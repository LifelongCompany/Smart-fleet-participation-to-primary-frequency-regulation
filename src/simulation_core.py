import pandas as pd
import numpy as np
import os
from src.data_loader import load_grid_data, load_driving_data, load_obc_data
from src.battery_model import solve_current, calc_aging_cycling_ac, calc_aging_cycling_dc, calc_aging_calendar

class SimulationCore:
    def __init__(self, output_dir='images', sim_end='2021-01-31 23:59:50'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load Data
        self.f_dev = load_grid_data()
        self.driving_df = load_driving_data()
        self.obc_df = load_obc_data()

        # Grid/Sim Params
        self.dt_sec = 10
        self.dt_h = 10.0 / 3600.0
        self.sim_index = pd.date_range(start='2021-01-01', end=sim_end, freq='10s')

        # Align Frequency to Sim Index (Tile/Crop)
        # f_dev is May 2019. Length ~ 2.6M seconds (30 days).
        # We need 31 days.
        f_val = self.f_dev.values
        required = len(self.sim_index)
        tiled_f = np.resize(f_val, required)
        self.y_red = 5.0 * tiled_f

        # Fleet Setup
        self.cars = self._setup_cars()
        self.n_cars = len(self.cars)
        self.car_ids = sorted(list(self.cars.keys()))

        # Interpolators
        self._setup_obc()

        # Physics Params
        self.BATTERY_CAP = 40.0 # kWh
        self.P_MAX = 7.0 # kW
        self.P_BID = 7.0 / 1.1
        self.K_FCR = 1.0 # P_req = P_bid * y_red (Usually K included in y_red def? Q1 says y_red is normalized 5*(f-50). P_req = P_bid * y_red.)
        # Wait, Q1 formula: P_RE = P_bid * y_red.

        # Aging Params (Thevenin R)
        # Assuming internal resistance R for Thevenin.
        # User didn't provide R. Q15 asks "Express I as function of P, R...".
        # I need a value for simulation.
        # Typical EV cell R ~ 1-2 mOhm. Pack 400V.
        # R_pack ~ 0.1 Ohm?
        # Let's assume R=0.1 Ohm for 40kWh pack (approx).
        # OCV curve? Simple linear model or constant?
        # Q15 says "U = OCV(SOC) - R I".
        # I'll use a simple OCV model: 300 + SOC% * 1.0 ? (300V to 400V).
        self.R_INTERNAL = 0.1

    def _setup_cars(self):
        # Organize driving data by ID
        # Add 'IS_AC' logic
        # Logic copied from behavior analysis for consistency
        df = self.driving_df.sort_values(['ID', 'START'])
        df['NEXT_START'] = df.groupby('ID')['START'].shift(-1)
        df['NEXT_SOC_START'] = df.groupby('ID')['SOC_START'].shift(-1)
        df['SOC_STOP'] = df['SOC_STOP']

        df['PARKING_DURATION_H'] = (df['NEXT_START'] - df['STOP']).dt.total_seconds() / 3600.0
        df['E_REQ'] = (100 - df['SOC_STOP']) / 100.0 * 40.0 # To Full

        # AC Logic
        def check_ac(row):
            if pd.isna(row['PARKING_DURATION_H']) or row['PARKING_DURATION_H'] <= 0: return False
            return (row['E_REQ'] / 7.0) <= row['PARKING_DURATION_H']

        df['IS_AC'] = df.apply(check_ac, axis=1)

        cars = {}
        for cid, group in df.groupby('ID'):
            # Store sessions as list of dicts
            sessions = []
            for _, row in group.iterrows():
                # Drive
                sessions.append({
                    'type': 'drive',
                    'start': row['START'],
                    'stop': row['STOP'],
                    'soc_start': row['SOC_START'],
                    'soc_stop': row['SOC_STOP']
                })
                # Park (if next start exists)
                if pd.notna(row['NEXT_START']):
                    p_type = 'park_ac' if row['IS_AC'] else 'park_dc'
                    sessions.append({
                        'type': p_type,
                        'start': row['STOP'],
                        'stop': row['NEXT_START'],
                        'soc_start': row['SOC_STOP'], # Start of parking is end of drive
                        'soc_end_target': row['NEXT_SOC_START'] # Target for next drive? Or just result?
                        # Actually "Vehicles must reach required SOC...".
                        # If Park DC, we charge to Target.
                        # If Park AC, we interact.
                    })
            cars[cid] = sessions
        return cars

    def _setup_obc(self):
        xp = self.obc_df['Power_kW'].values
        fp = self.obc_df['Efficiency'].values
        # Linear interp
        self.eff_func = lambda p: np.interp(p, xp, fp, left=0.0, right=fp.max())
        self.P_OPT = xp[np.argmax(fp)]
        self.ETA_MAX = np.max(fp)

    def run_scenario(self, strategy='uniform', bid_type='1h', fcr_active=True):
        print(f"Running Scenario: Strategy={strategy}, Bid={bid_type}, FCR={fcr_active}")

        # 1. Load Availability Profile (Capacity Limit)
        avail_file = f'data/availability_{bid_type}.csv'
        avail_df = pd.read_csv(avail_file, index_col=0, parse_dates=True)
        # Resample to simulation index (ffill)
        avail_series = avail_df.reindex(self.sim_index, method='ffill').fillna(0).values.flatten()
        # avail_series is N_bid_capacity(t)
        # Note: Availability in CSV is "Min cars connected".
        # This defines P_bid_total.

        if fcr_active:
            P_BID_TOTAL = avail_series * self.P_BID
            # 2. P_REQ Series
            P_REQ_TOTAL = P_BID_TOTAL * self.y_red # Vector
        else:
            P_REQ_TOTAL = np.zeros(len(self.sim_index))

        # 3. Initialize Car States
        # SOC Matrix: [Time, Cars]
        # To save memory, we might process in chunks or store just results.
        # But Q14 asks for "SOC Curve of every car".
        # 120 cars is fine.
        n_steps = len(self.sim_index)
        soc_matrix = np.zeros((n_steps, self.n_cars), dtype=np.float32)

        # Initialize SOCs at t=0
        # Look for first session of each car
        # If t=0 is middle of session, interp.
        # Simplified: Start at 50% if unknown.
        current_socs = np.full(self.n_cars, 50.0)

        # Pre-process "Available Matrix" for Strategy
        # We need to know which cars are "AC Connected" at each time.
        # Creating a boolean matrix [Time, Cars] is ~30MB. Cheap.
        is_avail_mat = np.zeros((n_steps, self.n_cars), dtype=bool)

        # Also need to know "Is Driving" or "Is DC" to update SOC independently.
        # We can handle "Non-Available" in the loop.

        # Fill availability matrix and Pre-fill Mandatory SOC (Drive/DC)
        print("Pre-processing availability and mandatory SOCs...")
        car_idx_map = {cid: i for i, cid in enumerate(self.car_ids)}

        # We pre-fill the SOC matrix for non-FCR periods (Drive/DC)
        # For AC periods, we leave it as 0 (or last value) to be simulated.
        # Actually, best to initialize with a baseline.

        for cid, sessions in self.cars.items():
            c_idx = car_idx_map[cid]
            for sess in sessions:
                start_i = self.sim_index.searchsorted(sess['start'])
                stop_i = self.sim_index.searchsorted(sess['stop'])

                # Safety clip
                if start_i >= n_steps: continue
                stop_i = min(stop_i, n_steps)
                if start_i >= stop_i: continue

                if sess['type'] == 'park_ac':
                    is_avail_mat[start_i:stop_i, c_idx] = True
                    # SOC will be simulated.
                    # Initial SOC for this session will be taken from previous step.

                else:
                    # Drive or Park DC
                    # Linear Interpolation of SOC
                    s_start = sess.get('soc_start', 50.0)
                    # For Drive, soc_stop is defined.
                    # For DC, soc_stop might be 'soc_end_target' or we assume full?
                    # Using 'soc_end_target' if exists, else 100?
                    s_end = sess.get('soc_stop')
                    if s_end is None: s_end = sess.get('soc_end_target', 100.0)

                    # Create linspace
                    # If this overlaps with simulation steps
                    length = stop_i - start_i
                    soc_segment = np.linspace(s_start, s_end, length)
                    soc_matrix[start_i:stop_i, c_idx] = soc_segment

                    # Also update current_socs if t=0 falls here?
                    # We handle t=0 init separately.

        print("Starting Time Loop...")

        # Simulation Loop
        # We iterate 15-min blocks for Decision, then 10s steps for Physics.
        # 15 min = 90 steps.

        block_size = 90

        # Stats
        total_aging_cal = 0.0
        total_aging_cyc = 0.0

        for t_idx in range(0, n_steps, block_size):
            end_idx = min(t_idx + block_size, n_steps)
            steps_in_block = end_idx - t_idx

            # 1. Decision (Strategy)
            # Who is available?
            # We look at the mask at start of block? Or every step?
            # Availability can change in 15 mins.
            # Strategy usually assigns *Capacity*.
            # But "Smart Strategy" sorts by SOC.
            # We resort at start of block.

            # Get SOCs at start of block
            block_start_socs = current_socs.copy() # Array of N_cars

            # Identify available cars in this block (use start of block or union?)
            # Prompt: "Decision step 15 min".
            # We determine priority list here.

            # Mask of cars available at t_idx
            mask_avail = is_avail_mat[t_idx]
            avail_indices = np.where(mask_avail)[0]

            # Sort indices by SOC
            # If Charge (P_req > 0): Priority Low SOC -> Ascending Order.
            # If Discharge (P_req < 0): Priority High SOC -> Descending Order.
            # We need two lists? Or one?
            # "Smart Logic: Based on SOC sorting... Discharge: High SOC. Charge: Low SOC."
            # So: Sorted Ascending = [Low ... High].
            # Charge List: Take from Start.
            # Discharge List: Take from End.

            sorted_indices = avail_indices[np.argsort(block_start_socs[avail_indices])]

            # Physics Loop (10s)
            for s in range(steps_in_block):
                curr_t = t_idx + s
                p_req_tot = P_REQ_TOTAL[curr_t] # Net Power (+ Charge, - Discharge) based on convention?
                # y_red > 0 -> Charge.

                # Active Set at this specific second
                # (Cars might plug out inside the 15m block)
                step_mask = is_avail_mat[curr_t]

                # Filter sorted list for currently valid ones
                # (Optimization: assume list valid for 15m? Or intersect?)
                # Intersection is safer.
                valid_sorted = [i for i in sorted_indices if step_mask[i]]
                n_valid = len(valid_sorted)

                # Distribute Power
                p_allocated = np.zeros(self.n_cars)

                if n_valid > 0:
                    if strategy == 'uniform':
                        # Distribute equally
                        p_per_car = p_req_tot / n_valid
                        p_allocated[valid_sorted] = p_per_car

                    elif strategy == 'smart':
                        # Stack
                        # P_req > 0 (Charge): Fill from valid_sorted[0] (Low SOC)
                        # P_req < 0 (Discharge): Fill from valid_sorted[-1] (High SOC)

                        rem_p = abs(p_req_tot)
                        sign = np.sign(p_req_tot)

                        if sign > 0: # Charge
                            order = valid_sorted # Low to High
                        else: # Discharge
                            order = valid_sorted[::-1] # High to Low

                        # Fill cars with P_OPT or P_MAX?
                        # Q6 analysis used P_OPT.
                        # Q14 says: "Priority SOC high/low". Doesn't specify power level.
                        # But efficient strategy uses P_OPT.
                        # Let's use P_OPT. If P_req huge, go to P_MAX.

                        for car_i in order:
                            if rem_p <= 0: break

                            # Determine power for this car
                            # Try P_OPT
                            if rem_p >= self.P_OPT:
                                p_car = min(rem_p, self.P_MAX) # Can go up to P_MAX if needed?
                                # Ideally we stick to P_OPT for efficiency.
                                # But if we have huge demand, we must use P_MAX.
                                # Let's assume we fill to P_OPT first.
                                # Wait, if we fill to P_OPT, do we loop again?
                                # Simple greedy: Take min(rem_p, P_MAX).
                                # But prefer P_OPT?
                                # Let's use: if rem_p < P_OPT, take rem_p.
                                # Else take P_OPT.
                                # (Multi-pass is better but slow).
                                # "Simplified Smart": Fill to P_MAX?
                                # Efficient Smart: Fill to P_OPT.
                                # Let's use P_OPT.
                                p_give = min(rem_p, self.P_OPT)
                                p_allocated[car_i] = p_give * sign
                                rem_p -= p_give
                            else:
                                p_allocated[car_i] = rem_p * sign
                                rem_p = 0

                        # If still remainder (Saturation), do we do a second pass?
                        # Yes, ramp up to P_MAX.
                        if rem_p > 1e-3:
                             for car_i in order:
                                current_p = abs(p_allocated[car_i])
                                if current_p < self.P_MAX:
                                    add = min(rem_p, self.P_MAX - current_p)
                                    p_allocated[car_i] += add * sign
                                    rem_p -= add
                                    if rem_p <= 1e-3: break

                # --- Vectorized Physics ---

                # 1. Non-Available Cars (Drive/DC)
                # Update SOC from pre-filled matrix
                # Compute Aging (DC Mode 4)

                mask_non_avail = ~step_mask
                if np.any(mask_non_avail):
                    soc_pre = current_socs[mask_non_avail]
                    soc_target = soc_matrix[curr_t, mask_non_avail]

                    # Update Current SOCs
                    current_socs[mask_non_avail] = soc_target

                    # Aging DC
                    delta_soc = soc_target - soc_pre
                    aging_cyc_dc = calc_aging_cycling_dc(delta_soc) # Vectorized
                    total_aging_cyc += np.sum(aging_cyc_dc)

                # 2. Available Cars (AC FCR)
                if np.any(step_mask):
                    soc_start = current_socs[step_mask]
                    p_cmd = p_allocated[step_mask]

                    # Constraints
                    # SOC <= 20 and Discharge -> Stop
                    # SOC >= 100 and Charge -> Stop
                    # Vectorized conditional
                    mask_stop_dis = (p_cmd < 0) & (soc_start <= 20.0)
                    mask_stop_chg = (p_cmd > 0) & (soc_start >= 100.0)
                    p_cmd[mask_stop_dis | mask_stop_chg] = 0.0

                    # Efficiency
                    eta = self.eff_func(np.abs(p_cmd)) # Vectorized interp

                    # Terminal Power
                    # Charge (p>0): -p*eta
                    # Discharge (p<0): abs(p)/eta
                    p_term = np.zeros_like(p_cmd)

                    mask_ch = p_cmd >= 0
                    mask_dis = ~mask_ch

                    p_term[mask_ch] = -(p_cmd[mask_ch] * eta[mask_ch]) * 1000.0
                    # Avoid div by zero
                    safe_eta = eta[mask_dis].copy()
                    safe_eta[safe_eta < 1e-6] = 1e-6 # Should not happen if p!=0
                    p_term[mask_dis] = (np.abs(p_cmd[mask_dis]) / safe_eta) * 1000.0

                    # Solve Current
                    ocv = 300.0 + (soc_start * 1.0)
                    i_amps = solve_current(p_term, ocv, self.R_INTERNAL)
                    i_amps = np.nan_to_num(i_amps) # Handle NaN

                    # Update SOC
                    power_internal = ocv * i_amps
                    energy_delta_wh = -(power_internal * self.dt_h)

                    soc_delta = (energy_delta_wh / 1000.0) / self.BATTERY_CAP * 100.0
                    soc_new = soc_start + soc_delta
                    soc_new = np.clip(soc_new, 0.0, 100.0)

                    # Write back
                    current_socs[step_mask] = soc_new
                    soc_matrix[curr_t, step_mask] = soc_new

                    # Aging AC (Mode 3)
                    aging_cyc_ac = calc_aging_cycling_ac(i_amps, self.dt_h)
                    total_aging_cyc += np.sum(aging_cyc_ac)

                # 3. Calendar Aging (All Cars)
                # Based on current_socs
                aging_cal = calc_aging_calendar(current_socs, 4*365, self.dt_h/24.0)
                total_aging_cal += np.sum(aging_cal)

        # Return Results
        return soc_matrix, total_aging_cyc, total_aging_cal

if __name__ == "__main__":
    sim = SimulationCore()
    # Run a test small scenario
    sim.run_scenario('uniform', '1h')
