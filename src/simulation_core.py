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
        f_val = self.f_dev.values
        required = len(self.sim_index)
        tiled_f = np.resize(f_val, required)
        # Q1/Q3: Limit reduced power to [-1, 1] (equivalent to f_dev >= 0.2Hz)
        self.y_red = np.clip(5.0 * tiled_f, -1.0, 1.0)

        # Physics Params
        self.BATTERY_CAP = 46.0 # kWh
        self.P_MAX = 7.0 # kW
        self.P_BID = 7.0 / 1.1
        self.R_INTERNAL = 0.0

        # Fleet Setup
        self.cars = self._setup_cars()
        self.n_cars = len(self.cars)
        self.car_ids = sorted(list(self.cars.keys()))

        # Interpolators
        self._setup_obc()

    def _setup_cars(self):
        df = self.driving_df.sort_values(['ID', 'START'])
        df['NEXT_START'] = df.groupby('ID')['START'].shift(-1)
        df['NEXT_SOC_START'] = df.groupby('ID')['SOC_START'].shift(-1)
        df['SOC_STOP'] = df['SOC_STOP']

        df['PARKING_DURATION_H'] = (df['NEXT_START'] - df['STOP']).dt.total_seconds() / 3600.0
        df['E_REQ'] = (100 - df['SOC_STOP']) / 100.0 * self.BATTERY_CAP

        def check_ac(row):
            if pd.isna(row['PARKING_DURATION_H']) or row['PARKING_DURATION_H'] <= 0: return False
            return (row['E_REQ'] / 7.0) <= row['PARKING_DURATION_H']

        df['IS_AC'] = df.apply(check_ac, axis=1)

        cars = {}
        for cid, group in df.groupby('ID'):
            sessions = []
            for _, row in group.iterrows():
                sessions.append({
                    'type': 'drive',
                    'start': row['START'],
                    'stop': row['STOP'],
                    'soc_start': row['SOC_START'],
                    'soc_stop': row['SOC_STOP']
                })
                if pd.notna(row['NEXT_START']):
                    p_type = 'park_ac' if row['IS_AC'] else 'park_dc'
                    sessions.append({
                        'type': p_type,
                        'start': row['STOP'],
                        'stop': row['NEXT_START'],
                        'soc_start': row['SOC_STOP'],
                        'soc_end_target': row['NEXT_SOC_START']
                    })
            cars[cid] = sessions
        return cars

    def _setup_obc(self):
        xp = self.obc_df['Power_kW'].values
        fp = self.obc_df['Efficiency'].values
        self.eff_func = lambda p: np.interp(p, xp, fp, left=0.0, right=fp.max())
        self.P_OPT = xp[np.argmax(fp)]
        self.ETA_MAX = np.max(fp)

    def run_scenario(self, strategy='uniform', bid_type='1h', fcr_active=True):
        print(f"Running Scenario: Strategy={strategy}, Bid={bid_type}, FCR={fcr_active}")

        avail_file = f'data/availability_{bid_type}.csv'
        if os.path.exists(avail_file):
            avail_df = pd.read_csv(avail_file, index_col=0, parse_dates=True)
            # Reindex and fill with 0
            avail_series = avail_df.reindex(self.sim_index, method='ffill')['N_avail'].fillna(0).values
        else:
            avail_series = np.zeros(len(self.sim_index))

        if fcr_active:
            P_BID_TOTAL = avail_series * self.P_BID
            P_REQ_TOTAL = P_BID_TOTAL * self.y_red
        else:
            P_REQ_TOTAL = np.zeros(len(self.sim_index))

        n_steps = len(self.sim_index)
        soc_matrix = np.zeros((n_steps, self.n_cars), dtype=np.float32)
        current_socs = np.full(self.n_cars, 50.0) # Initial guess, updated by first session

        # Pre-process Availability
        is_avail_mat = np.zeros((n_steps, self.n_cars), dtype=bool)
        car_idx_map = {cid: i for i, cid in enumerate(self.car_ids)}

        print("Pre-processing availability...")
        for cid, sessions in self.cars.items():
            c_idx = car_idx_map[cid]
            for sess in sessions:
                # Vectorized range check
                start_i = self.sim_index.searchsorted(sess['start'])
                stop_i = self.sim_index.searchsorted(sess['stop'])
                if start_i >= n_steps: continue
                stop_i = min(stop_i, n_steps)
                if start_i >= stop_i: continue

                if sess['type'] == 'park_ac':
                    is_avail_mat[start_i:stop_i, c_idx] = True
                    # SOC is constant (or managed by FCR), initialize if needed
                    if start_i == 0:
                         current_socs[c_idx] = sess.get('soc_start', 50.0)
                else:
                    # Drive or DC Park -> SOC is determined by schedule
                    s_start = sess.get('soc_start', 50.0)
                    s_end = sess.get('soc_stop')
                    if s_end is None: s_end = sess.get('soc_end_target', 100.0)

                    length = stop_i - start_i
                    soc_segment = np.linspace(s_start, s_end, length)
                    soc_matrix[start_i:stop_i, c_idx] = soc_segment

                    if start_i == 0:
                        current_socs[c_idx] = s_start

        print("Starting Time Loop...")
        # Optimization: Don't loop block by block if not necessary, but block helps with reporting progress
        total_aging_cal = 0.0
        total_aging_cyc = 0.0

        # Constants for vectorization
        P_OPT = self.P_OPT
        P_MAX = self.P_MAX
        CAP_REM = P_MAX - P_OPT

        # Pre-calc changes in availability to optimize
        # Actually, availability changes infrequently.
        # But SOC sorting changes every step.

        for t_idx in range(n_steps):
            if t_idx % 10000 == 0:
                print(f"  Step {t_idx}/{n_steps} ({t_idx/n_steps*100:.1f}%)")

            # 1. Handle Non-Available Cars (Drive/DC)
            # They already have SOC set in soc_matrix during pre-processing
            # We just update current_socs to match matrix for consistency

            mask_non_avail = ~is_avail_mat[t_idx]
            if np.any(mask_non_avail):
                # Calculate aging for DC/Drive (Mode 4)
                # SOC changed from prev step
                soc_target = soc_matrix[t_idx, mask_non_avail]
                soc_prev_vals = current_socs[mask_non_avail]

                # Check if we just entered this state (continuity handled by logic)
                # Delta SOC
                delta_soc = soc_target - soc_prev_vals

                # Update SOC tracker
                current_socs[mask_non_avail] = soc_target

                # Aging
                aging_cyc_dc = calc_aging_cycling_dc(delta_soc)
                total_aging_cyc += np.sum(aging_cyc_dc)

            # 2. Handle Available Cars (FCR)
            mask_avail = is_avail_mat[t_idx]
            if not np.any(mask_avail):
                # No cars available
                continue

            soc_start = current_socs[mask_avail]
            n_avail = len(soc_start)
            p_cmd_local = np.zeros(n_avail)

            if fcr_active:
                p_req_tot = P_REQ_TOTAL[t_idx]

                if abs(p_req_tot) > 1e-6:
                    if strategy == 'uniform':
                        # Uniform Distribution
                        p_per_car = p_req_tot / n_avail
                        p_cmd_local[:] = p_per_car

                    elif strategy == 'smart':
                        # Smart Distribution
                        # 1. Sort by SOC
                        # Map local indices [0..n_avail-1] to sorted order
                        sort_idx = np.argsort(soc_start)

                        if p_req_tot < 0: # Discharge: High SOC first
                            sort_idx = sort_idx[::-1]
                        # Else Charge: Low SOC first (default)

                        p_abs = abs(p_req_tot)
                        sign = np.sign(p_req_tot)

                        # Vectorized Allocation
                        p_vec_sorted = np.zeros(n_avail)

                        # Pass 1: Fill to P_OPT
                        k_opt = int(p_abs // P_OPT)
                        k_opt = min(k_opt, n_avail)

                        p_vec_sorted[:k_opt] = P_OPT
                        rem_after_opt = p_abs - k_opt * P_OPT

                        if k_opt < n_avail:
                            # Remainder goes to next car
                            # Wait, simple remainder check
                            # If k_opt cars took P_OPT, remaining power < P_OPT (unless k_opt==n_avail)
                            # Logic: floor division gives integer count.

                            p_vec_sorted[k_opt] = min(rem_after_opt, P_OPT)
                            # If rem > P_OPT (only if k_opt was capped by n_avail), handled below
                        else:
                            # All cars at P_OPT. Need to go higher?
                            # rem_after_opt is what's left.
                            # Pass 2: Fill to P_MAX
                            if rem_after_opt > 1e-6 and CAP_REM > 1e-6:
                                k_max = int(rem_after_opt // CAP_REM)
                                k_max = min(k_max, n_avail)

                                p_vec_sorted[:k_max] += CAP_REM

                                rem_final = rem_after_opt - k_max * CAP_REM
                                if k_max < n_avail:
                                    p_vec_sorted[k_max] += min(rem_final, CAP_REM)

                        # Apply Sign and Unsort
                        # p_vec_sorted corresponds to sort_idx
                        # We want p_cmd_local where p_cmd_local[sort_idx[i]] = p_vec_sorted[i]
                        p_cmd_local[sort_idx] = p_vec_sorted * sign

            # 3. Apply Limits & Calc Physics (Vectorized)
            # Stop Charging if SOC=100, Stop Discharging if SOC=20 (Grid constraint/Safety)
            # Or just clip SOC later?
            # Better to zero power to prevent illegal move

            # Mask local
            mask_stop_dis = (p_cmd_local < 0) & (soc_start <= 20.0)
            mask_stop_chg = (p_cmd_local > 0) & (soc_start >= 100.0)
            p_cmd_local[mask_stop_dis | mask_stop_chg] = 0.0

            # Efficiency
            p_abs_local = np.abs(p_cmd_local)
            eta = self.eff_func(p_abs_local)

            # Terminal Power
            # Charge (P<0 in battery perspective? No, P>0 is Grid->Car?)
            # Usually: P_grid > 0 is Charge.
            # Battery Model: P_term > 0 is Discharge (out of battery).
            # So:
            # If P_grid > 0 (Charge): P_term = - P_grid * eta
            # If P_grid < 0 (Discharge): P_term = |P_grid| / eta

            p_term = np.zeros_like(p_cmd_local)

            mask_ch = p_cmd_local >= 0
            mask_dis = ~mask_ch

            p_term[mask_ch] = -(p_cmd_local[mask_ch] * eta[mask_ch]) * 1000.0 # kW -> W

            # Safe division
            safe_eta = eta[mask_dis].copy()
            safe_eta[safe_eta < 1e-6] = 1e-6
            p_term[mask_dis] = (np.abs(p_cmd_local[mask_dis]) / safe_eta) * 1000.0

            # Solve Current
            ocv = 360.0 + (soc_start * 0.85)
            # R=0 assumed (or small)
            # If R=0, I = P/V
            i_amps = p_term / ocv # Vectorized

            # Update SOC
            # Energy (Wh) = - Power_internal * dt
            # Power_internal = V * I = P_term (since R=0)
            # Actually P_term is already V*I.
            # E_delta_wh = - P_term * dt_h

            energy_delta_wh = -(p_term * self.dt_h)
            soc_delta = (energy_delta_wh / 1000.0) / self.BATTERY_CAP * 100.0

            soc_new = soc_start + soc_delta
            soc_new = np.clip(soc_new, 0.0, 100.0)

            # Store Result
            current_socs[mask_avail] = soc_new
            soc_matrix[t_idx, mask_avail] = soc_new

            # Aging (AC)
            aging_cyc_ac = calc_aging_cycling_ac(i_amps, self.dt_h)
            total_aging_cyc += np.sum(aging_cyc_ac)

        # 4. Calendar Aging (All cars, All steps)
        # We can calculate this at the end vectorized over the whole matrix
        print("Calculating Calendar Aging...")
        # soc_matrix shape: (n_steps, n_cars)
        # We need a time vector for t_age?
        # t_age increases by dt each step.
        # But calc_aging_calendar uses t_age in days.
        # Approximation: t_age is roughly constant (0.75 exponent on years...)
        # Actually t_age should increase.
        # Construct t_age matrix? Too big?
        # Start at 4 years = 4*365 days.
        # End at 4*365 + 31 days.
        # Change is small. We can use average t_age or just integrate.

        # Or better: calc_aging_calendar supports vectorization.
        # t_age_days = 4*365 + (t_idx * dt_days)
        # Vector over time?
        # t_vec = 4*365 + np.arange(n_steps) * (self.dt_h / 24.0)
        # aging_cal = calc_aging_calendar(soc_matrix, t_vec[:, None], self.dt_h/24.0)
        # Memory heavy? (260k x 150) floats ~ 300MB. Fine.

        t_vec = 4*365 + np.arange(n_steps) * (self.dt_h / 24.0)
        # Broadcast t_vec to (n_steps, n_cars)
        # But we can just loop over chunks if memory is tight, or just do it.
        # 300MB is fine.

        aging_cal_mat = calc_aging_calendar(soc_matrix, t_vec[:, None], self.dt_h/24.0)
        total_aging_cal += np.sum(aging_cal_mat)

        return soc_matrix, total_aging_cyc, total_aging_cal

if __name__ == "__main__":
    sim = SimulationCore()
    sim.run_scenario('uniform', '1h')
