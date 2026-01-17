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
        self.R_INTERNAL = 0.1

    def _setup_cars(self):
        df = self.driving_df.sort_values(['ID', 'START'])
        df['NEXT_START'] = df.groupby('ID')['START'].shift(-1)
        df['NEXT_SOC_START'] = df.groupby('ID')['SOC_START'].shift(-1)
        df['SOC_STOP'] = df['SOC_STOP']

        df['PARKING_DURATION_H'] = (df['NEXT_START'] - df['STOP']).dt.total_seconds() / 3600.0
        df['E_REQ'] = (100 - df['SOC_STOP']) / 100.0 * 40.0

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
            avail_series = avail_df.reindex(self.sim_index, method='ffill').fillna(0).values.flatten()
        else:
            avail_series = np.zeros(len(self.sim_index))

        if fcr_active:
            P_BID_TOTAL = avail_series * self.P_BID
            P_REQ_TOTAL = P_BID_TOTAL * self.y_red
        else:
            P_REQ_TOTAL = np.zeros(len(self.sim_index))

        n_steps = len(self.sim_index)
        soc_matrix = np.zeros((n_steps, self.n_cars), dtype=np.float32)
        current_socs = np.full(self.n_cars, 50.0)

        # Pre-process Availability
        is_avail_mat = np.zeros((n_steps, self.n_cars), dtype=bool)
        car_idx_map = {cid: i for i, cid in enumerate(self.car_ids)}

        print("Pre-processing availability...")
        for cid, sessions in self.cars.items():
            c_idx = car_idx_map[cid]
            for sess in sessions:
                start_i = self.sim_index.searchsorted(sess['start'])
                stop_i = self.sim_index.searchsorted(sess['stop'])
                if start_i >= n_steps: continue
                stop_i = min(stop_i, n_steps)
                if start_i >= stop_i: continue

                if sess['type'] == 'park_ac':
                    is_avail_mat[start_i:stop_i, c_idx] = True
                else:
                    s_start = sess.get('soc_start', 50.0)
                    s_end = sess.get('soc_stop')
                    if s_end is None: s_end = sess.get('soc_end_target', 100.0)
                    length = stop_i - start_i
                    soc_segment = np.linspace(s_start, s_end, length)
                    soc_matrix[start_i:stop_i, c_idx] = soc_segment

                    if start_i == 0:
                        current_socs[c_idx] = s_start

        print("Starting Time Loop...")
        block_size = 90
        total_aging_cal = 0.0
        total_aging_cyc = 0.0

        for t_idx in range(0, n_steps, block_size):
            end_idx = min(t_idx + block_size, n_steps)
            steps_in_block = end_idx - t_idx

            # Strategy Decision (at start of block)
            sorted_indices = []
            if fcr_active:
                mask_avail = is_avail_mat[t_idx]
                avail_indices = np.where(mask_avail)[0]
                sorted_indices = avail_indices[np.argsort(current_socs[avail_indices])]

            for s in range(steps_in_block):
                curr_t = t_idx + s

                # Update Non-Available (Drive/DC)
                step_mask = is_avail_mat[curr_t]
                mask_non_avail = ~step_mask

                if np.any(mask_non_avail):
                    soc_pre = current_socs[mask_non_avail]
                    soc_target = soc_matrix[curr_t, mask_non_avail]
                    current_socs[mask_non_avail] = soc_target

                    delta_soc = soc_target - soc_pre
                    aging_cyc_dc = calc_aging_cycling_dc(delta_soc)
                    total_aging_cyc += np.sum(aging_cyc_dc)

                # Available (AC FCR)
                if np.any(step_mask):
                    soc_start = current_socs[step_mask]
                    p_cmd = np.zeros(np.sum(step_mask))

                    if fcr_active:
                        p_req_tot = P_REQ_TOTAL[curr_t]
                        valid_sorted = [i for i in sorted_indices if step_mask[i]]
                        n_valid = len(valid_sorted)

                        if n_valid > 0:
                            if strategy == 'uniform':
                                p_per_car = p_req_tot / n_valid
                                global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(np.where(step_mask)[0])}
                                for g_idx in valid_sorted:
                                    p_cmd[global_to_local[g_idx]] = p_per_car

                            elif strategy == 'smart':
                                rem_p = abs(p_req_tot)
                                sign = np.sign(p_req_tot)

                                if sign >= 0: # Charge: Low to High
                                    order = valid_sorted
                                else: # Discharge: High to Low
                                    order = valid_sorted[::-1]

                                global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(np.where(step_mask)[0])}

                                for car_i in order:
                                    if rem_p <= 1e-6: break

                                    if rem_p >= self.P_OPT:
                                        p_give = min(rem_p, self.P_OPT)
                                    else:
                                        p_give = rem_p

                                    p_cmd[global_to_local[car_i]] = p_give * sign
                                    rem_p -= p_give

                                if rem_p > 1e-3:
                                    for car_i in order:
                                        l_idx = global_to_local[car_i]
                                        current_p = abs(p_cmd[l_idx])
                                        if current_p < self.P_MAX:
                                            add = min(rem_p, self.P_MAX - current_p)
                                            p_cmd[l_idx] += add * sign
                                            rem_p -= add
                                            if rem_p <= 1e-3: break

                    mask_stop_dis = (p_cmd < 0) & (soc_start <= 20.0)
                    mask_stop_chg = (p_cmd > 0) & (soc_start >= 100.0)
                    p_cmd[mask_stop_dis | mask_stop_chg] = 0.0

                    eta = self.eff_func(np.abs(p_cmd))
                    p_term = np.zeros_like(p_cmd)
                    mask_ch = p_cmd >= 0
                    mask_dis = ~mask_ch

                    p_term[mask_ch] = -(p_cmd[mask_ch] * eta[mask_ch]) * 1000.0
                    safe_eta = eta[mask_dis].copy()
                    safe_eta[safe_eta < 1e-6] = 1e-6
                    p_term[mask_dis] = (np.abs(p_cmd[mask_dis]) / safe_eta) * 1000.0

                    ocv = 300.0 + (soc_start * 1.0)
                    i_amps = solve_current(p_term, ocv, self.R_INTERNAL)
                    i_amps = np.nan_to_num(i_amps)

                    power_internal = ocv * i_amps
                    energy_delta_wh = -(power_internal * self.dt_h)

                    soc_delta = (energy_delta_wh / 1000.0) / self.BATTERY_CAP * 100.0
                    soc_new = soc_start + soc_delta
                    soc_new = np.clip(soc_new, 0.0, 100.0)

                    current_socs[step_mask] = soc_new
                    soc_matrix[curr_t, step_mask] = soc_new

                    aging_cyc_ac = calc_aging_cycling_ac(i_amps, self.dt_h)
                    total_aging_cyc += np.sum(aging_cyc_ac)

                # Calendar Aging (All Cars)
                aging_cal = calc_aging_calendar(current_socs, 4*365, self.dt_h/24.0)
                total_aging_cal += np.sum(aging_cal)

        return soc_matrix, total_aging_cyc, total_aging_cal

if __name__ == "__main__":
    sim = SimulationCore()
    sim.run_scenario('uniform', '1h')
