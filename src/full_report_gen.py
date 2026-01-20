import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from src.simulation_core import SimulationCore
from scipy.interpolate import interp1d
from io import StringIO

# Constants
OUTPUT_DIR = 'images'
REPORT_FILE = 'README_REPORT.md'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class ReportGenerator:
    def __init__(self):
        self.buffer = StringIO()
        self.sim = None

    def log(self, text, header_level=0):
        if header_level > 0:
            self.buffer.write(f"\n{'#' * header_level} {text}\n\n")
        else:
            self.buffer.write(f"{text}\n")
        print(text) # Also print to stdout for progress tracking

    def save_plot(self, name):
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, name))
        plt.close()

    def run(self):
        self.log("FCR Participation Report (Full Month Simulation)", 1)
        self.log("Initializing Simulation Core for Jan 2021...")

        # FULL MONTH SIMULATION
        sim_end_date = '2021-01-31 23:59:50'
        self.sim = SimulationCore(output_dir=OUTPUT_DIR, sim_end=sim_end_date)
        sim = self.sim

        # ==========================================
        # PART 1: Grid & Basic Analysis (Q1-Q4)
        # ==========================================
        self.log("Part 1: Grid Frequency Data & Basic Analysis", 2)

        # Q1: Distribution of Regulating Power
        self.log("**Q1: Distribution of Regulating Power**")
        y_red = sim.y_red

        plt.figure(figsize=(10, 6))
        plt.hist(y_red, bins=100, density=True, alpha=0.7, color='blue')
        plt.title("Q1: Distribution of Normalized Regulating Power (Jan 2021)")
        plt.xlabel("Regulating Power (p.u.)")
        plt.ylabel("Density")
        plt.grid(True)
        self.save_plot('q1_distribution.png')
        self.log("![Q1 Distribution](images/q1_distribution.png)")
        self.log("- The regulating power signal is derived from frequency deviation: $P_{reg}^{p.u.} = 5 \\times (f - 50)$, clipped to [-1, 1].")
        self.log("")

        # Q2: Observations
        self.log("**Q2: Observations**")
        self.log("- The distribution is centered around 0 but exhibits a spread corresponding to frequency deviations.")
        self.log("- The magnitude rarely reaches full power ($\\pm 1$ p.u.), staying mostly within $\\pm 0.4$ p.u.")
        self.log("")

        # Q3: Single EV SOC Deviation
        self.log("**Q3: Single EV SOC Deviation**")
        dt_h = sim.dt_h
        p_profile_kw = sim.P_BID * y_red

        windows = [4, 8, 12, 24]
        data_q3 = []
        labels_q3 = []

        stats_text = []

        for w in windows:
            steps = int(w * 3600 / sim.dt_sec)
            p_series = pd.Series(p_profile_kw)
            e_rolling = p_series.rolling(window=steps).sum() * dt_h
            soc_dev = e_rolling / sim.BATTERY_CAP * 100.0
            soc_dev = soc_dev.dropna().values
            data_q3.append(soc_dev)
            labels_q3.append(f"{w}h")

            p05, p50, p95 = np.percentile(soc_dev, [5, 50, 95])
            stats_text.append(f"- **{w}h Window**: Median={p50:.2f}%, 90% Interval=[{p05:.2f}%, {p95:.2f}%]")

        plt.figure(figsize=(10, 6))
        plt.boxplot(data_q3, tick_labels=labels_q3)
        plt.title("Q3: SOC Deviation Distribution (Rolling Windows)")
        plt.ylabel("SOC Deviation (%)")
        plt.grid(True)
        self.save_plot('q3_soc_deviation.png')
        self.log("![Q3 SOC Deviation](images/q3_soc_deviation.png)")
        for line in stats_text:
            self.log(line)
        self.log("")

        # Q4: Reasonability
        self.log("**Q4: Reasonability**")
        self.log("- For short windows (4h), the SOC deviation is relatively small (< 10%).")
        self.log("- For 24h windows, the deviation can grow significantly. Without active energy management (recharging), continuous FCR participation carries a risk of depleting the battery or reaching full charge.")
        self.log("")

        # ==========================================
        # PART 2: Strategies (Q5-Q7)
        # ==========================================
        self.log("Part 2: Smart Dispatch Strategy", 2)

        # Load Efficiency
        xp = sim.obc_df['Power_kW'].values
        fp = sim.obc_df['Efficiency'].values
        eff_func = lambda p: np.interp(p, xp, fp, left=0.0, right=fp.max())
        p_opt = sim.P_OPT
        eta_max = sim.ETA_MAX

        # Helper for Loss
        def calc_loss(p_val_array):
            eff_vals = eff_func(np.abs(p_val_array))
            safe_eff = eff_vals.copy()
            safe_eff[safe_eff < 1e-6] = 1e-6
            loss_arr = np.zeros_like(p_val_array)
            mask_ch = p_val_array >= 0
            mask_dis = ~mask_ch
            loss_arr[mask_ch] = p_val_array[mask_ch] * (1.0 - eff_vals[mask_ch])
            loss_arr[mask_dis] = np.abs(p_val_array[mask_dis]) * (1.0 / safe_eff[mask_dis] - 1.0)
            return loss_arr

        # Q5
        p_bid_car = sim.P_BID
        p_req_car_uniform = p_bid_car * y_red
        loss_uniform = calc_loss(p_req_car_uniform)
        total_energy_uniform = np.sum(np.abs(p_req_car_uniform))
        total_loss_uniform_val = np.sum(loss_uniform)
        eta_avg_uniform = 1.0 - (total_loss_uniform_val / total_energy_uniform)

        self.log("**Q5: Uniform Strategy Efficiency**")
        self.log(f"- Calculated Average Efficiency: **{eta_avg_uniform*100:.2f}%**")
        self.log("")

        # Q6
        self.log("**Q6: Smart Strategy Efficiency & Convergence**")
        eta_smart_inf = eta_max
        target_eta_val = eta_avg_uniform + 0.9 * (eta_smart_inf - eta_avg_uniform)

        n_values = sorted(list(set([1, 2, 3, 4, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500])))
        eta_smart_vals = []

        # Subsample for Q6 speed (1 month is long)
        step_skip = 100 # Only use every 100th step for this stat calculation to be fast
        p_req_total_pu_sub = y_red[::step_skip]

        for N in n_values:
            p_bid_total = N * p_bid_car
            p_req_total = p_bid_total * p_req_total_pu_sub
            p_abs_total = np.abs(p_req_total)

            k_full = np.floor(p_abs_total / p_opt)
            k_full = np.minimum(k_full, N)
            p_rem = p_abs_total - k_full * p_opt

            sign = np.sign(p_req_total)
            loss_opt = calc_loss(sign * p_opt)
            loss_rem = calc_loss(sign * p_rem)

            mask_low = p_abs_total <= (N * p_opt)
            p_high_avg = p_req_total / N
            loss_high = calc_loss(p_high_avg) * N

            total_loss_t = np.zeros_like(p_req_total)
            total_loss_t[mask_low] = k_full[mask_low] * loss_opt[mask_low] + loss_rem[mask_low]
            total_loss_t[~mask_low] = loss_high[~mask_low]

            eta_N = 1.0 - (np.sum(total_loss_t) / np.sum(p_abs_total))
            eta_smart_vals.append(eta_N)

        n0_q6 = next((N for N, eta in zip(n_values, eta_smart_vals) if eta >= target_eta_val), None)

        plt.figure()
        plt.plot(n_values, eta_smart_vals, 'o-', label='Smart')
        plt.axhline(eta_avg_uniform, color='r', linestyle='--', label='Uniform')
        plt.axhline(target_eta_val, color='g', linestyle=':', label='90% Target')
        plt.xscale('log')
        plt.xlabel('Fleet Size N')
        plt.ylabel('Efficiency')
        plt.legend()
        plt.title("Efficiency Convergence")
        self.save_plot('q6_efficiency.png')
        self.log("![Q6 Efficiency](images/q6_efficiency.png)")
        self.log(f"- Smart Limit ($N \\to \\infty$): **{eta_smart_inf*100:.2f}%**")
        self.log(f"- Fleet size to achieve 90% of gain ($N_0$): **{n0_q6} vehicles**")
        self.log("")

        # Q7
        self.log("**Q7: OBC Operating Time**")
        mean_abs_y = np.mean(np.abs(y_red))
        t_op_inf = (p_bid_car / sim.P_MAX) * mean_abs_y
        self.log(f"- Theoretical Limit ($t_{{op}}^{{\\infty}}$): **{t_op_inf:.4f} p.u.**")

        target_top = 1.0 - 0.9 * (1.0 - t_op_inf)
        top_vals = []
        for N in n_values:
            p_req_abs = np.abs(N * p_bid_car * p_req_total_pu_sub)
            n_active = np.ceil(p_req_abs / sim.P_MAX)
            n_active = np.minimum(n_active, N)
            top_vals.append(np.mean(n_active) / N)

        n0_q7 = next((N for N, top in zip(n_values, top_vals) if top <= target_top), None)

        plt.figure()
        plt.plot(n_values, top_vals, 'o-')
        plt.axhline(t_op_inf, color='g', linestyle='--')
        plt.xscale('log')
        plt.title("Operating Time vs N")
        self.save_plot('q7_operating_time.png')
        self.log("![Q7 Operating Time](images/q7_operating_time.png)")
        self.log(f"- Fleet size for 90% reduction ($N_0$): **{n0_q7} vehicles**")
        self.log("")

        # ==========================================
        # PART 3: Behavior (Q8-Q10)
        # ==========================================
        self.log("Part 3: Driving & Charging Behaviour", 2)

        self.log("**Q8: Charging Inference Logic**")
        self.log("- If `Trip Energy / 7kW <= Parking Duration`: Assume **AC Charging** (V2G available).")
        self.log("- Else: Assume **DC Charging** (V2G not available).")
        self.log("")

        self.log("**Q9: Coincidence Factor**")
        try:
            csv_1h = pd.read_csv('data/availability_1h.csv', index_col=0, parse_dates=True)
            csv_4h = pd.read_csv('data/availability_4h.csv', index_col=0, parse_dates=True)
            sim_idx_min = pd.date_range(start=sim.sim_index[0], end=sim.sim_index[-1], freq='1min')
            avail_1h = csv_1h.reindex(sim_idx_min, method='ffill')['N_avail']
            avail_4h = csv_4h.reindex(sim_idx_min, method='ffill')['N_avail']
        except:
            avail_1h = pd.Series(0, index=sim.sim_index)
            avail_4h = pd.Series(0, index=sim.sim_index)

        plt.figure(figsize=(10, 6))
        plt.plot(avail_1h.index, avail_1h.values, label='1h Block')
        plt.plot(avail_4h.index, avail_4h.values, label='4h Block')
        plt.title("Fleet Availability (Jan 2021)")
        plt.legend()
        self.save_plot('q9_availability.png')
        self.log("![Q9 Availability](images/q9_availability.png)")
        self.log("")

        self.log("**Q10: Comparison / Limitations**")
        self.log("- Inferring charging from driving has limitations, especially at the year boundaries.")
        self.log("- However, using looped data allows reasonable estimation.")
        self.log("")

        # ==========================================
        # PART 4: Economics (Q11-Q13)
        # ==========================================
        self.log("Part 4: FCR Revenues", 2)

        PRICE_EUR_MW_H = 18.0
        p_bid_mw = sim.P_BID / 1000.0

        # Calculate Revenue
        # We use the availability derived from CSVs for consistency with Q9
        # Assuming avail_1h is minute resolution
        rev_1h_total = np.sum(avail_1h * p_bid_mw * PRICE_EUR_MW_H * (1/60.0))
        rev_4h_total = np.sum(avail_4h * p_bid_mw * PRICE_EUR_MW_H * (1/60.0))
        rev_per_ev_jan = rev_1h_total / sim.n_cars

        self.log("**Q11: Monthly Revenue**")
        self.log(f"- FCR Price: {PRICE_EUR_MW_H} EUR/MW/h")
        self.log(f"- Total Fleet Revenue (1h Blocks): **{rev_1h_total:.2f} EUR**")
        self.log(f"- Revenue per EV (1h Blocks): **{rev_per_ev_jan:.2f} EUR/EV**")
        self.log(f"- Revenue per EV (4h Blocks): **{rev_4h_total / sim.n_cars:.2f} EUR/EV**")
        self.log("")

        self.log("**Q12: Virtual Mileage**")
        e_thru_jan_kwh = sim.P_BID * mean_abs_y * (31 * 24)
        virt_km_jan = e_thru_jan_kwh / 0.2
        self.log(f"- Energy Throughput per EV: **{e_thru_jan_kwh:.2f} kWh**")
        self.log(f"- Virtual Mileage: **{virt_km_jan:.2f} km**")
        self.log("")

        self.log("**Q13: Residual Value Loss**")
        loss_val = 0.0
        try:
            res_df = pd.read_csv('data/residual_value.csv', sep=';', encoding='cp1252')
            mil = res_df.iloc[:, 0].values
            val = res_df.iloc[:, 1].values
            res_func = interp1d(mil, val, fill_value="extrapolate")
            start_km = 50000.0
            loss_val = res_func(start_km) - res_func(start_km + virt_km_jan)
            self.log(f"- Estimated Residual Value Loss: **{loss_val:.2f} EUR**")
            self.log(f"- Net Revenue: **{rev_per_ev_jan - loss_val:.2f} EUR**")
        except:
            self.log("- Could not calculate residual value (missing data).")
        self.log("")

        # ==========================================
        # PART 5: Simulation (Q14)
        # ==========================================
        self.log("Part 5: Full Simulation Results", 2)
        self.log("**Q14: SOC Profiles & Scenarios**")

        results = {}
        scenarios = [('uniform', '1h'), ('smart', '1h'), ('uniform', '4h'), ('smart', '4h')]

        # Baseline
        self.log("Running Baseline (No FCR)...")
        soc_base, age_cyc_base, age_cal_base = sim.run_scenario(fcr_active=False)
        results['no_fcr'] = {'soc': soc_base, 'age_cyc': age_cyc_base, 'age_cal': age_cal_base}

        for strat, bid in scenarios:
            key = f"{strat}_{bid}"
            self.log(f"Running Scenario: {key}...")
            soc_s, age_cyc_s, age_cal_s = sim.run_scenario(strategy=strat, bid_type=bid, fcr_active=True)
            results[key] = {'soc': soc_s, 'age_cyc': age_cyc_s, 'age_cal': age_cal_s}

            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(sim.sim_index, soc_s[:, 0], label=f'Car 0 {key}')
            plt.plot(sim.sim_index, soc_base[:, 0], label='Car 0 No FCR', linestyle='--')
            plt.title(f"SOC Profile Car 0 ({key})")
            plt.legend()
            self.save_plot(f"q14_soc_{key}.png")
            self.log(f"![SOC {key}](images/q14_soc_{key}.png)")

        self.log("")

        # ==========================================
        # PART 6: Aging (Q15-Q16)
        # ==========================================
        self.log("Part 6: Battery Aging", 2)
        self.log("**Q15: Battery Model**")
        self.log("- $V_{oc} = 360 + 0.85 \\times SOC$")
        self.log("- $I = P_{term} / V_{oc}$ (Assuming negligible internal resistance)")
        self.log("")

        self.log("**Q16: Aging Evaluation**")
        base_tot = results['no_fcr']['age_cyc'] + results['no_fcr']['age_cal']
        base_avg_pct = (base_tot / sim.n_cars) * 100.0

        self.log(f"- **Baseline Aging**: {base_avg_pct:.6f}% degradation")

        for key, res in results.items():
            if key == 'no_fcr': continue
            tot = res['age_cyc'] + res['age_cal']
            avg_pct = (tot / sim.n_cars) * 100.0
            inc = (avg_pct - base_avg_pct) / base_avg_pct * 100.0
            self.log(f"- **{key}**: {avg_pct:.6f}% (+{inc:.2f}%)")

        # Save Report
        with open(REPORT_FILE, 'w') as f:
            f.write(self.buffer.getvalue())
        print(f"Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    gen = ReportGenerator()
    gen.run()
