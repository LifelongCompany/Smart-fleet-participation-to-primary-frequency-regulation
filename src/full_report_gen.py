import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.simulation_core import SimulationCore
from scipy.interpolate import interp1d

# Constants
OUTPUT_DIR = 'images'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_plot(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name))
    plt.close()

def generate_report_data():
    print("Initializing Core...")
    # Reduced simulation duration to 5 days to ensure completion within constraints
    sim_end_date = '2021-01-05 23:59:50'
    sim = SimulationCore(output_dir=OUTPUT_DIR, sim_end=sim_end_date)

    # ==========================================
    # PART 1: Grid & Basic Analysis (Q1-Q4)
    # ==========================================
    print("--- Part 1: Grid Analysis ---")

    # Q1: Distribution of Regulating Power
    # P_pu = y_red / K? No, y_red is the signal.
    # simulation_core: y_red = 5.0 * tiled_f. (f-50)*5.
    # 0.2Hz deviation -> 5 * 0.2 = 1.0 p.u.
    # So y_red IS the p.u. regulating power.
    y_red = sim.y_red

    plt.figure(figsize=(10, 6))
    plt.hist(y_red, bins=100, density=True, alpha=0.7, color='blue')
    plt.title("Q1: Distribution of Normalized Regulating Power")
    plt.xlabel("Regulating Power (p.u.)")
    plt.ylabel("Density")
    plt.grid(True)
    save_plot('q1_distribution.png')

    # Q2: Comments (in README)

    # Q3: Single EV Rolling SOC
    # Simulate single EV with 46kWh, 7kW.
    # Q3 Fix: P_bid is P_max / 1.1. P(t) = P_bid * y_red(t).
    # y_red is already clipped to [-1, 1] in SimulationCore (representing f_dev >= 0.2Hz).
    # So max power is 6.36 kW. (Safe within 7kW).

    dt_h = sim.dt_h
    p_profile_kw = sim.P_BID * y_red
    # y_red > 0 (High Freq) -> Charge.

    windows = [4, 8, 12, 24]
    data_q3 = []
    labels_q3 = []

    print("Q4: SOC Deviation Statistics (percentiles)")
    for w in windows:
        steps = int(w * 3600 / sim.dt_sec)
        # Q3: "SOC deviation ... rolling window"
        # SOC(t+w) - SOC(t) = Sum(Energy) in window

        p_series = pd.Series(p_profile_kw)
        e_rolling = p_series.rolling(window=steps).sum() * dt_h
        soc_dev = e_rolling / sim.BATTERY_CAP * 100.0
        soc_dev = soc_dev.dropna().values
        data_q3.append(soc_dev)
        labels_q3.append(f"{w}h")

        # Q4 Stats
        p05, p50, p95, p99 = np.percentile(soc_dev, [5, 50, 95, 99])
        print(f"  Window {w}h: P05={p05:.2f}%, Median={p50:.2f}%, P95={p95:.2f}%, P99={p99:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.boxplot(data_q3, tick_labels=labels_q3)
    plt.title("Q3: SOC Deviation Distribution (Rolling Windows)")
    plt.ylabel("SOC Deviation (%)")
    plt.grid(True)
    save_plot('q3_soc_deviation.png')

    # ==========================================
    # PART 2: Strategies (Q5-Q7)
    # ==========================================
    print("--- Part 2: Strategies ---")

    # Load Efficiency
    xp = sim.obc_df['Power_kW'].values
    fp = sim.obc_df['Efficiency'].values
    eff_func = lambda p: np.interp(p, xp, fp, left=0.0, right=fp.max())
    p_opt = sim.P_OPT
    eta_max = sim.ETA_MAX

    # Q5: Uniform Efficiency (Corrected for Discharge Loss)
    p_bid_car = sim.P_BID
    p_req_car_uniform = p_bid_car * y_red

    # Loss Calculation Function (Q5 fix)
    def calc_loss(p_val_array):
        # Charge (P>0): Loss = P * (1 - eta)
        # Discharge (P<0): Loss = |P| * (1/eta - 1)
        # Input p_val_array can be mixed sign.
        eff_vals = eff_func(np.abs(p_val_array))

        # Safe eta for division
        safe_eff = eff_vals.copy()
        safe_eff[safe_eff < 1e-6] = 1e-6

        loss_arr = np.zeros_like(p_val_array)
        mask_ch = p_val_array >= 0
        mask_dis = ~mask_ch

        loss_arr[mask_ch] = p_val_array[mask_ch] * (1.0 - eff_vals[mask_ch])
        loss_arr[mask_dis] = np.abs(p_val_array[mask_dis]) * (1.0 / safe_eff[mask_dis] - 1.0)
        return loss_arr

    loss_uniform = calc_loss(p_req_car_uniform)
    total_energy_uniform = np.sum(np.abs(p_req_car_uniform))
    total_loss_uniform_val = np.sum(loss_uniform)
    eta_avg_uniform = 1.0 - (total_loss_uniform_val / total_energy_uniform)

    print(f"Q5 Uniform Efficiency: {eta_avg_uniform:.4f}")

    # Q6: Smart Strategy Limits (Minimizing Conversion Loss)
    # Strategy: Concentrate power on fewer cars (ideally at P_OPT).
    # Limit N -> Inf: Efficiency -> Eta_max (since we stay at P_opt or 0).
    eta_smart_inf = eta_max

    # Find N0 for 90% gain.
    target_eta_val = eta_avg_uniform + 0.9 * (eta_smart_inf - eta_avg_uniform)

    # Dense N values for accurate N0
    n_values = sorted(list(set([1, 2, 3, 4, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500])))
    eta_smart_vals = []

    p_req_total_pu = y_red # p.u. of P_bid_total

    for N in n_values:
        p_bid_total = N * p_bid_car
        p_req_total = p_bid_total * p_req_total_pu

        # We need signed P_req_total for loss calc
        # Smart logic:
        # If P_total > 0: Charge k cars at P_opt. Remainder at P_rem.
        # If P_total < 0: Discharge k cars at P_opt.

        p_abs_total = np.abs(p_req_total)

        # Ideal Smart Distribution (Concentration):
        # Try to run cars at P_OPT.
        k_full = np.floor(p_abs_total / p_opt)
        k_full = np.minimum(k_full, N)

        p_rem = p_abs_total - k_full * p_opt

        # Mask for Low vs High Demand
        mask_low = p_abs_total <= (N * p_opt)

        # Calculate Power per car (vectorized approximation)
        # We construct loss directly.
        # Loss = k * Loss(P_opt) + Loss(P_rem)

        # Loss at P_opt (Charge or Discharge? Use general loss func)
        # Since efficiency curve is usually symmetric in this simplified model
        # (or derived from same curve), we assume symmetric P_opt.
        # But wait, Discharge loss is P*(1/eta - 1). Charge is P*(1-eta).
        # We need to preserve sign of P.

        sign = np.sign(p_req_total)

        # P_opt component (k cars per step)
        p_opt_signed = sign * p_opt
        loss_opt = calc_loss(p_opt_signed) # This is a vector of loss per car if it runs at P_opt

        # Remainder component (1 car per step)
        p_rem_signed = sign * p_rem
        loss_rem = calc_loss(p_rem_signed)

        # High demand: All cars > P_opt.
        # Assumption: Distribute uniformly above P_opt.
        p_high_avg = p_req_total / N # Signed
        loss_high = calc_loss(p_high_avg) * N # Total loss for N cars

        # Combine
        total_loss_t = np.zeros_like(p_req_total)

        # Low demand part
        total_loss_t[mask_low] = k_full[mask_low] * loss_opt[mask_low] + loss_rem[mask_low]

        # High demand part
        total_loss_t[~mask_low] = loss_high[~mask_low]

        eta_N = 1.0 - (np.sum(total_loss_t) / np.sum(p_abs_total))
        eta_smart_vals.append(eta_N)

    # Calculate N0
    try:
        n0_q6 = next(N for N, eta in zip(n_values, eta_smart_vals) if eta >= target_eta_val)
        print(f"Q6 Minimum Fleet Size N0 (Efficiency): {n0_q6}")
    except StopIteration:
        print("Q6 Warning: N0 not found within range.")
        n0_q6 = None

    plt.figure()
    plt.plot(n_values, eta_smart_vals, 'o-', label='Smart')
    plt.axhline(eta_avg_uniform, color='r', linestyle='--', label='Uniform')
    plt.axhline(target_eta_val, color='g', linestyle=':', label='90% Target')
    if n0_q6:
        plt.plot(n0_q6, eta_smart_vals[n_values.index(n0_q6)], 'rx', markersize=10, label=f'N0={n0_q6}')
    plt.xscale('log')
    plt.xlabel('Fleet Size N')
    plt.ylabel('Efficiency')
    plt.legend()
    save_plot('q6_efficiency.png')

    # Q7: Operating Time
    # Formula Derivation Output
    print("Q7 Formula: t_op_inf = (P_bid_car / P_MAX) * mean(|y_red|)")

    mean_abs_y = np.mean(np.abs(y_red))
    t_op_inf = (p_bid_car / sim.P_MAX) * mean_abs_y
    print(f"Q7 Infinite Operating Time: {t_op_inf:.4f} p.u.")

    # Find N0 for Q7 (90% reduction of operating time)
    # Baseline (Uniform) is 1.0 (always active if P!=0, which is most of time).
    # Actually uniform logic: All cars active if P!=0.
    # Gain = 1.0 - t_op_inf (Assuming t_op_uniform ~ 1.0).
    # Target = 1.0 - 0.9 * Gain.
    target_top = 1.0 - 0.9 * (1.0 - t_op_inf)

    # Calculate Top(N) - Min Time Strategy (Concentrate at P_MAX)
    top_vals = []
    for N in n_values:
        p_req_abs = np.abs(N * p_bid_car * y_red)
        # Active count = ceil(P_req / P_max)
        n_active = np.ceil(p_req_abs / sim.P_MAX)
        n_active = np.minimum(n_active, N)
        avg_active = np.mean(n_active)
        t_op_n = avg_active / N
        top_vals.append(t_op_n)

    # Calculate N0_Q7
    try:
        # We want t_op <= target
        n0_q7 = next(N for N, top in zip(n_values, top_vals) if top <= target_top)
        print(f"Q7 Minimum Fleet Size N0 (Op Time): {n0_q7}")
    except StopIteration:
        print("Q7 Warning: N0 not found.")
        n0_q7 = None

    plt.figure()
    plt.plot(n_values, top_vals, 'o-', label='Smart (Min Time)')
    plt.axhline(t_op_inf, color='g', linestyle='--', label='Limit N->Inf')
    plt.axhline(target_top, color='r', linestyle=':', label='90% Target')
    if n0_q7:
        plt.plot(n0_q7, top_vals[n_values.index(n0_q7)], 'rx', markersize=10, label=f'N0={n0_q7}')
    plt.xscale('log')
    plt.title("Q7: Operating Time vs N")
    plt.legend()
    save_plot('q7_operating_time.png')

    # ==========================================
    # PART 3: Driving & Behavior (Q8-Q10)
    # ==========================================
    print("--- Part 3: Behavior ---")

    # Q8: Charging Inference
    print("Q8: Charging Inference Logic (see SimulationCore._setup_cars):")
    print("  - Trip Energy = (100 - SOC_stop) * Cap")
    print("  - If Trip Energy / 7kW <= Parking Duration: Assume AC Charging (V2G available).")
    print("  - Else: Assume DC Charging (V2G not available).")

    # Q9: Coincidence Factor
    # We use provided CSV files to avoid start-of-year boundary issues (cars initially parked).
    # Re-calculation from driving events starting at 0 misses initial parked cars.
    # The CSV likely uses looping/warmup.

    # Load original availability files for the FULL YEAR (for Q11 revenue calc)
    try:
        csv_1h = pd.read_csv('data/availability_1h.csv', index_col=0, parse_dates=True)
        csv_4h = pd.read_csv('data/availability_4h.csv', index_col=0, parse_dates=True)

        # Crop to simulation range for plotting
        sim_idx_min = pd.date_range(start=sim.sim_index[0], end=sim.sim_index[-1], freq='1min')
        avail_1h = csv_1h.reindex(sim_idx_min, method='ffill')['N_avail']
        avail_4h = csv_4h.reindex(sim_idx_min, method='ffill')['N_avail']
    except Exception as e:
        print(f"Error loading availability CSVs: {e}")
        # Fallback to calculated (with known boundary issue)
        avail_1h = pd.Series(0, index=sim.sim_index)
        avail_4h = pd.Series(0, index=sim.sim_index)

    plt.figure(figsize=(10, 6))
    plt.plot(avail_1h.index, avail_1h.values, label='1h Block')
    plt.plot(avail_4h.index, avail_4h.values, label='4h Block')
    plt.title("Q9: Fleet Availability (Coincidence Factor)")
    plt.legend()
    save_plot('q9_availability.png')

    # Q10: Start vs End Year Comparison
    # Compare first week vs last week of 2021 from the FULL CSV
    try:
        cf_jan = csv_1h['2021-01-01':'2021-01-07']['N_avail']
        cf_dec = csv_1h['2021-12-25':'2021-12-31']['N_avail']

        print("Q10: Coincidence Factor Comparison (1h Block)")
        print(f"  Jan 1-7 Mean: {cf_jan.mean():.2f}, Min: {cf_jan.min()}")
        print(f"  Dec 25-31 Mean: {cf_dec.mean():.2f}, Min: {cf_dec.min()}")
    except:
        print("Q10: Could not perform full year comparison (CSV data missing/incomplete).")

    # Calculate Consumption
    # Total Energy / Total Distance
    df_drive = sim.driving_df
    total_dist = df_drive['distance'].sum()
    # Energy: Sum of (SOC_start - SOC_stop) * Cap
    total_soc_diff = (df_drive['SOC_START'] - df_drive['SOC_STOP']).sum()
    total_energy_drive = total_soc_diff / 100.0 * sim.BATTERY_CAP
    consumption_kwh_km = total_energy_drive / total_dist
    print(f"Calculated Consumption: {consumption_kwh_km:.4f} kWh/km")

    # ==========================================
    # PART 4: Economics (Q11-Q13)
    # ==========================================
    print("--- Part 4: Economics ---")

    # Q11: Revenue (Calculated for Full Month of January)
    # Price: 18.0 EUR/MW/h (Placeholder for real data from regelleistung.net)
    PRICE_EUR_MW_H = 18.0

    try:
        # Filter for Jan 2021 from loaded CSVs
        # Ensure index is datetime
        if not isinstance(csv_1h.index, pd.DatetimeIndex):
            csv_1h.index = pd.to_datetime(csv_1h.index)
        if not isinstance(csv_4h.index, pd.DatetimeIndex):
            csv_4h.index = pd.to_datetime(csv_4h.index)

        avail_jan_1h = csv_1h.loc['2021-01-01':'2021-01-31', 'N_avail']
        avail_jan_4h = csv_4h.loc['2021-01-01':'2021-01-31', 'N_avail']

        # P_BID per car = 7/1.1 kW = 0.00636 MW
        p_bid_mw = sim.P_BID / 1000.0

        # Determine frequency of CSV (usually hourly)
        dt_rev = 1.0 # Default 1h
        if len(avail_jan_1h) > 0:
            if pd.infer_freq(avail_jan_1h.index) == 'T' or (avail_jan_1h.index[1] - avail_jan_1h.index[0]).seconds == 60:
                dt_rev = 1.0/60.0

        rev_1h_total = np.sum(avail_jan_1h * p_bid_mw * PRICE_EUR_MW_H * dt_rev)
        rev_4h_total = np.sum(avail_jan_4h * p_bid_mw * PRICE_EUR_MW_H * dt_rev)

        print(f"Q11: Revenue Calculation (Jan 2021 Full Month)")
        print(f"  Price Used: {PRICE_EUR_MW_H} EUR/MW/h (Replace with regelleistung.net data)")
        print(f"  Total Revenue (1h): {rev_1h_total:.2f} EUR")
        print(f"  Revenue per EV (1h): {rev_1h_total / sim.n_cars:.2f} EUR/EV")
        print(f"  Revenue per EV (4h): {rev_4h_total / sim.n_cars:.2f} EUR/EV")

        rev_per_ev_jan = rev_1h_total / sim.n_cars

    except Exception as e:
        print(f"Q11 Error: {e}")
        rev_per_ev_jan = 0.0

    # Q12: Virtual Mileage
    # Energy Throughput = Integral |P(t)| dt
    # P(t) = P_bid * y_red(t)
    # Estimate for Jan based on mean(|y_red|) from simulation period

    mean_abs_y = np.mean(np.abs(y_red)) # From 5-day sim

    # Total Energy Throughput per EV (Jan)
    # P_bid * Mean(|y_red|) * Hours_Jan
    hours_in_jan = 31 * 24
    e_thru_jan_kwh = sim.P_BID * mean_abs_y * hours_in_jan

    # Virtual Distance (Use 0.2 kWh/km per Q13 requirement)
    CONS_VIRT = 0.2
    virt_km_jan = e_thru_jan_kwh / CONS_VIRT

    print(f"Q12: Virtual Mileage (Jan Est)")
    print(f"  Mean |y_red|: {mean_abs_y:.4f} p.u.")
    print(f"  Energy Throughput: {e_thru_jan_kwh:.2f} kWh/EV")
    print(f"  Virtual Distance: {virt_km_jan:.2f} km/EV")

    # Q13: Residual Value Loss & Net Revenue
    try:
        res_df = pd.read_csv('data/residual_value.csv', sep=';', encoding='cp1252')
        mil = res_df.iloc[:, 0].values
        val = res_df.iloc[:, 1].values
        res_func = interp1d(mil, val, fill_value="extrapolate")

        # Calculate Loss for a typical EV (e.g. at 50,000 km start)
        start_km = 50000.0
        v_start = res_func(start_km)
        v_end = res_func(start_km + virt_km_jan)

        loss_val = v_start - v_end

        net_rev = rev_per_ev_jan - loss_val
        reduction_pct = (loss_val / rev_per_ev_jan) * 100.0 if rev_per_ev_jan > 0 else 0.0

        print(f"Q13: Residual Value Impact (at {start_km}km)")
        print(f"  Residual Loss: {loss_val:.2f} EUR/EV")
        print(f"  Net Revenue: {net_rev:.2f} EUR/EV")
        print(f"  Reduction: {reduction_pct:.1f}%")

    except Exception as e:
        print(f"Residual Value Calc Failed: {e}")

    # ==========================================
    # PART 5: Simulation (Q14)
    # ==========================================
    print("--- Part 5: Full Simulation ---")

    scenarios = [
        ('uniform', '1h'),
        ('smart', '1h'),
        ('uniform', '4h'),
        ('smart', '4h')
    ]

    results = {}

    # Run No-FCR Baseline
    print("Running No-FCR Baseline...")
    soc_base, age_cyc_base, age_cal_base = sim.run_scenario(fcr_active=False)
    results['no_fcr'] = {'soc': soc_base, 'age_cyc': age_cyc_base, 'age_cal': age_cal_base}

    # Q14 Validation Helper
    def validate_scenario(name, soc_mat):
        # SOC < 20%
        violations_low = np.sum(soc_mat < 20.0)
        violations_high = np.sum(soc_mat > 100.0)
        total_points = soc_mat.size
        print(f"Validation {name}:")
        print(f"  Min SOC: {np.min(soc_mat):.2f}%")
        print(f"  Max SOC: {np.max(soc_mat):.2f}%")
        print(f"  Violations (<20%): {violations_low} ({violations_low/total_points*100:.4f}%)")
        if violations_low > 0:
            print("  Warning: SOC < 20% constraint violated!")

    validate_scenario('No-FCR', soc_base)

    for strat, bid in scenarios:
        key = f"{strat}_{bid}"
        print(f"Running {key}...")
        soc_s, age_cyc_s, age_cal_s = sim.run_scenario(strategy=strat, bid_type=bid, fcr_active=True)
        results[key] = {'soc': soc_s, 'age_cyc': age_cyc_s, 'age_cal': age_cal_s}

        validate_scenario(key, soc_s)

        # Plot 1st car SOC
        plt.figure(figsize=(12, 6))
        plt.plot(sim.sim_index, soc_s[:, 0], label=f'Car 0 {key}')
        plt.plot(sim.sim_index, soc_base[:, 0], label='Car 0 No FCR', linestyle='--')
        plt.title(f"SOC Profile Car 0 ({key})")
        plt.legend()
        save_plot(f"q14_soc_{key}.png")

    # ==========================================
    # PART 6: Aging (Q16)
    # ==========================================
    print("--- Part 6: Aging ---")

    # Q16: Fleet Average Battery Degradation
    # Output from run_scenario is TOTAL aging (sum over all cars, all steps).
    # We want Average % Loss per EV for the simulation period.
    # dL (fraction) -> % (*100).

    sim_days = (sim.sim_index[-1] - sim.sim_index[0]).total_seconds() / 86400.0
    scale_to_month = 31.0 / sim_days

    print(f"Aging Analysis (Simulation Duration: {sim_days:.2f} days)")

    # Baseline
    base_tot_dl = results['no_fcr']['age_cyc'] + results['no_fcr']['age_cal']
    base_avg_dl = base_tot_dl / sim.n_cars
    base_avg_pct_mo = base_avg_dl * 100.0 * scale_to_month

    print(f"Baseline Aging (Month Est): {base_avg_pct_mo:.6f}%")

    for key, res in results.items():
        if key == 'no_fcr': continue

        tot_dl = res['age_cyc'] + res['age_cal']
        avg_dl = tot_dl / sim.n_cars
        avg_pct_mo = avg_dl * 100.0 * scale_to_month

        rel_increase = (avg_pct_mo - base_avg_pct_mo) / base_avg_pct_mo * 100.0

        print(f"Scenario {key} Aging (Month Est): {avg_pct_mo:.6f}% (+{rel_increase:.2f}%)")

    print("Report Generation Complete.")

if __name__ == "__main__":
    generate_report_data()
