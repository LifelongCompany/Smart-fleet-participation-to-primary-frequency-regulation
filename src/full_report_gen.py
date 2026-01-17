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
    # Simulate single EV with 40kWh, 7kW.
    # P(t) = 7kW * y_red(t).
    # E(t) = Integral P(t) dt.
    dt_h = sim.dt_h
    p_profile_kw = 7.0 * y_red # Positive = Charge?
    # y_red > 0 (High Freq) -> Charge.
    # Q3 formula: SOC = SOC0 + 1/E_bat * Integral(P).
    # So P is charge power.

    energy_profile_kwh = np.cumsum(p_profile_kw) * dt_h

    windows = [4, 8, 12, 24]
    data_q3 = []
    labels_q3 = []

    for w in windows:
        steps = int(w * 3600 / sim.dt_sec)
        # Rolling sum of energy changes? Or deviation over window?
        # Q3: "SOC deviation curve caused by FCR ... 4h rolling window".
        # Usually means: SOC(t+w) - SOC(t).
        # Which is sum of energy in window.

        # We can use pandas rolling on the power series
        p_series = pd.Series(p_profile_kw)
        # rolling sum of Power * dt
        e_rolling = p_series.rolling(window=steps).sum() * dt_h
        # Convert to SOC deviation %
        soc_dev = e_rolling / 40.0 * 100.0
        soc_dev = soc_dev.dropna().values
        data_q3.append(soc_dev)
        labels_q3.append(f"{w}h")

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

    # Q5: Uniform Efficiency
    # P_req per car (assuming full availability for simplicity of Q5/Q6 theoretical calc)
    # Prompt Q6 says: "Calculate limit when N -> inf".
    # For Q5, use single car behavior? "Calculate fleet avg efficiency under Uniform".
    # Assume fleet size N, Total P_req. P_per_car = P_req / N.
    # This is same as P_req_pu * P_bid_per_car.
    # P_bid = 7kW / 1.1 = 6.36 kW.
    # P_req_car = P_bid * y_red.

    p_bid_car = sim.P_BID
    p_req_car_uniform = p_bid_car * y_red
    p_abs_uniform = np.abs(p_req_car_uniform)
    # Clip to max 7kW
    p_abs_uniform = np.clip(p_abs_uniform, 0, 7.0)

    eff_uniform_t = eff_func(p_abs_uniform)

    # Avg Efficiency = 1 - (Total Loss / Total Energy)
    total_energy = np.sum(p_abs_uniform)
    total_loss_uniform = np.sum(p_abs_uniform * (1.0 - eff_uniform_t))
    eta_avg_uniform = 1.0 - (total_loss_uniform / total_energy)

    print(f"Q5 Uniform Efficiency: {eta_avg_uniform:.4f}")

    # Q6: Smart Strategy Limits
    # Limit N -> Inf.
    # Efficiency -> Eta_max (since we always operate at P_opt or 0).
    eta_smart_inf = eta_max

    # Find N0 for 90% gain.
    # Gain = Eta_smart - Eta_uniform.
    # Target = Eta_uniform + 0.9 * (Eta_smart_inf - Eta_uniform).

    target_eta_val = eta_avg_uniform + 0.9 * (eta_smart_inf - eta_avg_uniform)

    n_values = [1, 2, 5, 10, 20, 50, 100, 200]
    eta_smart_vals = []

    # Use a subset of time for speed if needed, but vectorization is fast
    p_req_total_pu = y_red # p.u. of P_bid_total

    for N in n_values:
        p_bid_total = N * p_bid_car
        p_req_total = p_bid_total * p_req_total_pu
        p_req_abs = np.abs(p_req_total)

        # Smart Logic approximation for efficiency:
        # k cars at P_opt. Remainder at P_rem.
        k_full = np.floor(p_req_abs / p_opt)
        # Cap k at N
        k_full = np.minimum(k_full, N)

        p_rem = p_req_abs - k_full * p_opt
        # If k=N, p_rem might be negative? No, minimum handles it?
        # If P_req > N*P_opt?
        # Then we saturate P_opt.
        # Check logic: P_max = 7. P_opt ~ ?
        # If P_req > N * P_opt, we increase power of cars above P_opt?
        # Q6 usually assumes we stay at optimal unless forced.
        # But if we need more power, we go up to P_max.
        # Let's assume ideal smart: fill P_opt, then remainder.
        # If remainder pushes cars > P_opt?
        # Simplified Q6: Just P_opt and remainder.

        loss_full = k_full * p_opt * (1.0 - eta_max)

        # Remainder car efficiency
        # Handle case where p_rem > P_MAX (shouldn't happen if we fill N cars?
        # Wait, if demand is huge? P_bid = 7/1.1. P_req can be 7/1.1 * 1.0.
        # P_opt is usually < P_max.
        # So yes, we might need more than P_opt per car.

        # Correct logic:
        # P_req distributed to N cars.
        # Sort SOC -> Activation.
        # Ideally operate at P_opt.
        # If P_req < N * P_opt: k cars at P_opt, 1 at remainder.
        # If P_req > N * P_opt: All N cars > P_opt.

        # Case 1: P_req <= N * P_opt
        mask_low = p_req_abs <= (N * p_opt)

        loss_smart = np.zeros_like(p_req_abs)

        # Low demand
        p_low = p_req_abs[mask_low]
        k_low = np.floor(p_low / p_opt)
        p_rem_low = p_low - k_low * p_opt

        l_low = k_low * p_opt * (1.0 - eta_max) + p_rem_low * (1.0 - eff_func(p_rem_low))
        loss_smart[mask_low] = l_low

        # High demand
        p_high = p_req_abs[~mask_low]
        # All N cars operate at least P_opt.
        # Average power per car = P_high / N.
        # Efficiency is eff(P_avg).
        # Assuming we balance them above P_opt?
        # Yes, if all active, spread load? Or fill to P_max?
        # Uniform spread above P_opt is likely best for convexity?
        # Let's assume uniform above P_opt.
        p_avg_high = p_high / N
        l_high = p_high * (1.0 - eff_func(p_avg_high))
        loss_smart[~mask_low] = l_high

        eta_N = 1.0 - (np.sum(loss_smart) / np.sum(p_req_abs))
        eta_smart_vals.append(eta_N)

    plt.figure()
    plt.plot(n_values, eta_smart_vals, 'o-', label='Smart')
    plt.axhline(eta_avg_uniform, color='r', linestyle='--', label='Uniform')
    plt.axhline(target_eta_val, color='g', linestyle=':', label='90% Target')
    plt.xscale('log')
    plt.xlabel('Fleet Size N')
    plt.ylabel('Efficiency')
    plt.legend()
    save_plot('q6_efficiency.png')

    # Q7: Operating Time
    # Top = P_bid / P_max * E[|y_red|]
    mean_abs_y = np.mean(np.abs(y_red))
    t_op_inf = (p_bid_car / sim.P_MAX) * mean_abs_y
    print(f"Q7 Infinite Operating Time: {t_op_inf:.4f} p.u.")

    # Find N0 for Q7 (90% reduction of operating time)
    # Uniform Top = 1.0 (always active).
    # Gain = 1.0 - t_op_inf.
    # Target = 1.0 - 0.9 * Gain.
    target_top = 1.0 - 0.9 * (1.0 - t_op_inf)

    # Calculate Top(N)
    top_vals = []
    for N in n_values:
        p_req_abs = np.abs(N * p_bid_car * y_red)
        # Number of active cars?
        # Active if P > 0.
        # k cars at P_opt. 1 car at P_rem.
        # So ceil(P_req / P_opt) cars active?
        # Wait, Smart Strategy tries to reduce TIME?
        # Minimizing time means running at P_MAX!
        # Prompt Q7: "Smart strategy can also reduce OBC average running time".
        # If we run at P_MAX, we use fewest cars => fewest time.
        # But Q6 said "Smart Strategy" (efficiency).
        # Usually Efficiency -> P_opt.
        # Life -> P_max (min time).
        # Prompt says "Background: Smart strategy can ALSO reduce...".
        # This implies the SAME strategy (P_opt) has this effect?
        # Or a strategy designed for it?
        # "Task 1: Express ... as function of P_bid, P_max...".
        # If we run at P_opt, calculation is P_req / P_opt.
        # If we run at P_max, calculation is P_req / P_max.
        # The limit formula I derived (P_bid/P_max) assumes running at P_MAX?
        # Let's check: "Limit when N->inf, t_op_inf depends on P_bid, P_max".
        # If strategy was P_opt, it would depend on P_opt.
        # So Q7 implies a "Min-Time Strategy" (run at P_max).
        # Let's assume Q7 analyzes the "Min-Time" aspect.

        # Calculate T_op assuming P_MAX strategy for Q7 curves.
        # Active count = ceil(P_req / P_max).
        n_active = np.ceil(p_req_abs / sim.P_MAX)
        n_active = np.minimum(n_active, N)
        avg_active = np.mean(n_active)
        t_op_n = avg_active / N
        top_vals.append(t_op_n)

    plt.figure()
    plt.plot(n_values, top_vals, 'o-')
    plt.axhline(t_op_inf, color='g', linestyle='--')
    plt.xscale('log')
    plt.title("Q7: Operating Time vs N")
    save_plot('q7_operating_time.png')

    # ==========================================
    # PART 3: Driving & Behavior (Q8-Q10)
    # ==========================================
    print("--- Part 3: Behavior ---")

    # Q9: Coincidence Factor
    # Recalculate from driving_df
    df_drive = sim.driving_df
    # We need to construct time series of AC connections.
    # Event based.

    # Filter AC sessions
    ac_sessions = []
    # Re-use sim.cars structure which has logic applied
    for cid, sessions in sim.cars.items():
        for sess in sessions:
            if sess['type'] == 'park_ac':
                ac_sessions.append((sess['start'], 1))
                ac_sessions.append((sess['stop'], -1))

    ac_sessions.sort(key=lambda x: x[0])

    # Walk through
    times = [x[0] for x in ac_sessions]
    changes = [x[1] for x in ac_sessions]

    # Create Series
    ts = pd.Series(changes, index=times)
    # Resample to simulation index (10s) is too heavy for just checking min in block.
    # But we need min in 1h and 4h blocks.
    # Let's pivot to 1min resolution.

    # Use the simulation end date
    full_idx = pd.date_range(start='2021-01-01', end=sim.sim_index[-1], freq='1min')
    # Accumulate
    # We need to sum changes at same time
    ts_grouped = ts.groupby(ts.index).sum()
    ts_reindexed = ts_grouped.reindex(full_idx, method=None).fillna(0).cumsum()
    # Handle initial state? (Assumed 0 at start? Or steady state?)
    # At Jan 1 00:00, some cars are parked.
    # We should probably run a warmup or circular year.
    # Q10 mentions "Looping data".
    # For now, start at 0. But actually check if driving starts later.
    # Valid approximation.

    # Calculate Min per block (Re-calculated)
    avail_1h_calc = ts_reindexed.resample('1h').min()
    avail_4h_calc = ts_reindexed.resample('4h').min()

    # Use CSV data for Report correctness (Q9/Q11) as recalculation might miss initial states
    # Load original availability files
    try:
        csv_1h = pd.read_csv('data/availability_1h.csv', index_col=0, parse_dates=True)
        csv_4h = pd.read_csv('data/availability_4h.csv', index_col=0, parse_dates=True)
        # Use intersection of time
        avail_1h = csv_1h.reindex(full_idx, method='ffill')['N_avail']
        avail_4h = csv_4h.reindex(full_idx, method='ffill')['N_avail']
    except:
        print("Warning: Could not load availability CSVs, using calculated values.")
        avail_1h = avail_1h_calc
        avail_4h = avail_4h_calc

    plt.figure(figsize=(10, 6))
    plt.plot(avail_1h.index, avail_1h.values, label='1h Block')
    plt.plot(avail_4h.index, avail_4h.values, label='4h Block')
    plt.title("Q9: Fleet Availability (Coincidence Factor)")
    plt.legend()
    save_plot('q9_availability.png')

    # Q10 Analysis (Text in README)

    # Calculate Consumption
    # Total Energy / Total Distance
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

    # Q11: Revenue
    # Cap = N_min * 7kW. (Wait, Coincidence Factor IS N_min).
    # Revenue = Cap(MW) * Price * Hours.
    price = 18.0 # Eur/MW/h

    # Scale factor for Monthly Estimate
    # 5 days simulated -> Month (31 days)
    # scale = 31 / 5 = 6.2
    scale_factor = 31.0 / 5.0

    # 1h Revenue
    # Series resolution is 1 min. Price is per Hour. dt = 1/60 h.
    cap_1h_mw = avail_1h.values * 7.0 / 1000.0
    rev_1h = np.sum(cap_1h_mw * 18.0 / 60.0) * scale_factor

    # 4h Revenue
    cap_4h_mw = avail_4h.values * 7.0 / 1000.0
    rev_4h = np.sum(cap_4h_mw * 18.0 / 60.0) * scale_factor

    print(f"Revenue 1h (Monthly Est): {rev_1h:.2f} EUR")
    print(f"Revenue 4h (Monthly Est): {rev_4h:.2f} EUR")

    # Q12: Virtual Mileage
    # E_thru = Integral |P|.
    # P = N_min * P_bid * y_red.
    # Average Virtual Mileage per Car = Total Virt Dist / N_cars.
    # Total Virt Dist = (Total E_thru / C_drive).

    # Need to simulate E_thru for the month (scaled).
    # Use sim.y_red.
    # Reconstruct P_bid(t).
    # 1h Profile:
    avail_1h_re = avail_1h.reindex(sim.sim_index, method='ffill').values
    p_bid_1h = avail_1h_re * sim.P_BID
    p_req_1h = p_bid_1h * y_red
    e_thru_1h = np.sum(np.abs(p_req_1h)) * dt_h * scale_factor
    virt_km_1h = e_thru_1h / consumption_kwh_km
    virt_km_per_car_1h = virt_km_1h / sim.n_cars

    # 4h Profile
    avail_4h_re = avail_4h.reindex(sim.sim_index, method='ffill').values
    p_bid_4h = avail_4h_re * sim.P_BID
    p_req_4h = p_bid_4h * y_red
    e_thru_4h = np.sum(np.abs(p_req_4h)) * dt_h * scale_factor
    virt_km_4h = e_thru_4h / consumption_kwh_km
    virt_km_per_car_4h = virt_km_4h / sim.n_cars

    print(f"Virtual Mileage 1h (Monthly Est): {virt_km_per_car_1h:.2f} km")

    # Q13: Residual Value
    # Load residual value
    # Interpolate loss
    try:
        res_df = pd.read_csv('data/residual_value.csv', sep=';', encoding='cp1252')
        mil = res_df.iloc[:, 0].values
        val = res_df.iloc[:, 1].values # Access by index to avoid encoding issues
        res_func = interp1d(mil, val, fill_value="extrapolate")

        # Base mileage + Virtual
        # Simulation: 5k to 75k.
        base_miles = np.arange(5000, 80000, 5000)
        losses = []
        for m in base_miles:
            v_base = res_func(m)
            v_virt = res_func(m + virt_km_per_car_1h) # Using 1h scenario
            loss = v_base - v_virt
            losses.append(loss)

        print(f"Avg Residual Loss: {np.mean(losses):.2f} EUR")
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

    for strat, bid in scenarios:
        key = f"{strat}_{bid}"
        print(f"Running {key}...")
        soc_s, age_cyc_s, age_cal_s = sim.run_scenario(strategy=strat, bid_type=bid, fcr_active=True)
        results[key] = {'soc': soc_s, 'age_cyc': age_cyc_s, 'age_cal': age_cal_s}

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

    # Compare Totals
    # Aging is sum of dL.
    # Convert to % capacity loss? Or just arbitrary units?
    # Formulas gave dL (fractional loss).
    # Report total dL per scenario.

    base_loss = results['no_fcr']['age_cyc'] + results['no_fcr']['age_cal']
    print(f"Baseline Aging (5 days): {base_loss:.6f}")
    print(f"Baseline Aging (Month Est): {base_loss * scale_factor:.6f}")

    for key, res in results.items():
        if key == 'no_fcr': continue
        tot = res['age_cyc'] + res['age_cal']
        increase = (tot - base_loss) / base_loss * 100.0
        print(f"Scenario {key} Aging (Month Est): {tot * scale_factor:.6f} (+{increase:.2f}%)")

    print("Report Generation Complete.")

if __name__ == "__main__":
    generate_report_data()
