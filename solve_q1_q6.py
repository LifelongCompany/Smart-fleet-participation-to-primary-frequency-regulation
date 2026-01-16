import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import timedelta
from scipy.interpolate import interp1d

# Constants
SIM_START_STR = '2021-01-01 00:00:00'
SIM_END_STR = '2021-01-31 23:59:50'
FREQ_STEP_SEC = 10
BATTERY_CAPACITY_KWH = 40  # Assumed generic EV capacity (e.g., Renault Zoe)
AC_CHARGER_KW = 7.0
FCR_GAIN_FACTOR = 1.1 # 1.1 (safety margin?)
K_FACTOR = 5.0 # Hz^-1

def load_and_preprocess_data():
    print("Loading data...")
    freq_df = pd.read_csv('data/france_2019_05.csv', parse_dates=[0], index_col=0)
    freq_df['Frequency'] = freq_df['Frequency'] / 1000.0 # mHz -> Hz

    driving_df = pd.read_csv('data/driving_sessions.csv', sep=';')
    driving_df['START'] = pd.to_datetime(driving_df['START'], format='mixed', utc=True)
    driving_df['STOP'] = pd.to_datetime(driving_df['STOP'], format='mixed', utc=True)
    driving_df['START'] = driving_df['START'].dt.tz_localize(None)
    driving_df['STOP'] = driving_df['STOP'].dt.tz_localize(None)

    obc_df = pd.read_csv('data/obc_efficiency.csv', sep=';', skiprows=1, names=['Power_kW', 'Efficiency'])
    # Ensure proper ordering and uniqueness for interpolation
    obc_df = obc_df.sort_values('Power_kW').drop_duplicates('Power_kW')

    try:
        residual_df = pd.read_csv('data/residual_value.csv', sep=';', encoding='latin1')
    except:
        residual_df = pd.read_csv('data/residual_value.csv', sep=';', encoding='utf-8')

    print("Data loaded successfully.")
    return freq_df, driving_df, obc_df, residual_df

def prepare_simulation_time_series(freq_df):
    print("Preparing simulation time series...")
    sim_index = pd.date_range(start=SIM_START_STR, end=SIM_END_STR, freq=f'{FREQ_STEP_SEC}s')
    freq_values = freq_df['Frequency'].values
    required_length = len(sim_index)
    tiled_freq = np.resize(freq_values, required_length)
    sim_df = pd.DataFrame(index=sim_index)
    sim_df['Frequency'] = tiled_freq
    print(f"Simulation time series created. Shape: {sim_df.shape}")
    return sim_df

def process_parking_sessions(driving_df):
    print("Processing parking sessions...")
    driving_df = driving_df.sort_values(by=['ID', 'START'])
    driving_df['NEXT_START'] = driving_df.groupby('ID')['START'].shift(-1)
    driving_df['NEXT_SOC_START'] = driving_df.groupby('ID')['SOC_START'].shift(-1)
    driving_df['NEXT_SOC_STOP'] = driving_df.groupby('ID')['SOC_STOP'].shift(-1)
    driving_df['PARKING_DURATION_H'] = (driving_df['NEXT_START'] - driving_df['STOP']).dt.total_seconds() / 3600.0
    driving_df['NEXT_E_TRIP_KWH'] = (driving_df['NEXT_SOC_START'] - driving_df['NEXT_SOC_STOP']) / 100.0 * BATTERY_CAPACITY_KWH

    def check_ac(row):
        if pd.isna(row['PARKING_DURATION_H']): return True
        if row['PARKING_DURATION_H'] <= 0: return False
        required_time = row['NEXT_E_TRIP_KWH'] / AC_CHARGER_KW
        return required_time <= row['PARKING_DURATION_H']

    driving_df['IS_AC'] = driving_df.apply(check_ac, axis=1)
    print("Parking sessions processed.")
    return driving_df

def calculate_fleet_availability(sim_index, driving_df):
    print("Calculating fleet availability...")
    ac_sessions = driving_df[driving_df['IS_AC'] == True].copy()
    events_start = pd.DataFrame({'time': ac_sessions['STOP'], 'change': 1})
    events_end = pd.DataFrame({'time': ac_sessions['NEXT_START'], 'change': -1})
    events = pd.concat([events_start, events_end]).sort_values('time')
    events_grouped = events.groupby('time')['change'].sum()
    events_cum = events_grouped.cumsum()
    availability = events_cum.reindex(sim_index, method='ffill').fillna(0)
    return availability

def perform_initial_analysis(sim_df, driving_df):
    print("Performing initial analysis (Q1-Q10)...")
    plt.figure(figsize=(10, 6))
    sim_df['Frequency'].hist(bins=100)
    plt.title("Frequency Deviation Distribution (Jan 2021)")
    plt.savefig("images/frequency_distribution.png")
    plt.close()

    drift = sim_df['Frequency'].cumsum() * FREQ_STEP_SEC
    plt.figure(figsize=(10, 6))
    plt.plot(sim_df.index, drift)
    plt.title("Theoretical Energy Drift (Integral of Frequency)")
    plt.savefig("images/theoretical_soc_drift.png")
    plt.close()

    availability_series = calculate_fleet_availability(sim_df.index, driving_df)
    plt.figure(figsize=(10, 6))
    plt.plot(availability_series.index, availability_series.values)
    plt.title("Number of Available AC-Connected EVs")
    plt.savefig("images/fleet_availability.png")
    plt.close()

    print("Initial analysis complete. Images saved.")
    return availability_series

# ==========================================
# Q1-Q6 Specific Analysis Functions
# ==========================================

def analyze_reduced_signal(sim_df):
    """Q1: Plot distribution of reduced regulating power."""
    print("Analyzing Reduced Signal (Q1)...")
    y_red = K_FACTOR * sim_df['Frequency']

    plt.figure(figsize=(10, 6))
    y_red.hist(bins=100, density=True)
    plt.title("Distribution of Reduced Regulating Power (p.u.)")
    plt.xlabel("Regulating Power (p.u. of P_bid)")
    plt.ylabel("Probability Density")
    plt.savefig("images/q1_reduced_power_dist.png")
    plt.close()
    return y_red

def analyze_rolling_windows(sim_df):
    """Q3: Plot SOC deviation for rolling windows."""
    print("Analyzing Rolling Windows (Q3)...")
    p_bid_ev = AC_CHARGER_KW / FCR_GAIN_FACTOR
    p_ev_series = p_bid_ev * K_FACTOR * sim_df['Frequency'] # kW
    dt_h = FREQ_STEP_SEC / 3600.0
    e_step = p_ev_series * dt_h # kWh

    windows_hours = [4, 8, 12, 24]

    plt.figure(figsize=(12, 6))
    data_to_plot = []
    labels = []

    for wh in windows_hours:
        steps = int(wh * 3600 / FREQ_STEP_SEC)
        rolling_e = e_step.rolling(window=steps).sum().dropna()
        data_to_plot.append(rolling_e.values)
        labels.append(f"{wh}h")

    plt.boxplot(data_to_plot, tick_labels=labels)
    plt.title("SOC Deviation for Rolling Windows (Single EV)")
    plt.ylabel("Energy Deviation (kWh)")
    plt.grid(True)
    plt.savefig("images/q3_rolling_soc_deviation.png")
    plt.close()

def analyze_smart_limits(obc_df, sim_df):
    """Q6: Analyze limits of smart strategy."""
    print("Analyzing Smart Strategy Limits (Q6)...")

    xp = obc_df['Power_kW'].values
    fp = obc_df['Efficiency'].values
    eff_func = interp1d(xp, fp, kind='linear', fill_value='extrapolate')

    p_scan = np.linspace(0, AC_CHARGER_KW, 1000)
    e_scan = eff_func(p_scan)
    eta_max = np.max(e_scan)
    p_opt = p_scan[np.argmax(e_scan)]

    eta_smart_inf = eta_max

    p_bid_ev = AC_CHARGER_KW / FCR_GAIN_FACTOR
    p_req_ev_profile = p_bid_ev * K_FACTOR * sim_df['Frequency'].values
    p_req_ev_abs = np.abs(p_req_ev_profile)

    eff_uniform_profile = eff_func(np.clip(p_req_ev_abs, 0, AC_CHARGER_KW))
    total_energy = np.sum(p_req_ev_abs)
    total_loss_uniform = np.sum(p_req_ev_abs * (1 - eff_uniform_profile))
    eta_uniform = 1.0 - (total_loss_uniform / total_energy)

    target_eta = eta_uniform + 0.9 * (eta_smart_inf - eta_uniform)

    n_values = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200]
    eta_smart_n = []

    for N in n_values:
        p_fleet = N * p_req_ev_abs
        k_full = np.floor(p_fleet / p_opt)
        p_rem = p_fleet - k_full * p_opt

        loss_full = k_full * p_opt * (1 - eta_max)
        eff_rem = eff_func(np.clip(p_rem, 0, AC_CHARGER_KW))
        loss_rem = p_rem * (1 - eff_rem)

        total_loss_n = np.sum(loss_full + loss_rem)
        total_energy_n = np.sum(p_fleet)

        eta_n = 1.0 - (total_loss_n / total_energy_n)
        eta_smart_n.append(eta_n)

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, eta_smart_n, marker='o', label='Smart Efficiency')
    plt.axhline(eta_uniform, color='r', linestyle='--', label='Uniform')
    plt.axhline(eta_smart_inf, color='g', linestyle='--', label='Smart Limit')
    plt.axhline(target_eta, color='orange', linestyle=':', label='90% Target')
    plt.xscale('log')
    plt.xlabel('Fleet Size (N)')
    plt.ylabel('Average Efficiency')
    plt.legend()
    plt.title("Efficiency Convergence with Smart Strategy")
    plt.savefig("images/q6_efficiency_convergence.png")
    plt.close()

    return {
        'eta_uniform': eta_uniform,
        'eta_smart_inf': eta_smart_inf,
        'n0': next((n for n, e in zip(n_values, eta_smart_n) if e >= target_eta), None),
        'p_opt': p_opt
    }

def perform_simulation_integrated(sim_df, driving_df, obc_df, availability_series, q6_results):
    print("Starting Integrated Simulation...")

    P_MAX_PER_CAR = 7.0
    dt_h = FREQ_STEP_SEC / 3600.0

    N_t = availability_series.values
    P_bid_total = N_t * (P_MAX_PER_CAR / FCR_GAIN_FACTOR)
    f_dev = sim_df['Frequency'].values
    P_RE_total = K_FACTOR * P_bid_total * f_dev

    xp = obc_df['Power_kW'].values
    fp = obc_df['Efficiency'].values

    # Uniform
    with np.errstate(divide='ignore', invalid='ignore'):
        p_per_car = np.abs(P_RE_total / N_t)
    p_per_car[N_t==0] = 0
    eff_uniform = np.interp(p_per_car, xp, fp)

    mask_ch = P_RE_total > 0
    mask_dis = P_RE_total < 0
    throughput_uniform_power = np.zeros_like(P_RE_total)
    throughput_uniform_power[mask_ch] = P_RE_total[mask_ch] * eff_uniform[mask_ch]
    with np.errstate(divide='ignore'):
        dis_val = np.abs(P_RE_total[mask_dis]) / eff_uniform[mask_dis]
    dis_val[np.isinf(dis_val)] = 0
    throughput_uniform_power[mask_dis] = dis_val
    total_throughput_uniform_kwh = np.sum(throughput_uniform_power) * dt_h

    # Smart
    p_opt = q6_results['p_opt']
    eta_max = np.max(fp)
    eff_func = interp1d(xp, fp, kind='linear', fill_value='extrapolate')

    p_re_abs = np.abs(P_RE_total)
    k_full = np.floor(p_re_abs / p_opt)
    k_full = np.minimum(k_full, N_t)
    p_rem = p_re_abs - k_full * p_opt

    loss_full = k_full * p_opt * (1 - eta_max)
    eff_rem = eff_func(np.clip(p_rem, 0, AC_CHARGER_KW))
    loss_rem = p_rem * (1 - eff_rem)
    total_loss_smart = loss_full + loss_rem

    throughput_smart_power = np.zeros_like(P_RE_total)
    throughput_smart_power[mask_ch] = P_RE_total[mask_ch] - total_loss_smart[mask_ch]
    throughput_smart_power[mask_dis] = np.abs(P_RE_total[mask_dis]) + total_loss_smart[mask_dis]
    total_throughput_smart_kwh = np.sum(throughput_smart_power) * dt_h

    # Economics
    FCR_PRICE_EUR_MW_H = 18.0
    BATTERY_COST_EUR_KWH = 150.0
    CYCLE_LIFE = 3000

    revenue = np.sum((P_bid_total / 1000.0) * FCR_PRICE_EUR_MW_H * dt_h)
    aging_cost_per_kwh = BATTERY_COST_EUR_KWH / (CYCLE_LIFE * 2)

    cost_uniform = total_throughput_uniform_kwh * aging_cost_per_kwh
    cost_smart = total_throughput_smart_kwh * aging_cost_per_kwh

    return {
        'revenue': revenue,
        'uniform': {'throughput': total_throughput_uniform_kwh, 'cost': cost_uniform, 'profit': revenue - cost_uniform},
        'smart': {'throughput': total_throughput_smart_kwh, 'cost': cost_smart, 'profit': revenue - cost_smart}
    }

if __name__ == "__main__":
    if not os.path.exists('images'):
        os.makedirs('images')

    freq_df, driving_df, obc_df, residual_df = load_and_preprocess_data()
    sim_df = prepare_simulation_time_series(freq_df)
    driving_df = process_parking_sessions(driving_df)

    analyze_reduced_signal(sim_df)
    availability_series = perform_initial_analysis(sim_df, driving_df)
    analyze_rolling_windows(sim_df)
    q6_results = analyze_smart_limits(obc_df, sim_df)
    econ_results = perform_simulation_integrated(sim_df, driving_df, obc_df, availability_series, q6_results)
