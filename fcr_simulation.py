import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import timedelta

# Constants
SIM_START_STR = '2021-01-01 00:00:00'
SIM_END_STR = '2021-01-31 23:59:50'
FREQ_STEP_SEC = 10
BATTERY_CAPACITY_KWH = 40  # Assumed generic EV capacity (e.g., Renault Zoe)
AC_CHARGER_KW = 7.0

def load_and_preprocess_data():
    print("Loading data...")
    freq_df = pd.read_csv('data/france_2019_05.csv', parse_dates=[0], index_col=0)
    freq_df['Frequency'] = freq_df['Frequency'] / 1000.0

    driving_df = pd.read_csv('data/driving_sessions.csv', sep=';')
    driving_df['START'] = pd.to_datetime(driving_df['START'], format='mixed', utc=True)
    driving_df['STOP'] = pd.to_datetime(driving_df['STOP'], format='mixed', utc=True)
    driving_df['START'] = driving_df['START'].dt.tz_localize(None)
    driving_df['STOP'] = driving_df['STOP'].dt.tz_localize(None)

    obc_df = pd.read_csv('data/obc_efficiency.csv', sep=';', skiprows=1, names=['Power_kW', 'Efficiency'])

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

def get_efficiency(power_kw, obc_df):
    return np.interp(np.abs(power_kw), obc_df['Power_kW'], obc_df['Efficiency'])

def perform_simulation(sim_df, driving_df, obc_df, availability_series):
    print("Starting Vectorized Simulation (Jan 2021)...")

    # Parameters
    P_MAX_PER_CAR = 7.0 # kW
    K_FACTOR = 5.0 # Hz^-1
    FCR_GAIN_FACTOR = 1.1
    dt_h = FREQ_STEP_SEC / 3600.0

    # 1. Calculate Aggregate Variables
    N_t = availability_series.values
    P_bid_total = N_t * (P_MAX_PER_CAR / FCR_GAIN_FACTOR)
    f_dev = sim_df['Frequency'].values
    P_RE_total = K_FACTOR * P_bid_total * f_dev

    # --- UNIFORM STRATEGY ---
    print("Simulating Uniform Strategy...")
    unique_ids = driving_df['ID'].unique()

    xp = obc_df['Power_kW'].values
    fp = obc_df['Efficiency'].values

    with np.errstate(divide='ignore', invalid='ignore'):
        P_per_car_profile = P_RE_total / N_t
    P_per_car_profile[N_t == 0] = 0.0

    Eff_uniform = np.interp(np.abs(P_per_car_profile), xp, fp)

    fleet_soc_energy = np.zeros(len(sim_df))
    base_fleet_energy = np.zeros(len(sim_df))
    fleet_energy_throughput = 0.0

    car_126_soc = None

    for car_id in unique_ids:
        car_sessions = driving_df[(driving_df['ID'] == car_id) & (driving_df['IS_AC'] == True)]
        starts = car_sessions['STOP']
        ends = car_sessions['NEXT_START']
        start_socs = car_sessions['SOC_STOP']

        for i in range(len(car_sessions)):
            t_start = starts.iloc[i]
            t_end = ends.iloc[i]
            soc_init_pct = start_socs.iloc[i]

            idx_start = sim_df.index.searchsorted(t_start)
            idx_end = sim_df.index.searchsorted(t_end)

            if idx_start >= len(sim_df) or idx_end <= 0: continue
            idx_start = max(0, idx_start)
            idx_end = min(len(sim_df), idx_end)
            if idx_start >= idx_end: continue

            # Base Energy (No FCR)
            e_val = soc_init_pct / 100.0 * BATTERY_CAPACITY_KWH
            base_fleet_energy[idx_start:idx_end] += e_val

            # FCR Delta
            p_slice = P_per_car_profile[idx_start:idx_end]
            eff_slice = Eff_uniform[idx_start:idx_end]

            mask_charge = p_slice > 0
            mask_discharge = p_slice < 0

            efficiency_factor = np.zeros_like(eff_slice)
            efficiency_factor[mask_charge] = eff_slice[mask_charge]
            with np.errstate(divide='ignore'):
                 inv_eff = 1.0 / eff_slice[mask_discharge]
            inv_eff[np.isinf(inv_eff)] = 0.0
            efficiency_factor[mask_discharge] = inv_eff

            delta_e_kwh = p_slice * efficiency_factor * dt_h
            e_trajectory = e_val + np.cumsum(delta_e_kwh)
            e_trajectory_clipped = np.clip(e_trajectory, 0, BATTERY_CAPACITY_KWH)

            fleet_soc_energy[idx_start:idx_end] += e_trajectory_clipped
            fleet_energy_throughput += np.sum(np.abs(delta_e_kwh))

            if car_id == 126:
                if car_126_soc is None:
                     car_126_soc = np.full(len(sim_df), np.nan)
                car_126_soc[idx_start:idx_end] = e_trajectory_clipped / BATTERY_CAPACITY_KWH * 100.0

    # --- SMART STRATEGY ---
    print("Simulating Smart Strategy (Unified)...")
    n_needed = np.ceil(np.abs(P_RE_total) / P_MAX_PER_CAR)
    n_needed = np.clip(n_needed, 1, np.maximum(1, N_t))

    p_smart_per_car = np.abs(P_RE_total) / n_needed
    p_smart_per_car[P_RE_total == 0] = 0.0

    Eff_smart = np.interp(p_smart_per_car, xp, fp)

    eff_factor_smart = np.zeros_like(Eff_smart)
    mask_ch_s = P_RE_total > 0
    mask_dis_s = P_RE_total < 0

    eff_factor_smart[mask_ch_s] = Eff_smart[mask_ch_s]
    with np.errstate(divide='ignore'):
         inv_eff_s = 1.0 / Eff_smart[mask_dis_s]
    inv_eff_s[np.isinf(inv_eff_s)] = 0.0
    eff_factor_smart[mask_dis_s] = inv_eff_s

    delta_e_smart_total = P_RE_total * eff_factor_smart * dt_h

    # Smart Soc Trajectory = Base + Cumsum(Delta)
    smart_soc_trajectory = base_fleet_energy + np.cumsum(delta_e_smart_total)
    smart_energy_throughput = np.sum(np.abs(delta_e_smart_total))

    print("Simulation complete.")

    return {
        'uniform_soc_fleet': fleet_soc_energy,
        'uniform_throughput': fleet_energy_throughput,
        'smart_soc_fleet': smart_soc_trajectory,
        'smart_throughput': smart_energy_throughput,
        'car_126_soc': car_126_soc,
        'base_energy': base_fleet_energy,
        'P_bid_total': P_bid_total,
        'P_RE_total': P_RE_total
    }

def perform_economics(sim_results):
    print("Performing Economics Analysis...")

    # Constants
    FCR_PRICE_EUR_MW_H = 18.0
    BATTERY_COST_EUR_KWH = 150.0
    CYCLE_LIFE = 3000
    DT_H = FREQ_STEP_SEC / 3600.0

    # 1. Revenue
    # Based on P_bid_total (kW)
    # Revenue = Sum(P_bid_MW * Price * dt)
    P_bid_total_kw = sim_results['P_bid_total']
    revenue = np.sum((P_bid_total_kw / 1000.0) * FCR_PRICE_EUR_MW_H * DT_H)

    # 2. Aging Costs
    # Factor = Cost_per_kWh / (Cycles * 2)
    # Throughput is sum of abs(delta_E)
    aging_cost_per_kwh_throughput = BATTERY_COST_EUR_KWH / (CYCLE_LIFE * 2)

    uniform_throughput = sim_results['uniform_throughput']
    smart_throughput = sim_results['smart_throughput']

    cost_uniform = uniform_throughput * aging_cost_per_kwh_throughput
    cost_smart = smart_throughput * aging_cost_per_kwh_throughput

    profit_uniform = revenue - cost_uniform
    profit_smart = revenue - cost_smart

    print("-" * 30)
    print(f"Total Revenue: {revenue:.2f} EUR")
    print(f"Uniform Strategy: Cost = {cost_uniform:.2f} EUR, Profit = {profit_uniform:.2f} EUR")
    print(f"Smart Strategy:   Cost = {cost_smart:.2f} EUR, Profit = {profit_smart:.2f} EUR")
    print("-" * 30)

    # Write results to text file for report
    with open("results_summary.txt", "w") as f:
        f.write("Simulation Results (Jan 2021)\n")
        f.write("===============================\n")
        f.write(f"Revenue: {revenue:.2f} EUR\n")
        f.write(f"Uniform Cost: {cost_uniform:.2f} EUR\n")
        f.write(f"Uniform Profit: {profit_uniform:.2f} EUR\n")
        f.write(f"Smart Cost: {cost_smart:.2f} EUR\n")
        f.write(f"Smart Profit: {profit_smart:.2f} EUR\n")
        f.write(f"Uniform Throughput: {uniform_throughput:.2f} kWh\n")
        f.write(f"Smart Throughput: {smart_throughput:.2f} kWh\n")

    return {
        'revenue': revenue,
        'cost_uniform': cost_uniform,
        'profit_uniform': profit_uniform,
        'cost_smart': cost_smart,
        'profit_smart': profit_smart
    }

if __name__ == "__main__":
    freq_df, driving_df, obc_df, residual_df = load_and_preprocess_data()
    sim_df = prepare_simulation_time_series(freq_df)
    driving_df = process_parking_sessions(driving_df)

    availability_series = perform_initial_analysis(sim_df, driving_df)

    sim_results = perform_simulation(sim_df, driving_df, obc_df, availability_series)

    # Plot Simulation Results (Preliminary)
    plt.figure(figsize=(10, 6))
    plt.plot(sim_df.index, sim_results['uniform_soc_fleet'], label='Uniform')
    plt.plot(sim_df.index, sim_results['smart_soc_fleet'], label='Smart', linestyle='--')
    plt.plot(sim_df.index, sim_results['base_energy'], label='Base (No FCR)', alpha=0.5)
    plt.title("Aggregate Fleet Energy (kWh)")
    plt.legend()
    plt.savefig("images/sim_aggregate_energy.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(sim_df.index, sim_results['car_126_soc'])
    plt.title("Car 126 SOC (%)")
    plt.savefig("images/sim_car_126_soc.png")
    plt.close()

    # Step 4: Economics
    econ_results = perform_economics(sim_results)
