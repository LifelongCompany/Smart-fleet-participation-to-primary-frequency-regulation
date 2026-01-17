import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.data_loader import load_grid_data

def run_q1_q4_analysis(output_dir='images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load Data (Hz deviation)
    f_dev = load_grid_data()

    # --- Q1: Reduced Regulating Power ---
    # y_red = 5 * (f - 50) = 5 * f_dev
    y_red = 5.0 * f_dev

    plt.figure(figsize=(10, 6))
    y_red.hist(bins=100, density=True, alpha=0.7, color='blue')
    plt.title("Distribution of Reduced Regulating Power (y_red)")
    plt.xlabel("Normalized Power (p.u.)")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "q1_reduced_power_dist.png"))
    plt.close()

    # --- Q2: Magnitude Stats ---
    # Calculate stats for the report
    stats = {
        'max_pos': y_red.max(),
        'max_neg': y_red.min(),
        'mean': y_red.mean(),
        'std': y_red.std(),
        'percent_full_power': (y_red.abs() >= 1.0).mean() * 100
    }

    # --- Q3: Rolling SOC Deviation ---
    # Parameters
    P_MAX = 7.0 # kW
    FCR_GAIN = 1.1
    P_BID = P_MAX / FCR_GAIN
    BATTERY_CAP = 40.0 # kWh

    # Power profile (kW) = P_BID * y_red
    # Note: y_red is p.u. (1.0 = full activation)
    # y_red > 0 usually means frequency is high -> Charge?
    # Wait, check physics.
    # High Freq -> Too much Gen -> Load should Increase (Charge).
    # y_red = 5 * (f - 50). If f > 50, y_red > 0.
    # So y_red > 0 implies Charging (Power into battery).
    # Power (kW) = P_BID * y_red.
    # Energy (kWh) = Integral P dt.
    # If P > 0 is charge, SOC increases.

    # Time step
    # The index is datetime. calculate delta t.
    # data is 10s usually?
    dt_seconds = (f_dev.index[1] - f_dev.index[0]).total_seconds()
    dt_hours = dt_seconds / 3600.0

    p_profile_kw = P_BID * y_red
    e_step_kwh = p_profile_kw * dt_hours # Positive = Charge = SOC Increase

    # Rolling windows
    windows_hours = [4, 8, 12, 24]
    results_q3 = {}

    plot_data = []
    plot_labels = []

    for wh in windows_hours:
        steps = int(wh / dt_hours)
        # Sum of energy steps over the window
        # We want the *deviation* from start of window to end of window.
        # Ideally, we want the range of SOC values?
        # Question Q3 says: "Plot the SOC deviation induced by FCR"
        # "Time window: Rolling 4h, 8h..."
        # Usually means: For every time t, what is E(t+window) - E(t)?
        # i.e., net energy change over that window.
        rolling_net_energy = e_step_kwh.rolling(window=steps).sum()

        # Drop NaNs
        rolling_valid = rolling_net_energy.dropna()

        plot_data.append(rolling_valid.values)
        plot_labels.append(f"{wh}h")

        # Stats for Q4
        results_q3[f'max_dev_{wh}h'] = rolling_valid.abs().max()
        results_q3[f'99_percentile_{wh}h'] = np.percentile(rolling_valid.abs(), 99)

    plt.figure(figsize=(10, 6))
    plt.boxplot(plot_data, tick_labels=plot_labels)
    plt.title("FCR-Induced Energy Deviation over Rolling Windows")
    plt.ylabel("Energy Change (kWh)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "q3_rolling_soc_deviation.png"))
    plt.close()

    # Return combined results
    return {
        'q2_stats': stats,
        'q3_stats': results_q3,
        'parameters': {
            'p_bid': P_BID,
            'battery_cap': BATTERY_CAP
        }
    }

if __name__ == "__main__":
    # Test run
    print(run_q1_q4_analysis())
