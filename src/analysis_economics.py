import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.data_loader import load_grid_data, load_residual_value_data, load_driving_data

def run_economics(output_dir='images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load Data
    f_dev = load_grid_data()
    avail_1h = pd.read_csv('data/availability_1h.csv', index_col=0, parse_dates=True)
    avail_4h = pd.read_csv('data/availability_4h.csv', index_col=0, parse_dates=True)
    res_df = load_residual_value_data()

    # Load Driving Data to get Total Fleet Size
    driving_df = load_driving_data()
    total_evs = driving_df['ID'].nunique()

    # Parameters
    FCR_PRICE = 18.0 # EUR/MW/h
    P_MAX = 7.0
    FCR_GAIN = 1.1
    P_BID_CAR = P_MAX / FCR_GAIN # kW
    C_CONS = 0.15 # kWh/km (Assumed)

    # Pre-process Residual Value Interpolator
    # Check column names
    # "Mileage (km)", "Residual value (\x80)" (encoding issue might happen, let's use iloc)
    km_vals = res_df.iloc[:, 0].values
    eur_vals = res_df.iloc[:, 1].values

    # Fit polynomial or interp?
    # Data looks linear-ish? 18645.2 -> 18644.8.
    # Diff is -0.4 EUR per 5000 km? That's tiny.
    # 5000km -> 18645.
    # 10000km -> 18644.
    # Loss = 0.4 EUR for 5000km? That's 0.00008 EUR/km.
    # This seems extremely low.
    # Let me check the file content again or assume I read it wrong.
    # Maybe the separator is ';' and I need to handle it.

    # Re-reading checking sample
    # 5000;18645.2
    # 10000;18644.8
    # It decreases by 0.4.
    # Wait. Maybe it's 18,645.2?
    # Let's check 75000 value later.
    # If the slope is shallow, the loss is negligible.
    # I'll proceed with interpolation.
    res_func = lambda k: np.interp(k, km_vals, eur_vals)

    # --- Q11: Revenue ---
    # We need to match Frequency Data (May 2019 / Jan 2021 Tiled) with Availability (Jan 2021).
    # We assume the availability index matches the simulation period.
    # avail_1h is full year?
    # We need to extract the month corresponding to grid data.
    # Grid data `f_dev` is May 2019.
    # But usually we map it to "Month 1" of simulation.
    # Let's assume we simulate Jan 2021 using May 2019 Frequency Profile.

    # Create simulation index for 1 month
    sim_start = '2021-01-01'
    sim_end = '2021-01-31 23:59:59'

    # Reindex availability to this month
    # Avail is 1h resolution. Freq is irregular (loaded as series).
    # Freq length ~ 1 month.
    # We align by length?

    # Total hours in May is 31*24 = 744.
    # We need sum of N_avail for each hour.

    # Get subset of availability for Jan 2021
    avail_1h_jan = avail_1h.loc[sim_start:sim_end]
    avail_4h_jan = avail_4h.loc[sim_start:sim_end]

    # Helper to calc revenue
    def calc_monthly_revenue(avail_series):
        # Revenue = Sum(N_avail_i * P_bid_car_MW * Price * 1h)
        # Avail series is hourly or 4-hourly.
        # We can just sum (N * dt_hours).
        # Be careful with 4h series: each point represents 4 hours?
        # If sampled at 4h freq, yes.
        # Check resampling: `resample('4h').min()` creates timestamps 00, 04, 08...
        # So each value applies for 4 hours.

        # Check frequency
        freq_str = pd.infer_freq(avail_series.index)
        if freq_str == 'h' or 'h' in str(freq_str).lower():
            dt = 1.0
        elif '4h' in str(freq_str).lower():
            dt = 4.0
        else:
            # Fallback
            diff = (avail_series.index[1] - avail_series.index[0]).total_seconds() / 3600.0
            dt = diff

        revenue = np.sum(avail_series.values * (P_BID_CAR / 1000.0) * FCR_PRICE * dt)
        return revenue

    rev_1h = calc_monthly_revenue(avail_1h_jan)
    rev_4h = calc_monthly_revenue(avail_4h_jan)

    # Per EV
    rev_1h_per_ev = rev_1h / total_evs
    rev_4h_per_ev = rev_4h / total_evs

    # --- Q12: Virtual Mileage ---
    # M_virt = (Integral |P(t)| dt) / C_cons
    # P(t) = P_bid * y_red(t)
    # y_red = 5 * f_dev

    y_red = 5.0 * f_dev
    # Integral |y_red| dt (in hours)
    # Grid data is 1 month.
    dt_grid_hours = (f_dev.index[1] - f_dev.index[0]).total_seconds() / 3600.0

    integral_y_abs = np.sum(np.abs(y_red)) * dt_grid_hours

    # Throughput per *Active* car (Capacity) = P_BID_CAR * Integral(|y|)
    throughput_per_bid_kw = P_BID_CAR * integral_y_abs

    # But a car is only active when it is available.
    # This is tricky.
    # Uniform Strategy: All Available cars share the load?
    # No, capacity is Sum(N). Load is Sum(N * P_bid * y).
    # So every available car delivers P_bid * y.
    # So if a car is available for fraction 'u' of the month, it does u * Throughput_Full.

    # Average Availability per car
    # u_avg = (Sum N_avail) / (Total_EVs * Total_Hours)
    # Total Hours in Jan = 744.
    hours_in_month = 31 * 24
    avg_avail_fraction = avail_1h_jan.mean().item() / total_evs # avail_1h_jan.mean() is average N
    # Actually simpler: Average N connected = X. Total fleet = Y.
    # Average duty cycle = X/Y.

    avg_virtual_mileage = (throughput_per_bid_kw * avg_avail_fraction) / C_CONS

    # --- Q13: Residual Value Loss ---
    # Base Mileages
    base_km = np.arange(5000, 80000, 5000)

    losses = []
    net_revenues = []

    # We use 1H revenue as baseline? Or 4H?
    # Usually 4H is more realistic (market blocks).
    # Let's use 4H Revenue per EV.
    base_revenue = rev_4h_per_ev

    for km in base_km:
        val_base = res_func(km)
        val_new = res_func(km + avg_virtual_mileage)
        loss = val_base - val_new
        losses.append(loss)
        net_revenues.append(base_revenue - loss)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(base_km, losses, marker='o', label='Resale Value Loss')
    plt.plot(base_km, net_revenues, marker='s', label='Net FCR Revenue (4h)')
    plt.axhline(base_revenue, color='g', linestyle='--', label='Gross Revenue')
    plt.xlabel("Annual Mileage (km)")
    plt.ylabel("EUR / EV / Month") # Wait, Loss is Lump Sum or Rate?
    # Resale Value is total asset value.
    # Loss is "Instantaneous drop in value due to extra km"?
    # If we add 1 month of virtual mileage, we lose some value.
    # Is the curve "Value vs Total Mileage"? Yes.
    # So delta V is correct.
    plt.title("Q13: Economic Impact of Virtual Mileage (1 Month)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "q13_economics.png"))
    plt.close()

    return {
        'q11_rev_1h': rev_1h_per_ev,
        'q11_rev_4h': rev_4h_per_ev,
        'q12_virt_km': avg_virtual_mileage,
        'q13_avg_loss_5k': losses[0],
        'q13_net_5k': net_revenues[0]
    }

if __name__ == "__main__":
    print(run_economics())
