import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.data_loader import load_driving_data

def process_behavior(output_dir='images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load Data
    driving_df = load_driving_data()

    # Sort
    driving_df = driving_df.sort_values(by=['ID', 'START'])

    # Calculate Parking Durations and Next Trip Energy
    # Shift -1 to get "Next Trip" details relative to current "Stop" (Parking Start)
    # The 'STOP' of row i is the start of parking. 'START' of row i+1 is end of parking.

    # We need to act on the *Parking Session*.
    # A row in driving_df represents a Drive.
    # Parking happens *after* the Drive (between STOP of row i and START of row i+1).

    driving_df['NEXT_START'] = driving_df.groupby('ID')['START'].shift(-1)
    driving_df['NEXT_SOC_START'] = driving_df.groupby('ID')['SOC_START'].shift(-1)

    # SOC is in %. Capacity 40kWh.
    # Trip Energy = (SOC_Start - SOC_Stop) * 40 / 100 ?
    # Actually, the *Next* Trip Energy determines if we can charge slowly.
    # Energy needed = Consumption of next trip.
    # Next trip starts at NEXT_START. Ends at NEXT_STOP.
    # Energy = (NEXT_SOC_START - NEXT_SOC_STOP) ? No.
    # Energy *consumed* during next trip is what needs to be replenished?
    # No, usually we need to replenish what was used *before*?
    # Or we need to reach a target SOC?
    # Q8 Logic: "If parking time is enough to fill up using 7kW, then AC."
    # Fill up to what? 100%? Or just enough for next trip?
    # "Usually logic: If parking time enough to use 7kW to full, then AC." -> "Or 7kW enough to charge?"
    # Let's read Q8 carefully: "(General logic: If parking time enough to full using 7kW -> AC)".
    # "To full" implies (100 - Current_SOC) * Capacity.
    # Current SOC is `SOC_STOP` of the drive that just ended.

    driving_df['PARKING_DURATION_H'] = (driving_df['NEXT_START'] - driving_df['STOP']).dt.total_seconds() / 3600.0

    BATTERY_CAP = 40.0
    AC_POWER = 7.0

    # Energy required to reach 100%
    driving_df['E_REQ_FULL'] = (100.0 - driving_df['SOC_STOP']) / 100.0 * BATTERY_CAP

    def check_ac(row):
        if pd.isna(row['PARKING_DURATION_H']):
            return False # End of data, assume no V2G
        if row['PARKING_DURATION_H'] <= 0:
            return False

        # Check if 7kW can fill it
        time_needed = row['E_REQ_FULL'] / AC_POWER

        # If time_needed <= parking_duration, we use AC (Slow).
        # Else, we use DC (Fast).
        return time_needed <= row['PARKING_DURATION_H']

    driving_df['IS_AC'] = driving_df.apply(check_ac, axis=1)

    # --- Q9: Coincidence Factors ---
    # Construct time series of AC availability
    # We need to span the full year 2021 based on data?
    # Or just the simulation month (Jan)?
    # Q10 asks for Start of Year vs End of Year.
    # So we should generate the profile for the whole dataset duration.

    min_time = driving_df['START'].min()
    max_time = driving_df['NEXT_START'].max()

    # Create an event-based calculation for speed
    ac_sessions = driving_df[driving_df['IS_AC'] == True].copy()
    ac_sessions = ac_sessions.dropna(subset=['STOP', 'NEXT_START'])

    events_start = pd.DataFrame({'time': ac_sessions['STOP'], 'change': 1})
    events_end = pd.DataFrame({'time': ac_sessions['NEXT_START'], 'change': -1})
    events = pd.concat([events_start, events_end]).sort_values('time')

    # Cumulative sum
    events['N_avail'] = events['change'].cumsum()

    # Resample to 1 minute or 15 mins to handle blocks
    # Event series is irregular. We need regular grid.
    # Let's resample to 1 min.

    # Ideally, we want "Minimum in 1h block".
    # Blocks: 00:00-01:00, 01:00-02:00...
    # We can use resampling '1h', apply 'min'.
    # But first we need a regular series.

    full_idx = pd.date_range(start=min_time.floor('h'), end=max_time.ceil('h'), freq='1min')

    # Forward fill availability
    # We use `asof` logic or reindex+ffill
    # Remove duplicates in events time to avoid reindex error
    events_unique = events.drop_duplicates('time', keep='last').set_index('time')

    avail_series = events_unique['N_avail'].reindex(full_idx, method='ffill').fillna(0)

    # 1-Hour Blocks (Min)
    # Resample 1h, min
    avail_1h = avail_series.resample('1h').min()

    # 4-Hour Blocks (Min)
    # Resample 4h, min (offset? 00:00-04:00)
    avail_4h = avail_series.resample('4h').min()

    # Plotting Q9
    # Just plot a representative week or month? Or full year?
    # Q9: "Plot curve".
    # Let's plot the first week of Jan and maybe full year statistics.

    plt.figure(figsize=(12, 6))
    subset_1h = avail_1h['2021-01-01':'2021-01-07']
    subset_4h = avail_4h['2021-01-01':'2021-01-07']

    # Need to plot 4h as steps
    plt.step(subset_1h.index, subset_1h.values, where='post', label='1h Block Min')
    plt.step(subset_4h.index, subset_4h.values, where='post', label='4h Block Min', linewidth=2)
    plt.title("Coincidence Factor (Availability) - First Week Jan 2021")
    plt.ylabel("Number of AC EVs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "q9_coincidence_factor_week.png"))
    plt.close()

    # --- Q10: Start vs End Analysis ---
    # Compare Jan 1-4 with Dec 28-31?
    # Or simply average profile start vs end.

    start_val = avail_1h['2021-01-01':'2021-01-02'].mean()
    end_val = avail_1h['2021-12-30':'2021-12-31'].mean()

    # Check for "loop" issues.
    # If the driving data doesn't loop perfectly, availability might drop at edges or mismatch.
    # Driving sessions usually just end.

    # Return 1h and 4h series for Simulation usage
    # We need to return them so `simulation_core` can use them.
    # We'll save them as CSV or return object.
    # Since we are separating processes, saving to disk is safer.

    avail_1h.to_csv('data/availability_1h.csv')
    avail_4h.to_csv('data/availability_4h.csv')

    return {
        'start_year_avg': start_val,
        'end_year_avg': end_val,
        'q10_comment': "Comparison showing potential boundary effects or seasonal drift."
    }

if __name__ == "__main__":
    print(process_behavior())
