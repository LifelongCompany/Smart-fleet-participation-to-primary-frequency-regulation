# Smart Fleet Participation in FCR - Project Report

## 1. Project Overview
This project simulates the participation of an Electric Vehicle (EV) fleet in the Frequency Containment Reserve (FCR) market. The goal is to evaluate the technical feasibility and economic viability of using Vehicle-to-Grid (V2G) technology for grid ancillary services.

The simulation covers **January 2021** with a time resolution of **10 seconds**, utilizing real driving session data and frequency deviation measurements.

## 2. Methodology

### 2.1 Data Sources
- **Driving Data:** `driving_sessions.csv` containing trip logs for 151 unique EVs.
- **Frequency Data:** `france_2019_05.csv` (tiled to cover Jan 2021). Values converted from mHz to Hz.
- **OBC Efficiency:** `obc_efficiency.csv`, modeling non-linear charger efficiency.
- **Battery Parameters:** Assumed 40 kWh capacity per EV.
- **Economic Parameters:**
  - FCR Capacity Payment: **18 EUR/MW/h**
  - Battery Cost: **150 EUR/kWh**
  - Cycle Life: **3000 cycles**

### 2.2 Simulation Logic
- **Time Domain:** January 1, 2021 to January 31, 2021 (10s steps).
- **Fleet Availability:** Cars are available for V2G only when parked and connected to an AC charger (7kW).
  - **AC/DC Logic:** If the energy required for the *next trip* can be replenished within the parking duration using 7kW AC, the car uses AC (V2G enabled). Otherwise, it uses DC fast charging (V2G disabled).
- **Power Calculation:**
  - Bid Capacity ($P_{bid}$): Sum of available cars $\times (P_{max} / 1.1)$.
  - Regulation Power ($P_{RE}$): $5 \times P_{bid} \times (f - 50)$.
- **Strategies Evaluated:**
  1.  **Uniform Strategy:** $P_{RE}$ is distributed evenly among all available cars. Efficiency is calculated per car based on its partial load.
  2.  **Smart Strategy (Unified):** The fleet is modeled as a unified battery. The dispatcher concentrates power requests to minimize the number of active chargers, operating them closer to peak efficiency (7kW).

## 3. Analysis & Results

### 3.1 Initial Analysis
- **Frequency Distribution:** The frequency deviation is centered around 0 Hz, typical of a stable grid.
- **Fleet Availability:** The number of connected EVs varies throughout the day, influencing the FCR bid capacity.
- **Theoretical Drift:** The integral of frequency deviation shows the theoretical energy drift of a battery performing FCR without correction.

![Fleet Availability](images/fleet_availability.png)
*Figure 1: Fleet Availability (Number of AC-connected EVs)*

### 3.2 Simulation Performance
The simulation was executed using vectorized operations (NumPy/Pandas) to ensure performance (runtime < 30s for 2.6M data points).

- **Uniform Strategy:** Distributed low-power operations resulted in lower charger efficiency (avg ~90-93%).
- **Smart Strategy:** Concentrating power allowed operations near peak efficiency (~97%).

![Aggregate Energy](images/sim_aggregate_energy.png)
*Figure 2: Aggregate Fleet Energy Evolution (Uniform vs Smart vs Base)*

### 3.3 Economic Assessment

The economic analysis compares the revenue generated from FCR capacity payments against the "Virtual Mileage" cost (battery degradation due to cycling).

**Parameters:**
- Aging Cost Factor: ~0.025 EUR/kWh of throughput.

**Results (January 2021):**

| Metric | Uniform Strategy | Smart Strategy |
| :--- | :--- | :--- |
| **Total Revenue** | **11,118.55 EUR** | **11,118.55 EUR** |
| Energy Throughput | 51,728 kWh | 48,738 kWh |
| Aging Cost | 1,293.21 EUR | 1,218.45 EUR |
| **Net Profit** | **9,825.35 EUR** | **9,900.10 EUR** |

### 4. Key Findings
1.  **Profitability:** FCR participation is highly profitable for this fleet. Revenue exceeds aging costs by a factor of ~8-9.
2.  **Smart Strategy Benefits:** By optimizing for charger efficiency, the Smart Strategy reduces energy throughput losses (and thus aging costs) by approximately **6%**, leading to higher net profit.
3.  **Feasibility:** The fleet maintains sufficient SOC levels (as seen in the aggregate plots) to perform the service while meeting driving requirements.

## 5. Artifacts
- Source Code: `fcr_simulation.py`
- Images: `images/` directory
