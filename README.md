# FCR Participation Report

## Part 1: Grid Frequency Data & Basic Analysis

**Q1:**
- **Data Source:** `data/france_2019_05.csv` (Frequency data).
- **Distribution of Regulating Power:**
  The normalized regulating power $P_{reg}^{p.u.}$ is calculated as $5 \times (f - 50)$.
  ![Q1 Distribution](images/q1_distribution.png)

**Q2:**
- **Observations:**
  The distribution is centered around 0 but exhibits a spread corresponding to frequency deviations. The magnitude rarely reaches full power ($\pm 1$ p.u.), staying mostly within $\pm 0.4$ p.u. The distribution is symmetric, indicating balanced regulation requirements over the month.

**Q3:**
- **Single EV SOC Deviation:**
  The SOC deviation for a single EV providing FCR (without energy management) is shown below for various rolling windows.
  ![Q3 SOC Deviation](images/q3_soc_deviation.png)

**Q4:**
- **Reasonability:**
  For short windows (4h), the SOC deviation is relatively small (< 10%). However, for 24h windows, the deviation can grow significantly. Without active energy management (recharging), continuous FCR participation for 24h+ carries a risk of depleting the battery or reaching full charge, confirming the need for a management strategy or limited participation windows.

---

## Part 2: Smart Dispatch Strategy

**Q5:**
- **Uniform Strategy Efficiency:**
  Under the Uniform Strategy, the average fleet efficiency is calculated to be **85.91%**. This is limited because many vehicles operate at low partial loads where the OBC efficiency is poor.

**Q6:**
- **Smart Strategy:**
  - **Proposal:** A "Smart" strategy concentrates the total power request onto a subset of vehicles, operating them closer to their optimal efficiency point ($P_{opt}$), while keeping others idle.
  - **Limit ($N \to \infty$):** The efficiency approaches $\eta_{max}$ (approx 94-95%).
  - **Convergence:** The efficiency improves rapidly with fleet size.
  ![Q6 Efficiency](images/q6_efficiency.png)
  - **Minimum Vehicles ($N_0$):** To achieve 90% of the potential gain, a fleet size of approximately **10-20 vehicles** is sufficient.

**Q7:**
- **OBC Operating Time:**
  - **Derivation:** The fraction of time the OBC is active, $t_{op}$, depends on the probability that a vehicle is needed. For a large fleet using a Smart (Min-Time) strategy, the fleet-wide operating ratio converges to:
    $$t_{op}^{\infty} = \frac{P_{bid}}{P_{max}} \mathbb{E}[|P_{reg}^{p.u.}|]$$
  - **Limit Value:** Calculated as **0.0669 p.u.** (approx 6.7% of time).
  - **Minimum Vehicles ($N_0$):**
  ![Q7 Operating Time](images/q7_operating_time.png)

---

## Part 3: Driving & Charging Behaviour

**Q8:**
- **Inference:** Charging sessions were inferred from `data/driving_sessions.csv`. Trips ending with enough parking time to charge via 7kW AC were labeled as AC (V2G available).

**Q9:**
- **Coincidence Factor:**
  The number of available AC-connected EVs varies over time.
  ![Q9 Availability](images/q9_availability.png)

**Q10:**
- **Limitations:**
  Inferring charging from driving has limitations, especially at the year boundaries. If the data is not cyclic, cars driving at the end of the year disappear, and cars at the start appear from nowhere. This leads to inaccurate availability estimates at the start/end of the simulation period (as seen in the availability drop/rise).

---

## Part 4: FCR Revenues

**Q11:**
- **Monthly Revenue (Estimated):**
  - **1-hour Blocks:** 7653.42 EUR (Total) $\to$ **51.02 EUR/EV**
  - **4-hour Blocks:** 7330.78 EUR (Total) $\to$ **48.87 EUR/EV**
  - *Note: 4-hour blocks have slightly lower revenue due to the stricter capacity constraint (minimum of 4 hours).*

**Q12:**
- **Virtual Mileage:**
  - The average virtual mileage per vehicle is approximately **1257 km** per month.
  - *Calculation based on energy throughput and fleet average consumption of 0.153 kWh/km.*

**Q13:**
- **Residual Value Loss:**
  - The estimated loss in residual value due to this virtual mileage is **11.09 EUR** per month.
  - **Net Revenue:** $51.02 - 11.09 \approx 39.93$ EUR/EV/month. FCR remains profitable.

---

## Part 5: Simulation Framework

**Q14:**
- **Full Simulation:**
  Simulations were run for Uniform and Smart strategies under 1h and 4h block constraints.
  - **SOC Profile (Example - Car 0, Smart 1h):**
    ![SOC Smart 1h](images/q14_soc_smart_1h.png)
  - **SOC Profile (Example - Car 0, Uniform 1h):**
    ![SOC Uniform 1h](images/q14_soc_uniform_1h.png)

---

## Part 6: Battery Aging

**Q15:**
- **Battery Model:**
  Current $I$ is derived from Power $P$, OCV $V_{oc}$, and Resistance $R$:
  $$I = \frac{V_{oc} - \sqrt{V_{oc}^2 - 4 R P_{term}}}{2 R}$$
  *(Choosing the root consistent with $I=0$ when $P=0$)*.

**Q16:**
- **Aging Evaluation:**
  Comparing 1-month equivalent aging against a baseline (No FCR, only Driving/Charging):
  - **Baseline Aging Score:** 10.72 (Arbitrary Units)
  - **Uniform Strategy:** +2.80% increase.
  - **Smart Strategy:** +7.47% increase.

  **Conclusion:** The Smart Strategy, while efficient for energy (grid side) and OBC life (time side), tends to stress the battery more (+7.5% aging vs +2.8% for Uniform) because it concentrates high power cycling on fewer vehicles at a time, and the cycling aging factor is convex with respect to current ($10^{0.003|I|}$).
