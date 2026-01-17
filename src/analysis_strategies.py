import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from src.data_loader import load_grid_data, load_obc_data

def run_q5_q7_analysis(output_dir='images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load Data
    f_dev = load_grid_data() # Hz deviation
    obc_df = load_obc_data()

    # Grid Parameters
    y_red = 5.0 * f_dev
    P_MAX = 7.0
    FCR_GAIN = 1.1
    P_BID_CAR = P_MAX / FCR_GAIN

    # OBC Interpolation Function
    xp = obc_df['Power_kW'].values
    fp = obc_df['Efficiency'].values
    # Handle P=0 -> Efficiency=0 or constant? Usually 0 power -> 0 loss, but efficiency undefined.
    # We care about Loss. Loss = P * (1 - eta).
    # If P=0, Loss=0.
    # We will interpolate Efficiency. For P very small, efficiency drops.
    eff_func = interp1d(xp, fp, kind='linear', fill_value='extrapolate')

    # Find P_opt (Max Efficiency Power)
    p_scan = np.linspace(0.1, P_MAX, 1000) # Start from 0.1 to avoid 0
    e_scan = eff_func(p_scan)
    idx_max = np.argmax(e_scan)
    P_OPT = p_scan[idx_max]
    ETA_MAX = e_scan[idx_max]

    # --- Q5: Uniform Strategy Efficiency ---
    # Each car follows full profile scaled
    p_req_uniform = np.abs(P_BID_CAR * y_red)
    # Clip to max (just in case)
    p_req_uniform = np.clip(p_req_uniform, 0, P_MAX)

    # Calculate weighted efficiency
    # Total Energy = Sum(P * dt)
    # Total Loss = Sum(P * (1-eta) * dt)
    # Eta_avg = 1 - Total Loss / Total Energy

    # Mask where P > epsilon
    mask = p_req_uniform > 1e-3
    p_active = p_req_uniform[mask]
    eta_active = eff_func(p_active)

    total_energy_uniform = np.sum(p_active)
    total_loss_uniform = np.sum(p_active * (1 - eta_active))

    eta_uniform = 1.0 - (total_loss_uniform / total_energy_uniform)

    # --- Q6: Smart Strategy Limit ---
    # Limit N -> inf: Efficiency -> Eta_max (assuming we can always run at P_opt)
    eta_smart_inf = ETA_MAX

    # Calculate N0 for 90% benefit
    # Benefit range: Eta_uniform to Eta_smart_inf
    target_eta = eta_uniform + 0.9 * (eta_smart_inf - eta_uniform)

    n_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50]
    eta_smart_n = []

    # Pre-calculate base profile (normalized per car)
    # Total Power(N) = N * P_req_uniform (vector)
    # We assume P_total scales with N.

    # For efficiency calculation, we can simulate one time step "types"?
    # Actually, simulating the whole time series for each N is fast enough vectorized.

    p_total_base = p_req_uniform # For N=1

    for N in n_values:
        p_fleet_total = N * p_total_base

        # Smart Logic:
        # Number of cars at P_OPT
        # k_full = floor(P_fleet / P_OPT)
        # Remainder P_rem = P_fleet - k_full * P_OPT
        # Constraint: k_full + (1 if rem>0) <= N

        k_full = np.floor(p_fleet_total / P_OPT)
        # We can't use more than N cars
        k_full = np.minimum(k_full, N)

        # Power covered by full cars
        p_full = k_full * P_OPT
        p_rem = p_fleet_total - p_full

        # If P_rem > P_MAX, we have a problem (undersizing).
        # But here capacity N*P_MAX >= P_fleet (since P_fleet = N*P_bid*y_red, and P_bid*y_red < P_max).
        # However, we prioritized P_OPT.
        # If P_OPT < P_MAX, we might saturate N cars at P_OPT and still have leftovers?
        # No, P_bid*y_red < P_MAX.
        # Wait, if P_OPT is small (e.g. 3kW) and P_req is 7kW.
        # k_full = 2 (6kW). Remainder 1kW. Total 2 cars + 1 car = 3 cars?
        # But we only have 1 car (if N=1).
        # So constraint k_full must account for max power.
        # Actually, "Smart Strategy" usually implies we fill cars to P_OPT first.
        # If we run out of cars, we increase power on them?
        # Or we maximize efficiency given N cars.
        # Let's stick to the prompt Q14 logic: "Sequential activation".
        # This implies filling car 1 to P_max? Or P_opt?
        # Q6 Task 1 says "Propose strategy to reduce losses".
        # Best strategy: Run k cars at P_OPT.
        # But we are limited by N.
        # If N is large, we never hit the N limit.
        # If N is small, we just do our best.

        # Implementation for finite N:
        # Calculate Loss.
        # Loss_full = k_full * P_OPT * (1 - ETA_MAX)
        # Loss_rem = P_rem * (1 - eff_func(P_rem)) (if P_rem > 0)
        # Wait, if we are capped by N?
        # If k_full == N, then P_rem is actually handled by these N cars?
        # No, if k_full=N, implies all N cars are at P_OPT.
        # If P_fleet > N*P_OPT, we must increase power on these N cars beyond P_OPT up to P_MAX.
        # So logic:
        # 1. Try to serve P_fleet using integer number of P_OPT cars.
        # 2. If P_fleet > N * P_OPT:
        #    All N cars must run > P_OPT.
        #    Average power = P_fleet / N.
        #    Loss = N * P_avg * (1 - eta(P_avg)).
        # 3. If P_fleet <= N * P_OPT:
        #    We use k = floor(P_fleet/P_OPT) cars at P_OPT.
        #    And 1 car at remainder.
        #    Loss = k*P_OPT*(1-ETA_MAX) + P_rem*(1-eta(P_rem)).

        mask_saturated = p_fleet_total > (N * P_OPT)

        loss_total = np.zeros_like(p_fleet_total)

        # Case 1: Saturated (High power demand compared to optimal capacity)
        # Distribute equally among N (since going above P_OPT, curve usually drops or flattens, better to share?)
        # Actually if curve is flat > P_opt, sharing is fine.
        # If curve drops, we should keep some at P_opt and push others?
        # Usually OBC efficiency is flat-ish after 20-30%.
        # Let's assume uniform distribution when > N*P_OPT is simplest and efficient enough.
        p_avg_sat = p_fleet_total[mask_saturated] / N
        loss_total[mask_saturated] = N * p_avg_sat * (1 - eff_func(p_avg_sat))

        # Case 2: Under-saturated (can optimize)
        # Use k_full cars at P_OPT, 1 at P_rem
        mask_under = ~mask_saturated
        p_under = p_fleet_total[mask_under]

        k_full_u = np.floor(p_under / P_OPT) # This will be < N
        p_rem_u = p_under - k_full_u * P_OPT

        loss_u = k_full_u * P_OPT * (1 - ETA_MAX)
        loss_u += p_rem_u * (1 - eff_func(p_rem_u))
        loss_total[mask_under] = loss_u

        eta_n = 1.0 - (np.sum(loss_total) / np.sum(p_fleet_total))
        eta_smart_n.append(eta_n)

    n0_q6 = next((n for n, e in zip(n_values, eta_smart_n) if e >= target_eta), None)

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, eta_smart_n, marker='o', label='Smart Efficiency')
    plt.axhline(eta_uniform, color='r', linestyle='--', label='Uniform Baseline')
    plt.axhline(eta_smart_inf, color='g', linestyle='--', label='Theoretical Limit')
    plt.axhline(target_eta, color='orange', linestyle=':', label='90% Target')
    plt.xscale('log')
    plt.xlabel('Fleet Size (N)')
    plt.ylabel('Efficiency')
    plt.title('Q6: Smart Strategy Efficiency Convergence')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.savefig(os.path.join(output_dir, "q6_efficiency_convergence.png"))
    plt.close()

    # --- Q7: OBC Operating Time ---
    # t_op_uniform = Fraction of time |y_red| > 0.
    t_op_uniform = np.mean(np.abs(y_red) > 1e-4) # Almost 1.0

    # t_op_smart_limit
    # Formula derived: u = (P_BID / P_OPT) * E[|y_red|]
    # (Assuming P_OPT is the target operating point)
    # If we use P_MAX, denominator is P_MAX.
    # To minimize time, we should run at P_MAX?
    # Requirement: "Smart strategy can also reduce OBC average running time".
    # Running at P_MAX minimizes time (Energy = Power * Time).
    # But Q6 said "Smart Strategy" (Efficiency).
    # Does "Smart Strategy" optimize Efficiency OR Time?
    # Usually Efficiency. P_OPT.
    # But User Q7 says: "Smart strategy can ALSO reduce...".
    # This implies we use the SAME strategy as Q6 (Efficiency optimized).
    # So we run at P_OPT.
    # Therefore, Limit is determined by P_OPT.

    # E[|y_red|]
    E_y_abs = np.mean(np.abs(y_red))
    t_op_smart_inf = (P_BID_CAR / P_OPT) * E_y_abs

    # Numerical check for finite N
    # Average Run Time = Sum(Active Cars per step) / (N * Total Steps)
    # = Mean(Active Cars) / N

    t_op_n = []

    for N in n_values:
        p_fleet_total = N * p_total_base

        # Logic matches Q6: k_full at P_OPT, 1 at P_rem.
        # Active cars = k_full + (1 if P_rem > 0 else 0)
        # Note: If saturated, Active = N.

        mask_saturated = p_fleet_total > (N * P_OPT)
        active_sat = np.full(np.sum(mask_saturated), N)

        mask_under = ~mask_saturated
        p_under = p_fleet_total[mask_under]
        k_full_u = np.floor(p_under / P_OPT)
        p_rem_u = p_under - k_full_u * P_OPT
        active_under = k_full_u + (p_rem_u > 1e-3).astype(int)

        total_active_steps = np.sum(active_sat) + np.sum(active_under)
        avg_t_op = total_active_steps / (len(p_fleet_total) * N)
        t_op_n.append(avg_t_op)

    # Target for Q7 (90% reduction benefit)
    # Reduction = t_uniform - t_smart
    # Target T = t_uniform - 0.9 * (t_uniform - t_smart_inf)

    target_t_op = t_op_uniform - 0.9 * (t_op_uniform - t_op_smart_inf)

    n0_q7 = next((n for n, t in zip(n_values, t_op_n) if t <= target_t_op), None)

    return {
        'q5_eta_uniform': eta_uniform,
        'q6_eta_limit': eta_smart_inf,
        'q6_n0': n0_q6,
        'q6_p_opt': P_OPT,
        'q7_t_op_uniform': t_op_uniform,
        'q7_t_op_limit': t_op_smart_inf,
        'q7_n0': n0_q7
    }

if __name__ == "__main__":
    print(run_q5_q7_analysis())
