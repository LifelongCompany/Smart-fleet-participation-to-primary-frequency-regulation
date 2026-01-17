import numpy as np

def solve_current(power_watts, ocv_volts, r_ohms):
    """
    Solves for current I (Amps) given Power P (Watts).
    Model: P = (OCV - R*I) * I
    Convention: P > 0 is Discharge, P < 0 is Charge.

    Returns: Current I (Amps).
             Returns NaN if power is too high (complex root).
    """
    # R*I^2 - OCV*I + P = 0
    # Discriminant delta = OCV^2 - 4*R*P
    delta = ocv_volts**2 - 4 * r_ohms * power_watts

    if np.isscalar(delta):
        if delta < 0:
            return np.nan
        sqrt_delta = np.sqrt(delta)
    else:
        # Check for validity (can't deliver more power than OCV^2/4R)
        valid_mask = delta >= 0
        sqrt_delta = np.full_like(delta, np.nan, dtype=float)
        sqrt_delta[valid_mask] = np.sqrt(delta[valid_mask])

    # We choose the root that gives I=0 when P=0.
    # I = (OCV - sqrt(delta)) / 2R
    i_amps = (ocv_volts - sqrt_delta) / (2 * r_ohms)

    return i_amps

def calc_aging_cycling_ac(current_amps, dt_hours):
    """
    Q16: Cycling Aging for FCR/AC Charging (Mode 3).
    Formula: dL = 2e-7 * |I| * 10^(0.003 * |I|) * dt
    dt in hours.
    """
    i_abs = np.abs(current_amps)
    # Avoid overflow if I is huge (shouldn't be for 7kW/350V ~ 20A)
    # 0.003 * 20 = 0.06. 10^0.06 is small. Safe.

    aging = 2e-7 * i_abs * (10**(0.003 * i_abs)) * dt_hours
    return aging

def calc_aging_cycling_dc(delta_soc_percent):
    """
    Q16: Cycling Aging for Driving/DC Charging (Mode 4).
    Formula: dL = 5e-5 * |delta_soc|
    """
    aging = 5e-5 * np.abs(delta_soc_percent)
    return aging

def calc_aging_calendar(soc_percent, t_age_days, dt_days):
    """
    Q16: Calendar Aging.
    Formula: dL = 2e-4 * 10^(0.004 * SOC) * alpha * t_age^(alpha-1) * dt
    alpha = 0.75
    """
    alpha = 0.75
    # Avoid division by zero if t_age_days is 0 (start of life)
    # Usually t_age is at least small positive.
    t_safe = np.maximum(t_age_days, 1e-9)

    term_soc = 10**(0.004 * soc_percent)
    term_time = alpha * (t_safe**(alpha - 1))

    aging = 2e-4 * term_soc * term_time * dt_days
    return aging
