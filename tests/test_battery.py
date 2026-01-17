import unittest
import numpy as np
from src.battery_model import solve_current, calc_aging_cycling_ac, calc_aging_cycling_dc, calc_aging_calendar

class TestBatteryModel(unittest.TestCase):
    def test_solve_current_discharge(self):
        # P = 1000W, OCV = 400V, R = 0.1
        # P_term = (OCV - RI) * I
        # 1000 = (400 - 0.1I)I
        # 0.1 I^2 - 400 I + 1000 = 0
        # Roots approx I = P/OCV = 2.5A
        # Check: 400*2.5 = 1000. R*I^2 = 0.1*6.25 = 0.6W (Negligible).
        # Exact: I = (400 - sqrt(160000 - 400)) / 0.2
        # sqrt(159600) = 399.49968
        # I = 0.5003 / 0.2 = 2.5015 A
        I = solve_current(1000, 400, 0.1)
        self.assertAlmostEqual(I, 2.5015, places=3)

    def test_solve_current_charge(self):
        # P = -1000W (Charge)
        # -1000 = (400 - 0.1I)I
        # 0.1 I^2 - 400 I - 1000 = 0
        # I approx -2.5A
        # Exact: I = (400 - sqrt(160000 + 400)) / 0.2
        # sqrt(160400) = 400.49968
        # I = -0.49968 / 0.2 = -2.4984 A
        I = solve_current(-1000, 400, 0.1)
        self.assertAlmostEqual(I, -2.4984, places=3)

    def test_aging_zero(self):
        # Zero current/SOC change -> Zero aging (Cycle)
        self.assertEqual(calc_aging_cycling_ac(0, 1), 0)
        self.assertEqual(calc_aging_cycling_dc(0), 0)

    def test_aging_calendar(self):
        # Check formula structure
        # dL = 2e-4 * 10^(0.004*SOC) * 0.75 * t^(-0.25) * dt
        # SOC=50, t=100 days, dt=1 day
        res = calc_aging_calendar(50, 100, 1)
        expected = 2e-4 * (10**(0.2)) * 0.75 * (100**(-0.25)) * 1
        self.assertAlmostEqual(res, expected, places=7)

if __name__ == '__main__':
    unittest.main()
