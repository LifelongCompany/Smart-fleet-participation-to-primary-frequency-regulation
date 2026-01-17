import unittest
from src.simulation_core import SimulationCore
import numpy as np

class TestSimulationCore(unittest.TestCase):
    def test_run_short(self):
        # Run for 1 hour
        sim_end = '2021-01-01 01:00:00'
        sim = SimulationCore(output_dir='tests/output', sim_end=sim_end)

        # Scenario Uniform 1h
        soc, cyc, cal = sim.run_scenario('uniform', '1h')

        # Check output shapes
        # 1 hour = 360 steps (+1 for end point?)
        self.assertGreater(len(soc), 300)
        self.assertEqual(soc.shape[1], sim.n_cars)
        self.assertGreater(sim.n_cars, 100)

        # Check aging is positive
        self.assertGreater(cyc + cal, 0)

        # Check SOC constraints (within 0-100)
        self.assertTrue(np.all(soc >= 0.0))
        self.assertTrue(np.all(soc <= 100.0))

        # Run Smart
        soc_s, cyc_s, cal_s = sim.run_scenario('smart', '1h')
        self.assertGreater(cyc_s + cal_s, 0)

if __name__ == '__main__':
    unittest.main()
