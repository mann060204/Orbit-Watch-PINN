"""Synthetic indexing fixtures only; no scientific benchmark is run here."""
import unittest

import numpy as np

from run_three_models import split_metrics


class ThreeModelSplitTests(unittest.TestCase):
    def setUp(self):
        self.reference_r = np.tile([7000.0, 0, 0], (181, 1))
        self.reference_v = np.tile([0, 7.5, 0], (181, 1))

    def test_splits_score_exactly_1_to_90_91_to_120_and_121_to_180(self):
        predicted_r, predicted_v = self.reference_r.copy(), self.reference_v.copy()
        indices = np.arange(181, dtype=float)
        predicted_r[:, 0] += indices
        predicted_v[:, 1] += indices / 1000
        metrics = split_metrics({"r": predicted_r, "v": predicted_v},
                                self.reference_r, self.reference_v)
        for name, first, last in [("train", 1, 90), ("validation", 91, 120), ("test", 121, 180)]:
            with self.subTest(split=name):
                scored = np.arange(first, last + 1, dtype=float)
                result = metrics[name]
                self.assertEqual(result["future_samples"], len(scored))
                self.assertAlmostEqual(result["position_vector_mae_km"], float(scored.mean()))
                self.assertAlmostEqual(result["position_vector_rmse_km"], float(np.sqrt(np.mean(scored**2))))
                self.assertAlmostEqual(result["velocity_vector_mae_km_s"], float(scored.mean() / 1000))
                self.assertAlmostEqual(result["initial_position_error_km"], first - 1)
                self.assertEqual(result["final_position_error_km"], last)
                np.testing.assert_allclose(result["position_rtn_mae_km"], [scored.mean(), 0, 0])

    def test_first_heldout_point_is_scored_and_preceding_boundary_is_excluded(self):
        predicted_r = self.reference_r.copy()
        predicted_r[120, 0] += 999
        predicted_r[121, 0] += 10
        metrics = split_metrics({"r": predicted_r, "v": self.reference_v.copy()},
                                self.reference_r, self.reference_v)["test"]
        self.assertEqual(metrics["future_samples"], 60)
        self.assertEqual(metrics["initial_position_error_km"], 999)
        self.assertAlmostEqual(metrics["position_vector_mae_km"], 10 / 60)
        self.assertAlmostEqual(metrics["position_vector_rmse_km"], 10 / np.sqrt(60))
        self.assertEqual(metrics["position_max_km"], 10)


if __name__ == "__main__":
    unittest.main()
