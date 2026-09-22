"""Analytic fixtures verify errors and sampling; they are not benchmark output."""
from datetime import datetime, timedelta, timezone
import unittest
import numpy as np

from orbit_truth import orbit_error_metrics, select_native_samples


class OrbitTruthTests(unittest.TestCase):
    def test_rotating_rtn_basis_uses_reference_orbit(self):
        angle = np.array([0, .3, .8, 1.2])
        radial = np.stack((np.cos(angle), np.sin(angle), np.zeros(4)), axis=1)
        transverse = np.stack((-np.sin(angle), np.cos(angle), np.zeros(4)), axis=1)
        r, v = 7000*radial, 7.5*transverse
        delta = radial+2*transverse+np.array([0, 0, 3])
        metrics, errors = orbit_error_metrics(r+delta, v, r, v)
        np.testing.assert_allclose(errors["position_rtn_km"], np.tile([1, 2, 3], (4, 1)), atol=1e-9)
        self.assertAlmostEqual(metrics["position_vector_rmse_km"], np.sqrt(14), places=9)
        self.assertAlmostEqual(metrics["position_vector_mae_km"], np.sqrt(14), places=9)
        np.testing.assert_allclose(metrics["position_rtn_mae_km"], [1, 2, 3], atol=1e-9)

    def test_initial_error_is_reported_but_excluded_from_aggregate(self):
        r = np.tile([7000., 0, 0], (3, 1))
        v = np.tile([0, 7.5, 0], (3, 1))
        prediction = r.copy()
        prediction[:, 0] += [100, 3, 4]
        metrics, _ = orbit_error_metrics(prediction, v, r, v)
        self.assertEqual(metrics["initial_position_error_km"], 100)
        self.assertAlmostEqual(metrics["position_vector_rmse_km"], np.sqrt(12.5))
        self.assertEqual(metrics["position_vector_mae_km"], 3.5)

    def test_native_sampling_does_not_round_epoch_or_interpolate(self):
        first = datetime(2026, 9, 13, tzinfo=timezone.utc)
        reference = {"times":[first+timedelta(seconds=10*i) for i in range(80)], "quality":np.array(["NOMINAL"]*80)}
        indices, seconds = select_native_samples(reference, first+timedelta(seconds=6), hours=.1, step_seconds=60)
        np.testing.assert_array_equal(indices, [1, 7, 13, 19, 25, 31, 37])
        np.testing.assert_array_equal(seconds, [0, 60, 120, 180, 240, 300, 360])
        with self.assertRaises(ValueError):
            select_native_samples(reference, first, hours=.1, step_seconds=12)

    def test_non_nominal_reference_is_rejected(self):
        first = datetime(2026, 9, 13, tzinfo=timezone.utc)
        reference = {"times":[first+timedelta(seconds=10*i) for i in range(80)], "quality":np.array(["NOMINAL"]*80)}
        reference["quality"][6] = "INVALID"
        with self.assertRaises(ValueError):
            select_native_samples(reference, first, hours=.1)

    def test_degenerate_reference_does_not_define_rtn(self):
        r = np.tile([7000., 0, 0], (3, 1))
        with self.assertRaises(ValueError):
            orbit_error_metrics(r, r, r, r)

    def test_both_models_get_independent_nonzero_errors(self):
        r = np.tile([7000., 0, 0], (3, 1))
        v = np.tile([0, 7.5, 0], (3, 1))
        a, _ = orbit_error_metrics(r+[1, 0, 0], v, r, v)
        b, _ = orbit_error_metrics(r+[2, 0, 0], v, r, v)
        self.assertEqual(a["position_vector_rmse_km"], 1)
        self.assertEqual(b["position_vector_rmse_km"], 2)


if __name__ == "__main__":
    unittest.main()
