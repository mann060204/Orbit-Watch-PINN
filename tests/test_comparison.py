"""Numerical fixtures for comparison logic; never exported as experiment data."""
import unittest
import numpy as np

from comparison_evaluation import flatten_metrics, match_encounters, physics_diagnostics, trajectory_details


def event(identifier, seconds, neural):
    return {"event_id":identifier, "object1_id":1, "object2_id":2,
            "offset_seconds" if neural else "tca_offset_seconds":seconds,
            "tca_utc":"2026-09-11T00:00:00Z", "miss_distance_km":1.0, "relative_speed_km_s":10.0}


class ComparisonTests(unittest.TestCase):
    def test_one_reference_cannot_match_multiple_neural_events(self):
        metric, rows = match_encounters([event("n1", 0, True), event("n2", 1, True)], [event("s1", 0, False)], 5)
        self.assertEqual(metric["matched_events"], 1)
        self.assertEqual(metric["pinn_only_events"], 1)
        self.assertEqual(metric["precision"], .5)
        self.assertEqual(len(rows), 2)

    def test_matching_maximizes_cardinality_before_time_proximity(self):
        metric, _ = match_encounters([event("n1", 0, True), event("n2", 5, True)],
                                     [event("s1", 4, False), event("s2", 10, False)], 6)
        self.assertEqual(metric["matched_events"], 2)
        self.assertEqual(metric["sgp4_only_events"], 0)

    def test_far_events_and_reference_only_events_are_retained(self):
        metric, rows = match_encounters([event("n1", 0, True)], [event("s1", 100, False)], 10)
        self.assertEqual(metric["matched_events"], 0)
        self.assertEqual({row["status"] for row in rows}, {"pinn_only", "sgp4_only"})
        self.assertIsNone(metric["tca_difference_seconds_mae"])

    def test_different_pairs_do_not_match(self):
        reference = event("s1", 0, False)
        reference["object2_id"] = 3
        metric, _ = match_encounters([event("n1", 0, True)], [reference])
        self.assertEqual(metric["matched_events"], 0)

    def test_empty_events_keep_undefined_scores(self):
        metric, rows = match_encounters([], [])
        self.assertIsNone(metric["precision"])
        self.assertIsNone(metric["recall"])
        self.assertEqual(rows, [])

    def test_rtn_components_and_vector_error(self):
        r = np.tile([7000., 0, 0], (3, 3, 1))
        v = np.tile([0, 7.5, 0], (3, 3, 1))
        splits = {"train":np.array([0]), "validation":np.array([1]), "test":np.array([2])}
        rows = [{"EPOCH":"2026-09-11T00:00:00Z"} for _ in range(3)]
        extra, times, objects, pe, ve, rtn = trajectory_details(r+[1, 2, 3], v, r, v, [1, 2, 3], rows, splits, [0, 30, 60])
        np.testing.assert_allclose(extra["test"]["position_rtn_rmse_km"], [1, 2, 3])
        np.testing.assert_allclose(pe, np.sqrt(14))
        np.testing.assert_allclose(ve, 0)
        self.assertEqual(len(times), 12)
        self.assertEqual(len(objects), 3)

    def test_physics_diagnostics_are_finite_and_identified(self):
        r = np.tile([7000., 0, 0], (1, 3, 1))
        v = np.tile([0, 7.5, 0], (1, 3, 1))
        diagnostic = physics_diagnostics(r, v, np.array([0., 30., 60.]))
        self.assertEqual(diagnostic["j2_energy_relative_drift_max"], 0)
        self.assertGreater(diagnostic["central_j2_acceleration_residual_vector_rms_km_s2"], 0)
        self.assertIn("not observational", diagnostic["interpretation"])

    def test_flat_metrics_preserve_unavailable_values(self):
        self.assertEqual(flatten_metrics({"SGP4":{"accuracy":None}}),
                         [{"metric":"SGP4.accuracy", "value":None, "status":"unavailable"}])


if __name__ == "__main__":
    unittest.main()
