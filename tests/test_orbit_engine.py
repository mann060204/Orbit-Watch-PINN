"""Numerical and screening tests using synthetic geometry and published test orbits."""
from datetime import datetime, timezone
import unittest
from unittest.mock import patch

import numpy as np
from sgp4.api import Satrec
from sgp4.exporter import export_omm

from evaluation import classification_metrics
from orbit_engine import (AnalysisConfig, interval_candidates, julian_dates, make_satellite,
                          propagate, reference_validation, refine_interval, screen)

LINE1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
LINE2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"


class OrbitTests(unittest.TestCase):
    def test_published_vallado_reference(self):
        self.assertTrue(reference_validation()["passed"])

    def test_omm_matches_tle_propagation(self):
        original = Satrec.twoline2rv(LINE1, LINE2)
        restored = make_satellite(export_omm(original, "TEST"))
        for offset in [0, 0.25, 1]:
            ea, ra, va = original.sgp4(original.jdsatepoch, original.jdsatepochF + offset)
            eb, rb, vb = restored.sgp4(original.jdsatepoch, original.jdsatepochF + offset)
            self.assertEqual(ea, eb)
            np.testing.assert_allclose(ra, rb, atol=1e-5, rtol=0)
            np.testing.assert_allclose(va, vb, atol=1e-8, rtol=0)

    def test_large_catalog_id_retained_in_input(self):
        row = export_omm(Satrec.twoline2rv(LINE1, LINE2), "TEST")
        row["NORAD_CAT_ID"] = 799123456
        sat = make_satellite(row)
        self.assertEqual(row["NORAD_CAT_ID"], 799123456)
        error, _, _ = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF)
        self.assertEqual(error, 0)

    def test_time_grid_crosses_midnight(self):
        jd, fraction = julian_dates(datetime(2026, 9, 10, 23, 59, tzinfo=timezone.utc), [0, 60, 120])
        self.assertTrue(np.all((fraction >= 0) & (fraction < 1)))
        np.testing.assert_allclose(np.diff(jd + fraction) * 86400, [60, 60], atol=0.0001)

    def test_fast_between_sample_encounter_is_retained(self):
        left = np.array([[-100., 0, 0], [100., 0, 0]])
        right = -left
        pairs, fractions, _ = interval_candidates(left, right, 1, 0.05, 60)
        np.testing.assert_array_equal(pairs, [[0, 1]])
        np.testing.assert_allclose(fractions, [0.5])

    def test_stationary_relative_trajectories(self):
        left = np.array([[0., 0, 0], [0., 0.5, 0], [1000., 0, 0]])
        right = left + [20., 0, 0]
        pairs, _, _ = interval_candidates(left, right, 1, 0.05, 10)
        np.testing.assert_array_equal(pairs, [[0, 1]])

    def test_swept_search_contains_all_bruteforce_linear_hits(self):
        rng = np.random.default_rng(42)
        left = rng.uniform(-100, 100, (50, 3))
        right = left + rng.uniform(-150, 150, (50, 3))
        pairs, _, _ = interval_candidates(left, right, 10, 0, 60)
        expected = set()
        for a in range(len(left)):
            for b in range(a + 1, len(left)):
                dr = left[a] - left[b]
                dv = right[a] - left[a] - right[b] + left[b]
                t = np.clip(-np.dot(dr, dv) / np.dot(dv, dv), 0, 1)
                if np.linalg.norm(dr + t * dv) <= 10:
                    expected.add((a, b))
        self.assertTrue(expected)
        self.assertEqual(set(map(tuple, pairs)), expected)

    @patch("orbit_engine.pair_state")
    def test_refinement_finds_interior_and_endpoint_minimum(self, pair):
        for tca in [3.234, -2., 12.]:
            pair.side_effect = lambda a,b,start,t: (np.array([t-tca, 0.2, 0]), np.zeros(3), np.zeros(3), np.zeros(3))
            time, distance = refine_interval(None, None, None, 0, 10)
            self.assertAlmostEqual(time, np.clip(tca, 0, 10), places=3)
            self.assertAlmostEqual(distance, np.hypot(time-tca, .2), places=6)

    def test_invalid_propagated_states_are_nan(self):
        sat = Satrec.twoline2rv(LINE1, LINE2)
        with patch("orbit_engine.SatrecArray") as array:
            array.return_value.sgp4.return_value = (np.array([[6]]), np.ones((1,1,3)), np.ones((1,1,3)))
            errors, position, velocity = propagate([sat], datetime.now(timezone.utc), [0])
        self.assertEqual(errors[0,0], 6)
        self.assertTrue(np.isnan(position).all() and np.isnan(velocity).all())

    def test_shared_model_orbits_are_separated(self):
        row = export_omm(Satrec.twoline2rv(LINE1, LINE2), "TEST")
        rows = [{**row, "NORAD_CAT_ID": id} for id in [1, 2]]
        positions = np.array([[[7000., 0, 0], [7000., 50, 0]]]*2)
        result = screen([None,None], rows, positions, np.zeros((2,2), dtype=int),
                        datetime(2000,6,28,tzinfo=timezone.utc), np.array([0.,10.]), AnalysisConfig(include_stale=True))
        self.assertEqual(result[0], [])
        self.assertEqual(result[-1]["shared_orbit_pairs_excluded"], 1)
        self.assertEqual(result[-1]["possible_pairs"], 0)

    def test_config_rejects_unbounded_or_nonfinite_work(self):
        for config in [AnalysisConfig(horizon_hours=0), AnalysisConfig(step_seconds=float("nan")),
                       AnalysisConfig(threshold_km=-1), AnalysisConfig(include_stale="false")]:
            with self.assertRaises(ValueError):
                config.validate()


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.summary={"config":{"threshold_km":5},"window_start_utc":"2026-09-10T00:00:00Z",
                      "window_end_utc":"2026-09-11T00:00:00Z",
                      "screening":{"eligible_ids":[1,2,3,4],"possible_pairs":6}}
        self.labels={"target":"close_approach_within_threshold","threshold_km":5,
                     "window_start_utc":self.summary["window_start_utc"],"window_end_utc":self.summary["window_end_utc"],
                     "reference_source":"Synthetic test fixture ONLY","pairs":[
                         {"object1_id":1,"object2_id":2,"label":1},
                         {"object1_id":1,"object2_id":3,"label":0},
                         {"object1_id":1,"object2_id":4,"label":1},
                         {"object1_id":2,"object2_id":3,"label":0}]}
        self.events=[{"object1_id":1,"object2_id":2},{"object1_id":1,"object2_id":3}]

    def test_no_labels_never_fabricates_accuracy(self):
        result=classification_metrics(self.events,self.summary)
        self.assertIsNone(result["accuracy"])
        self.assertIsNone(result["recall"])
        self.assertEqual(result["status"],"not_evaluable")

    def test_confusion_counts_and_ratios(self):
        result=classification_metrics(self.events,self.summary,self.labels)
        self.assertEqual(result["confusion_matrix"],{"true_positive":1,"true_negative":1,"false_positive":1,"false_negative":1})
        for key in ["accuracy","precision","recall","f1_score","specificity","balanced_accuracy"]:
            self.assertEqual(result[key],0.5)
        self.assertEqual(result["matthews_correlation_coefficient"],0)
        self.assertIsNone(result["roc_auc"])

    def test_undefined_denominators_are_null(self):
        self.labels["pairs"]=[{"object1_id":2,"object2_id":3,"label":0}]
        result=classification_metrics([],self.summary,self.labels)
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])
        self.assertEqual(result["accuracy"],1)

    def test_duplicate_labels_and_wrong_windows_rejected(self):
        self.labels["pairs"].append(self.labels["pairs"][0])
        with self.assertRaises(ValueError): classification_metrics(self.events,self.summary,self.labels)
        self.labels["pairs"].pop()
        self.labels["threshold_km"]=10
        with self.assertRaises(ValueError): classification_metrics(self.events,self.summary,self.labels)


if __name__ == "__main__":
    unittest.main()
