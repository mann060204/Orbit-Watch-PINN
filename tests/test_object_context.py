"""Object context preserves provenance, limited scope, and missing-data semantics."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import numpy as np

from object_context import build_object_info, _snapshot_index


NOW = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)
ROW = {
    "NORAD_CAT_ID": 38241, "OBJECT_NAME": "IRIDIUM 33 DEB", "OBJECT_ID": "1997-051TV",
    "EPOCH": "2026-09-14T00:00:00", "MEAN_MOTION": 14.5, "INCLINATION": 86.3,
    "ECCENTRICITY": 0.001, "RA_OF_ASC_NODE": 110.0, "ARG_OF_PERICENTER": 25.0,
    "MEAN_ANOMALY": 330.0, "SOURCE_GROUPS": ["iridium-33-debris"],
}


class ObjectContextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        _snapshot_index.cache_clear()
        self.addCleanup(_snapshot_index.cache_clear)
        self.service = SimpleNamespace(
            lock=threading.RLock(), catalog_lock=threading.RLock(), rows=[deepcopy(ROW)],
            satellites=[object()], index={38241: 0}, summary=None, events=[],
            manifest={"sources": [{"group": "iridium-33-debris", "fetched_at_utc": "2026-09-14T01:00:00Z"}]},
        )
        self.now = patch("object_context.utc_now", return_value=NOW)
        self.now.start()
        self.addCleanup(self.now.stop)
        self.propagation = patch("object_context.propagate", return_value=(
            np.array([[0]]), np.array([[[7000., 0., 0.]]]), np.array([[[0., 7.5, 0.]]])))
        self.propagation.start()
        self.addCleanup(self.propagation.stop)

    def build(self):
        return build_object_info(self.service, self.service.rows[0]["NORAD_CAT_ID"], self.root)

    def snapshot(self, number, rows):
        folder = self.root / "data" / "snapshots" / f"20260914T000000_{number:06d}Z"
        folder.mkdir(parents=True)
        (folder / "orbital_data.json").write_text(json.dumps(rows), encoding="utf-8")
        (folder / "manifest.json").write_text(json.dumps({"sources": [{
            "group": "iridium-33-debris", "fetched_at_utc": f"2026-09-14T01:{number % 60:02d}:00Z",
        }]}), encoding="utf-8")
        return folder

    def select(self, object_id, name, groups):
        self.service.rows[0].update(NORAD_CAT_ID=object_id, OBJECT_NAME=name, SOURCE_GROUPS=groups)
        self.service.index = {object_id: 0}

    def test_fragment_has_sourced_family_history_and_real_state_units(self):
        info = self.build()
        self.assertEqual(info["id"], 38241)
        self.assertEqual(info["element_age_hours"], 12)
        self.assertIn("Parent satellite", info["history_scope"])
        self.assertIn("individual fragment", info["history_scope"])
        self.assertEqual(info["history_events"][0]["date"], "2009-02-10")
        sources = {source["id"]: source for source in info["sources"]}
        for event in info["history_events"]:
            self.assertTrue(all(source in sources for source in event["source_ids"]))
        self.assertTrue(sources["nasa-collision"]["url"].startswith("https://www.nasa.gov/"))
        self.assertEqual(info["current_state"]["frame"], "TEME")
        self.assertAlmostEqual(info["current_state"]["altitude_km"], 621.863)
        self.assertEqual(info["current_state"]["speed_km_s"], 7.5)
        self.assertIn("not measured position tracks", info["catalog_history_note"])
        json.dumps(info, allow_nan=False)

    def test_original_satellite_designation_is_not_called_an_operating_satellite(self):
        self.select(24946, "IRIDIUM 33", ["iridium-33-debris"])
        info = self.build()
        self.assertEqual(info["category"], "Cataloged satellite remnant")
        self.assertIn("does not imply", info["history_scope"])
        self.assertIn("intact or operating", info["history_scope"])

    def test_unknown_station_does_not_inherit_iss_biography(self):
        self.select(99999, "UNKNOWN VISITOR", ["stations"])
        info = self.build()
        self.assertEqual(info["history_events"], [])
        self.assertIn("No source-verified mission biography", info["history_summary"])
        self.assertNotIn("nasa-iss", {source["id"] for source in info["sources"]})
        self.select(25544, "ISS (ZARYA)", ["stations"])
        self.assertEqual([event["date"] for event in self.build()["history_events"]],
                         ["1998-11-20", "1998-12-06", "2000-11-02"])

    def test_history_is_deduplicated_sorted_bounded_and_isolated_by_object(self):
        for number in range(12):
            row = {**ROW, "EPOCH": f"2026-09-14T00:{number:02d}:00"}
            unrelated = {**row, "NORAD_CAT_ID": 777, "MEAN_MOTION": 999}
            self.snapshot(number, [row, unrelated])
        self.snapshot(12, [{**ROW, "EPOCH": "2026-09-14T00:11:00"}])
        info = self.build()
        history = info["catalog_history"]
        self.assertEqual(len(history), 8)
        self.assertEqual(history[0]["epoch_utc"], "2026-09-14T00:11:00Z")
        self.assertEqual(history[-1]["epoch_utc"], "2026-09-14T00:04:00Z")
        self.assertEqual(history[0]["fetched_at_utc"], "2026-09-14T01:12:00Z")
        self.assertEqual({entry["mean_motion_rev_day"] for entry in history}, {14.5})
        history[0]["mean_motion_rev_day"] = -1
        self.assertEqual(self.build()["catalog_history"][0]["mean_motion_rev_day"], 14.5)

    def test_history_cache_invalidates_on_new_snapshot_and_skips_corrupt_file(self):
        self.snapshot(0, [ROW])
        self.assertEqual(len(self.build()["catalog_history"]), 1)
        self.snapshot(1, [{**ROW, "EPOCH": "2026-09-14T00:02:00"}])
        broken = self.snapshot(2, [ROW])
        (broken / "orbital_data.json").write_text("{broken", encoding="utf-8")
        info = self.build()
        self.assertEqual(len(info["catalog_history"]), 2)
        self.assertIn("1 unavailable or invalid snapshot(s)", info["catalog_history_note"])

    def test_failed_propagation_does_not_publish_nan_or_stale_coordinates(self):
        with patch("object_context.propagate", return_value=(np.array([[6]]),
                   np.full((1, 1, 3), np.nan), np.full((1, 1, 3), np.nan))):
            info = self.build()
        self.assertEqual(info["current_state"]["sgp4_error"], 6)
        for field in ("position_km", "velocity_km_s", "altitude_km", "speed_km_s"):
            self.assertIsNone(info["current_state"][field])
        json.dumps(info, allow_nan=False)

    def test_encounters_keep_matched_total_but_only_five_nearest_and_no_invented_probability(self):
        self.service.summary = {"run_id": "saved-run", "window_start_utc": "2026-09-14T00:00:00Z",
                                "window_end_utc": "2026-09-15T00:00:00Z", "config": {"threshold_km": 5},
                                "screening": {"eligible_ids": [38241], "shared_orbit_pairs": []}}
        self.service.events = [
            {"object1_id": 38241, "object2_id": 100 + n, "object2_name": f"DEB {n}",
             "miss_distance_km": n / 2, "relative_speed_km_s": 8.,
             "tca_utc": "2026-09-14T01:00:00Z", "distance_curve": [[0, 999]]}
            for n in reversed(range(8))]
        self.service.events.append({"object1_id": 1, "object2_id": 2, "miss_distance_km": 0})
        encounters = self.build()["encounters"]
        self.assertEqual(encounters["total"], 8)
        self.assertEqual(len(encounters["events"]), 5)
        self.assertEqual([event["other_id"] for event in encounters["events"]], list(range(100, 105)))
        self.assertEqual(encounters["threshold_km"], 5)
        self.assertTrue(encounters["screened"])
        self.assertTrue(all(event["collision_probability"] is None for event in encounters["events"]))
        self.assertNotIn("distance_curve", encounters["events"][0])
        self.service.summary["screening"]["eligible_ids"] = [1, 2]
        self.service.events = []
        encounters = self.build()["encounters"]
        self.assertFalse(encounters["screened"])
        self.assertIn("not included", encounters["note"])

    def test_unknown_id_raises_without_propagation(self):
        with patch("object_context.propagate") as propagate:
            with self.assertRaises(KeyError):
                build_object_info(self.service, 123, self.root)
        propagate.assert_not_called()

    def test_context_never_reads_private_benchmark_folders(self):
        for folder in ("comparison_results", "ground_truth_results"):
            (self.root / folder).mkdir()
            (self.root / folder / "latest.json").write_text('{"private": "benchmark"}')
        self.snapshot(0, [ROW])
        original = Path.read_text
        def read_text(path, *args, **kwargs):
            self.assertNotIn("comparison_results", path.parts)
            self.assertNotIn("ground_truth_results", path.parts)
            return original(path, *args, **kwargs)
        with patch.object(Path, "read_text", read_text):
            self.assertEqual(self.build()["id"], 38241)


if __name__ == "__main__":
    unittest.main()
