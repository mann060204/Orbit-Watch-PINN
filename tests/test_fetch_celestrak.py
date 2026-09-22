"""Offline tests; all orbital values below are synthetic fixtures, not live data."""

import contextlib
import csv
from datetime import timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import fetch_celestrak as app


def record(catalog_id=100123, epoch="2026-09-10T00:00:00.000000"):
    return {
        "NORAD_CAT_ID": catalog_id, "OBJECT_NAME": "SYNTHETIC TEST OBJECT",
        "EPOCH": epoch, "MEAN_MOTION": 15.0, "ECCENTRICITY": 0.001,
        "INCLINATION": 51.6, "RA_OF_ASC_NODE": 120.0,
        "ARG_OF_PERICENTER": 45.0, "MEAN_ANOMALY": 60.0,
        "BSTAR": 0.0001, "MEAN_MOTION_DOT": 0.00001, "MEAN_MOTION_DDOT": 0.0,
    }


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)

    def test_large_catalog_ids_and_optional_names(self):
        row = record(799123456)
        del row["OBJECT_NAME"]
        self.assertEqual(app.validate_records([row]), [row])

    def test_invalid_responses_are_rejected(self):
        bad_rows = [[], {"error": "no data"}, [None], [record(0)], [record(True)],
                    [record(epoch="yesterday")], [record(1000000000)]]
        for field, value in [("MEAN_MOTION", 0), ("ECCENTRICITY", 1),
                             ("INCLINATION", 181), ("BSTAR", float("nan")),
                             ("RA_OF_ASC_NODE", 360), ("MEAN_MOTION_DOT", None)]:
            bad_rows.append([{**record(), field: value}])
        missing = record()
        del missing["EPOCH"]
        bad_rows.append([missing])
        for payload in bad_rows:
            with self.subTest(payload=payload), self.assertRaises(app.DataError):
                app.validate_records(payload)

    def test_naive_epochs_are_utc(self):
        self.assertEqual(app.parse_epoch("2026-09-10T00:00:00"),
                         app.parse_epoch("2026-09-10T05:30:00+05:30"))

    @patch.object(app, "download_records", return_value=[record()])
    def test_repeat_runs_reuse_cache(self, download):
        first = app.fetch_group("stations", self.output)
        second = app.fetch_group("STATIONS", self.output)
        self.assertFalse(first["from_cache"])
        self.assertTrue(second["from_cache"])
        self.assertEqual(first["fetched_at_utc"], second["fetched_at_utc"])
        download.assert_called_once()
        self.assertIn("GROUP=stations&FORMAT=JSON", download.call_args.args[0])

    @patch.object(app, "download_records", return_value=[record()])
    def test_expired_cache_refreshes_and_offline_does_not(self, download):
        app.fetch_group("stations", self.output)
        cache = self.output / "cache" / "stations.json"
        payload = json.loads(cache.read_text())
        payload["fetched_at_utc"] = app.utc_text(app.utc_now() - timedelta(hours=3))
        app.write_json(cache, payload)
        offline = app.fetch_group("stations", self.output, offline=True)
        self.assertEqual(download.call_count, 1)
        self.assertEqual(offline["fetched_at_utc"], payload["fetched_at_utc"])
        app.fetch_group("stations", self.output)
        self.assertEqual(download.call_count, 2)

    @patch.object(app, "download_records")
    def test_missing_offline_cache_never_downloads(self, download):
        with self.assertRaises(app.DataError):
            app.fetch_group("stations", self.output, offline=True)
        download.assert_not_called()

    @patch.object(app, "download_records")
    def test_corrupt_cache_never_triggers_repeated_requests(self, download):
        app.write_json(self.output / "cache" / "stations.json", {})
        with self.assertRaises(app.DataError):
            app.fetch_group("stations", self.output)
        download.assert_not_called()

    @patch.object(app, "build_opener")
    def test_http_errors_stop_without_retry(self, opener):
        for code in [301, 403, 404, 429, 500]:
            with self.subTest(code=code):
                opener.return_value.open.reset_mock()
                opener.return_value.open.side_effect = HTTPError(
                    "https://celestrak.org", code, "test error", {}, io.BytesIO(b"error")
                )
                with self.assertRaisesRegex(app.DataError, f"HTTP {code}"):
                    app.download_records("https://celestrak.org", 30)
                opener.return_value.open.assert_called_once()

    @patch.object(app, "build_opener")
    def test_html_and_empty_json_responses_fail(self, opener):
        response = opener.return_value.open.return_value.__enter__.return_value
        response.status = 200
        for body in [b"<html>Service unavailable</html>", b"[]"]:
            with self.subTest(body=body), self.assertRaises(app.DataError):
                response.read.return_value = body
                app.download_records("https://celestrak.org", 30)

    @patch.object(app, "build_opener")
    def test_timeout_is_actionable(self, opener):
        opener.return_value.open.side_effect = URLError("timed out")
        with self.assertRaisesRegex(app.DataError, "No automatic retry"):
            app.download_records("https://celestrak.org", 30)

    def test_duplicates_use_newest_epoch_and_keep_memberships(self):
        older = record(epoch="2026-09-09T00:00:00")
        newer = record(epoch="2026-09-10T00:00:00")
        merged = app.merge_records([
            {"group": "stations", "records": [newer]},
            {"group": "active", "records": [older]},
        ])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["EPOCH"], newer["EPOCH"])
        self.assertEqual(merged[0]["SOURCE_GROUPS"], ["active", "stations"])
        self.assertNotIn("SOURCE_GROUPS", newer)

    @patch.object(app, "download_records", return_value=[record()])
    def test_snapshot_csv_json_and_latest_pointer_agree(self, download):
        data = app.fetch_group("stations", self.output)
        snapshot, manifest = app.save_snapshot([data], self.output)
        latest = json.loads((self.output / "latest.json").read_text())
        self.assertEqual(self.output / latest["snapshot"], snapshot)
        self.assertEqual(manifest["unique_objects"], 1)
        self.assertEqual(manifest["sources"][0]["fetched_at_utc"], data["fetched_at_utc"])
        self.assertEqual(json.loads((snapshot / "raw" / "stations.json").read_text()), [record()])
        with (snapshot / "orbital_data.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["NORAD_CAT_ID"], "100123")
        self.assertEqual(rows[0]["SOURCE_GROUPS"], "stations")

    @patch.object(app, "download_records", side_effect=[[record()], app.DataError("HTTP 403")])
    def test_group_failure_stops_run_and_preserves_previous_latest(self, download):
        app.write_json(self.output / "latest.json", {"previous": "snapshot"})
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = app.main(["--groups", "stations", "active", "fengyun-1c-debris",
                               "--output-dir", str(self.output)])
        self.assertEqual(status, 1)
        self.assertEqual(download.call_count, 2)
        self.assertEqual(json.loads((self.output / "latest.json").read_text()),
                         {"previous": "snapshot"})
        self.assertFalse((self.output / "snapshots").exists())


if __name__ == "__main__":
    unittest.main()
