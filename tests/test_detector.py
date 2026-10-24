"""No-network tests: color classifier, hysteresis, resync, and config validation."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest

import cv2
import numpy as np

from main import validate_config
from modules.detector import CalibrationError, StateTracker, classify, check_frame_geometry
from modules.events import emit

ROOT = Path(__file__).resolve().parents[1]


class TestIndicator(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config.json").read_text())
        for indicator in self.config["indicators"]:
            indicator["roi"] = [10, 10, 24, 24]
        self.config["video"].update({"reference_width": 48, "reference_height": 48,
                                     "panel_roi": [5, 5, 35, 35]})

    def lamp(self, bgr):
        frame = np.zeros((48, 48, 3), np.uint8)
        cv2.circle(frame, (22, 22), 10, bgr, -1)
        return frame

    def test_red_lens_off_is_not_an_alarm(self):
        red = self.config["indicators"][0]
        self.assertEqual(classify(self.lamp((0, 0, 75)), red, 0.82).state, "OFF")
        self.assertEqual(classify(self.lamp((0, 0, 245)), red, 0.82).state, "ON")

    def test_hysteresis_uncertain_light(self):
        red = self.config["indicators"][0]
        frame = np.zeros((48, 48, 3), np.uint8)
        frame[18:25, 18:25] = (0, 0, 245)
        observed = classify(frame, red, 0.82)
        self.assertEqual(observed.state, "UNKNOWN")

    def test_green_and_yellow(self):
        green, yellow = self.config["indicators"][1:3]
        self.assertEqual(classify(self.lamp((0, 235, 0)), green, 0.82).state, "ON")
        self.assertEqual(classify(self.lamp((0, 235, 235)), yellow, 0.82).state, "ON")

    def test_confirmation_and_no_repeated_events(self):
        track = StateTracker(self.config["indicators"], 2)
        key = "RED_1"
        self.assertIsNone(track.update(key, "OFF", "t1"))
        self.assertIsNone(track.update(key, "OFF", "t2"))
        self.assertIsNone(track.update(key, "ON", "t3"))
        event = track.update(key, "ON", "t4")
        self.assertEqual(event.description, "OFF -> ON")
        self.assertIsNone(track.update(key, "ON", "t5"))
        self.assertIsNone(track.update(key, "OFF", "t6"))
        self.assertEqual(track.update(key, "OFF", "t7").description, "ON -> OFF")

    def test_startup_mismatch_and_reconnect(self):
        track = StateTracker(self.config["indicators"], 2)
        self.assertIsNone(track.update("RED_1", "ON", "t1"))
        self.assertIn("STARTUP_MISMATCH", track.update("RED_1", "ON", "t2").description)
        track.invalidate()
        self.assertIsNone(track.update("RED_1", "OFF", "t3"))
        self.assertIsNone(track.update("RED_1", "OFF", "t4"))
        self.assertIsNone(track.update("RED_1", "ON", "t5"))
        self.assertEqual(track.update("RED_1", "ON", "t6").description, "OFF -> ON")

    def test_unknown_invalidates_transitions(self):
        track = StateTracker(self.config["indicators"], 2)
        track.update("RED_1", "OFF", "1")
        track.update("RED_1", "OFF", "2")
        track.update("RED_1", "UNKNOWN", "3")
        self.assertIsNone(track.update("RED_1", "ON", "4"))
        event = track.update("RED_1", "ON", "5")
        self.assertIsNotNone(event)
        self.assertNotIn("OFF -> ON", event.description)

    def test_geometry_and_config(self):
        validate_config(self.config, calibrated=True)
        check_frame_geometry(self.lamp((0, 0, 50)), self.config)
        broken = deepcopy(self.config)
        broken["video"]["reference_width"] = 50
        with self.assertRaises(CalibrationError):
            check_frame_geometry(self.lamp((0, 0, 50)), broken)
        broken = deepcopy(self.config)
        broken["indicators"][0]["on_fraction"] = -1
        with self.assertRaises(ValueError):
            validate_config(broken, calibrated=True)

    def test_failed_screenshot_never_claims_path(self):
        from datetime import datetime, timezone
        from modules.detector import Change
        dt = datetime(2026, 10, 23, 11, 31, 24, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            output = StringIO()
            with patch("modules.events.cv2.imwrite", return_value=False), redirect_stdout(output):
                emit([Change("RED_1", "System Failure", "OFF -> ON")],
                     self.lamp((0, 0, 245)), dt, Path(directory), 90)
            self.assertIn("SCREENSHOT_ERROR", output.getvalue())
            self.assertNotIn(".jpg", output.getvalue())
            self.assertEqual(list(Path(directory).rglob("*.jpg")), [])

    def test_shared_screenshot_for_simultaneous_events(self):
        from datetime import datetime, timezone
        from modules.detector import Change
        dt = datetime(2026, 10, 23, 11, 31, 24, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            output = StringIO()
            changes = [Change("RED_1", "System Failure", "OFF -> ON"),
                       Change("RED_2", "Pressure Failure", "OFF -> ON")]
            with redirect_stdout(output):
                emit(changes, self.lamp((0, 0, 245)), dt, Path(directory), 90)
            lines = output.getvalue().strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0].split(" | ")[-1], lines[1].split(" | ")[-1])
            self.assertEqual(len(list(Path(directory).rglob("*.jpg"))), 1)

    def test_screenshot_is_real_file(self):
        from datetime import datetime, timezone
        from modules.detector import Change
        with tempfile.TemporaryDirectory() as directory:
            dt = datetime(2026, 10, 23, 11, 31, 24, tzinfo=timezone.utc)
            emit([Change("RED_1", "System Failure", "OFF -> ON")],
                 self.lamp((0, 0, 245)), dt, Path(directory), 90)
            photos = list(Path(directory).rglob("*.jpg"))
            self.assertEqual(len(photos), 1)
            self.assertIsNotNone(cv2.imread(str(photos[0])))


if __name__ == "__main__":
    unittest.main()
