"""HSV illumination classification and independent confirmed-state tracking."""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

State = Literal["ON", "OFF", "UNKNOWN"]


@dataclass(frozen=True)
class Observation:
    state: State
    lit_fraction: float
    mean_brightness: float
    max_brightness: int


@dataclass(frozen=True)
class Change:
    indicator_id: str
    name: str
    description: str


class CalibrationError(ValueError):
    """The configured ROI or source dimensions are inconsistent."""


def check_frame_geometry(frame: np.ndarray, config: dict) -> None:
    """Fail closed rather than silently moving ROIs if the source format changes."""
    expected = (config["video"]["reference_height"], config["video"]["reference_width"])
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise CalibrationError("Expected a three-channel BGR video frame")
    if frame.shape[:2] != expected:
        raise CalibrationError(
            f"Frame size {frame.shape[1]}x{frame.shape[0]} differs from calibrated "
            f"{expected[1]}x{expected[0]}; capture and calibrate a new reference frame"
        )


def classify(frame: np.ndarray, indicator: dict, inner_ellipse_fraction: float) -> Observation:
    """Lit colored fraction within the core of the configured lamp ROI.

    Pure hue matching is insufficient for an unilluminated red lens: pixels
    must also exceed individually configured saturation and brightness minima.
    The hysteresis band between OFF and ON is deliberately UNKNOWN.
    """
    x, y, w, h = indicator["roi"]
    region = frame[y : y + h, x : x + w]
    if region.shape[:2] != (h, w):
        raise CalibrationError(f"Out-of-bounds ROI for {indicator['id']}")
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = np.zeros((h, w), dtype=np.uint8)
    for low_h, high_h in indicator["hue_ranges"]:
        mask |= cv2.inRange(
            hsv,
            (low_h, indicator["min_saturation"], indicator["min_brightness"]),
            (high_h, 255, 255),
        )
    # An ellipse rejects the surrounding mounting plate, borders and neighbors.
    core = np.zeros((h, w), dtype=np.uint8)
    radii = (max(1, int(w * inner_ellipse_fraction / 2)),
             max(1, int(h * inner_ellipse_fraction / 2)))
    cv2.ellipse(core, (w // 2, h // 2), radii, 0, 0, 360, 255, -1)
    pixel_count = cv2.countNonZero(core)
    if pixel_count == 0:
        raise CalibrationError(f"Empty sampling mask for {indicator['id']}")
    lit_fraction = cv2.countNonZero(cv2.bitwise_and(mask, core)) / pixel_count
    brightness = hsv[:, :, 2][core != 0]
    if lit_fraction >= indicator["on_fraction"]:
        state: State = "ON"
    elif lit_fraction <= indicator["off_fraction"]:
        state = "OFF"
    else:
        state = "UNKNOWN"
    return Observation(state, lit_fraction, float(np.mean(brightness)), int(np.max(brightness)))


class StateTracker:
    """Hold five independent states. A gap/UNKNOWN invalidates transition history."""

    def __init__(self, indicators: list[dict], confirmation_frames: int):
        self.indicators = {i["id"]: i for i in indicators}
        self.confirmation_frames = confirmation_frames
        self.confirmed: dict[str, State] = {key: "UNKNOWN" for key in self.indicators}
        self.candidate: dict[str, State] = {key: "UNKNOWN" for key in self.indicators}
        self.count = dict.fromkeys(self.indicators, 0)
        self.last_change: dict[str, str | None] = dict.fromkeys(self.indicators)
        self.baseline_reason = dict.fromkeys(self.indicators, "STARTUP_MISMATCH")

    def invalidate(self, reason: str = "RESYNC_MISMATCH") -> None:
        """Forget history across any loss of reliable observation."""
        for key in self.indicators:
            self.confirmed[key] = "UNKNOWN"
            self.candidate[key] = "UNKNOWN"
            self.count[key] = 0
            self.baseline_reason[key] = reason

    def update(self, indicator_id: str, observed: State, timestamp: str) -> Change | None:
        indicator = self.indicators[indicator_id]
        if observed == "UNKNOWN":
            if self.confirmed[indicator_id] != "UNKNOWN":
                self.baseline_reason[indicator_id] = "RESYNC_MISMATCH"
            self.confirmed[indicator_id] = "UNKNOWN"
            self.candidate[indicator_id] = "UNKNOWN"
            self.count[indicator_id] = 0
            return None

        if observed == self.confirmed[indicator_id]:
            self.candidate[indicator_id] = "UNKNOWN"
            self.count[indicator_id] = 0
            return None

        if self.candidate[indicator_id] == observed:
            self.count[indicator_id] += 1
        else:
            self.candidate[indicator_id] = observed
            self.count[indicator_id] = 1

        if self.count[indicator_id] < self.confirmation_frames:
            return None

        previous = self.confirmed[indicator_id]
        self.confirmed[indicator_id] = observed
        self.candidate[indicator_id] = "UNKNOWN"
        self.count[indicator_id] = 0
        self.last_change[indicator_id] = timestamp

        if previous == "UNKNOWN":
            if observed != indicator["normal_state"]:
                return Change(indicator_id, indicator["name"],
                              f"{self.baseline_reason[indicator_id]} expected={indicator['normal_state']} actual={observed}")
            return None
        return Change(indicator_id, indicator["name"], f"{previous} -> {observed}")
