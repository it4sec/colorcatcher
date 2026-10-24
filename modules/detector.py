"""Color-independent illumination measurements and independent sensor state machines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

State = Literal["ON", "OFF", "UNKNOWN"]


class CalibrationError(ValueError):
    """ROI geometry or video dimensions differ from calibrated configuration."""


@dataclass(frozen=True)
class Observation:
    state: State
    score: float


@dataclass(frozen=True)
class Change:
    indicator_id: str
    name: str
    description: str
    kind: str  # TRANSITION, STARTUP_MISMATCH, RESYNC_MISMATCH
    roi: tuple[int, int, int, int]
    previous_crop: np.ndarray | None


def check_frame_geometry(frame: np.ndarray, config: dict) -> None:
    """Never shift regions when a camera changes its frame resolution."""
    expected = (config["video"]["reference_height"], config["video"]["reference_width"])
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise CalibrationError("Expected a three-channel uint8 BGR frame")
    if frame.shape[:2] != expected:
        raise CalibrationError(
            f"Frame size {frame.shape[1]}x{frame.shape[0]} differs from "
            f"calibrated {expected[1]}x{expected[0]}"
        )


def crop_roi(frame: np.ndarray, roi: list[int] | tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    region = frame[y:y+h, x:x+w]
    if region.shape[:2] != (h, w) or region.ndim != 3:
        raise CalibrationError(f"ROI {roi} outside decoded frame")
    return region


def brightness_score(frame: np.ndarray, indicator: dict) -> tuple[float, float | None]:
    """Mean grayscale intensity of *all* ROI pixels, with optional ambient ROI.

    OpenCV's BGR-to-gray conversion is applied identically at calibration and
    runtime.  Each mean equals the sum of gray values / number of ROI pixels.
    """
    sensor_gray = cv2.cvtColor(crop_roi(frame, indicator["roi"]), cv2.COLOR_BGR2GRAY)
    brightness = float(cv2.mean(sensor_gray)[0])
    ref_roi = indicator.get("reference_roi")
    reference = None
    if ref_roi is not None:
        ref_gray = cv2.cvtColor(crop_roi(frame, ref_roi), cv2.COLOR_BGR2GRAY)
        reference = float(cv2.mean(ref_gray)[0])
    return brightness, reference


def classify(frame: np.ndarray, indicator: dict, change_percent: float) -> Observation:
    """Compare mean illumination with the saved single-frame baseline.

    A >N% positive OR negative deviation means DIFFERENT FROM BASELINE.
    The saved normal_state labels that baseline; its opposite labels the
    deviation.  This is change detection, not independent ON/OFF verification.
    """
    brightness, reference = brightness_score(frame, indicator)
    baseline = indicator.get("baseline_brightness")
    if baseline is None:
        raise CalibrationError(f"{indicator['id']} has no saved brightness baseline")
    delta = brightness - baseline
    if indicator.get("reference_roi") is not None:
        baseline_ref = indicator.get("baseline_reference_brightness")
        if baseline_ref is None or reference is None:
            raise CalibrationError(f"{indicator['id']} has no saved ambient baseline")
        delta -= reference - baseline_ref
    percentage = 100.0 * delta / max(baseline, 1.0)
    initial: State = indicator["baseline_state"]
    changed = abs(percentage) > change_percent
    state: State = (("ON" if initial == "OFF" else "OFF") if changed else initial)
    return Observation(state, percentage)


class StateTracker:
    """Confirmed states survive short uncertainty; stream interruptions require resync."""

    def __init__(self, indicators: list[dict], confirmation_frames: int, max_uncertain_seconds: float):
        self.indicators = {item["id"]: item for item in indicators}
        self.confirmation_frames = confirmation_frames
        self.max_uncertain_seconds = max_uncertain_seconds
        self.confirmed: dict[str, State] = {key: "UNKNOWN" for key in self.indicators}
        self.candidate: dict[str, State] = {key: "UNKNOWN" for key in self.indicators}
        self.count = dict.fromkeys(self.indicators, 0)
        self.uncertain_since: dict[str, float | None] = dict.fromkeys(self.indicators)
        self.requires_resync = dict.fromkeys(self.indicators, False)
        self.started = dict.fromkeys(self.indicators, False)
        self.previous_crop: dict[str, np.ndarray | None] = dict.fromkeys(self.indicators)
        self.last_change: dict[str, str | None] = dict.fromkeys(self.indicators)
        self.latest_observed: dict[str, State] = dict.fromkeys(self.indicators, "UNKNOWN")

    def invalidate(self) -> None:
        """Called after a video discontinuity, preserving diagnostic history."""
        for key in self.indicators:
            self.candidate[key] = "UNKNOWN"
            self.count[key] = 0
            self.uncertain_since[key] = None
            if self.started[key]:
                self.requires_resync[key] = True

    def update(self, indicator_id: str, observed: State, timestamp: str,
               clock: float, roi_image: np.ndarray) -> Change | None:
        indicator = self.indicators[indicator_id]
        self.latest_observed[indicator_id] = observed
        if observed == "UNKNOWN":
            if self.uncertain_since[indicator_id] is None:
                self.uncertain_since[indicator_id] = clock
            elif clock - self.uncertain_since[indicator_id] > self.max_uncertain_seconds:
                if self.started[indicator_id]:
                    self.requires_resync[indicator_id] = True
            self.candidate[indicator_id] = "UNKNOWN"
            self.count[indicator_id] = 0
            return None
        self.uncertain_since[indicator_id] = None
        old = self.confirmed[indicator_id]
        if observed == old and not self.requires_resync[indicator_id]:
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
        prior = self.previous_crop[indicator_id]
        self.confirmed[indicator_id] = observed
        self.previous_crop[indicator_id] = roi_image.copy()
        self.candidate[indicator_id] = "UNKNOWN"
        self.count[indicator_id] = 0
        resync = self.requires_resync[indicator_id]
        self.requires_resync[indicator_id] = False
        was_started = self.started[indicator_id]
        self.started[indicator_id] = True
        if not was_started:
            if observed == indicator["normal_state"]:
                return None
            kind, description = "STARTUP_MISMATCH", f"STARTUP_MISMATCH expected {indicator['normal_state']} observed {observed}"
        elif old == observed:
            return None
        elif resync:
            kind, description = "RESYNC_MISMATCH", f"RESYNC_MISMATCH {old} -> {observed} (unobserved gap)"
        else:
            kind, description = "TRANSITION", f"{old} -> {observed}"
        self.last_change[indicator_id] = timestamp
        return Change(indicator_id, indicator["name"], description, kind,
                      tuple(indicator["roi"]), prior.copy() if prior is not None else None)
