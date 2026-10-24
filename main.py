"""Factory Indicator Monitor MVP 0.1. All network access is opt-in at runtime."""

import argparse
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from urllib.parse import urlsplit

import cv2
import numpy as np

from modules.detector import CalibrationError, StateTracker, check_frame_geometry, classify
from modules.events import emit
from modules.stream import StreamError, StreamReader

LOG = logging.getLogger("factory_monitor")
IDS = ("RED_1", "GREEN", "YELLOW", "RED_2", "RED_3")


def _number(value, label, minimum, maximum, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: must be a finite number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{label}: must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label}: must be between {minimum} and {maximum}")


def _roi(value, label, frame_width, frame_height, *, allow_none):
    if value is None and allow_none:
        return
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} must be [x, y, width, height] (calibrate first)")
    x, y, w, h = value
    for coordinate in value:
        _number(coordinate, label, 0, max(frame_width, frame_height), integer=True)
    if w < 4 or h < 4 or x + w > frame_width or y + h > frame_height:
        raise ValueError(f"{label}: ROI is too small or outside calibrated video dimensions")


def panel_roi(config: dict):
    """Read the calibrated panel ROI from the root-level configuration field."""
    return config["panel_roi"]


def validate_config(config, *, calibrated=False):
    if not isinstance(config, dict):
        raise ValueError("Root of config.json must be an object")
    stream, video, detection, output = (config[key] for key in ("stream", "video", "detection", "output"))
    if not isinstance(stream["url"], str):
        raise ValueError("stream.url must be a string")
    for key in ("open_timeout_seconds", "read_timeout_seconds"):
        _number(stream[key], f"stream.{key}", 1, 120)
    _number(stream["reconnect_delay_seconds"], "stream.reconnect_delay_seconds", 0.1, 3600)
    _number(video["detection_fps"], "video.detection_fps", 0.1, 60)
    _number(video["min_panel_mean_brightness"], "video.min_panel_mean_brightness", 0, 255)
    _number(video["reference_width"], "video.reference_width", 0, 16384, integer=True)
    _number(video["reference_height"], "video.reference_height", 0, 16384, integer=True)
    _number(detection["confirmation_frames"], "detection.confirmation_frames", 1, 100, integer=True)
    _number(detection["inner_ellipse_fraction"], "detection.inner_ellipse_fraction", 0.3, 1)
    _number(output["jpeg_quality"], "output.jpeg_quality", 1, 100, integer=True)
    if not isinstance(output["screenshots_dir"], str) or not output["screenshots_dir"].strip():
        raise ValueError("output.screenshots_dir must be a non-empty path")
    if not isinstance(output["timezone"], str):
        raise ValueError("output.timezone must be local or an IANA timezone name")
    if output["timezone"] != "local":
        try:
            ZoneInfo(output["timezone"])
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Invalid or unavailable timezone: {output['timezone']}") from exc

    width, height = video["reference_width"], video["reference_height"]
    allow_none = not calibrated
    if calibrated and (width < 1 or height < 1):
        raise ValueError("Video reference dimensions are not calibrated")
    _roi(panel_roi(config), "panel_roi", width, height, allow_none=allow_none)
    indicators = config["indicators"]
    if not isinstance(indicators, list) or [item["id"] for item in indicators] != list(IDS):
        raise ValueError(f"Expected exactly five indicators in order: {', '.join(IDS)}")
    if len({item["name"] for item in indicators}) != 5:
        raise ValueError("Indicator names must be unique")
    for indicator in indicators:
        ident = indicator["id"]
        if not isinstance(indicator["name"], str) or not indicator["name"].strip():
            raise ValueError(f"{ident} requires a name")
        if indicator["color"] not in {"red", "green", "yellow"}:
            raise ValueError(f"{ident}: unsupported color")
        if indicator["normal_state"] not in {"ON", "OFF"}:
            raise ValueError(f"{ident}: normal_state must be ON/OFF")
        _roi(indicator["roi"], f"{ident}.roi", width, height, allow_none=allow_none)
        ranges = indicator["hue_ranges"]
        if not isinstance(ranges, list) or not ranges:
            raise ValueError(f"{ident}: hue_ranges must be non-empty")
        for interval in ranges:
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"{ident}: hue range should be [low, high]")
            for end in interval:
                _number(end, f"{ident}.hue_range", 0, 179, integer=True)
            if interval[0] > interval[1]:
                raise ValueError(f"{ident}: hue range low > high")
        for key in ("min_saturation", "min_brightness"):
            _number(indicator[key], f"{ident}.{key}", 0, 255, integer=True)
        for key in ("off_fraction", "on_fraction"):
            _number(indicator[key], f"{ident}.{key}", 0, 1)
        if indicator["off_fraction"] >= indicator["on_fraction"]:
            raise ValueError(f"{ident}: off_fraction must be below on_fraction")
        if calibrated:
            px, py, pw, ph = panel_roi(config)
            x, y, w, h = indicator["roi"]
            if not (px <= x and py <= y and x + w <= px + pw and y + h <= py + ph):
                raise ValueError(f"{ident}: indicator ROI must be inside panel ROI")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    validate_config(value)
    return value


def resolve_source(value: str, config_path: Path, *, command_line: bool) -> str:
    """Resolve relative config sources against the config file for portability."""
    if not value.strip() or urlsplit(value).scheme.lower() in {"http", "https"}:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((Path.cwd() if command_line else config_path.parent) / path)


def local_time(config: dict) -> datetime:
    zone = config["output"]["timezone"]
    return datetime.now().astimezone() if zone == "local" else datetime.now(ZoneInfo(zone))


def resolve_output_dir(config_path: Path, setting: str) -> Path:
    path = Path(setting).expanduser()
    return path if path.is_absolute() else config_path.parent.resolve() / path


def load_image(path: Path):
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError(f"Cannot read image: {path}")
    return frame


def save_image(path: Path, frame, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f"Image write failed: {path}")
    if os.name == "posix":
        path.chmod(0o600)


def calibrate(config: dict, config_path: Path, image_path: Path) -> None:
    frame = load_image(image_path)
    height, width = frame.shape[:2]
    config["video"]["reference_width"] = width
    config["video"]["reference_height"] = height
    try:
        print("Select entire panel, then SPACE/ENTER. Press C to cancel.")
        panel = tuple(map(int, cv2.selectROI("Select PANEL", frame, False, False)))
        if panel[2] < 4 or panel[3] < 4:
            raise CalibrationError("Panel ROI selection cancelled or too small")
        config["panel_roi"] = list(panel)
        x, y, w, h = panel
        panel_image = frame[y:y + h, x:x + w]
        for indicator in config["indicators"]:
            print(f"Select {indicator['name']} ({indicator['id']}) inside panel image")
            roi = tuple(map(int, cv2.selectROI(indicator["name"], panel_image, False, False)))
            if roi[2] < 4 or roi[3] < 4:
                raise CalibrationError(f"Selection cancelled or too small: {indicator['id']}")
            indicator["roi"] = [x + roi[0], y + roi[1], roi[2], roi[3]]
        validate_config(config, calibrated=True)
    finally:
        cv2.destroyAllWindows()
    # Atomic replace avoids partially overwriting a working configuration.
    tmp = config_path.with_name(config_path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2, ensure_ascii=False)
            file.write("\n")
        if os.name == "posix":
            tmp.chmod(0o600)  # URL may include an access token.
        os.replace(tmp, config_path)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"Calibration saved to: {config_path}")
    print("REQUIRED: inspect ON/OFF measurements and tune thresholds using real frame samples.")


def inspect_image(config: dict, image_path: Path, output_path: Path) -> None:
    frame = load_image(image_path)
    check_frame_geometry(frame, config)
    annotated = frame.copy()
    for indicator in config["indicators"]:
        observation = classify(frame, indicator, config["detection"]["inner_ellipse_fraction"])
        x, y, w, h = indicator["roi"]
        color = (0, 200, 0) if observation.state == "ON" else (
            (0, 0, 220) if observation.state == "OFF" else (0, 200, 255)
        )
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, indicator["id"], (x, max(15, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        print(f"{indicator['name']}: {observation.state:7} "
              f"lit_fraction={observation.lit_fraction:.3f} "
              f"mean_V={observation.mean_brightness:.1f} "
              f"max_V={observation.max_brightness}")
    save_image(output_path, annotated, config["output"]["jpeg_quality"])
    print(f"Annotated frame: {output_path}")


def capture_frame(config: dict, source: str, dest: Path) -> None:
    with StreamReader(source, config["stream"]["open_timeout_seconds"],
                      config["stream"]["read_timeout_seconds"]) as reader:
        frame = reader.read_first_frame()
    save_image(dest, frame, config["output"]["jpeg_quality"])
    print(f"First decoded frame: {dest} ({frame.shape[1]}x{frame.shape[0]})")


def monitor(config: dict, config_path: Path, source: str) -> None:
    tracker = StateTracker(config["indicators"], config["detection"]["confirmation_frames"])
    every = 1.0 / config["video"]["detection_fps"]
    shots_dir = resolve_output_dir(config_path, config["output"]["screenshots_dir"])
    live = source.startswith(("http://", "https://"))
    while True:
        reader = StreamReader(source, config["stream"]["open_timeout_seconds"],
                              config["stream"]["read_timeout_seconds"])
        try:
            reader.open()
            LOG.info("Monitoring source opened (%s)", "live HLS" if live else "local video")
            next_sample_at = 0.0
            last_grab = time.monotonic()
            last_decoded = last_grab
            corrupt_frame_reported = False
            panel_unavailable = False
            uncertain_indicators = set()
            while True:
                ok = reader.grab()  # Decode/grab EVERY frame to drain the backend queue.
                now = time.monotonic()
                if not ok:
                    if not live:
                        LOG.info("Local video reached EOF")
                        return
                    raise StreamError("No frames: stream closed or reception interrupted")
                if now - last_grab > config["stream"]["read_timeout_seconds"]:
                    raise StreamError("Frame acquisition stalled; discarded potentially stale frame")
                last_grab = now

                if live:
                    due = now >= next_sample_at
                    if due:
                        next_sample_at = now + every
                else:
                    # Offline files use their video timeline, not wall-clock decode speed.
                    media_seconds = (reader.frame_number - 1) / reader.source_fps
                    due = media_seconds + 1e-7 >= next_sample_at
                    if due:
                        next_sample_at = media_seconds + every
                if not due:
                    continue

                valid, frame = reader.retrieve()
                if not valid or frame is None or frame.size == 0:
                    if not corrupt_frame_reported:
                        LOG.warning("MONITOR_UNAVAILABLE: frame decode failed")
                        corrupt_frame_reported = True
                    tracker.invalidate()
                    if now - last_decoded >= config["stream"]["read_timeout_seconds"]:
                        raise StreamError("No valid decoded frames before configured timeout")
                    continue
                last_decoded = now
                if corrupt_frame_reported:
                    LOG.info("MONITOR_RESTORED: valid video frame received")
                    corrupt_frame_reported = False
                check_frame_geometry(frame, config)
                px, py, pw, ph = panel_roi(config)
                panel = frame[py : py + ph, px : px + pw]
                if float(np.mean(cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY))) < config["video"]["min_panel_mean_brightness"]:
                    tracker.invalidate()
                    if not panel_unavailable:
                        LOG.warning("MONITOR_UNAVAILABLE: panel too dark/occluded to classify")
                        panel_unavailable = True
                    continue
                if panel_unavailable:
                    LOG.info("MONITOR_RESTORED: panel visible; establishing baseline")
                    panel_unavailable = False
                timestamp = local_time(config)
                stamp = timestamp.isoformat(sep=" ", timespec="milliseconds")
                changes = []
                for indicator in config["indicators"]:
                    observation = classify(
                        frame, indicator, config["detection"]["inner_ellipse_fraction"]
                    )
                    ident = indicator["id"]
                    if observation.state == "UNKNOWN":
                        if ident not in uncertain_indicators:
                            LOG.warning("INDICATOR_UNCERTAIN: %s lit_fraction=%.3f; state is UNKNOWN",
                                        ident, observation.lit_fraction)
                            uncertain_indicators.add(ident)
                    elif ident in uncertain_indicators:
                        LOG.info("INDICATOR_RESTORED: %s; establishing stable state", ident)
                        uncertain_indicators.remove(ident)
                    change = tracker.update(ident, observation.state, stamp)
                    if change is not None:
                        changes.append(change)
                emit(changes, frame, timestamp, shots_dir, config["output"]["jpeg_quality"])
        except CalibrationError:
            # Camera resolution drift is an operator action, not a recoverable network error.
            raise
        except (StreamError, cv2.error) as exc:
            LOG.error("MONITOR_UNAVAILABLE: %s", exc)
            tracker.invalidate()
            if not live:
                raise
        finally:
            reader.close()
        LOG.info("Reconnecting after %.1fs", config["stream"]["reconnect_delay_seconds"])
        time.sleep(config["stream"]["reconnect_delay_seconds"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Five-light factory livestream monitor")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.json")
    parser.add_argument("--source", help="Authorized HLS URL or local video file (overrides config)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-config", action="store_true", help="Offline validation; no network")
    modes.add_argument("--capture-frame", action="store_true", help="Save one source frame (may access HLS)")
    modes.add_argument("--calibrate-image", type=Path, metavar="IMAGE", help="Offline interactive ROI selection")
    modes.add_argument("--inspect-image", type=Path, metavar="IMAGE", help="Offline indicator measurement")
    parser.add_argument("--output", type=Path, help="Output file for --capture-frame/--inspect-image")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config_path = args.config.resolve()
        config = load_config(config_path)
        source = resolve_source(
            args.source if args.source is not None else config["stream"]["url"],
            config_path,
            command_line=args.source is not None,
        )
        if args.calibrate_image is not None:
            calibrate(config, config_path, args.calibrate_image)
        elif args.check_config:
            validate_config(config, calibrated=True)
            print("VALID: five indicator ROIs, frame dimensions, and thresholds configured (offline only)")
            print("NOTE: actual detector accuracy cannot be verified without real reference frames")
        elif args.inspect_image is not None:
            validate_config(config, calibrated=True)
            inspect_image(config, args.inspect_image,
                          args.output or config_path.parent / "inspection.jpg")
        elif args.capture_frame:
            if not source.strip():
                raise ValueError("Provide --source or stream.url for capture")
            capture_frame(config, source, args.output or config_path.parent / "calibration_frame.jpg")
        else:
            validate_config(config, calibrated=True)
            if not source.strip():
                raise ValueError("Set stream.url or --source before monitoring")
            monitor(config, config_path, source)
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError,
            StreamError, cv2.error) as exc:
        LOG.error("%s: %s", type(exc).__name__, exc)
        return 2
    except KeyboardInterrupt:
        LOG.info("Stopped by operator")
        return 0


if __name__ == "__main__":
    sys.exit(main())
