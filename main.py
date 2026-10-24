"""ColorCatcher MVP 0.2: baseline-relative indicator monitor."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cv2

from modules.detector import (CalibrationError, StateTracker, brightness_score,
                              check_frame_geometry, classify, crop_roi)
from modules.events import emit
from modules.stream import StreamError, StreamReader

LOG = logging.getLogger("colorcatcher")

def _number(value: object, label: str, minimum: float, maximum: float, *, integer: bool = False) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (integer and not isinstance(value, int))
            or not minimum <= value <= maximum):
        raise ValueError(f"{label} must be {'an integer' if integer else 'a finite number'} in [{minimum}, {maximum}]")


def _roi(value: object, label: str, width: int, height: int, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} must be [x, y, width, height]")
    for part in value:
        _number(part, label, 0, max(width, height), integer=True)
    x, y, w, h = value
    if w < 4 or h < 4 or x + w > width or y + h > height:
        raise ValueError(f"{label}: ROI outside reference image or smaller than 4x4")


def validate_config(config: dict, *, operational: bool = True) -> None:
    """Single shared validator for runtime and calibration (unverified mode)."""
    if not isinstance(config, dict) or type(config.get("schema_version")) is not int or config["schema_version"] != 3:
        raise ValueError("Unsupported config.json; schema_version must be 3")
    count = config.get("sensor_count")
    _number(count, "sensor_count", 1, 100_000, integer=True)
    stream, video, det, output = (config[key] for key in ("stream", "video", "detection", "output"))
    if not isinstance(stream.get("url"), str):
        raise ValueError("stream.url must be a string")
    for field in ("open_timeout_seconds", "read_timeout_seconds"):
        _number(stream[field], f"stream.{field}", 0.1, 120)
    _number(stream["reconnect_delay_seconds"], "stream.reconnect_delay_seconds", 0.1, 3600)
    fps = video["detection_fps"]
    if fps is not None:
        _number(fps, "video.detection_fps", 0.1, 240)
    width, height = video["reference_width"], video["reference_height"]
    _number(width, "video.reference_width", 4, 16384, integer=True)
    _number(height, "video.reference_height", 4, 16384, integer=True)
    _number(det["confirmation_frames"], "detection.confirmation_frames", 1, 100, integer=True)
    _number(det["max_uncertain_seconds"], "detection.max_uncertain_seconds", .01, 3600)
    _number(output["jpeg_quality"], "output.jpeg_quality", 1, 100, integer=True)
    _number(output["screenshot_retries"], "output.screenshot_retries", 1, 10, integer=True)
    _number(det["brightness_change_percent"], "detection.brightness_change_percent", 0.01, 1000)
    if not isinstance(output.get("save_verification_preview"), bool):
        raise ValueError("output.save_verification_preview must be boolean")
    if not isinstance(output.get("screenshots_dir"), str) or not output["screenshots_dir"].strip():
        raise ValueError("output.screenshots_dir must be a nonempty path")
    if not isinstance(output.get("timezone"), str):
        raise ValueError("output.timezone must be local or an IANA timezone")
    if output["timezone"] != "local":
        try:
            ZoneInfo(output["timezone"])
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unsupported timezone: {output['timezone']}") from exc
    _roi(config.get("panel_roi"), "panel_roi", width, height, optional=True)
    indicators = config.get("indicators")
    if not isinstance(indicators, list) or len(indicators) != count:
        raise ValueError("len(indicators) must equal sensor_count")
    ids: set[str] = set()
    for index, item in enumerate(indicators):
        if not isinstance(item, dict):
            raise ValueError(f"indicators[{index}] must be an object")
        ident, name = item.get("id"), item.get("name")
        if not isinstance(ident, str) or not ident.strip() or ident in ids:
            raise ValueError(f"indicators[{index}].id must be unique and nonempty")
        ids.add(ident)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{ident}.name must be nonempty")
        if item.get("normal_state") not in ("ON", "OFF"):
            raise ValueError(f"{ident}.normal_state must be ON or OFF")
        if item.get("baseline_state") is not None and item["baseline_state"] not in ("ON", "OFF"):
            raise ValueError(f"{ident}.baseline_state must be ON or OFF")
        _roi(item.get("roi"), f"{ident}.roi", width, height, optional=not operational)
        _roi(item.get("reference_roi"), f"{ident}.reference_roi", width, height, optional=True)
        baseline = item.get("baseline_brightness")
        baseline_ref = item.get("baseline_reference_brightness")
        if baseline is not None:
            _number(baseline, f"{ident}.baseline_brightness", 0, 255)
        if baseline_ref is not None:
            _number(baseline_ref, f"{ident}.baseline_reference_brightness", 0, 255)
        if item.get("reference_roi") is None and baseline_ref is not None:
            raise ValueError(f"{ident}.baseline_reference_brightness requires reference_roi")
        if operational and (baseline is None or item.get("baseline_state") not in ("ON", "OFF") or
                            (item.get("reference_roi") is not None and baseline_ref is None)):
            raise ValueError(f"{ident}: brightness baseline missing; run calibrate.py")


def load_config(path: Path, *, operational: bool = True) -> dict:
    with path.open(encoding="utf-8") as source:
        config = json.load(source)
    validate_config(config, operational=operational)
    return config


def source_path(setting: str, config_path: Path, *, command_line: bool) -> str:
    if not setting.strip() or urlsplit(setting).scheme.lower() in {"http", "https"}:
        return setting
    path = Path(setting).expanduser()
    return str(path if path.is_absolute() else (Path.cwd() if command_line else config_path.parent) / path)


def output_path(config_path: Path, value: str) -> Path:
    directory = Path(value).expanduser()
    return directory if directory.is_absolute() else config_path.parent / directory


def local_time(config: dict) -> datetime:
    zone = config["output"]["timezone"]
    return datetime.now().astimezone() if zone == "local" else datetime.now(ZoneInfo(zone))


def inspect_image(config: dict, file: Path, out: Path) -> None:
    frame = cv2.imread(str(file), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Cannot read image {file}")
    check_frame_geometry(frame, config)
    marked = frame.copy()
    for sensor in config["indicators"]:
        bright, ref = brightness_score(frame, sensor)
        observation = (classify(frame, sensor, config["detection"]["brightness_change_percent"])
                       if sensor.get("baseline_brightness") is not None else None)
        status = observation.state if observation is not None else "UNCALIBRATED"
        x, y, w, h = sensor["roi"]
        cv2.rectangle(marked, (x, y), (x+w, y+h), (0, 255, 255), 1)
        cv2.putText(marked, sensor["id"], (x, max(y-5, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    .4, (0, 255, 255), 1)
        delta = f"{observation.score:+.2f}%" if observation is not None else "n/a"
        print(f"{sensor['id']}: {status:12} mean={bright:.2f} "
              f"calibration={sensor.get('baseline_brightness')} delta={delta} ambient={ref}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), marked):
        raise OSError(f"Cannot write {out}")
    print(f"Annotated image: {out}")


def capture_frame(config: dict, source: str, dest: Path) -> None:
    with StreamReader(source, config["stream"]["open_timeout_seconds"],
                      config["stream"]["read_timeout_seconds"]) as reader:
        frame = reader.read_first_frame()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest), frame):
        raise OSError(f"Cannot save decoded frame to {dest}")
    print(f"Frame captured: {dest} ({frame.shape[1]}x{frame.shape[0]})")


def monitor(config: dict, config_path: Path, source: str) -> None:
    tracker = StateTracker(config["indicators"], config["detection"]["confirmation_frames"],
                           config["detection"]["max_uncertain_seconds"])
    fps = config["video"]["detection_fps"]
    period = 1.0 / fps if fps is not None else None
    shot_directory = output_path(config_path, config["output"]["screenshots_dir"])
    live = urlsplit(source).scheme.lower() in {"http", "https"}
    while True:
        reader = StreamReader(source, config["stream"]["open_timeout_seconds"],
                              config["stream"]["read_timeout_seconds"])
        try:
            reader.open()
            LOG.info("Video source opened (%s)", "livestream" if live else "local file")
            next_due = 0.0
            last_grab = time.monotonic()
            last_decoded = last_grab
            decode_unavailable = False
            uncertain = set()
            while True:
                valid = reader.grab()
                now = time.monotonic()
                if not valid:
                    if not live:
                        LOG.info("Local video reached EOF")
                        return
                    raise StreamError("No new video frames")
                if now - last_grab > config["stream"]["read_timeout_seconds"]:
                    raise StreamError("Frame acquisition stalled")
                last_grab = now
                if period is not None:
                    position = now if live else (reader.frame_number - 1) / reader.source_fps
                    if position + 1e-7 < next_due:
                        continue
                    next_due = position + period
                valid, frame = reader.retrieve()
                if not valid or frame is None or frame.size == 0:
                    if not decode_unavailable:
                        LOG.warning("MONITOR_UNAVAILABLE: decode failure")
                        decode_unavailable = True
                    tracker.invalidate()
                    if now - last_decoded > config["stream"]["read_timeout_seconds"]:
                        raise StreamError("No decoded frames within read timeout")
                    continue
                check_frame_geometry(frame, config)
                last_decoded = now
                if decode_unavailable:
                    LOG.info("MONITOR_RESTORED: valid decoded frame")
                    decode_unavailable = False
                observation_clock = now if live else (reader.frame_number - 1) / reader.source_fps
                timestamp = local_time(config)
                clock_stamp = timestamp.isoformat(sep=" ", timespec="milliseconds")
                changes = []
                for indicator in config["indicators"]:
                    observed = classify(frame, indicator, config["detection"]["brightness_change_percent"])
                    ident = indicator["id"]
                    if observed.state == "UNKNOWN" and ident not in uncertain:
                        LOG.warning("INDICATOR_UNCERTAIN: %s delta=%.2f%%", ident, observed.score)
                        uncertain.add(ident)
                    elif observed.state != "UNKNOWN" and ident in uncertain:
                        LOG.info("INDICATOR_RESTORED: %s", ident)
                        uncertain.remove(ident)
                    change = tracker.update(ident, observed.state, clock_stamp, observation_clock,
                                            crop_roi(frame, indicator["roi"]))
                    if change is not None:
                        changes.append(change)
                if emit(changes, frame, timestamp, shot_directory,
                        config["output"]["jpeg_quality"], config["output"]["save_verification_preview"],
                        config["output"]["screenshot_retries"]):
                    LOG.error("EVIDENCE_DEGRADED: one or more screenshot writes failed")
        except CalibrationError:
            raise
        except (StreamError, cv2.error) as exc:
            LOG.error("MONITOR_UNAVAILABLE: %s", exc)
            tracker.invalidate()
            if not live:
                raise
        finally:
            reader.close()
        LOG.info("Reconnecting in %.1f seconds", config["stream"]["reconnect_delay_seconds"])
        time.sleep(config["stream"]["reconnect_delay_seconds"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ColorCatcher MVP 0.2")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--source", help="Authorized video URL or local video (override)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check-config", action="store_true", help="Validate operational config offline")
    group.add_argument("--check-schema", action="store_true", help="Validate uncalibrated config offline")
    group.add_argument("--inspect-image", type=Path, help="Measure an offline original-resolution image")
    group.add_argument("--capture-frame", action="store_true", help="Capture one frame (may open the video URL)")
    group.add_argument("--calibrate-image", type=Path, help="Open calibrate.py GUI with an offline image")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config_path = args.config.resolve()
        if args.calibrate_image:
            from calibrate import main as calibration_main
            return calibration_main(["--config", str(config_path), "--image", str(args.calibrate_image)])
        calibrated = not (args.check_schema or args.inspect_image or args.capture_frame)
        config = load_config(config_path, operational=calibrated)
        if args.check_schema or args.check_config:
            print("VALID: ColorCatcher v3 baseline configuration (offline; accuracy unverified)")
            return 0
        if args.inspect_image:
            inspect_image(config, args.inspect_image, args.output or config_path.parent / "inspection.jpg")
            return 0
        source = source_path(args.source if args.source is not None else config["stream"]["url"],
                             config_path, command_line=args.source is not None)
        if not source.strip():
            raise ValueError("Set stream.url or --source")
        if args.capture_frame:
            capture_frame(config, source, args.output or config_path.parent / "calibration_frame.jpg")
        else:
            monitor(config, config_path, source)
        return 0
    except (OSError, KeyError, TypeError, ValueError, StreamError, cv2.error) as exc:
        LOG.error("%s: %s", type(exc).__name__, exc)
        return 2
    except KeyboardInterrupt:
        LOG.info("Stopped by operator")
        return 0


if __name__ == "__main__":
    sys.exit(main())
