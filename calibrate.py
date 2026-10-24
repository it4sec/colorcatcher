#!/usr/bin/env python3
"""Point-and-click, single-frame calibration for the Factory Indicator Monitor MVP.

Run alongside config.json: poetry run python calibrate.py

Requires OpenCV built with GUI support (opencv-python, not headless).
No network connections are made until the user executes this script with a
stream configured. Use --image FILE for completely offline calibration.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2


INDICATORS = (
    ("RED_1", "System Failure"),
    ("GREEN", "Status OK"),
    ("YELLOW", "In Production"),
    ("RED_2", "Pressure Failure"),
    ("RED_3", "Machine Failure"),
)

# Pixel-coordinate ROIs: (x, y, width, height), in the ORIGINAL video frame.
Rect = tuple[int, int, int, int]


def _key(text: object) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _find_indicators(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = config.get("indicators")
    if isinstance(entries, list):
        candidates = [(entry.get("id", ""), entry) for entry in entries if isinstance(entry, dict)]
    elif isinstance(entries, dict):
        candidates = list(entries.items())
    else:
        raise ValueError("config.json must contain an 'indicators' list or object")

    aliases = {
        "RED_1": {"red1", "systemfailure", "upperred"},
        "GREEN": {"green", "statusok"},
        "YELLOW": {"yellow", "inproduction"},
        "RED_2": {"red2", "pressurefailure", "lowerred1"},
        "RED_3": {"red3", "machinefailure", "lowerred2"},
    }
    result: dict[str, dict[str, Any]] = {}
    claimed: set[int] = set()
    for ident, name in INDICATORS:
        matches = [
            obj for ref, obj in candidates
            if isinstance(obj, dict)
            and (aliases[ident] & {_key(ref), _key(obj.get("id", "")), _key(obj.get("name", ""))})
        ]
        if len(matches) != 1 or id(matches[0]) in claimed:
            raise ValueError(f"Cannot identify exactly one configured indicator for {ident} ({name})")
        result[ident] = matches[0]
        claimed.add(id(matches[0]))
    return result


def _stream_source(config: dict[str, Any]) -> str:
    for holder, fields in (
        (config.get("stream"), ("url", "hls_url", "uri", "source")),
        (config, ("stream_url", "hls_url", "video_url")),
        (config.get("video"), ("url", "stream_url", "source")),
    ):
        if isinstance(holder, dict):
            for field in fields:
                value = holder.get(field)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    raise ValueError("No stream URL found; use --source or set stream.url in config.json")


def _dimension_fields(node: dict[str, Any], location: str) -> list[tuple[str, str]]:
    """Find existing width/height fields without creating an unknown schema."""
    pairs = (
        ("reference_width", "reference_height"),
        ("video_reference_width", "video_reference_height"),
        ("frame_reference_width", "frame_reference_height"),
        ("reference_frame_width", "reference_frame_height"),
    )
    found = [pair for pair in pairs if pair[0] in node and pair[1] in node]
    if location and location.split(".")[-1] in {
        "video", "frame", "reference_dimensions", "reference_resolution",
        "reference_size", "video_reference", "frame_reference", "reference_frame",
    } and "width" in node and "height" in node:
        found.append(("width", "height"))
    return found


def _patch_dimensions(config: dict[str, Any], width: int, height: int) -> int:
    updated = 0

    def walk(node: Any, path: str = "") -> None:
        nonlocal updated
        if not isinstance(node, dict):
            return
        for wkey, hkey in _dimension_fields(node, path):
            node[wkey], node[hkey] = width, height
            updated += 1
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value, f"{path}.{key}" if path else key)
            elif key in {
                "reference_dimensions", "reference_resolution", "reference_size",
                "video_reference_dimensions", "frame_reference_dimensions",
            } and isinstance(value, list) and len(value) == 2:
                node[key] = [width, height]
                updated += 1
        # These flags only describe frame/ROI calibration, not brightness thresholds.
        for flag in ("reference_dimensions_calibrated", "roi_calibrated"):
            if flag in node and isinstance(node[flag], bool):
                node[flag] = True

    walk(config)
    if not updated:
        raise ValueError(
            "Cannot locate existing video reference dimensions in config.json. "
            "Expected e.g. video.reference_width/reference_height or "
            "video.reference_dimensions.width/height. No changes saved."
        )
    return updated


def _write_rect(target: dict[str, Any], field: str, rect: Rect, dimensions: tuple[int, int]) -> None:
    """Replace an existing ROI while preserving its representation and other settings."""
    if field not in target:
        raise ValueError(f"Missing '{field}' ROI field; cannot safely guess config schema")
    x, y, w, h = rect
    roi = target[field]
    normalized = "norm" in field.lower() or _key(target.get("roi_units", "")) in {
        "normalized", "relative", "fraction",
    }
    if normalized:
        frame_w, frame_h = dimensions
        a, b, c, d = x / frame_w, y / frame_h, w / frame_w, h / frame_h
    else:
        a, b, c, d = x, y, w, h

    if isinstance(roi, dict):
        names = set(roi)
        if {"x", "y", "w", "h"} <= names:
            roi.update(x=a, y=b, w=c, h=d)
        elif {"x", "y", "width", "height"} <= names:
            roi.update(x=a, y=b, width=c, height=d)
        elif {"left", "top", "right", "bottom"} <= names:
            roi.update(left=a, top=b, right=a + c, bottom=b + d)
        elif {"x1", "y1", "x2", "y2"} <= names:
            roi.update(x1=a, y1=b, x2=a + c, y2=b + d)
        elif {"cx", "cy", "radius"} <= names:
            roi.update(cx=a + c / 2, cy=b + d / 2, radius=min(c, d) / 2)
        else:
            raise ValueError(f"Unrecognized {field} format: {sorted(names)}")
    elif isinstance(roi, list) and len(roi) == 4:
        target[field] = [a, b, c, d]  # [x, y, width, height]
    elif roi is None:
        target[field] = {"x": a, "y": b, "w": c, "h": d}
    else:
        raise ValueError(f"Unsupported {field} ROI representation: {type(roi).__name__}")


def _patch_rois(config: dict[str, Any], panel: Rect, selections: dict[str, Rect], size: tuple[int, int]) -> None:
    for ident, item in _find_indicators(config).items():
        roi_field = next((name for name in ("roi", "roi_norm", "normalized_roi", "roi_normalized") if name in item), None)
        if roi_field is None:
            raise ValueError(f"Indicator {ident} has no ROI field in config.json")
        _write_rect(item, roi_field, selections[ident], size)

    if "panel_roi" in config:
        _write_rect(config, "panel_roi", panel, size)
    if isinstance(config.get("panel"), dict) and "roi" in config["panel"]:
        _write_rect(config["panel"], "roi", panel, size)


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError("config.json root must be a JSON object")
    return result


def _capture_frame(source: str):
    # FFMPEG timeouts are open-only parameters supported by recent OpenCV builds.
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10_000,
    ]
    capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
    try:
        if not capture.isOpened():
            raise RuntimeError("Cannot open configured video source")
        frame = None
        for _ in range(3):
            valid, candidate = capture.read()
            if not valid or candidate is None or candidate.size == 0:
                if frame is None:
                    raise RuntimeError("Stream opened but no decoded video frame was received")
                break
            frame = candidate
        if frame is None:
            raise RuntimeError("No usable video frames received")
        return frame
    finally:
        capture.release()


def _fit(image, max_w: int, max_h: int, upsample: bool = False):
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h)
    if not upsample:
        scale = min(1.0, scale)
    if scale == 1.0:
        return image.copy()
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)


# A chosen point is converted into a small, clipped rectangular ROI for the
# existing monitor. The GUI only draws circular markers; no panel box is needed.
Point = tuple[int, int, int]  # center_x, center_y, radius, in source-frame pixels


def _point_to_rect(point: Point, size: tuple[int, int]) -> Rect:
    """Return a bounded (x, y, w, h) pixel ROI around a marked point."""
    x, y, radius = point
    width, height = size
    if width <= 0 or height <= 0 or radius <= 0:
        raise ValueError("Invalid frame dimensions or sensor radius")
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Sensor center ({x}, {y}) is outside {width}x{height} frame")
    left = max(0, x - radius)
    top = max(0, y - radius)
    right = min(width, x + radius + 1)
    bottom = min(height, y + radius + 1)
    return left, top, right - left, bottom - top


def _panel_bounds(selections: dict[str, Rect]) -> Rect:
    """Bounding ROI containing every indicator, even widely separated ones."""
    x0 = min(x for x, y, w, h in selections.values())
    y0 = min(y for x, y, w, h in selections.values())
    x1 = max(x + w for x, y, w, h in selections.values())
    y1 = max(y + h for x, y, w, h in selections.values())
    return x0, y0, x1 - x0, y1 - y0


def _original_point(x: int, y: int, display_size: tuple[int, int],
                    frame_size: tuple[int, int]) -> tuple[int, int]:
    """Map a click on a scaled view back to original-frame coordinates."""
    dw, dh = display_size
    fw, fh = frame_size
    if not (0 <= x < dw and 0 <= y < dh):
        raise ValueError("Click was outside the image")
    return min(fw - 1, int((x + 0.5) * fw / dw)), min(fh - 1, int((y + 0.5) * fh / dh))


def _draw_markers(view, points: dict[str, Point], frame_size: tuple[int, int]):
    """Overlay numbered circles, never ROI rectangles or a panel selection."""
    result = view.copy()
    dw, dh = result.shape[1], result.shape[0]
    fw, fh = frame_size
    palette = (
        (40, 40, 255),   # red
        (40, 240, 40),   # green
        (0, 230, 255),   # yellow
        (40, 40, 255),
        (40, 40, 255),
    )
    for index, (ident, _name) in enumerate(INDICATORS):
        if ident not in points:
            continue
        x, y, radius = points[ident]
        dx = min(dw - 1, int((x + 0.5) * dw / fw))
        dy = min(dh - 1, int((y + 0.5) * dh / fh))
        dr = max(2, round(radius * min(dw / fw, dh / fh)))
        color = palette[index]
        cv2.circle(result, (dx, dy), dr, color, 2, cv2.LINE_AA)
        cv2.circle(result, (dx, dy), 2, color, -1, cv2.LINE_AA)
        cv2.putText(result, str(index + 1),
                    (min(dw - 14, dx + dr + 5), max(14, dy - dr - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
    return result


# Explicitly labeled buttons are rendered in the image itself. This avoids
# Qt's icon-only HighGUI toolbar and keeps controls visible without tooltips.
Button = tuple[str, str, Rect]  # action, visible label, (x, y, w, h)


def _toolbar_layout(display_width: int) -> tuple[int, list[Button]]:
    """Lay out all text-labeled controls, wrapping on narrower displays."""
    if display_width < 180:
        raise ValueError("Calibration image is too narrow for the on-screen controls")

    sensor_controls = [
        (f"sensor:{index}", f"{index + 1} {name}")
        for index, (_ident, name) in enumerate(INDICATORS)
    ]
    edit_controls = [
        ("undo", "Undo (U)"),
        ("clear", "Clear Sensor (D)"),
        ("reset", "Reset All (R)"),
        ("smaller", "Radius -"),
        ("larger", "Radius +"),
        ("save", "Save (S)"),
        ("quit", "Quit (Q)"),
    ]

    margin, gap, row_height, row_gap = 9, 7, 31, 6
    y = 29  # Space for an always-visible, plain-text instruction line.
    buttons: list[Button] = []
    for row in (sensor_controls, edit_controls):
        x = margin
        for action, label in row:
            text_width, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0]
            button_width = min(display_width - 2 * margin, text_width + 18)
            if x != margin and x + button_width > display_width - margin:
                x = margin
                y += row_height + row_gap
            buttons.append((action, label, (x, y, button_width, row_height)))
            x += button_width + gap
        y += row_height + row_gap
    return y + 5, buttons


def _button_at(x: int, y: int, buttons: list[Button]) -> str | None:
    """Return only a real button action; clicks elsewhere do nothing."""
    for action, _label, (bx, by, bw, bh) in buttons:
        if bx <= x < bx + bw and by <= y < by + bh:
            return action
    return None


def _draw_interface(view, points: dict[str, Point], active_index: int,
                    radius: int, frame_size: tuple[int, int]):
    """Paint labeled clickable controls above the image and status below it."""
    toolbar_height, buttons = _toolbar_layout(view.shape[1])
    status_height = 53
    canvas = cv2.copyMakeBorder(
        _draw_markers(view, points, frame_size),
        toolbar_height, status_height, 0, 0,
        cv2.BORDER_CONSTANT, value=(28, 28, 28),
    )
    cv2.putText(
        canvas, "Choose a sensor, then CLICK its center in the image to mark or move it.",
        (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.49,
        (245, 245, 245), 1, cv2.LINE_AA,
    )
    for action, label, (x, y, w, h) in buttons:
        if action == f"sensor:{active_index}":
            fill = (118, 85, 36)
        elif action.startswith("sensor:") and INDICATORS[int(action.split(':')[1])][0] in points:
            fill = (59, 108, 57)
        elif action == "save":
            fill = (60, 116, 49) if len(points) == len(INDICATORS) else (63, 63, 63)
        elif action == "reset":
            fill = (48, 69, 127)
        else:
            fill = (63, 63, 63)
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), fill, -1)
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1),
                      (165, 165, 165), 1)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        text_x = x + max(4, (w - text_size[0]) // 2)
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(canvas, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (255, 255, 255), 1, cv2.LINE_AA)

    footer_y = toolbar_height + view.shape[0]
    _ident, active_name = INDICATORS[active_index]
    messages = (
        f"Selected: {active_index + 1} {active_name}    Radius: {radius}px    "
        f"Marked: {len(points)}/5    Green buttons = marked sensors.",
        "Click sensor button to reposition | Undo reverses last click | "
        "Clear Sensor removes one | Reset All removes every point.",
    )
    for index, message in enumerate(messages):
        cv2.putText(canvas, message, (10, footer_y + 20 + index * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 245, 245),
                    1, cv2.LINE_AA)
    return canvas


def _select_points(frame, initial_radius: int) -> dict[str, Point]:
    """Place five independent centers using labeled buttons or keyboard."""
    view = _fit(frame, 1500, 830)
    frame_size = frame.shape[1], frame.shape[0]
    display_size = view.shape[1], view.shape[0]
    toolbar_height, buttons = _toolbar_layout(display_size[0])
    window = "ColorCatcher - sensor calibration"
    points: dict[str, Point] = {}
    history: list[tuple[str, Point | None]] = []
    active = 0
    radius = initial_radius
    requested_action: str | None = None

    def click(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        nonlocal active, requested_action
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if y < toolbar_height:
            requested_action = _button_at(x, y, buttons)
            return
        # The toolbar is ABOVE the image, and must not shift stored coordinates.
        image_y = y - toolbar_height
        if not (0 <= x < display_size[0] and 0 <= image_y < display_size[1]):
            return
        original_x, original_y = _original_point(x, image_y, display_size, frame_size)
        ident, name = INDICATORS[active]
        history.append((ident, points.get(ident)))
        points[ident] = (original_x, original_y, radius)
        print(f"{active + 1}. {name}: center=({original_x}, {original_y}), radius={radius}px", flush=True)
        for next_index in range(active + 1, len(INDICATORS)):
            if INDICATORS[next_index][0] not in points:
                active = next_index
                break
        else:
            for next_index, (next_ident, _) in enumerate(INDICATORS):
                if next_ident not in points:
                    active = next_index
                    break

    def perform(action: str) -> bool:
        """Apply an action. True means the completed calibration should be saved."""
        nonlocal active, radius
        if action.startswith("sensor:"):
            active = int(action.split(":", 1)[1])
            ident = INDICATORS[active][0]
            if ident in points:
                radius = points[ident][2]
        elif action == "smaller" or action == "larger":
            radius = max(1, radius - 1) if action == "smaller" else min(radius + 1, max(frame_size))
            ident = INDICATORS[active][0]
            if ident in points:
                x, y, _ = points[ident]
                points[ident] = x, y, radius
        elif action == "undo":
            if history:
                ident, previous = history.pop()
                if previous is None:
                    points.pop(ident, None)
                else:
                    points[ident] = previous
                active = next(i for i, (item_id, _) in enumerate(INDICATORS) if item_id == ident)
                radius = points[ident][2] if ident in points else initial_radius
        elif action == "clear":
            ident, name = INDICATORS[active]
            if ident in points:
                history.append((ident, points[ident]))
                del points[ident]
                print(f"Cleared selection for {active + 1}. {name}.", flush=True)
        elif action == "reset":
            points.clear()
            history.clear()
            active = 0
            radius = initial_radius
            print("All five selections cleared.", flush=True)
        elif action == "save":
            if len(points) != len(INDICATORS):
                print(f"Complete all five points before saving ({len(points)}/5).", flush=True)
            else:
                return True
        elif action == "quit":
            raise KeyboardInterrupt("Calibration cancelled")
        return False

    print("\nClick the labeled sensor button (1-5), then its CENTER anywhere in the image.")
    print("The buttons remain visible; you can reposition any sensor or clear it.")
    print("Undo (U), Clear Sensor (D), Reset All (R), Radius +/- , Save (S), Quit (Q).\n", flush=True)
    # On Qt builds, GUI_NORMAL disables the unlabeled OpenCV icon toolbar.
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    try:
        cv2.setMouseCallback(window, click)
        while True:
            cv2.imshow(window, _draw_interface(view, points, active, radius, frame_size))
            key = cv2.waitKey(30) & 0xFF
            action = requested_action
            requested_action = None
            if action is None:
                if key in (27, ord('q'), ord('Q')):
                    action = "quit"
                elif ord('1') <= key <= ord('5'):
                    action = f"sensor:{key - ord('1')}"
                else:
                    action = {
                        ord('+'): "larger", ord('='): "larger",
                        ord('-'): "smaller", ord('_'): "smaller",
                        ord('u'): "undo", ord('U'): "undo", 8: "undo",
                        ord('d'): "clear", ord('D'): "clear",
                        ord('r'): "reset", ord('R'): "reset",
                        ord('s'): "save", ord('S'): "save", 13: "save",
                    }.get(key)
            if action is not None and perform(action):
                return points.copy()
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    raise KeyboardInterrupt("Calibration window closed")
            except cv2.error as exc:
                raise KeyboardInterrupt("Calibration window closed") from exc
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


def _annotated_preview(frame, points: dict[str, Point]):
    return _draw_markers(frame, points, (frame.shape[1], frame.shape[0]))


def _write_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot save image to {path}")


def _atomic_save_config(path: Path, config: dict[str, Any]) -> None:
    # Preserve an original backup, then replace the config atomically on POSIX.
    backup = path.with_suffix(path.suffix + ".bak")
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.chmod(temp, stat.S_IMODE(path.stat().st_mode))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(config, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copy2(path, backup)
        os.replace(temp, path)
    finally:
        if fd != -1:
            os.close(fd)
        if os.path.exists(temp):
            os.unlink(temp)


def main() -> int:
    parser = argparse.ArgumentParser(description="Point-and-click calibration of five independent factory indicators")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.json")
    parser.add_argument("--source", help="Optional alternative video URL or local video filename")
    parser.add_argument("--image", type=Path, help="Calibrate entirely offline from an existing full-frame image")
    parser.add_argument("--radius", type=int, default=8,
                        help="Initial sensor-circle radius in original-frame pixels (default: 8)")
    args = parser.parse_args()
    if args.radius <= 0:
        parser.error("--radius must be a positive integer")
    config_path: Path = args.config.resolve()
    config = _load_config(config_path)

    # Refuse unsupported config layouts BEFORE opening a remote stream.
    trial = copy.deepcopy(config)
    _patch_dimensions(trial, 640, 480)
    dummy = (10, 10, 12, 12)
    _patch_rois(trial, dummy, {ident: dummy for ident, _ in INDICATORS}, (640, 480))

    if args.image:
        image_path = args.image.resolve()
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
    else:
        source = args.source or _stream_source(config)
        if not source.startswith(("http://", "https://", "rtsp://", "rtmp://")):
            candidate = Path(source).expanduser()
            if not candidate.is_absolute():
                candidate = config_path.parent / candidate
            source = str(candidate)
        print("Opening video source and capturing one frame (no continuous monitoring)...", flush=True)
        frame = _capture_frame(source)

    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Decoded frame has invalid dimensions")
    print(f"Reference frame resolution: {width} x {height}")
    snapshot = config_path.parent / "calibration_frame.jpg"
    _write_image(snapshot, frame)
    print(f"Original screenshot saved: {snapshot}")

    try:
        points = _select_points(frame, args.radius)
        selections = {
            ident: _point_to_rect(points[ident], (width, height))
            for ident, _ in INDICATORS
        }
        panel = _panel_bounds(selections)

        preview = _annotated_preview(frame, points)
        preview_path = config_path.parent / "calibration_preview.jpg"
        _write_image(preview_path, preview)
        print(f"Annotated preview saved: {preview_path}")
        for index, (ident, name) in enumerate(INDICATORS, 1):
            print(f"  {index}. {name:18s} {ident:7s} ROI {selections[ident]}")

        updated = copy.deepcopy(config)
        _patch_dimensions(updated, width, height)
        _patch_rois(updated, panel, selections, (width, height))
        _atomic_save_config(config_path, updated)
        print(f"Calibration saved to: {config_path}")
        print(f"Backup saved to: {config_path.with_suffix(config_path.suffix + '.bak')}")
        return 0
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Calibration cancelled. config.json was not modified.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, cv2.error) as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        sys.exit(1)
