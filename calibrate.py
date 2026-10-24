"""GUI calibration using one full-frame brightness baseline for each sensor.

Offline when --image is provided. Without --image, an operator explicitly
requests one frame from their configured source.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import cv2
import numpy as np

from modules.detector import brightness_score
from modules.stream import StreamReader


class Cancelled(Exception):
    """User closed the calibration UI without saving."""


def reconcile_count(config: dict, *, confirm_delete: bool = False) -> dict:
    """Increase config sensor list without dropping existing identities/settings.

    Requires an affirmative operator confirmation before removing entries.
    """
    from main import _number
    count = config["sensor_count"]
    _number(count, "sensor_count", 1, 100_000, integer=True)
    entries = config["indicators"]
    if not isinstance(entries, list):
        raise ValueError("indicators must be a list")
    existing = {item["id"] for item in entries}
    if len(entries) > count:
        if not confirm_delete:
            removed = ", ".join(item["id"] for item in entries[count:])
            choice = input(f"Delete sensor configurations [{removed}]? Type DELETE to confirm: ")
            if choice != "DELETE":
                raise Cancelled("Sensor deletion not authorized")
        del entries[count:]
    sequence = 1
    while len(entries) < count:
        while f"SENSOR_{sequence:03d}" in existing:
            sequence += 1
        identifier = f"SENSOR_{sequence:03d}"
        existing.add(identifier)
        entries.append({"id": identifier, "name": identifier, "normal_state": "OFF",
                        "roi": None, "reference_roi": None, "baseline_brightness": None,
                        "baseline_reference_brightness": None, "baseline_state": "OFF"})
    return config



def _invalidate(sensor: dict) -> None:
    sensor["baseline_brightness"] = None
    sensor["baseline_reference_brightness"] = None


def capture_baseline(sensor: dict, frame: np.ndarray) -> float:
    """Record the arithmetic mean of the marked area's grayscale pixels."""
    brightness, reference = brightness_score(frame, sensor)
    sensor.setdefault("baseline_state", sensor["normal_state"])
    sensor["baseline_brightness"] = round(brightness, 4)
    sensor["baseline_reference_brightness"] = (round(reference, 4)
                                               if reference is not None else None)
    return brightness


def _save_atomic(path: Path, data: dict) -> None:
    """Backup old configuration; fsync and atomically replace on same filesystem."""
    from main import validate_config
    validate_config(data, operational=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        if os.name == "posix":
            os.chmod(temp_path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = -1
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        os.replace(temp_path, path)
    finally:
        if fd != -1:
            os.close(fd)
        Path(temp_path).unlink(missing_ok=True)


def _images(args: argparse.Namespace, config: dict) -> list[np.ndarray]:
    paths = args.image or []
    if paths:
        frames = []
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Cannot open local image: {path}")
            frames.append(frame)
        return frames
    source = args.source or config["stream"]["url"]
    if not source:
        raise ValueError("Specify --image FILE [FILE ...] or an authorized --source")
    with StreamReader(source, config["stream"]["open_timeout_seconds"],
                      config["stream"]["read_timeout_seconds"]) as reader:
        return [reader.read_first_frame()]


class Calibrator:
    """OpenCV GUI retaining sensor buttons, click-to-place, undo, clear and radius tools."""

    def __init__(self, config: dict, frames: list[np.ndarray], config_path: Path, radius: int):
        self.config, self.frames, self.path = config, frames, config_path
        self.active = 0
        self.page = 0
        self.frame_index = 0
        self.radius = radius
        self.history: list[dict] = []
        self.message = "Select a sensor, click its CENTER. Save captures all missing brightness baselines."
        self.reference_mode = False
        self.reference_start: tuple[int, int] | None = None
        self.action: str | None = None
        self.window = "ColorCatcher - sensor calibration"
        self.view_max = (1450, 810)
        self.frame_size = (frames[0].shape[1], frames[0].shape[0])
        for frame in frames:
            if (frame.shape[1], frame.shape[0]) != self.frame_size:
                raise ValueError("Calibration images must have identical frame dimensions")
        original_size = (config["video"]["reference_width"],
                         config["video"]["reference_height"])
        if original_size != self.frame_size:
            for sensor in config["indicators"]:
                sensor["roi"] = None
                sensor["reference_roi"] = None
                _invalidate(sensor)
            self.message = "Video resolution changed: all sensor ROIs and thresholds require recalibration."
        config["video"]["reference_width"], config["video"]["reference_height"] = self.frame_size
        self.page_size = 8
        self.buttons: list[tuple[str, tuple[int, int, int, int]]] = []
        self.toolbar_height = 0
        self._view_size()

    @property
    def sensors(self) -> list[dict]:
        return self.config["indicators"]

    def _view_size(self) -> None:
        width, height = self.frame_size
        scale = min(1.0, self.view_max[0]/width, self.view_max[1]/height)
        self.display_size = (max(1, round(width*scale)), max(1, round(height*scale)))

    def _to_frame(self, x: int, y: int) -> tuple[int, int]:
        dw, dh = self.display_size
        fw, fh = self.frame_size
        return min(fw-1, int((x+0.5)*fw/dw)), min(fh-1, int((y+0.5)*fh/dh))

    def _snapshot(self) -> None:
        self.history.append({"indicators": copy.deepcopy(self.sensors)})
        if len(self.history) > 50:
            self.history.pop(0)

    def _buttons(self) -> tuple[int, list[tuple[str, tuple[int, int, int, int]]]]:
        width = self.display_size[0]
        buttons: list[tuple[str, tuple[int, int, int, int]]] = []
        rows: list[list[tuple[str, str]]] = []
        begin = self.page*self.page_size
        current = [(f"sensor:{i}", f"{i+1}:{s['id'][:14]}")
                   for i, s in enumerate(self.sensors) if begin <= i < begin+self.page_size]
        rows.append(current)
        rows.append([("page_prev", "< Sensors"), ("page_next", "Sensors >"),
                     ("frame_prev", "< Frame"), ("frame_next", "Frame >")])
        rows.append([("smaller", "Radius -"), ("larger", "Radius +"),
                     ("reference", "Ref ROI"), ("clear_ref", "Clear Ref"),
                     ("capture_baseline", "Set Baseline")])
        rows.append([("toggle_state", "Toggle ON/OFF"), ("undo", "Undo"), ("clear", "Clear Sensor"),
                     ("reset", "Reset All"), ("save", "Save"), ("quit", "Quit")])
        y = 31
        for row in rows:
            x = 8
            for action, label in row:
                btn_width = min(150, max(66, 10+len(label)*9))
                if x + btn_width > width:
                    x, y = 8, y+35
                buttons.append((action, (x, y, btn_width, 27)))
                x += btn_width+6
            y += 35
        return y+5, buttons

    def _mouse(self, event: int, x: int, y: int, flags: int, _: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and y < self.toolbar_height:
            self.action = next((a for a, (bx, by, bw, bh) in self.buttons
                                if bx <= x < bx+bw and by <= y < by+bh), None)
            return
        image_y = y - self.toolbar_height
        if not (0 <= x < self.display_size[0] and 0 <= image_y < self.display_size[1]):
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.reference_mode:
                self.reference_start = self._to_frame(x, image_y)
            else:
                sensor = self.sensors[self.active]
                fw, fh = self.frame_size
                px, py = self._to_frame(x, image_y)
                r = self.radius
                x0, y0 = max(0, px-r), max(0, py-r)
                rect = [x0, y0, min(fw, px+r+1)-x0, min(fh, py+r+1)-y0]
                self._snapshot()
                sensor["roi"] = rect
                capture_baseline(sensor, self.frames[self.frame_index])
                self.message = (f"{sensor['id']}: baseline mean "
                                f"{sensor['baseline_brightness']:.2f} captured.")
        elif event == cv2.EVENT_LBUTTONUP and self.reference_mode and self.reference_start:
            px, py = self._to_frame(x, image_y)
            x0, y0 = self.reference_start
            rect = [min(x0, px), min(y0, py), abs(px-x0)+1, abs(py-y0)+1]
            self.reference_start = None
            self.reference_mode = False
            if rect[2] < 4 or rect[3] < 4:
                self.message = "Reference ROI must be at least 4x4 pixels."
                return
            self._snapshot()
            sensor = self.sensors[self.active]
            sensor["reference_roi"] = rect
            capture_baseline(sensor, self.frames[self.frame_index])
            self.message = f"Ambient reference saved for {sensor['id']}; baseline updated."

    def _capture(self) -> None:
        sensor = self.sensors[self.active]
        if sensor["roi"] is None:
            self.message = f"Position {sensor['id']} before capturing a baseline."
            return
        self._snapshot()
        measured = capture_baseline(sensor, self.frames[self.frame_index])
        self.message = f"{sensor['id']}: baseline = {measured:.2f} (mean gray level)"

    def _action(self, action: str) -> bool:
        if action.startswith("sensor:"):
            self.active = int(action.partition(":")[2])
            self.page = self.active//self.page_size
            roi = self.sensors[self.active].get("roi")
            if roi:
                self.radius = max(2, (max(roi[2], roi[3])-1)//2)
        elif action in ("page_prev", "page_next"):
            pages = (len(self.sensors)-1)//self.page_size+1
            self.page = (self.page + (-1 if action == "page_prev" else 1)) % pages
        elif action in ("frame_prev", "frame_next"):
            self.frame_index = (self.frame_index + (-1 if action == "frame_prev" else 1)) % len(self.frames)
        elif action in ("smaller", "larger"):
            self.radius = max(2, min(200, self.radius+(-1 if action == "smaller" else 1)))
            sensor = self.sensors[self.active]
            if sensor.get("roi"):
                x, y, w, h = sensor["roi"]
                cx, cy = x+w//2, y+h//2
                fw, fh = self.frame_size
                self._snapshot()
                left, top = max(0, cx-self.radius), max(0, cy-self.radius)
                sensor["roi"] = [left, top, min(fw, cx+self.radius+1)-left,
                                 min(fh, cy+self.radius+1)-top]
                capture_baseline(sensor, self.frames[self.frame_index])
        elif action == "reference":
            self.reference_mode = True
            self.message = "Drag a rectangle on a stable DARK background (not an indicator)."
        elif action == "clear_ref":
            self._snapshot()
            self.sensors[self.active]["reference_roi"] = None
            if self.sensors[self.active]["roi"] is not None:
                capture_baseline(self.sensors[self.active], self.frames[self.frame_index])
        elif action == "capture_baseline":
            self._capture()
        elif action == "toggle_state":
            self._snapshot()
            sensor = self.sensors[self.active]
            sensor["baseline_state"] = "OFF" if sensor.get("baseline_state", sensor["normal_state"]) == "ON" else "ON"
            self.message = f"{sensor['id']}: baseline frame shows {sensor['baseline_state']} (operator-selected)"
        elif action == "undo":
            if self.history:
                self.config["indicators"] = self.history.pop()["indicators"]
        elif action == "clear":
            self._snapshot()
            self.sensors[self.active]["roi"] = None
            _invalidate(self.sensors[self.active])
        elif action == "reset":
            self._snapshot()
            for sensor in self.sensors:
                sensor["roi"] = None
                sensor["reference_roi"] = None
                _invalidate(sensor)
        elif action == "save":
            missing = [s["id"] for s in self.sensors if s["roi"] is None]
            if missing:
                self.message = f"Cannot save: position sensor(s): {', '.join(missing[:5])}"
                return False
            # Save automatically measures sensors without a baseline from the displayed frame.
            # Existing saved baselines remain stable until expressly replaced or ROI moved.
            for sensor in self.sensors:
                if sensor.get("baseline_brightness") is None or (sensor.get("reference_roi") is not None and sensor.get("baseline_reference_brightness") is None):
                    capture_baseline(sensor, self.frames[self.frame_index])
            # Panel ROI is optional visual reference, no containment requirement.
            x0 = min(s["roi"][0] for s in self.sensors)
            y0 = min(s["roi"][1] for s in self.sensors)
            x1 = max(s["roi"][0]+s["roi"][2] for s in self.sensors)
            y1 = max(s["roi"][1]+s["roi"][3] for s in self.sensors)
            self.config["panel_roi"] = [x0, y0, x1-x0, y1-y0]
            _save_atomic(self.path, self.config)
            calibrated = sum(s.get("baseline_brightness") is not None for s in self.sensors)
            print(f"Saved {len(self.sensors)} sensor ROIs to {self.path}; brightness baselines: {calibrated}/{len(self.sensors)}")
            print("Confirm each sensor's baseline_state matches its actual illumination in the captured frame.")
            return True
        elif action == "quit":
            raise Cancelled("Calibration cancelled; configuration unchanged")
        return False

    def _draw(self) -> np.ndarray:
        frame = self.frames[self.frame_index]
        dw, dh = self.display_size
        view = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        canvas = view.copy()
        fw, fh = self.frame_size
        for idx, sensor in enumerate(self.sensors):
            rect = sensor.get("roi")
            if rect:
                x, y, w, h = rect
                px, py = round(x*dw/fw), round(y*dh/fh)
                qx, qy = round((x+w)*dw/fw), round((y+h)*dh/fh)
                color = (0, 230, 255) if idx == self.active else (30, 210, 30)
                cv2.rectangle(canvas, (px, py), (qx, qy), color, 1 if idx != self.active else 2)
                cv2.putText(canvas, str(idx+1), (px, max(13, py-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
            if idx == self.active and sensor.get("reference_roi"):
                x, y, w, h = sensor["reference_roi"]
                cv2.rectangle(canvas, (round(x*dw/fw), round(y*dh/fh)),
                              (round((x+w)*dw/fw), round((y+h)*dh/fh)), (255, 160, 20), 2)
        self.toolbar_height, self.buttons = self._buttons()
        surface = cv2.copyMakeBorder(canvas, self.toolbar_height, 85, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(surface, "Select sensor -> click center | Set baseline (B) | Save",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1)
        for action, (x, y, w, h) in self.buttons:
            label = (f"{int(action.split(':')[1])+1}:{self.sensors[int(action.split(':')[1])]['id'][:14]}"
                     if action.startswith("sensor:") else {
                         "page_prev": "< Sensors", "page_next": "Sensors >", "frame_prev": "< Frame",
                         "frame_next": "Frame >", "smaller": "Radius -", "larger": "Radius +",
                         "reference": "Ref ROI", "clear_ref": "Clear Ref", "capture_baseline": "Set Baseline",
                         "undo": "Undo", "clear": "Clear Sensor",
                         "toggle_state": "Toggle ON/OFF", "reset": "Reset All", "save": "Save", "quit": "Quit"}[action])
            color = ((100, 90, 40) if action == f"sensor:{self.active}"
                     else (58, 115, 49) if action == "save"
                     else (65, 65, 65))
            cv2.rectangle(surface, (x,y), (x+w,y+h), color, -1)
            cv2.rectangle(surface, (x,y), (x+w,y+h), (175, 175, 175), 1)
            cv2.putText(surface, label, (x+3, y+19), cv2.FONT_HERSHEY_SIMPLEX,
                        .36 if len(label) > 14 else .43, (255, 255, 255), 1)
        sensor = self.sensors[self.active]
        footer = self.toolbar_height+dh
        summary = (f"Selected {self.active+1}/{len(self.sensors)} {sensor['id']} "
                   f"| Frame {self.frame_index+1}/{len(self.frames)} | Radius {self.radius}px "
                   f"| Baseline {sensor.get('baseline_brightness')} {sensor.get('baseline_state', sensor['normal_state'])}")
        cv2.putText(surface, summary[:125], (8, footer+22), cv2.FONT_HERSHEY_SIMPLEX,
                    .48, (245,245,245), 1)
        cv2.putText(surface, self.message[:150], (8, footer+44), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (180,225,245), 1)
        cv2.putText(surface, "Keys: S save, T toggle ON/OFF at baseline, Q quit, U undo, D clear, R reset, B baseline, A ambient ROI",
                    (8, footer+69), cv2.FONT_HERSHEY_SIMPLEX, .38, (210,210,210), 1)
        return surface

    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.window, self._mouse)
        try:
            while True:
                cv2.imshow(self.window, self._draw())
                key = cv2.waitKey(30) & 0xff
                action, self.action = self.action, None
                if action is None:
                    action = {27:"quit", ord('q'):"quit", ord('Q'):"quit",
                              ord('s'):"save", ord('S'):"save", 13:"save",
                              ord('u'):"undo", ord('U'):"undo", 8:"undo",
                              ord('d'):"clear", ord('D'):"clear",
                              ord('r'):"reset", ord('R'):"reset",
                              ord('+'):"larger", ord('='):"larger", ord('-'):"smaller",
                              ord('t'):"toggle_state", ord('T'):"toggle_state",
                              ord('b'):"capture_baseline", ord('B'):"capture_baseline",
                              ord('a'):"reference", ord('A'):"reference",
                              ord('n'):"frame_next", ord('p'):"frame_prev",
                              ord(']'):"page_next", ord('['):"page_prev"}.get(key)
                    if ord('1') <= key <= ord('8'):
                        index = self.page*self.page_size + key-ord('1')
                        if index < len(self.sensors):
                            action = f"sensor:{index}"
                if action and self._action(action):
                    return
                if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    raise Cancelled("Calibration window closed")
        finally:
            cv2.destroyWindow(self.window)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ColorCatcher baseline brightness calibration; GUI, 1..N sensors")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--image", nargs="+", type=Path,
                        help="One or more offline full-resolution calibration images")
    parser.add_argument("--source", help="Authorized video URL or local video file")
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--confirm-delete", action="store_true",
                        help="Explicitly authorize removing sensors exceeding sensor_count")
    args = parser.parse_args(argv)
    try:
        if args.radius < 2:
            raise ValueError("--radius must be >= 2")
        path = args.config.resolve()
        with path.open(encoding="utf-8") as source:
            config = json.load(source)
        reconcile_count(config, confirm_delete=args.confirm_delete)
        from main import validate_config
        validate_config(config, operational=False)
        frames = _images(args, config)
        Calibrator(config, frames, path, args.radius).run()
        return 0
    except Cancelled as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (OSError, KeyError, TypeError, ValueError, cv2.error) as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
