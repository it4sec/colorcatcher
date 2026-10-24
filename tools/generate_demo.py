"""Create a standalone, synthetic OFFLINE video and calibrated demo config.

This command never touches the network and never reads the client stream.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POINTS = {
    "RED_1": (69, 248),
    "GREEN": (118, 248),
    "YELLOW": (168, 248),
    "RED_2": (95, 300),
    "RED_3": (151, 300),
}


def make():
    video_path = ROOT / "demo.avi"
    config_path = ROOT / "demo_config.json"
    config = deepcopy(json.loads((ROOT / "config.json").read_text(encoding="utf-8")))
    config["stream"]["url"] = "demo.avi"
    config["video"].update({"reference_width": 640, "reference_height": 360,
                            "panel_roi": [35, 216, 185, 112]})
    config["output"]["screenshots_dir"] = "demo_screenshots"
    for indicator in config["indicators"]:
        x, y = POINTS[indicator["id"]]
        indicator["roi"] = [x - 12, y - 12, 24, 24]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (640, 360))
    if not writer.isOpened():
        raise RuntimeError("OpenCV MJPG video encoder unavailable")
    try:
        for n in range(210):
            t = n / 30.0
            state = {"RED_1": 1.2 <= t < 2.4,
                     "GREEN": not (3.2 <= t < 4.2),
                     "YELLOW": True,
                     "RED_2": t >= 5.0,
                     "RED_3": False}
            frame = np.full((360, 640, 3), (35, 40, 45), dtype=np.uint8)
            cv2.rectangle(frame, (35, 216), (220, 328), (100, 100, 100), -1)
            cv2.rectangle(frame, (35, 216), (220, 328), (175, 175, 175), 2)
            for key, (x, y) in POINTS.items():
                on = state[key]
                if key.startswith("RED_"):
                    color = (5, 5, 245) if on else (9, 9, 70)
                elif key == "GREEN":
                    color = (10, 250, 20) if on else (9, 65, 12)
                else:
                    color = (5, 240, 245) if on else (9, 67, 68)
                cv2.circle(frame, (x, y), 10, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 11, (15, 15, 15), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Offline demo created: {video_path}\nOffline configuration: {config_path}")
    print(f"Run: {sys.executable} main.py --config demo_config.json")


if __name__ == "__main__":
    make()
