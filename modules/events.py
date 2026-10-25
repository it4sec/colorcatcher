"""Atomic annotated screenshot evidence for each confirmed sensor event."""
from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4

import cv2
import numpy as np

from modules.detector import Change, crop_roi

LOG = logging.getLogger("colorcatcher.events")


def _safe_id(identifier: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", identifier)[:70]


def _persist_jpeg(path: Path, frame: np.ndarray, quality: int, retries: int = 2) -> None:
    """Encode, write, fsync and atomically replace. Never announce missing evidence."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok or encoded.size == 0:
        raise OSError("OpenCV JPEG encoding failed")
    data = encoded.tobytes()
    last_error: OSError | None = None
    for _ in range(retries):
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".evidence_", suffix=".tmp",
                                             delete=False) as handle:
                temporary = handle.name
                if os.name == "posix":
                    os.fchmod(handle.fileno(), 0o600)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            if not path.is_file() or path.stat().st_size == 0:
                raise OSError(f"Screenshot destination missing or empty: {path}")
            return
        except OSError as exc:
            last_error = exc
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)
    raise OSError(f"Screenshot persistence failed: {last_error}")


def _preview(frame: np.ndarray, event: Change) -> np.ndarray:
    """Annotated full frame and enlarged confirmed/previous ROI crops."""
    annotated = frame.copy()
    x, y, w, h = event.roi
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(annotated, event.indicator_id, (max(0, x-3), max(20, y-8)),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
    tile_side = 150
    current = cv2.resize(crop_roi(frame, event.roi), (tile_side, tile_side),
                         interpolation=cv2.INTER_NEAREST)
    previous = (cv2.resize(event.previous_crop, (tile_side, tile_side),
                           interpolation=cv2.INTER_NEAREST)
                if event.previous_crop is not None else np.zeros_like(current))
    strip = np.full((242, frame.shape[1], 3), 38, dtype=np.uint8)
    for label, image, left in (("PREVIOUS" if event.previous_crop is not None else "NO PREVIOUS", previous, 6),
                                ("CONFIRMED", current, 164)):
        if left + tile_side <= strip.shape[1]:
            strip[35:35+tile_side, left:left+tile_side] = image
            cv2.putText(strip, label, (left, 27), cv2.FONT_HERSHEY_SIMPLEX,
                        .47, (255, 255, 255), 1)
    # Show the actual configured name as well as the ID and observed change.
    # Scale each line to fit the frame width instead of silently truncating it.
    for label, baseline in ((f"{event.name} ({event.indicator_id})", 205),
                            (event.description, 233)):
        scale = 0.5
        while scale > 0.15 and cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                scale, 1)[0][0] > strip.shape[1] - 12:
            scale -= 0.02
        cv2.putText(strip, label, (6, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 1)
    return np.vstack((annotated, strip))


def emit(changes: list[Change], frame: np.ndarray, when: datetime, output_dir: Path,
         jpeg_quality: int, retries: int) -> bool:
    """Persist one annotated preview per event; False means evidence is intact."""
    degraded = False
    day_dir = output_dir / when.strftime("%Y-%m-%d")
    stamp = when.isoformat(sep=" ", timespec="milliseconds")
    for event in changes:
        safe = _safe_id(event.indicator_id)
        direction = event.description.split(" (")[0].replace(" -> ", "-").replace(" ", "_")
        direction = _safe_id(direction)
        unique = uuid4().hex[:12]
        name = f"{when:%H%M%S}_{when.microsecond:06d}_{safe}_{direction}_{unique}_preview.jpg"
        dest = day_dir / name
        try:
            _persist_jpeg(dest, _preview(frame, event), jpeg_quality, retries)
        except (OSError, cv2.error, ValueError) as exc:
            degraded = True
            LOG.error("SCREENSHOT_WRITE_FAILED: %s: %s", event.indicator_id, exc)
            print(f"{stamp} | {event.name} | {event.description} | SCREENSHOT_MISSING", flush=True)
            continue
        print(f"{stamp} | {event.name} | {event.description} | {dest}", flush=True)
    return degraded
