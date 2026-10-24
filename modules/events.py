"""Durable local screenshots and unambiguous console event output."""

from datetime import datetime
import logging
import os
from pathlib import Path
from uuid import uuid4

import cv2

from modules.detector import Change

LOG = logging.getLogger("factory_monitor.events")


def emit(changes: list[Change], frame, when: datetime, output_dir: Path, jpeg_quality: int) -> None:
    """One screenshot per confirming frame; emit no fictitious success paths."""
    if not changes:
        return
    stamp = when.isoformat(sep=" ", timespec="milliseconds")
    day_dir = output_dir / when.strftime("%Y-%m-%d")
    safe_name = changes[0].indicator_id.lower()
    filename = f"{when:%H%M%S}_{when.microsecond // 1000:03d}_{safe_name}_{uuid4().hex[:8]}.jpg"
    dest = day_dir / filename
    try:
        day_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        # OpenCV imwrite returns a boolean; it can also throw cv2.error/OS errors.
        if not cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
            raise OSError("cv2.imwrite returned False")
        if os.name == "posix":
            dest.chmod(0o600)
    except (OSError, cv2.error) as exc:
        LOG.error("SCREENSHOT_WRITE_FAILED: %s: %s", dest, exc)
        for event in changes:
            print(f"{stamp} | {event.name} | {event.description} | SCREENSHOT_ERROR", flush=True)
        return

    for event in changes:
        print(f"{stamp} | {event.name} | {event.description} | {dest}", flush=True)
