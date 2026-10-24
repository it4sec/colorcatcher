"""OpenCV FFmpeg frame reader. Never contacts a network until open() is called."""

from pathlib import Path
from urllib.parse import urlsplit

import cv2


class StreamError(RuntimeError):
    """Video source could not be opened or read."""


class StreamReader:
    def __init__(self, source: str, open_timeout_seconds: float, read_timeout_seconds: float):
        self.source = source
        self.open_timeout_seconds = open_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self.capture: cv2.VideoCapture | None = None
        self.is_live = urlsplit(source).scheme.lower() in {"https", "http"}
        self.frame_number = 0
        self.source_fps = 0.0

    def open(self) -> None:
        if not self.is_live and not Path(self.source).is_file():
            raise StreamError(f"Local source file not found: {self.source}")
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.open_timeout_seconds * 1000),
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.read_timeout_seconds * 1000),
        ]
        self.close()
        capture = cv2.VideoCapture()
        try:
            if not capture.open(self.source, cv2.CAP_FFMPEG, params):
                raise StreamError("Unable to open video with FFmpeg; verify source and authorization")
            if not capture.isOpened():
                raise StreamError("FFmpeg source was not opened")
            if capture.getBackendName() != "FFMPEG":
                raise StreamError("Source did not use the required FFmpeg backend")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not self.is_live and not (0 < fps < 1000):
                raise StreamError("Local video reports invalid FPS; cannot sample on media timeline")
            self.capture = capture
            self.source_fps = fps
            self.frame_number = 0
        except Exception:
            capture.release()
            raise

    def grab(self) -> bool:
        if self.capture is None:
            raise StreamError("Capture has not been opened")
        ok = self.capture.grab()
        if ok:
            self.frame_number += 1
        return ok

    def retrieve(self):
        if self.capture is None:
            raise StreamError("Capture has not been opened")
        return self.capture.retrieve()

    def read_first_frame(self):
        if not self.grab():
            raise StreamError("No decoded frames available")
        ok, frame = self.retrieve()
        if not ok or frame is None or frame.size == 0:
            raise StreamError("First frame could not be decoded")
        return frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
