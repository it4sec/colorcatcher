# Factory Indicator Monitor — Offline Verification Report

Scope: **Only local source code, local synthetic video, and local temporary files. No attempts to contact any factory/client server.**

## Architecture conformance

| Requirement | Implemented | Evidence |
|---|---|---|
| Single simple Python processing loop | Yes | `main.py:monitor` |
| OpenCV FFmpeg input | Yes | `modules/stream.py:StreamReader.open` |
| FPS-sampled classification | Yes | Live uses monotonic clock; offline uses video FPS |
| Five independently named detectors | Yes | `config.json` IDs; per-ROI threshold dictionaries |
| HSV color + saturation + brightness | Yes | `modules/detector.py:classify` |
| Unlit red is distinct from lit red | Algorithmic provision | Brightness gate (thresholds require real calibration) |
| Confirmation before transitions | Yes | `StateTracker.update` |
| No false transition on initial read/reconnect/UNKNOWN | Yes | UNKNOWN rebaseline; `STARTUP_MISMATCH`/`RESYNC_MISMATCH` |
| JPEG screenshot / console event | Yes | `modules/events.py:emit` |
| Report screenshot write failures | Yes | Explicit error, no success path |
| Detect loss / reconnect | Yes (software path) | FFmpeg read/open timeout, invalid-frame timeout, reconnect loop |
| Strict calibration geometry | Yes | Source-frame size and per-ROI range checks |
| No AI, DB, cloud or message queue | Yes | Imports/architecture reviewed |
| User-configurable behavior in JSON | Yes | `config.json` |
| Offline test support | Yes | `demo.avi`, `demo_config.json`, GUI calibrator, tests |

## Static checks

- All Python files compile with `python -m compileall -q .` (Python 3.13.5).
- Configuration validation ensures correct identifiers, ROI bounds within the panel and source frame, numeric threshold ranges, hysteresis ordering, FPS and timeout settings, allowed normal states, screenshot quality, and timezone.
- No client hostname, network secret, factory URL, browser session, remote write request or server-specific protocol assumptions are hardcoded.
- OpenCV URL acquisition occurs only through the user-initiated `--capture-frame` or default monitoring path.
- Source code uses no `eval`, `exec`, shell commands or background workers.

## Offline executed checks

- **10/10 `unittest` cases passed**: lit vs unlit red, green/yellow HSV, intermediate UNKNOWN, initial baseline mismatch, two-frame transition confirmation, reconnect resync, camera dimension and config checks, actual JPEG creation, JPEG failure detection, shared screenshot across events.
- Synthetic `demo.avi` (7 seconds, 30 FPS) used for **local end-to-end playback**; observed expected five transitions and a JPEG for each (real source not accessed).
- Example command for local replay: `python main.py --config demo_config.json`.

## Not validated without actual onsite inputs

1. Customer HLS authentication, redirects, HTTP headers, FFmpeg version-specific buffering behavior, and production network reconnect behavior.
2. Real 5-light ROI coordinates, source resolution, color/intensity thresholds and ambient-light impact.
3. Genuine illuminated red indicator appearance and alarm false-positive/negative rates.
4. Camera stability when frame dimensions do not change, HLS segment latency, and minimum detectable flash length.
5. End-to-end availability guarantees, since OpenCV/FFmpeg and the source server can buffer HLS segments.

**Before deployment:** calibrate ROIs using an actual decoded frame, inspect ON/OFF samples for every lamp, and run representative *offline* real-footage validation. This is visual monitoring software and not a safety-critical substitute for industrial alarms.
