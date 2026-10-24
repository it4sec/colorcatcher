# ColorCatcher

**ColorCatcher** is a lightweight Python application that monitors an authorized industrial livestream and detects changes in the illuminated status indicators of a fixed control panel. Each confirmed `ON → OFF` or `OFF → ON` transition is reported in the console with a local timestamp and a saved screenshot.

## What It Monitors

| Indicator | Color | Expected normal state |
| --- | --- | --- |
| Status OK | Green | ON |
| In Production | Yellow | ON |
| System Failure | Red | OFF |
| Pressure Failure | Red | OFF |
| Machine Failure | Red | OFF |

All five indicators are tracked independently.

## Minimal Architecture

```text
Authorized HLS stream (.m3u8)
             |
             v
    OpenCV / FFmpeg reader
             |
             v
   Configured indicator ROIs
             |
             v
    HSV color / brightness
        classification
             |
             v
   Confirmed state changes
             |
             v
   Console event + JPEG screenshot
```

- **`main.py`** — loads configuration, processes the video feed, detects confirmed indicator-state changes, and reports events.
- **`calibrate.py`** — provides a graphical tool for positioning the five indicator regions on a source-resolution frame; supports offline calibration from an image.
- **`config.json`** — holds the stream source, indicator positions and thresholds, detection settings, and output preferences.
- **`screenshots/`** — stores local JPEG evidence of detected events.

Processing is local and uses a simple video-analysis pipeline; no AI model, cloud processing service, or database is required.

## Technical Stack

| Technology | Purpose |
| --- | --- |
| Python 3.12+ | Application runtime |
| OpenCV (`opencv-python`) | Video frames, image processing, calibration GUI |
| FFmpeg backend | HLS stream decoding |
| JSON | Runtime configuration |
| Python standard library | State handling, logging, and file operations |
| JPEG | Event screenshots |
| Poetry | Python dependency management |

## Quick Start

1. Install the project dependencies:

   ```bash
   poetry install
   ```

2. Set the **authorized** video source and other settings in `config.json`.

3. Calibrate the five indicators, either from a local full-frame image (no network access):

   ```bash
   poetry run python calibrate.py --image /path/to/frame.jpg
   ```

   Or interactively using the configured video source:

   ```bash
   poetry run python calibrate.py
   ```

4. Start monitoring:

   ```bash
   poetry run python main.py
   ```

Changes are reported with the indicator name, state transition, local detection time, and screenshot path. Video buffering and change confirmation can introduce detection latency.

> **Note:** ColorCatcher assumes a stationary camera and requires calibration for the actual video resolution. It is an observational tool, not a safety-certified monitoring or alarm system.

## Credits

- **Denis Laskov** — project concept, requirements, and architecture.
- **ChatGPT Coding agent (OpenAI)** — AI-assisted implementation and development.
