# Factory Indicator Monitor — MVP 0.1

A small, local, deterministic Python 3.12+ application that reads an authorized HLS livestream and observes **five independently calibrated indicator lights**: System Failure (red), Status OK (green), In Production (yellow), Pressure Failure (red), and Machine Failure (red). It detects confirmed `ON -> OFF` and `OFF -> ON` transitions, saves the **full confirming frame** as JPEG, and prints an event with local detection time and the saved path.

No AI model, cloud service, database, outgoing event service, or web dashboard is needed. The application is read-only with respect to the video input. **No connection to the factory server occurs merely by importing or installing this project**; pip may separately download Python dependencies from package repositories. Capture/monitor modes connect to the source only when you explicitly execute them with a URL.

## Files

```text
factory_monitor/
├── main.py                    # CLI, configuration validation, main monitoring loop
├── config.json                # Real stream configuration (URL blank, ROIs uncalibrated)
├── demo_config.json           # Calibrated portable local-only demo configuration
├── demo.avi                   # Included synthetic video (7 seconds, 30 FPS)
├── modules/
│   ├── stream.py              # OpenCV FFmpeg stream reader
│   ├── detector.py            # HSV light detection and confirmed state machine
│   └── events.py              # JPEG screenshots and console events
├── tools/generate_demo.py     # Regenerate synthetic demo video/config locally
├── tests/test_detector.py     # Offline unit tests
├── screenshots/               # Real event screenshots (created at runtime)
├── requirements.txt
└── README.md
```

## 1. Installation (Ubuntu / Python 3.12+)

From inside the extracted `factory_monitor/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Requires a working OpenCV Python build with the FFmpeg backend. Official OpenCV pip wheels provide FFmpeg functionality for common supported platforms. The standalone `ffmpeg` executable is **not** invoked by the program. GUI calibration needs a desktop/GUI OpenCV installation.

## 2. First test: entirely offline

The ZIP **already includes** `demo.avi` and `demo_config.json` (no network and no credentials). You can regenerate them with `python tools/generate_demo.py`.

```bash
python main.py --config demo_config.json --check-config
python main.py --config demo_config.json
```

Expected event sequences (one JPEG per event under `demo_screenshots/YYYY-MM-DD/`):

```text
System Failure   OFF -> ON
System Failure   ON -> OFF
Status OK        ON -> OFF
Status OK        OFF -> ON
Pressure Failure OFF -> ON
```

The timestamps represent **when the program detected the change**, not the synthetic video's timestamps; replay of a local video is intentionally unthrottled. This checks the application logic but **does not** establish accuracy on actual camera footage.

Run included no-network unit tests:

```bash
python -m unittest discover -s tests -v
```

## 3. Capture a real calibration frame on YOUR machine

Do this **only when you are authorized** to connect to the factory HLS endpoint. Put the actual authorized `.m3u8` URL in `config.json` under `stream.url` (initially empty). Signed URLs with query parameters are supported. Alternatively pass `--source URL`; command-line URL arguments may be visible in process listings, so config file storage with restricted access is preferable.

```bash
chmod 600 config.json
python main.py --capture-frame --output calibration_frame.jpg
```

This opens the configured source, decodes the **first video frame**, saves it, and closes the source. It **will contact the endpoint** when YOU run it. No URL is bundled or guessed. If the HLS content requires browser session cookies, request headers, or rotating authentication that is not embedded in an authorized URL, this MVP's plain OpenCV URL ingestion may not be sufficient and must be extended based on the actual stream authentication mechanism.

For a pre-recorded **local** image or video, no network access is necessary:

```bash
python main.py --capture-frame --source /path/to/local_video.mp4 --output calibration_frame.jpg
```

## 4. Calibrate five indicator ROIs (offline)

```bash
python main.py --calibrate-image calibration_frame.jpg
```

An OpenCV GUI prompts for the panel rectangle, then displays that panel and asks for each indicator ROI in this exact order:

1. System Failure (upper red)
2. Status OK (green)
3. In Production (yellow)
4. Pressure Failure (lower red #1)
5. Machine Failure (lower red #2)

Drag around each **individual lamp** without adjacent lamps. Press **SPACE** or **ENTER** to accept each rectangle; press **C** to cancel. The final configuration stores all ROIs as full-source-frame pixel coordinates and stores actual source width/height. If you cancel, the configuration is not overwritten.

**Do not use coordinates copied from the webpage/browser screenshot.** Crop/rescaling in the webpage can shift them relative to the decoded HLS video. Camera movement or resolution changes require recalibration.

Check geometry, types, thresholds, and complete calibration locally:

```bash
python main.py --check-config
```

## 5. Inspect and tune detection (offline)

```bash
python main.py --inspect-image calibration_frame.jpg --output inspection.jpg
```

For each lamp this prints `ON`, `OFF` or `UNKNOWN`, `lit_fraction`, `mean_V`, and `max_V`. It saves an annotated frame with the ROIs. `lit_fraction` is the fraction of the lamp's **inner ellipse** whose pixels match **HSV hue + saturation + value** requirements.

Tune **each indicator's parameters** in `config.json` using representative frames of both ON and OFF conditions:

- `hue_ranges`: accepted OpenCV HSV hue ranges (0–179); red uses two ranges across 0/179.
- `min_saturation`: minimum HSV S to reject gray/white background.
- `min_brightness`: minimum HSV V to avoid confusing a dark **red lens** with an illuminated red lamp.
- `off_fraction`: below or equal -> `OFF`.
- `on_fraction`: above or equal -> `ON`.
- Between those two fractions -> `UNKNOWN` (no transition is emitted).
- `video.min_panel_mean_brightness`: minimum mean brightness of the *whole panel*; below this means camera blackout / obscuration, **not five lamps going OFF**.

The default thresholds are **starting guesses only**; the project cannot validate actual optical detection without real ON/OFF reference frames, especially a verified **illuminated red lamp** frame. Run `--inspect-image` separately on known ON and OFF examples for each indicator and select a safe hysteresis gap.

## 6. Start monitoring the authorized HLS stream

Once all five ROIs and illumination thresholds are calibrated:

```bash
python main.py --check-config
python main.py
```

Stop with `Ctrl+C`. An event example:

```text
2026-10-23 14:31:24.235+03:00 | System Failure | OFF -> ON | /path/to/factory_monitor/screenshots/2026-10-23/143124_235_red_1_f36e20bd.jpg
```

A camera starts with state `UNKNOWN`, establishes current states after two accepted observations and compares with the configured normal states. An abnormal initial state yields `STARTUP_MISMATCH`, **not a fabricated switch**. After loss of video, frame blackout, an ambiguous lamp, or reconnection, state history is invalidated; an abnormal re-established state yields `RESYNC_MISMATCH`, not an invented transition spanning the missing data. Monitoring problems are printed as separately labeled **log diagnostics**. A successful screenshot path is printed **only after the JPEG was written**.

Default config: **5 FPS** detection sampling, **2 consecutive frames** to confirm changes, **10-second** FFmpeg open/read timeouts, **3-second** reconnect delay, **90% JPEG quality**, and local machine timezone including its UTC offset. Set `output.timezone` to an IANA name such as `Asia/Jerusalem` if your OS timezone differs from the required local time. On Windows, the included `tzdata` wheel helps supply timezone data.

## Operational behavior and limitations

- Reads/grabs **every frame** to reduce backend backlog and only retrieves sampled frames for color analysis. HLS segment buffering and network latency are outside this program's control; **5 FPS classification is not a guaranteed 200 ms end-to-end alarm latency**.
- Uses OpenCV's documented FFmpeg `CAP_PROP_OPEN_TIMEOUT_MSEC` and `CAP_PROP_READ_TIMEOUT_MSEC` set at capture open. Actual timeout behavior depends on the installed OpenCV/FFmpeg build.
- Detects source frame-size changes and fully dark panels. **Camera movement with unchanged resolution, partial occlusion, reflections and exposure changes may still be undetected**; verify stable camera mounting.
- Frame loss, blackout or ambiguous observations are **not** interpreted as OFF. Unconfirmed states/changes are deliberately suppressed. Transient flashes shorter than the effective sample/confirmation interval may be missed.
- Local recording playback uses its media FPS for sampling, not wall-time decode rate; existing files end at EOF without retry. Live HLS sources reconnect indefinitely, with state reset.
- Screenshots contain original complete camera frames, potentially sensitive factory footage. On POSIX, new JPEGs and calibration outputs are made mode `0600`; protect config and the output directory and establish your own retention policy.
- Startup and resync mismatches save screenshots like change events. A failed screenshot is explicitly reported as `SCREENSHOT_ERROR`, never with a misleading output path.
- The program does **not** control equipment and is **not** a safety-certified industrial alarm or substitute for OEM pressure alarms.

## Offline verification performed for this release

- Python bytecode compilation/AST parsing of the project files.
- Configuration consistency and expected indicator ID checks.
- Ten offline `unittest` tests of hue/brightness differentiation, confirmation, startup/resync, invalid dimensions, and local JPEG writing.
- End-to-end playback of the bundled locally generated AVI: **five expected transitions and five JPEGs**.

**Not performed:** No factory server connections, no tests against the customer webpage/HLS stream, no real optical threshold validation, no real disconnection/reconnect timing tests. All of these require explicit operator execution and/or representative footage.

## Official technical references

- [OpenCV Video I/O flags / FFmpeg read & open timeout properties](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html)
- [OpenCV HSV and `inRange` color thresholding](https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html)
- [OpenCV `selectROI` documentation](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html)
- [FFmpeg HLS demuxer documentation](https://ffmpeg.org/ffmpeg-all.html#hls-2)
- [Python 3.12 `compileall`](https://docs.python.org/3.12/library/compileall.html)
- [Python `zoneinfo`](https://docs.python.org/3/library/zoneinfo.html)
- [Python `logging`](https://docs.python.org/3/library/logging.html)

The project deliberately follows the attached MVP architecture (one sequential loop, small modules, local JSON config, OpenCV/FFmpeg + standard Python), with **two usability additions**: offline GUI ROI calibration, and reproducible synthetic local-video testing.
