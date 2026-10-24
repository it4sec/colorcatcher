# ColorCatcher — Baseline Brightness Monitoring (MVP 0.2)

ColorCatcher monitors an authorized HLS stream or a local video recording for **mean-brightness changes** in any configured number of small indicator regions. It uses one OpenCV/FFmpeg video reader, one processing loop, and no AI or cloud analysis. It saves independent screenshot evidence whenever a sensor's confirmed state changes.

## What's changed

A single calibration frame is sufficient for each sensor; no separate verified ON and OFF image pair is required.

1. Select the indicator ROI with the calibration GUI.
2. The GUI captures its **mean grayscale brightness** (all ROI pixels, not a percentile, ellipse mask, or color filter) as the baseline.
3. Set the **baseline state** (ON/OFF) to what the indicator actually shows in the chosen frame. The initial assumption is `normal_state`; press `T` to switch it.
4. Runtime computes the same mean brightness, subtracts the saved baseline and calculates a percentage difference.
5. If the **absolute** deviation is **strictly greater than `detection.brightness_change_percent` (default 10%)**, the reading is classified as **different from baseline**; otherwise it is at the baseline state. A deviation in either direction counts.
6. The independent state tracker confirms a change using consecutive frames, then saves and reports an image. Returning inside ±10% of the baseline confirms the reverse change.

**Formula** (without ambient reference):

```
calibration_mean = mean(gray(sensor ROI))
current_mean     = mean(gray(sensor ROI))
delta_percent    = 100 * (current_mean - calibration_mean) / max(calibration_mean, 1)
changed          = abs(delta_percent) > brightness_change_percent
state            = opposite(baseline_state) if changed else baseline_state
```

For example, calibration mean 100 and current mean 111 gives +11%: changed; 89 gives −11%: changed; 110 is exactly +10% and does **not** trigger. A low baseline (<~10 grayscale units) can make small fluctuations disproportionately large, so verify reliability with a video recording.

### Important interpretation limitation

**Deviation is not proof of physical ON/OFF.** Because the algorithm intentionally accepts deviations in *either* direction, a light becoming darker than an already-OFF baseline can be reported as an inferred `OFF -> ON` change even if it did not actually turn on. `baseline_state` is a required **operator-provided label** of the captured baseline state; this does not magically distinguish ON from OFF. Brightness variation, reflections, camera auto-exposure, motion, and compression can produce false positives; a real indicator change smaller than 10% will be missed. For true physical-state identification, use two known-state brightness measurements or external instrumentation. **Not a safety-certified industrial alert system.**

### Optional ambient correction

An individual `reference_roi` can be selected on stable, unilluminated surrounding material. On calibration the mean brightness of that region is saved as `baseline_reference_brightness`. During monitoring:

```
adjusted_delta = (current_sensor - baseline_sensor)
               - (current_ambient - baseline_ambient)
delta_percent  = 100 * adjusted_delta / max(baseline_sensor, 1)
```

The reference ROI does not affect the denominator and is optional (`null` by default). It should never include an illuminated sensor.

## Install

Python >=3.12,<3.15:

```bash
poetry install
```

Or install with a local virtual environment: `pip install -r requirements.txt`.

## Configuration and calibration (offline-first)

The included `config.json` uses **schema version 3**, the current baseline-brightness JSON format for **MVP 0.2**. The schema version identifies the file structure, not the MVP release. It preserves the five supplied **IDs, names, normal states, frame resolution, and ROI coordinates**. Each sensor's `baseline_brightness` starts as `null` until calibrated from an actual frame.

Only the current schema is supported. Old color-threshold configurations must be replaced with the included format; there is no automatic legacy-format converter. The calibration tool creates a `.bak` copy before overwriting the configuration.

Use an existing full-frame offline image (recommended to avoid network requests):

```bash
poetry run python calibrate.py --image frame.jpg
```

If **you** choose to get a frame from the authorized HLS stream, running `poetry run python calibrate.py` without `--image` will open the configured stream (this was **not** done by the assistant).

### Calibration GUI

- Choose any sensor using the numbered buttons. Pagination supports 1..N indicators; the video frame stays at its native calibration coordinates.
- Click the sensor center to adjust its ROI. Baseline mean is captured immediately from the displayed frame.
- `B` / **Set Baseline** re-captures the selected sensor from the displayed frame without changing its ROI.
- **`T` / Toggle ON/OFF** labels what the selected sensor *actually looked like in that baseline image*. Verify this carefully; it determines whether baseline deviation is printed as `ON -> OFF` or `OFF -> ON`.
- `A` / **Ref ROI** enables optional ambient reference selection; **Clear Ref** removes it.
- **Save** automatically records baselines for any sensors that have an ROI but still have no baseline. Saved baselines are not overwritten when the GUI is reopened unless you change the ROI, reference or explicitly select Set Baseline.
- The existing Undo, Clear, Reset, Radius +/−, and multiple-frame navigation tools remain available.
- All calibration data is saved atomically; the previous config is backed up to `config.json.bak`.

The GUI prints e.g.:

```text
Saved 5 sensor ROIs to .../config.json; brightness baselines: 5/5
Confirm each sensor's baseline_state matches its actual illumination in the captured frame.
```

## Configuration

```json
{
  "schema_version": 3,
  "sensor_count": 1,
  "detection": {
    "confirmation_frames": 2,
    "max_uncertain_seconds": 2.0,
    "brightness_change_percent": 10.0
  },
  "indicators": [
    {
      "id": "SENSOR_001", "name": "Machine Indicator",
      "normal_state": "OFF", "baseline_state": "OFF",
      "roi": [140, 562, 17, 17], "reference_roi": null,
      "baseline_brightness": 65.4,
      "baseline_reference_brightness": null
    }
  ]
}
```

The example shows the *relevant fields only*: the actual JSON also needs `stream`, `video` and `output` blocks (see `config.json.example`). `baseline_brightness` is illustrative and must be measured, not copied. `normal_state` defines the expected equipment condition; `baseline_state` describes the observed calibration frame state. Source coordinates always refer to the original decoded frame.

```bash
poetry run python main.py --check-schema   # Validate uncalibrated JSON, offline
poetry run python main.py --check-config   # Requires all baselines, offline
poetry run python main.py --inspect-image frame.jpg --output inspection.jpg
```

The `--inspect-image` command prints actual mean brightness and signed percentage deviation for every configured sensor, and writes an ROI-marked image. It never opens the network.

## Offline tests

```bash
poetry run python -m pytest -q
poetry run python tools/generate_demo.py --directory offline_demo --count 12
poetry run python main.py --config offline_demo/config.json
```

The synthetic video demonstrates five-sensor-independent and N-sensor processing with four confirmed transitions (including simultaneous changes). No factory HLS traffic is needed. The demo uses simulated baselines and **does not** calibrate your factory's indicators.

## Production invocation (operator only)

After calibrating and verifying with local recordings:

```bash
poetry run python main.py --check-config
poetry run python main.py
```

Each confirmed transition saves its own unannotated, full-frame JPEG. Optional `_preview.jpg` compares the current crop against the previously confirmed crop. Startup and resync differences are **not** mislabeled as directly observed transitions. Missing video is not treated as an OFF event. Screenshot persistence is atomic with bounded retries; errors are logged explicitly.

The application does **not** contact the configured HLS endpoint for `--check-config`, `--check-schema`, `--inspect-image`, local video input, or `--image` calibration. Keep `config.json` private: it may contain an access-bearing stream URL. `.gitignore` excludes it by default.

## Official technical references

- [OpenCV color conversion / grayscale](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html)
- [OpenCV `mean` and masked mean](https://docs.opencv.org/4.x/d2/de8/group__core__array.html)
- [OpenCV VideoCapture backend and timeouts](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html)
- [Python `os.replace`](https://docs.python.org/3/library/os.html#os.replace)

## Credits

- Denis Laskov — project concept, requirements and architecture.
- ChatGPT Coding agent (OpenAI) — AI-assisted implementation and offline validation.
