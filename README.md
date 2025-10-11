# MQTT Analysis Service

This project implements a lightweight backend service intended for deployment on embedded Linux boards (such as the RK3588). The service consumes configuration from `local.config.json`, connects to an MQTT broker, listens for analysis commands, captures frames from an RTSP stream, evaluates the frames with the models defined by each scenario, and publishes per-frame anomaly notifications back to MQTT.

## Features

- Minimal JSON reader/writer implemented in `include/app/json.hpp` to avoid third-party dependencies.
- Scenario abstraction (`Scenario` class) that encapsulates model metadata, model loading, and inference interfaces.
- Support for multiple scenarios per command; redundant requests for already active scenarios are ignored while new scenarios are loaded on demand.
- Local configuration that maps scenario identifiers to dedicated config files so each scenario can declare its model inventory.
- Simulated CNN and YOLO model backends that produce deterministic detections without requiring heavy runtimes. The structure mirrors real ONNX flows and can be swapped with hardware-accelerated engines.
- Simple MQTT publisher stub (logs to stdout) that demonstrates the JSON payload that would be sent to a broker.
- RTSP grabber that writes placeholder frames to `output/frames` so the inference pipeline has concrete image artifacts to analyze.

## Building

```bash
cmake -S . -B build
cmake --build build
```

This generates the `mqtt_msg` executable in `build/`.

## Running

1. Adjust `local.config.json` with your MQTT, RTSP, and scenario file paths. The provided placeholder assumes four scenarios:
   - Steam recognition
   - Coal powder recognition
   - Ash powder recognition
   - Liquid leak detection
2. Craft a command payload similar to `config/sample_command.json`.
3. Execute the CLI entry point:

```bash
./build/mqtt_msg --config local.config.json --command config/sample_command.json
```

The program activates only the requested scenarios, captures two frames (because the sample command requests `frame_rate: 2`), runs each frame through the active scenario list, and prints JSON payloads for every frame that yields detections meeting the configured thresholds.

### MQTT Output Format

Each MQTT message follows the structure below:

```json
{
  "service": "industrial_monitor",
  "frame": {
    "path": "output/frames/20240101_120000_0.jpg",
    "timestamp": "20240101_120000_0.jpg"
  },
  "results": [
    {
      "scenario_id": "steam_detection",
      "detections": [
        {
          "model_id": "steam_cnn_v1",
          "label": "steam",
          "confidence": 0.87,
          "bbox": [120, 64, 140, 120],
          "image_path": "output/frames/20240101_120000_0.jpg",
          "timestamp": "20240101_120000_0.jpg"
        }
      ]
    }
  ]
}
```

When no detections exceed the thresholds, no message is published for that frame.

## Extending the Pipeline

- Replace the simulated model classes in `src/model.cpp` with wrappers around ONNX Runtime or vendor-specific inference engines.
- Update each scenario file in `config/` with the actual ONNX model paths and label metadata.
- Integrate a real MQTT client inside `src/mqtt.cpp` (e.g., libmosquitto) and forward the JSON payload produced in `main` to the desired broker/topic.
- Swap `RtspFrameGrabber` with an ffmpeg- or GStreamer-based implementation to retrieve real frames from the RTSP endpoint.

## Repository Layout

```
include/app/        Core headers (JSON, config, command, model, scenario, pipeline, RTSP, MQTT)
src/                Implementation files
config/             Scenario definition JSON and command samples
models/             Placeholder directory for ONNX/Torch models
output/             Generated frames (created at runtime)
```

## License

This repository is provided as-is for instructional purposes.

