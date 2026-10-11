# Changelog

## 0.1.0 — 2026-10-10

First version, split out of demoneye (where it started as `cam_tracker.py`
on the feature/camera-tracker branch, never merged).

- RTSP stream -> person detection on a USB Coral (MobileDet) -> nearest
  person published to `house/tracking/target` at up to 15/s; follows
  `house/show/command`.
- Lens-model mapping (`hfov_deg`, `gaze_max_deg`) or calibrated `points`;
  optional look-down from how low a person's feet are.
- ROI with automatic square tiles for wide strips; sticks with the person it
  is following.
- `--snapshot` and `--watch` for setup and calibration.
- Docker image (Debian bullseye, pycoral) and compose file for Unraid.
- Home Assistant status via MQTT discovery (running, tracking, stream,
  detector, last person seen).
