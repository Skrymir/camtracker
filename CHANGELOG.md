# Changelog

## Unreleased

- A single missed detection no longer moves the eyes to someone else in view:
  the followed person is waited for up to `lost_after_s` before switching.
- Lower stream latency: ffmpeg reads the RTSP stream with input buffering off
  (`fflags nobuffer`, `flags low_delay`).
- Docker image builds again: Google deleted the Coral apt repo, so the image
  is now Debian 12 (bookworm, Python 3.11) with libedgetpu from feranick's
  builds and Frigate's tflite_runtime wheel, both checksum-pinned. pycoral is
  gone; the detector calls tflite_runtime directly.
- The Coral works on the first start after a reboot: the container mounts the
  live `/dev/bus/usb` with a cgroup rule for USB devices, instead of a
  `--device` list that goes stale when the Coral changes its USB ID.
- `[camera] far_roi`: an optional second, finer pass over where people are
  far away, so `roi` can cover the whole height for people near the camera.
  The top of a near person seen by the far pass is merged into them.
- `[detector] ignore`: boxes for decorations the model takes for people (a
  ghost, a skeleton). A detection that matches one (IoU >= `ignore_iou`) is
  dropped; someone walking past it is still seen.
- `--snapshot` shows each box's score, draws near misses (just under
  `min_score`) in orange and ignored decorations in magenta, and prints each
  detection's box ready to paste into `ignore`.
- `--snapshot` waits (from 2 s in) for a frame with detail rather than the
  first one, which on H.265 can be grey until the next keyframe; it prints
  the frame's detail and warns if the picture is still blank.
- `--snapshot` drawing scales with the frame, so labels and the ruler are
  readable on a 4K stream; labels stay inside the picture.
- `--watch` prints each detection's box too, for `ignore`.
- Logs which stream it is opening (password hidden); `--snapshot` waits 45 s
  so ffmpeg's own open timeout is reported.

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
