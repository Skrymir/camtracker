#!/usr/bin/env python3
"""
Camera person tracker.

Reads a camera's RTSP stream, finds people with a Coral Edge TPU, and publishes
where the nearest one is as a look target over MQTT, 10-15 times a second.
Built for demoneye (projected eyes that follow people): it speaks the same
target message and on/off command as demoneye's mmWave bridge, so anything
that understands those can use it.

Mount the camera on the house between the eyes, looking at the street. Then a
person's left/right position in the picture is (almost exactly) the angle the
eyes need to look, with no distance needed.

Runs in Docker on a server with a USB Coral (see README.md).

    python camtracker.py                     # run (reads camtracker.ini)
    python camtracker.py --snapshot s.jpg    # one frame with the ROI, tiles and
                                             # detections drawn: set up the ROI
    python camtracker.py --watch             # print every detection's position
                                             # (for calibration); add
                                             # --no-publish to leave the eyes alone

Only the run/IO parts need OpenCV and tflite_runtime; the mapping, tracking
and detection-parsing logic (Mapper, pick, Tracker, ssd_people) is pure Python
and unit-tested on its own.
"""

import argparse
import configparser
import glob
import json
import math
import os
import re
import signal
import socket
import sys
import threading
import time

import status

DEFAULTS = {
    "camera": {
        # rtsp://... of the camera's substream, or Frigate's restream
        # (rtsp://<frigate-host>:8554/<camera>). CAMERA_URL env overrides.
        "url": "",
        "detect_fps": "12",
        # region of interest, fractions of the frame: x1, y1, x2, y2
        "roi": "0, 0, 1, 1",
        # optional second, finer pass over where people are far away (small
        # in the picture): x1, y1, x2, y2 like roi, tiled on its own
        "far_roi": "",
        # model passes across the ROI: auto = enough for roughly square tiles
        "tiles": "auto",
    },
    "detector": {
        "model": "/models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite",
        # auto = the first Edge TPU that isn't busy (Frigate holds its own)
        "device": "auto",
        "person_id": "0",
        "min_score": "0.5",
        # ignore boxes shorter than this fraction of the frame height
        "min_height": "0.04",
        # things that look like people but aren't (decorations): boxes as
        # "x1, y1, x2, y2; ..." in frame fractions, from --snapshot
        "ignore": "",
        # a detection overlapping an ignore box at least this much (IoU) is it
        "ignore_iou": "0.5",
        # regions drawn in Frigate's mask editor, its coordinates pasted as is:
        # "x1,y1,x2,y2,x3,y3,...; ..." (points as frame fractions)
        "ignore_masks": "",
        # a detection with at least this much of its box inside a mask is dropped
        "ignore_inside": "0.6",
    },
    "mqtt": {"host": "192.168.86.5", "port": "1883", "username": "", "password": "",
             "target_topic": "house/tracking/target",
             "command_topic": "house/show/command",
             "follow_command": "true", "start_enabled": "false"},
    "mapping": {
        "hfov_deg": "100",
        "gaze_max_deg": "35",
        # optional calibration: "u:x, u:x, ..." (u = 0..1 across the frame,
        # x = where the eye should look, -1..+1). Overrides hfov_deg/gaze_max_deg.
        "points": "",
        "invert_x": "false",
        # look down as a person gets closer (their feet lower in the frame)
        "y_from_feet": "false",
        "y_far": "0.45",
        "y_near": "0.95",
        # shape of the look-down between y_far and y_near: 1 = straight line,
        # below 1 = more of the look early (walking up), less at the end
        "y_curve": "1.0",
    },
    "tracking": {
        "smooth_factor": "0.5",
        "publish_hz": "15",
        "heartbeat_s": "0.75",
        "hold_s": "2.0",
        # keep the target through detection misses this long before clearing
        "lost_after_s": "0.8",
        # a detection further than this (fraction of the frame width) from the
        # tracked person is someone else
        "max_jump": "0.25",
    },
    "status": dict(status.STATUS_DEFAULTS, node="camera", refresh_s="60",
                   last_seen_every_s="10"),
}


def _clamp(v, lo=-1.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


def _floats(text, n=None, what="value"):
    vals = [float(p) for p in str(text).replace(";", ",").split(",") if p.strip()]
    if n is not None and len(vals) != n:
        raise ValueError(f"{what}: expected {n} numbers, got {text!r}")
    return vals


def _region(text, what):
    r = _floats(text, 4, what)
    x1, y1, x2, y2 = r
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"{what} must be x1,y1,x2,y2 inside 0..1 with x1<x2, y1<y2: {r}")
    return r


def parse_boxes(text):
    """'0.1, 0.2, 0.3, 0.4; 0.5, ...' -> [(0.1, 0.2, 0.3, 0.4), ...]."""
    boxes = []
    for part in str(text or "").split(";"):
        if not part.strip():
            continue
        b = tuple(_floats(part, 4, "ignore box"))
        if not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
            raise ValueError(f"ignore box must be x1,y1,x2,y2 inside 0..1: {part.strip()!r}")
        boxes.append(b)
    return boxes


def parse_masks(text):
    """Frigate mask coordinates, one mask per line or ";":
    '0.1,0.2,0.3,0.2,0.3,0.4; ...' -> [[(0.1, 0.2), (0.3, 0.2), (0.3, 0.4)], ...]."""
    masks = []
    for part in str(text or "").replace("\n", ";").split(";"):
        if not part.strip():
            continue
        v = _floats(part, what="ignore mask")
        if len(v) < 6 or len(v) % 2:
            raise ValueError(f"ignore mask needs x,y pairs, at least 3 points: {part.strip()!r}")
        if not all(0 <= c <= 1 for c in v):
            raise ValueError("ignore mask points must be frame fractions 0..1 (Frigate 0.14+ "
                             f"writes them so; older versions used pixels): {part.strip()!r}")
        masks.append(list(zip(v[0::2], v[1::2])))
    return masks


def parse_points(text):
    """'0.1:-1, 0.5:0, 0.9:1' -> [(0.1, -1.0), (0.5, 0.0), (0.9, 1.0)] sorted by u."""
    pts = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        u, sep, x = part.partition(":")
        if not sep:
            raise ValueError(f"points: {part!r} should be u:x")
        pts.append((float(u), float(x)))
    pts.sort()
    if len(pts) == 1:
        raise ValueError("points: need at least two")
    if any(b[0] == a[0] for a, b in zip(pts, pts[1:])):
        raise ValueError("points: two points have the same u")
    return pts


class Config:
    def __init__(self, path=None):
        cp = configparser.ConfigParser()
        cp.read_dict(DEFAULTS)
        if path and os.path.exists(path):
            cp.read(path)
            print(f"[cfg] loaded {path}", flush=True)
        else:
            print(f"[cfg] {path} not found — using defaults", flush=True)

        g = lambda s, k: cp.get(s, k).strip()          # noqa: E731
        self.url = os.environ.get("CAMERA_URL") or g("camera", "url")
        self.detect_fps = max(1.0, cp.getfloat("camera", "detect_fps"))
        self.roi = _region(g("camera", "roi"), "roi")
        far = g("camera", "far_roi")
        self.far_roi = _region(far, "far_roi") if far else None
        t = g("camera", "tiles").lower()
        self.tiles = 0 if t in ("", "auto") else max(1, int(t))

        self.model = g("detector", "model")
        self.device = g("detector", "device") or "auto"
        self.person_id = cp.getint("detector", "person_id")
        self.min_score = cp.getfloat("detector", "min_score")
        self.min_height = cp.getfloat("detector", "min_height")
        self.ignore = parse_boxes(g("detector", "ignore"))
        self.ignore_iou = cp.getfloat("detector", "ignore_iou")
        self.ignore_masks = parse_masks(g("detector", "ignore_masks"))
        self.ignore_inside = cp.getfloat("detector", "ignore_inside")

        self.mqtt_host = g("mqtt", "host")
        self.mqtt_port = cp.getint("mqtt", "port")
        self.mqtt_user = os.environ.get("MQTT_USERNAME") or g("mqtt", "username")
        self.mqtt_pass = os.environ.get("MQTT_PASSWORD") or g("mqtt", "password")
        self.topic = g("mqtt", "target_topic")
        self.command_topic = g("mqtt", "command_topic")
        self.follow_command = cp.getboolean("mqtt", "follow_command")
        self.start_enabled = cp.getboolean("mqtt", "start_enabled")

        self.hfov = cp.getfloat("mapping", "hfov_deg")
        if not 10 <= self.hfov < 179:
            raise ValueError(f"hfov_deg must be 10..179: {self.hfov}")
        self.gaze_max = max(1.0, cp.getfloat("mapping", "gaze_max_deg"))
        self.points = parse_points(g("mapping", "points"))
        self.invert_x = cp.getboolean("mapping", "invert_x")
        self.y_from_feet = cp.getboolean("mapping", "y_from_feet")
        self.y_far = cp.getfloat("mapping", "y_far")
        self.y_near = cp.getfloat("mapping", "y_near")
        self.y_curve = cp.getfloat("mapping", "y_curve")
        if self.y_curve <= 0:
            raise ValueError(f"y_curve must be above 0: {self.y_curve}")

        self.smooth = min(1.0, max(0.01, cp.getfloat("tracking", "smooth_factor")))
        self.min_interval = 1.0 / max(1.0, cp.getfloat("tracking", "publish_hz"))
        self.heartbeat = cp.getfloat("tracking", "heartbeat_s")
        self.hold = cp.getfloat("tracking", "hold_s")
        self.lost_after = cp.getfloat("tracking", "lost_after_s")
        self.max_jump = cp.getfloat("tracking", "max_jump")

        st = status.load_section(cp, "status")
        self.status_enabled = st["enabled"]
        self.status_node = status.resolve_node(st["node"])
        self.status_base = st["base_topic"]
        self.status_prefix = st["discovery_prefix"]
        self.status_refresh = st["refresh_s"]
        self.last_seen_every = cp.getfloat("status", "last_seen_every_s")


# ------------------------------------------------------------- pure logic
class Mapper:
    """A person's position in the frame -> the eyes' gaze (-1..+1 each way).

    u = horizontal position of the person (0 = left edge, 1 = right edge),
    v = their feet (bottom of the box, 0 = top, 1 = bottom).

    Lens model (no points): the camera sits between the eyes, so its bearing
    to the person is the eyes' bearing:
        bearing = atan((2u - 1) * tan(hfov / 2))      + = right in the picture
        x       = -bearing / gaze_max_deg
    The minus: the camera and the eyes face the street, so the picture's right
    is the audience's left, and the pupil moves screen-left (negative).

    points = "u:x, ..." replaces that with straight lines between calibrated
    spots (for a wide lens that bends the picture, or an off-centre camera).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._k = math.tan(math.radians(cfg.hfov) / 2.0)

    def bearing(self, u):
        return math.degrees(math.atan((2.0 * u - 1.0) * self._k))

    def gaze_x(self, u):
        pts = self.cfg.points
        if pts:
            if u <= pts[0][0]:
                x = pts[0][1]
            elif u >= pts[-1][0]:
                x = pts[-1][1]
            else:
                for (u0, x0), (u1, x1) in zip(pts, pts[1:]):
                    if u0 <= u <= u1:
                        x = x0 + (x1 - x0) * (u - u0) / (u1 - u0)
                        break
        else:
            x = -self.bearing(u) / self.cfg.gaze_max
        x = _clamp(x)
        return -x if self.cfg.invert_x else x

    def gaze_y(self, v):
        c = self.cfg
        if not c.y_from_feet or c.y_near == c.y_far:
            return 0.0
        # feet near the bottom of the frame = close = look down (-1)
        frac = _clamp((v - c.y_far) / (c.y_near - c.y_far), 0.0, 1.0)
        return -(frac ** c.y_curve)

    def __call__(self, u, v):
        return self.gaze_x(u), self.gaze_y(v)


def pick(dets, last_u=None, max_jump=0.25):
    """Choose who to look at. dets: [(score, x1, y1, x2, y2), ...] in frame
    fractions. Keeps following the person being tracked (the detection nearest
    last_u, if within max_jump); otherwise the tallest box, i.e. the nearest
    person. Returns (u, v) = (box centre x, box bottom) or None."""
    if not dets:
        return None
    if last_u is not None:
        best = min(dets, key=lambda d: abs((d[1] + d[3]) / 2 - last_u))
        if abs((best[1] + best[3]) / 2 - last_u) <= max_jump:
            return ((best[1] + best[3]) / 2, best[4])
    best = max(dets, key=lambda d: d[4] - d[2])
    return ((best[1] + best[3]) / 2, best[4])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets, thresh=0.5):
    """Merge duplicates (a person seen by two overlapping tiles)."""
    keep = []
    for d in sorted(dets, key=lambda d: -d[0]):
        if all(iou(d[1:], k[1:]) < thresh for k in keep):
            keep.append(d)
    return keep


def drop_contained(dets, thresh=0.7):
    """Drop a box that lies mostly (thresh of its area) inside a bigger one:
    the finer far_roi pass sees only the top of someone near the camera."""
    def area(d):
        return (d[3] - d[1]) * (d[4] - d[2])
    keep = []
    for d in sorted(dets, key=area, reverse=True):
        a = area(d)
        inside = lambda k: (max(0.0, min(d[3], k[3]) - max(d[1], k[1]))      # noqa: E731
                            * max(0.0, min(d[4], k[4]) - max(d[2], k[2])))
        if a <= 0 or all(inside(k) / a < thresh for k in keep):
            keep.append(d)
    return keep


def ssd_people(boxes, classes, scores, count, tile, frame_w, frame_h, cfg, min_score=None):
    """One tile's SSD output (boxes as ymin, xmin, ymax, xmax fractions of the
    tile) -> the people in it, [(score, x1, y1, x2, y2)] as frame fractions.
    min_score: instead of cfg.min_score (the snapshot looks lower)."""
    min_score = cfg.min_score if min_score is None else min_score
    tx1, ty1, tx2, ty2 = tile
    tw, th = tx2 - tx1, ty2 - ty1
    out = []
    for i in range(min(count, len(scores))):
        if int(classes[i]) != cfg.person_id or scores[i] < min_score:
            continue
        ymin, xmin, ymax, xmax = (float(b) for b in boxes[i])
        x1, y1 = (tx1 + max(0.0, xmin) * tw) / frame_w, (ty1 + max(0.0, ymin) * th) / frame_h
        x2, y2 = (tx1 + min(1.0, xmax) * tw) / frame_w, (ty1 + min(1.0, ymax) * th) / frame_h
        if (y2 - y1) >= cfg.min_height:
            out.append((float(scores[i]), x1, y1, x2, y2))
    return out


def _clip(pts, axis, limit, low):
    """The part of polygon pts on one side (x or y >= / <= limit) of a line."""
    keep = (lambda p: p[axis] >= limit) if low else (lambda p: p[axis] <= limit)
    out = []
    for i, q in enumerate(pts):
        p = pts[i - 1]
        if keep(p) != keep(q):                      # an edge crossing the line
            t = (limit - p[axis]) / (q[axis] - p[axis])
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        if keep(q):
            out.append(q)
    return out


def inside_fraction(box, poly):
    """How much of box (x1, y1, x2, y2) lies inside polygon poly [(x, y), ...]:
    the polygon is clipped to the box (Sutherland-Hodgman) and measured."""
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    pts = list(poly)
    for axis, limit, low in ((0, x1, True), (0, x2, False), (1, y1, True), (1, y2, False)):
        if not pts or area <= 0:
            return 0.0
        pts = _clip(pts, axis, limit, low)
    shoelace = sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(pts, pts[1:] + pts[:1]))
    return abs(shoelace) / 2 / area


def drop_ignored(dets, ignore, min_iou=0.5, masks=(), min_inside=0.6):
    """Detections that aren't one of the ignore boxes (a decoration seen as a
    person) or mostly inside an ignore mask. Someone walking past one has a
    different, bigger box, so is kept."""
    return [d for d in dets
            if all(iou(d[1:], b) < min_iou for b in ignore)
            and all(inside_fraction(d[1:], m) < min_inside for m in masks)]


def tile_boxes(roi, frame_w, frame_h, tiles=0, overlap=0.15):
    """Split the ROI into side-by-side tiles (pixels: x1, y1, x2, y2), each
    roughly square so the model's square input isn't squashed. tiles=0: auto."""
    x1, y1 = int(roi[0] * frame_w), int(roi[1] * frame_h)
    x2, y2 = int(roi[2] * frame_w), int(roi[3] * frame_h)
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    n = tiles or max(1, min(6, round(w / h)))
    if n == 1:
        return [(x1, y1, x2, y2)]
    tw = w / (n - (n - 1) * overlap)            # tile width so n overlapping tiles span w
    step = tw * (1 - overlap)
    out = []
    for i in range(n):
        a = x1 + int(round(i * step))
        b = x2 if i == n - 1 else min(x2, int(round(a + tw)))
        out.append((a, y1, b, y2))
    return out


def frame_tiles(cfg, frame_w, frame_h):
    """The tiles for one frame: the ROI's, then far_roi's (finer) if set."""
    tiles = tile_boxes(cfg.roi, frame_w, frame_h, cfg.tiles)
    if cfg.far_roi:
        tiles += tile_boxes(cfg.far_roi, frame_w, frame_h)
    return tiles


class Tracker:
    """Detections in, target / clear messages out (rate-limited, smoothed).

    update() once per processed frame with that frame's people; tick() often
    (it sends the heartbeat and the delayed clear). Both return a list of
    messages to publish: ("target", x, y) or ("clear",)."""

    def __init__(self, cfg, mapper=None, clock=time.monotonic):
        self.cfg = cfg
        self.mapper = mapper or Mapper(cfg)
        self.clock = clock
        self.have = False
        self.sx = self.sy = 0.0
        self.last_u = None
        self.last_seen = None
        self.last_pub = -1e9
        self.acquired = self.lost = 0           # counters, for logs / tests

    def reset(self):
        """Drop the target. Returns the clear message if one was held."""
        was = self.have
        self.have, self.last_u = False, None
        if was:
            self.lost += 1
        return [("clear",)] if was else []

    def update(self, dets):
        now = self.clock()
        p = pick(dets, self.last_u if self.have else None, self.cfg.max_jump)
        if p is None:
            return self.tick()
        u, v = p
        if (self.have and abs(u - self.last_u) > self.cfg.max_jump
                and now - self.last_seen < self.cfg.lost_after):
            # the person being followed was missed this frame and someone else
            # is in view: wait for them rather than flicking over
            return self.tick()
        gx, gy = self.mapper(u, v)
        self.last_u, self.last_seen = u, now
        if not self.have:                        # fresh: snap, no smoothing
            self.have = True
            self.sx, self.sy = gx, gy
            self.acquired += 1
            self.last_pub = now
            return [("target", self.sx, self.sy)]
        a = self.cfg.smooth
        self.sx += (gx - self.sx) * a
        self.sy += (gy - self.sy) * a
        if now - self.last_pub >= self.cfg.min_interval:
            self.last_pub = now
            return [("target", self.sx, self.sy)]
        return []

    def tick(self):
        if not self.have:
            return []
        now = self.clock()
        if now - self.last_seen >= self.cfg.lost_after:
            return self.reset()
        if now - self.last_pub >= self.cfg.heartbeat:
            self.last_pub = now
            return [("target", self.sx, self.sy)]
        return []


# ---------------------------------------------------------------- video IO
# ffmpeg options for the stream (OPENCV_FFMPEG_CAPTURE_OPTIONS overrides)
CAPTURE_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"


def redact(url):
    """rtsp://user:pass@host/... -> rtsp://***@host/... (for logs)."""
    return re.sub(r"//[^/]*@", "//***@", url, count=1)


class FrameGrabber:
    """Reads the stream on its own thread and keeps only the newest frame, so
    detection always sees "now" rather than a backlog. Reconnects on failure."""

    def __init__(self, url):
        self.url = url
        self.frame = None
        self.seq = 0
        self.ok = False
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="grabber", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None
        self.ok = False
        with self._cond:
            self.frame = None
            self._cond.notify_all()

    def latest(self, after_seq, timeout=1.0):
        """Wait for a frame newer than after_seq. Returns (seq, frame) or (after_seq, None)."""
        with self._cond:
            self._cond.wait_for(lambda: self.seq > after_seq or self._stop.is_set(), timeout)
            if self.seq > after_seq and self.frame is not None:
                return self.seq, self.frame
            return after_seq, None

    def _run(self):
        import cv2
        # no input buffering: ffmpeg otherwise queues frames, and the eyes lag
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", CAPTURE_OPTIONS)
        backoff = 1.0
        while not self._stop.is_set():
            print(f"[cam] opening {redact(self.url)}", flush=True)
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.ok = False
                print(f"[cam] can't open the stream — retrying in {backoff:.0f}s", flush=True)
                self._stop.wait(backoff)
                backoff = min(30.0, backoff * 2)
                continue
            print("[cam] stream open", flush=True)
            backoff = 1.0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    print("[cam] stream dropped — reconnecting", flush=True)
                    break
                self.ok = True
                with self._cond:
                    self.frame, self.seq = frame, self.seq + 1
                    self._cond.notify_all()
            self.ok = False
            cap.release()
            self._stop.wait(1.0)            # don't spin if it drops straight away


# USB Corals as (vendor, product): before and after their firmware is loaded
CORAL_USB_IDS = {("1a6e", "089a"), ("18d1", "9302")}


def find_tpus(sys_usb="/sys/bus/usb/devices", dev="/dev"):
    """Edge TPU device names to try: usb:0.. for each USB Coral, pci:0.. for
    each PCIe one. Empty if none can be seen."""
    n_usb = 0
    for d in glob.glob(os.path.join(sys_usb, "*")):
        try:
            ids = tuple(open(os.path.join(d, f)).read().strip()
                        for f in ("idVendor", "idProduct"))
        except OSError:
            continue
        n_usb += ids in CORAL_USB_IDS
    n_pci = len(glob.glob(os.path.join(dev, "apex_*")))
    return [f"usb:{i}" for i in range(n_usb)] + [f"pci:{i}" for i in range(n_pci)]


class CoralDetector:
    """SSD-style detection model on an Edge TPU (tflite_runtime + libedgetpu)."""

    def __init__(self, cfg):
        from tflite_runtime.interpreter import Interpreter, load_delegate
        self.cfg = cfg
        if cfg.device.lower() != "auto":
            candidates = [cfg.device]
        else:
            candidates = find_tpus()
            print(f"[tpu] found {len(candidates)} Edge TPU(s): {candidates}", flush=True)
        err = None
        for dev in candidates:
            try:
                interp = Interpreter(model_path=cfg.model, experimental_delegates=[
                    load_delegate("libedgetpu.so.1", {"device": dev})])
                interp.allocate_tensors()
            except Exception as e:                  # busy (Frigate) or absent
                print(f"[tpu] {dev}: {e}", flush=True)
                err = e
                continue
            self.interp, self.device = interp, dev
            inp = interp.get_input_details()[0]
            self._input = inp["index"]
            self.size = (int(inp["shape"][2]), int(inp["shape"][1]))     # w, h
            # SSD postprocess outputs: boxes, classes, scores, count
            self._outputs = [d["index"] for d in interp.get_output_details()[:4]]
            print(f"[tpu] using {dev}, model input {self.size[0]}x{self.size[1]}", flush=True)
            return
        raise RuntimeError(f"no usable Edge TPU (tried {candidates or 'none found'}): {err}")

    def detect(self, frame, tiles, min_score=None):
        """People in frame (BGR) within the tile rectangles ->
        [(score, x1, y1, x2, y2)] as fractions of the whole frame."""
        import cv2
        fh, fw = frame.shape[:2]
        out = []
        for tile in tiles:
            tx1, ty1, tx2, ty2 = tile
            crop = frame[ty1:ty2, tx1:tx2]
            rgb = cv2.cvtColor(cv2.resize(crop, self.size), cv2.COLOR_BGR2RGB)
            self.interp.set_tensor(self._input, rgb[None])
            self.interp.invoke()
            boxes, classes, scores, count = (self.interp.get_tensor(i)[0] for i in self._outputs)
            out += ssd_people(boxes, classes, scores, int(count), tile, fw, fh, self.cfg,
                              min_score)
        return drop_contained(nms(out))


# ------------------------------------------------------------------- service
def set_nodelay(client):
    try:
        sock = client.socket()
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except (OSError, AttributeError):
        pass


class CamTracker:
    ON_WORDS = set(status.ON_WORDS)
    OFF_WORDS = set(status.OFF_WORDS)

    def __init__(self, cfg, detector=None, grabber=None, publish=True, watch=False):
        self.cfg = cfg
        self.detector = detector
        self.grabber = grabber or FrameGrabber(cfg.url)
        self.tracker = Tracker(cfg)
        self.publish = publish
        self.watch = watch
        self.enabled = watch or (not cfg.follow_command) or cfg.start_enabled
        self.client = None
        self.status = None
        self._seen_pub = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._host = status.hostname()

    # -- MQTT --
    def _parse_command(self, payload):
        v = payload.decode(errors="ignore").strip().strip('"').lower()
        if v.startswith("{"):
            try:
                v = str(json.loads(v).get("state", "")).strip().lower()
            except Exception:
                return None
        return True if v in self.ON_WORDS else False if v in self.OFF_WORDS else None

    def _on_command(self, client, userdata, msg):
        want = self._parse_command(msg.payload)
        with self._lock:
            if want is None or want == self.enabled or self.watch:
                return
            self.enabled = want
            print(f"[cam] tracking {'enabled' if want else 'disabled'}", flush=True)
            if not want:
                self._send(self.tracker.reset())

    def connect(self):
        import paho.mqtt.client as mqtt
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"cam-tracker-{self._host}")
        if self.cfg.mqtt_user:
            c.username_pw_set(self.cfg.mqtt_user, self.cfg.mqtt_pass)
        if self.cfg.follow_command:
            c.message_callback_add(self.cfg.command_topic, self._on_command)
        if self.cfg.status_enabled and not self.watch:
            self.status = status.StatusPublisher(
                c, self.cfg.status_node,
                status.discovery(self.cfg.status_node, self.cfg.status_base,
                                           self.cfg.status_prefix),
                self.status_snapshot, base=self.cfg.status_base,
                prefix=self.cfg.status_prefix, refresh_s=self.cfg.status_refresh)

        def _on_connect(cl, u, flags, rc, props):
            if rc.is_failure:
                print(f"[mqtt] broker refused the connection: {rc} — check "
                      "MQTT_USERNAME / MQTT_PASSWORD", flush=True)
                return
            print(f"[mqtt] connected rc={rc}", flush=True)
            set_nodelay(cl)
            if self.cfg.follow_command:
                cl.subscribe(self.cfg.command_topic, qos=1)
            if self.status:
                self.status.on_connect()

        c.on_connect = _on_connect
        c.reconnect_delay_set(1, 30)
        c.connect_async(self.cfg.mqtt_host, self.cfg.mqtt_port, keepalive=30)
        c.loop_start()
        self.client = c
        if self.status:
            self.status.start()

    def status_snapshot(self):
        return {"tracking": bool(self.enabled and self.tracker.have),
                "enabled": bool(self.enabled),
                "stream_ok": bool(self.enabled and self.grabber.ok),
                "detector": getattr(self.detector, "device", "none"),
                "host": self._host, "version": status.VERSION}

    def _send(self, msgs):
        for m in msgs:
            if m[0] == "target":
                payload = json.dumps({"x": round(m[1], 4), "y": round(m[2], 4),
                                      "hold": self.cfg.hold})
                if self.tracker.acquired and self.status:
                    self._mark_seen()
            else:
                payload = "clear"
                if self.status:
                    self._mark_seen(force=True)
            if self.publish and self.client is not None:
                self.client.publish(self.cfg.topic, payload, qos=0)

    def _mark_seen(self, force=False):
        now = time.monotonic()
        if force or self._seen_pub is None or now - self._seen_pub >= self.cfg.last_seen_every:
            self.status.publish_last_seen()
            self._seen_pub = now

    # -- main loop --
    def run(self):
        tiles, seq, frames, t_log = None, 0, 0, time.monotonic()
        last_have = False
        while not self._stop.is_set():
            with self._lock:
                enabled = self.enabled
            if not enabled:
                self.grabber.stop()
                tiles = None
                self._stop.wait(0.5)
                continue
            self.grabber.start()
            t0 = time.monotonic()
            seq, frame = self.grabber.latest(seq, timeout=0.5)
            if frame is None:
                with self._lock:
                    self._send(self.tracker.tick())
                continue
            if tiles is None:
                h, w = frame.shape[:2]
                tiles = frame_tiles(self.cfg, w, h)
                print(f"[cam] frame {w}x{h}, {len(tiles)} tile(s): {tiles}", flush=True)
            dets = drop_ignored(self.detector.detect(frame, tiles), self.cfg.ignore,
                                self.cfg.ignore_iou, self.cfg.ignore_masks,
                                self.cfg.ignore_inside)
            with self._lock:
                if self.enabled:
                    self._send(self.tracker.update(dets))
                have = self.tracker.have
            if have != last_have:
                print(f"[tgt] {'acquired' if have else 'cleared'}"
                      + (f"  x={self.tracker.sx:+.2f} y={self.tracker.sy:+.2f}" if have else ""),
                      flush=True)
                last_have = have
            if self.watch:
                m = self.tracker.mapper
                for d in dets:
                    u, v = (d[1] + d[3]) / 2, d[4]
                    box = ", ".join(f"{c:.2f}" for c in d[1:])
                    print(f"  person {d[0]:.2f}  u={u:.3f} v={v:.3f}  "
                          f"bearing={m.bearing(u):+5.1f}°  -> x={m.gaze_x(u):+.2f} "
                          f"y={m.gaze_y(v):+.2f}  box {box}", flush=True)
            frames += 1
            if time.monotonic() - t_log >= 60:
                print(f"[cam] {frames / (time.monotonic() - t_log):.1f} detections/s", flush=True)
                frames, t_log = 0, time.monotonic()
            # pace to detect_fps (the grabber keeps only the newest frame)
            self._stop.wait(max(0.0, 1.0 / self.cfg.detect_fps - (time.monotonic() - t0)))
        self.grabber.stop()

    def stop(self):
        self._stop.set()

    def close(self):
        with self._lock:
            self._send(self.tracker.reset())
        if self.client is None:
            return
        if self.status:
            self.status.go_offline()
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass


def frame_detail(frame):
    """How much is in a picture (pixel spread): ~0 for a blank or grey frame,
    tens for a real one. Sampled, so cheap on 4K."""
    return float(frame[::8, ::8].std())


def snapshot(cfg, detector, path):
    """One frame with the ROI, tiles and detections drawn (for setting up)."""
    import cv2
    import numpy as np
    g = FrameGrabber(cfg.url)
    g.start()
    # longer than ffmpeg's ~30 s open timeout, so its error is printed first
    deadline = time.monotonic() + 45
    seq, frame = 0, None
    while frame is None and time.monotonic() < deadline:
        seq, frame = g.latest(seq, timeout=1.0)
    # the first frames can be grey until the next keyframe (H.265 joined
    # mid-GOP: "Could not find ref with POC"): from 2 s in, wait for detail
    settle = time.monotonic() + 2.0
    while frame is not None and time.monotonic() < deadline:
        if time.monotonic() >= settle and frame_detail(frame) >= 8:
            break
        seq, newer = g.latest(seq, timeout=1.0)
        frame = frame if newer is None else newer
    g.stop()
    if frame is None:
        raise SystemExit("!! no frame from the camera in 45 s — check [camera] url / CAMERA_URL")
    if frame_detail(frame) < 8:
        print("[snap] !! the picture is still blank/grey: the stream isn't decoding "
              "(try the camera's H.264 or a lower-resolution stream)", flush=True)
    h, w = frame.shape[:2]
    tiles = frame_tiles(cfg, w, h)
    n_main = len(tile_boxes(cfg.roi, w, h, cfg.tiles))
    img = frame.copy()
    k = w / 640                                     # drawing scale: readable at 4K too
    thin, thick = max(1, round(k)), max(2, round(2 * k))
    for i, (x1, y1, x2, y2) in enumerate(tiles):    # roi tiles blue, far_roi tiles cyan
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 160, 0) if i < n_main else (255, 255, 0),
                      thin)
    r = cfg.roi
    cv2.rectangle(img, (int(r[0] * w), int(r[1] * h)), (int(r[2] * w), int(r[3] * h)),
                  (0, 200, 255), thick)
    for b in cfg.ignore:                            # ignore boxes: magenta
        cv2.rectangle(img, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h)),
                      (255, 0, 255), thin)
    for poly in cfg.ignore_masks:                   # ignore masks: magenta too
        pts = np.array([(int(x * w), int(y * h)) for x, y in poly], np.int32)
        cv2.polylines(img, [pts], True, (255, 0, 255), thin)
    m = Mapper(cfg)
    # look a bit below min_score too, so near misses show (orange)
    low = max(0.2, cfg.min_score - 0.2)
    dets = detector.detect(frame, tiles, min_score=low) if detector else []
    kept = drop_ignored(dets, cfg.ignore, cfg.ignore_iou, cfg.ignore_masks, cfg.ignore_inside)
    n_people = 0
    for d in dets:
        u = (d[1] + d[3]) / 2
        if d not in kept:
            what, colour = "ignored", (255, 0, 255)
        elif d[0] < cfg.min_score:
            what, colour = "near miss", (0, 140, 255)
        else:
            what, colour = "person", (0, 0, 255)
            n_people += 1
        p1, p2 = (int(d[1] * w), int(d[2] * h)), (int(d[3] * w), int(d[4] * h))
        cv2.rectangle(img, p1, p2, colour, thick)
        label = f"{d[0]:.2f} u={u:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k, thin)
        cv2.putText(img, label, (min(p1[0], w - tw), max(th + thin, p1[1] - int(4 * k))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k, colour, thin)
        box = ", ".join(f"{v:.2f}" for v in d[1:])
        print(f"[snap] {what:9} score {d[0]:.2f}  u={u:.2f} x={m.gaze_x(u):+.2f}  "
              f"box {box}", flush=True)
    for i in range(1, 10):                          # u ruler along the bottom
        x = int(i / 10 * w)
        cv2.line(img, (x, h - int(14 * k)), (x, h), (255, 255, 255), thin)
        cv2.putText(img, f"{i / 10:.1f}", (x + int(3 * k), h - int(3 * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * k, (255, 255, 255), thin)
    cv2.imwrite(path, img)
    print(f"[snap] {w}x{h}, detail {frame_detail(frame):.0f}, {len(tiles)} tile(s), "
          f"{n_people} person(s) -> {path}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.environ.get(
        "CAMTRACKER_CONFIG", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "camtracker.ini")))
    ap.add_argument("--snapshot", metavar="JPG", help="save one annotated frame and exit")
    ap.add_argument("--watch", action="store_true",
                    help="track regardless of the show command and print every detection")
    ap.add_argument("--no-publish", action="store_true", help="don't send targets to the eyes")
    args = ap.parse_args(argv)

    cfg = Config(args.config)
    if not cfg.url:
        raise SystemExit("!! no camera: set [camera] url or the CAMERA_URL environment variable")
    detector = CoralDetector(cfg)
    if args.snapshot:
        snapshot(cfg, detector, args.snapshot)
        return

    t = CamTracker(cfg, detector, publish=not args.no_publish, watch=args.watch)

    def _sigterm(*_):
        t.stop()
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)
    t.connect()
    print(f"[cam] {'watching' if args.watch else 'ready'} — target {cfg.topic}, "
          f"following {cfg.command_topic if cfg.follow_command else 'nothing (always on)'}",
          flush=True)
    try:
        t.run()
    finally:
        t.close()


if __name__ == "__main__":
    sys.exit(main())
