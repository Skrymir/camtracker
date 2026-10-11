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

Only the run/IO parts need OpenCV and pycoral; the mapping and tracking logic
(Mapper, pick, Tracker) is pure Python and unit-tested on its own.
"""

import argparse
import configparser
import json
import math
import os
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
        self.roi = _floats(g("camera", "roi"), 4, "roi")
        x1, y1, x2, y2 = self.roi
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"roi must be x1,y1,x2,y2 inside 0..1 with x1<x2, y1<y2: {self.roi}")
        t = g("camera", "tiles").lower()
        self.tiles = 0 if t in ("", "auto") else max(1, int(t))

        self.model = g("detector", "model")
        self.device = g("detector", "device") or "auto"
        self.person_id = cp.getint("detector", "person_id")
        self.min_score = cp.getfloat("detector", "min_score")
        self.min_height = cp.getfloat("detector", "min_height")

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
        return -_clamp((v - c.y_far) / (c.y_near - c.y_far), 0.0, 1.0)

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


class CoralDetector:
    """SSD-style detection model on an Edge TPU (pycoral)."""

    def __init__(self, cfg):
        from pycoral.adapters import common, detect
        from pycoral.utils.edgetpu import list_edge_tpus, make_interpreter
        self._common, self._detect = common, detect
        self.cfg = cfg
        if cfg.device.lower() != "auto":
            candidates = [cfg.device]
        else:
            tpus = list_edge_tpus()
            print(f"[tpu] found {len(tpus)} Edge TPU(s): {tpus}", flush=True)
            n_usb = sum(1 for t in tpus if t.get("type") == "usb")
            n_pci = sum(1 for t in tpus if t.get("type") == "pci")
            candidates = [f"usb:{i}" for i in range(n_usb)] + [f"pci:{i}" for i in range(n_pci)]
        err = None
        for dev in candidates:
            try:
                interp = make_interpreter(cfg.model, device=dev)
                interp.allocate_tensors()
            except Exception as e:                  # busy (Frigate) or absent
                print(f"[tpu] {dev}: {e}", flush=True)
                err = e
                continue
            self.interp, self.device = interp, dev
            self.size = common.input_size(interp)
            print(f"[tpu] using {dev}, model input {self.size[0]}x{self.size[1]}", flush=True)
            return
        raise RuntimeError(f"no usable Edge TPU (tried {candidates or 'none found'}): {err}")

    def detect(self, frame, tiles):
        """People in frame (BGR) within the tile rectangles ->
        [(score, x1, y1, x2, y2)] as fractions of the whole frame."""
        import cv2
        fh, fw = frame.shape[:2]
        w, h = self.size
        out = []
        for (tx1, ty1, tx2, ty2) in tiles:
            crop = frame[ty1:ty2, tx1:tx2]
            rgb = cv2.cvtColor(cv2.resize(crop, (w, h)), cv2.COLOR_BGR2RGB)
            self._common.set_input(self.interp, rgb)
            self.interp.invoke()
            sx, sy = (tx2 - tx1) / w, (ty2 - ty1) / h
            for o in self._detect.get_objects(self.interp, self.cfg.min_score):
                if o.id != self.cfg.person_id:
                    continue
                b = o.bbox
                x1, y1 = (tx1 + b.xmin * sx) / fw, (ty1 + b.ymin * sy) / fh
                x2, y2 = (tx1 + b.xmax * sx) / fw, (ty1 + b.ymax * sy) / fh
                if (y2 - y1) >= self.cfg.min_height:
                    out.append((float(o.score), x1, y1, x2, y2))
        return nms(out)


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
                tiles = tile_boxes(self.cfg.roi, w, h, self.cfg.tiles)
                print(f"[cam] frame {w}x{h}, {len(tiles)} tile(s): {tiles}", flush=True)
            dets = self.detector.detect(frame, tiles)
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
                    print(f"  person {d[0]:.2f}  u={u:.3f} v={v:.3f}  "
                          f"bearing={m.bearing(u):+5.1f}°  -> x={m.gaze_x(u):+.2f} "
                          f"y={m.gaze_y(v):+.2f}", flush=True)
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


def snapshot(cfg, detector, path):
    """One frame with the ROI, tiles and detections drawn (for setting up)."""
    import cv2
    g = FrameGrabber(cfg.url)
    g.start()
    deadline = time.monotonic() + 20
    seq, frame = 0, None
    while frame is None and time.monotonic() < deadline:
        seq, frame = g.latest(seq, timeout=1.0)
    g.stop()
    if frame is None:
        raise SystemExit("!! no frame from the camera in 20 s — check [camera] url / CAMERA_URL")
    h, w = frame.shape[:2]
    tiles = tile_boxes(cfg.roi, w, h, cfg.tiles)
    img = frame.copy()
    for (x1, y1, x2, y2) in tiles:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 160, 0), 1)
    r = cfg.roi
    cv2.rectangle(img, (int(r[0] * w), int(r[1] * h)), (int(r[2] * w), int(r[3] * h)),
                  (0, 200, 255), 2)
    m = Mapper(cfg)
    dets = detector.detect(frame, tiles) if detector else []
    for d in dets:
        p1, p2 = (int(d[1] * w), int(d[2] * h)), (int(d[3] * w), int(d[4] * h))
        cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
        u = (d[1] + d[3]) / 2
        cv2.putText(img, f"u={u:.2f} x={m.gaze_x(u):+.2f}", (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    for i in range(1, 10):                          # u ruler along the bottom
        x = int(i / 10 * w)
        cv2.line(img, (x, h - 12), (x, h), (255, 255, 255), 1)
        cv2.putText(img, f"{i / 10:.1f}", (x + 2, h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 255), 1)
    cv2.imwrite(path, img)
    print(f"[snap] {w}x{h}, {len(tiles)} tile(s), {len(dets)} person(s) -> {path}", flush=True)


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
