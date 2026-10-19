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

While running, http://<server>:8080/ is a live view: the camera with the same
drawing as --snapshot, plus who is being followed ([web] port; 0 = off).

Only the run/IO parts need OpenCV and tflite_runtime; the mapping, tracking
and detection-parsing logic (Mapper, pick, Tracker, ssd_people) is pure Python
and unit-tested on its own.
"""

import argparse
import collections
import configparser
import glob
import json
import math
import os
import re
import shutil
import signal
import socket
import sys
import threading
import time

import status
import web

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
    "web": {
        # the live view (http://<server>:<port>/): the camera with what the
        # tracker sees drawn on it. 0 = off.
        "port": "8080",
        # the stream's frame rate and width (it's scaled down, to save CPU)
        "fps": "5",
        "width": "1280",
        # edit roi, far_roi and the ignore masks on the page (saved here)
        "edit": "true",
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


# -- zones: roi, far_roi and ignore_masks, as the live view edits them --------
def _num(v):
    return f"{round(float(v), 3):g}"


def format_region(r):
    return ", ".join(_num(v) for v in r) if r else ""


def format_masks(masks):
    """[[(x, y), ...], ...] -> Frigate's form, "x,y,x,y,...; ..."."""
    return "; ".join(",".join(_num(c) for p in m for c in p) for m in masks)


def zones(cfg):
    """The zones as JSON for the editor (ignore boxes are shown, not edited)."""
    return {"roi": list(cfg.roi), "far_roi": list(cfg.far_roi) if cfg.far_roi else None,
            "ignore_masks": [[list(p) for p in m] for m in cfg.ignore_masks],
            "ignore": [list(b) for b in cfg.ignore]}


def parse_zones(data):
    """The editor's JSON -> (roi, far_roi or None, masks), checked as if read
    from the config (so what's saved always loads). ValueError if not."""
    try:
        roi = _region(format_region(data["roi"]), "roi")
        far = data.get("far_roi")
        far = _region(format_region(far), "far_roi") if far else None
        masks = parse_masks(format_masks(data.get("ignore_masks") or []))
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"bad zones: {e!r}") from None
    return roi, far, masks


def set_ini_values(text, values):
    """Set keys in an ini file's text, keeping everything else (comments,
    order, other keys) as it is. values: {section: {key: value}}. A key's
    indented continuation lines go with it; a missing key is added at the end
    of its section, a missing section at the end of the file."""
    lines = text.splitlines()
    out, section, skipping, done = [], None, False, set()

    def finish(sec):                             # add the section's missing keys
        missing = [(k, v) for k, v in values.get(sec, {}).items() if (sec, k) not in done]
        if not missing:
            return
        at = len(out)
        while at > 0 and not out[at - 1].strip():
            at -= 1
        out[at:at] = [f"{k} = {v}".rstrip() for k, v in missing]
        done.update((sec, k) for k, _ in missing)

    for line in lines:
        if skipping and line[:1].isspace() and line.strip():
            continue                             # the old value's continuation
        skipping = False
        m = re.match(r"\[([^\]]+)\]\s*$", line)
        if m:
            finish(section)
            section = m.group(1).strip()
            out.append(line)
            continue
        m = re.match(r"([^\s#;=:][^=:]*?)\s*[=:]", line)
        if m and section in values and m.group(1) in values[section]:
            key = m.group(1)
            out.append(f"{key} = {values[section][key]}".rstrip())
            done.add((section, key))
            skipping = True
            continue
        out.append(line)
    finish(section)
    for sec in values:
        if any((sec, k) not in done for k in values[sec]):
            out += ["", f"[{sec}]"] + [f"{k} = {v}".rstrip() for k, v in values[sec].items()]
    return "\n".join(out) + "\n"


def write_zones(path, roi, far_roi, masks):
    """Save the zones into the config file (a copy of the old one goes to
    <path>.bak), replacing it in one go so a crash can't leave half a file."""
    old = ""
    if os.path.exists(path):
        with open(path) as f:
            old = f.read()
        with open(path + ".bak", "w") as f:
            f.write(old)
    new = set_ini_values(old, {"camera": {"roi": format_region(roi),
                                          "far_roi": format_region(far_roi)},
                               "detector": {"ignore_masks": format_masks(masks)}})
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(new)
    if old:
        shutil.copymode(path, tmp)
    os.replace(tmp, path)


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

        self.web_port = cp.getint("web", "port")
        self.web_fps = cp.getfloat("web", "fps")
        self.web_width = cp.getint("web", "width")
        self.web_edit = cp.getboolean("web", "edit")

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


def merge(dets):
    """One box per person from every tile's detections."""
    return drop_contained(nms(dets))


def split_detections(raw, cfg):
    """All tiles' detections (from detect_raw) -> (people, near, ignored).
    people are exactly what tracking sees (raw at min_score and up, merged,
    decorations dropped); near: merged boxes just under min_score, if raw
    went lower; ignored: either kind matching an ignore box or mask."""
    def keep(ds):
        return drop_ignored(ds, cfg.ignore, cfg.ignore_iou, cfg.ignore_masks,
                            cfg.ignore_inside)
    found = merge([d for d in raw if d[0] >= cfg.min_score])
    below = [d for d in merge(raw) if d[0] < cfg.min_score]
    people, near = keep(found), keep(below)
    ignored = [d for d in found + below if d not in people and d not in near]
    return people, near, ignored


def followed(people, last_u):
    """The person being followed: the box whose centre is nearest last_u."""
    if not people or last_u is None:
        return None
    return min(people, key=lambda d: abs((d[1] + d[3]) / 2 - last_u))


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
        return merge(self.detect_raw(frame, tiles, min_score))

    def detect_raw(self, frame, tiles, min_score=None):
        """Every tile's people, before duplicates are merged."""
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
        return out


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

    def __init__(self, cfg, detector=None, grabber=None, publish=True, watch=False, view=None):
        self.cfg = cfg
        self.detector = detector
        self.view = view
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
        self._retile = False
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

    def set_zones(self, roi, far_roi, masks):
        """New zones (the live view's editor), used from the next frame."""
        with self._lock:
            self.cfg.roi, self.cfg.far_roi, self.cfg.ignore_masks = roi, far_roi, masks
            self._retile = True
        print(f"[cfg] zones: roi {format_region(roi)}, far_roi {format_region(far_roi) or '-'}, "
              f"{len(masks)} mask(s)", flush=True)

    # -- main loop --
    def run(self):
        tiles, seq, frames, t_log = None, 0, 0, time.monotonic()
        last_have, recent = False, collections.deque(maxlen=30)    # detection times
        while not self._stop.is_set():
            with self._lock:
                enabled = self.enabled
            viewing = self.view is not None and self.view.wanted
            if not (enabled or viewing):
                self.grabber.stop()
                tiles = None
                recent.clear()
                self._stop.wait(0.5)
                continue
            self.grabber.start()
            t0 = time.monotonic()
            seq, frame = self.grabber.latest(seq, timeout=0.5)
            if frame is None:
                with self._lock:
                    self._send(self.tracker.tick())
                continue
            if tiles is None or self._retile:
                self._retile = False
                h, w = frame.shape[:2]
                tiles = frame_tiles(self.cfg, w, h)
                print(f"[cam] frame {w}x{h}, {len(tiles)} tile(s): {tiles}", flush=True)
            # someone on the live view: look lower too, so near misses show
            # (people, what tracking sees, is the same either way)
            raw = self.detector.detect_raw(frame, tiles,
                                           near_miss_score(self.cfg) if viewing else None)
            dets, near, ignored = split_detections(raw, self.cfg)
            with self._lock:
                if self.enabled:
                    self._send(self.tracker.update(dets))
                have, sx, sy = self.tracker.have, self.tracker.sx, self.tracker.sy
                last_u = self.tracker.last_u if have else None
            recent.append(time.monotonic())
            if viewing:
                rate = (len(recent) - 1) / max(1e-3, recent[-1] - recent[0])
                self.view.offer({"frame": frame, "people": dets, "near": near,
                                 "ignored": ignored, "follow": followed(dets, last_u),
                                 "status": view_status(enabled, have, sx, sy, rate)})
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


# colours (BGR)
ROI_C, TILE_C, FAR_TILE_C = (0, 200, 255), (255, 160, 0), (255, 255, 0)
PERSON_C, FOLLOW_C, NEAR_C, IGNORED_C = (0, 0, 255), (0, 220, 0), (0, 140, 255), (255, 0, 255)


def draw_overlay(img, cfg, frame_w, frame_h, people=(), near=(), ignored=(), follow=None,
                 status_line=None):
    """Draw the ROI, tiles, ignore boxes/masks, detections and the u ruler on
    img (the frame, or a smaller copy of it), in place. Returns
    [(what, det)] for each detection drawn."""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    k = w / 640                                     # drawing scale: readable at 4K too
    thin, thick = max(1, round(k)), max(2, round(2 * k))
    font = cv2.FONT_HERSHEY_SIMPLEX
    sx, sy = w / frame_w, h / frame_h
    n_main = len(tile_boxes(cfg.roi, frame_w, frame_h, cfg.tiles))
    for i, (x1, y1, x2, y2) in enumerate(frame_tiles(cfg, frame_w, frame_h)):
        cv2.rectangle(img, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)),
                      TILE_C if i < n_main else FAR_TILE_C, thin)

    def px(x, y):
        return int(x * w), int(y * h)

    r = cfg.roi
    cv2.rectangle(img, px(r[0], r[1]), px(r[2], r[3]), ROI_C, thick)
    for b in cfg.ignore:
        cv2.rectangle(img, px(b[0], b[1]), px(b[2], b[3]), IGNORED_C, thin)
    for poly in cfg.ignore_masks:
        cv2.polylines(img, [np.array([px(x, y) for x, y in poly], np.int32)], True,
                      IGNORED_C, thin)
    drawn = ([("ignored", d) for d in ignored] + [("near miss", d) for d in near]
             + [("following" if d == follow else "person", d) for d in people])
    colours = {"ignored": IGNORED_C, "near miss": NEAR_C, "person": PERSON_C,
               "following": FOLLOW_C}
    for what, d in drawn:
        p1, p2 = px(d[1], d[2]), px(d[3], d[4])
        cv2.rectangle(img, p1, p2, colours[what], thick)
        label = f"{d[0]:.2f} u={(d[1] + d[3]) / 2:.2f}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.45 * k, thin)
        cv2.putText(img, label, (min(p1[0], w - tw), max(th + thin, p1[1] - int(4 * k))),
                    font, 0.45 * k, colours[what], thin)
    for i in range(1, 10):                          # u ruler along the bottom
        x = int(i / 10 * w)
        cv2.line(img, (x, h - int(14 * k)), (x, h), (255, 255, 255), thin)
        cv2.putText(img, f"{i / 10:.1f}", (x + int(3 * k), h - int(3 * k)),
                    font, 0.4 * k, (255, 255, 255), thin)
    if status_line:                                 # top left, on a dark band
        (tw, th), base = cv2.getTextSize(status_line, font, 0.5 * k, thin)
        cv2.rectangle(img, (0, 0), (tw + int(12 * k), th + base + int(10 * k)), (0, 0, 0), -1)
        cv2.putText(img, status_line, (int(6 * k), th + int(5 * k)), font, 0.5 * k,
                    (255, 255, 255), thin)
    return drawn


def view_status(enabled, have, sx, sy, rate):
    """The live view's top line."""
    show = "show on" if enabled else "show off (not following)"
    who = f"following x={sx:+.2f} y={sy:+.2f}" if have else "nobody followed"
    return f"{show} | {who} | {rate:.0f} det/s" if enabled else f"{show} | {rate:.0f} det/s"


def render_view(cfg, state, width=None, overlay=True):
    """A live view frame (state from the loop) as JPEG, scaled to width, with
    what the tracker sees drawn on it (unless overlay is False)."""
    import cv2
    frame = state["frame"]
    fh, fw = frame.shape[:2]
    if width and fw > width:
        img = cv2.resize(frame, (width, round(fh * width / fw)), interpolation=cv2.INTER_AREA)
    else:
        img = frame.copy()
    if overlay:
        draw_overlay(img, cfg, fw, fh, state["people"], state["near"], state["ignored"],
                     state["follow"], state["status"])
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def near_miss_score(cfg):
    """How low the snapshot and live view look, so near misses show."""
    return max(0.2, cfg.min_score - 0.2)


def snapshot(cfg, detector, path):
    """One frame with the ROI, tiles and detections drawn (for setting up)."""
    import cv2
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
    raw = detector.detect_raw(frame, tiles, near_miss_score(cfg)) if detector else []
    people, near, ignored = split_detections(raw, cfg)
    img = frame.copy()
    m = Mapper(cfg)
    for what, d in draw_overlay(img, cfg, w, h, people, near, ignored):
        u = (d[1] + d[3]) / 2
        box = ", ".join(f"{v:.2f}" for v in d[1:])
        print(f"[snap] {what:9} score {d[0]:.2f}  u={u:.2f} x={m.gaze_x(u):+.2f}  "
              f"box {box}", flush=True)
    cv2.imwrite(path, img)
    print(f"[snap] {w}x{h}, detail {frame_detail(frame):.0f}, {len(tiles)} tile(s), "
          f"{len(people)} person(s) -> {path}", flush=True)


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

    view = None
    if cfg.web_port:
        view = web.LiveView(lambda state, width, overlay=True: render_view(cfg, state, width,
                                                                            overlay),
                            fps=cfg.web_fps, width=cfg.web_width, get_zones=lambda: zones(cfg))
        try:
            web.serve(view, cfg.web_port)
            print(f"[web] live view on port {cfg.web_port}: "
                  f"http://<this server>:{cfg.web_port}/", flush=True)
        except OSError as e:                        # port taken: track anyway
            print(f"[web] !! can't serve the live view on port {cfg.web_port}: {e}",
                  flush=True)
            view = None
    t = CamTracker(cfg, detector, publish=not args.no_publish, watch=args.watch, view=view)
    if view and cfg.web_edit:
        def save(data):
            roi, far, masks = parse_zones(data)
            write_zones(args.config, roi, far, masks)
            t.set_zones(roi, far, masks)
            return zones(cfg)
        view.save_zones = save

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
