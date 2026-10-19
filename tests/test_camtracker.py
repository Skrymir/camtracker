"""camtracker.py: frame position -> gaze, who to follow, and the target /
clear messages it sends. Pure logic: no OpenCV, Coral or broker needed."""
import math

import pytest

import camtracker as CT
import status


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def cfg(**kw):
    c = CT.Config(None)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def box(u, h=0.3, bottom=0.8, score=0.9, w=0.05):
    return (score, u - w / 2, bottom - h, u + w / 2, bottom)


# -- config ------------------------------------------------------------------
def test_template_loads_and_matches_defaults():
    c = CT.Config("camtracker.ini")
    d = CT.Config(None)
    for k in ("roi", "tiles", "hfov", "gaze_max", "points", "smooth", "hold", "topic",
              "command_topic", "status_base", "status_node", "min_score", "model"):
        assert getattr(c, k) == getattr(d, k), k
    assert c.status_node == "camera"


def test_secrets_from_environment(monkeypatch):
    monkeypatch.setenv("CAMERA_URL", "rtsp://u:p@cam/sub")
    monkeypatch.setenv("MQTT_USERNAME", "eyes")
    monkeypatch.setenv("MQTT_PASSWORD", "pw")
    c = CT.Config("camtracker.ini")
    assert (c.url, c.mqtt_user, c.mqtt_pass) == ("rtsp://u:p@cam/sub", "eyes", "pw")


def test_points_parsing():
    assert CT.parse_points("0.9:-1, 0.1:1;0.5:0") == [(0.1, 1.0), (0.5, 0.0), (0.9, -1.0)]
    assert CT.parse_points("") == []
    for bad in ("0.5", "0.5:0", "0.5:0, 0.5:1"):
        with pytest.raises(ValueError):
            CT.parse_points(bad)


def test_bad_roi_rejected(tmp_path):
    p = tmp_path / "c.ini"
    p.write_text("[camera]\nroi = 0.5, 0, 0.4, 1\n")
    with pytest.raises(ValueError):
        CT.Config(str(p))


# -- mapping -----------------------------------------------------------------
def test_centre_is_straight_ahead():
    m = CT.Mapper(cfg())
    assert m.gaze_x(0.5) == pytest.approx(0.0)


def test_lens_model_bearing_and_sign():
    m = CT.Mapper(cfg(hfov=90.0, gaze_max=45.0))
    # right edge of a 90 deg lens = 45 deg right of the camera; the audience
    # sees that as their left, so the pupil goes screen-left: -1
    assert m.bearing(1.0) == pytest.approx(45.0)
    assert m.gaze_x(1.0) == pytest.approx(-1.0)
    assert m.gaze_x(0.0) == pytest.approx(1.0)
    # a quarter of the way in is NOT a quarter of the angle (perspective)
    u = 0.75
    assert m.bearing(u) == pytest.approx(math.degrees(math.atan(0.5)))


def test_clamps_beyond_gaze_max_and_invert():
    m = CT.Mapper(cfg(hfov=100.0, gaze_max=20.0))
    assert m.gaze_x(0.98) == -1.0
    m.cfg.invert_x = True
    assert m.gaze_x(0.98) == 1.0


def test_calibration_points_interpolate_and_hold_ends():
    m = CT.Mapper(cfg(points=CT.parse_points("0.1:0.9, 0.5:0, 0.9:-0.9")))
    assert m.gaze_x(0.3) == pytest.approx(0.45)
    assert m.gaze_x(0.7) == pytest.approx(-0.45)
    assert m.gaze_x(0.0) == pytest.approx(0.9)
    assert m.gaze_x(1.0) == pytest.approx(-0.9)


def test_look_down_from_feet():
    m = CT.Mapper(cfg(y_from_feet=True, y_far=0.5, y_near=1.0))
    assert m.gaze_y(0.4) == 0.0
    assert m.gaze_y(0.75) == pytest.approx(-0.5)
    assert m.gaze_y(1.0) == -1.0
    assert CT.Mapper(cfg()).gaze_y(1.0) == 0.0          # off by default


def test_look_down_curve_moves_early_keeps_the_ends():
    m = CT.Mapper(cfg(y_from_feet=True, y_far=0.3, y_near=1.0, y_curve=0.5))
    assert m.gaze_y(0.3) == 0.0 and m.gaze_y(1.0) == -1.0
    assert m.gaze_y(0.475) == pytest.approx(-0.5)     # a quarter of the way: half the look
    assert m.gaze_y(0.65) == pytest.approx(-(0.5 ** 0.5))


# -- who to look at ----------------------------------------------------------
def test_pick_nearest_person_then_keeps_following():
    far, near = box(0.2, h=0.1), box(0.7, h=0.4)
    assert CT.pick([far, near])[0] == pytest.approx(0.7)
    # already following the far one: stay on them, even though someone nearer is in view
    assert CT.pick([far, near], last_u=0.22)[0] == pytest.approx(0.2)
    # the tracked person vanished: jump to the nearest
    assert CT.pick([near], last_u=0.2, max_jump=0.25)[0] == pytest.approx(0.7)
    assert CT.pick([]) is None


def test_nms_merges_tile_overlap():
    a = (0.9, 0.40, 0.2, 0.50, 0.8)
    b = (0.7, 0.41, 0.21, 0.51, 0.8)
    c = (0.8, 0.70, 0.2, 0.80, 0.8)
    assert CT.nms([b, a, c]) == [a, c]


def test_tiles_cover_the_roi_roughly_square():
    tiles = CT.tile_boxes((0, 0.4, 1, 0.7), 1280, 720)
    assert len(tiles) == 6                         # 1280 x 216 strip
    assert tiles[0][0] == 0 and tiles[-1][2] == 1280
    for a, b in zip(tiles, tiles[1:]):
        assert b[0] < a[2]                          # overlapping, no gaps
    assert CT.tile_boxes((0, 0, 1, 1), 640, 480) == [(0, 0, 640, 480)]
    assert len(CT.tile_boxes((0, 0, 1, 1), 640, 480, tiles=3)) == 3


# -- messages ----------------------------------------------------------------
def make(**kw):
    clk = Clock()
    c = cfg(**dict(dict(smooth=0.5, min_interval=0.1, heartbeat=0.75, lost_after=0.8), **kw))
    return CT.Tracker(c, clock=clk), clk


def test_acquire_snaps_then_smooths_and_rate_limits():
    t, clk = make()
    (m,) = t.update([box(0.5)])
    assert m[0] == "target" and m[1] == pytest.approx(0.0)
    clk.t += 0.05
    assert t.update([box(0.7)]) == []                # rate limit, but still smoothing
    assert t.sx == pytest.approx(0.5 * t.mapper.gaze_x(0.7))
    clk.t += 0.06
    (m,) = t.update([box(0.7)])
    assert m[0] == "target" and m[1] == pytest.approx(0.75 * t.mapper.gaze_x(0.7))


def test_brief_misses_hold_then_clear_once():
    t, clk = make()
    t.update([box(0.4)])
    clk.t += 0.5
    assert t.update([]) == []                        # a missed frame: keep looking
    assert t.have
    clk.t += 0.4                                     # 0.9 s since last seen
    assert t.update([]) == [("clear",)]
    assert t.tick() == [] and not t.have


def test_brief_miss_does_not_flick_to_someone_else():
    t, clk = make(max_jump=0.25)
    t.update([box(0.2, h=0.2)])                      # following A on the left
    clk.t += 0.1
    t.update([box(0.8, h=0.4)])                      # A missed once, B (nearer) in view
    assert t.last_u == pytest.approx(0.2)
    clk.t += 0.1
    t.update([box(0.22, h=0.2), box(0.8, h=0.4)])    # A is back: still on A
    assert t.last_u == pytest.approx(0.22)
    clk.t += 0.9                                      # A gone for good: move to B, no clear
    assert [m[0] for m in t.update([box(0.8, h=0.4)])] == ["target"]
    assert t.last_u == pytest.approx(0.8) and t.have


def test_heartbeat_for_someone_standing_still():
    t, clk = make()
    t.update([box(0.4)])
    clk.t += 0.3
    assert t.tick() == []
    t.update([box(0.4)])                             # still seen, rate-limited send
    clk.t += 0.76
    t.last_seen = clk.t                              # seen, but no new send since
    assert [m[0] for m in t.tick()] == ["target"]


def test_reset_only_clears_a_held_target():
    t, _ = make()
    assert t.reset() == []
    t.update([box(0.4)])
    assert t.reset() == [("clear",)]


# -- service glue (no broker) --------------------------------------------------
class FakeClient:
    def __init__(self):
        self.sent = []

    def publish(self, topic, payload, qos=0, retain=False):
        self.sent.append((topic, payload))


class FakeGrabber:
    ok = False

    def stop(self):
        pass


class Msg:
    def __init__(self, payload):
        self.payload = payload.encode()


def test_command_on_off_and_clear_on_off():
    c = cfg(status_enabled=False)
    s = CT.CamTracker(c, grabber=FakeGrabber())
    s.client = FakeClient()
    assert not s.enabled                              # follows the show, starts quiet
    s._on_command(None, None, Msg('{"state": "ON"}'))
    assert s.enabled
    s._send(s.tracker.update([box(0.5)]))
    s._on_command(None, None, Msg("off"))
    assert not s.enabled
    topics = [t for t, _ in s.client.sent]
    assert set(topics) == {c.topic}
    assert s.client.sent[-1][1] == "clear"
    s._on_command(None, None, Msg("banana"))          # unknown words are ignored
    assert not s.enabled


def test_status_discovery_uses_camera_node():
    d = status.discovery("camera", "house/status", "homeassistant")
    ids = {cfg_["default_entity_id"] for cfg_ in d.values()}
    assert "binary_sensor.camtracker_tracking" in ids
    assert all(t.startswith("homeassistant/") and "camtracker_camera_" in t for t in d)


def test_status_templates_render():
    jinja2 = pytest.importorskip("jinja2")
    import json
    env = jinja2.Environment()
    env.filters["tojson"] = json.dumps
    d = status.discovery()
    snap = json.dumps({"tracking": True, "enabled": True, "stream_ok": False,
                       "detector": "usb:1", "host": "tower", "version": "x"})
    want = {"camtracker_camera_tracking": "ON", "camtracker_camera_stream": "OFF",
            "camtracker_camera_detector": "usb:1"}
    for uid, value in want.items():
        (c,) = [c for c in d.values() if c["unique_id"] == uid]
        out = env.from_string(c["value_template"]).render(value_json=json.loads(snap))
        assert out.strip() == value, uid
    (det,) = [c for c in d.values() if c["unique_id"] == "camtracker_camera_detector"]
    attrs = env.from_string(det["json_attributes_template"]).render(value_json=json.loads(snap))
    assert json.loads(attrs) == {"enabled": True, "host": "tower", "version": "x"}


# -- detector output (no Coral) -----------------------------------------------
def test_ssd_people_maps_tile_boxes_to_frame():
    c = cfg(person_id=0, min_score=0.5, min_height=0.04)
    boxes = [(0.1, 0.2, 0.9, 0.4),            # person, in the right half's tile
             (0.1, 0.5, 0.9, 0.6),            # a car (class 2)
             (0.1, 0.6, 0.9, 0.7),            # a person, too unsure
             (0.5, 0.1, 0.52, 0.2),           # a person, too short
             (0.0, 0.0, 1.0, 1.0)]            # past count: ignored
    classes, scores = [0, 2, 0, 0, 0], [0.9, 0.9, 0.3, 0.9, 0.9]
    tile = (640, 0, 1280, 720)                # right half of a 1280x720 frame
    (p,) = CT.ssd_people(boxes, classes, scores, 4, tile, 1280, 720, c)
    assert p[0] == pytest.approx(0.9)
    assert p[1:] == pytest.approx((0.6, 0.1, 0.7, 0.9))


def test_find_tpus_counts_corals(tmp_path):
    usb, dev = tmp_path / "usb", tmp_path / "dev"
    for name, vid, pid in (("1-1", "1a6e", "089a"), ("1-2", "18d1", "9302"),
                           ("1-3", "046d", "c52b")):              # a mouse
        d = usb / name
        d.mkdir(parents=True)
        (d / "idVendor").write_text(vid + "\n")
        (d / "idProduct").write_text(pid + "\n")
    (usb / "usb1").mkdir()                                     # a hub: no ids
    dev.mkdir()
    (dev / "apex_0").write_text("")
    assert CT.find_tpus(str(usb), str(dev)) == ["usb:0", "usb:1", "pci:0"]
    assert CT.find_tpus(str(tmp_path / "none"), str(tmp_path / "none")) == []


def test_redact_hides_camera_login():
    assert CT.redact("rtsp://admin:p@ss@192.168.1.20:554/sub") == "rtsp://***@192.168.1.20:554/sub"
    assert CT.redact("rtsp://cam:554/sub") == "rtsp://cam:554/sub"


# -- decorations ----------------------------------------------------------------
def test_ignore_boxes_parse_and_validate(tmp_path):
    assert CT.parse_boxes("0.78, 0.13, 0.83, 0.29; 0.1,0.2,0.3,0.4") == [
        (0.78, 0.13, 0.83, 0.29), (0.1, 0.2, 0.3, 0.4)]
    assert CT.parse_boxes("") == []
    for bad in ("0.1, 0.2, 0.3", "0.5, 0.2, 0.4, 0.4"):
        with pytest.raises(ValueError):
            CT.parse_boxes(bad)
    p = tmp_path / "c.ini"
    p.write_text("[detector]\nignore = 0.78, 0.13, 0.83, 0.29\n")
    assert CT.Config(str(p)).ignore == [(0.78, 0.13, 0.83, 0.29)]


def test_decoration_dropped_person_walking_past_kept():
    ghost = (0.78, 0.13, 0.83, 0.29)
    seen_ghost = (0.8, 0.785, 0.135, 0.828, 0.288)      # jittered a little
    passer_by = (0.9, 0.76, 0.10, 0.80, 0.30)           # walking in front of it
    elsewhere = (0.9, 0.40, 0.10, 0.45, 0.30)
    assert CT.drop_ignored([seen_ghost, passer_by, elsewhere], [ghost]) == [passer_by, elsewhere]
    assert CT.drop_ignored([seen_ghost], []) == [seen_ghost]


def test_ignore_masks_parse_frigate_coordinates(tmp_path):
    frigate = "0.782,0.131,0.831,0.131,0.831,0.289,0.782,0.289"
    sq = [(0.782, 0.131), (0.831, 0.131), (0.831, 0.289), (0.782, 0.289)]
    assert CT.parse_masks(frigate) == [sq]
    assert CT.parse_masks(f"{frigate}; 0.1,0.1,0.2,0.1,0.15,0.2") == [
        sq, [(0.1, 0.1), (0.2, 0.1), (0.15, 0.2)]]
    assert CT.parse_masks("") == []
    for bad in ("0.1,0.1,0.2,0.2", "0.1,0.1,0.2,0.2,0.3", "120,40,300,40,300,200"):
        with pytest.raises(ValueError):
            CT.parse_masks(bad)
    p = tmp_path / "c.ini"
    p.write_text(f"[detector]\nignore_masks = {frigate}\n  0.1,0.1,0.2,0.1,0.15,0.2\n")
    assert len(CT.Config(str(p)).ignore_masks) == 2


def test_inside_fraction():
    sq = [(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]
    assert CT.inside_fraction((0.3, 0.3, 0.4, 0.4), sq) == pytest.approx(1.0)
    assert CT.inside_fraction((0.5, 0.3, 0.7, 0.4), sq) == pytest.approx(0.5)
    assert CT.inside_fraction((0.7, 0.7, 0.8, 0.8), sq) == 0.0
    tri = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]              # half the box, diagonally
    assert CT.inside_fraction((0.0, 0.0, 1.0, 1.0), tri) == pytest.approx(0.5)
    ell = [(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)]  # concave L
    assert CT.inside_fraction((0.5, 0.5, 1.0, 1.0), ell) == pytest.approx(0.0)
    assert CT.inside_fraction((0.0, 0.0, 1.0, 1.0), ell) == pytest.approx(0.75)


def test_decoration_in_mask_dropped_person_walking_past_kept():
    mask = [(0.77, 0.12), (0.84, 0.12), (0.84, 0.30), (0.77, 0.30)]  # drawn a bit loose
    seen_ghost = (0.8, 0.785, 0.135, 0.828, 0.288)
    passer_by = (0.9, 0.76, 0.10, 0.80, 0.40)
    elsewhere = (0.9, 0.40, 0.10, 0.45, 0.30)
    assert CT.drop_ignored([seen_ghost, passer_by, elsewhere], [], masks=[mask]) == [
        passer_by, elsewhere]


def test_ssd_people_min_score_override():
    c = cfg(person_id=0, min_score=0.5, min_height=0.0)
    args = ([(0.1, 0.1, 0.9, 0.2)], [0], [0.35], 1, (0, 0, 100, 100), 100, 100, c)
    assert CT.ssd_people(*args) == []
    assert len(CT.ssd_people(*args, min_score=0.3)) == 1


# -- near and far ----------------------------------------------------------------
def test_far_roi_adds_finer_tiles():
    c = cfg(roi=[0, 0.05, 1, 1], far_roi=None, tiles=0)
    assert len(CT.frame_tiles(c, 3840, 2160)) == 2
    c.far_roi = [0, 0.05, 1, 0.6]
    tiles = CT.frame_tiles(c, 3840, 2160)
    assert len(tiles) == 5
    assert all(t[3] <= int(0.6 * 2160) for t in tiles[2:])      # far tiles stay in the strip


def test_far_roi_config(tmp_path):
    p = tmp_path / "c.ini"
    p.write_text("[camera]\nroi = 0, 0.05, 1, 1\nfar_roi = 0, 0.05, 1, 0.6\n")
    assert CT.Config(str(p)).far_roi == [0, 0.05, 1, 0.6]
    assert CT.Config(None).far_roi is None
    p.write_text("[camera]\nfar_roi = 0, 0.7, 1, 0.6\n")
    with pytest.raises(ValueError):
        CT.Config(str(p))


def test_top_of_a_near_person_merged_into_them():
    near = (0.7, 0.40, 0.30, 0.50, 0.95)          # whole person, from the big tiles
    top = (0.9, 0.41, 0.30, 0.49, 0.60)           # the far strip's cut-off view of them
    far = (0.8, 0.10, 0.10, 0.13, 0.25)           # someone else, far away
    assert CT.drop_contained([top, near, far]) == [near, far]
    side_by_side = (0.8, 0.52, 0.30, 0.62, 0.95)
    assert len(CT.drop_contained([near, side_by_side])) == 2
