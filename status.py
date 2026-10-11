"""
Tracker status for Home Assistant, over MQTT discovery.

    <base>/<node>/availability   online | offline   (retained, LWT)
    <base>/<node>/state          JSON snapshot      (retained)
    <base>/<node>/last_seen      ISO timestamp      (retained)
    homeassistant/<component>/camtracker_<node>_<thing>/config   (retained)

Published on change (checked once a second) plus a slow refresh, never at
detection rate. A crash is caught by the MQTT last will; a clean stop
publishes offline straight away.

Started as demoneye's ha_status.py (the eyes' status publisher); the
publisher is the same, the discovery is the tracker's own.
"""

import json
import re
import socket
import threading
import time

VERSION = "0.1.0"

STATUS_DEFAULTS = {
    "enabled": "true",
    "node": "camera",
    "base_topic": "house/status",
    "discovery_prefix": "homeassistant",
    "refresh_s": "30",
}


ON_WORDS = ["on", "1", "true", "play", "start", "yes", "open"]


OFF_WORDS = ["off", "0", "false", "stop", "pause", "no", "close"]


def hostname():
    return socket.gethostname().split(".")[0]


def resolve_node(node, host=None):
    """'auto' -> the short hostname; anything else as given. Sanitised for a topic."""
    node = (node or "auto").strip().lower()
    if node == "auto":
        node = (host or hostname()).split(".")[0].lower()
    return re.sub(r"[^a-z0-9_]+", "_", node).strip("_") or "camera"


def utc_iso(ts=None):
    t = time.gmtime(time.time() if ts is None else ts)
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", t)


def topics(base, node):
    base = base.rstrip("/")
    return {"availability": f"{base}/{node}/availability",
            "state": f"{base}/{node}/state",
            "last_seen": f"{base}/{node}/last_seen"}


def _device(node, name, model):
    return {"identifiers": [f"camtracker_{node}"], "name": name,
            "manufacturer": "camtracker", "model": model, "sw_version": VERSION}


def _entity(prefix, component, uid, cfg):
    # default_entity_id (not the deprecated object_id) picks the entity_id on
    # first discovery; after that HA's entity registry owns it.
    return f"{prefix}/{component}/{uid}/config", dict(cfg, unique_id=uid)


def discovery(node="camera", base="house/status", prefix="homeassistant"):
    """Home Assistant discovery for the tracker's entities."""
    t = topics(base, node)
    dev = _device(node, "Camera Person Tracker", "Camera person tracker")
    avail = {"availability_topic": t["availability"]}
    return dict([
        _entity(prefix, "binary_sensor", f"camtracker_{node}_running", {
            "name": "Running", "device_class": "running",
            "state_topic": t["availability"], "payload_on": "online", "payload_off": "offline",
            "default_entity_id": "binary_sensor.camtracker_running", "device": dev,
        }),
        _entity(prefix, "binary_sensor", f"camtracker_{node}_tracking", {
            "name": "Tracking", "device_class": "occupancy", **avail,
            "state_topic": t["state"],
            "value_template": "{{ 'ON' if value_json.tracking else 'OFF' }}",
            "default_entity_id": "binary_sensor.camtracker_tracking", "device": dev,
        }),
        _entity(prefix, "sensor", f"camtracker_{node}_last_target_seen", {
            "name": "Last person seen", "device_class": "timestamp",
            "state_topic": t["last_seen"], "icon": "mdi:account-eye",
            "default_entity_id": "sensor.camtracker_last_person_seen", "device": dev,
        }),
        _entity(prefix, "binary_sensor", f"camtracker_{node}_stream", {
            "name": "Camera stream", "device_class": "connectivity", **avail,
            "state_topic": t["state"],
            "value_template": "{{ 'ON' if value_json.stream_ok else 'OFF' }}",
            "entity_category": "diagnostic",
            "default_entity_id": "binary_sensor.camtracker_stream", "device": dev,
        }),
        _entity(prefix, "sensor", f"camtracker_{node}_detector", {
            "name": "Detector", **avail, "state_topic": t["state"],
            "value_template": "{{ value_json.detector }}",
            "json_attributes_topic": t["state"],
            "json_attributes_template":
                "{{ {'enabled': value_json.enabled, 'host': value_json.host, "
                "'version': value_json.version} | tojson }}",
            "entity_category": "diagnostic", "icon": "mdi:chip",
            "default_entity_id": "sensor.camtracker_detector", "device": dev,
        }),
    ])


class StatusPublisher:
    """Owns one node's status on an existing paho client.

    Call it in this order:
        sp = StatusPublisher(client, node, discovery, snapshot_fn, ...)   # sets the LWT
        client.connect(...); client.loop_start()
        in on_connect:  sp.on_connect()
        sp.start()                     # background thread: on change + refresh
        on clean exit:  sp.go_offline()  (before loop_stop / disconnect)
    """

    def __init__(self, client, node, discovery, snapshot_fn, base="house/status",
                 prefix="homeassistant", refresh_s=30.0, tick_s=1.0, clock=time.monotonic):
        self.client = client
        self.node = node
        self.discovery = discovery
        self.snapshot_fn = snapshot_fn
        self.topics = topics(base, node)
        self.prefix = prefix
        self.refresh_s = max(5.0, float(refresh_s))
        self.tick_s = tick_s
        self.clock = clock
        self._last = None
        self._last_pub = None
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        # LWT: must be set before connect(). Retained, so HA sees "offline"
        # even if it (re)starts after the crash.
        client.will_set(self.topics["availability"], "offline", qos=1, retain=True)
        # HA announces itself on <prefix>/status when it starts: resend then, in
        # case the retained configs were cleared from the broker.
        client.message_callback_add(f"{prefix}/status", self._on_ha_status)

    # -- MQTT events --
    def on_connect(self):
        """Call from the client's on_connect (also runs after a reconnect)."""
        self.client.subscribe(f"{self.prefix}/status", qos=1)
        self.publish_discovery()
        self.client.publish(self.topics["availability"], "online", qos=1, retain=True)
        self.tick(force=True)

    def _on_ha_status(self, client, userdata, msg):
        if msg.payload.decode(errors="ignore").strip().lower() == "online":
            self.publish_discovery()
            self.client.publish(self.topics["availability"], "online", qos=1, retain=True)
            self.tick(force=True)

    def publish_discovery(self):
        for topic, cfg in self.discovery.items():
            self.client.publish(topic, json.dumps(cfg, separators=(",", ":")), qos=1, retain=True)

    # -- state --
    def tick(self, force=False):
        """Publish the snapshot if it changed or the refresh is due. Returns
        True if it published."""
        try:
            snap = self.snapshot_fn()
        except Exception as e:                      # never take the show down over status
            print(f"[status] snapshot failed: {e}", flush=True)
            return False
        now = self.clock()
        with self._lock:
            due = (force or snap != self._last or self._last_pub is None
                   or now - self._last_pub >= self.refresh_s)
            if not due:
                return False
        payload = json.dumps(snap, sort_keys=True)   # may raise: then nothing is marked sent
        self.client.publish(self.topics["state"], payload, qos=1, retain=True)
        with self._lock:
            self._last, self._last_pub = snap, now
        return True

    def publish_last_seen(self, ts=None):
        self.client.publish(self.topics["last_seen"], utc_iso(ts), qos=1, retain=True)

    # -- lifecycle --
    def start(self):
        if self._thread:
            return
        self._thread = threading.Thread(target=self._run, name=f"status-{self.node}", daemon=True)
        self._thread.start()

    def _run(self):
        # Nothing may end this loop but go_offline(): a dead status thread
        # leaves HA showing frozen state that still says "online".
        last_err, last_err_at = None, -1e9
        while not self._stop.wait(self.tick_s):
            try:
                self.tick()
            except Exception as e:
                now = self.clock()
                if repr(e) != last_err or now - last_err_at > 60:
                    print(f"[status] publish failed: {e!r} — will keep trying", flush=True)
                    last_err, last_err_at = repr(e), now

    def go_offline(self, timeout=1.0):
        """Clean shutdown: say so now rather than waiting for the LWT."""
        self._stop.set()
        try:
            info = self.client.publish(self.topics["availability"], "offline", qos=1, retain=True)
            info.wait_for_publish(timeout=timeout)
        except Exception:
            pass


def load_section(cp, section="status", overrides=None):
    """Read a [status] section (ConfigParser) over STATUS_DEFAULTS. Returns a
    plain dict of typed values."""
    vals = dict(STATUS_DEFAULTS, **(overrides or {}))
    if cp is not None and cp.has_section(section):
        vals.update({k: v for k, v in cp.items(section)})

    def _b(v):
        return str(v).strip().lower() in ("1", "yes", "true", "on")
    return {
        "enabled": _b(vals["enabled"]),
        "node": vals["node"].strip(),
        "base_topic": vals["base_topic"].strip().rstrip("/"),
        "discovery_prefix": vals["discovery_prefix"].strip().rstrip("/"),
        "refresh_s": float(vals["refresh_s"]),
    }

