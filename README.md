# camtracker

A camera person tracker for an x86-64 server with a spare USB Coral. It reads
a camera's RTSP stream, finds people (MobileDet, the model Frigate uses on a
Coral), and publishes where the nearest one is as a **look target** over MQTT,
10-15 times a second.

Built for [demoneye](https://github.com/Skrymir/demoneye) (projected eyes on a
house that follow people walking by), where it replaced an mmWave sensor that
couldn't reach the street. Anything that understands the same two topics can
use it.

## MQTT contract

| Topic | Direction | Payload |
|---|---|---|
| `house/tracking/target` | out | `{"x": -0.42, "y": 0.0, "hold": 2.0}` while someone is in view, `clear` when they're gone |
| `house/show/command` | in | `on` / `off` (also `1`/`0`, `play`/`stop`, `{"state": "ON"}` ...). Tracks only while on. |
| `house/status/camera/...` | out | Home Assistant status (retained, MQTT discovery) |

`x` and `y` run -1..+1 from the **audience's** point of view: `x = -1` is
their left, `y = -1` is down. `hold` is how long the receiver should keep the
target without an update. Topics are set in `camtracker.ini`.

## The math

Mount the camera on the house between the eyes (or whatever is doing the
looking), facing the street. For a person whose box centre is at `u` across
the picture (0 = left edge, 1 = right edge):

```
bearing = atan((2u - 1) * tan(hfov_deg / 2))     angle from the camera, + = picture's right
x       = -bearing / gaze_max_deg                clamped to -1..+1, flipped by invert_x
```

The camera sits where the eyes are, so its bearing is their bearing and no
distance is needed. The `tan` term is perspective: a quarter of the way in from
the edge is not a quarter of the angle. The minus: the camera and the eyes face
the street, so the picture's right is the audience's left. A wide lens bends
the picture, so `points = u:x, ...` can replace the formula with straight lines
between spots measured with `--watch`. With `y_from_feet`, the target also
moves down as a person's feet get lower in the frame (closer).

Who it looks at: the nearest person (tallest box), and it stays on them while
they're in view rather than flicking to whoever is closest each frame. Brief
missed detections (`lost_after_s`) don't drop the target or move it to
someone else.

## Running it on Unraid

### 1. The Coral

Plug the spare USB Coral into the Unraid server (a USB 3 port). Leave it with
Unraid itself: don't pass it through to the Home Assistant VM. Frigate's own
Coral is unaffected; the tracker uses the first Edge TPU that isn't busy.

Check Unraid sees it (terminal): `lsusb | grep -Ei "1a6e|18d1"`. It shows as
`Global Unichip` until first used, then `Google`. That change gives it a new
device node, which is why the container gets the live `/dev/bus/usb` and
permission for any USB device (`--device-cgroup-rule='c 189:* rmw'`) rather
than a fixed device list: otherwise the first start after a reboot fails.

### 2. Build the image

Clone this repo onto the server (any share), then in the Unraid terminal:

```bash
cd /mnt/user/appdata/camtracker-src     # wherever you cloned the repo
docker build -t camtracker .
```

Rebuild after pulling new code; the image holds camtracker.py.

### 3. Config and secrets

```bash
mkdir -p /mnt/user/appdata/camtracker
cp camtracker.ini /mnt/user/appdata/camtracker/camtracker.ini
cat > /mnt/user/appdata/camtracker/camtracker.env <<'ENV'
CAMERA_URL=rtsp://user:password@camera-ip:554/your-substream-path
MQTT_USERNAME=
MQTT_PASSWORD=
ENV
```

Edit `camtracker.ini`: `hfov_deg` from the camera's spec sheet, and leave the
rest for now. The camera's RTSP path depends on the make; Frigate's config has
it if the camera is already in Frigate (or use Frigate's restream,
`rtsp://<frigate-host>:8554/<camera-name>`).

### 4. Set up the picture (snapshot)

```bash
docker run --rm --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb \
  -v /mnt/user/appdata/camtracker:/config \
  --env-file /mnt/user/appdata/camtracker/camtracker.env \
  camtracker --snapshot /config/snap.jpg
```

Open `snap.jpg` (it's in the appdata share). Yellow = the ROI, blue = the
detection tiles (cyan = the finer `far_roi` tiles, if set), red = people found (with their score), orange = near misses
just under `min_score`, magenta = ignored decorations, with a 0.1-0.9 ruler
along the bottom. Each detection is also printed with its `box`: put a
decoration's box in `[detector] ignore` so the eyes don't stare at it, or
draw a snug mask around it in Frigate's mask editor (Settings → Masks / Zones
→ Object masks) and paste its coordinates into `ignore_masks`; masks are drawn
in magenta too. Set
`roi` to the part people walk in (skip the sky) and take another snapshot.
If people far away are missed while `roi` is tall, add `far_roi` over the far
strip: it is searched again with smaller tiles, so distant people look bigger
to the model.

### 5. Calibrate (watch)

Same command with `--watch --no-publish` instead of `--snapshot ...`. Have
someone stand at the left end, the middle and the right end of the stretch
you care about; it prints each person's `u` and the look it would send. If the
eyes should look somewhere else for a spot, write it as a point:

```ini
points = 0.12:0.9, 0.5:0, 0.88:-0.9
```

(x is the audience's view: -1 = their left, +1 = their right.) Drop
`--no-publish` to watch the eyes follow live.

### 6. Run it for good

`docker compose up -d` in the repo (Compose Manager plugin), or:

```bash
docker run -d --name camtracker --restart unless-stopped \
  --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb \
  -v /mnt/user/appdata/camtracker:/config \
  --env-file /mnt/user/appdata/camtracker/camtracker.env \
  camtracker
```

It's idle until the show turns on (`house/show/command`).
Logs: `docker logs -f camtracker` (`[tgt] acquired` / `cleared` as
people come and go). In Home Assistant it appears as **Camera Person
Tracker**: running, tracking, camera stream, detector, last person seen.

## One tracker at a time (demoneye)

demoneye's mmWave bridge publishes to the same topic. Turn it off once the camera
works, or the eyes will be pulled between the two:

```bash
ssh joshc@pi-eye-right.home 'sudo systemctl disable --now ha-bridge'
```

## Troubleshooting

- `no usable Edge TPU`: the Coral isn't visible to the container (check
  `lsusb` on the host and the `/dev/bus/usb` options above), or another
  program is using it.
- `can't open the stream`: wrong `CAMERA_URL`; test it in VLC. Some cameras
  allow only a couple of RTSP connections: use Frigate's restream instead.
- `[cam] opening ...` and then nothing for 30 s: the server can't reach the
  camera at all (wrong IP or port, or a firewall/VLAN in the way). From the
  server: `timeout 5 bash -c '</dev/tcp/<camera-ip>/554' && echo open`.
- People far away missed: use the main stream instead of the substream, or a
  narrower `roi` (more tiles = more detail).
- Eyes look away from people: `invert_x = true`, restart the container.
- Eyes lag behind people: use the camera's own substream rather than
  Frigate's restream (one hop less), and lower the camera's I-frame interval
  if it has the setting. The stream is read with ffmpeg's input buffering off
  (`OPENCV_FFMPEG_CAPTURE_OPTIONS`, set it in `camtracker.env` to override).

## Development

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements-dev.txt
./venv/bin/python -m pytest -q tests
```

The tests cover the mapping, who to follow, the target/clear messages, the
command handling and the Home Assistant templates; none need OpenCV, a Coral
or a broker. OpenCV and tflite_runtime are only imported by the stream and
detector classes, which run in the container.
