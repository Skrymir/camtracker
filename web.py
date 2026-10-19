"""
Live view: the camera's picture with what the tracker sees drawn on it (the
same drawing as --snapshot, plus who it is following), as a web page.

    /              the page
    /stream.mjpg   the live picture (MJPEG: a plain <img> shows it)
    /snapshot.jpg  one full-resolution frame

Nothing is drawn or encoded unless someone is looking. While someone is (and
for linger_s after), the tracker reads the camera even with the show off, so
the view works for setting up during the day. Pure stdlib: the drawing and
JPEG encoding are the render function the tracker passes in.
"""

import contextlib
import http.server
import threading
import time

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>camtracker</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #111; color: #ddd; font: 14px/1.4 system-ui, sans-serif; }
  img { display: block; width: 100%; height: auto; background: #000; }
  p { margin: 8px 16px; }
  .k { display: inline-block; width: .8em; height: .8em; margin: 0 4px 0 12px;
       border: 2px solid; vertical-align: -1px; }
  a { color: #9cf; }
</style></head><body>
<img src="stream.mjpg" alt="Live camera view">
<p><span class="k" style="border-color:#0d0"></span>following
<span class="k" style="border-color:#f00"></span>person
<span class="k" style="border-color:#f80"></span>near miss
<span class="k" style="border-color:#f0f"></span>ignored / masks
<span class="k" style="border-color:#fc0"></span>roi
<span class="k" style="border-color:#0af"></span>tiles
<span class="k" style="border-color:#0ff"></span>far_roi tiles</p>
<p><a href="snapshot.jpg">Full-resolution frame</a></p>
</body></html>
"""


class LiveView:
    """Between the detection loop (offer() each frame while wanted) and the
    web clients (frame_jpeg() / full_jpeg())."""

    def __init__(self, render, fps=5.0, width=1280, linger_s=10.0, clock=time.monotonic):
        self.render = render                # (state, width or None) -> JPEG bytes
        self.fps = max(0.5, fps)
        self.width = width
        self.linger = linger_s
        self.clock = clock
        self._cond = threading.Condition()
        self._render_lock = threading.Lock()
        self._state, self._seq = None, 0
        self._viewers, self._left = 0, -1e9
        self._jpeg = (0, None)              # the newest stream frame, shared by viewers

    @property
    def wanted(self):
        """Someone is looking (or was, a moment ago: a page reload shouldn't
        restart the camera)."""
        with self._cond:
            return self._viewers > 0 or self.clock() - self._left < self.linger

    @contextlib.contextmanager
    def viewer(self):
        with self._cond:
            self._viewers += 1
        try:
            yield
        finally:
            with self._cond:
                self._viewers -= 1
                self._left = self.clock()
                if not self._viewers:
                    self._state = None          # don't hold on to a 4K frame

    def offer(self, state):
        """The loop's newest frame and what it found (a dict for render)."""
        with self._cond:
            self._state, self._seq = state, self._seq + 1
            self._cond.notify_all()

    def wait(self, after_seq, timeout):
        """(seq, state) newer than after_seq, or (after_seq, None) on timeout."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > after_seq and self._state is not None,
                                timeout)
            if self._seq > after_seq and self._state is not None:
                return self._seq, self._state
            return after_seq, None

    def frame_jpeg(self, after_seq, timeout=10.0):
        """The next stream frame (scaled to width), rendered once for all
        viewers. (after_seq, None) if no new frame came in time."""
        seq, state = self.wait(after_seq, timeout)
        if state is None:
            return after_seq, None
        with self._render_lock:
            if self._jpeg[0] < seq:
                self._jpeg = (seq, self.render(state, self.width))
            return self._jpeg

    def last_jpeg(self):
        return self._jpeg[1]

    def full_jpeg(self, timeout=20.0):
        """A fresh frame at full resolution (waits for the camera if idle)."""
        with self.viewer():
            with self._cond:
                now = self._seq
            _, state = self.wait(now, timeout)
            return None if state is None else self.render(state, None)


def handler(view):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif path == "/stream.mjpg":
                self._stream()
            elif path == "/snapshot.jpg":
                jpg = view.full_jpeg()
                if jpg is None:
                    self._send(503, "text/plain", b"no picture from the camera yet\n")
                else:
                    self._send(200, "image/jpeg", jpg)
            else:
                self._send(404, "text/plain", b"not found\n")

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = 0
            try:
                with view.viewer():
                    while True:
                        t0 = time.monotonic()
                        seq, jpg = view.frame_jpeg(seq)
                        # camera down: resend the last picture, which also
                        # notices a viewer who has gone
                        jpg = jpg or view.last_jpeg()
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                                             + jpg + b"\r\n")
                            self.wfile.flush()
                        time.sleep(max(0.0, 1.0 / view.fps - (time.monotonic() - t0)))
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

        def log_message(self, *args):         # quiet: the stream is one long request
            pass

    return Handler


def serve(view, port, host="0.0.0.0"):
    """Start the web server on its own thread. Returns it (shutdown() to stop)."""
    srv = http.server.ThreadingHTTPServer((host, port), handler(view))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name="web", daemon=True).start()
    return srv
