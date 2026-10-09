"""Live web dashboard: counts, event stream and FPS in the browser.

Zero new dependencies -- stdlib http.server + Server-Sent Events.
The tracker pushes into DashboardState; the HTTP thread serves it.

    state = DashboardState()
    start_dashboard(state, port=8080)   # background thread
    ...
    state.update(tracks=tracks, fps=30.0, timings={...})
    state.push_event({"type": "loitering", "track_id": 7})
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cv-tracker live</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0b1020;color:#e2e8f0;font-family:system-ui,sans-serif;padding:20px}
h1{font-size:20px;margin-bottom:16px;color:#f8fafc}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:20px}
.card{background:#151d33;border:1px solid #243052;border-radius:12px;padding:16px}
.card .v{font-size:32px;font-weight:700;color:#38bdf8}
.card .k{font-size:12px;color:#94a3b8;margin-top:4px}
h2{font-size:14px;color:#94a3b8;margin:0 0 10px}
#events{list-style:none;max-height:50vh;overflow:auto}
#events li{background:#151d33;border:1px solid #243052;border-radius:8px;
  padding:10px 12px;margin-bottom:8px;font-size:13px}
#events li b{color:#f87171}
#events .t{color:#64748b;font-size:11px}
</style></head><body>
<h1>cv-tracker · 实时面板</h1>
<div class="grid">
<div class="card"><div class="v" id="tracks">0</div><div class="k">当前目标</div></div>
<div class="card"><div class="v" id="fps">0</div><div class="k">FPS</div></div>
<div class="card"><div class="v" id="total">0</div><div class="k">累计目标</div></div>
<div class="card"><div class="v" id="evcount">0</div><div class="k">事件总数</div></div>
</div>
<h2>事件流</h2>
<ul id="events"></ul>
<script>
const ev = new EventSource('/api/events');
let n = 0;
ev.onmessage = e => {
  const d = JSON.parse(e.data);
  n++;
  document.getElementById('evcount').textContent = n;
  const li = document.createElement('li');
  const time = new Date().toLocaleTimeString();
  li.innerHTML = '<b>' + d.type + '</b> #' + (d.track_id ?? '-')
    + ' <span class="t">' + time + '</span>';
  const ul = document.getElementById('events');
  ul.prepend(li);
  while (ul.children.length > 50) ul.lastChild.remove();
};
setInterval(async () => {
  const s = await (await fetch('/api/stats')).json();
  document.getElementById('tracks').textContent = s.tracks;
  document.getElementById('fps').textContent = s.fps.toFixed(1);
  document.getElementById('total').textContent = s.total_tracks;
}, 1000);
</script></body></html>
"""


class DashboardState:
    def __init__(self):
        self._lock = threading.Lock()
        self._stats = {"tracks": 0, "fps": 0.0, "total_tracks": 0}
        self._events = queue.Queue()
        self._seen_ids = set()

    def update(self, tracks, fps=0.0, timings=None):
        with self._lock:
            self._stats["tracks"] = len(tracks)
            self._stats["fps"] = fps
            for t in tracks:
                self._seen_ids.add(t.id)
            self._stats["total_tracks"] = len(self._seen_ids)

    def push_event(self, event):
        self._events.put(event)

    def snapshot(self):
        with self._lock:
            return dict(self._stats)


def _handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/stats":
                body = json.dumps(state.snapshot()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while True:
                        ev = state._events.get()
                        msg = f"data: {json.dumps(ev)}\n\n".encode()
                        self.wfile.write(msg)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_response(404)
                self.end_headers()
    return H


def start_dashboard(state, port=8080):
    """Start the dashboard in a background thread. Returns the server."""
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(state))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"dashboard: http://127.0.0.1:{port}")
    return server
