"""Dashboard tests: state updates and HTTP endpoints (no browser needed)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import threading
import urllib.request

from src.dashboard import DashboardState, start_dashboard


class StubTrack:
    def __init__(self, tid):
        self.id = tid


def _get(port, path):
    import urllib.error
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def test_state_counts_unique_tracks():
    s = DashboardState()
    s.update([StubTrack(1), StubTrack(2)], fps=30.0)
    s.update([StubTrack(2), StubTrack(3)], fps=29.0)
    snap = s.snapshot()
    assert snap["tracks"] == 2
    assert snap["total_tracks"] == 3  # unique IDs seen
    assert snap["fps"] == 29.0


def test_http_endpoints():
    s = DashboardState()
    s.update([StubTrack(7)], fps=25.0)
    server = start_dashboard(s, port=18081)
    try:
        status, body = _get(18081, "/")
        assert status == 200 and b"cv-tracker" in body
        status, body = _get(18081, "/api/stats")
        assert status == 200
        data = json.loads(body)
        assert data["tracks"] == 1 and data["total_tracks"] == 1
        status, _ = _get(18081, "/nope")
        assert status == 404
    finally:
        server.shutdown()


def test_sse_streams_pushed_events():
    s = DashboardState()
    server = start_dashboard(s, port=18082)
    received = []

    def reader():
        req = urllib.request.Request("http://127.0.0.1:18082/api/events")
        with urllib.request.urlopen(req, timeout=5) as r:
            while len(received) < 2:  # skip blank separator lines
                line = r.readline().decode()
                if line.startswith("data:"):
                    received.append(json.loads(line[5:]))

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    s.push_event({"type": "loitering", "track_id": 3})
    s.push_event({"type": "speeding", "track_id": 5})
    t.join(timeout=5)
    server.shutdown()
    assert len(received) == 2
    assert received[0]["type"] == "loitering"
    assert received[1]["track_id"] == 5
