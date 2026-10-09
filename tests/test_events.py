"""Behavioral event tests with synthetic histories."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.events import (CrowdingDetector, LoiteringDetector, SpeedingDetector,
                    WrongWayDetector)
from src.zones import Zone


class StubTrack:
    _id = 0

    def __init__(self, bboxes):
        StubTrack._id += 1
        self.id = StubTrack._id
        self.history = [tuple(b) for b in bboxes]
        self.bbox = list(bboxes[-1])

    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


def still_box(cx, cy, w=30, h=30):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def test_loitering_fires_and_fires_once():
    d = LoiteringDetector(radius_px=40, min_frames=20)
    t = StubTrack([still_box(100, 100) for _ in range(25)])
    evs = d.update([t], 25)
    assert len(evs) == 1 and evs[0]["type"] == "loitering"
    assert d.update([t], 26) == []  # no repeat firing


def test_loitering_ignores_moving_target():
    d = LoiteringDetector(radius_px=40, min_frames=20)
    t = StubTrack([still_box(100 + 10 * i, 100) for i in range(25)])
    assert d.update([t], 25) == []


def test_speeding_fires():
    d = SpeedingDetector(max_px_per_sec=300, fps=30, smooth=5)
    # 20 px/frame @30fps = 600 px/s > 300
    t = StubTrack([still_box(100 + 20 * i, 100) for i in range(10)])
    evs = d.update([t], 10)
    assert len(evs) == 1 and evs[0]["type"] == "speeding"
    assert evs[0]["px_per_sec"] > 300


def test_wrong_way_fires():
    d = WrongWayDetector(flow=(1, 0), min_frames=10, cos_thresh=-0.5)
    t = StubTrack([still_box(500 - 10 * i, 100) for i in range(15)])  # moving left
    evs = d.update([t], 15)
    assert len(evs) == 1 and evs[0]["type"] == "wrong_way"


def test_wrong_way_ignores_correct_flow():
    d = WrongWayDetector(flow=(1, 0), min_frames=10, cos_thresh=-0.5)
    t = StubTrack([still_box(100 + 10 * i, 100) for i in range(15)])
    assert d.update([t], 15) == []


def test_crowding_fires_after_sustained_overload():
    z = Zone("hall", [(0, 0), (200, 0), (200, 200), (0, 200)])
    d = CrowdingDetector(z, max_tracks=2, min_frames=5)
    tracks = [StubTrack([still_box(50 + 40 * i, 50)]) for i in range(3)]
    evs = []
    for f in range(10):
        evs += d.update(tracks, f)
    assert len(evs) == 1 and evs[0]["type"] == "crowding"
    assert evs[0]["count"] == 3
