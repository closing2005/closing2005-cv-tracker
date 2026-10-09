"""Behavioral event detection on top of raw tracks.

This is the layer that turns "boxes moving" into "something happened":
loitering, speeding, wrong-way motion and crowding. All detectors work
on track history only -- no extra model, no extra cost.

Each detector is deliberately small and explainable: thresholds live in
config.yaml, and every event carries the evidence (track id, frames,
measured value) so results are auditable, not magic.
"""

from __future__ import annotations

import math
from collections import deque


class LoiteringDetector:
    """Fires when a track stays inside a small radius for too long."""

    def __init__(self, radius_px=40, min_frames=150):
        self.radius_px = radius_px
        self.min_frames = min_frames
        self._fired = set()

    def update(self, tracks, frame_idx):
        events = []
        for t in tracks:
            if t.id in self._fired or len(t.history) < self.min_frames:
                continue
            recent = t.history[-self.min_frames:]
            cx = sum(b[0] + b[2] for b in recent) / (2 * len(recent))
            cy = sum(b[1] + b[3] for b in recent) / (2 * len(recent))
            spread = max(
                math.hypot((b[0] + b[2]) / 2 - cx, (b[1] + b[3]) / 2 - cy)
                for b in recent
            )
            if spread < self.radius_px:
                self._fired.add(t.id)
                events.append({
                    "type": "loitering", "track_id": t.id,
                    "frame": frame_idx, "spread_px": round(spread, 1),
                    "frames": self.min_frames,
                })
        return events


class SpeedingDetector:
    """Fires when a track's smoothed speed exceeds a threshold."""

    def __init__(self, max_px_per_sec=300, fps=30, smooth=5):
        self.max_v = max_px_per_sec
        self.fps = fps
        self.smooth = smooth
        self._fired = set()

    def _speed(self, track):
        h = track.history
        if len(h) < self.smooth + 1:
            return 0.0
        vs = []
        for a, b in zip(h[-(self.smooth + 1):-1], h[-self.smooth:]):
            ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
            bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            vs.append(math.hypot(bx - ax, by - ay) * self.fps)
        return sum(vs) / len(vs)

    def update(self, tracks, frame_idx):
        events = []
        for t in tracks:
            if t.id in self._fired:
                continue
            v = self._speed(t)
            if v > self.max_v:
                self._fired.add(t.id)
                events.append({
                    "type": "speeding", "track_id": t.id,
                    "frame": frame_idx, "px_per_sec": round(v, 1),
                })
        return events


class WrongWayDetector:
    """Fires when a track moves against a declared flow direction.

    flow: (dx, dy) unit-ish vector of the expected motion, e.g. (1, 0)
    means "everyone should move right".
    """

    def __init__(self, flow=(1, 0), min_frames=30, cos_thresh=-0.5):
        self.flow = flow
        self.min_frames = min_frames
        self.cos_thresh = cos_thresh
        self._fired = set()

    def update(self, tracks, frame_idx):
        fx, fy = self.flow
        fnorm = math.hypot(fx, fy) or 1.0
        events = []
        for t in tracks:
            if t.id in self._fired or len(t.history) < self.min_frames + 1:
                continue
            a, b = t.history[-self.min_frames - 1], t.history[-1]
            dx = (b[0] + b[2] - a[0] - a[2]) / 2
            dy = (b[1] + b[3] - a[1] - a[3]) / 2
            dnorm = math.hypot(dx, dy)
            if dnorm < 1e-6:
                continue
            cos_sim = (dx * fx + dy * fy) / (dnorm * fnorm)
            if cos_sim < self.cos_thresh:
                self._fired.add(t.id)
                events.append({
                    "type": "wrong_way", "track_id": t.id,
                    "frame": frame_idx, "cos_sim": round(cos_sim, 2),
                })
        return events


class CrowdingDetector:
    """Fires when more than `max_tracks` confirmed tracks share one zone."""

    def __init__(self, zone, max_tracks=5, min_frames=90):
        self.zone = zone
        self.max_tracks = max_tracks
        self.min_frames = min_frames
        self._over_since = None
        self._fired = False

    def update(self, tracks, frame_idx):
        inside = [t for t in tracks if self.zone.contains(t.center())]
        if len(inside) > self.max_tracks:
            if self._over_since is None:
                self._over_since = frame_idx
            elif frame_idx - self._over_since >= self.min_frames and not self._fired:
                self._fired = True
                return [{
                    "type": "crowding", "zone": self.zone.name,
                    "frame": frame_idx, "count": len(inside),
                    "track_ids": [t.id for t in inside],
                }]
        else:
            self._over_since = None
            self._fired = False
        return []


class EventBus:
    """Fan-out: one update() call runs every detector."""

    def __init__(self, detectors=()):
        self.detectors = list(detectors)
        self.log = deque(maxlen=10000)

    def update(self, tracks, frame_idx):
        fresh = []
        for d in self.detectors:
            evs = d.update(tracks, frame_idx)
            fresh.extend(evs)
            self.log.extend(evs)
        return fresh
