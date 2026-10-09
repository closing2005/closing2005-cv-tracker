"""Zones: polygon ROIs and tripwire lines for counting and dwell.

Zone: counts entries/exits, measures dwell time per track.
Tripwire: directed line crossing -- reports crossing direction
(IN vs OUT) by the sign of the cross product walk.
"""

from __future__ import annotations

import cv2
import numpy as np


class Zone:
    def __init__(self, name, polygon):
        """polygon: [(x, y), ...] in pixels."""
        self.name = name
        self.polygon = np.array(polygon, dtype=np.int32)
        self.entries = 0
        self.exits = 0
        self._inside = {}   # track_id -> bool
        self._enter_frame = {}  # track_id -> frame idx
        self.dwell = {}     # track_id -> frames spent inside

    def contains(self, point):
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0

    def update(self, tracks, frame_idx):
        """tracks: iterable of Track. Returns list of (event, track_id)."""
        events = []
        seen = set()
        for t in tracks:
            inside = self.contains(t.center())
            seen.add(t.id)
            was = self._inside.get(t.id, False)
            if inside and not was:
                self.entries += 1
                self._enter_frame[t.id] = frame_idx
                events.append(("enter", t.id))
            elif not inside and was:
                self.exits += 1
                enter = self._enter_frame.pop(t.id, frame_idx)
                self.dwell[t.id] = self.dwell.get(t.id, 0) + (frame_idx - enter)
                events.append(("exit", t.id))
            self._inside[t.id] = inside
        # Forget tracks that vanished (avoid unbounded growth)
        for tid in list(self._inside):
            if tid not in seen:
                self._inside.pop(tid, None)
                self._enter_frame.pop(tid, None)
        return events


class Tripwire:
    def __init__(self, name, p1, p2):
        """Directed line p1 -> p2. Stand at p1 looking toward p2:
        crossing from your left to your right counts as 'forward',
        the reverse as 'backward'."""
        self.name = name
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)
        self.forward = 0
        self.backward = 0
        self._prev_side = {}  # track_id -> sign

    def _side(self, point):
        # Sign of cross product (p2-p1) x (point-p1)
        d = self.p2 - self.p1
        v = np.array(point, dtype=float) - self.p1
        return float(np.sign(d[0] * v[1] - d[1] * v[0]))

    def update(self, tracks):
        events = []
        seen = set()
        for t in tracks:
            side = self._side(t.center())
            seen.add(t.id)
            prev = self._prev_side.get(t.id)
            if prev is not None and side != 0 and prev != 0 and side != prev:
                if side > 0:
                    self.forward += 1
                    events.append(("forward", t.id))
                else:
                    self.backward += 1
                    events.append(("backward", t.id))
            if side != 0:
                self._prev_side[t.id] = side
        for tid in list(self._prev_side):
            if tid not in seen:
                self._prev_side.pop(tid, None)
        return events
