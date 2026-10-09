"""Multi-object tracker: ByteTrack-style two-stage association.

Pipeline per frame:
  1. Kalman-predict every live track.
  2. Stage 1: match tracks against HIGH-confidence detections (IoU cost).
  3. Stage 2: match leftover tracks against LOW-confidence detections --
     this recovers briefly occluded targets instead of spawning new IDs
     (the core ByteTrack insight).
  4. Unmatched high detections seed new tentative tracks.

Track lifecycle: Tentative -> Confirmed -> Lost -> Deleted.
Only Confirmed tracks are reported, which kills most flicker ghosts.
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

from .hungarian import linear_sum_assignment
from .kalman import KalmanBoxTracker


class TrackState(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    LOST = auto()
    DELETED = auto()


def iou(a, b):
    """IoU of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    _next_id = 0

    def __init__(self, bbox, score, n_init=3):
        Track._next_id += 1
        self.id = Track._next_id
        self.kf = KalmanBoxTracker(bbox)
        self.bbox = list(bbox)
        self.score = score
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.n_init = n_init
        self.history = [tuple(bbox)]  # confirmed-box centers over time

    def predict(self):
        self.bbox = self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, score):
        self.bbox = self.kf.update(bbox)
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        self.history.append(tuple(self.bbox))
        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED  # re-acquired

    def mark_lost(self):
        if self.state != TrackState.DELETED:
            self.state = TrackState.LOST

    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class MultiTracker:
    def __init__(self, high_thresh=0.5, low_thresh=0.15, max_iou_dist=0.7,
                 n_init=3, max_age=30, max_lost=60):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_iou_dist = max_iou_dist  # min IoU = 1 - this
        self.n_init = n_init
        self.max_age = max_age      # frames before Tentative -> Deleted
        self.max_lost = max_lost    # frames before Lost -> Deleted
        self.tracks = []
        self.id_switches = 0  # self-diagnostic counter

    # -- public ------------------------------------------------------
    def update(self, detections):
        """detections: [[x1,y1,x2,y2,score], ...]. Returns confirmed Tracks."""
        high = [d for d in detections if d[4] >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d[4] < self.high_thresh]

        for t in self.tracks:
            t.predict()

        live = [t for t in self.tracks if t.state != TrackState.DELETED]

        # Stage 1: high-confidence detections
        m1, u_tracks, u_high = self._associate(live, high)
        for ti, di in m1:
            live[ti].update(high[di][:4], high[di][4])

        # Stage 2: low-confidence detections rescue unmatched tracks
        remaining = [live[i] for i in u_tracks]
        m2, u_tracks2, _ = self._associate(remaining, low)
        for ti, di in m2:
            remaining[ti].update(low[di][:4], low[di][4])

        rescued = {remaining[i] for i, _ in m2}
        for i in u_tracks2:
            remaining[i].mark_lost()

        # New tracks from unmatched high detections
        for di in u_high:
            self.tracks.append(Track(high[di][:4], high[di][4], self.n_init))

        # Age out
        for t in self.tracks:
            if t.state == TrackState.TENTATIVE and t.time_since_update > self.max_age:
                t.state = TrackState.DELETED
            elif t.state == TrackState.LOST and t.time_since_update > self.max_lost:
                t.state = TrackState.DELETED

        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]

    # -- internals ---------------------------------------------------
    def _associate(self, tracks, detections):
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        cost = np.ones((len(tracks), len(detections)), dtype=float)
        for i, t in enumerate(tracks):
            for j, d in enumerate(detections):
                cost[i, j] = 1.0 - iou(t.bbox, d[:4])
        rows, cols = linear_sum_assignment(cost.tolist())
        matches, u_tracks = [], list(range(len(tracks)))
        u_dets = list(range(len(detections)))
        for r, c in zip(rows, cols):
            if cost[r, c] <= self.max_iou_dist:
                matches.append((r, c))
                u_tracks.remove(r)
                u_dets.remove(c)
        return matches, u_tracks, u_dets

    def reset(self):
        self.tracks = []
        Track._next_id = 0
