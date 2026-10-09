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

try:
    from .appearance import AppearanceEmbedding, ReIDGallery
except ImportError:  # appearance is optional until first use
    AppearanceEmbedding, ReIDGallery = None, None


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
        self.embedding = None  # appearance vector (EMA-smoothed)

    def predict(self):
        self.bbox = self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, score, embedding=None):
        self.bbox = self.kf.update(bbox)
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        self.history.append(tuple(self.bbox))
        if embedding is not None:
            # Exponential moving average: adapt to slow appearance drift
            # (lighting, pose) while resisting single-frame noise.
            if self.embedding is None:
                self.embedding = embedding
            else:
                v = 0.3 * embedding + 0.7 * self.embedding
                n = float((v ** 2).sum() ** 0.5)
                self.embedding = v / n if n > 1e-9 else v
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
                 n_init=3, max_age=30, max_lost=60,
                 use_reid=False, reid_thresh=0.6):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_iou_dist = max_iou_dist  # min IoU = 1 - this
        self.n_init = n_init
        self.max_age = max_age      # frames before Tentative -> Deleted
        self.max_lost = max_lost    # frames before Lost -> Deleted
        self.use_reid = use_reid
        self.reid_thresh = reid_thresh
        self.tracks = []
        self.id_switches = 0  # self-diagnostic counter
        self._embedder = None
        self._gallery = None
        if use_reid:
            from .appearance import AppearanceEmbedding, ReIDGallery
            self._embedder = AppearanceEmbedding()
            self._gallery = ReIDGallery()

    # -- public ------------------------------------------------------
    def update(self, detections, frame=None):
        """detections: [[x1,y1,x2,y2,score], ...].
        frame: optional BGR image, needed for appearance embeddings.
        Returns confirmed Tracks."""
        high = [d for d in detections if d[4] >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d[4] < self.high_thresh]

        # Appearance vectors for high detections (stage-3 re-ID fuel)
        det_embs = []
        if self.use_reid and frame is not None:
            det_embs = [self._embedder.embed(frame, d[:4]) for d in high]
        else:
            det_embs = [None] * len(high)

        for t in self.tracks:
            t.predict()

        live = [t for t in self.tracks if t.state != TrackState.DELETED]

        # Stage 1: high-confidence detections
        m1, u_tracks, u_high = self._associate(live, high)
        for ti, di in m1:
            live[ti].update(high[di][:4], high[di][4], det_embs[di])
            if self.use_reid:
                self._gallery.update(live[ti].id, live[ti].embedding)

        # Stage 2: low-confidence detections rescue unmatched tracks
        remaining = [live[i] for i in u_tracks]
        m2, u_tracks2, _ = self._associate(remaining, low)
        for ti, di in m2:
            remaining[ti].update(low[di][:4], low[di][4])
        for i in u_tracks2:
            remaining[i].mark_lost()

        # Stage 3: appearance re-ID -- match leftover LOST tracks against
        # unmatched high detections by looks, not position. This is what
        # survives long occlusions where Kalman prediction has drifted.
        if self.use_reid and frame is not None:
            lost = [t for t in remaining
                    if t.state == TrackState.LOST and t.embedding is not None]
            u_high = self._reid_match(lost, high, u_high, det_embs)

        # New tracks from still-unmatched high detections
        for di in u_high:
            t = Track(high[di][:4], high[di][4], self.n_init)
            t.embedding = det_embs[di]
            self.tracks.append(t)

        # Age out
        for t in self.tracks:
            if t.state == TrackState.TENTATIVE and t.time_since_update > self.max_age:
                t.state = TrackState.DELETED
            elif t.state == TrackState.LOST and t.time_since_update > self.max_lost:
                t.state = TrackState.DELETED
                if self.use_reid:
                    self._gallery.forget(t.id)

        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]

    def _reid_match(self, lost_tracks, high, u_high, det_embs):
        """Greedy appearance matching. Returns remaining unmatched det indices."""
        if not lost_tracks or not u_high:
            return u_high
        scored = []  # (score, track, det_idx)
        for t in lost_tracks:
            for di in u_high:
                emb = det_embs[di]
                if emb is None:
                    continue
                s = AppearanceEmbedding.similarity(emb, t.embedding)
                if s >= self.reid_thresh:
                    scored.append((s, t, di))
        scored.sort(reverse=True)
        used_tracks, used_dets = set(), set()
        for s, t, di in scored:
            if t.id in used_tracks or di in used_dets:
                continue
            used_tracks.add(t.id)
            used_dets.add(di)
            t.update(high[di][:4], high[di][4], det_embs[di])
            if self.use_reid:
                self._gallery.update(t.id, t.embedding)
            self.id_switches += 0  # re-ID by design, not a switch
        return [di for di in u_high if di not in used_dets]

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
