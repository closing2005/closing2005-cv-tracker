"""Appearance embeddings for re-identification across long occlusions.

Zero-weight approach: HSV color histograms are surprisingly effective for
short-term re-ID (same camera, same lighting) and cost microseconds.
The interface is pluggable -- swap in a deep embedding (OSNet etc.) via
cv2.dnn without touching the tracker.

Design:
  AppearanceEmbedding  - frame crop -> L2-normalized feature vector
  ReIDGallery          - track_id -> smoothed embedding, cosine matching
"""

from __future__ import annotations

import cv2
import numpy as np


class AppearanceEmbedding:
    """HSV histogram embedding: H(30) x S(32) bins, L2-normalized."""

    def __init__(self, h_bins=30, s_bins=32):
        self.h_bins = h_bins
        self.s_bins = s_bins

    def embed(self, frame, bbox):
        """bbox: [x1, y1, x2, y2]. Returns normalized vector or None."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None,
                            [self.h_bins, self.s_bins], [0, 180, 0, 256])
        vec = hist.flatten().astype(np.float32)
        n = np.linalg.norm(vec)
        return vec / n if n > 1e-9 else None

    @staticmethod
    def similarity(a, b):
        """Cosine similarity of two normalized vectors."""
        return float(np.dot(a, b))


class ReIDGallery:
    """Per-track smoothed embeddings + best-match lookup."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha          # EMA update rate
        self._emb = {}              # track_id -> vector

    def update(self, track_id, vec):
        if vec is None:
            return
        if track_id in self._emb:
            v = self.alpha * vec + (1 - self.alpha) * self._emb[track_id]
            n = np.linalg.norm(v)
            self._emb[track_id] = v / n if n > 1e-9 else v
        else:
            self._emb[track_id] = vec

    def forget(self, track_id):
        self._emb.pop(track_id, None)

    def best_match(self, vec, candidates, thresh=0.6):
        """candidates: iterable of track ids. Returns (track_id, score)
        of the best match above thresh, else (None, 0.0)."""
        best_id, best_s = None, 0.0
        for tid in candidates:
            if tid not in self._emb:
                continue
            s = AppearanceEmbedding.similarity(vec, self._emb[tid])
            if s > best_s:
                best_id, best_s = tid, s
        if best_s >= thresh:
            return best_id, best_s
        return None, 0.0
