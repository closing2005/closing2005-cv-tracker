"""Multi-camera tracking: one global identity across camera views.

Each camera runs its own MultiTracker. A global ReID gallery maps
(cam_idx, local_id) -> global_id: when a *new* confirmed track appears
in camera B, its appearance is matched against the gallery of *other*
cameras. A strong match reuses the global ID -- the target "handed off"
from one view to the next.

This is the pragmatic subset of multi-camera tracking: no calibration,
no overlapping views required. It answers "is the person in cam 2 the
same one who left cam 1?" using appearance + a time window.
"""

from __future__ import annotations

from .appearance import AppearanceEmbedding, ReIDGallery
from .tracker import MultiTracker, TrackState


class MultiCameraTracker:
    def __init__(self, n_cams, reid_thresh=0.65, handover_window=300,
                 **tracker_kwargs):
        """handover_window: frames within which a cross-camera match is
        allowed after the target vanished from the other camera."""
        tracker_kwargs.setdefault("use_reid", True)
        self.trackers = [MultiTracker(**tracker_kwargs) for _ in range(n_cams)]
        self.n_cams = n_cams
        self.reid_thresh = reid_thresh
        self.handover_window = handover_window
        self._gallery = ReIDGallery()
        self._gid_of = {}        # (cam, local_id) -> global_id
        self._last_seen = {}     # global_id -> frame_idx
        self._next_gid = 1
        self._frame = 0

    def update(self, cam_idx, detections, frame=None):
        """Returns [(global_id, Track), ...] for confirmed tracks."""
        self._frame += 1
        tracks = self.trackers[cam_idx].update(detections, frame=frame)
        out = []
        for t in tracks:
            key = (cam_idx, t.id)
            if key not in self._gid_of:
                self._gid_of[key] = self._resolve_global_id(t)
            gid = self._gid_of[key]
            self._last_seen[gid] = self._frame
            if t.embedding is not None:
                self._gallery.update(gid, t.embedding)
            out.append((gid, t))
        return out

    def _resolve_global_id(self, track):
        # Try to match against other cameras' recent global IDs.
        if track.embedding is not None:
            candidates = [
                gid for gid, seen in self._last_seen.items()
                if self._frame - seen <= self.handover_window
            ]
            gid, score = self._gallery.best_match(
                track.embedding, candidates, thresh=self.reid_thresh)
            if gid is not None:
                return gid
        gid = self._next_gid
        self._next_gid += 1
        return gid

    def global_track_count(self):
        return self._next_gid - 1
