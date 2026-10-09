"""Tracker tests on synthetic detection sequences (no camera needed).

Scenario: two targets move on straight lines for 60 frames.
Target B is fully occluded for frames 30-34, with weak (low-score)
detections on frames 32-33 -- stage-2 association should rescue it
instead of spawning a new ID.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tracker import MultiTracker, TrackState


def box(cx, cy, w=40, h=40, score=0.9):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score]


def run_sequence():
    tr = MultiTracker(high_thresh=0.5, low_thresh=0.15, n_init=3,
                      max_age=30, max_lost=60)
    TrackState  # noqa - keep import used
    id_of_a, id_of_b = None, None
    for f in range(60):
        dets = [box(100 + 5 * f, 200)]                      # A: moving right
        if f < 30 or f > 34:
            dets.append(box(400, 100 + 4 * f))               # B: moving down
        elif f in (32, 33):
            dets.append(box(400, 100 + 4 * f, score=0.2))    # weak glimpse
        tracks = tr.update(dets)
        for t in tracks:
            cx = (t.bbox[0] + t.bbox[2]) / 2
            if cx < 300 and id_of_a is None:
                id_of_a = t.id
            if cx > 300 and id_of_b is None:
                id_of_b = t.id
    return tr, id_of_a, id_of_b


def test_two_targets_tracked():
    tr, _, _ = run_sequence()
    confirmed = [t for t in tr.tracks if t.state == TrackState.CONFIRMED]
    assert len(confirmed) == 2, f"expected 2 tracks, got {len(confirmed)}"


def test_no_id_switch_through_occlusion():
    tr, id_a, id_b = run_sequence()
    assert id_a is not None and id_b is not None and id_a != id_b
    # After the run, exactly the two original IDs should remain confirmed.
    ids = sorted(t.id for t in tr.tracks
                 if t.state == TrackState.CONFIRMED)
    assert ids == sorted([id_a, id_b]), f"ID switch detected: {ids}"


def test_tentative_ghosts_are_filtered():
    tr = MultiTracker(n_init=3, max_age=5)
    # Single-frame flicker: never enough hits to confirm.
    for _ in range(10):
        tr.update([box(50, 50)])
        tr.update([])
    assert tr.update([]) == []
    assert all(t.state != TrackState.CONFIRMED for t in tr.tracks)


def test_lost_tracks_eventually_deleted():
    tr = MultiTracker(n_init=1, max_lost=5)
    tr.update([box(100, 100)])
    assert len(tr.update([box(102, 100)])) == 1
    for _ in range(10):
        tr.update([])
    assert tr.tracks == []
