"""Re-ID tests: appearance embeddings + long-occlusion recovery.

Scenario: a RED target and a BLUE target. Red vanishes for 40 frames
(longer than Kalman alone can bridge reliably at speed) and returns
at a *different* position. Appearance matching should recover its ID
instead of spawning a new one.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.appearance import AppearanceEmbedding, ReIDGallery
from src.tracker import MultiTracker, TrackState


def solid_frame(color_bgr, box, size=(200, 200)):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    x1, y1, x2, y2 = [int(v) for v in box]
    img[y1:y2, x1:x2] = color_bgr
    return img


def det(cx, cy, w=30, h=30, score=0.9):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score]


def test_embedding_similarity():
    em = AppearanceEmbedding()
    red = solid_frame((0, 0, 255), (50, 50, 100, 100))
    red2 = solid_frame((0, 0, 255), (60, 60, 110, 110))
    blue = solid_frame((255, 0, 0), (50, 50, 100, 100))
    v_red = em.embed(red, (50, 50, 100, 100))
    v_red2 = em.embed(red2, (60, 60, 110, 110))
    v_blue = em.embed(blue, (50, 50, 100, 100))
    assert v_red is not None
    same = AppearanceEmbedding.similarity(v_red, v_red2)
    diff = AppearanceEmbedding.similarity(v_red, v_blue)
    assert same > 0.95, same
    assert diff < same - 0.3, (diff, same)


def test_embedding_rejects_tiny_box():
    em = AppearanceEmbedding()
    img = np.zeros((200, 200, 3), np.uint8)
    assert em.embed(img, (10, 10, 11, 11)) is None


def test_gallery_best_match():
    em = AppearanceEmbedding()
    g = ReIDGallery()
    red = solid_frame((0, 0, 255), (50, 50, 100, 100))
    blue = solid_frame((255, 0, 0), (50, 50, 100, 100))
    g.update(1, em.embed(red, (50, 50, 100, 100)))
    g.update(2, em.embed(blue, (50, 50, 100, 100)))
    tid, score = g.best_match(em.embed(red, (55, 55, 105, 105)), [1, 2])
    assert tid == 1 and score > 0.9
    tid, _ = g.best_match(em.embed(red, (55, 55, 105, 105)), [2])
    assert tid is None  # blue is the only candidate: no match
    g.forget(1)
    tid, _ = g.best_match(em.embed(red, (55, 55, 105, 105)), [1])
    assert tid is None


def test_reid_recovers_long_occlusion():
    tr = MultiTracker(n_init=2, max_lost=100, use_reid=True, reid_thresh=0.5)
    red, blue = (0, 0, 255), (255, 0, 0)
    red_id = None
    # Phase 1: both visible, let embeddings + IDs settle
    for f in range(10):
        frame = np.zeros((200, 200, 3), np.uint8)
        frame[80:110, 20 + f:50 + f] = red
        frame[80:110, 150 - f:180 - f] = blue
        tracks = tr.update([det(35 + f, 95), det(165 - f, 95)], frame=frame)
        for t in tracks:
            if abs((t.bbox[0] + t.bbox[2]) / 2 - (35 + f)) < 20:
                red_id = t.id
    assert red_id is not None
    # Phase 2: red gone for 40 frames (blue keeps moving)
    for f in range(40):
        frame = np.zeros((200, 200, 3), np.uint8)
        frame[80:110, 140 - f:170 - f] = blue
        tr.update([det(155 - f, 95)], frame=frame)
    # Phase 3: red returns at a NEW position (Kalman prediction is stale)
    for f in range(6):
        frame = np.zeros((200, 200, 3), np.uint8)
        frame[80:110, 140:170] = red
        frame[80:110, 100 - f:130 - f] = blue
        tracks = tr.update([det(155, 95), det(115 - f, 95)], frame=frame)
    ids = sorted(t.id for t in tracks if t.state == TrackState.CONFIRMED)
    assert red_id in ids, f"red got a new ID after long occlusion: {ids}"
    assert len(ids) == 2, f"expected 2 tracks, got {ids}"
