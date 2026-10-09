"""Multi-camera handover test: one red target, two cameras."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.multicam import MultiCameraTracker
from src.tracker import TrackState


def det(cx, cy, w=30, h=30, score=0.9):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score]


def red_frame(x, size=(200, 200)):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    img[80:110, x:x + 30] = (0, 0, 255)
    return img


def test_handover_keeps_global_id():
    mc = MultiCameraTracker(n_cams=2, reid_thresh=0.5, handover_window=300)
    gid_cam0 = None
    # Cam 0: red target walks across for 8 frames
    for f in range(8):
        res = mc.update(0, [det(40 + 10 * f, 95)], frame=red_frame(25 + 10 * f))
        for gid, t in res:
            gid_cam0 = gid
    assert gid_cam0 is not None
    # Gap: 20 frames, nobody anywhere
    for _ in range(20):
        mc.update(0, [], frame=np.zeros((200, 200, 3), np.uint8))
        mc.update(1, [], frame=np.zeros((200, 200, 3), np.uint8))
    # Cam 1: "same" red target appears (different local track)
    gid_cam1 = None
    for f in range(8):
        res = mc.update(1, [det(40 + 10 * f, 95)], frame=red_frame(25 + 10 * f))
        for gid, t in res:
            gid_cam1 = gid
    assert gid_cam1 is not None, "cam1 track never confirmed"
    assert gid_cam1 == gid_cam0, \
        f"handover failed: cam0 global {gid_cam0} != cam1 global {gid_cam1}"


def test_different_targets_get_different_ids():
    mc = MultiCameraTracker(n_cams=2, reid_thresh=0.5)
    gids = set()
    for f in range(8):
        for gid, _ in mc.update(0, [det(50, 95)], frame=red_frame(35)):
            gids.add(gid)
    blue = np.zeros((200, 200, 3), np.uint8)
    blue[80:110, 35:65] = (255, 0, 0)
    for f in range(8):
        for gid, _ in mc.update(1, [det(50, 95)], frame=blue):
            gids.add(gid)
    assert len(gids) == 2, f"expected 2 global IDs, got {gids}"


def test_expired_handover_gets_new_id():
    mc = MultiCameraTracker(n_cams=2, reid_thresh=0.5, handover_window=5)
    gid0 = None
    for f in range(8):
        for gid, _ in mc.update(0, [det(50, 95)], frame=red_frame(35)):
            gid0 = gid
    for _ in range(50):  # way beyond the handover window
        mc.update(0, [], frame=np.zeros((200, 200, 3), np.uint8))
        mc.update(1, [], frame=np.zeros((200, 200, 3), np.uint8))
    gid1 = None
    for f in range(8):
        for gid, _ in mc.update(1, [det(50, 95)], frame=red_frame(35)):
            gid1 = gid
    assert gid1 is not None and gid1 != gid0
