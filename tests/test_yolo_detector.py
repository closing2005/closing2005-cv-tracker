"""YOLO post-processing tests (no weights needed).

We test the math that turns a raw (84, 8400) tensor into image-space boxes:
letterbox geometry and output parsing. The DNN forward itself is OpenCV's
job, not ours.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.yolo_detector import _letterbox, _parse_output


def test_letterbox_keeps_aspect():
    import cv2
    img = np.zeros((480, 640, 3), np.uint8)  # 4:3 frame
    blob, scale, pw, ph = _letterbox(img, 640)
    assert blob.shape == (1, 3, 640, 640)
    assert abs(scale - 1.0) < 1e-6          # 640 wide -> scale 1.0
    assert pw == 0 and ph == 80             # 480 tall -> 80px top/bottom pad


def test_letterbox_portrait():
    import cv2
    img = np.zeros((800, 400, 3), np.uint8)
    blob, scale, pw, ph = _letterbox(img, 640)
    assert abs(scale - 0.8) < 1e-6
    assert pw == 160 and ph == 0


def _fake_output():
    # (84, 8400): one strong person detection, one weak car, rest noise
    out = np.zeros((84, 8400), dtype=np.float32)
    out[0, 0], out[1, 0], out[2, 0], out[3, 0] = 320, 320, 100, 200
    out[4 + 0, 0] = 0.9                       # class 0 = person
    out[0, 1], out[1, 1], out[2, 1], out[3, 1] = 100, 100, 50, 50
    out[4 + 2, 1] = 0.2                       # class 2 = car, weak
    return out


def test_parse_output_filters_by_conf():
    boxes, scores = _parse_output(_fake_output(), conf_thresh=0.5, keep=None)
    assert len(boxes) == 1
    assert boxes[0] == [320.0, 320.0, 100.0, 200.0]
    assert abs(scores[0] - 0.9) < 1e-6


def test_parse_output_class_filter():
    boxes, scores = _parse_output(_fake_output(), conf_thresh=0.1, keep={2})
    assert len(boxes) == 1          # only the car survives
    assert abs(scores[0] - 0.2) < 1e-6


def test_parse_output_empty():
    out = np.zeros((84, 8400), dtype=np.float32)
    boxes, scores = _parse_output(out, conf_thresh=0.5, keep=None)
    assert boxes == [] and scores == []
