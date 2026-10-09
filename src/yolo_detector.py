"""YOLO detector plug-in: DNN-based detection via cv2.dnn, no ultralytics needed.

Wires a YOLOv8/YOLO11 ONNX model into the Detector interface, so the tracker
gets real object detections (person, car, ...) instead of motion blobs.

Why this matters vs. MotionDetector:
  - works with a *moving* camera (no background model to corrupt)
  - detects *still* targets (a parked car, a standing person)
  - reports semantic classes, so you can track only people, only cars, ...

Trade-off: needs a ~11MB weight file and ~30-80ms/frame on CPU.
See docs/yolo_detector.md for the theory (output tensor, NMS, letterbox).
"""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detector

# COCO 80 classes, in model output order
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class YoloDetector(Detector):
    def __init__(self, weights_path, conf_thresh=0.5, nms_thresh=0.45,
                 input_size=640, classes=None):
        """weights_path: yolov8n.onnx / yolo11n.onnx etc.
        classes: None (all) or list like ["person", "car"].
        """
        self.net = cv2.dnn.readNet(weights_path)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        if classes is None:
            self.keep = None
        else:
            self.keep = {COCO_NAMES.index(c) for c in classes}

    # -- Detector interface -------------------------------------------
    def detect(self, frame):
        h0, w0 = frame.shape[:2]
        blob, scale, pad_w, pad_h = _letterbox(frame, self.input_size)
        self.net.setInput(blob)
        out = self.net.forward()[0]  # (84, 8400): 4 box + 80 classes
        boxes, scores = _parse_output(out, self.conf_thresh, self.keep)
        if not boxes:
            return []
        idx = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thresh,
                               self.nms_thresh)
        idx = np.array(idx).flatten()
        results = []
        for i in idx:
            cx, cy, w, h = boxes[i]
            # back to original image coordinates
            x1 = (cx - w / 2 - pad_w) / scale
            y1 = (cy - h / 2 - pad_h) / scale
            x2 = (cx + w / 2 - pad_w) / scale
            y2 = (cy + h / 2 - pad_h) / scale
            results.append([max(x1, 0), max(y1, 0),
                            min(x2, w0), min(y2, h0),
                            float(scores[i])])
        results.sort(key=lambda b: b[4], reverse=True)
        return results


def _letterbox(frame, size):
    """Resize keeping aspect ratio, pad with gray. Returns (blob, scale, pw, ph)."""
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))
    pad_w, pad_h = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
    blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (size, size),
                                 swapRB=True, crop=False)
    return blob, scale, pad_w, pad_h


def _parse_output(out, conf_thresh, keep):
    """out: (84, 8400). Returns (boxes_cxcywh_in_640px, scores)."""
    out = out.T  # (8400, 84)
    boxes, scores = [], []
    for row in out:
        cls_scores = row[4:]
        cls_id = int(np.argmax(cls_scores))
        score = float(cls_scores[cls_id])
        if score < conf_thresh:
            continue
        if keep is not None and cls_id not in keep:
            continue
        boxes.append([float(row[0]), float(row[1]),
                      float(row[2]), float(row[3])])
        scores.append(score)
    return boxes, scores
