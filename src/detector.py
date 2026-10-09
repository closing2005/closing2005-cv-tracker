"""Detectors: turn frames into [x1, y1, x2, y2, score] boxes.

The built-in MotionDetector needs no weights and runs anywhere --
background subtraction (MOG2) + morphology + contour filtering.
For real deployments, subclass Detector and plug in a DNN
(e.g. YOLO via cv2.dnn); the tracker only cares about the box list.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class Detector(ABC):
    @abstractmethod
    def detect(self, frame):
        """Return list of [x1, y1, x2, y2, score]."""
        raise NotImplementedError


class MotionDetector(Detector):
    """MOG2 background subtraction detector for static cameras."""

    def __init__(self, min_area=500, history=500, var_threshold=16,
                 open_ksize=3, close_ksize=15, learning_rate=-1):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True
        )
        self.min_area = min_area
        self.learning_rate = learning_rate
        self.open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
        )
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
        )

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg = self.bg.apply(gray, learningRate=self.learning_rate)
        # Drop shadows (gray 127) -> binary foreground
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.open_kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.close_kernel)

        contours, _ = cv2.findContours(
            fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        h, w = frame.shape[:2]
        frame_area = float(w * h)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Score: anything above min_area starts at 0.55 (motion blobs
            # are real, just often partial); saturates for large blobs.
            # Fill-ratio bonus: solid objects outrank wispy noise.
            rect_area = max(bw * bh, 1.0)
            fill = area / rect_area
            score = 0.55 + 0.25 * min(area / (frame_area * 0.02), 1.0) \
                + 0.15 * fill
            score = float(min(score, 0.95))
            boxes.append([float(x), float(y), float(x + bw), float(y + bh), score])
        # Largest first: helps the two-stage matcher prioritize.
        boxes.sort(key=lambda b: b[4], reverse=True)
        return boxes


class DnnDetector(Detector):
    """Template for weight-based detectors (plug your own model here).

    Example wiring (not bundled -- bring your own .onnx/.weights):
        net = cv2.dnn.readNet("yolov8n.onnx")
        det = DnnDetector(lambda f: run_yolo(net, f))
    """

    def __init__(self, infer_fn):
        self.infer_fn = infer_fn

    def detect(self, frame):
        return self.infer_fn(frame)
