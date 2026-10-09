"""Single-target Kalman filter with a constant-velocity motion model.

State: [cx, cy, w, h, vx, vy, vw, vh]  (center, size and their velocities)
Measurement: [cx, cy, w, h]

The process-noise covariance scales with the target's size so that large
boxes tolerate larger motion uncertainty -- a small adaptive touch that
keeps the filter stable across scales without per-scene tuning.
"""

from __future__ import annotations

import cv2
import numpy as np


class KalmanBoxTracker:
    def __init__(self, bbox, process_noise=1.0, meas_noise=10.0):
        """bbox: [x1, y1, x2, y2] initial detection."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = x2 - x1, y2 - y1

        self.kf = cv2.KalmanFilter(8, 4)
        # State transition: constant velocity
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0
        # Measurement: observe position/size only
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1.0
        self.kf.measurementMatrix[1, 1] = 1.0
        self.kf.measurementMatrix[2, 2] = 1.0
        self.kf.measurementMatrix[3, 3] = 1.0

        # Adaptive process noise: scale with box area so big, fast-moving
        # targets get a wider gate than small distant ones.
        area = max(w * h, 1.0)
        q_pos = np.float32(process_noise * (1.0 + np.log10(area) / 4.0))
        q = np.eye(8, dtype=np.float32) * q_pos
        q[4:, 4:] *= 4.0  # velocities are noisier than positions
        self.kf.processNoiseCov = q.astype(np.float32)

        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * meas_noise
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 10.0
        self.kf.statePost = np.array(
            [[cx], [cy], [w], [h], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32
        )

    def predict(self):
        """Advance one step; return predicted [x1, y1, x2, y2]."""
        s = self.kf.predict().flatten()
        return _state_to_bbox(s)

    def update(self, bbox):
        """Correct with a new detection [x1, y1, x2, y2]; return corrected box."""
        x1, y1, x2, y2 = bbox
        z = np.array(
            [[(x1 + x2) / 2.0], [(y1 + y2) / 2.0], [x2 - x1], [y2 - y1]],
            dtype=np.float32,
        )
        s = self.kf.correct(z).flatten()
        return _state_to_bbox(s)

    def velocity(self):
        """Current (vx, vy) in pixels/frame."""
        s = self.kf.statePost.flatten()
        return float(s[4]), float(s[5])


def _state_to_bbox(s):
    cx, cy, w, h = s[0], s[1], max(s[2], 1.0), max(s[3], 1.0)
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
