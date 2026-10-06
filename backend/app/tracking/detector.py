"""
detector.py — YOLO-based cricket ball detector.

THE ONLY BALL DETECTOR IN THE SYSTEM.
docs/UNIFIED_ARCHITECTURE.md §1 / C2: YOLO + Kalman run exactly once per
uploaded video, never once per delivery. No other detector exists in this
project, and the shot pipeline has no access to this class.

Responsibilities:
  - Load the YOLO11 model once.
  - Run inference on a resized frame (for speed).
  - Return the best ball bounding-box centre or None.
  - Reject obviously-wrong detections (size, aspect ratio).
  - Return the winning box's confidence alongside the centre so trajectory
    provenance can be persisted.
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from ..core.config import BALL_MODEL_PATH
from ..core.constants import (
    DETECTOR_CONF_THRESHOLD,
    DETECTOR_MAX_DIAG_PX,
    DETECTOR_MIN_DIAG_PX,
    DETECTOR_RESIZE_H,
    DETECTOR_RESIZE_W,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = str(BALL_MODEL_PATH)

CONF_THRESHOLD = DETECTOR_CONF_THRESHOLD   # minimum detection confidence
RESIZE_W       = DETECTOR_RESIZE_W         # inference width (keeps aspect-ratio)
RESIZE_H       = DETECTOR_RESIZE_H         # inference height

# Reject detections whose bounding-box diagonal is outside this pixel range
# (relative to RESIZE_W × RESIZE_H — helps filter sponsor logos / crowd noise)
MIN_DIAG_PX = DETECTOR_MIN_DIAG_PX
MAX_DIAG_PX = DETECTOR_MAX_DIAG_PX


class BallDetector:
    """Wraps YOLO inference for cricket ball detection."""

    def __init__(self, model_path: str | None = None):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model_path = model_path or MODEL_PATH
        print(f"🚀 Detector using device: {self.device}")
        try:
            self.model = YOLO(self.model_path)
            print("✅ YOLO model loaded")
        except Exception as exc:
            print(f"❌ Failed to load model: {exc}")
            self.model = None

    # ── Public API ────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> tuple | None:
        """
        Run inference on *frame* (full-resolution BGR).
        Returns (cx, cy) in full-resolution pixel coords, or None.

        The frame is internally downscaled to RESIZE_W×RESIZE_H for speed,
        then the result is scaled back to the original resolution.
        """
        cx, cy, _conf = self.detect_with_confidence(frame)
        if cx is None:
            return None
        return (cx, cy)

    def detect_with_confidence(self, frame: np.ndarray) -> tuple[int | None, int | None, float | None]:
        """
        Same as :meth:`detect` but also returns the winning box confidence.

        Returned coordinates are ALWAYS in full-resolution pixel space, scaled
        back by (w_full / RESIZE_W, h_full / RESIZE_H). Every downstream consumer
        — stored trajectory, bounce marker, overlay renderer — relies on that
        space, so nothing may resize a frame without rescaling coordinates by
        the same factor. See UNIFIED_ARCHITECTURE.md §6.3.
        """
        if self.model is None:
            return None, None, None

        h_full, w_full = frame.shape[:2]
        small = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        result = self.model.predict(
            source=small, conf=CONF_THRESHOLD, verbose=False, device=self.device
        )[0]

        sx = w_full / RESIZE_W
        sy = h_full / RESIZE_H

        best_center = None
        best_conf = -1.0

        if result.boxes is None:
            return None, None, None

        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = box
            if int(cls) != 0:
                continue

            # Size sanity check on resized coords
            diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if not (MIN_DIAG_PX <= diag <= MAX_DIAG_PX):
                continue

            # Prefer highest-confidence detection
            if score > best_conf:
                best_conf = score
                best_center = (
                    int((x1 + x2) / 2 * sx),
                    int((y1 + y2) / 2 * sy),
                )

        if best_center is None:
            return None, None, None

        return best_center[0], best_center[1], round(float(best_conf), 4)
