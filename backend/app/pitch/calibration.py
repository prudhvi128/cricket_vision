"""
calibration.py — From raw keypoints to a usable pitch calibration.

The pipeline already has a calibration concept: `analytics/calibration.py`
turns four OPERATOR-MEASURED corners into ground-plane metres, and it is
suppressed (`geometry_calibrated = False`) unless those corners were really
measured on this video. This module is a different thing and must not be
confused with it:

    analytics/calibration.py  = "where is the pitch in METRES?"  (operator)
    pitch/calibration.py      = "where is the pitch in THIS FRAME?" (model)

Nothing produced here feeds speed, length, line or swing. It produces a quad in
source pixels, an optional image→unit-square homography for normalised ball
positions, and the confidence/state a client needs to decide whether to draw
anything at all.

A VALID DETECTION IS EARNED, NOT ASSUMED
----------------------------------------
Every check below can refuse the model's answer: too few keypoints, a keypoint
outside the frame, a hull too small to be a pitch, a degenerate quad. A refusal
is a fact worth recording (`reason`), because "the model said nothing useful" is
the only honest output when the picture does not contain a readable pitch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..core.config import PitchConfig
from .detector import Keypoint, PitchDetection

Size = tuple[int, int]

# Ordered so the unit square reads top-left → top-right → bottom-right →
# bottom-left, i.e. the winding a perspective transform expects.
_CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")
_UNIT_SQUARE = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
)

# How far outside the frame a keypoint may sit before the detection is refused.
# Models routinely land a corner a few pixels past the edge; a corner halfway to
# the next frame is a different object.
_BOUND_SLACK_FRACTION = 0.02


@dataclass(frozen=True)
class PitchCalibration:
    """
    A validated pitch quad for one frame, in SOURCE pixel coordinates.

    `corners` and `homography` are empty/None whenever fewer than four usable
    keypoints survived validation — a three-keypoint model cannot describe a
    rectangle, and inventing the fourth corner would draw a boundary the model
    never reported.
    """

    frame_number: int
    image_size: Size
    keypoints: dict[str, tuple[float, float]]
    corners: dict[str, tuple[float, float]]
    confidence: Optional[float]
    homography: Optional[np.ndarray] = None

    @property
    def width(self) -> float:
        """Mean top/bottom span of the quad, in pixels."""
        return _span(self.corners, horizontal=True)

    @property
    def length(self) -> float:
        """Mean left/right span of the quad, in pixels."""
        return _span(self.corners, horizontal=False)

    @property
    def center(self) -> Optional[tuple[float, float]]:
        if self.corners:
            xs = [p[0] for p in self.corners.values()]
            ys = [p[1] for p in self.corners.values()]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        if self.keypoints:
            xs = [p[0] for p in self.keypoints.values()]
            ys = [p[1] for p in self.keypoints.values()]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        return None

    @property
    def hull_area_fraction(self) -> float:
        """Share of the frame the keypoint hull covers."""
        if len(self.keypoints) < 3:
            return 0.0
        pts = np.array(list(self.keypoints.values()), dtype=np.float32)
        area = float(cv2.contourArea(cv2.convexHull(pts)))
        w, h = self.image_size
        return area / float(w * h) if w > 0 and h > 0 else 0.0

    def image_to_pitch(self, x: float, y: float) -> Optional[tuple[float, float]]:
        """
        Source pixel → normalised pitch coordinates (0..1 across the quad).

        None when there is no homography: with three keypoints there is no
        transform, and a client must receive "unavailable" rather than a number
        computed from a quad that was never measured.
        """
        if self.homography is None:
            return None
        vec = np.array([x, y, 1.0], dtype=np.float64)
        mapped = self.homography @ vec
        w = mapped[2]
        if not np.isfinite(w) or abs(w) < 1e-9:
            return None
        u, v = float(mapped[0] / w), float(mapped[1] / w)
        if not (np.isfinite(u) and np.isfinite(v)):
            return None
        return u, v

    def inside_pitch(self, x: float, y: float) -> bool:
        mapped = self.image_to_pitch(x, y)
        if mapped is None:
            return False
        return -1e-6 <= mapped[0] <= 1 + 1e-6 and -1e-6 <= mapped[1] <= 1 + 1e-6

    def as_dict(self) -> dict:
        """
        The persisted form. Deliberately includes the homography: the frontend
        serializer projects it onto every trajectory point, and re-deriving it
        later would mean the stored calibration and the drawn positions could
        disagree.
        """
        return {
            "frame_number": self.frame_number,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "keypoints": {k: [round(v[0], 2), round(v[1], 2)] for k, v in self.keypoints.items()},
            "corners": {k: [round(v[0], 2), round(v[1], 2)] for k, v in self.corners.items()},
            "center": (
                [round(self.center[0], 2), round(self.center[1], 2)]
                if self.center else None
            ),
            "confidence": None if self.confidence is None else round(self.confidence, 4),
            "width_px": round(self.width, 2) if self.corners else None,
            "length_px": round(self.length, 2) if self.corners else None,
            "homography": (
                [[round(float(v), 8) for v in row] for row in self.homography]
                if self.homography is not None else None
            ),
        }


def build_calibration(
    detection: PitchDetection,
    *,
    cfg: PitchConfig,
    image_size: Size,
) -> tuple[Optional[PitchCalibration], str]:
    """
    Validate a detection and build the calibration it justifies.

    Returns `(calibration, "")` or `(None, reason)`; the reason is for the
    analysis record, never for the client (a client gets the state machine's
    four states, not the refusals behind them).
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        return None, "unknown_frame_size"

    points = _drop_low_confidence(detection.keypoints, cfg.min_confidence)
    points = _drop_out_of_bounds(points, image_size)
    if len(points) < max(1, cfg.min_keypoints):
        return None, f"only_{len(points)}_usable_keypoints"

    coords = [(p.x, p.y) for p in points]
    keypoints = {p.name: (p.x, p.y) for p in points}

    if _hull_area_fraction(coords, image_size) < cfg.min_area_fraction:
        return None, "keypoint_hull_too_small"

    # `derive_corners` has no rectangle in three points. The empty dict (not
    # None) is what the snapshot and `as_dict` iterate, so a three-keypoint
    # calibration is a first-class result rather than a crash.
    corners = derive_corners(coords) or {}
    confidence = detection.confidence
    if confidence is None:
        confs = [p.confidence for p in points if p.confidence is not None]
        confidence = sum(confs) / len(confs) if confs else None

    homography = None
    if corners:
        homography = build_homography(corners, image_size)
        if homography is None:
            corners = {}

    return (
        PitchCalibration(
            frame_number=detection.frame_number,
            image_size=image_size,
            keypoints=keypoints,
            corners=corners,
            confidence=None if confidence is None else round(float(confidence), 4),
            homography=homography,
        ),
        "",
    )


def derive_corners(
    coords: list[tuple[float, float]],
) -> Optional[dict[str, tuple[float, float]]]:
    """
    Order arbitrary keypoints into a perspective rectangle.

    Names are ignored on purpose: whether the model calls them `point_1..4` or
    `stump_left`, the four extremes of the convex hull are the same four corners.
    Three keypoints yield None — there is no rectangle to find in three points.
    """
    if len(coords) < 4:
        return None
    pts = np.array(coords, dtype=np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    candidates = [(float(p[0]), float(p[1])) for p in hull]
    if len(candidates) < 4:
        return None

    top_left = min(candidates, key=lambda p: p[0] + p[1])
    bottom_right = max(candidates, key=lambda p: p[0] + p[1])
    top_right = max(candidates, key=lambda p: p[0] - p[1])
    bottom_left = min(candidates, key=lambda p: p[0] - p[1])

    corners = {
        "top_left": top_left,
        "top_right": top_right,
        "bottom_left": bottom_left,
        "bottom_right": bottom_right,
    }
    if len(set(corners.values())) != 4:
        # Collinear/degenerate extremes: the hull is a line or a triangle.
        return None
    return corners


def build_homography(
    corners: dict[str, tuple[float, float]], image_size: Size
) -> Optional[np.ndarray]:
    """
    Source pixels → unit square. None when the quad cannot define one.

    The unit square is a PROPERTY OF THE QUAD, not of any particular real-world
    pitch dimensions: it says "this is where the ball was, as a fraction of the
    pitch we can see", which is resolution-independent and camera-independent —
    and, unlike metres, it does not pretend the model told us how wide the
    bowling end is.
    """
    if set(_CORNER_ORDER) - set(corners):
        return None
    src = np.array([corners[k] for k in _CORNER_ORDER], dtype=np.float64)
    if not np.isfinite(src).all():
        return None
    try:
        matrix = cv2.getPerspectiveTransform(src.astype(np.float32), _UNIT_SQUARE.astype(np.float32))
    except cv2.error:
        return None
    if matrix is None or not np.isfinite(matrix).all():
        return None
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        return None
    # Round trip: each corner must land on its own vertex of the square.
    for expected, point in zip(_UNIT_SQUARE, src):
        vec = np.array([point[0], point[1], 1.0])
        mapped = matrix @ vec
        if abs(mapped[2]) < 1e-12:
            return None
        got = mapped[:2] / mapped[2]
        if not np.allclose(got, expected, atol=1e-3):
            return None
    return matrix


class PitchSmoother:
    """
    Exponential moving average over the keypoints.

    A boundary drawn from raw model output jitters visibly between frames; alpha
    trades responsiveness for stability. Points that disappear are forgotten on
    the next step, so a renamed or dropped keypoint cannot drag the average
    toward a coordinate that no longer exists.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = min(1.0, max(0.0, alpha))
        self._points: dict[str, tuple[float, float]] = {}
        self._confidence: Optional[float] = None

    def apply(
        self, keypoints: dict[str, tuple[float, float]], confidence: Optional[float]
    ) -> tuple[dict[str, tuple[float, float]], Optional[float]]:
        if self.alpha >= 1.0 or not self._points:
            self._points = dict(keypoints)
        else:
            a = self.alpha
            blended: dict[str, tuple[float, float]] = {}
            for name, (x, y) in keypoints.items():
                prev = self._points.get(name)
                if prev is None:
                    blended[name] = (x, y)
                else:
                    blended[name] = (a * x + (1 - a) * prev[0], a * y + (1 - a) * prev[1])
            self._points = blended

        if confidence is None:
            self._confidence = None
        elif self._confidence is None or self.alpha >= 1.0:
            self._confidence = confidence
        else:
            a = self.alpha
            self._confidence = a * confidence + (1 - a) * self._confidence
        return dict(self._points), self._confidence

    def reset(self) -> None:
        self._points.clear()
        self._confidence = None


# ── helpers ───────────────────────────────────────────────────────────────────
def _drop_low_confidence(
    points: list[Keypoint], minimum: float
) -> list[Keypoint]:
    """
    Remove keypoints below the floor, but only when the response carried
    confidences at all. A response without confidences is not "zero confidence";
    it is silent, and rejecting every silent response would disable the feature
    for models that simply do not report per-point scores.
    """
    scored = [p for p in points if p.confidence is not None]
    if not scored:
        return list(points)
    return [p for p in points if p.confidence is None or p.confidence >= minimum]


def _drop_out_of_bounds(points: list[Keypoint], image_size: Size) -> list[Keypoint]:
    width, height = image_size
    slack_x = _BOUND_SLACK_FRACTION * width
    slack_y = _BOUND_SLACK_FRACTION * height
    return [
        p
        for p in points
        if -slack_x <= p.x <= width + slack_x and -slack_y <= p.y <= height + slack_y
    ]


def _hull_area_fraction(coords: list[tuple[float, float]], image_size: Size) -> float:
    """
    Share of the frame the keypoint hull covers.

    A SHARE, not an area: `cfg.min_area_fraction` is a fraction of the frame,
    so a hull measured in square pixels would pass any threshold that small and
    a cluster of keypoints the size of a postage stamp would be read as a pitch.
    """
    if len(coords) < 3:
        return 0.0
    width, height = image_size
    if width <= 0 or height <= 0:
        return 0.0
    pts = np.array(coords, dtype=np.float32)
    return float(cv2.contourArea(cv2.convexHull(pts))) / float(width * height)


def _span(corners: dict[str, tuple[float, float]], *, horizontal: bool) -> float:
    if not corners:
        return 0.0
    if horizontal:
        top = ("top_left", "top_right")
        bottom = ("bottom_left", "bottom_right")
    else:
        top = ("top_left", "bottom_left")
        bottom = ("top_right", "bottom_right")
    try:
        top_len = _distance(corners[top[0]], corners[top[1]])
        bottom_len = _distance(corners[bottom[0]], corners[bottom[1]])
    except KeyError:
        return 0.0
    return (top_len + bottom_len) / 2.0


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


__all__ = [
    "PitchCalibration",
    "PitchSmoother",
    "build_calibration",
    "build_homography",
    "derive_corners",
]
