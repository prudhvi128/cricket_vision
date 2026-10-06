"""
preprocessing.py — Turning a delivery clip into the exact tensor the shot model expects.

THIS IS NOT AN IMPROVEMENT SURFACE
----------------------------------
Everything here reproduces the upstream Adarsh pipeline bit-for-bit: the
`IMAGE_TRANSFORM` order, the 30-frame sample count, `np.linspace` sampling, and
the requirement that a short clip be handled by REPEATING indices. A different
preprocessing pipeline would silently invalidate `shot_classifier.ckpt`'s
accuracy while every test still passed, so these functions are moved as-is and
must not be "cleaned up".

WHY REPETITION IS REQUIRED, NOT INCREMENTAL
-------------------------------------------
`ImprovedSOTAModel` is a fixed-temporal-length transformer: `PositionalEncoding`
is a `(1, n_frames, d_model)` buffer and `forward()` raises on any
`T != n_frames`. A delivery clip shorter than 30 frames therefore has only two
honest options — repeat frames, or report no shot at all. Upstream repeats, and a
shot label is worth more than an exact frame count. It is never silent: the
`Shot` record carries `frames_repeated` and `unique_frames_sampled`, and
`inference.ShotClassifier` logs the sampled positions.

`sample_indices` is exposed on its own so the in-memory path and the clip-read
path provably agree. If they diverged, the same clip would classify differently
depending on which path ran — a bug that would be very hard to see.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Union

import cv2
import numpy as np
import torch
from torchvision import transforms

from ..core.constants import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SHOT_IMAGE_SIZE,
    SHOT_N_FRAMES,
)


# Identical to upstream's IMAGE_TRANSFORM, in the same order.
IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((SHOT_IMAGE_SIZE, SHOT_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A retained frame source is either a mapping of 0-based clip position to frame
# (what ClipWriterRegistry produces — a sparse subset of the clip) or a dense
# sequence indexed by clip position.
FrameSource = Union[Mapping[int, np.ndarray], Sequence[np.ndarray]]


class ShotModelError(Exception):
    pass


# ── Frame sampling ───────────────────────────────────────────────────────────
def sample_indices(total_frames: int, n: int = SHOT_N_FRAMES) -> list[int]:
    """
    Which frame indices to sample from a clip of *total_frames* frames.

    ALWAYS returns exactly `n` indices when the clip has at least one frame.

    This is `np.linspace(0, total_frames - 1, n)`, which is what the upstream
    `delivery_analyzer.load_30_frames` computes, and matching it exactly matters
    for two reasons:

      1. The architecture is a fixed-temporal-length transformer: `PositionalEncoding`
         is a `(1, n_frames, d_model)` buffer and `forward()` raises on any
         `T != n_frames`. There is no shorter-input path. A delivery clip shorter
         than 30 frames therefore has only two honest choices — repeat frames, or
         report no shot at all — and upstream repeats.
      2. A clip at or below `n` frames gets REPEATED INDICES rather than a short
         list. An earlier version of this module returned `range(total_frames)` in
         that case, which handed the model fewer than 30 frames and made
         `classify()` give up with "got 19 of 30 frames". On the real video that
         silently cost a shot label on every short delivery.

    Repetition is REQUIRED here, not incidental: the model cannot consume any
    other temporal length. It is never silent, though — `classify()` records
    `frames_repeated` and `unique_frames_sampled` on the Shot record and logs the
    sampled positions, so a label produced from a repeated sample is
    distinguishable from one produced from 30 distinct frames.

    A clip with zero frames still returns [], because there is genuinely nothing
    to sample and no amount of repeating invents a frame that does not exist.

    Exposed separately so the in-memory path and the clip-read path provably
    agree. If these ever diverge, the same clip would classify differently
    depending on which path ran — a bug that would be very hard to see.
    """
    if total_frames <= 0:
        return []
    if total_frames == 1:
        return [0] * n
    return [int(i) for i in np.linspace(0, total_frames - 1, n, dtype=np.int64)]


def frames_to_tensor(frames: Sequence[np.ndarray]) -> torch.Tensor:
    """
    BGR ndarray frames -> (1, n, 3, 224, 224) float tensor.

    Raises if the frame count differs from SHOT_N_FRAMES: the model expects a
    fixed temporal length, and padding or truncating silently would change what
    the network sees without anything recording it.
    """
    if len(frames) != SHOT_N_FRAMES:
        raise ValueError(
            f"Expected exactly {SHOT_N_FRAMES} frames, got {len(frames)}"
        )
    tensors = [
        IMAGE_TRANSFORM(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
    ]
    return torch.stack(tensors, dim=0).unsqueeze(0)


def tensor_from_clip_file(clip_path: Path | str, n: int = SHOT_N_FRAMES):
    """
    Sample frames from an already-written clip with ONE sequential decode.

    Deliberately a linear read rather than 30 seeks: for a short delivery clip a
    sequential pass is both simpler and faster, and it cannot land on the wrong
    frame the way `CAP_PROP_POS_FRAMES` can.

    `wanted` may contain REPEATED indices (a clip shorter than `n` frames), so the
    decode collects each distinct position once and the returned list is then
    rebuilt from `wanted` in order. Deduplicating the output instead — as an
    earlier version did — would return fewer than `n` frames for a short clip and
    silently reintroduce the defect this whole path exists to avoid.

    Returns (frames, indices) where indices are 0-based positions within the clip.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ShotModelError(f"Could not open clip: {clip_path}")
    try:
        wanted = sample_indices(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), n)
        if not wanted:
            return [], []
        wanted_set = set(wanted)
        collected: dict[int, np.ndarray] = {}
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted_set:
                collected[idx] = frame
            idx += 1
            if len(collected) == len(wanted_set):
                break
        missing = sorted(i for i in wanted_set if i not in collected)
        if missing:
            raise ShotModelError(
                f"clip ended before {len(missing)} sampled frame(s); first "
                f"missing clip position {missing[0]}"
            )
        # Rebuilt from `wanted`, not from `collected`, so repeated positions
        # produce repeated frames and the model always receives exactly n frames.
        return [collected[i] for i in wanted], wanted
    finally:
        cap.release()

__all__ = [
    "DEVICE",
    "IMAGE_TRANSFORM",
    "FrameSource",
    "ShotModelError",
    "frames_to_tensor",
    "sample_indices",
    "tensor_from_clip_file",
]
