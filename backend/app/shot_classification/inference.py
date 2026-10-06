"""
inference.py — Loading the shot classifier and labelling one delivery.

WHAT CHANGED FROM UPSTREAM, AND WHY
-----------------------------------
Upstream `delivery_analyzer.load_30_frames(clip_path)` samples 30 frames with
`np.linspace` and then issues **30 separate `cap.set(CAP_PROP_POS_FRAMES)`
seeks**. Each seek lands on the nearest preceding keyframe and decodes forward,
so 30 seeks re-decode most of the clip and cost far more than a linear read.

Three cases existed upstream, and this module keeps the one that is safe:

  * Re-open the SOURCE video and seek      -> FORBIDDEN. The source is decoded
                                              once, in TrackingService. Decision
                                              #4 and C1.
  * Re-open the CLIP and seek              -> permitted (decision #5) but
                                             wasteful, and the frames are
                                             usually still in memory.
  * Sample from frames already in memory   -> preferred, and the default.

`shot.input_source` records which path actually ran, so a consumer can always
tell whether the label came from memory or from a clip re-read.

THE MODEL AND ITS PREPROCESSING ARE UNCHANGED
----------------------------------------------
`IMAGE_TRANSFORM`, `ImprovedSOTAModel`, the 30-frame sample count and the
`strict=True` checkpoint load are reproduced faithfully. Preprocessing itself
lives in `preprocessing.py`; this module owns loading, the forward pass, and the
decision to fail one delivery rather than the whole analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F

from ..core.constants import SHOT_CLASSES, SHOT_IMAGE_SIZE, SHOT_N_FRAMES
from ..schemas.tracking import Shot
from .preprocessing import (
    DEVICE,
    FrameSource,
    ShotModelError,
    frames_to_tensor,
    sample_indices,
    tensor_from_clip_file,
)

log = logging.getLogger(__name__)


# ── Model ─────────────────────────────────────────────────────────────────────
def _extract_state_dict(checkpoint: Any) -> dict:
    if not isinstance(checkpoint, dict):
        raise ShotModelError("Unsupported checkpoint format; expected a dict")
    for key in ("state_dict", "model_state_dict"):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    if not isinstance(checkpoint, dict):
        raise ShotModelError("Checkpoint state_dict is not a dictionary")

    # Training wrappers prepend `module.` (DataParallel) or `model.` (Lightning).
    # Upstream strips them before loading and so does this: a checkpoint saved on
    # a multi-GPU box would otherwise fail `strict=True` for a reason that has
    # nothing to do with the architecture.
    cleaned: dict = {}
    for key, value in checkpoint.items():
        new_key = key
        while new_key.startswith("module.") or new_key.startswith("model."):
            new_key = new_key.split(".", 1)[1]
        # The deployment architecture creates this buffer itself, and the value
        # in the checkpoint is a class-weighted training artefact. `load_shot_model`
        # restores it from a fresh model, exactly as upstream does.
        if new_key == "class_weights":
            continue
        cleaned[new_key] = value
    return cleaned


def load_shot_model(checkpoint_path: Path | str, model_cls) -> Any:
    """
    Load the EfficientNet + transformer shot classifier.

    `strict=True` is deliberate and is kept from upstream: it turns an
    architecture/checkpoint mismatch into a loud failure at load time instead of
    silently random initialised layers producing confidently wrong shot labels.

    The `class_weights` buffer is absent from the checkpoint but is an
    `nn.Parameter` in the architecture, so it is restored from a freshly
    constructed model before loading. Same workaround as upstream.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise ShotModelError(f"Shot model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)

    model = model_cls(
        num_classes=len(SHOT_CLASSES),
        temporal_dim=256,
        n_frames=SHOT_N_FRAMES,
    )

    state_dict["class_weights"] = (
        model.state_dict()["class_weights"].detach().clone()
    )

    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE)
    model.eval()
    log.info("Shot model loaded on %s", DEVICE)
    return model

# ── Prediction ───────────────────────────────────────────────────────────────
@torch.inference_mode()
def predict_shot(model, frames: torch.Tensor) -> dict:
    """Run the classifier. Returns the winning class, confidence and all probs."""
    expected = (1, SHOT_N_FRAMES, 3, SHOT_IMAGE_SIZE, SHOT_IMAGE_SIZE)
    if tuple(frames.shape) != expected:
        raise ValueError(f"Expected tensor {expected}, got {tuple(frames.shape)}")

    frames = frames.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
    logits = model(frames)

    if logits.ndim != 2 or logits.shape[0] != 1:
        raise RuntimeError(f"Unexpected model output shape {tuple(logits.shape)}")
    if logits.shape[1] != len(SHOT_CLASSES):
        raise RuntimeError(
            f"Expected {len(SHOT_CLASSES)} classes, got {logits.shape[1]}"
        )

    probabilities = F.softmax(logits, dim=1)
    confidence, class_index = torch.max(probabilities, dim=1)
    predicted_index = int(class_index.item())

    return {
        "type": SHOT_CLASSES[predicted_index],
        "confidence": float(confidence.item()),
        "class_probabilities": {
            name: round(float(probabilities[0, i].item()), 6)
            for i, name in enumerate(SHOT_CLASSES)
        },
    }


# ── Stage entry point ────────────────────────────────────────────────────────
class ShotClassifier:
    """
    Classifies one delivery's shot.

    Tries, in order:
      1. Frames already retained in memory by the tracking pass.
      2. The written clip file, decoded sequentially (decision #5 permits this;
         the source video is never touched).

    A failure in either path is recorded on the Shot record rather than raised,
    because one unclassifiable delivery must not fail an otherwise valid
    analysis.
    """

    def __init__(self, model) -> None:
        self.model = model

    def classify(
        self,
        clip_path: Optional[Path],
        in_memory_frames: Optional[FrameSource] = None,
        expected_total_frames: Optional[int] = None,
        delivery_id: Optional[int] = None,
        frame_offset: Optional[int] = None,
    ) -> Shot:
        """
        Parameters
        ----------
        clip_path
            The written clip. Used only as a fallback source of pixels.
        in_memory_frames
            Frames the tracking pass retained. Preferred path.

            A **mapping** keyed by 0-based position within the clip is the
            natural form and is what the clip writer produces — the retained
            frames are a sparse subset of a ~150-frame clip, so their positions
            carry the sampling information. A dense **sequence** is also
            accepted, in which case index i means clip frame i.
        expected_total_frames
            Clip frame count used to compute sample indices. Required when
            relying on in-memory frames, since they are keyed by position.
        delivery_id
            Only used to make the mapping log line attributable.
        frame_offset
            `clip_start_frame - 1` for this clip, i.e. the same
            `ClipFrameMap.frame_offset` the rest of the system uses. When given,
            the sampled clip positions are also logged as ORIGINAL source frame
            numbers, which is the mapping a bug in the clip writer would corrupt.

        Never raises: a failure is recorded on the Shot so one unclassifiable
        delivery cannot fail an otherwise valid analysis.
        """
        frames, input_source, sampled_positions = self._gather(
            clip_path, in_memory_frames, expected_total_frames
        )

        if frames is None or len(frames) != SHOT_N_FRAMES:
            n = 0 if frames is None else len(frames)
            reason = (
                "clip unreadable" if frames is None
                else f"got {n} of {SHOT_N_FRAMES} frames"
            )
            log.warning(
                "Shot inference skipped for delivery %s: %s",
                delivery_id, reason,
            )
            return Shot(
                input_source=input_source,
                frames_sampled=n,
                error=reason,
            )

        unique_positions = len(set(sampled_positions))
        repeated = unique_positions < len(sampled_positions)

        # The mapping audit. Every value here is measured, not inferred, so it is
        # printed at DEBUG rather than guessed at from the clip duration.
        log.debug(
            "shot input: delivery=%s source=%s clip_positions=%s "
            "original_frames=%s unique=%d repeated=%s",
            delivery_id,
            input_source,
            sampled_positions,
            (
                [i + frame_offset for i in sampled_positions]
                if frame_offset is not None else None
            ),
            unique_positions,
            repeated,
        )
        if repeated:
            log.info(
                "Delivery %s clip has only %d frame(s) at or below the %d-frame "
                "model input, so sampled positions repeat; the model receives "
                "exactly %d frames",
                delivery_id, unique_positions, SHOT_N_FRAMES, SHOT_N_FRAMES,
            )

        tensor = frames_to_tensor(frames)
        log.debug(
            "shot tensor: delivery=%s shape=%s device=%s",
            delivery_id, tuple(tensor.shape), DEVICE,
        )

        try:
            result = predict_shot(self.model, tensor)
        except Exception as exc:
            log.error("Shot inference failed for delivery %s: %s", delivery_id, exc)
            return Shot(
                input_source=input_source,
                frames_sampled=len(frames),
                unique_frames_sampled=unique_positions,
                frames_repeated=repeated,
                error=str(exc),
            )

        return Shot(
            type=result["type"],
            confidence=round(result["confidence"], 6),
            class_probabilities=result["class_probabilities"],
            input_source=input_source,
            frames_sampled=len(frames),
            unique_frames_sampled=unique_positions,
            frames_repeated=repeated,
            sampled_clip_frames=sampled_positions,
            sampled_original_frames=(
                [i + frame_offset for i in sampled_positions]
                if frame_offset is not None else None
            ),
        )

    @staticmethod
    def _gather(
        clip_path: Optional[Path],
        in_memory_frames: Optional[FrameSource],
        expected_total_frames: Optional[int],
    ) -> tuple[Optional[list], str, list[int]]:
        """
        Return (frames, input_source, sampled_clip_positions).

        `frames` is None when unavailable. The positions are returned alongside
        the frames so the caller can audit the mapping without recomputing the
        sampling, and so both paths are provably identical.
        """
        # ── Path 1: in-memory ────────────────────────────────────────────────
        # The retained frames must be looked up BY POSITION. Collapsing them
        # into a dense list first would renumber them and silently change
        # which moments of the clip the model sees — a label that depends on
        # bookkeeping rather than on pixels.
        if in_memory_frames and expected_total_frames:
            idxs = sample_indices(int(expected_total_frames))
            if idxs:
                if isinstance(in_memory_frames, Mapping):
                    missing = [i for i in idxs if i not in in_memory_frames]
                    if not missing:
                        # Repeated positions resolve to the same retained frame,
                        # which is exactly the behaviour tensor_from_clip_file
                        # now implements too.
                        return (
                            [in_memory_frames[i] for i in idxs],
                            "in_memory",
                            idxs,
                        )
                    reason = (
                        f"{len(missing)} of {SHOT_N_FRAMES} sampled positions "
                        f"absent (e.g. {missing[0]})"
                    )
                else:
                    seq = in_memory_frames
                    if max(idxs) < len(seq):
                        picked = [seq[i] for i in idxs]
                        if len(picked) == SHOT_N_FRAMES:
                            return picked, "in_memory", idxs
                        reason = (
                            f"sampling yielded {len(picked)} of "
                            f"{SHOT_N_FRAMES} frames"
                        )
                    else:
                        reason = (
                            f"{len(seq)} frames retained but clip position "
                            f"{max(idxs)} was required"
                        )
                log.warning(
                    "In-memory shot frames unusable (%s); re-reading the clip "
                    "file (permitted — the source video is not reopened)",
                    reason,
                )

        # ── Path 2: the written clip, one sequential decode ───────────────────
        if clip_path is not None and Path(clip_path).is_file():
            try:
                frames, positions = tensor_from_clip_file(clip_path)
                if len(frames) == SHOT_N_FRAMES:
                    return frames, "clip_file", positions
                log.warning(
                    "Clip file yielded %d of %d frames",
                    len(frames), SHOT_N_FRAMES,
                )
            except Exception as exc:
                log.error("Clip-file sampling failed: %s", exc)

        return None, "unavailable", []

__all__ = [
    "DEVICE",
    "ShotClassifier",
    "ShotModelError",
    "frames_to_tensor",
    "load_shot_model",
    "predict_shot",
    "sample_indices",
    "tensor_from_clip_file",
]
