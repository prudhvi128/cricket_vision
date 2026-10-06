"""
shot_classification — The Adarsh shot model, unmodified.

This is the one part of the system inherited wholesale from a source project, and
it is kept intact on purpose. The architecture (`model.ImprovedSOTAModel`:
EfficientNet-B0 + transformer), the preprocessing (`preprocessing.IMAGE_TRANSFORM`,
30-frame sampling with repetition for short clips), the checkpoint
(`backend/models/shot_classifier.ckpt`) and the 10-class vocabulary are a single
artifact. Changing any one of them changes the meaning of the model's output while
leaving every test green.

  preprocessing.py  frames -> (1, 30, 3, 224, 224). Not an improvement surface.
  inference.py      checkpoint loading, the forward pass, and `ShotClassifier`,
                    which prefers frames already retained in memory and only
                    falls back to re-reading the written clip.
  model.py          the architecture definition, required to construct the model
                    at load time.

A failed classification is recorded on the `Shot` record, never raised: one
unclassifiable delivery must not fail an otherwise valid analysis.
"""

from .inference import (
    ShotClassifier,
    ShotModelError,
    frames_to_tensor,
    load_shot_model,
    predict_shot,
    sample_indices,
    tensor_from_clip_file,
)

__all__ = [
    "ShotClassifier",
    "ShotModelError",
    "frames_to_tensor",
    "load_shot_model",
    "predict_shot",
    "sample_indices",
    "tensor_from_clip_file",
]