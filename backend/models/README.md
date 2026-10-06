# Models

Two checkpoints. Neither is in git (see `.gitignore`); both are listed here with
the checksum that must match, because a silently wrong checkpoint produces
confidently wrong shot labels rather than an error.

| File | Size | Purpose | MD5 |
|---|---|---|---|
| `ball_detector.pt` | 6,225,450 B | YOLO11 ball detection, run once per uploaded video | `ADDE61B5409FCE5E…` |
| `shot_classifier.ckpt` | 72,010,842 B | Adarsh `ImprovedSOTAModel` shot classification, per delivery | `2974EFDCF449C8447BEF107D94633217` |

## ball_detector.pt — detector

- **Task:** object detection, single class (`ball`), loaded via
  `ultralytics.YOLO`.
- **Input:** BGR `np.ndarray` frames straight from the decode — no resizing, no
  normalisation, no colour conversion before the model. See
  `app/tracking/detector.py`.
- **Output:** boxes plus confidence. `detect_with_confidence()` returns both the
  detection and the confidence that segmentation thresholds against.
- **Origin:** CricketTracker `backend/tracker_modules/detector.py`, ported
  verbatim. Loads with `task=detect`.
- **Failure mode to watch:** a checkpoint loaded with the wrong `task` loads
  without error and returns tensors the tracker cannot read.

## shot_classifier.ckpt — shot classifier

- **Architecture:** `ImprovedSOTAModel` = EfficientNet-B0 per frame + temporal
  transformer, defined in `app/shot_classification/model.py`.
- **Input:** `(B, 30, 3, 224, 224)` float tensor. Exactly 30 frames, always. A
  shorter delivery clip has its sampled positions **repeated**; the model has a
  fixed temporal length and no shorter-input path.
- **Output:** logits `(B, 10)` → softmax → `{type, confidence, class_probabilities}`.
  The class vocabulary lives in `SHOT_CLASSES` (`app/core/constants.py`) and is
  published at `/api/reference`.
- **Preprocessing:** `app/shot_classification/preprocessing.py`. The
  `IMAGE_TRANSFORM` order and the 30-frame sampling are reproduced from the
  upstream training pipeline. **This is not an improvement surface** — changing it
  invalidates the checkpoint's accuracy while every test still passes.
- **Loading:** `strict=True` (`app/shot_classification/inference.py`), so an
  architecture/checkpoint mismatch fails loudly instead of leaving layers randomly
  initialised. The `class_weights` buffer is absent from the checkpoint and is
  restored from a fresh model, as upstream does.
- **Filename aliases:** the upstream project calls it
  `cricket_model_transformer.ckpt`. `resolve_shot_model_path()` accepts either
  name, preferring the upstream one, so a checkout that kept either runs
  unchanged.

## Verifying a copy

```bash
python - <<'PY'
import hashlib, pathlib
for name, expect in (
    ("ball_detector.pt",   "adde61b5409fce5e"),
    ("shot_classifier.ckpt", "2974efdcf449c8447bef107d94633217"),
):
    p = pathlib.Path("backend/models") / name
    h = hashlib.md5(p.read_bytes()).hexdigest()
    print(name, h, "OK" if h.startswith(expect) else "MISMATCH")
PY
```

## Pointing elsewhere

Both paths are overridable without touching code:

```bash
export CRICKET_MODELS_DIR=/mnt/models          # the directory
export CRICKET_BALL_MODEL=/mnt/models/det.pt   # or one file each
export CRICKET_SHOT_MODEL=/mnt/models/shot.ckpt
```

A missing checkpoint raises `FileNotFoundError` naming every path tried, at
startup, rather than surfacing later as "shot model failed to load".