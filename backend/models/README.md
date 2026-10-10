# Models

Three checkpoints, one of them optional. All three are tracked in git (they
have been since `.gitignore` stopped excluding `*.pt`/`*.ckpt`), and all are
listed here with the checksum that must match, because a silently wrong
checkpoint produces confidently wrong shot labels rather than an error.

| File | Size | Purpose | MD5 |
|---|---|---|---|
| `ball_detector.pt` | 6,225,450 B | YOLO11 ball detection, run once per uploaded video | `ADDE61B5409FCE5E…` |
| `shot_classifier.ckpt` | 72,010,842 B | Adarsh `ImprovedSOTAModel` shot classification, per delivery | `2974EFDCF449C8447BEF107D94633217` |
| `pitch_detect.pt` | 22,517,987 B | **Optional** local pitch-box detection, off unless configured | `fe980582cb5446062db5e65c238a8afb` |

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

## pitch_detect.pt — optional local pitch-box detector

- **Task:** object detection, single class (`Cricket_pitch`), loaded via
  `ultralytics.YOLO`. See `app/pitch/local_box_detector.py`.
- **Optional and off by default.** It is loaded only when
  `CRICKET_PITCH_BOX_MODEL` names it (a bare filename resolves against this
  directory); with the variable unset, nothing in the pipeline touches it and
  the output is byte-for-byte what it was before the detector existed.
- **Input:** BGR frame at its ORIGINAL resolution. Ultralytics letterboxes to
  `imgsz` (640) itself — the frame is never pre-resized, which is the mistake
  that previously squashed frames for the ball detector.
- **Output:** one box, highest confidence, in original frame pixels, converted
  to four APPROXIMATE corners by `app/pitch/box_corners.py`. Those corners are
  guesses: they are stored under `pitch.box_estimate` only, never become
  `pitch.corners`, never feed a homography, and can never make an analysis
  `geometry_calibrated`.
- **Measured on `real_cricket.mp4` (CUDA):** first inference ~3.7 s warm-up,
  then ~10–50 ms per frame, with the model returning no box on ~78% of sampled
  frames (replays and close-ups have no full pitch in view).

## Verifying a copy

```bash
python - <<'PY'
import hashlib, pathlib
for name, expect in (
    ("ball_detector.pt",   "adde61b5409fce5e"),
    ("shot_classifier.ckpt", "2974efdcf449c8447bef107d94633217"),
    ("pitch_detect.pt",     "fe980582cb5446062db5e65c238a8afb"),
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
export CRICKET_PITCH_BOX_MODEL=/mnt/models/pitch_detect.pt   # optional detector
```

A missing checkpoint raises `FileNotFoundError` naming every path tried, at
startup, rather than surfacing later as "shot model failed to load".

The optional pitch-box weights are the exception: `CRICKET_PITCH_BOX_MODEL`
naming a file that is not there is logged as an error at analysis start and the
run continues **without** the feature, exactly as if it had never been
configured, rather than failing an analysis over an auxiliary guess.