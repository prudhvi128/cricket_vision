"""
config.py — Runtime configuration and filesystem layout.

Every path is derived from one root so the two original repositories are never
written to. All tunables can be overridden per-call; nothing here reads or
mutates the source repositories.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .constants import (
    ACTIVITY_MIN_WINDOW_SECONDS,
    ACTIVITY_QUIET_HOLD_SECONDS,
    BLACK_FRAME_LUMINANCE_THRESHOLD,
    BLACK_FRAME_RUN_BARRIER,
    FRAGMENT_MERGE_CORRIDOR_RATIO,
    FRAGMENT_MERGE_MAX_GAP_SECONDS,
    FRAGMENT_MERGE_MIN_COMBINED_FRAMES,
    FRAGMENT_MERGE_REACH_BASE_FRACTION,
    FRAGMENT_MERGE_REACH_RATE_FRACTION,
    MAX_POST_ROLL_SECONDS,
    MAX_PRE_ROLL_SECONDS,
    MIN_DELIVERY_FRAMES,
    MIN_FRAMES_BETWEEN_DELIVERIES,
    PITCH_ATTEMPTS_PER_SEC_FLOOR,
    PITCH_DETECTION_INTERVAL_FRAMES,
    PITCH_HOMOGRAPHY_MIN_KPS,
    PITCH_MAX_ATTEMPTS_PER_SEC,
    PITCH_MAX_INPUT_SIZE,
    PITCH_MAX_LOST_FRAMES,
    PITCH_MIN_AREA_FRACTION,
    PITCH_MIN_CONFIDENCE,
    PITCH_MIN_KEYPOINTS,
    PITCH_MODEL_ID_DEFAULT,
    PITCH_REQUEST_TIMEOUT_SECONDS,
    PITCH_SAMPLING_MODE_DEFAULT,
    PITCH_SAMPLING_MODES,
    PITCH_SMOOTHING_ALPHA,
    POST_ROLL_SECONDS,
    PRE_ROLL_SECONDS,
    RING_BUFFER_CAPACITY_FRAMES,
    RING_BUFFER_JPEG_QUALITY,
    RING_BUFFER_MAX_BYTES,
    ROBOFLOW_API_URL_DEFAULT,
    SEG_AMBIGUOUS_CONFIDENCE,
    SEG_CLIP_MARGIN_SECONDS,
    SEG_HARD_SPLIT_SECONDS,
    SEG_MAX_COAST_SECONDS,
    SEG_REACH_BASE_FRACTION,
    SEG_REACH_RATE_FRACTION,
    SEG_REVERSAL_DEGREES,
    SEG_SPEED_JUMP_DOWN,
    SEG_SPEED_JUMP_UP,
    SEG_VELOCITY_WINDOW,
    SHOT_SAMPLING_MAX_IN_MEMORY_BYTES,
    VIDEO_CODEC_PREFERENCE,
    WEAK_CONFIRMATION_CONFIDENCE,
    WEAK_CORRIDOR_RATIO,
    WEAK_MAX_SILENCE_FRACTION,
    WEAK_MIN_CANDIDATE_FRAMES,
    WEAK_MIN_CONFIRMED_FRAMES,
    WEAK_MIN_CONTIGUOUS_FRACTION,
    WEAK_MIN_SIGNAL_AGREEMENT,
    WEAK_MIN_TOTAL_MEASUREMENTS,
    WEAK_REACH_TOLERANCE,
    WEAK_SPEED_RATIO_DOWN,
    WEAK_SPEED_RATIO_UP,
    WEAK_SPREAD_FRACTION,
    SHOT_MODEL_FILENAMES,
    SHOT_MODEL_PROVENANCE_NAME,
    BLACK_FRAME_LUMINANCE_THRESHOLD,
    BLACK_FRAME_RUN_BARRIER,
    SCENE_CUT_DIFF_THRESHOLD,
    MAX_PRE_ROLL_SECONDS,
    MAX_POST_ROLL_SECONDS,
    PITCH_ACTIVITY_QUIET_THRESHOLD,
    PITCH_ACTIVITY_ACTIVE_THRESHOLD,
    ACTIVITY_MIN_WINDOW_SECONDS,
    ACTIVITY_QUIET_HOLD_SECONDS,
)

# ── Filesystem layout ─────────────────────────────────────────────────────────
# backend/app/core/config.py -> backend/app/core -> backend/app -> backend
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


def _load_dotenv() -> None:
    """
    Read `<project>/.env` if it exists, never overriding the real environment.

    The API key for the optional pitch detector is a credential, so it lives in
    an untracked `.env` (see .gitignore / .env.example) rather than in code or
    in a command line. `override=False` means a shell-exported variable always
    wins, which is what lets CI and the tests set these values directly.

    Import failures are swallowed: python-dotenv is a convenience, and a
    missing package must not stop the server from starting.
    """
    env_file = PROJECT_DIR / ".env"
    if not env_file.is_file():
        return
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    try:
        load_dotenv(env_file, override=False)
    except Exception:
        return


_load_dotenv()


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw.strip() if raw is not None and raw.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_path(name: str, default: Path) -> Path:
    """
    Resolve a directory from the environment, falling back to the repo default.

    Only these roots are overridable, and only in one direction: a path may be
    pointed elsewhere, but a relative path is always resolved against the project
    directory so a service started from a different working directory still finds
    the same data. This is what lets a container mount `/data` without the code
    knowing it is in a container.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (PROJECT_DIR / p).resolve()


DATA_DIR = _env_path("CRICKET_DATA_DIR", PROJECT_DIR / "data")
MODELS_DIR = _env_path("CRICKET_MODELS_DIR", BACKEND_DIR / "models")
UPLOADS_DIR = _env_path("CRICKET_UPLOADS_DIR", DATA_DIR / "uploads")
RUNS_DIR = _env_path("CRICKET_RUNS_DIR", DATA_DIR / "runs")
SAMPLES_DIR = _env_path("CRICKET_SAMPLES_DIR", DATA_DIR / "samples")
DATASETS_DIR = _env_path("CRICKET_DATASETS_DIR", DATA_DIR / "datasets")
MANIFESTS_DIR = _env_path("CRICKET_MANIFESTS_DIR", DATA_DIR / "manifests")


def _env_file(name: str, default: Path) -> Path:
    """Same idea as `_env_path`, for an individual model file."""
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


BALL_MODEL_PATH = _env_file(
    "CRICKET_BALL_MODEL", MODELS_DIR / "ball_detector.pt"
)
# Kept for backwards compatibility with callers that imported it directly. Prefer
# `resolve_shot_model_path()`, which also finds the upstream filename.
SHOT_MODEL_PATH = _env_file(
    "CRICKET_SHOT_MODEL", MODELS_DIR / "shot_classifier.ckpt"
)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def resolve_shot_model_path(models_dir: Path | None = None) -> Path:
    """
    Locate the Adarsh shot checkpoint, accepting either known filename.

    The upstream repository calls it `cricket_model_transformer.ckpt`; this one
    keeps a byte-identical copy as `shot_classifier.ckpt` (same MD5). Both names are
    accepted so the backend runs against a checkout that kept either, and the
    upstream name is tried first because that is the documented name.

    Raises FileNotFoundError naming every path tried, rather than letting the
    pipeline report a generic "shot model failed to load" much later.
    """
    base = Path(models_dir) if models_dir is not None else MODELS_DIR
    tried = [base / name for name in SHOT_MODEL_FILENAMES]
    for candidate in tried:
        if candidate.is_file():
            return candidate
    tried.append(SHOT_MODEL_PATH)
    raise FileNotFoundError(
        "Shot model checkpoint not found. Tried:\n  "
        + "\n  ".join(str(p) for p in tried)
        + f"\nExpected the Adarsh {SHOT_MODEL_PROVENANCE_NAME} weights."
    )


@dataclass(frozen=True)
class PitchConfig:
    """
    Settings for the optional Roboflow pitch-keypoint detector.

    Environment-resolved ONCE at import (via `_load_dotenv` above), exactly like
    the rest of this module. Frozen, so a test can `dataclasses.replace` one
    field without inventing a new configuration object.

    `enabled` is deliberately just "is there an API key". Everything else —
    model id, interval, thresholds — is inert until a credential exists, so a
    checkout with no `.env` behaves as if the feature were never installed.
    """

    api_key: str = ""
    api_url: str = ROBOFLOW_API_URL_DEFAULT
    model_id: str = PITCH_MODEL_ID_DEFAULT
    # Submit one frame every N frames of the source video.
    interval_frames: int = PITCH_DETECTION_INTERVAL_FRAMES
    # How those submissions are spaced: `interval` (every Nth frame, always),
    # `adaptive` (the same N, scaled by delivery activity, calibration
    # coverage, corridor brightness and queue back-pressure) or `sequential`
    # (backpressure-aware: one inference at a time, never faster than the
    # detector answers, capped by `max_attempts_per_sec`).
    sampling_mode: str = PITCH_SAMPLING_MODE_DEFAULT
    # `sequential` mode only: the ceiling on detector attempts per second, and
    # therefore the cooldown between two submissions.
    max_attempts_per_sec: float = PITCH_MAX_ATTEMPTS_PER_SEC
    # Server-side and local confidence floor for a keypoint.
    min_confidence: float = PITCH_MIN_CONFIDENCE
    # Keypoints required to accept a detection at all.
    min_keypoints: int = PITCH_MIN_KEYPOINTS
    # Keypoints required before a homography can be computed (4 image points).
    homography_min_keypoints: int = PITCH_HOMOGRAPHY_MIN_KPS
    # Smallest share of the frame the keypoint hull may cover.
    min_area_fraction: float = PITCH_MIN_AREA_FRACTION
    # Frames after the last good detection that a calibration stays usable.
    max_lost_frames: int = PITCH_MAX_LOST_FRAMES
    # EMA weight applied to each accepted keypoint set.
    smoothing_alpha: float = PITCH_SMOOTHING_ALPHA
    # Ceiling on one Roboflow round trip.
    request_timeout_seconds: float = PITCH_REQUEST_TIMEOUT_SECONDS
    # Client-side downsize ceiling, in pixels, before upload.
    max_input_size: int = PITCH_MAX_INPUT_SIZE

    def __post_init__(self) -> None:
        # An unrecognised mode must not take the analysis down: it resolves to
        # the behaviour the field documents, and `metadata()` reports which one
        # actually ran. Spelling and case are forgiven for the same reason —
        # this value arrives from an environment file.
        resolved = str(self.sampling_mode).strip().lower()
        if resolved not in PITCH_SAMPLING_MODES:
            resolved = PITCH_SAMPLING_MODE_DEFAULT
        if resolved != self.sampling_mode:
            object.__setattr__(self, "sampling_mode", resolved)
        # The sequential cooldown divides by this: a zero (or negative) rate
        # would spin the sampler, and a rate that makes no sense is corrected
        # here rather than at the one place that uses it.
        rate = float(self.max_attempts_per_sec)
        if not rate > 0 or rate < PITCH_ATTEMPTS_PER_SEC_FLOOR:
            rate = PITCH_ATTEMPTS_PER_SEC_FLOOR
        if rate != self.max_attempts_per_sec:
            object.__setattr__(self, "max_attempts_per_sec", rate)

    @property
    def enabled(self) -> bool:
        """True only when a credential and a model id are both present."""
        return bool(self.api_key) and bool(self.model_id)

    def metadata(self) -> dict:
        """
        Reproducibility record for the analysis. The KEY IS NEVER INCLUDED —
        this dict is persisted to disk and returned by the API.
        """
        return {
            "enabled": self.enabled,
            "model_id": self.model_id,
            "api_url": self.api_url,
            "interval_frames": max(1, self.interval_frames),
            "sampling_mode": self.sampling_mode,
            "max_attempts_per_sec": self.max_attempts_per_sec,
            "min_confidence": self.min_confidence,
            "min_keypoints": self.min_keypoints,
            "homography_min_keypoints": self.homography_min_keypoints,
            "min_area_fraction": self.min_area_fraction,
            "max_lost_frames": self.max_lost_frames,
            "smoothing_alpha": self.smoothing_alpha,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_input_size": self.max_input_size,
            "api_key_configured": bool(self.api_key),
        }


PITCH_CONFIG = PitchConfig(
    api_key=_env_str("ROBOFLOW_API_KEY", ""),
    api_url=_env_str("ROBOFLOW_API_URL", ROBOFLOW_API_URL_DEFAULT).rstrip("/"),
    model_id=_env_str("ROBOFLOW_MODEL_ID", PITCH_MODEL_ID_DEFAULT),
    interval_frames=max(1, _env_int(
        "PITCH_DETECTION_INTERVAL", PITCH_DETECTION_INTERVAL_FRAMES
    )),
    sampling_mode=_env_str(
        "PITCH_SAMPLING_MODE", PITCH_SAMPLING_MODE_DEFAULT
    ).strip().lower(),
    max_attempts_per_sec=max(
        PITCH_ATTEMPTS_PER_SEC_FLOOR,
        _env_float("PITCH_MAX_ATTEMPTS_PER_SEC", PITCH_MAX_ATTEMPTS_PER_SEC),
    ),
    min_confidence=_env_float("PITCH_MIN_CONFIDENCE", PITCH_MIN_CONFIDENCE),
    min_keypoints=max(1, _env_int("PITCH_MIN_KEYPOINTS", PITCH_MIN_KEYPOINTS)),
    homography_min_keypoints=max(
        4, _env_int("PITCH_HOMOGRAPHY_MIN_KEYPOINTS", PITCH_HOMOGRAPHY_MIN_KPS)
    ),
    min_area_fraction=_env_float(
        "PITCH_MIN_AREA_FRACTION", PITCH_MIN_AREA_FRACTION
    ),
    max_lost_frames=max(1, _env_int("PITCH_MAX_LOST_FRAMES", PITCH_MAX_LOST_FRAMES)),
    smoothing_alpha=min(1.0, max(0.0, _env_float(
        "PITCH_SMOOTHING_ALPHA", PITCH_SMOOTHING_ALPHA
    ))),
    request_timeout_seconds=max(
        0.5, _env_float("PITCH_REQUEST_TIMEOUT_SECONDS", PITCH_REQUEST_TIMEOUT_SECONDS)
    ),
    max_input_size=max(64, _env_int("PITCH_MAX_INPUT_SIZE", PITCH_MAX_INPUT_SIZE)),
)


@dataclass
class SegmentationConfig:
    """
    Tunables for event-based delivery segmentation (services/event_segmenter.py).

    EVERY VALUE IS IN SECONDS OR IN FRACTIONS OF THE FRAME — never in frames and
    never a delivery count. `resolve()` converts them against the video being
    analysed, so the same configuration segments a 25 fps 640x360 clip and a
    50 fps 1080p clip on the same physical terms.

    There is deliberately no `expected_deliveries` field. The number of
    deliveries is an output. The earlier pixel analysis that counted 41
    contiguous footage regions is diagnostic input only, never a target.
    """

    # ── Detection-gap handling ───────────────────────────────────────────────
    # Longest silent gap still treated as the same ball (an occlusion behind the
    # bat, a fielder's arm, a momentary loss of focus).
    max_coast_seconds: float = SEG_MAX_COAST_SECONDS
    # Longest silent gap that can still be bridged by strong kinematics alone.
    # Beyond it, silence alone is conclusive: a new event has begun.
    hard_split_seconds: float = SEG_HARD_SPLIT_SECONDS

    # ── Kinematic tests (the "two events, no black gap" case) ────────────────
    # Slack on the reachable-radius calculation, as a fraction of the frame
    # diagonal, constant plus per-frame.
    reach_base_fraction: float = SEG_REACH_BASE_FRACTION
    reach_rate_fraction: float = SEG_REACH_RATE_FRACTION
    # Direction change beyond this many degrees reads as a different ball. A
    # bounce scatters the ball but does not reverse it, so this is generous.
    reversal_degrees: float = SEG_REVERSAL_DEGREES
    # Implied-speed ratios that count as a kinematic jump.
    speed_jump_up: float = SEG_SPEED_JUMP_UP
    speed_jump_down: float = SEG_SPEED_JUMP_DOWN
    velocity_window: int = SEG_VELOCITY_WINDOW

    # ── Event quality ────────────────────────────────────────────────────────
    min_event_confirmed_frames: int = MIN_DELIVERY_FRAMES
    ambiguous_confidence: float = SEG_AMBIGUOUS_CONFIDENCE
    clip_margin_seconds: float = SEG_CLIP_MARGIN_SECONDS

    # ── Coherent weak confirmation ───────────────────────────────────────────
    # An event that misses the confirmed-detection bar may still be confirmed if
    # gate-refused real measurements — evidence only, never detections — agree
    # with each other and with the event's own measurements across several
    # independent signals. None of these relaxes a threshold; they decide how much
    # corroboration a rescued event owes. See core/constants.py for the full
    # statement of what this channel may and may not become.
    weak_min_total_measurements: int = WEAK_MIN_TOTAL_MEASUREMENTS
    weak_min_confirmed_frames: int = WEAK_MIN_CONFIRMED_FRAMES
    weak_min_candidate_frames: int = WEAK_MIN_CANDIDATE_FRAMES
    weak_min_signal_agreement: int = WEAK_MIN_SIGNAL_AGREEMENT
    weak_max_silence_fraction: float = WEAK_MAX_SILENCE_FRACTION
    weak_min_contiguous_fraction: float = WEAK_MIN_CONTIGUOUS_FRACTION
    weak_reach_tolerance: float = WEAK_REACH_TOLERANCE
    weak_corridor_ratio: float = WEAK_CORRIDOR_RATIO
    weak_spread_fraction: float = WEAK_SPREAD_FRACTION
    weak_speed_ratio_down: float = WEAK_SPEED_RATIO_DOWN
    weak_speed_ratio_up: float = WEAK_SPEED_RATIO_UP
    weak_confirmation_confidence: float = WEAK_CONFIRMATION_CONFIDENCE

    # ── Dynamic extraction barriers ──────────────────────────────────────────
    black_frame_luminance_threshold: float = BLACK_FRAME_LUMINANCE_THRESHOLD
    black_frame_run_barrier: int = BLACK_FRAME_RUN_BARRIER
    scene_cut_diff_threshold: float = SCENE_CUT_DIFF_THRESHOLD
    max_pre_roll_seconds: float = MAX_PRE_ROLL_SECONDS
    max_post_roll_seconds: float = MAX_POST_ROLL_SECONDS
    pitch_activity_quiet_threshold: float = PITCH_ACTIVITY_QUIET_THRESHOLD
    pitch_activity_active_threshold: float = PITCH_ACTIVITY_ACTIVE_THRESHOLD
    activity_min_window_seconds: float = ACTIVITY_MIN_WINDOW_SECONDS
    activity_quiet_hold_seconds: float = ACTIVITY_QUIET_HOLD_SECONDS

    # ── Fragment merging ─────────────────────────────────────────────────────
    # Longest silence two adjacent candidates may span and still be judged one
    # ball. Bounded, and strictly inside `hard_split_seconds`.
    fragment_merge_max_gap_seconds: float = FRAGMENT_MERGE_MAX_GAP_SECONDS
    fragment_merge_min_combined_frames: int = FRAGMENT_MERGE_MIN_COMBINED_FRAMES
    fragment_merge_corridor_ratio: float = FRAGMENT_MERGE_CORRIDOR_RATIO
    fragment_merge_reach_base_fraction: float = FRAGMENT_MERGE_REACH_BASE_FRACTION
    fragment_merge_reach_rate_fraction: float = FRAGMENT_MERGE_REACH_RATE_FRACTION

    def resolve(self, fps: float, width: int, height: int) -> "ResolvedSegmentation":
        """Convert seconds and fractions into this video's frames and pixels."""
        fps = float(fps) if fps and fps > 0 else 25.0
        diagonal = float((int(width) ** 2 + int(height) ** 2) ** 0.5) or 1.0
        max_coast = max(1, int(round(self.max_coast_seconds * fps)))
        hard_split = max(max_coast + 1, int(round(self.hard_split_seconds * fps)))
        max_pre_roll = max(1, int(round(self.max_pre_roll_seconds * fps)))
        max_post_roll = max(1, int(round(self.max_post_roll_seconds * fps)))
        return ResolvedSegmentation(
            fps=fps,
            width=int(width),
            height=int(height),
            frame_diagonal=diagonal,
            max_coast_frames=max_coast,
            hard_split_frames=hard_split,
            reach_base_px=self.reach_base_fraction * diagonal,
            reach_rate_px=self.reach_rate_fraction * diagonal,
            reversal_degrees=self.reversal_degrees,
            speed_jump_up=self.speed_jump_up,
            speed_jump_down=self.speed_jump_down,
            velocity_window=self.velocity_window,
            min_event_confirmed_frames=self.min_event_confirmed_frames,
            ambiguous_confidence=self.ambiguous_confidence,
            clip_margin_frames=max(1, int(round(self.clip_margin_seconds * fps))),
            black_frame_luminance_threshold=self.black_frame_luminance_threshold,
            black_frame_run_barrier=self.black_frame_run_barrier,
            scene_cut_diff_threshold=self.scene_cut_diff_threshold,
            max_pre_roll_frames=max_pre_roll,
            max_post_roll_frames=max_post_roll,
            pitch_activity_quiet_threshold=self.pitch_activity_quiet_threshold,
            pitch_activity_active_threshold=self.pitch_activity_active_threshold,
            activity_min_window_frames=max(
                1, int(round(self.activity_min_window_seconds * fps))
            ),
            activity_quiet_hold_frames=max(
                1, int(round(self.activity_quiet_hold_seconds * fps))
            ),
            fragment_merge_max_gap_frames=max(
                1, int(round(self.fragment_merge_max_gap_seconds * fps))
            ),
            fragment_merge_min_combined_frames=self.fragment_merge_min_combined_frames,
            fragment_merge_corridor_ratio=self.fragment_merge_corridor_ratio,
            fragment_merge_reach_base_px=(
                self.fragment_merge_reach_base_fraction * diagonal
            ),
            fragment_merge_reach_rate_px=(
                self.fragment_merge_reach_rate_fraction * diagonal
            ),
            weak_min_total_measurements=self.weak_min_total_measurements,
            weak_min_confirmed_frames=self.weak_min_confirmed_frames,
            weak_min_candidate_frames=self.weak_min_candidate_frames,
            weak_min_signal_agreement=self.weak_min_signal_agreement,
            weak_max_silence_fraction=self.weak_max_silence_fraction,
            weak_min_contiguous_fraction=self.weak_min_contiguous_fraction,
            weak_reach_tolerance=self.weak_reach_tolerance,
            weak_corridor_ratio=self.weak_corridor_ratio,
            weak_spread_fraction=self.weak_spread_fraction,
            weak_speed_ratio_down=self.weak_speed_ratio_down,
            weak_speed_ratio_up=self.weak_speed_ratio_up,
            weak_confirmation_confidence=self.weak_confirmation_confidence,
        )

    def as_metadata(self) -> dict:
        return {
            "max_coast_seconds": self.max_coast_seconds,
            "hard_split_seconds": self.hard_split_seconds,
            "reach_base_fraction": self.reach_base_fraction,
            "reach_rate_fraction": self.reach_rate_fraction,
            "reversal_degrees": self.reversal_degrees,
            "speed_jump_up": self.speed_jump_up,
            "speed_jump_down": self.speed_jump_down,
            "velocity_window": self.velocity_window,
            "min_event_confirmed_frames": self.min_event_confirmed_frames,
            "ambiguous_confidence": self.ambiguous_confidence,
            "clip_margin_seconds": self.clip_margin_seconds,
            "max_pre_roll_seconds": self.max_pre_roll_seconds,
            "max_post_roll_seconds": self.max_post_roll_seconds,
            "weak_min_total_measurements": self.weak_min_total_measurements,
            "weak_min_confirmed_frames": self.weak_min_confirmed_frames,
            "weak_min_candidate_frames": self.weak_min_candidate_frames,
            "weak_min_signal_agreement": self.weak_min_signal_agreement,
            "weak_corridor_ratio": self.weak_corridor_ratio,
            "weak_confirmation_confidence": self.weak_confirmation_confidence,
        }


@dataclass(frozen=True)
class ResolvedSegmentation:
    """SegmentationConfig expressed in one specific video's units."""

    fps: float
    width: int
    height: int
    frame_diagonal: float
    max_coast_frames: int
    hard_split_frames: int
    reach_base_px: float
    reach_rate_px: float
    reversal_degrees: float
    speed_jump_up: float
    speed_jump_down: float
    velocity_window: int
    min_event_confirmed_frames: int
    ambiguous_confidence: float
    clip_margin_frames: int
    black_frame_luminance_threshold: float = BLACK_FRAME_LUMINANCE_THRESHOLD
    black_frame_run_barrier: int = BLACK_FRAME_RUN_BARRIER
    scene_cut_diff_threshold: float = SCENE_CUT_DIFF_THRESHOLD
    max_pre_roll_frames: int = 75
    max_post_roll_frames: int = 45
    pitch_activity_quiet_threshold: float = PITCH_ACTIVITY_QUIET_THRESHOLD
    pitch_activity_active_threshold: float = PITCH_ACTIVITY_ACTIVE_THRESHOLD
    activity_min_window_frames: int = 4
    activity_quiet_hold_frames: int = 4
    fragment_merge_max_gap_frames: int = 30
    fragment_merge_min_combined_frames: int = MIN_DELIVERY_FRAMES
    fragment_merge_corridor_ratio: float = FRAGMENT_MERGE_CORRIDOR_RATIO
    fragment_merge_reach_base_px: float = 0.0
    fragment_merge_reach_rate_px: float = 0.0
    # ── Coherent weak confirmation (see core/constants.py) ────────────────────
    weak_min_total_measurements: int = MIN_DELIVERY_FRAMES
    weak_min_confirmed_frames: int = 2
    weak_min_candidate_frames: int = 4
    weak_min_signal_agreement: int = 4
    weak_max_silence_fraction: float = 0.4
    weak_min_contiguous_fraction: float = 0.6
    weak_reach_tolerance: float = 1.5
    weak_corridor_ratio: float = 0.5
    weak_spread_fraction: float = 0.35
    weak_speed_ratio_down: float = 0.2
    weak_speed_ratio_up: float = 5.0
    weak_confirmation_confidence: float = 0.7

    @property
    def merge_horizon_frames(self) -> int:
        """
        How far ahead a closed candidate must be held before it can no longer be
        merged with anything.

        This is the deferral window the tracking loop needs: a candidate older
        than this relative to the current frame has no possible partner left, so
        it can be finalised. It is the merge gap, not a guess — a merge is
        bounded by `fragment_merge_max_gap_frames`, so waiting any longer cannot
        change the outcome.
        """
        return self.fragment_merge_max_gap_frames

    def as_metadata(self) -> dict:
        return {
            "fps": round(self.fps, 6),
            "frame_diagonal": round(self.frame_diagonal, 3),
            "max_coast_frames": self.max_coast_frames,
            "hard_split_frames": self.hard_split_frames,
            "reach_base_px": round(self.reach_base_px, 3),
            "reach_rate_px": round(self.reach_rate_px, 4),
            "reversal_degrees": self.reversal_degrees,
            "velocity_window": self.velocity_window,
            "min_event_confirmed_frames": self.min_event_confirmed_frames,
            "ambiguous_confidence": self.ambiguous_confidence,
            "clip_margin_frames": self.clip_margin_frames,
            "max_pre_roll_frames": self.max_pre_roll_frames,
            "max_post_roll_frames": self.max_post_roll_frames,
            "activity_min_window_frames": self.activity_min_window_frames,
            "activity_quiet_hold_frames": self.activity_quiet_hold_frames,
            "weak_min_total_measurements": self.weak_min_total_measurements,
            "weak_min_confirmed_frames": self.weak_min_confirmed_frames,
            "weak_min_candidate_frames": self.weak_min_candidate_frames,
            "weak_min_signal_agreement": self.weak_min_signal_agreement,
            "weak_confirmation_confidence": self.weak_confirmation_confidence,
        }


@dataclass
class PipelineConfig:
    """
    Per-analysis pipeline tunables.

    Defaults mirror app.core.constants. Callers may override for a specific
    analysis; the overrides are recorded in the analysis metadata so results
    remain reproducible.
    """

    # Delivery segmentation
    min_delivery_frames: int = MIN_DELIVERY_FRAMES
    min_frames_between_deliveries: int = MIN_FRAMES_BETWEEN_DELIVERIES
    # Event-based segmentation. `replace(cfg, segmentation=...)` overrides it.
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)

    # Ring buffer
    ring_capacity_frames: int = RING_BUFFER_CAPACITY_FRAMES
    ring_max_bytes: int = RING_BUFFER_MAX_BYTES
    ring_jpeg_quality: int = RING_BUFFER_JPEG_QUALITY

    # Clip window (seconds)
    pre_roll_seconds: float = PRE_ROLL_SECONDS
    post_roll_seconds: float = POST_ROLL_SECONDS

    # Video writing
    codec_preference: tuple = VIDEO_CODEC_PREFERENCE

    # Shot inference
    shot_sampling_max_in_memory_bytes: int = SHOT_SAMPLING_MAX_IN_MEMORY_BYTES

    # Missing-speed handling. Phase 3 decision D1: default OFF. A null speed is
    # reported as null; it is never replaced by a median or any other ball's speed.
    impute_missing_speeds: bool = False

    # ── Ground-plane geometry calibration ────────────────────────────────────
    # Pixel coordinates of the four pitch corners on THIS video's frames, as
    # {"TL": (x, y), "TR": ..., "BL": ..., "BR": ...}.
    #
    # `None` is the default and means NO calibration was performed, which
    # suppresses every ground-plane quantity (speed_kmh, length, line, swing,
    # bounce_angle, and all metre coordinates) to `null`. That suppression is not
    # a missing feature to be worked around: this repository has no corner
    # measurement procedure and no calibration input in any API, so a non-None
    # value here is only ever reachable by an operator who measured corners on
    # the specific video being analysed. The previous code passed nothing at all
    # and inherited a default of `True`, which meant the suppression branch was
    # dead code and every delivery carried a speed derived from corner fractions
    # measured on a different broadcast template.
    pitch_corners_px: Optional[dict] = None

    # ── Delivery purity gating ──────────────────────────────────────────────
    # Which post-pass validation verdicts may appear in `result.json`'s
    # `deliveries` list. See services/pipeline.py for the full rule; the short
    # version is that a clip judged to hold two batting events, or none, is never
    # presented as a delivery, and an undecidable one is presented only with its
    # AMBIGUOUS verdict attached so nothing is silently claimed valid.
    accepted_validation_classes: tuple = ("SINGLE_EVENT", "AMBIGUOUS")

    # Write the extracted raw clips alongside the overlays.
    keep_raw_clips: bool = True

    @property
    def geometry_calibrated(self) -> bool:
        """Whether real per-video pitch corners are available."""
        corners = self.pitch_corners_px
        return bool(corners) and {"TL", "TR", "BL", "BR"} <= set(corners)

    def pre_roll_frames(self, fps: float) -> int:
        return max(0, int(round(self.pre_roll_seconds * fps)))

    def post_roll_frames(self, fps: float) -> int:
        return max(0, int(round(self.post_roll_seconds * fps)))

    def as_metadata(self) -> dict:
        """Config snapshot for the persisted analysis — keeps runs reproducible."""
        return {
            "min_delivery_frames": self.min_delivery_frames,
            "min_frames_between_deliveries": self.min_frames_between_deliveries,
            "min_frames_between_deliveries_note": (
                "DEPRECATED and unused as a segmentation rule. Retained so the "
                "value upstream used is still visible in the run record."
            ),
            "segmentation": self.segmentation.as_metadata(),
            "ring_capacity_frames": self.ring_capacity_frames,
            "ring_max_bytes": self.ring_max_bytes,
            "ring_jpeg_quality": self.ring_jpeg_quality,
            "pre_roll_seconds": self.pre_roll_seconds,
            "post_roll_seconds": self.post_roll_seconds,
            "codec_preference": list(self.codec_preference),
            "shot_sampling_max_in_memory_bytes": self.shot_sampling_max_in_memory_bytes,
            "impute_missing_speeds": self.impute_missing_speeds,
            "geometry_calibrated": self.geometry_calibrated,
            "pitch_corners_px": (
                {
                    k: [float(v[0]), float(v[1])]
                    for k, v in self.pitch_corners_px.items()
                }
                if self.geometry_calibrated else None
            ),
            "accepted_validation_classes": list(self.accepted_validation_classes),
            "keep_raw_clips": self.keep_raw_clips,
        }


@dataclass
class Paths:
    """Filesystem layout for a single analysis."""

    analysis_id: str
    root: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root = analysis_root(self.analysis_id)

    @property
    def tracking_json(self) -> Path:
        return self.root / "tracking.json"

    @property
    def result_json(self) -> Path:
        return self.root / "result.json"

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"

    @property
    def overlays_dir(self) -> Path:
        return self.root / "overlays"

    @property
    def source_video(self) -> Path:
        return self.root / "source.mp4"

    def clip_path(self, delivery_id: int) -> Path:
        return self.clips_dir / f"delivery_{delivery_id:03d}.mp4"

    def overlay_path(self, delivery_id: int) -> Path:
        return self.overlays_dir / f"delivery_{delivery_id:03d}.mp4"

    def ensure(self) -> "Paths":
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.overlays_dir.mkdir(parents=True, exist_ok=True)
        return self


def analysis_root(analysis_id: str) -> Path:
    """
    Directory holding one analysis: `result.json`, `tracking.json`, `source.mp4`,
    `clips/` and `overlays/`.

    Stored under `data/runs/`, not `data/analyses/`. The on-disk word is "run"
    because the directory holds generated output that can be deleted and
    regenerated; the API and the persisted documents still say "analysis", which is
    the domain term. The two only meet here.
    """
    return RUNS_DIR / analysis_id


def ensure_base_dirs() -> None:
    for d in (DATA_DIR, UPLOADS_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Re-exported so services can import from one place.
__all__ = [
    "ALLOWED_VIDEO_EXTENSIONS",
    "BACKEND_DIR",
    "BALL_MODEL_PATH",
    "DATASETS_DIR",
    "DATA_DIR",
    "MANIFESTS_DIR",
    "MODELS_DIR",
    "PITCH_CONFIG",
    "PipelineConfig",
    "Paths",
    "PitchConfig",
    "PROJECT_DIR",
    "ResolvedSegmentation",
    "RUNS_DIR",
    "SAMPLES_DIR",
    "SHOT_MODEL_PATH",
    "SHOT_MODEL_PROVENANCE_NAME",
    "SegmentationConfig",
    "UPLOADS_DIR",
    "analysis_root",
    "ensure_base_dirs",
    "resolve_shot_model_path",
    "os",
]