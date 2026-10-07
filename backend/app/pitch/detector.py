"""
detector.py — Roboflow pitch keypoint detection. OPTIONAL and never blocking.

The tracking loop owns one decode of the source video and one YOLO call per
frame; nothing here may add a frame decode, a YOLO call, or a wait. So this
module does two things only:

  * turn one frame into an HTTP request to a hosted keypoint model, on a
    BACKGROUND thread (see service.py), with a hard ceiling on the round trip;
  * parse whatever comes back — defensively — into keypoints in SOURCE pixel
    coordinates.

THE RESPONSE IS NOT TRUSTED TO LOOK LIKE ANYTHING IN PARTICULAR
---------------------------------------------------------------
No live response could be inspected while writing this (the feature is inert
without ROBOFLOW_API_KEY), so the parser accepts every documented Roboflow
shape instead of the one shape someone remembered:

  * a bare list of predictions, or an object with `predictions`;
  * `predictions` nested one level deeper (some responses wrap it again);
  * keypoints under `keypoints`, under `points`, or as a plain
    {name: [x, y]} mapping;
  * a keypoint as {x, y, ...} or as a 2- or 3-element array;
  * names spelled `name`, `class` or `keypoint`, falling back to `point_1..N`;
  * confidences spelled `confidence`, `score` or `conf`.

Anything malformed is dropped rather than repaired: a keypoint that cannot be
read has no coordinate, and a coordinate that does not exist must never reach
the calibration, because the quad it would build is a fabrication drawn over
the video.

Nothing here logs, stores or transmits the API key. Errors are scrubbed before
they leave this module, so a URL carrying `api_key=...` can never reach a log
line or an analysis record.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ..core.config import PitchConfig

log = logging.getLogger(__name__)

# Frame size, as (width, height).
Size = tuple[int, int]


class PitchDetectError(Exception):
    """A detection attempt failed. Never fatal to the analysis."""


class PitchTimeout(PitchDetectError):
    """The round trip exceeded the configured ceiling."""


class PitchDetectorUnavailable(PitchDetectError):
    """No credential, or the SDK could not be imported."""


@dataclass(frozen=True)
class Keypoint:
    """One pitch keypoint, in SOURCE pixel coordinates."""

    name: str
    x: float
    y: float
    confidence: Optional[float] = None

    def as_pair(self) -> list[float]:
        return [round(self.x, 2), round(self.y, 2)]


@dataclass(frozen=True)
class PitchDetection:
    """
    One parsed model response, in SOURCE pixel coordinates.

    `confidence` is the mean over the keypoints that carried a confidence; it is
    None only when the response carried no confidences at all, which is itself
    information (the caller then treats it as "unknown", not as "high").
    """

    keypoints: list[Keypoint]
    frame_number: int
    image_size: Size
    confidence: Optional[float] = None

    @property
    def named(self) -> dict[str, tuple[float, float]]:
        return {k.name: (k.x, k.y) for k in self.keypoints}


# ── Parsing ───────────────────────────────────────────────────────────────────
def parse_pitch_response(
    payload: Any,
    *,
    frame_number: int,
    image_size: Size,
) -> Optional[PitchDetection]:
    """
    Turn a raw model response into keypoints, or None if it holds none.

    Pure and side-effect free: given the same bytes it always produces the same
    result, which is what makes it testable against fixtures rather than against
    a live endpoint.
    """
    predictions = _predictions(payload)
    if not predictions:
        return None

    best: list[Keypoint] = []
    best_conf: Optional[float] = None
    for pred in predictions:
        kps, conf = _keypoints_of(pred, image_size)
        if len(kps) > len(best):
            best, best_conf = kps, conf

    if not best:
        # No prediction carried keypoints, so the container we unwrapped may
        # hold them directly (`{"points": [{"name": ..., "x": ..., "y": ...}]}`)
        # rather than holding predictions that hold them. Read only the
        # keypoint-named containers: falling back to `predictions` would turn
        # a prediction object into a keypoint, which is exactly the repair this
        # parser refuses to do.
        best = _bare_keypoints(payload, image_size)
    if not best:
        return None
    best = _scale_if_normalised(best, image_size)
    if best_conf is None:
        best_conf = _mean([k.confidence for k in best if k.confidence is not None])
    return PitchDetection(
        keypoints=best,
        frame_number=frame_number,
        image_size=image_size,
        confidence=None if best_conf is None else round(float(best_conf), 4),
    )


def _scale_if_normalised(kps: list[Keypoint], image_size: Size) -> list[Keypoint]:
    """
    Unit coordinates become pixel coordinates, together or not at all.

    Decided over the WHOLE set rather than per point: scaling one point and not
    its neighbour would bend the quad, and a genuine pixel coordinate of 0 or 1
    only exists alongside other coordinates in the hundreds. If every coordinate
    sits inside [0, 1] the response is normalised, and the frame size is the only
    honest way to read it.
    """
    if not kps or image_size[0] <= 0 or image_size[1] <= 0:
        return kps
    if not all(0.0 <= k.x <= 1.0 and 0.0 <= k.y <= 1.0 for k in kps):
        return kps
    width, height = image_size
    return [
        Keypoint(k.name, k.x * width, k.y * height, k.confidence) for k in kps
    ]


def _bare_keypoints(payload: Any, image_size: Size) -> list[Keypoint]:
    """
    Keypoints read straight out of a keypoint-named container.

    Used only when no prediction in the response carried any: the container
    named `points`/`keypoints` then holds the keypoint set itself, either as a
    list of entries or as a `{name: [x, y]}` mapping.
    """
    if not isinstance(payload, dict):
        return []
    for key in ("keypoints", "points"):
        value = payload.get(key)
        if isinstance(value, (list, dict)):
            kps = _keypoint_container(value, image_size)
            if kps:
                return kps
    return []


def _predictions(payload: Any) -> list[Any]:
    """Every documented way a response can carry predictions."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("predictions", "points", "keypoints"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            # `{"predictions": {"predictions": [...]}}` — seen in wrapped responses.
            nested = value.get("predictions")
            if isinstance(nested, list):
                return nested
            return [value]
    # A single prediction object at the top level.
    if "x" in payload or "y" in payload:
        return [payload]
    return []


def _keypoints_of(pred: Any, image_size: Size) -> tuple[list[Keypoint], Optional[float]]:
    """Keypoints of one prediction, plus that prediction's own confidence."""
    if isinstance(pred, (list, tuple)):
        return _keypoint_list(pred, image_size), None
    if not isinstance(pred, dict):
        return [], None

    # A prediction that itself nests predictions (batched responses).
    for key in ("predictions", "instances"):
        nested = pred.get(key)
        if isinstance(nested, list) and nested:
            return _keypoints_of(nested[0], image_size)

    own_conf = _float(pred.get("confidence"))
    for key in ("keypoints", "points", "key_points"):
        value = pred.get(key)
        if value is None:
            continue
        kps = _keypoint_container(value, image_size)
        if kps:
            return kps, own_conf
    return [], own_conf


def _keypoint_container(value: Any, image_size: Size) -> list[Keypoint]:
    if isinstance(value, dict):
        # {name: [x, y]} or {name: {x, y}} or {name: [x, y, confidence]}
        out: list[Keypoint] = []
        for idx, (name, raw) in enumerate(value.items(), start=1):
            parsed = _point_from(raw, image_size)
            if parsed is not None:
                x, y, conf = parsed
                out.append(Keypoint(str(name), x, y, conf))
        if out:
            return out
        # A single {x, y} object rather than a name mapping.
        parsed = _point_from(value, image_size)
        if parsed is not None:
            x, y, conf = parsed
            return [Keypoint("point_1", x, y, conf)]
        return []
    if isinstance(value, (list, tuple)):
        # Either a list of keypoints, or ONE keypoint as [x, y, (confidence)].
        if value and all(isinstance(v, (int, float)) for v in value):
            parsed = _point_from(value, image_size)
            if parsed is None:
                return []
            x, y, conf = parsed
            return [Keypoint("point_1", x, y, conf)]
        return _keypoint_list(value, image_size)
    return []


def _keypoint_list(entries: Iterable[Any], image_size: Size) -> list[Keypoint]:
    out: list[Keypoint] = []
    for idx, entry in enumerate(entries, start=1):
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("class") or entry.get("keypoint")
            parsed = _point_from(entry, image_size)
            if parsed is None:
                continue
            x, y, conf = parsed
            out.append(Keypoint(str(name) if name else f"point_{idx}", x, y, conf))
            continue
        parsed = _point_from(entry, image_size)
        if parsed is None:
            continue
        x, y, conf = parsed
        out.append(Keypoint(f"point_{idx}", x, y, conf))
    return out


def _point_from(raw: Any, image_size: Size) -> Optional[tuple[float, float, Optional[float]]]:
    """
    One (x, y, confidence) triple, or None when it cannot be read.

    Unit-vs-pixel ambiguity is NOT resolved here: it is a property of the whole
    keypoint set, so it is settled once in `_scale_if_normalised`.
    """
    x = y = None
    conf: Optional[float] = None
    if isinstance(raw, dict):
        x, y = _float(raw.get("x")), _float(raw.get("y"))
        conf = _float(raw.get("confidence", raw.get("score", raw.get("conf"))))
    elif isinstance(raw, (list, tuple)):
        if len(raw) < 2:
            return None
        x, y = _float(raw[0]), _float(raw[1])
        if len(raw) >= 3:
            conf = _float(raw[2])
    if x is None or y is None:
        return None
    if x != x or y != y:  # NaN
        return None
    return float(x), float(y), conf


# ── The client ────────────────────────────────────────────────────────────────
class RoboflowPitchDetector:
    """
    Calls the hosted model for one frame. Constructed only when a key exists.

    The `inference-sdk` import happens on FIRST USE, not at module import, so a
    checkout without the package (and without a key) starts the server exactly
    as it did before this feature existed.
    """

    def __init__(self, cfg: PitchConfig) -> None:
        self.cfg = cfg
        self._client = None
        self._init_error: Optional[str] = None
        self._consecutive_timeouts = 0
        self.disabled_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.disabled_reason is None and self._client_or_error()

    def rearm(self) -> None:
        """
        Undo a timeout-driven disable so the client can be tried again.

        A disable means "three timeouts in a row", not "the credential is
        bad" — the service calls this once per re-arm window, and the same
        three timeouts will disable it again if the service really is gone.
        """
        self.disabled_reason = None
        self._consecutive_timeouts = 0

    def detect(self, frame, frame_number: int, image_size: Size) -> PitchDetection:
        """
        Run one detection. Raises PitchDetectError on any failure — the caller
        (the service) records it and moves on; the analysis is unaffected.

        The SDK has no per-request timeout of its own, so the call runs on a
        short-lived thread and is abandoned past the ceiling. Three consecutive
        abandonments disable the detector outright: a client that cannot answer
        in time is not going to start answering later, and leaving a thread per
        attempt would accumulate in a long-lived server.
        """
        client = self._client_or_error()
        if client is None:
            raise PitchDetectorUnavailable(self._init_error or "detector unavailable")

        box: dict[str, Any] = {}
        call = threading.Thread(
            target=self._invoke, args=(client, frame, box), daemon=True
        )
        call.start()
        call.join(self.cfg.request_timeout_seconds)
        if call.is_alive():
            self._consecutive_timeouts += 1
            if self._consecutive_timeouts >= 3:
                self.disabled_reason = "timeout"
            raise PitchTimeout(
                f"no response after {self.cfg.request_timeout_seconds}s "
                f"({self._consecutive_timeouts} consecutive)"
            )
        if "error" in box:
            raise PitchDetectError(scrub(str(box["error"]), self.cfg.api_key)) from box["error"]
        if "result" not in box:
            raise PitchDetectError("model returned nothing")

        detection = parse_pitch_response(
            box["result"], frame_number=frame_number, image_size=image_size
        )
        self._consecutive_timeouts = 0
        if detection is None:
            raise PitchDetectError("response held no keypoints")
        return detection

    # ── internals ────────────────────────────────────────────────────────────
    def _invoke(self, client, frame, box: dict[str, Any]) -> None:
        try:
            box["result"] = client.infer(frame, model_id=self.cfg.model_id)
        except Exception as exc:  # noqa: BLE001 — every failure is a soft failure
            box["error"] = exc

    def _client_or_error(self):
        if self._client is not None:
            return self._client
        if self._init_error is not None:
            return None
        if not self.cfg.enabled:
            self._init_error = "no ROBOFLOW_API_KEY configured"
            return None
        try:
            from inference_sdk import InferenceConfiguration, InferenceHTTPClient
        except Exception as exc:  # pragma: no cover - depends on the environment
            self._init_error = f"inference-sdk unavailable: {scrub(str(exc), self.cfg.api_key)}"
            log.warning("Pitch detection disabled: %s", self._init_error)
            return None
        try:
            client = InferenceHTTPClient(api_url=self.cfg.api_url, api_key=self.cfg.api_key)
            client.configure(
                InferenceConfiguration(
                    confidence_threshold=self.cfg.min_confidence,
                    keypoint_confidence_threshold=self.cfg.min_confidence,
                    default_max_input_size=self.cfg.max_input_size,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._init_error = f"client setup failed: {scrub(str(exc), self.cfg.api_key)}"
            log.warning("Pitch detection disabled: %s", self._init_error)
            return None
        self._client = client
        return self._client


# ── helpers ───────────────────────────────────────────────────────────────────
def scrub(text: str, api_key: Optional[str]) -> str:
    """
    Remove the credential from anything that might be logged or persisted.

    The SDK puts the key in the query string, so an exception message or a
    requests URL can contain it verbatim. It is a secret: it never leaves this
    process in any other form than the header/query it was sent in.
    """
    if not text:
        return text
    if api_key:
        text = text.replace(api_key, "***")
    if "api_key=" in text:
        head, _, tail = text.partition("api_key=")
        sep = tail.find("&")
        text = head + "api_key=***" + (tail[sep:] if sep >= 0 else "")
    return text


def _float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


__all__ = [
    "Keypoint",
    "PitchDetectError",
    "PitchDetection",
    "PitchDetectorUnavailable",
    "PitchTimeout",
    "RoboflowPitchDetector",
    "parse_pitch_response",
    "scrub",
]
