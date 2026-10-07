"""
pipeline.py — Orchestrates the stages around the single tracking pass.

Stage order, and what each stage is allowed to touch the SOURCE video for:

    1. tracking      source video: OPENED ONCE, decoded once   (C1)
    2. shot          source video: not touched. Reads frames retained
                     by stage 1, or the clip file it wrote     (C1, #4, #5)
    3. overlay       source video: not touched. Reads the clip file (C1, #5)

Nothing after stage 1 receives a path to the source video. This is enforced
structurally: `run()` takes the source path, uses it only to build stage 1, and
never passes it onwards.

MODEL LOADING IS SHARED
-----------------------
`run()` loads both models once and reuses them across deliveries within the
analysis. Loading a 72 MB transformer checkpoint per delivery would dominate the
runtime of a 40-ball video.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..core.config import PipelineConfig, Paths, resolve_shot_model_path
from ..core.constants import (
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    SHOT_MODEL_PROVENANCE_NAME,
)
from ..schemas.tracking import Delivery
from ..storage.persistence import atomic_write_json
from ..analytics.calibration import calibration_metadata
from ..video.overlay import OverlayRenderer
from ..shot_classification.inference import ShotClassifier, load_shot_model
from ..tracking.tracking_service import (
    ProgressFn,
    TrackingResult,
    TrackingService,
    _noop_progress,
)
from ..tracking.detector import BallDetector

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    analysis_id: str
    status: str = "completed"
    deliveries: list[Delivery] = field(default_factory=list)
    tracking: Optional[TrackingResult] = None
    shot_model_available: bool = False
    overlay_model_available: bool = True
    timings: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # Absolute path and filename of the checkpoint actually loaded, plus the
    # upstream model name reported in each delivery's provenance block.
    shot_model_checkpoint: Optional[str] = None
    shot_model_file: Optional[str] = None
    # Clips whose purity verdict keeps them out of `deliveries`. Persisted in full
    # rather than dropped, so "why is delivery 7 missing" is answerable from the
    # artefact alone.
    quarantined: list[dict] = field(default_factory=list)

    def result_document(self, paths: Paths, cfg: Optional[PipelineConfig] = None) -> dict:
        """
        The API-facing `result.json`.

        The same trajectory-derived numbers appear here, in the overlays and in
        the pitch map because they all read `Delivery.trajectory`. This document
        also carries the calibration block, so a client rendering `speed_kmh` has
        the calibration facts in the same payload and cannot present an estimate
        as a measurement.

        `cfg` carries this analysis's pitch corners. Without it the block would
        always describe an uncalibrated run, which would be wrong whenever an
        operator did supply corners — a result document that understates its own
        calibration is worse than no document.
        """
        cfg = cfg or PipelineConfig()
        deliveries = [d.as_dict() for d in self.deliveries]
        with_speed = [d for d in deliveries if d["bowling"].get("speed_kmh") is not None]
        imputed = [d for d in deliveries if d["bowling"].get("speed_imputed")]
        labelled = [d for d in deliveries if (d.get("shot") or {}).get("type")]
        by_validation = _tally(
            (d.get("validation") or {}).get("classification") for d in deliveries
        )
        shots = _tally((d.get("shot") or {}).get("type") for d in deliveries)

        return {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "analysis_id": self.analysis_id,
            # Prefix the clip URLs of this analysis are served under.
            "media_base_url": f"/api/analysis/{self.analysis_id}/clips",
            "calibration": calibration_metadata(
                cfg.pitch_corners_px,
                pitch_geometry=(
                    self.tracking.pitch_geometry_stats if self.tracking else None
                ),
            ),
            # The optional pitch keypoint detector's own record: whether it was
            # configured, how many submissions/detections/failures it made. Null
            # when the pass never ran. Configuration only — never a credential.
            "pitch_detection": self.tracking.pitch_summary if self.tracking else None,
            "summary": {
                "deliveries": len(deliveries),
                "quarantined": len(self.quarantined),
                "by_validation": by_validation,
                "with_speed": len(with_speed),
                "speed_imputed": len(imputed),
                "with_shot": len(labelled),
                "shot_failures": sum(
                    1 for d in deliveries
                    if d.get("shot") and not (d["shot"] or {}).get("type")
                ),
                "overlays_rendered": sum(
                    1 for d in deliveries if d["overlay"].get("rendered")
                ),
                "from_clip_fallback": sum(
                    1 for d in deliveries if d["tracking_source"] == "clip_fallback"
                ),
                "mean_speed_kmh": (
                    round(
                        sum(d["bowling"]["speed_kmh"] for d in with_speed)
                        / len(with_speed),
                        1,
                    )
                    if with_speed
                    else None
                ),
                "lengths": _tally(d["bowling"].get("length") for d in deliveries),
                "lines": _tally(d["bowling"].get("line") for d in deliveries),
                "shots": shots,
            },
            "performance": {
                "source_decodes": 1,
                "yolo_passes": 1,
                "kalman_passes": 1,
                "shot_model_loads": 1 if self.shot_model_available else 0,
                "shot_model_checkpoint": self.shot_model_checkpoint,
                "shot_model_file": self.shot_model_file,
                "shot_model_provenance_name": SHOT_MODEL_PROVENANCE_NAME,
                "timings_sec": {k: round(v, 3) for k, v in self.timings.items()},
            },
            "warnings": self.warnings,
            "errors": self.errors,
            "config": cfg.as_metadata(),
            "deliveries": deliveries,
            "quarantine": self.quarantined,
        }


def _tally(values) -> dict[str, int]:
    """Count occurrences, ignoring None. Unknown values are kept, not dropped."""
    out: dict[str, int] = {}
    for v in values:
        if v is None:
            continue
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


class UnifiedPipeline:
    """Runs tracking → shot → overlay for one analysis."""

    def __init__(
        self,
        detector: Optional[BallDetector] = None,
        shot_model=None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        # Models are injected so a server can hold them across requests.
        self.detector = detector
        self.shot_model = shot_model

    def run(
        self,
        source_video: str,
        paths: Paths,
        progress: Optional[ProgressFn] = None,
    ) -> PipelineResult:
        emit = progress or _noop_progress
        paths.ensure()
        t_total = time.perf_counter()

        result = PipelineResult(analysis_id=paths.analysis_id)

        # ── Stage 1: the one and only pass over the source video ────────────
        t0 = time.perf_counter()
        if self.detector is None:
            emit("models", 0, 1, "loading ball detector")
            self.detector = BallDetector()
        service = TrackingService(self.config)
        tracking = service.analyse(
            source_video, paths, progress=emit, detector=self.detector
        )
        result.tracking = tracking
        result.timings["tracking"] = time.perf_counter() - t0
        result.warnings.extend(tracking.warnings)

        # ── Purity gate on the assembled result ────────────────────────────
        # The tracking pass already refuses a window it can show is impure BEFORE
        # writing it, so this partition should find nothing to quarantine. It runs
        # anyway because the end-of-pass validator judges something the
        # pre-persistence gate structurally cannot: a clip whose frames were
        # individually clean but which the validator's independent re-segmentation
        # says hold two events. Both verdicts are kept; only the accepted ones
        # become deliveries.
        accepted, quarantined = self._partition_by_purity(result, tracking)
        result.deliveries = accepted
        result.quarantined = quarantined

        if not result.deliveries:
            result.timings["total"] = time.perf_counter() - t_total
            self._persist(paths, result, self.config)
            emit("complete", 0, 0, "no validated deliveries")
            return result

        # ── Stage 2: shot classification (never opens the source) ───────────
        t0 = time.perf_counter()
        classifier = self._build_classifier(emit, result)
        if classifier is not None:
            # Frames the tracking pass already retained, keyed by clip-frame
            # position. Preferred over re-reading the clip: same pixels, no
            # second decode.
            retained = {
                did: r.shot_frames
                for did, r in (tracking.clip_results or {}).items()
                if r.shot_frames
            }
            for d in result.deliveries:
                emit("shot", d.delivery_id, len(result.deliveries), "classifying")
                # Passed through with its position keys intact. The retained
                # frames are a sparse subset of the clip, so their positions are
                # what tells the sampler which moments to use; flattening them to a
                # list would renumber them and change the model's input.
                d.shot = classifier.classify(
                    clip_path=d.clip_path,
                    in_memory_frames=retained.get(d.delivery_id),
                    expected_total_frames=d.clip.frame_count if d.clip else None,
                    delivery_id=d.delivery_id,
                    # The SAME offset the overlay and the trajectory use, so the
                    # logged original-frame numbers are directly comparable.
                    frame_offset=d.clip.frame_offset if d.clip else None,
                )
                if d.shot.input_source == "clip_file":
                    result.warnings.append(
                        f"Delivery {d.delivery_id}: shot frames were not "
                        "available in memory, so the clip file was re-read "
                        "(permitted; the source video was not reopened)."
                    )
                if d.shot.error:
                    # One unclassifiable delivery must not fail the analysis, and
                    # it must not be reported as a successful label either.
                    result.warnings.append(
                        f"Delivery {d.delivery_id}: shot classification failed "
                        f"({d.shot.error}); shot is null for this delivery."
                    )
        result.timings["shot"] = time.perf_counter() - t0

        # ── Stage 3: overlays from the stored trajectory ────────────────────
        t0 = time.perf_counter()
        renderer = OverlayRenderer(codec_preference=self.config.codec_preference)
        for d in result.deliveries:
            emit("overlay", d.delivery_id, len(result.deliveries), "rendering")
            d.overlay = renderer.render(d, paths)
            if not d.overlay.rendered:
                # The clip and every measurement survive; only the video is
                # missing, and the reason travels with the record.
                result.warnings.append(
                    f"Delivery {d.delivery_id}: overlay not rendered "
                    f"({d.overlay.error}); clip and trajectory are unaffected."
                )
        result.timings["overlay"] = time.perf_counter() - t0

        result.timings["total"] = time.perf_counter() - t_total
        emit("persist", 1, 1, "writing result.json")
        self._persist(paths, result, self.config)
        result.timings["total"] = time.perf_counter() - t_total
        emit("complete", len(result.deliveries), len(result.deliveries), "done")
        return result

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _partition_by_purity(
        self, result: PipelineResult, tracking: TrackingResult
    ) -> tuple[list[Delivery], list[dict]]:
        """
        Split the tracking pass's deliveries by their purity verdict.

        ACCEPTED
            `PipelineConfig.accepted_validation_classes` — SINGLE_EVENT, and
            AMBIGUOUS. AMBIGUOUS is included deliberately: the requirement is not
            to *claim* an undecidable clip is valid, and it is not claimed — the
            verdict, its confidence and `requires_human_review` all travel with
            the record, and `summary.by_validation` counts them separately. A
            client that wants only unambiguous deliveries filters on
            `validation.classification`; the backend does not silently drop a
            third of a real over's deliveries to make its own output tidier.

        QUARANTINED
            MULTIPLE_EVENTS and NO_EVENT. These are not failures of the run, they
            are windows that violate the one-clip-one-event invariant. They are
            never presented as deliveries and are never deleted: the full record,
            including the reason and the frame range, is persisted under
            `quarantine`.
        """
        by_id = {
            rec.delivery_id: rec for rec in (
                tracking.validation.records if tracking.validation else []
            )
        }
        # Read from the live config, not a fresh default: a caller that tightened
        # or widened the accepted set must have that honoured.
        accepted_cls = set(self.config.accepted_validation_classes)

        accepted: list[Delivery] = []
        quarantined: list[dict] = []
        for d in tracking.deliveries:
            record = by_id.get(d.delivery_id)
            classification = (
                record.classification if record is not None else None
            )
            # No verdict at all is not a pass. An unvalidated clip is quarantined
            # rather than presented as a delivery on the strength of silence.
            if classification is None or classification in accepted_cls:
                if classification is None:
                    log.error(
                        "Delivery %d has no purity verdict; quarantining rather "
                        "than presenting an unvalidated clip as a delivery",
                        d.delivery_id,
                    )
                    quarantined.append({
                        "delivery_id": d.delivery_id,
                        "delivery_ref": d.delivery_ref,
                        "classification": None,
                        "reason": "the clip was never validated",
                        "requires_human_review": True,
                        "record": d.as_dict(),
                    })
                    continue
                accepted.append(d)
            else:
                quarantined.append({
                    "delivery_id": d.delivery_id,
                    "delivery_ref": d.delivery_ref,
                    "classification": classification,
                    "reason": record.reason if record else "",
                    "requires_human_review": bool(
                        record.requires_human_review if record else True
                    ),
                    "record": d.as_dict(),
                })
                result.warnings.append(
                    f"Delivery {d.delivery_id} quarantined: {classification} — "
                    f"{record.reason if record else 'no reason recorded'}"
                )

        if quarantined:
            log.error(
                "%d of %d clips quarantined by the purity gate",
                len(quarantined), len(tracking.deliveries),
            )
        return accepted, quarantined

    def _build_classifier(
        self, emit: ProgressFn, result: PipelineResult
    ) -> Optional[ShotClassifier]:
        """
        Load the shot model on first use and keep it for the process lifetime.

        One 72 MB transformer load per ANALYSIS, not per delivery: the pipeline
        holds it on `self` and `ModelCache` harvests it after the run, so a second
        upload to the same server reuses the weights.

        `resolve_shot_model_path` accepts the upstream checkpoint name
        (`cricket_model_transformer.ckpt`) as well as the local copy, so this works
        against a checkout that kept either filename.
        """
        if self.shot_model is None:
            emit("models", 0, 1, "loading shot classifier")
            try:
                from ..shot_classification.model import ImprovedSOTAModel  # local import: heavy

                checkpoint = resolve_shot_model_path()
                result.shot_model_checkpoint = str(checkpoint)
                result.shot_model_file = checkpoint.name
                self.shot_model = load_shot_model(checkpoint, ImprovedSOTAModel)
            except Exception as exc:
                msg = f"Shot classifier unavailable: {exc}"
                log.error(msg)
                result.errors.append(msg)
                result.shot_model_available = False
                return None
        else:
            # Weights were injected from the process-wide cache, so no load
            # happened here. The provenance still has to be recorded: a result
            # document that cannot say WHICH checkpoint produced its labels is
            # not reproducible, and the cached weights came from that file.
            try:
                checkpoint = resolve_shot_model_path()
                result.shot_model_checkpoint = str(checkpoint)
                result.shot_model_file = checkpoint.name
            except FileNotFoundError:
                # Cached weights whose checkpoint has since been removed. The
                # labels are still produced; only the provenance is unavailable,
                # and that is stated rather than filled in.
                result.warnings.append(
                    "Shot model weights were reused from the in-process cache but "
                    "the checkpoint file is no longer on disk; the loaded weights' "
                    "source file could not be recorded."
                )
        result.shot_model_available = True
        return ShotClassifier(self.shot_model)

    @staticmethod
    def _persist(
        paths: Paths, result: PipelineResult, cfg: Optional[PipelineConfig] = None
    ) -> None:
        atomic_write_json(
            paths.result_json, result.result_document(paths, cfg)
        )