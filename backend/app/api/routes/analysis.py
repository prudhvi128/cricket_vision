"""
analysis.py — Starting an analysis and following it to a result.

DESIGN NOTES THAT MATTER
------------------------
*The frontend talks to `/api/analyze` and `/api/analysis/{id}/...`.* That is the
documented contract in docs/API_CONTRACT.md. The earlier `/api/upload` +
`/api/analyses/{id}/...` pair is kept as a thin alias so the merged
CricketShot-Classification frontend keeps working; both call the same handlers, so
there is one implementation and one behaviour.

*Progress is Server-Sent Events as well as pollable JSON.* The merged
CricketShot-Classification frontend opened an `EventSource` against what was a
normal JSON endpoint, so progress could never have worked. `/progress` is a real
`text/event-stream` and `/progress/json` is the pollable equivalent, so either
client style works.

*Results are served from disk.* `result.json` is the persisted contract. The
frontend-facing `/result` route serialises it down to a clean array of
deliveries (app.schemas.frontend); `/result/internal` serves the document
itself, with the debugging and validation detail, for diagnostics. An analysis
survives a server restart as far as the filesystem is concerned.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from ...core.config import Paths
from ...schemas.frontend import DeliveryOut as FrontendDelivery
from ...schemas.frontend import serialize_result_document
from ...storage.persistence import read_json
from ..deps import get_jobs
from ..helpers import receive_upload, result_or_404

router = APIRouter()


# ── Upload / analyse ─────────────────────────────────────────────────────────
@router.post("/analyze", status_code=202)
async def analyze(
    request: Request,
    video: UploadFile = File(..., description="The cricket video to analyse"),
) -> dict:
    """
    Primary entry point. Accepts the upload, returns immediately.

        POST /api/analyze
        multipart/form-data
        video=<file>

    Responds `202 Accepted` with an `analysis_id`. Processing continues on a
    background thread; poll `/api/analysis/{analysis_id}/status`.
    """
    return await receive_upload(video, "video")


@router.post("/upload", status_code=202)
async def upload(file: UploadFile = File(...)) -> dict:
    """
    Alias of `/api/analyze` with a `file=` form field.

    Kept because the merged CricketShot-Classification frontend posts to
    `/api/upload`. Same handler, same behaviour, same `analysis_id` space.
    """
    return await receive_upload(file, "file")


# ── Status ────────────────────────────────────────────────────────────────────
@router.get("/analysis/{analysis_id}/status")
def analysis_status(analysis_id: str) -> dict:
    """
    Pollable processing state.

    `stage` is one of the documented PIPELINE_STAGES values; `substage` names
    interleaved activity inside the tracking pass; `progress` is 0..100 and only
    ever increases. `units` carries the real counts behind it.
    """
    jobs = get_jobs()
    job = jobs.get(analysis_id)
    if job is None:
        # A result on disk means the analysis completed in an earlier process.
        doc = read_json(Paths(analysis_id=analysis_id).result_json)
        if doc is not None:
            return {
                "analysis_id": analysis_id,
                "status": "completed",
                "stage": "completed",
                "substage": None,
                "progress": 100.0,
                "percent": 100.0,
                "units": {},
                "frames_done": None,
                "frames_total": None,
                "detail": (
                    f"{len(doc.get('deliveries', []))} deliveries "
                    f"(recovered from disk; not tracked in memory)"
                ),
                "error": None,
                "cancel_requested": False,
                "created_at": None,
                "started_at": None,
                "finished_at": None,
                "source_name": None,
            }
        raise HTTPException(
            404, f"Unknown analysis {analysis_id}."
        )
    return job.as_dict()


@router.get("/analyses/{analysis_id}/status")
def analysis_status_plural(analysis_id: str) -> dict:
    """
    Alias of `/analysis/{id}/status` for the legacy plural path.

    Keeps the old lenient contract: an unknown id answers `200` with
    `status: "unknown"` rather than a 404, because the merged frontend polls this
    spelling and treats a 404 as a transport failure. The documented singular
    route stays strict, so a client using the contract cannot mistake a typo for
    an analysis that is merely not finished.
    """
    try:
        return analysis_status(analysis_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            return {"analysis_id": analysis_id, "status": "unknown"}
        raise


# ── Results ───────────────────────────────────────────────────────────────────
@router.get("/analysis/{analysis_id}/result")
def analysis_result(analysis_id: str) -> list[FrontendDelivery]:
    """
    The frontend contract: a clean ARRAY of deliveries.

        [ delivery_1, delivery_2, ... ]

    Each entry is self-contained — video, trajectory, bounce, bowling analytics,
    shot classification and quality state — so a client does
    `deliveries[currentIndex]` for Previous/Next navigation and never has to
    reassemble a delivery from two internal objects.

    Everything the pipeline records for debugging stays internal: the segmenter
    evidence, purity validation verdicts, provenance, speed samples, clip frame
    maps and filesystem paths are all in `result.json` and on the
    `/result/internal` route below, never in this response.
    """
    return serialize_result_document(result_or_404(analysis_id))


@router.get("/analysis/{analysis_id}/result/internal")
def analysis_result_internal(analysis_id: str) -> dict:
    """
    The complete persisted `result.json` — summary, calibration, performance,
    warnings, quarantine and the full per-delivery records.

    This is the diagnostics route, not the frontend contract: it exposes
    `event`, `validation`, `provenance`, `clip_path` and every other internal
    field. The frontend must call `/result` instead.
    """
    return result_or_404(analysis_id)


@router.get("/analyses/{analysis_id}/result")
def analysis_result_plural(analysis_id: str) -> list[FrontendDelivery]:
    """Alias of `/analysis/{id}/result` for the legacy plural path."""
    return analysis_result(analysis_id)


@router.get("/analyses/{analysis_id}/result/internal")
def analysis_result_plural_internal(analysis_id: str) -> dict:
    """Alias of `/analysis/{id}/result/internal` for the legacy plural path."""
    return analysis_result_internal(analysis_id)


@router.get("/analyses/{analysis_id}/tracking")
def analysis_tracking(analysis_id: str) -> dict:
    """
    The tracker output alone, with no shot labels attached.

    Useful for auditing: it is what the single pass actually measured, before
    any inference ran on top of it.
    """
    doc = read_json(Paths(analysis_id=analysis_id).tracking_json)
    if doc is None:
        raise HTTPException(404, f"No tracking result for analysis {analysis_id}")
    return doc


# ── Progress ──────────────────────────────────────────────────────────────────
@router.get("/analysis/{analysis_id}/progress")
async def analysis_progress(analysis_id: str, request: Request) -> StreamingResponse:
    """
    Server-Sent Events stream of progress.

    Genuinely an event stream, unlike the merged frontend's original route.
    Polls the job registry rather than being pushed to, which keeps the
    registry the single source of truth and avoids threading a callback through
    the analysis for no benefit.
    """
    jobs = get_jobs()

    async def event_stream():
        last_payload = None
        # Poll faster than the analysis emits (every 0.5s) so the stream keeps up.
        while True:
            if await request.is_disconnected():
                break
            job = jobs.get(analysis_id)
            if job is None:
                yield f"event: error\ndata: {json.dumps({'error': 'unknown analysis'})}\n\n"
                break
            payload = job.as_dict()
            # Only emit on change, so an idle analysis is not a firehose.
            if payload != last_payload:
                last_payload = payload
                yield f"event: progress\ndata: {json.dumps(payload)}\n\n"
            if job.status in ("completed", "failed", "cancelled"):
                yield f"event: done\ndata: {json.dumps(payload)}\n\n"
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/analyses/{analysis_id}/progress")
async def analysis_progress_plural(analysis_id: str, request: Request):
    """Alias of `/analysis/{id}/progress` for the legacy plural path."""
    return await analysis_progress(analysis_id, request)


@router.get("/analysis/{analysis_id}/progress/json")
def analysis_progress_json(analysis_id: str) -> dict:
    """The same payload as `/status`, for clients that prefer an explicit route."""
    return analysis_status(analysis_id)


# ── Cancellation ──────────────────────────────────────────────────────────────
@router.post("/analysis/{analysis_id}/cancel")
def cancel_analysis(analysis_id: str) -> dict:
    """
    Request cancellation.

    Upstream CricketShot-Classification's frontend called `/api/cancel`, which did
    not exist on the backend — cancelling was impossible. This route exists and
    reports honestly whether the request was accepted.
    """
    ok = get_jobs().cancel(analysis_id)
    return {
        "analysis_id": analysis_id,
        "cancel_requested": ok,
        "detail": (
            "Cancellation requested; it takes effect after the current delivery."
            if ok
            else "Already finished, or unknown analysis."
        ),
    }


@router.post("/analyses/{analysis_id}/cancel")
def cancel_analysis_plural(analysis_id: str) -> dict:
    """Alias of `/analysis/{id}/cancel` for the legacy plural path."""
    return cancel_analysis(analysis_id)


@router.get("/analyses")
def list_analyses() -> dict:
    """Every analysis this process is tracking. Transients are not persisted."""
    return {"analyses": [j.as_dict() for j in get_jobs().list()]}