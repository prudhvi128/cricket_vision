"""
analysis.py — Starting an analysis and following it to a result.

DESIGN NOTES THAT MATTER
------------------------
*The frontend talks to `/api/analyze` and `/api/analysis/{id}/...`, and to
nothing else.* That is the whole contract (docs/API_CONTRACT.md): start an
analysis, poll its status, fetch its result. There is exactly one spelling of
each route and one handler behind it, so two frontends cannot disagree about
what the API means.

*Progress is pollable JSON.* `/status` answers queued | running | completed |
failed with `progress` 0..100 that only ever increases, `stage` from the
documented PIPELINE_STAGES vocabulary, and `error` when a run failed. There is
no second progress route to keep in step with it.

*Results are served from disk.* `result.json` is the persisted contract. The
`/result` route serialises it down to the frontend envelope
(app.schemas.frontend): analysis id, the pitch calibration in effect, and a
clean array of deliveries. Everything the pipeline records for debugging —
segmenter evidence, validation verdicts, provenance, speed samples, clip frame
maps, filesystem paths — stays in `result.json` and is never served. An
analysis survives a server restart as far as the filesystem is concerned.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from ...core.config import Paths
from ...schemas.frontend import ResultDocument, serialize_result_document
from ...storage.persistence import read_json
from ..deps import get_jobs
from ..helpers import receive_upload, result_or_404

router = APIRouter()


# ── Upload / analyse ─────────────────────────────────────────────────────────
@router.post("/analyze", status_code=202)
async def analyze(
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
    return await receive_upload(video)


@router.post("/upload", status_code=202)
async def upload(file: UploadFile = File(...)) -> dict:
    """
    The same handler under a `file=` form field, for clients that post to
    `POST /api/upload`. Same body, same behaviour, same `analysis_id` space.
    """
    return await receive_upload(file)


# ── Status ────────────────────────────────────────────────────────────────────
@router.get("/analysis/{analysis_id}/status")
def analysis_status(analysis_id: str) -> dict:
    """
    Pollable processing state: `queued | running | completed | failed`.

    `stage` is one of the documented PIPELINE_STAGES values; `substage` names
    interleaved activity inside the tracking pass; `progress` is 0..100 and only
    ever increases. `error` carries the failure reason on a failed run.
    """
    job = get_jobs().get(analysis_id)
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
                "frames_done": len(doc.get("deliveries", [])),
                "frames_total": len(doc.get("deliveries", [])),
                "detail": (
                    f"{len(doc.get('deliveries', []))} deliveries "
                    "(recovered from disk; not tracked in memory)"
                ),
                "error": None,
                "created_at": None,
                "started_at": None,
                "finished_at": None,
                "source_name": None,
            }
        raise HTTPException(
            404, f"Unknown analysis {analysis_id}."
        )
    return job.as_dict()


# ── Results ───────────────────────────────────────────────────────────────────
@router.get("/analysis/{analysis_id}/result")
def analysis_result(analysis_id: str) -> ResultDocument:
    """
    The frontend contract: one envelope.

        {
          "analysis_id": "...",
          "status": "completed",
          "pitch": { ...calibration in effect... },
          "deliveries": [ delivery_1, delivery_2, ... ]
        }

    Each delivery is self-contained — video, trajectory, bounce, bowling
    analytics, shot classification and quality state — so a client does
    `deliveries[currentIndex]` for Previous/Next navigation and never has to
    reassemble a delivery from two internal objects.

    The segmenter evidence, purity validation verdicts, provenance, speed
    samples, clip frame maps and filesystem paths stay in `result.json`, which
    is not served over HTTP.
    """
    return serialize_result_document(result_or_404(analysis_id))
