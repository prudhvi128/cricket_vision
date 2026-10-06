"""
helpers.py — Request parsing and result lookup shared by the route modules.

These were private functions inside a single 660-line `endpoints.py`. They are
here because three different route modules need them, and because they are the
places where the API's *contract* is actually decided:

  * how a delivery id is parsed (`3` and `delivery_003` are both accepted)
  * when a missing result is a 404 and when it is a 409
  * what a rejected upload is told, and what is cleaned up before answering
  * how a media file is served with byte-range support

Nothing here performs analysis. It validates, persists, and translates errors
into the documented HTTP shapes.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from fastapi import HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..core.config import ALLOWED_VIDEO_EXTENSIONS, Paths
from ..storage.persistence import read_json
from ..video.video_io import is_readable_video
from .deps import get_jobs, get_models

log = logging.getLogger(__name__)

# Uploads are moved into the analysis directory rather than streamed through
# memory. 2 GB ceiling is generous for a broadcast segment and bounded so a
# single upload cannot fill the disk.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

_DELIVERY_REF = re.compile(r"^(?:delivery_)?(\d{1,6})$")


def parse_delivery_id(raw: str) -> int:
    """
    Accept either `3` or `delivery_003` and return the integer id.

    Clients index deliveries positionally while the media files are named
    `delivery_003.mp4`, so both forms have to work in the same route. Anything else
    is a 404-shaped client error rather than a silent coercion.
    """
    if isinstance(raw, int):  # FastAPI path params arrive as str; be forgiving
        return raw
    m = _DELIVERY_REF.match(str(raw).strip())
    if not m:
        raise HTTPException(
            400,
            f"Malformed delivery id '{raw}'. Use the integer id or 'delivery_003'.",
        )
    return int(m.group(1))


async def receive_upload(file: UploadFile, field_name: str) -> dict:
    """
    Validate, persist and enqueue one upload. Shared by both route spellings.

    Returns the 202-shaped body. Never blocks on processing: the work happens on a
    background thread and the caller gets an `analysis_id` to poll.
    """
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type '{suffix or '(none)'}'. "
            f"Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    jobs = get_jobs()
    models = get_models()
    job = jobs.create(source_name=filename)

    paths = Paths(analysis_id=job.analysis_id).ensure()
    # Single flat name inside the analysis dir: the original filename is kept in
    # the job record rather than in the path, so a crafted name cannot escape.
    dest = paths.root / ("source" + suffix)

    written = 0
    try:
        with open(dest, "wb") as fh:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    fh.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, "Upload exceeds the 2 GB limit")
                fh.write(chunk)
                # Real progress on a real unit: bytes actually on disk.
                jobs.progress_cb(job.analysis_id)("uploading", written, 0, "receiving")
    except HTTPException:
        shutil.rmtree(paths.root, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(paths.root, ignore_errors=True)
        raise
    finally:
        await file.close()

    if written == 0:
        shutil.rmtree(paths.root, ignore_errors=True)
        raise HTTPException(400, "Uploaded file was empty")

    if not is_readable_video(dest):
        shutil.rmtree(paths.root, ignore_errors=True)
        raise HTTPException(
            400,
            "The uploaded file could not be opened as a video. It has the right "
            "extension but is not decodable.",
        )

    jobs.progress_cb(job.analysis_id)(
        "uploading", written, written, "upload complete"
    )
    jobs.start(job, dest, models)

    return {
        "analysis_id": job.analysis_id,
        "status": job.status,
        "progress": job.percent,
        "stage": job.stage,
        "bytes": written,
        # The documented routes, so a client does not have to assemble them.
        "status_url": f"/api/analysis/{job.analysis_id}/status",
        "deliveries_url": f"/api/analysis/{job.analysis_id}/deliveries",
        "result_url": f"/api/analysis/{job.analysis_id}/result",
        "progress_url": f"/api/analysis/{job.analysis_id}/progress",
        "cancel_url": f"/api/analysis/{job.analysis_id}/cancel",
    }


def result_or_404(analysis_id: str) -> dict:
    """
    Load a persisted result document, or explain precisely why it is not there.

    A still-running analysis is a 409, not a 404: the distinction is the
    difference between "come back later" and "this never happened".
    """
    doc = read_json(Paths(analysis_id=analysis_id).result_json)
    if doc is None:
        job = get_jobs().get(analysis_id)
        if job is not None and job.status in ("queued", "running"):
            raise HTTPException(
                409,
                f"Analysis {analysis_id} is still {job.status}; results are not "
                "written yet.",
            )
        raise HTTPException(
            404,
            f"No result for analysis {analysis_id}. It may have failed, or the id "
            "may be wrong.",
        )
    return doc


def find_delivery(doc: dict, delivery_id: int, analysis_id: str) -> dict:
    """
    Look up one delivery inside a result document.

    Absent means quarantined by purity validation or never produced — the two are
    not distinguishable from the outside, so the message must not pretend they
    are. Delivery ids are never renumbered.
    """
    for d in doc.get("deliveries", []):
        if d.get("delivery_id") == delivery_id:
            return d
    raise HTTPException(
        404,
        f"Delivery {delivery_id} is not in analysis {analysis_id}. It was either "
        "quarantined by purity validation or never produced.",
    )


def serve_file(path: Path, media_type: str, what: str) -> FileResponse:
    if not path.is_file():
        raise HTTPException(404, f"{what} not found")
    # Range support matters: browsers request byte ranges when seeking in a
    # <video> element, and without it scrubbing an overlay silently fails.
    # Starlette's FileResponse handles Range itself.
    return FileResponse(path, media_type=media_type)


__all__ = [
    "MAX_UPLOAD_BYTES",
    "find_delivery",
    "parse_delivery_id",
    "receive_upload",
    "result_or_404",
    "serve_file",
]