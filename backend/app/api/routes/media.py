"""
media.py — Serving the bytes behind a delivery clip.

One route, by filename, because that is what `video.clip_url` in the result
envelope points at:

    GET /api/analysis/{analysis_id}/clips/{name}

A clip that was never written is a 404 rather than a copy of another file, and
the name is validated before it touches the filesystem so a crafted filename
cannot walk out of the clips directory.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...core.config import Paths
from ..helpers import serve_file

router = APIRouter()


@router.get("/analysis/{analysis_id}/clips/{name}")
def serve_clip(analysis_id: str, name: str) -> FileResponse:
    """The delivery clip as a playable video, with byte-range support."""
    if not name.startswith("delivery_") or not name.endswith(".mp4"):
        raise HTTPException(400, "Invalid clip name")
    if any(sep in name for sep in ("/", "\\", "..")):
        raise HTTPException(400, "Invalid clip name")
    return serve_file(
        Paths(analysis_id=analysis_id).clips_dir / name, "video/mp4", "Clip"
    )
