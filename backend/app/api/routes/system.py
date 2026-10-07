"""
system.py — Liveness.

Deliberately tiny. `/api/health` is the one endpoint guaranteed to answer before
any analysis exists, so a client can check that the API is up and which schema
and pipeline version it is running.
"""

from __future__ import annotations

from fastapi import APIRouter

from ...core.constants import PIPELINE_STAGES, PIPELINE_VERSION, SCHEMA_VERSION
from ..deps import get_jobs

router = APIRouter()


@router.get("/health")
def health() -> dict:
    """
    Liveness plus the contract versions.

    `status` is the only field a load balancer needs. The versions are included
    because a client that renders a delivery record needs to know which schema it
    is looking at, and this is the one endpoint guaranteed to be reachable before
    any analysis exists.
    """
    jobs = get_jobs()
    return {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "analyses_tracked_in_memory": len(jobs.list()),
        # Published so a client can render a stage indicator without hardcoding
        # the vocabulary, and so a contract change is visible in a health check.
        "stages": list(PIPELINE_STAGES),
    }
