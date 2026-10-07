"""
routes — The HTTP surface, split by what a caller is trying to do.

  system.py      /api/health
  analysis.py    /api/analyze, /api/upload, /api/analysis/{id}/status,
                 /api/analysis/{id}/result
  media.py       /api/analysis/{id}/clips/{name}

`router` is the aggregate mounted by `main.py`. Six routes in total; see
docs/API_CONTRACT.md. There is one spelling per route — no legacy aliases — so a
route either exists as documented or does not exist at all.
"""

from fastapi import APIRouter

from . import analysis, media, system

router = APIRouter()
router.include_router(system.router)
router.include_router(analysis.router)
router.include_router(media.router)

__all__ = ["router"]
