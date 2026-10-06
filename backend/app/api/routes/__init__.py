"""
routes — The HTTP surface, split by what a caller is trying to do.

  system.py      health, and the self-describing reference document.
  analysis.py    start an analysis, follow its progress, fetch the result.
  deliveries.py  one delivery per request.
  media.py       clip and overlay bytes.

`router` is the aggregate mounted by `main.py`. The legacy plural paths
(`/api/analyses/...`, `/api/upload`) are preserved as aliases of the documented
singular ones throughout — see docs/API_CONTRACT.md — so both spellings call the
same handler and therefore cannot drift in behaviour.
"""

from fastapi import APIRouter

from . import analysis, deliveries, media, system

router = APIRouter()
router.include_router(system.router)
router.include_router(analysis.router)
router.include_router(deliveries.router)
router.include_router(media.router)

__all__ = ["router"]