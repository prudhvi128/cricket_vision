"""
conftest.py — test session setup.

Two jobs, both of which the suite used to leave to chance.

1. Keep generated output out of the source tree. The tests drive the real
   pipeline, which writes a complete analysis directory — clips, overlays,
   tracking.json, result.json — under CRICKET_RUNS_DIR. Before this file existed
   that was `data/runs/`, so every `pytest` invocation left several analysis
   directories of real output in the source tree.

   The redirect below has to happen before anything imports `app.core.config`,
   because that module reads the environment once at import time and freezes the
   resolved paths as module-level constants. Hence the import of the environment
   hook before the test collection imports anything from `app`.

2. Put the import root on sys.path. `backend/` is the import root for both
   `app.*` and `tests.*`, so tests must otherwise be run with `backend/` as the
   working directory. Doing it here means `pytest` works from the project root as
   well.
"""

import os
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

for _p in (BACKEND_DIR, PROJECT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Redirect runtime output before app.core.config is imported anywhere. The
# datasets and models stay where they are — only generated output moves.
_RUNS_DIR = Path(tempfile.mkdtemp(prefix="cricket_test_runs_"))
os.environ["CRICKET_RUNS_DIR"] = str(_RUNS_DIR)
os.environ.setdefault("CRICKET_UPLOADS_DIR", str(_RUNS_DIR / "uploads"))