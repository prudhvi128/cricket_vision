"""
io.py — Crash-safe JSON persistence.

Every artifact this service produces is written atomically: serialise to a
temporary file in the SAME directory, flush it to disk, then `os.replace` it
over the target.

`os.replace` is atomic on both NTFS and POSIX, so a reader either sees the
previous complete file or the new complete file — never a truncated one. A
plain `open(path, "w")` followed by `json.dump` leaves a half-written file if
the process dies mid-write, and `tracking.json` is the record the whole system
treats as truth, so a torn write is not an acceptable failure mode.

The temporary file is created in the destination directory rather than in the
system temp dir so the rename never crosses a filesystem boundary, which would
silently downgrade it to a non-atomic copy.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def atomic_write_json(path: Path | str, payload: Any, indent: int | None = 2) -> Path:
    """
    Write *payload* as JSON to *path* atomically. Returns the path.

    Raises on serialisation failure without touching the existing file — a
    half-written artifact is worse than a stale one, because a stale one can be
    detected by comparing timestamps while a truncated one cannot.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, ensure_ascii=False)
            fh.flush()
            # Force the bytes out of the OS cache so the rename cannot expose a
            # file whose contents have not actually landed.
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def read_json(path: Path | str) -> Any:
    """Read JSON, returning None if the file is absent or unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.error("Failed to read %s: %s", path, exc)
        return None


def update_json_inplace(path: Path | str, mutator) -> Any:
    """
    Read-modify-write a JSON document atomically.

    *mutator* receives the current document (or None) and returns the new one.
    Used to append a delivery to `tracking.json` after each delivery is
    finalised, so a crash mid-analysis still leaves every completed delivery
    recoverable instead of losing the whole run.
    """
    current = read_json(path)
    updated = mutator(current)
    if updated is None:
        return None
    atomic_write_json(path, updated)
    return updated