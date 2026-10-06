"""
storage — Durable JSON on disk.

`persistence.py` holds the only three filesystem-write primitives in the app:
`atomic_write_json`, `read_json`, `update_json_inplace`. Keeping them together is
what makes a partial write impossible to introduce: every persisted artifact goes
through the same temp-file-then-rename path.

Deliberately dumb. It does not know what an analysis or a delivery is — the
schema lives in `schemas/tracking.py`, and the paths live in `core/config.py`.
"""

from .persistence import atomic_write_json, read_json, update_json_inplace

__all__ = ["atomic_write_json", "read_json", "update_json_inplace"]