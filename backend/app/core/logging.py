"""
logging.py — Logging setup, in one place.

Every module in the app uses `logging.getLogger(__name__)`, which is the correct
default and needs no per-module configuration. What needs to happen exactly once,
at startup, is deciding the format and level. Keeping that here means a script,
the ASGI app, and a test all produce the same log lines, and it means the answer
to "how do I turn on debug logging" is one environment variable rather than a
search for `basicConfig`.
"""

from __future__ import annotations

import logging
import os

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: int | str | None = None) -> None:
    """
    Configure root logging. Safe to call more than once.

    `CRICKET_LOG_LEVEL` overrides the default of INFO, which is what a developer
    reaches for first when a delivery is not being detected; the level names
    (`DEBUG`, `INFO`, ...) and numeric forms both work.
    """
    if level is None:
        level = os.environ.get("CRICKET_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.strip().upper(), logging.INFO)
    logging.basicConfig(level=level, format=LOG_FORMAT)


__all__ = ["LOG_FORMAT", "configure_logging"]