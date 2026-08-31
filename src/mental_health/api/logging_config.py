"""Structured (JSON) logging setup — Phase 10 (Monitoring).

Plain-text logs are fine to read in a dev terminal, but hard to grep or
aggregate once this API runs anywhere else (Docker logs, and eventually a
log aggregator in Phase 13). This configures the root logger to emit one
JSON object per line instead, with a small set of fixed fields plus
whatever `extra=` a call site adds.

This does NOT change what gets logged, only how it's formatted — call
sites (main.py) are still solely responsible for never passing raw
request text into `extra`. See main.py's `predict()` for the one call
site that matters for privacy.
"""
from __future__ import annotations

import json
import logging
import sys

# Every attribute a stock LogRecord carries, so we can tell "extra" fields
# (added by a call site via `logger.info(..., extra={...})`) apart from the
# record's own built-in attributes.
_RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()) | {"message", "asctime"}


class JSONFormatter(logging.Formatter):
    """Renders each LogRecord as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        extras = {key: value for key, value in vars(record).items() if key not in _RESERVED}
        payload.update(extras)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Point the root logger at a single JSON-formatted stdout handler.

    Safe to call more than once (e.g. once at import time, and again if a
    test needs to reset it) — it always replaces the handler list rather
    than appending, so logs are never duplicated.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
