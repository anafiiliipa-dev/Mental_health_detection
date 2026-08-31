"""Unit tests for src/mental_health/api/logging_config.py."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mental_health.api.logging_config import JSONFormatter, configure_logging


def _make_record(msg: str = "hello", extra: dict | None = None) -> logging.LogRecord:
    record = logging.LogRecord(
        name="mental_health.api.main",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=None,
        exc_info=None,
    )
    for key, value in (extra or {}).items():
        setattr(record, key, value)
    return record


class TestJSONFormatter:
    def test_output_is_valid_json(self):
        payload = json.loads(JSONFormatter().format(_make_record()))
        assert isinstance(payload, dict)

    def test_fixed_fields_are_present(self):
        payload = json.loads(JSONFormatter().format(_make_record("predict request")))
        assert payload["event"] == "predict request"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "mental_health.api.main"
        assert "timestamp" in payload

    def test_extra_fields_are_merged_into_the_payload(self):
        extra = {"fingerprint": "abc123", "latency_ms": 12.5, "is_demo_fallback": False}
        payload = json.loads(JSONFormatter().format(_make_record(extra=extra)))
        assert payload["fingerprint"] == "abc123"
        assert payload["latency_ms"] == 12.5
        assert payload["is_demo_fallback"] is False

    def test_builtin_record_attributes_are_not_leaked_as_extras(self):
        # Only the fixed fields + genuine `extra=` values should appear —
        # never internal LogRecord bookkeeping like `pathname` or `args`.
        payload = json.loads(JSONFormatter().format(_make_record()))
        assert "pathname" not in payload
        assert "args" not in payload
        assert "msg" not in payload

    def test_exception_info_is_formatted_as_text(self):
        try:
            raise ValueError("boom")
        except ValueError:
            exc_info = sys.exc_info()
        record = _make_record()
        record.exc_info = exc_info
        payload = json.loads(JSONFormatter().format(record))
        assert "ValueError: boom" in payload["exc_info"]

    def test_never_includes_a_raw_text_field(self):
        # Privacy invariant: nothing named after request text should ever
        # come out of the formatter, even if a call site made a mistake —
        # this test documents the expectation, it doesn't enforce it at
        # the formatter level (call sites are responsible; see main.py).
        extra = {"fingerprint": "abc123", "text_length": 42}
        payload = json.loads(JSONFormatter().format(_make_record(extra=extra)))
        assert "text" not in payload


class TestConfigureLogging:
    def test_root_logger_gets_exactly_one_handler(self):
        configure_logging()
        root = logging.getLogger()
        assert len(root.handlers) == 1

    def test_handler_uses_the_json_formatter(self):
        configure_logging()
        root = logging.getLogger()
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_calling_it_twice_does_not_duplicate_handlers(self):
        configure_logging()
        configure_logging()
        root = logging.getLogger()
        assert len(root.handlers) == 1

    def test_sets_the_requested_level(self):
        configure_logging(level=logging.WARNING)
        assert logging.getLogger().level == logging.WARNING
        configure_logging(level=logging.INFO)  # restore default for other tests
