"""
Tests for src/ingest.py -- currently just _retry_transient (Phase 8's fix
for a real GitHub Actions failure: "Connection reset by peer" downloading
from nflverse-data's release CDN on a cold, cache-less CI checkout).
"""

import sys
import time
from pathlib import Path

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import _retry_transient  # noqa: E402


def _connection_error_with_404():
    response = requests.Response()
    response.status_code = 404
    http_404 = requests.exceptions.HTTPError(response=response)
    err = ConnectionError("Failed to download ...")
    err.__cause__ = http_404
    return err


def _connection_error_transient():
    return ConnectionError("('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))")


def test_retry_transient_succeeds_after_one_transient_failure(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)  # don't actually wait in tests

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _connection_error_transient()
        return "ok"

    result = _retry_transient(flaky, max_attempts=3, backoff_seconds=0.01)
    assert result == "ok"
    assert calls["n"] == 2


def test_retry_transient_reraises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)

    calls = {"n": 0}

    def always_flaky():
        calls["n"] += 1
        raise _connection_error_transient()

    with pytest.raises(ConnectionError):
        _retry_transient(always_flaky, max_attempts=3, backoff_seconds=0.01)
    assert calls["n"] == 3  # every attempt used, no more


def test_retry_transient_does_not_retry_a_404():
    """
    An HTTP 404 means the resource genuinely doesn't exist (e.g. an
    unpublished season) -- retrying wastes time before failing anyway, and
    src/pipeline.py's _is_unpublished_season_error relies on the 404
    surfacing immediately, not after 3 rounds of backoff.
    """
    calls = {"n": 0}

    def not_found():
        calls["n"] += 1
        raise _connection_error_with_404()

    with pytest.raises(ConnectionError):
        _retry_transient(not_found, max_attempts=3, backoff_seconds=0.01)
    assert calls["n"] == 1  # no retries for a 404


def test_retry_transient_passes_args_and_kwargs_through():
    def add(a, b, c=0):
        return a + b + c

    assert _retry_transient(add, 1, 2, c=3) == 6
