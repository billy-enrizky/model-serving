"""Smoke tests for gateway. Run with MODEL_API_KEY set."""

import os

import httpx
import pytest


BASE_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:8443")
API_KEY = os.getenv("MODEL_API_KEY")

requires_live = pytest.mark.skipif(
    not API_KEY, reason="MODEL_API_KEY not set; gateway not live"
)


@requires_live
def test_healthz():
    r = httpx.get(f"{BASE_URL}/healthz", timeout=10.0)
    assert r.status_code in (200, 503)
    assert "gateway" in r.json()


@requires_live
def test_auth_required():
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json={"model": "x"}, timeout=10.0)
    assert r.status_code == 401


@requires_live
def test_models_listed():
    r = httpx.get(
        f"{BASE_URL}/v1/models",
        headers={"x-api-key": API_KEY},
        timeout=10.0,
    )
    assert r.status_code == 200
    data = r.json()
    assert any(m["id"] == "gemma-4-E2B-it" for m in data.get("data", []))
