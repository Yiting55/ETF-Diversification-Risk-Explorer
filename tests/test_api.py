"""Tests for the pre-trade check API (skipped if FastAPI is not installed)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app, get_store  # noqa: E402
from tests.test_whatif import FakeStore  # noqa: E402

app.dependency_overrides[get_store] = lambda: FakeStore()
client = TestClient(app)
BODY = {"holdings": {"VOO": 60, "QQQ": 40}, "ticker": "TLT", "amount": 2500, "portfolio_value": 10000}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_pretrade_check_returns_all_periods():
    r = client.post("/v1/pretrade-check", json=BODY)
    assert r.status_code == 200
    data = r.json()
    assert data["ticker"] == "TLT"
    assert data["share_of_portfolio_after"] == pytest.approx(0.2)
    assert set(data["periods"]) == {"normal", "covid", "2022"}
    assert data["periods"]["normal"]["bets_after"] > data["periods"]["normal"]["bets_before"]
    assert data["best_alternative"]["ticker"] != "TLT"


def test_record_endpoint_is_plain_text():
    r = client.post("/v1/pretrade-check/record", json=BODY)
    assert r.status_code == 200 and "PRE-TRADE DIVERSIFICATION CHECK" in r.text


def test_bad_inputs_return_422():
    assert client.post("/v1/pretrade-check", json={**BODY, "ticker": "NOPE"}).status_code == 422
    assert client.post("/v1/pretrade-check", json={**BODY, "lookback_years": 4}).status_code == 422
    assert client.post("/v1/pretrade-check", json={**BODY, "amount": -5}).status_code == 422
