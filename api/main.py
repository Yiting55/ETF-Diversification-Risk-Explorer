"""Pre-trade diversification check as an HTTP API, for a broker or robo-advisor to call before an order.

Run:   uvicorn api.main:app --reload
Docs:  http://127.0.0.1:8000/docs
"""
import sys
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.data import PriceStore  # noqa: E402
from engine.scenarios import LOOKBACK_CHOICES  # noqa: E402
from engine.whatif import check_purchase, purchase_record, purchase_summary  # noqa: E402

app = FastAPI(
    title="Pre-trade diversification check",
    version="0.1.0",
    description="Shows what a planned purchase does to a portfolio's diversification over its historical "
                "lookback and in past crises, and compares it with putting the same amount elsewhere. "
                "Historical comparison only; not investment advice.",
)


class CheckRequest(BaseModel):
    holdings: dict[str, float] = Field(..., description="Ticker to weight or dollar value of current holdings.",
                                       examples=[{"VOO": 30, "QQQ": 25, "VTI": 15, "NVDA": 10, "AGG": 10, "GLD": 10}])
    ticker: str = Field(..., description="Ticker the client plans to buy.", examples=["QQQ"])
    amount: float = Field(..., gt=0, description="Purchase amount in US$.", examples=[2000])
    portfolio_value: float = Field(..., gt=0, description="Current portfolio value in US$.", examples=[10000])
    lookback_years: int = Field(5, description="Historical lookback: 3, 5 or 7 years.")


@lru_cache(maxsize=1)
def get_store() -> PriceStore:
    return PriceStore()


def _run(req: CheckRequest, store: PriceStore):
    if req.lookback_years not in LOOKBACK_CHOICES:
        raise HTTPException(422, f"lookback_years must be one of {list(LOOKBACK_CHOICES)}.")
    try:
        return check_purchase(req.holdings, req.ticker, req.amount, req.portfolio_value,
                              lookback_years=req.lookback_years, store=store)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/pretrade-check")
def pretrade_check(req: CheckRequest, store: PriceStore = Depends(get_store)) -> dict:
    """Impact of the planned purchase in each period, plus the best alternative for the same amount."""
    return purchase_summary(_run(req, store))


@app.post("/v1/pretrade-check/record", response_class=PlainTextResponse)
def pretrade_record(req: CheckRequest, store: PriceStore = Depends(get_store)) -> str:
    """Plain-text record of the check (inputs, impact and alternatives considered) for the client file."""
    return purchase_record(_run(req, store))
