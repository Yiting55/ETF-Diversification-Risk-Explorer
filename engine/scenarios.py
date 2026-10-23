"""Time windows the engine measures diversification over.

Normal markets use a trailing lookback (default 5 years of daily returns: long enough
for a stable correlation estimate, short enough to reflect today's market structure).

Stress windows run from the S&P 500's peak close to its trough close, so they capture
exactly the stretch where diversification was most needed.
"""
from dataclasses import dataclass

import pandas as pd

LOOKBACK_CHOICES = (3, 5, 7)   # years; 7 is the maximum every filtered ticker supports
DEFAULT_LOOKBACK = 5


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    description: str


STRESS_SCENARIOS = {
    "covid": Scenario(
        key="covid",
        label="2020 COVID-19 crash",
        start=pd.Timestamp("2020-02-19"),
        end=pd.Timestamp("2020-03-23"),
        description="S&P 500 peak to trough: -34% in 23 trading days, the fastest bear market "
                    "on record. Almost every risky asset fell together.",
    ),
    "2022": Scenario(
        key="2022",
        label="2022 inflation shock & Russia-Ukraine war",
        start=pd.Timestamp("2022-01-03"),
        end=pd.Timestamp("2022-10-12"),
        description="S&P 500 peak to trough: -25%. Russia invaded Ukraine on 24 Feb, energy "
                    "prices spiked and the Fed hiked rates; stocks and bonds fell together.",
    ),
}


def normal_window(as_of: pd.Timestamp, years: int) -> Scenario:
    return Scenario(
        key="normal",
        label=f"Historical overview (last {years} years)",
        start=as_of - pd.DateOffset(years=years),
        end=as_of,
        description=f"All trading days in the {years} years to {as_of:%d %b %Y}.",
    )


def custom_window(start, end) -> Scenario:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    return Scenario("custom", f"Custom ({start:%d %b %Y} - {end:%d %b %Y})", start, end,
                    "User-defined window.")
