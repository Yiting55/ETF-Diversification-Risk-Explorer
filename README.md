# Diversification X-Ray

IS4228 technical-track proof of concept (category: consumer fintech application).

Enter a portfolio of US stocks and ETFs and see how diversified it really is: how many
independent bets it behaves like, how much risk diversification removes, which holdings drive
the risk, and how all of that changed in the 2020 COVID crash and the 2022 inflation /
Russia-Ukraine sell-off.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

### Build the data (about 11 minutes, once)

```bash
python scripts/fetch_universe.py
```

This downloads every US-listed security from the Nasdaq Trader symbol directory, pulls daily
price history from Yahoo Finance, and filters it to:

* no leveraged / inverse / buffer / volatility funds or ETNs, no warrants, preferreds or units
* at least 7 years of history (covers the 2020 and 2022 stress windows)
* median daily dollar volume of at least US$1M over the last year; stocks priced at least $5

Result (Oct 2026): 967 ETFs (869 distinct exposures after merging near-duplicates) and 2,537
stocks. Outputs go to `data/universe/` (lists, plus an audit file with the reason every
excluded ticker was dropped) and `data/processed/` (daily and weekly adjusted closes).
Raw downloads in `data/raw/` are git-ignored. Re-running resumes from the last finished batch.

## Run

```bash
streamlit run app/streamlit_app.py
```

Tickers outside the filtered list (e.g. funds younger than 7 years such as SGOV) can still be
typed in; they are downloaded on first use and cached in `data/cache/`.

The **order ticket** page (sidebar navigation) shows the pre-trade check as it would appear in a
broker's order flow. It uses the holdings entered on the main page.

### Pre-trade check API

```bash
uvicorn api.main:app --reload       # docs at http://127.0.0.1:8000/docs
```

`POST /v1/pretrade-check` takes current holdings, the ticker to buy, the amount and the portfolio
value, and returns the effect on equivalent bets and worst fall in each period plus the best
alternative for the same amount. `POST /v1/pretrade-check/record` returns the same check as a
plain-text record.

### Validation

```bash
python validation/validate.py       # about a minute; writes validation/results/
```

Out-of-sample checks using only data from the five years before each crisis: whether a higher
score meant a smaller crisis fall, and whether the What-if top pick softened the fall.

## Tests

```bash
python -m pytest
```

Unit tests check the metrics against cases with known answers (uncorrelated assets = N bets,
perfectly correlated = 1 bet, risk shares sum to 100%, hedges exceed N). Integration tests
check the stress windows against the S&P 500's known peak-to-trough falls.

## Structure

```
app/streamlit_app.py      front end
engine/                   calculation engine (no Streamlit dependency, reusable behind an API)
  data.py                 price store: local parquet + on-demand Yahoo download with cache
  metrics.py              diversification ratio, effective bets, risk shares, covariance estimation
  scenarios.py            normal lookback and stress windows
  analyze.py              portfolio -> results, with date alignment and exclusion rules
scripts/fetch_universe.py data pipeline
api/main.py               pre-trade check API
validation/validate.py    out-of-sample checks
tests/                    pytest suite
```

## Method

### Correlation views

* **Historical overview:** rolling 60 / 120 / 252-trading-day correlation over the last
  seven years where available. Default is the unweighted mean of all distinct
  holding pairs; select a pair to inspect a specific relationship. Fixed holdings
  and complete return windows prevent changes in pair membership; missing or
  constant data leaves gaps. COVID and 2022 stress periods are shaded.
* **Stress test:** Pre-period baseline / Stress period / Change heatmaps use identical holdings with
  at least 95% price coverage and nonconstant returns in both periods. Pre-period baseline is
  the year immediately preceding the selected period, not necessarily crisis-free.
  Dates, return counts, exclusions and short-sample shrinkage are shown. Change is
  stress minus baseline, on a fixed -2 to +2 scale. The average-correlation metric
  uses this matched set; existing portfolio risk and return metrics retain the
  scenario's holdings. Falling together over a period does not necessarily mean
  high daily return correlation.

All metrics come from the diversification ratio DR = (sum of w_i * sigma_i) / sigma_p:

| Shown to user | Formula |
|---|---|
| Equivalent bets (volatility-based) | DR^2 |
| Risk removed by diversification | 1 - 1/DR |
| Holdings by weight | 1 / sum(w_i^2) |
| Share of risk per holding | w_i (Sigma w)_i / sigma_p^2 |
| Bad-year loss | 1.645 * sigma_p * portfolio value |

The score details compare DR² with a zero-correlation baseline, holding weights and
volatilities fixed: `(sum(w_i * sigma_i))^2 / sum((w_i * sigma_i)^2)`. A collapsed
section also compares 3 / 5 / 7-year scores and their actual dates. These overlapping
windows are a sensitivity check, not a confidence interval. DR² is an equivalent-bets
measure, not a literal count of independent risk factors; hedges can lift it above
the holding count. DR follows Choueifaty & Coignard (2008), *Toward Maximum Diversification*.

* **Historical lookback:** last 5 years of daily returns by default (3 / 5 / 7 selectable), including sell-offs.
* **Stress windows:** S&P 500 peak close to trough close, so 19 Feb - 23 Mar 2020 (COVID) and
  3 Jan - 12 Oct 2022 (inflation shock and Russia-Ukraine war); plus a custom date range.
* **Small samples:** with fewer than 3 trading days per holding, correlations are shrunk toward
  their average (Ledoit & Wolf 2004, constant-correlation target) and flagged low confidence.

## Known limitations

* Survivorship bias: only currently listed tickers are included.
* Yahoo Finance data is unofficial and fine for a proof of concept; a production version would
  license a data feed.
* The bad-year figure assumes normally distributed returns, which understates tail risk.
