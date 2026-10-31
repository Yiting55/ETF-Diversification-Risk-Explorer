"""What-if analysis: effect of adding one more holding to a portfolio."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analyze import DAYS_PER_HOLDING, WindowResult, analyze_window, clean_holdings
from .data import PriceStore, load_universe, normalize_ticker
from .metrics import diversification_metrics, replay
from .scenarios import DEFAULT_LOOKBACK, STRESS_SCENARIOS, Scenario, normal_window

CANDIDATE_SETS: dict[str, dict[str, str]] = {
    "US": {
        "IEF": "7-10Y US Treasuries",
        "TLT": "20+Y US Treasuries",
        "GLD": "Gold",
        "VNQ": "US REITs",
        "DBC": "Broad commodities",
        "EEM": "Emerging-market equities",
        "VXUS": "International equities ex-US",
    },
}
MARKET_PROXY = {"US": "VT"}
RF_PROXY = {"US": "BIL"}
DEFAULT_REGION = "US"
MRP_DEFAULT = 0.05
ALPHA_GRID = tuple(round(0.05 * i, 2) for i in range(1, 9))
ALPHA_DEFAULT = 0.10
VERDICT_THRESHOLD = 0.01  # change in equivalent bets treated as "little change"
TRADING_DAYS = 252


def new_weights(w: pd.Series, candidate: str, alpha: float) -> pd.Series:
    """(1 - alpha) * w + alpha * e_k"""
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    out = (1.0 - alpha) * (w / w.sum())
    out.loc[candidate] = out.get(candidate, 0.0) + alpha
    return out


def capm_expected_returns(prices: pd.DataFrame, start, end, market: str, rf: float, mrp: float) -> pd.Series:
    """E(R_i) = rf + beta_i * MRP, beta from daily returns over [start, end]."""
    r = prices.loc[start:end].pct_change(fill_method=None).iloc[1:]
    m = r[market]
    betas = {}
    for t in r.columns:
        pair = pd.concat([r[t], m], axis=1).dropna()
        betas[t] = pair.iloc[:, 0].cov(pair.iloc[:, 1]) / pair.iloc[:, 1].var()
    beta = pd.Series(betas)
    return rf + beta * mrp


def rf_from_proxy(prices: pd.DataFrame, proxy: str) -> float:
    """Trailing 12-month return of a T-bill ETF."""
    p = prices[proxy].dropna()
    return float(p.iloc[-1] / p.iloc[-(TRADING_DAYS + 1)] - 1)


@dataclass
class WhatIf:
    grid: pd.DataFrame
    base: dict[str, WindowResult]
    candidates: dict[str, str]
    rf: float
    mrp: float
    unavailable: dict[str, str] = field(default_factory=dict)
    held: dict[str, str] = field(default_factory=dict)
    notes: dict[str, list[str]] = field(default_factory=dict)
    prices: pd.DataFrame | None = None
    holdings: pd.Series | None = None


@dataclass
class _Measured:
    tickers: list[str]
    weights: np.ndarray
    metrics: dict
    replay: dict
    low_confidence: bool


def _row(win_key, k, a, res, er: pd.Series | None) -> dict:
    m = res.metrics
    exp = float(pd.Series(res.weights, index=res.tickers).mul(er.reindex(res.tickers)).sum()) if er is not None else np.nan
    return {"window": win_key, "candidate": k, "alpha": a,
            "bets": m["effective_bets"], "vol": m["portfolio_vol"], "er": exp,
            "max_dd": res.replay["max_drawdown"], "low_confidence": res.low_confidence}


def run_whatif(holdings: dict[str, float], lookback_years: int = DEFAULT_LOOKBACK,
               region: str = DEFAULT_REGION, candidates: dict[str, str] | None = None,
               alphas=ALPHA_GRID, rf: float | None = None, mrp: float = MRP_DEFAULT,
               store: PriceStore | None = None, stress: list[Scenario] | None = None,
               value: float = 10_000) -> WhatIf:
    w = clean_holdings(holdings)
    candidates = dict(candidates or CANDIDATE_SETS[region])
    market, rf_proxy = MARKET_PROXY[region], RF_PROXY[region]
    store = store or PriceStore()

    tickers = list(dict.fromkeys(list(w.index) + list(candidates) + [market, rf_proxy]))
    prices, unknown = store.get(tickers)
    w = w.drop([t for t in unknown if t in w.index])
    if w.empty:
        raise ValueError("No price data for any holding.")
    w = w / w.sum()
    for t in unknown:
        candidates.pop(t, None)
    if market in unknown:
        raise ValueError(f"No price data for the market proxy {market}.")

    if rf is None:
        rf = rf_from_proxy(prices, rf_proxy) if rf_proxy not in unknown else 0.0

    as_of = prices.index.max()
    windows = [normal_window(as_of, lookback_years)] + list(STRESS_SCENARIOS.values() if stress is None else stress)

    base, rows, unavailable, notes = {}, [], {}, {}
    er = None
    for s in windows:
        align = s.key == "normal"
        b = analyze_window(prices, w, s, value, align_start=align)
        if b is None:
            continue
        base[s.key] = b
        notes[s.key] = []
        shrink = b.n_obs < DAYS_PER_HOLDING * len(b.tickers)
        if align:
            er = capm_expected_returns(prices, b.start, b.end, market, rf, mrp)
        er_s = er if align else None
        rows.append(_row(s.key, "(current)", 0.0, b, er_s))
        for k in candidates:
            r = analyze_window(prices, new_weights(w, k, alphas[0]), s, value, align_start=align, shrink=shrink)
            if r is None or k in r.excluded:
                unavailable[f"{s.key}:{k}"] = (r.excluded.get(k) if r else "not enough data")
                continue
            if r.start != b.start or r.end != b.end:
                notes[s.key].append(f"{k} is measured from {r.start:%d %b %Y} (its first price), "
                                    f"not {b.start:%d %b %Y}.")
            rows.append(_row(s.key, k, alphas[0], r, er_s))
            # The covariance and dates do not depend on the weight, so reuse them for the other weights.
            vols = r.metrics["asset_vols"]
            cov = r.corr.to_numpy() * np.outer(vols, vols)
            px = prices.loc[r.start:r.end, r.tickers].dropna()
            for a in alphas[1:]:
                wa = new_weights(w, k, a).reindex(r.tickers)
                wa = (wa / wa.sum()).to_numpy()
                m = diversification_metrics(wa, cov)
                rows.append(_row(s.key, k, a, _Measured(r.tickers, wa, m, replay(px, wa), r.low_confidence), er_s))

    if "normal" not in base:
        raise ValueError("Not enough overlapping price history to measure correlations.")
    return WhatIf(grid=pd.DataFrame(rows), base=base, candidates=candidates, rf=rf, mrp=mrp,
                  unavailable=unavailable, prices=prices, holdings=w,
                  held=held_status(w, candidates), notes=notes)


def held_status(w: pd.Series, candidates: dict[str, str]) -> dict[str, str]:
    """Candidates already held, or held through a near-duplicate fund."""
    out = {k: "already held" for k in candidates if k in w.index}
    try:
        uni = load_universe().set_index("yf_symbol")["dup_cluster"]
    except (FileNotFoundError, KeyError):
        return out
    clusters = {t: uni.get(t) for t in w.index}
    for k in candidates:
        if k in out or k not in uni.index:
            continue
        twins = [t for t, c in clusters.items() if c and c == uni[k] and t != k]
        if twins:
            out[k] = f"similar to {', '.join(twins)}"
    return out


def ranking_table(wi: WhatIf, window: str = "normal", alpha: float = ALPHA_DEFAULT,
                  rank_by: str = "d_bets") -> pd.DataFrame:
    g = wi.grid[wi.grid["window"] == window]
    cur = g[g["candidate"] == "(current)"].iloc[0]
    g = g[g["candidate"] != "(current)"]
    at = g[np.isclose(g["alpha"], alpha)].set_index("candidate")
    if at.empty:
        return pd.DataFrame()
    tbl = pd.DataFrame({
        "bets": at["bets"],
        "d_bets": at["bets"] - cur["bets"],
        "d_vol": at["vol"] - cur["vol"],
        "d_er": at["er"] - cur["er"],
        "d_max_dd": at["max_dd"] - cur["max_dd"],
    })
    y = "d_er" if window == "normal" else "d_max_dd"
    crises = crisis_effects(wi, alpha).reindex(tbl.index)
    keys = list(crises.columns.drop("all_crises"))
    for key in keys:
        tbl[f"dd_{key}"] = crises[key]
    tbl["all_crises"] = crises["all_crises"].fillna(False).astype(bool)
    tbl["worst_crisis"] = crises[keys].min(axis=1, skipna=False) if keys else np.nan
    tbl = mark_pareto(tbl, "d_bets", y).sort_values(rank_by, ascending=False, na_position="last")
    tbl["held"] = [wi.held.get(k, "") for k in tbl.index]
    tbl.insert(0, "rank", range(1, len(tbl) + 1))
    return tbl


def crisis_effects(wi: WhatIf, alpha: float) -> pd.DataFrame:
    """Change in worst fall in each stress window at weight alpha (> 0 = shallower).
    all_crises is True when the candidate made the fall shallower in every stress window."""
    out = {}
    for key in wi.base:
        if key == "normal":
            continue
        g = wi.grid[wi.grid["window"] == key]
        cur = g.loc[g["candidate"] == "(current)", "max_dd"].iloc[0]
        at = g[(g["candidate"] != "(current)") & np.isclose(g["alpha"], alpha)].set_index("candidate")["max_dd"]
        out[key] = at - cur
    df = pd.DataFrame(out, index=list(wi.candidates))
    df["all_crises"] = df.notna().all(axis=1) & (df > 0).all(axis=1) if len(out) else False
    return df


def mark_pareto(tbl: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Pareto-efficient on (x, y)."""
    pts = tbl[[x, y]].to_numpy()
    eff = [not any((q >= p).all() and (q > p).any() for j, q in enumerate(pts) if j != i)
           for i, p in enumerate(pts)]
    out = tbl.copy()
    out["efficient"] = eff
    return out


def ranking_stability(wi: WhatIf, alpha: float = ALPHA_DEFAULT, n_boot: int = 200, block: int = 21,
                      seed: int = 42) -> pd.Series:
    """Share of block-bootstrap resamples in which each candidate adds the most bets."""
    b = wi.base["normal"]
    cands = list(wi.candidates)
    cols = list(dict.fromkeys(list(wi.holdings.index) + cands))
    r = wi.prices.loc[b.start:b.end, cols].pct_change(fill_method=None).iloc[1:].dropna().to_numpy()
    n = len(r)
    rng = np.random.default_rng(seed)
    idx_of = {t: i for i, t in enumerate(cols)}
    w0 = wi.holdings.reindex(cols).fillna(0.0).to_numpy()
    wins = pd.Series(0.0, index=cands)
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        sample = r[np.concatenate([np.arange(s, s + block) for s in starts])[:n]]
        cov = np.cov(sample, rowvar=False)
        base = diversification_metrics(w0, cov)["effective_bets"]
        gains = {}
        for k in cands:
            wk = (1 - alpha) * w0
            wk[idx_of[k]] += alpha
            gains[k] = diversification_metrics(wk, cov)["effective_bets"] - base
        wins[max(gains, key=gains.get)] += 1
    return (wins / n_boot).sort_values(ascending=False)


def summary_sentence(wi: WhatIf, tbl: pd.DataFrame, window: str, alpha: float,
                     rank_by: str = "d_bets", amount: float | None = None) -> str:
    """One-line reading of the top-ranked candidate. With `amount`, phrased as a purchase of new money."""
    if tbl.empty:
        return "No candidate could be tested in this period."
    k, row = tbl.index[0], tbl.iloc[0]
    cur = wi.base[window].metrics
    name = wi.candidates.get(k, k)
    held = f" (you {wi.held[k].replace('already held', 'already hold it')})" if k in wi.held else ""
    move = (f"Putting ${amount:,.0f} into {k} ({name}){held}" if amount
            else f"Moving {alpha:.0%} of the portfolio into {k} ({name}){held}")
    size = "this amount" if amount else "this weight"
    bets = f"equivalent bets go from {cur['effective_bets']:.2f} to {row['bets']:.2f}"

    if rank_by == "worst_crisis":
        worst = row["worst_crisis"]
        if not worst > 0:
            return f"At {size}, none of the candidate ETFs would have softened every past crisis tested."
        return (f"{move} would have made the worst fall shallower in every crisis tested, by at least "
                f"{worst * 100:.1f} pts; {bets}.")

    if rank_by == "d_max_dd":
        if row["d_max_dd"] <= 0.001:
            return f"At {size}, none of the candidate ETFs would have made the worst fall in this period shallower."
        return (f"{move} would have made the worst fall in this period {row['d_max_dd'] * 100:.1f} pts "
                f"shallower; {bets}.")

    if row["d_bets"] <= 0.01:
        return f"At {size}, none of the candidate ETFs would meaningfully add equivalent bets."
    if window == "normal":
        cost = row["d_er"] * 100
        tail = (f" and volatility changes by {row['d_vol'] * 100:+.1f} pts, "
                + (f"at an estimated cost of {abs(cost):.2f} pts of expected return a year (CAPM)."
                   if cost < 0 else f"with expected return changing by {cost:+.2f} pts a year (CAPM)."))
        missed = [STRESS_SCENARIOS[c[3:]].label for c in tbl.columns
                  if c.startswith("dd_") and c[3:] in STRESS_SCENARIOS
                  and not row[c] > 0]
        if missed:
            tail += f" It would not have softened the {' or the '.join(missed)}."
    else:
        dd = row["d_max_dd"] * 100
        tail = (f"; the worst fall in this period would have been {abs(dd):.1f} pts "
                f"{'shallower' if dd > 0 else 'deeper'}.")
    return f"{move}: {bets}{tail}"


def purchase_alpha(value: float, amount: float) -> float:
    """Buying A with new money on a portfolio worth V: alpha = A / (V + A)."""
    if value <= 0 or amount <= 0:
        raise ValueError("Portfolio value and purchase amount must be positive.")
    return amount / (value + amount)


def ticker_name(ticker: str) -> str:
    try:
        uni = load_universe().set_index("yf_symbol")["name"]
        return str(uni.get(ticker, ticker)) or ticker
    except (FileNotFoundError, KeyError):
        return ticker


@dataclass
class PurchaseCheck:
    ticker: str
    name: str
    amount: float
    value: float
    alpha: float
    impact: pd.DataFrame
    wi: WhatIf


def check_purchase(holdings: dict[str, float], ticker: str, amount: float, value: float = 10_000,
                   lookback_years: int = DEFAULT_LOOKBACK, region: str = DEFAULT_REGION,
                   rf: float | None = None, mrp: float = MRP_DEFAULT, store: PriceStore | None = None,
                   alternatives: dict[str, str] | None = None) -> PurchaseCheck:
    ticker = normalize_ticker(ticker)
    if not ticker:
        raise ValueError("Enter a ticker to check.")
    alpha = purchase_alpha(value, amount)
    name = ticker_name(ticker)
    cands = {ticker: name, **{k: v for k, v in (alternatives or CANDIDATE_SETS[region]).items() if k != ticker}}
    wi = run_whatif(holdings, lookback_years=lookback_years, region=region, candidates=cands,
                    alphas=(alpha,), rf=rf, mrp=mrp, store=store, value=value)
    if ticker not in wi.candidates:
        raise ValueError(f"No price data found for {ticker}.")
    return purchase_from(wi, ticker, amount, value)


def purchase_from(wi: WhatIf, ticker: str, amount: float, value: float) -> PurchaseCheck:
    """Before/after impact of buying `amount` of `ticker`, read from a What-if run that includes
    that candidate at weight amount / (value + amount)."""
    alpha = purchase_alpha(value, amount)
    rows = []
    for key, b in wi.base.items():
        g = wi.grid[(wi.grid.window == key)]
        after = g[(g.candidate == ticker) & np.isclose(g.alpha, alpha)]
        cur = g[g.candidate == "(current)"].iloc[0]
        a = after.iloc[0] if not after.empty else None
        rows.append({
            "window": key, "label": b.scenario.label,
            "bets_before": cur["bets"], "bets_after": a["bets"] if a is not None else np.nan,
            "vol_before": cur["vol"], "vol_after": a["vol"] if a is not None else np.nan,
            "er_before": cur["er"], "er_after": a["er"] if a is not None else np.nan,
            "dd_before": cur["max_dd"], "dd_after": a["max_dd"] if a is not None else np.nan,
            "loss_before": cur["max_dd"] * value,
            "loss_after": a["max_dd"] * (value + amount) if a is not None else np.nan,
        })
    return PurchaseCheck(ticker=ticker, name=wi.candidates.get(ticker, ticker), amount=amount, value=value,
                         alpha=alpha, impact=pd.DataFrame(rows).set_index("window"), wi=wi)


def purchase_record(pc: PurchaseCheck, as_of: str | None = None) -> str:
    """Plain-text record of a pre-trade check: inputs, impact and the alternatives considered."""
    wi, n = pc.wi, pc.wi.base["normal"]
    labels = {"normal": "Historical lookback", **{k: v.label for k, v in STRESS_SCENARIOS.items()}}
    lines = [
        "PRE-TRADE DIVERSIFICATION CHECK",
        f"Generated: {as_of or f'{pd.Timestamp.now():%Y-%m-%d %H:%M}'}",
        f"Price data to: {wi.prices.index.max():%Y-%m-%d}",
        "",
        "PORTFOLIO",
        f"Value before purchase: ${pc.value:,.0f}",
        *[f"  {t:<8}{w:>7.1%}" for t, w in wi.holdings.sort_values(ascending=False).items()],
        "",
        "PLANNED PURCHASE",
        f"{pc.ticker} ({pc.name}): ${pc.amount:,.0f}, {pc.alpha:.1%} of the portfolio after purchase",
        *([f"Note: {wi.held[pc.ticker]}"] if pc.ticker in wi.held else []),
        "",
        "IMPACT",
        f"{'Period':<46}{'Equiv. bets':>14}{'Worst fall':>18}{'Worst fall (US$)':>22}",
    ]
    for key, r in pc.impact.iterrows():
        if np.isnan(r["bets_after"]):
            lines.append(f"{labels.get(key, key):<46}{'n/a':>14}")
            continue
        lines.append(f"{labels.get(key, key):<46}"
                     f"{r['bets_before']:>6.2f} -> {r['bets_after']:<5.2f}"
                     f"{r['dd_before']:>8.1%} -> {r['dd_after']:<7.1%}"
                     f"{r['loss_before']:>10,.0f} -> {r['loss_after']:<9,.0f}")
    tbl = ranking_table(wi, "normal", pc.alpha)
    crisis_cols = [c for c in tbl.columns if c.startswith("dd_") and c[3:] in STRESS_SCENARIOS]
    lines += ["", f"ALTERNATIVES CONSIDERED (same ${pc.amount:,.0f})",
              f"{'Ticker':<8}{'Equiv. bets':>12}{'Change':>9}"
              + "".join(f"{'Fall ' + c[3:].upper():>13}" for c in crisis_cols) + f"{'Both crises':>13}"]
    for k, r in tbl.iterrows():
        mark = " *" if k == pc.ticker else ""
        lines.append(f"{k + mark:<8}{r['bets']:>12.2f}{r['d_bets']:>+9.2f}"
                     + "".join(f"{_pts(r[c]):>13}" for c in crisis_cols)
                     + f"{('yes' if r['all_crises'] else 'no'):>13}")
    lines += [
        "* planned purchase. Fall = change in worst fall (+ = shallower).",
        "",
        "METHOD",
        f"Equivalent bets = DR^2, DR = sum(w_i * sigma_i) / sigma_p, daily returns "
        f"{n.start:%Y-%m-%d} to {n.end:%Y-%m-%d}.",
        "Stress windows (S&P 500 peak to trough): " + "; ".join(
            f"{v.start:%Y-%m-%d} to {v.end:%Y-%m-%d}" for v in STRESS_SCENARIOS.values()) + ".",
        f"Expected return (CAPM): rf {wi.rf:.2%}, market risk premium {wi.mrp:.1%}, beta vs VT.",
        "Historical comparison only; not a recommendation.",
    ]
    return "\n".join(line.rstrip() for line in lines) + "\n"


def _pts(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    v = round(x * 100, 1)
    return "0.0 pts" if v == 0 else f"{v:+.1f} pts"


def diversification_verdict(change_in_bets: float | None) -> str:
    if change_in_bets is None or abs(change_in_bets) <= VERDICT_THRESHOLD:
        return "little_change"
    return "diversifies" if change_in_bets > 0 else "concentrates"


def purchase_summary(pc: PurchaseCheck) -> dict:
    """Structured result of a pre-trade check, shared by the API and the order-ticket demo."""
    periods = {}
    for key, r in pc.impact.iterrows():
        def num(x):
            return None if pd.isna(x) else round(float(x), 4)
        periods[key] = {
            "label": r["label"],
            "bets_before": num(r["bets_before"]), "bets_after": num(r["bets_after"]),
            "volatility_before": num(r["vol_before"]), "volatility_after": num(r["vol_after"]),
            "worst_fall_before": num(r["dd_before"]), "worst_fall_after": num(r["dd_after"]),
            "worst_fall_usd_before": num(r["loss_before"]), "worst_fall_usd_after": num(r["loss_after"]),
        }
    tbl = ranking_table(pc.wi, "normal", pc.alpha)
    alt = tbl.drop(index=pc.ticker, errors="ignore")
    robust = alt[alt["all_crises"]]
    best = (robust if not robust.empty else alt).index[0] if not alt.empty else None
    pick = tbl.loc[pc.ticker] if pc.ticker in tbl.index else None
    change = None if pick is None else round(float(pick["d_bets"]), 4)
    softened = None if pick is None else bool(pick["all_crises"])
    better = best is not None and pick is not None and (
        tbl.loc[best, "bets"] > pick["bets"] + VERDICT_THRESHOLD
        or (bool(tbl.loc[best, "all_crises"]) and not softened))
    return {
        "ticker": pc.ticker,
        "name": pc.name,
        "amount": pc.amount,
        "portfolio_value": pc.value,
        "share_of_portfolio_after": round(pc.alpha, 4),
        "already_held": pc.wi.held.get(pc.ticker, ""),
        "change_in_bets": change,
        "verdict": diversification_verdict(change),
        "softened_all_crises": softened,
        "periods": periods,
        "best_alternative": None if best is None else {
            "ticker": best,
            "name": pc.wi.candidates.get(best, best),
            "bets_after": round(float(tbl.loc[best, "bets"]), 4),
            "softened_all_crises": bool(tbl.loc[best, "all_crises"]),
            "is_better": bool(better),
        },
        "data_to": f"{pc.wi.prices.index.max():%Y-%m-%d}",
    }
