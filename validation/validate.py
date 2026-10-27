"""Out-of-sample checks for the diversification score and the What-if ranking.

Everything is estimated on the five years before each crisis and then compared with what
actually happened during the crisis (S&P 500 peak to trough).

    Test 1  Does a higher pre-crisis score (equivalent bets, DR^2) mean a smaller crisis fall?
    Test 2  Did the What-if top pick (most added bets, chosen before the crisis) soften the fall?
    Test 3  Before 2022, the best COVID cushion was known. Did that pick help in 2022?

Run from the repo root:  python validation/validate.py
Writes validation/results/*.csv and validation/results/summary.md (plus a chart if matplotlib is installed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.data import PriceStore  # noqa: E402
from engine.metrics import diversification_metrics, estimate_covariance  # noqa: E402
from engine.scenarios import STRESS_SCENARIOS  # noqa: E402
from engine.whatif import CANDIDATE_SETS  # noqa: E402

OUT = ROOT / "validation" / "results"
SEED = 7
N_TEST1 = 1000
N_TEST2 = 500
ALPHA = 0.10
LOOKBACK_YEARS = 5

CANDIDATES = list(CANDIDATE_SETS["US"])
# Broad universe for Test 1: all asset classes, all listed before 2015.
UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "MDY", "EFA", "EEM", "VGK", "EWJ", "VNQ", "TLT", "IEF", "SHY", "AGG",
            "LQD", "HYG", "TIP", "GLD", "SLV", "DBC", "XLE", "XLF", "XLK", "XLV", "XLU", "XLP", "XLY", "XLI",
            "XLB", "VIG"]
# Equity-heavy holdings for Test 2 (a typical DIY investor), excluding the What-if candidates.
EQUITY = ["SPY", "QQQ", "IWM", "DIA", "MDY", "EFA", "VGK", "EWJ", "XLE", "XLF", "XLK", "XLV", "XLU", "XLP",
          "XLY", "XLI", "XLB", "VIG"]
BONDS = ["AGG", "LQD", "TIP", "SHY"]


def windows(prices: pd.DataFrame, key: str):
    s = STRESS_SCENARIOS[key]
    pre = prices.loc[s.start - pd.DateOffset(years=LOOKBACK_YEARS): s.start - pd.Timedelta(days=1)]
    crisis = prices.loc[s.start:s.end]
    return pre, crisis


def pre_cov(pre: pd.DataFrame) -> pd.DataFrame:
    rets = pre.pct_change(fill_method=None).iloc[1:].dropna()
    cov, _, _ = estimate_covariance(rets, shrink=False)
    return pd.DataFrame(cov, index=pre.columns, columns=pre.columns)


def crisis_fall(crisis: pd.DataFrame, w: pd.Series) -> float:
    """Worst peak-to-trough fall of a buy-and-hold portfolio over the crisis window."""
    px = crisis[w.index].dropna()
    value = (px / px.iloc[0]) @ w.to_numpy()
    return float((value / value.cummax() - 1).min())


def random_weights(rng, tickers, k_min=3, k_max=8, must=None) -> pd.Series:
    k = int(rng.integers(k_min, k_max + 1))
    pick = list(rng.choice(tickers, size=k, replace=False))
    if must:
        pick = list(dict.fromkeys(must + pick))
    w = rng.dirichlet(np.ones(len(pick)))
    return pd.Series(w, index=pick)


def bets(w: pd.Series, cov: pd.DataFrame) -> tuple[float, float]:
    m = diversification_metrics(w.to_numpy(), cov.loc[w.index, w.index].to_numpy())
    return m["effective_bets"], m["portfolio_vol"]


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def partial_spearman(x: pd.Series, y: pd.Series, z: pd.Series) -> float:
    """Rank correlation of x and y after removing the linear effect of z from both ranks."""
    rx, ry, rz = x.rank(), y.rank(), z.rank()
    def resid(r):
        beta = np.polyfit(rz, r, 1)
        return r - np.polyval(beta, rz)
    return float(np.corrcoef(resid(rx), resid(ry))[0, 1])


# --------------------------------------------------------------------------- Test 1
def test1(prices, rng) -> pd.DataFrame:
    portfolios = [random_weights(rng, UNIVERSE) for _ in range(N_TEST1)]
    rows = []
    for key in STRESS_SCENARIOS:
        pre, crisis = windows(prices, key)
        cov = pre_cov(pre[UNIVERSE])
        for i, w in enumerate(portfolios):
            b, vol = bets(w, cov)
            rows.append({"crisis": key, "portfolio": i, "holdings": len(w), "pre_bets": b, "pre_vol": vol,
                         "crisis_fall": crisis_fall(crisis, w)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Test 2
def test2(prices, rng) -> pd.DataFrame:
    portfolios = []
    for _ in range(N_TEST2):
        w = random_weights(rng, EQUITY, 3, 7)
        if rng.random() < 0.5:                           # half of them also hold a bond fund
            w = pd.concat([w * 0.85, pd.Series({rng.choice(BONDS): 0.15})])
        portfolios.append(w)
    rows = []
    for key in STRESS_SCENARIOS:
        pre, crisis = windows(prices, key)
        cov = pre_cov(pre[EQUITY + BONDS + CANDIDATES])
        for i, w in enumerate(portfolios):
            base_bets, _ = bets(w, cov)
            base_fall = crisis_fall(crisis, w)
            for k in CANDIDATES:
                wk = (1 - ALPHA) * w
                wk.loc[k] = wk.get(k, 0.0) + ALPHA
                kb, _ = bets(wk, cov)
                rows.append({"crisis": key, "portfolio": i, "candidate": k,
                             "pre_d_bets": kb - base_bets,
                             "crisis_d_fall": crisis_fall(crisis, wk) - base_fall})
    return pd.DataFrame(rows)


def summarise(t1: pd.DataFrame, t2: pd.DataFrame) -> str:
    lines = ["# Out-of-sample validation", "",
             f"Pre-crisis window: {LOOKBACK_YEARS} years of daily returns ending the day before each crisis. "
             "Crisis windows: S&P 500 peak to trough (COVID 2020-02-19 to 2020-03-23; 2022-01-03 to 2022-10-12). "
             "Worst fall is buy-and-hold over the crisis window. Seed "
             f"{SEED}.", ""]

    lines += ["## Test 1: does a higher pre-crisis score mean a smaller crisis fall?", "",
              f"The same {N_TEST1} random portfolios in both crises, 3-8 ETFs from a 30-ETF universe, random weights.", "",
              "| Crisis | Spearman(score, fall) | Same, controlling for pre-crisis volatility | "
              "Avg fall, lowest-score fifth | Avg fall, highest-score fifth |", "|---|---|---|---|---|"]
    for key, g in t1.groupby("crisis", sort=False):
        q = pd.qcut(g["pre_bets"], 5, labels=False)
        lines.append(f"| {key} | {spearman(g['pre_bets'], g['crisis_fall']):+.2f} | "
                     f"{partial_spearman(g['pre_bets'], g['crisis_fall'], g['pre_vol']):+.2f} | "
                     f"{g.loc[q == 0, 'crisis_fall'].mean():.1%} | {g.loc[q == 4, 'crisis_fall'].mean():.1%} |")
    lines += ["", "Positive correlation = portfolios with more equivalent bets before the crisis fell less.", ""]

    lines += ["## Test 2: did the What-if top pick, chosen before the crisis, soften the fall?", "",
              f"The same {N_TEST2} equity-heavy portfolios in both crises (half also hold a bond fund). For each, the candidate "
              f"with the most added bets at {ALPHA:.0%}, using pre-crisis data only.", "",
              "| Crisis | Top pick softened the fall | Avg change in fall, top pick | "
              "Avg change in fall, all candidates | Most common top picks |", "|---|---|---|---|---|"]
    for key, g in t2.groupby("crisis", sort=False):
        top = g.loc[g.groupby("portfolio")["pre_d_bets"].idxmax()]
        picks = top["candidate"].value_counts(normalize=True).head(3)
        lines.append(f"| {key} | {(top['crisis_d_fall'] > 0).mean():.0%} | "
                     f"{top['crisis_d_fall'].mean() * 100:+.2f} pts | {g['crisis_d_fall'].mean() * 100:+.2f} pts | "
                     + ", ".join(f"{k} {v:.0%}" for k, v in picks.items()) + " |")
    lines += ["", "Change in fall > 0 means a shallower fall than the original portfolio.", ""]

    lines += ["### Each candidate's average effect", "",
              "| Candidate | COVID: avg change in fall | 2022: avg change in fall | Softened both (share of portfolios) |",
              "|---|---|---|---|"]
    piv = t2.pivot_table(index=["portfolio", "candidate"], columns="crisis", values="crisis_d_fall")
    for k in CANDIDATES:
        sub = piv.xs(k, level="candidate")
        lines.append(f"| {k} | {sub['covid'].mean() * 100:+.2f} pts | {sub['2022'].mean() * 100:+.2f} pts | "
                     f"{((sub['covid'] > 0) & (sub['2022'] > 0)).mean():.0%} |")
    lines.append("")

    lines += ["## Test 3: did the best COVID cushion also help in 2022?", ""]
    covid = t2[t2.crisis == "covid"].set_index(["portfolio", "candidate"])["crisis_d_fall"]
    y22 = t2[t2.crisis == "2022"].set_index(["portfolio", "candidate"])["crisis_d_fall"]
    best_covid = covid.groupby("portfolio").idxmax().map(lambda t: t[1])
    hits = [y22.loc[(p, k)] for p, k in best_covid.items() if (p, k) in y22.index]
    lines += [f"Across {len(hits)} portfolios, the candidate that cushioned COVID best was most often "
              f"{best_covid.value_counts().index[0]} ({best_covid.value_counts(normalize=True).iloc[0]:.0%}). "
              f"Holding that same pick in 2022 softened the fall in {np.mean(np.array(hits) > 0):.0%} of portfolios, "
              f"with an average change of {np.mean(hits) * 100:+.2f} pts.", ""]
    return "\n".join(lines)


def chart(t1: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, (key, g) in zip(axes, t1.groupby("crisis", sort=False)):
        ax.scatter(g["pre_bets"], g["crisis_fall"] * 100, s=6, alpha=0.4, color="#3987e5")
        ax.set_title({"covid": "COVID 2020", "2022": "2022 sell-off"}[key])
        ax.set_xlabel("Equivalent bets, 5 years before the crisis")
        ax.set_ylabel("Worst fall in the crisis (%)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "score_vs_fall.png", dpi=150)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tickers = list(dict.fromkeys(UNIVERSE + CANDIDATES))
    prices, unknown = PriceStore().get(tickers)
    if unknown:
        raise SystemExit(f"Missing price data: {unknown}")
    t1 = test1(prices, rng)
    t2 = test2(prices, rng)
    t1.to_csv(OUT / "test1_score_vs_fall.csv", index=False)
    t2.to_csv(OUT / "test2_whatif_candidates.csv", index=False)
    summary = summarise(t1, t2)
    (OUT / "summary.md").write_text(summary)
    chart(t1)
    print(summary)


if __name__ == "__main__":
    main()
