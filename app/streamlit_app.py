"""Diversification X-Ray - Streamlit front end.

Run:  streamlit run app/streamlit_app.py
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import STRESS_SCENARIOS, analyze, custom_window  # noqa: E402
from engine.data import PriceStore, load_universe, normalize_ticker  # noqa: E402
from engine.scenarios import DEFAULT_LOOKBACK, LOOKBACK_CHOICES  # noqa: E402

st.set_page_config(page_title="Diversification X-Ray", page_icon=":material/donut_small:", layout="wide")

EXAMPLES = {
    "Typical DIY investor": {"VOO": 30, "QQQ": 25, "VTI": 15, "NVDA": 10, "AGG": 10, "GLD": 10},
    "Classic 60/40": {"VTI": 40, "VXUS": 20, "AGG": 40},
    "All-weather style": {"VTI": 30, "TLT": 40, "IEF": 15, "GLD": 7.5, "DBC": 7.5},
}

# One palette for both themes: this blue/orange pair passes the lightness, CVD and contrast checks
# on the light AND dark surface, so charts never depend on detecting the viewer's theme (which can
# go stale, e.g. when a projector or OS switches mode mid-session).
COLORS = {"money": "#3987e5", "risk": "#d95926", "muted": "#8f8e88"}
# Diverging blue <-> neutral gray <-> red for correlations, equal steps per arm. Cells carry their
# own background, so the same scale reads on either theme.
PLOTLY_CONFIG = {"displayModeBar": False}
CORR_SCALE = [[0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#e4e3df"], [0.75, "#f3a5a3"], [1, "#9f2121"]]



# --- cached data / computation ------------------------------------------------
@st.cache_resource
def price_store() -> PriceStore:
    return PriceStore()


@st.cache_data
def universe() -> pd.DataFrame:
    return load_universe().set_index("yf_symbol")


@st.cache_data(show_spinner="Measuring correlations...")
def run_analysis(holdings: tuple, value: float, lookback: int, custom: tuple | None):
    """Cache the analysis including the score baseline and lookback comparison."""
    stress = list(STRESS_SCENARIOS.values())
    if custom:
        stress.append(custom_window(*custom))
    return analyze(dict(holdings), value=value, lookback_years=lookback, stress=stress, store=price_store())


# --- formatting helpers ---------------------------------------------------------
def money(x: float) -> str:
    # Escaped: Streamlit markdown treats a pair of bare "$" as LaTeX math delimiters.
    return f"\\${x:,.0f}"


def bets_text(m: dict, n: int) -> str:
    return f"{m['effective_bets']:.1f}" + (" (hedged)" if m["hedged"] else "")


def base_layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), hoverlabel=dict(namelength=-1),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0), **kw)
    fig.update_xaxes(showline=False, zeroline=False, gridwidth=1)
    fig.update_yaxes(showline=False, zeroline=False, gridwidth=1)
    return fig


# --- charts ---------------------------------------------------------------------------
def risk_vs_money_chart(res) -> go.Figure:
    c = COLORS
    df = pd.DataFrame({"ticker": res.tickers, "money": res.weights, "risk": res.metrics["risk_share"]})
    df = df.sort_values("risk")                          # biggest risk share on top
    fig = go.Figure()
    # Plotly stacks grouped horizontal bars bottom-up, so add "risk" first to put "money" on top,
    # and reverse the legend so it reads in the same order as the bars.
    for col, name in [("risk", "Share of your risk"), ("money", "Share of your money")]:
        fig.add_bar(y=df.ticker, x=df[col], name=name, orientation="h", marker_color=c[col],
                    marker_cornerradius=4, text=df[col].map("{:.0%}".format), textposition="outside",
                    cliponaxis=False, hovertemplate=f"<b>%{{y}}</b><br>{name}: %{{x:.1%}}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08, legend_traceorder="reversed")
    fig.update_xaxes(tickformat=".0%", rangemode="tozero")
    return base_layout(fig, 90 + 46 * len(df))


def cell_text_color(v: float) -> str:
    """Dark or light ink, whichever reads on the heatmap cell behind it (cells go dark past |0.6|)."""
    return "#ffffff" if abs(v) > 0.6 else "#0b0b0b"


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    scale = CORR_SCALE
    z = corr.to_numpy().copy()
    np.fill_diagonal(z, np.nan)                          # self-correlation carries no information
    fig = go.Figure(go.Heatmap(
        z=z, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale=scale, xgap=2, ygap=2,
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(title="Correlation", thickness=10, tickvals=[-1, 0, 1]),
    ))
    if len(corr) <= 10:   # per-cell values only while they fit; hover + table view cover the rest
        for i, row in enumerate(corr.index):
            for j, col in enumerate(corr.columns):
                if i != j:
                    fig.add_annotation(x=col, y=row, text=f"{z[i, j]:.2f}", showarrow=False,
                                       font=dict(size=12, color=cell_text_color(z[i, j])))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    return base_layout(fig, 120 + 42 * len(corr))


def growth_chart(res, value: float) -> go.Figure:
    c = COLORS
    fig = go.Figure()
    growth = res.replay["growth"] * value
    for i, t in enumerate(growth.columns):
        fig.add_scatter(x=growth.index, y=growth[t], name="Individual holdings", legendgroup="h",
                        showlegend=i == 0, line=dict(color=c["muted"], width=1),
                        hovertemplate=f"{t}: %{{y:$,.0f}}<extra></extra>")
    fig.add_scatter(x=growth.index, y=res.replay["portfolio"] * value, name="Your portfolio",
                    line=dict(color=c["money"], width=2.5), hovertemplate="Portfolio: %{y:$,.0f}<extra></extra>")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return base_layout(fig, 340)


def holding_returns_chart(res) -> go.Figure:
    c = COLORS
    r = res.replay["holding_returns"].sort_values()
    fig = go.Figure(go.Bar(y=r.index, x=r.values, orientation="h", marker_color=c["money"],
                           marker_cornerradius=4, text=r.map("{:+.0%}".format), textposition="outside",
                           cliponaxis=False, hovertemplate="<b>%{y}</b>: %{x:+.1%}<extra></extra>"))
    # Leave room past both bar ends so outside labels never run into the ticker axis.
    lo, hi = min(r.min(), 0), max(r.max(), 0)
    pad = 0.25 * (hi - lo or 0.1)
    fig.update_xaxes(tickformat=".0%", range=[lo - pad, hi + pad])
    return base_layout(fig, 70 + 38 * len(r), showlegend=False)


# --- page sections -------------------------------------------------------------------
def avg_corr(corr: pd.DataFrame) -> float:
    z = corr.to_numpy()
    return float(z[~np.eye(len(z), dtype=bool)].mean()) if len(z) > 1 else 1.0


def headline(res, value: float):
    m, n = res.metrics, len(res.tickers)
    st.markdown(f"#### Your {n} holdings provide **{bets_text(m, n)} equivalent bets**.")
    a, b, c = (col.container(border=True) for col in st.columns(3))
    a.metric("Equivalent bets (DR²)", bets_text(m, n),
             help="A volatility-based diversification measure, expressed as equivalent uncorrelated bets. "
                  "It is not a count of independent risk factors. Negative correlations can push it above the holding count.")
    a.caption(f"By weight alone you look like {m['weight_count']:.1f} holdings.")
    b.metric("Risk removed by diversification", f"{m['risk_reduction']:.0%}")
    b.caption(f"Your volatility is {m['portfolio_vol']:.1%} a year; with every holding moving in "
              f"lockstep it would be {m['undiversified_vol']:.1%}.")
    c.metric("A bad year (1-in-20) could cost", money(res.dollars["bad_year_loss"]).replace("\\", ""),
             help="Parametric estimate: 1.645 x annual volatility x portfolio value.")
    c.caption(f"Without any diversification it would be "
              f"{money(res.dollars['bad_year_loss_undiversified'])}.")


def score_details(result):
    """Keep score interpretation and sensitivity out of the main chart area."""
    res = result.normal
    m = res.metrics
    with st.expander("Understand your diversification score"):
        st.dataframe(pd.DataFrame({
            "Measure": ["Holdings", "Effective holdings by weight", "Equivalent bets (DR²)"],
            "Value": [str(len(res.tickers)), f"{m['weight_count']:.1f}", f"{m['effective_bets']:.1f}"],
            "Accounts for": ["Ticker count", "Capital concentration", "Weights, volatility and correlation"],
        }), hide_index=True, width="stretch")
        baseline = m["uncorrelated_bets"]
        st.markdown(f"With the **same weights and volatilities**, but zero correlations, your score would be "
                    f"**{baseline:.1f}**, compared with **{m['effective_bets']:.1f}** using observed correlations.")
        st.caption("Positive correlations can reduce the score; negative correlations can improve it. "
                   "A gap from the weight-only count can also reflect differences in volatility.")
        st.markdown("**Lookback sensitivity**")
        rows = []
        available = []
        for years in LOOKBACK_CHOICES:
            window = result.lookbacks.get(years)
            if window is None:
                rows.append({"Lookback": f"{years} years", "Equivalent bets": "Unavailable",
                             "Actual dates": "Insufficient or invalid data", "Confidence": "—"})
                continue
            available.append(window)
            rows.append({"Lookback": f"{years} years", "Equivalent bets": f"{window.metrics['effective_bets']:.2f}",
                         "Actual dates": f"{window.start:%d %b %Y} – {window.end:%d %b %Y}",
                         "Confidence": "Limited history / sample" if window.low_confidence else "Standard sample"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if len(available) > 1:
            scores = [w.metrics["effective_bets"] for w in available]
            st.caption(f"Score range: {min(scores):.2f}–{max(scores):.2f}. "
                       "These overlapping historical windows are a sensitivity check, not a confidence interval.")
            if len({(w.start, w.end) for w in available}) < len(available):
                st.caption("Some lookbacks use the same actual dates because overlapping price history is limited.")
        st.caption("All lookbacks include sell-offs within their dates. Scores depend on the historical sample.")


def notes(res):
    for n in res.notes:
        st.caption(f":material/info: {n}")


def detail_table(res, label: str):
    with st.expander(f"Table view: {label}"):
        st.dataframe(pd.DataFrame({
            "Weight": res.weights, "Annual volatility": res.metrics["asset_vols"],
            "Share of risk": res.metrics["risk_share"],
            "Return over window": res.replay["holding_returns"][res.tickers].to_numpy(),
        }, index=res.tickers).style.format("{:.1%}"), width="stretch")
        st.dataframe(res.corr.style.format("{:.2f}"), width="stretch")


def duplicate_warning(tickers: list[str]):
    uni = universe()
    known = [t for t in tickers if t in uni.index and uni.at[t, "dup_cluster"]]
    groups = pd.Series({t: uni.at[t, "dup_cluster"] for t in known})
    for cluster, members in groups.groupby(groups):
        if len(members) > 1:
            st.warning(f"**{', '.join(members.index)}** are near-duplicates: their daily returns are more "
                       f"than 99.5% correlated (same exposure as {cluster}). Holding several adds cost, "
                       "not diversification.", icon=":material/content_copy:")


# --- sidebar: portfolio input -----------------------------------------------------
def load_example():
    ex = EXAMPLES[st.session_state.example]
    st.session_state.tickers = list(ex)
    for t, w in ex.items():
        st.session_state[f"w_{t}"] = float(w)


def portfolio_sidebar():
    uni = universe()
    if "tickers" not in st.session_state:
        st.session_state.example = next(iter(EXAMPLES))
        load_example()

    with st.sidebar:
        st.header("Your portfolio")
        st.selectbox("Start from an example", EXAMPLES, key="example", on_change=load_example)
        options = list(uni.index) + [t for t in st.session_state.tickers if t not in uni.index]
        st.multiselect(
            "Holdings (US stocks and ETFs)", options, key="tickers", accept_new_options=True,
            format_func=lambda t: f"{t} · {uni.at[t, 'name']}" if t in uni.index else t,
            help="Search by ticker or name. You can also type any other US ticker; it is downloaded on first use.",
        )
        holdings = {}
        if st.session_state.tickers:
            st.caption("Weight of each holding (%)")
        for t in st.session_state.tickers:
            key = f"w_{t}"
            st.session_state.setdefault(key, round(100 / len(st.session_state.tickers), 1))
            holdings[normalize_ticker(t)] = st.number_input(t, min_value=0.0, max_value=100.0, step=1.0, key=key)
        total = sum(holdings.values())
        if holdings:
            st.caption(f"Total {total:.1f}%" + ("" if abs(total - 100) < 0.05 else " - rescaled to 100%"))

        st.divider()
        value = st.number_input("Portfolio value (US$)", min_value=100.0, value=10_000.0, step=1_000.0)
        lookback = st.segmented_control("Correlation lookback", LOOKBACK_CHOICES, default=DEFAULT_LOOKBACK,
                                        format_func=lambda y: f"{y} years",
                                        help="Normal-market window of daily returns. 5 years balances a stable "
                                             "estimate against reflecting today's market.") or DEFAULT_LOOKBACK
    return holdings, value, lookback


# --- main -----------------------------------------------------------------------------
def main():
    holdings, value, lookback = portfolio_sidebar()
    st.title("Diversification X-Ray")
    st.caption("How diversified is your portfolio really, in normal markets and when markets crash?")

    if not holdings or sum(holdings.values()) <= 0:
        st.info("Add at least one holding with a positive weight in the sidebar.", icon=":material/arrow_back:")
        return

    tab_normal, tab_stress = st.tabs(["Everyday markets", "Stress test"])

    with tab_stress:
        options = {**{k: s.label for k, s in STRESS_SCENARIOS.items()}, "custom": "Custom period"}
        choice = st.segmented_control("Scenario", options, format_func=options.get, default="covid",
                                      key="scenario") or "covid"
        custom = None
        if choice == "custom":
            picked = st.date_input("Period", value=(date(2008, 9, 1), date(2009, 3, 9)),
                                   min_value=date(2000, 1, 3), max_value=date.today())
            if len(picked) == 2:
                custom = (pd.Timestamp(picked[0]), pd.Timestamp(picked[1]))

    try:
        result = run_analysis(tuple(sorted(holdings.items())), value, lookback, custom)
    except ValueError as e:
        st.error(str(e))
        return
    if result.unknown:
        st.error(f"No price data found for {', '.join(result.unknown)}; they were left out.")

    with tab_normal:
        res = result.normal
        st.caption(f"Measured over {res.start:%d %b %Y} - {res.end:%d %b %Y} ({res.n_obs:,} trading days).")
        notes(res)
        duplicate_warning(res.tickers)
        headline(res, value)
        score_details(result)
        left, right = st.columns(2)
        with left:
            st.subheader("Where your risk comes from")
            st.caption("A holding whose orange bar is longer than its blue bar adds more risk than its size suggests.")
            st.plotly_chart(risk_vs_money_chart(res), width="stretch", config=PLOTLY_CONFIG)
        with right:
            st.subheader("How your holdings move together")
            st.caption("1 = always move together, 0 = unrelated, -1 = move opposite. Red cells are where "
                       "diversification is missing.")
            st.plotly_chart(correlation_heatmap(res.corr), width="stretch", config=PLOTLY_CONFIG)
        detail_table(res, "everyday markets")

    with tab_stress:
        key = "custom" if choice == "custom" else choice
        if choice == "custom" and custom is None:
            st.info("Pick a start and end date.")
        elif key not in result.stress:
            st.warning("Not enough overlapping price data for this period.")
        else:
            stress_section(result.normal, result.stress[key], value)

    with st.expander("How we calculate this"):
        st.markdown(METHODOLOGY)


def stress_section(normal, res, value: float):
    st.caption(f"{res.scenario.description}  \n{res.start:%d %b %Y} - {res.end:%d %b %Y} "
               f"({res.n_obs} trading days).")
    notes(res)
    nm, sm, n = normal.metrics, res.metrics, len(res.tickers)
    loss = res.replay["total_return"] * value
    st.markdown(f"#### In this period your portfolio would have gone from {money(value)} to "
                f"**{money(value + loss)}** ({res.replay['total_return']:+.1%}).")

    a, b, c, d = st.columns(4)
    a.metric("Equivalent bets (DR²)", bets_text(sm, n), delta=f"{sm['effective_bets'] - nm['effective_bets']:+.1f} vs normal",
             border=True)
    b.metric("Risk removed by diversification", f"{sm['risk_reduction']:.0%}",
             delta=f"{(sm['risk_reduction'] - nm['risk_reduction']) * 100:+.0f} pts vs normal", border=True)
    c.metric("Avg. correlation of holdings", f"{avg_corr(res.corr):.2f}",
             delta=f"{avg_corr(res.corr) - avg_corr(normal.corr):+.2f} vs normal", delta_color="inverse", border=True)
    d.metric("Worst peak-to-trough fall", f"{res.replay['max_drawdown']:.1%}", border=True,
             help="Largest drop from a high point within the period.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"What {money(value)} would have done")
        st.plotly_chart(growth_chart(res, value), width="stretch", config=PLOTLY_CONFIG)
    with right:
        st.subheader("Each holding over the period")
        st.plotly_chart(holding_returns_chart(res), width="stretch", config=PLOTLY_CONFIG)

    left, right = st.columns(2)
    with left:
        st.subheader("Where your risk came from")
        st.plotly_chart(risk_vs_money_chart(res), width="stretch", config=PLOTLY_CONFIG)
    with right:
        st.subheader("How your holdings moved together")
        st.plotly_chart(correlation_heatmap(res.corr), width="stretch", config=PLOTLY_CONFIG)
    detail_table(res, res.scenario.label)


METHODOLOGY = r"""
All numbers use **daily total returns** (dividends reinvested) from Yahoo Finance.

**Diversification ratio (DR)** = weighted average of each holding's volatility / the portfolio's actual volatility.
If all holdings moved in lockstep the two would be equal (DR = 1).

* **Equivalent bets** = DR², a volatility-based measure, not a literal count of independent risk factors.
  Five uncorrelated, equally risky holdings at equal weight give exactly 5; holdings that move in lockstep give 1.
  Negatively correlated hedges can push it above the number of holdings ("hedged").
* **Zero-correlation baseline** = (Σwᵢσᵢ)² / Σ(wᵢσᵢ)². It keeps weights and volatilities fixed to isolate the effect of correlation.
* **Risk removed** = 1 - 1/DR: the share of risk that offsetting moves cancel out.
* **Share of risk** for each holding = its contribution to portfolio variance, wᵢ(Σw)ᵢ / σₚ². These add up to 100%.
* **Bad year** = 1.645 × annual volatility × portfolio value, a 1-in-20 one-year loss under a normal approximation.

**Windows.** Everyday markets use the last 3, 5 or 7 years (default 5). Stress tests run from the S&P 500's
peak close to its trough close: COVID-19 (19 Feb - 23 Mar 2020) and the 2022 inflation shock and Russia-Ukraine
war (3 Jan - 12 Oct 2022). The stress replay is buy-and-hold from the starting weights.

**Small samples.** When a window has fewer than 3 trading days per holding, correlations are shrunk toward their
average (Ledoit & Wolf, 2004) and flagged as low confidence. Shrinking toward the average rather than toward zero
avoids making a short crash look more diversified than it was.

**Source.** Choueifaty & Coignard (2008), [Toward Maximum Diversification](https://www.tobam.fr/wp-content/uploads/2014/12/TOBAM-JoPM-Maximum-Div-2008.pdf),
defines DR. We use DR² as an equivalent-bets interpretation of the volatility reduction.
"""

main()
