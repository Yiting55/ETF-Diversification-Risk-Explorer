"""Monte Carlo outlook page, isolated from the original app tabs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.data import PriceStore, normalize_ticker  # noqa: E402
from engine.monte_carlo import MonteCarloResult, simulate_portfolio  # noqa: E402

st.set_page_config(page_title="Monte Carlo outlook", page_icon=":material/query_stats:", layout="wide")


@st.cache_resource
def price_store() -> PriceStore:
    return PriceStore()


@st.cache_data(show_spinner="Loading adjusted price history...", max_entries=16)
def load_prices(tickers: tuple[str, ...]):
    return price_store().get(list(tickers))


@st.cache_data(show_spinner="Simulating portfolio outcomes...", max_entries=16)
def run_simulation(returns: pd.DataFrame, weights: tuple[tuple[str, float], ...],
                   starting_value: float, horizon_days: int, n_simulations: int,
                   block_days: int, seed: int) -> MonteCarloResult:
    return simulate_portfolio(
        returns, dict(weights), starting_value=starting_value, horizon_days=horizon_days,
        n_simulations=n_simulations, block_days=block_days, seed=seed)


def parse_portfolio(ticker_text: str, weight_text: str) -> dict[str, float]:
    tickers = [normalize_ticker(t) for t in ticker_text.split(",") if t.strip()]
    raw_weights = [w.strip() for w in weight_text.split(",") if w.strip()]
    if not tickers:
        raise ValueError("Enter at least one ticker.")
    if len(set(tickers)) != len(tickers):
        raise ValueError("Enter each ticker only once.")
    if len(tickers) != len(raw_weights):
        raise ValueError("Enter one capital weight for each ticker, in the same order.")
    try:
        weights = [float(w) for w in raw_weights]
    except ValueError as exc:
        raise ValueError("Capital weights must be numeric percentages.") from exc
    if not np.isfinite(weights).all() or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Weights must be finite, non-negative percentages with a positive total.")
    return dict(zip(tickers, weights))


def path_chart(result: MonteCarloResult) -> go.Figure:
    paths = result.paths
    days = np.arange(paths.shape[1])
    q05, q25, q50, q75, q95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    fig = go.Figure()
    rng = np.random.default_rng(17)
    sample_n = min(30, len(paths))
    for i in rng.choice(len(paths), size=sample_n, replace=False):
        fig.add_scatter(x=days, y=paths[i], mode="lines", showlegend=False,
                        line=dict(color="rgba(100, 110, 125, 0.16)", width=1),
                        hoverinfo="skip")
    fig.add_scatter(x=days, y=q05, mode="lines", name="5th–95th percentile",
                    line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=days, y=q95, mode="lines", name="5th–95th percentile",
                    line=dict(width=0), fill="tonexty", fillcolor="rgba(57, 135, 229, 0.14)",
                    hoverinfo="skip")
    fig.add_scatter(x=days, y=q25, mode="lines", name="25th–75th percentile",
                    line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=days, y=q75, mode="lines", name="25th–75th percentile",
                    line=dict(width=0), fill="tonexty", fillcolor="rgba(57, 135, 229, 0.28)",
                    hoverinfo="skip")
    fig.add_scatter(x=days, y=q50, mode="lines", name="Median path",
                    line=dict(color="#1f5d9f", width=2.5),
                    hovertemplate="Day %{x}<br>Median value: $%{y:,.0f}<extra></extra>")
    fig.add_hline(y=result.starting_value, line_dash="dot", line_color="#8f8e88",
                  annotation_text="Starting value", annotation_position="top left")
    fig.update_layout(height=440, margin=dict(l=10, r=15, t=20, b=10),
                      hoverlabel=dict(namelength=-1), legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(title="Trading days simulated", showline=False, zeroline=False)
    fig.update_yaxes(title="Portfolio value (US$)", tickprefix="$", tickformat=",.0f",
                     showline=False, zeroline=False, gridwidth=1)
    return fig


def terminal_chart(result: MonteCarloResult) -> go.Figure:
    summary = result.summary
    fig = go.Figure(go.Histogram(
        x=result.terminal_values, nbinsx=45, marker_color="#3987e5", opacity=0.82,
        hovertemplate="Ending value: $%{x:,.0f}<br>Simulations: %{y}<extra></extra>"))
    fig.add_vline(x=result.starting_value, line_dash="dot", line_color="#8f8e88",
                  annotation_text="Starting value", annotation_position="top left")
    fig.add_vline(x=summary["terminal_median"], line_color="#d95926",
                  annotation_text="Median", annotation_position="top right")
    fig.update_layout(height=350, margin=dict(l=10, r=15, t=35, b=10), showlegend=False)
    fig.update_xaxes(title="Simulated ending portfolio value (US$)", tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title="Number of simulations")
    return fig


def render_results(result: MonteCarloResult):
    summary = result.summary
    st.caption(f"Resampled {result.n_observations:,} complete daily observations from "
               f"{result.history_start:%d %b %Y} through {result.history_end:%d %b %Y}. "
               f"{result.n_simulations:,} paths × {result.horizon_days:,} trading days; "
               f"{result.block_days}-day blocks.")
    st.info("These are historical resampling scenarios, not forecasts. Results depend on the selected "
            "historical sample and assume the portfolio is rebalanced daily to the entered weights.")

    a, b, c = st.columns(3)
    a.metric("Median ending value", f"${summary['terminal_median']:,.0f}", border=True)
    b.metric("Simulations below starting value", f"{summary['probability_below_start']:.1%}", border=True)
    c.metric("Median maximum drawdown", f"{summary['max_drawdown_median']:.1%}", border=True)

    st.subheader("Range of simulated portfolio paths")
    st.caption("Shaded bands show the 25th–75th and 5th–95th percentile ranges across simulations.")
    st.plotly_chart(path_chart(result), width="stretch", config={"displayModeBar": False})

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Simulated ending values")
        st.plotly_chart(terminal_chart(result), width="stretch", config={"displayModeBar": False})
    with right:
        st.subheader("Outcome ranges")
        table = pd.DataFrame([
            {"Percentile": "5th", "Ending value": summary["terminal_p05"]},
            {"Percentile": "25th", "Ending value": summary["terminal_p25"]},
            {"Percentile": "Median", "Ending value": summary["terminal_median"]},
            {"Percentile": "75th", "Ending value": summary["terminal_p75"]},
            {"Percentile": "95th", "Ending value": summary["terminal_p95"]},
        ])
        st.dataframe(table.style.format({"Ending value": "${:,.0f}"}),
                     hide_index=True, width="stretch")
        st.metric("5th percentile maximum drawdown", f"{summary['max_drawdown_p05']:.1%}",
                  help="Five percent of simulations had an even deeper maximum drawdown.")


def main():
    st.title("Monte Carlo outlook")
    st.caption("Explore a distribution of possible portfolio paths by resampling historical returns.")

    with st.form("monte_carlo_settings"):
        st.subheader("Portfolio and simulation settings")
        ticker_text = st.text_input("Holdings tickers, comma-separated", value="VOO, AGG")
        weight_text = st.text_input("Capital weights in the same order (%)", value="60, 40")
        left, middle, right = st.columns(3)
        starting_value = left.number_input("Starting portfolio value (US$)", min_value=100.0,
                                           value=10_000.0, step=1_000.0)
        lookback_years = middle.selectbox("Historical sample", [1, 3, 5, 7], index=2,
                                          format_func=lambda n: f"Last {n} years")
        horizon_years = right.selectbox("Simulation horizon", [1, 2, 3], index=0,
                                        format_func=lambda n: f"{n} year" + ("s" if n > 1 else ""))
        left, middle, right = st.columns(3)
        n_simulations = left.selectbox("Number of simulations", [500, 1_000, 2_000, 5_000], index=2)
        block_days = middle.selectbox("Return block length", [1, 5, 21], index=2,
                                      format_func=lambda n: {1: "1 day", 5: "1 week", 21: "1 month"}[n])
        seed = right.number_input("Random seed", min_value=0, max_value=2_147_483_647,
                                  value=42, step=1)
        submitted = st.form_submit_button("Run simulation", type="primary")

    if submitted:
        try:
            parsed = parse_portfolio(ticker_text, weight_text)
            st.session_state["mc_config"] = {
                "weights": parsed, "starting_value": float(starting_value),
                "lookback_years": int(lookback_years), "horizon_years": int(horizon_years),
                "n_simulations": int(n_simulations), "block_days": int(block_days),
                "seed": int(seed),
            }
            st.session_state.pop("mc_error", None)
        except ValueError as exc:
            st.session_state.pop("mc_config", None)
            st.session_state["mc_error"] = str(exc)

    if st.session_state.get("mc_error"):
        st.error(st.session_state["mc_error"])

    config = st.session_state.get("mc_config")
    if not config:
        st.info("Enter a portfolio and select **Run simulation** to generate the outcome range.")
        return

    holdings = config["weights"]
    prices, unknown = load_prices(tuple(holdings))
    if unknown:
        st.error(f"No adjusted price history found for: {', '.join(unknown)}. Check the ticker symbols and try again.")
        return
    if prices.empty:
        st.error("No price history is available for these holdings.")
        return

    as_of = prices.index.max()
    start = as_of - pd.DateOffset(years=config["lookback_years"])
    prices = prices.loc[start:as_of, list(holdings)]
    daily_returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if daily_returns.empty:
        st.error("There are no complete daily returns in the selected historical sample.")
        return
    try:
        result = run_simulation(
            daily_returns, tuple(holdings.items()), config["starting_value"],
            config["horizon_years"] * 252, config["n_simulations"], config["block_days"], config["seed"])
    except ValueError as exc:
        st.error(str(exc))
        return
    render_results(result)

    with st.expander("How this simulation works"):
        st.markdown(
            "The page converts adjusted closes to daily simple returns, combines each day across "
            "holdings using the entered weights, then samples contiguous historical blocks with "
            "replacement. Blocks preserve short-term clustering in the observed portfolio returns. "
            "The portfolio is assumed to rebalance to the entered weights each trading day. "
            "The displayed percentiles and loss frequency describe this resampling experiment; "
            "they are not probabilities calibrated to future markets and do not account for "
            "structural changes, fees, taxes, or trading costs."
        )


main()
