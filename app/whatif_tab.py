"""What-if tab: compare candidate ETFs, or check a planned purchase."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import data as edata
from engine import whatif as wf

PLOTLY_CONFIG = {"displayModeBar": False}
MUTED = "#8f8e88"
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
WINDOWS = {"normal": "Historical overview", "covid": "COVID 2020", "2022": "2022 sell-off"}
SHORT = {"covid": "COVID", "2022": "2022"}
MAX_CUSTOM = 3


def _defaults() -> dict:
    return wf.CANDIDATE_SETS[wf.DEFAULT_REGION]


@st.cache_data
def _universe() -> pd.DataFrame:
    return edata.load_universe().set_index("yf_symbol")


@st.cache_data(show_spinner="Testing each candidate ETF...")
def _run(holdings: tuple, lookback: int, mrp: float, rf: float | None, candidates: tuple, _store,
         engine_revision: str = ""):
    return wf.run_whatif(dict(holdings), lookback_years=lookback, mrp=mrp, rf=rf,
                      candidates=dict(candidates), store=_store)


@st.cache_data(show_spinner="Resampling history...")
def _stability(holdings: tuple, lookback: int, mrp: float, rf: float | None, candidates: tuple,
               alpha: float, _store, engine_revision: str = ""):
    return wf.ranking_stability(_run(holdings, lookback, mrp, rf, candidates, _store, engine_revision), alpha=alpha)


@st.cache_data(show_spinner="Checking this purchase...")
def _purchase(holdings: tuple, ticker: str, amount: float, value: float, lookback: int, mrp: float,
              rf: float | None, _store, engine_revision: str = ""):
    return wf.check_purchase(dict(holdings), ticker, amount, value, lookback_years=lookback, mrp=mrp, rf=rf,
                          store=_store)


def _layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), hoverlabel=dict(namelength=-1),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0), **kw)
    fig.update_xaxes(showline=False, zeroline=False, gridwidth=1)
    fig.update_yaxes(showline=False, zeroline=False, gridwidth=1)
    return fig


def _spread_labels(ends: dict, min_gap_share: float = 0.045):
    if not ends:
        return []
    ys = [v[1] for v in ends.values()]
    gap = max((max(ys) - min(ys)) * min_gap_share, 0.02)
    out, last = [], None
    for k, (x, y) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        lab = y if last is None else max(y, last + gap)
        out.append((k, (x, y), lab))
        last = lab
    return out


def _crisis_line(tbl: pd.DataFrame, pick: str | None = None) -> str:
    both = list(tbl.index[tbl["all_crises"]])
    if not both:
        line = "**None of these would have softened both COVID 2020 and the 2022 sell-off.**"
    else:
        line = f"**Softened both COVID 2020 and the 2022 sell-off:** {', '.join(both)}."
    if pick is not None and pick in tbl.index:
        line += (f" Your pick {pick} did too." if tbl.at[pick, "all_crises"]
                 else f" Your pick {pick} did not.")
    return line + " Different crises reward different diversifiers."


def _pts(x: float, digits: int = 1) -> str:
    v = round(x * 100, digits)
    return f"{0.0:.{digits}f} pts" if v == 0 else f"{v:+.{digits}f} pts"


def _money(x: float) -> str:
    return f"\\${x:,.0f}"


def _styles(cands) -> dict:
    order = list(_defaults()) + [k for k in cands if k not in _defaults()]
    extra = [("dash", "square"), ("dot", "diamond"), ("dashdot", "triangle-up")]
    out = {}
    for i, k in enumerate(order):
        if i < len(PALETTE):
            out[k] = {"color": PALETTE[i], "dash": "solid", "symbol": "circle"}
        else:
            d, sym = extra[(i - len(PALETTE)) % len(extra)]
            out[k] = {"color": MUTED, "dash": d, "symbol": sym}
    return out


def _assumptions():
    with st.popover("Return assumptions (CAPM)"):
        mrp = st.slider("Market risk premium", 0.02, 0.08, wf.MRP_DEFAULT, 0.005, format="%.3f", key="wi_mrp")
        own_rf = st.toggle("Set risk-free rate myself", key="wi_own_rf")
        rf = st.number_input("Risk-free rate", value=0.04, step=0.0025, format="%.4f",
                             key="wi_rf") if own_rf else None
    return mrp, rf


def _ticker_picker(label: str, key: str, exclude=(), multi=False, help=None):
    uni = _universe()
    options = [t for t in uni.index if t not in exclude]
    fmt = lambda t: f"{t} · {uni.at[t, 'name']}" if t in uni.index else t  # noqa: E731
    if multi:
        return st.multiselect(label, options, key=key, format_func=fmt, accept_new_options=True,
                              max_selections=MAX_CUSTOM, help=help)
    return st.selectbox(label, options, index=None, key=key, format_func=fmt, accept_new_options=True,
                        placeholder="Search by ticker or name", help=help)


def render_whatif(holdings: dict, lookback: int, store, value: float = 10_000, engine_revision: str = ""):
    mode = st.segmented_control("Mode", ["Compare ETFs", "Check a purchase"], default="Compare ETFs",
                                key="wi_mode", label_visibility="collapsed") or "Compare ETFs"
    key = tuple(sorted(holdings.items()))
    if mode == "Check a purchase":
        _render_purchase(key, lookback, store, value, engine_revision)
    else:
        _render_compare(key, lookback, store, engine_revision)


def _render_compare(key: tuple, lookback: int, store, rev: str):
    st.caption("Test moving part of your portfolio into one more ETF. Existing holdings are scaled down "
               "pro-rata, so the total amount invested stays the same. A historical comparison, not advice.")

    c1, c2, c3 = st.columns([4, 3, 2])
    with c1:
        window = st.segmented_control("Measured in", WINDOWS, format_func=WINDOWS.get, default="normal",
                                      key="wi_window") or "normal"
    with c2:
        alpha = st.select_slider("Share moved into the new ETF", options=list(wf.ALPHA_GRID), value=wf.ALPHA_DEFAULT,
                                 format_func=lambda a: f"{a:.0%}", key="wi_alpha")
    with c3:
        mrp, rf = _assumptions()
    custom = _ticker_picker(f"Add your own candidates (up to {MAX_CUSTOM})", "wi_custom", exclude=_defaults(),
                            multi=True, help="Any US stock or ETF. Compared on the same terms as the defaults.")
    custom = [edata.normalize_ticker(t) for t in custom if edata.normalize_ticker(t) not in _defaults()]
    candidates = {**_defaults(), **{t: wf.ticker_name(t) for t in custom}}

    try:
        wi = _run(key, lookback, mrp, rf, tuple(candidates.items()), store, rev)
    except ValueError as e:
        st.error(str(e))
        return
    missing = [t for t in custom if t not in wi.candidates]
    if missing:
        st.warning(f"No price data found for {', '.join(missing)}; left out.")
    if window not in wi.base:
        st.warning("Not enough overlapping price data for this period.")
        return

    cands = list(wi.candidates)
    style = _styles(cands)
    color = {k: v["color"] for k, v in style.items()}
    base = wi.base[window]
    normal = window == "normal"
    labels = ({"d_bets": "Most added bets", "worst_crisis": "Most crisis-proof"} if normal
              else {"d_bets": "Most added bets", "d_max_dd": "Shallowest fall"})
    rank_by = st.segmented_control("Rank by", labels, format_func=labels.get, default="d_bets",
                                   key=f"wi_rank_by_{'normal' if normal else 'stress'}") or "d_bets"
    if rank_by == "worst_crisis":
        st.caption("Most crisis-proof = ranked by the smaller improvement in worst fall across COVID 2020 "
                   "and the 2022 sell-off, so a candidate only ranks high if it helped in both.")
    tbl = wf.ranking_table(wi, window, alpha, rank_by)

    st.caption(f"{base.start:%d %b %Y} - {base.end:%d %b %Y} ({base.n_obs:,} trading days). "
               f"Risk-free rate {wi.rf:.2%}"
               + ("" if rf is not None else f" ({wf.RF_PROXY[wf.DEFAULT_REGION]} trailing 12-month return)")
               + f", market risk premium {wi.mrp:.1%}.")
    gone = [k.split(":")[1] for k in wi.unavailable if k.startswith(f"{window}:")]
    if gone:
        st.caption(f"Not testable in this period (no prices yet): {', '.join(gone)}.")
    for n in wi.notes.get(window, []):
        st.caption(n)
    if base.low_confidence:
        st.caption("Short window: correlations are estimated with low confidence (see the Stress test tab).")

    m = base.metrics
    a, b, c = st.columns(3)
    a.metric("Equivalent bets now", f"{m['effective_bets']:.1f}", border=True)
    b.metric("Volatility now", f"{m['portfolio_vol']:.1%}", border=True)
    if normal:
        cur_er = wi.grid[(wi.grid.window == "normal") & (wi.grid.candidate == "(current)")]["er"].iloc[0]
        c.metric("Expected return now (CAPM)", f"{cur_er:.2%}", border=True)
    else:
        c.metric("Worst fall in this period", f"{base.replay['max_drawdown']:.1%}", border=True)

    if tbl.empty:
        st.info("No candidate could be tested in this period.")
        return
    st.info(wf.summary_sentence(wi, tbl, window, alpha, rank_by), icon=":material/lightbulb:")
    st.subheader(f"Each candidate at {alpha:.0%}")
    cols = {
        "Rank": tbl["rank"],
        "ETF": [f"{k} · {wi.candidates.get(k, k)}" + (f" ({tbl.at[k, 'held']})" if tbl.at[k, "held"] else "")
                for k in tbl.index],
        "Bets after": tbl["bets"].round(2),
        "Δ bets": tbl["d_bets"].map(lambda x: f"{x:+.2f}"),
        "Δ volatility": tbl["d_vol"].map(_pts),
    }
    if normal:
        cols["Δ expected return"] = tbl["d_er"].map(lambda x: _pts(x, 2))
    else:
        cols["Δ worst fall"] = tbl["d_max_dd"].map(_pts)
    if normal:
        for c in [c for c in tbl.columns if c.startswith("dd_") and c[3:] in WINDOWS]:
            cols[f"Δ fall {SHORT[c[3:]]}"] = tbl[c].map(lambda x: "n/a" if pd.isna(x) else _pts(x))
    if not normal:
        cols["Most bets at"] = tbl["best_alpha"].map(lambda x: "40% or more" if x >= max(wf.ALPHA_GRID) else f"{x:.0%}")
    cols["Efficient"] = tbl["efficient"].map({True: "Yes", False: ""})
    cols["Both crises"] = tbl["all_crises"].map({True: "Yes", False: ""})
    st.dataframe(pd.DataFrame(cols), hide_index=True, width="stretch")
    st.markdown(_crisis_line(tbl))
    st.caption("Efficient = no other candidate is better on both added bets and "
               + ("expected return." if normal else "worst fall.")
               + (" A positive change in worst fall means a shallower drop." if not normal else "")
               + (" Candidates you already hold (or a near-identical fund of) get a larger weight, not a new line."
                  if tbl["held"].any() else "")
               + " Both crises = the worst fall would have been shallower in COVID 2020 and in the 2022 sell-off.")

    left, right = st.columns(2)
    with left:
        st.subheader("How much to move?")
        g = wi.grid[wi.grid.window == window]
        fig, ends = go.Figure(), {}
        for k in cands:
            d = g[g.candidate == k]
            if d.empty:
                continue
            x = np.r_[0.0, d["alpha"]]
            y = np.r_[m["effective_bets"], d["bets"]]
            fig.add_scatter(x=x, y=y, name=k, mode="lines+markers",
                            line=dict(width=2, color=color[k], dash=style[k]["dash"]),
                            marker=dict(size=8, symbol=style[k]["symbol"]), hovertemplate=f"{k}: %{{y:.2f}} bets at %{{x:.0%}}<extra></extra>")
            ends[k] = (x[-1], y[-1])
        for k, (x_end, _), y_lab in _spread_labels(ends):
            fig.add_annotation(x=x_end, y=y_lab, text=k, showarrow=False, xanchor="left", xshift=8)
        fig.add_hline(y=m["effective_bets"], line_dash="dot", line_color=MUTED,
                      annotation_text="now", annotation_position="bottom left")
        _layout(fig, 400, hovermode="x unified", showlegend=True,
                xaxis=dict(title="Share moved into the new ETF", tickformat=".0%", range=[-0.01, 0.47]),
                yaxis=dict(title="Equivalent bets"))
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

    with right:
        y_col, y_title = (("d_er", "Change in expected return (pts a year)") if normal
                          else ("d_max_dd", "Change in worst fall (pts, + = shallower)"))
        st.subheader("What does it cost?" if normal else "Did it actually protect?")
        fig2 = go.Figure()
        for k, r in tbl.iterrows():
            fig2.add_scatter(
                x=[r["d_bets"]], y=[r[y_col] * 100], mode="markers+text", text=[k], name=k,
                textposition="top center",
                marker=dict(size=14 if r["efficient"] else 10, color=color.get(k, MUTED), line=dict(width=2),
                            symbol=style.get(k, {}).get("symbol", "circle") + ("" if r["efficient"] else "-open")),
                hovertemplate=f"{k}<br>bets %{{x:+.2f}}<br>{y_title}: %{{y:+.2f}}<extra></extra>")
        fig2.add_hline(y=0, line_color=MUTED, line_width=1)
        fig2.add_vline(x=0, line_color=MUTED, line_width=1)
        _layout(fig2, 400, showlegend=False, xaxis=dict(title="Change in equivalent bets"),
                yaxis=dict(title=y_title))
        st.plotly_chart(fig2, width="stretch", config=PLOTLY_CONFIG)
        st.caption("Filled dots are efficient. Up and to the right is better."
                   + ("" if normal else " More bets does not always mean a smaller loss."))

    if normal:
        st.subheader("Is the ranking robust?")
        st.caption("Resamples the daily history 200 times in 21-day blocks and counts how often each ETF "
                   "adds the most bets.")
        if st.button("Run robustness check", key="wi_boot"):
            stab = _stability(key, lookback, mrp, rf, tuple(candidates.items()), alpha, store, rev)
            fig3 = go.Figure(go.Bar(x=stab.values, y=stab.index, orientation="h",
                                    marker=dict(color=[color.get(k, MUTED) for k in stab.index]),
                                    text=[f"{v:.0%}" for v in stab.values], textposition="outside",
                                    hovertemplate="%{y}: ranked first in %{x:.0%} of resamples<extra></extra>"))
            _layout(fig3, 280, xaxis=dict(tickformat=".0%", range=[0, 1.12], title="Share of resamples ranked first"),
                    yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig3, width="stretch", config=PLOTLY_CONFIG)


def _render_purchase(key: tuple, lookback: int, store, value: float, rev: str):
    st.caption("Before you buy: see what a planned purchase does to your diversification, over your historical "
               "lookback and in past crashes, and how the same amount would do elsewhere. "
               "A historical comparison, not advice.")
    c1, c2, c3 = st.columns([4, 2, 2])
    with c1:
        ticker = _ticker_picker("I plan to buy", "wi_buy_ticker",
                                help="Any US stock or ETF, including ones you already hold.")
    with c2:
        amount = st.number_input("Amount (US$)", min_value=50.0, value=2_000.0, step=500.0, key="wi_buy_amount")
    with c3:
        mrp, rf = _assumptions()
    if not ticker:
        st.info("Pick the stock or ETF you are about to buy.", icon=":material/search:")
        return

    try:
        pc = _purchase(key, edata.normalize_ticker(ticker), amount, value, lookback, mrp, rf, store, rev)
    except ValueError as e:
        st.error(str(e))
        return

    st.caption(f"Your portfolio is worth {_money(value)} (sidebar). After the purchase it is worth "
               f"{_money(value + amount)}, and the new {pc.ticker} purchase is {pc.alpha:.1%} of that. "
               "Buying with new money gives the same mix as moving that share out of your current holdings.")
    held = pc.wi.held.get(pc.ticker)
    if held:
        st.caption(f"Note: you {held.replace('already held', 'already hold it')}.")
    st.info(wf.purchase_sentence(pc), icon=":material/lightbulb:")
    st.download_button("Download check record", wf.purchase_record(pc),
                       file_name=f"pretrade_check_{pc.ticker}_{pd.Timestamp.now():%Y%m%d}.txt",
                       mime="text/plain", icon=":material/download:", key="wi_buy_record",
                       help="Inputs, impact in each period and the alternatives considered, as a plain-text record.")
    imp = pc.impact
    cards = st.columns(len(imp))
    for col, (win, r) in zip(cards, imp.iterrows()):
        title = WINDOWS.get(win, r["label"])
        if np.isnan(r["bets_after"]):
            col.metric(title, "n/a", help=f"{pc.ticker} has no prices for this period.", border=True)
            continue
        col.metric(f"{title}: equivalent bets", f"{r['bets_before']:.2f} → {r['bets_after']:.2f}",
                   delta=f"{r['bets_after'] - r['bets_before']:+.2f}", border=True)

    st.subheader("Before and after")
    rows = []
    for win, r in imp.iterrows():
        rows.append({
            "Period": WINDOWS.get(win, r["label"]),
            "Bets": f"{r['bets_before']:.2f} → {r['bets_after']:.2f}" if not np.isnan(r["bets_after"]) else "n/a",
            "Volatility": f"{r['vol_before']:.1%} → {r['vol_after']:.1%}" if not np.isnan(r["vol_after"]) else "n/a",
            "Worst fall": f"{r['dd_before']:.1%} → {r['dd_after']:.1%}" if not np.isnan(r["dd_after"]) else "n/a",
            "Worst fall in US$": (f"{r['loss_before']:,.0f} → {r['loss_after']:,.0f}"
                                  if not np.isnan(r["loss_after"]) else "n/a"),
            "Expected return (CAPM)": (f"{r['er_before']:.2%} → {r['er_after']:.2%}"
                                       if win == "normal" and not np.isnan(r["er_after"]) else ""),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Historical overview: worst fall over the lookback window. Dollar figures use your portfolio "
               "value before and after the purchase, so a bigger portfolio can lose more dollars even "
               "when its percentage fall is smaller.")
    st.subheader(f"The same {_money(amount)} elsewhere")
    window = st.segmented_control("Compare in", WINDOWS, format_func=WINDOWS.get, default="normal",
                                  key="wi_buy_window") or "normal"
    if window not in pc.wi.base:
        st.warning("Not enough overlapping price data for this period.")
        return
    tbl = wf.ranking_table(pc.wi, window, pc.alpha, "d_bets" if window == "normal" else "d_max_dd")
    if tbl.empty:
        st.info("No alternative could be tested in this period.")
        return
    cands = list(pc.wi.candidates)
    color = {k: v["color"] for k, v in _styles(cands).items()}
    now = pc.wi.base[window].metrics["effective_bets"]
    order = tbl.sort_values("bets")
    fig = go.Figure(go.Bar(
        x=order["bets"], y=order.index, orientation="h",
        marker=dict(color=[color.get(k, MUTED) for k in order.index],
                    pattern=dict(shape=["/" if k == pc.ticker else "" for k in order.index])),
        text=[f"{v:.2f}" + ("  (your pick)" if k == pc.ticker else "") for k, v in order["bets"].items()],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f} equivalent bets after buying<extra></extra>"))
    fig.add_vline(x=now, line_dash="dot", line_color=MUTED, annotation_text="now",
                  annotation_position="bottom right")
    _layout(fig, 70 + 34 * len(order), xaxis=dict(title="Equivalent bets after buying",
                                                  range=[min(now, order["bets"].min()) * 0.9,
                                                         order["bets"].max() * 1.15]))
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    if window != "normal":
        st.caption("Ranked by how much shallower the worst fall would have been: "
                   + ", ".join(f"{k} {_pts(v)}" for k, v in tbl["d_max_dd"].items()) + ".")
    st.markdown(_crisis_line(tbl, pc.ticker))
