"""What-if tab: what one more ETF would do to the portfolio, compared with the alternatives."""
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
MAX_PICKS = 3
RF_INPUT_DEFAULT = 0.04

HELP = {
    "mrp": "Extra annual return the stock market is assumed to earn over the risk-free rate. "
           "Only affects the CAPM expected return.",
    "own_rf": "By default the risk-free rate is the trailing 12-month return of BIL, a 1-3 month T-bill ETF.",
    "rf": "Annual rate as a decimal: 0.04 = 4%.",
    "rank": "Most added bets: biggest gain in equivalent bets. Most crisis-proof: ranked by the smaller "
            "improvement in worst fall across COVID 2020 and 2022, so only ETFs that helped in both rank high. "
            "Why both: in a back-test on 500 portfolios, TLT, the top pick before COVID, softened the 2022 "
            "fall in only 17% of them, while IEF and GLD helped in both crises almost every time.",
    "now": "Your portfolio today over the lookback in the sidebar. Volatility is annualised; expected return is a "
           "CAPM estimate (risk-free rate + beta × market risk premium, beta against VT), not a forecast.",
    "curve": "Each line is one ETF. Moving right puts more of the portfolio into it; higher means more "
             "diversified. The dotted line is your portfolio today and the shaded line is your amount. Where a "
             "line flattens, adding more brings little extra diversification.",
    "cost": "Each dot is one ETF. Right = more diversified; higher = more expected return kept. "
            "Filled dots are not beaten by another option on both; hollow dots are.",
    "protect": "Each dot is one ETF. Right = more diversified; higher = a shallower fall in the worse of the two "
               "crises. A dot far right but below zero added diversification without softening both crashes. "
               "Filled dots are not beaten by another option on both.",
    "picks": "Any US stock or ETF, including ones you already hold. Each is tested with the amount on the right "
             "and ranked against the default diversifiers.",
    "amount": "New money put into each ETF, one at a time. Your current portfolio value comes from the sidebar. "
              "Buying with new money gives the same mix as moving that share out of your current holdings.",
    "quick": "Common picks to add with one click.",
    "verdict": "Whether this purchase spreads your risk over more independent bets or concentrates it, and whether "
               "it would have softened both COVID 2020 and the 2022 sell-off. The numbers are in the table below.",
    "alternative": "The default diversifier that improves the portfolio most for the same amount, preferring ones "
                   "that softened both past crises.",
    "compare": "Your picks next to the default diversifiers. Each row puts the same amount into one ETF and "
               "measures the portfolio again. Green means better than today, red means worse. Hover over a column "
               "name for its definition.",
    "robust_btn": "Resamples the daily history in 21-day blocks, 200 times, and ranks the candidates again each "
                  "time. A long bar means that ETF stays the top pick across many versions of history.",
    "before_after": "Your portfolio before and after the purchase, in each period. For the historical overview, "
                    "worst fall is measured over your whole lookback window. Dollar figures use the portfolio value "
                    "before and after the purchase, so a bigger portfolio can lose more dollars even when its "
                    "percentage fall is smaller.",
}

COLUMN_HELP = {
    "Rank": "Order under the selected Rank by option.",
    "ETF": "★ marks the ETFs you picked. 'already held' means you own it or a near-identical fund, "
           "so its weight goes up.",
    "Bets after": "Equivalent bets after adding this ETF.",
    "Δ bets": "Change in equivalent bets compared with now. Higher means more diversified.",
    "Δ vol": "Change in volatility (annualised), in percentage points. Negative means a calmer portfolio.",
    "Δ return": "Change in expected return per year (CAPM), in percentage points. Often negative, "
                         "because diversifiers usually have lower beta.",
    "Δ fall COVID": "Change in the worst fall during COVID 2020 with the same amount. Positive means shallower.",
    "Δ fall 2022": "Change in the worst fall during the 2022 sell-off with the same amount. Positive means shallower.",
    "Both crises": "The worst fall would have been shallower in both COVID 2020 and the 2022 sell-off.",
    "Period": "Historical lookback, COVID 2020 or the 2022 sell-off.",
    "Bets": "Equivalent bets before → after the purchase.",
    "Volatility": "Annualised volatility before → after the purchase.",
    "Worst fall": "Largest drop from a high point in the period, before → after the purchase.",
    "Worst fall in US$": "Worst fall × portfolio value. 'After' uses the larger portfolio value after the purchase.",
    "Expected return (CAPM)": "CAPM expected return per year before → after the purchase.",
}


GOOD, BAD = "#0ca30c", "#d03b3b"
# +1: a positive change is better; -1: a negative change is better
DIRECTION = {"Δ bets": 1, "Δ return": 1, "Δ fall COVID": 1, "Δ fall 2022": 1, "Δ vol": -1}
SERIOUS = "#ec835a"
VERDICTS = {"concentrates": ("Concentrates your portfolio", "⚠", SERIOUS),
            "diversifies": ("Adds diversification", "✓", GOOD),
            "little_change": ("Little change in diversification", "•", MUTED)}


def _tone(value, direction: int) -> str:
    text = str(value).strip()
    if not text or text in ("n/a", "0.0 pts", "+0.00", "0.00 pts"):
        return ""
    sign = -1 if text.startswith("-") else 1 if text.startswith("+") else 0
    if sign == 0:
        return ""
    return f"color: {GOOD if sign * direction > 0 else BAD}"


def _table(df: pd.DataFrame):
    widths = {"Rank": "small", "ETF": 290}
    config = {c: st.column_config.Column(f"{c} ⓘ", help=COLUMN_HELP[c], width=widths.get(c))
              for c in df.columns if c in COLUMN_HELP}
    styler = df.style
    for col, direction in DIRECTION.items():
        if col in df.columns:
            styler = styler.map(lambda v, d=direction: _tone(v, d), subset=[col])
    st.dataframe(styler, hide_index=True, width="stretch", column_config=config)


def _defaults() -> dict:
    return wf.CANDIDATE_SETS[wf.DEFAULT_REGION]


@st.cache_data(show_spinner=False)
def _universe() -> pd.DataFrame:
    return edata.load_universe().set_index("yf_symbol")


@st.cache_data(show_spinner="Testing each candidate ETF...")
def _run(holdings: tuple, lookback: int, mrp: float, rf: float | None, candidates: tuple, alpha: float,
         _store, engine_revision: str = ""):
    alphas = (alpha, *[a for a in wf.ALPHA_GRID if not np.isclose(a, alpha)])
    return wf.run_whatif(dict(holdings), lookback_years=lookback, mrp=mrp, rf=rf,
                         candidates=dict(candidates), alphas=alphas, store=_store)


@st.cache_data(show_spinner="Resampling history...")
def _stability(holdings: tuple, lookback: int, mrp: float, rf: float | None, candidates: tuple,
               alpha: float, _store, engine_revision: str = ""):
    return wf.ranking_stability(_run(holdings, lookback, mrp, rf, candidates, alpha, _store, engine_revision),
                                alpha=alpha)


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
        st.slider("Market risk premium", 0.02, 0.08, wf.MRP_DEFAULT, 0.005, format="%.3f", key="wi_mrp",
                  help=HELP["mrp"])
        if st.toggle("Set risk-free rate myself", key="wi_own_rf", help=HELP["own_rf"]):
            st.number_input("Risk-free rate", value=RF_INPUT_DEFAULT, step=0.0025, format="%.4f", key="wi_rf", help=HELP["rf"])


def _ticker_picker(label: str, key: str, help=None):
    uni = _universe()
    options = list(uni.index) + [t for t in st.session_state.get(key, []) if t not in uni.index]
    fmt = lambda t: f"{t} · {uni.at[t, 'name']}" if t in uni.index else t  # noqa: E731
    return st.multiselect(label, options, key=key, format_func=fmt, accept_new_options=True,
                          max_selections=MAX_PICKS, placeholder="Search by ticker or name", help=help)


QUICK_PICKS = ["QQQ", "VOO", "GLD", "IEF", "DBC"]


def _add_quick_pick():
    pick = st.session_state.get("wi_quick")
    picks = st.session_state.setdefault("wi_picks", [])
    if pick and pick not in picks and len(picks) < MAX_PICKS:
        st.session_state["wi_picks"] = [*picks, pick]
    st.session_state["wi_quick"] = None


def _rf_setting() -> tuple[float, float | None]:
    """CAPM inputs set in the assumptions popover next to Rank by."""
    mrp = st.session_state.get("wi_mrp", wf.MRP_DEFAULT)
    rf = st.session_state.get("wi_rf", RF_INPUT_DEFAULT) if st.session_state.get("wi_own_rf") else None
    return mrp, rf


def render_whatif(holdings: dict, lookback: int, store, value: float = 10_000, engine_revision: str = ""):
    key = tuple(sorted(holdings.items()))
    c1, c2 = st.columns([3, 1])
    with c1:
        picked = _ticker_picker(f"ETFs you are considering (up to {MAX_PICKS})", "wi_picks", help=HELP["picks"])
    with c2:
        amount = st.number_input("Amount (US$)", min_value=50.0, value=2_000.0, step=500.0, key="wi_amount",
                                 help=HELP["amount"])
    st.pills("Quick picks", QUICK_PICKS, key="wi_quick", on_change=_add_quick_pick, help=HELP["quick"])
    alpha = wf.purchase_alpha(value, amount)
    st.caption(f"Portfolio value {_money(value)} → {_money(value + amount)}: each ETF is tested at "
               f"{alpha:.1%} of the new total.")

    picks = list(dict.fromkeys(edata.normalize_ticker(t) for t in picked))
    candidates = {**_defaults(), **{t: wf.ticker_name(t) for t in picks if t not in _defaults()}}
    mrp, rf = _rf_setting()
    try:
        wi = _run(key, lookback, mrp, rf, tuple(candidates.items()), alpha, store, engine_revision)
    except ValueError as e:
        st.error(str(e))
        return
    missing = [t for t in picks if t not in wi.candidates]
    if missing:
        st.warning(f"No price data found for {', '.join(missing)}; left out.")
    picks = [t for t in picks if t in wi.candidates]

    if picks:
        _render_picks(wi, picks, amount, value)
    _render_comparison(wi, picks, key, lookback, store, engine_revision, amount, alpha, rf)


def badge(text: str, icon: str, color: str):
    st.markdown(f"<span style='display:inline-block;padding:2px 10px;margin:2px 6px 2px 0;border-radius:12px;"
                f"border:1px solid {color};color:{color};font-weight:600'>{icon} {text}</span>",
                unsafe_allow_html=True)


def verdict_badge(verdict: str):
    badge(*VERDICTS[verdict])


def _render_picks(wi, picks: list, amount: float, value: float):
    st.subheader("Your picks", help=HELP["verdict"], anchor=False)
    cards = st.columns(len(picks))
    for col, k in zip(cards, picks):
        s = wf.purchase_summary(wf.purchase_from(wi, k, amount, value))
        with col.container(border=True):
            st.markdown(f"**{k}** · {s['name']}")
            if s["already_held"]:
                st.caption(f"You {s['already_held'].replace('already held', 'already hold it')}.")
            verdict_badge(s["verdict"])
            deeper = [WINDOWS[w] for w, p in s["periods"].items()
                      if w != "normal" and p["worst_fall_after"] is not None
                      and p["worst_fall_after"] < p["worst_fall_before"]]
            if s["softened_all_crises"]:
                badge("Softened both past crises", "✓", GOOD)
            elif deeper:
                badge(f"Deeper fall in {' and '.join(deeper)}", "⚠", SERIOUS)
            alt = s["best_alternative"]
            if alt and alt["is_better"]:
                st.markdown(f"Better for the same {_money(amount)}: **{alt['ticker']}** ({alt['name']})"
                            + (", which softened both crises." if alt["softened_all_crises"] else "."),
                            help=HELP["alternative"])

    with st.expander("Before and after, in detail", icon=":material/table_view:"):
        k = picks[0] if len(picks) == 1 else (
            st.segmented_control("ETF", picks, default=picks[0], key="wi_detail_pick") or picks[0])
        pc = wf.purchase_from(wi, k, amount, value)
        rows = []
        for win, r in pc.impact.iterrows():
            ok = not np.isnan(r["bets_after"])
            rows.append({
                "Period": WINDOWS.get(win, r["label"]),
                "Bets": f"{r['bets_before']:.2f} → {r['bets_after']:.2f}" if ok else "n/a",
                "Volatility": f"{r['vol_before']:.1%} → {r['vol_after']:.1%}" if ok else "n/a",
                "Worst fall": f"{r['dd_before']:.1%} → {r['dd_after']:.1%}" if ok else "n/a",
                "Worst fall in US$": f"{r['loss_before']:,.0f} → {r['loss_after']:,.0f}" if ok else "n/a",
                "Expected return (CAPM)": (f"{r['er_before']:.2%} → {r['er_after']:.2%}"
                                           if win == "normal" and ok else ""),
            })
        st.markdown(f"**{k}: before and after**", help=HELP["before_after"])
        _table(pd.DataFrame(rows))
        st.download_button("Download check record", wf.purchase_record(pc),
                           file_name=f"pretrade_check_{k}_{pd.Timestamp.now():%Y%m%d}.txt",
                           mime="text/plain", icon=":material/download:", key="wi_record",
                           help="Inputs, impact in each period and the alternatives considered, as a plain-text record.")


def _render_comparison(wi, picks: list, key: tuple, lookback: int, store, rev: str, amount: float,
                       alpha: float, rf: float | None):
    st.subheader(f"Compared with other diversifiers, {_money(amount)} each", help=HELP["compare"], anchor=False)
    labels = {"d_bets": "Most added bets", "worst_crisis": "Most crisis-proof"}
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    with c1:
        rank_by = st.segmented_control("Rank by", labels, format_func=labels.get, default="d_bets",
                                       key="wi_rank_by", help=HELP["rank"]) or "d_bets"
    with c2:
        _assumptions()

    cands = list(wi.candidates)
    style = _styles(cands)
    color = {k: v["color"] for k, v in style.items()}
    base = wi.base["normal"]
    m = base.metrics
    tbl = wf.ranking_table(wi, "normal", alpha, rank_by)
    cur_er = wi.grid[(wi.grid.window == "normal") & (wi.grid.candidate == "(current)")]["er"].iloc[0]

    st.markdown(f"Your portfolio now: **{m['effective_bets']:.2f}** equivalent bets · "
                f"**{m['portfolio_vol']:.1%}** volatility · **{cur_er:.2%}** expected return (CAPM)",
                help=HELP["now"])
    st.caption(f"{base.start:%d %b %Y} - {base.end:%d %b %Y} ({base.n_obs:,} trading days). "
               f"Risk-free rate {wi.rf:.2%}"
               + ("" if rf is not None else f" ({wf.RF_PROXY[wf.DEFAULT_REGION]} trailing 12-month return)")
               + f", market risk premium {wi.mrp:.1%}.")
    gone = sorted({k.split(":")[1] for k in wi.unavailable})
    if gone:
        st.caption(f"Not testable in every period (no prices yet): {', '.join(gone)}.")
    for n in wi.notes.get("normal", []):
        st.caption(n)

    if tbl.empty:
        st.info("No candidate could be tested.")
        return
    st.info(wf.summary_sentence(wi, tbl, "normal", alpha, rank_by, amount=amount), icon=":material/lightbulb:")
    cols = {
        "Rank": tbl["rank"],
        "ETF": [("★ " if k in picks else "") + f"{k} · {wi.candidates.get(k, k)}"
                + (f" ({tbl.at[k, 'held']})" if tbl.at[k, "held"] else "") for k in tbl.index],
        "Bets after": tbl["bets"].map(lambda x: f"{x:.2f}"),
        "Δ bets": tbl["d_bets"].map(lambda x: f"{x:+.2f}"),
        "Δ vol": tbl["d_vol"].map(_pts),
        "Δ return": tbl["d_er"].map(lambda x: _pts(x, 2)),
    }
    for col in [col for col in tbl.columns if col.startswith("dd_") and col[3:] in SHORT]:
        cols[f"Δ fall {SHORT[col[3:]]}"] = tbl[col].map(lambda x: "n/a" if pd.isna(x) else _pts(x))
    cols["Both crises"] = tbl["all_crises"].map({True: "Yes", False: ""})
    _table(pd.DataFrame(cols))

    left, right = st.columns(2)
    with left:
        st.subheader("How much to add?", help=HELP["curve"], anchor=False)
        g = wi.grid[wi.grid.window == "normal"]
        fig, ends = go.Figure(), {}
        for k in cands:
            d = g[g.candidate == k].sort_values("alpha")
            if d.empty:
                continue
            x = np.r_[0.0, d["alpha"]]
            y = np.r_[m["effective_bets"], d["bets"]]
            fig.add_scatter(x=x, y=y, name=k, mode="lines+markers",
                            line=dict(width=3 if k in picks else 2, color=color[k], dash=style[k]["dash"]),
                            marker=dict(size=8, symbol=style[k]["symbol"]),
                            hovertemplate=f"{k}: %{{y:.2f}} bets at %{{x:.0%}}<extra></extra>")
            ends[k] = (x[-1], y[-1])
        for k, (x_end, _), y_lab in _spread_labels(ends):
            fig.add_annotation(x=x_end, y=y_lab, text=k, showarrow=False, xanchor="left", xshift=8)
        fig.add_hline(y=m["effective_bets"], line_dash="dot", line_color=MUTED,
                      annotation_text="now", annotation_position="bottom left")
        fig.add_vline(x=alpha, line_width=8, line_color=MUTED, opacity=0.25,
                      annotation_text=_money(amount).replace("\\", ""), annotation_position="bottom right")
        x_max = max(max(wf.ALPHA_GRID), alpha) + 0.07
        _layout(fig, 400, hovermode="x unified", showlegend=True,
                xaxis=dict(title="Share of the portfolio in the new ETF", tickformat=".0%", range=[-0.01, x_max]),
                yaxis=dict(title="Equivalent bets"))
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

    with right:
        crisis = rank_by == "worst_crisis"
        pts = tbl.dropna(subset=["worst_crisis"]) if crisis else tbl
        if crisis:
            pts = wf.mark_pareto(pts, "d_bets", "worst_crisis")
        y_col, y_title = (("worst_crisis", "Change in fall, worse crisis (pts, + = shallower)") if crisis
                          else ("d_er", "Change in expected return (pts a year)"))
        st.subheader("Did it protect in both crises?" if crisis else "What does it cost?",
                     help=HELP["protect" if crisis else "cost"], anchor=False)
        fig2 = go.Figure()
        for k, r in pts.iterrows():
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

    with st.expander("Is the ranking robust?", icon=":material/insights:"):
        if st.button("Run robustness check", key="wi_boot", help=HELP["robust_btn"]):
            stab = _stability(key, lookback, wi.mrp, rf, tuple(wi.candidates.items()), alpha, store, rev)
            fig3 = go.Figure(go.Bar(x=stab.values, y=stab.index, orientation="h",
                                    marker=dict(color=[color.get(k, MUTED) for k in stab.index]),
                                    text=[f"{v:.0%}" for v in stab.values], textposition="outside",
                                    hovertemplate="%{y}: ranked first in %{x:.0%} of resamples<extra></extra>"))
            _layout(fig3, 280, xaxis=dict(tickformat=".0%", range=[0, 1.12],
                                          title="Share of resamples ranked first"),
                    yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig3, width="stretch", config=PLOTLY_CONFIG)
