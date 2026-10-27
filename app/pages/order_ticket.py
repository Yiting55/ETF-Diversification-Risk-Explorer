"""Order ticket demo: the pre-trade diversification check as it would appear inside a broker's order flow."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.data import PriceStore, load_universe, normalize_ticker  # noqa: E402
from engine.whatif import check_purchase, purchase_record, purchase_summary  # noqa: E402

st.set_page_config(page_title="Order ticket", page_icon=":material/receipt_long:", layout="wide")

EXAMPLE = {"VOO": 30, "QQQ": 25, "VTI": 15, "NVDA": 10, "AGG": 10, "GLD": 10}
GOOD, SERIOUS, NEUTRAL = "#0ca30c", "#ec835a", "#8f8e88"


@st.cache_resource
def store() -> PriceStore:
    return PriceStore()


@st.cache_data(show_spinner=False)
def universe() -> pd.DataFrame:
    return load_universe().set_index("yf_symbol")


@st.cache_data(show_spinner="Running pre-trade check...")
def run_check(holdings: tuple, ticker: str, amount: float, value: float):
    pc = check_purchase(dict(holdings), ticker, amount, value, store=store())
    return purchase_summary(pc), purchase_record(pc)


def current_portfolio() -> tuple[dict, float, bool]:
    """Holdings and value entered on the main page, or the example portfolio."""
    saved = st.session_state.get("portfolio", {})
    h = {t: float(w) for t, w in saved.get("holdings", {}).items() if w > 0}
    if h:
        return h, float(saved["value"]), True
    return dict(EXAMPLE), 10_000.0, False


def switch_to(ticker: str):
    st.session_state["ot_symbol"] = ticker
    st.session_state["ot_reviewed"] = True


def reset_review():
    st.session_state["ot_reviewed"] = False


def badge(text: str, icon: str, color: str):
    st.markdown(f"<span style='display:inline-block;padding:4px 10px;border-radius:12px;"
                f"border:1px solid {color};color:{color};font-weight:600'>{icon} {text}</span>",
                unsafe_allow_html=True)


holdings, portfolio_value, from_main = current_portfolio()
uni = universe()

st.title("Order ticket")
st.caption("Demo brokerage · illustration of the pre-trade diversification check, not a real trading screen.")

left, right = st.columns([2, 3], gap="large")

with left:
    with st.container(border=True):
        st.subheader("New order", help="A simplified market order. The check runs when you review the order.")
        account_value = st.number_input("Account value (US$)", min_value=100.0, value=portfolio_value, step=1_000.0,
                                        key="ot_value", on_change=reset_review,
                                        help="Current value of the holdings below.")
        with st.expander(f"Holdings ({len(holdings)})"):
            st.dataframe(pd.DataFrame({"Weight": pd.Series(holdings) / sum(holdings.values())}).style.format("{:.1%}"),
                         width="stretch")
            st.caption("From your portfolio on the main page." if from_main else "Example portfolio.")
        options = list(uni.index)
        symbol = st.selectbox("Symbol", options, index=options.index("QQQ") if "QQQ" in options else 0,
                              key="ot_symbol", accept_new_options=True, on_change=reset_review,
                              format_func=lambda t: f"{t} · {uni.at[t, 'name']}" if t in uni.index else t,
                              help="Stock or ETF to buy.")
        st.segmented_control("Side", ["Buy"], default="Buy", disabled=True, key="ot_side",
                             help="This demo covers buy orders.")
        amount = st.number_input("Amount (US$)", min_value=50.0, value=2_000.0, step=500.0, key="ot_amount",
                                 on_change=reset_review, help="Dollar amount to invest.")
        if st.button("Review order", type="primary", width="stretch", icon=":material/fact_check:"):
            st.session_state["ot_reviewed"] = True

with right:
    if not st.session_state.get("ot_reviewed"):
        st.info("Fill in the order and select **Review order** to see its effect on your portfolio.",
                icon=":material/arrow_back:")
        st.stop()
    try:
        s, record = run_check(tuple(sorted(holdings.items())), normalize_ticker(symbol), amount, account_value)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    n, covid, y22 = s["periods"]["normal"], s["periods"].get("covid"), s["periods"].get("2022")
    with st.container(border=True):
        st.subheader("Pre-trade diversification check",
                     help="What this order does to your portfolio's diversification over the last five years "
                          "and in two past crashes, compared with putting the same amount elsewhere. "
                          "Based on historical prices; not investment advice.")
        st.markdown(f"**Buy {s['ticker']} · \\${s['amount']:,.0f}** "
                    f"({s['share_of_portfolio_after']:.1%} of the account after the order)"
                    + (f" · {s['already_held']}" if s["already_held"] else ""))
        d = s["change_in_bets"] or 0.0
        if d < -0.01:
            badge("Concentrates your portfolio", "⚠", SERIOUS)
        elif d > 0.01:
            badge("Adds diversification", "✓", GOOD)
        else:
            badge("Little change in diversification", "•", NEUTRAL)

        a, b, c = st.columns(3)
        a.metric("Equivalent bets", f"{n['bets_after']:.2f}",
                 delta=f"{n['bets_after'] - n['bets_before']:+.2f}",
                 help="How many independent, equally risky holdings the portfolio behaves like after the "
                      "order, compared with now. Higher is more diversified.")
        for col, p, title in [(b, covid, "COVID 2020 fall"), (c, y22, "2022 fall")]:
            if p and p["worst_fall_after"] is not None:
                col.metric(title, f"{p['worst_fall_after']:.1%}",
                           delta=f"{(p['worst_fall_after'] - p['worst_fall_before']) * 100:+.1f} pts",
                           help="Largest drop from a high point during that crash if you had held the portfolio "
                                "after the order, compared with now. A positive change means a smaller drop.")
            else:
                col.metric(title, "n/a", help="No prices for this period.")

        now = [f"{n['bets_before']:.2f} equivalent bets"] + [
            f"{t} fall {p['worst_fall_before']:.1%}" for p, t in [(covid, "COVID"), (y22, "2022")] if p]
        st.caption("Your portfolio now: " + " · ".join(now))
        both = s["softened_all_crises"]
        st.markdown(f"**Softened both past crises:** {'Yes' if both else 'No'}",
                    help="Whether this order would have made the worst fall smaller in both COVID 2020 "
                         "and the 2022 sell-off. Helping in one crisis is not enough: in a back-test on 500 "
                         "portfolios, TLT cushioned COVID but softened the 2022 fall in only 17% of them.")

        alt = s["best_alternative"]
        if alt and alt["ticker"] != s["ticker"] and (alt["bets_after"] > n["bets_after"] + 0.01 or (alt["softened_all_crises"] and not both)):
            with st.container(border=True):
                st.markdown(f"**Alternative:** the same \\${s['amount']:,.0f} in **{alt['ticker']}** ({alt['name']}) "
                            f"gives {alt['bets_after']:.2f} equivalent bets"
                            + (" and softened both past crises." if alt["softened_all_crises"] else "."),
                            help="The default diversifier that improves the portfolio most for the same amount, "
                                 "preferring ones that softened both past crises.")
                st.button(f"Switch order to {alt['ticker']}", on_click=switch_to, args=(alt["ticker"],),
                          icon=":material/swap_horiz:")

        x, y = st.columns(2)
        if x.button("Place order", type="primary", width="stretch", icon=":material/send:"):
            st.toast("Demo only: no order was placed.", icon=":material/info:")
        y.download_button("Download check record", record, width="stretch", icon=":material/download:",
                          file_name=f"pretrade_check_{s['ticker']}_{pd.Timestamp.now():%Y%m%d}.txt",
                          mime="text/plain",
                          help="Inputs, impact and alternatives considered, for the client file.")
    st.caption(f"Prices to {s['data_to']}. Historical comparison, not investment advice.")
