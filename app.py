"""Interactive dashboard:  streamlit run app.py"""

from __future__ import annotations

import warnings

import streamlit as st

from stockpred import viz
from stockpred.data import DataError, load_prices
from stockpred.models import MODEL_INFO
from stockpred.pipeline import forecast_fan, run_experiment

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")
st.title("📈 Stock Price Predictor")
st.caption("Walk-forward validated, benchmarked against the random walk, with conformal prediction intervals.")

with st.sidebar:
    st.header("Setup")
    ticker = st.text_input("Ticker", "AAPL").strip().upper()
    source = st.selectbox("Data source", ["auto", "yfinance", "synthetic"], help="auto = Yahoo, falling back to simulated data offline")
    start = st.text_input("Start date", "2012-01-01")
    horizon = st.slider("Horizon (trading days)", 1, 10, 1)
    options = [m for m in MODEL_INFO if m != "naive"]
    models = st.multiselect("Models", options, default=["ridge", "hgb", "ensemble"], format_func=lambda m: f"{m}")
    coverage = st.select_slider("Interval coverage", [0.80, 0.90, 0.95], value=0.90)
    cost = st.slider("Transaction cost (bps)", 0.0, 25.0, 5.0)
    mode = st.radio("Strategy", ["long_only", "long_short"], horizontal=True)
    fast = st.checkbox("Fast mode (smaller ensembles)", True)
    run = st.button("Run experiment", type="primary", use_container_width=True)
    for m in models:
        st.caption(f"**{m}**: {MODEL_INFO[m]}")


@st.cache_data(show_spinner=False)
def _load(ticker, start, source):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = load_prices(ticker, start, None, source)
    return df, [str(w.message) for w in caught]


@st.cache_data(show_spinner="Running walk-forward validation...")
def _run(df, models, horizon, alpha, cost, mode, fast):
    exp = run_experiment(df, models=list(models), horizon=horizon, alpha=alpha, cost_bps=cost, mode=mode, fast=fast)
    fan = forecast_fan(df, exp.best_model, alpha=alpha, fast=fast)
    return exp, fan


if run or "ran" in st.session_state:
    st.session_state["ran"] = True
    try:
        df, warns = _load(ticker, start, source)
        for w in warns:
            st.warning(w)
        exp, fan = _run(df, tuple(models), horizon, 1 - coverage, cost, mode, fast)
    except (DataError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    if exp.source.startswith("synthetic"):
        st.info("Showing **synthetic** data: numbers illustrate the method, not a real market.")
    st.success(viz.headline(exp))

    last = float(df["Close"].iloc[-1])
    one = fan.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last close", f"{last:,.2f}")
    c2.metric(f"Next-day forecast ({exp.best_model})", f"{one['price']:,.2f}", f"{one['pred_return_pct']:+.2f}%")
    c3.metric(f"{int(coverage * 100)}% interval", f"{one['price_lo']:,.1f} - {one['price_hi']:,.1f}")
    best = exp.metrics.loc[exp.best_model]
    c4.metric("R²oos vs random walk", f"{best['r2_oos_pct']:.2f}%")

    t1, t2, t3, t4, t5 = st.tabs(["Forecast", "Accuracy", "Backtest", "Diagnostics", "Data"])
    with t1:
        st.plotly_chart(viz.fig_fan(df, fan, alpha=1 - coverage), use_container_width=True)
        st.dataframe(fan[["date", "pred_return_pct", "price", "price_lo", "price_hi"]].round(2))
        st.plotly_chart(viz.fig_forecast(exp), use_container_width=True)
    with t2:
        st.plotly_chart(viz.fig_model_compare(exp), use_container_width=True)
        st.markdown(viz.metrics_table_html(exp), unsafe_allow_html=True)
        st.caption("A model must beat the random walk (R²oos > 0, small DM p-value) to claim any skill. High R² on raw prices is meaningless.")
    with t3:
        st.plotly_chart(viz.fig_equity(exp), use_container_width=True)
        st.markdown(viz.strategy_table_html(exp), unsafe_allow_html=True)
        st.caption("Optimistic: fills at the close, no slippage beyond the cost setting, no borrow fees or taxes.")
    with t4:
        a, b = st.columns(2)
        a.plotly_chart(viz.fig_calibration(exp), use_container_width=True)
        b.plotly_chart(viz.fig_regime(exp), use_container_width=True)
        fi = viz.fig_importance(exp)
        if fi is not None:
            st.plotly_chart(fi, use_container_width=True)
    with t5:
        st.dataframe(df.tail(200))
        st.download_button("Download forecasts (CSV)", exp.results[exp.best_model].frame.to_csv(), f"{ticker}_forecasts.csv")
else:
    st.info("Pick a ticker and press **Run experiment**.")
st.caption("Educational project. Not investment advice.")
