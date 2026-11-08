"""Plotly figures shared by the HTML report and the Streamlit dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .models import BASELINES
from .pipeline import Experiment

PALETTE = {
    "price": "#1f2937",
    "pred": "#2563eb",
    "band": "rgba(37,99,235,0.15)",
    "bench": "#9ca3af",
    "good": "#059669",
    "bad": "#dc2626",
}
_LAYOUT = dict(template="plotly_white", margin=dict(l=40, r=20, t=50, b=40), hovermode="x unified")


def fig_forecast(exp: Experiment, model: str | None = None, last_n: int = 250) -> go.Figure:
    """Out-of-sample predicted price vs actual, with the conformal band."""
    model = model or exp.best_model
    f = exp.results[model].frame.tail(last_n)
    x = f["target_date"]
    fig = go.Figure()
    fig.add_scatter(x=x, y=f["price_hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(
        x=x, y=f["price_lo"], fill="tonexty", fillcolor=PALETTE["band"], mode="lines", line=dict(width=0),
        name=f"{int(100 * (1 - exp.alpha))}% conformal interval", hoverinfo="skip",
    )
    fig.add_scatter(x=x, y=f["price_true"], name="Actual", line=dict(color=PALETTE["price"], width=2))
    fig.add_scatter(
        x=x, y=f["price_pred"], name=f"Predicted ({model})",
        line=dict(color=PALETTE["pred"], width=1.5, dash="dot"),
    )
    fig.update_layout(title=f"Out-of-sample forecasts ({exp.horizon}-day horizon)", **_LAYOUT)
    return fig


def fig_equity(exp: Experiment) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    b = exp.benchmark
    fig.add_scatter(x=b.equity.index, y=b.equity, name="Buy & hold", line=dict(color=PALETTE["bench"]), row=1, col=1)
    for name, s in exp.strategies.items():
        w = 3 if name == exp.best_model else 1.2
        fig.add_scatter(x=s.equity.index, y=s.equity, name=name, line=dict(width=w), row=1, col=1)
    best = exp.strategies[exp.best_model]
    fig.add_scatter(x=b.drawdown.index, y=100 * b.drawdown, name="B&H drawdown", line=dict(color=PALETTE["bench"]), showlegend=False, row=2, col=1)
    fig.add_scatter(x=best.drawdown.index, y=100 * best.drawdown, name=f"{best.name} drawdown", line=dict(color=PALETTE["pred"]), showlegend=False, row=2, col=1)
    fig.update_yaxes(title_text="Growth of 1", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    cost = exp.settings.get("cost_bps", 0)
    fig.update_layout(title=f"Walk-forward strategy vs buy & hold (net of {cost:g} bps/trade)", **_LAYOUT)
    return fig


def fig_model_compare(exp: Experiment) -> go.Figure:
    m = exp.metrics.drop(index="naive")
    colors = [PALETTE["good"] if v > 0 else PALETTE["bad"] for v in m["r2_oos_pct"]]
    fig = go.Figure(go.Bar(x=m.index, y=m["r2_oos_pct"], marker_color=colors, text=m["r2_oos_pct"].round(2), textposition="outside"))
    fig.add_hline(y=0, line_color=PALETTE["bench"], annotation_text="random-walk benchmark")
    fig.update_layout(title="Out-of-sample R² vs random walk (%, positive = beats benchmark)", yaxis_title="R²_oos %", **{**_LAYOUT, "hovermode": "closest"})
    return fig


def fig_importance(exp: Experiment, top: int = 12) -> go.Figure | None:
    if exp.importance is None:
        return None
    imp = exp.importance.head(top)[::-1]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=PALETTE["pred"]))
    fig.update_layout(title=f"Permutation importance ({exp.best_model}): % MSE added when scrambled", **{**_LAYOUT, "hovermode": "closest"})
    return fig


def fig_calibration(exp: Experiment, model: str | None = None) -> go.Figure:
    """Rolling empirical coverage of the intervals against the nominal level."""
    f = exp.results[model or exp.best_model].frame.dropna(subset=["lo"])
    inside = ((f["y_true"] >= f["lo"]) & (f["y_true"] <= f["hi"])).astype(float)
    roll = 100 * inside.rolling(126, min_periods=60).mean()
    fig = go.Figure()
    fig.add_scatter(x=roll.index, y=roll, name="Rolling coverage (6m)", line=dict(color=PALETTE["pred"]))
    fig.add_hline(y=100 * (1 - exp.alpha), line_dash="dash", line_color=PALETTE["bench"], annotation_text="nominal")
    fig.update_layout(title="Interval calibration", yaxis_title="Coverage %", yaxis_range=[60, 100], **_LAYOUT)
    return fig


def fig_fan(df: pd.DataFrame, fan: pd.DataFrame, history: int = 90, alpha: float = 0.10) -> go.Figure:
    """Price history followed by the multi-horizon forecast fan."""
    hist = df["Close"].tail(history)
    t0, c0 = hist.index[-1], float(hist.iloc[-1])
    dates = [t0, *fan["date"]]
    hi = [c0, *fan["price_hi"]]
    lo = [c0, *fan["price_lo"]]
    mid = [c0, *fan["price"]]
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist, name="Close", line=dict(color=PALETTE["price"], width=2))
    fig.add_scatter(x=dates, y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=dates, y=lo, fill="tonexty", fillcolor=PALETTE["band"], mode="lines", line=dict(width=0), name=f"{int(100 * (1 - alpha))}% interval", hoverinfo="skip")
    fig.add_scatter(x=dates, y=mid, name="Forecast", line=dict(color=PALETTE["pred"], dash="dot"), mode="lines+markers")
    fig.update_layout(title="Forecast fan (direct multi-horizon, conformal intervals)", **_LAYOUT)
    return fig


def fig_regime(exp: Experiment) -> go.Figure:
    """Model error by volatility regime: is the forecast only 'working' in calm markets?"""
    f = exp.results[exp.best_model].frame
    b = exp.results["naive"].frame
    tercile = pd.qcut(f["sigma"], 3, labels=["low vol", "mid vol", "high vol"])
    mse_m = ((f["y_true"] - f["y_pred"]) ** 2).groupby(tercile, observed=True).mean()
    mse_b = (b["y_true"] ** 2).groupby(tercile, observed=True).mean()
    gain = 100 * (1 - mse_m / mse_b)
    fig = go.Figure(go.Bar(x=gain.index.astype(str), y=gain.values, marker_color=[PALETTE["good"] if v > 0 else PALETTE["bad"] for v in gain.values]))
    fig.update_layout(title="R²_oos by volatility regime (%)", **{**_LAYOUT, "hovermode": "closest"})
    return fig


def metrics_table_html(exp: Experiment) -> str:
    cols = {
        "rmse_ret_pct": "RMSE (ret %)", "r2_oos_pct": "R²oos %", "mape_pct": "MAPE %",
        "dir_acc": "Dir. acc", "dir_chance": "Chance", "p_better": "DM p-value", "coverage_pct": "Cover %",
    }
    t = exp.metrics.reindex(columns=list(cols)).rename(columns=cols).copy()
    for c in ("Dir. acc", "Chance"):
        t[c] = 100 * t[c]
    return t.round(3).to_html(na_rep="-", classes="tbl")


def strategy_table_html(exp: Experiment) -> str:
    rows = {"buy & hold": exp.benchmark.stats, **{k: v.stats for k, v in exp.strategies.items()}}
    t = pd.DataFrame(rows).T[["cagr_pct", "vol_pct", "sharpe", "max_dd_pct", "exposure_pct", "turnover_per_year"]]
    t.columns = ["CAGR %", "Vol %", "Sharpe", "Max DD %", "Exposure %", "Turnover/yr"]
    return t.astype(float).round(2).to_html(classes="tbl")


def headline(exp: Experiment) -> str:
    """One honest sentence about whether the best model really beats the random walk.

    The winner is picked *after* seeing results, so its raw p-value is optimistic. We report
    a Bonferroni-adjusted p-value over every model that competed for the title.
    """
    m = exp.metrics.loc[exp.best_model]
    p = m.get("p_better", np.nan)
    k = max(1, len([n for n in exp.metrics.index if n not in BASELINES]))
    p_adj = min(1.0, p * k) if np.isfinite(p) else np.nan
    verdict = (
        "statistically significant" if np.isfinite(p_adj) and p_adj < 0.05
        else "NOT statistically significant"
    )
    return (
        f"Best model: {exp.best_model}. Out-of-sample R² vs random walk = {m['r2_oos_pct']:.2f}% "
        f"({verdict}: Diebold-Mariano p = {p:.3f}, {p_adj:.3f} after adjusting for the "
        f"{k} models compared)."
    )
