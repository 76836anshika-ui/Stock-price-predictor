"""Self-contained HTML report (one file, shareable, no server)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import viz
from .pipeline import Experiment

CSS = """
body{font-family:Inter,system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#111827}
h1{margin-bottom:.2rem}.sub{color:#6b7280;margin-top:0}
.banner{padding:.8rem 1rem;border-radius:8px;background:#eff6ff;border:1px solid #bfdbfe;margin:1rem 0}
.warn{background:#fef3c7;border-color:#fcd34d}
.tbl{border-collapse:collapse;font-size:.85rem;margin:.5rem 0}
.tbl th,.tbl td{border-bottom:1px solid #e5e7eb;padding:.3rem .7rem;text-align:right}
.tbl th:first-child,.tbl td:first-child{text-align:left}
h2{margin-top:2rem;border-bottom:2px solid #e5e7eb;padding-bottom:.3rem}
small{color:#6b7280}
"""


def build_report(
    exp: Experiment,
    ticker: str,
    fan: pd.DataFrame | None = None,
    path: str | Path = "reports/report.html",
) -> Path:
    figs = [
        ("Forecast vs actual", viz.fig_forecast(exp)),
        ("Model comparison", viz.fig_model_compare(exp)),
        ("Strategy backtest", viz.fig_equity(exp)),
        ("Interval calibration", viz.fig_calibration(exp)),
        ("Regime analysis", viz.fig_regime(exp)),
        ("Feature importance", viz.fig_importance(exp)),
    ]
    if fan is not None:
        figs.insert(0, ("Forecast fan", viz.fig_fan(exp.df, fan, alpha=exp.alpha)))

    synthetic = exp.source.startswith("synthetic")
    warn = (
        '<div class="banner warn"><b>Synthetic data.</b> This run used simulated prices '
        "(no network or by request). Numbers demonstrate the method, not any real market.</div>"
        if synthetic else ""
    )
    parts = [
        f"<html><head><meta charset='utf-8'><title>{ticker} forecast report</title>",
        f"<style>{CSS}</style></head><body>",
        f"<h1>{ticker} forecast report</h1>",
        f"<p class='sub'>Data: {exp.source} · {exp.df.index[0]:%Y-%m-%d} to {exp.df.index[-1]:%Y-%m-%d} "
        f"· horizon {exp.horizon}d · walk-forward, retrained every {exp.settings['step']} bars</p>",
        warn,
        f"<div class='banner'>{viz.headline(exp)}</div>",
        "<h2>Forecast accuracy</h2>", viz.metrics_table_html(exp),
        "<small>R²oos is measured against the random-walk forecast (0% return). "
        "DM p-value: one-sided Diebold-Mariano test that the model beats it. "
        "Dir. acc is compared with the hit rate a skill-free forecaster would get.</small>",
        "<h2>Strategy backtest</h2>", viz.strategy_table_html(exp),
        f"<small>Long/flat on the forecast sign, {exp.settings['cost_bps']:g} bps per unit turnover, "
        "fills at the close, no borrow fees or taxes: an optimistic upper bound.</small>",
    ]
    for i, (title, fig) in enumerate(figs):
        if fig is None:
            continue
        parts.append(f"<h2>{title}</h2>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False))
    parts.append("<p><small>Educational project. Not investment advice.</small></p></body></html>")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
