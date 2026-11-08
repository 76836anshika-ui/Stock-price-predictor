"""Forecast evaluation with statistical rigour: error metrics, significance tests, coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano(loss_bench: np.ndarray, loss_model: np.ndarray, h: int = 1) -> dict:
    """Diebold-Mariano test (Harvey-Leybourne-Newbold small-sample correction).

    H0: equal predictive accuracy. ``p_better`` is one-sided: small values mean the model
    beats the benchmark. Overlapping h-step forecasts are handled with a Bartlett HAC variance.
    """
    d = np.asarray(loss_bench, float) - np.asarray(loss_model, float)
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return {"dm_stat": np.nan, "p_better": np.nan, "p_two_sided": np.nan}
    dbar = d.mean()
    dc = d - dbar
    var = np.dot(dc, dc) / n
    for k in range(1, h):
        w = 1 - k / h
        var += 2 * w * np.dot(dc[k:], dc[:-k]) / n
    var = max(var / n, 1e-18)
    stat = dbar / np.sqrt(var)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {
        "dm_stat": float(stat),
        "p_better": float(stats.t.sf(stat, df=n - 1)),
        "p_two_sided": float(2 * stats.t.sf(abs(stat), df=n - 1)),
    }


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Hit rate of the predicted sign, tested against what chance would deliver.

    Chance is not 50%: if 54% of days are up and the model says "up" 70% of the time, a
    skill-free forecaster hits ``0.54*0.70 + 0.46*0.30 = 51.6%``. The binomial test uses that.
    """
    up_true, up_pred = np.asarray(y_true) > 0, np.asarray(y_pred) > 0
    n = len(up_true)
    if n == 0 or np.ptp(y_pred) == 0:
        return {"dir_acc": np.nan, "dir_chance": np.nan, "dir_p": np.nan}
    hits = int((up_true == up_pred).sum())
    b, q = up_true.mean(), up_pred.mean()
    chance = float(b * q + (1 - b) * (1 - q))
    p = float(stats.binomtest(hits, n, chance, alternative="greater").pvalue)
    return {"dir_acc": hits / n, "dir_chance": chance, "dir_p": p}


def evaluate_frame(frame: pd.DataFrame, horizon: int, benchmark: pd.DataFrame | None) -> dict:
    """All headline numbers for one model's walk-forward frame."""
    y, p = frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()
    sse, sse_naive = np.sum((y - p) ** 2), np.sum(y**2)
    price_err = frame["price_pred"] - frame["price_true"]

    out = {
        "rmse_ret_pct": 100 * np.sqrt(np.mean((y - p) ** 2)),
        "mae_ret_pct": 100 * np.mean(np.abs(y - p)),
        "r2_oos_pct": 100 * (1 - sse / sse_naive),
        "price_rmse": float(np.sqrt(np.mean(price_err**2))),
        "price_mae": float(np.mean(np.abs(price_err))),
        "mape_pct": float(100 * np.mean(np.abs(price_err) / frame["price_true"])),
        **directional_accuracy(y, p),
    }
    if benchmark is not None:
        loss_b = (benchmark["y_true"] - benchmark["y_pred"]).to_numpy() ** 2
        out.update(diebold_mariano(loss_b, (y - p) ** 2, h=horizon))
    ok = frame["lo"].notna()
    if ok.any():
        inside = (frame.loc[ok, "y_true"] >= frame.loc[ok, "lo"]) & (
            frame.loc[ok, "y_true"] <= frame.loc[ok, "hi"]
        )
        out["coverage_pct"] = float(100 * inside.mean())
        width = (frame.loc[ok, "price_hi"] - frame.loc[ok, "price_lo"]) / frame.loc[ok, "close"]
        out["pi_width_pct"] = float(100 * width.mean())
    return out


def performance_stats(net_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Annualised performance summary for a series of simple periodic returns."""
    r = net_returns.dropna()
    if len(r) < 2:
        return {}
    equity = (1 + r).cumprod()
    years = len(r) / periods_per_year
    vol = r.std() * np.sqrt(periods_per_year)
    sharpe = r.mean() / r.std() * np.sqrt(periods_per_year) if r.std() > 0 else np.nan
    dd = equity / equity.cummax() - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    return {
        "cagr_pct": 100 * cagr,
        "vol_pct": 100 * vol,
        "sharpe": sharpe,
        "sharpe_t": sharpe * np.sqrt(years) if np.isfinite(sharpe) else np.nan,
        "max_dd_pct": 100 * dd.min(),
        "total_return_pct": 100 * (equity.iloc[-1] - 1),
    }
