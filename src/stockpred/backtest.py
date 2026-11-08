"""Walk-forward validation, conformal prediction intervals and a cost-aware strategy backtest."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import Dataset
from .metrics import performance_stats
from .models import make_model


@dataclass
class WalkForwardResult:
    model: str
    horizon: int
    alpha: float
    frame: pd.DataFrame
    n_refits: int


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1-alpha) quantile used by split conformal prediction."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))  # rank of the order statistic (1-indexed)
    return float(s[min(k, n) - 1])


def walk_forward(
    ds: Dataset,
    model: str | Callable[[], object] = "ridge",
    initial_train: int = 504,
    step: int = 21,
    alpha: float = 0.10,
    cal_window: int = 250,
    min_cal: int = 60,
    max_train: int | None = None,
    seed: int = 42,
    fast: bool = False,
) -> WalkForwardResult:
    """Expanding-window (or rolling, via ``max_train``) walk-forward evaluation.

    Leakage control: a model refit at row ``s`` may only train on rows whose *label* was
    already observable at that time, i.e. ``pos_j + h <= pos_s``. For a 5-day horizon that
    means the last 4 rows before ``s`` are excluded, because their labels reach into the test
    period. Intervals are volatility-normalised split-conformal: the score of a past forecast
    is ``|error| / sigma_t`` and the band is ``pred +/- q * sigma_t``, so it widens in
    turbulent markets while keeping ~(1-alpha) empirical coverage.
    """
    h, n = ds.horizon, len(ds.X)
    if n < initial_train + step:
        raise ValueError(
            f"Only {n} usable rows; need at least initial_train + step = {initial_train + step}. "
            "Use a longer date range or reduce initial_train."
        )
    name = model if isinstance(model, str) else getattr(model, "__name__", "custom")
    factory = (lambda: make_model(model, seed=seed, fast=fast)) if isinstance(model, str) else model

    preds = np.full(n, np.nan)
    n_refits = 0
    for s in range(initial_train, n, step):
        k = int(np.searchsorted(ds.pos, ds.pos[s] - h, side="right"))  # labels known at s
        a = max(0, k - max_train) if max_train else 0
        est = factory()
        est.fit(ds.X.iloc[a:k], ds.y.iloc[a:k])
        e = min(s + step, n)
        preds[s:e] = est.predict(ds.X.iloc[s:e])
        n_refits += 1

    y = ds.y.to_numpy()
    sigma = ds.sigma.to_numpy()
    score = np.abs(y - preds) / sigma  # NaN before initial_train

    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    for i in range(initial_train, n):
        kmax = int(np.searchsorted(ds.pos, ds.pos[i] - h, side="right"))
        cal = score[max(initial_train, kmax - cal_window) : kmax]
        if len(cal) >= min_cal:
            q = conformal_quantile(cal, alpha) * sigma[i]
            lo[i], hi[i] = preds[i] - q, preds[i] + q

    sl = slice(initial_train, n)
    close = ds.close.to_numpy()
    frame = pd.DataFrame(
        {
            "target_date": ds.target_dates[sl],
            "close": close[sl],
            "y_true": y[sl],
            "y_pred": preds[sl],
            "lo": lo[sl],
            "hi": hi[sl],
            "sigma": sigma[sl],
            "score": score[sl],
            "next_ret": ds.next_ret.to_numpy()[sl],
        },
        index=ds.X.index[sl],
    )
    frame["price_true"] = frame["close"] * np.exp(frame["y_true"])
    frame["price_pred"] = frame["close"] * np.exp(frame["y_pred"])
    frame["price_lo"] = frame["close"] * np.exp(frame["lo"])
    frame["price_hi"] = frame["close"] * np.exp(frame["hi"])
    return WalkForwardResult(name, h, alpha, frame, n_refits)


# --------------------------------------------------------------------------- strategy
@dataclass
class StrategyResult:
    name: str
    equity: pd.Series
    drawdown: pd.Series
    position: pd.Series
    stats: dict


def run_strategy(
    frame: pd.DataFrame,
    mode: str = "long_only",
    cost_bps: float = 5.0,
    threshold: float = 0.0,
    name: str = "strategy",
) -> StrategyResult:
    """Trade the sign of the forecast at the close, hold one day, pay costs on turnover.

    Optimistic assumptions (stated in the README): fills at the close price, no slippage
    beyond ``cost_bps`` per unit of turnover, no short borrow fees, no taxes.
    """
    signal = frame["y_pred"]
    if mode == "long_only":
        pos = (signal > threshold).astype(float)
    elif mode == "long_short":
        pos = np.sign(signal.where(signal.abs() > threshold, 0.0))
    else:
        raise ValueError("mode must be 'long_only' or 'long_short'")
    simple = np.expm1(frame["next_ret"])
    turnover = pos.diff().abs().fillna(pos.abs())
    net = pos * simple - turnover * cost_bps / 1e4
    equity = (1 + net).cumprod()
    s = performance_stats(net)
    s["exposure_pct"] = 100 * float((pos != 0).mean())
    s["turnover_per_year"] = float(turnover.sum() / (len(net) / 252))
    return StrategyResult(name, equity, equity / equity.cummax() - 1, pos, s)


def buy_and_hold(frame: pd.DataFrame, name: str = "buy & hold") -> StrategyResult:
    simple = np.expm1(frame["next_ret"])
    equity = (1 + simple).cumprod()
    s = performance_stats(simple)
    s["exposure_pct"] = 100.0
    s["turnover_per_year"] = 0.0
    return StrategyResult(name, equity, equity / equity.cummax() - 1, pd.Series(1.0, index=simple.index), s)
