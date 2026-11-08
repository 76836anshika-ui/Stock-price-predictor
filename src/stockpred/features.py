"""Leakage-safe feature engineering and dataset construction.

Contract: every feature in row ``t`` is a function of bars at or before ``t`` only. This is
enforced by a test that recomputes features on truncated history and demands identical values
(``tests/test_features.py::test_no_lookahead``). The *only* place the future is touched is the
target, ``y_t = log(C_{t+h} / C_t)``, built in :func:`build_dataset`.

We model log returns rather than raw prices: prices are non-stationary, so a regression on price
levels mostly rediscovers "tomorrow looks like today" and reports a flattering R^2 of ~0.99.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DataError

VOL_COL = "vol_21"


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / down)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Technical / statistical features, all strictly causal. Warm-up rows contain NaN."""
    o, h, l, c, v = (df[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    lc = np.log(c)
    r = lc.diff()
    f: dict[str, pd.Series] = {}

    # short-term return memory + multi-horizon momentum
    for k in range(5):
        f[f"ret_lag{k}"] = r.shift(k)
    for k in (5, 10, 21, 63, 126):
        f[f"mom_{k}"] = lc - lc.shift(k)
    f["mom_126_skip21"] = lc.shift(21) - lc.shift(126)  # classic 12-1 style momentum

    # volatility
    for k in (5, 10, 21, 63):
        f[f"vol_{k}"] = r.rolling(k).std()
    f["vol_ratio_5_63"] = f["vol_5"] / f["vol_63"]
    hl = np.log(h / l)
    f["parkinson_21"] = np.sqrt((hl**2).rolling(21).mean() / (4 * np.log(2)))
    true_range = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_14_pct"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / c

    # trend
    smas = {n: c.rolling(n).mean() for n in (10, 20, 50, 200)}
    for n, s in smas.items():
        f[f"dist_sma_{n}"] = c / s - 1
    f["sma_50_200"] = smas[50] / smas[200] - 1

    # oscillators
    f["rsi_14"] = _rsi(c) / 100
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / c
    f["macd"] = macd
    f["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    sd20 = c.rolling(20).std()
    f["bb_pctb"] = (c - smas[20]) / (2 * sd20)
    f["bb_width"] = 4 * sd20 / smas[20]
    f["stoch_k"] = (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min())

    # bar shape
    f["gap"] = np.log(o / c.shift())
    f["intraday"] = np.log(c / o)
    f["range_pct"] = hl
    day_range = h - l
    f["close_loc"] = ((c - l) / day_range).where(day_range > 0, 0.5)

    # volume (skipped for instruments that do not report it, e.g. some indices / FX)
    if (v > 0).mean() > 0.9:
        vm, vs = v.rolling(20).mean(), v.rolling(20).std()
        f["volume_z_20"] = (v - vm) / vs.replace(0, np.nan)
        f["volume_ratio_5_20"] = v.rolling(5).mean() / vm

    # calendar (cyclical encodings)
    dow, month = df.index.dayofweek.to_numpy(), df.index.month.to_numpy()
    f["dow_sin"] = pd.Series(np.sin(2 * np.pi * dow / 5), index=df.index)
    f["dow_cos"] = pd.Series(np.cos(2 * np.pi * dow / 5), index=df.index)
    f["month_sin"] = pd.Series(np.sin(2 * np.pi * (month - 1) / 12), index=df.index)
    f["month_cos"] = pd.Series(np.cos(2 * np.pi * (month - 1) / 12), index=df.index)

    return pd.DataFrame(f, index=df.index).replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class Dataset:
    """Aligned modelling tables for one forecast horizon.

    ``X``/``y`` contain only rows whose label is known. ``pos`` gives each row's integer
    position in the original price index; the walk-forward engine uses it to guarantee a
    training label is never used before it would have been observable.
    """

    X: pd.DataFrame
    y: pd.Series  # log(C_{t+h}/C_t)
    close: pd.Series  # C_t
    next_ret: pd.Series  # log(C_{t+1}/C_t), realised return used by the strategy backtest
    sigma: pd.Series  # volatility scale for conformal normalisation
    pos: np.ndarray
    target_dates: pd.DatetimeIndex
    horizon: int
    X_live: pd.DataFrame  # features at the final bar (the forecast origin)
    sigma_live: float
    last_close: float
    last_date: pd.Timestamp


def build_dataset(df: pd.DataFrame, horizon: int = 1) -> Dataset:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    feats = compute_features(df)
    complete = feats.dropna()
    if complete.empty:
        raise DataError("Not enough history to compute features (need roughly 200+ bars).")
    if complete.index[-1] != df.index[-1]:
        raise DataError("Latest bar has missing features (check for NaN/zero prices at the end).")

    lc = np.log(df["Close"])
    y_all = (lc.shift(-horizon) - lc).dropna()
    next_all = (lc.shift(-1) - lc)
    position = pd.Series(np.arange(len(df)), index=df.index)

    keep = complete.index[complete.index.isin(y_all.index)]
    X = complete.loc[keep]
    pos = position.loc[keep].to_numpy()
    sigma_all = np.maximum(complete[VOL_COL] * np.sqrt(horizon), 1e-4)

    return Dataset(
        X=X,
        y=y_all.loc[keep],
        close=df["Close"].loc[keep],
        next_ret=next_all.loc[keep],
        sigma=sigma_all.loc[keep],
        pos=pos,
        target_dates=df.index[pos + horizon],
        horizon=horizon,
        X_live=complete.iloc[[-1]],
        sigma_live=float(sigma_all.iloc[-1]),
        last_close=float(df["Close"].iloc[-1]),
        last_date=df.index[-1],
    )
