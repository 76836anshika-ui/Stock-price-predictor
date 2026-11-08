"""Price data acquisition.

Three sources are supported:

* ``yfinance``  - live daily bars from Yahoo Finance (cached on disk)
* ``csv``       - any local OHLCV file
* ``synthetic`` - a reproducible regime-switching market simulator with an optional,
  *planted* predictable component. It exists so the whole pipeline (and the CI test
  suite) runs offline, and so the leakage "canary" tests have a ground truth.

``source="auto"`` tries Yahoo first and falls back to synthetic data with a loud warning.
Every returned frame carries ``df.attrs["source"]`` so reports can label what they show.
"""

from __future__ import annotations

import hashlib
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class DataError(RuntimeError):
    """Raised when price data cannot be loaded or fails validation."""


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, sorted, tz-naive OHLCV frame or raise ``DataError``."""
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise DataError(f"Missing required columns: {missing}")
    out = df[OHLCV].apply(pd.to_numeric, errors="coerce").copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out.index = idx.normalize()
    out.index.name = "Date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out["Volume"] = out["Volume"].fillna(0.0)
    if out.empty:
        raise DataError("No valid rows after cleaning.")
    if (out[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise DataError("Non-positive prices found; cannot compute log returns.")
    prices = out[["Open", "High", "Low", "Close"]]
    out["High"] = prices.max(axis=1)  # repair inconsistent bars
    out["Low"] = prices.min(axis=1)
    return out


# --------------------------------------------------------------------------- synthetic
def make_synthetic_prices(
    n_days: int = 2600,
    seed: int = 7,
    start: str = "2015-01-05",
    s0: float = 100.0,
    signal_strength: float = 3.0,
) -> pd.DataFrame:
    """Simulate daily OHLCV bars with realistic stylised facts.

    Includes volatility regimes, fat tails (Student-t), volatility clustering, gaps and
    volume that co-moves with volatility. ``signal_strength`` scales a *planted*
    predictable component (slow mean-reverting drift + weak AR(1)):

    * ``0``  -> a pure random walk. No model can have a true edge. Used as a leakage canary.
    * ``3``  -> faint but real signal (default demo, out-of-sample R^2 well under 1%).
    * ``8+`` -> an easy problem that any working pipeline must detect.
    """
    rng = np.random.default_rng(seed)

    # 2-state Markov volatility regime (calm / turbulent), smoothed to cluster like GARCH
    p_stay = np.array([0.985, 0.95])
    state = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        state[t] = state[t - 1] if rng.random() < p_stay[state[t - 1]] else 1 - state[t - 1]
    sigma = pd.Series(np.where(state == 0, 0.008, 0.022)).ewm(span=8).mean().to_numpy()

    eps = rng.standard_t(5, n_days) / np.sqrt(5 / 3)  # unit-variance fat tails

    # planted signal: slow OU drift + AR(1) on the demeaned return
    theta, amp, phi = 1 / 60, 0.0004 * signal_strength, 0.03 * signal_strength
    drift = np.zeros(n_days)
    for t in range(1, n_days):
        drift[t] = drift[t - 1] * (1 - theta) + amp * np.sqrt(1 - (1 - theta) ** 2) * rng.normal()

    mu = 0.0003
    ret = np.zeros(n_days)
    for t in range(1, n_days):
        ret[t] = mu + drift[t - 1] + phi * (ret[t - 1] - mu) + sigma[t] * eps[t]

    close = s0 * np.exp(np.cumsum(ret))
    prev_close = np.concatenate([[s0], close[:-1]])
    gap = 0.3 * ret + rng.normal(0, 0.25, n_days) * sigma
    open_ = prev_close * np.exp(gap)
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    high = body_hi * np.exp(np.abs(rng.normal(0, 0.5, n_days)) * sigma)
    low = body_lo * np.exp(-np.abs(rng.normal(0, 0.5, n_days)) * sigma)
    volume = 1e7 * np.exp(0.3 * rng.normal(size=n_days) + 25 * np.abs(ret))

    idx = pd.bdate_range(start=start, periods=n_days, name="Date")
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx
    )
    df.attrs["source"] = "synthetic"
    return df


def _ticker_seed(ticker: str) -> int:
    return int(hashlib.sha256(ticker.upper().encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- yfinance
def _download_yfinance(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise DataError("yfinance is not installed (pip install yfinance).") from exc
    try:
        raw = yf.download(
            ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False
        )
    except Exception as exc:  # network / parsing problems
        raise DataError(f"Yahoo Finance download failed for {ticker}: {exc}") from exc
    if raw is None or raw.empty:
        raise DataError(f"Yahoo Finance returned no data for '{ticker}'.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw


def _cached_yfinance(
    ticker: str, start: str | None, end: str | None, cache_dir: Path, max_age_h: float
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{ticker.upper()}_{start or 'min'}_{end or 'max'}.csv"
    if path.exists() and (time.time() - path.stat().st_mtime) / 3600 < max_age_h:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    raw = _download_yfinance(ticker, start, end)
    raw.to_csv(path)
    return raw


# --------------------------------------------------------------------------- public API
def load_prices(
    ticker: str = "AAPL",
    start: str | None = "2012-01-01",
    end: str | None = None,
    source: str = "auto",
    csv_path: str | Path | None = None,
    cache_dir: str | Path = ".cache/prices",
    cache_hours: float = 12.0,
) -> pd.DataFrame:
    """Load validated daily OHLCV data. See module docstring for ``source`` options."""
    source = source.lower()
    if source not in {"auto", "yfinance", "csv", "synthetic"}:
        raise DataError(f"Unknown source '{source}'.")

    if source == "csv":
        if csv_path is None:
            raise DataError("source='csv' requires csv_path.")
        df = validate_ohlcv(pd.read_csv(csv_path, index_col=0, parse_dates=True))
        df.attrs["source"] = f"csv:{Path(csv_path).name}"
        return df

    if source in {"auto", "yfinance"}:
        try:
            df = validate_ohlcv(_cached_yfinance(ticker, start, end, Path(cache_dir), cache_hours))
            df.attrs["source"] = "yfinance"
            return df
        except DataError as exc:
            if source == "yfinance":
                raise
            warnings.warn(
                f"{exc} Falling back to SYNTHETIC data - results are for demonstration only.",
                stacklevel=2,
            )

    df = make_synthetic_prices(seed=_ticker_seed(ticker))
    df.attrs["source"] = "synthetic"
    return df
