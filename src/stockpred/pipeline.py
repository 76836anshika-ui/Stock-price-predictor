"""High-level orchestration shared by the CLI, the HTML report and the Streamlit dashboard."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .backtest import (
    StrategyResult,
    WalkForwardResult,
    buy_and_hold,
    conformal_quantile,
    run_strategy,
    walk_forward,
)
from .features import Dataset, build_dataset
from .metrics import evaluate_frame
from .models import BASELINES, make_model


@dataclass
class Experiment:
    df: pd.DataFrame
    source: str
    horizon: int
    alpha: float
    dataset: Dataset
    results: dict[str, WalkForwardResult]
    metrics: pd.DataFrame
    strategies: dict[str, StrategyResult]
    benchmark: StrategyResult
    best_model: str
    importance: pd.Series | None = None
    settings: dict = field(default_factory=dict)


def run_experiment(
    df: pd.DataFrame,
    models: Sequence[str] = ("naive", "ridge", "hgb", "ensemble"),
    horizon: int = 1,
    initial_train: int = 504,
    step: int = 21,
    alpha: float = 0.10,
    cost_bps: float = 5.0,
    mode: str = "long_only",
    fast: bool = False,
    seed: int = 42,
    with_importance: bool = True,
    progress: Callable[[str], None] | None = None,
) -> Experiment:
    """Run every model through the same walk-forward protocol and score them side by side."""
    names = ["naive", *[m for m in models if m != "naive"]]
    ds = build_dataset(df, horizon)

    results: dict[str, WalkForwardResult] = {}
    for m in names:
        if progress:
            progress(f"walk-forward: {m}")
        results[m] = walk_forward(
            ds, m, initial_train=initial_train, step=step, alpha=alpha, seed=seed, fast=fast
        )

    bench = results["naive"].frame
    rows = {}
    for m, res in results.items():
        rows[m] = evaluate_frame(res.frame, horizon, None if m == "naive" else bench)
    metrics = pd.DataFrame(rows).T

    contenders = [m for m in names if m not in BASELINES]
    best = (
        min(contenders, key=lambda m: metrics.loc[m, "rmse_ret_pct"]) if contenders else "naive"
    )

    strategies = {
        m: run_strategy(res.frame, mode=mode, cost_bps=cost_bps, name=m)
        for m, res in results.items()
        if m != "naive"
    }
    importance = None
    if with_importance and best not in BASELINES:
        if progress:
            progress(f"permutation importance: {best}")
        importance = feature_importance(ds, best, initial_train=initial_train, seed=seed, fast=fast)

    return Experiment(
        df=df,
        source=str(df.attrs.get("source", "unknown")),
        horizon=horizon,
        alpha=alpha,
        dataset=ds,
        results=results,
        metrics=metrics,
        strategies=strategies,
        benchmark=buy_and_hold(bench),
        best_model=best,
        importance=importance,
        settings={
            "initial_train": initial_train,
            "step": step,
            "cost_bps": cost_bps,
            "mode": mode,
            "fast": fast,
            "seed": seed,
        },
    )


def feature_importance(
    ds: Dataset,
    model: str,
    initial_train: int = 504,
    test_frac: float = 0.25,
    n_repeats: int = 5,
    seed: int = 42,
    fast: bool = False,
) -> pd.Series:
    """Permutation importance on a held-out final slice (model-agnostic, no extra deps)."""
    n = len(ds.X)
    cut = max(initial_train, int(n * (1 - test_frac)))
    k = int(np.searchsorted(ds.pos, ds.pos[cut] - ds.horizon, side="right"))
    est = make_model(model, seed=seed, fast=fast).fit(ds.X.iloc[:k], ds.y.iloc[:k])
    r = permutation_importance(
        est,
        ds.X.iloc[cut:],
        ds.y.iloc[cut:],
        scoring="neg_mean_squared_error",
        n_repeats=n_repeats,
        random_state=seed,
    )
    # express as % of out-of-sample MSE added when the feature is scrambled
    base_mse = float(np.mean((ds.y.iloc[cut:] - est.predict(ds.X.iloc[cut:])) ** 2))
    imp = pd.Series(100 * r.importances_mean / base_mse, index=ds.X.columns)
    return imp.sort_values(ascending=False)


def forecast_fan(
    df: pd.DataFrame,
    model: str = "ridge",
    horizons: Sequence[int] = (1, 2, 3, 5, 10),
    alpha: float = 0.10,
    initial_train: int = 504,
    step: int = 21,
    cal_window: int = 250,
    seed: int = 42,
    fast: bool = False,
) -> pd.DataFrame:
    """Direct multi-horizon forecast from the latest bar, with conformal intervals.

    One model is trained per horizon (direct strategy: no error compounding from recursion).
    Interval half-widths come from that horizon's walk-forward out-of-sample scores.
    """
    rows = []
    for h in horizons:
        ds = build_dataset(df, h)
        wf = walk_forward(
            ds, model, initial_train=initial_train, step=step, alpha=alpha, seed=seed, fast=fast
        )
        est = make_model(model, seed=seed, fast=fast).fit(ds.X, ds.y)
        pred = float(est.predict(ds.X_live)[0])
        q = conformal_quantile(wf.frame["score"].dropna().to_numpy()[-cal_window:], alpha)
        half = q * ds.sigma_live
        c0 = ds.last_close
        rows.append(
            {
                "horizon": h,
                "date": pd.bdate_range(ds.last_date, periods=h + 1)[-1],
                "pred_return_pct": 100 * (np.exp(pred) - 1),
                "price": c0 * np.exp(pred),
                "price_lo": c0 * np.exp(pred - half),
                "price_hi": c0 * np.exp(pred + half),
                "last_close": c0,
            }
        )
    return pd.DataFrame(rows).set_index("horizon")
