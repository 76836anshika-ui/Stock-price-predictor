"""Walk-forward protocol tests, including the leakage canaries."""

import numpy as np
import pytest

from stockpred.backtest import conformal_quantile, run_strategy, walk_forward
from stockpred.data import make_synthetic_prices
from stockpred.features import build_dataset
from stockpred.metrics import diebold_mariano, evaluate_frame


class Spy:
    """Linear model that records the maximum training row it ever saw."""

    log: list = []

    def fit(self, X, y):
        Spy.log.append(X.index.max())
        self.m = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


@pytest.mark.parametrize("h", [1, 5])
def test_training_labels_are_observable(prices, h):
    """A refit at test row s must not train on a label that ends after s's forecast origin."""
    ds = build_dataset(prices, h)
    Spy.log = []
    walk_forward(ds, Spy, initial_train=300, step=50)
    origins = ds.X.index[300 :: 50]
    for last_train, origin in zip(Spy.log, origins, strict=False):
        i = prices.index.get_loc(last_train)
        assert i + h <= prices.index.get_loc(origin)


def test_random_walk_has_no_edge():
    """Canary: on a pure random walk, no model may show significant out-of-sample skill."""
    df = make_synthetic_prices(n_days=1500, seed=3, signal_strength=0.0)
    ds = build_dataset(df, 1)
    bench = walk_forward(ds, "naive", initial_train=400, step=42).frame
    res = walk_forward(ds, "ridge", initial_train=400, step=42).frame
    m = evaluate_frame(res, 1, bench)
    assert m["r2_oos_pct"] < 1.0
    assert not m["p_better"] < 0.01


def test_planted_signal_is_detected():
    """Canary: with a strong planted signal the pipeline must find it (it is not blind)."""
    df = make_synthetic_prices(n_days=1800, seed=5, signal_strength=10.0)
    ds = build_dataset(df, 1)
    bench = walk_forward(ds, "naive", initial_train=400, step=42).frame
    res = walk_forward(ds, "ridge", initial_train=400, step=42).frame
    m = evaluate_frame(res, 1, bench)
    assert m["r2_oos_pct"] > 1.0
    assert m["p_better"] < 0.05


def test_conformal_coverage_close_to_nominal(prices):
    ds = build_dataset(prices, 1)
    f = walk_forward(ds, "ridge", initial_train=400, step=42, alpha=0.10).frame
    cover = evaluate_frame(f, 1, None)["coverage_pct"]
    assert 84 <= cover <= 96


def test_conformal_quantile_basic():
    s = np.arange(1, 101, dtype=float)
    assert conformal_quantile(s, 0.10) == 91.0


def test_naive_predicts_zero_and_price_flat(prices):
    ds = build_dataset(prices, 1)
    f = walk_forward(ds, "naive", initial_train=400, step=42).frame
    assert (f["y_pred"] == 0).all()
    np.testing.assert_allclose(f["price_pred"], f["close"])


def test_too_little_data_raises(prices):
    ds = build_dataset(prices.iloc[:500], 1)
    with pytest.raises(ValueError, match="usable rows"):
        walk_forward(ds, "ridge", initial_train=504)


def test_strategy_costs_reduce_returns(prices):
    ds = build_dataset(prices, 1)
    f = walk_forward(ds, "ridge", initial_train=400, step=42).frame
    free = run_strategy(f, cost_bps=0).stats["total_return_pct"]
    costly = run_strategy(f, cost_bps=50).stats["total_return_pct"]
    assert costly < free


def test_diebold_mariano_directional():
    rng = np.random.default_rng(0)
    good = rng.normal(0, 1, 500)
    bad = good + rng.normal(0, 1, 500)
    r = diebold_mariano(bad**2, good**2)
    assert r["p_better"] < 0.01
