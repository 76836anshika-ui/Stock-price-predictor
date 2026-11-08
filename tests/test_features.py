import numpy as np
import pandas as pd
import pytest

from stockpred.features import build_dataset, compute_features


def test_no_lookahead(prices):
    """Features at row t must be identical when all later bars are deleted."""
    full = compute_features(prices)
    for cut in (400, 700, 1000):
        trunc = compute_features(prices.iloc[:cut])
        pd.testing.assert_frame_equal(
            full.iloc[:cut], trunc, check_exact=False, rtol=1e-9, atol=1e-12
        )


def test_future_shock_does_not_move_past_features(prices):
    shocked = prices.copy()
    shocked.iloc[-30:, shocked.columns.get_loc("Close")] *= 3
    a, b = compute_features(prices), compute_features(shocked)
    pd.testing.assert_frame_equal(a.iloc[:-30], b.iloc[:-30])


@pytest.mark.parametrize("h", [1, 5])
def test_target_alignment(prices, h):
    ds = build_dataset(prices, h)
    lc = np.log(prices["Close"])
    t = ds.X.index[10]
    i = prices.index.get_loc(t)
    assert ds.y.loc[t] == pytest.approx(lc.iloc[i + h] - lc.iloc[i])
    assert ds.target_dates[10] == prices.index[i + h]
    assert ds.last_date == prices.index[-1]
    assert not ds.X.isna().any().any()


def test_live_row_is_latest_bar(prices):
    ds = build_dataset(prices, 1)
    assert ds.X_live.index[0] == prices.index[-1]
