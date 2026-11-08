"""Model zoo. Every model predicts the h-day log return; price forecasts are derived from it.

Baselines come first on purpose: a forecaster is only interesting if it beats ``naive``
(the random walk, i.e. "tomorrow's price = today's price"), which is notoriously hard.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_INFO = {
    "naive": "Random walk: predicts a 0% return (price stays flat). The benchmark to beat.",
    "hist_mean": "Predicts the historical average return (drift-only baseline).",
    "linear": "Ordinary least squares on winsorised, standardised features.",
    "ridge": "Ridge regression, penalty chosen by time-series cross-validation.",
    "rf": "Random forest with heavy regularisation (large leaves, feature subsampling).",
    "hgb": "Gradient-boosted shallow trees with L2 shrinkage.",
    "ensemble": "Equal-weight average of ridge, random forest and gradient boosting.",
}
BASELINES = {"naive", "hist_mean"}


class ZeroReturn(BaseEstimator, RegressorMixin):
    """Predicts zero return regardless of input (the random-walk forecast)."""

    def fit(self, X, y=None):
        self.n_features_in_ = np.shape(X)[1]
        return self

    def predict(self, X):
        return np.zeros(len(X))


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each feature to its training-set quantiles (tames fat-tailed return features)."""

    def __init__(self, q: float = 0.01):
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.q, axis=0)
        self.hi_ = np.nanquantile(X, 1 - self.q, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def _linear_pipeline(reg) -> Pipeline:
    return Pipeline([("clip", Winsorizer()), ("scale", StandardScaler()), ("reg", reg)])


def _ridge() -> Pipeline:
    return _linear_pipeline(RidgeCV(alphas=np.logspace(1, 5, 13), cv=TimeSeriesSplit(4)))


def _rf(seed: int, fast: bool) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=60 if fast else 150,
        min_samples_leaf=30,
        max_features=0.4,
        n_jobs=-1,
        random_state=seed,
    )


def _hgb(seed: int, fast: bool) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=60 if fast else 150,
        max_depth=3,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=seed,
    )


def make_model(name: str, seed: int = 42, fast: bool = False):
    """Build a fresh, unfitted estimator by registry name."""
    if name == "naive":
        return ZeroReturn()
    if name == "hist_mean":
        return DummyRegressor(strategy="mean")
    if name == "linear":
        return _linear_pipeline(LinearRegression())
    if name == "ridge":
        return _ridge()
    if name == "rf":
        return _rf(seed, fast)
    if name == "hgb":
        return _hgb(seed, fast)
    if name == "ensemble":
        return VotingRegressor(
            [("ridge", _ridge()), ("rf", _rf(seed, fast)), ("hgb", _hgb(seed, fast))]
        )
    raise ValueError(f"Unknown model '{name}'. Choose from {sorted(MODEL_INFO)}.")
