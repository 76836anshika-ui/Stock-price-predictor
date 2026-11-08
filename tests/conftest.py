import pytest

from stockpred.data import make_synthetic_prices


@pytest.fixture(scope="session")
def prices():
    return make_synthetic_prices(n_days=1400, seed=11)
