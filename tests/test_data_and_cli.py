import pandas as pd
import pytest

from stockpred.cli import main
from stockpred.data import DataError, load_prices, make_synthetic_prices, validate_ohlcv


def test_synthetic_is_deterministic_and_valid():
    a, b = make_synthetic_prices(300, seed=1), make_synthetic_prices(300, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert (a["High"] >= a[["Open", "Close"]].max(axis=1) - 1e-9).all()
    assert (a["Low"] <= a[["Open", "Close"]].min(axis=1) + 1e-9).all()


def test_validate_rejects_bad_input():
    with pytest.raises(DataError):
        validate_ohlcv(pd.DataFrame({"Close": [1.0]}))
    bad = make_synthetic_prices(50)
    bad.iloc[3, bad.columns.get_loc("Close")] = -1
    with pytest.raises(DataError):
        validate_ohlcv(bad)


def test_csv_roundtrip(tmp_path):
    df = make_synthetic_prices(120)
    p = tmp_path / "x.csv"
    df.to_csv(p)
    out = load_prices(source="csv", csv_path=p)
    assert len(out) == 120 and out.attrs["source"].startswith("csv")


def test_cli_end_to_end(tmp_path, capsys):
    out = tmp_path / "r.html"
    code = main(["DEMO", "--source", "synthetic", "--models", "ridge", "--fast",
                 "--out", str(out), "--initial-train", "400", "--step", "63"])
    assert code == 0 and out.exists() and "Diebold" in out.read_text()
    assert "Forecast from" in capsys.readouterr().out


def test_cli_reports_errors(capsys):
    assert main(["X", "--source", "csv"]) == 2
