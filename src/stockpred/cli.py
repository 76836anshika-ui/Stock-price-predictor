"""Command line interface: ``stockpred AAPL --horizon 1 --models ridge,hgb,ensemble``."""

from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

from .data import DataError, load_prices
from .models import MODEL_INFO
from .pipeline import forecast_fan, run_experiment
from .report import build_report
from .viz import headline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stockpred", description=__doc__)
    p.add_argument("ticker", nargs="?", default="AAPL")
    p.add_argument("--source", default="auto", choices=["auto", "yfinance", "csv", "synthetic"])
    p.add_argument("--csv", help="path to OHLCV csv (with --source csv)")
    p.add_argument("--start", default="2012-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--horizon", type=int, default=1, help="forecast horizon in trading days")
    p.add_argument("--models", default="ridge,hgb,ensemble",
                   help=f"comma list from: {', '.join(m for m in MODEL_INFO if m != 'naive')}")
    p.add_argument("--alpha", type=float, default=0.10, help="interval miscoverage (0.10 = 90%%)")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--mode", default="long_only", choices=["long_only", "long_short"])
    p.add_argument("--initial-train", type=int, default=504)
    p.add_argument("--step", type=int, default=21)
    p.add_argument("--fast", action="store_true", help="smaller ensembles for quick runs")
    p.add_argument("--out", default="reports/report.html")
    p.add_argument("--no-report", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pd.set_option("display.width", 200)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = load_prices(args.ticker, args.start, args.end, args.source, args.csv)
        for w in caught:
            print(f"WARNING: {w.message}", file=sys.stderr)
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        exp = run_experiment(
            df, models=models, horizon=args.horizon, initial_train=args.initial_train,
            step=args.step, alpha=args.alpha, cost_bps=args.cost_bps, mode=args.mode,
            fast=args.fast, progress=lambda s: print(f"  ... {s}", file=sys.stderr),
        )
    except (DataError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"\n{args.ticker} | data: {exp.source} | {len(df)} bars | horizon {args.horizon}d")
    cols = ["rmse_ret_pct", "r2_oos_pct", "dir_acc", "dir_chance", "p_better", "coverage_pct"]
    print(exp.metrics[cols].round(3).to_string())
    print("\n" + headline(exp))

    fan = None
    try:
        fan = forecast_fan(df, exp.best_model, alpha=args.alpha, fast=args.fast,
                           initial_train=args.initial_train, step=args.step)
        print(f"\nForecast from {df.index[-1]:%Y-%m-%d} close {df['Close'].iloc[-1]:.2f}:")
        show = fan[["date", "pred_return_pct", "price", "price_lo", "price_hi"]].copy()
        show["date"] = show["date"].dt.strftime("%Y-%m-%d")
        print(show.round(2).to_string())
    except ValueError as exc:
        print(f"(fan skipped: {exc})", file=sys.stderr)

    if not args.no_report:
        path = build_report(exp, args.ticker, fan, args.out)
        print(f"\nReport written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
