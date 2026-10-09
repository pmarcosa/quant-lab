#!/usr/bin/env python3
"""What three execution rules are worth to the momentum strategy.

    python scripts/compare_execution.py

Measured against one baseline (decide on the weekly close, fill at the next
open, IBKR's tiered commissions):

- a minimum order size, as a fraction of equity and as a number of dollars;
- deciding on the close and filling at that close (a closing-auction order);
- deciding on the open and filling at that open.

Then the minimum order size together with the better of the two fills.

The two fills at the decision price both look ahead a little, and say so. The
close is not known when a closing-auction order has to be sent, a few minutes
before it. The open is not known at all before the opening auction prints it, so
"decide on the open, fill at the open" cannot be traded as simulated; it is here
as the upper bound of what removing the weekend between decision and fill could
be worth. It runs on bars rebuilt from one open to the next, so that the
unchanged strategy sees the Monday open as its latest price.

The headline table is on the calendar the live account rotates on. Each variant
is also run on the three other four-week calendars, because a difference that
holds on one rotation calendar and not on the others is timing luck. Every run is recorded in the research ledger: a comparison of rules is a
set of trials, and the Deflated Sharpe Ratio needs to know.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.execution import is_stop_order  # noqa: E402
from contracts.identifiers import InstrumentId, RunId, StrategyVersion  # noqa: E402
from contracts.temporal import BarInterval  # noqa: E402
from data.bitemporal import BitemporalStore  # noqa: E402
from engine.decide import SizingPolicy  # noqa: E402
from execution.simulated import IBKR_FIXED, IBKR_TIERED, CostModel  # noqa: E402
from risk.rules import (  # noqa: E402
    GrossExposureLimit,
    NetExposureLimit,
    ProtectiveStop,
    RiskSupervisor,
    ShortSales,
)
from runtime.research import (  # noqa: E402
    periodic_dates,
    periodic_returns,
    precompute_indicators,
    run_once,
    trial_label,
)
from runtime.wiring import load_market, universe_list  # noqa: E402
from strategies.momentum import (  # noqa: E402
    CADENCE_EPOCH,
    STRATEGY_NAME,
    MomentumParams,
    WeeklyMomentum,
)
from validation.ledger import ResearchLedger, Study  # noqa: E402
from validation.metrics import summarise  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "var" / "store"
STUDY = "execution-rules"
SKIPPED_AS_SMALL = "below the minimum order size"
#: A decision the live account rotated on (the close before Monday 2026-09-21).
#: The live calendar counts four weeks from there; the strategy's own calendar
#: counts from ``CADENCE_EPOCH``, and the two need not be the same one of four.
LIVE_ROTATION = datetime(2026, 9, 18, 21, 15, tzinfo=timezone.utc)
_FAR_FUTURE = datetime(2200, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class ShiftedMomentum(WeeklyMomentum):
    """The same rules on another of the rotation calendars.

    A four-week cadence has four calendars. The strategy trades one of them;
    the other three are the same strategy started a week, two or three later.
    """

    shift: int = 0

    @property
    def version(self) -> StrategyVersion:
        if self.shift == 0:
            return super().version
        from dataclasses import asdict

        return StrategyVersion.of(
            STRATEGY_NAME, asdict(self.params), code_version=f"{self.code_version}+shift{self.shift}"
        )

    def rotates_at(self, moment: datetime) -> bool:
        weeks = (moment - CADENCE_EPOCH).days // 7
        return (weeks - self.shift) % self.params.rebalance_weeks == 0


def open_to_open_store(source: Path, target: Path, symbols) -> None:
    """Weekly bars that run from one open to the next.

    Bar *k* keeps its open, closes at bar *k+1*'s open, and its range is widened
    to include that price. A strategy deciding on such a bar's close is deciding
    on the next week's open, and an order filled at that close is filled at that
    open. Each bar keeps its time stamps, so both stores share one schedule; the
    last bar has no open after it and is dropped.
    """
    original = BitemporalStore(source, "bars_1week")
    rebuilt = BitemporalStore(target, "bars_1week")
    for instrument in original.instruments():
        if symbols is not None and str(instrument) not in symbols:
            continue
        bars = original.as_of(instrument, _FAR_FUTURE)
        if len(bars) < 2:
            continue
        first_known = original.first_known(instrument, _FAR_FUTURE).reindex(bars.index)
        next_open = bars["open"].shift(-1)
        frame = pd.DataFrame({
            "event_time": bars.index,
            "available_time": first_known.to_numpy(),
            "open": bars["open"].to_numpy(),
            "high": np.maximum(bars["high"], next_open).to_numpy(),
            "low": np.minimum(bars["low"], next_open).to_numpy(),
            "close": next_open.to_numpy(),
            "volume": bars["volume"].to_numpy() if "volume" in bars else 0.0,
        }).iloc[:-1]
        rebuilt.append(instrument, frame)
    universe = pd.read_csv(source / "universe_weekly.csv")
    universe.to_csv(target / "universe_weekly.csv", index=False)


def newey_west_t(difference: np.ndarray) -> float:
    """t-statistic of a mean difference with Newey-West errors (Bartlett kernel)."""
    d = np.asarray(difference, dtype=float)
    n = d.size
    if n < 3 or np.allclose(d, 0.0):
        return 0.0
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    centred = d - d.mean()
    variance = float(centred @ centred) / n
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        variance += 2.0 * weight * float(centred[lag:] @ centred[:-lag]) / n
    if variance <= 0:
        return 0.0
    return float(d.mean() / np.sqrt(variance / n))


def describe(result, market, capital) -> dict:
    """The figures a variant is compared on."""
    stats = summarise(result.equity_curve(), periods_per_year=market.periods_per_year)
    fills = [f for f in result.all_fills() if not is_stop_order(f.client_order_id)]
    years = (result.steps[-1].marked_at - result.steps[0].decision.decision_time).days / 365.25
    held_back = sum(
        1 for step in result.steps for reason in step.decision.skipped.values()
        if reason == SKIPPED_AS_SMALL
    )
    invested = [step.gross_after for step in result.steps]
    return {
        "cagr": stats.cagr, "volatility": stats.volatility, "sharpe": stats.sharpe,
        "max_drawdown": stats.max_drawdown, "final_equity": stats.final_equity,
        "capital": capital, "years": years,
        "orders": len(fills), "orders_per_year": len(fills) / years,
        "stops_fired": result.stops_fired(),
        "commission_paid": sum(f.commission for f in result.all_fills()),
        "turnover_per_year": sum(result.turnover()) / years,
        "orders_held_back": held_back,
        "average_invested": float(np.mean(invested)),
    }


def fees_at_todays_size(result, account: float, share_price: float) -> dict:
    """What each commission model would charge an account that stays ``account``.

    Every rotation order of the run is rescaled to that account (its share of
    equity is kept) and priced under each plan at a typical share price. In
    percent of the account a year.
    """
    years = (result.steps[-1].marked_at - result.steps[0].decision.decision_time).days / 365.25
    plans = {
        "flat 10 bp": None,
        "ibkr-fixed": IBKR_FIXED.with_reference_price(share_price),
        "ibkr-tiered": IBKR_TIERED.with_reference_price(share_price),
    }
    totals = dict.fromkeys(plans, 0.0)
    sizes: list[float] = []
    previous = result.opening_equity
    for step in result.steps:
        for fill in step.fills:
            value = fill.quantity * fill.price * account / previous
            if value < 0.01:
                continue
            sizes.append(value)
            quantity = value / fill.price
            for name, plan in plans.items():
                totals[name] += (
                    value * 0.0010 if plan is None else plan.fee(quantity, fill.price, fill.side)
                )
        previous = step.equity_after
    sizes_array = np.asarray(sizes)
    return {
        "account": account, "share_price": share_price,
        "orders_per_year": len(sizes) / years,
        "median_order": float(np.median(sizes_array)),
        "orders_below_1000": float(np.mean(sizes_array < 1_000.0)),
        "percent_of_account_per_year": {k: 100.0 * v / account / years for k, v in totals.items()},
        "dollars_per_order": {k: v / len(sizes) for k, v in totals.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument("--ledger", default=str(ROOT / "state" / "research.jsonl"))
    parser.add_argument("--universe", default="momentum")
    parser.add_argument("--capital", type=float, default=16_729.0,
                        help="Opening equity: the account's size today")
    parser.add_argument("--min-order", type=float, default=1_000.0)
    parser.add_argument("--share-price", type=float, default=100.0,
                        help="Typical share price per-share fees are charged at")
    parser.add_argument("--stop", type=float, default=0.12)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--start", default=None)
    parser.add_argument("--out", default=str(ROOT / "state" / "scratch" / "compare_execution.json"))
    args = parser.parse_args(argv)

    store = Path(args.store)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else None
    universe = universe_list(args.universe)
    symbols = universe.symbols if universe else None
    params = MomentumParams(rebalance_weeks=4, top_n=4, lookback_weeks=13)
    fraction = args.min_order / args.capital
    live = ((LIVE_ROTATION - CADENCE_EPOCH).days // 7) % params.rebalance_weeks

    market = load_market(store, interval=BarInterval.WEEK, start=start, symbols=symbols)
    scratch = Path(tempfile.mkdtemp(prefix="ql-open-to-open-"))
    open_to_open_store(store, scratch, set(symbols) if symbols else None)
    opened = load_market(scratch, interval=BarInterval.WEEK, start=start, symbols=symbols)

    warmup = params.warmup_weeks + params.min_history_weeks
    # One window for every variant: the open-to-open bars end a week earlier.
    last = min(market.schedule[-1], opened.schedule[-1])
    schedules = {
        "close": [m for m in list(market.schedule)[warmup:] if m <= last],
        "open": [m for m in list(opened.schedule)[warmup:] if m <= last],
    }
    indicators = {
        "close": precompute_indicators(market, params),
        "open": precompute_indicators(opened, params),
    }
    markets = {"close": market, "open": opened}

    tiered = CostModel(slippage_bps=args.slippage_bps,
                       schedule=IBKR_TIERED.with_reference_price(args.share_price))
    plain = SizingPolicy(cash_buffer=0.01, min_trade_fraction=0.005)
    scaled = replace(plain, min_order_fraction=fraction)

    # name: (bars, fill at the decision, sizing, costs, opening equity)
    variants = {
        "baseline": ("close", False, plain, tiered, args.capital),
        "min order (share of equity)": ("close", False, scaled, tiered, args.capital),
        "min order (dollars)": ("close", False, replace(plain, min_order_value=args.min_order),
                                tiered, args.capital),
        "decide and fill at the close": ("close", True, plain, tiered, args.capital),
        "decide and fill at the open": ("open", True, plain, tiered, args.capital),
        "baseline, flat 10 bp": ("close", False, plain,
                                 CostModel(commission_bps=10.0, slippage_bps=args.slippage_bps),
                                 args.capital),
        "baseline, ibkr-fixed": ("close", False, plain,
                                 CostModel(slippage_bps=args.slippage_bps,
                                           schedule=IBKR_FIXED.with_reference_price(args.share_price)),
                                 args.capital),
        "baseline, tiered at stored prices": ("close", False, plain,
                                              CostModel(slippage_bps=args.slippage_bps,
                                                        schedule=IBKR_TIERED),
                                              args.capital),
    }
    on_every_calendar = ["baseline", "min order (share of equity)",
                         "decide and fill at the close", "decide and fill at the open"]

    ledger = ResearchLedger(Path(args.ledger))
    output: dict = {"settings": {
        "universe": universe.describe() if universe else "store",
        "window": f"{schedules['close'][0].date()}..{schedules['close'][-1].date()}",
        "params": {"top_n": 4, "lookback_weeks": 13, "rebalance_weeks": 4},
        "stop": args.stop, "slippage_bps": args.slippage_bps, "capital": args.capital,
        "min_order": args.min_order, "min_order_fraction": fraction,
        "share_price": args.share_price, "live_calendar_shift": live,
    }, "variants": {}, "calendars": {}}
    series: dict[tuple[str, int], pd.Series] = {}
    results = {}

    def run(name: str, shift: int):
        bars, at_decision, policy, costs, capital = variants[name]
        strategy = ShiftedMomentum(params, precomputed=indicators[bars], shift=shift)
        supervisor = RiskSupervisor(
            rules=(ShortSales(allowed=False), GrossExposureLimit(1.0), NetExposureLimit(-1.0, 1.0)),
            stop=ProtectiveStop(args.stop) if args.stop > 0 else None,
        )
        schedule = schedules[bars]
        result = run_once(markets[bars], strategy, schedule, RunId(f"ex-{shift}"), costs, policy,
                          supervisor=supervisor, capital=capital, fill_at_decision=at_decision)
        returns = periodic_returns(result)
        dates = periodic_dates(result)
        note = (f"{name} shift={shift} stop={args.stop} slip={args.slippage_bps} "
                f"commission={costs.describe()} capital={capital:g} "
                f"min_order={policy.min_order_value:g} "
                f"min_order_fraction={policy.min_order_fraction:.4f} "
                f"bars={'open-to-open' if bars == 'open' else 'weekly'} "
                f"fill={'decision' if at_decision else 'next-open'} "
                f"universe={universe.name}:{universe.fingerprint}")
        label = trial_label(markets[bars], schedule)
        with Study(STUDY, ledger) as study:
            if study.existing(strategy.version, label, note) is None:
                sd = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
                study.evaluate(strategy.version, label, {
                    "sharpe": float(returns.mean() / sd) if sd > 0 else 0.0,
                    "final_equity": result.final_equity, "periods": float(returns.size),
                }, returns=returns, note=note, dates=dates)
        series[(name, shift)] = pd.Series(returns, index=pd.to_datetime(list(dates)))
        return result

    for name in variants:
        results[name] = run(name, live)
    best_fill = max(("decide and fill at the close", "decide and fill at the open"),
                    key=lambda n: summarise(results[n].equity_curve(), 52.0).sharpe)
    combined = f"min order + {best_fill.removeprefix('decide and ')}"
    bars, at_decision, _, costs, capital = variants[best_fill]
    variants[combined] = (bars, at_decision, scaled, costs, capital)
    results[combined] = run(combined, live)
    on_every_calendar.append(combined)
    output["settings"]["combined"] = combined
    output["settings"]["best_fill"] = best_fill

    def against_baseline(name: str, shift: int) -> dict:
        both = pd.concat([series[(name, shift)], series[("baseline", shift)]], axis=1,
                         join="inner").dropna()
        difference = (both.iloc[:, 0] - both.iloc[:, 1]).to_numpy()
        return {"difference_per_year": float(difference.mean() * 52.0),
                "t": newey_west_t(difference), "weeks": int(difference.size)}

    for name, result in results.items():
        bars, _, _, _, capital = variants[name]
        row = describe(result, markets[bars], capital)
        row["vs_baseline"] = against_baseline(name, live)
        output["variants"][name] = row

    for name in on_every_calendar:
        rows = []
        for shift in range(params.rebalance_weeks):
            if ("baseline", shift) not in series:
                run("baseline", shift)
            result = results[name] if shift == live else run(name, shift)
            bars = variants[name][0]
            row = describe(result, markets[bars], variants[name][4])
            row["shift"] = shift
            row["live_calendar"] = shift == live
            row["vs_baseline"] = against_baseline(name, shift)
            rows.append(row)
        pooled = np.concatenate([
            (pd.concat([series[(name, s)], series[("baseline", s)]], axis=1, join="inner")
             .dropna().pipe(lambda f: f.iloc[:, 0] - f.iloc[:, 1]).to_numpy())
            for s in range(params.rebalance_weeks)
        ])
        output["calendars"][name] = {
            "rows": rows,
            "mean_cagr": float(np.mean([r["cagr"] for r in rows])),
            "mean_sharpe": float(np.mean([r["sharpe"] for r in rows])),
            "mean_max_drawdown": float(np.mean([r["max_drawdown"] for r in rows])),
            "worst_max_drawdown": float(np.min([r["max_drawdown"] for r in rows])),
            "mean_difference_per_year": float(np.mean(
                [r["vs_baseline"]["difference_per_year"] for r in rows])),
            "calendars_ahead": int(sum(r["vs_baseline"]["difference_per_year"] > 0 for r in rows)),
            "pooled_weeks": int(pooled.size),
        }

    output["fees_today"] = [
        fees_at_todays_size(results["baseline"], args.capital, price)
        for price in (args.share_price, 300.0)
    ]
    # The benchmark over the same window, bought once and held.
    whole = load_market(store, interval=BarInterval.WEEK, start=start)
    spy = InstrumentId("SPY")
    marks = [(m, whole.window.opens_at(m).get(spy)) for m in schedules["close"][1:]]
    marks = [(m, float(v)) for m, v in marks if v is not None]
    if len(marks) > 1:
        output["benchmark"] = {"symbol": "SPY", **{
            k: getattr(summarise(marks, 52.0), k)
            for k in ("cagr", "volatility", "sharpe", "max_drawdown")}}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=1, default=str))

    print(f"window {output['settings']['window']} · universe {output['settings']['universe']}"
          f" · live rotation calendar (shift {live})")
    print(f"{'variant':38s} {'CAGR':>7s} {'Sharpe':>7s} {'max DD':>8s} {'orders/y':>9s} "
          f"{'fees':>8s} {'vs base':>8s} {'t':>6s}")
    for name, row in output["variants"].items():
        print(f"{name:38s} {row['cagr']:7.1%} {row['sharpe']:7.2f} {row['max_drawdown']:8.1%} "
              f"{row['orders_per_year']:9.1f} {row['commission_paid']:8.0f} "
              f"{row['vs_baseline']['difference_per_year']:+8.2%} {row['vs_baseline']['t']:6.2f}")
    print("\nacross the four rotation calendars")
    for name, block in output["calendars"].items():
        print(f"{name:38s} CAGR {block['mean_cagr']:6.1%} Sharpe {block['mean_sharpe']:5.2f} "
              f"max DD {block['mean_max_drawdown']:6.1%} (worst {block['worst_max_drawdown']:6.1%}) "
              f"vs base {block['mean_difference_per_year']:+6.2%} "
              f"ahead on {block['calendars_ahead']}/4")
    print("baseline by calendar: " + ", ".join(
        f"{r['cagr']:.1%}{' (live)' if r['live_calendar'] else ''}"
        for r in output["calendars"]["baseline"]["rows"]))
    print(f"\nledger: study '{STUDY}' holds {ledger.count(STUDY)} trials · details in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
