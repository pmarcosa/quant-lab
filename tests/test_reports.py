"""Reports: the document is data, the page is rendered only from it."""

from __future__ import annotations

import math
import re
from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from contracts.errors import ContractViolation
from reports.document import (
    SCHEMA,
    Chart,
    Metric,
    Report,
    Series,
    Status,
    Table,
    cumulative,
    drawdown,
    rolling_mean,
)
from reports.html import fmt, render

X = ("2026-01-02", "2026-01-09", "2026-01-16", "2026-01-23")


def sample(**over) -> Report:
    base = dict(
        kind="backtest", title="Sample", generated_at="2026-09-19T10:00:00+00:00",
        status=Status("warning", "REDUCE-ONLY", ("drawdown beyond P80",)),
        metrics=(Metric("CAGR", 0.12, "pct"), Metric("Sharpe", float("nan"), "num")),
        charts=(Chart("eq", "Equity", X, (Series("Strategy", (100.0, 101.0, None, 99.0)),
                                           Series("SPY", (100.0, 100.5, 100.7, 101.0))),
                      format="money"),),
        tables=(Table("Years", ("year", "ret"), (("2026", 0.05),), ("text", "pct")),),
        notes=("a note",),
    )
    base.update(over)
    return Report(**base)


def test_the_document_round_trips_through_json(tmp_path):
    report = sample()
    path = report.save(tmp_path / "r.json")
    again = Report.load(path)
    assert again.to_dict() == report.to_dict()
    assert again.to_dict()["schema"] == SCHEMA


def test_nan_is_written_as_null_not_as_invalid_json(tmp_path):
    data = sample().to_dict()
    assert data["metrics"][1]["value"] is None
    (tmp_path / "r.json").write_text("x")
    sample().save(tmp_path / "r.json")
    assert "NaN" not in (tmp_path / "r.json").read_text()


def test_an_unknown_schema_is_refused():
    data = sample().to_dict()
    data["schema"] = "quant-lab.report/0"
    with pytest.raises(ContractViolation, match="schema"):
        Report.from_dict(data)


@pytest.mark.parametrize("bad", [
    dict(metrics=(Metric("x", 1.0, "percent"),)),
    dict(status=Status("orange", "x")),
    dict(charts=(Chart("a", "A", X, (Series("s", (1.0,)),)),)),
    dict(charts=(Chart("a", "A", X, tuple(Series(str(i), (1.0,) * 4) for i in range(4))),)),
    dict(charts=(Chart("a", "A", X, (Series("s", (1.0,) * 4),)),
                 Chart("a", "B", X, (Series("s", (1.0,) * 4),)))),
    dict(tables=(Table("t", ("a", "b"), (("only one",),)),)),
])
def test_a_document_the_renderer_would_guess_about_is_refused(bad):
    with pytest.raises(ContractViolation):
        sample(**bad).validate()


def test_the_page_is_self_contained():
    page = render(sample())
    assert page.startswith("<!doctype html>")
    assert "<svg" in page and "Table view" in page
    # No network: no external scripts, stylesheets, images or fonts.
    assert not re.search(r"""(src|href)\s*=\s*['"]?https?:""", page)
    assert "@import" not in page


def test_the_page_escapes_what_it_is_given():
    page = render(sample(title="<script>alert(1)</script>"))
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_a_gap_in_a_series_breaks_the_line_instead_of_joining_it():
    page = render(sample())
    chart = page[page.index("id='eq'"):]
    strategy_paths = re.findall(r"class='line s0' d='([^']+)'", chart)
    assert len(strategy_paths) == 2


def test_status_is_never_colour_alone():
    page = render(sample())
    assert "REDUCE-ONLY" in page and "lvl-warning" in page
    assert "drawdown beyond P80" in page


def test_two_series_get_a_legend_and_direct_labels():
    page = render(sample())
    assert "class='legend'" in page
    assert page.count("class='endlabel'") == 2


def test_an_empty_chart_says_so():
    empty = Chart("e", "Empty", (), (Series("s", ()),))
    page = render(sample(charts=(empty,)))
    assert "No data yet" in page


@pytest.mark.parametrize("value,format,text", [
    (0.1234, "pct", "12.3%"), (-0.05, "pct_signed", "-5.0%"), (0.05, "pct_signed", "+5.0%"),
    (1234567.8, "money", "1,234,568"), (12.345, "bps", "12.3 bp"), (None, "pct", "—"),
    (float("nan"), "num", "—"), ("x", "text", "x"), (3.14159, "num", "3.14"),
])
def test_numbers_read_the_way_a_person_expects(value, format, text):
    assert fmt(value, format) == text


def test_series_helpers():
    assert drawdown((100.0, 120.0, 90.0, None, 130.0)) == (0.0, 0.0, -0.25, None, 0.0)
    assert cumulative((0.1, -0.5)) == pytest.approx((1.0, 1.1, 0.55))
    assert rolling_mean((1.0, 2.0, 3.0), 2) == (None, 1.5, 2.5)


# -- runtime adapters, over the fake gateway ------------------------------------------

pytest.importorskip("ib_async")


def test_a_backtest_becomes_a_report(tmp_path):
    from runtime.monitor import build_baseline  # noqa: F401  (import check)
    from runtime.reporting import backtest_report
    from tests.test_monitor_wiring import make_session

    session, *_ = make_session(tmp_path)
    from contracts.identifiers import RunId
    from engine.decide import SizingPolicy
    from execution.simulated import CostModel
    from runtime.research import run_once

    market = session.market()
    result = run_once(market, session.strategy(), list(market.schedule), RunId("bt-test"),
                      CostModel(), SizingPolicy())
    now = datetime(2026, 9, 19, tzinfo=timezone.utc)
    report = backtest_report(result, market, "Test backtest", {"top_n": 2}, now, benchmark="AAA")
    data = report.to_dict()
    assert data["kind"] == "backtest"
    assert {c["id"] for c in data["charts"]} == {"equity", "drawdown"}
    assert len(data["charts"][0]["x"]) == len(result.steps)
    page = render(report)
    assert "Calendar-year returns" in page


def test_the_live_monitor_becomes_a_dashboard(tmp_path):
    from runtime.monitor import baseline_path, build_baseline, run_monitor
    from runtime.reporting import live_report
    from tests.test_monitor_wiring import live_weeks, make_session

    session, gateway, clock, frames = make_session(tmp_path)
    build_baseline(session, clock.now).save(baseline_path(session))
    live_weeks(session, gateway, clock, frames, tmp_path / "store", 4)
    monitor = run_monitor(session, benchmark="AAA")
    report = live_report(monitor, session.status(), asdict(session.config.monitoring),
                         benchmark="AAA")
    data = report.to_dict()
    assert data["status"]["label"].startswith("NORMAL")
    ids = {c["id"] for c in data["charts"]}
    assert {"live-equity", "live-return", "live-drawdown"} <= ids
    positions = next(t for t in data["tables"] if t["title"].startswith("Positions"))
    assert positions["rows"], "held positions are listed"
    assert all(row[4] is not None for row in positions["rows"]), "each has a stop"
    labels = {m["label"] for m in data["metrics"]}
    assert {"Live drawdown", "Break probability", "Stop coverage"} <= labels
    assert not any(isinstance(v, float) and math.isnan(v)
                   for m in data["metrics"] for v in [m["value"]])
    render(report)
