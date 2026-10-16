"""Converting a vendor payload into the cache format.

The validation tests matter more than the happy path. A malformed payload that
writes successfully becomes a committed CSV, then a bitemporal store, then a
backtest result — and by then nothing looks wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from contracts.errors import ContractViolation
from data.vendor import (
    CACHE_COLUMNS,
    bars_from_connector,
    cache_inventory,
    check_coverage,
    write_cache_csv,
)

REAL_CACHE = Path(__file__).resolve().parent.parent / "data" / "ibkr_cache"

#: The IBKR cache is not in the repository (its licence forbids redistribution);
#: these tests run where a user has fetched it.
needs_cache = pytest.mark.skipif(
    not (REAL_CACHE / "weekly").is_dir(),
    reason="needs the IBKR cache; fetch it with `ql data fetch` (manual, step 2.2)",
)


def payload(n=5, **overrides):
    weeks = pd.date_range("2026-01-05", periods=n, freq="W-MON", tz="UTC")
    base = {
        "time": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in weeks],
        "open": [100.0 + i for i in range(n)],
        "high": [101.0 + i for i in range(n)],
        "low": [99.0 + i for i in range(n)],
        "close": [100.5 + i for i in range(n)],
        "volume": [1_000_000 + i for i in range(n)],
    }
    base.update(overrides)
    return base


def test_a_payload_becomes_an_ohlcv_frame():
    bars = bars_from_connector(payload())
    assert list(bars.columns) == list(CACHE_COLUMNS)
    assert len(bars) == 5
    assert bars.index.is_monotonic_increasing
    assert bars.index.name == "timestamp"
    assert bars["close"].iloc[0] == 100.5


def test_bars_come_back_in_order_however_they_arrived():
    scrambled = payload()
    scrambled["time"] = list(reversed(scrambled["time"]))
    bars = bars_from_connector(scrambled)
    assert bars.index.is_monotonic_increasing


def test_arrays_of_different_lengths_are_refused():
    """The failure that produces a perfectly ordinary-looking wrong file."""
    broken = payload()
    broken["close"] = broken["close"][:-1]
    with pytest.raises(ContractViolation, match="disagree in length"):
        bars_from_connector(broken)


def test_a_missing_field_is_refused():
    broken = payload()
    del broken["low"]
    with pytest.raises(ContractViolation, match="missing field"):
        bars_from_connector(broken)


def test_an_empty_payload_is_refused():
    with pytest.raises(ContractViolation, match="no bars"):
        bars_from_connector(payload(0))


def test_a_non_positive_price_is_refused():
    broken = payload()
    broken["close"][2] = 0.0
    with pytest.raises(ContractViolation, match="close is not positive"):
        bars_from_connector(broken)


def test_a_null_price_is_refused():
    broken = payload()
    broken["open"][1] = None
    with pytest.raises(ContractViolation, match="open is not positive"):
        bars_from_connector(broken)


def test_bars_that_are_not_internally_consistent_are_refused():
    broken = payload()
    broken["high"][3] = 1.0  # below the low
    with pytest.raises(ContractViolation, match="high is below"):
        bars_from_connector(broken)


def test_two_bars_for_one_timestamp_are_refused():
    broken = payload()
    broken["time"][2] = broken["time"][1]
    with pytest.raises(ContractViolation, match="two bars"):
        bars_from_connector(broken)


# -- writing -----------------------------------------------------------------


def test_writing_produces_the_format_the_ingest_reads(tmp_path):
    target = write_cache_csv("NFLX", bars_from_connector(payload()), tmp_path, "weekly")
    assert target == tmp_path / "weekly" / "NFLX.csv"
    text = target.read_text().splitlines()
    assert text[0] == "timestamp,open,high,low,close,volume"
    assert text[1].startswith("2026-01-05,")
    assert "T00:00:00" not in text[1], "dates only, so the file stays diffable"


@needs_cache
def test_the_written_file_matches_the_shape_of_the_cache():
    """A new instrument has to look exactly like the ones already there."""
    existing = (REAL_CACHE / "weekly" / "AAPL.csv").read_text().splitlines()
    assert existing[0] == "timestamp,open,high,low,close,volume"


def test_a_lowercase_or_odd_symbol_is_refused(tmp_path):
    bars = bars_from_connector(payload())
    with pytest.raises(ContractViolation, match="uppercase"):
        write_cache_csv("nflx", bars, tmp_path)
    with pytest.raises(ContractViolation, match="uppercase"):
        write_cache_csv("BRK.B", bars, tmp_path)


def test_an_unknown_frequency_is_refused(tmp_path):
    with pytest.raises(ContractViolation, match="frequency"):
        write_cache_csv("NFLX", bars_from_connector(payload()), tmp_path, "yearly")


def test_a_written_file_round_trips_through_the_ingest(tmp_path):
    from data.ingest import load_price_csv

    write_cache_csv("NFLX", bars_from_connector(payload()), tmp_path, "weekly")
    reloaded = load_price_csv(tmp_path / "weekly" / "NFLX.csv")
    assert len(reloaded) == 5
    assert {"open", "high", "low", "close"} <= set(reloaded.columns)


# -- inventory ---------------------------------------------------------------


def test_the_inventory_reports_what_is_cached(tmp_path):
    write_cache_csv("AAA", bars_from_connector(payload(40)), tmp_path, "weekly")
    write_cache_csv("BBB", bars_from_connector(payload(4)), tmp_path, "weekly")
    inventory = cache_inventory(tmp_path, "weekly")
    assert set(inventory["symbol"]) == {"AAA", "BBB"}
    assert inventory.set_index("symbol").loc["AAA", "bars"] == 40


def test_thin_coverage_is_named_rather_than_failed(tmp_path):
    """A short series is normal for a recent listing and suspicious otherwise."""
    write_cache_csv("AAA", bars_from_connector(payload(40)), tmp_path, "weekly")
    write_cache_csv("BBB", bars_from_connector(payload(4)), tmp_path, "weekly")
    assert check_coverage(cache_inventory(tmp_path, "weekly"), minimum_bars=27) == ("BBB",)
    assert check_coverage(pd.DataFrame()) == ()


@needs_cache
def test_the_real_cache_has_no_thin_series_that_should_not_be_thin():
    inventory = cache_inventory(REAL_CACHE, "weekly")
    assert len(inventory) >= 39
    # SPCX listed in 2026; anything else with under 27 weeks is worth a look.
    thin = set(check_coverage(inventory, minimum_bars=27))
    assert thin <= {"SPCX"}, thin


# -- the import script -------------------------------------------------------


def test_the_import_script_validates_before_writing(tmp_path, monkeypatch, capsys):
    import scripts.import_ibkr_json as importer

    good = tmp_path / "good.json"
    good.write_text(json.dumps(payload()))
    bad = tmp_path / "bad.json"
    broken = payload()
    broken["close"] = broken["close"][:-1]
    bad.write_text(json.dumps(broken))

    monkeypatch.setattr(importer, "CACHE", tmp_path / "cache")
    code = importer.main([f"GOOD={good}", f"BAD={bad}", "--freq", "weekly"])
    assert code == 0
    assert (tmp_path / "cache" / "weekly" / "GOOD.csv").exists()
    assert not (tmp_path / "cache" / "weekly" / "BAD.csv").exists()
    assert "SKIPPED" in capsys.readouterr().err


def test_a_dry_run_writes_nothing(tmp_path, monkeypatch):
    import scripts.import_ibkr_json as importer

    source = tmp_path / "x.json"
    source.write_text(json.dumps(payload()))
    monkeypatch.setattr(importer, "CACHE", tmp_path / "cache")
    assert importer.main([f"XYZ={source}", "--dry-run"]) == 0
    assert not (tmp_path / "cache").exists()


def test_isolated_bad_prints_are_brought_back_to_the_body():
    """BSX's week of 2009-01-12 as IBKR lists it: a low of 0.01 under 7.65-7.87."""
    import pandas as pd

    from data.vendor import clip_bad_prints

    index = pd.to_datetime(["2009-01-05", "2009-01-12", "2009-01-19", "2008-10-06"])
    bars = pd.DataFrame({
        "open": [7.50, 7.65, 7.80, 22.50], "high": [7.90, 7.95, 9000.0, 24.40],
        "low": [7.40, 0.01, 7.70, 6.71], "close": [7.66, 7.87, 7.90, 9.68],
        "volume": [1, 1, 1, 1]}, index=index)
    fixed, changes = clip_bad_prints(bars)
    assert fixed.loc["2009-01-12", "low"] == 7.65
    assert fixed.loc["2009-01-19", "high"] == 7.90
    # A genuine crash week (Morgan Stanley, October 2008) is left as it traded.
    assert fixed.loc["2008-10-06", "low"] == 6.71
    assert len(changes) == 2
    assert fixed[["open", "close", "volume"]].equals(bars[["open", "close", "volume"]])


def test_dividends_paid_before_a_split_are_counted_in_todays_shares():
    """NVIDIA lists 0.04 a share in March 2024, three months before its 10-for-1."""
    from datetime import date

    from data.vendor import connector_dividends

    payload = {"corp_actions": [
        {"type": "CashDividends", "date": "20240305", "value": "0.04", "currency": "USD"},
        {"type": "CashDividends", "date": "20240611", "value": "0.01", "currency": "USD"},
        {"type": "CashDividends", "date": "20200601", "value": "0.16", "currency": "USD"},
        {"type": "Splits", "date": "20240610", "value": "10.0"},
        {"type": "StockDividends", "date": "20210720", "value": "4.0"},
        {"type": "Splits", "date": "20110509", "value": "0.1"},       # a reverse split
        {"type": "CashDividends", "date": "20110101", "value": "0.01", "currency": "USD"},
    ]}
    paid = {ex: amount for ex, amount, _ in connector_dividends(payload)}
    assert paid[date(2024, 6, 11)] == pytest.approx(0.01)     # after the split: as listed
    assert paid[date(2024, 3, 5)] == pytest.approx(0.004)
    assert paid[date(2020, 6, 1)] == pytest.approx(0.004)     # 0.16 / 4 / 10
    assert paid[date(2011, 1, 1)] == pytest.approx(0.01 / (0.1 * 4 * 10))
    assert list(paid) == sorted(paid)


def test_a_share_action_without_a_positive_ratio_is_refused():
    from data.vendor import connector_dividends

    with pytest.raises(ContractViolation):
        connector_dividends({"corp_actions": [{"type": "Splits", "date": "20240610",
                                               "value": "0"}]})


def test_a_spin_off_listed_as_a_split_does_not_rescale_the_dividends():
    """Honeywell 2026: 'Splits 0.5', yet the dividend went from 1.19 to 0.70."""
    from datetime import date

    from data.vendor import connector_dividends

    spin = {"corp_actions": [
        {"type": "CashDividends", "date": "20260515", "value": "1.19"},
        {"type": "Splits", "date": "20260629", "value": "0.5"},
        {"type": "CashDividends", "date": "20260814", "value": "0.7"},
    ]}
    paid = {ex: amount for ex, amount, _ in connector_dividends(spin)}
    assert paid[date(2026, 5, 15)] == pytest.approx(1.19)

    reverse = {"corp_actions": [                              # Duke Energy, 2012
        {"type": "CashDividends", "date": "20120511", "value": "0.255"},
        {"type": "Splits", "date": "20120703", "value": "0.3333"},
        {"type": "CashDividends", "date": "20120814", "value": "0.765"},
    ]}
    paid = {ex: amount for ex, amount, _ in connector_dividends(reverse)}
    assert paid[date(2012, 5, 11)] == pytest.approx(0.255 / 0.3333)
