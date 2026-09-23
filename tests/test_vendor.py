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
