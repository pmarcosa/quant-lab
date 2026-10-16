"""The sector-gated momentum strategy.

Its only rule of its own is the gate: a stock may enter only while its sector
is trending. Everything else must stay the weekly momentum strategy's, so the
first test pins the two together when every sector is open.
"""

from __future__ import annotations

import numpy as np
import pytest

from contracts.errors import ContractViolation
from contracts.identifiers import InstrumentId
from contracts.targets import Holdings
from strategies.momentum import MomentumParams, WeeklyMomentum
from strategies.sector_momentum import SectorMomentum, SectorMomentumParams
from tests.test_strategy_momentum import _gathering_pace, series

UP, UP2, FLAT, DOWN = (InstrumentId(s) for s in ("up", "up2", "flat", "down"))
TECH, ENERGY, STAPLES = (InstrumentId(s) for s in ("XTECH", "XENER", "XSTAP"))
SECTORS = (("up", "XTECH"), ("up2", "XENER"), ("flat", "XSTAP"), ("down", "XSTAP"))
WEEKS = 120


def _frames(**overrides):
    frames = {
        UP: series(_gathering_pace(200)),
        UP2: series(_gathering_pace(300)),
        FLAT: series(np.linspace(100, 220, WEEKS)),     # up, steady: fails the pace filter
        DOWN: series(np.linspace(220, 100, WEEKS)),     # breaks its trend
        TECH: series(_gathering_pace(150)),             # a sector in trend
        ENERGY: series(np.linspace(150, 90, WEEKS)),    # a sector falling
        STAPLES: series(_gathering_pace(120)),
    }
    frames.update({InstrumentId(k): v for k, v in overrides.items()})
    return frames


class _Bars:
    """A filtration over made-up stocks and sector funds, pinned at the last bar."""

    def __init__(self, frames):
        self.frames = frames

    @property
    def decision_time(self):
        return self.frames[UP].index[-1].to_pydatetime()

    def universe(self, min_bars=1):
        return tuple(i for i, f in self.frames.items() if len(f) >= min_bars)

    def history(self, instrument, field, count):
        frame = self.frames.get(instrument)
        return frame[field].tail(count) if frame is not None else series([])[field]


def _strategy(**params):
    settings = dict(rebalance_weeks=1, top_n=0, sectors=SECTORS)
    settings.update(params)
    return SectorMomentum(SectorMomentumParams(**settings))


def test_with_every_sector_open_it_is_the_weekly_momentum_strategy():
    frames = _frames(XENER=series(_gathering_pace(160)))     # energy in trend as well
    gated = _strategy().target(_Bars(frames), Holdings())
    stocks = {i: f for i, f in frames.items() if i in (UP, UP2, FLAT, DOWN)}
    plain = WeeklyMomentum(MomentumParams(rebalance_weeks=1, top_n=0)).target(
        _Bars(stocks), Holdings())
    assert gated.weights == pytest.approx(plain.weights)
    assert set(gated.weights) == {UP, UP2}


def test_a_stock_whose_sector_is_falling_does_not_enter():
    target = _strategy().target(_Bars(_frames()), Holdings())
    assert UP in target.weights and UP2 not in target.weights      # energy is falling
    assert target.weights[UP] == pytest.approx(1.0)


def test_sector_funds_are_read_and_never_traded():
    table = _strategy().rank(_Bars(_frames()), Holdings())
    assert not set(table["instrument"]) & {TECH, ENERGY, STAPLES}
    assert dict(zip(table["instrument"], table["sector_open"], strict=True))[UP2] is False
    target = _strategy().target(_Bars(_frames()), Holdings())
    assert not set(target.weights) & {TECH, ENERGY, STAPLES}


def test_a_held_stock_in_a_closed_sector_is_judged_by_its_own_exits():
    """The gate is for entries. A held name stays frozen until an exit sells it."""
    held = Holdings({UP2: 0.30})
    target = _strategy(top_n=1).target(_Bars(_frames()), held)
    assert target.weights[UP2] == pytest.approx(0.30)
    assert target.weights[UP] == pytest.approx(0.70)


def test_top_k_keeps_only_the_best_sectors():
    frames = _frames(XENER=series(_gathering_pace(400)))     # energy now the strongest
    sectors = _strategy(sector_trend=False, sector_top_k=1).sector_table(_Bars(frames))
    assert list(sectors.loc[sectors["open"], "sector"]) == [ENERGY]
    target = _strategy(sector_trend=False, sector_top_k=1).target(_Bars(frames), Holdings())
    assert set(target.weights) == {UP2}


def test_a_sector_with_no_history_yet_is_left_open():
    frames = _frames()
    del frames[ENERGY]                                      # the fund had not listed
    target = _strategy().target(_Bars(frames), Holdings())
    assert {UP, UP2} <= set(target.weights)


def test_impossible_gates_are_refused():
    with pytest.raises(ContractViolation):
        SectorMomentumParams(sectors=SECTORS, sector_trend=False, sector_top_k=0)
    with pytest.raises(ContractViolation):
        SectorMomentumParams(sectors=SECTORS, sector_top_k=-1)
    with pytest.raises(ContractViolation):
        SectorMomentumParams(sectors=(("up", "XTECH"), ("up", "XENER")))
    with pytest.raises(ContractViolation):
        SectorMomentumParams(sectors=(("up", "XTECH"), ("XTECH", "XENER")))


def test_the_map_is_part_of_the_version():
    one = _strategy().version
    other = _strategy(sectors=SECTORS[:3]).version
    assert one != other
    assert one != WeeklyMomentum(MomentumParams(rebalance_weeks=1, top_n=0)).version


def test_the_registry_reads_the_committed_map():
    from runtime.strategies import build_strategy, sector_map

    pairs = sector_map("momentum-wide")
    assert ("XOM", "XLE") in pairs and ("AAPL", "XLK") in pairs
    assert len(pairs) == len({stock for stock, _ in pairs}) == 150
    strategy = build_strategy("sector-momentum", {"sectors": "momentum-wide", "top_n": 5})
    assert strategy.params.sectors == pairs
    with pytest.raises(ContractViolation):
        build_strategy("sector-momentum", {"sectors": "no-such-map"})


def test_the_wide_universe_and_its_map_agree():
    from runtime.strategies import sector_map
    from runtime.wiring import universe_list

    stocks = set(universe_list("momentum-wide").symbols)
    gated = set(universe_list("momentum-wide-gated").symbols)
    pairs = dict(sector_map("momentum-wide"))
    assert set(pairs) == stocks
    assert gated == stocks | set(pairs.values())
    assert set(universe_list("momentum").symbols) <= stocks
