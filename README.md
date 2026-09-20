# quant-lab

A framework for designing, testing, running and maintaining trading strategies.
The strategies are the variable part; the rigour around them is not.

The organising idea: **a backtest and a live run must be the same code reading the
same data through the same interface, differing only in what "now" is.** Almost
every way a backtest lies is a way that sentence stops being true — a lookahead,
a universe assembled with hindsight, a live path that quietly takes a shortcut the
backtest never exercised. So the framework makes the seam between them a single
object (a `Filtration`) instead of a convention people remember to follow.

Status: **phases 0 and 1 complete** — the contracts, the data layer, and the
architectural guard rails. There is no engine, no strategy and no broker yet, by
design. Those arrive on top of foundations that are already enforced.

## What is guaranteed today

Run it yourself:

```bash
pip install -e ".[dev]"
python scripts/ingest_ibkr_cache.py     # builds var/store from the committed CSVs
python scripts/demo_causality.py        # asserts the guarantees on that real data
pytest -q                               # 116 tests
```

`demo_causality.py` proves three things against 17 years of real IBKR bars:

1. **The universe is a function of the decision time.** A run standing at
   2011 sees 28 instruments; at 2021 it sees 35; today, 39. META, PLTR and CRCL
   are in the file and absent from the 2011 view because they had not listed.
2. **A pinned view cannot see past its decision time.** Ask for 100,000 AAPL
   closes at 2011-01-03 and you get 198, the last one dated 2010-12-27. There is
   no date-range argument to pass wrongly: the decision time is set when the view
   is constructed and is the only knob.
3. **Publication lag is separate from event time.** A bar that closed at 20:00
   is invisible at 20:07 and visible at 20:16. Knowing a price *later* than it
   happened is modelled, not assumed away.

Caveat, stated because it matters more than the guarantees: the universe is
derived from a cache assembled in 2026, so it contains no delisted names. It
removes *timing* bias, not *selection* bias. A real survivorship-free universe is
a data-vendor problem, not a code problem, and the docstring says so where
someone will actually read it.

## Layout

```
contracts/   Types and Protocols. Depends on nothing.
access/      Who may do what (a null-object local owner, for now).
data/        Bitemporal store, point-in-time universe, filtration, ingest.
strategies/  Strategies. Sees contracts only — never the engine, never a peer.
engine/      Backtest and live, one implementation.
risk/        Limits and kill switches.
validation/  The falsification funnel and the research ledger.
reports/     Presentation. Reads results, never computes them.
execution/   Broker adapters behind one port.
runtime/     The composition root. Imports everything; imported by nothing.
```

The dependency rules are in `tests/test_architecture.py:ALLOWED`, and they are
executable. A module that imports outside its allowance fails a test with the
file name and the offending package. This exists because the previous system
fused a momentum strategy into the same file as an HMM allocator, so importing
the strategy pulled in hmmlearn, scipy and sklearn — 738 modules. That is not a
speed problem, it is the layers announcing they can no longer be replaced
independently.

Two properties are enforced the same way:

- **No module-level mutable state.** A module-level dict is shared by every
  caller in the process, which breaks the first time two portfolios, two
  strategies or two runs exist at once. Capitals are not an exemption: a frozen
  non-empty lookup table nothing writes to is fine, `REGISTRY = {}` is not.
- **Nothing imports `runtime/`.** Wiring flows one way.

The guard rails are themselves tested. `test_the_import_rule_catches_a_crossed_layer`
and friends feed known violations to the same functions the real tests use and
require them to be caught — because a rule that passes on a clean codebase and a
rule that cannot fail look identical from the outside. The mutable-state rule
was found to have exactly that hole (`REGISTRY = {}` sailed through on the
strength of being uppercase) and was closed.

## Data

`data/ibkr_cache/` holds the raw CSVs and is committed: it is the reproducible
input. `var/store/` is derived, bitemporal and gitignored — rebuild it in about a
second with the ingest script.

The store is append-only. There is no `update` and no `delete`; a correction is a
new row with a later `available_time`, and `revise()` is deliberately identical
to `append()`. A split or dividend adjustment rewrites price history backwards,
which is the one restatement that actually happens to IBKR data, and the only
honest way to backtest across one is to be able to ask what the series looked
like *before* it was rewritten.

## What is not here yet

Phase 2 is the engine — one code path, driven by a `Filtration`, that runs both
a backtest and a live proposal. Phase 3 is the falsification funnel and the
research ledger; hyperparameter optimisation comes after the ledger exists, not
before, because optimisation without a count of trials is how a system discovers
something that was never there. Phases 4 and 5 add risk machinery and paper
trading.

Execution stays proposal-plus-manual-approval throughout. The system does not
send orders.
