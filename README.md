# quant-lab

A framework for designing, testing, running and maintaining trading strategies.
The strategies are the variable part; the rigour around them is not.

The organising idea: **a backtest and a live run must be the same code reading the
same data through the same interface, differing only in what "now" is.** Almost
every way a backtest lies is a way that sentence stops being true — a lookahead,
a universe assembled with hindsight, a live path that quietly takes a shortcut the
backtest never exercised. So the framework makes the seam between them a single
object (a `Filtration`) instead of a convention people remember to follow.

Status: **phases 0, 1 and 2 complete** — the contracts and data layer, then the
engine, a broker adapter and the first strategy. Risk machinery and the
falsification funnel are next; execution stays proposal-only throughout.

## What is guaranteed today

Run it yourself:

```bash
pip install -e ".[dev]"
python scripts/ingest_ibkr_cache.py       # builds var/store from the committed CSVs
python scripts/demo_causality.py          # asserts the guarantees on that real data
python scripts/reconcile_conventions.py   # prices the execution conventions
python scripts/backtest_momentum.py --freq weekly --every 4 --start 2009-02-24
pytest -q                                 # 196 tests
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

## One decision path

A backtest and a live session are not two implementations kept in agreement. They
are two callers of `engine.decide.decide`, differing only in where the filtration,
the book and the marks come from. `engine.run.propose` — the whole live path — is
four lines, because they are the same four lines the backtest runs.
`test_a_live_proposal_matches_the_backtest_decision` runs a backtest to a bar,
makes a live proposal standing at that same bar from the book the backtest had
reached, and requires the orders to match down to the client order id. The id is
a hash of the decision, so equal ids mean equal decisions rather than similar
ones.

`propose` takes no broker and no execution prices. The system proposes and a
person approves; wiring execution to it would be a design change, not a
configuration one, and a test asserts the signature.

The division of labour is fixed and each part is enforced somewhere:

- a **strategy** says what *fraction* of the book it wants, and is never told the
  equity. It receives a filtration and its own current weights — dimensionless,
  so the shape of the book reaches it and the size does not.
- the **engine** turns fractions into whole tradable quantities, given equity,
  marks and the broker's lot and tick rules.
- the **broker** decides what the fill actually costs.

### What the engine refuses to do

Filling a decision at the price it was decided on is the most common way a
backtest reports money that was never available. `scripts/reconcile_conventions.py`
runs both conventions through this same engine over 17 years of real bars:

| convention | CAGR | Sharpe | max drawdown |
|---|---|---|---|
| fill at the decision bar's close | 27.0% | 1.23 | −21.1% |
| fill at the next bar's open | 20.2% | 0.88 | −49.3% |

Nearly seven points of annual return and more than half the reported drawdown.
The previous system used the first convention. The script asserts the direction
so it stays a regression rather than an anecdote, and runs in CI.

Two more refusals, both found while building phase 2:

- **Prices are not forward filled for execution.** These instruments do not share
  a calendar — IBKR stamps a weekly bar on the first trading day of the week, so
  1,547 distinct timestamps cover 1,437 weeks. The previous system reindexed onto
  SPY's calendar and forward filled, which means some backtested fills happened at
  a price never quoted that week. Here bars are matched by ISO week and nothing is
  filled: a week an instrument did not print is a week it cannot be traded, and
  an order with no price expires rather than transacting at a stale one.
- **Priced and tradable are separate.** A held position needs a mark every period
  to be *valued*; that is not permission to transact in it. Collapsing the two
  either stops the book being valued or lets the backtest trade through a halt.

### Accounting

Equity is `cash + sum(quantity * price)`. There is no other formula in the system,
and only a fill moves the book. This is aimed at a measured failure: the previous
system tracked weights, and when a stop fired it renormalised the survivors to sum
to one — which deleted the loss from the total and read as "everything else just
got bigger". It produced a 58% CAGR. An accounting error does not look like an
error, it looks like a brilliant strategy, which is why it survives review.
`test_a_stop_does_not_inflate_the_rest_of_the_book` is the regression.

## Layout

```
contracts/   Types and Protocols. Depends on nothing.
access/      Who may do what (a null-object local owner, for now).
data/        Bitemporal store, point-in-time universe, filtration, ingest.
strategies/  Strategies. Sees contracts only — never the engine, never a peer.
engine/      Accounting, the decision, the loop. One implementation.
risk/        Limits and kill switches. Empty: phase 4.
validation/  Performance measurement; the funnel and ledger come in phase 3.
reports/     Presentation. Reads results, never computes them. Empty.
execution/   Broker adapters behind one port. Simulated one today.
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

**Phase 3** is the falsification funnel and the research ledger. Hyperparameter
optimisation comes after the ledger exists, not before: optimisation without a
count of trials is how a system discovers something that was never there.

**Phase 4** is risk machinery — protective stops among it. Today's numbers carry
no stop, which is why the honest drawdown above is −49%. The exit *rules* (trend
break, RSI reversion, drawdown from the rolling high) belong to the strategy and
are implemented; a protective stop is a risk mechanism and belongs in `risk/`.

**Phase 5** is paper trading and autonomy.

Execution stays proposal-plus-manual-approval throughout. The system does not
send orders.
