# quant-lab

A framework for designing, testing, running and maintaining trading strategies.
The strategies are the variable part; the rigour around them is not.

The organising idea: **a backtest and a live run must be the same code reading the
same data through the same interface, differing only in what "now" is.** Almost
every way a backtest lies is a way that sentence stops being true — a lookahead,
a universe assembled with hindsight, a live path that quietly takes a shortcut the
backtest never exercised. So the framework makes the seam between them a single
object (a `Filtration`) instead of a convention people remember to follow.

Status: **phases 0–5 complete** — contracts and data, the engine, the
falsification funnel and research ledger, a risk layer that may only reduce
exposure, and live trading through IBKR with monitoring and reports. Execution
is proposal-plus-typed-approval: the system computes orders, and only a person
typing the proposal's code sends them.

**To use it, read [docs/MANUAL.md](docs/MANUAL.md).** Everything runs through
one command, `ql` (`pip install -e ".[dev]"`, then `ql --help`).

## What is guaranteed today

Run it yourself:

```bash
pip install -e ".[dev]"
python scripts/ingest_ibkr_cache.py       # builds var/store from the committed CSVs
python scripts/demo_causality.py          # asserts the guarantees on that real data
python scripts/reconcile_conventions.py   # prices the execution conventions
python scripts/run_funnel.py              # five gates, on the real data
python scripts/compare_stops.py           # what a protective stop actually buys
python scripts/backtest_momentum.py --freq weekly --rebalance-weeks 4 --start 2009-02-24
pytest -q                                 # 583 tests
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

**[docs/MANUAL.md](docs/MANUAL.md)** is the operating manual: every command, how
to expand the universe, and an honest list of what is not built.

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
| fill at the decision bar's close | 27.5% | 1.23 | −20.0% |
| fill at the next bar's open | 27.1% | 1.13 | −26.2% |

Half a point of annual return, and six points of drawdown that were never there.
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
risk/        Limits, the protective stop, reduce-only.
validation/  Metrics, CPCV, DSR/PBO, the funnel, the ledger, live monitoring statistics.
reports/     Report documents (versioned JSON) and offline HTML. Never computes.
execution/   Broker adapters behind one port: simulated, and IBKR (ib_async).
runtime/     The composition root: the `ql` CLI, the live session and journal,
             monitoring, data refresh. Imports everything; imported by nothing.
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

## The funnel

Five gates, fixed in the design before any of them ran, which is the only honest
time to fix a threshold. `scripts/run_funnel.py` runs all of them on the real
data and writes every trial to the research ledger first.

| gate | verdict | detail |
|---|---|---|
| 1. monkey test | pass | 100% of 60 random controls, p=0.000 |
| 2. CPCV purged + embargoed | pass | 0% negative of 252 folds |
| 3. walk-forward efficiency | pass | WFE 0.95 over 14 windows |
| 4a. deflated Sharpe | pass | DSR 0.999 at N_eff 47.2 |
| 4b. probability of overfitting | pass | PBO 1.2% over 252 partitions |
| 5. jitter, noise, slippage | pass | 100% kept half the baseline |

Two caveats belong next to that PASS rather than in a footnote. The universe was
assembled in 2026 and holds no delisted names, so gate 1 neutralises selection
bias *within* the universe and not the universe's own construction. And the forty
configurations tried before the ledger existed have no recoverable return series,
so they are counted at face value rather than de-correlated.

### The ledger

The DSR needs to know how many strategies were tried before the winner was
chosen, and that number cannot be reconstructed afterwards: nobody remembers the
configurations that looked bad and were dropped, and those are exactly the ones
that make the survivor look good. So `validation/ledger.py` has no `skip`, no
`delete` and no filter. The only way to evaluate a strategy is through
`Study.evaluate`, and the only way to avoid recording a trial is not to run it.

Hyperparameter optimisation comes after the ledger, never before: optimising
without a trial count leaves the DSR uncomputable for that line of research,
permanently.

## Risk

`risk/` may **only reduce exposure**, and the contract enforces it on signed
positions, so it holds for longs and shorts alike: for every instrument, the
position risk approves must lie **between zero and the position the strategy
proposed**, inclusive. Risk may shrink an entry, stop a reversal at flat, or add
an order that closes a position. It may never enlarge an entry, and never shrink
an exit — watering an exit down leaves exposure on, which is an increase wearing
a limit's clothes. `RiskReview.__post_init__` refuses every violation, so a new
rule cannot break the invariant by being written carelessly.

The rules: gross exposure (longs plus shorts) and net exposure (longs minus
shorts) limits, which credit what an order closes before counting what it opens
and fail closed when a new name has no price; short sales, off unless a strategy
allows them and, live, cut to what the broker says can be borrowed; and
reduce-only, which lets positions shrink toward zero on either side and nothing
else.

The protective stop rests **at the broker**, not in the process: if the machine
running this is off, the positions are still protected. A long gets a sell stop
below its entry, a short a buy stop above it. It is a fixed 12% — the
same distance for a calm instrument and a volatile one, because volatility enters
through position *size* — and it is anchored to the **rotation price, never the
average cost**. Measured on the live book, a stop set from AAPL's average cost
sat 47% below the market and protected nothing, while one from DIS's sat above
the market and would have sold immediately. Average cost is information about the
holder's past, not the instrument's future.

A position a stop closes **stays closed** until the next rotation. And a resting
stop is cancelled before a rotation's own order on the same side reaches the
market, because two orders closing the same shares push the position through
zero.

### What the stop is worth

`scripts/compare_stops.py`, 2009–2026:

| stop | CAGR | Sharpe | max drawdown | stops fired |
|---|---|---|---|---|
| none | 25.1% | 1.10 | −25.2% | 0 |
| fixed 8% | 20.7% | 1.03 | −26.3% | 185 |
| fixed 10% | 22.3% | 1.05 | −22.6% | 130 |
| fixed 12% | 23.4% | 1.08 | −24.0% | 90 |
| fixed 15% | 21.9% | 1.01 | −24.6% | 63 |
| fixed 20% | 24.0% | 1.09 | −23.4% | 19 |

On this engine the stop does not pay for itself: every distance costs return, and
the deepest drawdown improves by at most 2.5 points — with one distance (8%)
making it *worse*. The ordering across distances is not monotonic, which is what
noise looks like. The default is 12% because that is what the live system runs;
whether to keep it is a judgement about sleeping at night, not a number this
table settles.

## Live trading and monitoring

`runtime/live.py` runs a **sleeve**: a fixed amount of capital inside an IBKR
account, whose positions are replayed from an append-only, integrity-checked
journal (`runtime/journal.py`). Each week, `ql live propose` computes orders
with the same engine the backtest uses. `ql live approve` sends them as
market-on-open orders, the fill the backtest assumes, only after the person
types the proposal's code. Before sending, it checks that the sleeve has not
changed since the proposal was made.

After every sync, the sleeve is reconciled against the account, on signed
positions. If the broker holds less than the sleeve believes, or on the other
side, the system halts.

**Several strategies.** Each deployed strategy has an id, its own config
(`configs/strategies/<id>.yaml`), its own journal and state folder, and its own
IBKR account. The id is written on every order it sends (IBKR's Order Ref is
`ql-<id>.<hash>`), so a strategy recognises its own fills and stops, and a
statement says which strategy sent what. Two strategies in one account are
refused by configuration: the broker would net their orders, and splitting fills
between virtual sub-accounts is machinery deliberately not built.

**Any bar size, either side.** The deployed strategy is weekly and long-only;
the system is neither. The bar size comes from the strategy and drives the data
refresh, the order type (opening auction for daily and weekly bars, day orders
intraday), how long a proposal stays valid, and the data-age limits. Monitoring
settings are calendar durations converted into bars, because a market's memory
is fixed in calendar time. Shorts need `allow_short`, a margin account, borrow
availability and a passing what-if margin check before a proposal is made.

`validation/monitoring.py` judges live results against distributions built from
the backtest, never against a single number. It uses four checks:

- the live drawdown's percentile among stationary-bootstrap paths of the same
  length;
- online changepoint detection;
- a robust trend with a bootstrap interval;
- implementation shortfall against the decision price and the modelled cost.

The checks drive a degradation ladder: NORMAL → REDUCE_ONLY → HALTED.
Monitoring can impose a halt, but only a person lifts one. The thresholds and
their measured false-alarm rates are in the manual, section 10.

`reports/` renders a backtest report and a live dashboard as self-contained HTML
from a versioned JSON document.

## What is not here yet

- **Delisted instruments.** The universe carries survivorship bias until
  point-in-time data is bought.
- **Borrow costs in backtests.** Shorts are backtested without borrow fees,
  margin interest or recalls, so a short strategy's backtest is optimistic.
- **Intraday readiness.** No exchange calendar (holidays, half days) and only
  IBKR's short intraday history. The plumbing handles hourly and minute bars;
  validating a strategy on them needs data this project does not have.
- **Borrow fees live.** The TWS API rarely reports them; the fee limit applies
  only when it does.
- **Automatic execution.** Not planned: approval stays manual by design.
