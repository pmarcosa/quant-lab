# quant-lab — Code Review and Study Guide

A review of the whole codebase (~13k lines of source, ~12k of tests), the changes
it led to, and the ideas behind both. It is written to be studied, not skimmed:
each finding is a lesson with a name, a mental model, and real code from this
repository.

**How to use it.** Read Part 1 for the big picture and Part 2 for the mental
models. Part 3 is the line-level feedback; read one finding at a time, with
`git diff` open beside it. Part 4 is what I recommend next but did not change.

**What was verified.** Every change was checked the same way:

| check | before | after |
|---|---|---|
| `ruff check .` | clean | clean |
| `pytest` | 679 passed | **689 passed** (10 new regression tests) |
| real 17-year backtest (`ql backtest`) | CAGR 23.8%, Sharpe 1.05, equity 9,010,971 | **byte-for-byte identical output** |
| backtest wall time (best of 4) | 2.58 s | **1.47 s** (1.75× faster) |
| `state/research.jsonl` (your trial ledger) | — | untouched (checksum verified) |

Each bug fix comes with a test that **fails on the old code and passes on the
new**. That was confirmed by temporarily reverting the fix (`git stash`), not
assumed.

---

## Part 1 — Code analysis

### The verdict in one paragraph

This is an unusually well-designed codebase. Its architecture enforces its own
rules with executable tests, its types refuse invalid states at construction,
its accounting is an event log replayed by pure functions, and its docstrings
explain *why* rather than *what*. The bugs I found are not careless ones. They
sit in three places that are hard for any team: **fast paths that must equal slow
paths**, **floating-point boundaries**, and **identifiers whose meaning drifted
as the system grew**. Two of them could cost real money or real research
validity, and both are now fixed and pinned by tests.

### Strengths — and the names of the techniques, so you can reuse them

**1. Layered architecture with executable dependency rules (Ports and Adapters).**
`contracts/` depends on nothing; `strategies/` sees only `contracts/`; `runtime/`
is the *composition root*, the one place where concrete things are wired
together. Most projects state such rules in a README. This one tests them
(`tests/test_architecture.py:ALLOWED`), and it also tests the tests: it feeds a
known violation to the checker and requires it to be caught. That second step is
rare and important. *A rule that passes on clean code and a rule that cannot fail
look identical from the outside.*

**2. "Make illegal states unrepresentable."** `Filtration` has no method that
accepts a date, so a strategy *cannot* ask for the future. It doesn't need
to remember not to. That is much stronger than a convention. The same idea runs
through the frozen dataclasses with `__post_init__` validation (`TargetIntent`,
`Fill`, `Membership`): once a value exists, it is valid, so nothing downstream
re-checks it. This pattern is often summarised as **"parse, don't validate"**:
convert untrusted input into a type that *proves* its validity, once, at the
boundary.

**3. A functional core with an imperative shell.** `Book.apply(fill)` returns a
*new* `Book` and changes nothing. Pure functions over immutable data make
accounting replayable (`replay(opening, fills)`), testable without mocks, and
safe to share. The I/O (broker, journal, files) is pushed to the edges in
`runtime/`.

**4. Event sourcing.** The live journal, the research ledger and the bitemporal
store are all **append-only**. State is *derived* by replaying events, never
edited. This gives you an audit trail for free and makes "what did we believe
on 2021-03-01?" an answerable question.

**5. Deterministic idempotency keys.** `client_order_id` hashes the decision, so
a retry after a network timeout produces the *same* id and the broker refuses the
duplicate. This is the standard cure for "the order was sent twice".

**6. Docstrings that carry the reasoning.** Almost every non-obvious line says
which failure it prevents, often with the measured number that motivated it
("a stop set from AAPL's average cost sat 47% below the market"). This is the
most valuable kind of comment: it survives refactors because it explains intent.

### Weaknesses — the themes behind every finding

| theme | where it showed up |
|---|---|
| **A fast path that must equal a slow path, written twice** | `ReplayFiltrations` vs `StoreFiltration` duplicated five methods, and the bug lived in the one part they did not share |
| **An identity that stopped meaning what it used to** | a stop's id named the *decision*, but stops are cancelled and re-placed for the same decision |
| **Exact comparisons at floating-point boundaries** | `total(hi) < 1.0` never became false; `int(0.3 / 0.1) == 2`; `x <= 0` lets NaN through |
| **Loops whose cost grows with history, not with what is live** | the simulator scanned every order ever sent; slices copied whole histories |
| **A magic string standing for a concept** | `"-stop"` in 13 places across 4 modules |

Keep these five themes in mind. They will help you review *other* code too.

---

## Part 2 — Mental models to develop

These are the habits of thought behind the findings. Each one is framed as a
question you can ask of any code you read.

### Model 1 — "Is this a cache of the *answer*, or of the *inputs*?"

A cache is safe when it stores something that is still correct for every
question it will be asked. `ReplayFiltrations` cached the **answer** to "what is
the latest version of each bar, as of the horizon?" and then reused it to answer
"what was the latest version *as of 2018*?" Those are different questions. When
a cache must serve many points of view, it has to store the **inputs** (every
version) and compute the answer per question.

> **Ask:** if I change the parameter this cache was built with, is the stored
> value still the right answer?

### Model 2 — "An idempotency key names an *intention*. Can the intention be re-issued?"

Idempotency says the same key means the same thing, so a repeat is ignored. That
holds only if *the same thing can never be wanted twice*. A rotation order is
wanted once. A protective stop can be wanted, cancelled, and then wanted again
for the same decision. The second wanting is a new intention, so it needs a new
key. The standard fix is a **generation counter**, derived deterministically
(here, from the journal) so retries still collide and genuine re-issues don't.

> **Ask:** what is the full lifecycle of the thing this key names? Can it go
> from "done" back to "wanted"?

### Model 3 — "NaN fails every comparison"

In IEEE-754 floating point, `nan < 0`, `nan > 0` and `nan == nan` are all
`False`. So a guard written as `if x <= 0: raise` *silently admits NaN*, because
the condition is false. A guard written as `if not x > 0: raise` rejects it. Even
better is to make the intent explicit with `math.isfinite(x) and x > 0`, which
rejects infinity too. This codebase already used the safe form in one place
(`Charge`: `if not self.amount >= 0`) and the unsafe form in others. Consistency
is the lesson: **pick the safe idiom and use it everywhere.**

```python
>>> nan = float("nan")
>>> nan <= 0          # the guard `if price <= 0: raise` does not fire
False
>>> not nan > 0       # the guard `if not price > 0: raise` does
True
```

### Model 4 — "Floats are not reals near a boundary"

Floating point is exact for small integers and approximate for almost
everything else. Problems appear exactly at the boundaries your code cares
about:

```python
>>> 0.3 / 0.1
2.9999999999999996      # int() of this is 2, not 3
>>> sum([0.1] * 10)     # Python 3.10/3.11
0.9999999999999999      # never >= 1.0
```

Note that the second one is *version dependent*: Python 3.12+ uses compensated
summation in `sum()`, so the same code passes on 3.12 and hangs on 3.10. Your
CI runs both, and your laptop runs 3.14. **Boundary bugs can hide behind the
interpreter you happen to test on.**

> **Ask:** is there an `==`, `<` or `int()` applied to a computed float, *at
> exactly the value the logic depends on*? If so, add a tolerance or reason in
> integers.

### Model 5 — "What does each iteration cost, and does it grow?"

Find the loop inside the loop. A backtest is a loop over bars. Anything inside
it that touches *all history* (all orders ever submitted, the full price
series) makes the total cost quadratic, *O(n²)*. That is invisible at 900 weekly
bars and painful at 100,000 minute bars. The cure is almost always to keep an
index of what is **live** (the open orders) or to use the **sort order you
already have** (binary search with `searchsorted` instead of a boolean mask).

> **Ask:** if the run were 100× longer, which line would dominate?

### Model 6 — "Prove the refactor, don't argue it"

The author already had this instinct. `test_the_cache_cannot_widen_what_is_visible`
pins the fast path to the slow one. I extended it in two ways you should adopt:

- **Regression tests that fail first.** Write the test, run it on the *old* code,
  and watch it fail. A test you never saw fail might not test anything.
- **Differential testing.** When rewriting a function for speed, run old and new
  side by side on randomised inputs (including NaN and edge values) and require
  identical output. I did this for `changed_rows` (40 random cases) and for the
  whole backtest (byte-identical report).

---

## Part 3 — Specific feedback

Findings are ordered by severity. Each one gives **what**, **why it matters**,
**the fix**, and **the lesson**.

### Critical

#### C1. The cached backtest path lost history after any dividend refresh
`data/filtration.py`, `data/bitemporal.py` · test:
`test_a_revision_after_the_decision_does_not_hide_the_original`

**What.** `ReplayFiltrations` loaded each instrument once with
`store.as_of(instrument, horizon)`, which keeps only the **latest revision** of
each bar. It then sliced that per decision time with
`available_time <= decision_time`.

`ql data refresh` records a dividend or split by **re-appending every stored bar**
with `available_time = now` (by design: the store is bitemporal). After that,
the latest revision of every 2015 bar is dated 2026. So a 2015 decision sliced
by `available_time <= 2015` sees **nothing**, even though the original 2015 bars
are right there in the file.

**Reproduced on a synthetic store:**

```
uncached (StoreFiltration)   bars visible at 2018-03-05: 166
cached   (ReplayFiltrations) bars visible at 2018-03-05: 0     <- before the fix
```

**Why it matters.** `load_market` (the path behind `ql backtest`, the funnel and
the monitoring baselines) uses the cached path. After the first automated
`ql live cycle` that refreshes across a dividend, every later backtest would have
**silently dropped every dividend payer** from its history: different results, no
error. Your current store has no revisions yet (I checked all 79 files), so
**your existing results are unaffected.**

**The fix.** Cache the inputs, not the answer (Model 1). The store gained
`knowable()`, which returns *every* version, and one shared definition of
"latest":

```python
def latest_revisions(versions: pd.DataFrame) -> pd.DataFrame:
    if versions.empty or versions.index.is_unique:
        return versions
    return versions[~versions.index.duplicated(keep="last")]
```

`as_of()` is now `latest_revisions(knowable(...))`, and the replay cache stores
`knowable(horizon)` and applies `latest_revisions` **per decision**.

**A second, subtler bug fixed on the way.** The old code chose the latest
version with `groupby("event_time").last()`. In pandas, `.last()` returns the
**last non-null value per column**, not the last row. A revision with a blank
field would have been stitched together with a field from the version it
replaced. `duplicated(keep="last")` keeps whole rows. The sort is also now
`kind="stable"`, so two rows with identical timestamps resolve deterministically
(file order) rather than by an unstable quicksort.

**Lesson.** Know your library's semantics precisely. `groupby().last()`,
`groupby().first()` and `.max()` all skip NaN by default. When you mean "the
last row", say so.

#### C2. A re-placed protective stop was silently not placed
`runtime/live.py`, `contracts/execution.py` · test:
`test_a_replaced_stop_is_really_placed[tighter|off_then_on]`

**What.** A stop's id was `client_order_id(run, portfolio, instrument,
decision_time, side, quantity) + "-stop"`. The stop *price* is not part of it.
When `place_stops` must move a stop, for example because you change
`stop_distance` from 12% to 10% (the README calls this "a judgement about
sleeping at night"), it cancels the old stop and submits a new one **with the
same id**. The IBKR adapter is correctly idempotent: it finds the existing trade
with that `orderRef`, now *cancelled*, and returns it instead of placing an
order.

**Reproduced against the project's own fake gateway:**

```
after rotation, working stops      : [('AAA', 214.11), ('BBB', 143.88)]
place_stops() reported placed=2 cancelled=2
after re-placement, working stops  : []                     <- before the fix
after re-placement, working stops  : [('AAA', 218.97), ('BBB', 147.15)]   <- after
```

The journal recorded two `STOP_PLACED` events, the command reported success, and
**both positions were unprotected**. Disabling stops and re-enabling them has the
same effect. The next `reconcile()` would warn "no protective stop resting at the
broker". That is good defence in depth, but only a warning, and only later.

**The fix.** A generation counter (Model 2), in the contracts layer so the
backtest and live paths share it:

```python
def stop_order_id(run, portfolio, instrument, decision_time, side, quantity,
                  placement: int = 0) -> str:
    extra = (f"placement {placement}",) if placement else ()
    return (order_prefix(portfolio)
            + _decision_hash(run, portfolio, instrument, decision_time, side, quantity, *extra)
            + STOP_SUFFIX)
```

`place_stops` counts earlier `STOP_PLACED` events for the same instrument and
rotation in the journal. That keeps the id **deterministic** (a retry after a
crash computes the same number) while a genuine re-placement gets a fresh one.
It also skips any id it has just cancelled, in case a stop was placed but never
journaled. **Placement 0 produces exactly the old id**, so stops already resting
at your broker and every backtest's order ids are unchanged.

**Lesson.** When a system grows a new lifecycle (cancel, then replace),
re-examine every identifier that assumed the old one. And treat a broker's reply
as data to check, not as a formality (see suggestion S2).

#### C3. `capped_proportional` could loop forever
`strategies/sizing.py` · test: `test_a_cap_of_exactly_equal_weight_terminates[10|49]`

**What.** The scale search brackets the solution with:

```python
while total(hi) < 1.0:   # old
    hi *= 2.0
```

`MomentumParams` explicitly allows `weight_cap_mult = 1.0`, which caps every
weight at exactly `1/n`. Then `total(hi)` is a sum of `n` copies of `1/n`, which
in floating point can be `0.9999999999999999` (Model 4). It is never `>= 1.0`,
so `hi` doubles to infinity and the loop never ends. On Python 3.10, which your
CI runs, this happens at `top_n = 10`. On every version it happens at 49.

**Confirmed:** the old code was killed by a 5-second alarm inside that exact
loop.

**The fix.** Bracket with the same tolerance the bisection already uses, and
bound every loop so a numerical surprise raises an error instead of hanging:

```python
for _ in range(_MAX_DOUBLINGS):
    if total(hi) >= 1.0 - tol:
        break
    hi *= 2.0
else:  # for...else: runs only if the loop never hit `break`
    raise ContractViolation(unbracketable)
```

**Lesson.** Every `while` loop whose exit depends on a float needs (a) a
tolerance and (b) an iteration bound. Note the `for … else` construct: Python's
`else` on a loop runs when the loop finishes **without** `break`. That is
exactly "we gave up", so it is the right place for the error.

### Robustness

#### R1. NaN and infinity passed `Fill` and `OrderIntent` validation
`contracts/execution.py` · test: `test_a_fill_or_an_order_without_a_finite_number_is_refused`

`if self.price <= 0: raise` admits NaN (Model 3). A NaN fill price would make
`cash` NaN, then equity NaN, then every metric NaN, far from the cause. The
fix is a small helper used consistently:

```python
def _positive(value: float) -> bool:
    """Finite and above zero. False for NaN and infinity, which ``<= 0`` misses."""
    return math.isfinite(value) and value > 0
```

#### R2. Lot rounding could drop a whole lot
`contracts/execution.py` · test: `test_an_exact_multiple_of_a_fractional_lot_is_not_rounded_down`

`int(abs(quantity) / lot_step)` truncates `0.3 / 0.1 = 2.9999999999999996` to
`2`. With US stocks (`lot_step = 1.0`) this never triggers, but it would for
fractional shares, FX or crypto. The fix adds a tolerance far below any real lot
(`1e-9`), so it rescues only exact multiples, and rounds the result so
`0.30000000000000004` does not leak out.

#### R3. `_usable(price)` accepted infinity and rejected NumPy integers
`engine/decide.py`

`isinstance(price, (int, float)) and price > 0 and price == price`: the
`price == price` trick catches NaN but not `inf`, and `np.int64` is not a
subclass of `int`. Now: `isinstance(price, numbers.Real) and math.isfinite(price)
and price > 0`. NumPy registers its scalar types with the `numbers` abstract base
classes, so `numbers.Real` is the idiomatic "any real number" check.

#### R4. `Trial` promised "finite" metrics but only rejected NaN
`validation/ledger.py`

The error message said *finite*; the check was `value != value`. Infinity would
be written to the ledger as the non-standard JSON token `Infinity`. Now uses
`math.isfinite`. **Lesson:** when a message and a check disagree, one of them is
lying.

#### R5. The Sortino ratio used a non-standard downside deviation
`validation/metrics.py` · test: `test_sortino_averages_the_shortfall_over_every_period`

The code averaged squared losses over **losing periods only**. The standard
definition (Sortino and Price, 1994; also `empyrical`/`pyfolio`) averages over
**all** periods, counting a gain as zero shortfall:

```python
shortfall = np.minimum(excess, 0.0)
downside_deviation = float(np.sqrt(np.mean(shortfall**2)))
```

The old version overstated the deviation by about `sqrt(N / losers)`, so it
**understated Sortino by about 1.4×** for a strategy that loses half its weeks.

> ⚠️ **This changes a reported number.** Sortino appears in HTML reports and two
> scripts; it feeds no gate in the funnel. Reports made before this change are
> not comparable with reports made after it. This is the only change in the
> review that alters any output. Your module's own principle ("the definition
> used is stated… the alternative is computed alongside") would suggest
> reporting both for a while if you compare against older reports.

### Performance

**Result: the 17-year weekly backtest runs 1.75× faster (2.58 s → 1.47 s) with
identical output.** More importantly, several fixes remove *quadratic* behaviour,
so the gain grows with run length (daily, intraday).

#### P1. The simulator rescanned every order ever submitted, every bar
`execution/simulated.py`

`advance()` looped over `self._orders` (all history) and skipped the finished
ones. A 1,100-bar run accumulates thousands of orders, so each bar did work
proportional to the whole past: *O(bars × orders)*. Now a `_working` dict holds
only open orders. Python dicts preserve insertion order, so fills still come out
in submission order, as the docstring promises.

```python
for oid, intent in list(self._working.items()):   # snapshot: we delete while iterating
    ...
    del self._working[oid]                         # on fill, expiry or cancel
```

Note the `list(...)` snapshot: deleting from a dict while iterating over it
raises `RuntimeError`.

#### P2. Binary search instead of boolean masks on sorted time series
`strategies/momentum.py`, `data/filtration.py`

`frame.loc[frame.index <= t]` compares *every* row and **copies** the matching
prefix, and the caller only needed its length and last row. On a sorted index,
the same rows are found by binary search and taken as a slice:

```python
index = frame.index
if index.is_monotonic_increasing:          # cached by pandas after the first call
    visible = frame.iloc[: index.searchsorted(t, side="right")]
else:
    visible = frame.loc[index <= t]        # still correct for unsorted data
```

The replay filtration applies the same idea to *both* time conditions: when
`available_time` also rises monotonically (any store that was never revised),
both conditions select a prefix, and the intersection of two prefixes is the
shorter one. This was the single biggest cost left in the profile (about 7,600
calls per backtest).

**Lesson.** Sorted data is an asset. `searchsorted` is *O(log n)* and
`iloc[:k]` avoids a copy. Keep the safe fallback for the unsorted case.

#### P3. The universe was recomputed for every instrument
`data/filtration.py`

`_eligible()` called `universe.members_at(t)`, which filters and sorts all
memberships, once **per instrument**. `universe()` calls it for every member, so
it was quadratic per decision. Membership at a fixed instant cannot change, so
it is now computed once per view and stored in a `frozenset` for *O(1)* lookup.

#### P4. Compute once, use many times
`engine/run.py`

`execution_at(execution_time)` was called up to four times per bar (fills,
financing, stop anchors, valuation), each call building a new dict from a
DataFrame row. It is now fetched once into `fill_prices`. Similarly
`marks_at(opening.as_of)` had been called *inside* a dict comprehension, once
per held position.

#### P5. Vectorised `changed_rows` (no more `iterrows`)
`runtime/refresh.py`

After a dividend, every stored bar is compared with the fresh download. The old
code did it row by row with `DataFrame.iterrows()`, which builds a `Series` per
row and is among the slowest things you can do in pandas. The new version aligns
the two frames and compares whole arrays:

```python
events = pd.DatetimeIndex(rows["event_time"])
is_new = ~events.isin(known.index)
current = rows[columns].to_numpy(dtype=float)
previous = known[columns].reindex(events).to_numpy(dtype=float)   # NaN for new rows
same = np.isclose(current, previous, rtol=1e-9, atol=1e-9).all(axis=1)
```

I verified it with a differential test against the old implementation (Model 6):
40 random cases with NaNs, new rows, revised rows, and changes inside the
tolerance gave identical output.

#### P6. Smaller ones
- `split_count` built every combination to count them; `math.comb(n, k)` gives
  the count directly.
- `LeverageSchedule.base_level` copied the whole return history (`list(...)`)
  to keep its last 104 values; it now slices first.

### Maintainability and style

#### M1. One name for one concept: `STOP_SUFFIX` and `is_stop_order()`
The string `"-stop"` appeared 13 times across `engine/`, `risk/` and `runtime/`.
Its siblings `ORDER_PREFIX` and `ORDER_ID_SEPARATOR` were already named
constants in `contracts/execution.py`, which is where it now lives too. A
predicate `is_stop_order(oid)` reads as intent; `oid.endswith("-stop")` reads as
mechanism. If the suffix ever changes, it changes in one place.

#### M2. The Template Method pattern for the two filtrations
`StoreFiltration` and `_SlicedFiltration` duplicated `history`, `frame`,
`is_available`, `universe` and `metadata` line for line. Bug C1 lived in the one
part that *wasn't* shared. Now a base class `_PinnedFiltration` owns everything,
and subclasses override a single hook:

```python
class _PinnedFiltration(Filtration):
    def _load(self, instrument) -> pd.DataFrame:      # the only thing that varies
        raise NotImplementedError
    def history(self, instrument, field, count): ...   # written once

class StoreFiltration(_PinnedFiltration):
    def _load(self, instrument):
        return self._store.as_of(instrument, self._decision_time)

class _SlicedFiltration(_PinnedFiltration):
    def _load(self, instrument):
        return self._source.visible(instrument, self._decision_time)
```

**Lesson.** When two implementations *must* agree, make it structurally hard
for them to disagree: share everything except the one varying step. The file
stayed about the same length (255 → 260 lines) while losing the duplication and
gaining a feature: the revision-aware cache, plus its fast path.

#### M3. `nonlocal` instead of a one-element list
In `GrossExposureLimit` and `NetExposureLimit`, the closures used
`room = [value]` and `room[0] -= …` to update an outer variable. That is a
Python 2 workaround. Python 3 has `nonlocal room`, which says exactly what is
meant.

#### M4. Small items
- **Type hints that match reality.** `place_stops` annotated a 5-tuple as a
  4-tuple, `RiskSupervisor.rules` was a bare `tuple` (now `tuple[RiskRule, ...]`),
  and `BarInterval.duration` and `base_level` lacked types. A type checker would
  catch all of these (suggestion S4).
- **Explicit `encoding="utf-8"`** on `read_text`/`write_text`. The default is the
  platform's locale encoding, which is not UTF-8 on Windows.
- **Imports at module top** (`itertools` in `overfitting.py`, `StrategyId` in
  `ledger.py`), unless deferring one is deliberate (to break a cycle or delay a
  heavy import), in which case say so in a comment.
- **Named constants for shared thresholds.** `0.05` appeared twice in
  `engine/run.py` and had to agree; it is now `MIN_GROSS_FOR_BASE_RETURN`.
  `PRICE_COLUMNS` was defined in two modules; `refresh.py` now imports it.

---

## Part 4 — Improvement suggestions (not applied)

Ordered by priority. I did not apply these, because each needs a design
decision that belongs to you, or a change to a contract that other code relies
on.

**S1. Decide how delisted names are priced, before adding point-in-time data.**
*(High; latent today.)* `MarketWindow.from_store` loads prices only for
`universe.survivors_only()`, but the filtration's universe includes names during
their membership window, delisted or not. Today no instrument has a delisting
date, so nothing is affected. When you buy survivorship-free data (first item in
the README's "not here yet"), a selected delisted name would have no mark and the
run would fail. The docstring of `survivors_only` even says it is "useful only
to quantify the bias avoided". Decide the delisting convention first: the last
traded price, a cash-out at a recovery value, or the vendor's delisting return.

**S2. Verify the broker's reply in `place_stops`.** Even with C2 fixed, a stop
can come back `REJECTED` (a bad price, permissions). `place_stops` records
whatever status comes back and counts it as placed. Treating a terminal status
as a finding, or a halt, would turn "the journal says protected" into "the
broker confirmed protected". It is the same defence-in-depth principle you
already apply in `RiskReview`.

**S3. Read each CSV once per `load_market`.** The horizon calculation, the replay
filtration and `MarketWindow` each read every instrument's file. A small
per-load cache of `knowable(horizon)` frames would cut the I/O to a third.

**S4. Add a static type checker to CI** (mypy or pyright, starting with
`--strict` on `contracts/`). The codebase is already well annotated, so the cost
is low, and it would catch the M4 class of mismatches automatically. Several
parameters are still untyped (`broker`, `constraints_for` in `RiskSupervision`).

**S5. Property-based tests for the numeric kernels.** Libraries like
Hypothesis generate hundreds of edge-case inputs per test. A property such as
"for any feasible scores and bounds, `capped_proportional` returns weights
within bounds that sum to 1" would have found C3 automatically. Good candidates:
`capped_proportional`, `round_quantity`, `split_legs`, `Book.apply` (replay
equals sequential apply), and `_changes` in the risk rules.

**S6. Run the test suite locally on Python 3.10.** C3 depended on the
interpreter version, and you develop on 3.14 while CI tests 3.10. A tool like
`uv` or `nox` makes running both a single command.

**S7. Let the risk rules respect lot sizes.** `_trim_to_budget` floors to whole
shares (`math.floor`) without knowing the instrument's `lot_step`. That is correct
for US stocks and wrong for anything traded in lots of 100 or in fractions.
Passing `constraints_for` to the rules would fix it.

**S8. `LeverageState` copies its whole history every bar.** `run_backtest` builds
`tuple(equity_history)` per step, and the policy rescans it for the running peak:
*O(n²)* over a run. That's fine at weekly scale, but it will show at intraday.
Carrying the running peak in the state (or a read-only view) would fix it. It's
a contract change, so it deserves a deliberate decision.

**S9. PBO silently drops the tail.** `probability_of_backtest_overfitting` uses
`periods // blocks` rows per block and discards the remainder, while
`purged_splits` uses `np.array_split`, which keeps every row. Use one convention,
or at least document the dropped rows.

**S10. Consider more ruff rule sets.** `PERF` (performance anti-patterns), `PD`
(pandas-vet) and `NPY` (NumPy) catch several of the patterns fixed here. Enable
them one at a time and read each finding before accepting it.

**S11. The journal is re-read on every query.** `place_stops` alone reads the
whole file several times (to rebuild the book, find the last rotation, the
anchors, the fills and the placements). At weekly scale that is the right trade-off: always truthful,
crash-safe, and multi-process-safe. If you go intraday, an in-process cache
invalidated by file size and modification time would keep those properties.

---

## Appendix — files changed

| file | change |
|---|---|
| `contracts/execution.py` | `stop_order_id`, `STOP_SUFFIX`, `is_stop_order`, `_decision_hash`; NaN-safe `Fill`/`OrderIntent`; lot rounding tolerance |
| `contracts/temporal.py` | return type on `BarInterval.duration` |
| `data/bitemporal.py` | `knowable()`, `latest_revisions()`; `as_of` rebuilt on them; stable sort |
| `data/filtration.py` | `_PinnedFiltration` base class; replay cache holds every version; binary-search fast path; cached membership |
| `data/universe.py` | explicit encoding |
| `engine/decide.py` | `_usable` uses `numbers.Real` and `math.isfinite` |
| `engine/run.py` | execution prices fetched once per bar; `MIN_GROSS_FOR_BASE_RETURN`; `is_stop_order` |
| `execution/simulated.py` | working-order index |
| `risk/leverage.py` | slice before copying; typed `base_level` |
| `risk/rules.py` | `stop_order_id`; `nonlocal`; typed `rules` |
| `runtime/live.py` | stop placement numbering; `is_stop_order`; annotation fix |
| `runtime/monitor.py` | `is_stop_order`; explicit encoding |
| `runtime/refresh.py` | vectorised `changed_rows`; shared `PRICE_COLUMNS` |
| `strategies/momentum.py` | binary-search slicing of precomputed indicators |
| `strategies/sizing.py` | bracketing with tolerance and iteration bounds |
| `validation/cpcv.py` | `math.comb` |
| `validation/ledger.py` | finite-metric check; top-level import |
| `validation/metrics.py` | standard Sortino downside deviation |
| `validation/overfitting.py` | top-level import |
| `tests/…` | 10 new regression tests in 5 files |

Nothing was committed. Review with `git diff`, and commit when you are satisfied.
