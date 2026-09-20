# quant-lab — user manual

How to run the system, what each part does, and what is not built yet.

Written for the person who owns it, not for a contributor. Every command below
is meant to be typed.

---

## 1. Where things are

```
contracts/    Types and Protocols. The vocabulary everything else shares.
access/       Who may do what. One local owner today.
data/         Bitemporal store, point-in-time universe, filtration, ingest,
              vendor conversion.
strategies/   The strategies. Sees contracts only.
engine/       Accounting, the decision, the loop. One implementation.
risk/         Limits and the protective stop.
validation/   Performance metrics, CPCV, DSR/PBO, the funnel, the ledger.
reports/      EMPTY. See section 7.
execution/    Broker adapters. Simulated only. See section 6.
runtime/      Wiring. Imports everything; imported by nothing.
scripts/      Everything you actually run.
docs/         This file.
data/ibkr_cache/   Committed raw CSVs. The reproducible input.
var/          Derived and gitignored: the store, the research ledger.
```

**The answer to "where is X" for two specific things:**

| you are looking for | it is at | honest status |
|---|---|---|
| IBKR **data** | `data/vendor.py`, `scripts/fetch_ibkr.py`, `scripts/import_ibkr_json.py` | works |
| IBKR **trading** | nowhere | **not built** — section 6 |
| **performance measurement** | `validation/metrics.py` | works |
| **performance reports** | nowhere | **not built** — section 7 |

---

## 2. First run

```bash
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/VIsual\ Studio/quant-lab
pip install -e ".[dev]"
python scripts/ingest_ibkr_cache.py     # builds var/store from the CSVs, ~2s
pytest -q                               # 324 tests, ~2s
```

`var/` is derived. Delete it and rebuild whenever you want a clean slate; the
committed CSVs are the source of truth.

---

## 3. The commands

### Run a backtest

```bash
python scripts/backtest_momentum.py --freq weekly --rebalance-weeks 4 --start 2009-02-24
```

| option | default | what it does |
|---|---|---|
| `--freq` | `weekly` | `weekly` or `daily` bars |
| `--rebalance-weeks` | `1` | Weeks between rotations. The book is marked **every** week regardless — marking only on rotation weeks hides what happened between them. |
| `--top` | `4` | Positions held |
| `--lookback` | `13` | Ranking horizon in weeks |
| `--stop` | `0.12` | Protective stop distance. `0` disables it. |
| `--cost-bps` / `--slippage-bps` | `10` / `10` | Per side |
| `--cash-buffer` | `0.01` | Held back from sizing, for commission and the gap between decision mark and fill |
| `--min-trade` | `0.005` | No-trade band, as a fraction of equity |
| `--capital` | `100000` | Starting cash |
| `--start` | none | Earliest decision date, ISO |

### Check the causality guarantees

```bash
python scripts/demo_causality.py
```

Asserts on real data that the universe is a function of the decision time, that
a pinned view cannot see past it, and that publication lag is modelled. Silence
is success.

### Price the execution conventions

```bash
python scripts/reconcile_conventions.py
```

Runs both fill conventions through the same engine, so the difference is the
convention and nothing else.

### Run the falsification funnel

```bash
python scripts/run_funnel.py                 # 60 controls, several minutes
python scripts/run_funnel.py --controls 20   # faster
```

Five gates on the real data. **Every backtest it runs is written to the research
ledger first** (`var/research.jsonl`). It resumes: if you interrupt it, running
it again continues from what is already recorded rather than repeating work or
double-counting trials.

Exit code `0` means the funnel passed, `2` means it did not.

### Compare stop distances

```bash
python scripts/compare_stops.py
```

---

## 4. Expanding the universe

This is the most valuable thing you can do to the system, and the reason is in
the funnel's own caveat: the current universe was assembled in 2026 from names
that had already done well, so the monkey test neutralises selection bias
*within* the universe but not the universe's own construction. A wider,
mechanically defined universe is what separates "momentum works" from "I picked
the winners of 2026".

**There is no universe list to edit.** The universe is *derived* from whatever is
in `data/ibkr_cache/`, with each instrument's listing date taken from its own
first bar. Adding a file adds a universe member; there is nothing else to keep in
sync. That is deliberate — a hand-maintained list is how a backtest ends up
trading names that had not listed yet.

### Path A — you have IB Gateway or TWS running

```bash
pip install -e ".[ibkr]"
python scripts/fetch_ibkr.py --symbols NFLX,ORCL,ADBE,CRM,NOW --freq weekly
rm -rf var && python scripts/ingest_ibkr_cache.py
```

Needs the gateway logged in with API socket clients enabled. Port `4001` for IB
Gateway, `7496` for TWS (`--port`).

### Path B — ask Claude

Claude reaches IBKR through the connector, which does not need the gateway
running. Say which instruments you want; Claude fetches the payloads and runs:

```bash
python scripts/import_ibkr_json.py --from-dir var/incoming --freq weekly
```

Either path validates before writing: mismatched array lengths, non-positive
prices, bars whose high is below their low, and duplicate timestamps are all
refused rather than written. A bad file that writes cleanly becomes a committed
CSV, then a store, then a result — and by then nothing looks wrong.

**Verified end to end on 2026-09-20.** NFLX was fetched through the connector
(1,181 weekly bars, 2004–2026), imported, and the store rebuilt: the universe
went from 39 to 40 members and the backtest picked it up. It was then reverted,
because one instrument moved the 17-year result from 23.4% to 25.5% CAGR — which
is the point below.

### After expanding

1. **Rebuild:** `rm -rf var && python scripts/ingest_ibkr_cache.py`
2. **Re-run the funnel.** A different universe is a different experiment; the
   previous gate results do not carry over.
3. **Expect the numbers to move.** One instrument changed the 17-year CAGR by two
   points. Thirty-five will change it more.
4. **Treat the expansion as a trial.** Choosing a universe *after* seeing which
   universe backtests better is selection bias with extra steps. Decide the
   membership rule first — an index, a liquidity screen, a sector spread — write
   it down, then fetch whatever it selects, including the names you would rather
   not own.

### How far back the data goes

IBKR caps a request at about 1,000 bars. Weekly reaches roughly 22 years; daily
only about 4. That is why the weekly series is the one with 17 years of history
and the daily one starts in 2022.

### What it still will not fix

The cache holds no **delisted** names, because it is assembled from instruments
that exist today. Adding more live instruments widens the universe without
removing survivorship bias. Fixing that is a data-vendor purchase, not a code
change, and the funnel's caveat stands until it is made.

---

## 5. Reading a result

```
years                17.52
CAGR                23.4%
volatility          21.5%
Sharpe                1.08     <- mean excess / std, annualised. The conventional one.
  geometric           1.09     <- (CAGR - rf) / vol. The previous system used this.
Sortino               1.12
max drawdown       -24.0%
final equity     3,970,080
```

The two Sharpe figures are both computable from the same curve and are not the
same number. The previous system reported the geometric one, which runs higher.
Quote the first when comparing against anything published.

`stops hit` counts positions a protective stop closed. `unfilled` counts orders
that expired because their instrument did not print at the execution bar.

---

## 6. IBKR trading: what is missing

There is **no IBKR execution adapter**. `execution/` contains one file,
`simulated.py`. `pyproject.toml` declares an optional `ibkr` extra, and until
section 4 nothing imported it.

What exists is the shape of the hole: `contracts/execution.py` defines
`ExecutionPort` — `capabilities`, `constraints`, `submit`, `poll`, `cancel`,
`positions` — and `SimulatedBroker` implements it in full, including the
inconvenient parts (orders are accepted and filled later, never both at once;
idempotency on the client order id; resting stops). An IBKR adapter is a new
implementation of that port. It is not a change to the engine, the risk layer or
any strategy, and the architecture test enforces that.

This is phase 5 work, and it stays **proposal-plus-manual-approval**:
`engine.run.propose` takes no broker and a test asserts the signature. Wiring
execution to it would be a design change, not a configuration one.

---

## 7. Performance reports: what is missing

`reports/` is empty. What exists is `validation/metrics.py`, which computes the
numbers; there is nothing that renders them.

The design calls for the generator to emit a **versioned JSON data document** with
rendering as a separate layer, showing the funnel state, DSR and PBO with N
visible, the trial count and the discrepancy metrics — not just an equity curve.
The separation matters: a report that computes its own numbers is a second
implementation that can disagree with the first.

For now, the scripts print to the terminal and `var/research.jsonl` holds every
trial as one JSON object per line, which is directly queryable:

```bash
python - <<'PY'
import json
rows = [json.loads(l) for l in open("var/research.jsonl")]
for r in sorted(rows, key=lambda r: -r["metrics"].get("sharpe", 0))[:5]:
    print(f'{r["metrics"]["sharpe"]:+.3f}  {r["note"]}')
PY
```

---

## 8. Things that will bite

**`rm -rf var` deletes the research ledger.** It lives at `var/research.jsonl`
and `var/` is gitignored. Copy it aside before rebuilding the store, or the trial
count behind your DSR is gone — and unlike the store, it cannot be rebuilt.
A safer habit:

```bash
cp var/research.jsonl /tmp/ && rm -rf var && python scripts/ingest_ibkr_cache.py
mkdir -p var && cp /tmp/research.jsonl var/
```

**The funnel takes minutes, not seconds.** Seventy-five backtests. It resumes, so
interrupting it is safe.

**Changing a parameter changes the strategy's identity.** `StrategyVersion` folds
a hash of the parameters into the version, so a 13-week and a 26-week lookback
are two strategies in the ledger. That is intentional: evidence earned by one
does not transfer to the other.

**The system does not send orders**, and no flag makes it.

---

## 9. Making a change safely

```bash
pytest -q && ruff check .
```

Both run in CI along with the ingest, the causality demo, the convention
reconciliation and a reduced funnel. If you add a package, add it to `ALLOWED` in
`tests/test_architecture.py` deliberately — a package that is not declared fails
a test, which is the point.

The house rule, applied throughout: **a guard that has never failed and a guard
that cannot fail look identical from the outside.** If you add a check, add a
test that makes it fail. Three real defects were found that way, including two
where my own measurement was wrong in a way that looked like a finding.
