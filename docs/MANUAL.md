# quant-lab — user manual

How to run every part of the system yourself, in the order you will need it:
set it up, look after the data, widen the universe, test a strategy, read the
reports, then trade it through IBKR, paper account first, and watch it live.

Each step says **what to type**, **what it does**, and **why it is done that
way**. The "why" matters most, because it tells you when a step can be skipped.
Usually it cannot.

Every command is typed in a terminal at the root of the repository:

```bash
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/VIsual\ Studio/quant-lab
```

---

## Contents

1. [The system in one page](#1-the-system-in-one-page)
2. [One-time setup](#2-one-time-setup)
3. [Price data](#3-price-data)
4. [Expanding the universe](#4-expanding-the-universe)
5. [Research: backtests, the funnel, the ledger](#5-research-backtests-the-funnel-the-ledger)
6. [Reading the reports](#6-reading-the-reports)
7. [Before trading: IB Gateway, the config, the baseline](#7-before-trading-ib-gateway-the-config-the-baseline)
8. [Opening the sleeve](#8-opening-the-sleeve)
9. [The weekly cycle](#9-the-weekly-cycle)
10. [Monitoring live performance](#10-monitoring-live-performance)
11. [Incidents: what to do when something is wrong](#11-incidents-what-to-do-when-something-is-wrong)
12. [From paper to live money](#12-from-paper-to-live-money)
13. [Files, state and backups](#13-files-state-and-backups)
14. [Things that will bite](#14-things-that-will-bite)
15. [Changing the code safely](#15-changing-the-code-safely)
16. [Command reference](#16-command-reference)

---

## 1. The system in one page

**What it trades.** The weekly momentum strategy ranks the universe by 13-week
momentum and holds the top 4, equally weighted. It rotates every 4 weeks. Each
position gets a protective stop 12% below its entry price. These are the
defaults in `configs/live.example.yaml`; your own settings live in
`configs/strategies/<id>.yaml` (section 7).

**One strategy, one account, one id.** Each deployed strategy has a short id
(for example `momentum`), its own config file, and its own IBKR account: a
linked account under the same login when you run more than one. The id is
written on every order the strategy sends. IBKR shows it in the *Order Ref*
column as `ql-momentum.3fa9…`, so you can tell in TWS or in a statement which
strategy sent an order, and the system recognises its own fills and stops by it.
Section 7.5.

**Not only weekly, not only long.** The momentum strategy is weekly and
long-only. The machinery is not: a strategy can decide on weekly, daily, hourly
or minute bars, and can hold short positions when its config allows it.
Section 7.6 says what changes.

**How it trades.** The system *proposes*, and you *approve* by typing a code.
That is the default, and nothing changes it but you, twice: the strategy's config
must allow automation (`automation.mode`), *and* you must arm it with a typed
phrase (`ql live auto arm`, step 9.9). Armed, `ql live cycle` sends what its
safety gates allow; a halt disarms it. Orders are market-on-open orders (MOO), so
they fill in Monday's opening auction. That is exactly how the backtest assumes they fill, so live
fills and simulated fills can be compared honestly.

**The sleeve.** The strategy manages a fixed amount of capital inside its IBKR
account, the `sleeve_capital`. It only knows the positions it opened or that you
handed to it, and it never trades anything else. By default the account is
*dedicated* to the strategy, so anything else found in it stops the system; with
`account_scope: shared`, other holdings are only reported (section 7.2).

**The journal.** Every proposal, approval, order, fill, stop, snapshot,
reconciliation, correction and state change is appended to one file:
`state/live/<id>/<mode>-journal.jsonl`. The sleeve's positions and cash are *replayed*
from that file every time they are needed. That makes the journal the source of
truth, and the one file you must never lose (section 13).

**The degradation ladder.** The system is always in one of three states:

| state | proposals | new exposure | who can put it there | who can lift it |
|---|---|---|---|---|
| `NORMAL` | yes | yes | — | — |
| `REDUCE_ONLY` | yes | **no**: only orders that shrink positions (selling a long, covering a short) and stops | monitoring, or you (`pause`) | monitoring lifts its own; you lift yours with `clear` |
| `HALTED` | only `--liquidate` | no | monitoring, a reconciliation mismatch, or you (`halt`) | **only you**, with `clear` and a written reason |

**Where things are:**

```
contracts/    Shared vocabulary: types, protocols, live states and journal events.
data/         Bitemporal store, point-in-time universe, ingest, IBKR file conversion.
strategies/   The strategies. They see contracts only, so they stay swappable.
engine/       Accounting and the decision loop. One implementation for backtest and live.
risk/         Limits, the protective stop, the reduce-only rule.
validation/   Metrics, CPCV, DSR/PBO, the funnel, the ledger, live monitoring statistics.
execution/    Broker adapters: simulated.py (backtests) and ibkr.py (IB Gateway / TWS).
reports/      Report documents (JSON) and their offline HTML pages.
runtime/      Wiring. cli.py is the `ql` command; live.py the live session;
              monitor.py the monitoring pass; journal.py the event journal;
              refresh.py the data refresh; reporting.py builds reports;
              strategies.py the registry of deployable strategies.
scripts/      The research scripts `ql` delegates to.
configs/      live.example.yaml (committed); strategies/<id>.yaml (yours, gitignored).
data/ibkr_cache/   Raw weekly and daily CSVs from IBKR: the input. Yours, never committed
                   (IBKR's licence forbids redistributing market data).
data/universe.txt  The symbols the momentum strategy trades: what to fetch.
var/store/    The bitemporal store, derived from the cache. Safe to delete and rebuild.
state/        Irreplaceable: journals, baselines, the research ledger, reports. Gitignored.
```

**The week at a glance** (details in section 9):

| when | command | what it does |
|---|---|---|
| Saturday | `ql data refresh` | adds the week that closed on Friday |
| Saturday | `ql live sync` | brings the journal up to date with the account |
| Saturday | `ql monitor run` | judges live results, may change the state, writes the dashboard |
| Saturday | `ql live propose` | computes this week's orders; sends nothing |
| Saturday to Monday before 09:28 New York | `ql live approve` | you type the code; MOO orders are sent |
| Monday after the open | `ql live sync` | records fills and places the new stops |
| any day | `ql live status` | state, equity, positions, stops |

---

## 2. One-time setup

### Step 2.1 — Install

```bash
pip install -e ".[dev]"
```

This installs the project and its dependencies (numpy, pandas, scipy, pyyaml,
`ib_async` for talking to IB Gateway, and `exchange_calendars`, the NYSE session
calendar that intraday strategies need). It also installs the **`ql`**
command. The `-e` (editable) flag means `ql` always runs the code in this folder,
so a `git pull` needs no reinstall.

If `ql` is not found afterwards, your Python's scripts folder is not on your
`PATH`. Use `python -m runtime.cli` instead: it is the same program, and every
`ql …` in this manual can be typed as `python -m runtime.cli …`.

### Step 2.2 — Get the price data, then build the store

The price data is not in the repository. It comes from IBKR, whose market-data
agreement forbids redistributing it, so each user fetches their own with their
own account. With IB Gateway running (step 7.1):

```bash
ql data fetch --symbols "$(paste -sd, data/universe.txt)" --freq weekly
ql data fetch --symbols "$(paste -sd, data/universe.txt)" --freq daily   # optional
```

This writes one CSV per symbol under `data/ibkr_cache/`, which git ignores,
with the prices as traded (split-adjusted), and one file of dividend factors
per symbol under `data/ibkr_cache/factors/` (section 3, "Splits and
dividends"). Without the gateway, import saved payloads instead (step 4.2b). Keep the folder
backed up with `state/` (section 13): it is the input every result was computed
from. Then:

```bash
ql data ingest --rebuild
```

This reads the CSVs in `data/ibkr_cache/`, multiplies each bar by its
dividend factor, and writes the bitemporal store in `var/store/`. It names
any symbol without factors (price-only). It is derived data: the CSVs are the input, and the store
is how the engine reads them without looking into the future. `--rebuild`
deletes `var/store` first, and nothing else.

### Step 2.3 — Run the tests

```bash
pytest -q && ruff check .
```

Both must pass (about 580 tests, about 25 seconds). The tests include the IBKR
adapter and the whole live cycle, run against a stand-in gateway: the weekly
long-only momentum strategy, and a daily long/short test strategy that exists
only to prove the machinery does not assume either. A red
test means the machine you are on differs from the one the code was verified on,
and nothing should be traded until it is green.

### Step 2.4 — Check the data

```bash
ql data status
```

It shows how many instruments are cached, how many bars the store holds, and
how old the latest complete bar is. It looks at the configured strategy's bar
size; `--interval daily` (or `weekly`, `hourly`, `minute`) chooses another. Use it whenever you are unsure what the
system is looking at.

---

## 3. Price data

### What the data is

For the momentum strategy, weekly bars, one per instrument per week. Each bar is
stamped at the moment it became knowable: **Friday 21:00 UTC**, after the US close. That time is a safe
bound all year round, because it is 16:00 or 17:00 in New York depending on
daylight saving. A backtest decision on Friday's close can therefore never use a
bar that was not yet complete.

Other bar sizes follow the same rule. A daily bar is stamped at 21:00 UTC on its
own day. An intraday bar is stamped at its *end*: IBKR labels hourly and minute
bars with their start time, and a bar stamped there would be visible to a
decision it had not finished forming for. Each bar size has its own store
(`bars_1week`, `bars_1day`, …) and its own cache folder
(`data/ibkr_cache/weekly`, `daily`, …).

### Splits and dividends

IBKR serves two price series. **TRADES** is adjusted for splits but not
dividends; it exists for every bar size and can be paged backwards. **ADJUSTED_LAST**
is adjusted for both, but only up to today and only for bars of a day or less.
Following the expert's rule (the primary copy is the series as traded; the
adjustment lives in a separate table and is applied when the data is read;
a pre-adjusted series is never the primary copy, because every dividend
rewrites all of its history):

| layer | holds | written by |
|---|---|---|
| `data/ibkr_cache/<frequency>/` | TRADES bars, as traded | `fetch`, `refresh`, `backfill` |
| `data/ibkr_cache/factors/` | one factor per session: ADJUSTED_LAST ÷ TRADES on its close, i.e. the product of (1 − dividend ÷ price) over every later ex-date; 1.0 on the day of the download | the same commands, from two daily requests |
| `var/store/` | TRADES × factor: the total-return series the engine reads | `ingest`, `refresh` |

Volume is not adjusted (splits are already in TRADES; a cash dividend does not
change the share count). A weekly bar is grouped from daily TRADES; in a week
with an ex-date, the sessions before it are put at the week-end scale, a ratio
that never changes once the week is over, and the store applies the factor of
the week's last session. Live stops need no conversion: they are anchored on
fills, or on the latest bar, whose factor is 1 right after a refresh.

Two checks join downloads made on different days:

- **A new dividend** since the last download multiplies every earlier factor by
  one constant. The stored factors are rescaled by it, and every stored bar is
  revised (the old values stay, as of before).
- **A split** since the last download shows as one constant ratio between the
  new and the cached closes. The symbol's whole history is downloaded again,
  never patched.

A ratio that is not one constant means the two downloads disagree about more
than a split or a dividend; the symbol is refused and its cache left as it was.

### The store

The store is **bitemporal**: every row carries both the week it describes and
the moment it was recorded. When IBKR restates history, for example after a
split, the new values are added as a revision and nothing is overwritten. A
backtest run "as of" an earlier date still sees what was known then.

### Step 3.1 — Refresh every week (needs IB Gateway, section 7)

```bash
ql data refresh
```

It works on the configured strategy's bar size (`--interval` chooses another).
For every instrument in the universe, it does four things:

1. Asks the gateway for recent TRADES bars: two years of daily bars grouped into
   weeks (IBKR cannot adjust bars longer than a day), one year of daily bars,
   ten days of hourly or two days of minute bars. Plus daily ADJUSTED_LAST over
   the same window, for the dividend factors. A symbol with no factors yet gets
   its whole history, once.
2. Drops the first, partial week of the window and the current bar while it is
   still incomplete.
3. Checks the new download against the cache: a new dividend rescales the
   factors, a split downloads the whole history again, anything else is refused
   (section 3, "Splits and dividends").
4. Updates the cache and the factors, and appends to the store only what is new
   or restated (TRADES × factor), stamped with the moment of the fetch. After a
   new dividend or a split, that is every bar.

The output lists new weeks, revisions and a note per instrument ("dividend
since the last refresh", "split … full history downloaded again"). An instrument
that failed is shown with its error; the others still update.

**Why two years and not one week.** The overlap with the cache is what the two
checks compare; a longer window also picks up a vendor's correction of recent
bars. The store keeps every version.

**Options:** `--symbols AAPL,MSFT` refreshes only those symbols. `--duration "5 Y"`
reaches further back. `-v` lists every instrument, including unchanged ones.

### Step 3.2 — Rebuild when in doubt

```bash
ql data ingest --rebuild
```

Rebuilding from the cache is always safe, and it is required after importing
new instruments (section 4). The difference from a refresh: a rebuild stamps
every bar as known at its own Friday close, with a single version per week. The
record of *when* a refresh actually saw a bar, and of restatements it picked
up, is lost. Research is unaffected: it assumes bars are known at the close
anyway. The live system keeps its own history in the journal.

---
## 4. Expanding the universe

Widening the universe is the most valuable research you can do, for the reason
the funnel prints in its own caveat. The current universe was assembled in 2026
from names that had already done well. The monkey test neutralises selection
bias *within* the universe, but not the bias in how the universe itself was
chosen. A wider universe, defined by a mechanical rule, is what separates
"momentum works" from "I picked the winners of 2026".

**There is no universe list to edit.** The universe is *derived* from whatever is
in `data/ibkr_cache/`. Each instrument's listing date is taken from its own
first bar. Adding a file adds a member, and there is nothing else to keep in
sync. This is deliberate: a hand-maintained list is how a backtest ends up
trading names that had not yet listed.

### Step 4.1 — Decide the rule before fetching anything

Write the membership rule down first. It might be an index's members, a
liquidity screen, or a sector spread. Then fetch *everything* the rule selects,
including the names you would rather not own. Choosing the universe *after*
seeing which one backtests better is selection bias with extra steps.

### Step 4.2a — Fetch through IB Gateway (if it is running, section 7)

```bash
ql data fetch --symbols NFLX,ORCL,ADBE,CRM,NOW --freq weekly
```

This asks the gateway for 22 years of daily TRADES and daily ADJUSTED_LAST for
each symbol (4 years with `--freq daily`), derives the dividend factors,
validates both, and writes the bars to `data/ibkr_cache/weekly/` (grouped into
weeks) and the factors to `data/ibkr_cache/factors/`. It replaces the symbol's
file. The host, port and client id come from the strategy's config (with several
configured, put `--strategy <name>` before `data`); `--port` overrides them.

**When a symbol comes back empty**, the output prints what IBKR said, with a
plain-language hint, and the gateway's data-farm status at connection:

| IBKR says | Meaning | What to do |
|---|---|---|
| no reply within Ns, or 366 | the request timed out and was cancelled | check the HMDS farm line; retry with `--timeout 300` or a shorter `--duration "10 Y"` |
| 162 … no market data permissions | the login has no data for it | paper accounts: turn on market-data sharing with the live account (Client Portal → Settings → Paper Trading Account); the live account needs a US stock subscription |
| 354 / 10168 | no subscription / no market data | as above |
| 321 … Multi day bar size not supported with adjusted last | an adjusted request for bars longer than a day | fixed in the code (weekly bars are grouped from daily); update the repo |
| 2105 / 2107 HMDS … broken / inactive | the historical data farm is down | wait and retry; `inactive` usually connects on the first request |

Options: `--duration` (IBKR syntax), `--timeout` seconds per request (default
120), `--attempts` per request (default 2).

### Step 4.2b — Or import saved payloads (no gateway needed)

Claude can fetch price history through the IBKR connector without the gateway.
Ask for the instruments, and the payloads are saved as JSON. Then import them:

```bash
ql data import --from-dir var/incoming --freq weekly
ql data import NFLX=nflx.json ORCL=orcl.json --freq weekly
```

The connector serves prices as traded but not the adjusted series, so imported
symbols have no dividend factors: `ingest` lists them as price-only until they
are fetched through the gateway.

Both paths validate before they write anything. Mismatched array lengths,
non-positive prices, bars whose high is below their low, and duplicate
timestamps are refused. A bad file that writes cleanly would become a cached
CSV, then a store, then a result, and by that point nothing would look wrong.

### Step 4.3 — Rebuild and check

```bash
ql data ingest --rebuild
ql data status -v
```

The verbose status lists every instrument with its number of bars and its date
range. Look for any file holding far less history than the others. That is the
most common failure, and it is silent.

### Step 4.4 — Treat the new universe as a new experiment

1. **Re-run the funnel** (section 5.3). A different universe is a different
   experiment, and earlier gate results do not carry over.
2. **Expect the numbers to move.** Adding one instrument (NFLX, in a test on
   2026-09-20) moved the 17-year CAGR by two points.
3. **If you are trading live, rebuild the monitoring baseline** (step 7.4). A
   strategy without a `strategy.universe` in its config trades whatever is in
   the store, so from the next proposal it trades the new universe. Monitoring
   must compare it against a backtest of the same universe. To keep a strategy
   on a fixed list whatever you fetch, give it a universe (step 4.5).
4. **Record the new universe.** Add the symbols to `data/universe.txt` and
   commit that list, so anyone with IBKR data can rebuild the same universe. The
   CSVs themselves stay out of git (step 2.2); back them up with `state/`.

### Step 4.5 — Name the universe a strategy chooses from

A universe is a file, `data/universes/<name>.txt`: one ticker per line, `#`
starts a comment. The repository ships `sector-etfs`, the eleven Select Sector
SPDRs. Use one:

```bash
ql backtest --universe sector-etfs
ql funnel --universe sector-etfs --top 3 --rebalance-weeks 2
```

or put it in a strategy's config, where it becomes part of the strategy:

```yaml
strategy:
  name: weekly-momentum
  params: {rebalance_weeks: 4, top_n: 4, lookback_weeks: 13}
  universe: sector-etfs        # a name or a path; omit for every instrument in the store
```

- Symbols on the list with no data are left out and named in the output; fetch
  them with `ql data fetch`. A symbol joins the point-in-time universe from its
  first bar (XLC from 2018).
- The ledger records the universe (name and a fingerprint of its symbols) on
  every trial.
- A live sleeve records the universe it was opened on. If the config later gives
  another one, or the file gains or loses a symbol, every `ql live` command
  refuses: another universe is another strategy, so open a new strategy id.
- Without a universe, a strategy chooses from every instrument in the store.
  That is the behaviour of configs written before universes existed.
- `universe` goes **under `strategy:`**, indented. A key the loader does not
  read, anywhere at the top level of a config, is refused with a hint rather
  than ignored. To check what is in force: `ql strategies` lists each
  strategy's universe, and `ql backtest` prints the universe it used and where
  it came from.

### What expanding will not fix

The cache holds no **delisted** names, because it is built from instruments that
exist today. More live instruments widen the universe but do not remove
survivorship bias. Fixing that means buying data from a vendor, not changing the
code. IBKR has no history for securities that no longer trade. Today's S&P 500
members over twenty years is the extreme case: the expert calls the effect on
momentum "catastrophic", because the backtest keeps "discovering" the names
that later became the largest. Until there is point-in-time membership data,
the options are the sector ETFs (the issuer rebalances them, so they carry no
survivorship bias), a 3-5 year window, or a haircut of 200-400 bp a year on the
result.

---

## 5. Research: backtests, the funnel, the ledger

### Step 5.1 — Run a backtest

```bash
ql backtest
ql backtest --report
ql backtest --top 5 --lookback 26 --stop 0 --start 2009-01-01 --report
```

This runs the strategy over the whole store with exactly the engine the live
system uses. It prints CAGR, volatility, Sharpe, maximum drawdown, final equity
and stops fired. With `--report` it also writes a report you can open in any
browser (section 6.1).

| option | default | what it does |
|---|---|---|
| `--top` | from config, else 4 | positions held |
| `--lookback` | from config, else 13 | momentum horizon, in weeks |
| `--rebalance-weeks` | from config, else 4 | weeks between rotations. The book is still marked every week. |
| `--stop` | from config, else 0.12 | protective stop distance; `0` disables it |
| `--cost-bps` / `--slippage-bps` | from config, else 10 / 10 | commission and slippage per side, in basis points |
| `--commission` | bps | `bps` charges `--cost-bps` of each order's value. `ibkr-tiered` and `ibkr-fixed` charge IBKR's published plan in dollars per order instead: per share, with the minimum, the 1% cap, and (tiered) exchange, clearing and regulatory fees |
| `--share-price` | 100 | with an IBKR plan: the typical share price per-share fees are charged at. The stored history is split-adjusted, so an old bar of a stock that later split shows many more shares than were traded; `0` charges on the stored quantity |
| `--cash-buffer` / `--no-trade-band` | from config, else 0.01 / 0.005 | the fraction of equity held back from sizing, and the size below which an adjustment is not sent |
| `--limit-band` | from config, else market orders | send limit orders this far through the decision price; an instrument that gaps past the band is not traded |
| `--stop-limit-offset` | from config, else a plain stop | make the stop a stop-limit this far under the stop. A stop-limit does not fill when the price gaps through its limit; a negative value forces a plain stop |
| `--min-order` | 0 | orders below this many dollars are not sent. Exits are exempt |
| `--min-order-fraction` | 0 | the same rule as a fraction of equity, for an account that stays its present size (1,000 dollars at 17,000 is 0.06) |
| `--fill` | next-open | `decision-close` fills rotation orders at the close they were decided on, as a closing-auction order would. It looks ahead by the last minutes before the auction |
| `--start` | the first week in the store | earliest week, as an ISO date |
| `--benchmark` | SPY | the comparison line in the report |
| `--universe` | from config, else the whole store | a universe name or file (step 4.5) |
| `--capital` | the config's `sleeve_capital`, else 100,000 | starting capital; see below |
| `--report` | off | writes `state/reports/<id>/backtest-<time>.html` and `.json` |

`--top`, `--lookback` and `--rebalance-weeks` are the momentum strategy's
parameters. For any other strategy the command refuses them and uses the
`strategy.params` in its config. With several strategies configured, say which:
`ql --strategy trend backtest`.

**Shorts in a backtest.** A strategy allowed to short (`risk.allow_short`) is
backtested with its gross and net exposure limits and with buy stops above its
shorts. The backtest charges **no borrow fees, no margin interest and no
recalls**, so its short side is optimistic; the output says so.

**Why the defaults come from the config.** A backtest you are going to act on
should be of the strategy you are actually going to trade.

**Every backtest is recorded.** Each run is written to the research ledger
(`state/research.jsonl`, study `manual`) before its result is shown.

**Re-running.** A trial's label holds its dates, a fingerprint of the prices
and a fingerprint of the code that produces the numbers (`engine/`, `risk/`,
`strategies/`, `data/`, `contracts/`, the simulated broker). So:

- the identical run again is recognised and recorded once;
- a run after a bug fix, a re-fetch or on another universe is a new trial. It
  should be: its numbers are different, and you saw them. If it is nearly the
  same series as the old one, it adds almost nothing to N_eff, because
  near-copies are de-correlated away;
- every trial stores the date of each of its returns, so runs on different
  windows (another `--start`) are cut to the weeks they share and
  de-correlated, rather than each counting as a whole trial. Older trials get
  their dates from their label's window when it states one exactly; the
  remaining undated ones count as whole trials.

Why record at all? The Deflated Sharpe Ratio discounts a result by the number of
things tried to find it. Twenty quick backtests while you "just look around"
are twenty trials. A ledger that does not see them overstates every later
result.

**Capital is fixed before validating, not searched.** The expert's rule: the
starting capital does not change the signal, so it is not a parameter like N or
K. With whole shares it does change results a little (a 25,000 book of four
names rounds differently from a 100,000 one). Choose the capital you will
trade, then validate. Running several capitals and keeping the one that
backtests best is selection on rounding noise, and those runs are trials. The
funnel varies it on purpose (half and double) as a friction test: the result
should not collapse. The ledger records every run anyway; runs that differ only
in capital are almost perfectly correlated, so they barely move N_eff.

**Note on the start date.** The cache reaches back to 1999 for some names. Early
years therefore have a thin universe: 6 names in 1999 against 39 today. A
backtest from the very start answers a different question from one starting
in 2009. Use `--start` to pick the window deliberately, and use the same window
when comparing variants.

### Step 5.2 — Check the causality guarantees (optional, after data changes)

```bash
python scripts/demo_causality.py
python scripts/reconcile_conventions.py
```

The first script checks, on the real data, three things: the universe is a
function of the decision time, a pinned view cannot see past it, and publication
lag is modelled. Silence means success. The second runs both fill conventions
through the same engine and prices the difference between them.

### Step 5.3 — Run the falsification funnel

```bash
ql funnel                    # 60 random-selection controls; several minutes
ql funnel --controls 20      # faster
```

The funnel puts the strategy through five gates on the real data, then prints
the verdict whatever it is. Exit code `0` means it passed and `2` means it did
not. Every backtest it runs is written to the ledger first. It also resumes: if
you interrupt it, running it again continues from what is already recorded,
without repeating work or counting a trial twice. A trial is only reused on the
same data: its label carries a fingerprint of the prices, so a re-fetch (with
dividends, say) or another universe runs afresh.

| option | default | what it does |
|---|---|---|
| `--top` / `--rebalance-weeks` / `--lookback` | 4 / 4 / 13 | the momentum parameters (N, K, lookback) |
| `--universe` | the whole store | a universe name or file (step 4.5) |
| `--start` | 2009-02-24 | first decision |
| `--capital` | 100,000 | fixed before validating; the funnel also runs half and double |
| `--controls` | 60 | random-selection controls (at least 20) |
| `--grid-top`, `--grid-rebalance` | off | grid mode (below) |

**The trial count is the whole research line.** The expert: another universe,
other N or K, a manual backtest, all belong to the same line of research, so
the Deflated Sharpe counts every trial of the strategy in the ledger, whatever
study recorded it (the random controls are not candidates and are left out).
Trials are aligned on their dates: the window most trials cover is chosen,
every trial covering it is cut to it and de-correlated (N_eff). Trials that
cannot be dated or do not cover it, and the 40 configurations tried before the
ledger existed, are counted at face value. The funnel prints how many were
aligned, on how many weeks, and how many count whole.

**Choosing N and K: the grid.**

```bash
ql funnel --universe sector-etfs --grid-top 1,2,3,4,5 --grid-rebalance 1,2,4,6,8
```

Every combination is backtested and recorded (study `grid`). For each, the 252
purged CPCV splits give a distribution of out-of-sample Sharpe, and its 10th
percentile (the worst tenth of splits) is the combination's score. The expert's
rule is a plateau, not a peak: each combination is judged by the worst score
among itself and its neighbours (adjacent N and K), and the best of those is the
plateau's centre. A peak that stands alone is printed separately, with a
warning. The grid's PBO says how often the in-sample best combination falls
below the median out of sample. Then validate the centre with the funnel; its
Deflated Sharpe already counts the grid. The expert suggests k-means clustering
for the plateau; on a grid of a few dozen points k-means depends on its seed,
so the neighbourhood rule asks the same question deterministically.

### Step 5.4 — Compare stop distances

```bash
python scripts/compare_stops.py
```

This runs the same strategy with several stop distances and records each run as
a trial. It shows what the stop actually buys: less drawdown, paid for in
positions cut just before they recovered.

### Step 5.5 — Compare execution rules

```bash
python scripts/compare_execution.py
```

This measures three execution rules against one baseline (decide on the weekly
close, fill at the next open, IBKR's tiered commissions, the account's present
size as opening capital):

- a minimum order size, as a share of equity and as a number of dollars;
- deciding on the close and filling at that close;
- deciding on the open and filling at that open;
- the minimum order size together with the better of the two fills.

It also prices the baseline's orders under each commission model for an account
that stays its present size, and repeats the main variants on the three other
four-week rotation calendars. Each variant is compared with the baseline on the
weekly difference of returns, with Newey-West errors. Every run is recorded as a
trial in the study `execution-rules`; the details go to
`state/scratch/compare_execution.json`.

Read two things before trusting a difference. "Decide on the open, fill at the
open" cannot be traded as simulated: the open is not known before the opening
auction prints it. And a difference that appears on one rotation calendar and
not on the others is timing luck.

### Step 5.6 — How many names to hold

```bash
python scripts/compare_positions.py
```

The strategy splits the book between the best `top_n` names that pass its
filters (`0`: every name that passes) and keeps -- freezes -- a held name that
is not among them but has triggered no exit, for at most `freeze_rotations`
rotations. This script takes the configured strategy with every name that
passes as its reference, then runs the best 4 to 10 names and each way of
handling frozen names (kept, kept for one or two rotations, halved, sold) on the
four rotation calendars, the pace filter both ways, and three single switches
(the stop-loss on cost, the stop type, limit orders). It reports how many
positions the book held and how small the smallest target was for an account of
the present size. It stops after `--budget` seconds and goes on where it left
off when run again. Every run is recorded in the study `position-count`.

The config in use since 2026-10-09 is `top_n: 5`, `freeze_rotations: 2`,
`pace_ratio_min: 0.1231` (pace compared per week) and `cost_stop_loss: 0.0`.

**One place for how the strategy trades.** `execution:` and `risk:` in the
strategy's config set the cash buffer, the no-trade band, limit or market
orders, the commission plan and the stop type. `ql backtest`, `ql monitor
baseline` and the live proposal all read them, so after changing one, rebuild
the baseline.

### Step 5.7 — Query the ledger directly

The ledger holds one JSON object per line and is easy to query:

```bash
python - <<'PY'
import json
rows = [json.loads(l) for l in open("state/research.jsonl")]
print(len(rows), "trials")
for r in sorted(rows, key=lambda r: -r["metrics"].get("sharpe", 0))[:5]:
    print(f'{r["metrics"]["sharpe"]:+.3f}  {r["study"]:<8} {r["window"]}  {r["note"]}')
PY
```

The Sharpe stored here is *weekly* (not annualised). It is the input the DSR
needs.

---

## 6. Reading the reports

Every report is written twice, as data (`.json`) and as a page (`.html`), side
by side in `state/reports/<id>/`, one folder per strategy. The page is drawn *only* from the data, so the two
can never disagree, and the JSON can be archived, diffed or loaded into anything.

The page is a single file with no internet dependency. Open it by
double-clicking it. It follows your system's light or dark mode, and the ◐
button switches between them.

Charts work the same way everywhere:

- **Hover** over a chart, or focus it with Tab and use ← →, to see every
  series' value at that week.
- Every chart has a **Table view** underneath with the exact numbers. A value
  in the tooltip is always also in the table.
- Colours are fixed: blue is the strategy or sleeve, orange the benchmark. Each
  line is also named at its right-hand end, so colour is never the only way to
  tell the lines apart.
- The four status colours (green, amber, orange, red) are used only for the
  state of the system. They always appear with a symbol (✓ ! ▲ ✕) and a word.

List recent reports, or re-draw a page from its data:

```bash
ql report list
ql report render state/reports/live-paper-20261003-101500.json
```

### 6.1 The backtest report

| element | what it tells you | how to read it |
|---|---|---|
| **CAGR**, with the benchmark underneath | compound annual growth | Compare over the same window. The benchmark line starts on the day its own data starts, at the strategy's equity on that day. |
| **Max drawdown** | the deepest fall from a peak | One path's worst case, not a bound. Monitoring uses the bootstrap to judge how deep a *normal* fall can be. |
| **Sharpe** | mean weekly excess return over its standard deviation, annualised | The conventional definition. The geometric variant, (CAGR − rf)/vol, is in the notes. It runs higher and is not comparable with published figures. |
| **Sortino** | as Sharpe, but against downside deviation only | |
| **Rotations, stops fired, commission** | how much the strategy traded | Stops fired is the price of protection. See step 5.4. |
| **Equity chart** | strategy against benchmark, same starting capital | Linear scale: for the size of falls, read the drawdown chart instead. |
| **Drawdown chart** | distance below the running peak | The shaded area is the strategy. Long, flat stretches below zero are the periods you would have had to sit through. |
| **Calendar-year returns** | year by year, with the difference against the benchmark | The first and last years are partial. |
| **Last rotations** | what the strategy chose to hold, and when | A sanity check: a strategy that holds the same names for years is not rotating. |
| **Settings** | exactly what was run | |

### 6.2 The live monitoring dashboard

Written by `ql monitor run` (section 10) to `state/reports/<id>/live-<mode>-<time>.html`.
Read it top to bottom. It is ordered by what needs your attention first.

1. **The status bar.** The state (NORMAL, REDUCE-ONLY or HALTED), whether it
   changed on this run, and every reason. Health problems, such as stale data,
   no recent sync or a mismatch, are listed too. They turn a NORMAL bar amber.
2. **The tiles.** Each tile is one check, marked with its own status where it
   has a threshold. Section 10.2 explains each check.
3. **Sleeve equity vs benchmark.** The sleeve's weekly equity from the
   snapshots. Deposits or corrections recorded with `ql live adjust --cash-delta`
   are removed from the returns, so they do not count as performance.
4. **Cumulative live return against the backtest's range.** The live cumulative
   return, with two dashed lines. They mark the 10th and 1st percentiles of the
   cumulative return that bootstrapped backtest paths reach over the *same
   number of weeks*. Sitting below P10 is unusual; below P1 is rare.
5. **Live drawdown.** How far below its peak the sleeve is.
6. **Implementation shortfall per rotation.** What each rotation cost against
   the decision price, with the backtest's modelled cost as a dashed line.
7. **Positions and protective stops.** Every position, its mark, its stop and the
   stop's distance below the mark. The monitor runs without the gateway, so
   these stops come from the journal; `ql live sync` checks them against the
   broker. An empty stop cell means a position is unprotected: run
   `ql live stops`.
8. **Rotations: cost and edge.** Per rotation: the shortfall, and the *alpha
   share*, meaning the shortfall as a fraction of the return a rotation is
   expected to earn.
9. **Thresholds in force.** The exact lines this run was judged against.
10. **Notes.** Especially "trend not judged before 26 weeks" and the warning
    that early statistics are wide.

For a strategy that is not weekly, the dashboard counts in its own bars (days,
hours): the title names the bar size, and "weekly returns" read as daily or
hourly ones. The thresholds' durations stay in calendar weeks (section 10.3).

---
## 7. Before trading: IB Gateway, the config, the baseline

### Step 7.1 — Install and configure IB Gateway

The system talks to IBKR through **IB Gateway** (or TWS) running on your Mac.
IB Gateway is the better choice: it is lighter, and it is built for API use.

1. Download IB Gateway (the "stable" version) from interactivebrokers.com and
   install it.
2. Log in to your **paper trading** account first. On the login screen, choose
   "Paper Trading". Your paper account number starts with `DU`.
3. In the gateway: *Configure → Settings → API → Settings*:
   - tick **Enable ActiveX and Socket Clients**;
   - leave the **Socket port** at **4002** (paper); the live default is 4001;
   - untick **Read-Only API**, otherwise orders are refused;
   - under **Trusted IPs**, keep `127.0.0.1`.
4. Market data: the API's historical bars need the same market-data permissions
   as the platform. For the paper account, in Client Portal enable sharing your
   live account's market data with the paper account (*Settings → Paper Trading
   Account*).

**Why the ports matter.** The system refuses a paper configuration pointed at a
live port (4001, 7496) and a live configuration pointed at a paper port (4002,
7497). A config pointed at the wrong gateway is how a test order becomes a real
one.

### Step 7.2 — Write your config

```bash
mkdir -p configs/strategies
cp configs/live.example.yaml configs/strategies/momentum.yaml
```

Edit `configs/strategies/momentum.yaml`. Name the file after the strategy's id.

| setting | what to put | why |
|---|---|---|
| `strategy_id` | `momentum` | Lowercase letters, digits and single dashes, at most 16 characters, starting with a letter. It is written on every order (`ql-momentum.…`) and names the strategy's state folder. **Never change it on a running sleeve**: the journal is filed under it, and the system refuses a journal that belongs to another id. |
| `mode` | `paper` | Always start here (section 12). |
| `account` | your `DU…` account | Checked against the accounts the gateway manages, and against the mode: `DU` is paper, `U` is live. |
| `sleeve_capital` | the capital the strategy manages | It sizes against this, not the whole account. |
| `gateway.port` | 4002 | Must match the gateway (step 7.1). |
| `gateway.client_id` | any number not used by another API program | Two programs with the same id disconnect each other. |
| `account_scope` | `dedicated` | The account belongs to this strategy: a position the sleeve does not know, or extra shares, stops the system. Use `shared` only if you must keep other holdings in the same account; they are then reported, not treated as an incident. |
| `strategy.name` | `weekly-momentum` | Which strategy. The deployable ones are registered in `runtime/strategies.py`; an unknown name is refused. |
| `strategy.params` | the values you researched | Changing any of these changes the strategy's identity and requires a new baseline. |
| `strategy.universe` | optional: a universe name or file | The list it chooses from (step 4.5). Omitted: every instrument in the store. An open sleeve refuses a different one. |
| `execution.time_in_force` | `auto` | `auto` sends orders to the opening auction (`opg`) for daily and weekly strategies, and as day orders for intraday ones. |
| `risk.stop_distance` | 0.12 | The protective stop: below a long, above a short. `0` disables it (not recommended). |
| `risk.max_gross` | 1.0 | Longs plus shorts, as a share of sleeve equity. |
| `risk.max_net` / `risk.min_net` | empty / −1.0 | Longs minus shorts. Empty `max_net` means "same as `max_gross`": for a long-only book net *is* gross, so a lower value would forbid leverage. |
| `risk.allow_short` | `false` | Section 7.6. Off, a strategy that asks for a short fails loudly instead of borrowing stock. |
| `risk.max_borrow_fee` | empty | Refuse a short whose annual borrow fee is above this, when IBKR reports a fee. |
| `risk.max_order_fraction` | 0.6 | No single order may *open* more than this share of the sleeve. It is a guard against unit errors, not a sizing rule. |
| `leverage.*` | the defaults (no borrowing) | Step 7.7. `leverage.target` above 1.0 borrows; it may not exceed `risk.max_gross`, the hard cap. |
| `financing.margin_rate` / `financing.borrow_fee` | 0.055 / 0.005 | Annual interest on borrowed cash and fee on borrowed stock, charged in backtests and accrued in the live sleeve. Check IBKR's current rate for your balance tier. |
| `automation.mode` | `manual` | The most the system may do without you: `manual`, `exits` or `full`. Nothing is automatic until you also arm it (step 9.9). |
| `automation.*` (other) | the defaults | The safety gates of the automatic mode, step 9.9. |
| `monitoring.*` | the defaults | Durations are in **calendar weeks** whatever the bar size. Section 10.3 explains each one and when to change it. |
| `monitoring.max_data_age_hours` | empty | Empty means by bar size: 240 hours for weekly bars, 100 for daily, 72 for intraday. |
| `proposal_ttl_hours` | 60 | A Saturday proposal must still be approvable on Monday morning. A proposal is also dead as soon as a newer bar has closed. |
| `unmanaged` | tickers the strategy must never touch | Only meaningful with `account_scope: shared`. |

`configs/strategies/` is gitignored because it names your accounts. An older
`configs/live.yaml` still works if you add a `strategy_id` to it; moving it to
`configs/strategies/<id>.yaml` is the tidy option.

Check what is configured:

```bash
ql strategies
```

It lists every strategy with its bar size, mode, account, capital and whether its
sleeve is open. It refuses, as every command does, when two configs share an id
in the same mode or two strategies share an account.

### Step 7.3 — Check the connection

With the gateway running and logged in:

```bash
ql live status
```

It first names the strategy, for example `strategy momentum (weekly-momentum,
weekly bars)`. Before the sleeve is opened it then prints `connected DU… (paper)`
with the account's net liquidation value, and `sleeve not open yet`. That means the gateway, the
port and the account all check out. If it fails, the message says why: the
gateway is not running, the port is wrong, or the account is not one this
gateway manages. The account is checked on every connection, not only here.

### Step 7.4 — Build the monitoring baseline

```bash
ql monitor baseline
```

**What it does.** It backtests the configured strategy (same universe, settings,
stop and costs) over the store and saves three things to
`state/live/<id>/<mode>-baseline.json`:

- the returns per bar (weekly, for momentum);
- the execution cost the backtest paid;
- the return an average rotation earned.

**Why.** Monitoring asks one question: *is what is happening live consistent
with what the backtest said would happen?* The baseline is "what the backtest
said".

- The baseline is **tied to the settings**. If you change a strategy parameter,
  the stop, an exposure limit or `allow_short`, monitoring refuses to run until
  you rebuild it. This stops live results from being compared against a different
  strategy.
- **Rebuild it** after changing the strategy settings, and after expanding the
  universe.
- **Don't rebuild it** week after week just to fold in new data. A reference
  that moves every week can quietly absorb a problem.

### Step 7.5 — Running more than one strategy

Each strategy runs in **its own IBKR account**. That is IBKR's own way of keeping
money apart: positions, cash, margin and orders never mix, and each strategy's
reconciliation stays "this account against this sleeve".

1. **Open an account for it.** In Client Portal, add a linked account under the
   same login and fund it with the strategy's capital. Trade it on paper first,
   like the first one.
2. **Write its config**, `configs/strategies/<id>.yaml`, with its own
   `strategy_id` and that account's number. Give it a `gateway.client_id` of
   its own.
3. **Gateway.** If the accounts share a login, one IB Gateway serves them all,
   and the system checks that each config's account is one the gateway manages.
   If they do not, run a second gateway on another port and put that port in the
   config.
4. **Check** with `ql strategies`.
5. **Name the strategy in every command** once more than one is configured:

   ```bash
   ql --strategy trend live sync
   ql --strategy trend live propose
   ql --strategy trend monitor run
   ```

   Without `--strategy`, a command refuses and lists the configured ones:
   guessing which account to trade in is not a default. `--strategy` names the
   config file, `configs/strategies/<name>.yaml`, which is normally named after
   the id. With a single strategy configured, no flag is needed.

What keeps two strategies apart:

| what | how |
|---|---|
| orders | Every order's reference starts with `ql-<id>.`. A strategy only claims fills, and only manages stops, that carry its own prefix. The dot cannot occur in an id, so `trend` can never claim `trend-fx`'s orders. |
| state | `state/live/<id>/` holds its journal and baseline; `state/reports/<id>/` its reports. |
| account | Checked against the gateway on every connection. The journal records the account at `init`, and the system refuses to run a sleeve against another one: a sleeve does not move between accounts. |
| config | No command runs while two configs share an id in the same mode, or two strategies share an account. |

**Why not two strategies in one account.** It is possible: the project's expert
describes it as virtual sub-accounts. But the broker would then net their orders
and hold one combined position per name, so the system would need to split every
fill between strategies, reconcile against a sum, and decide who owns a stop.
Each of those is a new way to be wrong about money. Separate accounts need none
of it, so it is not built, and `ql` refuses the configuration.

**Paper next to live, for one strategy.** Give it two files with the same
`strategy_id`: for example `momentum.yaml` (`mode: live`) and
`momentum-paper.yaml` (`mode: paper`, the paper account and port). Select the
paper one with `ql --strategy momentum-paper …`. Their journals are
`state/live/momentum/live-journal.jsonl` and `paper-journal.jsonl`.

### Step 7.6 — Strategies that are not weekly, or can be short

The momentum strategy is weekly and long-only. Nothing else in the system
assumes either.

**The bar size** comes from the strategy itself, and `ql strategies` shows it.
Everything periodic follows it:

| what | how it follows the bar size |
|---|---|
| data | `ql data refresh` fetches that bar size, stamped as in section 3. |
| the cycle | Section 9 runs once per bar: for a daily strategy, every evening after the close or every morning before the open. |
| orders | `execution.time_in_force: auto`: opening auction for daily and weekly strategies, day market orders for intraday ones. |
| proposals | A proposal is dead once a newer bar has closed, whatever `proposal_ttl_hours` says. |
| data age | `propose` refuses bars older than 240 hours (weekly), 100 hours (daily, so Friday's bar still serves on Monday) or 72 hours (intraday), unless `monitoring.max_data_age_hours` says otherwise. |
| monitoring | Its durations are calendar weeks and are converted into bars: a 6-week bootstrap block is 6 weekly bars or about 29 daily bars. The expert's rule: how long the market remembers, and how long a regime lasts, are properties of calendar time, not of how often you sample it. |

**Intraday: what it takes, and what is already done.** The project's expert
lists five prerequisites before an intraday strategy trades real money. Where
each stands:

| prerequisite | status | how |
|---|---|---|
| the exchange's session calendar: holidays, 13:00 half days, the regular session apart from pre- and after-market | **done** | `data/calendar.py`, from the maintained `exchange_calendars` package. Intraday bars outside the regular session are dropped; a bar ends at the session close if that comes first; `propose` refuses unless the last bar that should have closed is in the data. |
| 3-5 years of intraday history (the market's regimes are counted in years, not bars), cleaned: split-adjusted, no impossible bars, regular session only | **tooling done; the data is yours to fetch** | `ql data backfill --interval hour --years 5` pages back through IBKR's TRADES history at IBKR's pacing limit (60 requests per 10 minutes), resumably, and says where the broker's history ends. Two daily requests per symbol first give the dividend factors for the span (IBKR cannot page adjusted data); the first page reaches one day into the cache, and a split since the cache was written starts the symbol again from now. Hourly bars for the whole universe take roughly an hour; minute bars about half a day. Then `ql data ingest --rebuild`. `propose` refuses an intraday strategy with less than 3 years in the store. |
| a strategy validated on that data, with an intraday cost model (spread paid, square-root market impact, per-share commission) and the funnel's purge and embargo set by the trade's lifetime and the longest feature memory | **not started** | This is research, not plumbing: no intraday strategy exists yet. The funnel runs on any bar size; the cost model needs the spread and impact terms added before its results mean anything. |
| a process running through the session | **hourly: done; minute: not built** | `ql live auto schedule` writes a Mac launch job that runs `ql live cycle` every hour of the US session. Minute bars need a program that runs continuously; that is not built. |
| automation | **done** | A person cannot approve every hour. Step 9.9. |

Two things outside the code. **The pattern-day-trader rule** (at least $25,000
to day-trade a US margin account) was scrapped by FINRA with effect from
4 June 2026, but brokers may keep applying it until October 2027 while they
implement the replacement: check what IBKR shows for your account before
counting on it. **The Mac and IB Gateway** must be on through the US session
(15:30-22:00 in Spain) every trading day.

So the earliest date is set by the research, not the code: when an intraday
strategy has passed the funnel on 3-5 years of backfilled data with intraday
costs. Then paper, as for any strategy (section 12).

**Shorts** are off unless `risk.allow_short: true`. With them on:

1. **The account must be a margin account.** IBKR refuses short sales in a cash
   account.
2. **Before any proposal that opens a short, the system asks IBKR:**
   - *how many shares can be borrowed.* A short is cut to that number, and a
     name with no figure is treated as not borrowable;
   - *what the orders do to margin*, with a what-if order. The proposal is
     refused if the account cannot carry them;
   - *the borrow fee*, checked against `risk.max_borrow_fee`. The standard API
     usually does not report fees, so check hard-to-borrow names in TWS.
3. **Exposure** is limited by `max_gross` (longs plus shorts) and by
   `min_net`/`max_net` (longs minus shorts).
4. **Each short gets a buy stop** `stop_distance` above its entry. The expert's
   warnings about the short side: the loss is not bounded by zero; a gap up
   fills the stop above its level; crowded shorts get squeezed; and the lender
   can recall the shares, which forces a buy-in.
5. **Reduce-only** covers shorts but never opens or enlarges one. An order that
   would reverse a short into a long stops at flat. A liquidation buys to cover.
6. **Reconciliation compares signed positions.** A long where the sleeve holds a
   short is a mismatch.
7. **Backtests** charge a flat borrow fee (`financing.borrow_fee`) but not the
   higher fees of hard-to-borrow names, and do not model recalls.
8. **`ql live adjust`** takes negative quantities for shorts, and only for a
   strategy that allows them.

### Step 7.7 — Leverage

The system can hold more than 100% of the sleeve's equity, on a margin account.
It is off by default (`leverage.target: 1.0`). The rules, from the project's
expert:

1. **A hard cap: `risk.max_gross`.** The expert's range for a concentrated,
   four-name book with a 30-40% drawdown tolerance is 1.3x-1.5x. Nothing the
   rules below compute can exceed it.
2. **The level: fixed, or set by tail risk.** `leverage.target` is a fixed
   leverage. Alternatively `leverage.cvar_target` sets it so that the book's
   expected shortfall per bar (the average of its worst 5% of bars) matches the
   target, measured on the strategy's own history with the leverage divided
   out. Kelly-style sizing is not used: it assumes a return distribution known
   exactly, and with fat tails it overshoots any drawdown tolerance.
3. **Cut in drawdown, convexly.** Past a 10% drawdown, the borrowed part shrinks
   with the square of how far the drawdown has gone towards 35%; at 35% it is
   gone and the unlevered strategy is left. The expert scales the whole book
   towards zero; here only the borrowing is removed, because below 1x the
   strategy is no longer the one that was validated, and the unlevered
   strategy's drawdowns are governed by monitoring (`leverage.floor` below 1
   restores the expert's version).
4. **Cut at once, rebuild slowly.** A lower level applies at the next decision;
   a higher one rises at most 0.05x per calendar week, and only while equity is
   improving. Between rotations the book is only ever cut, never topped up.
5. **The margin cushion.** IBKR does not make margin calls: when the cushion
   (1 − maintenance margin ÷ net liquidation) reaches zero it sells, at market,
   without notice. Below a 25% cushion the system goes reduce-only and sizes
   the next decision back to 1x; below 35% it adds nothing. It lifts its own
   reduce-only when the cushion is back above 35%.
6. **It costs money every day.** Backtests charge interest on the debit balance
   and a fee on borrowed stock (`financing.*`); the live sleeve accrues the same
   estimate at each sync. Before a proposal that borrows, IBKR's what-if check
   confirms the account can margin it.

To use it: set `leverage.target` (and `risk.max_gross` at least as high), make
sure the account is a margin account, rebuild the baseline, and backtest first:

```bash
ql backtest --leverage 1.3 --start 2009-01-01
```

What it bought on the momentum strategy (September 2026, 5.5% interest, each
run recorded in the ledger):

| window | leverage | CAGR | Sharpe | max drawdown | interest paid |
|---|---|---|---|---|---|
| 2009-2026 | 1.0x | 20.3% | 0.95 | −24.0% | — |
| 2009-2026 | 1.3x | 24.9% | 0.93 | −30.6% | 148k |
| 2009-2026 | 1.5x | 27.3% | 0.89 | −42.4% | 294k |
| 2000-2026 | 1.0x | 12.1% | 0.60 | −69.8% | — |
| 2000-2026 | 1.3x | 13.8% | 0.60 | −74.1% | 108k |

Leverage buys return, not quality: the Sharpe ratio does not improve, and the
drawdown grows with it. At 1.5x the 2009-2026 drawdown is past a 40% tolerance.
The 2000-2009 years, with a thin early universe, show the base strategy's
drawdown is already beyond it unlevered.

---

## 8. Opening the sleeve

### Step 8.1 — Open it, once

```bash
ql live init
```

This records the sleeve's opening balance: `sleeve_capital` in cash and no
positions. It then runs a first sync: it snapshots the account and reconciles
the (empty) sleeve against it.

A sleeve opens once per strategy and mode. The paper and live sleeves have
separate journals (`paper-journal.jsonl`, `live-journal.jsonl`, in the
strategy's folder), so paper history never leaks into live. The opening entry
records the strategy id, its version, its bar size and the account.

### Step 8.2 — Or hand over positions you already hold

```bash
ql live init --adopt AAPL,MSFT
```

Adopted positions enter the sleeve at the quantity and average cost IBKR
reports. The sleeve's cash is whatever of `sleeve_capital` they do not already
use. From then on the strategy manages them like any other position: it may sell
them at the next rotation, and it protects them with stops anchored at the
opening price.

Adoption is refused when:

- the account does not hold the name;
- the name is listed as `unmanaged`;
- the name is not in the price data;
- the adopted positions are worth more than `sleeve_capital`.

**Why adopt rather than sell and rebuy.** It avoids a round trip of costs and
taxes for names the strategy would hold anyway.

---

## 9. The weekly cycle

The weekly bar closes on **Friday at the US close**. Everything below happens
between then and the **Monday open**. Rotations happen every
`rebalance_weeks` (4 by default). The other weeks are "hold weeks": the
cycle still runs, but the proposal is usually empty.

A strategy with another bar size runs the same steps once per bar (section 7.6).
With several strategies configured, add `--strategy <name>` to every command.

Keep IB Gateway running and logged in for the steps that need it (marked 🔌).

### Step 9.1 — Refresh the data 🔌 (Saturday)

```bash
ql data refresh
```

This adds the week that closed on Friday (section 3.1). **Why first:** the
proposal is made on the latest complete week. `propose` refuses if that week
closed more than 240 hours (10 days) ago, so a forgotten refresh cannot turn into
a stale decision. The limit depends on the bar size (section 7.6).

### Step 9.2 — Sync 🔌

```bash
ql live sync
```

Sync brings the journal up to date with the account, in this order:

1. **Fills.** Every execution of an order this strategy sent (recognised by
   its order reference, `ql-<id>.…`), including stops that fired, is recorded
   once. Duplicates are ignored by execution id. Orders placed by hand carry no
   such reference, so their fills are never claimed.
2. **Order statuses.** Orders that filled, were cancelled or were rejected are
   recorded.
3. **Stops.** Every sleeve position gets a good-till-cancelled protective stop:
   a sell stop below a long, a buy stop above a short. Positions that no longer
   need one have theirs cancelled.
4. **Snapshot.** The sleeve is marked at the latest closes. This is the weekly
   equity point that monitoring uses.
5. **Reconciliation.** The sleeve is compared with the account:
   - **OK**: they agree.
   - **WARN**: in a *shared* account, the account holds more than the sleeve,
     for example your own other positions. A missing stop and low account cash
     are also warnings.
   - **MISMATCH**: the account holds *less* than the sleeve believes, holds it on
     the other side (long where the sleeve is short, or the reverse), an order
     the system sent has vanished, or, in a *dedicated* account, it holds
     anything the sleeve does not. **The system halts**, because it would
     otherwise be sizing and protecting positions that do not exist. See 11.1.

Proposals require a reconciliation less than 24 hours old, and without a
mismatch. That is why sync comes before propose.

### Step 9.3 — Monitor (no gateway needed)

```bash
ql monitor run
```

This runs every check (section 10). It applies the resulting state change, if
any, and writes the dashboard. Open the page it prints. **Why before
proposing:** if monitoring moves the system to reduce-only or halted, the
proposal must already respect that.

### Step 9.4 — Propose 🔌

```bash
ql live propose
```

The strategy decides on the latest complete week, exactly as it did in the
backtest. **Nothing is sent.** The proposal is printed and recorded:

- the proposal id, what kind it is (ROTATION, HOLD WEEK or LIQUIDATION), and
  when it expires;
- the current and target weights of every name;
- each order: side, instrument, quantity, type (market, `opg` = opening
  auction), estimated price and value, and share of equity;
- any risk findings, for example orders removed because the system is
  reduce-only.

Things `propose` refuses, and why:

| refusal | why |
|---|---|
| no reconciliation, or one older than a day, or a mismatch | Sizing from a sleeve that may not match the account. |
| orders from the last approval still working | Two sets of orders for the same shares. |
| data older than its limit (10 days for weekly bars) | A decision on stale prices. |
| the system is halted | Only `--liquidate` is allowed (11.2). |
| a short the account cannot margin, or a broker that cannot report margin | A short sale IBKR would refuse, or one nobody checked (7.6). |
| a short target from a strategy without `allow_short` | A short by accident fails instead of borrowing stock. |

A short that cannot be borrowed is not a refusal: the order is cut to what can
be borrowed, possibly to nothing, and the finding says so.

Names that a stop closed since the last rotation are not bought back until the
next rotation. The backtest does the same.

### Step 9.5 — Approve, or reject 🔌 (before Monday 09:28 New York time)

```bash
ql live approve
```

This shows the pending proposal's orders and asks you to type its confirmation
code:

- in **paper** mode, the proposal id, e.g. `P2655AA`;
- in **live** mode, `LIVE` followed by the id, e.g. `LIVE P2655AA`. It is longer
  on purpose, so a live approval can never be a reflex.

Anything else sends nothing. Before sending, `approve` checks four things:

- the proposal is the latest undecided one;
- it has not expired;
- no newer bar has closed since it was decided (it would be a decision on old
  data);
- the sleeve has not changed since the proposal was computed (a fill, a stop or
  an adjustment in between makes its quantities wrong);
- the system has not been halted since.

It then does the following:

- cancels any resting stop on a name it is about to trade on the same side: a
  sell against a long's stop, a buy against a short's. Otherwise the stop and
  the order together would close the same shares twice, and push the position
  through zero;
- sends the orders that reduce exposure first, then those that add to it, as
  market-on-open orders, each tagged with the strategy's reference and a unique
  code so a retry can never send an order twice.

**Timing.** Market-on-open orders must reach IBKR before the opening auction
closes to new orders, at **09:28 New York time**. That is normally 15:28 in
Spain; for a couple of weeks in March and in October/November the offset is one
hour less. The proposal stays approvable for 60 hours, so Saturday to Monday
morning is comfortable.

**To decline** a proposal:

```bash
ql live reject P2655AA --reason "Earnings on Tuesday for two of the names; skipping this rotation"
```

A rejection is legitimate, but it is recorded as an **override**. Monitoring
measures overrides: how often you skip, and whether you skip buys and sells
equally (section 10.2).

### Step 9.6 — Sync after the open 🔌 (Monday, after 09:30 New York)

```bash
ql live sync
```

This records the opening-auction fills and places the new stops, anchored at
each position's actual fill price. **Until this runs, new positions have no
stop.** Run it as soon after the open as you can.

### Step 9.7 — Check

```bash
ql live status
```

This shows the state, sleeve equity and cash, the time since the last sync,
the last reconciliation, and every position with its mark and its stop. Every
position should show a stop.

### During the week

Nothing is required. Stops rest at IBKR and work without the system running.
If one fires, the next `ql live sync` records it as a fill. The system does
not rebuy that name until the next rotation.

### Step 9.8 — The whole cycle in one command, or on a schedule

```bash
ql live cycle
```

Runs steps 9.1 to 9.4 in order -- refresh, sync, monitor, propose -- and then,
only if automation is armed (step 9.9), sends what its gates allow. Disarmed, it
stops at the proposal and prints the command to approve it. It is safe to run as
often as you like: a bar is proposed on once, an order is sent once, and a run
with nothing new does nothing. `--no-refresh` and `--no-monitor` skip those
steps.

To have the Mac run it for you:

```bash
ql live auto schedule
```

This writes a macOS launch job (in the strategy's state folder) and prints the
two commands to install and to remove it. For a weekly strategy it runs on
Saturday at 10:00 (refresh, propose, and send if armed), on Monday at 16:05
(after the US open: record the fills, place the stops) and every weekday at
22:45 (record any stop that fired). Times are the Mac's local time. Run the
command on your Mac, not elsewhere: the job uses the Python it was written with.

What it depends on:

- **The Mac must be awake or asleep, not off.** A job missed during sleep runs
  when the Mac wakes; a Mac that is shut down misses it. *System Settings →
  Battery → Options* can keep a plugged-in Mac from sleeping, or
  `sudo pmset repeat wakeorpoweron MTWRFS 09:55:00` wakes it before the
  Saturday run.
- **IB Gateway must be running and logged in.** Set *Auto restart* in the
  gateway's settings; IBKR still requires a full login with two-factor
  authentication **once a week, from Monday**. Log in each Monday before the
  open. A run that cannot connect sends nothing and says so in the log
  (`state/live/<id>/cycle.log`).

### Step 9.9 — Automatic sending (off until you arm it)

The system can send orders without your typed approval, within limits, if you
allow it twice: in the config, and with a typed phrase.

**1. Allow it in the config** (the most it may ever do):

```yaml
automation:
  mode: exits    # manual | exits | full
```

**2. Arm it:**

```bash
ql live auto arm --scope exits     # type: AUTO EXITS momentum
ql live auto status                # what is armed, and the evidence so far
ql live auto disarm --reason "..."  # back to manual
```

In live mode the phrase starts with `LIVE`.

**The two scopes**, in the order the project's expert prescribes:

- **`exits`**: orders that reduce exposure -- selling a long, covering a short,
  cutting leverage -- are sent automatically. Orders that add exposure stay
  pending for your `ql live approve`, which then sends only what is left. This
  automates first what people do worst: the record shows skipped exits, not
  skipped buys.
- **`full`**: everything the gates pass. On paper you may arm it at once. With
  live money it needs evidence from the `exits` stage, from the live or the
  paper journal: at least 8 weeks armed for exits, no reconciliation mismatch,
  and automatic exits filling within 1.2x the modelled cost. `--override
  "reason"` proceeds without it; the reason is recorded.

**The gates** every automatic send passes. A gate that fails holds the orders
for you and says why; nothing is retried blindly.

| gate | holds | rule |
|---|---|---|
| proposal | everything | The same checks as your approval: latest, unexpired, no newer bar, sleeve unchanged, not halted. A liquidation is never automatic. |
| reconciliation | everything | A clean reconciliation within the last hour. |
| order count | everything | At most `automation.max_orders` (20). |
| ladder | entries | Reduce-only lets exits through, nothing else. |
| margin cushion | entries | At least `automation.min_cushion` (35%). |
| order size | entries | No order opening more than 10% above the largest order the backtest ever sent (or `automation.max_order_fraction`). |
| turnover | entries | No cycle trading more than both the backtest's 99th-percentile rotation and one whole book. |

And one breaker: **a bar whose return is below the backtest's 0.5% quantile
halts the system**, which disarms automation. It judges each bar once, so after
you have looked and cleared the halt it does not fire again for the same bar.

**What never happens automatically:** lifting a halt, re-arming after one,
trading on a book that did not reconcile, liquidating, or changing a limit.

**Why these numbers are not the expert's.** The expert's example caps are 5% of
equity per order and 30-40% turnover per cycle, and a 4% daily loss breaker.
They describe a strategy that slices its orders and seldom replaces its book. A
four-name momentum book opens 25% positions and replaces most of itself at a
rotation, and a 4% weekly loss is an ordinary week. So each limit is set from
the strategy's own backtest instead: what it has never done is what gets held.

### Step 9.10 — The weekly review where there is no gateway

Everything above needs IB Gateway on this Mac. `ql review` makes the same
decision where there is none: in a scheduled Claude session that reaches the
broker through the IBKR connector. The session fetches and reports; the rules
are this repository's, so the weekly review, the backtest and the paper sleeve
run one strategy.

```bash
ql review run --state state.json --state-out state.json --out review.json
```

**What it reads.** By default the running session's own log
(`~/.claude/projects/…`, and the logs of any helper the session started), where
every connector answer is already stored verbatim: retyping a price history
costs it twice and invites a wrong digit. `--inputs DIR` reads a folder of
saved answers instead (`--save-inputs DIR` writes one). The session has to have
asked for:

| answer | connector tool | for |
|---|---|---|
| the watchlist | `get_watchlist` | naming contracts |
| weekly bars, two years, with corporate actions, one call per stock and the benchmark | `get_price_history` | signals |
| account summary, positions, working orders | `get_account_summary`, `get_account_positions`, `get_account_orders` | the book, the stops |
| the last seven days' trades | `get_account_trades` | last rotation's fills |
| the performance series | `get_pa_performance_all_periods` | the live record |
| a quote per stock to trade or protect (rotations only) | `get_price_snapshot` | limit prices, stop levels |

**What it decides, and with what.** Nothing here is written twice:

| part | comes from |
|---|---|
| filters, exits, ranking, frozen positions, weights | `strategies.momentum.WeeklyMomentum` |
| whole shares, what is too small to send | `engine.decide.decide` |
| stop and limit levels | `risk.rules.ProtectiveStop` |
| the degradation ladder | `validation.monitoring.assess` |

**Prices.** The connector's bars are adjusted for splits, not dividends; the
store's are adjusted for both, and the expert's rule is that signals use the
total-return series in research and in production alike. The review rebuilds
it from the dividends the connector lists (every bar before an ex-dividend
week is scaled by `1 - dividend / previous week's close`). Against the store,
on the connector's own answers of 2026-10-09: the adjusted closes of all 53
stocks agree within 0.12% over two years (unadjusted, they are up to 11.6%
apart), and the scan of that week is the same stock for stock. Orders, stops
and the value of the book use prices as traded.

**A rotation takes two runs.** The first says which stocks it would trade or
protect and exits with status 3 (`NOT FINAL`); the session quotes them and
runs it again. Orders are day limit orders 0.5% through the quote
(`review.limit_offset`), sized by the engine; stops are replaced for the book
as it will be after the trades. A person approves each order at the broker:
the review sends nothing. A rotation chosen from part of the universe is not
final either: a stock with no current history is named, to be fetched.

**State.** A scheduled session remembers nothing, so `--state-out` writes one
small JSON document to keep until the next run (the weekly task stores it as a
project document): the ladder's state, each rotation's proposals and the fills
that followed, and the weekly returns. It is written only when the review is
final.

**Monitoring.** The live return is read each week from the broker's
time-weighted series, which a deposit does not move. Three things differ from
`ql monitor run`, all of them the expert's:

- the record starts at `review.monitor_from`, the first rotation traded on this
  strategy version: weeks traded under other rules are another strategy's;
- for the first `review.burn_in_weeks` (12) the drawdown and changepoint checks
  are shown but do not move the ladder; the execution-cost check and the loss
  breaker act from the first week;
- last week's return is cross-checked against what the positions held a week
  earlier did, and a gap with no trade to explain it is flagged.

A halted review proposes nothing and leaves the stops alone; reduce-only lets
exits through. Lifting either is a person's decision:

```bash
ql review clear --state state.json --to normal --reason "what you checked"
```

**The definition and the baseline.** `ql review` has no private config to read.
It runs from `configs/definitions/momentum.yaml` (the strategy with no account
in it; your private config takes the same file in with `extends:`) and from
`baselines/<strategy version>.json`. After changing a rule:

```bash
ql monitor baseline          # rebuild the reference on the store
ql review baseline           # publish it under the new version's name
git add configs/definitions baselines && git commit && git push
```

and move `review.monitor_from` to the first rotation on the new rules. A
review with no baseline for its version says so and judges nothing.

---
## 10. Monitoring live performance

### 10.1 What monitoring is for

A four-week rotation marked weekly gives about 52 observations a year. On 52
points, a mean or a t-statistic cannot tell a real loss of edge from a bad run.
Acting on one produces both kinds of error: switching off a healthy strategy
after bad luck, and keeping one whose edge has gone.

So monitoring never compares live results with one backtest number. It places
every live statistic inside a **distribution built from the backtest**, for a
window as long as the live record. The question it answers is always the same:
*how unusual is this for this strategy?*

Run it weekly (step 9.3), or at any time:

```bash
ql monitor run              # judge, apply the state, write the dashboard
ql monitor run --dry-run    # judge and write the dashboard; change nothing
ql monitor run --no-report  # judge and apply; no page
```

It needs neither the gateway nor market hours. It reads the journal and the
store.

### 10.2 The checks

**1. Drawdown against the bootstrap.**

- *What.* A stationary bootstrap (Politis–Romano) resamples the baseline's
  weekly returns in blocks of random length. It keeps the volatility
  clustering and persistence that an independent resample would destroy, and
  builds 5,000 synthetic paths as long as the live record (at most 52 weeks).
  The live drawdown's percentile among those paths' drawdowns is the tile
  *Live drawdown: deeper than N% of backtest paths*.
- *Why the matched length.* A 10-week live drawdown compared with one-year
  drawdowns would look mild when it is not.
- *Lines.* Reduce-only above the 80th percentile, halt above the 99th.
- *Same paths, second test.* The live cumulative return is compared with the
  paths' 10th and 1st percentiles (the dashed lines on the dashboard). Below
  P10 is reduce-only; below P1 is halt.

**2. Break probability (online changepoint detection).**

- *What.* Bayesian online changepoint detection (Adams–MacKay) runs over the
  backtest and then the live weeks. It reports the posterior probability that
  the return process *changed after going live*.
- *Lines.* Reduce-only at 20%, halt above 50%.
- *Known limit, measured.* It is sensitive to a change in *volatility* and
  weak at detecting a pure fall in the *mean* over a few months. That case is
  exactly what check 1 catches, which is why both exist.

**3. Trend.**

- *What.* The median weekly return of the live record, with a bootstrap
  confidence interval.
- *When it counts.* Only after 26 live weeks, and only when the **whole
  interval** is below zero. Before that, the tile says "not judged yet".
- *Line.* A significantly negative trend halts.

**4. Implementation shortfall.**

- *What.* For every rotation: what the fills cost against the price at
  decision time, in basis points of the traded value. It
  is compared with what the backtest paid.
- *Why.* It separates "the signal stopped working" from "the fills got worse".
  They call for different responses.
- *Supporting numbers.*
  - Execution drag: the fill against the bar's opening price. This is the part
    the backtest cannot model.
  - Markouts: the price move 1 and 4 calendar weeks after the fill, in the
    trade's favour (up after a buy, down after a sale). Persistently negative
    markouts mean you are trading at turning points: buying tops, or selling and
    shorting bottoms.
- *Lines.* Reduce-only at 1.5× the modelled cost. Halt when the cost exceeds
  half of the return a rotation is expected to earn, two rotations in a row:
  execution would then be eating the edge.

**5. Process: how you and the system work together.** These are not
thresholds, but they are shown on every dashboard:

| metric | meaning | watch for |
|---|---|---|
| approved / proposed, rejected, expired | how often proposals are acted on | Expiries are skipped weeks by default. |
| entry and exit compliance | share of proposed entries and exits that were executed. Entries open or add to a position (buying a long, selling a short); exits reduce or close one | |
| **asymmetry** | entry compliance minus exit compliance | Positive means you execute entries and skip exits. It is the most expensive override habit: it keeps losers the system wanted out of. Amber at +10%, red at +25%. |
| override cost | what the rejected orders would have made over the following week, in dollars | Positive means the overrides cost money. |
| median latency | hours from proposal to approval | |
| stop coverage | share of positions with a working stop | Anything under 100% is an unprotected position. |

**6. Health.** Data age (past the limit `propose` uses: 10 days for weekly
bars), time since the last sync (over one bar plus three days: 10 days for a
weekly strategy, 4 for a daily one), and whether the last reconciliation was a
mismatch. Health problems do
not change the state, but they turn the status bar amber: every other check is
only as good as the data it reads.

### 10.3 The thresholds, and when to change them

The defaults are in the strategy's config under `monitoring:`. The durations
(`bootstrap_block_weeks`, `changepoint_hazard_weeks`, `trend_min_weeks`,
`horizon_weeks`) are calendar weeks for every bar size (section 7.6). They come from the
project's expert, with one deliberate change: **the drawdown halt is at the 99th
percentile, not the 95th.**

Why: on healthy synthetic data, where nothing was wrong, the checks were run
thousands of times and their false alarms counted.

| drawdown halt at | false halts per weekly assessment (healthy system) | real collapse halted within 13 weeks / 26 weeks |
|---|---|---|
| 95th percentile | 5–13% | — |
| 99th percentile | 0–5% | 60% / 87% |

The reduce-only line at the 80th percentile trips on 27–37% of assessments of
a healthy system. That is intended: reduce-only is cheap, because existing
positions keep running with their stops. It lifts itself when the evidence
clears. A halt is expensive, so its line is set where healthy systems rarely
reach it. With a stated drawdown tolerance of 30–40%, the 99th percentile is
the expert's own "high tolerance" option.

**Change a threshold only in the config, only with a reason, and not in
response to the current reading.** Moving a line because it has just been
crossed is switching the alarm off.

### 10.4 What to do in each state

- **NORMAL.** Nothing. Keep the weekly cycle.
- **REDUCE_ONLY (set by monitoring).**
  - Proposals continue, but orders that add exposure are removed: selling
    longs, covering shorts, stops and exits still happen; new risk does not.
  - Read the reasons on the dashboard. Usually it is a drawdown past P80, which
    is common.
  - No action is needed: monitoring lifts it when the evidence clears. If you
    want to override, `ql live clear --to normal --reason "..."`.
- **REDUCE_ONLY (set by you, `ql live pause`).** Monitoring never lifts it. You
  paused for a reason monitoring cannot see, such as a holiday or earnings.
  Lift it yourself with `ql live clear`.
- **HALTED.** No rotations. Follow 11.2.

---

## 11. Incidents: what to do when something is wrong

### 11.1 Reconciliation mismatch

*Symptom:* `ql live sync` prints `MISMATCH` and the system is **HALTED**.

*Meaning:* the account holds **less** of something than the sleeve believes,
holds it on the other side, or an order the system sent has disappeared. In a
dedicated account, it can also mean the account holds something the sleeve does
not. Typical causes:

- you sold or bought shares manually in TWS or the app;
- a corporate action (a merger, a spin-off, a symbol change);
- IBKR cancelled an order the system still thinks is working.

*Steps:*

1. `ql live sync` again. A fill that arrived late may resolve it.
2. `ql live journal --tail 40` to see what the system last did, and compare it
   with *Trades* and *Portfolio* in TWS or Client Portal.
3. Correct the sleeve to the truth. Quantities are **absolute**, and a reason
   is required:
   ```bash
   ql live adjust --instrument AAPL --quantity 0 --cash-delta 18250.40 \
       --reason "Sold 100 AAPL manually on 3 Oct at 182.50; proceeds stay in the sleeve"
   ```
   `--cash-delta` adds (+) or removes (−) cash from the sleeve. Monitoring
   removes it from that week's return, so a correction does not count as
   performance.
4. `ql live reconcile`. It must now print `OK` or `WARN`.
5. `ql live clear --reason "Manual AAPL sale recorded; reconciliation OK"`.

**Why the system halts instead of fixing itself.** Guessing which side is right
is exactly the decision that needs a person.

### 11.2 The system is halted

*Meaning:* either monitoring found the strategy outside its bootstrapped range
(the dashboard lists the reason), or reconciliation failed (11.1), or you halted
it (`ql live halt --reason ...`).

While halted, nothing rotates. Existing positions keep their stops at IBKR, so
the downside of each position stays bounded.

*Steps:*

1. Read the dashboard: `ql monitor run --dry-run`, then open the page.
2. Decide among three options:
   - **Resume.** The cause is understood and does not invalidate the strategy
     (a data error, a one-off event):
     `ql live clear --reason "..."`. Clearing re-checks reconciliation first,
     and refuses if there is still a mismatch.
   - **Resume cautiously.** `ql live clear --to reduce_only --reason "..."`.
     Existing positions run and nothing new is bought.
   - **Exit.** `ql live propose --liquidate`, then `ql live approve` as usual.
     It proposes selling every sleeve position at the next open. Even an exit
     needs your typed approval.

**Why a person must lift a halt.** A halt means the evidence says the system is
no longer the one that was validated. Deciding it is fine again is a judgement,
and the written reason is the record of that judgement.

### 11.3 Data is stale

*Symptom:* `propose` refuses with "the latest complete week closed N hours ago",
or the dashboard shows a data-age warning.

*Steps:* `ql data refresh` (gateway needed), then continue the cycle. If the
refresh fails for some symbols, see the error column. A symbol that fails
repeatedly may have been delisted or renamed.

### 11.4 An order was not filled, or was rejected

*Symptom:* after Monday's sync, an order shows cancelled or rejected; or
`propose` says orders are still working.

*Steps:*

1. `ql live sync` records the final status. Rejections carry IBKR's message
   (for example no trading permission, or insufficient funds).
2. Once no orders are working, `ql live propose` computes a fresh proposal from
   the current sleeve. It contains whatever is still missing.
3. MOO orders placed after the auction cutoff are rejected by IBKR. The next
   approval window is the next morning's open.

### 11.5 A stop fired

This is normal, and nothing is required. The next `ql live sync` records the
fill. The name is not bought back until the next rotation.

### 11.6 You traded in the account yourself

- **In a dedicated account (the default)**, anything you bought that the sleeve
  does not hold, or extra shares of a name it does, is a mismatch and halts the
  system. The account is the strategy's, so a stray position is an incident.
  Undo the trade, or move the position to another account, then
  `ql live reconcile` and `ql live clear`.
- **In a shared account**, reconciliation only warns that it is "held at the
  broker but not by the sleeve". To silence it, list the ticker under
  `unmanaged:` in the config. The strategy will never touch it.
- **You changed a sleeve position.** Treat it as 11.1.

Orders you place by hand have no `ql-` reference, so the system never mistakes
them for its own.

### 11.7 Deposits and withdrawals

The sleeve does not see the account's cash movements. To give the strategy more
or less capital:

```bash
ql live adjust --cash-delta 10000 --reason "Added 10k to the strategy on 5 Oct"
```

Also update `sleeve_capital` in the config for the record.

### 11.8 The gateway disconnects, or you forget a week

Nothing breaks: stops rest at IBKR, and nothing is sent without you. When you
are back:

1. `ql data refresh`
2. `ql live sync`
3. `ql monitor run`
4. `ql live propose`

A proposal that expired stays in the journal as expired, which monitoring
counts.

### 11.9 A command refuses

Every refusal starts with `error:` and says what to do next. Refusals are the
system working. Nothing is half-done when a command refuses: the check runs
before anything is recorded or sent.

---

## 12. From paper to live money

**Minimum before switching:**

1. At least one full rotation cycle, four weeks, on paper, including one
   rotation with sells and buys. Every step of section 9 must have been done by
   you.
2. Every paper sync reconciled `OK` or `WARN`, never `MISMATCH`.
3. Stops were placed after each rotation, and `ql live status` showed a stop on
   every position.
4. `ql monitor run` produced a dashboard you could read without this manual
   open.

**The switch:**

1. In IB Gateway, log in to the **live** account. The API settings are the same
   (step 7.1), but the port is **4001**.
2. In the strategy's config set `mode: live`, `account: U…` (your live account)
   and `gateway.port: 4001`. Set `sleeve_capital` to the amount you intend to
   commit. Start smaller than the final amount.
3. `ql live status`. It must say `connected U… (live)`.
4. `ql monitor baseline`. Baselines are per strategy and mode, so this builds
   the live one.
5. `ql live init`, or `ql live init --adopt …`.
6. From now on, `approve` asks you to type `LIVE <id>`.
7. Automation, if you want it, starts again at `exits` (step 9.9): arming
   `full` with live money needs the exits-stage evidence, which the paper
   journal of the same strategy can supply.

The paper journal stays in `state/live/<id>/paper-journal.jsonl`. To keep paper
running alongside live, keep the paper config as a second file with the same id,
for example `configs/strategies/momentum-paper.yaml`, and select it with
`ql --strategy momentum-paper …` (step 7.5).

---

## 13. Files, state and backups

| path | what | if lost |
|---|---|---|
| `data/ibkr_cache/` | raw CSVs from IBKR, not in git | Re-fetch with `ql data fetch` (step 2.2); the store's recorded revisions go with `var/store`. Back it up. |
| `data/ibkr_cache/factors/` | dividend factors, one file per symbol, not in git | Re-fetch with `ql data fetch`; without them the store is price-only. Back it up with the cache. |
| `var/store/` | derived store | Rebuild: `ql data ingest --rebuild`. |
| `state/live/<id>/<mode>-journal.jsonl` | **every live event of one strategy; its sleeve is replayed from it** | **Not recoverable.** The sleeve's history, costs and overrides are gone. |
| `state/live/<id>/<mode>-baseline.json` | the monitoring reference | Rebuild with `ql monitor baseline`. |
| `state/research.jsonl` | every research trial | **Not recoverable.** The trial count behind the DSR is gone. |
| `state/reports/<id>/` | report pages and their data | Re-render from the JSON; re-run to regenerate. |
| `configs/strategies/*.yaml` | your configs, one per strategy and mode | Re-create from the example, with the **same** `strategy_id`. |

**Back up `state/`.** It is gitignored on purpose, because it names your account
and positions. The repository lives in iCloud Drive, which syncs it, but sync is
not a backup. Copy `state/` somewhere else after each weekly cycle.

**The journal refuses to be edited.** Each line carries a sequence number. A
truncated, reordered or hand-edited journal fails its integrity check, and the
system stops rather than trading on a history it cannot trust. Corrections go
through `ql live adjust`, which *appends* a correction with a reason. Never edit
the file.

**iCloud "Optimise Mac Storage".** If it is enabled, macOS may offload files it
thinks are unused. Right-click the `quant-lab` folder and choose **Keep
Downloaded**.

---

## 14. Things that will bite

- **Changing a strategy parameter changes the strategy's identity.** The
  version includes a hash of the parameters, so the ledger treats a 13-week and
  a 26-week lookback as two strategies. The monitoring baseline refuses
  settings it was not built for. Evidence earned by one does not transfer to
  the other.
- **The funnel takes minutes.** About 75 backtests. It resumes, so interrupting
  it is safe.
- **Market-on-open has a cutoff.** 09:28 New York time. An approval after that
  is rejected by IBKR and has to wait for the next open (11.4).
- **Two API programs with the same client id** disconnect each other. Keep
  `gateway.client_id` unique.
- **IB Gateway restarts daily** at the IBKR server reset, and needs a fresh
  login at least once a week. Check it is logged in before the weekly cycle.
- **A strategy id is for life.** Renaming it on a running sleeve orphans the
  journal and the orders at IBKR. For a new id, open a new sleeve.
- **Don't edit an order's reference in TWS.** The system recognises its orders
  by it; an edited one becomes a stranger's order.
- **A short strategy's backtest is optimistic.** No borrow fees, no margin
  interest, no recalls.
- **Automation needs the Mac and the gateway.** A shut-down Mac misses its
  scheduled runs, and IB Gateway needs a full login every Monday. An armed
  system that cannot connect sends nothing; check `state/live/<id>/cycle.log`.
- **Leverage costs every day, and drawdowns grow with it.** It does not
  improve the Sharpe ratio (step 7.7).
- **The system sends orders on its own only when you have allowed it in the
  config *and* armed it**, and never after a halt until you clear the halt and
  arm it again.

---

## 15. Changing the code safely

```bash
pytest -q && ruff check .
```

Both run in CI on every push, along with:

- the ingest;
- the causality demo;
- the convention reconciliation;
- `ql data status` and a `ql backtest --report`;
- a reduced funnel.

The layering is enforced by `tests/test_architecture.py`. For example,
`reports/` may import only `contracts` and `validation`, and `strategies/` only
`contracts`. A new package must be added there deliberately.

The live code is tested against a stand-in gateway (`tests/fake_gateway.py`)
that uses the real `ib_async` types, and the whole weekly cycle is driven
through the `ql` command in `tests/test_cli.py`.

The house rule: **a guard that has never failed and a guard that cannot fail
look identical from the outside.** If you add a check, add a test that makes it
fail.

---

## 16. Command reference

🔌 = needs IB Gateway running and logged in.

| command | what it does |
|---|---|
| `ql strategies` | every configured strategy: bar size, mode, account, capital, sleeve |
| `ql data status [-v] [--interval I]` | cache and store contents, data age |
| `ql data refresh [--symbols A,B] [--duration "2 Y"] [--interval I]` 🔌 | append the latest complete bars |
| `ql data fetch --symbols A,B --freq weekly` 🔌 | add instruments to the universe |
| `ql data import A=a.json … \| --from-dir DIR` | add instruments from saved payloads |
| `ql data ingest --rebuild` | rebuild the store from the cache |
| `ql data backfill --interval hour [--years 5] [--symbols A,B]` 🔌 | page back through intraday history into the cache |
| `ql backtest [options] [--universe U] [--capital C] [--leverage X] [--margin-rate R] [--report]` | backtest; recorded in the ledger |
| `ql funnel [--universe U] [--top N] [--rebalance-weeks K] [--capital C] [--controls N]` | the five research gates |
| `ql funnel --grid-top 1,2,3 --grid-rebalance 1,2,4 [--universe U]` | choose N and K: the plateau, not the peak |
| `ql live init [--adopt A,B]` 🔌 | open the sleeve (once per strategy and mode) |
| `ql live sync` 🔌 | fills, statuses, stops, snapshot, reconciliation |
| `ql live status [--offline]` | state, equity, positions, stops |
| `ql live propose [--liquidate]` 🔌 | compute orders; sends nothing |
| `ql live approve [ID] [--confirm CODE]` 🔌 | send the pending proposal after the typed code |
| `ql live reject ID --reason "…"` | decline; recorded as an override |
| `ql live stops` 🔌 | place any missing protective stops |
| `ql live reconcile` 🔌 | compare sleeve and account, and record it |
| `ql live adjust [--instrument T --quantity Q] [--average-cost C] [--cash-delta X] --reason "…"` | correct the sleeve |
| `ql live pause --reason "…"` | reduce-only until you clear it |
| `ql live halt --reason "…"` | no rotations until you clear it |
| `ql live clear [--to normal\|reduce_only] --reason "…"` 🔌 | lift a pause or halt (reconciles first) |
| `ql live journal [--tail N] [--kind K] [--full]` | read the event journal |
| `ql live cycle [--no-refresh] [--no-monitor]` 🔌 | refresh, sync, monitor, propose; sends only if armed |
| `ql live auto status` | what automation may do, what is armed, the evidence |
| `ql live auto arm --scope exits\|full [--override "…"]` | allow automatic sending (typed phrase) |
| `ql live auto disarm --reason "…"` | back to manual approval |
| `ql live auto schedule` | write the Mac launch job for `ql live cycle` |
| `ql review run [--state F] [--state-out F] [--out F] [--inputs DIR] [--as-of T]` | the week's decision from the broker connector's answers; sends nothing |
| `ql review clear --state F --to normal\|reduce_only --reason "…"` | lift the review's pause or halt |
| `ql review baseline [--source F]` | publish the monitoring baseline where the review reads it |
| `ql monitor baseline` | build the monitoring reference |
| `ql monitor run [--dry-run] [--no-report] [--benchmark SPY]` | every check, the state, the dashboard |
| `ql report list` | recent report pages |
| `ql report render FILE.json [--out FILE.html]` | re-render a saved report |

Global options, before the command: `--strategy NAME` (acts on
`configs/strategies/<NAME>.yaml`; needed when more than one is configured),
`--config PATH` (any config file) and `--store PATH` (default `var/store`).
`--interval` takes `weekly`, `daily`, `hourly` or `minute`. Every command has
`--help`.
