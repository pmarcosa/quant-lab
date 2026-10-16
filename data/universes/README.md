# Universes

One file per universe: `<name>.txt`, one ticker per line, `#` starts a comment.
Use one with `ql backtest --universe <name>`, `ql funnel --universe <name>`, or
in a strategy's config (`strategy.universe: <name>`). Without one, a strategy
chooses from every instrument in the store.

A universe is part of the strategy's definition. A live sleeve records the one
it was opened on and refuses to run on another. The symbols must be fetched
(`ql data fetch --symbols …`); symbols without data are left out and listed.

A strategy that reads a sector per stock (`sector-momentum`) takes a map beside
the universe: `<name>.sectors.csv`, one `SYMBOL,SECTOR` pair per line, where
the sector is the instrument the strategy reads for it (a Select Sector SPDR).
`momentum-wide.sectors.csv` maps the 150 stocks of `momentum-wide`; the
universe `momentum-wide-gated` is those stocks plus the eleven funds, which the
strategy reads and never trades.
