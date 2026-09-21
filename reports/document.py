"""The report as data: a versioned, plain JSON document.

A report is written twice — once as data, once as a page — and the page is only
ever rendered *from* the data. That keeps two promises. Anything a chart shows
can be re-read, diffed and archived as JSON without scraping HTML, and the
renderer cannot compute anything, because everything it receives is already a
number with a label.

This layer knows nothing about backtests or brokers. ``runtime/reporting.py``
turns a run or a monitoring pass into a :class:`Report`; this module only says
what a report may contain and checks that it does.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contracts.errors import ContractViolation

#: Bumped whenever a field changes meaning. Readers refuse other versions.
SCHEMA = "quant-lab.report/1"

#: How a number is shown. The renderer formats; the document never pre-formats.
FORMATS = ("pct", "pct_signed", "num", "int", "money", "bps", "hours", "days", "text")

#: The four reserved status levels. Never used for anything but state.
LEVELS = ("good", "warning", "serious", "critical", "neutral")


@dataclass(frozen=True, slots=True)
class Metric:
    """One headline number."""

    label: str
    value: float | str | None
    format: str = "num"
    note: str = ""
    level: str = "neutral"


@dataclass(frozen=True, slots=True)
class Series:
    name: str
    values: tuple[float | None, ...]


@dataclass(frozen=True, slots=True)
class Chart:
    """A line chart over a shared x axis. One y axis, always.

    Attributes:
        x: Labels of the x axis, usually ISO dates.
        series: At most three; each the same length as ``x``. ``None`` is a gap.
        format: How y values are shown.
        baseline: A reference level drawn as a rule (0 for returns, 1 for an index).
        bands: Optional horizontal reference lines, ``(label, value)``.
        area: Fill under the first series down to the baseline (drawdowns).
    """

    id: str
    title: str
    x: tuple[str, ...]
    series: tuple[Series, ...]
    format: str = "num"
    baseline: float | None = None
    bands: tuple[tuple[str, float], ...] = ()
    area: bool = False
    note: str = ""


@dataclass(frozen=True, slots=True)
class Table:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    formats: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True, slots=True)
class Status:
    """The one thing to read first: what state the system is in, and why."""

    level: str
    label: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Report:
    kind: str
    title: str
    generated_at: str
    subtitle: str = ""
    status: Status | None = None
    metrics: tuple[Metric, ...] = ()
    charts: tuple[Chart, ...] = ()
    tables: tuple[Table, ...] = ()
    notes: tuple[str, ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> Report:
        """Refuse a document the renderer would have to guess about."""
        for metric in self.metrics:
            _check_format(metric.format, metric.label)
            _check_level(metric.level, metric.label)
        if self.status is not None:
            _check_level(self.status.level, "status")
        seen: set[str] = set()
        for chart in self.charts:
            _check_format(chart.format, chart.id)
            if chart.id in seen:
                raise ContractViolation(f"duplicate chart id {chart.id!r}")
            seen.add(chart.id)
            if not chart.series:
                raise ContractViolation(f"chart {chart.id!r} has no series")
            if len(chart.series) > 3:
                raise ContractViolation(
                    f"chart {chart.id!r} has {len(chart.series)} series; three is the limit"
                )
            for s in chart.series:
                if len(s.values) != len(chart.x):
                    raise ContractViolation(
                        f"series {s.name!r} in {chart.id!r} has {len(s.values)} values "
                        f"for {len(chart.x)} x labels"
                    )
        for table in self.tables:
            if table.formats and len(table.formats) != len(table.columns):
                raise ContractViolation(f"table {table.title!r}: one format per column")
            for f in table.formats:
                _check_format(f, table.title)
            for row in table.rows:
                if len(row) != len(table.columns):
                    raise ContractViolation(f"table {table.title!r}: row width differs from columns")
        return self

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": self.kind,
            "title": self.title,
            "subtitle": self.subtitle,
            "generated_at": self.generated_at,
            "status": None if self.status is None else {
                "level": self.status.level, "label": self.status.label,
                "reasons": list(self.status.reasons),
            },
            "metrics": [
                {"label": m.label, "value": _clean(m.value), "format": m.format,
                 "note": m.note, "level": m.level}
                for m in self.metrics
            ],
            "charts": [
                {
                    "id": c.id, "title": c.title, "x": list(c.x), "format": c.format,
                    "baseline": c.baseline, "area": c.area, "note": c.note,
                    "bands": [[label, value] for label, value in c.bands],
                    "series": [{"name": s.name, "values": [_clean(v) for v in s.values]}
                               for s in c.series],
                }
                for c in self.charts
            ],
            "tables": [
                {"title": t.title, "columns": list(t.columns), "formats": list(t.formats),
                 "note": t.note, "rows": [[_clean(v) for v in row] for row in t.rows]}
                for t in self.tables
            ],
            "notes": list(self.notes),
            "context": {k: _clean(v) for k, v in self.context.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Report:
        if data.get("schema") != SCHEMA:
            raise ContractViolation(f"unknown report schema {data.get('schema')!r}; expected {SCHEMA}")
        status = data.get("status")
        return cls(
            kind=data["kind"], title=data["title"], subtitle=data.get("subtitle", ""),
            generated_at=data["generated_at"],
            status=None if status is None else Status(
                status["level"], status["label"], tuple(status.get("reasons", ()))
            ),
            metrics=tuple(Metric(m["label"], m["value"], m["format"], m.get("note", ""),
                                 m.get("level", "neutral")) for m in data.get("metrics", ())),
            charts=tuple(
                Chart(
                    id=c["id"], title=c["title"], x=tuple(c["x"]),
                    series=tuple(Series(s["name"], tuple(s["values"])) for s in c["series"]),
                    format=c["format"], baseline=c.get("baseline"),
                    bands=tuple((b[0], b[1]) for b in c.get("bands", ())),
                    area=c.get("area", False), note=c.get("note", ""),
                )
                for c in data.get("charts", ())
            ),
            tables=tuple(
                Table(t["title"], tuple(t["columns"]), tuple(tuple(r) for r in t["rows"]),
                      tuple(t.get("formats", ())), t.get("note", ""))
                for t in data.get("tables", ())
            ),
            notes=tuple(data.get("notes", ())),
            context=dict(data.get("context", {})),
        ).validate()

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.validate().to_dict(), indent=1), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> Report:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# -- series helpers: shaping, never judging ----------------------------------


def drawdown(values: Sequence[float | None]) -> tuple[float | None, ...]:
    """Distance below the running peak, as a (non-positive) fraction."""
    out: list[float | None] = []
    peak = -math.inf
    for v in values:
        if v is None:
            out.append(None)
            continue
        peak = max(peak, v)
        out.append(v / peak - 1.0 if peak > 0 else 0.0)
    return tuple(out)


def cumulative(returns: Sequence[float], start: float = 1.0) -> tuple[float, ...]:
    """Growth of ``start`` compounded through ``returns``; one more point than returns."""
    level = start
    out = [level]
    for r in returns:
        level *= 1.0 + r
        out.append(level)
    return tuple(out)


def rolling_mean(values: Sequence[float], window: int) -> tuple[float | None, ...]:
    out: list[float | None] = []
    for i in range(len(values)):
        if i + 1 < window:
            out.append(None)
        else:
            chunk = values[i + 1 - window:i + 1]
            out.append(sum(chunk) / window)
    return tuple(out)


def _check_format(fmt: str, where: str) -> None:
    if fmt not in FORMATS:
        raise ContractViolation(f"{where}: unknown format {fmt!r}; one of {FORMATS}")


def _check_level(level: str, where: str) -> None:
    if level not in LEVELS:
        raise ContractViolation(f"{where}: unknown level {level!r}; one of {LEVELS}")


def _clean(value: Any) -> Any:
    """JSON has no NaN or infinity; a missing number is ``null``."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _clean(value.item())
        except (TypeError, ValueError):
            return value
    return value
