"""A report rendered as one self-contained, offline HTML page.

No network, no libraries, no build step: the page is a file you can open from a
USB stick in five years. Charts are inline SVG drawn here; a few lines of inline
script add the hover readout, and every value the hover shows is also in the
table under each chart, so nothing depends on the script running.

Visual rules (from the data-visualisation method this project follows):

* one y axis per chart; two measures of different scale get two charts;
* categorical colours in a fixed order (strategy, benchmark, third), validated
  for colour-vision deficiency on both surfaces; identity never by colour alone —
  a legend plus a direct label at the end of each line;
* the four status colours are reserved for system state and always carry an
  icon and a word;
* light and dark themes are separate, chosen palettes, not an inversion.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Sequence

from reports.document import Chart, Metric, Report, Table

W, H = 720, 260
LEFT, RIGHT, TOP, BOTTOM = 64, 118, 14, 30

_ICON = {"good": "✓", "warning": "!", "serious": "▲", "critical": "✕", "neutral": "•"}


def fmt(value, format: str) -> str:
    """One number, as a person reads it. Mirrored by ``fmt`` in the page script."""
    if value is None:
        return "—"
    if format == "text" or isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    v = float(value)
    if not math.isfinite(v):
        return "—"
    if v == 0:
        v = 0.0  # never print "-0.0%"
    if format == "pct":
        return f"{v:.1%}"
    if format == "pct_signed":
        return f"{v:+.1%}"
    if format == "money":
        return f"{v:,.0f}"
    if format == "bps":
        return f"{v:.1f} bp"
    if format == "int":
        return f"{v:,.0f}"
    if format == "hours":
        return f"{v:.1f} h"
    if format == "days":
        return f"{v:.1f} d"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.2f}"


def render(report: Report) -> str:
    report.validate()
    e = html.escape
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{e(report.title)}</title><style>{_CSS}</style></head><body>",
        "<main>",
        f"<header><h1>{e(report.title)}</h1>",
        f"<p class='sub'>{e(report.subtitle)}</p>" if report.subtitle else "",
        f"<p class='muted'>Generated {e(report.generated_at)} · schema quant-lab.report/1</p>",
        "<button class='theme' type='button' aria-label='Toggle light or dark theme'>◐</button>",
        "</header>",
    ]
    if report.status is not None:
        s = report.status
        reasons = "".join(f"<li>{e(r)}</li>" for r in s.reasons)
        parts.append(
            f"<section class='status lvl-{s.level}' aria-label='System state'>"
            f"<span class='icon' aria-hidden='true'>{_ICON[s.level]}</span>"
            f"<div><strong>{e(s.label)}</strong>"
            + (f"<ul>{reasons}</ul>" if reasons else "")
            + "</div></section>"
        )
    if report.metrics:
        parts.append("<section class='tiles'>")
        parts.extend(_tile(m) for m in report.metrics)
        parts.append("</section>")
    for chart in report.charts:
        parts.append(_chart(chart))
    for table in report.tables:
        parts.append(_table(table))
    if report.notes:
        parts.append("<section class='notes'><h2>Notes</h2><ul>")
        parts.extend(f"<li>{e(n)}</li>" for n in report.notes)
        parts.append("</ul></section>")
    parts.append(f"</main><div class='tip' role='status' hidden></div><script>{_JS}</script>")
    parts.append("</body></html>")
    return "".join(parts)


def write(report: Report, path) -> None:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(report), encoding="utf-8")


# -- pieces --------------------------------------------------------------------


def _tile(m: Metric) -> str:
    e = html.escape
    icon = (
        f"<span class='icon lvl-{m.level}' aria-hidden='true'>{_ICON[m.level]}</span>"
        if m.level != "neutral" else ""
    )
    word = f"<span class='lvlword'>{m.level}</span>" if m.level != "neutral" else ""
    return (
        f"<div class='tile'><div class='label'>{e(m.label)}</div>"
        f"<div class='value'>{icon}{e(fmt(m.value, m.format))}</div>"
        f"{word}<div class='note'>{e(m.note)}</div></div>"
    )


def _ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if hi <= lo:
        # A flat series still needs a readable axis around it, not 0 to 100%.
        pad = abs(lo) * 0.05 or 0.01
        lo, hi = lo - pad, hi + pad
    raw = (hi - lo) / count
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    ticks = []
    t = start
    while t <= hi + step * 1e-9:
        ticks.append(round(t, 12))
        t += step
    if ticks[-1] < hi:
        ticks.append(ticks[-1] + step)
    return ticks


def _chart(c: Chart) -> str:
    e = html.escape
    n = len(c.x)
    values = [v for s in c.series for v in s.values if v is not None]
    extra = [b for _, b in c.bands] + ([c.baseline] if c.baseline is not None else [])
    if not values or n == 0:
        return (f"<section class='chart'><h2>{e(c.title)}</h2>"
                "<p class='muted'>No data yet.</p></section>")
    ticks = _ticks(min(values + extra), max(values + extra))
    lo, hi = ticks[0], ticks[-1]
    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM

    def sx(i: int) -> float:
        return LEFT + (pw * i / (n - 1) if n > 1 else pw / 2)

    def sy(v: float) -> float:
        return TOP + ph * (1 - (v - lo) / (hi - lo))

    svg = [f"<svg viewBox='0 0 {W} {H}' role='img' tabindex='0' "
           f"aria-label='{e(c.title)}. Use arrow keys to read values.'>"]
    for t in ticks:
        y = sy(t)
        svg.append(f"<line class='grid' x1='{LEFT}' x2='{W - RIGHT}' y1='{y:.1f}' y2='{y:.1f}'/>")
        svg.append(f"<text class='axis' x='{LEFT - 8}' y='{y + 4:.1f}' text-anchor='end'>"
                   f"{e(_tick(t, c.format, ticks[1] - ticks[0]))}</text>")
    xt = _x_ticks(n)
    short = [_short(c.x[i]) for i in xt]
    if len(set(short)) < len(short):  # a short span: months repeat, so show days
        short = [c.x[i] for i in xt]
    for i, text in zip(xt, short, strict=True):
        svg.append(f"<text class='axis' x='{sx(i):.1f}' y='{H - 8}' text-anchor='middle'>"
                   f"{e(text)}</text>")
    if c.baseline is not None and lo <= c.baseline <= hi:
        y = sy(c.baseline)
        svg.append(f"<line class='base' x1='{LEFT}' x2='{W - RIGHT}' y1='{y:.1f}' y2='{y:.1f}'/>")
    ends: list[tuple[float, str, int]] = []
    for label, value in c.bands:
        y = sy(value)
        svg.append(f"<line class='band' x1='{LEFT}' x2='{W - RIGHT}' y1='{y:.1f}' y2='{y:.1f}'/>")
        ends.append((y, label, -1))
    for k, s in enumerate(c.series):
        paths, current = [], []
        for i, v in enumerate(s.values):
            if v is None:
                if current:
                    paths.append(current)
                current = []
            else:
                current.append((sx(i), sy(v)))
        if current:
            paths.append(current)
        for pts in paths:
            d = "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            if c.area and k == 0 and c.baseline is not None:
                base = sy(c.baseline)
                svg.append(f"<path class='area s{k}' d='{d}L{pts[-1][0]:.1f},{base:.1f}"
                           f"L{pts[0][0]:.1f},{base:.1f}Z'/>")
            svg.append(f"<path class='line s{k}' d='{d}'/>")
        last = next(((i, v) for i, v in reversed(list(enumerate(s.values))) if v is not None), None)
        if last is not None:
            ends.append((sy(last[1]), s.name, k))
    # Direct labels at the line ends (and band names), nudged apart so none overlap.
    ends.sort()
    placed: list[float] = []
    for y, name, k in ends:
        y = max(y, placed[-1] + 14) if placed else y
        placed.append(y)
        if k < 0:
            svg.append(f"<text class='axis' x='{W - RIGHT + 6}' y='{y + 4:.1f}'>{e(name)}</text>")
        else:
            svg.append(f"<text class='endlabel' x='{W - RIGHT + 6}' y='{y + 4:.1f}'>"
                       f"<tspan class='sw s{k}'>■</tspan> {e(name)}</text>")
    svg.append(f"<line class='cross' x1='0' x2='0' y1='{TOP}' y2='{H - BOTTOM}' visibility='hidden'/>")
    for k in range(len(c.series)):
        svg.append(f"<circle class='dot s{k}' r='4.5' visibility='hidden'/>")
    svg.append(f"<rect class='hit' x='{LEFT}' y='{TOP}' width='{pw}' height='{ph}'/>")
    svg.append("</svg>")

    payload = {
        "x": list(c.x), "format": c.format,
        "geom": {"left": LEFT, "width": pw, "top": TOP, "height": ph, "lo": lo, "hi": hi},
        "series": [{"name": s.name, "values": list(s.values)} for s in c.series],
    }
    legend = ""
    if len(c.series) >= 2:
        legend = "<div class='legend'>" + "".join(
            f"<span><i class='sw s{k}'>■</i>{e(s.name)}</span>" for k, s in enumerate(c.series)
        ) + "</div>"
    table = Table(
        title="", columns=("x",) + tuple(s.name for s in c.series),
        rows=tuple((c.x[i],) + tuple(s.values[i] for s in c.series) for i in range(n)),
        formats=("text",) + (c.format,) * len(c.series),
    )
    return (
        f"<section class='chart' id='{e(c.id)}'><h2>{e(c.title)}</h2>{legend}"
        f"<figure data-chart>{''.join(svg)}"
        f"<script type='application/json'>{_json(payload)}</script></figure>"
        + (f"<p class='note'>{e(c.note)}</p>" if c.note else "")
        + f"<details><summary>Table view</summary>{_table_html(table)}</details></section>"
    )


def _table(t: Table) -> str:
    e = html.escape
    return (
        f"<section class='table'><h2>{e(t.title)}</h2>{_table_html(t)}"
        + (f"<p class='note'>{e(t.note)}</p>" if t.note else "")
        + "</section>"
    )


def _table_html(t: Table) -> str:
    e = html.escape
    formats = t.formats or ("text",) * len(t.columns)
    head = "".join(f"<th>{e(str(c))}</th>" for c in t.columns)
    if not t.rows:
        return f"<table><thead><tr>{head}</tr></thead><tbody><tr><td colspan='{len(t.columns)}' " \
               "class='muted'>Nothing to show.</td></tr></tbody></table>"
    body = "".join(
        "<tr>" + "".join(
            f"<td class='{'num' if f != 'text' else ''}'>{e(fmt(v, f))}</td>"
            for v, f in zip(row, formats, strict=True)
        ) + "</tr>"
        for row in t.rows
    )
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _tick(value: float, format: str, step: float) -> str:
    """An axis label with as many decimals as the step needs, and no more."""
    if format in ("pct", "pct_signed"):
        decimals = max(0, math.ceil(-math.log10(step * 100) + 1e-9)) if step > 0 else 1
        value = 0.0 if abs(value) < step * 1e-6 else value
        return f"{value * 100:.{decimals}f}%"
    if format == "bps":
        return f"{value:.0f} bp" if step >= 1 else f"{value:.1f} bp"
    return fmt(value, format)


def _x_ticks(n: int, count: int = 6) -> Sequence[int]:
    if n <= 1:
        return [0]
    step = max(1, math.ceil((n - 1) / (count - 1)))
    ticks = list(range(0, n, step))
    if n - 1 - ticks[-1] < step / 2:
        ticks[-1] = n - 1
    else:
        ticks.append(n - 1)
    return ticks


def _short(label: str) -> str:
    """ISO dates shortened to what an axis needs."""
    return label[:7] if len(label) >= 10 and label[4] == "-" else label


def _json(payload) -> str:
    # ``</`` inside a script block would end it early.
    return json.dumps(payload, allow_nan=False, default=str).replace("</", "<\\/")


_CSS = """
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--base:#c3c2b7;--s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a;--good:#0ca30c;--warning:#fab219;
--serious:#ec835a;--critical:#d03b3b;--panel:#f3f2ee;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;
--ink:#ffffff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--base:#383835;--s0:#3987e5;
--s1:#d95926;--s2:#199e70;--panel:#232322;color-scheme:dark}}
:root[data-theme="dark"]{--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--base:#383835;--s0:#3987e5;--s1:#d95926;--s2:#199e70;--panel:#232322;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
header{position:relative}h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:0 0 8px}
.sub{color:var(--ink2);margin:0 0 2px}.muted,.note{color:var(--muted);font-size:12px}
.theme{position:absolute;right:0;top:0;background:none;border:1px solid var(--grid);
color:var(--ink2);border-radius:6px;padding:2px 8px;cursor:pointer}
section{margin-top:24px}
.status{display:flex;gap:12px;align-items:flex-start;padding:12px 14px;border-radius:8px;
background:var(--panel);border-left:4px solid var(--muted)}
.status ul{margin:4px 0 0;padding-left:18px;color:var(--ink2)}
.status .icon{font-weight:700;font-size:18px;line-height:1.2}
.lvl-good{border-color:var(--good)}.lvl-warning{border-color:var(--warning)}
.lvl-serious{border-color:var(--serious)}.lvl-critical{border-color:var(--critical)}
.icon.lvl-good,.lvl-good>.icon{color:var(--good)}.icon.lvl-warning,.lvl-warning>.icon{color:var(--warning)}
.icon.lvl-serious,.lvl-serious>.icon{color:var(--serious)}.icon.lvl-critical,.lvl-critical>.icon{color:var(--critical)}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.tile{background:var(--panel);border-radius:8px;padding:10px 12px}
.tile .label{color:var(--ink2);font-size:12px}.tile .value{font-size:20px;font-weight:600;
font-variant-numeric:tabular-nums}.tile .icon{margin-right:6px;font-size:16px}
.lvlword{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink2)}
figure{margin:0;position:relative}svg{width:100%;height:auto;display:block;overflow:visible}
svg:focus{outline:2px solid var(--s0);outline-offset:2px}
.grid{stroke:var(--grid);stroke-width:1}.base{stroke:var(--base);stroke-width:1.5}
.band{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 4}
.axis{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.endlabel{fill:var(--ink2);font-size:12px}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.line.s0{stroke:var(--s0)}.line.s1{stroke:var(--s1)}.line.s2{stroke:var(--s2)}
.area.s0{fill:var(--s0);opacity:.14}
.sw.s0{fill:var(--s0);color:var(--s0)}.sw.s1{fill:var(--s1);color:var(--s1)}.sw.s2{fill:var(--s2);color:var(--s2)}
.dot{stroke:var(--surface);stroke-width:2}.dot.s0{fill:var(--s0)}.dot.s1{fill:var(--s1)}.dot.s2{fill:var(--s2)}
.cross{stroke:var(--muted);stroke-width:1}.hit{fill:transparent;cursor:crosshair}
.legend{display:flex;gap:16px;color:var(--ink2);font-size:12px;margin-bottom:4px}
.legend i{font-style:normal;margin-right:4px}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);
border:1px solid var(--grid);border-radius:6px;padding:6px 9px;font-size:12px;
box-shadow:0 2px 8px rgba(0,0,0,.12);z-index:10;font-variant-numeric:tabular-nums}
.tip .row{display:flex;gap:8px;align-items:baseline}.tip strong{min-width:64px}
.tip .when{color:var(--muted);margin-bottom:2px}
details{margin-top:6px}summary{color:var(--ink2);font-size:12px;cursor:pointer}
.scroll{max-height:340px;overflow:auto}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{text-align:left;color:var(--ink2);font-weight:600;border-bottom:1px solid var(--base);
padding:4px 8px;position:sticky;top:0;background:var(--surface)}
td{padding:3px 8px;border-bottom:1px solid var(--grid)}td.num{text-align:right;
font-variant-numeric:tabular-nums}
.notes li{color:var(--ink2)}
"""

_JS = r"""
(function(){
function fmt(v,f){if(v===null||v===undefined||!isFinite(v))return'—';
var p=function(x,s){return(s&&x>=0?'+':'')+(x*100).toFixed(1)+'%'};
if(f==='pct')return p(v,false);if(f==='pct_signed')return p(v,true);
if(f==='money'||f==='int')return Math.round(v).toLocaleString('en-US');
if(f==='bps')return v.toFixed(1)+' bp';if(f==='hours')return v.toFixed(1)+' h';
if(f==='days')return v.toFixed(1)+' d';
return Math.abs(v)>=1000?Math.round(v).toLocaleString('en-US'):v.toFixed(2)}
var tip=document.querySelector('.tip');
function show(fig,d,i,cx,cy){var g=d.geom,n=d.x.length;
var x=g.left+(n>1?g.width*i/(n-1):g.width/2);
var cross=fig.querySelector('.cross');cross.setAttribute('x1',x);cross.setAttribute('x2',x);
cross.setAttribute('visibility','visible');
var dots=fig.querySelectorAll('.dot');tip.textContent='';
var w=document.createElement('div');w.className='when';w.textContent=d.x[i];tip.appendChild(w);
d.series.forEach(function(s,k){var v=s.values[i];var dot=dots[k];
if(v===null){dot.setAttribute('visibility','hidden')}else{
dot.setAttribute('cx',x);dot.setAttribute('cy',g.top+g.height*(1-(v-g.lo)/(g.hi-g.lo)));
dot.setAttribute('visibility','visible')}
var r=document.createElement('div');r.className='row';
var b=document.createElement('strong');b.textContent=fmt(v,d.format);
var sw=document.createElement('span');sw.className='sw s'+k;sw.textContent='■ ';
var l=document.createElement('span');l.appendChild(sw);l.appendChild(document.createTextNode(s.name));
r.appendChild(b);r.appendChild(l);tip.appendChild(r)});
tip.hidden=false;var tw=tip.offsetWidth;
tip.style.left=Math.min(cx+14,window.innerWidth-tw-8)+'px';tip.style.top=(cy+14)+'px'}
function hide(fig){tip.hidden=true;fig.querySelector('.cross').setAttribute('visibility','hidden');
fig.querySelectorAll('.dot').forEach(function(d){d.setAttribute('visibility','hidden')})}
document.querySelectorAll('figure[data-chart]').forEach(function(fig){
var d=JSON.parse(fig.querySelector('script').textContent);var svg=fig.querySelector('svg');
var n=d.x.length,cur=n-1;
function idx(ev){var pt=svg.createSVGPoint();pt.x=ev.clientX;pt.y=ev.clientY;
var p=pt.matrixTransform(svg.getScreenCTM().inverse());
var t=(p.x-d.geom.left)/d.geom.width;return Math.max(0,Math.min(n-1,Math.round(t*(n-1))))}
svg.addEventListener('pointermove',function(ev){cur=idx(ev);show(fig,d,cur,ev.clientX,ev.clientY)});
svg.addEventListener('pointerleave',function(){hide(fig)});
svg.addEventListener('blur',function(){hide(fig)});
svg.addEventListener('keydown',function(ev){if(ev.key!=='ArrowLeft'&&ev.key!=='ArrowRight')return;
ev.preventDefault();cur=Math.max(0,Math.min(n-1,cur+(ev.key==='ArrowRight'?1:-1)));
var r=svg.getBoundingClientRect();show(fig,d,cur,r.left+r.width*((d.geom.left+
(n>1?d.geom.width*cur/(n-1):0))/720),r.top+20)})});
var btn=document.querySelector('.theme');if(btn)btn.addEventListener('click',function(){
var root=document.documentElement;var dark=root.getAttribute('data-theme')==='dark'||
(!root.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme: dark)').matches);
root.setAttribute('data-theme',dark?'light':'dark')});
})();
"""
