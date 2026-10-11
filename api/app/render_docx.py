"""Write DPage models to an editable .docx that keeps the original look:

* right-to-left paragraphs (w:bidi) and runs (w:rtl) for Arabic, mixed text kept in
  logical order so Word / LibreOffice shape and order it correctly;
* paragraph alignment, indentation, spacing and exact line pitch from the source;
* ruled tables rebuilt as real Word tables (column widths, row heights, merged
  cells, borders, cell shading, right-to-left column order);
* every non-text graphic (bands, logos, photos, stamps) on a page-sized picture
  anchored behind the text, so the page looks like the original while all text
  stays editable.
"""
from __future__ import annotations

import copy
import io
from dataclasses import dataclass, field

import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu, Pt

from .docmodel import (DLine, DPage, DWord, line_is_rtl, logical_order, needs_space,
                       space_advance, word_metrics)
from .textutil import has_arabic, is_rtl_text

EMU_PER_PT = 12700
TW = 20  # twips per point


# =============================================================== tables
@dataclass
class Seg:
    a: float   # start (x for horizontal, y for vertical)
    b: float   # end
    pos: float  # y for horizontal, x for vertical
    width: float
    color: tuple[int, int, int]


@dataclass
class Table:
    xs: list[float]
    ys: list[float]
    hsegs: list[Seg]
    vsegs: list[Seg]
    fills: list = field(default_factory=list)
    cells: dict = field(default_factory=dict)   # (r, c) -> list[DWord]

    @property
    def bbox(self):
        return self.xs[0], self.ys[0], self.xs[-1], self.ys[-1]


def _merge_segs(segs: list[Seg], tol: float = 2.0) -> list[Seg]:
    segs = sorted(segs, key=lambda s: (s.pos, s.a))
    out: list[Seg] = []
    for s in segs:
        for o in out:
            if abs(o.pos - s.pos) <= tol and s.a <= o.b + 3 and s.b >= o.a - 3:
                o.a, o.b = min(o.a, s.a), max(o.b, s.b)
                o.width = max(o.width, s.width)
                break
        else:
            out.append(Seg(s.a, s.b, s.pos, s.width, s.color))
    return out


def _cluster(vals: list[float], tol: float = 2.5) -> list[float]:
    out: list[list[float]] = []
    for v in sorted(vals):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [float(np.mean(c)) for c in out]


def detect_tables(page: DPage) -> list[Table]:
    hs, vs, fills = [], [], []
    for r in page.rects:
        if r.h <= 3.5 and r.w >= 15:
            hs.append(Seg(r.x, r.x + r.w, r.y + r.h / 2, r.h, r.color))
        elif r.w <= 3.5 and r.h >= 8:
            vs.append(Seg(r.y, r.y + r.h, r.x + r.w / 2, r.w, r.color))
        elif r.w > 3.5 and r.h > 3.5:
            fills.append(r)
    hs, vs = _merge_segs(hs), _merge_segs(vs)
    # union-find on intersections
    n = len(hs) + len(vs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, h in enumerate(hs):
        for j, v in enumerate(vs):
            if h.a - 3 <= v.pos <= h.b + 3 and v.a - 3 <= h.pos <= v.b + 3:
                parent[find(i)] = find(len(hs) + j)
    groups: dict[int, tuple[list[Seg], list[Seg]]] = {}
    for i, h in enumerate(hs):
        groups.setdefault(find(i), ([], []))[0].append(h)
    for j, v in enumerate(vs):
        groups.setdefault(find(len(hs) + j), ([], []))[1].append(v)
    tables = []
    for gh, gv in groups.values():
        if len(gh) < 2 or len(gv) < 2:
            continue
        ys = _cluster([h.pos for h in gh])
        xs = _cluster([v.pos for v in gv])
        if len(ys) < 2 or len(xs) < 2 or (len(ys) - 1) * (len(xs) - 1) < 2:
            continue
        t = Table(xs, ys, gh, gv)
        x0, y0, x1, y1 = t.bbox
        t.fills = [f for f in fills if f.x >= x0 - 3 and f.y >= y0 - 3
                   and f.x + f.w <= x1 + 3 and f.y + f.h <= y1 + 3]
        tables.append(t)
    return tables


def _has_v(t: Table, x: float, y0: float, y1: float) -> bool:
    mid = (y0 + y1) / 2
    return any(abs(v.pos - x) <= 3 and v.a - 2 <= mid <= v.b + 2 for v in t.vsegs)


def _has_h(t: Table, y: float, x0: float, x1: float) -> bool:
    mid = (x0 + x1) / 2
    return any(abs(h.pos - y) <= 3 and h.a - 2 <= mid <= h.b + 2 for h in t.hsegs)


# =============================================================== xml helpers
def _el(tag: str, **attrs) -> OxmlElement:
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k) if ":" in k else k, str(v))   # drawingml attributes are unqualified
    return e


def _hex(c) -> str:
    return "%02X%02X%02X" % tuple(int(v) for v in c[:3])


def _set_run_props(run, w: DWord, rtl: bool, spacing: float = 0.0) -> None:
    rpr = run._r.get_or_add_rPr()
    rpr.append(_el("w:rFonts", **{"w:ascii": w.font, "w:hAnsi": w.font, "w:cs": w.font,
                                  "w:eastAsia": w.font}))
    if w.bold:
        rpr.append(_el("w:b"))
        rpr.append(_el("w:bCs"))
    if w.italic:
        rpr.append(_el("w:i"))
        rpr.append(_el("w:iCs"))
    if tuple(w.color) != (0, 0, 0):
        rpr.append(_el("w:color", **{"w:val": _hex(w.color)}))
    if abs(spacing) >= 0.05:
        rpr.append(_el("w:spacing", **{"w:val": int(round(spacing * TW))}))
    pct = _scale_pct(w)
    if pct != 100:
        # character scaling: the run is exactly as wide as in the original
        rpr.append(_el("w:w", **{"w:val": max(1, min(600, pct))}))
    half = max(2, int(round(w.size * 2)))
    rpr.append(_el("w:sz", **{"w:val": half}))
    rpr.append(_el("w:szCs", **{"w:val": half}))
    if rtl:
        rpr.append(_el("w:rtl"))
        rpr.append(_el("w:lang", **{"w:bidi": "ar-SA"}))


def _scale_pct(w: DWord) -> int:
    return int(round(w.scale * 100))


def _style_key(w: DWord):
    return (w.bold, w.italic, tuple(w.color), round(w.size * 2), w.font, has_arabic(w.text),
            _scale_pct(w))


LRM = "\u200e"


def _visual(w: DWord, rtl_para: bool) -> str:
    """In a right-to-left paragraph, Latin words and numbers are wrapped in
    left-to-right marks: the PDF / scan gives their *visual* order, and the marks
    make every Word renderer show exactly that order (Arabic-context numbers such
    as 117-2025 would otherwise be re-ordered by the bidi algorithm)."""
    if rtl_para and not has_arabic(w.text) and any(c.isalnum() for c in w.text):
        return LRM + w.text + LRM
    return w.text


def _join(words: list[DWord], leading: bool, rtl_para: bool = False) -> str:
    out = ""
    prev = None
    for w in words:
        if prev is not None or leading:
            out += " " if needs_space(prev, w) else ""
        out += _visual(w, rtl_para)
        prev = w
    return out


def _dir_class(w: DWord, rtl_para: bool) -> bool:
    """True when the word is laid out right-to-left inside a paragraph of the
    given direction (Latin words / numbers in an RTL paragraph are LTR islands)."""
    if not rtl_para:
        return has_arabic(w.text)
    return has_arabic(w.text) or not any(c.isalnum() for c in w.text)


def _line_runs(par, words: list[DWord], rtl_para: bool, leading: bool = False) -> None:
    """Write one line so every word keeps its original width (character scale) and
    every gap its original width (character spacing on the space), measured in the
    font Word uses. Words are given in logical order."""
    if not words:
        return
    mets = [word_metrics(w, rtl_para or has_arabic(w.text)) for w in words]
    if any(m is None for m in mets):
        _add_runs(par, words, rtl_para)
        return
    scales = []
    for w, (adv, lb, rb, ink) in zip(words, mets):
        target = w.bbox[2] - w.bbox[0]
        natural = ink if w.ink_box else adv
        scales.append(float(np.clip(target / max(0.1, natural), 0.6, 1.6)))
    dirs = [_dir_class(w, rtl_para) for w in words]
    # blocks: maximal runs of words laid out against the paragraph direction
    block_of = list(range(len(words)))
    for i in range(1, len(words)):
        if dirs[i] != rtl_para and dirs[i - 1] != rtl_para:
            block_of[i] = block_of[i - 1]

    def block(i):
        return [j for j in range(len(words)) if block_of[j] == block_of[i] and dirs[j] != rtl_para] or [i]

    def bear(i, side):  # visual bearing (pt) of word i on its left / right side
        if not words[i].ink_box:
            return 0.0
        adv, lb, rb, ink = mets[i]
        return (lb if side == "left" else rb) * scales[i]

    def gap_spec(a, b):
        """(gap between the facing ink edges in the original, bearings of those edges)."""
        wa, wb = words[a], words[b]
        a_in = dirs[a] != rtl_para
        b_in = dirs[b] != rtl_para
        if a_in and b_in and block_of[a] == block_of[b] or (not a_in and not b_in):
            # same direction: a precedes b along that direction
            if dirs[a]:   # right-to-left: a is right of b
                return wa.bbox[0] - wb.bbox[2], bear(a, "left") + bear(b, "right")
            return wb.bbox[0] - wa.bbox[2], bear(a, "right") + bear(b, "left")
        if b_in:      # entering an island: the space touches the island's near edge
            blk = block(b)
            if rtl_para:
                near = max(blk, key=lambda j: words[j].bbox[2])
                return wa.bbox[0] - words[near].bbox[2], bear(a, "left") + bear(near, "right")
            near = min(blk, key=lambda j: words[j].bbox[0])
            return words[near].bbox[0] - wa.bbox[2], bear(a, "right") + bear(near, "left")
        blk = block(a)  # leaving an island
        if rtl_para:
            far = min(blk, key=lambda j: words[j].bbox[0])
            return words[far].bbox[0] - wb.bbox[2], bear(far, "left") + bear(b, "right")
        far = max(blk, key=lambda j: words[j].bbox[2])
        return wb.bbox[0] - words[far].bbox[2], bear(far, "right") + bear(b, "left")

    if leading:
        run = par.add_run(" ")
        _set_run_props(run, words[0], False)
    for i, w in enumerate(words):
        if i and needs_space(words[i - 1], w):
            gap, bearings = gap_spec(i - 1, i)
            sp = space_advance(words[i - 1])
            extra = float(np.clip(gap - bearings - sp, -0.95 * sp, 6 * w.size))
            # wider gap: character spacing on the space; narrower gap: a horizontally
            # scaled space (renderers ignore negative spacing on a space)
            sw = copy.copy(words[i - 1])
            sw.scale = 1.0 if extra >= 0 else max(0.05, (sp + extra) / sp)
            run = par.add_run(" ")
            _set_run_props(run, sw, has_arabic(words[i - 1].text), spacing=max(0.0, extra))
        w.scale = scales[i]
        run = par.add_run(_visual(w, rtl_para))
        _set_run_props(run, w, has_arabic(w.text))


def _add_runs(par, words: list[DWord], rtl_para: bool = False) -> None:
    groups: list[list[DWord]] = []
    for w in words:
        if groups and _style_key(groups[-1][-1]) == _style_key(w):
            groups[-1].append(w)
        else:
            groups.append([w])
    for gi, g in enumerate(groups):
        text = _join(g, leading=gi > 0, rtl_para=rtl_para)
        run = par.add_run(text)
        _set_run_props(run, g[0], has_arabic(text))


def _para_props(par, *, rtl: bool, align: str | None, before: float = 0, after: float = 0,
                line: float | None = None, ind_start: float = 0, ind_end: float = 0,
                tabs: list[float] | None = None) -> None:
    ppr = par._p.get_or_add_pPr()
    for tag in ("w:bidi", "w:spacing", "w:ind", "w:jc"):
        for old in ppr.findall(qn(tag)):
            ppr.remove(old)
    ppr.append(_el("w:widowControl", **{"w:val": "0"}))
    if tabs:
        # "left" == start in both directions; positions are measured from the start
        # margin (the right one in a right-to-left paragraph)
        t = _el("w:tabs")
        for pos in sorted(tabs):
            t.append(_el("w:tab", **{"w:val": "left", "w:pos": int(round(pos * TW))}))
        ppr.append(t)
    if rtl:
        ppr.append(_el("w:bidi"))
    sp = {"w:before": int(max(0, before) * TW), "w:after": int(max(0, after) * TW)}
    if line:
        sp.update({"w:line": int(line * TW), "w:lineRule": "exact"})
    ppr.append(_el("w:spacing", **sp))
    if ind_start or ind_end:
        # w:start / w:end are logical (right / left in an RTL paragraph)
        ppr.append(_el("w:ind", **{"w:start": int(ind_start * TW), "w:end": int(ind_end * TW)}))
    if align:
        ppr.append(_el("w:jc", **{"w:val": align}))


# =============================================================== paragraphs
def segments(ln: DLine) -> list[list[DWord]]:
    """Split a line at wide horizontal gaps (unruled tables, forms, columns).
    Segments are returned in reading order, words inside in logical order."""
    ws = sorted(ln.words, key=lambda w: w.bbox[0])
    gap_min = 1.5 * ln.size
    segs: list[list[DWord]] = [[ws[0]]]
    for w in ws[1:]:
        if w.bbox[0] - segs[-1][-1].bbox[2] > gap_min:
            segs.append([w])
        else:
            segs[-1].append(w)
    if ln.rtl:
        segs.reverse()
    return [logical_order(sg, ln.rtl) for sg in segs]


@dataclass
class Para:
    lines: list[DLine]

    @property
    def bbox(self):
        bs = [ln.bbox for ln in self.lines]
        return min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)

    @property
    def rtl(self) -> bool:
        return self.lines[0].rtl


def group_paragraphs(lines: list[DLine], col0: float, col1: float) -> list[Para]:
    """Consecutive lines form one paragraph only where the text really wrapped:
    the next line's first word would not have fitted at the end of the line above."""
    lines = sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0]))
    width = col1 - col0
    paras: list[Para] = []
    for ln in lines:
        if len(segments(ln)) > 1:          # tab-positioned line: its own paragraph
            paras.append(Para([ln]))
            paras[-1].tabbed = True
            continue
        if paras and not getattr(paras[-1], "tabbed", False):
            p = paras[-1]
            last = p.lines[-1]
            lb, cb = last.bbox, ln.bbox
            gap = cb[1] - lb[3]
            pitch = cb[1] - lb[1]
            same_style = abs(ln.size - last.size) < 0.15 * last.size and ln.rtl == last.rtl
            horiz = min(lb[2], cb[2]) - max(lb[0], cb[0]) > 0
            ok_pitch = len(p.lines) < 2 or abs(pitch - (lb[1] - p.lines[-2].bbox[1])) < 0.25 * pitch
            limit = (1.0 if ln.group == last.group else 0.75) * last.size
            first = ln.words[0]
            free = width - (lb[2] - lb[0])
            wrapped = free < (first.bbox[2] - first.bbox[0]) + 0.6 * last.size
            if same_style and horiz and -0.2 * last.size <= gap < limit and ok_pitch and wrapped:
                p.lines.append(ln)
                continue
        paras.append(Para([ln]))
    return paras


def _stacks_and_bands(paras: list[Para]) -> list[tuple[str, float, object]]:
    """Flow items: paragraphs, or 'bands' where blocks of text sit side by side
    (letterhead columns, signature blocks...) - written as a borderless layout table
    so each block keeps its own position instead of being stacked."""
    stacks: list[list[Para]] = []
    for p in sorted(paras, key=lambda p: p.bbox[1]):
        x0, y0, x1, y1 = p.bbox
        size = p.lines[0].size
        for st in stacks:
            sx0, _, sx1, _ = _union([q.bbox for q in st])
            last = st[-1].bbox
            ov = min(x1, sx1) - max(x0, sx0)
            if ov > 0.3 * min(x1 - x0, sx1 - sx0) and 0 <= y0 - last[3] < 2.5 * size:
                st.append(p)
                break
        else:
            stacks.append([p])
    bands: list[list[list[Para]]] = []
    for st in sorted(stacks, key=lambda st: _union([q.bbox for q in st])[1]):
        b = _union([q.bbox for q in st])
        for band in bands:
            bb = _union([q.bbox for s2 in band for q in s2])
            if min(b[3], bb[3]) - max(b[1], bb[1]) > 0:
                band.append(st)
                break
        else:
            bands.append([st])
    items: list[tuple[str, float, object]] = []
    for band in bands:
        boxes = sorted((_union([q.bbox for q in st]), i) for i, st in enumerate(band))
        disjoint = all(boxes[i][0][2] <= boxes[i + 1][0][0] + 2 for i in range(len(boxes) - 1))
        if len(band) > 1 and disjoint:
            ordered = [band[i] for _, i in boxes]          # left to right
            items.append(("b", min(b[1] for b, _ in boxes), ordered))
        else:
            for st in band:
                for p in st:
                    items.append(("p", p.bbox[1], p))
    return items


def _union(bs):
    return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))


def _alignment(p: Para, col0: float, col1: float, tol: float | None = None) -> tuple[str | None, float, float]:
    """(jc, indent_start, indent_end) for a paragraph inside the column [col0, col1].

    NB: in a bidi paragraph Word / LibreOffice read jc="left" as start (= right)
    and jc="right" as end (= left); w:ind start / end are logical too.  The ragged
    side never gets an indent so the text can breathe like in the original."""
    if tol is None:
        tol = max(4.0, 0.6 * p.lines[0].size)
    bbs = [ln.bbox for ln in p.lines]
    left_ok = [abs(b[0] - col0) < tol for b in bbs]
    right_ok = [abs(b[2] - col1) < tol for b in bbs]
    centre = (col0 + col1) / 2
    centred = [abs((b[0] + b[2]) / 2 - centre) < tol for b in bbs]
    x0 = min(b[0] for b in bbs)
    x1 = max(b[2] for b in bbs)
    same_x0 = all(abs(b[0] - x0) < tol for b in bbs)
    same_x1 = all(abs(b[2] - x1) < tol for b in bbs)
    if len(bbs) >= 2 and all(left_ok[:-1]) and all(right_ok[:-1]):
        return "both", 0, 0
    if all(centred) and not (all(left_ok) or all(right_ok)):
        return "center", 0, 0
    if p.rtl:
        if all(right_ok):
            return None, 0, 0
        if all(left_ok):
            return "right", 0, 0
        if same_x1 or not same_x0:
            return None, col1 - x1, 0
        return "right", 0, x0 - col0
    if all(left_ok):
        return None, 0, 0
    if all(right_ok):
        return "right", 0, 0
    if same_x0 or not same_x1:
        return None, x0 - col0, 0
    return "right", 0, col1 - x1


# =============================================================== pictures
def _anchor_picture(par, png: bytes, x_pt: float, y_pt: float, w_pt: float, h_pt: float,
                    z: int = 0, behind: bool = True) -> None:
    run = par.add_run()
    run.add_picture(io.BytesIO(png), width=Emu(int(w_pt * EMU_PER_PT)),
                    height=Emu(int(h_pt * EMU_PER_PT)))
    inline = run._r.find(".//" + qn("wp:inline"))
    cx, cy = inline.find(qn("wp:extent")).get("cx"), inline.find(qn("wp:extent")).get("cy")
    anchor = _el("wp:anchor", distT=0, distB=0, distL=0, distR=0, simplePos=0, relativeHeight=z,
                 behindDoc=1 if behind else 0, locked=0, layoutInCell=0, allowOverlap=1)
    anchor.append(_el("wp:simplePos", x=0, y=0))
    ph = _el("wp:positionH", relativeFrom="page")
    po = OxmlElement("wp:posOffset")
    po.text = str(int(x_pt * EMU_PER_PT))
    ph.append(po)
    pv = _el("wp:positionV", relativeFrom="page")
    po = OxmlElement("wp:posOffset")
    po.text = str(int(y_pt * EMU_PER_PT))
    pv.append(po)
    anchor.append(ph)
    anchor.append(pv)
    anchor.append(_el("wp:extent", cx=cx, cy=cy))
    anchor.append(_el("wp:effectExtent", l=0, t=0, r=0, b=0))
    anchor.append(_el("wp:wrapNone"))
    for tag in ("wp:docPr", "wp:cNvGraphicFramePr", "a:graphic"):
        e = inline.find(qn(tag))
        if e is not None:
            anchor.append(copy.deepcopy(e))
    inline.getparent().replace(inline, anchor)


_SHAPE = (
    '<w:drawing {ns}><wp:anchor distT="0" distB="0" distL="0" distR="0" simplePos="0" '
    'relativeHeight="{z}" behindDoc="1" locked="0" layoutInCell="0" allowOverlap="1">'
    '<wp:simplePos x="0" y="0"/>'
    '<wp:positionH relativeFrom="page"><wp:posOffset>{x}</wp:posOffset></wp:positionH>'
    '<wp:positionV relativeFrom="page"><wp:posOffset>{y}</wp:posOffset></wp:positionV>'
    '<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/><wp:wrapNone/>'
    '<wp:docPr id="{sid}" name="{name} {sid}"/><wp:cNvGraphicFramePr/>'
    '<a:graphic><a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">'
    '<wps:wsp><wps:cNvSpPr/><wps:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
    '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
    '<a:ln><a:noFill/></a:ln></wps:spPr><wps:bodyPr/></wps:wsp></a:graphicData></a:graphic>'
    '</wp:anchor></w:drawing>')
_SHAPE_NS = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
             'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
             'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
             'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"')


class _Ids:
    """Unique docPr ids for shapes (pictures get theirs from python-docx)."""
    def __init__(self) -> None:
        self.n = 10000

    def __call__(self) -> int:
        self.n += 1
        return self.n


def _anchor_shape(par, ids: _Ids, x: float, y: float, w: float, h: float, color, z: int,
                  name: str = "Box") -> None:
    """Native Word rectangle (DrawingML), positioned on the page, behind the text."""
    from docx.oxml import parse_xml

    e = EMU_PER_PT
    xml = _SHAPE.format(ns=_SHAPE_NS, z=z, x=int(round(x * e)), y=int(round(y * e)),
                        cx=max(1, int(round(w * e))), cy=max(1, int(round(h * e))),
                        sid=ids(), name=name, fill=_hex(color))
    par.add_run()._r.append(parse_xml(xml))


def _near(c1, c2, tol: int = 10) -> bool:
    return max(abs(int(a) - int(b)) for a, b in zip(c1[:3], c2[:3])) <= tol


def _place_graphics(par, page: DPage, tables: list[Table], ids: _Ids) -> None:
    """Page colour, boxes and rules as native shapes, then pictures, at their exact
    page positions, all behind the text, in the original stacking order."""
    z = 1
    if not _near(page.page_color, (255, 255, 255), 3):
        _anchor_shape(par, ids, 0, 0, page.width, page.height, page.page_color, z, "Page")
        z += 1

    def in_table(r) -> bool:
        for t in tables:
            x0, y0, x1, y1 = t.bbox
            if r.x >= x0 - 3 and r.y >= y0 - 3 and r.x + r.w <= x1 + 3 and r.y + r.h <= y1 + 3:
                return True
        return False

    page_area = page.width * page.height
    shapes = [r for r in page.rects if not in_table(r) and r.w > 0.1 and r.h > 0.1
              and not (_near(r.color, page.page_color, 3) and r.w * r.h > 0.3 * page_area)]
    # big boxes first so smaller boxes / rules stack on top (as drawn originally)
    for r in sorted(shapes, key=lambda r: -(r.w * r.h)):
        thin = min(r.w, r.h) <= 3.5
        _anchor_shape(par, ids, r.x, r.y, r.w, r.h, r.color, z, "Line" if thin else "Box")
        z += 1
    for pic in page.pictures:
        _anchor_picture(par, pic.png, pic.x, pic.y, pic.w, pic.h, z=z)
        z += 1


# =============================================================== tables -> docx
def _cell_paragraphs(words: list[DWord]) -> list[DLine]:
    rows: list[list[DWord]] = []
    for w in sorted(words, key=lambda w: (w.bbox[1] + w.bbox[3]) / 2):
        cy = (w.bbox[1] + w.bbox[3]) / 2
        for r in rows:
            ry0 = min(x.bbox[1] for x in r)
            ry1 = max(x.bbox[3] for x in r)
            if ry0 <= cy <= ry1:
                r.append(w)
                break
        else:
            rows.append([w])
    out = []
    for r in rows:
        rtl = line_is_rtl(r)
        out.append(DLine(logical_order(r, rtl), rtl))
    return out


def _border(t: Table) -> tuple[float, tuple[int, int, int]]:
    segs = t.hsegs + t.vsegs
    if not segs:
        return 0.75, (0, 0, 0)
    bw = float(np.median([s.width for s in segs]))
    bc = tuple(int(np.median([s.color[i] for s in segs])) for i in range(3))
    return bw, bc  # type: ignore[return-value]


CELL_MARGIN = 2.0  # pt, left / right cell padding


def _write_table(doc, t: Table, margin_left: float, margin_right_edge: float, page_color) -> None:
    nr, nc = len(t.ys) - 1, len(t.xs) - 1
    all_words = [w for ws in t.cells.values() for w in ws]
    rtl = sum(has_arabic(w.text) for w in all_words) * 2 > max(1, len(all_words))
    table = doc.add_table(rows=nr, cols=nc)
    tbl = table._tbl
    tblpr = tbl.tblPr
    for e in list(tblpr):
        if e.tag in (qn("w:tblStyle"),):
            tblpr.remove(e)
    width = t.xs[-1] - t.xs[0]
    tblpr.append(_el("w:tblW", **{"w:w": int(width * TW), "w:type": "dxa"}))
    if rtl:
        tblpr.append(_el("w:bidiVisual"))
    # the indent positions the cell *text*, so the border sits one cell margin
    # further out; in a right-to-left table it is measured from the right margin
    ind = (margin_right_edge - t.xs[-1]) if rtl else (t.xs[0] - margin_left)
    tblpr.append(_el("w:tblInd", **{"w:w": int(round((ind + CELL_MARGIN) * TW)), "w:type": "dxa"}))
    bw, bc = _border(t)
    borders = _el("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        borders.append(_el(f"w:{side}", **{"w:val": "single", "w:sz": max(2, int(round(bw * 8))),
                                           "w:space": 0, "w:color": _hex(bc)}))
    tblpr.append(borders)
    tblpr.append(_el("w:tblLayout", **{"w:type": "fixed"}))
    mar = _el("w:tblCellMar")
    for side, v in (("top", 0), ("left", int(CELL_MARGIN * TW)), ("bottom", 0),
                    ("right", int(CELL_MARGIN * TW))):
        mar.append(_el(f"w:{side}", **{"w:w": v, "w:type": "dxa"}))
    tblpr.append(mar)
    grid = tbl.tblGrid
    for gc in list(grid):
        grid.remove(gc)
    col_order = list(range(nc))[::-1] if rtl else list(range(nc))   # logical -> visual
    for lc in range(nc):
        vc = col_order[lc]
        grid.append(_el("w:gridCol", **{"w:w": int((t.xs[vc + 1] - t.xs[vc]) * TW)}))

    covered = set()
    for r in range(nr):
        tr = table.rows[r]._tr
        trpr = tr.get_or_add_trPr()
        # the renderer adds the border width to the row height
        row_h = t.ys[r + 1] - t.ys[r] - bw
        if r == nr - 1:
            row_h -= bw / 2      # the outer bottom border is drawn below the last row
        row_h = max(2.0, row_h)
        trpr.append(_el("w:trHeight", **{"w:val": int(row_h * TW), "w:hRule": "atLeast"}))
        for lc in range(nc):
            vc = col_order[lc]
            cell = table.cell(r, lc)
            if (r, vc) in covered:
                continue
            # horizontal span (visual direction) while no vertical rule separates
            span = 1
            while lc + span < nc:
                nvc = col_order[lc + span]
                xb = t.xs[max(vc, nvc)]
                if _has_v(t, xb, t.ys[r], t.ys[r + 1]):
                    break
                span += 1
            vis_cols = sorted(col_order[lc:lc + span])
            x0c, x1c = t.xs[vis_cols[0]], t.xs[vis_cols[-1] + 1]
            # vertical merge
            rspan = 1
            while r + rspan < nr and not _has_h(t, t.ys[r + rspan], x0c, x1c):
                rspan += 1
            if span > 1:
                cell = cell.merge(table.cell(r, lc + span - 1))
            if rspan > 1:
                cell = cell.merge(table.cell(r + rspan - 1, lc + span - 1))
            for rr in range(r, r + rspan):
                for c in vis_cols:
                    covered.add((rr, c))
            y0c, y1c = t.ys[r], t.ys[r + rspan]
            tcpr = cell._tc.get_or_add_tcPr()
            tcpr.append(_el("w:vAlign", **{"w:val": "top"}))
            for f in t.fills:
                ix = max(0, min(f.x + f.w, x1c) - max(f.x, x0c))
                iy = max(0, min(f.y + f.h, y1c) - max(f.y, y0c))
                if ix * iy > 0.6 * (x1c - x0c) * (y1c - y0c) and np.abs(np.array(f.color) - page_color).max() > 12:
                    tcpr.append(_el("w:shd", **{"w:val": "clear", "w:color": "auto", "w:fill": _hex(f.color)}))
                    break
            words = [w for rr in range(r, r + rspan) for c in vis_cols for w in t.cells.get((rr, c), [])]
            lines = _cell_paragraphs(words)
            first = cell.paragraphs[0]
            if not lines:
                _para_props(first, rtl=rtl, align=None)
                continue
            top = y0c + bw / 2
            for i, ln in enumerate(lines):
                par = first if i == 0 else cell.add_paragraph()
                b = ln.bbox
                tol = max(3.0, 0.5 * ln.size)
                ccx = (x0c + x1c) / 2
                if abs((b[0] + b[2]) / 2 - ccx) < tol:
                    align = "center"
                elif ln.rtl:   # see _alignment: "right" == physical left in RTL
                    align = None if abs(b[2] - x1c) < abs(b[0] - x0c) else "right"
                else:
                    align = None if abs(b[0] - x0c) < abs(b[2] - x1c) else "right"
                # exact line height so that baseline = line top + 0.8 L is where it was
                L = max(ln.size * 0.9, (ln.baseline - top) / 0.8)
                remaining = (y1c - top) - (ln.size * 1.0) * (len(lines) - i - 1)
                L = min(L, max(ln.size * 0.9, remaining))
                _para_props(par, rtl=ln.rtl, align=align, line=L)
                _line_runs(par, ln.words, ln.rtl)
                top += L


# =============================================================== document
def _write_para(par, p: Para, cursor: float, col0: float, col1: float,
                snap: float | None = None) -> float:
    """Write paragraph p into the (empty) paragraph `par` so its first baseline lands
    where it was, inside the column [col0, col1]. Returns the new vertical cursor."""
    size = float(np.median([ln.size for ln in p.lines]))
    if len(p.lines) > 1:
        pitch = float(np.median([b.bbox[1] - a.bbox[1] for a, b in zip(p.lines, p.lines[1:])]))
    else:
        pitch = max(size * 1.25, p.lines[0].bbox[3] - p.lines[0].bbox[1])
    # exact line spacing L puts the baseline at line_top + 0.8 L (Word and
    # LibreOffice agree), so the first baseline lands where it was
    line_top = p.lines[0].baseline - 0.8 * pitch
    if line_top < cursor and len(p.lines) == 1:
        # tight under the previous block: a smaller exact line keeps the baseline
        pitch = max(size, (p.lines[0].baseline - cursor) / 0.8)
        line_top = p.lines[0].baseline - 0.8 * pitch
    before = line_top - cursor
    if getattr(p, "tabbed", False):
        ln = p.lines[0]
        segs = segments(ln)

        def start_of(sg):   # distance of a segment's start edge from the margin
            if ln.rtl:
                return col1 - max(w.bbox[2] for w in sg)
            return min(w.bbox[0] for w in sg) - col0

        _para_props(par, rtl=ln.rtl, align=None, before=before, line=pitch,
                    ind_start=start_of(segs[0]), ind_end=-min(4.0, 0.3 * size),
                    tabs=[start_of(sg) for sg in segs[1:]])
        for si, sg in enumerate(segs):
            if si:
                par.add_run("\t")
            _line_runs(par, sg, ln.rtl)
        return max(cursor, line_top) + pitch
    jc, i_start, i_end = _alignment(p, col0, col1, snap)
    # font metric differences must never cause extra line wraps: the ragged side
    # may run a little into the margin (default jc: aligned on the start side,
    # ragged end; jc="right" is the end side in both directions, ragged start)
    slack = min(4.0, 0.3 * size)
    if jc is None and i_end == 0:
        i_end = -slack
    elif jc == "right" and i_start == 0:
        i_start = -slack
    _para_props(par, rtl=p.rtl, align=jc, before=before, line=pitch,
                ind_start=i_start, ind_end=i_end)
    for li, ln in enumerate(p.lines):
        _line_runs(par, ln.words, p.rtl, leading=li > 0)
    return max(cursor, line_top) + pitch * len(p.lines)


def _write_band(doc, stacks: list[list[Para]], cursor: float, col0: float, col1: float) -> float:
    """Side-by-side blocks as a one-row borderless table spanning the text column;
    every block is written in its own cell with the same exact positioning."""
    boxes = [_union([q.bbox for q in st]) for st in stacks]
    bounds = [col0] + [(boxes[i][2] + boxes[i + 1][0]) / 2 for i in range(len(boxes) - 1)] + [col1]
    table = doc.add_table(rows=1, cols=len(stacks))
    tbl = table._tbl
    tblpr = tbl.tblPr
    for e in list(tblpr):
        if e.tag == qn("w:tblStyle"):
            tblpr.remove(e)
    tblpr.append(_el("w:tblW", **{"w:w": int((col1 - col0) * TW), "w:type": "dxa"}))
    tblpr.append(_el("w:tblInd", **{"w:w": 0, "w:type": "dxa"}))
    borders = _el("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        borders.append(_el(f"w:{side}", **{"w:val": "nil"}))
    tblpr.append(borders)
    tblpr.append(_el("w:tblLayout", **{"w:type": "fixed"}))
    mar = _el("w:tblCellMar")
    for side in ("top", "left", "bottom", "right"):
        mar.append(_el(f"w:{side}", **{"w:w": 0, "w:type": "dxa"}))
    tblpr.append(mar)
    grid = tbl.tblGrid
    for gc in list(grid):
        grid.remove(gc)
    for i in range(len(stacks)):
        grid.append(_el("w:gridCol", **{"w:w": int((bounds[i + 1] - bounds[i]) * TW)}))
    end = cursor
    for i, st in enumerate(stacks):
        cell = table.cell(0, i)
        tcpr = cell._tc.get_or_add_tcPr()
        tcpr.append(_el("w:tcW", **{"w:w": int((bounds[i + 1] - bounds[i]) * TW), "w:type": "dxa"}))
        c = cursor
        for j, p in enumerate(sorted(st, key=lambda q: q.bbox[1])):
            par = cell.paragraphs[0] if j == 0 else cell.add_paragraph()
            # cell edges are arbitrary (mid-gap): position exactly, never snap to them
            c = _write_para(par, p, c, bounds[i], bounds[i + 1], snap=0.5)
        end = max(end, c)
    return end


def build_docx(pages: list[DPage]) -> bytes:
    doc = Document()
    body = doc.element.body
    for p in list(body):
        if p.tag == qn("w:p"):
            body.remove(p)
    ids = _Ids()
    for pi, page in enumerate(pages):
        section = doc.sections[0] if pi == 0 else doc.add_section(WD_SECTION.NEW_PAGE)
        tables = detect_tables(page)
        # words that belong to table cells
        free_lines: list[DLine] = []
        for ln in page.lines:
            rest = []
            for w in ln.words:
                cx, cy = (w.bbox[0] + w.bbox[2]) / 2, (w.bbox[1] + w.bbox[3]) / 2
                for t in tables:
                    x0, y0, x1, y1 = t.bbox
                    if x0 <= cx <= x1 and y0 <= cy <= y1:
                        c = max(i for i in range(len(t.xs) - 1) if t.xs[i] <= cx)
                        r = max(i for i in range(len(t.ys) - 1) if t.ys[i] <= cy)
                        t.cells.setdefault((r, c), []).append(w)
                        break
                else:
                    rest.append(w)
            if rest:
                free_lines.append(DLine(rest, ln.rtl, ln.group))

        bxs = [ln.bbox for ln in free_lines] + [t.bbox for t in tables]
        col0 = min((b[0] for b in bxs), default=56.0)
        col1 = max((b[2] for b in bxs), default=page.width - 56.0)
        top = min((b[1] for b in bxs), default=56.0)
        margin_l = max(0.0, min(col0, page.width / 3))
        margin_r = max(0.0, min(page.width - col1, page.width / 3))
        margin_t = max(0.0, min(top - 2, 36.0))
        section.page_width = Emu(int(page.width * EMU_PER_PT))
        section.page_height = Emu(int(page.height * EMU_PER_PT))
        section.left_margin = Emu(int(margin_l * EMU_PER_PT))
        section.right_margin = Emu(int(margin_r * EMU_PER_PT))
        section.top_margin = Emu(int(margin_t * EMU_PER_PT))
        section.bottom_margin = Emu(int(min(18.0, page.height * 0.02) * EMU_PER_PT))
        section.header_distance = Emu(0)
        section.footer_distance = Emu(0)

        page_color = page.page_color
        items = _stacks_and_bands(group_paragraphs(free_lines, col0, col1))
        items += [("t", t.bbox[1], t) for t in tables]
        items.sort(key=lambda it: it[1])

        cursor = margin_t
        first_par = None
        if not items:
            first_par = doc.add_paragraph()
        for kind, _y, obj in items:
            if kind == "t":
                # the outer top border is drawn above the first row
                gap = obj.bbox[1] - cursor - _border(obj)[0] / 2
                if gap > 1:
                    spacer = doc.add_paragraph()
                    _para_props(spacer, rtl=False, align=None, line=max(1.0, gap))
                    first_par = first_par or spacer
                _write_table(doc, obj, margin_l, page.width - margin_r, np.array(page_color))
                cursor = obj.bbox[3]
                continue
            if kind == "b":
                cursor = _write_band(doc, obj, cursor, col0, col1)
                continue
            par = doc.add_paragraph()
            first_par = first_par or par
            cursor = _write_para(par, obj, cursor, col0, col1)
        if first_par is None and (page.rects or page.pictures):
            first_par = doc.add_paragraph()      # anchor holder (graphics are page-positioned)
            _para_props(first_par, rtl=False, align=None, line=1.0)
        if first_par is not None and (page.rects or page.pictures
                                      or not _near(page.page_color, (255, 255, 255), 3)):
            _place_graphics(first_par, page, tables, ids)
    _normalise_schema_order(doc.element.body)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


_ORDER = {
    "pPr": ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl",
            "numPr", "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens",
            "kinsoku", "wordWrap", "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN",
            "bidi", "adjustRightInd", "snapToGrid", "spacing", "ind", "contextualSpacing",
            "mirrorIndents", "suppressOverlap", "jc", "textDirection", "textAlignment",
            "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr", "sectPr", "pPrChange"],
    "rPr": ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike",
            "outline", "shadow", "emboss", "imprint", "noProof", "snapToGrid", "vanish",
            "webHidden", "color", "spacing", "w", "kern", "position", "sz", "szCs", "highlight",
            "u", "effect", "bdr", "shd", "fitText", "vertAlign", "rtl", "cs", "em", "lang",
            "eastAsianLayout", "specVanish", "oMath"],
    "tblPr": ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize",
              "tblStyleColBandSize", "tblW", "jc", "tblCellSpacing", "tblInd", "tblBorders",
              "shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption", "tblDescription"],
    "tcPr": ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd", "noWrap",
             "tcMar", "textDirection", "tcFitText", "vAlign", "hideMark"],
    "trPr": ["cnfStyle", "divId", "gridBefore", "gridAfter", "wBefore", "wAfter", "cantSplit",
             "trHeight", "tblHeader", "tblCellSpacing", "jc", "hidden"],
}
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _normalise_schema_order(body) -> None:
    """Word refuses files whose property elements are out of schema order or
    duplicated (LibreOffice is lenient): sort them, keep the last duplicate."""
    for name, order in _ORDER.items():
        rank = {t: i for i, t in enumerate(order)}
        for el in body.iter(_W + name):
            kids = list(el)
            seen: dict[str, object] = {}
            for k in kids:
                seen[k.tag] = k            # last one wins (ours come after defaults)
            keep = [k for k in kids if seen.get(k.tag) is k]
            keep.sort(key=lambda k: rank.get(k.tag.replace(_W, ""), len(order)))
            for k in kids:
                el.remove(k)
            for k in keep:
                el.append(k)


def docx_has_rtl(data: bytes) -> bool:
    """Small helper for tests: does the document contain bidi paragraphs."""
    return b"w:bidi" in _document_xml(data)


def _document_xml(data: bytes) -> bytes:
    import zipfile
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return z.read("word/document.xml")


__all__ = ["build_docx", "detect_tables", "group_paragraphs", "is_rtl_text"]
