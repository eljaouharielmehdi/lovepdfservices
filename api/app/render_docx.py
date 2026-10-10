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
from PIL import Image, ImageDraw

from .docmodel import DLine, DPage, DWord, logical_order
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


def _set_run_props(run, w: DWord, rtl: bool) -> None:
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
    half = max(2, int(round(w.size * 2)))
    rpr.append(_el("w:sz", **{"w:val": half}))
    rpr.append(_el("w:szCs", **{"w:val": half}))
    if rtl:
        rpr.append(_el("w:rtl"))
        rpr.append(_el("w:lang", **{"w:bidi": "ar-SA"}))


def _style_key(w: DWord):
    return (w.bold, w.italic, tuple(w.color), round(w.size * 2), w.font, has_arabic(w.text))


_CLOSE = set(".,:;!?\u060c\u061b\u061f)]}\u00bb%")
_OPEN = set("([{\u00ab")


def _join(words: list[DWord], leading: bool) -> str:
    out = ""
    prev = None
    for w in words:
        if prev is not None or leading:
            glue = not (w.text and all(c in _CLOSE for c in w.text)) and not (
                prev is not None and prev.text and all(c in _OPEN for c in prev.text))
            out += " " if glue else ""
        out += w.text
        prev = w
    return out


def _add_runs(par, words: list[DWord]) -> None:
    groups: list[list[DWord]] = []
    for w in words:
        if groups and _style_key(groups[-1][-1]) == _style_key(w):
            groups[-1].append(w)
        else:
            groups.append([w])
    for gi, g in enumerate(groups):
        text = _join(g, leading=gi > 0)
        run = par.add_run(text)
        _set_run_props(run, g[0], has_arabic(text))


def _para_props(par, *, rtl: bool, align: str | None, before: float = 0, after: float = 0,
                line: float | None = None, ind_start: float = 0, ind_end: float = 0) -> None:
    ppr = par._p.get_or_add_pPr()
    for tag in ("w:bidi", "w:spacing", "w:ind", "w:jc"):
        for old in ppr.findall(qn(tag)):
            ppr.remove(old)
    ppr.append(_el("w:widowControl", **{"w:val": "0"}))
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


def group_paragraphs(lines: list[DLine]) -> list[Para]:
    lines = sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0]))
    paras: list[Para] = []
    for ln in lines:
        if paras:
            p = paras[-1]
            last = p.lines[-1]
            lb, cb = last.bbox, ln.bbox
            gap = cb[1] - lb[3]
            pitch = cb[1] - lb[1]
            same_style = abs(ln.size - last.size) < 0.15 * last.size and ln.rtl == last.rtl
            horiz = min(lb[2], cb[2]) - max(lb[0], cb[0]) > 0
            ok_pitch = len(p.lines) < 2 or abs(pitch - (lb[1] - p.lines[-2].bbox[1])) < 0.25 * pitch
            limit = (1.0 if ln.group == last.group else 0.75) * last.size
            if same_style and horiz and -0.2 * last.size <= gap < limit and ok_pitch:
                p.lines.append(ln)
                continue
        paras.append(Para([ln]))
    return paras


def _alignment(p: Para, col0: float, col1: float) -> tuple[str | None, float, float]:
    """(jc, indent_start, indent_end) for a paragraph inside the column [col0, col1].

    NB: in a bidi paragraph Word / LibreOffice read jc="left" as start (= right)
    and jc="right" as end (= left); w:ind start / end are logical too.  The ragged
    side never gets an indent so the text can breathe like in the original."""
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
                    behind: bool = True) -> None:
    run = par.add_run()
    run.add_picture(io.BytesIO(png), width=Emu(int(w_pt * EMU_PER_PT)),
                    height=Emu(int(h_pt * EMU_PER_PT)))
    inline = run._r.find(".//" + qn("wp:inline"))
    cx, cy = inline.find(qn("wp:extent")).get("cx"), inline.find(qn("wp:extent")).get("cy")
    anchor = _el("wp:anchor", distT=0, distB=0, distL=0, distR=0, simplePos=0, relativeHeight=0,
                 behindDoc=1 if behind else 0, locked=1, layoutInCell=1, allowOverlap=1)
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


def _page_color(img: Image.Image) -> tuple[int, int, int]:
    a = np.asarray(img.convert("RGB"))[::6, ::6].reshape(-1, 3)
    q = (a // 16).astype(np.int32)
    keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    sel = a[keys == vals[counts.argmax()]]
    return tuple(int(v) for v in np.median(sel, axis=0))  # type: ignore[return-value]


def _background_png(page: DPage, tables: list[Table]) -> tuple[bytes | None, tuple[int, int, int]]:
    if page.background is None:
        return None, (255, 255, 255)
    img = page.background.convert("RGB").copy()
    pc = _page_color(img)
    d = ImageDraw.Draw(img)
    s = page.bg_scale
    for t in tables:   # tables are rebuilt natively
        x0, y0, x1, y1 = t.bbox
        d.rectangle([(x0 - 3) * s, (y0 - 3) * s, (x1 + 3) * s, (y1 + 3) * s], fill=pc)
    a = np.asarray(img).astype(np.int16)
    if (np.abs(a - np.array(pc)).max(axis=-1) > 40).mean() < 0.0005:
        return None, pc   # nothing but blank paper
    # keep the file small: 150 dpi is plenty for backgrounds
    target = 150 / 72
    if s > target:
        img = img.resize((int(img.width * target / s), int(img.height * target / s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue(), pc


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
        rtl = any(has_arabic(w.text) for w in r)
        out.append(DLine(logical_order(r, rtl), rtl))
    return out


def _write_table(doc, t: Table, margin_left: float, page_color) -> None:
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
    tblpr.append(_el("w:tblInd", **{"w:w": int(max(0, t.xs[0] - margin_left) * TW), "w:type": "dxa"}))
    widths = [s.width for s in t.hsegs + t.vsegs]
    colors = [s.color for s in t.hsegs + t.vsegs]
    bw = float(np.median(widths)) if widths else 0.75
    bc = tuple(int(np.median([c[i] for c in colors])) for i in range(3)) if colors else (0, 0, 0)
    borders = _el("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        borders.append(_el(f"w:{side}", **{"w:val": "single", "w:sz": max(2, int(round(bw * 8))),
                                           "w:space": 0, "w:color": _hex(bc)}))
    tblpr.append(borders)
    tblpr.append(_el("w:tblLayout", **{"w:type": "fixed"}))
    mar = _el("w:tblCellMar")
    for side, v in (("top", 0), ("left", 40), ("bottom", 0), ("right", 40)):
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
        row_h = max(2.0, t.ys[r + 1] - t.ys[r] - bw)
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
                _add_runs(par, ln.words)
                top += L


# =============================================================== document
def build_docx(pages: list[DPage]) -> bytes:
    doc = Document()
    body = doc.element.body
    for p in list(body):
        if p.tag == qn("w:p"):
            body.remove(p)
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

        bg, page_color = _background_png(page, tables)
        items = [("p", p.bbox[1], p) for p in group_paragraphs(free_lines)]
        items += [("t", t.bbox[1], t) for t in tables]
        items.sort(key=lambda it: it[1])

        cursor = margin_t
        first_par = None
        if not items:
            first_par = doc.add_paragraph()
        for kind, _y, obj in items:
            if kind == "t":
                gap = obj.bbox[1] - cursor
                if gap > 1:
                    spacer = doc.add_paragraph()
                    _para_props(spacer, rtl=False, align=None, line=max(1.0, gap))
                    first_par = first_par or spacer
                _write_table(doc, obj, margin_l, np.array(page_color))
                cursor = obj.bbox[3]
                continue
            p: Para = obj
            par = doc.add_paragraph()
            first_par = first_par or par
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
            jc, i_start, i_end = _alignment(p, col0, col1)
            # font metric differences must never cause extra line wraps: the
            # ragged side may run a little into the margin
            # (default jc: aligned on the start side, ragged end; jc="right" is the
            # end side in both directions, ragged start)
            slack = min(4.0, 0.3 * size)
            if jc is None and i_end == 0:
                i_end = -slack
            elif jc == "right" and i_start == 0:
                i_start = -slack
            _para_props(par, rtl=p.rtl, align=jc, before=before, line=pitch,
                        ind_start=i_start, ind_end=i_end)
            for li, ln in enumerate(p.lines):
                if li:
                    par.add_run(" ")
                _add_runs(par, ln.words)
            cursor = max(cursor, line_top) + pitch * len(p.lines)
        if bg is not None and first_par is not None:
            _anchor_picture(first_par, bg, 0, 0, page.width, page.height, behind=True)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def docx_has_rtl(data: bytes) -> bool:
    """Small helper for tests: does the document contain bidi paragraphs."""
    return b"w:bidi" in _document_xml(data)


def _document_xml(data: bytes) -> bytes:
    import zipfile
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return z.read("word/document.xml")


__all__ = ["build_docx", "detect_tables", "group_paragraphs", "is_rtl_text"]
