"""Schiffs-grafik: die vektor-zeichnung aus dem design-mockup.

Uebernommen aus dem Claude-Design-projekt "Ship Mockup.dc.html"
(`7c1401de-e967-41a5-818a-742e9a6015b3`). Das mockup ist ein SVG mit
viewBox `0 0 300 170`; hier steht dieselbe geometrie noch einmal in
deklarativer form, damit sie ohne SVG-parser in GL-batches uebersetzt
werden kann.

Koordinaten
-----------
Die shape-listen unten sind in **SVG-koordinaten** notiert (x nach rechts,
y nach UNTEN, ursprung links oben), damit ein abgleich mit dem mockup
zeile fuer zeile moeglich bleibt. ``build()`` rechnet sie einmalig in den
**lokalen schiffsraum** um:

    local_x = svg_x - SVG_CX      # +x = nasenrichtung
    local_y = SVG_CY - svg_y      # +y nach OBEN (ortho-konvention)

Der ursprung liegt damit im sichtbaren mittelpunkt des schiffs, und
``rendering.Renderer._draw_ship_sprite`` muss nur noch drehen, skalieren
und verschieben. Die einheit bleibt "SVG-pixel"; ``SHIP_LENGTH`` sagt,
wieviele davon die gesamtlaenge sind, sodass der renderer auf eine
gewuenschte bildschirmlaenge normieren kann.

Alles hier ist rein numerisch -- kein pygame, kein moderngl, kein numba.
"""

import math

import numpy as np


# --- viewBox-bezug ----------------------------------------------------------
SVG_CX = 147.0          # mitte zwischen duesen-lippe (x=32) und nasenspitze (x=262)
SVG_CY = 85.0           # laengsachse des rumpfs
SHIP_LENGTH = 230.0     # x=32 .. x=262, die volle silhouette
SHIP_HEIGHT = 118.0     # y=26 .. y=144, inklusive der radiatoren

# Platzhalter fuer die akzentfarbe: ueberall dort, wo das mockup
# `{{ accent }}` einsetzt. `build()` ersetzt ihn durch den uebergebenen wert.
ACCENT = "@accent"

DEFAULT_ACCENT = "#4de1e8"


# --- kleine geometrie-helfer ------------------------------------------------

def _hex_rgb(value):
    """'#rrggbb' (oder '#rgb') -> (r, g, b) in 0..1."""
    text = str(value).strip().lstrip('#')
    if len(text) == 3:
        text = ''.join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError("ship_art: unbrauchbare farbe %r" % (value,))
    return tuple(int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rect(x, y, w, h):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _quad_bezier(p0, p1, p2, steps=10):
    """Quadratische bezier-kurve abtasten (ohne den startpunkt)."""
    out = []
    for i in range(1, steps + 1):
        t = i / float(steps)
        u = 1.0 - t
        out.append((
            u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0],
            u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1],
        ))
    return out


def _circle(cx, cy, r, steps=24):
    return [(cx + r * math.cos(2.0 * math.pi * i / steps),
             cy + r * math.sin(2.0 * math.pi * i / steps)) for i in range(steps)]


def _arc_half(cx, cy, r, upward, steps=12):
    """Halbkreis um (cx, cy) -- die beiden sensor-kuppeln des mockups.

    `upward=True` entspricht `a r,r 0 0 1 2r,0` (SVG-sweep im uhrzeigersinn,
    auf dem bildschirm also nach oben), `False` dem gespiegelten sweep 0.
    """
    sign = -1.0 if upward else 1.0
    pts = []
    for i in range(steps + 1):
        a = math.pi * i / steps
        pts.append((cx - r * math.cos(a), cy + sign * r * math.sin(a)))
    return pts


def _dash(points, on, off):
    """Eine polylinie in strich-segmente zerlegen (stroke-dasharray)."""
    runs = []
    pos = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 0.0:
            continue
        t = 0.0
        while t < seg:
            phase = math.fmod(pos + t, on + off)
            if phase < on:
                step = min(on - phase, seg - t)
                a = t / seg
                b = (t + step) / seg
                runs.append([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a),
                             (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)])
            else:
                step = min(on + off - phase, seg - t)
            t += max(step, 1e-6)
        pos += seg
    return runs


def _signed_area(pts):
    total = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def _cross3(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _in_triangle(p, a, b, c):
    d1 = _cross3(a, b, p)
    d2 = _cross3(b, c, p)
    d3 = _cross3(c, a, p)
    return ((d1 >= 0.0 and d2 >= 0.0 and d3 >= 0.0)
            or (d1 <= 0.0 and d2 <= 0.0 and d3 <= 0.0))


def _triangulate(polygon):
    """Ear-clipping fuer einfache polygone.

    Ein fan vom ersten punkt aus wuerde reichen, solange alles konvex ist --
    die triebwerksglocke (`M66,76 Q54,78 40,66 ...`) ist es nicht, ihre beiden
    bezier-flanken woelben sich nach innen. Ear-clipping deckt beide faelle ab
    und laeuft einmal beim aufbau, nicht pro frame.
    """
    pts = [(float(x), float(y)) for x, y in polygon]
    if (len(pts) >= 2 and math.isclose(pts[0][0], pts[-1][0])
            and math.isclose(pts[0][1], pts[-1][1])):
        pts.pop()
    if len(pts) < 3:
        return []
    if _signed_area(pts) < 0.0:
        pts.reverse()

    idx = list(range(len(pts)))
    tris = []
    guard = 4 * len(pts) + 16
    while len(idx) > 3 and guard > 0:
        guard -= 1
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if _cross3(a, b, c) <= 1e-12:
                continue
            if any(_in_triangle(pts[j], a, b, c)
                   for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((a, b, c))
            idx.pop(k)
            break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(pts[i] for i in idx))
    return tris


# --- shape-beschreibung -----------------------------------------------------
#
# Jeder eintrag ist (points, closed, fill, stroke, stroke_width, alpha).
# `fill`/`stroke` duerfen None sein; ACCENT wird in build() ersetzt.
#
# Die REIHENFOLGE ist die zeichenreihenfolge des mockups (painter's
# algorithmus) und darf nicht nach farbe sortiert werden: die rumpf-fuellung
# liegt bewusst ueber den radiator-streben, die akzent-instrumente ueber dem
# rumpf.

def _shapes():
    shapes = []

    def add(points, closed=True, fill=None, stroke=None, width=1.0, alpha=1.0):
        shapes.append((points, closed, fill, stroke, width, alpha))

    def vline(x, y0, y1, stroke, width):
        add([(x, y0), (x, y1)], False, None, stroke, width)

    def hline(y, x0, x1, stroke, width):
        add([(x0, y), (x1, y)], False, None, stroke, width)

    # --- radiator-paneele (oben/unten) --------------------------------------
    add(_rect(118, 26, 68, 38), True, "#1c2135", "#78809a", 1.1)
    add(_rect(118, 106, 68, 38), True, "#1c2135", "#78809a", 1.1)
    for x in range(126, 183, 8):
        vline(x, 26, 64, "#454d6b", 0.9)
    for x in range(126, 183, 8):
        vline(x, 106, 144, "#454d6b", 0.9)
    for y in (36, 54, 116, 134):
        hline(y, 118, 186, "#7d8496", 1.1)
    for x, y in ((128, 64), (168, 64), (128, 100), (168, 100)):
        add(_rect(x, y, 8, 6), True, "#3d4463", "#a8aec0", 1.0)

    # --- triebwerks-traeger --------------------------------------------------
    add([(96, 71), (70, 68), (70, 102), (96, 99)], True, "#282e46", "#c9ccd6", 1.4)
    add([(88, 69), (88, 101)], False, None, "#8f97ab", 0.9)
    add([(80, 68.5), (80, 101.5)], False, None, "#8f97ab", 0.9)
    add(_rect(74, 60, 16, 8), True, "#3d4463", "#a8aec0", 1.0)
    add(_rect(74, 102, 16, 8), True, "#3d4463", "#a8aec0", 1.0)

    # --- triebwerksglocke ----------------------------------------------------
    bell = [(66, 76)]
    bell += _quad_bezier((66, 76), (54, 78), (40, 66))
    bell += [(40, 104)]
    bell += _quad_bezier((40, 104), (54, 92), (66, 94))
    add(bell, True, "#2b3149", "#e9e9ed", 1.6)
    add([(64, 79)] + _quad_bezier((64, 79), (52, 80), (43, 70)),
        False, None, "#6b7288", 0.9)
    add([(64, 91)] + _quad_bezier((64, 91), (52, 90), (43, 100)),
        False, None, "#6b7288", 0.9)
    add([(47, 72), (47, 98)], False, None, "#6b7288", 0.9)
    add([(40, 70), (32, 85), (40, 100)], False, None, ACCENT, 1.6, 0.95)
    add(_rect(66, 74, 4, 22), True, "#3d4463", "#b3b9c9", 1.0)

    # --- hauptrumpf ----------------------------------------------------------
    hull = [(258, 79), (250, 76), (236, 74), (216, 71), (200, 70), (190, 69),
            (120, 69), (104, 70), (96, 72), (96, 98), (104, 100), (120, 101),
            (190, 101), (200, 100), (216, 99), (236, 96), (250, 94), (258, 91)]
    add(hull, True, "#2f3550", "#e9e9ed", 1.7)
    add([(258, 79), (262, 81), (262, 89), (258, 91)], True, "#3a4160", "#c9ccd6", 1.2)

    # --- rumpf-aufbauten -----------------------------------------------------
    add(_rect(122, 74, 30, 22), True, "#3d4463", "#a8aec0", 1.0)
    add(_rect(158, 76, 22, 18), True, "#3d4463", "#a8aec0", 1.0)
    hline(80, 122, 152, "#606a86", 0.85)
    hline(90, 122, 152, "#606a86", 0.85)
    vline(132, 74, 96, "#606a86", 0.85)
    vline(142, 74, 96, "#606a86", 0.85)
    vline(166, 76, 94, "#606a86", 0.85)
    vline(173, 76, 94, "#606a86", 0.85)

    # --- schotten und laengslinien -------------------------------------------
    for x, y0, y1 in ((190, 69, 101), (200, 70, 100), (216, 71, 99),
                      (236, 74, 96), (104, 70, 100), (120, 69, 101)):
        vline(x, y0, y1, "#8f97ab", 0.95)
    add([(96, 77), (250, 78.5)], False, None, "#5b6480", 0.75)
    add([(96, 93), (250, 91.5)], False, None, "#5b6480", 0.75)
    for run in _dash([(96, 85), (256, 85)], 4.0, 5.0):
        add(run, False, None, "#7b839a", 0.6)

    # --- akzent-modul --------------------------------------------------------
    add([(216, 71), (236, 74), (236, 96), (216, 99)], True, "#454d70", "#dcdfe8", 1.2)
    add(_rect(220, 79, 12, 2.4), True, ACCENT, None, 1.0, 0.9)
    add(_rect(220, 88.6, 12, 2.4), True, ACCENT, None, 1.0, 0.9)

    # --- akzent-instrumente --------------------------------------------------
    add([(256, 85), (246, 82), (246, 88), (256, 85)], False, None, ACCENT, 1.2)
    add(_circle(206, 85, 4), True, None, ACCENT, 1.2)
    add(_circle(168, 85, 5), True, None, ACCENT, 1.2, 0.65)
    add(_arc_half(194, 64, 7, upward=True), False, None, ACCENT, 1.2, 0.95)
    add(_arc_half(194, 106, 7, upward=False), False, None, ACCENT, 1.2, 0.95)
    add([(194, 69), (194, 64)], False, None, ACCENT, 1.2)
    add([(194, 101), (194, 106)], False, None, ACCENT, 1.2)

    # --- andockluken ---------------------------------------------------------
    add(_rect(172, 63, 10, 6), True, "#3d4463", "#b3b9c9", 1.0)
    add(_rect(172, 101, 10, 6), True, "#3d4463", "#b3b9c9", 1.0)
    add(_rect(106, 63, 10, 7), True, "#3d4463", "#b3b9c9", 1.0)
    add(_rect(106, 100, 10, 7), True, "#3d4463", "#b3b9c9", 1.0)

    return shapes


# --- die abgasfahne ---------------------------------------------------------
#
# Im mockup eine ellipse (cx 34, cy 85, rx 32, ry 21) mit radialem verlauf,
# dessen zentrum auf ihrer RECHTEN spitze sitzt (`cx="1" cy="0.5"` in
# objectBoundingBox-einheiten, also bei x = 66): hell an der duese, nach
# hinten auslaufend.
#
# Der ortho-zeichenweg kennt nur uniforme farben, also wird der verlauf aus
# ineinander liegenden kopien der ellipse gebaut -- jede zur duesen-spitze hin
# geschrumpft, jede mit derselben kleinen alpha. Wo alle N schichten
# uebereinander liegen (an der duese) ergibt sich PLUME_ALPHA, am hinteren
# ende traegt nur noch eine schicht.
PLUME_LAYERS = 18
PLUME_ALPHA = 0.35
PLUME_ANCHOR = (66.0, 85.0)


def _plume_polygons():
    ax, ay = PLUME_ANCHOR
    base = [(34.0 + 32.0 * ux, 85.0 + 21.0 * uy)
            for ux, uy in _circle(0.0, 0.0, 1.0, steps=40)]
    polys = []
    for k in range(PLUME_LAYERS):
        s = 1.0 - k / float(PLUME_LAYERS)
        polys.append([(ax + s * (px - ax), ay + s * (py - ay)) for px, py in base])
    return polys


def _to_local(points):
    return [(float(x) - SVG_CX, SVG_CY - float(y)) for x, y in points]


class ShipGeometry:
    """Fertig triangulierte schiffs-grafik im lokalen schiffsraum.

    `verts` haelt ALLE punkte (rumpf und fahne) in einem array; die eintraege
    in `ops` / `plume_ops` sind nur `(mode, rgba, width, start, count)`-slices
    darauf. Der renderer dreht und skaliert damit einmal pro frame ein
    einziges array statt dutzender kleiner.
    """

    __slots__ = ("verts", "ops", "plume_ops", "accent", "length", "height")

    def __init__(self, verts, ops, plume_ops, accent):
        self.verts = verts
        self.ops = ops
        self.plume_ops = plume_ops
        self.accent = accent
        self.length = SHIP_LENGTH
        self.height = SHIP_HEIGHT


def build(accent=DEFAULT_ACCENT):
    """Die grafik in GL-taugliche batches uebersetzen.

    accent: hex-farbe fuer alles, was im mockup `{{ accent }}` benutzt --
            duesen-lippe, modulstreifen, instrumente und die abgasfahne.
    """
    accent_rgb = _hex_rgb(accent)

    def resolve(color):
        return accent_rgb if color == ACCENT else _hex_rgb(color)

    verts = []

    def emit(bucket, mode, rgba, width, points):
        if not points:
            return
        start = len(verts)
        verts.extend(points)
        bucket.append((mode, rgba, float(width), start, len(points)))

    # --- fahne zuerst: sie liegt hinter allem anderen ------------------------
    #
    # Alle schichten gehen in EINEN draw-call. Bei src-alpha-blending ohne
    # tiefentest blendet jedes dreieck einzeln und in abgabereihenfolge gegen
    # den framebuffer -- die ueberlagerung der schichten, aus der der verlauf
    # entsteht, bleibt also erhalten.
    plume_ops = []
    layer_alpha = 1.0 - (1.0 - PLUME_ALPHA) ** (1.0 / PLUME_LAYERS)
    plume_tris = []
    for poly in _plume_polygons():
        for tri in _triangulate(_to_local(poly)):
            plume_tris.extend(tri)
    emit(plume_ops, "tris", accent_rgb + (layer_alpha,), 1.0, plume_tris)

    # --- rumpf ---------------------------------------------------------------
    #
    # Aufeinanderfolgende shapes mit gleichem stil werden zusammengefasst:
    # erst alle fuellungen der gruppe, dann alle konturen. Das sind im mockup
    # genau die `<g>`-geschwister (radiator-paneele, luken, streben), die
    # einander nicht ueberlappen -- die umsortierung innerhalb der gruppe ist
    # damit unsichtbar, spart aber gut ein drittel der draw-calls. Ueber
    # gruppengrenzen hinweg bleibt die zeichenreihenfolge des mockups
    # unangetastet.
    hull_ops = []
    group_key = None
    fills, strokes = [], []

    def flush_group():
        if group_key is None:
            return
        fill, stroke, width, alpha = group_key
        if fill is not None:
            emit(hull_ops, "tris", resolve(fill) + (alpha,), 1.0, fills)
        if stroke is not None:
            emit(hull_ops, "lines", resolve(stroke) + (alpha,), width, strokes)

    for points, closed, fill, stroke, width, alpha in _shapes():
        key = (fill, stroke, width, alpha)
        if key != group_key:
            flush_group()
            group_key = key
            fills, strokes = [], []
        local = _to_local(points)
        if fill is not None:
            for tri in _triangulate(local):
                fills.extend(tri)
        if stroke is not None:
            ring = local + local[:1] if closed else local
            for a, b in zip(ring, ring[1:]):
                strokes.append(a)
                strokes.append(b)
    flush_group()

    arr = np.ascontiguousarray(
        np.asarray(verts, dtype=np.float64).reshape((-1, 2)))
    return ShipGeometry(arr, tuple(hull_ops), tuple(plume_ops), accent)
