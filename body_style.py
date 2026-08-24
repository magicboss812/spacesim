# -*- coding: utf-8 -*-
"""Prozedurale vektor-optik der himmelskoerper.

Portierung des canvas-mockups (Claude-Design-projekt, ``Planet Mockup.dc.html``)
auf numpy + moderngl. Erzeugt aus EINEM ganzzahligen seed die komplette
zeichnung eines koerpers: eine icosphere, ein daraus abgeleitetes rausch-feld,
dessen quantisierung in stufen ("tiles"), die konturlinien zwischen den stufen,
innen-figuren auf den hellsten tiles und ein paar grosskreis-ringe.

Drei entwurfsentscheidungen, die den rest erklaeren:

* **Es bleibt vektorgeometrie, es wird nie eine textur.** Das ergebnis sind
  dreiecke und liniensegmente im EINHEITSKREIS (x, y in [-1, 1], y nach oben).
  Der renderer skaliert sie pro frame auf den bildschirmradius. Damit ist die
  zeichnung bei jeder zoomstufe scharf, und ein koerper kostet unabhaengig von
  seiner bildschirmgroesse dieselben vertices.
* **Die koerper drehen sich nicht.** Die zufaellige orientierung wird EINMAL
  beim bauen in die vertices gerechnet. Dadurch ist die rueckseite statisch
  bekannt und kann schon hier weggeworfen werden -- das halbiert die geometrie
  und macht die z-tests zur bauzeit statt pro frame.
* **Beleuchtung bleibt dynamisch.** Jeder vertex traegt die 3D-normale seiner
  facette; der lichtvektor ist ein uniform. Deshalb wandert der terminator
  korrekt mit der bahn, ohne dass ein einziger vertex neu gerechnet wird.

Die farbe kommt vollstaendig aus ``body.color``: hue und saettigung des
koerpers ersetzen den festen hue des mockups.
"""

from __future__ import annotations

import colorsys
import math

import numpy as np

TAU = 2.0 * math.pi
_MASK32 = 0xFFFFFFFF

#: Feld-modi (bestimmen, WIE die tiles ueber die kugel verteilt sind).
MODES = ('bands', 'plates', 'ridges', 'cells', 'shards')
#: Figuren auf den hellsten tiles (bestimmen, WAS in einem tile steht).
SHAPES = ('nested', 'inset', 'medial', 'dot', 'diamond')

#: Wunsch-standard: "bands - nested".
DEFAULT_MODE = 'bands'
DEFAULT_SHAPE = 'nested'

#: Unterteilungsstufe der icosphere je detailstufe -> 80 / 320 / 1280 facetten.
_SUBDIV = {'coarse': 1, 'medium': 2, 'fine': 3}

#: Die stufen, aufsteigend. Der renderer waehlt sie nach bildschirmgroesse.
DETAIL_LEVELS = ('coarse', 'medium', 'fine')

#: Gemessene facettenbreite als vielfaches des bildschirmradius, je stufe.
#: (Abgezaehlt an abzuegen bei 260 px: 6.5 / 13 / 26 facetten ueber den
#: durchmesser.) Daraus leitet der renderer ab, welche stufe bei welcher
#: groesse etwa gleich grosse facetten IN PIXELN ergibt -- die einzige
#: definition von "detail", die beim zoomen stabil bleibt.
FACET_FRACTION = {'coarse': 0.308, 'medium': 0.154, 'fine': 0.077}

#: Bezugsradius (px), gegen den die flaechen-schwelle geeicht ist. Figuren
#: unter dieser flaeche werden verworfen -- sie waeren am rand, wo die
#: verkuerzung die facetten zusammendrueckt, nur noch ein fleck.
_AREA_REFERENCE_PX = 260.0
_AREA_MIN_PX2 = 12.0

#: Anteil der hellen tiles, der eine figur traegt. Das mockup wuerfelt das je
#: planet zwischen 0.24 und 0.60 aus; das entwurfsbild, an dem die optik
#: abgenommen wurde, zeigt dagegen auf FAST jedem hellen tile eine figur --
#: das ist der eigentliche eindruck, und die abwechslung zwischen den koerpern
#: kommt ohnehin aus der verteilung der tiles, nicht aus deren fuellung.
DEFAULT_SHAPE_DENSITY = 0.9

#: Spaltenlayout der beiden ausgabe-arrays. Absichtlich flache float32-tabellen:
#: sie gehen ohne umbau in einen GL-buffer.
TRI_COLUMNS = 10   # px, py, nx, ny, nz, r, g, b, alpha, dark
SEG_COLUMNS = 13   # ax, ay, bx, by, nx, ny, nz, r, g, b, alpha, dark, width
VERT_COLUMNS = 15  # px, py, nx, ny, nz, r, g, b, alpha, dark, dx, dy, side, ext, half


# --------------------------------------------------------------------------
# Bit-genaue portierung des JS-rauschens
# --------------------------------------------------------------------------
# Die formeln stammen 1:1 aus dem mockup. Javascript rechnet in `Math.imul`,
# also in 32-bit-zweierkomplement; hier wird durchgehend unsigned maskiert,
# was dieselben bitmuster ergibt. Weicht man davon ab, sieht das ergebnis
# zwar immer noch nach planet aus, aber nicht mehr nach DIESEM planeten.

def _imul(a, b):
    return (int(a) * int(b)) & _MASK32


class _Rng:
    """Der `rng`-generator des mockups (xorshift-multiply, 32 bit)."""

    __slots__ = ('s',)

    def __init__(self, seed):
        self.s = (int(seed) & _MASK32) or 1

    def __call__(self):
        s = _imul(self.s ^ (self.s >> 15), 2246822507)
        s = _imul(s ^ (s >> 13), 3266489909)
        s = (s ^ (s >> 16)) & _MASK32
        self.s = s
        return s / 4294967296.0


def _u64(value):
    return np.asarray(value, dtype=np.int64).astype(np.uint64) & np.uint64(_MASK32)


def _hash3(x, y, z, seed):
    """`hash3` des mockups, vektorisiert. Liefert werte in [0, 1)."""
    with np.errstate(over='ignore'):
        h = (
            (_u64(x) * np.uint64(374761393)) & np.uint64(_MASK32)
        ) ^ (
            (_u64(y) * np.uint64(668265263)) & np.uint64(_MASK32)
        ) ^ (
            (_u64(z) * np.uint64(2147483647)) & np.uint64(_MASK32)
        ) ^ (
            (_u64(seed) * np.uint64(1274126177)) & np.uint64(_MASK32)
        )
        h = ((h ^ (h >> np.uint64(13))) * np.uint64(1274126177)) & np.uint64(_MASK32)
        h = h ^ (h >> np.uint64(16))
    return (h & np.uint64(_MASK32)).astype(np.float64) / 4294967296.0


def _smooth(t):
    return t * t * (3.0 - 2.0 * t)


def _noise3(x, y, z, seed):
    xi = np.floor(x)
    yi = np.floor(y)
    zi = np.floor(z)
    xf = _smooth(x - xi)
    yf = _smooth(y - yi)
    zf = _smooth(z - zi)
    xi = xi.astype(np.int64)
    yi = yi.astype(np.int64)
    zi = zi.astype(np.int64)

    acc = np.zeros(np.shape(x), dtype=np.float64)
    for dz in (0, 1):
        wz = zf if dz else (1.0 - zf)
        for dy in (0, 1):
            wy = yf if dy else (1.0 - yf)
            for dx in (0, 1):
                wx = xf if dx else (1.0 - xf)
                acc += wx * wy * wz * _hash3(xi + dx, yi + dy, zi + dz, seed)
    return acc


def _fbm(x, y, z, seed, octaves):
    total = np.zeros(np.shape(x), dtype=np.float64)
    amp = 0.5
    freq = 1.0
    norm = 0.0
    for i in range(octaves):
        total += amp * _noise3(x * freq, y * freq, z * freq, seed + i * 7919)
        norm += amp
        amp *= 0.5
        freq *= 2.07
    return total / norm


# --------------------------------------------------------------------------
# Geometrie
# --------------------------------------------------------------------------

def icosphere(subdiv):
    """Icosaeder, `subdiv` mal unterteilt und auf die einheitskugel projiziert."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    raw = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
           (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
           (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    verts = []
    for vx, vy, vz in raw:
        length = math.sqrt(vx * vx + vy * vy + vz * vz)
        verts.append([vx / length, vy / length, vz / length])

    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]

    for _ in range(int(subdiv)):
        cache = {}
        nxt = []

        def midpoint(a, b):
            key = (a, b) if a < b else (b, a)
            hit = cache.get(key)
            if hit is not None:
                return hit
            va, vb = verts[a], verts[b]
            mx, my, mz = va[0] + vb[0], va[1] + vb[1], va[2] + vb[2]
            length = math.sqrt(mx * mx + my * my + mz * mz)
            verts.append([mx / length, my / length, mz / length])
            cache[key] = len(verts) - 1
            return len(verts) - 1

        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            nxt.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
        faces = nxt

    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _midpoint_normal(a, b):
    """Normierte kugel-normale in der mitte einer kante.

    Bewusst `math.sqrt` statt `np.linalg.norm`: die kante ist ein 3-tupel, und
    numpy kostet auf so kleinen daten mehr overhead als rechnung -- gemessen
    war das der groesste einzelposten beim bauen.
    """
    nx = a[0] + b[0]
    ny = a[1] + b[1]
    nz = a[2] + b[2]
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length <= 1e-12:
        return 0.0, 0.0, 1.0
    return nx / length, ny / length, nz / length


def _euler_matrix(a, b, c):
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cc, sc = math.cos(c), math.sin(c)
    return np.asarray([
        [ca * cb * cc - sa * sc, -ca * cb * sc - sa * cc, ca * sb],
        [sa * cb * cc + ca * sc, -sa * cb * sc + ca * cc, sa * sb],
        [-sb * cc, sb * sc, cb],
    ], dtype=np.float64)


# --------------------------------------------------------------------------
# Farbe
# --------------------------------------------------------------------------

def _color_basis(rgb):
    """`body.color` -> (hue, saettigungs-faktor).

    Der faktor floort graue koerper: Mond & Co. haben s ~ 0.05 und wuerden
    sonst als reines grau in reines grau zeichnen, also unsichtbar. Nach oben
    bleibt er bei 1.0, damit ein saftiger planet die mockup-werte unveraendert
    behaelt.
    """
    r, g, b = (max(0.0, min(1.0, float(c) / 255.0)) for c in rgb[:3])
    hue, _light, sat = colorsys.rgb_to_hls(r, g, b)
    return hue, max(0.30, min(1.0, sat * 1.5))


def _hsl(hue, sat_factor, hue_offset_deg, sat_pct, light_pct):
    h = (hue + hue_offset_deg / 360.0) % 1.0
    s = max(0.0, min(1.0, (sat_pct / 100.0) * sat_factor))
    l = max(0.0, min(1.0, light_pct / 100.0))
    return colorsys.hls_to_rgb(h, l, s)


# --------------------------------------------------------------------------
# Ergebnis
# --------------------------------------------------------------------------

class PlanetStyle(object):
    """Fertige zeichnung eines koerpers im einheitskreis.

    `tri` sind die flaechen (stufen-fuellungen und figuren-fuellungen),
    `seg` die liniensegmente. `under_segments` trennt die segmente, die UNTER
    den fuellungen liegen (das gitternetz), von denen darueber (figuren-
    umrisse, konturen, ringe) -- die zeichenreihenfolge des mockups, und
    weil alphas sich addieren ist sie nicht beliebig.
    """

    __slots__ = ('tri', 'seg', 'under_segments', 'seed', 'mode', 'shape',
                 'accent_shape', 'tiers', 'subdiv')

    def __init__(self, tri, seg, under_segments, seed, mode, shape,
                 accent_shape, tiers, subdiv):
        self.tri = tri
        self.seg = seg
        self.under_segments = int(under_segments)
        self.seed = int(seed)
        self.mode = mode
        self.shape = shape
        self.accent_shape = accent_shape
        self.tiers = int(tiers)
        self.subdiv = int(subdiv)

    @property
    def triangle_count(self):
        return int(self.tri.shape[0] // 3)

    @property
    def segment_count(self):
        return int(self.seg.shape[0])

    def __repr__(self):
        return (f"PlanetStyle(seed={self.seed}, mode={self.mode!r}, "
                f"shape={self.shape!r}, tris={self.triangle_count}, "
                f"segs={self.segment_count})")


def seed_from_name(name):
    """Stabiler 32-bit-seed aus dem koerpernamen.

    Damit bekommt jeder koerper ohne zutun ein eigenes, ueber laeufe hinweg
    gleiches muster -- und `style_seed` in der JSON bleibt eine reine
    korrektur-moeglichkeit statt einer pflichtangabe.
    """
    h = 2166136261
    for ch in str(name).encode('utf-8'):
        h = _imul(h ^ ch, 16777619)
    return h & _MASK32


# --------------------------------------------------------------------------
# Bau
# --------------------------------------------------------------------------

def build_planet_style(seed, color=(255, 255, 255), mode=None, shape=None,
                       accent_shape=None, coverage=0.5, detail='medium',
                       accent_fraction=0.14, shape_density=None):
    """Baut die vektor-zeichnung eines koerpers.

    `mode` / `shape` / `accent_shape` = None heisst "aus dem seed waehlen".
    Der aufrufer setzt sie auf 'bands' / 'nested', was der gewuenschte
    standard ist; die zufallszahlen werden trotzdem gezogen, damit alle
    uebrigen parameter unabhaengig von dieser wahl gleich bleiben.
    """
    seed = int(seed) & _MASK32
    rnd = _Rng((seed * 2654435761 + 12345) & _MASK32)
    subdiv = _SUBDIV.get(str(detail), 2)
    verts, faces = icosphere(subdiv)
    s = seed & 0xffff

    picked_mode = MODES[int(rnd() * len(MODES))]
    picked_shape = SHAPES[int(rnd() * len(SHAPES))]
    mode = picked_mode if mode is None else str(mode)
    shape = picked_shape if shape is None else str(shape)
    if accent_shape is None:
        accent_shape = SHAPES[int(_hash3(s, 17, 91, seed + 6151) * len(SHAPES))]

    shape_gate = 0.3 + rnd() * 0.45
    if shape_density is None:
        shape_density = DEFAULT_SHAPE_DENSITY
    shape_gate = max(0.0, min(1.0, float(shape_density)))
    tiers = 2 + int(rnd() * 2)
    warp = 0.3 + rnd() * 0.7
    freq = 0.7 + rnd() * 0.9
    band_f = 1.6 + rnd() * 2.2
    scatter = 0.35 + rnd() * 0.5
    rnd()  # markAngle -- im mockup ungenutzt, hier nur mitgezogen
    coverage = float(coverage)

    rot = _euler_matrix(rnd() * TAU, math.acos(2.0 * rnd() - 1.0), rnd() * TAU)

    # ---- facetten-mittelpunkte (auf der kugel) -------------------------
    tri_verts = verts[faces]                      # (n, 3, 3)
    centers = tri_verts.mean(axis=1)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    cx, cy, cz = centers[:, 0], centers[:, 1], centers[:, 2]

    # ---- das feld ------------------------------------------------------
    if mode == 'bands':
        values = _fbm(cx * 0.45 * freq + 4.0, cy * band_f, cz * 0.45 * freq, s, 3)
    elif mode == 'ridges':
        values = 1.0 - np.abs(2.0 * _fbm(cx * freq * 1.8 + 9.0, cy * freq * 1.8,
                                         cz * freq * 1.8, s, 4) - 1.0)
    elif mode == 'shards':
        w = _fbm(cx * warp, cy * warp, cz * warp, s + 31, 2)
        values = _fbm(cx * freq * 2.4 + w * 2.0, cy * freq * 2.4,
                      cz * freq * 2.4 - w * 2.0, s, 2)
    elif mode == 'cells':
        a = _fbm(cx * freq * 2.1, cy * freq * 2.1, cz * freq * 2.1, s, 2)
        b = _fbm(cx * freq * 2.1 + 13.0, cy * freq * 2.1 + 7.0,
                 cz * freq * 2.1 + 5.0, s + 991, 2)
        values = np.abs(a - b) * 1.7
    else:  # 'plates'
        w = _fbm(cx * 1.6, cy * 1.6, cz * 1.6, s + 77, 2) - 0.5
        values = _fbm(cx * freq + w * warp, cy * freq + w * warp, cz * freq, s, 4)

    # Streuung pro facette: verteilt die belegten tiles, statt weiche flecken
    # zu lassen. Danach zweimal S-kurve -- die werte sammeln sich an den
    # extremen, die mittlere stufe bleibt ein duennes uebergangsband.
    values = values + (_hash3(faces[:, 0], faces[:, 1], faces[:, 2],
                              s + 5077) - 0.5) * scatter
    values = np.clip((values - 0.5) * 3.2 + coverage, 0.0, 1.0)
    values = _smooth(_smooth(values))
    steps = max(1, tiers - 1)
    values = np.round(values * steps) / steps

    # ---- kanten & nachbarn --------------------------------------------
    edge_map = {}
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_map.setdefault(key, []).append(fi)

    neighbours = [[] for _ in range(len(faces))]
    for adj in edge_map.values():
        if len(adj) == 2:
            neighbours[adj[0]].append(adj[1])
            neighbours[adj[1]].append(adj[0])

    # Ein glaettungsdurchgang: macht aus sprenkeln zusammenhaengende gebiete,
    # ohne die streuung wieder einzuebnen.
    src = values.copy()
    for f in range(len(values)):
        total = src[f] * 2.2
        count = 2.2
        for g in neighbours[f]:
            total += src[g]
            count += 1.0
        values[f] = round((total / count) * steps) / steps

    # Die meisten mittleren tiles in volle aufloesen; nur wenige ueberleben
    # als uebergang.
    mid = (values > 0.001) & (values < 0.999)
    if np.any(mid):
        hh = _hash3(faces[mid, 0], faces[mid, 1], faces[mid, 2], s + 9001)
        resolved = values[mid].copy()
        keep = hh < 0.18
        resolved[~keep] = np.where(hh[~keep] < 0.78, 1.0, 0.0)
        values[mid] = resolved

    # ---- orientierung einrechnen (danach steht der koerper fest) -------
    verts = verts @ rot.T
    centers = centers @ rot.T
    cz = centers[:, 2]

    hue, sat_factor = _color_basis(color)

    tri_rows = []
    seg_rows = []

    # ---- 1. gitternetz -------------------------------------------------
    # Eindeutige kanten statt drei kanten pro facette: gleiche optik, halbe
    # geometrie, und geteilte kanten bekommen nicht doppeltes alpha.
    col_far = _hsl(hue, sat_factor, 0.0, 55.0, 68.0)
    col_near = _hsl(hue, sat_factor, 0.0, 50.0, 70.0)
    vlist = verts.tolist()
    czlist = cz.tolist()
    for (ia, ib), adj in edge_map.items():
        depth = max(czlist[f] for f in adj)
        va, vb = vlist[ia], vlist[ib]
        nx, ny, nz = _midpoint_normal(va, vb)
        if depth > 0.02:
            seg_rows.append((va[0], va[1], vb[0], vb[1], nx, ny, nz,
                             col_near[0], col_near[1], col_near[2],
                             0.18, 1.0, 0.6))
        else:
            seg_rows.append((va[0], va[1], vb[0], vb[1], nx, ny, nz,
                             col_far[0], col_far[1], col_far[2],
                             0.05, 1.0, 0.6))
    under_segments = len(seg_rows)

    # ---- 2. stufen-fuellungen -----------------------------------------
    # Jede beleuchtete facette bekommt etwas farbe, damit niedrige stufen als
    # gelaende lesen und nicht als loch. Die helligkeit steckt NICHT in der
    # farbe, sondern im alpha -- deshalb kann der shader sie pro frame aus
    # der normale rechnen (`dark` = anteil, der auf der nachtseite bleibt).
    clist = centers.tolist()
    flist = faces.tolist()
    vals = values.tolist()
    for f in np.nonzero(cz > 0.02)[0].tolist():
        v = vals[f]
        col = _hsl(hue, sat_factor, 8.0, 44.0 + v * 18.0, 20.0 + v * 22.0)
        alpha = (0.22 + v * 0.3) * 1.31
        nx, ny, nz = clist[f]
        for idx in flist[f]:
            p = vlist[idx]
            tri_rows.append((p[0], p[1], nx, ny, nz,
                             col[0], col[1], col[2], alpha, 0.06 / 1.31))

    # ---- 3. figuren auf den hellsten tiles -----------------------------
    # Immer zum schwerpunkt hin eingerueckt, damit sie im dreieck bleiben.
    # Der einzug ist affin, also identisch, ob er hier im einheitskreis oder
    # spaeter in bildschirmpixeln gerechnet wird.
    shape_stroke = _hsl(hue, sat_factor, 6.0, 86.0, 82.0)
    area_min = _AREA_MIN_PX2 / (_AREA_REFERENCE_PX * _AREA_REFERENCE_PX)
    accent_fraction = max(0.0, min(1.0, float(accent_fraction)))
    gate_hash = _hash3(faces[:, 0], faces[:, 1], faces[:, 2], s + 4242).tolist()
    accent_hash = _hash3(faces[:, 0], faces[:, 1], faces[:, 2], s + 7717).tolist()
    for f in np.nonzero((cz > 0.06) & (values >= 0.99))[0].tolist():
        v = vals[f]
        if gate_hash[f] > shape_gate:
            continue
        ia, ib, ic = flist[f]
        pa, pb, pc = vlist[ia], vlist[ib], vlist[ic]
        area = abs((pb[0] - pa[0]) * (pc[1] - pa[1])
                   - (pc[0] - pa[0]) * (pb[1] - pa[1])) * 0.5
        if area < area_min:
            continue
        this_shape = shape
        if accent_fraction > 0.0 and accent_hash[f] < accent_fraction:
            this_shape = accent_shape
        n = clist[f]
        fill = _hsl(hue, sat_factor, 10.0, 62.0, 46.0 + v * 16.0)
        fill_alpha = (0.16 + v * 0.2) * 1.3
        line_alpha = (0.34 + v * 0.32) * 1.3
        _emit_shape(this_shape, pa, pb, pc, n, fill, fill_alpha,
                    shape_stroke, line_alpha, area, tri_rows, seg_rows)

    # ---- 4. konturen zwischen den stufen -- der eigentliche eindruck ---
    col_contour = _hsl(hue, sat_factor, 4.0, 88.0, 80.0)
    for (ia, ib), adj in edge_map.items():
        if len(adj) != 2:
            continue
        delta = abs(vals[adj[0]] - vals[adj[1]])
        if delta < 0.001:
            continue
        va, vb = vlist[ia], vlist[ib]
        if (va[2] + vb[2]) * 0.5 <= 0.03:
            continue
        strength = min(1.0, delta * 1.6)
        nx, ny, nz = _midpoint_normal(va, vb)
        seg_rows.append((va[0], va[1], vb[0], vb[1], nx, ny, nz,
                         col_contour[0], col_contour[1], col_contour[2],
                         (0.3 + strength * 0.4) * 1.19, 0.14 / 1.19, 2.0))

    # ---- 5. grosskreis-ringe ------------------------------------------
    col_ring = _hsl(hue, sat_factor, 0.0, 72.0, 80.0)
    for _ in range(1 + int(rnd() * 3)):
        theta = rnd() * TAU
        phi = math.acos(2.0 * rnd() - 1.0)
        axis = np.asarray([math.sin(phi) * math.cos(theta),
                           math.cos(phi),
                           math.sin(phi) * math.sin(theta)], dtype=np.float64)
        axis = rot @ axis
        _emit_ring(axis, col_ring, seg_rows)

    tri = (np.asarray(tri_rows, dtype=np.float32)
           if tri_rows else np.zeros((0, TRI_COLUMNS), dtype=np.float32))
    seg = (np.asarray(seg_rows, dtype=np.float32)
           if seg_rows else np.zeros((0, SEG_COLUMNS), dtype=np.float32))
    return PlanetStyle(tri, seg, under_segments, seed, mode, shape,
                       accent_shape, tiers, subdiv)


def _emit_shape(shape, pa, pb, pc, n, fill, fill_alpha, stroke, line_alpha,
                area, tri_rows, seg_rows):
    g = ((pa[0] + pb[0] + pc[0]) / 3.0, (pa[1] + pb[1] + pc[1]) / 3.0)

    def inset(p, k):
        return (g[0] + (p[0] - g[0]) * k, g[1] + (p[1] - g[1]) * k)

    def mid(p, q):
        return ((p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5)

    def add_fill(points):
        # Faecher ab dem ersten punkt -- alle figuren hier sind konvex.
        for i in range(1, len(points) - 1):
            for p in (points[0], points[i], points[i + 1]):
                tri_rows.append((p[0], p[1], n[0], n[1], n[2],
                                 fill[0], fill[1], fill[2], fill_alpha, 0.0))

    def add_outline(points, width=0.9):
        for i in range(len(points)):
            p = points[i]
            q = points[(i + 1) % len(points)]
            seg_rows.append((p[0], p[1], q[0], q[1], n[0], n[1], n[2],
                             stroke[0], stroke[1], stroke[2],
                             line_alpha, 0.0, width))

    if shape == 'dot':
        # Kreis als polygon: die figur bleibt vektor, nur eben mit 12 ecken.
        #
        # Begrenzt wird er vom INKREIS, nicht von der halben strecke zum
        # eckpunkt. Am rand der scheibe sind die facetten stark verkuerzt;
        # die halbe eckstrecke lag dort ausserhalb des dreiecks und damit
        # ausserhalb des einheitskreises -- gemessen 1.0045, also 0.45 %
        # ueber den rand, wo der koerper eigentlich schon zu ende ist.
        side_a = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        side_b = math.hypot(pc[0] - pb[0], pc[1] - pb[1])
        side_c = math.hypot(pa[0] - pc[0], pa[1] - pc[1])
        inradius = 2.0 * area / max(1e-12, side_a + side_b + side_c)
        radius = min(math.sqrt(max(area, 0.0)) * 0.48, inradius * 0.9)
        ring = [(g[0] + radius * math.cos(i / 12.0 * TAU),
                 g[1] + radius * math.sin(i / 12.0 * TAU)) for i in range(12)]
        add_fill(ring)
        add_outline(ring)
    elif shape == 'medial':
        pts = [mid(pa, pb), mid(pb, pc), mid(pc, pa)]
        add_fill(pts)
        add_outline(pts)
    elif shape == 'nested':
        outer = [inset(pa, 0.7), inset(pb, 0.7), inset(pc, 0.7)]
        add_fill(outer)
        add_outline(outer)
        add_outline([inset(pa, 0.34), inset(pb, 0.34), inset(pc, 0.34)])
    elif shape == 'diamond':
        pts = [mid(pa, pb), inset(pb, 0.72), mid(pb, pc), mid(pc, pa)]
        add_fill(pts)
        add_outline(pts)
    else:  # 'inset'
        pts = [inset(pa, 0.56), inset(pb, 0.56), inset(pc, 0.56)]
        add_fill(pts)
        add_outline(pts)


def _emit_ring(axis, color, seg_rows, samples=160):
    ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
    helper = (1.0, 0.0, 0.0) if abs(ax) < 0.9 else (0.0, 1.0, 0.0)
    u = np.asarray([helper[1] * az - helper[2] * ay,
                    helper[2] * ax - helper[0] * az,
                    helper[0] * ay - helper[1] * ax], dtype=np.float64)
    u /= max(1e-12, float(np.linalg.norm(u)))
    v = np.asarray([ay * u[2] - az * u[1],
                    az * u[0] - ax * u[2],
                    ax * u[1] - ay * u[0]], dtype=np.float64)

    t = np.linspace(0.0, TAU, samples + 1)
    pts = (u[None, :] * np.cos(t)[:, None]) + (v[None, :] * np.sin(t)[:, None])
    visible = (pts[:, 2] > 0.02).tolist()
    plist = pts.tolist()
    for i in range(samples):
        if not (visible[i] and visible[i + 1]):
            continue
        a, b = plist[i], plist[i + 1]
        nx, ny, nz = _midpoint_normal(a, b)
        seg_rows.append((a[0], a[1], b[0], b[1], nx, ny, nz,
                         color[0], color[1], color[2], 0.16, 1.0, 0.9))


# --------------------------------------------------------------------------
# GL-vertexdaten
# --------------------------------------------------------------------------

def expand_segments(seg):
    """Liniensegmente -> quads (6 vertices je segment).

    Die breite steht in PIXELN, nicht in weltmass: der vertex-shader dreht die
    normale erst nach der skalierung auf, deshalb bleibt eine linie bei jeder
    zoomstufe gleich dick. Das halbe mass ist um 0.5 px aufgeweitet -- dieser
    saum ist die kantenglaettung im fragment-shader.
    """
    seg = np.asarray(seg, dtype=np.float32)
    if seg.size == 0:
        return np.zeros((0, VERT_COLUMNS), dtype=np.float32)

    ax, ay = seg[:, 0], seg[:, 1]
    bx, by = seg[:, 2], seg[:, 3]
    dx, dy = bx - ax, by - ay
    length = np.hypot(dx, dy)
    keep = length > 1e-9
    if not np.any(keep):
        return np.zeros((0, VERT_COLUMNS), dtype=np.float32)
    seg = seg[keep]
    ax, ay, bx, by = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
    dx, dy = bx - ax, by - ay
    length = np.hypot(dx, dy)
    dx = dx / length
    dy = dy / length

    n = seg.shape[0]
    half = seg[:, 12] * 0.5 + 0.5

    # v0 = a(-1), v1 = a(+1), v2 = b(+1), v3 = b(-1); zwei dreiecke 0-1-2, 0-2-3
    corner_pos = np.asarray([0, 0, 1, 0, 1, 1], dtype=np.int8)      # 0 = a, 1 = b
    corner_side = np.asarray([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    corner_ext = np.asarray([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float32)

    out = np.empty((n * 6, VERT_COLUMNS), dtype=np.float32)
    for k in range(6):
        rows = out[k::6]
        at_b = bool(corner_pos[k])
        rows[:, 0] = bx if at_b else ax
        rows[:, 1] = by if at_b else ay
        rows[:, 2:10] = seg[:, 4:12]      # normale, farbe, alpha, dark
        rows[:, 10] = dx
        rows[:, 11] = dy
        rows[:, 12] = corner_side[k]
        rows[:, 13] = corner_ext[k]
        rows[:, 14] = half
    return out
