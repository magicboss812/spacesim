"""Die bauformen des schiffs und ihre farben.

Jede bauform beschreibt ihre formen mit FARBROLLEN statt hex-werten
(`"hull"`, `"cargo_a:2"` = rolle cargo_a, schattierung 2). Die farbe jeder
rolle steht in `renderer.ship_build_colors` (config `ship_build_colors`,
dev-UI "Ship"), mit den vorgaben aus `DEFAULT_COLORS`. `resolve()` macht
daraus die formliste, die `ship/art.py` in vektor- oder zellgeometrie
uebersetzt.

Schattierungen: grosse flaechen sind in platten zerlegt, jede platte nimmt
eine schattierung ihrer rolle (heller/dunkler, leicht im farbton gedreht).
So traegt eine rolle eine ganze familie naher toene, und der spieler stellt
trotzdem nur EINE farbe je rolle ein.

Lokale koordinaten (+x nase, +y oben), 230 einheiten lang (nase +115, duese
-115) wie das original -- `_ship_length_px` normiert darauf. Alle duesen
liegen bei x = -98 .. -115: die abgasfahne (`ship_pixel.frag`, `art._plume_
polygons`) sitzt fest dahinter.

Rein numerisch -- kein pygame, kein GL.
"""
import colorsys
import math

from ship import art

BUILDS = ("original", "ringschiff", "pfeil", "schlepper", "segler")

DEFAULT_COLORS = {
    "original": {
        "panel": "#1c2135", "hull": "#2f3550", "module": "#3d4463",
        "module_accent": "#454d70", "nose": "#3a4160", "bell": "#2b3149",
        "mount": "#282e46", "line": "#e9e9ed", "detail": "#8f97ab",
        "rail": "#7d8496", "accent": "#E71D36",
    },
    "ringschiff": {
        "hull": "#3a4a7a", "ring": "#3f4d78", "window": "#3a94a0",
        "hub": "#30808f", "cargo_a": "#a07a34", "cargo_b": "#8e4a38",
        "cargo_c": "#6e8a44", "steel": "#465a6e", "nose": "#c8403e",
        "heat": "#ec7a44", "bell": "#2b3149", "line": "#e9e9ed",
        "rail": "#7d8496", "accent": "#E71D36",
    },
    "pfeil": {
        "wing": "#42479a", "hull": "#3a4a7a", "livery": "#e0583e",
        "canopy": "#3a94a0", "steel": "#465a6e", "nose": "#c8403e",
        "bell": "#2b3149", "line": "#e9e9ed", "rail": "#7d8496",
        "accent": "#E71D36",
    },
    "schlepper": {
        "tank": "#b08636", "hazard": "#c49a44", "hazard_dark": "#a8343a",
        "core": "#465a6e", "module": "#30808f", "steel": "#3a4a5c",
        "grip": "#e0583e", "bell": "#2b3149", "line": "#e9e9ed",
        "rail": "#7d8496", "accent": "#E71D36",
    },
    "segler": {
        "sail": "#5a6e88", "boom": "#3e5064", "hull": "#3a4a7a",
        "nose": "#c8403e", "livery": "#e0583e", "radiator": "#2a6a7c",
        "steel": "#3a4a5c", "bell": "#2b3149", "line": "#e9e9ed",
        "rail": "#7d8496", "accent": "#E71D36",
    },
}

# Das original ist im SVG des mockups notiert und benutzt hex-werte; jede
# zaehlt zu einer rolle und wird mit ihr VERHAELTNISGLEICH umgefaerbt
# (farbe * neu / vorgabe), damit die vorgaben das mockup exakt treffen.
_ORIGINAL_ROLES = {
    "#1c2135": "panel", "#2f3550": "hull", "#3d4463": "module",
    "#454d70": "module_accent", "#3a4160": "nose", "#2b3149": "bell",
    "#282e46": "mount", "#7d8496": "rail",
    "#e9e9ed": "line", "#c9ccd6": "line", "#dcdfe8": "line",
    "#a8aec0": "line", "#b3b9c9": "line",
    "#78809a": "detail", "#454d6b": "detail", "#8f97ab": "detail",
    "#6b7288": "detail", "#606a86": "detail", "#5b6480": "detail",
    "#7b839a": "detail",
}


def colors_for(build, overrides=None):
    """Vorgaben der bauform, ueberschrieben mit gueltigen hex-werten aus
    `overrides` (unbekannte rollen und kaputte werte werden ignoriert)."""
    base = dict(DEFAULT_COLORS.get(build, DEFAULT_COLORS["original"]))
    for role, value in (overrides or {}).items():
        if role in base:
            try:
                art._hex_rgb(value)
            except (ValueError, TypeError):
                continue
            base[role] = str(value)
    return base


def shade(rgb, k):
    """Schattierung k einer farbe: je stufe ~9 % heller, ein hauch waermer
    im farbton, etwas weniger gesaettigt. k = 0 ist die farbe selbst."""
    if not k:
        return rgb
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    v = min(1.0, max(0.0, v * (1.0 + 0.09 * k)))
    s = min(1.0, max(0.0, s * (1.0 - 0.03 * k)))
    h = (h - 0.008 * k) % 1.0
    return colorsys.hsv_to_rgb(h, s, v)


def _kind(role):
    if role == "accent":
        return "accent"
    if role == "rail":
        return "rail"
    return "plain"


class _Spec:
    """Formen in rollen sammeln, im LOKALEN schiffsraum."""

    def __init__(self):
        self.shapes = []

    def fill(self, pts, role, stroke="line", width=1.2):
        self.shapes.append(([(float(x), float(y)) for x, y in pts], True, role, stroke, width))

    def line(self, pts, role, width=1.0, closed=False):
        self.shapes.append(([(float(x), float(y)) for x, y in pts], closed, None, role, width))

    def both(self, pts, role, stroke="line"):
        self.fill(pts, role, stroke)
        self.fill([(x, -y) for x, y in pts], role, stroke)

    def bell(self, x_in, x_out, y, r_in, r_out):
        self.fill([(x_in, y - r_in), (x_in, y + r_in), (x_out, y + r_out), (x_out, y - r_out)],
                  "bell", "line", 1.4)
        self.line([(x_out, y - r_out), (x_out, y + r_out)], "accent", 1.4)


def _rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _circle(cx, cy, r, n=20):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
            for i in range(n)]


def _plates(s, x0, y0, x1, y1, role, nx, ny, pattern):
    """Rechteck in nx x ny platten; `pattern[i]` ist die schattierung."""
    n = 0
    for i in range(nx):
        for j in range(ny):
            a = x0 + (x1 - x0) * i / nx
            b = x0 + (x1 - x0) * (i + 1) / nx
            c = y0 + (y1 - y0) * j / ny
            d = y0 + (y1 - y0) * (j + 1) / ny
            s.fill(_rect(a, c, b, d), f"{role}:{pattern[n % len(pattern)]}", None)
            n += 1


# --- 1 ringschiff: kiel, wohnring, containerreihen -------------------------

def _ringschiff():
    s = _Spec()
    cargo = ("cargo_a:0", "cargo_b:1", "cargo_c:0", "cargo_a:-2", "cargo_b:-1", "cargo_c:2")
    n = 0
    for x0 in (-78, -56, -34):
        for sgn in (1, -1):
            s.fill([(x0, sgn * 9), (x0 + 19, sgn * 9), (x0 + 19, sgn * 21), (x0, sgn * 21)], cargo[n])
            s.line([(x0 + 9.5, sgn * 9), (x0 + 9.5, sgn * 21)], "rail")
            n += 1
    _plates(s, -98, -6, 96, 6, "hull", 8, 1, (0, 2, -1, 1, -2, 0, 2, -1))
    s.line([(-98, -6), (96, -6), (96, 6), (-98, 6)], "line", 1.6, closed=True)
    cx, ro, ri, segs = 14.0, 55.0, 40.0, 36
    for i in range(segs):
        a0, a1 = 2 * math.pi * i / segs, 2 * math.pi * (i + 1) / segs
        seg = [(cx + ri * math.cos(a0), ri * math.sin(a0)), (cx + ro * math.cos(a0), ro * math.sin(a0)),
               (cx + ro * math.cos(a1), ro * math.sin(a1)), (cx + ri * math.cos(a1), ri * math.sin(a1))]
        role = f"window:{(i // 2) % 4 - 1}" if i % 4 in (1, 2) else f"ring:{(i * 5) % 6 - 2}"
        s.fill(seg, role, None)
    s.line(_circle(cx, 0, ro, 36), "line", 1.2, closed=True)
    s.line(_circle(cx, 0, ri, 36), "line", 1.2, closed=True)
    for ang in (40, 140):
        a = math.radians(ang)
        c, sn, w = math.cos(a), math.sin(a), 5.0
        s.both([(cx + 12 * c - w * sn, 12 * sn + w * c), (cx + ri * c - w * sn, ri * sn + w * c),
                (cx + ri * c + w * sn, ri * sn - w * c), (cx + 12 * c + w * sn, 12 * sn - w * c)],
               "steel:1")
    s.fill(_circle(cx, 0, 13, 16), "hub")
    s.fill(_circle(cx, 0, 8, 12), "hub:2", None)
    s.line(_circle(cx, 0, 5, 10), "accent", 1.2, closed=True)
    for ang in (0, 90, 180, 270):
        a = math.radians(ang)
        s.line(_circle(cx + 47.5 * math.cos(a), 47.5 * math.sin(a), 1.6, 6), "accent", 1.2, closed=True)
    _plates(s, 72, -11, 96, 11, "steel", 2, 2, (0, 2, 1, -1))
    s.fill([(96, -9), (108, -6), (115, 0), (108, 6), (96, 9)], "nose")
    s.line([(100, 0), (112, 0)], "accent", 1.2)
    s.fill(_rect(-98, -20, -86, 20), "steel:-1")
    s.fill(_rect(-96, -4, -88, 4), "heat", None)
    for y in (-12, 0, 12):
        s.bell(-98, -115, y, 3.5, 6.0)
    return s.shapes


# --- 2 pfeil: deltaflueger, zwei triebwerke --------------------------------

def _pfeil():
    s = _Spec()
    s.both([(38, 11), (-48, 56), (-62, 60), (-74, 56), (-80, 44), (-70, 11)], "wing:-1")
    for k, (ya, yb) in enumerate(((11, 24), (24, 38), (38, 60))):
        for sgn in (1, -1):
            xa = 38 - (ya - 11) * (86 / 45)
            xb = 38 - (yb - 11) * (86 / 45)
            band = [(xa, sgn * ya), (xb, sgn * yb), (-74, sgn * min(yb, 56)),
                    (-70 if ya < 12 else -76, sgn * ya)]
            s.fill(band, f"wing:{k}", None)
    for sgn in (1, -1):
        s.fill([(8, sgn * 24), (-12, sgn * 35), (-22, sgn * 35), (-2, sgn * 24)], "livery", None)
        s.fill([(-18, sgn * 38), (-34, sgn * 47), (-44, sgn * 47), (-28, sgn * 38)], "livery:1", None)
        s.fill([(-62, sgn * 52), (-74, sgn * 56), (-80, sgn * 44), (-70, sgn * 44)], "livery:-2", None)
    for y in (26, 40):
        x_lead = 38 - (y - 11) * (86 / 45)
        s.line([(x_lead - 2, y), (-72, y)], "rail")
        s.line([(x_lead - 2, -y), (-72, -y)], "rail")
    s.line([(30, 16), (-40, 52)], "accent", 1.2)
    s.line([(30, -16), (-40, -52)], "accent", 1.2)
    body = [(115, 0), (92, 8), (40, 12), (-62, 12), (-88, 10), (-88, -10), (-62, -12), (40, -12), (92, -8)]
    s.fill(body, "hull", "line", 1.6)
    _plates(s, -60, -11, 40, 11, "hull", 5, 2, (0, 2, -1, 1, -2, 1, 0, 2, -1, 1))
    s.fill([(115, 0), (98, 6), (92, 0), (98, -6)], "nose", None)
    s.both([(74, 7), (58, 22), (50, 22), (54, 9)], "nose")
    s.fill([(90, 0), (80, 5), (60, 5), (54, 0), (60, -5), (80, -5)], "canopy")
    s.fill([(84, 0), (78, 3), (66, 3), (62, 0), (66, -3), (78, -3)], "canopy:2", None)
    s.line([(62, 0), (84, 0)], "accent", 1.2)
    s.fill(_rect(-10, -7, 30, 7), "steel:1")
    s.fill(_rect(-50, -7, -18, 7), "steel")
    s.line([(-34, -7), (-34, 7)], "rail")
    s.fill(_rect(-100, -16, -84, 16), "steel:-1")
    for y in (-8, 8):
        s.bell(-100, -114, y, 5.0, 7.5)
    return s.shapes


# --- 3 schlepper: achteckiger kern, greifer, aussentanks -------------------

def _schlepper():
    s = _Spec()
    for sgn in (1, -1):
        s.fill([(-30, sgn * 22), (-20, sgn * 22), (-20, sgn * 34), (-30, sgn * 34)], "steel:1", None)
        s.fill([(8, sgn * 22), (18, sgn * 22), (18, sgn * 34), (8, sgn * 34)], "steel:1", None)
        s.fill([(-72, sgn * 34), (30, sgn * 34), (36, sgn * 39), (36, sgn * 49), (30, sgn * 54),
                (-72, sgn * 54), (-78, sgn * 49), (-78, sgn * 39)], "tank")
        xs = (-72, -50, -26, -2, 20, 30)
        for i, (a, b) in enumerate(zip(xs, xs[1:])):
            s.fill([(a, sgn * 35), (b, sgn * 35), (b, sgn * 53), (a, sgn * 53)],
                   f"tank:{(-1, 1, 0, -2, 1)[i]}", None)
            s.fill([(a, sgn * 42), (b, sgn * 42), (b, sgn * 46), (a, sgn * 46)], "tank:2", None)
        for x in (-50, -26, -2, 20):
            s.line([(x, sgn * 34), (x, sgn * 54)], "rail")
        s.line(_circle(33, sgn * 44, 1.6, 6), "accent", 1.2, closed=True)
    for sgn in (1, -1):
        s.fill([(30, sgn * 4), (30, sgn * 16), (88, sgn * 26), (104, sgn * 30), (115, sgn * 22),
                (98, sgn * 12), (40, sgn * 4)], "core:1")
        for k in range(4):
            x = 46 + k * 13
            yl = 4 + (x - 40) * (8 / 58)
            yh = 16 + (x - 30) * (10 / 58)
            s.fill([(x, sgn * yl), (x + 6, sgn * (yl + 0.8)), (x + 10, sgn * (yh + 0.8)), (x + 4, sgn * yh)],
                   "hazard" if k % 2 == 0 else "hazard_dark", None)
        s.fill([(100, sgn * 9), (110, sgn * 9), (110, sgn * 15), (100, sgn * 15)], "grip")
        s.line([(103, sgn * 11), (108, sgn * 11)], "accent", 1.2)
    s.fill([(34, -12), (34, 12), (18, 28), (-8, 28), (-24, 12), (-24, -12), (-8, -28), (18, -28)],
           "core", "line", 1.7)
    s.fill([(18, 28), (-8, 28), (-8, 12), (18, 12)], "core:2", None)
    s.fill([(18, -28), (-8, -28), (-8, -12), (18, -12)], "core:-1", None)
    s.fill(_rect(-12, -11, 22, 11), "module")
    s.fill(_rect(-8, -6, 18, 6), "module:2", None)
    s.line([(5, -11), (5, 11)], "rail")
    s.line(_circle(26, 0, 3.5, 10), "accent", 1.2, closed=True)
    s.fill(_rect(-62, -15, -24, 15), "steel")
    s.fill(_rect(-58, -15, -48, 15), "hazard_dark", None)
    s.line([(-43, -15), (-43, 15)], "rail")
    s.fill([(-62, -10), (-62, 10), (-115, 22), (-115, -22)], "bell", "line", 1.6)
    s.fill([(-80, -14), (-80, 14), (-100, 18), (-100, -18)], "bell:2", None)
    s.line([(-115, -22), (-115, 22)], "accent", 1.6)
    return s.shapes


# --- 4 segler: rautensegel an kreuzspieren, ionenantrieb -------------------

def _segler():
    s = _Spec()
    # Das segel: eine raute (gedrehtes quadrat) um (20, 0), in 4 x 4 felder
    # zerlegt, jedes eine schattierung -- folie, die das licht ungleich faengt.
    c = (20.0, 0.0)
    e1, e2 = (29.0, 29.0), (29.0, -29.0)

    def at(u, v):
        return (c[0] + u * e1[0] + v * e2[0], c[1] + u * e1[1] + v * e2[1])

    steps = (-1.0, -0.5, 0.0, 0.5, 1.0)
    pattern = ((1, -1, 2, 0), (-1, 2, 0, 1), (2, 0, 1, -2), (0, 1, -2, 2))
    for i in range(4):
        for j in range(4):
            u0, u1, v0, v1 = steps[i], steps[i + 1], steps[j], steps[j + 1]
            s.fill([at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)], f"sail:{pattern[i][j]}", None)
    s.line([at(-1, -1), at(1, -1), at(1, 1), at(-1, 1)], "line", 1.4, closed=True)
    for t in (-0.5, 0.5):
        s.line([at(t, -1), at(t, 1)], "rail")
        s.line([at(-1, t), at(1, t)], "rail")
    # spieren: quer (y) und laengs (x, der kiel)
    s.fill(_rect(17, -60, 23, 60), "boom")
    s.fill(_rect(-100, -4, 104, 4), "hull", "line", 1.4)
    _plates(s, -96, -4, 16, 4, "hull", 4, 1, (0, 2, -1, 1))
    s.fill(_rect(40, -5, 50, 5), "livery", None)
    s.fill(_rect(56, -5, 62, 5), "livery:1", None)
    # kommandokapsel
    s.fill([(86, -8), (104, -8), (112, -4), (115, 0), (112, 4), (104, 8), (86, 8)], "nose")
    s.fill([(96, -4), (106, -4), (110, 0), (106, 4), (96, 4)], "nose:2", None)
    # radiatorflossen am heck
    for sgn in (1, -1):
        s.fill([(-80, sgn * 4), (-56, sgn * 4), (-60, sgn * 16), (-76, sgn * 16)], "radiator")
        s.line([(-68, sgn * 5), (-68, sgn * 15)], "rail")
    # positionslichter an den segelspitzen
    for p in (at(1, 1), at(-1, -1), at(1, -1), at(-1, 1)):
        s.line(_circle(p[0], p[1], 1.6, 6), "accent", 1.2, closed=True)
    # ionenantrieb
    s.fill(_rect(-100, -10, -86, 10), "steel")
    s.bell(-100, -115, 0, 5.0, 9.0)
    return s.shapes


_BUILDERS = {
    "ringschiff": _ringschiff,
    "pfeil": _pfeil,
    "schlepper": _schlepper,
    "segler": _segler,
}


def resolve(build, colors):
    """Formliste mit aufgeloesten farben:
    [(lokale_punkte, geschlossen, fuellung, kontur, breite, alpha)], wobei
    fuellung/kontur None oder ((r, g, b), art) sind; art in
    {"accent", "rail", "plain"}. Reihenfolge = zeichenreihenfolge."""
    rgb = {role: art._hex_rgb(value) for role, value in colors.items()}
    out = []
    if build not in _BUILDERS:
        defaults = DEFAULT_COLORS["original"]
        base = {role: art._hex_rgb(value) for role, value in defaults.items()}

        def orig(color):
            if color is None:
                return None
            if color == art.ACCENT:
                return (rgb["accent"], "accent")
            role = _ORIGINAL_ROLES.get(color.lower())
            own = art._hex_rgb(color)
            if role is None:
                return (own, "plain")
            scaled = tuple(min(1.0, o * n / max(d, 1e-6))
                           for o, n, d in zip(own, rgb[role], base[role]))
            return (scaled, _kind(role))

        for points, closed, fill, stroke, width, alpha in art._shapes():
            out.append((art._to_local(points), closed, orig(fill), orig(stroke), width, alpha))
        return out

    def spec(value):
        if value is None:
            return None
        role, _, k = value.partition(":")
        return (shade(rgb[role], int(k or 0)), _kind(role))

    for points, closed, fill, stroke, width in _BUILDERS[build]():
        out.append((points, closed, spec(fill), spec(stroke), width, 1.0))
    return out
