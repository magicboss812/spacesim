# -*- coding: utf-8 -*-
"""Die POSITIONS-MARKE eines koerpers -- das icon beim herauszoomen.

Nicht zu verwechseln mit `body_style.py`: das ist die oberflaechen-optik des
grossen koerpers. Hier geht es um die marke, die ihn vertritt, sobald er unter
`renderer.body_icon_min_radius_px` schrumpft. Frueher war das eine flache scheibe
in koerperfarbe -- bei systemzoom also 27 gleiche punkte.

Zwei varianten, beide aus EINEM 32-bit-seed:

* ``rosette``   -- voller kern, der aeussere ring wird gewuerfelt (entwurf A)
* ``signature`` -- kern plus 2-4 gesaete zacken auf dem ring (entwurf D)

> **Warum die zellen im ICON-raum liegen und nicht im bildschirmraum.**
> Der hintergrund rastet bewusst auf den schirm (`background.frag`:
> ``frag = (floor(gl_FragCoord.xy/px)+0.5)*px``, jede kante ``step``). Damit
> rueckt sein muster pixelweise, sobald sich etwas bewegt. Fuer eine marke, die
> einem koerper folgt, waere das genau die stockende bewegung, die hier nicht
> gewollt ist. Die zellen liegen deshalb im einheitskreis der marke; der shader
> rechnet sie aus der interpolierten LOKAL-koordinate aus, die an der
> gleitkomma-position der marke haengt. Das muster kann daher gar nicht ueber
> die marke wandern -- es IST die marke.

Ausgabe ist bewusst winzig: ein ``(N, N)``-feld aus stufen (0 = leer, 1..3) und
dessen bit-packung in **vier uint32**. Gezeichnet wird die marke als EIN quad;
welche zelle ein fragment trifft, entscheidet der shader. Es gibt deshalb keine
aneinanderstossenden primitive und damit auch keine naht, an der zwei
teildeckungen uebereinander liegen koennten.
"""

from __future__ import annotations

import math

import numpy as np

from bodies.style import _Rng, _imul, _MASK32, seed_from_name  # noqa: F401

#: Die beiden gewaehlten entwuerfe.
VARIANTS = ('rosette', 'signature')
DEFAULT_VARIANT = 'rosette'

#: Kantenlaenge des zellgitters -- die EINZIGE stellschraube fuer den
#: detailgrad. Sie haengt bewusst NICHT an der bildschirmgroesse: eine groessere
#: marke zeigt dasselbe muster groesser, nie ein feineres.
#:
#: Ungerade, damit es eine mittelzelle gibt -- die entwuerfe sind radial um
#: sie herum gebaut. 16 waere gerade und haette keine; 15 ist der groesste
#: ungerade wert darunter und zugleich das, was die packung traegt
#: (16 x uint32 = 256 zellen, 15x15 = 225).
DEFAULT_GRID = 9
MAX_GRID = 15
CELL_WORDS = 16

#: Stufen: 0 = leer, 1 = dunkel, 2 = grund, 3 = hell. Zwei bits je zelle.
TIER_EMPTY, TIER_DIM, TIER_BASE, TIER_BRIGHT = 0, 1, 2, 3

#: Deckkraft je stufe. Die dunklen zellen sind das, was der marke tiefe gibt --
#: ohne sie ist sie ein farbfleck. Nicht "aufraeumen".
TIER_ALPHA = (0.0, 0.55, 0.85, 1.0)

#: Grund, gegen den die dunkle stufe gemischt wird (die spielfarbe des raums).
_GROUND = (8.0, 13.0, 21.0)
_WHITE = (255.0, 255.0, 255.0)

#: Seed-mischung. BEWUSST anders als in `body_style.py`, sonst korrelierte die
#: marke eines koerpers mit seiner oberflaeche.
_SEED_MIX = 2654435761
_SEED_ADD = 12345


def _mix(a, b, t):
    return tuple(a[k] + (b[k] - a[k]) * t for k in range(3))


def _sym(i, j):
    """Punktsymmetrischer schluessel: (i,j) und (-i,-j) teilen den zustand.

    Damit liegt der pixel-schwerpunkt der marke exakt in ihrer mitte --
    `tests/selection_camera_test.py` §7 misst genau den gegen die gerechnete
    position und laesst nur 5 px zu.
    """
    return (i, j) if (i > 0 or (i == 0 and j > 0)) else (-i, -j)


def _ring_positions(radius):
    """Die zellen, deren EUKLIDISCHER abstand auf `radius` rundet.

    Tschebyschow waere hier falsch: das gaebe einen quadratischen umriss.
    """
    out = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if int(round(math.hypot(i, j))) == radius:
                out.append((i, j))
    return out


# --------------------------------------------------------------------------
# Die beiden entwuerfe
# --------------------------------------------------------------------------

# > **Gewuerfelt wird je ZELLE, nicht aus einem Rauschfeld.** Ein FBM-Feld
# > wurde probiert, weil es zusammenhaengende Flecken statt Koernung gibt --
# > und war der falsche Weg: mit dem radialen Abfall, den die Marke braucht,
# > damit sie eine Scheibe bleibt, wurde aus jedem Koerper dieselbe Scheibe
# > mit hellem Kern. Genau die Eigenschaft, um die es hier geht, ging dabei
# > verloren. Der Zellwurf ist der des abgenommenen Entwurfs und bleibt.


def _rosette(rng, grid):
    """Entwurf A -- die Scheibe wird durchgehend gewuerfelt, radial gewichtet.

    Kein fester Kern. Ein solcher war zweimal falsch: fuenf feste Zellen sind
    bei 15x15 ein Punkt, und ein mitwachsender Kern ist eine glatte Scheibe,
    die genau die Textur auffrisst, um die es hier geht. Statt dessen haengen
    die WAHRSCHEINLICHKEITEN am Radius -- innen fast nur helle Zellen, nach
    aussen immer mehr leere. Das gibt einen dichten, hellen Kern mit
    unregelmaessigem Rand und einen ausfransenden Saum, und beides bleibt bei
    jedem Raster erhalten, weil es ein Anteil ist und keine Zellzahl.
    """
    radius = (grid - 1) // 2
    reach = radius + 0.35
    decided = {}
    cells = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            dist = math.hypot(i, j)
            if dist > reach:
                continue
            key = _sym(i, j)
            if key not in decided:
                t = dist / reach
                p_empty = 0.02 + 0.58 * t * t
                p_bright = max(0.0, 0.74 - 0.66 * t)
                r = rng()
                if r < p_empty:
                    decided[key] = TIER_EMPTY
                elif r < p_empty + p_bright:
                    decided[key] = TIER_BRIGHT
                elif r < p_empty + p_bright + (1.0 - p_empty - p_bright) * 0.62:
                    decided[key] = TIER_BASE
                else:
                    decided[key] = TIER_DIM
            if decided[key] != TIER_EMPTY:
                cells.append((i, j, decided[key]))
    return cells


def _signature(rng, grid):
    """Entwurf D -- Kern plus 2-4 gesaete Zacken.

    Die Zacken sitzen auf DEMSELBEN kreisfoermigen Ring wie `_ring_positions`.
    Ein frueherer Entwurf liess sie diagonal bis (2,2) laufen: noch im
    N x N-Kasten, aber der Umkreis stieg damit auf 3.54 Zellen, waehrend der
    Kern nur 1.5 braucht. Nach der Normierung auf den Einheitskreis blieb vom
    Kern ein 30-%-Puenktchen mit weit abgesprengten Flecken uebrig.
    """
    radius = (grid - 1) // 2
    core = max(1.0, radius * 0.34)
    decided = {}
    cells = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            d = math.hypot(i, j)
            if d > core:
                continue
            if d <= core * 0.5:
                cells.append((i, j, TIER_BRIGHT))
                continue
            key = _sym(i, j)
            if key not in decided:
                decided[key] = TIER_BRIGHT if rng() < 0.65 else TIER_BASE
            cells.append((i, j, decided[key]))

    pool = _ring_positions(radius)
    count = 2 + int(rng() * 3)
    for _ in range(count):
        if not pool:
            break
        i, j = pool.pop(int(rng() * len(pool)))
        cells.append((i, j, TIER_BASE))
        # Der Steg zwischen Kern und Zacke -- ohne ihn sind es lose Flecken.
        steps = max(1, int(round(radius * 0.5)))
        for step in range(1, steps + 1):
            mi = int(round(i * step / (steps + 1.0)))
            mj = int(round(j * step / (steps + 1.0)))
            if math.hypot(mi, mj) > core:
                cells.append((mi, mj, TIER_DIM))
    return cells


_BUILDERS = {'rosette': _rosette, 'signature': _signature}


# --------------------------------------------------------------------------
# Bau
# --------------------------------------------------------------------------

class IconCells(object):
    """Das gepackte zellfeld einer marke."""

    __slots__ = ('grid', 'words', 'unit', 'seed', 'variant', 'count')

    #: So viele uint32 traegt die packung -- fest, damit der uniform im
    #: shader eine feste laenge hat.

    def __init__(self, grid, words, unit, seed, variant, count):
        self.grid = int(grid)
        self.words = tuple(int(w) & _MASK32 for w in words)
        #: Zellbreite in EINHEITSKREIS-koordinaten.
        self.unit = float(unit)
        self.seed = int(seed) & _MASK32
        self.variant = str(variant)
        self.count = int(count)


def build_icon(seed, variant=None, grid=DEFAULT_GRID):
    """Eine marke aus einem seed. Rein rechnerisch, kein GL, kein pygame."""
    variant = str(variant or DEFAULT_VARIANT)
    if variant not in _BUILDERS:
        variant = DEFAULT_VARIANT
    grid = int(grid)
    if grid < 3:
        grid = 3
    if grid > MAX_GRID:
        grid = MAX_GRID
    if grid % 2 == 0:
        grid -= 1

    rng = _Rng((_imul(int(seed) & _MASK32, _SEED_MIX) + _SEED_ADD) & _MASK32)
    cells = _BUILDERS[variant](rng, grid)
    if not cells:
        cells = [(0, 0, TIER_BRIGHT)]

    # Auf den EINHEITSKREIS normieren: die am weitesten aussen liegende ECKE
    # bestimmt den massstab. Sonst raegte die marke ueber ihren eigenen radius
    # hinaus -- und damit ueber den greifradius, der daran haengt.
    max_ext = max(math.hypot(abs(i) + 0.5, abs(j) + 0.5) for i, j, _ in cells)
    unit = 1.0 / max_ext if max_ext > 0.0 else 1.0

    radius = (grid - 1) // 2
    words = [0] * CELL_WORDS
    for i, j, tier in cells:
        index = (j + radius) * grid + (i + radius)
        words[index >> 4] |= (int(tier) & 3) << ((index & 15) * 2)

    return IconCells(grid, words, unit, seed, variant, len(cells))


def icon_palette(color):
    """Die drei stufen-farben aus `body.color` (RGB 0..255).

    Der farbton bleibt erkennbar, aber die IDENTITAET traegt das muster:
    fuenf monde teilen sich praktisch dasselbe grau, und Ganymed und Oberon
    stehen in `solar_system.json` sogar auf demselben `#9c8f7c`.
    """
    base = tuple(float(c) for c in tuple(color)[:3])
    return (
        tuple(c / 255.0 for c in _mix(base, _GROUND, 0.42)),   # dunkel
        tuple(c / 255.0 for c in base),                        # grund
        tuple(c / 255.0 for c in _mix(base, _WHITE, 0.5)),     # hell
    )


def cells_array(icon):
    """Das feld als `(grid, grid)` int8 -- fuer tests und fehlersuche.

    Zeile 0 ist die UNTERSTE (y nach oben), wie im einheitskreis.
    """
    radius = (icon.grid - 1) // 2
    out = np.zeros((icon.grid, icon.grid), dtype=np.int8)
    for index in range(icon.grid * icon.grid):
        tier = (icon.words[index >> 4] >> ((index & 15) * 2)) & 3
        out[index // icon.grid, index % icon.grid] = tier
    return out


def seed_for(body, offset=0):
    """Der seed einer marke: `style_seed`, sonst der name -- plus ein globaler
    versatz.

    `offset` ist der ganze Zoo einer neuen Serie mit einem Knopf: er geht
    genauso durch die `_Rng`-mischung wie ein Körper-Seed selbst (siehe
    `build_icon`), nicht als simple Addition -- eine Addition würde
    benachbarte Offsets zu fast identischen Mustern machen, weil `_Rng` seinen
    ersten Wurf stark vom niedrigen Bit des Seeds abhängen lässt.
    """
    seed = getattr(body, 'style_seed', None)
    if seed is None:
        seed = seed_from_name(getattr(body, 'name', '?'))
    seed = int(seed) & _MASK32
    if offset:
        seed = (seed ^ (_imul(int(offset) & _MASK32, _SEED_MIX) + _SEED_ADD)) & _MASK32
    return seed
