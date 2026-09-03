"""Die POSITIONS-MARKE der koerper -- rechnung und echte pixel.

Das icon hatte bisher keinen eigenen test; geprueft wurde es nur nebenbei von
`selection_camera_test.py`. Seit es ein gesaetes zellmuster traegt
(`body_icon.py`), gibt es genug zuzusichern:

  1. Aus einem seed faellt immer dieselbe marke, aus zwei namen zwei
  2. Die marke bleibt im einheitskreis -- sonst raegt sie ueber den greifradius
  3. Die bit-packung gibt genau das feld zurueck, das hineinging
  4. ECHTE pixel: die marke setzt tinte, und `"disc"` gibt die alte scheibe
  5. **Icon-fest, nicht bildschirmfest:** ein halber pixel versatz aendert das
     bild, aber nicht seine helligkeit
  6. Die ueberblendung laeuft monoton; die marke traegt viele helligkeiten
  7. Die groesse skaliert mit dem echten radius -- min/max/einfluss, und der
     greifradius folgt genau dem, was gezeichnet wird

Aufruf: python tests/body_icon_test.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SDL_WINDOWS_DPI_AWARENESS', 'permonitorv2')

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import moderngl
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL

from bodies import icon as body_icon

W, H = 640, 480
FAILURES = []


def check(condition, label, detail=''):
    if condition:
        print(f"  ok   {label}{(' -- ' + detail) if detail else ''}")
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label} -- {detail}")


# =====================================================================
print("\n1. Aus einem seed faellt immer dieselbe marke")
# =====================================================================
for variant in body_icon.VARIANTS:
    a = body_icon.build_icon(body_icon.seed_from_name('Erde'), variant, 5)
    b = body_icon.build_icon(body_icon.seed_from_name('Erde'), variant, 5)
    c = body_icon.build_icon(body_icon.seed_from_name('Mars'), variant, 5)
    check(a.words == b.words,
          f"[{variant}] derselbe seed liefert dieselbe packung",
          f"{[hex(w) for w in a.words]}")
    check(a.words != c.words,
          f"[{variant}] zwei koerper bekommen verschiedene marken",
          f"Erde {hex(a.words[0])} gegen Mars {hex(c.words[0])}")

# Ganymed und Oberon stehen in solar_system.json auf DERSELBEN farbe
# (#9c8f7c). Wenn die marke sie nicht trennt, trennt sie nichts.
gan = body_icon.build_icon(body_icon.seed_from_name('Ganymed'), 'rosette', 5)
obe = body_icon.build_icon(body_icon.seed_from_name('Oberon'), 'rosette', 5)
check(gan.words != obe.words,
      "Ganymed und Oberon sind unterscheidbar -- sie teilen sich die farbe",
      f"{gan.count} gegen {obe.count} zellen")

# Der globale seed-versatz (renderer.body_icon_seed_offset): eine ganze serie
# neuer marken mit einem knopf, ohne style_seed anzufassen. Er geht durch
# dieselbe _Rng-mischung wie ein koerper-seed selbst -- eine simple addition
# haette benachbarte offsets zu fast identischen mustern gemacht, weil _Rng
# seinen ersten wurf stark vom niedrigsten bit des seeds abhaengen laesst.
ERDE_NAME = type('B', (), {'name': 'Erde', 'style_seed': None})()
seed_a = body_icon.seed_for(ERDE_NAME, 0)
seed_b = body_icon.seed_for(ERDE_NAME, 1)
seed_c = body_icon.seed_for(ERDE_NAME, 2)
check(seed_a == body_icon.seed_from_name('Erde'),
      "versatz 0 ist die namensbasierte vorgabe-marke",
      f"{hex(seed_a)}")
check(len({seed_a, seed_b, seed_c}) == 3,
      "drei verschiedene versaetze ergeben drei verschiedene seeds",
      f"{hex(seed_a)}  {hex(seed_b)}  {hex(seed_c)}")
check(bin(seed_a ^ seed_b).count('1') > 8,
      "und die seeds unterscheiden sich in vielen bits -- keine simple "
      "fortlaufende zahl",
      f"{bin(seed_a ^ seed_b).count('1')} bit unterschied")
check(body_icon.build_icon(seed_a, 'rosette', 9).words
      != body_icon.build_icon(seed_b, 'rosette', 9).words,
      "und damit auch verschiedene marken")

# =====================================================================
print("\n2. Die marke bleibt im einheitskreis")
# =====================================================================
# Der greifradius (`_pick_radius_px`) ist der marken-radius. Raegte das
# muster darueber hinaus, waeren sichtbare zellen nicht anklickbar.
for variant in body_icon.VARIANTS:
    for grid in (5, 7):
        worst = 0.0
        for name in ('Erde', 'Mond', 'Mars', 'Io', 'Titan', 'Charon'):
            icon = body_icon.build_icon(body_icon.seed_from_name(name),
                                        variant, grid)
            cells = body_icon.cells_array(icon)
            radius = (icon.grid - 1) // 2
            for row in range(icon.grid):
                for col in range(icon.grid):
                    if cells[row, col] == 0:
                        continue
                    i, j = col - radius, row - radius
                    corner = math.hypot((abs(i) + 0.5) * icon.unit,
                                        (abs(j) + 0.5) * icon.unit)
                    worst = max(worst, corner)
        check(worst <= 1.0 + 1e-9,
              f"[{variant} {grid}x{grid}] keine ecke ragt heraus",
              f"weiteste ecke {worst:.6f}")

# =====================================================================
print("\n3. Die bit-packung ist verlustfrei")
# =====================================================================
icon = body_icon.build_icon(body_icon.seed_from_name('Jupiter'), 'rosette', 5)
cells = body_icon.cells_array(icon)
check(int((cells > 0).sum()) == icon.count,
      "so viele belegte zellen wie gebaut",
      f"{int((cells > 0).sum())} gegen {icon.count}")
check(int(cells.max()) <= 3 and int(cells.min()) >= 0,
      "jede stufe liegt in 0..3 -- mehr passt nicht in zwei bit",
      f"min {int(cells.min())}, max {int(cells.max())}")
# MAX_GRID x MAX_GRID zellen a 2 bit muessen in CELL_WORDS uint32 passen.
big = body_icon.build_icon(12345, 'signature', body_icon.MAX_GRID)
check(len(big.words) == body_icon.CELL_WORDS
      and big.grid * big.grid <= body_icon.CELL_WORDS * 16,
      f"{body_icon.MAX_GRID}x{body_icon.MAX_GRID} passt in "
      f"{body_icon.CELL_WORDS} uint32",
      f"{big.grid * big.grid} zellen in {body_icon.CELL_WORDS * 16} plaetzen")
check(body_icon.build_icon(1, 'rosette', 99).grid == body_icon.MAX_GRID,
      "ein zu grosses raster wird auf MAX_GRID geklemmt",
      f"99 -> {body_icon.build_icon(1, 'rosette', 99).grid}")
check(body_icon.build_icon(1, 'rosette', 10).grid % 2 == 1,
      "ein gerades raster wird ungerade gemacht -- die entwuerfe sind radial "
      "um eine mittelzelle gebaut",
      f"10 -> {body_icon.build_icon(1, 'rosette', 10).grid}")

# =====================================================================
print("\n3b. Das RASTER bestimmt den detailgrad, nicht die bildschirmgroesse")
# =====================================================================
# Die zusage: eine groessere marke zeigt dasselbe muster groesser, nie ein
# feineres. Und ein feineres raster gibt MEHR muster, nicht mehr matsch.
counts = {}
for g in (5, 9, 13, body_icon.MAX_GRID):
    counts[g] = body_icon.build_icon(
        body_icon.seed_from_name('Erde'), 'rosette', g).count
check(all(counts[a] < counts[b]
          for a, b in zip(sorted(counts), sorted(counts)[1:])),
      "ein feineres raster liefert streng mehr zellen",
      "  ".join(f"{g}x{g}: {n}" for g, n in sorted(counts.items())))

# =====================================================================
# Ab hier: echte pixel.
# =====================================================================
# NIEMALS pygame.init() -- siehe CLAUDE.md. Nur display und font.
pygame.display.init()
pygame.font.init()
pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL, vsync=0)
gl = moderngl.create_context()

from config.loader import ConfigLoader                                # noqa: E402
from render.renderer import Renderer                                 # noqa: E402

renderer = Renderer(W, H, enable_fxaa=False, ctx=gl)
_config = ConfigLoader()
_config.load()
_config.apply_to_renderer(renderer)
renderer.background.enabled = False        # tinte gegen dunklen grund zaehlen
fbo = gl.simple_framebuffer((W, H))
fbo.use()
renderer.ctx.viewport = (0, 0, W, H)


class FakeBody(object):
    """Ein koerper, wie er aus solar_system.json faellt."""

    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.radius = 1.0
        self.style_seed = None


def shot(paint):
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    paint()
    return np.frombuffer(fbo.read(components=3),
                         dtype=np.uint8).reshape(H, W, 3).astype(int)


def ink(frame):
    """Gesamte helligkeit -- das mass, an dem sich kriechen zeigt."""
    return float(frame.sum())


ERDE = FakeBody('Erde', (68, 136, 255))
RADIUS = float(renderer.body_icon_min_radius_px)

# =====================================================================
print("\n4. Die marke setzt echte pixel, und 'disc' gibt die alte scheibe")
# =====================================================================
check(renderer.body_icon_style == 'pixel',
      "die geschiffte vorgabe ist das zellmuster",
      f"body_icon_style = {renderer.body_icon_style!r}")
_icon = body_icon.build_icon(body_icon.seed_from_name('Erde'),
                            renderer.body_icon_variant,
                            renderer.body_icon_grid)
CELL_PX = _icon.unit * RADIUS
check(int(renderer.body_icon_grid) >= 9,
      "die geschiffte vorgabe ist fein genug fuer textur",
      f"{renderer.body_icon_grid}x{renderer.body_icon_grid}, "
      f"{_icon.count} zellen, {CELL_PX:.2f} px je zelle")
_fade_end = RADIUS * float(renderer.body_icon_fade_factor)
check(_fade_end > RADIUS,
      "das ueberblend-band steht richtig herum",
      f"marke {RADIUS:.1f} px -> koerper {_fade_end:.1f} px "
      f"(faktor {renderer.body_icon_fade_factor:.2f})")

pixel_frame = shot(lambda: renderer._draw_body_icon(
    ERDE, W / 2 + 0.37, H / 2 + 0.21, RADIUS, 0.267, 0.533, 1.0, 1.0))
lit = int((pixel_frame.sum(axis=2) > 0).sum())
check(lit > 150,
      "das muster setzt tinte aufs bild",
      f"{lit} pixel bei radius {RADIUS:.0f}")

renderer.body_icon_style = 'disc'
disc_frame = shot(lambda: renderer._draw_body_icon(
    ERDE, W / 2 + 0.37, H / 2 + 0.21, RADIUS, 0.267, 0.533, 1.0, 1.0))
renderer.body_icon_style = 'pixel'
check(not np.array_equal(pixel_frame, disc_frame),
      "'disc' zeichnet etwas anderes -- der schalter wirkt",
      f"scheibe {int((disc_frame.sum(axis=2) > 0).sum())} pixel")

# Zwei koerper mit derselben FARBE muessen verschiedene bilder ergeben.
gan_frame = shot(lambda: renderer._draw_body_icon(
    FakeBody('Ganymed', (156, 143, 124)), W / 2, H / 2, RADIUS,
    0.61, 0.56, 0.49, 1.0))
obe_frame = shot(lambda: renderer._draw_body_icon(
    FakeBody('Oberon', (156, 143, 124)), W / 2, H / 2, RADIUS,
    0.61, 0.56, 0.49, 1.0))
differing = int((np.abs(gan_frame - obe_frame).sum(axis=2) > 0).sum())
check(differing > 20,
      "Ganymed und Oberon sehen trotz gleicher farbe verschieden aus",
      f"{differing} pixel unterschied")

# =====================================================================
print("\n5. Icon-fest, nicht bildschirmfest")
# =====================================================================
# Die zusage: das zellmuster klebt an der marke, nicht am schirm. Wandert
# die marke um bruchteile eines pixels, muss sich das BILD aendern (sie ist
# ja woanders) -- ihre HELLIGKEIT aber praktisch nicht. Rastete das muster
# auf den schirm, sprungen ganze zellen ein und aus und die tinte mit ihnen.
inks = []
frames = []
for step in range(8):
    dx = step / 8.0                      # ein ganzer pixel in acht schritten
    frame = shot(lambda: renderer._draw_body_icon(
        ERDE, W / 2 + dx, H / 2 + 0.21, RADIUS, 0.267, 0.533, 1.0, 1.0))
    inks.append(ink(frame))
    frames.append(frame)

mean = sum(inks) / len(inks)
spread = (max(inks) - min(inks)) / mean * 100.0
check(spread < 2.0,
      "die helligkeit bleibt ueber einen ganzen pixel drift stabil",
      f"±{spread:.2f} % (grenze 2 %)")
moved = int((np.abs(frames[4] - frames[0]).sum(axis=2) > 0).sum())
check(moved > 0,
      "und das bild aendert sich trotzdem -- sonst bewegte sich nichts",
      f"{moved} pixel bei einem halben pixel versatz")

# Gegenprobe: OHNE kantenglaettung springt die deckung pixelweise. Ohne die
# haette der test oben auch eine marke bestanden, die gar nichts glaettet.
renderer.body_icon_edge_px = 0.0
hard = []
for step in range(8):
    hard.append(ink(shot(lambda: renderer._draw_body_icon(
        ERDE, W / 2 + step / 8.0, H / 2 + 0.21, RADIUS,
        0.267, 0.533, 1.0, 1.0))))
renderer.body_icon_edge_px = 1.0
hard_spread = (max(hard) - min(hard)) / (sum(hard) / len(hard)) * 100.0
check(hard_spread > spread,
      "die gegenprobe mit harter kante schwankt staerker",
      f"±{hard_spread:.2f} % gegen ±{spread:.2f} %")

# =====================================================================
print("\n6. Die ueberblendung laeuft monoton")
# =====================================================================
lo = float(renderer.body_icon_min_radius_px)
hi = lo * float(renderer.body_icon_fade_factor)
check(hi > lo,
      "das ueberblend-band ist nicht leer",
      f"{lo:.1f} px -> {hi:.1f} px")

fades = [renderer._body_icon_fade(r)
         for r in np.linspace(lo - 2.0, hi + 2.0, 25)]
check(all(fades[k] >= fades[k + 1] - 1e-9 for k in range(len(fades) - 1)),
      "die deckkraft faellt monoton",
      f"{fades[0]:.2f} -> {fades[-1]:.2f}")
check(abs(fades[0] - 1.0) < 1e-9 and abs(fades[-1]) < 1e-9,
      "voll unterhalb der schwelle, ganz weg oberhalb des bandes")
jump = max(abs(fades[k] - fades[k + 1]) for k in range(len(fades) - 1))
check(jump < 0.25,
      "kein sprung -- genau das poppen, das es zu vermeiden gilt",
      f"groesster schritt {jump:.3f}")

# Der ZELLSPALT: zwischen zwei gleich hellen nachbarzellen muss eine
# dunklere linie stehen. Ohne sie verschmelzen sie zu einer flaeche, und die
# marke liest sich als klecks statt als raster -- genau das war die erste
# fassung, und genau das fiel im spiel auf.
lum = shot(lambda: renderer._draw_body_icon(
    ERDE, W / 2 + 0.37, H / 2 + 0.21, RADIUS,
    0.267, 0.533, 1.0, 1.0)).sum(axis=2)
mid = lum[int(H / 2)]
bright = mid[mid > 0].max() if (mid > 0).any() else 0
span = np.nonzero(mid > bright * 0.45)[0]
dips = []
if span.size > 4:
    seg = mid[span.min():span.max() + 1]
    dips = [seg[k] / bright for k in range(1, len(seg) - 1)
            if seg[k] <= seg[k - 1] and seg[k] <= seg[k + 1]
            and seg[k] < bright * 0.98]
check(len(dips) >= 2 and min(dips) < 0.75,
      "zwischen den zellen steht eine trennlinie -- die marke ist ein raster",
      f"{len(dips)} linien, tiefste bei {min(dips) * 100:.0f} % "
      f"(spalt {renderer.body_icon_cell_gap:.2f})")

# TIEFE: die marke muss viele verschiedene helligkeiten tragen. Drei stufen
# allein geben zu wenig -- gleich eingestufte nachbarn verschmelzen dann zu
# einer flaeche, und genau das sah im spiel flach aus, waehrend der entwurf
# im browser voller schattierungen war. Die eigene helligkeit je zelle
# (`body_icon_shade_jitter`) ist die quelle, nicht mehr der spalt.
def shade_levels(jitter):
    renderer.body_icon_shade_jitter = jitter
    frame = shot(lambda: renderer._draw_body_icon(
        ERDE, W / 2 + 0.37, H / 2 + 0.21, RADIUS, 0.267, 0.533, 1.0, 1.0))
    lit = frame.sum(axis=2)
    lit = lit[lit > 60]
    if lit.size == 0:
        return 0
    return int(np.unique(lit // 24).size)


shipped = float(renderer.body_icon_shade_jitter)
rich = shade_levels(shipped)
flat = shade_levels(0.0)
renderer.body_icon_shade_jitter = shipped
check(rich >= 14,
      "die marke traegt viele verschiedene helligkeiten",
      f"{rich} stufen bei streuung {shipped:.2f}")
check(rich > flat,
      "und die streuung je zelle ist die quelle -- ohne sie sind es weniger",
      f"{flat} stufen ohne streuung gegen {rich} mit")

# Die UMRISSE muessen in BEIDEN achsen gleich ankommen. Der Nutzer sah im
# Spiel nur waagerechte Linien, und das war kein Geschmack: solange die
# Umrissbreite ein ANTEIL der Zelle war, lag sie bei Radius 16 unter einem
# Pixel, und ob sie abgetastet wurde, hing an der Bruchteil-Position der
# Marke -- die in x und y verschieden ist. Als Bildschirmmass (`cell_rim` in
# Pixeln) kommen beide Achsen an.
def line_dips(profile):
    """Zahl der oertlichen Minima, die deutlich unter ihre Nachbarn fallen."""
    return sum(1 for k in range(1, len(profile) - 1)
               if profile[k] < profile[k - 1] * 0.93
               and profile[k] < profile[k + 1] * 0.93)


_lum = shot(lambda: renderer._draw_body_icon(
    ERDE, W / 2 + 0.37, H / 2 + 0.21, RADIUS, 0.267, 0.533, 1.0, 1.0)).sum(axis=2)
_ys, _xs = np.nonzero(_lum > 60)
_box = _lum[_ys.min():_ys.max() + 1, _xs.min():_xs.max() + 1]
# Spaltenmittel zeigt SENKRECHTE linien, Zeilenmittel WAAGERECHTE.
_vert = line_dips(_box.mean(axis=0))
_horz = line_dips(_box.mean(axis=1))
check(_vert >= 2 and _horz >= 2,
      "die zell-umrisse kommen in BEIDEN achsen an",
      f"{_vert} senkrechte, {_horz} waagerechte linien")
check(min(_vert, _horz) >= max(_vert, _horz) * 0.5,
      "und zwar etwa gleich stark -- keine achse faellt aus",
      f"senkrecht {_vert} gegen waagerecht {_horz}")

# Gegenprobe: ohne umriss verschwinden sie.
_rim = float(renderer.body_icon_cell_rim)
renderer.body_icon_cell_rim = 0.0
_flat = shot(lambda: renderer._draw_body_icon(
    ERDE, W / 2 + 0.37, H / 2 + 0.21, RADIUS, 0.267, 0.533, 1.0, 1.0)).sum(axis=2)
renderer.body_icon_cell_rim = _rim
_fbox = _flat[_ys.min():_ys.max() + 1, _xs.min():_xs.max() + 1]
check(line_dips(_fbox.mean(axis=0)) + line_dips(_fbox.mean(axis=1))
      < _vert + _horz,
      "ohne umriss sind es weniger -- sonst pruefte das oben nichts",
      f"{line_dips(_fbox.mean(axis=0))}+{line_dips(_fbox.mean(axis=1))} "
      f"gegen {_vert}+{_horz}")

# MEHR RASTER MUSS MEHR MUSTER HEISSEN, nicht mehr matsch. Der box-filter
# ist auf eine halbe zelle gedeckelt; ohne diesen deckel mittelte ein feineres
# raster sich selbst weg, und die marke wuerde bei 15x15 ein gleichmaessiger
# fleck. Gemessen wird die zahl der helligkeits-WECHSEL quer durch die marke.
def edge_count(grid):
    renderer.body_icon_grid = grid
    renderer._body_icon_cache.clear()
    frame = shot(lambda: renderer._draw_body_icon(
        ERDE, W / 2 + 0.37, H / 2 + 0.21, RADIUS, 0.267, 0.533, 1.0, 1.0))
    band = frame.sum(axis=2)[int(H / 2) - 3:int(H / 2) + 4]
    return int((np.abs(np.diff(band, axis=1)) > 18).sum())


# Bei der geschifften groesse: das raster muss ankommen.
near = {g: edge_count(g) for g in (5, 9)}
check(near[9] > near[5],
      "bei der geschifften groesse zeigt ein feineres raster mehr kanten",
      "  ".join(f"{g}x{g}: {n}" for g, n in sorted(near.items())))

# Und der deckel wirkt: bei einem radius, der 15x15 auch TRAGEN kann, muss
# das feinste raster auch das meiste zeigen. Ohne den deckel mittelte der
# 1-px-filter hier alles weg.
BIG_R = 24.0
wide = {}
for g in (5, 9, body_icon.MAX_GRID):
    renderer.body_icon_grid = g
    renderer._body_icon_cache.clear()
    frame = shot(lambda: renderer._draw_body_icon(
        ERDE, W / 2 + 0.37, H / 2 + 0.21, BIG_R, 0.267, 0.533, 1.0, 1.0))
    band = frame.sum(axis=2)[int(H / 2) - 3:int(H / 2) + 4]
    wide[g] = int((np.abs(np.diff(band, axis=1)) > 18).sum())
renderer.body_icon_grid = int(_config.get('renderer.body_icon_grid', 9))
renderer._body_icon_cache.clear()
check(wide[body_icon.MAX_GRID] > wide[9] > wide[5],
      f"bei radius {BIG_R:.0f} px traegt jedes feinere raster auch mehr muster",
      "  ".join(f"{g}x{g}: {n}" for g, n in sorted(wide.items())))

# Die GRENZE ehrlich benannt: unter rund 1.5 px je zelle kann der schirm das
# raster nicht mehr aufloesen -- kein filter der welt aendert das. Die
# vorgabe muss darueber liegen.
_cell_px = body_icon.build_icon(
    body_icon.seed_from_name('Erde'), renderer.body_icon_variant,
    renderer.body_icon_grid).unit * RADIUS
check(_cell_px >= 1.5,
      "die geschiffte kombination aus raster und groesse ist aufloesbar",
      f"{_cell_px:.2f} px je zelle -- bei {body_icon.MAX_GRID}x"
      f"{body_icon.MAX_GRID} waeren es "
      f"{body_icon.build_icon(1, 'rosette', body_icon.MAX_GRID).unit * RADIUS:.2f} px")

# Und die tinte der marke folgt der deckkraft wirklich.
faded = ink(shot(lambda: renderer._draw_body_icon(
    ERDE, W / 2, H / 2, RADIUS, 0.267, 0.533, 1.0, 0.5)))
full = ink(shot(lambda: renderer._draw_body_icon(
    ERDE, W / 2, H / 2, RADIUS, 0.267, 0.533, 1.0, 1.0)))
ratio = faded / full if full else 0.0
check(0.35 < ratio < 0.65,
      "fade 0.5 halbiert die tinte auch wirklich",
      f"{ratio:.3f} des vollen bildes")

# =====================================================================
print("\n7. Die marken-groesse skaliert mit dem PHYSISCHEN koerper-radius")
# =====================================================================
# Die erste fassung skalierte mit dem BILDSCHIRMradius (`true_radius_px`) und
# klemmte deshalb bei JEDEM einfluss-wert auf `min` zurueck: sobald ein
# koerper klein genug ist, um ueberhaupt eine marke zu sein, liegt sein
# bildschirmradius fast immer weit unter `min`, `scaled = min + (true-min)*
# einfluss` blieb also unter `min` und wurde wieder hochgeklemmt. Der regler
# hatte dadurch im spiel keine sichtbare wirkung -- ein test mit absichtlich
# GROSSEN werten weit ueber `min` sah trotzdem "richtig" aus und haette den
# fehler nicht gefunden. Jetzt haengt die groesse an `body.radius` (metern),
# unabhaengig vom zoom, log-skaliert ueber die spanne der GELADENEN koerper.
_lo = float(renderer.body_icon_min_radius_px)
_hi = float(renderer.body_icon_max_radius_px)

_saved_influence = float(renderer.body_icon_size_influence)
_saved_max = _hi
_saved_range = renderer._icon_radius_range_m

# Eine feste, bekannte spanne setzen statt eine echte koerperliste zu bauen --
# das macht die erwarteten werte von hand nachrechenbar.
LO_M, HI_M = 2.0e5, 2.0e8            # drei dekaden, wie im echten sonnensystem
renderer._icon_radius_range_m = (LO_M, HI_M)

renderer.body_icon_size_influence = 0.0
_flat_sizes = {r: renderer._body_icon_draw_radius_px(r)
              for r in (0.0, LO_M, math.sqrt(LO_M * HI_M), HI_M, HI_M * 100)}
check(all(abs(v - _lo) < 1e-9 for v in _flat_sizes.values()),
      "einfluss 0 -- jede marke bleibt bei body_icon_min_radius_px, "
      "unabhaengig vom koerper-radius",
      "  ".join(f"{k:.1e}->{v:.2f}" for k, v in _flat_sizes.items()))

renderer.body_icon_max_radius_px = 40.0
renderer.body_icon_size_influence = 1.0
_smallest = renderer._body_icon_draw_radius_px(LO_M)
_largest = renderer._body_icon_draw_radius_px(HI_M)
_mid_m = math.sqrt(LO_M * HI_M)      # geometrische mitte -- die MITTE im log
_middle = renderer._body_icon_draw_radius_px(_mid_m)
check(abs(_smallest - _lo) < 1e-9,
      "einfluss 1 -- der kleinste geladene koerper bekommt genau min",
      f"radius {LO_M:.1e} m -> marke {_smallest:.2f} px")
check(abs(_largest - 40.0) < 1e-9,
      "einfluss 1 -- der groesste geladene koerper bekommt genau max",
      f"radius {HI_M:.1e} m -> marke {_largest:.2f} px")
check(abs(_middle - (_lo + 40.0) / 2.0) < 1e-6,
      "einfluss 1 -- log-mitte der spanne liegt genau auf der mitte von "
      "[min, max]",
      f"radius {_mid_m:.2e} m -> marke {_middle:.2f} px "
      f"(mitte waere {(_lo + 40.0) / 2.0:.2f})")
check(abs(renderer._body_icon_draw_radius_px(HI_M * 1000) - 40.0) < 1e-9,
      "und ein koerper ausserhalb der spanne wird auf max geklemmt, "
      "nicht extrapoliert",
      f"{renderer._body_icon_draw_radius_px(HI_M * 1000):.2f} px")

renderer.body_icon_size_influence = 0.5
_mid_half = renderer._body_icon_draw_radius_px(HI_M)
_expect_half = _lo + (40.0 - _lo) * 1.0 * 0.5
check(abs(_mid_half - _expect_half) < 1e-9,
      "einfluss 0.5 -- linear zwischen 'immer min' und 'voll skaliert' "
      "gemischt",
      f"{_mid_half:.3f} px gegen erwartete {_expect_half:.3f} px")

# Ein Jupiter-aehnlicher koerper muss BEI JEDEM ZOOM sichtbar groesser
# bleiben als ein kleiner mond -- das war der ganze punkt der anfrage.
# `true_radius_px` (der bildschirmradius) taucht hier bewusst NICHT auf.
renderer.body_icon_size_influence = 1.0
_moon_radius_px = renderer._body_icon_draw_radius_px(2.0e6)     # ~mond-groesse
_planet_radius_px = renderer._body_icon_draw_radius_px(7.0e7)   # ~jupiter-groesse
check(_planet_radius_px > _moon_radius_px + 1.0,
      "ein grosser koerper bekommt eine sichtbar groessere marke als ein "
      "kleiner, UNABHAENGIG vom aktuellen zoom",
      f"mond {_moon_radius_px:.2f} px gegen planet {_planet_radius_px:.2f} px")

renderer.body_icon_size_influence = 1.0
_ship_pick = renderer._pick_radius_px(
    type('S', (), {'is_ship': True})(), type('C', (), {'scale': 1.0})())
check(abs(_ship_pick - 12.0) < 1e-9,
      "das schiff bleibt beim festen greifradius -- die skalierung gilt nur "
      "koerpern")
_body_stub = type('B', (), {'radius': HI_M, 'is_ship': False})()
_cam_stub = type('C', (), {'scale': 1e-9})()   # winziger bildschirmradius
_pick = renderer._pick_radius_px(_body_stub, _cam_stub)
_expected_draw = renderer._body_icon_draw_radius_px(HI_M)
check(abs(_pick - _expected_draw) < 1e-9 and _pick > 30.0,
      "der greifradius folgt der PHYSISCHEN groesse, nicht dem "
      "verschwindenden bildschirmradius -- sonst waere das klickziel fuer "
      "einen weit entfernten riesenkoerper winzig",
      f"{_pick:.2f} px bei bildschirmradius {HI_M * 1e-9:.2e} px")

renderer.body_icon_size_influence = _saved_influence
renderer.body_icon_max_radius_px = _saved_max
renderer._icon_radius_range_m = _saved_range

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for f in FAILURES:
        print(f"  {f}")
    sys.exit(1)
print("body icon: alle pruefungen bestanden")
