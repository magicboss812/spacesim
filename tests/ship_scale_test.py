"""Regressionstest fuer die zoom-abhaengige groesse des schiffs.

Das schiff wird in BILDSCHIRM-pixeln gezeichnet (bei echtem massstab waere
es auf jeder spielbaren zoomstufe kleiner als ein pixel), aber nicht mehr
auf jeder zoomstufe gleich gross: weit herausgezoomt faehrt der massstab auf
`ship_zoom_shrink_min` herunter, damit die silhouette nicht die halbe bahn
ueberdeckt.

Geprueft wird:
  1. Die kennlinie: 1.0 oberhalb des starts, der bodenwert unterhalb des
     endes, dazwischen monoton fallend
  2. Sie ist eine RAMPE, keine stufe -- und weder linear in der skala noch
     mit knick an den enden (smoothstep im log-raum)
  3. Kaputte oder vertauschte konfiguration schrumpft gar nicht, statt zu
     stuerzen
  4. (GL) Die gezeichneten pixel folgen der kennlinie: dasselbe schiff, zwei
     zoomstufen, gemessenes groessenverhaeltnis == gerechneter faktor. Mit
     gegenprobe bei abgeschalteter schrumpfung (dann bleibt es gleich gross)
  5. (GL) Das schrumpfen ist PROPORTIONAL: breite und hoehe folgen demselben
     faktor, das seitenverhaeltnis haelt. Nicht erst die hoehe und dann die
     breite

Aufruf: python tests/ship_scale_test.py
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

import numpy as np

W, H = 800, 500
FAILURES = []


def check(condition, label, detail=''):
    if condition:
        print(f"  ok   {label}{(' -- ' + detail) if detail else ''}")
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label} -- {detail}")


from render.renderer import Renderer


class Knobs:
    """Nur die felder, die `_ship_zoom_shrink_factor` anfasst.

    Die kennlinie wird OHNE GL geprueft -- sie ist reine rechnung, und ein
    kontext waere hier nur zeremonie.
    """

    def __init__(self, **over):
        self.ship_zoom_shrink_enabled = True
        self.ship_zoom_shrink_start_scale = 1e-6
        self.ship_zoom_shrink_end_scale = 1e-9
        self.ship_zoom_shrink_min = 0.55
        for key, value in over.items():
            setattr(self, key, value)

    def factor(self, scale):
        return Renderer._ship_zoom_shrink_factor(self, scale)


# =====================================================================
print("1. die kennlinie steht an den enden fest")
# =====================================================================

k = Knobs()
START, END, FLOOR = 1e-6, 1e-9, 0.55

check(k.factor(1e-3) == 1.0, "weit hineingezoomt: volle groesse")
check(k.factor(START) == 1.0, "genau am start: noch volle groesse")
check(k.factor(END) == FLOOR, "genau am ende: bodenwert", f"{k.factor(END):.4f}")
check(k.factor(1e-14) == FLOOR, "weit darunter: bleibt auf dem bodenwert")

mid = k.factor(math.sqrt(START * END))
check(FLOOR < mid < 1.0, "in der mitte liegt der faktor dazwischen",
      f"{mid:.4f}")

# 601 stufen ueber die ganzen drei dekaden.
xs = [10.0 ** (math.log10(START) + math.log10(END / START) * i / 600.0)
      for i in range(601)]
vals = [k.factor(x) for x in xs]
check(all(vals[i] >= vals[i + 1] - 1e-15 for i in range(len(vals) - 1)),
      "ueber die ganze rampe faellt der faktor monoton")


# =====================================================================
print("\n2. es ist eine rampe, keine stufe -- und keine gerade")
# =====================================================================

jump = max(abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1))
check(jump < 0.01, "kein sprung: der groesste schritt ist winzig",
      f"{jump:.5f} ueber 600 stufen")

# Smoothstep heisst: an BEIDEN enden ist die aenderungsrate ~0. Eine gerade
# haette dort dieselbe steigung wie in der mitte und knickte sichtbar ein.
d_start = abs(vals[1] - vals[0])
d_mid = abs(vals[301] - vals[300])
d_end = abs(vals[-1] - vals[-2])
check(d_start < 0.1 * d_mid and d_end < 0.1 * d_mid,
      "die enden sind knickfrei (aenderungsrate faellt dort auf ~0)",
      f"start {d_start:.2e}, mitte {d_mid:.2e}, ende {d_end:.2e}")

# Und sie laeuft im LOG-raum: eine linear in `scale` gerechnete rampe waere
# nach einer zehntel dekade schon fast am boden. Hier steht sie nach einer
# ganzen dekade noch deutlich ueber der mitte.
one_decade = k.factor(START / 10.0)
linear_would_be = 1.0 + (FLOOR - 1.0) * (1.0 - (START / 10.0) / START)
check(one_decade > 0.75,
      "eine dekade nach dem start ist erst ein kleiner teil verbraucht",
      f"{one_decade:.4f} (linear gerechnet waere es {linear_would_be:.4f})")


# =====================================================================
print("\n3. kaputte konfiguration schrumpft nicht, statt zu stuerzen")
# =====================================================================

check(Knobs(ship_zoom_shrink_enabled=False).factor(1e-14) == 1.0,
      "abgeschaltet bleibt die groesse fest")
check(Knobs(ship_zoom_shrink_end_scale=1e-3).factor(1e-8) == 1.0,
      "start und ende vertauscht: keine schrumpfung")
check(Knobs(ship_zoom_shrink_end_scale=1e-6).factor(1e-8) == 1.0,
      "start == ende: keine division durch null")
check(Knobs(ship_zoom_shrink_start_scale=0.0).factor(1e-8) == 1.0,
      "skala 0 in der konfiguration: keine schrumpfung")
for bad in (float('nan'), float('inf'), -1.0, None, 'x'):
    check(k.factor(bad) == 1.0, f"unbrauchbare kamera-skala {bad!r}")
check(Knobs(ship_zoom_shrink_min=-5.0).factor(1e-14) >= 0.05,
      "ein negativer bodenwert wird geklemmt, nicht uebernommen",
      f"{Knobs(ship_zoom_shrink_min=-5.0).factor(1e-14):.4f}")


# =====================================================================
print("\n4. gegen echte pixel")
# =====================================================================

import moderngl
import pygame
from pygame.locals import DOUBLEBUF, OPENGL

# Nur display+font -- pygame.init() zaehlt mixer- und joystick-geraete auf
# und kostet dabei ~45 s. Siehe runtime/window.py.
pygame.display.init()
pygame.font.init()
pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL, vsync=0)
gl = moderngl.create_context()
gl.enable(moderngl.BLEND)
gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

from ship.camera import Camera
from config.loader import ConfigLoader
from runtime.system_loader import SystemLoader
from physics.vec import Vec2
from physics.world import world as World

config = ConfigLoader()
world = World(float(config.get('physics.gravitational_constant', 6.6730831e-11)))
world.body = SystemLoader(config.get('simulation.system_file',
                                     'solar_system.json')).load()
config.apply_to_world(world)
world.update_planets(0.0)

renderer = Renderer(W, H, enable_fxaa=False, ctx=gl)
config.apply_to_renderer(renderer)
# Die hintergrund-ebene MUSS hier aus sein. `ink_extent` unten sucht helle,
# wenig gesaettigte pixel -- und genau so sehen die gitterlinien aus
# (0.10 * RGB(206,218,226) additiv). Mit hintergrund misst der test die
# volle bildbreite statt der schiffs-silhouette und meldet auf jeder
# zoomstufe dieselben 705 px.
renderer.background.enabled = False
camera = Camera(None, W, H)
config.apply_to_camera(camera)
camera.pan_inertia_enabled = False

SHIP = next(b for b in world.body if getattr(b, 'is_ship', False))
SHIP.theta = 0.0

# NUR das schiff zeichnen: sonst muesste die messung die ink eines koerpers
# von der des schiffs trennen, und bei den zoomstufen, um die es hier geht,
# liegen beide uebereinander.
ONLY_SHIP = [SHIP]


def draw_at(scale):
    """Einen frame mit dem schiff in der bildmitte zeichnen."""
    camera.scale = float(scale)
    camera.target_scale = float(scale)
    camera.target = None
    camera.position = Vec2(float(SHIP.position.x), float(SHIP.position.y))
    camera.target_position = camera.position.copy()
    gl.screen.use()
    gl.clear(0.0, 0.0, 0.0, 1.0)
    renderer.render(ONLY_SHIP, camera, None, sim_time=world.time, real_dt=0.0)
    data = gl.screen.read(viewport=(0, 0, W, H), components=3, dtype='f1')
    return np.frombuffer(data, dtype=np.uint8).reshape(H, W, 3)[::-1]


def ink_extent(image):
    """Breite der schiffs-silhouette in pixeln.

    Zwei dinge muessen dabei draussen bleiben. Die beschriftung (name,
    geschwindigkeit) steht ueber und unter dem schiff -- dagegen hilft der
    schmale streifen auf hoehe der bildmitte. Und aus dem schiff heraus
    zeigen immer die beiden orientierungs-vektoren des debug-overlays
    (gruen prograde, magenta normal-innen), die mit festen 55 px gezeichnet
    werden und beim schrumpfen stehen bleiben -- sie wuerden die messung
    nach unten glaetten. Sie sind KRAEFTIG bunt, der rumpf ist weiss/grau:
    ausgewertet werden deshalb nur die wenig gesaettigten pixel.
    """
    band = image[int(H * 0.5) - 6:int(H * 0.5) + 6].astype(np.int16)
    high = band.max(axis=2)
    low = band.min(axis=2)
    mask = (high > 40) & ((high - low) <= 45)
    if not mask.any():
        return None
    xs = np.nonzero(mask.any(axis=0))[0]
    return float(xs.max() - xs.min() + 1)


BIG, SMALL = 1e-5, 1e-11        # klar ueber dem start, klar unter dem ende

wide = ink_extent(draw_at(BIG))
narrow = ink_extent(draw_at(SMALL))
check(wide is not None and narrow is not None,
      "das schiff ist auf beiden zoomstufen im bild",
      f"{wide} px / {narrow} px")

if wide and narrow:
    ratio = narrow / wide
    check(abs(ratio - FLOOR) < 0.06,
          "ganz herausgezoomt ist das schiff auf den bodenwert geschrumpft",
          f"gemessen {ratio:.3f}, erwartet {FLOOR:.3f} "
          f"({wide:.0f} px -> {narrow:.0f} px)")

    # Eine zwischenstufe liegt dazwischen -- und zwar dort, wo die kennlinie
    # sie hinlegt. Das ist der eigentliche punkt: der uebergang ist ein
    # verlauf, kein umschalten zwischen zwei groessen.
    MID = math.sqrt(START * END)
    mid_px = ink_extent(draw_at(MID))
    expect = renderer._ship_zoom_shrink_factor(MID)
    check(mid_px is not None and abs(mid_px / wide - expect) < 0.06,
          "und eine zwischenstufe trifft die kennlinie",
          f"gemessen {mid_px / wide:.3f}, gerechnet {expect:.3f}")
    check(narrow < mid_px < wide,
          "die drei stufen stehen in der richtigen reihenfolge",
          f"{narrow:.0f} < {mid_px:.0f} < {wide:.0f} px")

# Gegenprobe: abgeschaltet bleibt das schiff auf jeder zoomstufe gleich
# gross -- sonst bewiese die messung oben nur, dass sich irgendwas aendert.
renderer.ship_zoom_shrink_enabled = False
try:
    off_wide = ink_extent(draw_at(BIG))
    off_narrow = ink_extent(draw_at(SMALL))
finally:
    renderer.ship_zoom_shrink_enabled = True
check(off_wide == off_narrow,
      "gegenprobe: abgeschaltet ist das schiff auf beiden stufen gleich breit",
      f"{off_wide} px / {off_narrow} px")

# `_ship_half_height_px` zieht mit demselben faktor mit -- alles, was sich am
# rand der silhouette ausrichtet (auswahl-marke, HUD-anschluesse), folgt so
# der schrumpfenden groesse statt in der luft zu haengen.
renderer._ship_zoom_factor = 1.0
full_h = renderer._ship_half_height_px()
renderer._ship_zoom_factor = FLOOR
small_h = renderer._ship_half_height_px()
check(abs(small_h / full_h - FLOOR) < 1e-9,
      "der silhouetten-halbabstand folgt demselben faktor",
      f"{full_h:.2f} px -> {small_h:.2f} px")
renderer._ship_zoom_factor = 1.0


# =====================================================================
print("\n5. das schrumpfen ist proportional (breite UND hoehe)")
# =====================================================================

# Die frage dahinter: schrumpft das schiff erst in der hoehe und dann in der
# breite? Gemessen wird die volle bounding box der silhouette, nicht nur ein
# streifen. Dafuer muss die beschriftung weg -- sie ist weiss, steht ueber
# und unter dem schiff und wuerde die box bestimmen statt der rumpf.
_blit = renderer._blit_text_topdown
renderer._blit_text_topdown = lambda *a, **k: None


def hull_bbox(image):
    """(breite, hoehe) der wenig gesaettigten pixel im ganzen bild.

    Die orientierungs-vektoren des debug-overlays sind kraeftig bunt und
    fallen ueber dieselbe saettigungs-schranke heraus wie in `ink_extent`.
    """
    patch = image.astype(np.int16)
    high = patch.max(axis=2)
    low = patch.min(axis=2)
    m = (high > 40) & ((high - low) <= 45)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return (float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))


try:
    boxes = []
    for scale in (BIG, math.sqrt(START * END), SMALL):
        bb = hull_bbox(draw_at(scale))
        boxes.append((renderer._ship_zoom_shrink_factor(scale), bb))
finally:
    renderer._blit_text_topdown = _blit

check(all(bb is not None for _f, bb in boxes),
      "die silhouette ist auf allen drei stufen messbar")

if all(bb is not None for _f, bb in boxes):
    ref_f, (ref_w, ref_h) = boxes[0]
    ref_aspect = ref_h / ref_w
    geo_aspect = renderer._ship_geometry().height / renderer._ship_geometry().length
    print(f"       grafik h/l = {geo_aspect:.4f}, gemessen bei faktor "
          f"{ref_f:.2f}: {ref_w:.0f} x {ref_h:.0f} px "
          f"(h/b = {ref_aspect:.4f})")

    for f, (bw, bh) in boxes:
        aspect = bh / bw
        # Die schranke ist grosszuegig, und zwar aus einem sachlichen grund:
        # die hoehe ist bei voller groesse nur ~42 px und ganz herausgezoomt
        # ~24 px. Ein einziges pixel rasterung sind dort schon 4 %. Ein
        # echtes "erst hoehe, dann breite" waere ein vielfaches davon.
        check(abs(aspect / ref_aspect - 1.0) < 0.12,
              f"faktor {f:.2f}: das seitenverhaeltnis haelt",
              f"{bw:.0f} x {bh:.0f} px, h/b = {aspect:.4f} "
              f"({100.0 * (aspect / ref_aspect - 1.0):+.1f} %)")
        # Und beide masse folgen DEMSELBEN faktor -- das ist die eigentliche
        # aussage: keine der beiden richtungen laeuft der anderen voraus.
        check(abs(bw / (ref_w * f) - 1.0) < 0.10,
              f"faktor {f:.2f}: die breite folgt der kennlinie",
              f"{bw / (ref_w * f):.3f}")
        check(abs(bh / (ref_h * f) - 1.0) < 0.10,
              f"faktor {f:.2f}: die hoehe folgt ihr genauso",
              f"{bh / (ref_h * f):.3f}")


# =====================================================================
print()
if FAILURES:
    print(f"{len(FAILURES)} FEHLER")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("schiffs-massstab: alle pruefungen bestanden")
