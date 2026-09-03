"""Der hintergrund auf dem ECHTEN framebuffer -- was numpy nicht sehen kann.

`background_test.py` rechnet die ganze ebene headless nach und war trotzdem
gruen, waehrend das sternenfeld im spiel **vollstaendig unsichtbar** war: die
zellmaske in `star.frag` las `gl_PointCoord`, und der NVIDIA-treiber dieses
rechners (4.6.0, 595.71) liefert darin in JEDEM fragment exakt (0, 0). Damit
war `q = abs(0 - 0.5) * 2 = 1`, jedes fragment fiel dem `discard` zum opfer,
und zwar genau ab `pixel_round > 0` -- also in der Vorgabe.

Kein rein rechnender test kann das finden. Diese datei zieht deshalb pixel:

  1. Bei den GESCHIFFTEN vorgabewerten setzen die sterne tinte aufs bild
  2. ... und zwar ueber jede rundung hinweg, nicht nur bei round = 0
  3. Das gitter setzt tinte, und `grid_enabled` schaltet genau sie ab
  4. `enabled = False` laesst den framebuffer voellig unberuehrt
  5. Ein schwenk verschiebt das gitter, nicht das ganze bild

Aufruf: python tests/background_gl_test.py
"""

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

W, H = 900, 700
FAILURES = []


def check(condition, label, detail=''):
    if condition:
        print(f"  ok   {label}{(' -- ' + detail) if detail else ''}")
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label} -- {detail}")


# NIEMALS pygame.init() -- siehe CLAUDE.md. Nur display und font.
pygame.display.init()
pygame.font.init()
pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL, vsync=0)
gl = moderngl.create_context()

from ship.camera import Camera                                  # noqa: E402
from config.loader import ConfigLoader                             # noqa: E402
from render.renderer import Renderer                              # noqa: E402

renderer = Renderer(W, H, enable_fxaa=False, ctx=gl)
camera = Camera(None, W, H)
fbo = gl.simple_framebuffer((W, H))
fbo.use()
renderer.ctx.viewport = (0, 0, W, H)

bg = renderer.background


# Zwei bilder lassen sich nur vergleichen, wenn NICHTS ausser der gepruef-
# ten schicht sich zwischen ihnen aendert. Das gitter blendet aber ueber
# `grid_fade` ein und die sterne funkeln ueber `u_time` -- beides laeuft mit
# jedem gezeichneten bild weiter. Deshalb: leerlauf-ausblenden praktisch
# abschalten, jeden schuss mit derselben zahl bilder aus demselben zustand
# fahren, und die zeit vorher zuruecksetzen.
bg.idle_fade_delay = 1.0e9
SETTLE = 90                      # bei 7/s ist grid_fade danach exakt 1.0


def shot(steps=SETTLE, dt=1 / 60.0):
    """Einen hintergrund aus definiertem zustand zeichnen und auslesen."""
    bg.grid_fade = 1.0
    bg.time_s = 0.0
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    for _ in range(steps):
        renderer._draw_background(camera, dt)
    return np.frombuffer(fbo.read(components=3),
                         dtype=np.uint8).reshape(H, W, 3).astype(int)


def ink(with_layer, without_layer):
    """Zahl der pixel, die die eine schicht gegenueber der anderen aendert."""
    return int((np.abs(with_layer - without_layer).sum(axis=2) > 0).sum())


# =====================================================================
print("\n1. Die GESCHIFFTEN vorgabewerte zeigen sterne")
# =====================================================================
# Genau das war kaputt. Geprueft wird nicht "irgendeine einstellung
# funktioniert", sondern die aus config.json -- die einzige, die der spieler
# ohne zutun zu sehen bekommt.
cfg = ConfigLoader()
cfg.load()
cfg.apply_to_background(bg)
shipped_round = float(bg.pixel_round)
check(shipped_round > 0.0,
      "die Vorgabe ist die leuchtpunkt-matrix, nicht der volle raster",
      f"pixel_round = {shipped_round}")

bg.grid_enabled = False          # nur die sterne sollen sich unterscheiden
bg.stars_enabled = True
with_stars = shot()
bg.stars_enabled = False
without_stars = shot()
bg.stars_enabled = True
bg.grid_enabled = True
star_ink = ink(with_stars, without_stars)
check(star_ink > 100,
      "das sternenfeld setzt bei den vorgabewerten tinte aufs bild",
      f"{star_ink} pixel gegen {bg.star_density} sterne")

# Und sie sind auch HELL genug, um sichtbar zu sein -- ein stern mit
# alpha 0.01 waere formal tinte und praktisch nichts.
delta = np.abs(with_stars - without_stars).sum(axis=2)
check(int(delta.max()) > 120,
      "und sie sind hell, nicht nur formal vorhanden",
      f"hellster stern +{int(delta.max())} (RGB-summe)")

# =====================================================================
print("\n2. Ueber JEDE rundung hinweg, nicht nur bei round = 0")
# =====================================================================
# Der fehler zeigte sich exakt an dieser stelle: bei 0 waren die sterne da,
# ab dem kleinsten wert darueber verschwanden alle auf einmal.
counts = {}
bg.grid_enabled = False
for value in (0.0, 0.1, 0.5, 1.0):
    bg.pixel_round = value
    bg.stars_enabled = True
    a = shot()
    bg.stars_enabled = False
    b = shot()
    counts[value] = ink(a, b)
bg.grid_enabled = True
bg.pixel_round = shipped_round
bg.stars_enabled = True
check(all(v > 100 for v in counts.values()),
      "jede rundung zeigt sterne",
      "  ".join(f"round={k}: {v}px" for k, v in sorted(counts.items())))
check(counts[1.0] < counts[0.0],
      "und die runde zelle setzt WENIGER pixel als die volle -- die maske "
      "greift ueberhaupt",
      f"{counts[1.0]} < {counts[0.0]}")

# =====================================================================
print("\n3. Das gitter setzt tinte, und grid_enabled schaltet sie ab")
# =====================================================================
bg.stars_enabled = False         # nur das gitter soll sich unterscheiden
bg.grid_enabled = True
with_grid = shot()
bg.grid_enabled = False
without_grid = shot()
bg.grid_enabled = True
bg.stars_enabled = True
grid_ink = ink(with_grid, without_grid)
check(grid_ink > 1000,
      "das gitter setzt tinte aufs bild",
      f"{grid_ink} pixel")

# =====================================================================
print("\n4. enabled = False laesst den framebuffer unberuehrt")
# =====================================================================
bg.enabled = False
off = shot()
bg.enabled = True
check(int(np.abs(off).sum()) == 0,
      "abgeschaltet zeichnet die ebene gar nichts",
      f"restsumme {int(np.abs(off).sum())}")

# =====================================================================
print("\n5. Ein schwenk verschiebt das gitter")
# =====================================================================
# Die zusage aus §8: ein schwenk schiebt das gitter um genau die
# schwenkstrecke. Hier wird nachgesehen, dass davon auch pixel ankommen --
# ein anker, der nie im shader landet, faellt sonst nicht auf (genau das ist
# mit u_level_phase schon einmal passiert).
camera.scale = camera.target_scale = 1e-6
before = shot()
camera.position.x += 120.0 / camera.scale        # 120 px schwenken
after = shot()
moved = int((np.abs(after - before).sum(axis=2) > 0).sum())
check(moved > W * H * 0.02,
      "ein schwenk aendert das bild -- der anker kommt im shader an",
      f"{moved} pixel geaendert ({100.0 * moved / (W * H):.1f} %)")

# =====================================================================
print("\n6. Ein verfolgter KOERPER bewegt die sterne -- durch den echten pfad")
# =====================================================================
# `_draw_background` liest die eigenbewegung aus der POSITION des blickziels,
# nicht aus `body.velocity` -- das ist fuer jeden himmelskoerper (0, 0). Hier
# haengt genau der plumbing-teil dran: focus_world_xy, focus_key, sim_time.


class FakeBody:
    """Ein koerper wie aus solar_system.json: position, velocity IMMER null."""

    def __init__(self, name):
        self.name = name
        self.position = type('V', (), {'x': 0.0, 'y': 0.0})()
        self.velocity = type('V', (), {'x': 0.0, 'y': 0.0})()


erde = FakeBody('Erde')
camera.target = erde
bg._prev_focus = None
bg.star_pan_px[:] = 0.0
V_ERDE = 2.98e4
for i in range(61):                       # eine sim-sekunde je bild
    erde.position.x = V_ERDE * i
    renderer._frame_time_s = float(i)
    renderer._draw_background(camera, 1 / 60.0)
star_moved = float(bg.star_pan_px[0])
camera.target = None
check(abs(star_moved - V_ERDE / 1000.0 * bg.star_motion_scale) < 1e-6,
      "ein koerper OHNE velocity-feld treibt die sterne durch den echten "
      "zeichenpfad",
      f"{star_moved:.4f} px gegen "
      f"{V_ERDE / 1000.0 * bg.star_motion_scale:.4f}")
check(abs(star_moved) > 1.0,
      "und zwar sichtbar, nicht in der siebten stelle",
      f"{star_moved:.2f} px je sekunde")

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for f in FAILURES:
        print(f"  {f}")
    sys.exit(1)
print("background (GL): alle pruefungen bestanden")
