"""Regressionstest fuer die zeitmessung des zeichenwegs und fuer lecks.

Anlass: `rend_calc` in der `TIMING:`-zeile sah dauerhaft zu hoch aus. Der
grund war keine langsame zeichnung, sondern die buchhaltung -- `present()`
hat `frame_ms` auf "render-start bis nach dem swap" umgeschrieben, und weil
`rend_calc` daraus als `frame_ms - swap` gebildet wurde, lief alles, was die
hauptschleife ZWISCHEN render() und present() zeichnet (vor allem das
spieler-HUD, gemessen ~8 ms median), stillschweigend unter "render calc".
Das war etwa die haelfte der zahl.

Geprueft wird:
  1. (GL) `frame_ms` ist und bleibt die dauer von render() selbst -- present()
     ruehrt es nicht mehr an -- und die luecke dazwischen steht als
     `overlay_ms` da. Mit gegenprobe gegen die alte rechnung, die sie
     verschluckt hat
  2. (GL) Die gemessenen phasen von render() decken `frame_ms` ab; es bleibt
     kein grosser unzugeordneter rest
  3. (GL) Ueber viele frames waechst nichts: keine GL-puffer, keine
     vertex-arrays, keine textur je frame, und die label-zwischenspeicher
     bleiben in ihren grenzen

Aufruf: python tests/render_budget_test.py
"""

import gc
import math
import os
import sys
import time

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
from render.renderer import Renderer
from physics.vec import Vec2
from physics.world import world as World

config = ConfigLoader()
world = World(float(config.get('physics.gravitational_constant', 6.6730831e-11)))
world.body = SystemLoader(config.get('simulation.system_file',
                                     'solar_system.json')).load()
config.apply_to_world(world)
world.update_planets(0.0)

renderer = Renderer(W, H, enable_fxaa=True, ctx=gl)
config.apply_to_renderer(renderer)
camera = Camera(None, W, H)
config.apply_to_camera(camera)
camera.pan_inertia_enabled = False

ERDE = next(b for b in world.body if b.name == 'Erde')
camera.position = Vec2(float(ERDE.position.x), float(ERDE.position.y))
camera.target_position = camera.position.copy()


def draw_frame(selected=None):
    gl.screen.use()
    gl.clear(0.0, 0.0, 0.0, 1.0)
    renderer.render(world.body, camera, None, sim_time=world.time,
                    real_dt=1.0 / 180.0, selected_body=selected)


def busy_wait(seconds):
    """Aktives warten. `time.sleep` waere hier zu grob (windows: ~15 ms)."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


# =====================================================================
print("1. frame_ms ist render() selbst, die luecke steht in overlay_ms")
# =====================================================================

for _ in range(30):                     # aufwaermen (shader, caches)
    draw_frame()
    renderer.present()

GAP_S = 0.006
draw_frame()
after_render = dict(renderer.last_frame_timings)
busy_wait(GAP_S)                        # das tut in runtime/loop.py ui_root.render()
renderer.present()
after_present = dict(renderer.last_frame_timings)

check(after_present['frame_ms'] == after_render['frame_ms'],
      "present() schreibt frame_ms NICHT mehr um",
      f"{after_render['frame_ms']:.3f} -> {after_present['frame_ms']:.3f} ms")

overlay = float(after_present['overlay_ms'])
check(abs(overlay - GAP_S * 1000.0) < 2.0,
      "overlay_ms misst die luecke zwischen render() und present()",
      f"{overlay:.3f} ms bei {GAP_S * 1000.0:.1f} ms wartezeit")

check(float(after_render['overlay_ms']) == 0.0,
      "direkt nach render() ist overlay_ms noch null")

# Die gegenprobe ist der eigentliche punkt: die ALTE rechnung haette die
# luecke in rend_calc gesteckt, und zwar in voller hoehe.
alt = float(after_present['frame_ms']) + overlay - float(after_present['swap_or_present_ms'])
neu = float(after_present['frame_ms'])
check(alt - neu > 0.8 * GAP_S * 1000.0,
      "gegenprobe: die alte rechnung haette die luecke rend_calc zugeschlagen",
      f"alt {alt:.3f} ms gegen neu {neu:.3f} ms")

# Ohne luecke darf die zahl nicht davonlaufen (kein stehengebliebener
# bezugspunkt aus dem vorigen frame).
draw_frame()
renderer.present()
check(float(renderer.last_frame_timings['overlay_ms']) < 2.0,
      "ohne luecke bleibt overlay_ms klein",
      f"{renderer.last_frame_timings['overlay_ms']:.3f} ms")


# =====================================================================
print("\n2. die phasen decken frame_ms ab")
# =====================================================================

PHASES = ('bodies_ms', 'reference_trails_ms', 'orbit_lines_ms', 'hud_ms',
          'fxaa_ms')
rests = []
for _ in range(60):
    draw_frame()
    renderer.present()
    t = renderer.last_frame_timings
    stats = getattr(renderer, '_last_prediction_render_stats', {}) or {}
    named = sum(float(t.get(k, 0.0)) for k in PHASES)
    named += float(stats.get('prepare_ms', 0.0)) + float(stats.get('draw_ms', 0.0))
    rests.append(float(t['frame_ms']) - named)

rests.sort()
median_rest = rests[len(rests) // 2]
check(median_rest >= -0.5,
      "die summe der phasen ueberschreitet frame_ms nicht",
      f"rest median {median_rest:.3f} ms")
check(median_rest < 3.0,
      "und laesst keinen grossen unzugeordneten rest stehen",
      f"rest median {median_rest:.3f} ms, max {rests[-1]:.3f} ms")


# =====================================================================
print("\n3. je frame waechst nichts")
# =====================================================================

GL_TYPES = ('Buffer', 'VertexArray', 'Framebuffer', 'Renderbuffer', 'Texture')


def gl_census():
    counts = dict.fromkeys(GL_TYPES, 0)
    for obj in gc.get_objects():
        name = type(obj).__name__
        if name in counts:
            counts[name] += 1
    return counts


def churn(n, start=0):
    """`n` frames mit wechselndem zoom, zeit und auswahl.

    Der wechsel ist absicht: ein standbild traefe jeden zwischenspeicher und
    bewiese nichts. So laufen beschriftungen (die geschwindigkeit aendert
    sich), die koerper-optik und die icon/scheibe-umschaltung durch.
    """
    for i in range(start, start + n):
        camera.scale = 10.0 ** (-6.0 - 2.0 * (0.5 + 0.5 * math.sin(i * 0.05)))
        camera.target_scale = camera.scale
        world.update_planets(world.time + 3600.0)
        renderer.set_frame_time(world.time)
        draw_frame(selected=world.body[i % len(world.body)])
        renderer.present()


churn(200)                              # aufwaermen: caches fuellen sich
gc.collect()
before = gl_census()
churn(400, start=200)
gc.collect()
after = gl_census()

for name in GL_TYPES:
    delta = after[name] - before[name]
    # Ein einzelnes objekt darf noch dazukommen (ein neuer beschriftungstext,
    # der vorher nicht vorkam). Was je frame leckt, waere hier vierstellig.
    check(delta <= 4, f"{name}: keine allokation je frame",
          f"{before[name]} -> {after[name]} (delta {delta:+d}) ueber 400 frames")

cache = getattr(renderer, '_label_texture_cache', {})
check(len(cache) <= renderer._label_texture_cache_max,
      "der beschriftungs-zwischenspeicher bleibt in seiner grenze",
      f"{len(cache)} von {renderer._label_texture_cache_max}")

check(len(renderer._deferred_labels) <= len(world.body),
      "die aufgeschobenen beschriftungen werden je frame geleert",
      f"{len(renderer._deferred_labels)} eintraege")

# Und die zeitmessung selbst waechst nicht mit: `last_frame_timings` ist ein
# dict fester groesse, kein protokoll.
check(len(renderer.last_frame_timings) <= 10,
      "last_frame_timings bleibt ein dict fester groesse",
      f"{len(renderer.last_frame_timings)} schluessel")


# =====================================================================
print()
if FAILURES:
    print(f"{len(FAILURES)} FEHLER")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("render-budget: alle pruefungen bestanden")
