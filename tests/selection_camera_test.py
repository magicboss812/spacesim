"""Regressionstest fuer auswahl per mausklick und den kamera-anflug.

Geprueft wird:
  1. focus_on() springt nicht und laeuft monoton auf den koerper zu
  2. focus_on() endet EXAKT auf einem BEWEGTEN koerper (kein v/k-rueckstand),
     mit gegenprobe gegen die naive glaettung der absoluten position
  3. focus_smoothing bestimmt die dauer, danach ist pan_smoothing zurueck
  4. Der zoom bewegt die kamera GAR NICHT (ankerung auf der bildmitte); der
     verfolgte koerper steht dabei exakt still -- mit gegenprobe gegen das
     frueher hier stehende zeiger-ankern
  5. Ein schwenk loest die verfolgung und laesst die kamera mit
     weltgeschwindigkeit 0 stehen; Home heftet wieder an
  6. Die auswahl ruehrt weder bezugskoerper noch aenderungs-benachrichtigung an
  7. (GL) pick_body trifft das, was GEZEICHNET wird -- gemessen am pixel-
     schwerpunkt des koerpers, im nicht-rotierenden UND im richtungsrahmen
  8. (GL) Die markierung liegt als vier pfeile um den koerper, und ohne
     auswahl ist der frame bit-identisch zum frame davor
  9. (GL) Dreh- und pulsphase sind bildratenunabhaengig

Aufruf: python tests/selection_camera_test.py
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

from camera import Camera
from vec import Vec2

W, H = 800, 500
DT = 1.0 / 180.0
SCALE = 5e-7          # 1 px ~ 2e6 m
FAILURES = []


def check(condition, label, detail=''):
    if condition:
        print(f"  ok   {label}{(' -- ' + detail) if detail else ''}")
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label} -- {detail}")


class FakeBody:
    """Minimaler koerper: nur was kamera und auswahl anfassen."""

    def __init__(self, x, y, vx=0.0, vy=0.0, radius=6.371e6, name='X'):
        self.position = Vec2(float(x), float(y))
        self.velocity = Vec2(float(vx), float(vy))
        self.radius = float(radius)
        self.name = name
        self.is_ship = False

    def advance(self, dt):
        self.position += self.velocity * dt


def make_camera(scale=SCALE):
    cam = Camera(None, W, H)
    cam.scale = scale
    cam.target_scale = scale
    cam.pan_inertia_enabled = False
    return cam


def px(world_length, scale=SCALE):
    return float(world_length) * float(scale)


# =====================================================================
print("1. focus_on: kein sprung, monotoner anflug")
# =====================================================================

body = FakeBody(1.0e9, -4.0e8, name='Ziel')
cam = make_camera()
cam.position = Vec2(0.0, 0.0)
cam.target_position = Vec2(0.0, 0.0)

before = cam.position.copy()
total = (body.position - before).magnitude()
cam.focus_on(body)

check(cam.position.x == before.x and cam.position.y == before.y,
      "der aufruf selbst bewegt das bild um exakt null",
      f"({cam.position.x - before.x}, {cam.position.y - before.y})")
check(cam.target is body, "die verfolgung sitzt sofort auf dem koerper")

distances = []
for _ in range(1200):
    cam.update(DT, ui_wants_keyboard=True)
    distances.append((body.position - cam.position).magnitude())

first_step = total - distances[0]
check(first_step / total < 0.03,
      "der erste frame nimmt nur einen bruchteil des weges",
      f"{100.0 * first_step / total:.2f} % von {px(total):.0f} px")

monotone = all(b <= a + 1e-9 for a, b in zip(distances, distances[1:]))
check(monotone, "der abstand faellt monoton -- kein ueberschwingen")

# Gemessen wird die SICHTBARE dauer, nicht der rechnerische stillstand: eine
# exponentielle annaeherung braucht fuer das letzte halbe pixel so lange wie
# fuer die ersten 95 %, und dieser rest ist nichts, was jemand sieht.
visible = next(i for i, d in enumerate(distances) if d < 0.05 * total)
check(0.4 < visible * DT < 1.0,
      "95 % des weges sind nach ~0.7 s zurueckgelegt",
      f"{visible * DT:.2f} s ({visible} frames)")

settled = next((i for i, d in enumerate(distances) if d == 0.0), None)
check(settled is not None and settled * DT < 2.5,
      "und der rest rastet innerhalb von 2.5 s ein",
      f"{settled * DT:.2f} s")


# =====================================================================
print("\n2. focus_on auf einen BEWEGTEN koerper endet exakt")
# =====================================================================

# 7.7 km/s ist LEO; mit sim_dt-raffung sieht die kamera ein vielfaches davon.
mover = FakeBody(1.0e9, 0.0, vx=0.0, vy=-3.0e7, name='Schnell')
cam = make_camera()
cam.position = Vec2(0.0, 0.0)
cam.target_position = Vec2(0.0, 0.0)
cam.focus_on(mover)

for _ in range(1200):
    mover.advance(DT)
    cam.update(DT, ui_wants_keyboard=True)

residual = (mover.position - cam.position).magnitude()
check(residual == 0.0,
      "kein bleibender rueckstand: die kamera sitzt EXAKT auf dem koerper",
      f"{residual} m")

# Gegenprobe: die naheliegende umsetzung -- die absolute position auf das
# bewegte ziel glaetten -- behaelt v/k. Ohne diese zeile bewiese der test
# oben nur, dass die glaettung ueberhaupt konvergiert.
naive_body = FakeBody(1.0e9, 0.0, vx=0.0, vy=-3.0e7)
naive_pos = Vec2(0.0, 0.0)
rate = cam.focus_smoothing
for _ in range(1200):
    naive_body.advance(DT)
    alpha = 1.0 - math.exp(-rate * DT)
    naive_pos += (naive_body.position - naive_pos) * alpha
naive_residual = (naive_body.position - naive_pos).magnitude()
expected = naive_body.velocity.magnitude() / rate
check(naive_residual > 1e6 and abs(naive_residual - expected) / expected < 0.05,
      "gegenprobe: die naive glaettung haengt um v/k hinterher",
      f"{px(naive_residual):.0f} px (v/k = {px(expected):.0f} px)")


# =====================================================================
print("\n3. focus_smoothing gilt nur waehrend des anflugs")
# =====================================================================

def settle_frames(rate_attr_value):
    b = FakeBody(1.0e9, 0.0)
    c = make_camera()
    c.position = Vec2(0.0, 0.0)
    c.target_position = Vec2(0.0, 0.0)
    c.focus_smoothing = rate_attr_value
    c.focus_on(b)
    for i in range(5000):
        c.update(DT, ui_wants_keyboard=True)
        if not c._focus_active:
            return i + 1
    return None


slow = settle_frames(4.5)
fast = settle_frames(20.0)
check(slow is not None and fast is not None and slow > fast * 3.5,
      "die langsamere rate braucht entsprechend laenger",
      f"{slow} frames bei 4.5 gegen {fast} bei 20.0")

b = FakeBody(1.0e9, 0.0)
cam = make_camera()
cam.position = Vec2(0.0, 0.0)
cam.target_position = Vec2(0.0, 0.0)
cam.focus_on(b)
check(cam._focus_active, "waehrend des anflugs ist die focus-rate aktiv")
for _ in range(2000):
    cam.update(DT, ui_wants_keyboard=True)
check(not cam._focus_active,
      "nach der ankunft ist pan_smoothing wieder zustaendig")


# =====================================================================
print("\n4. der zoom bewegt die kamera ueberhaupt nicht")
# =====================================================================

CURSOR = (200.0, 150.0)          # deutlich abseits der bildmitte
ORBIT_V = -3.0e7                 # bahngeschwindigkeit x zeitraffer


def zoom_sweep(cam, body, notch):
    """Schnelles rein/raus-zoomen ueber 240 frames, koerper faehrt mit.

    Gibt die groesste abweichung des koerpers von der BILDMITTE zurueck --
    genau das, was der spieler als 'die kamera holt das schiff ein' sieht.
    """
    worst = 0.0
    for i in range(240):
        if i % 3 == 0:
            notch(cam, i)
        body.advance(DT)
        cam.update(DT, ui_wants_keyboard=True)
        sx, sy = cam.world_to_screen(body.position)
        worst = max(worst, math.hypot(sx - W * 0.5, sy - H * 0.5))
    return worst


followed = FakeBody(3.0e9, 1.0e9, vy=ORBIT_V, name='Verfolgt')
cam = make_camera()
cam.follow(followed)
cam.snap_to_targets()
worst = zoom_sweep(cam, followed,
                   lambda c, i: c.zoom_by(1.5 ** (1 if (i // 30) % 2 == 0 else -1)))
check(worst == 0.0,
      "beim schnellen zoomen steht der verfolgte koerper EXAKT still",
      f"groesste abweichung von der mitte {worst} px")
check(cam.target is followed, "und die verfolgung besteht weiter")

# Gegenprobe: das alte zeiger-ankern. Es verschiebt bei jeder raste den
# versatz zum koerper, und der geglaettete versatz laeuft dem nach -- das ist
# das 'einholen'. Hier nachgebaut, weil der mechanismus (follow_offset) mit
# genau dieser aenderung entfallen ist.
legacy = FakeBody(3.0e9, 1.0e9, vy=ORBIT_V)
lcam = make_camera()
lcam.follow(legacy)
lcam.snap_to_targets()
legacy_offset = Vec2(0.0, 0.0)          # entspricht follow_offset
legacy_render = Vec2(0.0, 0.0)          # entspricht _render_follow_offset
legacy_worst = 0.0
for i in range(240):
    if i % 3 == 0:
        old_scale = lcam.target_scale
        new_scale = old_scale * (1.5 ** (1 if (i // 30) % 2 == 0 else -1))
        base = legacy.position + legacy_offset
        anchor_world = lcam._screen_to_world_with(CURSOR, base, old_scale)
        legacy_offset += Vec2(
            anchor_world.x - (CURSOR[0] - W * 0.5) / new_scale - base.x,
            anchor_world.y + (CURSOR[1] - H * 0.5) / new_scale - base.y)
        lcam.target_scale = new_scale
    legacy.advance(DT)
    lcam.update(DT, ui_wants_keyboard=True)      # nur fuer die skala-glaettung
    a = 1.0 - math.exp(-lcam.pan_smoothing * DT)
    legacy_render += (legacy_offset - legacy_render) * a
    legacy_worst = max(legacy_worst, legacy_render.magnitude() * lcam.scale)
check(legacy_worst > 50.0,
      "gegenprobe: mit zeiger-ankern laeuft der koerper aus der mitte",
      f"bis zu {legacy_worst:.0f} px daneben")

# Und die freie kamera wird vom zoom ebenfalls nicht verschoben.
cam_free = make_camera()
cam_free.position = Vec2(7.0e8, -2.0e8)
cam_free.target_position = cam_free.position.copy()
before_free = cam_free.target_position.copy()
for _ in range(12):
    cam_free.zoom_by(1.5)
check((cam_free.target_position - before_free).magnitude() == 0.0,
      "ohne verfolgung bleibt das kamera-ziel beim zoomen unveraendert")


# =====================================================================
print("\n5. ein schwenk loest die verfolgung und laesst die kamera stehen")
# =====================================================================

runner = FakeBody(1.0e9, 0.0, vy=ORBIT_V, name='Laeufer')
cam = make_camera()
cam.follow(runner)
cam.snap_to_targets()
start = cam.position.copy()

PAN = Vec2(2.0e8, 0.0)
cam._shift_target_position(PAN)
check(cam.target is None, "der erste schwenk loest die verfolgung")
check(cam.position.x == start.x and cam.position.y == start.y,
      "und erzeugt dabei keinen sprung",
      f"({cam.position.x - start.x}, {cam.position.y - start.y})")

# Mehrfach schwenken darf sich addieren -- das loesen passiert nur einmal.
cam._shift_target_position(PAN)
cam._shift_target_position(PAN)
expected = start + PAN * 3.0
check((cam.target_position - expected).magnitude() == 0.0,
      "weitere schwenks addieren sich, statt das ziel zurueckzusetzen")

for _ in range(900):
    runner.advance(DT)
    cam.update(DT, ui_wants_keyboard=True)

check((cam.position - expected).magnitude() == 0.0,
      "die kamera kommt zur ruhe und bleibt im weltraum stehen",
      f"{px((cam.position - expected).magnitude()):.3f} px vom ziel")
# Gegenprobe: der koerper ist in derselben zeit weit weggeflogen. Ohne das
# waere 'die kamera steht still' trivial erfuellt -- ein stehender koerper
# haette dieselbe messung bestanden.
flown_px = px((runner.position - start).magnitude())
check(flown_px > 50.0,
      "gegenprobe: der koerper ist derweil aus dem bild geflogen",
      f"{flown_px:.0f} px in 5 s")

# Ziehen geht denselben weg -- und zieht 1:1 mit, ohne gummiband.
drag_body = FakeBody(1.0e9, 0.0, vy=ORBIT_V)
cam = make_camera()
cam.follow(drag_body)
cam.snap_to_targets()
cam._begin_drag((400.0, 250.0))
cam._update_drag((460.0, 250.0), DT)
check(cam.target is None, "ziehen loest die verfolgung ebenfalls")
moved_px = (cam.position - (drag_body.position)).magnitude() * cam.scale
check(abs(moved_px - 60.0) < 1e-6,
      "und verschiebt die ansicht um genau den zeigerweg",
      f"{moved_px:.4f} px auf 60 px zeigerweg")

# Home holt zurueck -- geglaettet, und exakt.
cam.set_home_body(drag_body)
cam._end_drag()
cam.pan_inertia_enabled = False
cam._pan_velocity.clear()
cam.recentre()
check(cam.target is drag_body and cam._focus_active,
      "Home heftet wieder an und startet einen anflug")
for _ in range(900):
    drag_body.advance(DT)
    cam.update(DT, ui_wants_keyboard=True)
check((cam.position - drag_body.position).magnitude() == 0.0,
      "und landet exakt auf dem koerper")


# =====================================================================
print("\n6. auswahl ist ANSICHT, nicht bezug")
# =====================================================================

from ui.state import UIState

bodies = [FakeBody(0.0, 0.0, name='Sonne'),
          FakeBody(1.5e11, 0.0, name='Erde'),
          FakeBody(1.5e11, 4.0e8, name='Mond')]
for b in bodies:
    b.is_moon_of = None

changes = []
state = UIState(bodies, initial_reference_index=1,
                on_change=lambda s: changes.append(s.reference_index))

check(state.selected_index is None and state.selected_body is None,
      "am anfang ist nichts ausgewaehlt")
check(state.select_body(2) is True, "auswaehlen meldet die aenderung")
check(state.selected_body is bodies[2], "selected_body zeigt auf den koerper")
check(state.reference_index == 1,
      "der bezugskoerper bleibt unveraendert", f"{state.reference_index}")
check(changes == [],
      "die auswahl loest KEIN on_change aus (kein frame-/predictor-neuaufbau)",
      f"{changes}")
check(state.select_body(2) is False, "dieselbe auswahl noch einmal ist ein no-op")
check(state.clear_selection() is True and state.selected_index is None,
      "aufheben leert die auswahl")
check(state.select_body(99) is False and state.selected_index is None,
      "ein index ausserhalb der liste waehlt nichts aus")

# Gegenprobe: der bezugskoerper meldet sehr wohl.
state.set_reference_index(0)
check(changes == [0], "gegenprobe: ein bezugswechsel meldet weiterhin",
      f"{changes}")


# =====================================================================
print("\n7.-9. gegen echte pixel")
# =====================================================================

import moderngl
import pygame
from pygame.locals import DOUBLEBUF, OPENGL

# Nur display+font -- pygame.init() zaehlt mixer- und joystick-geraete auf
# und kostet dabei ~45 s. Siehe test.py.
pygame.display.init()
pygame.font.init()
pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL, vsync=0)
gl = moderngl.create_context()
gl.enable(moderngl.BLEND)
gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

from loader import ConfigLoader, SystemLoader
from reference_frames import (
    BODY_CENTRED_BODY_DIRECTION,
    BODY_CENTRED_NON_ROTATING,
    PlottingFrameAdapter,
    ReferenceFrameSelector,
)
from rendering import Renderer
from world import world as World

config = ConfigLoader()
world = World(float(config.get('physics.gravitational_constant', 6.6730831e-11)))
world.body = SystemLoader(config.get('simulation.system_file',
                                     'solar_system.json')).load()
config.apply_to_world(world)
world.update_planets(0.0)

renderer = Renderer(W, H, enable_fxaa=False, ctx=gl)
config.apply_to_renderer(renderer)
camera = Camera(None, W, H)
config.apply_to_camera(camera)
camera.pan_inertia_enabled = False

adapter = PlottingFrameAdapter(renderer, world.body)
selector = ReferenceFrameSelector(
    lambda params, target_body_index, target_reference_index:
        adapter.update_plotting_frame(
            params,
            target_body_index=target_body_index,
            target_reference_index=target_reference_index,
        )
)

ERDE_I = next(i for i, b in enumerate(world.body) if b.name == 'Erde')
MOND_I = next(i for i, b in enumerate(world.body) if b.name == 'Mond')
ERDE = world.body[ERDE_I]
MOND = world.body[MOND_I]


def read_pixels():
    data = gl.screen.read(viewport=(0, 0, W, H), components=3, dtype='f1')
    return np.frombuffer(data, dtype=np.uint8).reshape(H, W, 3)[::-1]


def draw(selected=None, real_dt=0.0, settle=False):
    """Einen frame zeichnen.

    Die koerper-optik wird NEBENLAEUFIG gebaut, ein einzelner frame waere
    also ein wettlauf. `settle` wartet auf das ergebnis statt auf eine feste
    zeit -- dieselbe vorgehensweise wie in body_style_test.
    """
    frames = 400 if settle else 1
    for _ in range(frames):
        gl.screen.use()
        gl.clear(0.0, 0.0, 0.0, 1.0)
        renderer.render(world.body, camera, None, sim_time=world.time,
                        real_dt=real_dt, selected_body=selected)
        if settle and not renderer._body_style_jobs:
            break
    return read_pixels()


def look_at(subject, radius_px, offset_px=(0.0, 0.0)):
    """Kamera so stellen, dass `subject` mit gegebenem radius dort erscheint.

    Der versatz ist in BILDSCHIRM-pixeln vom mittelpunkt aus angegeben und
    wird ueber die skala in weltmeter zurueckgerechnet -- unabhaengig vom
    plot-frame, weil kamera und koerper durch dieselbe transformation gehen.
    """
    camera.scale = float(radius_px) / float(subject.radius)
    camera.target_scale = camera.scale
    camera.target = None
    dx = -float(offset_px[0]) / camera.scale
    dy = float(offset_px[1]) / camera.scale
    camera.position = Vec2(float(subject.position.x) + dx,
                           float(subject.position.y) + dy)
    camera.target_position = camera.position.copy()


def blob_centroid(image, around, window):
    """Flaechen-schwerpunkt der beschriebenen pixel in einem fenster (top-down).

    UNGEWICHTET, mit absicht: eine helligkeits-gewichtung zoege den
    schwerpunkt auf die beleuchtete seite (gemessen 18 px bei 60 px radius)
    und wuerde die messung ueber den terminator statt ueber die geometrie
    fuehren. Das fenster ist so eng gewaehlt, dass die beschriftung ueber dem
    koerper nicht hineinfaellt.
    """
    cx, cy = int(round(around[0])), int(round(around[1]))
    x0, x1 = max(0, cx - window), min(W, cx + window)
    y0, y1 = max(0, cy - window), min(H, cy + window)
    patch = image[y0:y1, x0:x1].astype(np.int32).sum(axis=2)
    mask = patch > 24
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return (x0 + float(xs.mean()), y0 + float(ys.mean()))


print("\n7. pick_body trifft die GEZEICHNETE stelle")

SONNE_I = next(i for i, b in enumerate(world.body) if b.name == 'Sonne')

# Der zweite fall ist mit absicht NICHT "richtungsrahmen auf Erde, Erde
# angeklickt": in einem rahmen, dessen ursprung der koerper selbst ist,
# faellt jede drehung heraus, und der test waere buchstaeblich dieselbe
# rechnung wie im nicht-rotierenden fall (gemessen: pixelgleich). Getestet
# wird deshalb ein koerper, den der rahmen wirklich bewegt -- Erde in einem
# Sonne-Erde-richtungsrahmen, nach einem halben jahr weltzeit.
CASES = (
    ("nicht-rotierend", lambda: selector.set_to_body_non_rotating(ERDE_I)),
    ("richtungsrahmen", lambda: selector.set_to_body_direction(SONNE_I, ERDE_I)),
)

world.update_planets(0.5 * 365.25 * 86400.0)

for mode, setup in CASES:
    setup()

    RADIUS_PX = 60.0
    OFFSET = (150.0, -80.0)
    look_at(ERDE, RADIUS_PX, offset_px=OFFSET)
    renderer.set_frame_time(world.time)

    # Wo der rahmen den koerper hinlegt. Bei drehung ist das NICHT
    # mitte+versatz, deshalb wird es gerechnet statt angenommen.
    computed = renderer._world_to_screen_xy(
        float(ERDE.position.x), float(ERDE.position.y), camera,
        camera_frame_xy=renderer._frame_camera_xy(camera))

    image = draw(settle=True)
    centre = blob_centroid(image, computed, window=int(RADIUS_PX) + 10)
    check(centre is not None, f"[{mode}] der koerper ist im bild")
    if centre is None:
        continue

    # DAS ist der eigentliche punkt: die gezeichnete scheibe und die stelle,
    # gegen die pick_body rechnet, sind derselbe ort. Ohne diese messung
    # pruefte der treffer unten nur, dass eine funktion mit sich selbst
    # uebereinstimmt.
    offset = math.hypot(centre[0] - computed[0], centre[1] - computed[1])
    check(offset < 5.0,
          f"[{mode}] der pixel-schwerpunkt liegt auf der gerechneten position",
          f"{offset:.2f} px")

    hit = renderer.pick_body(centre, world.body, camera)
    check(hit == ERDE_I,
          f"[{mode}] ein klick auf den PIXEL-schwerpunkt trifft Erde",
          f"index {hit} bei ({centre[0]:.1f}, {centre[1]:.1f})")

    # Gegenprobe: die naheliegende, rahmen-BLINDE rechnung. Im rotierenden
    # rahmen liegt sie weit daneben -- genau deshalb geht pick_body durch
    # _world_to_screen_xy und nicht durch camera.world_to_screen.
    naive = camera.world_to_screen(ERDE.position)
    naive_off = math.hypot(naive[0] - centre[0], naive[1] - centre[1])
    if mode == "richtungsrahmen":
        check(naive_off > 50.0,
              f"[{mode}] gegenprobe: die rahmen-blinde rechnung liegt daneben",
              f"{naive_off:.0f} px")
    else:
        check(naive_off < 5.0,
              f"[{mode}] ohne drehung stimmen beide rechnungen ueberein",
              f"{naive_off:.2f} px")

    grab = (renderer._pick_radius_px(ERDE, camera)
            + renderer.ui_px(renderer.selection_pick_margin_px))
    far = (computed[0] + grab + 3.0, computed[1])
    check(renderer.pick_body(far, world.body, camera) != ERDE_I,
          f"[{mode}] 3 px jenseits des greifradius trifft Erde nicht mehr",
          f"greifradius {grab:.1f} px")

# Ueberdeckung: der naechste MITTELPUNKT gewinnt, nicht der groesste treffer.
# Mit zwei gestellten koerpern statt mit dem echten system -- die zoomstufe,
# bei der ein mond vor seinem planeten steht, laesst sich mit den echten
# bahnradien nicht herstellen, ohne dass beide zu icons schrumpfen.
selector.set_to_body_non_rotating(ERDE_I)
look_at(ERDE, 200.0)
s = camera.scale
gross = FakeBody(ERDE.position.x, ERDE.position.y,
                 radius=300.0 / s, name='Gross')
klein = FakeBody(ERDE.position.x + 120.0 / s, ERDE.position.y,
                 radius=10.0 / s, name='Klein')
paar = [gross, klein]
check(renderer.pick_body((W * 0.5 + 120.0, H * 0.5), paar, camera) == 1,
      "auf dem kleinen koerper gewinnt der kleine, obwohl beide getroffen sind")
check(renderer.pick_body((W * 0.5 + 250.0, H * 0.5), paar, camera) == 0,
      "gegenprobe: abseits davon gewinnt wieder der grosse")


print("\n8. die markierung liegt als vier pfeile um den koerper")

look_at(ERDE, 40.0)
plain = draw(settle=True)
plain_again = draw()
check(np.array_equal(plain, plain_again),
      "der frame ist ohne auswahl von sich aus stabil")

marked = draw(selected=ERDE)
diff = (np.abs(marked.astype(np.int16) - plain.astype(np.int16)).sum(axis=2) > 12)
check(diff.any(), "mit auswahl aendern sich pixel",
      f"{int(diff.sum())} pixel")

ys, xs = np.nonzero(diff)
radii = np.hypot(xs - W * 0.5, ys - H * 0.5)
body_r = renderer._pick_radius_px(ERDE, camera)
reach = (body_r + renderer.ui_px(renderer.selection_gap_px
                                 + renderer.selection_arrow_length_px) + 3.0)

ring = (radii >= body_r - 3.0) & (radii <= reach)
check(ring.any(), "pfeil-pixel liegen im ring um den koerper",
      f"{int(ring.sum())} von {int(diff.sum())} "
      f"(koerper {body_r:.1f}, reichweite {reach:.1f})")

# Was ausserhalb des rings liegt, darf NUR die beschriftung sein: sie rueckt
# hoch, damit der obere pfeil nicht in ihr steht. Also alles oberhalb des
# koerpers, nichts daneben oder darunter.
outside = ~ring
lift = renderer.selection_label_lift_px(ERDE)
check(not outside.any()
      or (ys[outside].max() < H * 0.5 - body_r
          and radii[outside].max() < reach + lift + 40.0),
      "ausserhalb des rings aendert sich nur die (hochgerueckte) beschriftung",
      f"{int(outside.sum())} pixel, hoechstens y={ys[outside].max() if outside.any() else '-'}")

# Bei phase 0 stehen die pfeile auf den achsen; jede der vier richtungen
# muss ink tragen, sonst waere es kein vierfach-marker.
angles = np.degrees(np.arctan2(ys[ring] - H * 0.5, xs[ring] - W * 0.5)) % 360.0
sectors = [((angles < 45.0) | (angles >= 315.0)),
           ((angles >= 45.0) & (angles < 135.0)),
           ((angles >= 135.0) & (angles < 225.0)),
           ((angles >= 225.0) & (angles < 315.0))]
counts = [int(s.sum()) for s in sectors]
check(all(c > 0 for c in counts),
      "alle vier richtungen tragen pfeil-pixel", f"{counts}")

# Und der obere pfeil steht wirklich frei. Gemessen wird weisse schrift im
# streifen, den der obere pfeil belegt: ohne auswahl steht dort das label
# (sonst bewiese die messung nichts), mit auswahl ist der streifen leer.
def white_pixels(image, y0, y1, x0, x1):
    patch = image[int(y0):int(y1), int(x0):int(x1)]
    return int((patch.min(axis=2) > 180).sum())


band = (H * 0.5 - 58, H * 0.5 - 44, W * 0.5 - 40, W * 0.5 + 40)
check(white_pixels(plain, *band) > 0,
      "gegenprobe: ohne auswahl steht die beschriftung im pfeil-streifen",
      f"{white_pixels(plain, *band)} px")
check(white_pixels(marked, *band) == 0,
      "mit auswahl ist der streifen des oberen pfeils frei von schrift",
      f"{white_pixels(marked, *band)} px, angehoben um {lift:.1f} px")

renderer.selection_marker_enabled = False
off = draw(selected=ERDE)
check(np.array_equal(off, plain),
      "abgeschaltet ist der frame BIT-IDENTISCH zum frame ohne auswahl")
renderer.selection_marker_enabled = True


print("\n9. drehung und puls haengen nicht an der bildrate")

renderer._selection_spin_phase = 0.0
renderer._selection_pulse_phase = 0.0
renderer._advance_selection_phases(0.1)
coarse = (renderer._selection_spin_phase, renderer._selection_pulse_phase)

renderer._selection_spin_phase = 0.0
renderer._selection_pulse_phase = 0.0
for _ in range(10):
    renderer._advance_selection_phases(0.01)
fine = (renderer._selection_spin_phase, renderer._selection_pulse_phase)

check(abs(coarse[0] - fine[0]) < 1e-9 and abs(coarse[1] - fine[1]) < 1e-9,
      "ein schritt von 0.1 s == zehn schritte von 0.01 s",
      f"{coarse} gegen {fine}")

renderer._selection_spin_phase = 0.0
renderer._selection_pulse_phase = 0.0
still = renderer._selection_marker_vertices(400.0, 250.0, 40.0)
renderer._advance_selection_phases(0.4)
moved = renderer._selection_marker_vertices(400.0, 250.0, 40.0)
shift = max(math.hypot(a[0] - b[0], a[1] - b[1])
            for a, b in zip(still, moved))
check(shift > 1.0, "die pfeile bewegen sich ueberhaupt", f"{shift:.2f} px")

# Der puls bleibt "leicht": ueber eine ganze periode darf der ring nicht
# mehr als ein paar pixel atmen, sonst wackelt die markierung.
renderer._selection_spin_phase = 0.0
extents = []
for k in range(24):
    renderer._selection_pulse_phase = 2.0 * math.pi * k / 24.0
    v = renderer._selection_marker_vertices(400.0, 250.0, 40.0)
    extents.append(max(math.hypot(p[0] - 400.0, p[1] - (H - 250.0)) for p in v))
swing = max(extents) - min(extents)
check(0.5 < swing < 8.0, "der puls atmet sichtbar, aber leicht",
      f"{swing:.2f} px")


# =====================================================================
print()
if FAILURES:
    print(f"{len(FAILURES)} FEHLER")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("alle pruefungen bestanden")
