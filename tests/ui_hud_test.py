"""Regressionstest des spieler-HUDs (Phase 4).

Drei ebenen, bewusst getrennt:

1. **Bahnelemente gegen ANALYTISCH bekannte bahnen.** Eine kreisbahn und
   eine ellipse mit von hand gerechneten sollwerten -- das ist die einzige
   art, die kepler-loesung zu pruefen, ohne sie noch einmal aufzuschreiben.
2. **Layout gegen echte fenstergroessen.** Alle gruppen muessen im bild
   liegen und duerfen sich nicht ueberlappen, breit wie schmal.
3. **Bedienung gegen den simulationszustand.** Ein klick auf einen
   zeitraffer-knopf muss camera.sim_dt aendern, nicht nur den knopf.

Aufruf: python tests/ui_hud_test.py
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
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE

from vec import Vec2, G
from ui.hud.telemetry import (
    OrbitalElements,
    compass_from_frame_direction,
    compass_from_theta,
)

FAILURES = []


def check(condition, label, detail=''):
    if condition:
        print(f"  ok   {label}{(' -- ' + detail) if detail else ''}")
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label} -- {detail}")


def close(a, b, tol=1e-6, label=''):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))


class Stub:
    def __init__(self, px, py, vx, vy, mass=0.0, radius=0.0):
        self.position = Vec2(px, py)
        self.velocity = Vec2(vx, vy)
        self.mass = mass
        self.radius = radius


# ═══════════════════════════════════════════ 1. bahnelemente (analytisch)

print("1. Bahnelemente gegen bekannte bahnen")

M_EARTH = 5.9722e24
MU = G * M_EARTH

# --- kreisbahn: e = 0, ap = pe = r, T = 2*pi*sqrt(r^3/mu) -----------------
R_CIRC = 7.0e6
V_CIRC = math.sqrt(MU / R_CIRC)
central = Stub(0.0, 0.0, 0.0, 0.0, mass=M_EARTH, radius=6.371e6)
ship = Stub(R_CIRC, 0.0, 0.0, V_CIRC)

elements = OrbitalElements().solve(ship, central, G)
check(elements.valid and elements.closed, 'kreisbahn erkannt als geschlossen')
check(abs(elements.eccentricity) < 1e-9, 'kreisbahn: exzentrizitaet 0',
      f"e = {elements.eccentricity:.3e}")
check(close(elements.apoapsis, R_CIRC, 1e-9), 'kreisbahn: AP = r',
      f"{elements.apoapsis:.6e} vs {R_CIRC:.6e}")
check(close(elements.periapsis, R_CIRC, 1e-9), 'kreisbahn: PE = r',
      f"{elements.periapsis:.6e} vs {R_CIRC:.6e}")
period_expected = 2.0 * math.pi * math.sqrt(R_CIRC ** 3 / MU)
check(close(elements.period, period_expected, 1e-9), 'kreisbahn: umlaufzeit',
      f"{elements.period:.3f}s vs {period_expected:.3f}s")

# --- ellipse: start im PERIAPSIS -----------------------------------------
RP = 7.0e6
RA = 1.4e7
A = 0.5 * (RP + RA)
ECC = (RA - RP) / (RA + RP)
VP = math.sqrt(MU * (1.0 + ECC) / RP)
ship = Stub(RP, 0.0, 0.0, VP)

elements = OrbitalElements().solve(ship, central, G)
check(close(elements.eccentricity, ECC, 1e-9), 'ellipse: exzentrizitaet',
      f"{elements.eccentricity:.9f} vs {ECC:.9f}")
check(close(elements.periapsis, RP, 1e-9), 'ellipse: PE',
      f"{elements.periapsis:.6e} vs {RP:.6e}")
check(close(elements.apoapsis, RA, 1e-9), 'ellipse: AP',
      f"{elements.apoapsis:.6e} vs {RA:.6e}")
check(close(elements.semi_major_axis, A, 1e-9), 'ellipse: grosse halbachse',
      f"{elements.semi_major_axis:.6e} vs {A:.6e}")
# Im periapsis ist die zeit bis zum apoapsis genau eine halbe umlaufzeit.
check(close(elements.time_to_apoapsis, elements.period * 0.5, 1e-6),
      'ellipse: T-AP im periapsis = halbe umlaufzeit',
      f"{elements.time_to_apoapsis:.3f}s vs {elements.period * 0.5:.3f}s")

# --- dieselbe ellipse, start im APOAPSIS ----------------------------------
VA = math.sqrt(MU * (1.0 - ECC) / RA)
ship = Stub(-RA, 0.0, 0.0, -VA)
elements = OrbitalElements().solve(ship, central, G)
check(close(elements.apoapsis, RA, 1e-9), 'apoapsis-start: AP unveraendert',
      f"{elements.apoapsis:.6e}")
check(elements.time_to_apoapsis is not None
      and (elements.time_to_apoapsis < 1e-3
           or abs(elements.time_to_apoapsis - elements.period) < 1e-3),
      'apoapsis-start: T-AP ist 0 (bzw. eine volle umlaufzeit)',
      f"{elements.time_to_apoapsis:.6f}s von {elements.period:.1f}s")

# --- hyperbel -------------------------------------------------------------
ship = Stub(R_CIRC, 0.0, 0.0, V_CIRC * 1.6)
elements = OrbitalElements().solve(ship, central, G)
check(elements.valid and not elements.closed, 'fluchtbahn ist nicht geschlossen')
check(elements.eccentricity > 1.0, 'fluchtbahn: exzentrizitaet > 1',
      f"e = {elements.eccentricity:.4f}")
check(elements.period is None and elements.apoapsis is None,
      'fluchtbahn hat weder umlaufzeit noch apoapsis')

# --- entartete eingaben ---------------------------------------------------
check(not OrbitalElements().solve(None, central, G).valid, 'kein schiff -> ungueltig')
check(not OrbitalElements().solve(ship, None, G).valid, 'kein bezugskoerper -> ungueltig')

print()
print("2. Kurs-umrechnung")
check(abs(compass_from_theta(0.0) - 90.0) < 1e-9,
      'theta 0 (nase nach rechts) = kurs 090')
check(abs(compass_from_theta(math.pi / 2) - 0.0) < 1e-9,
      'theta 90 (nase nach oben) = kurs 000')
check(abs(compass_from_theta(math.pi) - 270.0) < 1e-9,
      'theta 180 (nase nach links) = kurs 270')
# Eine frame-richtung muss denselben kurs liefern wie das theta, das
# _apply_orientation_snap daraus errechnet -- sonst stehen ring-marker und
# schiffsnase nicht uebereinander.
check(abs(compass_from_frame_direction(Vec2(1.0, 0.0)) - 90.0) < 1e-9,
      'frame-richtung +x = kurs 090')
check(abs(compass_from_frame_direction(Vec2(0.0, -1.0)) - 0.0) < 1e-9,
      'frame-richtung -y (bildschirm oben) = kurs 000')
check(compass_from_frame_direction(None) is None, 'None-richtung -> None')

# ═══════════════════════════════════════════════ 3. layout und bedienung

print()
print("3. Layout und bedienung (echtes GL-fenster)")

W, H = 1280, 800
# pygame.init() waere hier dasselbe wie im spiel: mixer und joystick
# zaehlen geraete auf und kosten zusammen ~45 s. Siehe test.py.
pygame.display.init()
pygame.font.init()
pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL | RESIZABLE, vsync=0)
gl = moderngl.create_context()
gl.enable(moderngl.BLEND)
gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

from camera import Camera
from loader import ConfigLoader, SystemLoader
from predictor import Predictor
from rendering import Renderer
from schiff import schiffcontrol
from ui import UIContext, UIRoot, UIState
from ui.hud import Hud
from ui.hud.layout import WARP_STEPS
from world import world as World

config = ConfigLoader(None)
config.load()
bodies = SystemLoader("solar_system.json").load()
world = World(G)
world.body = bodies
config.apply_to_world(world)
sim_ship = next(b for b in bodies if b.is_ship)
earth = next(b for b in bodies if b.name == "Erde")
control = schiffcontrol(sim_ship)
config.apply_to_ship_control(control)

camera = Camera(None, W, H)
config.apply_to_camera(camera)
camera.follow(sim_ship)
camera.snap_to_targets()
renderer = Renderer(W, H, enable_fxaa=False, ctx=gl)
config.apply_to_renderer(renderer)
predictor = Predictor(recompute_every_update=True, **config.predictor_kwargs())
config.apply_to_predictor(predictor)

state = UIState(world.body, initial_reference_index=world.body.index(earth))
ui = UIContext(gl, W, H, ui_scale=renderer.ui_scale)
root = UIRoot(ui)
hud = Hud(root, world, sim_ship, control, camera, renderer, predictor, state,
          tick_rate=60.0)


def frame(width=None, height=None):
    if width is not None:
        root.resize(width, height, ui_scale=1.0)
    hud.update()
    root.begin_frame(1 / 60)


frame()

GROUPS = {
    'badge': hud.badge, 'warp': hud.warp, 'palette': hud.palette_button,
    'koerperliste': hud.body_browser, 'elements': hud.elements,
    'target': hud.target, 'throttle': hud.throttle, 'frames': hud.frames,
    'ring': hud.ring, 'snaps': hud.snaps, 'zoom': hud.zoom,
}

for name, widget in GROUPS.items():
    r = widget.rect
    inside = (r.x >= -0.5 and r.y >= -0.5
              and r.right <= W + 0.5 and r.bottom <= H + 0.5)
    check(inside, f"'{name}' liegt im bild", f"{r}")

# Ueberlappungen zwischen den gruppen. Der ring darf die rahmenwahl nicht
# beruehren, die seitenpanels nicht die mitte, usw.
names = list(GROUPS)
for i, a_name in enumerate(names):
    for b_name in names[i + 1:]:
        a = GROUPS[a_name].rect
        b = GROUPS[b_name].rect
        overlap = (a.x < b.right and b.x < a.right
                   and a.y < b.bottom and b.y < a.bottom)
        if overlap:
            check(False, f"'{a_name}' ueberlappt '{b_name}'", f"{a} / {b}")
check(not any(f.startswith("'") and 'ueberlappt' in f for f in FAILURES),
      'keine gruppe ueberlappt eine andere')

print()
print("3b. Geschwindigkeit skriptgefuehrter koerper")
# world.update_planets() setzt bei Kepler-koerpern NUR die position -- ihr
# velocity-feld bleibt auf dem ladewert stehen. Wer es direkt liest, rechnet
# die bahnelemente gegen einen stillstehenden planeten: eine saubere
# kreisbahn um die Erde kam so als hyperbel mit e = 4.4 heraus.
for _ in range(3):
    world.update_dynamics(60.0)
    world.update_planets(60.0)

check(getattr(earth, 'scripted_orbit', False),
      'Erde ist ein skriptgefuehrter koerper (sonst prueft dieser test nichts)')
stored = math.hypot(float(earth.velocity.x), float(earth.velocity.y))
derived = hud.telemetry.body_velocity(earth)
derived_speed = math.hypot(*derived)
check(derived_speed > 1.0e3,
      'abgeleitete Erd-geschwindigkeit ist die echte bahngeschwindigkeit',
      f"{derived_speed:.1f} m/s (gespeichertes feld: {stored:.1f} m/s)")

# Gegenprobe: die abgeleitete geschwindigkeit muss zur tatsaechlichen
# positionsaenderung passen.
p0 = earth.position_at_time(world.time)
p1 = earth.position_at_time(world.time + 10.0)
measured = math.hypot(float(p1.x) - float(p0.x), float(p1.y) - float(p0.y)) / 10.0
check(abs(derived_speed - measured) / max(measured, 1.0) < 1e-3,
      'stimmt mit der positionsaenderung ueberein',
      f"{derived_speed:.3f} vs {measured:.3f} m/s")

# Und die entscheidende folge: eine gestellte kreisbahn muss auch als solche
# erkannt werden, nicht als fluchtbahn.
mu_earth = G * earth.mass
r_test = float(earth.radius) + 4.0e5
v_test = math.sqrt(mu_earth / r_test)
probe = Stub(float(earth.position.x) + r_test, float(earth.position.y),
             derived[0], derived[1] + v_test)
elements = OrbitalElements().solve(probe, earth, G, reference_velocity=derived)
check(elements.closed and elements.eccentricity < 1e-3,
      'kreisbahn um einen skriptgefuehrten koerper wird als kreis erkannt',
      f"e = {elements.eccentricity:.2e}, geschlossen = {elements.closed}")
naive = OrbitalElements().solve(probe, earth, G)   # ohne korrektur
check(not naive.closed or naive.eccentricity > 0.5,
      'ohne die korrektur waere dieselbe bahn eine fluchtbahn (der alte fehler)',
      f"e = {naive.eccentricity:.3f}, geschlossen = {naive.closed}")

print()
print("4. Responsive umschaltung")
frame(1280, 800)
check(hud._wide, 'bei 1280 breit: volles layout')
check(hud.elements.visible and not hud.elements_rail.visible,
      'breit: bahnelemente als panel, nicht als leiste')

frame(820, 620)
check(not hud._wide, 'bei 820 breit: kompaktes layout')
check(hud.elements_rail.visible and not hud.elements.visible,
      'schmal: bahnelemente als leiste')
check(hud.body_browser.visible, 'schmal: koerperliste bleibt erreichbar')
check(hud.throttle_compact.visible and not hud.throttle.visible,
      'schmal: schubregler hochkant')
for name, widget in (('leiste links', hud.elements_rail),
                     ('leiste rechts', hud.target_rail),
                     ('ring', hud.ring), ('snaps kompakt', hud.snaps_compact)):
    r = widget.rect
    check(r.x >= -0.5 and r.right <= 820.5 and r.y >= -0.5 and r.bottom <= 620.5,
          f"schmal: '{name}' liegt im bild", f"{r}")

frame(1280, 800)

print()
print("5. Bedienelemente wirken auf die simulation")


def click(widget, dx=0.5, dy=0.5):
    x = int(widget.rect.x + widget.rect.w * dx)
    y = int(widget.rect.y + widget.rect.h * dy)
    root._mouse_pos = (float(x), float(y))
    root.begin_frame(1 / 60)
    root.handle_event(pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, {'pos': (x, y), 'button': 1}))
    root.handle_event(pygame.event.Event(
        pygame.MOUSEBUTTONUP, {'pos': (x, y), 'button': 1}))
    return x, y


# --- zeitraffer -----------------------------------------------------------
before = camera.sim_dt
rects = hud.warp._option_rects(ui)
bx, by, bw, bh = rects[0]
root._mouse_pos = (bx + bw * 0.5, by + bh * 0.5)
root.begin_frame(1 / 60)
root.handle_event(pygame.event.Event(
    pygame.MOUSEBUTTONDOWN, {'pos': (int(bx + bw * 0.5), int(by + bh * 0.5)), 'button': 1}))
root.handle_event(pygame.event.Event(
    pygame.MOUSEBUTTONUP, {'pos': (int(bx + bw * 0.5), int(by + bh * 0.5)), 'button': 1}))
expected = WARP_STEPS[0][0] / 60.0   # sim-sekunden je tick
check(abs(camera.sim_dt - expected) < 1e-9,
      'klick auf die langsamste stufe setzt camera.sim_dt',
      f"{before} -> {camera.sim_dt} (erwartet {expected})")
frame()
check(hud._warp_index() == 0, 'zeitraffer-knopf liest den neuen wert zurueck',
      f"index {hud._warp_index()}")

# --- orientierungs-autopilot ---------------------------------------------
control.snap_mode = None
tx, ty, tw, th = hud.snaps._tile_rect(ui, 0)
root._mouse_pos = (tx + tw * 0.5, ty + th * 0.5)
root.begin_frame(1 / 60)
root.handle_event(pygame.event.Event(
    pygame.MOUSEBUTTONDOWN, {'pos': (int(tx + tw * 0.5), int(ty + th * 0.5)), 'button': 1}))
root.handle_event(pygame.event.Event(
    pygame.MOUSEBUTTONUP, {'pos': (int(tx + tw * 0.5), int(ty + th * 0.5)), 'button': 1}))
check(control.snap_mode == 'prograde', 'PRO-knopf rastet den autopiloten',
      f"snap_mode = {control.snap_mode}")

# --- schubregler ----------------------------------------------------------
maximum = hud.telemetry.thrust_max
hud.telemetry.set_thrust_level(0.5)
check(abs(control.thrust_acc - maximum * 0.5) < 1e-9,
      'schubstufe skaliert schiffcontrol.thrust_acc',
      f"{control.thrust_acc} von {maximum}")
hud.telemetry.set_thrust_level(2.0)
check(abs(control.thrust_acc - maximum) < 1e-9, 'schubstufe klemmt bei 100 %')
hud.telemetry.set_thrust_level(-1.0)
check(abs(control.thrust_acc) < 1e-9, 'schubstufe klemmt bei 0 %')

# --- rahmenwahl -----------------------------------------------------------
changes = []
state.on_change = lambda s: changes.append(s.view_mode())
hud._set_view_mode(0)
check(state.view_mode() == 0 and changes, 'SURFACE waehlt den body-direction-rahmen',
      f"modus {state.view_mode()}")
hud._set_view_mode(2)
check(state.target_overlay_enabled, 'TARGET schaltet das ziel-overlay ein')
hud._set_view_mode(1)
check(state.view_mode() == 1 and not state.target_overlay_enabled,
      'ORBITAL schaltet zurueck auf nicht rotierend')

# --- palettenknopf: aufklappen und ZEICHNEN -------------------------------
# Regression: die saetze in theme.PALETTE_SETS stehen als hex-ZEICHENKETTEN
# da (damit sie in theme.py lesbar bleiben), der zeichenpfad will aber
# float-tupel. Das popup starb deshalb beim oeffnen mit
# "could not convert string to float: '#'". Nur das oeffnen zu pruefen
# genuegt nicht -- der fehler schlaegt erst beim zeichnen zu.
hud.palette_button.open = True
frame()
try:
    gl.screen.use()
    gl.clear(0.0, 0.0, 0.0, 1.0)
    root.render()
    popup_ok, popup_error = True, ''
except Exception as exc:
    popup_ok, popup_error = False, f"{type(exc).__name__}: {exc}"
check(popup_ok, 'geoeffnete palettenauswahl laesst sich zeichnen', popup_error)

# Klick auf eine satz-zeile uebernimmt den satz und schliesst das popup.
hud.palette_button.open = True
frame()
rx, ry, rw, rh = hud.palette_button._set_row_rect(ui, 1)
row_pos = (int(rx + rw * 0.5), int(ry + rh * 0.5))
root._mouse_pos = (float(row_pos[0]), float(row_pos[1]))
root.begin_frame(1 / 60)
# DOWN und UP: UIRoot leitet ein loslassen nur an das widget weiter, das
# beim druecken aktiv wurde -- ein einzelnes MOUSEBUTTONUP verpufft.
root.handle_event(pygame.event.Event(
    pygame.MOUSEBUTTONDOWN, {'pos': row_pos, 'button': 1}))
root.handle_event(pygame.event.Event(
    pygame.MOUSEBUTTONUP, {'pos': row_pos, 'button': 1}))
check(ui.theme.palette.name == 'Ember', 'klick auf eine satz-zeile uebernimmt ihn',
      f"aktiv: {ui.theme.palette.name}")
check(not hud.palette_button.open, 'popup schliesst nach der auswahl')
check(all(isinstance(c, tuple) for c in ui.theme.palette.colors),
      'palette haelt nach dem wechsel float-tupel, keine zeichenketten')

# --- palette --------------------------------------------------------------
# Auf Baltic zuruecksetzen: der klick-test oben hat bereits auf Ember
# gestellt, sonst pruefte der wechsel unten gegen sich selbst.
ui.theme.set_palette_colors(('#22577a', '#38a3a5', '#57cc99', '#80ed99'), name='Baltic')
first = tuple(ui.theme.palette.colors[0])
ui.theme.set_palette_colors(('#3d2b56', '#c1462f', '#e0803c', '#f2c14e'), name='Ember')
check(tuple(ui.theme.palette.colors[0]) != first, 'palettenwechsel aendert die farben')
check(ui.theme.palette.name == 'Ember', 'palettenname uebernommen')
ring_color = ui.theme.palette.ring
check(max(ring_color[:3]) > 0.35,
      'aufgehellte rollenfarbe bleibt auf dunklem grund lesbar',
      f"ring = {tuple(round(c, 3) for c in ring_color)}")
ui.theme.set_palette_colors(('#22577a', '#38a3a5', '#57cc99', '#80ed99'), name='Baltic')

# --- kursbeschriftung weicht markern aus ---------------------------------
print()
print("6. Kursbeschriftung weicht den bahnmarkern aus")
hud.telemetry.marker_headings = {'prograde': 0.0, 'retrograde': 180.0,
                                 'normal_in': 90.0, 'antinormal_out': 270.0}
hidden = [deg for deg in (0, 90, 180, 270)
          if any(abs((deg - m + 180.0) % 360.0 - 180.0) < 16.0
                 for m in hud.telemetry.marker_headings.values())]
check(len(hidden) == 4, 'marker auf allen vier himmelsrichtungen -> alle verdeckt',
      f"{hidden}")
hud.telemetry.marker_headings = {'prograde': 45.0, 'retrograde': 225.0,
                                 'normal_in': 135.0, 'antinormal_out': 315.0}
hidden = [deg for deg in (0, 90, 180, 270)
          if any(abs((deg - m + 180.0) % 360.0 - 180.0) < 16.0
                 for m in hud.telemetry.marker_headings.values())]
check(len(hidden) == 0, 'marker zwischen den himmelsrichtungen -> alle sichtbar',
      f"{hidden}")

# --- ziehen am lagemesser -------------------------------------------------
print()
print("7. Ziehen am lagemesser")

frame(1280, 800)
ring = hud.ring
cx, cy = ring.rect.center_x, ring.rect.center_y


def cursor_at(compass_deg):
    """Ganzzahlige fensterkoordinate auf einem kompasswinkel am ring.

    GANZZAHLIG, weil UIRoot die position aus event.pos uebernimmt: ein
    float-anspruch waere um bruchteile eines pixels daneben und der test
    wuerde eine winzige winkelabweichung als fehler melden, die es im
    spiel gar nicht gibt.
    """
    rad = math.radians(compass_deg - 90.0)
    r = ring.rect.w * 0.35
    return (int(cx + math.cos(rad) * r), int(cy + math.sin(rad) * r))


# Der GEGRIFFENE RINGPUNKT muss unter dem cursor bleiben. Ein ringpunkt mit
# dem kurswert d wird bei (d - heading) gezeichnet; greift man bei cursor c,
# ist d = c + heading und muss waehrend des ganzen ziehens konstant bleiben.
# Vorher wurde stattdessen heading = c + offset gesetzt: der ring lief dann
# GEGENLAEUFIG zur maus.
hud.telemetry.heading = 0.0
control.snap_mode = None
sx, sy = cursor_at(40.0)
root._mouse_pos = (float(sx), float(sy))
root.begin_frame(1 / 60)
root.handle_event(pygame.event.Event(
    pygame.MOUSEBUTTONDOWN, {'pos': (sx, sy), 'button': 1}))
start_cursor = ring._cursor_compass(sx, sy)
grabbed = (start_cursor + hud.telemetry.heading) % 360.0

mx, my = cursor_at(70.0)
root._mouse_pos = (float(mx), float(my))
root.handle_event(pygame.event.Event(
    pygame.MOUSEMOTION, {'pos': (mx, my), 'rel': (0, 0), 'buttons': (1, 0, 0)}))
moved_cursor = ring._cursor_compass(mx, my)
held = (ring._manual_heading + moved_cursor) % 360.0
check(abs((held - grabbed + 180.0) % 360.0 - 180.0) < 1e-6,
      'ziehen haelt den gegriffenen ringpunkt unter dem cursor',
      f"gegriffen {grabbed:.2f} -> jetzt {held:.2f}")
swept = (moved_cursor - start_cursor + 180.0) % 360.0 - 180.0
expected_heading = (0.0 - swept) % 360.0
check(abs((ring._manual_heading - expected_heading + 180.0) % 360.0 - 180.0) < 1e-6,
      'cursor im uhrzeigersinn -> kurs zurueck (ring folgt der maus, laeuft ihr nicht entgegen)',
      f"cursor +{swept:.2f} Grad -> sollkurs {ring._manual_heading:.2f}")

# --- loslassen gibt die pfeiltasten wieder frei ---------------------------
# Der ring darf nach dem loslassen NICHT weiter orient_towards_angle()
# aufrufen: sobald der kurs erreicht ist, heftet das intern theta jeden
# frame fest (_snap_locked) und die tastatursteuerung ist tot.
sim_ship.theta = 0.0
for _ in range(240):
    ring.update(ui, 1 / 60)          # ziel anfahren, bis es gehalten wird
check(control._snap_locked, 'gehaltener zug heftet die nase (erwartet)',
      f"theta = {sim_ship.theta:.4f}")

root.handle_event(pygame.event.Event(
    pygame.MOUSEBUTTONUP, {'pos': (int(sx), int(sy)), 'button': 1}))
check(ring._manual_heading is None, 'loslassen loescht den sollkurs')
check(not control._snap_locked, 'loslassen loest die theta-heftung')

theta_before = sim_ship.theta
keys = {pygame.K_LEFT: 0, pygame.K_RIGHT: 1, pygame.K_UP: 0, pygame.K_DOWN: 0}
control.handle_rotation(keys, 1 / 60)
ring.update(ui, 1 / 60)              # der ring darf jetzt nicht dazwischenfunken
check(abs(sim_ship.theta - theta_before) > 1e-6,
      'nach dem loslassen drehen die pfeiltasten das schiff wieder',
      f"theta {theta_before:.4f} -> {sim_ship.theta:.4f}")

# --- koerperliste ---------------------------------------------------------
print()
print("8. Koerperliste (bezugskoerper)")

from ui.hud.body_browser import build_hierarchy

rows = build_hierarchy(bodies)
names = [b.name for _i, b, _d in rows]
depths = {b.name: d for _i, b, d in rows}
check('SaturnV' not in names, 'das schiff steht nicht in der liste', f"{names}")
check(names[0] == 'Sonne', 'wurzel ohne is_moon_of steht oben', f"{names}")
check(depths.get('Erde') == 1 and depths.get('Mond') == 2,
      'is_moon_of erzeugt die einrueckung', f"{depths}")
check(names.index('Erde') < names.index('Mars'),
      'geschwister nach abstand zum mutterkoerper sortiert', f"{names}")
check(names.index('Mond') == names.index('Erde') + 1,
      'ein mond folgt direkt seinem planeten', f"{names}")

# Zyklische is_moon_of-daten duerfen die gliederung nicht aufhaengen.
class Cyc:
    def __init__(self, name):
        self.name, self.mass, self.is_ship = name, 1.0, False
        self.semi_major_axis, self.is_moon_of = 1.0, None


a, b_ = Cyc('A'), Cyc('B')
a.is_moon_of, b_.is_moon_of = b_, a
check(len(build_hierarchy([a, b_])) == 2, 'is_moon_of-zyklus haengt nicht auf')

browser = hud.body_browser
frame(1280, 800)
browser.open = True
frame(1280, 800)
browser.draw(ui)          # muss zeichenbar sein (vgl. palette-absturz)
check(True, 'geoeffnete koerperliste laesst sich zeichnen')

mars_row = next(i for i, (_idx, b, _d) in enumerate(browser.rows())
                if b.name == 'Mars')
mars_index = browser.rows()[mars_row][0]
rx, ry, rw, rh = browser._row_rect(ui, mars_row)
px_, py_ = int(rx + rw * 0.5), int(ry + rh * 0.5)
root._mouse_pos = (float(px_), float(py_))
root.handle_event(pygame.event.Event(
    pygame.MOUSEBUTTONDOWN, {'pos': (px_, py_), 'button': 1}))
root.handle_event(pygame.event.Event(
    pygame.MOUSEBUTTONUP, {'pos': (px_, py_), 'button': 1}))
check(state.reference_index == mars_index,
      'klick auf eine zeile setzt den bezugskoerper',
      f"{state.reference_name}")
check(not browser.open, 'die liste schliesst nach der wahl')

state.set_reference_index(bodies.index(earth))

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for failure in FAILURES:
        print(f"  {failure}")
    pygame.quit()
    sys.exit(1)
print("ui/hud: alle pruefungen bestanden")
pygame.quit()
