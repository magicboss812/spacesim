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

from physics.vec import Vec2, G
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
# zaehlen geraete auf und kosten zusammen ~45 s. Siehe runtime/window.py.
pygame.display.init()
pygame.font.init()
pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL | RESIZABLE, vsync=0)
gl = moderngl.create_context()
gl.enable(moderngl.BLEND)
gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

from ship.camera import Camera
from config.loader import ConfigLoader
from runtime.system_loader import SystemLoader
from ship.predictor import Predictor
from render.renderer import Renderer
from ship.control import schiffcontrol
from ui import UIContext, UIRoot, UIState
from ui.hud import Hud
from ui.hud.layout import WARP_STEPS
from ui.hud.navball import GAUGE_RADIUS, THROTTLE_ARC
from ui import units
from physics.world import world as World

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


def frame(width=None, height=None, dt=1 / 60):
    if width is not None:
        root.resize(width, height, ui_scale=1.0)
    hud.update()
    root.begin_frame(dt)


frame()

GROUPS = {
    'badge': hud.badge, 'warp': hud.warp,
    'koerperliste': hud.body_browser, 'target': hud.target,
    'navball': hud.navball, 'frames': hud.frames,
    'snaps': hud.snaps, 'zoom': hud.zoom,
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
        # Ein pixel toleranz: navball-block und snap-rosette sind bewusst
        # aneinander GEDOCKT, ihre kanten duerfen sich beruehren.
        overlap = (a.x < b.right - 1.0 and b.x < a.right - 1.0
                   and a.y < b.bottom - 1.0 and b.y < a.bottom - 1.0)
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
check(hud.target.visible and not hud.target_rail.visible,
      'breit: ziel als panel, nicht als leiste')

frame(820, 620)
check(not hud._wide, 'bei 820 breit: kompaktes layout')
check(hud.target_rail.visible and not hud.target.visible,
      'schmal: ziel als leiste')
check(hud.body_browser.visible, 'schmal: koerperliste bleibt erreichbar')
check(hud.navball.visible,
      'schmal: der navball-block bleibt -- ohne ihn kann man nicht fliegen')
check(hud.snaps_compact.visible and not hud.snaps.visible,
      'schmal: snap-rosette geschrumpft')
for name, widget in (('leiste rechts', hud.target_rail),
                     ('navball', hud.navball), ('ring', hud.ring),
                     ('snaps kompakt', hud.snaps_compact)):
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

# --- die palette liegt fest ----------------------------------------------
# Der frueher hier gepruefte palettenwechsel ist ENTFALLEN: eine farbe, die
# sich neu verteilen laesst, kann nichts bedeuten (siehe theme.py). Geprueft
# wird stattdessen, dass es genau einen satz gibt und dass jede rollenfarbe
# auf dem dunklen grund noch lesbar ist -- das war der eigentliche zweck der
# aufhellung in theme.readable().
check(len(ui.theme.palette_sets()) == 1, 'es gibt genau EINEN farbsatz',
      f"{[name for name, _ in ui.theme.palette_sets()]}")
for role in ('ring', 'velocity', 'target', 'warp', 'throttle', 'snap', 'ship'):
    color = ui.theme.palette.accent_for(role)
    check(max(color[:3]) > 0.35,
          f"rollenfarbe '{role}' bleibt auf dunklem grund lesbar",
          f"{tuple(round(c, 3) for c in color)}")
check(all(isinstance(c, tuple) for c in ui.theme.palette.colors),
      'palette haelt float-tupel, keine zeichenketten')

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
# AUSKLAPPEN LAUFEN LASSEN. Seit die liste animiert aufgeht, reicht ihre
# trefferflaeche nur so weit, wie sie auch gezeichnet ist -- ein klick auf
# eine zeile, die noch gar nicht sichtbar ist, darf nicht wirken. Ein
# einzelner frame genuegt dafuer nicht mehr.
for _ in range(30):
    frame(1280, 800)
browser.draw(ui)          # muss zeichenbar sein (vgl. palette-absturz)
check(True, 'geoeffnete koerperliste laesst sich zeichnen')
check(browser._open_t > 0.98, 'und ist dann voll ausgeklappt',
      f"_open_t = {browser._open_t:.4f}")

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

# ═══════════════════════════════════ 9. die liste klappt nach UNTEN auf

print()
print("9. Die koerperliste klappt nach unten auf, nicht von der seite herein")

browser.open = False
for _ in range(30):
    frame()

full_height = browser._panel_rect(ui)[3]
browser.open = True
heights, lefts, widths, tops = [], [], [], []
for _ in range(30):
    frame()
    ax, ay, aw, ah = browser._panel_rect_open(ui)
    heights.append(ah)
    lefts.append(ax)
    widths.append(aw)
    tops.append(ay)

check(all(b >= a - 1e-6 for a, b in zip(heights, heights[1:])),
      'die hoehe waechst monoton',
      f"{heights[0]:.1f} -> {heights[-1]:.1f} von {full_height:.1f} px")
check(heights[0] < full_height * 0.5 and heights[-1] > full_height * 0.98,
      'sie faengt klein an und erreicht die volle hoehe')
# DAS ist die eigentliche forderung: x und breite duerfen sich NICHT
# bewegen, sonst faehrt die liste seitlich ein statt aufzuklappen.
check(len(set(round(v, 3) for v in lefts)) == 1
      and len(set(round(v, 3) for v in widths)) == 1,
      'x und breite bleiben fest -- sie kommt nicht von der seite',
      f"x = {lefts[0]:.1f}, breite = {widths[0]:.1f}")
check(len(set(round(v, 3) for v in tops)) == 1
      and abs(tops[0] - browser.rect.bottom) < ui.px(12) + 1.0,
      'die oberkante haengt an der unterkante des knopfes',
      f"panel bei y = {tops[0]:.1f}, knopf endet bei {browser.rect.bottom:.1f}")

browser.open = False
closing = []
for _ in range(30):
    frame()
    closing.append(browser._panel_rect_open(ui)[3])
check(all(b <= a + 1e-6 for a, b in zip(closing, closing[1:]))
      and closing[-1] == 0.0,
      'beim schliessen laeuft dieselbe bewegung rueckwaerts bis genau null',
      f"{closing[0]:.1f} -> {closing[-1]:.1f} px")


def _open_progress(dt, steps):
    browser.open = False
    browser._open_t = 0.0
    frame(dt=dt)
    browser.open = True
    for _ in range(steps):
        frame(dt=dt)
    return browser._open_t


# Framerate-unabhaengig, wie jede andere bewegung im projekt: doppelte
# bildrate, doppelt so viele frames, derselbe fortschritt.
slow = _open_progress(1 / 60.0, 10)
fast = _open_progress(1 / 120.0, 20)
check(abs(slow - fast) < 0.02, 'die bewegung ist framerate-unabhaengig',
      f"60 fps / 10 frames = {slow:.4f}, 120 fps / 20 frames = {fast:.4f}")
browser.open = False
for _ in range(30):
    frame()


# ═════════════════════════ 10. der ring faengt nur in seinem KREIS

print()
print("10. Trefferflaeche des lagemessers: rund, nicht quadratisch")

navball = hud.navball
ring = hud.ring
cx, cy = navball._center(ui)
outer = ring.rect.w / 220.0 * 103.0

for label, (sx, sy) in (('oben links', (-1, -1)), ('oben rechts', (1, -1)),
                        ('unten links', (-1, 1)), ('unten rechts', (1, 1))):
    x = ring.rect.center_x + sx * ring.rect.w * 0.47
    y = ring.rect.center_y + sy * ring.rect.h * 0.47
    check(not ring.hit_test(ui, x, y),
          f"die ecke '{label}' des quadrats gehoert dem ring nicht",
          f"abstand {math.hypot(x - cx, y - cy):.1f} px, radius {outer:.1f}")
check(ring.hit_test(ui, cx, cy) and ring.hit_test(ui, cx + outer * 0.9, cy),
      'der kreis selbst gehoert ihm weiterhin')

# Die konkrete folge, wegen der das ueberhaupt auffiel: der schubbogen liegt
# ausserhalb des rings, seine enden aber INNERHALB von dessen quadrat. Mit
# quadratischer trefferflaeche verschluckte der ring genau diese klicks, und
# der schub liess sich im oberen drittel nicht stellen.
low, high = THROTTLE_ARC
for label, fraction in (('unteres ende', 0.04), ('mitte', 0.5),
                        ('oberes ende', 0.96)):
    compass = low + (high - low) * fraction
    angle = math.radians(compass - 90.0)
    x = cx + math.cos(angle) * ui.px(GAUGE_RADIUS)
    y = cy + math.sin(angle) * ui.px(GAUGE_RADIUS)
    root._mouse_pos = (x, y)
    frame()
    picked = root.hovered_widget
    level = navball._throttle_from_point(ui, x, y)
    check(picked is navball and level is not None,
          f"schubbogen, {label}: der zeiger gehoert dem navball-block",
          f"getroffen: {type(picked).__name__ if picked else None}, "
          f"stufe {None if level is None else round(level, 3)}")

# Gegenprobe -- ohne sie koennte diese pruefung unbemerkt leerlaufen, falls
# der bogen einmal so weit nach aussen wandert, dass er das quadrat verlaesst.
compass = low + (high - low) * 0.96
angle = math.radians(compass - 90.0)
probe = (cx + math.cos(angle) * ui.px(GAUGE_RADIUS),
         cy + math.sin(angle) * ui.px(GAUGE_RADIUS))
check(ring.rect.contains(*probe),
      'und dieser punkt liegt wirklich im quadrat des rings',
      f"({probe[0]:.0f}, {probe[1]:.0f}) in {ring.rect}")


# ═════════════════════ 11. tabellenziffern -- der zaehler zuckt nicht

print()
print("11. Zaehler und messwerte wechseln die breite nicht")

# SB Liquid ist NICHT dicktengleich: die '1' ist schmaler als jede andere
# ziffer. Bei einem rechtsbuendigen countdown wandert damit die linke kante
# jedes mal, wenn eine '1' hinein- oder herauslaeuft -- im sekundentakt.
font = ui.text.font('value')
raw_widths = {font.size(str(d))[0] for d in range(10)}
check(len(raw_widths) > 1,
      'die schrift selbst ist nicht dicktengleich (sonst pruefte das hier nichts)',
      f"rohe ziffernbreiten {sorted(raw_widths)}")

for role in ('value', 'gauge', 'readout', 'warp'):
    seconds = {ui.text.measure(f"T-06:24:{s:02d}", role)[0] for s in range(60)}
    hours = {ui.text.measure(f"T-{h:02d}:24:19", role)[0] for h in range(24)}
    check(len(seconds) == 1 and len(hours) == 1,
          f"'{role}': gleiche breite bei jedem ziffernstand",
          f"sekunden {sorted(seconds)}, stunden {sorted(hours)}")

check(len({ui.text.measure(str(d) * 6, 'value')[0] for d in range(10)}) == 1,
      'auch jede ziffer einzeln belegt dieselbe breite')
# Die LESESCHRIFT bleibt davon unberuehrt -- dort waeren feste ziffern-
# breiten in gemischtem text nur eine luecke.
check(not ui.text._role_tabular('body'),
      'die leseschrift setzt weiterhin proportional')


# ═══════════ 12. der massstab der geschwindigkeitsnadel kommt aus der bahn

print()
print("12. Die geschwindigkeitsnadel misst gegen die bahn, nicht gegen "
      "eine feste zahl")

# Frueher stand im ring ein fester vollausschlag von 2600 m/s -- eine zahl,
# die zu keinem koerper gehoert. Jetzt ist der massstab v_flucht = sqrt(2) *
# sqrt(mu/r) an diesem ort, und eine kreisbahn liegt damit an JEDEM koerper
# auf derselben nadellaenge.
fractions = []
for body_name, altitude in (('Erde', 4.0e5), ('Mond', 1.0e5),
                            ('Jupiter', 1.0e6), ('Sonne', 1.496e11)):
    body = next((b for b in bodies if b.name == body_name), None)
    if body is None:
        continue
    state.set_reference_index(bodies.index(body))
    radius = float(getattr(body, 'radius', 0.0)) + altitude
    mu_body = G * float(body.mass)
    base = hud.telemetry.body_velocity(body)
    sim_ship.position.x = float(body.position.x) + radius
    sim_ship.position.y = float(body.position.y)
    sim_ship.velocity.x = base[0]
    sim_ship.velocity.y = base[1] + math.sqrt(mu_body / radius)
    frame()
    fractions.append((body_name, hud.telemetry.velocity_fraction()))

check(all(abs(f - fractions[0][1]) < 1e-3 for _n, f in fractions),
      'eine kreisbahn ergibt an jedem koerper dieselbe nadellaenge',
      ', '.join(f"{n} {f:.4f}" for n, f in fractions))
check(abs(fractions[0][1] - 1.0 / math.sqrt(2.0)) < 1e-3,
      'und zwar bei 1/sqrt(2) des vollausschlags',
      f"{fractions[0][1]:.4f}")

# Obergrenze: der vollausschlag IST die fluchtgeschwindigkeit.
state.set_reference_index(bodies.index(earth))
radius = float(earth.radius) + 4.0e5
mu_earth = G * float(earth.mass)
base = hud.telemetry.body_velocity(earth)
sim_ship.position.x = float(earth.position.x) + radius
sim_ship.position.y = float(earth.position.y)
for factor, label in ((math.sqrt(2.0), 'genau fluchtgeschwindigkeit'),
                      (3.0, 'weit darueber')):
    sim_ship.velocity.x = base[0]
    sim_ship.velocity.y = base[1] + math.sqrt(mu_earth / radius) * factor
    frame()
    value = hud.telemetry.velocity_fraction()
    check(abs(value - 1.0) < 1e-3, f"{label}: nadel am anschlag, nicht darueber",
          f"{value:.4f}")

sim_ship.velocity.x = base[0]
sim_ship.velocity.y = base[1]
frame()
check(hud.telemetry.velocity_fraction() < 1e-6,
      'ruhe relativ zum bezugskoerper: nadel auf null')

# Ohne bezugskoerper darf nichts sterben -- die richtung stimmt dann noch,
# der massstab nicht, und der ring zeichnet eine halbe nadel. Ueber UIState
# ist dieser zustand NICHT erreichbar (set_reference_index(None) ist dort
# ein no-op -- es gibt immer einen bezugskoerper); Telemetry laesst
# ui_state=None aber ausdruecklich zu, und genau dieser pfad wird geprueft.
saved_state = hud.telemetry.ui_state
hud.telemetry.ui_state = None
try:
    check(hud.telemetry.velocity_fraction() is None
          and hud.telemetry.circular_speed_fraction() is None,
          'ohne bezugskoerper liefert der massstab None statt zu stuerzen',
          f"{hud.telemetry.orbital_speed_scale()}")
finally:
    hud.telemetry.ui_state = saved_state
state.set_reference_index(bodies.index(earth))
frame()

# ═══════ 13. der zahlenstreifen laeuft bei keiner steigrate ueber

print()
print("13. Die zahlenstreifen unter den flanken laufen nicht ueber")

# Ihr inhalt haengt an der GROESSENORDNUNG des messwerts: '+14.41km/s'
# belegte in 15 px genau die volle streifenbreite und schob sich ueber das
# 'V/S' davor. Die ausweichstufe auf die kleine rolle allein reicht dafuer
# nicht -- die pixelschrift rastet ihre groesse auf fuenferschritte, ist
# also nicht ueberall proportional kleiner.
from ui.hud.navball import BOX_W as _BOX_W

MAGNITUDES = (0.0, 0.4, 7.0, 99.0, 109.0, 999.0, 1234.0, 14410.0,
              99900.0, 250000.0, 4.2e6)


def _strip_width(probe, caption, text):
    """Bildet NavballCluster._strip nach: drei stufen, wert gewinnt."""
    available = (probe.px(_BOX_W) - 2.0 * probe.px(3)) - 2.0 * probe.px(9)
    gap = probe.px(6)
    caption_w = probe.text.measure(caption, 'caption')[0]
    for candidate in ('throttle_value', 'caption'):
        width = caption_w + gap + probe.text.measure(text, candidate)[0]
        if width <= available:
            return width, available, True
    return probe.text.measure(text, 'caption')[0], available, False


worst = 0.0
labelled = 0
for probe_scale in (1.0, 1.25, 1.44, 2.0):
    probe = UIContext(gl, W, H, ui_scale=probe_scale)
    for magnitude in MAGNITUDES:
        for sign in (1.0, -1.0):
            radial = magnitude * sign
            value, unit = units.split_speed(abs(radial), digits=1)
            text = f"{'+' if radial >= 0.0 else '-'}{value}{unit}"
            width, available, with_caption = _strip_width(probe, 'V/S', text)
            labelled += 1 if with_caption else 0
            worst = max(worst, width / available)
            if width > available + 0.01:
                check(False, 'streifen laeuft ueber',
                      f"ui_scale {probe_scale}, {text!r}: "
                      f"{width:.1f} von {available:.1f} px")
    for text in ('0%', '100%', 'HOLD'):
        width, available, _ = _strip_width(probe, 'THR', text)
        if width > available + 0.01:
            check(False, 'THR-streifen laeuft ueber',
                  f"ui_scale {probe_scale}, {text!r}")

check(worst <= 1.0 + 1e-6,
      'kein zahlenstreifen laeuft ueber -- ueber alle groessenordnungen '
      'und vier ui_scales',
      f"engster fall belegt {worst * 100:.1f} % des platzes")
# Gegenprobe: die beschriftung darf nicht IMMER wegfallen, sonst waere die
# pruefung erfuellt, ohne dass man je 'V/S' zu sehen bekaeme.
check(labelled > 0.5 * len(MAGNITUDES) * 2 * 4,
      'und die beschriftung bleibt im regelfall stehen',
      f"{labelled} von {len(MAGNITUDES) * 2 * 4} faellen mit beschriftung")

# Der bogen daneben misst gegen die geschwindigkeit RELATIV ZUM
# BEZUGSKOERPER. Gegen frame_speed waere er vom gewaehlten plotting-rahmen
# abhaengig -- eine anzeige, die sich neu skaliert, weil jemand die ANSICHT
# umstellt.
# Erst eine bahn mit ECHTER radialrate herstellen -- steht das schiff
# still, sind beide werte null und die pruefung liefe leer.
base_v = hud.telemetry.body_velocity(earth)
radius = float(earth.radius) + 4.0e5
sim_ship.position.x = float(earth.position.x) + radius
sim_ship.position.y = float(earth.position.y)
speed = math.sqrt(G * float(earth.mass) / radius)
sim_ship.velocity.x = base_v[0] + speed * 0.6      # radial, nach aussen
sim_ship.velocity.y = base_v[1] + speed * 0.8
frame()
check(abs(hud.telemetry.radial_fraction() or 0.0) > 0.1,
      'die probe hat eine echte radialrate (sonst prueft das folgende nichts)',
      f"anteil {hud.telemetry.radial_fraction():.4f}")

hud.telemetry.frame_speed = 1.0e9
loose = hud.telemetry.radial_fraction()
hud.telemetry.frame_speed = 1.0
tight = hud.telemetry.radial_fraction()
check(loose is not None and tight is not None and abs(loose - tight) < 1e-9,
      'der steigraten-bogen haengt nicht an frame_speed',
      f"{loose:.6f} gegen {tight:.6f}")
frame()

# ═══════════════════════════════════ 14. horizont-regler ist verdrahtet
print()
print("14. Horizont-Regler")

# Eigener Hud mit ECHTEN closures ueber eine veraenderliche box -- so
# schreibt ein simulierter griff den manuellen faktor wirklich zurueck.
_hmult = [1.0]


def predictor_manual_mult_getter():
    return _hmult[0]


def _hmult_set(m):
    _hmult[0] = max(0.25, min(4.0, float(m)))


hud_h = Hud(root, world, sim_ship, control, camera, renderer, predictor, state,
            tick_rate=60.0,
            horizon_mult_get=predictor_manual_mult_getter,
            horizon_mult_set=_hmult_set,
            horizon_mult_min=0.25, horizon_mult_max=4.0, horizon_sweep_s=2.5)

check(getattr(hud_h, 'horizon', None) is not None,
      "hud.horizon existiert", str(getattr(hud_h, 'horizon', None)))
check(hud_h.horizon in hud_h.left_stack.children,
      "sitzt im linken stapel (unter FRAME + zoom)", "")
check(hud_h.horizon.is_grabbing is False,
      "startet ungegriffen", "")

# ein simulierter griff nach rechts hebt den manuellen faktor
_start = predictor_manual_mult_getter()
hud_h.horizon.pressed = True
hud_h.horizon._offset = 1.0
for _ in range(30):
    hud_h.horizon.update(hud_h.ctx, 1.0 / 60.0)
check(predictor_manual_mult_getter() > _start,
      "griff nach rechts verlaengert den horizont",
      f"{_start:.3f} -> {predictor_manual_mult_getter():.3f}")
check(hud_h.horizon.is_grabbing is True, "waehrend des griffs: is_grabbing", "")

hud_h.horizon.pressed = False
for _ in range(30):
    hud_h.horizon.update(hud_h.ctx, 1.0 / 60.0)
check(abs(hud_h.horizon._offset) < 0.06,
      "loslassen federt in die mitte zurueck",
      f"offset {hud_h.horizon._offset:.4f}")

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for failure in FAILURES:
        print(f"  {failure}")
    pygame.quit()
    sys.exit(1)
print("ui/hud: alle pruefungen bestanden")
pygame.quit()
