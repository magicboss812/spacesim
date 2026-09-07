"""Die manoever-zeichnung auf einem echten GL-context.

Geprueft wird NICHT, wie es aussieht, sondern was das HUD danach anfassen
kann: `renderer.maneuver_node_hits`. Der ziehgriff (ui/hud/maneuver.py)
trifft genau diese schirmpositionen -- rechnete er sie selbst nach, laege
der griff neben dem gezeichneten pfeil, und zwar in jedem bewegten
bezugsrahmen anders.

Dieselbe bauart wie `tests/background_gl_test.py`.

Aufruf: python tests/maneuver_render_test.py
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

W, H = 1024, 640
FAILURES = []


def _empty(curve):
    """Leer, egal ob None oder ein array der laenge 0.

    Die schirmkurve ist seit der projektion in EINEM rutsch ein (n,3)-array
    -- `== []` waere darauf ein elementweiser vergleich, kein test.
    """
    return curve is None or len(curve) == 0


def check(condition, label, detail=''):
    status = 'OK  ' if condition else 'FEHL'
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ''))
    if not condition:
        FAILURES.append(label)


# NUR display+font -- pygame.init() zaehlt mixer/joystick auf (siehe CLAUDE.md).
pygame.display.init()
pygame.font.init()
pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
gl = moderngl.create_context()
gl.enable(moderngl.BLEND)
gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

from config.loader import ConfigLoader
from physics.world import world as World
from render.maneuver import _GREEN
from render.renderer import Renderer
from runtime.system_loader import SystemLoader
from ship.camera import Camera
from ship.maneuver.plan import ManeuverNode, ManeuverPlan
from ship.maneuver.preview import ManeuverPreview
from ship.predictor import Predictor
from runtime.bootstrap import FrameController
from ui.state import UIState

_GREEN_R, _GREEN_G, _GREEN_B = _GREEN

config = ConfigLoader()
config.load()
bodies = SystemLoader(config.get('simulation.system_file', 'solar_system.json')).load()
w = World(float(config.get('physics.gravitational_constant', 6.6730831e-11)))
w.body = bodies
config.apply_to_world(w)
w.update_planets(0.0)

erde = next(b for b in w.body if b.name.lower().startswith('erde'))
ship = next(b for b in w.body if getattr(b, 'is_ship', False))
mu = w.G * erde.mass
r = float(getattr(erde, 'radius', 6.371e6)) + 4.0e6
ahead = erde.position_at_time(1.0)
behind = erde.position_at_time(-1.0)
ship.position.x = erde.position.x + r
ship.position.y = erde.position.y
ship.velocity.x = (float(ahead.x) - float(behind.x)) / 2.0
ship.velocity.y = (float(ahead.y) - float(behind.y)) / 2.0 + math.sqrt(mu / r)

camera = Camera(None, W, H)
config.apply_to_camera(camera)
camera.target = ship
camera.scale = 2.0e-5
camera.snap_to_targets()

predictor = Predictor(**config.predictor_kwargs())
config.apply_to_predictor(predictor)
predictor.async_compute = False
predictor.initialize(ship, w)
predictor.update(ship, w)

renderer = Renderer(W, H, enable_fxaa=False, ctx=gl)
config.apply_to_renderer(renderer)
renderer.set_frame_time(w.time)
renderer.current_reference_body = erde

# DEN PLOT-FRAME AUFSETZEN, wie es runtime/bootstrap.py tut. Ohne ihn
# rechnet der renderer barycentrisch, und ein knoten eine dreiviertel
# umlaufzeit spaeter liegt dann Gigameter neben dem bild -- die Erde ist
# inzwischen weitergeflogen. Der treffertest meldete dann null marker,
# obwohl die kette voellig in ordnung war.
ui_state = UIState(w.body, initial_reference_index=w.body.index(erde))
frames = FrameController(w, ui_state, predictor, camera, renderer)
frames.apply()
# apply() macht die gehaltene vorhersage ungueltig -- danach ist
# get_points() leer, bis der predictor einmal neu gerechnet hat.
predictor.update(ship, w)

base = predictor.get_points()
t0 = float(base[0, 2])
t_end = float(base[len(base) - 1, 2])
t_node = t0 + (t_end - t0) * 0.25

plan = ManeuverPlan()
plan.add(ManeuverNode(t_node, dv_prograde=150.0, dv_normal=-40.0))
preview = ManeuverPreview(max_points=4000)
preview.rebuild(plan, ship, w, predictor, erde, 10.0, 0.6)

renderer.maneuver_plan = plan
renderer.maneuver_preview = preview
renderer._maneuver_predictor = predictor

print("\n1) die vorgabewerte stehen")
check(hasattr(renderer, 'maneuver_node_hits'), "maneuver_node_hits existiert", "")
check(renderer.maneuver_node_hits == [], "und beginnt leer", "")
check(renderer.maneuver_drag_active is False, "kein zug im gange", "")
check(renderer.maneuver_drag_curve is False, "und keiner am MARKER", "")
check(renderer.maneuver_drag_handle is None, "kein griff ausgelenkt", "")
check(_empty(renderer.maneuver_curve_screen), "keine schirmkurve abgelegt", "")

print("\n2) ein zeichendurchgang meldet den knoten ans HUD")
gl.screen.use()
gl.viewport = (0, 0, W, H)
gl.clear(0.0, 0.0, 0.0, 1.0)
renderer.draw_maneuver(camera, w.body)
hits = renderer.maneuver_node_hits
check(len(hits) == 1, "ein treffer je knoten", f"{len(hits)}")
if hits:
    hit = hits[0]
    check(math.isfinite(hit['sx']) and math.isfinite(hit['sy']),
          "die schirmposition ist endlich", f"({hit['sx']:.1f}, {hit['sy']:.1f})")
    check(0.0 <= hit['sx'] <= W and 0.0 <= hit['sy'] <= H,
          "und liegt im bild", f"({hit['sx']:.1f}, {hit['sy']:.1f})")
    check(len(hit['handles']) == 4, "vier griffe", f"{len(hit['handles'])}")
    kinds = sorted(h['kind'] for h in hit['handles'])
    check(kinds == ['antinormal', 'normal', 'prograde', 'retrograde'],
          "und zwar die vier erwarteten", f"{kinds}")

print("\n3) die griffe stehen paarweise GEGENUEBER")
# prograde und retrograde muessen entgegengesetzte schirmrichtungen tragen,
# normal und antinormal ebenso. Ein vertauschtes vorzeichen faellt hier auf.
if hits and len(hits[0]['handles']) == 4:
    by_kind = {h['kind']: h for h in hits[0]['handles']}
    for a, b in (('prograde', 'retrograde'), ('normal', 'antinormal')):
        dot = (by_kind[a]['dir_sx'] * by_kind[b]['dir_sx']
               + by_kind[a]['dir_sy'] * by_kind[b]['dir_sy'])
        check(dot < -0.999, f"{a} steht {b} gegenueber", f"skalarprodukt {dot:.6f}")
    dot = (by_kind['prograde']['dir_sx'] * by_kind['normal']['dir_sx']
           + by_kind['prograde']['dir_sy'] * by_kind['normal']['dir_sy'])
    check(abs(dot) < 1e-6, "prograde steht senkrecht auf normal",
          f"skalarprodukt {dot:.3e}")
    for handle in hits[0]['handles']:
        length = math.hypot(handle['dir_sx'], handle['dir_sy'])
        check(abs(length - 1.0) < 1e-6, f"{handle['kind']}: einheitsrichtung",
              f"{length:.9f}")

print("\n3b) die schirmpositionen kommen aus EINER projektion")
# Der array-weg (`to_this_frame_xy_arrays`) muss dasselbe liefern wie der
# punktweise -- er ist nur schneller, nicht anders. Gemessen war die
# punktweise projektion von 900 linienpunkten je frame der grund, warum die
# bildrate beim verstellen eines knotens von 100 auf 40 fiel.
renderer.maneuver_drag_curve = True
renderer.draw_maneuver(camera, w.body)
fast = renderer.maneuver_curve_screen
base_pts = predictor.get_points()
stride = max(1, int(math.ceil(len(base_pts)
                              / float(max(2, renderer.maneuver_curve_screen_points)))))
cam_xy = renderer._frame_camera_xy(camera)
indices = list(range(0, len(base_pts), stride))
# Der LETZTE punkt gehoert dazu, auch wenn der stride ihn ueberspringt --
# sonst endet die gezeichnete linie vor ihrem eigenen ende.
if indices[-1] != len(base_pts) - 1:
    indices.append(len(base_pts) - 1)
slow = []
for i in indices:
    px_, py_ = renderer._world_to_screen_xy_at_time(
        float(base_pts[i, 0]), float(base_pts[i, 1]), camera,
        float(base_pts[i, 2]), cam_xy)
    slow.append((px_, py_))
check(len(fast) == len(slow), "gleich viele punkte",
      f"{len(fast)} gegen {len(slow)}")
if len(fast) == len(slow) and len(fast):
    diff = float(np.max(np.abs(np.asarray(slow) - fast[:, :2])))
    check(diff < 1e-6, "und dieselben schirmpositionen",
          f"groesste abweichung {diff:.3e} px")
renderer.maneuver_drag_curve = False

def max_deviation_px(run, truth):
    """Groesster abstand eines WAHREN kurvenpunkts vom gezeichneten zug.

    Punkt-zu-STRECKE, nicht punkt-zu-punkt: die linie wird als polygonzug
    gezeichnet, und was man sieht, ist ihr abstand zur kurve.
    """
    worst = 0.0
    ax = run[:-1, 0]
    ay = run[:-1, 1]
    bx = run[1:, 0]
    by = run[1:, 1]
    ex = bx - ax
    ey = by - ay
    length2 = ex * ex + ey * ey
    length2 = np.where(length2 > 1e-12, length2, 1e-12)
    for px_, py_ in truth:
        if not (-200.0 <= px_ <= W + 200.0 and -200.0 <= py_ <= H + 200.0):
            continue          # ausserhalb wird ohnehin nicht verfeinert
        t = ((px_ - ax) * ex + (py_ - ay) * ey) / length2
        t = np.clip(t, 0.0, 1.0)
        d = np.hypot(px_ - (ax + t * ex), py_ - (ay + t * ey))
        worst = max(worst, float(np.min(d)))
    return worst


print("\n3c) die gezeichnete linie bleibt GLATT, egal wie lang sie ist")
# DAS WAR DER FEHLER: die vorschau setzt ihre punkte in gleichem
# BOGENABSTAND, und der waechst mit der eingestellten reichweite. Ein
# fester stride darauf machte die linie mit jeder verlaengerung kantiger,
# bis sie sichtbar aus geraden stuecken bestand. Die verfeinerung
# (`_hermite_refine_world`) haengt die aufloesung an den BILDSCHIRM statt
# an die linienlaenge.
plan.clear()
plan.add(ManeuverNode(t_node, dv_prograde=150.0, dv_normal=-40.0))
ratios = []
for mult in (1.0, 4.0, 16.0):
    preview.set_length_mult(mult)
    preview.rebuild(plan, ship, w, predictor, erde, 10.0, 0.6)
    pts = preview.points
    if pts is None or len(pts) < 8:
        check(False, f"x{mult:g}: die vorschau liefert eine linie", "")
        continue
    # Die WAHRHEIT: jeder gerechnete punkt, einzeln zu seiner eigenen zeit.
    truth = []
    for i in range(len(pts)):
        truth.append(renderer._world_to_screen_xy_at_time(
            float(pts[i, 0]), float(pts[i, 1]), camera, float(pts[i, 2]),
            cam_xy))
    fine = renderer._maneuver_screen_polyline(
        pts, camera, cam_xy, int(renderer.maneuver_max_draw_points))
    coarse = renderer._maneuver_screen_polyline(
        pts, camera, cam_xy, int(renderer.maneuver_coarse_points),
        refine=False)
    d_fine = max_deviation_px(fine, truth)
    d_coarse = max_deviation_px(coarse, truth)
    print(f"       x{mult:<5g} verfeinert {len(fine):4d} punkte, "
          f"abweichung {d_fine:7.3f} px  (grob {len(coarse):3d} / "
          f"{d_coarse:8.3f} px)")
    check(d_fine < 2.0, f"x{mult:g}: verfeinert bleibt unter 2 px",
          f"{d_fine:.3f} px")
    check(len(fine) <= int(renderer.maneuver_max_draw_points) + 8,
          f"x{mult:g}: und haelt das punktbudget ein", f"{len(fine)}")
    # GEGENPROBE: ohne verfeinerung ist es schlechter. Ohne sie koennte der
    # abschnitt bestehen, weil die kurve zufaellig gerade ist. NICHT je
    # stufe ein fester faktor -- bei grosser reichweite liegt der groesste
    # teil der linie ausserhalb des bildes, dort wird ohnehin nicht
    # verfeinert und das grobgitter ist zufaellig fast so gut. Der faktor
    # wird deshalb ueber die stufen gesammelt und danach einmal geprueft.
    # Nicht scharf `>=`: die zwischenpunkte sind KUBISCHE naeherungen, keine
    # integrierten zustaende, und bei sehr grossem punktabstand weicht die
    # naeherung selbst um bruchteile eines pixels ab. Ein zehntel pixel
    # spielraum, damit die pruefung eine aussage ueber die GLATTHEIT bleibt
    # und nicht ueber das rauschen der interpolation.
    check(d_coarse >= d_fine - 0.1,
          f"x{mult:g}: das grobgitter ist nicht besser",
          f"{d_coarse:.3f} gegen {d_fine:.3f} px")
    ratios.append(d_coarse / max(d_fine, 1e-9))
check(max(ratios) > 2.0,
      "und wo die kurve wirklich biegt, ist es MEHRFACH schlechter",
      f"groesster faktor {max(ratios):.2f}")
preview.set_length_mult(1.0)
preview.rebuild(plan, ship, w, predictor, erde, 10.0, 0.6)

print("\n3d) die endkappen stehen am ENDE DES PLANS")
# Dieselbe aussage wie bei den bahnlinien, nur eine bahn weiter: der kreis
# traegt den ECHTEN koerperradius, damit ablesbar ist, ob das schiff zur
# planendzeit IM koerper steckt.
drawn = []
real_disc = renderer._draw_body_disc_outline
real_cap = renderer._draw_end_cap


def spy_disc(sx_, sy_, r_px, color):
    drawn.append(('disc', sx_, sy_, r_px, color))
    return real_disc(sx_, sy_, r_px, color)


def spy_cap(sx_, sy_, color, size_px):
    drawn.append(('cap', sx_, sy_, size_px, color))
    return real_cap(sx_, sy_, color, size_px)


renderer._draw_body_disc_outline = spy_disc
renderer._draw_end_cap = spy_cap
renderer.draw_maneuver(camera, w.body)
renderer._draw_body_disc_outline = real_disc
renderer._draw_end_cap = real_cap

caps = [d for d in drawn if d[0] == 'cap']
discs = [d for d in drawn if d[0] == 'disc']
check(len(caps) == 1, "genau eine endkappe fuer die geplante linie",
      f"{len(caps)}")
check(bool(discs), "und mindestens ein koerper-radiuskreis", f"{len(discs)}")
if discs:
    scale = abs(float(camera.scale))
    expected = float(getattr(erde, 'radius', 0.0)) * scale
    hit = [d for d in discs if abs(d[3] - expected) < 1e-6]
    check(bool(hit), "der kreis traegt den ECHTEN radius, keinen pixelwert",
          f"erwartet {expected:.1f} px, gezeichnet {[round(d[3], 1) for d in discs]}")
for entry in caps + discs:
    check(abs(entry[4][0] - _GREEN_R) < 1e-6 and abs(entry[4][1] - _GREEN_G) < 1e-6,
          f"{entry[0]}: in gruen, der farbe der geplanten bahn", "")

renderer.maneuver_end_caps = False
drawn.clear()
renderer._draw_body_disc_outline = spy_disc
renderer._draw_end_cap = spy_cap
renderer.draw_maneuver(camera, w.body)
renderer._draw_body_disc_outline = real_disc
renderer._draw_end_cap = real_cap
check(not drawn, "abgeschaltet zeichnet keine kappen", f"{len(drawn)}")
renderer.maneuver_end_caps = True

print("\n4) die trefferliste wird JEDEN durchgang geleert")
# Sonst steht ein griff noch im bild, den es nicht mehr gibt -- dieselbe
# regel wie bei Renderer.apsis_marker_hits.
plan.clear()
preview.rebuild(plan, ship, w, predictor, erde, 10.0, 0.6)
renderer.draw_maneuver(camera, w.body)
check(renderer.maneuver_node_hits == [],
      "ohne knoten bleibt nichts stehen", f"{len(renderer.maneuver_node_hits)}")

print("\n5) die schirmkurve entsteht NUR waehrend eines MARKER-zugs")
# Nicht bei jedem zug: ein GRIFF verschiebt den knoten gar nicht, er
# aendert nur sein delta-v -- und die kurve kostet eine projektion je
# punkt, also genau waehrend der eingabe, bei der es auf bildrate ankommt.
plan.add(ManeuverNode(t_node, dv_prograde=150.0))
preview.rebuild(plan, ship, w, predictor, erde, 10.0, 0.6)
renderer.draw_maneuver(camera, w.body)
check(_empty(renderer.maneuver_curve_screen),
      "ohne zug wird sie nicht gebaut", "")
renderer.maneuver_drag_active = True
renderer.draw_maneuver(camera, w.body)
check(_empty(renderer.maneuver_curve_screen),
      "ein GRIFF-zug allein baut sie ebenfalls nicht", "")
renderer.maneuver_drag_curve = True
renderer.draw_maneuver(camera, w.body)
curve = renderer.maneuver_curve_screen
check(len(curve) > 8, "erst der marker-zug tut es", f"{len(curve)} punkte")
if len(curve):
    check(bool(np.all(np.isfinite(curve[:, 0]))
               and np.all(np.isfinite(curve[:, 1]))),
          "und ist durchweg endlich", "")
    times = curve[:, 2]
    check(bool(np.all(np.diff(times) >= -1e-9)),
          "die zeiten laufen vorwaerts", "")
renderer.maneuver_drag_active = False
renderer.maneuver_drag_curve = False

print("\n6) abgeschaltet wird nichts gemeldet")
renderer.maneuver_enabled = False
renderer.draw_maneuver(camera, w.body)
check(renderer.maneuver_node_hits == [], "maneuver_enabled = False schweigt", "")
renderer.maneuver_enabled = True

print("\n7) die schubrichtung erscheint als fuenfte orientierungs-richtung")
# Daran haengt der autopilot: _apply_orientation_snap sucht 'node' im
# richtungs-woerterbuch und haelt die nase darauf.
_frame, directions = renderer.orbital_frame_directions(
    ship, reference_body=erde, prediction_points=base)
check('node' not in directions, "ohne scharfen knoten kein eintrag",
      f"{sorted(directions)}")
renderer.maneuver_burn_direction = (0.0, 1.0)
_frame, directions = renderer.orbital_frame_directions(
    ship, reference_body=erde, prediction_points=base)
check('node' in directions, "mit scharfem knoten schon", f"{sorted(directions)}")
if 'node' in directions:
    d = directions['node']
    check(abs(math.hypot(float(d.x), float(d.y)) - 1.0) < 1e-9,
          "und sie ist ein einheitsvektor", "")
renderer.maneuver_burn_direction = None

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for failure in FAILURES:
        print(f"  {failure}")
    pygame.quit()
    sys.exit(1)
print("render/maneuver: alle pruefungen bestanden")
pygame.quit()
