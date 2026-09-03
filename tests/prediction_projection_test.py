"""Der schnelle zeichenweg muss EXAKT dasselbe liefern wie der langsame.

Die projektion der predictor-punkte laeuft seit der optimierung ueber drei
stapel-pfade statt ueber Python-schleifen:

    Renderer._project_prediction_batch     punkte -> bildschirm, en bloc
    Renderer._build_prediction_indices     stichprobe der rohpunkte, numpy
    Renderer._build_clipped_polyline_runs  vorab-aussortierung per maske

Alle drei sind reine beschleunigungen ohne erlaubte abweichung. Dieser test
haelt das fest, indem er dieselbe szene zweimal zeichnet -- einmal mit
abgeschaltetem stapelweg -- und die entstandenen polylinien punktweise
vergleicht. Eine abweichung ist ein fehler, kein rundungsdetail: beide wege
rechnen dieselben operationen in derselben reihenfolge.

Geprueft wird ueber mehrere zoomstufen UND beide bewegten bezugsrahmen, denn
die rahmen bringen je eine eigene stapel-transformation mit.

Aufruf: python tests/prediction_projection_test.py
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
import moderngl
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE

FAILURES = []


def check(condition, label, detail=''):
    if condition:
        print(f"  ok   {label}{(' -- ' + detail) if detail else ''}")
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label} -- {detail}")


W, H = 1280, 800
# Nur display+font -- pygame.init() zaehlt mixer- und joystick-geraete auf
# und kostet dabei ~45 s. Siehe runtime/window.py.
pygame.display.init()
pygame.font.init()
pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL | RESIZABLE, vsync=0)
gl = moderngl.create_context()
gl.enable(moderngl.BLEND)
gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

from physics.vec import G
from ship.camera import Camera
from config.loader import ConfigLoader
from runtime.system_loader import SystemLoader
from ship.predictor import Predictor
from physics.reference_frames import PlottingFrameAdapter, ReferenceFrameSelector
from render.renderer import Renderer
from ship.control import schiffcontrol
from physics.world import world as World

config = ConfigLoader(None)
config.load()
bodies = SystemLoader("solar_system.json").load()
w = World(G)
w.body = bodies
config.apply_to_world(w)
ship = next(b for b in bodies if b.is_ship)
earth = next(b for b in bodies if b.name == "Erde")
sun = next(b for b in bodies if b.name == "Sonne")
control = schiffcontrol(ship)
config.apply_to_ship_control(control)
camera = Camera(None, W, H)
config.apply_to_camera(camera)
camera.follow(ship)
camera.snap_to_targets()
renderer = Renderer(W, H, enable_fxaa=False, ctx=gl)
config.apply_to_renderer(renderer)
predictor = Predictor(recompute_every_update=True, **config.predictor_kwargs())
config.apply_to_predictor(predictor)
predictor.set_length(predictor.num_points * predictor.precision)

adapter = PlottingFrameAdapter(renderer, w.body)
selector = ReferenceFrameSelector(
    lambda fp, tb, tr: adapter.update_plotting_frame(
        fp, target_body_index=tb, target_reference_index=tr)
)
earth_index = w.body.index(earth)
sun_index = w.body.index(sun)


def draw_runs(scale, use_batch, hermite=False, refresh=True):
    """Zeichnet einmal und gibt die entstandenen polylinien zurueck.

    `hermite` ist hier standardmaessig AUS. Die kubische verfeinerung ist
    absichtlich an den stapelweg gebunden (sie braucht
    `to_this_frame_xy_arrays`), also gibt es zu ihr gar keine skalare
    fassung, gegen die man sie vergleichen koennte. Sie muss fuer diesen
    vergleich abgeschaltet sein, damit hier wirklich die PROJEKTION geprueft
    wird und nicht zufaellig nur, dass die verfeinerung in dieser szene
    ohnehin nichts zu tun findet.
    """
    camera.target_scale = scale
    camera.snap_to_targets()
    predictor.set_view_scale(scale)
    # refresh=False laesst die punkteliste des vorherigen aufrufs unveraendert
    # stehen. Beide renders eines vergleichspaars muessen EXAKT dasselbe
    # punkte-array sehen: ein update() im zweiten aufruf koennte ein frisch
    # fertig gewordenes async-ergebnis einwechseln (oder synchron neu rechnen)
    # und die abweichung waere dann rechen-timing, nicht projektion. Frueher
    # war das implizit garantiert, weil jeder zoomwert einen synchronen
    # neuaufbau im ERSTEN aufruf erzwang und der zweite nichts mehr tat.
    if refresh:
        predictor.update(ship, w)
    renderer._frame_time_s = w.time

    original = renderer._project_prediction_batch
    was_hermite = renderer.prediction_hermite_enabled
    renderer.prediction_hermite_enabled = bool(hermite)
    if not use_batch:
        renderer._project_prediction_batch = lambda *a, **k: None
    try:
        renderer.render(w.body, camera, predictor.get_points(),
                        predictor=predictor, sim_time=w.time,
                        ship_control=control, real_dt=1 / 60)
    finally:
        renderer._project_prediction_batch = original
        renderer.prediction_hermite_enabled = was_hermite
    runs = renderer._prediction_line_cache_points or []
    return [np.asarray(r, dtype=np.float64) for r in runs]


print("1. Stapel-projektion == skalare projektion")

for frame_label, setup in (
    ('body-centred non-rotating', lambda: selector.set_to_body_non_rotating(earth_index)),
    ('body-direction', lambda: selector.set_to_body_direction(earth_index, sun_index)),
):
    setup()
    for scale in (2e-9, 2e-8, 6e-8, 2e-7, 2e-6, 2e-5):
        # Beide wege muessen denselben weltzustand sehen -- also erst die
        # welt vorruecken, dann zweimal zeichnen ohne dazwischen zu rechnen.
        w.update_planets(240.0)
        w.update_dynamics(240.0)
        slow = draw_runs(scale, use_batch=False)
        fast = draw_runs(scale, use_batch=True, refresh=False)

        same_runs = len(slow) == len(fast)
        worst = 0.0
        if same_runs:
            for a, b in zip(slow, fast):
                if a.shape != b.shape:
                    same_runs = False
                    break
                if a.size:
                    worst = max(worst, float(np.abs(a - b).max()))
        total = sum(len(r) for r in fast)
        check(same_runs and worst == 0.0,
              f"{frame_label} @ {scale:.0e}: linie identisch",
              f"{len(fast)} laeufe, {total} punkte, groesste abweichung {worst:.3e} px")

print()
print("2. Stichprobe der rohpunkte (numpy == schleife)")


def reference_indices(count, max_scan):
    if count <= 0:
        return []
    if max_scan <= 0 or count <= max_scan:
        return list(range(count))
    if max_scan == 1:
        return [0]
    step = (count - 1) / float(max_scan - 1)
    out = []
    last = -1
    for i in range(max_scan):
        idx = max(0, min(count - 1, int(round(i * step))))
        if idx != last:
            out.append(idx)
            last = idx
    return out


import random

random.seed(20260816)
cases = [(10000, 3000), (9994, 3000), (3000, 3000), (3001, 3000), (5, 3000),
         (1, 3000), (0, 3000), (7, 3), (3000, 1), (12345, 777)]
cases += [(random.randint(1, 20000), random.randint(1, 4000)) for _ in range(200)]
mismatches = [
    (c, m) for c, m in cases
    if list(reference_indices(c, m)) != [int(x) for x in renderer._build_prediction_indices(c, m)]
]
check(not mismatches, "stichprobe stimmt in allen faellen ueberein",
      f"{len(cases)} faelle geprueft, {len(mismatches)} abweichungen")

print()
print("3. Getrennte laeufe beim verlassen und wiederbetreten des bildes")

# Der urspruengliche fehlerbericht: die linie hoert an der ersten bildkante
# auf. Sie MUSS in zwei laeufe zerfallen und darf NICHT quer ueber den
# schirm verbunden werden.
path = [(500, 400), (700, 400), (3000, 400), (6000, 400),
        (3000, 600), (700, 600), (500, 600), (-4000, 600)]
runs = renderer._build_clipped_polyline_runs(path, margin_px=0.0)
check(len(runs) == 2, "verlassen und wiederbetreten ergibt zwei laeufe",
      f"{len(runs)} laeufe")
if len(runs) == 2:
    gap = math.hypot(runs[1][0][0] - runs[0][-1][0], runs[1][0][1] - runs[0][-1][1])
    check(gap > 100.0, "die beiden laeufe sind nicht verbunden",
          f"abstand {gap:.0f} px")

# Mit vorab-maske muss dasselbe herauskommen wie ohne.
coords = (np.array([p[0] for p in path], dtype=np.float64),
          np.array([p[1] for p in path], dtype=np.float64))
runs_masked = renderer._build_clipped_polyline_runs(path, margin_px=0.0, coords=coords)
check([[tuple(p) for p in r] for r in runs] == [[tuple(p) for p in r] for r in runs_masked],
      "vorab-maske aendert das ergebnis nicht",
      f"{len(runs)} vs {len(runs_masked)} laeufe")

print()
print("4. Jede stapel-transformation gegen ihre skalare fassung")

# Direkt auf den rahmen-klassen, damit auch die ueberlagerungs-rahmen
# abgedeckt sind, die nicht als haupt-plotrahmen auftreten. Der
# target-rahmen benennt seine koerper target/reference statt
# primary/secondary -- eine vertauschung faellt genau hier auf.
from physics.reference_frames import (
    BodyCentredBodyDirectionReferenceFrame,
    BodyCentredNonRotatingReferenceFrame,
    IdentityReferenceFrame,
    TargetBodyDirectionReferenceFrame,
)

times = np.linspace(w.time, w.time + 4.0e5, 512)
xs = np.linspace(1.0e11, 1.6e11, 512)
ys = np.linspace(-3.0e10, 5.0e10, 512)

for label, frame in (
    ('Identity', IdentityReferenceFrame()),
    ('BodyCentredNonRotating', BodyCentredNonRotatingReferenceFrame(earth)),
    ('BodyCentredBodyDirection', BodyCentredBodyDirectionReferenceFrame(earth, sun)),
    ('TargetBodyDirection', TargetBodyDirectionReferenceFrame(ship, earth)),
):
    if hasattr(frame, 'set_epoch_time'):
        frame.set_epoch_time(w.time)
    if hasattr(frame, 'set_origin_interp_window'):
        # Fenster aktivieren, sonst gibt es kein knotengitter und der
        # stapelweg meldet (korrekt) 'kann ich nicht'.
        frame.set_origin_interp_window(float(times[0]), float(times[-1]), len(times) * 4)

    batch = frame.to_this_frame_xy_arrays(times, xs, ys)
    if batch is None:
        check(True, f"{label}: kein stapelweg, faellt sauber zurueck")
        continue
    bx, by = batch
    worst = 0.0
    for i in range(0, len(times), 7):
        sx, sy = frame.to_this_frame_xy(float(times[i]), float(xs[i]), float(ys[i]))
        worst = max(worst, abs(sx - float(bx[i])), abs(sy - float(by[i])))
    scale = max(1.0, float(np.abs(np.asarray(batch)).max()))
    check(worst / scale < 1e-12, f"{label}: stapel == skalar",
          f"groesste abweichung {worst:.3e} m (relativ {worst / scale:.2e})")

print()
print("5. Die verfeinerung folgt der linie, statt sie zu verbiegen")

# Die zugesagte GENAUIGKEIT wird gegen die analytische wahrheit gemessen --
# in tests/prediction_detail_test.py, wo die bahn ein bekannter kreis ist.
# Hier geht es um das, was nur die bewegten rahmen brechen koennen: die
# zwischenpunkte werden aus (p0, v0, p1, v1) und dt gebildet, und ein
# falscher dt-massstab (oder eine an der falschen zeit ausgewertete
# rahmen-transformation) laesst sie weit neben die kurve laufen.
#
# Das faellt an der BOGENLAENGE auf, und zwar frei von jeder annahme ueber die
# form der bahn: zwischenpunkte auf der kurve verlaengern den polygonzug nur
# minimal (er schmiegt sich der sehne an), waehrend ausreisser ihn vervielfachen.
selector.set_to_body_non_rotating(earth_index)


def run_length_px(runs):
    total = 0.0
    for run in runs:
        for i in range(len(run) - 1):
            total += math.hypot(run[i + 1][0] - run[i][0],
                                run[i + 1][1] - run[i][1])
    return total


for frame_label, setup in (
    ('body-centred non-rotating', lambda: selector.set_to_body_non_rotating(earth_index)),
    ('body-direction', lambda: selector.set_to_body_direction(earth_index, sun_index)),
):
    setup()
    for scale in (2e-9, 2e-7, 2e-5):
        w.update_planets(240.0)
        w.update_dynamics(240.0)
        coarse = draw_runs(scale, use_batch=True, hermite=False)
        fine = draw_runs(scale, use_batch=True, hermite=True, refresh=False)
        stats = dict(renderer._last_prediction_render_stats or {})
        lc = run_length_px(coarse)
        lf = run_length_px(fine)
        ratio = lf / max(lc, 1e-9)
        # Eine kurve ist laenger als ihre sehnen, aber nur um weniges; das
        # doppelte waere schon grob unphysikalisch.
        check(0.95 <= ratio <= 1.05,
              f"{frame_label} @ {scale:.0e}: bogenlaenge bleibt plausibel",
              f"{ratio:.4f} der groben laenge, "
              f"{stats.get('hermite_added', 0)} zwischenpunkte, "
              f"sprosse {stats.get('detail_eps_m')} m")

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for failure in FAILURES:
        print(f"  {failure}")
    pygame.quit()
    sys.exit(1)
print("prediction-projektion: alle pruefungen bestanden")
pygame.quit()
