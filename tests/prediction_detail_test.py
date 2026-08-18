"""Die vorhersagelinie wird so fein gezeichnet, wie der schirm es zeigt.

Gemessen wird eine GEOMETRIE, keine implementierung: der abstand der
GEZEICHNETEN polylinie von der wahren bahn. Die szene ist deshalb ein
synthetischer kreis -- ein schwerer koerper im ursprung, ein schiff auf einer
exakt kreisfoermigen bahn -- denn nur dort ist "die wahrheit" analytisch
bekannt und der fehler nicht selbst wieder ein integrationsergebnis.

Der gezeichnete fehler zerfaellt in zwei UNABHAENGIGE anteile:

    fehler = |Hermite - wahrheit|   +   |polygon - Hermite|
             (haengt am PUNKTABSTAND)    (haengt an der UNTERTEILUNG)

Die zeichenzeit-unterteilung kann nur den zweiten term druecken. Der erste ist
die klassische schranke des kubischen polynoms, ``c^4 / (384 R^3)``. Wer das
verwechselt, verspricht der leiter sprossen (1 mm, 1 cm), die ohne NEUE
integration gar nicht erreichbar sind -- deshalb hat dieser test einen eigenen
abschnitt fuer den boden.

Geprueft wird:

1. **Die toleranz-leiter rastet richtig.** Gewaehlt wird die groesste sprosse,
   die den bildschirm-wunsch noch einhaelt; darunter/darueber wird geklemmt.
2. **Der interpolations-boden stimmt mit dem gesetz ueberein.** Sonst meldet
   der renderer eine genauigkeit, die die linie nicht hat.
3. **Die gezeichnete linie haelt die zugesagte toleranz ein** -- und die
   kubische fassung ist um groessenordnungen naeher an der bahn als die
   lineare, bei GLEICHEM punktabstand.
4. **Der horizont ueberlebt jede zoomstufe.** Detail darf die linie nie
   verkuerzen (derselbe fehler, den `_horizon_spacing_floor` schon einmal
   abfangen musste).
5. **Ein knappes punktbudget macht die linie GROEBER, nicht KUERZER.**
6. **Was ausserhalb des bildes liegt, wird nicht verfeinert.**

Aufruf: python tests/prediction_detail_test.py
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
# und kostet dabei ~45 s. Siehe test.py.
pygame.display.init()
pygame.font.init()
pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL | RESIZABLE, vsync=0)
gl = moderngl.create_context()
gl.enable(moderngl.BLEND)
gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

from vec import Vec2, G
from bodies import body, schiff
from camera import Camera
from loader import ConfigLoader
from predictor import Predictor
from rendering import Renderer
from schiff import schiffcontrol
from world import world as World

# ---------------------------------------------------------------- szene ----
# Ein kreis, dessen radius und mittelpunkt wir exakt kennen. Erdmasse und ein
# erdnaher radius, weil genau dort die sehnen-naeherung am schlimmsten ist:
# c^2/8R = 17.8 km bei 1000 km punktabstand.
CENTRAL_MASS = 5.972e24
ORBIT_R = 7.0e6
ORBIT_V = math.sqrt(G * CENTRAL_MASS / ORBIT_R)
PERIOD = 2.0 * math.pi * ORBIT_R / ORBIT_V
SPACING = 1.0e6          # punktabstand des predictors
# Knapp unter einer umrundung. Der horizont ist hier absichtlich kurz und die
# integration auf "accurate" gestellt: gemessen werden soll die INTERPOLATION,
# und der eigene fehler des integrators muss dafuer unter deren boden bleiben.
# Ueber neun umrundungen (n = 400) waren es 1316 m gegen einen boden von
# 7.6 m -- die messung haette dann den integrator beschrieben, nicht die linie.
NUM_POINTS = 40

config = ConfigLoader(None)
config.load()

planet = body("Zentralkoerper", CENTRAL_MASS, 6.371e6,
              Vec2(0.0, 0.0), Vec2(0.0, 0.0), fixed=True)
ship = schiff("Testschiff", Vec2(ORBIT_R, 0.0), Vec2(0.0, ORBIT_V))

w = World(G)
w.body = [planet, ship]
config.apply_to_world(w)

control = schiffcontrol(ship)
config.apply_to_ship_control(control)

camera = Camera(None, W, H)
config.apply_to_camera(camera)
camera.follow(ship)

renderer = Renderer(W, H, enable_fxaa=False, ctx=gl)
config.apply_to_renderer(renderer)
renderer.hud_enabled = False

predictor = Predictor(recompute_every_update=True, **config.predictor_kwargs())
config.apply_to_predictor(predictor)
predictor.async_compute = False
# Zoom-automatik aus: dieser test will einen BEKANNTEN punktabstand messen,
# nicht einen, den der zoom nebenbei verschiebt.
predictor.auto_precision_from_zoom = False
predictor.set_integrator_quality("accurate")
predictor.set_precision(SPACING)
predictor.set_num_points(NUM_POINTS)
predictor.set_length(NUM_POINTS * SPACING)


def frame(scale, hermite=True, detail_scale=1.0, budget=None):
    """Eine szene zeichnen und die entstandenen polylinien zurueckgeben."""
    camera.target_scale = scale
    camera.snap_to_targets()
    predictor.set_view_scale(scale)
    predictor.reset()
    predictor.update(ship, w)
    renderer._frame_time_s = w.time

    was_h = renderer.prediction_hermite_enabled
    was_d = renderer.prediction_detail_scale
    was_b = renderer.prediction_render_max_draw_points
    was_s = renderer.prediction_sampling_max_points
    renderer.prediction_hermite_enabled = bool(hermite)
    renderer.prediction_detail_scale = float(detail_scale)
    if budget is not None:
        renderer.prediction_render_max_draw_points = int(budget)
        renderer.prediction_sampling_max_points = int(budget)
    try:
        renderer.render(w.body, camera, predictor.get_points(),
                        predictor=predictor, sim_time=w.time,
                        ship_control=control, real_dt=1 / 60)
    finally:
        renderer.prediction_hermite_enabled = was_h
        renderer.prediction_detail_scale = was_d
        renderer.prediction_render_max_draw_points = was_b
        renderer.prediction_sampling_max_points = was_s

    runs = renderer._prediction_line_cache_points or []
    stats = dict(renderer._last_prediction_render_stats or {})
    return [np.asarray(r, dtype=np.float64) for r in runs], stats


def screen_to_world(points, scale):
    """Bildschirm -> welt. Gilt nur beim identitaets-rahmen (siehe unten)."""
    cam = renderer._frame_camera_xy(camera)
    x = cam[0] + (points[:, 0] - W * 0.5) / scale
    y = cam[1] - (points[:, 1] - H * 0.5) / scale
    return x, y


def drawn_radius_error(runs, scale, samples_per_segment=24):
    """Groesster abstand der GEZEICHNETEN linie vom wahren kreis, in metern.

    Es genuegt NICHT, die abweichung an den stuetzstellen zu messen: gezeichnet
    wird die strecke dazwischen, und genau dort sitzt der sehnenfehler. Also
    wird jedes segment abgetastet.
    """
    worst = 0.0
    for run in runs:
        if run.shape[0] < 2:
            continue
        x, y = screen_to_world(run, scale)
        for i in range(run.shape[0] - 1):
            t = np.linspace(0.0, 1.0, samples_per_segment)
            px = x[i] + (x[i + 1] - x[i]) * t
            py = y[i] + (y[i + 1] - y[i]) * t
            r = np.hypot(px, py)
            worst = max(worst, float(np.abs(r - ORBIT_R).max()))
    return worst


def drawn_arc_length(runs, scale):
    """Gezeichnete bogenlaenge in metern (ueber alle laeufe)."""
    total = 0.0
    for run in runs:
        if run.shape[0] < 2:
            continue
        x, y = screen_to_world(run, scale)
        total += float(np.hypot(np.diff(x), np.diff(y)).sum())
    return total


# Zoomstufe, bei der die ganze bahn ins bild passt.
FIT_SCALE = H / (2.6 * ORBIT_R)

print("1. Die toleranz-leiter rastet auf der richtigen sprosse")

ladder = renderer.prediction_error_ladder_m
print(f"     leiter: {ladder}")
cases = [
    # (view_scale px/m, erwartete sprosse in m, was das darstellt)
    (0.3 / 150000.0, 1000.0, "wunsch 150 km -> auf 1 km geklemmt"),
    (0.3 / 300.0, 100.0, "wunsch 300 m -> sprosse 100 m"),
    (0.3 / 3.0, 1.0, "wunsch 3 m -> sprosse 1 m"),
    (0.3 / 0.05, 0.01, "wunsch 5 cm -> sprosse 1 cm"),
    (0.3 / 1e-6, 0.001, "wunsch 1 um -> auf 1 mm geklemmt"),
]


class _FakeCam:
    def __init__(self, scale):
        self.scale = scale


for scale, expected, label in cases:
    got = renderer._prediction_error_budget(_FakeCam(scale))
    check(got is not None and abs(got[0] - expected) < 1e-12,
          f"leiter: {label}",
          f"view_scale {scale:.3e} -> {None if got is None else got[0]} m (erwartet {expected} m)")

# Die sprosse ist ein VERSPRECHEN, kein wunsch: sie darf nie groeber sein als
# angefragt, solange die leiter reicht.
violations = []
for scale in np.logspace(-9, 3, 200):
    got = renderer._prediction_error_budget(_FakeCam(float(scale)))
    if got is None:
        continue
    wanted = 0.3 / float(scale)
    if got[0] > wanted and got[0] > ladder[0]:
        violations.append((scale, wanted, got[0]))
check(not violations, "leiter ist nie groeber als angefragt (ausser am boden)",
      f"200 zoomstufen geprueft, {len(violations)} verletzungen")

# Quantisierung: ueber eine ganze zoom-geste darf sich das ziel nur wenige
# male aendern -- sonst wird jede zoom-stufe neu gerechnet.
rungs_seen = {renderer._prediction_error_budget(_FakeCam(float(s)))[0]
              for s in np.logspace(-7, -4, 400)}
check(len(rungs_seen) <= 4,
      "quantisierung: das ziel springt selten, statt stetig zu wandern",
      f"{len(rungs_seen)} verschiedene sprossen ueber 400 zoomstufen")

print()
print("2. Der gemeldete interpolations-boden folgt c^4 / (384 R^3)")

predictor.set_view_scale(FIT_SCALE)
predictor.reset()
predictor.update(ship, w)
pts = predictor.get_points()

check(pts.shape[1] >= 5, "die punkteliste traegt die tangenten",
      f"{pts.shape[1]} spalten")
check(bool(np.isfinite(pts[:, 3:5]).all()),
      "alle tangenten sind endlich",
      f"{int(np.isfinite(pts[:, 3:5]).all(axis=1).sum())} von {pts.shape[0]}")

# Die stuetzstellen selbst muessen auf dem kreis liegen -- sonst misst
# abschnitt 3 den integrator und nicht die interpolation.
raw_err = float(np.abs(np.hypot(pts[:, 0], pts[:, 1]) - ORBIT_R).max())
spacing = float(np.median(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))))
law = spacing ** 4 / (384.0 * ORBIT_R ** 3)
reported = predictor.interpolation_error_floor()

check(raw_err < law, "der integrator ist genauer als der interpolations-boden",
      f"integrator {raw_err:.4g} m < boden {law:.4g} m")
check(reported is not None and abs(reported - law) / law < 0.05,
      "gemeldeter boden == gesetz",
      f"gemeldet {reported:.4g} m, gesetz {law:.4g} m, "
      f"punktabstand {spacing:.4g} m")

# Der boden haengt am PUNKTABSTAND, nicht an der laenge der liste. Wer beim
# ausduennen ganze stuetzstellen ueberspringt, misst einen groesseren abstand
# und meldet einen zu hohen boden (c^4 -- ein faktor 1.6 im abstand sind
# schon 6.5 im ergebnis).
coarse = predictor.interpolation_error_floor(sample_limit=8)
check(abs(coarse - reported) / reported < 0.05,
      "der boden haengt nicht davon ab, wie fein man ihn abtastet",
      f"8 tripel: {coarse:.4g} m, alle: {reported:.4g} m")

print()
print("3. Die gezeichnete linie haelt die zugesagte toleranz ein")

for zoom_factor in (1.0, 4.0, 16.0):
    scale = FIT_SCALE * zoom_factor
    eps_m = renderer._prediction_error_budget(_FakeCam(scale))[0]

    runs_lin, _ = frame(scale, hermite=False)
    runs_cub, stats = frame(scale, hermite=True)
    err_lin = drawn_radius_error(runs_lin, scale)
    err_cub = drawn_radius_error(runs_cub, scale)

    check(err_cub <= eps_m,
          f"zoom x{zoom_factor:g}: kubisch bleibt unter der sprosse",
          f"fehler {err_cub:.4g} m, sprosse {eps_m:g} m "
          f"({err_cub * scale:.3f} px)")
    check(err_cub * 10.0 < err_lin,
          f"zoom x{zoom_factor:g}: kubisch schlaegt linear um >10x",
          f"linear {err_lin:.4g} m -> kubisch {err_cub:.4g} m "
          f"(faktor {err_lin / max(err_cub, 1e-12):.0f}x), "
          f"{stats.get('hermite_added', 0)} punkte ergaenzt")
    # Mehr detail-budget muss die linie auch wirklich naeher an die bahn
    # bringen -- bis an den interpolations-boden heran und nicht weiter. Das
    # ist der beweis, dass die UNTERTEILUNG die schranke ist und nicht
    # irgendeine andere stufe der kette, die still dagegenarbeitet.
    runs_max, _ = frame(scale, hermite=True, detail_scale=100.0)
    err_max = drawn_radius_error(runs_max, scale)
    check(err_max <= 3.0 * (law + raw_err),
          f"zoom x{zoom_factor:g}: mit vollem budget laeuft es auf den boden zu",
          f"fehler {err_max:.4g} m gegen boden+integrator {law + raw_err:.4g} m "
          f"(ohne extra-budget {err_cub:.4g} m)")

print()
print("4. Detail verkuerzt den horizont nicht")

horizon = NUM_POINTS * SPACING
for scale in (FIT_SCALE * 0.02, FIT_SCALE * 0.2, FIT_SCALE,
              FIT_SCALE * 8.0, FIT_SCALE * 64.0):
    runs, _ = frame(scale, hermite=True)
    # Nur der sichtbare teil wird gezeichnet; verglichen wird deshalb gegen
    # den sichtbaren teil der LINEAREN fassung, die denselben horizont hat.
    runs_lin, _ = frame(scale, hermite=False)
    drawn = drawn_arc_length(runs, scale)
    drawn_lin = drawn_arc_length(runs_lin, scale)
    ratio = drawn / max(drawn_lin, 1e-9)
    check(ratio >= 0.99,
          f"view_scale {scale:.2e}: verfeinerung kuerzt die linie nicht",
          f"{ratio:.4f} der linearen laenge "
          f"({drawn:.3e} von {horizon:.3e} m horizont)")

print()
print("5. Ein knappes budget macht die linie groeber, nicht kuerzer")

scale = FIT_SCALE * 16.0
full, _ = frame(scale, hermite=True)
full_len = drawn_arc_length(full, scale)
for budget in (2000, 600, 200):
    tight, stats = frame(scale, hermite=True, budget=budget)
    tight_len = drawn_arc_length(tight, scale)
    drawn_points = sum(len(r) for r in tight)
    ratio = tight_len / max(full_len, 1e-9)
    check(ratio >= 0.99,
          f"budget {budget}: die linie behaelt ihre laenge",
          f"{ratio:.4f} der vollen laenge, {drawn_points} punkte gezeichnet")

print()
print("6. Was ausserhalb des bildes liegt, wird nicht verfeinert")

# Zwei laeufe bei GLEICHER zoomstufe: einmal auf die bahn zentriert, einmal so
# weit daneben, dass fast nichts mehr im bild ist. Verfeinert werden darf nur,
# was man sieht.
scale = FIT_SCALE * 16.0
_, stats_centre = frame(scale, hermite=True)
added_centre = int(stats_centre.get('hermite_added', 0))

camera.follow(None)
camera.target_position = Vec2(400.0 * ORBIT_R, 0.0)
camera.snap_to_targets()
predictor.set_view_scale(scale)
predictor.reset()
predictor.update(ship, w)
renderer._frame_time_s = w.time
renderer.render(w.body, camera, predictor.get_points(), predictor=predictor,
                sim_time=w.time, ship_control=control, real_dt=1 / 60)
stats_off = dict(renderer._last_prediction_render_stats or {})
added_off = int(stats_off.get('hermite_added', 0))
camera.follow(ship)
camera.snap_to_targets()

check(added_off * 20 < max(added_centre, 1),
      "ausserhalb des bildes faellt die verfeinerung praktisch weg",
      f"im bild {added_centre} zwischenpunkte, daneben {added_off}")

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for failure in FAILURES:
        print(f"  {failure}")
    pygame.quit()
    sys.exit(1)
print("vorhersage-detail: alle pruefungen bestanden")
pygame.quit()
