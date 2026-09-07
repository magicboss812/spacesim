"""Die manoever-vorschau: was die geplanten knoten aus der bahn machen.

Die integration selbst ist in `maneuver_burn_kernel_test.py` geprueft. Hier
geht es um die KETTE darum herum, und die hat genau drei stellen, an denen
sie falsch sein kann:

1. Ein knoten OHNE delta-v darf die bahn nicht veraendern -- tut er es,
   stimmt der zustand nicht, den die kette an der knotenzeit abliest.
2. Die VORZEICHEN muessen stimmen: prograde hebt das apoapsis, retrograde
   senkt das periapsis. Das ist die pruefung, die einen vertauschten
   basisvektor findet.
3. Der ZWEITE knoten muss auf dem ergebnis des ersten sitzen, nicht auf der
   ursprungsbahn. Ohne diese pruefung waere die kette gar keine.

Aufruf: python tests/maneuver_preview_test.py
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np

from config.loader import ConfigLoader
from physics.world import world as World
from runtime.system_loader import SystemLoader
from ship.maneuver.plan import ManeuverNode, ManeuverPlan
from ship.maneuver.preview import ManeuverPreview, state_on_curve
from ship.predictor import Predictor

FAILURES = []


def check(condition, name, detail=''):
    status = 'OK  ' if condition else 'FEHL'
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ''))
    if not condition:
        FAILURES.append(name)


def build():
    """Welt + schiff auf einer kreisbahn um die Erde + predictor."""
    config = ConfigLoader()
    config.load()
    bodies = SystemLoader(
        config.get('simulation.system_file', 'solar_system.json')).load()
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
    evx = (float(ahead.x) - float(behind.x)) / 2.0
    evy = (float(ahead.y) - float(behind.y)) / 2.0
    ship.position.x = erde.position.x + r
    ship.position.y = erde.position.y
    ship.velocity.x = evx
    ship.velocity.y = evy + math.sqrt(mu / r)

    predictor = Predictor(**config.predictor_kwargs())
    config.apply_to_predictor(predictor)
    predictor.async_compute = False
    predictor.initialize(ship, w)
    predictor.update(ship, w)
    return w, erde, ship, predictor


def radius_range(points, erde):
    """Kleinster und groesster abstand der linie zur Erde."""
    lo, hi = None, None
    for i in range(0, len(points), 3):
        t = float(points[i, 2])
        ep = erde.position_at_time(t)
        d = math.hypot(float(points[i, 0]) - float(ep.x),
                       float(points[i, 1]) - float(ep.y))
        lo = d if lo is None else min(lo, d)
        hi = d if hi is None else max(hi, d)
    return lo, hi


print("\n0) aufbau")
w, erde, ship, predictor = build()
base = predictor.get_points()
check(base is not None and len(base) > 100,
      "der predictor liefert eine grundlinie",
      f"{0 if base is None else len(base)} punkte")
t0 = float(base[0, 2])
t_end = float(base[len(base) - 1, 2])
t_node = t0 + (t_end - t0) * 0.35
print(f"      horizont {t0:.1f} .. {t_end:.1f} s, knoten bei {t_node:.1f} s")

print("\n1) state_on_curve trifft die stuetzstellen exakt")
i = len(base) // 3
st = state_on_curve(base, float(base[i, 2]))
check(st is not None, "eine stuetzstelle laesst sich lesen", "")
err = math.hypot(st[0] - float(base[i, 0]), st[1] - float(base[i, 1]))
check(err < 1e-6, "auf der stuetzstelle == die stuetzstelle", f"{err:.3e} m")
check(state_on_curve(base, t0 - 1000.0) is None, "vor der linie -> None", "")
check(state_on_curve(base, t_end + 1000.0) is None, "hinter der linie -> None", "")

print("\n2) ein knoten OHNE delta-v laesst die bahn stehen")
plan = ManeuverPlan()
plan.add(ManeuverNode(t_node))
preview = ManeuverPreview(max_points=4000)
ok = preview.rebuild(plan, ship, w, predictor, erde, 10.0, 0.6)
check(ok, "die vorschau meldet erfolg", "")
check(len(preview.node_markers) == 1, "ein marker steht", "")
check(preview.points is None,
      "aber KEINE linie -- ein platzhalter aendert nichts an der bahn", "")

print("\n3) VORZEICHEN -- prograde hebt, retrograde senkt")
base_lo, base_hi = radius_range(base, erde)
print(f"      grundbahn r = {base_lo:.4e} .. {base_hi:.4e} m")

plan = ManeuverPlan()
plan.add(ManeuverNode(t_node, dv_prograde=+400.0))
preview.rebuild(plan, ship, w, predictor, erde, 10.0, 0.6)
check(preview.points is not None, "mit delta-v entsteht eine linie", "")
pro_lo, pro_hi = radius_range(preview.points, erde)
check(pro_hi > base_hi * 1.02, "prograde hebt das apoapsis",
      f"{base_hi:.4e} -> {pro_hi:.4e}")

plan = ManeuverPlan()
plan.add(ManeuverNode(t_node, dv_prograde=-400.0))
preview.rebuild(plan, ship, w, predictor, erde, 10.0, 0.6)
ret_lo, ret_hi = radius_range(preview.points, erde)
check(ret_lo < base_lo * 0.98, "retrograde senkt das periapsis",
      f"{base_lo:.4e} -> {ret_lo:.4e}")

print("\n4) normal wirkt SENKRECHT -- es kippt die bahn")
plan = ManeuverPlan()
plan.add(ManeuverNode(t_node, dv_normal=+400.0))
preview.rebuild(plan, ship, w, predictor, erde, 10.0, 0.6)
nrm_lo, nrm_hi = radius_range(preview.points, erde)
# Ein rein einwaerts gerichteter impuls auf einer kreisbahn senkt das
# periapsis UND hebt das apoapsis: er dreht die bahn, statt sie nur zu
# vergroessern. Prograde tut das nicht.
check(nrm_lo < base_lo * 0.98 and nrm_hi > base_hi * 1.005,
      "beide apsiden wandern",
      f"lo {base_lo:.3e}->{nrm_lo:.3e}, hi {base_hi:.3e}->{nrm_hi:.3e}")

print("\n5) der knotenmarker sitzt auf der bahn")
plan = ManeuverPlan()
plan.add(ManeuverNode(t_node, dv_prograde=250.0))
preview.rebuild(plan, ship, w, predictor, erde, 10.0, 0.6)
check(len(preview.node_markers) == 1, "ein marker je knoten", "")
m = preview.node_markers[0]
ref = state_on_curve(base, m['t_node'])
d = math.hypot(m['x'] - ref[0], m['y'] - ref[1])
check(d < 1e-3 * base_hi, "der marker liegt auf der grundlinie", f"{d:.1f} m")
check(abs(math.hypot(m['dir_x'], m['dir_y']) - 1.0) < 1e-9,
      "die schubrichtung ist ein einheitsvektor", "")
check(abs(m['dv'] - 250.0) < 1e-9, "der betrag ist das delta-v", f"{m['dv']}")
check(abs(m['pro_x'] * m['nrm_x'] + m['pro_y'] * m['nrm_y']) < 1e-9,
      "prograde und normal stehen senkrecht aufeinander", "")

print("\n6) KETTE -- der zweite knoten sitzt auf dem ergebnis des ersten")
plan = ManeuverPlan()
plan.add(ManeuverNode(t_node, dv_prograde=400.0))
preview.rebuild(plan, ship, w, predictor, erde, 10.0, 0.6)
first_reach = float(preview.points[len(preview.points) - 1, 2])
print(f"      die gleitphase nach knoten 1 reicht bis {first_reach:.1f} s")
# Den zweiten knoten INNERHALB dieser reichweite setzen -- weiter draussen
# hat die kette keinen zustand zum ablesen (siehe abschnitt 10).
t_node2 = t_node + (first_reach - t_node) * 0.5
plan = ManeuverPlan()
plan.add(ManeuverNode(t_node, dv_prograde=400.0))
plan.add(ManeuverNode(t_node2, dv_prograde=100.0))
preview.rebuild(plan, ship, w, predictor, erde, 10.0, 0.6)
check(len(preview.node_markers) == 2, "zwei marker",
      f"{len(preview.node_markers)}")
if len(preview.node_markers) == 2:
    m2 = preview.node_markers[1]
    on_base = state_on_curve(base, m2['t_node'])
    dist_to_base = (math.hypot(m2['x'] - on_base[0], m2['y'] - on_base[1])
                    if on_base is not None else float('inf'))
    check(dist_to_base > 1e-2 * base_hi,
          "der zweite marker liegt NICHT mehr auf der ursprungsbahn",
          f"{dist_to_base:.3e} m entfernt")

print("\n7) die linie ist zeitlich monoton und endlich")
check(bool(np.all(np.isfinite(preview.points))), "keine NaN/inf in der linie", "")
times = preview.points[:, 2]
check(bool(np.all(np.diff(times) >= -1e-9)), "die zeit laeuft nur vorwaerts", "")
check(len(preview.points) <= 4000 + 600, "das punktbudget wird eingehalten",
      f"{len(preview.points)}")

print("\n8) maybe_rebuild rechnet NICHT ohne aenderung")
# Eine kette aus fuenf integrationen je frame waere der teuerste posten im
# bild. Sie darf nur laufen, wenn plan, grundbahn oder reichweite sich
# bewegt haben. Synchron geprueft, damit die antwort nicht davon abhaengt,
# wie schnell der arbeiter gerade fertig wird.
gate = ManeuverPreview(max_points=1500, async_compute=False)
did = gate.maybe_rebuild(plan, ship, w, predictor, erde, 10.0, 0.6,
                         now_wall=1000.0)
check(did is True, "der erste aufruf rechnet", "")
did = gate.maybe_rebuild(plan, ship, w, predictor, erde, 10.0, 0.6,
                         now_wall=1000.5)
check(did is False, "unveraendert -> kein neuaufbau", "")
plan.nodes[0].dv_prograde = 401.0
plan.touch()
did = gate.maybe_rebuild(plan, ship, w, predictor, erde, 10.0, 0.6,
                         now_wall=1001.0)
check(did is True, "nach touch() -> neuaufbau", "")
did = gate.maybe_rebuild(plan, ship, w, predictor, erde, 10.0, 0.6,
                         now_wall=1001.5)
check(did is False, "und danach wieder nicht", "")
gate.set_length_mult(gate.length_mult * 2.0)
did = gate.maybe_rebuild(plan, ship, w, predictor, erde, 10.0, 0.6,
                         now_wall=1002.0)
check(did is True, "eine neue REICHWEITE zaehlt genauso", "")
gate.set_length_mult(1.0)

print("\n8b) nebenlaeufig gerechnet, aber mit demselben ergebnis")
# Der neuaufbau kostet gemessen 7-15 ms. Im hauptthread gerechnet fiel die
# bildrate beim ziehen eines griffs von 100 auf 40 -- er laeuft deshalb in
# einem arbeitsthread (alle kernel sind nogil). Was dabei NICHT passieren
# darf: eine andere linie als auf dem synchronen weg.
sync = ManeuverPreview(max_points=1500, async_compute=False)
sync.rebuild(plan, ship, w, predictor, erde, 10.0, 0.6)
job = ManeuverPreview(max_points=1500, async_compute=True)
t_call = time.perf_counter()
job.maybe_rebuild(plan, ship, w, predictor, erde, 10.0, 0.6, now_wall=2000.0)
call_ms = (time.perf_counter() - t_call) * 1000.0
check(job.points is None, "der aufruf gibt SOFORT zurueck, noch ohne linie",
      f"{call_ms:.3f} ms im hauptthread")
check(job.wait(10.0) is True, "das ergebnis kommt nach", "")
check(job.points is not None and len(job.points) == len(sync.points),
      "gleich viele punkte wie synchron",
      f"{0 if job.points is None else len(job.points)} gegen {len(sync.points)}")
if job.points is not None and len(job.points) == len(sync.points):
    diff = float(np.max(np.abs(job.points - sync.points)))
    check(diff == 0.0, "und exakt dieselben zahlen", f"groesste abweichung {diff}")
check(call_ms < sync.last_rebuild_ms * 0.5,
      "der hauptthread bezahlt einen bruchteil der rechnung",
      f"{call_ms:.3f} ms gegen {sync.last_rebuild_ms:.2f} ms")
job.shutdown()

print("\n8c) die REICHWEITE haengt am eigenen regler, nicht an der punktzahl")
# Sie sitzt im punktabstand (`precision`, eine bogenlaenge), nicht in der
# punktzahl: reichweite = abstand x punkte, und nur der abstand ist ohne
# kosten je punkt zu haben.
spans = []
for mult in (1.0, 2.0, 4.0):
    far = ManeuverPreview(max_points=1500, length_mult=mult,
                          async_compute=False)
    far.rebuild(plan, ship, w, predictor, erde, 10.0, 0.6)
    spans.append((mult, far.span_seconds, len(far.points)))
    print(f"       x{mult:<4g} spanne {far.span_seconds / 3600.0:8.2f} h, "
          f"{len(far.points)} punkte")
check(all(s is not None for _m, s, _n in spans), "jede stufe liefert eine spanne", "")
check(spans[1][1] > spans[0][1] * 1.8 and spans[2][1] > spans[1][1] * 1.8,
      "verdoppelter multiplikator = rund verdoppelte reichweite",
      f"{spans[0][1]:.0f} -> {spans[1][1]:.0f} -> {spans[2][1]:.0f} s")
check(len({n for _m, _s, n in spans}) == 1,
      "und die punktzahl bleibt dabei GLEICH",
      f"{[n for _m, _s, n in spans]}")

print("\n9) ein knoten AUSSERHALB der reichweite wird ehrlich fallengelassen")
# Die kette bricht ab, statt einen zustand zu erfinden. Das ist die
# dokumentierte grenze, nicht ein fehler -- geprueft, damit sie nicht
# unbemerkt zu geratenen zahlen wird.
plan = ManeuverPlan()
plan.add(ManeuverNode(t_node, dv_prograde=400.0))
plan.add(ManeuverNode(t_end - 1.0, dv_prograde=100.0))
preview.rebuild(plan, ship, w, predictor, erde, 10.0, 0.6)
check(len(preview.node_markers) == 1,
      "nur der erreichbare knoten bekommt einen marker",
      f"{len(preview.node_markers)}")

print("\n10) ein leerer plan liefert keine linie")
preview.rebuild(ManeuverPlan(), ship, w, predictor, erde, 10.0, 0.6)
check(preview.points is None and not preview.valid,
      "kein knoten -> nichts zu zeichnen", "")

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for failure in FAILURES:
        print(f"  {failure}")
    sys.exit(1)
print("ship/maneuver/preview: alle pruefungen bestanden")
