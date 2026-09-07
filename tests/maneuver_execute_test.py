"""Der manoever-autopilot: scharfschalten, zuenden, brennen, aufraeumen.

Der test, an dem das ganze feature haengt, ist abschnitt 8: die GEFLOGENE
bahn muss die VORGESCHAUTE treffen. Alles davor sind die voraussetzungen
dafuer.

Abschnitt 1 klaert die einheitenfrage, die man leicht uebersieht:
schiffcontrol.apply_thrust rechnet in ECHT-sekunden, die welt laeuft in
SIM-sekunden und auf der untersten raffungsstufe 60x schneller. Die
beschleunigung des profils ist deshalb thrust_acc/realtime_warp_max und
nicht thrust_acc -- sonst braennte der autopilot sechzigmal so hart wie die
pfeiltaste, und die vorschau zeigte diesen brennvorgang auch noch korrekt.

Aufruf: python tests/maneuver_execute_test.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from config.loader import ConfigLoader
from physics.world import world as World
from runtime.system_loader import SystemLoader
from ship.control import schiffcontrol
from ship.maneuver.executor import ARMED, BURNING, DONE, IDLE, ManeuverExecutor
from ship.maneuver.plan import ManeuverNode, ManeuverPlan
from ship.maneuver.preview import ManeuverPreview, state_on_curve
from ship.predictor import Predictor

FAILURES = []


def check(condition, name, detail=''):
    status = 'OK  ' if condition else 'FEHL'
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ''))
    if not condition:
        FAILURES.append(name)


TICK_RATE = 180.0
REALTIME_WARP_MAX = 60.0
THRUST_ACC = 600.0
RAMP_SECONDS = 0.6
A_MAX_SIM = THRUST_ACC / REALTIME_WARP_MAX
MAX_SUBSTEP = 1000.0


def build():
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
    ship.position.x = erde.position.x + r
    ship.position.y = erde.position.y
    ship.velocity.x = (float(ahead.x) - float(behind.x)) / 2.0
    ship.velocity.y = (float(ahead.y) - float(behind.y)) / 2.0 + math.sqrt(mu / r)

    control = schiffcontrol(ship)
    config.apply_to_ship_control(control)

    predictor = Predictor(**config.predictor_kwargs())
    config.apply_to_predictor(predictor)
    predictor.async_compute = False
    predictor.initialize(ship, w)
    predictor.update(ship, w)
    return config, w, erde, ship, control, predictor


def make_executor(plan, ship, control):
    return ManeuverExecutor(
        plan, ship, control,
        thrust_acc_max=THRUST_ACC,
        realtime_warp_max=REALTIME_WARP_MAX,
        tick_rate=TICK_RATE,
        ramp_seconds=RAMP_SECONDS,
        max_accel=None,
        orient_lead_seconds=5.0,
        burn_step_max_s=0.5,
    )


def armed_setup(dv_prograde=120.0, fraction=0.3):
    """Welt + plan + scharfer ausfuehrer, an einem knoten weit vorn."""
    config, w, erde, ship, control, predictor = build()
    base = predictor.get_points()
    t0 = float(base[0, 2])
    t_node = t0 + (float(base[len(base) - 1, 2]) - t0) * fraction
    plan = ManeuverPlan()
    plan.add(ManeuverNode(t_node, dv_prograde=dv_prograde))
    preview = ManeuverPreview(max_points=4000)
    preview.rebuild(plan, ship, w, predictor, erde, A_MAX_SIM, RAMP_SECONDS)
    ex = make_executor(plan, ship, control)
    ex.arm(w, erde, preview)
    return w, erde, ship, control, predictor, plan, preview, ex, t_node


print("\n1) EINHEITEN -- der autopilot brennt so hart wie die pfeiltaste")
_config, w, erde, ship, control, predictor = build()
plan = ManeuverPlan()
ex = make_executor(plan, ship, control)
a_max = ex.a_max_sim()
check(abs(a_max - A_MAX_SIM) < 1e-12,
      "a_max = thrust_acc / realtime_warp_max", f"{a_max} m/s^2 (sim)")
# Gegenrechnung: eine echtsekunde volle pfeiltaste liefert thrust_acc m/s
# und laesst die welt realtime_warp_max sim-sekunden vorruecken. Dieselbe
# spanne beim autopiloten muss dasselbe delta-v ergeben.
manual_dv = THRUST_ACC * 1.0
auto_dv = a_max * (REALTIME_WARP_MAX * 1.0)
check(abs(manual_dv - auto_dv) < 1e-9,
      "eine echtsekunde beider wege liefert dasselbe delta-v",
      f"{manual_dv} vs {auto_dv}")

print("\n2) ein knoten ohne delta-v laesst sich nicht scharfschalten")
plan.add(ManeuverNode(w.time + 600.0))
check(ex.can_arm() is False, "can_arm() ist False", "")
check(ex.arm(w, erde) is False, "arm() lehnt ab", "")
check(ex.state == IDLE, "und bleibt im leerlauf", f"{ex.state}")

print("\n3) scharfschalten setzt zuendzeitpunkt und richtung")
w, erde, ship, control, predictor, plan, preview, ex, t_node = armed_setup()
check(ex.state == ARMED, "zustand ARMED", f"{ex.state}")
check(abs(ex.t_ignition - (t_node - ex.profile.total_time * 0.5)) < 1e-9,
      "zuendung liegt eine halbe brenndauer vor dem knoten",
      f"t_ign={ex.t_ignition:.3f}, t_node={t_node:.3f}, "
      f"dauer={ex.profile.total_time:.3f}")
check(abs(math.hypot(ex.dir_x, ex.dir_y) - 1.0) < 1e-9,
      "die schubrichtung ist ein einheitsvektor", "")
check(getattr(control, 'snap_mode', None) == 'node',
      "die nase wird auf den knoten gerastet", f"{control.snap_mode}")

print("\n4) die schrittklemme springt NIE ueber die zuendung")
# Ohne sie ruecken ein paar frames im zeitraffer um stunden vor und der
# brennvorgang faellt komplett zwischen zwei bilder.
worst_overshoot = -1e18
t = w.time
guard = 0
while t < ex.t_ignition and guard < 200000:
    nominal = 3600.0            # eine stunde je frame -- grober zeitraffer
    cap = ex.max_sim_seconds_value(t)
    step = nominal if cap is None else min(nominal, cap)
    worst_overshoot = max(worst_overshoot, (t + step) - ex.t_ignition)
    t += step
    guard += 1
check(worst_overshoot <= 1e-9, "kein schritt landet hinter der zuendung",
      f"groesster ueberschuss {worst_overshoot:.3e} s")
check(abs(t - ex.t_ignition) < 1e-6, "die kette landet exakt darauf",
      f"{t:.6f} vs {ex.t_ignition:.6f}")

print("\n5) der volle brennvorgang liefert GENAU das geplante delta-v")
w, erde, ship, control, predictor, plan, preview, ex, t_node = armed_setup()
guard = 0
while ex.is_active and guard < 2000000:
    cap = ex.max_sim_seconds(w)
    step = 1.0 if cap is None else min(1.0, max(1e-9, cap))
    ex.update(w, step)
    w.step(step, MAX_SUBSTEP)
    guard += 1
check(ex.state == DONE, "der brennvorgang endet von selbst", f"{ex.state}")
check(abs(ex.dv_delivered - 120.0) < 1e-6,
      "geliefertes delta-v == geplantes", f"{ex.dv_delivered:.9f}")

print("\n6) aufraeumen nach dem brennen")
check(len(plan) == 0, "der geflogene knoten ist aus dem plan verschwunden",
      f"{len(plan)}")
check(getattr(control, 'snap_mode', None) is None,
      "die rastung ist geloest", f"{control.snap_mode}")
check(abs(float(control.thrust_acc) - THRUST_ACC) < 1e-9,
      "die schubstufe steht wieder auf ihrem alten wert",
      f"{control.thrust_acc}")

print("\n7) abbruch durch handeingabe")
w, erde, ship, control, predictor, plan, preview, ex, t_node = armed_setup()
guard = 0
while ex.state != BURNING and guard < 2000000:
    cap = ex.max_sim_seconds(w)
    step = 1.0 if cap is None else min(1.0, max(1e-9, cap))
    ex.update(w, step)
    w.step(step, MAX_SUBSTEP)
    guard += 1
ex.update(w, 0.2)
w.step(0.2, MAX_SUBSTEP)
partial = ex.dv_delivered
check(partial > 0.0, "es wurde bereits etwas geliefert", f"{partial:.4f}")
check(partial < 120.0, "aber noch nicht alles", f"{partial:.4f}")
ex.notify_manual_input()
check(not ex.is_active, "der autopilot ist aus", f"{ex.state}")
check(len(plan) == 1, "der knoten BLEIBT im plan", f"{len(plan)}")
v_at_abort = (float(ship.velocity.x), float(ship.velocity.y))
ex.update(w, 1.0)
check(abs(float(ship.velocity.x) - v_at_abort[0]) < 1e-12
      and abs(float(ship.velocity.y) - v_at_abort[1]) < 1e-12,
      "nach dem abbruch wird nicht mehr beschleunigt", "")

print("\n8) DIE HAUPTSACHE -- geflogen == vorgeschaut")
w, erde, ship, control, predictor, plan, preview, ex, t_node = armed_setup()
t_check = t_node + 900.0        # 15 minuten nach dem knoten
guard = 0
while w.time < t_check and guard < 2000000:
    cap = ex.max_sim_seconds(w)
    step = 1.0 if cap is None else min(1.0, max(1e-9, cap))
    remaining = t_check - w.time
    if remaining > 1e-9:
        step = min(step, remaining)
    ex.update(w, step)
    w.step(step, MAX_SUBSTEP)
    guard += 1
check(ex.state == DONE, "der brennvorgang ist durch", f"{ex.state}")

expected = state_on_curve(preview.points, t_check)
check(expected is not None, "die vorschau reicht bis zum pruefzeitpunkt", "")
if expected is not None:
    ep = erde.position_at_time(w.time)
    r_flown = math.hypot(float(ship.position.x) - float(ep.x),
                         float(ship.position.y) - float(ep.y))
    dp = math.hypot(float(ship.position.x) - expected[0],
                    float(ship.position.y) - expected[1])
    dv = math.hypot(float(ship.velocity.x) - expected[2],
                    float(ship.velocity.y) - expected[3])
    speed = math.hypot(expected[2], expected[3])
    # Der rest ist die kick-then-drift-diskretisierung: der ausfuehrer legt
    # das delta-v eines schrittes VOR dem schritt an, die vorschau
    # integriert den schub stetig. Der fehler ist durch burn_step_max_s
    # begrenzt und waechst danach nicht mehr.
    check(dv < 1e-3 * speed,
          "die geschwindigkeit trifft die vorschau (< 0.1 %)",
          f"{dv:.4f} m/s auf {speed:.1f} m/s")
    check(dp < 5e-3 * r_flown,
          "die position trifft die vorschau (< 0.5 % des bahnradius)",
          f"{dp:.1f} m auf r={r_flown:.4e} m")

print("\n9) zwei knoten -- EXECUTE nimmt immer den naechsten")
_config, w, erde, ship, control, predictor = build()
base = predictor.get_points()
t0 = float(base[0, 2])
span = float(base[len(base) - 1, 2]) - t0
plan = ManeuverPlan()
early = ManeuverNode(t0 + span * 0.3, dv_prograde=60.0)
late = ManeuverNode(t0 + span * 0.45, dv_prograde=60.0)
plan.add(late)
plan.add(early)
preview = ManeuverPreview(max_points=4000)
preview.rebuild(plan, ship, w, predictor, erde, A_MAX_SIM, RAMP_SECONDS)
ex = make_executor(plan, ship, control)
check(ex.arm(w, erde, preview) is True, "scharfschalten gelingt", "")
check(ex.node is early, "scharf ist der zeitlich naechste knoten", "")
check(len(plan) == 2, "der andere bleibt unangetastet", f"{len(plan)}")

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for failure in FAILURES:
        print(f"  {failure}")
    sys.exit(1)
print("ship/maneuver/executor: alle pruefungen bestanden")
