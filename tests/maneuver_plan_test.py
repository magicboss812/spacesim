"""Der manoeverplan: knoten, reihenfolge, und die orbitale basis.

Was hier geprueft wird, ist NICHT die physik der bahn, sondern die
buchhaltung darum herum -- und die eine geometrische festlegung, an der
rosette, ring-marker und ziehgriffe gemeinsam haengen: das VORZEICHEN von
normal. Laeuft es hier anders als in
reference_frames.apparent_orbital_directions(), zeigt der griff nach oben
und das schiff brennt nach unten.

Aufruf: python tests/maneuver_plan_test.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from ship.maneuver.plan import (
    MIN_EXECUTABLE_DV,
    ManeuverNode,
    ManeuverPlan,
    burn_direction_world,
    orbital_basis,
)

FAILURES = []


def check(condition, name, detail=''):
    status = 'OK  ' if condition else 'FEHL'
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ''))
    if not condition:
        FAILURES.append(name)


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


print("\n1) die orbitale basis einer kreisbahn")
# Schiff bei (r, 0) relativ zum bezugskoerper, geschwindigkeit nach +y.
# Prograde ist dann (0, 1); normal zeigt NACH INNEN, also (-1, 0).
basis = orbital_basis(1.0e7, 0.0, 0.0, 7.5e3)
check(basis is not None, "basis existiert", "")
px, py, nx, ny = basis
check(close(px, 0.0) and close(py, 1.0),
      "prograde = einheitsvektor der relativgeschwindigkeit", f"({px:.6f}, {py:.6f})")
check(close(nx, -1.0) and close(ny, 0.0),
      "normal zeigt NACH INNEN (zum bezugskoerper)", f"({nx:.6f}, {ny:.6f})")
check(close(px * nx + py * ny, 0.0), "prograde steht senkrecht auf normal", "")
check(close(math.hypot(px, py), 1.0) and close(math.hypot(nx, ny), 1.0),
      "beide sind einheitsvektoren", "")

print("\n2) das vorzeichen von normal bleibt einwaerts, egal wo auf der bahn")
worst = 1.0
for k in range(16):
    ang = 2.0 * math.pi * k / 16.0
    rx, ry = 1.0e7 * math.cos(ang), 1.0e7 * math.sin(ang)
    vx, vy = -7.5e3 * math.sin(ang), 7.5e3 * math.cos(ang)
    b = orbital_basis(rx, ry, vx, vy)
    r = math.hypot(rx, ry)
    dot = b[2] * (-rx / r) + b[3] * (-ry / r)
    worst = min(worst, dot)
check(worst > 0.999, "normal . einwaerts > 0 an 16 punkten", f"kleinster wert {worst:.6f}")

print("\n3) entartete faelle")
check(orbital_basis(1.0e7, 0.0, 0.0, 0.0) is None,
      "geschwindigkeit null -> keine basis", "")
# Radial fliegend: es gibt keine eindeutige 'innen'-richtung senkrecht zu v.
# Die funktion muss trotzdem eine ORTHONORMALE basis liefern und darf nicht
# None werden, sonst laesst sich auf einer radialbahn kein knoten setzen.
b = orbital_basis(1.0e7, 0.0, 5.0e3, 0.0)
check(b is not None, "radial fliegend -> basis existiert trotzdem", "")
check(close(b[0] * b[2] + b[1] * b[3], 0.0), "und ist orthogonal", "")

print("\n4) die richtung aus (prograde, normal)")
basis = orbital_basis(1.0e7, 0.0, 0.0, 7.5e3)
dx, dy, mag = burn_direction_world(basis, 100.0, 0.0)
check(close(dx, 0.0) and close(dy, 1.0) and close(mag, 100.0, 1e-9),
      "reines prograde = die bahnrichtung", f"({dx:.6f}, {dy:.6f}) |{mag:.4f}|")
dx, dy, mag = burn_direction_world(basis, -100.0, 0.0)
check(close(dx, 0.0) and close(dy, -1.0) and close(mag, 100.0, 1e-9),
      "negatives prograde = retrograde", f"({dx:.6f}, {dy:.6f})")
dx, dy, mag = burn_direction_world(basis, 0.0, 100.0)
check(close(dx, -1.0) and close(dy, 0.0), "reines normal = einwaerts",
      f"({dx:.6f}, {dy:.6f})")
dx, dy, mag = burn_direction_world(basis, 0.0, -100.0)
check(close(dx, 1.0), "negatives normal = auswaerts", f"({dx:.6f}, {dy:.6f})")
dx, dy, mag = burn_direction_world(basis, 30.0, 40.0)
check(close(mag, 50.0, 1e-9), "betrag ist der pythagoras der beiden achsen",
      f"{mag:.6f}")
check(close(math.hypot(dx, dy), 1.0), "richtung ist ein einheitsvektor", "")
check(close(dx, -0.8, 1e-9) and close(dy, 0.6, 1e-9),
      "und mischt beide achsen richtig", f"({dx:.6f}, {dy:.6f})")
check(burn_direction_world(basis, 0.0, 0.0) == (0.0, 0.0, 0.0),
      "kein delta-v -> keine richtung", "")
check(burn_direction_world(None, 10.0, 0.0) == (0.0, 0.0, 0.0),
      "keine basis -> keine richtung", "")

print("\n5) der knoten")
n = ManeuverNode(1000.0)
check(close(n.dv_total, 0.0), "frischer knoten hat kein delta-v", "")
check(not n.is_executable(), "und ist damit NICHT ausfuehrbar", "")
n.dv_prograde = 3.0
n.dv_normal = 4.0
check(close(n.dv_total, 5.0, 1e-9), "delta-v ist der pythagoras", f"{n.dv_total}")
check(n.is_executable(), "mit delta-v ist er ausfuehrbar", "")
tiny = ManeuverNode(0.0, dv_prograde=MIN_EXECUTABLE_DV * 0.5)
check(not tiny.is_executable(), "unter der schwelle bleibt er gesperrt",
      f"{tiny.dv_total:g}")
prof = n.profile(10.0, 0.6)
check(close(prof.dv, 5.0, 1e-9), "profile() nimmt das delta-v des knotens", "")
check(prof.total_time > 0.0, "und liefert eine echte dauer", f"{prof.total_time:.4f}")

print("\n6) der plan haelt die knoten nach ZEIT sortiert")
plan = ManeuverPlan(max_nodes=5)
for t in (300.0, 100.0, 200.0):
    plan.add(ManeuverNode(t, dv_prograde=1.0))
check([node.t_node for node in plan] == [100.0, 200.0, 300.0],
      "einfuegen sortiert", f"{[node.t_node for node in plan]}")
check(plan.first().t_node == 100.0, "first() ist der zeitlich naechste", "")

print("\n7) hoechstens fuenf knoten")
plan = ManeuverPlan(max_nodes=5)
added = [plan.add(ManeuverNode(float(i))) for i in range(7)]
check(added == [True] * 5 + [False, False], "der sechste wird abgelehnt", f"{added}")
check(len(plan) == 5, "es bleiben fuenf", f"{len(plan)}")
check(plan.is_full, "is_full meldet das", "")

print("\n8) die version zaehlt JEDE aenderung")
plan = ManeuverPlan()
v0 = plan.version
plan.add(ManeuverNode(10.0))
check(plan.version > v0, "add zaehlt hoch", f"{v0} -> {plan.version}")
v1 = plan.version
plan.nodes[0].dv_prograde = 50.0
check(plan.version == v1, "eine direkte feldaenderung zaehlt NICHT von selbst", "")
plan.touch()
check(plan.version > v1, "touch() zaehlt sie nach", f"{v1} -> {plan.version}")
v2 = plan.version
plan.remove_at(0)
check(plan.version > v2, "remove zaehlt hoch", "")
v3 = plan.version
plan.add(ManeuverNode(1.0))
plan.clear()
check(plan.version > v3 and len(plan) == 0, "clear zaehlt hoch und leert", "")

print("\n9) touch() sortiert nach, wenn ein knoten verschoben wurde")
plan = ManeuverPlan()
a = ManeuverNode(100.0, dv_prograde=1.0)
b = ManeuverNode(200.0, dv_prograde=1.0)
plan.add(a)
plan.add(b)
b.t_node = 50.0
plan.touch()
check(plan.first() is b, "der verschobene knoten ist jetzt der erste",
      f"{[node.t_node for node in plan]}")

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for failure in FAILURES:
        print(f"  {failure}")
    sys.exit(1)
print("ship/maneuver/plan: alle pruefungen bestanden")
