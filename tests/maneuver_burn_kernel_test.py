"""Der brennbogen-kernel: schwerkraft plus konstant gerichteter schub.

Die eine sache, die dieser kernel exakt treffen MUSS, ist das gelieferte
delta-v: die vorschau zeigt die bahn danach, und wenn der bogen 3 % zuviel
liefert, zeigt sie die falsche. Geprueft wird gegen zwei unabhaengige
referenzen:

1. **Ohne schwerkraft** (ein masseloser koerper): der bogen ist dann reine
   kinematik und das ergebnis steht analytisch fest -- die
   geschwindigkeitsaenderung ist das geplante delta-v, exakt in
   schubrichtung.
2. **Mit schwerkraft, aber ohne schub**: der bogen muss dieselbe bahn
   liefern wie der vorhandene RK4-kern
   `_compute_distance_points_numba_state`. Ohne diese gegenprobe bestuende
   punkt 1 auch dann, wenn die schwerkraft schlicht fehlte.

Aufruf: python tests/maneuver_burn_kernel_test.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np

from physics.kernels.burn import _burn_arc_numba
from physics.kernels.propagate import _compute_distance_points_numba_state
from ship.maneuver.profile import BurnProfile

FAILURES = []


def check(condition, name, detail=''):
    status = 'OK  ' if condition else 'FEHL'
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ''))
    if not condition:
        FAILURES.append(name)


G = 6.6730831e-11
EARTH_MASS = 5.97219e24
EARTH_MU = G * EARTH_MASS

NO_MEMO = np.zeros((0, 10), dtype=np.float64)


def bodies(mass):
    """Ein einzelner, ruhender koerper im ursprung."""
    return (
        np.array([0.0], dtype=np.float64),      # x
        np.array([0.0], dtype=np.float64),      # y
        np.array([mass], dtype=np.float64),     # m
        np.array([1], dtype=np.int64),          # fixed
        np.array([0], dtype=np.int64),          # scripted
        np.zeros(1, dtype=np.float64),          # a
        np.zeros(1, dtype=np.float64),          # e
        np.zeros(1, dtype=np.float64),          # theta
        np.zeros(1, dtype=np.float64),          # arg
        np.array([-1], dtype=np.int64),         # parent
    )


def run_burn(profile, dir_x, dir_y, px, py, vx, vy, mass, steps=400):
    bx, by, bm, bfix, bscr, ba, be, bth, barg, bpar = bodies(mass)
    return _burn_arc_numba(
        px, py, vx, vy, 0.0,
        dir_x, dir_y,
        profile.a_peak, profile.ramp_time, profile.hold_time,
        profile.total_time, profile.ramp_rate,
        -1, 0.0, 0.0,
        bx, by, bm, bfix, bscr, ba, be, bth, barg, bpar,
        G, 0, NO_MEMO,
        steps,
    )


print("\n1) ohne schwerkraft ist der bogen reine kinematik")
prof = BurnProfile(120.0, 10.0, 0.6)
dir_x, dir_y = 0.6, 0.8      # einheitsvektor (3-4-5)
out, count = run_burn(prof, dir_x, dir_y, 1.0e7, 0.0, 0.0, 7.5e3, 0.0)
check(count >= 2, "der bogen liefert punkte", f"{count}")
dvx = float(out[count - 1, 3]) - 0.0
dvy = float(out[count - 1, 4]) - 7.5e3
delivered = math.hypot(dvx, dvy)
check(abs(delivered - prof.dv) <= 1e-6 * prof.dv,
      "geliefertes delta-v == geplantes", f"{delivered:.9f} vs {prof.dv}")
check(abs(dvx / delivered - dir_x) < 1e-9 and abs(dvy / delivered - dir_y) < 1e-9,
      "und liegt exakt in schubrichtung",
      f"({dvx/delivered:.9f}, {dvy/delivered:.9f})")
check(abs(float(out[count - 1, 2]) - prof.total_time) < 1e-9,
      "die letzte stuetzstelle liegt am brennende",
      f"{float(out[count-1, 2]):.9f} vs {prof.total_time:.9f}")

print("\n2) der schub baut sich WEICH auf, er springt nicht")
# Gemessen wird der zuwachs LAENGS DER SCHUBRICHTUNG, nicht |v|: der schub
# steht hier schraeg zur bahngeschwindigkeit, und |v| waechst deshalb
# nichtlinear -- an |v| gemessen sieht sogar ein sauberes profil nach einem
# ueberschwinger aus. Die projektion ist das, was der schub wirklich tut.
along = [float(out[i, 3]) * dir_x + float(out[i, 4]) * dir_y
         for i in range(count)]
steps_dv = [along[i + 1] - along[i] for i in range(count - 1)]
mid = steps_dv[len(steps_dv) // 2]
check(steps_dv[0] < mid * 0.2, "erster schritt liefert < 20 % eines mittleren",
      f"{steps_dv[0]:.6f} vs {mid:.6f}")
check(steps_dv[-1] < mid * 0.2, "letzter schritt ebenso", f"{steps_dv[-1]:.6f}")
check(max(steps_dv) <= mid * 1.0001,
      "die haltephase ist das maximum, kein ueberschwinger",
      f"{max(steps_dv):.9f} vs {mid:.9f}")
check(sum(steps_dv) > 0.0 and min(steps_dv) >= -1e-12,
      "und kein schritt liefert negatives delta-v", f"{min(steps_dv):.3e}")

print("\n3) HALBES delta-v bei halber brenndauer")
# Die stuetzstelle wird ueber die ZEIT gesucht, nicht ueber den index: die
# drei phasen bekommen unterschiedlich viele schritte (siehe modulkopf von
# physics/kernels/burn.py), der mittlere index liegt also nicht auf der
# halben zeit.
half_t = prof.total_time * 0.5
half_index = min(range(count), key=lambda i: abs(float(out[i, 2]) - half_t))
v_mid = math.hypot(float(out[half_index, 3]) - 0.0,
                   float(out[half_index, 4]) - 7.5e3)
check(abs(float(out[half_index, 2]) - half_t) < prof.total_time / count * 1.01,
      "eine stuetzstelle liegt auf der halben brenndauer",
      f"t={float(out[half_index, 2]):.6f} vs {half_t:.6f}")
check(abs(v_mid - prof.dv * 0.5) <= 2e-3 * prof.dv,
      "und dort ist die haelfte geliefert",
      f"{v_mid:.6f} vs {prof.dv * 0.5:.6f}")

print("\n4) GEGENPROBE -- ohne schub gegen die ANALYTISCHE kreisbahn")
# Ohne diesen abschnitt bestuende abschnitt 1 auch dann, wenn die
# schwerkraft im kernel gar nicht ankaeme. Die referenz ist hier kein
# zweiter integrator, sondern die geschlossene loesung: eine kreisbahn um
# einen ruhenden massenpunkt. Startpunkt (r, 0), geschwindigkeit +y, also
# nach t = 600 s der winkel omega*t mit omega = v_circ/r.
R0 = 1.0e7
bx, by, bm, bfix, bscr, ba, be, bth, barg, bpar = bodies(EARTH_MASS)
v_circ = math.sqrt(EARTH_MU / R0)
COAST_S = 600.0
out_c, count_c = _burn_arc_numba(
    R0, 0.0, 0.0, v_circ, 0.0,
    1.0, 0.0,
    0.0, 0.0, 0.0, COAST_S, 0.0,          # a_peak 0 -> kein schub
    -1, 0.0, 0.0,
    bx, by, bm, bfix, bscr, ba, be, bth, barg, bpar,
    G, 0, NO_MEMO,
    2000,
)
end_x, end_y = float(out_c[count_c - 1, 0]), float(out_c[count_c - 1, 1])
omega = v_circ / R0
exact_x = R0 * math.cos(omega * COAST_S)
exact_y = R0 * math.sin(omega * COAST_S)
err = math.hypot(end_x - exact_x, end_y - exact_y)
check(err < 1.0e-8 * R0,
      "gleitbogen == analytische kreisbahn nach 600 s",
      f"{err:.6f} m auf r={R0:.3e} m")
check(abs(float(out_c[count_c - 1, 2]) - COAST_S) < 1e-9,
      "und die endzeit stimmt", f"{float(out_c[count_c-1, 2]):.9f}")

print("\n5) und derselbe gleitbogen deckt sich mit dem vorhandenen RK4-kern")
# Zweite, unabhaengige gegenprobe: `precision` ist ein BOGENABSTAND, kein
# zeitschritt -- bei 1 m je punkt decken 3000 punkte nur 3 km ab, also
# 0.5 s. Der abstand muss also aus der wirklich geflogenen strecke kommen.
span_m = v_circ * COAST_S
ref_out, ref_used = _compute_distance_points_numba_state(
    R0, 0.0, 0.0, v_circ, 0.0,
    0, 0.0, 0.0, bx, by, bm, bfix, G,
    COAST_S / 2000.0,
    span_m / 1500.0,
    3000, 200000,
)
best = None
for i in range(ref_used):
    dt = abs(float(ref_out[i, 2]) - COAST_S)
    if best is None or dt < best[0]:
        best = (dt, float(ref_out[i, 0]), float(ref_out[i, 1]))
check(best is not None and best[0] < 1.0, "die referenz reicht bis 600 s",
      f"naechster punkt {best[0]:.4f} s daneben" if best else "leer")
if best is not None and best[0] < 1.0:
    # Der naechstliegende referenzpunkt ist bis zu best[0] sekunden
    # daneben; auf dieser bahn sind das v_circ*dt meter. Genau darauf wird
    # die schranke gelegt, statt eine willkuerliche zahl zu waehlen.
    err = math.hypot(end_x - best[1], end_y - best[2])
    check(err < v_circ * best[0] + 1.0e-6 * R0,
          "gleitbogen == RK4-referenz (innerhalb ihres eigenen punktabstands)",
          f"{err:.3f} m, erlaubt {v_circ * best[0] + 1.0e-6 * R0:.3f} m")

print("\n6) der bogen bleibt auf einer plausiblen bahn")
r_end = math.hypot(end_x, end_y)
check(abs(r_end - R0) < 1.0e-6 * R0,
      "kreisbahn bleibt kreisfoermig ueber 600 s",
      f"r {R0:.3e} -> {r_end:.3e}")

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for failure in FAILURES:
        print(f"  {failure}")
    sys.exit(1)
print("physics/kernels/burn: alle pruefungen bestanden")
