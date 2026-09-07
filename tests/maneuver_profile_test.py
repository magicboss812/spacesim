"""Das schubprofil eines manoeverknotens.

Drei ebenen:

1. **Die geschlossene form gegen von hand gerechnete werte.** Beide zweige
   (trapez und dreieck) mit sollwerten, die im test selbst hergeleitet sind.
2. **Die symmetrie**, die den zuendzeitpunkt traegt: bei total_time/2 ist
   genau die haelfte des delta-v geliefert. Ohne sie brennt der knoten
   einseitig und die vorschau zeigt eine andere bahn als geflogen wird.
3. **Der njit-zwilling gegen die Python-fassung.** _profile_accel_numba muss
   bitgleich zu BurnProfile.accel_at sein -- der kernel kann die methode
   nicht aufrufen, also gibt es sie zweimal, und genau deshalb wird die
   gleichheit geprueft.

Aufruf: python tests/maneuver_profile_test.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from physics.kernels.burn import _profile_accel_numba
from ship.maneuver.profile import BurnProfile

FAILURES = []


def check(condition, name, detail=''):
    status = 'OK  ' if condition else 'FEHL'
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ''))
    if not condition:
        FAILURES.append(name)


def close(a, b, rel=1e-9, abs_=1e-12):
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


print("\n1) trapez-zweig -- delta-v gross genug, dass a_max erreicht wird")
# a_max 600, ramp 0.5 -> rate 1200 m/s^3. Die spitze wird ab
# dv >= a_max*ramp = 300 m/s erreicht.
p = BurnProfile(1200.0, 600.0, 0.5)
check(close(p.ramp_rate, 1200.0), "rampenrate = a_max/ramp", f"{p.ramp_rate}")
check(close(p.a_peak, 600.0), "spitze = a_max", f"{p.a_peak}")
check(close(p.ramp_time, 0.5), "rampenzeit = ramp_seconds", f"{p.ramp_time}")
check(close(p.hold_time, 1.5), "haltezeit", f"{p.hold_time}")
check(close(p.total_time, 2.5), "gesamtdauer = ramp + dv/a_max", f"{p.total_time}")

print("\n2) dreieck-zweig -- delta-v zu klein fuer die spitze")
q = BurnProfile(100.0, 600.0, 0.5)
check(close(q.a_peak, math.sqrt(100.0 * 1200.0)), "spitze = sqrt(dv*rate)",
      f"{q.a_peak:.6f}")
check(q.a_peak < 600.0, "spitze bleibt unter a_max", f"{q.a_peak:.3f}")
check(close(q.hold_time, 0.0), "keine haltephase", f"{q.hold_time}")
check(close(q.total_time, 2.0 * math.sqrt(100.0 / 1200.0)),
      "gesamtdauer = 2*sqrt(dv/rate)", f"{q.total_time:.6f}")

print("\n3) die beiden zweige treffen sich stetig")
lo = BurnProfile(299.999, 600.0, 0.5)
hi = BurnProfile(300.001, 600.0, 0.5)
check(abs(lo.total_time - hi.total_time) < 1e-5,
      "kein sprung in der dauer an der nahtstelle",
      f"{lo.total_time:.8f} vs {hi.total_time:.8f}")
check(abs(lo.a_peak - hi.a_peak) < 1e-2,
      "kein sprung in der spitze an der nahtstelle",
      f"{lo.a_peak:.6f} vs {hi.a_peak:.6f}")

print("\n4) das profil liefert GENAU das geplante delta-v")
for profile in (p, q, BurnProfile(0.75, 600.0, 0.5), BurnProfile(9000.0, 600.0, 0.5)):
    check(close(profile.dv_delivered(profile.total_time), profile.dv),
          f"dv_delivered(total) == dv  (dv={profile.dv})",
          f"{profile.dv_delivered(profile.total_time):.9f}")
    check(close(profile.dv_delivered(profile.total_time * 10.0), profile.dv),
          f"ueber das ende hinaus bleibt es stehen  (dv={profile.dv})", "")

print("\n5) SYMMETRIE -- bei total/2 ist die haelfte geliefert")
for profile in (p, q, BurnProfile(3.0, 600.0, 0.5), BurnProfile(4321.0, 600.0, 0.5)):
    half = profile.dv_delivered(profile.total_time * 0.5)
    check(close(half, profile.dv * 0.5, rel=1e-9),
          f"halbes delta-v bei halber zeit  (dv={profile.dv})",
          f"{half:.9f} vs {profile.dv * 0.5:.9f}")
    check(close(profile.lead_time, profile.total_time * 0.5),
          f"lead_time == total/2  (dv={profile.dv})", "")
    check(close(profile.ignition_time(1000.0), 1000.0 - profile.total_time * 0.5),
          f"ignition_time zieht lead_time ab  (dv={profile.dv})", "")

print("\n6) dv_between ist bildratenunabhaengig")
for steps in (7, 61, 1000):
    total = 0.0
    dt = p.total_time / steps
    for i in range(steps):
        total += p.dv_between(i * dt, (i + 1) * dt)
    check(close(total, p.dv, rel=1e-9),
          f"summe ueber {steps} schritte == dv", f"{total:.9f}")

print("\n7) randfaelle")
zero = BurnProfile(0.0, 600.0, 0.5)
check(close(zero.total_time, 0.0), "delta-v null -> dauer null", "")
check(close(zero.accel_at(0.0), 0.0) and close(zero.accel_at(5.0), 0.0),
      "delta-v null -> nie schub", "")
check(close(zero.dv_between(0.0, 10.0), 0.0), "delta-v null -> nie delta-v", "")
check(close(p.accel_at(-1.0), 0.0), "vor der zuendung kein schub", "")
check(close(p.accel_at(p.total_time + 1.0), 0.0), "nach dem ende kein schub", "")
peak = max(p.accel_at(p.total_time * i / 500.0) for i in range(501))
check(peak <= p.a_max + 1e-9, "schub ueberschreitet a_max nie", f"{peak:.6f}")
check(close(p.throttle_at(p.total_time * 0.5), 1.0),
      "voller hebel in der haltephase", f"{p.throttle_at(p.total_time*0.5):.6f}")

print("\n8) der njit-zwilling ist bitgleich zur Python-fassung")
# Der kernel kann BurnProfile.accel_at nicht aufrufen (numba nimmt keine
# Python-objekte), also steht dieselbe formel zweimal da. Genau deshalb
# wird sie hier verglichen -- ohne diese pruefung koennten vorschau und
# ausfuehrung unbemerkt auseinanderlaufen.
worst = 0.0
for profile in (p, q, BurnProfile(3.0, 600.0, 0.5), BurnProfile(9000.0, 600.0, 0.5)):
    span = profile.total_time * 1.2 + 0.1
    for i in range(501):
        tau = -0.05 + span * i / 500.0
        a_py = profile.accel_at(tau)
        a_nb = _profile_accel_numba(
            tau, profile.a_peak, profile.ramp_time, profile.hold_time,
            profile.total_time, profile.ramp_rate,
        )
        worst = max(worst, abs(a_py - a_nb))
check(worst == 0.0, "njit == Python an 2004 stuetzstellen, exakt",
      f"groesste abweichung {worst:g}")

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for failure in FAILURES:
        print(f"  {failure}")
    sys.exit(1)
print("ship/maneuver/profile: alle pruefungen bestanden")
