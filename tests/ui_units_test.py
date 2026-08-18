"""Regressionstest fuer ui/units.py.

Reine funktionen -- laeuft ohne fenster, ohne GL, ohne pygame.
Aufruf: python tests/ui_units_test.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Die windows-konsole faehrt standardmaessig cp1252; '°' und aehnliche
# zeichen wuerden den test sonst an der AUSGABE scheitern lassen statt an
# einer echten abweichung.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from ui import units

FAILURES = []


def check(actual, expected, label):
    if actual != expected:
        FAILURES.append(f"{label}: {actual!r} != {expected!r}")
        print(f"  FAIL {label}: {actual!r} != {expected!r}")
    else:
        print(f"  ok   {label}: {actual!r}")


print("distance")
check(units.distance(950.0), '950m', 'unter 1 km')
check(units.distance(1500.0), '1.50km', 'km')
check(units.distance(6.371e6), '6.37Mm', 'erdradius')
check(units.distance(3.844e8), '384.40Mm', 'mondbahn')
check(units.distance(1.496e11), '149.60Gm', 'AE')
check(units.distance(None), '--', 'None')
check(units.distance(float('nan')), '--', 'NaN')
check(units.distance(float('inf')), '--', 'inf')

print("altitude")
check(units.altitude(-1500.0), '-1.50km', 'negativ')
check(units.altitude(400000.0), '400.00km', 'LEO')

print("speed")
check(units.speed(7660.0), '7.66km/s', 'orbitalgeschwindigkeit')
check(units.speed(12.5), '12.5m/s', 'langsam')

print("delta_v")
check(units.delta_v(3200.0), '3200.0m/s', 'unter 10k')
check(units.delta_v(15000.0), '15 000m/s', 'ueber 10k')

print("duration")
check(units.duration(0.0), '00:00:00', 'null')
check(units.duration(59.0), '00:00:59', 'sekunden')
check(units.duration(3661.0), '01:01:01', 'stunde')
check(units.duration(90 * 60.0), '01:30:00', 'LEO-umlauf')
check(units.duration(2 * 86400.0 + 3661.0), '2d 01:01:01', 'tage')
check(units.duration(365.25 * 86400.0 + 23 * 86400.0 + 3661.0),
      '1y 23d 01:01:01', 'jahre + tage')
check(units.duration(-3661.0), '-01:01:01', 'negativ')

print("countdown")
check(units.countdown(125.0), 'T-00:02:05', 'vor dem manoever')
check(units.countdown(-125.0), 'T+00:02:05', 'nach dem manoever')

print("mass")
check(units.mass(5.9722e24), '5.97e+24kg', 'erdmasse')
check(units.mass(1.989e30), '1.99e+30kg', 'sonne')
check(units.mass(2500.0), '2.50t', 'rakete')
check(units.mass(12.0), '12kg', 'klein')

print("angle")
check(units.angle(0.0), '0.0°', 'null')
check(units.angle(math.pi), '180.0°', 'pi')
check(units.angle(-math.pi / 2), '270.0°', 'normalisiert auf [0,360)')
check(units.signed_angle(-math.pi / 2), '-90.0°', 'signiert')
check(units.signed_angle(math.radians(350.0)), '-10.0°', 'signiert wickelt um')

print("eccentricity / scientific / time_warp")
check(units.eccentricity(0.0167), '0.017', 'erdbahn')
check(units.scientific(1.23456e-6), '1.23e-06', 'skala')
check(units.time_warp(1.0), '1x', 'echtzeit')
check(units.time_warp(1000.0), '1 000x', 'tausend')
check(units.time_warp(1e6), '1.0e+06x', 'million')

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for failure in FAILURES:
        print(f"  {failure}")
    sys.exit(1)
print("ui/units.py: alle pruefungen bestanden")
