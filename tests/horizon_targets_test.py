"""Kopftest fuer test.horizon_targets -- die reine laengen-regel des
vorhersage-horizonts. Modulebene, damit sie hier messbar ist statt
nachgebaut (dieselbe begruendung wie bei predictor_horizon_lengths).

Aufruf: python tests/horizon_targets_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from test import horizon_targets, predictor_horizon_lengths

FAILURES = []


def check(condition, label, detail=''):
    if condition:
        print(f"  ok   {label}{(' -- ' + detail) if detail else ''}")
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label} -- {detail}")


BASE = 1.0e10          # 10000 punkte x 1e6 m abstand
MAXP = 40000
SPACING = 1.0e6
CEIL = 4.0

# --- 1. ohne grabbing exakt wie predictor_horizon_lengths -------------
print("1. deckungsgleich ohne grabbing")
for mult in (0.25, 1.0, 2.0, 4.0):
    for warp in (1.0, 4.0, 64.0):
        a = horizon_targets(BASE, mult, warp, MAXP, SPACING)
        b = predictor_horizon_lengths(BASE, mult, warp, MAXP, SPACING)
        check(a == b, f"mult={mult} warp={warp}", f"{a} vs {b}")

# --- 2. beim grabbing ist `wanted` konstant, `drawn` folgt der auslenkung
print("2. grabbing pinnt die gerechnete laenge an die decke")
prev_wanted = None
for mult in (0.5, 1.0, 2.0, 3.7):
    drawn, wanted = horizon_targets(BASE, mult, 1.0, MAXP, SPACING,
                                    grabbing=True, ceiling_mult=CEIL)
    exp_drawn, _ = predictor_horizon_lengths(BASE, mult, 1.0, MAXP, SPACING)
    check(abs(drawn - exp_drawn) < 1.0, f"drawn folgt mult={mult}",
          f"{drawn} vs {exp_drawn}")
    if prev_wanted is not None:
        check(wanted == prev_wanted, f"wanted konstant bei mult={mult}",
              f"{wanted} vs {prev_wanted}")
    prev_wanted = wanted

_, ceil_wanted = predictor_horizon_lengths(BASE, CEIL, 1.0, MAXP, SPACING)
check(prev_wanted == ceil_wanted, "wanted == die decken-laenge",
      f"{prev_wanted} vs {ceil_wanted}")

# --- 3. grabbing ohne ceiling_mult faellt auf das alte verhalten -----
print("3. grabbing ohne decke = normalverhalten")
a = horizon_targets(BASE, 2.0, 1.0, MAXP, SPACING, grabbing=True,
                    ceiling_mult=None)
b = predictor_horizon_lengths(BASE, 2.0, 1.0, MAXP, SPACING)
check(a == b, "kein ceiling_mult -> unveraendert", f"{a} vs {b}")

# --- 4. mit realem Predictor: set_length genau einmal je griff -------
print("4. griff-kontrakt gegen einen echten Predictor")
import numpy as np
from predictor import Predictor
from vec import G


class _Body:
    def __init__(self, x, y, mass, radius):
        self.position = type('P', (), {'x': x, 'y': y})()
        self.velocity = type('V', (), {'x': 0.0, 'y': 0.0})()
        self.mass = mass
        self.radius = radius
        self.is_ship = False
        self.name = 'Erde'
        self.fixed = True
        self.is_moon_of = None


class _World:
    def __init__(self):
        self.G = G
        self.time = 0.0
        m = 5.9722e24
        self.erde = _Body(0.0, 0.0, m, 6.371e6)
        self.body = [self.erde]

    def update_planets(self, dt):
        pass


w = _World()
mu = G * w.erde.mass
r = 6.371e6 + 4.0e5
ship = _Body(r, 0.0, 0.0, 0.0)
ship.is_ship = True
ship.name = 'Schiff'
ship.fixed = False
import math as _m
ship.velocity = type('V', (), {'x': 0.0, 'y': _m.sqrt(mu / r) * 1.2})()

pred = Predictor(async_compute=False, num_points=2000, precision=1.0e6)
pred.initialize(ship, w)

calls = {'set_length': 0, 'set_display_length': 0}
_orig_sl, _orig_sdl = pred.set_length, pred.set_display_length
pred.set_length = lambda m, _o=_orig_sl, _c=calls: (_c.__setitem__('set_length', _c['set_length'] + 1), _o(m))[1]
pred.set_display_length = lambda m, _o=_orig_sdl, _c=calls: (_c.__setitem__('set_display_length', _c['set_display_length'] + 1), _o(m))[1]

BASE = pred.num_points * pred.precision
CEIL = 4.0


def apply(mult, grabbing):
    """Der schwanz von test.apply_predictor_horizon, nachgebaut."""
    drawn, wanted = horizon_targets(BASE, mult, 1.0, 40000, 1.0e6,
                                    grabbing=grabbing, ceiling_mult=CEIL)
    pred.set_display_length(drawn if wanted > drawn else None)
    pw = int(min(40000, max(1, _m.ceil(wanted / 1.0e6))))
    if pw != int(pred.num_points):
        pred.set_num_points(pw, soft=True)
    cur = pred.length
    if cur is not None and abs(cur - wanted) <= wanted * 1e-9:
        return
    pred.set_length(wanted)


apply(1.0, grabbing=False)                       # ausgangslage
calls['set_length'] = 0
calls['set_display_length'] = 0

# 30 frames "ziehen": mult waechst, griff gehalten
for i in range(30):
    apply(1.0 + i * 0.1, grabbing=True)
check(calls['set_length'] <= 1, "set_length hoechstens 1x waehrend des griffs",
      f"{calls['set_length']}x")
check(calls['set_display_length'] == 30, "set_display_length jeden frame",
      f"{calls['set_display_length']}x")

# loslassen: ein weiterer set_length auf den settled-wert
before = calls['set_length']
apply(3.9, grabbing=False)
check(calls['set_length'] == before + 1, "beim loslassen genau ein set_length",
      f"{calls['set_length']} (war {before})")

pred.close()

# --- 5. view-identitaet bleibt stehen innerhalb eines quantums --------
# Der regler ruft set_display_length() JEDEN frame waehrend des ziehens.
# get_points() gibt self.points[:count] heraus; aendert sich count je frame,
# entsteht je frame eine neue view -- der renderer-cache und
# get_apsis_markers() (beide auf id(points)) verfehlen dann jeden frame.
# _display_quantum rundet count auf ein vielfaches, so bleibt die view-
# identitaet ueber die meisten zug-frames stehen.
print("5. get_points()-identitaet")
pred2 = Predictor(async_compute=False, rolling_mode=False,
                  num_points=2000, precision=1.0e6)
pred2.initialize(ship, w)
pred2.set_length(2000 * 1.0e6)
pred2.update(ship, w)

pred2.set_display_length(1.0e9)
a = pred2.get_points()
pred2.set_display_length(1.0e9 + 1.0e6)      # < 8 punkte weiter
b = pred2.get_points()
check(a is b, "identische view innerhalb eines quantums", f"{id(a)} vs {id(b)}")
pred2.set_display_length(1.0e9 + 5.0e7)      # deutlich weiter
c = pred2.get_points()
check(c is not b, "quantum-grenze ueberschritten -> neue view", "")
pred2.close()

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for f in FAILURES:
        print(f"  {f}")
    sys.exit(1)
print("test.horizon_targets: alle pruefungen bestanden")
