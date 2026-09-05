"""Kopftest fuer ship.horizon.horizon_targets -- die reine laengen-regel des
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

from ship.horizon import (horizon_compute_rung, horizon_targets,
                          predictor_horizon_lengths)

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
GRAB_STEP = 4.0

# --- 1. ohne grabbing exakt wie predictor_horizon_lengths -------------
print("1. deckungsgleich ohne grabbing")
for mult in (0.25, 1.0, 2.0, 4.0, 64.0, 256.0):
    for warp in (1.0, 4.0, 64.0):
        a = horizon_targets(BASE, mult, warp, MAXP, SPACING)
        b = predictor_horizon_lengths(BASE, mult, warp, MAXP, SPACING)
        check(a == b, f"mult={mult} warp={warp}", f"{a} vs {b}")

# --- 2. die raste: sprossen liegen fest an BASE -----------------------
print("2. horizon_compute_rung")
check(horizon_compute_rung(BASE, BASE, GRAB_STEP) == BASE,
      "genau auf einer sprosse -> die sprosse selbst",
      f"{horizon_compute_rung(BASE, BASE, GRAB_STEP)}")
check(horizon_compute_rung(BASE, BASE * 1.01, GRAB_STEP) == BASE * 4.0,
      "knapp darueber -> naechste sprosse", "")
check(horizon_compute_rung(BASE, BASE * 4.0, GRAB_STEP) == BASE * 4.0,
      "exakt die naechste sprosse -> keine weitere", "")
check(horizon_compute_rung(BASE, BASE * 5.0, GRAB_STEP) == BASE * 16.0,
      "darueber -> uebernaechste", "")
# Die sprosse DECKT `wanted` immer -- 0.3x liegt zwischen 0.25x und 1x, also
# ist 1x die richtige. Nie darunter: eine zu kurze kurve waere sichtbar, eine
# zu lange schneidet der zeichen-clip weg.
check(horizon_compute_rung(BASE, BASE * 0.3, GRAB_STEP) == BASE,
      "unter BASE -> die naechste sprosse darueber",
      f"{horizon_compute_rung(BASE, BASE * 0.3, GRAB_STEP)}")
check(horizon_compute_rung(BASE, BASE * 0.2, GRAB_STEP) == BASE * 0.25,
      "weiter unten -> sprosse nach unten", "")
# Ueber die ganze spanne 0.25x..256x sind es hoechstens 6 verschiedene werte.
rungs = {horizon_compute_rung(BASE, BASE * (0.25 * (1024.0 ** (i / 400.0))),
                              GRAB_STEP) for i in range(401)}
check(len(rungs) <= 6, "hoechstens 6 sprossen ueber 0.25x..256x",
      f"{len(rungs)} sprossen")

# --- 3. beim grabbing waechst `wanted` nur, `drawn` folgt dem knauf ---
print("3. grabbing rastet nach oben und schrumpft nie")
cur = BASE
seen = []
for i in range(200):                       # zug nach aussen, 1x -> 256x
    mult = 1.0 * (256.0 ** (i / 199.0))
    drawn, cur = horizon_targets(BASE, mult, 1.0, MAXP, SPACING,
                                 grabbing=True, current_length=cur,
                                 grab_step_factor=GRAB_STEP)
    exp_drawn, _ = predictor_horizon_lengths(BASE, mult, 1.0, MAXP, SPACING)
    if abs(drawn - exp_drawn) > 1.0:
        check(False, f"drawn folgt mult={mult:g}", f"{drawn} vs {exp_drawn}")
        break
    if not seen or seen[-1] != cur:
        seen.append(cur)
else:
    check(True, "drawn folgt dem knauf ueber den ganzen zug", "200 frames")
check(all(b > a for a, b in zip(seen, seen[1:])), "wanted waechst monoton",
      f"{[f'{v:.3g}' for v in seen]}")
check(len(seen) <= 6, "hoechstens 6 wechsel = 6 set_length ueber den zug",
      f"{len(seen)} wechsel")
check(cur >= BASE * 256.0, "am ende deckt die raste den knauf",
      f"{cur:.3g} vs {BASE * 256.0:.3g}")

# zug nach innen: kein einziger wechsel
inward = cur
changes = 0
for i in range(200):
    mult = 256.0 / (256.0 ** (i / 199.0))
    _, new = horizon_targets(BASE, mult, 1.0, MAXP, SPACING, grabbing=True,
                             current_length=inward, grab_step_factor=GRAB_STEP)
    if new != inward:
        changes += 1
    inward = new
check(changes == 0, "zug nach innen aendert die gerechnete laenge nie",
      f"{changes} wechsel")

# --- 3b. grabbing ohne current_length faellt auf das normalverhalten --
print("3b. grabbing ohne current_length = normalverhalten")
a = horizon_targets(BASE, 2.0, 1.0, MAXP, SPACING, grabbing=True,
                    current_length=None)
b = predictor_horizon_lengths(BASE, 2.0, 1.0, MAXP, SPACING)
check(a == b, "kein current_length -> unveraendert", f"{a} vs {b}")

# --- 4. mit realem Predictor: set_length genau einmal je griff -------
print("4. griff-kontrakt gegen einen echten Predictor")
import numpy as np
from ship.predictor import Predictor
from physics.vec import G


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
GRAB_STEP = 4.0


def apply(mult, grabbing):
    """Der schwanz von test.apply_predictor_horizon, nachgebaut."""
    drawn, wanted = horizon_targets(BASE, mult, 1.0, 40000, 1.0e6,
                                    grabbing=grabbing,
                                    current_length=pred.length,
                                    grab_step_factor=GRAB_STEP)
    pred.set_display_length(drawn)
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

# 30 frames "ziehen": mult waechst 1x -> 4x, griff gehalten
for i in range(30):
    apply(1.0 + i * 0.1, grabbing=True)
check(calls['set_length'] <= 1, "set_length hoechstens 1x waehrend des griffs",
      f"{calls['set_length']}x")
check(calls['set_display_length'] == 30, "set_display_length jeden frame",
      f"{calls['set_display_length']}x")

# DER FEHLER: beim loslassen darf der clip nicht ausgehen. `wanted` faellt auf
# `drawn` zurueck, waehrend die lange kurve noch im speicher liegt -- mit dem
# alten `drawn if wanted > drawn else None` wurde sie fuer die paar frames bis
# zum eintreffen des kurzen auftrags ungeschnitten gezeichnet.
before = calls['set_length']
apply(3.9, grabbing=False)
check(calls['set_length'] == before + 1, "beim loslassen genau ein set_length",
      f"{calls['set_length']} (war {before})")
drawn_after, _ = predictor_horizon_lengths(BASE, 3.9, 1.0, 40000, 1.0e6)
check(pred.display_length is not None
      and abs(pred.display_length - drawn_after) < 1.0,
      "der clip bleibt nach dem loslassen auf dem knauf stehen",
      f"{pred.display_length} vs {drawn_after}")

# Gegenprobe fuer die kosten der raste: ein voller zug bis 256x kostet nur
# eine handvoll set_length, nicht eines je frame.
apply(1.0, grabbing=False)
calls['set_length'] = 0
for i in range(200):
    apply(1.0 * (256.0 ** (i / 199.0)), grabbing=True)
check(calls['set_length'] <= 6, "voller zug bis 256x: hoechstens 6 set_length",
      f"{calls['set_length']}x")

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
print("ship.horizon.horizon_targets: alle pruefungen bestanden")
