"""Regressionen fuer den stillstand der Ap/Pe-marker.

Alle pruefungen messen eine GROESSE, keine implementierung.

Der fehler, um den es geht: zwischen zwei hintergrund-auffrischungen wurde die
gespeicherte kurve STARR so verschoben, dass ihr kopf auf dem schiff sitzt.
Der referenzkoerper wandert dabei nicht mit -- uebrig bleibt die RELATIV-
bewegung schiff<->koerper, und die legt die ganze kegelschnittbahn seitlich
neben den koerper. Genau das ist die periapsis-hoehe. Weil der verschiebe-
betrag am ALTER des schnappschusses haengt und das alter mit der rechen-
latenz schwankt, sprang der angezeigte Pe/Ap-abstand von frame zu frame
(gemeldet: 500 gegen 510 km an einem Mond-vorbeiflug). Im zeitraffer trat es
nie auf, weil dort der halt die kurve VERBRAUCHT statt sie zu verschieben.

1. **Der marker steht still, solange die bahn stillsteht.** Ohne schub ist
   die vorhersage eine eigenschaft der BAHN, nicht des augenblicks -- das
   schiff rutscht an ihr entlang, sie selbst bewegt sich nicht. Gemessen
   ueber viele frames ohne auffrischung; verglichen wird gegen die strecke,
   die das schiff in derselben zeit zuruecklegt (das ist der betrag, um den
   die alte starre verschiebung danebenlag).
2. **Die kurve waechst und schrumpft dabei nicht.** Der selbst voran-
   gestellte kopf muss vor dem naechsten schnitt wieder weg, sonst waechst
   die liste je frame um einen punkt.
3. **Eine verschobene zeitspalte verfaelscht den koerper nicht.** Wer die
   punktzeiten doch verschiebt (der fallback), muss den versatz mitfuehren
   -- sonst propagiert der apsis-scan den referenzkoerper um genau diesen
   betrag zu weit und meldet einen anderen abstand fuer dieselbe geometrie.
4. **Der zeitraffer-halt bleibt, was er war.** Er benutzt jetzt dieselbe
   mechanik; sein stillstand darf sich davon nicht verschlechtern.

Aufruf: python tests/apsis_stability_test.py
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

from vec import G, Vec2
from loader import ConfigLoader, SystemLoader
from predictor import Predictor
from world import world as World

FAILURES = []


def check(condition, label, detail=''):
    if condition:
        print(f"  ok   {label}{(' -- ' + detail) if detail else ''}")
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label} -- {detail}")


MAX_SUBSTEP = 1000.0

# Die unterste zeitraffer-stufe des spiels: `REALTIME_WARP_MAX` = 60 sim-s je
# echtsekunde, bei 180 bildern also 1/3 s je bild. Das ist die stufe, auf der
# der fehler gemeldet wurde -- der halt ist hier noch AUS.
REALTIME_RATE = 60.0
FPS = 180.0
FRAME_DT = REALTIME_RATE / FPS


def advance(w, sim_seconds):
    """Ein weltschritt -- in der reihenfolge des spiels (dynamics, planets)."""
    steps = max(1, int(math.ceil(sim_seconds / MAX_SUBSTEP)))
    dt = sim_seconds / steps
    for _ in range(steps):
        w.update_dynamics(dt)
        w.update_planets(dt)


def build():
    config = ConfigLoader(None)
    config.load()
    bodies = SystemLoader("solar_system.json").load()
    w = World(G)
    w.body = bodies
    config.apply_to_world(w)
    ship = next(b for b in bodies if b.is_ship)
    p = Predictor(recompute_every_update=True, **config.predictor_kwargs())
    config.apply_to_predictor(p)
    p.set_length(p.num_points * p.precision)
    p.async_compute = False
    return w, ship, p


def body_velocity(b, t, h=1.0):
    """Bahngeschwindigkeit eines GESKRIPTETEN koerpers per zentralem differenzen-
    quotienten -- `body.velocity` wird fuer sie nicht fortgeschrieben."""
    a = b.position_at_time(t - h)
    c = b.position_at_time(t + h)
    return Vec2((c.x - a.x) / (2.0 * h), (c.y - a.y) / (2.0 * h))


def place_lunar_flyby(w, ship, alt_pe=500e3, ecc=0.6):
    """Schiff auf eine Mond-umlaufbahn mit periapsis `alt_pe` setzen, START IM
    APOAPSIS -- die periapsis liegt damit voraus, mitten in der vorhersage.

    Rueckgabe: (index des Mondes, mu, erwarteter periapsis-radius).
    """
    idx = next(i for i, b in enumerate(w.body) if b.name == 'Mond')
    mond = w.body[idx]
    mu = G * mond.mass
    rp = float(mond.radius) + float(alt_pe)
    a = rp / (1.0 - ecc)
    ra = a * (1.0 + ecc)
    v_ap = math.sqrt(mu * (2.0 / ra - 1.0 / a))

    mpos = mond.position_at_time(w.time)
    mvel = body_velocity(mond, w.time)
    ship.position = Vec2(mpos.x + ra, mpos.y)
    ship.velocity = Vec2(mvel.x, mvel.y + v_ap)
    ship.acceleration = Vec2(0.0, 0.0)
    return idx, mu, rp


def periapsis_r(p):
    """Kleinster gemeldeter periapsis-abstand, oder None."""
    m = p.get_apsis_markers()
    if m is None or len(m) == 0:
        return None
    pe = m[m[:, 3] == 0.0]
    if pe.shape[0] == 0:
        return None
    return float(pe[0, 4])


# ═══════════════ 1. der marker steht still, solange die bahn stillsteht

print("1. Pe-marker steht still zwischen zwei auffrischungen (echtzeit)")

w, ship, p = build()
ref_index, mu, rp_expected = place_lunar_flyby(w, ship)
p.set_reference_body_index(ref_index)
p.set_hold(False)
p.initialize(ship, w)

r0 = periapsis_r(p)
check(r0 is not None, "die vorhersage findet ueberhaupt eine periapsis",
      f"r={r0:.6e} m (erwartet ~{rp_expected:.6e} m)" if r0 else "keine")

if r0 is not None:
    # Grob plausibel? Der scan misst gegen den GESKRIPTETEN Mond, die welt
    # integriert das schiff -- ein paar prozent abweichung sind normal.
    rel = abs(r0 - rp_expected) / rp_expected
    check(rel < 0.10, "und sie liegt auf der gesetzten periapsis",
          f"{r0:.6e} m gegen {rp_expected:.6e} m ({rel * 100:.2f} %)")

    start = Vec2(ship.position.x, ship.position.y)
    n0 = len(p.get_points())
    radii = [r0]
    counts = [n0]
    # KEINE auffrischung in dieser schleife: genau der zustand zwischen zwei
    # hintergrund-ergebnissen, in dem die alte fassung starr verschoben hat.
    for _ in range(120):
        advance(w, FRAME_DT)
        p._anchor_first_point(ship, w)
        r = periapsis_r(p)
        if r is not None:
            radii.append(r)
        counts.append(len(p.get_points()))

    flown = math.hypot(ship.position.x - start.x, ship.position.y - start.y)
    spread = max(radii) - min(radii)
    # DIE SCHRANKE IST NICHT NULL, UND DAS HAT EINEN BENANNTEN GRUND.
    #
    # Die verbliebenen stuetzstellen stehen bit-genau still, der scan selbst
    # aber nicht ganz: er loest den teuren Kepler-solve fuer den referenz-
    # koerper nur an jedem n-ten punkt und interpoliert dazwischen linear
    # ueber die zeit (siehe `stride_max` / `time_window` in
    # _find_apsis_markers_numba). Diese knoten haengen am INDEX, und der
    # verschiebt sich um die vorn verbrauchten punkte -- die interpolierte
    # Mond-position wackelt damit um den dort veranschlagten betrag
    # ("erde/mond: zehner meter"). Gemessen 1.0e+01 m, also genau in dieser
    # groessenordnung und rund fuenf zehnerpotenzen unter dem fehler, um den
    # es hier geht (gemeldet 500 gegen 510 km).
    check(spread < 100.0, "der gemeldete Pe-abstand bewegt sich nicht",
          f"streuung {spread:.3e} m ueber {len(radii)} frames "
          f"(schiff flog {flown:.3e} m)")
    # Die aussage, auf die es ankommt: die streuung haengt NICHT mehr an der
    # geflogenen strecke. Frueher war sie von derselben groessenordnung.
    check(spread < flown * 1e-3,
          "und zwar unabhaengig davon, wie weit das schiff inzwischen flog",
          f"{spread:.3e} m gegen {flown:.3e} m geflogen")

    # ═══════════ 2. die kurve waechst und schrumpft dabei nicht
    print()
    print("2. Der vorangestellte kopf haeuft sich nicht an")
    check(max(counts) - min(counts) <= 2,
          "die punktzahl bleibt stehen, waehrend das schiff die kurve entlang rutscht",
          f"min {min(counts)} max {max(counts)} ueber {len(counts)} frames")

# ═══════════ 3. verschobene zeitspalte verfaelscht den koerper nicht

print()
print("3. Eine verschobene zeitspalte liest den referenzkoerper nicht zu weit vorn")

w, ship, p = build()
ref_index, mu, rp_expected = place_lunar_flyby(w, ship)
p.set_reference_body_index(ref_index)
p.set_hold(False)
p.initialize(ship, w)

r_before = periapsis_r(p)
snap = p._last_swapped_snapshot
check(bool(snap.get("use_time_dependent_bodies", False)),
      "der scan propagiert den Mond ueberhaupt (sonst misst diese pruefung nichts)",
      f"use_time_dependent_bodies={snap.get('use_time_dependent_bodies')}")

# REIN ZEITLICHE verschiebung: die geometrie bleibt punktgenau stehen, nur die
# zeitspalte wandert -- wie beim starren nachziehen. Der gemeldete abstand
# muss davon unberuehrt bleiben, denn es ist dieselbe kurve und derselbe
# koerper-zustand. Der betrag ist eine halbe Mond-umlaufzeit geteilt durch
# 400, gross genug dass der Mond spuerbar weiterlaeuft.
shift = 5000.0
p.points[:, 2] += shift
p._points_time_offset = float(getattr(p, '_points_time_offset', 0.0)) + shift
p._invalidate_derived_caches()
r_after = periapsis_r(p)

if r_before is not None and r_after is not None:
    moved = abs(r_after - r_before)
    mond = w.body[ref_index]
    mond_flew = math.hypot(*(lambda a, b: (b.x - a.x, b.y - a.y))(
        mond.position_at_time(w.time), mond.position_at_time(w.time + shift)))
    check(moved < 1.0,
          "derselbe abstand fuer dieselbe geometrie",
          f"{r_before:.6e} -> {r_after:.6e} m (delta {moved:.3e} m; "
          f"der Mond lief in {shift:.0f} s um {mond_flew:.3e} m weiter)")

# ═══════════ 4. der zeitraffer-halt bleibt, was er war

print()
print("4. Der halt haelt weiterhin still")

w, ship, p = build()
ref_index, mu, rp_expected = place_lunar_flyby(w, ship)
p.set_reference_body_index(ref_index)
p.initialize(ship, w)
p.set_hold(True)

radii = []
counts = []
for _ in range(200):
    advance(w, FRAME_DT * 8.0)
    p.update(ship, w)
    r = periapsis_r(p)
    if r is not None:
        radii.append(r)
    counts.append(len(p.get_points()))

if radii:
    spread = max(radii) - min(radii)
    # Der halt rechnet zwischendurch nach (drift-schranke, vorrat), deshalb
    # nicht bit-genau -- aber weit unterhalb eines punktabstands.
    check(spread < 1e4, "auch im zeitraffer bewegt sich der Pe-abstand nicht",
          f"streuung {spread:.3e} m ueber {len(radii)} frames")
    check(min(counts) > 0, "und die linie laeuft nicht aus",
          f"kleinste punktzahl {min(counts)}")
else:
    check(False, "auch im zeitraffer bewegt sich der Pe-abstand nicht",
          "im halt wurde nie eine periapsis gemeldet")

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for f in FAILURES:
        print(f"  {f}")
    sys.exit(1)
print("Alle pruefungen bestanden.")
