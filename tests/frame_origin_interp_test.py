"""Der ursprung eines plot-rahmens wird KUBISCH interpoliert, nicht linear.

Gezeichnet wird `schiff(t) - ursprung(t)`. Der ursprung ist die position des
bezugskoerpers, und die wird nicht je punkt propagiert, sondern zwischen
hoechstens `frame_origin_interp_max_knots` (256) exakten stuetzstellen ueber
das zeitfenster der vorhersage interpoliert.

War diese interpolation LINEAR, dann steckte in der gezeichneten linie die
woelbung, die der bahn des BEZUGSKOERPERS fehlt: null auf jedem knoten,
maximal dazwischen -- also gleichfoermige baeuche mit harten ECKEN auf den
knoten. Und zwar ohne dass die vorhergesagten punkte selbst falsch waeren,
weshalb weder ein groesseres zeichenbudget noch eine feinere unterteilung
etwas half: beide zeichnen dieselbe verbogene kurve nur glatter.

Der fehler waechst mit dem ZEITFENSTER, also mit jedem '+'-druck auf den
horizont -- `R*theta^2/8` mit `theta = 2*pi*q/T`. Fuer die Erde bei 512x
horizont (fenster 5.3 jahre) sind das 3.2e8 m, im bild des fehlerberichts
rund 40 px.

Geprueft wird:

1. **Die knotenwerte selbst bleiben exakt.** Bei s = 0 und s = 1 muss die
   formel die stuetzstelle bitgleich zurueckgeben.
2. **Kubisch ist um groessenordnungen naeher an der wahrheit als linear** --
   mit einer gegenprobe, die den alten, linearen weg nachrechnet und
   durchfallen muss.
3. **Stapel- und skalarweg sind bitgleich.** Das ist die hausregel fuer jede
   vektorisierte fassung hier (siehe `_origin_xy_arrays`).
4. **Ohne zeitfenster (q <= 0) bleibt der exakte weg unangetastet.**

Aufruf: python tests/frame_origin_interp_test.py
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

from loader import SystemLoader
from reference_frames import BodyCentredNonRotatingReferenceFrame

FAILURES = []


def check(condition, label, detail=''):
    if condition:
        print(f"  ok   {label}{(' -- ' + detail) if detail else ''}")
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label} -- {detail}")


bodies = SystemLoader("solar_system.json").load()
erde = next(b for b in bodies if b.name == "Erde")
ERD_R = 1.496e11
ERD_T = 365.25 * 86400.0


def make_frame():
    f = BodyCentredNonRotatingReferenceFrame(erde)
    f.set_epoch_time(0.0)
    return f


def linear_reference(frame, body, t):
    """Der ALTE weg: sehne zwischen den beiden umgebenden knoten."""
    q = frame._origin_interp_q
    t0 = frame._origin_interp_t0
    n = math.floor((t - t0) / q)
    klo = t0 + n * q
    xlo, ylo = frame._body_world_position_exact(body, klo, None)
    xhi, yhi = frame._body_world_position_exact(body, klo + q, None)
    frac = (t - klo) / q
    return (xlo + (xhi - xlo) * frac, ylo + (yhi - ylo) * frac)


print("1. Die knotenwerte bleiben exakt")

f = make_frame()
p0, p1, p2, p3 = 3.0, 5.0, 11.0, 17.0
check(f._cubic_4pt(p0, p1, p2, p3, 0.0) == p1,
      "s = 0 gibt genau den linken knoten",
      f"{f._cubic_4pt(p0, p1, p2, p3, 0.0)!r} == {p1!r}")
check(f._cubic_4pt(p0, p1, p2, p3, 1.0) == p2,
      "s = 1 gibt genau den rechten knoten",
      f"{f._cubic_4pt(p0, p1, p2, p3, 1.0)!r} == {p2!r}")

# Eine echte kubik muss exakt reproduziert werden -- das ist der unterschied
# zu Catmull-Rom, das die steigung nur schaetzt und deshalb 3. ordnung ist.
def cubic(s):
    return 2.0 - 0.5 * s + 0.25 * s * s - 0.125 * s * s * s


worst = max(abs(f._cubic_4pt(cubic(-1.0), cubic(0.0), cubic(1.0), cubic(2.0), s)
                - cubic(s))
            for s in np.linspace(0.0, 1.0, 21))
check(worst < 1e-14, "eine kubik wird exakt wiedergegeben (4. ordnung)",
      f"groesster fehler {worst:.2e}")


print("\n2. Kubisch schlaegt linear um groessenordnungen")

# Zeitfenster wie bei langem horizont: 512x der grundreichweite entspricht
# rund 5.3 jahren. Bei 256 knoten ist q davon ein 256stel.
for label, span_years, bound_m in (("32x  (0.33 j)", 0.33, 2.0e5),
                                   ("128x (1.33 j)", 1.33, 2.0e6),
                                   ("512x (5.33 j)", 5.33, 3.0e7)):
    span = span_years * ERD_T
    f = make_frame()
    f.set_origin_interp_window(0.0, span, 4096)
    q = f._origin_interp_q
    # Mitten in einem intervall, dort ist der fehler beider verfahren maximal.
    ts = [q * (k + 0.5) for k in range(3, 40)]
    e_cub = e_lin = 0.0
    for t in ts:
        ex, ey = f._body_world_position_exact(erde, t, None)
        cx, cy = f._body_world_position_at_time(erde, t)
        lx, ly = linear_reference(f, erde, t)
        e_cub = max(e_cub, math.hypot(cx - ex, cy - ey))
        e_lin = max(e_lin, math.hypot(lx - ex, ly - ey))
    theta = 2.0 * math.pi * q / ERD_T
    check(e_cub < bound_m,
          f"{label}: kubischer fehler unter der schranke",
          f"{e_cub:.3e} m < {bound_m:.0e} m  (theta = {theta:.4f} rad)")
    # GEGENPROBE: der alte weg muss hier durchfallen, sonst misst der test
    # nichts (bei kleinem fenster sind beide gut genug).
    check(e_lin > bound_m,
          f"{label}: der lineare weg reisst dieselbe schranke",
          f"linear {e_lin:.3e} m, also {e_lin / max(e_cub, 1e-30):.0f}x schlechter")


print("\n3. Stapelweg und skalarweg sind bitgleich")

f = make_frame()
f.set_origin_interp_window(0.0, 5.33 * ERD_T, 4096)
q = f._origin_interp_q
times = np.array([q * (k + frac)
                  for k in range(2, 60)
                  for frac in (0.0, 0.13, 0.5, 0.87, 0.999)], dtype=np.float64)
batch = f._origin_xy_arrays(erde, times)
check(batch is not None, "der stapelweg greift ueberhaupt",
      "sonst prueft dieser abschnitt nichts")
if batch is not None:
    bx, by = batch
    sx = np.empty_like(bx)
    sy = np.empty_like(by)
    for i, t in enumerate(times):
        sx[i], sy[i] = f._body_world_position_at_time(erde, float(t))
    dx = np.abs(bx - sx).max()
    dy = np.abs(by - sy).max()
    check(dx == 0.0 and dy == 0.0,
          "stapel == skalar, bis aufs letzte bit",
          f"groesste abweichung {dx:.3e} / {dy:.3e} m ueber {times.size} zeiten")


print("\n4. Ohne zeitfenster bleibt der exakte weg unberuehrt")

f = make_frame()
f.set_origin_interp_window(0.0, 0.0)          # entartet -> q <= 0
same = True
for t in (0.0, 1234.5, 9.9e6):
    a = f._body_world_position_at_time(erde, t)
    b = f._body_world_position_exact(erde, t, None)
    same = same and a[0] == b[0] and a[1] == b[1]
check(same, "q <= 0 liefert exakt dieselben zahlen wie die exakte fassung")


print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for f_ in FAILURES:
        print(f"  {f_}")
    sys.exit(1)
print("rahmen-ursprung interpolation: alle pruefungen bestanden")
