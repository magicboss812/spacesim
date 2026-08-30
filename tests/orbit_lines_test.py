"""Regressionstest fuer orbit_lines.py -- die bahn-linien der koerper.

Reine funktionen: kein fenster, kein GL, kein pygame.
Aufruf: python tests/orbit_lines_test.py
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

from bodies import body as Body
from vec import Vec2, G
from reference_frames import BodyCentredNonRotatingReferenceFrame
import orbit_lines

FAILURES = []


def check(condition, label, detail=''):
    if condition:
        print(f"  ok   {label}" + (f": {detail}" if detail else ''))
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label}: {detail}")


def close(actual, expected, tol, label):
    err = abs(float(actual) - float(expected))
    check(err <= tol, label, f"{actual:.6e} vs {expected:.6e} (|d|={err:.3e} <= {tol:.3e})")


def make_body(name, mass, radius, a=None, e=0.0, arg=0.0, parent=None):
    b = Body(name=name, mass=mass, radius=radius,
             position=Vec2(0.0, 0.0), velocity=Vec2(0.0, 0.0),
             fixed=parent is not None, semi_major_axis=a, eccentricity=e,
             is_moon_of=parent)
    b.arg_periapsis = float(arg)
    return b


# ----------------------------------------------------------------------
print("\n§2  zukunfts-spur == das modell des PLOT-FRAMES (Kepler)")
# ----------------------------------------------------------------------
# Gezeichnet wird `spur(t)` durch den plot-frame, also `spur(t) -
# ursprung(t)`. Der ursprung kommt aus `reference_frames`, und das rechnet
# Kepler (mittlere anomalie + Newton) -- genau wie der praediktor, gegen
# dessen linie die spur ja gelesen wird. Die spur MUSS demselben modell
# folgen; jede abweichung landet ungefiltert in der gezeichneten linie.
#
# Frueher stand hier die forderung, `future_track` sei bitgleich zu
# `bodies.body.position_at_time`. Das war die falsche zwillingsfunktion:
# jenes modell rechnete mit KONSTANTER WINKELRATE und gehoerte dem
# welt-integrator, nicht dem praediktor. Der fehler war im spiel als
# davonfliegender Mond sichtbar.
#
# SEIT 2026-08-27 GIBT ES DIESEN UNTERSCHIED NICHT MEHR: die welt loest
# ebenfalls Kepler (`bodies.kepler_relative_xy`), weil die konstante rate
# ihre schrittweite von der chunk-groesse bezog und damit die physik an die
# raffungsstufe haengte. `position_at_time` deckt sich jetzt bis auf 1 mm
# mit dem rahmen-modell -- als gegenprobe taugt es also nicht mehr, und die
# alte naeherung wird hier eigens nachgebaut. Sie steht damit nur noch an
# der einen stelle, an der sie hingehoert: als das, was NICHT gerechnet
# werden soll.


def _constant_rate_position_at_time(b, t):
    """Die bis 2026-08-27 benutzte naeherung -- `theta` mit fester rate.

    Wortgleich zur alten `bodies.body.position_at_time`, damit die
    gegenproben unten weiter einen echten unterschied messen.
    """
    if b.is_moon_of is None or not b.semi_major_axis:
        return b.position.x, b.position.y
    a = float(b.semi_major_axis)
    e = float(b.eccentricity) if b.eccentricity else 0.0
    mu = G * b.is_moon_of.mass
    if mu <= 0.0:
        return b.position.x, b.position.y
    ref_theta = b._kepler_ref_theta
    dt = t - b._kepler_ref_time
    r_ref = a * (1.0 - e * e) / (1.0 + e * math.cos(ref_theta))
    v_ref = math.sqrt(max(0.0, mu * (2.0 / r_ref - 1.0 / a)))
    theta_t = ref_theta + (v_ref / max(1e-12, r_ref)) * dt
    r_t = a * (1.0 - e * e) / (1.0 + e * math.cos(theta_t))
    x_orb = r_t * math.cos(theta_t)
    y_orb = r_t * math.sin(theta_t)
    c = math.cos(b.arg_periapsis)
    sn = math.sin(b.arg_periapsis)
    px, py = x_orb * c - y_orb * sn, x_orb * sn + y_orb * c
    qx, qy = _constant_rate_position_at_time(b.is_moon_of, t)
    return px + qx, py + qy
sonne = make_body("Sonne", 1.989e30, 6.957e8)
saturn = make_body("Saturn", 5.683e26, 5.823e7, a=1.433e12, e=0.056, arg=0.31,
                   parent=sonne)
titan = make_body("Titan", 1.345e23, 2.575e6, a=1.222e9, e=0.029, arg=1.9,
                  parent=saturn)
for b, th in ((saturn, 0.83), (titan, 2.41)):
    b.theta = th
    b._kepler_ref_theta = th
    b._kepler_ref_time = 12345.0

times = np.linspace(12345.0, 12345.0 + 3.0 * 365.25 * 86400.0, 97)

_frame = BodyCentredNonRotatingReferenceFrame(sonne)
_frame.set_epoch_time(12345.0)

for b, label, scene in ((saturn, "ein glied (Saturn)", 1.433e12),
                        (titan, "zwei glieder (Titan)", 1.433e12)):
    track = orbit_lines.future_track(b, times)
    check(track.shape == (times.size, 2), f"{label}: form", str(track.shape))

    # Das modell, aus dem der rahmen seinen ursprung zieht.
    ref = np.array([_frame._body_world_position_exact(b, float(t)) for t in times])
    worst = float(np.max(np.hypot(track[:, 0] - ref[:, 0], track[:, 1] - ref[:, 1])))
    check(worst <= 1.0, f"{label}: deckt sich mit dem rahmen-modell",
          f"max|d| = {worst:.3e} m = {worst / scene:.2e} der szene")

    # Gegenprobe: das welt-modell reisst dieselbe schranke um himmelweite
    # betraege -- sonst koennte der test den fehler nicht von der korrektur
    # unterscheiden.
    old = np.array([_constant_rate_position_at_time(b, float(t)) for t in times])
    old_worst = float(np.max(np.hypot(old[:, 0] - ref[:, 0], old[:, 1] - ref[:, 1])))
    check(old_worst > 1e6 * worst and old_worst > 1e9,
          f"{label}: gegenprobe -- die konstante winkelrate reisst sie klar",
          f"max|d| = {old_worst:.3e} m ({old_worst / max(worst, 1e-9):.1e}x)")

# Und der punkt, an dem es im spiel schiefging: ein mond darf im rahmen
# seines eigenen planeten seine bahnschale NIE verlassen -- bei keinem
# horizont. Unter dem welt-modell tat er genau das.
_erde = make_body("Erde", 5.972e24, 6.371e6, a=1.496e11, e=0.0167, parent=sonne)
_mond = make_body("Mond", 7.342e22, 1.737e6, a=3.844e8, e=0.0549, parent=_erde)
for b in (_erde, _mond):
    b.theta = 0.0
    b._kepler_ref_theta = 0.0
    b._kepler_ref_time = 0.0

_ef = BodyCentredNonRotatingReferenceFrame(_erde)
_ef.set_epoch_time(0.0)
_lo, _hi = 3.844e8 * (1.0 - 0.0549), 3.844e8 * (1.0 + 0.0549)
for _days in (3.87, 30.0, 90.0, 365.0):
    _t = np.linspace(0.0, _days * 86400.0, 193)
    _o = np.array([_ef._body_world_position_exact(_erde, float(x)) for x in _t])
    _m = orbit_lines.future_track(_mond, _t)
    _r = np.hypot(_m[:, 0] - _o[:, 0], _m[:, 1] - _o[:, 1])
    check(_r.min() >= _lo * 0.999 and _r.max() <= _hi * 1.001,
          f"Mond bleibt im Erd-rahmen in seiner schale ({_days:g} d)",
          f"r {_r.min():.4e}..{_r.max():.4e} (schale {_lo:.4e}..{_hi:.4e})")

# Gegenprobe auf dem laengsten fenster: unter dem welt-modell war es weit
# daneben -- der Mond lief bis auf ein vielfaches der apoapsis hinaus.
_t = np.linspace(0.0, 365.0 * 86400.0, 193)
_o = np.array([_ef._body_world_position_exact(_erde, float(x)) for x in _t])
_old_m = np.array([_constant_rate_position_at_time(_mond, float(x)) for x in _t])
_old_r = np.hypot(_old_m[:, 0] - _o[:, 0], _old_m[:, 1] - _o[:, 1])
check(_old_r.max() > 10.0 * _hi,
      "gegenprobe: unter der konstanten winkelrate flog der Mond davon",
      f"r bis {_old_r.max():.4e} m = {_old_r.max() / _hi:.0f}x apoapsis")

# Und die aussage, die daraus geworden ist: das WELT-modell tut es nicht
# mehr -- es ist jetzt dasselbe Kepler-modell wie das des rahmens.
_world_m = np.array([[p.x, p.y] for p in (_mond.position_at_time(float(x)) for x in _t)])
_world_r = np.hypot(_world_m[:, 0] - _o[:, 0], _world_m[:, 1] - _o[:, 1])
check(_world_r.min() >= _lo * 0.999 and _world_r.max() <= _hi * 1.001,
      "das welt-modell bleibt jetzt selbst in der schale (365 d)",
      f"r {_world_r.min():.4e}..{_world_r.max():.4e}")

# Der stapelweg darf den koerper-zustand nicht anfassen -- er laeuft im
# render-frame, waehrend die welt dieselben objekte integriert.
check(saturn.theta == 0.83 and saturn._kepler_ref_time == 12345.0,
      "spur veraendert den koerper nicht",
      f"theta={saturn.theta} ref_t={saturn._kepler_ref_time}")

# Einzel- und stapelaufruf muessen bitgleich sein -- es ist EIN rechenweg.
_batch = orbit_lines.future_tracks((titan,), times)
check(float(np.max(np.abs(_batch[id(titan)] - orbit_lines.future_track(titan, times)))) == 0.0,
      "future_track und future_tracks sind bitgleich", "0.0 m")

# Koerper ohne elter (Sonne) hat keine bahn: konstante position.
sun_track = orbit_lines.future_track(sonne, times)
check(float(np.max(np.abs(sun_track - np.array([0.0, 0.0])))) == 0.0,
      "elternloser koerper steht still", "0.0 m")

# ----------------------------------------------------------------------
print("\n§3  einflusssphaere und deckkraft-rampe")
# ----------------------------------------------------------------------
erde = make_body("Erde", 5.972e24, 6.371e6, a=1.496e11, e=0.0167, parent=sonne)
mond = make_body("Mond", 7.342e22, 1.737e6, a=3.844e8, e=0.0549, parent=erde)

soi = orbit_lines.soi_radius(mond)
close(soi, 3.844e8 * (7.342e22 / 5.972e24) ** 0.4, 1.0, "Mond-SOI = a*(m/M)^0.4")
check(6.5e7 < soi < 6.8e7, "Mond-SOI liegt bei ~66 000 km", f"{soi:.4e} m")
check(orbit_lines.soi_radius(sonne) is None, "elternloser koerper hat keine SOI",
      repr(orbit_lines.soi_radius(sonne)))

FULL, FADE, AMAX, FLOOR = 1.0, 3.0, 0.85, 0.10


def alpha_at(multiple):
    return orbit_lines.approach_alpha(multiple * soi, soi, FULL, FADE, AMAX, FLOOR)


close(alpha_at(5.0), FLOOR, 1e-12, "weit draussen (5 SOI) -> boden")
close(alpha_at(3.0), FLOOR, 1e-12, "genau am rand (3 SOI) -> boden")
close(alpha_at(2.0), 0.5 * AMAX, 1e-12, "mitte (2 SOI) -> halbe helligkeit")
close(alpha_at(1.0), AMAX, 1e-12, "1 SOI -> volle helligkeit")
close(alpha_at(0.0), AMAX, 1e-12, "treffer -> volle helligkeit")

# Monoton fallend nach aussen -- sonst flackert die linie beim anfliegen.
samples = [alpha_at(m) for m in np.linspace(0.0, 4.0, 81)]
check(all(samples[i] >= samples[i + 1] - 1e-15 for i in range(len(samples) - 1)),
      "rampe ist monoton", f"{samples[0]:.3f} .. {samples[-1]:.3f}")

# Der boden gewinnt immer, auch wenn die naehe weniger hergibt.
close(orbit_lines.approach_alpha(9e9, soi, FULL, FADE, AMAX, 0.35), 0.35, 1e-12,
      "boden des referenzkoerpers gewinnt")
# Keine linie (kein praediktor): miss=inf faellt auf den boden.
close(orbit_lines.approach_alpha(float('inf'), soi, FULL, FADE, AMAX, FLOOR),
      FLOOR, 1e-12, "ohne annaeherung -> boden")

# ----------------------------------------------------------------------
print("\n§4  dichteste annaeherung, unterabtastung verfeinert")
# ----------------------------------------------------------------------
# Geradliniger vorbeiflug: das schiff zieht mit v in +x an einem ruhenden
# koerper vorbei, kleinster abstand B bei t=0. Die stichproben sind BEWUSST
# so gelegt, dass t=0 zwischen zwei von ihnen faellt -- genau der fall, in
# dem das rohe argmin daneben liegt.
B = 6.0e7
V = 1.0e4
step = 3600.0
t_s = np.arange(-6, 7, dtype=np.float64) * step + 0.37 * step
ship = np.stack([V * t_s, np.full_like(t_s, B)], axis=1)
bodyp = np.zeros_like(ship)

miss, t_min = orbit_lines.closest_approach(t_s, ship, bodyp)
raw = float(np.min(np.hypot(ship[:, 0], ship[:, 1])))

close(miss, B, 1.0, "verfeinertes minimum trifft den analytischen wert")
close(t_min, 0.0, 1e-3, "und den zeitpunkt")
# GEGENPROBE: ohne verfeinerung ist derselbe fall deutlich daneben.
check(abs(raw - B) > 1.0e6, "gegenprobe: rohes argmin verfehlt es klar",
      f"roh {raw:.4e} vs wahr {B:.4e} (|d|={abs(raw - B):.3e} m)")
check(abs(miss - B) < abs(raw - B) / 1000.0, "verfeinerung ist >1000x besser",
      f"{abs(miss - B):.3e} m gegen {abs(raw - B):.3e} m")

# Randfaelle: minimum am anfang/ende -> kein nachbar, roher wert.
t_edge = np.arange(0, 5, dtype=np.float64) * step
rising = np.stack([np.arange(5, dtype=np.float64) * 1e7 + 1e7,
                   np.zeros(5)], axis=1)
m_edge, _ = orbit_lines.closest_approach(t_edge, rising, np.zeros_like(rising))
close(m_edge, 1e7, 1e-9, "minimum am rand bleibt der stichprobenwert")

# Ein einziger punkt darf nicht krachen.
m_one, _ = orbit_lines.closest_approach(
    np.array([5.0]), np.array([[3.0, 4.0]]), np.zeros((1, 2)))
close(m_one, 5.0, 1e-12, "ein einzelner punkt")

# Leere eingabe -> unendlich, damit approach_alpha auf den boden faellt.
m_none, _ = orbit_lines.closest_approach(
    np.zeros(0), np.zeros((0, 2)), np.zeros((0, 2)))
check(m_none == float('inf'), "leere linie -> inf", str(m_none))

# ----------------------------------------------------------------------
print("\n§5  OrbitLineSet: ziele, boeden, einblendung, takt")
# ----------------------------------------------------------------------
T0 = 0.0
WINDOW = 12.0 * 86400.0
N_PTS = 1000
pt_t = np.linspace(T0, T0 + WINDOW, N_PTS)


def line_missing_moon_by(distance_m):
    """Praediktor-punkte, die der ZUKUNFTS-position des Mondes ueberall um
    mindestens `distance_m` ausweichen -- das minimum liegt in der mitte."""
    track = orbit_lines.future_track(mond, pt_t)
    u = (np.arange(N_PTS) - N_PTS * 0.5) / (N_PTS * 0.5)
    d = distance_m * (1.0 + 0.5 * u * u)
    pts = np.zeros((N_PTS, 5), dtype=np.float64)
    pts[:, 0] = track[:, 0] + d
    pts[:, 1] = track[:, 1]
    pts[:, 2] = pt_t
    return pts


mars = make_body("Mars", 6.417e23, 3.390e6, a=2.279e11, e=0.0934, parent=sonne)
bodies_list = [sonne, erde, mond, mars]
CFG = dict(track_samples=192, soi_full=FULL, soi_fade=FADE, alpha_max=AMAX,
           alpha_floor=FLOOR, alpha_floor_focus=0.35, fade_rate=6.0)

# a) Treffer-naehe: der Mond wird hell, die Erde bleibt auf ihrem boden.
oset = orbit_lines.OrbitLineSet(**CFG)
pts = line_missing_moon_by(0.5 * soi)
for _ in range(400):
    oset.update(bodies_list, pts, sim_time=T0, real_dt=1.0 / 60.0)
close(oset.target_alpha(mond), AMAX, 1e-9, "Mond bei 0.5 SOI -> ziel = alpha_max")
close(oset.alpha(mond), AMAX, 1e-3, "und die eingeblendete deckkraft folgt")
close(oset.alpha(mars), FLOOR, 1e-3, "Mars weit weg -> boden")
# Die Erde dagegen ist HELL, und das ist richtig: der Mond umlaeuft sie in
# 3.84e8 m, ihre SOI misst 9.25e8 m -- wer beim Mond steht, steht INNERHALB
# der Erd-SOI. Ihre bahnlinie ist dort die relevanteste ueberhaupt.
check(oset.miss(erde) < orbit_lines.soi_radius(erde),
      "Erde: schiff liegt innerhalb ihrer SOI",
      f"{oset.miss(erde):.3e} m < {orbit_lines.soi_radius(erde):.3e} m")
close(oset.alpha(erde), AMAX, 1e-3, "und ist daher voll sichtbar")
check(oset.get(sonne) is None, "Sonne (kein elter) bekommt keine linie",
      repr(oset.get(sonne)))
close(oset.miss(mond), 0.5 * soi, 1e-3 * soi, "gemessener fehlabstand",)

# b) Weit daneben -> boden.
oset_far = orbit_lines.OrbitLineSet(**CFG)
pts_far = line_missing_moon_by(4.0 * soi)
for _ in range(400):
    oset_far.update(bodies_list, pts_far, sim_time=T0, real_dt=1.0 / 60.0)
close(oset_far.alpha(mond), FLOOR, 1e-3, "Mond bei 4 SOI -> boden")

# c) Referenz- und auswahlkoerper bekommen den hoeheren boden.
oset_ref = orbit_lines.OrbitLineSet(**CFG)
for _ in range(400):
    oset_ref.update(bodies_list, pts_far, sim_time=T0, real_dt=1.0 / 60.0,
                    reference_body=mars, selected_body=mond)
close(oset_ref.alpha(mars), 0.35, 1e-3, "referenzkoerper -> fokus-boden")
close(oset_ref.alpha(mond), 0.35, 1e-3, "ausgewaehlter koerper -> fokus-boden")
# Gegenprobe: ohne fokus faellt derselbe Mars auf den niedrigen boden.
close(oset_far.alpha(mars), FLOOR, 1e-3, "gegenprobe: ohne fokus nur FLOOR")

# d) Die einblendung ist bildratenunabhaengig (1 - exp(-rate*dt)).
slow = orbit_lines.OrbitLineSet(**CFG)
fast = orbit_lines.OrbitLineSet(**CFG)
for _ in range(6):
    slow.update(bodies_list, pts, sim_time=T0, real_dt=0.10)
for _ in range(60):
    fast.update(bodies_list, pts, sim_time=T0, real_dt=0.01)
close(slow.alpha(mond), fast.alpha(mond), 2e-3,
      "6x100ms == 60x10ms (bildratenunabhaengig)")

# e) Der takt: ohne aenderung wird NICHT neu gerechnet.
paced = orbit_lines.OrbitLineSet(**CFG)
paced.update(bodies_list, pts, sim_time=T0, real_dt=0.016, generation=7)
first = paced.recomputes
for _ in range(50):
    paced.update(bodies_list, pts, sim_time=T0, real_dt=0.016, generation=7)
check(paced.recomputes == first, "50 ruhige frames -> 0 neuberechnungen",
      f"{paced.recomputes} == {first}")
paced.update(bodies_list, pts, sim_time=T0, real_dt=0.016, generation=8)
check(paced.recomputes == first + 1, "neue generation -> genau eine",
      f"{paced.recomputes}")
# ... und wenn die simzeit um mehr als einen stichproben-schritt vorrueckt.
paced.update(bodies_list, pts, sim_time=T0 + WINDOW, real_dt=0.016, generation=8)
check(paced.recomputes == first + 2, "simzeit-sprung -> genau eine",
      f"{paced.recomputes}")

# f) Ohne praediktor-linie faellt alles auf den boden zurueck, ohne krach.
oset.update(bodies_list, None, sim_time=T0, real_dt=1.0)
for _ in range(400):
    oset.update(bodies_list, None, sim_time=T0, real_dt=1.0 / 60.0)
close(oset.alpha(mond), FLOOR, 1e-3, "praediktor aus -> boden")


# ----------------------------------------------------------------------
print("\n§6  kosten am echten sonnensystem")
# ----------------------------------------------------------------------
from loader import SystemLoader

real_bodies = SystemLoader("solar_system.json").load()
n_lines = sum(1 for b in real_bodies
              if not getattr(b, 'is_ship', False)
              and orbit_lines.soi_radius(b) is not None)
check(n_lines >= 20, "koerper mit bahnlinie", f"{n_lines} von {len(real_bodies)}")

real_pts = np.zeros((10000, 5), dtype=np.float64)
real_pts[:, 2] = np.linspace(0.0, 4.0 * 86400.0, 10000)
real_pts[:, 0] = 1.496e11 + np.linspace(0.0, 4.0e8, 10000)

real_set = orbit_lines.OrbitLineSet(track_samples=192)
real_set.update(real_bodies, real_pts, sim_time=0.0, real_dt=0.016)  # warmlaufen

samples = []
for i in range(20):
    real_set.update(real_bodies, real_pts, sim_time=0.0, real_dt=0.016,
                    generation=100 + i)
    samples.append(real_set.last_recompute_ms)
samples.sort()
median_ms = samples[len(samples) // 2]
check(real_set.recomputes == 21, "je generation genau eine neuberechnung",
      str(real_set.recomputes))
check(median_ms < 5.0, "eine volle neuberechnung unter 5 ms",
      f"median {median_ms:.3f} ms, schlechteste {samples[-1]:.3f} ms")

# Der ruhige frame -- der normalfall -- darf praktisch nichts kosten.
t0 = time.perf_counter()
for _ in range(200):
    real_set.update(real_bodies, real_pts, sim_time=0.0, real_dt=0.016,
                    generation=999)
quiet_ms = (time.perf_counter() - t0) * 1000.0 / 200.0
check(quiet_ms < 0.20, "ruhiger frame (nur nachblenden) unter 0.20 ms",
      f"{quiet_ms:.4f} ms")

# ----------------------------------------------------------------------
print("\n§7  projektion je ZEIT statt je PUNKT")
# ----------------------------------------------------------------------
from reference_frames import (BodyCentredNonRotatingReferenceFrame,
                              BodyCentredBodyDirectionReferenceFrame)

r_sonne = next(b for b in real_bodies if b.name == 'Sonne')
r_erde = next(b for b in real_bodies if b.name == 'Erde')
r_mond = next(b for b in real_bodies if b.name == 'Mond')

proj_t = np.linspace(0.0, 30.0 * 86400.0, 233)
proj_track = orbit_lines.future_track(r_mond, proj_t)
proj_x = np.ascontiguousarray(proj_track[:, 0])
proj_y = np.ascontiguousarray(proj_track[:, 1])


class CountingFrame:
    """Zaehlt die skalaren aufrufe und reicht sie an einen echten rahmen weiter."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def to_this_frame_xy(self, t, x, y):
        self.calls += 1
        return self.inner.to_this_frame_xy(t, x, y)


for inner, label in (
        (BodyCentredNonRotatingReferenceFrame(r_erde), "nicht-rotierend"),
        (BodyCentredBodyDirectionReferenceFrame(r_erde, r_sonne), "richtungs-frame"),
):
    fx, fy = orbit_lines.frame_project(inner, proj_t, proj_x, proj_y)
    ref = [inner.to_this_frame_xy(float(t), float(x), float(y))
           for t, x, y in zip(proj_t, proj_x, proj_y)]
    ref_x = np.array([p[0] for p in ref])
    ref_y = np.array([p[1] for p in ref])
    worst = float(max(np.max(np.abs(fx - ref_x)), np.max(np.abs(fy - ref_y))))
    # Nicht bitgleich, sondern rundungsgleich: der affine weg rechnet
    # dieselbe starre abbildung mit anderen operationen. Der massstab sind
    # 1e11 m, 1e-4 m davon sind 1e-15 relativ -- und 1e-9 pixel bei jedem
    # zoom, den das spiel zulaesst.
    check(worst < 1.0e-4, f"{label}: gleich bis auf rundung",
          f"max|d| = {worst:.3e} m von ~1e11 m")

    # Die eigentliche pointe: drei sondierungen JE ZEIT, nicht drei je punkt.
    counter = CountingFrame(inner)
    one_t = np.full(512, 4321.0)
    ell_x = np.linspace(1.0e11, 1.1e11, 512)
    ell_y = np.linspace(0.0, 1.0e10, 512)
    orbit_lines.frame_project(counter, one_t, ell_x, ell_y)
    check(counter.calls == 3, f"{label}: 512 punkte, EINE zeit -> 3 aufrufe",
          f"{counter.calls} aufrufe")

    counter = CountingFrame(inner)
    orbit_lines.frame_project(counter, proj_t, proj_x, proj_y)
    check(counter.calls == 3 * 233, f"{label}: 233 zeiten -> 3 je zeit",
          f"{counter.calls} aufrufe")

    # Der bild-cache teilt die transformation zwischen den koerpern: der
    # zweite koerper auf DERSELBEN zeitachse kostet gar nichts mehr.
    shared = {}
    counter = CountingFrame(inner)
    orbit_lines.frame_project(counter, proj_t, proj_x, proj_y, cache=shared)
    first_calls = counter.calls
    orbit_lines.frame_project(counter, proj_t, proj_x * 1.01, proj_y, cache=shared)
    check(counter.calls == first_calls,
          f"{label}: zweiter koerper aus dem cache -> 0 zusaetzliche aufrufe",
          f"{first_calls} -> {counter.calls}")

# Eine nicht-starre abbildung darf NICHT durchgehen -- sonst wuerde der
# schnelle weg still etwas anderes zeichnen als der rahmen meint.
class _ScalingFrame:
    def to_this_frame_xy(self, t, x, y):
        return x * 2.0, y * 3.0


check(orbit_lines.frame_affine_at(_ScalingFrame(), 0.0) is None,
      "skalierende abbildung wird als nicht-starr erkannt")
sx_, sy_ = orbit_lines.frame_project(_ScalingFrame(), proj_t, proj_x, proj_y)
check(float(np.max(np.abs(sx_ - proj_x * 2.0))) == 0.0,
      "und faellt exakt auf die punktweise schleife zurueck", "0.0 m")

# Eine reine verschiebung ist starr und muss den schnellen weg nehmen.
class _ShiftFrame:
    def to_this_frame_xy(self, t, x, y):
        return x - 1000.0 * t, y + 2000.0 * t


fx, fy = orbit_lines.frame_project(_ShiftFrame(), proj_t, proj_x, proj_y)
check(float(np.max(np.abs(fx - (proj_x - 1000.0 * proj_t)))) < 1e-4,
      "reine verschiebung: starr, exakt", "< 1e-4 m")

# ----------------------------------------------------------------------
print("\n§9  skalare zeit: die ellipse gilt zu EINEM zeitpunkt")
# ----------------------------------------------------------------------
# Die ellipse ist ein momentaner ort. Ein array voller identischer zeiten
# dafuer zu bauen und durch np.unique zu schicken ist arbeit fuer nichts --
# `frame_project` nimmt deshalb auch einen skalar.
sc_frame = BodyCentredBodyDirectionReferenceFrame(r_erde, r_sonne)
ell_x = np.linspace(1.0e11, 1.1e11, 2048)
ell_y = np.linspace(0.0, 1.0e10, 2048)

ax_, ay_ = orbit_lines.frame_project(sc_frame, 4321.0, ell_x, ell_y)
bx_, by_ = orbit_lines.frame_project(sc_frame, np.full(2048, 4321.0), ell_x, ell_y)
check(float(np.max(np.abs(ax_ - bx_))) == 0.0
      and float(np.max(np.abs(ay_ - by_))) == 0.0,
      "skalar == array voller derselben zeit", "0.0 m")

counter = CountingFrame(sc_frame)
orbit_lines.frame_project(counter, 4321.0, ell_x, ell_y)
check(counter.calls == 3, "2048 punkte, skalare zeit -> 3 aufrufe",
      f"{counter.calls} aufrufe")

t0 = time.perf_counter()
for _ in range(200):
    orbit_lines.frame_project(sc_frame, 4321.0, ell_x, ell_y, cache={(id(sc_frame), 4321.0): orbit_lines.frame_affine_at(sc_frame, 4321.0)})
scalar_us = (time.perf_counter() - t0) * 1e6 / 200.0
check(scalar_us < 200.0, "2048 punkte projizieren unter 200 us",
      f"{scalar_us:.1f} us")

# ----------------------------------------------------------------------
print("\n§10 alle koerper teilen sich EINE zeitachse")
# ----------------------------------------------------------------------
# Die zukunfts-spuren aller koerper stehen auf demselben zeit-array. Der
# erste koerper darf die transformationstabelle bauen, die restlichen 25
# muessen sie geschenkt bekommen -- sonst kostet jeder von ihnen wieder
# eine Python-schleife ueber alle stichprobenzeiten. Gemessen waren das
# 5.0 ms je bild fuer am ende DREI gezeichnete kurven.
tt = np.ascontiguousarray(np.linspace(0.0, 12.0 * 86400.0, 192))
tracks = [orbit_lines.future_track(b, tt)
          for b in real_bodies if orbit_lines.soi_radius(b) is not None]
check(len(tracks) >= 20, "spuren fuer den test", f"{len(tracks)}")

shared_cache = {}
tf = BodyCentredBodyDirectionReferenceFrame(r_erde, r_sonne)

t0 = time.perf_counter()
orbit_lines.frame_project(tf, tt, np.ascontiguousarray(tracks[0][:, 0]),
                          np.ascontiguousarray(tracks[0][:, 1]),
                          cache=shared_cache)
first_us = (time.perf_counter() - t0) * 1e6

t0 = time.perf_counter()
for tr in tracks[1:]:
    orbit_lines.frame_project(tf, tt, np.ascontiguousarray(tr[:, 0]),
                              np.ascontiguousarray(tr[:, 1]),
                              cache=shared_cache)
rest_us = (time.perf_counter() - t0) * 1e6 / max(1, len(tracks) - 1)

check(rest_us < 25.0, "jeder weitere koerper unter 25 us",
      f"erster {first_us:.1f} us, danach je {rest_us:.1f} us")
check(rest_us < first_us / 5.0, "und mindestens 5x billiger als der erste",
      f"{rest_us:.1f} us gegen {first_us:.1f} us")

# Das ergebnis muss dasselbe sein wie ohne cache -- sonst waere es nur schnell.
a1 = orbit_lines.frame_project(tf, tt, np.ascontiguousarray(tracks[3][:, 0]),
                               np.ascontiguousarray(tracks[3][:, 1]),
                               cache=shared_cache)
a2 = orbit_lines.frame_project(tf, tt, np.ascontiguousarray(tracks[3][:, 0]),
                               np.ascontiguousarray(tracks[3][:, 1]))
check(float(np.max(np.abs(a1[0] - a2[0]))) == 0.0
      and float(np.max(np.abs(a1[1] - a2[1]))) == 0.0,
      "mit und ohne cache bitgleich", "0.0 m")

# ----------------------------------------------------------------------
print("\n§11 aufloesung der zukunfts-spur")
# ----------------------------------------------------------------------
# Die spur wird fuer die MESSUNG mit 192 stichproben gebraucht, fuers
# ZEICHNEN fast nie. Der stride kommt aus derselben fehlerrechnung wie die
# segmentzahl der ellipse, und er rastet auf eine leiter ein, damit die
# koerper sich ihre zeitpunkte teilen (der cache liegt auf (rahmen, zeit)).
DIAG2 = math.hypot(1280.0, 800.0)


def sag_of(stride, arc_px, r_px, n=192):
    pts = len(range(0, n, stride))
    return (arc_px / max(1, pts - 1)) ** 2 / (8.0 * r_px)


for arc_px, r_px, label in ((1.0e4, 1.5e5, "Erdbahn, kurzer bogen"),
                            (9.2e4, 1.0e5, "Mondbahn, weit hineingezoomt"),
                            (300.0, 400.0, "kleine bahn, klein am schirm"),
                            (5.0, 50.0, "fast ein punkt")):
    st = orbit_lines.polyline_stride(192, arc_px, r_px, DIAG2, 0.3)
    check(1 <= st <= 192, f"{label}: stride im bereich", f"stride={st}")
    check(sag_of(st, arc_px, r_px) <= 0.3 * 1.35,
          f"{label}: pfeilhoehe haelt die toleranz",
          f"{sag_of(st, arc_px, r_px):.4f} px bei stride {st}")

# Die pointe: ein kurzer bogen darf grob abgetastet werden, ein langer nicht.
st_short = orbit_lines.polyline_stride(192, 1.0e4, 1.5e5, DIAG2, 0.3)
st_long = orbit_lines.polyline_stride(192, 9.2e4, 1.0e5, DIAG2, 0.3)
check(st_short > st_long, "kurzer bogen bekommt groesseren stride",
      f"{st_short} > {st_long}")
check(st_short > 1, "und spart wirklich etwas", f"stride={st_short}")

# Die leiter: nur teilerfreundliche werte, damit die zeitpunkte sich decken.
seen = {orbit_lines.polyline_stride(192, a, r, DIAG2, 0.3)
        for a in (1e2, 1e3, 1e4, 1e5, 1e6) for r in (1e2, 1e4, 1e6)}
check(seen <= set(orbit_lines.STRIDE_LADDER), "alle strides von der leiter",
      f"{sorted(seen)}")

# --- und der boden-vergleich, den der renderer benutzt ------------------
oset_arc = orbit_lines.OrbitLineSet(**CFG)
for _ in range(5):
    oset_arc.update(bodies_list, pts, sim_time=T0, real_dt=1.0 / 60.0)
e_mond = oset_arc.get(mond)
e_mars = oset_arc.get(mars)
check(e_mond.target > e_mond.floor + 1e-6,
      "Mond mit annaeherung: ueber seinem boden -> heller bogen",
      f"ziel {e_mond.target:.3f} > boden {e_mond.floor:.3f}")
check(e_mars.target <= e_mars.floor + 1e-6,
      "Mars ohne annaeherung: genau auf dem boden -> kein bogen",
      f"ziel {e_mars.target:.3f} == boden {e_mars.floor:.3f}")
check(e_mond.track_len > 0.0, "spurlaenge wird mitgefuehrt",
      f"{e_mond.track_len:.3e} m")

# ----------------------------------------------------------------------
print("\n§12 die abtastung darf das ENDE nicht verlieren")
# ----------------------------------------------------------------------
# `track[::stride]` schneidet den schwanz ab, wenn (n-1) nicht durch stride
# teilbar ist -- bei 192 punkten und stride 64 endet die kurve auf index
# 128 von 191, also fehlen 33 %. Genau das ist der endpunkt, an dem die
# ganze funktion haengt: er MUSS der koerper zur endzeit des praediktors
# sein, sonst zeigt er auf nichts.
for n in (192, 191, 100, 65, 33, 9):
    for stride in orbit_lines.STRIDE_LADDER:
        idx = orbit_lines.stride_indices(n, stride)
        check(idx[0] == 0 and idx[-1] == n - 1,
              f"n={n} stride={stride}: erster 0, letzter {n-1}",
              f"{idx[0]} .. {idx[-1]} ({idx.size} punkte)")
        check(bool(np.all(np.diff(idx) > 0)), f"n={n} stride={stride}: streng steigend")

# GEGENPROBE: die nackte scheibe verliert das ende, sonst prueft §12 nichts.
naive = np.arange(0, 192, 64)
check(naive[-1] != 191, "gegenprobe: track[::64] endet NICHT auf dem letzten punkt",
      f"{naive[-1]} statt 191")

# ----------------------------------------------------------------------
print("\n§13 FrameAffineTable: transformation je FENSTER statt je punkt")
# ----------------------------------------------------------------------
WIN_T0 = 0.0
for win_days, label in ((3.8, "1x horizont (3.8 d)"), (240.0, "64x horizont (8 mon)")):
    win_t1 = WIN_T0 + win_days * 86400.0
    probe_t = np.linspace(WIN_T0, win_t1, 401)
    px_ = np.full(401, 1.0e11)
    py_ = np.linspace(-2.0e10, 2.0e10, 401)

    for inner, fl in (
            (BodyCentredNonRotatingReferenceFrame(r_erde), "nicht-rotierend"),
            (BodyCentredBodyDirectionReferenceFrame(r_erde, r_sonne), "richtung"),
    ):
        tab = orbit_lines.FrameAffineTable(inner, WIN_T0, win_t1)
        check(tab.valid, f"{label} / {fl}: tabelle gebaut",
              f"{tab.knots} knoten, {tab.probes} sondierungen")
        fx, fy = tab.project(probe_t, px_, py_)
        ref = [inner.to_this_frame_xy(float(t), float(x), float(y))
               for t, x, y in zip(probe_t, px_, py_)]
        worst = float(max(np.max(np.abs(fx - np.array([p[0] for p in ref]))),
                          np.max(np.abs(fy - np.array([p[1] for p in ref])))))
        # Der massstab ist NICHT ein absoluter meterwert, sondern was davon
        # auf dem schirm ankommt. Die referenz ist die knoten-interpolation
        # des rahmens selbst: sie steht laut CLAUDE.md bei 0.54 px (512x
        # horizont, 256 knoten). Bei einem zoom, der die ganze bahn ins bild
        # bringt (~5.4e-7 px/m), sind 1e-6 von 1e11 m gerade 0.054 px --
        # also eine groessenordnung besser als das, was die gezeichnete
        # vorhersagelinie ohnehin schon mitbringt.
        rel = worst / 1.0e11
        check(rel < 1.0e-6,
              f"{label} / {fl}: relativ zur szene unter 1e-6",
              f"max|d| = {worst:.3e} m = {rel:.2e} rel = {worst * 5.4e-7:.4f} px")
        # Die sondierungen muessen VIEL weniger sein als die punkte, sonst
        # ist nichts gewonnen.
        check(tab.probes < 3 * 401, f"{label} / {fl}: billiger als punktweise",
              f"{tab.probes} gegen {3*401}")

# GEGENPROBE: mit fest 2 knoten reisst dasselbe fenster die schranke -- sonst
# wuerde die adaptive knotenzahl gar nichts beweisen.
win_t1 = WIN_T0 + 240.0 * 86400.0
probe_t = np.linspace(WIN_T0, win_t1, 401)
px_ = np.full(401, 1.0e11)
py_ = np.linspace(-2.0e10, 2.0e10, 401)
inner = BodyCentredNonRotatingReferenceFrame(r_erde)
coarse = orbit_lines.FrameAffineTable(inner, WIN_T0, win_t1, knot_min=2, knot_max=2)
fx, fy = coarse.project(probe_t, px_, py_)
ref = [inner.to_this_frame_xy(float(t), float(x), float(y))
       for t, x, y in zip(probe_t, px_, py_)]
worst_coarse = float(max(np.max(np.abs(fx - np.array([p[0] for p in ref]))),
                         np.max(np.abs(fy - np.array([p[1] for p in ref])))))
check(worst_coarse > 1.0e6, "gegenprobe: 2 knoten reissen die schranke klar",
      f"max|d| = {worst_coarse:.3e} m")

# Ein NICHT rotierender rahmen hat drehwinkel 0 und trotzdem einen
# gekruemmten ursprung -- wer nur die drehung misst, nimmt hier den
# minimalwert und liegt daneben.
tab_nr = orbit_lines.FrameAffineTable(
    BodyCentredNonRotatingReferenceFrame(r_erde), WIN_T0, win_t1)
check(tab_nr.knots > 8, "nicht-rotierend: knoten kommen aus der URSPRUNGS-bewegung",
      f"{tab_nr.knots} knoten (minimum waere 8)")

# Auf den knoten selbst darf nichts verschoben werden.
tab = orbit_lines.FrameAffineTable(
    BodyCentredBodyDirectionReferenceFrame(r_erde, r_sonne), WIN_T0, win_t1)
knot_t = np.array([WIN_T0 + j * tab.q for j in range(1, tab.knots)])
kx = np.full(knot_t.size, 1.0e11)
ky = np.zeros(knot_t.size)
fx, fy = tab.project(knot_t, kx, ky)
ref = [tab_frame.to_this_frame_xy(float(t), float(x), float(y))
       for tab_frame, t, x, y in zip([BodyCentredBodyDirectionReferenceFrame(r_erde, r_sonne)] * knot_t.size, knot_t, kx, ky)]
worst_knot = float(max(np.max(np.abs(fx - np.array([p[0] for p in ref]))),
                       np.max(np.abs(fy - np.array([p[1] for p in ref])))))
check(worst_knot < 1.0, "auf den knoten exakt (<1 m)", f"{worst_knot:.3e} m")


# ----------------------------------------------------------------------
print("\n§14 enthuellung: wie VIEL der linie sichtbar ist")
# ----------------------------------------------------------------------
R_FULL, R_FADE = 10.0, 30.0


def reveal_at(mult):
    return orbit_lines.reveal_fraction(mult * soi, soi, R_FULL, R_FADE)


close(reveal_at(40.0), 0.0, 1e-12, "weit weg -> gar nichts")
close(reveal_at(30.0), 0.0, 1e-12, "am rand des bandes -> gar nichts")
close(reveal_at(20.0), 0.5, 1e-12, "mitte -> halbe linie")
close(reveal_at(10.0), 1.0, 1e-12, "10 SOI -> ganze linie")
close(reveal_at(1.0), 1.0, 1e-12, "nah -> ganze linie")
close(orbit_lines.reveal_fraction(float('inf'), soi, R_FULL, R_FADE), 0.0, 1e-12,
      "ohne annaeherung -> nichts")
rr = [reveal_at(m) for m in np.linspace(0.0, 40.0, 81)]
check(all(rr[i] >= rr[i + 1] - 1e-15 for i in range(len(rr) - 1)),
      "monoton fallend", f"{rr[0]:.2f} .. {rr[-1]:.2f}")
# Die enthuellung ist WEITER als die helligkeit: die endkappe soll da sein,
# waehrend man noch steuert.
check(reveal_at(5.0) == 1.0 and orbit_lines.approach_alpha(
          5.0 * soi, soi, FULL, FADE, AMAX, 0.0) == 0.0,
      "bei 5 SOI: linie ganz da, aber noch dunkel -- genau der punkt",
      "reveal 1.0 / alpha 0.0")

# ----------------------------------------------------------------------
print("\n§15 die linie gehoert dem PLOT-FRAME, nicht dem weltraum")
# ----------------------------------------------------------------------
W_T0, W_T1 = 0.0, 3.8 * 86400.0
w_times = np.linspace(W_T0, W_T1, 192)

# a) Der ursprungskoerper des rahmens steht in seinem eigenen rahmen still.
#    Eine linie fuer ihn ist definitionsgemaess sinnlos -- und genau das war
#    der fehlerbericht: die Erde zog im Erd-rahmen eine bahn um die Sonne.
mond_frame = BodyCentredNonRotatingReferenceFrame(r_mond)
mond_frame.set_epoch_time(W_T0)
check(orbit_lines.frame_origin_body(mond_frame) is r_mond,
      "ursprungskoerper wird erkannt (nicht-rotierend)")
check(orbit_lines.frame_origin_body(
          BodyCentredBodyDirectionReferenceFrame(r_erde, r_sonne)) is r_erde,
      "ursprungskoerper wird erkannt (richtungs-frame)")
check(orbit_lines.frame_origin_body(None) is None, "kein rahmen -> kein ursprung")

# b) Und der REST muss sich mitdrehen. Die Erde im Mond-rahmen zeigt die
#    relativbewegung; die alte starre fassung (alles zu EINER zeit) zeigt
#    stattdessen die ungedrehte bahn.
erde_track = orbit_lines.future_track(r_erde, w_times)
ex = np.ascontiguousarray(erde_track[:, 0])
ey = np.ascontiguousarray(erde_track[:, 1])

tab = orbit_lines.FrameAffineTable(mond_frame, W_T0, W_T1)
fx_t, fy_t = tab.project(w_times, ex, ey)          # zeitrichtig
fx_s, fy_s = orbit_lines.frame_project(mond_frame, W_T0, ex, ey)  # alt: eine zeit

# Zeitrichtig: der abstand Erde-Mond bleibt ueber das fenster fast konstant
# (beide laufen zusammen um die Sonne), die kurve ist also kurz.
len_t = float(np.hypot(np.diff(fx_t), np.diff(fy_t)).sum())
len_s = float(np.hypot(np.diff(fx_s), np.diff(fy_s)).sum())
check(len_t < len_s / 10.0,
      "zeitrichtig ist die Erde im Mond-rahmen eine KURZE kurve",
      f"{len_t:.3e} m gegen {len_s:.3e} m starr ({len_s/len_t:.0f}x)")
r_t = np.hypot(fx_t, fy_t)
check(r_t.max() / r_t.min() < 1.5,
      "und sie bleibt auf Mond-abstand, statt um die Sonne zu laufen",
      f"r zwischen {r_t.min():.3e} und {r_t.max():.3e} m")

# c) Die linie STARTET auf dem koerper: bei t=now stimmen das modell des
#    praediktors und das des rahmens exakt ueberein.
own = orbit_lines.future_track(r_mond, w_times)
ox_, oy_ = tab.project(w_times, np.ascontiguousarray(own[:, 0]),
                       np.ascontiguousarray(own[:, 1]))
close(float(np.hypot(ox_[0], oy_[0])), 0.0, 1.0,
      "der ursprungskoerper sitzt bei t=now exakt im ursprung")

# d) DIE endkappe: beide enden auf einem punkt heisst treffer.
#    Konstruiert: ein schiff, das genau die Mondbahn nachfaehrt.
hit = np.zeros((192, 5))
hit[:, 0] = own[:, 0]
hit[:, 1] = own[:, 1]
hit[:, 2] = w_times
oset_cap = orbit_lines.OrbitLineSet(track_samples=192)
oset_cap.update([r_sonne, r_erde, r_mond], hit, sim_time=W_T0, real_dt=1.0)
e = oset_cap.get(r_mond)
cap_gap = float(np.hypot(e.track[-1, 0] - hit[-1, 0], e.track[-1, 1] - hit[-1, 1]))
close(cap_gap, 0.0, 1.0, "treffer: beide endkappen auf demselben punkt")
close(e.miss, 0.0, 1.0, "und der fehlabstand ist null")

# GEGENPROBE: dasselbe schiff eine halbe stunde phasenverschoben trifft nicht.
late = np.zeros((192, 5))
shifted = orbit_lines.future_track(r_mond, w_times + 1800.0)
late[:, 0] = shifted[:, 0]
late[:, 1] = shifted[:, 1]
late[:, 2] = w_times
oset_late = orbit_lines.OrbitLineSet(track_samples=192)
oset_late.update([r_sonne, r_erde, r_mond], late, sim_time=W_T0, real_dt=1.0)
e2 = oset_late.get(r_mond)
gap2 = float(np.hypot(e2.track[-1, 0] - late[-1, 0], e2.track[-1, 1] - late[-1, 1]))
check(gap2 > 1.0e6, "gegenprobe: 30 min versatz trennt die endkappen deutlich",
      f"{gap2:.3e} m")
check(e2.miss > 1.0e6, "und der fehlabstand steigt entsprechend",
      f"{e2.miss:.3e} m")

# ----------------------------------------------------------------------
print("\n§16 faint volllinie -- EIN ganzer umlauf, zeitrichtig im plot-frame")
# ----------------------------------------------------------------------
# `orbital_period` = 2*pi*sqrt(a^3/mu). Gegen den analytischen wert einer
# bekannten bahn.
_per_mond = orbit_lines.orbital_period(mond)
_mu_em = G * erde.mass
_per_analytic = 2.0 * math.pi * math.sqrt((3.844e8) ** 3 / _mu_em)
close(_per_mond, _per_analytic, 1.0, "orbital_period(Mond) = 2*pi*sqrt(a^3/mu)")
check(orbit_lines.orbital_period(sonne) is None,
      "elternloser koerper hat keine periode", repr(orbit_lines.orbital_period(sonne)))

# Der elternrelative offset des Mondes ist eine exakte Kepler-ellipse und
# schliesst sich ueber eine periode auf sich selbst -- das ist der beleg,
# dass die periode stimmt. (Die WELTbahn tut das nicht: die Erde traegt den
# Mond ueber 27 tage ~4.5e10 m um die Sonne. Genau darum ist die volllinie
# eine zeitkurve im plot-frame, keine starr transformierte weltbahn.)
_t0 = 5000.0
_full_t = _t0 + np.linspace(0.0, _per_mond, 256)
_ft = orbit_lines.future_track(mond, _full_t)
_par = orbit_lines.future_track(erde, _full_t)
_rel = _ft - _par
_gap = float(np.hypot(_rel[-1, 0] - _rel[0, 0], _rel[-1, 1] - _rel[0, 1]))
check(_gap < 1e-6 * 3.844e8,
      "Mond: elternrelative bahn schliesst sich ueber eine periode",
      f"luecke {_gap:.3e} m von a=3.844e8 m")
_gap_world = float(np.hypot(_ft[-1, 0] - _ft[0, 0], _ft[-1, 1] - _ft[0, 1]))
check(_gap_world > 1e10, "gegenprobe: die WELTbahn schliesst sich nicht",
      f"luecke {_gap_world:.3e} m (Erde traegt den Mond fort)")
# GEGENPROBE: eine um 10 % falsche periode reisst die schleife auf.
_wrong_t = _t0 + np.linspace(0.0, 1.1 * _per_mond, 256)
_relw = orbit_lines.future_track(mond, _wrong_t) - orbit_lines.future_track(erde, _wrong_t)
_gap_w = float(np.hypot(_relw[-1, 0] - _relw[0, 0], _relw[-1, 1] - _relw[0, 1]))
check(_gap_w > 0.05 * 3.844e8, "gegenprobe: 10 % falsche periode -> klare luecke",
      f"luecke {_gap_w:.3e} m")

# Zeitrichtig im plot-frame: im nicht-rotierenden Sonnen-rahmen ist die
# Erdbahn ueber ein jahr eine geschlossene schleife -- durch DIESELBE
# transformationspipeline wie die enthuellte spur.
_erde2 = make_body("Erde", 5.972e24, 6.371e6, a=1.496e11, e=0.0167, parent=sonne)
_erde2.theta = 0.0
_erde2._kepler_ref_theta = 0.0
_erde2._kepler_ref_time = 0.0
_per_erde = orbit_lines.orbital_period(_erde2)
_sf = BodyCentredNonRotatingReferenceFrame(sonne)
_sf.set_epoch_time(0.0)
_et = np.linspace(0.0, _per_erde, 256)
_etrack = orbit_lines.future_track(_erde2, _et)
_tab = orbit_lines.FrameAffineTable(_sf, 0.0, _per_erde, knot_angle=0.12)
check(_tab.valid, "volllinien-tabelle gebaut (jahr, groeber)",
      f"{_tab.knots} knoten, {_tab.probes} sondierungen")
_fx, _fy = _tab.project(_et, np.ascontiguousarray(_etrack[:, 0]),
                        np.ascontiguousarray(_etrack[:, 1]))
_loop_gap = float(np.hypot(_fx[-1] - _fx[0], _fy[-1] - _fy[0]))
_loop_r = float(np.hypot(_fx, _fy).mean())
check(_loop_gap < 1e-4 * _loop_r,
      "Erdbahn schliesst sich im Sonnen-rahmen zur schleife",
      f"luecke {_loop_gap:.3e} m von r~{_loop_r:.3e} m")
check(1.4e11 < _loop_r < 1.6e11, "und sie liegt bei ~1 AE",
      f"r~{_loop_r:.3e} m")

# OrbitLineSet fuellt full_track: fenster startet am praediktor-anfang und
# spannt genau eine periode.
oset16 = orbit_lines.OrbitLineSet(**CFG, full_orbit_enabled=True,
                                  full_samples=256, full_max_span_s=4.0e8)
pts16 = line_missing_moon_by(0.5 * soi)
for _ in range(5):
    oset16.update(bodies_list, pts16, sim_time=T0, real_dt=1.0 / 60.0)
e16 = oset16.get(mond)
check(e16.full_track is not None and e16.full_track.shape == (256, 2),
      "Mond: full_track gefuellt", str(None if e16.full_track is None else e16.full_track.shape))
close(float(e16.full_track_t[0]), float(pts16[0, 2]), 1.0,
      "full_track startet am fensteranfang des praediktors")
close(float(e16.full_track_t[-1] - e16.full_track_t[0]), _per_mond, 1.0,
      "und spannt genau eine umlaufperiode")
check(e16.full_track_len > 0.0, "full_track_len wird mitgefuehrt",
      f"{e16.full_track_len:.3e} m")

# Der deckel: ein koerper mit periode ueber full_max_span_s bekommt keine.
oset_cap16 = orbit_lines.OrbitLineSet(**CFG, full_orbit_enabled=True,
                                      full_max_span_s=1.0e6)
for _ in range(5):
    oset_cap16.update(bodies_list, pts16, sim_time=T0, real_dt=1.0 / 60.0)
check(oset_cap16.get(mond).full_track is None,
      "periode ueber dem deckel -> keine volllinie",
      f"periode {_per_mond:.3e} s > 1e6 s")

# Abgeschaltet -> nichts.
oset_off16 = orbit_lines.OrbitLineSet(**CFG, full_orbit_enabled=False)
for _ in range(5):
    oset_off16.update(bodies_list, pts16, sim_time=T0, real_dt=1.0 / 60.0)
check(oset_off16.get(mond).full_track is None, "full_orbit_enabled=False -> keine volllinie")

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for failure in FAILURES:
        print(f"  {failure}")
    sys.exit(1)
print("orbit_lines.py: alle pruefungen bestanden")
