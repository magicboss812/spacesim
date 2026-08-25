"""Regressionen fuer horizont, welt-integrator und zeitraffer-halt.

Alle drei pruefungen messen eine GROESSE, nicht eine implementierung -- sie
bleiben also gueltig, wenn die umsetzung sich aendert:

1. **Horizont ueberlebt den zoom.** Ein feinerer punktabstand darf die
   vorhersage nicht verkuerzen. Das war der fehler, bei dem die linie an der
   ersten bildkante endete und Ap/Pe sowie CLOSEST/T-CA mit verschwanden.
2. **Numba-integrator == Python-integrator.** Bit fuer bit, sonst ist die
   uebersetzung falsch (und nicht etwa "etwas ungenauer").
3. **Die kurve zittert nicht und laeuft nicht aus.** Gemessen wird die
   bewegung eines punktes FESTER SIM-ZEIT zwischen zwei frames -- bei einer
   stabilen kurve ist das null. Gilt seit dem verbrauchen statt verschieben
   fuer beide betriebsarten, nicht mehr nur fuer den zeitraffer-halt.
4. **Echtzeit ist bei jeder bildrate erreichbar.** Der sim_dt-boden ist je
   TICK angegeben, die warp-stufen je ECHTSEKUNDE -- bei 180 fps sperrte der
   boden sonst die unterste stufe aus und der schub blieb ueberall gesperrt.
5. **Nach reset() kommt die linie zurueck.** Sonst faellt der navball auf die
   geradeaus-tangente zurueck und die marker springen.
6. **Die tangente ueberlebt die schrumpfende kopfsehne.** Im halt laeuft die
   erste sehne stetig auf null; ohne MINDESTLAENGE teilt sie den seitlichen
   versatz des schiffs und die richtung schlaegt in jede richtung aus.

Aufruf: python tests/warp_predictor_test.py
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

from vec import G
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


TICK = 60.0
MAX_SUBSTEP = 1000.0


def build(async_compute=False, fast_integrator=True):
    config = ConfigLoader(None)
    config.load()
    bodies = SystemLoader("solar_system.json").load()
    w = World(G)
    w.body = bodies
    config.apply_to_world(w)
    w.use_fast_integrator = fast_integrator
    ship = next(b for b in bodies if b.is_ship)
    p = Predictor(recompute_every_update=True, **config.predictor_kwargs())
    config.apply_to_predictor(p)
    p.set_length(p.num_points * p.precision)
    p.async_compute = bool(async_compute)
    return w, ship, p


def advance(w, sim_seconds):
    """Ein weltschritt -- IN DER REIHENFOLGE DES SPIELS.

    `test.py::update` ruft erst `update_dynamics`, dann `update_planets`.
    Diese reihenfolge ist nicht beliebig: `update_planets` schreibt
    `body.theta` um einen ganzen chunk vor und merkt sich `self.time` als
    epoche -- die dann bereits das chunk-ende ist, weil `update_dynamics`
    die uhr schon gestellt hat. Vertauscht liegt der bookmark einen chunk
    daneben, und `position_at_time` liefert jeden geskripteten koerper einen
    chunk ZU FRUEH BZW. ZU SPAET. Gemessen (erdumlaufbahn rp 2e7 m, e = 0.3,
    abstand der welt von der analytisch propagierten predictor-linie nach
    4800 s):

        chunk                       1000 s    300 s     5 s
        dynamics, planets (spiel)   5.2e1 m   5.2e1 m   5.2e1 m
        planets, dynamics           9.4e6 m   3.9e6 m   7.4e4 m

    Dieser helfer hatte die reihenfolge vertauscht und hat den halt damit
    gegen eine welt gemessen, die das spiel so gar nicht rechnet. §18 haelt
    die aussage jetzt ausdruecklich fest.
    """
    steps = max(1, int(math.ceil(sim_seconds / MAX_SUBSTEP)))
    dt = sim_seconds / steps
    for _ in range(steps):
        w.update_dynamics(dt)
        w.update_planets(dt)


def arc_length(points):
    a = np.asarray(points)
    if len(a) < 2:
        return 0.0
    d = np.diff(a[:, :2], axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


# ══════════════════════════════════ 1. horizont ueberlebt den zoom

print("1. Vorhersage-horizont bleibt beim hineinzoomen erhalten")

w, ship, p = build()
horizon = float(p.length)
for scale in (2e-9, 2e-7, 2e-6, 2e-5, 2e-4, 2e-3):
    p._view_scale = None
    p.set_view_scale(scale)
    p.reset()
    p.update(ship, w)
    traced = arc_length(p.get_points())
    ratio = traced / horizon
    # 2e-5 und 2e-4 waren vorher bei 10 % bzw. 1 %.
    check(ratio > 0.95, f"view_scale {scale:.0e}: horizont zu >95 % gezeichnet",
          f"{ratio * 100:.1f} % ({traced:.3e} von {horizon:.3e} m)")

p._view_scale = None
p.set_view_scale(2e-4)
check(p._effective_precision() >= p._horizon_spacing_floor() - 1e-9,
      "punktabstand faellt nicht unter die horizont-schranke",
      f"eff={p._effective_precision():.4g} floor={p._horizon_spacing_floor():.4g}")

# ═══════════════════════════ 2. numba-integrator == python-integrator

print()
print("2. Schneller integrator liefert exakt dasselbe wie der langsame")

for label, total in (("1000 s", 1000.0), ("100 800 s (7d/s-frames)", 100800.0)):
    wa, sa, _ = build(fast_integrator=False)
    wb, sb, _ = build(fast_integrator=True)
    advance(wa, total)
    advance(wb, total)
    dpos = math.hypot(sa.position.x - sb.position.x, sa.position.y - sb.position.y)
    dvel = math.hypot(sa.velocity.x - sb.velocity.x, sa.velocity.y - sb.velocity.y)
    check(dpos == 0.0 and dvel == 0.0,
          f"{label}: position und geschwindigkeit identisch",
          f"|dpos|={dpos:.3e} m |dvel|={dvel:.3e} m/s")
    check(wa.integrator_last_substeps == wb.integrator_last_substeps,
          f"{label}: gleiche anzahl teilschritte",
          f"{wa.integrator_last_substeps} vs {wb.integrator_last_substeps}")

# ═══════════════════════════════════ 3. zeitraffer-halt

print()
print("3. Zeitraffer-halt: kein zittern, kein auslaufen")


def probe(a, t):
    """Kurvenpunkt zur ABSOLUTEN sim-zeit t (linear interpoliert)."""
    ts = a[:, 2]
    if t < ts[0] or t > ts[-1]:
        return None
    i = max(1, min(int(np.searchsorted(ts, t)), len(ts) - 1))
    f = (t - ts[i - 1]) / max(1e-9, ts[i] - ts[i - 1])
    return (a[i - 1, 0] + (a[i, 0] - a[i - 1, 0]) * f,
            a[i - 1, 1] + (a[i, 1] - a[i - 1, 1]) * f)


def run_warp(hold, frames=200, warp=604800.0):
    w, ship, p = build(async_compute=True)
    p.set_view_scale(2e-9)
    p.update(ship, w)
    p.set_hold(hold)
    sim_dt = warp / TICK
    prev = np.asarray(p.get_points()).copy()
    jumps = []
    min_lead = 1e30
    min_points = 10 ** 9
    for _ in range(frames):
        advance(w, sim_dt * TICK * (1 / 60.0))
        p.update(ship, w)
        cur = np.asarray(p.get_points())
        if len(cur) < 2:
            return None, None, 0, 0
        lead = float(cur[-1, 2]) - w.time
        min_lead = min(min_lead, lead)
        min_points = min(min_points, len(cur))
        t = w.time + min(1.0e5, max(0.0, lead) * 0.5)
        a1, a2 = probe(prev, t), probe(cur, t)
        if a1 and a2:
            jumps.append(math.hypot(a2[0] - a1[0], a2[1] - a1[1]))
        prev = cur.copy()
    return np.array(jumps), min_lead, min_points, len(prev)


loose, _, _, _ = run_warp(False)
held, min_lead, min_points, final_n = run_warp(True)

median_loose = float(np.median(loose))
median_held = float(np.median(held))
p99_held = float(np.percentile(held, 99))

# FRUEHER WAR DAS EIN VERGLEICH hold GEGEN nicht-hold, und der ging um zwei
# groessenordnungen aus. Ohne halt zog `_anchor_first_point` die kurve jeden
# frame STARR ans schiff -- hier gemessene 3.176e+07 m median je frame, gegen
# 0.0 m mit halt. Inzwischen VERBRAUCHT auch die echtzeit die kurve, statt sie
# zu verschieben (Predictor._advance_points_along_curve), und beide wege stehen
# gleich still. Ein relativvergleich misst damit nichts mehr (0 < 0/100 ist
# falsch). Die groesse, auf die es ankommt, ist der ABSOLUTE versatz eines
# punktes fester sim-zeit -- und der gehoert in pixeln gemessen, weil nur das
# sichtbar ist.
check(median_held * 2e-6 < 0.5 and median_loose * 2e-6 < 0.5,
      "die kurve steht still -- mit halt wie ohne",
      f"median {median_held:.3e} m gehalten vs {median_loose:.3e} m ohne halt "
      f"(= {median_held * 2e-6:.4f} / {median_loose * 2e-6:.4f} px)")
# Bei 2e-6 px/m (zoomstufe des fehler-screenshots) sind 5e5 m rund 1 px.
check(p99_held * 2e-6 < 0.5,
      "gehaltene kurve bleibt auch im 99. perzentil unter einem halben pixel",
      f"p99 = {p99_held:.3e} m = {p99_held * 2e-6:.4f} px")
check(min_points > 4 and min_lead > 0.0,
      "die linie laeuft im halt nie aus (failsafe greift)",
      f"kleinste punktzahl {min_points}, kleinster zeitvorlauf {min_lead:.0f} s")

# Der halt darf nicht dauerhaft rechnen: gezaehlt wird, wie oft er
# durchgereicht statt gehalten hat.
w, ship, p = build(async_compute=True)
p.set_view_scale(2e-9)
p.update(ship, w)
p.set_hold(True)
original = p._hold_advance
stats = {'calls': 0, 'held': 0}


def counting(ship_arg, world_arg):
    stats['calls'] += 1
    result = original(ship_arg, world_arg)
    stats['held'] += 1 if result else 0
    return result


p._hold_advance = counting
for _ in range(300):
    advance(w, (604800.0 / TICK) * TICK * (1 / 60.0))
    p.update(ship, w)

refreshes = stats['calls'] - stats['held']
frames_per_refresh = stats['calls'] / max(1, refreshes)
check(frames_per_refresh >= 20.0,
      "nachgerechnet wird selten, nicht jeden frame",
      f"alle {frames_per_refresh:.0f} frames ({frames_per_refresh / 60.0:.2f} s bei 60 fps)")
check(stats['held'] / stats['calls'] > 0.9,
      "ueber 90 % der frames werden gehalten",
      f"{100.0 * stats['held'] / stats['calls']:.0f} %")

print()
print("4. Die gehaltene kurve rueckt STETIG vor, nicht in stufen")

# Stuetzstellen lassen sich nur ganz wegwerfen. Wuerde als kopf die naechste
# stuetzstelle VOR dem schiff stehenbleiben und der rest per abklingender
# korrektur nachgezogen, liefe dieser rest zwischen zwei verbrauchten
# stuetzstellen um bis zu EINER PUNKTWEITE hin und her -- ein saegezahn, der
# auf jeder zoomstufe gleich aussieht, weil er eine weltlaenge ist.
# Stattdessen wird die schiffsposition als kopf vorangestellt.
w, ship, p = build(async_compute=False)
p.set_view_scale(2e-9)
p.update(ship, w)
p.set_hold(True)

head_gap = 0.0
tail_move = 0.0
first_seg = []
previous = None
previous_was_hold = False
for _ in range(120):
    advance(w, (604800.0 / TICK) * TICK * (1 / 60.0))
    p.update(ship, w)
    a = np.asarray(p.get_points())
    if len(a) < 4:
        break
    head_gap = max(head_gap, math.hypot(a[0, 0] - ship.position.x,
                                        a[0, 1] - ship.position.y))
    span = math.hypot(a[2, 0] - a[1, 0], a[2, 1] - a[1, 1])
    first_seg.append(math.hypot(a[1, 0] - a[0, 0], a[1, 1] - a[0, 1]) / max(span, 1e-9))

    # Die punktzahl faellt beim verbrauchen und springt beim auffrischen.
    is_hold_step = previous is not None and len(a) < len(previous)
    if is_hold_step and previous_was_hold:
        # Die eigentliche zusicherung: die stuetzstellen hinter dem kopf sind
        # DIESELBEN wie im vorframe -- nicht 'aehnlich', sondern bitgleich.
        # Verglichen werden die ueberlappenden stuetzstellen ueber ihre zeit,
        # der angestueckelte kopf bleibt dabei aussen vor.
        tail, previous_tail = a[1:], previous[1:]
        offset = int(np.searchsorted(previous_tail[:, 2], tail[0, 2]))
        count = min(len(previous_tail) - offset, len(tail))
        if count > 0:
            moved = np.hypot(
                previous_tail[offset:offset + count, 0] - tail[:count, 0],
                previous_tail[offset:offset + count, 1] - tail[:count, 1])
            tail_move = max(tail_move, float(moved.max()))
    previous = a.copy()
    previous_was_hold = is_hold_step

check(head_gap == 0.0, "der kopf IST die schiffsposition",
      f"groesster abstand {head_gap:.3e} m")
check(tail_move == 0.0, "hinter dem kopf bewegt sich beim halten nichts",
      f"groesste bewegung {tail_move:.3e} m")
first_seg = np.array(first_seg)
check(first_seg.min() < 0.15 and first_seg.max() > 0.85,
      "das erste segment laeuft stetig von voller auf null laenge",
      f"{first_seg.min():.2f} .. {first_seg.max():.2f} punktweiten")

print()
print("5. Die linie rollt: vorn verbraucht, hinten verlaengert")

# Ohne verlaengerung wird die gehaltene kurve nur von vorn aufgebraucht. Sie
# schrumpft dann sichtbar, bis eine vollberechnung sie schlagartig wieder auf
# volle laenge bringt -- die linie pulsiert im takt der auffrischung. Mit
# fortsetzung bleibt der horizont konstant und es gibt (fast) keine
# vollberechnungen mehr.
for warp_label, warp in (('1h/s', 3600.0), ('1d/s', 86400.0), ('7d/s', 604800.0)):
    w, ship, p = build(async_compute=True)
    p.set_view_scale(2e-9)
    p.update(ship, w)
    p.set_hold(True)

    # Zaehler als liste, weil das hier modulebene ist (kein nonlocal).
    full_computes = [0]
    original_full = p._compute_full

    def counting_full(ship_arg, world_arg, _orig=original_full, _n=full_computes):
        _n[0] += 1
        return _orig(ship_arg, world_arg)

    p._compute_full = counting_full

    lengths = []
    worst_seg_ratio = 0.0
    for f in range(240):
        advance(w, (warp / TICK) * TICK * (1 / 60.0))
        p.update(ship, w)
        a = np.asarray(p.get_points())
        if len(a) < 8:
            break
        lengths.append(arc_length(a) / p.length)
        if f > 3:
            # Ohne das angestueckelte kopf-teilsegment: alle uebrigen
            # abstaende muessen der punktweite entsprechen. Eine naht mit
            # luecke faellt hier sofort auf.
            seg = np.hypot(np.diff(a[1:, 0]), np.diff(a[1:, 1]))
            worst_seg_ratio = max(worst_seg_ratio,
                                  float(seg.max()) / max(float(np.median(seg)), 1e-9))

    lengths = np.array(lengths[3:])
    check(lengths.min() > 0.95 and lengths.max() < 1.05,
          f"{warp_label}: horizont bleibt konstant",
          f"laenge {lengths.min():.3f} .. {lengths.max():.3f} des horizonts")
    check(worst_seg_ratio < 1.5,
          f"{warp_label}: die naht ist luecklos",
          f"groesstes segment = {worst_seg_ratio:.2f} x punktweite")
    check(full_computes[0] <= 3,
          f"{warp_label}: kaum noch vollberechnungen",
          f"{full_computes[0]} in 240 frames")

print()
print("6. Echtzeit ist bei jeder bildrate erreichbar")

# min_sim_dt ist sim-sekunden JE TICK, die warp-stufen sind sim-sekunden je
# ECHTSEKUNDE. Der config-boden 1.0 passte zu 60 fps; bei window.fps = 180
# war die langsamste erreichbare rate 180 s/s -- dauerhaft ueber
# realtime_warp_max. Im spiel hiess das: der schub war in JEDER stufe
# gesperrt, der regler zeigte staendig "HOLD", und die vorhersage kam nie aus
# dem zeitraffer-halt heraus.
from camera import Camera
from ui.hud.layout import WARP_STEPS

_config = ConfigLoader(None)
_config.load()
REALTIME_MAX = float(_config.get('simulation.realtime_warp_max', 60.0))
SLOWEST_STEP = float(WARP_STEPS[0][0])

check(abs(SLOWEST_STEP - REALTIME_MAX) < 1e-9,
      "unterste warp-stufe == realtime_warp_max",
      f"stufe {SLOWEST_STEP:.1f} s/s, grenze {REALTIME_MAX:.1f} s/s")

locked_at = []
for fps in (30, 60, 90, 120, 144, 180, 240):
    cam = Camera(None, 100, 100)
    _config.apply_to_camera(cam)
    tick = float(fps)
    cam.allow_warp_rate(REALTIME_MAX, tick)

    # Genau der weg, den Hud._set_warp fuer die unterste stufe nimmt.
    cam.sim_dt = max(cam.min_sim_dt, min(cam.max_sim_dt, SLOWEST_STEP / tick))
    rate = cam.sim_dt * tick
    if rate > REALTIME_MAX * 1.001:
        locked_at.append((fps, rate))

check(not locked_at, "unterste stufe laesst den schub frei",
      f"gesperrt bei {locked_at}" if locked_at
      else "geprueft bei 30..240 fps, ueberall genau 60.00 s/s")

# Und PageDown muss ebenfalls dorthin kommen -- der boden gilt fuer beide wege.
cam = Camera(None, 100, 100)
_config.apply_to_camera(cam)
cam.allow_warp_rate(REALTIME_MAX, 180.0)
cam.sim_dt = 900.0
for _ in range(60):
    cam.sim_dt = max(cam.sim_dt / cam.sim_dt_factor, cam.min_sim_dt)
check(abs(cam.sim_dt * 180.0 - REALTIME_MAX) < 1e-6,
      "PageDown erreicht denselben boden",
      f"{cam.sim_dt * 180.0:.2f} s/s")

# Und zuletzt das, was der spieler wirklich sieht: der schubregler. Er liest
# Telemetry.thrust_locked und schreibt dann "HOLD" statt des prozentwerts.
from ui.hud.telemetry import Telemetry

_w, _ship, _p = build()
_cam = Camera(None, 100, 100)
_config.apply_to_camera(_cam)
_cam.allow_warp_rate(REALTIME_MAX, 180.0)
_tel = Telemetry(_w, _ship, None, _cam, None, _p, None, tick_rate=180.0)
_tel.realtime_warp_max = REALTIME_MAX

_cam.sim_dt = max(_cam.min_sim_dt, SLOWEST_STEP / 180.0)
_tel.sample()
check(not _tel.thrust_locked,
      "regler zeigt in der untersten stufe den prozentwert",
      f"{_tel.warp_factor:.2f} s/s -> '{_tel.text_throttle()}'")

_cam.sim_dt = 604800.0 / 180.0
_tel.sample()
check(_tel.thrust_locked, "regler zeigt bei 7d/s weiterhin HOLD",
      f"{_tel.warp_factor:.0f} s/s")

print()
print("7. Nach reset() kommt die linie zurueck -- auch im zeitraffer")

# predictor.reset() haengt an '9'/'0'/'+'/'-' (test.py). Danach vergleicht
# update() die schiffsgeschwindigkeit gegen _last_swapped_snapshot, um schub
# zu erkennen. Blieb der vermerk nach dem reset stehen, wich die
# geschwindigkeit jeden frame weiter davon ab -> jeden frame eine neue
# trajektorien-version -> der laufende hintergrund-auftrag wurde verworfen,
# bevor er fertig war. Gemessen: 20 frames, 20 auftraege, 0 eingewechselt,
# die linie kam NIE zurueck. Ohne linie faellt der navball von der
# gezeichneten bahn auf die geradeaus-tangente zurueck -- das springen der
# marker.
w, ship, p = build(async_compute=True)
p.set_view_scale(2e-6)
p.update(ship, w)
p.set_hold(True)
for _ in range(12):
    advance(w, 86400.0 / 60.0)
    p.update(ship, w)

check(len(p.get_points()) > 0, "vorlauf: linie steht",
      f"{len(p.get_points())} punkte")

p.set_precision(max(1.0, p.precision / 2.0))
p.reset()
check(p._last_swapped_snapshot is None,
      "reset() loescht den vermerk des letzten ergebnisses")

recovered = -1
versions = []
for f in range(30):
    advance(w, 86400.0 / 60.0)
    p.set_hold(True)
    p.update(ship, w)
    versions.append(int(p._trajectory_version))
    time.sleep(0.004)
    if recovered < 0 and len(p.get_points()) > 0:
        recovered = f

# Die schranke ist bewusst grosszuegig: die schleife pollt mit 4 ms je frame,
# der neuaufbau laeuft im worker und dauert mit dem vollen sonnensystem ~45 ms
# -- die frame-zahl bis zur rueckkehr ist also eine WANDUHR-groesse und damit
# maschinenabhaengig (gemessen 10 .. 11 auf demselben rechner, mit und ohne
# jede aenderung am predictor). Geprueft wird die EIGENSCHAFT, die der fehler
# verletzte: die linie kommt ueberhaupt zurueck, und zwar in einem bruchteil
# des fensters -- nicht nie (recovered = -1) und nicht erst am ende (29).
check(0 <= recovered <= 15, "die linie ist nach wenigen frames wieder da",
      f"nach {recovered} frames" if recovered >= 0 else "gar nicht")
check(len(set(versions)) == 1,
      "keine trajektorien-invalidierung je frame",
      f"version {versions[0]} .. {versions[-1]}, {len(set(versions))} verschiedene")

print()
print("8. Die tangente ueberlebt die schrumpfende kopfsehne")

# Im halt ist points[0] die exakte schiffsposition, points[1] eine stehende
# stuetzstelle: die kopfsehne c laeuft zwischen zwei verbrauchten stuetzstellen
# stetig von einer vollen punktweite auf NULL. Das schiff liegt dabei nie exakt
# auf der kurve (welt und predictor propagieren die planeten leicht
# verschieden, ~37 m). Der winkelfehler ist ~ d/c und wird mit c -> 0
# beliebig gross -- das sind die navball-marker, die nach ein paar sekunden
# zeitraffer in jede richtung springen.
from reference_frames import IdentityReferenceFrame, _prograde_from_line

_frame = IdentityReferenceFrame()
_S = 1.0e6        # stuetzweite
_D = 37.0         # dokumentierter seitlicher versatz je frame
_X0 = 1.496e11    # heliozentrische groessenordnung -- hier zaehlt die aufloesung


def _line_with_head(c, d):
    """Gerade in +x mit stuetzweite _S; kopf c davor und d seitlich versetzt."""
    n = 20
    pts = np.empty((n + 1, 3), dtype=np.float64)
    pts[0] = (_X0 - c, d, 0.0)
    for i in range(n):
        pts[i + 1] = (_X0 + i * _S, 0.0, (c + i * _S) / 7546.0)
    return pts


worst_angle = 0.0
worst_frac = None
for frac in (1.0, 0.5, 0.25, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6):
    direction = _prograde_from_line(_frame, _line_with_head(_S * frac, _D))
    if direction is None:
        worst_angle = 999.0
        worst_frac = frac
        break
    angle = abs(math.degrees(math.atan2(direction.y, direction.x)))
    if angle > worst_angle:
        worst_angle = angle
        worst_frac = frac

check(worst_angle < 0.1,
      "kopfsehne bis auf 1e-6 punktweiten: richtung bleibt stabil",
      f"groesste abweichung {worst_angle:.4f} Grad (bei c/weite = {worst_frac:.0e})")


def _first_chord_direction(points):
    """Die alte regel: erste sehne ueber 1e-12 m, ohne mindestlaenge."""
    p0 = points[0]
    fx0, fy0 = _frame.to_this_frame_xy(float(p0[2]), float(p0[0]), float(p0[1]))
    for i in range(1, min(len(points), 16)):
        pi = points[i]
        fxi, fyi = _frame.to_this_frame_xy(float(pi[2]), float(pi[0]), float(pi[1]))
        dx, dy = fxi - fx0, fyi - fy0
        mag = math.hypot(dx, dy)
        if mag > 1e-12:
            return dx / mag, dy / mag
    return None


# Gegenprobe: bei GLEICHMAESSIGEN stuetzweiten -- also ueberall ausserhalb des
# halts -- muss exakt dasselbe herauskommen wie vorher. Die mindestlaenge darf
# nur dort greifen, wo der kopf kuenstlich nah sitzt.
_uniform = np.empty((20, 3), dtype=np.float64)
for i in range(20):
    _a = i * 0.01
    _uniform[i] = (_X0 + math.cos(_a) * _S * i, math.sin(_a) * _S * i, i * 132.0)

_new = _prograde_from_line(_frame, _uniform)
_old = _first_chord_direction(_uniform)
check(_new is not None and _old is not None
      and _new.x == _old[0] and _new.y == _old[1],
      "gleichmaessige stuetzweiten: bit-identisch zur alten regel",
      f"{_new.x:.17g}, {_new.y:.17g}")

# Und dasselbe am echten lauf: ueber viele zeitraffer-frames darf der fehler
# gegen die wahre bahntangente weder gross werden noch mit der zeit wachsen.
from reference_frames import (
    BodyCentredNonRotatingReferenceFrame,
    apparent_orbital_directions,
)

w, ship, p = build()
earth = next(b for b in w.body if b.name == "Erde")
p.set_view_scale(2e-5)
p.set_reference_body_index(w.body.index(earth))
live_frame = BodyCentredNonRotatingReferenceFrame(earth)
live_frame.set_epoch_time(w.time)


def _earth_velocity():
    """Skriptgefuehrte koerper haben velocity == 0 -- zentraldifferenz noetig."""
    a = earth.position_at_time(w.time + 1.0)
    b = earth.position_at_time(w.time - 1.0)
    return (a.x - b.x) * 0.5, (a.y - b.y) * 0.5


SIM_PER_FRAME = 3600.0 / 180.0     # 1h/s bei 180 fps
p.update(ship, w)
p.set_hold(True)
for _ in range(20):
    advance(w, SIM_PER_FRAME)
    p.update(ship, w)

angle_errors = []
for f in range(900):
    advance(w, SIM_PER_FRAME)
    p.set_hold(True)
    p.update(ship, w)
    directions = apparent_orbital_directions(
        live_frame, w.time, ship.position, ship.velocity,
        ref_pos=earth.position, points=p.get_points(),
    )
    pro = directions.get('prograde')
    if pro is None:
        angle_errors.append(999.0)
        continue
    evx, evy = _earth_velocity()
    truth = math.degrees(math.atan2(-(ship.velocity.y - evy),
                                    ship.velocity.x - evx)) % 360.0
    drawn = math.degrees(math.atan2(-pro.y, pro.x)) % 360.0
    angle_errors.append(abs(((drawn - truth + 180) % 360) - 180))

angle_errors = np.array(angle_errors)
check(angle_errors.max() < 0.1,
      "echter zeitraffer-lauf: prograde bleibt auf der bahntangente",
      f"900 frames, fehler median {np.median(angle_errors):.4f}, "
      f"max {angle_errors.max():.4f} Grad")

first_half = angle_errors[:450].max()
second_half = angle_errors[450:].max()
check(second_half < max(first_half * 2.0, 0.02),
      "der fehler waechst nicht mit der zeit im zeitraffer",
      f"max erste haelfte {first_half:.4f}, zweite haelfte {second_half:.4f} Grad")

print()
print("9. Schub blockiert den hauptthread nicht")

# Schub reisst die geschwindigkeit in JEDEM frame ueber die toleranz. Der alte
# weg hat daraufhin jedes mal _compute_full SYNCHRON laufen lassen: gemessen
# mit dem vollen sonnensystem 0.12 ms im gleitflug gegen 59 ms unter schub --
# ~14 fps, solange die pfeiltaste gedrueckt ist. Zugleich wurde die laufende
# hintergrundrechnung verworfen und die linie geleert, also kam unter
# dauerschub NIE ein ergebnis durch (dasselbe muster wie bei reset(), §7).
#
# Geprueft werden die drei eigenschaften, nicht der weg dorthin: der
# hauptthread bleibt frei, die linie bleibt stehen, und die vorhersage wird
# waehrend des brennens tatsaechlich erneuert.

w, ship, p = build(async_compute=True)
p.set_view_scale(2e-9)
FRAME = 1.0 / 180.0
for _ in range(40):
    advance(w, 3.0)
    p.update(ship, w)
    time.sleep(0.005)


def _measure(frames, thrust):
    times = []
    counts = []
    swaps0 = int(p._jobs_swapped)
    for _ in range(frames):
        advance(w, 3.0)
        if thrust:
            # schiff.apply_thrust: 600 m/s^2 ueber einen echten frame
            ship.velocity.y += 600.0 * FRAME
        t0 = time.perf_counter()
        p.update(ship, w)
        times.append((time.perf_counter() - t0) * 1000.0)
        counts.append(len(p.get_points()))
        time.sleep(FRAME)
    return np.array(times), counts, int(p._jobs_swapped) - swaps0


# Vergleichslinie: derselbe verlauf, aber SYNCHRON gerechnet -- also genau
# die linie, die der alte weg in jedem frame erzwungen hat.
_, _, ref = build(async_compute=False)
ref.set_view_scale(2e-9)


def _deviation():
    ref.reset()
    ref._compute_full(ship, w)
    ref._anchor_first_point(ship, w)
    a = np.asarray(p.get_points())
    b = np.asarray(ref.get_points())
    n = min(len(a), len(b))
    if n < 10:
        return float('inf')
    return float(np.hypot(a[:n, 0] - b[:n, 0], a[:n, 1] - b[:n, 1]).max())


coast_ms, coast_counts, _ = _measure(90, thrust=False)

# Grundrauschen VOR dem brennen: der hintergrund-rechner ist auch im
# gleitflug nie exakt, sein ergebnis ist immer eine rechenzeit alt. Gegen
# diese schranke wird nach brennschluss gemessen -- eine bit-genaue forderung
# waere fuer einen asynchronen weg unerfuellbar.
coast_dev = _deviation()

versions0 = int(p._trajectory_version)
thrust_ms, thrust_counts, thrust_swaps = _measure(180, thrust=True)

# Ein voller neuaufbau kostet hier ~60 ms; ein 180-fps-frame hat 5.6 ms
# budget INSGESAMT. Die schranke ist bewusst als bruchteil eines frames
# formuliert und nicht als "schneller als vorher".
check(np.median(thrust_ms) < 5.0,
      "update() bleibt unter schub im frame-budget",
      f"median {np.median(thrust_ms):.2f} ms, p90 {np.percentile(thrust_ms, 90):.2f} ms "
      f"(gleitflug {np.median(coast_ms):.2f} ms)")

# Die kurve ATMET jetzt um ein, zwei punkte: sie wird vorn verbraucht (die
# stuetzstellen, deren zeit vergangen ist) und bekommt dafuer das schiff als
# kopf vorangestellt, statt starr mitgezogen zu werden. Geprueft wird deshalb,
# dass sie nicht EINBRICHT -- das war der fehler --, und nicht, dass die zahl
# auf den punkt genau stehen bleibt.
check(min(thrust_counts) >= max(coast_counts) - 2,
      "die linie wird unter schub nicht geleert",
      f"kleinste punktzahl unter schub {min(thrust_counts)} von {max(coast_counts)}")

check(thrust_swaps >= 5,
      "die vorhersage wird waehrend des brennens erneuert",
      f"{thrust_swaps} ergebnisse in {180 * FRAME:.2f} s eingewechselt")

check(int(p._trajectory_version) == versions0,
      "kein trajektorien-neustart je schub-frame",
      f"version {versions0} -> {int(p._trajectory_version)}")

# Waehrend des brennens haengt die gezeichnete linie um eine rechenzeit
# hinterher. Nach brennschluss MUSS dieser abstand wieder auf das
# gleitflug-rauschen zurueckgehen -- sonst waere die vorhersage dauerhaft
# falsch, und der gewonnene bildratenvorteil waere erkauft.
burn_dev = _deviation()
check(burn_dev > coast_dev * 3.0,
      "unter schub haengt die linie messbar hinterher (die eingegangene wette)",
      f"{burn_dev:.3e} m gegen {coast_dev:.3e} m gleitflug-rauschen")

dev_now = burn_dev
settled = -1
for f in range(30):
    advance(w, 3.0)
    p.update(ship, w)
    time.sleep(FRAME)
    dev_now = _deviation()
    if dev_now <= coast_dev * 3.0:
        settled = f
        break

check(0 <= settled <= 10,
      "nach brennschluss rastet die linie wieder ein",
      (f"nach {settled} frames: {burn_dev:.3e} m unter schub -> {dev_now:.3e} m "
       f"(gleitflug-rauschen {coast_dev:.3e} m)"
       if settled >= 0 else f"nicht eingerastet (zuletzt {dev_now:.3e} m)"))

print()
print("10. Der koerper-notizblock rechnet BIT-IDENTISCH")

# 96 % der rechenzeit einer vorhersage steckt im aufstellen der 28 koerper
# (gemessen: 61.7 ms gegen 0.6 ms mit eingefrorenen koerpern). Der notizblock
# in _body_position_at_time_numba entfernt daraus die reine wiederholung --
# gleiche zeit, gleicher koerper, schon gerechnet -- und der kernel-vorlauf
# hebt die zeitunabhaengigen bahngroessen heraus. Beides darf am ergebnis
# NICHTS aendern; `Predictor.use_body_memo = False` rechnet wie zuvor.
#
# Der schalter hat schon einen echten fehler gefangen: mit `NaN` als
# "noch nichts gerechnet"-merker war der vergleich unter `fastmath=True`
# (das LLVMs nnan einschaltet) wirkungslos, jeder koerper stand im ursprung
# und die bahn wich um 1.8e6 m ab -- ohne diese pruefung waere das nur als
# "irgendwie andere linie" aufgefallen.

def _line(memo, view_scale, ref_name=None, thrust=0.0):
    w_, ship_, p_ = build(async_compute=False)
    p_.use_body_memo = memo
    p_.set_view_scale(view_scale)
    if ref_name is not None:
        for i, b in enumerate(w_.body):
            if getattr(b, 'name', None) == ref_name:
                p_.reference_body_index = i
                p_.reference_body_enabled = True
    if thrust:
        ship_.velocity.y += thrust
    t0 = time.perf_counter()
    p_._compute_full(ship_, w_)
    return np.asarray(p_.get_points()).copy(), (time.perf_counter() - t0) * 1000.0


_slow_ms = 0.0
_fast_ms = 0.0
for _label, _kw in (
    ("weit gezoomt", dict(view_scale=2e-9)),
    ("nah gezoomt", dict(view_scale=2e-4)),
    ("bezug Erde", dict(view_scale=2e-9, ref_name="Erde")),
    ("bezug Titan (mond, zweigliedrige kette)", dict(view_scale=2e-9, ref_name="Titan")),
    ("nach schub", dict(view_scale=2e-9, thrust=900.0)),
):
    _a, _ta = _line(False, **_kw)
    _b, _tb = _line(True, **_kw)
    _slow_ms += _ta
    _fast_ms += _tb
    _same = _a.shape == _b.shape and np.array_equal(_a, _b, equal_nan=True)
    _diff = 0.0 if _same else (
        float(np.nanmax(np.abs(_a - _b))) if _a.shape == _b.shape else float('nan'))
    check(_same, f"notizblock bit-identisch -- {_label}",
          f"{_a.shape[0]} punkte, groesste abweichung {_diff:.3e}")

check(_fast_ms < _slow_ms * 0.75,
      "und er ist deutlich schneller",
      f"{_slow_ms:.0f} ms -> {_fast_ms:.0f} ms ({_slow_ms / max(_fast_ms, 1e-9):.1f}x)")

print()
print("11. Ap/Pe-marker sitzen auf den analytisch bekannten radien")

# Diese pruefung existiert, weil die marker schon einmal SPURLOS verschwanden:
# _find_apsis_markers_numba liess sich nicht mehr uebersetzen (ein globales
# numpy-array ist fuer numba `readonly`, siehe predictor._no_body_memo), die
# ausnahme wurde gefangen, und get_apsis_markers() lieferte still null marker.
# Im spiel fehlten die rauten und die HUD-zahlen, sonst nichts.
#
# Gemessen wird deshalb gegen eine bahn, deren Pe und Ap man ausrechnen kann:
# eine ellipse mit e = 0.5 um die Erde.

w, ship, p = build(async_compute=False)
w.update_planets(0.0)
_erde = next(b for b in w.body if getattr(b, 'name', None) == 'Erde')
_ei = [i for i, b in enumerate(w.body) if getattr(b, 'name', None) == 'Erde'][0]

_mu = G * _erde.mass
_rp = 8.0e6
_ecc = 0.5
_ra = _rp * (1.0 + _ecc) / (1.0 - _ecc)
_vp = math.sqrt(_mu * (1.0 + _ecc) / _rp)
_period = 2.0 * math.pi * math.sqrt((_rp / (1.0 - _ecc)) ** 3 / _mu)

# Die Erde hat kein gueltiges velocity-feld (CLAUDE.md) -- zentral
# differenzieren, sonst sitzt das schiff in einer voellig anderen bahn.
_h = 1.0
_pa = _erde.position_at_time(w.time - _h)
_pb = _erde.position_at_time(w.time + _h)
_evx = (_pb.x - _pa.x) / (2.0 * _h)
_evy = (_pb.y - _pa.y) / (2.0 * _h)
ship.position.x = _erde.position.x + _rp
ship.position.y = _erde.position.y
ship.velocity.x = _evx
ship.velocity.y = _evy + _vp

p.reference_body_index = _ei
p.reference_body_enabled = True
# Horizont ueber gut zwei umlaeufe. Die bogenlaenge zaehlt BARYZENTRISCH,
# und dort dominiert die 30 km/s der Erde -- ein umlauf sind also rund
# 6e8 m bahn, nicht 1e8.
p.set_precision(150000.0)
p.set_length(p.num_points * 150000.0)
p.set_view_scale(2e-5)
p._compute_full(ship, w)

_marks = np.asarray(p.get_apsis_markers())
check(_marks.shape[0] >= 2, "marker werden ueberhaupt gefunden",
      f"{_marks.shape[0]} marker ueber {p.get_points()[-1, 2] - p.get_points()[0, 2]:.0f} s "
      f"({_period:.0f} s umlaufzeit)")

if _marks.shape[0] >= 2:
    _pe = _marks[_marks[:, 3] == 0.0]
    _ap = _marks[_marks[:, 3] == 1.0]
    check(_pe.shape[0] >= 1 and _ap.shape[0] >= 1,
          "sowohl periapsis als auch apoapsis dabei",
          f"{_pe.shape[0]} Pe, {_ap.shape[0]} Ap")
    if _pe.shape[0] >= 1:
        _err = abs(float(_pe[:, 4].min()) - _rp) / _rp
        check(_err < 1e-3, "periapsis-radius stimmt",
              f"{float(_pe[:, 4].min()):.4e} m gegen {_rp:.4e} m ({_err * 100:.3f} %)")
    if _ap.shape[0] >= 1:
        _err = abs(float(_ap[:, 4].max()) - _ra) / _ra
        check(_err < 1e-3, "apoapsis-radius stimmt",
              f"{float(_ap[:, 4].max()):.4e} m gegen {_ra:.4e} m ({_err * 100:.3f} %)")

print()
print("12. Die schub-pipeline erneuert oefter und NIE rueckwaerts")

# Eine vorhersage dauert laenger als ein bild (~17 ms gegen ~7 ms), also kann
# eine einzeln gerechnete linie hoechstens jedes dritte bild neu sein -- das
# ist das ruckeln waehrend eines burns. Mehrere zeitversetzt gestartete laeufe
# heben den DURCHSATZ, ohne die dauer einer einzelnen rechnung anzuruehren.
#
# Der preis dafuer ist eine echte gefahr: laeufe koennen in beliebiger
# reihenfolge fertig werden, und ein aelteres ergebnis wuerde die linie
# ZURUECKSPRINGEN lassen -- optisch genau das zittern, das der zeitraffer-halt
# beseitigt hat. Deshalb wird hier nicht die (maschinenabhaengige) rate
# geprueft, sondern die ORDNUNG.


def _burn(depth, frames=360):
    w_, ship_, p_ = build(async_compute=True)
    p_.set_view_scale(2e-9)
    p_.thrust_pipeline_depth = depth
    for _ in range(40):
        advance(w_, 1.0)
        p_.update(ship_, w_)
        time.sleep(0.01)
    ids = []
    vys = []
    last = -1
    queue_max = 0
    swaps0 = int(p_._jobs_swapped)
    t0 = time.perf_counter()
    for _ in range(frames):
        advance(w_, 1.0)
        ship_.velocity.y += 600.0 * FRAME
        p_.update(ship_, w_)
        queue_max = max(queue_max, len(getattr(p_, '_pending_futures', [])))
        jid = int(getattr(p_, '_last_swapped_job_id', -1))
        if jid != last:
            ids.append(jid)
            snap = p_._last_swapped_snapshot
            vys.append(float(snap.get('ship_vy', 0.0)) if snap else float('nan'))
            last = jid
        time.sleep(FRAME)
    return {
        'rate': (int(p_._jobs_swapped) - swaps0) / (time.perf_counter() - t0),
        'ids': ids,
        'vys': np.array(vys),
        'queue_max': queue_max,
        'depth': depth,
    }


_one = _burn(1)
_deep = _burn(3)

check(all(b > a for a, b in zip(_deep['ids'], _deep['ids'][1:])),
      "eingewechselte auftraege sind streng aufsteigend",
      f"{len(_deep['ids'])} wechsel, kein rueckwaertssprung")

# Der schub erhoeht die geschwindigkeit monoton. Ein schnappschuss, dessen
# geschwindigkeit unter der zuletzt gezeichneten liegt, waere ein rueckschritt
# der VORHERSAGE selbst -- schaerfer als die reine auftrags-nummer.
check(bool(np.all(np.diff(_deep['vys']) > 0.0)),
      "jede neue linie gehoert zu einem NEUEREN schiffszustand",
      f"{_deep['vys'].shape[0]} schnappschuesse, kleinste zunahme "
      f"{float(np.min(np.diff(_deep['vys']))) if _deep['vys'].shape[0] > 1 else float('nan'):.3f} m/s")

check(_deep['queue_max'] <= 3,
      "die warteschlange laeuft nicht voll",
      f"hoechstens {_deep['queue_max']} gleichzeitige auftraege bei tiefe 3")

check(_deep['rate'] > _one['rate'] * 1.4,
      "tiefe 3 erneuert deutlich oefter als tiefe 1",
      f"{_one['rate']:.0f}/s -> {_deep['rate']:.0f}/s "
      f"(alle {1000.0 / max(_one['rate'], 1e-9):.0f} ms -> alle {1000.0 / max(_deep['rate'], 1e-9):.0f} ms)")

# Die tiefe ist nicht fest, sondern folgt rechenzeit/bildzeit -- eine feste
# zahl waere beim kurzen horizont verschwenderisch und beim langen zu klein.
# Geprueft wird die RECHENVORSCHRIFT, nicht ein zahlenwert: der doppelte
# horizont kostet doppelt so viel rechenzeit und braucht daher mehr laeufe.
_, _, _pd = build(async_compute=True)
_pd.thrust_pipeline_depth = 8
_pd._update_interval_ms = 11.0

_pd.last_compute_ms = 17.0
_shallow = _pd._target_pipeline_depth()
_pd.last_compute_ms = 74.0
_steep = _pd._target_pipeline_depth()

check(_shallow < _steep,
      "die tiefe waechst mit der rechenzeit",
      f"17 ms -> tiefe {_shallow}, 74 ms -> tiefe {_steep} (bei 11 ms bildzeit)")

check(_shallow * _pd._update_interval_ms >= 17.0,
      "die gewaehlte tiefe reicht fuer ein ergebnis je bild",
      f"{_shallow} laeufe x 11 ms = {_shallow * 11.0:.0f} ms >= 17 ms rechenzeit")

_pd.thrust_pipeline_depth = 1
check(_pd._target_pipeline_depth() == 1,
      "tiefe 1 in der konfiguration schaltet die pipeline wirklich ab",
      "genau ein lauf, wie vor der aenderung")

print()
print("13. Schub wird auch NAHE DER PERIAPSIS erkannt")

# Die schub-erkennung verglich frueher den gesamten geschwindigkeitssprung
# gegen eine schranke von der groesse des SCHWERKRAFT-anteils
# (4 * |g| * dt). Nahe der periapsis ist das 65 m/s gegen 6.7 m/s vollschub
# je bild -- der schub verschwand vollstaendig darunter, es wurde keine
# neuberechnung mehr angefordert, und die linie fiel auf den langsamen
# einzelbetrieb zurueck (gemessen bei vierfachem horizont: 66 -> 18
# erneuerungen je sekunde, sobald das schiff in die naehe der periapsis kam).
# Genau dieser wechsel ist das ruckartige nachziehen.
#
# Geprueft wird die trennschaerfe selbst: im gleitflug darf nichts feuern
# (sonst rechnet der predictor dauernd umsonst), unter schub muss es feuern --
# an JEDER stelle der bahn.

_wp, _shipp, _pp = build(async_compute=True)
_wp.update_planets(0.0)
_erde_p = next(b for b in _wp.body if getattr(b, 'name', None) == 'Erde')
_ei_p = [i for i, b in enumerate(_wp.body) if getattr(b, 'name', None) == 'Erde'][0]
_pp.reference_body_index = _ei_p
_pp.reference_body_enabled = True
_pp.set_view_scale(2e-5)

_mu_p = G * _erde_p.mass
_h = 1.0
_pa = _erde_p.position_at_time(_wp.time - _h)
_pb = _erde_p.position_at_time(_wp.time + _h)
_evx = (_pb.x - _pa.x) / (2.0 * _h)
_evy = (_pb.y - _pa.y) / (2.0 * _h)

_SIM_STEP = 2.0            # sim-sekunden je bild, wie test.py bei 90 fps
_THRUST_DV = 600.0 / 90.0  # vollschub je bild


def _place_on_ellipse(nu, rp=7.0e6, ecc=0.7):
    a = rp / (1.0 - ecc)
    r = a * (1.0 - ecc * ecc) / (1.0 + ecc * math.cos(nu))
    k = math.sqrt(_mu_p / (a * (1.0 - ecc * ecc)))
    vr = k * ecc * math.sin(nu)
    vt = k * (1.0 + ecc * math.cos(nu))
    c, s = math.cos(nu), math.sin(nu)
    _shipp.position.x = _erde_p.position.x + r * c
    _shipp.position.y = _erde_p.position.y + r * s
    _shipp.velocity.x = _evx + vr * c - vt * s
    _shipp.velocity.y = _evy + vr * s + vt * c
    return r


def _fires(nu, thrust, frames=12):
    """Wie oft fordert die schub-erkennung in `frames` bildern eine
    neuberechnung an? `_pipeline_depth_used` wird ausschliesslich von
    _request_thrust_recompute gesetzt und ist deshalb der direkte anzeiger."""
    _place_on_ellipse(nu)
    for _ in range(3):
        advance(_wp, _SIM_STEP)
        _pp.update(_shipp, _wp)
    hits = 0
    for _ in range(frames):
        advance(_wp, _SIM_STEP)
        if thrust:
            _shipp.velocity.y += _THRUST_DV
        _pp._pipeline_depth_used = 0
        _pp.update(_shipp, _wp)
        if int(_pp._pipeline_depth_used) > 0:
            hits += 1
    return hits


for _deg in (0, 30, 90, 180):
    _nu = math.radians(_deg)
    _r = _place_on_ellipse(_nu)
    _coast = _fires(_nu, thrust=False)
    _burn = _fires(_nu, thrust=True)
    check(_burn >= 11 and _coast <= 1,
          f"bahnpunkt {_deg}deg (r = {_r / 1e6:.1f} Mm): schub erkannt, gleitflug nicht",
          f"schub {_burn}/12 bilder, gleitflug {_coast}/12")

# Und die schranke muss im ZEITRAFFER weiter dichthalten: ueber einen
# 28-stunden-schritt erklaert `g * dt` die aenderung nicht mehr, der rest
# waechst auf ~28 000 m/s -- die schranke waechst mit der kruemmung von g
# aber auf ~3.4e6 m/s mit. Ohne das wuerde die gehaltene kurve jeden frame
# zerrissen.
_place_on_ellipse(0.0)
for _ in range(3):
    advance(_wp, _SIM_STEP)
    _pp.update(_shipp, _wp)
_warp_hits = 0
for _ in range(6):
    advance(_wp, 100800.0)          # 7 d/s bei 60 fps
    _pp._pipeline_depth_used = 0
    _pp.update(_shipp, _wp)
    if int(_pp._pipeline_depth_used) > 0:
        _warp_hits += 1
check(_warp_hits == 0,
      "im zeitraffer feuert die erkennung NICHT (der halt bleibt heil)",
      f"{_warp_hits}/6 bilder bei 100800 s je schritt")

print()
print("14. Die linie zieht GLEICHMAESSIG nach, nicht in doppelschritten")

# Die laeufe der schub-pipeline werden gleichmaessig gestartet, werden aber
# nicht ganz gleichmaessig fertig (schwankende rechenzeit). Wer immer sofort
# das NEUESTE ergebnis nimmt, macht daraus stillstand-plus-doppelschritt: die
# kurvenform springt in einem bild doppelt so weit wie in seinen nachbarn.
# Rund 1.5 % der bilder waren betroffen -- etwa eines je sekunde. Eine
# gleichmaessig niedrige rate sieht man nicht, so einen ausreisser schon
# ("wie hohe netzwerk-latenz").
#
# Geprueft wird deshalb die GLEICHMAESSIGKEIT des fortschritts, nicht seine
# hoehe: aufeinanderfolgende einwechslungen duerfen keine nummern
# ueberspringen, solange der puffer nicht ueberlaeuft.

_wj, _shipj, _pj = build(async_compute=True)
_wj.update_planets(0.0)
_erde_j = next(b for b in _wj.body if getattr(b, 'name', None) == 'Erde')
_pj.reference_body_index = [i for i, b in enumerate(_wj.body)
                            if getattr(b, 'name', None) == 'Erde'][0]
_pj.reference_body_enabled = True
_pj.set_view_scale(2e-5)

_mu_j = G * _erde_j.mass
_pa = _erde_j.position_at_time(_wj.time - 1.0)
_pb = _erde_j.position_at_time(_wj.time + 1.0)
_evxj = (_pb.x - _pa.x) / 2.0
_evyj = (_pb.y - _pa.y) / 2.0


def _at_periapsis():
    rp = 7.0e6
    vp = math.sqrt(_mu_j * 1.7 / rp)
    _shipj.position.x = _erde_j.position.x + rp
    _shipj.position.y = _erde_j.position.y
    _shipj.velocity.x = _evxj
    _shipj.velocity.y = _evyj + vp


def _advance_pattern(backlog, frames=240):
    _pj.swap_backlog_max = backlog
    _at_periapsis()
    for _ in range(25):
        advance(_wj, 2.0)
        _pj.update(_shipj, _wj)
        time.sleep(0.02)
    _at_periapsis()
    last = int(_pj._last_swapped_job_id)
    steps = []
    for _ in range(frames):
        advance(_wj, 2.0)
        _shipj.velocity.y += 600.0 / 90.0
        _pj.update(_shipj, _wj)
        jid = int(_pj._last_swapped_job_id)
        steps.append(jid - last)
        last = jid
        time.sleep(1.0 / 90.0)
    return np.array(steps)


_paced = _advance_pattern(1)
_doubles = int(np.sum(_paced >= 2))
check(_doubles <= 2,
      "kaum doppelschritte beim einwechseln (puffer 1)",
      f"{_doubles} von {_paced.shape[0]} bildern springen um mehr als einen auftrag")

_smooth = _advance_pattern(2)
check(int(np.sum(_smooth >= 2)) <= _doubles,
      "ein groesserer puffer glaettet weiter",
      f"puffer 2: {int(np.sum(_smooth >= 2))} doppelschritte gegen {_doubles} bei puffer 1")

# Der puffer darf die linie NICHT anhalten: es muss weiterhin in der grossen
# mehrheit der bilder vorangehen.
check(float(np.mean(_paced >= 1)) > 0.9,
      "der puffer haelt die linie nicht auf",
      f"{100.0 * float(np.mean(_paced >= 1)):.0f} % der bilder gehen voran")

# Und die reihenfolge bleibt streng: kein ergebnis darf zurueckspringen.
check(bool(np.all(_paced >= 0)) and bool(np.all(_smooth >= 0)),
      "auch mit puffer springt nie ein aelteres ergebnis ein",
      "alle schritte >= 0")

print()
print("15. Die schrittweiten-deckelung haengt NICHT davon ab, WO auf der bahn")
print("    das schiff gerade steht")

# Der ferne teil der bahn wird mit einer gedeckelten schrittweite integriert,
# und dieser deckel kommt aus der frage "wieviel ZEIT deckt der horizont ab?".
# Wird die aus der MOMENTANGESCHWINDIGKEIT geschaetzt, ist sie im perihel
# (schnellster punkt) zu kurz und im aphel (langsamster punkt) zu lang -- auf
# derselben bahn, mit demselben bogen. Der lauf kostet dann im perihel ein
# vielfaches, die linie faellt dort unter die bildrate und stockt sichtbar,
# waehrend im aphel nichts davon zu merken ist. Genau diese asymmetrie wird
# hier gemessen.

_wf, _shipf, _pf = build(async_compute=False)
_sonne_f = next(b for b in _wf.body if b.name.lower().startswith('sonne'))
_mu_f = _wf.G * _sonne_f.mass
_RPE, _RAP = 29.16e9, 128.8e9
_af = 0.5 * (_RPE + _RAP)
_HORIZONT = 3.2e11                      # entspricht ~5x '+' -- die ganze ellipse


def _kosten(r, laeufe=3):
    """schritte + zeitspanne eines eingeschwungenen laufs bei radius r."""
    v = math.sqrt(_mu_f * (2.0 / r - 1.0 / _af))
    _pf.reset()
    _pf.set_length(_HORIZONT)
    _pf.set_precision(_HORIZONT / _pf.num_points)
    _shipf.position.x = _sonne_f.position.x + r
    _shipf.position.y = _sonne_f.position.y
    _shipf.velocity.x = 0.0
    _shipf.velocity.y = v
    for _ in range(laeufe):
        _pf._compute_full(_shipf, _wf)
    pts = _pf.get_points()
    spanne = float(pts[-1, 2] - pts[0, 2])
    return int(_pf.rkn_last_accepted_steps), spanne, float(_pf.rkn_last_max_dt)


_st_pe, _sp_pe, _md_pe = _kosten(_RPE)
_st_ap, _sp_ap, _md_ap = _kosten(_RAP)
_verhaeltnis = max(_st_pe, _st_ap) / max(1.0, min(_st_pe, _st_ap))
check(_verhaeltnis < 1.6,
      "perihel und aphel kosten gleich viele integrationsschritte",
      f"perihel {_st_pe}, aphel {_st_ap} -- faktor {_verhaeltnis:.2f}")

# ... und zwar beide in der naehe des vorgegebenen budgets, nicht irgendwo.
_ziel = float(_pf.rkn_far_field_target_steps)
check(0.6 * _ziel <= _st_pe <= 1.8 * _ziel and 0.6 * _ziel <= _st_ap <= 1.8 * _ziel,
      "beide treffen das schrittbudget",
      f"budget {_ziel:.0f}, perihel {_st_pe}, aphel {_st_ap}")

# Die gemessene groesse muss die ECHTE zeitspanne sein, nicht bogen/v_jetzt.
_gemessen = float(_pf._horizon_time_per_arc)
check(abs(_gemessen * _HORIZONT - _sp_ap) / _sp_ap < 0.05,
      "die zurueckgemeldete groesse ist die echte zeitspanne des horizonts",
      f"gemeldet {_gemessen * _HORIZONT / 86400.0:.2f} d, gemessen "
      f"{_sp_ap / 86400.0:.2f} d")

# Ein reset() muss sie loeschen -- nach einem teleport/reparenting waere sie
# schlicht falsch, und der erste lauf danach faellt bewusst auf den alten,
# vorsichtigeren schaetzer zurueck.
_pf.reset()
check(float(_pf._horizon_time_per_arc) == 0.0,
      "reset() loescht die gemessene zeitspanne",
      f"nach reset: {_pf._horizon_time_per_arc}")

# Kurze horizonte bleiben UNBERUEHRT: dort greift der feste boden rkn_max_dt,
# die rueckmeldung darf daran nichts aendern.
_pf.reset()
_pf.set_length(_pf.num_points * 1.0e6)
_pf.set_precision(1.0e6)
_shipf.position.x = _sonne_f.position.x + _RPE
_shipf.position.y = _sonne_f.position.y
_shipf.velocity.x = 0.0
_shipf.velocity.y = math.sqrt(_mu_f * (2.0 / _RPE - 1.0 / _af))
_pf._compute_full(_shipf, _wf)
_erste = _pf.get_points().copy()
_pf._compute_full(_shipf, _wf)
_zweite = _pf.get_points().copy()
check(_erste.shape == _zweite.shape and bool(np.array_equal(_erste, _zweite)),
      "beim standard-horizont aendert die rueckmeldung nichts (bit-identisch)",
      f"max abweichung {float(np.abs(_erste - _zweite).max()):.3e}")

# ═══════════════════════════ 16. schrittweiten-decke und bahn-riegel

print()
print("16. Die schrittweiten-decke traegt den zeitraffer, ohne die bahn zu")
print("    verlieren")

_w16, _ship16, _p16 = build()
_p16.close()

# (a) In echtzeit bleibt die decke EXAKT die konfigurierte -- nur so rechnet
#     der integrator dort dieselben floats wie vor der aenderung
#     (tests/energy_test.py haengt daran).
_base16 = float(_w16.integrator_max_step)
_w16.set_warp_step_ceiling(60.0 / 180.0)
check(_w16.effective_max_step() == _base16,
      "in echtzeit bleibt die decke unveraendert",
      f"{_w16.effective_max_step()} == {_base16}")

# (b) Im zeitraffer waechst sie proportional zur sim-zeit je frame und zielt
#     auf integrator_warp_substep_target teilschritte.
_frame16 = 31557600.0 / 180.0            # ein frame bei 1 y/s
_ceil16 = _w16.set_warp_step_ceiling(_frame16)
_target16 = float(_w16.integrator_warp_substep_target)
check(abs(_ceil16 - _frame16 / _target16) <= 1e-6 * _frame16,
      "im zeitraffer zielt die decke auf die gewuenschte schrittzahl",
      f"decke {_ceil16:.1f} s = {_frame16:.0f} / {_target16:.0f}")

# (c) Und sie macht den schritt wirklich billiger.
def _cost16(world_obj, ship_obj, sim_seconds, use_ceiling):
    if use_ceiling:
        chunk = max(MAX_SUBSTEP, world_obj.set_warp_step_ceiling(sim_seconds))
    else:
        world_obj.integrator_max_step_effective = 0.0
        chunk = MAX_SUBSTEP
    steps = max(1, int(math.ceil(sim_seconds / chunk)))
    dt = sim_seconds / steps
    total = 0
    for _ in range(steps):
        world_obj.update_dynamics(dt)
        world_obj.update_planets(dt)
        total += world_obj.integrator_last_substeps
    return total

_wa, _sa, _pa = build(); _pa.close()
_wa.update_planets(0.0)
_subs_off = _cost16(_wa, _sa, _frame16, False)
_wb, _sb, _pb = build(); _pb.close()
_wb.update_planets(0.0)
_subs_on = _cost16(_wb, _sb, _frame16, True)
check(_subs_on * 20 < _subs_off,
      "die decke senkt die teilschritte bei 1 y/s um mehr als das 20-fache",
      f"ohne decke {_subs_off}, mit decke {_subs_on} "
      f"(faktor {_subs_off / max(1, _subs_on):.0f})")

# (d) NAHE AN EINEM KOERPER darf die decke nichts ausrichten -- dort haelt die
#     fehlerkontrolle die zuegel. Das ist die eigentliche sicherheitsaussage:
#     ohne sie wuerde eine angehobene decke tiefe bahnen zerlegen.
def _leo_world(max_step):
    wl, shipl, pl = build()
    pl.close()
    wl.update_planets(0.0)          # KOERPER AUF IHRE BAHN SETZEN -- ohne das
                                    # steht Erde noch im ursprung und das
                                    # schiff umkreist nichts.
    erde = next(b for b in wl.body if b.name == 'Erde')
    h = 1.0
    p1 = erde.position_at_time(wl.time + h)
    p0 = erde.position_at_time(wl.time - h)
    evx = (p1.x - p0.x) / (2.0 * h)
    evy = (p1.y - p0.y) / (2.0 * h)
    R = erde.radius + 400e3
    shipl.position.x = erde.position.x + R
    shipl.position.y = erde.position.y
    shipl.velocity.x = evx
    shipl.velocity.y = evy + math.sqrt(G * erde.mass / R)
    wl.integrator_max_step_effective = float(max_step)
    T = 2.0 * math.pi * math.sqrt(R ** 3 / (G * erde.mass))
    return wl, shipl, erde, R, T

_eff = {}
for _ms in (30.0, 100000.0):
    _wl, _shipl, _erde_l, _Rl, _Tl = _leo_world(_ms)
    _total = 5.0 * _Tl
    _n = 200
    _subs = 0
    for _ in range(_n):
        _wl.integrator_max_step_effective = float(_ms)
        _wl.update_dynamics(_total / _n)
        _wl.update_planets(_total / _n)
        _subs += _wl.integrator_last_substeps
    _alt = ((_shipl.position - _erde_l.position).magnitude() - _erde_l.radius) / 1e3
    _eff[_ms] = (_total / max(1, _subs), _alt)

_step_low, _alt_low = _eff[30.0]
_step_high, _alt_high = _eff[100000.0]
# Die schranke war 2.0, solange RKN4 faelschlich 3. ordnung war (27.7 -> 34.7 s).
# Seit dem koeffizienten-fix (world.py:226 / world_kernels.py:147) erlaubt
# dieselbe 1-m-toleranz groessere schritte, gemessen 27.7 -> 69.3 s. Die aussage
# bleibt dieselbe: die schrittweite setzt die fehlerkontrolle, nicht die decke --
# 69.3 s liegen 1443x unter den 100 000 s. Die eigentliche sicherheitsaussage
# steht im naechsten check (die bahn darf nicht wandern).
check(_step_high < _step_low * 3.0 and _step_high < 0.01 * 100000.0,
      "im 400-km-orbit aendert eine 3300x hoehere decke die schrittweite kaum",
      f"{_step_low:.1f} s -> {_step_high:.1f} s "
      f"(decke 100000 s, also faktor {100000.0 / _step_high:.0f} darunter)")
check(abs(_alt_high - 400.0) < 5.0 and abs(_alt_low - 400.0) < 5.0,
      "und die bahn bleibt in beiden faellen stehen",
      f"hoehe nach 5 umlaeufen: {_alt_low:.3f} km / {_alt_high:.3f} km")

# (e) Der bahn-riegel: characteristic_timescale ist T/2pi.
_wt, _shipt, _erde_t, _Rt, _Tt = _leo_world(30.0)
_tchar = _wt.characteristic_timescale(_shipt)
check(abs(_tchar - _Tt / (2.0 * math.pi)) < 0.01 * _Tt / (2.0 * math.pi),
      "characteristic_timescale liefert T/2pi",
      f"gemessen {_tchar:.1f} s, erwartet {_Tt / (2.0 * math.pi):.1f} s")

# Und im fernfeld ist sie so gross, dass sie nichts sperrt.
_wf16, _shipf16, _pf16 = build(); _pf16.close()
_wf16.update_planets(0.0)
_shipf16.position.x = 1.496e11
_shipf16.position.y = 0.0
_tchar_far = _wf16.characteristic_timescale(_shipf16)
check(_tchar_far / 3.0 * 180.0 > 31557600.0,
      "im heliozentrischen fernfeld sperrt der riegel 1 y/s NICHT",
      f"grenze {_tchar_far / 3.0 * 180.0:.3e} s/s gegen 3.156e+07 s/s")
check(_tchar / 3.0 * 180.0 < 86400.0,
      "im 400-km-orbit sperrt er dagegen alles ab 1 d/s",
      f"grenze {_tchar / 3.0 * 180.0:.3e} s/s")



# ══════════════════════════ 17. stufenwechsel blockiert den hauptthread nicht

print()
print("17. Der wechsel der zeitraffer-stufe haelt den hauptthread nicht an")

# WARUM. Die stufe bestimmt ueber predictor_warp_length_mult() den horizont
# (1x/4x/16x/64x ab 7d/s). Jeder wechsel ruft set_length(), das den halt
# entwertet -- und der halt-zweig in update() beantwortete das mit einem
# SYNCHRONEN _compute_full im hauptthread. Gemessen mit dem vollen
# sonnensystem bei 180 fps: 44 ms auf 7d/s->30d/s, 34 ms auf 30d/s->100d/s,
# 45 ms auf 100d/s->1y/s und 75-82 ms auf dem rueckweg -- gegen 0.4 ms in den
# nachbar-frames. Das ist der ruckler beim umschalten.
#
# Richtig ist dieselbe antwort wie beim schub (siehe _request_thrust_recompute):
# die neue laenge wird ANGEFORDERT, nicht erzwungen. Die alte kurve ist
# geometrisch weiterhin gueltig -- sie ist nur zu kurz bzw. zu lang -- also
# wird sie weiter gehalten, bis das asynchrone ergebnis da ist.
#
# Gemessen wird deshalb dreierlei, und alles drei ist eine GROESSE, keine
# implementierung: (a) die zeit im hauptthread auf dem wechsel-frame,
# (b) dass die linie waehrend des wechsels nie leer wird, (c) dass die neue
# laenge auch wirklich ankommt (der wechsel darf nicht verschluckt werden).

_W17_TICK = 180.0
_W17_FRAME_DT = 1.0 / _W17_TICK
_W17_BASE = None
# Punktbudget-parameter wie in test.py: der punktabstand des grundzustands
# und die decke aus config.json.
_W17_SPACING = None
_W17_MAX_POINTS = 40000


def _w17_mult(rate):
    """predictor_warp_length_mult() aus test.py."""
    ratio = rate / 604800.0
    if ratio <= 1.0:
        return 1.0
    return float(1 << min(6, max(0, int(round(math.log2(ratio))))))


def _w17_step(w, sim_seconds):
    """step_simulation() aus test.py."""
    if sim_seconds <= 0.0:
        return
    ceiling = w.set_warp_step_ceiling(sim_seconds)
    chunk = max(MAX_SUBSTEP, ceiling)
    steps = max(1, int(math.ceil(sim_seconds / chunk)))
    sub = sim_seconds / steps
    for _ in range(steps):
        # reihenfolge wie test.py::update -- siehe advance()
        w.update_dynamics(sub)
        w.update_planets(sub)


def _w17_frame(w, ship, p, rate):
    """Ein frame der hauptschleife -- nur der zeitraffer-relevante teil."""
    drawn = _W17_BASE
    wanted = drawn * _w17_mult(rate)
    changed = False
    if hasattr(p, 'set_display_length'):
        p.set_display_length(drawn if wanted > drawn else None)
    # Das punktbudget waechst mit dem horizont mit (apply_predictor_horizon):
    # auch das muss WEICH gehen, sonst kostet jeder stufenwechsel wieder
    # einen synchronen neuaufbau im hauptthread.
    _budget = int(min(_W17_MAX_POINTS,
                      max(1, math.ceil(wanted / _W17_SPACING))))
    if _budget != int(p.num_points):
        p.set_num_points(_budget, soft=True)
        changed = True
    if p.length is None or abs(p.length - wanted) > wanted * 1e-9:
        p.set_length(wanted)
        changed = True
    _w17_step(w, rate * _W17_FRAME_DT)
    p.set_hold(rate > 60.0 * 1.001)
    p.set_view_scale(2e-9)
    t0 = time.perf_counter()
    p.update(ship, w)
    return (time.perf_counter() - t0) * 1000.0, changed


def _w17_transition(w, ship, p, from_rate, to_rate, settle=25, after=140):
    """Auf from_rate einschwingen, dann auf to_rate wechseln.

    Rueckgabe: (ms auf dem wechsel-frame, ruhe-median davor, kleinste
    punktzahl danach, frames bis die neue laenge gezeichnet wird).
    """
    quiet = []
    for _ in range(settle):
        ms, _c = _w17_frame(w, ship, p, from_rate)
        quiet.append(ms)
    quiet_median = sorted(quiet)[len(quiet) // 2]

    switch_ms = None
    min_points = None
    min_drawn = None
    arrived = None
    for i in range(after):
        ms, changed = _w17_frame(w, ship, p, to_rate)
        # Der teure frame ist der erste nach dem wechsel -- egal ob ihn der
        # horizont ausloest (set_length) oder der halt selbst (set_hold).
        if switch_ms is None:
            switch_ms = ms
        if switch_ms is not None:
            n = int(p.points.shape[0]) if p.points is not None else 0
            min_points = n if min_points is None else min(min_points, n)
            # Was der spieler SIEHT: die gezeichnete bogenlaenge, gemessen
            # gegen den un-geraften basis-horizont. Sie darf waehrend des
            # wechsels nicht einbrechen und zurueckspringen.
            drawn_arc = arc_length(p.get_points())
            frac = drawn_arc / _W17_BASE
            min_drawn = frac if min_drawn is None else min(min_drawn, frac)
            # Die neue laenge ist angekommen, sobald die gerechnete kurve den
            # neuen bogen wirklich ueberspannt (nicht nur die punktzahl).
            if arrived is None and p.points is not None and n >= 4:
                if arc_length(p.points) >= 0.9 * float(p.length):
                    arrived = i
    return switch_ms, quiet_median, min_points, min_drawn, arrived


# 10m/s -> 1h/s ist der schritt, bei dem der HALT anspringt (ueber
# realtime_warp_max). Der horizont aendert sich dort nicht -- gemessen wurden
# trotzdem 14.1 ms, weil set_hold() die kurve hart entwertete.
_UP = [(600.0, 3600.0, "10m/s -> 1h/s (halt springt an)"),
       (604800.0, 2592000.0, "7d/s -> 30d/s"),
       (2592000.0, 8640000.0, "30d/s -> 100d/s"),
       (8640000.0, 31557600.0, "100d/s -> 1y/s")]
_DOWN = [(31557600.0, 8640000.0, "1y/s -> 100d/s"),
         (8640000.0, 2592000.0, "100d/s -> 30d/s"),
         (2592000.0, 604800.0, "30d/s -> 7d/s")]

for _from, _to, _label in _UP + _DOWN:
    _w17, _ship17, _p17 = build(async_compute=True)
    _W17_BASE = float(_p17.length)
    _W17_SPACING = _W17_BASE / max(1, int(_p17.num_points))
    try:
        _sw, _quiet, _minpts, _mindrawn, _arrived = _w17_transition(_w17, _ship17, _p17, _from, _to)
        # (a) Der wechsel-frame darf nicht aus der reihe fallen. Die schranke
        #     ist grosszuegig (10 ms gegen 34-82 ms vorher), damit sie nicht
        #     auf die tagesform der maschine reagiert.
        check(_sw is not None and _sw < 10.0,
              f"{_label}: wechsel-frame bleibt im hauptthread billig",
              f"{_sw:.1f} ms (ruhe-median {_quiet:.1f} ms)")
        # (b) Die linie darf waehrend des wechsels weder verschwinden noch
        #     pulsieren. Ohne den festgehaltenen fortsetzungs-zustand
        #     (_hold_resume_context) schrumpfte sie auf 61 % und sprang beim
        #     einwechseln zurueck -- kein ruckler, aber auch nicht nahtlos.
        check(_minpts is not None and _minpts >= 4,
              f"{_label}: die linie bleibt waehrend des wechsels stehen",
              f"kleinste punktzahl {_minpts}")
        check(_mindrawn is not None and _mindrawn >= 0.95,
              f"{_label}: der gezeichnete horizont pulsiert nicht",
              f"kleinster gezeichneter anteil {_mindrawn * 100:.1f} %")
        # (c) ... und die neue laenge muss ankommen, nicht verschluckt werden.
        check(_arrived is not None,
              f"{_label}: die neue laenge kommt an",
              f"nach {_arrived} frames" if _arrived is not None
              else f"nie erreicht (laenge {_p17.length:.3e} m)")
    finally:
        _p17.close()


# ══════════════ 18. die welt haelt sich an dieselbe bahn wie die vorhersage

print()
print("18. Welt und vorhersage rechnen dieselbe bahn")

# ZWEI AUSSAGEN, UND DIE ERSTE IST EINE UEBER DIE REIHENFOLGE.
#
# `update_planets(dt)` schreibt `body.theta` um einen ganzen chunk vor und
# merkt sich `self.time` als epoche. Das stimmt genau dann, wenn
# `update_dynamics(dt)` VORHER lief und die uhr schon gestellt hat -- die
# reihenfolge aus `test.py::update`. Vertauscht liegt der bookmark einen
# chunk daneben und `position_at_time(tau)` liefert systematisch die
# position bei `tau + dt`: jeder geskriptete koerper steht fuer die
# kraftrechnung des schiffs einen chunk in der zukunft.
#
# Der fehler ist ERSTER ORDNUNG im chunk -- und `step_simulation()` rechnet
# in `max_substep_seconds` = 1000 s grossen chunks, im zeitraffer sogar
# `max(MAX_SUBSTEP, ceiling)`. Er faellt in echtzeit nicht auf, weil die
# linie jeden frame neu gerechnet und am schiff verankert wird; im
# zeitraffer steht sie still und das schiff rutscht an ihr entlang, dann
# sieht man ihn sofort.
#
# Geprueft wird die groesse selbst, nicht die umsetzung: eine konvergierte
# integration darf nicht davon abhaengen, wie fein man den weg zerlegt.

from vec import Vec2 as _Vec2


def _earth_orbit_scene(rp=2.0e7, ecc=0.3):
    """Schiff auf eine bekannte bahn um die Erde setzen."""
    config = ConfigLoader(None)
    config.load()
    bs = SystemLoader("solar_system.json").load()
    w = World(G)
    w.body = bs
    config.apply_to_world(w)
    ship = next(b for b in bs if b.is_ship)
    erde = next(b for b in bs if b.name == "Erde")
    erde_index = bs.index(erde)
    w.update_planets(1.0)
    q0 = erde.position_at_time(w.time)
    q1 = erde.position_at_time(w.time + 10.0)
    mu = G * erde.mass
    ship.position = _Vec2(erde.position.x + rp, erde.position.y)
    ship.velocity = _Vec2((q1.x - q0.x) / 10.0,
                          (q1.y - q0.y) / 10.0
                          + math.sqrt(mu * (1.0 + ecc) / rp))
    p = Predictor(recompute_every_update=True, **config.predictor_kwargs())
    config.apply_to_predictor(p)
    p.set_length(p.num_points * p.precision)
    p.async_compute = True
    p.set_reference_body_index(erde_index)
    period = 2.0 * math.pi * math.sqrt((rp / (1.0 - ecc)) ** 3 / mu)
    return w, ship, p, erde, period


def _hermite_at(points, t):
    """Die linie KUBISCH auswerten -- so zeichnet der renderer sie auch.

    Linear zwischen den stuetzstellen zu interpolieren misst die SEHNE, nicht
    die kurve: bei 1e6 m punktabstand und r = 2e7 m sind das schon
    c^2/8R = 6 km, und jede aussage hier ginge darin unter.
    """
    ts = points[:, 2]
    i = int(np.searchsorted(ts, t))
    if i <= 0 or i >= len(ts):
        return None
    t0 = float(ts[i - 1]); t1 = float(ts[i])
    h = t1 - t0
    if h <= 0.0:
        return None
    u = (t - t0) / h
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    return (h00 * points[i - 1, 0] + h10 * h * points[i - 1, 3]
            + h01 * points[i, 0] + h11 * h * points[i, 3],
            h00 * points[i - 1, 1] + h10 * h * points[i - 1, 4]
            + h01 * points[i, 1] + h11 * h * points[i, 4])


# (a) DIE ZERLEGUNG IN CHUNKS DARF DIE BAHN NICHT VERSCHIEBEN -- in der
#     reihenfolge des spiels. Wer die beiden aufrufe vertauscht, faellt hier
#     um drei groessenordnungen durch.
_TOTAL18 = 4800.0


def _walk18(chunk, dyn_first=True):
    w, ship, p, _erde, _T = _earth_orbit_scene()
    try:
        p.async_compute = False
        p.set_view_scale(1e-5)
        p.update(ship, w)
        line = np.array(p.points, dtype=np.float64)
    finally:
        p.close()
    steps = int(round(_TOTAL18 / chunk))
    dt = _TOTAL18 / steps
    for _ in range(steps):
        if dyn_first:
            w.update_dynamics(dt)
            w.update_planets(dt)
        else:
            w.update_planets(dt)
            w.update_dynamics(dt)
    q = _hermite_at(line, float(w.time))
    if q is None:
        return None
    return math.hypot(float(ship.position.x) - q[0],
                      float(ship.position.y) - q[1])


_d_game = {c: _walk18(c) for c in (1000.0, 300.0, 5.0)}
_d_swapped = _walk18(1000.0, dyn_first=False)

check(max(_d_game.values()) < 1.0e3,
      "das schiff bleibt auf der vorhergesagten linie, bei jeder chunk-groesse",
      "  ".join(f"{c:.0f}s: {d:.3e} m" for c, d in sorted(_d_game.items())))
check(abs(_d_game[1000.0] - _d_game[5.0]) < 1.0,
      "und die bahn haengt nicht davon ab, wie fein man sie zerlegt",
      f"1000 s gegen 5 s: {abs(_d_game[1000.0] - _d_game[5.0]):.3e} m")
# Gegenprobe: mit vertauschten aufrufen muss es deutlich schlechter werden,
# sonst prueft (a) gar nicht die reihenfolge.
check(_d_swapped > _d_game[1000.0] * 100.0,
      "vertauschte reihenfolge (planets vor dynamics) faellt durch",
      f"{_d_swapped:.3e} m gegen {_d_game[1000.0]:.3e} m")


# (b) UND SO SIEHT ES DER SPIELER: im zeitraffer, ueber mehrere umlaeufe.
#
# Im halt wird die kurve NICHT neu gerechnet -- das schiff rutscht an einer
# stehenden linie entlang, jede abweichung steht also unmittelbar auf dem
# schirm. `hold_drift_max_px` ist die schranke dagegen: wird der seitliche
# versatz groesser als ein halbes pixel, fordert der halt ASYNCHRON eine
# neue kurve an (siehe Predictor._hold_advance) statt weiter zu driften.

VIEW_SCALE_18 = 1e-5          # px/m -- eine erdumlaufbahn fuellt damit den schirm
WARP_18 = 3600.0              # 1 h/s
FRAMES_18 = 1200


def _lateral_drift(points, ship):
    ax = float(points[1, 0]); ay = float(points[1, 1])
    bx = float(points[2, 0]); by = float(points[2, 1])
    dx = bx - ax; dy = by - ay
    chord = math.hypot(dx, dy)
    if chord <= 0.0:
        return 0.0
    return abs((float(ship.position.x) - ax) * dy
               - (float(ship.position.y) - ay) * dx) / chord


w, ship, p, erde, period18 = _earth_orbit_scene()
p.set_view_scale(VIEW_SCALE_18)
p.update(ship, w)
p.set_hold(True)
drifts = []
false_markers = 0
try:
    for _ in range(FRAMES_18):
        advance(w, WARP_18 / 60.0)
        p.update(ship, w)
        pts = p.points
        if pts.shape[0] < 6:
            break
        drifts.append(_lateral_drift(pts, ship))
        dt_pt = float(pts[3, 2]) - float(pts[2, 2])
        for row in p.get_apsis_markers():
            offset = float(row[2]) - float(w.time)
            if not (0.0 <= offset <= 4.0 * dt_pt):
                continue
            # Auf einem echten apsis-durchgang gehoert die fahne dorthin.
            # Falsch ist sie, wenn der abstand an dieser stelle noch monoton
            # laeuft -- dann gibt es dort gar kein extremum.
            r = []
            for i in range(6):
                q = erde.position_at_time(float(pts[i, 2]))
                r.append(math.hypot(float(pts[i, 0]) - q.x,
                                    float(pts[i, 1]) - q.y))
            rising = all(r[i + 1] > r[i] for i in range(5))
            falling = all(r[i + 1] < r[i] for i in range(5))
            if (rising and float(row[3]) >= 0.5) or (falling and float(row[3]) < 0.5):
                false_markers += 1
finally:
    p.close()

drifts = np.array(drifts)
worst_px = float(drifts.max()) * VIEW_SCALE_18
check(worst_px < 0.5,
      "im zeitraffer bleibt das schiff auf der gehaltenen linie",
      f"groesster seitlicher versatz {drifts.max():.3e} m = {worst_px:.3f} px "
      f"ueber {len(drifts)} frames "
      f"({len(drifts) * (WARP_18 / 60.0) / period18:.1f} umlaeufe)")
check(false_markers == 0,
      "und keine Ap/Pe-fahne sitzt auf dem schiff, wo es gar kein extremum gibt",
      f"{false_markers} von {len(drifts)} frames")

# (c) Die schranke ist ein PIXELMASS, kein weltmass -- und sie laesst sich
#     abschalten. Eine weltlaenge waere auf jeder zoomstufe etwas anderes:
#     dieselben 50 km sind einmal unsichtbar und einmal fingerdick.
_wl18, _shipl18, _pl18, _el18, _Tl18 = _earth_orbit_scene()
try:
    _pl18.hold_drift_max_px = 0.5
    _pl18.set_view_scale(1e-5)
    check(abs(_pl18._hold_drift_limit_m() - 0.5 / 1e-5) < 1e-6,
          "die schranke rechnet pixel in meter um",
          f"0.5 px bei 1e-5 px/m = {_pl18._hold_drift_limit_m():.1f} m")
    _pl18.set_view_scale(1e-3)
    check(abs(_pl18._hold_drift_limit_m() - 0.5 / 1e-3) < 1e-6,
          "und zwar mit dem zoom, nicht fest",
          f"0.5 px bei 1e-3 px/m = {_pl18._hold_drift_limit_m():.1f} m")
    _pl18.hold_drift_max_px = 0.0
    check(_pl18._hold_drift_limit_m() == float('inf'),
          "0 px heisst AUS, nicht 'null meter erlaubt'",
          f"{_pl18._hold_drift_limit_m()}")
finally:
    _pl18.close()


# ═════════════════ 19. der vorangestellte kopf faelscht die apsis-suche nicht

print()
print("19. Der selbst vorangestellte kopf faelscht die apsis-suche nicht")

# Der kopf im halt ist die WELT-position des schiffs, nicht ein punkt dieser
# kurve. Als startwert des trend-scans gelesen kippt sein sprung die
# richtung, und die suche meldet ein extremum bei index 1 -- eine fahne
# direkt auf dem schiff. Geprueft wird am kern selbst: dieselbe kurve, ein
# kuenstlich versetzter kopf, einmal mit und einmal ohne `skip_head`.

from predictor import _find_apsis_markers_numba as _apsis_kernel

_w19, _ship19, _p19, _erde19, _ = _earth_orbit_scene()
try:
    _p19.async_compute = False
    _p19.set_view_scale(VIEW_SCALE_18)
    _p19.update(_ship19, _w19)
    _snap19 = _p19._last_swapped_snapshot
    _pts19 = np.array(_p19.points, dtype=np.float64)

    def _scan(points, skip_head):
        out, count = _apsis_kernel(
            points, float(_snap19.get("sim_time", 0.0)),
            int(_snap19.get("reference_body_index", -1)),
            _snap19["body_x"], _snap19["body_y"], _snap19["body_m"],
            _snap19["body_scripted"], _snap19["body_a"], _snap19["body_e"],
            _snap19["body_theta"], _snap19["body_arg"], _snap19["body_parent"],
            float(_snap19["G"]),
            1 if bool(_snap19.get("use_time_dependent_bodies", True)) else 0,
            int(_p19.apsis_max_markers), int(skip_head),
        )
        return out[:int(count)].copy()

    clean = _scan(_pts19, 0)

    # DEN KOPF SO VERSETZEN, WIE ES DER HALT TUT: RADIAL.
    #
    # Der kopf ist die welt-position des schiffs; sie weicht radial von der
    # kurve ab. Entscheidend ist die RICHTUNG: laeuft der abstand am anfang
    # der kurve nach oben, muss der kopf nach AUSSEN, dann sieht der scan
    # zuerst ein fallen und meldet gleich darauf eine periapsis. Laeuft er
    # nach unten, umgekehrt. Der betrag (30 punktweiten) ist die
    # groessenordnung, die im halt ueber ein paar umlaeufe wirklich entstand.
    _erde_at = _erde19.position_at_time(float(_pts19[0, 2]))
    _rx = float(_pts19[0, 0]) - _erde_at.x
    _ry = float(_pts19[0, 1]) - _erde_at.y
    _rlen = math.hypot(_rx, _ry)
    _erde_at1 = _erde19.position_at_time(float(_pts19[3, 2]))
    _r1 = math.hypot(float(_pts19[3, 0]) - _erde_at1.x,
                     float(_pts19[3, 1]) - _erde_at1.y)
    _outward = 1.0 if _r1 > _rlen else -1.0
    _step_len = math.hypot(float(_pts19[2, 0]) - float(_pts19[1, 0]),
                           float(_pts19[2, 1]) - float(_pts19[1, 1]))
    bumped = _pts19.copy()
    bumped[0, 0] += _outward * (_rx / _rlen) * _step_len * 30.0
    bumped[0, 1] += _outward * (_ry / _rlen) * _step_len * 30.0

    naive = _scan(bumped, 0)
    guarded = _scan(bumped, 1)

    def _first_index(markers, points):
        """Wie weit vorn sitzt die erste fahne, in punkten?"""
        if markers.shape[0] == 0:
            return None
        dt = float(points[3, 2]) - float(points[2, 2])
        return (float(markers[0, 2]) - float(points[0, 2])) / max(dt, 1e-9)

    naive_first = _first_index(naive, bumped)
    guarded_first = _first_index(guarded, bumped)
    clean_first = _first_index(clean, _pts19)

    check(naive_first is not None and naive_first < 4.0,
          "ohne die absicherung erzeugt der versetzte kopf eine fahne am schiff",
          f"erste fahne bei {naive_first:.2f} punkten "
          f"(unversetzt {clean_first:.1f})")
    check(guarded_first is not None and guarded_first > 4.0,
          "mit skip_head bleibt die fahne dort, wo sie ohne versatz war",
          f"erste fahne bei {guarded_first:.2f} punkten "
          f"(unversetzt {clean_first:.2f})")
    check(guarded.shape[0] == clean.shape[0]
          and bool(np.allclose(guarded[:, 2], clean[:, 2], rtol=0.0, atol=1e-6)),
          "und sie meldet dieselben fahnen wie die ungestoerte kurve",
          f"{guarded.shape[0]} gegen {clean.shape[0]} fahnen")
    check(bool(np.array_equal(_scan(_pts19, 0), clean)),
          "ohne halt (skip_head = 0) ist die suche bit-identisch",
          "gleicher aufruf, gleiches ergebnis")
finally:
    _p19.close()


# ═════════════ 20. die horizont-decke springt nicht ueber die bahn hinweg

print()
print("20. Ein langer horizont verbiegt die bahn nicht")

# `rkn_adaptive_far_maxdt` hebt die schrittweiten-decke mit dem HORIZONT an,
# damit ein weiter blick nicht mit der laenge teurer wird. Die decke kennt
# aber die bahn nicht: nach ein paar '+'-druecken deckt EIN schritt einen
# nennenswerten teil der umlaufzeit ab, und dann bestimmt nicht mehr die
# fehlerkontrolle die schrittweite, sondern die decke. Gemessen in einer
# erdumlaufbahn (rp 2e7 m, e = 0.6, T = 97 h) bei 64x horizont: die linie
# wich gegen dieselbe rechnung mit fester decke um bis zu 6.0e7 m ab --
# mehr, als die bahn gross ist.


def _long_horizon_points(mult, adaptive, ecc=0.6):
    w, ship, p, _erde, _period = _earth_orbit_scene(ecc=ecc)
    try:
        p.async_compute = False
        p.auto_precision_from_zoom = False
        p.rkn_adaptive_far_maxdt = bool(adaptive)
        p.set_length(p.num_points * p.precision * mult)
        p.reset()
        p.update(ship, w)
        return np.array(p.points, dtype=np.float64)
    finally:
        p.close()


def _probe20(points, t):
    ts = points[:, 2]
    if t < ts[0] or t > ts[-1]:
        return None
    i = max(1, min(int(np.searchsorted(ts, t)), len(ts) - 1))
    f = (t - ts[i - 1]) / max(1e-9, ts[i] - ts[i - 1])
    return (points[i - 1, 0] + (points[i, 0] - points[i - 1, 0]) * f,
            points[i - 1, 1] + (points[i, 1] - points[i - 1, 1]) * f)


for _mult in (16, 64):
    _fixed = _long_horizon_points(_mult, adaptive=False)
    _raised = _long_horizon_points(_mult, adaptive=True)
    _span20 = float(_fixed[-1, 2] - _fixed[0, 2])
    _worst20 = 0.0
    for _frac in (0.1, 0.5, 0.999):
        _t20 = float(_fixed[0, 2]) + _span20 * _frac
        _a20 = _probe20(_raised, _t20)
        _b20 = _probe20(_fixed, _t20)
        if _a20 and _b20:
            _worst20 = max(_worst20, math.hypot(_a20[0] - _b20[0],
                                                _a20[1] - _b20[1]))
    check(_worst20 == 0.0,
          f"{_mult}x horizont: dieselbe bahn wie mit fester schrittdecke",
          f"groesste abweichung {_worst20:.3e} m ueber "
          f"{_span20 / 3600.0:.0f} h")

# Gegenprobe: im FERNFELD muss die decke weiterhin greifen, sonst waere der
# riegel oben einfach ein "aus" fuer die ganze mechanik.
_wf, _shipf, _pf = build()
try:
    _tchar = _wf.characteristic_timescale(_shipf)
    _orbit_cap = _tchar / _pf.rkn_max_dt_timescale_divisor
    check(_orbit_cap > _pf.rkn_max_dt_ceiling,
          "im fernfeld bleibt der bahn-riegel weit ueber der decke",
          f"bahn-riegel {_orbit_cap:.3e} s gegen decke "
          f"{_pf.rkn_max_dt_ceiling:.3e} s (t_char {_tchar:.3e} s)")
finally:
    _pf.close()


print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for failure in FAILURES:
        print(f"  {failure}")
    sys.exit(1)
print("zeitraffer/predictor: alle pruefungen bestanden")
