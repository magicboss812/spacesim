"""Abnahmepruefung fuer die transfer-vorhersage (TASKS.md, Task 1 Teil B).

Faehrt die app hoch wie `tools/game_shot.py`, fliegt aus dem Erde-parkorbit
einen Hohmann-transfer zu Saturn oder Neptun -- als ausgefuehrter
manoeverknoten oder von hand mit vollschub -- und misst vier dinge:

  1. die BRENNPHASE: anteil der bilder mit frischer linie, alter der
     gezeigten linie, abweichung von einer synchronen referenz (px),
     hauptthread-kosten des predictors je bild;
  2. BRENNSCHLUSS: wie die linie einrastet (sprung des Ap-markers, rest
     gegen die exakte gleitlinie);
  3. das VORHERGESAGTE Ap direkt nach brennschluss (ort, zeit, abstand zur
     Sonne) samt rechenzeit und schrittzahl einer vorhersage an diesem
     horizont;
  4. den WARP der welt bis dorthin mit dem echten schrittmuster
     (`world.step` mit dem sim_step der schleife, an den stufen des HUDs):
     wo das schiff sein abstands-extremum wirklich erreicht, und wie weit
     der marker unterwegs wandert.

Alles, was die schleife je bild tut, laeuft in derselben reihenfolge wie in
`runtime/loop.py` (dieselben hilfsfunktionen), gezeichnet wird nicht. Die
bilder laufen im takt von 180 fps (die bildrate des spielers), weil die
asynchrone pipeline des predictors an der WANDzeit haengt.

    xvfb-run -a -s "-screen 0 1280x800x24" \\
        python tools/transfer_bench.py --target Saturn --mode node -o out.json

Pixelangaben gelten fuer einen 2560x1440-schirm in zwei zoomstufen:
`soi` (die einflusssphaere des ziels fuellt die schirmhoehe -- "die
Ap-gegend fuellt den schirm") und `full` (die ganze gezeichnete linie
passt auf den schirm).
"""
import argparse
import collections
import json
import math
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SPACESIM = os.path.dirname(HERE)
sys.path.insert(0, SPACESIM)
os.chdir(SPACESIM)

parser = argparse.ArgumentParser()
parser.add_argument('--target', default='Saturn')
parser.add_argument('--mode', default='node', choices=('node', 'manual'))
parser.add_argument('--mult', type=float, default=None,
                    help='manueller horizont-faktor (sonst je ziel gewaehlt)')
parser.add_argument('--ref-every', type=int, default=30,
                    help='jedes wievielte brenn-bild gegen eine synchrone '
                         'referenz vergleichen')
parser.add_argument('--settle-frames', type=int, default=360)
parser.add_argument('--settle-max-s', type=float, default=240.0)
parser.add_argument('-o', '--out', default=None)
parser.add_argument('--dump-snapshot', default=None,
                    help='schnappschuss nach brennschluss als pickle ablegen '
                         '(fuer profile des kernels)')
parser.add_argument('--snapshots-after', default=None,
                    help='komma-liste von sim-sekunden nach brennschluss: zu '
                         'jeder einen schnappschuss des spiels ablegen (mit '
                         '--dump-snapshots). Die Ap-genauigkeit haengt am '
                         'moment, von dem aus gerechnet wird.')
parser.add_argument('--dump-snapshots', default=None,
                    help='pickle-datei fuer die schnappschuss-reihe')
parser.add_argument('--coast-frames', type=int, default=0,
                    help='nach dem einrasten so viele bilder gleitflug in '
                         'echtzeit und dabei JEDEN Ap/Pe-marker je bild '
                         'mitschreiben (zittern ohne schub und ohne warp)')
parser.add_argument('--rtol', type=float, default=None,
                    help='rkn_rtol des predictors ueberschreiben')
parser.add_argument('--no-warp', action='store_true',
                    help='ohne den zeitraffer bis zum Ap (nur brand und '
                         'einrasten messen)')
args = parser.parse_args()

W, H = 1280, 800
FPS = 180.0
FRAME_DT = 1.0 / FPS
SCREEN_W, SCREEN_H = 2560.0, 1440.0

with open(os.path.join(SPACESIM, 'config', 'config.json'), 'r',
          encoding='utf-8-sig') as fh:
    cfg = json.load(fh)
cfg['window']['width'] = W
cfg['window']['height'] = H
cfg['window']['vsync'] = False
cfg['window']['fps'] = int(FPS)
cfg['debug']['print_frame_timings'] = False
cfg['debug']['print_loader_info'] = False
tmp_cfg = os.path.join(tempfile.gettempdir(), 'spacesim_transfer_config.json')
with open(tmp_cfg, 'w', encoding='utf-8') as fh:
    json.dump(cfg, fh)
os.environ['SPACESIM_CONFIG'] = tmp_cfg

import numpy as np
import pygame

from runtime.bootstrap import build_app, load_config
from runtime.loop import (_apply_horizon, _apply_maneuver, _clamp_warp,
                          _update_maneuver_preview, _update_predictor)
from ship.maneuver import ManeuverNode
from ship.maneuver.preview import body_state_at, state_on_curve
from ship.maneuver.profile import BurnProfile
from ship.predictor import Predictor
from ui import units
from ui.hud.layout import WARP_STEPS

# Horizont-faktor je ziel: gerade so, dass das Ap auf der gezeichneten linie
# liegt (halber ellipsenumfang: Saturn ~2.0e12 m, Neptun ~5.5e12 m).
DEFAULT_MULT = {'saturn': 256.0, 'neptun': 1024.0}


# ------------------------------------------------------------------ aufbau

app = build_app(load_config())
world = app.world
ship = app.ship
pred = app.predictor

# Jede ein- oder abgewiesene einwechselung mitschreiben (der predictor loggt
# sie selbst nur mit `debug`): (sim-zeit, angenommen, grund, vorlauf des
# ergebnisses). Damit laesst sich einer abweichung ihre ursache zuordnen.
swap_log = []
_orig_log_snapshot_result = pred._log_snapshot_result


def _record_snapshot_result(accepted, reason, snapshot, *a, **kw):
    lead_s = 0.0 if snapshot is None else float(snapshot.get('lead_s', 0.0) or 0.0)
    swap_log.append((float(world.time), bool(accepted), str(reason), lead_s))
    return _orig_log_snapshot_result(accepted, reason, snapshot, *a, **kw)


pred._log_snapshot_result = _record_snapshot_result
bodies = {b.name.lower(): b for b in world.body}
sonne = bodies['sonne']
erde = bodies['erde']
target = bodies[args.target.lower()]
target_a = float(target.semi_major_axis)
r_soi = target_a * (float(target.mass) / float(sonne.mass)) ** 0.4
ZOOM_SOI = SCREEN_H / (2.0 * r_soi)

# Bezugsrahmen Sonne: die Ap-marker messen den abstand zur Sonne.
app.ui_state.set_reference_index(world.body.index(sonne))
app.camera.scale = ZOOM_SOI
app.camera.target_scale = ZOOM_SOI
mult = args.mult if args.mult else DEFAULT_MULT.get(target.name.lower(), 256.0)
app.horizon.manual_mult = float(mult)

# Eigene, nie laufende instanzen nur fuer die synchronen referenzen: sie
# rechnen aus einem schnappschuss, ohne am zustand des spiels zu drehen. Eine
# je faden -- die referenzen laufen parallel (alle kernel sind `nogil`).
_tls = threading.local()


def ref_predictor():
    p = getattr(_tls, 'pred', None)
    if p is None:
        p = Predictor(async_compute=False, debug=False)
        p.apsis_max_markers = pred.apsis_max_markers
        _tls.pred = p
    return p


class _Keys(collections.defaultdict):
    pass


THRUST_KEYS = _Keys(int, {pygame.K_UP: 1})
NO_KEYS = _Keys(int)


def set_warp(rate):
    app.camera.sim_dt = float(rate) / app.tick_rate


def allowed_rate_cap():
    t_char = world.characteristic_timescale(ship)
    if not t_char:
        return app.realtime_warp_max
    return max(t_char / app.warp_timescale_divisor * app.tick_rate,
               app.realtime_warp_max)


def highest_step(limit_rate):
    cap = min(allowed_rate_cap(), limit_rate) * 1.001
    best = WARP_STEPS[0][0]
    for rate, _label in WARP_STEPS:
        if rate <= cap:
            best = rate
    return best


def prograde_theta_rel(body):
    bx, by, bvx, bvy = body_state_at(body, world.time)
    vx = ship.velocity.x - bvx
    vy = ship.velocity.y - bvy
    # schiffsnase = (cos theta, -sin theta), siehe ship/control.py
    return math.atan2(-vy, vx)


def state_time(snapshot):
    """Zu welcher sim-zeit der STARTZUSTAND eines laufs gehoert."""
    if snapshot is None:
        return None
    return float(snapshot.get('sim_time', 0.0)) + float(
        snapshot.get('lead_s', 0.0) or 0.0)


def is_post_burnout(snapshot, t_burnout):
    """Beschreibt diese linie den zustand NACH brennschluss?

    Ja, wenn ihr schnappschuss danach genommen wurde -- oder wenn ihr vorlauf
    das PROFIL des ausfuehrers war und ueber den brennschluss reichte (das
    profil kennt ihn). Ein vorlauf mit gehaltener eingabe ueber den
    brennschluss hinaus nimmt dagegen schub an, den es nicht gab.
    """
    if snapshot is None:
        return False
    t_s = float(snapshot.get('sim_time', 0.0))
    if t_s >= t_burnout - 1e-6:
        return True
    lead = snapshot.get('lead') or {}
    return int(lead.get('mode', 0)) == 2 and state_time(snapshot) >= t_burnout - 1e-6


def pick_ap(markers, t_hint=None):
    """Das Ap an der zielentfernung (kind 1, r ueber der halben bahn)."""
    if markers is None or len(markers) == 0:
        return None
    best = None
    for m in markers:
        if int(round(m[3])) != 1 or m[4] < 0.5 * target_a:
            continue
        if best is None:
            best = m
        elif t_hint is not None and abs(m[2] - t_hint) < abs(best[2] - t_hint):
            best = m
        elif t_hint is None and m[2] < best[2]:
            best = m
    return None if best is None else [float(v) for v in best]


def reference_line(snapshot):
    """Synchrone gleitlinie aus einem schnappschuss + ihr Ap."""
    rp = ref_predictor()
    result = rp._compute_from_snapshot_impl(snapshot)
    pts = result['points']
    rp.points = pts
    rp._last_swapped_snapshot = snapshot
    rp._synthetic_head = False
    rp._points_time_offset = 0.0
    rp.display_length = None
    rp._clear_apsis_markers()
    return pts, pick_ap(rp.get_apsis_markers())


def line_deviation(drawn, ref_pts, where=False):
    """Groesste abweichung der gezeichneten punkte von der referenz bei
    GLEICHER zeit, in metern (mit `where` auch die zeit dieses punkts)."""
    worst = 0.0
    t_worst = None
    for x, y, t in drawn:
        s = state_on_curve(ref_pts, t)
        if s is None:
            continue
        d = math.hypot(x - s[0], y - s[1])
        if d > worst:
            worst = d
            t_worst = t
    return (worst, t_worst) if where else worst


def eps_rel(b):
    """Spezifische bahnenergie des schiffs relativ zu `b` (J/kg); >0 heisst
    ungebunden."""
    bx, by, bvx, bvy = body_state_at(b, world.time)
    r = math.hypot(ship.position.x - bx, ship.position.y - by)
    v = math.hypot(ship.velocity.x - bvx, ship.velocity.y - bvy)
    return 0.5 * v * v - world.G * b.mass / r


def subsample(pts, n=400):
    k = len(pts)
    if k <= n:
        idx = np.arange(k)
    else:
        idx = np.unique(np.linspace(0, k - 1, n).astype(int))
    return [(float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])) for i in idx]


def line_zoom_full(pts):
    x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
    y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
    return min(SCREEN_W / max(x1 - x0, 1.0), SCREEN_H / max(y1 - y0, 1.0))


# ------------------------------------------------------------ ein bild

_wall = [0.0]


def frame(thrust=False, pace=True, warp=None):
    """Ein bild, in der reihenfolge von runtime/loop.py::run()."""
    t_start = time.perf_counter()
    if warp is not None:
        set_warp(warp)
    if thrust:
        ship.theta = prograde_theta_rel(erde)
        if app.thrust_allowed():
            app.ship_control.apply_thrust(THRUST_KEYS, FRAME_DT)
    _clamp_warp(app)
    _apply_maneuver(app)
    _apply_horizon(app)
    sim_step = app.camera.sim_dt * app.tick_rate * FRAME_DT
    cap = app.maneuver_executor.max_sim_seconds(world)
    if cap is not None:
        sim_step = min(sim_step, max(1e-9, cap))
    app.maneuver_executor.update(world, sim_step)
    world.step(sim_step, app.max_substep)
    app.camera.update(FRAME_DT, ui_wants_keyboard=False)
    t_pred0 = time.perf_counter()
    swapped_before = pred._jobs_swapped
    _update_predictor(app)
    # HUD und renderer lesen die marker je bild (beide auf dem hauptthread).
    markers = pred.get_apsis_markers()
    pred_ms = (time.perf_counter() - t_pred0) * 1000.0
    _update_maneuver_preview(app)
    _wall[0] += FRAME_DT
    if pace:
        remaining = FRAME_DT - (time.perf_counter() - t_start)
        if remaining > 0.0:
            time.sleep(remaining)
    return {
        'sim_step': sim_step,
        'swapped': pred._jobs_swapped != swapped_before,
        'pred_ms': pred_ms,
        'markers': markers,
    }


# ----------------------------------------------------- den transfer planen

def plan_transfer():
    """Knotenzeit und delta-v fuer einen tangentialen abflug (patched conic).

    Die asymptote der abflughyperbel soll in richtung der Erdbahngeschwindig-
    keit zeigen; der brennpunkt liegt deshalb um den asymptotenwinkel
    nu_inf = acos(-1/e) davor.
    """
    mu_e = world.G * erde.mass
    mu_s = world.G * sonne.mass
    t0 = world.time
    ex, ey, evx, evy = body_state_at(erde, t0)
    rx, ry = ship.position.x - ex, ship.position.y - ey
    vx, vy = ship.velocity.x - evx, ship.velocity.y - evy
    r = math.hypot(rx, ry)
    v = math.hypot(vx, vy)
    h = rx * vy - ry * vx
    sense = 1.0 if h > 0.0 else -1.0
    n = abs(h) / (r * r)
    r_e = math.hypot(ex - sonne.position.x, ey - sonne.position.y)
    v_e = math.hypot(evx, evy)
    v_p = math.sqrt(2.0 * mu_s * target_a / (r_e * (r_e + target_a)))
    v_inf = v_p - v_e
    dv = math.sqrt(v_inf * v_inf + 2.0 * mu_e / r) - v
    ecc = 1.0 + r * v_inf * v_inf / mu_e
    nu_inf = math.acos(-1.0 / ecc)
    psi = math.atan2(evy, evx)
    phi_b = psi - sense * nu_inf
    phi0 = math.atan2(ry, rx)
    dphi = ((phi_b - phi0) * sense) % (2.0 * math.pi)
    t_node = t0 + dphi / n
    profile = BurnProfile(dv, app.maneuver_executor.a_max_sim(),
                          app.maneuver_config['ramp_seconds'])
    # Genug vorlauf fuer ausrichten und die halbe brenndauer.
    while t_node - profile.total_time * 0.5 - t0 < 120.0:
        t_node += 2.0 * math.pi / n
    return t_node, dv, profile


def main():
    report = {'target': target.name, 'mode': args.mode,
              'horizon_mult': float(mult), 'zoom_soi_px_per_m': ZOOM_SOI,
              'r_soi_m': r_soi, 'target_a_m': target_a,
              'cpu_count': os.cpu_count()}

    if args.rtol is not None:
        pred.rkn_rtol = float(args.rtol)
    # Einschwingen in echtzeit: die linie steht, die JITs sind warm.
    set_warp(app.realtime_warp_max)
    for _ in range(120):
        frame()

    t_node, dv, profile = plan_transfer()
    t_ign = profile.ignition_time(t_node)
    report['plan'] = {'t_node': t_node, 'dv': dv,
                      'burn_s': profile.total_time, 't_ignition': t_ign}
    print(f"PLAN: knoten T+{t_node - world.time:.0f} s, dv {dv:.1f} m/s, "
          f"brenndauer {profile.total_time:.1f} s")

    if args.mode == 'node':
        app.maneuver_plan.add(ManeuverNode(t_node, dv_prograde=dv,
                                           dv_normal=0.0))
        # Knoten im Erde-rahmen aufloesen: prograde heisst hier relativ zur
        # Erde (der parkorbit), nicht zur Sonne.
        app.maneuver_preview.wait(10.0)
        app.maneuver_preview.rebuild(
            app.maneuver_plan, ship, world, pred, erde,
            app.maneuver_executor.a_max_sim(),
            app.maneuver_config['ramp_seconds'])
        ok = app.maneuver_executor.arm(world, erde, app.maneuver_preview)
        if not ok:
            pts = pred.get_points()
            raise SystemExit(
                'knoten liess sich nicht scharfschalten: vorschau '
                f'valid={app.maneuver_preview.valid} '
                f'marker={len(app.maneuver_preview.node_markers)} '
                f'linie {len(pts)} punkte bis T+{pts[-1, 2] - world.time:.0f} s')
        # Ab jetzt baut die vorschau gegen den anzeige-rahmen; die richtung
        # des ausfuehrers ist beim scharfschalten festgehalten.

    # -- vorlauf bis zur zuendung (zeitraffer wie ein spieler) -------------
    while world.time < t_ign - 1.0:
        left = t_ign - 5.0 - world.time
        limit = max(app.realtime_warp_max, left / 3.0 / FRAME_DT)
        frame(pace=False, warp=highest_step(limit))
    set_warp(app.realtime_warp_max)
    for _ in range(3):
        frame()

    # -- brennen ------------------------------------------------------------
    manual_frames = 0
    if args.mode == 'manual':
        sim_step_rt = app.realtime_warp_max / app.tick_rate * app.tick_rate * FRAME_DT
        manual_frames = int(round((dv / app.maneuver_executor.a_max_sim())
                                  / sim_step_rt))
    burn = {'frames': 0, 'fresh': 0, 'age_sim_s': [], 'pred_ms': [],
            'depth': [], 'compute_ms': [], 'trace': []}
    samples = []
    last_burn_ap = None
    t_burn0 = world.time
    burn_submitted0 = pred._jobs_submitted
    i = 0
    while True:
        if args.mode == 'node':
            if not app.maneuver_executor.is_active:
                break
            info = frame()
        else:
            if i >= manual_frames:
                break
            info = frame(thrust=True)
        i += 1
        last_burn_ap = pick_ap(info['markers'])
        burn['frames'] += 1
        burn['fresh'] += 1 if info['swapped'] else 0
        st = state_time(pred._last_swapped_snapshot)
        if st is not None:
            burn['age_sim_s'].append(world.time - st)
        burn['pred_ms'].append(info['pred_ms'])
        burn['depth'].append(int(getattr(pred, '_pipeline_depth_used', 1)))
        burn['compute_ms'].append(float(pred.last_compute_ms))
        burn['trace'].append((world.time - t_burn0,
                              int(pred._async_jobs_in_flight()),
                              int(pred._async_jobs_in_flight(exclude_coast=True)),
                              int(pred._jobs_submitted), int(pred._jobs_swapped),
                              float(getattr(pred, '_lead_lag_wall_ema', 0.0) or 0.0),
                              float(pred._expected_lead_s())))
        if i % max(1, args.ref_every) == 0:
            pts = pred.get_points()
            if pts is not None and len(pts) > 2:
                samples.append({
                    'phase': 'burn', 't': world.time,
                    'snapshot': pred._make_snapshot(
                        ship, world, pred._get_target_point_cap()),
                    'drawn': subsample(pts),
                    'ap': pick_ap(info['markers']),
                    'age_sim_s': (None if st is None else world.time - st),
                    'eps_erde': eps_rel(erde),
                })
    t_burnout = world.time
    print(f"BRENNSCHLUSS nach {burn['frames']} bildern, "
          f"{t_burnout - t_burn0:.1f} sim-s")

    # -- einrasten nach brennschluss ---------------------------------------
    # Solange, bis die erste linie aus einem zustand NACH brennschluss steht
    # (hoechstens `--settle-max-s` wandzeit), dann noch `--settle-frames`
    # bilder fuer die spruenge danach.
    settle_aps = []
    first_post = None
    series_at = sorted(float(v) for v in args.snapshots_after.split(',')) \
        if args.snapshots_after else []
    series = []
    settle_t0 = time.perf_counter()
    k = -1
    post_frames = 0
    while post_frames < args.settle_frames:
        k += 1
        if first_post is None and time.perf_counter() - settle_t0 > args.settle_max_s:
            break
        info = frame()
        while series_at and world.time >= t_burnout + series_at[0]:
            series_at.pop(0)
            series.append(pred._make_snapshot(ship, world,
                                              pred._get_target_point_cap()))
        ap = pick_ap(info['markers'])
        post = is_post_burnout(pred._last_swapped_snapshot, t_burnout)
        settle_aps.append({'t': world.time, 'swapped': info['swapped'],
                           'post': bool(post), 'ap': ap})
        if post and first_post is None and ap is not None:
            first_post = {'t': world.time, 'ap': ap,
                          'compute_ms': float(pred.last_compute_ms),
                          'steps': int(pred.rkn_last_accepted_steps),
                          'wall_s': time.perf_counter() - settle_t0}
        if first_post is not None and not series_at:
            post_frames += 1
        if first_post is not None and post_frames % max(1, args.ref_every) == 0:
            pts = pred.get_points()
            samples.append({
                'phase': 'settle', 't': world.time,
                'snapshot': pred._make_snapshot(
                    ship, world, pred._get_target_point_cap()),
                'drawn': subsample(pts),
                'ap': ap,
            })
    if args.coast_frames > 0:
        report['coast'] = coast_markers(args.coast_frames)
    final_snapshot = pred._make_snapshot(ship, world, pred._get_target_point_cap())
    if args.dump_snapshot:
        import pickle
        with open(args.dump_snapshot, 'wb') as fh:
            pickle.dump(final_snapshot, fh)
    if args.dump_snapshots:
        import pickle
        with open(args.dump_snapshots, 'wb') as fh:
            pickle.dump({'t_burnout': t_burnout, 'snapshots': series}, fh)
    pts_final = pred.get_points()
    zoom_full = line_zoom_full(pts_final)
    report['zoom_full_px_per_m'] = zoom_full

    # -- rechenkosten am transfer-horizont (einzeln, ohne konkurrenz) ------
    # Erst die noch laufenden auftraege des spiels abwarten, sonst misst die
    # rechnung gegen sie.
    t_wait = time.perf_counter()
    while pred._async_jobs_in_flight() and time.perf_counter() - t_wait < 120.0:
        time.sleep(0.01)
    times, steps = [], []
    for _ in range(5):
        t0 = time.perf_counter()
        res = ref_predictor()._compute_from_snapshot_impl(final_snapshot)
        times.append((time.perf_counter() - t0) * 1000.0)
        steps.append(int(res['rkn_stats'][0]))
    report['compute'] = {
        'ms_median': float(np.median(times)), 'ms_min': float(min(times)),
        'steps': steps[0], 'rejected': int(res['rkn_stats'][1]),
        'points': int(len(res['points'])),
        'max_dt': float(res['rkn_stats'][3]),
    }
    ref_final_pts, ref_final_ap = reference_line(final_snapshot)

    # -- burn-auswertung gegen synchrone referenzen -------------------------
    # Parallel: die referenzen der gebundenen phase kosten je sekunden.
    with ThreadPoolExecutor(max_workers=max(1, os.cpu_count() or 1)) as pool:
        refs = list(pool.map(lambda smp: reference_line(smp['snapshot']),
                             samples))
    for s, (ref_pts, ref_ap) in zip(samples, refs):
        d, s['t_worst'] = line_deviation(s['drawn'], ref_pts, where=True)
        s['dev_m'] = d
        s['dev_full_px'] = d * zoom_full
        if s['ap'] is not None and ref_ap is not None:
            s['dev_ap_soi_px'] = math.hypot(s['ap'][0] - ref_ap[0],
                                            s['ap'][1] - ref_ap[1]) * ZOOM_SOI
        else:
            s['dev_ap_soi_px'] = None
    burn_s = [s for s in samples if s['phase'] == 'burn']
    settle_s = [s for s in samples if s['phase'] == 'settle']
    # Je probe: wann im brand, wie alt die gezeigte linie, wie weit voraus
    # der schlimmste punkt liegt und ob das schiff an der Erde noch gebunden
    # ist -- damit sich ein p95 einer ursache zuordnen laesst.
    burn_log = [e for e in swap_log if t_burn0 <= e[0] <= t_burnout]
    reasons = collections.Counter(
        ('ok' if e[1] else 'rej') + ':' + e[2] for e in burn_log)
    report['burn_swaps'] = {
        'reasons': dict(reasons),
        'submitted': int(pred._jobs_submitted - burn_submitted0),
        'log': [{'t_rel': e[0] - t_burn0, 'ok': e[1], 'reason': e[2],
                 'lead_s': e[3]} for e in burn_log],
    }
    report['burn_trace'] = [
        {'t_rel': t, 'in_flight': f, 'in_flight_lead': fl, 'submitted': a,
         'swapped': b, 'lag_ema': lg, 'lead': ld}
        for t, f, fl, a, b, lg, ld in burn['trace']]
    report['burn_samples'] = [
        {'t_rel': s['t'] - t_burn0, 'dev_full_px': s['dev_full_px'],
         'dev_m': s['dev_m'], 'age_sim_s': s.get('age_sim_s'),
         'worst_ahead_s': (None if s['t_worst'] is None
                           else s['t_worst'] - s['t']),
         'eps_erde': s.get('eps_erde')} for s in burn_s]

    def _stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        a = np.asarray(vals, dtype=float)
        return {'median': float(np.median(a)), 'p95': float(np.percentile(a, 95)),
                'max': float(a.max()), 'n': int(a.size)}

    fr = burn['frames']
    report['burn'] = {
        'frames': fr,
        'fresh_share': burn['fresh'] / max(1, fr),
        'age_ms': _stats([a / app.realtime_warp_max * 1000.0
                          for a in burn['age_sim_s']]),
        'pred_main_ms': _stats(burn['pred_ms']),
        'depth': _stats(burn['depth']),
        'compute_ms_under_load': _stats(burn['compute_ms']),
        'dev_full_px': _stats([s['dev_full_px'] for s in burn_s]),
        'dev_ap_soi_px': _stats([s['dev_ap_soi_px'] for s in burn_s]),
    }

    # Einrasten: spruenge des markers zwischen aufeinanderfolgenden linien
    # NACH der ersten nach-brennschluss-linie, und der rest gegen die exakte
    # gleitlinie am ende.
    jumps = []
    prev = None
    last_pre = last_burn_ap
    jump_in = None
    for e in settle_aps:
        if not e['post']:
            if e['ap'] is not None:
                last_pre = e['ap']
            continue
        if e['ap'] is None:
            continue
        if prev is None and last_pre is not None:
            jump_in = math.hypot(e['ap'][0] - last_pre[0],
                                 e['ap'][1] - last_pre[1]) * ZOOM_SOI
        if prev is not None:
            jumps.append(math.hypot(e['ap'][0] - prev[0],
                                    e['ap'][1] - prev[1]) * ZOOM_SOI)
        prev = e['ap']
    last_ap = settle_aps[-1]['ap']
    report['settle'] = {
        'first_post_frame': next((k for k, e in enumerate(settle_aps)
                                  if e['post']), None),
        'first_post_wall_s': None if first_post is None else first_post['wall_s'],
        'jump_into_post_soi_px': jump_in,
        'jump_soi_px': _stats(jumps),
        'residual_soi_px': (None if last_ap is None or ref_final_ap is None else
                            math.hypot(last_ap[0] - ref_final_ap[0],
                                       last_ap[1] - ref_final_ap[1]) * ZOOM_SOI),
        'dev_full_px': _stats([s['dev_full_px'] for s in settle_s]),
        'dev_ap_soi_px': _stats([s['dev_ap_soi_px'] for s in settle_s]),
    }
    if first_post is None:
        raise SystemExit('kein Ap an der zielentfernung nach brennschluss -- '
                         'horizont zu kurz?')
    ap0 = first_post['ap']
    report['ap_pred'] = {'x': ap0[0], 'y': ap0[1], 't_abs': ap0[2],
                         'r': ap0[4], 'eta_s': ap0[2] - first_post['t'],
                         'compute_ms_in_game': first_post['compute_ms'],
                         'steps_in_game': first_post['steps']}
    report['ap_ref_final'] = (None if ref_final_ap is None else
                              {'x': ref_final_ap[0], 'y': ref_final_ap[1],
                               't_abs': ref_final_ap[2], 'r': ref_final_ap[4]})
    print(f"AP VORHERGESAGT: r {ap0[4]:.6e} m, T+{ap0[2] - world.time:.0f} s, "
          f"rechnung {report['compute']['ms_median']:.1f} ms, "
          f"{report['compute']['steps']} schritte")

    # -- zeitraffer bis zum Ap -------------------------------------------------
    drift = {'dpos_px': 0.0, 'dt_s': 0.0, 'dr_m': 0.0, 'labels': set(),
             'frames': 0, 'lost_frames': 0}
    label0 = units.distance(ap0[4])
    prev_state = (world.time, ship.position.x, ship.position.y,
                  ship.velocity.x, ship.velocity.y)
    extremum = None
    t_limit = world.time if args.no_warp else ap0[2] + 0.5 * (ap0[2] - t_burnout)
    warp_steps_used = collections.Counter()
    while world.time < t_limit:
        rate = highest_step(float('inf'))
        warp_steps_used[rate] += 1
        info = frame(warp=rate)
        drift['frames'] += 1
        ap = pick_ap(info['markers'], t_hint=ap0[2])
        # Nur DERSELBE marker zaehlt: ein Ap mehr als 5 % der flugzeit
        # daneben ist ein anderes (etwa das des naechsten umlaufs), und dass
        # das eigene fehlt, zaehlt als fehlend.
        if ap is not None and abs(ap[2] - ap0[2]) > 0.05 * max(ap0[2] - t_burnout, 1.0):
            ap = None
        if ap is not None and ap[2] > world.time:
            drift['dpos_px'] = max(drift['dpos_px'], math.hypot(
                ap[0] - ap0[0], ap[1] - ap0[1]) * ZOOM_SOI)
            drift['dt_s'] = max(drift['dt_s'], abs(ap[2] - ap0[2]))
            drift['dr_m'] = max(drift['dr_m'], abs(ap[4] - ap0[4]))
            drift['labels'].add(units.distance(ap[4]))
        elif ap0[2] > world.time:
            drift['lost_frames'] += 1
        state = (world.time, ship.position.x, ship.position.y,
                 ship.velocity.x, ship.velocity.y)
        rv0 = ((prev_state[1] - sonne.position.x) * prev_state[3]
               + (prev_state[2] - sonne.position.y) * prev_state[4])
        rv1 = ((state[1] - sonne.position.x) * state[3]
               + (state[2] - sonne.position.y) * state[4])
        # Erst im letzten viertel des anflugs suchen: beim abflug schwankt
        # der abstand zur Sonne, solange das schiff die Erde noch umrundet.
        near_ap = world.time > ap0[2] - 0.25 * (ap0[2] - t_burnout)
        if near_ap and rv0 > 0.0 >= rv1:
            extremum = find_extremum(prev_state, state)
            break
        prev_state = state
    report['warp'] = {
        'frames': drift['frames'],
        'steps_used': {label: warp_steps_used[r] for r, label in WARP_STEPS
                       if warp_steps_used[r]},
        'drift_soi_px': drift['dpos_px'],
        'drift_t_s': drift['dt_s'],
        'drift_r_m': drift['dr_m'],
        'label_first': label0,
        'labels_seen': sorted(drift['labels']),
        'marker_missing_frames': drift['lost_frames'],
    }
    if extremum is None:
        report['error'] = 'kein abstands-extremum gefunden'
    else:
        te, xe, ye = extremum
        r_act = math.hypot(xe - sonne.position.x, ye - sonne.position.y)
        dpos = math.hypot(xe - ap0[0], ye - ap0[1])
        report['ap_actual'] = {'t': te, 'x': xe, 'y': ye, 'r': r_act}
        report['ap_error'] = {
            'dt_s': te - ap0[2], 'dpos_m': dpos, 'dr_m': r_act - ap0[4],
            'dpos_soi_px': dpos * ZOOM_SOI,
        }
        print(f"AP ERREICHT: dt {te - ap0[2]:+.1f} s, dpos {dpos:.4e} m "
              f"({dpos * ZOOM_SOI:.3f} px), dr {r_act - ap0[4]:+.4e} m")

    out = args.out or os.path.join(tempfile.gettempdir(),
                                   f'transfer_{target.name}_{args.mode}.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=1, default=float)
    print('geschrieben:', out)
    print(json.dumps({k: report[k] for k in ('compute', 'burn', 'settle',
                                             'ap_error', 'warp')
                      if k in report}, indent=1, default=float))


def coast_markers(n_frames):
    """Gleitflug in echtzeit: wie weit wandert jeder marker, obwohl die bahn
    stillsteht?

    Ein marker wird ueber die bilder an seiner ZEIT wiedererkannt (art gleich,
    zeit innerhalb 2 % der restflugzeit). Gemeldet je marker: spannweite von
    abstand r und zeit t, spannweite der lage in px bei `soi`-zoom, wie oft
    sich der angezeigte wert (HUD-rundung) geaendert hat, und zum vergleich
    die zahl der einwechselungen in derselben zeit.
    """
    tracks = []
    swaps = 0
    t0 = world.time
    for _ in range(n_frames):
        info = frame()
        swaps += 1 if info['swapped'] else 0
        ms = info['markers']
        if ms is None:
            continue
        for m in ms:
            kind, t_abs, r = int(round(m[3])), float(m[2]), float(m[4])
            if t_abs <= world.time:
                continue
            tr = None
            for cand in tracks:
                if cand['kind'] == kind and abs(cand['t'][-1] - t_abs) <= \
                        0.02 * max(t_abs - world.time, 1.0):
                    tr = cand
                    break
            if tr is None:
                tr = {'kind': kind, 't': [], 'r': [], 'x': [], 'y': [],
                      'label': []}
                tracks.append(tr)
            tr['t'].append(t_abs)
            tr['r'].append(r)
            tr['x'].append(float(m[0]))
            tr['y'].append(float(m[1]))
            tr['label'].append(units.distance(r))
    out = {'frames': n_frames, 'sim_s': world.time - t0, 'swaps': swaps,
           'markers': []}
    for tr in tracks:
        if len(tr['t']) < max(3, n_frames // 2):
            continue
        x, y = np.asarray(tr['x']), np.asarray(tr['y'])
        changes = sum(1 for a, b in zip(tr['label'], tr['label'][1:]) if a != b)
        out['markers'].append({
            'kind': 'Ap' if tr['kind'] == 1 else 'Pe',
            'eta_s': tr['t'][0] - t0, 'r_m': float(np.median(tr['r'])),
            'r_span_m': float(max(tr['r']) - min(tr['r'])),
            't_span_s': float(max(tr['t']) - min(tr['t'])),
            'pos_span_soi_px': float(math.hypot(x.max() - x.min(),
                                                y.max() - y.min()) * ZOOM_SOI),
            'label_changes': changes, 'labels': sorted(set(tr['label'])),
            'frames': len(tr['t']),
        })
    return out


def find_extremum(s0, s1):
    """Zeitpunkt, an dem |r| zwischen zwei bildern maximal wird.

    Kubische Hermite ueber die beiden zustaende (dieselbe kurve, die die
    punkteliste beschreibt), nullstelle von r . v per bisektion.
    """
    t0, x0, y0, vx0, vy0 = s0
    t1, x1, y1, vx1, vy1 = s1
    pts = np.array([[x0, y0, t0, vx0, vy0], [x1, y1, t1, vx1, vy1]])
    lo, hi = t0, t1
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        px, py, vx, vy = state_on_curve(pts, mid)
        if (px - sonne.position.x) * vx + (py - sonne.position.y) * vy > 0.0:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    px, py, _vx, _vy = state_on_curve(pts, t)
    return t, px, py


try:
    main()
finally:
    try:
        app.maneuver_preview.shutdown()
    except Exception:
        pass
    pred.close()
    app.window.close()
