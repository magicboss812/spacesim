"""Messläufe zu Abschnitt 4.4 (KI-erstellt, 15.10.2026).

Fliegt in der Weltphysik des Spiels (physics/world.py, adaptives RKN4 mit den
Toleranzen aus config/config.json, alle 27 Himmelskörper auf ihren
Kepler-Bahnen, Monde aktiv) vier Läufe von der Parkbahn um die Erde:

  direkt     Hohmann-Transfer Erde-Saturn
  v1         Hohmann-Transfer zum Jupiter, Vorbeiflug mit Apoapsis auf der Saturnbahn
  v2         wie v1, Periapsis am Jupiter 5 R_J
  kontrolle  Abflug und erste Korrektur wie v1, Jupiter um 60 Grad verschoben

Ablauf je Lauf (wie Abschnitt 3.4): Erde, Jupiter und Saturn auf Kreisbahnen
(Exzentrizität 0, große Halbachsen und Massen wie im Spiel). Die Startwinkel
werden aus der Rechnung mit den Spielparametern so gelegt, dass die Planeten
zum richtigen Zeitpunkt am richtigen Ort stehen. Abflugimpuls nach einer halben
Stunde in der Parkbahn als idealer Impuls. Korrektur 1 am achten Tag in
Flugrichtung (relativ zur Sonne), Betrag durch wiederholtes Nachfliegen so
gewählt, dass die Periapsis am Saturn (direkt) bzw. am Jupiter (v2: 5 R_J) auf
dem Sollwert liegt; in v1 so, dass das Aphel der Bahn nach dem Vorbeiflug auf
der Saturnbahn liegt (Definition von Variante 1). Saturns Startwinkel wird für
v1/v2 aus dem tatsächlich geflogenen Weg gelegt und iteriert. Korrektur 2 (v1, v2) beim Verlassen der Einflusssphäre Jupiters
in Flugrichtung, Betrag so, dass die Periapsis am Saturn auf der Zielbahn liegt.
Einschwenkimpuls aus der Periapsisgeschwindigkeit nach Gl. (32).

Aufruf:  python messlaeufe_4_4.py <pfad zum spacesim-repo> [ausgabe.json]
"""
import json
import math
import os
import pickle
import sys

SP = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '.')
OUT = sys.argv[2] if len(sys.argv) > 2 else 'messlaeufe_4_4.json'
sys.path.insert(0, SP)
os.chdir(SP)

from runtime.bootstrap import load_config          # noqa: E402
from runtime.system_loader import SystemLoader      # noqa: E402
from physics.world import world as World            # noqa: E402
from physics.vec import Vec2                        # noqa: E402

DAY = 86400.0
YEAR = 365.25 * DAY
T_DEP = 1800.0                 # abflug nach einer halben stunde in der parkbahn
T_CORR1 = T_DEP + 8 * DAY      # korrektur 1 am achten tag
R0_E = 6378.1366e3 + 300e3     # parkbahn, abschnitt 3.2
R_J_EQ = 71492e3               # [17, Tab. 9]
R0_SA = 60268e3 + 300e3        # zielbahn am saturn
RP_V2 = 5 * R_J_EQ

CFG = load_config()
G = float(CFG.get('physics.gravitational_constant', 6.6730831e-11))


def load_system():
    bodies = SystemLoader('solar_system.json').load()
    w = World(G)
    w.body = bodies
    CFG.apply_to_world(w)
    return w, {b.name: b for b in bodies}


_w0, _B0 = load_system()
M = {n: _B0[n].mass for n in ('Sonne', 'Erde', 'Jupiter', 'Saturn')}
A = {n: float(_B0[n].semi_major_axis) for n in ('Erde', 'Jupiter', 'Saturn')}
MU_S = G * M['Sonne']
MU = {n: G * M[n] for n in ('Erde', 'Jupiter', 'Saturn')}
N = {n: math.sqrt(G * (M['Sonne']) / A[n] ** 3) for n in A}   # kepler im spiel: mu = G*m_sonne
R_SOI = {n: A[n] * (MU[n] / MU_S) ** 0.4 for n in A}


# ------------------------------------------------------------------ rechnung

def park_impulse(mu, r0, vinf):
    return math.sqrt(2 * mu / r0 + vinf ** 2) - math.sqrt(mu / r0)


def hohmann(r1, r2):
    at = (r1 + r2) / 2
    v1 = math.sqrt(MU_S * (2 / r1 - 1 / at))
    v2 = math.sqrt(MU_S * (2 / r2 - 1 / at))
    return v1 - math.sqrt(MU_S / r1), math.sqrt(MU_S / r2) - v2, math.pi * math.sqrt(at ** 3 / MU_S)


def kepler_time(a, e, nu):
    """zeit seit periapsis bis zur wahren anomalie nu (ellipse)."""
    E = 2 * math.atan2(math.sqrt(1 - e) * math.sin(nu / 2), math.sqrt(1 + e) * math.cos(nu / 2))
    return (E - e * math.sin(E)) / math.sqrt(MU_S / a ** 3)


def after_flyby(delta, vinf):
    vk = math.sqrt(MU_S / A['Jupiter'])
    vperp = vk - vinf * math.cos(delta)
    vr = vinf * math.sin(delta)
    return vperp, vr


def helio_orbit(r, vperp, vr):
    v2 = vperp ** 2 + vr ** 2
    eps = v2 / 2 - MU_S / r
    a = -MU_S / (2 * eps)
    h = r * vperp
    e = math.sqrt(max(0.0, 1 - h * h / (MU_S * a)))
    p = h * h / MU_S
    nu0 = math.atan2(vr * h / MU_S, p / r - 1)       # wahre anomalie am vorbeiflug
    return a, e, h, nu0


def rechnung():
    rE, rJ, rS = A['Erde'], A['Jupiter'], A['Saturn']
    out = {}
    d1, d2, tES = hohmann(rE, rS)
    dvE = park_impulse(MU['Erde'], R0_E, d1)
    dvS = park_impulse(MU['Saturn'], R0_SA, d2)
    out['direkt'] = dict(vinf_E=d1, dv_E=dvE, vinf_Sa=d2, dv_Sa=dvS,
                         summe=dvE + dvS, flugzeit_d=tES / DAY, t_EJ=None)
    j1, j2, tEJ = hohmann(rE, rJ)
    dvE_J = park_impulse(MU['Erde'], R0_E, j1)

    # variante 1: delta so, dass die apoapsis auf der saturnbahn liegt (gl. 42)
    def f(delta):
        vperp, vr = after_flyby(delta, j2)
        return (vperp ** 2 + vr ** 2) / 2 - MU_S / rJ - ((rJ * vperp / rS) ** 2 / 2 - MU_S / rS)
    lo, hi = 1.0, 2.2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    d_v1 = 0.5 * (lo + hi)
    eJ = 1 / math.sin(d_v1 / 2)
    rp_v1 = (eJ - 1) * MU['Jupiter'] / j2 ** 2

    for name, rp in (('v1', rp_v1), ('v2', RP_V2)):
        eJ = 1 + rp * j2 ** 2 / MU['Jupiter']
        delta = 2 * math.asin(1 / eJ)
        vperp, vr = after_flyby(delta, j2)
        a, e, h, nu0 = helio_orbit(rJ, vperp, vr)
        if name == 'v1':
            nu_arr = math.pi
        else:
            nu_arr = math.acos((h * h / MU_S / rS - 1) / e)
        t2 = kepler_time(a, e, nu_arr) - kepler_time(a, e, nu0)
        vperp_S = h / rS
        v_S = math.sqrt(2 * (-MU_S / (2 * a) + MU_S / rS))
        vr_S = math.sqrt(max(0.0, v_S ** 2 - vperp_S ** 2))
        vinf_S = math.hypot(math.sqrt(MU_S / rS) - vperp_S, vr_S)
        dvS = park_impulse(MU['Saturn'], R0_SA, vinf_S)
        out[name] = dict(vinf_E=j1, dv_E=dvE_J, vinf_J=j2, rp_J=rp, e_J=eJ,
                         delta_deg=math.degrees(delta), v_ein=math.sqrt(MU_S / rJ) - j2,
                         v_aus=math.hypot(vperp, vr), dv_vb=2 * j2 / eJ,
                         a=a, e=e, nu0=nu0, nu_arr=nu_arr,
                         vinf_Sa=vinf_S, dv_Sa=dvS, summe=dvE_J + dvS,
                         t_EJ_d=tEJ / DAY, t_JS_d=t2 / DAY, flugzeit_d=(tEJ + t2) / DAY)
    out['direkt']['t_ES'] = tES
    out['_tEJ'] = tEJ
    return out


R = rechnung()


# ------------------------------------------------------------------ aufbau

def wrap(x):
    return math.atan2(math.sin(x), math.cos(x))


def setup(lauf, theta_S=None):
    """welt mit den startwinkeln des laufs; liefert (welt, koerper, winkel)."""
    w, B = load_system()
    for n in ('Erde', 'Jupiter', 'Saturn'):
        B[n].eccentricity = 0.0
    rJ = A['Jupiter']
    th = {}
    th['Erde'] = wrap(0.0 - N['Erde'] * T_DEP)
    if lauf == 'direkt':
        tS = T_DEP + R['direkt']['t_ES']
        th['Saturn'] = wrap(math.pi - N['Saturn'] * tS)
        th['Jupiter'] = float(B['Jupiter'].theta)          # wie im spiel
        vinf = R['direkt']['vinf_E']
    else:
        var = 'v1' if lauf in ('v1', 'kontrolle') else 'v2'
        r = R[var]
        tJ = T_DEP + R['_tEJ']
        th['Jupiter'] = wrap(math.pi - N['Jupiter'] * tJ)
        if lauf == 'kontrolle':
            th['Jupiter'] = wrap(th['Jupiter'] + math.radians(60.0))
        # neue bahn: am jupiter (winkel pi) mit wahrer anomalie nu0
        omega = math.pi - r['nu0']
        ang_arr = omega + r['nu_arr']
        tS = tJ + r['t_JS_d'] * DAY
        th['Saturn'] = wrap(ang_arr - N['Saturn'] * tS)
        vinf = r['vinf_E']
    if theta_S is not None:
        th['Saturn'] = theta_S
    for n, v in th.items():
        B[n].theta = v
        B[n]._kepler_ref_theta = v
    w.time = 0.0
    w.update_planets(0.0)
    # schiff in der parkbahn, so dass es nach T_DEP am zuendpunkt steht
    e_h = 1 + R0_E * vinf ** 2 / MU['Erde']
    nu_inf = math.acos(-1 / e_h)
    phi_p = math.pi / 2 - nu_inf               # erde bei winkel 0 laeuft in +y
    om0 = math.sqrt(MU['Erde'] / R0_E ** 3)
    phi0 = phi_p - om0 * T_DEP
    ship = next(b for b in w.body if b.is_ship)
    E = B['Erde']
    vE = body_vel(E, 0.0)
    ship.position = Vec2(E.position.x + R0_E * math.cos(phi0), E.position.y + R0_E * math.sin(phi0))
    vc = math.sqrt(MU['Erde'] / R0_E)
    ship.velocity = Vec2(vE[0] - vc * math.sin(phi0), vE[1] + vc * math.cos(phi0))
    return w, B, ship, th


def body_pos(b, t):
    p = b.position_at_time(t)
    return (p.x, p.y)


def body_vel(b, t):
    a = b.position_at_time(t - 0.5)
    c = b.position_at_time(t + 0.5)
    return (c.x - a.x, c.y - a.y)


def rel_state(ship, b, t):
    bp = body_pos(b, t)
    bv = body_vel(b, t)
    return (ship.position.x - bp[0], ship.position.y - bp[1],
            ship.velocity.x - bv[0], ship.velocity.y - bv[1])


def osculating(rx, ry, vx, vy, mu):
    r = math.hypot(rx, ry)
    v2 = vx * vx + vy * vy
    eps = v2 / 2 - mu / r
    hz = rx * vy - ry * vx
    h = abs(hz)
    e = math.sqrt(max(0.0, 1 + 2 * eps * h * h / mu ** 2))
    rp = h * h / mu / (1 + e)
    # exzentrizitaetsvektor
    ex = ((v2 - mu / r) * rx - (rx * vx + ry * vy) * vx) / mu
    ey = ((v2 - mu / r) * ry - (rx * vx + ry * vy) * vy) / mu
    return dict(r=r, eps=eps, hz=hz, h=h, e=e, rp=rp, vp=h / rp if rp > 0 else float('nan'),
                ex=ex, ey=ey)


def asymptote_v(osc, mu, sign):
    """v-infinity-vektor der oskulierenden hyperbel; sign=-1 ein, +1 aus."""
    e = osc['e']
    nu = sign * math.acos(-1 / e)
    px, py = osc['ex'] / e, osc['ey'] / e
    s = 1.0 if osc['hz'] > 0 else -1.0
    qx, qy = -s * py, s * px
    k = mu / osc['h']
    return (k * (-math.sin(nu) * px + (e + math.cos(nu)) * qx),
            k * (-math.sin(nu) * py + (e + math.cos(nu)) * qy))


def step(w, dt):
    w.step(dt, 1000.0)


def advance(w, t_end, chunk=DAY):
    while w.time < t_end - 1e-6:
        step(w, min(chunk, t_end - w.time))


def chunk_near(w, ship, b):
    rx, ry, vx, vy = rel_state(ship, b, w.time)
    d = math.hypot(rx, ry)
    v = max(math.hypot(vx, vy), 1.0)
    return max(300.0, min(DAY, d / v / 200.0)), d


def snapshot(w, ship):
    s = {'time': w.time, 'hint': w._integrator_h_hint, 'bodies': []}
    for b in w.body:
        s['bodies'].append((b.position.x, b.position.y, b.velocity.x, b.velocity.y,
                            b.theta, b._kepler_ref_theta, b._kepler_ref_time))
    return s


def restore(w, s):
    w.time = s['time']
    w._integrator_h_hint = s['hint']
    for b, (px, py, vx, vy, th, krt, krtime) in zip(w.body, s['bodies']):
        b.position = Vec2(px, py)
        b.velocity = Vec2(vx, vy)
        b.theta, b._kepler_ref_theta, b._kepler_ref_time = th, krt, krtime


def prograde_kick(ship, B, dv, t):
    sx, sy = ship.velocity.x, ship.velocity.y
    n = math.hypot(sx, sy)
    ship.velocity = Vec2(sx + dv * sx / n, sy + dv * sy / n)


def kick_rel(ship, b, dv, t):
    rx, ry, vx, vy = rel_state(ship, b, t)
    n = math.hypot(vx, vy)
    ship.velocity = Vec2(ship.velocity.x + dv * vx / n, ship.velocity.y + dv * vy / n)


def fly_to_periapsis(w, ship, b, t_max, after_exit=False, log=None, moons=()):
    """bis zur naechsten periapsis an b. liefert dict (oskulierend am minimum).
    after_exit: danach weiter bis zum verlassen der einflusssphaere."""
    mu = G * b.mass
    rsoi = R_SOI[b.name]
    best = None
    entry = None
    exit_ = None
    prev_d = None
    moon_min = {m.name: (float('inf'), None) for m in moons}
    while w.time < t_max:
        ch, d = chunk_near(w, ship, b)
        if entry is None and d < rsoi:
            rs = rel_state(ship, b, w.time)
            entry = dict(t=w.time, osc=osculating(*rs, mu), rel=rs,
                         helio=(ship.velocity.x, ship.velocity.y, ship.position.x, ship.position.y))
        if best is None or d < best['d']:
            rs = rel_state(ship, b, w.time)
            best = dict(d=d, t=w.time, osc=osculating(*rs, mu), snap=None)
        for m in moons:
            mp = body_pos(m, w.time)
            dm = math.hypot(ship.position.x - mp[0], ship.position.y - mp[1])
            if dm < moon_min[m.name][0]:
                moon_min[m.name] = (dm, w.time)
        if prev_d is not None and d > prev_d and best is not None and d > best['d'] * 1.05:
            if not after_exit:
                break
            if d > rsoi:
                rs = rel_state(ship, b, w.time)
                exit_ = dict(t=w.time, osc=osculating(*rs, mu), rel=rs,
                             helio=(ship.velocity.x, ship.velocity.y, ship.position.x, ship.position.y))
                break
        prev_d = d
        if log is not None:
            log.append((w.time, ship.position.x, ship.position.y))
        step(w, ch)
    # periapsiszeit verfeinern: zeit bis zur periapsis aus der oskulierenden bahn am minimum
    if best is not None:
        o = best['osc']
        best['rp'] = o['rp']
        best['vp'] = o['vp']
        best['signed_rp'] = o['rp'] * (1 if o['hz'] > 0 else -1)
        best['t_peri'] = best['t'] + time_to_peri(o, mu)
    return best, entry, exit_, moon_min


def time_to_peri(o, mu):
    """zeit vom zustand bis zur periapsis der oskulierenden hyperbel (vorzeichenbehaftet)."""
    e = o['e']
    if e <= 1.0:
        return 0.0
    r = o['r']
    a = o['h'] ** 2 / mu / (e * e - 1)
    coshF = (1 + r / a) / e
    F = math.acosh(max(1.0, coshF))
    # vorzeichen: annaeherung, wenn r*v < 0 -- hier nur betrag am minimum, klein
    return 0.0 if F < 1e-9 else 0.0


def secant(fun, x0, x1, tol, max_it=30, label=''):
    f0 = fun(x0)
    f1 = fun(x1)
    hist = [(x0, f0), (x1, f1)]
    for _ in range(max_it):
        if abs(f1) < tol:
            break
        if f1 == f0:
            x1 += 0.01
            f1 = fun(x1)
            continue
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        # daempfung gegen ausreisser
        if abs(x2 - x1) > 20.0:
            x2 = x1 + math.copysign(20.0, x2 - x1)
        x0, f0 = x1, f1
        x1 = x2
        f1 = fun(x1)
        hist.append((x1, f1))
        print(f'   {label} c={x1:+.6f} m/s  f={f1/1e3:+.3f} km', flush=True)
    return x1, f1, hist


# ------------------------------------------------------------------ laeufe

def run(lauf, c1=None, c2=None):
    w, B, ship, th = setup(lauf)
    J, S = B['Jupiter'], B['Saturn']
    moons = [B[n] for n in ('Io', 'Europa', 'Ganymed', 'Kallisto')]
    res = {'lauf': lauf, 'theta0': {k: v for k, v in th.items()}}
    advance(w, T_DEP, chunk=300.0)
    var = 'direkt' if lauf == 'direkt' else ('v2' if lauf == 'v2' else 'v1')
    dvE = R[var]['dv_E']
    kick_rel(ship, B['Erde'], dvE, w.time)
    res['dv_E'] = dvE
    advance(w, T_CORR1, chunk=3600.0)
    s_c1 = snapshot(w, ship)

    if lauf == 'direkt':
        def f1(c):
            restore(w, s_c1)
            prograde_kick(ship, B, c, w.time)
            best, *_ = fly_to_periapsis(w, ship, S, T_DEP + 4000 * DAY)
            return best['signed_rp'] - sign_target[0] * R0_SA
        sign_target = [1.0]
        restore(w, s_c1)
        b0, *_ = fly_to_periapsis(w, ship, S, T_DEP + 4000 * DAY)
        sign_target[0] = 1.0 if b0['signed_rp'] > 0 else -1.0
        if c1 is None:
            c1, _, _ = secant(f1, 0.0, 1.0, 1e3, label='direkt K1')
        restore(w, s_c1)
        prograde_kick(ship, B, c1, w.time)
        best, entry, _, _ = fly_to_periapsis(w, ship, S, T_DEP + 4000 * DAY)
        res.update(c1=c1, c2=0.0, saturn=pack(best, entry, None, MU['Saturn']))
        finish(res, best)
        return res

    target = R[var]['rp_J']
    if lauf == 'kontrolle':
        restore(w, s_c1)
        prograde_kick(ship, B, c1, w.time)
        log = []
        t_end = T_DEP + 1800 * DAY
        rmax = (0.0, 0.0)
        while w.time < t_end:
            step(w, DAY)
            r = math.hypot(ship.position.x, ship.position.y)
            if r > rmax[0]:
                rmax = (r, w.time)
            log.append((w.time, ship.position.x, ship.position.y))
        res.update(c1=c1, r_max=rmax[0], t_rmax_d=(rmax[1] - T_DEP) / DAY,
                   min_dist_jupiter=min(math.hypot(x - body_pos(J, t)[0], y - body_pos(J, t)[1])
                                        for t, x, y in log))
        return res

    theta_S = None
    c1_guess = c1 if c1 is not None else 0.0
    c2_start = 0.0
    for it in range(3):
        w, B, ship, th = setup(lauf, theta_S)
        J, S = B['Jupiter'], B['Saturn']
        moons = [B[n] for n in ('Io', 'Europa', 'Ganymed', 'Kallisto')]
        advance(w, T_DEP, chunk=300.0)
        kick_rel(ship, B['Erde'], dvE, w.time)
        advance(w, T_CORR1, chunk=3600.0)
        s_c1 = snapshot(w, ship)

        def f1(c):
            restore(w, s_c1)
            prograde_kick(ship, B, c, w.time)
            best, *_ = fly_to_periapsis(w, ship, J, T_DEP + 1500 * DAY)
            return best['signed_rp'] - target

        def f1_aphel(c):
            # variante 1 in der simulation: periapsis am jupiter so, dass das
            # aphel der neuen bahn auf der saturnbahn liegt (saturn masselos)
            restore(w, s_c1)
            prograde_kick(ship, B, c, w.time)
            fly_to_periapsis(w, ship, J, T_DEP + 1500 * DAY, after_exit=True)
            mS0 = S.mass
            S.mass = 0.0
            r = probe(w, ship, 'max')[2]
            S.mass = mS0
            return r - A['Saturn']
        if lauf == 'v1':
            c1a, f1a = (-25.805417, None) if it == 0 else (c1_guess, None)
            c1, _, _ = secant(f1_aphel, c1a, c1a - 0.05, 2e8, label=f'v1 K1 aphel (it {it})')
        else:
            c1, _, _ = secant(f1, c1_guess, c1_guess + 0.01, 2e3, label=f'{lauf} K1 (it {it})')
        c1_guess = c1
        restore(w, s_c1)
        prograde_kick(ship, B, c1, w.time)
        bJ, eJ, xJ, mm = fly_to_periapsis(w, ship, J, T_DEP + 1500 * DAY, after_exit=True, moons=moons)
        s_c2 = snapshot(w, ship)
        mS = S.mass
        S.mass = 0.0
        restore(w, s_c2)
        prograde_kick(ship, B, c2_start, w.time)
        t_x, ang_x, r_x = probe(w, ship, 'max' if lauf == 'v1' else 'cross')
        S.mass = mS
        theta_new = wrap(ang_x - N['Saturn'] * t_x)
        print(f'   {lauf} it {it}: kreuzung t={(t_x-T_DEP)/DAY:.2f} d, r={r_x/1e9:.4f} Gm, '
              f'theta0_Saturn={theta_new:.6f} (vorher {th["Saturn"]:.6f})', flush=True)
        if theta_S is not None and abs(wrap(theta_new - theta_S)) < 1e-6:
            break
        theta_S = theta_new
    res['theta0'] = dict(th)
    res['c1'] = c1
    res['jupiter'] = pack(bJ, eJ, xJ, MU['Jupiter'])
    res['monde'] = {k: dict(min_abstand_km=v[0] / 1e3, t_rel_peri_h=(v[1] - bJ['t']) / 3600.0)
                    for k, v in mm.items()}
    res['monde_bei_peri'] = {}
    jp = body_pos(J, bJ['t'])
    for m in moons:
        mp = body_pos(m, bJ['t'])
        res['monde_bei_peri'][m.name] = dict(
            abstand_jupiter_km=math.hypot(mp[0] - jp[0], mp[1] - jp[1]) / 1e3,
            winkel_deg=math.degrees(math.atan2(mp[1] - jp[1], mp[0] - jp[0])))

    def f2(c):
        restore(w, s_c2)
        prograde_kick(ship, B, c, w.time)
        best, *_ = fly_to_periapsis(w, ship, S, T_DEP + 4500 * DAY)
        return best['signed_rp'] - sgn[0] * R0_SA
    sgn = [1.0]
    restore(w, s_c2)
    prograde_kick(ship, B, c2_start, w.time)
    b0, *_ = fly_to_periapsis(w, ship, S, T_DEP + 4500 * DAY)
    sgn[0] = 1.0 if b0['signed_rp'] > 0 else -1.0
    res['saturn_start_rp_km'] = b0['signed_rp'] / 1e3
    if c2 is None:
        c2, _, _ = secant(f2, c2_start, c2_start + 0.01, 1e3, label=f'{lauf} K2')
    restore(w, s_c2)
    prograde_kick(ship, B, c2, w.time)
    bS, eS, _, _ = fly_to_periapsis(w, ship, S, T_DEP + 4500 * DAY)
    res['c2'] = c2
    res['saturn'] = pack(bS, eS, None, MU['Saturn'])
    finish(res, bS)
    return res


def probe(w, ship, mode):
    """schiff ohne saturn weiterfliegen: 'max' -> groesster sonnenabstand,
    'cross' -> erstes erreichen der saturnbahn. liefert (t, winkel, r)."""
    prev = None
    t_end = w.time + 4000 * DAY
    while w.time < t_end:
        x, y = ship.position.x, ship.position.y
        r = math.hypot(x, y)
        cur = (w.time, math.atan2(y, x), r)
        if mode == 'max' and prev is not None and r < prev[2]:
            return prev
        if mode == 'cross' and r >= A['Saturn']:
            if prev is None:
                return cur
            f = (A['Saturn'] - prev[2]) / (r - prev[2])
            da = wrap(cur[1] - prev[1])
            return (prev[0] + f * (cur[0] - prev[0]), prev[1] + f * da, A['Saturn'])
        prev = cur
        step(w, DAY / 4)
    return prev


def pack(best, entry, exit_, mu):
    d = dict(t_peri_d=(best['t'] - T_DEP) / DAY, rp_km=best['rp'] / 1e3,
             signed_rp_km=best['signed_rp'] / 1e3, vp=best['vp'],
             vinf_peri=math.sqrt(max(0.0, 2 * best['osc']['eps'])),
             min_sample_d_km=best['d'] / 1e3)
    if entry is not None:
        o = entry['osc']
        d['vinf_ein'] = math.sqrt(max(0.0, 2 * o['eps']))
        d['v_helio_ein'] = math.hypot(entry['helio'][0], entry['helio'][1])
        d['eps_helio_ein'] = d['v_helio_ein'] ** 2 / 2 - MU_S / math.hypot(entry['helio'][2], entry['helio'][3])
        d['t_ein_d'] = (entry['t'] - T_DEP) / DAY
        if o['e'] > 1:
            d['_vin_vec'] = asymptote_v(o, mu, -1)
    if exit_ is not None:
        o = exit_['osc']
        d['vinf_aus'] = math.sqrt(max(0.0, 2 * o['eps']))
        d['v_helio_aus'] = math.hypot(exit_['helio'][0], exit_['helio'][1])
        d['eps_helio_aus'] = d['v_helio_aus'] ** 2 / 2 - MU_S / math.hypot(exit_['helio'][2], exit_['helio'][3])
        d['t_aus_d'] = (exit_['t'] - T_DEP) / DAY
        if o['e'] > 1:
            d['_vout_vec'] = asymptote_v(o, mu, +1)
        if '_vin_vec' in d and '_vout_vec' in d:
            a, b = d['_vin_vec'], d['_vout_vec']
            cosd = (a[0] * b[0] + a[1] * b[1]) / (math.hypot(*a) * math.hypot(*b))
            d['delta_deg'] = math.degrees(math.acos(max(-1, min(1, cosd))))
            d['dv_vb'] = math.hypot(b[0] - a[0], b[1] - a[1])
        # heliozentrische bahnenergie nach dem austritt, umgerechnet auf v am jupiterbahnradius
        d['v_aus_aequivalent'] = math.sqrt(2 * (d['eps_helio_aus'] + MU_S / A['Jupiter']))
        d['v_ein_aequivalent'] = math.sqrt(2 * (d['eps_helio_ein'] + MU_S / A['Jupiter']))
    for k in [k for k in d if k.startswith('_')]:
        d[k] = list(d[k])
    return d


def finish(res, bS):
    vk0 = math.sqrt(MU['Saturn'] / R0_SA)
    dvS = bS['vp'] - vk0
    res['dv_Sa'] = dvS
    res['summe'] = res['dv_E'] + abs(res['c1']) + abs(res.get('c2', 0.0)) + dvS
    res['flugzeit_d'] = (bS['t'] - T_DEP) / DAY


if __name__ == '__main__':
    which = os.environ.get('LAEUFE', 'direkt,v1,v2,kontrolle').split(',')
    out = {'rechnung_spielparameter': R, 'spielparameter': dict(
        G=G, m=M, a=A, mu_sonne=MU_S, mu=MU, r_soi=R_SOI, r0_E=R0_E, r0_Sa=R0_SA,
        rp_v2=RP_V2, pos_tol=_w0.integrator_position_tolerance,
        vel_tol=_w0.integrator_velocity_tolerance, max_step=_w0.integrator_max_step)}
    if os.path.exists(OUT):
        out.update(json.load(open(OUT)))
    for lauf in which:
        print('==', lauf, flush=True)
        kw = {}
        if lauf == 'kontrolle':
            kw['c1'] = out.get('v1', {}).get('c1', 0.0)
        out[lauf] = run(lauf, **kw)
        json.dump(out, open(OUT, 'w'), indent=1, default=float)
        print(json.dumps({k: v for k, v in out[lauf].items() if not isinstance(v, dict)}, default=float), flush=True)
