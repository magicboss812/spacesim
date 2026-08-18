"""Numba-fassung des welt-integrators. WORTGLEICH zur python-fassung.

WARUM ES DIESE DATEI GIBT. `world.update_dynamics` rueckt die freien koerper
in schritten von hoechstens `integrator_max_step` (30 s) vor. Bei hohem
zeitraffer ist das der teuerste teil des ganzen spiels und waechst LINEAR
mit der raffung:

    7d/s  = 604800 sim-s je echtsekunde
          = 10080 sim-s je frame bei 60 fps
          = 336 adaptive schritte je frame

Jeder dieser schritte macht schritt-verdopplung (drei RKN4-auswertungen zu je
vier beschleunigungen), jede beschleunigung laeuft ueber alle koerper und legt
dabei Vec2-objekte an -- gemessen 47.4 ms je frame allein hierfuer, gegenueber
0.17 ms bei 1m/s. Das ist die ursache der 1-2 fps bei hoher raffung, nicht der
predictor (der lag im selben messlauf bei 5.6 ms) und nicht das zeichnen.

WAS HIER NICHT PASSIERT: die genauigkeit wird NICHT angetastet. Es sind
dieselben formeln, dieselben koeffizienten, dieselbe schrittsteuerung,
dieselben toleranzen und dieselbe summationsreihenfolge wie in world.py --
nur ohne Python-objekte. `tests/energy_test.py` muss danach exakt dieselben
werte liefern; tut es das nicht, ist die uebertragung falsch, nicht "etwas
ungenauer".

ZWEI STELLEN, AN DENEN MAN LEICHT DANEBENGREIFT:

1. **Nicht die predictor-kernel wiederverwenden.** `predictor.py` propagiert
   die scripted-koerper mit einer echten Kepler-loesung (mittlere anomalie +
   Newton-iteration). `bodies.position_at_time`, das der welt-integrator
   benutzt, rechnet statt dessen mit KONSTANTER winkelgeschwindigkeit
   (`theta_t = theta_ref + omega_ref * dt`). Die beiden modelle stimmen nicht
   ueberein. Hier wird bewusst das der welt nachgebildet -- sonst aendert
   sich die physik, und genau das war ausgeschlossen.
2. **Die summe laeuft in koerper-reihenfolge.** Gleitkomma-addition ist nicht
   assoziativ; eine andere reihenfolge gibt andere letzte bits und damit eine
   andere energiedrift.
"""

import math

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:                                    # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):                       # type: ignore
        def wrap(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return wrap


# Wie tief `position_at_time` durch is_moon_of-ketten laeuft (Mond -> Erde ->
# Sonne sind drei). Die schranke ersetzt die rekursion der python-fassung und
# schuetzt zugleich vor einem zyklischen is_moon_of.
_MAX_PARENT_DEPTH = 16


@njit(cache=True, nogil=True, fastmath=False)
def _body_pos_at_time(index, t, bx, by, k_has, k_a, k_e, k_arg, k_parent,
                      k_ref_theta, k_ref_time, k_mu):
    """Nachbau von bodies.body.position_at_time -- iterativ statt rekursiv.

    Ein koerper OHNE gueltige bahn liefert seine gespeicherte position; ein
    koerper MIT bahn liefert seine relativposition plus die position seines
    mutterkoerpers zur selben zeit. Genau diese kette wird hier abgelaufen.
    """
    acc_x = 0.0
    acc_y = 0.0
    idx = index
    for _ in range(_MAX_PARENT_DEPTH):
        if k_has[idx] == 0:
            return acc_x + bx[idx], acc_y + by[idx]

        a = k_a[idx]
        e = k_e[idx]
        mu = k_mu[idx]

        ref_theta = k_ref_theta[idx]
        delta_t = t - k_ref_time[idx]

        # Konstante winkelgeschwindigkeit aus vis-viva am referenzpunkt --
        # bewusst dieselbe naeherung wie in bodies.py, siehe modulkopf.
        r_ref = a * (1.0 - e * e) / (1.0 + e * math.cos(ref_theta))
        v_ref = math.sqrt(max(0.0, mu * (2.0 / r_ref - 1.0 / a)))
        omega_ref = v_ref / max(1e-12, r_ref)
        theta_t = ref_theta + omega_ref * delta_t

        r_t = a * (1.0 - e * e) / (1.0 + e * math.cos(theta_t))
        x_orb = r_t * math.cos(theta_t)
        y_orb = r_t * math.sin(theta_t)

        c = math.cos(k_arg[idx])
        s = math.sin(k_arg[idx])
        acc_x += x_orb * c - y_orb * s
        acc_y += x_orb * s + y_orb * c

        idx = k_parent[idx]
        if idx < 0:
            return acc_x, acc_y

    return acc_x + bx[idx], acc_y + by[idx]


@njit(cache=True, nogil=True, fastmath=False)
def _acceleration_at(target, px, py, t, bx, by, bm, k_has, k_a, k_e, k_arg,
                     k_parent, k_ref_theta, k_ref_time, k_mu, G):
    """Nachbau von world.acceleration_at. Reihenfolge = koerper-reihenfolge."""
    ax = 0.0
    ay = 0.0
    for j in range(bx.shape[0]):
        if j == target:
            continue
        ox, oy = _body_pos_at_time(j, t, bx, by, k_has, k_a, k_e, k_arg,
                                   k_parent, k_ref_theta, k_ref_time, k_mu)
        dx = ox - px
        dy = oy - py
        r2 = dx * dx + dy * dy
        if r2 < 1e-10:
            continue
        r = math.sqrt(r2)
        f = G * bm[j] / (r2 * r)
        ax += dx * f
        ay += dy * f
    return ax, ay


@njit(cache=True, nogil=True, fastmath=False)
def _rkn4_step(target, px, py, vx, vy, t0, h, bx, by, bm, k_has, k_a, k_e,
               k_arg, k_parent, k_ref_theta, k_ref_time, k_mu, G):
    """world._rkn4_step_body_state."""
    a1x, a1y = _acceleration_at(target, px, py, t0, bx, by, bm, k_has, k_a,
                                k_e, k_arg, k_parent, k_ref_theta, k_ref_time,
                                k_mu, G)

    hh = h * h
    p2x = px + vx * (h * 0.5) + a1x * (hh * 0.125)
    p2y = py + vy * (h * 0.5) + a1y * (hh * 0.125)
    a2x, a2y = _acceleration_at(target, p2x, p2y, t0 + h * 0.5, bx, by, bm,
                                k_has, k_a, k_e, k_arg, k_parent, k_ref_theta,
                                k_ref_time, k_mu, G)

    p3x = px + vx * (h * 0.5) + a2x * (hh * 0.125)
    p3y = py + vy * (h * 0.5) + a2y * (hh * 0.125)
    a3x, a3y = _acceleration_at(target, p3x, p3y, t0 + h * 0.5, bx, by, bm,
                                k_has, k_a, k_e, k_arg, k_parent, k_ref_theta,
                                k_ref_time, k_mu, G)

    p4x = px + vx * h + a3x * (hh * 0.5)
    p4y = py + vy * h + a3y * (hh * 0.5)
    a4x, a4y = _acceleration_at(target, p4x, p4y, t0 + h, bx, by, bm, k_has,
                                k_a, k_e, k_arg, k_parent, k_ref_theta,
                                k_ref_time, k_mu, G)

    new_px = px + vx * h + (a1x + a2x + a3x) * (hh / 6.0)
    new_py = py + vy * h + (a1y + a2y + a3y) * (hh / 6.0)
    new_vx = vx + (a1x + 2.0 * a2x + 2.0 * a3x + a4x) * (h / 6.0)
    new_vy = vy + (a1y + 2.0 * a2y + 2.0 * a3y + a4y) * (h / 6.0)
    return new_px, new_py, new_vx, new_vy


@njit(cache=True, nogil=True, fastmath=False)
def _verlet_step(target, px, py, vx, vy, t0, h, bx, by, bm, k_has, k_a, k_e,
                 k_arg, k_parent, k_ref_theta, k_ref_time, k_mu, G):
    """world._verlet_step_body_state (Stoermer-Verlet, KDK)."""
    a0x, a0y = _acceleration_at(target, px, py, t0, bx, by, bm, k_has, k_a,
                                k_e, k_arg, k_parent, k_ref_theta, k_ref_time,
                                k_mu, G)
    p1x = px + vx * h + a0x * (0.5 * h * h)
    p1y = py + vy * h + a0y * (0.5 * h * h)
    a1x, a1y = _acceleration_at(target, p1x, p1y, t0 + h, bx, by, bm, k_has,
                                k_a, k_e, k_arg, k_parent, k_ref_theta,
                                k_ref_time, k_mu, G)
    v1x = vx + (a0x + a1x) * (0.5 * h)
    v1y = vy + (a0y + a1y) * (0.5 * h)
    return p1x, p1y, v1x, v1y


@njit(cache=True, nogil=True, fastmath=False)
def _step_once(mode, target, px, py, vx, vy, t0, h, bx, by, bm, k_has, k_a,
               k_e, k_arg, k_parent, k_ref_theta, k_ref_time, k_mu, G):
    if mode == 1:
        return _verlet_step(target, px, py, vx, vy, t0, h, bx, by, bm, k_has,
                            k_a, k_e, k_arg, k_parent, k_ref_theta,
                            k_ref_time, k_mu, G)
    return _rkn4_step(target, px, py, vx, vy, t0, h, bx, by, bm, k_has, k_a,
                      k_e, k_arg, k_parent, k_ref_theta, k_ref_time, k_mu, G)


@njit(cache=True, nogil=True, fastmath=False)
def advance_dynamics(dyn, dyn_px, dyn_py, dyn_vx, dyn_vy, t_start, total_dt,
                     bx, by, bm, k_has, k_a, k_e, k_arg, k_parent,
                     k_ref_theta, k_ref_time, k_mu, G, mode, max_step,
                     min_step, pos_tol, vel_tol, h_hint):
    """world.update_dynamics -- die aeussere schrittsteuerung, 1:1.

    `bx`/`by` werden bei jedem ANGENOMMENEN schritt fuer die freien koerper
    nachgezogen: die python-fassung schreibt b.position ebenfalls erst nach
    `accepted_all`, waehrend eines versuchs sehen sich die freien koerper
    also gegenseitig noch am alten ort.

    `h_hint` ist die zuletzt ANGENOMMENE schrittweite (0 = keine). Ohne sie
    beginnt jeder aeussere durchlauf wieder bei `max_step` und muss sich per
    ablehnung erneut nach unten arbeiten -- gemessen in einem 2000-km-orbit bei
    1 y/s: 3794 angenommene gegen 19810 abgelehnte schritte, 629 ms je frame.
    Mit dem hinweis faengt der naechste durchlauf dort an, wo der letzte
    aufgehoert hat, und waechst nach einem erfolg wieder um das doppelte.

    Rueckgabe: (substeps, rejections, forced, worst_pos_err, worst_vel_err,
                h_hint_out).
    """
    n = dyn.shape[0]
    substeps = 0
    rejections = 0
    forced = 0
    worst_pos_all = 0.0
    worst_vel_all = 0.0

    direction = 1.0 if total_dt >= 0.0 else -1.0
    remaining = abs(total_dt)
    t = t_start

    try_px = np.empty(n, dtype=np.float64)
    try_py = np.empty(n, dtype=np.float64)
    try_vx = np.empty(n, dtype=np.float64)
    try_vy = np.empty(n, dtype=np.float64)

    hint = max_step if h_hint <= 0.0 else min(h_hint, max_step)

    while remaining > 1e-12:
        h = min(hint, remaining) * direction

        while True:
            accepted_all = True
            worst_pos = 0.0
            worst_vel = 0.0

            for i in range(n):
                target = dyn[i]
                p0x = dyn_px[i]
                p0y = dyn_py[i]
                v0x = dyn_vx[i]
                v0y = dyn_vy[i]

                fpx, fpy, fvx, fvy = _step_once(
                    mode, target, p0x, p0y, v0x, v0y, t, h, bx, by, bm, k_has,
                    k_a, k_e, k_arg, k_parent, k_ref_theta, k_ref_time, k_mu, G)

                half = h * 0.5
                h1px, h1py, h1vx, h1vy = _step_once(
                    mode, target, p0x, p0y, v0x, v0y, t, half, bx, by, bm,
                    k_has, k_a, k_e, k_arg, k_parent, k_ref_theta, k_ref_time,
                    k_mu, G)
                h2px, h2py, h2vx, h2vy = _step_once(
                    mode, target, h1px, h1py, h1vx, h1vy, t + half, half, bx,
                    by, bm, k_has, k_a, k_e, k_arg, k_parent, k_ref_theta,
                    k_ref_time, k_mu, G)

                pos_err = math.sqrt((h2px - fpx) * (h2px - fpx)
                                    + (h2py - fpy) * (h2py - fpy))
                vel_err = math.sqrt((h2vx - fvx) * (h2vx - fvx)
                                    + (h2vy - fvy) * (h2vy - fvy))

                if pos_err > worst_pos:
                    worst_pos = pos_err
                if vel_err > worst_vel:
                    worst_vel = vel_err

                accepted = pos_err <= pos_tol and vel_err <= vel_tol

                try_px[i] = h2px
                try_py[i] = h2py
                try_vx[i] = h2vx
                try_vy[i] = h2vy

                # Wie in python: beim ersten abgelehnten koerper wird die
                # schleife verlassen, worst_* zaehlt also nur bis dorthin.
                if (not accepted) and abs(h) > min_step:
                    accepted_all = False
                    break

            if worst_pos > worst_pos_all:
                worst_pos_all = worst_pos
            if worst_vel > worst_vel_all:
                worst_vel_all = worst_vel

            if accepted_all:
                for i in range(n):
                    dyn_px[i] = try_px[i]
                    dyn_py[i] = try_py[i]
                    dyn_vx[i] = try_vx[i]
                    dyn_vy[i] = try_vy[i]
                    bx[dyn[i]] = try_px[i]
                    by[dyn[i]] = try_py[i]
                substeps += 1
                t += h
                remaining -= abs(h)
                hint = min(abs(h) * 2.0, max_step)
                break

            rejections += 1
            h *= 0.5

            if abs(h) <= min_step:
                # Erzwungener schritt mit der minimalen weite -- ohne
                # fehlerpruefung, genau wie in der python-fassung.
                for i in range(n):
                    target = dyn[i]
                    npx, npy, nvx, nvy = _step_once(
                        mode, target, dyn_px[i], dyn_py[i], dyn_vx[i],
                        dyn_vy[i], t, h, bx, by, bm, k_has, k_a, k_e, k_arg,
                        k_parent, k_ref_theta, k_ref_time, k_mu, G)
                    try_px[i] = npx
                    try_py[i] = npy
                    try_vx[i] = nvx
                    try_vy[i] = nvy
                for i in range(n):
                    dyn_px[i] = try_px[i]
                    dyn_py[i] = try_py[i]
                    dyn_vx[i] = try_vx[i]
                    dyn_vy[i] = try_vy[i]
                    bx[dyn[i]] = try_px[i]
                    by[dyn[i]] = try_py[i]
                substeps += 1
                forced += 1
                t += h
                remaining -= abs(h)
                hint = min(abs(h) * 2.0, max_step)
                break

    return substeps, rejections, forced, worst_pos_all, worst_vel_all, hint
