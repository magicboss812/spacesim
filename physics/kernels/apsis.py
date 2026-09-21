"""Ap/Pe-suche auf einer fertigen punktreihe.

Die marker sitzen auf den EXTREMA des abstands zum bezugskoerper, nicht auf
stuetzstellen: `_refine_apsis_numba` legt eine parabel durch die drei punkte um
das gefundene minimum/maximum und nimmt deren scheitel. Ohne das springt der
marker um eine ganze stuetzweite, sobald sich die abtastung verschiebt.

`tests/warp_predictor_test.py` §11 misst sie gegen eine analytische
e=0.5-ellipse (8.0002e6 gegen 8.0e6 und 2.4002e7 gegen 2.4e7).
"""
import math

import numpy as np
from numba import njit

from physics.kernels.kepler import _body_position_at_time_numba


@njit(cache=True, nogil=True, fastmath=True)
def _refine_apsis_numba(pts, d2_arr, idx, use_tangents):
    # parabolische verfeinerung des diskreten extremums bei `idx`: die
    # rohe "nächster punkt"-wahl hat einen quantisierungsfehler von der
    # größenordnung (punktabstand)^2 / (2*krümmungsradius), der mit dem
    # phasenversatz des abtastrasters zur wahren apsis schwankt. fit einer
    # parabel auf d^2 durch die drei punkte um idx liefert den scheitel.
    n = pts.shape[0]
    x = pts[idx, 0]
    y = pts[idx, 1]
    t = pts[idx, 2]
    r = math.sqrt(d2_arr[idx])
    if idx <= 0 or idx >= n - 1:
        return x, y, t, r

    d2_m = d2_arr[idx - 1]
    d2_0 = d2_arr[idx]
    d2_p = d2_arr[idx + 1]
    if not (math.isfinite(d2_m) and math.isfinite(d2_0) and math.isfinite(d2_p)):
        return x, y, t, r

    denom = d2_m - 2.0 * d2_0 + d2_p
    # denom ~ 0 heißt lokal (fast) linear/flach — keine verlässliche
    # scheitel-schätzung möglich, roher punkt bleibt bestehen.
    if not math.isfinite(denom) or abs(denom) < 1e-12 * max(d2_0, 1.0):
        return x, y, t, r

    k = 0.5 * (d2_m - d2_p) / denom
    if k > 0.5:
        k = 0.5
    elif k < -0.5:
        k = -0.5

    refined_d2 = d2_0 - 0.25 * (d2_m - d2_p) * k
    if not math.isfinite(refined_d2) or refined_d2 < 0.0:
        return x, y, t, r
    r = math.sqrt(refined_d2)

    # POSITION AUF DERSELBEN KUBIK, DIE DER RENDERER ZEICHNET -- nicht auf
    # ihrer SEHNE, die auf einem langen horizont um die pfeilhoehe (viele
    # pixel) daneben liegt. Bezier-form des Hermite-polynoms, wortgleich zu
    # `_hermite_refine_world` im renderer:
    #   b0 = p0, b1 = p0 + v0*dt/3, b2 = p1 - v1*dt/3, b3 = p1
    if k >= 0.0:
        i0 = idx
        i1 = idx + 1
        s = k
    else:
        i0 = idx - 1
        i1 = idx
        s = 1.0 + k

    t0 = pts[i0, 2]
    dt = pts[i1, 2] - t0
    t = t0 + dt * s

    x = pts[i0, 0] + (pts[i1, 0] - pts[i0, 0]) * s
    y = pts[i0, 1] + (pts[i1, 1] - pts[i0, 1]) * s

    # Ohne endliche tangenten an BEIDEN enden gibt es kein polynom -- die
    # sehnen-kernel (ASPI, blankes RK4) schreiben dort absichtlich NaN,
    # und ihre punkte werden auch gezeichnet wie eine gerade. Dann bleibt
    # es bei der linearen form oben, und das ist wieder genau richtig.
    #
    # OB DAS SO IST, WIRD DRAUSSEN ENTSCHIEDEN UND ALS `use_tangents`
    # HEREINGEREICHT -- hier laesst es sich nicht pruefen: unter
    # `fastmath=True` (LLVM `nnan`) liefern `math.isfinite(nan)` und
    # `nan == nan` beide True, ein NaN-guard hier waere wirkungslos.
    if use_tangents != 0 and pts.shape[1] >= 5 and dt > 0.0:
        third = dt / 3.0
        b0x = pts[i0, 0]
        b0y = pts[i0, 1]
        b3x = pts[i1, 0]
        b3y = pts[i1, 1]
        b1x = b0x + pts[i0, 3] * third
        b1y = b0y + pts[i0, 4] * third
        b2x = b3x - pts[i1, 3] * third
        b2y = b3y - pts[i1, 4] * third
        u = 1.0 - s
        w0 = u * u * u
        w1 = 3.0 * u * u * s
        w2 = 3.0 * u * s * s
        w3 = s * s * s
        x = w0 * b0x + w1 * b1x + w2 * b2x + w3 * b3x
        y = w0 * b0y + w1 * b1y + w2 * b2y + w3 * b3y

    return x, y, t, r


@njit(cache=True, nogil=True, fastmath=True)
def _find_apsis_markers_numba(
    pts,
    base_sim_time,
    ref_index,
    body_x,
    body_y,
    body_m,
    body_scripted,
    body_a,
    body_e,
    body_theta,
    body_arg,
    body_parent,
    G,
    use_time_dependent_bodies,
    max_markers,
    skip_head,
    use_tangents,
):
    # sucht lokale extrema des abstands schiff<->referenzkörper entlang der
    # predictor-punkte (pts: x, y, absolute sim-zeit[, vx, vy]). der
    # diskrete extrempunkt wird per parabel-fit über seine nachbarn zum
    # wahren scheitel verfeinert (_refine_apsis_numba).
    # rückgabe: (out, count); out-zeilen: x, y, t_abs, kind, r wobei
    # kind 0.0 = periapsis (lokales minimum), 1.0 = apoapsis (maximum).
    out = np.empty((max_markers, 5), dtype=np.float64)
    count = 0
    n = pts.shape[0]
    if n < 3 or ref_index < 0 or ref_index >= body_x.shape[0]:
        return out, count

    # pass 1: quadrat-abstand zum referenzkörper pro punkt. der teure
    # kepler-solve (scripted refs) läuft nur an stützstellen, dazwischen
    # wird die ref-position linear über die zeit interpoliert: fehler
    # ~0.5*a_ref*(window/2)^2 (erde/mond: zehner meter), weit unter dem
    # punktabstand und der integrator-toleranz — die extremum-wahl
    # zwischen nachbarpunkten bleibt davon unberührt.
    d2_arr = np.empty(n, dtype=np.float64)
    # Lokal angelegt, NICHT als modul-konstante: siehe _no_body_memo().
    empty_memo = np.zeros((0, 10), dtype=np.float64)
    if use_time_dependent_bodies != 0:
        stride_max = 64
        time_window = 240.0
        ia = 0
        rax, ray = _body_position_at_time_numba(
            ref_index, pts[0, 2] - base_sim_time,
            body_x, body_y, body_m, body_scripted,
            body_a, body_e, body_theta, body_arg, body_parent, G,
            empty_memo,
        )
        while ia < n - 1:
            ib = ia + stride_max
            if ib > n - 1:
                ib = n - 1
            # zeitfenster einhalten (punktzeiten sind monoton)
            while ib > ia + 1 and pts[ib, 2] - pts[ia, 2] > time_window:
                ib = ia + (ib - ia) // 2
            rbx, rby = _body_position_at_time_numba(
                ref_index, pts[ib, 2] - base_sim_time,
                body_x, body_y, body_m, body_scripted,
                body_a, body_e, body_theta, body_arg, body_parent, G,
                empty_memo,
            )
            ta = pts[ia, 2]
            span = pts[ib, 2] - ta
            inv_span = 1.0 / span if span > 0.0 else 0.0
            for i in range(ia, ib):
                s = (pts[i, 2] - ta) * inv_span
                rx = rax + (rbx - rax) * s
                ry = ray + (rby - ray) * s
                dx = pts[i, 0] - rx
                dy = pts[i, 1] - ry
                d2_arr[i] = dx * dx + dy * dy
            ia = ib
            rax = rbx
            ray = rby
        dx = pts[n - 1, 0] - rax
        dy = pts[n - 1, 1] - ray
        d2_arr[n - 1] = dx * dx + dy * dy
    else:
        rx = body_x[ref_index]
        ry = body_y[ref_index]
        for i in range(n):
            dx = pts[i, 0] - rx
            dy = pts[i, 1] - ry
            d2_arr[i] = dx * dx + dy * dy

    # pass 2: trend-scan über den abstandsverlauf
    #
    # `skip_head` sagt, ab welchem index die punkte einer GEMEINSAMEN
    # rechnung entstammen. `_hold_advance` stellt der kurve die
    # schiffsposition aus der WELT als kopf voran; sie weicht um ein
    # vielfaches eines punktschritts von der kurve ab und wuerde als
    # startwert des trends ein schein-extremum auf dem schiff erzeugen.
    # Der kopf wird deshalb nicht gelesen; `best_idx > start` unterdrückt
    # zusätzlich ein extremum unmittelbar dahinter.
    start = skip_head
    if start < 0:
        start = 0
    if start > n - 2:
        start = n - 2
    best_d2 = d2_arr[start]
    best_idx = start
    trend = 0  # 0 unbestimmt, 1 steigend, -1 fallend

    for i in range(start + 1, n):
        if count >= max_markers:
            break
        d2 = d2_arr[i]
        if not math.isfinite(d2):
            continue

        # relative hysterese: richtungswechsel erst ab signifikanter
        # abstandsänderung werten, sonst erzeugen interpolations-wobble
        # und quasi-kreisbahnen serienweise schein-extrema.
        hyst = 1e-4 * best_d2

        if trend == 0:
            if d2 > best_d2 + hyst:
                trend = 1
                best_d2 = d2
                best_idx = i
            elif d2 < best_d2 - hyst:
                trend = -1
                best_d2 = d2
                best_idx = i
        elif trend == 1:
            if d2 >= best_d2:
                best_d2 = d2
                best_idx = i
            elif d2 < best_d2 - hyst:
                # trend kippt nach unten: verfolgtes maximum = apoapsis.
                # best_idx == start (schiffsposition) wird unterdrückt.
                if best_idx > start:
                    rx, ry, rt, rr = _refine_apsis_numba(pts, d2_arr, best_idx, use_tangents)
                    out[count, 0] = rx
                    out[count, 1] = ry
                    out[count, 2] = rt
                    out[count, 3] = 1.0
                    out[count, 4] = rr
                    count += 1
                trend = -1
                best_d2 = d2
                best_idx = i
        else:
            if d2 <= best_d2:
                best_d2 = d2
                best_idx = i
            elif d2 > best_d2 + hyst:
                if best_idx > start:
                    rx, ry, rt, rr = _refine_apsis_numba(pts, d2_arr, best_idx, use_tangents)
                    out[count, 0] = rx
                    out[count, 1] = ry
                    out[count, 2] = rt
                    out[count, 3] = 0.0
                    out[count, 4] = rr
                    count += 1
                trend = 1
                best_d2 = d2
                best_idx = i

    return out, count
