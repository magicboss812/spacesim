# predictor.py

from vec import Vec2
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numba import njit
# Predictor ist absichtlich Numba-only
NUMBA_AVAILABLE = True


#: Spaltenzahl der punkteliste: x, y, t_abs, vx, vy.
#:
#: Die beiden geschwindigkeits-spalten machen aus der liste eine stueckweise
#: KUBISCHE kurve statt einer folge von positionen -- der renderer kann sie
#: damit zur zeichenzeit beliebig fein auswerten (Hermite), ohne dass hier ein
#: integrationsschritt mehr faellt. Der rkn-kernel rechnet die tangente
#: ohnehin (als ableitung genau des polynoms, mit dem er die position
#: interpoliert) und warf sie bisher weg.
#:
#: Kernel, die ihre punkte LINEAR auf die schrittsehne setzen, schreiben hier
#: NaN. Das ist kein fehlerfall, sondern die wahrheit: ein sehnenpunkt hat
#: keine tangente, die zu ihm passt. Der renderer zeichnet solche abschnitte
#: dann als geraden, statt eine kruemmung zu erfinden.
POINT_COLUMNS = 5

#: Spaltenzahl des koerper-notizblocks: [t, x, y, gueltig] + die fuenf
#: zeitunabhaengigen bahngroessen + deren gueltigkeitsmerker.
#: Siehe `_body_position_at_time_numba`.
BODY_MEMO_COLUMNS = 10


def _no_body_memo():
    """Leerer notizblock fuer aufrufer, bei denen sich das merken nicht lohnt.

    `_body_position_at_time_numba` erkennt an der zeilenzahl 0, dass kein
    notizblock vorliegt, und rechnet wie zuvor.

    **Das MUSS eine funktion sein, keine modul-konstante.** Numba behandelt
    ein globales array als compile-zeit-konstante und typisiert es
    `readonly` -- die (per `use_memo` ohnehin nie erreichten) schreibzugriffe
    im rumpf lassen sich dann nicht mehr typisieren, und der GANZE aufrufende
    kernel scheitert beim uebersetzen. Genau so verschwanden die Ap/Pe-marker:
    `_find_apsis_markers_numba` warf `NumbaTypeError`, der aufrufer fing die
    ausnahme, und `get_apsis_markers()` lieferte stillschweigend null marker
    -- im spiel sichtbar nur daran, dass die rauten und die HUD-zahlen fehlten.
    """
    return np.zeros((0, BODY_MEMO_COLUMNS), dtype=np.float64)


def _empty_points():
    """Leere punkteliste in der kanonischen breite."""
    if np is None:
        return []
    return np.empty((0, POINT_COLUMNS), dtype=np.float64)


def _widen_points(points):
    """Punkte auf POINT_COLUMNS bringen; fehlende tangenten werden NaN."""
    if np is None or not isinstance(points, np.ndarray) or points.ndim != 2:
        return points
    have = int(points.shape[1])
    if have >= POINT_COLUMNS:
        return points
    wide = np.empty((points.shape[0], POINT_COLUMNS), dtype=np.float64)
    wide[:, :have] = points
    wide[:, have:] = np.nan
    return wide


if NUMBA_AVAILABLE:
    @njit(cache=True, nogil=True, fastmath=True)
    def _compute_acc_numba(x, y, body_x, body_y, body_m, body_fixed, G):
        ax = 0.0
        ay = 0.0
        for i in range(body_x.shape[0]):
            if body_fixed[i] == 0:
                continue
            dx = body_x[i] - x
            dy = body_y[i] - y
            dist2 = dx * dx + dy * dy
            if dist2 < 1e-12:
                continue
            invd = 1.0 / math.sqrt(dist2)
            accm = G * body_m[i] / dist2
            ax += dx * invd * accm
            ay += dy * invd * accm
        return ax, ay


    @njit(cache=True, nogil=True, fastmath=True)
    def _compute_acc_nearest_numba(x, y, body_x, body_y, body_m, body_fixed, G):
        ax = 0.0
        ay = 0.0
        nearest_r = 1e30
        for i in range(body_x.shape[0]):
            if body_fixed[i] == 0:
                continue
            dx = body_x[i] - x
            dy = body_y[i] - y
            dist2 = dx * dx + dy * dy
            if dist2 < 1e-12:
                continue
            dist = math.sqrt(dist2)
            if dist < nearest_r:
                nearest_r = dist
            invd = 1.0 / dist
            accm = G * body_m[i] / dist2
            ax += dx * invd * accm
            ay += dy * invd * accm
        acc_mag = math.sqrt(ax * ax + ay * ay)
        return ax, ay, nearest_r, acc_mag


    @njit(cache=True, nogil=True, fastmath=True)
    def _rk4_step_numba(
        px,
        py,
        vx,
        vy,
        dt,
        ref_enabled,
        ref_px,
        ref_py,
        body_x,
        body_y,
        body_m,
        body_fixed,
        G,
    ):
        ref_ax = 0.0
        ref_ay = 0.0
        if ref_enabled != 0:
            ref_ax, ref_ay = _compute_acc_numba(ref_px, ref_py, body_x, body_y, body_m, body_fixed, G)

        k1_ax_raw, k1_ay_raw = _compute_acc_numba(px, py, body_x, body_y, body_m, body_fixed, G)
        k1_ax = k1_ax_raw - ref_ax
        k1_ay = k1_ay_raw - ref_ay
        k1_vx, k1_vy = vx, vy

        p2x = px + k1_vx * (dt / 2.0)
        p2y = py + k1_vy * (dt / 2.0)
        v2x = vx + k1_ax * (dt / 2.0)
        v2y = vy + k1_ay * (dt / 2.0)
        k2_ax_raw, k2_ay_raw = _compute_acc_numba(p2x, p2y, body_x, body_y, body_m, body_fixed, G)
        k2_ax = k2_ax_raw - ref_ax
        k2_ay = k2_ay_raw - ref_ay
        k2_vx, k2_vy = v2x, v2y

        p3x = px + k2_vx * (dt / 2.0)
        p3y = py + k2_vy * (dt / 2.0)
        v3x = vx + k2_ax * (dt / 2.0)
        v3y = vy + k2_ay * (dt / 2.0)
        k3_ax_raw, k3_ay_raw = _compute_acc_numba(p3x, p3y, body_x, body_y, body_m, body_fixed, G)
        k3_ax = k3_ax_raw - ref_ax
        k3_ay = k3_ay_raw - ref_ay
        k3_vx, k3_vy = v3x, v3y

        p4x = px + k3_vx * dt
        p4y = py + k3_vy * dt
        v4x = vx + k3_ax * dt
        v4y = vy + k3_ay * dt
        k4_ax_raw, k4_ay_raw = _compute_acc_numba(p4x, p4y, body_x, body_y, body_m, body_fixed, G)
        k4_ax = k4_ax_raw - ref_ax
        k4_ay = k4_ay_raw - ref_ay
        k4_vx, k4_vy = v4x, v4y

        next_px = px + (k1_vx + 2.0 * k2_vx + 2.0 * k3_vx + k4_vx) * (dt / 6.0)
        next_py = py + (k1_vy + 2.0 * k2_vy + 2.0 * k3_vy + k4_vy) * (dt / 6.0)
        next_vx = vx + (k1_ax + 2.0 * k2_ax + 2.0 * k3_ax + k4_ax) * (dt / 6.0)
        next_vy = vy + (k1_ay + 2.0 * k2_ay + 2.0 * k3_ay + k4_ay) * (dt / 6.0)

        return next_px, next_py, next_vx, next_vy


    @njit(cache=True, nogil=True, fastmath=True)
    def _rkn_acc_numba(x, y, ref_ax, ref_ay, body_x, body_y, body_m, body_fixed, G):
        ax, ay = _compute_acc_numba(x, y, body_x, body_y, body_m, body_fixed, G)
        return ax, ay


    @njit(cache=True, nogil=True, fastmath=True)
    def _rkn4_step_numba(
        px,
        py,
        vx,
        vy,
        dt,
        ref_enabled,
        ref_px,
        ref_py,
        body_x,
        body_y,
        body_m,
        body_fixed,
        G,
    ):
        ref_ax = 0.0
        ref_ay = 0.0
        if ref_enabled != 0:
            ref_ax, ref_ay = _compute_acc_numba(ref_px, ref_py, body_x, body_y, body_m, body_fixed, G)

        dt2 = dt * dt
        half_dt = 0.5 * dt

        k1_ax, k1_ay = _rkn_acc_numba(px, py, ref_ax, ref_ay, body_x, body_y, body_m, body_fixed, G)

        p2x = px + half_dt * vx + 0.125 * dt2 * k1_ax
        p2y = py + half_dt * vy + 0.125 * dt2 * k1_ay
        k2_ax, k2_ay = _rkn_acc_numba(p2x, p2y, ref_ax, ref_ay, body_x, body_y, body_m, body_fixed, G)

        # k3 teilt sein argument mit k2 -- siehe _rkn4_step_time_numba. Der
        # ausdruck war derselbe, die funktion ist rein, also war die dritte
        # kraftauswertung bit fuer bit k2 und damit ein viertel der arbeit
        # umsonst.
        k3_ax = k2_ax
        k3_ay = k2_ay

        p4x = px + dt * vx + 0.5 * dt2 * k3_ax
        p4y = py + dt * vy + 0.5 * dt2 * k3_ay
        k4_ax, k4_ay = _rkn_acc_numba(p4x, p4y, ref_ax, ref_ay, body_x, body_y, body_m, body_fixed, G)

        next_px = px + dt * vx + (dt2 / 6.0) * (k1_ax + k2_ax + k3_ax)
        next_py = py + dt * vy + (dt2 / 6.0) * (k1_ay + k2_ay + k3_ay)
        next_vx = vx + (dt / 6.0) * (k1_ax + 2.0 * k2_ax + 2.0 * k3_ax + k4_ax)
        next_vy = vy + (dt / 6.0) * (k1_ay + 2.0 * k2_ay + 2.0 * k3_ay + k4_ay)

        return next_px, next_py, next_vx, next_vy


    @njit(cache=True, nogil=True, fastmath=True)
    def _rkn_adaptive_step_numba(
        px,
        py,
        vx,
        vy,
        dt,
        min_dt,
        max_dt,
        rtol,
        atol_pos,
        atol_vel,
        safety,
        min_factor,
        max_factor,
        max_rejects,
        ref_enabled,
        ref_px,
        ref_py,
        body_x,
        body_y,
        body_m,
        body_fixed,
        G,
    ):
        if (not math.isfinite(min_dt)) or min_dt <= 0.0:
            min_dt = 1e-9
        if (not math.isfinite(max_dt)) or max_dt <= 0.0:
            max_dt = min_dt
        if max_dt < min_dt:
            max_dt = min_dt
        if (not math.isfinite(rtol)) or rtol < 0.0:
            rtol = 0.0
        if (not math.isfinite(atol_pos)) or atol_pos <= 0.0:
            atol_pos = 1e-12
        if (not math.isfinite(atol_vel)) or atol_vel <= 0.0:
            atol_vel = 1e-12
        if (not math.isfinite(safety)) or safety <= 0.0:
            safety = 0.9
        if (not math.isfinite(min_factor)) or min_factor <= 0.0:
            min_factor = 0.2
        if (not math.isfinite(max_factor)) or max_factor < min_factor:
            max_factor = min_factor
        if max_rejects < 0:
            max_rejects = 0

        step_dt = dt
        if (not math.isfinite(step_dt)) or step_dt <= 0.0:
            step_dt = max_dt
        if step_dt < min_dt:
            step_dt = min_dt
        if step_dt > max_dt:
            step_dt = max_dt

        rejected_count = 0

        while True:
            half_dt = 0.5 * step_dt

            full_px, full_py, full_vx, full_vy = _rkn4_step_numba(
                px,
                py,
                vx,
                vy,
                step_dt,
                ref_enabled,
                ref_px,
                ref_py,
                body_x,
                body_y,
                body_m,
                body_fixed,
                G,
            )
            half1_px, half1_py, half1_vx, half1_vy = _rkn4_step_numba(
                px,
                py,
                vx,
                vy,
                half_dt,
                ref_enabled,
                ref_px,
                ref_py,
                body_x,
                body_y,
                body_m,
                body_fixed,
                G,
            )
            half2_px, half2_py, half2_vx, half2_vy = _rkn4_step_numba(
                half1_px,
                half1_py,
                half1_vx,
                half1_vy,
                half_dt,
                ref_enabled,
                ref_px,
                ref_py,
                body_x,
                body_y,
                body_m,
                body_fixed,
                G,
            )

            finite_state = (
                math.isfinite(full_px)
                and math.isfinite(full_py)
                and math.isfinite(full_vx)
                and math.isfinite(full_vy)
                and math.isfinite(half2_px)
                and math.isfinite(half2_py)
                and math.isfinite(half2_vx)
                and math.isfinite(half2_vy)
            )

            if finite_state:
                pos_dx = half2_px - full_px
                pos_dy = half2_py - full_py
                vel_dx = half2_vx - full_vx
                vel_dy = half2_vy - full_vy

                pos_err = math.sqrt(pos_dx * pos_dx + pos_dy * pos_dy) / 15.0
                vel_err = math.sqrt(vel_dx * vel_dx + vel_dy * vel_dy) / 15.0

                cur_r = math.sqrt(px * px + py * py)
                next_r = math.sqrt(half2_px * half2_px + half2_py * half2_py)
                cur_speed = math.sqrt(vx * vx + vy * vy)
                next_speed = math.sqrt(half2_vx * half2_vx + half2_vy * half2_vy)
                motion_scale = cur_speed * step_dt

                pos_ref = cur_r
                if next_r > pos_ref:
                    pos_ref = next_r
                if motion_scale > pos_ref:
                    pos_ref = motion_scale
                if pos_ref < 1.0:
                    pos_ref = 1.0

                vel_ref = cur_speed
                if next_speed > vel_ref:
                    vel_ref = next_speed
                if vel_ref < 1.0:
                    vel_ref = 1.0

                pos_scale = atol_pos + rtol * pos_ref
                vel_scale = atol_vel + rtol * vel_ref
                if pos_scale <= 0.0 or not math.isfinite(pos_scale):
                    pos_scale = 1e-30
                if vel_scale <= 0.0 or not math.isfinite(vel_scale):
                    vel_scale = 1e-30

                pos_norm = pos_err / pos_scale
                vel_norm = vel_err / vel_scale
                err_norm = pos_norm
                if vel_norm > err_norm:
                    err_norm = vel_norm
            else:
                err_norm = 1e300

            if math.isfinite(err_norm) and err_norm <= 1.0:
                if err_norm <= 1e-300:
                    factor = max_factor
                else:
                    factor = safety * err_norm ** (-0.2)
                    if factor < min_factor:
                        factor = min_factor
                    if factor > max_factor:
                        factor = max_factor

                proposed_next_dt = step_dt * factor
                if proposed_next_dt < min_dt:
                    proposed_next_dt = min_dt
                if proposed_next_dt > max_dt:
                    proposed_next_dt = max_dt

                return (
                    half2_px,
                    half2_py,
                    half2_vx,
                    half2_vy,
                    step_dt,
                    proposed_next_dt,
                    err_norm,
                    1,
                    rejected_count,
                    0,
                )

            if not math.isfinite(err_norm):
                err_norm = 1e300

            if step_dt <= min_dt * (1.0 + 1e-12):
                return (
                    px,
                    py,
                    vx,
                    vy,
                    0.0,
                    min_dt,
                    err_norm,
                    0,
                    rejected_count,
                    6,
                )

            if rejected_count >= max_rejects:
                return (
                    px,
                    py,
                    vx,
                    vy,
                    0.0,
                    step_dt,
                    err_norm,
                    0,
                    rejected_count,
                    2,
                )

            if err_norm <= 1e-300:
                factor = min_factor
            else:
                factor = safety * err_norm ** (-0.2)
            if factor < min_factor:
                factor = min_factor
            if factor > max_factor:
                factor = max_factor

            next_dt = step_dt * factor
            if next_dt >= step_dt:
                next_dt = step_dt * min_factor
            if next_dt < min_dt:
                next_dt = min_dt
            if next_dt > max_dt:
                next_dt = max_dt

            rejected_count += 1
            step_dt = next_dt


    @njit(cache=True, nogil=True, fastmath=True)
    def _body_kepler_constants_numba(index, body_m, body_a, body_e, body_theta, body_arg, body_parent, G):
        """Die ZEITUNABHAENGIGEN groessen einer skriptierten bahn.

        M0, mittlere bewegung, sqrt(1-e^2) und cos/sin des periapsis-arguments
        haengen nur von den bahnelementen ab -- sie wurden bisher bei jeder
        einzelnen auswertung neu gerechnet, acht trigonometrie- bzw.
        wurzel-operationen von rund neunzehn. Die reihenfolge der rechnungen
        ist WORT FUER WORT die des inline-weges unten, damit beide exakt
        dieselben gleitkommazahlen erzeugen.

        Rueckgabe: (M0, mittlere bewegung, sqrt(1-e^2), cos arg, sin arg, ok).
        """
        parent = body_parent[index]
        if parent < 0 or parent >= body_m.shape[0]:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0

        a = body_a[index]
        e = body_e[index]
        parent_mass = body_m[parent]
        if a <= 0.0 or e < 0.0 or e >= 1.0 or parent_mass <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0

        mu = G * parent_mass
        if mu <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0

        nu0 = body_theta[index]
        arg = body_arg[index]

        cos_nu0 = math.cos(nu0)
        sin_nu0 = math.sin(nu0)
        denom = 1.0 + e * cos_nu0
        if abs(denom) <= 1e-14:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0

        sqrt_one_minus_e2 = math.sqrt(max(0.0, 1.0 - e * e))
        sin_e0 = sqrt_one_minus_e2 * sin_nu0 / denom
        cos_e0 = (e + cos_nu0) / denom
        ecc_anomaly0 = math.atan2(sin_e0, cos_e0)
        mean_anomaly0 = ecc_anomaly0 - e * math.sin(ecc_anomaly0)

        mean_motion = math.sqrt(mu / (a * a * a))
        return (mean_anomaly0, mean_motion, sqrt_one_minus_e2,
                math.cos(arg), math.sin(arg), 1)


    @njit(cache=True, nogil=True, fastmath=True)
    def _body_scripted_relative_xy_numba(index, local_t, body_m, body_a, body_e, body_theta, body_arg, body_parent, G, body_memo):
        # Schneller weg: der kernel hat die zeitunabhaengigen groessen im
        # vorlauf nach body_memo[:, 4:10] gelegt (spalte 9: 0 = nicht
        # vorberechnet, 1 = gueltig, -1 = bahn unbrauchbar). Ohne notizblock
        # -- oder mit abgeschaltetem `use_body_memo` -- laeuft der zweig
        # darunter, der alles wie frueher selbst rechnet.
        if body_memo.shape[0] == body_m.shape[0] and body_memo[index, 9] != 0.0:
            if body_memo[index, 9] < 0.0:
                return 0.0, 0.0, 0
            a = body_a[index]
            e = body_e[index]
            mean_anomaly0 = body_memo[index, 4]
            mean_motion = body_memo[index, 5]
            sqrt_one_minus_e2 = body_memo[index, 6]
            c = body_memo[index, 7]
            s = body_memo[index, 8]
        else:
            (mean_anomaly0, mean_motion, sqrt_one_minus_e2, c, s,
             const_ok) = _body_kepler_constants_numba(
                index, body_m, body_a, body_e, body_theta, body_arg, body_parent, G,
            )
            if const_ok == 0:
                return 0.0, 0.0, 0
            a = body_a[index]
            e = body_e[index]

        mean_anomaly = mean_anomaly0 + mean_motion * local_t
        two_pi = 2.0 * math.pi
        mean_anomaly = (mean_anomaly + math.pi) % two_pi
        if mean_anomaly < 0.0:
            mean_anomaly += two_pi
        mean_anomaly -= math.pi

        ecc_anomaly = mean_anomaly
        for _ in range(12):
            f = ecc_anomaly - e * math.sin(ecc_anomaly) - mean_anomaly
            fp = 1.0 - e * math.cos(ecc_anomaly)
            if abs(fp) <= 1e-14:
                break
            delta = f / fp
            ecc_anomaly -= delta
            if abs(delta) <= 1e-13:
                break

        cos_e = math.cos(ecc_anomaly)
        sin_e = math.sin(ecc_anomaly)
        r = a * (1.0 - e * cos_e)
        if r <= 0.0 or not math.isfinite(r):
            return 0.0, 0.0, 0

        nu = math.atan2(sqrt_one_minus_e2 * sin_e, cos_e - e)
        x_orb = r * math.cos(nu)
        y_orb = r * math.sin(nu)
        rel_x = x_orb * c - y_orb * s
        rel_y = x_orb * s + y_orb * c
        return rel_x, rel_y, 1


    @njit(cache=True, nogil=True, fastmath=True)
    def _body_position_at_time_numba(
        index,
        local_t,
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
        body_memo,
    ):
        # `body_memo` ist ein (n,3)-notizblock [t, x, y] je koerper. Er ist die
        # einzige optimierung, die diese funktion kennt, und sie ist
        # BIT-IDENTISCH: gemerkt wird genau der wert, den derselbe rechenweg
        # eben erzeugt hat, und getroffen wird nur bei EXAKT gleicher zeit.
        #
        # Sie lohnt sich, weil dieselbe koerperposition pro integrationsschritt
        # mehrfach gebraucht wird, und zwar aus drei unabhaengigen gruenden:
        #   1. mit bezugskoerper stellt _rkn_acc_time_numba ALLE koerper zweimal
        #      zur selben zeit auf (einmal fuer das schiff, einmal fuer den
        #      bezugspunkt);
        #   2. jeder mond loest die kepler-gleichung seines planeten selbst noch
        #      einmal -- Saturn wurde je auswertung sechsmal geloest;
        #   3. die schrittverdopplung wertet t, t+h/2 und t+h mehrfach aus.
        # Spalten: [t, x, y, gueltig]. Ein aufrufer ohne notizblock uebergibt
        # ein (0,4)-array.
        #
        # Die vierte spalte ist NICHT redundant. Der naheliegende weg -- die
        # zeitspalte auf NaN setzen und sich darauf verlassen, dass
        # `NaN == local_t` falsch ist -- funktioniert hier NICHT: alle kernel
        # laufen mit `fastmath=True`, und das schaltet LLVMs `nnan` ein, also
        # die zusicherung, dass keine NaN auftreten. Der vergleich darf dann
        # zu true gefaltet werden. Gemessen: der allererste zugriff meldete
        # einen treffer und lieferte die uninitialisierten nullen zurueck --
        # jeder koerper stand im ursprung, die bahn wich um 1.8e6 m ab.
        n = body_x.shape[0]
        if index < 0 or index >= n:
            return 0.0, 0.0

        use_memo = body_memo.shape[0] == n
        if use_memo and body_memo[index, 3] != 0.0 and body_memo[index, 0] == local_t:
            return body_memo[index, 1], body_memo[index, 2]

        # Kein `np.empty(n)` mehr fuer die elternkette: das war eine
        # HALDEN-ANFORDERUNG pro aufruf, und aufgerufen wird je koerper und
        # auswertungszeit -- gemessen ueber 200 000 mal pro vorhersage. Die
        # kette ist hoechstens drei glieder lang (mond -> planet -> stern),
        # also wird beim abstieg einfach neu hochgezaehlt: O(tiefe^2)
        # zeigerschritte gegen eine allokation, und die reihenfolge der
        # summanden bleibt exakt dieselbe.
        chain_count = 0
        cur = index
        memo_hit = 0
        hit_x = 0.0
        hit_y = 0.0

        while cur >= 0 and cur < n and chain_count < n:
            # Ein vorfahr, der zu DIESER zeit schon berechnet wurde, beendet
            # den aufstieg: sein absolutwert ist die basis, auf die die
            # restlichen glieder addiert werden -- dieselbe summe wie zuvor,
            # in derselben reihenfolge.
            if use_memo and body_memo[cur, 3] != 0.0 and body_memo[cur, 0] == local_t:
                memo_hit = 1
                hit_x = body_memo[cur, 1]
                hit_y = body_memo[cur, 2]
                break
            parent = body_parent[cur]
            if body_scripted[cur] == 0 or body_a[cur] <= 0.0 or parent < 0 or parent >= n:
                break
            chain_count += 1
            cur = parent

        if cur < 0 or cur >= n:
            cur = index
            chain_count = 0
            memo_hit = 0

        if memo_hit != 0:
            wx = hit_x
            wy = hit_y
        else:
            wx = body_x[cur]
            wy = body_y[cur]

        for chain_pos in range(chain_count - 1, -1, -1):
            # `chain[chain_pos]` war der koerper, der chain_pos schritte
            # UEBER `index` liegt -- hier wieder erlaufen statt gespeichert.
            child = index
            for _up in range(chain_pos):
                child = body_parent[child]
            rel_x, rel_y, ok = _body_scripted_relative_xy_numba(
                child,
                local_t,
                body_m,
                body_a,
                body_e,
                body_theta,
                body_arg,
                body_parent,
                G,
                body_memo,
            )
            if ok == 0:
                return body_x[index], body_y[index]
            wx += rel_x
            wy += rel_y
            # Jedes zwischenglied ist selbst eine gueltige koerperposition --
            # merken, damit die geschwister-monde denselben planeten nicht
            # noch einmal loesen.
            if use_memo:
                body_memo[child, 0] = local_t
                body_memo[child, 1] = wx
                body_memo[child, 2] = wy
                body_memo[child, 3] = 1.0

        return wx, wy


    @njit(cache=True, nogil=True, fastmath=True)
    def _compute_acc_time_numba(
        x,
        y,
        local_t,
        body_x,
        body_y,
        body_m,
        body_fixed,
        body_scripted,
        body_a,
        body_e,
        body_theta,
        body_arg,
        body_parent,
        G,
        use_time_dependent_bodies,
        body_memo,
    ):
        ax = 0.0
        ay = 0.0
        for i in range(body_x.shape[0]):
            if body_fixed[i] == 0:
                continue

            if use_time_dependent_bodies != 0:
                source_x, source_y = _body_position_at_time_numba(
                    i,
                    local_t,
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
                    body_memo,
                )
            else:
                source_x = body_x[i]
                source_y = body_y[i]

            dx = source_x - x
            dy = source_y - y
            dist2 = dx * dx + dy * dy
            if dist2 < 1e-12:
                continue
            invd = 1.0 / math.sqrt(dist2)
            accm = G * body_m[i] / dist2
            ax += dx * invd * accm
            ay += dy * invd * accm
        return ax, ay


    @njit(cache=True, nogil=True, fastmath=True)
    def _local_timescale_numba(
        x,
        y,
        local_t,
        body_x,
        body_y,
        body_m,
        body_fixed,
        body_scripted,
        body_a,
        body_e,
        body_theta,
        body_arg,
        body_parent,
        G,
        use_time_dependent_bodies,
        body_memo,
    ):
        """`min_i sqrt(r_i^3 / (G m_i))` am ORT (x, y) zur zeit local_t.

        Wort fuer wort dieselbe groesse wie `world.characteristic_timescale`
        -- fuer eine kreisbahn um einen koerper exakt T/2pi. Es gibt bewusst
        nur EINE definition davon im projekt; diese hier ist ihre numba-form,
        weil der kernel die welt nicht fragen kann.

        MINIMUM ueber alle koerper, NIE argmax(g): jenseits von r ~ 2.6e8 m
        von der Erde ist die sonnenbeschleunigung groesser als die der Erde,
        waehrend die Erd-SOI bis 9.2e8 m reicht -- die auswahl kippte dort auf
        die Sonne und meldete deren zeitskala. Ueber einen 2-%-radiusschritt
        gemessen ein sprung um das 97-fache. Das minimum stetiger funktionen
        hat den sprung nicht (hoechstens 1.03x). Siehe die ausfuehrliche
        herleitung in `.claude/rules/physics-world.md`.

        Rueckgabe 0.0, wenn kein koerper eine zeitskala liefert -- der aufrufer
        laesst seine decke dann unveraendert.
        """
        best = 0.0
        have = 0
        for i in range(body_x.shape[0]):
            if body_fixed[i] == 0:
                continue
            mass = body_m[i]
            if mass <= 0.0:
                continue

            if use_time_dependent_bodies != 0:
                source_x, source_y = _body_position_at_time_numba(
                    i,
                    local_t,
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
                    body_memo,
                )
            else:
                source_x = body_x[i]
                source_y = body_y[i]

            dx = source_x - x
            dy = source_y - y
            r2 = dx * dx + dy * dy
            if r2 < 1e-6:
                continue
            # sqrt(r^3/mu), als sqrt(r)*r/sqrt(mu) waere es eine wurzel mehr.
            t = math.sqrt(math.sqrt(r2) * r2 / (G * mass))
            if have == 0 or t < best:
                best = t
                have = 1
        if have == 0:
            return 0.0
        return best


    @njit(cache=True, nogil=True, fastmath=True)
    def _rkn_acc_time_numba(
        x,
        y,
        local_t,
        ref_enabled,
        ref_index,
        ref_px,
        ref_py,
        body_x,
        body_y,
        body_m,
        body_fixed,
        body_scripted,
        body_a,
        body_e,
        body_theta,
        body_arg,
        body_parent,
        G,
        use_time_dependent_bodies,
        body_memo,
    ):
        ref_ax = 0.0
        ref_ay = 0.0
        if ref_enabled != 0:
            if use_time_dependent_bodies != 0 and ref_index >= 0 and ref_index < body_x.shape[0]:
                rpx, rpy = _body_position_at_time_numba(
                    ref_index,
                    local_t,
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
                    body_memo,
                )
            else:
                rpx = ref_px
                rpy = ref_py
            # Diese zweite aufstellung ALLER koerper laeuft zur exakt selben
            # zeit wie die darunter -- ohne notizblock war sie eine volle
            # verdopplung der teuersten schleife im predictor.
            ref_ax, ref_ay = _compute_acc_time_numba(
                rpx,
                rpy,
                local_t,
                body_x,
                body_y,
                body_m,
                body_fixed,
                body_scripted,
                body_a,
                body_e,
                body_theta,
                body_arg,
                body_parent,
                G,
                use_time_dependent_bodies,
                body_memo,
            )

        ax, ay = _compute_acc_time_numba(
            x,
            y,
            local_t,
            body_x,
            body_y,
            body_m,
            body_fixed,
            body_scripted,
            body_a,
            body_e,
            body_theta,
            body_arg,
            body_parent,
            G,
            use_time_dependent_bodies,
            body_memo,
        )
        return ax, ay


    @njit(cache=True, nogil=True, fastmath=True)
    def _rkn4_step_time_numba(
        px,
        py,
        vx,
        vy,
        local_t,
        dt,
        ref_enabled,
        ref_index,
        ref_px,
        ref_py,
        body_x,
        body_y,
        body_m,
        body_fixed,
        body_scripted,
        body_a,
        body_e,
        body_theta,
        body_arg,
        body_parent,
        G,
        use_time_dependent_bodies,
        body_memo,
    ):
        dt2 = dt * dt
        half_dt = 0.5 * dt
        mid_t = local_t + half_dt
        end_t = local_t + dt

        k1_ax, k1_ay = _rkn_acc_time_numba(
            px, py, local_t, ref_enabled, ref_index, ref_px, ref_py,
            body_x, body_y, body_m, body_fixed, body_scripted, body_a, body_e,
            body_theta, body_arg, body_parent, G, use_time_dependent_bodies,
            body_memo
        )

        p2x = px + half_dt * vx + 0.125 * dt2 * k1_ax
        p2y = py + half_dt * vy + 0.125 * dt2 * k1_ay
        k2_ax, k2_ay = _rkn_acc_time_numba(
            p2x, p2y, mid_t, ref_enabled, ref_index, ref_px, ref_py,
            body_x, body_y, body_m, body_fixed, body_scripted, body_a, body_e,
            body_theta, body_arg, body_parent, G, use_time_dependent_bodies,
            body_memo
        )

        # DRITTE STUFE: k1, NICHT k2.
        #
        # Im klassischen RKN4 teilen sich k2 und k3 ihr argument -- genau
        # diese identitaet ist der ganze vorteil des verfahrens (ordnung 4
        # aus 3 kraftauswertungen, wo RK4 vier braucht). Mit `k2` hier
        # unterscheiden sich die stufen, die ordnungsbedingung bricht, und
        # man bezahlt die vierte auswertung fuer 3. ordnung.
        #
        # Die welt hat diesen fehler am 2026-08-18 verloren
        # (world.py + world_kernels.py:147), der predictor NICHT -- er ist
        # bis hierher 3. ordnung geblieben. Zwei verfahren auf derselben
        # physik laufen aber auseinander, und weil die vorhersage die einzige
        # ist, deren ergebnis man SIEHT, sah man es an ihr: gemessen in einer
        # erdumlaufbahn (rp 2e7 m, e = 0.3) lag ihr apoapsis bei 3.7133e7 m,
        # das der welt bei 3.7692e7 m -- 5.5e5 m auseinander nach EINEM
        # umlauf, und beide fuer sich in der schrittweite auskonvergiert.
        # Im zeitraffer, wo die kurve stehenbleibt und das schiff an ihr
        # entlangrutscht, ist genau das das "schiff loest sich von der linie".
        #
        # UND WEIL k3 SEIN ARGUMENT MIT k2 TEILT, IST ES DASSELBE k.
        # `p3 == p2` stand hier als eigener ausdruck, wurde aber aus denselben
        # summanden in derselben reihenfolge gebildet -- und
        # `_rkn_acc_time_numba` ist eine reine funktion von (ort, zeit). Die
        # dritte auswertung lieferte also denselben wert wie die zweite. Das
        # ist die klassische 3-stufen-form (ordnung 4 aus 3 auswertungen), es
        # geht keine genauigkeit verloren -- eine verdopplung faellt weg.
        #
        # ES SIND ABER NICHT DIE ERHOFFTEN 25 %, UND DER GRUND IST DER
        # NOTIZBLOCK. `.claude/rules/physics-world.md` beziffert diese stufe
        # mit "~25 % der integratorkosten"; das gilt fuer den WELT-kernel, der
        # keinen `body_memo` hat. Hier lief k3 zur exakt selben zeit wie k2,
        # traf also fuer jeden koerper den notizblock und bezahlte nur noch die
        # 28 nachschlage plus die kraftsumme -- die teuren kepler-loesungen
        # waren laengst gespart. Gemessen ueber die neun messlagen von
        # `tests/warp_predictor_test.py` §24, gegen denselben lauf mit wieder
        # eingesetzter dritter auswertung: **1.02x bis 1.15x, median 1.12x**.
        #
        # Und es ist BIT-IDENTISCH -- in allen neun lagen groesste abweichung
        # 0.000e+00 bei gleicher schrittzahl. (Auf einer bahn, die numerisch
        # davonlaeuft, verstaerkt sich unter `fastmath` eine unterschiedliche
        # rundung von `k1 + k2 + k3` gegen `k1 + 2*k2` durchaus bis auf
        # millimeter; auf den bahnen, die das spiel zeichnet, tut sie es
        # nicht.)
        k3_ax = k2_ax
        k3_ay = k2_ay

        p4x = px + dt * vx + 0.5 * dt2 * k3_ax
        p4y = py + dt * vy + 0.5 * dt2 * k3_ay
        k4_ax, k4_ay = _rkn_acc_time_numba(
            p4x, p4y, end_t, ref_enabled, ref_index, ref_px, ref_py,
            body_x, body_y, body_m, body_fixed, body_scripted, body_a, body_e,
            body_theta, body_arg, body_parent, G, use_time_dependent_bodies,
            body_memo
        )

        next_px = px + dt * vx + (dt2 / 6.0) * (k1_ax + k2_ax + k3_ax)
        next_py = py + dt * vy + (dt2 / 6.0) * (k1_ay + k2_ay + k3_ay)
        next_vx = vx + (dt / 6.0) * (k1_ax + 2.0 * k2_ax + 2.0 * k3_ax + k4_ax)
        next_vy = vy + (dt / 6.0) * (k1_ay + 2.0 * k2_ay + 2.0 * k3_ay + k4_ay)

        return next_px, next_py, next_vx, next_vy


    @njit(cache=True, nogil=True, fastmath=True)
    def _rkn_adaptive_step_time_numba(
        px,
        py,
        vx,
        vy,
        local_t,
        dt,
        min_dt,
        max_dt,
        rtol,
        atol_pos,
        atol_vel,
        safety,
        min_factor,
        max_factor,
        max_rejects,
        ref_enabled,
        ref_index,
        ref_px,
        ref_py,
        body_x,
        body_y,
        body_m,
        body_fixed,
        body_scripted,
        body_a,
        body_e,
        body_theta,
        body_arg,
        body_parent,
        G,
        use_time_dependent_bodies,
        body_memo,
        max_dt_floor,
        timescale_divisor,
    ):
        # DIE DECKE IST ORTLICH, NICHT GLOBAL.
        #
        # `max_dt` kommt als die vom HORIZONT abgeleitete decke herein (viele
        # tausend sekunden bei langer vorausschau). Sie darf aber nicht ueber
        # die bahn springen, und wie eng sie sein muss, haengt davon ab, wo das
        # schiff GERADE ist -- nicht davon, wo es beim anlegen des
        # schnappschusses stand. Genau das war der fehler: `_make_snapshot`
        # rechnete `t_char/divisor` EINMAL am schiff und legte das ergebnis
        # ueber den ganzen lauf. Auf einer abflugbahn (Erdorbit -> Jupiter)
        # ist das die zeitskala der ERDE, und die galt dann auch fuer die
        # 2.85 jahre heliozentrischen reiseflugs, wo die fehlerkontrolle
        # muehelos 30 000 s schritte nimmt. Gemessen bei 128x horizont:
        # **24 633 schritte / 899 ms gegen 1 276 / 56 ms**, dieselbe bahn --
        # das 16-fache, und praktisch die gesamte rechenzeit lag im fernfeld,
        # wo sie nichts kauft.
        #
        # Ortlich gerechnet ist die decke nahe der Erde genauso eng wie zuvor
        # (der boden `max_dt_floor` bindet dort ohnehin) und oeffnet sich erst,
        # wenn das schiff die Erde wirklich verlassen hat. Fuer einen lauf, der
        # in EINEM regime bleibt -- jede geschlossene umlaufbahn, also auch die
        # lage aus §20 -- ist das bit fuer bit die alte rechnung.
        #
        # Die kosten sind fast null, und zwar wegen der REIHENFOLGE: die
        # zeitskala wird zur zeit `local_t` ausgewertet, also genau der zeit,
        # zu der gleich darauf k1 alle koerper braucht. Sie WAERMT damit den
        # notizblock, statt zusaetzliche kepler-loesungen zu bezahlen; was
        # bleibt, sind 28 wurzeln je schritt gegen 12 x 28 kepler-loesungen.
        if timescale_divisor > 0.0:
            t_char = _local_timescale_numba(
                px,
                py,
                local_t,
                body_x,
                body_y,
                body_m,
                body_fixed,
                body_scripted,
                body_a,
                body_e,
                body_theta,
                body_arg,
                body_parent,
                G,
                use_time_dependent_bodies,
                body_memo,
            )
            if t_char > 0.0:
                orbit_cap = t_char / timescale_divisor
                # Der boden ist die voreingestellte schrittdecke der
                # qualitaetsstufe. Er darf nicht unterschritten werden -- sonst
                # wuerde die ortliche decke im nahfeld STRENGER als die alte
                # globale und der nahfeld-lauf teurer statt gleich teuer.
                if orbit_cap < max_dt_floor:
                    orbit_cap = max_dt_floor
                if orbit_cap < max_dt:
                    max_dt = orbit_cap

        if use_time_dependent_bodies == 0:
            return _rkn_adaptive_step_numba(
                px,
                py,
                vx,
                vy,
                dt,
                min_dt,
                max_dt,
                rtol,
                atol_pos,
                atol_vel,
                safety,
                min_factor,
                max_factor,
                max_rejects,
                ref_enabled,
                ref_px,
                ref_py,
                body_x,
                body_y,
                body_m,
                body_fixed,
                G,
            )

        if (not math.isfinite(min_dt)) or min_dt <= 0.0:
            min_dt = 1e-9
        if (not math.isfinite(max_dt)) or max_dt <= 0.0:
            max_dt = min_dt
        if max_dt < min_dt:
            max_dt = min_dt
        if (not math.isfinite(rtol)) or rtol < 0.0:
            rtol = 0.0
        if (not math.isfinite(atol_pos)) or atol_pos <= 0.0:
            atol_pos = 1e-12
        if (not math.isfinite(atol_vel)) or atol_vel <= 0.0:
            atol_vel = 1e-12
        if (not math.isfinite(safety)) or safety <= 0.0:
            safety = 0.9
        if (not math.isfinite(min_factor)) or min_factor <= 0.0:
            min_factor = 0.2
        if (not math.isfinite(max_factor)) or max_factor < min_factor:
            max_factor = min_factor
        if max_rejects < 0:
            max_rejects = 0

        step_dt = dt
        if (not math.isfinite(step_dt)) or step_dt <= 0.0:
            step_dt = max_dt
        if step_dt < min_dt:
            step_dt = min_dt
        if step_dt > max_dt:
            step_dt = max_dt

        rejected_count = 0

        while True:
            half_dt = 0.5 * step_dt

            full_px, full_py, full_vx, full_vy = _rkn4_step_time_numba(
                px, py, vx, vy, local_t, step_dt, ref_enabled, ref_index, ref_px, ref_py,
                body_x, body_y, body_m, body_fixed, body_scripted, body_a, body_e,
                body_theta, body_arg, body_parent, G, use_time_dependent_bodies,
                body_memo
            )
            half1_px, half1_py, half1_vx, half1_vy = _rkn4_step_time_numba(
                px, py, vx, vy, local_t, half_dt, ref_enabled, ref_index, ref_px, ref_py,
                body_x, body_y, body_m, body_fixed, body_scripted, body_a, body_e,
                body_theta, body_arg, body_parent, G, use_time_dependent_bodies,
                body_memo
            )
            half2_px, half2_py, half2_vx, half2_vy = _rkn4_step_time_numba(
                half1_px, half1_py, half1_vx, half1_vy, local_t + half_dt, half_dt,
                ref_enabled, ref_index, ref_px, ref_py, body_x, body_y, body_m, body_fixed,
                body_scripted, body_a, body_e, body_theta, body_arg, body_parent, G,
                use_time_dependent_bodies, body_memo
            )

            finite_state = (
                math.isfinite(full_px)
                and math.isfinite(full_py)
                and math.isfinite(full_vx)
                and math.isfinite(full_vy)
                and math.isfinite(half2_px)
                and math.isfinite(half2_py)
                and math.isfinite(half2_vx)
                and math.isfinite(half2_vy)
            )

            if finite_state:
                pos_dx = half2_px - full_px
                pos_dy = half2_py - full_py
                vel_dx = half2_vx - full_vx
                vel_dy = half2_vy - full_vy

                pos_err = math.sqrt(pos_dx * pos_dx + pos_dy * pos_dy) / 15.0
                vel_err = math.sqrt(vel_dx * vel_dx + vel_dy * vel_dy) / 15.0

                cur_r = math.sqrt(px * px + py * py)
                next_r = math.sqrt(half2_px * half2_px + half2_py * half2_py)
                cur_speed = math.sqrt(vx * vx + vy * vy)
                next_speed = math.sqrt(half2_vx * half2_vx + half2_vy * half2_vy)
                motion_scale = cur_speed * step_dt

                pos_ref = cur_r
                if next_r > pos_ref:
                    pos_ref = next_r
                if motion_scale > pos_ref:
                    pos_ref = motion_scale
                if pos_ref < 1.0:
                    pos_ref = 1.0

                vel_ref = cur_speed
                if next_speed > vel_ref:
                    vel_ref = next_speed
                if vel_ref < 1.0:
                    vel_ref = 1.0

                pos_scale = atol_pos + rtol * pos_ref
                vel_scale = atol_vel + rtol * vel_ref
                if pos_scale <= 0.0 or not math.isfinite(pos_scale):
                    pos_scale = 1e-30
                if vel_scale <= 0.0 or not math.isfinite(vel_scale):
                    vel_scale = 1e-30

                pos_norm = pos_err / pos_scale
                vel_norm = vel_err / vel_scale
                err_norm = pos_norm
                if vel_norm > err_norm:
                    err_norm = vel_norm
            else:
                err_norm = 1e300

            if math.isfinite(err_norm) and err_norm <= 1.0:
                if err_norm <= 1e-300:
                    factor = max_factor
                else:
                    factor = safety * err_norm ** (-0.2)
                    if factor < min_factor:
                        factor = min_factor
                    if factor > max_factor:
                        factor = max_factor

                proposed_next_dt = step_dt * factor
                if proposed_next_dt < min_dt:
                    proposed_next_dt = min_dt
                if proposed_next_dt > max_dt:
                    proposed_next_dt = max_dt

                return (
                    half2_px,
                    half2_py,
                    half2_vx,
                    half2_vy,
                    step_dt,
                    proposed_next_dt,
                    err_norm,
                    1,
                    rejected_count,
                    0,
                )

            if not math.isfinite(err_norm):
                err_norm = 1e300

            if step_dt <= min_dt * (1.0 + 1e-12):
                return (px, py, vx, vy, 0.0, min_dt, err_norm, 0, rejected_count, 6)

            if rejected_count >= max_rejects:
                return (px, py, vx, vy, 0.0, step_dt, err_norm, 0, rejected_count, 2)

            if err_norm <= 1e-300:
                factor = min_factor
            else:
                factor = safety * err_norm ** (-0.2)
            if factor < min_factor:
                factor = min_factor
            if factor > max_factor:
                factor = max_factor

            next_dt = step_dt * factor
            if next_dt >= step_dt:
                next_dt = step_dt * min_factor
            if next_dt < min_dt:
                next_dt = min_dt
            if next_dt > max_dt:
                next_dt = max_dt

            rejected_count += 1
            step_dt = next_dt


    @njit(cache=True, nogil=True, fastmath=True)
    def _compute_distance_points_rkn_numba(
        init_px,
        init_py,
        init_vx,
        init_vy,
        ref_enabled,
        ref_px,
        ref_py,
        body_x,
        body_y,
        body_m,
        body_fixed,
        body_scripted,
        body_a,
        body_e,
        body_theta,
        body_arg,
        body_parent,
        G,
        base_dt,
        precision,
        max_points,
        max_iters,
        min_dt,
        max_dt,
        rtol,
        atol_pos,
        atol_vel,
        safety,
        min_factor,
        max_factor,
        max_rejects,
        use_time_dependent_bodies,
        ref_index,
        init_t,
        init_accumulated,
        init_proposed_dt,
        use_body_memo,
        max_dt_floor,
        timescale_divisor,
    ):
        # init_t / init_accumulated / init_proposed_dt machen den kernel
        # FORTSETZBAR: mit dem zustand, den ein frueherer lauf in stats[7:]
        # hinterlassen hat, rechnet er exakt dort weiter, wo er aufgehoert
        # hat -- dieselbe schnappschuss-epoche, dieselbe schrittweite,
        # derselbe reststrecken-zaehler. Das ist die grundlage dafuer, die
        # vorhersage im zeitraffer hinten stueckweise zu verlaengern, statt
        # sie periodisch ganz neu zu rechnen.
        # Spalten 3/4 sind die GESCHWINDIGKEIT am ausgegebenen punkt. Sie
        # kostet nichts -- die emissions-schleife unten rechnet sie ohnehin
        # (als ableitung desselben Hermite-polynoms, mit dem sie die position
        # interpoliert) und warf bisher alle bis auf die letzte weg. Mit ihr
        # ist die punkteliste keine folge von positionen mehr, sondern eine
        # stueckweise KUBISCHE kurve: der renderer kann sie zur zeichenzeit
        # beliebig fein auswerten, ohne dass hier ein schritt mehr faellt.
        out = np.empty((max_points, 5), dtype=np.float64)
        out[0, 0] = init_px
        out[0, 1] = init_py
        out[0, 2] = init_t
        out[0, 3] = init_vx
        out[0, 4] = init_vy

        # Notizblock fuer koerperpositionen: [t, x, y, gueltig] je koerper,
        # EINMAL fuer den ganzen lauf angelegt und ueber alle schritte hinweg
        # gueltig. Nullen heisst "noch nichts gerechnet" -- ein NaN-merker
        # waere unter fastmath wirkungslos, siehe
        # _body_position_at_time_numba. Die kepler-aufstellung der
        # koerper ist 99 % der rechenzeit dieses kernels -- gemessen 61.7 ms
        # gegen 0.6 ms mit eingefrorenen koerpern -- und ein grossteil davon
        # war reine wiederholung derselben zeit. Siehe
        # _body_position_at_time_numba.
        # `use_body_memo = 0` legt ihn mit null zeilen an: dann greift in
        # _body_position_at_time_numba kein einziger treffer und der kernel
        # rechnet exakt wie vor der einfuehrung. Das ist der A/B-schalter fuer
        # den bit-vergleich (Predictor.use_body_memo), nach demselben muster
        # wie world.use_fast_integrator.
        _memo_rows = body_x.shape[0] if use_body_memo != 0 else 0
        body_memo = np.zeros((_memo_rows, 10), dtype=np.float64)
        # Vorlauf: die zeitunabhaengigen bahngroessen EINMAL je koerper.
        # Spalte 9 traegt das ergebnis: 1 = brauchbar, -1 = bahn unbrauchbar
        # (dann liefert die auswertung wie zuvor sofort ok = 0).
        for _bi in range(_memo_rows):
            (_m0, _mm, _s1e2, _ca, _sa, _cok) = _body_kepler_constants_numba(
                _bi, body_m, body_a, body_e, body_theta, body_arg, body_parent, G,
            )
            if _cok == 0:
                body_memo[_bi, 9] = -1.0
            else:
                body_memo[_bi, 4] = _m0
                body_memo[_bi, 5] = _mm
                body_memo[_bi, 6] = _s1e2
                body_memo[_bi, 7] = _ca
                body_memo[_bi, 8] = _sa
                body_memo[_bi, 9] = 1.0

        stats = np.zeros(14, dtype=np.float64)

        # Fortsetz-punkt = LETZTER AUSGEGEBENER punkt (nicht das ende des
        # letzten integrationsschritts). Nur so ist die reststrecke dort
        # definitionsgemaess 0 und kann beim fortsetzen nicht groesser als
        # der punktabstand werden -- genau daran scheiterte die naht sonst.
        resume_px = init_px
        resume_py = init_py
        resume_vx = init_vx
        resume_vy = init_vy
        resume_t = init_t

        count = 1
        px = init_px
        py = init_py
        vx = init_vx
        vy = init_vy
        t = init_t
        accumulated = init_accumulated
        proposed_dt = init_proposed_dt if init_proposed_dt > 0.0 else base_dt

        accepted_steps = 0.0
        rejected_steps = 0.0
        min_used_dt = 1e300
        max_used_dt = 0.0
        max_error_norm = 0.0
        failure_code = 0.0

        for _ in range(max_iters):
            if count >= max_points:
                break
            if (
                not math.isfinite(px)
                or not math.isfinite(py)
                or not math.isfinite(vx)
                or not math.isfinite(vy)
            ):
                failure_code = 1.0
                break

            if use_time_dependent_bodies == 0:
                (
                    next_px,
                    next_py,
                    next_vx,
                    next_vy,
                    used_dt,
                    next_proposed_dt,
                    err_norm,
                    accepted_flag,
                    rejected_count,
                    step_failure_code,
                ) = _rkn_adaptive_step_numba(
                    px,
                    py,
                    vx,
                    vy,
                    proposed_dt,
                    min_dt,
                    max_dt,
                    rtol,
                    atol_pos,
                    atol_vel,
                    safety,
                    min_factor,
                    max_factor,
                    max_rejects,
                    ref_enabled,
                    ref_px,
                    ref_py,
                    body_x,
                    body_y,
                    body_m,
                    body_fixed,
                    G,
                )
            else:
                (
                    next_px,
                    next_py,
                    next_vx,
                    next_vy,
                    used_dt,
                    next_proposed_dt,
                    err_norm,
                    accepted_flag,
                    rejected_count,
                    step_failure_code,
                ) = _rkn_adaptive_step_time_numba(
                    px,
                    py,
                    vx,
                    vy,
                    t,
                    proposed_dt,
                    min_dt,
                    max_dt,
                    rtol,
                    atol_pos,
                    atol_vel,
                    safety,
                    min_factor,
                    max_factor,
                    max_rejects,
                    ref_enabled,
                    ref_index,
                    ref_px,
                    ref_py,
                    body_x,
                    body_y,
                    body_m,
                    body_fixed,
                    body_scripted,
                    body_a,
                    body_e,
                    body_theta,
                    body_arg,
                    body_parent,
                    G,
                    use_time_dependent_bodies,
                    body_memo,
                    max_dt_floor,
                    timescale_divisor,
                )

            rejected_steps += float(rejected_count)

            if accepted_flag == 0:
                failure_code = float(step_failure_code)
                if failure_code == 0.0:
                    failure_code = 2.0
                if math.isfinite(err_norm) and err_norm > max_error_norm:
                    max_error_norm = err_norm
                break

            if (
                used_dt <= 0.0
                or not math.isfinite(used_dt)
                or not math.isfinite(next_px)
                or not math.isfinite(next_py)
                or not math.isfinite(next_vx)
                or not math.isfinite(next_vy)
            ):
                failure_code = 3.0
                break

            accepted_steps += 1.0
            if used_dt < min_used_dt:
                min_used_dt = used_dt
            if used_dt > max_used_dt:
                max_used_dt = used_dt
            if math.isfinite(err_norm) and err_norm > max_error_norm:
                max_error_norm = err_norm

            seg_dx = next_px - px
            seg_dy = next_py - py
            seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)

            if seg_len > 0.0 and math.isfinite(seg_len):
                placed = 0.0
                rem_len = seg_len

                while rem_len + accumulated >= precision and count < max_points:
                    if rem_len <= 0.0:
                        break

                    distance_to_place = precision - accumulated
                    placed += distance_to_place
                    s = placed / seg_len
                    if s < 0.0:
                        s = 0.0
                    if s > 1.0:
                        s = 1.0

                    linear_px = px + seg_dx * s
                    linear_py = py + seg_dy * s

                    s2 = s * s
                    s3 = s2 * s
                    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
                    h10 = s3 - 2.0 * s2 + s
                    h01 = -2.0 * s3 + 3.0 * s2
                    h11 = s3 - s2

                    sample_px = h00 * px + h10 * used_dt * vx + h01 * next_px + h11 * used_dt * next_vx
                    sample_py = h00 * py + h10 * used_dt * vy + h01 * next_py + h11 * used_dt * next_vy
                    if not math.isfinite(sample_px) or not math.isfinite(sample_py):
                        sample_px = linear_px
                        sample_py = linear_py

                    sample_t = t + s * used_dt

                    if (
                        not math.isfinite(sample_px)
                        or not math.isfinite(sample_py)
                        or not math.isfinite(sample_t)
                    ):
                        failure_code = 3.0
                        break

                    # Geschwindigkeit am ausgegebenen punkt: ableitung
                    # DESSELBEN Hermite-polynoms, mit dem oben die position
                    # interpoliert wurde -- also konsistent, nicht genaehert.
                    # Sie wird VOR dem schreiben gerechnet, weil sie jetzt
                    # mit in die zeile geht (spalten 3/4) und nicht mehr nur
                    # den fortsetz-zustand fuellt.
                    d00 = 6.0 * s2 - 6.0 * s
                    d10 = 3.0 * s2 - 4.0 * s + 1.0
                    d01 = -6.0 * s2 + 6.0 * s
                    d11 = 3.0 * s2 - 2.0 * s
                    if used_dt != 0.0:
                        resume_vx = (d00 * px + d01 * next_px) / used_dt + d10 * vx + d11 * next_vx
                        resume_vy = (d00 * py + d01 * next_py) / used_dt + d10 * vy + d11 * next_vy
                    else:
                        resume_vx = vx
                        resume_vy = vy
                    resume_px = sample_px
                    resume_py = sample_py
                    resume_t = sample_t

                    out[count, 0] = sample_px
                    out[count, 1] = sample_py
                    out[count, 2] = sample_t
                    out[count, 3] = resume_vx
                    out[count, 4] = resume_vy
                    count += 1

                    accumulated = 0.0
                    rem_len = seg_len - placed

                if failure_code != 0.0:
                    break

                # Reststrecke IMMER mitzaehlen. Bisher geschah das nur, wenn
                # die emissions-schleife regulaer endete; brach sie ab, weil
                # das punktbudget voll war, ging die restliche strecke des
                # segments verloren. Fuer einen einmaligen lauf war das
                # folgenlos (danach bricht auch die aeussere schleife ab und
                # `accumulated` wird nicht mehr gelesen) -- beim FORTSETZEN
                # dagegen sass der fortsetz-punkt dann bis zu einer ganzen
                # schrittweite hinter dem letzten ausgegebenen punkt, und an
                # der nahtstelle klaffte eine luecke (gemessen 2.6e7 m bei
                # 1e6 m punktabstand).
                accumulated += rem_len

            px = next_px
            py = next_py
            vx = next_vx
            vy = next_vy
            t += used_dt
            proposed_dt = next_proposed_dt

        if min_used_dt == 1e300:
            min_used_dt = 0.0
        if count < max_points and failure_code == 0.0:
            failure_code = 4.0

        stats[0] = accepted_steps
        stats[1] = rejected_steps
        stats[2] = min_used_dt
        stats[3] = max_used_dt
        stats[4] = max_error_norm
        stats[5] = failure_code
        stats[6] = t
        # Fortsetz-zustand (siehe kopf der funktion). Reststrecke ist am
        # ausgegebenen punkt per definition 0.
        stats[7] = resume_px
        stats[8] = resume_py
        stats[9] = resume_vx
        stats[10] = resume_vy
        stats[11] = 0.0
        stats[12] = proposed_dt
        stats[13] = resume_t

        return out, count, stats


    @njit(cache=True, nogil=True, fastmath=True)
    def _refine_apsis_numba(pts, d2_arr, idx, use_tangents):
        # parabolische verfeinerung des diskreten extremums bei `idx`: die
        # rohe "nächster punkt"-wahl hat einen quantisierungsfehler von der
        # größenordnung (punktabstand)^2 / (2*krümmungsradius) — der bei
        # jedem predictor-neuaufbau anders ausfällt, weil das arc-length-
        # sampling-raster jedes mal neu am schiff verankert wird (anderer
        # phasenversatz zur wahren apsis). das lässt den angezeigten
        # Pe/Ap-abstand bei UNVERÄNDERTER bahn zwischen neuberechnungen
        # spürbar schwanken (stärker an einer scharfen periapsis, schwächer
        # an einer flachen apoapsis). fit einer parabel durch die drei
        # punkte um idx liefert den echten scheitel und eliminiert das.
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
        # ihrer SEHNE.
        #
        # Hier stand eine lineare interpolation entlang des nachbar-segments,
        # mit der begruendung, der marker liege damit "exakt auf der
        # gezeichneten linie". Gezeichnet wird die linie aber als kubisches
        # Hermite-polynom durch dieselben punkte (`_hermite_refine_world`),
        # und das weicht von der sehne um die pfeilhoehe ab. Solange der
        # punktabstand klein gegen den kruemmungsradius ist, sind das
        # bruchteile eines pixels; auf einem langen horizont ist es das nicht
        # mehr: bei punktabstand 1.125e8 m und einer periapsis bei 1.69e8 m
        # (Erde -> Neptun, siehe .claude/rules/predictor.md) betraegt die
        # pfeilhoehe R*(1-cos(c/2R)) = 9.4e6 m -- rund 10 px im bild. Der
        # marker sass damit sichtbar NEBEN der linie, mal darueber, mal
        # darunter, je nachdem wie das abtastraster gerade zur wahren apsis
        # stand. Auf der kubik ausgewertet liegt er dort per konstruktion.
        #
        # Bezier-form des Hermite-polynoms, wortgleich zu
        # rendering._hermite_refine_world:
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
        # OB DAS SO IST, WIRD DRAUSSEN ENTSCHIEDEN UND HEREINGEREICHT -- hier
        # laesst es sich nicht pruefen. Dieser kernel ist `fastmath=True`,
        # also verspricht er LLVM, dass keine NaN auftreten (`nnan`), und
        # dann darf jede NaN-abfrage wegoptimiert werden. Gemessen mit
        # numba auf dieser maschine: unter fastmath liefert BEIDES
        #     math.isfinite(nan) -> True        nan == nan -> True
        # Ein guard an dieser stelle haette also nichts abgefangen und die
        # kubik mit NaN gerechnet -- der marker waere verschwunden. Es ist
        # dieselbe falle, die weiter oben schon die `valid`-spalte des
        # body_memo erzwungen hat.
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
        # predictor-punkte (pts: (n,3) mit x, y, absoluter sim-zeit). der
        # diskrete extrempunkt wird per parabel-fit über seine nachbarn zum
        # wahren scheitel verfeinert (_refine_apsis_numba) — sonst hängt der
        # angezeigte Pe/Ap-abstand vom zufälligen phasenversatz des arc-
        # length-samplings ab (das raster wird bei jedem predictor-neuaufbau
        # neu am schiff verankert) und schwankt bei unveränderter bahn.
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
        # Lokal angelegt, NICHT als modul-konstante: numba typisiert ein
        # globales array `readonly`, und dann scheitert schon die
        # uebersetzung von _body_position_at_time_numba an dessen (hier nie
        # erreichten) schreibzugriffen -- siehe _no_body_memo().
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
        # rechnung entstammen. Im zeitraffer-halt stellt `_hold_advance` der
        # gehaltenen kurve die tatsächliche schiffsposition als kopf voran --
        # die stammt aus der WELT, nicht aus dieser kurve, und weicht deshalb
        # um ein vielfaches eines normalen punktschritts von ihr ab (gemessen
        # 37 km gegen 1.3 km reguläre schrittweite in einer erdumlaufbahn).
        # Als startwert des trends gelesen kippt dieser sprung die richtung
        # und der scan meldet ein extremum bei index 1 -- ein Ap/Pe-marker
        # DIREKT AUF DEM SCHIFF, der von frame zu frame an- und ausgeht, weil
        # der sprung die hysterese mal reisst und mal nicht. Der kopf wird
        # deshalb gar nicht erst gelesen; `best_idx > skip_head` unterdrückt
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


    @njit(cache=True, nogil=True, fastmath=True)
    def _leapfrog_step_numba(
        px,
        py,
        vx,
        vy,
        ax,
        ay,
        dt,
        ref_enabled,
        ref_ax,
        ref_ay,
        body_x,
        body_y,
        body_m,
        body_fixed,
        G,
    ):
        hvx = vx + 0.5 * ax * dt
        hvy = vy + 0.5 * ay * dt

        next_px = px + hvx * dt
        next_py = py + hvy * dt

        next_ax_raw, next_ay_raw, nearest_r, acc_mag = _compute_acc_nearest_numba(
            next_px, next_py, body_x, body_y, body_m, body_fixed, G
        )
        next_ax = next_ax_raw - ref_ax
        next_ay = next_ay_raw - ref_ay

        next_vx = hvx + 0.5 * next_ax * dt
        next_vy = hvy + 0.5 * next_ay * dt

        return next_px, next_py, next_vx, next_vy, next_ax, next_ay, nearest_r, acc_mag


    @njit(cache=True, nogil=True, fastmath=True)
    def _compute_distance_points_aspi_numba(
        init_px,
        init_py,
        init_vx,
        init_vy,
        ref_enabled,
        ref_px,
        ref_py,
        body_x,
        body_y,
        body_m,
        body_fixed,
        G,
        base_dt,
        precision,
        max_points,
        max_iters,
        min_dt,
        max_dt,
        safety_g,
        safety_m,
        close_acc_threshold,
        use_rk4_fallback,
    ):
        # Fuenf spalten wie im rkn-kernel, aber die geschwindigkeit bleibt
        # NaN: dieser pfad setzt seine punkte LINEAR auf die schrittsehne,
        # es gibt also gar keine tangente, die sie beschreiben koennte. NaN
        # sagt dem renderer genau das -- er zeichnet diese abschnitte dann
        # als geraden statt eine kruemmung zu erfinden, die die punkte nicht
        # haben.
        out = np.empty((max_points, 5), dtype=np.float64)
        out[0, 0] = init_px
        out[0, 1] = init_py
        out[0, 2] = 0.0
        out[0, 3] = np.nan
        out[0, 4] = np.nan

        count = 1
        px = init_px
        py = init_py
        vx = init_vx
        vy = init_vy
        accumulated = 0.0
        t = 0.0

        ref_ax = 0.0
        ref_ay = 0.0
        if ref_enabled != 0:
            ref_ax, ref_ay = _compute_acc_numba(ref_px, ref_py, body_x, body_y, body_m, body_fixed, G)

        raw_ax, raw_ay, nearest_r, acc_mag = _compute_acc_nearest_numba(
            px, py, body_x, body_y, body_m, body_fixed, G
        )
        ax = raw_ax - ref_ax
        ay = raw_ay - ref_ay

        # ASPI is for visual prediction, not a replacement for ship physics.
        # The trajectory is still sequential; speed comes from smarter steps,
        # not from point-level parallelism.
        for _ in range(max_iters):
            if count >= max_points:
                break
            if (
                not math.isfinite(px)
                or not math.isfinite(py)
                or not math.isfinite(vx)
                or not math.isfinite(vy)
                or not math.isfinite(ax)
                or not math.isfinite(ay)
            ):
                break

            if ref_enabled != 0:
                ref_ax, ref_ay = _compute_acc_numba(ref_px, ref_py, body_x, body_y, body_m, body_fixed, G)
            else:
                ref_ax = 0.0
                ref_ay = 0.0

            speed = math.sqrt(vx * vx + vy * vy)
            dt_g = safety_g * math.sqrt(nearest_r / max(acc_mag, 1e-30))
            dt_m = safety_m * precision / max(speed, 1e-30)

            step_dt = max_dt
            if dt_g < step_dt:
                step_dt = dt_g
            if dt_m < step_dt:
                step_dt = dt_m

            if not math.isfinite(step_dt) or step_dt <= 0.0:
                step_dt = base_dt
            if not math.isfinite(step_dt) or step_dt <= 0.0:
                step_dt = min_dt

            if step_dt < min_dt:
                step_dt = min_dt
            if step_dt > max_dt:
                step_dt = max_dt

            if use_rk4_fallback and acc_mag > close_acc_threshold:
                # RK4 is kept as a local-accuracy fallback in strong gravity.
                next_px, next_py, next_vx, next_vy = _rk4_step_numba(
                    px,
                    py,
                    vx,
                    vy,
                    step_dt,
                    ref_enabled,
                    ref_px,
                    ref_py,
                    body_x,
                    body_y,
                    body_m,
                    body_fixed,
                    G,
                )
                next_raw_ax, next_raw_ay, next_nearest_r, next_acc_mag = _compute_acc_nearest_numba(
                    next_px, next_py, body_x, body_y, body_m, body_fixed, G
                )
                next_ax = next_raw_ax - ref_ax
                next_ay = next_raw_ay - ref_ay
            else:
                # Velocity Verlet/KDK leapfrog is symplectic and behaves well
                # for long visual orbit predictions with bounded step sizes.
                (
                    next_px,
                    next_py,
                    next_vx,
                    next_vy,
                    next_ax,
                    next_ay,
                    next_nearest_r,
                    next_acc_mag,
                ) = _leapfrog_step_numba(
                    px,
                    py,
                    vx,
                    vy,
                    ax,
                    ay,
                    step_dt,
                    ref_enabled,
                    ref_ax,
                    ref_ay,
                    body_x,
                    body_y,
                    body_m,
                    body_fixed,
                    G,
                )

            if (
                not math.isfinite(next_px)
                or not math.isfinite(next_py)
                or not math.isfinite(next_vx)
                or not math.isfinite(next_vy)
                or not math.isfinite(next_ax)
                or not math.isfinite(next_ay)
            ):
                break

            seg_dx = next_px - px
            seg_dy = next_py - py
            seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)

            if seg_len <= 0.0 or not math.isfinite(seg_len):
                px = next_px
                py = next_py
                vx = next_vx
                vy = next_vy
                ax = next_ax
                ay = next_ay
                nearest_r = next_nearest_r
                acc_mag = next_acc_mag
                t += step_dt
                continue

            local_px = px
            local_py = py
            rem_dx = seg_dx
            rem_dy = seg_dy
            rem_len = seg_len

            while rem_len + accumulated >= precision and count < max_points:
                if rem_len <= 0.0:
                    break

                distance_to_place = precision - accumulated
                frac = distance_to_place / rem_len

                sample_px = local_px + rem_dx * frac
                sample_py = local_py + rem_dy * frac
                sample_t = t + frac * step_dt

                if (
                    not math.isfinite(sample_px)
                    or not math.isfinite(sample_py)
                    or not math.isfinite(sample_t)
                ):
                    break

                out[count, 0] = sample_px
                out[count, 1] = sample_py
                out[count, 2] = sample_t
                # Linear auf der sehne gesetzt -> keine tangente vorhanden.
                out[count, 3] = np.nan
                out[count, 4] = np.nan
                count += 1

                local_px = sample_px
                local_py = sample_py

                rem_dx = next_px - local_px
                rem_dy = next_py - local_py
                rem_len = math.sqrt(rem_dx * rem_dx + rem_dy * rem_dy)
                accumulated = 0.0

            if rem_len + accumulated < precision:
                accumulated += rem_len

            px = next_px
            py = next_py
            vx = next_vx
            vy = next_vy
            ax = next_ax
            ay = next_ay
            nearest_r = next_nearest_r
            acc_mag = next_acc_mag
            t += step_dt

        return out, count


    @njit(cache=True, nogil=True, fastmath=True)
    def _compute_distance_points_numba(
        init_px,
        init_py,
        init_vx,
        init_vy,
        ref_enabled,
        ref_px,
        ref_py,
        body_x,
        body_y,
        body_m,
        body_fixed,
        G,
        dt,
        precision,
        max_points,
        max_iters,
    ):
        # Wie im ASPI-kernel: fuenf spalten, geschwindigkeit NaN, weil die
        # punkte linear auf der schrittsehne sitzen.
        out = np.empty((max_points, 5), dtype=np.float64)
        out[0, 0] = init_px
        out[0, 1] = init_py
        out[0, 2] = 0.0
        out[0, 3] = np.nan
        out[0, 4] = np.nan

        count = 1
        px = init_px
        py = init_py
        vx = init_vx
        vy = init_vy
        accumulated = 0.0

        t = 0.0

        for _ in range(max_iters):
            if count >= max_points:
                break

            ref_ax = 0.0
            ref_ay = 0.0
            if ref_enabled != 0:
                ref_ax, ref_ay = _compute_acc_numba(ref_px, ref_py, body_x, body_y, body_m, body_fixed, G)

            k1_ax_raw, k1_ay_raw = _compute_acc_numba(px, py, body_x, body_y, body_m, body_fixed, G)
            k1_ax = k1_ax_raw - ref_ax
            k1_ay = k1_ay_raw - ref_ay
            k1_vx, k1_vy = vx, vy

            p2x = px + k1_vx * (dt / 2.0)
            p2y = py + k1_vy * (dt / 2.0)
            v2x = vx + k1_ax * (dt / 2.0)
            v2y = vy + k1_ay * (dt / 2.0)
            k2_ax_raw, k2_ay_raw = _compute_acc_numba(p2x, p2y, body_x, body_y, body_m, body_fixed, G)
            k2_ax = k2_ax_raw - ref_ax
            k2_ay = k2_ay_raw - ref_ay
            k2_vx, k2_vy = v2x, v2y

            p3x = px + k2_vx * (dt / 2.0)
            p3y = py + k2_vy * (dt / 2.0)
            v3x = vx + k2_ax * (dt / 2.0)
            v3y = vy + k2_ay * (dt / 2.0)
            k3_ax_raw, k3_ay_raw = _compute_acc_numba(p3x, p3y, body_x, body_y, body_m, body_fixed, G)
            k3_ax = k3_ax_raw - ref_ax
            k3_ay = k3_ay_raw - ref_ay
            k3_vx, k3_vy = v3x, v3y

            p4x = px + k3_vx * dt
            p4y = py + k3_vy * dt
            v4x = vx + k3_ax * dt
            v4y = vy + k3_ay * dt
            k4_ax_raw, k4_ay_raw = _compute_acc_numba(p4x, p4y, body_x, body_y, body_m, body_fixed, G)
            k4_ax = k4_ax_raw - ref_ax
            k4_ay = k4_ay_raw - ref_ay
            k4_vx, k4_vy = v4x, v4y

            next_px = px + (k1_vx + 2.0 * k2_vx + 2.0 * k3_vx + k4_vx) * (dt / 6.0)
            next_py = py + (k1_vy + 2.0 * k2_vy + 2.0 * k3_vy + k4_vy) * (dt / 6.0)
            next_vx = vx + (k1_ax + 2.0 * k2_ax + 2.0 * k3_ax + k4_ax) * (dt / 6.0)
            next_vy = vy + (k1_ay + 2.0 * k2_ay + 2.0 * k3_ay + k4_ay) * (dt / 6.0)

            seg_dx = next_px - px
            seg_dy = next_py - py
            seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)

            if seg_len <= 0.0:
                px = next_px
                py = next_py
                vx = next_vx
                vy = next_vy
                continue

            local_px = px
            local_py = py
            local_vx = vx
            local_vy = vy
            rem_dx = seg_dx
            rem_dy = seg_dy
            rem_len = seg_len

            while rem_len + accumulated >= precision and count < max_points:
                if rem_len <= 0.0:
                    break

                distance_to_place = precision - accumulated
                frac = distance_to_place / rem_len

                sample_px = local_px + rem_dx * frac
                sample_py = local_py + rem_dy * frac
                sample_t = t + frac * dt

                out[count, 0] = sample_px
                out[count, 1] = sample_py
                out[count, 2] = sample_t
                # Linear auf der sehne gesetzt -> keine tangente vorhanden.
                out[count, 3] = np.nan
                out[count, 4] = np.nan
                count += 1

                local_px = sample_px
                local_py = sample_py

                rem_dx = next_px - local_px
                rem_dy = next_py - local_py
                rem_len = math.sqrt(rem_dx * rem_dx + rem_dy * rem_dy)
                accumulated = 0.0

            if rem_len + accumulated < precision:
                accumulated += rem_len

            px = next_px
            py = next_py
            vx = next_vx
            vy = next_vy
            t += dt

        return out, count


    @njit(cache=True, nogil=True, fastmath=True)
    def _compute_distance_points_numba_state(
        init_px,
        init_py,
        init_vx,
        init_vy,
        init_t,
        ref_enabled,
        ref_px,
        ref_py,
        body_x,
        body_y,
        body_m,
        body_fixed,
        G,
        dt,
        precision,
        max_points,
        max_iters,
    ):
        # spalten: x, y, t, vx, vy
        out = np.empty((max_points, 5), dtype=np.float64)
        out[0, 0] = init_px
        out[0, 1] = init_py
        out[0, 2] = init_t
        out[0, 3] = init_vx
        out[0, 4] = init_vy

        count = 1
        px = init_px
        py = init_py
        vx = init_vx
        vy = init_vy
        accumulated = 0.0

        t = init_t

        for _ in range(max_iters):
            if count >= max_points:
                break

            ref_ax = 0.0
            ref_ay = 0.0
            if ref_enabled != 0:
                ref_ax, ref_ay = _compute_acc_numba(ref_px, ref_py, body_x, body_y, body_m, body_fixed, G)

            k1_ax_raw, k1_ay_raw = _compute_acc_numba(px, py, body_x, body_y, body_m, body_fixed, G)
            k1_ax = k1_ax_raw - ref_ax
            k1_ay = k1_ay_raw - ref_ay
            k1_vx, k1_vy = vx, vy

            p2x = px + k1_vx * (dt / 2.0)
            p2y = py + k1_vy * (dt / 2.0)
            v2x = vx + k1_ax * (dt / 2.0)
            v2y = vy + k1_ay * (dt / 2.0)
            k2_ax_raw, k2_ay_raw = _compute_acc_numba(p2x, p2y, body_x, body_y, body_m, body_fixed, G)
            k2_ax = k2_ax_raw - ref_ax
            k2_ay = k2_ay_raw - ref_ay
            k2_vx, k2_vy = v2x, v2y

            p3x = px + k2_vx * (dt / 2.0)
            p3y = py + k2_vy * (dt / 2.0)
            v3x = vx + k2_ax * (dt / 2.0)
            v3y = vy + k2_ay * (dt / 2.0)
            k3_ax_raw, k3_ay_raw = _compute_acc_numba(p3x, p3y, body_x, body_y, body_m, body_fixed, G)
            k3_ax = k3_ax_raw - ref_ax
            k3_ay = k3_ay_raw - ref_ay
            k3_vx, k3_vy = v3x, v3y

            p4x = px + k3_vx * dt
            p4y = py + k3_vy * dt
            v4x = vx + k3_ax * dt
            v4y = vy + k3_ay * dt
            k4_ax_raw, k4_ay_raw = _compute_acc_numba(p4x, p4y, body_x, body_y, body_m, body_fixed, G)
            k4_ax = k4_ax_raw - ref_ax
            k4_ay = k4_ay_raw - ref_ay
            k4_vx, k4_vy = v4x, v4y

            next_px = px + (k1_vx + 2.0 * k2_vx + 2.0 * k3_vx + k4_vx) * (dt / 6.0)
            next_py = py + (k1_vy + 2.0 * k2_vy + 2.0 * k3_vy + k4_vy) * (dt / 6.0)
            next_vx = vx + (k1_ax + 2.0 * k2_ax + 2.0 * k3_ax + k4_ax) * (dt / 6.0)
            next_vy = vy + (k1_ay + 2.0 * k2_ay + 2.0 * k3_ay + k4_ay) * (dt / 6.0)

            seg_dx = next_px - px
            seg_dy = next_py - py
            seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)

            if seg_len <= 0.0:
                px = next_px
                py = next_py
                vx = next_vx
                vy = next_vy
                t += dt
                continue

            local_px = px
            local_py = py
            local_vx = vx
            local_vy = vy
            rem_dx = seg_dx
            rem_dy = seg_dy
            rem_len = seg_len

            while rem_len + accumulated >= precision and count < max_points:
                if rem_len <= 0.0:
                    break

                distance_to_place = precision - accumulated
                frac = distance_to_place / rem_len

                sample_px = local_px + rem_dx * frac
                sample_py = local_py + rem_dy * frac
                sample_t = t + frac * dt
                sample_vx = local_vx + (next_vx - local_vx) * frac
                sample_vy = local_vy + (next_vy - local_vy) * frac

                out[count, 0] = sample_px
                out[count, 1] = sample_py
                out[count, 2] = sample_t
                out[count, 3] = sample_vx
                out[count, 4] = sample_vy
                count += 1

                local_px = sample_px
                local_py = sample_py
                local_vx = sample_vx
                local_vy = sample_vy

                rem_dx = next_px - local_px
                rem_dy = next_py - local_py
                rem_len = math.sqrt(rem_dx * rem_dx + rem_dy * rem_dy)
                accumulated = 0.0

            if rem_len + accumulated < precision:
                accumulated += rem_len

            px = next_px
            py = next_py
            vx = next_vx
            vy = next_vy
            t += dt

        return out, count


class Predictor:
    def __init__(
        self,
        num_points=5000,
        dt=60.0,
        workers=None,
        debug=True,
        recompute_every_update=True,
        precision=1e6,
        length=None,
        use_numba=True,
        async_compute=True,
        rolling_mode=None,
        integrator_mode="rkn",
        aspi_min_dt=1.0,
        aspi_max_dt=120.0,
        aspi_safety_g=0.05,
        aspi_safety_m=0.5,
        aspi_close_acc_threshold=0.02,
        aspi_use_rk4_fallback=True,
        rkn_min_dt=0.1,
        rkn_max_dt=1500.0,
        rkn_rtol=1e-7,
        rkn_atol_pos=10.0,
        rkn_atol_vel=1e-4,
        rkn_safety=0.9,
        rkn_min_factor=0.2,
        rkn_max_factor=5.0,
        rkn_max_rejects=32,
        strict_snapshot_matching=True,
        use_time_dependent_bodies=True,
        use_reference_acceleration_correction=False,
    ):
        
        self.num_points = int(num_points)
        self.dt = float(dt)
        self.precision = float(precision)
        self.base_precision = float(precision)
        self.length = None if length is None else float(length)
        self.integrator_mode = self._normalize_integrator_mode(integrator_mode)
        self.aspi_min_dt = float(aspi_min_dt)
        self.aspi_max_dt = float(aspi_max_dt)
        self.aspi_safety_g = float(aspi_safety_g)
        self.aspi_safety_m = float(aspi_safety_m)
        self.aspi_close_acc_threshold = float(aspi_close_acc_threshold)
        self.aspi_use_rk4_fallback = bool(aspi_use_rk4_fallback)
        self.rkn_min_dt = float(rkn_min_dt)
        self.rkn_max_dt = float(rkn_max_dt)
        self.rkn_rtol = float(rkn_rtol)
        self.rkn_atol_pos = float(rkn_atol_pos)
        self.rkn_atol_vel = float(rkn_atol_vel)
        self.rkn_safety = float(rkn_safety)
        self.rkn_min_factor = float(rkn_min_factor)
        self.rkn_max_factor = float(rkn_max_factor)
        self.rkn_max_rejects = int(rkn_max_rejects)
        # Intervall-gekoppelte schrittweite (Option A): wird das abtast-
        # intervall (effektive precision) gröber als base_precision, darf der
        # integrator proportional größere schritte mit proportional lockererer
        # toleranz nehmen, sodass die schrittzahl ~an die punktzahl statt an die
        # bogenlänge gekoppelt ist. kosten bleiben ~konstant über das intervall;
        # drift wächst bewusst mit dem intervall. nahe vorbeiflügen bleibt die
        # adaptive kontrolle wirksam. bei/unter base_precision: exakte identität.
        # Off by default: the look-ahead horizon is set by `length`, not by
        # `precision` (see _get_target_point_cap / get_display_length). Interval
        # coupling's premise ("coarser spacing => longer horizon => trade
        # accuracy for cost") no longer holds once the two are decoupled, so
        # coarsening `precision` must stay purely cosmetic — same cost, same
        # accuracy. Re-enable only for a deliberate fast, low-accuracy preview.
        self.rkn_interval_coupling = False
        self.rkn_interval_tol_exponent = 8.0
        # Horizon-scaled far-field step size. A long look-ahead over a smooth arc
        # is otherwise integrated at the fixed max_dt cap, costing ~arc/max_dt
        # steps (e.g. ~240 ms at a ~240-day horizon). Scaling max_dt with the
        # horizon to target a bounded step budget keeps long-horizon compute
        # roughly constant. Tied to the HORIZON, not `precision`, so the spacing
        # decouple is preserved; floored at the preset max_dt so short horizons
        # stay fully accurate; capped by the ceiling for close-approach safety
        # (the adaptive tolerance + step-doubling still refine near planets).
        # See _make_snapshot.
        self.rkn_adaptive_far_maxdt = True
        self.rkn_far_field_target_steps = 1250.0
        # Bruchteil der bahn-zeitskala (sqrt(r/|g|) = T/2pi auf der kreisbahn),
        # den ein integrator-schritt hoechstens ueberdecken darf, wenn der
        # horizont die decke anhebt. Siehe _make_snapshot.
        self.rkn_max_dt_timescale_divisor = 30.0
        # Absolute obergrenze der schrittdecke. Sie war 30000 s, solange die
        # bahn-klammer GLOBAL war und deshalb auf einer abflugbahn ausfiel --
        # dann war dies der einzige schutz. Ortlich gerechnet binden bereits
        # zwei PHYSIKALISCHE schranken (`desired` aus dem horizont und
        # `t_char_local/30` aus der bahn), und diese dritte, unphysikalische
        # war nur noch teuer: gemessen auf der Jupiter-abflugbahn bei 128x
        # 2848 schritte / 103 ms gegen 1280 / 51 ms bei 120000 s, wobei 1280
        # genau das schrittbudget `rkn_far_field_target_steps` ist -- darueber
        # bindet `desired` und der wert saettigt (300000 s misst dasselbe).
        # Der preis gegen eine referenz mit 300-s-decke: 2.501e6 -> 2.665e6 m
        # auf 1.28e12 m horizont, also **0.0025 -> 0.0027 px**, wenn der ganze
        # bogen im bild steht. Das NAHFELD ist bit-identisch (leo/ecc/mond bei
        # 30k gegen 300k: gleiche schrittzahl, groesste abweichung 0.0), weil
        # die bahn-klammer dort um zwei groessenordnungen tiefer liegt.
        self.rkn_max_dt_ceiling = 120000.0
        # ORTLICHE statt globale schrittdecke. False stellt den alten weg her:
        # `t_char/divisor` EINMAL am schiff gemessen und ueber den ganzen lauf
        # gelegt. Das ist der A/B-schalter fuer den vergleich (dieselbe rolle
        # wie `use_body_memo` und `world.use_fast_integrator`) -- mit ihm
        # zeigt `tests/warp_predictor_test.py` §24, dass beide wege auf jeder
        # bahn, die IHR REGIME NICHT VERLAESST, bit-identisch rechnen, und dass
        # der unterschied genau dort auftritt, wo er auftreten soll.
        self.use_local_step_ceiling = True
        # Gemessene MITTLERE inverse geschwindigkeit ueber den horizont (s/m):
        # zeitspanne des letzten laufs geteilt durch seine bogenlaenge. 0.0 =
        # noch unbekannt, dann faellt _make_snapshot auf die momentangeschwin-
        # digkeit zurueck. Siehe _make_snapshot; wird in _compute_from_snapshot
        # aus dem ergebnis nachgezogen und in reset() geloescht.
        self._horizon_time_per_arc = 0.0
        self.rkn_last_accepted_steps = 0
        self.rkn_last_rejected_steps = 0
        self.rkn_last_min_dt = 0.0
        self.rkn_last_max_dt = 0.0
        self.rkn_last_max_error_norm = 0.0
        self.rkn_last_failed = False
        self.rkn_last_failure_reason = ""
        self.strict_snapshot_matching = bool(strict_snapshot_matching)
        self.use_time_dependent_bodies = bool(use_time_dependent_bodies)
        self.use_reference_acceleration_correction = False
        self.debug_moving_sources = False
        # wall-clock duration (ms) of the most recent trajectory compute
        # (_compute_from_snapshot). In async mode this runs on a worker thread,
        # so it reflects the real line-calculation cost even though it overlaps
        # rendering. Read by the per-frame TIMING line in test.py.
        self.last_compute_ms = 0.0
        self._trajectory_version = 0
        self._last_seen_px = None
        self._last_seen_py = None
        self._last_seen_vx = None
        self._last_seen_vy = None
        self._last_seen_sim_time = None
        # Beschleunigung des letzten bildes -- daraus wird die KRUEMMUNG von g
        # ueber einen schritt geschaetzt, die schranke fuer den
        # schwerkraft-bereinigten rest (siehe _handle_trajectory_branch_change).
        self._last_seen_gx = None
        self._last_seen_gy = None
        self.velocity_invalidation_abs_tol = 1.0
        self.velocity_invalidation_rel_tol = 1e-5
        self.position_invalidation_abs_tol = 100.0
        self.sync_recompute_on_velocity_change = True
        # OBERGRENZE fuer gleichzeitig laufende vorhersagen unter schub. Wie
        # viele es tatsaechlich werden, ergibt sich aus dem messwert:
        # gebraucht werden `rechenzeit / bildzeit` laeufe, damit je bild genau
        # ein ergebnis fertig wird (siehe _target_pipeline_depth). Beim
        # gleitflug bleibt es immer bei einer einzigen rechnung.
        # 1 = abgeschaltet, wie vor der pipeline.
        self.thrust_pipeline_depth = 6
        # Wie viele FERTIGE, noch nicht eingewechselte ergebnisse warten
        # duerfen (siehe _swap_ready_result). Der klassische kompromiss eines
        # jitter-puffers: mehr puffer = gleichmaessigeres nachziehen, aber
        # aeltere linie. Gemessen an der periapsis unter vollschub, je 300
        # bilder, und der abstand der gezeichneten zur synchron gerechneten
        # linie:
        #
        #     0 -> 4 doppelschritte,  alter 2 s,   8.4 px abstand
        #     1 -> 1 doppelschritt,   alter 4 s,  10.3 px
        #     2 -> 0 doppelschritte,  alter 6 s,  17.5 px
        #
        # Voreinstellung 1: drei viertel der ausreisser weg fuer knapp 2 px.
        # Die STILLSTAENDE (3 je 300 bilder) bleiben in allen faellen -- sie
        # sind die andere haelfte derselben sache, denn es kann nie mehr als
        # ein ergebnis je bild ankommen. Ein stillstand faellt aber kaum auf,
        # ein doppelsprung schon.
        self.swap_backlog_max = 1
        # Gleitender mittelwert des abstands zwischen zwei update()-aufrufen,
        # also der bildzeit -- der predictor bekommt sie sonst nicht mit.
        self._update_interval_ms = 0.0
        self._last_update_ts = None
        self._pipeline_depth_used = 1
        # Notizblock fuer koerperpositionen im rkn-kernel. False rechnet jede
        # kepler-aufstellung wie frueher einzeln -- der A/B-schalter fuer den
        # bit-vergleich (tests/warp_predictor_test.py §10), nach demselben
        # muster wie world.use_fast_integrator. Gemessen 61.7 -> 15.8 ms.
        self.use_body_memo = True
        # A coasting ship's velocity changes by ~|g|*dt each step from gravity
        # alone; only a jump BEYOND that (real thrust) should invalidate the
        # trajectory. Without this the detector fires every frame and forces a
        # synchronous full recompute, bypassing the async worker. Margin covers
        # accel changing across the step / large sim_dt. See
        # _handle_trajectory_branch_change.
        self.gravity_dv_safety_factor = 4.0
        self.max_async_sim_age = max(2.0 * self.dt, 1.0)
        # Freshness of accepted async results is gated by WALL age (seconds since
        # the worker finished) rather than sim-time age: sim-time age scales with
        # sim_dt and horizon and wrongly rejected every result, forcing the
        # blocking sync path. The per-frame anchor + rebase keep position exact,
        # so a wall-fresh result is always safe to accept. See _swap_ready_result.
        self.max_async_wall_age = 1.5
        # Throttle redundant async re-submissions to ~this rate (wall seconds).
        # When a compute is cheaper than one frame, recompute_every_update would
        # otherwise submit 60x/s; ~25 Hz refresh looks identical and frees CPU.
        # Heavier computes self-throttle via single-flight (pending skips submit).
        self.async_submit_min_interval = 0.04
        self._last_submit_wall = 0.0

        self.points: "np.ndarray | list" = _empty_points()
        self.debug = debug
        # suppress frequent computed debug lines by default; set False to enable
        self._suppress_dbg_computed = True
        self.initialized = False
        self.recompute_every_update = recompute_every_update

        try:
            requested_workers = 1 if workers is None else int(workers)
        except Exception:
            requested_workers = 1
        self._requested_workers = int(requested_workers)
        self._predictor_worker_threads = 1
        self.workers = 1
        if self._requested_workers != 1 and self.debug:
            print(
                f"PRED_DBG_THREAD: requested_workers={self._requested_workers} clamped_workers=1",
                flush=True,
            )
        self.use_numba = bool(use_numba)

        self.auto_precision_from_zoom = True
        self.target_screen_step_px = 2.0
        self.min_precision = 1.0
        self._view_scale = None

        self.async_compute = bool(async_compute)
        if rolling_mode is None:
            # default: async path when async is enabled, rolling path otherwise
            self.rolling_mode = not self.async_compute
        else:
            self.rolling_mode = bool(rolling_mode)
        if self.rolling_mode and self.async_compute:
            # rolling mode computes in the update loop and does not use async jobs
            self.async_compute = False
        self._roll_states = np.empty((0, 5), dtype=np.float64) if np is not None else []
        self._executor = None
        self._pending_future = None
        self._pending_futures = []
        self._pending_job_id = 0
        self._next_job_id = 1
        self._last_swapped_job_id = 0
        self._jobs_submitted = 0
        self._jobs_swapped = 0
        self._single_flight = True

        self._computed_since_last_update = 0
        
        # debug counters / thresholds
        self._frame_dbg_counter = 0
        self._frame_dbg_freq = 10  # print PRED_DBG_FRAME every N frames (or when view changed)
        self._update_rolling_warn_threshold = 0.01  # only log UPDATE_ROLLING if > threshold (s)

        self._last_swapped_snapshot = None
        self._integrator_debug_seen = set()

        self.snapshot_velocity_rel_tol = self.velocity_invalidation_rel_tol
        self.snapshot_velocity_abs_tol = self.velocity_invalidation_abs_tol

        self.snapshot_position_abs_tol = 1000.0
        self.snapshot_sim_time_abs_tol = self.max_async_sim_age

        self.force_sync_on_stale = False


        self.view_change_cooldown = 0.0
        self._view_change_cooldown_until = 0.0

        self.snapshot_view_rel_tol = 1e-6

        self._view_scale_changed = False

        # optionale übersetzung des referenzrahmens. wenn gesetzt, berechnet predictor
        # bewegung in einem körper-zentrierten nicht-rotierenden rahmen durch subtraktion
        # der referenzkörper-beschleunigung.
        self.reference_body_index = None
        self._rolling_rkn_warning_printed = False

        # apoapsis/periapsis-marker entlang der prädiktionslinie (relativ zum
        # referenzkörper). lazy berechnet in get_apsis_markers() und über die
        # punkte-identität gecacht, damit pro trajektorie nur ein O(n)-scan läuft.
        self.apsis_markers_enabled = True
        self.apsis_max_markers = 16
        self._apsis_markers = self._empty_apsis_array()
        self._apsis_cache_key = None
        # Zeitraffer-halt: die gehaltene kurve aendert sich pro frame nur am
        # kopf (verbraucht) und schwanz (angestueckelt) -- die marker der
        # verbleibenden punkte sind bit-identisch. Statt jeden frame alle
        # 10 000 punkte neu zu scannen (2x pro frame: HUD + renderer),
        # werden die marker hoechstens alle `apsis_hold_rescan_s` neu
        # gerechnet und dazwischen nur um abgelaufene gefiltert. Ein neuer
        # marker am ENDE des horizonts erscheint damit maximal diese spanne
        # spaeter -- am fernen ende einer tagelangen vorhersage unsichtbar.
        self._apsis_soft_stale = False
        self._apsis_last_scan_ts = 0.0
        self.apsis_hold_rescan_s = 0.25
        # Generation der punkteliste: steigt bei jeder NEUEN kurve (swap,
        # neuberechnung, reset), nicht beim verbrauchen/anstueckeln im halt.
        self._points_generation = 0
        self._apsis_scan_generation = -1

        # Zeitraffer-halt (siehe _hold_advance). Standardmaessig AUS -- die
        # hauptschleife schaltet ihn ein, sobald der zeitraffer ueber die
        # unterste stufe geht.
        self.hold_enabled = False
        self._hold_invalidated = False
        # WEICHE entwertung: die gehaltene kurve ist nicht falsch, sondern nur
        # ueberholt (horizont/punktabstand verstellt). Sie darf weiterlaufen,
        # waehrend die neue im hintergrund entsteht -- siehe _hold_advance.
        self._hold_soft_invalidated = False
        # Laeuft gerade so ein hintergrund-auftrag fuer den halt?
        self._hold_pending_swap = False
        # Fortsetzungs-zustand der GEHALTENEN kurve, solange der wechsel
        # unterwegs ist (siehe _request_hold_recompute).
        self._hold_resume_context = None
        # Ob points[0] der selbst vorangestellte kopf ist (siehe
        # _advance_points_along_curve) -- er muss vor der naechsten suche
        # wieder weg. Gilt fuer BEIDE wege: den zeitraffer-halt und die
        # echtzeit, die dieselbe mechanik benutzt.
        self._synthetic_head = False
        # UM WIEVIEL WURDE DIE ZEITSPALTE GEGEN IHREN SCHNAPPSCHUSS VERSCHOBEN?
        #
        # `points[:, 2]` ist absolute sim-zeit, gerechnet als
        # `snapshot["sim_time"] + lokale zeit`. Wer die kurve starr nachzieht
        # (der fallback in _anchor_first_point), verschiebt auch diese spalte
        # -- der schnappschuss aber bleibt, wo er ist. Jede auswertung, die
        # aus einer punktzeit eine LOKALE zeit zurueckrechnet (der
        # apsis-scan propagiert damit den referenzkoerper), braucht deshalb
        # diesen versatz, sonst liest sie die koerper um genau diesen betrag
        # zu weit vorn. Siehe get_apsis_markers().
        self._points_time_offset = 0.0
        # Ab welchem restvorrat (anteil des punktbudgets) waehrend des halts
        # nachgerechnet wird. Das ist die failsafe-schwelle, die verhindert,
        # dass die linie ausläuft.
        self.hold_refresh_fraction = 0.25
        # Ueber wie viele punkte die kopf-korrektur abklingt.
        self.hold_taper_points = 64
        # Hoechstens so viele punkte je frame hinten anstueckeln (siehe
        # _hold_advance): verteilt einen budget-sprung ueber mehrere frames.
        self.hold_extend_max_points = 1000
        # WIE WEIT DARF DAS SCHIFF NEBEN DER GEHALTENEN KURVE LIEGEN?
        # In PIXELN, weil genau das die groesse ist, die man sieht -- und
        # weil eine weltlaenge auf jeder zoomstufe etwas anderes bedeutet.
        # Siehe _hold_advance.
        self.hold_drift_max_px = 0.5
        # Untergrenze in metern, damit ein extremer zoom nicht in jedem frame
        # nachrechnen laesst (dann waere der halt wirkungslos).
        self.hold_drift_min_m = 1.0
        # Letzter gemessener seitlicher versatz (diagnose/tests).
        self.hold_drift_m = 0.0
        # GERECHNETE laenge und GEZEICHNETE laenge sind nicht dasselbe.
        # Im zeitraffer muss die kurve viel weiter reichen, als man sieht --
        # sonst laeuft der halt leer (bei 1 y/s frisst EIN frame den ganzen
        # basis-horizont). Sichtbar soll die linie aber ueberall gleich lang
        # sein, sonst wickelt sie sich im zeitraffer mehrfach um die bahn,
        # waehrend sie in echtzeit einen einzigen bogen zeigt.
        # None = alles zeichnen. Siehe set_display_length().
        self.display_length = None
        # Die GEZEICHNETE punktzahl wird auf ein vielfaches hiervon gerundet,
        # damit ein langsamer regler-zug nicht in JEDEM frame eine neue view
        # (self.points[:count]) erzeugt -- der renderer-cache und
        # get_apsis_markers() haengen an id(points). Aus config gesetzt
        # (predictor.display_length_quantum_points), getattr-default weil eine
        # vor __init__ gebaute (oder entpickelte) instanz sie nicht traegt.
        self._display_quantum = 8
        self._display_view = None
        self._display_view_base = None
        self._display_view_limit = -1

        if self.async_compute and not self.rolling_mode:
            self._ensure_executor()
            self._pending_futures = []

    @staticmethod
    def _normalize_integrator_mode(mode):
        try:
            mode = str(mode).strip().lower()
        except Exception:
            mode = "rkn"
        if mode in ("rkn", "rkn_adaptive", "rkn_adaptive_sd"):
            return "rkn"
        if mode not in ("rk4", "aspi", "aspi_rk4_fallback"):
            return "rkn"
        return mode

    def _ensure_executor(self):
        if getattr(self, "_executor", None) is not None:
            return
        # Beim GLEITFLUG laeuft hier genau eine rechnung; die tiefe wird nur
        # unter schub ausgereizt (siehe _request_thrust_recompute).
        #
        # Warum ueberhaupt mehrere: eine vorhersage dauert ~17 ms, ein bild
        # ~7 ms. Nacheinander gerechnet kann die linie also hoechstens jedes
        # dritte bild neu sein -- das ist das ruckeln waehrend eines burns.
        # Die dauer EINER rechnung laesst sich nicht weiter druecken, ihr
        # DURCHSATZ aber schon: mehrere zeitversetzt gestartete laeufe geben
        # alle ~17/tiefe ms ein ergebnis. Erlaubt ist das, weil alle kernel
        # `nogil=True` sind -- sie laufen wirklich nebenlaeufig und nehmen dem
        # hauptthread nichts weg (gemessen: gleiche hauptthread-arbeit 0.25 ms
        # bei leerlaufendem gegen 0.27 ms bei ausgelastetem worker).
        # Der pool wird auf die OBERGRENZE ausgelegt; wie viele davon
        # tatsaechlich beschaeftigt sind, entscheidet _target_pipeline_depth
        # bild fuer bild aus rechenzeit/bildzeit. Leerlaufende threads kosten
        # nichts.
        workers = self._pipeline_depth_cap()
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="predictor-worker")
        self._predictor_worker_threads = workers
        if self.debug:
            try:
                print(f"PRED_DBG_THREAD: predictor worker max_workers={workers}", flush=True)
            except Exception:
                pass

    def _debug_integrator_mode(self, action, snapshot):
        if not self.debug:
            return
        try:
            mode = self._normalize_integrator_mode(snapshot.get("integrator_mode", self.integrator_mode))
            fallback = bool(snapshot.get("aspi_use_rk4_fallback", self.aspi_use_rk4_fallback))
            key = (str(action), mode, fallback)
            seen = getattr(self, "_integrator_debug_seen", set())
            if key in seen:
                return
            seen.add(key)
            self._integrator_debug_seen = seen
            if mode == "aspi" or mode == "aspi_rk4_fallback":
                print(f"PRED_DBG_INTEGRATOR: {action} mode={mode} aspi_rk4_fallback={fallback}", flush=True)
            else:
                print(f"PRED_DBG_INTEGRATOR: {action} mode={mode}", flush=True)
        except Exception:
            pass

    def set_integrator_quality(self, quality: str):
        old = (
            self.integrator_mode,
            self.aspi_min_dt,
            self.aspi_max_dt,
            self.aspi_safety_g,
            self.aspi_safety_m,
            self.aspi_close_acc_threshold,
            self.aspi_use_rk4_fallback,
            self.rkn_min_dt,
            self.rkn_max_dt,
            self.rkn_rtol,
            self.rkn_atol_pos,
            self.rkn_atol_vel,
            self.rkn_safety,
            self.rkn_min_factor,
            self.rkn_max_factor,
            self.rkn_max_rejects,
        )

        q = str(quality).strip().lower()
        if q == "fast":
            self.integrator_mode = "rkn"
            self.rkn_min_dt = 0.5
            self.rkn_max_dt = 3000.0
            self.rkn_rtol = 1e-5
            self.rkn_atol_pos = 1000.0
            self.rkn_atol_vel = 1e-2
        elif q == "balanced":
            self.integrator_mode = "rkn"
            self.rkn_min_dt = 0.1
            self.rkn_max_dt = 1500.0
            self.rkn_rtol = 1e-7
            self.rkn_atol_pos = 10.0
            self.rkn_atol_vel = 1e-4
        elif q == "accurate":
            self.integrator_mode = "rkn"
            self.rkn_min_dt = 0.01
            self.rkn_max_dt = 500.0
            self.rkn_rtol = 1e-9
            self.rkn_atol_pos = 0.1
            self.rkn_atol_vel = 1e-6
        elif q == "rk4":
            self.integrator_mode = "rk4"
        else:
            raise ValueError("quality must be one of: fast, balanced, accurate, rk4")

        new = (
            self.integrator_mode,
            self.aspi_min_dt,
            self.aspi_max_dt,
            self.aspi_safety_g,
            self.aspi_safety_m,
            self.aspi_close_acc_threshold,
            self.aspi_use_rk4_fallback,
            self.rkn_min_dt,
            self.rkn_max_dt,
            self.rkn_rtol,
            self.rkn_atol_pos,
            self.rkn_atol_vel,
            self.rkn_safety,
            self.rkn_min_factor,
            self.rkn_max_factor,
            self.rkn_max_rejects,
        )
        if new != old:
            self.reset()

    @staticmethod
    def _rkn_failure_reason(code):
        try:
            code = int(code)
        except Exception:
            code = 0
        if code == 0:
            return ""
        if code == 1:
            return "non-finite input state"
        if code == 2:
            return "adaptive step rejected too often"
        if code == 3:
            return "non-finite adaptive step"
        if code == 4:
            return "maximum predictor iterations reached"
        if code == 6:
            return "minimum dt could not satisfy tolerance"
        return f"failure code {code}"

    def _apply_rkn_stats(self, stats):
        if stats is None:
            return
        try:
            self.rkn_last_accepted_steps = int(stats[0])
            self.rkn_last_rejected_steps = int(stats[1])
            self.rkn_last_min_dt = float(stats[2])
            self.rkn_last_max_dt = float(stats[3])
            self.rkn_last_max_error_norm = float(stats[4])
            failure_code = int(stats[5])
            self.rkn_last_failed = failure_code != 0
            self.rkn_last_failure_reason = self._rkn_failure_reason(failure_code)
            if self.debug and not getattr(self, "_suppress_dbg_computed", False):
                print(
                    "PRED_DBG_RKN: "
                    f"accepted={self.rkn_last_accepted_steps} "
                    f"rejected={self.rkn_last_rejected_steps} "
                    f"min_dt={self.rkn_last_min_dt:.6g} "
                    f"max_dt={self.rkn_last_max_dt:.6g} "
                    f"max_err={self.rkn_last_max_error_norm:.6g} "
                    f"failed={self.rkn_last_failed}",
                    flush=True,
                )
        except Exception:
            pass

    def reset(self):
        self._cancel_pending_job()
        self.points = _empty_points()
        self._roll_states = np.empty((0, 5), dtype=np.float64) if np is not None else []
        self.initialized = False
        self._clear_apsis_markers()
        # Der halt haelt eine kurve fest, die es nach dem reset nicht mehr gibt.
        self._synthetic_head = False
        # Ohne punkte gibt es auch keinen versatz ihrer zeitspalte.
        self._points_time_offset = 0.0
        # Eine WEICHE entwertung setzt darauf, dass die alte kurve noch da ist
        # -- nach dem reset ist sie es nicht. Sonst wuerde der halt beim
        # naechsten frame auf einer leeren kurve weiterhalten wollen.
        self._hold_soft_invalidated = False
        self._hold_pending_swap = False
        self._hold_resume_context = None
        self._resume_context = None
        # WICHTIG: auch den vermerk loeschen, WELCHER zustand die punkte erzeugt
        # hat. Er wird nur beim einwechseln eines ergebnisses gesetzt, nach dem
        # reset gibt es aber keins mehr -- er stuende also als luege da.
        #
        # Das ist kein aufraeumen, sondern behebt eine selbsterhaltende
        # blockade: update() vergleicht die schiffsgeschwindigkeit gegen genau
        # diesen vermerk und wirft die bahn weg, sobald sie abweicht. Bleibt er
        # alt stehen, weicht sie JEDEN frame weiter ab (im zeitraffer um
        # ~24 m/s je frame), also wird jeden frame die trajektorien-version
        # erhoeht und der laufende hintergrund-auftrag verworfen -- der aber
        # laenger als einen frame braucht. Gemessen nach einem druck auf
        # '9'/'0'/'+'/'-': 20 frames, 20 auftraege abgeschickt, KEINER
        # eingewechselt, die linie kam nie zurueck. Ohne linie faellt der
        # navball auf die geradeaus-tangente zurueck statt auf die gezeichnete
        # bahn -- das ist das springen der marker.
        self._last_swapped_snapshot = None
        # Die gemessene bahn-zeitspanne gehoert zu der kurve, die es nicht mehr
        # gibt. Nach einem reparenting/teleport waere sie schlicht falsch.
        self._horizon_time_per_arc = 0.0

    def set_reference_body_index(self, index: int | None):
        if index is None:
            new_index = None
        else:
            new_index = int(index)

        if new_index == self.reference_body_index:
            return

        self.reference_body_index = new_index
            # Frame-Änderung macht aktuell gespeicherte Prädiktor-Punkte ungültig.
        self.reset()

    def _resolve_reference_body(self, world):
        idx = self.reference_body_index
        if idx is None:
            return 0, 0.0, 0.0

        try:
            idx = int(idx)
        except Exception:
            return 0, 0.0, 0.0

        try:
            if idx < 0 or idx >= len(world.body):
                return 0, 0.0, 0.0
            ref = world.body[idx]
            return 1, float(ref.position.x), float(ref.position.y)
        except Exception:
            return 0, 0.0, 0.0

    def _snapshot_body_index_by_names(self, snapshot, names):
        body_names = snapshot.get("body_names", None) if snapshot is not None else None
        if body_names is None:
            return -1
        wanted = set(names)
        for i, name in enumerate(body_names):
            key = str(name).strip().lower()
            if key in wanted:
                return int(i)
        return -1

    def _snapshot_body_position_at_local_t(self, snapshot, index, local_t):
        if snapshot is None or index < 0:
            return None
        try:
            if bool(snapshot.get("use_time_dependent_bodies", True)):
                return _body_position_at_time_numba(
                    int(index),
                    float(local_t),
                    snapshot["body_x"],
                    snapshot["body_y"],
                    snapshot["body_m"],
                    snapshot["body_scripted"],
                    snapshot["body_a"],
                    snapshot["body_e"],
                    snapshot["body_theta"],
                    snapshot["body_arg"],
                    snapshot["body_parent"],
                    float(snapshot["G"]),
                    _no_body_memo(),
                )
            return float(snapshot["body_x"][index]), float(snapshot["body_y"][index])
        except Exception:
            return None

    def _snapshot_body_velocity_at_local_t(self, snapshot, index, local_t):
        dt = 1.0
        p0 = self._snapshot_body_position_at_local_t(snapshot, index, float(local_t) - dt)
        p1 = self._snapshot_body_position_at_local_t(snapshot, index, float(local_t) + dt)
        if p0 is None or p1 is None:
            return None
        return (float(p1[0]) - float(p0[0])) / (2.0 * dt), (float(p1[1]) - float(p0[1])) / (2.0 * dt)

    def _debug_moving_source_snapshot(self, snapshot):
        if not getattr(self, "debug_moving_sources", False):
            return
        try:
            labels = [
                ("Earth", ("earth", "erde")),
                ("Mun", ("mun", "moon", "mond")),
            ]
            parts = []
            for label, names in labels:
                idx = self._snapshot_body_index_by_names(snapshot, names)
                if idx < 0:
                    continue
                samples = []
                for t in (0.0, 3600.0, 7200.0):
                    pos = self._snapshot_body_position_at_local_t(snapshot, idx, t)
                    if pos is not None:
                        samples.append(f"t=+{t:.0f} {label}=({pos[0]:.6e},{pos[1]:.6e})")
                if samples:
                    parts.append(" ".join(samples))
            if parts:
                print("PRED_SOURCE_DBG: " + " | ".join(parts), flush=True)
        except Exception:
            pass

    def _debug_predictor_energy(self, snapshot, points):
        if not getattr(self, "debug_moving_sources", False):
            return
        try:
            if points is None:
                return
            if np is not None and isinstance(points, np.ndarray):
                n = int(points.shape[0])
                if n < 3:
                    return
                get_point = lambda i: (float(points[i, 0]), float(points[i, 1]), float(points[i, 2]))
            else:
                n = len(points)
                if n < 3:
                    return
                get_point = lambda i: (float(points[i][0]), float(points[i][1]), float(points[i][2]))

            earth_idx = self._snapshot_body_index_by_names(snapshot, ("earth", "erde"))
            if earth_idx < 0:
                return

            base_t = float(snapshot.get("sim_time", 0.0))
            earth_mass = float(snapshot["body_m"][earth_idx])
            G = float(snapshot["G"])
            indices = [0, n // 2, n - 1]
            parts = []
            for idx in indices:
                px, py, abs_t = get_point(idx)
                local_t = abs_t - base_t
                earth_pos = self._snapshot_body_position_at_local_t(snapshot, earth_idx, local_t)
                earth_vel = self._snapshot_body_velocity_at_local_t(snapshot, earth_idx, local_t)
                if earth_pos is None or earth_vel is None:
                    continue

                if idx <= 0:
                    px2, py2, t2 = get_point(1)
                    dt = max(1e-9, t2 - abs_t)
                    ship_vx = (px2 - px) / dt
                    ship_vy = (py2 - py) / dt
                elif idx >= n - 1:
                    px0, py0, t0 = get_point(n - 2)
                    dt = max(1e-9, abs_t - t0)
                    ship_vx = (px - px0) / dt
                    ship_vy = (py - py0) / dt
                else:
                    px0, py0, t0 = get_point(idx - 1)
                    px2, py2, t2 = get_point(idx + 1)
                    dt = max(1e-9, t2 - t0)
                    ship_vx = (px2 - px0) / dt
                    ship_vy = (py2 - py0) / dt

                rel_x = px - float(earth_pos[0])
                rel_y = py - float(earth_pos[1])
                rel_vx = ship_vx - float(earth_vel[0])
                rel_vy = ship_vy - float(earth_vel[1])
                r = math.hypot(rel_x, rel_y)
                if r <= 1e-9:
                    continue
                energy = 0.5 * (rel_vx * rel_vx + rel_vy * rel_vy) - G * earth_mass / r
                parts.append(f"i={idx} t=+{local_t:.3f}s E={energy:.6e}")
            if parts:
                print("PRED_ENERGY_DBG: " + " | ".join(parts), flush=True)
        except Exception:
            pass

    def _points_count(self):
        if np is not None and isinstance(self.points, np.ndarray):
            return int(self.points.shape[0])
        return len(self.points)

    def _current_reference_body_index(self):
        try:
            if self.reference_body_index is None:
                return -1
            return int(self.reference_body_index)
        except Exception:
            return -1

    def _empty_points_array(self) -> "np.ndarray | list":
        return _empty_points()

    def _empty_apsis_array(self):
        return np.empty((0, 5), dtype=np.float64) if np is not None else []

    def _clear_apsis_markers(self):
        self._apsis_markers = self._empty_apsis_array()
        self._apsis_cache_key = None

    def _clear_prediction_points(self):
        self.points = self._empty_points_array()
        self._roll_states = np.empty((0, 5), dtype=np.float64) if np is not None else []
        self.initialized = False
        self._points_time_offset = 0.0
        self._synthetic_head = False
        self._clear_apsis_markers()

    def _allowed_velocity_delta(self, speed):
        try:
            speed = float(speed)
        except Exception:
            speed = 0.0
        return max(
            float(self.velocity_invalidation_abs_tol),
            float(self.velocity_invalidation_rel_tol) * max(abs(speed), 1.0),
        )

    def _remember_ship_state(self, ship, world=None):
        if ship is None:
            return
        try:
            self._last_seen_px = float(ship.position.x)
            self._last_seen_py = float(ship.position.y)
            self._last_seen_vx = float(ship.velocity.x)
            self._last_seen_vy = float(ship.velocity.y)
        except Exception:
            return
        try:
            self._last_seen_sim_time = float(world.time) if world is not None else None
        except Exception:
            self._last_seen_sim_time = None

    def _warn_rolling_rkn_once(self):
        if not self.debug:
            return
        if not self.rolling_mode or self.integrator_mode != "rkn":
            return
        if getattr(self, "_rolling_rkn_warning_printed", False):
            return
        self._rolling_rkn_warning_printed = True
        try:
            print("PRED_DBG_WARNING: rolling_mode uses RK4 state helper, not adaptive RKN", flush=True)
        except Exception:
            pass

    def _handle_trajectory_branch_change(self, ship, world):
        if ship is None:
            return False

        try:
            cur_px = float(ship.position.x)
            cur_py = float(ship.position.y)
            cur_vx = float(ship.velocity.x)
            cur_vy = float(ship.velocity.y)
        except Exception:
            return False

        last_px = self._last_seen_px
        last_py = self._last_seen_py
        last_vx = self._last_seen_vx
        last_vy = self._last_seen_vy
        if last_px is None or last_py is None or last_vx is None or last_vy is None:
            self._remember_ship_state(ship, world)
            return False

        dvx_seen = cur_vx - float(last_vx)
        dvy_seen = cur_vy - float(last_vy)
        delta_speed = math.hypot(dvx_seen, dvy_seen)
        cur_speed = math.hypot(cur_vx, cur_vy)
        allowed_speed = self._allowed_velocity_delta(cur_speed)

        delta_pos = math.hypot(cur_px - float(last_px), cur_py - float(last_py))
        try:
            cur_time = float(world.time) if world is not None else None
        except Exception:
            cur_time = None
        last_time = self._last_seen_sim_time
        if cur_time is not None and last_time is not None:
            dt_age = abs(cur_time - float(last_time))
        else:
            dt_age = abs(float(self.dt))
        last_speed = math.hypot(float(last_vx), float(last_vy))
        expected_motion = max(cur_speed, last_speed, 1.0) * max(dt_age, 0.0)
        allowed_pos = max(float(self.position_invalidation_abs_tol), expected_motion * 4.0)

        # Die schwerkraft wird HERAUSGERECHNET, nicht mit einer schranke
        # ueberdeckt.
        #
        # Frueher stand hier `allowed_speed = max(allowed, 4 * |g| * dt)`: der
        # gesamte geschwindigkeitssprung wurde gegen eine schranke von der
        # groesse des schwerkraft-anteils gehalten. Fern vom planeten geht das
        # auf, NAHE DER PERIAPSIS nicht: dort ist |g| = 8.1 m/s^2, ueber einen
        # 2-sekunden-schritt also 16 m/s schwerkraft gegen 6.7 m/s vollschub
        # je bild -- die schranke lag bei 65 m/s und der schub verschwand
        # vollstaendig darunter. Die vorhersagelinie wurde in genau dem
        # moment nicht mehr angefordert, in dem sie sich am staerksten
        # aendert, und sprang erst wieder an, wenn das schiff weit genug weg
        # war. Das ist das ruckartige nachziehen nahe der periapsis.
        #
        # Richtig ist der REST: was bleibt von der geschwindigkeitsaenderung
        # uebrig, wenn man abzieht, was die schwerkraft erklaert. Gemessen auf
        # einer bahn mit e = 0.7 um die Erde, je bild:
        #
        #     periapsis   gleitflug 0.023 m/s   |   schub 6.69 m/s
        #     apoapsis    gleitflug 0.000 m/s   |   schub 6.67 m/s
        #
        # Der schub steht damit ueberall gleich deutlich da (faktor ~290 ueber
        # dem grundrauschen), und die feste toleranz von 1 m/s trennt beides
        # sauber. Nebenbei faengt der test auch den fall, in dem schub der
        # schwerkraft ENTGEGEN zeigt und die summe klein ist: bei nu = 90 Grad
        # betraegt der gesamtsprung 0.98 m/s -- unter der toleranz -- der rest
        # aber 6.66 m/s.
        #
        # Die restschranke muss mit der KRUEMMUNG von g mitwachsen, sonst
        # feuert sie im zeitraffer: `g * dt` erklaert einen 28-stunden-schritt
        # nicht mehr. `|g_jetzt - g_vorher| * dt` waechst genau mit diesem
        # fehler mit -- gemessen im gleitflug von 0.5 s bis 100800 s (7 d/s)
        # bleibt der rest bei jedem schritt unter der schranke.
        residual_speed = delta_speed
        if world is not None:
            try:
                g = world.acceleration_at(ship, ship.position, cur_time)
                gx = float(g.x)
                gy = float(g.y)
                span = max(dt_age, 0.0)
                residual_speed = math.hypot(dvx_seen - gx * span, dvy_seen - gy * span)

                last_gx = self._last_seen_gx
                last_gy = self._last_seen_gy
                # DIE KRUEMMUNGS-SCHRANKE GILT NUR, SOLANGE DER SCHRITT DIE
                # BAHN UEBERHAUPT AUFLOEST.
                #
                # `|g_jetzt - g_vorher|` misst, wie stark sich die schwerkraft
                # ueber den schritt geaendert hat -- ein gutes mass fuer den
                # fehler von `g * dt`, solange sich g dazwischen stetig
                # bewegt. Deckt ein schritt aber MEHRERE UMLAEUFE ab (7 d/s
                # sind 28 stunden je bild, auf einer 2-stunden-bahn), dann
                # sind anfangs- und endwert unkorreliert: sie koennen zufaellig
                # dicht beieinander liegen, die schranke faellt zusammen und
                # der ganz normale gleitflug reisst sie. Gemessen auf der
                # e = 0.7-bahn bei 100800 s je schritt: rest 2.46e3 m/s gegen
                # eine schranke von 1.20e3 m/s -- ein bild von sechs, ohne
                # jeden schub.
                #
                # Oberhalb der bahn-zeitskala (`sqrt(r/|g|)`, auf der
                # kreisbahn T/2pi -- dieselbe groesse, die den zeitraffer
                # deckelt) traegt der vergleich also nichts mehr, und es
                # bleibt die alte, grosszuegige schranke. Das ist die
                # richtige seite des irrtums: schub gibt es dort ohnehin
                # nicht (`test.py` sperrt ihn oberhalb von
                # `realtime_warp_max`), eine verpasste anforderung kostet
                # nichts -- eine falsche zerreisst die gehaltene kurve.
                resolves_orbit = True
                try:
                    t_char = self._characteristic_timescale(world, ship)
                    resolves_orbit = t_char is None or span <= t_char
                except Exception:
                    resolves_orbit = True
                if last_gx is None or last_gy is None or not resolves_orbit:
                    curvature_dv = math.hypot(gx, gy) * span
                else:
                    curvature_dv = math.hypot(gx - float(last_gx), gy - float(last_gy)) * span
                allowed_speed = max(allowed_speed,
                                    float(self.gravity_dv_safety_factor) * curvature_dv)
                self._last_seen_gx = gx
                self._last_seen_gy = gy
            except Exception:
                residual_speed = delta_speed

        reason = None
        if residual_speed > allowed_speed:
            reason = "velocity"
        elif delta_pos > allowed_pos:
            reason = "position"

        if reason is None:
            self._remember_ship_state(ship, world)
            return False

        # Schub ist KEIN bruch der bahn, sondern ihre stetige veraenderung: die
        # gezeichnete linie ist danach ein paar dutzend millisekunden alt, aber
        # nicht falsch. Sie deshalb zu leeren und synchron neu zu rechnen kostet
        # 59 ms pro frame (voller sonnensystem-satz) und verwarf zugleich jedes
        # asynchrone ergebnis, weil die version im naechsten frame schon wieder
        # weiter war. Ein echter POSITIONS-sprung (teleport, reparenting) ist
        # dagegen ein bruch -- dort bleibt der harte weg unten.
        if reason == "velocity" and self._request_thrust_recompute(ship, world):
            self._remember_ship_state(ship, world)
            if self.debug:
                try:
                    print(
                        "PRED_DBG_TRAJECTORY_REFRESH: "
                        f"reason=velocity rest={residual_speed:.6e} (roh {delta_speed:.6e}) "
                        f"allowed={allowed_speed:.6e} "
                        "mode=async-coalesced",
                        flush=True,
                    )
                except Exception:
                    pass
            # Nicht kurzschliessen: update() soll normal weiterlaufen, damit
            # ein fertiges ergebnis eingewechselt und die linie ans schiff
            # geheftet wird.
            return False

        old_version = int(self._trajectory_version)
        self._trajectory_version = old_version + 1
        if self.debug:
            try:
                if reason == "velocity":
                    print(
                        "PRED_DBG_TRAJECTORY_INVALIDATED: "
                        f"reason=velocity rest={residual_speed:.6e} allowed={allowed_speed:.6e} "
                        f"old_version={old_version} new_version={self._trajectory_version}",
                        flush=True,
                    )
                else:
                    print(
                        "PRED_DBG_TRAJECTORY_INVALIDATED: "
                        f"reason=position dp={delta_pos:.6e} allowed={allowed_pos:.6e} "
                        f"old_version={old_version} new_version={self._trajectory_version}",
                        flush=True,
                    )
            except Exception:
                pass

        self._cancel_pending_job()
        self._clear_prediction_points()
        self._remember_ship_state(ship, world)

        if self.sync_recompute_on_velocity_change and world is not None:
            self._compute_full(ship, world)
        elif self.async_compute and world is not None and self.num_points > 0:
            self._submit_async_compute(ship, world, self._get_target_point_cap())

        return True

    def _rebase_points_to_current_snapshot(self, points, snapshot, current_ship):
        if points is None or snapshot is None or current_ship is None:
            return points
        try:
            dx = float(current_ship.position.x) - float(snapshot.get("ship_px", 0.0))
            dy = float(current_ship.position.y) - float(snapshot.get("ship_py", 0.0))
        except Exception:
            return points

        if not math.isfinite(dx) or not math.isfinite(dy):
            return points

        if np is not None and isinstance(points, np.ndarray):
            rebased = points.copy()
            if rebased.shape[0] <= 0 or rebased.shape[1] < 2:
                return rebased
            rebased[:, 0] += dx
            rebased[:, 1] += dy
            rebased[0, 0] = float(current_ship.position.x)
            rebased[0, 1] = float(current_ship.position.y)
            return rebased

        try:
            rebased = []
            for idx, p in enumerate(points):
                if idx == 0:
                    x = float(current_ship.position.x)
                    y = float(current_ship.position.y)
                else:
                    x = float(p[0]) + dx
                    y = float(p[1]) + dy
                if hasattr(p, "__len__") and len(p) >= 3:
                    rebased.append((x, y, float(p[2])))
                else:
                    rebased.append((x, y))
            return rebased
        except Exception:
            return points

    def _log_snapshot_result(self, accepted, reason, snapshot, cur_sim_time, sim_age, pos_delta, vel_delta):
        if not self.debug:
            return
        try:
            snap_sim_time = float(snapshot.get("sim_time", 0.0)) if snapshot is not None else 0.0
        except Exception:
            snap_sim_time = 0.0
        try:
            cur_time = float(cur_sim_time) if cur_sim_time is not None else float("nan")
        except Exception:
            cur_time = float("nan")
        try:
            age = float(sim_age) if sim_age is not None else float("nan")
        except Exception:
            age = float("nan")
        try:
            pd = float(pos_delta)
        except Exception:
            pd = float("nan")
        try:
            vd = float(vel_delta)
        except Exception:
            vd = float("nan")
        try:
            snapshot_version = int(snapshot.get("trajectory_version", -1)) if snapshot is not None else -1
        except Exception:
            snapshot_version = -1
        try:
            current_version = int(self._trajectory_version)
        except Exception:
            current_version = -1

        if accepted:
            print(
                "PRED_DBG_ACCEPT_SNAPSHOT: "
                f"reason={reason} "
                f"version={current_version} "
                f"sim_age={age:.6e} "
                f"pos_delta={pd:.6e} "
                f"vel_delta={vd:.6e} "
                f"snapshot_sim_time={snap_sim_time:.6f} "
                f"current_world_time={cur_time:.6f}",
                flush=True,
            )
        else:
            print(
                "PRED_DBG_REJECT_SNAPSHOT: "
                f"reason={reason} "
                f"snapshot_version={snapshot_version} "
                f"current_version={current_version} "
                f"sim_age={age:.6e} "
                f"pos_delta={pd:.6e} "
                f"vel_delta={vd:.6e} "
                f"snapshot_sim_time={snap_sim_time:.6f} "
                f"current_world_time={cur_time:.6f}",
                flush=True,
            )

    def _advance_points_along_curve(self, ship, now):
        """Kurve VERBRAUCHEN statt starr verschieben.

        Rueckgabe: die zahl der vorn verbrauchten stuetzstellen, oder None,
        wenn es nicht geht (keine/zu kurze kurve, zeit abgelaufen) -- dann
        muss der aufrufer den alten weg gehen.

        Die vorhersage ist eine eigenschaft der BAHN, nicht des augenblicks.
        Ohne schub bleibt sie stehen und das schiff rutscht an ihr entlang.
        Also werden vorn die punkte weggeworfen, deren zeit bereits vergangen
        ist (die zeitspalte ist absolute sim-zeit, das ist exakt und per
        suchlauf billig), und der rest bleibt in ORT UND ZEIT stehen, wo er
        ist.

        DIE KURVE WIRD VORN ANGESTUECKELT, NICHT VERBOGEN.

        Stuetzstellen lassen sich nur GANZ wegwerfen -- eine halbe gibt es
        nicht. Bliebe als kopf immer die naechste stuetzstelle VOR dem schiff
        stehen, liefe der rest zwischen zwei verbrauchten stuetzstellen von 0
        auf eine volle punktweite und spraenge dann zurueck: ein saegezahn mit
        der amplitude EINER PUNKTWEITE. Weil das eine weltlaenge ist und der
        zoom welt und linie gleich vergroessert, saehe es auf JEDER zoomstufe
        gleich aus -- die linie rueckte sichtbar in stufen statt stetig vor.

        Richtig ist, dem unveraenderten rest die aktuelle schiffsposition als
        neuen kopf voranzustellen. Das erste segment ist dann ein echtes
        teilstueck, das stetig kuerzer wird, bis die naechste stuetzstelle
        verbraucht ist. Kein punkt hinter dem kopf bewegt sich dabei
        ueberhaupt -- und genau darauf beruht, dass die Ap/Pe-marker
        stillstehen.
        """
        points = self.points
        if np is None or not isinstance(points, np.ndarray) or points.ndim != 2:
            return None
        if points.shape[0] < 4 or points.shape[1] < 3:
            return None
        if ship is None or not math.isfinite(now):
            return None

        # Den selbst vorangestellten kopf aus dem vorframe wieder entfernen,
        # damit unten immer auf den UNVERAENDERTEN stuetzstellen gesucht wird
        # (und die liste nicht bei jedem frame um einen punkt waechst).
        had_head = bool(getattr(self, '_synthetic_head', False)) and points.shape[0] >= 3
        if had_head:
            points = points[1:]

        times = points[:, 2]
        if not (math.isfinite(float(times[0])) and math.isfinite(float(times[-1]))):
            return None
        # Reicht die kurve zeitlich ueberhaupt noch in die zukunft?
        if float(times[-1]) <= now:
            return None

        # Erster punkt, der ECHT in der zukunft liegt. Die zeitspalte ist
        # monoton steigend, also genuegt eine binaere suche.
        #
        # 'right', nicht 'left': eine stuetzstelle GENAU auf `now` ist die
        # gegenwart, und die gegenwart ist der kopf, den wir gleich davor
        # setzen. Bliebe sie stehen, saessen zwei punkte aufeinander und das
        # erste segment haette laenge null -- mitsamt seiner tangente, an der
        # der navball haengt. Exakte gleichheit ist kein grenzfall, sondern
        # der regelfall: eine FRISCH gerechnete kurve beginnt per konstruktion
        # bei ship@world.time, und _anchor_first_point laeuft unmittelbar
        # danach.
        drop = int(np.searchsorted(times, now, side='right'))
        drop = max(0, min(drop, points.shape[0] - 2))

        if drop == 0 and had_head:
            # NICHTS VERBRAUCHT -> NICHTS UMKOPIEREN.
            #
            # Der regelfall in echtzeit: die stuetzstellen liegen auf festem
            # BOGENabstand (bei spielueblichem zoom hunderte kilometer),
            # waehrend ein bild nur bruchteile davon vorrueckt -- es wird also
            # ueber viele bilder hinweg gar keine stuetzstelle faellig. Dann
            # genuegt es, den vorhandenen kopf nachzufuehren.
            #
            # Das spart nicht bloss die kopie: es haelt auch die IDENTITAET
            # des arrays fest, und daran haengen zwei caches, die sonst in
            # jedem bild leerliefen -- der apsis-scan (id(pts), siehe
            # get_apsis_markers) und die abtastung der linie im renderer
            # (_make_prediction_line_cache_key). Die marker duerfen dabei
            # stehen bleiben, weil der scan den kopf ohnehin ueberspringt
            # (skip_head) und hinter ihm kein punkt bewegt wurde.
            self.points[0, 0] = float(ship.position.x)
            self.points[0, 1] = float(ship.position.y)
            self.points[0, 2] = now
            if self.points.shape[1] > 3:
                self.points[0, 3] = float(getattr(ship.velocity, 'x', 0.0))
                self.points[0, 4] = float(getattr(ship.velocity, 'y', 0.0))
            return 0

        tail = points[drop:] if drop > 0 else points
        head = np.empty((1, points.shape[1]), dtype=np.float64)
        head[0, 0] = float(ship.position.x)
        head[0, 1] = float(ship.position.y)
        head[0, 2] = now
        if points.shape[1] > 3:
            # Der kopf IST das schiff -- also auch seine tangente. Frueher
            # wurde die der naechsten stuetzstelle uebernommen; damit haette
            # das erste (stetig kuerzer werdende) teilstueck eine tangente
            # getragen, die zur falschen stelle der bahn gehoert.
            head[0, 3] = float(getattr(ship.velocity, 'x', 0.0))
            head[0, 4] = float(getattr(ship.velocity, 'y', 0.0))

        self.points = np.concatenate((head, tail), axis=0)
        self._synthetic_head = True
        # Die zeitspalte der verbliebenen punkte ist unangetastet -- ihr
        # versatz gegen den schnappschuss bleibt also, was er war.
        self._invalidate_derived_caches(soft=True)
        return drop

    def _anchor_first_point(self, ship, world):
        """Setzt den kurvenanfang auf das schiff.

        DER REGELFALL IST DAS VERBRAUCHEN, NICHT DAS VERSCHIEBEN. Erste wahl
        ist `_advance_points_along_curve` -- die kurve bleibt stehen und das
        schiff rutscht an ihr entlang. Die starre verschiebung unten ist nur
        noch der fallback fuer den rolling-modus und fuer eine kurve, deren
        zeit abgelaufen ist (dann steht ohnehin gleich eine neuberechnung an).

        WARUM NICHT MEHR STARR. Die verschiebung zieht die GANZE kurve um den
        kopfversatz mit, und der ist nicht der versatz je frame, sondern der
        ueber das ganze alter des schnappschusses -- `max_async_wall_age`
        laesst 1.5 s echtzeit zu, bei 60 s/s also bis zu 90 sim-sekunden
        bahnbewegung. Der referenzkoerper wandert dabei NICHT mit. Was bleibt,
        ist die RELATIVbewegung schiff<->referenzkoerper: die ganze kegel-
        schnittbahn liegt um diesen betrag seitlich neben dem koerper, und
        damit steht die periapsis-hoehe falsch. Weil das alter mit der
        rechenlatenz schwankt, schwankt der angezeigte Pe/Ap-abstand mit --
        das ist das hin- und herspringen der marker in echtzeit, und es
        verschwand im zeitraffer nur deshalb, weil dort der halt schon
        verbraucht statt verschoben hat.

        Wird doch starr verschoben, muss die ZEITSPALTE mitwandern. Sie ist
        bei der berechnung auf die damalige `world.time` bezogen worden
        (_compute_from_snapshot). Ohne die zeit-korrektur faellt die zeitbasis
        pro frame um ein sim_dt zurueck (gemessen 900-2700 s). Der renderer
        waehlt daraus ueber _world_to_screen_xy_at_time die epoche des
        plot-frames: bei einem bewegten frame-ursprung (body-centred
        non-rotating) landet dieselbe weltposition dadurch neben dem schiff --
        gemessen 54.5 px bei 2e-6 px/m, exakt der drift von Erde ueber 900 s.
        Der betrag wird in `_points_time_offset` mitgeschrieben, weil die
        punktzeiten damit nicht mehr zum schnappschuss passen und jeder, der
        aus ihnen eine lokale zeit zurueckrechnet, das wissen muss.
        """
        if self._points_count() == 0:
            return
        sx = float(ship.position.x)
        sy = float(ship.position.y)
        try:
            st = float(world.time) if world is not None else None
        except Exception:
            st = None

        # IM ZEITRAFFER NICHT STARR VERSCHIEBEN. Diese methode zieht sonst
        # die ganze kurve um den kopfversatz mit. Bei gehaltener kurve ist
        # dieser versatz gross (das gespeicherte ergebnis ist mehrere frames
        # alt und das schiff je frame ~1e8 m weiter), die kurve wuerde also
        # jeden frame quer durchs bild wandern -- und genau das macht sie
        # anschliessend fuer den halt unbrauchbar, weil ihre zeitspalte dann
        # nicht mehr zu ihrer geometrie passt (gemessen: kopfabstand 3.2e6 m
        # statt der punktweite 1e6 m, obwohl die echte abweichung zwischen
        # welt und predictor nur 37 m je frame betraegt).
        if (self._hold_active() and np is not None
                and isinstance(self.points, np.ndarray)
                and self.points.ndim == 2 and self.points.shape[0] >= 2
                and st is not None):
            dx = sx - float(self.points[0, 0])
            dy = sy - float(self.points[0, 1])
            dt = st - float(self.points[0, 2]) if self.points.shape[1] >= 3 else 0.0
            if math.isfinite(dx) and math.isfinite(dy) and math.isfinite(dt):
                self._apply_head_taper(self.points, sx, sy, st, dx, dy, dt)
                self._invalidate_derived_caches(soft=True)
            return

        # ECHTZEIT: DIESELBE MECHANIK WIE DER HALT.
        #
        # Der rolling-modus fuehrt in `_roll_states` einen zweiten, parallel
        # gehaltenen zustand mit, der punktweise zu `points` passen muss --
        # der bleibt beim alten weg. Alles andere verbraucht.
        if not self.rolling_mode and st is not None:
            if self._advance_points_along_curve(ship, st) is not None:
                return

        if np is not None and isinstance(self.points, np.ndarray):
            dx = sx - float(self.points[0, 0])
            dy = sy - float(self.points[0, 1])
            if math.isfinite(dx) and math.isfinite(dy):
                self.points[:, 0] += dx
                self.points[:, 1] += dy
                self.points[0, 0] = sx
                self.points[0, 1] = sy
                if st is not None and self.points.shape[1] >= 3:
                    dt = st - float(self.points[0, 2])
                    if math.isfinite(dt):
                        self.points[:, 2] += dt
                        self.points[0, 2] = st
                        # Die punktzeiten passen jetzt um `dt` nicht mehr zu
                        # `snapshot["sim_time"]` -- siehe _points_time_offset.
                        self._points_time_offset = float(
                            getattr(self, '_points_time_offset', 0.0)) + dt
                try:
                    if (
                        np is not None
                        and isinstance(self._roll_states, np.ndarray)
                        and self._roll_states.shape[0] == self.points.shape[0]
                        and self._roll_states.shape[1] >= 2
                    ):
                        self._roll_states[:, 0] += dx
                        self._roll_states[:, 1] += dy
                        self._roll_states[0, 0] = sx
                        self._roll_states[0, 1] = sy
                except Exception:
                    pass
        else:
            try:
                t0 = float(self.points[0][2])
            except Exception:
                t0 = 0.0
            # zeitbasis mitziehen (siehe docstring); ohne world.time bleibt sie
            # wie bisher stehen.
            dt = (st - t0) if st is not None else 0.0
            if not math.isfinite(dt):
                dt = 0.0
            t0 += dt
            if dt:
                self._points_time_offset = float(
                    getattr(self, '_points_time_offset', 0.0)) + dt
            try:
                dx = sx - float(self.points[0][0])
                dy = sy - float(self.points[0][1])
                for i, p in enumerate(self.points):
                    if i == 0:
                        self.points[i] = (sx, sy, t0)
                    elif hasattr(p, "__len__") and len(p) >= 3:
                        self.points[i] = (float(p[0]) + dx, float(p[1]) + dy, float(p[2]) + dt)
                    else:
                        self.points[i] = (float(p[0]) + dx, float(p[1]) + dy)
            except Exception:
                self.points[0] = (sx, sy, t0)

    def _count_recomputed_points(self, old_points, new_points, tol=1e-6):
        """Gibt die Anzahl der Einträge in `new_points` zurück, die sich von `old_points` unterscheiden.

        Der vergleich überspringt den ersten punkt (anker) und behandelt
        einen zusätzlichen "tail" in `new_points` gegenüber `old_points`
        als neu berechnet.
        """
        try:
            if old_points is None:
                old_len = 0
            else:
                if np is not None and isinstance(old_points, np.ndarray):
                    old_len = int(old_points.shape[0])
                else:
                    old_len = len(old_points)
        except Exception:
            old_len = 0

        try:
            if new_points is None:
                return 0
            if np is not None and isinstance(new_points, np.ndarray):
                new_len = int(new_points.shape[0])
            else:
                new_len = len(new_points)
        except Exception:
            return 0

        if old_len <= 0:
            return max(0, new_len)

        try:
            if np is not None and isinstance(new_points, np.ndarray) and isinstance(old_points, np.ndarray):
                old_arr = old_points
                new_arr = new_points
            else:
                old_arr = np.array(old_points, dtype=np.float64)
                new_arr = np.array(new_points, dtype=np.float64)
        except Exception:
            try:
                old_arr = np.array(old_points, dtype=np.float64)
                new_arr = np.array(new_points, dtype=np.float64)
            except Exception:
                return max(0, new_len)

        min_len = min(int(old_arr.shape[0]), int(new_arr.shape[0]))

        if min_len <= 1:
            changed_in_overlap = 0
        else:
            a = old_arr[1:min_len, :2]
            b = new_arr[1:min_len, :2]
            diffs = np.abs(a - b) > float(tol)
            rows_changed = np.any(diffs, axis=1)
            changed_in_overlap = int(np.count_nonzero(rows_changed))

        added_tail = max(0, int(new_arr.shape[0]) - int(old_arr.shape[0]))

        return changed_in_overlap + added_tail

    def set_view_scale(self, scale: float):
        try:
            scale = float(scale)
        except Exception:
            return
        if scale > 0.0:
            old = self._view_scale

            if old is not None:
                try:
                    rel_change = abs(scale - old) / max(abs(old), 1e-30)
                except Exception:
                    rel_change = 0.0
                if rel_change <= self.snapshot_view_rel_tol:
                    return

            # Neu rechnen lohnt nur, wenn der zoom die WIRKSAME punktdichte
            # veraendert. Mit angepinntem horizont (length = num_points *
            # precision, siehe test.py) klemmt _horizon_spacing_floor() die
            # zoom-verfeinerung bei JEDEM zoomwert auf exakt `precision` --
            # der synchrone _compute_full lieferte dann eine bit-identische
            # linie und kostete trotzdem die volle rechenzeit im hauptthread,
            # einmal pro mausrad-raste. Das war der massive fps-einbruch beim
            # zoomen. Aendert der zoom die dichte wirklich (nicht
            # angepinnter horizont), bleibt das verhalten wie zuvor.
            try:
                eff_old = float(self._effective_precision())
            except Exception:
                eff_old = None

            self._view_scale = scale

            try:
                eff_new = float(self._effective_precision())
            except Exception:
                eff_new = None
            eff_changed = True
            if eff_old is not None and eff_new is not None:
                eff_changed = (
                    abs(eff_new - eff_old) / max(abs(eff_old), 1e-30)
                    > self.snapshot_view_rel_tol
                )
            # Nur ueberspringen, wenn es eine linie zum BEHALTEN gibt: ohne
            # punkte garantierte der alte weg, dass der naechste update()
            # synchron eine baut -- diese zusicherung bleibt bestehen.
            try:
                has_points = self._points_count() > 0
            except Exception:
                has_points = False
            if not eff_changed and has_points:
                if self.debug:
                    print("PRED_DBG_VIEW_SCALE: eff_precision unchanged, no recompute")
                return

            try:
                self._view_scale_changed = True
                if self.debug:
                    print("PRED_DBG_VIEW_CHANGED: flagged for sync recompute")
            except Exception:
                pass
            if self.debug:
                try:
                    eff = self._effective_precision()
                except Exception:
                    eff = self.precision
                print(f"PRED_DBG_VIEW_SCALE: old={old} new={self._view_scale} eff_precision={eff}")

  
            try:
                if old is not None and self.async_compute and len(getattr(self, "_pending_futures", [])) > 0:
                    rel = abs(self._view_scale - old) / max(abs(old), 1e-30)
                    if rel > 0.02:
                        if self.debug:
                            print(f"PRED_DBG_CANCEL_PENDING: zoom rel_change={rel:.3f} canceling pending job")
                        self._cancel_pending_job()
            except Exception:
                pass

            try:
                self._view_change_cooldown_until = time.time() + float(self.view_change_cooldown)
                if self.debug:
                    print(f"PRED_DBG_VIEW_COOLDOWN: until={self._view_change_cooldown_until:.6f}")
            except Exception:
                pass

    # ------------------------------------------------------ zeitraffer-halt

    def set_hold(self, enabled):
        """Zeitraffer-halt ein/aus. Ausschalten erzwingt eine neuberechnung.

        Die beiden richtungen sind NICHT symmetrisch.

        AUSSCHALTEN (zurueck in die echtzeit) entwertet hart: der spieler darf
        von da an sofort wieder schub geben, und die gehaltene kurve weiss
        davon nichts.

        EINSCHALTEN dagegen uebernimmt eine kurve, die der asynchrone weg bis
        zum vorigen frame in jedem frame frisch gehalten hat -- sie ist also
        genau so gut wie eine neu gerechnete. Hart zu entwerten kostete dort
        gemessen 14.1 ms im hauptthread beim schritt 10m/s -> 1h/s (der
        stufe, bei der der halt anspringt), gegen 0.2 ms in den nachbarn.
        Also weich: neu ANFORDERN und derweil weiterhalten, wie beim
        stufenwechsel (siehe _request_hold_recompute).
        """
        enabled = bool(enabled)
        if enabled == getattr(self, 'hold_enabled', False):
            return
        self.hold_enabled = enabled
        self._synthetic_head = False
        self._hold_pending_swap = False
        self._hold_invalidated = True
        if enabled:
            self._hold_soft_invalidated = True
            # `_resume_context` stammt vom letzten asynchronen lauf und gehoert
            # damit zur kurve, die jetzt gehalten wird -- stehen lassen, sonst
            # kann sie waehrend des anlaufens nicht nachlegen. Festgehalten
            # wird er in _request_hold_recompute, dem einzigen besitzer.
        else:
            self._hold_soft_invalidated = False
            self._resume_context = None
            self._hold_resume_context = None

    def invalidate_hold(self, soft=False):
        """Die gehaltene kurve ist ueberholt (schub, rahmenwechsel, ...).

        `soft=True` heisst: die kurve ist GEOMETRISCH weiterhin richtig, nur
        ihre parameter stimmen nicht mehr (horizont oder punktabstand
        verstellt). Das schiff sitzt weiter auf ihr, sie reicht weiter in die
        zukunft -- sie ist bloss zu kurz oder zu lang. Ein solcher wechsel
        darf deshalb NACHGEREICHT werden, statt den hauptthread anzuhalten;
        siehe _hold_advance und _request_hold_recompute.

        Der harte weg bleibt fuer alles, was die kurve wirklich unbrauchbar
        macht: sprung, reparenting, rahmenwechsel, ende des halts.
        """
        self._hold_invalidated = True
        self._hold_soft_invalidated = bool(soft) and not self.rolling_mode
        self._synthetic_head = False
        if not soft:
            # Weiterrechnen geht nur auf einer kurve, die noch gilt.
            self._resume_context = None
            self._hold_resume_context = None
            self._hold_pending_swap = False

    def _hold_active(self):
        if not bool(getattr(self, 'hold_enabled', False)):
            return False
        if self.rolling_mode:
            return False
        if not self.initialized:
            return False
        # Eine zoom-aenderung veraendert die punktdichte und muss deshalb
        # durch den normalen rechenweg -- der halt darf sie nicht schlucken.
        if getattr(self, '_view_scale_changed', False):
            return False
        return True

    def _hold_advance(self, ship, world):
        """Kurve VERBRAUCHEN statt neu rechnen. True = frame ist erledigt.

        WARUM. Ohne halt ruft update() bei jedem frame eine neuberechnung an
        und `_anchor_first_point` schiebt die gespeicherte kurve STARR so,
        dass ihr kopf auf dem schiff sitzt. Bei 1m/s ist der versatz je frame
        winzig. Bei 7d/s rueckt das schiff je frame um ~10 000 sim-sekunden
        bahn weiter -- die ganze kurve wird also um diesen betrag quer
        verschoben und springt zurueck, sobald ein frisch gerechnetes
        ergebnis eintrifft. Genau dieser wechsel ist das "zittern" der linie
        und der Ap/Pe-marker.

        Richtig ist: die vorhersage ist eine eigenschaft der BAHN, nicht des
        augenblicks. Ohne schub bleibt sie stehen und das schiff rutscht an
        ihr entlang. Also werden vorn die punkte weggeworfen, deren zeit
        bereits vergangen ist (die zeitspalte ist absolute sim-zeit, das ist
        exakt und per suchlauf billig), und der rest bleibt, wo er ist.

        Der kopf wird trotzdem an das schiff gezogen, aber ABKLINGEND ueber
        die ersten `hold_taper_points` punkte -- welt und predictor
        propagieren die planeten leicht unterschiedlich, ohne korrektur
        klafft am schiff eine luecke. Die korrektur voll auf die ganze kurve
        zu legen waere wieder die starre verschiebung von oben.

        Failsafe: laeuft der vorrat unter `hold_refresh_fraction`, gibt die
        methode False zurueck und der normale weg rechnet nach. Die linie
        kann also nicht auslaufen.
        """
        if getattr(self, '_hold_invalidated', False):
            # WEICHE entwertung -> ANFORDERN statt anhalten. Die kurve, die
            # hier steht, ist geometrisch weiterhin richtig; nur ihr horizont
            # bzw. punktabstand ist ueberholt. Sie darf also weiterlaufen,
            # waehrend die neue im hintergrund entsteht. Gelingt das nicht
            # (kein async, keine kurve, kein worker frei), bleibt der harte
            # weg -- die zusicherung "update() baut synchron eine, wenn keine
            # da ist" gilt unveraendert.
            if (getattr(self, '_hold_soft_invalidated', False)
                    and self._request_hold_recompute(ship, world)):
                self._hold_invalidated = False
                self._hold_soft_invalidated = False
                # und weiter unten ganz normal verbrauchen
            else:
                self._hold_invalidated = False
                self._hold_soft_invalidated = False
                self._synthetic_head = False
                return False
        if ship is None or world is None or np is None:
            return False

        try:
            now = float(world.time)
        except Exception:
            return False

        # VERBRAUCHEN statt verschieben -- dieselbe mechanik, die inzwischen
        # auch die echtzeit benutzt (siehe _advance_points_along_curve).
        #
        # Das ergebnis wird IMMER uebernommen, auch wenn gleich darauf
        # abgebrochen wird: sonst rastet der halt ein. Bricht er ab, bevor der
        # schnitt steht, bleibt die kurve stehen, waehrend das schiff
        # weiterfliegt -- der kopfabstand waechst dann jeden frame weiter
        # (gemessen 6.4e5 -> 3.2e6 m in fuenf frames) und die
        # abbruchbedingung ist von da an dauerhaft erfuellt.
        drop = self._advance_points_along_curve(ship, now)
        if drop is None:
            return False

        points = self.points
        sx = float(ship.position.x)
        sy = float(ship.position.y)

        # HINTEN ANSTUECKELN, was vorn verbraucht wurde -> der horizont
        # bleibt konstant und die linie wandert mit, statt zu schrumpfen und
        # bei jeder auffrischung zurueckzuspringen.
        if drop > 0:
            budget = self._get_target_point_cap()
            missing = int(budget) - int(self.points.shape[0])
            # JE FRAME NUR EIN STUECK. Normal sind das die punkte, die vorn
            # gerade verbraucht wurden (bei 7d/s rund 170) -- die schranke
            # merkt man dort nicht. Sie greift, wenn das BUDGET springt:
            # `apply_predictor_horizon` zieht mit dem zeitraffer-schritt auch
            # das punktbudget mit, beim wechsel 7d/s -> 30d/s von 10 000 auf
            # 40 000. Die fehlenden 30 000 punkte in EINEM frame anzustueckeln
            # kostete gemessen 40.3 ms im hauptthread (nachbarframes 0.3 ms) --
            # genau der ruckler, den §17 fuer set_length schon beseitigt hat.
            # Verteilt ueber ein paar frames faellt er nicht auf, und die
            # bestellte neue kurve ist ohnehin schon unterwegs.
            cap = int(getattr(self, 'hold_extend_max_points', 1000) or 0)
            if cap > 0 and missing > cap:
                missing = cap
            if missing > 0:
                self._hold_extend_tail(missing)
            points = self.points

        target_points = self._get_target_point_cap()
        remaining = points.shape[0]
        refresh_at = max(4, int(target_points * float(getattr(
            self, 'hold_refresh_fraction', 0.25))))
        # LAEUFT SCHON EINE NEUE KURVE, IST DIE SCHWELLE EINE ANDERE.
        #
        # Sie misst den vorrat am ANGEPEILTEN budget. Waechst das budget
        # sprunghaft -- der zeitraffer-schritt zieht ueber
        # `apply_predictor_horizon` den horizont UND das punktbudget mit, beim
        # wechsel 7d/s -> 30d/s von 10 000 auf 40 000 --, dann rutscht die
        # noch vollstaendige kurve allein durch die neue bezugsgroesse unter
        # die schwelle, und der halt rechnet SYNCHRON nach: gemessen 43.8 ms
        # im hauptthread gegen 0.3 ms in den nachbarframes.
        #
        # Ist der ersatz bereits unterwegs (`_hold_pending_swap`), kann die
        # linie gar nicht auslaufen -- dann genuegt eine absolute
        # not-schwelle. Kommt der auftrag nicht an, raeumt `update()` das
        # flag ab und der harte weg steht im naechsten frame wieder offen.
        if getattr(self, '_hold_pending_swap', False):
            refresh_at = max(4, int(target_points * 0.02))
        if remaining < refresh_at:
            # Vorrat zu klein -> normaler weg rechnet nach (und der halt
            # greift danach wieder). Das ist die failsafe-schwelle.
            self._synthetic_head = False
            return False

        # Weicht das schiff von der gehaltenen kurve ab, stimmt sie nicht
        # mehr (schub, sprung, rahmenwechsel) -- dann lieber neu rechnen als
        # eine falsche kurve weiterzeichnen. Gemessen wird gegen die ZWEITE
        # stuetzstelle, denn die erste ist ja das schiff selbst. Regulaer
        # liegt es hoechstens eine punktweite davor; der spielraum darueber
        # faengt ab, dass welt und predictor die planeten nicht voellig
        # gleich propagieren (gemessen ~37 m je frame).
        if points.shape[0] >= 3:
            span = math.hypot(float(points[2, 0]) - float(points[1, 0]),
                              float(points[2, 1]) - float(points[1, 1]))
            gap = math.hypot(float(points[1, 0]) - sx,
                             float(points[1, 1]) - sy)
            if gap > max(span * 4.0, 1.0):
                self._synthetic_head = False
                return False

            # DER HALT BRAUCHT EINE OBERGRENZE FUER DEN SEITLICHEN VERSATZ.
            #
            # Die pruefung darueber misst den abstand ENTLANG der bahn und
            # laesst vier punktweiten zu -- an einer seitlichen abweichung
            # geht sie deshalb blind vorbei. Und der vorrat laeuft nie leer,
            # weil `_hold_extend_tail` hinten nachlegt: gemessen 0 volle
            # neuberechnungen in 3000 frames. Die gehaltene kurve wurde also
            # EINMAL gerechnet und danach nie wieder mit der welt verglichen.
            #
            # Welt und predictor rechnen aber nicht dasselbe (andere
            # schrittweiten, und die welt setzt die planeten ueber
            # `bodies.position_at_time` mit konstanter winkelrate, der
            # predictor mit echtem Kepler-solve). Der unterschied summiert
            # sich. Gemessen in einer erdumlaufbahn (rp 2e7 m, e = 0.3) bei
            # 1 h/s ueber 2.5 umlaeufe: das schiff steht am ende **3.9e5 m
            # = 1.96 % des bahnradius** neben der linie, und in einer
            # sonnenumlaufbahn ueber 350 tage 4.2e5 m. Das ist genau das
            # "schiff loest sich von der linie" -- und es verschwindet beim
            # verlassen des zeitraffers, weil `set_hold(False)` hart entwertet.
            #
            # Gemessen wird SENKRECHT zur kurve (die laengsrichtung ist
            # bereits durch den kopf abgedeckt) an den beiden ersten echten
            # stuetzstellen -- der kopf ist ja das schiff selbst.
            #
            # Die schwelle ist ein PIXELMASS, weil nur das sichtbar ist:
            # dieselbe weltlaenge ist zoom-abhaengig entweder unsichtbar oder
            # fingerdick. Angefordert wird ASYNCHRON (derselbe weg wie beim
            # stufenwechsel) -- die alte kurve bleibt stehen, bis die neue da
            # ist, es gibt also keinen ruckler und kein springen.
            dxs = float(points[2, 0]) - float(points[1, 0])
            dys = float(points[2, 1]) - float(points[1, 1])
            chord = math.hypot(dxs, dys)
            if chord > 0.0:
                drift = abs((sx - float(points[1, 0])) * dys
                            - (sy - float(points[1, 1])) * dxs) / chord
            else:
                drift = 0.0
            self.hold_drift_m = drift
            #
            # Getaktet wird das NICHT ueber eine uhr, sondern ueber
            # `_hold_pending_swap`: solange ein auftrag laeuft, wird kein
            # zweiter gestellt. Damit stellt sich die auffrischrate von
            # selbst auf "eine je rechendauer" ein -- dieselbe selbstregelung
            # wie beim schub. Eine feste echtzeit-sperre (0.25 s) war
            # nachweislich zu grob: gemessen 4 auffrischungen ueber 1500
            # frames, und der versatz lief zwischendurch wieder auf 4.4e5 m.
            if (drift > self._hold_drift_limit_m()
                    and not getattr(self, '_hold_pending_swap', False)):
                self._request_hold_recompute(ship, world)

        return True

    def _hold_drift_limit_m(self):
        """Erlaubter seitlicher versatz des schiffs von der gehaltenen kurve.

        Ein PIXELMASS, in meter umgerechnet -- siehe _hold_advance. Ohne
        bekannte zoomstufe bleibt nur die untergrenze.
        """
        px = float(getattr(self, 'hold_drift_max_px', 0.5) or 0.0)
        # px <= 0 heisst AUS -- keine anforderung, egal wie weit es auseinander
        # laeuft. (Das ist auch der schalter, mit dem die gegenprobe im test
        # das alte verhalten wiederherstellt.)
        if px <= 0.0:
            return float('inf')
        scale = getattr(self, '_view_scale', None)
        if scale is None or not math.isfinite(scale) or scale <= 0.0:
            # Ohne bekannte zoomstufe gibt es kein pixelmass -- dann lieber
            # nichts tun als eine weltlaenge zu raten.
            return float('inf')
        floor = float(getattr(self, 'hold_drift_min_m', 1.0) or 0.0)
        return max(floor, px / scale)

    def _hold_extend_tail(self, wanted):
        """Hinten so viele punkte anstueckeln, wie vorn verbraucht wurden.

        Damit bleibt der HORIZONT konstant. Ohne das wird die gehaltene kurve
        nur von vorn aufgebraucht, schrumpft also sichtbar, bis die
        auffrischung sie schlagartig wieder auf volle laenge bringt -- die
        linie pulsiert dann im takt der auffrischung, statt gleichmaessig
        mitzuwandern.

        Gerechnet wird als FORTSETZUNG desselben laufs: derselbe
        schnappschuss, derselbe integrator-zustand, dieselbe schrittweite
        (siehe _resume_context und die init_*-parameter des kernels). Die
        angehaengten punkte sind deshalb genau die, die eine von vornherein
        laengere rechnung geliefert haette -- kein bruch an der nahtstelle.

        Kostet nur die tatsaechlich verbrauchten punkte (bei 7d/s rund 170 je
        frame) statt der vollen neuberechnung von 10 000.
        """
        wanted = int(wanted)
        if wanted <= 0 or np is None:
            return 0
        # Waehrend ein stufenwechsel unterwegs ist, gehoert `_resume_context`
        # bereits zur NEUEN kurve (der worker setzt ihn beim fertigwerden).
        # Angesetzt werden muss aber an die kurve, die gerade gehalten wird.
        context = None
        if getattr(self, '_hold_pending_swap', False):
            context = getattr(self, '_hold_resume_context', None)
        if not context:
            context = getattr(self, '_resume_context', None)
        if not context:
            return 0
        if _compute_distance_points_rkn_numba is None:
            return 0
        points = self.points
        if not isinstance(points, np.ndarray) or points.shape[0] < 2:
            return 0

        snapshot = context['snapshot']
        px, py, vx, vy = context['state']
        if not all(math.isfinite(v) for v in (px, py, vx, vy)):
            return 0

        try:
            out, used, stats = _compute_distance_points_rkn_numba(
                px, py, vx, vy,
                0,
                float(snapshot.get("ref_px", 0.0)),
                float(snapshot.get("ref_py", 0.0)),
                snapshot["body_x"], snapshot["body_y"],
                snapshot["body_m"], snapshot["body_fixed"],
                context['body_scripted'], context['body_a'], context['body_e'],
                context['body_theta'], context['body_arg'], context['body_parent'],
                snapshot["G"], context['base_dt'], snapshot["precision"],
                int(wanted) + 1, int(max(10000, (wanted + 1) * 100)),
                context['min_dt'], context['max_dt'],
                context['rtol'], context['atol_pos'], context['atol_vel'],
                context['safety'], context['min_factor'], context['max_factor'],
                context['max_rejects'],
                context['use_time_dependent_bodies'], context['ref_index'],
                context['kernel_t'], context['accumulated'], context['proposed_dt'],
                1 if getattr(self, 'use_body_memo', True) else 0,
                float(context.get('max_dt_floor', context['max_dt'])),
                float(context.get('timescale_divisor', 0.0)),
            )
        except Exception:
            return 0

        used = int(used)
        if used <= 1:
            return 0

        # out[0] ist der fortsetz-punkt selbst und steht schon in der liste.
        addition = out[1:used].copy()
        addition[:, 2] += float(snapshot.get("sim_time", 0.0))
        self.points = np.concatenate((points, addition), axis=0)

        context['state'] = (float(stats[7]), float(stats[8]),
                            float(stats[9]), float(stats[10]))
        context['accumulated'] = float(stats[11])
        context['proposed_dt'] = float(stats[12])
        context['kernel_t'] = float(stats[13])
        self._invalidate_derived_caches(soft=True)
        return int(addition.shape[0])

    def _apply_head_taper(self, points, sx, sy, now, dx, dy, dt):
        """Kopf ans schiff ziehen -- ABKLINGEND ueber die ersten punkte.

        Der unterschied zu `_anchor_first_point` ist der ganze punkt der
        sache: dort wird die KOMPLETTE kurve starr um (dx, dy) verschoben,
        hier klingt die korrektur ueber `hold_taper_points` punkte auf null
        ab. Das fernfeld bleibt also stehen, wo es steht.
        """
        taper = int(max(1, min(int(getattr(self, 'hold_taper_points', 64)),
                               points.shape[0])))
        weights = np.zeros(points.shape[0], dtype=np.float64)
        weights[:taper] = np.linspace(1.0, 0.0, taper, endpoint=False)

        points[:, 0] += dx * weights
        points[:, 1] += dy * weights
        if points.shape[1] >= 3:
            points[:, 2] += dt * weights
        points[0, 0] = sx
        points[0, 1] = sy
        if points.shape[1] >= 3:
            points[0, 2] = now

    def _invalidate_derived_caches(self, soft=False):
        """Von der punkteliste abgeleitete zwischenergebnisse verwerfen.

        soft=True heisst: die kurve wurde nur vorn verbraucht / hinten
        verlaengert / am kopf angeschmiegt (zeitraffer-halt). Die apsis-
        marker der verbliebenen punkte sind dann weiterhin gueltig und
        get_apsis_markers() darf sie gefiltert weiterreichen, statt neu zu
        scannen.
        """
        for attr in ('_apsis_cache_key', '_apsis_cache_value'):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
        if not soft:
            # NEUE GEOMETRIE -> die gemerkten marker gehoeren zu einer kurve,
            # die es nicht mehr gibt. Der zaehler ist das, was der weiche weg
            # in get_apsis_markers() dagegenhaelt: er darf nur filtern,
            # solange die kurve DIESELBE ist, auf der er zuletzt gesucht hat.
            self._points_generation = int(getattr(self, '_points_generation', 0)) + 1
        self._apsis_soft_stale = bool(soft)

    def _horizon_spacing_floor(self):
        """Feinste punktdichte, die den HORIZONT noch traegt -- oder None.

        Die kernel setzt punkte in festem ABSTAND (`_effective_precision`)
        und hoechstens `num_points` viele. Der wirklich gezeichnete bogen
        ist damit `num_points * spacing`. Sobald das kleiner als `length`
        wird, endet die linie einfach vorzeitig -- ein engerer punktabstand
        VERKUERZT also die vorhersage.

        Der bodenwert `length / num_points` ist genau der abstand, bei dem
        das punktbudget den horizont gerade noch ausfuellt.
        """
        if self.length is None:
            return None
        try:
            budget = int(self.num_points)
            horizon = float(self.length)
        except Exception:
            return None
        if budget <= 0 or not (horizon > 0.0):
            return None
        return horizon / float(budget)

    def _effective_precision(self):
        effective = float(self.precision)
        if self.auto_precision_from_zoom and self._view_scale is not None:
            zoom_precision = self.target_screen_step_px / max(self._view_scale, 1e-30)
            effective = min(effective, max(self.min_precision, zoom_precision))

        # HORIZONT VOR PUNKTDICHTE. Ohne diese schranke frisst das zoom-
        # abhaengige verfeinern den horizont auf: gemessen blieb bei
        # view_scale 2e-5 noch 10 % der vorhersage uebrig, bei 2e-4 noch
        # 1 %. Auf dem schirm sieht das aus, als wuerde die linie an der
        # ersten bildkante abgeschnitten und komme nie zurueck -- samt
        # verschwundener Ap/Pe-marker und leerem CLOSEST/T-CA, weil beide
        # ueber dieselbe punkteliste laufen.
        #
        # Bei erreichtem budget wird also GROEBER gezeichnet statt KUERZER.
        # Das ist die richtige seite des tauschs: die groebere teilung fehlt
        # nur dort, wo die bahn ohnehin kaum kruemmt (der bogenfehler einer
        # sehne waechst mit c^2/8R und ist im fernfeld weit unter einem
        # pixel), waehrend ein fehlender horizont die anzeige unbrauchbar
        # macht. Mehr naehe-detail gibt es ueber ein groesseres
        # `predictor.num_points`, nicht ueber einen kuerzeren horizont.
        floor = self._horizon_spacing_floor()
        if floor is not None and effective < floor:
            effective = floor
        return effective

    def _serialize_bodies_numba(self, world):
        count = len(world.body)
        body_x = np.empty(count, dtype=np.float64)
        body_y = np.empty(count, dtype=np.float64)
        body_m = np.empty(count, dtype=np.float64)
        body_fixed = np.empty(count, dtype=np.uint8)
        for i, b in enumerate(world.body):
            body_x[i] = float(b.position.x)
            body_y[i] = float(b.position.y)
            body_m[i] = float(b.mass)
            body_fixed[i] = 1 if getattr(b, "fixed", True) else 0
        return body_x, body_y, body_m, body_fixed

    def _serialize_body_orbits_numba(self, world):
        count = len(world.body)
        body_scripted = np.empty(count, dtype=np.uint8)
        body_a = np.empty(count, dtype=np.float64)
        body_e = np.empty(count, dtype=np.float64)
        body_theta = np.empty(count, dtype=np.float64)
        body_arg = np.empty(count, dtype=np.float64)
        body_parent = np.empty(count, dtype=np.int64)

        body_to_index = {}
        for i, b in enumerate(world.body):
            body_to_index[b] = int(i)

        for i, b in enumerate(world.body):
            try:
                a = float(getattr(b, "semi_major_axis", 0.0) or 0.0)
            except Exception:
                a = 0.0
            try:
                e = float(getattr(b, "eccentricity", 0.0) or 0.0)
            except Exception:
                e = 0.0
            try:
                theta = float(getattr(b, "theta", 0.0) or 0.0)
            except Exception:
                theta = 0.0
            try:
                arg = float(getattr(b, "arg_periapsis", 0.0) or 0.0)
            except Exception:
                arg = 0.0

            parent = getattr(b, "is_moon_of", None)
            parent_index = body_to_index.get(parent, -1)
            scripted = bool(getattr(b, "scripted_orbit", False)) or (a > 0.0 and parent_index >= 0)

            body_scripted[i] = 1 if scripted else 0
            body_a[i] = a
            body_e[i] = e
            body_theta[i] = theta
            body_arg[i] = arg
            body_parent[i] = int(parent_index)

        return body_scripted, body_a, body_e, body_theta, body_arg, body_parent

    def _cancel_pending_job(self):
    # alle wartenden futures abbrechen (unterstützt multi-worker-modus).
        pending = getattr(self, "_pending_futures", [])

        # cancel any futures in the list
        for job_id, fut in list(pending):
            try:
                if not fut.done():
                    fut.cancel()
            except Exception:
                pass
        pending.clear()
        self._pending_job_id = 0

        # also cancel legacy single future if present
        pf = getattr(self, '_pending_future', None)
        if pf is not None:
            try:
                if not pf.done():
                    pf.cancel()
            except Exception:
                pass
            self._pending_future = None
            self._pending_job_id = 0

    def _characteristic_timescale(self, world, ship):
        """sqrt(r_dominant / |g_total|) am schiff, in sekunden -- oder None.

        Dieselbe groesse, die `world.characteristic_timescale` fuer die
        zeitraffer-obergrenze benutzt; sie wird hier durchgereicht, damit es
        nur EINE definition davon gibt. Faellt die welt aus (tests reichen
        manchmal nur ein objekt herein), gibt es None und die decke bleibt,
        wie sie war.
        """
        if world is None or ship is None:
            return None
        fn = getattr(world, 'characteristic_timescale', None)
        if fn is None:
            return None
        try:
            value = fn(ship)
        except Exception:
            return None
        if value is None:
            return None
        try:
            value = float(value)
        except Exception:
            return None
        return value if math.isfinite(value) and value > 0.0 else None

    def _make_snapshot(self, ship, world, max_points):
        effective_precision = self._effective_precision()
        ref_enabled, ref_px, ref_py = self._resolve_reference_body(world)
        physics_ref_enabled = 0
        ref_index = self._current_reference_body_index()

        # Horizon-scaled far-field step ceiling. A long look-ahead over a smooth
        # arc is otherwise integrated at the fixed max_dt cap, so cost grows
        # ~arc/max_dt. Raise max_dt for long horizons to target a bounded step
        # budget (roughly constant compute cost); the adaptive tolerance +
        # step-doubling still refine near planets, so only the smooth far field
        # coarsens. Floored at the preset max_dt (short horizons unchanged) and
        # capped by the ceiling (close-approach safety). Tied to the HORIZON
        # (arc = max_points × precision), not to `precision`, so the spacing
        # decouple holds.
        eff_max_dt = float(self.rkn_max_dt)
        if self.rkn_adaptive_far_maxdt and float(self.rkn_far_field_target_steps) > 0.0:
            horizon_arc = float(max_points) * float(effective_precision)
            # Wieviel ZEIT deckt dieser bogen ab? Genau das braucht die
            # schrittzahl-schaetzung -- und genau das darf NICHT aus der
            # momentangeschwindigkeit kommen. Auf einer exzentrischen bahn ist
            # sie im perihel das MAXIMUM und im aphel das MINIMUM der ganzen
            # bahn, der fehler geht also in beide richtungen und ausgerechnet
            # im perihel nach unten: die schaetzung faellt zu kurz aus, die
            # schrittweite wird zu klein gedeckelt und der lauf kostet ein
            # vielfaches. Gemessen auf Pe 29 Gm / Ap 129 Gm bei 32x horizont:
            # 6663 schritte / 256 ms im perihel gegen 1160 / 43 ms im aphel --
            # dieselbe bahn, derselbe bogen, 6x. Genau das ist das stocken der
            # linie am perihel (die auffrischung faellt unter die bildrate)
            # und genau deshalb ist am aphel nichts davon zu merken.
            #
            # Die ehrliche groesse ist die MITTLERE inverse geschwindigkeit
            # ueber den bogen, und die kennt der letzte lauf bereits exakt:
            # seine zeitspanne durch seine bogenlaenge. Als verhaeltnis
            # gespeichert ueberlebt sie auch ein '+'/'-' auf den horizont.
            # Rueckkopplung ohne ruecklauf: die zeitspanne ist eine eigenschaft
            # der bahn, nicht der schrittweite -- ein groesseres max_dt
            # verandert sie nicht, es gibt also keinen regelkreis.
            time_per_arc = float(getattr(self, "_horizon_time_per_arc", 0.0) or 0.0)
            if time_per_arc <= 0.0:
                # Erster lauf: nichts gemessen, also der alte schaetzer.
                speed = math.hypot(float(ship.velocity.x), float(ship.velocity.y))
                if speed > 1.0:
                    time_per_arc = 1.0 / speed
            if time_per_arc > 0.0 and horizon_arc > 0.0:
                desired = (horizon_arc * time_per_arc) / float(self.rkn_far_field_target_steps)
                ceiling = float(self.rkn_max_dt_ceiling)
                # DIE DECKE DARF DIE BAHN NICHT UEBERSPRINGEN.
                #
                # `desired` kennt nur den horizont, nicht die bahn. Bei vielen
                # '+'-druecken wird sie deshalb groesser als ein nennenswerter
                # bruchteil der umlaufzeit -- und dann liegt die schrittweite
                # nicht mehr an der fehlerkontrolle, sondern an der decke.
                # Gemessen in einer erdumlaufbahn (rp 2e7 m, e = 0.6, T = 97 h)
                # bei 64x horizont: die linie weicht gegen dieselbe rechnung
                # mit fester decke (1500 s) um bis zu **6.0e7 m** ab, mehr als
                # die bahn selbst gross ist -- die vorhersage zeigt dann
                # schlicht eine andere bahn.
                #
                # Dieselbe schranke, die schon der zeitraffer benutzt:
                # `sqrt(r_dominant/|g|)`, fuer eine kreisbahn genau T/2pi.
                # Im FERNFELD (heliozentrisch, t_char ~ 5e6 s) ist sie um
                # groessenordnungen groesser als die decke und aendert nichts
                # -- der fernfeld-gewinn bleibt also unangetastet.
                #
                # SIE WIRD HIER NICHT MEHR EINGERECHNET, SONDERN IM KERNEL JE
                # SCHRITT. Hier war sie EINE zahl fuer den ganzen lauf, gemessen
                # am schiff, wie es beim anlegen des schnappschusses stand --
                # und damit falsch fuer jede bahn, die ihr regime verlaesst. Auf
                # einer abflugbahn (Erdorbit -> Jupiter) galt die zeitskala der
                # ERDE fuer die ganzen 2.85 jahre reiseflug: 24 633 schritte /
                # 899 ms statt 1 276 / 56 ms. Der kernel wertet dieselbe formel
                # jetzt am jeweiligen ORT aus (`_local_timescale_numba`), womit
                # die klammer im nahfeld unveraendert greift und sich erst
                # oeffnet, wenn das schiff den koerper wirklich verlassen hat.
                if not self.use_local_step_ceiling:
                    t_char = self._characteristic_timescale(world, ship)
                    if t_char is not None and t_char > 0.0:
                        orbit_cap = t_char / max(1e-9, float(self.rkn_max_dt_timescale_divisor))
                        if orbit_cap < ceiling:
                            ceiling = orbit_cap
                eff_max_dt = max(eff_max_dt, min(desired, ceiling))

        snapshot = {
            "ship_px": float(ship.position.x),
            "ship_py": float(ship.position.y),
            "ship_vx": float(ship.velocity.x),
            "ship_vy": float(ship.velocity.y),
            "ref_enabled": int(physics_ref_enabled),
            "reference_body_index": int(ref_index),
            "trajectory_version": int(self._trajectory_version),
            "ref_px": float(ref_px),
            "ref_py": float(ref_py),
            "G": float(world.G),
            "dt": float(self.dt),
            "precision": float(effective_precision),
            "max_points": int(max_points),
            "max_iters": int(max(10000, max_points * 100)),
            "numba": True,
            "integrator_mode": str(self.integrator_mode),
            "aspi_min_dt": float(self.aspi_min_dt),
            "aspi_max_dt": float(self.aspi_max_dt),
            "aspi_safety_g": float(self.aspi_safety_g),
            "aspi_safety_m": float(self.aspi_safety_m),
            "aspi_close_acc_threshold": float(self.aspi_close_acc_threshold),
            "aspi_use_rk4_fallback": bool(self.aspi_use_rk4_fallback),
            "rkn_min_dt": float(self.rkn_min_dt),
            "rkn_max_dt": float(eff_max_dt),
            # Boden und teiler der ORTLICHEN decke. Der boden ist die
            # schrittdecke der qualitaetsstufe -- die ortliche klammer darf nie
            # darunter, sonst wuerde sie das nahfeld strenger rechnen als der
            # alte globale weg. Teiler 0 = klammer aus.
            "rkn_max_dt_floor": float(self.rkn_max_dt),
            "rkn_max_dt_timescale_divisor": (
                float(self.rkn_max_dt_timescale_divisor)
                if (self.use_local_step_ceiling
                    and self.rkn_adaptive_far_maxdt
                    and float(self.rkn_max_dt_timescale_divisor) > 0.0)
                else 0.0
            ),
            "rkn_rtol": float(self.rkn_rtol),
            "rkn_atol_pos": float(self.rkn_atol_pos),
            "rkn_atol_vel": float(self.rkn_atol_vel),
            "rkn_safety": float(self.rkn_safety),
            "rkn_min_factor": float(self.rkn_min_factor),
            "rkn_max_factor": float(self.rkn_max_factor),
            "rkn_max_rejects": int(self.rkn_max_rejects),
            "base_precision": float(self.base_precision),
            "rkn_interval_coupling": bool(self.rkn_interval_coupling),
            "rkn_interval_tol_exponent": float(self.rkn_interval_tol_exponent),
            "strict_snapshot_matching": bool(self.strict_snapshot_matching),
            "use_time_dependent_bodies": bool(self.use_time_dependent_bodies),
            "use_reference_acceleration_correction": False,
        }

        try:
            snapshot["sim_time"] = float(world.time)
        except Exception:
            snapshot["sim_time"] = 0.0
        try:
            snapshot["submit_ts"] = float(time.time())
        except Exception:
            snapshot["submit_ts"] = 0.0

        try:
            snapshot["view_scale"] = float(self._view_scale) if self._view_scale is not None else None
        except Exception:
            snapshot["view_scale"] = None
        try:
            snapshot["eff_precision"] = float(self._effective_precision())
        except Exception:
            snapshot["eff_precision"] = None
        # Muss ueber den schnappschuss laufen, nicht ueber self: der kernel
        # laeuft im worker-thread und darf den schalter nicht mitten im lauf
        # wechseln sehen.
        snapshot["use_body_memo"] = bool(getattr(self, "use_body_memo", True))
        body_x, body_y, body_m, body_fixed = self._serialize_bodies_numba(world)
        snapshot["body_x"] = body_x
        snapshot["body_y"] = body_y
        snapshot["body_m"] = body_m
        snapshot["body_fixed"] = body_fixed
        (
            body_scripted,
            body_a,
            body_e,
            body_theta,
            body_arg,
            body_parent,
        ) = self._serialize_body_orbits_numba(world)
        snapshot["body_scripted"] = body_scripted
        snapshot["body_a"] = body_a
        snapshot["body_e"] = body_e
        snapshot["body_theta"] = body_theta
        snapshot["body_arg"] = body_arg
        snapshot["body_parent"] = body_parent
        snapshot["body_names"] = [str(getattr(b, "name", "")) for b in world.body]
        if getattr(self, "debug_moving_sources", False):
            self._debug_moving_source_snapshot(snapshot)
        return snapshot

    def _compute_from_snapshot(self, snapshot):
        # Thin timing shim: record the wall-clock cost of the actual trajectory
        # compute into self.last_compute_ms (single choke point for both the
        # async worker and the synchronous paths). See last_compute_ms in __init__.
        _t0 = time.perf_counter()
        try:
            result = self._compute_from_snapshot_impl(snapshot)
            self._record_horizon_time_per_arc(result, snapshot)
            return result
        finally:
            self.last_compute_ms = (time.perf_counter() - _t0) * 1000.0

    def _record_horizon_time_per_arc(self, result, snapshot):
        """Mittlere inverse geschwindigkeit ueber den horizont mitschreiben.

        Einzige quelle fuer die schrittweiten-deckelung in `_make_snapshot`
        (dort steht, warum die momentangeschwindigkeit dafuer untauglich ist).
        Laeuft auf dem worker-thread; es ist eine einzelne float-zuweisung,
        also unter der GIL atomar -- der hauptthread liest nie einen halben
        wert. Nur volle laeufe zaehlen: eine kurze fortsetzung (der
        schwanz-anbau im zeitraffer) misst nur ihr eigenes stueck bahn und
        wuerde die mittelung wieder auf einen momentanwert zusammenziehen.
        """
        try:
            points = result.get("points") if isinstance(result, dict) else None
            if points is None or len(points) < 3:
                return
            precision = float(snapshot.get("precision", 0.0) or 0.0)
            max_points = int(snapshot.get("max_points", 0) or 0)
            n = int(len(points))
            if precision <= 0.0 or max_points <= 0 or n < max(3, max_points // 2):
                return
            arc = float(n - 1) * precision
            span = float(points[-1, 2]) - float(points[0, 2])
            if arc > 0.0 and math.isfinite(span) and span > 0.0:
                self._horizon_time_per_arc = span / arc
        except Exception:
            pass

    def _compute_from_snapshot_impl(self, snapshot):
        mode = self._normalize_integrator_mode(snapshot.get("integrator_mode", "rkn"))
        self._debug_integrator_mode("compute", snapshot)
        rkn_stats = None

        if mode == "rkn":
            min_dt = float(snapshot.get("rkn_min_dt", 0.1))
            max_dt = float(snapshot.get("rkn_max_dt", 1500.0))
            base_dt = float(snapshot.get("dt", 60.0))
            rtol = float(snapshot.get("rkn_rtol", 1e-7))
            atol_pos = float(snapshot.get("rkn_atol_pos", 10.0))
            atol_vel = float(snapshot.get("rkn_atol_vel", 1e-4))
            safety = float(snapshot.get("rkn_safety", 0.9))
            min_factor = float(snapshot.get("rkn_min_factor", 0.2))
            max_factor = float(snapshot.get("rkn_max_factor", 5.0))
            max_rejects = int(snapshot.get("rkn_max_rejects", 32))

            if (not math.isfinite(min_dt)) or min_dt <= 0.0:
                min_dt = 0.1
            if (not math.isfinite(max_dt)) or max_dt <= 0.0:
                max_dt = 1500.0
            if max_dt < min_dt:
                max_dt = min_dt
            if (not math.isfinite(base_dt)) or base_dt <= 0.0:
                base_dt = max_dt
            if (not math.isfinite(rtol)) or rtol < 0.0:
                rtol = 1e-7
            if (not math.isfinite(atol_pos)) or atol_pos <= 0.0:
                atol_pos = 10.0
            if (not math.isfinite(atol_vel)) or atol_vel <= 0.0:
                atol_vel = 1e-4
            if (not math.isfinite(safety)) or safety <= 0.0:
                safety = 0.9
            if (not math.isfinite(min_factor)) or min_factor <= 0.0:
                min_factor = 0.2
            if (not math.isfinite(max_factor)) or max_factor < min_factor:
                max_factor = max(min_factor, 5.0)
            if max_rejects < 0:
                max_rejects = 0

            # --- Option A: intervall-gekoppelte schrittweite + toleranz -------
            # Koppelt die schrittzahl an die punktzahl statt an die bogenlänge.
            # Die max_dt-decke begrenzt schritte auf ~ein abtast-intervall pro
            # schritt (kosten-obergrenze ~ num_points auf glatten bögen); die
            # toleranz-lockerung sorgt dafür, dass diese decke auf glatten bögen
            # tatsächlich bindet, statt unnötig fein zu unterteilen. Nahe
            # vorbeiflügen übersteigt der fehler auch die gelockerte toleranz
            # weiterhin → unterteilung bis min_dt bleibt erhalten (sicherheit).
            # base_precision >= effektive precision → coarsen==1 → identität.
            if bool(snapshot.get("rkn_interval_coupling", False)):
                base_precision = float(snapshot.get("base_precision", 0.0))
                precision_val = float(snapshot.get("precision", 0.0))
                if base_precision > 0.0 and precision_val > base_precision:
                    coarsen = precision_val / base_precision
                    speed = math.hypot(
                        float(snapshot.get("ship_vx", 0.0)),
                        float(snapshot.get("ship_vy", 0.0)),
                    )
                    # zielschrittweite: ~ein abtast-intervall arc pro schritt
                    if speed > 1e-9:
                        dt_target = precision_val / speed
                    else:
                        dt_target = max_dt * coarsen
                    # decke nur anheben, nie senken; gegen absurde werte kappen
                    eff_max_dt = max(max_dt, min(dt_target, max_dt * coarsen))
                    if math.isfinite(eff_max_dt) and eff_max_dt > max_dt:
                        max_dt = eff_max_dt
                        base_dt = max_dt
                    # toleranz mit der vergröberung lockern (RKN4: fehler ~ dt^p)
                    exponent = float(snapshot.get("rkn_interval_tol_exponent", 4.0))
                    tol_scale = coarsen ** exponent
                    if math.isfinite(tol_scale) and tol_scale > 1.0:
                        rtol = rtol * tol_scale
                        atol_pos = atol_pos * tol_scale
                        atol_vel = atol_vel * tol_scale

            body_scripted = snapshot.get("body_scripted", None)
            body_a = snapshot.get("body_a", None)
            body_e = snapshot.get("body_e", None)
            body_theta = snapshot.get("body_theta", None)
            body_arg = snapshot.get("body_arg", None)
            body_parent = snapshot.get("body_parent", None)
            body_count = snapshot["body_x"].shape[0]
            if body_scripted is None:
                body_scripted = np.zeros(body_count, dtype=np.uint8)
            if body_a is None:
                body_a = np.zeros(body_count, dtype=np.float64)
            if body_e is None:
                body_e = np.zeros(body_count, dtype=np.float64)
            if body_theta is None:
                body_theta = np.zeros(body_count, dtype=np.float64)
            if body_arg is None:
                body_arg = np.zeros(body_count, dtype=np.float64)
            if body_parent is None:
                body_parent = np.full(body_count, -1, dtype=np.int64)

            use_time_dependent_bodies = 1 if bool(snapshot.get("use_time_dependent_bodies", True)) else 0
            ref_index = int(snapshot.get("reference_body_index", -1))

            # Die ORTLICHE schrittdecke (siehe _rkn_adaptive_step_time_numba).
            # `max_dt_floor` ist die schrittdecke der qualitaetsstufe -- unter
            # sie darf die ortliche rechnung nie gehen, damit das nahfeld exakt
            # so teuer bleibt wie zuvor. `timescale_divisor` = 0 schaltet die
            # ganze ortliche klammer ab (der A/B-schalter fuer den bit-vergleich
            # und der zustand, in dem `rkn_adaptive_far_maxdt` aus ist).
            max_dt_floor = float(snapshot.get("rkn_max_dt_floor", max_dt))
            timescale_divisor = float(snapshot.get("rkn_max_dt_timescale_divisor", 0.0))
            if not math.isfinite(max_dt_floor) or max_dt_floor <= 0.0:
                max_dt_floor = max_dt
            if not math.isfinite(timescale_divisor) or timescale_divisor <= 0.0:
                timescale_divisor = 0.0

            out, used, rkn_stats = _compute_distance_points_rkn_numba(
                snapshot["ship_px"],
                snapshot["ship_py"],
                snapshot["ship_vx"],
                snapshot["ship_vy"],
                0,
                float(snapshot.get("ref_px", 0.0)),
                float(snapshot.get("ref_py", 0.0)),
                snapshot["body_x"],
                snapshot["body_y"],
                snapshot["body_m"],
                snapshot["body_fixed"],
                body_scripted,
                body_a,
                body_e,
                body_theta,
                body_arg,
                body_parent,
                snapshot["G"],
                base_dt,
                snapshot["precision"],
                snapshot["max_points"],
                snapshot["max_iters"],
                min_dt,
                max_dt,
                rtol,
                atol_pos,
                atol_vel,
                safety,
                min_factor,
                max_factor,
                max_rejects,
                use_time_dependent_bodies,
                ref_index,
                float(snapshot.get("resume_t", 0.0)),
                float(snapshot.get("resume_accumulated", 0.0)),
                float(snapshot.get("resume_proposed_dt", 0.0)),
                1 if snapshot.get("use_body_memo", True) else 0,
                max_dt_floor,
                timescale_divisor,
            )
            # Alles aufheben, was noetig ist, um GENAU HIER weiterzurechnen.
            # Entscheidend ist, dass der SCHNAPPSCHUSS mitgehalten wird: die
            # koerper-arrays sind auf seine epoche bezogen und werden im
            # kernel analytisch fortgeschrieben. Mit einem frischeren
            # schnappschuss weiterzurechnen waere ein anderer lauf.
            self._resume_context = {
                'snapshot': snapshot,
                'base_dt': base_dt,
                'min_dt': min_dt,
                'max_dt': max_dt,
                # Die ortliche decke muss mit fortgesetzt werden, sonst rechnet
                # `_hold_extend_tail` den angehaengten schwanz nach einer
                # anderen regel als den rest der kurve -- genau die naht, die
                # der fortsetzbare kernel vermeiden soll.
                'max_dt_floor': max_dt_floor,
                'timescale_divisor': timescale_divisor,
                'rtol': rtol,
                'atol_pos': atol_pos,
                'atol_vel': atol_vel,
                'safety': safety,
                'min_factor': min_factor,
                'max_factor': max_factor,
                'max_rejects': max_rejects,
                'body_scripted': body_scripted,
                'body_a': body_a,
                'body_e': body_e,
                'body_theta': body_theta,
                'body_arg': body_arg,
                'body_parent': body_parent,
                'use_time_dependent_bodies': use_time_dependent_bodies,
                'ref_index': ref_index,
                'state': (float(rkn_stats[7]), float(rkn_stats[8]),
                          float(rkn_stats[9]), float(rkn_stats[10])),
                'accumulated': float(rkn_stats[11]),
                'proposed_dt': float(rkn_stats[12]),
                'kernel_t': float(rkn_stats[13]),
            }
        elif mode == "aspi" or mode == "aspi_rk4_fallback":
            min_dt = float(snapshot.get("aspi_min_dt", 1.0))
            max_dt = float(snapshot.get("aspi_max_dt", 120.0))
            base_dt = float(snapshot.get("dt", 60.0))
            safety_g = float(snapshot.get("aspi_safety_g", 0.05))
            safety_m = float(snapshot.get("aspi_safety_m", 0.5))
            close_acc_threshold = float(snapshot.get("aspi_close_acc_threshold", 0.02))

            if (not math.isfinite(min_dt)) or min_dt <= 0.0:
                min_dt = 1.0
            if (not math.isfinite(max_dt)) or max_dt <= 0.0:
                max_dt = 120.0
            if max_dt < min_dt:
                max_dt = min_dt
            if (not math.isfinite(base_dt)) or base_dt <= 0.0:
                base_dt = min_dt
            if (not math.isfinite(safety_g)) or safety_g <= 0.0:
                safety_g = 0.05
            if (not math.isfinite(safety_m)) or safety_m <= 0.0:
                safety_m = 0.5
            if (not math.isfinite(close_acc_threshold)) or close_acc_threshold < 0.0:
                close_acc_threshold = 0.02

            out, used = _compute_distance_points_aspi_numba(
                snapshot["ship_px"],
                snapshot["ship_py"],
                snapshot["ship_vx"],
                snapshot["ship_vy"],
                0,
                float(snapshot.get("ref_px", 0.0)),
                float(snapshot.get("ref_py", 0.0)),
                snapshot["body_x"],
                snapshot["body_y"],
                snapshot["body_m"],
                snapshot["body_fixed"],
                snapshot["G"],
                base_dt,
                snapshot["precision"],
                snapshot["max_points"],
                snapshot["max_iters"],
                min_dt,
                max_dt,
                safety_g,
                safety_m,
                close_acc_threshold,
                bool(snapshot.get("aspi_use_rk4_fallback", True)),
            )
        else:
            out, used = _compute_distance_points_numba(
                snapshot["ship_px"],
                snapshot["ship_py"],
                snapshot["ship_vx"],
                snapshot["ship_vy"],
                0,
                float(snapshot.get("ref_px", 0.0)),
                float(snapshot.get("ref_py", 0.0)),
                snapshot["body_x"],
                snapshot["body_y"],
                snapshot["body_m"],
                snapshot["body_fixed"],
                snapshot["G"],
                snapshot["dt"],
                snapshot["precision"],
                snapshot["max_points"],
                snapshot["max_iters"],
            )
        points = out[:int(used)].copy()
        computed_count = int(used)

        try:
            base_sim_time = float(snapshot.get("sim_time", 0.0)) if snapshot is not None else 0.0
        except Exception:
            base_sim_time = 0.0

        try:
            if np is not None and isinstance(points, np.ndarray) and points.shape[1] >= 3:
                points = points.copy()
                points[:, 2] = points[:, 2] + base_sim_time
            else:

                pts = []
                for p in points:
                    try:
                        pts.append((float(p[0]), float(p[1]), float(p[2]) + base_sim_time))
                    except Exception:
                        pts.append((float(p[0]), float(p[1]), base_sim_time))
                points = pts
        except Exception:
            pass

        if getattr(self, "debug_moving_sources", False):
            self._debug_predictor_energy(snapshot, points)

        return {"points": points, "snapshot": snapshot, "computed": computed_count, "rkn_stats": rkn_stats}

    def _compute_full_rolling(self, ship, world):
        start_ts = time.time()
        try:
            if self.num_points <= 0:
                self.points = _empty_points()
                self._roll_states = np.empty((0, 5), dtype=np.float64) if np is not None else []
                self.initialized = True
                return

            if self.precision <= 0.0:
                raise ValueError("Predictor precision must be > 0")

            max_points = self._get_target_point_cap()
            snapshot = self._make_snapshot(ship, world, max_points)
            base_t = float(snapshot.get("sim_time", 0.0))

            # Rolling mode keeps the existing RK4 state path for now.
            out, used = _compute_distance_points_numba_state(
                snapshot["ship_px"],
                snapshot["ship_py"],
                snapshot["ship_vx"],
                snapshot["ship_vy"],
                base_t,
                int(snapshot.get("ref_enabled", 0)),
                float(snapshot.get("ref_px", 0.0)),
                float(snapshot.get("ref_py", 0.0)),
                snapshot["body_x"],
                snapshot["body_y"],
                snapshot["body_m"],
                snapshot["body_fixed"],
                snapshot["G"],
                snapshot["dt"],
                snapshot["precision"],
                snapshot["max_points"],
                snapshot["max_iters"],
            )

            states = out[:int(used)].copy()
            # Alle fuenf spalten uebernehmen (frueher [:, :3]): die
            # geschwindigkeiten sind echte RK4-werte an den stuetzstellen und
            # taugen als tangente fuer die zeichenzeit-verfeinerung.
            new_points = states.copy() if (np is not None and isinstance(states, np.ndarray) and states.shape[0] > 0) else _empty_points()

            try:
                old_points = self.points if (np is not None and isinstance(self.points, np.ndarray)) else np.array(self.points, dtype=np.float64) if self.points is not None else None
            except Exception:
                old_points = None
            try:
                changed = int(self._count_recomputed_points(old_points, new_points))
            except Exception:
                changed = int(new_points.shape[0]) if (hasattr(new_points, 'shape')) else 0
            try:
                self._computed_since_last_update += changed
            except Exception:
                pass
            self._roll_states = states
            if np is not None and isinstance(states, np.ndarray) and states.shape[0] > 0:
                self.points = new_points.copy()
            else:
                self.points = _empty_points()
            self.initialized = True
            self._last_swapped_snapshot = snapshot
        finally:
            try:
                if self.debug:
                    dur = time.time() - start_ts
                    try:
                        rsn = self._roll_states.shape[0] if (isinstance(getattr(self, '_roll_states', None), np.ndarray)) else 'n/a'
                    except Exception:
                        rsn = 'n/a'
                    print(f"PRED_DBG_COMPUTE_FULL_ROLLING: took {dur:.3f}s roll_states={rsn}", flush=True)
            except Exception:
                pass

    def _append_rolling_tail(self, world, missing_points):
        if missing_points <= 0:
            return 0
        if np is None or not isinstance(self._roll_states, np.ndarray) or self._roll_states.shape[0] == 0:
            return 0

        tail = self._roll_states[-1]
        init_px = float(tail[0])
        init_py = float(tail[1])
        init_t = float(tail[2])
        init_vx = float(tail[3])
        init_vy = float(tail[4])

        body_x, body_y, body_m, body_fixed = self._serialize_bodies_numba(world)
        ref_enabled, ref_px, ref_py = self._resolve_reference_body(world)
        ref_enabled = 0
        max_new_points = int(missing_points) + 1  # include seed sample at index 0
        max_iters = int(max(10000, max_new_points * 100))

        # Rolling tail extension intentionally stays on the RK4 state helper.
        out, used = _compute_distance_points_numba_state(
            init_px,
            init_py,
            init_vx,
            init_vy,
            init_t,
            int(ref_enabled),
            float(ref_px),
            float(ref_py),
            body_x,
            body_y,
            body_m,
            body_fixed,
            float(world.G),
            float(self.dt),
            float(self._effective_precision()),
            max_new_points,
            max_iters,
        )

        if int(used) <= 1:
            return 0

        to_add = out[1:int(used)].copy()
        if to_add.shape[0] > missing_points:
            to_add = to_add[:missing_points]
        if to_add.shape[0] <= 0:
            return 0

        self._roll_states = np.concatenate((self._roll_states, to_add), axis=0)
        self.points = self._roll_states.copy()
        added = int(to_add.shape[0])
        try:
            self._computed_since_last_update += added
        except Exception:
            pass
        return added

    def _update_rolling(self, ship, world):
        # On first run or when zoom changed (auto precision), rebuild once.
        if (not self.initialized) or ( np is None or not isinstance(self._roll_states, np.ndarray) or self._roll_states.shape[0] == 0) or getattr(self, "_view_scale_changed", False):
            self._compute_full_rolling(ship, world)
            self._view_scale_changed = False
        else:
            removed = self.remove_passed_points(ship)

            target_points = self._get_target_point_cap()
            missing = target_points - self._points_count()
            if missing > 0:
                self._append_rolling_tail(world, missing)

        self._anchor_first_point(ship, world)
        if np is not None and isinstance(self._roll_states, np.ndarray) and self._roll_states.shape[0] > 0:
            self._roll_states[0, 0] = float(ship.position.x)
            self._roll_states[0, 1] = float(ship.position.y)
            try:
                self._roll_states[0, 2] = float(world.time)
            except Exception:
                pass
            self._roll_states[0, 3] = float(ship.velocity.x)
            self._roll_states[0, 4] = float(ship.velocity.y)

    def _async_jobs_in_flight(self):
        """Wie viele auftraege rechnen gerade?

        `_pending_futures` enthaelt auch bereits FERTIGE futures, die nur noch
        nicht eingewechselt wurden -- die zaehlen hier nicht als "in arbeit".
        """
        count = 0
        pending = getattr(self, "_pending_futures", None)
        if pending:
            for _job_id, fut in list(pending):
                try:
                    if not fut.done():
                        count += 1
                except Exception:
                    count += 1
        if count == 0:
            pf = getattr(self, "_pending_future", None)
            if pf is not None and not any(pf is f for _j, f in (pending or [])):
                try:
                    if not pf.done():
                        count += 1
                except Exception:
                    count += 1
        return count

    def _async_job_in_flight(self):
        """Rechnet ueberhaupt ein auftrag? (bequemlichkeit fuer altes verhalten)"""
        return self._async_jobs_in_flight() > 0

    def _pipeline_depth_cap(self):
        """Obergrenze: konfiguration und verfuegbare kerne."""
        cap = int(max(1, getattr(self, "thrust_pipeline_depth", 1)))
        try:
            cores = int(os.cpu_count() or 2)
        except Exception:
            cores = 2
        # Einen kern fuer haupt- und darstellungs-thread frei lassen.
        return max(1, min(cap, max(1, cores - 1)))

    def _target_pipeline_depth(self):
        """So viele gleichzeitige laeufe, dass je BILD eines fertig wird.

        Die dauer einer vorhersage laesst sich nicht unter die bildzeit
        druecken -- sie haengt am horizont (17 ms bei der grundeinstellung,
        ~74 ms bei vierfachem horizont) und ein bild dauert 11 ms. Wie oft
        sich die linie erneuert, haengt aber nicht an dieser dauer, sondern am
        DURCHSATZ: bei `n` zeitversetzt gestarteten laeufen wird alle
        rechenzeit/n ein ergebnis fertig. Gebraucht werden also

            n = rechenzeit / bildzeit

        laeufe (aufgerundet, plus einer als puffer gegen schwankungen), damit
        in jedem bild genau einer ankommt. Eine feste zahl kann das nicht
        leisten: sie ist beim kurzen horizont verschwenderisch und beim langen
        zu klein -- genau das war bei vierfachem horizont noch sichtbar
        (3 laeufe / 74 ms = 40 erneuerungen je sekunde bei 90 bildern).

        Die bildzeit misst der predictor selbst am abstand seiner eigenen
        aufrufe; die rechenzeit ist der letzte messwert aus
        `_compute_from_snapshot`.
        """
        cap = self._pipeline_depth_cap()
        if cap <= 1:
            self._pipeline_depth_used = 1
            return 1

        frame_ms = float(getattr(self, "_update_interval_ms", 0.0) or 0.0)
        compute_ms = float(getattr(self, "last_compute_ms", 0.0) or 0.0)
        if frame_ms <= 0.0 or compute_ms <= 0.0:
            # Noch nichts gemessen: bescheiden anfangen, nicht mit voller
            # breitseite -- der erste messwert kommt schon im naechsten bild.
            depth = min(2, cap)
        else:
            depth = int(math.ceil(compute_ms / frame_ms)) + 1
            depth = max(1, min(cap, depth))
        self._pipeline_depth_used = depth
        return depth

    def _request_thrust_recompute(self, ship, world):
        """Schub-neuberechnung ANFORDERN statt sie im hauptthread zu erzwingen.

        Waehrend eines brennmanoevers reisst der schub die geschwindigkeit in
        JEDEM frame ueber die toleranz. Der alte weg hat daraufhin jedes mal
        `_compute_full` synchron laufen lassen: gemessen mit dem vollen
        sonnensystem **0.12 ms im gleitflug gegen 59 ms unter schub**, also
        ~14 fps, solange die pfeiltaste gedrueckt ist. Ausserdem wurde die
        laufende asynchrone rechnung jedes mal verworfen und die linie
        geleert -- unter dauerschub kam also nie ein ergebnis durch.

        Statt dessen wird die anforderung ZUSAMMENGEFASST: laeuft schon ein
        auftrag, passiert nichts (er ist ohnehin schon aktueller als die
        gezeichnete linie); laeuft keiner, wird genau einer abgeschickt. Die
        alte linie bleibt sichtbar und wird wie immer per
        `_anchor_first_point` ans schiff geheftet, bis das neue ergebnis da
        ist. Damit erneuert sich die vorhersage waehrend des brennens etwa
        alle 60 ms (statt gar nicht) und der hauptthread bleibt frei.

        Rueckgabe: True = zusammengefasst, der aufrufer laesst die vorhandene
        linie stehen. False = der aufrufer muss den alten, harten weg gehen
        (kein async, rolling-modus, oder es gibt gar keine linie, die man
        behalten koennte -- dann gilt weiterhin die zusicherung, dass
        update() synchron eine baut).
        """
        if world is None or ship is None:
            return False
        if self.rolling_mode or not self.async_compute:
            return False
        if self.num_points <= 0:
            return False
        try:
            if self._points_count() <= 0:
                return False
        except Exception:
            return False

        # Weil je bild hoechstens einer dazukommt, starten die laeufe
        # automatisch um eine bildzeit versetzt -- und liefern deshalb auch um
        # eine bildzeit versetzt ab, statt gebuendelt.
        depth = self._target_pipeline_depth()
        if self._async_jobs_in_flight() < depth:
            try:
                self._submit_async_compute(
                    ship, world, self._get_target_point_cap(), max_in_flight=depth,
                )
            except Exception:
                return False
        return True

    def _request_hold_recompute(self, ship, world):
        """Neue horizont-/abstands-kurve ANFORDERN, ohne den halt aufzugeben.

        Dasselbe muster wie `_request_thrust_recompute`, fuer den anderen
        ausloeser: den WECHSEL DER ZEITRAFFER-STUFE. Die stufe bestimmt ueber
        `predictor_warp_length_mult()` den horizont (1x/4x/16x/64x ab 7d/s),
        jeder wechsel ruft `set_length()`, und der halt-zweig in `update()`
        hat das bisher mit einem synchronen `_compute_full` beantwortet.
        Gemessen mit dem vollen sonnensystem bei 180 fps, gegen 0.3-0.5 ms in
        den nachbar-frames:

            7d/s -> 30d/s     47.6 ms      1y/s  -> 100d/s   30.6 ms
            30d/s -> 100d/s   31.1 ms      100d/s -> 30d/s   48.2 ms
            100d/s -> 1y/s    40.6 ms      30d/s  -> 7d/s    14.9 ms

        Das ist der ruckler beim umschalten -- und er ist unnoetig, denn die
        gehaltene kurve ist zu diesem zeitpunkt nicht falsch. Sie ist bloss
        zu kurz (hoch) oder zu lang (runter). Zu kurz heisst nur, dass sie
        frueher nachgerechnet werden muss; bis dahin zeigt sie dieselbe bahn.
        Zu lang heisst gar nichts -- `set_display_length()` zeichnet ohnehin
        nur den un-geraffen anteil.

        Also: genau EINEN auftrag abschicken und den halt normal weiterlaufen
        lassen. `update()` wechselt das ergebnis ein, sobald es da ist (siehe
        `_hold_pending_swap`). Ein zweiter auftrag brauchte es nicht -- unter
        dem halt aendert sich der schiffszustand nicht sprunghaft, ein
        laufender auftrag ist also immer schon der richtige.

        Rueckgabe: True = angefordert, der aufrufer haelt weiter. False = der
        aufrufer muss den harten (synchronen) weg gehen.
        """
        if world is None or ship is None:
            return False
        if self.rolling_mode or not self.async_compute:
            return False
        if self.num_points <= 0:
            return False
        # Ohne kurve gibt es nichts zu halten -- dann gilt weiterhin die
        # zusicherung, dass update() synchron eine baut.
        try:
            if self._points_count() <= 0:
                return False
        except Exception:
            return False
        # WICHTIG: den fortsetzungs-zustand der ALTEN kurve festhalten, nicht
        # wegwerfen. Ohne ihn kann `_hold_extend_tail` waehrend der wartezeit
        # nicht mehr nachlegen, die gehaltene kurve wird also nur noch von
        # vorn verbraucht -- gemessen 10 000 -> 6 075 punkte ueber 16 frames
        # beim wechsel 7d/s -> 30d/s, also eine um 39 % kuerzere linie, die
        # beim einwechseln zurueckspringt. Genau dieses pulsieren beseitigt
        # `_hold_extend_tail` ja.
        #
        # Er wird GESONDERT gehalten, weil der worker `self._resume_context`
        # schon beim fertigwerden ueberschreibt -- also ein bis zwei frames
        # bevor das ergebnis eingewechselt ist. Mit dem waere der schwanz mit
        # dem NEUEN punktabstand angesetzt worden, waehrend der rest noch den
        # alten traegt; ein solcher sprung im abstand macht sowohl den
        # index-anteil in `_display_point_count` als auch die mindest-sehne
        # der tangente falsch (beide setzen festen abstand voraus).
        self._hold_resume_context = getattr(self, '_resume_context', None)
        try:
            self._submit_async_compute(
                ship, world, self._get_target_point_cap(), max_in_flight=1,
            )
        except Exception:
            return False
        self._hold_pending_swap = True
        return True

    def _submit_async_compute(self, ship, world, max_points, max_in_flight=1):
        pending = getattr(self, "_pending_futures", [])

        if self._single_flight and max_in_flight <= 1:
            if len(pending) > 0:
                return
        elif self._async_jobs_in_flight() >= max_in_flight:
            # Gezaehlt wird, was RECHNET. `len(pending)` waere falsch: darin
            # stehen auch schon fertige, nur noch nicht eingewechselte
            # ergebnisse, und die haben keinen worker mehr belegt. Sie
            # mitzuzaehlen haette den nachschub genau in den bildern
            # blockiert, in denen gerade eines fertig geworden ist.
            return

        snapshot = self._make_snapshot(ship, world, max_points)
        self._debug_integrator_mode("submit", snapshot)

        # ensure executor exists (lazy creation)
        if getattr(self, '_executor', None) is None:
            self._ensure_executor()

        job_id = self._next_job_id
        fut = self._executor.submit(self._compute_from_snapshot, snapshot)
        if self.debug and not getattr(self, "_suppress_dbg_computed", False):
            try:
                print(
                    "PRED_DBG_SUBMIT: "
                    f"job={job_id} "
                    f"version={int(snapshot.get('trajectory_version', -1))} "
                    f"sim_time={float(snapshot.get('sim_time', 0.0)):.6f} "
                    f"vx={float(snapshot.get('ship_vx', 0.0)):.6e} "
                    f"vy={float(snapshot.get('ship_vy', 0.0)):.6e} "
                    "thread=worker",
                    flush=True,
                )
            except Exception:
                pass

        # mirror single-future state for legacy code paths
        try:
            self._pending_future = fut
            self._pending_job_id = job_id
        except Exception:
            pass

        # Ersetze Queue statt endlos anzuhängen
        if self._single_flight and max_in_flight <= 1:
            self._pending_futures = [(job_id, fut)]
        else:
            pending.append((job_id, fut))
            self._pending_futures = pending

        self._next_job_id += 1
        self._jobs_submitted += 1

    def _swap_ready_result(self, current_ship=None, current_world=None, allow_rebase=True):
        pending = getattr(self, "_pending_futures", [])

        if not pending:
            if self._pending_future is None or not self._pending_future.done():
                return False
            finished_future = self._pending_future
            finished_job_id = self._pending_job_id
            self._pending_future = None
            self._pending_job_id = 0
        else:
            finished_future = None
            finished_job_id = None

            # GLEICHMAESSIG einwechseln -- ein ergebnis je bild, das AELTESTE
            # zuerst.
            #
            # Immer das neueste zu nehmen liegt nahe (es ist ja das aktuellste),
            # macht das nachziehen aber ruckartig: die laeufe werden zwar
            # gleichmaessig gestartet, aber nicht ganz gleichmaessig fertig.
            # In einem bild wird keines fertig, im naechsten zwei -- und
            # "neuestes zuerst" macht daraus einen stillstand gefolgt von einem
            # DOPPELSCHRITT. Gemessen unter vollschub an der periapsis: der
            # sprung der kurvenform ist in so einem bild doppelt so gross wie in
            # seinen nachbarn, und das alter des gezeigten zustands faellt dabei
            # von 6 auf 4 sekunden. Rund 2 % der bilder waren betroffen, also
            # etwa jede sekunde eines -- das ist das stockende, "wie hohe
            # netzwerk-latenz" wirkende nachziehen. Eine gleichmaessig zu
            # langsame bildrate sieht man nicht, einen ausreisser alle 90 bilder
            # schon.
            #
            # Die abhilfe ist dieselbe wie bei genau diesem netzwerk-problem:
            # ein kleiner puffer, aus dem in gleichmaessigen schritten
            # entnommen wird. Schwankende ankunft wird so zu gleichmaessiger
            # ausgabe, bezahlt mit etwas mehr, aber KONSTANTER verzoegerung.
            # `swap_backlog_max` begrenzt, wie viele fertige ergebnisse warten
            # duerfen; darueber hinaus wird uebersprungen, damit die
            # verzoegerung nicht davonlaeuft.
            done_entries = []
            for idx, (jid, fut) in enumerate(pending):
                try:
                    done = fut.done()
                except Exception:
                    done = False
                if done:
                    done_entries.append((jid, idx))

            if not done_entries:
                return False

            done_entries.sort()
            backlog_max = int(max(0, getattr(self, "swap_backlog_max", 1)))
            # So weit vorspulen, dass hoechstens `backlog_max` ergebnisse
            # zurueckbleiben -- im normalfall ist das 0 und es wird schlicht
            # das aelteste genommen.
            skip = max(0, len(done_entries) - 1 - backlog_max)
            finished_job_id, newest_idx = done_entries[skip]
            finished_future = pending[newest_idx][1]

            keep = []
            for idx, entry in enumerate(pending):
                if idx == newest_idx:
                    continue
                jid, fut = entry
                try:
                    done = fut.done()
                except Exception:
                    done = False
                if done and jid < finished_job_id:
                    continue
                keep.append(entry)
            pending[:] = keep

            # Ein ergebnis, das AELTER ist als die gezeichnete linie, darf sie
            # nicht ersetzen -- sonst laeuft die vorhersage rueckwaerts.
            try:
                last_swapped = int(getattr(self, "_last_swapped_job_id", -1))
            except Exception:
                last_swapped = -1
            if finished_job_id is not None and finished_job_id < last_swapped:
                return False

        try:
            result = finished_future.result()


            if isinstance(result, dict):
                points = result.get("points")
                snapshot = result.get("snapshot")
                rkn_stats = result.get("rkn_stats")
            else:
                points = result
                snapshot = None
                rkn_stats = None

            if points is None:
                return False

            if snapshot is not None:
                try:
                    snapshot_version = int(snapshot.get("trajectory_version", -1))
                except Exception:
                    snapshot_version = -1
                current_version = int(self._trajectory_version)
                if snapshot_version != current_version:
                    self._log_snapshot_result(False, "trajectory_version", snapshot, None, None, float("nan"), float("nan"))
                    return False

            if snapshot is not None and current_ship is not None:
                svx = float(snapshot.get("ship_vx", 0.0))
                svy = float(snapshot.get("ship_vy", 0.0))
                cur_vx = float(current_ship.velocity.x)
                cur_vy = float(current_ship.velocity.y)

                dvx = cur_vx - svx
                dvy = cur_vy - svy
                delta_speed = math.hypot(dvx, dvy)
                cur_speed = math.hypot(cur_vx, cur_vy)
                allowed_speed = self._allowed_velocity_delta(cur_speed)


                spx = float(snapshot.get("ship_px", 0.0))
                spy = float(snapshot.get("ship_py", 0.0))
                cur_px = float(current_ship.position.x)
                cur_py = float(current_ship.position.y)
                pos_delta = math.hypot(cur_px - spx, cur_py - spy)

                sim_age = None
                snap_sim_time = None
                cur_sim_time = None
                if current_world is not None:
                    try:
                        snap_sim_time = float(snapshot.get("sim_time", 0.0))
                        cur_sim_time = float(current_world.time)
                        sim_age = cur_sim_time - snap_sim_time
                    except Exception:
                        sim_age = None

                allowed_pos = float(self.snapshot_position_abs_tol)


                # Veraltet ist ein ergebnis erst, wenn der zoom die WIRKSAME
                # punktdichte veraendert hat -- der rohe view-scale-vergleich
                # verwarf ergebnisse auch dann, wenn die dichte durch
                # _horizon_spacing_floor() ohnehin festgeklemmt ist und die
                # linie identisch waere (siehe set_view_scale).
                is_stale_view = False
                try:
                    snap_eff = snapshot.get("eff_precision", None)
                    if snap_eff is not None:
                        cur_eff = float(self._effective_precision())
                        rel_eff = abs(float(snap_eff) - cur_eff) / max(abs(cur_eff), 1e-30)
                        if rel_eff > float(self.snapshot_view_rel_tol):
                            is_stale_view = True
                    else:
                        snap_view = snapshot.get("view_scale", None)
                        if snap_view is not None and self._view_scale is not None:
                            rel_view = abs(snap_view - self._view_scale) / max(abs(self._view_scale), 1e-30)
                            if rel_view > float(self.snapshot_view_rel_tol):
                                is_stale_view = True
                except Exception:
                    is_stale_view = False

                current_ref_index = self._current_reference_body_index()
                try:
                    snapshot_ref_index = int(snapshot.get("reference_body_index", -1))
                except Exception:
                    snapshot_ref_index = -1
                is_stale_reference = snapshot_ref_index != current_ref_index
                wall_age = 0.0
                try:
                    wall_age = max(0.0, time.time() - float(snapshot.get("submit_ts", time.time())))
                except Exception:
                    wall_age = 0.0
                max_wall_age = float(getattr(self, "max_async_wall_age", 1.5))

                # Freshness is gated by WALL age (seconds since the worker
                # finished) — sim-time age scales with sim_dt and horizon and
                # wrongly rejected every result, forcing the blocking sync path.
                # Thrust since the snapshot is already caught by the
                # trajectory_version check above; zoom / frame changes by the
                # view / reference checks. The per-frame anchor + whole-curve
                # rebase correct for the ship's motion during compute, so any
                # wall-fresh, version/view/reference-matching result is safe.
                is_stale_wall_age = wall_age > max_wall_age

                reject_reason = None
                if is_stale_view:
                    reject_reason = "view_scale"
                elif is_stale_reference:
                    reject_reason = "reference_frame"
                elif is_stale_wall_age:
                    reject_reason = "wall_age"

                if reject_reason is not None:
                    self._log_snapshot_result(False, reject_reason, snapshot, cur_sim_time, sim_age, pos_delta, delta_speed)

                    if (
                        reject_reason == "wall_age"
                        and self.force_sync_on_stale
                        and allow_rebase
                        and current_world is not None
                    ):
                        self._compute_full(current_ship, current_world)
                        self._last_swapped_job_id = finished_job_id
                        self._jobs_swapped += 1
                        self._log_snapshot_result(True, "force_sync_on_stale", snapshot, cur_sim_time, sim_age, pos_delta, delta_speed)
                        return True
                    return False

                # Rebase the whole curve to the current ship position (corrects
                # for motion during compute). The per-frame anchor in update()
                # keeps the start glued to the ship between swaps.
                #
                # UNTER DEM HALT NICHT. Dort ist genau diese starre
                # verschiebung der fehler, den der halt beseitigt: bei 30 d/s
                # rueckt das schiff waehrend der rechnung um ~1.3 tage bahn
                # vor, und die kurve um diesen sehnen-vektor quer zu schieben
                # legt sie neben die bahn. Richtig ist, sie in absoluter lage
                # UND zeit stehen zu lassen -- `_hold_advance` wirft danach
                # die punkte weg, deren zeit vergangen ist, und stellt dem
                # rest das schiff als kopf voran. Das ist dieselbe mechanik,
                # die den halt ueberhaupt traegt.
                needs_rebase = (allow_rebase and pos_delta > 1e-9
                                and math.isfinite(pos_delta))
                if needs_rebase:
                    points = self._rebase_points_to_current_snapshot(points, snapshot, current_ship)
                    self._log_snapshot_result(True, "rebased", snapshot, cur_sim_time, sim_age, pos_delta, delta_speed)
                else:
                    self._log_snapshot_result(True, "matched", snapshot, cur_sim_time, sim_age, pos_delta, delta_speed)


            try:
                old_points = self.points if (np is not None and isinstance(self.points, np.ndarray)) else np.array(self.points, dtype=np.float64) if self.points is not None else None
            except Exception:
                old_points = None

         
            try:
                changed = int(self._count_recomputed_points(old_points, points))
            except Exception:
           
                changed = None
                if isinstance(result, dict):
                    changed = result.get('computed', None)
                if changed is None:
                    try:
                        changed = int(points.shape[0]) if (np is not None and hasattr(points, 'shape')) else int(len(points))
                    except Exception:
                        changed = 0
            try:
                self._computed_since_last_update += int(changed)
            except Exception:
                pass

            self.points = points
            # Frisch gerechnet: die zeitspalte ist wieder exakt auf
            # `snapshot["sim_time"]` bezogen, und points[0] ist die echte
            # stuetzstelle des laufs, kein selbst vorangestellter kopf.
            self._points_time_offset = 0.0
            self._synthetic_head = False
            # NEUE kurve -> abgeleitete zwischenergebnisse (apsis-marker) sind
            # nicht bloss verschoben, sondern gehoeren zu einer anderen
            # geometrie. Ohne das reichte der weiche weg im halt bis zu
            # `apsis_hold_rescan_s` lang die marker der ALTEN kurve weiter --
            # gemessen ein Pe/Ap-marker mit r = 3.71e7 m, waehrend das schiff
            # bei 3.79e7 m stand und der abstand noch stieg. Auf dem schirm
            # ist das die fahne, die fuer einen frame beim schiff auftaucht
            # und wieder verschwindet.
            self._invalidate_derived_caches()
            self.initialized = True
            self._last_swapped_job_id = finished_job_id
            self._jobs_swapped += 1
            self._last_swapped_snapshot = snapshot
            self._apply_rkn_stats(rkn_stats)
            if self.debug:
                try:
                    cnt = points.shape[0] if (np is not None and hasattr(points, "shape")) else len(points)
                except Exception:
                    cnt = 0
                if snapshot is not None:
                    svx = float(snapshot.get("ship_vx", 0.0))
                    svy = float(snapshot.get("ship_vy", 0.0))
                    stime = snapshot.get("time", 0.0)
            return True
        except Exception as exc:
            return False

    def _get_target_point_cap(self):

        if self.num_points <= 0:
            return 0

        if self.length is None:
            return self.num_points

        # Cap the point count by the target horizon using the SAME (effective)
        # spacing the kernel samples at, so the traced arc = max_points *
        # eff_precision = length, independent of `precision`. This decouples the
        # look-ahead horizon (`length`, the thing that costs) from point spacing
        # (`precision`, purely cosmetic). num_points stays the safety ceiling.
        spacing_for_cap = self._effective_precision()
        if not (spacing_for_cap > 0.0):
            spacing_for_cap = self.base_precision if self.base_precision > 0.0 else self.precision
        max_by_length = max(1, int(self.length / spacing_for_cap) + 1)
        return min(self.num_points, max_by_length)

    def _compute_full(self, ship, world):
        if self.rolling_mode:
            self._compute_full_rolling(ship, world)
            return

        if self.num_points <= 0:
            self.points = _empty_points()
            self.initialized = True
            return

        if self.precision <= 0.0:
            raise ValueError("Predictor precision must be > 0")

        max_points = self._get_target_point_cap()

        snapshot = self._make_snapshot(ship, world, max_points)

        try:
            old_points = self.points if (np is not None and isinstance(self.points, np.ndarray)) else np.array(self.points, dtype=np.float64) if self.points is not None else None
        except Exception:
            old_points = None

        result = self._compute_from_snapshot(snapshot)
        if isinstance(result, dict):
            new_points = result["points"]
            self.points = new_points
            self._last_swapped_snapshot = result.get("snapshot")
            self._apply_rkn_stats(result.get("rkn_stats"))
        else:
            new_points = result
            self.points = new_points

        # Siehe _swap_ready_result: neue kurve, neue marker -- und eine
        # zeitspalte, die wieder auf ihrem eigenen schnappschuss sitzt.
        self._points_time_offset = 0.0
        self._synthetic_head = False
        self._invalidate_derived_caches()
        self.initialized = True
 
        try:
            changed = int(self._count_recomputed_points(old_points, new_points))
        except Exception:
  
            changed = None
            if isinstance(result, dict):
                changed = result.get('computed', None)
            if changed is None:
                try:
                    changed = int(self.points.shape[0]) if (np is not None and hasattr(self.points, 'shape')) else int(len(self.points))
                except Exception:
                    changed = 0
        try:
            self._computed_since_last_update += int(changed)
        except Exception:
            pass

    def initialize(self, ship, world):
        self.reset()
        if self.rolling_mode:
            self._compute_full_rolling(ship, world)
            self._anchor_first_point(ship, world)
            if np is not None and isinstance(self._roll_states, np.ndarray) and self._roll_states.shape[0] > 0:
                self._roll_states[0, 0] = float(ship.position.x)
                self._roll_states[0, 1] = float(ship.position.y)
                try:
                    self._roll_states[0, 2] = float(world.time)
                except Exception:
                    pass
                self._roll_states[0, 3] = float(ship.velocity.x)
                self._roll_states[0, 4] = float(ship.velocity.y)
            return
        self._compute_full(ship, world)
        self._anchor_first_point(ship, world)

    def update(self, ship, world):
        # Bildzeit mitschreiben: update() laeuft genau einmal je bild, der
        # abstand zweier aufrufe IST also die bildzeit. Sie bestimmt, wie
        # viele vorhersagen gleichzeitig laufen muessen, damit je bild eine
        # fertig wird (_target_pipeline_depth). Gleitender mittelwert, weil
        # einzelne bilder stark schwanken; ausreisser (fenster verschoben,
        # pause) werden verworfen.
        try:
            now_ts = time.perf_counter()
            last_ts = self._last_update_ts
            self._last_update_ts = now_ts
            if last_ts is not None:
                gap_ms = (now_ts - last_ts) * 1000.0
                if 0.05 <= gap_ms <= 250.0:
                    prev = float(self._update_interval_ms or 0.0)
                    self._update_interval_ms = gap_ms if prev <= 0.0 else (prev * 0.9 + gap_ms * 0.1)
        except Exception:
            pass

        try:
            self._computed_since_last_update = 0
        except Exception:
            pass

        if self.num_points <= 0:
            self.reset()
            if self.debug and not getattr(self, "_suppress_dbg_computed", False):
                try:
                    print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                except Exception:
                    pass
            self._computed_since_last_update = 0
            return

        if self.precision <= 0.0:
            raise ValueError("Predictor precision must be > 0")

        self._warn_rolling_rkn_once()
        if self._handle_trajectory_branch_change(ship, world):
            if self.debug and not getattr(self, "_suppress_dbg_computed", False):
                try:
                    print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                except Exception:
                    pass
            self._computed_since_last_update = 0
            return

        if self.rolling_mode:
            # Detect sudden ship velocity changes (thrust) even in rolling
            # mode by tracking the last observed ship velocity. If a large
            # delta is detected, rebuild the full rolling state so stored
            # points don't remain stale.
            if ship is not None:
                cur_vx = float(ship.velocity.x)
                cur_vy = float(ship.velocity.y)
                last_vx = getattr(self, '_last_ship_vx', None)
                last_vy = getattr(self, '_last_ship_vy', None)
                if last_vx is not None and last_vy is not None:
                    dvx = cur_vx - float(last_vx)
                    dvy = cur_vy - float(last_vy)
                    delta_speed = math.hypot(dvx, dvy)
                    cur_speed = math.hypot(cur_vx, cur_vy)
                    allowed_speed = max(self.snapshot_velocity_abs_tol, self.snapshot_velocity_rel_tol * max(cur_speed, 1.0))
                    if delta_speed >= allowed_speed:
                        if self.debug:
                            try:
                                print(f"PRED_DBG_VEL_CHANGE: dv={delta_speed:.6e} allowed={allowed_speed:.6e}", flush=True)
                            except Exception:
                                pass
                        # Rebuild entire rolling prediction synchronously.
                        self._compute_full_rolling(ship, world)
                        self._anchor_first_point(ship, world)
                        # Update remembered velocity and report
                        self._last_ship_vx = cur_vx
                        self._last_ship_vy = cur_vy
                        if self.debug and not getattr(self, "_suppress_dbg_computed", False):
                            try:
                                print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                            except Exception:
                                pass
                        self._computed_since_last_update = 0
                        return
                # remember velocity for next update
                self._last_ship_vx = cur_vx
                self._last_ship_vy = cur_vy

            # instrumentation: compact frame summary (throttled) and timed update_rolling
            try:
                self._frame_dbg_counter += 1
                rs = getattr(self, "_roll_states", None)
                try:
                    rsn = rs.shape[0] if (rs is not None and hasattr(rs, 'shape')) else (len(rs) if rs is not None else 'n/a')
                except Exception:
                    rsn = 'n/a'
                view_changed = getattr(self,'_view_scale_changed',False)
                if view_changed or (self._frame_dbg_counter % max(1, self._frame_dbg_freq) == 0):
                    try:
                        print(f"PRED_DBG_FRAME: rolling_mode={self.rolling_mode} num_points={self.num_points} initialized={self.initialized} roll_states={rsn} view_changed={view_changed}", flush=True)
                    except Exception:
                        pass
            except Exception:
                pass
            t0 = time.time()
            self._update_rolling(ship, world)
            t1 = time.time()
            dur = t1 - t0
            if self.debug and dur >= getattr(self, '_update_rolling_warn_threshold', 0.0):
                try:
                    print(f"PRED_DBG_UPDATE_ROLLING: took {dur:.6f}s", flush=True)
                except Exception:
                    pass
            self._computed_since_last_update = 0
            return

        # ------------------------------------------------ zeitraffer-halt
        # Der halt uebernimmt den frame VOLLSTAENDIG -- er laeuft vor beiden
        # rechenwegen und kehrt in jedem fall zurueck. Das ist absicht: der
        # asynchrone weg wuerde sonst weiterhin jeden frame ein mehrere
        # frames altes ergebnis einwechseln und `_anchor_first_point`
        # darauf loslassen, und genau diese starre verschiebung einer
        # veralteten kurve ist das zittern, das der halt beseitigen soll.
        #
        # Aufgefrischt wird EINMAL SYNCHRON, wenn der vorrat zur neige geht
        # (siehe _hold_advance). Das kostet ~6 ms und faellt bei 7d/s etwa
        # alle 40 frames an -- deterministisch, statt jeden frame ein
        # bisschen.
        if self._hold_active():
            # Ein angeforderter stufenwechsel wird eingewechselt, sobald er da
            # ist -- OHNE starre verschiebung (siehe _request_hold_recompute
            # und der `allow_rebase`-zweig in _swap_ready_result). Das ist die
            # einzige stelle, an der der halt ein asynchrones ergebnis
            # uebernimmt: nicht jeden frame, sondern genau einmal je wechsel.
            # Jeden frame einzuwechseln waere wieder das zittern, das der halt
            # beseitigt.
            if getattr(self, '_hold_pending_swap', False):
                if self._swap_ready_result(ship, world, allow_rebase=False):
                    self._hold_pending_swap = False
                    self._hold_resume_context = None
                    # Die neue kurve traegt keinen selbst vorangestellten kopf.
                    self._synthetic_head = False
                elif not self._async_jobs_in_flight():
                    # Verworfen (version/zoom/rahmen) und nichts mehr
                    # unterwegs: nicht ewig warten, sondern den harten weg
                    # wieder zulassen.
                    self._hold_pending_swap = False
                    self._hold_resume_context = None
                    self._hold_invalidated = True
                    self._hold_soft_invalidated = False
            if not self._hold_advance(ship, world):
                self._cancel_pending_job()
                self._hold_pending_swap = False
                self._hold_resume_context = None
                self._compute_full(ship, world)
                self._anchor_first_point(ship, world)
                self._view_scale_changed = False
            if self.debug and not getattr(self, "_suppress_dbg_computed", False):
                try:
                    print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update} (hold)")
                except Exception:
                    pass
            self._computed_since_last_update = 0
            return

        if not self.async_compute:
            if not self.initialized:
                self.initialize(ship, world)
                if self.debug and not getattr(self, "_suppress_dbg_computed", False):
                    try:
                        print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                    except Exception:
                        pass
                self._computed_since_last_update = 0
                return

            if self.recompute_every_update:
                self._compute_full(ship, world)
                self._anchor_first_point(ship, world)
                # Die zoom-anforderung ist mit dem vollen neuaufbau erfuellt.
                # Nur der asynchrone weg hat das flag bisher zurueckgesetzt;
                # synchron blieb es stehen und haette den zeitraffer-halt
                # dauerhaft blockiert.
                self._view_scale_changed = False
                if self.debug and not getattr(self, "_suppress_dbg_computed", False):
                    try:
                        print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                    except Exception:
                        pass
                self._computed_since_last_update = 0
                return

            removed = self.remove_passed_points(ship)
            target_points = self._get_target_point_cap()
            if self._points_count() < target_points:
                self._compute_full(ship, world)
            self._anchor_first_point(ship, world)
            self._view_scale_changed = False
            if self.debug and not getattr(self, "_suppress_dbg_computed", False):
                try:
                    print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                except Exception:
                    pass
            self._computed_since_last_update = 0
            return


        if getattr(self, '_view_scale_changed', False):
            if ship is not None and world is not None:
                self._cancel_pending_job()
                self._compute_full(ship, world)
                self._anchor_first_point(ship, world)
                self._view_scale_changed = False
                if self.debug and not getattr(self, "_suppress_dbg_computed", False):
                    try:
                        print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                    except Exception:
                        pass
                self._computed_since_last_update = 0
                return

        # Detect large ship state changes (e.g. player thrust) and force
        # a recompute so stored predictor points don't remain stale.
        try:
            if (not self.recompute_every_update) and ship is not None and self._last_swapped_snapshot is not None:
                svx = float(self._last_swapped_snapshot.get("ship_vx", 0.0))
                svy = float(self._last_swapped_snapshot.get("ship_vy", 0.0))
                cur_vx = float(ship.velocity.x)
                cur_vy = float(ship.velocity.y)

                dvx = cur_vx - svx
                dvy = cur_vy - svy
                delta_speed = math.hypot(dvx, dvy)
                cur_speed = math.hypot(cur_vx, cur_vy)
                allowed_speed = max(self.snapshot_velocity_abs_tol, self.snapshot_velocity_rel_tol * max(cur_speed, 1.0))

                # Dieselbe zusammenfassung wie in
                # _handle_trajectory_branch_change, nur gegen den zuletzt
                # EINGEWECHSELTEN zustand gemessen. Dieser melder ist der,
                # der die linie nach dem brennschluss wieder exakt macht:
                # solange die gezeichnete kurve noch zum vor-schub-zustand
                # gehoert, fordert er weiter nach, bis ein passendes ergebnis
                # eingewechselt ist -- dann liegt dv wieder in der toleranz
                # und er verstummt von selbst.
                if delta_speed >= allowed_speed and self._request_thrust_recompute(ship, world):
                    pass
                elif delta_speed >= allowed_speed:
                    old_version = int(self._trajectory_version)
                    self._trajectory_version = old_version + 1
                    if self.debug:
                        try:
                            print(
                                "PRED_DBG_TRAJECTORY_INVALIDATED: "
                                f"reason=velocity dv={delta_speed:.6e} allowed={allowed_speed:.6e} "
                                f"old_version={old_version} new_version={self._trajectory_version}",
                                flush=True,
                            )
                        except Exception:
                            pass

                    # Cancel pending work and either recompute synchronously
                    # (rolling mode / non-async) or submit a fresh async job.
                    try:
                        self._cancel_pending_job()
                    except Exception:
                        pass
                    self._clear_prediction_points()
                    self._remember_ship_state(ship, world)

                    if self.rolling_mode:
                        self._compute_full_rolling(ship, world)
                        self._anchor_first_point(ship, world)
                        if self.debug and not getattr(self, "_suppress_dbg_computed", False):
                            try:
                                print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                            except Exception:
                                pass
                        self._computed_since_last_update = 0
                        return

                    target_points = self._get_target_point_cap()
                    if self.async_compute:
                        try:
                            self._submit_async_compute(ship, world, target_points)
                        except Exception:
                            pass
                        if self.debug and not getattr(self, "_suppress_dbg_computed", False):
                            try:
                                print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                            except Exception:
                                pass
                        self._computed_since_last_update = 0
                        return
                    else:
                        self._compute_full(ship, world)
                        self._anchor_first_point(ship, world)
                        if self.debug and not getattr(self, "_suppress_dbg_computed", False):
                            try:
                                print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                            except Exception:
                                pass
                        self._computed_since_last_update = 0
                        return
        except Exception:
            pass

        # OHNE STARRE VERSCHIEBUNG -- genau wie unter dem halt. Der lauf ist
        # von einem zustand ausgegangen, der beim eintreffen ein paar frames
        # alt ist; die kurve deswegen quer zu schieben legt sie neben die
        # bahn (siehe _anchor_first_point). Richtig ist, sie in absoluter
        # lage und zeit stehen zu lassen -- `_anchor_first_point` wirft
        # gleich darauf die punkte weg, deren zeit vergangen ist, und stellt
        # dem rest das schiff als kopf voran.
        swapped = self._swap_ready_result(ship, world, allow_rebase=False)
        target_points = self._get_target_point_cap()

        if not self.initialized:
            self._submit_async_compute(ship, world, target_points)
            if self.debug and not getattr(self, "_suppress_dbg_computed", False):
                try:
                    print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                except Exception:
                    pass
            self._computed_since_last_update = 0
            return

        if not self.recompute_every_update:
            self.remove_passed_points(ship)

        # Submit a fresh background job, but throttle redundant re-submissions to
        # ~async_submit_min_interval so a cheap compute doesn't rerun 60x/s.
        # Always resubmit immediately when a result was just consumed (swapped)
        # or the line is short of its target length. Heavier computes are
        # additionally self-throttled by single-flight (submit skips if pending).
        now = time.perf_counter()
        need_more_points = self._points_count() < target_points
        throttle_ready = (now - self._last_submit_wall) >= self.async_submit_min_interval
        if need_more_points or swapped or (self.recompute_every_update and throttle_ready):
            self._submit_async_compute(ship, world, target_points)
            self._last_submit_wall = now

        # Keep the drawn line's start glued to the ship every frame (cheap rigid
        # shift of the whole curve). Between background refreshes the curve then
        # tracks the ship smoothly instead of lagging and snapping on each swap;
        # the shape itself refreshes at the worker's cadence.
        if self._points_count() > 0:
            self._anchor_first_point(ship, world)

        if self.debug and not getattr(self, "_suppress_dbg_computed", False):
            try:
                print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
            except Exception:
                pass
        self._computed_since_last_update = 0

    def set_display_length(self, metres):
        """Wie viel von der gerechneten kurve GEZEICHNET wird, in metern.

        None oder >= der gerechneten laenge heisst: alles. Betrifft nur, was
        get_points() herausgibt -- der halt, das anstueckeln und die
        fortsetzung arbeiten unveraendert auf der vollen kurve (self.points).
        """
        if metres is None:
            self.display_length = None
        else:
            value = float(metres)
            self.display_length = value if value > 0.0 else None
        # KEIN _clear_display_view() hier: der regler ruft diese methode jeden
        # frame, und ein harter cache-reset erzwaenge trotz gerundetem count
        # (siehe _display_point_count) je frame eine neue view. get_points()
        # verwirft die view schon selbst, sobald sich der gerundete count,
        # das zugrundeliegende array oder der zeichne-alles-fall aendert.

    def _clear_display_view(self):
        self._display_view = None
        self._display_view_base = None
        self._display_view_limit = -1

    def _display_point_count(self):
        """Zahl der zu zeichnenden fuehrenden punkte, oder None fuer alle.

        Ueber den INDEXANTEIL, nicht ueber die aufsummierte bogenlaenge: der
        kernel legt seine stuetzstellen auf festen abstand, der anteil ist
        also der laengenanteil -- und er kostet O(1) statt O(n) je frame.

        Der abstand wird an den GESPEICHERTEN punkten gemessen, nicht aus
        `self.length` erschlossen. Das ist derselbe O(1)-griff, aber er fragt
        die kurve, die wirklich da liegt, statt die zuletzt ANGEFORDERTE
        laenge fuer sie zu halten. Beim zeitraffer-stufenwechsel fallen die
        beiden fuer ein paar frames auseinander: `set_length()` stellt schon
        auf 4x, waehrend der halt noch die alte 1x-kurve zeigt (die neue
        entsteht im hintergrund, siehe _request_hold_recompute). Mit
        `self.length` als bezug waeren davon 1/4 gezeichnet worden -- die
        linie waere auf 25 % zusammengefallen und beim einwechseln wieder
        aufgesprungen. Gemessen: 25.0 % / 24.9 % / 24.8 % auf den drei
        aufwaerts-wechseln.

        Gemessen wird in der MITTE der kurve: points[0] ist im halt das
        schiff selbst und traegt ein absichtlich verkuerztes erstes
        teilstueck (siehe _hold_advance), waere als massstab also zu klein.
        """
        limit = self.display_length
        if limit is None:
            return None
        pts = self.points
        if np is None or not isinstance(pts, np.ndarray):
            return None
        n = int(pts.shape[0])
        if n <= 2:
            return None

        mid = n // 2
        spacing = math.hypot(float(pts[mid, 0]) - float(pts[mid - 1, 0]),
                             float(pts[mid, 1]) - float(pts[mid - 1, 1]))
        if spacing > 0.0 and math.isfinite(spacing):
            if limit >= spacing * (n - 1):
                return None
            count = int(math.ceil(limit / spacing)) + 1
            q = max(1, int(getattr(self, '_display_quantum', 8)))
            count = int(round(count / q)) * q
            return max(2, min(n, count))

        # Entartete kurve (stillstand, NaN) -- alter weg als rueckfall.
        total = self.length
        if total is None or total <= 0.0:
            total = float(self.num_points) * float(self.precision)
        if total <= 0.0 or limit >= total:
            return None
        count = int(math.ceil(n * (limit / total)))
        q = max(1, int(getattr(self, '_display_quantum', 8)))
        count = int(round(count / q)) * q
        return max(2, min(n, count))

    def get_points(self):
        """Die zu ZEICHNENDE kurve (siehe set_display_length).

        Der ausschnitt wird gemerkt und nur neu gebildet, wenn sich das
        zugrundeliegende array oder die zahl aendert. Das ist kein geiz:
        Renderer._make_prediction_line_cache_key nimmt `id(path_points)` in
        den schluessel auf, ein frisch erzeugter view je aufruf wuerde den
        cache also in JEDEM frame verfehlen.
        """
        count = self._display_point_count()
        if count is None:
            self._clear_display_view()
            return self.points
        # `is` statt id(): so haelt der vergleich eine referenz und kann nicht
        # auf eine wiederverwendete id hereinfallen.
        if (self._display_view_base is not self.points
                or self._display_view_limit != count
                or self._display_view is None):
            self._display_view = self.points[:count]
            self._display_view_base = self.points
            self._display_view_limit = count
        return self._display_view

    @staticmethod
    def _points_have_tangents(pts):
        """Traegt diese punkteliste brauchbare geschwindigkeits-spalten?

        Alles oder nichts, und das ist keine vereinfachung: ob die tangenten
        geschrieben werden, haengt am KERNEL, nicht am einzelnen punkt --
        die RKN-pfade schreiben sie durchgehend, die sehnen-pfade (ASPI,
        blankes RK4) durchgehend nicht. Geprueft wird mit numpy, weil der
        scan-kernel unter `fastmath` keine NaN erkennen kann (siehe
        _refine_apsis_numba).
        """
        if np is None or not isinstance(pts, np.ndarray):
            return False
        if pts.ndim != 2 or pts.shape[1] < 5 or pts.shape[0] < 2:
            return False
        return bool(np.all(np.isfinite(pts[:, 3:5])))

    def get_apsis_markers(self):
        """Apoapsis/Periapsis-Marker der aktuellen Prädiktionslinie.

        Rückgabe: ndarray (m, 5) mit spalten x, y, t_abs, kind, r —
        kind 0.0 = periapsis, 1.0 = apoapsis; r = abstand zum referenz-
        körper in metern. x/y sind absolute (baryzentrische) weltkoords
        des zugehörigen predictor-punkts, t_abs die absolute sim-zeit,
        damit der renderer zeitabhängig frame-transformieren kann.
        Leer wenn kein referenzkörper gewählt ist. Ergebnis wird über die
        identität des punkte-arrays gecacht: der O(n)-scan läuft nur wenn
        eine neue trajektorie geswappt (oder geschnitten) wurde.
        """
        if not self.apsis_markers_enabled:
            return self._empty_apsis_array()
        # Bewusst die GEZEICHNETE kurve: marker jenseits des sichtbaren
        # endes haetten keine linie unter sich.
        pts = self.get_points()
        if np is None or not isinstance(pts, np.ndarray) or pts.shape[0] < 3:
            return self._empty_apsis_array()
        snapshot = self._last_swapped_snapshot
        if snapshot is None:
            return self._empty_apsis_array()
        try:
            ref_index = int(snapshot.get("reference_body_index", -1))
        except Exception:
            ref_index = -1
        if ref_index < 0:
            return self._empty_apsis_array()

        cache_key = (id(pts), int(pts.shape[0]), ref_index)
        if cache_key == self._apsis_cache_key:
            return self._apsis_markers

        # Zeitraffer-halt: nur weich invalidiert (vorn verbraucht, hinten
        # angestueckelt, kopf angeschmiegt) -- die uebrigen punkte stehen
        # bit-identisch, also stehen auch ihre marker. Innerhalb des
        # rescan-fensters nur die abgelaufenen marker herausfiltern statt
        # alle punkte neu zu scannen; das lief sonst ZWEIMAL pro frame
        # (HUD-telemetrie und renderer sehen je ein anderes array).
        now_ts = time.perf_counter()
        if (getattr(self, '_apsis_soft_stale', False)
                and bool(getattr(self, 'hold_enabled', False))
                and isinstance(self._apsis_markers, np.ndarray)
                and int(getattr(self, '_apsis_scan_generation', -1))
                == int(getattr(self, '_points_generation', 0))
                and (now_ts - float(getattr(self, '_apsis_last_scan_ts', 0.0))
                     < float(getattr(self, 'apsis_hold_rescan_s', 0.25)))):
            markers = self._apsis_markers
            if markers.shape[0] > 0 and pts.shape[1] >= 3:
                head_t = float(pts[0, 2])
                if float(markers[:, 2].min()) < head_t:
                    markers = markers[markers[:, 2] >= head_t]
                    self._apsis_markers = markers
            self._apsis_cache_key = cache_key
            return self._apsis_markers

        try:
            markers, count = _find_apsis_markers_numba(
                pts,
                # DIE BEZUGSZEIT MUSS ZU DEN PUNKTZEITEN PASSEN, NICHT ZUR UHR.
                #
                # Der kernel rechnet `pts[i, 2] - base_sim_time` in eine
                # LOKALE zeit zurueck und propagiert damit den referenz-
                # koerper aus `body_x`/`body_y` -- und die wurden bei
                # `snapshot["sim_time"]` abgegriffen. Hat jemand die
                # zeitspalte seither verschoben (die starre nachfuehrung in
                # _anchor_first_point tut das), passt die spalte nicht mehr
                # zum schnappschuss: der koerper wird um genau diesen versatz
                # zu weit vorn gelesen, waehrend die kurve nur um die
                # SCHIFFSbewegung mitgezogen wurde. Was uebrig bleibt, ist die
                # relativbewegung -- und die schlaegt voll auf den
                # angezeigten Pe/Ap-abstand durch.
                float(snapshot.get("sim_time", 0.0))
                + float(getattr(self, "_points_time_offset", 0.0)),
                ref_index,
                snapshot["body_x"],
                snapshot["body_y"],
                snapshot["body_m"],
                snapshot["body_scripted"],
                snapshot["body_a"],
                snapshot["body_e"],
                snapshot["body_theta"],
                snapshot["body_arg"],
                snapshot["body_parent"],
                float(snapshot["G"]),
                1 if bool(snapshot.get("use_time_dependent_bodies", True)) else 0,
                int(self.apsis_max_markers),
                # Der selbst vorangestellte kopf ist die WELT-position des
                # schiffs und gehoert nicht zu dieser kurve -- siehe den
                # trend-scan im kernel. Das gilt jetzt in BEIDEN betriebsarten
                # (echtzeit wie halt), weil beide dieselbe mechanik benutzen;
                # ohne kopf steht das flag auf False und der aufruf ist
                # bit-identisch zu vorher.
                1 if bool(getattr(self, '_synthetic_head', False)) else 0,
                # OB DIE TANGENTEN-SPALTEN BRAUCHBAR SIND, WIRD HIER
                # ENTSCHIEDEN, NICHT IM KERNEL. Der marker wird auf derselben
                # kubik plaziert, die der renderer zeichnet -- die braucht die
                # geschwindigkeits-spalten. Die sehnen-kernel (ASPI, blankes
                # RK4) schreiben dort absichtlich NaN, und im kernel laesst
                # sich das nicht abfragen: er ist `fastmath=True`, und
                # gemessen liefern dort SOWOHL `math.isfinite(nan)` ALS AUCH
                # `nan == nan` den wert True. Also numpy, ausserhalb.
                1 if self._points_have_tangents(pts) else 0,
            )
            self._apsis_markers = markers[:int(count)].copy()
        except Exception as exc:
            # Dieser fang hat schon einen uebersetzungsfehler des kernels
            # verschluckt (readonly-notizblock, siehe _no_body_memo): die
            # marker verschwanden im spiel ohne jede meldung. Ein leeres
            # ergebnis ist ein voellig normaler zustand -- eine AUSNAHME ist
            # es nicht, also wird sie einmal gemeldet.
            if not getattr(self, "_apsis_scan_error_logged", False):
                self._apsis_scan_error_logged = True
                try:
                    print(f"PREDICTOR: apsis-scan fehlgeschlagen: {exc!r}", flush=True)
                except Exception:
                    pass
            self._apsis_markers = self._empty_apsis_array()
        self._apsis_cache_key = cache_key
        self._apsis_soft_stale = False
        self._apsis_last_scan_ts = now_ts
        self._apsis_scan_generation = int(getattr(self, '_points_generation', 0))
        return self._apsis_markers

    def interpolation_error_floor(self, sample_limit=256):
        """Kleinster zeichenfehler, den diese punkteliste ueberhaupt zulaesst.

        Der gezeichnete fehler zerfaellt in zwei UNABHAENGIGE anteile:

            fehler = |Hermite - wahrheit|  +  |polygon - Hermite|
                     (haengt am PUNKTABSTAND)  (haengt an der UNTERTEILUNG)

        Die zeichenzeit-unterteilung drueckt nur den zweiten term. Der erste
        ist die klassische schranke des kubischen Hermite-polynoms,
        ``h^4/384 * max|f''''|``; fuer eine bahn mit lokalem kruemmungsradius
        ``R`` und punktabstand ``c`` ist das

            floor ~ c^4 / (384 * R^3)

        Gegen numerisch integrierte kreisboegen ueber drei zehnerpotenzen von
        ``R`` und ``c`` geprueft: passt auf 0.4 % genau.

        Das ist der grund, warum die feinen sprossen der toleranz-leiter
        (1 mm, 1 cm) mit dem heutigen punktabstand NICHT erreichbar sind --
        dafuer muesste neu integriert werden, nicht feiner ausgewertet. Die
        methode gibt den wert zurueck, damit der renderer die angeforderte
        sprosse daran klemmen und den ERREICHTEN wert anzeigen kann, statt
        eine genauigkeit zu behaupten, die die linie nicht hat.

        Rueckgabe: meter, oder ``None`` wenn nicht bestimmbar.
        """
        points = self.points
        if np is None or not isinstance(points, np.ndarray):
            return None
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
            return None

        n = int(points.shape[0])
        # AUSDUENNEN DARF DEN PUNKTABSTAND NICHT VERAENDERN. Ein einfaches
        # `linspace` ueber die liste tut genau das: die drei punkte eines
        # tripels liegen dann nicht mehr eine, sondern mehrere stuetzweiten
        # auseinander -- und weil der boden mit c^4 geht, kommt ein vielfaches
        # heraus (gemessen 120 m statt 7.6 m auf derselben bahn). Gezogen
        # werden deshalb ANFANGSINDIZES, und jedes tripel bleibt benachbart.
        triples = n - 2
        if triples < 1:
            return None
        limit = int(max(1, min(int(sample_limit), triples)))
        if limit < triples:
            starts = np.unique(np.linspace(0, triples - 1, limit).astype(np.int64))
        else:
            starts = np.arange(triples, dtype=np.int64)

        p0 = points[starts, :2]
        p1 = points[starts + 1, :2]
        p2 = points[starts + 2, :2]

        a = p1 - p0
        b = p2 - p1
        la = np.hypot(a[:, 0], a[:, 1])
        lb = np.hypot(b[:, 0], b[:, 1])
        lc = np.hypot(p2[:, 0] - p0[:, 0], p2[:, 1] - p0[:, 1])

        # Umkreisradius der drei aufeinanderfolgenden punkte = lokaler
        # kruemmungsradius. Vierfache dreiecksflaeche ueber das kreuzprodukt.
        cross = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        with np.errstate(divide='ignore', invalid='ignore'):
            radius = (la * lb * lc) / (2.0 * cross)
            spacing = 0.5 * (la + lb)
            floor = (spacing ** 4) / (384.0 * radius ** 3)

        floor = floor[np.isfinite(floor)]
        if floor.size == 0:
            # Perfekt gerade linie: ein kubisches polynom trifft sie exakt.
            return 0.0
        return float(np.max(floor))

    def get_precision_factor(self):
        if self.base_precision <= 0.0:
            return 1.0
        return self.precision / self.base_precision

    def get_display_length(self):
        # The true traced horizon: length pins it, but the num_points ceiling
        # can clip it when spacing is finer than length/num_points. Report what
        # is actually integrated, not length * coarsen (which only held under
        # the old precision<->horizon coupling).
        if self.length is None:
            return None
        eff = self._effective_precision()
        if not (eff > 0.0):
            return self.length
        return min(self.length, self.num_points * eff)

    def set_precision(self, meters: float):
        meters = float(meters)
        if meters <= 0.0:
            raise ValueError("precision must be > 0")
        self.precision = meters
        # Die gehaltene kurve traegt den ALTEN punktabstand. Ohne diesen
        # vermerk schluckt der zeitraffer-halt die umstellung vollstaendig --
        # gemessen: punktzahl und richtung aendern sich um exakt 0.
        #
        # WEICH: der abstand ist kosmetisch, die kurve bleibt bis zum
        # eintreffen der neuen richtig. Kein grund, den hauptthread
        # anzuhalten (siehe invalidate_hold).
        self.invalidate_hold(soft=True)
        if self.rolling_mode:
            self.reset()
        elif self.async_compute:
            self._cancel_pending_job()

    def set_length(self, meters: float | None):
        if meters is None:
            self.length = None
            self.invalidate_hold()
            if self.rolling_mode:
                self.reset()
            elif self.async_compute:
                self._cancel_pending_job()
            return
        meters = float(meters)
        if meters <= 0.0:
            raise ValueError("length must be > 0")
        self.length = meters
        # Der horizont steuert ueber _horizon_spacing_floor() auch den
        # punktabstand -- die gehaltene kurve passt danach nicht mehr.
        #
        # WEICH: sie ist deswegen aber nicht falsch, sondern nur zu kurz bzw.
        # zu lang. Genau das ist der zeitraffer-stufenwechsel, bei dem
        # apply_predictor_horizon() den faktor 1x/4x/16x/64x umstellt -- der
        # harte weg hat dort 34-82 ms im hauptthread gekostet.
        self.invalidate_hold(soft=True)
        if self.rolling_mode:
            self.reset()
        elif self.async_compute:
            self._cancel_pending_job()

    def set_num_points(self, count: int, soft: bool = False):
        """Punktbudget setzen. `soft=True` behaelt die vorhandene kurve.

        Der harte weg (`reset()`) ist richtig, wenn das budget aus einem
        grund wechselt, der die kurve entwertet -- der `P`-umschalter etwa.

        WEICH ist er, wenn das budget nur MITWAECHST, weil der horizont sich
        geaendert hat (siehe `apply_predictor_horizon` in test.py). Die kurve,
        die dann dasteht, ist geometrisch weiterhin richtig; sie hat bloss zu
        wenige oder zu viele punkte. Genau dieselbe lage wie bei
        `set_length()` -- und dort hat der harte weg im zeitraffer 34-82 ms
        im hauptthread gekostet, weil `update()` sofort synchron neu rechnete
        (§17). Der zeitraffer-schritt verstellt den horizont bei JEDEM
        stufenwechsel, das budget also mit.
        """
        self.num_points = max(0, int(count))
        if not soft:
            self.reset()
            return
        self.invalidate_hold(soft=True)
        if self.rolling_mode:
            self.reset()
        elif self.async_compute:
            self._cancel_pending_job()

    def advance_state(self, world=None):

        if self.async_compute:
            self._swap_ready_result(None, world)

    def get_async_status(self):
        return {
            "enabled": self.async_compute,
            "pending": len(getattr(self, "_pending_futures", [])) > 0,
            "submitted_jobs": self._jobs_submitted,
            "swapped_jobs": self._jobs_swapped,
            "last_swapped_job_id": self._last_swapped_job_id,
            "effective_precision": self._effective_precision(),
            "trajectory_version": int(getattr(self, "_trajectory_version", 0)),
            "strict_snapshot_matching": bool(getattr(self, "strict_snapshot_matching", True)),
            "use_time_dependent_bodies": bool(getattr(self, "use_time_dependent_bodies", False)),
            "use_reference_acceleration_correction": bool(getattr(self, "use_reference_acceleration_correction", False)),
            "worker_threads": int(getattr(self, "_predictor_worker_threads", 1)),
            "requested_workers": int(getattr(self, "_requested_workers", 1)),
        }

    def close(self):
        self._cancel_pending_job()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def remove_passed_points(self, ship):
        # Robust removal based on projection onto path segments.
        if self._points_count() < 2:
            return 0

        sx = float(ship.position.x)
        sy = float(ship.position.y)

        # If in rolling mode and roll_states is available, operate on it
        # so that _roll_states and points remain consistent.
        try:
            if getattr(self, 'rolling_mode', False) and np is not None and isinstance(self._roll_states, np.ndarray) and self._roll_states.shape[0] > 1:
                n = int(self._roll_states.shape[0])
                coords = self._roll_states[:, :2]
                remove_count = 0
                for i in range(n - 1):
                    x0 = float(coords[i, 0]); y0 = float(coords[i, 1])
                    x1 = float(coords[i + 1, 0]); y1 = float(coords[i + 1, 1])
                    vx = x1 - x0; vy = y1 - y0
                    wx = sx - x0; wy = sy - y0
                    denom = vx * vx + vy * vy
                    if denom <= 1e-12:
                        remove_count += 1
                        continue
                    t = (wx * vx + wy * vy) / denom
                    if t >= 1.0:
                        remove_count += 1
                        continue
                    break

                remove_count = min(remove_count, max(0, n - 1))
                if remove_count > 0:
                    # Siehe unten: der vorangestellte kopf ist mit weg.
                    self._synthetic_head = False
                    try:
                        self._roll_states = self._roll_states[remove_count:]
                        if isinstance(self._roll_states, np.ndarray) and self._roll_states.shape[0] > 0:
                            self.points = self._roll_states.copy()
                        else:
                            self.points = _empty_points()
                    except Exception:
                        try:
                            self._roll_states = np.array(self._roll_states[remove_count:], dtype=np.float64)
                            self.points = np.array(self.points[remove_count:], dtype=np.float64)
                        except Exception:
                            pass
                    return int(remove_count)
                return 0
        except Exception:
            pass

        # Numpy-optimized path: iterate segments until ship projection is < 1.0
        if np is not None and isinstance(self.points, np.ndarray):
            n = int(self.points.shape[0])
            if n <= 1:
                return 0

            coords = self.points[:, :2]
            remove_count = 0
            for i in range(n - 1):
                x0 = float(coords[i, 0]); y0 = float(coords[i, 1])
                x1 = float(coords[i + 1, 0]); y1 = float(coords[i + 1, 1])
                vx = x1 - x0; vy = y1 - y0
                wx = sx - x0; wy = sy - y0
                denom = vx * vx + vy * vy
                if denom <= 1e-12:
                    remove_count += 1
                    continue
                t = (wx * vx + wy * vy) / denom
                if t >= 1.0:
                    remove_count += 1
                    continue
                break

            remove_count = min(remove_count, max(0, n - 1))
            if remove_count > 0:
                # Siehe unten: der vorangestellte kopf ist mit weg.
                self._synthetic_head = False
                try:
                    self.points = self.points[remove_count:]
                except Exception:
                    self.points = np.array(self.points[remove_count:], dtype=np.float64)
            return int(remove_count)

        # List / generic fallback: use same projection logic.
        # self.points can't be an ndarray here (the isinstance branch above
        # always returns) but Pyright doesn't narrow attribute access across
        # that control flow, so alias to a local it can narrow.
        pts = self.points
        if isinstance(pts, np.ndarray):
            return 0
        try:
            n = len(pts)
            if n <= 1:
                return 0
        except Exception:
            return 0

        remove_count = 0
        try:
            for i in range(n - 1):
                p0 = pts[i]
                p1 = pts[i + 1]
                try:
                    x0 = float(p0[0]); y0 = float(p0[1])
                    x1 = float(p1[0]); y1 = float(p1[1])
                except Exception:
                    x0 = float(getattr(p0, 'x', p0[0])); y0 = float(getattr(p0, 'y', p0[1]))
                    x1 = float(getattr(p1, 'x', p1[0])); y1 = float(getattr(p1, 'y', p1[1]))

                vx = x1 - x0; vy = y1 - y0
                wx = sx - x0; wy = sy - y0
                denom = vx * vx + vy * vy
                if denom <= 1e-12:
                    remove_count += 1
                    continue
                t = (wx * vx + wy * vy) / denom
                if t >= 1.0:
                    remove_count += 1
                    continue
                break
        except Exception:
            remove_count = 0

        remove_count = min(remove_count, max(0, n - 1))
        if remove_count > 0:
            # Der selbst vorangestellte kopf (siehe
            # _advance_points_along_curve) ist mit weggeschnitten worden --
            # sonst wuerde die naechste runde ihn ein zweites mal entfernen
            # und dabei eine ECHTE stuetzstelle verlieren, jeden frame eine.
            self._synthetic_head = False
            try:
                del pts[:remove_count]
            except Exception:
                for _ in range(remove_count):
                    try:
                        pts.pop(0)
                    except Exception:
                        break
        return int(remove_count)
