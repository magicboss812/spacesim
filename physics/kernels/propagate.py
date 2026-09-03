"""Die punktreihen: eine bahn in stuetzstellen ausrollen.

Vier varianten, weil vier fragen gestellt werden -- gleichmaessiger ABSTAND
gegen gleichmaessige ZEIT, mit und ohne bewegte quellen. Die rkn-variante ist
der normalfall; sie schreibt neben x/y auch die tangente (vx/vy), womit die
reihe eine stueckweise KUBISCHE kurve wird und der renderer sie zur zeichenzeit
beliebig fein auswerten kann, ohne dass hier ein schritt mehr faellt.

Kerne, die ihre punkte linear auf die schrittsehne setzen, schreiben dort NaN
-- kein fehlerfall, sondern die wahrheit: ein sehnenpunkt hat keine tangente,
die zu ihm passt.
"""
import math

import numpy as np
from numba import njit

from physics.kernels import BODY_MEMO_COLUMNS, POINT_COLUMNS
from physics.kernels.kepler import (
    _body_kepler_constants_numba,
    _body_position_at_time_numba,
)
from physics.kernels.integrators import (
    _compute_acc_nearest_numba,
    _compute_acc_numba,
    _compute_acc_time_numba,
    _leapfrog_step_numba,
    _local_timescale_numba,
    _rk4_step_numba,
    _rkn_acc_numba,
    _rkn_acc_time_numba,
    _rkn_adaptive_step_numba,
    _rkn_adaptive_step_time_numba,
    _rkn4_step_numba,
    _rkn4_step_time_numba,
)


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
