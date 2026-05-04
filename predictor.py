# predictor.py

from vec import Vec2
import math
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numba import njit
# Predictor ist absichtlich Numba-only
NUMBA_AVAILABLE = True


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
        return ax - ref_ax, ay - ref_ay


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

        p3x = px + half_dt * vx + 0.125 * dt2 * k2_ax
        p3y = py + half_dt * vy + 0.125 * dt2 * k2_ay
        k3_ax, k3_ay = _rkn_acc_numba(p3x, p3y, ref_ax, ref_ay, body_x, body_y, body_m, body_fixed, G)

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
    def _body_scripted_relative_xy_numba(index, local_t, body_m, body_a, body_e, body_theta, body_arg, body_parent, G):
        parent = body_parent[index]
        if parent < 0 or parent >= body_m.shape[0]:
            return 0.0, 0.0, 0

        a = body_a[index]
        e = body_e[index]
        parent_mass = body_m[parent]
        if a <= 0.0 or e < 0.0 or e >= 1.0 or parent_mass <= 0.0:
            return 0.0, 0.0, 0

        mu = G * parent_mass
        if mu <= 0.0:
            return 0.0, 0.0, 0

        nu0 = body_theta[index]
        arg = body_arg[index]

        cos_nu0 = math.cos(nu0)
        sin_nu0 = math.sin(nu0)
        denom = 1.0 + e * cos_nu0
        if abs(denom) <= 1e-14:
            return 0.0, 0.0, 0

        sqrt_one_minus_e2 = math.sqrt(max(0.0, 1.0 - e * e))
        sin_e0 = sqrt_one_minus_e2 * sin_nu0 / denom
        cos_e0 = (e + cos_nu0) / denom
        ecc_anomaly0 = math.atan2(sin_e0, cos_e0)
        mean_anomaly0 = ecc_anomaly0 - e * math.sin(ecc_anomaly0)

        mean_motion = math.sqrt(mu / (a * a * a))
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
        c = math.cos(arg)
        s = math.sin(arg)
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
    ):
        n = body_x.shape[0]
        if index < 0 or index >= n:
            return 0.0, 0.0

        chain = np.empty(n, dtype=np.int64)
        chain_count = 0
        cur = index

        while cur >= 0 and cur < n and chain_count < n:
            parent = body_parent[cur]
            if body_scripted[cur] == 0 or body_a[cur] <= 0.0 or parent < 0 or parent >= n:
                break
            chain[chain_count] = cur
            chain_count += 1
            cur = parent

        if cur < 0 or cur >= n:
            cur = index
            chain_count = 0

        wx = body_x[cur]
        wy = body_y[cur]

        for chain_pos in range(chain_count - 1, -1, -1):
            child = chain[chain_pos]
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
            )
            if ok == 0:
                return body_x[index], body_y[index]
            wx += rel_x
            wy += rel_y

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
                )
            else:
                rpx = ref_px
                rpy = ref_py
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
        )
        return ax - ref_ax, ay - ref_ay


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
    ):
        dt2 = dt * dt
        half_dt = 0.5 * dt
        mid_t = local_t + half_dt
        end_t = local_t + dt

        k1_ax, k1_ay = _rkn_acc_time_numba(
            px, py, local_t, ref_enabled, ref_index, ref_px, ref_py,
            body_x, body_y, body_m, body_fixed, body_scripted, body_a, body_e,
            body_theta, body_arg, body_parent, G, use_time_dependent_bodies
        )

        p2x = px + half_dt * vx + 0.125 * dt2 * k1_ax
        p2y = py + half_dt * vy + 0.125 * dt2 * k1_ay
        k2_ax, k2_ay = _rkn_acc_time_numba(
            p2x, p2y, mid_t, ref_enabled, ref_index, ref_px, ref_py,
            body_x, body_y, body_m, body_fixed, body_scripted, body_a, body_e,
            body_theta, body_arg, body_parent, G, use_time_dependent_bodies
        )

        p3x = px + half_dt * vx + 0.125 * dt2 * k2_ax
        p3y = py + half_dt * vy + 0.125 * dt2 * k2_ay
        k3_ax, k3_ay = _rkn_acc_time_numba(
            p3x, p3y, mid_t, ref_enabled, ref_index, ref_px, ref_py,
            body_x, body_y, body_m, body_fixed, body_scripted, body_a, body_e,
            body_theta, body_arg, body_parent, G, use_time_dependent_bodies
        )

        p4x = px + dt * vx + 0.5 * dt2 * k3_ax
        p4y = py + dt * vy + 0.5 * dt2 * k3_ay
        k4_ax, k4_ay = _rkn_acc_time_numba(
            p4x, p4y, end_t, ref_enabled, ref_index, ref_px, ref_py,
            body_x, body_y, body_m, body_fixed, body_scripted, body_a, body_e,
            body_theta, body_arg, body_parent, G, use_time_dependent_bodies
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
    ):
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
                body_theta, body_arg, body_parent, G, use_time_dependent_bodies
            )
            half1_px, half1_py, half1_vx, half1_vy = _rkn4_step_time_numba(
                px, py, vx, vy, local_t, half_dt, ref_enabled, ref_index, ref_px, ref_py,
                body_x, body_y, body_m, body_fixed, body_scripted, body_a, body_e,
                body_theta, body_arg, body_parent, G, use_time_dependent_bodies
            )
            half2_px, half2_py, half2_vx, half2_vy = _rkn4_step_time_numba(
                half1_px, half1_py, half1_vx, half1_vy, local_t + half_dt, half_dt,
                ref_enabled, ref_index, ref_px, ref_py, body_x, body_y, body_m, body_fixed,
                body_scripted, body_a, body_e, body_theta, body_arg, body_parent, G,
                use_time_dependent_bodies
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
    ):
        out = np.empty((max_points, 3), dtype=np.float64)
        out[0, 0] = init_px
        out[0, 1] = init_py
        out[0, 2] = 0.0

        stats = np.zeros(7, dtype=np.float64)

        count = 1
        px = init_px
        py = init_py
        vx = init_vx
        vy = init_vy
        t = 0.0
        accumulated = 0.0
        proposed_dt = base_dt

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

                    out[count, 0] = sample_px
                    out[count, 1] = sample_py
                    out[count, 2] = sample_t
                    count += 1

                    accumulated = 0.0
                    rem_len = seg_len - placed

                if failure_code != 0.0:
                    break

                if rem_len + accumulated < precision:
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

        return out, count, stats


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
        out = np.empty((max_points, 3), dtype=np.float64)
        out[0, 0] = init_px
        out[0, 1] = init_py
        out[0, 2] = 0.0

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
        out = np.empty((max_points, 3), dtype=np.float64)
        out[0, 0] = init_px
        out[0, 1] = init_py
        out[0, 2] = 0.0

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
        self.rkn_last_accepted_steps = 0
        self.rkn_last_rejected_steps = 0
        self.rkn_last_min_dt = 0.0
        self.rkn_last_max_dt = 0.0
        self.rkn_last_max_error_norm = 0.0
        self.rkn_last_failed = False
        self.rkn_last_failure_reason = ""
        self.strict_snapshot_matching = bool(strict_snapshot_matching)
        self.use_time_dependent_bodies = bool(use_time_dependent_bodies)
        self.use_reference_acceleration_correction = bool(use_reference_acceleration_correction)
        self._trajectory_version = 0
        self._last_seen_px = None
        self._last_seen_py = None
        self._last_seen_vx = None
        self._last_seen_vy = None
        self._last_seen_sim_time = None
        self.velocity_invalidation_abs_tol = 1.0
        self.velocity_invalidation_rel_tol = 1e-5
        self.position_invalidation_abs_tol = 100.0
        self.sync_recompute_on_velocity_change = True
        self.max_async_sim_age = max(2.0 * self.dt, 1.0)
        self.max_async_wall_age = 0.5

        self.points = np.empty((0, 3), dtype=np.float64) if np is not None else []
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
        # The predictor uses exactly one dedicated worker thread. The main
        # simulation/render thread remains separate, and one trajectory is
        # integrated sequentially inside this worker.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="predictor-worker")
        self._predictor_worker_threads = 1
        if self.debug:
            try:
                print("PRED_DBG_THREAD: predictor worker max_workers=1", flush=True)
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
        self.points = np.empty((0, 3), dtype=np.float64) if np is not None else []
        self._roll_states = np.empty((0, 5), dtype=np.float64) if np is not None else []
        self.initialized = False

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

    def _empty_points_array(self):
        return np.empty((0, 3), dtype=np.float64) if np is not None else []

    def _clear_prediction_points(self):
        self.points = self._empty_points_array()
        self._roll_states = np.empty((0, 5), dtype=np.float64) if np is not None else []
        self.initialized = False

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

        delta_speed = math.hypot(cur_vx - float(last_vx), cur_vy - float(last_vy))
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

        reason = None
        if delta_speed > allowed_speed:
            reason = "velocity"
        elif delta_pos > allowed_pos:
            reason = "position"

        if reason is None:
            self._remember_ship_state(ship, world)
            return False

        old_version = int(self._trajectory_version)
        self._trajectory_version = old_version + 1
        if self.debug:
            try:
                if reason == "velocity":
                    print(
                        "PRED_DBG_TRAJECTORY_INVALIDATED: "
                        f"reason=velocity dv={delta_speed:.6e} allowed={allowed_speed:.6e} "
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

    def _anchor_first_point(self, ship):
        if self._points_count() == 0:
            return
        sx = float(ship.position.x)
        sy = float(ship.position.y)
        if np is not None and isinstance(self.points, np.ndarray):
            dx = sx - float(self.points[0, 0])
            dy = sy - float(self.points[0, 1])
            if math.isfinite(dx) and math.isfinite(dy):
                self.points[:, 0] += dx
                self.points[:, 1] += dy
                self.points[0, 0] = sx
                self.points[0, 1] = sy
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
            # timestamp beibehalten falls beim ersten punkt vorhanden
            try:
                t0 = float(self.points[0][2])
            except Exception:
                t0 = 0.0
            try:
                dx = sx - float(self.points[0][0])
                dy = sy - float(self.points[0][1])
                for i, p in enumerate(self.points):
                    if i == 0:
                        self.points[i] = (sx, sy, t0)
                    elif hasattr(p, "__len__") and len(p) >= 3:
                        self.points[i] = (float(p[0]) + dx, float(p[1]) + dy, float(p[2]))
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


            self._view_scale = scale
 
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

    def _effective_precision(self):
        effective = float(self.precision)
        if self.auto_precision_from_zoom and self._view_scale is not None:
            zoom_precision = self.target_screen_step_px / max(self._view_scale, 1e-30)
            effective = min(effective, max(self.min_precision, zoom_precision))
        return effective

    def _serialize_bodies(self, world):
        lst = []
        for b in world.body:
            lst.append((b.position.x, b.position.y, b.mass, getattr(b, "fixed", True)))
        return lst

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

    def _compute_acc(self, position, bodies, G):
        total = Vec2(0.0, 0.0)
        for bx, by, mass, fixed in bodies:
            if not fixed:
                continue
            dirv = Vec2(bx, by) - position
            dist2 = dirv.magnitude_squared()
            if dist2 < 1e-12:
                continue
            invd = 1.0 / math.sqrt(dist2)
            fdir = dirv * invd
            accm = G * mass / dist2
            total += fdir * accm
        return total

    def _rk4_step(self, p, v, bodies, G):
        dt = self.dt

        k1_a = self._compute_acc(p, bodies, G)
        k1_v = v

        p2 = p + k1_v * (dt / 2)
        v2 = v + k1_a * (dt / 2)
        k2_a = self._compute_acc(p2, bodies, G)
        k2_v = v2

        p3 = p + k2_v * (dt / 2)
        v3 = v + k2_a * (dt / 2)
        k3_a = self._compute_acc(p3, bodies, G)
        k3_v = v3

        p4 = p + k3_v * dt
        v4 = v + k3_a * dt
        k4_a = self._compute_acc(p4, bodies, G)
        k4_v = v4

        new_p = p + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * (dt / 6)
        new_v = v + (k1_a + 2 * k2_a + 2 * k3_a + k4_a) * (dt / 6)

        return new_p, new_v

    def _compute_full_python(self, ship, world, max_points):
        bodies = self._serialize_bodies(world)
        G = world.G

        pos = ship.position.copy()
        vel = ship.velocity.copy()
        pts = [pos.copy()]

        accumulated = 0.0

        safety_iters = max(100000, max_points * 100)
        iters = 0

        while len(pts) < max_points and iters < safety_iters:
            iters += 1
            next_pos, next_vel = self._rk4_step(pos, vel, bodies, G)

            seg_vec = next_pos - pos
            seg_len = seg_vec.magnitude()

            if seg_len <= 0:
                pos = next_pos
                vel = next_vel
                continue

            local_pos = pos
            local_vel = vel
            remaining_vec = seg_vec
            remaining_len = seg_len

            while remaining_len + accumulated >= self.precision and len(pts) < max_points:
                if remaining_len <= 0:
                    break

                distance_to_place = self.precision - accumulated
                fraction = distance_to_place / remaining_len

                sample_point = local_pos + remaining_vec * fraction
                sample_vel = local_vel + (next_vel - local_vel) * fraction

                pts.append(sample_point.copy())

                local_pos = sample_point
                local_vel = sample_vel

                remaining_vec = next_pos - local_pos
                remaining_len = remaining_vec.magnitude()
                accumulated = 0.0

            if remaining_len + accumulated < self.precision:
                accumulated += remaining_len

            pos = next_pos
            vel = next_vel

        return pts

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

    def _make_snapshot(self, ship, world, max_points):
        effective_precision = self._effective_precision()
        ref_enabled, ref_px, ref_py = self._resolve_reference_body(world)
        physics_ref_enabled = int(ref_enabled) if self.use_reference_acceleration_correction else 0
        ref_index = self._current_reference_body_index()
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
            "rkn_max_dt": float(self.rkn_max_dt),
            "rkn_rtol": float(self.rkn_rtol),
            "rkn_atol_pos": float(self.rkn_atol_pos),
            "rkn_atol_vel": float(self.rkn_atol_vel),
            "rkn_safety": float(self.rkn_safety),
            "rkn_min_factor": float(self.rkn_min_factor),
            "rkn_max_factor": float(self.rkn_max_factor),
            "rkn_max_rejects": int(self.rkn_max_rejects),
            "strict_snapshot_matching": bool(self.strict_snapshot_matching),
            "use_time_dependent_bodies": bool(self.use_time_dependent_bodies),
            "use_reference_acceleration_correction": bool(self.use_reference_acceleration_correction),
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
        return snapshot

    def _compute_from_snapshot(self, snapshot):
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

            out, used, rkn_stats = _compute_distance_points_rkn_numba(
                snapshot["ship_px"],
                snapshot["ship_py"],
                snapshot["ship_vx"],
                snapshot["ship_vy"],
                int(snapshot.get("ref_enabled", 0)),
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
            )
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
                int(snapshot.get("ref_enabled", 0)),
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

        return {"points": points, "snapshot": snapshot, "computed": computed_count, "rkn_stats": rkn_stats}

    def _compute_full_rolling(self, ship, world):
        start_ts = time.time()
        try:
            if self.num_points <= 0:
                self.points = np.empty((0, 3), dtype=np.float64) if np is not None else []
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
            new_points = states[:, :3].copy() if (np is not None and isinstance(states, np.ndarray) and states.shape[0] > 0) else np.empty((0, 3), dtype=np.float64)

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
                self.points = np.empty((0, 3), dtype=np.float64) if np is not None else []
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
        if not self.use_reference_acceleration_correction:
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
        self.points = self._roll_states[:, :3].copy()
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

        self._anchor_first_point(ship)
        if np is not None and isinstance(self._roll_states, np.ndarray) and self._roll_states.shape[0] > 0:
            self._roll_states[0, 0] = float(ship.position.x)
            self._roll_states[0, 1] = float(ship.position.y)
            try:
                self._roll_states[0, 2] = float(world.time)
            except Exception:
                pass
            self._roll_states[0, 3] = float(ship.velocity.x)
            self._roll_states[0, 4] = float(ship.velocity.y)

    def _submit_async_compute(self, ship, world, max_points):
        pending = getattr(self, "_pending_futures", [])

        if self._single_flight:
            if len(pending) > 0:
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
        if self._single_flight:
            self._pending_futures = [(job_id, fut)]
        else:
            self._pending_futures.append((job_id, fut))

        self._next_job_id += 1
        self._jobs_submitted += 1

    def _swap_ready_result(self, current_ship=None, current_world=None):
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

            if len(pending) > 2:

                pending[:] = pending[-2:]

            # find first completed future
            for idx, (jid, fut) in enumerate(pending):
                try:
                    done = fut.done()
                except Exception:
                    done = False
                if done:
                    finished_future = fut
                    finished_job_id = jid
                    pending.pop(idx)
                    break

            if finished_future is None:
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


                snap_view = snapshot.get("view_scale", None)
                is_stale_view = False
                try:
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
                is_stale_speed = delta_speed > allowed_speed
                is_stale_pos = pos_delta > allowed_pos
                max_async_sim_age = float(getattr(self, "max_async_sim_age", max(2.0 * float(self.dt), 1.0)))
                is_stale_sim_time = sim_age is not None and abs(float(sim_age)) > max_async_sim_age
                wall_age = 0.0
                try:
                    wall_age = max(0.0, time.time() - float(snapshot.get("submit_ts", time.time())))
                except Exception:
                    wall_age = 0.0
                max_wall_age = float(getattr(self, "max_async_wall_age", 0.5))
                is_stale_wall_age = sim_age is None and wall_age > max_wall_age

                reject_reason = None
                if is_stale_view:
                    reject_reason = "view_scale"
                elif is_stale_reference:
                    reject_reason = "reference_frame"
                elif is_stale_sim_time:
                    reject_reason = "sim_age"
                elif is_stale_wall_age:
                    reject_reason = "wall_age"
                elif is_stale_speed:
                    reject_reason = "velocity"
                elif is_stale_pos:
                    reject_reason = "position"

                if reject_reason is not None:
                    self._log_snapshot_result(False, reject_reason, snapshot, cur_sim_time, sim_age, pos_delta, delta_speed)

                    if (
                        reject_reason != "view_scale"
                        and reject_reason in ("sim_age", "wall_age")
                        and self.force_sync_on_stale
                        and current_world is not None
                    ):
                        self._compute_full(current_ship, current_world)
                        self._last_swapped_job_id = finished_job_id
                        self._jobs_swapped += 1
                        self._log_snapshot_result(True, "force_sync_on_stale", snapshot, cur_sim_time, sim_age, pos_delta, delta_speed)
                        return True
                    return False

                needs_rebase = pos_delta > 1e-9
                if needs_rebase:
                    sim_time_small = sim_age is None or abs(float(sim_age)) <= max_async_sim_age
                    pos_delta_safe = math.isfinite(pos_delta) and pos_delta <= max(
                        allowed_pos,
                        max(cur_speed, 1.0) * max_async_sim_age * 2.0,
                    )
                    if delta_speed <= allowed_speed and (not is_stale_reference) and sim_time_small and pos_delta_safe:
                        points = self._rebase_points_to_current_snapshot(points, snapshot, current_ship)
                        self._log_snapshot_result(True, "rebased", snapshot, cur_sim_time, sim_age, pos_delta, delta_speed)
                    else:
                        reason = "unsafe_rebase"
                        if not sim_time_small:
                            reason = "unsafe_rebase_sim_time"
                        elif delta_speed > allowed_speed:
                            reason = "unsafe_rebase_velocity"
                        elif not pos_delta_safe:
                            reason = "unsafe_rebase_position"
                        self._log_snapshot_result(False, reason, snapshot, cur_sim_time, sim_age, pos_delta, delta_speed)
                        return False
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

        spacing_for_cap = self.base_precision if self.base_precision > 0.0 else self.precision
        max_by_length = max(1, int(self.length / spacing_for_cap) + 1)
        return min(self.num_points, max_by_length)

    def _compute_full(self, ship, world):
        if self.rolling_mode:
            self._compute_full_rolling(ship, world)
            return

        if self.num_points <= 0:
            self.points = np.empty((0, 3), dtype=np.float64) if np is not None else []
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
            self._anchor_first_point(ship)
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
        self._anchor_first_point(ship)

    def update(self, ship, world):
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
                        self._anchor_first_point(ship)
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
                self._anchor_first_point(ship)
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
            self._anchor_first_point(ship)
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
                self._anchor_first_point(ship)
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

                if delta_speed >= allowed_speed:
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
                        self._anchor_first_point(ship)
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
                        self._anchor_first_point(ship)
                        if self.debug and not getattr(self, "_suppress_dbg_computed", False):
                            try:
                                print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                            except Exception:
                                pass
                        self._computed_since_last_update = 0
                        return
        except Exception:
            pass

        swapped = self._swap_ready_result(ship, world)
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

        if self.recompute_every_update or self._points_count() < target_points or swapped:
            self._submit_async_compute(ship, world, target_points)

        # Async results are accepted only after strict snapshot matching, with
        # whole-curve rebasing applied in _swap_ready_result when safe.
        # Do not anchor here; that would hide stale curve tails.
        if self.debug and not getattr(self, "_suppress_dbg_computed", False):
            try:
                print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
            except Exception:
                pass
        self._computed_since_last_update = 0

    def get_points(self):
        return self.points

    def get_precision_factor(self):
        if self.base_precision <= 0.0:
            return 1.0
        return self.precision / self.base_precision

    def get_display_length(self):
        if self.length is None:
            return None
        return self.length * self.get_precision_factor()

    def set_precision(self, meters: float):
        meters = float(meters)
        if meters <= 0.0:
            raise ValueError("precision must be > 0")
        self.precision = meters
        if self.rolling_mode:
            self.reset()
        elif self.async_compute:
            self._cancel_pending_job()

    def set_length(self, meters: float | None):
        if meters is None:
            self.length = None
            if self.rolling_mode:
                self.reset()
            elif self.async_compute:
                self._cancel_pending_job()
            return
        meters = float(meters)
        if meters <= 0.0:
            raise ValueError("length must be > 0")
        self.length = meters
        if self.rolling_mode:
            self.reset()
        elif self.async_compute:
            self._cancel_pending_job()

    def set_num_points(self, count: int):
        self.num_points = max(0, int(count))
        self.reset()

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
                    try:
                        self._roll_states = self._roll_states[remove_count:]
                        if isinstance(self._roll_states, np.ndarray) and self._roll_states.shape[0] > 0:
                            self.points = self._roll_states[:, :3].copy()
                        else:
                            self.points = np.empty((0, 3), dtype=np.float64)
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
                try:
                    self.points = self.points[remove_count:]
                except Exception:
                    self.points = np.array(self.points[remove_count:], dtype=np.float64)
            return int(remove_count)

        # List / generic fallback: use same projection logic
        try:
            n = len(self.points)
            if n <= 1:
                return 0
        except Exception:
            return 0

        remove_count = 0
        try:
            for i in range(n - 1):
                p0 = self.points[i]
                p1 = self.points[i + 1]
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
            try:
                del self.points[:remove_count]
            except Exception:
                for _ in range(remove_count):
                    try:
                        self.points.pop(0)
                    except Exception:
                        break
        return int(remove_count)
