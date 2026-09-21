"""Beschleunigung und schrittverfahren.

Zwei familien, und der unterschied ist tragend:

  * die ZEITLOSEN kerne (`_compute_acc_numba`, `_rkn4_step_numba`, ...) sehen
    die koerper an festen orten -- fuer kurze bogen, in denen sich die quellen
    nicht messbar bewegen;
  * die ZEIT-kerne (`*_time_numba`) setzen die koerper zu jedem teilschritt
    ueber `_body_position_at_time_numba` neu -- das ist der pfad, den eine
    lange vorhersage braucht.

`_rkn_adaptive_step_*` steuert die schrittweite ueber den fehler; die decke
greift nur im fernfeld. Predictor und World teilen sich diese verfahren --
eine integrator-konstante hier zu aendern heisst, sie auch in
`physics/world_kernels.py` zu aendern.
"""
import math

from numba import njit

from physics.kernels.kepler import _body_position_at_time_numba


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

    # k3 teilt sein argument mit k2 (klassisches RKN4) und ist damit
    # dasselbe k -- siehe _rkn4_step_time_numba.
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

    MINIMUM ueber alle koerper, NIE argmax(g): der koerper mit der
    groessten anziehung ist nicht immer der umlaufene (jenseits von
    ~2.6e8 m von der Erde zieht die Sonne staerker), und ein minimum
    stetiger funktionen springt nicht. Siehe
    `.claude/rules/physics-world.md`.

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
        # zeit wie die darunter und trifft deshalb den notizblock.
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

    # DRITTE STUFE: im klassischen RKN4 teilen sich k2 und k3 ihr argument
    # (p3 = p0 + v0*h/2 + k1*h^2/8 = p2) -- genau diese identitaet gibt
    # ordnung 4 aus 3 kraftauswertungen. Weil `_rkn_acc_time_numba` eine
    # reine funktion von (ort, zeit) ist, IST k3 damit k2; die auswertung
    # entfaellt. Die summen unten behalten die form `k1 + k2 + k3`, damit
    # die rundung dieselbe bleibt wie mit ausgewertetem k3.
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
    # tausend sekunden bei langer vorausschau). Sie darf nicht ueber die
    # bahn springen, und wie eng sie sein muss, haengt davon ab, wo das
    # schiff GERADE ist: auf einer abflugbahn gilt nahe der Erde deren
    # zeitskala, im heliozentrischen reiseflug die viel groessere der
    # Sonne. Deshalb `t_char/timescale_divisor` je schritt am aktuellen ort.
    #
    # Die zeitskala wird zur zeit `local_t` ausgewertet, also genau der
    # zeit, zu der gleich darauf k1 alle koerper braucht -- sie waermt damit
    # den notizblock, statt zusaetzliche kepler-loesungen zu kosten.
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
            # qualitaetsstufe: die ortliche decke macht das nahfeld nie
            # strenger als diese.
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
