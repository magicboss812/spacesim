"""Der brennbogen: schwerkraft plus konstant gerichteter schub.

`_profile_accel_numba` ist der ZWILLING von
`ship/maneuver/profile.py::BurnProfile.accel_at`. Er steht hier ein zweites
mal, weil numba keine Python-objekte annimmt und die methode im kernel
deshalb nicht aufrufbar ist. Die doppelung ist bewusst und wird bewacht:
`tests/maneuver_profile_test.py` abschnitt 8 vergleicht beide an 2004
stuetzstellen auf EXAKTE gleichheit. Wer eine der beiden aendert, aendert
beide.

DIE SCHRITTE LIEGEN AUF DEN PHASENGRENZEN, und das ist keine feinheit.
Das profil ist an den beiden rampenknicken stetig, aber nicht
differenzierbar. RK4 wertet je schritt an drei zeitpunkten aus und wiegt
sie wie Simpson -- ueber einen knick hinweg ist das nur noch erster
ordnung. Gemessen bei 400 gleichverteilten schritten ueber ein 12.6-s-
profil: **119.999775 statt 120.000000 m/s**, also 1.9e-6 relativ. Innerhalb
einer phase ist die beschleunigung dagegen linear oder konstant, und
Simpson integriert beides EXAKT -- mit phasenweiser schrittung landet das
gelieferte delta-v auf der letzten stelle. Deshalb bekommt jede der drei
phasen ihre eigene, gleichmaessige schrittweite statt einer gemeinsamen.
"""

import numpy as np
from numba import njit

from .integrators import _rkn_acc_time_numba


@njit(cache=True, nogil=True, fastmath=True)
def _profile_accel_numba(tau, a_peak, ramp_time, hold_time, total_time, ramp_rate):
    if tau <= 0.0 or tau >= total_time:
        return 0.0
    if tau < ramp_time:
        return ramp_rate * tau
    if tau < ramp_time + hold_time:
        return a_peak
    return ramp_rate * (total_time - tau)


@njit(cache=True, nogil=True, fastmath=True)
def _burn_arc_numba(
    init_px,
    init_py,
    init_vx,
    init_vy,
    init_t,
    dir_x,
    dir_y,
    a_peak,
    ramp_time,
    hold_time,
    total_time,
    ramp_rate,
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
    steps,
):
    """RK4 ueber den brennbogen: schwerkraft plus konstant gerichteter schub.

    FESTE schrittweite je phase, anders als die coast-kerne. Der bogen ist
    kurz und seine beschleunigung ist stueckweise linear -- eine adaptive
    steuerung haette hier nichts zu regeln und wuerde nur ihre eigene
    fehlerschaetzung bezahlen. `steps` ist das GESAMTbudget; es wird nach
    dauer auf rampe / halten / rampe verteilt (siehe modulkopf).

    Die schubrichtung ist KONSTANT, in der welt festgehalten: sie wird beim
    scharfschalten einmal aufgeloest und aendert sich waehrend des brennens
    nicht mehr. Genau das macht vorschau und ausfuehrung identisch -- eine
    mitdrehende richtung kann die vorschau nicht im voraus kennen.

    `init_t` ist LOKAL zur schnappschuss-epoche, wie bei
    `_compute_distance_points_rkn_numba`. Spalten: x, y, t, vx, vy.

    `body_memo` wird mit NULL ZEILEN uebergeben. Der bogen ist ein paar
    hundert schritte lang, der notizblock haette also kaum wiederholungen zu
    sparen -- und den vorlauf, den der grosse kernel dafuer faehrt, hier zu
    wiederholen hiesse ihn zu verdoppeln.
    """
    budget = steps if steps > 0 else 1

    # Die drei phasendauern. Die abfahrrampe ist so lang wie die anfahrt,
    # wird aber aus der differenz genommen, damit die summe exakt
    # total_time ergibt.
    d1 = ramp_time
    d2 = hold_time
    d3 = total_time - ramp_time - hold_time
    if d3 < 0.0:
        d3 = 0.0

    if total_time <= 0.0:
        out = np.empty((1, 5), dtype=np.float64)
        out[0, 0] = init_px
        out[0, 1] = init_py
        out[0, 2] = init_t
        out[0, 3] = init_vx
        out[0, 4] = init_vy
        return out, 1

    n1 = 0
    n2 = 0
    n3 = 0
    if d1 > 0.0:
        n1 = int(budget * d1 / total_time)
        if n1 < 1:
            n1 = 1
    if d2 > 0.0:
        n2 = int(budget * d2 / total_time)
        if n2 < 1:
            n2 = 1
    if d3 > 0.0:
        n3 = int(budget * d3 / total_time)
        if n3 < 1:
            n3 = 1
    n_total = n1 + n2 + n3
    if n_total < 1:
        n1 = 1
        n_total = 1

    out = np.empty((n_total + 1, 5), dtype=np.float64)

    px = init_px
    py = init_py
    vx = init_vx
    vy = init_vy
    t = init_t

    out[0, 0] = px
    out[0, 1] = py
    out[0, 2] = t
    out[0, 3] = vx
    out[0, 4] = vy

    ref_enabled = 0
    written = 0

    for phase in range(3):
        if phase == 0:
            n_phase = n1
            tau_start = 0.0
            tau_end = d1
        elif phase == 1:
            n_phase = n2
            tau_start = d1
            tau_end = d1 + d2
        else:
            n_phase = n3
            tau_start = d1 + d2
            tau_end = total_time
        if n_phase <= 0:
            continue

        h = (tau_end - tau_start) / n_phase

        for i in range(n_phase):
            tau = tau_start + i * h
            # Die absolute kernelzeit aus der phasenzeit ableiten, nicht
            # aufaddieren: sonst sammelt sich ueber ein paar hundert
            # schritte ein rundungsdrift in der zeit, mit der die
            # koerperpositionen ausgewertet werden.
            t = init_t + tau

            gx, gy = _rkn_acc_time_numba(
                px, py, t, ref_enabled, ref_index, ref_px, ref_py,
                body_x, body_y, body_m, body_fixed, body_scripted,
                body_a, body_e, body_theta, body_arg, body_parent,
                G, use_time_dependent_bodies, body_memo,
            )
            th = _profile_accel_numba(tau, a_peak, ramp_time, hold_time,
                                      total_time, ramp_rate)
            k1ax = gx + dir_x * th
            k1ay = gy + dir_y * th
            k1vx = vx
            k1vy = vy

            p2x = px + k1vx * (h * 0.5)
            p2y = py + k1vy * (h * 0.5)
            v2x = vx + k1ax * (h * 0.5)
            v2y = vy + k1ay * (h * 0.5)
            gx, gy = _rkn_acc_time_numba(
                p2x, p2y, t + h * 0.5, ref_enabled, ref_index, ref_px, ref_py,
                body_x, body_y, body_m, body_fixed, body_scripted,
                body_a, body_e, body_theta, body_arg, body_parent,
                G, use_time_dependent_bodies, body_memo,
            )
            th = _profile_accel_numba(tau + h * 0.5, a_peak, ramp_time,
                                      hold_time, total_time, ramp_rate)
            k2ax = gx + dir_x * th
            k2ay = gy + dir_y * th
            k2vx = v2x
            k2vy = v2y

            p3x = px + k2vx * (h * 0.5)
            p3y = py + k2vy * (h * 0.5)
            v3x = vx + k2ax * (h * 0.5)
            v3y = vy + k2ay * (h * 0.5)
            gx, gy = _rkn_acc_time_numba(
                p3x, p3y, t + h * 0.5, ref_enabled, ref_index, ref_px, ref_py,
                body_x, body_y, body_m, body_fixed, body_scripted,
                body_a, body_e, body_theta, body_arg, body_parent,
                G, use_time_dependent_bodies, body_memo,
            )
            th = _profile_accel_numba(tau + h * 0.5, a_peak, ramp_time,
                                      hold_time, total_time, ramp_rate)
            k3ax = gx + dir_x * th
            k3ay = gy + dir_y * th
            k3vx = v3x
            k3vy = v3y

            p4x = px + k3vx * h
            p4y = py + k3vy * h
            v4x = vx + k3ax * h
            v4y = vy + k3ay * h
            gx, gy = _rkn_acc_time_numba(
                p4x, p4y, t + h, ref_enabled, ref_index, ref_px, ref_py,
                body_x, body_y, body_m, body_fixed, body_scripted,
                body_a, body_e, body_theta, body_arg, body_parent,
                G, use_time_dependent_bodies, body_memo,
            )
            # Am phasenende faellt `tau + h` genau auf den knick. Das ist
            # unbedenklich: das profil ist dort STETIG (ramp_rate*ramp_time
            # == a_peak), beide zweige liefern denselben wert. Ein
            # einseitiger grenzwert waere hier eine scheinbare vorsicht.
            th = _profile_accel_numba(tau + h, a_peak, ramp_time, hold_time,
                                      total_time, ramp_rate)
            k4ax = gx + dir_x * th
            k4ay = gy + dir_y * th
            k4vx = v4x
            k4vy = v4y

            px = px + (k1vx + 2.0 * k2vx + 2.0 * k3vx + k4vx) * (h / 6.0)
            py = py + (k1vy + 2.0 * k2vy + 2.0 * k3vy + k4vy) * (h / 6.0)
            vx = vx + (k1ax + 2.0 * k2ax + 2.0 * k3ax + k4ax) * (h / 6.0)
            vy = vy + (k1ay + 2.0 * k2ay + 2.0 * k3ay + k4ay) * (h / 6.0)

            written += 1
            out[written, 0] = px
            out[written, 1] = py
            out[written, 2] = init_t + tau_start + (i + 1) * h
            out[written, 3] = vx
            out[written, 4] = vy

    return out, written + 1
