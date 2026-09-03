"""Skriptierte koerperbahnen -- das EINE bahnmodell dieses projekts.

ES GIBT GENAU EIN MODELL, UND ES IST KEPLER. Exakte propagation heisst: ein
schritt und hundert schritte geben dieselbe antwort. Genau das ist es, was
zeitraffer daran hindert, die planeten zu verschieben. Eine variante mit
konstanter rate oder Euler darf hier nie wieder entstehen -- diese spaltung
WAR der fehler (siehe .claude/rules/physics-world.md).

Dieselbe rechnung steht als Python-referenz in `bodies/body.py`
(`kepler_relative_xy`) und in `physics/world_kernels.py`; die drei muessen
bit-identisch bleiben.
"""
import math

import numpy as np
from numba import njit

from physics.kernels import BODY_MEMO_COLUMNS


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
