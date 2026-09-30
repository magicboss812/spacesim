"""Skriptierte koerperbahnen -- das EINE bahnmodell dieses projekts.

ES GIBT GENAU EIN MODELL, UND ES IST KEPLER. Exakte propagation heisst: ein
schritt und hundert schritte geben dieselbe antwort. Genau das ist es, was
zeitraffer daran hindert, die planeten zu verschieben. Keine variante mit
konstanter rate oder Euler einfuehren (siehe .claude/rules/physics-world.md).

Dieselbe rechnung steht als Python-referenz in `bodies/body.py`
(`kepler_relative_xy`) und in `physics/world_kernels.py`; die drei muessen
bit-identisch bleiben.
"""
import math

from numba import njit


@njit(cache=True, nogil=True, fastmath=True)
def _body_kepler_constants_numba(index, body_m, body_a, body_e, body_theta, body_arg, body_parent, G):
    """Die ZEITUNABHAENGIGEN groessen einer skriptierten bahn.

    M0, mittlere bewegung, sqrt(1-e^2) und cos/sin des periapsis-arguments
    haengen nur von den bahnelementen ab und koennen deshalb einmal je lauf
    in body_memo vorberechnet werden. Die rechnung ist WORT FUER WORT die
    von bodies.body.kepler_relative_xy, damit alle wege exakt dieselben
    gleitkommazahlen erzeugen.

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
def _kepler_rel_consts(local_t, a, e, mean_anomaly0, mean_motion,
                       sqrt_one_minus_e2, c, s):
    """Die kepler-loesung selbst, NUR AUS SKALAREN.

    Wort fuer wort der rechenweg von `bodies.body.kepler_relative_xy`. Als
    eigene funktion ohne array-argumente, weil jede numba-funktion, die einen
    nicht eingebetteten aufruf mit arrays enthaelt, bei JEDEM aufruf
    ~120 ns fuer die referenzzaehlung dieser arrays zahlt -- auch wenn der
    aufruf gar nicht ausgefuehrt wird (gemessen: dieselbe tafelabfrage 9 ns
    ohne, 130 ns mit einem solchen aufruf im rumpf). Rueckgabe
    (rel_x, rel_y, ok).
    """
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
def _body_scripted_relative_xy_numba(index, local_t, body_m, body_a, body_e, body_theta, body_arg, body_parent, G, body_memo):
    # Schneller weg: der kernel hat die zeitunabhaengigen groessen im
    # vorlauf nach body_memo[:, 4:10] gelegt (spalte 9: 0 = nicht
    # vorberechnet, 1 = gueltig, -1 = bahn unbrauchbar). Ohne notizblock
    # -- oder mit abgeschaltetem `use_body_memo` -- rechnet der zweig
    # darunter alles selbst.
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

    return _kepler_rel_consts(local_t, a, e, mean_anomaly0, mean_motion,
                              sqrt_one_minus_e2, c, s)


#: Spaltenaufteilung des notizblocks (`physics/kernels/__init__.py`,
#: BODY_MEMO_COLUMNS): 0-3 platz 0 [t, x, y, gueltig], 4-9 die
#: zeitunabhaengigen bahngroessen, 10-13 die gruppierung ferner mondsysteme
#: [wurzel, systemmasse, fern-schwelle^2, fern-merker] (siehe
#: `integrators._compute_acc_time_numba`), ab MEMO_SLOT_BASE die weiteren
#: plaetze zu je vier spalten, ab MEMO_TAB die planetentafel
#: [knotenabstand, 1/knotenabstand, nah-schwelle^2] und MEMO_TAB_KNOTS knoten
#: zu je [index + MEMO_TAB_OFFSET, x, y], dann art/eltern/a/e je koerper
#: (`_setup_body_kinds`), die aufstellung des laufenden auswertungszeitpunkts
#: [x, y, ausgelassen] (`_place_bodies_numba`), zuletzt der schreibzeiger.
MEMO_GROUP = 10
MEMO_SLOT_BASE = 14
MEMO_SLOTS = 5
MEMO_TAB = MEMO_SLOT_BASE + 4 * (MEMO_SLOTS - 1)
MEMO_TAB_KNOTS = 16
MEMO_TAB_OFFSET = 1073741824.0
MEMO_KIND = MEMO_TAB + 3 + 3 * MEMO_TAB_KNOTS
MEMO_PARENT = MEMO_KIND + 1
MEMO_A = MEMO_KIND + 2
MEMO_E = MEMO_KIND + 3
MEMO_POS = MEMO_KIND + 4
MEMO_CURSOR = MEMO_POS + 3
MEMO_COLUMNS = MEMO_CURSOR + 1


@njit(cache=True, nogil=True, fastmath=True)
def _memo_slots(body_memo):
    """Wie viele zeiten der notizblock je koerper haelt (ein (n, 10)-block: 1)."""
    if body_memo.shape[1] >= MEMO_COLUMNS:
        return MEMO_SLOTS
    return 1


@njit(cache=True, nogil=True, fastmath=True)
def _memo_find(body_memo, index, local_t, slots):
    """Spalte des platzes, der `index` zu GENAU `local_t` haelt, sonst -1."""
    if body_memo[index, 3] != 0.0 and body_memo[index, 0] == local_t:
        return 0
    for k in range(1, slots):
        col = MEMO_SLOT_BASE + 4 * (k - 1)
        if body_memo[index, col + 3] != 0.0 and body_memo[index, col] == local_t:
            return col
    return -1


@njit(cache=True, nogil=True, fastmath=True)
def _memo_store(body_memo, index, local_t, x, y, slots):
    """Reihum in den naechsten platz schreiben (der aelteste faellt heraus)."""
    if slots <= 1:
        col = 0
    else:
        cursor_col = MEMO_CURSOR
        k = int(body_memo[index, cursor_col])
        if k < 0 or k >= slots:
            k = 0
        col = 0 if k == 0 else MEMO_SLOT_BASE + 4 * (k - 1)
        body_memo[index, cursor_col] = float((k + 1) % slots)
    body_memo[index, col] = local_t
    body_memo[index, col + 1] = x
    body_memo[index, col + 2] = y
    body_memo[index, col + 3] = 1.0


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
    # `body_memo` ist ein (n, BODY_MEMO_COLUMNS)-notizblock je koerper;
    # spalten 0-3 sind [t, x, y, gueltig] (4-9: siehe
    # _body_scripted_relative_xy_numba). Er ist BIT-IDENTISCH: gemerkt wird
    # genau der wert, den derselbe rechenweg eben erzeugt hat, und getroffen
    # wird nur bei EXAKT gleicher zeit. Er lohnt sich, weil dieselbe
    # koerperposition pro integrationsschritt mehrfach gebraucht wird
    # (schiff und bezugspunkt zur selben zeit, jeder mond braucht seinen
    # planeten, die schrittverdopplung wertet dieselben zeiten mehrfach aus).
    # Ein aufrufer ohne notizblock uebergibt ein array mit 0 zeilen.
    #
    # MEHRERE ZEITEN JE KOERPER (spalten ab MEMO_SLOT_BASE, siehe _memo_slots). Ein
    # schrittverdoppelter schritt fragt t, t+h/2, t+h (voller schritt),
    # t, t+h/4, t+h/2 und t+h/2, t+3h/4, t+h (die haelften) -- fuenf
    # verschiedene zeiten, aber mit EINEM platz je koerper sind es sieben
    # kepler-loesungen, weil t, t+h/2 und t+h verdraengt sind, bevor sie
    # wieder gefragt werden. Fuenf plaetze, reihum ueberschrieben: vier.
    #
    # Die gueltig-spalte ist NICHT redundant: eine NaN-zeitspalte als
    # "leer"-marke funktioniert nicht, weil `fastmath=True` LLVMs `nnan`
    # einschaltet und `NaN == local_t` dann zu true gefaltet werden darf.
    n = body_x.shape[0]
    if index < 0 or index >= n:
        return 0.0, 0.0

    use_memo = body_memo.shape[0] == n
    slots = _memo_slots(body_memo) if use_memo else 1
    if use_memo:
        col = _memo_find(body_memo, index, local_t, slots)
        if col >= 0:
            return body_memo[index, col + 1], body_memo[index, col + 2]

    # Die elternkette wird nicht gespeichert (das waere eine allokation je
    # aufruf), sondern beim abstieg neu erlaufen: sie ist hoechstens drei
    # glieder lang (mond -> planet -> stern), also O(tiefe^2) zeigerschritte.
    chain_count = 0
    cur = index
    memo_hit = 0
    hit_x = 0.0
    hit_y = 0.0

    while cur >= 0 and cur < n and chain_count < n:
        # Ein vorfahr, der zu DIESER zeit schon berechnet wurde, beendet
        # den aufstieg: sein absolutwert ist die basis, auf die die
        # restlichen glieder addiert werden -- dieselbe summe in derselben
        # reihenfolge wie ohne treffer.
        if use_memo:
            col = _memo_find(body_memo, cur, local_t, slots)
            if col >= 0:
                memo_hit = 1
                hit_x = body_memo[cur, col + 1]
                hit_y = body_memo[cur, col + 2]
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
        # Der koerper, der chain_pos schritte UEBER `index` liegt.
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
            _memo_store(body_memo, child, local_t, wx, wy, slots)

    return wx, wy



@njit(cache=True, nogil=True, fastmath=True)
def _table_knot_col(index, kj, body_memo):
    """Spalte des tafelknotens `kj`, oder -1, wenn er (noch) nicht da ist."""
    col = MEMO_TAB + 3 + 3 * (kj % MEMO_TAB_KNOTS)
    if body_memo[index, col] != float(kj) + MEMO_TAB_OFFSET:
        return -1
    return col


@njit(cache=True, nogil=True, fastmath=True)
def _table_fill_numba(
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
    """Die fehlenden der vier knoten um `local_t` EXAKT rechnen und ablegen."""
    h = body_memo[index, MEMO_TAB]
    k = int(math.floor(local_t * body_memo[index, MEMO_TAB + 1]))
    for j in range(4):
        kj = k - 1 + j
        if _table_knot_col(index, kj, body_memo) >= 0:
            continue
        kx, ky = _body_position_at_time_numba(
            index, float(kj) * h, body_x, body_y, body_m, body_scripted,
            body_a, body_e, body_theta, body_arg, body_parent, G, body_memo,
        )
        col = MEMO_TAB + 3 + 3 * (kj % MEMO_TAB_KNOTS)
        body_memo[index, col] = float(kj) + MEMO_TAB_OFFSET
        body_memo[index, col + 1] = kx
        body_memo[index, col + 2] = ky


@njit(cache=True, nogil=True, fastmath=True)
def _table_interp_numba(index, local_t, body_memo):
    """Lagrange-kubik ueber die knoten k-1 .. k+2; (x, y, ok).

    Fehler hoechstens (9/16)/24 * h^4 * max|d4x/dt4| -- aus diesem wert leitet
    `propagate._setup_planet_table` die nah-schwelle ab, unterhalb der exakt
    gerechnet wird. ok = 0, wenn ein knoten fehlt.
    """
    u = local_t * body_memo[index, MEMO_TAB + 1]
    k = int(math.floor(u))
    s = u - k
    c0 = _table_knot_col(index, k - 1, body_memo)
    c1 = _table_knot_col(index, k, body_memo)
    c2 = _table_knot_col(index, k + 1, body_memo)
    c3 = _table_knot_col(index, k + 2, body_memo)
    if c0 < 0 or c1 < 0 or c2 < 0 or c3 < 0:
        return 0.0, 0.0, 0
    w0 = -s * (s - 1.0) * (s - 2.0) / 6.0
    w1 = (s + 1.0) * (s - 1.0) * (s - 2.0) / 2.0
    w2 = -(s + 1.0) * s * (s - 2.0) / 2.0
    w3 = (s + 1.0) * s * (s - 1.0) / 6.0
    x = (w0 * body_memo[index, c0 + 1] + w1 * body_memo[index, c1 + 1]
         + w2 * body_memo[index, c2 + 1] + w3 * body_memo[index, c3 + 1])
    y = (w0 * body_memo[index, c0 + 2] + w1 * body_memo[index, c1 + 2]
         + w2 * body_memo[index, c2 + 2] + w3 * body_memo[index, c3 + 2])
    return x, y, 1


@njit(cache=True, nogil=True, fastmath=True)
def _memo_kepler_rel(index, local_t, body_memo):
    """`_kepler_rel_consts` mit den vorberechneten groessen aus dem notizblock."""
    return _kepler_rel_consts(
        local_t, body_memo[index, MEMO_A], body_memo[index, MEMO_E],
        body_memo[index, 4], body_memo[index, 5], body_memo[index, 6],
        body_memo[index, 7], body_memo[index, 8],
    )


@njit(cache=True, nogil=True, fastmath=True)
def _setup_body_kinds(body_memo, body_x, body_scripted, body_a, body_e,
                      body_parent):
    """Je koerper, auf welchem weg `_place_bodies_numba` ihn aufstellt.

    0 = fest (kein eigener bahnweg: die Sonne), 1 = planet (umlaeuft einen
    festen koerper), 2 = mond (umlaeuft einen planeten), 3 = alles andere --
    das rechnet weiter `_body_position_at_time_numba`. Die einteilung bildet
    genau die elternkette nach, die jene funktion hinaufsteigt, und setzt die
    gueltigen bahngroessen (spalte 9 = 1) voraus.
    """
    n = body_x.shape[0]
    for i in range(n):
        body_memo[i, MEMO_A] = body_a[i]
        body_memo[i, MEMO_E] = body_e[i]
        p = body_parent[i]
        body_memo[i, MEMO_PARENT] = float(p)
        own = body_scripted[i] != 0 and body_a[i] > 0.0 and p >= 0 and p < n
        if not own:
            body_memo[i, MEMO_KIND] = 0.0
            continue
        body_memo[i, MEMO_KIND] = 3.0
        if body_memo[i, 9] <= 0.0:
            continue
        pp = body_parent[p]
        parent_own = body_scripted[p] != 0 and body_a[p] > 0.0 and pp >= 0 and pp < n
        if not parent_own:
            body_memo[i, MEMO_KIND] = 1.0
            continue
        if body_memo[p, 9] <= 0.0:
            continue
        ppp = body_parent[pp]
        grand_own = body_scripted[pp] != 0 and body_a[pp] > 0.0 and ppp >= 0 and ppp < n
        if not grand_own:
            body_memo[i, MEMO_KIND] = 2.0


@njit(cache=True, nogil=True, fastmath=True)
def _place_bodies_numba(
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
    body_memo,
):
    """Alle quellen EINMAL zur zeit `local_t` aufstellen, in MEMO_POS.

    Der schnelle weg fuer einen notizblock in voller breite: die kraft- und
    zeitskalen-schleifen lesen danach nur noch MEMO_POS. Er rechnet planeten
    und monde selbst (`_memo_kepler_rel`, nur skalare) statt ueber
    `_body_position_at_time_numba`, weil jeder aufruf dorthin ~120 ns an
    referenzzaehlung kostet (siehe `_kepler_rel_consts`) -- bei gleichem
    rechenweg und gleicher summationsreihenfolge, also bit-identisch.

    Dazu die beiden naeherungen, die der lauf einschalten kann: ferne
    planeten aus der tafel (MEMO_TAB) und ferne mondsysteme als ein koerper
    (MEMO_GROUP, der planet steht im koerper-array vor seinen monden).
    MEMO_POS + 2 = 1 heisst: dieser mond zieht nicht einzeln.
    """
    n = body_x.shape[0]
    for i in range(n):
        body_memo[i, MEMO_POS + 2] = 0.0
        if body_fixed[i] == 0:
            continue
        root = int(body_memo[i, MEMO_GROUP]) - 1
        if root >= 0 and body_memo[root, MEMO_GROUP + 3] != 0.0:
            body_memo[i, MEMO_POS + 2] = 1.0
            continue

        placed = 0
        px = 0.0
        py = 0.0
        if body_memo[i, MEMO_TAB] > 0.0:
            px, py, ok = _table_interp_numba(i, local_t, body_memo)
            if ok == 0:
                _table_fill_numba(i, local_t, body_x, body_y, body_m,
                                  body_scripted, body_a, body_e, body_theta,
                                  body_arg, body_parent, G, body_memo)
                px, py, ok = _table_interp_numba(i, local_t, body_memo)
            if ok != 0:
                tdx = px - x
                tdy = py - y
                if tdx * tdx + tdy * tdy >= body_memo[i, MEMO_TAB + 2]:
                    placed = 1

        if placed == 0:
            col = _memo_find(body_memo, i, local_t, MEMO_SLOTS)
            if col >= 0:
                px = body_memo[i, col + 1]
                py = body_memo[i, col + 2]
            else:
                kind = int(body_memo[i, MEMO_KIND])
                if kind == 0:
                    px = body_x[i]
                    py = body_y[i]
                elif kind == 1:
                    p = int(body_memo[i, MEMO_PARENT])
                    rx, ry, ok = _memo_kepler_rel(i, local_t, body_memo)
                    if ok != 0:
                        px = body_x[p]
                        py = body_y[p]
                        px += rx
                        py += ry
                        _memo_store(body_memo, i, local_t, px, py, MEMO_SLOTS)
                    else:
                        px = body_x[i]
                        py = body_y[i]
                elif kind == 2:
                    p = int(body_memo[i, MEMO_PARENT])
                    cp = _memo_find(body_memo, p, local_t, MEMO_SLOTS)
                    okp = 1
                    if cp >= 0:
                        bx = body_memo[p, cp + 1]
                        by = body_memo[p, cp + 2]
                    else:
                        g = int(body_memo[p, MEMO_PARENT])
                        rx, ry, okp = _memo_kepler_rel(p, local_t, body_memo)
                        bx = body_x[g]
                        by = body_y[g]
                        if okp != 0:
                            bx += rx
                            by += ry
                            _memo_store(body_memo, p, local_t, bx, by, MEMO_SLOTS)
                    if okp != 0:
                        rx, ry, ok = _memo_kepler_rel(i, local_t, body_memo)
                        if ok != 0:
                            px = bx + rx
                            py = by + ry
                            _memo_store(body_memo, i, local_t, px, py, MEMO_SLOTS)
                        else:
                            px = body_x[i]
                            py = body_y[i]
                    else:
                        px = body_x[i]
                        py = body_y[i]
                else:
                    px, py = _body_position_at_time_numba(
                        i, local_t, body_x, body_y, body_m, body_scripted,
                        body_a, body_e, body_theta, body_arg, body_parent, G,
                        body_memo,
                    )

        body_memo[i, MEMO_POS] = px
        body_memo[i, MEMO_POS + 1] = py
        if body_memo[i, MEMO_GROUP + 2] > 0.0:
            dx = px - x
            dy = py - y
            body_memo[i, MEMO_GROUP + 3] = (
                1.0 if dx * dx + dy * dy > body_memo[i, MEMO_GROUP + 2] else 0.0)
