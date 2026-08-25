"""Bahn-linien der himmelskoerper -- reine geometrie, kein GL.

Zwei kurven je koerper, beide in WELT-koordinaten:

* die volle kepler-ellipse (`ellipse_offsets`), elternrelativ und damit
  zeitlos -- sie haengt nur an (a, e, arg_periapsis) und wird gecacht;
* die zukunfts-spur (`future_track`), also wo der koerper waehrend des
  prognose-fensters wirklich stehen wird.

Die deckkraft beider kommt aus der dichtesten annaeherung zwischen der
prognostizierten schiffsbahn und der ZUKUENFTIGEN position des koerpers zur
JEWEILS GLEICHEN zeit -- nicht aus dem abstand zur bahnlinie selbst.

Wie `body_style.py`: reines numpy, damit der ganze block headless testbar
bleibt. Das zeichnen liegt in `rendering.py`.
"""

import math
import time

import numpy as np

from vec import G


def future_track(body, times, memo=None):
    """Wo `body` zu den zeiten `times` stehen wird, (k, 2) in weltkoordinaten.

    Stapelfassung von `bodies.body.position_at_time`, OPERATION FUER
    OPERATION -- dieselben zwischengroessen in derselben reihenfolge. Das
    ist kein selbstzweck: genau dieses modell (konstante winkelrate ab dem
    epochen-lesezeichen) benutzt der praediktor fuer die schwerkraft, also
    trifft die gezeichnete linie die hier gezeichnete spur auch wirklich.
    Eine "genauere" kepler-loesung waere hier die FALSCHE antwort.

    `memo` ist ein dict fuer EINEN durchlauf mit EINEM `times`-array: die
    elternkette wird darin abgelegt, damit Saturn nicht einmal je mond neu
    geloest wird. Ein fremdes `times` leert es.
    """
    times = np.ascontiguousarray(times, dtype=np.float64)
    if memo is None:
        memo = {}
    elif memo.get('_times_id') != id(times):
        memo.clear()
    memo['_times_id'] = id(times)

    key = id(body)
    hit = memo.get(key)
    if hit is not None:
        return hit

    track = _track_uncached(body, times, memo)
    memo[key] = track
    return track


def _constant_track(body, times):
    out = np.empty((times.shape[0], 2), dtype=np.float64)
    out[:, 0] = float(body.position.x)
    out[:, 1] = float(body.position.y)
    return out


def _track_uncached(body, times, memo):
    # Reihenfolge der abbrueche wie im skalaren original.
    parent = getattr(body, 'is_moon_of', None)
    if parent is None:
        return _constant_track(body, times)

    a = getattr(body, 'semi_major_axis', None)
    if a is None or a == 0.0:
        return _constant_track(body, times)

    a = float(a)
    e = float(body.eccentricity) if body.eccentricity else 0.0
    mu = G * parent.mass
    if mu <= 0.0:
        return _constant_track(body, times)

    ref_theta = body._kepler_ref_theta
    delta_t = times - body._kepler_ref_time

    # Skalar bleibt skalar: r_ref/v_ref/omega_ref haengen nicht an der zeit,
    # und `math.*` liefert hier dieselben bits wie das original.
    r_ref = a * (1.0 - e * e) / (1.0 + e * math.cos(ref_theta))
    v_ref = math.sqrt(max(0.0, mu * (2.0 / r_ref - 1.0 / a)))
    omega_ref = v_ref / max(1e-12, r_ref)
    theta_t = ref_theta + omega_ref * delta_t

    r_t = a * (1.0 - e * e) / (1.0 + e * np.cos(theta_t))
    x_orb = r_t * np.cos(theta_t)
    y_orb = r_t * np.sin(theta_t)

    c = math.cos(body.arg_periapsis)
    s = math.sin(body.arg_periapsis)
    out = np.empty((times.shape[0], 2), dtype=np.float64)
    out[:, 0] = x_orb * c - y_orb * s
    out[:, 1] = x_orb * s + y_orb * c

    if hasattr(parent, 'position_at_time'):
        out += future_track(parent, times, memo)
    else:
        out[:, 0] += float(parent.position.x)
        out[:, 1] += float(parent.position.y)
    return out


def soi_radius(body):
    """Radius der einflusssphaere, `a * (m / m_elter)^0.4`, oder None.

    Der massstab, an dem "nah" gemessen wird. Er ist massenbewusst: ein
    mond und ein brocken gleicher groesse im selben abstand sind eben
    NICHT gleich relevant. Ohne elter (die Sonne) gibt es weder bahn noch
    SOI -- solche koerper bekommen gar keine linie.
    """
    parent = getattr(body, 'is_moon_of', None)
    if parent is None:
        return None
    a = getattr(body, 'semi_major_axis', None)
    if a is None:
        return None
    a = float(a)
    if a <= 0.0:
        return None
    m = float(getattr(body, 'mass', 0.0) or 0.0)
    m_parent = float(getattr(parent, 'mass', 0.0) or 0.0)
    if m <= 0.0 or m_parent <= 0.0:
        return None
    return a * (m / m_parent) ** 0.4


def approach_alpha(miss, soi, soi_full, soi_fade, alpha_max, floor):
    """Deckkraft aus der dichtesten annaeherung, in SOI-vielfachen.

    `miss <= soi_full * soi` -> `alpha_max`, `miss >= soi_fade * soi` ->
    `floor`, dazwischen ein smoothstep. Der boden gewinnt immer, damit die
    linie nie ganz verschwindet (und die des referenzkoerpers deutlich
    sichtbar bleibt).
    """
    floor = float(floor)
    if soi is None or not (soi > 0.0):
        return floor
    if not math.isfinite(miss):
        return floor

    lo = float(soi_full)
    hi = float(soi_fade)
    if hi <= lo:
        return alpha_max if miss <= lo * soi else floor

    u = (hi - (float(miss) / float(soi))) / (hi - lo)
    if u <= 0.0:
        return floor
    if u >= 1.0:
        return max(floor, float(alpha_max))
    return max(floor, float(alpha_max) * u * u * (3.0 - 2.0 * u))


def closest_approach(sample_t, ship_xy, body_xy):
    """Kleinster abstand zwischen schiff und koerper ZUR GLEICHEN ZEIT.

    Alle drei arrays sind gleich lang und auf dieselben zeiten bezogen.
    Gibt `(abstand, zeitpunkt)` zurueck; bei leerer eingabe `(inf, nan)`,
    damit `approach_alpha` sauber auf den boden faellt.

    Verfeinert wird auf dem QUADRAT des abstands, nicht auf dem abstand:
    bei geradliniger relativbewegung ist d^2 exakt eine parabel in t, die
    scheitelpunkt-formel also exakt -- waehrend d selbst eine hyperbel ist.
    Das kostet nichts und holt den fehler weg, den die grobe abtastung
    sonst hinterlaesst (gemessen: ein faktor >1000 beim vorbeiflug).
    """
    n = int(sample_t.shape[0])
    if n == 0:
        return float('inf'), float('nan')

    dx = ship_xy[:, 0] - body_xy[:, 0]
    dy = ship_xy[:, 1] - body_xy[:, 1]
    d2 = dx * dx + dy * dy
    i = int(np.argmin(d2))

    raw = math.sqrt(float(d2[i]))
    if i <= 0 or i >= n - 1:
        # Kein nachbarpaar: das minimum liegt am rand des fensters, dort
        # gibt es nichts zu interpolieren.
        return raw, float(sample_t[i])

    t0 = float(sample_t[i - 1]); y0 = float(d2[i - 1])
    t1 = float(sample_t[i]);     y1 = float(d2[i])
    t2 = float(sample_t[i + 1]); y2 = float(d2[i + 1])

    # Scheitelpunkt der parabel durch die drei (moeglicherweise ungleich
    # weit auseinanderliegenden) stuetzstellen. Die praediktor-punkte sind
    # nach BOGENLAENGE gesetzt, ihre zeitschritte also nicht uniform.
    d01 = t0 - t1
    d21 = t2 - t1
    denom = d01 * d21 * (d01 - d21)
    if denom == 0.0:
        return raw, t1
    a2 = (d21 * (y0 - y1) - d01 * (y2 - y1)) / denom
    b2 = (d01 * d01 * (y2 - y1) - d21 * d21 * (y0 - y1)) / denom
    if not (a2 > 0.0):
        # Nach oben offen ist die bedingung fuer ein minimum; sonst war die
        # stichprobe schon der beste wert.
        return raw, t1

    dt = -0.5 * b2 / a2
    if not (d01 <= dt <= d21 or d21 <= dt <= d01):
        return raw, t1

    y_min = y1 + b2 * dt + a2 * dt * dt
    if y_min < 0.0:
        y_min = 0.0
    return math.sqrt(y_min), t1 + dt


class OrbitLineEntry:
    """Was zu EINEM koerper je bild bekannt ist."""

    __slots__ = ('body', 'alpha', 'target', 'floor', 'reveal',
                 'reveal_target', 'miss', 't_min',
                 'track', 'track_t', 'track_len')

    def __init__(self, body):
        self.body = body
        # Von null hochblenden: beim ersten auftauchen soll die linie
        # erscheinen, nicht einfach dastehen.
        self.alpha = 0.0
        self.target = 0.0
        self.floor = 0.0
        self.reveal = 0.0
        self.reveal_target = 0.0
        self.miss = float('inf')
        self.t_min = float('nan')
        self.track = None
        self.track_t = None
        self.track_len = 0.0


class OrbitLineSet:
    """Deckkraft und zukunfts-spur aller koerper, ueber die frames gehalten.

    Teilt die arbeit in zwei takte, weil sie sehr verschieden teuer ist:

    * NEUBERECHNUNG (spuren + dichteste annaeherung) nur, wenn der
      praediktor eine neue linie geliefert hat oder die simulationszeit um
      mehr als einen stichproben-schritt vorgerueckt ist. Letzteres skaliert
      sich selbst mit dem horizont und faengt den zeitraffer-halt ab, in dem
      die generation stillsteht.
    * JE BILD nur das nachziehen der deckkraft (1 - exp(-rate*dt), wie
      ueberall im projekt) und das neu-setzen der boeden -- referenz- und
      auswahlkoerper duerfen wechseln, ohne dass irgendetwas neu gerechnet
      wird. Genau das macht den groben takt unsichtbar.
    """

    def __init__(self, *, track_samples=192, soi_full=1.0, soi_fade=3.0,
                 reveal_full=10.0, reveal_fade=30.0,
                 alpha_max=0.85, alpha_floor=0.10, alpha_floor_focus=0.35,
                 fade_rate=6.0):
        self.track_samples = max(8, int(track_samples))
        self.soi_full = float(soi_full)
        self.soi_fade = float(soi_fade)
        self.reveal_full = float(reveal_full)
        self.reveal_fade = float(reveal_fade)
        self.alpha_max = float(alpha_max)
        self.alpha_floor = float(alpha_floor)
        self.alpha_floor_focus = float(alpha_floor_focus)
        self.fade_rate = float(fade_rate)

        self._entries = {}
        self._last_key = None
        self._last_sim_time = None
        self._sample_step = 0.0
        self.recomputes = 0
        self.last_recompute_ms = 0.0

    # -- abfragen -----------------------------------------------------
    def get(self, body):
        return self._entries.get(id(body))

    def alpha(self, body):
        entry = self._entries.get(id(body))
        return 0.0 if entry is None else entry.alpha

    def target_alpha(self, body):
        entry = self._entries.get(id(body))
        return 0.0 if entry is None else entry.target

    def reveal(self, body):
        entry = self._entries.get(id(body))
        return 0.0 if entry is None else entry.reveal

    def miss(self, body):
        entry = self._entries.get(id(body))
        return float('inf') if entry is None else entry.miss

    def entries(self):
        return self._entries.values()

    # -- takt ---------------------------------------------------------
    def update(self, bodies, points, sim_time, real_dt,
               reference_body=None, selected_body=None, generation=None):
        if self._needs_recompute(points, sim_time, generation):
            self._recompute(bodies, points)
            self._last_sim_time = float(sim_time)
        self._retarget(reference_body, selected_body)
        self._ease(real_dt)

    def _needs_recompute(self, points, sim_time, generation):
        count = 0 if points is None else int(np.asarray(points).shape[0])
        key = (generation, None if points is None else id(points), count)
        if key != self._last_key:
            self._last_key = key
            return True
        if self._last_sim_time is None:
            return True
        if self._sample_step > 0.0:
            return abs(float(sim_time) - self._last_sim_time) >= self._sample_step
        return False

    def _recompute(self, bodies, points):
        t0 = time.perf_counter()
        self.recomputes += 1

        sample_t = None
        ship_xy = None
        self._sample_step = 0.0
        if points is not None:
            arr = np.asarray(points, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] >= 3 and arr.shape[0] >= 2:
                n = arr.shape[0]
                k = min(self.track_samples, n)
                # Stichproben AUF den praediktor-punkten, nicht dazwischen:
                # so ist die schiffsposition zur stichprobenzeit exakt und
                # muss nicht interpoliert werden.
                idx = np.unique(np.linspace(0, n - 1, k).astype(np.int64))
                sample_t = np.ascontiguousarray(arr[idx, 2])
                ship_xy = np.ascontiguousarray(arr[idx, 0:2])
                span = float(sample_t[-1] - sample_t[0])
                self._sample_step = abs(span) / max(1, idx.size - 1)

        memo = {}
        active = set()
        for b in bodies:
            if getattr(b, 'is_ship', False):
                continue
            if soi_radius(b) is None:
                continue
            body_id = id(b)
            active.add(body_id)
            entry = self._entries.get(body_id)
            if entry is None:
                entry = OrbitLineEntry(b)
                self._entries[body_id] = entry
            entry.body = b

            if sample_t is None:
                entry.track = None
                entry.track_t = None
                entry.track_len = 0.0
                entry.miss = float('inf')
                entry.t_min = float('nan')
                continue

            track = future_track(b, sample_t, memo)
            entry.track = track
            entry.track_t = sample_t
            # Bogenlaenge fuer die zeichen-aufloesung -- einmal hier statt
            # je bild im renderer.
            d = np.diff(track, axis=0)
            entry.track_len = float(np.sum(np.hypot(d[:, 0], d[:, 1]))) if d.size else 0.0
            entry.miss, entry.t_min = closest_approach(sample_t, ship_xy, track)

        for stale in [key for key in self._entries if key not in active]:
            del self._entries[stale]

        self.last_recompute_ms = (time.perf_counter() - t0) * 1000.0

    def _retarget(self, reference_body, selected_body):
        focus = set()
        if reference_body is not None:
            focus.add(id(reference_body))
        if selected_body is not None:
            focus.add(id(selected_body))
        for body_id, entry in self._entries.items():
            is_focus = body_id in focus
            floor = self.alpha_floor_focus if is_focus else self.alpha_floor
            entry.floor = floor
            soi = soi_radius(entry.body)
            entry.target = approach_alpha(
                entry.miss, soi, self.soi_full, self.soi_fade,
                self.alpha_max, floor)
            # Referenz- und auswahlkoerper sind immer kandidaten: sie sind
            # das, worauf der spieler ausdruecklich gezeigt hat.
            entry.reveal_target = 1.0 if is_focus else reveal_fraction(
                entry.miss, soi, self.reveal_full, self.reveal_fade)

    def _ease(self, real_dt):
        dt = float(real_dt)
        if dt <= 0.0:
            return
        f = 1.0 - math.exp(-self.fade_rate * dt)
        for entry in self._entries.values():
            entry.alpha += (entry.target - entry.alpha) * f
            entry.reveal += (entry.reveal_target - entry.reveal) * f


# Toleranz, ab der eine frame-transformation nicht mehr als starr
# durchgeht. 1e-9 ist grosszuegig gegen rundung (erwartet ~1e-16) und eng
# genug, dass eine echte skalierung oder scherung auffliegt.
_RIGID_TOL = 1e-9


def frame_affine_at(frame, t):
    """Die starre transformation des frames zur zeit `t`.

    Gibt `(r00, r01, r10, r11, tx, ty)` mit `f(p) = R p + t`, oder None,
    wenn `R` keine drehung ist -- dann rechnet der aufrufer punktweise
    weiter, statt etwas falsches zu rechnen.

    Drei sondierungen genuegen, weil die abbildung affin ist. Die
    sondierlaenge `L` haengt an der GROESSE des ursprungs: `f(0)` ist bei
    einem koerperzentrierten rahmen die (negative) position des
    bezugskoerpers, also bis zu 1e12 m. Eine sondierung bei L = 1 wuerde
    diese zahl von sich selbst abziehen und nur noch rundungsrauschen
    uebrig lassen -- gemessen 1e-4 relativ statt 1e-16.
    """
    to_xy = getattr(frame, 'to_this_frame_xy', None)
    if to_xy is None:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    t = float(t)
    tx, ty = to_xy(t, 0.0, 0.0)
    tx = float(tx)
    ty = float(ty)
    L = max(1.0, math.hypot(tx, ty))

    ax, ay = to_xy(t, L, 0.0)
    bx, by = to_xy(t, 0.0, L)
    r00 = (float(ax) - tx) / L
    r10 = (float(ay) - ty) / L
    r01 = (float(bx) - tx) / L
    r11 = (float(by) - ty) / L

    # Starrheit nachweisen, nicht annehmen.
    if (abs(r00 * r00 + r10 * r10 - 1.0) > _RIGID_TOL
            or abs(r01 * r01 + r11 * r11 - 1.0) > _RIGID_TOL
            or abs(r00 * r01 + r10 * r11) > _RIGID_TOL):
        return None
    return (r00, r01, r10, r11, tx, ty)


def frame_project(frame, times, xs, ys, cache=None):
    """Weltpunkte in den plot-frame, je ZEIT statt je PUNKT.

    Die ellipse besteht aus hunderten punkten zu EINER zeit und die
    zukunfts-spuren aller koerper teilen sich DASSELBE zeit-array. Punktweise
    ausgewertet ist das hundertfach dieselbe rechnung: gemessen ~27 000
    skalare frame-aufrufe je bild und 17 ms, weil
    `to_this_frame_xy_arrays` fuer zeiten ausserhalb seines
    knoten-fensters None liefert und die schleife uebernimmt.

    Hier wird stattdessen `frame_affine_at` je EINDEUTIGER zeit bestimmt
    und dann in numpy angewandt. Das ist zugleich genauer als der
    stapelweg des rahmens, der den ursprung zwischen knoten kubisch
    interpoliert -- hier wird er exakt ausgewertet.

    `cache` ist ein dict fuer EIN bild: die koerper-positionen wandern mit
    der welt, die transformation gilt also nur innerhalb eines bildes.
    """
    xs = np.ascontiguousarray(xs, dtype=np.float64)
    ys = np.ascontiguousarray(ys, dtype=np.float64)
    n = int(xs.shape[0])
    if n == 0:
        return xs.copy(), ys.copy()

    # Skalare zeit (die ellipse gilt zu EINEM zeitpunkt): eine
    # transformation, keine tabelle, kein np.unique. Ausdruecklich
    # ausgeschrieben statt sich auf das broadcasting eines 0-d-arrays zu
    # verlassen -- das haengt an der numpy-fassung.
    if np.isscalar(times) or np.asarray(times).ndim == 0:
        t = float(times)
        got = cache.get((id(frame), t)) if cache is not None else None
        if got is None:
            try:
                got = frame_affine_at(frame, t)
            except Exception:
                got = None
            if got is None:
                return _frame_project_scalar(
                    frame, np.full(n, t, dtype=np.float64), xs, ys)
            if cache is not None:
                cache[(id(frame), t)] = got
        r00, r01, r10, r11, tx, ty = got
        return (tx + r00 * xs + r01 * ys, ty + r10 * xs + r11 * ys)

    frame_key = id(frame)
    times = np.ascontiguousarray(times, dtype=np.float64)

    # Der cache haelt die FERTIG AUFGEZOGENE koeffiziententabelle, nicht
    # einzelne transformationen je zeit. Alle koerper stehen auf derselben
    # zeitachse, also baut sie der erste und die restlichen 25 lesen sie --
    # gemessen 1811 us fuer den ersten und danach 6 us statt 113 us. Die
    # tabelle je koerper neu zusammenzusetzen war eine Python-schleife ueber
    # alle stichprobenzeiten und hat 5.0 ms je bild gekostet.
    key = None
    if cache is not None and times.shape[0] == n:
        key = ('coef', frame_key, id(times), n,
               float(times[0]), float(times[-1]))
        hit = cache.get(key)
        if hit is not None:
            a, b, c, d, tx, ty = hit
            return (tx + a * xs + b * ys, ty + c * xs + d * ys)

    uniq, inverse = np.unique(times, return_inverse=True)
    rows = []
    for t in uniq.tolist():
        got = None
        if cache is not None:
            got = cache.get((frame_key, t))
        if got is None:
            try:
                got = frame_affine_at(frame, t)
            except Exception:
                got = None
            if got is None:
                return _frame_project_scalar(frame, times, xs, ys)
            if cache is not None:
                cache[(frame_key, t)] = got
        rows.append(got)

    # In EINEM zug in ein array statt sechs numpy-skalarzuweisungen je
    # zeit -- das war der eigentliche kostenpunkt.
    table = np.asarray(rows, dtype=np.float64)[inverse.reshape(-1)]
    a = np.ascontiguousarray(table[:, 0])
    b = np.ascontiguousarray(table[:, 1])
    c = np.ascontiguousarray(table[:, 2])
    d = np.ascontiguousarray(table[:, 3])
    tx = np.ascontiguousarray(table[:, 4])
    ty = np.ascontiguousarray(table[:, 5])
    if key is not None:
        cache[key] = (a, b, c, d, tx, ty)
    return (tx + a * xs + b * ys, ty + c * xs + d * ys)

def _frame_project_scalar(frame, times, xs, ys):
    """Punktweise rueckfallebene -- langsam, aber immer richtig."""
    out_x = np.empty(xs.shape[0], dtype=np.float64)
    out_y = np.empty(ys.shape[0], dtype=np.float64)
    scalar = getattr(frame, 'to_this_frame_xy', None)
    if scalar is None:
        return xs.copy(), ys.copy()
    for i in range(xs.shape[0]):
        try:
            fx, fy = scalar(float(times[i]), float(xs[i]), float(ys[i]))
        except Exception:
            fx, fy = float(xs[i]), float(ys[i])
        out_x[i] = fx
        out_y[i] = fy
    return out_x, out_y


# Nur teilerfreundliche werte: `track[::stride]` zweier verschiedener
# strides soll sich moeglichst decken, denn der transformations-cache liegt
# auf (rahmen, ZEIT). Zwei koerper mit stride 4 und 8 teilen sich so jede
# zweite auswertung, statt zwei disjunkte saetze zu bezahlen.
STRIDE_LADDER = (1, 2, 4, 8, 16, 32, 64)


def polyline_stride(n_points, arc_len_px, r_px, view_diagonal_px, tolerance_px):
    """Groesster stride, mit dem die gezeichnete spur die toleranz haelt.

    Dieselbe rechnung wie `ellipse_segments`, nur andersherum: gegeben sind
    die stichproben, gesucht ist, wie viele davon ueberhaupt gezeichnet
    werden muessen. Die spur wird mit 192 punkten GEMESSEN, weil die
    dichteste annaeherung das braucht -- gezeichnet werden muss sie fast
    nie so fein, und jeder gezeichnete punkt kostet eine
    frame-transformation.
    """
    n_points = int(n_points)
    if n_points <= 2:
        return 1
    r_px = float(r_px)
    tol = max(1e-6, float(tolerance_px))
    diag = max(1.0, float(view_diagonal_px))
    arc = max(0.0, float(arc_len_px))

    if not math.isfinite(r_px) or r_px <= 0.0 or arc <= 0.0:
        return STRIDE_LADDER[-1]

    step = min(diag, math.sqrt(8.0 * r_px * tol))
    needed = int(math.ceil(arc / step)) + 1
    if needed >= n_points:
        return 1

    stride = (n_points - 1) // max(1, needed - 1)
    best = 1
    for candidate in STRIDE_LADDER:
        if candidate <= stride:
            best = candidate
        else:
            break
    return best


def stride_indices(n_points, stride):
    """Indizes `0, stride, 2*stride, ..., n-1` -- der letzte IMMER dabei.

    `arr[::stride]` verliert den schwanz, sobald `(n-1) % stride != 0`: bei
    192 punkten und stride 64 endet die kurve auf index 128 von 191, also
    fehlt ein drittel. Der letzte punkt ist aber genau der, auf den die
    ganze anzeige hinauslaeuft -- der koerper zur ENDZEIT des praediktors.
    Faellt er weg, zeigt die endkappe auf nichts.
    """
    n = int(n_points)
    if n <= 1:
        return np.zeros(1, dtype=np.int64) if n == 1 else np.zeros(0, dtype=np.int64)
    stride = max(1, int(stride))
    idx = np.arange(0, n, stride, dtype=np.int64)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def reveal_fraction(miss, soi, reveal_full, reveal_fade):
    """Wie VIEL der linie gezeichnet wird, 0..1 -- neben der deckkraft.

    Zweites, unabhaengiges band auf derselben gemessenen annaeherung. Es ist
    bewusst WEITER als das der helligkeit: die endkappe -- der koerper zur
    endzeit des praediktors -- ist das, womit gezielt wird, und sie muss
    schon da sein, waehrend man noch steuert. Naeher kommen aendert dann nur
    noch die helligkeit, nicht mehr die laenge.
    """
    if soi is None or not (soi > 0.0) or not math.isfinite(miss):
        return 0.0
    lo = float(reveal_full)
    hi = float(reveal_fade)
    if hi <= lo:
        return 1.0 if miss <= lo * soi else 0.0
    u = (hi - (float(miss) / float(soi))) / (hi - lo)
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return u * u * (3.0 - 2.0 * u)


def _cubic_4pt(p0, p1, p2, p3, s):
    """Kubik DURCH ALLE VIER knoten (Lagrange), ausgewertet auf [p1, p2].

    Wortgleich zu `reference_frames._BodyEphemerisMixin._cubic_4pt` -- diesselbe
    kurve, damit die bahnlinien nicht anders interpolieren als der ursprung
    der plot-frames es tut. Nicht Catmull-Rom: dessen sehnen-steigung ist auf
    einem kreis um `sin(t)/t` zu kurz und macht das schema 3. ordnung.
    """
    return (p0 * (-(s * (s - 1.0) * (s - 2.0)) / 6.0)
            + p1 * (((s + 1.0) * (s - 1.0) * (s - 2.0)) / 2.0)
            + p2 * (-((s + 1.0) * s * (s - 2.0)) / 2.0)
            + p3 * (((s + 1.0) * s * (s - 1.0)) / 6.0))


class FrameAffineTable:
    """Die starre transformation eines plot-frames ueber ein ZEITFENSTER.

    Alle koerper stehen jetzt auf demselben fenster (dem des praediktors),
    also lohnt es sich, die transformation EINMAL auf einem knotengitter zu
    bestimmen und fuer jeden koerper daraus zu interpolieren -- statt sie je
    stichprobenzeit neu zu sondieren.

    Gespeichert wird der WINKEL, nicht die matrix. Interpoliert man `cos` und
    `sin` getrennt, ist das ergebnis keine drehung mehr (die spalten
    verlieren ihre laenge); ueber den winkel bleibt sie es exakt, bei
    gleichem aufwand.
    """

    __slots__ = ('t0', 'q', 'theta', 'tx', 'ty', 'knots', 'probes', 'valid')

    def __init__(self, frame, t0, t1, knot_angle=0.05, knot_min=8, knot_max=256):
        self.valid = False
        self.probes = 0
        t0 = float(t0)
        t1 = float(t1)
        span = t1 - t0
        if not math.isfinite(span) or span <= 0.0:
            span = 0.0

        knots = self._knot_count(frame, t0, t1, knot_angle, knot_min, knot_max)
        if knots is None:
            return
        self.knots = knots

        # Ein knoten mehr auf jeder seite: die kubik braucht vier stuetzen je
        # intervall, und an den raendern gaebe es sonst keine.
        self.q = span / knots if knots > 0 else 0.0
        if self.q <= 0.0:
            self.q = 1.0
        self.t0 = t0

        n = knots + 3
        theta = np.empty(n, dtype=np.float64)
        tx = np.empty(n, dtype=np.float64)
        ty = np.empty(n, dtype=np.float64)
        for k in range(n):
            got = frame_affine_at(frame, t0 + (k - 1) * self.q)
            self.probes += 3
            if got is None:
                return
            r00, r01, r10, r11, ox, oy = got
            theta[k] = math.atan2(r10, r00)
            tx[k] = ox
            ty[k] = oy
        # Sprungfrei machen, sonst interpoliert die kubik ueber den
        # +pi/-pi-schnitt hinweg quer durch den kreis.
        self.theta = np.unwrap(theta)
        self.tx = tx
        self.ty = ty
        self.valid = True

    @staticmethod
    def _knot_count(frame, t0, t1, knot_angle, knot_min, knot_max):
        """Knotenzahl aus dem, was sich im fenster WIRKLICH dreht.

        Zwei winkel, und beide zaehlen: der rahmen selbst kann rotieren
        (richtungs-frame), und sein URSPRUNG laeuft um -- ein
        koerperzentrierter, nicht rotierender rahmen hat winkel 0 und
        trotzdem eine gekruemmte verschiebung. Nur den ersten zu messen
        gaebe dort 8 knoten fuer einen halben umlauf.
        """
        a = frame_affine_at(frame, t0)
        b = frame_affine_at(frame, 0.5 * (t0 + t1))
        c = frame_affine_at(frame, t1)
        if a is None or b is None or c is None:
            return None

        def sweep(p, q):
            # Winkel zwischen zwei verschiebungsvektoren; bei (fast) null
            # laenge gibt es keinen, dann zaehlt nur die drehung.
            n0 = math.hypot(p[4], p[5])
            n1 = math.hypot(q[4], q[5])
            if n0 <= 1e-9 or n1 <= 1e-9:
                return 0.0
            dot = (p[4] * q[4] + p[5] * q[5]) / (n0 * n1)
            return math.acos(max(-1.0, min(1.0, dot)))

        rot = abs(math.atan2(b[2], b[0]) - math.atan2(a[2], a[0]))
        rot += abs(math.atan2(c[2], c[0]) - math.atan2(b[2], b[0]))
        shift = sweep(a, b) + sweep(b, c)
        total = max(rot, shift)

        want = int(math.ceil(total / max(1e-6, float(knot_angle))))
        return max(int(knot_min), min(int(knot_max), max(1, want)))

    def project(self, times, xs, ys):
        """(fx, fy) fuer beliebige zeiten im fenster, vollstaendig in numpy.

        Das gitter beginnt EINEN knoten vor `t0`, damit auch das erste
        intervall vier stuetzen hat: array-index `j` traegt die knotenzeit
        `t0 + (j-1)*q`. Fuer ein `t` im intervall `k = floor(u)` sind das
        die indizes `k .. k+3`, mit `p1 = k+1` genau auf dem knoten -- und
        dort ist die basis exakt (0,1,0,0), der knotenwert wird also nicht
        um ein bit verschoben.
        """
        t = np.ascontiguousarray(times, dtype=np.float64)
        xs = np.ascontiguousarray(xs, dtype=np.float64)
        ys = np.ascontiguousarray(ys, dtype=np.float64)

        u = (t - self.t0) / self.q
        last = self.theta.shape[0] - 4
        i = np.clip(np.floor(u), 0, last).astype(np.int64)
        # Ausserhalb des fensters wird extrapoliert statt geklemmt -- das
        # passiert nur, wenn jemand zeiten aus einem anderen fenster
        # hereinreicht, und eine sichtbar falsche kurve ist besser als eine
        # unsichtbar geklemmte.
        s = u - i.astype(np.float64)

        th = _cubic_4pt(self.theta[i], self.theta[i + 1],
                        self.theta[i + 2], self.theta[i + 3], s)
        ox = _cubic_4pt(self.tx[i], self.tx[i + 1],
                        self.tx[i + 2], self.tx[i + 3], s)
        oy = _cubic_4pt(self.ty[i], self.ty[i + 1],
                        self.ty[i + 2], self.ty[i + 3], s)

        cos_a = np.cos(th)
        sin_a = np.sin(th)
        return (ox + cos_a * xs - sin_a * ys,
                oy + sin_a * xs + cos_a * ys)


def frame_origin_body(frame):
    """Der koerper, der im ursprung dieses plot-frames sitzt, oder None.

    Er bekommt KEINE bahnlinie: in seinem eigenen rahmen bewegt er sich
    definitionsgemaess nicht, eine linie fuer ihn zeigt also nur noch den
    unterschied zwischen zwei kepler-modellen (siehe die notiz in
    CLAUDE.md). Genau das war der fehlerbericht -- die Erde zog im
    Erd-rahmen eine bahn um die Sonne.

    `TargetBodyDirectionReferenceFrame` nennt ihn `target`, nicht `primary`;
    beide namen werden geprueft, in der reihenfolge ihrer eindeutigkeit.
    """
    if frame is None:
        return None
    for attr in ('primary_body', 'target_body', 'child_body'):
        body = getattr(frame, attr, None)
        if body is not None:
            return body
    return None
