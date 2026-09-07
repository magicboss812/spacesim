"""Die gezeichnete folge des plans: coast -> burn -> coast -> ...

WIE DIE KETTE LAEUFT. Fuer jeden knoten der reihe nach:

    1. den ZUSTAND zur zuendzeit von der bisherigen linie ablesen
       (kubische Hermite-auswertung, `state_on_curve`) -- fuer den ersten
       knoten ist das die vorhersagelinie des predictors, fuer jeden
       weiteren das ergebnis des vorigen knotens. GENAU DAS macht sie zu
       einer kette.
    2. den BRENNBOGEN integrieren (`_burn_arc_numba`)
    3. von dort weiter GLEITEN (`_compute_distance_points_rkn_numba`), bis
       zum naechsten knoten oder bis das punktbudget aufgebraucht ist

Es wird NICHTS neu propagiert, was der predictor schon gerechnet hat: der
abschnitt vor dem ersten knoten ist die vorhandene linie und wird nur
gelesen. Ein knoten jenseits des gezeichneten horizonts hat keinen
ablesbaren zustand -- die kette bricht dort ab, statt zu raten.

KOSTEN, UND WARUM SIE NEBENHER ANFALLEN. Ein neuaufbau ist eine kette aus
bis zu fuenf integrationen und kostet gemessen 7-15 ms. Er laeuft nur, wenn
der plan (`plan.version`), die grundbahn (`predictor._trajectory_version`)
oder die eingestellte reichweite sich bewegt haben -- und er laeuft in
einem ARBEITSTHREAD, weil er sonst genau die eingabe bezahlt, die ihn
ausloest: im hauptthread gerechnet fiel die bildrate beim ziehen eines
griffs von 100 auf 40. Wer die auftraege stellt und was dabei im
hauptthread bleiben MUSS, steht an `ManeuverPreview`.

Zeitrechnung: die kernel rechnen LOKAL zur schnappschuss-epoche
(`snapshot["sim_time"]`), die punktelisten hier aussen tragen ABSOLUTE
sim-zeit in spalte 2 -- dieselbe konvention wie `ship/predictor/`.
"""

import math
import time

import numpy as np

from physics.kernels.burn import _burn_arc_numba
from physics.kernels.propagate import _compute_distance_points_rkn_numba

from .plan import burn_direction_world, orbital_basis
from .profile import BurnProfile


def state_on_curve(points, t_abs):
    """Zustand `(x, y, vx, vy)` auf einer `(n,5)`-linie zur absoluten zeit.

    Kubische Hermite zwischen den beiden umgebenden stuetzstellen -- die
    linie IST eine stueckweise kubische kurve, ihre spalten 3/4 sind die
    tangenten (siehe physics/kernels/__init__.py, POINT_COLUMNS). Wo diese
    spalten NaN tragen (sehnen-kernel), faellt die auswertung auf linear
    zurueck; das ist kein fehlerfall, sondern die wahrheit ueber solche
    punkte.

    `None`, wenn `t_abs` ausserhalb der linie liegt.
    """
    if points is None:
        return None
    try:
        n = len(points)
    except Exception:
        return None
    if n < 2:
        return None

    t = float(t_abs)
    times = points[:, 2]
    if t < float(times[0]) or t > float(times[n - 1]):
        return None

    # searchsorted statt einer schleife: die linie hat bis zu 40 000 punkte
    # und die vorschau liest sie mehrmals je neuaufbau.
    i = int(np.searchsorted(times, t, side='right')) - 1
    i = max(0, min(i, n - 2))

    t0 = float(times[i])
    t1 = float(times[i + 1])
    dt = t1 - t0
    if dt <= 0.0:
        return (float(points[i, 0]), float(points[i, 1]),
                float(points[i, 3]), float(points[i, 4]))

    s = (t - t0) / dt
    p0x, p0y = float(points[i, 0]), float(points[i, 1])
    p1x, p1y = float(points[i + 1, 0]), float(points[i + 1, 1])
    v0x, v0y = float(points[i, 3]), float(points[i, 4])
    v1x, v1y = float(points[i + 1, 3]), float(points[i + 1, 4])

    if not (math.isfinite(v0x) and math.isfinite(v0y)
            and math.isfinite(v1x) and math.isfinite(v1y)):
        # Sehnenpunkt: geradlinig interpolieren, geschwindigkeit aus der
        # sehne. Eine kruemmung zu erfinden waere schlimmer als keine.
        return (p0x + (p1x - p0x) * s, p0y + (p1y - p0y) * s,
                (p1x - p0x) / dt, (p1y - p0y) / dt)

    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    px = h00 * p0x + h10 * dt * v0x + h01 * p1x + h11 * dt * v1x
    py = h00 * p0y + h10 * dt * v0y + h01 * p1y + h11 * dt * v1y

    d00 = 6.0 * s2 - 6.0 * s
    d10 = 3.0 * s2 - 4.0 * s + 1.0
    d01 = -6.0 * s2 + 6.0 * s
    d11 = 3.0 * s2 - 2.0 * s
    vx = (d00 * p0x + d01 * p1x) / dt + d10 * v0x + d11 * v1x
    vy = (d00 * p0y + d01 * p1y) / dt + d10 * v0y + d11 * v1y
    return (px, py, vx, vy)


def body_state_at(body, t_abs, h=1.0):
    """Position und geschwindigkeit eines koerpers zur absoluten zeit.

    Die geschwindigkeit kommt als ZENTRALE DIFFERENZ aus
    `position_at_time()`, nie aus `body.velocity`: skriptgefuehrte koerper
    behalten dort ihren ladewert 0, und die orbitale basis waere dann gegen
    einen stillstehenden planeten gerechnet. Dieselbe falle steht in
    `ui/hud/telemetry.py::body_velocity`.
    """
    if body is None:
        return (0.0, 0.0, 0.0, 0.0)
    try:
        here = body.position_at_time(float(t_abs))
        px, py = float(here.x), float(here.y)
    except Exception:
        try:
            px, py = float(body.position.x), float(body.position.y)
        except Exception:
            return (0.0, 0.0, 0.0, 0.0)
    if not getattr(body, 'scripted_orbit', False):
        try:
            return (px, py, float(body.velocity.x), float(body.velocity.y))
        except Exception:
            return (px, py, 0.0, 0.0)
    try:
        ahead = body.position_at_time(float(t_abs) + h)
        behind = body.position_at_time(float(t_abs) - h)
        return (px, py,
                (float(ahead.x) - float(behind.x)) / (2.0 * h),
                (float(ahead.y) - float(behind.y)) / (2.0 * h))
    except Exception:
        return (px, py, 0.0, 0.0)


class ManeuverPreview:
    """Die kette, ihre marker, und der weg, auf dem sie NEBENHER laeuft.

    NEBENLAEUFIG, weil sie sonst die eingabe bezahlt, die sie ausloest. Ein
    neuaufbau kostet gemessen 7-15 ms; im hauptthread gerechnet fiel die
    bildrate beim ziehen eines griffs von 100 auf 40. Alle beteiligten
    kernel sind `nogil=True` -- dieselbe voraussetzung, unter der schon die
    vorhersagelinie ausgelagert ist (`ship/predictor/jobs.py`).

    DIE ARBEITSTEILUNG IST DIE GANZE SCHWIERIGKEIT. Im hauptthread
    entsteht der AUFTRAG: schnappschuss, eine kopie der basislinie, je
    knoten zeit/delta-v und der zustand des bezugskoerpers zu dieser zeit.
    Alles davon liest Python-objekte (`world`, `body.position_at_time`),
    die der hauptthread im selben moment weiterschreibt. Der arbeiter
    bekommt danach nur noch arrays und zahlen und ruehrt kein objekt mehr
    an. Ein auftrag zur zeit; das ergebnis wird mit EINER zuweisung
    eingewechselt (`self.points = ...`), also nie halb sichtbar.

    Der riegel ist damit nicht mehr die uhr, sondern der auftrag selbst:
    solange einer laeuft, wird kein zweiter gestellt. Die auffrischrate ist
    dadurch 1/rechenzeit und STETIG -- der feste mindestabstand von vorher
    (0.15 s) rastete gegen die unregelmaessigen versions-spruenge des
    predictors und liess die linie ungleichmaessig nachziehen.
    """

    def __init__(self, max_points=1500, burn_step_s=0.25, burn_min_steps=16,
                 burn_max_steps=512, min_interval_s=0.0, burn_draw_points=24,
                 length_mult=1.0, length_mult_min=0.25, length_mult_max=64.0,
                 async_compute=True):
        self.max_points = max(64, int(max_points))
        self.burn_step_s = max(1e-3, float(burn_step_s))
        self.burn_min_steps = max(4, int(burn_min_steps))
        self.burn_max_steps = max(self.burn_min_steps, int(burn_max_steps))
        self.min_interval_s = max(0.0, float(min_interval_s))
        #: Wieviele punkte des brennbogens ueberhaupt in die GEZEICHNETE
        #: linie wandern. Der bogen wird mit hunderten schritten integriert
        #: -- das ist genauigkeit --, aber er ist ein paar sekunden lang und
        #: am schirm eine handbreit lange kruemmung. Alle schritte zu
        #: zeichnen verbrannte das punktbudget genau dort, wo es nichts
        #: bringt: gemessen 400 von 1200 punkten fuer einen 12-sekunden-
        #: bogen, worauf die anschliessende gleitphase zu kurz wurde, um den
        #: NAECHSTEN knoten noch zu erreichen -- die kette brach nach dem
        #: ersten glied ab.
        self.burn_draw_points = max(2, int(burn_draw_points))

        #: Reichweite der vorschau als VIELFACHES der punktdichte, die der
        #: predictor gerade fuer die vorhersagelinie benutzt.
        #:
        #: EIGENER REGLER, weil die vorschau eine andere frage stellt als
        #: die vorhersagelinie. Die linie zeigt, wo das schiff GLEICH ist --
        #: da will man aufloesung. Der plan zeigt, wo ein knoten den vierten
        #: umlauf HINLEGT -- da will man weite, und die genauigkeit
        #: dazwischen interessiert nicht. Beide an einen regler zu haengen
        #: hiesse, fuer weite immer aufloesung mitzukaufen.
        #:
        #: Er wirkt auf den PUNKTABSTAND (`precision`, eine bogenlaenge),
        #: nicht auf die punktzahl: die reichweite ist abstand x punkte, und
        #: nur der abstand ist ohne rechenkosten je punkt zu haben.
        self.length_mult_min = max(1e-3, float(length_mult_min))
        self.length_mult_max = max(self.length_mult_min, float(length_mult_max))
        self.length_mult = self.clamp_length_mult(length_mult)

        self.points = None
        self.node_markers = []
        self.valid = False
        self.last_rebuild_ms = 0.0
        #: Zeitspanne der gezeichneten kette in sim-sekunden -- der regler
        #: liest sie als seinen anzeigewert zurueck.
        self.span_seconds = None

        self.async_compute = bool(async_compute)
        self._executor = None
        self._future = None
        self._submitted_key = None
        self._plan_version = None
        self._trajectory_version = None
        self._last_wall = -1e18

    # ------------------------------------------------------------- reichweite

    def clamp_length_mult(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 1.0
        if not math.isfinite(value):
            return 1.0
        return max(self.length_mult_min, min(self.length_mult_max, value))

    def set_length_mult(self, value):
        """Reichweite setzen. Der naechste neuaufbau uebernimmt sie."""
        self.length_mult = self.clamp_length_mult(value)
        return self.length_mult

    # ---------------------------------------------------------------- riegel

    def invalidate(self):
        self._plan_version = None
        self._trajectory_version = None
        self._submitted_key = None

    def shutdown(self):
        executor = self._executor
        self._executor = None
        self._future = None
        if executor is not None:
            executor.shutdown(wait=False)

    def _ensure_executor(self):
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor
            # EIN arbeiter. Mehr braechte nichts: es laeuft ohnehin nur ein
            # auftrag zur zeit, und zwei ergebnisse gleichzeitig waeren zwei
            # antworten auf dieselbe frage.
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix='maneuver-preview')
        return self._executor

    def _state_key(self, plan, predictor):
        plan_version = int(getattr(plan, 'version', 0)) if plan is not None else -1
        traj_version = int(getattr(predictor, '_trajectory_version', 0))
        return (plan_version, traj_version, round(float(self.length_mult), 6))

    def _collect(self):
        """Ein fertiges ergebnis einwechseln. True, wenn eines kam."""
        future = self._future
        if future is None or not future.done():
            return False
        self._future = None
        try:
            result = future.result()
        except Exception:
            result = None
        if result is None:
            self.points = None
            self.node_markers = []
            self.valid = False
            self.span_seconds = None
            return True
        points, markers, elapsed_ms = result
        self.points = points
        self.node_markers = markers
        self.valid = True
        self.last_rebuild_ms = elapsed_ms
        self.span_seconds = (
            float(points[len(points) - 1, 2]) - float(points[0, 2])
            if points is not None and len(points) >= 2 else None)
        return True

    def wait(self, timeout=5.0):
        """Auf einen laufenden auftrag warten und ihn einwechseln.

        Nur fuer tests und abzuege: die hauptschleife wartet nie, sie holt
        das ergebnis im naechsten frame ab.
        """
        future = self._future
        if future is not None:
            try:
                future.result(timeout)
            except TypeError:
                pass
            except Exception:
                pass
        return self._collect()

    def maybe_rebuild(self, plan, ship, world, predictor, reference_body,
                      a_max, ramp_seconds, now_wall=None):
        """Nur rechnen, wenn sich etwas bewegt hat. True, wenn gerechnet wurde.

        Reihenfolge: erst ein fertiges ergebnis einwechseln, dann pruefen,
        ob der zustand seit dem ABGESCHICKTEN auftrag ein neuer ist. So
        holt ein zug am griff, der waehrend eines laufenden auftrags
        weitergeht, im naechsten frame von selbst nach.
        """
        collected = self._collect()
        if now_wall is None:
            now_wall = time.perf_counter()

        key = self._state_key(plan, predictor)
        if key == self._submitted_key:
            return collected
        if self._future is not None:
            # Einer zur zeit -- das IST der riegel.
            return collected
        if (float(now_wall) - self._last_wall) < self.min_interval_s:
            return collected

        job = self._make_job(plan, ship, world, predictor, reference_body,
                             a_max, ramp_seconds)
        self._last_wall = float(now_wall)
        self._submitted_key = key
        self._plan_version, self._trajectory_version = key[0], key[1]

        if job is None:
            self.points = None
            self.node_markers = []
            self.valid = False
            self.span_seconds = None
            return True

        if not self.async_compute:
            self._future = None
            result = _run_preview_job(job)
            self._future = _Done(result)
            self._collect()
            return True

        self._future = self._ensure_executor().submit(_run_preview_job, job)
        return True

    # ----------------------------------------------------------- neuaufbau

    def rebuild(self, plan, ship, world, predictor, reference_body,
                a_max, ramp_seconds):
        """SYNCHRON rechnen und sofort einwechseln (tests, screenshots)."""
        job = self._make_job(plan, ship, world, predictor, reference_body,
                             a_max, ramp_seconds)
        self._future = None
        if job is None:
            self.points = None
            self.node_markers = []
            self.valid = False
            self.span_seconds = None
            return False
        self._future = _Done(_run_preview_job(job))
        self._collect()
        return self.valid

    def _make_job(self, plan, ship, world, predictor, reference_body,
                  a_max, ramp_seconds):
        """Der auftrag -- IM HAUPTTHREAD, weil hier objekte gelesen werden.

        Alles, was danach im arbeiter passiert, sieht nur noch arrays und
        zahlen: der schnappschuss, eine KOPIE der basislinie (der predictor
        schreibt sein array fort, waehrend gerechnet wird) und je knoten der
        zustand des bezugskoerpers zu seiner zeit.
        """
        if plan is None or len(plan) == 0 or ship is None or predictor is None:
            return None
        base = predictor.get_points()
        if base is None or len(base) < 2:
            return None
        try:
            snapshot = predictor.make_maneuver_snapshot(ship, world,
                                                        self.max_points)
        except Exception:
            return None

        nodes = []
        for index, node in enumerate(plan.nodes):
            t_node = float(node.t_node)
            nodes.append({
                'index': index,
                'node': node,
                't_node': t_node,
                'dv_prograde': float(node.dv_prograde),
                'dv_normal': float(node.dv_normal),
                'ref': body_state_at(reference_body, t_node),
            })

        return {
            'base': np.array(base, dtype=np.float64, copy=True),
            'snapshot': snapshot,
            'nodes': nodes,
            'a_max': float(a_max),
            'ramp_seconds': float(ramp_seconds),
            'length_mult': float(self.length_mult),
            'max_points': int(self.max_points),
            'burn_step_s': float(self.burn_step_s),
            'burn_min_steps': int(self.burn_min_steps),
            'burn_max_steps': int(self.burn_max_steps),
            'burn_draw_points': int(self.burn_draw_points),
        }


class _Done:
    """Ein bereits erfuelltes future -- fuer den synchronen weg."""

    __slots__ = ('_value',)

    def __init__(self, value):
        self._value = value

    def done(self):
        return True

    def result(self):
        return self._value


def _burn_steps(job, total_time):
    steps = int(math.ceil(float(total_time) / job['burn_step_s']))
    return max(job['burn_min_steps'], min(job['burn_max_steps'], steps))


def _thin_arc(arc, draw_points):
    """Den brennbogen auf `draw_points` zeichenpunkte ausduennen.

    Erster und letzter punkt bleiben immer stehen: der erste ist der
    anschluss an die gleitphase davor, der letzte der zustand, mit dem die
    naechste gleitphase startet.
    """
    n = len(arc)
    if n <= draw_points:
        return arc
    idx = np.linspace(0, n - 1, draw_points)
    idx = np.unique(np.rint(idx).astype(np.int64))
    return np.ascontiguousarray(arc[idx])


def _run_preview_job(job):
    """Die kette rechnen. NUR arrays und zahlen -- laeuft im arbeiter.

    Rueckgabe `(points, markers, ms)` oder None.
    """
    t_start = time.perf_counter()
    try:
        result = _chain(job)
    except Exception:
        result = None
    if result is None:
        return None
    points, markers = result
    return (points, markers, (time.perf_counter() - t_start) * 1000.0)


def _chain(job):
    snapshot = job['snapshot']
    epoch = float(snapshot.get('sim_time', 0.0))
    ref_index = int(snapshot.get('reference_body_index', -1))
    no_memo = np.zeros((0, 10), dtype=np.float64)
    use_time = 1 if snapshot.get('use_time_dependent_bodies', True) else 0
    a_max = job['a_max']
    ramp_seconds = job['ramp_seconds']

    # Die REICHWEITE sitzt im punktabstand, nicht in der punktzahl (siehe
    # ManeuverPreview.length_mult). `max_iters` waechst mit: der adaptive
    # integrator braucht fuer einen laengeren bogen mehr schritte, und ohne
    # die anhebung braeche er auf halber strecke ab, statt weiter zu
    # reichen.
    mult = max(1e-3, float(job.get('length_mult', 1.0)))
    precision = float(snapshot['precision']) * mult
    max_iters = int(snapshot['max_iters'] * max(1.0, mult))

    segments = []
    markers = []
    source = job['base']
    budget = int(job['max_points'])

    for entry in job['nodes']:
        # KEIN budget-abbruch hier oben. Die vorige gleitphase hat sich das
        # ganze restbudget genommen und gibt es erst zurueck, wenn sie an
        # dieser zuendung abgeschnitten wird -- ein test davor sah also
        # immer null und liess die kette nach dem ersten glied abbrechen,
        # obwohl reichlich platz da war. Geprueft wird erst unmittelbar vor
        # der integration.
        t_node = entry['t_node']

        # -- 1. zustand an der knotenzeit, von der bisherigen linie
        at_node = state_on_curve(source, t_node)
        if at_node is None:
            # Der knoten liegt jenseits dessen, was gezeichnet ist. Raten
            # waere schlimmer als abbrechen.
            break

        rpx, rpy, rvx, rvy = entry['ref']
        basis = orbital_basis(at_node[0] - rpx, at_node[1] - rpy,
                              at_node[2] - rvx, at_node[3] - rvy)
        dir_x, dir_y, dv = burn_direction_world(
            basis, entry['dv_prograde'], entry['dv_normal'])
        profile = BurnProfile(dv, a_max, ramp_seconds)

        markers.append({
            'index': entry['index'],
            'node': entry['node'],
            't_node': t_node,
            'x': at_node[0], 'y': at_node[1],
            'dir_x': dir_x, 'dir_y': dir_y,
            'pro_x': basis[0] if basis else 0.0,
            'pro_y': basis[1] if basis else 0.0,
            'nrm_x': basis[2] if basis else 0.0,
            'nrm_y': basis[3] if basis else 0.0,
            'dv': dv,
            'profile': profile,
        })

        if profile.total_time <= 0.0:
            # Platzhalter-knoten: er markiert eine stelle, veraendert aber
            # nichts. Die kette laeuft auf derselben linie weiter.
            continue

        # -- 2. der brennbogen, ab der zuendung
        t_ign = profile.ignition_time(t_node)
        at_ign = state_on_curve(source, t_ign)
        if at_ign is None:
            break
        # DIE VORIGE GLEITPHASE ENDET HIER, nicht am knoten. Die zuendung
        # liegt eine halbe brenndauer VOR dem knoten -- schnitte man die
        # gleitphase erst an der knotenzeit ab, ueberlappten sich die beiden
        # abschnitte um genau diese spanne und die zusammengefuegte linie
        # liefe an der nahtstelle zeitlich rueckwaerts. Die dabei frei
        # werdenden punkte gehen ans budget zurueck.
        if segments:
            prev = segments[-1]
            keep = max(1, int(np.searchsorted(prev[:, 2], t_ign, side='right')))
            if keep < len(prev):
                budget += len(prev) - keep
                segments[-1] = prev[:keep]

        if budget <= 32:
            # Jetzt ist die frage ehrlich beantwortbar: der platz ist
            # wirklich alle. Der marker steht bereits, die bahn dahinter
            # bleibt ungezeichnet.
            break

        steps = _burn_steps(job, profile.total_time)
        arc, arc_count = _burn_arc_numba(
            at_ign[0], at_ign[1], at_ign[2], at_ign[3], t_ign - epoch,
            dir_x, dir_y,
            profile.a_peak, profile.ramp_time, profile.hold_time,
            profile.total_time, profile.ramp_rate,
            ref_index, float(snapshot.get('ref_px', 0.0)),
            float(snapshot.get('ref_py', 0.0)),
            snapshot['body_x'], snapshot['body_y'], snapshot['body_m'],
            snapshot['body_fixed'], snapshot['body_scripted'],
            snapshot['body_a'], snapshot['body_e'], snapshot['body_theta'],
            snapshot['body_arg'], snapshot['body_parent'],
            snapshot['G'], use_time, no_memo,
            steps,
        )
        arc = np.array(arc[:arc_count], dtype=np.float64, copy=True)
        arc[:, 2] += epoch
        # Der ZUSTAND am brennende kommt vom letzten INTEGRATIONSschritt,
        # nicht vom letzten gezeichneten punkt -- erst danach wird
        # ausgeduennt.
        end = arc[len(arc) - 1].copy()
        arc = _thin_arc(arc, job['burn_draw_points'])
        segments.append(arc)
        budget -= len(arc)

        # -- 3. weiter gleiten, ab dem brennende
        #
        # MIT DEM VOLLEN RESTBUDGET, und erst hinterher abgeschnitten (siehe
        # oben). Das budget vorab unter den knoten aufzuteilen war der
        # naheliegende weg und ist der falsche: eine halbierte gleitphase
        # reicht zeitlich kuerzer, und ein knoten, der ein paar sekunden
        # hinter ihrem ende liegt, faellt aus der kette -- gemessen fiel er
        # bei einem abstand von 65 062 s aus, obwohl die ungeteilte phase
        # 244 434 s weit reichte. Verbraucht werden ohnehin nur die punkte,
        # die nach dem abschneiden uebrig bleiben; die integration ist
        # derselbe eine kernelaufruf.
        coast_points = max(16, min(budget, int(job['max_points'])))
        out, used, _stats = _compute_distance_points_rkn_numba(
            float(end[0]), float(end[1]), float(end[3]), float(end[4]),
            0,
            float(snapshot.get('ref_px', 0.0)),
            float(snapshot.get('ref_py', 0.0)),
            snapshot['body_x'], snapshot['body_y'], snapshot['body_m'],
            snapshot['body_fixed'], snapshot['body_scripted'],
            snapshot['body_a'], snapshot['body_e'], snapshot['body_theta'],
            snapshot['body_arg'], snapshot['body_parent'],
            snapshot['G'], snapshot['dt'], precision,
            coast_points, max_iters,
            snapshot['rkn_min_dt'], snapshot['rkn_max_dt'],
            snapshot['rkn_rtol'], snapshot['rkn_atol_pos'],
            snapshot['rkn_atol_vel'], snapshot['rkn_safety'],
            snapshot['rkn_min_factor'], snapshot['rkn_max_factor'],
            snapshot['rkn_max_rejects'], use_time, ref_index,
            float(end[2]) - epoch, 0.0, 0.0,
            1 if snapshot.get('use_body_memo', True) else 0,
            snapshot['rkn_max_dt_floor'],
            snapshot['rkn_max_dt_timescale_divisor'],
        )
        if used < 2:
            break
        coast = np.array(out[:used], dtype=np.float64, copy=True)
        coast[:, 2] += epoch

        segments.append(coast)
        budget -= len(coast)

        # Der naechste knoten liest seinen zustand von DIESER linie -- von
        # der VOLLEN, nicht von der oben spaeter abgeschnittenen. Das
        # abschneiden betrifft nur, was GEZEICHNET wird; ablesen darf die
        # kette bis ans ende der gerechneten phase.
        source = coast

    if not segments:
        # Nur marker (platzhalter-knoten ohne delta-v): eine gueltige
        # antwort, nur ohne eigene linie.
        return (None, markers) if markers else None

    return (np.concatenate(segments, axis=0), markers)
