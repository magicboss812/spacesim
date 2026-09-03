"""Die vorausberechnete bahnlinie des schiffs.

Die REINE ZAHLENARBEIT liegt nicht mehr hier, sondern in `physics/kernels/`:
integratoren, das Kepler-bahnmodell, die Ap/Pe-suche und die propagation. Das
waren 2400 der 6800 zeilen dieser datei, und sie gehoeren der physik, nicht dem
schiff. Was bleibt, ist die BUCHFUEHRUNG: wann wird neu gerechnet, was wird
gehalten, was gezeichnet.

Zwei regeln, die beide teuer erkauft sind:

  * EINE PROGNOSEKURVE WIRD VERBRAUCHT, NIE VERSCHOBEN. Der vorausblick ist
    eine eigenschaft der BAHN, nicht des augenblicks. Siehe CLAUDE.md und
    tests/apsis_stability_test.py.
  * Predictor und World teilen sich die integratoren -- eine konstante in
    `physics/kernels/integrators.py` zu aendern heisst, sie auch in
    `physics/world_kernels.py` zu aendern.
"""
from physics.vec import Vec2
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numba import njit

from physics.kernels import (
    BODY_MEMO_COLUMNS,
    POINT_COLUMNS,
    _empty_points,
    _no_body_memo,
    _widen_points,
)
from physics.kernels.apsis import _find_apsis_markers_numba, _refine_apsis_numba
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
from physics.kernels.kepler import (
    _body_kepler_constants_numba,
    _body_position_at_time_numba,
    _body_scripted_relative_xy_numba,
)
from physics.kernels.propagate import (
    _compute_distance_points_aspi_numba,
    _compute_distance_points_numba,
    _compute_distance_points_numba_state,
    _compute_distance_points_rkn_numba,
)

from ship.predictor.hold import HoldMixin
from ship.predictor.compute import ComputeMixin
from ship.predictor.jobs import JobsMixin
from ship.predictor.view import ViewMixin

# Predictor ist absichtlich Numba-only
NUMBA_AVAILABLE = True












































class Predictor(HoldMixin, ComputeMixin, JobsMixin, ViewMixin):
    """Die vorausberechnete bahnlinie -- zusammengesetzt aus mixins.

    Die klasse war 4280 zeilen. Sie ist jetzt ueber `ship/predictor/` verteilt,
    bleibt aber EIN objekt: die mixins teilen sich `self` und die felder, die
    `__init__` hier anlegt. Bewusst keine komposition -- das haette hunderte
    attributzugriffe in der zweitgroessten datei des projekts umgeschrieben.

    Was hier BLEIBT, ist der kern: der zustand, die integrator-guete und der
    frame-einstieg `update()`, der entscheidet, welcher der vier wege (halt,
    rollend, asynchron, synchron voll) diesen frame gegangen wird.
    """
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







    def _current_reference_body_index(self):
        try:
            if self.reference_body_index is None:
                return -1
            return int(self.reference_body_index)
        except Exception:
            return -1














































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












    def advance_state(self, world=None):

        if self.async_compute:
            self._swap_ready_result(None, world)


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

