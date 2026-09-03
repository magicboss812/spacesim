"""Die asynchrone rechen-pipeline des predictors.

Hier wird nicht gerechnet, hier wird VERWALTET: auftraege stellen, stornieren,
tiefe steuern, fertige ergebnisse einwechseln.
"""
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from physics.vec import Vec2
from physics.kernels import POINT_COLUMNS, _empty_points, _widen_points


class JobsMixin:
    """Die asynchrone rechen-pipeline.

    Mehrere rechnungen laufen VERSETZT nebeneinander, der durchsatz ist
    deshalb hoeher als 1/rechenzeit -- das ist der grund, warum die linie unter
    schub nicht einfriert, obwohl ein voller neuaufbau ~60 ms kostet und ein
    180-fps-frame 5.6 ms hat.

    `_swap_ready_result` ist der heikle teil: ein fertiges ergebnis darf nur
    eingewechselt werden, wenn es zu einem NEUEREN schiffszustand gehoert als
    das, was schon liegt -- sonst springt die linie zurueck."""

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
