"""Der warp-halt und das fortschreiben der bestehenden kurve.

Im zeitraffer wird die kurve GEHALTEN statt jeden frame neu gerechnet -- sonst
zieht `_anchor_first_point` sie je frame um die volle bahnbewegung starr mit
und sie zittert. Der halt ist damit kein sparmodus, sondern die bedingung
dafuer, dass die linie im raffer ueberhaupt stillsteht.
"""
import math

import numpy as np

from physics.vec import Vec2
from physics.kernels import POINT_COLUMNS, _empty_points, _widen_points
from physics.kernels.propagate import _compute_distance_points_rkn_numba


class HoldMixin:
    """Der halt, das verbrauchen der kurve und das umschalten auf einen
    neuen bahnast.

    EINE PROGNOSEKURVE WIRD VERBRAUCHT, NIE VERSCHOBEN.
    `_advance_points_along_curve` wirft die stuetzstellen weg, deren zeit
    vergangen ist, und stellt das schiff als neuen kopf voran -- nichts
    dahinter bewegt sich. Die kurve per `points[:, 0/1]` wieder ans schiff zu
    schieben ist der fehler, den `tests/apsis_stability_test.py` bewacht: der
    bezugskoerper wandert dabei NICHT mit, der kegelschnitt steht danach um
    genau die relativbewegung versetzt, und das gemeldete perigaeum ist um
    ebenso viel falsch."""

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

        dvx_seen = cur_vx - float(last_vx)
        dvy_seen = cur_vy - float(last_vy)
        delta_speed = math.hypot(dvx_seen, dvy_seen)
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

        # Die schwerkraft wird HERAUSGERECHNET, nicht mit einer schranke
        # ueberdeckt.
        #
        # Frueher stand hier `allowed_speed = max(allowed, 4 * |g| * dt)`: der
        # gesamte geschwindigkeitssprung wurde gegen eine schranke von der
        # groesse des schwerkraft-anteils gehalten. Fern vom planeten geht das
        # auf, NAHE DER PERIAPSIS nicht: dort ist |g| = 8.1 m/s^2, ueber einen
        # 2-sekunden-schritt also 16 m/s schwerkraft gegen 6.7 m/s vollschub
        # je bild -- die schranke lag bei 65 m/s und der schub verschwand
        # vollstaendig darunter. Die vorhersagelinie wurde in genau dem
        # moment nicht mehr angefordert, in dem sie sich am staerksten
        # aendert, und sprang erst wieder an, wenn das schiff weit genug weg
        # war. Das ist das ruckartige nachziehen nahe der periapsis.
        #
        # Richtig ist der REST: was bleibt von der geschwindigkeitsaenderung
        # uebrig, wenn man abzieht, was die schwerkraft erklaert. Gemessen auf
        # einer bahn mit e = 0.7 um die Erde, je bild:
        #
        #     periapsis   gleitflug 0.023 m/s   |   schub 6.69 m/s
        #     apoapsis    gleitflug 0.000 m/s   |   schub 6.67 m/s
        #
        # Der schub steht damit ueberall gleich deutlich da (faktor ~290 ueber
        # dem grundrauschen), und die feste toleranz von 1 m/s trennt beides
        # sauber. Nebenbei faengt der test auch den fall, in dem schub der
        # schwerkraft ENTGEGEN zeigt und die summe klein ist: bei nu = 90 Grad
        # betraegt der gesamtsprung 0.98 m/s -- unter der toleranz -- der rest
        # aber 6.66 m/s.
        #
        # Die restschranke muss mit der KRUEMMUNG von g mitwachsen, sonst
        # feuert sie im zeitraffer: `g * dt` erklaert einen 28-stunden-schritt
        # nicht mehr. `|g_jetzt - g_vorher| * dt` waechst genau mit diesem
        # fehler mit -- gemessen im gleitflug von 0.5 s bis 100800 s (7 d/s)
        # bleibt der rest bei jedem schritt unter der schranke.
        residual_speed = delta_speed
        if world is not None:
            try:
                g = world.acceleration_at(ship, ship.position, cur_time)
                gx = float(g.x)
                gy = float(g.y)
                span = max(dt_age, 0.0)
                residual_speed = math.hypot(dvx_seen - gx * span, dvy_seen - gy * span)

                last_gx = self._last_seen_gx
                last_gy = self._last_seen_gy
                # DIE KRUEMMUNGS-SCHRANKE GILT NUR, SOLANGE DER SCHRITT DIE
                # BAHN UEBERHAUPT AUFLOEST.
                #
                # `|g_jetzt - g_vorher|` misst, wie stark sich die schwerkraft
                # ueber den schritt geaendert hat -- ein gutes mass fuer den
                # fehler von `g * dt`, solange sich g dazwischen stetig
                # bewegt. Deckt ein schritt aber MEHRERE UMLAEUFE ab (7 d/s
                # sind 28 stunden je bild, auf einer 2-stunden-bahn), dann
                # sind anfangs- und endwert unkorreliert: sie koennen zufaellig
                # dicht beieinander liegen, die schranke faellt zusammen und
                # der ganz normale gleitflug reisst sie. Gemessen auf der
                # e = 0.7-bahn bei 100800 s je schritt: rest 2.46e3 m/s gegen
                # eine schranke von 1.20e3 m/s -- ein bild von sechs, ohne
                # jeden schub.
                #
                # Oberhalb der bahn-zeitskala (`sqrt(r/|g|)`, auf der
                # kreisbahn T/2pi -- dieselbe groesse, die den zeitraffer
                # deckelt) traegt der vergleich also nichts mehr, und es
                # bleibt die alte, grosszuegige schranke. Das ist die
                # richtige seite des irrtums: schub gibt es dort ohnehin
                # nicht (`test.py` sperrt ihn oberhalb von
                # `realtime_warp_max`), eine verpasste anforderung kostet
                # nichts -- eine falsche zerreisst die gehaltene kurve.
                resolves_orbit = True
                try:
                    t_char = self._characteristic_timescale(world, ship)
                    resolves_orbit = t_char is None or span <= t_char
                except Exception:
                    resolves_orbit = True
                if last_gx is None or last_gy is None or not resolves_orbit:
                    curvature_dv = math.hypot(gx, gy) * span
                else:
                    curvature_dv = math.hypot(gx - float(last_gx), gy - float(last_gy)) * span
                allowed_speed = max(allowed_speed,
                                    float(self.gravity_dv_safety_factor) * curvature_dv)
                self._last_seen_gx = gx
                self._last_seen_gy = gy
            except Exception:
                residual_speed = delta_speed

        reason = None
        if residual_speed > allowed_speed:
            reason = "velocity"
        elif delta_pos > allowed_pos:
            reason = "position"

        if reason is None:
            self._remember_ship_state(ship, world)
            return False

        # Schub ist KEIN bruch der bahn, sondern ihre stetige veraenderung: die
        # gezeichnete linie ist danach ein paar dutzend millisekunden alt, aber
        # nicht falsch. Sie deshalb zu leeren und synchron neu zu rechnen kostet
        # 59 ms pro frame (voller sonnensystem-satz) und verwarf zugleich jedes
        # asynchrone ergebnis, weil die version im naechsten frame schon wieder
        # weiter war. Ein echter POSITIONS-sprung (teleport, reparenting) ist
        # dagegen ein bruch -- dort bleibt der harte weg unten.
        if reason == "velocity" and self._request_thrust_recompute(ship, world):
            self._remember_ship_state(ship, world)
            if self.debug:
                try:
                    print(
                        "PRED_DBG_TRAJECTORY_REFRESH: "
                        f"reason=velocity rest={residual_speed:.6e} (roh {delta_speed:.6e}) "
                        f"allowed={allowed_speed:.6e} "
                        "mode=async-coalesced",
                        flush=True,
                    )
                except Exception:
                    pass
            # Nicht kurzschliessen: update() soll normal weiterlaufen, damit
            # ein fertiges ergebnis eingewechselt und die linie ans schiff
            # geheftet wird.
            return False

        old_version = int(self._trajectory_version)
        self._trajectory_version = old_version + 1
        if self.debug:
            try:
                if reason == "velocity":
                    print(
                        "PRED_DBG_TRAJECTORY_INVALIDATED: "
                        f"reason=velocity rest={residual_speed:.6e} allowed={allowed_speed:.6e} "
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

    def _advance_points_along_curve(self, ship, now):
        """Kurve VERBRAUCHEN statt starr verschieben.

        Rueckgabe: die zahl der vorn verbrauchten stuetzstellen, oder None,
        wenn es nicht geht (keine/zu kurze kurve, zeit abgelaufen) -- dann
        muss der aufrufer den alten weg gehen.

        Die vorhersage ist eine eigenschaft der BAHN, nicht des augenblicks.
        Ohne schub bleibt sie stehen und das schiff rutscht an ihr entlang.
        Also werden vorn die punkte weggeworfen, deren zeit bereits vergangen
        ist (die zeitspalte ist absolute sim-zeit, das ist exakt und per
        suchlauf billig), und der rest bleibt in ORT UND ZEIT stehen, wo er
        ist.

        DIE KURVE WIRD VORN ANGESTUECKELT, NICHT VERBOGEN.

        Stuetzstellen lassen sich nur GANZ wegwerfen -- eine halbe gibt es
        nicht. Bliebe als kopf immer die naechste stuetzstelle VOR dem schiff
        stehen, liefe der rest zwischen zwei verbrauchten stuetzstellen von 0
        auf eine volle punktweite und spraenge dann zurueck: ein saegezahn mit
        der amplitude EINER PUNKTWEITE. Weil das eine weltlaenge ist und der
        zoom welt und linie gleich vergroessert, saehe es auf JEDER zoomstufe
        gleich aus -- die linie rueckte sichtbar in stufen statt stetig vor.

        Richtig ist, dem unveraenderten rest die aktuelle schiffsposition als
        neuen kopf voranzustellen. Das erste segment ist dann ein echtes
        teilstueck, das stetig kuerzer wird, bis die naechste stuetzstelle
        verbraucht ist. Kein punkt hinter dem kopf bewegt sich dabei
        ueberhaupt -- und genau darauf beruht, dass die Ap/Pe-marker
        stillstehen.
        """
        points = self.points
        if np is None or not isinstance(points, np.ndarray) or points.ndim != 2:
            return None
        if points.shape[0] < 4 or points.shape[1] < 3:
            return None
        if ship is None or not math.isfinite(now):
            return None

        # Den selbst vorangestellten kopf aus dem vorframe wieder entfernen,
        # damit unten immer auf den UNVERAENDERTEN stuetzstellen gesucht wird
        # (und die liste nicht bei jedem frame um einen punkt waechst).
        had_head = bool(getattr(self, '_synthetic_head', False)) and points.shape[0] >= 3
        if had_head:
            points = points[1:]

        times = points[:, 2]
        if not (math.isfinite(float(times[0])) and math.isfinite(float(times[-1]))):
            return None
        # Reicht die kurve zeitlich ueberhaupt noch in die zukunft?
        if float(times[-1]) <= now:
            return None

        # Erster punkt, der ECHT in der zukunft liegt. Die zeitspalte ist
        # monoton steigend, also genuegt eine binaere suche.
        #
        # 'right', nicht 'left': eine stuetzstelle GENAU auf `now` ist die
        # gegenwart, und die gegenwart ist der kopf, den wir gleich davor
        # setzen. Bliebe sie stehen, saessen zwei punkte aufeinander und das
        # erste segment haette laenge null -- mitsamt seiner tangente, an der
        # der navball haengt. Exakte gleichheit ist kein grenzfall, sondern
        # der regelfall: eine FRISCH gerechnete kurve beginnt per konstruktion
        # bei ship@world.time, und _anchor_first_point laeuft unmittelbar
        # danach.
        drop = int(np.searchsorted(times, now, side='right'))
        drop = max(0, min(drop, points.shape[0] - 2))

        if drop == 0 and had_head:
            # NICHTS VERBRAUCHT -> NICHTS UMKOPIEREN.
            #
            # Der regelfall in echtzeit: die stuetzstellen liegen auf festem
            # BOGENabstand (bei spielueblichem zoom hunderte kilometer),
            # waehrend ein bild nur bruchteile davon vorrueckt -- es wird also
            # ueber viele bilder hinweg gar keine stuetzstelle faellig. Dann
            # genuegt es, den vorhandenen kopf nachzufuehren.
            #
            # Das spart nicht bloss die kopie: es haelt auch die IDENTITAET
            # des arrays fest, und daran haengen zwei caches, die sonst in
            # jedem bild leerliefen -- der apsis-scan (id(pts), siehe
            # get_apsis_markers) und die abtastung der linie im renderer
            # (_make_prediction_line_cache_key). Die marker duerfen dabei
            # stehen bleiben, weil der scan den kopf ohnehin ueberspringt
            # (skip_head) und hinter ihm kein punkt bewegt wurde.
            self.points[0, 0] = float(ship.position.x)
            self.points[0, 1] = float(ship.position.y)
            self.points[0, 2] = now
            if self.points.shape[1] > 3:
                self.points[0, 3] = float(getattr(ship.velocity, 'x', 0.0))
                self.points[0, 4] = float(getattr(ship.velocity, 'y', 0.0))
            return 0

        tail = points[drop:] if drop > 0 else points
        head = np.empty((1, points.shape[1]), dtype=np.float64)
        head[0, 0] = float(ship.position.x)
        head[0, 1] = float(ship.position.y)
        head[0, 2] = now
        if points.shape[1] > 3:
            # Der kopf IST das schiff -- also auch seine tangente. Frueher
            # wurde die der naechsten stuetzstelle uebernommen; damit haette
            # das erste (stetig kuerzer werdende) teilstueck eine tangente
            # getragen, die zur falschen stelle der bahn gehoert.
            head[0, 3] = float(getattr(ship.velocity, 'x', 0.0))
            head[0, 4] = float(getattr(ship.velocity, 'y', 0.0))

        self.points = np.concatenate((head, tail), axis=0)
        self._synthetic_head = True
        # Die zeitspalte der verbliebenen punkte ist unangetastet -- ihr
        # versatz gegen den schnappschuss bleibt also, was er war.
        self._invalidate_derived_caches(soft=True)
        return drop

    def _anchor_first_point(self, ship, world):
        """Setzt den kurvenanfang auf das schiff.

        DER REGELFALL IST DAS VERBRAUCHEN, NICHT DAS VERSCHIEBEN. Erste wahl
        ist `_advance_points_along_curve` -- die kurve bleibt stehen und das
        schiff rutscht an ihr entlang. Die starre verschiebung unten ist nur
        noch der fallback fuer den rolling-modus und fuer eine kurve, deren
        zeit abgelaufen ist (dann steht ohnehin gleich eine neuberechnung an).

        WARUM NICHT MEHR STARR. Die verschiebung zieht die GANZE kurve um den
        kopfversatz mit, und der ist nicht der versatz je frame, sondern der
        ueber das ganze alter des schnappschusses -- `max_async_wall_age`
        laesst 1.5 s echtzeit zu, bei 60 s/s also bis zu 90 sim-sekunden
        bahnbewegung. Der referenzkoerper wandert dabei NICHT mit. Was bleibt,
        ist die RELATIVbewegung schiff<->referenzkoerper: die ganze kegel-
        schnittbahn liegt um diesen betrag seitlich neben dem koerper, und
        damit steht die periapsis-hoehe falsch. Weil das alter mit der
        rechenlatenz schwankt, schwankt der angezeigte Pe/Ap-abstand mit --
        das ist das hin- und herspringen der marker in echtzeit, und es
        verschwand im zeitraffer nur deshalb, weil dort der halt schon
        verbraucht statt verschoben hat.

        Wird doch starr verschoben, muss die ZEITSPALTE mitwandern. Sie ist
        bei der berechnung auf die damalige `world.time` bezogen worden
        (_compute_from_snapshot). Ohne die zeit-korrektur faellt die zeitbasis
        pro frame um ein sim_dt zurueck (gemessen 900-2700 s). Der renderer
        waehlt daraus ueber _world_to_screen_xy_at_time die epoche des
        plot-frames: bei einem bewegten frame-ursprung (body-centred
        non-rotating) landet dieselbe weltposition dadurch neben dem schiff --
        gemessen 54.5 px bei 2e-6 px/m, exakt der drift von Erde ueber 900 s.
        Der betrag wird in `_points_time_offset` mitgeschrieben, weil die
        punktzeiten damit nicht mehr zum schnappschuss passen und jeder, der
        aus ihnen eine lokale zeit zurueckrechnet, das wissen muss.
        """
        if self._points_count() == 0:
            return
        sx = float(ship.position.x)
        sy = float(ship.position.y)
        try:
            st = float(world.time) if world is not None else None
        except Exception:
            st = None

        # IM ZEITRAFFER NICHT STARR VERSCHIEBEN. Diese methode zieht sonst
        # die ganze kurve um den kopfversatz mit. Bei gehaltener kurve ist
        # dieser versatz gross (das gespeicherte ergebnis ist mehrere frames
        # alt und das schiff je frame ~1e8 m weiter), die kurve wuerde also
        # jeden frame quer durchs bild wandern -- und genau das macht sie
        # anschliessend fuer den halt unbrauchbar, weil ihre zeitspalte dann
        # nicht mehr zu ihrer geometrie passt (gemessen: kopfabstand 3.2e6 m
        # statt der punktweite 1e6 m, obwohl die echte abweichung zwischen
        # welt und predictor nur 37 m je frame betraegt).
        if (self._hold_active() and np is not None
                and isinstance(self.points, np.ndarray)
                and self.points.ndim == 2 and self.points.shape[0] >= 2
                and st is not None):
            dx = sx - float(self.points[0, 0])
            dy = sy - float(self.points[0, 1])
            dt = st - float(self.points[0, 2]) if self.points.shape[1] >= 3 else 0.0
            if math.isfinite(dx) and math.isfinite(dy) and math.isfinite(dt):
                self._apply_head_taper(self.points, sx, sy, st, dx, dy, dt)
                self._invalidate_derived_caches(soft=True)
            return

        # ECHTZEIT: DIESELBE MECHANIK WIE DER HALT.
        #
        # Der rolling-modus fuehrt in `_roll_states` einen zweiten, parallel
        # gehaltenen zustand mit, der punktweise zu `points` passen muss --
        # der bleibt beim alten weg. Alles andere verbraucht.
        if not self.rolling_mode and st is not None:
            if self._advance_points_along_curve(ship, st) is not None:
                return

        if np is not None and isinstance(self.points, np.ndarray):
            dx = sx - float(self.points[0, 0])
            dy = sy - float(self.points[0, 1])
            if math.isfinite(dx) and math.isfinite(dy):
                self.points[:, 0] += dx
                self.points[:, 1] += dy
                self.points[0, 0] = sx
                self.points[0, 1] = sy
                if st is not None and self.points.shape[1] >= 3:
                    dt = st - float(self.points[0, 2])
                    if math.isfinite(dt):
                        self.points[:, 2] += dt
                        self.points[0, 2] = st
                        # Die punktzeiten passen jetzt um `dt` nicht mehr zu
                        # `snapshot["sim_time"]` -- siehe _points_time_offset.
                        self._points_time_offset = float(
                            getattr(self, '_points_time_offset', 0.0)) + dt
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
            try:
                t0 = float(self.points[0][2])
            except Exception:
                t0 = 0.0
            # zeitbasis mitziehen (siehe docstring); ohne world.time bleibt sie
            # wie bisher stehen.
            dt = (st - t0) if st is not None else 0.0
            if not math.isfinite(dt):
                dt = 0.0
            t0 += dt
            if dt:
                self._points_time_offset = float(
                    getattr(self, '_points_time_offset', 0.0)) + dt
            try:
                dx = sx - float(self.points[0][0])
                dy = sy - float(self.points[0][1])
                for i, p in enumerate(self.points):
                    if i == 0:
                        self.points[i] = (sx, sy, t0)
                    elif hasattr(p, "__len__") and len(p) >= 3:
                        self.points[i] = (float(p[0]) + dx, float(p[1]) + dy, float(p[2]) + dt)
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

    # ------------------------------------------------------ zeitraffer-halt

    def set_hold(self, enabled):
        """Zeitraffer-halt ein/aus. Ausschalten erzwingt eine neuberechnung.

        Die beiden richtungen sind NICHT symmetrisch.

        AUSSCHALTEN (zurueck in die echtzeit) entwertet hart: der spieler darf
        von da an sofort wieder schub geben, und die gehaltene kurve weiss
        davon nichts.

        EINSCHALTEN dagegen uebernimmt eine kurve, die der asynchrone weg bis
        zum vorigen frame in jedem frame frisch gehalten hat -- sie ist also
        genau so gut wie eine neu gerechnete. Hart zu entwerten kostete dort
        gemessen 14.1 ms im hauptthread beim schritt 10m/s -> 1h/s (der
        stufe, bei der der halt anspringt), gegen 0.2 ms in den nachbarn.
        Also weich: neu ANFORDERN und derweil weiterhalten, wie beim
        stufenwechsel (siehe _request_hold_recompute).
        """
        enabled = bool(enabled)
        if enabled == getattr(self, 'hold_enabled', False):
            return
        self.hold_enabled = enabled
        self._synthetic_head = False
        self._hold_pending_swap = False
        self._hold_invalidated = True
        if enabled:
            self._hold_soft_invalidated = True
            # `_resume_context` stammt vom letzten asynchronen lauf und gehoert
            # damit zur kurve, die jetzt gehalten wird -- stehen lassen, sonst
            # kann sie waehrend des anlaufens nicht nachlegen. Festgehalten
            # wird er in _request_hold_recompute, dem einzigen besitzer.
        else:
            self._hold_soft_invalidated = False
            self._resume_context = None
            self._hold_resume_context = None

    def invalidate_hold(self, soft=False):
        """Die gehaltene kurve ist ueberholt (schub, rahmenwechsel, ...).

        `soft=True` heisst: die kurve ist GEOMETRISCH weiterhin richtig, nur
        ihre parameter stimmen nicht mehr (horizont oder punktabstand
        verstellt). Das schiff sitzt weiter auf ihr, sie reicht weiter in die
        zukunft -- sie ist bloss zu kurz oder zu lang. Ein solcher wechsel
        darf deshalb NACHGEREICHT werden, statt den hauptthread anzuhalten;
        siehe _hold_advance und _request_hold_recompute.

        Der harte weg bleibt fuer alles, was die kurve wirklich unbrauchbar
        macht: sprung, reparenting, rahmenwechsel, ende des halts.
        """
        self._hold_invalidated = True
        self._hold_soft_invalidated = bool(soft) and not self.rolling_mode
        self._synthetic_head = False
        if not soft:
            # Weiterrechnen geht nur auf einer kurve, die noch gilt.
            self._resume_context = None
            self._hold_resume_context = None
            self._hold_pending_swap = False

    def _hold_active(self):
        if not bool(getattr(self, 'hold_enabled', False)):
            return False
        if self.rolling_mode:
            return False
        if not self.initialized:
            return False
        # Eine zoom-aenderung veraendert die punktdichte und muss deshalb
        # durch den normalen rechenweg -- der halt darf sie nicht schlucken.
        if getattr(self, '_view_scale_changed', False):
            return False
        return True

    def _hold_advance(self, ship, world):
        """Kurve VERBRAUCHEN statt neu rechnen. True = frame ist erledigt.

        WARUM. Ohne halt ruft update() bei jedem frame eine neuberechnung an
        und `_anchor_first_point` schiebt die gespeicherte kurve STARR so,
        dass ihr kopf auf dem schiff sitzt. Bei 1m/s ist der versatz je frame
        winzig. Bei 7d/s rueckt das schiff je frame um ~10 000 sim-sekunden
        bahn weiter -- die ganze kurve wird also um diesen betrag quer
        verschoben und springt zurueck, sobald ein frisch gerechnetes
        ergebnis eintrifft. Genau dieser wechsel ist das "zittern" der linie
        und der Ap/Pe-marker.

        Richtig ist: die vorhersage ist eine eigenschaft der BAHN, nicht des
        augenblicks. Ohne schub bleibt sie stehen und das schiff rutscht an
        ihr entlang. Also werden vorn die punkte weggeworfen, deren zeit
        bereits vergangen ist (die zeitspalte ist absolute sim-zeit, das ist
        exakt und per suchlauf billig), und der rest bleibt, wo er ist.

        Der kopf wird trotzdem an das schiff gezogen, aber ABKLINGEND ueber
        die ersten `hold_taper_points` punkte -- welt und predictor
        propagieren die planeten leicht unterschiedlich, ohne korrektur
        klafft am schiff eine luecke. Die korrektur voll auf die ganze kurve
        zu legen waere wieder die starre verschiebung von oben.

        Failsafe: laeuft der vorrat unter `hold_refresh_fraction`, gibt die
        methode False zurueck und der normale weg rechnet nach. Die linie
        kann also nicht auslaufen.
        """
        if getattr(self, '_hold_invalidated', False):
            # WEICHE entwertung -> ANFORDERN statt anhalten. Die kurve, die
            # hier steht, ist geometrisch weiterhin richtig; nur ihr horizont
            # bzw. punktabstand ist ueberholt. Sie darf also weiterlaufen,
            # waehrend die neue im hintergrund entsteht. Gelingt das nicht
            # (kein async, keine kurve, kein worker frei), bleibt der harte
            # weg -- die zusicherung "update() baut synchron eine, wenn keine
            # da ist" gilt unveraendert.
            if (getattr(self, '_hold_soft_invalidated', False)
                    and self._request_hold_recompute(ship, world)):
                self._hold_invalidated = False
                self._hold_soft_invalidated = False
                # und weiter unten ganz normal verbrauchen
            else:
                self._hold_invalidated = False
                self._hold_soft_invalidated = False
                self._synthetic_head = False
                return False
        if ship is None or world is None or np is None:
            return False

        try:
            now = float(world.time)
        except Exception:
            return False

        # VERBRAUCHEN statt verschieben -- dieselbe mechanik, die inzwischen
        # auch die echtzeit benutzt (siehe _advance_points_along_curve).
        #
        # Das ergebnis wird IMMER uebernommen, auch wenn gleich darauf
        # abgebrochen wird: sonst rastet der halt ein. Bricht er ab, bevor der
        # schnitt steht, bleibt die kurve stehen, waehrend das schiff
        # weiterfliegt -- der kopfabstand waechst dann jeden frame weiter
        # (gemessen 6.4e5 -> 3.2e6 m in fuenf frames) und die
        # abbruchbedingung ist von da an dauerhaft erfuellt.
        drop = self._advance_points_along_curve(ship, now)
        if drop is None:
            return False

        points = self.points
        sx = float(ship.position.x)
        sy = float(ship.position.y)

        # HINTEN ANSTUECKELN, was vorn verbraucht wurde -> der horizont
        # bleibt konstant und die linie wandert mit, statt zu schrumpfen und
        # bei jeder auffrischung zurueckzuspringen.
        if drop > 0:
            budget = self._get_target_point_cap()
            missing = int(budget) - int(self.points.shape[0])
            # JE FRAME NUR EIN STUECK. Normal sind das die punkte, die vorn
            # gerade verbraucht wurden (bei 7d/s rund 170) -- die schranke
            # merkt man dort nicht. Sie greift, wenn das BUDGET springt:
            # `apply_predictor_horizon` zieht mit dem zeitraffer-schritt auch
            # das punktbudget mit, beim wechsel 7d/s -> 30d/s von 10 000 auf
            # 40 000. Die fehlenden 30 000 punkte in EINEM frame anzustueckeln
            # kostete gemessen 40.3 ms im hauptthread (nachbarframes 0.3 ms) --
            # genau der ruckler, den §17 fuer set_length schon beseitigt hat.
            # Verteilt ueber ein paar frames faellt er nicht auf, und die
            # bestellte neue kurve ist ohnehin schon unterwegs.
            cap = int(getattr(self, 'hold_extend_max_points', 1000) or 0)
            if cap > 0 and missing > cap:
                missing = cap
            if missing > 0:
                self._hold_extend_tail(missing)
            points = self.points

        target_points = self._get_target_point_cap()
        remaining = points.shape[0]
        refresh_at = max(4, int(target_points * float(getattr(
            self, 'hold_refresh_fraction', 0.25))))
        # LAEUFT SCHON EINE NEUE KURVE, IST DIE SCHWELLE EINE ANDERE.
        #
        # Sie misst den vorrat am ANGEPEILTEN budget. Waechst das budget
        # sprunghaft -- der zeitraffer-schritt zieht ueber
        # `apply_predictor_horizon` den horizont UND das punktbudget mit, beim
        # wechsel 7d/s -> 30d/s von 10 000 auf 40 000 --, dann rutscht die
        # noch vollstaendige kurve allein durch die neue bezugsgroesse unter
        # die schwelle, und der halt rechnet SYNCHRON nach: gemessen 43.8 ms
        # im hauptthread gegen 0.3 ms in den nachbarframes.
        #
        # Ist der ersatz bereits unterwegs (`_hold_pending_swap`), kann die
        # linie gar nicht auslaufen -- dann genuegt eine absolute
        # not-schwelle. Kommt der auftrag nicht an, raeumt `update()` das
        # flag ab und der harte weg steht im naechsten frame wieder offen.
        if getattr(self, '_hold_pending_swap', False):
            refresh_at = max(4, int(target_points * 0.02))
        if remaining < refresh_at:
            # Vorrat zu klein -> normaler weg rechnet nach (und der halt
            # greift danach wieder). Das ist die failsafe-schwelle.
            self._synthetic_head = False
            return False

        # Weicht das schiff von der gehaltenen kurve ab, stimmt sie nicht
        # mehr (schub, sprung, rahmenwechsel) -- dann lieber neu rechnen als
        # eine falsche kurve weiterzeichnen. Gemessen wird gegen die ZWEITE
        # stuetzstelle, denn die erste ist ja das schiff selbst. Regulaer
        # liegt es hoechstens eine punktweite davor; der spielraum darueber
        # faengt ab, dass welt und predictor die planeten nicht voellig
        # gleich propagieren (gemessen ~37 m je frame).
        if points.shape[0] >= 3:
            span = math.hypot(float(points[2, 0]) - float(points[1, 0]),
                              float(points[2, 1]) - float(points[1, 1]))
            gap = math.hypot(float(points[1, 0]) - sx,
                             float(points[1, 1]) - sy)
            if gap > max(span * 4.0, 1.0):
                self._synthetic_head = False
                return False

            # DER HALT BRAUCHT EINE OBERGRENZE FUER DEN SEITLICHEN VERSATZ.
            #
            # Die pruefung darueber misst den abstand ENTLANG der bahn und
            # laesst vier punktweiten zu -- an einer seitlichen abweichung
            # geht sie deshalb blind vorbei. Und der vorrat laeuft nie leer,
            # weil `_hold_extend_tail` hinten nachlegt: gemessen 0 volle
            # neuberechnungen in 3000 frames. Die gehaltene kurve wurde also
            # EINMAL gerechnet und danach nie wieder mit der welt verglichen.
            #
            # Welt und predictor rechnen aber nicht dasselbe (andere
            # schrittweiten, und die welt setzt die planeten ueber
            # `bodies.position_at_time` mit konstanter winkelrate, der
            # predictor mit echtem Kepler-solve). Der unterschied summiert
            # sich. Gemessen in einer erdumlaufbahn (rp 2e7 m, e = 0.3) bei
            # 1 h/s ueber 2.5 umlaeufe: das schiff steht am ende **3.9e5 m
            # = 1.96 % des bahnradius** neben der linie, und in einer
            # sonnenumlaufbahn ueber 350 tage 4.2e5 m. Das ist genau das
            # "schiff loest sich von der linie" -- und es verschwindet beim
            # verlassen des zeitraffers, weil `set_hold(False)` hart entwertet.
            #
            # Gemessen wird SENKRECHT zur kurve (die laengsrichtung ist
            # bereits durch den kopf abgedeckt) an den beiden ersten echten
            # stuetzstellen -- der kopf ist ja das schiff selbst.
            #
            # Die schwelle ist ein PIXELMASS, weil nur das sichtbar ist:
            # dieselbe weltlaenge ist zoom-abhaengig entweder unsichtbar oder
            # fingerdick. Angefordert wird ASYNCHRON (derselbe weg wie beim
            # stufenwechsel) -- die alte kurve bleibt stehen, bis die neue da
            # ist, es gibt also keinen ruckler und kein springen.
            dxs = float(points[2, 0]) - float(points[1, 0])
            dys = float(points[2, 1]) - float(points[1, 1])
            chord = math.hypot(dxs, dys)
            if chord > 0.0:
                drift = abs((sx - float(points[1, 0])) * dys
                            - (sy - float(points[1, 1])) * dxs) / chord
            else:
                drift = 0.0
            self.hold_drift_m = drift
            #
            # Getaktet wird das NICHT ueber eine uhr, sondern ueber
            # `_hold_pending_swap`: solange ein auftrag laeuft, wird kein
            # zweiter gestellt. Damit stellt sich die auffrischrate von
            # selbst auf "eine je rechendauer" ein -- dieselbe selbstregelung
            # wie beim schub. Eine feste echtzeit-sperre (0.25 s) war
            # nachweislich zu grob: gemessen 4 auffrischungen ueber 1500
            # frames, und der versatz lief zwischendurch wieder auf 4.4e5 m.
            if (drift > self._hold_drift_limit_m()
                    and not getattr(self, '_hold_pending_swap', False)):
                self._request_hold_recompute(ship, world)

        return True

    def _hold_drift_limit_m(self):
        """Erlaubter seitlicher versatz des schiffs von der gehaltenen kurve.

        Ein PIXELMASS, in meter umgerechnet -- siehe _hold_advance. Ohne
        bekannte zoomstufe bleibt nur die untergrenze.
        """
        px = float(getattr(self, 'hold_drift_max_px', 0.5) or 0.0)
        # px <= 0 heisst AUS -- keine anforderung, egal wie weit es auseinander
        # laeuft. (Das ist auch der schalter, mit dem die gegenprobe im test
        # das alte verhalten wiederherstellt.)
        if px <= 0.0:
            return float('inf')
        scale = getattr(self, '_view_scale', None)
        if scale is None or not math.isfinite(scale) or scale <= 0.0:
            # Ohne bekannte zoomstufe gibt es kein pixelmass -- dann lieber
            # nichts tun als eine weltlaenge zu raten.
            return float('inf')
        floor = float(getattr(self, 'hold_drift_min_m', 1.0) or 0.0)
        return max(floor, px / scale)

    def _hold_extend_tail(self, wanted):
        """Hinten so viele punkte anstueckeln, wie vorn verbraucht wurden.

        Damit bleibt der HORIZONT konstant. Ohne das wird die gehaltene kurve
        nur von vorn aufgebraucht, schrumpft also sichtbar, bis die
        auffrischung sie schlagartig wieder auf volle laenge bringt -- die
        linie pulsiert dann im takt der auffrischung, statt gleichmaessig
        mitzuwandern.

        Gerechnet wird als FORTSETZUNG desselben laufs: derselbe
        schnappschuss, derselbe integrator-zustand, dieselbe schrittweite
        (siehe _resume_context und die init_*-parameter des kernels). Die
        angehaengten punkte sind deshalb genau die, die eine von vornherein
        laengere rechnung geliefert haette -- kein bruch an der nahtstelle.

        Kostet nur die tatsaechlich verbrauchten punkte (bei 7d/s rund 170 je
        frame) statt der vollen neuberechnung von 10 000.
        """
        wanted = int(wanted)
        if wanted <= 0 or np is None:
            return 0
        # Waehrend ein stufenwechsel unterwegs ist, gehoert `_resume_context`
        # bereits zur NEUEN kurve (der worker setzt ihn beim fertigwerden).
        # Angesetzt werden muss aber an die kurve, die gerade gehalten wird.
        context = None
        if getattr(self, '_hold_pending_swap', False):
            context = getattr(self, '_hold_resume_context', None)
        if not context:
            context = getattr(self, '_resume_context', None)
        if not context:
            return 0
        if _compute_distance_points_rkn_numba is None:
            return 0
        points = self.points
        if not isinstance(points, np.ndarray) or points.shape[0] < 2:
            return 0

        snapshot = context['snapshot']
        px, py, vx, vy = context['state']
        if not all(math.isfinite(v) for v in (px, py, vx, vy)):
            return 0

        try:
            out, used, stats = _compute_distance_points_rkn_numba(
                px, py, vx, vy,
                0,
                float(snapshot.get("ref_px", 0.0)),
                float(snapshot.get("ref_py", 0.0)),
                snapshot["body_x"], snapshot["body_y"],
                snapshot["body_m"], snapshot["body_fixed"],
                context['body_scripted'], context['body_a'], context['body_e'],
                context['body_theta'], context['body_arg'], context['body_parent'],
                snapshot["G"], context['base_dt'], snapshot["precision"],
                int(wanted) + 1, int(max(10000, (wanted + 1) * 100)),
                context['min_dt'], context['max_dt'],
                context['rtol'], context['atol_pos'], context['atol_vel'],
                context['safety'], context['min_factor'], context['max_factor'],
                context['max_rejects'],
                context['use_time_dependent_bodies'], context['ref_index'],
                context['kernel_t'], context['accumulated'], context['proposed_dt'],
                1 if getattr(self, 'use_body_memo', True) else 0,
                float(context.get('max_dt_floor', context['max_dt'])),
                float(context.get('timescale_divisor', 0.0)),
            )
        except Exception:
            return 0

        used = int(used)
        if used <= 1:
            return 0

        # out[0] ist der fortsetz-punkt selbst und steht schon in der liste.
        addition = out[1:used].copy()
        addition[:, 2] += float(snapshot.get("sim_time", 0.0))
        self.points = np.concatenate((points, addition), axis=0)

        context['state'] = (float(stats[7]), float(stats[8]),
                            float(stats[9]), float(stats[10]))
        context['accumulated'] = float(stats[11])
        context['proposed_dt'] = float(stats[12])
        context['kernel_t'] = float(stats[13])
        self._invalidate_derived_caches(soft=True)
        return int(addition.shape[0])

    def _apply_head_taper(self, points, sx, sy, now, dx, dy, dt):
        """Kopf ans schiff ziehen -- ABKLINGEND ueber die ersten punkte.

        Der unterschied zu `_anchor_first_point` ist der ganze punkt der
        sache: dort wird die KOMPLETTE kurve starr um (dx, dy) verschoben,
        hier klingt die korrektur ueber `hold_taper_points` punkte auf null
        ab. Das fernfeld bleibt also stehen, wo es steht.
        """
        taper = int(max(1, min(int(getattr(self, 'hold_taper_points', 64)),
                               points.shape[0])))
        weights = np.zeros(points.shape[0], dtype=np.float64)
        weights[:taper] = np.linspace(1.0, 0.0, taper, endpoint=False)

        points[:, 0] += dx * weights
        points[:, 1] += dy * weights
        if points.shape[1] >= 3:
            points[:, 2] += dt * weights
        points[0, 0] = sx
        points[0, 1] = sy
        if points.shape[1] >= 3:
            points[0, 2] = now

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
                    # Siehe unten: der vorangestellte kopf ist mit weg.
                    self._synthetic_head = False
                    try:
                        self._roll_states = self._roll_states[remove_count:]
                        if isinstance(self._roll_states, np.ndarray) and self._roll_states.shape[0] > 0:
                            self.points = self._roll_states.copy()
                        else:
                            self.points = _empty_points()
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
                # Siehe unten: der vorangestellte kopf ist mit weg.
                self._synthetic_head = False
                try:
                    self.points = self.points[remove_count:]
                except Exception:
                    self.points = np.array(self.points[remove_count:], dtype=np.float64)
            return int(remove_count)

        # List / generic fallback: use same projection logic.
        # self.points can't be an ndarray here (the isinstance branch above
        # always returns) but Pyright doesn't narrow attribute access across
        # that control flow, so alias to a local it can narrow.
        pts = self.points
        if isinstance(pts, np.ndarray):
            return 0
        try:
            n = len(pts)
            if n <= 1:
                return 0
        except Exception:
            return 0

        remove_count = 0
        try:
            for i in range(n - 1):
                p0 = pts[i]
                p1 = pts[i + 1]
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
            # Der selbst vorangestellte kopf (siehe
            # _advance_points_along_curve) ist mit weggeschnitten worden --
            # sonst wuerde die naechste runde ihn ein zweites mal entfernen
            # und dabei eine ECHTE stuetzstelle verlieren, jeden frame eine.
            self._synthetic_head = False
            try:
                del pts[:remove_count]
            except Exception:
                for _ in range(remove_count):
                    try:
                        pts.pop(0)
                    except Exception:
                        break
        return int(remove_count)
