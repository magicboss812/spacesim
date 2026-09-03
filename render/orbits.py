"""Bahnlinien und referenzspuren.

Die kurven selbst rechnet `bodies/orbit_lines.py` (reines numpy, kein GL); hier
steht die projektion in den aktiven plot-rahmen und der zeichen-pfad. Siehe
.claude/rules/orbit-lines.md.
"""
import math

import numpy as np

from bodies import orbit_lines


class OrbitDrawMixin:
    """Die bahnlinien der koerper und die aufgezeichneten referenzspuren."""

    def _reset_reference_trajectories(self):
        self._reference_traj_points = {}
        self._reference_traj_last_sample_time = None
        # frame-wechsel: gecachte kamera-frame-position ist nicht mehr gültig.
        self._camera_frame_xy_key = None

    def _record_reference_trajectories(self, bodies):
        if not self.reference_trajectories_enabled:
            return

        sample_step = max(0.0, float(self.reference_trajectories_sample_step_s))
        if self._reference_traj_last_sample_time is not None and sample_step > 0.0:
            if abs(float(self._frame_time_s) - float(self._reference_traj_last_sample_time)) < sample_step:
                return

        active_ids = set()
        for body in bodies:
            if getattr(body, 'is_ship', False):
                continue

            body_id = id(body)
            active_ids.add(body_id)
            trail = self._reference_traj_points.get(body_id)
            if trail is None:
                # Fester numpy-puffer statt deque von tupeln: das zeichnen
                # braucht die spur als array, und np.asarray über eine
                # tupel-liste kostete pro körper und frame spürbar zeit
                # (27 körper x bis zu 300 punkte, jeden frame neu gewandelt).
                cap = max(64, int(self.reference_trajectories_max_points))
                trail = {'buf': np.empty((cap, 2), dtype=np.float64), 'n': 0}
                self._reference_traj_points[body_id] = trail

            try:
                fx, fy = self._frame_transform_xy(float(body.position.x), float(body.position.y))
            except Exception:
                continue

            buf = trail['buf']
            n = trail['n']
            if n > 0:
                dx = fx - buf[n - 1, 0]
                dy = fy - buf[n - 1, 1]
                if dx * dx + dy * dy < 1e-18:
                    continue
            if n < buf.shape[0]:
                buf[n, 0] = fx
                buf[n, 1] = fy
                trail['n'] = n + 1
            else:
                buf[:-1] = buf[1:]
                buf[-1, 0] = fx
                buf[-1, 1] = fy

        stale_ids = [k for k in self._reference_traj_points.keys() if k not in active_ids]
        for stale_id in stale_ids:
            del self._reference_traj_points[stale_id]

        self._reference_traj_last_sample_time = float(self._frame_time_s)

    def _draw_reference_trajectories(self, bodies, camera):
        if not self.reference_trajectories_enabled:
            return

        camera_frame_xy = self._frame_camera_xy(camera)
        half_w = self.width * 0.5
        half_h = self.height * 0.5
        scale = float(camera.scale)

        for body in bodies:
            if getattr(body, 'is_ship', False):
                continue

            trail = self._reference_traj_points.get(id(body))
            if trail is None or trail['n'] < 2:
                continue

            # Vektorisiert statt python-schleife: spuren haben bis zu
            # reference_trajectories_max_points punkte pro körper und frame.
            # `buf[:n]` ist eine sicht auf den ringpuffer -- keine wandlung.
            arr = trail['buf'][:trail['n']]
            sxs = half_w + (arr[:, 0] - camera_frame_xy[0]) * scale
            sys_ = half_h - (arr[:, 1] - camera_frame_xy[1]) * scale
            min_sx = float(sxs.min()); max_sx = float(sxs.max())
            min_sy = float(sys_.min()); max_sy = float(sys_.max())

            # Größen-schwelle: kollabiert die ganze spur auf eine sub-pixel-fläche
            # (z. B. weit herausgezoomt), ist sie ohnehin unsichtbar -> nicht
            # zeichnen. Die position des körpers zeigt dann sein icon.
            min_px = float(self.reference_traj_min_screen_px)
            if (max_sx - min_sx) < min_px and (max_sy - min_sy) < min_px:
                continue

            # Komplett off-screen liegende spur: weder punkte-liste bauen noch
            # pro segment clippen.
            margin = float(self.prediction_visibility_margin_px)
            right = self.width + margin
            bottom = self.height + margin
            if max_sx < -margin or min_sx > right or max_sy < -margin or min_sy > bottom:
                continue

            base = getattr(body, 'color', (200, 200, 200))
            cr = min(1.0, max(0.0, base[0] / 255.0))
            cg = min(1.0, max(0.0, base[1] / 255.0))
            cb = min(1.0, max(0.0, base[2] / 255.0))

            # ALS SPALTEN WEITERREICHEN. Die punkte liegen schon als arrays
            # vor; die tupel-liste, die hier stand, wurde vom klipper und
            # von _draw_polyline sofort wieder in arrays zurueckverwandelt
            # -- bei bis zu 300 punkten je koerper und frame reine arbeit
            # ohne ergebnis.
            screen_points = np.empty((sxs.shape[0], 2), dtype=np.float64)
            screen_points[:, 0] = sxs
            screen_points[:, 1] = sys_

            if min_sx >= -margin and max_sx <= right and min_sy >= -margin and max_sy <= bottom:
                # Spur liegt vollständig im sichtfenster: Liang-Barsky wäre für
                # jedes segment ein no-op und lieferte exakt einen run.
                runs = (screen_points,)
            else:
                runs = self._visible_window_runs(screen_points, margin_px=margin,
                                                 coords=(sxs, sys_))
            for run in runs:
                if len(run) < 2:
                    continue
                self._draw_polyline(run, color=(cr, cg, cb, 0.42), width=1.0)

    # ------------------------------------------------------------------
    # Bahnlinien der koerper (rechnung in orbit_lines.py)
    # ------------------------------------------------------------------

    def _ensure_orbit_line_set(self):
        """Der zustandsbehaftete teil, ueber frames hinweg gehalten.

        Die konfiguration wird JE FRAME hineingeschrieben statt das objekt
        neu zu bauen -- ein neubau wuerde die eingeblendeten deckkraefte
        verwerfen, und dann blitzt beim drehen an einem regler die ganze
        szene auf.
        """
        oset = self._orbit_line_set
        if oset is None:
            oset = orbit_lines.OrbitLineSet()
            self._orbit_line_set = oset
        oset.track_samples = max(8, int(self.orbit_line_track_samples))
        # Der winkel-boden der spur -- siehe OrbitLineSet.__init__.
        oset.samples_per_period = max(8.0, float(self.orbit_line_samples_per_period))
        oset.max_track_samples = max(int(self.orbit_line_track_samples),
                                     int(self.orbit_line_max_track_samples))
        oset.max_periods_drawn = max(0.25, float(self.orbit_line_max_periods_drawn))
        oset.soi_full = float(self.orbit_line_soi_full)
        oset.soi_fade = float(self.orbit_line_soi_fade)
        oset.reveal_full = float(self.orbit_line_reveal_full)
        oset.reveal_fade = float(self.orbit_line_reveal_fade)
        oset.alpha_max = float(self.orbit_line_alpha_max)
        oset.alpha_floor = float(self.orbit_line_alpha_floor)
        oset.alpha_floor_focus = float(self.orbit_line_alpha_floor_focus)
        oset.fade_rate = float(self.orbit_line_fade_rate)
        oset.full_orbit_enabled = bool(self.orbit_line_full_orbit_enabled)
        oset.full_samples = max(16, int(self.orbit_line_full_samples))
        oset.full_max_span_s = float(self.orbit_line_full_max_span_s)
        return oset

    def _draw_frame_polyline(self, screen_x, screen_y, color, width,
                             min_screen_px=0.0):
        """Fertig projizierte bildschirmpunkte als polylinie, mit culling."""
        n = int(screen_x.shape[0])
        if n < 2:
            return False
        if not (np.all(np.isfinite(screen_x)) and np.all(np.isfinite(screen_y))):
            return False

        min_sx = float(screen_x.min()); max_sx = float(screen_x.max())
        min_sy = float(screen_y.min()); max_sy = float(screen_y.max())

        margin = float(self.prediction_visibility_margin_px)
        right = self.width + margin
        bottom = self.height + margin
        if max_sx < -margin or min_sx > right or max_sy < -margin or min_sy > bottom:
            return False
        if (max_sx - min_sx) < min_screen_px and (max_sy - min_sy) < min_screen_px:
            return False

        pts = np.empty((n, 2), dtype=np.float64)
        pts[:, 0] = screen_x
        pts[:, 1] = screen_y

        if min_sx >= -margin and max_sx <= right and min_sy >= -margin and max_sy <= bottom:
            runs = (pts,)
        else:
            runs = self._build_clipped_polyline_runs(
                pts, margin_px=margin, coords=(screen_x, screen_y))

        drew = False
        for run in runs:
            if len(run) >= 2:
                self._draw_polyline(run, color=color, width=width)
                drew = True
        return drew

    def _draw_end_cap(self, sx, sy, color, size_px):
        """Kleine raute auf dem endpunkt einer linie.

        Die endkappen sind der eigentliche messwert der ganzen funktion:
        liegt die des koerpers auf der des schiffs, ist das schiff zur
        endzeit der vorhersage dort, wo der koerper dann steht.
        """
        if not (math.isfinite(sx) and math.isfinite(sy)):
            return
        if sx < -size_px or sx > self.width + size_px:
            return
        if sy < -size_px or sy > self.height + size_px:
            return
        r = float(size_px)
        diamond = [(sx, sy - r), (sx + r, sy), (sx, sy + r), (sx - r, sy),
                   (sx, sy - r)]
        self._draw_polyline(diamond, color=color, width=1.0)

    def _draw_body_disc_outline(self, sx, sy, r_px, color):
        """Kreis-umriss mit dem ECHTEN radius des koerpers auf dem linienende.

        Das ist der messwert der bahnlinie: liegt dieser kreis ueber der
        weissen schiffs-endkappe, steckt das schiff zur endzeit der vorhersage
        im koerper. Anders als die alte raute ist er KEIN fester pixelwert --
        er ist `body.radius * camera.scale` und schrumpft mit heraus-zoomen
        auf nichts, genau wie die koerperscheibe selbst.
        """
        if not (math.isfinite(sx) and math.isfinite(sy) and math.isfinite(r_px)):
            return
        if r_px < 0.75:
            return
        if (sx < -r_px or sx > self.width + r_px
                or sy < -r_px or sy > self.height + r_px):
            return
        seg = max(12, min(64, int(r_px)))
        ang = np.linspace(0.0, 2.0 * math.pi, seg + 1)
        ring = np.empty((seg + 1, 2), dtype=np.float64)
        ring[:, 0] = float(sx) + float(r_px) * np.cos(ang)
        ring[:, 1] = float(sy) + float(r_px) * np.sin(ang)
        self._draw_polyline(ring, color=color, width=1.0)

    def _draw_orbit_lines(self, bodies, camera, predictor=None, real_dt=0.0):
        """Wo jeder koerper waehrend des VORHERSAGE-FENSTERS entlanglaeuft.

        Dieselbe zeitspanne wie die schiffslinie, punkt fuer punkt in dem
        plot-frame, den seine EIGENE zeit aufspannt. Damit steht die
        endkappe des koerpers fuer "hier ist er, wenn das schiff am ende
        seiner linie ankommt" -- fallen die beiden endkappen zusammen,
        trifft man. Das ist der ganze zweck, und es funktioniert nur, wenn
        beide linien durch dieselbe transformation gehen.

        Eine feste ellipse waere hier schlicht falsch: ein plot-frame ist
        eine ZEITABHAENGIGE abbildung, eine starr transformierte ellipse
        zeigt die bahn also so, wie sie JETZT gerade laege. Im Erd-rahmen
        kam dabei eine Erdbahn um die Sonne heraus, obwohl die Erde dort im
        ursprung steht.
        """
        self.debug_info['orbit_lines_drawn'] = 0
        if not self.orbit_lines_enabled:
            return

        oset = self._ensure_orbit_line_set()

        points = None
        generation = None
        if predictor is not None:
            try:
                points = predictor.get_points()
            except Exception:
                points = None
            generation = getattr(predictor, '_points_generation', None)

        # Rahmen ZUERST: der ursprungskoerper bestimmt mit, wie fein die
        # spuren abgetastet werden muessen (relative_min_period) -- im
        # Titania-rahmen laeuft Uranus einmal je 8.7 tagen um den
        # bildmittelpunkt, nicht einmal je 84 jahren.
        frame = self._active_frame()
        origin_body = orbit_lines.frame_origin_body(frame)

        oset.update(
            bodies, points,
            sim_time=self._frame_time_s, real_dt=real_dt,
            reference_body=self.current_reference_body,
            selected_body=self.selected_body,
            generation=generation,
            origin_body=origin_body,
        )

        # Der ursprungskoerper bekommt keine linie: er steht in seinem
        # eigenen rahmen still.
        drawable = [e for e in oset.entries()
                    if e.reveal > 0.002 and e.alpha > 0.004
                    and e.track is not None and e.track_t is not None
                    and e.track.shape[0] >= 2
                    and e.body is not origin_body]
        if not drawable:
            return

        # EINE tabelle fuer alle: sie stehen alle auf dem fenster des
        # praediktors, also wird die transformation einmal auf einem
        # knotengitter bestimmt statt je koerper und stichprobe.
        track_t = drawable[0].track_t
        table = orbit_lines.FrameAffineTable(
            frame, float(track_t[0]), float(track_t[-1]),
            knot_angle=float(self.orbit_line_knot_angle))
        if not table.valid:
            return

        # Knotentabellen fuer die faint volllinien: EINE JE KOERPER, ueber
        # SEINE periode. Kein gemeinsames gitter wie bei der spur-tabelle --
        # die perioden liegen um groessenordnungen auseinander (Mond 27 d,
        # ein planet jahre), ein ueber die laengste gespanntes gitter liesse
        # der kurzen periode zu wenige knoten und die ursprungs-interpolation
        # explodiert. Gecacht ueber die frames auf (rahmen, fenster), gebaut
        # nur wenn `_recompute` eine neue `full_track_t` geliefert hat.
        full_enabled = bool(self.orbit_line_full_orbit_enabled)
        full_alpha_mult = float(self.orbit_line_full_alpha_mult)
        full_knot_angle = float(self.orbit_line_full_knot_angle)
        table_cache = self._full_orbit_tables
        # Bei jedem frame-wechsel (R, 1, 2) den ganzen cache verwerfen -- so
        # kann eine wiederverwendete id() des frame-objekts keiner alten
        # tabelle einen falschen treffer geben.
        if frame is not getattr(self, '_full_orbit_tables_frame', None):
            table_cache.clear()
            self._full_orbit_tables_frame = frame
        frame_key = id(frame)
        live_keys = set()

        def _table_for(times, knot_angle):
            # Schluessel ueber die FENSTERGRENZEN, nicht id(times): eine
            # freigegebene array-id kann wiederverwendet werden und gaebe der
            # alten tabelle einen falschen treffer. (rahmen, fenster,
            # knotenwinkel) bestimmt die affine tabelle vollstaendig -- eine
            # kollision ist harmlos.
            key = (frame_key, round(float(times[0]), 3),
                   round(float(times[-1]), 3), float(knot_angle))
            live_keys.add(key)
            hit = table_cache.get(key)
            if hit is not None:
                return hit if hit.valid else None
            tab = orbit_lines.FrameAffineTable(
                frame, float(times[0]), float(times[-1]),
                knot_angle=knot_angle)
            table_cache[key] = tab
            return tab if tab.valid else None

        def _full_table_for(entry):
            return _table_for(entry.full_track_t, full_knot_angle)

        camera_frame_xy = self._frame_camera_xy(camera)
        scale = abs(float(camera.scale))
        margin = float(self.prediction_visibility_margin_px)
        view_diag = math.hypot(self.width + 2.0 * margin,
                               self.height + 2.0 * margin)
        min_px = float(self.orbit_line_min_screen_px)
        half_w = self.width * 0.5
        half_h = self.height * 0.5
        cap_px = self.ui_px(self.orbit_line_end_cap_px)
        show_caps = bool(self.orbit_line_end_caps)
        drawn = 0
        any_full = False

        for entry in drawable:
            body = entry.body
            base = getattr(body, 'color', (200, 200, 200))
            try:
                cr = min(1.0, max(0.0, float(base[0]) / 255.0 * 0.85))
                cg = min(1.0, max(0.0, float(base[1]) / 255.0 * 0.85))
                cb = min(1.0, max(0.0, float(base[2]) / 255.0 * 0.85))
            except Exception:
                cr = cg = cb = 0.75

            # Faint volllinie zuerst -- ein ganzer umlauf, HINTER der hellen
            # spur, damit die enthuellte linie oben liegt. Gleiche
            # transformations-pipeline `koerper(t) - ursprung(t)`, also im
            # plot-frame automatisch richtig.
            if (full_enabled and getattr(entry, 'full_track', None) is not None
                    and entry.full_track_t is not None
                    and entry.full_track.shape[0] >= 2):
                fa = float(entry.alpha) * full_alpha_mult
                full_table = _full_table_for(entry) if fa > 0.003 else None
                if full_table is not None:
                    ftrack = entry.full_track
                    # ALLE stichproben projizieren -- kein stride. Der stride
                    # oben schaetzt die zeichen-aufloesung aus `track_len`, und
                    # das ist die WELT-bogenlaenge; ueber eine ganze periode
                    # traegt die eltern-heliozentrik da das zehn- bis
                    # hundertfache der plot-frame-laenge hinein. Es sind ohnehin
                    # nur 0-3 volllinien, `table.project` ist numpy-vektorisiert.
                    ffx, ffy = full_table.project(
                        entry.full_track_t,
                        np.ascontiguousarray(ftrack[:, 0]),
                        np.ascontiguousarray(ftrack[:, 1]))
                    fsx = half_w + (ffx - camera_frame_xy[0]) * scale
                    fsy = half_h - (ffy - camera_frame_xy[1]) * scale
                    self._draw_frame_polyline(
                        fsx, fsy, (cr, cg, cb, fa),
                        float(self.orbit_line_width), min_screen_px=min_px)

            # DAS GEZEICHNETE GITTER, nicht das gemessene. Fuer koerper,
            # deren umlaufzeit das praediktor-fenster ueberdauert, sind das
            # dieselben arrays und dieselbe tabelle wie bisher; ein mond mit
            # hunderten umlaeufen im fenster bekommt ein eigenes, feineres
            # gitter ueber die letzten umlaeufe (OrbitLineSet._build_draw_track).
            d_track = getattr(entry, 'draw_track', None)
            d_track_t = getattr(entry, 'draw_track_t', None)
            if d_track is None or d_track_t is None or d_track.shape[0] < 2:
                d_track, d_track_t = entry.track, entry.track_t
                d_len, d_shared = entry.track_len, True
            else:
                d_len = float(getattr(entry, 'draw_track_len', entry.track_len))
                d_shared = bool(getattr(entry, 'draw_shared', True))

            d_table = table if d_shared else _table_for(
                d_track_t, float(self.orbit_line_knot_angle))
            if d_table is None:
                continue

            # Enthuellung: die linie rollt sich VOM KOERPER AUS ab.
            total = int(d_track.shape[0])
            reveal = max(0.0, min(1.0, float(entry.reveal)))
            n_show = max(2, int(math.ceil(reveal * total)))

            # Gezeichnet wird nur so fein, wie die fehlerschranke verlangt.
            # Als kruemmungsradius dient die bogenlaenge selbst -- fuer eine
            # fast gerade kurve ist das eine unterschaetzung, also zu viele
            # punkte statt zu wenige.
            arc_px = d_len * scale * reveal
            r_eff = max(1.0, arc_px / (2.0 * math.pi))
            stride = orbit_lines.polyline_stride(
                n_show, arc_px, r_eff, view_diag,
                float(self.orbit_line_tolerance_px))
            idx = orbit_lines.stride_indices(n_show, stride)

            fx, fy = d_table.project(
                d_track_t[idx],
                np.ascontiguousarray(d_track[idx, 0]),
                np.ascontiguousarray(d_track[idx, 1]))
            sx = half_w + (fx - camera_frame_xy[0]) * scale
            sy = half_h - (fy - camera_frame_xy[1]) * scale

            if self._draw_frame_polyline(
                    sx, sy, (cr, cg, cb, float(entry.alpha)),
                    float(self.orbit_line_width), min_screen_px=min_px):
                drawn += 1

            # Die endkappe steht fuer den koerper zur ENDZEIT -- solange die
            # linie nicht ganz da ist, endet sie irgendwo dazwischen und
            # duerfte nicht als messpunkt gelesen werden.
            if show_caps and reveal > 0.995:
                any_full = True
                body_r_px = float(getattr(body, 'radius', 0.0) or 0.0) * scale
                self._draw_body_disc_outline(
                    float(sx[-1]), float(sy[-1]), body_r_px,
                    (cr, cg, cb, float(entry.alpha)))

        # Die kappe des schiffs nur, wenn es etwas zu vergleichen gibt --
        # und ueber den weg der GEZEICHNETEN linie, damit sie auf deren ende
        # sitzt und nicht daneben.
        if show_caps and any_full and points is not None and len(points) >= 2:
            try:
                last = points[-1]
                csx, csy = self._world_to_screen_xy_at_time(
                    float(last[0]), float(last[1]), camera, float(last[2]),
                    camera_frame_xy=camera_frame_xy)
                self._draw_end_cap(csx, csy, (1.0, 1.0, 1.0, 0.9), cap_px)
            except Exception:
                pass

        self.debug_info['orbit_lines_drawn'] = drawn

        # Volllinien-tabellen aufraeumen: alles, was dieses bild nicht mehr
        # gebraucht hat (frame gewechselt, neue full_track_t nach recompute).
        if len(table_cache) > len(live_keys):
            for stale in [k for k in table_cache if k not in live_keys]:
                del table_cache[stale]
