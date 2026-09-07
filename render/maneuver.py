"""Die fuenfte zeichen-domaene: der manoeverplan.

Sie zeichnet drei dinge und meldet EINES:

    - die geplante bahn (die kette aus ship/maneuver/preview.py)
    - je knoten einen marker auf der aktuellen linie
    - je knoten vier ziehgriffe: +/- prograde, +/- normal

Gemeldet wird `renderer.maneuver_node_hits` -- die SCHIRMpositionen von
marker und griffen. Dasselbe muster wie `apsis_marker_hits`, und aus
demselben grund: nur der renderer kennt die zeitabhaengige
frame-transformation, die den marker auf der gezeichneten linie haelt. Ein
HUD-widget, das die position selbst nachrechnete, laege in jedem bewegten
bezugsrahmen daneben -- siehe `.claude/rules/hud-ui.md`.

DIE GRIFFRICHTUNGEN KOMMEN AUS `to_this_frame_vector_xy`, nicht aus einer
differenz zweier transformierter punkte. Ein epsilon-schritt neben einem ort
bei 1.5e11 m verliert sechs stellen, bevor er ueberhaupt gerechnet ist; der
frame kann den vektor exakt.

FARBEN. Sie stehen hier als zahlen, weil der renderer die HUD-palette nicht
kennt -- und sie sind WORTGLEICH zu `ui/theme.py::SCHEME`. Aendert sich das
schema, aendern sich diese drei zeilen mit, sonst traegt der griff an der
linie eine andere farbe als derselbe knopf in der rosette.
"""

import math

import numpy as np

#: = SCHEME[3] '#48d97c', gruen -- eingerastet/bereit: autopilot und schiff.
_GREEN = (0.282, 0.851, 0.486)
#: = SCHEME[1] '#d9519f', magenta -- die zweite achse: normal/antinormal.
_MAGENTA = (0.851, 0.318, 0.624)
#: = SCHEME[2] '#eda63c', amber -- energie: ein knoten ist ein brennvorgang.
_AMBER = (0.929, 0.651, 0.235)

#: Leere schirmlinie -- dieselbe form wie ein ergebnis, damit die aufrufer
#: nicht zwischen liste und array unterscheiden muessen.
_EMPTY_SCREEN = np.zeros((0, 3), dtype=np.float64)


class ManeuverDrawMixin:
    """Geplante bahn, knotenmarker und ziehgriffe."""

    # ------------------------------------------------------------ hilfsmittel

    def _maneuver_screen_direction(self, world_dx, world_dy, t_abs):
        """Weltrichtung -> SCHIRMrichtung (einheitsvektor), oder None.

        Zwei schritte: der bezugsrahmen dreht den vektor
        (`to_this_frame_vector_xy`), dann kippt die y-achse -- die welt wird
        top-down gezeichnet (siehe `.claude/rules/rendering.md`). Der
        kamera-massstab faellt beim normieren heraus und steht deshalb gar
        nicht erst darin.
        """
        try:
            frame = self._active_frame()
            fx, fy = frame.to_this_frame_vector_xy(
                float(t_abs), float(world_dx), float(world_dy))
        except Exception:
            fx, fy = float(world_dx), float(world_dy)
        sx = fx
        sy = -fy
        length = math.hypot(sx, sy)
        if not math.isfinite(length) or length <= 1e-30:
            return None
        return (sx / length, sy / length)

    def _maneuver_project(self, xs, ys, ts, camera, camera_frame_xy):
        """(x, y, t) in weltkoordinaten -> (sx, sy) in schirmpixeln.

        IN EINEM RUTSCH, ueber `to_this_frame_xy_arrays`. Punktweise war
        genau der fehler, den `render/prediction.py` fuer die
        vorhersagelinie schon einmal behoben hat: dort lagen 3000 einzelne
        aufrufe bei 5.6 ms je frame, praktisch alles davon
        Python-aufruf-overhead. Hier waren es 900 punkte je frame -- plus
        400 weitere, solange ein griff gezogen wurde -- und das war die
        gemessene ursache dafuer, dass die bildrate beim verstellen eines
        knotens von 100 auf 40 fiel.

        JEDER punkt wird zu SEINER eigenen zeit abgebildet, auch die
        zwischenpunkte der verfeinerung: ein bewegter oder drehender
        plot-rahmen bildet zwei nachbarpunkte sonst verschieden ab.
        """
        frame = self._active_frame()
        transform = getattr(frame, 'to_this_frame_xy_arrays', None)
        frame_x = frame_y = None
        if transform is not None:
            try:
                transformed = transform(ts, xs, ys)
            except Exception:
                transformed = None
            if transformed is not None:
                frame_x, frame_y = transformed
        if frame_x is None:
            # Ein rahmen ohne array-weg: punktweise, wie frueher. Kein
            # fehlerfall -- nur der langsame zweig.
            frame_x = np.empty(len(ts), dtype=np.float64)
            frame_y = np.empty(len(ts), dtype=np.float64)
            for i in range(len(ts)):
                fx, fy = frame.to_this_frame_xy(
                    float(ts[i]), float(xs[i]), float(ys[i]))
                frame_x[i] = fx
                frame_y[i] = fy

        scale = float(camera.scale)
        sx = self.width * 0.5 + (np.asarray(frame_x) - camera_frame_xy[0]) * scale
        sy = self.height * 0.5 - (np.asarray(frame_y) - camera_frame_xy[1]) * scale
        return sx, sy

    def _maneuver_screen_polyline(self, points, camera, camera_frame_xy,
                                  max_points, refine=True):
        """Eine (n,5)-linie in schirmpunkte -- grob abgetastet, dann VERFEINERT.

        NUR AUSDUENNEN REICHT NICHT, und das ist der ganze punkt dieser
        funktion. Die vorschau setzt ihre punkte in gleichem BOGENABSTAND,
        und dieser abstand waechst mit der eingestellten reichweite
        (`length_mult`); ein fester stride darauf macht die linie mit jeder
        verlaengerung kantiger, bis sie sichtbar aus geraden stuecken
        besteht. Gemessen war der knick zwischen zwei punkten bei x2
        reichweite und nahem zoom mehrere hundert pixel von der wahren bahn
        entfernt.

        Der ausweg ist derselbe, den die vorhersagelinie nimmt: grob
        abtasten und die segmente per KUBISCHER HERMITE so weit
        unterteilen, wie eine flachheitsschranke in PIXELN es verlangt
        (`_hermite_refine_world`, `_prediction_error_budget`). Die
        aufloesung haengt damit am bildschirm statt an der linienlaenge --
        laenger heisst nicht mehr grober, und die kosten bleiben gedeckelt,
        weil unsichtbare segmente gar nicht erst unterteilt werden und das
        budget (`max_points`) gleichmaessig gedrueckt statt abgeschnitten
        wird.

        Rueckgabe: ein (n,3)-array `(sx, sy, t_abs)`. Ein array, keine
        tupel-liste: der zeichenweg macht daraus sofort wieder ein array,
        und der marker-zug sucht mit `argmin` darin statt in einer schleife.
        """
        try:
            count = len(points)
        except Exception:
            return _EMPTY_SCREEN
        if count < 2 or not isinstance(points, np.ndarray):
            return _EMPTY_SCREEN

        # GROBGITTER. Es muss nur die form tragen, nicht die glattheit --
        # die kommt aus der verfeinerung. Weniger grobe punkte heisst mehr
        # budget fuer die stellen, an denen die kurve wirklich biegt.
        #
        # OHNE verfeinerung ist `max_points` die punktzahl selbst: dann gibt
        # es nichts, was ein grobgitter spaeter wieder auffuellte.
        coarse = max(8, int(self.maneuver_coarse_points if refine
                            else max_points))
        stride = max(1, int(math.ceil(count / float(coarse))))
        idx = np.arange(0, count, stride, dtype=np.int64)
        if idx[len(idx) - 1] != count - 1:
            idx = np.append(idx, count - 1)

        sub = points[idx]
        xs = np.ascontiguousarray(sub[:, 0], dtype=np.float64)
        ys = np.ascontiguousarray(sub[:, 1], dtype=np.float64)
        ts = np.ascontiguousarray(sub[:, 2], dtype=np.float64)
        sx, sy = self._maneuver_project(xs, ys, ts, camera, camera_frame_xy)

        if refine and points.ndim == 2 and points.shape[1] >= 5:
            budget = self._prediction_error_budget(camera)
            if budget is not None:
                dense = None
                try:
                    dense = self._hermite_refine_world(
                        points, idx, (sx, sy), camera, budget[1],
                        float(self.prediction_visibility_margin_px),
                        int(max_points), {})
                except Exception:
                    dense = None
                if dense is not None and len(dense) >= 2:
                    xs = np.ascontiguousarray(dense[:, 0])
                    ys = np.ascontiguousarray(dense[:, 1])
                    ts = np.ascontiguousarray(dense[:, 2])
                    sx, sy = self._maneuver_project(
                        xs, ys, ts, camera, camera_frame_xy)

        keep = np.isfinite(sx) & np.isfinite(sy)
        if not bool(np.all(keep)):
            sx = sx[keep]
            sy = sy[keep]
            ts = ts[keep]
        if len(sx) < 2:
            return _EMPTY_SCREEN
        return np.column_stack((sx, sy, ts))

    # ---------------------------------------------------------- frame-einstieg

    def draw_maneuver(self, camera, bodies=None):
        """Den plan zeichnen und seine schirmpositionen melden.

        Die trefferliste wird bei JEDEM aufruf geleert, auch wenn nichts
        gezeichnet wird -- sonst haengt ein griff im bild, den es nicht mehr
        gibt (dieselbe regel wie bei `_draw_apsis_markers`).
        """
        self.maneuver_node_hits = []
        self.maneuver_curve_screen = _EMPTY_SCREEN
        if not getattr(self, 'maneuver_enabled', True):
            return
        preview = getattr(self, 'maneuver_preview', None)
        if preview is None or not getattr(preview, 'valid', False):
            return

        camera_frame_xy = self._frame_camera_xy(camera)

        # -- die geplante bahn
        points = getattr(preview, 'points', None)
        if points is not None and len(points) >= 2:
            run = self._maneuver_screen_polyline(
                points, camera, camera_frame_xy,
                int(self.maneuver_max_draw_points))
            if len(run) >= 2:
                self._draw_polyline(
                    np.ascontiguousarray(run[:, :2]),
                    (_GREEN[0], _GREEN[1], _GREEN[2],
                     float(self.maneuver_path_alpha)),
                    width=float(self.maneuver_path_width),
                )
                self._draw_maneuver_end_caps(
                    bodies, camera, camera_frame_xy, points, run)

        # -- waehrend eines MARKER-zugs: die BASISlinie in schirmkoordinaten,
        #    damit der marker an ihr entlang geschoben werden kann.
        #
        #    NUR beim marker-zug, nicht bei jedem zug. Frueher hing das an
        #    `maneuver_drag_active`, das auch ein GRIFF setzt -- und ein
        #    griff verschiebt den knoten gar nicht, er aendert nur sein
        #    delta-v. Die zusaetzlichen 400 projektionen je frame liefen
        #    also genau waehrend der eingabe, bei der die bildrate einbrach.
        if getattr(self, 'maneuver_drag_curve', False):
            predictor = getattr(self, '_maneuver_predictor', None)
            base = None
            if predictor is not None:
                try:
                    base = predictor.get_points()
                except Exception:
                    base = None
            if base is not None:
                # OHNE verfeinerung: diese kurve wird nicht gezeichnet,
                # sie wird nur durchsucht (naechste stuetzstelle zum
                # zeiger). Glattheit kauft dort nichts und kostet
                # projektionen in genau dem moment, in dem gezogen wird.
                self.maneuver_curve_screen = self._maneuver_screen_polyline(
                    base, camera, camera_frame_xy,
                    int(self.maneuver_curve_screen_points), refine=False)

        # -- marker und griffe
        selected = int(getattr(self, 'maneuver_selected_index', 0))
        marker_r = float(self.maneuver_marker_radius_px)
        offset = float(self.maneuver_handle_offset_px)
        handle_r = float(self.maneuver_handle_radius_px)

        for entry in getattr(preview, 'node_markers', ()) or ():
            t_abs = float(entry['t_node'])
            sx, sy = self._world_to_screen_xy_at_time(
                float(entry['x']), float(entry['y']), camera, t_abs,
                camera_frame_xy)
            if not (math.isfinite(sx) and math.isfinite(sy)):
                continue
            if not self._is_on_screen(sx, sy, offset + handle_r + 16.0):
                continue

            active = int(entry['index']) == selected
            self._draw_maneuver_marker(sx, sy, marker_r, active)

            pro = self._maneuver_screen_direction(
                entry['pro_x'], entry['pro_y'], t_abs)
            nrm = self._maneuver_screen_direction(
                entry['nrm_x'], entry['nrm_y'], t_abs)
            if pro is None or nrm is None:
                self.maneuver_node_hits.append({
                    'index': int(entry['index']), 'sx': sx, 'sy': sy,
                    'radius_px': marker_r, 'handles': [],
                })
                continue

            # Der gezogene griff steht AUSGELENKT -- er ist ein steuer-
            # knueppel, kein schieberegler, und ohne sichtbaren ausschlag
            # gaebe es keine rueckmeldung darueber, wie schnell der wert
            # gerade laeuft. Die trefferposition bleibt trotzdem die
            # RUHElage: der zeiger haelt den knueppel, der knueppel folgt
            # nicht dem zeiger.
            grab = getattr(self, 'maneuver_drag_handle', None)
            handles = []
            for kind, (dx, dy), color in (
                ('prograde', pro, _GREEN),
                ('retrograde', (-pro[0], -pro[1]), _GREEN),
                ('normal', nrm, _MAGENTA),
                ('antinormal', (-nrm[0], -nrm[1]), _MAGENTA),
            ):
                hx = sx + dx * offset
                hy = sy + dy * offset
                push = 0.0
                if (grab is not None and int(grab[0]) == int(entry['index'])
                        and grab[1] == kind):
                    push = float(grab[2])
                self._draw_maneuver_handle(
                    sx, sy, hx + dx * push, hy + dy * push, dx, dy, handle_r,
                    color, active, deflection=push)
                handles.append({
                    'kind': kind, 'sx': hx, 'sy': hy,
                    'radius_px': handle_r, 'dir_sx': dx, 'dir_sy': dy,
                })

            self.maneuver_node_hits.append({
                'index': int(entry['index']), 'sx': sx, 'sy': sy,
                'radius_px': marker_r, 'handles': handles,
            })

    def _draw_maneuver_end_caps(self, bodies, camera, camera_frame_xy,
                                points, run):
        """Wo die koerper stehen, wenn das schiff am ENDE DES PLANS ankommt.

        Dieselbe aussage wie bei den bahnlinien (`render/orbits.py`), nur
        eine bahn weiter: dort steht die kappe fuer das ende der
        VORHERSAGElinie, hier fuer das ende der GEPLANTEN. Der plan reicht
        weiter, und ohne eigene kappen liesse sich gar nicht ablesen, wo ein
        knoten das schiff gegenueber den koerpern hinlegt.

        Der kreis traegt den ECHTEN radius (`body.radius * camera.scale`),
        keinen festen pixelwert: liegt die gruene raute des linienendes in
        ihm, steckt das schiff zur planendzeit im koerper. Ein fester punkt
        koennte das nicht sagen.

        Beides in gruen, der farbe der geplanten bahn -- die weissen und
        koerperfarbenen kappen gehoeren der vorhersage, und zwei
        bedeutungen in einer farbe waeren keine.
        """
        if not getattr(self, 'maneuver_end_caps', True):
            return
        try:
            t_end = float(points[len(points) - 1, 2])
        except Exception:
            return
        alpha = float(self.maneuver_path_alpha)
        color = (_GREEN[0], _GREEN[1], _GREEN[2], alpha)

        scale = abs(float(camera.scale))
        for body in bodies or ():
            if getattr(body, 'is_ship', False):
                continue
            radius_px = float(getattr(body, 'radius', 0.0) or 0.0) * scale
            # Derselbe boden wie in `_draw_body_disc_outline`; darunter
            # waere der kreis kleiner als der punkt, den er umschliesst.
            if radius_px < 0.75:
                continue
            try:
                pos = body.position_at_time(t_end)
                bx, by = float(pos.x), float(pos.y)
            except Exception:
                continue
            sx, sy = self._world_to_screen_xy_at_time(
                bx, by, camera, t_end, camera_frame_xy)
            self._draw_body_disc_outline(sx, sy, radius_px, color)

        # Die kappe der LINIE zuletzt, damit sie ueber den kreisen liegt.
        self._draw_end_cap(float(run[len(run) - 1, 0]),
                           float(run[len(run) - 1, 1]), color,
                           self.ui_px(self.orbit_line_end_cap_px))

    # ------------------------------------------------------------- die formen

    def _draw_maneuver_marker(self, sx, sy, radius, active):
        """Ein sechseck -- die form, die sonst nichts im bild traegt.

        Die raute gehoert Ap/Pe, der kreis den koerpern, der pfeil dem
        schiff. Ein knoten braucht eine eigene silhouette, sonst ist er auf
        einen blick nicht von einer apsis zu unterscheiden.
        """
        alpha = 1.0 if active else 0.72
        color = (_AMBER[0], _AMBER[1], _AMBER[2], alpha)
        pts = []
        for i in range(7):
            ang = math.pi / 6.0 + i * (math.pi / 3.0)
            pts.append((sx + math.cos(ang) * radius,
                        sy + math.sin(ang) * radius))
        self._draw_polyline(pts, color, width=2.0 if active else 1.4)
        if active:
            inner = []
            for i in range(7):
                ang = math.pi / 6.0 + i * (math.pi / 3.0)
                inner.append((sx + math.cos(ang) * radius * 0.42,
                              sy + math.sin(ang) * radius * 0.42))
            self._draw_polyline(inner, color, width=1.4)

    def _draw_maneuver_handle(self, sx, sy, hx, hy, dx, dy, radius, color,
                              active, deflection=0.0):
        """Ein stiel vom marker weg und eine pfeilspitze am ende.

        `deflection` ist der ausschlag des KNUEPPELS in pixeln: er schiebt
        die spitze nach aussen und laesst den stiel mitwachsen, damit der
        ausschlag zu sehen ist, mit dem der wert gerade laeuft.
        """
        alpha = 0.95 if active else 0.5
        if deflection:
            alpha = 1.0
        rgba = (color[0], color[1], color[2], alpha)
        # Der stiel beginnt AUSSERHALB des markers, sonst laeuft er durch
        # dessen sechseck hindurch.
        start_x = sx + dx * (radius * 1.3)
        start_y = sy + dy * (radius * 1.3)
        tip_x = hx + dx * radius
        tip_y = hy + dy * radius
        px, py = -dy, dx
        self._draw_line_segments(
            [(start_x, start_y), (hx, hy),
             (tip_x, tip_y), (hx + px * radius * 0.62, hy + py * radius * 0.62),
             (tip_x, tip_y), (hx - px * radius * 0.62, hy - py * radius * 0.62)],
            rgba, width=2.0 if active else 1.4,
        )
