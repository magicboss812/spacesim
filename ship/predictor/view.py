"""Die ausgabeseite des predictors.

`get_points()` liefert die GEZEICHNETE kurve -- moeglicherweise kuerzer als die
gerechnete, weil der zeitraffer einen vorrat anlegt, der nicht gezeichnet wird
(siehe `ship/horizon.py`).
"""
import math
import time

import numpy as np

from physics.vec import Vec2
from physics.kernels import POINT_COLUMNS, _empty_points, _widen_points
from physics.kernels.apsis import _find_apsis_markers_numba


class ViewMixin:
    """Was HERAUSKOMMT: punkte, Ap/Pe-marker, laenge, abstand, zoom.

    Horizont (`length`) und punktabstand (`precision`) sind zwei ENTKOPPELTE
    regler: der horizont ist der kostenknopf (kosten ~ integrierter bogen), der
    abstand ist kosmetisch (mehr oder weniger gezeichnete punkte im selben
    horizont, gleiche rechenzeit, gleiche genauigkeit)."""

    def _points_count(self):
        if np is not None and isinstance(self.points, np.ndarray):
            return int(self.points.shape[0])
        return len(self.points)

    def _empty_points_array(self) -> "np.ndarray | list":
        return _empty_points()

    def _empty_apsis_array(self):
        return np.empty((0, 5), dtype=np.float64) if np is not None else []

    def _clear_apsis_markers(self):
        self._apsis_markers = self._empty_apsis_array()
        self._apsis_cache_key = None

    def _clear_prediction_points(self):
        self.points = self._empty_points_array()
        self._roll_states = np.empty((0, 5), dtype=np.float64) if np is not None else []
        self.initialized = False
        self._points_time_offset = 0.0
        self._synthetic_head = False
        self._clear_apsis_markers()

    def set_view_scale(self, scale: float):
        try:
            scale = float(scale)
        except Exception:
            return
        if scale > 0.0:
            old = self._view_scale

            if old is not None:
                try:
                    rel_change = abs(scale - old) / max(abs(old), 1e-30)
                except Exception:
                    rel_change = 0.0
                if rel_change <= self.snapshot_view_rel_tol:
                    return

            # Neu rechnen lohnt nur, wenn der zoom die WIRKSAME punktdichte
            # veraendert. Mit angepinntem horizont (length = num_points *
            # precision, siehe test.py) klemmt _horizon_spacing_floor() die
            # zoom-verfeinerung bei JEDEM zoomwert auf exakt `precision` --
            # der synchrone _compute_full lieferte dann eine bit-identische
            # linie und kostete trotzdem die volle rechenzeit im hauptthread,
            # einmal pro mausrad-raste. Das war der massive fps-einbruch beim
            # zoomen. Aendert der zoom die dichte wirklich (nicht
            # angepinnter horizont), bleibt das verhalten wie zuvor.
            try:
                eff_old = float(self._effective_precision())
            except Exception:
                eff_old = None

            self._view_scale = scale

            try:
                eff_new = float(self._effective_precision())
            except Exception:
                eff_new = None
            eff_changed = True
            if eff_old is not None and eff_new is not None:
                eff_changed = (
                    abs(eff_new - eff_old) / max(abs(eff_old), 1e-30)
                    > self.snapshot_view_rel_tol
                )
            # Nur ueberspringen, wenn es eine linie zum BEHALTEN gibt: ohne
            # punkte garantierte der alte weg, dass der naechste update()
            # synchron eine baut -- diese zusicherung bleibt bestehen.
            try:
                has_points = self._points_count() > 0
            except Exception:
                has_points = False
            if not eff_changed and has_points:
                if self.debug:
                    print("PRED_DBG_VIEW_SCALE: eff_precision unchanged, no recompute")
                return

            try:
                self._view_scale_changed = True
                if self.debug:
                    print("PRED_DBG_VIEW_CHANGED: flagged for sync recompute")
            except Exception:
                pass
            if self.debug:
                try:
                    eff = self._effective_precision()
                except Exception:
                    eff = self.precision
                print(f"PRED_DBG_VIEW_SCALE: old={old} new={self._view_scale} eff_precision={eff}")

  
            try:
                if old is not None and self.async_compute and len(getattr(self, "_pending_futures", [])) > 0:
                    rel = abs(self._view_scale - old) / max(abs(old), 1e-30)
                    if rel > 0.02:
                        if self.debug:
                            print(f"PRED_DBG_CANCEL_PENDING: zoom rel_change={rel:.3f} canceling pending job")
                        self._cancel_pending_job()
            except Exception:
                pass

            try:
                self._view_change_cooldown_until = time.time() + float(self.view_change_cooldown)
                if self.debug:
                    print(f"PRED_DBG_VIEW_COOLDOWN: until={self._view_change_cooldown_until:.6f}")
            except Exception:
                pass

    def _invalidate_derived_caches(self, soft=False):
        """Von der punkteliste abgeleitete zwischenergebnisse verwerfen.

        soft=True heisst: die kurve wurde nur vorn verbraucht / hinten
        verlaengert / am kopf angeschmiegt (zeitraffer-halt). Die apsis-
        marker der verbliebenen punkte sind dann weiterhin gueltig und
        get_apsis_markers() darf sie gefiltert weiterreichen, statt neu zu
        scannen.
        """
        for attr in ('_apsis_cache_key', '_apsis_cache_value'):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
        if not soft:
            # NEUE GEOMETRIE -> die gemerkten marker gehoeren zu einer kurve,
            # die es nicht mehr gibt. Der zaehler ist das, was der weiche weg
            # in get_apsis_markers() dagegenhaelt: er darf nur filtern,
            # solange die kurve DIESELBE ist, auf der er zuletzt gesucht hat.
            self._points_generation = int(getattr(self, '_points_generation', 0)) + 1
        self._apsis_soft_stale = bool(soft)

    def _horizon_spacing_floor(self):
        """Feinste punktdichte, die den HORIZONT noch traegt -- oder None.

        Die kernel setzt punkte in festem ABSTAND (`_effective_precision`)
        und hoechstens `num_points` viele. Der wirklich gezeichnete bogen
        ist damit `num_points * spacing`. Sobald das kleiner als `length`
        wird, endet die linie einfach vorzeitig -- ein engerer punktabstand
        VERKUERZT also die vorhersage.

        Der bodenwert `length / num_points` ist genau der abstand, bei dem
        das punktbudget den horizont gerade noch ausfuellt.
        """
        if self.length is None:
            return None
        try:
            budget = int(self.num_points)
            horizon = float(self.length)
        except Exception:
            return None
        if budget <= 0 or not (horizon > 0.0):
            return None
        return horizon / float(budget)

    def _effective_precision(self):
        effective = float(self.precision)
        if self.auto_precision_from_zoom and self._view_scale is not None:
            zoom_precision = self.target_screen_step_px / max(self._view_scale, 1e-30)
            effective = min(effective, max(self.min_precision, zoom_precision))

        # HORIZONT VOR PUNKTDICHTE. Ohne diese schranke frisst das zoom-
        # abhaengige verfeinern den horizont auf: gemessen blieb bei
        # view_scale 2e-5 noch 10 % der vorhersage uebrig, bei 2e-4 noch
        # 1 %. Auf dem schirm sieht das aus, als wuerde die linie an der
        # ersten bildkante abgeschnitten und komme nie zurueck -- samt
        # verschwundener Ap/Pe-marker und leerem CLOSEST/T-CA, weil beide
        # ueber dieselbe punkteliste laufen.
        #
        # Bei erreichtem budget wird also GROEBER gezeichnet statt KUERZER.
        # Das ist die richtige seite des tauschs: die groebere teilung fehlt
        # nur dort, wo die bahn ohnehin kaum kruemmt (der bogenfehler einer
        # sehne waechst mit c^2/8R und ist im fernfeld weit unter einem
        # pixel), waehrend ein fehlender horizont die anzeige unbrauchbar
        # macht. Mehr naehe-detail gibt es ueber ein groesseres
        # `predictor.num_points`, nicht ueber einen kuerzeren horizont.
        floor = self._horizon_spacing_floor()
        if floor is not None and effective < floor:
            effective = floor
        return effective

    def set_display_length(self, metres):
        """Wie viel von der gerechneten kurve GEZEICHNET wird, in metern.

        None oder >= der gerechneten laenge heisst: alles. Betrifft nur, was
        get_points() herausgibt -- der halt, das anstueckeln und die
        fortsetzung arbeiten unveraendert auf der vollen kurve (self.points).
        """
        if metres is None:
            self.display_length = None
        else:
            value = float(metres)
            self.display_length = value if value > 0.0 else None

        # KEIN _clear_display_view() hier: der regler ruft diese methode jeden
        # frame, und ein harter cache-reset erzwaenge trotz gerundetem count
        # (siehe _display_point_count) je frame eine neue view. get_points()
        # verwirft die view schon selbst, sobald sich der gerundete count,
        # das zugrundeliegende array oder der zeichne-alles-fall aendert.

    def _clear_display_view(self):
        self._display_view = None
        self._display_view_base = None
        self._display_view_limit = -1

    def _display_point_count(self):
        """Zahl der zu zeichnenden fuehrenden punkte, oder None fuer alle.

        Ueber den INDEXANTEIL, nicht ueber die aufsummierte bogenlaenge: der
        kernel legt seine stuetzstellen auf festen abstand, der anteil ist
        also der laengenanteil -- und er kostet O(1) statt O(n) je frame.

        Der abstand wird an den GESPEICHERTEN punkten gemessen, nicht aus
        `self.length` erschlossen. Das ist derselbe O(1)-griff, aber er fragt
        die kurve, die wirklich da liegt, statt die zuletzt ANGEFORDERTE
        laenge fuer sie zu halten. Beim zeitraffer-stufenwechsel fallen die
        beiden fuer ein paar frames auseinander: `set_length()` stellt schon
        auf 4x, waehrend der halt noch die alte 1x-kurve zeigt (die neue
        entsteht im hintergrund, siehe _request_hold_recompute). Mit
        `self.length` als bezug waeren davon 1/4 gezeichnet worden -- die
        linie waere auf 25 % zusammengefallen und beim einwechseln wieder
        aufgesprungen. Gemessen: 25.0 % / 24.9 % / 24.8 % auf den drei
        aufwaerts-wechseln.

        Gemessen wird in der MITTE der kurve: points[0] ist im halt das
        schiff selbst und traegt ein absichtlich verkuerztes erstes
        teilstueck (siehe _hold_advance), waere als massstab also zu klein.
        """
        limit = self.display_length
        if limit is None:
            return None
        pts = self.points
        if np is None or not isinstance(pts, np.ndarray):
            return None
        n = int(pts.shape[0])
        if n <= 2:
            return None

        mid = n // 2
        spacing = math.hypot(float(pts[mid, 0]) - float(pts[mid - 1, 0]),
                             float(pts[mid, 1]) - float(pts[mid - 1, 1]))
        if spacing > 0.0 and math.isfinite(spacing):
            q = max(1, int(getattr(self, '_display_quantum', 8)))
            # EIN QUANTUM SPIELRAUM, sonst kostet der dauerhaft gesetzte clip
            # das kurvenende. test.apply_predictor_horizon() ruft
            # set_display_length(drawn) inzwischen in JEDEM frame, auch wenn
            # die kurve genau auf `drawn` gerechnet wurde; `spacing` ist aber
            # eine EINZELNE sehne aus der kurvenmitte, also nur ein schaetzer
            # fuer den mittleren abstand. Ohne spielraum kippt der vergleich
            # bei der kleinsten abweichung nach unten, und die rundung auf das
            # quantum schnitt dann bis zu q punkte ab -- bei 8 punkten x 1 Mm
            # grundabstand acht sichtbar fehlende Mm am linienende.
            if limit >= spacing * (n - 1 - q):
                return None
            count = int(math.ceil(limit / spacing)) + 1
            count = int(round(count / q)) * q
            return max(2, min(n, count))

        # Entartete kurve (stillstand, NaN) -- alter weg als rueckfall.
        total = self.length
        if total is None or total <= 0.0:
            total = float(self.num_points) * float(self.precision)
        if total <= 0.0 or limit >= total:
            return None
        count = int(math.ceil(n * (limit / total)))
        q = max(1, int(getattr(self, '_display_quantum', 8)))
        count = int(round(count / q)) * q
        return max(2, min(n, count))

    def get_points(self):
        """Die zu ZEICHNENDE kurve (siehe set_display_length).

        Der ausschnitt wird gemerkt und nur neu gebildet, wenn sich das
        zugrundeliegende array oder die zahl aendert. Das ist kein geiz:
        Renderer._make_prediction_line_cache_key nimmt `id(path_points)` in
        den schluessel auf, ein frisch erzeugter view je aufruf wuerde den
        cache also in JEDEM frame verfehlen.
        """
        count = self._display_point_count()
        if count is None:
            self._clear_display_view()
            return self.points
        # `is` statt id(): so haelt der vergleich eine referenz und kann nicht
        # auf eine wiederverwendete id hereinfallen.
        if (self._display_view_base is not self.points
                or self._display_view_limit != count
                or self._display_view is None):
            self._display_view = self.points[:count]
            self._display_view_base = self.points
            self._display_view_limit = count
        return self._display_view

    @staticmethod
    def _points_have_tangents(pts):
        """Traegt diese punkteliste brauchbare geschwindigkeits-spalten?

        Alles oder nichts, und das ist keine vereinfachung: ob die tangenten
        geschrieben werden, haengt am KERNEL, nicht am einzelnen punkt --
        die RKN-pfade schreiben sie durchgehend, die sehnen-pfade (ASPI,
        blankes RK4) durchgehend nicht. Geprueft wird mit numpy, weil der
        scan-kernel unter `fastmath` keine NaN erkennen kann (siehe
        _refine_apsis_numba).
        """
        if np is None or not isinstance(pts, np.ndarray):
            return False
        if pts.ndim != 2 or pts.shape[1] < 5 or pts.shape[0] < 2:
            return False
        return bool(np.all(np.isfinite(pts[:, 3:5])))

    def get_apsis_markers(self):
        """Apoapsis/Periapsis-Marker der aktuellen Prädiktionslinie.

        Rückgabe: ndarray (m, 5) mit spalten x, y, t_abs, kind, r —
        kind 0.0 = periapsis, 1.0 = apoapsis; r = abstand zum referenz-
        körper in metern. x/y sind absolute (baryzentrische) weltkoords
        des zugehörigen predictor-punkts, t_abs die absolute sim-zeit,
        damit der renderer zeitabhängig frame-transformieren kann.
        Leer wenn kein referenzkörper gewählt ist. Ergebnis wird über die
        identität des punkte-arrays gecacht: der O(n)-scan läuft nur wenn
        eine neue trajektorie geswappt (oder geschnitten) wurde.
        """
        if not self.apsis_markers_enabled:
            return self._empty_apsis_array()
        # Bewusst die GEZEICHNETE kurve: marker jenseits des sichtbaren
        # endes haetten keine linie unter sich.
        pts = self.get_points()
        if np is None or not isinstance(pts, np.ndarray) or pts.shape[0] < 3:
            return self._empty_apsis_array()
        snapshot = self._last_swapped_snapshot
        if snapshot is None:
            return self._empty_apsis_array()
        try:
            ref_index = int(snapshot.get("reference_body_index", -1))
        except Exception:
            ref_index = -1
        if ref_index < 0:
            return self._empty_apsis_array()

        cache_key = (id(pts), int(pts.shape[0]), ref_index)
        if cache_key == self._apsis_cache_key:
            return self._apsis_markers

        # Zeitraffer-halt: nur weich invalidiert (vorn verbraucht, hinten
        # angestueckelt, kopf angeschmiegt) -- die uebrigen punkte stehen
        # bit-identisch, also stehen auch ihre marker. Innerhalb des
        # rescan-fensters nur die abgelaufenen marker herausfiltern statt
        # alle punkte neu zu scannen; das lief sonst ZWEIMAL pro frame
        # (HUD-telemetrie und renderer sehen je ein anderes array).
        now_ts = time.perf_counter()
        if (getattr(self, '_apsis_soft_stale', False)
                and bool(getattr(self, 'hold_enabled', False))
                and isinstance(self._apsis_markers, np.ndarray)
                and int(getattr(self, '_apsis_scan_generation', -1))
                == int(getattr(self, '_points_generation', 0))
                and (now_ts - float(getattr(self, '_apsis_last_scan_ts', 0.0))
                     < float(getattr(self, 'apsis_hold_rescan_s', 0.25)))):
            markers = self._apsis_markers
            if markers.shape[0] > 0 and pts.shape[1] >= 3:
                head_t = float(pts[0, 2])
                if float(markers[:, 2].min()) < head_t:
                    markers = markers[markers[:, 2] >= head_t]
                    self._apsis_markers = markers
            self._apsis_cache_key = cache_key
            return self._apsis_markers

        try:
            markers, count = _find_apsis_markers_numba(
                pts,
                # DIE BEZUGSZEIT MUSS ZU DEN PUNKTZEITEN PASSEN, NICHT ZUR UHR.
                #
                # Der kernel rechnet `pts[i, 2] - base_sim_time` in eine
                # LOKALE zeit zurueck und propagiert damit den referenz-
                # koerper aus `body_x`/`body_y` -- und die wurden bei
                # `snapshot["sim_time"]` abgegriffen. Hat jemand die
                # zeitspalte seither verschoben (die starre nachfuehrung in
                # _anchor_first_point tut das), passt die spalte nicht mehr
                # zum schnappschuss: der koerper wird um genau diesen versatz
                # zu weit vorn gelesen, waehrend die kurve nur um die
                # SCHIFFSbewegung mitgezogen wurde. Was uebrig bleibt, ist die
                # relativbewegung -- und die schlaegt voll auf den
                # angezeigten Pe/Ap-abstand durch.
                float(snapshot.get("sim_time", 0.0))
                + float(getattr(self, "_points_time_offset", 0.0)),
                ref_index,
                snapshot["body_x"],
                snapshot["body_y"],
                snapshot["body_m"],
                snapshot["body_scripted"],
                snapshot["body_a"],
                snapshot["body_e"],
                snapshot["body_theta"],
                snapshot["body_arg"],
                snapshot["body_parent"],
                float(snapshot["G"]),
                1 if bool(snapshot.get("use_time_dependent_bodies", True)) else 0,
                int(self.apsis_max_markers),
                # Der selbst vorangestellte kopf ist die WELT-position des
                # schiffs und gehoert nicht zu dieser kurve -- siehe den
                # trend-scan im kernel. Das gilt jetzt in BEIDEN betriebsarten
                # (echtzeit wie halt), weil beide dieselbe mechanik benutzen;
                # ohne kopf steht das flag auf False und der aufruf ist
                # bit-identisch zu vorher.
                1 if bool(getattr(self, '_synthetic_head', False)) else 0,
                # OB DIE TANGENTEN-SPALTEN BRAUCHBAR SIND, WIRD HIER
                # ENTSCHIEDEN, NICHT IM KERNEL. Der marker wird auf derselben
                # kubik plaziert, die der renderer zeichnet -- die braucht die
                # geschwindigkeits-spalten. Die sehnen-kernel (ASPI, blankes
                # RK4) schreiben dort absichtlich NaN, und im kernel laesst
                # sich das nicht abfragen: er ist `fastmath=True`, und
                # gemessen liefern dort SOWOHL `math.isfinite(nan)` ALS AUCH
                # `nan == nan` den wert True. Also numpy, ausserhalb.
                1 if self._points_have_tangents(pts) else 0,
            )
            self._apsis_markers = markers[:int(count)].copy()
        except Exception as exc:
            # Dieser fang hat schon einen uebersetzungsfehler des kernels
            # verschluckt (readonly-notizblock, siehe _no_body_memo): die
            # marker verschwanden im spiel ohne jede meldung. Ein leeres
            # ergebnis ist ein voellig normaler zustand -- eine AUSNAHME ist
            # es nicht, also wird sie einmal gemeldet.
            if not getattr(self, "_apsis_scan_error_logged", False):
                self._apsis_scan_error_logged = True
                try:
                    print(f"PREDICTOR: apsis-scan fehlgeschlagen: {exc!r}", flush=True)
                except Exception:
                    pass
            self._apsis_markers = self._empty_apsis_array()
        self._apsis_cache_key = cache_key
        self._apsis_soft_stale = False
        self._apsis_last_scan_ts = now_ts
        self._apsis_scan_generation = int(getattr(self, '_points_generation', 0))
        return self._apsis_markers

    def interpolation_error_floor(self, sample_limit=256):
        """Kleinster zeichenfehler, den diese punkteliste ueberhaupt zulaesst.

        Der gezeichnete fehler zerfaellt in zwei UNABHAENGIGE anteile:

            fehler = |Hermite - wahrheit|  +  |polygon - Hermite|
                     (haengt am PUNKTABSTAND)  (haengt an der UNTERTEILUNG)

        Die zeichenzeit-unterteilung drueckt nur den zweiten term. Der erste
        ist die klassische schranke des kubischen Hermite-polynoms,
        ``h^4/384 * max|f''''|``; fuer eine bahn mit lokalem kruemmungsradius
        ``R`` und punktabstand ``c`` ist das

            floor ~ c^4 / (384 * R^3)

        Gegen numerisch integrierte kreisboegen ueber drei zehnerpotenzen von
        ``R`` und ``c`` geprueft: passt auf 0.4 % genau.

        Das ist der grund, warum die feinen sprossen der toleranz-leiter
        (1 mm, 1 cm) mit dem heutigen punktabstand NICHT erreichbar sind --
        dafuer muesste neu integriert werden, nicht feiner ausgewertet. Die
        methode gibt den wert zurueck, damit der renderer die angeforderte
        sprosse daran klemmen und den ERREICHTEN wert anzeigen kann, statt
        eine genauigkeit zu behaupten, die die linie nicht hat.

        Rueckgabe: meter, oder ``None`` wenn nicht bestimmbar.
        """
        points = self.points
        if np is None or not isinstance(points, np.ndarray):
            return None
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
            return None

        n = int(points.shape[0])
        # AUSDUENNEN DARF DEN PUNKTABSTAND NICHT VERAENDERN. Ein einfaches
        # `linspace` ueber die liste tut genau das: die drei punkte eines
        # tripels liegen dann nicht mehr eine, sondern mehrere stuetzweiten
        # auseinander -- und weil der boden mit c^4 geht, kommt ein vielfaches
        # heraus (gemessen 120 m statt 7.6 m auf derselben bahn). Gezogen
        # werden deshalb ANFANGSINDIZES, und jedes tripel bleibt benachbart.
        triples = n - 2
        if triples < 1:
            return None
        limit = int(max(1, min(int(sample_limit), triples)))
        if limit < triples:
            starts = np.unique(np.linspace(0, triples - 1, limit).astype(np.int64))
        else:
            starts = np.arange(triples, dtype=np.int64)

        p0 = points[starts, :2]
        p1 = points[starts + 1, :2]
        p2 = points[starts + 2, :2]

        a = p1 - p0
        b = p2 - p1
        la = np.hypot(a[:, 0], a[:, 1])
        lb = np.hypot(b[:, 0], b[:, 1])
        lc = np.hypot(p2[:, 0] - p0[:, 0], p2[:, 1] - p0[:, 1])

        # Umkreisradius der drei aufeinanderfolgenden punkte = lokaler
        # kruemmungsradius. Vierfache dreiecksflaeche ueber das kreuzprodukt.
        cross = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        with np.errstate(divide='ignore', invalid='ignore'):
            radius = (la * lb * lc) / (2.0 * cross)
            spacing = 0.5 * (la + lb)
            floor = (spacing ** 4) / (384.0 * radius ** 3)

        floor = floor[np.isfinite(floor)]
        if floor.size == 0:
            # Perfekt gerade linie: ein kubisches polynom trifft sie exakt.
            return 0.0
        return float(np.max(floor))

    def get_precision_factor(self):
        if self.base_precision <= 0.0:
            return 1.0
        return self.precision / self.base_precision

    def get_display_length(self):
        # The true traced horizon: length pins it, but the num_points ceiling
        # can clip it when spacing is finer than length/num_points. Report what
        # is actually integrated, not length * coarsen (which only held under
        # the old precision<->horizon coupling).
        if self.length is None:
            return None
        eff = self._effective_precision()
        if not (eff > 0.0):
            return self.length
        return min(self.length, self.num_points * eff)

    def set_precision(self, meters: float):
        meters = float(meters)
        if meters <= 0.0:
            raise ValueError("precision must be > 0")
        self.precision = meters
        # Die gehaltene kurve traegt den ALTEN punktabstand. Ohne diesen
        # vermerk schluckt der zeitraffer-halt die umstellung vollstaendig --
        # gemessen: punktzahl und richtung aendern sich um exakt 0.
        #
        # WEICH: der abstand ist kosmetisch, die kurve bleibt bis zum
        # eintreffen der neuen richtig. Kein grund, den hauptthread
        # anzuhalten (siehe invalidate_hold).
        self.invalidate_hold(soft=True)
        if self.rolling_mode:
            self.reset()
        elif self.async_compute:
            self._cancel_pending_job()

    def set_length(self, meters: float | None):
        if meters is None:
            self.length = None
            self.invalidate_hold()
            if self.rolling_mode:
                self.reset()
            elif self.async_compute:
                self._cancel_pending_job()
            return
        meters = float(meters)
        if meters <= 0.0:
            raise ValueError("length must be > 0")
        self.length = meters
        # Der horizont steuert ueber _horizon_spacing_floor() auch den
        # punktabstand -- die gehaltene kurve passt danach nicht mehr.
        #
        # WEICH: sie ist deswegen aber nicht falsch, sondern nur zu kurz bzw.
        # zu lang. Genau das ist der zeitraffer-stufenwechsel, bei dem
        # apply_predictor_horizon() den faktor 1x/4x/16x/64x umstellt -- der
        # harte weg hat dort 34-82 ms im hauptthread gekostet.
        self.invalidate_hold(soft=True)
        if self.rolling_mode:
            self.reset()
        elif self.async_compute:
            self._cancel_pending_job()

    def set_num_points(self, count: int, soft: bool = False):
        """Punktbudget setzen. `soft=True` behaelt die vorhandene kurve.

        Der harte weg (`reset()`) ist richtig, wenn das budget aus einem
        grund wechselt, der die kurve entwertet -- der `P`-umschalter etwa.

        WEICH ist er, wenn das budget nur MITWAECHST, weil der horizont sich
        geaendert hat (siehe `apply_predictor_horizon` in test.py). Die kurve,
        die dann dasteht, ist geometrisch weiterhin richtig; sie hat bloss zu
        wenige oder zu viele punkte. Genau dieselbe lage wie bei
        `set_length()` -- und dort hat der harte weg im zeitraffer 34-82 ms
        im hauptthread gekostet, weil `update()` sofort synchron neu rechnete
        (§17). Der zeitraffer-schritt verstellt den horizont bei JEDEM
        stufenwechsel, das budget also mit.
        """
        self.num_points = max(0, int(count))
        if not soft:
            self.reset()
            return
        self.invalidate_hold(soft=True)
        if self.rolling_mode:
            self.reset()
        elif self.async_compute:
            self._cancel_pending_job()
