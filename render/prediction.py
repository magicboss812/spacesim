"""Das zeichnen der vorhersagelinie.

Der teure teil ist nicht das zeichnen, sondern die AUSWAHL: aus bis zu 40 000
gerechneten punkten die paar hundert zu finden, die die kurve auf diesem
bildschirm bei diesem zoom tragen. Siehe .claude/rules/predictor.md und
tests/prediction_detail_test.py.
"""
import math
import time

import numpy as np

from render.line_kernels import (
    _LINE_KERNELS_OK,
    _compact_min_step_numba,
    _densify_numba,
    _max_gap_refine_numba,
    _rdp_keep_numba,
)


class PredictionDrawMixin:
    """Die vorhersagelinie: abtastung, Hermite-verfeinerung, Ap/Pe-marker.

    EINE PROGNOSEKURVE WIRD VERBRAUCHT, NIE VERSCHOBEN. Die punkte gehoeren
    der BAHN, nicht dem augenblick -- siehe CLAUDE.md und
    tests/apsis_stability_test.py."""

    def _points_count(self, points):
        if points is None:
            return 0
        try:
            return len(points)
        except Exception:
            return 0

    def _point_xy(self, point):
        if hasattr(point, 'x') and hasattr(point, 'y'):
            return float(point.x), float(point.y)
        return float(point[0]), float(point[1])

    def _effective_sampling_tolerance(self, camera):
        scale = abs(float(camera.scale))
        reference_scale = max(self.prediction_sampling_reference_scale, 1e-30)
        zoom_factor = max(1.0, scale / reference_scale)
        tolerance = self.prediction_sampling_tolerance_px / zoom_factor
        tolerance = min(self.prediction_sampling_max_tolerance_px, tolerance)
        return max(self.prediction_sampling_min_tolerance_px, tolerance)

    def _effective_max_segment_step(self, camera):
        scale = abs(float(camera.scale))
        reference_scale = max(self.prediction_sampling_reference_scale, 1e-30)
        zoom_factor = max(1.0, scale / reference_scale)
        step = self.prediction_sampling_max_segment_px / math.sqrt(zoom_factor)
        # Allow smaller max-segment when zoomed in; keep a small floor to
        # avoid degenerate zero-length subdivisions.
        return max(0.5, step)

    def _compact_min_step_indices(self, xs, ys, min_step2):
        """Python-fassung von `_compact_min_step_numba` (fallback ohne numba).

        Gibt die zu behaltenden INDIZES zurueck, nicht die punkte -- so
        arbeitet der zeichenweg auch ohne numba durchgehend auf spalten.
        """
        n = len(xs)
        if n == 0:
            return []
        keep = [0]
        lx = float(xs[0])
        ly = float(ys[0])
        for i in range(1, n):
            sx = float(xs[i])
            sy = float(ys[i])
            dx = sx - lx
            dy = sy - ly
            if dx * dx + dy * dy >= min_step2:
                keep.append(i)
                lx = sx
                ly = sy
        last = keep[-1]
        if xs[last] != xs[n - 1] or ys[last] != ys[n - 1]:
            keep.append(n - 1)
        return keep

    def _max_gap_refine_indices(self, keep_indices, xs, ys, max_seg):
        """Python-fassung von `_max_gap_refine_numba` (fallback ohne numba)."""
        refined = [int(keep_indices[0])]
        for i in range(1, len(keep_indices)):
            start_idx = refined[-1]
            end_idx = int(keep_indices[i])
            if end_idx <= start_idx:
                continue

            seg_dx = float(xs[end_idx]) - float(xs[start_idx])
            seg_dy = float(ys[end_idx]) - float(ys[start_idx])
            seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)

            if seg_len > max_seg:
                steps = max(2, int(math.ceil(seg_len / max_seg)))
                for step_i in range(1, steps):
                    candidate = start_idx + int(round((end_idx - start_idx) * (step_i / steps)))
                    if candidate <= refined[-1]:
                        candidate = refined[-1] + 1
                    if candidate >= end_idx:
                        break
                    refined.append(candidate)

            if end_idx > refined[-1]:
                refined.append(end_idx)

        return np.asarray(refined, dtype=np.int64)

    def _densify_screen_columns(self, xs, ys, max_segment_px):
        """Wie `_densify_screen_run`, aber auf spalten -- gibt (n, 2) zurueck."""
        n = int(len(xs))
        out = np.empty((n, 2), dtype=np.float64)
        if n < 2:
            if n:
                out[0, 0] = xs[0]
                out[0, 1] = ys[0]
            return out

        max_segment = max(0.5, float(max_segment_px))
        if _LINE_KERNELS_OK:
            dx, dy = _densify_numba(
                np.ascontiguousarray(xs, dtype=np.float64),
                np.ascontiguousarray(ys, dtype=np.float64),
                max_segment)
            dense = np.empty((dx.shape[0], 2), dtype=np.float64)
            dense[:, 0] = dx
            dense[:, 1] = dy
            return dense

        run = np.empty((n, 2), dtype=np.float64)
        run[:, 0] = xs
        run[:, 1] = ys
        return np.asarray(
            self._densify_screen_run([tuple(p) for p in run], max_segment_px),
            dtype=np.float64)

    def _densify_screen_run(self, run, max_segment_px):
        if len(run) < 2:
            return run

        max_segment = max(0.5, float(max_segment_px))
        dense = [run[0]]
        for i in range(len(run) - 1):
            x0, y0 = run[i]
            x1, y1 = run[i + 1]
            dx = x1 - x0
            dy = y1 - y0
            seg_len = math.sqrt(dx * dx + dy * dy)

            if seg_len > max_segment:
                parts = int(math.ceil(seg_len / max_segment))
                parts = max(2, min(256, parts))
                for p in range(1, parts):
                    t = p / parts
                    dense.append((x0 + dx * t, y0 + dy * t))

            dense.append((x1, y1))

        return dense

    def _prediction_point_key(self, points, index):
        try:
            point = points[index]
            if hasattr(point, 'x') and hasattr(point, 'y'):
                return (float(point.x), float(point.y), None)
            t = None
            try:
                if hasattr(point, '__len__') and len(point) >= 3:
                    t = float(point[2])
            except Exception:
                t = None
            return (float(point[0]), float(point[1]), t)
        except Exception:
            return None

    def _make_prediction_line_cache_key(self, path_points, input_count, camera, anchor_world):
        shape = getattr(path_points, 'shape', None)
        if shape is not None:
            try:
                shape_key = tuple(int(v) for v in shape)
            except Exception:
                shape_key = (int(input_count),)
        else:
            shape_key = (int(input_count),)

        anchor_key = None
        if anchor_world is not None:
            try:
                anchor_key = (float(anchor_world[0]), float(anchor_world[1]))
            except Exception:
                anchor_key = None

        active_frame = self._active_frame()
        return (
            id(path_points),
            shape_key,
            int(input_count),
            self._prediction_point_key(path_points, 0),
            self._prediction_point_key(path_points, input_count - 1),
            float(camera.position.x),
            float(camera.position.y),
            float(camera.scale),
            int(self.width),
            int(self.height),
            id(active_frame),
            getattr(active_frame, 'label', None),
            id(self._target_frame),
            self._target_frame_label,
            self._plotting_frame_label,
            float(self._frame_time_s),
            anchor_key,
            float(self.prediction_sampling_tolerance_px),
            float(self.prediction_sampling_min_step_px),
            float(self.prediction_sampling_max_points),
            float(self.prediction_sampling_max_segment_px),
            float(self.prediction_sampling_reference_scale),
            float(self.prediction_visibility_margin_px),
            int(self.prediction_render_max_raw_scan),
            int(self.prediction_render_max_draw_points),
            None if self.prediction_render_max_world_length is None else float(self.prediction_render_max_world_length),
            None if self.prediction_render_max_screen_length_px is None else float(self.prediction_render_max_screen_length_px),
            bool(self.prediction_hermite_enabled),
            float(self.prediction_detail_scale),
            int(self.prediction_hermite_max_subdiv),
            tuple(float(v) for v in self.prediction_error_ladder_m),
        )

    def _prediction_scan_indices(self, raw_count, stats):
        try:
            max_scan = int(self.prediction_render_max_raw_scan)
        except Exception:
            max_scan = 0
        indices = self._iter_prediction_indices_evenly(raw_count, max_scan)
        if len(indices) >= 2:
            stride_est = max(1, int(round((int(raw_count) - 1) / float(len(indices) - 1))))
        else:
            stride_est = 1
        stats['raw_stride'] = stride_est
        stats['skipped_by_stride'] = max(0, int(raw_count) - len(indices))
        return indices

    def _refocus_scan_indices(self, indices, coords, raw_count, margin_px, stats):
        """Das ROH-scan-budget dorthin legen, wo die linie im BILD liegt.

        `_prediction_scan_indices` verteilt seine 3000 stichproben
        GLEICHMAESSIG ueber die ganze punkteliste. Solange der horizont kurz
        ist, faellt das nicht auf; sobald er es nicht mehr ist, ist es der
        ganze fehler. Gemessen auf einem transfer Erde -> Neptun (horizont
        4.5e12 m, 40000 gespeicherte punkte, punktabstand 1.125e8 m):

            stride 13.3  ->  GEZEICHNETER punktabstand 1.5e9 m

        Bei dem massstab des screenshots (1.06e-6 px/m) sind das 1590 px je
        stuetzstelle. Die ganze begegnung mit Neptun -- bogen ~6.8e8 m, also
        6 GESPEICHERTE punkte -- bekommt damit **0.45 gezeichnete** und wird
        zu einer einzigen sehne, die am planeten vorbeischiesst. Genau das
        sind die langen geraden im bild.

        Zwei folgen, und beide stehen im bericht:

        1. Die kubische Hermite-verfeinerung (`_hermite_refine_world`)
           ueberbrueckt dann eine sehne von 8.9 rad bahnwinkel. Innerhalb
           eines bogens von ~0.7 rad liegt ihr fehler bei 0.09 px, darueber
           ist sie schlicht eine andere kurve.
        2. Die auswahl WANDERT. `count` faellt im zeitraffer jeden frame um
           die vorn verbrauchten punkte (`Predictor._hold_advance`), und
           `step = (count-1)/(max_scan-1)` haengt daran -- gemessen springen
           die gewaehlten absoluten indizes bis zu einen ganzen punkt weit,
           also 119 px, von frame zu frame. Die stuetzstellen der
           gezeichneten linie huepfen damit seitlich hin und her, waehrend
           die gespeicherten punkte bit-identisch stehen: das ist das
           "schwingende seil" an jeder kurve und jedem vorbeiflug.

        Der ausweg ist nicht mehr budget, sondern ein besser verteiltes:
        was im bild liegt, wird MIT STRIDE 1 abgetastet -- dann gibt es gar
        keine phase mehr, die wandern koennte --, und der rest behaelt die
        grobe gleichverteilung. Ist das sichtbare stueck zu gross fuer das
        budget (herausgezoomt, die ganze bahn im bild), faellt es stetig auf
        einen groesseren stride zurueck; dann ist ein gespeicherter punkt
        ohnehin unter einem pixel breit.

        Rueckgabe: neues index-array, oder ``None`` -- dann bleibt alles wie
        bisher (kurze linien, in denen ohnehin jeder punkt abgetastet wird,
        kommen hier gar nicht erst an).
        """
        if np is None:
            return None
        idx = np.asarray(indices, dtype=np.int64)
        n = int(idx.size)
        raw_count = int(raw_count)
        if n < 2 or n >= raw_count:
            return None
        try:
            max_scan = int(self.prediction_render_max_raw_scan)
        except Exception:
            return None
        if max_scan <= 0:
            return None
        if coords is None:
            return None
        sx, sy = coords
        sx = np.asarray(sx, dtype=np.float64)
        sy = np.asarray(sy, dtype=np.float64)
        if sx.shape[0] != n or sy.shape[0] != n:
            return None

        margin = float(margin_px)
        x_lo, x_hi = -margin, float(self.width) + margin
        y_lo, y_hi = -margin, float(self.height) + margin

        # SEGMENT gegen das sichtfeld, nicht punkt. Bei stride 13 ist eine
        # sehne 1590 px lang und der schirm 1280 -- die sehne, die ueber die
        # begegnung laeuft, hat oft KEINEN eigenen endpunkt im bild. Ein
        # reiner punkttest fände dort nichts und liesse alles beim alten.
        # Der huellkoerper-test ist bewusst konservativ: er nimmt zu viel
        # mit, nie zu wenig.
        ax, bx = sx[:-1], sx[1:]
        ay, by = sy[:-1], sy[1:]
        finite = (np.isfinite(ax) & np.isfinite(bx)
                  & np.isfinite(ay) & np.isfinite(by))
        seg_hit = (finite
                   & (np.minimum(ax, bx) <= x_hi) & (np.maximum(ax, bx) >= x_lo)
                   & (np.minimum(ay, by) <= y_hi) & (np.maximum(ay, by) >= y_lo))
        if not seg_hit.any():
            return None

        keep = np.zeros(n, dtype=bool)
        keep[:-1] |= seg_hit
        keep[1:] |= seg_hit
        # Eine grobe stuetzweite luft nach beiden seiten: der uebergang
        # zwischen feinem und grobem stueck soll ausserhalb des bildes liegen.
        dilated = keep.copy()
        dilated[1:] |= keep[:-1]
        dilated[:-1] |= keep[1:]
        keep = dilated
        if keep.all():
            # Alles im bild -- die gleichverteilung IST hier die richtige
            # antwort, und ein feinerer stride passt ohnehin nicht ins budget.
            return None

        # Rohbereiche, die fein abgetastet werden sollen.
        flips = np.flatnonzero(np.diff(keep.astype(np.int8)) != 0) + 1
        bounds = np.concatenate(([0], flips, [n]))
        fine_ranges = []
        fine_total = 0
        for start, end in zip(bounds[:-1], bounds[1:]):
            if not keep[start]:
                continue
            lo = int(idx[start])
            hi = int(idx[end - 1])
            fine_ranges.append((lo, hi))
            fine_total += hi - lo + 1
        if not fine_ranges or fine_total <= 0:
            return None

        # Ein viertel des budgets bleibt der groben abtastung ausserhalb des
        # bildes. Sie traegt nichts zum bild bei -- der laufweg wird am
        # bildrand ohnehin geschnitten (`_build_clipped_polyline_runs`) --,
        # muss aber die luecken ueberbruecken, damit ein wiedereintritt ins
        # bild an der richtigen stelle sitzt.
        outside = idx[~keep]
        out_budget = max(2, max_scan // 4)
        if outside.size > out_budget:
            pick = np.unique(np.rint(
                np.linspace(0, outside.size - 1, out_budget)).astype(np.int64))
            outside = outside[pick]
        fine_budget = max(2, max_scan - int(outside.size))

        stride = 1 if fine_total <= fine_budget else int(
            math.ceil(fine_total / float(fine_budget)))
        # Der grobe stride, den wir ersetzen wollen. Bringt der feine nichts,
        # bleibt es beim alten -- eine zweite stapel-projektion umsonst.
        coarse_stride = raw_count / float(n)
        if stride >= coarse_stride:
            return None

        parts = [outside, np.array([0, raw_count - 1], dtype=np.int64)]
        for lo, hi in fine_ranges:
            run = np.arange(lo, hi + 1, stride, dtype=np.int64)
            if run.size == 0:
                run = np.array([lo], dtype=np.int64)
            if run[-1] != hi:
                run = np.append(run, hi)
            parts.append(run)
        merged = np.unique(np.concatenate(parts))

        if stats is not None:
            stats['raw_focus_stride'] = int(stride)
            stats['raw_focus_points'] = int(merged.size)
            stats['raw_focus_ranges'] = int(len(fine_ranges))
            stats['skipped_by_stride'] = max(0, raw_count - int(merged.size))
        return merged

    def _iter_prediction_indices_evenly(self, count, max_scan):
        """Gleichmaessige stichprobe der rohpunkte -- GEMERKT, nicht neu gebaut.

        Das ergebnis haengt einzig an (count, max_scan). Der cache greift
        aber NUR, solange beide gleich bleiben -- und `count` aendert sich im
        zeitraffer jeden frame, weil der halt die punkteliste vorn beschneidet
        (siehe Predictor._hold_advance). Deshalb muss auch der neubau billig
        sein; er laeuft ueber numpy statt ueber eine Python-schleife.
        """
        count = int(count)
        max_scan = int(max_scan)

        key = (count, max_scan)
        cached = getattr(self, '_prediction_indices_cache', None)
        if cached is not None and cached[0] == key:
            return cached[1]

        indices = self._build_prediction_indices(count, max_scan)
        self._prediction_indices_cache = (key, indices)
        return indices

    def _build_prediction_indices(self, count, max_scan):
        if count <= 0:
            return []

        if max_scan <= 0 or count <= max_scan:
            return list(range(count))

        if max_scan == 1:
            return [0]

        step = (count - 1) / float(max_scan - 1)

        if np is not None:
            # np.rint rundet wie Pythons round() zur GERADEN zahl hin, die
            # stichprobe ist damit dieselbe wie in der schleife unten.
            idx = np.rint(np.arange(max_scan, dtype=np.float64) * step)
            idx = idx.astype(np.int64)
            np.clip(idx, 0, count - 1, out=idx)
            # Nur AUFEINANDERFOLGENDE wiederholungen fallen weg -- genau das
            # tut die schleife mit ihrem `last`.
            keep = np.empty(idx.shape, dtype=bool)
            keep[0] = True
            np.not_equal(idx[1:], idx[:-1], out=keep[1:])
            return idx[keep]

        indices = []
        last = -1
        for i in range(max_scan):
            idx = int(round(i * step))
            idx = max(0, min(count - 1, idx))
            if idx != last:
                indices.append(idx)
                last = idx
        return indices

    def _cap_runs_by_screen_length(self, runs, max_screen_length_px, stats):
        if max_screen_length_px is None:
            return runs
        try:
            max_length = float(max_screen_length_px)
        except Exception:
            return runs
        if max_length <= 0.0:
            stats['clipped_or_rejected'] = stats.get('clipped_or_rejected', 0) + sum(len(run) for run in runs)
            return []

        run_lengths = []
        total_length = 0.0
        for run in runs:
            if len(run) < 2:
                run_lengths.append(0.0)
                continue
            length = 0.0
            for i in range(len(run) - 1):
                lx, ly = run[i]
                sx, sy = run[i + 1]
                dx = float(sx) - float(lx)
                dy = float(sy) - float(ly)
                length += math.sqrt(dx * dx + dy * dy)
            run_lengths.append(length)
            total_length += length

        if total_length <= max_length:
            return runs

        capped = []
        rejected = 0
        for run, run_length in zip(runs, run_lengths):
            if len(run) < 2 or run_length <= 1e-12:
                rejected += len(run)
                continue

            remaining = max_length * (run_length / total_length)
            if remaining <= 1e-12:
                rejected += len(run)
                continue

            current = [run[0]]
            for i in range(len(run) - 1):
                lx, ly = current[-1]
                sx, sy = run[i + 1]
                dx = float(sx) - float(lx)
                dy = float(sy) - float(ly)
                seg_len = math.sqrt(dx * dx + dy * dy)
                if seg_len <= remaining:
                    current.append((sx, sy))
                    remaining -= seg_len
                    continue
                if seg_len > 1e-12 and remaining > 0.0:
                    frac = remaining / seg_len
                    current.append((lx + dx * frac, ly + dy * frac))
                break
            if len(current) >= 2:
                capped.append(current)
            rejected += max(0, len(run) - len(current))

        stats['clipped_or_rejected'] = stats.get('clipped_or_rejected', 0) + rejected
        return capped

    def _cap_runs_by_point_budget(self, runs, max_points, stats):
        capped = self._limit_polyline_runs_evenly(runs, max_points)
        rejected = max(0, sum(len(run) for run in runs) - sum(len(run) for run in capped))
        stats['clipped_or_rejected'] = stats.get('clipped_or_rejected', 0) + rejected
        return capped

    def _limit_polyline_runs_evenly(self, runs, max_points):
        max_points = int(max_points)
        if max_points <= 0:
            return []

        total = sum(len(run) for run in runs)
        if total <= max_points:
            return runs

        limited = []
        points_left = max_points
        runs_left = len(runs)

        for run in runs:
            if runs_left <= 0 or points_left <= 1:
                break

            budget = max(2, points_left // runs_left)
            if len(run) <= budget:
                limited.append(run)
                points_left -= len(run)
            elif np is not None and isinstance(run, np.ndarray):
                # Gleiche stichprobe wie unten, nur als index-rechnung.
                # np.round rundet -- wie Pythons round() -- die haelfte zur
                # geraden zahl, die gewaehlten indizes sind also dieselben.
                step = (len(run) - 1) / float(budget - 1)
                idx = np.round(np.arange(budget, dtype=np.float64) * step)
                idx = np.clip(idx, 0.0, float(len(run) - 1)).astype(np.int64)
                if idx.shape[0] > 1:
                    unique = np.empty(idx.shape[0], dtype=bool)
                    unique[0] = True
                    np.not_equal(idx[1:], idx[:-1], out=unique[1:])
                    idx = idx[unique]
                if idx.shape[0] >= 2:
                    limited.append(run[idx])
                points_left -= int(idx.shape[0])
            else:
                step = (len(run) - 1) / float(budget - 1)
                sampled = []
                last = -1
                for i in range(budget):
                    idx = int(round(i * step))
                    idx = max(0, min(len(run) - 1, idx))
                    if idx != last:
                        sampled.append(run[idx])
                        last = idx
                if len(sampled) >= 2:
                    limited.append(sampled)
                points_left -= len(sampled)

            runs_left -= 1

        return limited

    def draw_prediction(self, path_points, camera, anchor_world=None, predictor=None):

        input_count = self._points_count(path_points)
        stats = {
            'raw_in': int(input_count),
            'raw_points': int(input_count),
            'scanned': 0,
            'scanned_points': 0,
            'visible': 0,
            'runs': 0,
            'draw_points': 0,
            'drawn': 0,
            'skipped_by_stride': 0,
            'clipped_or_rejected': 0,
            'prepare_ms': 0.0,
            'draw_ms': 0.0,
            'cache_hit': False,
        }
        if input_count == 0:
            self.debug_info['prediction_points_in'] = 0
            self.debug_info['prediction_points_drawn'] = 0
            self._last_prediction_render_stats = stats
            return

        # Blending ist global aktiv (ctx.enable in _init_opengl); die alten
        # textur-/blend-state-resets der fixed-function-pipeline entfallen.
        prepare_t0 = time.perf_counter()
        half_w = self.width * 0.5
        half_h = self.height * 0.5
        camera_frame_xy = self._frame_camera_xy(camera)
        self._debug_prediction_frame_transform(path_points, predictor=predictor)

        # debug-ausgabe: schiff-welt-position und ersten predictor-punkt anzeigen
        try:
            pred0_x, pred0_y = self._point_xy(path_points[0])
            ship_world_x, ship_world_y = (float(anchor_world[0]), float(anchor_world[1])) if anchor_world is not None else (pred0_x, pred0_y)
            if self.debug_predictor:
                print(f"PRED_DBG_POS: ship=({ship_world_x:.6e},{ship_world_y:.6e}) predictor_first=({pred0_x:.6e},{pred0_y:.6e})")
        except Exception:
            pass

        effective_tolerance = self._effective_sampling_tolerance(camera)
        effective_min_step = max(0.05, min(self.prediction_sampling_min_step_px, effective_tolerance * 0.6))
        effective_max_segment = self._effective_max_segment_step(camera)
        max_draw_points = max(2, min(int(self.prediction_sampling_max_points), int(self.prediction_render_max_draw_points)))

        # EIN BUDGET FUER DIE GANZE KETTE. Der zeichenweg hat drei stellen, an
        # denen er punkte wegwirft (min-schritt-verdichtung, RDP, run-kappung)
        # und eine, an der er welche setzt (kubische unterteilung). Solange die
        # wegwerfenden ihre eigene, aelteren zoom-heuristik folgende toleranz
        # benutzen, macht die eine haelfte zunichte, was die andere aufbaut:
        # gemessen 1990 m abweichung bei einer zusage von 1000 m, weil die
        # RDP-toleranz bis 0.25 px gehen darf -- bei 4.4e-5 px/m sind das
        # 5700 m. Also bekommen alle stufen ihren anteil an DERSELBEN zusage.
        self._prediction_detail_budget = None
        if self.prediction_hermite_enabled:
            budget = self._prediction_error_budget(camera)
            if budget is not None:
                self._prediction_detail_budget = budget
                eps_px = budget[1]
                # Die anteile addieren sich im schlimmsten fall, also muessen
                # sie zusammen unter 1 bleiben: 0.5 unterteilung + 0.25 RDP +
                # 0.1 verdichtung = 0.85. Mit 0.5/0.5/0.25 (summe 1.25) lag die
                # gemessene abweichung bei 1.12 der zusage -- knapp darueber.
                effective_tolerance = max(1e-3, min(effective_tolerance, eps_px * 0.25))
                effective_min_step = max(1e-3, min(effective_min_step, eps_px * 0.1))

        cache_key = self._make_prediction_line_cache_key(path_points, input_count, camera, anchor_world)
        if cache_key == self._prediction_line_cache_key_value and self._prediction_line_cache_points is not None:
            sampled_runs = self._prediction_line_cache_points
            stats.update(dict(self._prediction_line_cache_stats))
            stats['raw_in'] = int(input_count)
            stats['raw_points'] = int(input_count)
            stats['prepare_ms'] = (time.perf_counter() - prepare_t0) * 1000.0
            stats['cache_hit'] = True
        else:
            # Bewegte origin-frames (z.B. Erde): origin-position über das
            # predictor-zeitfenster interpolieren statt pro punkt propagieren.
            # Auf das aktive frame begrenzt und danach wieder gelöscht, damit
            # körper/spuren (aktuelle zeit) exakt transformiert bleiben.
            active_frame = self._active_frame()
            interp_window_set = False
            try:
                try:
                    p_first = path_points[0]
                    p_last = path_points[input_count - 1]
                    t_first = float(p_first[2]) if hasattr(p_first, '__len__') and len(p_first) >= 3 else None
                    t_last = float(p_last[2]) if hasattr(p_last, '__len__') and len(p_last) >= 3 else None
                except Exception:
                    t_first = t_last = None
                if t_first is not None and t_last is not None and t_last > t_first:
                    active_frame.set_origin_interp_window(t_first, t_last, int(input_count))
                    interp_window_set = True
                sampled_runs = self._adaptive_prediction_screen_points(
                    path_points,
                    camera,
                    tolerance_px=effective_tolerance,
                    min_step_px=effective_min_step,
                    max_segment_px=effective_max_segment,
                    max_points=max_draw_points,
                    margin_px=self.prediction_visibility_margin_px,
                    anchor_world=anchor_world,
                    stats=stats,
                    camera_frame_xy=camera_frame_xy,
                )
            finally:
                if interp_window_set:
                    active_frame.set_origin_interp_window(0.0, 0.0, 0)
            stats['prepare_ms'] = (time.perf_counter() - prepare_t0) * 1000.0

        self.debug_info['prediction_points_in'] = input_count
        self.debug_info['prediction_points_drawn'] = sum(len(run) for run in sampled_runs)

        # Store small sample for debugging and optionally print it.
        try:
            if sampled_runs and len(sampled_runs[0]) > 0:
                sample_n = min(5, len(sampled_runs[0]))
                screen_samples = [sampled_runs[0][i] for i in range(sample_n)]
                frame_samples = []
                for sx, sy in screen_samples:
                    fx = camera_frame_xy[0] + (sx - half_w) / float(camera.scale)
                    fy = camera_frame_xy[1] - (sy - half_h) / float(camera.scale)
                    frame_samples.append((fx, fy))
                self.debug_info['prediction_sample_screen'] = screen_samples
                self.debug_info['prediction_sample_frame'] = frame_samples
                if self.debug_predictor:
                    print('PRED_DBG: in=', input_count, 'drawn=', self.debug_info['prediction_points_drawn'])
                    print('PRED_DBG: screen_samples=', screen_samples)
                    print('PRED_DBG: frame_samples=', frame_samples)
        except Exception:
            pass

        if len(sampled_runs) == 0:
            stats['drawn'] = 0
            self._last_prediction_render_stats = stats
            return

        sampled_runs = self._cap_runs_by_point_budget(sampled_runs, max_draw_points, stats)
        self._prediction_line_cache_key_value = cache_key
        self._prediction_line_cache_points = sampled_runs
        self._prediction_line_cache_stats = dict(stats)
        draw_t0 = time.perf_counter()
        for run in sampled_runs:
            if len(run) < 2:
                continue
            self._draw_polyline(run, color=(1.0, 1.0, 1.0, 0.6), width=2.0)
        self._draw_apsis_markers(predictor, camera, camera_frame_xy)
        stats['draw_ms'] = (time.perf_counter() - draw_t0) * 1000.0
        stats['drawn'] = sum(len(run) for run in sampled_runs)
        stats['draw_points'] = int(stats['drawn'])
        stats['runs'] = len(sampled_runs)
        self.debug_info['prediction_points_drawn'] = int(stats['drawn'])

        # WAS DIE LINIE WIRKLICH KANN, NICHT WAS SIE VERSPRICHT. Die sprosse
        # der leiter ist das ziel; der interpolations-boden des predictors
        # (punktabstand^4 / 384 R^3) kann darueber liegen, und dann ist ER die
        # erreichte genauigkeit. Beides anzuzeigen ist der einzige weg, eine
        # zugesagte praezision nicht still zu erfinden.
        floor = None
        try:
            get_floor = getattr(predictor, 'interpolation_error_floor', None)
            if get_floor is not None:
                floor = get_floor()
        except Exception:
            floor = None
        stats['detail_floor_m'] = floor
        eps_m = stats.get('detail_eps_m')
        if eps_m is not None:
            achieved = eps_m if floor is None else max(eps_m, floor)
            stats['detail_achieved_m'] = achieved
            self.debug_info['prediction_detail_target_m'] = eps_m
            self.debug_info['prediction_detail_achieved_m'] = achieved
            self.debug_info['prediction_detail_added'] = int(
                stats.get('hermite_added', 0))

        self._last_prediction_render_stats = stats

    def _format_apsis_distance(self, r):
        if r >= 1e9:
            return f"{r / 1e9:.2f}Gm"
        if r >= 1e6:
            return f"{r / 1e6:.2f}Mm"
        if r >= 1e3:
            return f"{r / 1e3:.1f}km"
        return f"{r:.0f}m"

    def _draw_apsis_markers(self, predictor, camera, camera_frame_xy=None):
        """Zeichnet apoapsis/periapsis-marker des predictors auf die linie.

        Marker kommen als (m, 5)-array (x, y, t_abs, kind, r) aus
        predictor.get_apsis_markers(); die transformation nutzt die
        zeitabhängige frame-transformation, damit die marker in bewegten
        plot-frames auf der gezeichneten linie bleiben.
        """
        # Die trefferliste wird bei JEDEM aufruf geleert, auch wenn nichts
        # gezeichnet wird: sonst stuende der schwebezettel des HUDs noch ueber
        # einem marker, den es gar nicht mehr gibt.
        self.apsis_marker_hits.clear()
        if predictor is None or not self.show_apsis_markers:
            return
        get_markers = getattr(predictor, 'get_apsis_markers', None)
        if get_markers is None:
            return
        try:
            markers = get_markers()
        except Exception:
            return
        count = self._points_count(markers)
        if count == 0:
            return

        r_px = float(self.apsis_marker_radius_px)

        # AUSBLENDEN NACH SCHIRMGROESSE DER BAHN, NICHT NACH ZOOM-SCHWELLE.
        # `m[4]` ist der apsis-abstand zum bezugskoerper in metern, `* scale`
        # also der apsis-radius in PIXELN -- ein direktes mass dafuer, wie
        # gross die bahn am schirm ist. Wird sie klein, ruecken Pe/Ap an die
        # schiffs- und die Erde-marke heran; das smoothstep zwischen
        # `fade_min_px` und `fade_full_px` blendet sie dann sauber weg, statt
        # sie uebereinanderzustapeln. Damit ist die alte "ein draw je farbe"-
        # buendelung hin (jeder marker hat jetzt seine eigene deckkraft) --
        # bei real 1 Pe + 1 Ap, selten je zwei, ist das ein draw je marker
        # und faellt nicht ins gewicht.
        scale = abs(float(getattr(camera, 'scale', 0.0)))
        fade_min = float(self.apsis_marker_fade_min_px)
        fade_full = float(self.apsis_marker_fade_full_px)

        for i in range(count):
            try:
                m = markers[i]
                wx = float(m[0])
                wy = float(m[1])
                t_abs = float(m[2])
                is_apo = float(m[3]) >= 0.5
                dist = float(m[4])
            except Exception:
                continue

            size_px = dist * scale
            if fade_full > fade_min:
                u = (size_px - fade_min) / (fade_full - fade_min)
                u = max(0.0, min(1.0, u))
                alpha_mult = u * u * (3.0 - 2.0 * u)
            else:
                alpha_mult = 1.0 if size_px >= fade_min else 0.0
            if alpha_mult <= 0.003:
                continue

            sx, sy = self._world_to_screen_xy_at_time(wx, wy, camera, t_abs, camera_frame_xy)
            if not (math.isfinite(sx) and math.isfinite(sy)):
                continue
            if not self._is_on_screen(sx, sy, 32.0):
                continue

            # Die schirmposition an das HUD melden -- der schwebezettel
            # (ui/hud/apsis_tooltip.py) trifft damit genau die raute, die
            # hier gezeichnet wird, und nicht eine selbst nachgerechnete.
            if self.apsis_tooltip_enabled:
                self.apsis_marker_hits.append(
                    (sx, sy, r_px, bool(is_apo), dist, t_abs, alpha_mult))

            label = "Ap" if is_apo else "Pe"
            base = (0.45, 0.75, 1.0) if is_apo else (1.0, 0.62, 0.25)
            col = (base[0], base[1], base[2], 0.95 * alpha_mult)

            # Nord -> Ost -> Sued -> West -> Nord, als vier strecken.
            north = (sx, sy - r_px)
            east = (sx + r_px, sy)
            south = (sx, sy + r_px)
            west = (sx - r_px, sy)
            self._draw_line_segments(
                (north, east, east, south, south, west, west, north),
                color=col, width=2.0)

            text = f"{label} {self._format_apsis_distance(dist)}"
            try:
                # Diamant + linie laufen über den line-shader (top-down, sy
                # wächst nach unten), text über die ortho-konvention
                # (y nach oben). _blit_text_topdown rechnet das um.
                entry = self._get_label_texture(text, self.font_small)
                tw = float(entry[1]) if entry else 0.0
                label_x = sx - tw / 2.0
                self._blit_text_topdown(text, label_x, sy + r_px + 4.0,
                                        self.font_small,
                                        color=(1.0, 1.0, 1.0, alpha_mult))
            except Exception:
                pass

    def _squared_point_line_distance(self, px, py, ax, ay, bx, by):
        abx = bx - ax
        aby = by - ay
        ab2 = abx * abx + aby * aby
        if ab2 <= 1e-18:
            dx = px - ax
            dy = py - ay
            return dx * dx + dy * dy

        apx = px - ax
        apy = py - ay
        t = (apx * abx + apy * aby) / ab2
        t = max(0.0, min(1.0, t))
        proj_x = ax + t * abx
        proj_y = ay + t * aby
        dx = px - proj_x
        dy = py - proj_y
        return dx * dx + dy * dy

    def _rdp_indices(self, points, tolerance_px):
        n = len(points)
        if n <= 2:
            return [0, n - 1] if n == 2 else [0]

        tol2 = tolerance_px * tolerance_px
        keep = [False] * n
        keep[0] = True
        keep[-1] = True
        stack = [(0, n - 1)]

        while stack:
            start, end = stack.pop()
            if end <= start + 1:
                continue

            ax, ay = points[start]
            bx, by = points[end]
            max_d2 = -1.0
            index = -1

            for i in range(start + 1, end):
                px, py = points[i]
                d2 = self._squared_point_line_distance(px, py, ax, ay, bx, by)
                if d2 > max_d2:
                    max_d2 = d2
                    index = i

            if max_d2 > tol2 and index != -1:
                keep[index] = True
                stack.append((start, index))
                stack.append((index, end))

        return [i for i, k in enumerate(keep) if k]

    def _adaptive_prediction_screen_points(self,
                                           path_points,
                                           camera,
                                           tolerance_px,
                                           min_step_px,
                                           max_segment_px,
                                           max_points,
                                           margin_px,
                                           anchor_world=None,
                                           stats=None,
                                           camera_frame_xy=None):
        if stats is None:
            stats = {}
        half_w = self.width * 0.5
        half_h = self.height * 0.5
        if camera_frame_xy is None:
            camera_frame_xy = self._frame_camera_xy(camera)
        scale = float(camera.scale)

        screen_points = []

        raw_count = len(path_points)
        indices = self._prediction_scan_indices(raw_count, stats)
        max_world_length = self.prediction_render_max_world_length
        try:
            max_world_length = None if max_world_length is None else max(0.0, float(max_world_length))
        except Exception:
            max_world_length = None

        # Schneller weg: alle punkte auf einmal projizieren. Greift nur, wenn
        # keine weltlaengen-begrenzung aktiv ist (die braucht den laufenden
        # summenwert und damit die schleife) und der rahmen eine stapel-
        # transformation anbietet. Sonst faellt es auf die schleife darunter
        # zurueck -- dieselbe rechnung, nur langsam.
        batch = None
        if max_world_length is None:
            batch = self._project_prediction_batch(
                path_points, indices, camera, camera_frame_xy, margin_px)

        if batch is not None:
            # `screen_points` ist auf diesem weg IMMER None: die spalten in
            # `coords` sind die punkte. Die tupel-liste, die hier frueher
            # entstand, wurde weiter unten ohnehin wieder in arrays
            # zurueckverwandelt (gemessen 4000 tupel je frame, nur um sie
            # danach wegzuwerfen).
            screen_points, visible_count, coords = batch

            # ERST DAS ROH-BUDGET UMVERTEILEN, DANN VERFEINERN. Die kubische
            # nachverdichtung unten kann nur so gut sein wie die stuetzstellen,
            # zwischen denen sie interpoliert; ueber eine sehne von mehreren
            # radianten bahnwinkel ist sie eine andere kurve. Siehe
            # _refocus_scan_indices -- dort stehen die messwerte.
            focused = self._refocus_scan_indices(
                indices, coords, raw_count, margin_px, stats)
            if focused is not None:
                focused_batch = self._project_prediction_batch(
                    path_points, focused, camera, camera_frame_xy, margin_px)
                if focused_batch is not None:
                    indices = focused
                    screen_points, visible_count, coords = focused_batch

            # ZWISCHEN den groben stuetzstellen kubisch nachlegen -- so fein,
            # wie der bildschirm es zeigt, und nur dort, wo etwas zu sehen
            # ist. Schlaegt das fehl (kein tangenten-paar, kein stapel-
            # rahmen, kein budget), bleibt es bei den groben punkten und die
            # linie sieht aus wie vorher.
            budget = getattr(self, '_prediction_detail_budget', None)
            if budget is not None:
                eps_m, eps_px, rung = budget
                stats['detail_eps_m'] = eps_m
                stats['detail_eps_px'] = eps_px
                stats['detail_rung'] = rung
                refined = self._hermite_refine_world(
                    path_points, indices, coords, camera,
                    eps_px * 0.5, margin_px,
                    int(self.prediction_render_max_draw_points), stats)
                if refined is not None:
                    dense = self._project_prediction_batch(
                        refined, np.arange(refined.shape[0], dtype=np.int64),
                        camera, camera_frame_xy, margin_px)
                    if dense is not None:
                        screen_points, visible_count, coords = dense

            scanned_count = (len(coords[0]) if screen_points is None
                             else len(screen_points))
            stats['scanned'] = stats.get('scanned', 0) + scanned_count
            stats['scanned_points'] = stats.get('scanned_points', 0) + scanned_count
            stats['visible'] = stats.get('visible', 0) + visible_count
            return self._runs_from_screen_points(
                screen_points, camera, tolerance_px, min_step_px,
                max_segment_px, max_points, margin_px, stats, coords=coords)

        prev_world = None
        prev_time = None
        world_accum = 0.0

        for i in indices:
            point = path_points[i]
            px, py = self._point_xy(point)

            # If point includes timestamp (x,y,t), use time-aware projection.
            sample_time = None
            try:
                if hasattr(point, '__len__') and len(point) >= 3:
                    sample_time = float(point[2])
            except Exception:
                sample_time = None

            stop_after_point = False
            if max_world_length is not None and prev_world is not None:
                seg_dx_world = px - prev_world[0]
                seg_dy_world = py - prev_world[1]
                seg_len_world = math.sqrt(seg_dx_world * seg_dx_world + seg_dy_world * seg_dy_world)
                if world_accum + seg_len_world > max_world_length:
                    remaining_world = max_world_length - world_accum
                    if seg_len_world > 1e-12 and remaining_world > 0.0:
                        frac = remaining_world / seg_len_world
                        px = prev_world[0] + seg_dx_world * frac
                        py = prev_world[1] + seg_dy_world * frac
                        if sample_time is not None and prev_time is not None:
                            sample_time = prev_time + (sample_time - prev_time) * frac
                    else:
                        px, py = prev_world
                        sample_time = prev_time
                    stop_after_point = True
                else:
                    world_accum += seg_len_world

            if sample_time is not None:
                sx, sy = self._world_to_screen_xy_at_time(
                    px,
                    py,
                    camera,
                    sample_time,
                    camera_frame_xy=camera_frame_xy,
                )
            else:
                frame_x, frame_y = self._frame_transform_xy(px, py)
                sx = half_w + (frame_x - camera_frame_xy[0]) * scale
                sy = half_h - (frame_y - camera_frame_xy[1]) * scale
            stats['scanned'] = stats.get('scanned', 0) + 1
            stats['scanned_points'] = stats.get('scanned_points', 0) + 1

            near_visible = self._is_on_screen(sx, sy, margin_px)
            if near_visible:
                stats['visible'] = stats.get('visible', 0) + 1

            screen_points.append((sx, sy))

            prev_world = (px, py)
            prev_time = sample_time
            if stop_after_point:
                stats['clipped_or_rejected'] = stats.get('clipped_or_rejected', 0) + max(0, raw_count - i - 1)
                break

        return self._runs_from_screen_points(
            screen_points, camera, tolerance_px, min_step_px, max_segment_px,
            max_points, margin_px, stats)

    def _project_prediction_batch(self, path_points, indices, camera,
                                  camera_frame_xy, margin_px):
        """Alle stichproben-punkte in EINEM rutsch projizieren.

        Gibt ``(None, sichtbar_anzahl, (sx, sy))`` zurueck oder None, wenn
        der schnelle weg nicht anwendbar ist. Rechnerisch identisch zur
        schleife in _adaptive_prediction_screen_points -- dieselbe rahmen-
        transformation, derselbe massstab, dieselbe y-spiegelung.

        Der erste eintrag ist bewusst None: die punkte leben ab hier nur
        noch in den spalten (sx, sy). Die frueher hier gebaute liste aus
        (x, y)-tupeln kostete bei 4000 punkten je frame mehr als die
        projektion selbst und wurde vom zeichenweg sofort wieder in arrays
        zurueckverwandelt.

        Motivation: gemessen lag die punktweise projektion bei 5.6 ms je
        frame, praktisch alles davon Python-aufruf-overhead ueber 3000
        punkte, von denen am ende 196 gezeichnet werden.
        """
        if np is None:
            return None
        if not isinstance(path_points, np.ndarray):
            return None
        if path_points.ndim != 2 or path_points.shape[1] < 3:
            return None
        if len(indices) == 0:
            empty = np.empty(0, dtype=np.float64)
            return (None, 0, (empty, empty))

        frame = self._active_frame()
        transform = getattr(frame, 'to_this_frame_xy_arrays', None)
        if transform is None:
            return None

        idx = np.asarray(indices, dtype=np.int64)
        sub = path_points[idx]
        xs = np.ascontiguousarray(sub[:, 0], dtype=np.float64)
        ys = np.ascontiguousarray(sub[:, 1], dtype=np.float64)
        ts = np.ascontiguousarray(sub[:, 2], dtype=np.float64)

        try:
            transformed = transform(ts, xs, ys)
        except Exception:
            return None
        if transformed is None:
            return None
        frame_x, frame_y = transformed

        scale = float(camera.scale)
        sx = self.width * 0.5 + (frame_x - camera_frame_xy[0]) * scale
        sy = self.height * 0.5 - (frame_y - camera_frame_xy[1]) * scale

        # Gleiche schranke wie _is_on_screen -- rein diagnostisch (stats).
        margin = float(margin_px)
        visible = int(np.count_nonzero(
            (sx >= -margin) & (sx <= self.width + margin)
            & (sy >= -margin) & (sy <= self.height + margin)))
        return (None, visible, (sx, sy))

    # ------------------------------------------------------------------
    # Aufloesungsgetriebene verfeinerung
    # ------------------------------------------------------------------

    def _prediction_error_budget(self, camera):
        """Erlaubte abweichung der gezeichneten linie -- in metern und pixeln.

        Der bildschirm gibt den wunsch vor (`eps_px / view_scale`), die
        toleranz-leiter quantisiert ihn. Gewaehlt wird die GROESSTE sprosse,
        die den wunsch noch einhaelt; darunter/darueber wird geklemmt. Damit
        ist die zugesagte genauigkeit nie schlechter als angefordert, und sie
        aendert sich nur an wenigen diskreten zoom-schwellen statt stetig.

        Rueckgabe: ``(eps_m, eps_px, rung_index)`` oder ``None``.
        """
        try:
            scale = abs(float(camera.scale))
        except Exception:
            return None
        if not (scale > 0.0) or not math.isfinite(scale):
            return None

        try:
            detail = float(self.prediction_detail_scale)
        except Exception:
            detail = 1.0
        detail = max(1e-3, detail)
        eps_px = 0.3 / detail

        ladder = self.prediction_error_ladder_m
        try:
            rungs = sorted(float(v) for v in ladder if float(v) > 0.0)
        except Exception:
            rungs = []
        if not rungs:
            return (eps_px / scale, eps_px, -1)

        wanted_m = eps_px / scale
        index = 0
        for i, rung in enumerate(rungs):
            if rung <= wanted_m:
                index = i
            else:
                break
        if wanted_m < rungs[0]:
            index = 0
        eps_m = rungs[index]
        # Die sprosse ist das versprechen; die pixel-toleranz muss sich ihr
        # beugen, sonst wird feiner gezeichnet als zugesagt (kostet) oder
        # groeber (bricht das versprechen).
        return (eps_m, eps_m * scale, index)

    def _hermite_refine_world(self, path_points, indices, coords, camera,
                              eps_px, margin_px, budget, stats):
        """Kubische zwischenpunkte setzen -- nur sichtbar, nur so fein wie noetig.

        Arbeitet in WELTKOORDINATEN und gibt ein dichteres ``(m, 3)``-array
        ``[x, y, t]`` zurueck, das anschliessend durch dieselbe
        stapel-projektion laeuft wie die groben punkte. In *screen*-space zu
        unterteilen waere falsch: jeder zwischenpunkt gehoert zu einer eigenen
        zeit, und ein bewegter oder rotierender plot-rahmen bildet ihn deshalb
        anders ab als seine nachbarn.

        Die unterteilungszahl je segment kommt aus der flachheits-schranke fuer
        kubische kurven (siehe unten). Das ist eine heuristik fuer die ANZAHL --
        die gezeichneten punkte selbst werden einzeln zu ihrer eigenen zeit
        projiziert und sind damit exakt.

        Rueckgabe: ``(m, 3)``-array oder ``None`` (dann bleibt alles wie bisher).
        """
        if np is None or not self.prediction_hermite_enabled:
            return None
        if not isinstance(path_points, np.ndarray) or path_points.ndim != 2:
            return None
        # Ohne die geschwindigkeits-spalten gibt es keine tangente und damit
        # nichts zu interpolieren.
        if path_points.shape[1] < 5:
            return None
        if coords is None:
            return None

        idx = np.asarray(indices, dtype=np.int64)
        if idx.size < 2:
            return None

        sx, sy = coords
        if sx.shape[0] != idx.shape[0]:
            return None

        sub = path_points[idx]
        x0 = sub[:-1, 0]; y0 = sub[:-1, 1]; t0 = sub[:-1, 2]
        x1 = sub[1:, 0];  y1 = sub[1:, 1];  t1 = sub[1:, 2]
        vx0 = sub[:-1, 3]; vy0 = sub[:-1, 4]
        vx1 = sub[1:, 3];  vy1 = sub[1:, 4]
        dt = t1 - t0

        sx0 = sx[:-1]; sy0 = sy[:-1]
        sx1 = sx[1:];  sy1 = sy[1:]

        third = dt / 3.0

        # Ohne endliche tangente an BEIDEN enden gibt es kein polynom -- der
        # abschnitt bleibt dann eine gerade. Genau dafuer schreiben die
        # sehnen-kernel NaN in die geschwindigkeitsspalten.
        usable = (np.isfinite(vx0) & np.isfinite(vy0)
                  & np.isfinite(vx1) & np.isfinite(vy1)
                  & np.isfinite(dt) & (dt > 0.0))
        if not np.any(usable):
            return None

        # SICHTBARKEIT ZUERST, DANN RECHNEN. Die flachheits-schaetzung kostet
        # zwei zusaetzliche rahmen-transformationen je segment -- bei 3000
        # segmenten gemessen 2.3 ms je frame, waehrend am ende fuenf punkte
        # dazukamen, weil nur ~200 segmente ueberhaupt im bild lagen. Die
        # vorauswahl laeuft deshalb allein auf den schon vorhandenen
        # bildschirm-endpunkten.
        #
        # Als grosszuegige schranke fuer die auslenkung dient die sehnenlaenge
        # selbst: eine kubische kurve liegt in der konvexen huelle ihrer
        # kontrollpunkte, und die liegen bei einer glatten bahn rund eine
        # drittel sehne neben den endpunkten.
        margin = float(margin_px)
        chord = np.hypot(sx1 - sx0, sy1 - sy0)
        chord = np.where(np.isfinite(chord), chord, 0.0)
        lo_x = np.minimum(sx0, sx1) - chord
        hi_x = np.maximum(sx0, sx1) + chord
        lo_y = np.minimum(sy0, sy1) - chord
        hi_y = np.maximum(sy0, sy1) + chord
        visible = ((hi_x >= -margin) & (lo_x <= self.width + margin)
                   & (hi_y >= -margin) & (lo_y <= self.height + margin))
        candidate = visible & usable
        if not np.any(candidate):
            return None

        # Zweite differenzen der Bezier-kontrollpunkte
        # (b0 = p0, b1 = p0 + v0*dt/3, b2 = p1 - v1*dt/3, b3 = p1), ausmultipliziert
        # -- in WELTKOORDINATEN und dann mit dem massstab in pixel umgerechnet,
        # statt die kontrollpunkte eigens auf den schirm zu projizieren.
        #
        # Das darf man, weil alle plot-rahmen STARR sind (verschiebung plus
        # drehung): beides laesst laengen unveraendert, und eine zweite
        # differenz ist eine laenge. Gemessen kostete die eigene projektion
        # zwei zusaetzliche rahmen-transformationen ueber ~3000 segmente und
        # damit 1.9 ms je frame -- fuer eine ZAHL, die ohnehin nur die
        # unterteilungsstufe waehlt.
        #
        # Nicht erfasst wird die zusaetzliche kruemmung, die ein ROTIERENDER
        # rahmen ueber die dauer eines segments selbst erzeugt. Fuer den
        # Erde-Sonne-richtungsrahmen sind das ~0.14 m gegen 0.83 m echte
        # woelbung; erst bei sehr schnell drehenden rahmen waere das relevant,
        # und dort faengt `prediction_sampling_max_segment_px` die luecke ab.
        d1x = (x1 - x0) - (2.0 * vx0 + vx1) * third
        d1y = (y1 - y0) - (2.0 * vy0 + vy1) * third
        d2x = (x0 - x1) + (vx0 + 2.0 * vx1) * third
        d2y = (y0 - y1) + (vy0 + 2.0 * vy1) * third
        scale = abs(float(camera.scale))
        second = np.maximum(np.hypot(d1x, d1y), np.hypot(d2x, d2y)) * scale
        second = np.where(np.isfinite(second) & candidate, second, 0.0)

        # Wie viele teilstuecke braucht es, damit der polygonzug hoechstens
        # `tol` von der kurve abweicht?
        #
        #   d <= max|B''| / (8 n^2)     und     max|B''| <= 6 M
        #   =>  d <= 0.75 M / n^2       =>  n = ceil(sqrt(0.75 M / tol))
        #
        # Der faktor ist nachgerechnet, nicht geraten: mit dem in vielen
        # rasterisierern kursierenden sqrt(3)/8 statt 3/4 wird um 1.86 zu
        # grob unterteilt (n = 4 statt 8), und genau das war messbar -- 1122 m
        # abweichung bei einer zusage von 1000 m, exakt sehne/n^2.
        tol = max(1e-4, float(eps_px))
        n_seg = np.ceil(np.sqrt(0.75 * second / tol))
        n_seg = np.where(np.isfinite(n_seg), n_seg, 1.0)
        counts = np.clip(n_seg - 1.0, 0.0,
                         float(max(0, int(self.prediction_hermite_max_subdiv)))
                         ).astype(np.int64)
        counts[~candidate] = 0

        total = int(counts.sum())
        if total <= 0:
            return None

        # BUDGET GLEICHMAESSIG DRUECKEN, NIE ABSCHNEIDEN. Wer das budget von
        # vorn aufbraucht, verliert das ende der linie -- und ein fehlender
        # horizont macht die anzeige unbrauchbar (Ap/Pe weg, CLOSEST leer),
        # waehrend eine gleichmaessig groebere linie nur etwas kantiger ist.
        room = int(budget) - int(idx.shape[0])
        if room < 0:
            room = 0
        if total > room:
            if room <= 0:
                return None
            factor = room / float(total)
            counts = np.floor(counts * factor).astype(np.int64)
            total = int(counts.sum())
            if total <= 0:
                return None
            stats['hermite_budget_limited'] = True

        # Auswertungsstellen: je segment `counts[i]` innere parameter.
        seg_id = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
        offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
        local = np.arange(total, dtype=np.int64) - offsets[seg_id]
        s = (local + 1).astype(np.float64) / (counts[seg_id] + 1).astype(np.float64)

        s2 = s * s
        s3 = s2 * s
        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2

        seg_dt = dt[seg_id]
        ix = (h00 * x0[seg_id] + h10 * seg_dt * vx0[seg_id]
              + h01 * x1[seg_id] + h11 * seg_dt * vx1[seg_id])
        iy = (h00 * y0[seg_id] + h10 * seg_dt * vy0[seg_id]
              + h01 * y1[seg_id] + h11 * seg_dt * vy1[seg_id])
        it = t0[seg_id] + s * seg_dt

        # Grobe und neue punkte in EINE aufsteigende reihenfolge bringen.
        n_coarse = int(idx.shape[0])
        out = np.empty((n_coarse + total, 3), dtype=np.float64)
        before = np.concatenate(([0], np.cumsum(counts)))
        coarse_slots = np.arange(n_coarse, dtype=np.int64) + before
        out[coarse_slots, 0] = sub[:, 0]
        out[coarse_slots, 1] = sub[:, 1]
        out[coarse_slots, 2] = sub[:, 2]
        insert_slots = coarse_slots[seg_id] + 1 + local
        out[insert_slots, 0] = ix
        out[insert_slots, 1] = iy
        out[insert_slots, 2] = it

        stats['hermite_added'] = total
        stats['hermite_segments'] = int(np.count_nonzero(counts))
        return out

    def _runs_from_screen_points(self, screen_points, camera, tolerance_px,
                                 min_step_px, max_segment_px, max_points,
                                 margin_px, stats, coords=None):
        runs = self._build_clipped_polyline_runs(screen_points, margin_px,
                                                 coords=coords)
        stats['runs'] = len(runs)
        stats['clipped_runs'] = len(runs)
        if not runs:
            stats['draw_points'] = 0
            return []

        # ARRAYS STATT TUPEL-LISTEN. Verdichtung, RDP, luecken-auffuellung
        # und verdichtung-nach-innen sind alle indexoperationen auf denselben
        # zwei koordinaten-spalten. Frueher wanderte die spur nach jeder
        # stufe durch `list(zip(...tolist()))` und zurueck durch
        # `np.asarray` -- bei 4000 punkten je frame gemessen der zweit-
        # groesste einzelposten des zeichenwegs nach dem klippen selbst.
        if coords is not None:
            origin_x = float(coords[0][0])
            origin_y = float(coords[1][0])
        else:
            origin_x = float(screen_points[0][0])
            origin_y = float(screen_points[0][1])

        min_step2 = float(min_step_px) * float(min_step_px)
        tol2 = float(tolerance_px) * float(tolerance_px)
        max_seg = max(0.5, float(max_segment_px))

        sampled_runs = []

        for run in runs:
            run = np.asarray(run, dtype=np.float64)
            if run.ndim != 2 or run.shape[0] < 2:
                continue
            rxs = np.ascontiguousarray(run[:, 0])
            rys = np.ascontiguousarray(run[:, 1])

            run_starts_at_path_origin = (
                abs(rxs[0] - origin_x) < 1e-9 and
                abs(rys[0] - origin_y) < 1e-9
            )

            # Verdichtung: punkte naeher als min_step fallen weg.
            if _LINE_KERNELS_OK:
                cidx = _compact_min_step_numba(rxs, rys, min_step2)
            else:
                cidx = np.asarray(
                    self._compact_min_step_indices(rxs, rys, min_step2),
                    dtype=np.int64)
            cx = rxs[cidx]
            cy = rys[cidx]

            if cx.shape[0] > 2:
                if _LINE_KERNELS_OK:
                    keep_mask = _rdp_keep_numba(cx, cy, tol2)
                    keep_indices = np.nonzero(keep_mask)[0]
                else:
                    keep_indices = np.asarray(
                        self._rdp_indices(list(zip(cx.tolist(), cy.tolist())),
                                          tolerance_px),
                        dtype=np.int64)

                if run_starts_at_path_origin:
                    preserve_count = min(32, cx.shape[0])
                    # Die vereinigung mit 0..preserve_count-1 ist hier KEIN
                    # allgemeiner mengen-schnitt: der zweite operand ist ein
                    # LUECKENLOSER anfang, und `keep_indices` ist bereits
                    # sortiert und doppelfrei. Damit ist das ergebnis genau
                    # "0..preserve-1, dahinter alle keeps >= preserve" --
                    # eine suche plus ein anhaengen, ohne die hash-tabelle,
                    # die np.union1d aufbaut (gemessen 0.44 ms je aufruf,
                    # zwei aufrufe je frame).
                    cut = int(np.searchsorted(keep_indices, preserve_count,
                                              side='left'))
                    tail = keep_indices[cut:]
                    merged = np.empty(preserve_count + tail.shape[0],
                                      dtype=np.int64)
                    merged[:preserve_count] = np.arange(preserve_count,
                                                        dtype=np.int64)
                    merged[preserve_count:] = tail
                    keep_indices = merged

                # Guard against over-aggressive simplification by enforcing
                # a maximum screen-space gap between consecutive kept points.
                if keep_indices.shape[0] > 1:
                    keep_indices = np.ascontiguousarray(
                        keep_indices, dtype=np.int64)
                    if _LINE_KERNELS_OK:
                        keep_indices = _max_gap_refine_numba(
                            keep_indices, cx, cy, max_seg)
                    else:
                        keep_indices = self._max_gap_refine_indices(
                            keep_indices, cx, cy, max_seg)

                sampled_x = cx[keep_indices]
                sampled_y = cy[keep_indices]
            else:
                sampled_x = cx
                sampled_y = cy

            # Densify only the RDP-kept points, not the raw scan.
            # Pre-RDP densification of sparse predictors could expand 3000 samples
            # to 75 000+ linearly-interpolated dummies that RDP discards anyway,
            # making _rdp_indices O(N²) on a huge but information-free array.
            sampled = self._densify_screen_columns(
                sampled_x, sampled_y, max_segment_px)

            if sampled.shape[0] >= 2:
                sampled_runs.append(sampled)

        sampled_runs = self._cap_runs_by_screen_length(
            sampled_runs,
            self.prediction_render_max_screen_length_px,
            stats,
        )
        sampled_runs = self._cap_runs_by_point_budget(sampled_runs, max_points, stats)
        stats['drawn'] = sum(len(run) for run in sampled_runs)
        stats['draw_points'] = stats['drawn']
        stats['runs'] = len(sampled_runs)
        return sampled_runs
