"""Zeichen-primitive: polylinien, ortho-formen, texturen, clipping.

Die clipping-routinen halten die punktzahl klein, BEVOR etwas an die GPU geht
-- eine linie mit 40 000 punkten, von der 30 auf dem schirm liegen, kostet
sonst die volle uebertragung je frame.
"""
import math

import moderngl
import numpy as np
import pygame

from render.line_kernels import _LINE_KERNELS_OK, _clip_runs_numba


class DrawMixin:
    """Die zeichen-primitive: linien, ortho-formen, texturen, clipping.

    Alles hier nimmt BILDSCHIRM-koordinaten. Zwei Y-konventionen treffen sich
    an dieser stelle -- die welt zeichnet top-down (der y-flip sitzt in
    line.vert/body.vert), text und schiff zeichnen ortho (y-up, ursprung unten
    links). `_ortho_y()` ist der uebergang; siehe .claude/rules/rendering.md."""

    def _write_poly_vertices(self, arr):
        """Lädt ein (N,2)-float32-array in den geteilten dynamischen VBO.

        orphan() reallokiert bei bedarf nur den speicher und behält das
        buffer-objekt — die VAOs der line-/ortho-pipeline bleiben gültig.
        """
        self._ensure_poly_vbo()
        data_size = int(arr.nbytes)
        if data_size > int(self._poly_vbo_size):
            self._poly_vbo.orphan(data_size)
            self._poly_vbo_size = data_size
        self._poly_vbo.write(arr)
        return int(arr.shape[0])

    def _draw_polyline(self, run, color=(1.0, 1.0, 1.0, 1.0), width=1.0):
        """Zeichnet eine bildschirm-space polyline (top-down-konvention) via GLSL+VBO."""
        n = len(run)
        if n < 2 or self._line_vao is None:
            return

        try:
            arr = np.asarray(run, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 2:
                arr = arr.reshape((-1, 2))
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr, dtype=np.float32)
        except Exception:
            return

        n = self._write_poly_vertices(arr)
        self._set_line_width(width)
        self._set_uniform(self._line_program, 'u_viewport', '_line_viewport',
                          (float(self.width), float(self.height)))
        self._set_uniform(self._line_program, 'u_color', '_line_color', (
            float(color[0]), float(color[1]), float(color[2]), float(color[3])
        ))
        self._line_vao.render(moderngl.LINE_STRIP, vertices=n)

    def _draw_line_segments(self, points, color=(1.0, 1.0, 1.0, 1.0), width=1.0):
        """Zeichnet PAARWEISE strecken (GL_LINES) in der top-down-konvention.

        Dieselbe pipeline und dieselbe abbildung wie `_draw_polyline`, nur
        ohne den zwang, dass alle punkte EINEN zug bilden. Damit gehen
        mehrere kleine, unverbundene figuren (etwa alle apsis-rauten einer
        farbe) in einem einzigen draw an die GPU.
        """
        n = len(points)
        if n < 2 or self._line_vao is None:
            return

        try:
            arr = np.asarray(points, dtype=np.float32).reshape((-1, 2))
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr, dtype=np.float32)
        except Exception:
            return
        # GL_LINES verbraucht die punkte paarweise; ein einzelner ueberzaehliger
        # punkt wuerde verworfen, hier gar nicht erst hochgeladen.
        if arr.shape[0] % 2:
            arr = arr[:-1]
        if arr.shape[0] < 2:
            return

        n = self._write_poly_vertices(arr)
        self._set_line_width(width)
        self._set_uniform(self._line_program, 'u_viewport', '_line_viewport',
                          (float(self.width), float(self.height)))
        self._set_uniform(self._line_program, 'u_color', '_line_color', (
            float(color[0]), float(color[1]), float(color[2]), float(color[3])
        ))
        self._line_vao.render(moderngl.LINES, vertices=n)

    def _draw_ortho_shape(self, points, color, mode, width=1.0):
        """Zeichnet geometrie in der alten ortho-konvention (y nach oben).

        Ersatz für die früheren immediate-mode-aufrufe unter
        gluOrtho2D(0, w, 0, h): identische pixel-abbildung, nur via shader.
        """
        n = len(points)
        if n < 2 or self._ortho_vao is None:
            return

        try:
            arr = np.asarray(points, dtype=np.float32).reshape((-1, 2))
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr, dtype=np.float32)
        except Exception:
            return

        n = self._write_poly_vertices(arr)
        if mode in (moderngl.LINES, moderngl.LINE_STRIP):
            self._set_line_width(width)
        self._set_uniform(self._ortho_program, 'u_viewport', '_ortho_viewport',
                          (float(self.width), float(self.height)))
        self._set_uniform(self._ortho_program, 'u_color', '_ortho_color', (
            float(color[0]), float(color[1]), float(color[2]), float(color[3])
        ))
        self._ortho_vao.render(mode, vertices=n)

    def _clip_segment_to_rect(self, x0, y0, x1, y1, left, top, right, bottom):
        """
        Liang-Barsky clipping for screen-space line segments.
        Returns (cx0, cy0, cx1, cy1) or None if the segment is fully outside.
        Screen coordinates: x right, y down.
        """
        dx = x1 - x0
        dy = y1 - y0

        u1 = 0.0
        u2 = 1.0

        # Liang-Barsky gegen die vier kanten. Bewusst ohne zwischen-listen/zip:
        # diese funktion läuft pro segment jeder spur-, orbit- und vorhersage-
        # linie und ist damit der meistaufgerufene pro-frame-pfad. die (pi, qi)-
        # paare sind exakt wie zuvor (links, rechts, oben, unten), nur skalar.

        # links: pi = -dx, qi = x0 - left
        pi = -dx
        qi = x0 - left
        if pi == 0.0:
            if qi < 0.0:
                return None
        else:
            t = qi / pi
            if pi < 0.0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            elif t < u1:
                return None
            elif t < u2:
                u2 = t

        # rechts: pi = dx, qi = right - x0
        pi = dx
        qi = right - x0
        if pi == 0.0:
            if qi < 0.0:
                return None
        else:
            t = qi / pi
            if pi < 0.0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            elif t < u1:
                return None
            elif t < u2:
                u2 = t

        # oben: pi = -dy, qi = y0 - top
        pi = -dy
        qi = y0 - top
        if pi == 0.0:
            if qi < 0.0:
                return None
        else:
            t = qi / pi
            if pi < 0.0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            elif t < u1:
                return None
            elif t < u2:
                u2 = t

        # unten: pi = dy, qi = bottom - y0
        pi = dy
        qi = bottom - y0
        if pi == 0.0:
            if qi < 0.0:
                return None
        else:
            t = qi / pi
            if pi < 0.0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            elif t < u1:
                return None
            elif t < u2:
                u2 = t

        return (
            x0 + u1 * dx,
            y0 + u1 * dy,
            x0 + u2 * dx,
            y0 + u2 * dy,
        )

    def _build_clipped_polyline_runs(self, screen_points, margin_px=128.0,
                                     coords=None):
        """
        Converts one logical predictor polyline into multiple visible screen-space runs.
        Important: preserve original segment topology. Never connect visible points
        across an offscreen gap.

        `coords` sind dieselben punkte als (sx, sy)-arrays. Liegen sie vor
        (und ist numba da), laeuft die ganze zustandsmaschine als EIN
        kernel-aufruf -- vorher war das die teuerste einzelne funktion des
        frames (gemessen ~15 ms bei 4000 segmenten, praktisch alles
        Python-schleifen-overhead). Ohne `coords` oder ohne numba bleibt der
        Python-weg darunter, zeichenweise identisch.

        Rueckgabe: liste von ``(n, 2)``-float64-arrays. Der ganze
        linien-zeichenweg rechnet auf arrays weiter; die frueheren listen
        aus (x, y)-tupeln wurden auf dem weg zur GPU ohnehin wieder in
        arrays umgewandelt.

        `screen_points` darf ``None`` sein, WENN `coords` vorliegt -- dann
        sind die spalten die einzige darstellung der punkte und es wird gar
        keine tupel-liste mehr gebaut.
        """
        have_coords = coords is not None and np is not None
        if screen_points is None:
            if not have_coords:
                return []
            point_count = len(coords[0])
        else:
            point_count = len(screen_points)
        if point_count < 2:
            return []

        left = -float(margin_px)
        top = -float(margin_px)
        right = float(self.width) + float(margin_px)
        bottom = float(self.height) + float(margin_px)

        coords_match = have_coords and len(coords[0]) == point_count

        # Aufrufer ohne spalten (bahnlinien, referenz-spuren) bekommen sie
        # hier einmalig -- sonst laufen genau die durch den langsamen
        # Python-klipper, waehrend die vorhersagelinie den kernel nutzt.
        if not coords_match and np is not None and _LINE_KERNELS_OK:
            try:
                arr = np.asarray(screen_points, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] == point_count:
                    coords = (np.ascontiguousarray(arr[:, 0]),
                              np.ascontiguousarray(arr[:, 1]))
                    have_coords = True
                    coords_match = True
            except Exception:
                pass

        if coords_match and _LINE_KERNELS_OK:
            sx, sy = coords
            return self._clipped_runs_from_arrays(
                sx, sy, left, top, right, bottom)

        if screen_points is None:
            # Ohne numba braucht der Python-weg unten die punkte einzeln.
            sx, sy = coords
            screen_points = list(zip(np.asarray(sx).tolist(),
                                     np.asarray(sy).tolist()))

        segment_indices = None
        if have_coords:
            sx, sy = coords
            if coords_match:
                out_left = sx < left
                out_right = sx > right
                out_top = sy < top
                out_bottom = sy > bottom
                trivially_out = (
                    (out_left[:-1] & out_left[1:])
                    | (out_right[:-1] & out_right[1:])
                    | (out_top[:-1] & out_top[1:])
                    | (out_bottom[:-1] & out_bottom[1:])
                )
                segment_indices = np.flatnonzero(~trivially_out)

        runs = []
        run = []

        if segment_indices is None:
            iterator = range(len(screen_points) - 1)
        else:
            iterator = segment_indices

        previous_index = None
        for i in iterator:
            i = int(i)
            # Uebersprungene segmente sind verworfene segmente: der lauf
            # bricht dort ab, sonst wuerde ueber die luecke hinweg verbunden.
            if previous_index is not None and i != previous_index + 1:
                if len(run) >= 2:
                    runs.append(run)
                run = []
            previous_index = i

            x0, y0 = screen_points[i]
            x1, y1 = screen_points[i + 1]

            clipped = self._clip_segment_to_rect(
                float(x0), float(y0),
                float(x1), float(y1),
                left, top, right, bottom
            )

            if clipped is None:
                if len(run) >= 2:
                    runs.append(run)
                run = []
                continue

            cx0, cy0, cx1, cy1 = clipped

            if not run:
                run = [(cx0, cy0), (cx1, cy1)]
                continue

            last_x, last_y = run[-1]
            gap_px = math.hypot(cx0 - last_x, cy0 - last_y)

            if gap_px > 2.0:
                if len(run) >= 2:
                    runs.append(run)
                run = [(cx0, cy0), (cx1, cy1)]
            else:
                run.append((cx1, cy1))

        if len(run) >= 2:
            runs.append(run)

        # Einheitliche rueckgabe mit dem kernel-weg: (n, 2)-arrays.
        if np is None:
            return runs
        return [np.asarray(r, dtype=np.float64) for r in runs]

    def _clipped_runs_from_arrays(self, sx, sy, left, top, right, bottom):
        """Kernel-weg von `_build_clipped_polyline_runs`.

        Ein numba-aufruf statt einer Python-schleife ueber alle segmente;
        die laeufe werden anschliessend nur noch als sichten auf den
        ausgabepuffer herausgeschnitten.
        """
        xs = np.ascontiguousarray(sx, dtype=np.float64)
        ys = np.ascontiguousarray(sy, dtype=np.float64)
        ox, oy, starts, counts = _clip_runs_numba(
            xs, ys, float(left), float(top), float(right), float(bottom))
        if starts.shape[0] == 0:
            return []

        runs = []
        for k in range(starts.shape[0]):
            a = int(starts[k])
            b = a + int(counts[k])
            run = np.empty((b - a, 2), dtype=np.float64)
            run[:, 0] = ox[a:b]
            run[:, 1] = oy[a:b]
            runs.append(run)
        return runs

    def _draw_texture_ortho(self, texture, x, y, width, height, color=(1.0, 1.0, 1.0, 1.0)):
        """Zeichnet eine textur als quad in der ortho-konvention (y nach oben).

        Ersatz für die früheren immediate-mode glTexCoord/glVertex-quads unter
        gluOrtho2D(0, w, 0, h): (x, y) ist die untere linke ecke, texcoord
        (0, 0) liegt ebendort (texturen werden vertikal geflippt hochgeladen).

        color toent die textur multiplikativ (texquad.frag, u_color). Der
        uniform MUSS gesetzt werden -- GL initialisiert uniforms mit 0, ein
        ausgelassenes u_color zeichnet also nichts.
        """
        if self._texquad_vao is None or texture is None:
            return
        # AUF DAS PIXELRASTER RASTEN. Die weltabgeleiteten label-positionen
        # sind subpixelgenau (Erde z. B. bei y=113.7048). Bei LINEAR-filterung
        # verteilt ein solcher versatz jede glyphenzeile auf ZWEI pixelzeilen:
        # der text wird weich und bekommt eine schwache kopie darueber/darunter
        # -- sieht aus wie eine zweite zahl unter der zahl. Gemessen faellt der
        # anteil voll deckender pixel von 19.5 % auf 9 %.
        # Das HUD war nie betroffen, weil es ganzzahlige ursprungswerte nutzt.
        # Die textur wird 1:1 gezeichnet, deshalb genuegt das runden der ecke.
        self._texquad_program['u_rect'].value = (
            round(float(x)), round(float(y)), float(width), float(height)
        )
        # u_viewport ist ueber den ganzen frame konstant, u_color ueber
        # ganze gruppen von beschriftungen -- nur wechsel schreiben.
        self._set_uniform(self._texquad_program, 'u_viewport',
                          '_texquad_viewport',
                          (float(self.width), float(self.height)))
        self._set_uniform(self._texquad_program, 'u_color', '_texquad_color', (
            float(color[0]), float(color[1]), float(color[2]), float(color[3])
        ))
        texture.use(location=0)
        self._texquad_vao.render(moderngl.TRIANGLE_STRIP)

    def _ortho_y(self, y_topdown):
        """Top-down bildschirm-Y (wie _world_to_screen_xy liefert) -> ortho-Y.

        Die welt wird top-down gezeichnet (line.vert flippt y), text und
        schiffs-pfeil laufen dagegen ueber die ortho-konvention (y nach oben,
        ursprung unten links). Ohne diese umrechnung landet alles, was aus
        weltkoordinaten kommt, an der ueber die BILDSCHIRMMITTE gespiegelten
        position -- unsichtbar solange das objekt genau mittig steht, und mit
        wachsendem abstand zur mitte immer weiter daneben.
        """
        return float(self.height) - float(y_topdown)

    def _draw_hud_quad(self, x, y, width, height):
        """Zeichnet die persistente HUD-textur als quad (ohne re-upload)."""
        if self._hud_texture is None:
            return
        self._draw_texture_ortho(self._hud_texture, x, y, width, height)

    def _blit_pygame_surface(self, surface, x, y):
        """Lädt eine pygame Surface in die persistente HUD-textur und zeichnet sie.

        Der upload (tostring + texture.write) ist der teure teil. Aufrufer,
        deren inhalt sich gegenüber dem vorframe nicht geändert hat, überspringen
        diese methode und rufen direkt _draw_hud_quad.
        """
        texture_data = pygame.image.tostring(surface, 'RGBA', True)
        width, height = surface.get_size()

        # Create or resize HUD texture
        if self._hud_texture is None or self._hud_texture_size != (width, height):
            if self._hud_texture is not None:
                try:
                    self._hud_texture.release()
                except Exception:
                    pass
            self._hud_texture = self.ctx.texture((width, height), 4, texture_data)
            self._hud_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._hud_texture_size = (width, height)
        else:
            self._hud_texture.write(texture_data)

        # Textur rendern
        self._draw_hud_quad(x, y, width, height)

    def _is_on_screen(self, sx, sy, margin_px):
        return (-margin_px <= sx <= self.width + margin_px and
                -margin_px <= sy <= self.height + margin_px)

    def _visible_window_runs(self, screen_points, margin_px, coords=None):
        return self._build_clipped_polyline_runs(screen_points, margin_px,
                                                 coords=coords)
