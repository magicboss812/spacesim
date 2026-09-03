"""Das zeichnen der himmelskoerper.

Die detail-leiter (`_body_detail_levels`) entscheidet, ob ein koerper als
GLSL-scheibe, als vektor-form oder nur noch als gesaete pixel-marke erscheint
-- siehe .claude/rules/body-art.md.
"""
import math
from concurrent.futures import ThreadPoolExecutor

import moderngl

import numpy as np

from bodies import icon as body_icon
from bodies import style as body_style


class BodyDrawMixin:
    """Die koerper: scheibe, prozedurale marke, vektor-look, beschriftung,
    auswahlmarke und der treffertest.

    Die GEOMETRIE kommt aus `bodies/style.py` und `bodies/icon.py` (reines
    numpy); hier steht nur, wie sie auf den schirm kommt."""

    def _draw_body_glsl(self, x, y, radius, base_color, atmos_color, atmos_density,
                        light_intensity, light=(0.0, 0.0, 1.0), emissive=1.0,
                        surface_mix=0.0, glow=0.0):
        """Zeichnet einen körper als shader-gesteuertes quad (scheibe + optional atmosphäre + glow).

        `light` ist die richtung ZUR lichtquelle im scheiben-raum (y nach oben);
        `emissive` = 1 schaltet die schattierung ab (stern, positions-icon).
        `surface_mix` verdunkelt die scheibe, sobald die vektor-zeichnung
        darueber liegt -- ohne das leuchtet die volle koerperfarbe durch die
        linien hindurch und die facetten verschwinden.
        """
        if self._body_vao is None:
            return False

        radius_px = max(1.0, float(radius))
        radius_scale = max(0.5, min(2.0, radius_px / 50.0))

        outer_radius = radius_px
        atmos_alpha = 0.0
        atmos_radius = radius_px
        if atmos_density > 0.0:
            # Enger als frueher (war 2.0): mit der neuen kugelschattierung ist
            # der koerper selbst dunkel, und ein halo von zwei radien breite
            # ueberstrahlte dann die halbe bildflaeche.
            atmos_radius = radius_px * 1.22
            outer_radius = max(outer_radius, atmos_radius)
            atmos_alpha = min(float(atmos_density) / 100.0, 1.0) * min(radius_scale, 1.0)

        glow_alpha = 0.0
        if light_intensity > 0.0:
            # Stern: grosser halo. Die alte formel teilte durch 1000 und kam
            # damit auf alpha 4e-4 -- der glow war rechnerisch da und optisch
            # nie zu sehen.
            glow_radius = radius_px * 3.0
            outer_radius = max(outer_radius, glow_radius)
            glow_alpha = min(1.0, 0.22 + float(light_intensity) * 0.30) * radius_scale
        elif glow > 0.0:
            glow_radius = radius_px * 1.28
            outer_radius = max(outer_radius, glow_radius)
            glow_alpha = min(1.0, float(glow))

        core_norm = max(0.001, min(1.0, radius_px / max(outer_radius, 1e-6)))
        if atmos_alpha > 0.0:
            atmos_norm = max(core_norm, min(1.0, atmos_radius / max(outer_radius, 1e-6)))
        else:
            atmos_norm = core_norm

        try:
            prog = self._body_program
            prog['u_center_px'].value = (float(x), float(y))
            prog['u_outer_radius_px'].value = float(outer_radius)
            prog['u_viewport'].value = (float(self.width), float(self.height))
            prog['u_base_color'].value = (
                float(base_color[0]), float(base_color[1]), float(base_color[2])
            )
            prog['u_atmos_color'].value = (
                float(atmos_color[0]), float(atmos_color[1]), float(atmos_color[2])
            )
            prog['u_core_radius_norm'].value = float(core_norm)
            prog['u_atmos_radius_norm'].value = float(atmos_norm)
            prog['u_atmos_alpha'].value = float(atmos_alpha)
            prog['u_glow_alpha'].value = float(glow_alpha)
            prog['u_light'].value = (float(light[0]), float(light[1]), float(light[2]))
            prog['u_ambient'].value = float(self.body_ambient)
            prog['u_emissive'].value = float(emissive)
            prog['u_surface_mix'].value = float(surface_mix)

            self._body_vao.render(moderngl.TRIANGLE_STRIP)
            return True
        except Exception:
            return False

    def _body_icon_entry(self, body):
        """Gepacktes zellfeld und farbstufen dieser marke, gecacht.

        Der schluessel ist wie bei `_body_style_key` bewusst NICHT `id(body)`,
        sondern das, was das muster bestimmt: ein neu geladener koerper mit
        denselben angaben bekommt dieselbe marke. Hoechstens ein eintrag je
        koerper und variante, gebaut im hauptthread -- der bau sind ein paar
        dutzend zellen, kein grund fuer einen worker wie bei body_style.
        """
        seed = body_icon.seed_for(body, self.body_icon_seed_offset)
        color = tuple(int(c) for c in tuple(getattr(body, 'color', (255, 255, 255)))[:3])
        key = (seed, str(self.body_icon_variant), int(self.body_icon_grid), color)
        entry = self._body_icon_cache.get(key)
        if entry is None:
            try:
                cells = body_icon.build_icon(
                    seed, self.body_icon_variant, self.body_icon_grid)
                entry = (cells, body_icon.icon_palette(color))
            except Exception as exc:
                # Wie bei body_style: einmal scheitern heisst nie wieder
                # versuchen. Sonst kostet ein kaputter bau jeden frame.
                self.debug_info['body_icon_error'] = f"{type(exc).__name__}: {exc}"
                entry = False
            self._body_icon_cache[key] = entry
        return entry

    def _draw_body_icon(self, body, x, y, radius, r, g, b, fade=1.0):
        """Positions-marke eines körpers, konstanter bildschirmgröße.

        `radius` ist der GEZEICHNETE marken-radius -- siehe
        `_body_icon_draw_radius_px` fuer die skalierung mit dem echten
        koerper; `fade` blendet die marke über dem echten körper aus, siehe
        `_body_icon_fade`.

        Zwei wege. `body_icon_style = "disc"` zeichnet die alte flache scheibe
        über denselben GLSL-körper-shader wie der volle körper: der
        vertex-shader (body.vert) erwartet top-down-screen-koordinaten und
        spiegelt y intern (`ndc.y = 1 - 2*y/h`) — dieselbe konvention wie die
        körper-position. Mit glow/atmosphäre = 0 ergibt der shader
        (core_radius_norm == 1.0) eine flache scheibe in körperfarbe.

        `"pixel"` (voreinstellung) zeichnet statt dessen das gesäte zellmuster
        aus `body_icon.py` — EIN quad, das muster löst der fragment-shader aus
        der icon-lokalen koordinate auf. Es gibt deshalb keine aneinander-
        stossenden primitive und damit keine naht, und weil die koordinate an
        der gleitkomma-position der marke hängt, kann das muster nicht über
        die marke wandern.
        """
        if (self.body_icon_style != "pixel"
                or self._body_icon_program is None
                or self._body_icon_vao is None):
            self._draw_body_glsl(x, y, float(radius), (r, g, b), (r, g, b), 0.0, 0.0)
            return

        entry = self._body_icon_entry(body)
        if not entry:
            self._draw_body_glsl(x, y, float(radius), (r, g, b), (r, g, b), 0.0, 0.0)
            return

        cells, palette = entry
        prog = self._body_icon_program

        # Was ueber alle koerper gleich bleibt, geht ueber den vergleichenden
        # cache; nur position, muster, farbe und ueberblendung je koerper.
        self._set_uniform(prog, 'u_viewport', '_icon_viewport',
                          (float(self.width), float(self.height)))
        self._set_uniform(prog, 'u_tier_alpha', '_icon_tier_alpha',
                          tuple(body_icon.TIER_ALPHA[1:]))
        self._set_uniform(prog, 'u_grid', '_icon_grid', int(cells.grid))
        self._set_uniform(prog, 'u_edge_px', '_icon_edge',
                          float(self.body_icon_edge_px))
        self._set_uniform(prog, 'u_cell_gap', '_icon_gap',
                          float(self.body_icon_cell_gap))
        self._set_uniform(prog, 'u_cell_rim', '_icon_rim',
                          float(self.body_icon_cell_rim))
        self._set_uniform(prog, 'u_cell_rim_dark', '_icon_rim_dark',
                          float(self.body_icon_cell_rim_dark))
        self._set_uniform(prog, 'u_halo_alpha', '_icon_halo',
                          float(self.body_icon_halo_alpha))
        self._set_uniform(prog, 'u_extent', '_icon_extent',
                          float(self.ICON_QUAD_EXTENT))
        self._set_uniform(prog, 'u_radius_px', '_icon_radius', float(radius))
        self._set_uniform(prog, 'u_unit', '_icon_unit', float(cells.unit))

        self._write_uniform(prog, 'u_center_px', (float(x), float(y)))
        # Ein uniform-ARRAY: moderngl schreibt es mit glUniform1uiv, also
        # dicht gepackt -- deshalb .write() statt .value.
        self._write_uniform_array(prog, 'u_cells', cells.words)
        self._write_uniform(prog, 'u_tier_dim', palette[0])
        self._write_uniform(prog, 'u_tier_base', palette[1])
        self._write_uniform(prog, 'u_tier_bright', palette[2])
        self._write_uniform(prog, 'u_fade', float(fade))
        self._write_uniform(prog, 'u_seed', int(cells.seed) & 0xFFFFFFFF)
        self._set_uniform(prog, 'u_shade', '_icon_shade',
                          float(self.body_icon_shade_jitter))

        self._body_icon_vao.render(moderngl.TRIANGLE_STRIP)

    def _update_icon_radius_range(self, bodies):
        """Die spanne der PHYSISCHEN koerper-radien im geladenen system.

        Einmal je frame aus der echten koerperliste bestimmt (28 koerper,
        eine schleife -- kostet nichts), NICHT aus einer festen konstante:
        so passt sich die skalierung automatisch an, welches system gerade
        geladen ist, statt eine zahl aus DIESEM sonnensystem im code zu
        verstecken. Das schiff zaehlt nicht mit -- sein `radius` ist ein
        rein technischer platzhalter (1.0 m), keine physische groesse.
        """
        lo = hi = None
        for body in bodies:
            if getattr(body, 'is_ship', False):
                continue
            r = float(getattr(body, 'radius', 0.0))
            if r <= 0.0:
                continue
            if lo is None or r < lo:
                lo = r
            if hi is None or r > hi:
                hi = r
        if lo is None:
            lo = hi = 1.0
        self._icon_radius_range_m = (lo, hi)

    def _body_icon_size_factor(self, body_radius_m):
        """0..1: wo dieser koerper-radius innerhalb der GELADENEN spanne liegt.

        LOG-skaliert: planeten- und mond-radien liegen ueber mehrere
        groessenordnungen (in `solar_system.json` von Mimas' 2.0e5 m bis
        Sonnes 7.0e8 m -- 3.5 dekaden). Linear interpoliert wuerde alles
        ausser der Sonne auf denselben punkt nahe 0 zusammendruecken.
        """
        lo_m, hi_m = self._icon_radius_range_m
        if hi_m <= lo_m:
            return 0.0
        r = max(lo_m, min(hi_m, float(body_radius_m)))
        return (math.log10(r) - math.log10(lo_m)) / (math.log10(hi_m) - math.log10(lo_m))

    def _body_icon_draw_radius_px(self, body_radius_m):
        """Der GEZEICHNETE radius der marke -- ein je koerper KONSTANTER wert
        aus seinem PHYSISCHEN radius, unabhaengig vom zoom.

        > **Bewusst nicht aus dem aktuellen bildschirmradius abgeleitet --
        > das war die erste, falsche fassung.** `true_radius_px` schrumpft mit
        > jedem herauszoomen gegen null, und genau dort, wo ein koerper zur
        > marke wird, liegt er fast immer weit unter `body_icon_min_radius_px`
        > -- eine mischung `min + (true - min) * einfluss` klemmte deshalb bei
        > JEDEM einfluss-wert exakt auf `min` zurueck, weil `true - min`
        > negativ blieb. Der regler hatte dadurch im spiel keine sichtbare
        > wirkung, obwohl er in einem test mit handgesetzten grossen radien
        > (bewusst weit ueber `min`) korrekt aussah. Die groesse haengt jetzt
        > an `body.radius` selbst -- der bleibt bei jedem zoom derselbe, ein
        > Jupiter-aehnlicher koerper ist also IMMER sichtbar groesser als ein
        > kleiner mond, nicht nur kurz waehrend der ueberblendung.

        `body_icon_size_influence` (0..1) mischt zwischen "immer
        `body_icon_min_radius_px`" (0 -- jede marke gleich gross) und "voll
        nach dem log-skalierten koerper-radius, bis `body_icon_max_radius_px`"
        (1).
        """
        lo = float(self.body_icon_min_radius_px)
        hi = max(lo, float(self.body_icon_max_radius_px))
        influence = max(0.0, min(1.0, float(self.body_icon_size_influence)))
        if influence <= 0.0:
            return lo
        factor = self._body_icon_size_factor(body_radius_m)
        return lo + (hi - lo) * factor * influence

    def _body_icon_fade(self, true_radius_px):
        """Deckkraft der marke bei diesem echten bildschirmradius.

        1.0 unterhalb der schwelle, dann linear auf 0 bis
        `body_icon_min_radius_px * body_icon_fade_factor`. Der echte koerper
        wird in diesem band ganz normal gezeichnet und die marke darueber
        ausgeblendet -- das ist die ueberblendung, und sie kostet den
        koerper-zeichenweg keine zeile.
        """
        lo = float(self.body_icon_min_radius_px)
        hi = lo * float(self.body_icon_fade_factor)
        if true_radius_px < lo:
            return 1.0
        if hi <= lo or true_radius_px >= hi:
            return 0.0
        return 1.0 - (float(true_radius_px) - lo) / (hi - lo)

    # ------------------------------------------------------------------
    # Prozedurale vektor-optik der koerper (D2)
    # ------------------------------------------------------------------

    def _body_style_key(self, body, detail):
        """Cache-schluessel: alles, was die zeichnung bestimmt.

        Bewusst NICHT `id(body)`: der schluessel soll einen neu geladenen
        koerper mit denselben angaben auf dieselbe zeichnung fuehren.
        """
        seed = getattr(body, 'style_seed', None)
        if seed is None:
            seed = body_style.seed_from_name(getattr(body, 'name', '?'))
        mode = getattr(body, 'style_mode', None) or body_style.DEFAULT_MODE
        shape = getattr(body, 'style_shape', None) or body_style.DEFAULT_SHAPE
        color = tuple(int(c) for c in tuple(getattr(body, 'color', (255, 255, 255)))[:3])
        return (int(seed) & 0xFFFFFFFF, str(mode), str(shape), color,
                str(detail), float(self.body_vector_shape_density))

    def _body_detail_levels(self, radius_px):
        """[(stufe, gewicht), ...] fuer diesen bildschirmradius.

        Die stufe, deren facetten am naechsten an `body_vector_facet_px`
        liegen, gewinnt; in einem band um den wechsel herum laufen ZWEI
        stufen mit summe 1 -- das ist die ueberblendung. Gerechnet wird in
        log-groesse, weil die stufen sich in der facettenbreite jeweils
        halbieren, also geometrisch und nicht linear liegen.
        """
        forced = self.body_vector_detail
        levels = body_style.DETAIL_LEVELS
        if forced:
            return ((str(forced), 1.0),)

        radius_px = max(1e-3, float(radius_px))
        target = max(1.0, float(self.body_vector_facet_px))
        blend = max(1e-3, min(0.9, float(self.body_vector_detail_blend)))

        position = 0.0
        for index in range(len(levels) - 1):
            # Bildschirmradius, bei dem stufe index+1 dieselbe facettenbreite
            # traefe wie das ziel.
            switch = target / body_style.FACET_FRACTION[levels[index + 1]]
            low = math.log(switch / (1.0 + blend))
            high = math.log(switch * (1.0 + blend))
            position += max(0.0, min(1.0,
                                     (math.log(radius_px) - low) / (high - low)))

        base = int(math.floor(position))
        frac = position - base
        if base >= len(levels) - 1:
            return ((levels[-1], 1.0),)
        if frac <= 1e-3:
            return ((levels[base], 1.0),)
        return ((levels[base], 1.0 - frac), (levels[base + 1], frac))

    def _body_style_entry(self, body, detail):
        """Gebaute + hochgeladene zeichnung eines koerpers, oder None.

        None heisst 'diesen frame noch nicht' -- entweder ist das budget
        aufgebraucht (dann kommt sie im naechsten frame) oder der bau ist
        fehlgeschlagen (dann nie wieder, der fehler steht in debug_info).
        """
        if not self.body_vector_style or self._body_surface_program is None:
            return None
        key = self._body_style_key(body, detail)
        entry = self._body_style_gpu.get(key)
        if entry is not None:
            return entry or None
        job = self._body_style_jobs.get(key)
        if job is not None:
            if not job.done():
                return None
            del self._body_style_jobs[key]
            return self._finish_body_style(key, body, job.result)

        if len(self._body_style_jobs) >= int(self._body_style_build_budget):
            return None
        executor = self._body_style_executor
        if executor is None:
            try:
                executor = ThreadPoolExecutor(max_workers=1,
                                              thread_name_prefix='bodystyle')
                self._body_style_executor = executor
            except Exception:
                executor = None
        args = (key[0],)
        kwargs = dict(color=key[3], mode=key[1], shape=key[2],
                      coverage=float(self.body_vector_coverage),
                      detail=key[4], shape_density=key[5])
        if executor is None:
            # Ohne threads lieber einen ruckler als gar keine zeichnung.
            return self._finish_body_style(
                key, body, lambda: body_style.build_planet_style(*args, **kwargs))
        self._body_style_jobs[key] = executor.submit(
            body_style.build_planet_style, *args, **kwargs)
        return None

    def _finish_body_style(self, key, body, produce):
        """Ergebnis eines baus in GL-puffer legen. Laeuft im hauptthread."""
        try:
            entry = self._upload_body_style(produce())
        except Exception as exc:
            self.debug_info['body_style_error'] = f"{getattr(body, 'name', '?')}: {exc}"
            print(f"Body style build failed ({getattr(body, 'name', '?')}): {exc}")
            entry = False
        self._body_style_gpu[key] = entry
        return entry or None

    def _upload_body_style(self, style):
        """PlanetStyle -> GL-puffer.

        Die linien werden GETRENNT expandiert: `expand_segments` wirft
        entartete segmente weg, und danach waere die grenze zwischen den
        segmenten unter und ueber den fuellungen nicht mehr bekannt. Diese
        reihenfolge ist nicht kosmetisch -- alphas addieren sich, das
        gitternetz gehoert unter die fuellungen.
        """
        tri = np.ascontiguousarray(style.tri, dtype='f4')
        under = body_style.expand_segments(style.seg[:style.under_segments])
        over = body_style.expand_segments(style.seg[style.under_segments:])
        lines = np.ascontiguousarray(np.concatenate([under, over], axis=0), dtype='f4')

        tri_vbo = self.ctx.buffer(tri.tobytes()) if tri.shape[0] else None
        line_vbo = self.ctx.buffer(lines.tobytes()) if lines.shape[0] else None

        tri_vao = None
        if tri_vbo is not None:
            tri_vao = self.ctx.vertex_array(
                self._body_surface_program,
                [(tri_vbo, '2f 3f 3f 1f 1f',
                  'a_pos', 'a_nrm', 'a_col', 'a_alpha', 'a_dark')],
            )
        line_vao = None
        if line_vbo is not None:
            line_vao = self.ctx.vertex_array(
                self._body_line_program,
                [(line_vbo, '2f 3f 3f 1f 1f 2f 1f 1f 1f',
                  'a_pos', 'a_nrm', 'a_col', 'a_alpha', 'a_dark',
                  'a_dir', 'a_side', 'a_ext', 'a_half')],
            )
        return {
            'tri_vao': tri_vao,
            'tri_count': int(tri.shape[0]),
            'line_vao': line_vao,
            'under_count': int(under.shape[0]),
            'over_count': int(over.shape[0]),
            'buffers': [b for b in (tri_vbo, line_vbo) if b is not None],
            'style': style,
        }

    def _body_detail_fade(self, radius_px):
        """0 unter der schwelle, 1 ab voller groesse, dazwischen linear."""
        lo = float(self.body_vector_min_radius_px)
        hi = max(lo + 1e-6, float(self.body_vector_full_radius_px))
        return max(0.0, min(1.0, (float(radius_px) - lo) / (hi - lo)))

    def _body_light_dir(self, body, x, y):
        """Richtung zur lichtquelle im scheiben-raum, plus emissiv-flag.

        Die richtung wird im BILDSCHIRM gemessen, nicht in weltkoordinaten:
        so folgt die beleuchtung automatisch jedem rotierenden plotting-frame.
        z bleibt 0, das licht liegt also in der bahnebene -- genau das ergibt
        von oben auf das system gesehen die richtige phase.
        """
        source = self._light_screen_xy
        if (not self.body_light_enabled or source is None
                or body is self._light_source_body):
            return (0.0, 0.0, 1.0), 1.0
        dx = float(source[0]) - float(x)
        dy = float(source[1]) - float(y)
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return (0.0, 0.0, 1.0), 1.0
        tilt = max(0.0, min(1.0, float(self.body_light_tilt)))
        plane = math.sqrt(max(0.0, 1.0 - tilt * tilt)) / length
        # bildschirm zaehlt y nach unten, die scheibe nach oben
        return (dx * plane, -dy * plane, tilt), 0.0

    def _draw_body_vector(self, entry, x, y, radius_px, light, emissive, fade):
        """Zeichnet die vektor-zeichnung eines koerpers: drei draw-calls.

        Gitternetz -> fuellungen -> konturen/figuren/ringe, in genau dieser
        reihenfolge (siehe `_upload_body_style`).
        """
        prog_surface = self._body_surface_program
        prog_line = self._body_line_program
        if prog_surface is None or prog_line is None:
            return False
        try:
            for prog in (prog_surface, prog_line):
                prog['u_center_px'].value = (float(x), float(y))
                prog['u_radius_px'].value = float(radius_px)
                prog['u_viewport'].value = (float(self.width), float(self.height))
                prog['u_light'].value = (float(light[0]), float(light[1]), float(light[2]))
                prog['u_light_exp'].value = float(self.body_light_exponent)
                prog['u_fade'].value = float(fade)
                prog['u_emissive'].value = float(emissive)

            line_vao = entry.get('line_vao')
            under = int(entry.get('under_count', 0))
            over = int(entry.get('over_count', 0))
            if line_vao is not None and under > 0:
                line_vao.render(moderngl.TRIANGLES, vertices=under, first=0)
            tri_vao = entry.get('tri_vao')
            if tri_vao is not None and entry.get('tri_count', 0) > 0:
                tri_vao.render(moderngl.TRIANGLES)
            if line_vao is not None and over > 0:
                line_vao.render(moderngl.TRIANGLES, vertices=over, first=under)
            return True
        except Exception as exc:
            self.debug_info['body_style_error'] = f"draw: {exc}"
            return False

    def _find_light_source(self, bodies):
        """Der koerper, der das system beleuchtet.

        `light_intensity > 0` gewinnt; sonst der massereichste koerper. Der
        fallback ist absicht: ein selbst gebautes system ohne das feld soll
        trotzdem beleuchtet aussehen, und der schwerste koerper ist dort
        praktisch immer der stern.
        """
        best = None
        best_mass = -1.0
        for candidate in bodies:
            if getattr(candidate, 'is_ship', False):
                continue
            if float(getattr(candidate, 'light_intensity', 0.0)) > 0.0:
                return candidate
            mass = float(getattr(candidate, 'mass', 0.0))
            if mass > best_mass:
                best = candidate
                best_mass = mass
        return best

    def _draw_body(self, body, camera):
        camera_frame_xy = self._frame_camera_xy(camera)
        x, y = self._world_to_screen_xy(
            float(body.position.x),
            float(body.position.y),
            camera,
            camera_frame_xy=camera_frame_xy,
        )
        screen_pos = (x, y)
        r, g, b = body.color[0] / 255.0, body.color[1] / 255.0, body.color[2] / 255.0
        x, y = float(screen_pos[0]), float(screen_pos[1])

        if body.is_ship:
            # Schiff: feste bildschirmgröße (pfeil), nie gecullt, nie als icon.
            self.debug_info['bodies_rendered'] += 1
            theta_frame = float(getattr(body, 'theta', 0.0))
            try:
                theta_frame = self._active_frame().transform_heading(self._frame_time_s, theta_frame)
            except Exception:
                pass
            self._draw_ship_sprite(body, x, y, r, g, b, theta_override=theta_frame)
            # Das Schiff traegt KEINEN schwebenden text mehr -- name und
            # geschwindigkeit standen frueher fest ueber/unter der silhouette.
            # Beide leben im spieler-HUD (navball-cluster); der name erscheint
            # ueber das auswahl-label, wenn das schiff angeklickt wird.
            return

        # --- Nicht-Schiff-Körper: off-screen-cull + größen-schwelle (icon-swap) ---
        # Echter, UNgeklemmter bildschirmradius. Statt den körper (alt) auf
        # min. 3px zu klemmen und dauerhaft als winzige scheibe zu zeichnen,
        # lassen wir ihn unter die schwelle schrumpfen und tauschen ihn dann
        # nahtlos gegen ein positions-icon konstanter größe.
        icon_min_radius_px = float(self.body_icon_min_radius_px)
        true_radius_px = float(body.radius) * float(camera.scale)
        as_icon = true_radius_px < icon_min_radius_px

        # Off-screen-cull (NUR rendering, physik unberührt): die marge deckt für
        # sichtbare körper den glow (~2.5x radius) ab, damit randständige große
        # körper nicht fälschlich verschwinden. Vollständig off-screen-körper
        # werden gar nicht erst gezeichnet (kein shader-/icon-aufruf).
        cull_margin_px = (icon_min_radius_px if as_icon else true_radius_px * 2.5) + 8.0
        if not self._is_on_screen(x, y, cull_margin_px):
            self.debug_info['bodies_culled'] = self.debug_info.get('bodies_culled', 0) + 1
            return

        self.debug_info['bodies_rendered'] += 1

        if as_icon:
            # Körper komplett de-rendern; nur die positions-marke zeichnen.
            # Die groesse haengt am PHYSISCHEN radius, nicht am (hier winzigen
            # bis nahe-null) bildschirmradius -- siehe `_body_icon_draw_radius_px`.
            self.debug_info['bodies_as_icon'] = self.debug_info.get('bodies_as_icon', 0) + 1
            icon_draw_radius_px = self._body_icon_draw_radius_px(float(body.radius))
            self._draw_body_icon(body, x, y, icon_draw_radius_px, r, g, b, 1.0)
            # Der name haengt NICHT an der zeichengroesse: ein angewaehlter
            # mond soll auch als marke lesbar beschriftet sein. Der zoom-modus
            # dagegen misst den ECHTEN radius, nicht die marke.
            self._queue_body_label(body, x, y, icon_draw_radius_px, screen_pos,
                                   size_radius_px=true_radius_px)
            return

        # --- Voller körper (disc + glow + atmosphäre) bei echter größe ---
        # Gleitkomma-Radius für Label-Anker beibehalten, um 1-Pixel-Flackern beim
        # Zoomen zu vermeiden. radius_px >= icon_radius_px ist hier garantiert.
        radius_px = true_radius_px
        radius = max(3, int(round(radius_px)))  # integer radius for geometry

        if hasattr(body, 'atmosphere_color'):
            r1, g1, b1 = body.atmosphere_color[0] / 255.0, body.atmosphere_color[1] / 255.0, body.atmosphere_color[2] / 255.0
        else:
            r1, g1, b1 = r, g, b

        has_atmos = bool(getattr(body, 'has_atmosphere', False))
        atmos_density = float(getattr(body, 'atmos_density', 0.0)) if has_atmos else 0.0
        light_intensity = float(getattr(body, 'light_intensity', 0.0))

        # Lichtrichtung und detailgrad bestimmen, BEVOR die scheibe gezeichnet
        # wird: `fade` verdunkelt die scheibe genau so weit, wie die vektor-
        # zeichnung darueber sie ersetzt.
        light, emissive = self._body_light_dir(body, x, y)
        fade = self._body_detail_fade(radius_px)
        style_layers = []
        if fade > 0.0:
            for detail, weight in self._body_detail_levels(radius_px):
                entry = self._body_style_entry(body, detail)
                if entry is not None:
                    style_layers.append((entry, weight))
        if not style_layers:
            # Noch nicht gebaut (budget) oder abgeschaltet: die alte flache
            # scheibe bleibt stehen, statt einen leeren dunklen kreis zu zeigen.
            fade = 0.0
        else:
            # Waehrend einer ueberblendung fehlt die zweite stufe vielleicht
            # noch. Dann traegt die vorhandene das volle bild, statt dass die
            # zeichnung fuer einen frame halb durchsichtig wird.
            total = sum(weight for _entry, weight in style_layers)
            if total > 1e-6:
                style_layers = [(entry, weight / total)
                                for entry, weight in style_layers]

        # GLSL-Shader zeichnet Scheibe + Glow + Atmosphäre in einem Quad.
        # (Kein immediate-mode-fallback mehr: ohne body-shader wird der körper
        # nicht gezeichnet, der fehler steht in debug_info['shader_error'].)
        self._draw_body_glsl(
            x,
            y,
            radius_px,
            (r, g, b),
            (r1, g1, b1),
            atmos_density,
            light_intensity,
            light=light,
            emissive=emissive,
            surface_mix=fade,
            glow=float(self.body_glow_alpha) * fade,
        )

        drawn = False
        for entry, weight in style_layers:
            if self._draw_body_vector(entry, x, y, radius_px,
                                      light, emissive, fade * weight):
                drawn = True
        if drawn:
            self.debug_info['bodies_vector'] = (
                self.debug_info.get('bodies_vector', 0) + 1)

        # --- Ueberblendung marke -> koerper -------------------------------
        # Knapp ueber der schwelle ist der koerper zwar schon "echt", sieht
        # aber noch nicht danach aus: eine 8-px-scheibe mit limbus ist etwas
        # anderes als ein zellmuster, und ein harter tausch bei exakt gleichem
        # radius poppt trotzdem. Der koerper ist oben also ganz normal
        # gezeichnet, und die marke wird DARUEBER ausgeblendet -- eine echte
        # ueberblendung, ohne dass der koerper-zeichenweg davon etwas wissen
        # muss.
        icon_fade = self._body_icon_fade(true_radius_px)
        if icon_fade > 0.0:
            icon_draw_radius_px = self._body_icon_draw_radius_px(float(body.radius))
            self._draw_body_icon(body, x, y, icon_draw_radius_px, r, g, b, icon_fade)

        self._queue_body_label(body, x, y, radius_px, screen_pos)

    def _wants_body_label(self, body, radius_px):
        """Ob der name dieses koerpers gerade angeschrieben wird.

        `body_label_mode` entscheidet, WAS die beschriftung ausloest --
        `"selected"` die auswahl, `"zoom"` der bildschirmradius (das alte
        verhalten), `"both"` beides. Der auswahl-fall haengt bewusst NICHT
        an der groesse: sonst haette gerade der weit entfernte koerper, den
        man anklickt, um ihn zu finden, keinen namen.
        """
        mode = str(getattr(self, 'body_label_mode', 'selected')).strip().lower()
        selected = (body is not None and body is self.selected_body)
        try:
            big = float(radius_px) > float(self.body_label_min_radius_px)
        except (TypeError, ValueError):
            big = False
        if mode == 'zoom':
            return big
        if mode == 'both':
            return selected or big
        return selected

    def _body_label_style(self, name):
        """(text, font, kantenglaettung, laufweite) fuer einen koerpernamen.

        VERSAL UND IN DER HAUSSCHRIFT. Der name des ausgewaehlten koerpers
        ist die einzige beschriftung mitten im bild; in der system-groteske
        gesetzt las er sich als etwas, das nicht zu dieser oberflaeche
        gehoert. Jetzt traegt er dieselbe form wie jede display-beschriftung
        des HUDs -- versal, gesperrt, hart gerastert (siehe
        _build_body_label_font und .claude/rules/ui-hud.md).

        Faellt die schriftdatei aus, bleibt es bei der systemschrift -- und
        dann auch bei ihrer kantenglaettung und ohne sperrung, denn beides
        gehoert zur pixelschrift, nicht zum namen.
        """
        font = self.font_body_label or self.font_small
        if font is None or font is self.font_small:
            return (str(name), self.font_small, True, 0.0)
        text = str(name).upper() if self.body_label_uppercase else str(name)
        tracking = float(self.body_label_tracking_em) * float(font.get_height())
        return (text, font, False, tracking)

    def _queue_body_label(self, body, lx, ly, radius_px, screen_pos=None,
                          size_radius_px=None):
        """Den namen eines koerpers fuer die zeichnung NACH dem FXAA vormerken.

        NICHT sofort zeichnen: koerper laufen in den FXAA-FBO, und FXAA ist
        ein kantenfilter -- ueber gerastertem text macht er aus 34.7 % voll
        deckenden pixeln 5.3 % und verschmiert die glyphen ueber 55 % mehr
        pixel. Die beschriftung wird deshalb gesammelt und in render() NACH
        dem FXAA-resolve gezeichnet, so wie schiff und apsis-marker es schon
        immer wurden.

        `lx, ly` ist die FRAME-AWARE bildschirmposition aus
        `_world_to_screen_xy`, nicht `camera.world_to_screen`: in rotierenden
        plot-frames weichen beide voneinander ab und das label loest sich vom
        koerper. `radius_px` ist der bezugsradius, an dem der text haengt --
        der echte bildschirmradius beim vollen koerper, die marken-groesse bei
        der marke.

        `size_radius_px` ist davon getrennt: es ist die GROESSE, nach der
        `body_label_mode = "zoom"` entscheidet, und das ist immer der echte
        bildschirmradius des koerpers. Beides zu vermengen war lange folgenlos,
        weil die marke mit 4 px unter `body_label_min_radius_px` (5) lag --
        mit 8 px lag sie darueber, und ploetzlich trug im zoom-modus jeder
        winzige mond seinen namen. Der anker haengt an der ZEICHNUNG, die
        entscheidung am KOERPER.
        """
        if size_radius_px is None:
            size_radius_px = radius_px
        if not self._wants_body_label(body, size_radius_px):
            return
        text, font, antialias, tracking = self._body_label_style(body.name)
        try:
            # Bei ausgewaehltem koerper steht ueber ihm der obere
            # auswahl-pfeil -- `lift` hebt den text darueber hinweg.
            lift = self.selection_label_lift_px(body)
            entry = self._get_label_texture(text, font, antialias=antialias,
                                            tracking=tracking)
            if entry:
                _, w, h = entry
                label_x = float(lx) - (float(w) / 2.0)
                # ueber den koerper setzen: top-down ist "oben" kleineres y
                label_y = float(ly) - float(radius_px) - lift - 6.0 - float(h)
                self._deferred_labels.append(
                    (text, label_x, label_y, font, antialias, tracking))
            else:
                self._deferred_labels.append(
                    (text,
                     float(lx) + float(radius_px) + lift + 2.0,
                     float(ly) - 8.0, font, antialias, tracking))
        except Exception:
            try:
                self._draw_body_label(
                    body.name,
                    screen_pos if screen_pos is not None else (lx, ly),
                    radius_px)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Auswahl: anklicken und markieren
    # ------------------------------------------------------------------

    def _pick_radius_px(self, body, camera):
        """Greifradius eines koerpers in bildschirm-pixeln.

        Deckungsgleich mit dem, was `_draw_body` zeichnet: der echte radius,
        nach unten auf die icon-groesse geklemmt (darunter IST der koerper das
        icon). Das schiff ist ein pfeil fester bildschirmgroesse und bekommt
        deshalb einen festen wert.
        """
        if getattr(body, 'is_ship', False):
            return 12.0
        true_radius_px = float(getattr(body, 'radius', 0.0)) * float(camera.scale)
        # Dieselbe funktion wie beim zeichnen: das klickziel deckt sich mit
        # dem, was tatsaechlich zu sehen ist, auch wenn body_icon_size_influence
        # die marke groesser als body_icon_min_radius_px zeichnet.
        icon_radius_px = self._body_icon_draw_radius_px(
            float(getattr(body, 'radius', 0.0)))
        return max(true_radius_px, icon_radius_px)

    def pick_body(self, screen_pos, bodies, camera):
        """Index des koerpers unter `screen_pos` (top-down pixel), sonst None.

        Rechnet ueber DENSELBEN pfad wie das zeichnen (`_world_to_screen_xy`
        mit `_frame_camera_xy`), damit die trefferflaeche im rotierenden wie
        im nicht-rotierenden rahmen genau dort liegt, wo der koerper zu sehen
        ist. Eine eigene, "einfachere" rechnung ueber `camera.world_to_screen`
        waere in jedem bewegten plot-frame daneben.

        Laeuft NUR beim klick, nicht je frame: 28 transformationen.
        """
        if not bodies:
            return None
        try:
            cx = float(screen_pos[0])
            cy = float(screen_pos[1])
        except Exception:
            return None

        camera_frame_xy = self._frame_camera_xy(camera)
        margin = self.ui_px(self.selection_pick_margin_px)

        best_index = None
        best_distance = 0.0
        best_radius = 0.0
        for index, body in enumerate(bodies):
            try:
                sx, sy = self._world_to_screen_xy(
                    float(body.position.x), float(body.position.y),
                    camera, camera_frame_xy=camera_frame_xy,
                )
            except Exception:
                continue
            if not (math.isfinite(sx) and math.isfinite(sy)):
                continue
            radius = self._pick_radius_px(body, camera)
            grab = radius + margin
            dx = sx - cx
            dy = sy - cy
            distance = math.hypot(dx, dy)
            if distance > grab:
                continue
            # Naechster MITTELPUNKT gewinnt, nicht der groesste treffer: sonst
            # verschluckt eine bildfuellende Sonne jeden mond, der als icon
            # davor steht. Bei gleichem abstand der kleinere koerper -- das
            # ist der spezifischere treffer.
            if (best_index is None
                    or distance < best_distance
                    or (distance == best_distance and radius < best_radius)):
                best_index = index
                best_distance = distance
                best_radius = radius
        return best_index

    def selection_label_lift_px(self, body):
        """Wieviel die beschriftung eines koerpers hoeher sitzen muss.

        Der obere pfeil steht genau dort, wo `_draw_body` sonst das label
        anheftet -- sichtbar als text mit einem dreieck darin. Bewusst OHNE
        den puls gerechnet, mit fester zugabe: eine mitatmende beschriftung
        waere unruhiger als die ueberdeckung, die sie behebt.
        """
        if body is not self.selected_body or not self.selection_marker_enabled:
            return 0.0
        span = self.ui_px(float(self.selection_gap_px)
                          + float(self.selection_arrow_length_px))
        return span * (1.0 + 2.0 * float(self.selection_pulse_amount)) + 4.0

    def _advance_selection_phases(self, real_dt):
        """Dreh- und pulsphase der markierung fortschreiben.

        Ueber das ECHTE frame-delta, nicht um einen festen betrag je frame:
        sonst haengt die drehzahl an der bildrate.
        """
        dt = max(0.0, float(real_dt))
        two_pi = 2.0 * math.pi
        spin = math.radians(float(self.selection_spin_deg_per_s)) * dt
        self._selection_spin_phase = (self._selection_spin_phase + spin) % two_pi
        period = max(float(self.selection_pulse_period_s), 1e-3)
        self._selection_pulse_phase = (
            (self._selection_pulse_phase + two_pi * dt / period) % two_pi
        )

    def _selection_marker_vertices(self, cx, cy, body_radius_px):
        """Die 12 ortho-eckpunkte der vier pfeile (4 dreiecke).

        Gibt None zurueck, wenn nichts zu zeichnen ist. Reine rechnung, damit
        der test sie ohne GL-kontext pruefen kann.
        """
        pulse = 1.0 + (float(self.selection_pulse_amount)
                       * math.sin(self._selection_pulse_phase))
        length = self.ui_px(self.selection_arrow_length_px) * pulse
        half_width = 0.5 * self.ui_px(self.selection_arrow_width_px) * pulse
        gap = self.ui_px(self.selection_gap_px)
        # Das atmen sitzt im ABSTAND, nicht in der pfeilgroesse allein -- bei
        # einem bildfuellenden koerper waeren 7 % von 13 px sonst unsichtbar.
        breathe = length * float(self.selection_pulse_amount) * 2.0 * math.sin(
            self._selection_pulse_phase)

        radius = min(max(float(body_radius_px),
                         self.ui_px(self.selection_min_radius_px)),
                     self.ui_px(self.selection_max_radius_px))
        ring = radius + gap + breathe
        if not math.isfinite(ring) or ring <= 0.0:
            return None

        verts = []
        base = self._selection_spin_phase
        for k in range(4):
            angle = base + k * (math.pi * 0.5)
            dx = math.cos(angle)
            dy = math.sin(angle)
            # Spitze zeigt nach INNEN, auf den koerper.
            tip_x = cx + dx * ring
            tip_y = cy + dy * ring
            back_x = cx + dx * (ring + length)
            back_y = cy + dy * (ring + length)
            # Normale zur pfeilachse, fuer die basisbreite.
            nx = -dy * half_width
            ny = dx * half_width
            verts.append((tip_x, self._ortho_y(tip_y)))
            verts.append((back_x + nx, self._ortho_y(back_y + ny)))
            verts.append((back_x - nx, self._ortho_y(back_y - ny)))
        return verts

    def _draw_selection_marker(self, camera):
        """Vier pfeile um den ausgewaehlten koerper. EIN zeichenaufruf.

        Wird nach dem FXAA-resolve gezeichnet (wie die koerper-beschriftungen):
        ein kantenfilter ueber vier duenne dreiecke verwaescht genau die
        spitzen, die auf den koerper zeigen sollen.
        """
        body = self.selected_body
        if body is None or not self.selection_marker_enabled:
            return
        if self._ortho_vao is None:
            return
        try:
            sx, sy = self._world_to_screen_xy(
                float(body.position.x), float(body.position.y), camera,
                camera_frame_xy=self._frame_camera_xy(camera),
            )
        except Exception:
            return
        if not (math.isfinite(sx) and math.isfinite(sy)):
            return
        # Ausserhalb des bildes gibt es nichts zu markieren. Die marge deckt
        # die pfeile ab, die noch hereinragen koennen.
        reach = self.ui_px(self.selection_arrow_length_px
                           + self.selection_gap_px
                           + self.selection_max_radius_px)
        if not self._is_on_screen(sx, sy, reach):
            return

        verts = self._selection_marker_vertices(
            sx, sy, self._pick_radius_px(body, camera))
        if not verts:
            return
        self._draw_ortho_shape(verts, self.selection_marker_color,
                               moderngl.TRIANGLES)
