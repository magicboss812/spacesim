"""Das zeichnen der hintergrund-ebene.

Die geometrie (sterne, gitter) rechnet `render/background.py` in reinem numpy;
hier steht der GL-pfad dazu.
"""
import math

import moderngl

import numpy as np

from render import background


class BackgroundDrawMixin:
    """Sternenfeld und dekaden-gitter.

    NICHTS HIER SKALIERT MIT camera.scale -- der hintergrund ist eine feste
    pixel-ebene, sonst zoomt das sternenfeld mit und der eindruck von tiefe
    geht verloren. Siehe .claude/rules/background.md."""

    def _ensure_star_buffer(self):
        """Laedt die sterntabelle in den instanz-VBO, wenn die dichte wechselt.

        Der puffer ist STATISCH: parallaxe und funkelphase stehen je stern
        darin, drift und zeit sind uniforms. Es wird also nur bei einer
        dichteaenderung geschrieben, nicht je bild.

        Gezeichnet wird als INSTANZIERTES quad, nicht als punkt-sprite --
        `gl_PointCoord` liefert auf dem NVIDIA-treiber dieses rechners
        konstant (0, 0) und liess damit die zellmaske jedes sternfragment
        verwerfen. Begruendung in shaders/star.vert.
        """
        if self._star_program is None:
            return None
        table = self.background.star_table()
        if table is None or table.shape[0] == 0:
            return None
        if not self.background.take_stars_dirty() and self._star_vao is not None:
            return self._star_vao

        data = np.ascontiguousarray(table, dtype='f4')
        if self._star_vbo is not None:
            try:
                self._star_vbo.release()
            except Exception:
                pass
        if self._star_vao is not None:
            try:
                self._star_vao.release()
            except Exception:
                pass
        try:
            if self._star_corner_vbo is None:
                # Einheitsquadrat 0..1 in TRIANGLE_STRIP-reihenfolge, von
                # allen instanzen geteilt.
                corner = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                  dtype='f4')
                self._star_corner_vbo = self.ctx.buffer(corner.tobytes())
            self._star_vbo = self.ctx.buffer(data.tobytes())
            self._star_vao = self.ctx.vertex_array(
                self._star_program,
                [
                    (self._star_corner_vbo, '2f', 'a_corner'),
                    # '/i' = je INSTANZ ein satz, nicht je vertex.
                    (self._star_vbo, '2f 4f 1f/i',
                     'a_pos', 'a_param', 'a_phase'),
                ],
            )
            self._star_vbo_count = int(data.shape[0])
        except Exception as exc:
            self.debug_info['shader_error'] = f"star buffer: {exc}"
            print(f"Star buffer fallback: {exc}")
            self._star_vao = None
            self._star_vbo = None
            self._star_vbo_count = 0
        return self._star_vao

    def _draw_background(self, camera, real_dt):
        """Zeichnet sternenfeld und gitter -- die unterste schicht.

        Laeuft VOR allem anderen in denselben framebuffer und ersetzt damit
        praktisch den clear (der bleibt trotzdem stehen, falls die ebene
        abgeschaltet oder ein programm ausgefallen ist).

        Die ebene liegt bewusst INNERHALB des FXAA-passes: sie ist die
        unterste schicht, alles andere spaeter herauszuziehen wuerde die
        reihenfolge zerreissen. Bei deckkraeften um 3 % ist der kantenfilter
        hier ohnehin nahe an einer identitaet -- anders als bei text, siehe
        .claude/rules/rendering.md.
        """
        bg = self.background
        if not bg.enabled:
            return

        # Das STERNENFELD haengt an der echten eigenbewegung des verfolgten
        # koerpers (absolut, damit ein rahmenwechsel es nicht ruckt); steht die
        # kamera frei, uebernimmt der schwenk.
        #
        # ACHTUNG: hier stand einmal `focus.velocity`. Das ist fuer
        # himmelskoerper IMMER (0, 0) -- solar_system.json setzt es so, und
        # world.update_planets schreibt nur die kepler-POSITION. Das feld stand
        # damit bei jedem koerper ausser dem Schiff still. Uebergeben wird
        # deshalb die position, abgeleitet wird in background._focus_speed.
        focus = getattr(camera, 'target', None)
        focus_world_xy = None
        focus_frame_xy = None
        if focus is not None:
            position = getattr(focus, 'position', None)
            if position is not None:
                focus_world_xy = (float(position.x), float(position.y))
                focus_frame_xy = self._frame_transform_xy(*focus_world_xy)

        # Das GITTER ist ein festes lattice im aktiven plot-frame. Sein anker
        # ist damit schlicht die kameraposition darin -- der bezugskoerper
        # steht darauf still, mond und schiff wandern darueber, ein schwenk
        # schiebt es genau so weit wie die welt. Der bezugskoerper muss hier
        # nicht gesondert hinein: er STECKT bereits in der frame-transform.
        cam_xy = self._frame_camera_xy(camera)
        grid_target = bg.grid_target_xy(cam_xy, focus_frame_xy)
        # Wogegen der anker gemessen ist. Wechselt der schluessel (R / 1 / 2,
        # oder das blickziel bei anchor="focus"), ist der sprung im ziel kein
        # flug -- die ebene uebernimmt ihn dann, statt ihn abzufahren.
        frame = self._active_frame()
        grid_key = (frame.__class__.__name__,
                    str(getattr(frame, 'label', '')),
                    bg.grid_anchor,
                    getattr(focus, 'name', None)
                    if bg.grid_anchor == "focus" else None)

        bg.update(
            real_dt,
            camera.scale,
            getattr(camera, 'target_scale', camera.scale),
            (float(camera.position.x), float(camera.position.y)),
            focus_world_xy=focus_world_xy,
            # Nur der KOERPER, nicht der rahmen: die sterne rechnen in
            # absoluten weltkoordinaten, ein rahmenwechsel aendert daran
            # nichts und darf die ableitung nicht neu ansetzen.
            focus_key=(id(focus), getattr(focus, 'name', None)),
            sim_time=self._frame_time_s,
            grid_target=grid_target,
            grid_key=grid_key,
            viewport=(self.width, self.height),
        )

        anchor_xy = bg.anchor_xy()

        viewport = (float(self.width), float(self.height))
        accent = bg.accent_rgb()
        # Virtuelle pixelgroesse in DESIGN-einheiten -- wie jede andere
        # UI-groesse, sonst zerfaellt das raster bei anderer aufloesung.
        pixel = max(1.0, self.ui_px(bg.pixel_size))
        pixel_round = min(1.0, max(0.0, float(bg.pixel_round)))

        # ------------------------------------------------- gitter/grundflaeche
        if self._background_program is not None and self._background_vao is not None:
            levels = bg.levels(camera.scale, anchor_xy[0], anchor_xy[1]) \
                if bg.grid_enabled else []
            count = min(len(levels), background.MAX_LEVELS)

            program = self._background_program
            self._set_uniform(program, 'u_viewport', '_background_viewport',
                              viewport)
            self._write_uniform(program, 'u_accent', accent)
            self._write_uniform(program, 'u_grid_opacity', float(bg.grid_opacity))
            self._write_uniform(program, 'u_pixel', pixel)
            self._write_uniform(program, 'u_pixel_round', pixel_round)
            self._write_uniform(program, 'u_level_count', int(count))
            if count:
                # Die uniform-arrays werden IMMER voll geschrieben: ein rest
                # aus dem letzten bild wuerde sonst mitgezeichnet, sobald
                # u_level_count wieder steigt.
                pad = background.MAX_LEVELS - count
                self._write_uniform(program, 'u_level_sp',
                                    [lv.spacing_px for lv in levels[:count]] + [0.0] * pad)
                self._write_uniform(program, 'u_level_alpha',
                                    [lv.alpha for lv in levels[:count]] + [0.0] * pad)
                self._write_uniform(program, 'u_level_node',
                                    [lv.node_alpha for lv in levels[:count]] + [0.0] * pad)
                # ACHTUNG: `u_level_phase` ist ein vec2-ARRAY. moderngl will
                # dafuer eine liste von PAAREN -- eine flache liste wirft
                # "Value after * must be an iterable, not float". Das ist
                # genau der fehler, der hier einmal drin war: der schreib-
                # versuch schlug still fehl, die phasen blieben null, und das
                # gitter klebte am bildschirm statt an der welt.
                phases = [(lv.phase_a, lv.phase_b) for lv in levels[:count]]
                phases.extend([(0.0, 0.0)] * pad)
                self._write_uniform(program, 'u_level_phase', phases)

            # Das quad ueberschreibt jeden pixel -- ohne blending, sonst
            # mischt es sich mit der clear-farbe.
            self.ctx.disable(moderngl.BLEND)
            self._background_vao.render(moderngl.TRIANGLE_STRIP)
            self.ctx.enable(moderngl.BLEND)

        # --------------------------------------------------------- sternenfeld
        if not bg.stars_enabled:
            return
        vao = self._ensure_star_buffer()
        if vao is None or self._star_vbo_count <= 0:
            return

        program = self._star_program
        self._set_uniform(program, 'u_viewport', '_star_viewport', viewport)
        self._write_uniform(program, 'u_pan',
                            (float(bg.star_pan_px[0]), float(bg.star_pan_px[1])))
        self._write_uniform(program, 'u_time', float(bg.time_s))
        self._write_uniform(program, 'u_opacity', float(bg.star_opacity))
        self._write_uniform(program, 'u_star_zoom', float(bg.star_zoom))
        self._write_uniform(program, 'u_zoom_amount', float(bg.zoom_amount()))
        self._write_uniform(program, 'u_pixel', pixel)
        self._write_uniform(program, 'u_pixel_round', pixel_round)

        # Ein quad je stern, alle vier ecken aus demselben puffer. Kein
        # PROGRAM_POINT_SIZE, kein gl_PointCoord -- siehe star.vert.
        vao.render(moderngl.TRIANGLE_STRIP, vertices=4,
                   instances=self._star_vbo_count)
