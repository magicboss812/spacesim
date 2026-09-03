"""Die OpenGL-pipelines des Renderers und der GL-zustandscache.

Die GLSL-quellen liegen in `render/gl/` und werden ueber `render.GL_DIR`
gefunden -- `_load_shader_source` ist die einzige stelle, die sie oeffnet.
"""
import os
import struct

import moderngl
import numpy as np


class ShaderPipelineMixin:
    """Shader uebersetzen, VAOs/VBOs anlegen, GL-zustand cachen.

    Die `_init_*_pipeline`-methoden laufen genau einmal beim aufbau; die
    uniform- und zustands-setter laufen tausende male je frame und halten
    deshalb caches, damit ein unveraenderter wert keinen GL-aufruf kostet."""

    def _init_gpu_helpers(self):
        """Erstellt wiederverwendbare puffer, programme und VAOs für kritische render-pfade."""
        self._ensure_poly_vbo()
        self._ensure_quad_vbo()
        self._init_line_pipeline()
        self._init_ortho_pipeline()
        self._init_body_pipeline()
        self._init_texquad_pipeline()
        self._init_background_pipeline()

    def _load_shader_source(self, filename):
        path = os.path.join(self._shader_dir, filename)
        with open(path, 'r', encoding='utf-8') as shader_file:
            return shader_file.read()

    def _compile_shader_program(self, vertex_filename, fragment_filename, label):
        """Lädt und linkt ein GLSL-programm; None bei fehler (pipeline degradiert)."""
        try:
            vertex_source = self._load_shader_source(vertex_filename)
            fragment_source = self._load_shader_source(fragment_filename)
            return self.ctx.program(
                vertex_shader=vertex_source,
                fragment_shader=fragment_source,
            )
        except Exception as exc:
            self.debug_info['shader_error'] = f"{label}: {exc}"
            print(f"Shader pipeline fallback ({label}): {exc}")
            return None

    def _init_line_pipeline(self):
        """Linien in top-down-bildschirmkoordinaten (y-flip in line.vert)."""
        program = self._compile_shader_program('line.vert', 'line.frag', 'line')
        if program is None:
            self._line_program = None
            self._line_vao = None
            return

        try:
            self._line_vao = self.ctx.vertex_array(
                program, [(self._poly_vbo, '2f', 'a_pos')]
            )
            self._line_program = program
        except Exception as exc:
            self.debug_info['shader_error'] = f"line: {exc}"
            print(f"Shader pipeline fallback (line): {exc}")
            try:
                program.release()
            except Exception:
                pass
            self._line_program = None
            self._line_vao = None

    def _init_ortho_pipeline(self):
        """Geometrie in der alten fixed-function-ortho-konvention (y nach oben).

        Ersetzt die früheren immediate-mode-pfade unter gluOrtho2D(0, w, 0, h)
        (schiffspfeil, debug-kreuze): exakt dieselbe pixel-abbildung, nur via
        shader (ortho.vert, OHNE den y-flip von line.vert). Der konventions-
        unterschied zwischen line- und ortho-pfad ist absichtlich und
        dokumentiert (CLAUDE.md, render-convention caveat).
        """
        program = self._compile_shader_program('ortho.vert', 'line.frag', 'ortho')
        if program is None:
            self._ortho_program = None
            self._ortho_vao = None
            return

        try:
            self._ortho_vao = self.ctx.vertex_array(
                program, [(self._poly_vbo, '2f', 'a_pos')]
            )
            self._ortho_program = program
        except Exception as exc:
            self.debug_info['shader_error'] = f"ortho: {exc}"
            print(f"Shader pipeline fallback (ortho): {exc}")
            try:
                program.release()
            except Exception:
                pass
            self._ortho_program = None
            self._ortho_vao = None

    def _init_body_pipeline(self):
        program = self._compile_shader_program('body.vert', 'body.frag', 'body')
        if program is None:
            self._body_program = None
            self._body_vao = None
            return

        try:
            self._body_vao = self.ctx.vertex_array(
                program, [(self._ensure_quad_vbo(), '2f', 'a_corner')]
            )
            self._body_program = program
        except Exception as exc:
            self.debug_info['shader_error'] = f"body: {exc}"
            print(f"Shader pipeline fallback (body): {exc}")
            try:
                program.release()
            except Exception:
                pass
            self._body_program = None
            self._body_vao = None

        self._init_body_style_pipeline()
        self._init_body_icon_pipeline()

    def _init_body_icon_pipeline(self):
        """Programm der positions-marke.

        Teilt sich das statische einheits-quad mit der body- und der
        FXAA-pipeline; die marke braucht keine eigene geometrie, weil das
        zellmuster im fragment-shader aufgeloest wird.
        """
        program = self._compile_shader_program(
            'body_icon.vert', 'body_icon.frag', 'body_icon')
        if program is None:
            self._body_icon_program = None
            self._body_icon_vao = None
            return
        try:
            self._body_icon_vao = self.ctx.vertex_array(
                program, [(self._ensure_quad_vbo(), '2f', 'a_corner')]
            )
            self._body_icon_program = program
        except Exception as exc:
            self.debug_info['shader_error'] = f"body_icon: {exc}"
            print(f"Shader pipeline fallback (body_icon): {exc}")
            try:
                program.release()
            except Exception:
                pass
            self._body_icon_program = None
            self._body_icon_vao = None

    def _init_body_style_pipeline(self):
        """Programme fuer die vektor-zeichnung der koerper.

        Anders als die uebrigen pipelines gibt es hier KEIN gemeinsames VAO:
        jeder koerper hat seine eigene geometrie und damit seinen eigenen
        puffer (siehe `_upload_body_style`).
        """
        self._body_surface_program = self._compile_shader_program(
            'body_surface.vert', 'body_surface.frag', 'body_surface')
        self._body_line_program = self._compile_shader_program(
            'body_line.vert', 'body_line.frag', 'body_line')
        if self._body_surface_program is None or self._body_line_program is None:
            self._body_surface_program = None
            self._body_line_program = None

    def _init_texquad_pipeline(self):
        """Texturierte quads (labels, HUD) in der ortho-konvention (y nach oben)."""
        program = self._compile_shader_program('texquad.vert', 'texquad.frag', 'texquad')
        if program is None:
            self._texquad_program = None
            self._texquad_vao = None
            return

        try:
            program['u_texture'].value = 0
            self._texquad_vao = self.ctx.vertex_array(
                program, [(self._ensure_quad_vbo(), '2f', 'a_corner')]
            )
            self._texquad_program = program
        except Exception as exc:
            self.debug_info['shader_error'] = f"texquad: {exc}"
            print(f"Shader pipeline fallback (texquad): {exc}")
            try:
                program.release()
            except Exception:
                pass
            self._texquad_program = None
            self._texquad_vao = None

    def _init_background_pipeline(self):
        """Hintergrund-ebene: vollbild-quad (gitter) + punkt-sprites (sterne).

        Beide programme degradieren einzeln zu None; fehlt eines, zeichnet der
        hintergrund die jeweils andere schicht weiter.
        """
        program = self._compile_shader_program(
            'background.vert', 'background.frag', 'background')
        if program is None:
            self._background_program = None
            self._background_vao = None
        else:
            try:
                self._background_vao = self.ctx.vertex_array(
                    program, [(self._ensure_quad_vbo(), '2f', 'a_pos')]
                )
                self._background_program = program
            except Exception as exc:
                self.debug_info['shader_error'] = f"background: {exc}"
                print(f"Shader pipeline fallback (background): {exc}")
                try:
                    program.release()
                except Exception:
                    pass
                self._background_program = None
                self._background_vao = None

        star = self._compile_shader_program('star.vert', 'star.frag', 'star')
        if star is None:
            self._star_program = None
            self._star_vao = None
            return
        self._star_program = star
        self._star_vao = None      # entsteht beim ersten VBO-schreiben

    def _ensure_poly_vbo(self):
        """Geteilter dynamischer vertex-puffer für polylines und ortho-geometrie."""
        if self._poly_vbo is None:
            initial_size = 4096 * 8  # bytes; wächst bei bedarf via orphan()
            self._poly_vbo = self.ctx.buffer(reserve=initial_size, dynamic=True)
            self._poly_vbo_size = initial_size
        return self._poly_vbo

    def _ensure_quad_vbo(self):
        """Statisches einheits-quad (-1..1, TRIANGLE_STRIP-reihenfolge).

        Geteilt von body-, FXAA- und texquad-pipeline.
        """
        if self._quad_vbo is None:
            quad = np.array(
                [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32
            )
            self._quad_vbo = self.ctx.buffer(quad.tobytes())
        return self._quad_vbo

    # ---- GL-zustandscache -------------------------------------------------
    #
    # Jedes `program['u_x'].value = ...` und jedes `ctx.line_width = ...` geht
    # als eigener aufruf in den treiber. Der linien-zeichenweg setzt beides
    # bei JEDEM aufruf neu -- gemessen ~300 uniform-schreibvorgaenge je frame,
    # von denen sich die allermeisten gegenueber dem vorigen aufruf gar nicht
    # geaendert haben (u_viewport ist ueber den ganzen frame konstant, u_color
    # ueber ganze gruppen von linien). Der cache haelt nur den zuletzt
    # GESCHRIEBENEN wert; geschrieben wird weiterhin jeder wechsel, die
    # sichtbare ausgabe ist also unveraendert.

    def _set_uniform(self, program, name, cache_attr, value):
        if getattr(self, cache_attr, None) == value:
            return
        try:
            program[name].value = value
        except Exception:
            return
        setattr(self, cache_attr, value)

    def _write_uniform(self, program, name, value):
        """Uniform ohne cache schreiben.

        Gegenstueck zu `_set_uniform`: fuer werte, die sich ohnehin JEDES bild
        aendern (gitterphasen, sterndrift, zeit) waere der vergleich teurer
        als der schreibvorgang.

        Ein fehlschlag wird NICHT verschluckt, sondern einmal je uniform in
        `debug_info` vermerkt und einmal gedruckt. Ein still fehlschlagender
        schreibversuch sieht sonst aus wie ein shader-fehler: der uniform
        behaelt seinen wert (in der GL: null), und man sucht die ursache im
        GLSL statt im aufrufer. Genau so ging einmal `u_level_phase` als
        flache liste statt als liste von paaren durch.
        """
        try:
            program[name].value = value
        except Exception as exc:
            key = f"uniform:{name}"
            if key not in self.debug_info:
                self.debug_info[key] = f"{type(exc).__name__}: {exc}"
                print(f"Uniform write failed ({name}): {exc}")

    def _write_uniform_array(self, program, name, values):
        """Ein uint-array-uniform in EINEM aufruf schreiben.

        Gegenstueck zu `_write_uniform` fuer arrays: `.value` nimmt bei einem
        array keine liste, `.write()` will die rohbytes.
        """
        try:
            program[name].write(struct.pack(f'{len(values)}I', *values))
        except Exception as exc:
            key = f"uniform:{name}"
            if key not in self.debug_info:
                self.debug_info[key] = f"{type(exc).__name__}: {exc}"
                print(f"Uniform array write failed ({name}): {exc}")

    def _set_line_width(self, width):
        width = float(width)
        if self._gl_line_width == width:
            return
        self.ctx.line_width = width
        self._gl_line_width = width

    def _invalidate_gl_state_cache(self):
        """Nach fenster-/kontextwechseln: alles wieder als unbekannt fuehren."""
        self._line_viewport = None
        self._line_color = None
        self._ortho_viewport = None
        self._ortho_color = None
        self._texquad_viewport = None
        self._texquad_color = None
        self._background_viewport = None
        self._star_viewport = None
        # Die marke haengt mit `u_viewport` an der fenstergroesse -- genau der
        # wert, ueber den ein cache sonst stale wuerde.
        self._icon_viewport = None
        self._icon_tier_alpha = None
        self._icon_grid = None
        self._icon_edge = None
        self._icon_gap = None
        self._icon_rim = None
        self._icon_rim_dark = None
        self._icon_shade = None
        self._icon_halo = None
        self._icon_extent = None
        self._icon_radius = None
        self._icon_unit = None
        self._gl_line_width = None
