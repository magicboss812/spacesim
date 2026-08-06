"""
OpenGL-Renderer für die Weltraumsimulation.
Verwendet pygame für Fensterverwaltung und HUD, moderngl (OpenGL) für Rendering.
"""

import pygame
from pygame.locals import *
import moderngl
import math
import os
from collections import deque
import time

import numpy as np

from reference_frames import IdentityReferenceFrame, apparent_orbital_directions


class Renderer:
    def __init__(self, width, height, enable_fxaa=True, ctx=None):

        self.width = width
        self.height = height
        self.enable_fxaa = enable_fxaa

        # moderngl-context: hängt sich an den von pygame/SDL erstellten
        # GL-context. Aufrufer (test.py) können ihren bereits erstellten
        # wrapper übergeben, damit nicht zwei moderngl-contexte denselben
        # GL-state verwalten.
        self.ctx = ctx if ctx is not None else moderngl.create_context()

        # gpu-helpers: wiederverwendbare VBOs, programme und VAOs (erstellt in
        # _init_gpu_helpers). _quad_vbo (statisches einheits-quad) wird auch
        # vom FXAA-pfad genutzt und muss deshalb schon vor _init_fxaa
        # deklariert sein (lazy erstellt via _ensure_quad_vbo).
        self._poly_vbo = None
        self._poly_vbo_size = 0
        self._quad_vbo = None
        self._line_program = None
        self._line_vao = None
        self._ortho_program = None
        self._ortho_vao = None
        self._body_program = None
        self._body_vao = None
        self._texquad_program = None
        self._texquad_vao = None

        # FXAA framebuffer, textur, shader-programm und VAO
        self.fbo = None
        self.fbo_texture = None
        self.fxaa_program = None
        self._fxaa_vao = None

        # OpenGL initialisieren
        self._init_opengl()

        # FXAA initialisieren wenn aktiviert
        if self.enable_fxaa:
            self._init_fxaa()
        
        # Pygame Fonts für HUD
        pygame.font.init()
        self.font_small = pygame.font.SysFont(None, 16)
        self.font_medium = pygame.font.SysFont(None, 20)
        
        # Debug-Info
        self.debug_info = {
            'shader_error': None,
            'bodies_rendered': 0,
            'bodies_culled': 0,
            'bodies_as_icon': 0,
            'prediction_points_in': 0,
            'prediction_points_drawn': 0,
        }
        self.render_benchmark_debug = False
        self.render_benchmark_every_n_frames = 60
        self._render_benchmark_frame = 0
        self._last_prediction_render_stats = {}
        # per-phase timings of the most recent render() call (frame_ms,
        # bodies_ms, swap_or_present_ms, ...). Read by the per-frame TIMING
        # line in test.py to split render calc vs. present cost.
        self.last_frame_timings = {}

        # optionales predictor-debug: wenn True druckt kleine beispiele der predictor-
        # punkte (bildschirm und rekonstruierte welt-koords) in die konsole.
        self.debug_predictor = False

        # principia-ähnliche visuelle sampling-kontrollen: linien-strip-rendering behalten,
        # aber punktdichte an bildschirm-krümmung/-fehler anpassen.
        self.prediction_sampling_tolerance_px = 1.5
        self.prediction_sampling_min_step_px = 0.35
        self.prediction_sampling_max_points = 1000
        # sehr feine bildschirm-toleranz beim reingezoomt erlauben.
        # kleinere werte ermöglichen mehr detail bei extremen zoom-stufen.
        self.prediction_sampling_min_tolerance_px = 0.005
        self.prediction_sampling_max_tolerance_px = 0.25
        self.prediction_sampling_max_segment_px = 4.0
        self.prediction_sampling_reference_scale = 1e-6
        self.prediction_visibility_margin_px = 128.0
        self.prediction_bypass_fxaa = True
        self.prediction_render_max_raw_scan = 3000
        self.prediction_render_max_draw_points = 1000
        self.prediction_render_max_world_length = None
        self.prediction_render_max_screen_length_px = None
        # apoapsis/periapsis-marker auf der prädiktionslinie (vom predictor
        # geliefert, hier nur gezeichnet).
        self.show_apsis_markers = True
        self.apsis_marker_radius_px = 5.0
        self._prediction_line_cache_key_value = None
        self._prediction_line_cache_points = None
        self._prediction_line_cache_stats = {}
        self._prediction_frame_transform_debug_key = None
        self._current_body_index_by_id = {}
        self.current_reference_body = None
        self.ship_velocity_vector_length_px = 70.0

        # Körper-icon-schwelle (bildschirm-pixel). Sobald der ECHTE bildschirm-
        # radius eines (nicht-schiff-)körpers unter diesen wert fällt, wird der
        # volle körper (disc + glow + atmosphäre) de-rendert und stattdessen ein
        # positions-icon konstanter bildschirmgröße gezeichnet. Dieser eine wert
        # ist zugleich swap-schwelle UND icon-radius -> der körper schrumpft exakt
        # bis zu dieser größe und das icon übernimmt nahtlos bei identischer größe
        # (kein leerer frame, keine doppelzeichnung). Beim weiteren herauszoomen
        # bleibt das icon konstant groß (skaliert nicht mehr mit der zoom-stufe).
        self.body_icon_radius_px = 4.0
        # Bildschirm-bounding-box kleiner als dieser wert (px) => referenz-spur
        # wird nicht gezeichnet (sub-pixel, ohnehin unsichtbar).
        self.reference_traj_min_screen_px = 2.0

        # frame-status (principia-ähnlich): physik bleibt absolut, rendering
        # wendet den aktuell ausgewählten plotting-frame plus optionales target-
        # overlay-frame an.
        self._plotting_frame = IdentityReferenceFrame()
        self._plotting_frame_label = "Barycentric"
        self._target_frame = None
        self._target_frame_label = None
        self._frame_time_s = 0.0
        # debugging: aktivieren um periodisch aktives frame und ausgewählte
        # körper welt/frame-koordinaten zur inspektion zu drucken.
        self.debug_frame = False
        self._frame_debug_counter = 0
        self._frame_debug_period = 30

        # reference-frame trajectorien-spuren (historie im frame-raum).
        # diese ersetzen statische scripted-orbit-ellipsen und zeigen relative
        # epizykel-bewegung für alle körper im aktiven frame.
        self.reference_trajectories_enabled = True
        self.reference_trajectories_max_points = 2400
        self.reference_trajectories_sample_step_s = 1.0
        self._reference_traj_last_sample_time = None
        self._reference_traj_points = {}
        self._shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')

        self._label_texture_cache = {}
        # obergrenze für gecachte label-texturen: ständig wechselnde texte
        # (z. B. das schiffs-speed-label, das sich fast jeden frame ändert)
        # würden sonst unbegrenzt GL-texturen anhäufen (vram-leak).
        self._label_texture_cache_max = 256
        # pro-frame-memo der kamera-position im aktiven frame: dieselbe
        # transformation wird sonst von _draw_body (pro körper!), trails und
        # prediction mehrfach pro frame berechnet.
        self._camera_frame_xy_key = None
        self._camera_frame_xy_value = (0.0, 0.0)
        # cache der pro-zeile gerenderten HUD-surfaces: ändert sich nur eine
        # zeile (z. B. kamera-position), müssen die übrigen nicht erneut durch
        # font.render laufen.
        self._hud_line_surface_cache = {}
        self._hud_texture = None
        self._hud_texture_size = (0, 0)
        # HUD-memoization: solange die formatierten textzeilen identisch sind,
        # bleibt die persistente HUD-textur gültig und muss weder neu gerastert
        # (font.render/Surface/tostring) noch hochgeladen werden.
        self._hud_cache_key = None
        # GPU-helpers initialisieren (VBOs, programme, VAOs). Kein blanket-
        # try/except mehr: ohne diese pipelines gibt es keinen fixed-function-
        # fallback, ein fehler hier soll sofort sichtbar sein. Einzelne
        # pipelines degradieren weiterhin kontrolliert (programm = None).
        self._init_gpu_helpers()
    
    def _init_opengl(self):
        """Initialisiert OpenGL-Einstellungen (moderngl-state)."""
        self.ctx.viewport = (0, 0, self.width, self.height)

        # Blending aktivieren; depth test wird nie aktiviert (2D)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # Hintergrundfarbe (dunkelblau)
        self._clear_color = (0.0, 0.0, 0.05, 1.0)

        # VSync kommt vom fenster-swap: pygame.display.set_mode(..., vsync=1)
        # bzw. SDL_VIDEO_VSYNC in test.py. Der alte wgl/glX-hack entfällt.

    def _create_fxaa_targets(self):
        """Erstellt FBO-textur und framebuffer in aktueller fenstergröße."""
        self.fbo_texture = self.ctx.texture((self.width, self.height), 4)
        self.fbo_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo_texture.repeat_x = False  # CLAMP_TO_EDGE
        self.fbo_texture.repeat_y = False
        self.fbo = self.ctx.framebuffer(color_attachments=[self.fbo_texture])

    def _release_fxaa_targets(self):
        for name in ('fbo', 'fbo_texture'):
            obj = getattr(self, name, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
            setattr(self, name, None)

    def _init_fxaa(self):
        """Initialisiert FXAA Framebuffer und Shader."""
        try:
            self._create_fxaa_targets()

            # FXAA Shader laden
            self._load_fxaa_shaders()

            print("FXAA initialized successfully")
        except Exception as e:
            print(f"FXAA initialization failed: {e}")
            self._release_fxaa_targets()
            self.enable_fxaa = False
    
    def _load_fxaa_shaders(self):
        fxaa_vertex_source = """
        #version 330
        in vec2 a_pos;
        out vec2 v_texcoord;
        void main() {
            v_texcoord = a_pos * 0.5 + 0.5;
            gl_Position = vec4(a_pos, 0.0, 1.0);
        }
        """

        fxaa_fragment_source = """
        #version 330
        uniform sampler2D u_texture;
        uniform vec2 u_resolution;
        in vec2 v_texcoord;
        out vec4 fragColor;

        float luminance(vec3 c) {
            return dot(c, vec3(0.299, 0.587, 0.114));
        }

        void main() {
            vec2 texel_size = 1.0 / u_resolution;
            vec2 uv = v_texcoord;

            vec3 center = texture(u_texture, uv).rgb;
            float center_luma = luminance(center);

            vec3 nw = texture(u_texture, uv + vec2(-1.0, -1.0) * texel_size).rgb;
            vec3 ne = texture(u_texture, uv + vec2(1.0, -1.0) * texel_size).rgb;
            vec3 sw = texture(u_texture, uv + vec2(-1.0, 1.0) * texel_size).rgb;
            vec3 se = texture(u_texture, uv + vec2(1.0, 1.0) * texel_size).rgb;

            float luma_nw = luminance(nw);
            float luma_ne = luminance(ne);
            float luma_sw = luminance(sw);
            float luma_se = luminance(se);

            float luma_min = min(center_luma, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
            float luma_max = max(center_luma, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));
            float luma_range = luma_max - luma_min;

            if (luma_range < 0.0312) {
                fragColor = vec4(center, 1.0);
                return;
            }

            float gradient_nw_se = abs(luma_nw - luma_se);
            float gradient_ne_sw = abs(luma_ne - luma_sw);
            float contrast = max(gradient_nw_se, gradient_ne_sw);

            if (contrast < 0.0625) {
                fragColor = vec4(center, 1.0);
                return;
            }

            vec2 dir;
            dir.x = -((luma_nw + luma_ne) - (luma_sw + luma_se));
            dir.y = ((luma_nw + luma_sw) - (luma_ne + luma_se));

            float dir_reduce = max((luma_nw + luma_ne + luma_sw + luma_se) * 0.25, 0.125);
            float rcp_dir_min = 1.0 / (min(abs(dir.x), abs(dir.y)) + dir_reduce);

            dir = min(vec2(8.0), max(vec2(-8.0), dir * rcp_dir_min)) * texel_size;

            vec3 result_a = 0.5 * (
                texture(u_texture, uv + dir * (1.0/3.0 - 0.5)).rgb +
                texture(u_texture, uv + dir * (2.0/3.0 - 0.5)).rgb
            );
            vec3 result_b = result_a * 0.5 + 0.25 * (
                texture(u_texture, uv + dir * -0.5).rgb +
                texture(u_texture, uv + dir * 0.5).rgb
            );

            float luma_b = luminance(result_b);

            if (luma_b < luma_min || luma_b > luma_max) {
                fragColor = vec4(result_a, 1.0);
            } else {
                fragColor = vec4(result_b, 1.0);
            }
        }
        """

        # moderngl kompiliert und linkt in einem schritt; compile-/link-fehler
        # werfen und werden vom aufrufer (_init_fxaa) behandelt.
        self.fxaa_program = self.ctx.program(
            vertex_shader=fxaa_vertex_source,
            fragment_shader=fxaa_fragment_source,
        )

        # Uniforms einmalig setzen (textur-unit 0; auflösung bei resize
        # aktualisiert) statt pro frame in _apply_fxaa.
        self.fxaa_program['u_texture'].value = 0
        self.fxaa_program['u_resolution'].value = (float(self.width), float(self.height))

        # Vollbild-quad (TRIANGLE_STRIP über das geteilte einheits-quad)
        self._fxaa_vao = self.ctx.vertex_array(
            self.fxaa_program, [(self._ensure_quad_vbo(), '2f', 'a_pos')]
        )

        print("FXAA Shader loaded successfully")

    def _apply_fxaa(self):
        """Wendet FXAA Post-Processing an.

        Erwartet, dass der ziel-framebuffer (screen) bereits gebunden ist.
        Das vollbild-quad überschreibt jeden pixel, daher ohne blending.
        """
        if not self.enable_fxaa or self.fbo_texture is None or self._fxaa_vao is None:
            return

        self.ctx.disable(moderngl.BLEND)
        self.fbo_texture.use(location=0)
        self._fxaa_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.BLEND)

    def set_plotting_frame(self, frame, label=None):
        self._plotting_frame = frame if frame is not None else IdentityReferenceFrame()
        if label is not None:
            self._plotting_frame_label = str(label)
        else:
            self._plotting_frame_label = getattr(self._plotting_frame, 'label', 'Barycentric')
        self._reset_reference_trajectories()

    def set_target_frame(self, frame, label=None):
        self._target_frame = frame
        if frame is None:
            self._target_frame_label = None
            self._reset_reference_trajectories()
            return
        if label is not None:
            self._target_frame_label = str(label)
        else:
            self._target_frame_label = getattr(frame, 'label', 'Target overlay')
        self._reset_reference_trajectories()

    def clear_target_frame(self):
        self._target_frame = None
        self._target_frame_label = None
        self._reset_reference_trajectories()

    def set_frame_time(self, time_s):
        try:
            self._frame_time_s = float(time_s)
        except Exception:
            self._frame_time_s = 0.0

        for frame in (self._plotting_frame, self._target_frame):
            if frame is None:
                continue
            try:
                frame.set_epoch_time(self._frame_time_s)
            except Exception:
                pass

    def _active_frame(self):
        return self._target_frame if self._target_frame is not None else self._plotting_frame

    def _frame_transform_xy(self, x, y):
        frame = self._active_frame()
        try:
            return frame.to_this_frame_xy(self._frame_time_s, float(x), float(y))
        except Exception:
            return float(x), float(y)

    def _frame_camera_xy(self, camera):
        # memoisiert: ergebnis hängt nur von aktivem frame, frame-zeit und
        # kamera-position ab — alles konstant innerhalb eines render-frames.
        cam_x = float(camera.position.x)
        cam_y = float(camera.position.y)
        key = (id(self._active_frame()), self._frame_time_s, cam_x, cam_y)
        if key == self._camera_frame_xy_key:
            return self._camera_frame_xy_value
        value = self._frame_transform_xy(cam_x, cam_y)
        self._camera_frame_xy_key = key
        self._camera_frame_xy_value = value
        return value

    def _world_to_screen_xy(self, world_x, world_y, camera, camera_frame_xy=None):
        if camera_frame_xy is None:
            camera_frame_xy = self._frame_camera_xy(camera)
        frame_x, frame_y = self._frame_transform_xy(world_x, world_y)
        scale = float(camera.scale)
        sx = self.width * 0.5 + (frame_x - camera_frame_xy[0]) * scale
        sy = self.height * 0.5 - (frame_y - camera_frame_xy[1]) * scale
        return sx, sy

    def _world_to_screen_xy_at_time(self, world_x, world_y, camera, time_s, camera_frame_xy=None):
        """Konvertiert einen Welt-Punkt zu einer bestimmten Sim-Zeit in Bildschirmkoordinaten.

        Diese nutzt die zeitabhängige Transformation des aktiven Frames, sodass
        Prädiktor-Punkte (die pro Sample Sim-Zeiten enthalten) korrekt in einen
        sich bewegenden/rotierenden Plot-Frame projiziert werden.
        """
        frame = self._active_frame()
        try:
            frame_x, frame_y = frame.to_this_frame_xy(float(time_s), float(world_x), float(world_y))
        except Exception:
            # Fallback: auf aktuelle Frame-Transformation zurückfallen
            frame_x, frame_y = self._frame_transform_xy(world_x, world_y)

        # Keep the camera origin in the current render frame. Prediction samples
        # are time-tagged future world points; transforming the camera at the
        # sample time would cancel a moving reference-frame origin back out.
        if camera_frame_xy is None:
            camera_frame_xy = self._frame_camera_xy(camera)

        scale = float(camera.scale)
        sx = self.width * 0.5 + (frame_x - camera_frame_xy[0]) * scale
        sy = self.height * 0.5 - (frame_y - camera_frame_xy[1]) * scale
        return sx, sy

    def _prediction_frame_transform_mode(self):
        frame = self._active_frame()
        name = frame.__class__.__name__
        label = str(getattr(frame, 'label', '') or '')
        label_l = label.lower()
        if isinstance(frame, IdentityReferenceFrame) or name == 'IdentityReferenceFrame':
            return 'world'
        if 'NonRotating' in name or 'non-rotating' in label_l:
            return 'body_centered_non_rotating'
        if 'BodyDirection' in name or 'direction' in label_l:
            return 'body_centered_body_direction'
        return 'custom_frame'

    def _debug_prediction_frame_transform(self, path_points, predictor=None):
        if not getattr(self, 'debug_predictor', False):
            return
        try:
            count = self._points_count(path_points)
            if count <= 0:
                return
            mode = self._prediction_frame_transform_mode()
            active_frame = self._active_frame()
            ref_index = getattr(predictor, 'reference_body_index', None) if predictor is not None else None
            if ref_index is None:
                primary = getattr(active_frame, 'primary_body', None)
                if primary is None:
                    primary = getattr(active_frame, 'target_body', None)
                body_index_by_id = getattr(self, '_current_body_index_by_id', {}) or {}
                ref_index = body_index_by_id.get(id(primary)) if primary is not None else None
            key = (mode, int(count), id(active_frame), ref_index)
            if key == getattr(self, '_prediction_frame_transform_debug_key', None):
                return
            self._prediction_frame_transform_debug_key = key
            if mode == 'world':
                print(f"PRED_DBG_FRAME_TRANSFORM: mode=world points={int(count)}", flush=True)
            else:
                print(
                    "PRED_DBG_FRAME_TRANSFORM: "
                    f"mode={mode} "
                    f"ref_index={ref_index if ref_index is not None else 'n/a'} "
                    f"points={int(count)}",
                    flush=True,
                )
        except Exception:
            pass

    def _reset_reference_trajectories(self):
        self._reference_traj_points = {}
        self._reference_traj_last_sample_time = None
        # frame-wechsel: gecachte kamera-frame-position ist nicht mehr gültig.
        self._camera_frame_xy_key = None

    def _init_gpu_helpers(self):
        """Erstellt wiederverwendbare puffer, programme und VAOs für kritische render-pfade."""
        self._ensure_poly_vbo()
        self._ensure_quad_vbo()
        self._init_line_pipeline()
        self._init_ortho_pipeline()
        self._init_body_pipeline()
        self._init_texquad_pipeline()

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
        self.ctx.line_width = float(width)
        self._line_program['u_viewport'].value = (float(self.width), float(self.height))
        self._line_program['u_color'].value = (
            float(color[0]), float(color[1]), float(color[2]), float(color[3])
        )
        self._line_vao.render(moderngl.LINE_STRIP, vertices=n)

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
            self.ctx.line_width = float(width)
        self._ortho_program['u_viewport'].value = (float(self.width), float(self.height))
        self._ortho_program['u_color'].value = (
            float(color[0]), float(color[1]), float(color[2]), float(color[3])
        )
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

    def _build_clipped_polyline_runs(self, screen_points, margin_px=128.0):
        """
        Converts one logical predictor polyline into multiple visible screen-space runs.
        Important: preserve original segment topology. Never connect visible points
        across an offscreen gap.
        """
        if screen_points is None or len(screen_points) < 2:
            return []

        left = -float(margin_px)
        top = -float(margin_px)
        right = float(self.width) + float(margin_px)
        bottom = float(self.height) + float(margin_px)

        runs = []
        run = []

        for i in range(len(screen_points) - 1):
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

        return runs

    def _draw_body_glsl(self, x, y, radius, base_color, atmos_color, atmos_density, light_intensity):
        """Zeichnet einen körper als shader-gesteuertes quad (scheibe + optional atmosphäre + glow)."""
        if self._body_vao is None:
            return False

        radius_px = max(1.0, float(radius))
        radius_scale = max(0.5, min(2.0, radius_px / 50.0))

        outer_radius = radius_px
        atmos_alpha = 0.0
        atmos_radius = radius_px
        if atmos_density > 0.0:
            atmos_radius = radius_px * 2.0
            outer_radius = max(outer_radius, atmos_radius)
            atmos_alpha = min(float(atmos_density) / 100.0, 1.0) * radius_scale

        glow_alpha = 0.0
        if light_intensity > 0.0:
            glow_radius = radius_px * 2.5
            outer_radius = max(outer_radius, glow_radius)
            glow_alpha = min(float(light_intensity) / 1000.0, 1.0) * 0.5 * radius_scale * 0.8

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

            self._body_vao.render(moderngl.TRIANGLE_STRIP)
            return True
        except Exception:
            return False

    def _draw_body_icon(self, x, y, radius, r, g, b):
        """Positions-icon eines körpers: flache scheibe konstanter bildschirmgröße.

        `radius` ist sowohl die swap-schwelle als auch der icon-radius
        (`body_icon_radius_px`). Wird gezeichnet, sobald der echte bildschirm-
        radius des körpers unter die schwelle fällt.

        WICHTIG: das icon läuft über denselben GLSL-körper-shader wie der
        volle körper. Der vertex-shader (body.vert) erwartet top-down-screen-
        koordinaten und spiegelt y intern (`ndc.y = 1 - 2*y/h`) — dieselbe
        konvention wie die körper-position. Mit glow/atmosphäre = 0 ergibt der
        shader (core_radius_norm == 1.0) eine flache scheibe in körperfarbe --
        positionsgenau und am übergang nahtlos zum körper.
        """
        self._draw_body_glsl(x, y, float(radius), (r, g, b), (r, g, b), 0.0, 0.0)

    def _get_label_texture(self, text, font):
        key = (text, font.get_height())
        entry = self._label_texture_cache.get(key)
        if entry:
            return entry  # (texture, w, h)
        try:
            surface = font.render(text, True, (255, 255, 255))
            texture_data = pygame.image.tostring(surface, 'RGBA', True)
            w, h = surface.get_size()
            texture = self.ctx.texture((w, h), 4, texture_data)
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            # cache deckeln (FIFO): ständig wechselnde texte (speed-label)
            # würden sonst unbegrenzt GL-texturen anhäufen. Stabile labels
            # (körpernamen) werden nach einer eviction einfach neu erzeugt.
            if len(self._label_texture_cache) >= self._label_texture_cache_max:
                evict_n = max(1, self._label_texture_cache_max // 4)
                for old_key in list(self._label_texture_cache.keys())[:evict_n]:
                    old_texture = self._label_texture_cache.pop(old_key)[0]
                    try:
                        old_texture.release()
                    except Exception:
                        pass
            self._label_texture_cache[key] = (texture, w, h)
            return (texture, w, h)
        except Exception:
            return None

    def _draw_texture_ortho(self, texture, x, y, width, height):
        """Zeichnet eine textur als quad in der ortho-konvention (y nach oben).

        Ersatz für die früheren immediate-mode glTexCoord/glVertex-quads unter
        gluOrtho2D(0, w, 0, h): (x, y) ist die untere linke ecke, texcoord
        (0, 0) liegt ebendort (texturen werden vertikal geflippt hochgeladen).
        """
        if self._texquad_vao is None or texture is None:
            return
        self._texquad_program['u_rect'].value = (
            float(x), float(y), float(width), float(height)
        )
        self._texquad_program['u_viewport'].value = (float(self.width), float(self.height))
        texture.use(location=0)
        self._texquad_vao.render(moderngl.TRIANGLE_STRIP)

    def _blit_cached_text(self, text, x, y, font):
        entry = self._get_label_texture(text, font)
        if not entry:
            # fallback: one-shot-textur ohne cache erzeugen, zeichnen, freigeben
            try:
                surface = font.render(text, True, (255, 255, 255))
                texture_data = pygame.image.tostring(surface, 'RGBA', True)
                w, h = surface.get_size()
                texture = self.ctx.texture((w, h), 4, texture_data)
                texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self._draw_texture_ortho(texture, x, y, w, h)
                texture.release()
            except Exception:
                pass
            return
        texture, w, h = entry
        self._draw_texture_ortho(texture, x, y, w, h)

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
                trail = deque(maxlen=max(64, int(self.reference_trajectories_max_points)))
                self._reference_traj_points[body_id] = trail

            try:
                fx, fy = self._frame_transform_xy(float(body.position.x), float(body.position.y))
            except Exception:
                continue

            if trail:
                lx, ly = trail[-1]
                dx = fx - lx
                dy = fy - ly
                if dx * dx + dy * dy < 1e-18:
                    continue
            trail.append((float(fx), float(fy)))

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
            if trail is None or len(trail) < 2:
                continue

            # Vektorisiert statt python-schleife: spuren haben bis zu
            # reference_trajectories_max_points punkte pro körper und frame.
            arr = np.asarray(trail, dtype=np.float64)
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

            screen_points = list(zip(sxs.tolist(), sys_.tolist()))

            if min_sx >= -margin and max_sx <= right and min_sy >= -margin and max_sy <= bottom:
                # Spur liegt vollständig im sichtfenster: Liang-Barsky wäre für
                # jedes segment ein no-op und lieferte exakt einen run.
                runs = (screen_points,)
            else:
                runs = self._visible_window_runs(screen_points, margin_px=margin)
            for run in runs:
                if len(run) < 2:
                    continue
                self._draw_polyline(run, color=(cr, cg, cb, 0.42), width=1.0)

    def _emit_render_benchmark(self, timings):
        if not self.render_benchmark_debug:
            return
        try:
            self._render_benchmark_frame += 1
            every = max(1, int(self.render_benchmark_every_n_frames))
            if self._render_benchmark_frame % every != 0:
                return
            pred = dict(getattr(self, "_last_prediction_render_stats", {}) or {})
            print(
                "RENDER_BENCH: "
                f"frame_ms={timings.get('frame_ms', 0.0):.3f} "
                f"bodies_ms={timings.get('bodies_ms', 0.0):.3f} "
                f"predictor_prepare_ms={pred.get('prepare_ms', 0.0):.3f} "
                f"predictor_draw_ms={pred.get('draw_ms', 0.0):.3f} "
                f"predictor_raw_in={pred.get('raw_in', 0)} "
                f"scanned={pred.get('scanned', 0)} "
                f"visible={pred.get('visible', 0)} "
                f"drawn={pred.get('drawn', 0)} "
                f"skipped_by_stride={pred.get('skipped_by_stride', 0)} "
                f"clipped_or_rejected={pred.get('clipped_or_rejected', 0)} "
                f"cache_hit={pred.get('cache_hit', False)} "
                f"reference_trails_ms={timings.get('reference_trails_ms', 0.0):.3f} "
                f"hud_ms={timings.get('hud_ms', 0.0):.3f} "
                f"fxaa_ms={timings.get('fxaa_ms', 0.0):.3f} "
                f"swap_or_present_ms={timings.get('swap_or_present_ms', 0.0):.3f}",
                flush=True,
            )
        except Exception:
            pass

    def render(self, bodies, camera, prediction_points=None, predictor=None, sim_time=None, reference_body=None, ship_control=None, real_dt=0.0):
        frame_t0 = time.perf_counter()
        timings = {
            'bodies_ms': 0.0,
            'reference_trails_ms': 0.0,
            'hud_ms': 0.0,
            'fxaa_ms': 0.0,
            'swap_or_present_ms': 0.0,
        }

        if sim_time is not None:
            self.set_frame_time(sim_time)
        self.current_reference_body = reference_body
        self._dbg_ship_control = ship_control
        try:
            self._current_body_index_by_id = {id(body): idx for idx, body in enumerate(bodies)}
        except Exception:
            self._current_body_index_by_id = {}

        reference_t0 = time.perf_counter()
        self._record_reference_trajectories(bodies)
        timings['reference_trails_ms'] += (time.perf_counter() - reference_t0) * 1000.0

        # Optional periodic debug output to inspect frame transforms.
        if getattr(self, 'debug_frame', False):
            self._frame_debug_counter += 1
            if self._frame_debug_counter % getattr(self, '_frame_debug_period', 30) == 0:
                try:
                    sun = next((b for b in bodies if 'sonn' in getattr(b, 'name', '').lower() or getattr(b, 'name', '').lower() in ('sun', 'sonne')), None)
                    earth = next((b for b in bodies if getattr(b, 'name', '').lower() in ('earth', 'erde')), None)
                    active = self._active_frame()
                    label = getattr(active, 'label', None)
                    if sun is not None and earth is not None:
                        swx, swy = float(sun.position.x), float(sun.position.y)
                        exx, exy = float(earth.position.x), float(earth.position.y)
                        sfx, sfy = self._frame_transform_xy(swx, swy)
                        efx, efy = self._frame_transform_xy(exx, exy)
                        print(f"FRAME_DBG: label={label} time={self._frame_time_s:.3f} sun_world=({swx:.6e},{swy:.6e}) sun_frame=({sfx:.6e},{sfy:.6e}) earth_world=({exx:.6e},{exy:.6e}) earth_frame=({efx:.6e},{efy:.6e})")
                except Exception:
                    pass

        self.debug_info['bodies_rendered'] = 0
        self.debug_info['bodies_culled'] = 0
        self.debug_info['bodies_as_icon'] = 0
        self.debug_info['prediction_points_in'] = 0
        self.debug_info['prediction_points_drawn'] = 0
        self._last_prediction_render_stats = {
            'raw_in': self._points_count(prediction_points),
            'scanned': 0,
            'visible': 0,
            'drawn': 0,
            'skipped_by_stride': 0,
            'clipped_or_rejected': 0,
            'prepare_ms': 0.0,
            'draw_ms': 0.0,
            'cache_hit': False,
        }
        
        # falls FXAA aktiviert ist, rendern nicht-schiff-körper in das FBO und
        # FXAA anwenden. Schiffe werden danach direkt in den haupt-framebuffer
        # gerendert damit predictor (ebenfalls im hauptpuffer gerendert) und
        # das schiff-marker exakt dieselben pixel-koordinaten teilen.
        ship_body = next((b for b in bodies if getattr(b, 'is_ship', False)), None)

        if self.enable_fxaa and self.fbo:
            target_fbo = self.fbo
        else:
            target_fbo = self.ctx.screen
        target_fbo.use()
        target_fbo.clear(*self._clear_color)

        reference_t0 = time.perf_counter()
        self._draw_reference_trajectories(bodies, camera)
        timings['reference_trails_ms'] += (time.perf_counter() - reference_t0) * 1000.0

        # Render all non-ship bodies first (they may be FXAA-processed).
        bodies_t0 = time.perf_counter()
        for body in bodies:
            if getattr(body, 'is_ship', False):
                continue

            self._draw_body(body, camera)
        timings['bodies_ms'] += (time.perf_counter() - bodies_t0) * 1000.0

        prediction_has_points = self._points_count(prediction_points) > 0
        prediction_drawn = False

        if prediction_has_points and self.enable_fxaa and self.fbo and not self.prediction_bypass_fxaa:
            self.draw_prediction(prediction_points, camera, predictor=predictor)
            prediction_drawn = True

        if self.enable_fxaa and self.fbo:
            # Zurück zum Standard-Framebuffer and apply FXAA post-process
            self.ctx.screen.use()
            fxaa_t0 = time.perf_counter()
            self._apply_fxaa()
            timings['fxaa_ms'] += (time.perf_counter() - fxaa_t0) * 1000.0

        # Ab hier wird direkt in den haupt-framebuffer gezeichnet (predictor,
        # schiff, HUD). Blending ist global aktiv (ctx.enable in _init_opengl,
        # von _apply_fxaa wiederhergestellt); die alten projektions-resets der
        # fixed-function-pipeline entfallen.
        if prediction_has_points and not prediction_drawn:
            self.draw_prediction(prediction_points, camera, predictor=predictor)

        # Schiff-Marker im Haupt-Framebuffer zeichnen, damit er visuell
        # genau mit dem Prädiktor-Startpunkt übereinstimmt.
        if ship_body is not None:
            bodies_t0 = time.perf_counter()
            # Orientierungs-snap ANWENDEN bevor der pfeil gezeichnet wird, mit
            # demselben frame + _frame_time_s wie die vektoren/der pfeil — so
            # ist die nase exakt an die gezeichneten prograde/normal-vektoren
            # gebunden (keine zeit-/konventionsdrift).
            self._apply_orientation_snap(
                ship_body, ship_control, reference_body, prediction_points, real_dt
            )
            self._draw_body(ship_body, camera)
            self.draw_ship_velocity_vector(ship_body, camera, reference_body=reference_body)
            self.draw_ship_thrust_vector(ship_body, camera)
            self.draw_ship_orientation_debug_vectors(
                ship_body, camera, reference_body=reference_body,
                prediction_points=prediction_points,
            )
            timings['bodies_ms'] += (time.perf_counter() - bodies_t0) * 1000.0

        hud_t0 = time.perf_counter()
        self._render_hud(camera, predictor)
        timings['hud_ms'] += (time.perf_counter() - hud_t0) * 1000.0
        swap_t0 = time.perf_counter()
        pygame.display.flip()
        timings['swap_or_present_ms'] = (time.perf_counter() - swap_t0) * 1000.0
        timings['frame_ms'] = (time.perf_counter() - frame_t0) * 1000.0
        self.last_frame_timings = timings
        self._emit_render_benchmark(timings)

    def draw_ship_velocity_vector(self, ship, camera, reference_body=None):
        if ship is None:
            return

        try:
            vx = float(ship.velocity.x)
            vy = float(ship.velocity.y)
            if reference_body is None:
                reference_body = getattr(self, "current_reference_body", None)
            if reference_body is not None:
                try:
                    ref_vx = float(reference_body.velocity.x)
                    ref_vy = float(reference_body.velocity.y)
                except Exception:
                    ref_vx = float(getattr(getattr(reference_body, "velocity", None), "x", 0.0) or 0.0)
                    ref_vy = float(getattr(getattr(reference_body, "velocity", None), "y", 0.0) or 0.0)
                vx -= ref_vx
                vy -= ref_vy

            frame = self._active_frame()
            try:
                vx, vy = frame.to_this_frame_vector_xy(self._frame_time_s, vx, vy)
            except Exception:
                pass

            mag = math.hypot(vx, vy)
            if mag <= 1e-12:
                return

            dir_x = vx / mag
            dir_y = vy / mag

            sx, sy = self._world_to_screen_xy(float(ship.position.x), float(ship.position.y), camera)
            length_px = max(20.0, min(90.0, float(getattr(self, "ship_velocity_vector_length_px", 70.0))))
            ex = sx + dir_x * length_px
            ey = sy - dir_y * length_px

            self._draw_polyline([(sx, sy), (ex, ey)], color=(0.2, 0.8, 1.0, 0.9), width=2.0)
        except Exception:
            return

    def draw_ship_thrust_vector(self, ship, camera):
        if ship is None:
            return

        try:
            direction = getattr(ship, "last_thrust_direction", None)
            if direction is None:
                return

            vx = float(direction.x)
            vy = float(direction.y)

            frame = self._active_frame()
            try:
                vx, vy = frame.to_this_frame_vector_xy(self._frame_time_s, vx, vy)
            except Exception:
                pass

            mag = math.hypot(vx, vy)
            if mag <= 1e-12:
                return

            vx /= mag
            vy /= mag

            sx, sy = self._world_to_screen_xy(float(ship.position.x), float(ship.position.y), camera)
            length_px = 45.0
            ex = sx + vx * length_px
            ey = sy - vy * length_px

            self._draw_polyline([(sx, sy), (ex, ey)], color=(1.0, 0.5, 0.1, 0.95), width=2.0)
        except Exception:
            return

    def active_plotting_frame(self):
        """Public accessor for the frame the ship control uses to hold a snap."""
        return self._active_frame()

    def orbital_frame_directions(self, ship, reference_body=None, prediction_points=None):
        """The frame-space orbital directions used to draw the overlay vectors.

        Single source of truth for both the debug vectors and the orientation
        snap: prograde/normal_in are the tangent/inward of the *drawn* predictor
        line in the active plotting frame; retrograde/antinormal are their
        opposites. Evaluated at the renderer's current ``_frame_time_s`` — the
        same instant the ship arrow is drawn.
        """
        frame = self._active_frame()
        if reference_body is None:
            reference_body = getattr(self, "current_reference_body", None)
        ref_pos = getattr(reference_body, "position", None)
        return frame, apparent_orbital_directions(
            frame, self._frame_time_s, ship.position, ship.velocity, ref_pos,
            points=prediction_points,
        )

    def _apply_orientation_snap(self, ship, ship_control, reference_body,
                                prediction_points, real_dt):
        """Tie the ship nose to the drawn orbital vector for the latched snap.

        Computes the world heading whose *drawn* arrow coincides with the
        frame-space snap vector, using ``heading_from_this_frame`` at the SAME
        ``_frame_time_s`` that ``_draw_body`` uses to draw the arrow. This makes
        the nose lock onto the on-screen prograde/normal vector exactly, with no
        dependence on sim_dt or frame rotation rate. The ship's stored ``theta``
        stays in world space (physics remains absolute); only the render-time
        transform is inverted here.
        """
        if ship is None or ship_control is None:
            return
        mode = getattr(ship_control, "snap_mode", None)
        if not mode:
            return
        try:
            frame, directions = self.orbital_frame_directions(
                ship, reference_body, prediction_points
            )
            d = directions.get(mode)
            if d is None:
                return
            # The orbital vectors render top-down (line shader, y flipped in
            # line.vert) while the ship arrow renders bottom-up (fixed-function
            # gluOrtho2D). So the arrow's on-screen heading is the vertical
            # MIRROR of the vector for the same frame-space direction: for a
            # target frame-heading a, the drawn arrow lands on the vector only
            # when a is measured with d.y negated. Without the shader both paths
            # are bottom-up and no flip is needed. This is what ties the snapped
            # nose to the *drawn* prograde/normal vector (not its mirror).
            dy_sign = -1.0 if getattr(self, "_line_program", None) else 1.0
            ang_frame = math.atan2(dy_sign * float(d.y), float(d.x))
            try:
                theta_target = frame.heading_from_this_frame(self._frame_time_s, ang_frame)
            except Exception:
                theta_target = ang_frame
            ship_control.orient_towards_angle(theta_target, real_dt)
        except Exception:
            return

    def draw_ship_orientation_debug_vectors(self, ship, camera, reference_body=None,
                                            prediction_points=None):
        """Debug overlay: always draws prograde (green) + normal-inward (magenta).

        Directions come from ``apparent_orbital_directions`` fed the actual
        predictor polyline (``prediction_points``) — i.e. the tangent of the
        drawn line as it appears in the active plotting frame — so they already
        live in frame space and only need the screen y-flip, exactly like the
        drawn trajectory. This keeps them glued to the predictor line as it
        changes shape, including in rotating/translating frames.
        """
        if ship is None:
            return

        try:
            frame, directions = self.orbital_frame_directions(
                ship, reference_body, prediction_points
            )

            sx, sy = self._world_to_screen_xy(float(ship.position.x), float(ship.position.y), camera)
            length_px = 55.0

            for key, color in (
                ("prograde", (0.2, 1.0, 0.35, 0.95)),
                ("normal_in", (0.9, 0.3, 1.0, 0.95)),
            ):
                d = directions.get(key)
                if d is None:
                    continue
                ex = sx + float(d.x) * length_px
                ey = sy - float(d.y) * length_px
                # Stash the ACTUAL drawn pixel direction of each vector so the
                # diagnostic can compare raw screen geometry (non-derived).
                if key == "normal_in":
                    self._last_normal_screen_dir = (ex - sx, ey - sy)
                elif key == "prograde":
                    self._last_prograde_screen_dir = (ex - sx, ey - sy)
                self._draw_polyline([(sx, sy), (ex, ey)], color=color, width=2.0)

            self._debug_orientation_angles(ship, camera, frame, directions, sx, sy)
        except Exception:
            return

    def _debug_orientation_angles(self, ship, camera, frame, directions, sx, sy):
        """Env-guarded (SPACESIM_DEBUG_ORIENT=1) screen-space angle report.

        Prints, in one common screen convention, the heading of: the actual
        predictor orbit line, my green prograde, the blue velocity vector, and
        the ship nose. Whichever one disagrees is the culprit for the reported
        45 deg offset. Behavior-neutral: only prints, throttled.
        """
        if os.environ.get("SPACESIM_DEBUG_ORIENT", "0").strip().lower() in ("0", "", "false", "off", "no"):
            return
        self._debug_orient_counter = getattr(self, "_debug_orient_counter", 0) + 1
        if self._debug_orient_counter % 30 != 1:
            return

        def sdeg(dx, dy):
            # Screen-space heading: vectors are drawn as (dx, -dy), so the
            # on-screen angle of a frame-space direction is atan2(-dy, dx).
            return math.degrees(math.atan2(-dy, dx))

        parts = []

        # RAW drawn pixel angles (from the actual vertices, y-down screen space).
        def rawdeg(v):
            if v is None:
                return None
            return math.degrees(math.atan2(v[1], v[0]))

        norm_v = getattr(self, "_last_normal_screen_dir", None)
        arrow_v = getattr(self, "_last_arrow_screen_dir", None)
        norm_deg = rawdeg(norm_v)
        # DISPLAYED arrow angle: the arrow renders under gluOrtho2D bottom-up
        # while vectors render via the line shader top-down, so the arrow's
        # on-screen y is the negation of its input y.
        arrow_deg = None if arrow_v is None else math.degrees(math.atan2(-arrow_v[1], arrow_v[0]))

        if norm_deg is not None:
            parts.append(f"magenta_raw={norm_deg:8.3f}")
        if arrow_deg is not None:
            parts.append(f"arrow_raw={arrow_deg:8.3f}")

        # Per-sample rotation of each, so co-rotation vs counter-rotation is
        # directly visible (this is the user's actual complaint).
        prev = getattr(self, "_dbg_prev_raw", None)
        if prev is not None and norm_deg is not None and arrow_deg is not None:
            dmag = (norm_deg - prev[0] + 180.0) % 360.0 - 180.0
            darr = (arrow_deg - prev[1] + 180.0) % 360.0 - 180.0
            sense = "SAME" if (dmag * darr) >= 0 else "OPPOSITE"
            parts.append(f"d_mag={dmag:+7.3f} d_arrow={darr:+7.3f} [{sense}]")
        if norm_deg is not None and arrow_deg is not None:
            self._dbg_prev_raw = (norm_deg, arrow_deg)
            gap = (arrow_deg - norm_deg + 180.0) % 360.0 - 180.0
            parts.append(f"gap={gap:+7.3f}")

        mode = getattr(getattr(self, "_dbg_ship_control", None), "snap_mode", None)
        frame_label = getattr(frame, "label", frame.__class__.__name__)
        print(f"ORIENT_DBG: snap={mode} " + "  ".join(parts) + f"  frame='{frame_label}'")

    def _ship_relative_speed_m_s(self, ship, reference_body=None):
        if ship is None:
            return None

        try:
            vx = float(ship.velocity.x)
            vy = float(ship.velocity.y)
        except Exception:
            return None

        if reference_body is not None:
            try:
                vx -= float(reference_body.velocity.x)
                vy -= float(reference_body.velocity.y)
            except Exception:
                pass

        return math.hypot(vx, vy)

    def _ship_frame_speed_m_s(self, ship, dt_s=1.0):
        """
        Returns the ship's apparent speed in the active plotting frame.

        This respects translated, rotating, target-overlay, and time-dependent
        frames by finite-differencing the active frame transform. It does not
        use the clamped visual velocity vector length.
        """
        if ship is None:
            return None

        try:
            t0 = float(self._frame_time_s)
            dt = max(1e-3, float(dt_s))

            x0 = float(ship.position.x)
            y0 = float(ship.position.y)
            vx = float(ship.velocity.x)
            vy = float(ship.velocity.y)

            frame = self._active_frame()

            fx0, fy0 = frame.to_this_frame_xy(t0, x0, y0)
            fx1, fy1 = frame.to_this_frame_xy(
                t0 + dt,
                x0 + vx * dt,
                y0 + vy * dt,
            )

            dvx = float(fx1) - float(fx0)
            dvy = float(fy1) - float(fy0)

            return math.hypot(dvx, dvy) / dt
        except Exception:
            return None

    def _format_speed_label(self, speed_m_s):
        if speed_m_s is None:
            return ""

        speed_m_s = float(speed_m_s)
        if speed_m_s >= 1000.0:
            return f"{speed_m_s / 1000.0:.2f} km/s"

        return f"{speed_m_s:.1f} m/s"
    
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
            self._draw_ship_arrow(body, x, y, r, g, b, theta_override=theta_frame)
            # Schiffs-Label mit camera.world_to_screen zeichnen, um
            # konsistente Welt->Bildschirm-Abbildung zu gewährleisten und
            # FBO/Projektions-Inkonsistenzen zu vermeiden, die Label-Flackern
            # beim Umschalten der Verfolgung verursachen können.
            try:
                lx, ly = camera.world_to_screen(body.position)
                entry = self._get_label_texture(body.name, self.font_small)
                if entry:
                    _, w, h = entry
                    label_x = float(lx) - (float(w) / 2.0)
                    label_y = float(ly) - 16.0
                    self._blit_cached_text(body.name, label_x, label_y, self.font_small)
                else:
                    self._blit_cached_text(body.name, float(lx) + 12.0, float(ly) - 8.0, self.font_small)
                speed = self._ship_frame_speed_m_s(body)
                if getattr(self, "debug_frame", False):
                    try:
                        period = max(1, int(getattr(self, "_frame_debug_period", 30)))
                        if int(getattr(self, "_frame_debug_counter", 0)) % period == 0:
                            backend_rel = self._ship_relative_speed_m_s(body, getattr(self, "current_reference_body", None))
                            frame_speed = speed
                            if backend_rel is not None and frame_speed is not None:
                                print(
                                    f"SHIP_SPEED_DBG: "
                                    f"backend_rel={backend_rel:.3f} m/s "
                                    f"frame={frame_speed:.3f} m/s "
                                    f"frame_label={getattr(self._active_frame(), 'label', '?')}"
                                )
                    except Exception:
                        pass
                speed_text = self._format_speed_label(speed)
                if speed_text:
                    speed_entry = self._get_label_texture(speed_text, self.font_small)
                    if speed_entry:
                        _, sw, _ = speed_entry
                        self._blit_cached_text(speed_text, float(lx) - (float(sw) / 2.0), float(ly) + 24.0, self.font_small)
                    else:
                        self._blit_cached_text(speed_text, float(lx) + 12.0, float(ly) + 24.0, self.font_small)
            except Exception:
                try:
                    self._draw_body_label(body.name, screen_pos, 12)
                except Exception:
                    pass
            return

        # --- Nicht-Schiff-Körper: off-screen-cull + größen-schwelle (icon-swap) ---
        # Echter, UNgeklemmter bildschirmradius. Statt den körper (alt) auf
        # min. 3px zu klemmen und dauerhaft als winzige scheibe zu zeichnen,
        # lassen wir ihn unter die schwelle schrumpfen und tauschen ihn dann
        # nahtlos gegen ein positions-icon konstanter größe.
        icon_radius_px = float(self.body_icon_radius_px)
        true_radius_px = float(body.radius) * float(camera.scale)
        as_icon = true_radius_px < icon_radius_px

        # Off-screen-cull (NUR rendering, physik unberührt): die marge deckt für
        # sichtbare körper den glow (~2.5x radius) ab, damit randständige große
        # körper nicht fälschlich verschwinden. Vollständig off-screen-körper
        # werden gar nicht erst gezeichnet (kein shader-/icon-aufruf).
        cull_margin_px = (icon_radius_px if as_icon else true_radius_px * 2.5) + 8.0
        if not self._is_on_screen(x, y, cull_margin_px):
            self.debug_info['bodies_culled'] = self.debug_info.get('bodies_culled', 0) + 1
            return

        self.debug_info['bodies_rendered'] += 1

        if as_icon:
            # Körper komplett de-rendern; nur das positions-icon zeichnen.
            # icon-größe == swap-schwelle => exakt nahtloser tausch (keine lücke,
            # keine doppelzeichnung), konstante bildschirmgröße beim herauszoomen.
            self.debug_info['bodies_as_icon'] = self.debug_info.get('bodies_as_icon', 0) + 1
            self._draw_body_icon(x, y, icon_radius_px, r, g, b)
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
        )

        if radius > 5:
            # Label-Position mittels camera.world_to_screen berechnen, um
            # inkonsistente Koordinatensysteme zwischen FBO und Hauptpuffer zu vermeiden.
            try:
                lx, ly = camera.world_to_screen(body.position)
                entry = self._get_label_texture(body.name, self.font_small)
                if entry:
                    _, w, h = entry
                    label_x = float(lx) - (float(w) / 2.0)
                    label_y = float(ly) + float(radius_px) + 6.0
                    self._blit_cached_text(body.name, label_x, label_y, self.font_small)
                else:
                    self._blit_cached_text(body.name, float(lx) + float(radius_px) + 2.0, float(ly) - 8.0, self.font_small)
            except Exception:
                try:
                    self._draw_body_label(body.name, screen_pos, radius_px)
                except Exception:
                    pass

    def _draw_ship_arrow(self, body, x, y, r, g, b, theta_override=None):
        # in festen bildschirm-pixeln zeichnen damit schiffgröße beim zoomen konstant bleibt.
        arrow_length = 18.0
        arrow_half_width = 7.0
        tail_offset = 6.0

        theta = float(theta_override) if theta_override is not None else float(getattr(body, 'theta', 0.0))

        # Match camera.world_to_screen() y-inversion to keep visual heading correct.
        hx = math.cos(theta)
        hy = -math.sin(theta)
        nx = -hy
        ny = hx
        # Stash the ACTUAL drawn nose screen-direction so diagnostics can compare
        # the real arrow pixels against the drawn vectors (non-circular check).
        self._last_arrow_screen_dir = (hx, hy)

        # ursprung anpassen damit der dreiecks-schwerpunkt an (x, y) liegt.
        # der schwerpunkt des dreiecks aus nase und schwanz-ecken liegt
        # entlang der richtung versetzt um (arrow_length - 2*tail_offset)/3
        # in bildschirm-pixeln. verschiebe den lokalen ursprung zurück um diesen
        # betrag damit die welt-position des schiffs dem visuellen mittelpunkt des pfeils entspricht.
        centroid_offset = (arrow_length - 2.0 * tail_offset) / 3.0
        origin_x = x - hx * centroid_offset
        origin_y = y - hy * centroid_offset

        nose_x = origin_x + hx * arrow_length
        nose_y = origin_y + hy * arrow_length
        tail_x = origin_x - hx * tail_offset
        tail_y = origin_y - hy * tail_offset

        left_x = tail_x + nx * arrow_half_width
        left_y = tail_y + ny * arrow_half_width
        right_x = tail_x - nx * arrow_half_width
        right_y = tail_y - ny * arrow_half_width

        self._draw_ortho_shape(
            [(nose_x, nose_y), (left_x, left_y), (right_x, right_y)],
            color=(r, g, b, 1.0),
            mode=moderngl.TRIANGLES,
        )
        # debug: kleine marker zeichnen und einzeilige info ausgeben die
        # den dreiecks-schwerpunkt mit der übergebenen screen-position vergleicht.
        try:
            if self.debug_predictor:
                centroid_x = (nose_x + left_x + right_x) / 3.0
                centroid_y = (nose_y + left_y + right_y) / 3.0
                print(f"PRED_DBG_DRAW: centroid=({centroid_x:.6f},{centroid_y:.6f}) screen_pos=({x:.6f},{y:.6f})")
                # magenta cross = centroid, cyan cross = passed screen pos
                size = 3.0
                self._draw_ortho_shape(
                    [(centroid_x - size, centroid_y), (centroid_x + size, centroid_y),
                     (centroid_x, centroid_y - size), (centroid_x, centroid_y + size)],
                    color=(1.0, 0.0, 1.0, 1.0),
                    mode=moderngl.LINES,
                )
                self._draw_ortho_shape(
                    [(x - size, y), (x + size, y),
                     (x, y - size), (x, y + size)],
                    color=(0.0, 1.0, 1.0, 1.0),
                    mode=moderngl.LINES,
                )
        except Exception:
            pass

    def _draw_orbit(self, body, camera, segments=None, color=None, width=1.0):
        """Zeichnet eine vollständige Orbit-Ellipse für Körper mit scripted-Orbits.
        Verwendet `semi_major_axis` (a) und `eccentricity` (e). Wenn `is_moon_of`
        auf ein Eltern-Objekt gesetzt ist, wird die Bahn um die Position des Elternteils
        gezeichnet; andernfalls ist der Fokus im Weltursprung.
        Die Anzahl der Segmente wird adaptiv anhand des Umfangs in Bildschirmpixeln
        gewählt, um die Linie glatt aber performant zu halten.
        """
        a = getattr(body, 'semi_major_axis', None)
        e = getattr(body, 'eccentricity', None)
        if a is None or e is None:
            return
        try:
            a = float(a)
            e = float(e)
        except Exception:
            return
        if a <= 0:
            return

        parent = getattr(body, 'is_moon_of', None)
        if parent is not None and hasattr(parent, 'position'):
            cx = float(parent.position.x)
            cy = float(parent.position.y)
        else:
            cx, cy = 0.0, 0.0

        # screen transform params
        half_w = self.width * 0.5
        half_h = self.height * 0.5
        cam_frame_x, cam_frame_y = self._frame_camera_xy(camera)
        scale = abs(float(camera.scale))

        # adaptive segment count: aim for ~1 segment per ~4px of circumference
        circ_px = 2.0 * math.pi * a * scale
        if segments is None:
            est = int(max(48, min(1024, circ_px / 4.0))) if circ_px > 0 else 128
            segments = max(48, min(2048, est))

        # build screen-space polyline
        screen_points = []
        for i in range(segments + 1):
            phi = 2.0 * math.pi * i / segments
            r = a * (1.0 - e * e) / (1.0 + e * math.cos(phi))
            wx = cx + r * math.cos(phi)
            wy = cy + r * math.sin(phi)
            frame_x, frame_y = self._frame_transform_xy(wx, wy)
            sx = half_w + (frame_x - cam_frame_x) * scale
            sy = half_h - (frame_y - cam_frame_y) * scale
            screen_points.append((sx, sy))

        # Nur sichtbare Abschnitte zeichnen, um unnötige GPU-Arbeit zu vermeiden
        runs = self._visible_window_runs(screen_points, margin_px=self.prediction_visibility_margin_px)
        if not runs:
            return

        if color is None:
            base = getattr(body, 'color', (200, 200, 200))
            cr, cg, cb = (base[0] / 255.0 * 0.8, base[1] / 255.0 * 0.8, base[2] / 255.0 * 0.8)
        else:
            cr, cg, cb = color

        for run in runs:
            if len(run) < 2:
                continue
            self._draw_polyline(run, color=(cr, cg, cb, 0.6), width=width)

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

    def _is_on_screen(self, sx, sy, margin_px):
        return (-margin_px <= sx <= self.width + margin_px and
                -margin_px <= sy <= self.height + margin_px)

    def _visible_window_runs(self, screen_points, margin_px):
        return self._build_clipped_polyline_runs(screen_points, margin_px)

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

    def _iter_prediction_indices_evenly(self, count, max_scan):
        count = int(count)
        max_scan = int(max_scan)

        if count <= 0:
            return []

        if max_scan <= 0 or count <= max_scan:
            return list(range(count))

        if max_scan == 1:
            return [0]

        step = (count - 1) / float(max_scan - 1)
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
            sx, sy = self._world_to_screen_xy_at_time(wx, wy, camera, t_abs, camera_frame_xy)
            if not (math.isfinite(sx) and math.isfinite(sy)):
                continue
            if not self._is_on_screen(sx, sy, 32.0):
                continue

            if is_apo:
                color = (0.45, 0.75, 1.0, 0.95)
                label = "Ap"
            else:
                color = (1.0, 0.62, 0.25, 0.95)
                label = "Pe"

            run = [
                (sx, sy - r_px),
                (sx + r_px, sy),
                (sx, sy + r_px),
                (sx - r_px, sy),
                (sx, sy - r_px),
            ]
            self._draw_polyline(run, color=color, width=2.0)

            text = f"{label} {self._format_apsis_distance(dist)}"
            try:
                # Diamant + linie werden über den line-shader in Y-nach-unten-
                # bildschirmkoordinaten gezeichnet (sy wächst nach unten). Text
                # dagegen läuft über die fixed-function-ortho aus render()
                # (gluOrtho2D(0, w, 0, h), Y-nach-oben) mit vertikal geflippter
                # textur. Ohne umrechnung landet das label an der über die
                # bildschirmmitte gespiegelten position — bei markern fernab der
                # mitte weit neben ihrem diamant. Daher sy -> ortho-Y flippen.
                entry = self._get_label_texture(text, self.font_small)
                th = float(entry[2]) if entry else float(self.font_small.get_height())
                tw = float(entry[1]) if entry else 0.0
                # oberkante des labels knapp unter die untere diamant-spitze
                # (bildschirm-abwärts = kleineres ortho-Y): ortho_top = H - (sy + r + 4)
                label_x = sx - tw / 2.0
                label_y = float(self.height) - (sy + r_px + 4.0) - th
                self._blit_cached_text(text, label_x, label_y, self.font_small)
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

        runs = self._build_clipped_polyline_runs(screen_points, margin_px)
        stats['runs'] = len(runs)
        stats['clipped_runs'] = len(runs)
        if not runs:
            stats['draw_points'] = 0
            return []

        sampled_runs = []

        for run in runs:
            run_starts_at_path_origin = (
                abs(run[0][0] - screen_points[0][0]) < 1e-9 and
                abs(run[0][1] - screen_points[0][1]) < 1e-9
            )

            min_step2 = min_step_px * min_step_px
            compact = [run[0]]
            for sx, sy in run[1:]:
                lx, ly = compact[-1]
                dx = sx - lx
                dy = sy - ly
                if dx * dx + dy * dy >= min_step2:
                    compact.append((sx, sy))
            if compact[-1] != run[-1]:
                compact.append(run[-1])

            if len(compact) > 2:
                keep_indices = self._rdp_indices(compact, tolerance_px)
                if run_starts_at_path_origin:
                    preserve_count = min(32, len(compact))
                    forced = set(range(preserve_count))
                    merged = set(keep_indices)
                    merged.update(forced)
                    keep_indices = sorted(merged)

                # Guard against over-aggressive simplification by enforcing
                # a maximum screen-space gap between consecutive kept points.
                if len(keep_indices) > 1:
                    max_seg = max(0.5, float(max_segment_px))
                    refined = [keep_indices[0]]
                    for i in range(1, len(keep_indices)):
                        start_idx = refined[-1]
                        end_idx = keep_indices[i]
                        if end_idx <= start_idx:
                            continue

                        sx0, sy0 = compact[start_idx]
                        sx1, sy1 = compact[end_idx]
                        seg_dx = sx1 - sx0
                        seg_dy = sy1 - sy0
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

                    keep_indices = refined

                sampled = [compact[i] for i in keep_indices]
            else:
                sampled = compact

            # Densify only the RDP-kept points, not the raw scan.
            # Pre-RDP densification of sparse predictors could expand 3000 samples
            # to 75 000+ linearly-interpolated dummies that RDP discards anyway,
            # making _rdp_indices O(N²) on a huge but information-free array.
            sampled = self._densify_screen_run(sampled, max_segment_px)

            if len(sampled) >= 2:
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

    def _draw_body_label(self, name, screen_pos, radius):
        # Label mit gecachten GL-Texturen zeichnen, um pro-Frame GL-Allocationen zu vermeiden.
        # Label horizontal zentrieren und über dem Körper platzieren, um
        # Fehlausrichtungen beim Zoomen oder bei Radiusänderungen zu vermeiden.
        try:
            entry = self._get_label_texture(name, self.font_small)
            if entry:
                _, w, h = entry
                label_x = float(screen_pos[0]) - (float(w) / 2.0)
                # Label über dem Körper platzieren; Bildschirm-Y wächst nach oben.
                label_y = float(screen_pos[1]) + float(radius) + 6.0
                self._blit_cached_text(name, label_x, label_y, self.font_small)
                return
        except Exception:
            pass

        # Fallback: previous heuristic
        label_x = screen_pos[0] + radius + 2
        label_y = screen_pos[1] - 8
        self._blit_cached_text(name, label_x, label_y, self.font_small)
    
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

    def _render_hud(self, camera, predictor=None):
        # HUD-Texte vorbereiten
        def _fmt_dist(n):
            if n is None:
                return 'auto'
            try:
                n = float(n)
            except Exception:
                return str(n)
            if n >= 1e9:
                return f"{n/1e9:.2f}Gm"
            if n >= 1e6:
                return f"{n/1e6:.2f}Mm"
            if n >= 1e3:
                return f"{n/1e3:.2f}km"
            return f"{n:.0f}m"

        texts = [
            f"Scale: {camera.scale:.2e} px/m",
            f"Position: ({camera.position.x:.2e}, {camera.position.y:.2e})",
            f"Target: {camera.target.name if camera.target else 'None'}",
            f"Plot frame: {self._plotting_frame_label}",
            f"Target overlay: {self._target_frame_label if self._target_frame_label else 'OFF'}",
            f"Ref trails: {'ON' if self.reference_trajectories_enabled else 'OFF'}",
            f"Time step: {camera.sim_dt:.2e} s/step",
            f"Bodies rendered: {self.debug_info['bodies_rendered']}",
            f"FXAA: {'ON' if self.enable_fxaa else 'OFF'}",
        ]

        if predictor is not None:
            precision_factor = predictor.get_precision_factor() if hasattr(predictor, 'get_precision_factor') else 1.0
            display_length = predictor.get_display_length() if hasattr(predictor, 'get_display_length') else predictor.length
            texts += [
                f"Predictor len: {_fmt_dist(display_length)} ([+/-])",
                f"Predictor spacing: {_fmt_dist(predictor.precision)} ([9/0])",
                f"Predictor precision factor: {precision_factor:.2f}x",
                f"Pred points: {len(predictor.get_points())}/{predictor.num_points}",
                f"Pred draw points: {self.debug_info['prediction_points_drawn']}/{self.debug_info['prediction_points_in']}",
            ]
            if hasattr(predictor, 'get_async_status'):
                async_status = predictor.get_async_status()
                texts.append(
                    f"Pred async: {'ON' if async_status['enabled'] else 'OFF'} "
                    f"pending={async_status['pending']} swapped={async_status['swapped_jobs']}"
                )

        texts.append("[WASD] Move | [F] Unfollow | [Scroll] Zoom | [R] Cycle ref | [1]/[2] Frame mode | [T] Target overlay")
        
        # Pygame Surface für HUD erstellen
        line_height = 16
        hud_width = 560
        hud_height = max(40, len(texts) * line_height + 8)
        origin_x = 10
        origin_y = self.height - hud_height - 10

        # Bei unverändertem text bleibt die persistente HUD-textur gültig:
        # font.render (~1 pro zeile), Surface-allokation, tostring und der
        # textur-upload entfallen, es genügt ein redraw der bestehenden textur.
        cache_key = (tuple(texts), int(self.width), int(self.height))
        if cache_key == self._hud_cache_key and self._hud_texture is not None:
            self._draw_hud_quad(origin_x, origin_y, *self._hud_texture_size)
            return

        hud_surface = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)

        # Zeilen-surfaces cachen: ändert sich nur eine zeile (z. B. kamera-
        # position beim verfolgen), durchlaufen die übrigen kein font.render.
        # Der cache wird pro frame auf die aktuellen texte reduziert und kann
        # daher nicht wachsen.
        new_line_cache = {}
        for i, text in enumerate(texts):
            text_surface = self._hud_line_surface_cache.get(text)
            if text_surface is None:
                text_surface = self.font_medium.render(text, True, (255, 255, 255))
            new_line_cache[text] = text_surface
            hud_surface.blit(text_surface, (0, i * line_height))
        self._hud_line_surface_cache = new_line_cache

        # HUD in OpenGL rendern
        self._blit_pygame_surface(hud_surface, origin_x, origin_y)
        self._hud_cache_key = cache_key
    
    def resize(self, width, height):

        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
        try:
            self.ctx.screen.viewport = (0, 0, width, height)
        except Exception:
            pass

        # Framebuffer neu erstellen wenn FXAA aktiviert (programm + VAO
        # bleiben; nur textur/FBO hängen von der fenstergröße ab).
        if self.enable_fxaa:
            self._release_fxaa_targets()
            try:
                self._create_fxaa_targets()
                if self.fxaa_program is not None:
                    self.fxaa_program['u_resolution'].value = (float(width), float(height))
            except Exception as e:
                print(f"FXAA resize failed: {e}")
                self._release_fxaa_targets()
                self.enable_fxaa = False
        # Clear HUD and label texture caches (will be recreated lazily)
        try:
            for entry in list(self._label_texture_cache.values()):
                texture = entry[0]
                if texture:
                    try:
                        texture.release()
                    except Exception:
                        pass
        except Exception:
            pass
        self._label_texture_cache = {}
        if getattr(self, '_hud_texture', None):
            try:
                self._hud_texture.release()
            except Exception:
                pass
            self._hud_texture = None
            self._hud_texture_size = (0, 0)
        # HUD-memoization invalidieren: textur und viewport haben sich geändert.
        self._hud_cache_key = None
        # Der poly-VBO ist größen-unabhängig und bleibt (samt VAOs) bestehen.
