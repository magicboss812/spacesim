"""
OpenGL-Renderer für die Weltraumsimulation.
Verwendet pygame für Fensterverwaltung und HUD, OpenGL für Rendering.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import os
from collections import deque
import ctypes
import time

import numpy as np

from reference_frames import IdentityReferenceFrame


class Renderer:
    def __init__(self, width, height, enable_fxaa=True):

        self.width = width
        self.height = height
        self.enable_fxaa = enable_fxaa
        
        # FXAA Framebuffer und Textur
        self.fbo = None
        self.fbo_texture = None
        self.fxaa_program = None
        self.fxaa_vertex_shader = None
        self.fxaa_fragment_shader = None
        
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
            'prediction_points_in': 0,
            'prediction_points_drawn': 0,
        }
        self.render_benchmark_debug = False
        self.render_benchmark_every_n_frames = 60
        self._render_benchmark_frame = 0
        self._last_prediction_render_stats = {}

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
        self._prediction_line_cache_key_value = None
        self._prediction_line_cache_points = None
        self._prediction_line_cache_stats = {}

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
        self.reference_trajectories_max_points = 400
        self.reference_trajectories_sample_step_s = 10.0
        self._reference_traj_last_sample_time = None
        self._reference_traj_points = {}
        self._shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')

        # gpu-helpers: wiederverwendbare VBO- und texture-caches
        self._poly_vbo = None
        self._poly_vbo_size = 0
        self._line_program = None
        self._line_a_pos = -1
        self._line_u_viewport = -1
        self._line_u_color = -1
        self._body_program = None
        self._body_a_corner = -1
        self._body_u_center_px = -1
        self._body_u_outer_radius_px = -1
        self._body_u_viewport = -1
        self._body_u_base_color = -1
        self._body_u_atmos_color = -1
        self._body_u_core_radius_norm = -1
        self._body_u_atmos_radius_norm = -1
        self._body_u_atmos_alpha = -1
        self._body_u_glow_alpha = -1
        self._body_quad_vbo = None

        self._label_texture_cache = {}
        self._hud_texture = None
        self._hud_texture_size = (0, 0)
        # Initialize GPU helpers (VBOs, caches)
        try:
            self._init_gpu_helpers()
        except Exception:
            pass
    
    def _init_opengl(self):
        """Initialisiert OpenGL-Einstellungen."""
        glViewport(0, 0, self.width, self.height)
        
        # Orthogonale Projektion (2D)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Blending aktivieren
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Depth Test deaktivieren (2D)
        glDisable(GL_DEPTH_TEST)
        
        # Hintergrundfarbe (dunkelblau)
        glClearColor(0.0, 0.0, 0.05, 1.0)
        
        # VSync aktivieren (falls verfügbar)
        try:
            # Versuche verschiedene OpenGL-Funktionen für VSync
            import ctypes
            try:
                # Windows
                wgl = ctypes.windll.opengl32
                wgl.wglSwapIntervalEXT(1)
            except (AttributeError, OSError):
                try:
                    # Linux/Mac
                    glX = ctypes.CDLL('libGL.so.1')
                    glX.glXSwapIntervalMESA(1)
                except (AttributeError, OSError):
                    # Versuche PyOpenGL
                    from OpenGL import GL
                    if hasattr(GL, 'glSwapIntervalEXT'):
                        GL.glSwapIntervalEXT(1)
                    elif hasattr(GL, 'glXSwapIntervalMESA'):
                        GL.glXSwapIntervalMESA(1)
        except Exception:
            pass  # VSync nicht verfügbar
    
    def _init_fxaa(self):
        """Initialisiert FXAA Framebuffer und Shader."""
        try:
            # Framebuffer erstellen
            self.fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
            
            # Textur für Framebuffer erstellen
            self.fbo_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.fbo_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 
                        0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            
            # Textur an Framebuffer anhängen
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                                   GL_TEXTURE_2D, self.fbo_texture, 0)
            
            # Framebuffer-Status prüfen
            status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
            if status != GL_FRAMEBUFFER_COMPLETE:
                print(f"FXAA Framebuffer error: {status}")
                self.enable_fxaa = False
                return
            
            # Zurück zum Standard-Framebuffer
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            
            # FXAA Shader laden
            self._load_fxaa_shaders()
            
            print("FXAA initialized successfully")
        except Exception as e:
            print(f"FXAA initialization failed: {e}")
            self.enable_fxaa = False
    
    def _load_fxaa_shaders(self):
        fxaa_vertex_source = """
        #version 120
        varying vec2 v_texcoord;
        void main() {
            v_texcoord = gl_Vertex.xy * 0.5 + 0.5;
            gl_Position = gl_Vertex;
        }
        """
        
        fxaa_fragment_source = """
        #version 120
        uniform sampler2D u_texture;
        uniform vec2 u_resolution;
        varying vec2 v_texcoord;
        
        float luminance(vec3 c) {
            return dot(c, vec3(0.299, 0.587, 0.114));
        }
        
        void main() {
            vec2 texel_size = 1.0 / u_resolution;
            vec2 uv = v_texcoord;
            
            vec3 center = texture2D(u_texture, uv).rgb;
            float center_luma = luminance(center);
            
            vec3 nw = texture2D(u_texture, uv + vec2(-1.0, -1.0) * texel_size).rgb;
            vec3 ne = texture2D(u_texture, uv + vec2(1.0, -1.0) * texel_size).rgb;
            vec3 sw = texture2D(u_texture, uv + vec2(-1.0, 1.0) * texel_size).rgb;
            vec3 se = texture2D(u_texture, uv + vec2(1.0, 1.0) * texel_size).rgb;
            
            float luma_nw = luminance(nw);
            float luma_ne = luminance(ne);
            float luma_sw = luminance(sw);
            float luma_se = luminance(se);
            
            float luma_min = min(center_luma, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
            float luma_max = max(center_luma, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));
            float luma_range = luma_max - luma_min;
            
            if (luma_range < 0.0312) {
                gl_FragColor = vec4(center, 1.0);
                return;
            }
            
            float gradient_nw_se = abs(luma_nw - luma_se);
            float gradient_ne_sw = abs(luma_ne - luma_sw);
            float contrast = max(gradient_nw_se, gradient_ne_sw);
            
            if (contrast < 0.0625) {
                gl_FragColor = vec4(center, 1.0);
                return;
            }
            
            vec2 dir;
            dir.x = -((luma_nw + luma_ne) - (luma_sw + luma_se));
            dir.y = ((luma_nw + luma_sw) - (luma_ne + luma_se));
            
            float dir_reduce = max((luma_nw + luma_ne + luma_sw + luma_se) * 0.25, 0.125);
            float rcp_dir_min = 1.0 / (min(abs(dir.x), abs(dir.y)) + dir_reduce);
            
            dir = min(vec2(8.0), max(vec2(-8.0), dir * rcp_dir_min)) * texel_size;
            
            vec3 result_a = 0.5 * (
                texture2D(u_texture, uv + dir * (1.0/3.0 - 0.5)).rgb +
                texture2D(u_texture, uv + dir * (2.0/3.0 - 0.5)).rgb
            );
            vec3 result_b = result_a * 0.5 + 0.25 * (
                texture2D(u_texture, uv + dir * -0.5).rgb +
                texture2D(u_texture, uv + dir * 0.5).rgb
            );
            
            float luma_b = luminance(result_b);
            
            if (luma_b < luma_min || luma_b > luma_max) {
                gl_FragColor = vec4(result_a, 1.0);
            } else {
                gl_FragColor = vec4(result_b, 1.0);
            }
        }
        """
        
        # Vertex Shader kompilieren
        self.fxaa_vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(self.fxaa_vertex_shader, fxaa_vertex_source)
        glCompileShader(self.fxaa_vertex_shader)
        
        # Prüfen ob Vertex Shader kompiliert wurde
        if not glGetShaderiv(self.fxaa_vertex_shader, GL_COMPILE_STATUS):
            log = glGetShaderInfoLog(self.fxaa_vertex_shader).decode('utf-8')
            print(f"FXAA Vertex Shader Error: {log}")
            self.enable_fxaa = False
            return
        
        # Fragment Shader kompilieren
        self.fxaa_fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(self.fxaa_fragment_shader, fxaa_fragment_source)
        glCompileShader(self.fxaa_fragment_shader)
        
        # Prüfen ob Fragment Shader kompiliert wurde
        if not glGetShaderiv(self.fxaa_fragment_shader, GL_COMPILE_STATUS):
            log = glGetShaderInfoLog(self.fxaa_fragment_shader).decode('utf-8')
            print(f"FXAA Fragment Shader Error: {log}")
            self.enable_fxaa = False
            return
        
        # Programm erstellen
        self.fxaa_program = glCreateProgram()
        glAttachShader(self.fxaa_program, self.fxaa_vertex_shader)
        glAttachShader(self.fxaa_program, self.fxaa_fragment_shader)
        glLinkProgram(self.fxaa_program)
        
        # Prüfen ob Programm gelinkt wurde
        if not glGetProgramiv(self.fxaa_program, GL_LINK_STATUS):
            log = glGetProgramInfoLog(self.fxaa_program).decode('utf-8')
            print(f"FXAA Program Link Error: {log}")
            self.enable_fxaa = False
            return
        
        # Shader aufräumen
        glDeleteShader(self.fxaa_vertex_shader)
        glDeleteShader(self.fxaa_fragment_shader)
        
        print("FXAA Shader loaded successfully")
    
    def _apply_fxaa(self):
        """Wendet FXAA Post-Processing an."""
        if not self.enable_fxaa or not self.fbo_texture or not self.fxaa_program:
            return
        
        # OpenGL Status zurücksetzen
        glUseProgram(0)
        glDisable(GL_BLEND)
        
        # Projektion für Vollbild-Quad zurücksetzen
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # FXAA Shader aktivieren
        glUseProgram(self.fxaa_program)
        
        # Textur binden
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.fbo_texture)
        
        # Uniforms setzen
        glUniform1i(glGetUniformLocation(self.fxaa_program, 'u_texture'), 0)
        glUniform2f(glGetUniformLocation(self.fxaa_program, 'u_resolution'), 
                    float(self.width), float(self.height))
        
        # Vollbild-Quad rendern
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(-1, 1)
        glEnd()
        
        # Shader deaktivieren
        glUseProgram(0)
        
        # Blending wieder aktivieren
        glEnable(GL_BLEND)
        
        # Projektion zurücksetzen
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

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
        return self._frame_transform_xy(float(camera.position.x), float(camera.position.y))

    def _world_to_screen_xy(self, world_x, world_y, camera, camera_frame_xy=None):
        if camera_frame_xy is None:
            camera_frame_xy = self._frame_camera_xy(camera)
        frame_x, frame_y = self._frame_transform_xy(world_x, world_y)
        scale = float(camera.scale)
        sx = self.width * 0.5 + (frame_x - camera_frame_xy[0]) * scale
        sy = self.height * 0.5 - (frame_y - camera_frame_xy[1]) * scale
        return sx, sy

    def _world_to_screen_xy_at_time(self, world_x, world_y, camera, time_s):
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

        # Kamera-Ursprung zur selben Zeit im Frame-Raum berechnen
        try:
            cam_fx, cam_fy = frame.to_this_frame_xy(float(time_s), float(camera.position.x), float(camera.position.y))
        except Exception:
            cam_fx, cam_fy = self._frame_transform_xy(float(camera.position.x), float(camera.position.y))

        scale = float(camera.scale)
        sx = self.width * 0.5 + (frame_x - cam_fx) * scale
        sy = self.height * 0.5 - (frame_y - cam_fy) * scale
        return sx, sy

    def _reset_reference_trajectories(self):
        self._reference_traj_points = {}
        self._reference_traj_last_sample_time = None

    def _init_gpu_helpers(self):
        """Erstellt wiederverwendbare puffer und GLSL-programme für kritische render-pfade."""
        self._ensure_poly_vbo()
        self._init_line_pipeline()
        self._init_body_pipeline()

    def _load_shader_source(self, filename):
        path = os.path.join(self._shader_dir, filename)
        with open(path, 'r', encoding='utf-8') as shader_file:
            return shader_file.read()

    def _decode_gl_log(self, value):
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='replace')
        return str(value)

    def _compile_shader_program(self, vertex_filename, fragment_filename, label):
        vertex_shader = None
        fragment_shader = None
        program = None
        try:
            vertex_source = self._load_shader_source(vertex_filename)
            fragment_source = self._load_shader_source(fragment_filename)

            vertex_shader = glCreateShader(GL_VERTEX_SHADER)
            glShaderSource(vertex_shader, vertex_source)
            glCompileShader(vertex_shader)
            if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
                log = self._decode_gl_log(glGetShaderInfoLog(vertex_shader))
                raise RuntimeError(f"vertex compile failed: {log}")

            fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
            glShaderSource(fragment_shader, fragment_source)
            glCompileShader(fragment_shader)
            if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
                log = self._decode_gl_log(glGetShaderInfoLog(fragment_shader))
                raise RuntimeError(f"fragment compile failed: {log}")

            program = glCreateProgram()
            glAttachShader(program, vertex_shader)
            glAttachShader(program, fragment_shader)
            glLinkProgram(program)
            if not glGetProgramiv(program, GL_LINK_STATUS):
                log = self._decode_gl_log(glGetProgramInfoLog(program))
                raise RuntimeError(f"program link failed: {log}")

            return program
        except Exception as exc:
            self.debug_info['shader_error'] = f"{label}: {exc}"
            print(f"Shader pipeline fallback ({label}): {exc}")
            if program is not None:
                try:
                    glDeleteProgram(program)
                except Exception:
                    pass
            return None
        finally:
            if vertex_shader is not None:
                try:
                    glDeleteShader(vertex_shader)
                except Exception:
                    pass
            if fragment_shader is not None:
                try:
                    glDeleteShader(fragment_shader)
                except Exception:
                    pass

    def _init_line_pipeline(self):
        program = self._compile_shader_program('line.vert', 'line.frag', 'line')
        if program is None:
            self._line_program = None
            return

        try:
            a_pos = glGetAttribLocation(program, 'a_pos')
            u_viewport = glGetUniformLocation(program, 'u_viewport')
            u_color = glGetUniformLocation(program, 'u_color')
            if a_pos < 0 or u_viewport < 0 or u_color < 0:
                raise RuntimeError('missing line shader attribute/uniform')

            self._line_program = program
            self._line_a_pos = a_pos
            self._line_u_viewport = u_viewport
            self._line_u_color = u_color
        except Exception as exc:
            self.debug_info['shader_error'] = f"line: {exc}"
            print(f"Shader pipeline fallback (line): {exc}")
            try:
                glDeleteProgram(program)
            except Exception:
                pass
            self._line_program = None

    def _init_body_pipeline(self):
        program = self._compile_shader_program('body.vert', 'body.frag', 'body')
        if program is None:
            self._body_program = None
            return

        try:
            a_corner = glGetAttribLocation(program, 'a_corner')
            u_center_px = glGetUniformLocation(program, 'u_center_px')
            u_outer_radius_px = glGetUniformLocation(program, 'u_outer_radius_px')
            u_viewport = glGetUniformLocation(program, 'u_viewport')
            u_base_color = glGetUniformLocation(program, 'u_base_color')
            u_atmos_color = glGetUniformLocation(program, 'u_atmos_color')
            u_core_radius_norm = glGetUniformLocation(program, 'u_core_radius_norm')
            u_atmos_radius_norm = glGetUniformLocation(program, 'u_atmos_radius_norm')
            u_atmos_alpha = glGetUniformLocation(program, 'u_atmos_alpha')
            u_glow_alpha = glGetUniformLocation(program, 'u_glow_alpha')

            if (a_corner < 0 or u_center_px < 0 or u_outer_radius_px < 0 or
                    u_viewport < 0 or u_base_color < 0 or u_atmos_color < 0 or
                    u_core_radius_norm < 0 or u_atmos_radius_norm < 0 or
                    u_atmos_alpha < 0 or u_glow_alpha < 0):
                raise RuntimeError('missing body shader attribute/uniform')

            self._body_program = program
            self._body_a_corner = a_corner
            self._body_u_center_px = u_center_px
            self._body_u_outer_radius_px = u_outer_radius_px
            self._body_u_viewport = u_viewport
            self._body_u_base_color = u_base_color
            self._body_u_atmos_color = u_atmos_color
            self._body_u_core_radius_norm = u_core_radius_norm
            self._body_u_atmos_radius_norm = u_atmos_radius_norm
            self._body_u_atmos_alpha = u_atmos_alpha
            self._body_u_glow_alpha = u_glow_alpha

            if self._body_quad_vbo is None:
                quad_data = (ctypes.c_float * 8)(
                    -1.0, -1.0,
                    1.0, -1.0,
                    -1.0, 1.0,
                    1.0, 1.0,
                )
                self._body_quad_vbo = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, self._body_quad_vbo)
                glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(quad_data), quad_data, GL_STATIC_DRAW)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
        except Exception as exc:
            self.debug_info['shader_error'] = f"body: {exc}"
            print(f"Shader pipeline fallback (body): {exc}")
            try:
                glDeleteProgram(program)
            except Exception:
                pass
            self._body_program = None

    def _ensure_poly_vbo(self):
        if self._poly_vbo is None:
            try:
                self._poly_vbo = glGenBuffers(1)
            except Exception:
                self._poly_vbo = None

    def _draw_polyline(self, run, color=(1.0, 1.0, 1.0, 1.0), width=1.0):
        """Zeichnet eine bildschirm-space polyline via GLSL+VBO mit legacy-fallback."""
        n = len(run)
        if n < 2:
            return
        self._ensure_poly_vbo()

        arr = None
        try:
            arr = np.asarray(run, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 2:
                arr = arr.reshape((-1, 2))
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr, dtype=np.float32)
            n = int(arr.shape[0])
        except Exception:
            arr = None

        glLineWidth(float(width))
        try:
            if self._line_program and self._poly_vbo and arr is not None:
                glDisable(GL_TEXTURE_2D)
                glUseProgram(self._line_program)
                glUniform2f(self._line_u_viewport, float(self.width), float(self.height))
                glUniform4f(self._line_u_color,
                            float(color[0]), float(color[1]), float(color[2]), float(color[3]))

                glBindBuffer(GL_ARRAY_BUFFER, self._poly_vbo)
                data_size = int(arr.nbytes)
                if data_size > int(self._poly_vbo_size):
                    glBufferData(GL_ARRAY_BUFFER, data_size, arr, GL_DYNAMIC_DRAW)
                    self._poly_vbo_size = int(data_size)
                else:
                    glBufferSubData(GL_ARRAY_BUFFER, 0, data_size, arr)
                glEnableVertexAttribArray(self._line_a_pos)
                glVertexAttribPointer(self._line_a_pos, 2, GL_FLOAT, GL_FALSE, 0, None)
                glDrawArrays(GL_LINE_STRIP, 0, n)
                glDisableVertexAttribArray(self._line_a_pos)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                glUseProgram(0)
                return
        except Exception:
            try:
                glUseProgram(0)
            except Exception:
                pass

        # fallback auf immediate-mode wenn GLSL-pfad nicht verfügbar ist.
        glColor4f(float(color[0]), float(color[1]), float(color[2]), float(color[3]))
        glDisable(GL_TEXTURE_2D)
        glLineWidth(float(width))
        glBegin(GL_LINE_STRIP)
        for sx, sy in run:
            glVertex2f(float(sx), float(sy))
        glEnd()

    def _draw_body_glsl(self, x, y, radius, base_color, atmos_color, atmos_density, light_intensity):
        """Zeichnet einen körper als shader-gesteuertes quad (scheibe + optional atmosphäre + glow)."""
        if self._body_program is None or self._body_quad_vbo is None:
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
            glDisable(GL_TEXTURE_2D)
            glUseProgram(self._body_program)

            glUniform2f(self._body_u_center_px, float(x), float(y))
            glUniform1f(self._body_u_outer_radius_px, float(outer_radius))
            glUniform2f(self._body_u_viewport, float(self.width), float(self.height))
            glUniform3f(self._body_u_base_color,
                        float(base_color[0]), float(base_color[1]), float(base_color[2]))
            glUniform3f(self._body_u_atmos_color,
                        float(atmos_color[0]), float(atmos_color[1]), float(atmos_color[2]))
            glUniform1f(self._body_u_core_radius_norm, float(core_norm))
            glUniform1f(self._body_u_atmos_radius_norm, float(atmos_norm))
            glUniform1f(self._body_u_atmos_alpha, float(atmos_alpha))
            glUniform1f(self._body_u_glow_alpha, float(glow_alpha))

            glBindBuffer(GL_ARRAY_BUFFER, self._body_quad_vbo)
            glEnableVertexAttribArray(self._body_a_corner)
            glVertexAttribPointer(self._body_a_corner, 2, GL_FLOAT, GL_FALSE, 0, None)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            glDisableVertexAttribArray(self._body_a_corner)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glUseProgram(0)
            return True
        except Exception:
            try:
                glUseProgram(0)
            except Exception:
                pass
            return False

    def _draw_body_legacy(self, x, y, radius, r, g, b):
        glColor4f(r, g, b, 1.0)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(x, y)
        segments = max(16, min(64, int(radius * 2)))
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            px = x + math.cos(angle) * radius
            py = y + math.sin(angle) * radius
            glVertex2f(px, py)
        glEnd()

    def _get_label_texture(self, text, font):
        key = (text, font.get_height())
        entry = self._label_texture_cache.get(key)
        if entry:
            return entry  # (texid, w, h)
        try:
            surface = font.render(text, True, (255, 255, 255))
            texture_data = pygame.image.tostring(surface, 'RGBA', True)
            w, h = surface.get_size()
            texid = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texid)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
            self._label_texture_cache[key] = (texid, w, h)
            return (texid, w, h)
        except Exception:
            return None

    def _blit_cached_text(self, text, x, y, font):
        entry = self._get_label_texture(text, font)
        if not entry:
            # fallback: immediate per-call blit
            surface = font.render(text, True, (255,255,255))
            texture_data = pygame.image.tostring(surface, 'RGBA', True)
            w, h = surface.get_size()
            tid = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tid)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
            glEnable(GL_TEXTURE_2D)
            glColor4f(1.0, 1.0, 1.0, 1.0)
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex2f(x, y)
            glTexCoord2f(1, 0); glVertex2f(x + w, y)
            glTexCoord2f(1, 1); glVertex2f(x + w, y + h)
            glTexCoord2f(0, 1); glVertex2f(x, y + h)
            glEnd()
            glDisable(GL_TEXTURE_2D)
            try:
                glDeleteTextures(1, [tid])
            except Exception:
                pass
            return
        texid, w, h = entry
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texid)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x, y)
        glTexCoord2f(1, 0); glVertex2f(x + w, y)
        glTexCoord2f(1, 1); glVertex2f(x + w, y + h)
        glTexCoord2f(0, 1); glVertex2f(x, y + h)
        glEnd()
        glDisable(GL_TEXTURE_2D)

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

        glDisable(GL_TEXTURE_2D)

        for body in bodies:
            if getattr(body, 'is_ship', False):
                continue

            trail = self._reference_traj_points.get(id(body))
            if trail is None or len(trail) < 2:
                continue

            base = getattr(body, 'color', (200, 200, 200))
            cr = min(1.0, max(0.0, base[0] / 255.0))
            cg = min(1.0, max(0.0, base[1] / 255.0))
            cb = min(1.0, max(0.0, base[2] / 255.0))

            screen_points = []
            for fx, fy in trail:
                sx = half_w + (fx - camera_frame_xy[0]) * scale
                sy = half_h - (fy - camera_frame_xy[1]) * scale
                screen_points.append((sx, sy))

            runs = self._visible_window_runs(screen_points, margin_px=self.prediction_visibility_margin_px)
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

    def render(self, bodies, camera, prediction_points=None, predictor=None, sim_time=None):
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
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
            glViewport(0, 0, self.width, self.height)

        glClear(GL_COLOR_BUFFER_BIT)

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

        # Schiffsanker einmal berechnen (ship_body wurde oben ermittelt)
        ship_anchor_world = (float(ship_body.position.x), float(ship_body.position.y)) if ship_body is not None else None
        prediction_has_points = self._points_count(prediction_points) > 0
        prediction_drawn = False

        if prediction_has_points and self.enable_fxaa and self.fbo and not self.prediction_bypass_fxaa:
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluOrtho2D(0, self.width, 0, self.height)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            self.draw_prediction(prediction_points, camera, anchor_world=ship_anchor_world)
            prediction_drawn = True

        if self.enable_fxaa and self.fbo:
            # Zurück zum Standard-Framebuffer and apply FXAA post-process
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0, 0, self.width, self.height)
            fxaa_t0 = time.perf_counter()
            self._apply_fxaa()
            timings['fxaa_ms'] += (time.perf_counter() - fxaa_t0) * 1000.0
        
        # Projektion-Matrix für Trajektorienwiedergabe wiederherstellen
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Blending für Trajektorien wieder aktivieren
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        if prediction_has_points and not prediction_drawn:
            self.draw_prediction(prediction_points, camera, anchor_world=ship_anchor_world)

        # Schiff-Marker im Haupt-Framebuffer zeichnen, damit er visuell
        # genau mit dem Prädiktor-Startpunkt übereinstimmt.
        if ship_body is not None:
            bodies_t0 = time.perf_counter()
            self._draw_body(ship_body, camera)
            timings['bodies_ms'] += (time.perf_counter() - bodies_t0) * 1000.0

        hud_t0 = time.perf_counter()
        self._render_hud(camera, predictor)
        timings['hud_ms'] += (time.perf_counter() - hud_t0) * 1000.0
        swap_t0 = time.perf_counter()
        pygame.display.flip()
        timings['swap_or_present_ms'] = (time.perf_counter() - swap_t0) * 1000.0
        timings['frame_ms'] = (time.perf_counter() - frame_t0) * 1000.0
        self._emit_render_benchmark(timings)
    
    def _draw_body(self, body, camera):
        camera_frame_xy = self._frame_camera_xy(camera)
        x, y = self._world_to_screen_xy(
            float(body.position.x),
            float(body.position.y),
            camera,
            camera_frame_xy=camera_frame_xy,
        )
        screen_pos = (x, y)
        # Gleitkomma-Radius in Pixeln für Label-Anker beibehalten, um
        # 1-Pixel-Flackern zu vermeiden, wenn sich der Radius beim Zoomen leicht ändert.
        radius_px = max(3.0, float(body.radius) * float(camera.scale))
        radius = max(3, int(round(radius_px)))  # integer radius for geometry
        
        self.debug_info['bodies_rendered'] += 1
        r, g, b = body.color[0] / 255.0, body.color[1] / 255.0, body.color[2] / 255.0
        x, y = float(screen_pos[0]), float(screen_pos[1])

        if body.is_ship:
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
                    label_y = float(ly) + 12.0 + 6.0
                    self._blit_cached_text(body.name, label_x, label_y, self.font_small)
                else:
                    self._blit_cached_text(body.name, float(lx) + 12.0, float(ly) - 8.0, self.font_small)
            except Exception:
                try:
                    self._draw_body_label(body.name, screen_pos, 12)
                except Exception:
                    pass
            return

        if hasattr(body, 'atmosphere_color'):
            r1, g1, b1 = body.atmosphere_color[0] / 255.0, body.atmosphere_color[1] / 255.0, body.atmosphere_color[2] / 255.0
        else:
            r1, g1, b1 = r, g, b

        has_atmos = bool(getattr(body, 'has_atmosphere', False))
        atmos_density = float(getattr(body, 'atmos_density', 0.0)) if has_atmos else 0.0
        light_intensity = float(getattr(body, 'light_intensity', 0.0))

        # Primärer Pfad: GLSL-Shader zeichnet Scheibe + Glow + Atmosphäre in einem Quad.
        drawn_with_shader = self._draw_body_glsl(
            x,
            y,
            radius_px,
            (r, g, b),
            (r1, g1, b1),
            atmos_density,
            light_intensity,
        )

        # Fallback path: preserve previous immediate-mode behavior.
        if not drawn_with_shader:
            if light_intensity > 0:
                self._draw_glow(x, y, radius_px, r, g, b, light_intensity)
            if has_atmos and atmos_density > 0:
                self._draw_atmosphere(x, y, radius_px, r1, g1, b1, atmos_density)
            self._draw_body_legacy(x, y, radius, r, g, b)
        
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

        glColor4f(r, g, b, 1.0)
        glBegin(GL_TRIANGLES)
        glVertex2f(nose_x, nose_y)
        glVertex2f(left_x, left_y)
        glVertex2f(right_x, right_y)
        glEnd()
        # debug: kleine marker zeichnen und einzeilige info ausgeben die
        # den dreiecks-schwerpunkt mit der übergebenen screen-position vergleicht.
        try:
            if self.debug_predictor:
                centroid_x = (nose_x + left_x + right_x) / 3.0
                centroid_y = (nose_y + left_y + right_y) / 3.0
                print(f"PRED_DBG_DRAW: centroid=({centroid_x:.6f},{centroid_y:.6f}) screen_pos=({x:.6f},{y:.6f})")
                # magenta cross = centroid, cyan cross = passed screen pos
                glColor4f(1.0, 0.0, 1.0, 1.0)
                size = 3.0
                glBegin(GL_LINES)
                glVertex2f(centroid_x - size, centroid_y); glVertex2f(centroid_x + size, centroid_y)
                glVertex2f(centroid_x, centroid_y - size); glVertex2f(centroid_x, centroid_y + size)
                glEnd()
                glColor4f(0.0, 1.0, 1.0, 1.0)
                glBegin(GL_LINES)
                glVertex2f(x - size, y); glVertex2f(x + size, y)
                glVertex2f(x, y - size); glVertex2f(x, y + size)
                glEnd()
        except Exception:
            pass
    
    def _draw_atmosphere(self, x, y, radius, r, g, b, density):

        multiplier = 2.0
        atmos_radius = radius * multiplier

        radius_scale = max(0.5, min(2.0, radius / 50.0))

        base_density = min(density / 100.0, 1.0)
        density_factor = base_density * radius_scale  # 100 = gut sichtbar
        
        segments = max(16, min(64, int(atmos_radius * 2)))
        
        # Ring zeichnen: von radius bis atmos_radius
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            
            # Innerer Punkt (am Planetenrand) - volle Sichtbarkeit
            inner_x = x + math.cos(angle) * radius
            inner_y = y + math.sin(angle) * radius
            glColor4f(r, g, b, density_factor)
            glVertex2f(inner_x, inner_y)
            
            # Äußerer Punkt (am Atmosphärenrand) - transparent
            outer_x = x + math.cos(angle) * atmos_radius
            outer_y = y + math.sin(angle) * atmos_radius
            glColor4f(r, g, b, 0.0)
            glVertex2f(outer_x, outer_y)
        glEnd()
    
    def _draw_glow(self, x, y, radius, r, g, b, intensity):
        glow_radius = radius * 2.5
        
        # Automatische Skalierung mit Radius: Kleinere Planeten bekommen relativ stärkeren Glow
        radius_scale = max(0.5, min(2.0, radius / 50.0))
        intensity_factor = min(intensity / 1000.0, 1.0) * 0.5 * radius_scale
        
        segments = max(16, min(64, int(glow_radius * 2)))
        
        # Glow als Ring rendern (nicht vom Zentrum)
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            
            # Innerer Punkt (am Planetenrand)
            inner_x = x + math.cos(angle) * radius
            inner_y = y + math.sin(angle) * radius
            glColor4f(r, g, b, intensity_factor * 0.8)
            glVertex2f(inner_x, inner_y)
            
            # Äußerer Punkt (am Glow-Rand)
            outer_x = x + math.cos(angle) * glow_radius
            outer_y = y + math.sin(angle) * glow_radius
            glColor4f(r, g, b, 0.0)
            glVertex2f(outer_x, outer_y)
        glEnd()
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

        glDisable(GL_TEXTURE_2D)
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

    def _segment_intersects_rect(self, x0, y0, x1, y1, xmin, xmax, ymin, ymax):
        left = 1
        right = 2
        bottom = 4
        top = 8

        def outcode(x, y):
            code = 0
            if x < xmin:
                code |= left
            elif x > xmax:
                code |= right
            if y < ymin:
                code |= bottom
            elif y > ymax:
                code |= top
            return code

        c0 = outcode(x0, y0)
        c1 = outcode(x1, y1)

        while True:
            if (c0 | c1) == 0:
                return True
            if (c0 & c1) != 0:
                return False

            out = c0 if c0 != 0 else c1
            if out & top:
                if y1 == y0:
                    return False
                x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0)
                y = ymax
            elif out & bottom:
                if y1 == y0:
                    return False
                x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0)
                y = ymin
            elif out & right:
                if x1 == x0:
                    return False
                y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
                x = xmax
            else:
                if x1 == x0:
                    return False
                y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0)
                x = xmin

            if out == c0:
                x0, y0 = x, y
                c0 = outcode(x0, y0)
            else:
                x1, y1 = x, y
                c1 = outcode(x1, y1)

    def _clip_segment_to_rect(self, x0, y0, x1, y1, xmin, xmax, ymin, ymax):
        left = 1
        right = 2
        bottom = 4
        top = 8

        def outcode(x, y):
            code = 0
            if x < xmin:
                code |= left
            elif x > xmax:
                code |= right
            if y < ymin:
                code |= bottom
            elif y > ymax:
                code |= top
            return code

        c0 = outcode(x0, y0)
        c1 = outcode(x1, y1)

        while True:
            if (c0 | c1) == 0:
                return x0, y0, x1, y1
            if (c0 & c1) != 0:
                return None

            out = c0 if c0 != 0 else c1
            if out & top:
                if y1 == y0:
                    return None
                x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0)
                y = ymax
            elif out & bottom:
                if y1 == y0:
                    return None
                x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0)
                y = ymin
            elif out & right:
                if x1 == x0:
                    return None
                y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
                x = xmax
            else:
                if x1 == x0:
                    return None
                y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0)
                x = xmin

            if out == c0:
                x0, y0 = x, y
                c0 = outcode(x0, y0)
            else:
                x1, y1 = x, y
                c1 = outcode(x1, y1)

    def _visible_window_runs(self, screen_points, margin_px):
        n = len(screen_points)
        if n < 2:
            return []

        xmin = -margin_px
        xmax = self.width + margin_px
        ymin = -margin_px
        ymax = self.height + margin_px

        runs = []
        current = []
        for i in range(n - 1):
            x0, y0 = screen_points[i]
            x1, y1 = screen_points[i + 1]
            clipped = self._clip_segment_to_rect(
                x0, y0, x1, y1, xmin, xmax, ymin, ymax
            )
            if clipped is not None:
                cx0, cy0, cx1, cy1 = clipped
                if not current:
                    current.append((cx0, cy0))
                elif current[-1] != (cx0, cy0):
                    current.append((cx0, cy0))
                current.append((cx1, cy1))
            else:
                if len(current) >= 2:
                    runs.append(current)
                current = []

        if len(current) >= 2:
            runs.append(current)

        return runs

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

    def _prediction_line_cache_key(self, path_points, input_count, camera, anchor_world):
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
        max_scan = int(max(1, self.prediction_render_max_raw_scan))
        if raw_count <= max_scan:
            stats['raw_stride'] = 1
            stats['skipped_by_stride'] = 0
            return list(range(raw_count))

        stride = max(1, int(math.ceil(raw_count / max_scan)))
        indices = list(range(0, raw_count, stride))
        if not indices or indices[0] != 0:
            indices.insert(0, 0)
        if indices[-1] != raw_count - 1 and len(indices) < max_scan:
            indices.append(raw_count - 1)

        stats['raw_stride'] = stride
        stats['skipped_by_stride'] = max(0, raw_count - len(indices))
        return indices

    def _cap_runs_by_screen_length(self, runs, max_screen_length_px, stats):
        if max_screen_length_px is None:
            return runs
        try:
            remaining = float(max_screen_length_px)
        except Exception:
            return runs
        if remaining <= 0.0:
            stats['clipped_or_rejected'] = stats.get('clipped_or_rejected', 0) + sum(len(run) for run in runs)
            return []

        capped = []
        rejected = 0
        for run in runs:
            if len(run) < 2:
                rejected += len(run)
                continue
            current = [run[0]]
            for sx, sy in run[1:]:
                lx, ly = current[-1]
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
                rejected += max(0, len(run) - len(current))
                remaining = 0.0
                break
            if len(current) >= 2:
                capped.append(current)
            if remaining <= 0.0:
                break

        stats['clipped_or_rejected'] = stats.get('clipped_or_rejected', 0) + rejected
        return capped

    def _cap_runs_by_point_budget(self, runs, max_points, stats):
        max_points = max(2, int(max_points))
        total = sum(len(run) for run in runs)
        if total <= max_points:
            return runs

        capped = []
        remaining = max_points
        rejected = 0
        for run in runs:
            if remaining < 2:
                rejected += len(run)
                break
            if len(run) <= remaining:
                capped.append(run)
                remaining -= len(run)
                continue
            target = max(2, remaining)
            step = (len(run) - 1) / (target - 1)
            reduced = []
            last_idx = -1
            for i in range(target):
                idx = int(round(i * step))
                idx = max(0, min(len(run) - 1, idx))
                if idx != last_idx:
                    reduced.append(run[idx])
                    last_idx = idx
            if len(reduced) >= 2:
                capped.append(reduced)
            rejected += max(0, len(run) - len(reduced))
            remaining = 0
            break

        stats['clipped_or_rejected'] = stats.get('clipped_or_rejected', 0) + rejected
        return capped

    def draw_prediction(self, path_points, camera, anchor_world=None):

        input_count = self._points_count(path_points)
        stats = {
            'raw_in': int(input_count),
            'scanned': 0,
            'visible': 0,
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

        # Textur deaktivieren falls HUD sie aktiviert hat
        glDisable(GL_TEXTURE_2D)
        
        # blend aktivieren
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        prepare_t0 = time.perf_counter()
        ship_x, ship_y = self._point_xy(path_points[0])
        half_w = self.width * 0.5
        half_h = self.height * 0.5
        camera_frame_xy = self._frame_camera_xy(camera)

        # debug-ausgabe: schiff-welt-position und ersten predictor-punkt anzeigen
        try:
            pred0_x, pred0_y = self._point_xy(path_points[0])
            ship_world_x, ship_world_y = (float(anchor_world[0]), float(anchor_world[1])) if anchor_world is not None else (pred0_x, pred0_y)
            if self.debug_predictor:
                print(f"PRED_DBG_POS: ship=({ship_world_x:.6e},{ship_world_y:.6e}) predictor_first=({pred0_x:.6e},{pred0_y:.6e})")
        except Exception:
            pass

        # wenn das schiff identifiziert werden kann, dessen exakte position zum verankern verwenden.
        # dies zwingt den ersten gerenderten predictor-punkt dazu am live-schiff zu liegen.
        if anchor_world is not None:
            ship_x = float(anchor_world[0])
            ship_y = float(anchor_world[1])

        ship_screen = self._world_to_screen_xy(
            ship_x,
            ship_y,
            camera,
            camera_frame_xy=camera_frame_xy,
        )

        effective_tolerance = self._effective_sampling_tolerance(camera)
        effective_min_step = max(0.05, min(self.prediction_sampling_min_step_px, effective_tolerance * 0.6))
        effective_max_segment = self._effective_max_segment_step(camera)
        max_draw_points = max(2, min(int(self.prediction_sampling_max_points), int(self.prediction_render_max_draw_points)))

        anchor_first_run = True
        if input_count >= 2:
            n0x, n0y = self._point_xy(path_points[1])
            # If predictor points include timestamps, project using per-sample time
            sample_time = None
            try:
                if hasattr(path_points[1], '__len__') and len(path_points[1]) >= 3:
                    sample_time = float(path_points[1][2])
            except Exception:
                sample_time = None

            if sample_time is not None:
                next_screen = self._world_to_screen_xy_at_time(n0x, n0y, camera, sample_time)
            else:
                next_screen = self._world_to_screen_xy(
                    n0x,
                    n0y,
                    camera,
                    camera_frame_xy=camera_frame_xy,
                )
            margin = self.prediction_visibility_margin_px
            anchor_first_run = self._segment_intersects_rect(
                ship_screen[0],
                ship_screen[1],
                next_screen[0],
                next_screen[1],
                -margin,
                self.width + margin,
                -margin,
                self.height + margin,
            )

        cache_key = self._prediction_line_cache_key(path_points, input_count, camera, anchor_world)
        if cache_key == self._prediction_line_cache_key_value and self._prediction_line_cache_points is not None:
            sampled_runs = self._prediction_line_cache_points
            stats.update(dict(self._prediction_line_cache_stats))
            stats['raw_in'] = int(input_count)
            stats['prepare_ms'] = (time.perf_counter() - prepare_t0) * 1000.0
            stats['cache_hit'] = True
        else:
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
            )
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

        # If the ship is visible, force exact predictor start at ship with no offset.
        # If it is not visible, keep the guarded behavior to avoid false connectors.
        first_run = sampled_runs[0]
        ship_is_visible = self._is_on_screen(ship_screen[0], ship_screen[1], 0.0)
        if ship_is_visible and len(first_run) >= 1:
            first_run[0] = (float(ship_screen[0]), float(ship_screen[1]))
        elif anchor_first_run and len(first_run) >= 1:
            dx = first_run[0][0] - ship_screen[0]
            dy = first_run[0][1] - ship_screen[1]
            if dx * dx + dy * dy <= 1e-10:
                first_run[0] = (float(ship_screen[0]), float(ship_screen[1]))
            else:
                first_run.insert(0, (float(ship_screen[0]), float(ship_screen[1])))

        sampled_runs = self._cap_runs_by_point_budget(sampled_runs, max_draw_points, stats)
        self._prediction_line_cache_key_key_value = cache_key
        self._prediction_line_cache_points = sampled_runs
        self._prediction_line_cache_stats = dict(stats)
        draw_t0 = time.perf_counter()
        for run in sampled_runs:
            if len(run) < 2:
                continue
            self._draw_polyline(run, color=(1.0, 1.0, 1.0, 0.6), width=2.0)
        stats['draw_ms'] = (time.perf_counter() - draw_t0) * 1000.0
        stats['drawn'] = sum(len(run) for run in sampled_runs)
        self.debug_info['prediction_points_drawn'] = int(stats['drawn'])
        self._last_prediction_render_stats = stats

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
                                           stats=None):
        if stats is None:
            stats = {}
        half_w = self.width * 0.5
        half_h = self.height * 0.5
        camera_frame_xy = self._frame_camera_xy(camera)
        scale = float(camera.scale)

        # If an anchor_world is provided, compute the world-space delta
        # between the requested anchor and the predictor's original first
        # point, and translate all projected points by that delta. This
        # avoids a visual 'gap' when the predictor points were computed
        # from an earlier snapshot (async) while we anchor the first
        # point to the ship's current position.
        screen_points = []
        delta_world_x = 0.0
        delta_world_y = 0.0
        if anchor_world is not None and len(path_points) > 0:
            try:
                orig_px, orig_py = self._point_xy(path_points[0])
                delta_world_x = float(anchor_world[0]) - float(orig_px)
                delta_world_y = float(anchor_world[1]) - float(orig_py)
            except Exception:
                delta_world_x = 0.0
                delta_world_y = 0.0

        raw_count = len(path_points)
        indices = self._prediction_scan_indices(raw_count, stats)
        max_world_length = self.prediction_render_max_world_length
        try:
            max_world_length = None if max_world_length is None else max(0.0, float(max_world_length))
        except Exception:
            max_world_length = None

        xmin = -margin_px
        xmax = self.width + margin_px
        ymin = -margin_px
        ymax = self.height + margin_px
        prev_world = None
        prev_time = None
        world_accum = 0.0
        prev_screen_all = None
        prev_near_all = False
        pending_offscreen = None

        for i in indices:
            point = path_points[i]
            if i == 0 and anchor_world is not None:
                px = float(anchor_world[0])
                py = float(anchor_world[1])
            else:
                px, py = self._point_xy(point)
                if anchor_world is not None:
                    px = float(px) + delta_world_x
                    py = float(py) + delta_world_y

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
                sx, sy = self._world_to_screen_xy_at_time(px, py, camera, sample_time)
            else:
                frame_x, frame_y = self._frame_transform_xy(px, py)
                sx = half_w + (frame_x - camera_frame_xy[0]) * scale
                sy = half_h - (frame_y - camera_frame_xy[1]) * scale
            stats['scanned'] = stats.get('scanned', 0) + 1

            near_visible = self._is_on_screen(sx, sy, margin_px)
            if near_visible:
                stats['visible'] = stats.get('visible', 0) + 1

            current_screen = (sx, sy)
            if not screen_points:
                screen_points.append(current_screen)
            elif near_visible:
                if pending_offscreen is not None and screen_points[-1] != pending_offscreen:
                    screen_points.append(pending_offscreen)
                screen_points.append(current_screen)
                pending_offscreen = None
            else:
                crosses_view = False
                if prev_screen_all is not None:
                    crosses_view = self._segment_intersects_rect(
                        prev_screen_all[0],
                        prev_screen_all[1],
                        sx,
                        sy,
                        xmin,
                        xmax,
                        ymin,
                        ymax,
                    )
                if crosses_view:
                    if prev_screen_all is not None and screen_points[-1] != prev_screen_all:
                        screen_points.append(prev_screen_all)
                    screen_points.append(current_screen)
                    pending_offscreen = None
                elif prev_near_all:
                    screen_points.append(current_screen)
                    pending_offscreen = None
                else:
                    pending_offscreen = current_screen
                    stats['clipped_or_rejected'] = stats.get('clipped_or_rejected', 0) + 1

            prev_world = (px, py)
            prev_time = sample_time
            prev_screen_all = current_screen
            prev_near_all = near_visible
            if stop_after_point:
                stats['clipped_or_rejected'] = stats.get('clipped_or_rejected', 0) + max(0, raw_count - i - 1)
                break

        runs = self._visible_window_runs(screen_points, margin_px)
        if not runs:
            return []

        sampled_runs = []
        remaining_budget = max(2, int(max_points))

        for run in runs:
            if remaining_budget < 2:
                break

            run = self._densify_screen_run(run, max_segment_px)

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

            if len(sampled) > remaining_budget:
                target = max(2, remaining_budget)
                step = (len(sampled) - 1) / (target - 1)
                reduced = []
                last_idx = -1
                for i in range(target):
                    idx = int(round(i * step))
                    idx = max(0, min(len(sampled) - 1, idx))
                    if idx != last_idx:
                        reduced.append(sampled[idx])
                        last_idx = idx
                if reduced[0] != sampled[0]:
                    reduced.insert(0, sampled[0])
                if reduced[-1] != sampled[-1]:
                    reduced.append(sampled[-1])
                sampled = reduced

            if len(sampled) >= 2:
                sampled_runs.append(sampled)
                remaining_budget -= len(sampled)

        sampled_runs = self._cap_runs_by_screen_length(
            sampled_runs,
            self.prediction_render_max_screen_length_px,
            stats,
        )
        sampled_runs = self._cap_runs_by_point_budget(sampled_runs, max_points, stats)
        stats['drawn'] = sum(len(run) for run in sampled_runs)
        return sampled_runs

    def _is_visible(self, screen_pos, radius):
        x, y = screen_pos
        margin = radius + 100
        return (-margin < x < self.width + margin and 
                -margin < y < self.height + margin)
    
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
    
    def _blit_pygame_surface(self, surface, x, y):
        """Rendert eine pygame Surface an der angegebenen Position."""
        # Persistent HUD texture: reuse gl texture and update via glTexSubImage2D
        texture_data = pygame.image.tostring(surface, 'RGBA', True)
        width, height = surface.get_size()

        # Create or resize HUD texture
        if self._hud_texture is None or self._hud_texture_size != (width, height):
            if self._hud_texture is not None:
                try:
                    glDeleteTextures(1, [self._hud_texture])
                except Exception:
                    pass
            try:
                self._hud_texture = glGenTextures(1)
                self._hud_texture_size = (width, height)
                glBindTexture(GL_TEXTURE_2D, self._hud_texture)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
            except Exception:
                # Fallback: create one-shot texture
                tex = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, tex)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
                self._hud_texture = tex
                self._hud_texture_size = (width, height)
        else:
            glBindTexture(GL_TEXTURE_2D, self._hud_texture)
            try:
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
            except Exception:
                # Some drivers don't support TexSubImage with certain formats; fallback to TexImage
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

        # Textur rendern
        glEnable(GL_TEXTURE_2D)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x, y)
        glTexCoord2f(1, 0); glVertex2f(x + width, y)
        glTexCoord2f(1, 1); glVertex2f(x + width, y + height)
        glTexCoord2f(0, 1); glVertex2f(x, y + height)
        glEnd()
        glDisable(GL_TEXTURE_2D)
    
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
        hud_surface = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
        
        for i, text in enumerate(texts):
            text_surface = self.font_medium.render(text, True, (255, 255, 255))
            hud_surface.blit(text_surface, (0, i * line_height))
        
        # HUD in OpenGL rendern
        self._blit_pygame_surface(hud_surface, 10, self.height - hud_height - 10)
    
    def resize(self, width, height):

        self.width = width
        self.height = height
        glViewport(0, 0, width, height)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, width, 0, height)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Framebuffer neu erstellen wenn FXAA aktiviert
        if self.enable_fxaa:
            # Alten Framebuffer löschen
            if self.fbo:
                glDeleteFramebuffers(1, [self.fbo])
            if self.fbo_texture:
                glDeleteTextures(1, [self.fbo_texture])
            
            # Framebuffer neu erstellen
            self._init_fxaa()
        # Clear HUD and label texture caches (will be recreated lazily)
        try:
            for entry in list(self._label_texture_cache.values()):
                texid = entry[0]
                if texid:
                    try:
                        glDeleteTextures(1, [texid])
                    except Exception:
                        pass
        except Exception:
            pass
        self._label_texture_cache = {}
        if getattr(self, '_hud_texture', None):
            try:
                glDeleteTextures(1, [self._hud_texture])
            except Exception:
                pass
            self._hud_texture = None
            self._hud_texture_size = (0, 0)
        # Delete poly VBO so it can be recreated for the new context/size
        try:
            if getattr(self, '_poly_vbo', None):
                try:
                    glDeleteBuffers(1, [self._poly_vbo])
                except Exception:
                    pass
        except Exception:
            pass
        self._poly_vbo = None
        self._poly_vbo_size = 0
