"""
OpenGL-Renderer für die Weltraumsimulation.
Verwendet pygame für Fensterverwaltung und HUD, OpenGL für Rendering.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math


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

        # Opt-in predictor debug: when True, prints small samples of predictor
        # points (screen and reconstructed world coords) to the console.
        self.debug_predictor = False

        # Principia-like visual sampling controls: keep line-strip rendering,
        # but adapt point density to screen-space curvature/error.
        self.prediction_sampling_tolerance_px = 1.5
        self.prediction_sampling_min_step_px = 0.35
        self.prediction_sampling_max_points = 5000
        # Allow very fine screen-space tolerance when zoomed in.
        # Lowering this enables more detail at extreme zoom levels.
        self.prediction_sampling_min_tolerance_px = 0.005
        self.prediction_sampling_max_tolerance_px = 0.25
        self.prediction_sampling_max_segment_px = 4.0
        self.prediction_sampling_reference_scale = 1e-6
        self.prediction_visibility_margin_px = 128.0
    
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
    
    def render(self, bodies, camera, prediction_points=None, predictor=None):

        self.debug_info['bodies_rendered'] = 0
        self.debug_info['prediction_points_in'] = 0
        self.debug_info['prediction_points_drawn'] = 0
        
        # If FXAA is enabled, render non-ship bodies into the FBO and
        # apply FXAA. Ships are rendered afterwards directly to the main
        # framebuffer so the predictor (also rendered to the main buffer)
        # and the ship marker share the exact same pixel-space coordinates.
        ship_body = next((b for b in bodies if getattr(b, 'is_ship', False)), None)

        if self.enable_fxaa and self.fbo:
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
            glViewport(0, 0, self.width, self.height)

        glClear(GL_COLOR_BUFFER_BIT)

        # Render all non-ship bodies first (they may be FXAA-processed).
        for body in bodies:
            if getattr(body, 'is_ship', False):
                continue
            self._draw_body(body, camera)

        if self.enable_fxaa and self.fbo:
            # Zurück zum Standard-Framebuffer and apply FXAA post-process
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0, 0, self.width, self.height)
            self._apply_fxaa()
        
        # Restore projection matrix for trajectory rendering
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Re-enable blending for trajectory
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Compute ship anchor once (we computed ship_body earlier)
        ship_anchor_world = (float(ship_body.position.x), float(ship_body.position.y)) if ship_body is not None else None

        if self._points_count(prediction_points) > 0:
            self.draw_prediction(prediction_points, camera, anchor_world=ship_anchor_world)

        # Draw the ship marker on the main framebuffer so it visually
        # matches the predictor start point exactly.
        if ship_body is not None:
            self._draw_body(ship_body, camera)

        self._render_hud(camera, predictor)
        pygame.display.flip()
    
    def _draw_body(self, body, camera):

        screen_pos = camera.world_to_screen(body.position)
        radius = max(3, int(body.radius * camera.scale))  # Mindestens 3 Pixel
        
        self.debug_info['bodies_rendered'] += 1
        r, g, b = body.color[0] / 255.0, body.color[1] / 255.0, body.color[2] / 255.0
        x, y = float(screen_pos[0]), float(screen_pos[1])

        if body.is_ship:
            self._draw_ship_arrow(body, x, y, r, g, b)
            self._draw_body_label(body.name, screen_pos, 12)
            return

        if body.light_intensity > 0:
            self._draw_glow(x, y, radius, r, g, b, body.light_intensity)
        
        if hasattr(body, 'atmosphere_color'):
            r1, g1, b1 = body.atmosphere_color[0] / 255.0, body.atmosphere_color[1] / 255.0, body.atmosphere_color[2] / 255.0
        else:
            r1, g1, b1 = r, g, b
        if body.has_atmosphere and body.atmos_density > 0:
            self._draw_atmosphere(x, y, radius, r1, g1, b1, body.atmos_density)
        
        glColor4f(r, g, b, 1.0)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(x, y)  # Zentrum
        
        segments = max(16, min(64, radius * 2))
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            px = x + math.cos(angle) * radius
            py = y + math.sin(angle) * radius
            glVertex2f(px, py)
        glEnd()
        
        if radius > 5:
            self._draw_body_label(body.name, screen_pos, radius)

    def _draw_ship_arrow(self, body, x, y, r, g, b):
        # Draw in fixed screen pixels so ship size stays constant while zooming.
        arrow_length = 18.0
        arrow_half_width = 7.0
        tail_offset = 6.0

        theta = float(getattr(body, 'theta', 0.0))

        # Match camera.world_to_screen() y-inversion to keep visual heading correct.
        hx = math.cos(theta)
        hy = -math.sin(theta)
        nx = -hy
        ny = hx

        # Adjust origin so the triangle's centroid is located at (x, y).
        # The centroid of the triangle formed by nose and tail corners lies
        # offset along the heading by (arrow_length - 2*tail_offset)/3 in
        # screen pixels. Move the local origin back by that amount so the
        # ship's world position corresponds to the visual center of the arrow.
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
        # Debug: draw small markers and emit one-line info comparing
        # the triangle centroid with the provided screen position.
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

    def draw_prediction(self, path_points, camera, anchor_world=None):

        input_count = self._points_count(path_points)
        if input_count == 0:
            self.debug_info['prediction_points_in'] = 0
            self.debug_info['prediction_points_drawn'] = 0
            return

        # Textur deaktivieren falls HUD sie aktiviert hat
        glDisable(GL_TEXTURE_2D)
        
        # Ensure blend is enabled
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glColor4f(1.0, 1.0, 1.0, 0.6)
        glLineWidth(2.0)

        ship_x, ship_y = self._point_xy(path_points[0])
        half_w = self.width * 0.5
        half_h = self.height * 0.5

        # Debug output: show ship world position and predictor's first point
        try:
            pred0_x, pred0_y = self._point_xy(path_points[0])
            ship_world_x, ship_world_y = (float(anchor_world[0]), float(anchor_world[1])) if anchor_world is not None else (pred0_x, pred0_y)
            if self.debug_predictor:
                print(f"PRED_DBG_POS: ship=({ship_world_x:.6e},{ship_world_y:.6e}) predictor_first=({pred0_x:.6e},{pred0_y:.6e})")
        except Exception:
            pass

        # If we can identify the ship body, use that exact position for anchoring.
        # This forces the first rendered predictor point to be at the live ship.
        if anchor_world is not None:
            ship_x = float(anchor_world[0])
            ship_y = float(anchor_world[1])

        ship_screen = (
            half_w + (ship_x - float(camera.position.x)) * float(camera.scale),
            half_h - (ship_y - float(camera.position.y)) * float(camera.scale),
        )

        effective_tolerance = self._effective_sampling_tolerance(camera)
        effective_min_step = max(0.05, min(self.prediction_sampling_min_step_px, effective_tolerance * 0.6))
        effective_max_segment = self._effective_max_segment_step(camera)

        anchor_first_run = True
        if input_count >= 2:
            n0x, n0y = self._point_xy(path_points[1])
            next_screen = (
                half_w + (n0x - float(camera.position.x)) * float(camera.scale),
                half_h - (n0y - float(camera.position.y)) * float(camera.scale),
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

        sampled_runs = self._adaptive_prediction_screen_points(
            path_points,
            camera,
            tolerance_px=effective_tolerance,
            min_step_px=effective_min_step,
            max_segment_px=effective_max_segment,
            max_points=self.prediction_sampling_max_points,
            margin_px=self.prediction_visibility_margin_px,
            anchor_world=anchor_world,
        )

        self.debug_info['prediction_points_in'] = input_count
        self.debug_info['prediction_points_drawn'] = sum(len(run) for run in sampled_runs)

        # Store small sample for debugging and optionally print it.
        try:
            if sampled_runs and len(sampled_runs[0]) > 0:
                sample_n = min(5, len(sampled_runs[0]))
                screen_samples = [sampled_runs[0][i] for i in range(sample_n)]
                world_samples = []
                for sx, sy in screen_samples:
                    wx = float(camera.position.x) + (sx - half_w) / float(camera.scale)
                    wy = float(camera.position.y) - (sy - half_h) / float(camera.scale)
                    world_samples.append((wx, wy))
                self.debug_info['prediction_sample_screen'] = screen_samples
                self.debug_info['prediction_sample_world'] = world_samples
                if self.debug_predictor:
                    print('PRED_DBG: in=', input_count, 'drawn=', self.debug_info['prediction_points_drawn'])
                    print('PRED_DBG: screen_samples=', screen_samples)
                    print('PRED_DBG: world_samples=', world_samples)
        except Exception:
            pass

        if len(sampled_runs) == 0:
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

        for run in sampled_runs:
            if len(run) < 2:
                continue
            glBegin(GL_LINE_STRIP)
            for sx, sy in run:
                glVertex2f(float(sx), float(sy))
            glEnd()

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
                                           anchor_world=None):
        half_w = self.width * 0.5
        half_h = self.height * 0.5
        cam_x = float(camera.position.x)
        cam_y = float(camera.position.y)
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

        for i, point in enumerate(path_points):
            if i == 0 and anchor_world is not None:
                px = float(anchor_world[0])
                py = float(anchor_world[1])
            else:
                px, py = self._point_xy(point)
                if anchor_world is not None:
                    px = float(px) + delta_world_x
                    py = float(py) + delta_world_y

            sx = half_w + (px - cam_x) * scale
            sy = half_h - (py - cam_y) * scale
            screen_points.append((sx, sy))

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

        return sampled_runs

    def _is_visible(self, screen_pos, radius):
        x, y = screen_pos
        margin = radius + 100
        return (-margin < x < self.width + margin and 
                -margin < y < self.height + margin)
    
    def _draw_body_label(self, name, screen_pos, radius):

        # Label mit pygame rendern
        name_surface = self.font_small.render(name, True, (255, 255, 255))
        label_x = screen_pos[0] + radius + 2
        label_y = screen_pos[1] - 8
        
        # Surface zu OpenGL-Textur konvertieren
        self._blit_pygame_surface(name_surface, label_x, label_y)
    
    def _blit_pygame_surface(self, surface, x, y):
        """Rendert eine pygame Surface an der angegebenen Position."""
        # Textur-Daten extrahieren
        texture_data = pygame.image.tostring(surface, 'RGBA', True)
        width, height = surface.get_size()
        
        # OpenGL Textur erstellen
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 
                     0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
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
        glDeleteTextures([texture_id])
    
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

        texts.append("[WASD] Move | [F] Unfollow | [Scroll] Zoom")
        
        # Pygame Surface für HUD erstellen
        line_height = 22
        hud_width = 520
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
