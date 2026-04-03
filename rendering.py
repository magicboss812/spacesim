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
            'bodies_rendered': 0
        }
    
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
    
    def render(self, bodies, camera, prediction_points=None):

        self.debug_info['bodies_rendered'] = 0
        
        if self.enable_fxaa and self.fbo:
            # In Framebuffer rendern (nur Planeten)
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
            glViewport(0, 0, self.width, self.height)
        
        glClear(GL_COLOR_BUFFER_BIT)
        
        for body in bodies:
            self._draw_body(body, camera)
        
        if self.enable_fxaa and self.fbo:
            # Zurück zum Standard-Framebuffer
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
        
        if prediction_points:
            self.draw_prediction(prediction_points, camera)
        self._render_hud(camera)
        pygame.display.flip()
    
    def _draw_body(self, body, camera):

        screen_pos = camera.world_to_screen(body.position)
        radius = max(3, int(body.radius * camera.scale))  # Mindestens 3 Pixel
        
        self.debug_info['bodies_rendered'] += 1
        r, g, b = body.color[0] / 255.0, body.color[1] / 255.0, body.color[2] / 255.0
        x, y = float(screen_pos[0]), float(screen_pos[1])
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
    def draw_prediction(self, path_points, camera):

        if not path_points:
            return

        # Textur deaktivieren falls HUD sie aktiviert hat
        glDisable(GL_TEXTURE_2D)
        
        # Ensure blend is enabled
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glColor4f(1.0, 1.0, 1.0, 0.6)
        glLineWidth(2.0)

        glBegin(GL_LINE_STRIP)

        for point in path_points:
            screen = camera.world_to_screen(point)
            glVertex2f(float(screen[0]), float(screen[1]))

        glEnd()

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
    
    def _render_hud(self, camera):
        # HUD-Texte vorbereiten
        texts = [
            f"Scale: {camera.scale:.2e} px/m",
            f"Position: ({camera.position.x:.2e}, {camera.position.y:.2e})",
            f"Target: {camera.target.name if camera.target else 'None'}",
            f"Bodies rendered: {self.debug_info['bodies_rendered']}",
            f"FXAA: {'ON' if self.enable_fxaa else 'OFF'}",
            "[WASD] Move | [F] Unfollow | [Scroll] Zoom"
            
        ]
        
        # Pygame Surface für HUD erstellen
        hud_surface = pygame.Surface((300, 150), pygame.SRCALPHA)
        
        for i, text in enumerate(texts):
            text_surface = self.font_medium.render(text, True, (255, 255, 255))
            hud_surface.blit(text_surface, (0, i * 22))
        
        # HUD in OpenGL rendern
        self._blit_pygame_surface(hud_surface, 10, self.height - 160)
    
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
