"""Die GL-geraeteschicht des Renderers.

War teil der 5900-zeiligen `Renderer`-klasse in `rendering.py`. Als MIXIN
herausgeloest, nicht als eigenes objekt: die methoden greifen auf dutzende
`self._*`-felder zu, die `Renderer.__init__` anlegt, und eine echte
komposition haette hunderte zugriffe umgeschrieben -- ein grosses risiko fuer
eine rein strukturelle aenderung.
"""
import os
import time

import moderngl
import numpy as np
import pygame


class GLDeviceMixin:
    """Context, FXAA-ziele, present und resize -- die geraeteschicht.

    Liegt in `runtime/`, nicht in `render/`: das hier redet mit dem FENSTER
    (viewport, puffergroessen, swap), nicht mit der szene. Der FXAA-resolve
    gehoert dazu, weil er auf denselben ziel-texturen sitzt."""

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

    def present(self):
        """Fuehrt den buffer-swap aus und schreibt die swap-zeit in die timings.

        Von der hauptschleife aufzurufen, NACHDEM alle overlays gezeichnet sind.

        `frame_ms` bleibt dabei stehen: es ist die dauer von render() SELBST.
        Frueher wurde es hier auf "render-start bis nach dem swap" gesetzt,
        und weil `rend_calc` daraus als `frame_ms - swap` gebildet wird, lief
        alles, was zwischen render() und present() gezeichnet wird -- vor
        allem das spieler-HUD (`ui_root.render()`, gemessen ~8 ms median) --
        stillschweigend unter "render calc". Das ist die haelfte der zahl,
        und sie stand an der falschen stelle. Die luecke heisst jetzt
        `overlay_ms` und wird getrennt ausgewiesen.
        """
        swap_t0 = time.perf_counter()
        pygame.display.flip()
        swap_ms = (time.perf_counter() - swap_t0) * 1000.0
        timings = self.last_frame_timings
        if isinstance(timings, dict):
            timings['swap_or_present_ms'] = swap_ms
            end = getattr(self, '_render_end', None)
            if end is not None:
                overlay_ms = (swap_t0 - end) * 1000.0
                # Wer render() ohne present() aufruft (die GL-tests tun das),
                # hinterlaesst einen alten `_render_end` -- dann waere die
                # differenz unsinnig gross oder negativ.
                timings['overlay_ms'] = overlay_ms if overlay_ms >= 0.0 else 0.0

    def resize(self, width, height):

        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
        # u_viewport haengt an der fenstergroesse -- der zustandscache waere
        # sonst genau ueber diesen wert veraltet.
        self._invalidate_gl_state_cache()
        # WICHTIG: moderngl erkennt die groesse von ctx.screen nur EINMAL beim
        # anlegen des contexts. Nach einem resize meldet ctx.screen.size noch
        # die alte fenstergroesse -- und jedes ctx.screen.use() stellt daraus
        # viewport UND scissor wieder her. Ohne die explizite neuzuweisung
        # unten klemmt der scissor nach dem FXAA-pass (render() ruft dort
        # ctx.screen.use()) alles nachfolgende -- predictor-linie, schiff, HUD
        # -- auf das alte fenster-rechteck: beim maximieren ist dann nur noch
        # ein ausschnitt des spiels sichtbar.
        # ctx.screen.scissor = None hilft NICHT: das setzt den scissor auf die
        # (weiterhin veraltete) eigengroesse des framebuffers zurueck.
        try:
            self.ctx.screen.viewport = (0, 0, width, height)
            self.ctx.screen.scissor = (0, 0, width, height)
        except Exception as exc:
            print(f"RENDERER WARNING: screen viewport/scissor resize failed ({exc})")

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
        # UI-skala an die neue fensterhöhe anpassen. Nur bei echter änderung
        # die fonts neu aufbauen -- _rebuild_fonts leert die text-caches
        # ohnehin, und beim reinen breiten-resize wäre das verschwendet.
        if self._recompute_ui_scale():
            self._rebuild_fonts()

        # Clear HUD and label texture caches (will be recreated lazily)
        try:
            self._clear_text_caches()
        except Exception:
            pass
        if getattr(self, '_hud_texture', None):
            try:
                self._hud_texture.release()
            except Exception:
                pass
            self._hud_texture = None
            self._hud_texture_size = (0, 0)
        # HUD-memoization invalidieren: textur und viewport haben sich geändert.
        self._hud_cache_key = None
