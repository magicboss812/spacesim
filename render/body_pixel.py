"""Die koerper im pixel-stil des HUD.

Die zeichnung bleibt GEOMETRIE (`bodies/style.py`); nur ihr raster ist grob,
und zwar das raster des KOERPERS, nicht des schirms. Je koerper:

1. zellpass: scheibe und fuellungen in einen kleinen puffer mit einer zelle
   je pixel (`body_cell_*.frag`) -- deckend, je zelle eine farbe aus der
   rampe des koerpers, dazu halbton-groesse und regions-kennung;
2. zusammenbau in VOLLER aufloesung (`body_pixel.frag`): gefaste LED-zellen
   mit fuge, 1-px-striche zwischen regionen und am lichtrand, atmosphaere
   und halo als punktmatrix -- in eine eigene ebene in fenstergroesse;
3. die ebene kommt nach dem FXAA-resolve ungefiltert auf den schirm
   (`_end_body_layer`), sonst verwischte der kantenfilter die treppen.

Regeln und messwerte: .claude/rules/body-art.md, "Pixel bodies".
"""
import math
import zlib

import moderngl

from ui import theme as ui_theme


def _hex_rgb(value):
    rgba = ui_theme.rgba(value)
    return (float(rgba[0]), float(rgba[1]), float(rgba[2]))


_HUD_GROUND = _hex_rgb('#04070c')
_HUD_INK = _hex_rgb('#dceaf4')


def _mix(a, b, t):
    return tuple(float(x) + (float(y) - float(x)) * t for x, y in zip(a, b))


class BodyPixelMixin:
    """Zellpass, zusammenbau und ebene der pixel-koerper."""

    # ---- an / aus, zellgroesse ---------------------------------------

    def _body_pixel_ready(self):
        return (float(self.body_pixel_size) > 0.0
                and self._body_cell_disc_program is not None
                and self._body_cell_disc_vao is not None
                and self._body_cell_fill_program is not None
                and self._body_pixel_program is not None
                and self._body_pixel_vao is not None
                and self._body_layer_program is not None
                and self._body_layer_vao is not None)

    def _body_pixel_cell_px(self, radius_px):
        """Zellkante k in ganzen pixeln fuer einen koerper dieses radius.

        Zwei stufen auf EINEM raster: die grosse zelle ist das doppelte der
        kleinen, und die kleine ist die teilung des hintergrund-rasters.
        Ein kleiner koerper bekommt die kleine zelle, sonst bliebe von ihm
        ein klecks aus vier zellen."""
        big = max(2, int(round(self.ui_px(float(self.body_pixel_size)))))
        small = max(2, int(round(self.ui_px(float(self.body_pixel_size_small)))))
        if float(radius_px) < float(self.body_pixel_small_cells) * big:
            return min(small, big)
        return big

    def _body_facet_target_px(self, radius_px):
        """Facettenziel der detailleiter: im pixel-stil in ZELLEN, damit eine
        region immer viele zellen gross ist und nie zu brei zerfaellt."""
        target = max(1.0, float(self.body_vector_facet_px))
        if self._body_layer_active:
            cells = float(self.body_pixel_facet_cells)
            target = max(target, cells * self._body_pixel_cell_px(radius_px))
        return target

    @staticmethod
    def _body_pixel_seed(body):
        return zlib.crc32(str(getattr(body, 'name', '')).encode('utf-8')) & 0xFFFF

    # ---- puffer --------------------------------------------------------

    def _ensure_body_pixel_targets(self):
        w, h = int(self.width), int(self.height)
        if self._body_layer_size != (w, h) or self._body_layer_fbo is None:
            self._release_body_pixel_targets()
            tex = self.ctx.texture((w, h), 4)
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._body_layer_tex = tex
            self._body_layer_fbo = self.ctx.framebuffer(color_attachments=[tex])
            self._body_layer_size = (w, h)
            self._body_layer_dirty = (0, 0, w, h)
            # Der zellpuffer traegt hoechstens den schirm bei der KLEINSTEN
            # zelle (2 px) plus rand; danach wird nur ein ausschnitt benutzt.
            cap = (w // 2 + 8, h // 2 + 8)
            cells = self.ctx.texture(cap, 4)
            ids = self.ctx.texture(cap, 4)
            for t in (cells, ids):
                t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._body_cell_textures = (cells, ids)
            self._body_cell_fbo = self.ctx.framebuffer(color_attachments=[cells, ids])
            self._body_cell_cap = cap

    def _release_body_pixel_targets(self):
        for obj in [self._body_layer_fbo, self._body_layer_tex,
                    self._body_cell_fbo, *self._body_cell_textures]:
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self._body_layer_fbo = None
        self._body_layer_tex = None
        self._body_cell_fbo = None
        self._body_cell_textures = ()
        self._body_cell_cap = (0, 0)
        self._body_layer_size = None

    # ---- ebene ------------------------------------------------------------

    def _begin_body_layer(self):
        """Schaltet den pixel-stil fuer diesen frame ein. False = glatter
        alter zeichenweg (abgeschaltet oder ein shader fehlt)."""
        self._body_layer_active = False
        self._body_layer_rect = None
        self._deferred_icons = []
        if not self._body_pixel_ready():
            return False
        try:
            self._ensure_body_pixel_targets()
            # Nur das rechteck des letzten frames ist schmutzig.
            x0, y0, x1, y1 = self._body_layer_dirty or (0, 0, 0, 0)
            if x1 > x0 and y1 > y0:
                self._body_layer_fbo.clear(0.0, 0.0, 0.0, 0.0,
                                           viewport=(x0, y0, x1 - x0, y1 - y0))
            self._body_layer_dirty = None
        except Exception as exc:
            self.debug_info['body_layer_error'] = f"{type(exc).__name__}: {exc}"
            self._release_body_pixel_targets()
            return False
        self._body_layer_active = True
        return True

    def _end_body_layer(self, dest_fbo):
        """Bringt die ebene ungefiltert auf `dest_fbo`, danach die marken."""
        if not self._body_layer_active:
            return
        self._body_layer_active = False
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        dest_fbo.use()
        rect = self._body_layer_rect
        if rect is not None:
            w, h = float(self.width), float(self.height)
            x0, y0, x1, y1 = rect
            try:
                prog = self._body_layer_program
                prog['u_cell'].value = 1.0
                prog['u_height_px'].value = h
                prog['u_cells'].value = (int(w), int(h))
                prog['u_rect'].value = (x0 / w * 2.0 - 1.0, 1.0 - y1 / h * 2.0,
                                        x1 / w * 2.0 - 1.0, 1.0 - y0 / h * 2.0)
                self._body_layer_tex.use(location=0)
                self.ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
                self._body_layer_vao.render(moderngl.TRIANGLE_STRIP)
            except Exception as exc:
                self.debug_info['body_layer_error'] = f"{type(exc).__name__}: {exc}"
            finally:
                self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            # fuer den naechsten clear, in GL-koordinaten (von unten)
            hi = int(h)
            self._body_layer_dirty = (int(x0), hi - int(y1), int(x1), hi - int(y0))
        icons, self._deferred_icons = self._deferred_icons, []
        for args in icons:
            self._draw_body_icon(*args)

    # ---- ein koerper ------------------------------------------------------

    def _draw_body_pixel(self, body, x, y, radius_px, style_layers, fade,
                         light, emissive, rgb, atmos_rgb, has_atmos,
                         light_intensity):
        """Zellpass + zusammenbau eines koerpers. True, wenn gezeichnet."""
        k = self._body_pixel_cell_px(radius_px)
        w, h = int(self.width), int(self.height)
        # Die mitte auf ganze pixel: der koerper bewegt sich in 1-px-schritten
        # mit seiner bahnlinie, und das zellraster haengt an IHM.
        ox, oy = int(round(x)), int(round(y))
        is_star = bool(getattr(body, 'is_star', False))
        kind = 1 if is_star else (2 if getattr(body, 'is_gas', False) else 0)

        reach = float(radius_px) + 2.0 * k
        i0 = math.floor((max(0.0, ox - reach) - ox) / k)
        i1 = math.ceil((min(float(w), ox + reach) - ox) / k)
        j0 = math.floor((max(0.0, oy - reach) - oy) / k)
        j1 = math.ceil((min(float(h), oy + reach) - oy) / k)
        gw, gh = i1 - i0, j1 - j0
        if gw <= 0 or gh <= 0:
            return False
        cap = self._body_cell_cap
        if gw > cap[0] or gh > cap[1]:
            return False
        origin = (ox + i0 * k, oy + j0 * k)
        center_rel = (float(ox - origin[0]), float(oy - origin[1]))
        viewport = (float(gw * k), float(gh * k))
        seed = float(self._body_pixel_seed(body))

        ctx = self.ctx
        try:
            # --- 1. zellpass ----------------------------------------------
            fbo = self._body_cell_fbo
            fbo.use()
            fbo.viewport = (0, 0, gw, gh)
            fbo.clear(0.0, 0.0, 0.0, 0.0, viewport=(0, 0, gw, gh))
            ctx.disable(moderngl.BLEND)

            prog = self._body_cell_disc_program
            prog['u_center_px'].value = center_rel
            prog['u_outer_radius_px'].value = float(radius_px)
            prog['u_viewport'].value = viewport
            prog['u_base_color'].value = tuple(float(c) for c in rgb)
            prog['u_light'].value = tuple(float(c) for c in light)
            prog['u_light_exp'].value = float(self.body_light_exponent)
            prog['u_emissive'].value = float(emissive)
            prog['u_ground'].value = 0.0 if is_star else (1.0 if style_layers and fade >= 0.5 else 0.0)
            prog['u_cell'].value = float(k)
            prog['u_seed'].value = seed
            prog['u_dither_range'].value = (0.0, 1.0)
            self._body_cell_disc_vao.render(moderngl.TRIANGLE_STRIP)

            # KEINE ueberblendung im pixel-stil: zwei detailstufen (oder
            # zeichnung und grund) im bayer-wechsel ergaben je zelle eine
            # andere region, und jede zelle bekam ihren eigenen grenzstrich.
            # Die staerkere stufe gewinnt, die zeichnung kommt ab halber
            # einblendung ganz.
            prog = self._body_cell_fill_program
            prog['u_center_px'].value = center_rel
            prog['u_radius_px'].value = float(radius_px)
            prog['u_viewport'].value = viewport
            prog['u_light'].value = tuple(float(c) for c in light)
            prog['u_light_exp'].value = float(self.body_light_exponent)
            prog['u_emissive'].value = float(emissive)
            # Ein-/ausblenden geht ueber den dither-bereich, nicht ueber alpha.
            prog['u_fade'].value = 1.0
            prog['u_cell'].value = float(k)
            prog['u_seed'].value = seed
            prog['u_dither_range'].value = (0.0, 1.0)
            if style_layers and fade >= 0.5:
                entry = max(style_layers, key=lambda item: item[1])[0]
                vao = self._body_cell_vao(entry)
                if vao is not None:
                    vao.render(moderngl.TRIANGLES)
        except Exception as exc:
            self.debug_info['body_pixel_error'] = f"cells: {type(exc).__name__}: {exc}"
            ctx.enable(moderngl.BLEND)
            return False
        finally:
            ctx.enable(moderngl.BLEND)

        # --- 2. zusammenbau in voller aufloesung ------------------------
        if is_star and float(light_intensity) > 0.0:
            halo_r, halo_a, halo_d = 3.0, 0.85, float(self.body_pixel_halo_density) * 1.4
        else:
            glow = float(self.body_glow_alpha) * fade
            halo_r = 1.45 if glow > 0.0 else 0.0
            halo_a = min(1.0, glow * 4.5)
            halo_d = float(self.body_pixel_halo_density)
        outer = float(radius_px) * max(1.0, halo_r) + 2.0 * k
        x0 = max(0, int(math.floor(ox - outer)))
        y0 = max(0, int(math.floor(oy - outer)))
        x1 = min(w, int(math.ceil(ox + outer)))
        y1 = min(h, int(math.ceil(oy + outer)))
        if x1 <= x0 or y1 <= y0:
            return False

        # Der lichtrand des ANGEWAEHLTEN koerpers in HUD-cyan gehoert zur
        # auswahlmarke und geht mit ihr aus (selection_camera_test §8).
        selected = (body is getattr(self, 'selected_body', None)
                    and bool(getattr(self, 'selection_marker_enabled', True)))
        rim = (_mix(ui_theme.rgba(ui_theme.SCHEME[0])[:3], _HUD_INK, 0.25) if selected
               else _mix(tuple(min(1.0, c * 1.15) for c in rgb), _HUD_INK, 0.45))
        grout = _mix(tuple(c * 2.0 for c in _HUD_GROUND), rgb, 0.10)
        halo_rgb = _mix(rgb, _HUD_INK, 0.2)

        try:
            self._body_layer_fbo.use()
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA,
                              moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            prog = self._body_pixel_program
            fw, fh = float(w), float(h)
            prog['u_rect'].value = (x0 / fw * 2.0 - 1.0, 1.0 - y1 / fh * 2.0,
                                    x1 / fw * 2.0 - 1.0, 1.0 - y0 / fh * 2.0)
            prog['u_target'].value = (fw, fh)
            prog['u_origin'].value = (float(origin[0]), float(origin[1]))
            prog['u_grid'].value = (int(gw), int(gh))
            prog['u_k'].value = int(k)
            prog['u_center'].value = (float(ox), float(oy))
            prog['u_radius'].value = float(radius_px)
            prog['u_light'].value = tuple(float(c) for c in light)
            prog['u_emissive'].value = float(emissive)
            prog['u_kind'].value = int(kind)
            prog['u_seed'].value = seed
            prog['u_grout'].value = grout
            prog['u_rim'].value = rim
            prog['u_atmos'].value = (float(atmos_rgb[0]), float(atmos_rgb[1]),
                                     float(atmos_rgb[2]), 1.0 if has_atmos else 0.0)
            prog['u_halo'].value = (halo_rgb[0], halo_rgb[1], halo_rgb[2], float(halo_a))
            prog['u_halo_radius'].value = float(halo_r)
            prog['u_halo_density'].value = float(halo_d)
            cells, ids = self._body_cell_textures
            cells.use(location=0)
            ids.use(location=1)
            self._body_pixel_vao.render(moderngl.TRIANGLE_STRIP)
        except Exception as exc:
            self.debug_info['body_pixel_error'] = f"composite: {type(exc).__name__}: {exc}"
            return False
        finally:
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        rect = self._body_layer_rect
        if rect is not None:
            x0, y0 = min(x0, rect[0]), min(y0, rect[1])
            x1, y1 = max(x1, rect[2]), max(y1, rect[3])
        self._body_layer_rect = (x0, y0, x1, y1)
        return True

    def _body_cell_vao(self, entry):
        """VAO der fuellungen fuer den zellpass, einmal je zeichnung."""
        vao = entry.get('cell_vao')
        if vao is None and entry.get('tri_vbo') is not None:
            try:
                vao = self.ctx.vertex_array(
                    self._body_cell_fill_program,
                    [(entry['tri_vbo'], '2f 3f 3f 1f 1f',
                      'a_pos', 'a_nrm', 'a_col', 'a_alpha', 'a_dark')],
                )
            except Exception as exc:
                self.debug_info['body_pixel_error'] = f"vao: {exc}"
                vao = False
            entry['cell_vao'] = vao
        return vao or None
