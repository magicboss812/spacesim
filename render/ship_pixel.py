"""Das schiff im pixel-stil der koerper.

Dieselben zwei schritte wie `render/body_pixel.py`, nur mit drehung:

1. zellpass: die flaechen aus `ship/art.py::build_cells` gedreht und
   skaliert in einen kleinen puffer mit einer zelle je pixel
   (`ship_cell.{vert,frag}`), danach der putz-pass der koerper;
2. zusammenbau in VOLLER aufloesung (`ship_pixel.frag`) direkt ins ziel --
   das schiff kommt ohnehin erst nach dem FXAA-resolve.

Das zellraster haengt an der auf ganze pixel gerundeten schiffsmitte und
steht achsparallel zum schirm wie das der koerper; dreht das schiff, wird
die silhouette neu gerastert. `renderer.ship_pixel_style` waehlt eine von
fuenf auspraegungen, 0 ist die vektor-zeichnung. Regeln:
.claude/rules/body-art.md, "Pixel ship".
"""
import math

import moderngl

from render.body_pixel import _HUD_GROUND, _HUD_INK, _mix
from ship import art as ship_art
from ui import theme as ui_theme

# Groesster puffer in zellen; ein schiff, das mehr braucht, bekommt groessere
# zellen statt eines groesseren puffers.
_SHIP_CELL_CAP = 160


class ShipPixelMixin:

    def _ship_pixel_ready(self):
        return (int(self.ship_pixel_style) > 0
                and self._ship_cell_program is not None
                and self._ship_pixel_program is not None
                and self._ship_pixel_vao is not None
                and self._body_cell_clean_program is not None
                and self._body_cell_clean_vao is not None)

    def _ship_cell_size(self, length_px):
        """Zellkante in ganzen pixeln: die kleine koerper-zelle, solange das
        schiff mindestens `ship_pixel_min_cells` zellen lang bleibt; bei der
        zoom-schrumpfung werden die zellen mit kleiner, bis 1 px."""
        small = max(1, int(round(self.ui_px(float(self.body_pixel_size_small)))))
        fit = int(float(length_px) / max(1.0, float(self.ship_pixel_min_cells)))
        return max(1, min(small, fit))

    def _ship_cells(self):
        geo = self._ship_cells_cache
        if geo is not None and geo.accent == self.ship_accent_color:
            return geo
        self._release_ship_cell_buffers()
        geo = ship_art.build_cells(self.ship_accent_color)
        prog = self._ship_cell_program
        vaos = []
        for table, mode in ((geo.tris, moderngl.TRIANGLES), (geo.lines, moderngl.LINES)):
            if len(table) == 0:
                continue
            vbo = self.ctx.buffer(table.tobytes())
            vao = self.ctx.vertex_array(
                prog, [(vbo, '2f 3f 1f', 'a_pos', 'a_col', 'a_tag')])
            vaos.append((vbo, vao, mode, len(table)))
        self._ship_cell_vaos = vaos
        self._ship_cells_cache = geo
        return geo

    def _release_ship_cell_buffers(self):
        for vbo, vao, _mode, _n in self._ship_cell_vaos:
            for obj in (vao, vbo):
                try:
                    obj.release()
                except Exception:
                    pass
        self._ship_cell_vaos = []
        self._ship_cells_cache = None

    def _ensure_ship_cell_targets(self):
        if self._ship_cell_fbo is not None:
            return
        cap = (_SHIP_CELL_CAP, _SHIP_CELL_CAP)

        def pair():
            texs = (self.ctx.texture(cap, 4), self.ctx.texture(cap, 4))
            for t in texs:
                t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            return texs, self.ctx.framebuffer(color_attachments=list(texs))

        self._ship_cell_textures, self._ship_cell_fbo = pair()
        self._ship_clean_textures, self._ship_clean_fbo = pair()

    def _draw_ship_pixel(self, body, x, y, theta, rgb, dest_fbo):
        """Das schiff im pixel-stil. x/y top-down. True, wenn gezeichnet."""
        if not self._ship_pixel_ready():
            return False
        try:
            self._ensure_ship_cell_targets()
            geo = self._ship_cells()
        except Exception as exc:
            self.debug_info['ship_pixel_error'] = f"setup: {type(exc).__name__}: {exc}"
            self.ship_pixel_style = 0
            return False

        w, h = int(self.width), int(self.height)
        length_px = self._ship_length_px()
        k = self._ship_cell_size(length_px)
        scale = length_px / geo.length
        # Reichweite um die mitte: nase bis ende der fahne (x = 2 im SVG,
        # also 145 einheiten hinter der mitte), radiatoren +-59.
        reach = 147.0 * scale + 2.0 * k
        while k < 64 and 2.0 * reach / k + 4 > _SHIP_CELL_CAP:
            k += 1
        hx, hy = math.cos(theta), -math.sin(theta)
        oy_up = self._ortho_y(y)
        ox, oy = int(round(x)), int(round(oy_up))
        i0 = math.floor(-reach / k) - 1
        j0 = i0
        n = 2 * (-i0) + 1
        origin = (ox + i0 * k, oy + j0 * k)
        if (origin[0] > w or origin[1] > h
                or origin[0] + n * k < 0 or origin[1] + n * k < 0):
            return True

        light, emissive = self._body_light_dir(body, x, y)
        light3 = tuple(float(c) for c in light)
        style = int(self.ship_pixel_style)
        ctx = self.ctx
        try:
            fbo = self._ship_cell_fbo
            fbo.use()
            fbo.viewport = (0, 0, n, n)
            fbo.clear(0.0, 0.0, 0.0, 0.0, viewport=(0, 0, n, n))
            ctx.disable(moderngl.BLEND)
            prog = self._ship_cell_program
            prog['u_head'].value = (hx, hy)
            prog['u_scale'].value = float(scale)
            prog['u_center'].value = (float(ox - origin[0]), float(oy - origin[1]))
            prog['u_cell'].value = float(k)
            prog['u_grid'].value = (float(n), float(n))
            prog['u_light'].value = light3
            prog['u_light_exp'].value = float(self.body_light_exponent)
            prog['u_emissive'].value = float(emissive)
            prog['u_style'].value = style
            # Eine zelle breit, auch im hochaufgeloesten foto (dort sind die
            # ZELLEN k-fach, nicht die linie im zellraster).
            self._set_line_width(1.0 / float(getattr(self, 'capture_scale', 1.0) or 1.0))
            for _vbo, vao, mode, _count in self._ship_cell_vaos:
                vao.render(mode)

            clean = self._ship_clean_fbo
            clean.use()
            clean.viewport = (0, 0, n, n)
            cprog = self._body_cell_clean_program
            cprog['u_rect'].value = (-1.0, -1.0, 1.0, 1.0)
            cprog['u_grid'].value = (n, n)
            self._ship_cell_textures[0].use(location=0)
            self._ship_cell_textures[1].use(location=1)
            self._body_cell_clean_vao.render(moderngl.TRIANGLE_STRIP)
        except Exception as exc:
            self.debug_info['ship_pixel_error'] = f"cells: {type(exc).__name__}: {exc}"
            return False
        finally:
            ctx.enable(moderngl.BLEND)

        plume = self._ship_plume_intensity(body, getattr(self, '_frame_real_dt', 0.0))
        accent = ui_theme.rgba(self.ship_accent_color)[:3]
        hull_rgb = tuple(float(c) for c in rgb)
        rim = _mix(_mix((0.80, 0.84, 0.92), hull_rgb, 0.3), _HUD_INK, 0.5)
        ink = _mix(ui_theme.rgba(ui_theme.SCHEME[0])[:3], _HUD_INK, 0.35) \
            if style == 3 else _HUD_INK
        grout = tuple(c * 2.0 for c in _HUD_GROUND)

        x0, y0 = max(0, origin[0]), max(0, origin[1])
        x1, y1 = min(w, origin[0] + n * k), min(h, origin[1] + n * k)
        if x1 <= x0 or y1 <= y0:
            return True
        try:
            dest_fbo.use()
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA,
                              moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            prog = self._ship_pixel_program
            fw, fh = float(w), float(h)
            prog['u_rect'].value = (x0 / fw * 2.0 - 1.0, y0 / fh * 2.0 - 1.0,
                                    x1 / fw * 2.0 - 1.0, y1 / fh * 2.0 - 1.0)
            prog['u_origin'].value = (float(origin[0]), float(origin[1]))
            prog['u_grid'].value = (n, n)
            prog['u_k'].value = int(k)
            prog['u_style'].value = style
            prog['u_light'].value = light3
            prog['u_emissive'].value = float(emissive)
            prog['u_grout'].value = grout
            prog['u_rim'].value = rim
            prog['u_ink'].value = tuple(float(c) for c in ink)
            prog['u_accent'].value = tuple(float(c) for c in accent)
            prog['u_plume'].value = float(max(0.0, min(1.0, plume)))
            prog['u_center'].value = (float(ox), float(oy))
            prog['u_head'].value = (hx, hy)
            prog['u_scale'].value = float(scale)
            prog['u_alpha'].value = 1.0
            self._ship_clean_textures[0].use(location=0)
            self._ship_clean_textures[1].use(location=1)
            self._ship_pixel_vao.render(moderngl.TRIANGLE_STRIP)
        except Exception as exc:
            self.debug_info['ship_pixel_error'] = f"compose: {type(exc).__name__}: {exc}"
            return False
        finally:
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        return True
