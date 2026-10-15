"""Das schiff im pixel-stil der koerper.

Dieselben zwei schritte wie `render/body_pixel.py`, nur mit drehung:

1. zellpass: die flaechen aus `ship/art.py::build_cells` gedreht und
   skaliert in einen kleinen puffer mit einer zelle je pixel
   (`ship_cell.{vert,frag}`), danach der putz-pass der koerper;
2. zusammenbau in VOLLER aufloesung (`ship_pixel.frag`) direkt ins ziel --
   das schiff kommt ohnehin erst nach dem FXAA-resolve.

Das zellraster haengt an der auf ganze pixel gerundeten schiffsmitte und
steht achsparallel zum schirm wie das der koerper; dreht das schiff, wird
die silhouette neu gerastert. `renderer.ship_pixel_style` waehlt "halbton"
oder "sprite"; "vector" ist die vektor-zeichnung. Regeln:
.claude/rules/body-art.md, "Pixel ship".
"""
import math

import moderngl

from render.body_pixel import _HUD_GROUND, _HUD_INK, _mix
from ship import art as ship_art
from ship import builds as ship_builds
from ship import plume as ship_plume

# Groesster puffer in zellen; ein schiff, das mehr braucht, bekommt groessere
# zellen statt eines groesseren puffers.
_SHIP_CELL_CAP = 160

# Zeichenweise -> u_style im shader. Andere werte (auch "vector") zeichnen
# die vektor-grafik.
SHIP_PIXEL_STYLES = {"halbton": 2, "sprite": 4}


class ShipPixelMixin:

    def _ship_pixel_ready(self):
        return (str(self.ship_pixel_style) in SHIP_PIXEL_STYLES
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
        key, shapes, _accent = self._ship_shapes()
        geo = self._ship_cells_cache
        if geo is not None and geo.key == key:
            return geo
        self._release_ship_cell_buffers()
        geo = ship_art.build_cells(shapes, key)
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
        heat = self.ctx.texture(cap, 2, dtype='f2')
        # LINEAR fuer den weichzeichner (zwei texel je abgriff); der
        # zusammenbau liest die hitze mit texelFetch, ungefiltert.
        heat.filter = (moderngl.LINEAR, moderngl.LINEAR)
        heat.repeat_x = heat.repeat_y = False
        self._ship_heat_tex = heat
        self._ship_heat_fbo = self.ctx.framebuffer(color_attachments=[heat])
        glow = []
        for _ in range(2):
            t = self.ctx.texture(cap, 1, dtype='f2')
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
            glow.append(t)
        self._ship_glow_tex = tuple(glow)
        self._ship_glow_fbo = tuple(self.ctx.framebuffer(color_attachments=[t]) for t in glow)

    def _ship_glow_pass(self, sigma_cells, n):
        """Die hitze in zwei richtungen weichzeichnen: hitze -> a -> b."""
        prog = self._ship_glow_program
        prog['u_rect'].value = (-1.0, -1.0, 1.0, 1.0)
        prog['u_grid'].value = (n, n)
        prog['u_sigma'].value = float(sigma_cells)
        src = self._ship_heat_tex
        for i, d in enumerate(((1, 0), (0, 1))):
            fbo = self._ship_glow_fbo[i]
            fbo.use()
            fbo.viewport = (0, 0, n, n)
            prog['u_dir'].value = d
            src.use(location=0)
            self._ship_glow_vao.render(moderngl.TRIANGLE_STRIP)
            src = self._ship_glow_tex[i]

    def _ship_glow_on(self):
        return (self._ship_glow_vao is not None
                and float(getattr(self, 'ship_plume_glow', 0.0)) > 0.0)

    def _ship_plume_step(self, body, build, head):
        """Die partikel einen echtzeit-schritt weiter; im foto (capture)
        steht die fahne still, sie zeigt den zustand des fensters."""
        if not bool(getattr(self, 'ship_plume_particles', True)) \
                or self._ship_plume_program is None:
            return None
        if self._ship_plume is None:
            self._ship_plume = ship_plume.PlumeParticles()
        sim = self._ship_plume
        capture = float(getattr(self, 'capture_scale', 1.0) or 1.0) != 1.0
        dt = 0.0 if capture else float(getattr(self, '_frame_real_dt', 0.0))
        sim.step(dt, self._ship_thrust_target(body),
                 max(0.0, min(1.0, float(self.ship_plume_idle))),
                 head, ship_builds.NOZZLES.get(build, ()),
                 float(self.ship_plume_rate), float(self.ship_plume_speed),
                 float(self.ship_plume_life_s), float(self.ship_plume_spread))
        return sim

    def _ship_heat_pass(self, sim, scale, center, k, n):
        """Partikel als punkte additiv in den hitze-puffer (zellraster)."""
        fbo = self._ship_heat_fbo
        fbo.use()
        fbo.viewport = (0, 0, n, n)
        fbo.clear(0.0, 0.0, 0.0, 0.0, viewport=(0, 0, n, n))
        if sim is None:
            return
        pts = sim.render_points()
        if len(pts) == 0:
            return
        ctx = self.ctx
        need = pts.nbytes
        if self._ship_plume_vbo is None or self._ship_plume_vbo.size < need:
            for obj in (self._ship_plume_vao, self._ship_plume_vbo):
                if obj is not None:
                    obj.release()
            self._ship_plume_vbo = ctx.buffer(reserve=max(need, (ship_plume.MAX_PARTICLES + 64) * 20))
            self._ship_plume_vao = ctx.vertex_array(
                self._ship_plume_program,
                [(self._ship_plume_vbo, '2f 3f', 'a_pos', 'a_attr')])
        self._ship_plume_vbo.write(pts.tobytes())
        prog = self._ship_plume_program
        prog['u_scale'].value = float(scale)
        prog['u_center'].value = (float(center[0]), float(center[1]))
        prog['u_cell'].value = float(k)
        prog['u_grid'].value = (float(n), float(n))
        ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        try:
            self._ship_plume_vao.render(moderngl.POINTS, vertices=len(pts))
        finally:
            ctx.disable(moderngl.PROGRAM_POINT_SIZE)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    def _draw_ship_pixel(self, body, x, y, theta, rgb, dest_fbo):
        """Das schiff im pixel-stil. x/y top-down. True, wenn gezeichnet."""
        if not self._ship_pixel_ready():
            return False
        try:
            self._ensure_ship_cell_targets()
            geo = self._ship_cells()
        except Exception as exc:
            self.debug_info['ship_pixel_error'] = f"setup: {type(exc).__name__}: {exc}"
            self.ship_pixel_style = "vector"
            return False

        w, h = int(self.width), int(self.height)
        length_px = self._ship_length_px()
        k = self._ship_cell_size(length_px)
        scale = length_px / geo.length
        hx, hy = math.cos(theta), -math.sin(theta)
        build = self._ship_build_name()
        sim = self._ship_plume_step(body, build, (hx, hy))
        # Reichweite um die mitte: nase (115) bzw. so weit, wie die fahne
        # gerade reicht (gedeckelt); jede bauform bleibt in |y| <= 60.
        span = 147.0 if sim is None else min(320.0, max(147.0, sim.extent()))
        # Der schein erst, wenn das triebwerk laeuft: im leerlauf ist er
        # unsichtbar und kostete zwei weichzeichner-paesse und ein groesseres
        # raster. Ueber 1.5 sigma hinaus bleibt er unter der ersten stufe.
        glow = self._ship_glow_on() and sim is not None and sim.spool > 0.12
        glow_r = float(self.ship_plume_glow_radius) if glow else 0.0
        reach = (span + 1.5 * glow_r) * scale + 2.0 * k
        while k < 64 and 2.0 * reach / k + 4 > _SHIP_CELL_CAP:
            k += 1
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
        style = SHIP_PIXEL_STYLES[str(self.ship_pixel_style)]
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
            accent = self._ship_shapes()[2]
            lights = sim.light_sources(8) if sim is not None else []
            packed = [tuple(float(v) for v in L) for L in lights]
            packed += [(0.0, 0.0, 0.0, 1.0)] * (8 - len(packed))
            prog['u_plights'].value = packed
            prog['u_plight_n'].value = len(lights)
            prog['u_glow'].value = tuple(float(c) for c in _mix(accent, (1.0, 0.86, 0.45), 0.5))
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

            self._ship_heat_pass(sim, scale, (ox - origin[0], oy - origin[1]), k, n)
            glow_gain = 0.0
            if glow:
                ctx.disable(moderngl.BLEND)
                self._ship_glow_pass(max(0.5, glow_r * scale / k), n)
                # Mit dem hochlaufen einblenden, nicht an einer schwelle
                # anschalten: smoothstep(0.12, 0.55) des spool.
                t = min(1.0, max(0.0, (sim.spool - 0.12) / 0.43))
                glow_gain = float(self.ship_plume_glow) * t * t * (3.0 - 2.0 * t)
            else:
                fbo = self._ship_glow_fbo[1]
                fbo.use()
                fbo.clear(0.0, 0.0, 0.0, 0.0, viewport=(0, 0, n, n))
        except Exception as exc:
            self.debug_info['ship_pixel_error'] = f"cells: {type(exc).__name__}: {exc}"
            return False
        finally:
            ctx.enable(moderngl.BLEND)

        accent = self._ship_shapes()[2]
        hull_rgb = tuple(float(c) for c in rgb)
        rim = _mix(_mix((0.80, 0.84, 0.92), hull_rgb, 0.3), _HUD_INK, 0.5)
        ink = _HUD_INK
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
            prog['u_alpha'].value = 1.0
            self._ship_clean_textures[0].use(location=0)
            self._ship_clean_textures[1].use(location=1)
            self._ship_heat_tex.use(location=2)
            self._ship_glow_tex[1].use(location=3)
            prog['u_glow_gain'].value = glow_gain
            self._ship_pixel_vao.render(moderngl.TRIANGLE_STRIP)
        except Exception as exc:
            self.debug_info['ship_pixel_error'] = f"compose: {type(exc).__name__}: {exc}"
            return False
        finally:
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        return True
