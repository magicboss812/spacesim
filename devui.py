"""Dear ImGui entwickler-oberflaeche (moderngl-nativ).

Bewusst NUR fuer entwicklung und debugging: laufzeit-verstellung der werte aus
config.json, integrator-tuning, predictor-innereien, debug-flags. Die
spieler-oberflaeche (HUD) ist ein eigenes, gestaltetes system -- ImGui sieht
absichtlich nach werkzeugkasten aus und soll das auch bleiben.

Warum ein eigener renderer statt des mitgelieferten backends:
`imgui_bundle.python_backends.pygame_backend.PygameRenderer` erbt von
`ProgrammablePipelineRenderer`, und das zieht **PyOpenGL** herein. Dieses
projekt ist bewusst von PyOpenGL auf moderngl portiert (und die hiesige
PyOpenGL_accelerate-installation ist kaputt). Deshalb ist hier
* das zeichnen moderngl-nativ ueber denselben geteilten context, und
* die eingabe-uebersetzung von pygame nach imgui selbst geschrieben.

Damit bleibt der GL-zustand vollstaendig unter der kontrolle von moderngl.
"""

import ctypes

import moderngl
import numpy as np

try:
    from imgui_bundle import imgui
    IMGUI_AVAILABLE = True
except Exception as _exc:  # pragma: no cover - optionale abhaengigkeit
    imgui = None
    IMGUI_AVAILABLE = False
    _IMGUI_IMPORT_ERROR = _exc


VERTEX_SHADER = """
#version 330
uniform mat4 ProjMtx;
in vec2 Position;
in vec2 UV;
in vec4 Color;
out vec2 Frag_UV;
out vec4 Frag_Color;
void main() {
    Frag_UV = UV;
    Frag_Color = Color;
    gl_Position = ProjMtx * vec4(Position.xy, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
uniform sampler2D Texture;
in vec2 Frag_UV;
in vec4 Frag_Color;
out vec4 Out_Color;
void main() {
    Out_Color = Frag_Color * texture(Texture, Frag_UV.st);
}
"""


QUALITY_PRESETS = ("fast", "balanced", "accurate", "rk4")


class DevContext:
    """Sammelt die objekte, die die entwickler-panels verstellen duerfen.

    Bewusst ein schlichter halter statt globaler zugriffe: so ist an einer
    stelle sichtbar, was die oberflaeche anfassen kann.
    """

    __slots__ = ("world", "camera", "predictor", "renderer", "ship_control",
                 "ship", "frame_dt", "sim_step_s", "tick_rate", "notes")

    def __init__(self, world=None, camera=None, predictor=None, renderer=None,
                 ship_control=None, ship=None, frame_dt=0.0, sim_step_s=0.0,
                 tick_rate=60.0):
        self.world = world
        self.camera = camera
        self.predictor = predictor
        self.renderer = renderer
        self.ship_control = ship_control
        self.ship = ship
        self.frame_dt = frame_dt
        self.sim_step_s = sim_step_s
        self.tick_rate = tick_rate
        self.notes = []


def _fmt_si(value, unit="m"):
    """Kompakte SI-darstellung fuer die readouts."""
    try:
        value = float(value)
    except Exception:
        return str(value)
    a = abs(value)
    for limit, suffix, div in ((1e12, "T", 1e12), (1e9, "G", 1e9),
                               (1e6, "M", 1e6), (1e3, "k", 1e3)):
        if a >= limit:
            return f"{value / div:.3f} {suffix}{unit}"
    return f"{value:.3f} {unit}"


def _slider(label, obj, attr, lo, hi, fmt="%.3f", log=False, tooltip=None):
    """slider_float direkt auf ein attribut. Gibt True bei aenderung zurueck."""
    if obj is None or not hasattr(obj, attr):
        return False
    flags = imgui.SliderFlags_.logarithmic if log else 0
    changed, value = imgui.slider_float(label, float(getattr(obj, attr)), lo, hi, fmt, flags)
    if changed:
        setattr(obj, attr, value)
    if tooltip and imgui.is_item_hovered():
        imgui.set_tooltip(tooltip)
    return changed


def _checkbox(label, obj, attr, tooltip=None):
    if obj is None or not hasattr(obj, attr):
        return False
    changed, value = imgui.checkbox(label, bool(getattr(obj, attr)))
    if changed:
        setattr(obj, attr, value)
    if tooltip and imgui.is_item_hovered():
        imgui.set_tooltip(tooltip)
    return changed


def draw_dev_panels(c: "DevContext"):
    """Die eigentlichen panels. Rein werkzeug -- kein spieler-HUD."""
    imgui.set_next_window_size(imgui.ImVec2(460, 720), imgui.Cond_.first_use_ever)
    imgui.set_next_window_pos(imgui.ImVec2(20, 20), imgui.Cond_.first_use_ever)
    expanded, _ = imgui.begin("spacesim - dev tools")
    if not expanded:
        imgui.end()
        return

    # ImGui setzt die beschriftung RECHTS neben das widget. Ohne reservierten
    # platz laufen laengere labels aus dem fenster ("move speed (scre...").
    # Negativer wert = abstand vom rechten fensterrand.
    imgui.push_item_width(-190.0)

    io = imgui.get_io()
    imgui.text(f"{io.framerate:6.1f} FPS   ({1000.0 / max(io.framerate, 1e-3):5.2f} ms/frame)")
    if c.world is not None:
        imgui.text(f"sim time: {_fmt_si(getattr(c.world, 'time', 0.0), 's')}")

    # ---------------------------------------------------------- Simulation
    if imgui.collapsing_header("Simulation", imgui.TreeNodeFlags_.default_open):
        cam = c.camera
        if cam is not None:
            changed, value = imgui.slider_float(
                "sim_dt (zeitraffer, ACHTUNG: Spiel kann abstürzen!!!)", float(cam.sim_dt),
                float(cam.min_sim_dt), min(float(cam.max_sim_dt), 1e9),
                "%.1f", imgui.SliderFlags_.logarithmic,
            )
            if changed:
                cam.sim_dt = value
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Zeitraffer-regler. Die simulation rueckt zeitproportional vor,\n"
                    f"die rate ist sim_dt * {c.tick_rate:.0f} = "
                    f"{cam.sim_dt * c.tick_rate:,.0f} sim-s pro echtsekunde\n"
                    "-- unabhaengig von der bildrate."
                )
            imgui.text(f"rate: {_fmt_si(cam.sim_dt * c.tick_rate, 's')}/s wall")
        imgui.text(f"this frame: +{_fmt_si(c.sim_step_s, 's')} sim "
                   f"(over {c.frame_dt * 1000.0:.1f} ms real)")
        world = c.world
        if world is not None and hasattr(world, 'integrator_mode'):
            modes = ("rkn4", "verlet")
            current = modes.index(world.integrator_mode) if world.integrator_mode in modes else 0
            changed, idx = imgui.combo("world integrator", current, modes)
            if changed:
                world.integrator_mode = modes[idx]
                c.notes.append(f"world.integrator_mode = {modes[idx]}")
            _slider("pos tolerance", world, 'integrator_position_tolerance',
                    1e-3, 1e4, "%.4f", log=True)
            _slider("vel tolerance", world, 'integrator_velocity_tolerance',
                    1e-6, 1e2, "%.6f", log=True)

    # -------------------------------------------------------------- Camera
    if imgui.collapsing_header("Camera"):
        cam = c.camera
        if cam is not None:
            imgui.text(f"scale  {cam.scale:.4e} px/m")
            imgui.text(f"target {cam.target_scale:.4e} px/m")
            imgui.text(f"pos    ({cam.position.x:.4e}, {cam.position.y:.4e})")
            imgui.text(f"follow {getattr(cam.target, 'name', 'free')}")
            imgui.separator_text("feel")
            _slider("zoom smoothing", cam, 'zoom_smoothing', 1.0, 60.0, "%.1f",
                    tooltip="Hoeher = direkter. Die glaettung ist bildratenunabhaengig.")
            _slider("pan smoothing", cam, 'pan_smoothing', 1.0, 60.0, "%.1f")
            _slider("zoom factor/notch", cam, 'zoom_factor', 1.05, 3.0, "%.2f")
            _slider("move speed (screens/s)", cam, 'move_speed', 0.05, 5.0, "%.2f")
            _checkbox("zoom to cursor", cam, 'zoom_to_cursor')
            _checkbox("pan inertia", cam, 'pan_inertia_enabled')
            _slider("inertia damping", cam, 'pan_inertia_damping', 0.5, 20.0, "%.1f")
            if imgui.button("recentre on target"):
                cam.follow_offset.clear()
            imgui.same_line()
            if imgui.button("snap (no easing)"):
                cam.snap_to_targets()

    # ----------------------------------------------------------- Predictor
    if imgui.collapsing_header("Predictor"):
        p = c.predictor
        if p is not None:
            current = QUALITY_PRESETS.index(getattr(p, '_quality', 'balanced')) \
                if getattr(p, '_quality', None) in QUALITY_PRESETS else 1
            changed, idx = imgui.combo("quality", current, QUALITY_PRESETS)
            if changed:
                try:
                    p.set_integrator_quality(QUALITY_PRESETS[idx])
                    p._quality = QUALITY_PRESETS[idx]
                    p.reset()
                    c.notes.append(f"predictor quality = {QUALITY_PRESETS[idx]}")
                except Exception as exc:
                    c.notes.append(f"quality failed: {exc}")

            imgui.text(f"mode: {getattr(p, 'integrator_mode', '?')}")
            imgui.text(f"points: {len(p.get_points())} / {p.num_points}")
            imgui.text(f"horizon: {_fmt_si(p.length)}")
            imgui.text(f"spacing: {_fmt_si(p.precision)}")
            imgui.text(f"last compute: {getattr(p, 'last_compute_ms', 0.0):.1f} ms")

            if hasattr(p, 'get_async_status'):
                st = p.get_async_status()
                imgui.text(f"async: {'on' if st['enabled'] else 'off'}  "
                           f"pending={st['pending']}  swapped={st['swapped_jobs']}")

            imgui.separator_text("tolerances")
            _slider("rkn rtol", p, 'rkn_rtol', 1e-12, 1e-3, "%.2e", log=True)
            _slider("rkn atol pos", p, 'rkn_atol_pos', 1e-3, 1e4, "%.3f", log=True)
            _slider("rkn atol vel", p, 'rkn_atol_vel', 1e-8, 1e0, "%.2e", log=True)
            _slider("rkn max dt", p, 'rkn_max_dt', 1.0, 30000.0, "%.0f", log=True)

            imgui.separator_text("behaviour")
            _checkbox("auto precision from zoom", p, 'auto_precision_from_zoom')
            _checkbox("apsis markers (compute)", p, 'apsis_markers_enabled')
            _checkbox("time-dependent bodies", p, 'use_time_dependent_bodies')
            if imgui.button("reset predictor"):
                p.reset()

    # ------------------------------------------------------------ Renderer
    if imgui.collapsing_header("Renderer"):
        r = c.renderer
        if r is not None:
            imgui.text(f"viewport: {r.width} x {r.height}")
            imgui.text(f"ui_scale: {r.ui_scale:.3f}")
            changed, value = imgui.slider_float(
                "ui scale (user)", float(r.ui_scale_user), 0.5, 3.0, "%.2f")
            if changed:
                r.set_ui_scale_user(value)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Multiplikativ auf die automatische, aus der "
                                  "fensterhoehe abgeleitete skala.")

            imgui.separator_text("visuals")
            _checkbox("FXAA", r, 'enable_fxaa')
            _checkbox("apsis markers (draw)", r, 'show_apsis_markers')
            _checkbox("reference trails", r, 'reference_trajectories_enabled')
            _checkbox("prediction bypasses FXAA", r, 'prediction_bypass_fxaa')
            _slider("apsis marker px", r, 'apsis_marker_radius_px', 1.0, 20.0, "%.1f")
            _slider("body icon px", r, 'body_icon_radius_px', 1.0, 20.0, "%.1f")
            _slider("velocity vector px", r, 'ship_velocity_vector_length_px',
                    10.0, 300.0, "%.0f")

            imgui.separator_text("prediction sampling")
            _slider("tolerance px", r, 'prediction_sampling_tolerance_px',
                    0.01, 10.0, "%.3f", log=True)
            _slider("max segment px", r, 'prediction_sampling_max_segment_px',
                    0.5, 40.0, "%.1f")
            imgui.text(f"drawn: {r.debug_info['prediction_points_drawn']}"
                       f" / {r.debug_info['prediction_points_in']}")
            imgui.text(f"bodies: {r.debug_info['bodies_rendered']} rendered, "
                       f"{r.debug_info['bodies_culled']} culled")

    # --------------------------------------------------------------- Ship
    if imgui.collapsing_header("Ship"):
        sc, ship = c.ship_control, c.ship
        if ship is not None:
            imgui.text(f"pos ({ship.position.x:.4e}, {ship.position.y:.4e})")
            imgui.text(f"vel ({ship.velocity.x:.2f}, {ship.velocity.y:.2f}) m/s")
            imgui.text(f"speed {ship.velocity.magnitude():.2f} m/s")
            imgui.text(f"theta {getattr(ship, 'theta', 0.0):.4f} rad")
        if sc is not None:
            imgui.text(f"snap: {sc.snap_mode or 'off'}")
            _slider("rotation speed", sc, 'rotation_speed', 0.1, 20.0, "%.2f")
            _slider("thrust acc (m/s2)", sc, 'thrust_acc', 1.0, 5000.0, "%.0f", log=True)

    # -------------------------------------------------------------- Debug
    if imgui.collapsing_header("Debug flags"):
        _checkbox("world.integrator_debug", c.world, 'integrator_debug')
        _checkbox("predictor.debug", c.predictor, 'debug')
        _checkbox("predictor.debug_moving_sources", c.predictor, 'debug_moving_sources')
        _checkbox("renderer.debug_predictor", c.renderer, 'debug_predictor')
        _checkbox("renderer.debug_frame", c.renderer, 'debug_frame')
        _checkbox("renderer.render_benchmark_debug", c.renderer, 'render_benchmark_debug')

    if c.notes:
        imgui.separator()
        for note in c.notes[-4:]:
            imgui.text_wrapped(note)

    imgui.pop_item_width()
    imgui.end()


class ImguiLayer:
    """ImGui-overlay auf dem geteilten moderngl-context."""

    def __init__(self, ctx, width, height, enabled=False):
        self.ctx = ctx
        self.width = int(width)
        self.height = int(height)
        self.available = IMGUI_AVAILABLE
        # Sichtbarkeit der panels. Standardmaessig aus -- das werkzeug soll
        # dem spiel nicht im weg stehen.
        self.visible = bool(enabled)

        self._program = None
        self._vbo = None
        self._ibo = None
        self._vao = None
        self._textures = {}          # imgui tex_id -> moderngl.Texture
        self._next_tex_id = 1
        self._frame_open = False

        if not self.available:
            print(f"DEVUI: imgui-bundle nicht verfuegbar ({_IMGUI_IMPORT_ERROR}) "
                  f"-- entwickler-oberflaeche deaktiviert")
            return

        imgui.create_context()
        io = imgui.get_io()
        io.display_size = imgui.ImVec2(float(self.width), float(self.height))
        # imgui 1.92 verwaltet texturen selbst und erwartet, dass das backend
        # want_create/want_updates/want_destroy bedient (siehe _sync_textures).
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        # renderer_has_vtx_offset wird BEWUSST nicht gesetzt: moderngls
        # VertexArray.render() kennt kein base_vertex, also darf imgui die
        # draw-listen nicht mit einem vtx_offset != 0 ausliefern. Ohne das
        # flag teilt imgui die listen selbst auf.
        # Kein imgui.ini neben der exe ablegen.
        io.set_ini_filename("")

        self._apply_style()
        self._init_gpu()
        self._key_map = self._build_key_map()

    # ------------------------------------------------------------------
    # Aufbau
    # ------------------------------------------------------------------

    def _apply_style(self):
        style = imgui.get_style()
        style.window_rounding = 6.0
        style.frame_rounding = 4.0
        style.grab_rounding = 4.0
        style.scrollbar_rounding = 4.0
        style.window_border_size = 1.0
        style.window_padding = imgui.ImVec2(10.0, 10.0)
        style.item_spacing = imgui.ImVec2(8.0, 6.0)
        imgui.style_colors_dark()

    def _init_gpu(self):
        self._program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )
        # Dynamische puffer; wachsen ueber orphan(), die VAO-bindung bleibt
        # dabei gueltig (gleiches muster wie _poly_vbo in rendering.py).
        self._vbo = self.ctx.buffer(reserve=65536, dynamic=True)
        self._ibo = self.ctx.buffer(reserve=65536, dynamic=True)
        # ImDrawVert: vec2 pos, vec2 uv, uint32 rgba -> '2f 2f 4f1' (20 byte)
        self._vao = self.ctx.vertex_array(
            self._program,
            [(self._vbo, "2f 2f 4f1", "Position", "UV", "Color")],
            index_buffer=self._ibo,
            index_element_size=imgui.INDEX_SIZE,
        )

    def _build_key_map(self):
        import pygame
        return {
            pygame.K_LEFT: imgui.Key.left_arrow,
            pygame.K_RIGHT: imgui.Key.right_arrow,
            pygame.K_UP: imgui.Key.up_arrow,
            pygame.K_DOWN: imgui.Key.down_arrow,
            pygame.K_PAGEUP: imgui.Key.page_up,
            pygame.K_PAGEDOWN: imgui.Key.page_down,
            pygame.K_HOME: imgui.Key.home,
            pygame.K_END: imgui.Key.end,
            pygame.K_INSERT: imgui.Key.insert,
            pygame.K_DELETE: imgui.Key.delete,
            pygame.K_BACKSPACE: imgui.Key.backspace,
            pygame.K_SPACE: imgui.Key.space,
            pygame.K_RETURN: imgui.Key.enter,
            pygame.K_ESCAPE: imgui.Key.escape,
            pygame.K_TAB: imgui.Key.tab,
            pygame.K_LCTRL: imgui.Key.left_ctrl,
            pygame.K_RCTRL: imgui.Key.right_ctrl,
            pygame.K_LSHIFT: imgui.Key.left_shift,
            pygame.K_RSHIFT: imgui.Key.right_shift,
            pygame.K_LALT: imgui.Key.left_alt,
            pygame.K_RALT: imgui.Key.right_alt,
        }

    # ------------------------------------------------------------------
    # Eingabe-vorfahrt
    # ------------------------------------------------------------------

    @property
    def wants_mouse(self):
        if not self.available or not self.visible:
            return False
        return bool(imgui.get_io().want_capture_mouse)

    @property
    def wants_keyboard(self):
        if not self.available or not self.visible:
            return False
        return bool(imgui.get_io().want_capture_keyboard)

    def process_event(self, event):
        """pygame-ereignis nach imgui uebersetzen."""
        if not self.available:
            return
        import pygame

        io = imgui.get_io()

        if event.type == pygame.MOUSEMOTION:
            io.add_mouse_pos_event(float(event.pos[0]), float(event.pos[1]))
        elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            # pygame: 1=links 2=mitte 3=rechts -> imgui: 0=links 1=rechts 2=mitte
            button = {1: 0, 2: 2, 3: 1}.get(event.button)
            if button is not None:
                io.add_mouse_button_event(button, event.type == pygame.MOUSEBUTTONDOWN)
        elif event.type == pygame.MOUSEWHEEL:
            io.add_mouse_wheel_event(float(event.x), float(event.y))
        elif event.type == pygame.TEXTINPUT:
            for ch in event.text:
                io.add_input_character(ord(ch))
        elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
            key = self._key_map.get(event.key)
            if key is not None:
                io.add_key_event(key, event.type == pygame.KEYDOWN)
        elif event.type == pygame.WINDOWSIZECHANGED:
            self.resize(event.x, event.y)

    def resize(self, width, height):
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        if self.available:
            imgui.get_io().display_size = imgui.ImVec2(float(self.width), float(self.height))

    def toggle(self):
        self.visible = not self.visible

    # ------------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------------

    def new_frame(self, dt):
        if not self.available:
            return
        io = imgui.get_io()
        io.display_size = imgui.ImVec2(float(self.width), float(self.height))
        io.delta_time = max(1e-4, float(dt))
        imgui.new_frame()
        self._frame_open = True

    def render(self):
        """imgui-frame abschliessen und ueber moderngl zeichnen."""
        if not self.available or not self._frame_open:
            return
        imgui.render()
        self._frame_open = False
        self._draw(imgui.get_draw_data())

    # ------------------------------------------------------------------
    # Texturen (imgui 1.92 backend-protokoll)
    # ------------------------------------------------------------------

    def _sync_textures(self):
        for tex in imgui.get_platform_io().textures:
            status = tex.status
            if status == imgui.ImTextureStatus.want_create:
                self._create_texture(tex)
            elif status == imgui.ImTextureStatus.want_updates:
                self._update_texture(tex)
            elif status == imgui.ImTextureStatus.want_destroy:
                self._destroy_texture(tex)

    def _create_texture(self, tex):
        pixels = tex.get_pixels_array()
        texture = self.ctx.texture((tex.width, tex.height), 4, np.ascontiguousarray(pixels).tobytes())
        # Bilineare filterung ist fuer die gebackenen linien-texturen des
        # atlas vorausgesetzt (sonst franst antialiasing aus).
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex_id = self._next_tex_id
        self._next_tex_id += 1
        self._textures[tex_id] = texture
        tex.set_tex_id(tex_id)
        tex.set_status(imgui.ImTextureStatus.ok)

    def _update_texture(self, tex):
        texture = self._textures.get(tex.get_tex_id())
        if texture is None:
            self._create_texture(tex)
            return
        # Teil-updates existieren, aber die atlas-texturen sind klein genug,
        # dass ein vollstaendiger re-upload einfacher und schnell genug ist.
        pixels = tex.get_pixels_array()
        texture.write(np.ascontiguousarray(pixels).tobytes())
        tex.set_status(imgui.ImTextureStatus.ok)

    def _destroy_texture(self, tex):
        texture = self._textures.pop(tex.get_tex_id(), None)
        if texture is not None:
            try:
                texture.release()
            except Exception:
                pass
        tex.set_tex_id(0)
        tex.set_status(imgui.ImTextureStatus.destroyed)

    # ------------------------------------------------------------------
    # Zeichnen
    # ------------------------------------------------------------------

    def _draw(self, draw_data):
        if draw_data is None:
            return

        self._sync_textures()

        fb_width = int(self.width)
        fb_height = int(self.height)
        if fb_width <= 0 or fb_height <= 0:
            return

        ctx = self.ctx
        # GL-zustand sichern. moderngl cacht zustand; scissor und viewport
        # muessen anschliessend exakt so zurueck, wie der haupt-renderer sie
        # erwartet, sonst verschwindet nach dem ersten ImGui-frame das bild.
        prev_scissor = ctx.scissor
        prev_viewport = ctx.viewport

        ctx.enable(ctx.BLEND)
        ctx.blend_func = ctx.SRC_ALPHA, ctx.ONE_MINUS_SRC_ALPHA
        ctx.disable(ctx.DEPTH_TEST | ctx.CULL_FACE)
        ctx.viewport = (0, 0, fb_width, fb_height)

        # Orthographische projektion, y nach unten (imgui-konvention).
        left, right = 0.0, float(fb_width)
        top, bottom = 0.0, float(fb_height)
        projection = np.array([
            2.0 / (right - left), 0.0, 0.0, 0.0,
            0.0, 2.0 / (top - bottom), 0.0, 0.0,
            0.0, 0.0, -1.0, 0.0,
            -1.0, 1.0, 0.0, 1.0,
        ], dtype='f4')
        self._program["ProjMtx"].write(projection.tobytes())
        self._program["Texture"].value = 0

        for cmd_list in draw_data.cmd_lists:
            n_vtx = cmd_list.vtx_buffer.size()
            n_idx = cmd_list.idx_buffer.size()
            if n_vtx == 0 or n_idx == 0:
                continue

            vtx_bytes = ctypes.string_at(
                cmd_list.vtx_buffer.data_address(), n_vtx * imgui.VERTEX_SIZE
            )
            idx_bytes = ctypes.string_at(
                cmd_list.idx_buffer.data_address(), n_idx * imgui.INDEX_SIZE
            )

            if len(vtx_bytes) > self._vbo.size:
                self._vbo.orphan(len(vtx_bytes))
            if len(idx_bytes) > self._ibo.size:
                self._ibo.orphan(len(idx_bytes))
            self._vbo.write(vtx_bytes)
            self._ibo.write(idx_bytes)

            for command in cmd_list.cmd_buffer:
                if command.elem_count == 0:
                    continue

                texture = self._textures.get(command.tex_ref.get_tex_id())
                if texture is None:
                    continue
                texture.use(location=0)

                x0, y0, x1, y1 = command.clip_rect
                sx = int(max(0.0, x0))
                sy = int(max(0.0, y0))
                sw = int(min(float(fb_width), x1) - sx)
                sh = int(min(float(fb_height), y1) - sy)
                if sw <= 0 or sh <= 0:
                    continue
                # moderngl-scissor rechnet von UNTEN links, imgui von oben.
                ctx.scissor = (sx, fb_height - sy - sh, sw, sh)

                self._vao.render(
                    mode=ctx.TRIANGLES,
                    vertices=command.elem_count,
                    first=command.idx_offset,
                )

        # Zustand zurueckgeben.
        ctx.scissor = prev_scissor
        ctx.viewport = prev_viewport

    def build(self, ctxobj):
        """Zeichnet die entwickler-panels. `ctxobj` ist ein DevContext."""
        if not self.available or not self.visible:
            return
        draw_dev_panels(ctxobj)

    def shutdown(self):
        if not self.available:
            return
        for texture in self._textures.values():
            try:
                texture.release()
            except Exception:
                pass
        self._textures.clear()
        for obj in (self._vao, self._vbo, self._ibo, self._program):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass
