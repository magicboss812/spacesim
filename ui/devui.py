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
import math

import moderngl
import numpy as np

from render import background
from bodies import icon as body_icon

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


# Die serien der zeitmessung. Reihenfolge = zeilen im ringpuffer; sie ist
# teil des dateiformats des puffers und darf nicht umsortiert werden, ohne
# TimingHistory.push() mitzuziehen.
#
# Es sind GENAU die vier groessen, die test.py in der `TIMING:`-zeile
# ausgibt, aus derselben quelle gelesen -- graph und ausgabe koennen sich
# damit nicht widersprechen. `frame` ist die fuenfte, aber keine gezeichnete:
# sie ist nur der bezug (budget-strich, textzeile).
TIMING_SERIES = ("pred_compute", "pred_draw", "rend_calc", "rend_draw",
                 "frame", "ui_calc")
TIMING_PLOTTED = ("pred_compute", "pred_draw", "rend_calc", "ui_calc",
                  "rend_draw")
_TIMING_INDEX = {name: i for i, name in enumerate(TIMING_SERIES)}

# Wie schnell der achsen-spitzenwert zerfaellt. Nur eine hysterese gegen das
# flackern der achse: ohne sie springt die sprosse bei einem messwert, der um
# eine leiter-grenze zittert, im wechsel zwischen zwei hoehen. Mit 0.5 s ist
# die achse nach einem ausreisser in unter einer sekunde wieder unten.
TIMING_AXIS_DECAY_TAU_S = 0.5


def _nice_ceiling(value):
    """Naechstgroessere sprosse der leiter 1 / 2 / 5 je dekade.

    Der achsen-maximalwert MUSS gerastert sein. Eine achse, die dem
    momentanen maximum stufenlos folgt, haelt die kurve optisch immer gleich
    hoch -- man sieht dann jede aenderung der form und keine einzige der
    groesse, was bei einer zeitmessung genau das falsche herum ist.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(v) or v <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(v))
    decade = 10.0 ** exponent
    mantissa = v / decade
    # log10 ist nicht exakt: fuer v = 1000 kann der exponent als 2.9999...
    # herauskommen und die mantisse damit als 10.0.
    if mantissa >= 10.0:
        mantissa /= 10.0
        decade *= 10.0
    for step in (1.0, 2.0, 5.0):
        if mantissa <= step * (1.0 + 1e-9):
            return step * decade
    return 10.0 * decade


def _ms(value):
    """Robuste ms-zahl. Fehlende/kaputte werte werden zu 0, nicht zu NaN.

    Ein NaN in einem einzigen frame vergiftet sonst mittelwert UND achse des
    ganzen fensters, und zwar still -- der graph ist dann leer statt falsch.
    Der `type(...) is float`-pfad haelt den normalfall auf zwei vergleichen;
    diese funktion laeuft fuenfmal je frame.
    """
    if type(value) is float:
        return value if -1e9 < value < 1e9 else 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return v if -1e9 < v < 1e9 else 0.0


class TimingHistory:
    """Ringpuffer der per-frame zeitmessung, direkt fuer imgui.plot_lines.

    Ein zusammenhaengendes `(serien, kapazitaet)`-float32-feld; jede zeile ist
    eine serie und damit eine gueltige eingabe fuer `plot_lines`, ohne kopie.
    Geschrieben wird ueber eine schreibmarke, NICHT ueber np.roll oder eine
    liste mit append: das abtasten laeuft in jedem frame -- auch wenn das
    panel zu ist -- und darf im bildbudget nicht auftauchen. Gemessen liegt
    `push()` bei ~1-2 us, also unter 0.04 % eines 5.6-ms-frames.

    `window()` liefert das feld zusammen mit dem offset der AELTESTEN probe,
    weil imgui `values[(i + offset) % n]` liest. Zeigt der offset auf die
    neueste probe, laeuft der graph rueckwaerts -- das sieht plausibel aus
    und ist trotzdem falsch.
    """

    __slots__ = ("_data", "_rows", "_capacity", "_cursor", "_count",
                 "_peaks", "paused", "shared_scale", "manual_max",
                 "graph_height")

    def __init__(self, capacity=240):
        self._capacity = max(2, int(capacity))
        self._data = np.zeros((len(TIMING_SERIES), self._capacity), dtype='f4')
        self._rows = tuple(self._data[i] for i in range(len(TIMING_SERIES)))
        self._cursor = 0
        self._count = 0
        self._peaks = [0.0] * len(TIMING_SERIES)
        # Anzeige-einstellungen des panels. Sie leben hier, weil das panel
        # selbst eine reine funktion ist und keinen zustand halten kann.
        self.paused = False
        self.shared_scale = False
        self.manual_max = 0.0        # 0 = automatisch
        self.graph_height = 58.0

    # ------------------------------------------------------------- schreiben

    def push(self, pred_compute, pred_draw, rend_calc, rend_draw, frame,
             ui_calc=0.0):
        """Eine probe. Heisser pfad -- keine allokation, kein dict, kein try.

        `ui_calc` steht hinten, obwohl es im panel zwischen rend_calc und
        rend_draw gezeichnet wird: die zeilen-reihenfolge ist TIMING_SERIES,
        die zeichen-reihenfolge TIMING_PLOTTED, und das anhaengen haelt die
        vorhandenen zeilen-indizes stabil.
        """
        if self.paused:
            return
        i = self._cursor
        rows = self._rows
        rows[0][i] = pred_compute
        rows[1][i] = pred_draw
        rows[2][i] = rend_calc
        rows[3][i] = rend_draw
        rows[4][i] = frame
        rows[5][i] = ui_calc
        i += 1
        if i >= self._capacity:
            i = 0
        self._cursor = i
        if self._count < self._capacity:
            self._count += 1

    def reset(self):
        self._data[:, :] = 0.0
        self._cursor = 0
        self._count = 0
        for i in range(len(self._peaks)):
            self._peaks[i] = 0.0

    def resize(self, capacity):
        """Laenge umstellen und dabei die JUENGSTEN proben behalten."""
        capacity = max(2, int(capacity))
        if capacity == self._capacity:
            return
        keep = min(self._count, capacity)
        data = np.zeros((len(TIMING_SERIES), capacity), dtype='f4')
        if keep:
            for i, name in enumerate(TIMING_SERIES):
                data[i, :keep] = self.series(name)[-keep:]
        self._data = data
        self._rows = tuple(data[i] for i in range(len(TIMING_SERIES)))
        self._capacity = capacity
        self._count = keep
        self._cursor = keep % capacity

    # --------------------------------------------------------------- lesen

    @property
    def capacity(self):
        return self._capacity

    @property
    def count(self):
        return self._count

    def window(self, name):
        """`(feld, offset)` fuer plot_lines. Offset = aelteste probe."""
        row = self._rows[_TIMING_INDEX[name]]
        if self._count < self._capacity:
            return row[:self._count], 0
        return row, self._cursor

    def series(self, name):
        """Kopie in chronologischer reihenfolge (alt -> neu). Nur fuer tests
        und ausgaben -- der zeichenweg benutzt window()."""
        values, offset = self.window(name)
        if offset == 0:
            return values.copy()
        return np.concatenate((values[offset:], values[:offset]))

    def stats(self, name):
        """`(cur, avg, max)` ueber den GEFUELLTEN teil des puffers.

        Der ungefuellte rest ist null und wuerde den mittelwert eines gerade
        zurueckgesetzten puffers sonst nach unten ziehen.
        """
        values, offset = self.window(name)
        if values.size == 0:
            return (0.0, 0.0, 0.0)
        # offset zeigt auf die aelteste probe, also liegt die juengste davor.
        # Bei offset 0 greift die negative indizierung auf das ende -- in
        # beiden faellen (voll und teilgefuellt) das richtige element.
        return (float(values[offset - 1]), float(values.mean()),
                float(values.max()))

    def peak(self, name, dt=0.0):
        """Zerfallender spitzenwert der serie. Nicht gerastert.

        Je frame HOECHSTENS EINMAL je serie aufrufen -- der aufruf schreibt
        den zerfall fort. Deshalb ruft der zeichenweg das hier einmal und
        leitet einzel- wie gemeinsame achse daraus ab.
        """
        index = _TIMING_INDEX[name]
        current = self._peaks[index]
        if dt > 0.0 and current > 0.0:
            current *= math.exp(-float(dt) / TIMING_AXIS_DECAY_TAU_S)
        values, _ = self.window(name)
        if values.size:
            window_max = float(values.max())
            if window_max > current:
                current = window_max
        self._peaks[index] = current
        return current

    def axis_max(self, name, dt=0.0):
        """Gerasterter achsen-maximalwert einer serie."""
        return _nice_ceiling(self.peak(name, dt))

    def shared_axis_max(self, dt=0.0):
        """Gemeinsame achse ueber die vier GEZEICHNETEN serien.

        `frame` bleibt bewusst draussen: es ist die summe der uebrigen und
        wuerde die gemeinsame achse so hoch ziehen, dass die einzelnen
        anteile flach am boden liegen.
        """
        highest = 0.0
        for name in TIMING_PLOTTED:
            value = self.peak(name, dt)
            if value > highest:
                highest = value
        return _nice_ceiling(highest)


class DevContext:
    """Sammelt die objekte, die die entwickler-panels verstellen duerfen.

    Bewusst ein schlichter halter statt globaler zugriffe: so ist an einer
    stelle sichtbar, was die oberflaeche anfassen kann.
    """

    __slots__ = ("world", "camera", "predictor", "renderer", "ship_control",
                 "ship", "frame_dt", "sim_step_s", "tick_rate", "notes",
                 "timings")

    def __init__(self, world=None, camera=None, predictor=None, renderer=None,
                 ship_control=None, ship=None, frame_dt=0.0, sim_step_s=0.0,
                 tick_rate=60.0, timing_capacity=240):
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
        self.timings = TimingHistory(timing_capacity)

    def sample_timings(self, frame_ms=0.0):
        """Eine probe der vier zeitreihen. Je frame einmal, aus test.py.

        BEWUSST ausserhalb von draw_dev_panels: das panel ist meistens zu
        (F1), und ein puffer, der nur gefuellt wird, waehrend man hinschaut,
        ist beim aufklappen leer -- also genau dann, wenn man ihn braucht.
        Der preis dafuer ist, dass das hier in JEDEM frame laeuft, weshalb
        es nichts alloziert und nur werte abliest, die render() und der
        predictor ohnehin schon geschrieben haben.

        Die vier groessen sind dieselben wie in der `TIMING:`-zeile:
          pred_compute -- dauer EINER predictor-rechnung. Laeuft auf einem
                          worker-thread, gehoert also NICHT ins bildbudget.
          pred_draw    -- projizieren/abtasten + zeichnen der linie (haupt-thread)
          rend_calc    -- renderer.render() selbst (haupt-thread)
          ui_calc      -- spieler-HUD + devtools, zwischen render() und
                          present() (haupt-thread)
          rend_draw    -- der swap selbst, inklusive der VSync-wartezeit
        """
        history = self.timings
        if history.paused:
            return

        predictor = self.predictor
        pred_compute = (_ms(getattr(predictor, 'last_compute_ms', 0.0))
                        if predictor is not None else 0.0)

        pred_draw = 0.0
        rend_calc = 0.0
        rend_draw = 0.0
        ui_calc = 0.0
        renderer = self.renderer
        if renderer is not None:
            stats = getattr(renderer, '_last_prediction_render_stats', None)
            if type(stats) is dict:
                pred_draw = _ms(stats.get('prepare_ms')) + _ms(stats.get('draw_ms'))
            timings = getattr(renderer, 'last_frame_timings', None)
            if type(timings) is dict:
                rend_draw = _ms(timings.get('swap_or_present_ms'))
                # frame_ms IST render() selbst -- present() schreibt es nicht
                # mehr um. Nichts abzuziehen, nichts abzuschneiden.
                rend_calc = _ms(timings.get('frame_ms'))
                ui_calc = _ms(timings.get('overlay_ms'))

        history.push(pred_compute, pred_draw, rend_calc, rend_draw,
                     _ms(frame_ms), ui_calc)


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


# Die vier gezeichneten graphen: (serie, titel, farbe, budget-strich?).
# Der budget-strich fehlt bei `pred_compute` mit absicht -- die serie misst
# einen worker-thread, sie hat mit dem bildbudget nichts zu tun, und ein
# strich darueber wuerde genau die verwechslung nahelegen, die die
# beschriftung vermeiden soll.
_TIMING_GRAPHS = (
    ("pred_compute", "predictor compute (worker thread)",
     (0.42, 0.72, 1.00, 1.0), False,
     "Dauer EINER trajektorien-rechnung, nicht die last des haupt-threads.\n"
     "Laeuft in einem ThreadPoolExecutor (alle kernel sind nogil), also\n"
     "auf einem anderen kern -- deshalb kein budget-strich. Der wert steht\n"
     "still, bis die naechste rechnung fertig ist; die stufen sind echt."),
    ("pred_draw", "predictor draw",
     (0.40, 0.88, 0.55, 1.0), True,
     "Haupt-thread: prepare_ms + draw_ms aus draw_prediction().\n"
     "prepare = projizieren, abtasten, kubische unterteilung, RDP.\n"
     "draw = die polylinien und die Ap/Pe-marker."),
    ("rend_calc", "render calc",
     (1.00, 0.74, 0.32, 1.0), True,
     "Haupt-thread: renderer.render() selbst (timings['frame_ms']).\n"
     "Enthaelt koerper, bahnlinien, spuren, FXAA und den predictor-anteil\n"
     "oben -- NICHT mehr das spieler-HUD, das steht jetzt in ui_calc."),
    ("ui_calc", "ui calc (spieler-HUD + devtools)",
     (0.62, 0.55, 1.00, 1.0), True,
     "Haupt-thread: alles zwischen render() und present() -- ui_root.render()\n"
     "und diese oberflaeche hier. Lief frueher unsichtbar in rend_calc mit\n"
     "und war dort etwa die haelfte der zahl."),
    ("rend_draw", "render draw (swap + VSync-wartezeit)",
     (0.92, 0.48, 0.85, 1.0), True,
     "present() -> pygame.display.flip(). Bei aktivem VSync ist das\n"
     "groesstenteils WARTEN auf das naechste bild. Ein hoher wert hier\n"
     "heisst also luft im budget, kein problem -- ausser die anderen\n"
     "serien sind gleichzeitig hoch."),
)


def _draw_timing_graphs(c: "DevContext"):
    """Die vier zeitreihen als graphen. Nur anzeige -- misst selbst nichts."""
    history = getattr(c, 'timings', None)
    if history is None:
        return

    delta_time = float(imgui.get_io().delta_time)
    tick_rate = max(1.0, float(getattr(c, 'tick_rate', 60.0) or 60.0))
    budget_ms = 1000.0 / tick_rate

    frame_cur, frame_avg, frame_max = history.stats('frame')
    imgui.text(f"frame {frame_cur:6.2f} ms   avg {frame_avg:6.2f}   "
               f"max {frame_max:6.2f}   budget {budget_ms:5.2f} @ {tick_rate:.0f} fps")
    if frame_max > budget_ms * 1.05 and history.count > 0:
        imgui.text_colored((1.0, 0.72, 0.25, 1.0),
                           f"  ueber budget in der spitze (+{frame_max - budget_ms:.2f} ms)")

    changed, value = imgui.checkbox("pause##timing", history.paused)
    if changed:
        history.paused = value
    imgui.same_line()
    changed, value = imgui.checkbox("shared scale##timing", history.shared_scale)
    if changed:
        history.shared_scale = value
    if imgui.is_item_hovered():
        imgui.set_tooltip("Eine achse fuer alle vier -- vergleicht die GROESSEN.\n"
                          "Aus: jede serie skaliert selbst und zeigt ihre FORM.")
    imgui.same_line()
    if imgui.button("reset##timing"):
        history.reset()

    changed, value = imgui.slider_int("history (frames)", history.capacity, 60, 600)
    if changed:
        history.resize(value)
    changed, value = imgui.slider_float(
        "max ms (0 = auto)", float(history.manual_max), 0.0, 120.0, "%.1f")
    if changed:
        history.manual_max = value
    if imgui.is_item_hovered():
        imgui.set_tooltip("Feste achse statt der automatischen leiter --\n"
                          "noetig, wenn zwei laeufe verglichen werden sollen.")

    # Der zerfallende spitzenwert wird je serie GENAU EINMAL je frame
    # fortgeschrieben; einzel- und gemeinsame achse leiten sich daraus ab.
    # Zweimal aufrufen liesse die achse doppelt so schnell zerfallen.
    peaks = [history.peak(name, delta_time) for name, _, _, _, _ in _TIMING_GRAPHS]
    shared_scale = _nice_ceiling(max(peaks)) if peaks else 1.0
    manual = float(history.manual_max)

    graph_width = max(120.0, float(imgui.get_content_region_avail().x))
    graph_size = imgui.ImVec2(graph_width, float(history.graph_height))
    budget_color = imgui.color_convert_float4_to_u32(
        imgui.ImVec4(1.0, 0.35, 0.35, 0.55))

    for index, (name, title, color, show_budget, tooltip) in enumerate(_TIMING_GRAPHS):
        if manual > 0.0:
            scale = manual
        elif history.shared_scale:
            scale = shared_scale
        else:
            scale = _nice_ceiling(peaks[index])

        cur, avg, peak = history.stats(name)
        imgui.text_colored(color, title)
        imgui.text(f"  cur {cur:6.2f}   avg {avg:6.2f}   max {peak:6.2f} ms"
                   f"   [0 .. {scale:g}]")

        values, offset = history.window(name)
        if values.size < 2:
            imgui.text_disabled("  (sammelt proben ...)")
            continue

        imgui.push_style_color(imgui.Col_.plot_lines, imgui.ImVec4(*color))
        imgui.plot_lines(f"##timing_{name}", values, offset,
                         f"{cur:.2f} ms", 0.0, scale, graph_size)
        imgui.pop_style_color()
        hovered = imgui.is_item_hovered()

        # Der budget-strich MUSS nach plot_lines kommen: er wird ueber die
        # rechteck-masse des zuletzt gezeichneten elements gelegt, weil
        # plot_lines selbst keine linien-annotation kennt. Ohne ihn ist die
        # achse eine reine zahl und man muss im kopf gegen die bildrate
        # rechnen -- genau das soll der graph abnehmen.
        if show_budget and 0.0 < budget_ms < scale:
            top_left = imgui.get_item_rect_min()
            bottom_right = imgui.get_item_rect_max()
            y = bottom_right.y - (bottom_right.y - top_left.y) * (budget_ms / scale)
            imgui.get_window_draw_list().add_line(
                imgui.ImVec2(top_left.x, y), imgui.ImVec2(bottom_right.x, y),
                budget_color, 1.0)

        if hovered:
            imgui.set_tooltip(tooltip)


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

    # ------------------------------------------------------------- Timing
    if imgui.collapsing_header("Timing", imgui.TreeNodeFlags_.default_open):
        _draw_timing_graphs(c)

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
            _slider("focus smoothing", cam, 'focus_smoothing', 1.0, 30.0, "%.1f",
                    tooltip="Rate des anflugs (focus_on / Home), nicht des schwenks.")
            _checkbox("pan inertia", cam, 'pan_inertia_enabled')
            _slider("inertia damping", cam, 'pan_inertia_damping', 0.5, 20.0, "%.1f")
            if imgui.button("home (ship)"):
                cam.recentre()
            imgui.same_line()
            if imgui.button("snap (no easing)"):
                cam.snap_to_targets()

    # ---------------------------------------------------------- Background
    if imgui.collapsing_header("Background"):
        bg = getattr(c.renderer, 'background', None)
        if bg is None:
            imgui.text_disabled("keine hintergrund-ebene am renderer")
        else:
            # Diese liste ist absichtlich genau der `background`-abschnitt aus
            # config.json -- kein regler mehr, kein regler weniger. Siehe
            # .claude/rules/background.md.
            _checkbox("enabled", bg, 'enabled',
                      tooltip="Ganze ebene aus: dann bleibt nur die clear-farbe.")
            _checkbox("grid enabled", bg, 'grid_enabled')
            imgui.same_line()
            _checkbox("stars enabled", bg, 'stars_enabled')

            rgb = background.parse_hex_color(bg.accent_color)
            changed, col = imgui.color_edit3("accent color", list(rgb))
            if changed:
                bg.accent_color = '#%02x%02x%02x' % tuple(
                    int(round(min(1.0, max(0.0, float(v))) * 255.0)) for v in col
                )
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Faerbt die tiefenglut UND die gitterknoten.\n"
                    f"aktuell: {bg.accent_color}"
                )

            _slider("grid opacity", bg, 'grid_opacity', 0.0, 2.0, "%.2f",
                    tooltip="Vielfaches der grund-deckkraft des gitters.")
            anchors = background.GRID_ANCHORS
            current = anchors.index(bg.grid_anchor) \
                if bg.grid_anchor in anchors else 0
            changed, idx = imgui.combo("grid anchor", current, list(anchors))
            if changed:
                bg.grid_anchor = anchors[idx]
                c.notes.append(f"background.grid_anchor = {anchors[idx]}")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "frame: festes lattice im aktiven plot-frame. Der\n"
                    "bezugskoerper steht darauf still, mond und schiff\n"
                    "wandern darueber, ein schwenk schiebt es genau so\n"
                    "weit wie die welt.\n"
                    "focus: klebt am verfolgten koerper und steht immer\n"
                    "still -- reine massstabsanzeige ohne tempo."
                )

            _slider("idle fade delay", bg, 'idle_fade_delay', 0.0, 10.0, "%.1f s",
                    tooltip="Sekunden ohne ZOOM, bis das gitter ausblendet.\n"
                            "Schwenken zaehlt bewusst nicht.")
            _slider("grid max speed", bg, 'grid_max_speed_px', 0.0, 4000.0,
                    "%.0f px/s",
                    tooltip="Obergrenze, mit der der gitteranker dem wahren\n"
                            "wert nachlaeuft. DARUNTER ist das gitter exakt\n"
                            "weltfest; darueber gleitet es nur noch mit\n"
                            "dieser rate -- das haelt es bei extremem zoom\n"
                            "lesbar. Muss ueber der schwenkrate liegen\n"
                            "(camera.move_speed x schirmhoehe, ~800 px/s),\n"
                            "sonst haengt es beim schwenken hinterher.\n"
                            "0 friert es ein.")
            _slider("pixel size", bg, 'pixel_size', 1.0, 8.0, "%.1f du",
                    tooltip="Kantenlaenge des virtuellen pixels, in DESIGN-\n"
                            "einheiten (x ui_scale). 1 = glatt, 3 = die\n"
                            "koernung der HUD-anzeigeschrift.")
            _slider("pixel round", bg, 'pixel_round', 0.0, 1.0, "%.2f",
                    tooltip="Form EINER rasterzelle.\n"
                            "0 = volles quadrat (nahtloser pixelraster).\n"
                            "1 = runder leuchtpunkt mit spalt -- die\n"
                            "leuchtpunkt-matrix. Trifft nur die tinte\n"
                            "(gitter, knoten, sterne), nie den grundverlauf.")

            dens_changed, dens = imgui.slider_int(
                "star density", int(bg.star_density), 0, 900)
            if dens_changed:
                bg.star_density = int(dens)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Zahl der sterne. Aenderung schreibt den VBO\n"
                                  "einmal neu, nicht je bild.")

            _slider("star opacity", bg, 'star_opacity', 0.0, 1.0, "%.2f")
            _slider("star motion scale", bg, 'star_motion_scale', 0.0, 5.0, "%.2f",
                    tooltip="Pixel je sekunde bei 1 km/s eigengeschwindigkeit\n"
                            "des verfolgten koerpers. Die parallaxe je stern\n"
                            "(0.05..0.55) kommt obendrauf. Unabhaengig vom\n"
                            "zoom -- 0 = feststehendes feld.")
            _slider("star zoom influence", bg, 'star_zoom_influence', 0.0, 1.0,
                    "%.2f",
                    tooltip="Wie stark das feld beim zoomen mitatmet.\n"
                            "0 = starr. 1 = die kachel dehnt sich voll, sterne\n"
                            "blenden dabei aus und andere ein, sodass die\n"
                            "dichte konstant bleibt.")

            imgui.separator()
            imgui.text(f"grid fade: {bg.grid_fade:.3f}   "
                       f"star zoom: {bg.star_zoom:+.3f} oct")
            cam = c.camera
            if cam is not None:
                # Denselben anker wie der renderer benutzen, sonst zeigt das
                # panel andere phasen als das bild.
                focus = getattr(cam, 'target', None)
                imgui.text(f"focus: {getattr(focus, 'name', 'frei')}")
                levels = bg.levels(cam.scale)
                if levels:
                    for lv in levels:
                        imgui.text(f"  10^{lv.k:<3d} = {_fmt_si(lv.spacing_m)}  "
                                   f"{lv.spacing_px:7.1f}px  a={lv.alpha:.3f}  "
                                   f"node={lv.node_alpha:.3f}")
                else:
                    imgui.text_disabled("  keine sichtbare dekade")
            imgui.text(f"star pan: ({bg.star_pan_px[0]:+7.1f}, "
                       f"{bg.star_pan_px[1]:+7.1f}) px")
            imgui.text(f"grid anchor: ({_fmt_si(bg.grid_anchor_m[0])}, "
                       f"{_fmt_si(bg.grid_anchor_m[1])})")
            imgui.text(f"  rueckstand: {bg.grid_lag_px:8.1f} px "
                       f"(grenze {bg.grid_max_speed_px:.0f} px/s)")
            if cam is not None:
                imgui.text(f"zoom ceiling: {cam._scale_ceiling():.3e} px/m "
                           f"(schirm >= {getattr(cam, 'min_visible_span_m', 0.0):.0f} m)")

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
            _slider("apsis fade min px", r, 'apsis_marker_fade_min_px', 0.0, 60.0, "%.1f",
                    tooltip="Apsis-radius am schirm (nicht zoom), unter dem der\n"
                            "marker unsichtbar ist. So stapelt er sich bei\n"
                            "weit-sicht nicht auf die schiffs-/Erde-marke.")
            _slider("apsis fade full px", r, 'apsis_marker_fade_full_px', 4.0, 160.0, "%.1f",
                    tooltip="Ab diesem apsis-radius am schirm ist der marker\n"
                            "voll deckend; dazwischen ein smoothstep.")
            _slider("orbit full-loop alpha", r, 'orbit_line_full_alpha_mult', 0.0, 1.0, "%.2f",
                    tooltip="Deckkraft der faint volllinie (ein ganzer umlauf,\n"
                            "hinter der enthuellten spur) als anteil der spur-\n"
                            "deckkraft. 0 = aus.")
            _slider("body icon min px", r, 'body_icon_min_radius_px', 1.0, 20.0, "%.1f",
                    tooltip="MINDESTgroesse der positions-marke -- und zugleich\n"
                            "die schwelle, unter der ein koerper geometrisch\n"
                            "komplett gegen sie getauscht wird. Bei 4 px waere\n"
                            "eine zelle des musters 1.1 px breit; bei 8 px\n"
                            "sind es 3.2 px.")
            _slider("body icon max px", r, 'body_icon_max_radius_px', 8.0, 128.0, "%.1f",
                    tooltip="HOECHSTgroesse, bis zu der die marke nach dem\n"
                            "PHYSISCHEN koerper-radius wachsen darf (siehe\n"
                            "'icon groessen-einfluss').")
            _slider("icon groessen-einfluss", r, 'body_icon_size_influence',
                    0.0, 1.0, "%.2f",
                    tooltip="Wie stark der PHYSISCHE koerper-radius (meter,\n"
                            "nicht der aktuelle bildschirmradius) die marken-\n"
                            "groesse skaliert -- log-skaliert ueber die spanne\n"
                            "aller geladenen koerper. 0 = jede marke bleibt bei\n"
                            "'body icon min px', egal wie gross der koerper\n"
                            "ist. 1 = ein jupiter-grosser koerper ist BEI JEDEM\n"
                            "ZOOM sichtbar groesser als ein kleiner mond,\n"
                            "geklemmt auf [min, max].")
            imgui.separator()
            imgui.text("positions-marke (body_icon.py)")
            styles = ("pixel", "disc")
            cur = styles.index(r.body_icon_style) \
                if getattr(r, 'body_icon_style', None) in styles else 0
            changed, idx = imgui.combo("icon style", cur, list(styles))
            if changed:
                r.body_icon_style = styles[idx]
                c.notes.append(f"renderer.body_icon_style = {styles[idx]}")
            variants = body_icon.VARIANTS
            cur = variants.index(r.body_icon_variant) \
                if getattr(r, 'body_icon_variant', None) in variants else 0
            changed, idx = imgui.combo("icon variante", cur, list(variants))
            if changed:
                r.body_icon_variant = variants[idx]
                c.notes.append(f"renderer.body_icon_variant = {variants[idx]}")
            changed, value = imgui.input_int(
                "icon seed-versatz", int(getattr(r, 'body_icon_seed_offset', 0)))
            if changed:
                r.body_icon_seed_offset = int(value) & 0xFFFFFFFF
                r._body_icon_cache.clear()
                c.notes.append(
                    f"renderer.body_icon_seed_offset = {r.body_icon_seed_offset}")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Globaler versatz auf JEDEN koerper-seed. Wuerfelt eine\n"
                    "ganz neue serie von marken, ohne style_seed in\n"
                    "solar_system.json anzufassen. 0 = die namensbasierten\n"
                    "vorgabe-marken.")
            if imgui.button("wuerfeln##icon_seed"):
                r.body_icon_seed_offset = (
                    int(getattr(r, 'body_icon_seed_offset', 0)) + 1) & 0xFFFFFFFF
                r._body_icon_cache.clear()
                c.notes.append(
                    f"renderer.body_icon_seed_offset = {r.body_icon_seed_offset}")
            _slider("icon ueberblend-faktor", r, 'body_icon_fade_factor',
                    1.05, 4.0, "%.2f",
                    tooltip="Das ueberblend-band endet bei 'body icon min px'\n"
                            "MAL diesem faktor -- z.b. min=16, faktor=1.6 ->\n"
                            "band bis 25.6 px echter radius. Ein FAKTOR statt\n"
                            "eines festen pixelwerts, weil der sonst beim\n"
                            "verstellen von 'min' von hand nachgezogen werden\n"
                            "musste (und das zweimal verkehrt herum stand).\n"
                            "1.0 oder darunter heisst: harter tausch, kein band.")
            _slider("icon halo", r, 'body_icon_halo_alpha', 0.0, 1.0, "%.2f",
                    tooltip="Weicher schein hinter der marke. Traegt sie gegen\n"
                            "das sternenfeld, ohne selbst als form zu lesen.")
            _slider("icon kante px", r, 'body_icon_edge_px', 0.0, 3.0, "%.2f",
                    tooltip="Breite der umriss-glaettung. 0 = harte kante und\n"
                            "damit pixelweise bewegung, 1 = ein pixel rampe.")
            changed, value = imgui.slider_int(
                "icon raster", int(getattr(r, 'body_icon_grid', 9)),
                5, body_icon.MAX_GRID)
            if changed:
                r.body_icon_grid = int(value)
                c.notes.append(f"renderer.body_icon_grid = {int(value)}")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Zellen je kante -- der DETAILGRAD, unabhaengig von der\n"
                    "bildschirmgroesse. Feiner heisst mehr muster, nicht\n"
                    "groessere marke. Achtung: unter rund 1.5 px je zelle\n"
                    "kann der schirm das raster nicht mehr aufloesen; bei\n"
                    "radius 8 ist das etwa bei 9 erreicht.")
            _slider("icon schattierung", r, 'body_icon_shade_jitter',
                    0.0, 0.6, "%.2f",
                    tooltip="Eigene helligkeit je zelle. Drei stufen allein\n"
                            "geben zu wenig tiefe -- gleich eingestufte\n"
                            "nachbarn verschmelzen sonst zu einer flaeche.\n"
                            "0 = alle zellen einer stufe gleich hell.")
            _slider("icon umriss px", r, 'body_icon_cell_rim', 0.0, 3.0, "%.2f",
                    tooltip="Breite des umrisses JEDER zelle, in pixeln.\n"
                            "Als anteil der zelle war er unter einem pixel\n"
                            "breit und dann phasenabhaengig -- eine achse\n"
                            "zeigte ihn, die andere nicht. 0 = kein umriss.")
            _slider("icon umriss dunkel", r, 'body_icon_cell_rim_dark',
                    0.0, 1.0, "%.2f",
                    tooltip="Wie dunkel der umriss wird. 1 = unsichtbar,\n"
                            "0 = schwarz.")
            _slider("icon zellspalt", r, 'body_icon_cell_gap', 0.0, 0.5, "%.2f",
                    tooltip="Anteil einer zelle, der als spalt frei bleibt.\n"
                            "0 = gleichfarbige nachbarn verschmelzen zu einer\n"
                            "flaeche und die marke wird ein klecks; 0.22 macht\n"
                            "sie wieder als raster lesbar.")

            imgui.separator_text("prediction sampling")
            _slider("tolerance px", r, 'prediction_sampling_tolerance_px',
                    0.01, 10.0, "%.3f", log=True)
            _slider("max segment px", r, 'prediction_sampling_max_segment_px',
                    0.5, 40.0, "%.1f")
            imgui.text(f"drawn: {r.debug_info['prediction_points_drawn']}"
                       f" / {r.debug_info['prediction_points_in']}")

            imgui.separator_text("prediction detail")
            _checkbox("cubic refinement", r, 'prediction_hermite_enabled')
            _slider("detail scale", r, 'prediction_detail_scale',
                    0.1, 20.0, "%.2f", log=True)
            _slider("max subdiv", r, 'prediction_hermite_max_subdiv',
                    1.0, 256.0, "%.0f")
            # Ziel = sprosse der toleranz-leiter, erreicht = das, was der
            # punktabstand des predictors ueberhaupt hergibt. Weichen sie
            # voneinander ab, ist NICHT die unterteilung die schranke,
            # sondern die integration -- dann hilft nur ein feinerer
            # punktabstand, kein hoeherer detail-regler.
            target = r.debug_info.get('prediction_detail_target_m')
            achieved = r.debug_info.get('prediction_detail_achieved_m')
            if target is None or achieved is None:
                imgui.text("tolerance: --")
            else:
                imgui.text(f"tolerance: {_fmt_si(target)} target"
                           f" / {_fmt_si(achieved)} achieved")
                if achieved > target * 1.001:
                    imgui.text_colored((1.0, 0.72, 0.25, 1.0),
                                       "  limited by point spacing, not subdivision")
            imgui.text(f"sub-points added: {r.debug_info.get('prediction_detail_added', 0)}")
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
        if c.renderer is not None:
            imgui.separator()
            imgui.text("darstellung (ship_art.py)")
            _checkbox("vektor-grafik", c.renderer, 'ship_sprite_enabled',
                      "Aus = der alte dreiecks-pfeil.")
            _slider("scale", c.renderer, 'ship_render_scale', 0.25, 6.0, "%.2f",
                    tooltip="Groesse der schiffs-grafik.\n"
                            "Sie haengt an der bildschirm-, nicht an der\n"
                            "welt-geometrie: der regler aendert nichts an der\n"
                            "physik, und die groesse bleibt ueber jede\n"
                            "zoomstufe hinweg gleich.")
            _slider("laenge (design-px)", c.renderer, 'ship_length_px', 20.0, 240.0, "%.0f",
                    tooltip="Basislaenge bei scale = 1, in design-einheiten\n"
                            "(ui_px() rechnet sie auf die aufloesung um).")
            _slider("fahne im leerlauf", c.renderer, 'ship_plume_idle', 0.0, 1.0, "%.2f",
                    tooltip="Grundhelligkeit der abgasfahne ohne schub.\n"
                            "Unter schub faehrt sie immer auf 1.0.")
            imgui.separator()
            imgui.text("zoom-schrumpfung")
            _checkbox("aktiv", c.renderer, 'ship_zoom_shrink_enabled',
                      "Aus = feste bildschirmgroesse auf jeder zoomstufe.")
            zoom_now = getattr(c.renderer, '_ship_zoom_factor', 1.0)
            imgui.text(f"faktor jetzt {zoom_now:.3f}")
            _slider("kleinster faktor", c.renderer, 'ship_zoom_shrink_min',
                    0.1, 1.0, "%.2f",
                    tooltip="Groesse ganz herausgezoomt, als anteil der\n"
                            "vollen groesse.")
            _slider("start (px/m)", c.renderer, 'ship_zoom_shrink_start_scale',
                    1e-12, 1e-3, "%.3e", log=True,
                    tooltip="Kamera-skala, ab der das schiff zu schrumpfen\n"
                            "beginnt. Darueber volle groesse.")
            _slider("ende (px/m)", c.renderer, 'ship_zoom_shrink_end_scale',
                    1e-15, 1e-4, "%.3e", log=True,
                    tooltip="Kamera-skala, ab der der kleinste faktor steht.\n"
                            "Muss unter dem start liegen, sonst wird die\n"
                            "schrumpfung ignoriert.")

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
