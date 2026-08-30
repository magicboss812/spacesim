"""Hintergrund-ebene -- sternenfeld und rekursives Dreiecksgitter.

Zwei schichten, beide unter allem anderen gezeichnet:

* das **sternenfeld**: eine feste, deterministisch erzeugte tabelle, im
  bildschirm-raster gekachelt. Es driftet mit der ECHTEN geschwindigkeit des
  verfolgten koerpers durch den raum -- nicht mit der bildschirmbewegung, und
  darum voellig unabhaengig vom zoom. Zusaetzlich atmet es beim zoomen: die
  kachel dehnt sich, sterne blenden dabei aus, andere blenden ein, sodass die
  dichte konstant bleibt.
* das **gitter**: ein dreiecks-lattice, dessen zellweite eine zehnerpotenz in
  METERN ist. Zwei bis drei dekaden sind gleichzeitig sichtbar, jede mit
  eigener deckkraft -- beim zoomen uebergibt eine dekade stetig an die
  naechste. Nach `idle_fade_delay` sekunden ohne zoom blendet es aus.

Wie `orbit_lines.py`: reines numpy, kein GL, kein pygame -- damit der ganze
block headless testbar bleibt. Das zeichnen liegt in `rendering.py`, die
rasterung in `shaders/background.frag` bzw. `shaders/star.vert`.

Die vollstaendige begruendung steht in `.claude/rules/background.md`; die vier
punkte, die man beim aendern kennen muss, auch hier:

1. **Die knoten liegen auf `x = q*ws/sqrt(3)`, `y = p*ws` mit `p + q` GERADE.**
   Die drei linienscharen (normalen `(0,1)`, `(-sqrt(3)/2, 1/2)`,
   `(sqrt(3)/2, 1/2)`) sind konkurrent -- ihr schnittpunktgitter ist genau
   diese menge. Ohne die paritaets-bedingung landet die haelfte aller knoten
   auf halben vielfachen von `ws`, also MITTEN in den dreiecken. Prueft §1
   des tests.
2. **Nirgends in der dekaden-kette steht eine harte schwelle.** Deckkraft,
   knoten und ein-/ausblenden laufen ueber `smoothstep`; eine dekade betritt
   und verlaesst `levels()` exakt dort, wo ihre deckkraft null ist. Prueft §2.
3. **Das gitter ist ein festes lattice im PLOT-FRAME.** Es haengt nicht an
   einer geschwindigkeit, sondern an der kameraposition im aktiven
   bezugsrahmen: der bezugskoerper steht darauf still, mond und schiff
   wandern darueber, und ein schwenk schiebt es exakt so weit wie die welt.
   Damit das bei extremem zoom kein schmierstreifen wird, laeuft der anker
   dem wahren wert nur mit hoechstens `grid_max_speed_px` bildschirmpixeln je
   sekunde nach. Prueft §8 und §8b.
4. **Die optik ist an das spieler-HUD gebunden** (`ui/theme.py`): dunkles
   marineblau, die HUD-kantenfarbe fuer die linien, cyan fuer die knoten --
   und alles auf ein VIRTUELLES pixelraster gerastert, ohne kantenglaettung.
   Das HUD setzt seine anzeigeschrift genauso.
"""

import math

import numpy as np


# ---------------------------------------------------------------- konstanten

#: Zellweiten sind zehnerpotenzen in metern -- `10**k`.
LEVEL_BASE = 10.0

#: Bildschirm-zellweite (px), ueber die eine dekade ein-/ausblendet.
LEVEL_FADE_IN = (24.0, 90.0)
LEVEL_FADE_OUT = (600.0, 2600.0)

#: Die akzent-knoten kommen SPAETER als die linien ihrer dekade.
NODE_FADE_IN = (90.0, 200.0)

#: `LEVEL_FADE_OUT[1] / LEVEL_FADE_IN[0]` sind gut zwei dekaden, also nie mehr
#: als drei gleichzeitig sichtbare stufen; vier deckt die raender ab.
MAX_LEVELS = 4

#: Radialer zerfall zum bildrand hin, in einheiten der halben bilddiagonale.
DISSOLVE = (0.35, 1.05)

#: Ein-/ausblendrate des ganzen gitters (1/s), bewusst asymmetrisch.
GRID_FADE_IN_RATE = 7.0
GRID_FADE_OUT_RATE = 1.1

#: Relative aenderung von `camera.target_scale`, ab der ein bild als "zoomt
#: gerade" gilt. Schwenken zaehlt bewusst NICHT.
ZOOM_ACTIVITY_EPS = 8.0e-4

#: Seed des sternenfeldes.
STAR_SEED = 90210

#: Wie schnell das sternenfeld beim zoomen eine oktave weiterrueckt:
#: `s = log2(scale) * STAR_ZOOM_RATE`. 0.5 = eine oktave je vierfachem zoom.
#: Die STAERKE des effekts regelt `star_zoom_influence`, nicht dieser wert --
#: sonst haetten beide dieselbe wirkung und man koennte sie nicht trennen.
STAR_ZOOM_RATE = 0.5

#: Obergrenze der sterndrift je bild, als anteil der bildschirmdiagonale.
#: Reine notbremse: das geschwindigkeitsmodell kann sie im normalbetrieb
#: nicht erreichen.
STAR_PAN_CLAMP_FRAC = 0.06

#: `star_motion_scale` wird in "px je sekunde bei 1 km/s" gemessen -- eine
#: zahl, die man lesen kann. Hier die umrechnung nach m/s.
STAR_SPEED_UNIT = 1000.0

#: Bei FREIER kamera gibt es keine eigengeschwindigkeit; dann treibt die
#: bildschirmbewegung des schwenks die sterne, gedaempft um diesen faktor mal
#: `star_motion_scale`. In weltmetern gerechnet waere der schwenk bei kleinem
#: zoom astronomisch (0.8 schirme/s bei 1e12 m je schirm) und liefe dauerhaft
#: in die notbremse -- genau so rasten die sterne beim schwenken davon.
FREE_PAN_GAIN = 0.3


_SQRT3 = math.sqrt(3.0)

#: "frame" = das gitter ist ein festes lattice im aktiven plot-frame,
#: "focus" = es klebt am verfolgten koerper (reine massstabsanzeige).
GRID_ANCHORS = ("frame", "focus")


def _smoothstep(edge0, edge1, x):
    """Wie GLSL `smoothstep` -- C1-stetig, damit nichts poppt."""
    if edge1 == edge0:
        return 0.0 if x < edge0 else 1.0
    t = (float(x) - edge0) / (edge1 - edge0)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return t * t * (3.0 - 2.0 * t)


#: HUD-cyan, `ui/theme.py` SCHEME[0]. Als exakter bruch, nicht gerundet --
#: sonst weicht der ausweichwert von der geparsten farbe ab.
DEFAULT_ACCENT_HEX = "#17b2c4"
DEFAULT_ACCENT_RGB = (0x17 / 255.0, 0xB2 / 255.0, 0xC4 / 255.0)


def parse_hex_color(text, default=DEFAULT_ACCENT_RGB):
    """`"#17b2c4"` -> `(r, g, b)` in 0..1. Bei unsinn: `default` (HUD-cyan)."""
    try:
        raw = str(text).strip().lstrip('#')
        if len(raw) == 3:
            raw = ''.join(c + c for c in raw)
        if len(raw) != 6:
            return default
        value = int(raw, 16)
    except (TypeError, ValueError):
        return default
    return (
        ((value >> 16) & 0xFF) / 255.0,
        ((value >> 8) & 0xFF) / 255.0,
        (value & 0xFF) / 255.0,
    )


def fold_spans(scale):
    """Die kleinste GITTERTRANSLATION, in der `(x, y)` gefaltet werden darf.

    Der anker darf um jeden vektor verschoben werden, der fuer JEDE sichtbare
    dekade eine gittertranslation ist -- dann aendert sich kein einziges
    pixel. Die groebste ueberhaupt sichtbare zellweite ist `ws = 10^k_hi`, und
    fuer sie sind die perioden:

    * in x: `2*ws/sqrt(3)`. Die knoten stehen bei `x = q*ws/sqrt(3)`, und `q`
      muss um eine GERADE zahl springen, sonst kippt die paritaet.
    * in y: `2*ws`, aus demselben grund fuer `p`.

    Fuer eine feinere dekade `10^j` sind das `2*10^(k-j)` zellen, also
    ebenfalls gerade -- die faltung ist auf jeder sichtbaren stufe exakt.

    > Das `sqrt(3)` in der x-periode ist der punkt. Wer hier schlicht modulo
    > `10^k` faltet, verschiebt das muster um `sqrt(3)*n` zellen -- eine
    > irrationale zahl, also NIE ein gitterpunkt. §8d ueberfuehrt das.
    """
    scale = float(scale)
    if not (scale > 0.0) or not math.isfinite(scale):
        return None
    ws = math.pow(LEVEL_BASE,
                  math.ceil(math.log10(LEVEL_FADE_OUT[1] / scale)))
    if not (ws > 0.0) or not math.isfinite(ws):
        return None
    return (2.0 * ws / _SQRT3, 2.0 * ws)


def fold(value, span):
    """`value` in `(-span/2, +span/2]` zurueckholen."""
    if not (span > 0.0) or not math.isfinite(span):
        return float(value)
    rest = math.fmod(float(value), span)
    if rest > span * 0.5:
        rest -= span
    elif rest < -span * 0.5:
        rest += span
    return rest


def lattice_vertices(spacing, q_range, p_range):
    """Die knoten des dreiecksgitters als `(n, 2)`-array in weltkoordinaten.

    `x = q*ws/sqrt(3)`, `y = p*ws`, und `p + q` muss GERADE sein. Nur fuer den
    test und zum nachrechnen; der renderer findet seine knoten im shader.
    """
    ws = float(spacing)
    out = []
    for p in range(int(p_range[0]), int(p_range[1]) + 1):
        for q in range(int(q_range[0]), int(q_range[1]) + 1):
            if (p + q) % 2:
                continue
            out.append((q * ws / _SQRT3, p * ws))
    if not out:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def family_normals():
    """Die drei linien-normalen."""
    return (
        (0.0, 1.0),
        (-_SQRT3 / 2.0, 0.5),
        (_SQRT3 / 2.0, 0.5),
    )


def build_star_table(count, seed=STAR_SEED):
    """`(n, 7)`-float32: x, y, radius, alpha, parallaxe, funkelphase, zoomphase.

    Linearer kongruenzgenerator, ziehreihenfolge fest -- gleicher seed ergibt
    dieselbe tabelle bis aufs bit. Eine geaenderte dichte haengt nur hinten an
    bzw. schneidet ab; die ersten `min(alt, neu)` sterne bleiben stehen.

    Die **zoomphase** (spalte 6) ist der schluessel zum atmenden feld: sie ist
    gleichverteilt in [0, 1), also liegt zu JEDER zoomstufe ein gleichmaessig
    verteilter querschnitt der sterne in jedem stadium des dehnens. Die
    sichtbare dichte ist damit exakt zoomunabhaengig -- siehe star.vert.
    """
    n = max(0, int(count))
    out = np.empty((n, 7), dtype=np.float32)
    state = int(seed)

    def rnd():
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    for i in range(n):
        out[i, 0] = rnd()                    # x im einheitsquadrat
        out[i, 1] = rnd()                    # y im einheitsquadrat
        out[i, 2] = 0.4 + rnd() * 1.5        # radius (virtuelle pixel)
        out[i, 3] = 0.25 + rnd() * 0.75      # grundhelligkeit
        out[i, 4] = 0.05 + rnd() * 0.5       # parallaxen-tiefe
        out[i, 5] = rnd() * 6.28             # funkelphase
        out[i, 6] = rnd()                    # zoomphase
    return out


class Level:
    """Eine sichtbare gitter-dekade."""

    __slots__ = ('k', 'spacing_m', 'spacing_px', 'alpha', 'node_alpha',
                 'phase_a', 'phase_b')

    def __init__(self, k, spacing_m, spacing_px, alpha, node_alpha,
                 phase_a=0.0, phase_b=0.0):
        self.k = int(k)
        self.spacing_m = float(spacing_m)
        self.spacing_px = float(spacing_px)
        self.alpha = float(alpha)
        self.node_alpha = float(node_alpha)
        self.phase_a = float(phase_a)
        self.phase_b = float(phase_b)

    def __repr__(self):
        return (f"Level(k={self.k}, sp={self.spacing_px:.1f}px, "
                f"a={self.alpha:.3f}, node={self.node_alpha:.3f})")


class BackgroundLayer:
    """Zustand und rechnung der hintergrund-ebene. Kein GL.

    `update()` einmal je bild aufrufen, danach `levels()`, `star_pan_px` und
    `star_zoom` abfragen. Alle konfigurierbaren groessen sind schlichte
    attribute, damit `loader.ConfigLoader.apply_to_background` und das
    ImGui-panel dieselbe menge an schaltern sehen.
    """

    def __init__(self):
        # ------------------------------------------------ konfigurierbar
        self.enabled = True
        self.grid_enabled = True
        self.stars_enabled = True
        #: HUD-cyan (`ui/theme.py` SCHEME[0]). Faerbt tiefenglut UND knoten.
        self.accent_color = "#17b2c4"
        self.grid_opacity = 1.0
        #: "frame" = festes lattice im aktiven plot-frame. Der bezugskoerper
        #: steht darauf still, mond und schiff wandern darueber, ein schwenk
        #: schiebt es genau so weit wie die welt. Vorgabe.
        #: "focus" = es klebt am verfolgten koerper und steht immer still --
        #: reine massstabsanzeige ohne jedes tempo.
        self.grid_anchor = "frame"
        self.idle_fade_delay = 2.2
        self.star_density = 260
        self.star_opacity = 0.55
        #: px je sekunde bei 1 km/s eigengeschwindigkeit.
        self.star_motion_scale = 0.5
        #: Obergrenze, mit der der gitteranker dem wahren wert nachlaeuft --
        #: in DESIGN-pixeln je sekunde. Solange die wahre bewegung darunter
        #: bleibt, ist das gitter exakt weltfest; darueber gleitet es nur noch
        #: mit dieser rate. Der schwenk (`camera.move_speed` schirmhoehen je
        #: sekunde, also ~800 px/s) muss DARUNTER liegen, sonst haengt das
        #: gitter beim schwenken hinterher. 0 friert es ein.
        self.grid_max_speed_px = 1500.0
        #: 0 = starres feld, 1 = volles atmen beim zoomen.
        self.star_zoom_influence = 0.35
        #: Kantenlaenge des virtuellen pixels, in DESIGN-einheiten.
        self.pixel_size = 3.0
        #: 0 = volle quadratische zelle (wie ein reiner pixelraster),
        #: 1 = runder punkt mit spalt (leuchtpunkt-matrix). Dazwischen wird
        #: die zelle abgerundet UND der spalt waechst mit. Der shader gleicht
        #: den fuellgrad aus, `grid_opacity` bedeutet also bei jeder rundung
        #: dasselbe.
        self.pixel_round = 1.0

        # ------------------------------------------------------- zustand
        self._stars = None
        self._stars_count = -1
        self._stars_dirty = True

        #: Aufsummierte sterndrift in TOP-DOWN-bildschirmpixeln.
        self.star_pan_px = np.zeros(2, dtype=np.float64)
        #: Der GEZEIGTE gitteranker in plot-frame-metern. Laeuft dem wahren
        #: anker (`grid_target_xy`) nach, begrenzt auf `grid_max_speed_px`.
        self.grid_anchor_m = np.zeros(2, dtype=np.float64)
        #: Wie weit der gezeigte anker dem wahren hinterherhaengt, in
        #: bildschirmpixeln. Nur zum ablesen -- 0 heisst "exakt weltfest".
        self.grid_lag_px = 0.0
        self._anchor_ready = False
        self._prev_target = (0.0, 0.0)
        self._grid_key = None
        #: Oktavenzaehler des atmenden feldes (`log2(scale) * rate`).
        self.star_zoom = 0.0

        self._prev_cam_world = None
        self._prev_target_scale = None
        self._prev_focus = None
        self._prev_sim_time = None
        self._focus_key = None

        self.grid_fade = 0.0
        self._idle_s = 0.0
        self.time_s = 0.0

    # ------------------------------------------------------------- sterne

    def star_table(self):
        """Die sterntabelle, bei bedarf neu erzeugt."""
        count = max(0, int(self.star_density))
        if self._stars is None or self._stars_count != count:
            self._stars = build_star_table(count)
            self._stars_count = count
            self._stars_dirty = True
        return self._stars

    def take_stars_dirty(self):
        """True (einmalig), wenn der VBO neu geschrieben werden muss."""
        dirty = self._stars_dirty
        self._stars_dirty = False
        return dirty

    def accent_rgb(self):
        return parse_hex_color(self.accent_color)

    def zoom_amount(self):
        """Staerke des atmenden feldes, auf [0, 1] geklemmt."""
        value = float(self.star_zoom_influence)
        return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)

    # ------------------------------------------------ eigengeschwindigkeit

    def _focus_speed(self, focus_world_xy, focus_key, sim_time):
        """Die geschwindigkeit des verfolgten koerpers, aus seiner POSITION.

        > **Ein `velocity`-feld gibt es fuer himmelskoerper nicht.** In
        > `solar_system.json` steht bei JEDEM geskripteten koerper
        > `"velocity": [0, 0]`, und `world.update_planets` schreibt nur
        > `position` (Kepler), nie `velocity`. Wer `body.velocity` liest,
        > bekommt fuer Erde, Mond, Mars ... exakt null -- nur das integrierte
        > Schiff traegt einen echten wert. Genau daran stand das sternenfeld
        > still, sobald man irgendetwas ausser dem Schiff anschaute.

        Deshalb wird abgeleitet: `dpos / dsim_t`. Zwei dinge muessen dabei
        stimmen:

        * **Durch die SIM-zeit teilen, nicht durch die echte.** Sonst wird die
          gemessene geschwindigkeit mit dem zeitraffer multipliziert, und bei
          1 y/s stroben die sterne. Der eigentliche schritt unten nimmt dann
          wieder `real_dt` -- das feld laeuft also in echtzeit, wie zugesagt.
        * **Beim koerperwechsel NICHT ableiten.** Der sprung von Schiff zu
          Mars sind 1e11 m in einem bild; als geschwindigkeit gelesen ginge
          das direkt in die notbremse und risse das feld quer ueber den
          schirm. `focus_key` benennt den koerper, ein wechsel setzt nur neu
          an.
        """
        if focus_world_xy is None:
            self._prev_focus = None
            self._focus_key = focus_key
            return None

        fx = float(focus_world_xy[0])
        fy = float(focus_world_xy[1])
        now = None if sim_time is None else float(sim_time)

        derived = None
        if (focus_key == self._focus_key and self._prev_focus is not None
                and now is not None and self._prev_sim_time is not None):
            dt_sim = now - self._prev_sim_time
            if dt_sim > 0.0 and math.isfinite(dt_sim):
                derived = ((fx - self._prev_focus[0]) / dt_sim,
                           (fy - self._prev_focus[1]) / dt_sim)

        self._prev_focus = (fx, fy)
        self._prev_sim_time = now
        self._focus_key = focus_key
        return derived

    # -------------------------------------------------------------- takt

    def update(self, real_dt, scale, target_scale, cam_world_xy,
               focus_world_xy=None, focus_key=None, sim_time=None,
               focus_velocity=None, grid_target=None, grid_key=None,
               viewport=(1280.0, 800.0)):
        """Einen bildschritt weiterdrehen.

        `cam_world_xy` ist die kameraposition in ABSOLUTEN weltkoordinaten.

        Die eigengeschwindigkeit des verfolgten koerpers kommt aus seiner
        POSITION (`focus_world_xy`, `focus_key`, `sim_time`), nicht aus einem
        `velocity`-feld -- siehe `_focus_speed`. `focus_velocity` ueberschreibt
        sie, wenn eine echte geschwindigkeit vorliegt.

        Die sterne rechnen bewusst ABSOLUT und nicht im plot-frame: sie stehen
        im raum fest, also antwortet das feld auf die echte eigenbewegung, und
        ein frame-wechsel (R / 1 / 2) kann es nicht rucken lassen.

        `grid_target` ist der WAHRE gitteranker in plot-frame-metern, wie ihn
        `grid_target_xy()` liefert. Er wird nicht uebernommen, sondern
        NACHGEFAHREN -- siehe unten. `grid_key` benennt, WOGEGEN er gemessen
        ist (plot-frame, ankermodus, blickziel); aendert sich der schluessel,
        ist der sprung kein flug, sondern ein neuer bezug.
        """
        dt = float(real_dt)
        if not (dt > 0.0):
            dt = 0.0
        self.time_s += dt

        scale = float(scale)
        cam_x = float(cam_world_xy[0])
        cam_y = float(cam_world_xy[1])
        prev_cam = self._prev_cam_world
        self._prev_cam_world = (cam_x, cam_y)

        limit = STAR_PAN_CLAMP_FRAC * math.hypot(
            float(viewport[0]), float(viewport[1]))

        # --- sterndrift ------------------------------------------------
        # Ein fester weltpunkt wandert bei eigenbewegung um (-vx, +vy) ueber
        # den schirm (top-down-y). Der verbraucher negiert seinen drift,
        # deshalb hier (+vx, -vy).
        if focus_velocity is None:
            focus_velocity = self._focus_speed(focus_world_xy, focus_key,
                                               sim_time)

        step_x = step_y = 0.0
        if focus_velocity is not None:
            gain = float(self.star_motion_scale) / STAR_SPEED_UNIT
            step_x = float(focus_velocity[0]) * gain * dt
            step_y = -float(focus_velocity[1]) * gain * dt
        elif prev_cam is not None and scale > 0.0 and math.isfinite(scale):
            # FREIE KAMERA. Hier gibt es keine eigengeschwindigkeit, und die
            # kamerabewegung in WELTMETERN taugt nicht als ersatz: bei
            # 1e-9 px/m sind 0.8 schirme/s rund 1e12 m/s, das ist tausendfach
            # ueber der notbremse -- die sterne rasten dann konstant mit
            # klammergeschwindigkeit davon, egal wie langsam man schwenkt.
            # Der schwenk ist eine BILDSCHIRM-bewegung, also wird er auch als
            # solche gelesen und nur gedaempft.
            gain = float(self.star_motion_scale) * FREE_PAN_GAIN
            step_x = (cam_x - prev_cam[0]) * scale * gain
            step_y = -(cam_y - prev_cam[1]) * scale * gain
        length = math.hypot(step_x, step_y)
        if length > limit > 0.0:
            trim = limit / length
            step_x *= trim
            step_y *= trim
        self.star_pan_px[0] += step_x
        self.star_pan_px[1] += step_y

        # Klein halten: der shader kachelt modulo kachelgroesse, aber so
        # bleibt die float32-genauigkeit des uniforms erhalten. Der bezug ist
        # die groesste kachel (viewport * 2), sonst springt das feld beim
        # falten, wenn es gerade gedehnt ist.
        for i in (0, 1):
            span = float(viewport[i]) * 2.0
            if span > 0.0:
                self.star_pan_px[i] = math.fmod(self.star_pan_px[i], span)

        # --- gitteranker: dem wahren wert NACHFAHREN -------------------
        # Das gitter ist ein festes lattice im plot-frame; der anker ist
        # schlicht die kameraposition darin. Uebernaehme man ihn direkt, waere
        # das gitter exakt weltfest -- richtig, aber bei 1e2 px/m rast ein
        # schiff mit 7.7 km/s dann um 8e5 px/s vorbei. Deshalb wird die
        # BEWEGUNG je bild begrenzt, nicht die position korrigiert:
        #
        #   * unter `grid_max_speed_px` ist die bewegung EXAKT die wahre --
        #     ein schwenk schiebt das gitter um genau die schwenkstrecke;
        #   * darueber gleitet es mit dieser rate in der WAHREN richtung.
        #
        # Der rueckstand bleibt dann stehen, und das ist richtig so: ein
        # unendliches lattice hat keinen ursprung, seine absolute lage ist
        # unbeobachtbar. Sichtbar ist nur die bewegung -- und die stimmt.
        # (Den rueckstand stattdessen aufzuholen hiesse, den FEHLER falten zu
        # muessen; das gitter zoege dann bei extremem zoom zur naechsten
        # gitteraequivalenten stelle statt in flugrichtung und zappelte.)
        if grid_target is not None and scale > 0.0 and math.isfinite(scale):
            tx = float(grid_target[0])
            ty = float(grid_target[1])
            spans = fold_spans(scale)
            if not self._anchor_ready:
                self.grid_anchor_m[0] = tx
                self.grid_anchor_m[1] = ty
                self.grid_lag_px = 0.0
                self._anchor_ready = True
            else:
                dx = tx - self._prev_target[0]
                dy = ty - self._prev_target[1]
                step_px = math.hypot(dx, dy) * scale
                if grid_key != self._grid_key:
                    # Kein flug, sondern ein NEUER BEZUG (R / 1 / 2, oder ein
                    # fokuswechsel bei anchor="focus"): uebernehmen statt
                    # abfahren -- sonst gliten bis zu 1e11 m minutenlang
                    # durchs bild. Der anflug verdeckt den versatz.
                    #
                    # Warum ein SCHLUESSEL und keine sprunghoehe: beides
                    # ueberlappt. Der wechsel Erde->Mond misst bei 1e-4 px/m
                    # 3.8e4 px je bild, ein vorbeiflug am zoomanschlag
                    # 8.3e4 px -- jede schwelle dazwischen trifft einmal das
                    # falsche. Der schluessel weiss es ohne zu raten.
                    self.grid_anchor_m[0] += dx
                    self.grid_anchor_m[1] += dy
                    self.grid_lag_px = 0.0
                else:
                    budget = float(self.grid_max_speed_px) * dt
                    take = 1.0
                    if step_px > 0.0:
                        take = budget / step_px
                        if take > 1.0:
                            take = 1.0
                    self.grid_anchor_m[0] += dx * take
                    self.grid_anchor_m[1] += dy * take
                    # NACH dem schritt gemessen: 0 heisst "exakt weltfest".
                    self.grid_lag_px = step_px * (1.0 - take)
            self._prev_target = (tx, ty)
            self._grid_key = grid_key
            # Den anker klein halten. Die faltung ist eine exakte
            # GITTERTRANSLATION der groebsten sichtbaren dekade und damit auf
            # jeder sichtbaren stufe unsichtbar (`fold_spans`); ohne sie
            # gingen bei 1e11 m die letzten stellen der phase verloren.
            # Beim HERAUSzoomen waechst die periode, der bereits gefaltete
            # wert liegt dann ohnehin darin -- es kann also nie zurueckspringen.
            if spans is not None:
                self.grid_anchor_m[0] = fold(self.grid_anchor_m[0], spans[0])
                self.grid_anchor_m[1] = fold(self.grid_anchor_m[1], spans[1])

        # --- atmendes feld ---------------------------------------------
        if scale > 0.0 and math.isfinite(scale):
            self.star_zoom = math.log2(scale) * STAR_ZOOM_RATE

        # --- gitter: zoom-aktivitaet und leerlauf-ausblenden ------------
        target = float(target_scale)
        zooming = False
        if self._prev_target_scale is None:
            self._prev_target_scale = target
        elif self._prev_target_scale > 0.0 and target > 0.0:
            rel = abs(math.log(target / self._prev_target_scale))
            zooming = rel > ZOOM_ACTIVITY_EPS
            self._prev_target_scale = target
        else:
            self._prev_target_scale = target

        if zooming:
            self._idle_s = 0.0
        else:
            self._idle_s += dt

        goal = 0.0 if self._idle_s > float(self.idle_fade_delay) else 1.0
        rate = GRID_FADE_IN_RATE if goal > self.grid_fade else GRID_FADE_OUT_RATE
        blend = dt * rate
        if blend > 1.0:
            blend = 1.0
        self.grid_fade += (goal - self.grid_fade) * blend

    # ------------------------------------------------------------ gitter

    def grid_target_xy(self, cam_frame_xy, focus_frame_xy=None):
        """Der WAHRE gitteranker -- wo das lattice stehen muesste.

        "frame" (Vorgabe): die kameraposition im aktiven plot-frame. Das
        gitter ist damit ein festes lattice IN DIESEM RAHMEN -- der
        bezugskoerper steht darauf still, mond und schiff wandern darueber,
        und eine kreisbahn zeichnet einen kreis. Ein schwenk verschiebt es um
        genau die strecke, um die sich die welt verschiebt.

        "focus": zusaetzlich die position des verfolgten koerpers abgezogen.
        Das gitter klebt dann am blickziel und steht immer still; es misst nur
        noch den abstand vom ziel.
        """
        x = float(cam_frame_xy[0])
        y = float(cam_frame_xy[1])
        if self.grid_anchor == "focus" and focus_frame_xy is not None:
            x -= float(focus_frame_xy[0])
            y -= float(focus_frame_xy[1])
        return (x, y)

    def anchor_xy(self):
        """Der GEZEIGTE anker -- was `levels()` als phase bekommt."""
        return (float(self.grid_anchor_m[0]), float(self.grid_anchor_m[1]))

    def levels(self, scale, anchor_x=0.0, anchor_y=0.0):
        """Die sichtbaren dekaden, hellste zuerst, hoechstens `MAX_LEVELS`.

        Eine dekade taucht genau dann auf, wenn ihre deckkraft echt groesser
        null ist -- die liste kann also nicht springen.
        """
        scale = float(scale)
        if not (scale > 0.0) or not math.isfinite(scale):
            return []
        if not self.grid_enabled or self.grid_fade <= 0.0:
            return []

        k_lo = int(math.floor(math.log10(LEVEL_FADE_IN[0] / scale)))
        k_hi = int(math.ceil(math.log10(LEVEL_FADE_OUT[1] / scale)))

        found = []
        for k in range(k_lo, k_hi + 1):
            spacing_m = math.pow(LEVEL_BASE, k)
            spacing_px = spacing_m * scale
            alpha = (_smoothstep(LEVEL_FADE_IN[0], LEVEL_FADE_IN[1], spacing_px)
                     * (1.0 - _smoothstep(LEVEL_FADE_OUT[0], LEVEL_FADE_OUT[1],
                                          spacing_px)))
            if alpha <= 0.0:
                continue
            node_alpha = alpha * _smoothstep(NODE_FADE_IN[0], NODE_FADE_IN[1],
                                             spacing_px)
            found.append(Level(
                k, spacing_m, spacing_px,
                alpha * self.grid_fade,
                node_alpha * self.grid_fade,
                *self._phases(spacing_m, anchor_x, anchor_y),
            ))

        found.sort(key=lambda lv: lv.alpha, reverse=True)
        return found[:MAX_LEVELS]

    @staticmethod
    def _phases(spacing_m, anchor_x, anchor_y):
        """Gitterphase in `(a, b)`-koordinaten, modulo 2.

        `a = x*sqrt(3)/ws`, `b = y/ws`; an einem knoten sind beide ganzzahlig
        mit gerader summe. Modulo **2**, nicht 1, weil die paritaet die
        periode des lattice ist: `a += 2` verschiebt `(m, n)` um `(1, 1)` und
        ist damit eine gittertranslation.

        In float64 gerechnet und als kleine zahl an den shader gegeben: der
        anker liegt bei bis zu 1e11 m, davon bliebe in float32 nichts uebrig.
        """
        ws = float(spacing_m)
        if not (ws > 0.0) or not math.isfinite(ws):
            return 0.0, 0.0
        phase_a = math.fmod(_SQRT3 * float(anchor_x) / ws, 2.0)
        phase_b = math.fmod(float(anchor_y) / ws, 2.0)
        return phase_a, phase_b
