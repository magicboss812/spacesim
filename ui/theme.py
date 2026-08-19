"""Farbpalette, typo-stufen, abstaende und eckformen des spieler-HUDs.

FORMSPRACHE: FASE STATT RUNDUNG. Die vorlage ist die instrumententafel aus
Kerbal Space Program 2 -- und, was die entscheidung erst zwingend macht, die
hausschrift SB Liquid selbst: beide schneiden ecken unter 45 grad ab, statt
sie zu runden, und oft nur EINIGE ecken. Diese asymmetrie ist der
eigentliche traeger des eindrucks "konstruiert" statt "dekoriert".
Umgesetzt ist sie im SDF-shader ueber ein VORZEICHEN -- ein negativer
eckwert ist eine fase (siehe shaders/ui_rect.frag, chamfer_box) --, also
ohne zusaetzliches attribut und ohne zweiten shader.

ZWEI SCHRIFTEN, ZWEI AUFGABEN. `display` ist SB Liquid: eine gefaste
pixelschrift, gerastert OHNE kantenglaettung und auf ein vielfaches von
fuenf pixel gerundet -- so bleiben die stege ueberall gleich breit und die
kanten hart. Sie traegt beschriftungen, messwerte und knopftexte, also
alles, was nach instrument aussehen soll. `text` ist Oxanium: dieselben
quadratischen grundformen, aber mit weich gerundeten ecken und echter
kantenglaettung. Sie traegt namen, gemischte schreibung und alles, was
gelesen statt abgelesen wird. Beide haben bei gleicher nenngroesse
praktisch dieselbe versalhoehe und dieselbe laufweite (gemessen: 7 bzw.
8 px versalhoehe und 68 bzw. 64 px vorschub bei groesse 10), lassen sich
also in einer zeile mischen, ohne dass eine der beiden herausfaellt.

TYPO-KONTRAST TRAEGT DIE HIERARCHIE, NICHT DIE FARBE: gesperrte
10-px-beschriftungen gegen 25-30-px-messwerte. Das ist der auffaelligste
einzelzug der vorlage und der grund, warum dort kaum farbe noetig ist.

DIE PALETTE IST FESTGELEGT. Vier farben, jede mit EINER bedeutung --
cyan = daten, magenta = zweite achse/ziel, amber = achtung und energie,
gruen = eingerastet/bereit. Der frueher hier moegliche palettenwechsel ist
entfallen: eine farbe, die sich neu verteilen laesst, kann nichts bedeuten.

Alle groessen sind DESIGN-EINHEITEN; die umrechnung auf echte pixel passiert
ausschliesslich ueber UIContext.px().
"""

import colorsys

# --------------------------------------------------------------- farbhilfen


def rgba(hex_color, alpha=1.0):
    """'#0e1420' oder '0e1420' -> (r, g, b, a) als floats in [0, 1]."""
    text = str(hex_color).lstrip('#')
    if len(text) == 3:
        text = ''.join(ch * 2 for ch in text)
    if len(text) == 8:
        alpha = int(text[6:8], 16) / 255.0 * alpha
        text = text[:6]
    if len(text) != 6:
        raise ValueError(f"ungueltige farbe: {hex_color!r}")
    return (
        int(text[0:2], 16) / 255.0,
        int(text[2:4], 16) / 255.0,
        int(text[4:6], 16) / 255.0,
        float(alpha),
    )


def with_alpha(color, alpha):
    """Dieselbe farbe mit anderer deckkraft."""
    return (color[0], color[1], color[2], float(alpha))


def mix(color_a, color_b, t):
    """Lineare mischung; t = 0 -> a, t = 1 -> b. Fuer hover-/fokus-uebergaenge."""
    t = max(0.0, min(1.0, float(t)))
    return tuple(a + (b - a) * t for a, b in zip(color_a, color_b))


def _relative_luminance(color):
    """WCAG-luminanz. Erwartet lineare sRGB-eingaben in [0, 1]."""
    channels = []
    for value in color[:3]:
        value = max(0.0, min(1.0, float(value)))
        channels.append(
            value / 12.92 if value <= 0.03928
            else ((value + 0.055) / 1.055) ** 2.4
        )
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def readable(color, background_luminance=0.0055, target_ratio=4.5):
    """Hellt eine palettenfarbe auf, bis sie auf dem dunklen grund lesbar ist.

    Der entwurf benutzt genau diese regel fuer text und duenne striche. Ohne
    sie faellt eine dunkle palettenfarbe (etwa #22577a) auf dem fast schwarzen
    hintergrund auf ~1.6:1 zusammen und ist als beschriftung unbrauchbar --
    die palette bliebe damit auf grosse flaechen beschraenkt.

    Gemischt wird richtung WEISS, nicht ueber die helligkeit im HSL-raum:
    aufhellen soll den farbton erhalten und nur saettigung abgeben.
    """
    result = color
    step = 0.0
    while step <= 0.8001:
        result = mix(color, (1.0, 1.0, 1.0, color[3]), step)
        ratio = (_relative_luminance(result) + 0.05) / (background_luminance + 0.05)
        if ratio >= target_ratio:
            return result
        step += 0.08
    return result


def ink_on(color, threshold=0.32):
    """Vorder-auf-hintergrund: dunkle schrift auf hellen flaechen, hell auf
    dunklen. Derselbe schwellwert wie im entwurf."""
    if _relative_luminance(color) > threshold:
        return rgba('#07090d')
    return rgba('#f2f7fc')


def shift_hue(color, degrees):
    """Farbton drehen, saettigung und helligkeit behalten."""
    h, l, s = colorsys.rgb_to_hls(color[0], color[1], color[2])
    h = (h + float(degrees) / 360.0) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b, color[3])




# ----------------------------------------------------------------- palette

#: DIE vier farben. Kein zweiter satz, kein wechselknopf -- siehe modulkopf.
#: Reihenfolge = bedeutung, nicht helligkeit.
SCHEME_NAME = 'Kerbin'
SCHEME = (
    '#17b2c4',   # 0 CYAN    -- daten: bahn, ring, geschwindigkeit, rahmen
    '#d9519f',   # 1 MAGENTA -- zweite achse: normal/antinormal, ziel
    '#eda63c',   # 2 AMBER   -- energie und achtung: schub, hoehe, zeitraffer
    '#48d97c',   # 3 GRUEN   -- eingerastet / bereit: autopilot, schiff
)

#: Rolle -> index in SCHEME. Fest, weil eine farbe sonst nichts bedeutet.
ROLE_INDEX = {
    'body': 0,        # himmelskoerper-fuellung
    'orbit': 0,       # bahnlinien, system-map
    'ring': 0,        # attitude-ring und teilstriche
    'elem': 0,        # bahnelemente
    'velocity': 0,    # geschwindigkeitsnadel und -wert
    'frame': 0,       # bezugsrahmen-auswahl
    'target': 1,      # ziel
    'normal': 1,      # normal-/antinormal-achse
    'warp': 2,        # zeitraffer
    'throttle': 2,    # schub
    'altitude': 2,    # hoehe ueber dem bezugskoerper
    'star': 2,        # zentralgestirn
    'snap': 3,        # orientierungs-autopilot
    'ship': 3,        # schiff
}


class Palette:
    """Die vier bedeutungsfarben plus die feste, dunkle grundierung.

    Der grund (panel, kante, schrift) ist BEWUSST nicht teil der palette:
    er muss unter jeder farbe gleich lesbar bleiben, und ein mitgefaerbter
    hintergrund wuerde die vier akzente entwerten.
    """

    def __init__(self, colors=None, name=SCHEME_NAME):
        self.name = name
        self.set_colors(colors or SCHEME)

    # ------------------------------------------------------ feste grundierung

    # Flaechen. Deutlich dunkler und BLAEULICHER als zuvor: die vorlage setzt
    # ihre instrumente auf ein sehr dunkles marineblau, nicht auf neutrales
    # schwarz -- das ist es, was die cyan-daten darauf leuchten laesst.
    ground = rgba('#04070c')
    panel = rgba('#080f18', 0.88)
    panel_pill = rgba('#0a1420', 0.92)
    panel_popup = rgba('#0a1420', 0.97)
    panel_sunken = rgba('#000308', 0.75)
    ring_face = rgba('#061019', 0.94)
    #: Innenflaeche eines doppelt gerahmten blocks (aussenlinie, spalt, kern).
    panel_core = rgba('#0c1826', 0.94)

    # Kanten. Die vorlage rahmt DOPPELT: eine haarfeine aussenlinie, ein
    # schmaler spalt, dann der kern. edge ist die aussenlinie, edge_inner die
    # innere; edge_strong traegt den aktiven zustand.
    edge = rgba('#7fb4cc', 0.34)
    edge_strong = rgba('#a8d8ea', 0.55)
    edge_inner = rgba('#7fb4cc', 0.16)

    # Schrift.
    text = rgba('#dceaf4')
    text_muted = rgba('#8fa8bc')
    text_dim = rgba('#6f8698')
    text_dimmer = rgba('#4c5f70')
    text_inverse = rgba('#04070c')

    # Interaktionszustaende.
    hover = rgba('#a8d8ea', 0.10)
    active = rgba('#a8d8ea', 0.20)
    idle_fill = rgba('#a8d8ea', 0.045)
    disabled = rgba('#a8d8ea', 0.03)
    shadow = rgba('#000000', 0.70)

    # Aliase, die die allgemeinen widgets aus ui/widgets/ erwarten.
    panel_raised = rgba('#101d2c', 0.94)
    divider = rgba('#7fb4cc', 0.14)
    border = edge
    border_strong = edge_strong
    warning = rgba('#eda63c')
    danger = rgba('#ff5f6b')

    def set_colors(self, colors):
        """Setzt die vier farben und leitet alle rollen neu ab."""
        self.colors = [rgba(c) if isinstance(c, str) else tuple(c) for c in colors]
        while len(self.colors) < 4:
            self.colors.append(self.colors[-1])
        self.colors = self.colors[:4]

        # Rohfarben fuer FLAECHEN, aufgehellte fuer STRICHE UND SCHRIFT.
        self.raw = {role: self.colors[i] for role, i in ROLE_INDEX.items()}
        self.role = {role: readable(color) for role, color in self.raw.items()}

    # Bequemer zugriff: palette.ring statt palette.role['ring'].
    def __getattr__(self, name):
        role = self.__dict__.get('role')
        if role and name in role:
            return role[name]
        raise AttributeError(name)

    def raw_of(self, role):
        """Ungehellte farbe -- fuer flaechen, nie fuer schrift."""
        return self.raw.get(role, self.colors[0])

    def glow(self, role, intensity=0.6):
        """Der farbige schein hinter einem block -- ein schlagschatten mit
        versatz null und weiter weichzeichnung.

        DEUTLICH SCHWAECHER als frueher (0.28 -> 0.13): der schein war das,
        was die alte oberflaeche nach "jedes element schwebt einzeln"
        aussehen liess. Er soll einen block vom sternenfeld abheben, nicht
        ihn zum leuchtobjekt machen.
        """
        return with_alpha(self.role.get(role, self.colors[0]), 0.13 * float(intensity))

    def accent_for(self, role):
        return self.role.get(role, self.colors[0])

    # Namen, die ui/widgets/ generisch benutzt.
    @property
    def accent(self):
        return self.role['velocity']

    @property
    def accent_strong(self):
        return self.role['ship']

    @property
    def accent_soft(self):
        return with_alpha(self.role['velocity'], 0.18)

    @property
    def focus_ring(self):
        return with_alpha(self.role['velocity'], 0.65)


class Role:
    """Eine typo-rolle: groesse, laufweite, strichstaerke, schriftfamilie.

    FAMILIE ist hier die eigentliche entscheidung: 'display' ist die gefaste
    pixelschrift (hart gerastert, groesse auf fuenferschritte gerundet),
    'text' die weich gerundete leseschrift. Siehe modulkopf.

    LAUFWEITE traegt den entwurf: versal-beschriftungen laufen auf .16em,
    einheiten auf .20em. Ohne sperrung sieht die oberflaeche voellig anders
    aus.
    """

    __slots__ = ('size', 'tracking', 'bold', 'family')

    def __init__(self, size, tracking=0.0, bold=False, family='display'):
        self.size = float(size)
        self.tracking = float(tracking)   # in em
        self.bold = bool(bold)
        self.family = str(family)

    @property
    def mono(self):
        """Rueckwaertskompatibler alias -- die pixelschrift IST dicktengleich."""
        return self.family == 'display'


class TypeScale:
    """Rollen, nicht groessen -- widgets fragen nach 'section', nicht nach 10.

    Die groessen der display-rollen liegen bewusst auf der FUENFER-LEITER
    (10/15/20/25/30). Die pixelschrift wird auf das naechste vielfache von
    fuenf gerundet gerastert; wer hier 11 oder 13 einsetzt, bekommt denselben
    pixelwert wie ein nachbar und verliert die stufe.
    """

    # -------------------------------------------------- instrument (display)
    #: Die notch-tabs auf der panelkante: ORBITAL.INFO, SNAP.CONTROL.
    tab = Role(10, tracking=0.16)
    #: Abschnitts-beschriftung im panel.
    section = Role(10, tracking=0.16)
    #: Kleinstbeschriftung unter einem knopf oder neben einem wert.
    caption = Role(10, tracking=0.12)
    #: Einheit hinter einem messwert.
    unit = Role(10, tracking=0.20)
    #: Wert einer messwert-zeile.
    value = Role(15, tracking=0.02)
    #: Der grosse messwert in den navball-flanken (KSP2: 01527 / 00876).
    gauge = Role(25, tracking=0.02)
    #: Der grosse wert unter dem instrument.
    readout = Role(25, tracking=0.02)
    #: Kurs ueber dem ring (KSP2: 357 grad).
    heading_big = Role(20, tracking=0.04)
    #: Zeitraffer-stufen, kurs-nabe, timer.
    warp = Role(10, tracking=0.06)
    hdg = Role(10, tracking=0.10)
    #: Knopfbeschriftungen.
    button = Role(10, tracking=0.14)
    button_sm = Role(10, tracking=0.12)
    #: Teilung und marker des rings.
    ring_caption = Role(10, tracking=0.20)
    ring_unit = Role(10, tracking=0.20)
    ring_tick = Role(10, tracking=0.06)
    ring_marker = Role(10, tracking=0.0)
    throttle_value = Role(15, tracking=0.02)
    glyph = Role(15, tracking=0.0)

    # ------------------------------------------------------- lesetext (text)
    #: Schluessel einer messwert-zeile (AP, PE, ECC) -- gemischt lesbar.
    key = Role(11, tracking=0.10, family='text')
    #: Schiffs-plakette und ihr untertitel.
    badge = Role(13, tracking=0.06, bold=True, family='text')
    badge_sub = Role(11, tracking=0.10, family='text')
    #: Ziel-/koerpername.
    title = Role(16, tracking=0.02, bold=True, family='text')
    #: Marke wie LOCKED.
    pill = Role(10, tracking=0.10, family='text')
    #: Eintraege der koerperliste.
    body = Role(12, tracking=0.01, family='text')
    label = Role(11, tracking=0.04, family='text')
    heading = Role(15, tracking=0.02, bold=True, family='text')

    # Von den allgemeinen widgets erwartete namen.
    mono_readout = Role(15, tracking=0.02)
    display = Role(30, tracking=0.02)
    ring_speed = Role(25, tracking=0.02)


class Spacing:
    """Abstands-leiter. Grob gestuft, damit nicht jedes widget seinen
    eigenen wert erfindet."""

    none = 0
    xs = 3
    sm = 5
    md = 8
    lg = 11
    xl = 15
    xxl = 22
    section = 30


class Radius:
    """Eckformen in design-einheiten.

    NEGATIV = FASE, positiv = rundung (siehe modulkopf und
    shaders/ui_rect.frag). Die oberflaeche benutzt bis auf echte kreise
    ausschliesslich fasen -- durchgaengige rundung war der staerkste
    einzelne "das hat eine maschine entworfen"-hinweis der alten fassung.
    """

    none = 0
    #: Kleine fase an knoepfen und kacheln.
    cut_sm = -4
    #: Regelfase an panels und rahmen.
    cut = -7
    #: Grosse fase an den tragenden bloecken.
    cut_lg = -11
    #: Sanfte rundung -- nur dort, wo etwas ausdruecklich weich sein soll.
    sm = 3
    md = 5
    #: Vollkreis; wird im shader auf die halbe kante geklemmt.
    pill = 999

    # Alte namen, damit die allgemeinen widgets weiterlaufen. Sie zeigen
    # jetzt auf FASEN -- eine rundung soll nirgends mehr versehentlich
    # zurueckkommen.
    lg = -7
    xl = -11


def cut_corners(size, top_left=True, top_right=True,
                bottom_right=True, bottom_left=True):
    """Eckwert-tupel, bei dem NUR die gewaehlten ecken gefast sind.

    Die vorlage fast fast nie alle vier ecken -- genau die asymmetrie laesst
    einen block konstruiert wirken. Beispiel: ein panel mit notch-tab oben
    links laesst diese ecke scharf und fast die drei anderen.
    """
    cut = -abs(float(size))
    return (
        cut if top_left else 0.0,
        cut if top_right else 0.0,
        cut if bottom_right else 0.0,
        cut if bottom_left else 0.0,
    )


class Motion:
    """Uebergangsraten fuer exponentielles easing (1 - exp(-rate * dt)).

    RATEN, keine dauern -- framerate-unabhaengig, wie das kamera-easing.
    """

    instant = 0.0
    fast = 22.0
    normal = 14.0
    slow = 8.0


class Theme:
    """Buendelt die stufen. Erreichbar ueber UIContext.theme."""

    def __init__(self, palette=None, glow_intensity=0.6, compact_breakpoint=900):
        self.palette = palette if palette is not None else Palette()
        self.glow_intensity = float(glow_intensity)
        # Unter dieser fensterbreite klappen die seitenpanels zu schmalen
        # leisten zusammen.
        self.compact_breakpoint = float(compact_breakpoint)

    type_scale = TypeScale
    spacing = Spacing
    radius = Radius
    motion = Motion

    # Standard-metriken.
    control_height = 24
    control_height_sm = 18
    control_height_lg = 32
    panel_padding = 12
    panel_width = 190
    border_width = 1.0
    #: Breite des spalts zwischen aussen- und innenlinie eines doppelrahmens.
    frame_gap = 3
    shadow_offset = (0.0, -4.0)   # ortho-konvention: negativ = nach unten
    shadow_softness = 14.0

    # Schrift-familien. Die beiden TTF in ui/assets/ sind die vorgabe; die
    # systemnamen dahinter greifen nur, wenn eine datei fehlt.
    #: SB Liquid -- gefaste pixelschrift, OHNE kantenglaettung gerastert.
    font_family_display = ('Consolas', 'DejaVu Sans Mono', 'Courier New')
    #: Oxanium -- quadratisch mit weichen ecken.
    font_family_text = ('Oxanium', 'Chakra Petch', 'Segoe UI', 'DejaVu Sans')

    # Rueckwaertskompatible namen.
    font_family = font_family_text
    font_family_mono = font_family_display

    def palette_sets(self):
        """Nur noch der EINE satz -- der wechselknopf ist entfallen."""
        return ((SCHEME_NAME, SCHEME),)

    def set_palette_colors(self, colors, name=None):
        self.palette.set_colors(colors)
        if name:
            self.palette.name = name

    def glow(self, role):
        return self.palette.glow(role, self.glow_intensity)


DEFAULT_THEME = Theme()
