"""Farbpalette, typo-stufen, abstaende und radien des spieler-HUDs.

Uebertragen aus dem entwurf "Spacesim 2D gameplay GUI mockup"
(claude.ai/design, projekt 8790e76e). Alle groessen sind DESIGN-EINHEITEN
und entsprechen 1:1 den pixelwerten des entwurfs; die umrechnung auf echte
pixel passiert ausschliesslich ueber UIContext.px().

KERNIDEE DES ENTWURFS: die oberflaeche zieht ihre farbe aus GENAU VIER
werten. Jede rolle -- ring, geschwindigkeit, schub, ziel, schiff, rahmen,
snap, bahn -- greift auf eine davon zu. Der grund steht im entwurf selbst:
"The ground stays dark; the palette only tints chrome, glow and data."
Ein palettenwechsel faerbt damit das gesamte HUD um, ohne dass eine einzige
widget-datei angefasst werden muss.

Die vier farben sind fuer FLAECHEN gewaehlt, nicht fuer text. Duenne striche
und schrift laufen deshalb durch readable(): das hellt eine farbe so lange
richtung weiss auf, bis sie auf dem dunklen grund ~4.5:1 erreicht.
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


# ------------------------------------------------------------ palettensaetze

PALETTE_SETS = (
    ('Baltic', ('#22577a', '#38a3a5', '#57cc99', '#80ed99')),
    ('Ember', ('#3d2b56', '#c1462f', '#e0803c', '#f2c14e')),
    ('Ion', ('#2b3a67', '#496ddb', '#8b5cf6', '#c2b0ff')),
)

# Welche der vier farben welche rolle traegt. Der entwurf verteilt das per
# zufallsseed neu ("SHUFFLE DISTRIBUTION"), was zum ERKUNDEN gedacht ist --
# im spiel ist eine feste, bewusst gewaehlte zuordnung richtig, sonst
# bedeutet dieselbe farbe von sitzung zu sitzung etwas anderes.
#
# Index 0 ist die dunkelste, 3 die hellste farbe des satzes.
ROLE_INDEX = {
    'body': 0,        # himmelskoerper-fuellung, dunkelste
    'orbit': 1,       # bahnlinien, system-map
    'ring': 1,        # attitude-ring und teilstriche
    'warp': 2,        # zeitraffer
    'frame': 2,       # bezugsrahmen-auswahl
    'snap': 2,        # orientierungs-autopilot
    'throttle': 2,    # schub
    'elem': 2,        # bahnelemente
    'velocity': 3,    # geschwindigkeitsnadel, hellste
    'target': 3,      # ziel
    'ship': 3,        # schiff
    'star': 3,        # zentralgestirn
}


class Palette:
    """Die vier palettenfarben plus die feste, dunkle grundierung.

    Der grund (panel, kante, schrift) ist BEWUSST nicht teil der palette:
    er muss unter jeder palette gleich lesbar bleiben, und ein mitgefaerbter
    hintergrund wuerde die vier akzente entwerten.
    """

    def __init__(self, colors=None, name='Baltic'):
        self.name = name
        self.set_colors(colors or PALETTE_SETS[0][1])

    # ------------------------------------------------------ feste grundierung

    # Flaechen. Der entwurf setzt panels auf rgba(10,15,22,.70) ueber einem
    # #07090d-grund; pillen liegen etwas dichter, popups fast deckend.
    ground = rgba('#07090d')
    panel = rgba('#0a0f16', 0.70)
    panel_pill = rgba('#0a0f16', 0.76)
    panel_popup = rgba('#0a0f16', 0.92)
    panel_sunken = rgba('#ffffff', 0.09)
    ring_face = rgba('#080c12', 0.82)

    # Kanten. Weisse hairlines statt farbiger raender -- so bleibt die kante
    # unter jeder palette gleich stark.
    edge = rgba('#ffffff', 0.12)
    edge_strong = rgba('#ffffff', 0.18)
    edge_inner = rgba('#ffffff', 0.07)

    # Schrift.
    text = rgba('#e8eef5')
    text_muted = rgba('#8b9cb0')
    text_dim = rgba('#7f8fa3')
    text_dimmer = rgba('#5c6c80')
    text_inverse = rgba('#07090d')

    # Interaktionszustaende.
    hover = rgba('#ffffff', 0.08)
    active = rgba('#ffffff', 0.16)
    idle_fill = rgba('#ffffff', 0.05)
    disabled = rgba('#ffffff', 0.04)
    shadow = rgba('#000000', 0.55)

    # Aliase, die die allgemeinen widgets aus ui/widgets/ erwarten.
    panel_raised = rgba('#131b26', 0.86)
    divider = rgba('#ffffff', 0.08)
    border = edge
    border_strong = edge_strong
    warning = rgba('#f5c451')
    danger = rgba('#ff6b6b')

    def set_colors(self, colors):
        """Setzt die vier palettenfarben und leitet alle rollen neu ab."""
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
        """Ungehellte palettenfarbe -- fuer flaechen, nie fuer schrift."""
        return self.raw.get(role, self.colors[2])

    def glow(self, role, intensity=0.6):
        """Der farbige schein um ein panel. Im entwurf ein box-shadow;
        hier eine sehr schwache, weiche fuellung hinter dem panel."""
        return with_alpha(self.role.get(role, self.colors[2]), 0.28 * float(intensity))

    def accent_for(self, role):
        return self.role.get(role, self.colors[2])

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

    LAUFWEITE ist hier keine feinheit, sondern traegt den entwurf: die
    abschnitts-beschriftungen laufen auf .18em, die zahlen-einheiten auf
    .24em. Ohne sperrung sieht die oberflaeche voellig anders aus.
    """

    __slots__ = ('size', 'tracking', 'bold', 'mono')

    def __init__(self, size, tracking=0.0, bold=False, mono=False):
        self.size = float(size)
        self.tracking = float(tracking)   # in em
        self.bold = bool(bold)
        self.mono = bool(mono)


class TypeScale:
    """Rollen, nicht groessen -- widgets fragen nach 'section', nicht nach 9."""

    # Abschnitts-ueberschriften in panels ("ORBITAL ELEMENTS", "TARGET").
    section = Role(9, tracking=0.18)
    # Schluessel einer messwert-zeile ("AP", "PE", "ECC").
    key = Role(11, tracking=0.06)
    # Wert einer messwert-zeile. Tabellenziffern: eine zahl, die sich jeden
    # frame aendert, darf die spaltenbreite nicht verschieben.
    value = Role(13, tracking=0.0, bold=True, mono=True)

    # Schiffs-plakette oben links.
    badge = Role(13, tracking=0.05, bold=True)
    badge_sub = Role(11, tracking=0.12)
    # Ziel-name.
    title = Role(16, tracking=0.02, bold=True)
    # "LOCKED"-marke.
    pill = Role(9, tracking=0.10)

    # Bedienelemente.
    button = Role(11, tracking=0.12, bold=True)
    button_sm = Role(10, tracking=0.12, bold=True)
    warp = Role(11, tracking=0.0, bold=True, mono=True)
    glyph = Role(14, tracking=0.0, bold=True)
    caption = Role(9, tracking=0.10)

    # Attitude-ring.
    ring_caption = Role(9, tracking=0.24)
    ring_speed = Role(30, tracking=-0.015, bold=True, mono=True)
    ring_unit = Role(10, tracking=0.24)
    # Der grosse messwert in der geschwindigkeits-plakette unter dem ring.
    readout = Role(24, tracking=-0.01, bold=True, mono=True)
    ring_tick = Role(11, tracking=0.09, bold=True)
    ring_marker = Role(9, tracking=0.0, bold=True)
    hdg = Role(11, tracking=0.11, bold=True, mono=True)

    # Schub.
    throttle_value = Role(14, tracking=0.0, bold=True, mono=True)

    # Von den allgemeinen widgets erwartete namen.
    body = Role(13, tracking=0.0)
    label = Role(11, tracking=0.06)
    heading = Role(16, tracking=0.02, bold=True)
    mono_readout = Role(13, tracking=0.0, bold=True, mono=True)
    display = Role(30, tracking=-0.015, bold=True)


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
    """Eckradien in design-einheiten, direkt aus dem entwurf."""

    none = 0
    sm = 6
    md = 12       # snap-kacheln
    lg = 16       # panels
    xl = 18       # popups, fensterrahmen
    pill = 999    # wird im shader auf die halbe kante geklemmt


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
        # leisten zusammen. Der wert kommt aus dem entwurf.
        self.compact_breakpoint = float(compact_breakpoint)

    type_scale = TypeScale
    spacing = Spacing
    radius = Radius
    motion = Motion

    # Standard-metriken.
    control_height = 26
    control_height_sm = 20
    control_height_lg = 34
    panel_padding = 14
    panel_width = 196
    border_width = 1.0
    shadow_offset = (0.0, -4.0)   # ortho-konvention: negativ = nach unten
    shadow_softness = 14.0

    # Schrift-familien in prioritaetsreihenfolge. Der entwurf benutzt
    # "Chakra Petch" -- eine Google-schrift, die auf Windows nicht
    # vorinstalliert ist. ui/text.py nimmt sie, WENN sie installiert ist,
    # sonst die naechste der liste. Eine datei in ui/assets/ui-sans.ttf
    # (bzw. ui-sans-bold.ttf) hat vorrang vor allem hier.
    font_family = ('Chakra Petch', 'Segoe UI', 'Inter', 'Roboto', 'DejaVu Sans', 'Arial')
    font_family_mono = ('Chakra Petch', 'Consolas', 'DejaVu Sans Mono', 'Courier New')

    def palette_sets(self):
        return PALETTE_SETS

    def set_palette_colors(self, colors, name=None):
        self.palette.set_colors(colors)
        if name:
            self.palette.name = name

    def glow(self, role):
        return self.palette.glow(role, self.glow_intensity)


DEFAULT_THEME = Theme()
