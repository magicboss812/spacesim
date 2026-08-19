"""Textdarstellung der UI-schicht.

Uebernimmt die rolle von Renderer._get_label_texture / _blit_cached_text,
aber mit drei unterschieden:

1. **Rollen statt roher fonts.** Aufrufer fragen nach 'label' oder
   'mono_readout'; welche familie und welche pixelgroesse daraus wird,
   entscheidet das theme zusammen mit der ui_scale.
2. **Farbe.** Der text wird WEISS gerastert und ueber u_color (texquad.frag)
   getoent. Eine weisse textur pro text bedient damit beliebig viele farben --
   waere die farbe teil der rasterung, muesste sie in den cache-schluessel und
   jeder hover-zustand wuerde eine eigene GL-textur belegen.
3. **Top-down koordinaten.** Die ganze UI-schicht rechnet in bildschirm-
   konvention (ursprung oben links, y nach unten) -- genau wie pygames
   mauskoordinaten. Die umrechnung in die ortho-konvention passiert HIER,
   an der grenze, und nirgends sonst (siehe CLAUDE.md).

Zwei fallen aus Phase 0-2 sind hier fest eingebaut und duerfen nicht
aufgeweicht werden:

- **Auf das pixelraster rasten.** Subpixel-positionen verteilen bei
  LINEAR-filterung jede glyphenzeile auf zwei pixelzeilen: der text wird
  weich und bekommt eine geisterkopie. Gemessen fiel der anteil voll
  deckender pixel von 19.5 % auf 9 %.
- **Nicht durch FXAA.** Ein kantenfilter ueber glyphen schmiert sie ueber
  55 % mehr pixel (34.7 % -> 5.3 % voll deckend). Text gehoert IMMER hinter
  den FXAA-resolve; defer()/flush() ist der mechanismus dafuer.
"""

import math
import os

import moderngl
import numpy as np
import pygame

from .theme import DEFAULT_THEME

_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
_SHADER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shaders'
)


class TextRenderer:
    """Font-verwaltung, label-textur-cache und texturiertes blitten."""

    def __init__(self, ctx, width, height, theme=DEFAULT_THEME,
                 ui_scale=1.0, cache_max=256):
        self.ctx = ctx
        self.width = int(width)
        self.height = int(height)
        self.theme = theme
        self.ui_scale = float(ui_scale)
        self.cache_max = int(cache_max)

        self._program = None
        self._vao = None
        self._quad_vbo = None
        self._fonts = {}            # rolle -> pygame.Font
        self._cache = {}            # (text, font_key) -> (texture, w, h)
        self._deferred = []
        self._font_paths = None
        self._digit_widths = {}

        self._init_pipeline()
        self._rebuild_fonts()

    # ------------------------------------------------------------------ GL

    def _init_pipeline(self):
        """Eigene texquad-pipeline. Bewusst NICHT die des Renderers geteilt:
        die UI-schicht soll ohne Renderer-instanz testbar bleiben."""
        try:
            with open(os.path.join(_SHADER_DIR, 'texquad.vert'), 'r', encoding='utf-8') as f:
                vertex_source = f.read()
            with open(os.path.join(_SHADER_DIR, 'texquad.frag'), 'r', encoding='utf-8') as f:
                fragment_source = f.read()
            program = self.ctx.program(
                vertex_shader=vertex_source, fragment_shader=fragment_source
            )
            program['u_texture'].value = 0

            quad = np.array(
                [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4'
            )
            self._quad_vbo = self.ctx.buffer(quad.tobytes())
            self._vao = self.ctx.vertex_array(
                program, [(self._quad_vbo, '2f', 'a_corner')]
            )
            self._program = program
        except Exception as exc:
            print(f"UI TEXT WARNING: texquad-pipeline nicht verfuegbar ({exc})")
            self._program = None
            self._vao = None

    # --------------------------------------------------------------- fonts

    def _resolve_font_paths(self):
        """Sucht die schriftdateien EINMAL, je familie und strichstaerke.

        Reihenfolge: mitgelieferte TTF in ui/assets/ -> systemschrift aus der
        familienliste des themes -> pygames eingebauter fallback (None).

        Zwei familien, zwei aufgaben (siehe theme.py, modulkopf):
        'display' = SB Liquid, die gefaste pixelschrift der instrumente;
        'text'    = Oxanium, die weich gerundete leseschrift.
        """
        if self._font_paths is not None:
            return self._font_paths

        pygame.font.init()

        def pick(families, bundled_names, bold):
            for name in bundled_names:
                candidate = os.path.join(_ASSET_DIR, name)
                if os.path.isfile(candidate):
                    return candidate
            for family in families:
                try:
                    path = pygame.font.match_font(family, bold=bold)
                except Exception:
                    path = None
                if path:
                    return path
            return None

        display = self.theme.font_family_display
        text = self.theme.font_family_text
        self._font_paths = {
            # Die pixelschrift hat keinen eigenen fetten schnitt und braucht
            # auch keinen: sie ist bereits ein solid-schnitt, synthetisches
            # fetten wuerde nur das pixelraster verschmieren.
            ('display', False): pick(display, ('ui-display.ttf', 'ui-mono.ttf'), False),
            ('display', True): pick(display, ('ui-display.ttf', 'ui-mono.ttf'), False),
            ('text', False): pick(text, ('ui-text.ttf', 'ui-sans.ttf'), False),
            ('text', True): pick(text, ('ui-text-bold.ttf', 'ui-text.ttf'), True),
        }
        return self._font_paths

    def _role(self, name):
        role = getattr(self.theme.type_scale, name, None)
        if role is None or not hasattr(role, 'size'):
            role = self.theme.type_scale.body
        return role

    #: Rasterstufe der pixelschrift und ihre kleinste brauchbare groesse.
    #: Beides gemessen, siehe _role_pixel_size.
    _PIXEL_STEP = 5
    _PIXEL_MIN = 10

    def _role_pixel_size(self, name):
        """Rollengroesse -> pixelgroesse, fuer die pixelschrift GERASTET.

        SB Liquid ist auf einem pixelraster gezeichnet. Bei einer beliebigen
        pixelgroesse fallen ihre stege unterschiedlich breit aus -- gemessen
        an 'HHIHIH1111' liefern die groessen 11 bis 13 stege von 1 UND 2 px
        nebeneinander, was die schrift trotz harter kanten unruhig macht.
        Auf vielfachen von fuenf (10/15/20/25/30/35/40/45/50) ist die
        stegbreite dagegen durchgaengig gleich. Darunter, bei 9 px, bleiben
        von einer versalie nur noch 6 px ink -- deshalb der boden bei 10.

        Die leseschrift wird NICHT gerastet: sie hat keine rasterbindung,
        und ein sprung von 5 px waere dort nur ein grober typo-sprung.
        """
        role = self._role(name)
        raw = role.size * self.ui_scale
        if role.family != 'display':
            return max(6, int(round(raw)))
        step = self._PIXEL_STEP
        return max(self._PIXEL_MIN, int(round(raw / step)) * step)

    def _role_antialias(self, name):
        """Die pixelschrift wird OHNE kantenglaettung gerastert.

        Gemessen ueber 'ABCDEFG0123456789': mit glaettung traegt SB Liquid
        bei jeder groesse einen halbdeckenden saum (6-22 % der pixel),
        ohne sie genau ZWEI alphawerte -- 0 und 255. Genau das ist der
        unterschied zwischen "pixelig" und "unscharf pixelig".
        """
        return self._role(name).family != 'display'

    def _role_tabular(self, name):
        """Die instrumentenschrift setzt ZIFFERN AUF FESTER BREITE.

        SB Liquid ist nicht dicktengleich: gemessen bei 15 px ist die '1'
        neun pixel breit, jede andere ziffer zehn. Bei einem rechtsbuendigen
        zaehler wandert damit die LINKE kante jedes mal, wenn eine '1' in
        die anzeige laeuft oder sie verlaesst -- der AP/PE-countdown zuckte
        so im sekundentakt.

        Behoben wird das wie in jeder echten schrift mit tabellenziffern:
        jede ziffer bekommt die breite der breitesten und wird darin
        zentriert. Nur ziffern -- die uebrigen zeichen behalten ihre
        natuerliche breite, denn sie stehen in einem festen format ohnehin
        immer an derselben stelle.
        """
        return self._role(name).family == 'display'

    def _digit_width(self, font):
        """Breite der breitesten ziffer dieser schrift, einmal gemessen."""
        key = id(font)
        width = self._digit_widths.get(key)
        if width is None:
            try:
                width = max(font.size(digit)[0] for digit in '0123456789')
            except Exception:
                width = 0
            self._digit_widths[key] = width
        return width

    def _rebuild_fonts(self):
        """Rastert alle rollen in der aktuellen skalierten pixelgroesse NEU.

        Fertige texturen zu strecken waere billiger, sieht beim hochskalieren
        aber unscharf aus -- deshalb neu rastern. Der textur-cache ist nach
        pixelgroesse verschluesselt und wird dabei ungueltig.
        """
        paths = self._resolve_font_paths()
        self._fonts = {}
        self._digit_widths = {}
        for name in dir(self.theme.type_scale):
            if name.startswith('_'):
                continue
            role = getattr(self.theme.type_scale, name, None)
            if role is None or not hasattr(role, 'size'):
                continue
            size_px = self._role_pixel_size(name)
            path = paths.get((role.family, role.bold))
            try:
                if path:
                    font = pygame.font.Font(path, size_px)
                else:
                    font = pygame.font.SysFont(None, size_px, bold=role.bold)
            except Exception:
                try:
                    font = pygame.font.SysFont(None, size_px, bold=role.bold)
                except Exception:
                    continue
            # Fand die suche keine EIGENE fette schnittdatei (der eintrag ist
            # derselbe wie fuer regular), synthetisch fetten -- sonst faellt
            # die gewichts-hierarchie in sich zusammen. Die PIXELSCHRIFT ist
            # davon ausgenommen: synthetisches fetten verbreitert ihre stege
            # um genau ein halbes pixel und zerstoert damit das raster.
            if (role.bold and role.family != 'display' and path is not None
                    and paths.get((role.family, False)) == path):
                try:
                    font.set_bold(True)
                except Exception:
                    pass
            self._fonts[name] = font
        self.clear_cache()

    def font(self, role='body'):
        font = self._fonts.get(role)
        if font is None:
            font = self._fonts.get('body')
        return font

    def line_height(self, role='body'):
        font = self.font(role)
        return float(font.get_height()) if font else float(self._role_pixel_size(role))

    def tracking_px(self, role='body'):
        """Laufweite dieser rolle in pixeln (em-wert * schriftgroesse)."""
        return self._role(role).tracking * self._role_pixel_size(role)

    # --------------------------------------------------------------- cache

    def clear_cache(self):
        for entry in list(self._cache.values()):
            try:
                entry[0].release()
            except Exception:
                pass
        self._cache = {}

    def _texture_for(self, text, role):
        font = self.font(role)
        if font is None:
            return None
        key = (text, role, font.get_height())
        entry = self._cache.get(key)
        if entry is not None:
            return entry
        antialias = self._role_antialias(role)
        try:
            # IMMER weiss rastern -- eingefaerbt wird im shader (u_color).
            surface = self._render_tracked(
                text, font, self.tracking_px(role), antialias,
                tabular=self._role_tabular(role),
            )
            data = pygame.image.tostring(surface, 'RGBA', True)
            w, h = surface.get_size()
            texture = self.ctx.texture((w, h), 4, data)
            # Die pixelschrift wird NEAREST gefiltert. _blit zeichnet zwar
            # 1:1 und auf ganze pixel gerastet, wo LINEAR dasselbe ergaebe --
            # aber eine harte rasterung, die von der genauigkeit der
            # texturkoordinaten abhaengt, ist ein unnoetiges risiko.
            mode = moderngl.LINEAR if antialias else moderngl.NEAREST
            texture.filter = (mode, mode)
        except Exception:
            return None

        # FIFO-deckel: staendig wechselnde texte (geschwindigkeits-anzeige,
        # timer) wuerden sonst unbegrenzt GL-texturen anhaeufen. Stabile
        # labels werden nach einer verdraengung einfach neu erzeugt.
        if len(self._cache) >= self.cache_max:
            for old_key in list(self._cache.keys())[: max(1, self.cache_max // 4)]:
                try:
                    self._cache.pop(old_key)[0].release()
                except Exception:
                    pass

        self._cache[key] = (texture, w, h)
        return self._cache[key]

    def _render_tracked(self, text, font, tracking, antialias=True,
                        tabular=False):
        """Rastert text, bei bedarf mit LAUFWEITE (letter-spacing).

        Die oberflaeche sperrt ihre beschriftungen stark (.16em bei
        abschnitts-titeln, .20em bei einheiten) -- ohne das sieht sie voellig
        anders aus. pygames font.render kann keine laufweite, also werden die
        zeichen einzeln gesetzt.

        Ohne sperrung laeuft weiter der normale einzel-render: er behaelt
        das kerning, das beim zeichenweisen setzen verloren geht. Bei
        gesperrter versal-beschriftung faellt kerning nicht auf, bei
        normalem fliesstext schon.

        antialias=False rastert hart -- so wird die pixelschrift gesetzt.
        pygame liefert dann eine palettierte flaeche; convert_alpha() ist
        ohne display-modus nicht sicher, deshalb wird sie ueber einen
        SRCALPHA-zwischenschritt mit farbschluessel transparent gemacht.

        tabular=True setzt ZIFFERN auf feste breite (siehe _role_tabular).
        """
        if abs(tracking) < 0.05 and not tabular:
            return self._render_plain(text, font, antialias)

        digit_width = self._digit_width(font) if tabular else 0
        glyphs = []
        total = 0.0
        height = font.get_height()
        for index, char in enumerate(text):
            glyph = self._render_plain(char, font, antialias)
            natural = font.size(char)[0]
            if tabular and char.isdigit() and digit_width > natural:
                # In der zelle ZENTRIEREN, nicht linksbuendig setzen: sonst
                # sitzt eine schmale '1' sichtbar links in ihrem feld.
                advance = digit_width
                offset = total + (digit_width - natural) * 0.5
            else:
                advance = natural
                offset = total
            # Nach dem LETZTEN zeichen keine sperrung: sonst hinge rechts
            # ein leerraum, der zentrierten text sichtbar nach links zieht.
            if index < len(text) - 1:
                advance += tracking
            glyphs.append((glyph, offset))
            total += advance

        surface = pygame.Surface(
            (max(1, int(math.ceil(total))), height), pygame.SRCALPHA
        )
        for glyph, offset in glyphs:
            surface.blit(glyph, (int(round(offset)), 0))
        return surface

    @staticmethod
    def _render_plain(text, font, antialias):
        """Ein render-aufruf, mit alphakanal auch im hart gerasterten fall.

        Ohne kantenglaettung liefert pygame eine 8-bit-palettenflaeche mit
        durchsichtigem hintergrund. Sie laesst sich blitten, aber
        image.tostring('RGBA') gaebe deckende schwarze pixel um jede glyphe.
        Der umweg ueber eine SRCALPHA-flaeche macht daraus einen echten
        alphakanal mit genau zwei werten -- 0 und 255.
        """
        if antialias:
            return font.render(text, True, (255, 255, 255))
        raw = font.render(text, False, (255, 255, 255), (0, 0, 0))
        surface = pygame.Surface(raw.get_size(), pygame.SRCALPHA)
        raw.set_colorkey((0, 0, 0))
        surface.blit(raw, (0, 0))
        return surface

    def measure(self, text, role='body'):
        """Groesse des gerenderten textes in pixeln, ohne zu zeichnen."""
        entry = self._texture_for(text, role)
        if entry is None:
            font = self.font(role)
            return (0.0, float(font.get_height()) if font else 0.0)
        return (float(entry[1]), float(entry[2]))

    # -------------------------------------------------------------- zeichnen

    def draw(self, text, x, y, role='body', color=(1.0, 1.0, 1.0, 1.0),
             align='left', valign='top'):
        """Zeichnet text an TOP-DOWN koordinaten.

        align:  'left' | 'center' | 'right'  -- bezogen auf x
        valign: 'top'  | 'middle' | 'bottom' -- bezogen auf y

        Gibt das belegte rechteck (x, y, w, h) in top-down koordinaten
        zurueck, damit aufrufer daraus trefferflaechen bauen koennen.
        """
        if not text:
            return (float(x), float(y), 0.0, 0.0)
        entry = self._texture_for(text, role)
        if entry is None:
            return (float(x), float(y), 0.0, 0.0)
        texture, w, h = entry

        left = float(x)
        if align == 'center':
            left -= w * 0.5
        elif align == 'right':
            left -= w

        top = float(y)
        if valign == 'middle':
            top -= h * 0.5
        elif valign == 'bottom':
            top -= h

        self._blit(texture, left, top, w, h, color)
        return (left, top, float(w), float(h))

    def defer(self, text, x, y, role='body', color=(1.0, 1.0, 1.0, 1.0),
              align='left', valign='top'):
        """Wie draw(), aber erst beim naechsten flush() ausgefuehrt.

        Der einzige zweck: alles, was WAEHREND des FXAA-passes anfaellt (etwa
        weltverankerte marker), darf nicht in das FXAA-FBO gezeichnet werden.
        flush() laeuft nach dem resolve.
        """
        self._deferred.append((text, x, y, role, color, align, valign))

    def flush(self):
        """Zeichnet und leert die aufgeschobene warteschlange."""
        if not self._deferred:
            return
        queued = self._deferred
        self._deferred = []
        for text, x, y, role, color, align, valign in queued:
            self.draw(text, x, y, role=role, color=color, align=align, valign=valign)

    def _blit(self, texture, left, top, w, h, color):
        """Top-down -> ortho und auf das pixelraster rasten."""
        if self._vao is None or self._program is None:
            return
        # Gesammelte UI-rechtecke zuerst zeichnen (UIDraw stapelt fuer einen
        # instanzierten draw) -- sonst laege dieser text UNTER flaechen, die
        # vor ihm angefordert wurden. Gesetzt von UIContext.
        rect_flush = getattr(self, 'rect_flush', None)
        if rect_flush is not None:
            rect_flush()
        # AUF DAS PIXELRASTER RASTEN, siehe modul-docstring. Die textur wird
        # 1:1 gezeichnet, deshalb genuegt das runden der ecke.
        ortho_x = round(float(left))
        ortho_y = round(float(self.height) - float(top) - float(h))
        self._program['u_rect'].value = (ortho_x, ortho_y, float(w), float(h))
        self._program['u_viewport'].value = (float(self.width), float(self.height))
        self._program['u_color'].value = (
            float(color[0]), float(color[1]), float(color[2]), float(color[3])
        )
        texture.use(location=0)
        self._vao.render(moderngl.TRIANGLE_STRIP)

    # --------------------------------------------------------------- resize

    def resize(self, width, height, ui_scale=None):
        self.width = int(width)
        self.height = int(height)
        if ui_scale is not None and abs(float(ui_scale) - self.ui_scale) > 1e-3:
            self.ui_scale = float(ui_scale)
            self._rebuild_fonts()

    def release(self):
        self.clear_cache()
        for obj in (self._vao, self._quad_vbo, self._program):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass
        self._vao = None
        self._quad_vbo = None
        self._program = None
