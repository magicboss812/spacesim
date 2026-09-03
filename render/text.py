"""Schriften und der label-textur-cache des Renderers.

Das spieler-HUD hat mit `ui/text.py` seinen eigenen, gleich gebauten cache --
die beiden schichten teilen bewusst nichts ausser der skala (`ui_scale`).
"""
import math
import os

import moderngl
import pygame


class TextMixin:
    """Schriften, der label-textur-cache und getoentes blitten.

    UI-groessen sind DESIGN-EINHEITEN, nie pixel: `ui_px()` rechnet sie ueber
    die fensterhoehe um. Ein literaler pixelwert in widget-code ist bei jeder
    aufloesung ausser der referenz ein fehler."""

    def _recompute_ui_scale(self):
        """Leitet ui_scale aus der fensterhöhe ab. Gibt True bei änderung zurück.

        Skaliert nach höhe (nicht nach fläche oder breite): breite fenster
        sollen die oberfläche nicht aufblähen, hohe fenster schon. Das ist die
        übliche konvention für auflösungsunabhängige spiel-UIs.
        """
        reference = max(float(getattr(self, 'ui_scale_reference_height', 1000.0)), 1.0)
        raw = float(self.height) / reference
        lo = float(getattr(self, 'ui_scale_min', 1.0))
        hi = float(getattr(self, 'ui_scale_max', 3.0))
        scale = max(lo, min(hi, raw)) * float(getattr(self, 'ui_scale_user', 1.0))
        scale = max(0.1, scale)

        previous = getattr(self, 'ui_scale', None)
        self.ui_scale = scale
        # Winzige schwankungen ignorieren: ein neuaufbau aller fonts und ein
        # leeren des textur-caches pro pixel fenster-höhe wäre verschwendung.
        return previous is None or abs(scale - previous) > 1e-3

    def ui_px(self, design_units):
        """Design-einheiten -> tatsächliche pixel bei aktueller ui_scale."""
        return float(design_units) * self.ui_scale

    def _rebuild_fonts(self):
        """Erzeugt die HUD-fonts in der aktuell skalierten pixelgröße neu.

        Neu rastern statt die fertigen texturen zu strecken: gestreckte
        glyphen werden beim hochskalieren unscharf. Der label-textur-cache
        wird dabei ungültig (er ist nach schrifthöhe verschlüsselt) und muss
        geleert werden.
        """
        small_px = max(6, int(round(self.ui_px(self.hud_font_size_small))))
        medium_px = max(6, int(round(self.ui_px(self.hud_font_size_medium))))
        try:
            pygame.font.init()
            self.font_small = pygame.font.SysFont(None, small_px)
            self.font_medium = pygame.font.SysFont(None, medium_px)
        except Exception as exc:
            print(f"RENDERER WARNING: HUD-fonts konnten nicht erzeugt werden ({exc})")
            return
        self.font_body_label = self._build_body_label_font()
        self._clear_text_caches()

    def _build_body_label_font(self):
        """SB Liquid in der gerasteten groesse -- die schrift der koerpernamen.

        Dieselbe datei, die auch das HUD benutzt (ui/assets/ui-display.ttf).
        Gefunden wird sie ueber den pfad, nicht ueber den familiennamen: eine
        mitgelieferte schrift ist nicht installiert, und pygame.font.match_font
        sieht nur installierte.

        Faellt die datei aus, bleibt es bei font_small. Ein koerpername ohne
        schrift waere schlimmer als einer in der falschen.
        """
        raw = self.ui_px(self.hud_font_size_body_label)
        step = self._DISPLAY_PIXEL_STEP
        size_px = max(self._DISPLAY_PIXEL_MIN, int(round(raw / step)) * step)
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'ui', 'assets',
            'ui-display.ttf',
        )
        if not os.path.isfile(path):
            return self.font_small
        try:
            return pygame.font.Font(path, size_px)
        except Exception as exc:
            print(f"RENDERER WARNING: koerpernamen-schrift nicht ladbar ({exc})")
            return self.font_small

    def _clear_text_caches(self):
        """Gibt alle text-abhängigen GL-texturen frei (font- oder größenwechsel)."""
        for entry in list(getattr(self, '_label_texture_cache', {}).values()):
            texture = entry[0]
            if texture is not None:
                try:
                    texture.release()
                except Exception:
                    pass
        self._label_texture_cache = {}
        # Der pool haelt texturen der ALTEN schriftgroesse -- nach einem
        # font-wechsel passt keine davon mehr, also mit weg.
        for bucket in getattr(self, '_label_texture_pool', {}).values():
            for texture in bucket:
                try:
                    texture.release()
                except Exception:
                    pass
        self._label_texture_pool = {}
        self._label_texture_pool_count = 0
        self._hud_line_surface_cache = {}
        self._hud_cache_key = None

    def set_hud_font_sizes(self, small=None, medium=None, body_label=None):
        """Setzt die DESIGN-schriftgrößen und baut die fonts neu auf."""
        if small is not None:
            self.hud_font_size_small = int(small)
        if medium is not None:
            self.hud_font_size_medium = int(medium)
        if body_label is not None:
            self.hud_font_size_body_label = int(body_label)
        self._rebuild_fonts()

    def set_ui_scale_user(self, factor):
        """Benutzer-skalenfaktor (multiplikativ auf die automatische skala)."""
        self.ui_scale_user = max(0.1, float(factor))
        if self._recompute_ui_scale():
            self._rebuild_fonts()

    def _acquire_label_texture(self, size, data):
        """Beschriftungs-textur besorgen -- moeglichst eine wiederverwendete.

        Wie in ui/text.py: der teure teil einer neuen beschriftung ist die
        GL-allokation, nicht das rastern. Die apsis-labels tragen eine
        entfernung im text und wechseln damit in JEDEM frame; ihre
        verdraengten texturen haben aber immer wieder dieselbe groesse und
        werden deshalb eingesammelt und per `write()` neu befuellt.
        """
        bucket = self._label_texture_pool.get(size)
        if bucket:
            texture = bucket.pop()
            self._label_texture_pool_count -= 1
            try:
                texture.write(data)
                texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                return texture
            except Exception:
                try:
                    texture.release()
                except Exception:
                    pass
        texture = self.ctx.texture(size, 4, data)
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return texture

    def _retire_label_texture(self, texture, size):
        if texture is None:
            return
        if self._label_texture_pool_count < self._LABEL_TEXTURE_POOL_MAX:
            self._label_texture_pool.setdefault(size, []).append(texture)
            self._label_texture_pool_count += 1
            return
        try:
            texture.release()
        except Exception:
            pass

    @staticmethod
    def _render_label_surface(text, font, antialias=True, tracking=0.0):
        """Rastert eine beschriftung -- bei bedarf HART und GESPERRT.

        Zwei zugestaendnisse an die hausschrift, beide aus ui/text.py
        uebernommen und dort gemessen:

        - `antialias=False` rastert ohne kantenglaettung. SB Liquid traegt
          mit glaettung bei jeder groesse einen halbdeckenden saum; ohne sie
          genau zwei alphawerte. pygame liefert dann eine palettierte
          flaeche, die ueber einen SRCALPHA-zwischenschritt mit farbschluessel
          erst zu einem echten alphakanal wird -- `image.tostring('RGBA')`
          gaebe sonst deckende schwarze pixel um jede glyphe.
        - `tracking` sperrt die zeichen. pygames font.render kann keine
          laufweite, also werden die glyphen einzeln gesetzt. Nach dem
          LETZTEN zeichen wird nicht gesperrt, sonst haengt rechts ein
          leerraum, der zentrierten text sichtbar nach links zieht.
        """
        def render_one(chunk):
            if antialias:
                return font.render(chunk, True, (255, 255, 255))
            raw = font.render(chunk, False, (255, 255, 255), (0, 0, 0))
            surface = pygame.Surface(raw.get_size(), pygame.SRCALPHA)
            raw.set_colorkey((0, 0, 0))
            surface.blit(raw, (0, 0))
            return surface

        if abs(float(tracking)) < 0.05 or not text:
            return render_one(text)

        glyphs = []
        cursor = 0.0
        for index, char in enumerate(text):
            glyphs.append((render_one(char), cursor))
            cursor += font.size(char)[0]
            if index < len(text) - 1:
                cursor += float(tracking)
        surface = pygame.Surface(
            (max(1, int(math.ceil(cursor))), font.get_height()), pygame.SRCALPHA
        )
        for glyph, offset in glyphs:
            surface.blit(glyph, (int(round(offset)), 0))
        return surface

    def _get_label_texture(self, text, font, antialias=True, tracking=0.0):
        # DIE SCHRIFT GEHOERT IN DEN SCHLUESSEL, NICHT NUR IHRE HOEHE. Seit
        # die koerpernamen ueber eine zweite schriftdatei laufen, koennen
        # zwei fonts dieselbe hoehe melden -- der cache haette dann die
        # gerasterten glyphen der einen unter dem namen der anderen
        # ausgeliefert.
        key = (text, id(font), font.get_height(), bool(antialias),
               round(float(tracking), 2))
        entry = self._label_texture_cache.get(key)
        if entry:
            return entry  # (texture, w, h)
        try:
            surface = self._render_label_surface(
                text, font, antialias=antialias, tracking=tracking)
            texture_data = pygame.image.tostring(surface, 'RGBA', True)
            w, h = surface.get_size()
            # cache deckeln (FIFO): ständig wechselnde texte (speed-label)
            # würden sonst unbegrenzt GL-texturen anhäufen. Stabile labels
            # (körpernamen) werden nach einer eviction einfach neu erzeugt.
            # Die verdraengten TEXTUREN wandern in den pool, statt dass
            # jedes frame neue angelegt werden.
            if len(self._label_texture_cache) >= self._label_texture_cache_max:
                evict_n = max(1, self._label_texture_cache_max // 4)
                for old_key in list(self._label_texture_cache.keys())[:evict_n]:
                    old = self._label_texture_cache.pop(old_key)
                    self._retire_label_texture(old[0], (int(old[1]), int(old[2])))
            texture = self._acquire_label_texture((w, h), texture_data)
            self._label_texture_cache[key] = (texture, w, h)
            return (texture, w, h)
        except Exception:
            return None

    def _blit_text_topdown(self, text, x_left, y_top, font, color=(1.0, 1.0, 1.0, 1.0),
                           antialias=True, tracking=0.0):
        """Text an TOP-DOWN koordinaten zeichnen (x = links, y = oberkante).

        Nimmt dem aufrufer die ortho-umrechnung ab: _draw_texture_ortho
        erwartet die UNTERE linke ecke in ortho-Y. `color` toent multiplikativ
        (texquad.frag) -- der alphakanal blendet den text aus.
        """
        entry = self._get_label_texture(text, font, antialias=antialias,
                                        tracking=tracking)
        text_h = float(entry[2]) if entry else float(font.get_height())
        self._blit_cached_text(text, x_left, self._ortho_y(y_top) - text_h, font,
                               color=color, antialias=antialias,
                               tracking=tracking)

    def _blit_cached_text(self, text, x, y, font, color=(1.0, 1.0, 1.0, 1.0),
                          antialias=True, tracking=0.0):
        entry = self._get_label_texture(text, font, antialias=antialias,
                                        tracking=tracking)
        if not entry:
            # fallback: one-shot-textur ohne cache erzeugen, zeichnen, freigeben
            try:
                surface = self._render_label_surface(
                    text, font, antialias=antialias, tracking=tracking)
                texture_data = pygame.image.tostring(surface, 'RGBA', True)
                w, h = surface.get_size()
                texture = self.ctx.texture((w, h), 4, texture_data)
                texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self._draw_texture_ortho(texture, x, y, w, h, color=color)
                texture.release()
            except Exception:
                pass
            return
        texture, w, h = entry
        self._draw_texture_ortho(texture, x, y, w, h, color=color)

    def _draw_body_label(self, name, screen_pos, radius):
        # Label mit gecachten GL-Texturen zeichnen, um pro-Frame GL-Allocationen zu vermeiden.
        # Label horizontal zentrieren und über dem Körper platzieren, um
        # Fehlausrichtungen beim Zoomen oder bei Radiusänderungen zu vermeiden.
        text, font, antialias, tracking = self._body_label_style(name)
        try:
            entry = self._get_label_texture(text, font, antialias=antialias,
                                            tracking=tracking)
            if entry:
                _, w, h = entry
                label_x = float(screen_pos[0]) - (float(w) / 2.0)
                # screen_pos ist TOP-DOWN; ueber dem koerper heisst kleineres y.
                label_y = float(screen_pos[1]) - float(radius) - 6.0 - float(h)
                self._blit_text_topdown(text, label_x, label_y, font,
                                        antialias=antialias, tracking=tracking)
                return
        except Exception:
            pass

        # Fallback: previous heuristic
        label_x = screen_pos[0] + radius + 2
        label_y = screen_pos[1] - 8
        self._blit_text_topdown(text, label_x, label_y, font,
                                antialias=antialias, tracking=tracking)
