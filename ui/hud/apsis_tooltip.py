"""Der schwebezettel an einer Ap/Pe-raute auf der bahnlinie.

Die raute selbst und ihre abstands-fahne zeichnet der renderer, weil nur er
die zeitabhaengige frame-transformation kennt, die den marker auf der
gezeichneten linie haelt. Was dort FEHLTE, sind die beiden zahlen, die man
beim planen eines manoevers wirklich braucht: WANN das schiff dort ankommt
und WIE SCHNELL es dann ist. Beide gehoeren nicht dauerhaft ins bild -- vier
zusaetzliche zeilen ueber der bahn waeren genau die textwand, die das HUD
ersetzt hat --, also erscheinen sie beim ueberfahren der raute.

WARUM HIER UND NICHT IM RENDERER. Der zettel braucht drei dinge, die
allesamt in ui/ liegen und nicht in rendering.py: die hausschrift
(ui-text.ttf ueber die rolle 'tip_*'), den SDF-shader fuer seine flaeche
und die maus. Der renderer legt darum nur die schirmpositionen seiner
marker in `renderer.apsis_marker_hits` ab; getroffen wird hier.

DER ZETTEL VERBRAUCHT DIE MAUS NICHT (`blocks_mouse` bleibt False). Er sitzt
mitten im bild ueber der bahn, und ein element, das dort klicks abfaengt,
haette das anklicken von koerpern und den kameraschwenk unter sich
blockiert -- fuer eine reine anzeige. Die trefferpruefung laeuft deshalb in
update() gegen ctx.mouse_x/y statt ueber das eingabe-routing der UIRoot.
"""

import math

from ..core import Widget, ease
from ..theme import with_alpha
from .. import units

# Entwurfseinheiten, wie ueberall im HUD.

#: Abstand zwischen der unterkante der raute und der oberkante des zettels.
_GAP = 7.0
_PAD_X = 9.0
_PAD_Y = 7.0
_ROW_GAP = 3.0
#: Mindestabstand zwischen schluessel- und wertspalte.
_COLUMN_GAP = 14.0
#: Abstand, den der zettel zum fensterrand haelt.
_SCREEN_MARGIN = 8.0

#: Die farben der beiden rauten, WORTGLEICH zu Renderer._draw_apsis_markers.
#: Zettel und raute muessen dieselbe farbe tragen, sonst ist nicht zu sehen,
#: zu welchem der beiden marker er gehoert.
_APOAPSIS_COLOR = (0.45, 0.75, 1.0, 1.0)
_PERIAPSIS_COLOR = (1.0, 0.62, 0.25, 1.0)


class ApsisTooltip(Widget):
    """Zwei zeilen -- ankunftszeit und bahntempo -- unter der raute."""

    def __init__(self, telemetry, **kwargs):
        # Groesse null: der zettel steht an einer WELTposition, nicht an
        # einer ecke des fensters. Er bekommt sein rechteck jeden frame aus
        # der marker-position, nicht aus dem verankerungs-layout.
        kwargs.setdefault('size', (0.0, 0.0))
        kwargs.setdefault('z', 60)
        super().__init__(**kwargs)
        self.telemetry = telemetry
        self.blocks_mouse = False
        #: Der zuletzt getroffene marker. Bleibt ueber das ausblenden hinweg
        #: stehen -- sonst verschwaende die haelfte der bewegung an einem
        #: zettel ohne inhalt.
        self._marker = None
        self._t = 0.0

    # ---------------------------------------------------------------- daten

    def _renderer(self):
        return getattr(self.telemetry, 'renderer', None)

    def _hover_radius(self, ctx, marker_radius_px):
        """Trefferkreis um die raute, in echten pixeln.

        Grosszuegiger als die raute selbst: ihre 5 px sind auf 4 K-schirmen
        eine flaeche, die man mit der maus nicht zuverlaessig trifft.
        """
        renderer = self._renderer()
        extra = float(getattr(renderer, 'apsis_tooltip_hover_px', 14.0) or 0.0)
        return max(float(marker_radius_px), 0.0) + ctx.px(extra)

    def _pick(self, ctx):
        """Der marker unter dem zeiger, oder None.

        Bei zwei dicht beieinander liegenden rauten gewinnt die NAEHERE,
        nicht die erste der liste -- sonst haengt die auswahl an der
        reihenfolge, in der der predictor sie gefunden hat.
        """
        renderer = self._renderer()
        if renderer is None or not getattr(renderer, 'apsis_tooltip_enabled', False):
            return None
        # Liegt der zeiger auf einem bedienelement, gehoert er diesem. Ein
        # zettel, der unter dem navball-block hervorkommt, weil dort zufaellig
        # eine raute liegt, waere schlicht falsch.
        root = self.parent
        if getattr(root, 'hovered_widget', None) is not None:
            return None

        mouse_x = float(ctx.mouse_x)
        mouse_y = float(ctx.mouse_y)
        best = None
        best_distance = None
        for hit in getattr(renderer, 'apsis_marker_hits', ()) or ():
            try:
                sx, sy, radius_px, is_apo, distance_m, t_abs, alpha = hit
            except (TypeError, ValueError):
                continue
            if alpha <= 0.05:
                continue
            reach = self._hover_radius(ctx, radius_px)
            gap = math.hypot(mouse_x - sx, mouse_y - sy)
            if gap > reach:
                continue
            if best_distance is None or gap < best_distance:
                best_distance = gap
                best = {
                    'x': float(sx), 'y': float(sy),
                    'radius': float(radius_px),
                    'apoapsis': bool(is_apo),
                    'distance': float(distance_m),
                    't_abs': float(t_abs),
                }
        return best

    def _rows(self):
        """Die beiden zeilen als (schluessel, wert).

        ETA ist SIMULATIONSZEIT bis zur ankunft, nicht echtzeit: die
        raffungsstufe aendert, wie lange man darauf wartet, aber nicht, wann
        das schiff dort ist. Genau so lesen sich auch die AP/PE-countdowns
        im navball-block.
        """
        marker = self._marker
        if marker is None:
            return ()
        telemetry = self.telemetry
        try:
            now = float(getattr(telemetry.world, 'time', 0.0))
        except Exception:
            now = 0.0
        eta = marker['t_abs'] - now
        speed = telemetry.speed_at_radius(marker['distance'])
        return (
            ('ETA', units.duration(eta if eta > 0.0 else 0.0)),
            ('REL V', units.speed(speed)),
        )

    # -------------------------------------------------------------- ablauf

    def update(self, ctx, dt):
        super().update(ctx, dt)
        picked = self._pick(ctx)
        if picked is not None:
            self._marker = picked
        # motion.fast, nicht .normal: der zettel folgt dem zeiger, und alles,
        # was auf eine zeigerbewegung antwortet, muss sofort da sein. 22
        # entspricht ~0.14 s bis 95 %.
        self._t = ease(self._t, 1.0 if picked is not None else 0.0,
                       ctx.theme.motion.fast, dt)
        if picked is None and self._t < 0.004:
            self._t = 0.0
            self._marker = None

    # ------------------------------------------------------------ zeichnen

    def _flag_height(self):
        """Hoehe der abstands-fahne, die der renderer unter die raute setzt."""
        font = getattr(self._renderer(), 'font_small', None)
        try:
            return float(font.get_height()) + 4.0
        except Exception:
            return 20.0

    def _measure(self, ctx, rows):
        pad_x = ctx.px(_PAD_X)
        pad_y = ctx.px(_PAD_Y)
        gap = ctx.px(_ROW_GAP)
        key_w = 0.0
        value_w = 0.0
        height = 0.0
        for index, (key, value) in enumerate(rows):
            kw, kh = ctx.text.measure(key, 'tip_key')
            vw, vh = ctx.text.measure(value, 'tip_value')
            key_w = max(key_w, kw)
            value_w = max(value_w, vw)
            height += max(kh, vh)
            if index < len(rows) - 1:
                height += gap
        width = key_w + ctx.px(_COLUMN_GAP) + value_w + pad_x * 2.0
        return width, height + pad_y * 2.0

    def draw(self, ctx):
        if self._t <= 0.004 or self._marker is None:
            return
        rows = self._rows()
        if not rows:
            return

        palette = ctx.theme.palette
        marker = self._marker
        color = _APOAPSIS_COLOR if marker['apoapsis'] else _PERIAPSIS_COLOR
        fade = max(0.0, min(1.0, self._t))

        width, height = self._measure(ctx, rows)
        # UNTER der raute, wie gefordert -- und um die halbe breite nach
        # links, damit er auf ihr zentriert steht.
        #
        # Der abstand ist NICHT nur der zettel-spalt: direkt unter der raute
        # steht bereits die abstands-fahne des renderers ("Ap 16.37Mm", bei
        # sy + radius + 4 px, gesetzt in hud_font_size_small). Der zettel
        # muss unter IHR beginnen, sonst schreibt er darueber. Die hoehe der
        # fahne wird beim renderer erfragt statt geraten -- sie haengt an
        # dessen eigener ui_scale.
        top = marker['y'] + marker['radius'] + ctx.px(_GAP) + self._flag_height()
        left = marker['x'] - width * 0.5
        # Im bild halten: eine raute am fensterrand haette ihren zettel sonst
        # zur haelfte ausserhalb.
        margin = ctx.px(_SCREEN_MARGIN)
        left = max(margin, min(left, ctx.width - width - margin))
        if top + height > ctx.height - margin:
            # Kein platz darunter: dann darueber, aber nur dann.
            top = marker['y'] - marker['radius'] - ctx.px(_GAP) - height
        top = max(margin, top)

        # Die kleine bewegung beim erscheinen: der zettel faehrt zwei
        # entwurfseinheiten aus der raute heraus. Ohne sie erscheint er
        # einfach, und das liest sich als bildfehler statt als antwort.
        top += (1.0 - fade) * ctx.px(4.0)

        # SEHR SCHWACH GEFUELLT, ABER MIT SCHATTEN. Die vorgabe ist eine
        # geringe deckkraft -- ueber der hell gezeichneten bahnlinie waere
        # eine nur getoente flaeche aber unlesbar. Der schlagschatten
        # darunter dunkelt die linie ab, ohne dass der zettel selbst deckend
        # werden muss.
        cut = -ctx.px(5.0)
        ctx.draw.rect(
            left, top, width, height,
            fill=with_alpha(palette.panel_popup, 0.62 * fade),
            radius=(cut, 0.0, cut, 0.0),
            border_color=with_alpha(color, 0.42 * fade),
            border_width=ctx.theme.border_width,
            shadow=with_alpha(palette.shadow, 0.85 * fade),
            shadow_offset=(0.0, 0.0),
            shadow_softness=ctx.px(12.0),
        )
        # Der farbige steg an der linken kante bindet den zettel an die
        # raute -- dieselbe farbe, dieselbe kante, kein zweites zeichen noetig.
        ctx.draw.rect(left, top, max(1.0, ctx.px(2.0)), height,
                      fill=with_alpha(color, 0.85 * fade))

        pad_x = ctx.px(_PAD_X)
        gap = ctx.px(_ROW_GAP)
        y = top + ctx.px(_PAD_Y)
        for key, value in rows:
            row_h = max(ctx.text.measure(key, 'tip_key')[1],
                        ctx.text.measure(value, 'tip_value')[1])
            middle = y + row_h * 0.5
            ctx.text.draw(key, left + pad_x, middle, role='tip_key',
                          color=with_alpha(palette.text_dim, fade),
                          valign='middle')
            ctx.text.draw(value, left + width - pad_x, middle,
                          role='tip_value', color=with_alpha(color, fade),
                          align='right', valign='middle')
            y += row_h + gap
