"""Der schwebezettel an einer Ap/Pe-raute auf der bahnlinie.

Die raute selbst und ihre abstands-fahne zeichnet der renderer, weil nur er
die zeitabhaengige frame-transformation kennt, die den marker auf der
gezeichneten linie haelt. Der zettel ergaenzt die beiden zahlen, die man
beim planen eines manoevers wirklich braucht: WANN das schiff dort ankommt
und WIE SCHNELL es dann ist. Beide gehoeren nicht dauerhaft ins bild --
zusaetzliche zeilen ueber der bahn waeren eine textwand --, also erscheinen
sie beim ueberfahren der raute.

WARUM HIER UND NICHT IM RENDERER. Der zettel braucht drei dinge, die
allesamt in ui/ liegen und nicht im renderer: die hausschrift
(ui-text.ttf ueber die rolle 'tip_*'), den SDF-shader fuer seine flaeche
und die maus. Der renderer legt darum nur die schirmpositionen seiner
marker in `renderer.apsis_marker_hits` ab; getroffen wird hier.

DER ZETTEL VERBRAUCHT DIE MAUS NUR AUF SEINEM KNOPF. Er sitzt mitten im
bild ueber der bahn, und ein element, das dort klicks abfaengt, haette das
anklicken von koerpern und den kameraschwenk unter sich blockiert -- fuer
eine anzeige. Das ueberfahren laeuft deshalb in update() gegen
ctx.mouse_x/y; `hit_test` ist nur auf dem "WARP TO"-knopf eines
ANGEHEFTETEN zettels wahr.

ANHEFTEN. Ein klick auf eine raute heftet den zettel an
(`InputRouter.handle_world_click` -> `pin_at`); er folgt dann seinem marker
und schliesst, wenn der vergangen oder verschwunden ist, oder beim naechsten
klick irgendwo sonst (`dismiss`, von UIRoot bei jedem mausdruck gerufen).
Angeheftet traegt er "WARP TO" (runtime/warp_to.py): ziel ist die apsis
minus `warp_to.apsis_margin_s`. Nicht an rauten der GEPLANTEN bahn -- dort
ist das schiff nie, solange der plan nicht geflogen ist -- und abgeblendet,
wenn die marge schon angebrochen ist.

Die rauten der geplanten bahn (`renderer.preview_apsis_marker_hits`, Task 3)
trifft er genauso; ihr tempo kommt dann aus der vorschau selbst, nicht aus
der oskulierenden bahn von jetzt.
"""

import math

from ..core import Widget, ease
from ..theme import ink_on, with_alpha
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
#: Hoehe des "WARP TO"-knopfs und sein abstand zur letzten zeile.
_BUTTON_H = 20.0
_BUTTON_GAP = 6.0
#: So lange (echtzeit) darf der angeheftete marker fehlen, bevor der zettel
#: schliesst -- ein einzelner frame ohne marker (neue linie, kurz ausserhalb
#: des bildes) ist noch kein verschwinden.
_MISSING_GRACE_S = 0.5

#: Die farben der beiden rauten, WORTGLEICH zu Renderer._draw_apsis_marker_rows.
#: Zettel und raute muessen dieselbe farbe tragen, sonst ist nicht zu sehen,
#: zu welchem der beiden marker er gehoert.
_APOAPSIS_COLOR = (0.45, 0.75, 1.0, 1.0)
_PERIAPSIS_COLOR = (1.0, 0.62, 0.25, 1.0)
#: ... und die der GEPLANTEN bahn, wortgleich zu render/prediction.py
#: (_PREVIEW_AP_RGB / _PREVIEW_PE_RGB: 45 % zum gruen der planlinie).
_PREVIEW_MIX = 0.45
_PREVIEW_GREEN = (0.282, 0.851, 0.486)
_PREVIEW_AP_COLOR = tuple(a + (g - a) * _PREVIEW_MIX for a, g in
                          zip((0.45, 0.75, 1.0), _PREVIEW_GREEN)) + (1.0,)
_PREVIEW_PE_COLOR = tuple(a + (g - a) * _PREVIEW_MIX for a, g in
                          zip((1.0, 0.62, 0.25), _PREVIEW_GREEN)) + (1.0,)


def _hit_dict(hit):
    """Eine trefferzeile des renderers als woerterbuch, oder None.

    Sieben felder fuer die vorhersagelinie, neun fuer die geplante bahn
    (`..., True, v_rel`).
    """
    try:
        sx, sy, radius_px, is_apo, distance_m, t_abs, alpha = hit[:7]
    except (TypeError, ValueError):
        return None
    preview = len(hit) > 7 and bool(hit[7])
    speed = float(hit[8]) if len(hit) > 8 else math.nan
    return {
        'x': float(sx), 'y': float(sy),
        'radius': float(radius_px),
        'apoapsis': bool(is_apo),
        'distance': float(distance_m),
        't_abs': float(t_abs),
        'alpha': float(alpha),
        'preview': preview,
        'speed': speed,
    }


class ApsisTooltip(Widget):
    """Ankunftszeit und bahntempo unter der raute; angeheftet mit WARP TO."""

    #: Der zettel einer raute haengt UNTER ihrer abstands-fahne.
    HAS_FLAG = True

    def __init__(self, telemetry, **kwargs):
        # Groesse null: der zettel steht an einer WELTposition, nicht an
        # einer ecke des fensters. Er bekommt sein rechteck jeden frame aus
        # der marker-position, nicht aus dem verankerungs-layout.
        kwargs.setdefault('size', (0.0, 0.0))
        kwargs.setdefault('z', 60)
        super().__init__(**kwargs)
        self.telemetry = telemetry
        # Nur der knopf eines angehefteten zettels nimmt die maus (hit_test).
        self.blocks_mouse = True
        #: Der zuletzt getroffene marker. Bleibt ueber das ausblenden hinweg
        #: stehen -- sonst verschwaende die haelfte der bewegung an einem
        #: zettel ohne inhalt.
        self._marker = None
        self._t = 0.0
        self.pinned = False
        self._missing_s = 0.0
        self._ctx = None
        self._button = None
        self._button_hover = False
        self._stops_running = False

    # ---------------------------------------------------------------- daten

    def _renderer(self):
        return getattr(self.telemetry, 'renderer', None)

    def _now(self):
        try:
            return float(getattr(self.telemetry.world, 'time', 0.0))
        except Exception:
            return 0.0

    def _hover_radius(self, ctx, marker_radius_px):
        """Trefferkreis um die raute, in echten pixeln.

        Grosszuegiger als die raute selbst: ihre 5 px sind auf 4 K-schirmen
        eine flaeche, die man mit der maus nicht zuverlaessig trifft.
        """
        renderer = self._renderer()
        extra = float(getattr(renderer, 'apsis_tooltip_hover_px', 14.0) or 0.0)
        return max(float(marker_radius_px), 0.0) + ctx.px(extra)

    def _hits(self):
        renderer = self._renderer()
        if renderer is None or not getattr(renderer, 'apsis_tooltip_enabled', False):
            return []
        out = []
        for name in ('apsis_marker_hits', 'preview_apsis_marker_hits'):
            for hit in getattr(renderer, name, ()) or ():
                marker = _hit_dict(hit)
                if marker is not None and marker['alpha'] > 0.05:
                    out.append(marker)
        return out

    def _pick_at(self, ctx, x, y):
        """Die naechste raute im trefferkreis um (x, y), oder None.

        Bei zwei dicht beieinander liegenden rauten gewinnt die NAEHERE,
        nicht die erste der liste -- sonst haengt die auswahl an der
        reihenfolge, in der der predictor sie gefunden hat.
        """
        best = None
        best_distance = None
        for marker in self._hits():
            gap = math.hypot(float(x) - marker['x'], float(y) - marker['y'])
            if gap > self._hover_radius(ctx, marker['radius']):
                continue
            if best_distance is None or gap < best_distance:
                best_distance = gap
                best = marker
        return best

    def _pick(self, ctx):
        """Der marker unter dem zeiger (ueberfahren), oder None."""
        # Liegt der zeiger auf einem bedienelement, gehoert er diesem. Ein
        # zettel, der unter dem navball-block hervorkommt, weil dort zufaellig
        # eine raute liegt, waere schlicht falsch.
        root = self.parent
        hovered = getattr(root, 'hovered_widget', None)
        if hovered is not None and hovered is not self:
            return None
        return self._pick_at(ctx, ctx.mouse_x, ctx.mouse_y)

    # ------------------------------------------------------------ anheften

    def pin_at(self, x, y):
        """Klick in die welt bei (x, y): trifft er eine raute, anheften."""
        if self._ctx is None or not self.drawn:
            return False
        marker = self._pick_at(self._ctx, x, y)
        if marker is None:
            return False
        self._marker = marker
        self.pinned = True
        self._missing_s = 0.0
        return True

    def unpin(self):
        self.pinned = False
        self._missing_s = 0.0

    def dismiss(self):
        """Irgendwo anders gedrueckt -- der angeheftete zettel schliesst."""
        self.unpin()

    def _follow(self, dt):
        """Den angehefteten marker im neuen frame wiederfinden.

        Gleiche art (Ap/Pe, linie/plan), zeitlich naechster -- die zeit eines
        markers wandert unter schub ein wenig, eine andere bahn verschiebt
        sie um mehr als die toleranz, und dann ist es nicht mehr derselbe.
        """
        pinned = self._marker
        now = self._now()
        if pinned is None or pinned['t_abs'] <= now:
            self.unpin()
            return
        tolerance = max(120.0, 0.02 * (pinned['t_abs'] - now))
        best = None
        for marker in self._hits():
            if (marker['apoapsis'] != pinned['apoapsis']
                    or marker['preview'] != pinned['preview']):
                continue
            gap = abs(marker['t_abs'] - pinned['t_abs'])
            if gap <= tolerance and (best is None or gap < best[0]):
                best = (gap, marker)
        if best is None:
            self._missing_s += dt
            if self._missing_s > _MISSING_GRACE_S:
                self.unpin()
            return
        self._missing_s = 0.0
        self._marker = best[1]

    # ------------------------------------------------------------ warp to

    def _warp_to(self):
        return getattr(self.telemetry, 'warp_to', None)

    def warp_offered(self):
        """Der knopf steht nur am angehefteten zettel einer echten raute."""
        marker = self._marker
        return (self.pinned and marker is not None and not marker['preview']
                and self._warp_to() is not None)

    def warp_target(self):
        warp = self._warp_to()
        margin = float(getattr(warp, 'apsis_margin_s', 0.0)) if warp else 0.0
        return self._marker['t_abs'] - margin

    def warp_label(self):
        return 'Ap' if self._marker and self._marker['apoapsis'] else 'Pe'

    def warp_enabled(self):
        return self.warp_offered() and self.warp_target() > self._now() + 1e-6

    def warp_running(self):
        warp = self._warp_to()
        return (warp is not None and warp.active and self._marker is not None
                and abs(float(warp.target) - self.warp_target()) < 1e-3)

    def hit_test(self, ctx, x, y):
        if not self.visible or self._button is None or not self.warp_offered():
            return False
        bx, by, bw, bh = self._button
        return bx <= x < bx + bw and by <= y < by + bh

    def on_mouse_move(self, ctx, x, y):
        self._button_hover = self.hit_test(ctx, x, y)
        return True

    def on_mouse_down(self, ctx, x, y, button):
        # VOR dem abbruch der hauptschleife (jeder klick bricht "WARP TO"
        # ab): laeuft schon die raffung zu GENAU diesem ziel, ist der klick
        # ein STOPP und darf sie beim loslassen nicht neu starten.
        self._stops_running = self.warp_running()
        return True

    def on_mouse_up(self, ctx, x, y, button):
        if button != 1 or not self.hit_test(ctx, x, y):
            return True
        if self._stops_running or not self.warp_enabled():
            return True
        self._warp_to().start(self.warp_target(), self._now(),
                              label=self.warp_label())
        return True

    def _rows(self):
        """Die zeilen als (schluessel, wert).

        ETA ist SIMULATIONSZEIT bis zur ankunft, nicht echtzeit: die
        raffungsstufe aendert, wie lange man darauf wartet, aber nicht, wann
        das schiff dort ist. Genau so lesen sich auch die AP/PE-countdowns
        im navball-block.
        """
        marker = self._marker
        if marker is None:
            return ()
        eta = marker['t_abs'] - self._now()
        speed = marker.get('speed', math.nan)
        if not marker.get('preview') or not math.isfinite(speed):
            speed = self.telemetry.speed_at_radius(marker['distance'])
        return (
            ('ETA', units.duration(eta if eta > 0.0 else 0.0)),
            ('REL V', units.speed(speed)),
        )

    def _color(self, ctx):
        marker = self._marker
        if marker.get('preview'):
            return _PREVIEW_AP_COLOR if marker['apoapsis'] else _PREVIEW_PE_COLOR
        return _APOAPSIS_COLOR if marker['apoapsis'] else _PERIAPSIS_COLOR

    # -------------------------------------------------------------- ablauf

    def update(self, ctx, dt):
        super().update(ctx, dt)
        self._ctx = ctx
        if self.pinned:
            self._follow(dt)
        shown = self.pinned
        if not self.pinned:
            picked = self._pick(ctx)
            if picked is not None:
                self._marker = picked
                shown = True
        # motion.fast, nicht .normal: der zettel folgt dem zeiger, und alles,
        # was auf eine zeigerbewegung antwortet, muss sofort da sein. 22
        # entspricht ~0.14 s bis 95 %.
        self._t = ease(self._t, 1.0 if shown else 0.0,
                       ctx.theme.motion.fast, dt)
        if not shown and self._t < 0.004:
            self._t = 0.0
            self._marker = None
        if not self.warp_offered():
            self._button = None

    # ------------------------------------------------------------ zeichnen

    def _flag_height(self):
        """Hoehe der abstands-fahne, die der renderer unter die raute setzt."""
        if not self.HAS_FLAG:
            return 0.0
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
        if self.warp_offered():
            width = max(width, ctx.text.measure('WARP TO', 'button_sm')[0]
                        + pad_x * 4.0)
            height += ctx.px(_BUTTON_GAP + _BUTTON_H)
        return width, height + pad_y * 2.0

    def _draw_button(self, ctx, left, width, bottom, color, fade):
        """Der "WARP TO"-knopf in der zustandsregel der HUD-knoepfe
        (ui/hud/controls.py): aktiv gefuellt, sonst schwach, gesperrt blass."""
        palette = ctx.theme.palette
        pad_x = ctx.px(_PAD_X)
        h = ctx.px(_BUTTON_H)
        x = left + pad_x
        w = width - pad_x * 2.0
        y = bottom - ctx.px(_PAD_Y) - h
        self._button = (x, y, w, h)
        running = self.warp_running()
        enabled = self.warp_enabled()
        cut = -ctx.px(4.0)
        if running:
            fill = with_alpha(color, 0.88 * fade)
            text_color = with_alpha(ink_on(color), fade)
            label = 'WARPING'
        elif enabled:
            hover = 1.0 if self._button_hover else 0.0
            fill = with_alpha(color, (0.16 + 0.22 * hover) * fade)
            text_color = with_alpha(color, fade)
            label = 'WARP TO'
        else:
            fill = with_alpha(palette.disabled, 0.6 * fade)
            text_color = with_alpha(palette.text_dimmer, fade)
            label = 'WARP TO'
        ctx.draw.rect(x, y, w, h, fill=fill, radius=(cut, 0.0, cut, 0.0),
                      border_color=with_alpha(color, (0.55 if enabled else 0.2) * fade),
                      border_width=ctx.theme.border_width)
        ctx.text.draw(label, x + w * 0.5, y + h * 0.5, role='button_sm',
                      color=text_color, align='center', valign='middle')

    def _anchor(self, ctx, width, height):
        """Linke obere ecke des zettels in pixeln."""
        marker = self._marker
        # UNTER der raute -- und um die halbe breite nach links, damit er auf
        # ihr zentriert steht.
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
        return left, max(margin, top)

    def _refresh_position(self):
        """Die schirmposition aus den treffern DIESES bildes nachziehen.

        update() laeuft vor `renderer.render()`, sieht also die treffer des
        VORIGEN bildes; gezeichnet wird nach render(). Bei verfolgter kamera
        im zeitraffer wandert die ganze szene je bild um den weg des
        schiffs (1 d/s: ein viertel umlauf), und ein zettel mit der
        position von vorhin stuende sichtbar daneben. Gesucht wird derselbe
        marker (art, linie/plan, zeit auf eine sekunde).
        """
        marker = self._marker
        for hit in self._hits():
            if (hit['apoapsis'] == marker['apoapsis']
                    and hit['preview'] == marker['preview']
                    and abs(hit['t_abs'] - marker['t_abs']) < 1.0):
                marker['x'] = hit['x']
                marker['y'] = hit['y']
                return

    def draw(self, ctx):
        if self._t <= 0.004 or self._marker is None:
            return
        self._refresh_position()
        rows = self._rows()
        if not rows:
            return

        palette = ctx.theme.palette
        color = self._color(ctx)
        fade = max(0.0, min(1.0, self._t))

        width, height = self._measure(ctx, rows)
        left, top = self._anchor(ctx, width, height)

        # Die kleine bewegung beim erscheinen: der zettel faehrt zwei
        # entwurfseinheiten aus der raute heraus. Ohne sie erscheint er
        # einfach, und das liest sich als bildfehler statt als antwort.
        top += (1.0 - fade) * ctx.px(4.0)

        # SEHR SCHWACH GEFUELLT, ABER MIT SCHATTEN. Der zettel soll kaum
        # decken -- ueber der hell gezeichneten bahnlinie waere eine nur
        # getoente flaeche aber unlesbar. Der schlagschatten
        # darunter dunkelt die linie ab, ohne dass der zettel selbst deckend
        # werden muss. ANGEHEFTET deckt er etwas mehr und traegt einen
        # kraeftigeren rahmen: er bleibt stehen und wird bedient.
        cut = -ctx.px(5.0)
        ctx.draw.rect(
            left, top, width, height,
            fill=with_alpha(palette.panel_popup,
                            (0.80 if self.pinned else 0.62) * fade),
            radius=(cut, 0.0, cut, 0.0),
            border_color=with_alpha(color, (0.70 if self.pinned else 0.42) * fade),
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

        if self.warp_offered():
            self._draw_button(ctx, left, width, top + height, color, fade)
