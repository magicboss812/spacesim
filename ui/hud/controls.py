"""Die bedienelemente des HUDs: zeitraffer, bezugsrahmen, autopilot, zoom.

ALLE folgen derselben zustandsregel:

    aktiv    -> flaeche in der rollenfarbe (88 %), schrift in der
                gegenfarbe (ink_on)
    inaktiv  -> flaeche sehr schwach, schrift in der rollenfarbe
    gesperrt -> keine flaeche, sehr blasse schrift

Das ist bewusst EINE regel fuer alle knoepfe: sobald jeder knopf seinen
eigenen aktiv-stil erfindet, liest sich die leiste nicht mehr auf einen
blick.

WICHTIG -- ANZEIGE, NICHT ZUSTAND: jedes element liest seinen wert ueber
ein callable aus der simulation zurueck, statt ihn selbst zu halten. Nur so
stimmen HUD und tastatur ueberein, wenn dieselbe groesse ueber beide wege
verstellt wird (etwa PageUp/PageDown und der zeitraffer-knopf).

FORM: die bauteile aus chrome.py -- gefaste kanten, doppelrahmen,
notch-tabs. Kein element bringt eine eigene kontur mit.
"""

import math

from ..core import Widget
from ..theme import ink_on, mix, with_alpha
from . import chrome


def _button_colors(ctx, active, color, hover_t=0.0, press_t=0.0):
    """Die gemeinsame zustandsregel. Siehe modul-docstring."""
    palette = ctx.theme.palette
    if active:
        fill = with_alpha(color, 0.88)
        text = ink_on(color)
        border = with_alpha(color, 0.90)
    else:
        fill = palette.idle_fill
        text = with_alpha(color, 0.80)
        border = palette.edge
    fill = mix(fill, palette.hover, hover_t * 0.6)
    fill = mix(fill, palette.active, press_t * 0.6)
    return fill, text, border


class SegmentBar(Widget):
    """Waagerechte zellenleiste mit sich ausschliessenden optionen.

    Traegt den zeitraffer und die rahmenauswahl -- dieselbe form, andere
    rollenfarbe und andere beschriftung. Die zellen sind GEFASTE rechtecke
    mit sichtbarem spalt dazwischen, nicht pillen in einer wanne: eine
    zellenreihe liest sich als stufenschalter, eine pillenreihe als
    web-navigation.

    Die beschriftung reitet als notch-tab auf der oberkante.
    """

    def __init__(self, options, value, on_select, color_role='frame',
                 caption=None, role='button_sm', min_option_width=0.0,
                 pad_x=9, pad_y=7, gap=2, container_pad=5, enabled=None,
                 tab_edge='top', cumulative=False, **kwargs):
        kwargs.setdefault('size', (None, None))
        super().__init__(**kwargs)
        self.options = list(options)
        self.value = value
        self.on_select = on_select
        self.color_role = color_role
        self.caption = caption
        self.role = role
        self.min_option_width = float(min_option_width)
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.gap = gap
        self.container_pad = container_pad
        self.tab_edge = tab_edge
        # PEGEL statt auswahl: bei cumulative=True bekommen auch alle zellen
        # UNTERHALB der gewaehlten eine (schwaechere) fuellung. Genau so
        # zeigt die vorlage ihre raffung -- als reihe gruener winkel, die
        # bis zur aktuellen stufe reicht. Eine raffungsstufe IST ein pegel;
        # als radiogruppe gezeichnet sagt sie weniger, als sie weiss.
        self.cumulative = bool(cumulative)
        # Praedikat index -> bool. None heisst: alles erlaubt. Wird fuer den
        # zeitraffer benutzt, dessen obere stufen nahe an einem koerper die
        # bahn nicht mehr aufloesen (siehe Hud._warp_step_enabled).
        self.enabled = enabled
        self.blocks_mouse = True
        self._hover_index = -1

    def resolve_options(self):
        return list(self.options() if callable(self.options) else self.options)

    def resolve_value(self):
        return int(self.value() if callable(self.value) else self.value)

    def option_enabled(self, index):
        if self.enabled is None:
            return True
        try:
            return bool(self.enabled(index))
        except Exception:
            return True

    def tab_height(self, ctx):
        """Platz, den der notch-tab AUSSERHALB des rahmens braucht.

        Er wird in measure() mitgezaehlt und in _bar_rect() wieder abgezogen.
        Ohne diese reservierung ragte der tab in den nachbarn -- an der
        zeitraffer-leiste verdeckte er die beiden ersten stufen.
        """
        if not self.caption:
            return 0.0
        return ctx.text.measure(self.caption, 'tab')[1] + ctx.px(4)

    def _bar_rect(self, ctx):
        """Die flaeche des eigentlichen rahmens, ohne das tab-band."""
        tab_h = self.tab_height(ctx)
        if self.tab_edge == 'top':
            return (self.rect.x, self.rect.y + tab_h,
                    self.rect.w, self.rect.h - tab_h)
        return (self.rect.x, self.rect.y, self.rect.w, self.rect.h - tab_h)

    def _metrics(self, ctx):
        options = self.resolve_options()
        pad_x = ctx.px(self.pad_x)
        widths = []
        for option in options:
            width = ctx.text.measure(str(option), self.role)[0] + pad_x * 2.0
            widths.append(max(width, ctx.px(self.min_option_width)))
        height = ctx.text.measure('X', self.role)[1] + ctx.px(self.pad_y) * 2.0
        return widths, height

    def measure(self, ctx):
        widths, height = self._metrics(ctx)
        pad = ctx.px(self.container_pad)
        gap = ctx.px(self.gap)
        total = sum(widths) + gap * max(0, len(widths) - 1) + pad * 2.0
        return (total, height + pad * 2.0 + self.tab_height(ctx))

    def _option_rects(self, ctx):
        widths, height = self._metrics(ctx)
        pad = ctx.px(self.container_pad)
        gap = ctx.px(self.gap)
        bx, by, _bw, _bh = self._bar_rect(ctx)
        x = bx + pad
        y = by + pad
        rects = []
        for width in widths:
            rects.append((x, y, width, height))
            x += width + gap
        return rects

    def on_mouse_move(self, ctx, x, y):
        self._hover_index = -1
        for index, (bx, by, bw, bh) in enumerate(self._option_rects(ctx)):
            if bx <= x < bx + bw and by <= y < by + bh:
                self._hover_index = index
                break
        return True

    def on_mouse_up(self, ctx, x, y, button):
        if button != 1:
            return True
        for index, (bx, by, bw, bh) in enumerate(self._option_rects(ctx)):
            if bx <= x < bx + bw and by <= y < by + bh:
                if self.on_select is not None and self.option_enabled(index):
                    self.on_select(index)
                break
        return True

    def draw(self, ctx):
        palette = ctx.theme.palette
        color = palette.accent_for(self.color_role)
        fx, fy, fw, fh = self._bar_rect(ctx)
        chrome.frame(ctx, fx, fy, fw, fh, glow_role=self.color_role)

        selected = self.resolve_value()
        options = self.resolve_options()
        cut = -ctx.px(3.0)
        for index, (bx, by, bw, bh) in enumerate(self._option_rects(ctx)):
            active = index == selected
            usable = self.option_enabled(index)
            hover = 1.0 if (self.hovered and index == self._hover_index
                            and usable) else 0.0
            below = self.cumulative and index < selected
            fill, text_color, _ = _button_colors(ctx, active, color, hover)
            if not usable:
                # Gesperrt: keine flaeche, nur sehr blasse schrift -- der
                # knopf bleibt sichtbar (die stufe existiert ja), sagt aber
                # deutlich, dass er hier nicht zu haben ist.
                text_color = palette.text_dimmer
            elif below:
                ctx.draw.rect(bx, by, bw, bh, fill=with_alpha(color, 0.26),
                              radius=(cut, 0.0, cut, 0.0))
                text_color = with_alpha(color, 0.95)
            elif active or hover:
                # Die AKTIVE zelle wird nur oben links und unten rechts
                # gefast. Diagonal gegenueberliegende schnitte lesen sich
                # als richtung -- die zelle bekommt damit eine lage in der
                # reihe, nicht nur eine markierung.
                ctx.draw.rect(bx, by, bw, bh, fill=fill,
                              radius=(cut, 0.0, cut, 0.0))
            ctx.text.draw(
                str(options[index]) if index < len(options) else '',
                bx + bw * 0.5, by + bh * 0.5, role=self.role,
                color=text_color, align='center', valign='middle',
            )

        if self.caption:
            edge_y = fy if self.tab_edge == 'top' else fy + fh
            chrome.tab(ctx, self.caption, fx + ctx.px(14), edge_y,
                       color=color, edge=self.tab_edge)


class WarpBar(SegmentBar):
    """Der zeitraffer -- die zellenleiste plus die laufende missionszeit.

    Die uhr gehoert dazu und nicht woandershin: eine raffungsstufe ohne die
    zeit, die sie erzeugt, ist eine zahl ohne wirkung. Die vorlage stellt
    beides ebenfalls in einen block.
    """

    def __init__(self, telemetry, clock_height=22, **kwargs):
        super().__init__(**kwargs)
        self.telemetry = telemetry
        self.clock_height = float(clock_height)

    def measure(self, ctx):
        width, height = super().measure(ctx)
        return (width, height + ctx.px(self.clock_height))

    def _bar_rect(self, ctx):
        """Der rahmen endet ueber dem uhrstreifen."""
        x, y, w, h = super()._bar_rect(ctx)
        return (x, y, w, h - ctx.px(self.clock_height))

    def draw(self, ctx):
        palette = ctx.theme.palette
        clock_h = ctx.px(self.clock_height)
        super().draw(ctx)
        fx, fy, fw, fh = self._bar_rect(ctx)
        y = fy + fh
        chrome.plate(ctx, fx, y, fw, clock_h, fill=palette.panel_sunken,
                     corners=(False, False, True, True))
        middle = y + clock_h * 0.5
        pad = ctx.px(9)
        ctx.text.draw('UT', fx + pad, middle, role='caption',
                      color=palette.text_dimmer, valign='middle')
        ctx.text.draw(self.telemetry.text_mission_time(), fx + fw - pad,
                      middle, role='warp', color=palette.warp,
                      align='right', valign='middle')


class SnapRosette(Widget):
    """Der orientierungs-autopilot als rosette -- vier knoepfe um das schiff.

    Bildet exakt die tasten I / K / J / L ab und liest den aktiven modus aus
    schiffcontrol.snap_mode zurueck; tastatur und HUD koennen deshalb nicht
    auseinanderlaufen.

    Warum eine ROSETTE und kein 2x2-raster: die vier richtungen sind keine
    liste, sondern ein achsenkreuz. Prograde steht dem retrograden
    gegenueber, normal dem antinormalen -- im raster ist diese beziehung
    nicht ablesbar, in der rosette steht sie da. Die vorlage ordnet sie aus
    demselben grund so an.

    Die dritte raumachse der vorlage (radial in/out) fehlt: diese
    simulation ist zweidimensional, es gibt sie schlicht nicht.

    Die symbole werden GEZEICHNET, nicht gesetzt -- die zeichen der vorlage
    (U+25C9 / U+2297) fehlen in vielen schriften und erschienen als kaestchen.
    """

    #: (modus, kuerzel, winkel in kompassgrad, farbrolle)
    MODES = (
        ('prograde', 'PRO', 0.0, 'snap'),
        ('normal_in', 'NORM', 90.0, 'normal'),
        ('retrograde', 'RETRO', 180.0, 'snap'),
        ('antinormal_out', 'ANTI', 270.0, 'normal'),
    )

    SIZE = 132.0
    TILE = 34.0
    ORBIT = 38.0

    def __init__(self, telemetry, ship_control, compact=False, **kwargs):
        kwargs.setdefault('size', (None, None))
        super().__init__(**kwargs)
        self.telemetry = telemetry
        self.ship_control = ship_control
        self.compact = bool(compact)
        self.blocks_mouse = True
        self._hover_index = -1

    def _scale(self):
        return 0.76 if self.compact else 1.0

    def measure(self, ctx):
        size = ctx.px(self.SIZE * self._scale())
        return (size, size)

    def _tile_rect(self, ctx, index):
        scale = self._scale()
        tile = ctx.px(self.TILE * scale)
        orbit = ctx.px(self.ORBIT * scale)
        _mode, _label, compass, _role = self.MODES[index]
        cx, cy = chrome.polar(self.rect.center_x, self.rect.center_y,
                              orbit, compass)
        return (cx - tile * 0.5, cy - tile * 0.5, tile, tile)

    def on_mouse_move(self, ctx, x, y):
        self._hover_index = -1
        for index in range(len(self.MODES)):
            bx, by, bw, bh = self._tile_rect(ctx, index)
            if bx <= x < bx + bw and by <= y < by + bh:
                self._hover_index = index
                break
        return True

    def on_mouse_up(self, ctx, x, y, button):
        if button != 1 or self.ship_control is None:
            return True
        for index in range(len(self.MODES)):
            bx, by, bw, bh = self._tile_rect(ctx, index)
            if bx <= x < bx + bw and by <= y < by + bh:
                try:
                    self.ship_control.toggle_snap(self.MODES[index][0])
                except Exception:
                    pass
                break
        return True

    def draw(self, ctx):
        palette = ctx.theme.palette
        cx, cy = self.rect.center_x, self.rect.center_y
        scale = self._scale()

        # Der TRAeGER ist ein oktogon, kein rechteck: eine rosette in einer
        # kiste sieht aus wie ein raster, das man rund gestellt hat. Der
        # entwurf von Claude Design hatte hier eine viel zu grosse
        # hintergrundflaeche -- diese hier umschliesst die vier knoepfe
        # gerade eben.
        span_units = (self.ORBIT + self.TILE * 0.5 + 7.0) * scale * 2.0
        span = ctx.px(span_units)
        chrome.frame(ctx, cx - span * 0.5, cy - span * 0.5, span, span,
                     cut=span_units * 0.26, glow_role='snap')

        # Das schiff in der mitte -- der bezugspunkt, auf den sich die vier
        # richtungen beziehen.
        self._ship_glyph(ctx, cx, cy, ctx.px(11.0 * scale), palette.ship)

        active_mode = getattr(self.ship_control, 'snap_mode', None)
        for index, (mode, label, _compass, role) in enumerate(self.MODES):
            bx, by, bw, bh = self._tile_rect(ctx, index)
            active = mode == active_mode
            hover = 1.0 if (self.hovered and index == self._hover_index) else 0.0
            color = palette.accent_for(role)
            fill, text_color, border = _button_colors(ctx, active, color, hover)
            chrome.plate(ctx, bx, by, bw, bh, fill=fill, line=border,
                         cut=6.0 * scale)
            self._glyph(ctx, mode, bx + bw * 0.5, by + bh * 0.5,
                        ctx.px(7.0 * scale), text_color)

        if not self.compact:
            label = (dict((m, l) for m, l, _c, _r in self.MODES).get(active_mode)
                     if active_mode else 'FREE')
            chrome.tab(ctx, chrome.tab_text('SNAP', label),
                       cx, cy + span * 0.5,
                       color=palette.snap if active_mode else palette.text_dim,
                       align='center', edge='bottom')

    def _ship_glyph(self, ctx, x, y, radius, color):
        """Ein schlanker pfeil nach oben -- dieselbe silhouette wie die
        schiffsnase am ring."""
        width = max(1.0, ctx.px(1.6))
        ctx.draw.line(x, y - radius, x - radius * 0.52, y + radius * 0.72,
                      color, width=width, cap='round')
        ctx.draw.line(x, y - radius, x + radius * 0.52, y + radius * 0.72,
                      color, width=width, cap='round')
        ctx.draw.line(x - radius * 0.52, y + radius * 0.72,
                      x + radius * 0.52, y + radius * 0.72,
                      color, width=width, cap='round')

    def _glyph(self, ctx, mode, x, y, radius, color):
        width = max(1.0, ctx.px(1.5))
        if mode == 'prograde':
            ctx.draw.ring(x, y, radius, width, color)
            ctx.draw.circle(x, y, radius * 0.36, fill=color)
        elif mode == 'retrograde':
            ctx.draw.ring(x, y, radius, width, color)
            arm = radius * 0.60
            ctx.draw.line(x - arm, y - arm, x + arm, y + arm, color,
                          width=width, cap='round')
            ctx.draw.line(x - arm, y + arm, x + arm, y - arm, color,
                          width=width, cap='round')
        else:
            # Normal / antinormal: das dreieck der vorlage, mit der spitze
            # nach oben bzw. unten.
            up = mode == 'normal_in'
            tip = y - radius if up else y + radius
            base = y + radius * 0.72 if up else y - radius * 0.72
            ctx.draw.line(x - radius * 0.92, base, x + radius * 0.92, base,
                          color, width=width, cap='round')
            ctx.draw.line(x - radius * 0.92, base, x, tip, color,
                          width=width, cap='round')
            ctx.draw.line(x + radius * 0.92, base, x, tip, color,
                          width=width, cap='round')


class ZoomButtons(Widget):
    """SYSTEM / LOCAL -- zwei zoomstufen mit einem klick.

    Die stufen werden aus den ECHTEN koerperabstaenden gerechnet, nicht aus
    festen zahlen: SYSTEM rahmt den aeussersten koerper ein, LOCAL den
    bezugskoerper mitsamt schiff. Ein hart kodierter massstab waere beim
    ersten anderen system falsch.
    """

    def __init__(self, telemetry, camera, compact=False, **kwargs):
        kwargs.setdefault('size', (None, None))
        super().__init__(**kwargs)
        self.telemetry = telemetry
        self.camera = camera
        self.compact = bool(compact)
        self.blocks_mouse = True
        self._hover_index = -1
        self._mode = 'local'

    def measure(self, ctx):
        if self.compact:
            return (ctx.px(36) * 2.0 + ctx.px(4), ctx.px(36))
        return (ctx.px(168), ctx.px(34))

    def _button_rect(self, ctx, index):
        if self.compact:
            size = ctx.px(36)
            gap = ctx.px(4)
            return (self.rect.x + index * (size + gap), self.rect.y, size, size)
        gap = ctx.px(4)
        width = (self.rect.w - gap) * 0.5
        return (self.rect.x + index * (width + gap), self.rect.y,
                width, self.rect.h)

    def on_mouse_move(self, ctx, x, y):
        self._hover_index = -1
        for index in range(2):
            bx, by, bw, bh = self._button_rect(ctx, index)
            if bx <= x < bx + bw and by <= y < by + bh:
                self._hover_index = index
                break
        return True

    def on_mouse_up(self, ctx, x, y, button):
        if button != 1:
            return True
        for index in range(2):
            bx, by, bw, bh = self._button_rect(ctx, index)
            if bx <= x < bx + bw and by <= y < by + bh:
                self._apply_zoom('system' if index == 0 else 'local')
                break
        return True

    def _apply_zoom(self, mode):
        camera = self.camera
        telemetry = self.telemetry
        if camera is None:
            return
        self._mode = mode

        span = None
        bodies = getattr(telemetry.world, 'body', None) or []
        if mode == 'system':
            far = 0.0
            for candidate in bodies:
                try:
                    far = max(far, math.hypot(float(candidate.position.x),
                                              float(candidate.position.y)))
                except Exception:
                    continue
            span = far * 2.2
        else:
            reference = telemetry.ui_state.reference_body if telemetry.ui_state else None
            distance = telemetry.target_distance
            radius = float(getattr(reference, 'radius', 0.0) or 0.0)
            span = max((distance or 0.0) * 3.0, radius * 8.0)

        if not span or span <= 0.0 or not math.isfinite(span):
            return
        height = float(getattr(camera, 'height', 800) or 800)
        # Ziel-massstab setzen, NICHT den gezeichneten: camera.update() laesst
        # ihn weich nachlaufen, und der predictor bekommt nur das ziel zu
        # sehen (sonst ein synchroner neuaufbau pro animations-frame).
        try:
            camera.target_scale = max(
                float(getattr(camera, 'min_scale', 1e-30)),
                min(float(getattr(camera, 'max_scale', 1e10)), height / span),
            )
        except Exception:
            pass

    def draw(self, ctx):
        palette = ctx.theme.palette
        color = palette.orbit
        cut = -ctx.px(4.0)
        for index, label in enumerate(('SYSTEM', 'LOCAL')):
            bx, by, bw, bh = self._button_rect(ctx, index)
            active = (self._mode == ('system' if index == 0 else 'local'))
            hover = 1.0 if (self.hovered and index == self._hover_index) else 0.0
            fill, text_color, border = _button_colors(ctx, active, color, hover)
            # Aussenkante gefast, innenkante scharf -- die beiden knoepfe
            # lesen sich damit als EIN geteiltes bauteil.
            radius = ((cut, 0.0, 0.0, cut) if index == 0
                      else (0.0, cut, cut, 0.0))
            ctx.draw.rect(bx, by, bw, bh, fill=fill, radius=radius,
                          border_color=border,
                          border_width=ctx.theme.border_width)

            icon_r = ctx.px(6)
            if self.compact:
                icon_x, icon_y = bx + bw * 0.5, by + bh * 0.5
            else:
                text_w = ctx.text.measure(label, 'button')[0]
                block = icon_r * 2.0 + ctx.px(7) + text_w
                icon_x = bx + (bw - block) * 0.5 + icon_r
                icon_y = by + bh * 0.5
                ctx.text.draw(label, icon_x + icon_r + ctx.px(7), icon_y,
                              role='button', color=text_color, valign='middle')
            self._magnifier(ctx, icon_x, icon_y, icon_r, text_color,
                            plus=index == 1)

    def _magnifier(self, ctx, x, y, radius, color, plus):
        width = max(1.0, ctx.px(1.8))
        ctx.draw.ring(x, y, radius, width, color)
        ctx.draw.line(x + radius * 0.72, y + radius * 0.72,
                      x + radius * 1.5, y + radius * 1.5, color,
                      width=width, cap='round')
        ctx.draw.line(x - radius * 0.45, y, x + radius * 0.45, y, color,
                      width=width, cap='round')
        if plus:
            ctx.draw.line(x, y - radius * 0.45, x, y + radius * 0.45, color,
                          width=width, cap='round')
