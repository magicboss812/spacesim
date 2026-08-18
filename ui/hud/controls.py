"""Die bedienelemente des HUDs.

Zeitraffer, bezugsrahmen, orientierungs-autopilot, schub, zoom und die
palettenauswahl. Alle folgen derselben zustandsregel aus dem entwurf:

    aktiv    -> flaeche in der rollenfarbe (88 %), schrift in der
                gegenfarbe (ink_on), farbiger schein
    inaktiv  -> flaeche weiss auf 5 %, schrift in der rollenfarbe (78 %)

Das ist bewusst EINE regel fuer alle knoepfe: sobald jeder knopf seinen
eigenen aktiv-stil erfindet, liest sich die leiste nicht mehr auf einen
blick.

WICHTIG -- ANZEIGE, NICHT ZUSTAND: jedes element liest seinen wert ueber
ein callable aus der simulation zurueck, statt ihn selbst zu halten. Nur so
stimmen HUD und tastatur ueberein, wenn dieselbe groesse ueber beide wege
verstellt wird (etwa PageUp/PageDown und der zeitraffer-knopf).
"""

import math

from ..core import Widget
from ..theme import ink_on, mix, rgba, with_alpha


def _button_colors(ctx, active, color, hover_t=0.0, press_t=0.0):
    """Die gemeinsame zustandsregel. Siehe modul-docstring."""
    palette = ctx.theme.palette
    if active:
        fill = with_alpha(color, 0.88)
        text = ink_on(color)
        border = with_alpha(color, 0.90)
    else:
        fill = palette.idle_fill
        text = with_alpha(color, 0.78)
        border = palette.edge
    fill = mix(fill, palette.hover, hover_t * 0.6)
    fill = mix(fill, palette.active, press_t * 0.6)
    return fill, text, border


class SegmentBar(Widget):
    """Waagerechte pillenleiste mit sich ausschliessenden optionen.

    Traegt im entwurf sowohl den zeitraffer als auch die rahmenauswahl --
    dieselbe form, andere rollenfarbe und andere beschriftung.
    """

    def __init__(self, options, value, on_select, color_role='frame',
                 caption=None, role='button_sm', min_option_width=0.0,
                 pad_x=14, pad_y=8, gap=5, container_pad=5, enabled=None,
                 **kwargs):
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

    def _metrics(self, ctx):
        options = self.resolve_options()
        pad_x = ctx.px(self.pad_x)
        widths = []
        for option in options:
            width = ctx.text.measure(str(option), self.role)[0] + pad_x * 2.0
            widths.append(max(width, ctx.px(self.min_option_width)))
        height = ctx.text.measure('X', self.role)[1] + ctx.px(self.pad_y) * 2.0
        caption_w = 0.0
        if self.caption:
            caption_w = (ctx.text.measure(self.caption, 'caption')[0]
                         + ctx.px(7) * 2.0)
        return widths, height, caption_w

    def measure(self, ctx):
        widths, height, caption_w = self._metrics(ctx)
        pad = ctx.px(self.container_pad)
        gap = ctx.px(self.gap)
        total = (sum(widths) + gap * max(0, len(widths) - 1)
                 + caption_w + pad * 2.0)
        return (total, height + pad * 2.0)

    def _option_rects(self, ctx):
        widths, height, caption_w = self._metrics(ctx)
        pad = ctx.px(self.container_pad)
        gap = ctx.px(self.gap)
        x = self.rect.x + pad + caption_w
        y = self.rect.y + pad
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
        radius = self.rect.h * 0.5

        ctx.draw.rect(
            self.rect.x, self.rect.y, self.rect.w, self.rect.h,
            fill=palette.panel_pill, radius=radius,
            border_color=palette.edge, border_width=ctx.theme.border_width,
            shadow=ctx.theme.glow(self.color_role),
            shadow_offset=(0.0, 0.0), shadow_softness=ctx.px(22.0),
        )

        if self.caption:
            ctx.text.draw(
                self.caption, self.rect.x + ctx.px(self.container_pad) + ctx.px(7),
                self.rect.center_y, role='caption', color=palette.text_dim,
                valign='middle',
            )

        selected = self.resolve_value()
        options = self.resolve_options()
        for index, (bx, by, bw, bh) in enumerate(self._option_rects(ctx)):
            active = index == selected
            usable = self.option_enabled(index)
            hover = 1.0 if (self.hovered and index == self._hover_index
                            and usable) else 0.0
            fill, text_color, _ = _button_colors(ctx, active, color, hover)
            if not usable:
                # Gesperrt: keine flaeche, nur sehr blasse schrift -- der
                # knopf bleibt sichtbar (die stufe existiert ja), sagt aber
                # deutlich, dass er hier nicht zu haben ist.
                text_color = palette.text_dimmer
            elif active or hover:
                ctx.draw.rect(bx, by, bw, bh, fill=fill, radius=bh * 0.5)
            ctx.text.draw(
                str(options[index]) if index < len(options) else '',
                bx + bw * 0.5, by + bh * 0.5, role=self.role,
                color=text_color, align='center', valign='middle',
            )


class SnapGrid(Widget):
    """Zwei-mal-zwei-raster fuer den rastenden orientierungs-autopiloten.

    Bildet exakt die tasten I / K / J / L ab und liest den aktiven modus
    aus schiffcontrol.snap_mode zurueck -- tastatur und HUD koennen deshalb
    nicht auseinanderlaufen.

    Die symbole des entwurfs (◉ / ⊗) werden GEZEICHNET, nicht gesetzt: beide
    fehlen in vielen oberflaechen-schriften und erschienen als kaestchen.
    """

    MODES = (
        ('prograde', 'PRO'),
        ('retrograde', 'RETRO'),
        ('normal_in', 'NORM'),
        ('antinormal_out', 'ANTI'),
    )

    def __init__(self, telemetry, ship_control, compact=False, **kwargs):
        kwargs.setdefault('size', (None, None))
        super().__init__(**kwargs)
        self.telemetry = telemetry
        self.ship_control = ship_control
        self.compact = bool(compact)
        self.blocks_mouse = True
        self._hover_index = -1

    def _tile(self, ctx):
        if self.compact:
            return ctx.px(34), ctx.px(34), ctx.px(5), ctx.px(6)
        return ctx.px(60), ctx.px(46), ctx.px(6), ctx.px(7)

    def measure(self, ctx):
        tw, th, gap, pad = self._tile(ctx)
        return (tw * 2.0 + gap + pad * 2.0, th * 2.0 + gap + pad * 2.0)

    def _tile_rect(self, ctx, index):
        tw, th, gap, pad = self._tile(ctx)
        col = index % 2
        row = index // 2
        return (self.rect.x + pad + col * (tw + gap),
                self.rect.y + pad + row * (th + gap), tw, th)

    def on_mouse_move(self, ctx, x, y):
        self._hover_index = -1
        for index in range(4):
            bx, by, bw, bh = self._tile_rect(ctx, index)
            if bx <= x < bx + bw and by <= y < by + bh:
                self._hover_index = index
                break
        return True

    def on_mouse_up(self, ctx, x, y, button):
        if button != 1 or self.ship_control is None:
            return True
        for index in range(4):
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
        color = palette.snap
        radius = ctx.px(18 if self.compact else 16)

        ctx.draw.rect(
            self.rect.x, self.rect.y, self.rect.w, self.rect.h,
            fill=palette.panel, radius=radius,
            border_color=palette.edge, border_width=ctx.theme.border_width,
            shadow=ctx.theme.glow('snap'),
            shadow_offset=(0.0, 0.0), shadow_softness=ctx.px(26.0),
        )

        active_mode = getattr(self.ship_control, 'snap_mode', None)
        for index, (mode, label) in enumerate(self.MODES):
            bx, by, bw, bh = self._tile_rect(ctx, index)
            active = mode == active_mode
            hover = 1.0 if (self.hovered and index == self._hover_index) else 0.0
            fill, text_color, border = _button_colors(ctx, active, color, hover)
            tile_radius = bh * 0.5 if self.compact else ctx.px(12)
            ctx.draw.rect(bx, by, bw, bh, fill=fill, radius=tile_radius,
                          border_color=border,
                          border_width=ctx.theme.border_width)

            if self.compact:
                self._glyph(ctx, mode, bx + bw * 0.5, by + bh * 0.5,
                            ctx.px(7), text_color)
            else:
                self._glyph(ctx, mode, bx + bw * 0.5, by + ctx.px(15),
                            ctx.px(7), text_color)
                ctx.text.draw(label, bx + bw * 0.5, by + bh - ctx.px(8),
                              role='caption', color=text_color,
                              align='center', valign='middle')

    def _glyph(self, ctx, mode, x, y, radius, color):
        width = max(1.0, ctx.px(1.6))
        if mode == 'prograde':
            ctx.draw.ring(x, y, radius, width, color)
            ctx.draw.circle(x, y, radius * 0.38, fill=color)
        elif mode == 'retrograde':
            ctx.draw.ring(x, y, radius, width, color)
            arm = radius * 0.62
            ctx.draw.line(x - arm, y - arm, x + arm, y + arm, color,
                          width=width, cap='round')
            ctx.draw.line(x - arm, y + arm, x + arm, y - arm, color,
                          width=width, cap='round')
        else:
            ctx.text.draw('N' if mode == 'normal_in' else 'A', x, y,
                          role='glyph', color=color,
                          align='center', valign='middle')


class ThrottleControl(Widget):
    """Schubstufe.

    Das schiff kennt keinen dauerschub -- 'Up'/'Down' geben pro frame einen
    festen delta-v-impuls ueber schiffcontrol.thrust_acc. Dieser regler
    skaliert genau diesen wert und ist damit eine echte, wirksame
    steuerung: bei 0 % bleibt der schub aus, bei 100 % liegt er auf dem in
    config.json eingestellten maximum.
    """

    def __init__(self, telemetry, compact=False, **kwargs):
        kwargs.setdefault('size', (None, None))
        super().__init__(**kwargs)
        self.telemetry = telemetry
        self.compact = bool(compact)
        self.blocks_mouse = True

    def measure(self, ctx):
        if self.compact:
            return (ctx.px(46), ctx.px(112 + 52))
        return (ctx.px(132 + 28), ctx.px(64))

    def _track_rect(self, ctx):
        if self.compact:
            width = ctx.px(10)
            height = ctx.px(112)
            return (self.rect.center_x - width * 0.5,
                    self.rect.y + ctx.px(26), width, height)
        width = ctx.px(132)
        height = ctx.px(10)
        return (self.rect.x + ctx.px(14), self.rect.bottom - ctx.px(20),
                width, height)

    def _apply(self, ctx, x, y):
        tx, ty, tw, th = self._track_rect(ctx)
        if self.compact:
            level = (ty + th - float(y)) / max(th, 1e-6)
        else:
            level = (float(x) - tx) / max(tw, 1e-6)
        self.telemetry.set_thrust_level(level)

    def on_mouse_down(self, ctx, x, y, button):
        if button == 1:
            self._apply(ctx, x, y)
        return True

    def on_mouse_move(self, ctx, x, y):
        if self.pressed:
            self._apply(ctx, x, y)
            return True
        return False

    def on_wheel(self, ctx, dx, dy):
        if not dy:
            return False
        self.telemetry.set_thrust_level(
            self.telemetry.thrust_level + 0.05 * float(dy)
        )
        return True

    def draw(self, ctx):
        palette = ctx.theme.palette
        # Im zeitraffer ist der schub gesperrt. Der regler wird dann
        # ausgegraut und beschriftet -- ohne das drueckt der spieler 'Up',
        # es passiert nichts, und nichts auf dem schirm sagt warum.
        locked = bool(getattr(self.telemetry, 'thrust_locked', False))
        color = palette.text_dimmer if locked else palette.throttle
        level = self.telemetry.thrust_level

        ctx.draw.rect(
            self.rect.x, self.rect.y, self.rect.w, self.rect.h,
            fill=palette.panel, radius=ctx.px(16),
            border_color=palette.edge, border_width=ctx.theme.border_width,
            shadow=ctx.theme.glow('throttle'),
            shadow_offset=(0.0, 0.0), shadow_softness=ctx.px(26.0),
        )

        tx, ty, tw, th = self._track_rect(ctx)
        ctx.draw.rect(tx, ty, tw, th, fill=palette.panel_sunken,
                      radius=min(tw, th) * 0.5)

        if self.compact:
            ctx.text.draw(self.telemetry.text_throttle(), self.rect.center_x,
                          self.rect.y + ctx.px(13), role='throttle_value',
                          color=color, align='center', valign='middle')
            filled = th * level
            if filled > 0.5:
                ctx.draw.rect(tx, ty + th - filled, tw, filled, fill=color,
                              radius=tw * 0.5)
            ctx.text.draw('HOLD' if locked else 'THR', self.rect.center_x,
                          self.rect.bottom - ctx.px(12), role='caption',
                          color=palette.text_dim, align='center', valign='middle')
            return

        ctx.text.draw('THROTTLE', self.rect.x + ctx.px(14),
                      self.rect.y + ctx.px(16), role='section',
                      color=palette.text_dim, valign='middle')
        # Gesperrt steht rechts 'HOLD' STATT des prozentwerts -- eine
        # laengere ueberschrift lief in die zahl hinein. Die eingestellte
        # stufe bleibt am fuellstand des balkens ablesbar und die zahl
        # kommt zurueck, sobald wieder in echtzeit geflogen wird.
        ctx.text.draw('HOLD' if locked else self.telemetry.text_throttle(),
                      self.rect.right - ctx.px(14), self.rect.y + ctx.px(16),
                      role='throttle_value', color=color,
                      align='right', valign='middle')

        filled = tw * level
        if filled > 0.5:
            ctx.draw.rect(tx, ty, filled, th, fill=color, radius=th * 0.5)
        knob_x = tx + filled
        ctx.draw.circle(knob_x, ty + th * 0.5, ctx.px(8),
                        fill=palette.text_dim if locked else palette.text)


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
            return (ctx.px(40) * 2.0 + ctx.px(6), ctx.px(40))
        return (ctx.px(196), ctx.px(42))

    def _button_rect(self, ctx, index):
        if self.compact:
            size = ctx.px(40)
            gap = ctx.px(6)
            return (self.rect.x + index * (size + gap), self.rect.y, size, size)
        gap = ctx.px(6)
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
        for index, label in enumerate(('SYSTEM', 'LOCAL')):
            bx, by, bw, bh = self._button_rect(ctx, index)
            active = (self._mode == ('system' if index == 0 else 'local'))
            hover = 1.0 if (self.hovered and index == self._hover_index) else 0.0
            fill, text_color, border = _button_colors(ctx, active, color, hover)
            ctx.draw.rect(bx, by, bw, bh, fill=fill, radius=bh * 0.5,
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


class PaletteButton(Widget):
    """Die vier farbpunkte oben rechts -- oeffnet die palettenauswahl.

    Der entwurf setzt genau hierhin sein zentrales versprechen: die
    oberflaeche zieht ihre farbe aus vier werten, und diese vier lassen sich
    wechseln. Umgesetzt ist die auswahl aus den drei benannten saetzen des
    entwurfs; ein farbwaehler pro einzelfarbe braeuchte ein eingabe-widget,
    das es in dieser oberflaeche noch nicht gibt.
    """

    def __init__(self, theme, on_change=None, **kwargs):
        kwargs.setdefault('size', (None, 34))
        super().__init__(**kwargs)
        self.theme_ref = theme
        self.on_change = on_change
        self.blocks_mouse = True
        self.open = False
        self.z = max(self.z, 200)
        self._hover_set = -1

    def measure(self, ctx):
        return (ctx.px(7) * 2.0 + ctx.px(12) * 4.0 + ctx.px(5) * 3.0, ctx.px(34))

    # -------------------------------------------------------------- popup

    def _popup_rect(self, ctx):
        width = ctx.px(236)
        sets = self.theme_ref.palette_sets()
        row = ctx.px(30)
        height = ctx.px(14) * 2.0 + ctx.px(18) + ctx.px(8) + len(sets) * row
        x = self.rect.right - width
        y = self.rect.bottom + ctx.px(9)
        return (x, y, width, height)

    def _set_row_rect(self, ctx, index):
        px, py, pw, ph = self._popup_rect(ctx)
        row = ctx.px(30)
        top = py + ctx.px(14) + ctx.px(18) + ctx.px(8) + index * row
        return (px + ctx.px(14), top, pw - ctx.px(28), row - ctx.px(6))

    def hit_test(self, ctx, x, y):
        if not self.visible:
            return False
        if self.rect.contains(x, y):
            return True
        if not self.open:
            return False
        px, py, pw, ph = self._popup_rect(ctx)
        return px <= x < px + pw and py <= y < py + ph

    def dismiss(self):
        self.open = False
        self._hover_set = -1

    def on_mouse_move(self, ctx, x, y):
        self._hover_set = -1
        if self.open:
            for index in range(len(self.theme_ref.palette_sets())):
                bx, by, bw, bh = self._set_row_rect(ctx, index)
                if bx <= x < bx + bw and by <= y < by + bh:
                    self._hover_set = index
                    break
        return True

    def on_mouse_up(self, ctx, x, y, button):
        if button != 1:
            return True
        if self.open:
            for index, (name, colors) in enumerate(self.theme_ref.palette_sets()):
                bx, by, bw, bh = self._set_row_rect(ctx, index)
                if bx <= x < bx + bw and by <= y < by + bh:
                    self.theme_ref.set_palette_colors(colors, name=name)
                    if self.on_change is not None:
                        self.on_change(name, colors)
                    self.open = False
                    return True
            if not self.rect.contains(x, y):
                return True
        if self.rect.contains(x, y):
            self.open = not self.open
        return True

    # ------------------------------------------------------------ zeichnen

    def draw(self, ctx):
        palette = ctx.theme.palette
        ctx.draw.rect(
            self.rect.x, self.rect.y, self.rect.w, self.rect.h,
            fill=palette.panel_pill, radius=self.rect.h * 0.5,
            border_color=palette.accent if self.open else palette.edge,
            border_width=ctx.theme.border_width,
        )
        dot = ctx.px(6)
        gap = ctx.px(5)
        x = self.rect.x + ctx.px(7) + dot
        for color in palette.colors:
            ctx.draw.circle(x, self.rect.center_y, dot, fill=color)
            x += dot * 2.0 + gap

        if self.open:
            self._draw_popup(ctx)

    def _draw_popup(self, ctx):
        palette = ctx.theme.palette
        px, py, pw, ph = self._popup_rect(ctx)
        ctx.draw.rect(
            px, py, pw, ph, fill=palette.panel_popup, radius=ctx.px(18),
            border_color=palette.edge_strong, border_width=ctx.theme.border_width,
            shadow=palette.shadow, shadow_offset=(0.0, ctx.px(-6)),
            shadow_softness=ctx.px(18),
        )
        ctx.text.draw('UI PALETTE - 4 COLOURS', px + ctx.px(14),
                      py + ctx.px(14), role='section', color=palette.text_dim)

        for index, (name, colors) in enumerate(self.theme_ref.palette_sets()):
            bx, by, bw, bh = self._set_row_rect(ctx, index)
            selected = name == palette.name
            if index == self._hover_set:
                ctx.draw.rect(bx, by, bw, bh, fill=palette.hover,
                              radius=bh * 0.5)
            swatch_w = ctx.px(13)
            total = swatch_w * 4.0
            sx = bx + ctx.px(8)
            for offset, color in enumerate(colors):
                left_r = bh * 0.35 if offset == 0 else 0.0
                right_r = bh * 0.35 if offset == 3 else 0.0
                # PALETTE_SETS haelt die farben als HEX-ZEICHENKETTEN, damit
                # die saetze lesbar in theme.py stehen. Der zeichenpfad will
                # float-tupel -- ohne diese wandlung stirbt der shader-aufruf
                # an "could not convert string to float: '#'".
                ctx.draw.rect(
                    sx + offset * swatch_w, by + ctx.px(5), swatch_w,
                    bh - ctx.px(10), fill=rgba(color),
                    radius=(left_r, right_r, right_r, left_r),
                )
            ctx.text.draw(
                name.upper(), sx + total + ctx.px(12), by + bh * 0.5,
                role='caption',
                color=palette.text if selected else palette.text_muted,
                valign='middle',
            )
