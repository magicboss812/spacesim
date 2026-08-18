"""Schaltflaeche und rastender umschalter."""

from ..core import Widget, ease
from ..theme import mix, with_alpha


class Button(Widget):
    """Klickbare schaltflaeche.

    Der klick loest beim LOSLASSEN aus, und nur wenn der zeiger dabei noch
    ueber der flaeche steht -- so kann man einen versehentlichen klick
    zurueckziehen, indem man wegzieht, bevor man loslaesst.
    """

    def __init__(self, text='', on_click=None, role='label', accent=False,
                 radius=None, size=(None, None), padding=None, **kwargs):
        super().__init__(size=size, **kwargs)
        self.text = text
        self.on_click = on_click
        self.role = role
        self.accent = accent
        self.radius = radius
        self.padding = padding
        self.blocks_mouse = True

    def resolve_text(self):
        value = self.text() if callable(self.text) else self.text
        return '' if value is None else str(value)

    def measure(self, ctx):
        pad = ctx.px(
            self.padding if self.padding is not None else ctx.theme.spacing.lg
        )
        text_w, _ = ctx.text.measure(self.resolve_text(), self.role)
        return (text_w + pad * 2.0, ctx.px(ctx.theme.control_height))

    def on_mouse_up(self, ctx, x, y, button):
        if button != 1 or not self.enabled:
            return True
        if self.rect.contains(x, y) and self.on_click is not None:
            self.on_click(self)
        return True

    def _colors(self, ctx):
        palette = ctx.theme.palette
        if not self.enabled:
            return palette.disabled, palette.border, palette.text_dim
        if self.accent:
            base = palette.accent_soft
            border = palette.accent
            text = palette.accent_strong
        else:
            base = palette.panel_raised
            border = palette.border
            text = palette.text
        fill = mix(base, palette.hover, self._hover_t)
        fill = mix(fill, palette.active, self._press_t)
        return fill, border, text

    def draw(self, ctx):
        theme = ctx.theme
        fill, border, text_color = self._colors(ctx)
        ctx.draw.rect(
            self.rect.x, self.rect.y, self.rect.w, self.rect.h,
            fill=fill,
            radius=ctx.px(self.radius if self.radius is not None else theme.radius.md),
            border_color=border,
            border_width=theme.border_width,
        )
        if self.focused:
            ctx.draw.rect(
                self.rect.x - 1, self.rect.y - 1, self.rect.w + 2, self.rect.h + 2,
                fill=None, radius=ctx.px(theme.radius.md) + 1,
                border_color=theme.palette.focus_ring, border_width=theme.border_width,
            )
        ctx.text.draw(
            self.resolve_text(), self.rect.center_x, self.rect.center_y,
            role=self.role, color=text_color, align='center', valign='middle',
        )


class Toggle(Widget):
    """Rastender schalter mit schiebeknopf.

    value darf ein aufrufbares objekt sein (dann ist der schalter nur eine
    ANZEIGE des externen zustands) -- on_change traegt die eigentliche
    aenderung. So kann derselbe zustand weiter per tastatur umgeschaltet
    werden, ohne dass der schalter danach falsch steht.
    """

    def __init__(self, text='', value=False, on_change=None, role='label',
                 size=(None, None), **kwargs):
        super().__init__(size=size, **kwargs)
        self.text = text
        self.value = value
        self.on_change = on_change
        self.role = role
        self.blocks_mouse = True
        self._knob_t = 1.0 if self.resolve_value() else 0.0

    def resolve_value(self):
        return bool(self.value() if callable(self.value) else self.value)

    def measure(self, ctx):
        track_w = ctx.px(ctx.theme.control_height) * 1.7
        text_w, _ = ctx.text.measure(str(self.text), self.role)
        gap = ctx.px(ctx.theme.spacing.md) if text_w else 0.0
        return (track_w + gap + text_w, ctx.px(ctx.theme.control_height))

    def update(self, ctx, dt):
        super().update(ctx, dt)
        self._knob_t = ease(
            self._knob_t, 1.0 if self.resolve_value() else 0.0,
            ctx.theme.motion.fast, dt,
        )

    def on_mouse_up(self, ctx, x, y, button):
        if button != 1 or not self.enabled:
            return True
        if self.rect.contains(x, y):
            new_value = not self.resolve_value()
            if not callable(self.value):
                self.value = new_value
            if self.on_change is not None:
                self.on_change(new_value)
        return True

    def draw(self, ctx):
        theme = ctx.theme
        palette = theme.palette
        on = self.resolve_value()

        track_h = min(self.rect.h, ctx.px(theme.control_height_sm))
        track_w = track_h * 1.9
        track_y = self.rect.center_y - track_h * 0.5

        off_color = mix(palette.panel_sunken, palette.hover, self._hover_t)
        track_fill = mix(off_color, palette.accent, self._knob_t)
        ctx.draw.rect(
            self.rect.x, track_y, track_w, track_h,
            fill=track_fill, radius=track_h * 0.5,
            border_color=palette.accent if on else palette.border,
            border_width=theme.border_width,
        )

        inset = ctx.px(2.0)
        knob_r = track_h * 0.5 - inset
        knob_x = (
            self.rect.x + inset + knob_r
            + (track_w - 2.0 * (inset + knob_r)) * self._knob_t
        )
        ctx.draw.circle(
            knob_x, self.rect.center_y, knob_r,
            fill=palette.text if on else palette.text_muted,
        )

        if self.text:
            ctx.text.draw(
                str(self.text), self.rect.x + track_w + ctx.px(theme.spacing.md),
                self.rect.center_y, role=self.role,
                color=palette.text if self.enabled else palette.text_dim,
                valign='middle',
            )


class SegmentedControl(Widget):
    """Mehrere sich gegenseitig ausschliessende optionen in einer leiste.

    Die form fuer kleine, feste auswahlmengen -- etwa die rahmen-modi
    (non-rotating / body-direction). Sichtbar sind alle optionen gleichzeitig,
    anders als beim aufklappmenue.
    """

    def __init__(self, options=(), value=0, on_change=None, role='label',
                 size=(None, None), **kwargs):
        super().__init__(size=size, **kwargs)
        self.options = list(options)
        self.value = value
        self.on_change = on_change
        self.role = role
        self.blocks_mouse = True

    def resolve_value(self):
        return int(self.value() if callable(self.value) else self.value)

    def measure(self, ctx):
        pad = ctx.px(ctx.theme.spacing.lg) * 2.0
        widest = 0.0
        for option in self.options:
            widest = max(widest, ctx.text.measure(str(option), self.role)[0])
        count = max(1, len(self.options))
        return ((widest + pad) * count, ctx.px(ctx.theme.control_height))

    def _segment_rect(self, index):
        count = max(1, len(self.options))
        width = self.rect.w / count
        return (self.rect.x + width * index, self.rect.y, width, self.rect.h)

    def on_mouse_up(self, ctx, x, y, button):
        if button != 1 or not self.enabled:
            return True
        for index in range(len(self.options)):
            sx, sy, sw, sh = self._segment_rect(index)
            if sx <= x < sx + sw and sy <= y < sy + sh:
                if not callable(self.value):
                    self.value = index
                if self.on_change is not None:
                    self.on_change(index)
                break
        return True

    def draw(self, ctx):
        theme = ctx.theme
        palette = theme.palette
        radius = ctx.px(theme.radius.md)
        selected = self.resolve_value()

        ctx.draw.rect(
            self.rect.x, self.rect.y, self.rect.w, self.rect.h,
            fill=palette.panel_sunken, radius=radius,
            border_color=palette.border, border_width=theme.border_width,
        )

        for index, option in enumerate(self.options):
            sx, sy, sw, sh = self._segment_rect(index)
            active = index == selected
            if active:
                inset = ctx.px(2.0)
                ctx.draw.rect(
                    sx + inset, sy + inset, sw - 2 * inset, sh - 2 * inset,
                    fill=with_alpha(palette.accent, 0.22),
                    radius=max(0.0, radius - inset),
                    border_color=palette.accent, border_width=theme.border_width,
                )
            ctx.text.draw(
                str(option), sx + sw * 0.5, sy + sh * 0.5, role=self.role,
                color=palette.accent_strong if active else palette.text_muted,
                align='center', valign='middle',
            )
