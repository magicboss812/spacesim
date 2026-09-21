"""Schaltflaeche und rastender umschalter."""

from ..core import Widget, ease
from ..theme import mix


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

