"""Schieberegler, linear oder logarithmisch."""

import math

from ..core import Rect, Widget
from ..theme import mix


class Slider(Widget):
    """Horizontaler regler.

    log=True fuer groessen, die ueber mehrere zehnerpotenzen laufen
    (zeitraffer, kamera-zoom, predictor-horizont). Ein linearer regler ist
    dort unbrauchbar: 99 % des weges lieferten dann werte, die man nie will.
    """

    def __init__(self, value=0.0, minimum=0.0, maximum=1.0, on_change=None,
                 log=False, step=None, label='', fmt=None, role='label',
                 size=(120.0, None), **kwargs):
        super().__init__(size=size, **kwargs)
        self.value = value
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.on_change = on_change
        self.log = bool(log)
        self.step = step
        self.label = label
        self.fmt = fmt
        self.role = role
        self.blocks_mouse = True

    # ---------------------------------------------------------- umrechnung

    def resolve_value(self):
        return float(self.value() if callable(self.value) else self.value)

    def _to_fraction(self, value):
        lo, hi = self.minimum, self.maximum
        if self.log:
            lo = max(lo, 1e-30)
            hi = max(hi, lo * (1.0 + 1e-9))
            value = min(max(float(value), lo), hi)
            return (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
        if hi - lo <= 0.0:
            return 0.0
        return min(max((float(value) - lo) / (hi - lo), 0.0), 1.0)

    def _from_fraction(self, fraction):
        fraction = min(max(float(fraction), 0.0), 1.0)
        lo, hi = self.minimum, self.maximum
        if self.log:
            lo = max(lo, 1e-30)
            hi = max(hi, lo * (1.0 + 1e-9))
            value = math.exp(math.log(lo) + fraction * (math.log(hi) - math.log(lo)))
        else:
            value = lo + fraction * (hi - lo)
        if self.step:
            value = round(value / self.step) * self.step
        return value

    # -------------------------------------------------------------- layout

    def measure(self, ctx):
        return (ctx.px(120.0), ctx.px(ctx.theme.control_height))

    def _track_rect(self, ctx):
        height = ctx.px(4.0)
        top = self.rect.y + self.rect.h * 0.5 - height * 0.5
        if self.label:
            top = self.rect.bottom - ctx.px(ctx.theme.spacing.md) - height
        return Rect(self.rect.x, top, self.rect.w, height)

    # ------------------------------------------------------------- eingabe

    def _apply_from_x(self, ctx, x):
        track = self._track_rect(ctx)
        if track.w <= 0.0:
            return
        fraction = (float(x) - track.x) / track.w
        new_value = self._from_fraction(fraction)
        if not callable(self.value):
            self.value = new_value
        if self.on_change is not None:
            self.on_change(new_value)

    def on_mouse_down(self, ctx, x, y, button):
        if button == 1 and self.enabled:
            self._apply_from_x(ctx, x)
        return True

    def on_mouse_move(self, ctx, x, y):
        # Nur waehrend des ziehens. UIRoot leitet bewegungen an das AKTIVE
        # widget weiter, auch wenn der zeiger dabei herauswandert -- deshalb
        # bleibt der griff beim schnellen ziehen haengen.
        if self.pressed and self.enabled:
            self._apply_from_x(ctx, x)
            return True
        return False

    def on_wheel(self, ctx, dx, dy):
        if not self.enabled or not dy:
            return False
        fraction = self._to_fraction(self.resolve_value()) + 0.04 * float(dy)
        new_value = self._from_fraction(fraction)
        if not callable(self.value):
            self.value = new_value
        if self.on_change is not None:
            self.on_change(new_value)
        return True

    # ------------------------------------------------------------ zeichnen

    def draw(self, ctx):
        theme = ctx.theme
        palette = theme.palette
        track = self._track_rect(ctx)
        fraction = self._to_fraction(self.resolve_value())

        if self.label or self.fmt:
            text_y = self.rect.y
            if self.label:
                ctx.text.draw(
                    str(self.label), self.rect.x, text_y, role=self.role,
                    color=palette.text_muted,
                )
            if self.fmt:
                ctx.text.draw(
                    self.fmt(self.resolve_value()), self.rect.right, text_y,
                    role='mono_readout', color=palette.text, align='right',
                )

        ctx.draw.rect(
            track.x, track.y, track.w, track.h,
            fill=palette.panel_sunken, radius=track.h * 0.5,
        )
        if fraction > 0.0:
            ctx.draw.rect(
                track.x, track.y, max(track.h, track.w * fraction), track.h,
                fill=palette.accent if self.enabled else palette.text_dim,
                radius=track.h * 0.5,
            )

        knob_r = ctx.px(7.0) + ctx.px(1.5) * self._hover_t
        knob_x = track.x + track.w * fraction
        ctx.draw.circle(
            knob_x, track.center_y, knob_r,
            fill=mix(palette.text, palette.accent_strong, self._press_t),
            border_color=palette.panel_sunken, border_width=theme.border_width,
        )
