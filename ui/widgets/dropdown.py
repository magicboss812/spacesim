"""Aufklappmenue -- fuer auswahlmengen, die zu gross fuer eine leiste sind
(referenzkoerper bei vielen himmelskoerpern)."""

from ..core import Rect, Widget
from ..theme import mix, with_alpha


class Dropdown(Widget):
    """Auswahlfeld mit aufklappliste.

    Die liste wird vom widget SELBST gezeichnet, nicht als eigenes kind:
    sie muss ueber allem liegen, was danach im baum kommt, und sie muss
    treffer entgegennehmen, obwohl sie ausserhalb des eigenen rechtecks
    liegt -- deshalb erweitern hit_test() und draw() sich im offenen zustand
    um die listenflaeche.
    """

    def __init__(self, options=(), value=0, on_change=None, role='label',
                 max_visible=8, size=(160.0, None), **kwargs):
        super().__init__(size=size, **kwargs)
        self.options = list(options)
        self.value = value
        self.on_change = on_change
        self.role = role
        self.max_visible = int(max_visible)
        self.open = False
        self.blocks_mouse = True
        self.z = max(self.z, 100)
        self._highlight = -1

    # ------------------------------------------------------------ ableitung

    def resolve_options(self):
        return list(self.options() if callable(self.options) else self.options)

    def resolve_value(self):
        return int(self.value() if callable(self.value) else self.value)

    def selected_text(self):
        options = self.resolve_options()
        index = self.resolve_value()
        if 0 <= index < len(options):
            return str(options[index])
        return '--'

    def measure(self, ctx):
        pad = ctx.px(ctx.theme.spacing.lg) * 2.0 + ctx.px(14.0)
        widest = ctx.text.measure(self.selected_text(), self.role)[0]
        for option in self.resolve_options():
            widest = max(widest, ctx.text.measure(str(option), self.role)[0])
        return (widest + pad, ctx.px(ctx.theme.control_height))

    def _row_height(self, ctx):
        return ctx.px(ctx.theme.control_height)

    def _list_rect(self, ctx):
        count = min(len(self.resolve_options()), self.max_visible)
        row = self._row_height(ctx)
        pad = ctx.px(ctx.theme.spacing.sm)
        height = count * row + pad * 2.0
        top = self.rect.bottom + ctx.px(ctx.theme.spacing.sm)
        # Nach oben aufklappen, wenn unten kein platz mehr ist.
        if top + height > ctx.height:
            top = self.rect.y - ctx.px(ctx.theme.spacing.sm) - height
        return Rect(self.rect.x, top, self.rect.w, height)

    def _row_index_at(self, ctx, x, y):
        area = self._list_rect(ctx)
        if not area.contains(x, y):
            return -1
        pad = ctx.px(ctx.theme.spacing.sm)
        row = self._row_height(ctx)
        index = int((float(y) - area.y - pad) // row)
        options = self.resolve_options()
        if 0 <= index < min(len(options), self.max_visible):
            return index
        return -1

    # ------------------------------------------------------------- eingabe

    def hit_test(self, ctx, x, y):
        if not self.visible:
            return False
        if self.rect.contains(x, y):
            return True
        # Im offenen zustand gehoert die liste mit dazu, obwohl sie ausserhalb
        # des eigenen rechtecks liegt -- sonst laege der klick auf einem
        # eintrag fuer die UI im leeren und ginge an die kamera durch.
        return bool(self.open and self._list_rect(ctx).contains(x, y))

    def on_mouse_down(self, ctx, x, y, button):
        if button != 1 or not self.enabled:
            return True
        if self.open:
            index = self._row_index_at(ctx, x, y)
            if index >= 0:
                if not callable(self.value):
                    self.value = index
                if self.on_change is not None:
                    self.on_change(index)
            self.open = False
        elif self.rect.contains(x, y):
            self.open = True
        return True

    def dismiss(self):
        self.open = False
        self._highlight = -1

    def on_mouse_move(self, ctx, x, y):
        if self.open:
            self._highlight = self._row_index_at(ctx, x, y)
            return True
        self._highlight = -1
        return False

    # ------------------------------------------------------------ zeichnen

    def draw(self, ctx):
        theme = ctx.theme
        palette = theme.palette
        radius = ctx.px(theme.radius.md)

        fill = mix(palette.panel_raised, palette.hover, self._hover_t)
        ctx.draw.rect(
            self.rect.x, self.rect.y, self.rect.w, self.rect.h,
            fill=fill, radius=radius,
            border_color=palette.accent if self.open else palette.border,
            border_width=theme.border_width,
        )
        pad = ctx.px(theme.spacing.lg)
        ctx.text.draw(
            self.selected_text(), self.rect.x + pad, self.rect.center_y,
            role=self.role, color=palette.text, valign='middle',
        )

        # Pfeil als zwei linien, gedreht ueber die offen/zu-lage.
        cx = self.rect.right - pad
        cy = self.rect.center_y
        arm = ctx.px(4.0)
        direction = -1.0 if self.open else 1.0
        ctx.draw.line(cx - arm, cy - arm * 0.5 * direction,
                      cx, cy + arm * 0.5 * direction,
                      palette.text_muted, width=ctx.px(1.5), cap='round')
        ctx.draw.line(cx, cy + arm * 0.5 * direction,
                      cx + arm, cy - arm * 0.5 * direction,
                      palette.text_muted, width=ctx.px(1.5), cap='round')

        if self.open:
            self._draw_list(ctx)

    def _draw_list(self, ctx):
        theme = ctx.theme
        palette = theme.palette
        area = self._list_rect(ctx)
        radius = ctx.px(theme.radius.md)

        ctx.draw.rect(
            area.x, area.y, area.w, area.h,
            fill=palette.panel_raised, radius=radius,
            border_color=palette.border_strong, border_width=theme.border_width,
            shadow=palette.shadow,
            shadow_offset=(ctx.px(0.0), ctx.px(-4.0)),
            shadow_softness=ctx.px(10.0),
        )

        pad = ctx.px(theme.spacing.sm)
        row = self._row_height(ctx)
        selected = self.resolve_value()
        for index, option in enumerate(self.resolve_options()[: self.max_visible]):
            top = area.y + pad + index * row
            if index == self._highlight:
                ctx.draw.rect(
                    area.x + pad, top, area.w - pad * 2.0, row,
                    fill=palette.hover, radius=ctx.px(theme.radius.sm),
                )
            elif index == selected:
                ctx.draw.rect(
                    area.x + pad, top, area.w - pad * 2.0, row,
                    fill=with_alpha(palette.accent, 0.16),
                    radius=ctx.px(theme.radius.sm),
                )
            ctx.text.draw(
                str(option), area.x + pad + ctx.px(theme.spacing.md), top + row * 0.5,
                role=self.role,
                color=palette.accent_strong if index == selected else palette.text,
                valign='middle',
            )
