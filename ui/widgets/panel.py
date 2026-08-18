"""Panel: die grundflaeche, auf der HUD-gruppen sitzen."""

from ..core import Rect, Widget


class Panel(Widget):
    """Abgerundete flaeche mit rahmen, schatten und innenabstand.

    blocks_mouse ist absichtlich True: ein klick auf ein panel darf NICHT
    zur kamera durchfallen, sonst schwenkt man beim bedienen des HUDs die
    ansicht.
    """

    def __init__(self, padding=None, radius=None, fill=None, border=None,
                 shadow=True, gradient_to=None, **kwargs):
        super().__init__(**kwargs)
        self.padding = padding
        self.radius = radius
        self.fill = fill
        self.border = border
        self.shadow = shadow
        self.gradient_to = gradient_to
        self.blocks_mouse = True

    def content_rect(self, ctx):
        pad = ctx.px(
            self.padding if self.padding is not None else ctx.theme.panel_padding
        )
        return self.rect.inset(pad)

    def draw(self, ctx):
        theme = ctx.theme
        palette = theme.palette
        ctx.draw.rect(
            self.rect.x, self.rect.y, self.rect.w, self.rect.h,
            fill=self.fill if self.fill is not None else palette.panel,
            gradient_to=self.gradient_to,
            radius=ctx.px(self.radius if self.radius is not None else theme.radius.lg),
            border_color=self.border if self.border is not None else palette.border,
            border_width=theme.border_width,
            shadow=palette.shadow if self.shadow else None,
            shadow_offset=(
                ctx.px(theme.shadow_offset[0]), ctx.px(theme.shadow_offset[1])
            ),
            shadow_softness=ctx.px(theme.shadow_softness),
        )


class Group(Widget):
    """Unsichtbarer container. Nur zum verankern -- verbraucht keine maus."""

    def __init__(self, padding=0.0, **kwargs):
        super().__init__(**kwargs)
        self.padding = padding

    def content_rect(self, ctx):
        return self.rect.inset(ctx.px(self.padding))


class Stack(Widget):
    """Unsichtbarer container, der seine kinder aneinanderreiht.

    Die verankerung aus ui/core.py setzt EIN widget an EINE ecke -- fuer
    gruppen, die zusammen an einer ecke sitzen (zeitraffer + palette,
    rahmenwahl + ring, snaps + zoom), braucht es eine reihung. Die groesse
    folgt den kindern, damit die gruppe als ganzes verankert werden kann.

    align steuert die QUERachse: bei einer senkrechten reihung also die
    waagerechte ausrichtung.
    """

    def __init__(self, gap=8, horizontal=False, align='start', **kwargs):
        kwargs.setdefault('size', (None, None))
        super().__init__(**kwargs)
        self.gap = gap
        self.horizontal = bool(horizontal)
        self.align = align

    def _visible_children(self):
        return [child for child in self.children if child.visible]

    def measure(self, ctx):
        gap = ctx.px(self.gap)
        children = self._visible_children()
        if not children:
            return (0.0, 0.0)
        sizes = [child.desired_size(ctx) for child in children]
        along = sum(s[0] if self.horizontal else s[1] for s in sizes)
        along += gap * (len(sizes) - 1)
        across = max((s[1] if self.horizontal else s[0]) for s in sizes)
        return (along, across) if self.horizontal else (across, along)

    def layout_children(self, ctx):
        gap = ctx.px(self.gap)
        rect = self.rect
        cursor = rect.x if self.horizontal else rect.y
        for child in self._visible_children():
            width, height = child.desired_size(ctx)
            if self.horizontal:
                offset = self._cross_offset(rect.h, height)
                child.layout(ctx, Rect(cursor, rect.y + offset, width, height))
                cursor += width + gap
            else:
                offset = self._cross_offset(rect.w, width)
                child.layout(ctx, Rect(rect.x + offset, cursor, width, height))
                cursor += height + gap

    def _cross_offset(self, available, size):
        if self.align == 'center':
            return (available - size) * 0.5
        if self.align == 'end':
            return available - size
        return 0.0
