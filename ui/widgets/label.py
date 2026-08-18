"""Text-widgets: einfaches label und die zweispaltige messwert-zeile."""

from ..core import Widget


class Label(Widget):
    """Text. Die groesse folgt standardmaessig dem inhalt (size=(None, None)).

    text darf ein aufrufbares objekt sein -- dann wird es pro frame
    ausgewertet. Das ist der weg fuer messwerte, die sich staendig aendern,
    ohne dass jemand von aussen setText() aufrufen muss.
    """

    def __init__(self, text='', role='body', color=None, align='left',
                 valign='top', size=(None, None), **kwargs):
        super().__init__(size=size, **kwargs)
        self.text = text
        self.role = role
        self.color = color
        self.align = align
        self.valign = valign

    def resolve_text(self):
        value = self.text() if callable(self.text) else self.text
        return '' if value is None else str(value)

    def measure(self, ctx):
        return ctx.text.measure(self.resolve_text(), self.role)

    def draw(self, ctx):
        color = self.color if self.color is not None else ctx.theme.palette.text
        if not self.enabled:
            color = ctx.theme.palette.text_dim

        if self.align == 'center':
            x = self.rect.center_x
        elif self.align == 'right':
            x = self.rect.right
        else:
            x = self.rect.x

        if self.valign == 'middle':
            y = self.rect.center_y
        elif self.valign == 'bottom':
            y = self.rect.bottom
        else:
            y = self.rect.y

        ctx.text.draw(
            self.resolve_text(), x, y, role=self.role, color=color,
            align=self.align, valign=self.valign,
        )


class Readout(Widget):
    """Beschriftung links, wert rechts -- die standardzeile der info-panels.

    Der wert laeuft in der mono-rolle: eine zahl, die sich jeden frame
    aendert, darf die spaltenbreite nicht verschieben, sonst zappelt das
    ganze panel.
    """

    def __init__(self, label='', value='', value_color=None,
                 label_role='label', value_role='mono_readout',
                 size=(None, None), **kwargs):
        super().__init__(size=size, **kwargs)
        self.label = label
        self.value = value
        self.value_color = value_color
        self.label_role = label_role
        self.value_role = value_role

    def resolve_value(self):
        value = self.value() if callable(self.value) else self.value
        return '' if value is None else str(value)

    def measure(self, ctx):
        label_w, label_h = ctx.text.measure(str(self.label), self.label_role)
        value_w, value_h = ctx.text.measure(self.resolve_value(), self.value_role)
        gap = ctx.px(ctx.theme.spacing.lg)
        return (label_w + gap + value_w, max(label_h, value_h))

    def draw(self, ctx):
        palette = ctx.theme.palette
        middle = self.rect.center_y
        ctx.text.draw(
            str(self.label), self.rect.x, middle, role=self.label_role,
            color=palette.text_muted, valign='middle',
        )
        ctx.text.draw(
            self.resolve_value(), self.rect.right, middle, role=self.value_role,
            color=self.value_color if self.value_color is not None else palette.text,
            align='right', valign='middle',
        )
