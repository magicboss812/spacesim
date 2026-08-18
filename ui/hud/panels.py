"""Die informations-panels des HUDs: bahnelemente und ziel.

Alle teilen die form aus dem entwurf: abgerundete flaeche (radius 16) auf
rgba(10,15,22,.70), hairline-kante in weiss, und ein farbiger SCHEIN in der
rollenfarbe des panels. Der schein ist das, was die vier palettenfarben ueber
die ganze oberflaeche traegt, ohne dass die flaechen selbst bunt werden --
"the palette only tints chrome, glow and data".

Technisch ist der schein derselbe schlagschatten wie sonst, nur ohne versatz
und mit grosser weichzeichnung: ein 0-offset-schatten IST ein glow.
"""

from ..core import FILL, Rect, Widget
from ..theme import with_alpha
from ..widgets import Readout
from ..widgets.panel import Panel


class HudPanel(Panel):
    """Panel im entwurfs-stil: hairline-kante plus farbiger schein.

    Stapelt seine kinder senkrecht mit festem abstand -- die info-panels
    sind alle einfache listen, und jedem kind von hand einen y-versatz zu
    geben waere nur eine fehlerquelle beim umbauen.
    """

    def __init__(self, glow_role='elem', gap=None, **kwargs):
        kwargs.setdefault('radius', 16)
        kwargs.setdefault('padding', 14)
        # (None, None) = groesse aus measure(), also aus den kindern. Der
        # Widget-standardwert (0, 0) waere hier ein unsichtbares panel.
        kwargs.setdefault('size', (None, None))
        super().__init__(**kwargs)
        self.glow_role = glow_role
        self.gap = gap

    def _gap_px(self, ctx):
        return ctx.px(self.gap if self.gap is not None else ctx.theme.spacing.lg)

    def _content_probe(self, ctx):
        """Fiktive inhaltsflaeche fuer die messung.

        Die zeilen sind (FILL, None) breit -- ihre HOEHE braucht die breite
        nicht, aber _resolve_size will trotzdem eine flaeche. Die echte
        breite steht waehrend measure() noch nicht fest (self.rect ist noch
        leer), also wird sie hier aus der entwurfsbreite abgeleitet.
        """
        pad = ctx.px(self.padding if self.padding is not None else 14)
        return Rect(0.0, 0.0, ctx.px(ctx.theme.panel_width) - pad * 2.0, 0.0)

    def measure(self, ctx):
        """Hoehe aus den kindern, breite aus der vorgabe.

        Haengt bewusst NICHT von self.rect ab -- sonst waere die groessen-
        aufloesung zirkulaer (rect braucht die groesse, die groesse das rect).
        """
        pad = ctx.px(self.padding if self.padding is not None else 14)
        gap = self._gap_px(ctx)
        probe = self._content_probe(ctx)
        total = 0.0
        visible = [c for c in self.children if c.visible]
        for index, child in enumerate(visible):
            total += child.desired_size(ctx, probe)[1]
            if index < len(visible) - 1:
                total += gap
        return (ctx.px(ctx.theme.panel_width), total + pad * 2.0)

    def layout_children(self, ctx):
        content = self.content_rect(ctx)
        gap = self._gap_px(ctx)
        cursor = content.y
        for child in self.children:
            if not child.visible:
                continue
            height = child.desired_size(ctx, content)[1]
            child.layout(ctx, Rect(content.x, cursor, content.w, height))
            cursor += height + gap

    def draw(self, ctx):
        palette = ctx.theme.palette
        theme = ctx.theme
        radius = ctx.px(self.radius)
        # Der farbige schein: gleiche form, kein versatz, weit weichgezeichnet.
        ctx.draw.rect(
            self.rect.x, self.rect.y, self.rect.w, self.rect.h,
            fill=self.fill if self.fill is not None else palette.panel,
            radius=radius,
            border_color=self.border if self.border is not None else palette.edge,
            border_width=theme.border_width,
            shadow=theme.glow(self.glow_role),
            shadow_offset=(0.0, 0.0),
            shadow_softness=ctx.px(28.0),
        )


class SectionLabel(Widget):
    """Die gesperrte versal-ueberschrift eines panels."""

    def __init__(self, text='', color_role=None, **kwargs):
        kwargs.setdefault('size', (FILL, None))
        super().__init__(**kwargs)
        self.text = text
        self.color_role = color_role

    def measure(self, ctx):
        return ctx.text.measure(str(self.text), 'section')

    def draw(self, ctx):
        palette = ctx.theme.palette
        color = (palette.accent_for(self.color_role) if self.color_role
                 else palette.text_dim)
        ctx.text.draw(str(self.text), self.rect.x, self.rect.y,
                      role='section', color=color)


class TargetHeader(Widget):
    """'TARGET' links, 'LOCKED'-marke rechts."""

    def __init__(self, telemetry, **kwargs):
        kwargs.setdefault('size', (FILL, None))
        super().__init__(**kwargs)
        self.telemetry = telemetry

    def measure(self, ctx):
        return (0.0, max(ctx.text.measure('TARGET', 'section')[1],
                         ctx.text.measure('LOCKED', 'pill')[1] + ctx.px(6)))

    def draw(self, ctx):
        palette = ctx.theme.palette
        middle = self.rect.center_y
        ctx.text.draw('TARGET', self.rect.x, middle, role='section',
                      color=palette.text_dim, valign='middle')

        locked = self.telemetry.target_locked
        label = 'LOCKED' if locked else 'NONE'
        color = palette.target if locked else palette.text_dimmer
        text_w, text_h = ctx.text.measure(label, 'pill')
        pad_x = ctx.px(8)
        pad_y = ctx.px(3)
        width = text_w + pad_x * 2.0
        height = text_h + pad_y * 2.0
        x = self.rect.right - width
        y = middle - height * 0.5
        ctx.draw.rect(x, y, width, height, fill=None, radius=height * 0.5,
                      border_color=with_alpha(color, 0.55),
                      border_width=ctx.theme.border_width)
        ctx.text.draw(label, x + width * 0.5, middle, role='pill', color=color,
                      align='center', valign='middle')


class TargetName(Widget):
    """Der ziel-name in der zielfarbe."""

    def __init__(self, telemetry, **kwargs):
        kwargs.setdefault('size', (FILL, None))
        super().__init__(**kwargs)
        self.telemetry = telemetry

    def _text(self):
        state = self.telemetry.ui_state
        return (state.reference_name if state else '--').upper()

    def measure(self, ctx):
        return ctx.text.measure(self._text(), 'title')

    def draw(self, ctx):
        ctx.text.draw(self._text(), self.rect.x, self.rect.y, role='title',
                      color=ctx.theme.palette.target)


def _row(label, value, value_color=None):
    return Readout(
        label=label, value=value, value_color=value_color,
        label_role='key', value_role='value', size=(FILL, None),
    )


def build_elements_panel(telemetry, **kwargs):
    """Bahnelemente: AP, PE, ECC, PERIODE, T-AP.

    Genau die fuenf zeilen des entwurfs. Die werte kommen aus der
    zweikoerper-loesung in telemetry.py, nicht aus abgelesenen
    predictor-punkten -- sie sollen auch stimmen, wenn der predictor
    gerade neu rechnet oder aus ist.
    """
    panel = HudPanel(glow_role='elem', gap=10, **kwargs)
    panel.add(SectionLabel('ORBITAL ELEMENTS', color_role='elem'))
    panel.add(_row('AP', telemetry.text_apoapsis))
    panel.add(_row('PE', telemetry.text_periapsis))
    panel.add(_row('ECC', telemetry.text_eccentricity))
    panel.add(_row('PERIOD', telemetry.text_period))
    panel.add(_row('T-AP', telemetry.text_time_to_apoapsis))
    return panel


def build_target_panel(telemetry, **kwargs):
    """Ziel-panel. Ziel ist der aktive BEZUGSKOERPER -- die simulation kennt
    keine eigene zielauswahl, und der bezugskoerper ist genau der, auf den
    sich alle uebrigen anzeigen beziehen."""
    panel = HudPanel(glow_role='target', gap=10, **kwargs)
    panel.add(TargetHeader(telemetry))
    panel.add(TargetName(telemetry))
    panel.add(_row('DIST', telemetry.text_target_distance))
    panel.add(_row('REL V', telemetry.text_target_relative_speed))
    panel.add(_row('CLOSEST', telemetry.text_closest))
    panel.add(_row('T-CA', telemetry.text_time_to_closest))
    return panel


class ShipBadge(Widget):
    """Die pille oben links: leuchtpunkt, schiffsname, bezugskoerper.

    Im entwurf steht rechts die baureihe ('MK-II'). Die simulation hat keine
    baureihen, wohl aber einen aktiven bezugskoerper -- und der gehoert
    ohnehin dauerhaft sichtbar, weil saemtliche bahnwerte relativ zu ihm
    gelten.
    """

    def __init__(self, telemetry, **kwargs):
        kwargs.setdefault('size', (None, 34))
        super().__init__(**kwargs)
        self.telemetry = telemetry

    def _texts(self):
        ship = self.telemetry.ship
        state = self.telemetry.ui_state
        name = str(getattr(ship, 'name', 'SHIP')).upper()
        reference = (state.reference_name if state else '--').upper()
        return name, reference

    def measure(self, ctx):
        name, reference = self._texts()
        width = (ctx.px(15) * 2.0 + ctx.px(7) + ctx.px(11)
                 + ctx.text.measure(name, 'badge')[0]
                 + ctx.px(ctx.theme.spacing.lg)
                 + ctx.text.measure(reference, 'badge_sub')[0])
        return (width, ctx.px(34))

    def draw(self, ctx):
        palette = ctx.theme.palette
        name, reference = self._texts()
        height = self.rect.h

        ctx.draw.rect(
            self.rect.x, self.rect.y, self.rect.w, height,
            fill=palette.panel_pill, radius=height * 0.5,
            border_color=palette.edge, border_width=ctx.theme.border_width,
            shadow=with_alpha(palette.ship, 0.28 * ctx.theme.glow_intensity),
            shadow_offset=(0.0, 0.0), shadow_softness=ctx.px(22.0),
        )

        pad = ctx.px(15)
        middle = self.rect.center_y
        dot_x = self.rect.x + pad
        ctx.draw.circle(dot_x, middle, ctx.px(3.5), fill=palette.ship)

        text_x = dot_x + ctx.px(11)
        ctx.text.draw(name, text_x, middle, role='badge', color=palette.text,
                      valign='middle')
        ctx.text.draw(reference, self.rect.right - pad, middle,
                      role='badge_sub', color=palette.text_dim,
                      align='right', valign='middle')


class IconRail(Widget):
    """Schmale senkrechte leiste fuer das kompakte layout.

    Ersetzt unter der umbruchbreite ein ganzes info-panel: die werte selbst
    haetten dort keinen platz mehr, die kuerzel als abrufbare knoepfe schon.
    """

    def __init__(self, entries, color_role='elem', **kwargs):
        kwargs.setdefault('size', (46, None))
        super().__init__(**kwargs)
        self.entries = list(entries)
        self.color_role = color_role
        self.blocks_mouse = True
        self._hover_index = -1

    def measure(self, ctx):
        button = ctx.px(34)
        gap = ctx.px(4)
        pad = ctx.px(8)
        return (ctx.px(46), pad * 2.0 + len(self.entries) * button
                + max(0, len(self.entries) - 1) * gap)

    def _button_rect(self, ctx, index):
        button = ctx.px(34)
        gap = ctx.px(4)
        pad = ctx.px(8)
        return (self.rect.center_x - button * 0.5,
                self.rect.y + pad + index * (button + gap), button, button)

    def on_mouse_move(self, ctx, x, y):
        self._hover_index = -1
        for index in range(len(self.entries)):
            bx, by, bw, bh = self._button_rect(ctx, index)
            if bx <= x < bx + bw and by <= y < by + bh:
                self._hover_index = index
                break
        return True

    def on_mouse_up(self, ctx, x, y, button):
        if button != 1:
            return True
        for index, entry in enumerate(self.entries):
            bx, by, bw, bh = self._button_rect(ctx, index)
            if bx <= x < bx + bw and by <= y < by + bh:
                action = entry.get('action')
                if action is not None:
                    action()
                break
        return True

    def draw(self, ctx):
        palette = ctx.theme.palette
        color = palette.accent_for(self.color_role)
        ctx.draw.rect(
            self.rect.x, self.rect.y, self.rect.w, self.rect.h,
            fill=palette.panel, radius=self.rect.w * 0.5,
            border_color=palette.edge, border_width=ctx.theme.border_width,
            shadow=ctx.theme.glow(self.color_role),
            shadow_offset=(0.0, 0.0), shadow_softness=ctx.px(22.0),
        )
        for index, entry in enumerate(self.entries):
            bx, by, bw, bh = self._button_rect(ctx, index)
            if index == self._hover_index and self.hovered:
                ctx.draw.circle(bx + bw * 0.5, by + bh * 0.5, bw * 0.5,
                                fill=palette.hover)
            ctx.text.draw(str(entry['key']), bx + bw * 0.5, by + bh * 0.5,
                          role='button_sm', color=color,
                          align='center', valign='middle')
            value = entry.get('value')
            if value is not None:
                text = value() if callable(value) else str(value)
                ctx.text.draw(text, bx + bw * 0.5, by + bh - ctx.px(1),
                              role='caption', color=palette.text_dimmer,
                              align='center', valign='bottom')
