"""Die informations-bloecke am bildschirmrand: schiff und ziel.

Alle benutzen dasselbe bauteil aus chrome.py -- gefaste ecken, doppelter
rahmen, notch-tab auf der kante. Ein panel bringt hier KEINE eigene kontur
mehr mit; genau das hielt die alte fassung als sammlung schwebender
lozenges zusammen statt als eine tafel.

Die bahnelemente stehen nicht mehr hier, sondern im navball-block
(hud/navball.py): AP und PE liest man waehrend eines brennmanoevers, also
genau dann, wenn der blick ohnehin auf dem instrument liegt.
"""

from ..core import FILL, Rect, Widget
from ..theme import with_alpha
from ..widgets import Readout
from ..widgets.panel import Panel
from . import chrome


class HudPanel(Panel):
    """Gefaster doppelrahmen mit notch-tab auf der unterkante.

    Stapelt seine kinder senkrecht mit festem abstand -- die info-panels
    sind alle einfache listen, und jedem kind von hand einen y-versatz zu
    geben waere nur eine fehlerquelle beim umbauen.
    """

    def __init__(self, glow_role='elem', gap=None, tab=None, **kwargs):
        kwargs.setdefault('radius', -7)
        kwargs.setdefault('padding', 12)
        # (None, None) = groesse aus measure(), also aus den kindern. Der
        # Widget-standardwert (0, 0) waere hier ein unsichtbares panel.
        kwargs.setdefault('size', (None, None))
        super().__init__(**kwargs)
        self.glow_role = glow_role
        self.gap = gap
        self.tab = tab

    def _gap_px(self, ctx):
        return ctx.px(self.gap if self.gap is not None else ctx.theme.spacing.lg)

    def tab_height(self, ctx):
        """Platz fuer den notch-tab UNTERHALB des rahmens.

        Er sitzt ausserhalb der kante, nicht darauf -- sonst verdeckt er die
        letzte zeile des panels.
        """
        if not self.tab:
            return 0.0
        return ctx.text.measure(self.tab, 'tab')[1] + ctx.px(4)

    def _frame_rect(self, ctx):
        return Rect(self.rect.x, self.rect.y, self.rect.w,
                    self.rect.h - self.tab_height(ctx))

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
        pad = ctx.px(self.padding if self.padding is not None else 12)
        gap = self._gap_px(ctx)
        probe = self._content_probe(ctx)
        total = 0.0
        visible = [c for c in self.children if c.visible]
        for index, child in enumerate(visible):
            total += child.desired_size(ctx, probe)[1]
            if index < len(visible) - 1:
                total += gap
        return (ctx.px(ctx.theme.panel_width),
                total + pad * 2.0 + self.tab_height(ctx))

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

    def content_rect(self, ctx):
        pad = ctx.px(self.padding if self.padding is not None else 12)
        return self._frame_rect(ctx).inset(pad)

    def draw(self, ctx):
        palette = ctx.theme.palette
        frame_rect = self._frame_rect(ctx)
        chrome.frame(
            ctx, frame_rect.x, frame_rect.y, frame_rect.w, frame_rect.h,
            cut=self.radius, fill=self.fill, glow_role=self.glow_role,
        )
        if self.tab:
            chrome.tab(ctx, self.tab, frame_rect.x + ctx.px(14),
                       frame_rect.bottom,
                       color=palette.accent_for(self.glow_role), edge='bottom')


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
        chrome.plate(ctx, x, y, width, height, cut=3,
                     fill=with_alpha(color, 0.14),
                     line=with_alpha(color, 0.55))
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


def build_target_panel(telemetry, **kwargs):
    """Ziel-block. Ziel ist der aktive BEZUGSKOERPER -- die simulation kennt
    keine eigene zielauswahl, und der bezugskoerper ist genau der, auf den
    sich alle uebrigen anzeigen beziehen.

    Er sitzt jetzt in der LINKEN spalte unter der schiffs-plakette statt
    frei am rechten rand: schiff, bezugskoerper und die werte dazwischen
    gehoeren zusammen, und die rechte bildschirmhaelfte bleibt so frei fuer
    die bahn.
    """
    panel = HudPanel(glow_role='target', gap=9,
                     tab=chrome.tab_text('TARGET', 'INFO'), **kwargs)
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

        # Die untere LINKE ecke bleibt scharf: dort dockt der koerper-
        # waehler an, und zwei gefaste ecken uebereinander saehen aus wie
        # zwei getrennte teile.
        ix, iy, iw, ih = chrome.frame(
            ctx, self.rect.x, self.rect.y, self.rect.w, height,
            glow_role='ship', corners=(True, True, True, False),
        )

        pad = ctx.px(11)
        middle = self.rect.center_y
        dot_x = ix + pad
        # Ein gefastes quadrat statt eines punktes -- dieselbe formsprache
        # wie alles andere.
        ctx.draw.rect(dot_x - ctx.px(3.5), middle - ctx.px(3.5),
                      ctx.px(7), ctx.px(7), fill=palette.ship,
                      radius=-ctx.px(2.0))

        text_x = dot_x + ctx.px(11)
        ctx.text.draw(name, text_x, middle, role='badge', color=palette.text,
                      valign='middle')
        ctx.text.draw(reference, ix + iw - pad, middle,
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
        chrome.frame(ctx, self.rect.x, self.rect.y, self.rect.w, self.rect.h,
                     glow_role=self.color_role)
        for index, entry in enumerate(self.entries):
            bx, by, bw, bh = self._button_rect(ctx, index)
            if index == self._hover_index and self.hovered:
                chrome.plate(ctx, bx, by, bw, bh, cut=4, fill=palette.hover,
                             line=(0.0, 0.0, 0.0, 0.0))
            ctx.text.draw(str(entry['key']), bx + bw * 0.5, by + bh * 0.5,
                          role='button_sm', color=color,
                          align='center', valign='middle')
            value = entry.get('value')
            if value is not None:
                text = value() if callable(value) else str(value)
                ctx.text.draw(text, bx + bw * 0.5, by + bh - ctx.px(1),
                              role='caption', color=palette.text_dimmer,
                              align='center', valign='bottom')
