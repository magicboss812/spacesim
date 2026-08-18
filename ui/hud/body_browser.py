"""Ausklappbare koerperliste zur wahl des BEZUGSKOERPERS.

Bisher liess sich der bezugskoerper nur mit der taste R durchblaettern --
eine reihenfolge, die man nicht sieht, ohne ziel und ohne rueckweg. Das ist
genau die art undurchschaubarer tastenbelegung, die dieser HUD ersetzen
soll: der bezugskoerper bestimmt saemtliche bahnwerte, er gehoert direkt
waehlbar.

AUFBAU DER LISTE. Die gliederung wird NICHT von hand gepflegt, sondern aus
den ``is_moon_of``-verweisen gelesen, die der loader ohnehin schon zu
objektverweisen aufloest (zweiter durchgang in SystemLoader). Daraus folgt:

- koerper ohne ``is_moon_of`` sind wurzeln und stehen oben, nach masse
  absteigend -- das zentralgestirn zuerst
- jeder weitere koerper haengt eingerueckt unter seinem mutterkoerper
- geschwister sind nach ``semi_major_axis`` sortiert, also nach ihrem
  abstand zum mutterkoerper (der sonne am naechsten zuerst)

Damit ordnet sich ein erweitertes ``solar_system.json`` von selbst ein; es
gibt keine zweite, mitzupflegende reihenfolge.

Das schiff steht NICHT in der liste. Ein bezugsrahmen, der auf dem schiff
sitzt, macht die eigene bahn zu einem punkt -- dieselbe begruendung wie bei
``UIState.celestial_indices``, und die auswahl wird gegen diese liste
geprueft.
"""

import math

from ..core import Widget
from ..theme import readable, with_alpha
from .. import units

# Entwurfseinheiten, wie ueberall im HUD -- niemals pixel.
_BUTTON = 34.0
_PANEL_WIDTH = 214.0
_ROW_HEIGHT = 26.0
_PADDING = 12.0
_HEADER = 18.0
_INDENT = 13.0
_DOT_RADIUS = 4.0
_GAP = 8.0


def _body_dot_color(body):
    """``body.color`` (0..255-tripel aus dem JSON) -> zeichenfarbe.

    Bewusst UNVERAENDERT uebernommen: der punkt ist eine FLAECHE, und fuer
    flaechen gilt die rohfarbe (nur schrift und duenne striche laufen ueber
    theme.readable). Er soll den koerper wiedererkennbar machen, wie er auch
    in der welt gezeichnet wird.
    """
    raw = getattr(body, 'color', None) or (255, 255, 255)
    try:
        r, g, b = (float(raw[0]), float(raw[1]), float(raw[2]))
    except Exception:
        r = g = b = 255.0
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)


def build_hierarchy(bodies):
    """Flache liste von zeilen: ``(index, body, tiefe)``, in anzeigereihenfolge.

    Flach statt verschachtelt, weil das zeichnen und das treffer-suchen
    ohnehin ueber laufende zeilennummern gehen. Die einrueckung steckt in
    der tiefe.

    Robust gegen kaputte daten: ein ``is_moon_of``-zyklus oder ein
    mutterkoerper ausserhalb der liste wuerde eine rekursion sonst nie
    beenden. Alles, was nach dem durchlauf nicht besucht wurde, wird darum
    am ende als wurzel angehaengt -- lieber falsch eingerueckt als unsichtbar.
    """
    entries = [
        (index, b) for index, b in enumerate(bodies)
        if not getattr(b, 'is_ship', False)
    ]
    known = {id(b): index for index, b in entries}

    children = {}
    roots = []
    for index, b in entries:
        parent = getattr(b, 'is_moon_of', None)
        if parent is not None and id(parent) in known:
            children.setdefault(id(parent), []).append((index, b))
        else:
            roots.append((index, b))

    def orbit_key(item):
        _index, b = item
        a = getattr(b, 'semi_major_axis', None)
        try:
            a = float(a)
        except (TypeError, ValueError):
            a = None
        if a is None or not math.isfinite(a) or a <= 0.0:
            # Kein bahnradius bekannt -> ans ende, aber stabil nach namen.
            return (1, 0.0, str(getattr(b, 'name', '')))
        return (0, a, str(getattr(b, 'name', '')))

    # Wurzeln nach MASSE absteigend: das zentralgestirn zuerst. Ein
    # bahnradius steht ihnen nicht zur verfuegung -- sie umkreisen nichts.
    roots.sort(key=lambda item: -float(getattr(item[1], 'mass', 0.0) or 0.0))

    rows = []
    seen = set()

    def walk(item, depth):
        index, b = item
        if id(b) in seen:
            return
        seen.add(id(b))
        rows.append((index, b, depth))
        for child in sorted(children.get(id(b), []), key=orbit_key):
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
    for item in entries:
        if id(item[1]) not in seen:
            walk(item, 0)
    return rows


class BodyBrowser(Widget):
    """Symbolknopf plus die dahinter liegende koerperliste.

    EIN widget statt knopf + panel, aus demselben grund wie bei
    ``PaletteButton``: die liste ist ein ueberlagerndes ausklapp-element.
    Als eigenstaendiges widget muesste ihre trefferflaeche mit der des
    knopfes von hand synchron gehalten werden; so erweitert schlicht
    ``hit_test`` die flaeche, solange die liste offen ist.
    """

    def __init__(self, telemetry, ui_state, side='left', **kwargs):
        kwargs.setdefault('size', (_BUTTON, _BUTTON))
        kwargs.setdefault('z', 150)
        super().__init__(**kwargs)
        self.telemetry = telemetry
        self.ui_state = ui_state
        self.side = side
        self.blocks_mouse = True
        self.open = False
        self._hover_row = -1
        self._rows = None
        self._rows_source = None

    # ------------------------------------------------------------- gliederung

    def rows(self):
        """Gecachte gliederung. Die koerperliste aendert sich zur laufzeit
        nicht, der aufbau muss also nicht pro frame passieren."""
        bodies = self.ui_state.bodies if self.ui_state is not None else []
        if self._rows is None or self._rows_source is not bodies:
            self._rows = build_hierarchy(bodies)
            self._rows_source = bodies
        return self._rows

    # ---------------------------------------------------------------- geometrie

    def _panel_rect(self, ctx):
        rows = self.rows()
        width = ctx.px(_PANEL_WIDTH)
        pad = ctx.px(_PADDING)
        height = (pad * 2.0 + ctx.px(_HEADER) + ctx.px(_GAP)
                  + len(rows) * ctx.px(_ROW_HEIGHT))
        # Die liste haengt unter dem knopf und richtet sich an DERSELBEN
        # kante aus wie er -- links verankert nach rechts, rechts verankert
        # nach links. Sonst ragt sie bei rechter verankerung aus dem bild.
        if self.side == 'right':
            x = self.rect.right - width
        else:
            x = self.rect.x
        y = self.rect.bottom + ctx.px(_GAP)
        # Nach unten begrenzen: bei vielen koerpern liefe die liste sonst
        # aus dem fenster, und die untersten zeilen waeren unerreichbar.
        max_height = max(ctx.px(_ROW_HEIGHT), ctx.height - y - ctx.px(14))
        return (x, y, width, min(height, max_height))

    def _row_rect(self, ctx, index):
        px, py, pw, _ph = self._panel_rect(ctx)
        pad = ctx.px(_PADDING)
        top = py + pad + ctx.px(_HEADER) + ctx.px(_GAP) + index * ctx.px(_ROW_HEIGHT)
        return (px + pad * 0.5, top, pw - pad, ctx.px(_ROW_HEIGHT))

    def hit_test(self, ctx, x, y):
        if not self.visible:
            return False
        if self.rect.contains(x, y):
            return True
        if not self.open:
            return False
        px, py, pw, ph = self._panel_rect(ctx)
        return px <= x < px + pw and py <= y < py + ph

    # ------------------------------------------------------------------ eingabe

    def dismiss(self):
        self.open = False
        self._hover_row = -1

    def on_mouse_move(self, ctx, x, y):
        self._hover_row = -1
        if self.open:
            for index in range(len(self.rows())):
                bx, by, bw, bh = self._row_rect(ctx, index)
                if bx <= x < bx + bw and by <= y < by + bh:
                    self._hover_row = index
                    break
        return True

    def on_mouse_up(self, ctx, x, y, button):
        if button != 1:
            return True
        if self.open:
            for row_index, (body_index, _body, _depth) in enumerate(self.rows()):
                bx, by, bw, bh = self._row_rect(ctx, row_index)
                if bx <= x < bx + bw and by <= y < by + bh:
                    if self.ui_state is not None:
                        self.ui_state.set_reference_index(body_index)
                    # Nach der wahl schliessen: die liste hat ihren zweck
                    # erfuellt und verdeckt sonst den blick auf genau die
                    # werte, die sich soeben geaendert haben.
                    self.dismiss()
                    return True
            if not self.rect.contains(x, y):
                return True
        if self.rect.contains(x, y):
            self.open = not self.open
            self._hover_row = -1
        return True

    # ----------------------------------------------------------------- zeichnen

    def draw(self, ctx):
        palette = ctx.theme.palette
        radius = self.rect.h * 0.5
        active = self.open or self.hovered
        ctx.draw.rect(
            self.rect.x, self.rect.y, self.rect.w, self.rect.h,
            fill=palette.panel_pill, radius=radius,
            border_color=palette.accent_for('target') if self.open else palette.edge,
            border_width=ctx.theme.border_width,
            shadow=ctx.theme.glow('target') if active else None,
            shadow_offset=(0.0, 0.0), shadow_softness=ctx.px(18.0),
        )
        self._draw_icon(ctx)
        if self.open:
            self._draw_panel(ctx)

    def _draw_icon(self, ctx):
        """Ein system in klein: zentralkoerper, bahn, trabant.

        Gezeichnet statt gesetzt -- dieselbe entscheidung wie bei den
        bahnmarkern des rings. Ein passendes zeichen (U+1F784 o. ae.) fehlt
        in den meisten oberflaechen-schriften; ein vektor stimmt immer.
        """
        palette = ctx.theme.palette
        cx, cy = self.rect.center_x, self.rect.center_y
        color = readable(palette.accent_for('target'))
        tint = color if self.open or self.hovered else palette.text_dim
        ctx.draw.circle(cx, cy, ctx.px(3.0), fill=tint)
        orbit = ctx.px(9.0)
        ctx.draw.ring(cx, cy, orbit, max(1.0, ctx.px(1.2)), with_alpha(tint, 0.75))
        ctx.draw.circle(cx + orbit * 0.707, cy - orbit * 0.707, ctx.px(2.0),
                        fill=tint)

    def _draw_panel(self, ctx):
        palette = ctx.theme.palette
        px, py, pw, ph = self._panel_rect(ctx)
        ctx.draw.rect(
            px, py, pw, ph, fill=palette.panel_popup, radius=ctx.px(16),
            border_color=palette.edge_strong, border_width=ctx.theme.border_width,
            shadow=palette.shadow, shadow_offset=(0.0, ctx.px(-6)),
            shadow_softness=ctx.px(20),
        )
        pad = ctx.px(_PADDING)
        ctx.text.draw('REFERENCE BODY', px + pad, py + pad, role='section',
                      color=palette.text_dim)

        reference = self.ui_state.reference_index if self.ui_state else None
        bottom = py + ph
        for row_index, (body_index, body, depth) in enumerate(self.rows()):
            bx, by, bw, bh = self._row_rect(ctx, row_index)
            if by + bh > bottom:
                # Ueber die begrenzte hoehe hinaus wird nicht gezeichnet --
                # sonst laege die zeile ausserhalb des panels in der luft.
                break
            selected = body_index == reference
            if selected:
                ctx.draw.rect(bx, by, bw, bh,
                              fill=with_alpha(palette.accent_for('target'), 0.20),
                              radius=bh * 0.5)
            elif row_index == self._hover_row:
                ctx.draw.rect(bx, by, bw, bh, fill=palette.hover,
                              radius=bh * 0.5)

            middle = by + bh * 0.5
            dot_x = bx + ctx.px(10.0) + ctx.px(_INDENT) * depth
            ctx.draw.circle(dot_x, middle, ctx.px(_DOT_RADIUS),
                            fill=_body_dot_color(body))

            name = str(getattr(body, 'name', '?'))
            ctx.text.draw(
                name, dot_x + ctx.px(11.0), middle, role='body',
                color=palette.text if selected else palette.text_muted,
                valign='middle',
            )

            # Der bahnradius rechts belegt genau die sortierung dieser zeile.
            a = getattr(body, 'semi_major_axis', None)
            try:
                a = float(a)
            except (TypeError, ValueError):
                a = None
            if a and math.isfinite(a) and a > 0.0:
                ctx.text.draw(
                    units.distance(a, digits=1), bx + bw - ctx.px(10.0), middle,
                    role='caption', color=palette.text_dimmer,
                    align='right', valign='middle',
                )
