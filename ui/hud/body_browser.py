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

from ..core import Widget, ease
from ..theme import readable, with_alpha
from . import chrome
from .. import units

# Entwurfseinheiten, wie ueberall im HUD -- niemals pixel.
#
# Der knopf ist so BREIT wie das ziel-panel darunter und die plakette
# darueber. Er war einmal ein 34x34-quadrat mit einem symbol darin und sass
# damit als einzelnes kleines kaestchen zwischen zwei breiten bloecken -- die
# linke spalte las sich als drei zufaellig gestapelte teile statt als eine
# spalte. Gleiche breite ist hier die ganze arbeit.
_BUTTON = 34.0
_BUTTON_WIDTH = 190.0
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
        kwargs.setdefault('size', (_BUTTON_WIDTH, _BUTTON))
        kwargs.setdefault('z', 150)
        super().__init__(**kwargs)
        self.telemetry = telemetry
        self.ui_state = ui_state
        self.side = side
        self.blocks_mouse = True
        self.open = False
        # Aufklapp-fortschritt, 0..1. Getrennt von `open`, weil `open` das
        # ZIEL ist und dieser wert der gezeichnete zustand -- beim schliessen
        # laeuft die bewegung damit rueckwaerts, statt einfach zu verschwinden.
        self._open_t = 0.0
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

    def _panel_rect_open(self, ctx):
        """Das panel in seiner MOMENTANEN aufklapp-hoehe.

        Es waechst aus der unterkante des knopfes NACH UNTEN heraus: x, y
        und breite bleiben fest, nur die hoehe laeuft. Ein einflug von der
        seite haette so ausgesehen, als kaeme die liste von woanders her --
        sie gehoert aber zu genau diesem knopf, und das soll die bewegung
        sagen.
        """
        x, y, w, h = self._panel_rect(ctx)
        return (x, y, w, h * self._open_t)

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
        # Nur die AUFGEKLAPPTE flaeche faengt, und nur solange die liste
        # offen sein SOLL: eine zuklappende liste ist noch sichtbar, darf
        # aber keine klicks mehr verschlucken.
        if not self.open:
            return False
        px, py, pw, ph = self._panel_rect_open(ctx)
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

    def update(self, ctx, dt):
        super().update(ctx, dt)
        # Dieselbe framerate-unabhaengige formel wie ueberall sonst
        # (1 - exp(-rate * dt)); ein fester schritt je frame waere bei
        # 180 fps eine andere bewegung als bei 60.
        # motion.normal, nicht .fast: bei rate 22 springt der erste frame um
        # 30 % der hoehe, und die bewegung liest sich als schnappen statt als
        # aufklappen. Rate 14 sind ~0.21 s bis 95 % -- lang genug, dass man
        # die richtung sieht, kurz genug, dass niemand darauf wartet.
        self._open_t = ease(self._open_t, 1.0 if self.open else 0.0,
                            ctx.theme.motion.normal, dt)
        if not self.open and self._open_t < 0.004:
            self._open_t = 0.0

    # ----------------------------------------------------------------- zeichnen

    def draw(self, ctx):
        palette = ctx.theme.palette
        active = self.open or self.hovered
        # OBEN LINKS SCHARF -- dort sitzt die schiffs-plakette darueber, die
        # ihre untere linke ecke aus demselben grund scharf laesst. Die
        # beiden lesen sich damit als eine spalte, nicht als zwei kaesten.
        chrome.plate(
            ctx, self.rect.x, self.rect.y, self.rect.w, self.rect.h,
            fill=palette.panel_pill,
            line=palette.accent_for('target') if active else palette.edge,
            corners=(False, True, True, True),
        )
        self._draw_icon(ctx)
        # Gezeichnet wird nach _open_t, nicht nach `open` -- sonst
        # verschwaende das zuklappen ohne bewegung.
        if self._open_t > 0.004:
            self._draw_panel(ctx)

    def _draw_icon(self, ctx):
        """Symbol, beschriftung und der aufklapp-winkel.

        Das systemsymbol (zentralkoerper, bahn, trabant) ist GEZEICHNET,
        nicht gesetzt -- dieselbe entscheidung wie bei den bahnmarkern des
        rings. Ein passendes zeichen (U+1F784 o. ae.) fehlt in den meisten
        oberflaechen-schriften; ein vektor stimmt immer.
        """
        palette = ctx.theme.palette
        middle = self.rect.center_y
        color = readable(palette.accent_for('target'))
        tint = color if self.open or self.hovered else palette.text_dim

        orbit = ctx.px(9.0)
        cx = self.rect.x + ctx.px(_PADDING) + orbit
        ctx.draw.circle(cx, middle, ctx.px(3.0), fill=tint)
        ctx.draw.ring(cx, middle, orbit, max(1.0, ctx.px(1.2)),
                      with_alpha(tint, 0.75))
        ctx.draw.circle(cx + orbit * 0.707, middle - orbit * 0.707,
                        ctx.px(2.0), fill=tint)

        # Der knopf sagt jetzt selbst, was er tut. Vorher stand hier nur das
        # symbol, und der einzige hinweis auf den bezugskoerper war das
        # kuerzel in der plakette darueber.
        ctx.text.draw('REFERENCE', cx + orbit + ctx.px(11), middle,
                      role='caption', color=palette.text_dim, valign='middle')
        arrow = self.rect.right - ctx.px(_PADDING)
        span = ctx.px(4.0)
        # Winkel nach unten (zu) bzw. nach oben (offen).
        direction = -1.0 if self.open else 1.0
        ctx.draw.line(arrow - span * 2.0, middle - span * 0.5 * direction,
                      arrow - span, middle + span * 0.5 * direction, tint,
                      width=max(1.0, ctx.px(1.4)), cap='round')
        ctx.draw.line(arrow, middle - span * 0.5 * direction,
                      arrow - span, middle + span * 0.5 * direction, tint,
                      width=max(1.0, ctx.px(1.4)), cap='round')

    def _draw_panel(self, ctx):
        palette = ctx.theme.palette
        px, py, pw, ph = self._panel_rect_open(ctx)
        chrome.frame(ctx, px, py, pw, ph, fill=palette.panel_popup,
                     line=palette.edge_strong, glow_role='target')
        pad = ctx.px(_PADDING)
        if ph > pad + ctx.px(_HEADER):
            ctx.text.draw('REFERENCE BODY', px + pad, py + pad, role='section',
                          color=with_alpha(palette.text_dim,
                                           min(1.0, self._open_t * 2.5)))

        reference = self.ui_state.reference_index if self.ui_state else None
        bottom = py + ph
        row_h = ctx.px(_ROW_HEIGHT)
        for row_index, (body_index, body, depth) in enumerate(self.rows()):
            bx, by, bw, bh = self._row_rect(ctx, row_index)
            if by >= bottom:
                # Ueber die momentane hoehe hinaus wird nicht gezeichnet --
                # sonst laege die zeile ausserhalb des panels in der luft.
                break
            # Jede zeile blendet GENAU DANN auf, wenn die wachsende
            # unterkante sie ueberstreicht. Ohne das erschiene sie
            # schlagartig, und das waere das einzige an der bewegung, was
            # noch ruckelt.
            reveal = max(0.0, min(1.0, (bottom - by) / max(row_h, 1e-6)))
            selected = body_index == reference
            cut = -ctx.px(4.0)
            if selected:
                ctx.draw.rect(bx, by, bw, bh,
                              fill=with_alpha(palette.accent_for('target'),
                                              0.20 * reveal),
                              radius=(cut, 0.0, cut, 0.0))
            elif row_index == self._hover_row and self.open:
                ctx.draw.rect(bx, by, bw, bh,
                              fill=with_alpha(palette.hover,
                                              palette.hover[3] * reveal),
                              radius=(cut, 0.0, cut, 0.0))

            middle = by + bh * 0.5
            dot_x = bx + ctx.px(10.0) + ctx.px(_INDENT) * depth
            ctx.draw.circle(dot_x, middle, ctx.px(_DOT_RADIUS),
                            fill=with_alpha(_body_dot_color(body), reveal))

            name = str(getattr(body, 'name', '?'))
            ctx.text.draw(
                name, dot_x + ctx.px(11.0), middle, role='body',
                color=with_alpha(
                    palette.text if selected else palette.text_muted, reveal),
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
                    role='caption',
                    color=with_alpha(palette.text_dimmer, reveal),
                    align='right', valign='middle',
                )
