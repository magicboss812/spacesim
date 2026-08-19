"""Der Attitude-Ring -- das Herzstueck des HUDs.

Ein 2D-lagemesser statt einer navball-kugel: die simulation kennt GENAU
EINEN orientierungswinkel (schiff.theta). Eine schattierte kugel wuerde
nick- und rollachsen anzeigen, die es in dieser physik nicht gibt -- in
einer Seminararbeit waeren das erfundene daten. Der ring zeigt nur, was
wirklich existiert: kurs, bahnrichtungen und geschwindigkeit.

Aufbau (masse in den einheiten des entwurfs, viewBox 220, gezeichnet mit 212):
- aussenring r=103, teilung r=92 mit strichen alle 7.5 Grad
- kursbeschriftung bei 000/090/180/270 auf r=68, immer aufrecht
- vier bahnmarker (prograde, retrograde, normal, antinormal) auf r=76
- geschwindigkeitsnadel aus der mitte, laenge nach betrag
- schiffsnase fest oben -- der RING dreht sich, nicht die nase

KONVENTION: kurse sind kompasswinkel (0 = oben, im uhrzeigersinn). Die
umrechnung aus theta macht telemetry.compass_from_theta -- an genau einer
stelle, damit ring, pfeil und autopilot nicht auseinanderlaufen.
"""

import math

from ..core import Widget, ease
from ..theme import with_alpha

# Der entwurf ist in einer 220er viewBox gezeichnet; alle rohmasse unten
# stehen in diesen einheiten und werden mit size/220 skaliert.
_VIEWBOX = 220.0
_CENTER = 110.0
_TICK_RADIUS = 92.0
_OUTER_RADIUS = 103.0
# Radien wie im entwurf. Die beschriftung MUSS auf 68 bleiben: weiter innen
# laeuft sie in den mittleren messwertblock (die kurs-plakette sitzt bei
# y+38..+58), weiter aussen in die teilung. Dass marker und beschriftung sich
# gelegentlich treffen, wird stattdessen in _draw_tick_labels abgefangen.
_LABEL_RADIUS = 68.0
_MARKER_RADIUS = 76.0
# Ab dieser winkeldifferenz verdeckt ein marker eine himmelsrichtung.
# Marker (halbwinkel ~7 Grad bei r=76) plus beschriftung (~9 Grad bei r=68).
_LABEL_HIDE_DEG = 16.0
_INNER_RADIUS = 74.0

# Laengenbereich der geschwindigkeitsnadel, in denselben viewBox-einheiten.
# Die obergrenze bleibt INNERHALB der teilung (_TICK_RADIUS = 92), damit
# eine fluchtbahn die nadel nicht in den kursring laufen laesst.
_NEEDLE_MIN = 22.0
_NEEDLE_MAX = 66.0

# Die farbe folgt der ACHSE, nicht der reihenfolge: prograde und retrograde
# sind dieselbe achse und tragen deshalb dieselbe farbe, normal und
# antinormal ebenso. Vier verschiedene farben fuer zwei achsenpaare haetten
# die zusammengehoerigkeit gerade verdeckt -- und dieselbe zuordnung
# benutzt die snap-rosette (controls.SnapRosette.MODES).
_MARKER_ORDER = (
    ('prograde', 'PRO', 'velocity'),
    ('retrograde', 'RETRO', 'velocity'),
    ('normal_in', 'NORM', 'normal'),
    ('antinormal_out', 'ANTI', 'normal'),
)


def _polar(center_x, center_y, radius, compass_deg):
    """Kompasswinkel -> punkt in TOP-DOWN pixeln (0 Grad = oben, im uhrzeigersinn)."""
    angle = math.radians(float(compass_deg) - 90.0)
    return (
        center_x + math.cos(angle) * radius,
        center_y + math.sin(angle) * radius,
    )


class AttitudeRing(Widget):
    """Lagemesser mit kursring, bahnmarkern und geschwindigkeitsanzeige.

    Ziehen dreht das schiff, SOLANGE DIE TASTE GEHALTEN WIRD: der gegriffene
    ringpunkt bleibt unter dem cursor, daraus folgt der sollkurs, den
    schiffcontrol.orient_towards_angle mit der konfigurierten drehrate
    anfaehrt. Direkt theta zu schreiben waere unphysikalisch -- ein
    raumfahrzeug dreht sich nicht sprunghaft -- und wuerde ausserdem gegen
    den rastenden autopiloten arbeiten.

    Beim loslassen gibt der ring die steuerung vollstaendig zurueck (siehe
    on_mouse_up); ein hier haengengebliebener sollwert sperrt sonst die
    pfeiltasten aus.
    """

    def __init__(self, telemetry, ship_control=None, size=(212, 212),
                 hub_only=False, **kwargs):
        super().__init__(size=size, **kwargs)
        self.telemetry = telemetry
        self.ship_control = ship_control
        self.blocks_mouse = True
        self._manual_heading = None
        self._drag_offset = 0.0
        self._display_heading = 0.0
        # Im navball-block steht der kurs in einer plakette UEBER der kugel
        # (so wie in der vorlage); die ringmitte traegt dann nur noch eine
        # kleine nabe, die das innerste stueck der nadel abdeckt.
        self.hub_only = bool(hub_only)

    # --------------------------------------------------------------- geometrie

    def _scale(self, ctx):
        return min(self.rect.w, self.rect.h) / _VIEWBOX

    def _center(self):
        return (self.rect.center_x, self.rect.center_y)

    # ----------------------------------------------------------------- eingabe

    def hit_test(self, ctx, x, y):
        """RUND, nicht quadratisch.

        Das widget ist ein quadrat, der ring darin ein kreis -- die vier
        ecken des quadrats sind leer, gehoerten der trefferpruefung aber
        trotzdem. Da der ring im navball-block VOR den zellenbogen liegt,
        verschluckte er damit genau die klicks auf deren obere und untere
        enden: der schub liess sich in seinem oberen drittel nicht stellen,
        obwohl der bogen sichtbar frei lag. Eine ecke, die nichts zeichnet,
        darf auch nichts fangen.
        """
        if not self.visible:
            return False
        cx, cy = self._center()
        outer = _OUTER_RADIUS * self._scale(ctx)
        return math.hypot(float(x) - cx, float(y) - cy) <= outer

    def _cursor_compass(self, x, y):
        cx, cy = self._center()
        dx = float(x) - cx
        dy = float(y) - cy
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        # atan2(dx, -dy): 0 = oben, waechst im uhrzeigersinn.
        return math.degrees(math.atan2(dx, -dy)) % 360.0

    def on_mouse_down(self, ctx, x, y, button):
        if button != 1 or self.ship_control is None:
            return True
        cursor = self._cursor_compass(x, y)
        if cursor is None:
            return True
        # Ziehen loest den rastenden autopiloten -- sonst zoege der spieler
        # gegen einen regler, der die nase jeden frame zurueckreisst.
        try:
            self.ship_control.clear_snap()
        except Exception:
            pass
        # GEGRIFFEN WIRD DER RING, NICHT DIE NASE. Die teilung wird bei
        # (deg - heading) gezeichnet, ein kurs steigt also, wenn sich der
        # ring GEGEN den uhrzeigersinn dreht. Wer den ring anfasst und im
        # uhrzeigersinn zieht, sah ihn deshalb rueckwaerts laufen.
        #
        # Festgehalten wird der ringpunkt unter dem cursor: der kurswert
        # dort ist (cursor + heading) und bleibt waehrend des ziehens
        # konstant -- genau das heisst "der punkt bleibt unter dem finger".
        current = self.telemetry.heading
        self._drag_offset = (current + cursor) % 360.0
        self._manual_heading = current
        return True

    def on_mouse_move(self, ctx, x, y):
        if not self.pressed or self.ship_control is None:
            return False
        cursor = self._cursor_compass(x, y)
        if cursor is None:
            return False
        self._manual_heading = (self._drag_offset - cursor) % 360.0
        return True

    def on_mouse_up(self, ctx, x, y, button):
        """Loslassen gibt das schiff SOFORT wieder frei.

        Ohne das blieb _manual_heading stehen und update() rief weiter
        orient_towards_angle() auf. Das ist nicht bloss ein weiterlaufender
        sollwert: sobald der kurs erreicht ist, setzt orient_towards_angle
        intern _snap_locked und heftet theta danach JEDEN frame auf den
        zielwert. Die pfeiltasten schrieben theta zwar noch, der wert wurde
        aber im selben frame wieder ueberschrieben -- das schiff liess sich
        nach einem einzigen zieh-vorgang nur noch ueber den ring steuern.
        """
        if button != 1:
            return True
        self._manual_heading = None
        if self.ship_control is not None:
            try:
                # Hebt die _snap_locked-heftung auf, die das ziehen gesetzt
                # haben kann. snap_mode ist bereits seit on_mouse_down None.
                self.ship_control.clear_snap()
            except Exception:
                pass
        return True

    def update(self, ctx, dt):
        super().update(ctx, dt)
        # Sollkurs mit der konfigurierten drehrate anfahren. Gesetzt ist er
        # nur zwischen on_mouse_down und on_mouse_up; ein zwischendurch
        # gerasteter autopilot hat vorrang und beendet das ziehen.
        if self._manual_heading is not None and self.ship_control is not None:
            if getattr(self.ship_control, 'snap_mode', None) is not None:
                self._manual_heading = None
            else:
                # kompass -> theta ist die umkehrung von compass_from_theta.
                target_theta = math.radians(90.0 - self._manual_heading)
                try:
                    self.ship_control.orient_towards_angle(target_theta, dt)
                except Exception:
                    self._manual_heading = None

        # Die ANZEIGE laeuft dem echten kurs weich nach. Bei hohem zeitraffer
        # springt theta pro frame um mehrere grad; ungeglaettet flackert die
        # ganze teilung.
        target = self.telemetry.heading
        delta = (target - self._display_heading + 180.0) % 360.0 - 180.0
        self._display_heading = (
            ease(self._display_heading, self._display_heading + delta,
                 ctx.theme.motion.fast, dt) % 360.0
        )

    # ---------------------------------------------------------------- zeichnen

    def draw(self, ctx):
        palette = ctx.theme.palette
        scale = self._scale(ctx)
        cx, cy = self._center()
        heading = self._display_heading

        outer = _OUTER_RADIUS * scale

        # DOPPELTER RAHMEN, wie an jedem block der tafel: eine haarfeine
        # aussenlinie, ein spalt, dann die flaeche. Ein einzelner kreis mit
        # fuellung sieht aus wie ein widget, zwei sehen aus wie ein geraet.
        ctx.draw.ring(
            cx, cy, outer, max(1.0, 1.0 * scale), palette.edge,
        )
        ctx.draw.circle(
            cx, cy, outer - 3.0 * scale,
            fill=palette.ring_face,
            border_color=with_alpha(palette.ring, 0.55),
            border_width=max(1.0, 1.4 * scale),
            shadow=palette.shadow,
            shadow_offset=(0.0, -3.0 * scale),
            shadow_softness=14.0 * scale,
        )

        self._draw_grid(ctx, cx, cy, scale)
        self._draw_ticks(ctx, cx, cy, scale, heading)
        self._draw_tick_labels(ctx, cx, cy, scale, heading)

        ctx.draw.ring(
            cx, cy, _INNER_RADIUS * scale, max(1.0, 1.0 * scale),
            palette.edge_inner,
        )

        self._draw_markers(ctx, cx, cy, scale, heading)
        self._draw_velocity_needle(ctx, cx, cy, scale, heading)
        self._draw_readout(ctx, cx, cy, scale)
        self._draw_nose(ctx, cx, cy, scale, outer)

    def _draw_grid(self, ctx, cx, cy, scale):
        """Das schwache polargitter auf der ringflaeche.

        Es macht die flaeche zu einem instrument statt zu einer scheibe --
        und es ist, anders als eine schattierte kugel, KEINE erfundene
        angabe: speichen und kreise sind nur das koordinatennetz derselben
        kompassebene, die der ring ohnehin zeigt.
        """
        palette = ctx.theme.palette
        spoke = with_alpha(palette.ring, 0.13)
        radius = _INNER_RADIUS * scale
        for index in range(6):        # alle 30 grad, gegenueberliegend
            deg = index * 30.0
            x0, y0 = _polar(cx, cy, radius, deg)
            x1, y1 = _polar(cx, cy, radius, deg + 180.0)
            ctx.draw.line(x0, y0, x1, y1, spoke, width=max(1.0, 1.0 * scale))
        for fraction in (0.34, 0.67):
            ctx.draw.ring(cx, cy, radius * fraction,
                          max(1.0, 1.0 * scale), spoke)

    def _draw_ticks(self, ctx, cx, cy, scale, heading):
        palette = ctx.theme.palette
        radius = _TICK_RADIUS * scale
        minor_color = ctx.theme.palette.text_dimmer

        step = 7.5
        count = int(round(360.0 / step))
        for index in range(count):
            deg = index * step
            major = (index % 6) == 0        # alle 45 Grad
            mid = (index % 2) == 0          # alle 15 Grad
            length = (16.0 if major else 10.0 if mid else 5.0) * scale
            width = (2.2 if major else 1.6 if mid else 1.2) * scale
            if major:
                color = palette.ring
            elif mid:
                color = with_alpha(palette.ring, 0.6)
            else:
                color = with_alpha(minor_color, 0.75)

            on_ring = deg - heading
            x0, y0 = _polar(cx, cy, radius, on_ring)
            x1, y1 = _polar(cx, cy, radius - length, on_ring)
            ctx.draw.line(x0, y0, x1, y1, color, width=max(1.0, width), cap='round')

    def _draw_tick_labels(self, ctx, cx, cy, scale, heading):
        """Die vier himmelsrichtungen, immer aufrecht.

        Eine richtung wird WEGGELASSEN, wenn ein bahnmarker davor steht.
        Beide sitzen auf fast demselben radius, und sobald man prograde
        haelt -- der normalfall -- rastet der marker genau auf eine
        himmelsrichtung ein. Der marker ist dann die wichtigere information;
        die zahl darunter waere ohnehin nur noch als bruchstueck lesbar.
        """
        palette = ctx.theme.palette
        radius = _LABEL_RADIUS * scale
        occupied = [
            value for value in self.telemetry.marker_headings.values()
            if value is not None
        ]
        for deg in (0, 90, 180, 270):
            if any(abs((deg - marker + 180.0) % 360.0 - 180.0) < _LABEL_HIDE_DEG
                   for marker in occupied):
                continue
            x, y = _polar(cx, cy, radius, deg - heading)
            # Die beschriftung dreht NICHT mit: eine kopfstehende '180' ist
            # unlesbar, und der entwurf dreht sie im SVG ebenfalls zurueck.
            ctx.text.draw(
                f"{deg:03d}", x, y, role='ring_tick', color=palette.ring,
                align='center', valign='middle',
            )

    def _draw_markers(self, ctx, cx, cy, scale, heading):
        palette = ctx.theme.palette
        radius = _MARKER_RADIUS * scale
        marker_r = 9.0 * scale
        active = self.telemetry.snap_mode

        for key, _label, role in _MARKER_ORDER:
            compass = self.telemetry.marker_headings.get(key)
            if compass is None:
                continue
            color = palette.accent_for(role)
            x, y = _polar(cx, cy, radius, compass - heading)

            highlight = (key == active)
            # Deckende fuellung: der marker liegt ueber der kursbeschriftung
            # und muss sie sauber abdecken, nicht durchscheinen lassen.
            ctx.draw.circle(
                x, y, marker_r,
                fill=with_alpha(color, 0.30) if highlight
                else with_alpha(palette.ring_face, 1.0),
                border_color=color,
                border_width=max(1.0, (2.4 if highlight else 1.8) * scale),
            )
            self._draw_marker_glyph(ctx, key, x, y, marker_r, color, scale)

    def _draw_marker_glyph(self, ctx, key, x, y, radius, color, scale):
        """Die vier bahnsymbole als VEKTOREN, nicht als schriftzeichen.

        Der entwurf setzt hier die zeichen ◉ und ⊗ (U+25C9 / U+2297). Beide
        fehlen in vielen oberflaechen-schriften und erschienen dann als
        leeres kaestchen -- gezeichnet sind sie unter jeder schrift korrekt.
        """
        if key == 'prograde':
            ctx.draw.circle(x, y, radius * 0.34, fill=color)
        elif key == 'retrograde':
            arm = radius * 0.42
            width = max(1.0, 1.6 * scale)
            ctx.draw.line(x - arm, y - arm, x + arm, y + arm, color,
                          width=width, cap='round')
            ctx.draw.line(x - arm, y + arm, x + arm, y - arm, color,
                          width=width, cap='round')
        else:
            ctx.text.draw(
                'N' if key == 'normal_in' else 'A', x, y,
                role='ring_marker', color=color, align='center', valign='middle',
            )

    def _draw_velocity_needle(self, ctx, cx, cy, scale, heading):
        palette = ctx.theme.palette
        compass = self.telemetry.marker_headings.get('prograde')
        if compass is None:
            return
        # MASSSTAB AUS DER BAHN, nicht fest: die nadel misst gegen das
        # kreisbahn- und das fluchttempo AN DIESEM ORT (siehe
        # Telemetry.orbital_speed_scale). Vorher stand hier ein fester
        # vollausschlag von 2600 m/s, der zu keinem koerper gehoerte -- im
        # Erdorbit klebte die nadel am anschlag, um einen kleinen mond
        # schlug sie gar nicht aus.
        span = self.telemetry.velocity_fraction()
        if span is None:
            # Kein bezugskoerper: die richtung stimmt trotzdem, also wird
            # die nadel mit halber laenge und ohne massstab gezeichnet.
            span = 0.5
        length = (_NEEDLE_MIN + span * (_NEEDLE_MAX - _NEEDLE_MIN)) * scale

        # Die bezugsmarke fuer die KREISBAHN. Ohne sie ist eine skalierte
        # nadel nur ein zappelnder strich -- mit ihr liest man auf einen
        # blick ab, ob die bahn gerade unter, auf oder ueber der kreisbahn
        # liegt, und das ist beim zirkularisieren genau die frage.
        circular = self.telemetry.circular_speed_fraction()
        if circular is not None:
            mark = (_NEEDLE_MIN + circular * (_NEEDLE_MAX - _NEEDLE_MIN)) * scale
            ctx.draw.ring(cx, cy, mark, max(1.0, 1.0 * scale),
                          with_alpha(palette.velocity, 0.22))

        x, y = _polar(cx, cy, length, compass - heading)
        ctx.draw.line(cx, cy, x, y, palette.velocity,
                      width=max(1.5, 3.0 * scale), cap='round')
        ctx.draw.circle(x, y, 5.0 * scale, fill=palette.velocity)

    def _draw_readout(self, ctx, cx, cy, scale):
        """Nur noch die kurs-plakette, mittig als nabe des rings.

        Der GESCHWINDIGKEITSWERT sass hier urspruenglich (so wie im entwurf),
        wurde aber von der geschwindigkeitsnadel ueberschrieben: die nadel
        laeuft aus der mitte heraus und legt sich damit je nach kurs quer
        ueber die dreissig pixel grosse zahl. Im entwurf faellt das nicht auf,
        weil dort nur ein einziger kurs abgebildet ist. Der wert steht jetzt
        in VelocityReadout unterhalb des rings.

        Die plakette wird NACH der nadel gezeichnet und deckt deren innerstes
        stueck ab -- das liest sich als nabe, nicht als fehler.
        """
        palette = ctx.theme.palette
        if self.hub_only:
            # Nur die nabe. Der kurs steht im navball-block darueber.
            ctx.draw.circle(cx, cy, 7.0 * scale, fill=palette.ring_face,
                            border_color=with_alpha(palette.ring, 0.7),
                            border_width=max(1.0, 1.4 * scale))
            ctx.draw.circle(cx, cy, 2.2 * scale, fill=palette.ring)
            return
        pill_w = 92.0 * scale
        pill_h = 22.0 * scale
        pill_x = cx - pill_w * 0.5
        pill_y = cy - pill_h * 0.5
        ctx.draw.rect(
            pill_x, pill_y, pill_w, pill_h,
            fill=palette.panel_popup, radius=-6.0 * scale,
            border_color=with_alpha(palette.frame, 0.5),
            border_width=ctx.theme.border_width,
        )
        ctx.text.draw(
            f"HDG {self.telemetry.text_heading()}", cx, cy,
            role='hdg', color=palette.frame, align='center', valign='middle',
        )

    def _draw_nose(self, ctx, cx, cy, scale, outer):
        """Die schiffsnase steht FEST oben -- der ring dreht sich darunter.

        Das ist die konvention jedes lagemessers: der bezugsrahmen bewegt
        sich, das eigene fahrzeug bleibt in der mitte des blickfelds.
        """
        palette = ctx.theme.palette
        tip_y = cy - outer - 2.0 * scale
        half = 6.0 * scale
        height = 11.0 * scale
        # Als schmales dreieck aus zwei linien mit spitzem zulauf: das
        # SDF-rechteck kann keine dreiecke, zwei linien zur spitze schon.
        ctx.draw.line(cx - half, tip_y + height, cx, tip_y, palette.ship,
                      width=max(1.5, 2.4 * scale), cap='round')
        ctx.draw.line(cx + half, tip_y + height, cx, tip_y, palette.ship,
                      width=max(1.5, 2.4 * scale), cap='round')
        ctx.draw.line(cx - half, tip_y + height, cx + half, tip_y + height,
                      palette.ship, width=max(1.5, 2.4 * scale), cap='round')


class VelocityReadout(Widget):
    """Die geschwindigkeitsanzeige als eigene plakette unter dem ring.

    Im entwurf steht der wert in der ringmitte. Das geht dort auf, weil nur
    EIN kurs abgebildet ist -- im spiel wandert die geschwindigkeitsnadel
    ueber den vollen kreis und schneidet dabei zwangslaeufig durch die
    dreissig pixel hohe zahl. Ausgelagert bleibt beides jederzeit lesbar,
    und der ring behaelt seine nabe (die kurs-plakette).

    Die zahlenspalte ist auf eine FESTE breite reserviert. Ohne das wuerde
    die plakette bei jedem stellenwechsel (999 -> 1 000 m/s) ihre breite
    aendern, und weil sie mittig verankert ist, zuckte sie dabei seitlich.
    """

    #: Reservierte zahlenbreite -- die breiteste realistisch auftretende
    #: zeichenfolge. Die mono-rolle haelt alle ziffern gleich breit.
    _WIDTH_SAMPLE = '000 000'

    def __init__(self, telemetry, size=(None, 44), **kwargs):
        super().__init__(size=size, **kwargs)
        self.telemetry = telemetry

    def _parts(self, ctx):
        caption = f"{self.telemetry.view_mode_label()} VELOCITY"
        return caption, self.telemetry.text_speed(), self.telemetry.text_speed_unit()

    def measure(self, ctx):
        caption, _speed, unit = self._parts(ctx)
        pad = ctx.px(18)
        gap = ctx.px(12)
        width = (pad * 2.0 + gap * 2.0
                 + ctx.text.measure(caption, 'ring_caption')[0]
                 + ctx.text.measure(self._WIDTH_SAMPLE, 'readout')[0]
                 + ctx.text.measure(unit, 'ring_unit')[0])
        return (width, ctx.px(44))

    def draw(self, ctx):
        palette = ctx.theme.palette
        caption, speed, unit = self._parts(ctx)
        height = self.rect.h
        middle = self.rect.center_y
        pad = ctx.px(18)
        gap = ctx.px(12)

        ctx.draw.rect(
            self.rect.x, self.rect.y, self.rect.w, height,
            fill=palette.panel_pill, radius=height * 0.5,
            border_color=palette.edge, border_width=ctx.theme.border_width,
            shadow=ctx.theme.glow('velocity'),
            shadow_offset=(0.0, 0.0), shadow_softness=ctx.px(24.0),
        )

        cursor = self.rect.x + pad
        ctx.text.draw(caption, cursor, middle, role='ring_caption',
                      color=palette.text_dim, valign='middle')
        cursor += ctx.text.measure(caption, 'ring_caption')[0] + gap

        # Rechtsbuendig in die reservierte spalte: die zahl waechst damit
        # nach links und der einheiten-text bleibt stehen.
        column = ctx.text.measure(self._WIDTH_SAMPLE, 'readout')[0]
        ctx.text.draw(speed, cursor + column, middle, role='readout',
                      color=palette.velocity, align='right', valign='middle')
        cursor += column + gap

        ctx.text.draw(unit, cursor, middle, role='ring_unit',
                      color=palette.text_muted, valign='middle')
