"""Der navball-block: EIN instrument statt sechs schwebender kaesten.

Nachgebaut nach der instrumententafel von Kerbal Space Program 2. Deren
eigentliche leistung ist nicht die kugel, sondern die VERDICHTUNG: kurs,
geschwindigkeit, hoehe, schub, steigrate und die beiden apsiden sitzen in
einem einzigen, zusammenhaengenden block von der groesse einer handflaeche.
Vorher lagen dieselben werte hier in fuenf einzelnen panels ueber den
ganzen bildschirm verteilt -- und die mitte, wo die bahn gezeichnet wird,
war zugebaut.

    ┌──────────┐        ╭─ 357° ─╮        ┌──────────┐
    │ ORB      │      ╱            ╲      │      ALT │
    │ 11.69    │  ╭─ teilungsring ─╮  │   │    5.36  │
    │      km/s│  │   ●  kompass   │  │   │       Gm │
    └──────────┘  ╰────────────────╯  │   └──────────┘
       schub ◜────────────────────────◝ steigrate
              ┌───────────────────────┐
              │ AP  976.4 km  T-00:21 │
              │ PE  708.9 km  T-01:09 │
              └────[ ORBITAL.INFO ]───┘

AUFTEILUNG DER BEIDEN FLANKEN. Links steht, was man SETZT (schub), rechts,
was daraus FOLGT (steigrate). Beide bogen sind zellen, keine balken -- ein
zellenbogen ist abzaehlbar, und die vorlage benutzt fuer beides dieselbe
form.

Was bewusst FEHLT: eine schattierte kugel mit horizont. Die simulation
kennt genau einen orientierungswinkel; nick und roll gibt es hier nicht.
Eine kugel wuerde also achsen anzeigen, die in dieser physik nicht
existieren -- in einer Seminararbeit waeren das erfundene daten. Der
kompassring zeigt nur, was wirklich da ist. Ebenso fehlen tank, RCS und
SAS-modi: nichts davon hat eine entsprechung in der simulation.

KOORDINATEN top-down, WINKEL in kompassgrad (0 = oben, im uhrzeigersinn).
"""

import math

from .. import units
from ..core import Widget
from ..theme import with_alpha
from . import chrome
from .attitude import AttitudeRing

# Alle masse in DESIGN-EINHEITEN. Der block ist als ganzes verankert, die
# einzelteile sitzen relativ zu seiner mitte -- deshalb steht hier eine
# geschlossene rechnung und keine liste von absolutwerten.
RING_SIZE = 196.0          # kantenlaenge des kompassrings
RING_OUTER = 92.0          # dessen aeusserer radius (103/220 * RING_SIZE)
GAUGE_RADIUS = 109.0       # radius der beiden zellenbogen
GAUGE_WIDTH = 9.0
GAUGE_SPAN = 104.0         # ueberstrichener winkel je bogen
BOX_W = 124.0
BOX_H = 58.0
BADGE_W = 78.0
BADGE_H = 26.0
INFO_H = 58.0
FLANK_GAP = 6.0
STRIP_H = 20.0     # der zahlenstreifen unter jeder flanke

WIDTH = 2.0 * (GAUGE_RADIUS + GAUGE_WIDTH * 0.5 + FLANK_GAP + BOX_W)
HEIGHT = BADGE_H + 4.0 + 2.0 * RING_OUTER + 10.0 + INFO_H

#: Der schubbogen laeuft ueber die LINKE flanke und fuellt von UNTEN nach
#: oben -- aufwaerts heisst mehr, wie an jedem schubhebel. In kompassgrad
#: ist unten-links 218 und oben-links 322, der bogen beginnt also bei 218.
THROTTLE_ARC = (270.0 - GAUGE_SPAN * 0.5, 270.0 + GAUGE_SPAN * 0.5)
#: Der steigraten-bogen ueber die RECHTE. Er fuellt von der MITTE aus, und
#: die mitte liegt auf 90 grad, also genau waagerecht -- steigen fuellt nach
#: oben, sinken nach unten.
RADIAL_ARC = (90.0 + GAUGE_SPAN * 0.5, 90.0 - GAUGE_SPAN * 0.5)


class NavballCluster(Widget):
    """Kompassring, beide flanken und die bahndaten als EIN block.

    Der ring ist ein echtes kind-widget: er nimmt das ziehen entgegen und
    bringt seine eigene, muehsam erarbeitete eingabelogik mit. Alles uebrige
    zeichnet dieser block selbst -- als kinder waeren es wieder einzelne
    kaesten, und genau das soll die verdichtung ja abschaffen.
    """

    def __init__(self, telemetry, ship_control=None, **kwargs):
        kwargs.setdefault('size', (WIDTH, HEIGHT))
        super().__init__(**kwargs)
        self.telemetry = telemetry
        # Der block als ganzes faengt die maus ab: sonst schwenkt ein klick
        # zwischen ring und flanke die kamera darunter.
        self.blocks_mouse = True
        self._dragging_throttle = False
        self.ring = self.add(AttitudeRing(
            telemetry, ship_control, size=(RING_SIZE, RING_SIZE), hub_only=True,
        ))

    # -------------------------------------------------------------- geometrie

    def _center(self, ctx):
        """Mittelpunkt des kompassrings."""
        return (self.rect.center_x,
                self.rect.y + ctx.px(BADGE_H + 4.0 + RING_OUTER))

    def layout_children(self, ctx):
        cx, cy = self._center(ctx)
        size = ctx.px(RING_SIZE)
        self.ring.layout(ctx, self.rect)
        # Der ring wird nach der normalen verankerung UEBERSCHRIEBEN: seine
        # lage folgt aus der blockrechnung oben, nicht aus einem anker.
        self.ring.rect.x = cx - size * 0.5
        self.ring.rect.y = cy - size * 0.5
        self.ring.rect.w = size
        self.ring.rect.h = size

    # --------------------------------------------------------------- zeichnen

    def draw(self, ctx):
        cx, cy = self._center(ctx)
        self._draw_gauges(ctx, cx, cy)
        self._draw_flank(ctx, cx, cy, side='left')
        self._draw_flank(ctx, cx, cy, side='right')
        self._draw_gauge_labels(ctx)
        self._draw_heading_badge(ctx, cx)
        self._draw_info(ctx)

    # ------------------------------------------------------------- eingabe

    def _throttle_from_point(self, ctx, x, y):
        """Punkt auf dem linken bogen -> schubstufe, oder None.

        Getroffen wird ein RING-AUSSCHNITT, nicht das umschliessende
        quadrat: sonst faengt der bogen klicks, die eigentlich der kugel
        oder der flanke gelten.
        """
        cx, cy = self._center(ctx)
        dx = float(x) - cx
        dy = float(y) - cy
        distance = math.hypot(dx, dy)
        band = ctx.px(GAUGE_WIDTH) * 1.6
        if not (ctx.px(GAUGE_RADIUS) - band <= distance
                <= ctx.px(GAUGE_RADIUS) + band):
            return None
        compass = math.degrees(math.atan2(dx, -dy)) % 360.0
        low, high = THROTTLE_ARC
        if not (low <= compass <= high):
            return None
        return (compass - low) / max(high - low, 1e-6)

    def _throttle_strip_hit(self, ctx, x, y):
        sx, sy, sw, sh = self._strip_rect(ctx, 'left')
        return sx <= x < sx + sw and sy <= y < sy + sh

    def on_mouse_down(self, ctx, x, y, button):
        if button != 1:
            return True
        level = self._throttle_from_point(ctx, x, y)
        if level is None and self._throttle_strip_hit(ctx, x, y):
            sx, _sy, sw, _sh = self._strip_rect(ctx, 'left')
            level = (float(x) - sx) / max(sw, 1e-6)
        if level is not None:
            self.telemetry.set_thrust_level(level)
            self._dragging_throttle = True
        return True

    def on_mouse_move(self, ctx, x, y):
        if not (self.pressed and getattr(self, '_dragging_throttle', False)):
            return False
        level = self._throttle_from_point(ctx, x, y)
        if level is None:
            sx, _sy, sw, _sh = self._strip_rect(ctx, 'left')
            level = (float(x) - sx) / max(sw, 1e-6)
        self.telemetry.set_thrust_level(level)
        return True

    def on_mouse_up(self, ctx, x, y, button):
        self._dragging_throttle = False
        return True

    def on_wheel(self, ctx, dx, dy):
        """Das rad stellt den schub -- aber NUR ueber dem bogen und dem
        streifen. Ueberall sonst muss es zur kamera durchfallen, sonst
        laesst sich mit dem zeiger auf dem instrument nicht mehr zoomen."""
        if not dy:
            return False
        over_arc = self._throttle_from_point(ctx, ctx.mouse_x, ctx.mouse_y)
        if over_arc is None and not self._throttle_strip_hit(
                ctx, ctx.mouse_x, ctx.mouse_y):
            return False
        self.telemetry.set_thrust_level(
            self.telemetry.thrust_level + 0.05 * float(dy)
        )
        return True

    # ------------------------------------------------------------- die bogen

    def _draw_gauges(self, ctx, cx, cy):
        palette = ctx.theme.palette
        telemetry = self.telemetry
        radius = ctx.px(GAUGE_RADIUS)
        width = ctx.px(GAUGE_WIDTH)

        # LINKS: schub. Gesperrt (im zeitraffer) wird der bogen blass und
        # behaelt trotzdem seinen fuellstand -- die eingestellte stufe gilt
        # ja weiter, sie ist nur gerade nicht abrufbar.
        locked = bool(getattr(telemetry, 'thrust_locked', False))
        throttle_color = (palette.text_dimmer if locked else palette.throttle)
        chrome.segment_arc(
            ctx, cx, cy, radius, width, THROTTLE_ARC[0], THROTTLE_ARC[1],
            telemetry.thrust_level, throttle_color, count=14,
        )
        # RECHTS: steigrate, null in der bogenmitte.
        chrome.segment_arc(
            ctx, cx, cy, radius, width, RADIAL_ARC[0], RADIAL_ARC[1],
            telemetry.radial_fraction(), palette.ring, count=14,
            bidirectional=True,
        )
        # Die nullmarke des rechten bogens -- ohne sie ist eine beidseitige
        # anzeige nicht ablesbar.
        self._tick(ctx, cx, cy, radius, width, 90.0, palette.text_muted)

        # Aussenteilung an beiden bogen, wie in der vorlage.
        outer = radius + width * 0.5 + ctx.px(2.0)
        chrome.arc_ruler(ctx, cx, cy, outer, with_alpha(palette.throttle, 0.55),
                         THROTTLE_ARC[0], THROTTLE_ARC[1], count=15,
                         major_every=7, major=7.0, minor=3.5, inward=False)
        chrome.arc_ruler(ctx, cx, cy, outer, with_alpha(palette.ring, 0.55),
                         RADIAL_ARC[0], RADIAL_ARC[1], count=15,
                         major_every=7, major=7.0, minor=3.5, inward=False)

    def _tick(self, ctx, cx, cy, radius, width, compass, color):
        x0, y0 = chrome.polar(cx, cy, radius - width * 0.5 - ctx.px(2.0), compass)
        x1, y1 = chrome.polar(cx, cy, radius + width * 0.5 + ctx.px(2.0), compass)
        ctx.draw.line(x0, y0, x1, y1, color, width=max(1.0, ctx.px(1.4)))

    # ---------------------------------------------------------- die flanken

    def _flank_rect(self, ctx, side):
        width = ctx.px(BOX_W)
        height = ctx.px(BOX_H)
        cx, cy = self._center(ctx)
        if side == 'left':
            x = self.rect.x
        else:
            x = self.rect.right - width
        return (x, cy - height * 0.5, width, height)

    def _draw_flank(self, ctx, cx, cy, side):
        """Eine der beiden messwert-flanken.

        Aufbau der vorlage: winzige gesperrte beschriftung oben, darunter
        die zahl in 25 px, die einheit klein daneben. Der groessensprung --
        nicht die farbe -- macht die hierarchie.

        Die zur kugel zeigende ecke bleibt SCHARF: der block sieht damit
        angesetzt aus statt danebengestellt.
        """
        palette = ctx.theme.palette
        telemetry = self.telemetry
        x, y, w, h = self._flank_rect(ctx, side)

        if side == 'left':
            caption = telemetry.view_mode_label()[:3]
            value, unit = telemetry.gauge_speed()
            color = palette.velocity
            role_color = palette.velocity
            # Rechte ecken scharf (dort steht die kugel).
            corners = (True, False, False, True)
        else:
            caption = 'ALT'
            value, unit = telemetry.gauge_altitude()
            color = palette.altitude
            role_color = palette.altitude
            corners = (False, True, True, False)

        ix, iy, iw, ih = chrome.frame(ctx, x, y, w, h, corners=corners)
        pad = ctx.px(9)

        # KOPFZEILE: rolle links, einheit rechts. Die einheit steht bewusst
        # NICHT neben der zahl -- ein 25-px-wert und eine 10-px-einheit auf
        # einer zeile lassen in 124 einheiten breite keinen platz, und ein
        # kollidierender einheiten-text ist schlimmer als gar keiner.
        head = iy + ctx.px(11)
        ctx.text.draw(caption, ix + pad, head, role='caption',
                      color=with_alpha(role_color, 0.9), valign='middle')
        ctx.text.draw(unit, ix + iw - pad, head, role='unit',
                      color=palette.text_dim, align='right', valign='middle')
        # Eine feine linie darunter trennt beschriftung von messwert -- die
        # vorlage setzt an derselben stelle eine.
        ctx.draw.rect(ix + pad, iy + ctx.px(19), iw - pad * 2.0,
                      max(1.0, ctx.theme.border_width),
                      fill=palette.edge_inner)
        # Der WERT sitzt linksbuendig in beiden flanken. Rechtsbuendig auf
        # der linken seite haette gespiegelt gewirkt, aber die zahl waechst
        # dann in richtung kugel und stoesst dort an.
        ctx.text.draw(value, ix + pad, iy + ih - ctx.px(19), role='gauge',
                      color=color, valign='middle')

    def _strip_rect(self, ctx, side):
        """Der schmale streifen unter einer flanke."""
        x, y, w, h = self._flank_rect(ctx, side)
        return (x, y + h + ctx.px(6.0), w, ctx.px(STRIP_H))

    def _draw_gauge_labels(self, ctx):
        """Was die beiden bogen zeigen, noch einmal als zahl.

        Ein zellenbogen sagt die GROESSENORDNUNG auf einen blick, aber
        "62 %" liest man daran nicht ab -- und beim schub ist der genaue
        wert die eigentliche einstellung. Wichtiger noch: im zeitraffer ist
        der schub gesperrt, der bogen wird blass und saehe ohne diesen
        streifen einfach nur kaputt aus. HOLD sagt, dass es absicht ist.
        """
        palette = ctx.theme.palette
        telemetry = self.telemetry
        locked = bool(getattr(telemetry, 'thrust_locked', False))

        self._strip(ctx, 'left', 'THR',
                    'HOLD' if locked else telemetry.text_throttle(),
                    palette.text_dimmer if locked else palette.throttle,
                    corners=(True, False, False, True))

        # Mit VORZEICHEN: das ist die ganze aussage der groesse. Ein '+'
        # vorweg, weil ein blosses '38' nicht sagt, ob es steigt oder faellt.
        #
        # EINE nachkommastelle, nicht zwei: '+14.41km/s' ist bei 15 px
        # genau so breit wie der ganze streifen und schob sich damit ueber
        # die beschriftung. Die zweite stelle sagt an einer anzeige, die
        # daneben ohnehin einen bogen hat, auch nichts.
        radial = telemetry.radial_speed
        if radial is None:
            text = '--'
        else:
            value, unit = units.split_speed(abs(radial), digits=1)
            text = f"{'+' if radial >= 0.0 else '-'}{value}{unit}"
        self._strip(ctx, 'right', 'V/S', text, palette.ring,
                    corners=(False, True, True, False))

    def _strip(self, ctx, side, caption, text, color, corners):
        """Ein zahlenstreifen -- beschriftung links, wert rechts.

        DREI STUFEN, weil der inhalt von der groessenordnung des messwerts
        abhaengt: grosser wert mit beschriftung -> kleiner wert mit
        beschriftung -> kleiner wert ALLEIN. Der wert gewinnt immer; die
        beschriftung ist die zugabe.

        Der anlass war '+14.41km/s': bei 15 px belegte das die vollen 100 px
        des streifens, und das 'V/S' verschwand darunter. Die kleine stufe
        allein reicht dafuer nicht -- die pixelschrift rastet ihre groesse
        auf fuenferschritte, die "kleine" rolle ist also nicht ueberall
        proportional kleiner. Bei 250 km/s laeuft auch sie noch ueber, und
        genau dann faellt die beschriftung weg.
        """
        palette = ctx.theme.palette
        sx, sy, sw, sh = self._strip_rect(ctx, side)
        chrome.plate(ctx, sx, sy, sw, sh, fill=palette.panel_sunken,
                     corners=corners)
        pad = ctx.px(9)
        gap = ctx.px(6)
        middle = sy + sh * 0.5
        available = sw - pad * 2.0
        caption_w = ctx.text.measure(caption, 'caption')[0]

        role = None
        for candidate in ('throttle_value', 'caption'):
            if caption_w + gap + ctx.text.measure(text, candidate)[0] <= available:
                role = candidate
                break
        if role is not None:
            ctx.text.draw(caption, sx + pad, middle, role='caption',
                          color=palette.text_dim, valign='middle')
        else:
            role = 'caption'

        ctx.text.draw(text, sx + sw - pad, middle, role=role, color=color,
                      align='right', valign='middle')

    # ----------------------------------------------------------- kurs-plakette

    def _draw_heading_badge(self, ctx, cx):
        """Der kurs ueber der kugel -- die auffaelligste einzelzahl der vorlage.

        Er steht bewusst NICHT mehr in der ringmitte: dort lief die
        geschwindigkeitsnadel quer durch die ziffern.
        """
        palette = ctx.theme.palette
        width = ctx.px(BADGE_W)
        height = ctx.px(BADGE_H)
        x = cx - width * 0.5
        y = self.rect.y
        chrome.plate(ctx, x, y, width, height, fill=palette.panel_popup,
                     line=palette.edge_strong)
        ctx.text.draw(self.telemetry.text_heading(), cx, y + height * 0.5,
                      role='heading_big', color=palette.text,
                      align='center', valign='middle')

    # ------------------------------------------------------------ bahndaten

    def _info_rect(self, ctx):
        """Der rahmen ohne das tab-band, das unter ihm sitzt."""
        inset = ctx.px(BOX_W * 0.5)
        tab_h = ctx.text.measure('X', 'tab')[1] + ctx.px(4)
        return (self.rect.x + inset,
                self.rect.bottom - ctx.px(INFO_H),
                self.rect.w - inset * 2.0, ctx.px(INFO_H) - tab_h)

    def _draw_info(self, ctx):
        """AP und PE mit ihren restzeiten -- der ORBITAL.INFO-block.

        Das ersetzt das frueher eigenstaendige seitenpanel. Die beiden
        apsiden gehoeren neben das instrument, nicht an den bildschirmrand:
        man liest sie waehrend eines brennmanoevers, also genau dann, wenn
        der blick auf dem kurs liegt.
        """
        palette = ctx.theme.palette
        telemetry = self.telemetry
        x, y, w, h = self._info_rect(ctx)
        ix, iy, iw, ih = chrome.frame(ctx, x, y, w, h, glow_role='elem')

        pad = ctx.px(12)
        rows = (
            ('AP', telemetry.text_apoapsis(), telemetry.text_countdown_to_apoapsis()),
            ('PE', telemetry.text_periapsis(), telemetry.text_time_to_periapsis()),
        )
        row_h = ih / 2.0
        for index, (key, value, countdown) in enumerate(rows):
            middle = iy + row_h * (index + 0.5)
            ctx.text.draw(key, ix + pad, middle, role='caption',
                          color=palette.elem, valign='middle')
            ctx.text.draw(value, ix + pad + ctx.px(34), middle, role='value',
                          color=palette.text, valign='middle')
            ctx.text.draw('T-', ix + iw - pad - ctx.px(62), middle,
                          role='caption', color=palette.text_dimmer,
                          valign='middle')
            ctx.text.draw(countdown, ix + iw - pad, middle, role='value',
                          color=palette.text_muted, align='right',
                          valign='middle')

        chrome.tab(ctx, chrome.tab_text('ORBITAL', 'INFO'),
                   x + w * 0.5, y + h, color=palette.elem,
                   align='center', edge='bottom')
