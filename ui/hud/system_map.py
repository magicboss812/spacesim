"""Die system-karte -- der ueberblick ueber das ganze system in einer kachel.

Sie sitzt an der rechten flanke unter dem zeitraffer, und das ist kein
zufaelliger platz: der zeitraffer sagt, WIE SCHNELL die zeit laeuft, die
karte zeigt, WAS sich dabei bewegt. Beide zusammen sind das
system-instrument, waehrend die untere bildmitte das flug-instrument bleibt.

WAS SIE ZEIGT UND WAS NICHT

Der ueberblick zeigt AUSSCHLIESSLICH die oberste hierarchie-ebene: das
zentralgestirn in der mitte, die planeten darum. Alle 19 monde des systems
gleichzeitig einzuzeichnen ergaebe bei dieser groesse einen fleck -- Mimas
laege 24 bildschirm-tausendstel neben Saturn. Monde erscheinen deshalb
erst, wenn man ihren planeten anwaehlt, und der rest der karte tritt
dafuer zurueck.

DIE BAHNEN SIND KREISE, DIE POSITIONEN NICHT ERFUNDEN

Gezeichnet werden perfekte kreise -- die echten exzentrizitaeten liegen
zwischen 0.007 (Venus) und 0.25 (Pluto) und waeren bei einem kartenradius
von gut hundert pixeln ohnehin nicht zu sehen; ein leicht ovaler ring saehe
nur nach zeichenfehler aus. Der WINKEL auf dem kreis ist dagegen der echte:
er kommt aus der momentanen position des koerpers relativ zu seinem
mutterkoerper. Damit laeuft die karte automatisch in derselben zeit wie das
spiel, folgt jeder zeitraffer-stufe und braucht keine eigene uhr, keinen
eigenen integrator und keinen abgleich -- es gibt schlicht nichts, was
auseinanderlaufen koennte.

Die bahnradien sind UEBERWIEGEND LOGARITHMISCH verteilt. Merkur und Pluto
liegen um den faktor 102 auseinander; linear aufgetragen laegen die vier
inneren planeten alle auf demselben pixel. Ein anteil gleichabstaendiger
verteilung (`_RANK_MIX`) zieht ausserdem auseinander, was auch logarithmisch
noch zusammenfaellt -- Neptun und Pluto trennen sonst vier hundertstel des
kartenradius. Die karte ist eine uebersicht, kein massstab.

VOLLE RINGE, KEINE SPUREN

Ein ring ist EINE instanz des SDF-shaders und damit exakt rund -- kein
polygonzug, also auch keine ecken, egal wie gross die karte gezogen wird.
Er steht ausserdem still: eine schleppe hinter dem planeten waere eine
zweite, widerspruechliche zeitanzeige neben dem planeten selbst.

ZWEI GROESSEN, EINE GESTE

Zugeklappt ist die karte eine kachel und nichts weiter -- ein klick darauf
faehrt sie auf ihre volle groesse aus, und ERST DANN nehmen koerper-treffer,
hover-namen und die mond-ansicht ihre arbeit auf. Ein klick daneben faehrt
sie wieder ein (ueber `dismiss()`, denselben weg, den auch die koerperliste
benutzt). Ohne diese trennung faenge eine dauerhaft grosse karte am rand
staendig klicks ab, die dem schwenk gehoeren.

Beim ausfahren wachsen KOERPER UND RAHMEN VERSCHIEDEN SCHNELL (`_DOT_COLLAPSED`).
Die grosse karte ist nicht die kachel unter der lupe, sondern die einzige
ansicht, in der ein punkt angeklickt und ein mondsystem gelesen wird -- die
punkte muessen darin eine flaeche haben, und der rahmen soll dafuer nicht
noch weiter ueber den bildrand wandern. Die kachel bleibt umgekehrt so eng
um ihren kreis, wie das notch-tab es zulaesst: ihre breite folgt dem
kartendurchmesser, nicht der breite des zeitraffers darueber.

DIE AUSWAHL IST DIESELBE WIE IM SPIEL

Ein klick auf einen koerper waehlt ihn aus, ein zweiter fliegt ihn an --
wortgleich zu `test.py::handle_world_click`. Die karte ist damit ein
zweiter weg zu derselben handlung, kein eigener zustand: was hier
angeklickt wird, traegt in der welt sofort die vier auswahl-pfeile.
"""

import math

from ..core import Rect, Widget, ease
from ..theme import with_alpha
from . import chrome

# ------------------------------------------------------- entwurfseinheiten

#: Zugeklappt und ausgefahren. Die zugeklappte breite ist die des
#: zeitraffers darueber nicht ganz, aber ihre eigene: die kachel soll als
#: eigenes bauteil lesbar bleiben und nicht als dessen fortsetzung.
#:
#: BEIDE MASSE FOLGEN DEM KREIS, nicht umgekehrt. Der plotradius ist
#: ``min(breite, hoehe)/2 - _PAD`` -- jedes pixel, um das die kachel breiter
#: als hoch ist, ist deshalb leerer rand und sonst nichts. Bei 196x150 waren
#: das 46 px auf jeder flanke, mehr als der halbe kartenradius; die kachel
#: las sich als grosses bauteil mit einer kleinen zeichnung darin. Die
#: breite liegt jetzt nur noch so weit ueber der rahmenhoehe, wie das
#: notch-tab ueber der kante braucht.
_COLLAPSED = (134.0, 138.0)
_EXPANDED = (292.0, 300.0)

#: Abstand zum widget darueber (dem zeitraffer).
_GAP_ABOVE = 10.0
#: Innenabstand zwischen rahmen und der aeussersten bahn.
_PAD = 10.0

#: Kleinster und groesster bahnradius als anteil des verfuegbaren radius.
_ORBIT_INNER = 0.30
_ORBIT_OUTER = 1.0

#: Punktgroessen der koerper, GEMESSEN AN DER AUSGEFAHRENEN KARTE. Bewusst
#: schmal gestuft -- der groessenunterschied Sonne/Merkur ist 285-fach, als
#: flaeche aufgetragen bliebe von Merkur ein halbes pixel. Die punkte
#: ordnen, sie messen nicht.
_DOT_MIN = 2.9
_DOT_MAX = 7.0
_STAR_DOT = 11.0
_MOON_DOT = 3.4

#: Punktgroesse der ZUGEKLAPPTEN kachel als anteil davon. Die punkte wachsen
#: also mit dem ausfahren mit, der rahmen aber nicht mehr im selben mass --
#: genau darum geht es: die ausgefahrene karte soll nicht die kachel unter
#: der lupe sein, sondern dieselbe flaeche mit LESBAREN koerpern. Bei 0.75
#: bleibt die kachel bei ihren gewohnten gut zwei bis fuenf pixeln.
_DOT_COLLAPSED = 0.75

#: Trefferkreis um einen punkt. Deutlich groesser als der punkt selbst, sonst
#: ist ein 3-px-mond mit der maus nicht zu fassen.
_HIT_RADIUS = 11.0

#: Deckkraft der zurueckgetretenen karte, wenn ein planet angewaehlt ist.
_DIMMED = 0.22

#: Mond-bahnen: anteil des abstands zur naechst inneren planetenbahn, dazu
#: ein boden und eine decke als anteil des kartenradius. Der abstand allein
#: genuegt nicht -- zwischen Mars und Jupiter ist er gross, zwischen Neptun
#: und Pluto fast null, und ein mondsystem von drei pixeln waere unlesbar.
#:
#: Boden und decke sind bewusst gross: die mond-ansicht ist der einzige
#: zweck, fuer den die karte ueberhaupt ausfaehrt, und bei 0.15/0.26 des
#: kartenradius war das Jupiter-system 21 bis 37 pixel breit -- vier ringe
#: und vier punkte darin sind auf einem laptop nicht mehr auseinander zu
#: halten. Der ganze rest der karte steht waehrenddessen ohnehin auf
#: `_DIMMED`, das mondsystem darf also ueber die nachbarbahnen laufen.
_MOON_SPAN = 0.9
_MOON_SPAN_MIN = 0.24
_MOON_SPAN_MAX = 0.42
_MOON_INNER = 0.38

#: Wieviel der bahnverteilung aus der REIHENFOLGE statt aus dem logarithmus
#: kommt. Rein logarithmisch liegen Neptun (4.5e12 m) und Pluto (5.9e12 m)
#: vier hundertstel des kartenradius auseinander -- zwei ringe, die als
#: einer erscheinen. Ein anteil gleichabstaendiger verteilung zieht sie
#: auseinander, ohne dass die karte ihren groessenordnungs-eindruck verliert.
_RANK_MIX = 0.45


def _body_color(body):
    """``body.color`` (0..255 aus dem JSON) -> zeichenfarbe.

    Unveraendert uebernommen, wie in der koerperliste: der punkt ist eine
    FLAECHE, und fuer flaechen gilt die rohfarbe. Er soll denselben koerper
    meinen, den man in der welt sieht.
    """
    raw = getattr(body, 'color', None) or (255, 255, 255)
    try:
        r, g, b = float(raw[0]), float(raw[1]), float(raw[2])
    except Exception:
        r = g = b = 255.0
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)


def _log_spread(values, low, high, rank_mix=0.0):
    """Werte auf [low, high] verteilen -- logarithmisch, optional nach rang.

    `rank_mix` mischt eine GLEICHABSTAENDIGE verteilung nach der reihenfolge
    unter die logarithmische. Rein logarithmisch bildet die karte die echten
    groessenverhaeltnisse ab, draengt aber alles zusammen, was dicht
    beieinander liegt; rein nach rang ist jeder ring gleich weit vom
    nachbarn entfernt, sagt dafuer nichts mehr ueber abstaende. Die mischung
    ist der lesbare kompromiss, und sie steht hier als zahl, damit man ihn
    verstellen kann.

    Ein einzelner wert (oder lauter gleiche) landet in der MITTE statt am
    rand -- am rand saehe eine einplanetige karte aus, als fehlte etwas.
    """
    logs = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        logs.append(math.log(value) if value > 0.0 else None)
    known = [v for v in logs if v is not None]
    count = len(logs)
    if not known or count == 0:
        return [(low + high) * 0.5] * count
    lo, hi = min(known), max(known)
    span = hi - lo
    # Der rang wird ueber die SORTIERUNG der bekannten werte vergeben, nicht
    # ueber die listenposition: die aufrufer sortieren zwar bereits nach
    # bahnradius, aber ein koerper ohne radius darf die reihe nicht
    # verschieben.
    order = sorted(range(count), key=lambda i: (logs[i] is None, logs[i] or 0.0))
    rank_of = {index: slot for slot, index in enumerate(order)}
    mix = max(0.0, min(1.0, float(rank_mix)))
    out = []
    for index, value in enumerate(logs):
        if value is None:
            out.append((low + high) * 0.5)
            continue
        by_log = (value - lo) / span if span > 1e-9 else 0.5
        by_rank = rank_of[index] / (count - 1) if count > 1 else 0.5
        out.append(low + (high - low) * (by_log * (1.0 - mix) + by_rank * mix))
    return out


def _true_angle(body, parent):
    """Der ECHTE bahnwinkel des koerpers um seinen mutterkoerper.

    Aus den momentanen positionen, nicht aus ``body.theta``: theta ist die
    wahre anomalie ab dem periapsis und haette gegenueber der gezeichneten
    lage noch die drehung ``arg_periapsis`` offen. Die differenz der
    positionen hat sie bereits drin und gilt ausserdem fuer freigelassene
    koerper weiter, die gar keine kepler-elemente mehr fortschreiben.

    Vorzeichen wie in der welt: der renderer zeichnet top-down (line.vert
    kippt y), ein positives welt-y erscheint also UNTEN. Die karte
    uebernimmt das, damit dieselbe konstellation hier und im bild gleich
    herum steht.
    """
    try:
        dx = float(body.position.x) - float(parent.position.x)
        dy = float(body.position.y) - float(parent.position.y)
    except Exception:
        return 0.0
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return math.atan2(dy, dx)


class SystemMap(Widget):
    """Die karte als widget: kachel, ausfahren, treffer, mond-ansicht."""

    def __init__(self, telemetry, ui_state, camera, below=None, **kwargs):
        kwargs.setdefault('size', (None, None))
        # Ueber den panels, unter den aufklapp-listen (die liegen bei 150):
        # ausgefahren ueberlappt die karte den rand, aber eine geoeffnete
        # koerperliste muss weiterhin darueber liegen.
        kwargs.setdefault('z', 40)
        super().__init__(**kwargs)
        self.telemetry = telemetry
        self.ui_state = ui_state
        self.camera = camera
        #: Das widget, unter dem die karte haengt (der zeitraffer). Seine
        #: hoehe steht erst nach dessen layout fest, deshalb ein verweis und
        #: kein fester abstand -- eine andere schriftgroesse aendert sie.
        self.below = below
        self.blocks_mouse = True

        self.expanded = False
        self._expand_t = 0.0
        #: Index des planeten, dessen monde gerade gezeigt werden.
        self._focus_index = None
        self._focus_t = 0.0
        self._hover_index = None
        #: Trefferflaechen des zuletzt gezeichneten frames:
        #: index -> (x, y, radius). Aus dem ZEICHNEN gewonnen, nicht neu
        #: gerechnet -- getroffen wird damit garantiert das, was man sieht.
        self._hits = {}
        self._structure = None
        self._structure_source = None

    # -------------------------------------------------------- systemaufbau

    def structure(self):
        """(zentralkoerper, planeten, monde je planet), einmal aufgebaut.

        Die koerperliste aendert sich zur laufzeit nicht; der aufbau muss
        also nicht pro frame laufen. Verschluesselt ist der cache ueber die
        IDENTITAET der liste, damit ein neu geladenes system ihn verwirft.
        """
        bodies = list(getattr(self.telemetry.world, 'body', None) or [])
        if self._structure is not None and self._structure_source is bodies:
            return self._structure
        source = bodies

        celestial = [(i, b) for i, b in enumerate(bodies)
                     if not getattr(b, 'is_ship', False)]
        roots = [(i, b) for i, b in celestial
                 if getattr(b, 'is_moon_of', None) is None]
        # Schwerster wurzelkoerper ist das zentralgestirn -- dieselbe regel
        # wie in der koerperliste (body_browser.build_hierarchy).
        roots.sort(key=lambda item: -float(getattr(item[1], 'mass', 0.0) or 0.0))
        star = roots[0] if roots else None

        planets = []
        moons = {}
        if star is not None:
            for index, candidate in celestial:
                parent = getattr(candidate, 'is_moon_of', None)
                if parent is star[1]:
                    planets.append((index, candidate))
            planets.sort(key=lambda item: float(
                getattr(item[1], 'semi_major_axis', 0.0) or 0.0))
            for planet_index, planet in planets:
                own = [(i, b) for i, b in celestial
                       if getattr(b, 'is_moon_of', None) is planet]
                own.sort(key=lambda item: float(
                    getattr(item[1], 'semi_major_axis', 0.0) or 0.0))
                moons[planet_index] = own
            # Weitere wurzeln (ein zweites gestirn, ein freier koerper)
            # laufen als planeten mit -- lieber falsch einsortiert als
            # unsichtbar, dieselbe entscheidung wie in build_hierarchy.
            for index, candidate in roots[1:]:
                planets.append((index, candidate))
                moons.setdefault(index, [])

        self._structure = (star, planets, moons)
        self._structure_source = source
        return self._structure

    # ------------------------------------------------------------- geometrie

    def measure(self, ctx):
        """Groesse aus dem ausfahr-fortschritt, nicht aus dem schalter.

        Zwischen kachel und voller karte wird INTERPOLIERT, damit das
        ausfahren eine bewegung ist und kein sprung. Der anker liegt oben
        rechts, das wachstum geht also nach links und unten -- weg vom rand.
        """
        t = self._expand_t
        width = _COLLAPSED[0] + (_EXPANDED[0] - _COLLAPSED[0]) * t
        height = _COLLAPSED[1] + (_EXPANDED[1] - _COLLAPSED[1]) * t
        return (ctx.px(width), ctx.px(height))

    def layout(self, ctx, parent_rect):
        super().layout(ctx, parent_rect)
        below = self.below
        if below is None or not getattr(below, 'visible', False):
            return
        # UNTER dem zeitraffer, gemessen an dessen fertigem rechteck. Ein
        # fester y-abstand waere hier falsch: die hoehe der zeitraffer-leiste
        # folgt ihrer schriftgroesse und ihrem notch-tab, aendert sich also
        # mit der UI-skala.
        top = below.rect.bottom + ctx.px(_GAP_ABOVE)
        self.rect = Rect(self.rect.x, top, self.rect.w, self.rect.h)
        self.layout_children(ctx)

    def _frame_rect(self, ctx):
        """Die flaeche des rahmens, ohne das tab-band darueber."""
        tab_h = ctx.text.measure('X', 'tab')[1] + ctx.px(4)
        return (self.rect.x, self.rect.y + tab_h,
                self.rect.w, self.rect.h - tab_h)

    def _plot(self, ctx):
        """(mittelpunkt_x, mittelpunkt_y, aeusserer radius) der karte."""
        fx, fy, fw, fh = self._frame_rect(ctx)
        pad = ctx.px(_PAD)
        cx = fx + fw * 0.5
        cy = fy + fh * 0.5
        radius = max(ctx.px(8.0), min(fw, fh) * 0.5 - pad)
        return cx, cy, radius

    def _dot_scale(self):
        """Punktgroesse relativ zur ausgefahrenen karte.

        Die punkte wachsen MIT dem ausfahren, und zwar staerker, als der
        rahmen es tut: die grosse karte ist keine vergroesserung der kachel,
        sondern die einzige ansicht, in der koerper angeklickt und monde
        gelesen werden -- dafuer muessen sie eine flaeche haben.
        """
        return _DOT_COLLAPSED + (1.0 - _DOT_COLLAPSED) * self._expand_t

    # --------------------------------------------------------------- eingabe

    def _pick(self, x, y):
        """Index des koerpers unter dem zeiger, oder None."""
        best = None
        best_gap = None
        for index, (px, py, reach) in self._hits.items():
            gap = math.hypot(float(x) - px, float(y) - py)
            if gap > reach:
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best = index
        return best

    def on_mouse_move(self, ctx, x, y):
        # Hover-erkennung erst in der ausgefahrenen karte: in der kachel
        # liegen die planeten so dicht, dass ein name nur im weg staende.
        self._hover_index = self._pick(x, y) if self.expanded else None
        return True

    def on_mouse_up(self, ctx, x, y, button):
        if button != 1:
            return True
        if not self.expanded:
            self.expanded = True
            return True

        index = self._pick(x, y)
        if index is None:
            # Klick auf den leeren teil der karte: zurueck in die uebersicht.
            # Die karte bleibt ausgefahren -- sie zu schliessen ist die
            # aufgabe eines klicks DANEBEN (dismiss).
            self._focus_index = None
            return True

        self._activate(index)
        return True

    def _activate(self, index):
        """Auswahl, anflug und mond-ansicht -- die drei folgen eines klicks.

        Die ersten beiden sind WORTGLEICH zu `test.py::handle_world_click`:
        erster klick waehlt aus, ein zweiter auf denselben koerper fliegt
        ihn an. Die karte ist ein zweiter weg zu derselben handlung, kein
        zweiter zustand.
        """
        ui_state = self.ui_state
        bodies = list(getattr(self.telemetry.world, 'body', None) or [])
        if not (0 <= index < len(bodies)):
            return

        if ui_state is not None:
            if index == ui_state.selected_index:
                if self.camera is not None:
                    try:
                        self.camera.focus_on(bodies[index])
                    except Exception:
                        pass
            else:
                ui_state.select_body(index)

        # Die mond-ansicht folgt der HIERARCHIE, nicht dem klick: waehlt man
        # einen mond an, bleibt sein planet der mittelpunkt der ansicht --
        # sonst verschwaende der eigene klick genau die monde, unter denen
        # man gerade einen ausgesucht hat.
        _star, planets, _moons = self.structure()
        planet_indices = {i for i, _b in planets}
        if index in planet_indices:
            self._focus_index = index
            return
        parent = getattr(bodies[index], 'is_moon_of', None)
        for planet_index, planet in planets:
            if planet is parent:
                self._focus_index = planet_index
                return

    def dismiss(self):
        """Woanders geklickt -- die karte faehrt ein.

        Derselbe weg, den die koerperliste benutzt: UIRoot ruft dismiss()
        auf jedem widget ausser dem angeklickten, auch bei einem klick ins
        leere. Die mond-ansicht bleibt gemerkt, damit die karte beim
        naechsten oeffnen dort weitermacht, wo man sie verlassen hat.
        """
        self.expanded = False
        self._hover_index = None

    def update(self, ctx, dt):
        super().update(ctx, dt)
        motion = ctx.theme.motion
        # motion.normal (rate 14, ~0.21 s bis 95 %) -- dieselbe rate wie beim
        # aufklappen der koerperliste, und aus demselben grund: bei rate 22
        # springt der erste frame um 30 % und die bewegung liest sich als
        # schnappen statt als ausfahren.
        self._expand_t = ease(self._expand_t, 1.0 if self.expanded else 0.0,
                              motion.normal, dt)
        if not self.expanded and self._expand_t < 0.004:
            self._expand_t = 0.0
        # Die monde blenden LANGSAMER ein als die karte ausfaehrt: sie sind
        # die antwort auf den klick und sollen als eigene bewegung lesbar
        # sein, nicht im ausfahren untergehen.
        self._focus_t = ease(
            self._focus_t,
            1.0 if (self._focus_index is not None and self._expand_t > 0.5) else 0.0,
            motion.slow, dt)

    # -------------------------------------------------------------- zeichnen

    def draw(self, ctx):
        palette = ctx.theme.palette
        self._hits = {}
        fx, fy, fw, fh = self._frame_rect(ctx)
        chrome.frame(ctx, fx, fy, fw, fh, glow_role='orbit')
        chrome.tab(ctx, chrome.tab_text('SYSTEM', 'MAP'), fx + ctx.px(14), fy,
                   color=palette.orbit, edge='top')

        star, planets, moons = self.structure()
        if star is None:
            return

        cx, cy, radius = self._plot(ctx)
        focus = self._focus_index if self._focus_t > 0.004 else None
        # Der rest der karte tritt zurueck, wenn ein planet angewaehlt ist --
        # anteilig am einblend-fortschritt, damit beides EINE bewegung ist.
        dim = 1.0 - (1.0 - _DIMMED) * self._focus_t if focus is not None else 1.0

        star_index, star_body = star
        radii = _log_spread(
            [getattr(b, 'semi_major_axis', 0.0) for _i, b in planets],
            radius * _ORBIT_INNER, radius * _ORBIT_OUTER,
            rank_mix=_RANK_MIX,
        )
        scale = self._dot_scale()
        dots = _log_spread(
            [getattr(b, 'radius', 0.0) for _i, b in planets],
            ctx.px(_DOT_MIN) * scale, ctx.px(_DOT_MAX) * scale,
        )

        # --- bahnen zuerst, damit kein ring ueber einem punkt liegt --------
        line = max(1.0, ctx.px(1.0))
        for slot, (index, body) in enumerate(planets):
            alpha = dim if index != focus else 1.0
            ctx.draw.ring(cx, cy, radii[slot], line,
                          with_alpha(palette.orbit, 0.30 * alpha))

        # --- zentralgestirn ------------------------------------------------
        star_color = _body_color(star_body)
        star_r = ctx.px(_STAR_DOT) * scale
        ctx.draw.circle(cx, cy, star_r * 2.4,
                        fill=with_alpha(star_color, 0.10 * dim))
        ctx.draw.circle(cx, cy, star_r, fill=with_alpha(star_color, dim))
        self._register_hit(ctx, star_index, cx, cy, star_r)
        self._draw_selection_ring(ctx, star_index, cx, cy, star_r, dim)

        # --- planeten -------------------------------------------------------
        for slot, (index, body) in enumerate(planets):
            angle = _true_angle(body, star_body)
            px = cx + math.cos(angle) * radii[slot]
            py = cy + math.sin(angle) * radii[slot]
            alpha = 1.0 if index == focus else dim
            color = _body_color(body)
            ctx.draw.circle(px, py, dots[slot], fill=with_alpha(color, alpha))
            self._register_hit(ctx, index, px, py, dots[slot])
            self._draw_selection_ring(ctx, index, px, py, dots[slot], alpha)
            if index == focus:
                inner = radii[slot - 1] if slot > 0 else radius * _ORBIT_INNER * 0.4
                span = min(max((radii[slot] - inner) * _MOON_SPAN,
                               radius * _MOON_SPAN_MIN),
                           radius * _MOON_SPAN_MAX)
                self._draw_moons(ctx, index, body, px, py, span,
                                 moons.get(index, ()))

        # --- name unter dem zeiger ------------------------------------------
        if self.expanded and self._hover_index is not None:
            self._draw_hover_name(ctx)

    def _register_hit(self, ctx, index, x, y, dot=0.0):
        """Trefferflaeche merken -- nur in der ausgefahrenen karte.

        In der kachel darf ein klick NICHT auf einem koerper landen: er soll
        dort ausschliesslich ausfahren, sonst waehlt schon die oeffnende
        geste einen zufaelligen planeten aus.

        `_HIT_RADIUS` ist ein BODEN, kein mass: die sonne ist mit elf pixeln
        gezeichnet, und ein trefferkreis von genau elf pixeln liesse ihren
        eigenen rand daneben gehen -- man klickte sichtbar auf den koerper
        und traefe nichts.
        """
        if self.expanded and self._expand_t > 0.6:
            reach = max(ctx.px(_HIT_RADIUS), float(dot) + ctx.px(3.0))
            self._hits[index] = (x, y, reach)

    def _draw_selection_ring(self, ctx, index, x, y, dot_radius, alpha):
        """Der ring um den koerper, der IM SPIEL ausgewaehlt ist.

        Er zeigt denselben zustand wie die vier auswahl-pfeile in der welt --
        die karte erfindet keine eigene markierung, sie liest
        ``ui_state.selected_index``.
        """
        if self.ui_state is None or index != self.ui_state.selected_index:
            return
        color = ctx.theme.palette.accent_for('target')
        ctx.draw.ring(x, y, dot_radius + ctx.px(4.0), max(1.0, ctx.px(1.4)),
                      with_alpha(color, 0.95 * alpha))

    def _draw_moons(self, ctx, planet_index, planet, px, py, span, moons):
        """Das mondsystem eines planeten, aufgefaltet um seinen punkt.

        Der platz dafuer ist der abstand zur NAECHST INNEREN bahn, nicht ein
        fester wert: sonst deckte ein mondsystem bei einem eng gestaffelten
        planetenpaar den nachbarn zu.
        """
        if not moons:
            return
        palette = ctx.theme.palette
        fade = self._focus_t
        # ... und der zweite grenzwert ist der RAHMEN. Pluto und Neptun
        # stehen auf der aeussersten bahn, keine drei fingerbreit von der
        # kante; ein mondsystem in voller groesse haengt dort halb aus der
        # karte heraus, was wie ein zeichenfehler aussieht. Der boden von
        # einem zehntel kartenradius laesst es die kante hoechstens
        # beruehren, statt es auf unlesbare paar pixel zusammenzudruecken.
        fx, fy, fw, fh = self._frame_rect(ctx)
        room = min(px - fx, fx + fw - px, py - fy, fy + fh - py) - ctx.px(3.0)
        _cx, _cy, plot_radius = self._plot(ctx)
        span = min(span, max(room, plot_radius * 0.10))
        radii = _log_spread(
            [getattr(b, 'semi_major_axis', 0.0) for _i, b in moons],
            span * _MOON_INNER, span,
        )
        line = max(1.0, ctx.px(0.9))
        for slot, (index, moon) in enumerate(moons):
            # Die bahnen wachsen aus dem planeten HERAUS: bei fade = 0 liegen
            # sie auf seinem punkt, bei 1 auf ihrem platz. Ein blosses
            # aufblenden an fester stelle saehe aus, als waeren sie immer da
            # gewesen und nur uebersehen worden.
            r = radii[slot] * (0.45 + 0.55 * fade)
            ctx.draw.ring(px, py, r, line, with_alpha(palette.orbit, 0.34 * fade))
            angle = _true_angle(moon, planet)
            mx = px + math.cos(angle) * r
            my = py + math.sin(angle) * r
            ctx.draw.circle(mx, my, ctx.px(_MOON_DOT),
                            fill=with_alpha(_body_color(moon), fade))
            if fade > 0.6:
                self._register_hit(ctx, index, mx, my, ctx.px(_MOON_DOT))
            self._draw_selection_ring(ctx, index, mx, my, ctx.px(_MOON_DOT), fade)

    def _draw_hover_name(self, ctx):
        """Der name des koerpers unter dem zeiger, als kleines schild.

        Er erscheint nur beim ueberfahren: dauerhaft angeschrieben waeren die
        planeten bei dieser groesse eine textwand, und die karte soll auf
        einen blick die KONSTELLATION zeigen, nicht eine liste sein.
        """
        hit = self._hits.get(self._hover_index)
        if hit is None:
            return
        bodies = list(getattr(self.telemetry.world, 'body', None) or [])
        if not (0 <= self._hover_index < len(bodies)):
            return
        palette = ctx.theme.palette
        name = str(getattr(bodies[self._hover_index], 'name', '?'))
        x, y, _reach = hit
        text_w, text_h = ctx.text.measure(name, 'map_body')
        pad = ctx.px(6.0)
        width = text_w + pad * 2.0
        height = text_h + ctx.px(4.0)
        left = x - width * 0.5
        top = y - ctx.px(13.0) - height
        # Im rahmen halten -- ein schild, das aus der karte ragt, sieht aus
        # wie ein zeichenfehler.
        fx, fy, fw, fh = self._frame_rect(ctx)
        left = max(fx + ctx.px(3.0), min(left, fx + fw - width - ctx.px(3.0)))
        if top < fy + ctx.px(3.0):
            top = y + ctx.px(13.0)
        chrome.plate(ctx, left, top, width, height,
                     fill=with_alpha(palette.panel_popup, 0.92), cut=3.0)
        ctx.text.draw(name, left + width * 0.5, top + height * 0.5,
                      role='map_body', color=palette.text, align='center',
                      valign='middle')
