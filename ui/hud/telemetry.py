"""Datenschicht des HUDs -- rechnet die anzeigewerte EINMAL pro frame aus.

Warum getrennt von den widgets: die bahnelemente kosten ein paar dutzend
gleitkomma-operationen, und AP, PE, ECC, PERIODE und T-AP kommen alle aus
derselben zwischenrechnung. Wuerde jedes label seinen wert selbst holen,
liefe dieselbe kepler-loesung fuenfmal pro frame. Die widgets lesen hier nur
noch fertige felder ab.

INVARIANTEN:
- Nur LESEN. Diese datei fasst keinen simulationszustand an.
- Die physik bleibt absolut (baryzentrisch). Was hier relativ zu einem
  bezugskoerper gerechnet wird, ist reine anzeige.
- SI ueberall. Die formatierung nach km/h:m:s macht ui/units.py.
"""

import math

from .. import units

_TAU = 2.0 * math.pi


def compass_from_theta(theta):
    """Schiffs-theta -> kompasskurs (0 = oben, im uhrzeigersinn).

    theta ist IM UHRZEIGERSINN gemessen und die weltrichtung der nase ist
    (cos theta, -sin theta) -- siehe CLAUDE.md. theta = 0 zeigt damit nach
    rechts, also kompass 90 Grad; daraus folgt kompass = 90 - theta.
    Wer hier ein vorzeichen dreht, dreht die steuerung um.
    """
    return (90.0 - math.degrees(float(theta))) % 360.0


def compass_from_frame_direction(direction):
    """Frame-raum-richtung -> kompasskurs.

    atan2(-d.y, d.x) ist genau die groesse, die auch
    Renderer._apply_orientation_snap gegen theta vergleicht. Beide muessen
    dieselbe formel benutzen, sonst stehen ring-marker und nase nicht
    uebereinander.
    """
    if direction is None:
        return None
    try:
        theta_like = math.atan2(-float(direction.y), float(direction.x))
    except Exception:
        return None
    return (90.0 - math.degrees(theta_like)) % 360.0


class OrbitalElements:
    """Zweikoerper-bahnelemente aus dem zustandsvektor.

    Bewusst aus position und geschwindigkeit gerechnet und NICHT aus den
    predictor-punkten gelesen: die elemente sollen auch dann stimmen, wenn
    der predictor gerade neu rechnet oder abgeschaltet ist. Es ist die
    klassische zweikoerper-naeherung -- bei einem N-koerper-integrator eine
    momentaufnahme des osculating orbit, was genau das ist, was ein HUD
    anzeigen soll.
    """

    __slots__ = ('valid', 'apoapsis', 'periapsis', 'eccentricity', 'period',
                 'semi_major_axis', 'time_to_apoapsis', 'time_to_periapsis',
                 'altitude', 'radius', 'speed', 'closed')

    def __init__(self):
        self.valid = False
        self.closed = False
        self.apoapsis = None
        self.periapsis = None
        self.eccentricity = None
        self.period = None
        self.semi_major_axis = None
        self.time_to_apoapsis = None
        self.time_to_periapsis = None
        self.altitude = None
        self.radius = None
        self.speed = None

    def solve(self, ship, reference_body, gravitational_constant,
              reference_velocity=None):
        self.__init__()
        if ship is None or reference_body is None:
            return self
        if reference_velocity is None:
            reference_velocity = (float(reference_body.velocity.x),
                                  float(reference_body.velocity.y))
        try:
            rx = float(ship.position.x) - float(reference_body.position.x)
            ry = float(ship.position.y) - float(reference_body.position.y)
            vx = float(ship.velocity.x) - float(reference_velocity[0])
            vy = float(ship.velocity.y) - float(reference_velocity[1])
            mu = float(gravitational_constant) * float(reference_body.mass)
        except Exception:
            return self

        r = math.hypot(rx, ry)
        v = math.hypot(vx, vy)
        if r <= 0.0 or mu <= 0.0 or not math.isfinite(r) or not math.isfinite(v):
            return self

        self.valid = True
        self.radius = r
        self.speed = v
        self.altitude = r - float(getattr(reference_body, 'radius', 0.0) or 0.0)

        # Spezifische bahnenergie -> grosse halbachse.
        energy = 0.5 * v * v - mu / r
        # Exzentrizitaetsvektor.
        rv = rx * vx + ry * vy
        factor = v * v - mu / r
        ex = (factor * rx - rv * vx) / mu
        ey = (factor * ry - rv * vy) / mu
        ecc = math.hypot(ex, ey)
        self.eccentricity = ecc

        if energy >= -1e-12:
            # Parabolisch oder hyperbolisch: es gibt kein apoapsis und keine
            # umlaufzeit. Das periapsis existiert weiterhin.
            self.closed = False
            if ecc > 1.0:
                p = (rx * vy - ry * vx) ** 2 / mu
                self.periapsis = p / (1.0 + ecc)
            return self

        self.closed = True
        a = -mu / (2.0 * energy)
        self.semi_major_axis = a
        self.apoapsis = a * (1.0 + ecc)
        self.periapsis = a * (1.0 - ecc)
        self.period = _TAU * math.sqrt(a * a * a / mu)

        # Wahre anomalie -> exzentrische -> mittlere anomalie. Daraus die
        # zeit bis zum naechsten apoapsis bzw. periapsis.
        if ecc > 1e-9:
            cos_nu = (ex * rx + ey * ry) / (ecc * r)
            cos_nu = max(-1.0, min(1.0, cos_nu))
            nu = math.acos(cos_nu)
            if rv < 0.0:
                nu = _TAU - nu
        else:
            # Kreisbahn: kein exzentrizitaetsvektor, also der winkel selbst.
            nu = math.atan2(ry, rx) % _TAU

        half = math.tan(nu * 0.5)
        eccentric = 2.0 * math.atan2(
            math.sqrt(max(0.0, 1.0 - ecc)) * half,
            math.sqrt(max(1e-12, 1.0 + ecc)),
        )
        mean = (eccentric - ecc * math.sin(eccentric)) % _TAU
        n = _TAU / self.period
        self.time_to_apoapsis = ((math.pi - mean) % _TAU) / n
        self.time_to_periapsis = ((_TAU - mean) % _TAU) / n
        return self


class Telemetry:
    """Alle HUD-werte eines frames.

    sample() laeuft einmal pro frame aus der hauptschleife heraus; die
    widgets lesen danach nur noch attribute.
    """

    def __init__(self, world, ship, ship_control, camera, renderer, predictor,
                 ui_state, tick_rate=60.0):
        self.world = world
        self.ship = ship
        self.ship_control = ship_control
        self.camera = camera
        self.renderer = renderer
        self.predictor = predictor
        self.ui_state = ui_state
        self.tick_rate = float(tick_rate)

        self.elements = OrbitalElements()

        self.frame_speed = None
        self.relative_speed = None
        self.heading = 0.0
        self.marker_headings = {}
        self.snap_mode = None

        # RADIALGESCHWINDIGKEIT -- der anteil der bahngeschwindigkeit, der
        # vom bezugskoerper WEG zeigt (v . r_dach). Positiv heisst steigend.
        #
        # Das ist die 2D-entsprechung der senkrechtgeschwindigkeit, die in
        # KSP2 auf dem rechten bogen neben dem navball steht, und sie ist
        # hier die ehrlichere groesse: die simulation kennt keine
        # oberflaechennormale, wohl aber die verbindungslinie zum koerper.
        # Am Ap und Pe ist sie genau null -- das macht den bogen zu einer
        # brauchbaren apsis-anzeige.
        self.radial_speed = None

        self.target_distance = None
        self.target_relative_speed = None
        self.closest_approach = None
        self.time_to_closest = None
        self.target_locked = False

        # Standardgravitationsparameter des bezugskoerpers und die spezifische
        # bahnenergie des schiffs darin. Beide stehen hier, weil sie EINE
        # frage beantworten, die die bahnelemente allein nicht koennen: wie
        # schnell ist das schiff an einem anderen ort SEINER bahn? Genau das
        # will der schwebezettel an der Ap/Pe-raute wissen.
        self.reference_mu = None
        self.specific_energy = None

        self.warp_factor = 1.0
        # Ab welcher raffung (sim-sekunden je echtsekunde) der schub gesperrt
        # ist. Wird von test.py aus simulation.realtime_warp_max gesetzt und
        # ist hier nur vorbelegt, damit das HUD auch allein lauffaehig bleibt.
        self.realtime_warp_max = 60.0
        self.thrust_locked = False

        # Hoechste raffung, bei der die BAHN noch aufgeloest ist. Nahe an einem
        # koerper ist das keine frage der rechenleistung, sondern der physik:
        # ein frame bei 1 y/s rueckt um 48 stunden vor, das sind ~24 umlaeufe
        # eines 2-stunden-orbits. Die kann kein integrator in 40 schritten
        # abbilden -- gemessen 5120 teilschritte und 270 ms je frame.
        # Grenze ist world.characteristic_timescale() / warp_timescale_divisor.
        self.warp_timescale_divisor = 3.0
        self.max_warp_rate = None

        # Schubstufe: das schiff kennt keinen dauerschub, nur impulse pro
        # frame ueber schiffcontrol.thrust_acc. Der regler skaliert genau
        # diesen wert -- er ist damit eine echte, wirksame steuerung und
        # keine attrappe.
        self.thrust_max = float(getattr(ship_control, 'thrust_acc', 600.0) or 600.0)
        self.thrust_level = 1.0

    # ------------------------------------------------------------- abtastung

    def sample(self):
        world = self.world
        ship = self.ship
        reference = self.ui_state.reference_body if self.ui_state else None
        reference_velocity = self.body_velocity(reference)

        self.elements.solve(
            ship, reference, getattr(world, 'G', 6.6730831e-11),
            reference_velocity=reference_velocity,
        )

        self._sample_attitude(ship, reference)
        self._sample_radial(ship, reference, reference_velocity)
        self._sample_target(ship, reference, reference_velocity)
        self._sample_conic(reference)

        self.warp_factor = float(getattr(self.camera, 'sim_dt', 1.0)) * self.tick_rate
        # Im zeitraffer ist der schub gesperrt (siehe test.py): ein impuls je
        # frame waere dort weder dosierbar noch reproduzierbar. Der regler
        # zeigt das an, sonst drueckt der spieler 'Up' und nichts geschieht.
        self.thrust_locked = self.warp_factor > self.realtime_warp_max * 1.001
        self._sample_warp_limit(ship)
        self.snap_mode = getattr(self.ship_control, 'snap_mode', None)

    def _sample_warp_limit(self, ship):
        """Obergrenze der raffung aus der bahn-zeitskala."""
        fn = getattr(self.world, 'characteristic_timescale', None)
        if fn is None:
            self.max_warp_rate = None
            return
        try:
            t_char = fn(ship)
        except Exception:
            t_char = None
        if not t_char or t_char <= 0.0:
            self.max_warp_rate = None
            return
        divisor = max(float(self.warp_timescale_divisor or 3.0), 1e-6)
        # Nie unter die echtzeit-stufe klemmen -- sonst waere das schiff in
        # einem sehr tiefen orbit gar nicht mehr steuerbar.
        self.max_warp_rate = max(t_char / divisor * self.tick_rate,
                                 self.realtime_warp_max)

    def warp_step_allowed(self, rate):
        """Darf diese stufe gewaehlt werden?"""
        cap = self.max_warp_rate
        return cap is None or float(rate) <= cap * 1.001

    def _sample_conic(self, reference):
        """mu und spezifische bahnenergie -- die grundlage von vis-viva."""
        self.reference_mu = None
        self.specific_energy = None
        e = self.elements
        if reference is None or not e.valid or not e.radius or e.radius <= 0.0:
            return
        try:
            mu = float(getattr(self.world, 'G', 6.6730831e-11)) * float(reference.mass)
        except Exception:
            return
        if mu <= 0.0 or e.speed is None:
            return
        self.reference_mu = mu
        self.specific_energy = 0.5 * e.speed * e.speed - mu / e.radius

    def speed_at_radius(self, radius):
        """Bahntempo relativ zum bezugskoerper in `radius` metern abstand.

        Vis-viva auf der OSKULIERENDEN bahn: v = sqrt(2 (E + mu/r)). Genau
        dieselbe zweikoerper-momentaufnahme, aus der auch AP, PE und die
        beiden countdowns kommen -- der schwebezettel an der raute sagt damit
        nichts anderes als der rest des HUDs.

        Nicht aus den predictor-punkten interpoliert, obwohl die raute von
        dort kommt: deren geschwindigkeits-spalten sind bei den sehnen-
        kernen NaN (siehe .claude/rules/predictor.md), und ein wert, der je
        nach qualitaetsstufe da ist oder nicht, ist als anzeige unbrauchbar.
        """
        mu = self.reference_mu
        energy = self.specific_energy
        try:
            radius = float(radius)
        except (TypeError, ValueError):
            return None
        if not mu or energy is None or radius <= 0.0:
            return None
        v_squared = 2.0 * (energy + mu / radius)
        if v_squared <= 0.0 or not math.isfinite(v_squared):
            return None
        return math.sqrt(v_squared)

    def _sample_attitude(self, ship, reference):
        renderer = self.renderer
        if ship is None or renderer is None:
            return

        self.heading = compass_from_theta(getattr(ship, 'theta', 0.0))

        try:
            self.frame_speed = renderer._ship_frame_speed_m_s(ship)
        except Exception:
            self.frame_speed = None
        try:
            self.relative_speed = renderer._ship_relative_speed_m_s(
                ship, reference_body=reference
            )
        except Exception:
            self.relative_speed = None

        # Die marker sitzen auf den GEZEICHNETEN orbital-vektoren, nicht auf
        # frisch gerechneten: sonst zeigt der ring woanders hin als der pfeil
        # und der autopilot, der an dieselbe quelle gebunden ist.
        try:
            points = self.predictor.get_points() if self.predictor else None
            _, directions = renderer.orbital_frame_directions(
                ship, reference_body=reference, prediction_points=points
            )
        except Exception:
            directions = {}

        self.marker_headings = {
            name: compass_from_frame_direction(vector)
            for name, vector in (directions or {}).items()
        }

    def body_velocity(self, body):
        """Geschwindigkeit eines koerpers -- auch eines skriptgefuehrten.

        ACHTUNG, das ist keine feinheit: world.update_planets() setzt bei
        koerpern mit Kepler-elementen (fixed=True, wie Erde, Mond, Mars) NUR
        die position neu. Ihr velocity-feld behaelt den ladewert, meist
        (0, 0). Wer es direkt liest, rechnet gegen einen stillstehenden
        planeten -- eine saubere kreisbahn um die Erde kommt dann als
        hyperbel mit exzentrizitaet 4 heraus.

        Deshalb hier eine zentrale differenz ueber position_at_time(), also
        genau die funktion, die auch der integrator fuer bewegte
        gravitationsquellen benutzt.
        """
        if body is None:
            return (0.0, 0.0)
        if not getattr(body, 'scripted_orbit', False):
            try:
                return (float(body.velocity.x), float(body.velocity.y))
            except Exception:
                return (0.0, 0.0)
        try:
            now = float(getattr(self.world, 'time', 0.0))
            h = 1.0
            ahead = body.position_at_time(now + h)
            behind = body.position_at_time(now - h)
            return ((float(ahead.x) - float(behind.x)) / (2.0 * h),
                    (float(ahead.y) - float(behind.y)) / (2.0 * h))
        except Exception:
            try:
                return (float(body.velocity.x), float(body.velocity.y))
            except Exception:
                return (0.0, 0.0)

    def _sample_radial(self, ship, reference, reference_velocity=None):
        """v . r_dach gegen den bezugskoerper -- steig- bzw. sinkrate.

        Die geschwindigkeit des KOERPERS wird abgezogen, und zwar ueber
        body_velocity(): das gespeicherte velocity-feld skriptgefuehrter
        koerper steht auf null (siehe dort), womit die rate bei einem
        planeten auf sonnenbahn um dessen 30 km/s danebenlaege.
        """
        self.radial_speed = None
        if ship is None or reference is None:
            return
        try:
            rx = float(ship.position.x) - float(reference.position.x)
            ry = float(ship.position.y) - float(reference.position.y)
            distance = math.hypot(rx, ry)
            if distance <= 0.0:
                return
            base = reference_velocity or (0.0, 0.0)
            vx = float(ship.velocity.x) - float(base[0])
            vy = float(ship.velocity.y) - float(base[1])
            self.radial_speed = (vx * rx + vy * ry) / distance
        except Exception:
            self.radial_speed = None

    def _sample_target(self, ship, reference, reference_velocity=None):
        self.target_locked = reference is not None
        if ship is None or reference is None:
            self.target_distance = None
            self.target_relative_speed = None
            self.closest_approach = None
            self.time_to_closest = None
            return
        if reference_velocity is None:
            reference_velocity = self.body_velocity(reference)

        try:
            self.target_distance = math.hypot(
                float(ship.position.x) - float(reference.position.x),
                float(ship.position.y) - float(reference.position.y),
            )
            self.target_relative_speed = math.hypot(
                float(ship.velocity.x) - float(reference_velocity[0]),
                float(ship.velocity.y) - float(reference_velocity[1]),
            )
        except Exception:
            self.target_distance = None
            self.target_relative_speed = None

        # Naechste annaeherung = das naechste PERIAPSIS der vorhersagelinie.
        # Der predictor sucht diese marker ohnehin schon (kind 0.0 =
        # periapsis, r = abstand zum referenzkoerper), also wird hier nichts
        # doppelt gerechnet.
        self.closest_approach = None
        self.time_to_closest = None
        predictor = self.predictor
        if predictor is None or not hasattr(predictor, 'get_apsis_markers'):
            return
        try:
            markers = predictor.get_apsis_markers()
            now = float(getattr(self.world, 'time', 0.0))
            best_t = None
            for row in markers:
                if float(row[3]) != 0.0:      # nur periapsis
                    continue
                dt = float(row[2]) - now
                if dt < 0.0:
                    continue
                if best_t is None or dt < best_t:
                    best_t = dt
                    self.closest_approach = float(row[4])
                    self.time_to_closest = dt
        except Exception:
            pass

    # ------------------------------------------------------ formatierte werte

    # AP/PE SIND ABSTAENDE ZUM MITTELPUNKT, NICHT HOEHEN UEBER GRUND.
    #
    # Sie standen einmal als `apsis - koerperradius` da, also als hoehe. Die
    # Ap/Pe-fahnen auf der linie zeigen aber `r` aus
    # `Predictor.get_apsis_markers()`, und das ist der abstand zum
    # MITTELPUNKT des referenzkoerpers. Zwei zahlen fuer denselben punkt, die
    # sich um einen ganzen koerperradius unterscheiden (bei Erde 6371 km) --
    # und nichts an der anzeige sagt, welche welche ist. Beide messen jetzt
    # vom mittelpunkt; die hoehe ueber grund steht weiterhin im ALT-feld,
    # wo sie hingehoert (siehe text_altitude).
    def text_apoapsis(self):
        e = self.elements
        if not e.valid or e.apoapsis is None:
            return '--'
        return units.distance(e.apoapsis)

    def text_periapsis(self):
        e = self.elements
        if not e.valid or e.periapsis is None:
            return '--'
        return units.distance(e.periapsis)

    def text_eccentricity(self):
        return units.eccentricity(self.elements.eccentricity)

    def text_period(self):
        if not self.elements.closed:
            return 'ESCAPE'
        return units.duration(self.elements.period)

    def text_time_to_apoapsis(self):
        if not self.elements.closed:
            return '--'
        return units.duration(self.elements.time_to_apoapsis)

    def text_speed(self):
        speed = self.frame_speed
        if speed is None:
            return '--'
        if speed >= 100000.0:
            return f"{speed / 1000.0:,.0f}".replace(',', ' ')
        return f"{speed:,.0f}".replace(',', ' ')

    def text_speed_unit(self):
        speed = self.frame_speed
        return 'KM/S' if speed is not None and speed >= 100000.0 else 'M/S'

    def text_heading(self):
        return f"{int(round(self.heading)) % 360:03d}°"

    def text_target_distance(self):
        return units.distance(self.target_distance)

    def text_target_relative_speed(self):
        return units.speed(self.target_relative_speed)

    def text_closest(self):
        # Dieselbe groesse wie die Pe-fahne (siehe text_periapsis): sie kommt
        # aus demselben marker, also auch vom mittelpunkt gemessen.
        if self.closest_approach is None:
            return '--'
        return units.distance(self.closest_approach)

    def text_time_to_closest(self):
        return units.duration(self.time_to_closest)

    def text_warp(self):
        return units.time_warp(self.warp_factor)

    def text_mission_time(self):
        """Die laufende weltzeit als 'T+000y 012d 04:05'.

        Sie gehoert neben den zeitraffer: eine raffungsstufe ohne die zeit,
        die sie erzeugt, ist eine zahl ohne sichtbare wirkung -- und bei
        1 y/s ist die jahresstelle die einzige, an der man ueberhaupt noch
        etwas laufen sieht.
        """
        try:
            seconds = max(0.0, float(getattr(self.world, 'time', 0.0)))
        except Exception:
            return 'T+--'
        years, rest = divmod(int(seconds), 31557600)
        days, rest = divmod(rest, 86400)
        hours, rest = divmod(rest, 3600)
        minutes = rest // 60
        return f"T+{years:03d}y {days:03d}d {hours:02d}:{minutes:02d}"

    def view_mode_label(self):
        """SURFACE / ORBITAL / TARGET -- identisch zur rahmenwahl im HUD."""
        if self.ui_state is None:
            return 'ORBITAL'
        return ('SURFACE', 'ORBITAL', 'TARGET')[self.ui_state.view_mode()]

    def text_throttle(self):
        return f"{int(round(self.thrust_level * 100.0))}%"

    # ------------------------------------------- die beiden navball-flanken

    def gauge_speed(self):
        """Bahngeschwindigkeit als (zahl, einheit) fuer die linke flanke."""
        return units.split_speed(self.frame_speed)

    def gauge_altitude(self):
        """Hoehe ueber der oberflaeche des bezugskoerpers, als (zahl, einheit).

        Ueber der OBERFLAECHE, nicht ueber dem mittelpunkt: alles andere
        waere neben einem planeten mit 6 371 km radius sinnlos. AP und PE
        messen dagegen vom MITTELPUNKT -- so wie die fahnen auf der linie
        (siehe text_periapsis).
        """
        e = self.elements
        if not e.valid or e.altitude is None:
            return ('--', '')
        return units.split_distance(e.altitude)

    def text_time_to_periapsis(self):
        if not self.elements.closed:
            return '--'
        return units.countdown(self.elements.time_to_periapsis)

    def text_countdown_to_apoapsis(self):
        if not self.elements.closed:
            return '--'
        return units.countdown(self.elements.time_to_apoapsis)

    def orbital_speed_scale(self):
        """(bahntempo, kreisbahn-tempo, fluchttempo) am aktuellen ort.

        Der MASSSTAB der geschwindigkeitsnadel, und er wird aus der bahn
        gerechnet statt festgelegt. Vorher stand im ring ein fester
        vollausschlag von 2600 m/s -- eine zahl, die zu nichts gehoert: im
        niedrigen Erdorbit fliegt man 7.7 km/s und die nadel klebt am
        anschlag, um Pluto 4.7 km/s und ebenfalls, waehrend sie um einen
        kleinen mond ueberhaupt nicht ausschlaegt.

        Die beiden ehrlichen bezugsgroessen liefert der zentralkoerper
        selbst:

            v_kreis = sqrt(mu / r)          eine kreisbahn hier
            v_flucht = sqrt(2) * v_kreis    gerade noch gebunden

        Damit bedeutet dieselbe nadellaenge an JEDEM koerper dasselbe --
        kurz = unterhalb der kreisbahn, drei viertel = kreisbahn, voll =
        fluchtgeschwindigkeit. Das ist die "an jeden planeten angepasste
        proportion mit realistischer obergrenze".

        Gerechnet wird gegen die geschwindigkeit RELATIV zum bezugskoerper
        (OrbitalElements.speed), nicht gegen die baryzentrische: sonst
        traegt eine Erdumlaufbahn die 30 km/s der Erde um die Sonne mit
        und die nadel steht immer am anschlag.
        """
        e = self.elements
        reference = self.ui_state.reference_body if self.ui_state else None
        if not e.valid or reference is None or not e.radius or e.radius <= 0.0:
            return (None, None, None)
        try:
            mu = float(getattr(self.world, 'G', 6.6730831e-11)) * float(reference.mass)
        except Exception:
            return (None, None, None)
        if mu <= 0.0:
            return (None, None, None)
        circular = math.sqrt(mu / e.radius)
        return (e.speed, circular, circular * math.sqrt(2.0))

    def velocity_fraction(self):
        """Nadellaenge auf [0, 1]: 0 = ruhe, ~0.71 = kreisbahn, 1 = flucht."""
        speed, _circular, escape = self.orbital_speed_scale()
        if speed is None or not escape or escape <= 0.0:
            return None
        return max(0.0, min(1.0, float(speed) / float(escape)))

    def circular_speed_fraction(self):
        """Wo auf derselben skala die kreisbahn liegt -- immer 1/sqrt(2).

        Steht hier trotzdem als funktion, weil der ring die marke daraus
        zeichnet: eine skalierte nadel ohne bezugsmarke ist nur ein
        zappelnder strich.
        """
        _speed, circular, escape = self.orbital_speed_scale()
        if circular is None or not escape or escape <= 0.0:
            return None
        return float(circular) / float(escape)

    def radial_fraction(self, full_scale=None):
        """Radialgeschwindigkeit auf [-1, 1], fuer den rechten bogen.

        MASSSTAB AUS DER BAHN, nicht fest: die auftretenden raten reichen
        von metern je sekunde im niedrigen orbit bis zu kilometern je
        sekunde auf einem transfer. Ein fester vollausschlag waere in einem
        der beiden faelle immer nutzlos. Bezug ist deshalb die
        bahngeschwindigkeit selbst -- auf einer kreisbahn ist die radialrate
        null, auf einer stark elliptischen erreicht sie einen guten teil
        davon. Vollausschlag heisst damit "die gesamte bewegung geht
        senkrecht vom koerper weg oder auf ihn zu".

        Bezugsgroesse ist OrbitalElements.speed, also die geschwindigkeit
        RELATIV ZUM BEZUGSKOERPER -- nicht frame_speed. Die haengt am
        gewaehlten plotting-rahmen, und dann skalierte sich ein bogen neu,
        weil jemand die ANSICHT umgestellt hat. Zaehler und nenner muessen
        aus derselben bewegung kommen, sonst misst der bogen nichts.
        """
        value = self.radial_speed
        if value is None:
            return None
        if full_scale is None:
            full_scale = self.elements.speed if self.elements.valid else None
            if not full_scale:
                full_scale = self.relative_speed or self.frame_speed
        if not full_scale or full_scale <= 0.0:
            return None
        return max(-1.0, min(1.0, float(value) / float(full_scale)))

    # ------------------------------------------------------------- steuerung

    def set_thrust_level(self, level):
        """Skaliert die schubbeschleunigung des schiffs (0 .. 1 der obergrenze)."""
        self.thrust_level = max(0.0, min(1.0, float(level)))
        if self.ship_control is not None:
            self.ship_control.thrust_acc = self.thrust_max * self.thrust_level
