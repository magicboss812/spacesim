import math
from operator import pos
from physics.vec import Vec2, vec, G


def kepler_relative_xy(a, e, nu0, arg, mu, dt):
    """Bahnposition relativ zum mutterkoerper nach `dt`, EXAKT geloest.

    Rueckgabe `(x, y, nu_t)` -- die relativposition und die wahre anomalie am
    ende, oder `None`, wenn die elemente keine geschlossene bahn beschreiben.

    > **Das war einmal ein Euler-schritt, und das ist der grund, warum der
    > zeitraffer die physik verstellt hat.** Vorher stand hier
    > `theta += (v/r)*dt` mit `v` aus vis-viva -- erster ordnung, mit der
    > CHUNK-GROESSE als schrittweite. `test.py::step_simulation` zerlegt den
    > frame in `max(max_substep_seconds, warp-decke)` grosse stuecke, also
    > 1000 s in echtzeit und 4375 s bei 1 y/s: die schrittweite des
    > koerpermodells haengt damit an der raffungsstufe. Gemessen ueber
    > 20 tage wanderte der Mond dadurch um 7.5e4 m (chunk 1000) bis 1.3e6 m
    > (chunk 16000) gegen den feinen lauf, und auf einer mondtransferbahn
    > kam ueber 25 tage ein perigaeum von 9.05e6 m (chunk 4375) gegen
    > 6.76e6 m (chunk 30) heraus.
    >
    > Zwei fehler steckten darin. `omega = v/r` ist nur an den apsiden die
    > wahre winkelrate -- dazwischen enthaelt `v` die RADIALKOMPONENTE, die
    > sich gar nicht dreht, der koerper lief also systematisch zu schnell
    > (~e^2/4 je umlauf). Und die position wurde aus dem radius zum ALTEN
    > winkel mit dem NEUEN winkel gebildet, der radius hinkte also einen
    > schritt hinterher.
    >
    > Exakt geloest ist die fortschreibung von der schrittweite unabhaengig,
    > weil die zusammensetzung exakter schritte wieder der exakte schritt
    > ist. Der rechenweg ist WORT FUER WORT der von
    > `predictor._body_kepler_constants_numba` /
    > `_body_scripted_relative_xy_numba` -- welt und vorhersage rechnen
    > damit dasselbe koerpermodell, was der ganze punkt ist: die linie zeigt
    > sonst eine bahn, die die welt gar nicht fliegt.
    """
    if a is None or a <= 0.0 or mu is None or mu <= 0.0:
        return None
    e = 0.0 if not e else float(e)
    if e < 0.0 or e >= 1.0:
        return None

    cos_nu0 = math.cos(nu0)
    sin_nu0 = math.sin(nu0)
    denom = 1.0 + e * cos_nu0
    if abs(denom) <= 1e-14:
        return None

    sqrt_one_minus_e2 = math.sqrt(max(0.0, 1.0 - e * e))
    sin_e0 = sqrt_one_minus_e2 * sin_nu0 / denom
    cos_e0 = (e + cos_nu0) / denom
    ecc_anomaly0 = math.atan2(sin_e0, cos_e0)
    mean_anomaly0 = ecc_anomaly0 - e * math.sin(ecc_anomaly0)
    mean_motion = math.sqrt(mu / (a * a * a))

    mean_anomaly = mean_anomaly0 + mean_motion * dt
    two_pi = 2.0 * math.pi
    # Auf [-pi, pi) falten -- ohne das startet Newton bei grossem dt auf
    # einer weit entfernten wurzel.
    mean_anomaly = (mean_anomaly + math.pi) % two_pi
    if mean_anomaly < 0.0:
        mean_anomaly += two_pi
    mean_anomaly -= math.pi

    ecc_anomaly = mean_anomaly
    for _ in range(12):
        f = ecc_anomaly - e * math.sin(ecc_anomaly) - mean_anomaly
        fp = 1.0 - e * math.cos(ecc_anomaly)
        if abs(fp) <= 1e-14:
            break
        delta = f / fp
        ecc_anomaly -= delta
        if abs(delta) <= 1e-13:
            break

    cos_e = math.cos(ecc_anomaly)
    sin_e = math.sin(ecc_anomaly)
    r = a * (1.0 - e * cos_e)
    if r <= 0.0 or not math.isfinite(r):
        return None

    nu = math.atan2(sqrt_one_minus_e2 * sin_e, cos_e - e)
    x_orb = r * math.cos(nu)
    y_orb = r * math.sin(nu)
    c = math.cos(arg)
    s = math.sin(arg)
    return x_orb * c - y_orb * s, x_orb * s + y_orb * c, nu


class body:
    def __init__(self, name, mass, radius, position, velocity, fixed=False, 
                 semi_major_axis=None, eccentricity=None, period=None, theta0=0.0, 
                 is_moon_of=None, color=(255, 255, 255),
                 has_atmosphere=False,atmosphere_color=(255, 255, 255), atmos_density=0.0, light_intensity=0.0, is_ship=False,
                 style_seed=None, style_mode=None, style_shape=None):
        self.name = name
        self.mass = float(mass)
        self.radius = float(radius)

        self.position = position.copy()
        self.velocity = velocity.copy()

        self.acceleration = vec(0.0, 0.0)
        self.fixed = bool(fixed)  
        self.is_ship = bool(is_ship)  # Korrekt aus Parameter setzen
        # Orbit-Parameter nur für Planeten
        self.semi_major_axis = semi_major_axis
        self.eccentricity = eccentricity
        self.period = period
        self.theta = theta0
        self.is_moon_of = is_moon_of
        self.scripted_orbit = fixed
        # Argument of periapsis / orbit rotation (radians)
        self.arg_periapsis = 0.0
        self.released = False
        # Kepler epoch for time-accurate force evaluation (set by world.update_planets)
        self._kepler_ref_time = 0.0
        self._kepler_ref_theta = float(theta0)
        self.color = color
        
        # Atmosphäre und Glow-Eigenschaften
        self.has_atmosphere = bool(has_atmosphere)
        self.atmos_density = float(atmos_density) if self.has_atmosphere else 0.0
        self.light_intensity = float(light_intensity)
        self.atmosphere_color = atmosphere_color if self.has_atmosphere else (0, 0, 0)

        # Prozedurale vektor-optik (siehe body_style.py). Alle drei duerfen
        # None bleiben: der seed wird dann aus dem NAMEN abgeleitet, muster
        # und figur stehen auf 'bands' / 'nested'. Das feld existiert nur,
        # damit ein einzelner koerper bewusst anders aussehen kann.
        self.style_seed = None if style_seed is None else (int(style_seed) & 0xFFFFFFFF)
        self.style_mode = style_mode
        self.style_shape = style_shape
    def position_at_time(self, t):
        """Return this body's Kepler position at simulation time t without modifying state.

        Uses the epoch (_kepler_ref_time, _kepler_ref_theta) written by world.update_planets
        to extrapolate analytically.  Safe to call from inside the integrator's force loops.
        Bodies without a parent orbit (Sun, fixed bodies) return their current .position.
        """
        if self.is_moon_of is None:
            return self.position
        if self.semi_major_axis is None or self.semi_major_axis == 0.0:
            return self.position

        a = float(self.semi_major_axis)
        e = float(self.eccentricity) if self.eccentricity else 0.0
        mu = G * self.is_moon_of.mass
        if mu <= 0.0:
            return self.position

        # Exakt aus der epoche fortgeschrieben -- dieselbe loesung, die
        # update_planets und der predictor benutzen. Die frühere fassung
        # extrapolierte mit KONSTANTER winkelrate ab dem bookmark; ihr fehler
        # wuchs also mit dem alter des bookmarks, und das ist die chunk-
        # groesse, und die haengt an der raffung (siehe kepler_relative_xy).
        solved = kepler_relative_xy(
            a, e, self._kepler_ref_theta, self.arg_periapsis, mu,
            t - self._kepler_ref_time,
        )
        if solved is None:
            return self.position
        pos = Vec2(solved[0], solved[1])

        parent = self.is_moon_of
        if hasattr(parent, 'position_at_time'):
            pos += parent.position_at_time(t)
        else:
            pos += parent.position
        return pos

    def orbit_position(self, dt, parent_position=None, mu=None):
            """Position nach `dt` auf der eigenen bahn -- exakt, siehe
            kepler_relative_xy(). Schreibt `self.theta` mit fort."""
            if self.semi_major_axis == 0.0 or mu is None or mu == 0.0:
                return self.position

            solved = kepler_relative_xy(
                float(self.semi_major_axis), float(self.eccentricity),
                self.theta, self.arg_periapsis, mu, dt,
            )
            if solved is None:
                return self.position
            x, y, nu = solved
            self.theta = nu

            pos = Vec2(x, y)
            if parent_position is not None:
                pos += parent_position
            return pos
class schiff(body):
    def __init__(self, name, position, velocity, color=(255, 255, 255)):
        super().__init__(name=name, mass=0, radius=0, position=position, velocity=velocity, fixed=False)
        self.is_ship = True
        self.color = color

