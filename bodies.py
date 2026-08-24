import math
from operator import pos
from vec import Vec2, vec, G

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

        ref_theta = self._kepler_ref_theta
        delta_t = t - self._kepler_ref_time

        r_ref = a * (1.0 - e * e) / (1.0 + e * math.cos(ref_theta))
        v_ref = math.sqrt(max(0.0, mu * (2.0 / r_ref - 1.0 / a)))
        omega_ref = v_ref / max(1e-12, r_ref)
        theta_t = ref_theta + omega_ref * delta_t

        r_t = a * (1.0 - e * e) / (1.0 + e * math.cos(theta_t))
        x_orb = r_t * math.cos(theta_t)
        y_orb = r_t * math.sin(theta_t)

        c = math.cos(self.arg_periapsis)
        s = math.sin(self.arg_periapsis)
        pos = Vec2(x_orb * c - y_orb * s, x_orb * s + y_orb * c)

        parent = self.is_moon_of
        if hasattr(parent, 'position_at_time'):
            pos += parent.position_at_time(t)
        else:
            pos += parent.position
        return pos

    def orbit_position(self, dt, parent_position=None, mu=None):
            """Berechnet Position basierend auf Orbit, nur für Planeten"""
            if self.semi_major_axis == 0.0 or mu is None or mu == 0.0:
                return self.position

            a = float(self.semi_major_axis)
            e = float(self.eccentricity)

            # aktueller radius aus kepler-formel
            r = a * (1.0 - e * e) / (1.0 + e * math.cos(self.theta))

            # momentane geschwindigkeit (approx) und winkelgeschwindigkeit
            # vis-viva für geschwindigkeitsbetrag verwenden; winkelgeschwindigkeit approx = v / r
            v = math.sqrt(max(0.0, mu * (2.0 / r - 1.0 / a)))
            omega = v / max(1e-12, r)

            # wahre anomalie voranschreiten
            self.theta += omega * dt

            # position in orbitalebene (periapsis bei winkel 0)
            x_orb = r * math.cos(self.theta)
            y_orb = r * math.sin(self.theta)

            # mit periapsis-argument in weltkoordinaten rotieren
            c = math.cos(self.arg_periapsis)
            s = math.sin(self.arg_periapsis)
            x = x_orb * c - y_orb * s
            y = x_orb * s + y_orb * c

            pos = Vec2(x, y)
            if parent_position is not None:
                pos += parent_position
            return pos
class schiff(body):
    def __init__(self, name, position, velocity, color=(255, 255, 255)):
        super().__init__(name=name, mass=0, radius=0, position=position, velocity=velocity, fixed=False)
        self.is_ship = True
        self.color = color

