import math
from operator import pos
from vec import Vec2, vec

class body:
    def __init__(self, name, mass, radius, position, velocity, fixed=False, 
                 semi_major_axis=None, eccentricity=None, period=None, theta0=0.0, 
                 is_moon_of=None, color=(255, 255, 255),
                 has_atmosphere=False,atmosphere_color=(255, 255, 255), atmos_density=0.0, light_intensity=0.0, is_ship=False):
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
        self.color = color
        
        # Atmosphäre und Glow-Eigenschaften
        self.has_atmosphere = bool(has_atmosphere)
        self.atmos_density = float(atmos_density) if self.has_atmosphere else 0.0
        self.light_intensity = float(light_intensity)
        self.atmosphere_color = atmosphere_color if self.has_atmosphere else (0, 0, 0)
    def orbit_position(self, dt, parent_position=None, mu=None):
            """Berechnet Position basierend auf Orbit, nur für Planeten"""
            if self.semi_major_axis == 0.0 or mu is None or mu == 0.0:
                return self.position

            a = float(self.semi_major_axis)
            e = float(self.eccentricity)

            # Aktueller Radius aus Kepler-Formel
            r = a * (1.0 - e * e) / (1.0 + e * math.cos(self.theta))

            # Momentane Geschwindigkeit (approx) und Winkelgeschwindigkeit
            # Use vis-viva for speed magnitude; angular speed approx = v / r
            v = math.sqrt(max(0.0, mu * (2.0 / r - 1.0 / a)))
            omega = v / max(1e-12, r)

            # Advance true anomaly
            self.theta += omega * dt

            # Position in orbital plane (periapsis at angle 0)
            x_orb = r * math.cos(self.theta)
            y_orb = r * math.sin(self.theta)

            # Rotate by argument of periapsis to world coordinates
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

