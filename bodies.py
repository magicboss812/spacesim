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
        self.released = False
        self.color = color
        
        # Atmosphäre und Glow-Eigenschaften
        self.has_atmosphere = bool(has_atmosphere)
        self.atmos_density = float(atmos_density) if self.has_atmosphere else 0.0
        self.light_intensity = float(light_intensity)
        self.atmosphere_color = atmosphere_color if self.has_atmosphere else (0, 0, 0)
    def orbit_position(self, dt, parent_position=None, mu=None):
            """Berechnet Position basierend auf Orbit, nur für Planeten"""
            if self.semi_major_axis == 0.0 or mu == 0.0:
                return self.position
            a = self.semi_major_axis
            e = self.eccentricity
            # Aktueller Radius
            r = a * (1 - e**2) / (1 + e * math.cos(self.theta))
            # Momentane Geschwindigkeit tangential
            v = math.sqrt(mu * (2/r - 1/a))
            # Winkelgeschwindigkeit: omega = v / r
            omega = v / r
            # true anomaly update
            self.theta += omega * dt
            x = r * math.cos(self.theta)
            y = r * math.sin(self.theta)
            pos = Vec2(x, y)
            if parent_position is not None:
                pos += parent_position
            return pos
class schiff(body):
    def __init__(self, name, position, velocity, color=(255, 255, 255)):
        super().__init__(name=name, mass=0, radius=0, position=position, velocity=velocity, fixed=False)
        self.is_ship = True
        self.color = color

