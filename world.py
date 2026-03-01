import math
import pygame
from schiff import schiffcontrol
from vec import Vec2, vec
from bodies import body
G = 6.6730831e-11

class world:

    def __init__(self, G):
        self.G = G
        self.body = []
        self.time = 0.0

# Mithilfe von should_release und release_body wird geprüft, ob ein Körper zu weit von seinem Bezugskörper entfernt ist.
# Für zu hohe Abstände ergibt es keinen Sinn mehr, wenn der Körper dennoch um seinen Bezugskörper kreist, da die Gravitation zu schwach wäre.
# Es wird die Gravitationsbeschleunigung am aktuellen Abstand berechnet und determiniert ob sie unter einem definierten Schwellenwert liegt
# Hier werden beide Funktionen erstmal aufgestellt und definiert, später in update() werden sie aufgerufen und ausgeführt
# WARUM: Besonders hilfreich, wenn es um Custom Systeme geht, bei denen der Spieler ausversehen zu hohe Abstände definiert, die dann nicht mehr physikalisch korrekt
# So gibt es zumindest immer noch eine gewisse "Schwierigkeit" für den Spieler. Der Körper habe dann eine "komische" Bahn und gälte dann als eine extra Herausforderung

    def should_release(self, body):
        if body.is_moon_of is None:
            return False

        parent = body.is_moon_of
        
        r = (body.position - parent.position).magnitude()
        if r < 1e-10:
            return False
            
        gravitational_acc = self.G * parent.mass / (r * r)
        
        MIN_GRAVITY_THRESHOLD = 1e-3 # m/s^2 
        
        return gravitational_acc < MIN_GRAVITY_THRESHOLD

    def release_body(self, body):
        if body.is_ship is True:
            return False
        else:
            parent = body.is_moon_of

            # Radiusvektor (from parent to body)
            delta = body.position - parent.position
            r = delta.magnitude()

            # Orbital parameters
            a = body.semi_major_axis
            e = body.eccentricity if body.eccentricity else 0.0
            mu = self.G * parent.mass
            
            theta = body.theta
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            
            p = a * (1 - e * e)
            
            h = math.sqrt(mu * p)
            
            v_r = (mu / h) * e * sin_theta
            v_t = (mu / h) * (1 + e * cos_theta)
            
            radial = delta.normalize()
            tangent = Vec2(-radial.y, radial.x)
        
            body.velocity = radial * v_r + tangent * v_t

            body.scripted_orbit = False
            body.is_moon_of = None
            body.released = True
    def update_planets(self, dt):
        for body in self.body:
            # Überspringe Schiffe komplett - sie haben keine orbit_position
            if body.is_ship:
                continue
            if not body.scripted_orbit:
                continue
            parent_pos = body.is_moon_of.position if body.is_moon_of else None
            mu = self.G * body.is_moon_of.mass if body.is_moon_of else None
            
            # ERST Position aktualisieren
            body.position = body.orbit_position(dt, parent_pos, mu)
            
            # DANN prüfen ob Release nötig
            if self.should_release(body):
                self.release_body(body)
    def calculate_forces(self):

        for body in self.body:
            if body.scripted_orbit:
                continue
            body.acceleration.clear()
            for other in self.body:
                if other is body:
                    continue
                delta = other.position - body.position
                r2 = delta.magnitude_squared()
                if r2 < 1e-10:
                    continue
                r = math.sqrt(r2)
                factor = self.G * other.mass / (r2 * r)
                body.acceleration += delta * factor
    def update_dynamics(self, dt):
        for body in self.body:
            if body.scripted_orbit:
                continue
            old_acc = body.acceleration.copy()
            body.position += body.velocity * dt + 0.5 * old_acc * dt**2
            body._old_acc = old_acc
        self.calculate_forces()
        for body in self.body:
            if body.scripted_orbit:
                continue
            body.velocity += 0.5 * (body._old_acc + body.acceleration) * dt
        self.time += dt