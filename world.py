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
            
            # RK4 Stage 1
            self.calculate_forces()
            k1_v = body.acceleration.copy()
            k1_p = body.velocity.copy()
            
            # RK4 Stage 2
            body.position += k1_p * (dt / 2)
            body.velocity += k1_v * (dt / 2)
            self.calculate_forces()
            k2_v = body.acceleration.copy()
            k2_p = body.velocity.copy()
            
            # RK4 Stage 3
            body.position += k2_p * (dt / 2) - k1_p * (dt / 2)
            body.velocity += k2_v * (dt / 2) - k1_v * (dt / 2)
            self.calculate_forces()
            k3_v = body.acceleration.copy()
            k3_p = body.velocity.copy()
            
            # RK4 Stage 4
            body.position += k3_p * dt - k2_p * (dt / 2)
            body.velocity += k3_v * dt - k2_v * (dt / 2)
            self.calculate_forces()
            k4_v = body.acceleration.copy()
            k4_p = body.velocity.copy()
            
            # Combine all stages (weighted average)
            body.position += (k1_p + 2*k2_p + 2*k3_p + k4_p) * (dt / 6) - k3_p * dt
            body.velocity += (k1_v + 2*k2_v + 2*k3_v + k4_v) * (dt / 6) - k3_v * dt
        
        self.time += dt