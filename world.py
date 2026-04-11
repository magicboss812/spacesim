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
        # Epicycle (Ptolemaic) mode state. When enabled, top-level bodies
        # (those with no parent) will be reparented to the chosen center
        # body so that the resulting motion produces epicycles relative
        # to that center.
        self._epicycle_enabled = False
        self._epicycle_center = None
        self._epicycle_saved = {}

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

        # Hinweis: Der Epizykel-Modus wird durch Umparenting der Top-Level-
        # Körper zum gewählten Zentrum via `enable_epicycles()` aktiviert; 
        # update_planets folgt einfach den aktuell gesetzten Elternbeziehungen.

    def _rv_to_orbital(self, r_vec, v_vec, mu):
        """konvertiert position/geschwindigkeit (relativ zum parent) in orbitale elementen.

        gibt (a, e, theta, arg_peri) zurück wobei theta die wahre anomalie ist gemessen
        vom periapsis und arg_peri das periapsis-argument (radian) ist.
        """
        r = r_vec.magnitude()
        v = v_vec.magnitude()
        if r <= 0.0 or mu is None or mu <= 0.0:
            return None

        # specific angular momentum (scalar z-component)
        h = r_vec.x * v_vec.y - r_vec.y * v_vec.x

        # specific energy
        eps = 0.5 * v * v - mu / r
        if abs(eps) < 1e-20:
            # Division durch Null vermeiden; Fallback: kreisförmige Bahn
            a = r
        else:
            a = -mu / (2.0 * eps)

        # eccentricity vector: (v x h)/mu - r_hat
        # v x h_vec (2D) => (h * v_y, -h * v_x)
        vxh_x = h * v_vec.y
        vxh_y = -h * v_vec.x
        evec_x = vxh_x / mu - r_vec.x / r
        evec_y = vxh_y / mu - r_vec.y / r
        e = math.sqrt(evec_x * evec_x + evec_y * evec_y)

        # argument of periapsis (direction of e-vector)
        arg_peri = math.atan2(evec_y, evec_x) if e > 1e-12 else 0.0

        # true anomaly measured from periapsis: angle(r) - arg_peri
        theta_world = math.atan2(r_vec.y, r_vec.x)
        theta = theta_world - arg_peri
        # normalize to 0..2pi
        theta = (theta + 2.0 * math.pi) % (2.0 * math.pi)

        return a, e, theta, arg_peri
    def calculate_forces(self):

        for body in self.body:
            # überspringe körper die scripted sind (deren positionen durch
            # orbit-skripte vorgegeben werden) oder explizit als `fixed` markiert sind —
            # fixe körper sollten nicht vom dynamik-solver integriert werden.
            if body.scripted_orbit or getattr(body, 'fixed', False):
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
            # scripted-orbit körper oder als fixed markierte körper nicht integrieren —
            # diese sollten an ihren scripted/initial positionen bleiben.
            if body.scripted_orbit or getattr(body, 'fixed', False):
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

    def enable_epicycles(self, center):
        """epizykel-modus aktivieren mit wurzel in `center`.

        speichert den aktuellen eltern/orbit-zustand für alle körper und setzt dann
        jedes top-level körper (deren `is_moon_of` None ist) als child von `center`.
        der gespeicherte zustand wird so abgelegt, dass `disable_epicycles()`
        die ursprüngliche konfiguration wiederherstellen kann.
        """
        if center is None:
            return False

        # If already enabled with same center, do nothing
        if self._epicycle_enabled and self._epicycle_center is center:
            return True

        # If enabled with another center, restore first
        if self._epicycle_enabled:
            self.disable_epicycles()

        saved = {}
        for b in self.body:
            saved[b] = {
                'is_moon_of': getattr(b, 'is_moon_of', None),
                'semi_major_axis': getattr(b, 'semi_major_axis', None),
                'eccentricity': getattr(b, 'eccentricity', None),
                'scripted_orbit': getattr(b, 'scripted_orbit', False),
                'released': getattr(b, 'released', False),
                'theta': getattr(b, 'theta', 0.0),
                'arg_periapsis': getattr(b, 'arg_periapsis', 0.0),
                'position': b.position.copy() if hasattr(b, 'position') else None,
                'velocity': b.velocity.copy() if hasattr(b, 'velocity') else None,
            }

        # Apply epicycle reparenting
        for b in self.body:
            if b is center:
                # keep center stationary under scripted orbit
                b.is_moon_of = None
                b.semi_major_axis = 0.0
                b.eccentricity = 0.0
                b.scripted_orbit = True
                b.released = False
                continue

            orig_parent = saved[b]['is_moon_of']
            # Skip ships entirely (they should remain dynamic)
            if getattr(b, 'is_ship', False):
                continue

            # Only reparent bodies that were scripted_orbit originally (planetary
            # bodies defined with orbital elements). Do not change purely
            # dynamic bodies.
            if orig_parent is None and saved[b]['scripted_orbit']:
                # Berechne relatives r/v zum Zentrum und leite neue orbitale Elemente ab
                
                    rel_r = b.position - center.position
                    # If center has velocity attribute, use relative velocity, else assume 0
                    center_v = getattr(center, 'velocity', Vec2(0.0, 0.0))
                    rel_v = b.velocity - center_v if hasattr(b, 'velocity') else Vec2(0.0, 0.0)
                    mu = self.G * getattr(center, 'mass', 0.0)
                    elems = self._rv_to_orbital(rel_r, rel_v, mu)
                    if elems is not None:
                        a, e, theta_rel, arg_peri = elems
                        b.is_moon_of = center
                        b.semi_major_axis = float(max(0.0, a))
                        b.eccentricity = float(max(0.0, min(0.999999, e)))
                        b.theta = float(theta_rel)
                        b.arg_periapsis = float(arg_peri)
                        b.scripted_orbit = True
                        b.released = False
                    else:
                        # Fallback: setze kreisförmige Bahn mit dem aktuellen Abstand
                        try:
                            r = (b.position - center.position).magnitude()
                        except Exception:
                            r = float(getattr(b, 'semi_major_axis', 0.0) or 0.0)
                        b.is_moon_of = center
                        b.semi_major_axis = float(max(0.0, r))
                        b.eccentricity = 0.0
                        b.theta = math.atan2((b.position - center.position).y, (b.position - center.position).x)
                        b.arg_periapsis = 0.0
                        b.scripted_orbit = True
                        b.released = False
            else:
                # preserve moons' parent relationships and dynamic bodies; ensure
                # scripted bodies stay scripted if they were originally.
                if saved[b]['scripted_orbit']:
                    b.scripted_orbit = True

        self._epicycle_saved = saved
        self._epicycle_enabled = True
        self._epicycle_center = center
        return True

    def disable_epicycles(self):
        """gespeicherten zustand der körper wiederherstellen und epizykel-modus deaktivieren."""
        if not self._epicycle_enabled:
            return False

        for b in self.body:
            saved = self._epicycle_saved.get(b)
            if saved is None:
                continue
            b.is_moon_of = saved['is_moon_of']
            b.semi_major_axis = saved['semi_major_axis']
            b.eccentricity = saved['eccentricity']
            b.scripted_orbit = saved['scripted_orbit']
            b.released = saved['released']
            # restore angular state
            if 'theta' in saved:
                try:
                    b.theta = float(saved['theta'])
                except Exception:
                    pass
            if 'arg_periapsis' in saved:
                try:
                    b.arg_periapsis = float(saved['arg_periapsis'])
                except Exception:
                    pass

        self._epicycle_saved = {}
        self._epicycle_enabled = False
        self._epicycle_center = None
        return True

    def set_epicycle_center_by_name(self, name):
        """komfortfunktion: epizykel für den körper mit `name` aktivieren.

        wenn `name` None ist oder nicht gefunden wird, werden epizykel deaktiviert.
        """
        if name is None:
            return self.disable_epicycles()
        target = next((b for b in self.body if getattr(b, 'name', '').lower() == name.lower()), None)
        if target is None:
            return False
        return self.enable_epicycles(target)