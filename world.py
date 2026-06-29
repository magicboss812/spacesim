import math
from vec import Vec2
from bodies import body
G = 6.6730831e-11

class world:

    def __init__(self, G):
        self.G = G
        self.body = []
        self.time = 0.0
        self.integrator_max_step = 30.0
        self.integrator_min_step = 0.01
        self.integrator_position_tolerance = 1.0
        self.integrator_velocity_tolerance = 0.001
        self.integrator_debug = False
        self.integrator_mode = "rkn4"  # "rkn4" | "verlet"
        self.integrator_last_substeps = 0
        self.integrator_last_rejections = 0
        self.integrator_last_min_step_forced = 0
        self.integrator_last_worst_pos_error = 0.0
        self.integrator_last_worst_vel_error = 0.0
        # Epicycle (Ptolemaic) mode state. When enabled, top-level bodies
        # (those with no parent) will be reparented to the chosen center
        # body so that the resulting motion produces epicycles relative
        # to that center.
        self._epicycle_enabled = False
        self._epicycle_center = None
        self._epicycle_saved = {}

    def update_planets(self, dt):
        for body in self.body:
            # Überspringe Schiffe komplett - sie haben keine orbit_position
            if body.is_ship:
                continue
            if not body.scripted_orbit:
                continue
            parent_pos = body.is_moon_of.position if body.is_moon_of else None
            mu = self.G * body.is_moon_of.mass if body.is_moon_of else None

            # Advance position; record epoch so position_at_time can extrapolate
            # correctly during the next update_dynamics call.
            body.position = body.orbit_position(dt, parent_pos, mu)
            body._kepler_ref_theta = body.theta
            body._kepler_ref_time = self.time

            # DANN prüfen ob Release nötig

        # Hinweis: Der Epizykel-Modus wird durch Umparenting der Top-Level-
        # Körper zum gewählten Zentrum via `enable_epicycles()` aktiviert; 
        # update_planets folgt einfach den aktuell gesetzten Elternbeziehungen.

    def _body_position_at_time(self, body, time_s):
        """
        Return the body's world position at a given simulation time.

        For now:
        - if the body has a time-aware orbit method later, use it here
        - otherwise fall back to current body.position

        This keeps the integrator ready for scripted moving planets at intermediate stages.
        """
        try:
            if hasattr(body, "position_at_time"):
                return body.position_at_time(time_s)
        except Exception:
            pass

        return body.position

    def acceleration_at(self, target_body, position, time_s=None):
        acc = Vec2(0.0, 0.0)

        if time_s is None:
            time_s = self.time

        for other in self.body:
            if other is target_body:
                continue

            try:
                other_pos = self._body_position_at_time(other, time_s)
            except Exception:
                other_pos = other.position

            delta = other_pos - position
            r2 = delta.magnitude_squared()
            if r2 < 1e-10:
                continue

            r = math.sqrt(r2)
            acc += delta * (self.G * other.mass / (r2 * r))

        return acc

    def _rkn4_step_body_state(self, body, p0, v0, t0, h):
        """
        One explicit RKN4-style step for r'' = a(r, t).

        p0: initial position
        v0: initial velocity
        t0: initial simulation time
        h: step size in seconds

        Returns:
            new_position, new_velocity
        """
        a1 = self.acceleration_at(body, p0, t0)

        p2 = p0 + v0 * (h * 0.5) + a1 * (h * h * 0.125)
        a2 = self.acceleration_at(body, p2, t0 + h * 0.5)

        p3 = p0 + v0 * (h * 0.5) + a2 * (h * h * 0.125)
        a3 = self.acceleration_at(body, p3, t0 + h * 0.5)

        p4 = p0 + v0 * h + a3 * (h * h * 0.5)
        a4 = self.acceleration_at(body, p4, t0 + h)

        new_p = p0 + v0 * h + (a1 + a2 + a3) * (h * h / 6.0)
        new_v = v0 + (a1 + 2.0 * a2 + 2.0 * a3 + a4) * (h / 6.0)

        return new_p, new_v

    def _verlet_step_body_state(self, body, p0, v0, t0, h):
        """Störmer-Verlet (KDK leapfrog) — 2nd-order symplectic RKN."""
        a0 = self.acceleration_at(body, p0, t0)
        p1 = p0 + v0 * h + a0 * (0.5 * h * h)
        a1 = self.acceleration_at(body, p1, t0 + h)
        v1 = v0 + (a0 + a1) * (0.5 * h)
        return p1, v1

    def _adaptive_rkn_step_body_state(
        self,
        body,
        p0,
        v0,
        t0,
        h,
        pos_tol=1.0,
        vel_tol=0.001,
    ):
        """
        Adaptive embedded RKN-style step by step-doubling.

        Compares:
        - one full step h
        - two half steps h/2

        Returns:
            accepted, new_position, new_velocity, pos_error, vel_error
        """
        step_fn = self._verlet_step_body_state if getattr(self, "integrator_mode", "rkn4") == "verlet" \
            else self._rkn4_step_body_state

        p_full, v_full = step_fn(body, p0, v0, t0, h)

        half = h * 0.5
        p_half, v_half = step_fn(body, p0, v0, t0, half)
        p_two, v_two = step_fn(body, p_half, v_half, t0 + half, half)

        pos_err = (p_two - p_full).magnitude()
        vel_err = (v_two - v_full).magnitude()

        accepted = pos_err <= pos_tol and vel_err <= vel_tol
        return accepted, p_two, v_two, pos_err, vel_err

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
    def update_dynamics(self, dt):
        self.integrator_last_substeps = 0
        self.integrator_last_rejections = 0
        self.integrator_last_min_step_forced = 0
        self.integrator_last_worst_pos_error = 0.0
        self.integrator_last_worst_vel_error = 0.0

        dynamic_bodies = [
            b for b in self.body
            if not getattr(b, "scripted_orbit", False) and not getattr(b, "fixed", False)
        ]

        total_dt = float(dt)
        if not dynamic_bodies:
            self.time += total_dt
            return

        direction = 1.0 if total_dt >= 0.0 else -1.0
        remaining = abs(total_dt)

        max_step = max(float(getattr(self, "integrator_max_step", 30.0)), 1e-9)
        min_step = max(float(getattr(self, "integrator_min_step", 0.01)), 1e-12)
        pos_tol = max(float(getattr(self, "integrator_position_tolerance", 1.0)), 1e-12)
        vel_tol = max(float(getattr(self, "integrator_velocity_tolerance", 0.001)), 1e-12)

        t = float(self.time)

        while remaining > 1e-12:
            h = min(max_step, remaining) * direction

            while True:
                saved_states = [
                    (b, b.position.copy(), b.velocity.copy())
                    for b in dynamic_bodies
                ]

                accepted_all = True
                worst_pos_err = 0.0
                worst_vel_err = 0.0
                new_states = []

                for b, p0, v0 in saved_states:
                    accepted, p_new, v_new, pos_err, vel_err = self._adaptive_rkn_step_body_state(
                        b,
                        p0,
                        v0,
                        t,
                        h,
                        pos_tol=pos_tol,
                        vel_tol=vel_tol,
                    )

                    worst_pos_err = max(worst_pos_err, pos_err)
                    worst_vel_err = max(worst_vel_err, vel_err)

                    if not accepted and abs(h) > min_step:
                        accepted_all = False
                        break

                    new_states.append((b, p_new, v_new))

                self.integrator_last_worst_pos_error = max(self.integrator_last_worst_pos_error, worst_pos_err)
                self.integrator_last_worst_vel_error = max(self.integrator_last_worst_vel_error, worst_vel_err)

                if accepted_all:
                    for b, p_new, v_new in new_states:
                        b.position = p_new
                        b.velocity = v_new

                    self.integrator_last_substeps += 1
                    t += h
                    remaining -= abs(h)
                    break

                for b, p_old, v_old in saved_states:
                    b.position = p_old
                    b.velocity = v_old

                self.integrator_last_rejections += 1
                h *= 0.5

                if abs(h) <= min_step:
                    step_fn = self._verlet_step_body_state if getattr(self, "integrator_mode", "rkn4") == "verlet" \
                        else self._rkn4_step_body_state
                    new_states = []
                    for b, p0, v0 in saved_states:
                        p_new, v_new = step_fn(b, p0, v0, t, h)
                        new_states.append((b, p_new, v_new))

                    for b, p_new, v_new in new_states:
                        b.position = p_new
                        b.velocity = v_new

                    self.integrator_last_substeps += 1
                    self.integrator_last_min_step_forced += 1
                    t += h
                    remaining -= abs(h)
                    break

        if getattr(self, "integrator_debug", False):
            print(
                "INTEGRATOR_DBG: "
                f"dt={total_dt:.6g} "
                f"substeps={self.integrator_last_substeps} "
                f"rejections={self.integrator_last_rejections} "
                f"forced={self.integrator_last_min_step_forced} "
                f"worst_pos={self.integrator_last_worst_pos_error:.6e} "
                f"worst_vel={self.integrator_last_worst_vel_error:.6e}"
            )

        self.time += total_dt

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

