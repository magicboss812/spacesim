import pygame
import math
from vec import Vec2


# Latched orientation-hold modes, keyed by the field returned from
# reference_frames.apparent_orbital_directions(). The renderer computes the
# directions and ties the ship nose to them (see Renderer._apply_orientation_snap).
SNAP_MODES = ("prograde", "retrograde", "normal_in", "antinormal_out")


class schiffcontrol:
    def __init__(self, schiff):
        self.schiff = schiff
        self.rotation_speed = 3.0
        self.thrust_acc = 600.0
        self.last_thrust_direction = None
        # Active orientation autopilot (one of SNAP_MODES) or None. Latched:
        # toggled on/off by a key tap; the renderer computes the target heading
        # each frame (Renderer._apply_orientation_snap) and calls
        # orient_towards_angle() to hold the nose smoothly on the drawn vector.
        self.snap_mode = None
        # Once the nose reaches the snapped vector, pin it exactly (see
        # orient_towards_angle) so velocity changes can't make it lag/flip.
        self._snap_locked = False
        if self.schiff is not None:
            setattr(self.schiff, "last_thrust_direction", None)

    def handle_rotation(self, keys, real_dt, frame=None, time_s=0.0):
        """rotation mit echtem (wanduhr-)delta behandeln damit das drehen sich glatt anfühlt.

        real_dt: in echt verstrichene sekunden (frame_dt)
        """
        rotation_input = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
        if rotation_input:
            self.schiff.theta += rotation_input * self.rotation_speed * real_dt

    def apply_thrust(self, keys, real_dt=1.0):
        """Manual nose thrust as acceleration per real frame.

        dies stellt sicher, dass der vom spieler angewendete schub unabhängig
        von der gewählten simulationstimestep identisch ist. der controller
        sollte dies einmal pro echtem frame beim verarbeiten von eingaben aufrufen.
        """
        direction = Vec2(math.cos(self.schiff.theta), -math.sin(self.schiff.theta))
        thrust_input = float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN])
        if thrust_input:
            thrust_direction = direction if thrust_input > 0.0 else direction * -1.0
            self.last_thrust_direction = thrust_direction.copy()
            setattr(self.schiff, "last_thrust_direction", thrust_direction.copy())
            delta_v = thrust_direction * (abs(thrust_input) * self.thrust_acc * float(real_dt))
            self.schiff.velocity += delta_v

    def _safe_normalized(self, v):
        mag = math.hypot(v.x, v.y)
        if mag <= 1e-12:
            return None
        return Vec2(v.x / mag, v.y / mag)

    def _body_velocity(self, body):
        if body is None:
            return Vec2(0.0, 0.0)
        try:
            return body.velocity.copy()
        except Exception:
            return Vec2(
                float(getattr(getattr(body, "velocity", None), "x", 0.0) or 0.0),
                float(getattr(getattr(body, "velocity", None), "y", 0.0) or 0.0),
            )

    def _body_position(self, body):
        if body is None:
            return Vec2(0.0, 0.0)
        try:
            return body.position.copy()
        except Exception:
            return Vec2(
                float(getattr(getattr(body, "position", None), "x", 0.0) or 0.0),
                float(getattr(getattr(body, "position", None), "y", 0.0) or 0.0),
            )

    def relative_velocity_to(self, reference_body=None):
        ref_v = self._body_velocity(reference_body)
        return self.schiff.velocity - ref_v

    def relative_position_to(self, reference_body=None):
        ref_p = self._body_position(reference_body)
        return self.schiff.position - ref_p

    def apply_directional_thrust(self, direction, amount, real_dt):
        if direction is None:
            return
        self.last_thrust_direction = direction.copy()
        setattr(self.schiff, "last_thrust_direction", direction.copy())
        self.schiff.velocity += direction * (float(amount) * float(real_dt))

    def apply_prograde_thrust(self, reference_body, amount, real_dt):
        rel_v = self.relative_velocity_to(reference_body)
        direction = self._safe_normalized(rel_v)
        self.apply_directional_thrust(direction, amount, real_dt)

    def apply_retrograde_thrust(self, reference_body, amount, real_dt):
        rel_v = self.relative_velocity_to(reference_body)
        direction = self._safe_normalized(rel_v)
        if direction is not None:
            direction = direction * -1.0
        self.apply_directional_thrust(direction, amount, real_dt)

    def apply_radial_out_thrust(self, reference_body, amount, real_dt):
        rel_pos = self.relative_position_to(reference_body)
        direction = self._safe_normalized(rel_pos)
        self.apply_directional_thrust(direction, amount, real_dt)

    def apply_radial_in_thrust(self, reference_body, amount, real_dt):
        rel_pos = self.relative_position_to(reference_body)
        direction = self._safe_normalized(rel_pos)
        if direction is not None:
            direction = direction * -1.0
        self.apply_directional_thrust(direction, amount, real_dt)

    def toggle_snap(self, mode):
        """Latch/unlatch an orientation-hold. Tapping the active mode clears it."""
        if mode not in SNAP_MODES:
            return
        self.snap_mode = None if self.snap_mode == mode else mode
        # Re-acquire smoothly whenever the latched mode changes.
        self._snap_locked = False

    def clear_snap(self):
        self.snap_mode = None
        self._snap_locked = False

    def orient_towards_angle(self, target_theta, real_dt):
        """Hold the ship nose on a world-space heading supplied by the renderer.

        The heading is computed so the *drawn* arrow lands exactly on the drawn
        orbital vector (see ``Renderer._apply_orientation_snap``). Two phases:

        - **Acquire** (just after a tap): rotate smoothly toward the target at
          ``rotation_speed`` (shortest path), rate-limited by the real
          (wall-clock) delta so the turn feels consistent regardless of sim_dt.
        - **Locked** (once the target is reached): pin ``theta`` directly onto
          the target every frame, so the nose stays glued to the vector even
          when a velocity change swings the vector faster than ``rotation_speed``
          could follow. Without this the nose lagged (and, near a sign flip,
          appeared to rotate the opposite way) during speed changes.

        Only world-space ``theta`` is stored, so physics stays absolute.
        """
        target = float(target_theta)
        delta = (target - self.schiff.theta + math.pi) % (2.0 * math.pi) - math.pi
        if getattr(self, "_snap_locked", False):
            self.schiff.theta = target
            return
        step = self.rotation_speed * float(real_dt)
        if abs(delta) <= step or step <= 0.0:
            self.schiff.theta = target
            self._snap_locked = True
        else:
            self.schiff.theta += math.copysign(step, delta)

    def orient_towards(self, direction_world, real_dt):
        """Smoothly rotate toward a world-space direction vector (see above)."""
        if direction_world is None:
            return
        self.orient_towards_angle(
            math.atan2(float(direction_world.y), float(direction_world.x)), real_dt
        )

