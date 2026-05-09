import pygame
import math
from vec import Vec2


class schiffcontrol:
    def __init__(self, schiff):
        self.schiff = schiff
        self.rotation_speed = 3.0
        self.thrust_acc = 600.0

    def handle_rotation(self, keys, real_dt, frame=None, time_s=0.0):
        """rotation mit echtem (wanduhr-)delta behandeln damit das drehen sich glatt anfühlt.

        real_dt: in echt verstrichene sekunden (frame_dt)
        """
        rotation_input = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
        if rotation_input:
            self.schiff.theta += rotation_input * self.rotation_speed * real_dt

    def apply_thrust(self, keys, real_dt=1.0):
        """schub als beschleunigung pro echtem frame anwenden.

        dies stellt sicher, dass der vom spieler angewendete schub unabhängig
        von der gewählten simulationstimestep identisch ist. der controller
        sollte dies einmal pro echtem frame beim verarbeiten von eingaben aufrufen.
        """
        direction = Vec2(math.cos(self.schiff.theta), math.sin(self.schiff.theta))
        thrust_input = float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN])
        if thrust_input:
            delta_v = direction * (thrust_input * self.thrust_acc * float(real_dt))
            self.schiff.velocity += delta_v

    def _safe_normalized(self, v):
        mag = math.hypot(v.x, v.y)
        if mag <= 1e-12:
            return None
        return Vec2(v.x / mag, v.y / mag)

    def apply_directional_thrust(self, direction, amount, real_dt):
        if direction is None:
            return
        self.schiff.velocity += direction * (float(amount) * float(real_dt))

    def apply_prograde_thrust(self, reference_body, amount, real_dt):
        rel_v = self.schiff.velocity - reference_body.velocity
        direction = self._safe_normalized(rel_v)
        self.apply_directional_thrust(direction, amount, real_dt)

    def apply_retrograde_thrust(self, reference_body, amount, real_dt):
        rel_v = self.schiff.velocity - reference_body.velocity
        direction = self._safe_normalized(rel_v)
        if direction is not None:
            direction = direction * -1.0
        self.apply_directional_thrust(direction, amount, real_dt)

    def apply_radial_out_thrust(self, reference_body, amount, real_dt):
        rel_pos = self.schiff.position - reference_body.position
        direction = self._safe_normalized(rel_pos)
        self.apply_directional_thrust(direction, amount, real_dt)

    def apply_radial_in_thrust(self, reference_body, amount, real_dt):
        rel_pos = self.schiff.position - reference_body.position
        direction = self._safe_normalized(rel_pos)
        if direction is not None:
            direction = direction * -1.0
        self.apply_directional_thrust(direction, amount, real_dt)

    def handle_input(self, keys, sim_dt, real_dt, frame=None, time_s=0.0):
        """Abwärtskompatibler Helfer: rotation (real_dt) und schub (einmal pro frame) anwenden."""
        self.handle_rotation(keys, real_dt, frame=frame, time_s=time_s)
        self.apply_thrust(keys, real_dt)
