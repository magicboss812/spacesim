import pygame
import math
from vec import Vec2


class schiffcontrol:
    def __init__(self, schiff):
        self.schiff = schiff
        self.rotation_speed = 3.0
        self.thrust_acc = 100.0

    def handle_rotation(self, keys, real_dt):
        """Handle rotation using real (wall-clock) delta so turning feels smooth.

        real_dt: seconds elapsed in real time (frame_dt)
        """
        rotation_input = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
        if rotation_input:
            self.schiff.theta += rotation_input * self.rotation_speed * real_dt

    def apply_thrust(self, keys):
        """Apply thrust as a fixed delta-v per call (does not scale with sim_dt).

        This ensures the player's applied thrust is identical regardless of
        the simulation timestep chosen. The controller should call this once
        per real-frame when handling input.
        """
        direction = Vec2(math.cos(self.schiff.theta), math.sin(self.schiff.theta))
        thrust_input = float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN])
        if thrust_input:
            delta_v = direction * (thrust_input * self.thrust_acc)
            self.schiff.velocity += delta_v

    def handle_input(self, keys, sim_dt, real_dt):
        """Backward-compatible helper: apply rotation (real_dt) and thrust (once per frame)."""
        self.handle_rotation(keys, real_dt)
        self.apply_thrust(keys)
