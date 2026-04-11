import pygame
import math
from vec import Vec2


class schiffcontrol:
    def __init__(self, schiff):
        self.schiff = schiff
        self.rotation_speed = 3.0
        self.thrust_acc = 100.0

    def handle_rotation(self, keys, real_dt):
        """rotation mit echtem (wanduhr-)delta behandeln damit das drehen sich glatt anfühlt.

        real_dt: in echt verstrichene sekunden (frame_dt)
        """
        rotation_input = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
        if rotation_input:
            self.schiff.theta += rotation_input * self.rotation_speed * real_dt

    def apply_thrust(self, keys):
        """schub als festen delta-v pro aufruf anwenden (skaliert nicht mit sim_dt).

        dies stellt sicher, dass der vom spieler angewendete schub unabhängig
        von der gewählten simulationstimestep identisch ist. der controller
        sollte dies einmal pro echtem frame beim verarbeiten von eingaben aufrufen.
        """
        direction = Vec2(math.cos(self.schiff.theta), math.sin(self.schiff.theta))
        thrust_input = float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN])
        if thrust_input:
            delta_v = direction * (thrust_input * self.thrust_acc)
            self.schiff.velocity += delta_v

    def handle_input(self, keys, sim_dt, real_dt):
        """Abwärtskompatibler Helfer: rotation (real_dt) und schub (einmal pro frame) anwenden."""
        self.handle_rotation(keys, real_dt)
        self.apply_thrust(keys)
