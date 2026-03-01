import pygame
import math
from vec import Vec2
class schiffcontrol:
    def __init__(self, schiff):
        self.schiff = schiff
        self.rotation_speed = 0.05 
        self.thrust_acc = 1000
    def handle_input(self, keys):
        if keys[pygame.K_LEFT]:
            self.schiff.theta -= self.rotation_speed
        if keys[pygame.K_RIGHT]:
            self.schiff.theta += self.rotation_speed
        
        if keys[pygame.K_UP]:
            direction = Vec2(math.cos(self.schiff.theta), math.sin(self.schiff.theta))
            self.schiff.acceleration += direction * self.thrust_acc
        if keys[pygame.K_DOWN]:
            direction = Vec2(math.cos(self.schiff.theta), math.sin(self.schiff.theta))
            self.schiff.acceleration -= direction * self.thrust_acc
