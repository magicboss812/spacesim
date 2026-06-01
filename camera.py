import pygame
from vec import Vec2
import math

class Camera:
    """Verwaltet die Ansicht der Simulation: Position, Zoom und Koordinatentransformation."""
    
    def __init__(self, screen, width, height, sim_dt=9000.0):
        self.screen = screen  # Wird für eventuelle Direktzugriffe behalten
        self.width = width
        self.height = height

        # Simulation time step (seconds per simulation update)
        self.sim_dt = float(sim_dt)
        self.min_sim_dt = 1.0
        self.max_sim_dt = 1e12
        self.sim_dt_factor = 1.5
        
        # Kamera-Position (Weltkoordinaten des Bildzentrums)
        self.position = Vec2(0.0, 0.0)
        
        # Zoom: Pixel pro Meter (höher = mehr vergrößert)
        self.scale = 1e-6  # Standard: 1 Pixel = 1.000.000 Meter
        
        # Verfolgtes Objekt (None = freie Kamera)
        self.target = None
        
        # Bewegungsgeschwindigkeit für freie Kamera
        self.move_speed = 3 # Meter pro Sekunde
        
        # Zoom-Grenzen
        self.min_scale = 1e-30
        self.max_scale = 1e+10
    
    def world_to_screen(self, world_pos):
        rel = world_pos - self.position
        
        # Float-Berechnung
        screen_x = self.width / 2 + rel.x * self.scale
        screen_y = self.height / 2 - rel.y * self.scale
        
        # PRÜFUNG vor Rückgabe
        if not (math.isfinite(screen_x) and math.isfinite(screen_y)):
            print(f"WARNING: Invalid screen coords: {screen_x}, {screen_y}")
            return (self.width / 2.0, self.height / 2.0)
        
        # Subpixel-Präzision behalten, damit Bewegung nicht stufig wirkt.
        return (screen_x, screen_y)
    def screen_to_world(self, screen_pos):
        """Wandelt Bildschirmkoordinaten in Weltkoordinaten um."""
        screen_x, screen_y = screen_pos
        world_x = (screen_x - self.width / 2) / self.scale + self.position.x
        world_y = -(screen_y - self.height / 2) / self.scale + self.position.y
        return Vec2(world_x, world_y)
    
    def follow(self, target_body):
        """Setzt ein Objekt zur Verfolgung."""
        self.target = target_body
    
    def unfollow(self):
        """Beendet die Objektverfolgung."""
        self.target = None
    
    def update(self, dt):
        """Aktualisiert die Kameraposition (im Loop aufrufen)."""
        if self.target is not None:
            # Kamera folgt dem Ziel
            self.position = self.target.position.copy()
        else:
            # Freie Kamerasteuerung mit Tasten
            keys = pygame.key.get_pressed()
            # gewünschte bildschirm-geschwindigkeit (`move_speed`, interpretiert
            # als pixel pro sekunde) in welt-meter umrechnen unter verwendung der
            # aktuellen `scale` (px/m). Mit `dt` (sekunden) multiplizieren damit
            # bewegung framerate-unabhängig ist. sichere untergrenze für `scale`
            # verwenden um riesige sprung oder division durch null zu vermeiden.
            scale_safe = max(self.scale, 1e-30)
            move = (self.move_speed / scale_safe) * float(dt)

            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.position.x -= move
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.position.x += move
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                self.position.y -= move
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                self.position.y += move
        
        # Zoom mit Mausrad (wird in handle_event verarbeitet)
    
    def handle_event(self, event):
        """Verarbeitet Eingabeereignisse (Zoom, Klicks)."""
        if event.type == pygame.MOUSEWHEEL:
            # Mausrad für Zoom
            zoom_factor = 1.5 if event.y > 0 else 1 / 1.5
            self.scale *= zoom_factor
            self.scale = max(self.min_scale, min(self.max_scale, self.scale))
        
        
        elif event.type == pygame.KEYDOWN:
            # Verfolgung umschalten
            if event.key == pygame.K_f:
                if self.target is not None:
                    self.unfollow()
            # simulation timestep steuerung (PageUp/PageDown)
            if event.key == pygame.K_PAGEUP:
                self.sim_dt *= self.sim_dt_factor
                self.sim_dt = min(self.sim_dt, self.max_sim_dt)
            elif event.key == pygame.K_PAGEDOWN:
                self.sim_dt /= self.sim_dt_factor
                self.sim_dt = max(self.sim_dt, self.min_sim_dt)
