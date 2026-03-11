import time
import pygame
from pygame.locals import *
from OpenGL.GL import *
from loader import SystemLoader
from vec import Vec2, G, vec
from bodies import body
from world import world
from camera import Camera
from rendering import Renderer
from predictor import Predictor

def main():
    import os
    # VSync über Umgebungsvariable aktivieren
    os.environ['SDL_VIDEO_VSYNC'] = '1'

    # Starte Pygame mit OpenGL
    pygame.init()
    WIDTH, HEIGHT = 1920, 1000

    # OpenGL-Flag für pygame Display
    screen = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL, vsync=1)
    pygame.display.set_caption("Orbital Mechanics - OpenGL Renderer")
    clock = pygame.time.Clock()
    FPS = 60

    # System laden
    loader = SystemLoader("solar_system.json")
    bodies = loader.load()

    # Debug: Geladene Körper anzeigen
    print("=== Geladene Körper (nach Loader) ===")
    for b in bodies:
        print(f"  {b.name}: pos={b.position}, vel={b.velocity}, is_ship={b.is_ship}, fixed={b.fixed}")

    # Körper in die World-Class einfügen (world.py)
    w = world(G)
    w.body = bodies

    # Parameter für die Simulation
    dt = 900  # zeitschritt in Sekunden (1 Schritt = 15 Minuten)
    running = True  # Hauptschleife der Simulation

    # Kamera initialisieren
    camera = Camera(screen, WIDTH, HEIGHT)
    camera.follow(bodies[4])  # Erde folgen

    # Predictor initialisieren
    # num_points: Anzahl der Punkte (bestimmt die Reichweite)
    # distance_interval: Abstand zwischen Punkten in Metern (kleiner = genauer)
    predictor = Predictor(num_points=1000, dt=10000.0)  # 1M Meter pro Punkt

    # Schiff-Steuerung initialisieren
    # OpenGL Renderer initialisieren
    renderer = Renderer(WIDTH, HEIGHT, enable_fxaa=True)
    print("=== Renderer initialisiert ===")


    def update(world, dt):
        """Aktualisiert die Simulation."""
        world.update_planets(dt)
        world.update_dynamics(dt)


    # Hauptschleife
    frame_count = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # Taste P für Predictive Orbit umschalten
                elif event.key == pygame.K_p:
                    if predictor.num_points > 0:
                        predictor.reset()
                    else:
                        predictor.set_num_points(30)
            camera.handle_event(event)
        
        # Schiff-Steuerung
        
        
        # Simulation aktualisieren
        w.calculate_forces()
        update(w, dt)
        camera.update(dt)
        
        # Orbit-Prediction berechnen (für das Schiff oder einen Körper)
        points = []

        if predictor.num_points > 0:
            ship = None
            for b in w.body:
                if b.is_ship:
                    ship = b
                    break

            if ship:
                predictor.update(ship, w)
                points = predictor.get_points()
            else:
                for b in w.body:
                    if not b.fixed:
                        predictor.update(b, w)
                        points = predictor.get_points()
                        break # Ergebnisse der Vorhersage abrufen
        points = predictor.get_points()
        if frame_count % 60 == 0:
            print(f"Vorhersagepunkte: {len(points)}")
            if len(points) > 0:
                print(f"DEBUG: First point: {points[0]}, Last point: {points[-1]}")
                print(f"DEBUG: Camera position: {camera.position}, scale: {camera.scale}")
        # Rendern

        
        renderer.render(w.body, camera, points)
        frame_count += 1
        
        clock.tick(FPS)  # Vorhersageprozess beenden
    pygame.quit()

if __name__ == "__main__":
    main()