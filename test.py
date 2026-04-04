import time
import math
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
from schiff import schiffcontrol

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
    dt = 9000  # zeitschritt in Sekunden (1 Schritt = 15 Minuten)
    running = True  # Hauptschleife der Simulation

    # Kamera initialisieren
    camera = Camera(screen, WIDTH, HEIGHT, sim_dt=dt)
    camera.follow(bodies[4])  # Erde folgen

    # Predictor initialisieren
    # num_points: Anzahl der Punkte (bestimmt die Reichweite)
    # distance_interval: Abstand zwischen Punkten in Metern (kleiner = genauer)
    predictor = Predictor(num_points=10000, dt=1000.0, recompute_every_update=True)  # 1M Meter pro Punkt
    # Diagnostic: force synchronous recompute on stale async snapshots
    predictor.force_sync_on_stale = True
    print("PREDICTOR DEBUG: force_sync_on_stale = True")

    # Schiff-Steuerung initialisieren
    ship = next((b for b in w.body if b.is_ship), None)
    ship_control = schiffcontrol(ship) if ship else None

    # OpenGL Renderer initialisieren
    renderer = Renderer(WIDTH, HEIGHT, enable_fxaa=True)
    renderer.debug_predictor = True
    print("=== Renderer initialisiert ===")


    def update(world, dt):
        """Aktualisiert die Simulation."""
        world.update_planets(dt)
        world.update_dynamics(dt)


    # Hauptschleife
    frame_count = 0
    while running:
        frame_dt = clock.tick(FPS) / 1000.0

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

                # Predictor length/precision controls: '+' / '-' adjust length, '9' / '0' adjust spacing
                ch = event.unicode
                if ch == '+' or event.key == pygame.K_KP_PLUS:
                    # double length (or initialize if None)
                    new_len = predictor.length * 2 if predictor.length is not None else predictor.num_points * predictor.precision * 2
                    predictor.set_length(new_len)
                    predictor.reset()
                    print(f"PREDICTOR: length set to {predictor.length}")
                elif ch == '-' or event.key == pygame.K_KP_MINUS:
                    cur = predictor.length if predictor.length is not None else predictor.num_points * predictor.precision
                    new_len = max(predictor.precision, cur / 2)
                    predictor.set_length(new_len)
                    predictor.reset()
                    print(f"PREDICTOR: length set to {predictor.length}")
                elif ch == '9':
                    # increase precision (finer = smaller spacing)
                    new_prec = max(1.0, predictor.precision / 2)
                    predictor.set_precision(new_prec)
                    predictor.reset()
                    print(f"PREDICTOR: precision set to {predictor.precision}")
                elif ch == '0':
                    new_prec = predictor.precision * 2
                    predictor.set_precision(new_prec)
                    predictor.reset()
                    print(f"PREDICTOR: precision set to {predictor.precision}")

            camera.handle_event(event)
        
        # Schiff-Steuerung
        keys = pygame.key.get_pressed()
        if ship_control:
            # rotation: smooth in real time
            ship_control.handle_rotation(keys, frame_dt)
            # thrust: apply a fixed delta-v once per real frame (independent of sim_dt)
            # capture velocity before thrust to report only thrust-caused change
            if ship is not None:
                old_v = ship.velocity.copy()
                ship_control.apply_thrust(keys)
                dv = ship.velocity - old_v
                if dv.x != 0.0 or dv.y != 0.0:
                    mag = math.hypot(dv.x, dv.y)
                    print(f"THRUST_DELTA: dx={dv.x:.6e}, dy={dv.y:.6e}, |dv|={mag:.6e}")
            else:
                ship_control.apply_thrust(keys)

        # Simulation aktualisieren (split into substeps for dynamics only)
        total_sim = camera.sim_dt
        MAX_SUBSTEP = 1000.0
        if total_sim <= MAX_SUBSTEP:
            update(w, total_sim)
        else:
            steps = int(math.ceil(total_sim / MAX_SUBSTEP))
            sub_dt = total_sim / steps
            for _ in range(steps):
                update(w, sub_dt)

        camera.update(total_sim)
        
        # Orbit-Prediction berechnen (für das Schiff oder einen Körper)
        points = []

        if predictor.num_points > 0:
            target = ship if ship else next((b for b in w.body if not b.fixed), None)

            if target:
                if hasattr(predictor, 'set_view_scale'):
                    predictor.set_view_scale(camera.scale)
                predictor.update(target, w)

        points = predictor.get_points()
        # Rendern

        
        renderer.render(w.body, camera, points, predictor=predictor)
        frame_count += 1

    pygame.quit()

if __name__ == "__main__":
    main()