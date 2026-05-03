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
from reference_frames import (
    BODY_CENTRED_BODY_DIRECTION,
    BODY_CENTRED_NON_ROTATING,
    PlottingFrameAdapter,
    ReferenceFrameSelector,
    resolve_plotting_camera_target_index,
)

def main():
    import os
    # VSync über Umgebungsvariable aktivieren
    os.environ['SDL_VIDEO_VSYNC'] = '1'

    # Starte Pygame mit OpenGL
    pygame.init()
    WIDTH, HEIGHT = 1920, 1000

    # OpenGL-Flag für pygame Display
    screen = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL, vsync=1)
    print(glGetString(GL_VENDOR), glGetString(GL_RENDERER), glGetString(GL_VERSION))
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
    earth = next((b for b in bodies if getattr(b, 'name', '').lower() in ('earth', 'erde')), None)
    ship = next((b for b in bodies if getattr(b, 'is_ship', False)), None)
    camera.follow(earth)

    # Predictor initialisieren
    # num_points: Anzahl der Punkte (bestimmt die Reichweite)
    # distance_interval: Abstand zwischen Punkten in Metern (kleiner = genauer)
    predictor = Predictor(num_points=10000, dt=1000.0, recompute_every_update=True)  # 1M Meter pro Punkt
    # diagnostik: synchrone neuberechnung bei veralteten asynchronen snapshots erzwingen
    predictor.force_sync_on_stale = False
    predictor.set_integrator_quality("accurate")
    print(f"PREDICTOR DEBUG: force_sync_on_stale = {predictor.force_sync_on_stale}")

    # Schiff-Steuerung initialisieren
    ship = next((b for b in w.body if b.is_ship), None)
    ship_control = schiffcontrol(ship) if ship else None

    # OpenGL Renderer initialisieren
    renderer = Renderer(WIDTH, HEIGHT, enable_fxaa=True)
    renderer.debug_predictor = False
    print("=== Renderer initialisiert ===")
    # predictor-darstellung bei gleichem zoom detaillierter machen
    renderer.prediction_sampling_tolerance_px = 2.5
    renderer.prediction_sampling_min_tolerance_px = 0.002
    renderer.prediction_sampling_max_points = 1000

    # zoom-empfindlichkeit verstärken (renderer nutzt camera.scale/reference zur zoom-berechnung)
    renderer.prediction_sampling_reference_scale = 5e-7

    # principia-ähnliche frame-pipeline:
    # selector (eingabe) -> adapter (factory/dispatch) -> renderer (projektion).
    celestial_indices = [i for i, b in enumerate(w.body) if not getattr(b, 'is_ship', False)]
    if not celestial_indices:
        celestial_indices = list(range(len(w.body)))

    ship_index = next((i for i, b in enumerate(w.body) if getattr(b, 'is_ship', False)), None)

    if earth is not None:
        reference_index = w.body.index(earth)
    else:
        reference_index = celestial_indices[0] if celestial_indices else 0
    reference_cursor = celestial_indices.index(reference_index) if reference_index in celestial_indices else 0

    frame_extension = BODY_CENTRED_NON_ROTATING
    target_overlay_enabled = False

    def choose_secondary(primary_index):
        primary = w.body[primary_index]
        parent = getattr(primary, 'is_moon_of', None)
        if parent is not None:
            for idx, candidate in enumerate(w.body):
                if candidate is parent and not getattr(candidate, 'is_ship', False):
                    return idx
        for idx in celestial_indices:
            if idx != primary_index:
                return idx
        return primary_index

    frame_adapter = PlottingFrameAdapter(renderer, w.body)

    def on_frame_change(frame_parameters, target_body_index, target_reference_index):
        frame_adapter.update_plotting_frame(
            frame_parameters,
            target_body_index=target_body_index,
            target_reference_index=target_reference_index,
        )

    frame_selector = ReferenceFrameSelector(on_frame_change)

    def apply_frame_selection():
        secondary_index = choose_secondary(reference_index)
        if frame_extension == BODY_CENTRED_BODY_DIRECTION:
            frame_selector.set_to_body_direction(reference_index, secondary_index)
            mode_text = (
                f"body-direction ({w.body[reference_index].name} -> "
                f"{w.body[secondary_index].name})"
            )
        else:
            frame_selector.set_to_body_non_rotating(reference_index)
            mode_text = f"body-centred non-rotating ({w.body[reference_index].name})"

        # predictor-physik-korrektur für translierte nicht-rotierende rahmen:
        # referenzkörper-beschleunigung nur in diesem modus subtrahieren.
        try:
            if hasattr(predictor, 'set_reference_body_index'):
                if frame_extension == BODY_CENTRED_NON_ROTATING:
                    predictor.set_reference_body_index(reference_index)
                else:
                    predictor.set_reference_body_index(None)
        except Exception:
            pass

        # kamera am schiff verankert halten damit frame/target-änderungen nicht springen
        # zum ausgewählten referenzkörper.
        if ship is not None:
            camera.follow(ship)
            camera_follow_name = ship.name
        else:
            active_params = frame_selector.frame_parameters()
            follow_index = resolve_plotting_camera_target_index(active_params, w.body)
            camera.follow(w.body[follow_index])
            camera_follow_name = w.body[follow_index].name

        if target_overlay_enabled and ship_index is not None:
            frame_selector.set_target_frame(ship_index, reference_index)
            overlay_text = f"ON ({w.body[ship_index].name} vs {w.body[reference_index].name})"
        else:
            overlay_text = "OFF"

        print(
            f"FRAME: {mode_text} | target_overlay={overlay_text} "
            f"| camera_follow={camera_follow_name}"
        )

    apply_frame_selection()


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

                # Taste E: epizykel-modus umschalten (zentriert auf kameraziel oder Erde)
                elif event.key == pygame.K_e:
                    center = camera.target
                    if center is None:
                        center = next((b for b in w.body if getattr(b, 'name', '').lower() in ('earth', 'erde')), None)
                    if center is None:
                        print("EPICYCLE: No center found (camera target or Earth).")
                    else:
                        if getattr(w, '_epicycle_enabled', False) and getattr(w, '_epicycle_center', None) is center:
                            w.disable_epicycles()
                            print("EPICYCLE: disabled")
                        else:
                            w.enable_epicycles(center)
                            print(f"EPICYCLE: enabled (center={center.name})")

                elif event.key == pygame.K_r and celestial_indices:
                    reference_cursor = (reference_cursor + 1) % len(celestial_indices)
                    reference_index = celestial_indices[reference_cursor]
                    apply_frame_selection()

                elif event.key == pygame.K_1:
                    frame_extension = BODY_CENTRED_NON_ROTATING
                    apply_frame_selection()

                elif event.key == pygame.K_2:
                    frame_extension = BODY_CENTRED_BODY_DIRECTION
                    apply_frame_selection()

                elif event.key == pygame.K_t:
                    if ship_index is None:
                        print("FRAME: no ship available for target overlay")
                    else:
                        target_overlay_enabled = not target_overlay_enabled
                        apply_frame_selection()

                # predictor länge/präzision steuerung: '+' / '-' länge anpassen, '9' / '0' abstand anpassen
                ch = event.unicode
                if ch == '+' or event.key == pygame.K_KP_PLUS:
                    # länge verdoppeln (oder initialisieren falls None)
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
                    # präzision erhöhen (feiner = kleinere abstände)
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
            # rotation: in echtzeit sanft
            ship_control.handle_rotation(keys, frame_dt)
            # schub: einmal pro echtem frame festen delta-v anwenden (unabhängig von sim_dt)
            # geschwindigkeit vor dem schub erfassen um nur schub-verursachte änderung zu melden
            if ship is not None:
                old_v = ship.velocity.copy()
                ship_control.apply_thrust(keys)
                dv = ship.velocity - old_v
                if dv.x != 0.0 or dv.y != 0.0:
                    mag = math.hypot(dv.x, dv.y)
                    print(f"THRUST_DELTA: dx={dv.x:.6e}, dy={dv.y:.6e}, |dv|={mag:.6e}")
            else:
                ship_control.apply_thrust(keys)

        # Simulation aktualisieren (nur für dynamik in unterschritte aufteilen)
        total_sim = camera.sim_dt
        MAX_SUBSTEP = 1000.0
        if total_sim <= MAX_SUBSTEP:
            update(w, total_sim)
        else:
            steps = int(math.ceil(total_sim / MAX_SUBSTEP))
            sub_dt = total_sim / steps
            for _ in range(steps):
                update(w, sub_dt)

        # kamera mit echtem frame-delta für interaktives panning aktualisieren
        camera.update(frame_dt)
        
        # orbit-prognose berechnen (für das Schiff oder einen Körper)
        points = []

        if predictor.num_points > 0:
            target = ship if ship else next((b for b in w.body if not b.fixed), None)

            if target:
                if hasattr(predictor, 'set_view_scale'):
                    predictor.set_view_scale(camera.scale)
                predictor.update(target, w)

        points = predictor.get_points()
        # Rendern

        
        renderer.render(w.body, camera, points, predictor=predictor, sim_time=w.time)
        frame_count += 1

    pygame.quit()

if __name__ == "__main__":
    main()