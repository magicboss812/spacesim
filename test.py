import math
import pygame
from pygame.locals import *
import moderngl
from loader import ConfigLoader, SystemLoader
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
    import time

    # Zentrale Konfiguration laden. ALLE spielbaren Parameter stehen in
    # config.json; der ConfigLoader verteilt sie auf world/camera/schiff/
    # predictor/renderer. Hier werden nur noch die Werte gelesen, die die
    # Hauptschleife selbst braucht (Fenster, Tastenschritte, Debug-Ausgaben).
    # SPACESIM_CONFIG kann eine alternative Konfigurationsdatei vorgeben
    # (z. B. für Messläufe), sonst wird spacesim/config.json verwendet.
    config = ConfigLoader(os.environ.get("SPACESIM_CONFIG") or None)
    config.load()
    config.apply_globals()

    verbose = bool(config.get('debug.print_loader_info', True))
    print_timings = bool(config.get('debug.print_frame_timings', True))

    # VSync über Umgebungsvariable aktivieren
    vsync_enabled = bool(config.get('window.vsync', True))
    os.environ['SDL_VIDEO_VSYNC'] = '1' if vsync_enabled else '0'
    max_frames = int(config.get('simulation.max_frames', 0) or 0)
    try:
        # Umgebungsvariable hat Vorrang vor der Konfiguration (Messläufe).
        env_max_frames = int(os.environ.get("SPACESIM_MAX_FRAMES", "0") or "0")
    except Exception:
        env_max_frames = 0
    if env_max_frames > 0:
        max_frames = env_max_frames
    max_frames = max(0, max_frames)

    # Starte Pygame mit OpenGL
    pygame.init()
    WIDTH = int(config.get('window.width', 800))
    HEIGHT = int(config.get('window.height', 800))

    # OpenGL-Flag für pygame Display; moderngl hängt sich an den von
    # pygame/SDL erstellten GL-context (ein wrapper, geteilt mit dem Renderer)
    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT), DOUBLEBUF | OPENGL, vsync=1 if vsync_enabled else 0
    )
    gl_ctx = moderngl.create_context()
    print(gl_ctx.info['GL_VENDOR'], gl_ctx.info['GL_RENDERER'], gl_ctx.info['GL_VERSION'])
    pygame.display.set_caption(str(config.get('window.caption', "Orbital Mechanics - OpenGL Renderer")))
    clock = pygame.time.Clock()
    FPS = int(config.get('window.fps', 60))

    # System laden
    loader = SystemLoader(config.get('simulation.system_file', "solar_system.json"))
    bodies = loader.load()

    # Debug: Geladene Körper anzeigen
    if verbose:
        print("=== Geladene Körper (nach Loader) ===")
        for b in bodies:
            print(f"  {b.name}: pos={b.position}, vel={b.velocity}, is_ship={b.is_ship}, fixed={b.fixed}")

    # Körper in die World-Class einfügen (world.py)
    w = world(float(config.get('physics.gravitational_constant', 6.6730831e-11)))
    w.body = bodies
    config.apply_to_world(w)

    running = True  # Hauptschleife der Simulation

    # Kamera initialisieren (sim_dt, Zoom und Schwenkrate kommen aus config.json)
    camera = Camera(screen, WIDTH, HEIGHT)
    config.apply_to_camera(camera)
    focus_name = str(config.get('simulation.focus_body', "Erde")).strip().lower()
    focus_aliases = {focus_name, 'earth', 'erde'}
    earth = next((b for b in bodies if getattr(b, 'name', '').lower() in focus_aliases), None)
    ship = next((b for b in bodies if getattr(b, 'is_ship', False)), None)
    camera.follow(earth)

    # Predictor initialisieren
    # num_points: Anzahl der Punkte (bestimmt die Reichweite)
    # precision: Abstand zwischen Punkten in Metern (kleiner = feinere Linie)
    predictor_kwargs = config.predictor_kwargs()
    env_async = os.environ.get("SPACESIM_PREDICTOR_ASYNC")
    if env_async is not None:
        # Umgebungsvariable überschreibt die Konfiguration (Messläufe).
        predictor_kwargs['async_compute'] = env_async.strip().lower() not in ("0", "false", "no", "off")
    predictor = Predictor(recompute_every_update=True, **predictor_kwargs)
    # Übrige Predictor-Parameter (Qualität, Toleranzen, Apsis-Marker) aus config.json
    config.apply_to_predictor(predictor)
    # Look-ahead horizon (length) is the cost knob; point spacing (precision) is
    # cosmetic. Pin the horizon from startup so changing spacing ('9'/'0') no
    # longer moves the horizon (and thus no longer changes compute cost).
    # Default = num_points * base precision, so initial output is unchanged.
    predictor.set_length(predictor.num_points * predictor.precision)
    predictor_enabled = bool(config.get('predictor.enabled', True))
    if not predictor_enabled:
        predictor.reset()
        predictor.set_num_points(0)
    # Tastenschritte für '+'/'-' (Reichweite) und '9'/'0' (Punktabstand)
    length_step = max(float(config.get('predictor.length_step_factor', 2.0)), 1.0 + 1e-9)
    precision_step = max(float(config.get('predictor.precision_step_factor', 2.0)), 1.0 + 1e-9)
    predictor_toggle_points = int(config.get('predictor.toggle_num_points', 30))
    predictor_min_precision = float(config.get('predictor.min_precision', 1.0))
    if verbose:
        print(f"PREDICTOR DEBUG: async_compute = {predictor.async_compute}")
        print(f"PREDICTOR DEBUG: force_sync_on_stale = {predictor.force_sync_on_stale}")

    # Schiff-Steuerung initialisieren (Drehrate/Schub aus config.json)
    ship = next((b for b in w.body if b.is_ship), None)
    ship_control = schiffcontrol(ship) if ship else None
    config.apply_to_ship_control(ship_control)

    # OpenGL Renderer initialisieren (moderngl, geteilter context)
    renderer = Renderer(
        WIDTH, HEIGHT,
        enable_fxaa=bool(config.get('window.enable_fxaa', True)),
        ctx=gl_ctx,
    )
    # Darstellungsparameter (Linienqualität, Marker, Spuren, HUD) aus config.json
    config.apply_to_renderer(renderer)
    if verbose:
        print("=== Renderer initialisiert ===")
        print(f"=== Konfiguration: {config.filepath.name} ===")
        if config.unknown_keys:
            print(f"CONFIG: unbekannte Schlüssel ignoriert: {', '.join(sorted(set(config.unknown_keys)))}")

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
        world.update_dynamics(dt)
        world.update_planets(dt)


    # Größter Physik-Teilschritt (aus config.json): große sim_dt werden für die
    # Dynamik in mehrere Stücke zerlegt, damit der Integrator stabil bleibt.
    MAX_SUBSTEP = max(float(config.get('simulation.max_substep_seconds', 1000.0)), 1e-6)

    # Hauptschleife
    frame_count = 0
    while running:
        frame_dt = clock.tick(FPS) / 1000.0
        loop_t0 = time.perf_counter()

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
                        predictor.set_num_points(predictor_toggle_points)

                # Taste E: epizykel-modus umschalten (zentriert auf kameraziel oder Fokuskörper)
                elif event.key == pygame.K_e:
                    center = camera.target
                    if center is None:
                        center = next((b for b in w.body if getattr(b, 'name', '').lower() in focus_aliases), None)
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

                # I/K/J/L: orientierungs-snap (rastender autopilot) umschalten.
                # Tippen rastet ein, erneutes Tippen löst; render() hält die Nase
                # smooth an den gezeichneten orbital-vektoren im aktiven Frame.
                elif event.key in (pygame.K_i, pygame.K_k, pygame.K_j, pygame.K_l):
                    if ship_control is not None:
                        snap_mode = {
                            pygame.K_i: 'prograde',
                            pygame.K_k: 'retrograde',
                            pygame.K_j: 'normal_in',
                            pygame.K_l: 'antinormal_out',
                        }[event.key]
                        ship_control.toggle_snap(snap_mode)
                        print(f"SNAP: {ship_control.snap_mode or 'off'}")

                # predictor-steuerung (zwei entkoppelte regler):
                #   '+' / '-' -> look-ahead HORIZONT (predictor.length). Das ist
                #               der kosten-regler: kosten ~ integrierter bogen ~ horizont.
                #   '9' / '0' -> punkt-ABSTAND (predictor.precision). Rein kosmetisch:
                #               mehr/weniger gezeichnete punkte im festen horizont,
                #               gleiche rechenzeit und gleiche genauigkeit.
                ch = event.unicode
                if ch == '+' or event.key == pygame.K_KP_PLUS:
                    # länge verlängern (schrittfaktor aus config.json)
                    base_len = predictor.length if predictor.length is not None else predictor.num_points * predictor.precision
                    predictor.set_length(base_len * length_step)
                    predictor.reset()
                    print(f"PREDICTOR: length set to {predictor.length}")
                elif ch == '-' or event.key == pygame.K_KP_MINUS:
                    cur = predictor.length if predictor.length is not None else predictor.num_points * predictor.precision
                    new_len = max(predictor.precision, cur / length_step)
                    predictor.set_length(new_len)
                    predictor.reset()
                    print(f"PREDICTOR: length set to {predictor.length}")
                elif ch == '9':
                    # präzision erhöhen (feiner = kleinere abstände)
                    new_prec = max(predictor_min_precision, predictor.precision / precision_step)
                    predictor.set_precision(new_prec)
                    predictor.reset()
                    print(f"PREDICTOR: precision set to {predictor.precision}")
                elif ch == '0':
                    new_prec = predictor.precision * precision_step
                    predictor.set_precision(new_prec)
                    predictor.reset()
                    print(f"PREDICTOR: precision set to {predictor.precision}")

            camera.handle_event(event)
        
        # Schiff-Steuerung
        keys = pygame.key.get_pressed()
        reference_body = w.body[reference_index] if reference_index is not None else None
        if ship_control:
            ship_control.last_thrust_direction = None
            if ship is not None:
                setattr(ship, "last_thrust_direction", None)
            # rotation: in echtzeit sanft
            ship_control.handle_rotation(keys, frame_dt)
            # schub: einmal pro echtem frame festen delta-v anwenden (unabhängig von sim_dt)
            ship_control.apply_thrust(keys, frame_dt)

        # Simulation aktualisieren (nur für dynamik in unterschritte aufteilen)
        total_sim = camera.sim_dt
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

        # Rendern. Der Orientierungs-snap wird INNERHALB von render() angewendet,
        # unmittelbar bevor der Schiffspfeil gezeichnet wird, mit demselben Frame
        # und derselben Frame-Zeit wie die gezeichneten prograde/normal-Vektoren
        # — so ist die Nase exakt an diese Vektoren gebunden. ship_control und
        # frame_dt werden dafür durchgereicht.
        renderer.render(
            w.body, camera, points, predictor=predictor, sim_time=w.time,
            reference_body=reference_body, ship_control=ship_control, real_dt=frame_dt,
        )

        # Per-frame timing debug line. Splits the frame into predictor line
        # calculation vs. drawing, and the render pipeline into CPU calculation
        # vs. present-on-screen (swap/flip; includes the VSync wait).
        if print_timings:
            rt = getattr(renderer, 'last_frame_timings', {}) or {}
            ps = getattr(renderer, '_last_prediction_render_stats', {}) or {}
            rend_total = float(rt.get('frame_ms', 0.0))
            rend_draw = float(rt.get('swap_or_present_ms', 0.0))
            rend_calc = rend_total - rend_draw
            pred_calc = float(getattr(predictor, 'last_compute_ms', 0.0))
            pred_draw = float(ps.get('prepare_ms', 0.0)) + float(ps.get('draw_ms', 0.0))
            frame_ms = (time.perf_counter() - loop_t0) * 1000.0
            print(
                f"TIMING: pred_calc={pred_calc:.1f}ms pred_draw={pred_draw:.1f}ms "
                f"rend_calc={rend_calc:.1f}ms rend_draw={rend_draw:.1f}ms "
                f"frame={frame_ms:.1f}ms",
                flush=True,
            )

        frame_count += 1
        if max_frames > 0 and frame_count >= max_frames:
            running = False

    pygame.quit()

if __name__ == "__main__":
    main()
