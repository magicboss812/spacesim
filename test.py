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
from devui import DevContext, ImguiLayer
from ui import UIContext, UIRoot, UIState
from ui.hud import Hud
from reference_frames import (
    BODY_CENTRED_BODY_DIRECTION,
    BODY_CENTRED_NON_ROTATING,
    PlottingFrameAdapter,
    ReferenceFrameSelector,
)


def predictor_horizon_lengths(base_length, manual_mult, warp_mult,
                              max_points, base_spacing):
    """(gezeichnete laenge, gerechnete laenge) fuer den vorhersage-horizont.

    Modulebene und rein, damit `tests/warp_predictor_test.py` §23 sie messen
    kann statt die regel nachzubauen. Die begruendung fuer den deckel steht
    bei `apply_predictor_horizon()`.
    """
    drawn = float(base_length) * float(manual_mult)
    budget_length = float(max_points) * float(base_spacing)
    warp_mult = float(warp_mult)
    if drawn > 0.0:
        warp_mult = min(warp_mult, max(1.0, budget_length / drawn))
    return drawn, drawn * warp_mult


def horizon_targets(base_length, manual_mult, warp_mult, max_points,
                    base_spacing, *, grabbing=False, ceiling_mult=None):
    """Wie `predictor_horizon_lengths`, aber mit dem slider-griff.

    Solange der spieler den horizont-regler HAELT (`grabbing`), wird die
    GERECHNETE laenge so bestimmt, als staende der manuelle faktor auf
    `ceiling_mult` -- sie aendert sich dann nicht, waehrend der knauf
    wandert, also greift die 1e-9-schranke in `apply_predictor_horizon`
    genau einmal und der einzige aufruf pro frame ist das O(1)
    `set_display_length`. Die GEZEICHNETE laenge folgt weiter dem echten
    faktor. Siehe plans/predictor_horizon_slider_design.md.
    """
    drawn, wanted = predictor_horizon_lengths(
        base_length, manual_mult, warp_mult, max_points, base_spacing,
    )
    if grabbing and ceiling_mult is not None:
        _, wanted = predictor_horizon_lengths(
            base_length, float(ceiling_mult), warp_mult, max_points, base_spacing,
        )
    return drawn, wanted


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
    # Zaehlerstand fuer pred_hz (erneuerungen der vorhersagelinie je sekunde,
    # ueber ein halbsekunden-fenster gemittelt -- je bild gezaehlt waere es
    # nur 0 oder 1).
    timing_hz_t0 = None
    timing_hz_swaps = 0
    timing_pred_hz = 0.0

    # VSync über Umgebungsvariable aktivieren
    vsync_enabled = bool(config.get('window.vsync', True))
    os.environ['SDL_VIDEO_VSYNC'] = '1' if vsync_enabled else '0'

    # Windows-DPI-Awareness VOR pygame.init() setzen. python.exe ist ohne
    # Manifest standardmäßig DPI-unaware; ohne diesen Hint skaliert der DWM
    # das fertige (bereits scharf gerenderte) Fenster per Bitmap-Stretch auf
    # die physische Auflösung hoch, sobald die Windows-Skalierung > 100% ist
    # -> das ganze Fenster inkl. HUD-Text wirkt unscharf. SDL_WINDOWS_DPI_AWARENESS
    # ist der von SDL2 unterstützte Hint dafür und muss vor SDL_Init/pygame.init
    # gesetzt sein.
    if os.name == 'nt':
        os.environ.setdefault('SDL_WINDOWS_DPI_AWARENESS', 'permonitorv2')
    max_frames = int(config.get('simulation.max_frames', 0) or 0)
    try:
        # Umgebungsvariable hat Vorrang vor der Konfiguration (Messläufe).
        env_max_frames = int(os.environ.get("SPACESIM_MAX_FRAMES", "0") or "0")
    except Exception:
        env_max_frames = 0
    if env_max_frames > 0:
        max_frames = env_max_frames
    max_frames = max(0, max_frames)

    # Ab welcher raffung (sim-sekunden je echtsekunde) gilt der lauf als
    # "gerafft": darueber ist der schub gesperrt und die vorhersage wird
    # gehalten statt neu gerechnet. Standard ist genau die unterste
    # zeitraffer-stufe des HUDs (60 sim-s/s), also "echtzeit" in diesem spiel.
    # Muss VOR dem HUD-aufbau stehen -- der schubregler zeigt die sperre an.
    REALTIME_WARP_MAX = float(config.get('simulation.realtime_warp_max', 60.0))
    # Wie viele bahn-zeitskalen ein frame hoechstens vorruecken darf.
    # Muss VOR dem Hud-aufbau stehen -- das HUD blendet damit stufen ab.
    WARP_TIMESCALE_DIVISOR = max(
        float(config.get('simulation.warp_timescale_divisor', 3.0)), 1e-6)

    # Starte Pygame mit OpenGL.
    #
    # NUR display und font -- NICHT pygame.init(). pygame.init() faehrt JEDES
    # untermodul hoch, auch mixer und joystick, und beide zaehlen dabei die
    # geraete des rechners auf. Auf diesem system kostet das gemessen
    # 25.2 s (mixer) + 20.1 s (joystick) = 45.3 s, in denen das fenster noch
    # gar nicht existiert -- der start wirkt schlicht wie ein absturz.
    # Die dauer haengt an audio-/HID-treibern, nicht am spiel: sie kann sich
    # jederzeit wieder aendern. Deshalb wird hier gar nicht erst geraten,
    # sondern nur initialisiert, was das spiel wirklich benutzt.
    # Verwendet werden ausschliesslich display, event, font, image, key,
    # mouse und time; von denen brauchen nur display und font ein init.
    pygame.display.init()
    pygame.font.init()
    WIDTH = int(config.get('window.width', 800))
    HEIGHT = int(config.get('window.height', 800))

    # OpenGL-Flag für pygame Display; moderngl hängt sich an den von
    # pygame/SDL erstellten GL-context (ein wrapper, geteilt mit dem Renderer).
    # RESIZABLE: die auflösung ist dynamisch -- das fenster darf frei skaliert
    # oder maximiert werden, viewport/FXAA-targets/UI-skala folgen über den
    # WINDOWSIZECHANGED-handler in der hauptschleife.
    window_flags = DOUBLEBUF | OPENGL
    if bool(config.get('window.resizable', True)):
        window_flags |= RESIZABLE
    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT), window_flags, vsync=1 if vsync_enabled else 0
    )
    gl_ctx = moderngl.create_context()
    print(gl_ctx.info['GL_VENDOR'], gl_ctx.info['GL_RENDERER'], gl_ctx.info['GL_VERSION'])
    pygame.display.set_caption(str(config.get('window.caption', "Orbital Mechanics - OpenGL Renderer")))
    clock = pygame.time.Clock()
    FPS = int(config.get('window.fps', 180))

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
    # Wer verfolgt wird, entscheidet apply_frame_selection() weiter unten.
    # Hier stand einmal camera.follow(earth) -- wirkungslos, weil die
    # rahmenwahl es sofort wieder auf das schiff setzte. Seit sie eine
    # bestehende verfolgung BEIBEHAELT (damit ein angeflogener planet eine
    # taste R ueberlebt), waere diese zeile keine tote mehr: sie wuerde den
    # start auf die Erde statt auf das schiff legen.

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
    # Auf wie viele punkte die GEZEICHNETE laenge gerundet wird -- haelt die
    # view-identitaet (get_points()) waehrend eines langsamen regler-zugs stehen.
    predictor._display_quantum = int(config.get('predictor.display_length_quantum_points', 8))
    # Look-ahead horizon (length) is the cost knob; point spacing (precision) is
    # cosmetic. Pin the horizon from startup so changing spacing ('9'/'0') no
    # longer moves the horizon (and thus no longer changes compute cost).
    # Default = num_points * base precision, so initial output is unchanged.
    PREDICTOR_BASE_LENGTH = predictor.num_points * predictor.precision
    predictor.set_length(PREDICTOR_BASE_LENGTH)
    # DAS PUNKTBUDGET WAECHST MIT DEM HORIZONT.
    #
    # `_horizon_spacing_floor()` ist `length / num_points` -- bei festem
    # budget verdoppelt jedes '+' also nicht nur den bogen, sondern auch den
    # PUNKTABSTAND. Die zahl der stuetzstellen JE UMLAUF halbiert sich damit
    # bei jedem druck, und irgendwann ueberspannt eine stuetzweite einen
    # nennenswerten teil der bahn: das kubische Hermite-polynom zwischen zwei
    # solchen punkten ist die bahn dann nicht mehr, die linie wird zu beulen
    # mit knicken dazwischen. In einer 2e7-m-erdumlaufbahn sind das 180
    # stuetzstellen je umlauf im grundzustand, 22 bei 8x und 5.6 bei 32x.
    #
    # Also waechst `num_points` mit, bis zur decke `predictor.max_num_points`
    # -- der punktabstand bleibt dann konstant und mit ihm das detail je
    # umlauf. Das ist billig: der integrator muss denselben bogen ueberdecken
    # wie vorher, die schrittzahl aendert sich also nicht; teurer werden nur
    # die ausgabe und die arrays.
    PREDICTOR_BASE_SPACING = (PREDICTOR_BASE_LENGTH
                              / max(1, int(predictor.num_points)))
    PREDICTOR_MAX_POINTS = max(
        int(predictor.num_points),
        int(config.get('predictor.max_num_points', 40000)),
    )
    # Der horizont ist ein PRODUKT: basis * manuell ('+'/'-') * raffung.
    # Der raffungs-anteil kommt aus predictor_warp_length_mult(); ohne ihn
    # deckt die linie bei 1 y/s nur 2.1 tage einer 45-jahre-reise ab und der
    # halt laeuft staendig leer (gemessen 277 volle neuberechnungen je 600
    # frames). Siehe apply_predictor_horizon().
    predictor_manual_mult = 1.0
    predictor_enabled = bool(config.get('predictor.enabled', True))
    if not predictor_enabled:
        predictor.reset()
        predictor.set_num_points(0)
    # Tastenschritte für '+'/'-' (Reichweite) und '9'/'0' (Punktabstand)
    length_step = max(float(config.get('predictor.length_step_factor', 2.0)), 1.0 + 1e-9)
    precision_step = max(float(config.get('predictor.precision_step_factor', 2.0)), 1.0 + 1e-9)
    predictor_toggle_points = int(config.get('predictor.toggle_num_points', 30))
    predictor_min_precision = float(config.get('predictor.min_precision', 1.0))
    HORIZON_MULT_MIN = float(config.get('predictor.horizon_slider_min_mult', 0.25))
    # Die decke MUSS auf dem punktbudget liegen: darueber vergroebert der
    # griff den punktabstand fuer die ganze zugdauer und hebt die
    # fernfeld-schrittdecke -- genau der artefakt, den §23 verhindert.
    HORIZON_MULT_MAX = min(
        float(config.get('predictor.horizon_slider_max_mult', 4.0)),
        PREDICTOR_MAX_POINTS * PREDICTOR_BASE_SPACING
        / max(PREDICTOR_BASE_LENGTH, 1e-9),
    )
    HORIZON_SWEEP_S = float(config.get('predictor.horizon_slider_sweep_seconds', 2.5))
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

    # Entwickler-oberflaeche (Dear ImGui, moderngl-nativ auf demselben context).
    # Standardmaessig unsichtbar; F1 blendet sie ein. Rein werkzeug -- das
    # spieler-HUD ist ein eigenes system.
    devui = ImguiLayer(
        gl_ctx, WIDTH, HEIGHT,
        enabled=bool(config.get('debug.devui_visible', False)),
    )
    devui_toggle_key = pygame.K_F1

    # Spieler-HUD (eigene schicht, siehe spacesim/ui/). Die ui_scale kommt
    # vom renderer, damit HUD und weltbeschriftungen dieselbe skala teilen.
    ui_ctx = UIContext(
        gl_ctx, WIDTH, HEIGHT,
        ui_scale=renderer.ui_scale,
        label_cache_max=int(config.get('renderer.label_texture_cache_max', 256)),
    )
    ui_root = UIRoot(ui_ctx)

    if verbose:
        print("=== Renderer initialisiert ===")
        print(f"=== Konfiguration: {config.filepath.name} ===")
        if config.unknown_keys:
            print(f"CONFIG: unbekannte Schlüssel ignoriert: {', '.join(sorted(set(config.unknown_keys)))}")

    # principia-ähnliche frame-pipeline:
    # selector (eingabe) -> adapter (factory/dispatch) -> renderer (projektion).
    # Ansichts-zustand (bezugsrahmen, referenzkoerper, ziel-overlay).
    #
    # Lag frueher als LOKALE VARIABLEN genau hier -- kein objekt kam daran,
    # ein HUD-bedienelement haette sie nicht lesen und nicht setzen koennen.
    # Jetzt in ui/state.py, mit aenderungs-benachrichtigung: tastatur und
    # HUD schreiben denselben zustand und koennen nicht auseinanderlaufen.
    ui_state = UIState(
        w.body,
        initial_reference_index=(w.body.index(earth) if earth is not None else None),
    )

    frame_adapter = PlottingFrameAdapter(renderer, w.body)

    def on_frame_change(frame_parameters, target_body_index, target_reference_index):
        frame_adapter.update_plotting_frame(
            frame_parameters,
            target_body_index=target_body_index,
            target_reference_index=target_reference_index,
        )

    frame_selector = ReferenceFrameSelector(on_frame_change)

    def apply_frame_selection(state=None):
        state = state if state is not None else ui_state
        reference_index = state.reference_index
        secondary_index = state.secondary_index()
        if state.frame_extension == BODY_CENTRED_BODY_DIRECTION:
            frame_selector.set_to_body_direction(reference_index, secondary_index)
            mode_text = (
                f"body-direction ({w.body[reference_index].name} -> "
                f"{w.body[secondary_index].name})"
            )
        else:
            frame_selector.set_to_body_non_rotating(reference_index)
            mode_text = f"body-centred non-rotating ({w.body[reference_index].name})"

        # Ein rahmen-/bezugskoerperwechsel macht eine gehaltene vorhersage
        # ungueltig: die kurve wird gegen den neuen bezug gerechnet.
        try:
            if hasattr(predictor, 'invalidate_hold'):
                predictor.invalidate_hold()
        except Exception:
            pass

        # predictor-physik-korrektur für translierte nicht-rotierende rahmen:
        # referenzkörper-beschleunigung nur in diesem modus subtrahieren.
        try:
            if hasattr(predictor, 'set_reference_body_index'):
                if state.frame_extension == BODY_CENTRED_NON_ROTATING:
                    predictor.set_reference_body_index(reference_index)
                else:
                    predictor.set_reference_body_index(None)
        except Exception:
            pass

        # DIE KAMERA WIRD HIER NICHT MEHR ANGEFASST. Frueher stand hier
        # bedingungslos `camera.follow(ship)`, damit ein rahmenwechsel die
        # ansicht nicht springen laesst -- das ist aber gar nicht noetig:
        # bildmitte ist `frame(camera.position)`, und kamera wie inhalt gehen
        # durch dieselbe starre transformation, ein rahmenwechsel verschiebt
        # also beide gleich. Wer die kamera bewegt, ist ausschliesslich der
        # spieler: klick auf einen koerper, Home, WASD/ziehen. Sonst haette
        # jede taste R oder 1/2 einen angeflogenen planeten wieder verlassen
        # oder einen freien schwenk zurueckgerissen.
        camera_follow_name = getattr(camera.target, 'name', 'frei')

        if state.target_overlay_enabled and state.ship_index is not None:
            frame_selector.set_target_frame(state.ship_index, reference_index)
            overlay_text = (
                f"ON ({w.body[state.ship_index].name} vs "
                f"{w.body[reference_index].name})"
            )
        else:
            overlay_text = "OFF"

        print(
            f"FRAME: {mode_text} | target_overlay={overlay_text} "
            f"| camera_follow={camera_follow_name}"
        )

    # Startbindung: die kamera haengt am schiff. Das ist die EINZIGE stelle,
    # die sie ohne zutun des spielers anheftet -- ab hier entscheiden nur noch
    # klick, Home und schwenk darueber. Muss VOR dem ersten
    # apply_frame_selection() stehen, sonst meldet dessen ausgabe 'frei'.
    camera.follow(ship if ship is not None else w.body[ui_state.reference_index])

    ui_state.on_change = apply_frame_selection
    apply_frame_selection()

    def get_predictor_horizon_mult():
        return predictor_manual_mult

    def set_predictor_horizon_mult(mult):
        nonlocal predictor_manual_mult
        predictor_manual_mult = max(HORIZON_MULT_MIN,
                                    min(HORIZON_MULT_MAX, float(mult)))

    # Spieler-HUD aufbauen. Muss NACH ui_state und dem frame-selector
    # entstehen: die rahmenwahl im HUD schreibt in ui_state, und dessen
    # aenderungs-benachrichtigung ist erst jetzt verdrahtet.
    # renderer.hud_enabled = false baut das HUD GAR NICHT erst auf (statt es
    # nur zu verstecken): so laesst sich der reine welt-render sauber gegen
    # den mit HUD messen, und wer die alte darstellung will, zahlt nichts.
    hud = None
    if bool(config.get('renderer.hud_enabled', True)):
        hud = Hud(
            ui_root, w, ship, ship_control, camera, renderer, predictor, ui_state,
            tick_rate=float(max(1, FPS)),
            realtime_warp_max=REALTIME_WARP_MAX,
            warp_timescale_divisor=WARP_TIMESCALE_DIVISOR,
            horizon_mult_get=get_predictor_horizon_mult,
            horizon_mult_set=set_predictor_horizon_mult,
            horizon_mult_min=HORIZON_MULT_MIN,
            horizon_mult_max=HORIZON_MULT_MAX,
            horizon_sweep_s=HORIZON_SWEEP_S,
        )

    # Taste Home holt die ansicht zum schiff zurueck -- der weg heraus aus
    # einem angeflogenen planeten, ohne neue tastenbelegung.
    camera.set_home_body(ship)

    # Startansicht ohne einflug: zoom und position sofort auf ihre ziele
    # setzen, statt sie aus dem ursprung heranlaufen zu lassen.
    camera.snap_to_targets()

    # --- anklicken von koerpern --------------------------------------------
    # Erster klick waehlt aus (der renderer setzt vier pfeile darum), ein
    # zweiter klick auf denselben koerper fliegt die kamera hin. Ein klick auf
    # leeren raum hebt die auswahl auf.
    #
    # Die geste wird ueber DOWN/UP zusammengesetzt statt auf MOUSEBUTTONDOWN
    # allein zu reagieren: sonst wuerde jeder schwenk-anfang, der zufaellig auf
    # einem koerper beginnt, die auswahl umwerfen. Erst ein zeigerweg unter
    # `CLICK_SLOP_PX` gilt als klick.
    CLICK_SLOP_PX = 4.0
    click_press_pos = None

    def handle_world_click(screen_pos):
        """Auswahl / anflug. Aendert WEDER bezugskoerper NOCH bezugsrahmen."""
        index = renderer.pick_body(screen_pos, w.body, camera)
        if index is None:
            ui_state.clear_selection()
            return
        if index == ui_state.selected_index:
            # Zweiter klick auf den bereits gewaehlten koerper: hinfliegen.
            camera.focus_on(w.body[index])
            return
        ui_state.select_body(index)


    def update(world, dt):
        """Aktualisiert die Simulation."""
        world.update_dynamics(dt)
        world.update_planets(dt)


    # Größter Physik-Teilschritt (aus config.json): große sim_dt werden für die
    # Dynamik in mehrere Stücke zerlegt, damit der Integrator stabil bleibt.
    MAX_SUBSTEP = max(float(config.get('simulation.max_substep_seconds', 1000.0)), 1e-6)

    # Simulationstakt vom render-takt entkoppeln.
    #
    # Vorher wurde die simulation genau einmal pro gezeichnetem frame um
    # camera.sim_dt vorgerückt -- die simulationsgeschwindigkeit hing damit
    # direkt an der bildrate. Ein einbruch (fenster ziehen, resize, GC) hat die
    # simulation verlangsamt, danach sprang sie.
    #
    # Jetzt wird ZEITPROPORTIONAL vorgerückt: pro frame um
    #     camera.sim_dt * TICK_RATE * frame_dt
    # Die simulationsrate ist damit konstant camera.sim_dt * TICK_RATE
    # sim-sekunden pro echtsekunde, unabhängig von der bildrate.
    #
    # Ein akkumulator mit FESTEN ticks waere die lehrbuch-loesung, ist hier
    # aber falsch: die tick-rate liegt bei der bildrate, also quantisiert der
    # akkumulator gegen den vsync-jitter. Gemessen ueber 600 frames: 9 frames
    # ruecken GAR NICHT vor, 7 frames DOPPELT (16.4 % streuung statt 4.2 %).
    # Sichtbar wird das als stotterndes schiff, das gegen die jeden frame neu
    # gezeichnete predictor-linie springt. Der integrator ist adaptiv (RKN mit
    # schrittweitensteuerung) und step_simulation() zerlegt ohnehin in
    # MAX_SUBSTEP-stuecke, ein variables aeusseres dt ist also unproblematisch.
    TICK_RATE = float(max(1, FPS))

    # Echtzeit muss bei JEDER bildrate erreichbar sein -- sonst bleibt der
    # schub dauerhaft gesperrt. Begruendung steht bei Camera.allow_warp_rate.
    camera.allow_warp_rate(REALTIME_WARP_MAX, TICK_RATE)

    # Obergrenze für das echte frame-delta (simulation, kamera-easing, schub).
    # Nach einem stall darf kein einzelner riesiger schritt eingespeist werden.
    MAX_FRAME_DT = max(1e-4, float(config.get('simulation.max_frame_dt', 0.1)))

    def step_simulation(sim_seconds):
        """Rückt die welt um sim_seconds vor, aufgeteilt in stücke.

        Die stückgrösse ist MAX_SUBSTEP -- ausser im zeitraffer, wo die
        integrator-decke darüber liegt. Das ist kein detail: solange die stücke
        1000 s bleiben, kostet JEDES stück mindestens einen teilschritt, und die
        decke aus world.set_warp_step_ceiling() kann gar nicht wirken. Gemessen
        bei 365 d/s: die teilschritt-zahl bleibt bei 176 (= 175200/1000) egal wie
        hoch die decke gesetzt wird. Es sind also ZWEI decken, und beide müssen
        steigen, sonst bringt keine etwas.
        """
        if sim_seconds <= 0.0:
            return
        # Decke aus der raffung ableiten (in echtzeit bleibt sie bei 30 s, der
        # integrator rechnet dann bit-identisch wie bisher).
        ceiling = w.set_warp_step_ceiling(sim_seconds)
        chunk = max(MAX_SUBSTEP, ceiling)
        if sim_seconds <= chunk:
            update(w, sim_seconds)
            return
        steps = int(math.ceil(sim_seconds / chunk))
        sub_dt = sim_seconds / steps
        for _ in range(steps):
            update(w, sub_dt)

    def warp_rate():
        """Aktuelle raffung in sim-sekunden je echtsekunde."""
        return float(camera.sim_dt) * TICK_RATE


    def clamp_warp_to_orbit():
        """Raffung auf das begrenzen, was die BAHN noch aufloest.

        Nahe an einem koerper ist die obergrenze keine frage der rechen-
        leistung: ein frame bei 1 y/s rueckt um 48 stunden vor, das sind rund
        24 umlaeufe eines 2-stunden-orbits. Gemessen in einem 2000-km-orbit
        bei 1 y/s: 5120 teilschritte und 270 ms je frame -- und die waeren
        auch dann noetig, wenn sie billig waeren, weil sonst schlicht die
        bahn verloren geht.

        Das HUD blendet gesperrte stufen bereits ab; das hier ist der riegel
        fuer PageUp/PageDown und die dev-oberflaeche, die daran vorbeigehen.
        """
        fn = getattr(w, 'characteristic_timescale', None)
        if fn is None or ship is None:
            return
        try:
            t_char = fn(ship)
        except Exception:
            return
        if not t_char or t_char <= 0.0:
            return
        cap_rate = max(t_char / WARP_TIMESCALE_DIVISOR * TICK_RATE,
                       REALTIME_WARP_MAX)
        if warp_rate() > cap_rate:
            camera.sim_dt = max(float(getattr(camera, 'min_sim_dt', 1e-6)),
                                cap_rate / TICK_RATE)

    def thrust_allowed():
        # Kleine toleranz, damit die unterste stufe nicht an rundung scheitert.
        return warp_rate() <= REALTIME_WARP_MAX * 1.001

    def predictor_warp_length_mult():
        """Horizont-faktor aus der raffung -- zweierpotenz, gedeckelt.

        Bei hoher raffung frisst das schiff den horizont schneller als der
        halt ihn nachziehen kann, und jeder leerlauf kostet eine SYNCHRONE
        volle neuberechnung. Ein laengerer horizont ist deshalb bei raffung
        nicht teurer, sondern BILLIGER -- gemessen bei 1 y/s ueber 600 frames:

            faktor  median   p99     max    volle neuberechnungen
              1x    4.20 ms  5.69   8.29         277
             16x    1.62 ms  4.17   4.45           0
             64x    0.70 ms  3.86   4.84           0
            256x    0.28 ms  1.12  54.25           1   <-- sichtbarer hakler

        Der deckel bei 64 ist also nicht willkuerlich: darueber werden die
        neuberechnungen zwar noch seltener, aber die EINZELNE kostet dann so
        viel, dass sie als ruckler sichtbar wird. Zweierpotenzen sorgen
        ausserdem dafuer, dass sich der wert nur beim stufenwechsel aendert --
        set_length() verwirft den halt, das darf nicht jeden frame passieren.
        """
        rate = warp_rate()
        ratio = rate / 604800.0          # ab 7 d/s waechst der horizont mit
        if ratio <= 1.0:
            return 1.0
        # RUNDEN, nicht abschneiden: abgeschnitten faellt 1 y/s auf 32x, und
        # das ist gemessen die schlechtere stufe (max 14.7 ms gegen 4.8 ms bei
        # 64x). Gerundet ergibt die reihe genau die gemessenen guten werte
        # 7d/s->1, 30d/s->4, 100d/s->16, 1y/s->64.
        exp = min(6, max(0, int(round(math.log2(ratio)))))
        return float(1 << exp)

    def apply_predictor_horizon():
        """Horizont neu setzen, wenn sich basis*manuell*raffung geaendert hat."""
        if predictor.num_points <= 0:
            return
        # DIE RAFFUNGS-VERLAENGERUNG IST EIN VORRAT, KEIN BILD -- UND SIE DARF
        # DIE GEZEICHNETE LINIE NICHT ANFASSEN.
        #
        # `wanted` geht an zwei stellen weiter, die beide auf die SICHTBARE
        # kurve durchschlagen, obwohl der verlaengerte teil gar nicht
        # gezeichnet wird (set_display_length unten):
        #
        # 1. `points_wanted` ist bei `PREDICTOR_MAX_POINTS` gedeckelt. Ist der
        #    deckel erreicht, vergroebert jede weitere verlaengerung den
        #    PUNKTABSTAND -- gemessen bei manuell 8x und 64x raffung: 40000
        #    punkte auf dem gezeichneten stueck werden zu **626**, und die
        #    gezeichnete kurve weicht dann selbst mit der kubischen
        #    Hermite-auswertung um **2.3e6 m** von derselben bahn ab (linear
        #    waeren es 6.8e6 m). Auf einer bahn mit perigaeum 1e7 m ist das
        #    eine sichtbar andere linie.
        # 2. `horizon_arc` in `Predictor._make_snapshot` ist `punkte x abstand`
        #    und hebt damit die fernfeld-schrittdecke an. Gemessen dieselbe
        #    lage: decke 2163 -> 8676 s, und die INTEGRIERTE bahn verschiebt
        #    sich um **2.3e6 m** (mit fester decke: 8.4e4 m).
        #
        # Beides zusammen ist der bericht "die vorhersage sieht im zeitraffer
        # ganz anders aus" -- rund 4.6e6 m auf einer linie, deren perigaeum
        # 1e7 m misst, allein vom druck auf die raffungstaste.
        #
        # Der vorrat wird deshalb auf das begrenzt, was das PUNKTBUDGET beim
        # basis-abstand noch traegt. Dann ist `wanted` nie groesser als
        # `PREDICTOR_MAX_POINTS x PREDICTOR_BASE_SPACING`, der abstand bleibt
        # exakt der der echtzeit -- und mit ihm `horizon_arc` und die decke.
        # Hat der spieler mit '+' bereits ueber das budget hinaus verlaengert,
        # faellt der raffungsfaktor auf 1: seine eigene vergroeberung bleibt
        # (die ist gewollt und dokumentiert), die der raffung kommt nicht dazu.
        grabbing = (hud is not None
                    and getattr(hud, 'horizon', None) is not None
                    and hud.horizon.is_grabbing)
        drawn, wanted = horizon_targets(
            PREDICTOR_BASE_LENGTH, predictor_manual_mult,
            predictor_warp_length_mult(),
            PREDICTOR_MAX_POINTS, PREDICTOR_BASE_SPACING,
            grabbing=grabbing, ceiling_mult=HORIZON_MULT_MAX,
        )
        # GEZEICHNET wird immer nur der un-geraffte horizont. Ohne das wickelt
        # sich die linie im zeitraffer mehrfach um die bahn, waehrend sie in
        # echtzeit einen einzigen bogen zeigt -- und die Ap/Pe-fahnen stapeln
        # sich uebereinander. GERECHNET wird trotzdem die volle laenge, weil
        # genau die den halt am leben haelt (siehe predictor_warp_length_mult).
        if hasattr(predictor, 'set_display_length'):
            predictor.set_display_length(drawn if wanted > drawn else None)
        # Punktbudget zuerst, damit `set_length` gleich darauf arbeitet.
        # WEICH: der zeitraffer-schritt verstellt den horizont bei jedem
        # stufenwechsel und damit auch das budget -- ein harter reset waere
        # genau der ruckler, den set_length(soft) schon einmal beseitigt hat
        # (34-82 ms im hauptthread, siehe §17).
        points_wanted = int(min(
            PREDICTOR_MAX_POINTS,
            max(1, math.ceil(wanted / max(PREDICTOR_BASE_SPACING, 1e-9))),
        ))
        if points_wanted != int(predictor.num_points):
            predictor.set_num_points(points_wanted, soft=True)
        current = predictor.length
        if current is not None and abs(current - wanted) <= wanted * 1e-9:
            return
        predictor.set_length(wanted)

    # Was die dev-oberflaeche verstellen darf (siehe devui.DevContext).
    dev_ctx = DevContext(
        world=w, camera=camera, predictor=predictor, renderer=renderer,
        ship_control=ship_control, ship=ship, tick_rate=TICK_RATE,
    )

    # Hauptschleife
    frame_count = 0
    while running:
        raw_frame_dt = clock.tick(FPS) / 1000.0
        frame_dt = min(raw_frame_dt, MAX_FRAME_DT)
        loop_t0 = time.perf_counter()

        # Eingabe-vorfahrt für diesen frame: custom-UI -> ImGui -> welt.
        #
        # begin_frame() macht layout und hover-ermittlung des HUDs und MUSS
        # vor der ereignisschleife laufen -- sonst wird der treffertest gegen
        # das layout des vorframes gemacht.
        # Telemetrie abtasten und die responsive umschaltung anwenden, BEVOR
        # begin_frame() das layout rechnet -- die panelhoehen haengen an den
        # gerade gemessenen texten.
        if hud is not None:
            hud.update()
        ui_root.begin_frame(frame_dt)
        devui.new_frame(frame_dt)
        ui_wants_mouse = ui_root.wants_mouse or devui.wants_mouse
        ui_wants_keyboard = ui_root.wants_keyboard or devui.wants_keyboard

        for event in pygame.event.get():
            # Das spieler-HUD sieht jedes ereignis ZUERST. Verbraucht es das
            # ereignis, bekommen weder ImGui noch die welt es zu sehen.
            consumed_by_hud = ui_root.handle_event(event)

            # ImGui sieht die uebrigen ereignisse, damit es seinen eigenen
            # eingabezustand fuehren kann. Ob es die eingabe auch VERBRAUCHT,
            # entscheiden weiter unten ui_wants_mouse / ui_wants_keyboard.
            if not consumed_by_hud:
                devui.process_event(event)

            if event.type == pygame.QUIT:
                running = False

            # Dynamische auflösung: fenstergröße geändert (ziehen, maximieren,
            # DPI-wechsel). Renderer.resize() setzt viewport, baut die
            # FXAA-targets neu auf, leert die text-caches und leitet ui_scale
            # neu ab; die kamera braucht die neuen maße für world_to_screen.
            elif event.type == pygame.WINDOWSIZECHANGED:
                new_w = max(1, int(event.x))
                new_h = max(1, int(event.y))
                if (new_w, new_h) != (renderer.width, renderer.height):
                    renderer.resize(new_w, new_h)
                    camera.width = new_w
                    camera.height = new_h
                    devui.resize(new_w, new_h)
                    # ui_scale kommt vom renderer, damit HUD und
                    # weltbeschriftungen exakt dieselbe skala benutzen.
                    ui_root.resize(new_w, new_h, ui_scale=renderer.ui_scale)

            elif event.type == pygame.KEYDOWN:
                # F1 schaltet die dev-oberflaeche IMMER um, auch wenn ImGui
                # gerade die tastatur haelt -- sonst liesse sie sich nicht
                # mehr schliessen, sobald ein eingabefeld fokussiert ist.
                if event.key == devui_toggle_key:
                    devui.toggle()
                    continue
                if ui_wants_keyboard:
                    # Tastatur gehoert der oberflaeche (texteingabe o.ae.).
                    continue
                if event.key == pygame.K_ESCAPE:
                    running = False
                # Taste P für Predictive Orbit umschalten
                elif event.key == pygame.K_p:
                    if predictor.num_points > 0:
                        predictor.reset()
                    else:
                        predictor.set_num_points(predictor_toggle_points)

                # Taste O: bahnlinien der koerper umschalten
                elif event.key == pygame.K_o:
                    renderer.orbit_lines_enabled = not renderer.orbit_lines_enabled
                    print(f"ORBIT LINES: {'on' if renderer.orbit_lines_enabled else 'off'}")

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

                # R / 1 / 2 / T schreiben denselben zustand wie die
                # HUD-bedienelemente (ui/state.py) -- deshalb kein direktes
                # setzen mehr, sondern die methoden des zustands. Das
                # anwenden loest die aenderungs-benachrichtigung aus.
                elif event.key == pygame.K_r:
                    ui_state.cycle_reference()

                elif event.key == pygame.K_1:
                    ui_state.set_frame_extension(BODY_CENTRED_NON_ROTATING)

                elif event.key == pygame.K_2:
                    ui_state.set_frame_extension(BODY_CENTRED_BODY_DIRECTION)

                elif event.key == pygame.K_t:
                    if not ui_state.toggle_target_overlay():
                        print("FRAME: no ship available for target overlay")

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
                    # '+'/'-' verstellen den MANUELLEN faktor, nicht die laenge
                    # direkt -- sonst wuerde apply_predictor_horizon() die
                    # eingabe im naechsten frame wieder ueberschreiben.
                    predictor_manual_mult *= length_step
                    apply_predictor_horizon()
                    predictor.reset()
                    print(f"PREDICTOR: length set to {predictor.length} "
                          f"(manuell x{predictor_manual_mult:g}, "
                          f"raffung x{predictor_warp_length_mult():g})")
                elif ch == '-' or event.key == pygame.K_KP_MINUS:
                    lowest = predictor.precision / max(PREDICTOR_BASE_LENGTH, 1e-9)
                    predictor_manual_mult = max(lowest, predictor_manual_mult / length_step)
                    apply_predictor_horizon()
                    predictor.reset()
                    print(f"PREDICTOR: length set to {predictor.length} "
                          f"(manuell x{predictor_manual_mult:g}, "
                          f"raffung x{predictor_warp_length_mult():g})")
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

            # Eingabe-vorfahrt: custom-UI -> ImGui -> welt (kamera/schiff).
            # consumed_by_hud faengt den fall ab, dass ein HUD-element das
            # ereignis in DIESEM frame beansprucht hat; ui_wants_* deckt den
            # allgemeinen zustand ab (z. B. ein regler, der gerade gezogen
            # wird, auch wenn der zeiger ihn verlassen hat).
            if consumed_by_hud:
                continue

            # Linke maustaste: koerper auswaehlen / anfliegen. Die kamera
            # selbst zieht mit der mittleren/rechten taste, hier gibt es also
            # keinen streit um die geste -- nur der weg des zeigers zwischen
            # DOWN und UP entscheidet, ob es ein klick war.
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                click_press_pos = (
                    None if ui_wants_mouse
                    else (float(event.pos[0]), float(event.pos[1]))
                )
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if click_press_pos is not None:
                    dx = float(event.pos[0]) - click_press_pos[0]
                    dy = float(event.pos[1]) - click_press_pos[1]
                    if (dx * dx + dy * dy) <= CLICK_SLOP_PX * CLICK_SLOP_PX:
                        handle_world_click(event.pos)
                click_press_pos = None

            camera.handle_event(
                event,
                ui_wants_mouse=ui_wants_mouse,
                ui_wants_keyboard=ui_wants_keyboard,
            )

        # Schiff-Steuerung
        keys = pygame.key.get_pressed()
        reference_body = ui_state.reference_body
        if ship_control:
            ship_control.last_thrust_direction = None
            if ship is not None:
                setattr(ship, "last_thrust_direction", None)
            # Schiff-steuerung liest den tastaturzustand direkt (polling, keine
            # ereignisse) -- deshalb muss die eingabe-vorfahrt hier gesondert
            # geprueft werden, sonst fliegt das schiff waehrend einer
            # texteingabe in der dev-oberflaeche mit.
            if not ui_wants_keyboard:
                # rotation: in echtzeit sanft. DREHEN BLEIBT IMMER ERLAUBT,
                # auch im zeitraffer -- es aendert die bahn nicht.
                ship_control.handle_rotation(keys, frame_dt)
                # SCHUB NUR IN ECHTZEIT. Oberhalb der untersten zeitraffer-
                # stufe rueckt die welt je frame um stunden bis tage vor; ein
                # impuls "einmal pro frame" waere dort weder dosierbar noch
                # reproduzierbar (er haenge an der bildrate), und er macht die
                # gehaltene vorhersage in jedem frame ungueltig. Deshalb ist
                # der schub gesperrt, solange gerafft wird -- der spieler
                # geht zum manoevrieren auf die unterste stufe zurueck.
                if thrust_allowed():
                    ship_control.apply_thrust(keys, frame_dt)

        # Raffung auf die bahn-zeitskala begrenzen (siehe clamp_warp_to_orbit).
        clamp_warp_to_orbit()
        # Horizont an die raffung anpassen (no-op, solange die stufe steht).
        apply_predictor_horizon()

        # Simulation zeitproportional vorrücken (siehe TICK_RATE oben).
        # frame_dt ist bereits auf MAX_FRAME_DT gekappt, ein stall kann also
        # keinen riesigen sprung einspeisen.
        step_simulation(camera.sim_dt * TICK_RATE * frame_dt)

        # kamera mit echtem frame-delta für interaktives panning aktualisieren
        # (zoom/schwenk laufen ihren zielen geglättet nach)
        camera.update(frame_dt, ui_wants_keyboard=ui_wants_keyboard)

        # orbit-prognose berechnen (für das Schiff oder einen Körper)
        points = []

        if predictor.num_points > 0:
            target = ship if ship else next((b for b in w.body if not b.fixed), None)

            if target:
                # Im zeitraffer die kurve HALTEN statt jeden frame neu
                # rechnen -- sonst zieht _anchor_first_point sie je frame um
                # die volle bahnbewegung starr mit und sie zittert. Siehe
                # Predictor._hold_advance.
                predictor.set_hold(not thrust_allowed())
                if hasattr(predictor, 'set_view_scale'):
                    # WICHTIG: das zoom-ZIEL einspeisen, nicht die gerade
                    # nachlaufende skala. set_view_scale() setzt bei jeder
                    # änderung > snapshot_view_rel_tol (1e-6) das flag
                    # _view_scale_changed, was in Predictor.update() einen
                    # SYNCHRONEN _compute_full() im hauptthread auslöst. Mit
                    # der animierten skala wäre das ein voller neuaufbau der
                    # trajektorie in JEDEM frame einer zoom-animation. Das ziel
                    # ist während der animation konstant -> genau ein
                    # neuaufbau pro mausrad-raste.
                    predictor.set_view_scale(camera.target_scale)
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
            selected_body=ui_state.selected_body,
        )

        # Overlays NACH der welt und VOR dem swap. render() macht den swap
        # nicht mehr selbst -- das uebernimmt renderer.present() unten.
        #
        # Reihenfolge: spieler-HUD zuerst, entwicklerwerkzeuge darueber. Das
        # HUD landet damit hinter dem FXAA-resolve (render() ist fertig) --
        # ein kantenfilter ueber UI-text und 1px-rahmen wuerde beides
        # verschmieren.
        ui_root.render()

        dev_ctx.frame_dt = frame_dt
        dev_ctx.sim_step_s = camera.sim_dt * TICK_RATE * frame_dt
        devui.build(dev_ctx)
        devui.render()

        renderer.present()

        # Zeitreihen fuer die graphen der dev-oberflaeche (F1 -> Timing).
        #
        # NACH present(): render() setzt swap_or_present_ms auf 0.0 und erst
        # present() traegt den echten wert nach -- davor abgetastet waere
        # `render draw` konstant null. Das panel zeigt damit den stand des
        # VORIGEN frames, was bei 180 fps niemand sieht.
        #
        # Laeuft unbedingt, auch mit geschlossenem panel: ein puffer, der nur
        # gefuellt wird, waehrend man hinschaut, ist beim aufklappen leer.
        # Kostet dafuer gemessene 1.0 us je frame (0.02 % eines 5.6-ms-frames),
        # siehe tests/devui_timing_test.py.
        frame_ms = (time.perf_counter() - loop_t0) * 1000.0
        dev_ctx.sample_timings(frame_ms)

        # Per-frame timing debug line. Splits the frame into predictor line
        # calculation vs. drawing, and the render pipeline into CPU calculation
        # vs. present-on-screen (swap/flip; includes the VSync wait). Dieselben
        # vier groessen wie die graphen oben, aus derselben quelle.
        if print_timings:
            rt = getattr(renderer, 'last_frame_timings', {}) or {}
            ps = getattr(renderer, '_last_prediction_render_stats', {}) or {}
            # `frame_ms` ist die dauer von render() selbst (present() ruehrt
            # es nicht mehr an), also ist rend_calc genau das. Was zwischen
            # render() und present() gezeichnet wird -- spieler-HUD und
            # dev-oberflaeche -- steht getrennt als ui_calc; frueher lief es
            # unsichtbar unter rend_calc und hat die zahl verdoppelt.
            rend_calc = float(rt.get('frame_ms', 0.0))
            rend_draw = float(rt.get('swap_or_present_ms', 0.0))
            ui_calc = float(rt.get('overlay_ms', 0.0))
            pred_calc = float(getattr(predictor, 'last_compute_ms', 0.0))
            pred_draw = float(ps.get('prepare_ms', 0.0)) + float(ps.get('draw_ms', 0.0))

            # Wie oft die LINIE selbst neu wird -- das ist eine andere groesse
            # als pred_calc (die dauer EINER rechnung) und die eigentlich
            # interessante beim schub: mehrere rechnungen laufen versetzt
            # nebeneinander, der durchsatz ist deshalb hoeher als 1/pred_calc.
            # Ziel ist ein wert nahe der bildrate; `pipe` zeigt, wie viele
            # rechnungen der predictor dafuer gerade parallel faehrt.
            now_hz = time.perf_counter()
            swaps_now = int(getattr(predictor, '_jobs_swapped', 0))
            if timing_hz_t0 is None:
                timing_hz_t0 = now_hz
                timing_hz_swaps = swaps_now
            elapsed_hz = now_hz - timing_hz_t0
            if elapsed_hz >= 0.5:
                timing_pred_hz = (swaps_now - timing_hz_swaps) / elapsed_hz
                timing_hz_t0 = now_hz
                timing_hz_swaps = swaps_now
            print(
                f"TIMING: pred_calc={pred_calc:.1f}ms pred_draw={pred_draw:.1f}ms "
                f"rend_calc={rend_calc:.1f}ms ui_calc={ui_calc:.1f}ms "
                f"rend_draw={rend_draw:.1f}ms "
                f"frame={frame_ms:.1f}ms "
                f"pred_hz={timing_pred_hz:.0f} pipe={int(getattr(predictor, '_pipeline_depth_used', 1))}",
                flush=True,
            )

        frame_count += 1
        if max_frames > 0 and frame_count >= max_frames:
            running = False

    devui.shutdown()
    pygame.quit()

if __name__ == "__main__":
    main()
