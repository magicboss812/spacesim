"""Den ganzen apparat zusammenbauen: welt, kamera, predictor, renderer, UI.

Das war der mittelteil von `main()` in `test.py`. DIE REIHENFOLGE HIER IST
TRAGEND, an drei stellen sogar begruendet:

  * `camera.follow(ship)` muss VOR dem ersten `apply_frame_selection()` stehen,
    sonst meldet dessen ausgabe `camera_follow=frei`;
  * das HUD entsteht NACH `ui_state` und dem frame-selector, weil die
    rahmenwahl im HUD in `ui_state` schreibt und dessen benachrichtigung erst
    dann verdrahtet ist;
  * `REALTIME_WARP_MAX` und `WARP_TIMESCALE_DIVISOR` stehen vor dem HUD-aufbau,
    weil der schubregler die sperre anzeigt und das HUD stufen damit abblendet.
"""
import os

from config.loader import ConfigLoader
from runtime.system_loader import SystemLoader
from physics.world import world
from physics.reference_frames import (
    BODY_CENTRED_BODY_DIRECTION,
    BODY_CENTRED_NON_ROTATING,
    PlottingFrameAdapter,
    ReferenceFrameSelector,
)
from ship.camera import Camera
from ship.control import schiffcontrol
from ship.horizon import HorizonPolicy
from ship.predictor import Predictor
from render.renderer import Renderer
from runtime.window import Window
from ui import UIContext, UIRoot, UIState
from ui.devui import DevContext, ImguiLayer
from ui.hud import Hud


def load_config():
    """Zentrale Konfiguration laden.

    ALLE spielbaren parameter stehen in config.json; der ConfigLoader verteilt
    sie auf world/camera/schiff/predictor/renderer. SPACESIM_CONFIG kann eine
    alternative datei vorgeben (z. B. fuer messlaeufe).
    """
    config = ConfigLoader(os.environ.get("SPACESIM_CONFIG") or None)
    config.load()
    config.apply_globals()
    return config


class FrameController:
    """Bezugsrahmen und bezugskoerper anwenden.

    Haengt an `UIState.on_change`, wird also von tastatur (R / 1 / 2 / T) und
    HUD gleichermassen ausgeloest -- die beiden koennen deshalb nicht
    auseinanderlaufen.
    """

    def __init__(self, world_obj, ui_state, predictor, camera, renderer):
        self.world = world_obj
        self.ui_state = ui_state
        self.predictor = predictor
        self.camera = camera
        self.adapter = PlottingFrameAdapter(renderer, world_obj.body)
        self.selector = ReferenceFrameSelector(self._on_frame_change)

    def _on_frame_change(self, frame_parameters, target_body_index,
                         target_reference_index):
        self.adapter.update_plotting_frame(
            frame_parameters,
            target_body_index=target_body_index,
            target_reference_index=target_reference_index,
        )

    def apply(self, state=None):
        state = state if state is not None else self.ui_state
        bodies = self.world.body
        reference_index = state.reference_index
        secondary_index = state.secondary_index()
        if state.frame_extension == BODY_CENTRED_BODY_DIRECTION:
            self.selector.set_to_body_direction(reference_index, secondary_index)
            mode_text = (
                f"body-direction ({bodies[reference_index].name} -> "
                f"{bodies[secondary_index].name})"
            )
        else:
            self.selector.set_to_body_non_rotating(reference_index)
            mode_text = f"body-centred non-rotating ({bodies[reference_index].name})"

        # Ein rahmen-/bezugskoerperwechsel macht eine gehaltene vorhersage
        # ungueltig: die kurve wird gegen den neuen bezug gerechnet.
        try:
            if hasattr(self.predictor, 'invalidate_hold'):
                self.predictor.invalidate_hold()
        except Exception:
            pass

        # predictor-physik-korrektur fuer translierte nicht-rotierende rahmen:
        # referenzkoerper-beschleunigung nur in diesem modus subtrahieren.
        try:
            if hasattr(self.predictor, 'set_reference_body_index'):
                if state.frame_extension == BODY_CENTRED_NON_ROTATING:
                    self.predictor.set_reference_body_index(reference_index)
                else:
                    self.predictor.set_reference_body_index(None)
        except Exception:
            pass

        # DIE KAMERA WIRD HIER NICHT ANGEFASST. Frueher stand hier
        # bedingungslos `camera.follow(ship)`, damit ein rahmenwechsel die
        # ansicht nicht springen laesst -- das ist aber gar nicht noetig:
        # bildmitte ist `frame(camera.position)`, und kamera wie inhalt gehen
        # durch dieselbe starre transformation, ein rahmenwechsel verschiebt
        # also beide gleich. Wer die kamera bewegt, ist ausschliesslich der
        # spieler: klick auf einen koerper, Home, WASD/ziehen. Sonst haette
        # jede taste R oder 1/2 einen angeflogenen planeten wieder verlassen
        # oder einen freien schwenk zurueckgerissen.
        camera_follow_name = getattr(self.camera.target, 'name', 'frei')

        if state.target_overlay_enabled and state.ship_index is not None:
            self.selector.set_target_frame(state.ship_index, reference_index)
            overlay_text = (
                f"ON ({bodies[state.ship_index].name} vs "
                f"{bodies[reference_index].name})"
            )
        else:
            overlay_text = "OFF"

        print(
            f"FRAME: {mode_text} | target_overlay={overlay_text} "
            f"| camera_follow={camera_follow_name}"
        )


class App:
    """Alles, was die hauptschleife braucht -- ein sack voll verweise.

    Bewusst keine logik: was hier als methode stuende, gehoerte in das modul,
    dem das feld gehoert.
    """

    __slots__ = (
        'config', 'window', 'world', 'camera', 'ship', 'ship_control',
        'predictor', 'renderer', 'devui', 'ui_ctx', 'ui_root', 'ui_state',
        'hud', 'horizon', 'frames', 'dev_ctx', 'focus_aliases',
        'tick_rate', 'max_substep', 'max_frame_dt', 'realtime_warp_max',
        'warp_timescale_divisor', 'max_frames', 'verbose', 'print_timings',
        'predictor_toggle_points', 'predictor_min_precision', 'precision_step',
    )

    def warp_rate(self):
        return self.camera.warp_rate(self.tick_rate)

    def thrust_allowed(self):
        return self.camera.thrust_allowed(self.tick_rate, self.realtime_warp_max)


def build_app(config):
    app = App()
    app.config = config
    app.verbose = bool(config.get('debug.print_loader_info', True))
    app.print_timings = bool(config.get('debug.print_frame_timings', True))

    max_frames = int(config.get('simulation.max_frames', 0) or 0)
    try:
        # Umgebungsvariable hat Vorrang vor der Konfiguration (Messlaeufe).
        env_max_frames = int(os.environ.get("SPACESIM_MAX_FRAMES", "0") or "0")
    except Exception:
        env_max_frames = 0
    if env_max_frames > 0:
        max_frames = env_max_frames
    app.max_frames = max(0, max_frames)

    # Ab welcher raffung (sim-sekunden je echtsekunde) gilt der lauf als
    # "gerafft": darueber ist der schub gesperrt und die vorhersage wird
    # gehalten statt neu gerechnet. Standard ist genau die unterste
    # zeitraffer-stufe des HUDs (60 sim-s/s), also "echtzeit" in diesem spiel.
    # Muss VOR dem HUD-aufbau stehen -- der schubregler zeigt die sperre an.
    app.realtime_warp_max = float(config.get('simulation.realtime_warp_max', 60.0))
    # Wie viele bahn-zeitskalen ein frame hoechstens vorruecken darf.
    # Muss VOR dem Hud-aufbau stehen -- das HUD blendet damit stufen ab.
    app.warp_timescale_divisor = max(
        float(config.get('simulation.warp_timescale_divisor', 3.0)), 1e-6)
    # Groesster Physik-Teilschritt: grosse sim_dt werden fuer die dynamik in
    # mehrere stuecke zerlegt, damit der integrator stabil bleibt.
    app.max_substep = max(
        float(config.get('simulation.max_substep_seconds', 1000.0)), 1e-6)
    # Obergrenze fuer das echte frame-delta (simulation, kamera-easing, schub).
    # Nach einem stall darf kein einzelner riesiger schritt eingespeist werden.
    app.max_frame_dt = max(1e-4, float(config.get('simulation.max_frame_dt', 0.1)))

    app.window = Window(config)
    gl_ctx = app.window.ctx
    width, height = app.window.width, app.window.height

    # -- welt ---------------------------------------------------------------
    loader = SystemLoader(config.get('simulation.system_file', "solar_system.json"))
    bodies = loader.load()
    if app.verbose:
        print("=== Geladene Körper (nach Loader) ===")
        for b in bodies:
            print(f"  {b.name}: pos={b.position}, vel={b.velocity}, "
                  f"is_ship={b.is_ship}, fixed={b.fixed}")

    w = world(float(config.get('physics.gravitational_constant', 6.6730831e-11)))
    w.body = bodies
    config.apply_to_world(w)
    app.world = w

    # -- kamera und die beiden gesuchten koerper ----------------------------
    app.camera = Camera(app.window.screen, width, height)
    config.apply_to_camera(app.camera)
    focus_name = str(config.get('simulation.focus_body', "Erde")).strip().lower()
    app.focus_aliases = {focus_name, 'earth', 'erde'}
    earth = next((b for b in bodies
                  if getattr(b, 'name', '').lower() in app.focus_aliases), None)
    app.ship = next((b for b in w.body if getattr(b, 'is_ship', False)), None)

    # -- predictor ----------------------------------------------------------
    predictor_kwargs = config.predictor_kwargs()
    env_async = os.environ.get("SPACESIM_PREDICTOR_ASYNC")
    if env_async is not None:
        # Umgebungsvariable ueberschreibt die Konfiguration (Messlaeufe).
        predictor_kwargs['async_compute'] = (
            env_async.strip().lower() not in ("0", "false", "no", "off"))
    predictor = Predictor(recompute_every_update=True, **predictor_kwargs)
    config.apply_to_predictor(predictor)
    # Auf wie viele punkte die GEZEICHNETE laenge gerundet wird -- haelt die
    # view-identitaet (get_points()) waehrend eines langsamen regler-zugs stehen.
    predictor._display_quantum = int(
        config.get('predictor.display_length_quantum_points', 8))
    app.predictor = predictor

    # Setzt zugleich den basis-horizont am predictor (siehe HorizonPolicy).
    app.horizon = HorizonPolicy(predictor, config)

    if not bool(config.get('predictor.enabled', True)):
        predictor.reset()
        predictor.set_num_points(0)

    # Tastenschritte fuer '9'/'0' (punktabstand); '+'/'-' liegen an der
    # HorizonPolicy, weil sie den manuellen faktor verstellen.
    app.precision_step = max(
        float(config.get('predictor.precision_step_factor', 2.0)), 1.0 + 1e-9)
    app.predictor_toggle_points = int(config.get('predictor.toggle_num_points', 30))
    app.predictor_min_precision = float(config.get('predictor.min_precision', 1.0))

    if app.verbose:
        print(f"PREDICTOR DEBUG: async_compute = {predictor.async_compute}")
        print(f"PREDICTOR DEBUG: force_sync_on_stale = {predictor.force_sync_on_stale}")

    # -- schiff und darstellung ---------------------------------------------
    app.ship_control = schiffcontrol(app.ship) if app.ship else None
    config.apply_to_ship_control(app.ship_control)

    app.renderer = Renderer(
        width, height,
        enable_fxaa=bool(config.get('window.enable_fxaa', True)),
        ctx=gl_ctx,
    )
    config.apply_to_renderer(app.renderer)

    # Entwickler-oberflaeche (Dear ImGui, moderngl-nativ auf demselben
    # context). Standardmaessig unsichtbar; F1 blendet sie ein. Rein werkzeug
    # -- das spieler-HUD ist ein eigenes system.
    app.devui = ImguiLayer(
        gl_ctx, width, height,
        enabled=bool(config.get('debug.devui_visible', False)),
    )

    # Spieler-HUD (eigene schicht, siehe spacesim/ui/). Die ui_scale kommt vom
    # renderer, damit HUD und weltbeschriftungen dieselbe skala teilen.
    app.ui_ctx = UIContext(
        gl_ctx, width, height,
        ui_scale=app.renderer.ui_scale,
        label_cache_max=int(config.get('renderer.label_texture_cache_max', 256)),
    )
    app.ui_root = UIRoot(app.ui_ctx)

    if app.verbose:
        print("=== Renderer initialisiert ===")
        print(f"=== Konfiguration: {config.filepath.name} ===")
        if config.unknown_keys:
            print("CONFIG: unbekannte Schlüssel ignoriert: "
                  f"{', '.join(sorted(set(config.unknown_keys)))}")

    # -- ansichts-zustand und rahmen-pipeline -------------------------------
    # principia-aehnliche pipeline:
    # selector (eingabe) -> adapter (factory/dispatch) -> renderer (projektion).
    #
    # Lag frueher als LOKALE VARIABLEN in main() -- kein objekt kam daran, ein
    # HUD-bedienelement haette sie nicht lesen und nicht setzen koennen. Jetzt
    # in ui/state.py, mit aenderungs-benachrichtigung: tastatur und HUD
    # schreiben denselben zustand und koennen nicht auseinanderlaufen.
    app.ui_state = UIState(
        w.body,
        initial_reference_index=(w.body.index(earth) if earth is not None else None),
    )
    app.frames = FrameController(w, app.ui_state, predictor, app.camera, app.renderer)

    # Startbindung: die kamera haengt am schiff. Das ist die EINZIGE stelle,
    # die sie ohne zutun des spielers anheftet -- ab hier entscheiden nur noch
    # klick, Home und schwenk darueber. Muss VOR dem ersten apply() stehen,
    # sonst meldet dessen ausgabe 'frei'.
    app.camera.follow(app.ship if app.ship is not None
                      else w.body[app.ui_state.reference_index])

    app.ui_state.on_change = app.frames.apply
    app.frames.apply()

    # -- HUD ----------------------------------------------------------------
    # Muss NACH ui_state und dem frame-selector entstehen. hud_enabled = false
    # baut das HUD GAR NICHT erst auf (statt es nur zu verstecken): so laesst
    # sich der reine welt-render sauber gegen den mit HUD messen.
    app.tick_rate = float(max(1, app.window.fps))
    app.hud = None
    if bool(config.get('renderer.hud_enabled', True)):
        app.hud = Hud(
            app.ui_root, w, app.ship, app.ship_control, app.camera,
            app.renderer, predictor, app.ui_state,
            tick_rate=app.tick_rate,
            realtime_warp_max=app.realtime_warp_max,
            warp_timescale_divisor=app.warp_timescale_divisor,
            horizon_mult_get=app.horizon.get_mult,
            horizon_mult_set=app.horizon.set_mult,
            horizon_mult_min=app.horizon.mult_min,
            horizon_mult_max=app.horizon.mult_max,
            horizon_sweep_s=app.horizon.sweep_s,
        )

    # Taste Home holt die ansicht zum schiff zurueck -- der weg heraus aus
    # einem angeflogenen planeten, ohne neue tastenbelegung.
    app.camera.set_home_body(app.ship)
    # Startansicht ohne einflug: zoom und position sofort auf ihre ziele
    # setzen, statt sie aus dem ursprung heranlaufen zu lassen.
    app.camera.snap_to_targets()

    # Echtzeit muss bei JEDER bildrate erreichbar sein -- sonst bleibt der
    # schub dauerhaft gesperrt. Begruendung steht bei Camera.allow_warp_rate.
    app.camera.allow_warp_rate(app.realtime_warp_max, app.tick_rate)

    # Was die dev-oberflaeche verstellen darf (siehe ui/devui.py).
    app.dev_ctx = DevContext(
        world=w, camera=app.camera, predictor=predictor, renderer=app.renderer,
        ship_control=app.ship_control, ship=app.ship, tick_rate=app.tick_rate,
    )
    return app
