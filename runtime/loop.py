"""Die hauptschleife.

Die REIHENFOLGE in `run()` ist an mehreren stellen begruendet und darf nicht
umsortiert werden -- die begruendungen stehen jeweils an der zeile.
"""
import time

import pygame

from runtime.input import InputRouter


class FrameTimingPrinter:
    """Die `TIMING:`-zeile je frame.

    Zerlegt den frame in vorhersage-rechnung gegen -zeichnung und die
    render-pipeline in CPU-rechnung gegen present-on-screen (swap/flip,
    inklusive der VSync-wartezeit). Dieselben vier groessen wie die graphen
    der dev-oberflaeche, aus derselben quelle.
    """

    def __init__(self):
        # Zaehlerstand fuer pred_hz (erneuerungen der vorhersagelinie je
        # sekunde, ueber ein halbsekunden-fenster gemittelt -- je bild gezaehlt
        # waere es nur 0 oder 1).
        self._hz_t0 = None
        self._hz_swaps = 0
        self._pred_hz = 0.0

    def emit(self, renderer, predictor, frame_ms):
        rt = getattr(renderer, 'last_frame_timings', {}) or {}
        ps = getattr(renderer, '_last_prediction_render_stats', {}) or {}
        # rend_calc ist die dauer von render() selbst. Was zwischen render()
        # und present() gezeichnet wird -- spieler-HUD und dev-oberflaeche --
        # steht getrennt als ui_calc.
        rend_calc = float(rt.get('frame_ms', 0.0))
        rend_draw = float(rt.get('swap_or_present_ms', 0.0))
        ui_calc = float(rt.get('overlay_ms', 0.0))
        pred_calc = float(getattr(predictor, 'last_compute_ms', 0.0))
        pred_draw = float(ps.get('prepare_ms', 0.0)) + float(ps.get('draw_ms', 0.0))

        # Wie oft die LINIE selbst neu wird -- das ist eine andere groesse als
        # pred_calc (die dauer EINER rechnung) und die eigentlich interessante
        # beim schub: mehrere rechnungen laufen versetzt nebeneinander, der
        # durchsatz ist deshalb hoeher als 1/pred_calc. Ziel ist ein wert nahe
        # der bildrate; `pipe` zeigt, wie viele rechnungen der predictor dafuer
        # gerade parallel faehrt.
        now_hz = time.perf_counter()
        swaps_now = int(getattr(predictor, '_jobs_swapped', 0))
        if self._hz_t0 is None:
            self._hz_t0 = now_hz
            self._hz_swaps = swaps_now
        elapsed_hz = now_hz - self._hz_t0
        if elapsed_hz >= 0.5:
            self._pred_hz = (swaps_now - self._hz_swaps) / elapsed_hz
            self._hz_t0 = now_hz
            self._hz_swaps = swaps_now
        print(
            f"TIMING: pred_calc={pred_calc:.1f}ms pred_draw={pred_draw:.1f}ms "
            f"rend_calc={rend_calc:.1f}ms ui_calc={ui_calc:.1f}ms "
            f"rend_draw={rend_draw:.1f}ms "
            f"frame={frame_ms:.1f}ms "
            f"pred_hz={self._pred_hz:.0f} "
            f"pipe={int(getattr(predictor, '_pipeline_depth_used', 1))}",
            flush=True,
        )


def _handle_resize(app, event):
    """Dynamische aufloesung: fenstergroesse geaendert (ziehen, maximieren, DPI).

    Renderer.resize() setzt viewport, baut die FXAA-targets neu auf, leert die
    text-caches und leitet ui_scale neu ab; die kamera braucht die neuen masse
    fuer world_to_screen.
    """
    new_w = max(1, int(event.x))
    new_h = max(1, int(event.y))
    if (new_w, new_h) == (app.renderer.width, app.renderer.height):
        return
    app.renderer.resize(new_w, new_h)
    app.camera.width = new_w
    app.camera.height = new_h
    app.devui.resize(new_w, new_h)
    # ui_scale kommt vom renderer, damit HUD und weltbeschriftungen exakt
    # dieselbe skala benutzen.
    app.ui_root.resize(new_w, new_h, ui_scale=app.renderer.ui_scale)


def run(app):
    """Die schleife. Laeuft, bis Esc, das fensterkreuz oder max_frames greift."""
    router = InputRouter(app)
    # Die HUD-knoepfe rufen ueber diese box denselben router wie die tasten.
    if getattr(app, 'maneuver_router_ref', None) is not None:
        app.maneuver_router_ref[0] = router
    timing = FrameTimingPrinter()
    devui_toggle_key = pygame.K_F1

    running = True
    frame_count = 0
    while running:
        raw_frame_dt = app.window.tick()
        frame_dt = min(raw_frame_dt, app.max_frame_dt)
        loop_t0 = time.perf_counter()

        # Eingabe-vorfahrt fuer diesen frame: custom-UI -> ImGui -> welt.
        #
        # Telemetrie abtasten und die responsive umschaltung anwenden, BEVOR
        # begin_frame() das layout rechnet -- die panelhoehen haengen an den
        # gerade gemessenen texten. begin_frame() macht layout und
        # hover-ermittlung des HUDs und MUSS vor der ereignisschleife laufen --
        # sonst wird der treffertest gegen das layout des vorframes gemacht.
        if app.hud is not None:
            app.hud.update()
        app.ui_root.begin_frame(frame_dt)
        app.devui.new_frame(frame_dt)
        ui_wants_mouse = app.ui_root.wants_mouse or app.devui.wants_mouse
        ui_wants_keyboard = app.ui_root.wants_keyboard or app.devui.wants_keyboard

        for event in pygame.event.get():
            # Das spieler-HUD sieht jedes ereignis ZUERST. Verbraucht es das
            # ereignis, bekommen weder ImGui noch die welt es zu sehen.
            consumed_by_hud = app.ui_root.handle_event(event)

            # ImGui sieht die uebrigen ereignisse, damit es seinen eigenen
            # eingabezustand fuehren kann. Ob es die eingabe auch VERBRAUCHT,
            # entscheiden ui_wants_mouse / ui_wants_keyboard.
            if not consumed_by_hud:
                app.devui.process_event(event)

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.WINDOWSIZECHANGED:
                _handle_resize(app, event)

            elif event.type == pygame.KEYDOWN:
                # F1 schaltet die dev-oberflaeche IMMER um, auch wenn ImGui
                # gerade die tastatur haelt -- sonst liesse sie sich nicht mehr
                # schliessen, sobald ein eingabefeld fokussiert ist.
                if event.key == devui_toggle_key:
                    app.devui.toggle()
                    continue
                if ui_wants_keyboard:
                    # Tastatur gehoert der oberflaeche (texteingabe o.ae.).
                    continue
                if not router.handle_keydown(event):
                    running = False

            # Eingabe-vorfahrt: custom-UI -> ImGui -> welt (kamera/schiff).
            # consumed_by_hud faengt den fall ab, dass ein HUD-element das
            # ereignis in DIESEM frame beansprucht hat; ui_wants_* deckt den
            # allgemeinen zustand ab (z. B. ein regler, der gerade gezogen
            # wird, auch wenn der zeiger ihn verlassen hat).
            if consumed_by_hud:
                continue

            router.handle_mouse(event, ui_wants_mouse)
            app.camera.handle_event(
                event,
                ui_wants_mouse=ui_wants_mouse,
                ui_wants_keyboard=ui_wants_keyboard,
            )

        # -- schiff-steuerung ------------------------------------------------
        keys = pygame.key.get_pressed()
        reference_body = app.ui_state.reference_body
        if app.ship_control:
            app.ship_control.last_thrust_direction = None
            if app.ship is not None:
                setattr(app.ship, "last_thrust_direction", None)
            # Schiff-steuerung liest den tastaturzustand direkt (polling, keine
            # ereignisse) -- deshalb muss die eingabe-vorfahrt hier gesondert
            # geprueft werden, sonst fliegt das schiff waehrend einer
            # texteingabe in der dev-oberflaeche mit.
            if not ui_wants_keyboard:
                # DREHEN BLEIBT IMMER ERLAUBT, auch im zeitraffer -- es
                # aendert die bahn nicht.
                app.ship_control.handle_rotation(keys, frame_dt)
                # SCHUB NUR IN ECHTZEIT: im zeitraffer rueckt die welt je frame
                # um stunden bis tage vor, ein impuls je frame waere weder
                # dosierbar noch bildratenunabhaengig.
                # Handeingabe schlaegt den autopiloten. Geprueft werden alle
                # vier steuertasten: eine gedrehte nase macht die restliche
                # brenndauer ebenso ungueltig wie schub.
                if (app.maneuver_executor is not None
                        and app.maneuver_executor.is_active
                        and (keys[pygame.K_UP] or keys[pygame.K_DOWN]
                             or keys[pygame.K_LEFT] or keys[pygame.K_RIGHT])):
                    app.maneuver_executor.notify_manual_input()
                    print("MANEUVER: abgebrochen (handeingabe)")
                if app.thrust_allowed():
                    app.ship_control.apply_thrust(keys, frame_dt)

        # Raffung auf die bahn-zeitskala begrenzen. Das HUD blendet gesperrte
        # stufen bereits ab; das hier ist der riegel fuer PageUp/PageDown und
        # die dev-oberflaeche, die daran vorbeigehen.
        _clamp_warp(app)
        _apply_maneuver(app)
        # Horizont an die raffung anpassen (no-op, solange die stufe steht).
        _apply_horizon(app)

        # Simulation ZEITPROPORTIONAL vorruecken: je frame um
        #     camera.sim_dt * TICK_RATE * frame_dt
        # Die simulationsrate ist damit konstant camera.sim_dt * TICK_RATE
        # sim-sekunden pro echtsekunde, unabhaengig von der bildrate.
        #
        # Bewusst KEIN akkumulator mit festen ticks: die tick-rate liegt bei
        # der bildrate, feste ticks quantisierten gegen den vsync-jitter (frames
        # ohne und mit doppeltem vorruecken -> stotterndes schiff). Der
        # integrator ist adaptiv und world.step() zerlegt ohnehin in stuecke.
        #
        # frame_dt ist bereits auf max_frame_dt gekappt, ein stall kann also
        # keinen riesigen sprung einspeisen.
        sim_step = app.camera.sim_dt * app.tick_rate * frame_dt
        # Ein scharfgeschalteter knoten deckelt den schritt: sonst rueckt ein
        # zeitraffer-frame um stunden vor und der ganze brennvorgang faellt
        # zwischen zwei bilder. Waehrend des brennens haelt die decke die
        # kick-then-drift-naeherung klein (siehe ship/maneuver/executor.py).
        cap = (app.maneuver_executor.max_sim_seconds(app.world)
               if app.maneuver_executor is not None else None)
        if cap is not None:
            sim_step = min(sim_step, max(1e-9, cap))
        # Das delta-v des SCHRITTES anlegen, BEVOR er gegangen wird -- so
        # traegt die positionsintegration dieses schrittes es bereits.
        if app.maneuver_executor is not None:
            app.maneuver_executor.update(app.world, sim_step)
        app.world.step(sim_step, app.max_substep)

        # kamera mit echtem frame-delta fuer interaktives panning aktualisieren
        # (zoom/schwenk laufen ihren zielen geglaettet nach)
        app.camera.update(frame_dt, ui_wants_keyboard=ui_wants_keyboard)

        # -- orbit-prognose ---------------------------------------------------
        points = _update_predictor(app)
        # NACH dem predictor: die vorschau liest dessen linie, und mit der
        # linie des vorframes saesse sie einen frame lang daneben.
        _update_maneuver_preview(app)

        # Rendern. Der Orientierungs-snap wird INNERHALB von render() angewendet,
        # unmittelbar bevor der Schiffspfeil gezeichnet wird, mit demselben Frame
        # und derselben Frame-Zeit wie die gezeichneten prograde/normal-Vektoren
        # -- so ist die Nase exakt an diese Vektoren gebunden. ship_control und
        # frame_dt werden dafuer durchgereicht.
        app.renderer.render(
            app.world.body, app.camera, points, predictor=app.predictor,
            sim_time=app.world.time, reference_body=reference_body,
            ship_control=app.ship_control, real_dt=frame_dt,
            selected_body=app.ui_state.selected_body,
        )

        # Overlays NACH der welt und VOR dem swap (renderer.present() unten).
        #
        # Reihenfolge: spieler-HUD zuerst, entwicklerwerkzeuge darueber. Das
        # HUD landet damit hinter dem FXAA-resolve (render() ist fertig) --
        # ein kantenfilter ueber UI-text und 1px-rahmen wuerde beides
        # verschmieren.
        app.ui_root.render()

        app.dev_ctx.frame_dt = frame_dt
        app.dev_ctx.sim_step_s = app.camera.sim_dt * app.tick_rate * frame_dt
        app.devui.build(app.dev_ctx)
        app.devui.render()

        app.renderer.present()

        # Zeitreihen fuer die graphen der dev-oberflaeche (F1 -> Timing).
        #
        # NACH present(): render() setzt swap_or_present_ms auf 0.0 und erst
        # present() traegt den echten wert nach -- davor abgetastet waere
        # `render draw` konstant null. Das panel zeigt damit den stand des
        # VORIGEN frames, was bei 180 fps niemand sieht.
        #
        # Laeuft unbedingt, auch mit geschlossenem panel, damit der puffer
        # beim aufklappen schon gefuellt ist (kosten: tests/devui_timing_test.py).
        frame_ms = (time.perf_counter() - loop_t0) * 1000.0
        app.dev_ctx.sample_timings(frame_ms)

        if app.print_timings:
            timing.emit(app.renderer, app.predictor, frame_ms)

        frame_count += 1
        if app.max_frames > 0 and frame_count >= app.max_frames:
            running = False

    app.devui.shutdown()
    if getattr(app, 'maneuver_preview', None) is not None:
        # Der arbeiter der vorschau ist kein daemon-thread: ohne dieses
        # abmelden haengt der prozess am ende an einem laufenden neuaufbau.
        app.maneuver_preview.shutdown()
    app.window.close()


def _clamp_warp(app):
    """Raffung auf das begrenzen, was die BAHN noch aufloest."""
    fn = getattr(app.world, 'characteristic_timescale', None)
    if fn is None or app.ship is None:
        return
    try:
        t_char = fn(app.ship)
    except Exception:
        return
    app.camera.clamp_warp_to_timescale(
        t_char, app.tick_rate, app.warp_timescale_divisor,
        app.realtime_warp_max)


def _apply_maneuver(app):
    """Raffung auf echtzeit ziehen, sobald die zuendung naeherrueckt.

    Der spieler darf zum knoten hin raffen; kurz davor muss die welt aber
    zurueck auf echtzeit, weil der schub nur dort ueberhaupt erlaubt ist
    (app.thrust_allowed()). Ohne diesen griff steht der spieler bei 1 d/s
    vor einem brennvorgang, der nie beginnt.
    """
    executor = getattr(app, 'maneuver_executor', None)
    if executor is None or not executor.wants_realtime(app.world):
        return
    ceiling = app.realtime_warp_max / max(1.0, app.tick_rate)
    if app.camera.sim_dt > ceiling:
        app.camera.sim_dt = ceiling


def _update_maneuver_preview(app):
    """Die geplante bahn nachfuehren -- nur bei aenderung, nie je frame."""
    preview = getattr(app, 'maneuver_preview', None)
    if preview is not None and app.ship is not None:
        preview.maybe_rebuild(
            app.maneuver_plan, app.ship, app.world, app.predictor,
            app.ui_state.reference_body,
            app.maneuver_executor.a_max_sim(),
            app.maneuver_config['ramp_seconds'],
        )
    executor = getattr(app, 'maneuver_executor', None)
    # Die schubrichtung eines laufenden manoevers veroeffentlichen -- der
    # orientierungs-snap in render/ship.py holt sie sich dort ab.
    app.renderer.maneuver_burn_direction = (
        (executor.dir_x, executor.dir_y)
        if executor is not None and executor.is_active else None)
    app.renderer.maneuver_selected_index = int(
        getattr(app, 'selected_node_index', 0))


def _apply_horizon(app):
    grabbing = (app.hud is not None
                and getattr(app.hud, 'horizon', None) is not None
                and app.hud.horizon.is_grabbing)
    app.horizon.apply(app.predictor, app.warp_rate(), grabbing=grabbing)


def _update_predictor(app):
    """Die vorhersagelinie fortschreiben und ihre punkte liefern."""
    predictor = app.predictor
    if predictor.num_points > 0:
        target = app.ship
        if not target:
            target = next((b for b in app.world.body if not b.fixed), None)
        if target:
            # Im zeitraffer die kurve HALTEN statt jeden frame neu rechnen.
            # Siehe Predictor._hold_advance.
            predictor.set_hold(not app.thrust_allowed())
            if hasattr(predictor, 'set_view_scale'):
                # Das zoom-ZIEL einspeisen, nicht die nachlaufende skala: jede
                # skalenaenderung loest einen synchronen neuaufbau aus, und das
                # ziel ist waehrend der zoom-animation konstant -> genau ein
                # neuaufbau pro mausrad-raste.
                predictor.set_view_scale(app.camera.target_scale)
            predictor.update(target, app.world)
    return predictor.get_points()
