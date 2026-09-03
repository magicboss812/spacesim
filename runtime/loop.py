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
        # `frame_ms` ist die dauer von render() selbst (present() ruehrt es
        # nicht mehr an), also ist rend_calc genau das. Was zwischen render()
        # und present() gezeichnet wird -- spieler-HUD und dev-oberflaeche --
        # steht getrennt als ui_calc; frueher lief es unsichtbar unter
        # rend_calc und hat die zahl verdoppelt.
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
                # rotation: in echtzeit sanft. DREHEN BLEIBT IMMER ERLAUBT,
                # auch im zeitraffer -- es aendert die bahn nicht.
                app.ship_control.handle_rotation(keys, frame_dt)
                # SCHUB NUR IN ECHTZEIT. Oberhalb der untersten zeitraffer-
                # stufe rueckt die welt je frame um stunden bis tage vor; ein
                # impuls "einmal pro frame" waere dort weder dosierbar noch
                # reproduzierbar (er haenge an der bildrate), und er macht die
                # gehaltene vorhersage in jedem frame ungueltig. Deshalb ist
                # der schub gesperrt, solange gerafft wird -- der spieler geht
                # zum manoevrieren auf die unterste stufe zurueck.
                if app.thrust_allowed():
                    app.ship_control.apply_thrust(keys, frame_dt)

        # Raffung auf die bahn-zeitskala begrenzen. Das HUD blendet gesperrte
        # stufen bereits ab; das hier ist der riegel fuer PageUp/PageDown und
        # die dev-oberflaeche, die daran vorbeigehen.
        _clamp_warp(app)
        # Horizont an die raffung anpassen (no-op, solange die stufe steht).
        _apply_horizon(app)

        # Simulation ZEITPROPORTIONAL vorruecken: je frame um
        #     camera.sim_dt * TICK_RATE * frame_dt
        # Die simulationsrate ist damit konstant camera.sim_dt * TICK_RATE
        # sim-sekunden pro echtsekunde, unabhaengig von der bildrate.
        #
        # Ein akkumulator mit FESTEN ticks waere die lehrbuch-loesung, ist hier
        # aber falsch: die tick-rate liegt bei der bildrate, also quantisiert
        # der akkumulator gegen den vsync-jitter. Gemessen ueber 600 frames:
        # 9 frames ruecken GAR NICHT vor, 7 frames DOPPELT (16.4 % streuung
        # statt 4.2 %). Sichtbar wird das als stotterndes schiff, das gegen die
        # jeden frame neu gezeichnete predictor-linie springt. Der integrator
        # ist adaptiv und world.step() zerlegt ohnehin in stuecke, ein
        # variables aeusseres dt ist also unproblematisch.
        #
        # frame_dt ist bereits auf max_frame_dt gekappt, ein stall kann also
        # keinen riesigen sprung einspeisen.
        app.world.step(app.camera.sim_dt * app.tick_rate * frame_dt,
                       app.max_substep)

        # kamera mit echtem frame-delta fuer interaktives panning aktualisieren
        # (zoom/schwenk laufen ihren zielen geglaettet nach)
        app.camera.update(frame_dt, ui_wants_keyboard=ui_wants_keyboard)

        # -- orbit-prognose ---------------------------------------------------
        points = _update_predictor(app)

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

        # Overlays NACH der welt und VOR dem swap. render() macht den swap
        # nicht mehr selbst -- das uebernimmt renderer.present() unten.
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
        # Laeuft unbedingt, auch mit geschlossenem panel: ein puffer, der nur
        # gefuellt wird, waehrend man hinschaut, ist beim aufklappen leer.
        # Kostet dafuer gemessene 1.0 us je frame (0.02 % eines 5.6-ms-frames),
        # siehe tests/devui_timing_test.py.
        frame_ms = (time.perf_counter() - loop_t0) * 1000.0
        app.dev_ctx.sample_timings(frame_ms)

        if app.print_timings:
            timing.emit(app.renderer, app.predictor, frame_ms)

        frame_count += 1
        if app.max_frames > 0 and frame_count >= app.max_frames:
            running = False

    app.devui.shutdown()
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
            # Im zeitraffer die kurve HALTEN statt jeden frame neu rechnen --
            # sonst zieht _anchor_first_point sie je frame um die volle
            # bahnbewegung starr mit und sie zittert. Siehe
            # Predictor._hold_advance.
            predictor.set_hold(not app.thrust_allowed())
            if hasattr(predictor, 'set_view_scale'):
                # WICHTIG: das zoom-ZIEL einspeisen, nicht die gerade
                # nachlaufende skala. set_view_scale() setzt bei jeder
                # aenderung > snapshot_view_rel_tol (1e-6) das flag
                # _view_scale_changed, was in Predictor.update() einen
                # SYNCHRONEN _compute_full() im hauptthread ausloest. Mit der
                # animierten skala waere das ein voller neuaufbau der
                # trajektorie in JEDEM frame einer zoom-animation. Das ziel ist
                # waehrend der animation konstant -> genau ein neuaufbau pro
                # mausrad-raste.
                predictor.set_view_scale(app.camera.target_scale)
            predictor.update(target, app.world)
    return predictor.get_points()
