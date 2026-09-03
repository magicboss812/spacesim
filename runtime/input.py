"""Tastenbelegung und die klick-geste.

Stand als eine 100-zeilige if/elif-kette mitten in der ereignisschleife von
`test.py`. Die vorfahrt (custom-UI -> ImGui -> welt) entscheidet weiterhin die
schleife in `loop.py`; hier steht nur, WAS eine taste tut.

Die volle tastenreferenz steht in `.claude/rules/camera-input.md` und im HUD --
das HUD liest seine werte aus der simulation zurueck, die beiden koennen
deshalb nicht auseinanderlaufen.
"""
import pygame

from physics.reference_frames import (
    BODY_CENTRED_BODY_DIRECTION,
    BODY_CENTRED_NON_ROTATING,
)
from ship.horizon import warp_length_mult

# Erst ein zeigerweg unter dieser laenge gilt als klick (siehe handle_mouse).
CLICK_SLOP_PX = 4.0

_SNAP_KEYS = {
    pygame.K_i: 'prograde',
    pygame.K_k: 'retrograde',
    pygame.K_j: 'normal_in',
    pygame.K_l: 'antinormal_out',
}


class InputRouter:
    """Uebersetzt ereignisse in aenderungen an welt, kamera und predictor."""

    def __init__(self, app):
        self.app = app
        self._click_press_pos = None

    # -- maus ---------------------------------------------------------------

    def handle_world_click(self, screen_pos):
        """Auswahl / anflug. Aendert WEDER bezugskoerper NOCH bezugsrahmen."""
        app = self.app
        index = app.renderer.pick_body(screen_pos, app.world.body, app.camera)
        if index is None:
            app.ui_state.clear_selection()
            return
        if index == app.ui_state.selected_index:
            # Zweiter klick auf den bereits gewaehlten koerper: hinfliegen.
            app.camera.focus_on(app.world.body[index])
            return
        app.ui_state.select_body(index)

    def handle_mouse(self, event, ui_wants_mouse):
        """Linke maustaste: koerper auswaehlen / anfliegen.

        Die kamera zieht mit der mittleren/rechten taste, hier gibt es also
        keinen streit um die geste. Die geste wird ueber DOWN/UP zusammengesetzt
        statt auf MOUSEBUTTONDOWN allein zu reagieren: sonst wuerde jeder
        schwenk-anfang, der zufaellig auf einem koerper beginnt, die auswahl
        umwerfen.
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._click_press_pos = (
                None if ui_wants_mouse
                else (float(event.pos[0]), float(event.pos[1]))
            )
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._click_press_pos is not None:
                dx = float(event.pos[0]) - self._click_press_pos[0]
                dy = float(event.pos[1]) - self._click_press_pos[1]
                if (dx * dx + dy * dy) <= CLICK_SLOP_PX * CLICK_SLOP_PX:
                    self.handle_world_click(event.pos)
            self._click_press_pos = None

    # -- tastatur -----------------------------------------------------------

    def handle_keydown(self, event):
        """Eine taste anwenden. Liefert False, wenn das spiel enden soll."""
        app = self.app
        key = event.key

        if key == pygame.K_ESCAPE:
            return False

        # Taste P fuer Predictive Orbit umschalten
        if key == pygame.K_p:
            if app.predictor.num_points > 0:
                app.predictor.reset()
            else:
                app.predictor.set_num_points(app.predictor_toggle_points)

        # Taste O: bahnlinien der koerper umschalten
        elif key == pygame.K_o:
            app.renderer.orbit_lines_enabled = not app.renderer.orbit_lines_enabled
            print(f"ORBIT LINES: {'on' if app.renderer.orbit_lines_enabled else 'off'}")

        # Taste E: epizykel-modus umschalten (zentriert auf kameraziel oder
        # Fokuskoerper)
        elif key == pygame.K_e:
            self._toggle_epicycles()

        # R / 1 / 2 / T schreiben denselben zustand wie die HUD-bedienelemente
        # (ui/state.py) -- deshalb kein direktes setzen mehr, sondern die
        # methoden des zustands. Das anwenden loest die aenderungs-
        # benachrichtigung aus.
        elif key == pygame.K_r:
            app.ui_state.cycle_reference()

        elif key == pygame.K_1:
            app.ui_state.set_frame_extension(BODY_CENTRED_NON_ROTATING)

        elif key == pygame.K_2:
            app.ui_state.set_frame_extension(BODY_CENTRED_BODY_DIRECTION)

        elif key == pygame.K_t:
            if not app.ui_state.toggle_target_overlay():
                print("FRAME: no ship available for target overlay")

        # I/K/J/L: orientierungs-snap (rastender autopilot) umschalten.
        # Tippen rastet ein, erneutes Tippen loest; render() haelt die Nase
        # smooth an den gezeichneten orbital-vektoren im aktiven Frame.
        elif key in _SNAP_KEYS:
            if app.ship_control is not None:
                app.ship_control.toggle_snap(_SNAP_KEYS[key])
                print(f"SNAP: {app.ship_control.snap_mode or 'off'}")

        # predictor-steuerung (zwei entkoppelte regler):
        #   '+' / '-' -> look-ahead HORIZONT (predictor.length). Das ist der
        #               kosten-regler: kosten ~ integrierter bogen ~ horizont.
        #   '9' / '0' -> punkt-ABSTAND (predictor.precision). Rein kosmetisch:
        #               mehr/weniger gezeichnete punkte im festen horizont,
        #               gleiche rechenzeit und gleiche genauigkeit.
        #
        # Eigenes `if` statt eines weiteren `elif`: die zeichen kommen aus
        # event.unicode, nicht aus event.key, und duerfen die kette oben nicht
        # verschlucken.
        ch = event.unicode
        if ch == '+' or key == pygame.K_KP_PLUS:
            self._step_horizon(app.horizon.length_step)
        elif ch == '-' or key == pygame.K_KP_MINUS:
            self._step_horizon(1.0 / app.horizon.length_step)
        elif ch == '9':
            # praezision erhoehen (feiner = kleinere abstaende)
            self._set_precision(app.predictor.precision / app.precision_step)
        elif ch == '0':
            self._set_precision(app.predictor.precision * app.precision_step)

        return True

    # -- die etwas laengeren einzelfaelle ------------------------------------

    def _toggle_epicycles(self):
        app = self.app
        w = app.world
        center = app.camera.target
        if center is None:
            center = next((b for b in w.body
                           if getattr(b, 'name', '').lower() in app.focus_aliases),
                          None)
        if center is None:
            print("EPICYCLE: No center found (camera target or Earth).")
            return
        if (getattr(w, '_epicycle_enabled', False)
                and getattr(w, '_epicycle_center', None) is center):
            w.disable_epicycles()
            print("EPICYCLE: disabled")
        else:
            w.enable_epicycles(center)
            print(f"EPICYCLE: enabled (center={center.name})")

    def _step_horizon(self, factor):
        """'+'/'-' verstellen den MANUELLEN faktor, nicht die laenge direkt.

        Sonst wuerde die horizont-regel die eingabe im naechsten frame wieder
        ueberschreiben.
        """
        app = self.app
        app.horizon.step_mult(factor, app.predictor)
        app.horizon.apply(app.predictor, app.warp_rate())
        app.predictor.reset()
        print(f"PREDICTOR: length set to {app.predictor.length} "
              f"(manuell x{app.horizon.manual_mult:g}, "
              f"raffung x{warp_length_mult(app.warp_rate()):g})")

    def _set_precision(self, new_precision):
        app = self.app
        app.predictor.set_precision(
            max(app.predictor_min_precision, new_precision))
        app.predictor.reset()
        print(f"PREDICTOR: precision set to {app.predictor.precision}")
