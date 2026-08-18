"""Aufbau des kompletten HUDs und die verankerung seiner gruppen.

Die anker und abstaende stammen 1:1 aus dem entwurf (16 px zu den seiten,
14 px oben und unten). Verankert wird IMMER an einer ecke oder kante, nie
an absoluten koordinaten -- das ist der grund, warum das layout eine freie
fenstergroesse ueberhaupt ueberlebt.

RESPONSIVES VERHALTEN: unterhalb von theme.compact_breakpoint (900 design-
einheiten breite) klappen die beiden seitenpanels zu 46 px schmalen leisten
zusammen, der schubregler stellt sich hochkant und snap- wie zoomknoepfe
schrumpfen auf reine symbolflaechen. Der attitude-ring in der mitte bleibt
in jeder groesse -- er ist das einzige element, ohne das man nicht fliegen
kann; die koerperliste bleibt, weil sie der einzige mausweg zum
bezugskoerper ist.
"""

import math

from ..core import (
    BOTTOM_CENTER,
    BOTTOM_LEFT,
    BOTTOM_RIGHT,
    CENTER_LEFT,
    CENTER_RIGHT,
    TOP_LEFT,
    TOP_RIGHT,
)
from ..widgets import Stack
from .attitude import AttitudeRing, VelocityReadout
from .body_browser import BodyBrowser
from .controls import PaletteButton, SegmentBar, SnapGrid, ThrottleControl, ZoomButtons
from .panels import (
    IconRail,
    ShipBadge,
    build_elements_panel,
    build_target_panel,
)
from .telemetry import Telemetry

# Zeitraffer-stufen als SIM-ZEIT JE ECHTSEKUNDE.
#
# Der entwurf beschriftet die stufen mit 1x/5x/50x/1k/10k. Das waere hier aus
# zwei gruenden falsch:
#
# 1. "1x" ist gar nicht erreichbar. Die simulation rueckt pro tick um
#    camera.sim_dt sim-sekunden vor, und config.json setzt min_sim_dt = 1.0.
#    Bei 60 ticks/s ist die LANGSAMSTE einstellung damit bereits 60-fache
#    echtzeit -- ein knopf mit der aufschrift "1x" wuerde schlicht luegen.
# 2. Ein vielfaches sagt bei bahnmechanik nichts. "10000x" beantwortet die
#    eigentliche frage nicht; "1h/s" (eine simulierte stunde je echtsekunde)
#    beantwortet sie sofort, weil man umlaufzeiten in stunden und tagen denkt.
#
# Die erste stufe entspricht genau min_sim_dt bei 60 fps und ist damit immer
# erreichbar; _set_warp klemmt zusaetzlich auf die kamera-grenzen.
#
# Die oberen drei stufen (30d/s, 100d/s, 1y/s) kamen 2026-08-18 dazu. Ohne sie
# ist das eigentliche thema der arbeit nicht spielbar: ein Hohmann-transfer zu
# Pluto dauert ~45 jahre, bei 7 d/s also 75 minuten echtzeit. Bei 1 y/s sind es
# 45 sekunden.
#
# Bezahlt wurden sie NICHT mit bildrate, sondern mit der schrittweiten-decke --
# siehe world.set_warp_step_ceiling(). Gemessen bei 365 d/s, 28 koerper,
# 180 fps: welt+predictor 172.9 ms -> 4.3 ms.
WARP_STEPS = (
    (60.0, '1m/s'),
    (600.0, '10m/s'),
    (3600.0, '1h/s'),
    (86400.0, '1d/s'),
    (604800.0, '7d/s'),
    (2592000.0, '30d/s'),
    (8640000.0, '100d/s'),
    (31557600.0, '1y/s'),
)

VIEW_MODES = ('SURFACE', 'ORBITAL', 'TARGET')


class Hud:
    """Baut den widget-baum und haelt ihn pro frame aktuell.

    Besitzt die telemetrie und die responsive umschaltung; die hauptschleife
    ruft nur noch update() auf.
    """

    def __init__(self, ui_root, world, ship, ship_control, camera, renderer,
                 predictor, ui_state, tick_rate=60.0, realtime_warp_max=60.0,
                 warp_timescale_divisor=3.0):
        self.root = ui_root
        self.ctx = ui_root.ui
        self.camera = camera
        self.ui_state = ui_state
        self.telemetry = Telemetry(
            world, ship, ship_control, camera, renderer, predictor, ui_state,
            tick_rate=tick_rate,
        )
        # Schwelle, ab der der schub gesperrt ist -- der schubregler zeigt
        # das an (siehe ThrottleControl.draw).
        self.telemetry.realtime_warp_max = float(realtime_warp_max)
        # Dieselbe zahl wie der riegel in test.py -- sonst blendet das HUD
        # andere stufen ab als die hauptschleife zulaesst.
        self.telemetry.warp_timescale_divisor = float(warp_timescale_divisor)
        self._wide = None
        self._build(ship_control)

    # ----------------------------------------------------------------- aufbau

    def _build(self, ship_control):
        root = self.root
        telemetry = self.telemetry
        theme = self.ctx.theme

        # --- oben links: schiffs-plakette, darunter die koerperliste -----
        self.badge = root.add(ShipBadge(
            telemetry, anchor=TOP_LEFT, offset=(16, 14),
        ))
        # Die koerperliste sitzt direkt unter der plakette, weil dort auch
        # der aktive bezugskoerper steht -- knopf und angezeigter wert
        # gehoeren zusammen. Sie ist in JEDER fenstergroesse erreichbar:
        # ohne sie liesse sich der bezugskoerper ueberhaupt nur noch mit
        # der taste R wechseln.
        self.body_browser = root.add(BodyBrowser(
            telemetry, self.ui_state, side='left',
            anchor=TOP_LEFT, offset=(16, 56),
        ))

        # --- oben rechts: zeitraffer + palette ---------------------------
        top_right = root.add(Stack(
            gap=9, align='end', anchor=TOP_RIGHT, offset=(16, 14), z=10,
        ))
        warp_row = top_right.add(Stack(gap=8, horizontal=True, align='center'))
        self.warp = warp_row.add(SegmentBar(
            options=[label for _, label in WARP_STEPS],
            value=self._warp_index,
            on_select=self._set_warp,
            enabled=self._warp_step_enabled,
            color_role='warp', caption='WARP', role='warp',
            min_option_width=30, pad_x=6, pad_y=7, gap=2, container_pad=6,
        ))
        self.palette_button = warp_row.add(PaletteButton(theme))

        # --- seitenpanels (breit) bzw. leisten (schmal) ------------------
        self.elements = root.add(build_elements_panel(
            telemetry, anchor=CENTER_LEFT, offset=(16, 0),
        ))
        self.target = root.add(build_target_panel(
            telemetry, anchor=CENTER_RIGHT, offset=(16, 0),
        ))
        self.elements_rail = root.add(IconRail(
            [{'key': 'AP'}, {'key': 'PE'}, {'key': 'E'}, {'key': 'T'}],
            color_role='elem', anchor=CENTER_LEFT, offset=(16, 0),
        ))
        self.target_rail = root.add(IconRail(
            [{'key': 'TG'}, {'key': 'D'}, {'key': 'V'}],
            color_role='target', anchor=CENTER_RIGHT, offset=(16, 0),
        ))

        # --- unten links: schub -----------------------------------------
        self.throttle = root.add(ThrottleControl(
            telemetry, anchor=BOTTOM_LEFT, offset=(16, 14),
        ))
        self.throttle_compact = root.add(ThrottleControl(
            telemetry, compact=True, anchor=BOTTOM_LEFT, offset=(16, 14),
        ))

        # --- unten mitte: rahmenwahl ueber dem lagemesser ----------------
        center = root.add(Stack(
            gap=9, align='center', anchor=BOTTOM_CENTER, offset=(0, 14),
        ))
        self.frames = center.add(SegmentBar(
            options=VIEW_MODES,
            value=self.ui_state.view_mode,
            on_select=self._set_view_mode,
            color_role='frame', role='button_sm',
            pad_x=14, pad_y=8, gap=5, container_pad=5,
        ))
        self.ring = center.add(AttitudeRing(telemetry, ship_control))
        # Der geschwindigkeitswert sitzt UNTER dem ring, nicht darin: die
        # nadel laeuft aus der ringmitte heraus und schnitt sonst quer durch
        # die zahl (siehe VelocityReadout).
        self.velocity = center.add(VelocityReadout(telemetry))

        # --- unten rechts: autopilot + zoom ------------------------------
        self.right_stack = root.add(Stack(
            gap=8, align='end', anchor=BOTTOM_RIGHT, offset=(16, 14),
        ))
        self.snaps = self.right_stack.add(SnapGrid(telemetry, ship_control))
        self.snaps_compact = self.right_stack.add(
            SnapGrid(telemetry, ship_control, compact=True)
        )
        self.zoom = self.right_stack.add(ZoomButtons(telemetry, self.camera))
        self.zoom_compact = self.right_stack.add(
            ZoomButtons(telemetry, self.camera, compact=True)
        )

    # ----------------------------------------------------------------- ablauf

    def update(self):
        """Einmal pro frame VOR ui_root.begin_frame() aufrufen.

        Erst abtasten, dann umschalten: die sichtbarkeit haengt nur an der
        fensterbreite, aber die widgets lesen im selben frame bereits die
        frischen telemetriewerte.
        """
        self.telemetry.sample()
        self._apply_responsive()

    def _apply_responsive(self):
        wide = self.ctx.width >= self.ctx.px(self.ctx.theme.compact_breakpoint)
        if wide == self._wide:
            return
        self._wide = wide

        self.elements.visible = wide
        self.target.visible = wide
        self.throttle.visible = wide
        self.snaps.visible = wide
        self.zoom.visible = wide

        self.elements_rail.visible = not wide
        self.target_rail.visible = not wide
        self.throttle_compact.visible = not wide
        self.snaps_compact.visible = not wide
        self.zoom_compact.visible = not wide

    # ------------------------------------------------------------- aktionen

    def _warp_index(self):
        """Naechstliegende zeitraffer-stufe zum AKTUELLEN sim_dt.

        Zurueckgelesen statt gemerkt: PageUp/PageDown und das dev-panel
        verstellen sim_dt ebenfalls, und der knopf soll dann mitwandern.
        """
        current = max(self.telemetry.warp_factor, 1e-9)
        best = 0
        best_error = None
        for index, (factor, _label) in enumerate(WARP_STEPS):
            # Vergleich im LOGARITHMUS: die stufen liegen vier zehnerpotenzen
            # auseinander, linear waere die groesste immer die "naechste".
            error = abs(math.log(factor) - math.log(current))
            if best_error is None or error < best_error:
                best_error = error
                best = index
        return best

    def _warp_step_enabled(self, index):
        """Ist diese stufe momentan erlaubt? (bahn-zeitskala, siehe
        Telemetry._sample_warp_limit)"""
        if index < 0 or index >= len(WARP_STEPS):
            return False
        return self.telemetry.warp_step_allowed(WARP_STEPS[index][0])

    def _set_warp(self, index):
        index = max(0, min(index, len(WARP_STEPS) - 1))
        # Gesperrte stufe: nichts tun. Der knopf ist bereits abgeblendet, ein
        # klick darf die bahn nicht zerlegen.
        if not self._warp_step_enabled(index):
            return
        factor = WARP_STEPS[index][0]
        camera = self.camera
        if camera is None:
            return
        tick_rate = max(1.0, self.telemetry.tick_rate)
        target = factor / tick_rate
        low = float(getattr(camera, 'min_sim_dt', 1e-6) or 1e-6)
        high = float(getattr(camera, 'max_sim_dt', 1e12) or 1e12)
        camera.sim_dt = max(low, min(high, target))

    def _set_view_mode(self, index):
        self.ui_state.apply_view_mode(
            ('surface', 'orbital', 'target')[max(0, min(index, 2))]
        )
