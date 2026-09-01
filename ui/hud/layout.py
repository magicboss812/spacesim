"""Aufbau des kompletten HUDs und die verankerung seiner gruppen.

VIER BLOECKE, VIER ECKEN -- und die mitte bleibt frei. Das ist die
eigentliche aenderung gegenueber der ersten fassung, in der acht einzelne
elemente ueber den schirm verteilt lagen und die untere bildmitte so hoch
baute, dass sie auf der bahn sass:

    oben links   schiffs-plakette, koerperwaehler, ziel-block
    oben rechts  zeitraffer mit missionsuhr, darunter die system-karte
    unten mitte  der navball-block (kurs, schub, steigrate, AP/PE)
                 und rechts daneben, angedockt, die snap-rosette
    unten links  bezugsrahmen und zoom

Verankert wird IMMER an einer ecke oder kante, nie an absoluten
koordinaten -- das ist der grund, warum das layout eine freie fenstergroesse
ueberhaupt ueberlebt.

RESPONSIVES VERHALTEN: unterhalb von theme.compact_breakpoint klappt der
ziel-block zu einer schmalen leiste zusammen und die snap-rosette
schrumpft. Der navball-block bleibt in jeder groesse -- er ist das einzige
element, ohne das man nicht fliegen kann; die koerperliste bleibt, weil sie
der einzige mausweg zum bezugskoerper ist.
"""

import math

from ..core import (
    BOTTOM_CENTER,
    BOTTOM_LEFT,
    CENTER_LEFT,
    TOP_LEFT,
    TOP_RIGHT,
)
from ..widgets import Stack
from .apsis_tooltip import ApsisTooltip
from .body_browser import BodyBrowser
from .controls import SegmentBar, SnapRosette, WarpBar, ZoomButtons
from .navball import WIDTH as NAVBALL_WIDTH, NavballCluster
from .panels import IconRail, ShipBadge, build_target_panel
from .system_map import SystemMap
from .telemetry import Telemetry
from . import chrome

# Zeitraffer-stufen als SIM-ZEIT JE ECHTSEKUNDE.
#
# Ein vielfaches ("10000x") sagt bei bahnmechanik nichts; "1h/s" -- eine
# simulierte stunde je echtsekunde -- beantwortet die eigentliche frage
# sofort, weil man umlaufzeiten in stunden und tagen denkt.
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

VIEW_MODES = ('SRF', 'ORB', 'TGT')

#: Rand zum bildschirm. Ein wert, ueberall -- ungleiche raender sind das,
#: was eine oberflaeche "irgendwie zusammengeschoben" aussehen laesst.
MARGIN = 16

#: Abstand zwischen navball-block und snap-rosette. Klein genug, dass die
#: beiden als EINE instrumentenreihe gelesen werden.
DOCK_GAP = 12


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
        # Schwelle, ab der der schub gesperrt ist -- der schubbogen im
        # navball-block zeigt das an.
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

        # --- oben links: plakette, koerperwaehler, ziel ------------------
        self.badge = root.add(ShipBadge(
            telemetry, anchor=TOP_LEFT, offset=(MARGIN, MARGIN),
        ))
        # Die koerperliste sitzt direkt unter der plakette, weil dort auch
        # der aktive bezugskoerper steht -- knopf und angezeigter wert
        # gehoeren zusammen. Sie ist in JEDER fenstergroesse erreichbar:
        # ohne sie liesse sich der bezugskoerper ueberhaupt nur noch mit
        # der taste R wechseln.
        self.body_browser = root.add(BodyBrowser(
            telemetry, self.ui_state, side='left',
            anchor=TOP_LEFT, offset=(MARGIN, MARGIN + 40),
        ))
        self.target = root.add(build_target_panel(
            telemetry, anchor=TOP_LEFT, offset=(MARGIN, MARGIN + 82),
        ))
        # Der schwebezettel an den Ap/Pe-rauten. Er verbraucht die maus NICHT
        # und steht deshalb ausserhalb jeder gruppe -- sein platz kommt aus
        # der weltposition des markers, nicht aus der verankerung.
        self.apsis_tooltip = root.add(ApsisTooltip(telemetry))
        self.target_rail = root.add(IconRail(
            [{'key': 'TG'}, {'key': 'D'}, {'key': 'V'}],
            color_role='target', anchor=CENTER_LEFT, offset=(MARGIN, 0),
        ))

        # --- oben rechts: zeitraffer + missionsuhr -----------------------
        self.warp = root.add(WarpBar(
            telemetry,
            options=[label for _, label in WARP_STEPS],
            value=self._warp_index,
            on_select=self._set_warp,
            enabled=self._warp_step_enabled,
            color_role='warp', caption=chrome.tab_text('TIME', 'WARP'),
            role='warp', min_option_width=34, cumulative=True,
            anchor=TOP_RIGHT, offset=(MARGIN, MARGIN), z=10,
        ))
        # Die karte haengt UNTER dem zeitraffer, und zwar an dessen fertigem
        # rechteck statt an einem festen y-abstand: die hoehe der
        # zeitraffer-leiste folgt schriftgroesse und notch-tab und aendert
        # sich mit der UI-skala (siehe SystemMap.layout). Verankert ist sie
        # trotzdem oben rechts -- so waechst sie beim ausfahren nach links
        # und unten, weg vom bildrand.
        self.system_map = root.add(SystemMap(
            telemetry, self.ui_state, self.camera, below=self.warp,
            anchor=TOP_RIGHT, offset=(MARGIN, MARGIN),
        ))

        # --- unten mitte: der navball-block ------------------------------
        self.navball = root.add(NavballCluster(
            telemetry, ship_control, anchor=BOTTOM_CENTER,
            offset=(0, MARGIN),
        ))
        # Die rosette dockt RECHTS an den block an. Verankert ist sie an
        # derselben unteren mitte, nur um die halbe blockbreite plus den
        # dock-abstand nach rechts geschoben -- so bleibt der navball
        # bildschirmmittig und die rosette klebt trotzdem an ihm.
        snap_offset = NAVBALL_WIDTH * 0.5 + DOCK_GAP + SnapRosette.SIZE * 0.5
        self.snaps = root.add(SnapRosette(
            telemetry, ship_control, anchor=BOTTOM_CENTER,
            offset=(snap_offset, MARGIN + 22),
        ))
        # Bequemer durchgriff: der ring ist ein KIND des blocks, aber
        # tastatur-tests und die hauptschleife wollen ihn direkt.
        self.ring = self.navball.ring
        self.snaps_compact = root.add(SnapRosette(
            telemetry, ship_control, compact=True, anchor=BOTTOM_CENTER,
            offset=(NAVBALL_WIDTH * 0.5 + DOCK_GAP
                    + SnapRosette.SIZE * 0.76 * 0.5, MARGIN + 22),
        ))

        # --- unten links: bezugsrahmen und zoom --------------------------
        self.left_stack = root.add(Stack(
            gap=8, align='start', anchor=BOTTOM_LEFT, offset=(MARGIN, MARGIN),
        ))
        self.frames = self.left_stack.add(SegmentBar(
            options=VIEW_MODES,
            value=self.ui_state.view_mode,
            on_select=self._set_view_mode,
            color_role='frame', role='button_sm',
            caption=chrome.tab_text('FRAME'), min_option_width=44,
        ))
        self.zoom = self.left_stack.add(ZoomButtons(telemetry, self.camera))
        self.zoom_compact = self.left_stack.add(
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

        self.target.visible = wide
        self.snaps.visible = wide
        self.zoom.visible = wide
        # In der schmalen fassung faellt die karte weg -- sie ist ein
        # ueberblick, kein fluginstrument, und in einem 800 px breiten
        # fenster nimmt sie den platz weg, den die bahn braucht.
        self.system_map.visible = wide
        if not wide:
            self.system_map.expanded = False

        self.target_rail.visible = not wide
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
