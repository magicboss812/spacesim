"""Aufbau des kompletten HUDs und die verankerung seiner gruppen.

VIER BLOECKE, VIER ECKEN -- und die mitte bleibt frei fuer die bahn:

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
from ..widgets.rate_slider import HorizonSlider
from .apsis_tooltip import ApsisTooltip
from .body_browser import BodyBrowser
from .maneuver import (ManeuverAxesBlock, ManeuverBurnBlock, ManeuverGizmo,
                       ManeuverNodesBar, ManeuverPlanBlock)
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
# Die oberen drei stufen (30d/s, 100d/s, 1y/s) machen lange transfers
# spielbar: ein Hohmann-transfer zu Pluto dauert ~45 jahre, bei 7 d/s also
# 75 minuten echtzeit, bei 1 y/s 45 sekunden. Bezahlbar sind sie ueber die
# schrittweiten-decke, siehe world.set_warp_step_ceiling().
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
                 warp_timescale_divisor=3.0,
                 horizon_mult_get=None, horizon_mult_set=None,
                 horizon_mult_min=0.25, horizon_mult_max=4.0,
                 horizon_sweep_s=2.5,
                 maneuver_plan=None, maneuver_preview=None,
                 maneuver_executor=None, maneuver_selected_get=None,
                 maneuver_selected_set=None, maneuver_add=None,
                 maneuver_delete=None, maneuver_execute=None,
                 maneuver_dv_step_fine=1.0, maneuver_dv_step_coarse=10.0,
                 maneuver_dv_rate=260.0, maneuver_handle_travel_px=96.0,
                 maneuver_length_get=None, maneuver_length_set=None,
                 maneuver_length_min=0.25, maneuver_length_max=32.0,
                 maneuver_length_sweep_s=2.5):
        self.root = ui_root
        self.ctx = ui_root.ui
        self.camera = camera
        self.ui_state = ui_state
        self.telemetry = Telemetry(
            world, ship, ship_control, camera, renderer, predictor, ui_state,
            tick_rate=tick_rate,
            maneuver_plan=maneuver_plan,
            maneuver_preview=maneuver_preview,
            maneuver_executor=maneuver_executor,
            maneuver_selected_get=maneuver_selected_get,
            maneuver_selected_set=maneuver_selected_set,
            maneuver_add=maneuver_add,
            maneuver_delete=maneuver_delete,
            maneuver_execute=maneuver_execute,
            maneuver_dv_step_fine=maneuver_dv_step_fine,
            maneuver_dv_step_coarse=maneuver_dv_step_coarse,
        )
        self._maneuver_dv_rate = float(maneuver_dv_rate)
        self._maneuver_travel_px = float(maneuver_handle_travel_px)
        self._maneuver_length_get = maneuver_length_get
        self._maneuver_length_set = maneuver_length_set
        self._maneuver_length_min = float(maneuver_length_min)
        self._maneuver_length_max = float(maneuver_length_max)
        self._maneuver_length_sweep_s = float(maneuver_length_sweep_s)
        # Schwelle, ab der der schub gesperrt ist -- der schubbogen im
        # navball-block zeigt das an.
        self.telemetry.realtime_warp_max = float(realtime_warp_max)
        # Dieselbe zahl wie der riegel der hauptschleife -- sonst blendet das HUD
        # andere stufen ab als die hauptschleife zulaesst.
        self.telemetry.warp_timescale_divisor = float(warp_timescale_divisor)
        self._horizon_mult_get = horizon_mult_get
        self._horizon_mult_set = horizon_mult_set
        self._horizon_mult_min = float(horizon_mult_min)
        self._horizon_mult_max = float(horizon_mult_max)
        self._horizon_sweep_s = float(horizon_sweep_s)
        self.horizon = None
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
        # sonst liesse sich der bezugskoerper nur mit der taste R wechseln.
        self.body_browser = root.add(BodyBrowser(
            telemetry, self.ui_state,
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

        # --- das manoever-werkzeug IM navball-raster ---------------------
        #
        # Vier plaettchen in den vier freien feldern des blocks: ueber der
        # ORB-flanke die brenndaten, unter dem THR-streifen die beiden
        # delta-v-achsen, ueber der ALT-flanke die reichweite der vorschau,
        # unter dem V/S-streifen die knotenwahl. Sie bekommen KEINE eigene
        # verankerung -- `layout()` liest breite und x-lage aus
        # `NavballCluster.flank_rect()` / `strip_rect()`, sonst waeren es
        # zwei layouts fuer eine flanke. Begruendung: ui/hud/maneuver.py.
        self.maneuver_burn = root.add(ManeuverBurnBlock(
            telemetry, navball=self.navball))
        self.maneuver_axes = root.add(ManeuverAxesBlock(
            telemetry, navball=self.navball))
        self.maneuver_nodes = root.add(ManeuverNodesBar(
            telemetry, navball=self.navball))
        if (self._maneuver_length_get is not None
                and self._maneuver_length_set is not None):
            self.maneuver_length = root.add(ManeuverPlanBlock(
                telemetry, navball=self.navball,
                value=self._maneuver_length_get,
                on_change=self._maneuver_length_set,
                minimum=self._maneuver_length_min,
                maximum=self._maneuver_length_max,
                sweep_seconds=self._maneuver_length_sweep_s,
            ))
        else:
            self.maneuver_length = None
        # Der ziehgriff steht AUSSERHALB jeder gruppe -- sein platz kommt aus
        # der weltposition des knotens, nicht aus der verankerung. Wie der
        # Ap/Pe-schwebezettel.
        self.gizmo = root.add(ManeuverGizmo(
            telemetry, dv_rate=self._maneuver_dv_rate,
            travel_px=self._maneuver_travel_px))

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

        # Der vorhersage-horizont: ein mittenzentrierter raten-regler. Nach
        # rechts ziehen verlaengert die gezeichnete linie, nach links
        # verkuerzt sie -- die auslenkung ist die geschwindigkeit. Er
        # aendert NUR predictor.set_display_length (O(1), kein neuaufbau).
        if self._horizon_mult_get is not None and self._horizon_mult_set is not None:
            self.horizon = self.left_stack.add(HorizonSlider(
                value=self._horizon_mult_get,
                minimum=self._horizon_mult_min,
                maximum=self._horizon_mult_max,
                on_change=self._horizon_mult_set,
                predictor=self.telemetry.predictor,
                sweep_seconds=self._horizon_sweep_s,
            ))

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

        # Unter dem umbruch fallen alle manoever-plaettchen weg -- sie
        # sind ein PLANUNGSwerkzeug, kein fluginstrument, und in einem
        # schmalen fenster nehmen sie den platz weg, den die bahn braucht.
        # Die tasten N / Shift+N / X bleiben in jeder groesse erreichbar,
        # und der ziehgriff an der linie ebenso: der haengt an der bahn,
        # nicht an der chrome.
        self.maneuver_axes.visible = wide
        self.maneuver_nodes.visible = wide
        self.maneuver_burn.visible = wide
        if self.maneuver_length is not None:
            self.maneuver_length.visible = wide

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
