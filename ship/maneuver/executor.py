"""Der autopilot, der einen knoten wirklich fliegt.

    IDLE --arm()--> ARMED --zuendung--> BURNING --profil zuende--> DONE
                      |                    |
                      +---- abort() -------+--> ABORTED

WAS DER AUSFUEHRER NICHT TUT: er rechnet weder richtung noch dauer selbst.
Die richtung kommt aus dem marker, den die vorschau aufgeloest hat, die
dauer aus `BurnProfile`. Stuende hier eine zweite rechnung, floege das
schiff etwas anderes als die linie zeigt -- und genau das ist der fehler,
den dieses ganze modul verhindern soll.

DIE EINHEITENFRAGE. `schiffcontrol.apply_thrust` addiert
`thrust_acc * real_dt` -- 600 m/s^2 mal ECHTsekunden -- waehrend die welt um
`sim_dt * tick_rate * real_dt` SIM-sekunden vorrueckt. Auf der untersten
raffungsstufe sind das 60 sim-sekunden je echtsekunde. In SIM-zeit
gerechnet betraegt die beschleunigung der pfeiltaste also
`600 / 60 = 10 m/s^2`. Die vorschau integriert in sim-zeit, folglich muss
das profil dieselbe zahl benutzen:

    a_max_sim = thrust_acc / realtime_warp_max

`a_max` wird beim scharfschalten im profil festgehalten und nicht aus
`ship_control.thrust_acc` gelesen -- das feld schreibt der schubregler des
HUDs. `config.maneuver.max_accel` ueberschreibt die ableitung.

KICK-THEN-DRIFT. `update()` legt das delta-v EINES schrittes an, BEVOR
`world.step()` diesen schritt geht. Die vorschau integriert den schub
dagegen stetig (RK4). Der unterschied ist erster ordnung in der
schrittweite, und deshalb ist die schrittweite waehrend des brennens auf
`burn_step_max_s` gedeckelt. Der verbleibende rest wird in
`tests/maneuver_execute_test.py` abschnitt 8 gegen die vorschau gemessen.
"""

from .plan import MIN_EXECUTABLE_DV
from .profile import BurnProfile

IDLE = 'idle'
ARMED = 'armed'
BURNING = 'burning'
DONE = 'done'
ABORTED = 'aborted'

#: Wert fuer `schiffcontrol.snap_mode`, waehrend der autopilot laeuft.
#: Absichtlich NICHT in `ship/control.py::SNAP_MODES` -- die rosette bildet
#: die vier richtungen ab, die der spieler selbst waehlt, und diese hier
#: waehlt der autopilot.
NODE_SNAP_MODE = 'node'


class ManeuverExecutor:
    """Scharfschalten, ausrichten, zuenden, brennen, aufraeumen."""

    def __init__(self, plan, ship, ship_control, camera=None, telemetry=None,
                 thrust_acc_max=600.0, realtime_warp_max=60.0, tick_rate=180.0,
                 ramp_seconds=0.6, max_accel=None, orient_lead_seconds=5.0,
                 burn_step_max_s=0.5, min_executable_dv=MIN_EXECUTABLE_DV):
        self.plan = plan
        self.ship = ship
        self.ship_control = ship_control
        self.camera = camera
        self.telemetry = telemetry

        self.thrust_acc_max = float(thrust_acc_max)
        self.realtime_warp_max = max(1e-9, float(realtime_warp_max))
        self.tick_rate = max(1.0, float(tick_rate))
        self.ramp_seconds = max(1e-6, float(ramp_seconds))
        self.max_accel = None if max_accel is None else float(max_accel)
        self.orient_lead_seconds = max(0.0, float(orient_lead_seconds))
        self.burn_step_max_s = max(1e-3, float(burn_step_max_s))
        self.min_executable_dv = float(min_executable_dv)

        self.state = IDLE
        self.node = None
        self.profile = None
        self.dir_x = 0.0
        self.dir_y = 0.0
        self.t_ignition = 0.0
        self.dv_delivered = 0.0
        self.abort_reason = None

        self._thrust_acc_before = None
        self._thrust_level_before = None

    # ------------------------------------------------------------- ableitung

    def a_max_sim(self):
        """Schubbeschleunigung in SIM-sekunden (siehe modulkopf)."""
        if self.max_accel is not None:
            return max(1e-9, self.max_accel)
        return max(1e-9, self.thrust_acc_max / self.realtime_warp_max)

    @property
    def is_active(self):
        return self.state in (ARMED, BURNING)

    def can_arm(self):
        node = self.plan.first() if self.plan is not None else None
        return node is not None and node.is_executable(self.min_executable_dv)

    # ----------------------------------------------------------- schalten

    def _marker_for(self, preview, node):
        if preview is None:
            return None
        for marker in getattr(preview, 'node_markers', ()) or ():
            if marker.get('node') is node:
                return marker
        return None

    def arm(self, world, reference_body=None, preview=None):
        """Den naechsten knoten scharfschalten. False, wenn das nicht geht."""
        if self.is_active:
            return False
        node = self.plan.first() if self.plan is not None else None
        if node is None or not node.is_executable(self.min_executable_dv):
            return False

        marker = self._marker_for(preview, node)
        if marker is None:
            # Ohne aufgeloeste richtung gibt es nichts zu fliegen. Sie kommt
            # aus der vorschau, weil dort der zustand an der knotenzeit
            # bereits steht -- hier ein zweites mal zu propagieren waere
            # eine zweite wahrheit.
            return False

        a_max = self.a_max_sim()
        profile = BurnProfile(node.dv_total, a_max, self.ramp_seconds)
        t_ignition = profile.ignition_time(node.t_node)
        now = float(getattr(world, 'time', 0.0))
        if now > t_ignition + profile.total_time:
            # Der knoten liegt vollstaendig in der vergangenheit.
            return False

        self.node = node
        self.profile = profile
        self.dir_x = float(marker['dir_x'])
        self.dir_y = float(marker['dir_y'])
        self.t_ignition = t_ignition
        self.dv_delivered = 0.0
        self.abort_reason = None
        self.state = ARMED

        if self.ship_control is not None:
            self._thrust_acc_before = float(
                getattr(self.ship_control, 'thrust_acc', self.thrust_acc_max))
            # Die nase auf die schubrichtung rasten. Direkt gesetzt, nicht
            # ueber toggle_snap: 'node' ist keiner der vier rosetten-modi.
            self.ship_control.snap_mode = NODE_SNAP_MODE
            self.ship_control._snap_locked = False
        if self.telemetry is not None:
            self._thrust_level_before = float(
                getattr(self.telemetry, 'thrust_level', 1.0))
        return True

    def abort(self, reason='aborted'):
        if not self.is_active:
            return False
        self._restore()
        self.state = ABORTED
        self.abort_reason = str(reason)
        return True

    def notify_manual_input(self):
        """Von der hauptschleife gerufen, sobald der spieler selbst steuert.

        Handeingabe schlaegt den autopiloten, immer. Ein brennvorgang, den
        man nicht durch anfassen der steuerung stoppen kann, ist eine falle.
        """
        return self.abort('manual input')

    def _restore(self):
        if self.ship_control is not None:
            try:
                self.ship_control.clear_snap()
            except Exception:
                self.ship_control.snap_mode = None
            if self._thrust_acc_before is not None:
                self.ship_control.thrust_acc = self._thrust_acc_before
        if self.telemetry is not None and self._thrust_level_before is not None:
            try:
                self.telemetry.set_thrust_level(self._thrust_level_before)
            except Exception:
                pass
        self._thrust_acc_before = None
        self._thrust_level_before = None

    # -------------------------------------------------------- zeitschritte

    def max_sim_seconds_value(self, now):
        """Die schrittklemme fuer eine gegebene sim-zeit (testbar ohne welt)."""
        if self.state == ARMED:
            remaining = self.t_ignition - float(now)
            return remaining if remaining > 0.0 else self.burn_step_max_s
        if self.state == BURNING:
            return self.burn_step_max_s
        return None

    def max_sim_seconds(self, world):
        """Wie weit die welt in DIESEM frame hoechstens vorruecken darf.

        Scharf: hoechstens bis zur zuendung -- ein zeitraffer-frame rueckt
        sonst um stunden vor und der ganze brennvorgang faellt zwischen zwei
        bilder. Brennend: `burn_step_max_s`, damit die kick-then-drift-
        naeherung klein bleibt.
        """
        return self.max_sim_seconds_value(float(getattr(world, 'time', 0.0)))

    def wants_realtime(self, world):
        """Muss die raffung jetzt auf echtzeit herunter?

        Ja, sobald die zuendung naeher liegt als die zeit, die das schiff
        zum drehen braucht -- gerechnet mit `orient_lead_seconds`, das eine
        halbe umdrehung bei `rotation_speed` mit reserve abdeckt.
        """
        if self.state == BURNING:
            return True
        if self.state != ARMED:
            return False
        remaining = self.t_ignition - float(getattr(world, 'time', 0.0))
        return remaining <= self.orient_lead_seconds

    # ------------------------------------------------------------- fahren

    def update(self, world, sim_seconds):
        """VOR `world.step(sim_seconds)` aufrufen.

        Legt das delta-v des bevorstehenden schrittes an, damit die
        positionsintegration dieses schrittes es bereits traegt.
        """
        if not self.is_active or self.profile is None:
            return
        now = float(getattr(world, 'time', 0.0))
        step = max(0.0, float(sim_seconds))

        if self.state == ARMED:
            if now + 1e-9 < self.t_ignition:
                return
            self.state = BURNING

        tau0 = now - self.t_ignition
        tau1 = tau0 + step

        dv = self.profile.dv_between(tau0, tau1)
        if dv > 0.0 and self.ship is not None:
            self.ship.velocity.x += self.dir_x * dv
            self.ship.velocity.y += self.dir_y * dv
            self.dv_delivered += dv

        # Der schubbogen des HUDs zeigt damit die rampe -- kostenlos, weil
        # der regler ohnehin gezeichnet wird.
        if self.telemetry is not None:
            try:
                self.telemetry.set_thrust_level(
                    self.profile.throttle_at(max(0.0, (tau0 + tau1) * 0.5)))
            except Exception:
                pass

        if tau1 >= self.profile.total_time:
            self._finish()

    def _finish(self):
        node = self.node
        self._restore()
        self.state = DONE
        if node is not None and self.plan is not None:
            self.plan.remove(node)
        self.node = None
