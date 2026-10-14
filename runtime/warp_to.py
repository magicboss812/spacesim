"""Zeitraffer bis zu einem zeitpunkt -- "WARP TO" an Ap/Pe und an einem bahnpunkt.

Gestartet vom angehefteten schwebezettel (`ui/hud/apsis_tooltip.py`,
`ui/hud/point_tooltip.py`), gefahren von `runtime/loop.py`. Drei regeln:

1. STUFENWEISE HERUNTER. Je frame die hoechste stufe aus `WARP_STEPS`, die
   die bahn zulaesst (dieselbe grenze wie `Camera.clamp_warp_to_timescale`)
   UND bei der die restzeit noch mindestens `lead_real_seconds` echtzeit
   dauert. Je naeher das ziel, desto tiefer die stufe -- ohne sprung am ende.
2. NIE UEBER DAS ZIEL. `max_sim_seconds()` deckelt den sim-schritt des
   frames auf die restzeit, dasselbe muster wie
   `ManeuverExecutor.max_sim_seconds`. Die stufenwahl allein koennte das
   nicht garantieren: ein langsamer frame (max_frame_dt) rueckt weiter vor.
3. JEDE EIGENE ENTSCHEIDUNG DES SPIELERS BRICHT AB. Eine andere stufe
   (PageUp/PageDown, leiste, dev-oberflaeche), die pause, schub und jeder
   neue klick. Erkannt wird die stufenwahl daran, dass `camera.sim_dt` zu
   beginn des frames nicht mehr der wert ist, den die schleife am ende des
   vorigen frames stehen liess (`note_applied`).

ANKUNFT: auf die unterste stufe (1m/s, echtzeit), nicht in die pause. Wer
zu einer apsis raffft, will dort meist brennen -- und schub gibt es nur in
echtzeit; eine pause muesste er erst wieder aufheben. Die marge vor einer
apsis (`apsis_margin_s`, 30 min) laesst dafuer zeit.
"""


class WarpTo:
    """Ein ziel zur zeit. Kein ziel heisst: tut nichts."""

    def __init__(self, camera, tick_rate, warp_steps, realtime_warp_max,
                 warp_timescale_divisor, apsis_margin_s=1800.0,
                 point_margin_s=0.0, lead_real_seconds=1.5):
        self.camera = camera
        self.tick_rate = max(1.0, float(tick_rate))
        #: (sim-s je echtsekunde, beschriftung), aufsteigend.
        self.steps = tuple(warp_steps)
        self.realtime_warp_max = float(realtime_warp_max)
        self.divisor = max(1e-9, float(warp_timescale_divisor))
        self.apsis_margin_s = max(0.0, float(apsis_margin_s))
        self.point_margin_s = max(0.0, float(point_margin_s))
        self.lead_real_seconds = max(0.0, float(lead_real_seconds))
        self.target = None
        self.label = ''
        self._applied = None
        self._paused = False
        #: (welt-zeit bei ankunft, ziel) der letzten ankunft -- fuer messungen.
        self.last_arrival = None

    @property
    def active(self):
        return self.target is not None

    # -------------------------------------------------------------- steuern

    def start(self, t_stop, now, label=''):
        """Ziel setzen. False, wenn es nicht mehr in der zukunft liegt."""
        t_stop = float(t_stop)
        if not (t_stop > float(now) + 1e-6):
            return False
        self.target = t_stop
        self.label = str(label)
        self._applied = None
        self.camera.paused = False
        print(f"WARP TO: {self.label} bei t={t_stop:.3f} s "
              f"(in {t_stop - float(now):.1f} s)")
        return True

    def cancel(self, reason=''):
        if self.target is None:
            return
        print(f"WARP TO: abgebrochen ({reason})")
        self.target = None
        self._applied = None

    # ----------------------------------------------------------- je frame

    def _rate_cap(self, world, ship):
        """Hoechste raffung, die die bahn noch aufloest (sim-s je echtsekunde).

        Dieselbe rechnung wie `Camera.clamp_warp_to_timescale`, damit die
        hier gewaehlte stufe nicht im selben frame von `_clamp_warp` wieder
        gekappt wird.
        """
        fn = getattr(world, 'characteristic_timescale', None)
        if fn is None or ship is None:
            return None
        try:
            t_char = fn(ship)
        except Exception:
            return None
        if not t_char or t_char <= 0.0:
            return None
        return max(float(t_char) / self.divisor * self.tick_rate,
                   self.realtime_warp_max)

    def choose_rate(self, remaining, rate_cap=None):
        """Die stufe fuer `remaining` sim-sekunden restzeit (regel 1)."""
        best = self.steps[0][0]
        for rate, _label in self.steps:
            if rate_cap is not None and rate > rate_cap * 1.001:
                break
            if remaining / rate >= self.lead_real_seconds:
                best = rate
        return best

    def update(self, world, ship=None):
        """VOR `_clamp_warp`: abbruch erkennen, stufe setzen."""
        if self.target is None:
            return
        camera = self.camera
        if self._applied is not None and (
                camera.sim_dt != self._applied or camera.paused):
            self.cancel('raffung von hand geaendert'
                        if not camera.paused else 'pause')
            return
        remaining = self.target - float(world.time)
        rate = self.choose_rate(remaining, self._rate_cap(world, ship))
        low = float(getattr(camera, 'min_sim_dt', 1e-6) or 1e-6)
        high = float(getattr(camera, 'max_sim_dt', 1e12) or 1e12)
        camera.sim_dt = max(low, min(high, rate / self.tick_rate))
        camera.paused = False

    def note_applied(self):
        """NACH `_clamp_warp`/`_apply_maneuver`: den stehengelassenen wert
        merken. Ein abweichender wert zu beginn des naechsten frames kam
        dann von hand (regel 3)."""
        if self.target is not None:
            self._applied = self.camera.sim_dt

    def max_sim_seconds(self, world):
        """Deckel des sim-schritts (regel 2), oder None."""
        if self.target is None:
            return None
        return max(0.0, self.target - float(world.time))

    def after_step(self, world):
        """NACH `world.step`: angekommen? Dann unterste stufe, ziel weg."""
        if self.target is None:
            return False
        now = float(world.time)
        # Toleranz: ein paar ulp der weltzeit. world.step zerlegt den schritt
        # in stuecke, deren summe in der letzten stelle vom deckel abweichen
        # kann; ohne toleranz bliebe ein rest von 1e-12 s als eigener frame.
        if now < self.target - (1e-9 + 1e-14 * abs(self.target)):
            return False
        target = self.target
        self.target = None
        self._applied = None
        low = float(getattr(self.camera, 'min_sim_dt', 1e-6) or 1e-6)
        self.camera.sim_dt = max(low, self.steps[0][0] / self.tick_rate)
        self.last_arrival = (now, target)
        print(f"WARP TO: angekommen bei t={now:.6f} s, ziel {target:.6f} s "
              f"(differenz {now - target:+.3e} s)")
        return True
