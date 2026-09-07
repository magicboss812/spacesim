"""Das schubprofil eines geplanten manoevers.

DIE EINE QUELLE fuer brenndauer, zuendzeitpunkt und geliefertes delta-v.
Vorschau (`ship/maneuver/preview.py`) und ausfuehrung
(`ship/maneuver/executor.py`) rufen dieselben funktionen auf. Stuende die
dauer an zwei stellen, zeigte die gezeichnete linie ein anderes manoever
als geflogen wird -- und zwar erst dann, wenn es zu spaet ist, es zu
bemerken.

KEIN IMPULS. Der schub faehrt mit fester RATE hoch, haelt, und faehrt mit
derselben rate wieder herunter:

    r = a_max / ramp_seconds        [m/s^3]

DIE RATE ist die erhaltene groesse, nicht die rampenZEIT. Bei einem kleinen
delta-v wird a_max nie erreicht; mit fester rampenzeit gaebe es dort einen
sprung in der dauer (das profil muesste ploetzlich ein dreieck sein, ohne
dass die formel es merkt). Mit fester rate gehen trapez und dreieck stetig
ineinander ueber -- siehe `tests/maneuver_profile_test.py` abschnitt 3.

    a                              a
    |      ____________            |    /\\
    | a_p /            \\           |a_p/  \\
    +----+--------------+--- t     +--+----+--- t
      ramp   hold   ramp             r'   r'

BEIDE FORMEN SIND SYMMETRISCH, und daraus folgt der zuendzeitpunkt: bei
total_time/2 ist genau die haelfte des delta-v geliefert. Der knoten sitzt
also in der MITTE des brennvorgangs, halb davor, halb danach -- dieselbe
regel, die KSP benutzt, und die einzige, bei der eine endliche brenndauer
die geplante bahn trifft statt sie zu verfehlen.

Einheiten: a_max ist eine beschleunigung in SIMULATIONS-sekunden, nicht in
echtsekunden. Warum, steht in `.claude/rules/maneuver.md` (D2) und im kopf
von `ship/maneuver/executor.py`.
"""

import math


class BurnProfile:
    """Trapez- oder dreiecksprofil fuer ein gegebenes delta-v."""

    __slots__ = ('dv', 'a_max', 'ramp_rate', 'a_peak', 'ramp_time',
                 'hold_time', 'total_time')

    def __init__(self, dv, a_max, ramp_seconds):
        dv = abs(float(dv))
        a_max = max(1e-9, float(a_max))
        ramp_seconds = max(1e-9, float(ramp_seconds))
        rate = a_max / ramp_seconds

        self.dv = dv
        self.a_max = a_max
        self.ramp_rate = rate

        if dv <= 0.0:
            self.a_peak = 0.0
            self.ramp_time = 0.0
            self.hold_time = 0.0
            self.total_time = 0.0
            return

        # min() deckt beide zweige ab: unterhalb von dv = a_max*ramp liegt
        # sqrt(dv*rate) unter a_max und das profil wird ein dreieck.
        self.a_peak = min(a_max, math.sqrt(dv * rate))
        self.ramp_time = self.a_peak / rate
        self.hold_time = max(0.0, dv / self.a_peak - self.ramp_time)
        self.total_time = 2.0 * self.ramp_time + self.hold_time

    # ------------------------------------------------------------- ablesen

    @property
    def lead_time(self):
        """Zeit von der zuendung bis zum knoten -- die haelfte der dauer.

        Gilt, weil beide profilformen symmetrisch sind; nachgemessen in
        `tests/maneuver_profile_test.py` abschnitt 5.
        """
        return self.total_time * 0.5

    def ignition_time(self, t_node):
        """Absolute sim-zeit, zu der der schub einsetzen muss."""
        return float(t_node) - self.lead_time

    def accel_at(self, tau):
        """Schubbeschleunigung `tau` sekunden nach der zuendung.

        ACHTUNG: `physics/kernels/burn.py::_profile_accel_numba` ist der
        zwilling dieser funktion fuer den kernel (numba nimmt keine
        Python-objekte). Wer hier etwas aendert, aendert es DORT auch --
        abschnitt 8 des tests vergleicht beide exakt.
        """
        tau = float(tau)
        if tau <= 0.0 or tau >= self.total_time:
            return 0.0
        if tau < self.ramp_time:
            return self.ramp_rate * tau
        if tau < self.ramp_time + self.hold_time:
            return self.a_peak
        return self.ramp_rate * (self.total_time - tau)

    def throttle_at(self, tau):
        """Hebelstellung 0..1 -- was der schubbogen des HUDs anzeigt."""
        return max(0.0, min(1.0, self.accel_at(tau) / self.a_max))

    def dv_delivered(self, tau):
        """Aufsummiertes delta-v von der zuendung bis `tau`."""
        tau = float(tau)
        if tau <= 0.0:
            return 0.0
        if tau >= self.total_time:
            return self.dv
        if tau < self.ramp_time:
            return 0.5 * self.ramp_rate * tau * tau
        ramp_dv = 0.5 * self.a_peak * self.ramp_time
        if tau < self.ramp_time + self.hold_time:
            return ramp_dv + self.a_peak * (tau - self.ramp_time)
        rest = self.total_time - tau
        return self.dv - 0.5 * self.ramp_rate * rest * rest

    def dv_between(self, tau0, tau1):
        """Delta-v ueber ein zeitfenster, GESCHLOSSEN gerechnet.

        Der ausfuehrer benutzt das statt `accel_at(tau) * dt`: nur so ist
        das gelieferte delta-v unabhaengig von der bildrate und trifft am
        ende exakt den plan.
        """
        return self.dv_delivered(tau1) - self.dv_delivered(tau0)

    def __repr__(self):
        return (f"BurnProfile(dv={self.dv:.3f}, a_peak={self.a_peak:.3f}, "
                f"total={self.total_time:.3f}s)")
