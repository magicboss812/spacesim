"""Die laengenregel des vorhersage-horizonts.

Die modulfunktionen sind rein, damit `tests/horizon_targets_test.py` und
`tests/warp_predictor_test.py` §23 sie einzeln messen koennen;
`HorizonPolicy` haelt den zustand.

Der horizont ist ein PRODUKT: basis * manuell ('+'/'-') * raffung.
"""
import math


def predictor_horizon_lengths(base_length, manual_mult, warp_mult,
                              max_points, base_spacing):
    """(gezeichnete laenge, gerechnete laenge) fuer den vorhersage-horizont.

    Modulebene und rein, damit `tests/warp_predictor_test.py` §23 sie messen
    kann statt die regel nachzubauen. Die begruendung fuer den deckel steht
    bei `HorizonPolicy.apply()`.
    """
    drawn = float(base_length) * float(manual_mult)
    budget_length = float(max_points) * float(base_spacing)
    warp_mult = float(warp_mult)
    if drawn > 0.0:
        warp_mult = min(warp_mult, max(1.0, budget_length / drawn))
    return drawn, drawn * warp_mult


def horizon_compute_rung(base_length, wanted, step_factor):
    """Naechsthoehere sprosse einer groben leiter ueber `wanted`.

    Die leiter haengt an `base_length` (nicht am aktuellen wert), also liegen
    ihre sprossen fest und wandern waehrend eines langen zugs nicht mit:
    base, base*f, base*f^2, ...
    """
    base = float(base_length)
    want = float(wanted)
    step = max(float(step_factor), 1.0 + 1e-9)
    if base <= 0.0 or want <= 0.0:
        return want
    n = math.ceil(math.log(want / base) / math.log(step) - 1e-12)
    return base * step ** n


def horizon_targets(base_length, manual_mult, warp_mult, max_points,
                    base_spacing, *, grabbing=False, current_length=None,
                    grab_step_factor=4.0):
    """Wie `predictor_horizon_lengths`, aber mit dem slider-griff.

    DIE GERECHNETE LAENGE DARF WAEHREND DES ZUGS NICHT AM KNAUF KLEBEN:
    `set_length()` verwirft den halt und storniert den laufenden auftrag, je
    frame gerufen kaeme also nie eine kurve an. Die gerechnete laenge auf die
    regler-decke zu pinnen machte dagegen jedes antippen so teuer wie die
    decke.

    Deshalb eine RASTE: waehrend des griffs faehrt die gerechnete laenge nur
    auf groben sprossen (`grab_step_factor`, an `base_length` verankert) nach
    OBEN mit und schrumpft gar nicht -- zu lang ist harmlos, das schneidet der
    zeichen-clip weg. Ueber einen vollen zug sind das log_f(spanne) aufrufe
    statt einer je frame. Beim loslassen faellt `wanted` in einem schritt auf
    den genauen wert.
    """
    drawn, wanted = predictor_horizon_lengths(
        base_length, manual_mult, warp_mult, max_points, base_spacing,
    )
    if not grabbing:
        return drawn, wanted
    if current_length is None or float(current_length) <= 0.0:
        return drawn, wanted
    current = float(current_length)
    if wanted <= current:
        # NACH UNTEN passiert waehrend des zugs nichts: die vorhandene kurve
        # ist dann nur zu lang, und `drawn` schneidet sie ohnehin. Null
        # aufrufe, null stornierte auftraege.
        return drawn, current
    return drawn, max(current, horizon_compute_rung(base_length, wanted,
                                                    grab_step_factor))


def warp_length_mult(rate):
    """Horizont-faktor aus der raffung -- zweierpotenz, gedeckelt.

    `rate` ist die raffung in sim-sekunden je echtsekunde (Camera.warp_rate).

    Bei hoher raffung frisst das schiff den horizont schneller als der
    halt ihn nachziehen kann, und jeder leerlauf kostet eine SYNCHRONE
    volle neuberechnung. Ein laengerer horizont ist deshalb bei raffung
    BILLIGER. Der deckel bei 64: darueber kostet die einzelne, seltene
    neuberechnung so viel, dass sie als ruckler sichtbar wird.
    Zweierpotenzen sorgen dafuer, dass sich der wert nur beim stufenwechsel
    aendert -- set_length() verwirft den halt, das darf nicht jeden frame
    passieren.
    """
    ratio = float(rate) / 604800.0   # ab 7 d/s waechst der horizont mit
    if ratio <= 1.0:
        return 1.0
    # RUNDEN, nicht abschneiden: ergibt 7d/s->1, 30d/s->4, 100d/s->16,
    # 1y/s->64 (abgeschnitten fiele 1 y/s auf die schlechtere stufe 32x).
    exp = min(6, max(0, int(round(math.log2(ratio)))))
    return float(1 << exp)


class HorizonPolicy:
    """Haelt den horizont-zustand und setzt ihn am predictor durch.

    Der manuelle faktor ('+'/'-' und der HUD-regler) lebt hier, nicht im
    predictor: er ist eine ABSICHT des spielers, waehrend `predictor.length`
    das ergebnis ist, in das auch die raffung eingeht.
    """

    def __init__(self, predictor, config):
        # Look-ahead horizon (length) is the cost knob; point spacing
        # (precision) is cosmetic. The horizon is pinned from startup so
        # changing spacing ('9'/'0') does not move it (and so does not change
        # compute cost). Default = num_points * base precision.
        self.base_length = predictor.num_points * predictor.precision
        predictor.set_length(self.base_length)

        # DAS PUNKTBUDGET WAECHST MIT DEM HORIZONT.
        #
        # `_horizon_spacing_floor()` ist `length / num_points` -- bei festem
        # budget verdoppelte jedes '+' auch den PUNKTABSTAND, bis eine
        # stuetzweite einen nennenswerten teil der bahn ueberspannt und die
        # Hermite-linie beulen und knicke zeigt. Deshalb waechst `num_points`
        # mit, bis zur decke `predictor.max_num_points`; der punktabstand und
        # das detail je umlauf bleiben konstant. Die schrittzahl des
        # integrators aendert sich dadurch nicht, nur ausgabe und arrays.
        self.base_spacing = (self.base_length
                             / max(1, int(predictor.num_points)))
        self.max_points = max(
            int(predictor.num_points),
            int(config.get('predictor.max_num_points', 40000)),
        )

        self.manual_mult = 1.0
        self.mult_min = float(config.get('predictor.horizon_slider_min_mult', 0.25))
        # DIE DECKE IST DIE DES SPIELERS, NICHT DIE DES PUNKTBUDGETS.
        # Oberhalb des budgets vergroebert der punktabstand -- das ist fuer
        # den manuellen faktor gewollt. Die vergroeberung durch die RAFFUNG
        # klemmt predictor_horizon_lengths() unabhaengig von dieser zahl. Eine
        # hohe decke kostet im ruhezustand nichts: die gerechnete laenge folgt
        # dem knauf ueber die raste in horizon_targets(), nicht der decke.
        self.mult_max = max(
            self.mult_min * (1.0 + 1e-9),
            float(config.get('predictor.horizon_slider_max_mult', 256.0)),
        )
        self.sweep_s = float(
            config.get('predictor.horizon_slider_sweep_seconds', 3.5))
        # Sprossenweite der raste waehrend des griffs. Grob halten: jede
        # sprosse kostet ein set_length() und damit einen stornierten auftrag.
        # Bei 4.0 sind es ueber die spanne 0.25x..256x genau fuenf.
        self.grab_step = max(
            float(config.get('predictor.horizon_grab_step_factor', 4.0)),
            1.0 + 1e-9)
        # Tastenschritt fuer '+'/'-'.
        self.length_step = max(
            float(config.get('predictor.length_step_factor', 2.0)), 1.0 + 1e-9)

    # -- der manuelle faktor (HUD-regler und '+'/'-') ----------------------

    def get_mult(self):
        return self.manual_mult

    def set_mult(self, mult):
        self.manual_mult = max(self.mult_min, min(self.mult_max, float(mult)))

    def step_mult(self, factor, predictor):
        """'+' / '-': den MANUELLEN faktor verstellen, nicht die laenge direkt.

        Sonst wuerde `apply()` die eingabe im naechsten frame ueberschreiben.
        Nach unten begrenzt der punktabstand, nicht `mult_min`: unter einem
        einzigen punkt gibt es keine linie mehr.
        """
        if factor >= 1.0:
            self.manual_mult *= factor
        else:
            lowest = predictor.precision / max(self.base_length, 1e-9)
            self.manual_mult = max(lowest, self.manual_mult * factor)

    # -- durchsetzen -------------------------------------------------------

    def apply(self, predictor, warp_rate, grabbing=False):
        """Horizont neu setzen, wenn sich basis*manuell*raffung geaendert hat."""
        if predictor.num_points <= 0:
            return
        # DIE RAFFUNGS-VERLAENGERUNG IST EIN VORRAT, KEIN BILD -- UND SIE DARF
        # DIE GEZEICHNETE LINIE NICHT ANFASSEN.
        #
        # `wanted` schlaegt an zwei stellen auf die SICHTBARE kurve durch,
        # obwohl der verlaengerte teil nicht gezeichnet wird: jenseits von
        # `max_points` vergroebert es den PUNKTABSTAND, und `horizon_arc` in
        # `Predictor._make_snapshot` (punkte x abstand) hebt die
        # fernfeld-schrittdecke an -- beides verschiebt die integrierte bahn.
        #
        # Der vorrat wird deshalb auf das begrenzt, was das PUNKTBUDGET beim
        # basis-abstand noch traegt: `wanted` ist nie groesser als
        # `max_points x base_spacing`, der abstand bleibt exakt der der
        # echtzeit -- und mit ihm `horizon_arc` und die decke. Hat der spieler
        # mit '+' bereits ueber das budget hinaus verlaengert, faellt der
        # raffungsfaktor auf 1.
        drawn, wanted = horizon_targets(
            self.base_length, self.manual_mult, warp_length_mult(warp_rate),
            self.max_points, self.base_spacing,
            grabbing=grabbing, current_length=predictor.length,
            grab_step_factor=self.grab_step,
        )
        # GEZEICHNET wird immer nur der un-geraffte horizont, sonst wickelt
        # sich die linie im zeitraffer mehrfach um die bahn. GERECHNET wird
        # trotzdem die volle laenge, weil genau die den halt am leben haelt
        # (siehe warp_length_mult).
        #
        # IMMER `drawn`, nie `None`: der clip gehoert an die GEWOLLTE laenge,
        # nicht an die gerade angeforderte -- eine noch vorhandene laengere
        # kurve (etwa nach dem loslassen des reglers) bleibt so geschnitten.
        # Der dauerhaft gesetzte clip ist O(1).
        if hasattr(predictor, 'set_display_length'):
            predictor.set_display_length(drawn)
        # Punktbudget zuerst, damit `set_length` gleich darauf arbeitet.
        # WEICH: der zeitraffer-schritt verstellt den horizont bei jedem
        # stufenwechsel und damit auch das budget -- ein harter reset waere
        # ein ruckler im hauptthread (siehe warp_predictor_test §17).
        points_wanted = int(min(
            self.max_points,
            max(1, math.ceil(wanted / max(self.base_spacing, 1e-9))),
        ))
        if points_wanted != int(predictor.num_points):
            predictor.set_num_points(points_wanted, soft=True)
        current = predictor.length
        if current is not None and abs(current - wanted) <= wanted * 1e-9:
            return
        predictor.set_length(wanted)
