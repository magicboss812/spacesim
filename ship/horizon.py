"""Die laengenregel des vorhersage-horizonts.

Lag als lose funktionen und closures in `test.py` -- also in der datei, die
das spiel STARTET, obwohl jede zeile davon dem predictor gehoert. Die drei
modulfunktionen bleiben modulfunktionen (`tests/horizon_targets_test.py` und
`tests/warp_predictor_test.py` §23 messen sie einzeln, statt die regel
nachzubauen); `HorizonPolicy` haelt den zustand, der vorher in `main()` als
lokale variablen lag.

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
    ihre sprossen fest und wandern nicht mit: base, base*f, base*f^2, ...
    Sonst driftete sie waehrend eines langen zugs immer weiter mit.
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

    DIE GERECHNETE LAENGE DARF WAEHREND DES ZUGS NICHT AM KNAUF KLEBEN.
    `set_length()` verwirft den halt und storniert den laufenden auftrag
    (predictor.set_length -> _cancel_pending_job); je frame gerufen kommt
    deshalb NIE eine kurve an, und der zug zeigt bis zum loslassen dieselbe
    alte linie. Gleichzeitig ist die naive gegenrichtung -- die gerechnete
    laenge waehrend des griffs auf die REGLER-DECKE pinnen -- der grund fuer
    genau zwei fehler:

      * der ablesewert las `predictor.length` zurueck und stand deshalb den
        ganzen zug ueber auf der decke statt auf dem knauf;
      * beim loslassen fiel `wanted` auf `drawn` zurueck, der clip wurde
        damit abgeschaltet (`drawn if wanted > drawn else None`) -- waehrend
        die decken-lange kurve noch im speicher lag. Fuer die paar frames,
        bis der kurze auftrag ankam, wurde sie ungeschnitten gezeichnet: die
        linie sprang auf die decke und wieder zurueck.

    Und die decke skaliert nicht: je hoeher `horizon_slider_max_mult`, desto
    teurer wird JEDES antippen des reglers, auch wenn der spieler nur von 1x
    auf 1.2x will.

    Deshalb eine RASTE: waehrend des griffs faehrt die gerechnete laenge nur
    auf groben sprossen (`grab_step_factor`, an `base_length` verankert) nach
    OBEN mit und schrumpft gar nicht -- zu lang ist harmlos, das schneidet der
    zeichen-clip weg. Ueber einen vollen zug sind das log_f(spanne) aufrufe
    statt einer je frame, und die kosten haengen an dem horizont, den der
    spieler wirklich waehlt, nicht an der decke. Beim loslassen faellt
    `wanted` in einem schritt auf den genauen wert.

    Siehe plans/predictor_horizon_slider_design.md.
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
    nicht teurer, sondern BILLIGER -- gemessen bei 1 y/s ueber 600 frames:

        faktor  median   p99     max    volle neuberechnungen
          1x    4.20 ms  5.69   8.29         277
         16x    1.62 ms  4.17   4.45           0
         64x    0.70 ms  3.86   4.84           0
        256x    0.28 ms  1.12  54.25           1   <-- sichtbarer hakler

    Der deckel bei 64 ist also nicht willkuerlich: darueber werden die
    neuberechnungen zwar noch seltener, aber die EINZELNE kostet dann so
    viel, dass sie als ruckler sichtbar wird. Zweierpotenzen sorgen
    ausserdem dafuer, dass sich der wert nur beim stufenwechsel aendert --
    set_length() verwirft den halt, das darf nicht jeden frame passieren.
    """
    ratio = float(rate) / 604800.0   # ab 7 d/s waechst der horizont mit
    if ratio <= 1.0:
        return 1.0
    # RUNDEN, nicht abschneiden: abgeschnitten faellt 1 y/s auf 32x, und
    # das ist gemessen die schlechtere stufe (max 14.7 ms gegen 4.8 ms bei
    # 64x). Gerundet ergibt die reihe genau die gemessenen guten werte
    # 7d/s->1, 30d/s->4, 100d/s->16, 1y/s->64.
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
        # (precision) is cosmetic. Pin the horizon from startup so changing
        # spacing ('9'/'0') no longer moves the horizon (and thus no longer
        # changes compute cost). Default = num_points * base precision, so
        # initial output is unchanged.
        self.base_length = predictor.num_points * predictor.precision
        predictor.set_length(self.base_length)

        # DAS PUNKTBUDGET WAECHST MIT DEM HORIZONT.
        #
        # `_horizon_spacing_floor()` ist `length / num_points` -- bei festem
        # budget verdoppelt jedes '+' also nicht nur den bogen, sondern auch
        # den PUNKTABSTAND. Die zahl der stuetzstellen JE UMLAUF halbiert sich
        # damit bei jedem druck, und irgendwann ueberspannt eine stuetzweite
        # einen nennenswerten teil der bahn: das kubische Hermite-polynom
        # zwischen zwei solchen punkten ist die bahn dann nicht mehr, die linie
        # wird zu beulen mit knicken dazwischen. In einer 2e7-m-erdumlaufbahn
        # sind das 180 stuetzstellen je umlauf im grundzustand, 22 bei 8x und
        # 5.6 bei 32x.
        #
        # Also waechst `num_points` mit, bis zur decke
        # `predictor.max_num_points` -- der punktabstand bleibt dann konstant
        # und mit ihm das detail je umlauf. Das ist billig: der integrator muss
        # denselben bogen ueberdecken wie vorher, die schrittzahl aendert sich
        # also nicht; teurer werden nur die ausgabe und die arrays.
        self.base_spacing = (self.base_length
                             / max(1, int(predictor.num_points)))
        self.max_points = max(
            int(predictor.num_points),
            int(config.get('predictor.max_num_points', 40000)),
        )

        self.manual_mult = 1.0
        self.mult_min = float(config.get('predictor.horizon_slider_min_mult', 0.25))
        # DIE DECKE IST DIE DES SPIELERS, NICHT DIE DES PUNKTBUDGETS.
        #
        # Sie lag frueher auf `max_num_points * base_spacing / base_length`
        # (= 4x, 40 Gm), weil oberhalb davon der punktabstand vergroebert. Das
        # ist aber dieselbe vergroeberung, die '+' seit jeher erlaubt --
        # gewollt und dokumentiert. Was §23 verhindert, ist die vergroeberung
        # durch die RAFFUNG, und die klemmt weiterhin in
        # predictor_horizon_lengths(): dort wird `warp_mult` aufs budget
        # gedeckelt, unabhaengig von dieser zahl.
        #
        # Ohne diesen deckel kostet eine hohe zahl hier auch nichts im
        # ruhezustand: die gerechnete laenge folgt dem knauf ueber die raste in
        # horizon_targets(), nicht der decke.
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
        # `wanted` geht an zwei stellen weiter, die beide auf die SICHTBARE
        # kurve durchschlagen, obwohl der verlaengerte teil gar nicht
        # gezeichnet wird (set_display_length unten):
        #
        # 1. `points_wanted` ist bei `max_points` gedeckelt. Ist der deckel
        #    erreicht, vergroebert jede weitere verlaengerung den PUNKTABSTAND
        #    -- gemessen bei manuell 8x und 64x raffung: 40000 punkte auf dem
        #    gezeichneten stueck werden zu **626**, und die gezeichnete kurve
        #    weicht dann selbst mit der kubischen Hermite-auswertung um
        #    **2.3e6 m** von derselben bahn ab (linear waeren es 6.8e6 m). Auf
        #    einer bahn mit perigaeum 1e7 m ist das eine sichtbar andere linie.
        # 2. `horizon_arc` in `Predictor._make_snapshot` ist `punkte x abstand`
        #    und hebt damit die fernfeld-schrittdecke an. Gemessen dieselbe
        #    lage: decke 2163 -> 8676 s, und die INTEGRIERTE bahn verschiebt
        #    sich um **2.3e6 m** (mit fester decke: 8.4e4 m).
        #
        # Beides zusammen ist der bericht "die vorhersage sieht im zeitraffer
        # ganz anders aus" -- rund 4.6e6 m auf einer linie, deren perigaeum
        # 1e7 m misst, allein vom druck auf die raffungstaste.
        #
        # Der vorrat wird deshalb auf das begrenzt, was das PUNKTBUDGET beim
        # basis-abstand noch traegt. Dann ist `wanted` nie groesser als
        # `max_points x base_spacing`, der abstand bleibt exakt der der
        # echtzeit -- und mit ihm `horizon_arc` und die decke. Hat der spieler
        # mit '+' bereits ueber das budget hinaus verlaengert, faellt der
        # raffungsfaktor auf 1: seine eigene vergroeberung bleibt (die ist
        # gewollt und dokumentiert), die der raffung kommt nicht dazu.
        drawn, wanted = horizon_targets(
            self.base_length, self.manual_mult, warp_length_mult(warp_rate),
            self.max_points, self.base_spacing,
            grabbing=grabbing, current_length=predictor.length,
            grab_step_factor=self.grab_step,
        )
        # GEZEICHNET wird immer nur der un-geraffte horizont. Ohne das wickelt
        # sich die linie im zeitraffer mehrfach um die bahn, waehrend sie in
        # echtzeit einen einzigen bogen zeigt -- und die Ap/Pe-fahnen stapeln
        # sich uebereinander. GERECHNET wird trotzdem die volle laenge, weil
        # genau die den halt am leben haelt (siehe warp_length_mult).
        #
        # IMMER `drawn`, nie `None`: der clip gehoert an die GEWOLLTE laenge,
        # nicht an die gerade angeforderte. `wanted > drawn` als bedingung
        # schaltete ihn in dem moment ab, in dem `wanted` zurueckfiel -- also
        # beim loslassen des reglers, waehrend die lange kurve noch dalag; die
        # linie sprang dann fuer ein paar frames auf deren volle laenge.
        # `set_display_length` ist O(1) und `_display_point_count` gibt bei
        # einer kurve, die ohnehin nicht laenger ist als `drawn`, None zurueck
        # -- der dauerhaft gesetzte clip kostet also nichts.
        if hasattr(predictor, 'set_display_length'):
            predictor.set_display_length(drawn)
        # Punktbudget zuerst, damit `set_length` gleich darauf arbeitet.
        # WEICH: der zeitraffer-schritt verstellt den horizont bei jedem
        # stufenwechsel und damit auch das budget -- ein harter reset waere
        # genau der ruckler, den set_length(soft) schon einmal beseitigt hat
        # (34-82 ms im hauptthread, siehe §17).
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
