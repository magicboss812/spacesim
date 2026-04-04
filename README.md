# Was gehört alles zu diesem Update?
Stand 04.04.2026 (an diesem Update wird noch weiter gearbeitet)

## Schiffkontrolle
- der Geschwindigkeitsvektor vom Schiff kann aktiv geändert werden
- der Vektor wird basierend auf der Rotation des Schiffes gedreht
- die Schubbeschleunigung skaliert dann den Vektor entsprechend

## Predictor Präzision und Länge
- der Predictor hat zwei neue Faktore:
        1. Predictor Länge beschreibt wie viele Zukunftspunkte es ingesamt gibt
        2. Predictor Präzision beschreibt den Intervall zwischen den Punkten
       **Die Länge im HUD wird dann mithilfe des Intervals und der Punkten insgesamt berechnet**
## Camera Culling und Zoom Performance
- nicht alle Punkte des Predictors werden auf dem Bildschirm auf einmal gerendert (aber alle werden auf einmal berechnet)
- das bedeutet:
> nur so viele, wie viele **Pixel** auf dem Bildschirm möglich zu sehen sind
- das reduziert also unnötige Belastung auf den Prozessor bzw. auf das Grafikrendering
- nicht nur Punkte außerhalb des Bildschirmes werden ausgelassen, **sondern** auch der Detailgrad des Predictors, den man bei hohem Zoom-Out eh nicht sieht

## Steuerung
- somit kann die Länge sowie die Präzision mit Tastaturtasten kontrolliert werden

### Warum die Länge und die Präzision?
- Der Predictor beschreibt die mögliche Position des Schiffes in ferner und naher Zukunft
- da das Schiff von vielen Körpern angezogen wird, heißt es, dass diese Linie **keine** komplette Elipsenbahn ist, sondern eine immer veränderte **Kurve**
- in sehr hohen Umlaufbahnen, also Jupiter und weiter, müsste der Predictor deutlich länger sein
- dies wiederum erhöht auch die Auslastung auf die PC-Komponente, da mehr Punkte berechnet und gerendert werden müssen
- andererseits kann der Spieler sehen, wo sich das Raumschiff in sehr ferner Zukunft befinden wird. Dies ermöglicht das Einsehen von Gravitationsmanövern und Transfers (**äußerst wichtig für die Seminararbeit!!!**)
- so kann der Spieler dies mühelos temporär regulieren. Es wurden Inspirationen aus dem KSP-Mod "Principia" genommen

## Weitere Ideen und Updates
- Hilfspunkte, Planetenorbits und Transferpunkte
- Perspektivwechsel für den Predictor (Wie wäre es, wenn der Predictor relativ zur Erde verläuft, nicht zur Sonne?)
- GPU Rendering (weniger Belastung für die CPU)
