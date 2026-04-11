# OSTERUPDATE: Was gehört alles zu diesem Update?
Stand 04.04.2026 (an diesem Update wird noch weiter gearbeitet)

**Schiffkontrolle**
- der Geschwindigkeitsvektor vom Schiff kann aktiv geändert werden
- der Vektor wird basierend auf der Rotation des Schiffes gedreht
- die Schubbeschleunigung skaliert dann den Vektor entsprechend

**Predictor Präzision und Länge**
- der Predictor hat zwei neue Faktore:
        1. Predictor Länge beschreibt wie viele Zukunftspunkte es ingesamt gibt
        2. Predictor Präzision beschreibt den Intervall zwischen den Punkten
       **Die Länge im HUD wird dann mithilfe des Intervals und der Punkten insgesamt berechnet**
**Camera Culling und Zoom Performance**
- nicht alle Punkte des Predictors werden auf dem Bildschirm auf einmal gerendert (aber alle werden auf einmal berechnet)
- das bedeutet: `nur so viele, wie viele Pixel auf dem Bildschirm möglich zu sehen sind`
- das reduziert also unnötige Belastung auf den Prozessor bzw. auf das Grafikrendering
> [!IMPORTANT]
> nicht nur Punkte außerhalb des Bildschirmes werden ausgelassen, **sondern** auch der Detailgrad des Predictors, den man bei hohem Zoom-Out eh nicht sieht

**Steuerung des Predictors**
- somit kann die Länge sowie die Präzision mit Tastaturtasten kontrolliert werden

**Warum die Länge und die Präzision?**
- Der Predictor beschreibt die mögliche Position des Schiffes in ferner und naher Zukunft
- da das Schiff von vielen Körpern angezogen wird, heißt es, dass diese Linie **keine** komplette Elipsenbahn ist, sondern eine immer veränderte **Kurve**
- in sehr hohen Umlaufbahnen, also Jupiter und weiter, müsste der Predictor deutlich länger sein
- dies wiederum erhöht auch die Auslastung auf die PC-Komponente, da mehr Punkte berechnet und gerendert werden müssen
- andererseits kann der Spieler sehen, wo sich das Raumschiff in sehr ferner Zukunft befinden wird. Dies ermöglicht das Einsehen von Gravitationsmanövern und Transfers (**äußerst wichtig für die Seminararbeit!!!**)
- so kann der Spieler dies mühelos temporär regulieren. Es wurden Inspirationen aus dem KSP-Mod "Principia" genommen

**Weitere Ideen und Updates**
- Hilfspunkte, Planetenorbits und Transferpunkte
- Perspektivwechsel für den Predictor (Wie wäre es, wenn der Predictor relativ zur Erde verläuft, nicht zur Sonne?)
- GPU Rendering (weniger Belastung für die CPU)

# 11.04.2026 Finales Update für die Osterferien:

Stand 11.04.2026, weitere Änderungen werden wahrscheinlich erstmal lokal oder auf `main` committed.

## BIGGEST CHANGES:

### Reference Körper:

* Der Spieler kann einen Körper als „Mittelpunkt aller Umlaufbahnen“ auswählen. **Das bedeutet konkret:** Die Erde kann beispielsweise als Zentrum gewählt werden, sodass es so wirkt, als würden die Sonne und alle anderen Körper um die Erde kreisen. Dadurch entstehen sogenannte „Epizykel“, bestehend aus Trägerkreis (Deferent) und Schleifenbewegung.
  ![Beispiel von Epizykeln](https://i.imgur.com/r0UbrMp.png)

* Dies ermöglicht einen „einfacheren“ Transfer zu Planeten: Die relative Geschwindigkeit zum Reference Body wird eliminiert (Gleichsetzung der Geschwindigkeitsvektoren), sodass das Schiff relativ zu diesem Körper „stillsteht“. Durch die Gravitation fällt das Schiff dann in den Körper hinein und bildet kurz davor einen stabilen Orbit. Das ist ein ineffizienter, aber schneller Transfer.

> [!WARNING]
> Das System befindet sich noch in einer **Testphase** und ist teilweise fehlerhaft, zeigt jedoch bereits wichtige Kernelemente. Der Predictor ist noch nicht auf das Reference-Body-System ausgelegt und liefert daher fehlerhafte Ergebnisse.

### Small Changes:

**Planetenorbits**

* Planeten zeichnen nun ihre vergangenen Positionen bzw. ihren Bewegungspfad auf.
* Noch unvollständig, da die eigentliche Umlaufbahn nicht vollständig dargestellt wird.
* Noch zu erledigen: Bei Auswahl eines Planeten in der Kameraansicht soll dessen Orbit vollständig angezeigt werden. Die vergangene Trajectory soll nach einer gewissen Zeit ausfaden (transparent verschwinden).

**Runge-Kutta 4 Integrator**

* Wechsel auf den Runge-Kutta-4-Integrator
* Energie wird deutlich besser erhalten, das Schiff erhält keinen ungewollten „Schub“ mehr
* Nachteil: Die Performance ist deutlich gesunken, auch bedingt durch das Reference-Body-System

**Performance Optimierung (angeblich)**

* Der Predictor wurde auf RK4 optimiert, indem Snapshots der gesamten Linie gespeichert werden
* Die Liste der Predictor-Punkte wird dynamisch erweitert, während alte Punkte pro Loop entfernt werden. Zusätzlich wird die gesamte Liste pro Loop um einen Index verschoben
* Statt: 10.000 Punkte pro Main Loop → 10–100 Punkte, abhängig vom Zeitschritt

> [!WARNING]
> Diese Optimierung befindet sich ebenfalls in einer Testphase. Der Predictor hört nach einiger Zeit auf, neue Punkte zu berechnen, wodurch der aktuelle Snapshot „stehen bleibt“, was nicht passieren sollte.

**Grafik**

* Umstellung auf OpenGL mit GPU-orientiertem Rendering
* Aktuell noch nicht funktionsfähig und vollständig CPU-basiert
* Vertexberechnung und Rendering werden derzeit vom CPU-Main-Thread übernommen

## Evaluation / Probleme

**Grafik / Performance**

* Rendering und Berechnung laufen aktuell vollständig auf der CPU, was zu erheblichen Performance-Problemen führt. Der Main Thread wird stark belastet, obwohl moderne CPUs mehrere Threads (z. B. 12) zur Verfügung stellen. Das Spiel nutzt davon aktuell nur einen Bruchteil.
* Da die Engine komplett selbst entwickelt wird und die Komplexität bereits hoch ist, ist Optimierung ein zentraler Faktor.
* Möglicherweise erfolgt das Rendering auf einer falschen Skalierungsebene. Gemeint ist nicht 2D vs. 3D, sondern eine unpassende Dimensions- bzw. Maßstabsebene, die anschließend in Pixel umgerechnet wird.
* Das Rendering muss dynamisch und nativ auf Basis der tatsächlich sichtbaren Pixel erfolgen (abhängig von Kamera und Zoom). Derzeit werden vermutlich zu viele Details berechnet, obwohl nur ein kleiner Teil sichtbar ist.

**Predictor**

* Der Predictor ist nicht auf das Reference-Body-System ausgelegt und bleibt an den ursprünglichen Mittelpunkt gebunden. Die Physik selbst bleibt unverändert. Beispiel: Wenn die Sonne bei (0,0) ohne Geschwindigkeit liegt, bleibt das physikalisch so, auch wenn sie visuell im Erde-zentrierten Modus anders erscheint.
* Anfangs zeigte der Predictor keine Schleifenbewegung, sondern verhielt sich weiterhin so, als wäre die Sonne das Zentrum.
* Dies wurde gepatcht, jedoch nicht korrekt: Der Predictor zeigt nun Schleifen, aber diese verändern sich entlang des Orbits und liefern eine sichtbar falsche Vorhersage.

![Veranschaulichung des Problems](https://i.imgur.com/BGl4xWm.png)

# Was kann behoben werden und wie?

* Das Rendering muss explizit parallelisiert und für die GPU ausgelegt werden. GPUs arbeiten mit vielen Kernen und sind für parallele Berechnungen wie Pixel-Rendering optimiert. Die Engine muss entsprechend angepasst werden, insbesondere durch Verlagerung von Vertexberechnung und Rendering in GLSL (OpenGL Shading Language).
* Problematisch ist, dass GLSL nur begrenzte Präzision bei Gleitkommazahlen bietet. Das Spiel arbeitet jedoch mit realen Größenordnungen (auf Meter skaliert), was hohe Genauigkeit erfordert. Aktuell ist die Berechnung möglichst exakt, leidet aber stark unter Performanceproblemen.
* Der Predictor selbst ist nicht das Hauptproblem der Performance. Das Rendering stellt den größten Engpass dar. Durch bessere Nutzung von CPU-Threads und Integration der GPU in den Render-Loop kann die Performance deutlich gesteigert werden.
* Die Anzahl der Predictor-Punkte kann reduziert werden. Aktuell werden Punkte gesetzt und durch gerade Linien verbunden, was bei wenigen Punkten zu sichtbaren Kanten führt.
* Lösung: Verwendung von Bézier-Kurven zur Approximation. Dadurch kann die Punktanzahl reduziert werden, ohne dass visuelle Qualität stark leidet.
* Ansatz: Für jeden Punkt wird der Geschwindigkeitsvektor berücksichtigt und als Kontrollpunkt für die Bézier-Kurve genutzt. Dadurch entsteht eine genauere Approximation der tatsächlichen Bahn.
* Nachteil: Das Rendering der Bézier-Kurve selbst ist ebenfalls rechenintensiv, da es abhängig von Zoomlevel, Sichtbereich und gewünschter Genauigkeit dynamisch berechnet werden muss.
