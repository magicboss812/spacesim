# Performance Update für Anfang Mai

Stand: Anfang Mai 2026

## Kurzüberblick

Dieses Update konzentriert sich auf die Performance des Predictors und des Renderings. Nach mehreren Tests wurde klar: Die reine Berechnung der Flugbahn war nicht mehr der größte Engpass. Besonders das Zeichnen der Predictor-Linie hat unnötig viel Leistung gekostet, vor allem bei langen Vorhersagen über große Distanzen.

Deshalb wurden zwei zentrale Bereiche überarbeitet:

1. der Integrator des Predictors
2. das Rendering der Predictor-Linie

---

## Neuer Predictor-Integrator: ASPI

Der bisherige Runge-Kutta-4-Integrator wurde für den Predictor durch **ASPI** ersetzt.

**ASPI** ist ein adaptiver Predictor-Integrator. Das bedeutet: Die Schrittweite wird nicht mehr überall gleich behandelt, sondern abhängig von der Situation angepasst.

In einfachen Bereichen der Flugbahn, zum Beispiel weit entfernt von starken Gravitationseinflüssen, kann der Predictor größere Schritte verwenden. In kritischeren Bereichen, etwa in der Nähe eines massereichen Körpers, wird genauer gerechnet.

Dadurch wird Rechenleistung gezielter eingesetzt:

- weniger unnötige Berechnung in ruhigen Bereichen
- mehr Genauigkeit dort, wo die Bahn stärker beeinflusst wird
- bessere Balance zwischen Performance und sichtbarer Qualität

Der Predictor berechnet also nicht mehr stur überall gleich detailliert, sondern passt sich stärker an die tatsächliche Situation an.

---

## Rendering-Optimierung

Nach dem Integrator-Wechsel zeigte sich, dass die Predictor-Berechnung bereits schnell genug war. Das eigentliche Problem lag danach im Rendering.

Vorher konnte eine lange Predictor-Linie viele Punkte erzeugen, die zwar berechnet wurden, aber auf dem Bildschirm nicht immer sinnvoll sichtbar waren. Trotzdem musste der Renderer diese Punkte verarbeiten. Dadurch sank die Performance besonders dann, wenn der Predictor über große Welt-Distanzen angezeigt wurde.

Das Rendering wurde deshalb gezielter begrenzt und optimiert:

- Es werden nicht mehr automatisch alle Predictor-Punkte pro Frame verarbeitet.
- Die Anzahl der tatsächlich gezeichneten Punkte wird begrenzt.
- Punkte außerhalb oder weit außerhalb des sichtbaren Bereichs werden stärker gefiltert.
- Die Predictor-Linie wird stärker im Bildschirmraum vereinfacht.
- Die Datenübergabe an OpenGL wurde verbessert.
- Die Predictor-Linie kann getrennt vom FXAA-Postprocessing gerendert werden, damit sie nicht unnötig weichgezeichnet wird.
- Zusätzliche Render-Benchmarks helfen dabei, die tatsächlichen Engpässe zu erkennen.

Damit hängt die Performance weniger stark davon ab, wie lang der Predictor in Weltmetern ist.

---

## Ergebnis

Die Simulation läuft dadurch spürbar stabiler, besonders bei langen Predictor-Linien. Der Predictor kann weiterhin große zukünftige Bahnabschnitte anzeigen, ohne dass automatisch jeder einzelne Punkt auch teuer gerendert werden muss.

Wichtig ist dabei die Trennung zwischen:

- berechneter Predictor-Länge
- gespeicherten Predictor-Punkten
- tatsächlich gerenderten Punkten

Diese Trennung macht das System deutlich flexibler. Der Predictor kann intern weiterhin viele Informationen besitzen, während der Renderer nur das zeichnet, was für die aktuelle Ansicht sinnvoll ist.

---

## Gelöste Probleme

Der Predictor ist nun deutlich stabiler, weil die alte ASPI-Lösung durch einen **RK45-Integrator** ersetzt wurde. RK45 arbeitet mit adaptiven Schritten: In einfachen Bereichen kann größer gerechnet werden, während in schwierigeren Bereichen automatisch kleinere und genauere Schritte verwendet werden.

Zusätzlich besitzt RK45 eine Fehlerkontrolle. Dadurch wird nicht nur ungefähr geschätzt, wie groß ein Schritt sein sollte, sondern die Berechnung kann ihre eigene Genauigkeit besser überprüfen und bei Bedarf korrigieren. Das sorgt für eine deutlich zuverlässigere und stabilere Predictor-Linie, auch wenn sich Geschwindigkeit, Entfernung oder Gravitationseinfluss verändern.

Auch das Rendering des Predictors wurde überarbeitet. Die Predictor-Linie berücksichtigt nun Reference-Körper korrekt. Dadurch wird die Vorhersage nicht mehr nur im absoluten Raum gezeichnet, sondern passend zum gewählten Referenzsystem dargestellt. Wenn also ein Reference-Körper ausgewählt ist, wird dessen Bewegung in der Darstellung der zukünftigen Flugbahn mit einbezogen.

Damit sind zwei zentrale Probleme gelöst:

- Der Predictor rechnet nun stabiler und genauer durch adaptive RK45-Schritte mit Fehlerkontrolle.
- Die Darstellung der Predictor-Linie ist nun mit Reference-Körpern kompatibel und passt sich dem gewählten Referenzsystem an.
