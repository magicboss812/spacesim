## Inhaltsverzeichnis

1. [Ausgangslage im April 2026](#ausgangslage-im-april-2026)
2. [Entwicklung nach dem April-Stand](#entwicklung-nach-dem-april-stand)
   - [Reference-Frame-System](#reference-frame-system)
   - [Orbit- und Bahndaten](#orbit--und-bahndaten)
   - [Zeitbasierte Predictor-Darstellung](#zeitbasierte-predictor-darstellung)
   - [Predictor und Integratoren](#predictor-und-integratoren)
   - [Rendering der Predictor-Linie](#rendering-der-predictor-linie)
   - [Schiffskontrolle](#schiffskontrolle)
   - [Weltphysik](#weltphysik)
3. [Aktueller Entwicklungsstand](#aktueller-entwicklungsstand)
4. [Aktuelle Probleme](#aktuelle-probleme)
5. [Spielstatus](#spielstatus)

---

# Ausgangslage im April 2026

Der Stand `04/2026` war ein wichtiger Zwischenstand, da dort mehrere Kernideen der Simulation bereits vorhanden waren. Viele dieser Systeme waren aber noch prototypisch, technisch instabil oder nicht vollständig miteinander verbunden.

## Reference-Body-System

Das **Reference-Body-System** war bereits als zentrale Idee vorhanden. Körper konnten als Bezugspunkt der Darstellung ausgewählt werden. Dadurch sollte es möglich werden, die Simulation nicht nur aus einem absoluten, sonnenzentrierten Blickwinkel zu betrachten, sondern auch relativ zu einem ausgewählten Planeten.

Beispiel:

```text
Erde als Reference Body
→ Erde wirkt als Zentrum der Darstellung
→ Sonne und andere Körper bewegen sich relativ zur Erde
```

Das System war zu diesem Zeitpunkt jedoch nur teilweise funktional.

### Probleme im April-Stand

- Das System war noch stark prototypisch.
- Der Predictor war noch nicht sauber auf Reference Bodies ausgelegt.
- Dadurch entstanden visuelle Fehler und Konflikte in der Berechnung.
- Die Predictor-Linie bildete noch keine korrekten Schleifen in körperzentrierten Bezugssystemen.
- Die Energieerhaltung bzw. Bahndarstellung des Predictors war sichtbar ungenau.
- Die dargestellte Umlaufbahn des Schiffs wich teilweise stark von der erwarteten Bahn ab.

---

## Predictor

Der Predictor wurde im April-Stand bereits auf einen **vierstufigen Runge-Kutta-Integrator (RK4)** umgestellt. Dadurch wurde die Flugbahn genauer als mit einfacheren Integrationsverfahren. Besonders die Energieerhaltung wurde dadurch verbessert.

### Vorteile

- genauere Vorausberechnung der Flugbahn
- bessere Energieerhaltung als vorher
- stabilere Bahnkurven bei normalen Bedingungen

### Nachteile

- fixe Schrittweite
- keine adaptive Fehlerkontrolle
- hohe Kosten bei langen Zukunftsvorhersagen
- schwache Behandlung weit in der Zukunft liegender Punkte
- unzuverlässige Verbindung mit dem Reference-Body-System

---

## Snapshot-Konzept des Predictors

Um die Performance zu verbessern, wurde ein Snapshot-Konzept umgesetzt.

Die Grundidee:

```text
Nicht jedes Frame 10.000 Punkte neu berechnen,
sondern:
1. komplette Linie einmal berechnen
2. alte Punkte entfernen
3. neue Punkte am Ende ergänzen
4. vorhandene Punkte weiterverwenden
```

Dadurch sollte verhindert werden, dass der Predictor pro Frame vollständig neu berechnet werden muss.

### Problem

Das Snapshot-System stoppte nach einiger Zeit teilweise ohne erkennbaren Grund. Neue Punkte wurden dann nicht mehr korrekt am Ende ergänzt. Dadurch blieb der Predictor stehen oder zeigte veraltete Daten an.

---

## Screen-Culling im April-Stand

Das erste Screen-Culling entfernte Predictor-Punkte außerhalb des sichtbaren Bildschirms aus dem Rendering.

Das war grundsätzlich sinnvoll, führte aber zu einem neuen Problem:

```text
sichtbarer Punkt unten rechts
→ viele Punkte außerhalb des Bildschirms
→ sichtbarer Punkt oben links
```

Der Renderer verband dann den letzten sichtbaren Punkt unten rechts direkt mit dem nächsten sichtbaren Punkt oben links. Dadurch entstand eine gerade Linie quer über den Bildschirm, obwohl die eigentliche Umlaufbahn außerhalb des sichtbaren Bereichs weiterverlief.

### Beispiel

```text
eigentliche Bahn:
sichtbar → außerhalb → außerhalb → sichtbar

fehlerhafte Darstellung:
sichtbar ───────────── sichtbar
```

Dieses Problem entstand, weil sichtbare Punkte weiterhin wie eine zusammenhängende Linie behandelt wurden, obwohl unsichtbare Abschnitte dazwischenlagen.

---

## Rendering

Im April-Stand wurde das Rendering bereits auf **OpenGL** umgestellt. Außerdem wurden Orbit-Lines ergänzt.

### Bereits vorhandene Funktionen

- OpenGL-basierte Darstellung
- erste GPU-orientierte Struktur
- Orbit-Lines
- farbliche Anpassung der Orbit-Lines an die jeweiligen Körper
- hierarchiekompatible Darstellung von Bahnen

### Probleme

- Die OpenGL-Umstellung war noch nicht vollständig funktional ausgenutzt.
- Viele Berechnungen liefen weiterhin auf der CPU.
- Der Main-Thread wurde stark belastet.
- Das Rendering war besonders bei langen Predictor-Linien ein Performance-Problem.

---

## Schiffskontrolle

Die Schiffskontrolle konnte bereits den Geschwindigkeitsvektor des Schiffs verändern. Die Beschleunigung wurde durch einen Thrust-Vector bestimmt, der sich aus Richtung und Stärke des Schubs ergab.

### Problem

Die Schiffskontrolle war noch nicht ausreichend mit dem Reference-Body-System verbunden. Dadurch konnte die visuelle Ausrichtung des Schubs von der erwarteten Richtung im gewählten Bezugssystem abweichen.

---

## Performance-Problem

Ein großes Problem war die Performance bei aktivem Reference-System, besonders wenn der erste Körper, also ein absoluter Elternkörper wie die Sonne, als Reference Body ausgewählt wurde.

Das führte zu hoher Auslastung, weil viele Elemente relativ zu diesem Bezugssystem transformiert oder neu berechnet werden mussten.

---

# Entwicklung nach dem April-Stand

Nach dem Stand `04/2026` verschob sich der Schwerpunkt der Entwicklung deutlich.

Im April ging es vor allem darum, zentrale Funktionen überhaupt umzusetzen. Danach ging es zunehmend darum, diese Funktionen stabiler, genauer und performanter miteinander zu verbinden.

---

## Reference-Frame-System

Das Reference-System wurde nach dem April-Stand stärker als eigenständiges **Plotting-Frame-System** verstanden.

Dabei gilt:

```text
Physik bleibt im absoluten Raum.
Rendering transformiert die Darstellung in das gewählte Bezugssystem.
```

Das System beeinflusst also vor allem die visuelle Darstellung:

- Predictor-Linien
- Körperpositionen im Bild
- Text- und HUD-Elemente
- Vektoren
- relative Geschwindigkeiten
- Orientierung des Schiffs

Die Idee wurde unter anderem von der KSP-Modifikation **Principia** inspiriert. Dort werden ebenfalls verschiedene Bezugssysteme für die Darstellung orbitaler Bewegungen genutzt.

### Ziel

Das Reference-Frame-System sollte nicht nur Positionen verschieben, sondern langfristig auch weitere Größen sinnvoll transformieren:

- Position
- Orientierung
- Geschwindigkeit
- Predictor-Darstellung
- Thrust-Vector
- Velocity-Vector
- visuelle Bahnlinien

---

## Orbit- und Bahndaten

Auch die Bahndarstellung wurde erweitert. Besonders wichtig war das **Argument der Periapsis**.

Damit kann eine Umlaufbahn nicht nur durch Größe und Exzentrizität beschrieben werden, sondern auch durch ihre Ausrichtung.

### Warum war das wichtig?

Für Reference Frames und Predictor-Darstellung reicht eine einfache Umlaufbahn um den Ursprung nicht aus. Bahnen müssen korrekt gedreht werden können.

Vereinfacht:

```text
vorher:
Orbit liegt immer in Standardausrichtung

nachher:
Orbit kann durch Argument der Periapsis gedreht werden
```

Dadurch lassen sich elliptische Bahnen realistischer und flexibler darstellen.

---

## Zeitbasierte Predictor-Darstellung

Ein zentraler Fortschritt war die Erweiterung der Predictor-Punkte um Zeitinformationen.

Vorher enthielt ein Predictor-Punkt hauptsächlich:

```text
x, y
```

Später wurde daraus:

```text
x, y, t
```

Das bedeutet: Jeder Predictor-Punkt enthält nicht nur seine Position, sondern auch den zukünftigen Zeitpunkt, zu dem diese Position erreicht wird.

### Warum ist das wichtig?

Für einen Reference Body reicht es nicht, nur die zukünftige Position des Schiffs zu kennen. Auch die zukünftige Position des Bezugskörpers muss bekannt sein.

Korrekt ist:

```text
Schiff(t) - Erde(t)
```

Problematisch ist:

```text
Schiff(t) - Erde(jetzt)
```

Wenn die Erde als Bezugskörper gewählt wird, muss die Predictor-Linie relativ zur zukünftigen Erdposition dargestellt werden. Sonst entstehen falsche Bahnen oder falsch gedrehte Schleifen.

---

## Predictor und Integratoren

Nach dem April-Stand wurde der Predictor mehrfach überarbeitet. Dabei ging es zunächst um Performance, später stärker um Genauigkeit und Stabilität.

---

### Rolling-System

Anfangs lag der Fokus auf einem **Rolling-System**.

Die Idee:

```text
alte Predictor-Punkte entfernen
neue Punkte am Ende ergänzen
bestehende Punkte weiterverwenden
```

Dadurch sollte die Performance deutlich verbessert werden, weil nicht die gesamte Predictor-Linie ständig neu berechnet werden muss.

### Ergebnis

Die Resultate waren widersprüchlich.

- Die erwartete Performance-Verbesserung war nicht eindeutig.
- Das System wurde komplexer.
- Es entstanden neue Fehler durch veraltete oder nicht korrekt ergänzte Punkte.
- Es wurde unklar, ob die Berechnung selbst oder das Rendering der eigentliche Engpass war.

Diese Phase war trotzdem wichtig, weil sie gezeigt hat, dass das Performance-Problem nicht nur im Predictor selbst lag.

---

### ASPI-Idee

Danach wurde die Idee eines adaptiven Integrators verfolgt: **ASPI**.

ASPI steht hier für:

```text
Adaptive Symplectic Predictor Integrator
```

Die Grundidee war eine Kombination aus:

- Leapfrog für große Distanzen
- RK4 für Bereiche nahe an massereichen Körpern

### Gedanke

```text
weit weg von Körpern:
→ geringere Genauigkeit reicht aus
→ Leapfrog spart Rechenleistung

nah an Körpern:
→ hohe Genauigkeit nötig
→ RK4 wird verwendet
```

### Problem

Der Übergang zwischen Leapfrog und RK4 war nicht glatt genug. Außerdem ist die Integration sequenziell. Das bedeutet:

```text
spätere Punkte hängen von früheren Punkten ab
```

Wenn Leapfrog in einem frühen Abschnitt bereits ungenaue Ergebnisse liefert, kann RK4 später nicht mehr vollständig korrigieren, weil es auf diesen ungenauen Ausgangswerten aufbaut.

Dadurch war ASPI als Ansatz zwar interessant, aber praktisch nicht stabil genug.

---

### Wechsel auf RK45

Ein Wechsel auf **RK45** war deshalb sinnvoller.

RK45 arbeitet ähnlich wie RK4, besitzt aber eine eingebaute Fehlerabschätzung. Dabei werden Ergebnisse unterschiedlicher Ordnung verglichen, um den lokalen Fehler zu bestimmen.

### Vorteile

- adaptive Schrittweite
- bessere Fehlerkontrolle
- höhere Genauigkeit in kritischen Bereichen
- größere Schritte in einfachen Bereichen
- bessere Balance zwischen Performance und Genauigkeit

Vereinfacht:

```text
Fehler klein
→ größerer Zeitschritt möglich

Fehler groß
→ kleinerer Zeitschritt nötig
```

Dadurch wurde der Predictor deutlich stabiler.

---

### Zeitabhängige Körperpositionen im Predictor

Ein weiteres Problem war die ungenaue Prediction im Reference-Frame-System. Die Umlaufbahn schien sich im Verlauf teilweise zu drehen oder hatte keinen sauberen Bezug zum Reference Body.

Die Lösung war, zukünftige Körperpositionen in die Predictor-Darstellung einzubeziehen.

Dadurch konnte der Predictor nicht nur die zukünftige Position des Schiffs darstellen, sondern auch die zukünftige Bewegung des Bezugskörpers berücksichtigen.

Das löste das zentrale Problem:

```text
Predictor relativ zu einem bewegten Reference Body
```

---

## Predictor RKN

Später wurde zusätzlich das **Runge-Kutta-Nyström-Verfahren (RKN)** untersucht und eingebaut.

Dieses Verfahren eignet sich besonders für Systeme, bei denen die Beschleunigung direkt von Position und Zeit abhängt. Genau das ist bei orbitaler Bewegung der Fall.

### Warum RKN?

Die Bewegung in der Simulation folgt im Kern:

```text
Position → Beschleunigung → neue Geschwindigkeit → neue Position
```

RKN-Verfahren sind für solche Gleichungen oft effizienter als allgemeine RK-Verfahren, weil sie direkt mit Gleichungen zweiter Ordnung arbeiten.

### Vorteile

- effizienter für gravitative Bewegung
- bessere Kontrolle über Positions- und Geschwindigkeitsfehler
- sinnvoll für enge Umlaufbahnen
- adaptive Zeitschritte möglich
- bessere Stabilität bei kritischen Bahnbereichen

### Bedeutung für den Predictor

Der Predictor kann dadurch genauer und kontrollierter berechnen, wann kleinere Schritte nötig sind. Besonders bei engen Umlaufbahnen um Planeten ist das wichtig, weil dort kleine Fehler schnell sichtbar werden.

---

## Rendering der Predictor-Linie

Das Rendering stellte sich nach weiteren Tests als einer der wichtigsten Performance-Engpässe heraus.

Anfangs wurde angenommen, dass vor allem die Berechnung des Predictors teuer sei. Später wurde klar, dass auch das Zeichnen der Linie selbst sehr viel Leistung kosten kann.

---

### Problem: zu viele Punkte im Bildschirmraum

Predictor-Punkte wurden teilweise zu detailliert gerendert. Das heißt: Es wurden viele Punkte verarbeitet, obwohl der Unterschied auf dem Bildschirm kaum sichtbar war.

Problematisch war dabei die Trennung zwischen:

```text
Weltkoordinaten
```

und

```text
Bildschirmkoordinaten
```

Eine Predictor-Linie kann in Weltkoordinaten extrem lang sein, aber auf dem Bildschirm nur wenige Pixel einnehmen. Umgekehrt kann ein kleiner Weltabschnitt bei starkem Zoom sehr groß erscheinen.

Deshalb muss das Rendering abhängig vom Bildschirmraum arbeiten, nicht nur abhängig von Weltmetern.

---

### Verbesserung: sichtbare Abschnitte statt einer Punktliste

Das Screen-Culling wurde erweitert.

Vorher wurden Punkte außerhalb des Bildschirms entfernt. Das führte aber zu falschen Verbindungen zwischen getrennten sichtbaren Abschnitten.

Nachher wird die Linie stärker in sichtbare Teilabschnitte getrennt.

```text
vorher:
sichtbarer Punkt → unsichtbare Punkte → sichtbarer Punkt
= direkte Verbindung

nachher:
sichtbarer Abschnitt 1
sichtbarer Abschnitt 2
= getrennte Linien
```

Dadurch wurden die falschen Linien quer über den Bildschirm weitgehend beseitigt.

---

### Verbesserung: begrenzte Punktanzahl beim Rendering

Der Renderer verarbeitet nicht mehr automatisch alle Predictor-Punkte. Stattdessen wird kontrolliert:

- wie viele Rohpunkte existieren
- wie viele Punkte geprüft werden
- wie viele Punkte sichtbar sind
- wie viele Punkte tatsächlich gezeichnet werden

Dadurch wird verhindert, dass eine sehr lange Predictor-Linie das Rendering unnötig stark belastet.

---

### Verbesserung: Zwischenspeicherung der gerenderten Linie

Die vorbereitete Predictor-Linie wird zwischengespeichert. Wenn sich Kamera, Zoom oder Predictor-Daten nicht wesentlich ändern, muss die Linie nicht jedes Frame komplett neu vorbereitet werden.

Das reduziert die CPU-Last.

---

### Weiterhin bestehendes Problem

Das Culling scheint teilweise noch falsch skaliert zu sein. Besonders bei starkem Zoom-in werden offenbar zu viele Punkte entfernt, obwohl auf dem Bildschirm eigentlich genug Platz für eine detailliertere Darstellung vorhanden wäre.

Das betrifft zum Beispiel Planetenorbits oder enge lokale Bahnabschnitte.

---

## Schiffskontrolle

Die Schiffskontrolle war im April-Stand noch kaum auf das Reference-Frame-System ausgelegt.

Später wurde die Ausrichtung des Thrust-Vectors besser an den gewählten Reference Body angepasst.

### Verbesserung

Der Thrust-Vector wird nun passend zum Reference Frame rotiert. Dadurch wirkt die Beschleunigung in der erwarteten visuellen Richtung auf den Velocity-Vektor.

Das ist wichtig, weil der Spieler das Schiff aus dem aktuell gewählten Bezugssystem steuert. Wenn Darstellung und Schubrichtung nicht zusammenpassen, fühlt sich die Steuerung falsch an.

### Zusätzliche Visualisierung

Im Zuge der Fehlerbehebung wurden auch visuelle Hilfen ergänzt:

- Velocity-Vector
- Thrust-Vector
- Anzeige der relativen Geschwindigkeit
- Geschwindigkeitsanzeige unter dem Schiff

Dadurch sind Transfers bereits besser nachvollziehbar und teilweise praktisch möglich.

---

## Weltphysik

Um die echte Bewegung des Schiffs besser mit dem Predictor vergleichen zu können, wurde auch die Weltphysik angepasst.

Das RK4-Verfahren für dynamische Körper, insbesondere das Schiff, wurde durch ein RKN-Verfahren ersetzt.

### Ziel

Predictor und echte Simulation sollen numerisch besser vergleichbar sein.

Wenn der Predictor ein anderes Verfahren nutzt als die reale Bewegung des Schiffs, können Abweichungen entstehen, die nicht aus der Physik, sondern aus unterschiedlichen Integrationsmethoden stammen.

---

# Aktueller Entwicklungsstand

**Wichtige Fortschritte**

| Bereich | Fortschritt |
|---|---|
| Reference Frames | stärkeres eigenes Transformationssystem |
| Predictor | zeitbasierte Punkte und adaptive Integration |
| Rendering | bessere Culling-Logik und Punktbegrenzung |
| Schiffskontrolle | bessere Ausrichtung im Reference Frame |
| Weltphysik | RKN-Verfahren für dynamische Körper |
| Debugging | bessere Messbarkeit von Fehlern und Performance |

---

# Aktuelle Probleme

---

## Integration

Die Integration ist bei Körpern, die um die Sonne kreisen, teilweise noch ungenau. Das Problem ist nicht immer gravierend, aber sichtbar.

Tests mit Sonne, Erde und Mond zeigten, dass das Schiff in der Nähe der Erde teilweise eine ungenaue Energieentwicklung aufweist.

Mögliche Ursache:

- hohe Geschwindigkeit der Planeten
- hohe relative Geschwindigkeit des Schiffs
- komplexe Überlagerung von Sonnen- und Erdgravitation
- Unterschiede zwischen visueller Reference-Frame-Darstellung und absoluter Backend-Physik

Wichtig ist:

```text
Das Reference-System ist derzeit hauptsächlich visuell.
Die Physik selbst bleibt im absoluten Raum.
```

Das bedeutet: In einem geozentrischen Modus wirkt es für den Betrachter so, als würde sich die Sonne um die Erde bewegen. Im Backend bleibt die Sonne aber weiterhin der stationäre bzw. absolute Zentralkörper.

Das ist bewusst einfacher als ein vollständig transformiertes physikalisches Bezugssystem, aber es erzeugt technische Grenzen.

---

## Performance

Die Performance wurde verbessert, ist aber noch nicht vollständig gelöst.

Besonders die Länge des Predictors in Weltkoordinaten beeinflusst weiterhin die Leistung. Das gilt auch dann, wenn nicht mehr direkt mehr Punkte in `predictor.py` erzeugt werden.

Das zeigt, dass weiterhin nicht nur die Berechnung, sondern auch die Verarbeitung und Darstellung langer Linien eine Rolle spielt.

---

## Rendering-Culling

Das Screen-Culling funktioniert besser als vorher, ist aber noch nicht perfekt.

### Noch bestehendes Problem

Bei starkem Zoom-in werden teilweise zu viele Predictor-Punkte entfernt. Dadurch können Planetenorbits oder enge Bahnabschnitte weniger detailliert erscheinen, als es der verfügbare Bildschirmplatz eigentlich erlauben würde.

Das deutet darauf hin, dass die Culling- oder Sampling-Skalierung noch nicht vollständig korrekt an den Kamerazoom angepasst ist.

---

## Spielstatus

Das Spiel ist nun wirklich spielbar. Abgesehen von klaren Performance Problemen, sind Planeten-Transfers und Flybys möglich.

Das Hierarchien-System erlaubt es die eigene Erstellung von weiteren Systemen oder die Änderung von Werten. Die `solar_system.json` Datei zeigt schon einigermaßen, wie Systeme erstellt werden können.

**Wie man das Spiel startet:** 

Requirements: `pip install pyopengl pyopengl_accelerate pygame astropy poliastro numba`

Starten: `python test.py`
