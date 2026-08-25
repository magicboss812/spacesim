---
status: entwurf
stand: 2026-08-25
---

# Gliederung

## Leitfrage

> Inwieweit lässt sich reale Himmelsmechanik in einer interaktiven Echtzeitsimulation physikalisch korrekt abbilden, und wo liegen die Grenzen dieser Abbildung?

Funktion der Leitfrage: Sie macht das Spiel zum **Untersuchungsgegenstand** statt zum Anhang. Jedes Theoriekapitel bekommt dadurch einen Grund, im Spiel überprüft zu werden. Ohne diese Klammer zerfällt die Arbeit in "Astro-Referat" plus "Devlog".

## Gewichtung

Basis: 15 Seiten Fließtext. Prozentwerte skalieren, falls der Umfang anders ausfällt.

| Kap. | Titel | Seiten | Anteil |
|---|---|---|---|
| 1 | Einleitung | 1,0 | 7 % |
| 2 | Physikalische Grundlagen | 3,0 | 20 % |
| 3 | Näherung und Navigation: Apollo bis heute | 2,0 | 13 % |
| 4 | Bahnmanöver: Transfers und Swing-by | 3,5 | 23 % |
| 5 | Entwicklung der Simulation | 3,5 | 23 % |
| 6 | Validierung: Wie genau ist die Simulation? | 1,5 | 10 % |
| 7 | Fazit und Ausblick | 0,5 | 4 % |

Astronomischer Anteil (2, 3, 4, 6): 10 Seiten, 66 %.
Eigenanteil-Dokumentation (5): 3,5 Seiten, 23 %.
Damit bleibt es formal eine Astro-Arbeit, obwohl das Spiel überall drinsteckt.

---

## 1 Einleitung

- Ausgangspunkt: Warum eine eigene Simulation statt Literaturarbeit
- Leitfrage
- Abgrenzung: keine Softwaretechnik, kein Code-Listing im Fließtext
- Aufbau der Arbeit

## 2 Physikalische Grundlagen

- 2.1 Newtonsches Gravitationsgesetz, Superposition der Beschleunigungen
- 2.2 Zweikörperproblem, Kegelschnitte, Bahnelemente, Vis-Viva
- 2.3 Das N-Körperproblem: keine geschlossene Lösung ab N = 3
- 2.4 Numerische Integration als Antwort darauf
  - Anfangswertproblem, Schrittweite, lokaler und globaler Fehler
  - Euler, Verlet, RK4, RKN4
  - symplektisch vs. nicht-symplektisch, Energieerhaltung als Prüfgröße
- **Im Spiel**: Screenshot der Beschleunigungsvektoren, Wahl RKN4 kurz benannt, Detail folgt in Kap. 5

Material vorhanden: `Physik.md`, `Integrator.md`. Deckt 2.1 und 2.4 zu etwa 70 % ab.

## 3 Näherung und Navigation: Apollo bis heute

- 3.1 Patched Conics: das N-Körperproblem stückweise als Zweikörperproblem
- 3.2 Apollo: Bordrechner mit begrenzter Rechenleistung, Bodenverfolgung als eigentliche Navigationsquelle, Free-Return-Bahn als Sicherheitsnäherung
- 3.3 Heute: numerische Vollintegration, Ephemeriden, Deep Space Network
- 3.4 Übertrag: Das Spiel integriert vollständig numerisch, nutzt also das moderne Verfahren, nicht Patched Conics

Zweck dieses Kapitels: Es beweist, dass Näherungsverfahren keine Notlösung des Spiels sind, sondern Raumfahrtpraxis. Es verbindet Kapitel 2 mit Kapitel 4. Kurz halten, 2 Seiten.

## 4 Bahnmanöver: Transfers und Swing-by

Ein Kapitel, nicht zwei. Der Swing-by braucht zwingend das Bezugssystem-Argument aus dem Transferteil, getrennt wird beides redundant.

- 4.1 Impulsive Manöver, Delta-v, prograd und retrograd
- 4.2 Hohmann-Transfer: Herleitung, Delta-v-Bilanz, Transferzeit
- 4.3 Bi-elliptischer Transfer, wann er günstiger ist
- 4.4 Gravity Assist
  - planetozentrisch: Betrag der Geschwindigkeit bleibt erhalten, Richtung ändert sich
  - heliozentrisch: Energieübertrag vom Planeten auf die Sonde
  - der scheinbare Widerspruch löst sich über das Bezugssystem auf
- **Im Spiel**: hier liegt der Schwerpunkt der Screenshots und der eigenen Grafiken

Grafikbedarf: Hohmann-Schema, Delta-v-Vergleich Hohmann/bi-elliptisch, Swing-by in zwei Bezugssystemen nebeneinander. Letzteres ist die wichtigste Grafik der ganzen Arbeit.

Hinweis: 2D-Simulation, also Ebenenwechsel weglassen oder in einem Satz als Grenze nennen.

## 5 Entwicklung der Simulation

- 5.1 Zielsetzung: was die Simulation können musste und was bewusst nicht
- 5.2 Architektur in einem Absatz plus Hierarchiegrafik, mehr nicht
- 5.3 Entwicklungsverlauf entlang der Probleme
  - der gescheiterte rein KI-generierte Prototyp und was daraus folgte
  - Integratorwahl: Euler verworfen, Verlet verworfen, RKN4 gewählt
  - Zeitbeschleunigung gegen Schrittweite: der eigentliche Zielkonflikt
  - Referenzsysteme: warum ein wechselbares Bezugssystem nötig war
  - Predictor: Vorhersage der Bahn und deren Fehlerverhalten
  - Custom-Systeme über Konfigurationsdateien
- 5.4 Kompromisse und bewusste Vereinfachungen: 2D, Punktmassen, keine Relativistik, impulsive gegen kontinuierliche Schubmodelle

Aufbau jedes Unterpunkts in 5.3 immer gleich: **Problem, verworfene Option mit Grund, gewählte Option mit Grund, Folge.** Das ist der Unterschied zwischen einer Bewertung als Dokumentation und einer Bewertung als Argumentation.

Material vorhanden: `Anfangsidee.md`, LogBuch.

## 6 Validierung: Wie genau ist die Simulation?

- 6.1 Prüfgrößen: Energiedrift, Positionsabweichung, Phasendrift
- 6.2 Messergebnisse: Drift-Verhältnis 4,23e-7, Toleranzen 1,0 m und 0,001 m/s
- 6.3 Vergleich mit der analytischen Keplerlösung im Zweikörperfall
- 6.4 Ergebnis: sichtbare Abweichung frühestens nach 46 Jahren, im Bestfall nach 92 Jahren, bei stark exzentrischem Orbit
- 6.5 Antwort auf die Leitfrage

Dieses Kapitel entscheidet über die Note. Es ist der Punkt, an dem aus einem Spiel ein Messinstrument wird. Ohne Kapitel 6 ist die Arbeit ein Erfahrungsbericht.

## 7 Fazit und Ausblick

- Antwort auf die Leitfrage in drei Sätzen
- Grenzen
- Ausblick: 3D, weitere Körper, Schubmodelle

---

## Dateiorganisation

```
Obsidian Vault/
  _Arbeit/
    00 Gliederung.md
    01 Einleitung.md
    02 Grundlagen.md
    03 Navigation.md
    04 Manoever.md
    05 Entwicklung.md
    06 Validierung.md
    07 Fazit.md
  Evaluation/          <- Notizen, bleibt
  Material/            <- Screenshots und Grafiken
  Quellen.md
```

Trennung: `_Arbeit` enthält Fließtext, der so in die Abgabe wandert. `Evaluation` enthält Rohmaterial, Rechnungen, Begründungen, Verworfenes. Nicht vermischen.

## Notizdateien in Evaluation

Vorhanden und behalten: Anfangsidee, Integrator, Physik.

Anlegen, weil sie ein Kapitel tragen:

- `Referenzsystem.md` – trägt 4.4 und 5.3, hohe Priorität
- `Zeitbeschleunigung.md` – trägt den zentralen Zielkonflikt in 5.3
- `Predictor.md` – trägt 5.3 und 6.1
- `Manoever.md` – Formeln und Delta-v-Rechnungen für Kapitel 4
- `Validierung.md` – die Messwerte, sauber protokolliert, für Kapitel 6

Anlegen, aber klein halten:

- `Systemkonfiguration.md` statt "Planeten" – Custom-Systeme, JSON-Struktur, Körperparameter. "Planeten" allein wäre nur eine Datentabelle.

Nicht anlegen:

- `Performance.md` – Performance ist unkritisch, das trägt kein Kapitel. Ein Satz in 5.4 reicht.

## Sofort machbar

1. Kapitel 2 entwerfen. `Physik.md` und `Integrator.md` sind zu etwa 70 % Rohmaterial dafür.
2. Vorher drei fachliche Korrekturen in `Integrator.md` und `Physik.md`, siehe unten.
3. `Referenzsystem.md` und `Zeitbeschleunigung.md` als Notiz füllen, danach ist Kapitel 5 schreibbar.

## Korrekturen an den Bestandsnotizen

- `Integrator.md`: "eine eindeutige Lösung bzw. Funktion ist nicht bekannt (irrational)" ist sachlich schief. Korrekt: Für N >= 3 existiert keine geschlossene analytische Lösung in elementaren Funktionen.
- `Integrator.md`: RKN4 ist **nicht symplektisch**. Der Satz "deutlich niedrigere Fehlertoleranz sowie stabilere Lösungen auf Langzeit" ist gegenüber Verlet nicht haltbar. Korrektes Argument: RKN4 hat einen kleineren lokalen Fehler bei weniger Kraftauswertungen pro Schritt und ist damit für die geforderte Genauigkeit effizienter. Der säkulare Energiedrift bleibt vorhanden, ist mit 4,23e-7 aber über die relevante Spieldauer irrelevant. Genau das ist das eigentliche Argument, und es ist stärker als das falsche.
- `Physik.md`: G steht dort als 6,6730831e-11. Der CODATA-Wert ist 6,67430e-11. Prüfen, was im Code steht, und in der Arbeit den Literaturwert zitieren.
- `Physik.md`: Malzeichen als `\cdot` statt `\times`.
