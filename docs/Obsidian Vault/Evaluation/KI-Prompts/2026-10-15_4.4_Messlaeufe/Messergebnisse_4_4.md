# Messläufe zu Abschnitt 4.4 (KI-erstellt, 15.10.2026)

Skripte in diesem Ordner: `messlaeufe_4_4.py` (Läufe), `zustaende_4_4.py` (Zustände für die Aufnahmen), `aufnahmen_4_4.py` (8K-Aufnahmen aus dem Spiel). Rohwerte: `ergebnisse_4_4.json`.

## Konfiguration

- Weltphysik des Spiels (`physics/world.py`, adaptives RKN4, Toleranzen 0,1 m / 0,001 m/s, Höchstschritt 3000 s aus `config.json`), alle 27 Himmelskörper auf ihren Kepler-Bahnen, Monde aktiv.
- Erde, Jupiter, Saturn mit Exzentrizität 0, große Halbachsen und Massen wie in `solar_system.json`. Alle übrigen Körper unverändert.
- Parkbahn r = 6 678,1366 km, Zielbahn am Saturn r = 60 568 km, Variante 2: r_p,J = 357 460 km (absolute Radien, da das Spiel mit mittleren Radien rechnet).
- Abflug nach 30 min in der Parkbahn als idealer Impuls (Betrag aus der Rechnung mit Spielparametern), Zündpunkt so, dass die Asymptote der Abflughyperbel parallel zur Bahngeschwindigkeit der Erde liegt.
- Korrektur 1 am achten Tag in Flugrichtung (relativ zur Sonne). Ziel: direkter Transfer → Periapsis am Saturn auf der Zielbahn; Variante 2 → r_p,J = 5 R_J; Variante 1 → Periapsis am Jupiter so, dass das Aphel der neuen Bahn auf der Saturnbahn liegt (Definition von Variante 1, in der Simulation nachgeflogen).
- Korrektur 2 (Varianten) beim Verlassen der Einflusssphäre Jupiters in Flugrichtung, Ziel: Periapsis am Saturn auf der Zielbahn.
- Startwinkel: Erde bei 0° im Abflugzeitpunkt, Jupiter bei 180° nach der Hohmann-Flugzeit. Saturns Startwinkel aus dem tatsächlich geflogenen Weg (wo und wann das Schiff die Saturnbahn bzw. sein Aphel erreicht), iteriert.
- Einschwenkimpuls aus der Periapsisgeschwindigkeit nach Gl. (32), v_k,0 = √(μ_Sa/r_0) mit dem Gravitationsparameter des Spiels.
- Rechenwerte unten mit den Spielparametern (G·m, Bahnradien des Spiels), daher leicht anders als Tab. 4.

## Geschwindigkeitsaufwand und Flugzeit

| Transfer | Rechnung in m/s | Simulation in m/s | Differenz | Flugzeit Rechnung in d | Flugzeit Simulation in d |
|---|---|---|---|---|---|
| direkt | 18 069,0 | 18 070,7 | +1,6 | 2 222,7 | 2 214,7 |
| Variante 1 | 16 747,2 | 16 770,2 | +23,0 | 3 464,4 | 3 194,1 |
| Variante 2 | 17 545,1 | 17 623,6 | +78,5 | 2 059,3 | 1 998,7 |

Zusammensetzung der Simulationswerte (m/s):

| Transfer | Abflug | Korrektur 1 | Korrektur 2 | Einschwenken | v∞ am Saturn (Rechnung / Sim.) |
|---|---|---|---|---|---|
| direkt | 7 288,4 | −5,53 | – | 10 776,7 | 5 439 / 5 415 |
| Variante 1 | 6 299,5 | −26,55 | +1,48 | 10 442,7 | 2 420 / 2 346 |
| Variante 2 | 6 299,5 | −17,26 | +2,33 | 11 304,5 | 7 944 / 8 208 |

## Einsparung durch das Gravitationsmanöver

| | Rechnung | Simulation | Abweichung |
|---|---|---|---|
| Variante 1 | 1 321,8 m/s (7,32 %) | 1 300,4 m/s (7,20 %) | −1,6 % |
| Variante 2 | 523,9 m/s (2,90 %) | 447,0 m/s (2,47 %) | −14,7 % |

## Vorbeiflug am Jupiter

| Größe | V1 Rechnung | V1 Simulation | V2 Rechnung | V2 Simulation |
|---|---|---|---|---|
| Periapsisabstand in km | 1 548 558 | 1 684 896 (+8,8 %) | 357 460 | 357 460 (gesetzt) |
| Periapsisgeschwindigkeit (REL V) in km/s | – | 13,45 | – | 27,19 |
| v∞ beim Eintritt in die Einflusssphäre in m/s | 5 644 | 5 465 | 5 644 | 5 483 |
| v∞ beim Austritt in m/s | 5 644 | 5 631 | 5 644 | 5 589 |
| Umlenkwinkel δ (Asymptoten der oskulierenden Hyperbeln) | 92,07° | 90,62° | 133,14° | 133,04° |
| v relativ zur Sonne nach dem Vorbeiflug, auf r_J umgerechnet, in m/s | 14 410 | 14 498 | 17 409 | 17 889 |
| Periapsisdurchgang nach Abflug in d | 997,8 | 982,5 | 997,8 | 975,9 |

v∞ ist beim Austritt um 166 m/s (V1) bzw. 106 m/s (V2) größer als beim Eintritt: Die Sonne wirkt während der Wochen in Jupiters Einflusssphäre (rund 4,8·10¹⁰ m) mit, die Patched-Conics-Näherung vernachlässigt das. Die Bahn nach dem Vorbeiflug ist energiereicher als gerechnet; in Variante 1 läge das Aphel bei gerechnetem Periapsisabstand rund 0,5 AE hinter der Saturnbahn (Korrektur dafür wäre > 100 m/s gewesen), deshalb wurde der Periapsisabstand selbst nachgeführt.

Monde: kleinster Abstand der Sonde zu einem Galileischen Mond V1 Io 1 282 762 km, V2 Europa 507 547 km (Io 687 819 km). Lage der Monde beim Periapsisdurchgang in `ergebnisse_4_4.json` (`monde_bei_peri`).

## Kontrolllauf

Abflug und Korrektur 1 wie Variante 1, Jupiter um 60° verschoben. Größter Sonnenabstand 773,3 Gm nach 990 d, knapp innerhalb der Jupiterbahn (778,6 Gm); Jupiter kommt nie näher als 5,5·10¹¹ m. Die Sonde kehrt auf einer Ellipse mit Pe 151 Gm um und erreicht den Saturn nicht.

## Aufnahmen (8K, Grafiken & Bilder)

- `vorbeiflug_jupitersystem.png`: nicht rotierendes Bezugssystem Jupiters, a) Variante 1, b) Variante 2, gleicher Maßstab, jeweils 24 h vor dem Periapsisdurchgang, Tooltip der Pe-Marke (ETA, REL V).
- `vorbeiflug_sonnensystem.png`: Bezugssystem Sonne, gleicher Ausschnitt um die Sonde 15 d vor dem Periapsisdurchgang, Linie = vorausberechnete Bahn mit Knick am Jupiter, gelbliche Linie = Jupiterbahn.
- `kontrolllauf_saturn.png`: Bezugssystem Sonne, gleicher Maßstab, 150 d nach dem Periapsisdurchgang von Variante 1: a) Variante 1 (Ap 1 433,70 Gm auf der Saturnbahn), b) Kontrolllauf (Ap 772,93 Gm, Rückkehr nach innen).
