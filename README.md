---

# Stand bis 31.05.2026

> [!IMPORTANT]
> Dieser Abschnitt dokumentiert alle wesentlichen Änderungen und Verbesserungen, die seit dem Commit vom **9. Mai 2026** bis zum **31. Mai 2026** in die Simulation eingeflossen sind.

---

## Inhaltsverzeichnis

- [Inhaltsverzeichnis](#inhaltsverzeichnis)
- [Physik der Schiffsintegration](#physik-der-schiffsintegration)
  - [Gravitationsquellen bewegen sich mit der Zeit](#gravitationsquellen-bewegen-sich-mit-der-zeit)
  - [Auswirkung](#auswirkung)
- [Adaptive Präzision im Predictor](#adaptive-präzision-im-predictor)
  - [Schrittweite gekoppelt an den Abtastabstand](#schrittweite-gekoppelt-an-den-abtastabstand)
- [Darstellung weit entfernter Körper](#darstellung-weit-entfernter-körper)
  - [Icon-Swap bei kleinem Bildschirmradius](#icon-swap-bei-kleinem-bildschirmradius)
  - [Auswirkung im Überblick](#auswirkung-im-überblick)
- [Orbit-Spur-Culling](#orbit-spur-culling)
- [Rendering-Pipeline der Vorhersagelinie](#rendering-pipeline-der-vorhersagelinie)
  - [Reihenfolge: RDP vor Densify](#reihenfolge-rdp-vor-densify)
- [Weitere Performance-Verbesserungen](#weitere-performance-verbesserungen)
- [Gesamtbild](#gesamtbild)
  - [Status-Checkliste (31.05.2026)](#status-checkliste-31052026)

---

## Physik der Schiffsintegration

### Gravitationsquellen bewegen sich mit der Zeit

Bisher verhielten sich alle Planeten und Monde aus Sicht des Integrators als **ruhende Massepunkte** — ihre Positionen wurden für den gesamten Integrationsschritt eingefroren, unabhängig davon, wie weit in der Zukunft der jeweilige Punkt liegt.

Physikalisch bedeutete das: Das Gravitationsfeld, das auf das Schiff wirkt, gehörte zum Zeitpunkt $t_0$, nicht zum Zeitpunkt $t$.

```
vorher:
  F_Schiff(t) = G·M / |r_Schiff(t) - r_Planet(t₀)|²

jetzt:
  F_Schiff(t) = G·M / |r_Schiff(t) - r_Planet(t)|²
```

Planeten und Monde extrapolieren ihre eigene Kepler-Position jetzt **analytisch für jeden beliebigen Zeitpunkt**, ohne den Simulationszustand zu verändern. Der Epoch-Bookmark (`_kepler_ref_time`, `_kepler_ref_theta`) wird nach jedem Weltschritt gesetzt, sodass die Extrapolation immer vom aktuell bestätigten Bahnzustand ausgeht.

> [!NOTE]
> Diese Änderung betrifft ausschließlich die Präzision der Schiffsbewegung über längere Zeiträume. Die Planetenbahnen selbst bleiben unverändert skriptgesteuert.

### Auswirkung

| Szenario | Vorher | Jetzt |
|---|---|---|
| Kurzmanöver (Sekunden) | kein sichtbarer Unterschied | kein sichtbarer Unterschied |
| Langer Transfer (Stunden–Tage) | Planetenpositionen statisch eingefroren | Gravitationsfeld korrekt zeitabhängig |
| Flyby an schnell bewegtem Körper | Kraftrichtung leicht falsch | Kraftrichtung zur korrekten Körperposition |

---

## Adaptive Präzision im Predictor

### Schrittweite gekoppelt an den Abtastabstand

Der Predictor berechnet eine Flugbahn bis weit in die Zukunft. Nahe Punkte (wenige Minuten) brauchen hohe Genauigkeit; Punkte, die Stunden oder Tage entfernt liegen, werden mit einem viel gröberen Zeitraster abgetastet.

Bisher war die Integratorschrittweite unabhängig von diesem Abtastraster. Das Ergebnis: Der Integrator arbeitete auf weit entfernten Abschnitten genauso fein wie nahe am Schiff — die Arbeit wurde geleistet, die Punkte aber trotzdem zu selten gespeichert.

Jetzt gilt: Ist das effektive Abtastintervall gröber als die Basispräzision, darf der Integrator **proportional größere Schritte mit proportional gelockerter Toleranz** nehmen.

```
Abtastintervall ≤ Basispräzision
  → Identisches Verhalten wie zuvor (kein Unterschied)

Abtastintervall > Basispräzision
  → max_dt ↑ proportional zum Verhältnis
  → Toleranz ↑ skaliert mit Verhältnis^8
  → Schrittzahl ~ Punktzahl statt ~ Bogenlänge
```

> [!NOTE]
> Nahe Vorbeiflüge an Planeten erzeugen lokal hohe Beschleunigungsgradienten. Dort übersteigt der tatsächliche Fehler auch die gelockerte Toleranz, und der Integrator unterteilt weiterhin bis zur Mindestschrittweite — die Sicherheit bei kritischen Passagen bleibt erhalten.

---

## Darstellung weit entfernter Körper

### Icon-Swap bei kleinem Bildschirmradius

Beim Herauszoomen schrumpfen Körper auf dem Bildschirm. Unterhalb einer Schwelle von **4 Pixeln echtem Bildschirmradius** wird der vollständige Körper (Scheibe, Atmosphäre, Glow) nicht mehr gezeichnet. Stattdessen erscheint ein **Positions-Icon konstanter Bildschirmgröße**.

```
true_radius_px ≥ 4 px  →  voller Körper (Scheibe + Glow + Atmosphäre)
true_radius_px  < 4 px  →  Positions-Icon (4 px, konstant, kein weiteres Schrumpfen)
```

Wichtig: Die Schwelle und der Icon-Radius sind identisch. Der Übergang ist deshalb nahtlos — kein leerer Frame, keine Doppelzeichnung.

Körper, die vollständig außerhalb des Bildschirms liegen, werden darüber hinaus **komplett übersprungen** (Off-screen-Culling), ohne dass Shader oder Geometrie aufgerufen werden.

### Auswirkung im Überblick

| Zoom-Stufe | Darstellung |
|---|---|
| Nah (echter Radius ≥ 4 px) | Vollständiger Körper mit Atmosphäre und Glow |
| Mittel (Übergang bei ~4 px) | Nahtloser Wechsel zum Icon |
| Weit (Icon-Modus) | Konstant 4 px großes Positionsmarker-Icon |
| Vollständig off-screen | Kein Render-Aufruf |

---

## Orbit-Spur-Culling

Wenn eine Orbit- oder Referenzspur beim aktuellen Zoom einen Bildschirmbereich von weniger als **2 Pixeln** abdeckt, wird sie vollständig übersprungen. Eine sub-pixel-große Spur ist für das Auge ohnehin nicht wahrnehmbar; die Position des Körpers ist durch sein Icon repräsentiert.

```
Bounding-Box der Spur auf dem Bildschirm < 2 px  →  Spur wird nicht gezeichnet
```

---

## Rendering-Pipeline der Vorhersagelinie

### Reihenfolge: RDP vor Densify

> [!NOTE]
**RDP (Ramer-Douglas-Peucker)** ist ein Algorithmus zur Linienvereinfachung. Er nimmt eine Folge von Punkten, die eine Kurve beschreiben, und entfernt alle Punkte, die geometrisch nicht notwendig sind — also Punkte, die so nah an der Verbindungslinie ihrer Nachbarn liegen, dass sie auf dem Bildschirm keinen sichtbaren Unterschied machen. Das Ergebnis ist eine geometrisch nahezu identische Kurve mit deutlich weniger Punkten.

```
Eingabe:  ● ● ● ● ● ● ● ● ● ● ● ●   (viele dichte Punkte)
Ausgabe:  ●         ●     ●     ●     (nur geometrisch wichtige Punkte)
```

Die Pipeline zur Vorbereitung der Predictor-Linie lief bisher in dieser Reihenfolge:

```
Rohpunkte (z. B. 3 000)
  → Densify  (Lücken auffüllen → bis zu 75 000+ Punkte)
  → RDP      (Punkte reduzieren — arbeitet auf dem riesigen Array)
```

Das Densify-Schritt füllte geometrische Lücken zwischen dünn abgetasteten Punkten auf. Dazu wurden viele lineare Zwischenpunkte eingefügt. RDP musste anschließend auf diesem aufgeblähten Array arbeiten, obwohl die neuen Punkte keinerlei zusätzliche Information trugen.

Jetzt:

```
Rohpunkte (z. B. 3 000)
  → RDP      (arbeitet auf dem kleinen, informativen Array)
  → Densify  (füllt nur noch die RDP-Ergebnispunkte auf)
```

RDP behält ausschließlich geometrisch signifikante Punkte. Densify läuft anschließend nur auf diesem kompakten Ergebnis.

> [!NOTE]
> Der Liang-Barsky-Clipping-Algorithmus, der für jedes einzelne Liniensegment jeder Spur, Orbit- und Vorhersagelinie aufgerufen wird, wurde zusätzlich von einer generischen Schleife zu ausgerollten skalaren Operationen umgeschrieben.

---

## Weitere Performance-Verbesserungen

<details>
<summary>FXAA Uniform-Location-Caching</summary>

GLSL-Shader-Uniform-Locations (`u_texture`, `u_resolution`) werden einmalig beim Linken des Shader-Programms abgefragt und gecacht. Bisher wurde `glGetUniformLocation()` bei jedem Frame aufgerufen.

</details>

<details>
<summary>HUD-Memoization</summary>

Das Heads-Up-Display wird als persistente GPU-Textur gehalten. Solange sich die angezeigten Textzeilen nicht ändern, findet kein erneutes Rastern und kein CPU→GPU-Upload statt. Die Textur wird nur bei tatsächlicher Inhaltsänderung oder Fenstergrößenänderung neu erzeugt.

</details>

---

## Gesamtbild

```
Physik            Gravitationsfeld des Schiffs ist jetzt zeitabhängig korrekt
Predictor         Rechenaufwand skaliert mit dem Abtastraster, nicht mit der Bogenlänge
Darstellung       Körper wechseln nahtlos zu Positions-Icons; off-screen entfällt komplett
Rendering         RDP-vor-Densify beseitigt quadratischen Aufwand bei langen Vorhersagen
Performance       GPU-Overhead (FXAA, HUD) durch Caching dauerhaft reduziert
```

### Status-Checkliste (31.05.2026)

- [x] Zeitgenaue Planetenpositionen in der Schiffsintegration
- [x] Intervall-gekoppelte Schrittweite im Predictor
- [x] Nahtloser Icon-Swap für weit entfernte Körper
- [x] Off-screen-Culling für Körper
- [x] Orbit-Spur-Culling bei sub-pixel-Ausdehnung
- [x] RDP-vor-Densify in der Predictor-Rendering-Pipeline
- [x] FXAA Uniform-Caching
- [x] HUD-Memoization
