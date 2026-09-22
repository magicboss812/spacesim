---
paths:
  - "spacesim/render/background.py"
  - "spacesim/render/background_draw.py"
  - "spacesim/render/gl/background.frag"
  - "spacesim/render/gl/background.vert"
  - "spacesim/render/gl/star.vert"
  - "spacesim/render/gl/star.frag"
  - "spacesim/tests/background_test.py"
---

# Hintergrund-Ebene — Sternenfeld und rekursives Dreiecksgitter

Die unterste Zeichenschicht. `render/background.py` rechnet (pure numpy, kein GL,
headless testbar wie `bodies/orbit_lines.py`), `render/renderer.py::_draw_background`
zeichnet, `render/gl/background.frag` und `render/gl/star.vert` rastern.

Aufgebaut nach einem Claude-Design-Mockup (`2D Spacesim Background Design`),
aber an vier Stellen bewusst davon abgewichen — jede unten begründet und
durch eine Messung gedeckt.

## Die eine Regel, die alles andere erklärt

> **Keine Größe dieser Ebene darf UNBEGRENZT mit `camera.scale` wachsen.**

`camera.scale` läuft von 1e-30 bis 1e10 px/m. Jede Größe, die man mit ihr
multipliziert und dann ungebremst zeichnet, ist bei einer Zoomstufe unsichtbar
und bei einer anderen ein Schmierstreifen. Jeder Bewegungsfehler, den diese
Ebene hatte, war eine Verletzung davon — und zwar in **beide** Richtungen:

| Fassung | Fehler | Wirkung |
|---|---|---|
| 1 | Sterndrift = `Δwelt · camera.scale` | Sterne rasen beim Hineinzoomen |
| 2 | Gitterphase = absolute Kameraposition | Gitter rast beim Hineinzoomen |
| 3 | Gitterdrift in **Pixeln** in die Phase | Zoom-Fixpunkt wandert aus der Bildmitte |
| 4 | Gitter an der **Geschwindigkeit** statt an der Position | Gitter steht still, sobald Blickziel = Bezugskörper |
| 5 | Freie Kamera: Schwenk als **Weltgeschwindigkeit** gelesen | Sterne rasen bei jedem Schwenk in die Klammer |
| 6 | Eigenbewegung aus `body.velocity` gelesen | Sterne stehen bei **jedem** Körper außer dem Schiff |

Fassung 6 gehört nicht ganz hierher — sie ist keine Zoom-Sache, sondern eine
falsche Quelle (siehe unten) —, steht aber in derselben Zeile Symptome und
wurde beim selben Durchgang gemeldet.

Fassung 4 ist die lehrreiche: sie hielt die Regel ein und war trotzdem falsch.
Ein Tempo-Anteil aus `v` kann nicht zugleich still stehen, wenn man den Bezug
anschaut, *und* laufen, wenn man einen Mond anschaut — er kennt den Unterschied
nicht. Die Position kennt ihn.

Heute gilt deshalb: **das Gitter zeigt die Position im Plot-Frame, und nur die
RATE ist gedeckelt** (`grid_max_speed_px`). Gemessen, Schwenk 800 px/s als
Maßstab:

| Zoom (px/m) | Welt (m/s) | wahr (px/s) | gezeigt (px/s) |
|---|---|---|---|
| 1e-9 | 3.0e4 | 0.0 | 0.0 |
| 1e-4 | 7.7e3 | 0.8 | 0.8 |
| 1e-2 | 7.7e3 | 77.0 | 77.0 |
| 1e+0 | 7.7e3 | 7700 | **1500** |
| 1e+2 | 7.7e3 | 770000 | **1500** |

Und der **Schwenk** liegt über elf Zehnerpotenzen exakt auf 800 px/s — das
Gitter klebt beim Schwenken an der Welt, wie es soll. `background_test.py` §4,
§8 und §8b pinnen beides.

## Sterndrift — an der Eigengeschwindigkeit, nicht am Bild

```
star_px += (v_focus / 1 km/s) * star_motion_scale * real_dt
```

`v_focus` ist die **absolute (barycentrische)** Geschwindigkeit des von der
Kamera verfolgten Körpers (`camera.target`). `star_motion_scale` ist damit in
„px pro Sekunde bei 1 km/s" abzulesen, nicht in einer nackten Konstante.

> **`body.velocity` ist für Himmelskörper IMMER (0, 0) — nimm die Position.**
> `solar_system.json` setzt `"velocity": [0, 0]` bei **allen 27** geskripteten
> Körpern (§4c prüft genau das nach), und `world.update_planets` schreibt nur
> die Kepler-*Position* zurück. Nur das integrierte Schiff trägt einen echten
> Wert. Wer das Feld liest, bekommt für Erde, Mond, Mars … exakt null — das
> Sternenfeld stand deshalb bei jedem Körper außer dem Schiff still.
>
> `BackgroundLayer._focus_speed` leitet stattdessen ab, `Δpos / Δsim_t`, mit
> zwei Bedingungen: **durch die Sim-Zeit teilen** (sonst multipliziert der
> Zeitraffer die gemessene Geschwindigkeit und die Sterne stroben bei 1 y/s —
> der Schritt selbst nimmt weiterhin `real_dt`), und beim **Körperwechsel
> nicht ableiten** (Schiff → Mars sind 1e11 m in einem Bild; als Flug gelesen
> genau die Klammer, gemessen 90.6 px). `focus_key` benennt den Körper.
> Gemessen: Erde treibt die Sterne mit 14.9 px/s, bei 1 s/Bild wie bei
> 1 d/Bild identisch.

> **Dasselbe Loch steckt noch woanders.** `Renderer._ship_relative_speed_m_s`
> zieht `reference_body.velocity` ab — also null. Die angezeigte
> „Relativgeschwindigkeit" zum Bezugskörper ist damit die **absolute**. Nicht
> im Zuge dieser Ebene angefasst; wer es angeht, tut es in `physics/world.py` an der
> Quelle (Kepler kann die Geschwindigkeit analytisch mitliefern), nicht durch
> weitere Ableitungen an den Verbrauchern.

> **Steht die Kamera frei, treibt der SCHWENK die Sterne — als
> Bildschirmbewegung gelesen, nicht als Weltgeschwindigkeit.** Das war
> Fassung 5 oben: `Δwelt/dt` sind bei 1e-9 px/m rund 1e12 m/s für einen ganz
> normalen Schwenk, also tausendfach über der Klammer. Die Sterne rasten dann
> bei *jedem* Schwenk mit voller Klammergeschwindigkeit davon, egal wie langsam
> man schwenkte — und bei 1e5 px/m gar nicht. Heute:
> `star_px += Δschirm · star_motion_scale · FREE_PAN_GAIN`, über vierzehn
> Zehnerpotenzen identisch (§4).

Drei Eigenschaften fallen aus dieser Wahl heraus, und alle drei sind gewollt:

- **Zoom bewegt die Sterne nicht** — die Kameraposition ändert sich beim
  Zoomen nicht, also auch `v` nicht.
- **Ein Bezugsrahmen-Wechsel (`R`/`1`/`2`) kann nichts anrichten**, weil das
  Modell in absoluten Größen rechnet, die der Wechsel nicht anfasst. Der
  frühere `frame_key`-Neuabgleich ist deshalb ersatzlos entfallen.
- **Zeitraffer beschleunigt sie nicht** (`real_dt`, nicht Sim-Zeit). Das ist
  eine bewusste Ruhe-Entscheidung: bei 1 y/s wäre jede physikalisch ehrliche
  Drift ein Strobe.

Die Klammer (`STAR_PAN_CLAMP_FRAC`, 6 % der Diagonale je Bild) bleibt als
reine Notbremse; das Geschwindigkeitsmodell erreicht sie im Normalbetrieb
nicht.

## Das atmende Feld

Beim Hineinzoomen soll das Feld mitzoomen, ohne beim Herauszoomen zu
verklumpen. Jeder Stern trägt dafür eine eigene, gleichverteilte **Zoomphase**
(Spalte 6 der Tabelle). In `star.vert`:

```
f = fract(star_zoom + zoomphase)        // star_zoom = log2(scale)*STAR_ZOOM_RATE
e = mix(1, exp2(f), amount)             // Kacheldehnung, 1..2
w = mix(1, fenster(f), amount)          // Ausblendfenster
kachel = viewport * e
```

Die Kachel eines Sterns ist `viewport · e`, seine Sichtbarkeitswahrschein-
lichkeit also `1/e²`. Wächst `e` auf 2, fällt es zurück auf 1 — an **beiden**
Enden ist `w` exakt null, der Sprung ist also unsichtbar. Weil die Zoomphasen
gleichverteilt sind, verschiebt `star_zoom` diese Verteilung nur; die
erwartete sichtbare Sternzahl ist damit **vom Zoom unabhängig**. Gemessen über
acht Oktaven: 0.67 % Schwankung bei `amount = 0.35`, 1.41 % bei 1.0 (§4b).

> **`amount = 0` muss exakt das starre Feld ergeben.** Deshalb wird gegen den
> Neutralwert *gemischt* (`mix(1, …, amount)`) und nicht etwa `f` auf null
> gezwungen — letzteres machte jeden Stern nach seiner festen Phase dauerhaft
> verschieden hell.

`STAR_ZOOM_RATE` (Modulkonstante, 0.5 = eine Oktave je vierfachem Zoom) regelt
die *Geschwindigkeit*, `star_zoom_influence` (Config) die *Stärke*. Getrennt,
weil sie sonst dasselbe täten und man sie nicht auseinanderhalten könnte.

## Gitteranker — ein festes Lattice im Plot-Frame

Der Anker ist die **Kameraposition im aktiven Plot-Frame**, sonst nichts. Alles,
was der Spieler sehen will, fällt daraus heraus, ohne eigens gebaut zu werden:

- Der **Bezugskörper steht still** — im körperfesten Frame liegt er im
  Ursprung, sitzt die Kamera auf ihm, ist der Anker konstant null.
- **Mond und Schiff wandern darüber**, und zwar auf ihrer echten Bahn: eine
  Kreisbahn zeichnet einen Kreis (gemessen: Achsverhältnis 1.000000, schließt
  auf 1.7e-9 m gegen 7e6 m Radius).
- Ein **Schwenk** schiebt es um genau die Schwenkstrecke (gemessen: 0.000e+00 m
  Abweichung über 30 km, und 800 px/s über elf Zehnerpotenzen).
- Ein **Rahmenwechsel** (`R`/`1`/`2`) wirkt, weil er die Transform ändert — der
  Bezugskörper muss nirgends gesondert hineingereicht werden.

`grid_anchor` wählt zwischen:

- **`"frame"` (Vorgabe)** — genau das eben Beschriebene.
- **`"focus"`** — zusätzlich die Position des verfolgten Körpers abgezogen. Das
  Gitter klebt dann am Blickziel und steht **immer** still; es misst nur noch
  den Abstand vom Ziel. Reine Maßstabsanzeige.

> **Warum nicht an der Geschwindigkeit.** Die vorige Fassung addierte einen
> Tempo-Anteil aus `v_focus − v_referenz`. Der steht auf null, sobald Blickziel
> und Bezugskörper derselbe Körper sind — also in dem Fall, den man am
> häufigsten hat: Erde anschauen, Erde als Bezug. Das Gitter stand dann bei
> *jedem* Körper außer dem Schiff völlig still. Eine Geschwindigkeit kann den
> Unterschied zwischen „ich schaue den Bezug an" und „ich schaue einen Mond an"
> nicht kennen; die Position kennt ihn.

### Die Geschwindigkeitsgrenze

Positionstreu ist ehrlich, aber bei 1e2 px/m rast ein Schiff mit 7.7 km/s um
**770 000 px/s** vorbei. Gedeckelt wird deshalb die **Bewegung je Bild**, nicht
die Position:

```
d      = ziel - ziel_vorheriges_bild        # in plot-frame-metern
budget = grid_max_speed_px * real_dt        # in bildschirmpixeln
anker += d * min(1, budget / (|d| * scale))
```

- Unter der Grenze ist die Bewegung **exakt** die wahre — `grid_lag_px` steht
  auf 0, ein Schwenk kommt Pixel für Pixel an.
- Darüber gleitet es mit genau dieser Rate in der **wahren Richtung**.

Der Rückstand bleibt dann stehen, und das ist richtig so: ein unendliches
Lattice hat keinen Ursprung, seine absolute Lage ist **unbeobachtbar** —
sichtbar ist nur die Bewegung, und die stimmt.

> **Deshalb wird die BEWEGUNG begrenzt und nicht die Position nachgeführt.**
> Eine Positions-Nachführung müsste den *Fehler* auf die nächste
> Gittertranslation falten (sonst holt sie 1e11 m nie ein) — und zöge das
> Gitter dann zur nächsten gitteräquivalenten Stelle statt in Flugrichtung.
> Bei extremem Zoom kippt die Richtung dabei mehrmals je Sekunde: das Gitter
> zappelt, statt zu gleiten. Gemessen, ehe das auffiel: −10 bis +80 px/s
> statt der verlangten 600.

`grid_max_speed_px` (Vorgabe **1500**) muss **über der Schwenkrate** liegen
(`camera.move_speed` × Bildschirmhöhe, also rund 800 px/s), sonst hängt das
Gitter beim Schwenken hinterher — genau der Eindruck, den die Grenze beheben
soll. §8b pinnt die Ordnung. `0` friert das Gitter ein.

### Ein Sprung ist kein Flug — der `grid_key`

Ein Rahmen- oder Bezugswechsel verschiebt das Ziel um bis zu 1e11 m. Mit
1500 px/s abgefahren wären das Minuten. `render/renderer.py` reicht deshalb einen
**Schlüssel** mit — Frame-Klasse, Frame-Label, Ankermodus und (bei `"focus"`)
der Name des Blickziels. Ändert er sich, wird der Sprung *übernommen* statt
abgefahren; der Anflug verdeckt den Versatz.

> **Eine Sprunghöhen-Schwelle taugt dafür nicht.** Die Bereiche überlappen: der
> Wechsel Erde→Mond misst bei 1e-4 px/m 3.8e4 px je Bild, ein Vorbeiflug am
> Zoomanschlag 8.3e4 px. Jede Schwelle dazwischen trifft einmal das Falsche.
> Der Schlüssel weiß es, ohne zu raten.

### Die Faltung trägt ein √3

Der Anker wird jedes Bild modulo `fold_spans(scale)` klein gehalten — sonst
gingen bei 1e11 m die letzten Stellen der Phase verloren. Erlaubt ist nur eine
echte **Gittertranslation** der gröbsten sichtbaren Dekade `ws = 10^k_hi`:

```
x-periode = 2·ws/√3      y-periode = 2·ws
```

Beides mal `2`, weil `p + q` gerade bleiben muss (§1) — und das `√3` in x, weil
die Knoten bei `x = q·ws/√3` stehen, nicht bei `q·ws`.

> **Modulo `10^k` zu falten ist FALSCH**, und zwar genau in x: das verschiebt
> das Muster um `√3·n` Zellen — eine irrationale Zahl, also nie ein Gitterpunkt.
> §8d misst den Versatz zu **0.79 Zellen**. Die frühere Fassung tat das (am
> Drift-Akkumulator statt am Anker) und ist damit sichtbar gesprungen.
>
> Beim **Heraus**zoomen wächst die Periode; der bereits gefaltete Wert liegt
> dann ohnehin darin, es kann also nie zurückspringen.

## Die Gittergeometrie — und der Fehler, den das Mockup hatte

Drei Linienscharen mit den Normalen `(0,1)`, `(-√3/2, ½)`, `(√3/2, ½)` und
Weltabstand `ws`. Sie sind **konkurrent**: der Schnittpunkt von Schar 1 und 2
liegt automatisch auf Schar 3 (`i₃ = i₁ − i₂`). Die Knotenmenge ist

```
x = q · ws/√3 ,   y = p · ws ,   p + q GERADE
```

Die Paritätsbedingung ist der Kern: ohne sie landet die Hälfte aller Knoten
auf **halben** Vielfachen von `ws`, also mitten in den Dreiecken statt auf den
Kreuzungen. Genau das tat das Mockup (`wy = (j+m)·ws/2`); §1 überführt die
Formel mit **0.5 Zellen** Versatz. Mit Parität stimmt auch die
Gleichseitigkeit: alle sechs Nachbarn bei `2/√3 · ws = 1.1547 · ws`.

Im Shader fällt daraus eine Vereinfachung. Mit

```
a = √3·(frag.x − w/2)/sp + phase_a      b = (frag.y − h/2)/sp + phase_b
m = (a + b)/2                            n = (a − b)/2
```

sind die drei Scharen **exakt `b`, `m` und `n`** — der Abstand zur nächsten
Linie ist jeweils `|t − round(t)| · sp`. Die Konkurrenz der Scharen ist daran
direkt ablesbar.

> **Der Anker darf niemals als Weltkoordinate in den Shader.** Er liegt bei
> bis zu 1e11 m; in float32 bliebe davon nichts übrig. Es geht nur die
> **Phase** hinein (`_phases`, float64): `fmod(√3·x/ws, 2)` und
> `fmod(y/ws, 2)`. Modulo **2**, nicht 1 — die Parität ist die Periode des
> Lattice, `a += 2` verschiebt `(m, n)` um `(1, 1)` und ist eine
> Gittertranslation.

## Keine harte Schwelle in der Dekaden-Kette

Zellweiten sind Zehnerpotenzen in Metern:

```
alpha = smoothstep(24, 90, sp) · (1 − smoothstep(600, 2600, sp))   # sp in px
node  = alpha · smoothstep(90, 200, sp)
```

Dekaden liegen um Faktor 10 auseinander, es sind also 2–3 sichtbar mit einer
klar dominanten. Eine Dekade betritt und verlässt `levels()` **genau dort, wo
ihre Deckkraft null ist**. §2 fährt zwölf Dekaden in 4000 Schritten ab: max.
Sprung 0.0096.

> **Deshalb ist das Gitter ein Fragment-Shader und keine Geometrie.** Als
> Linien gezeichnet braucht es eine Obergrenze für die Linienzahl (das Mockup
> steigt bei 900 aus) — und die lässt beim Hineinzoomen eine ganze Dekade in
> EINEM Bild verschwinden.

## Der Frühausstieg trägt den ganzen Shader

Ein Knoten liegt auf **allen drei** Scharen. Ist ein Pixel von auch nur einer
weiter entfernt als der Knotenradius, kann kein Knoten in Reichweite sein —
und die drei Abstände stehen für die Linien ohnehin schon da:

```glsl
if (max(db, max(dm, dn)) > NODE_RADIUS) continue;   // wirft ~99 % der Pixel raus
```

Ohne ihn lief die 3×3-Knotensuche für **jeden** Pixel: gemessen **9.3 ms je
Bild statt 0.35 ms**. Die 3×3-Nachbarschaft selbst ist nötig, weil das Lattice
in `(m, n)` ein 120°-Basisgitter ist, dessen nächster Punkt sich nicht durch
schlichtes Runden findet.

## Optik: an das HUD gebunden, nicht an den Entwurf

Das Mockup war warm (brauner Grund, orangeroter Akzent) und weich
kantengeglättet. Das Spieler-HUD ist das Gegenteil: sehr dunkles Marineblau,
Cyan/Magenta/Amber, und eine Anzeigeschrift, die laut `ui/theme.py`
„gerastert OHNE kantenglaettung und auf ein vielfaches von fuenf pixel
gerundet" ist. Nebeneinander las sich das als zwei verschiedene Produkte.

- **Virtuelles Pixelraster.** Der ganze Fragment-Pass rechnet auf
  `(floor(frag/px) + 0.5) * px` mit `px = ui_px(pixel_size)` — also in
  Design-Einheiten, wie jede andere UI-Größe. Alle Kanten sind `step`, nicht
  `smoothstep`: hier wäre die weiche Kante der Fremdkörper, nicht der
  Treppeneffekt. Auch die Sterne rasten Position und Größe darauf ein.
- **Fase statt Rundung** (`ui/theme.py`). Die Knoten sind darum
  Pixel-**Rauten** — Manhattan-Abstand im virtuellen Raster —, keine runden
  Punkte: eine Raute ist die 45°-Fase in ihrer kleinsten Form.
- **Zellform (`pixel_round`).** Ein reiner Pixelraster (`0`) füllt jede Zelle
  voll aus; benachbarte Zellen stoßen nahtlos aneinander. Eine dünne Diagonale
  ist dann eine Kette einzelner, schwacher Blöcke und liest sich als *weiches
  Raster*, nicht als gezeichnete Linie — genau die „unscharf gepixelte"
  Wirkung, die den ersten Entwurf verdorben hat. Bei `1` wird jede Zelle zum
  runden Punkt mit Spalt, also zu einer **Leuchtpunkt-Matrix**: die einzelne
  Marke wird als Punkt lesbar. Das ist die Vorgabe.

> **Die Zellmaske trifft nur die TINTE.** Gitterlinien, Knoten und Sterne
> werden getrennt in `ink` gesammelt und erst am Ende maskiert; Grundverlauf,
> Glut und Vignette bleiben flächig. Ein löchriger Himmel wäre etwas anderes
> als eine Leuchtpunkt-Matrix.

> **Der Füllgrad wird ausgeglichen.** Bei `pixel_round = 1` bleiben nur
> **52.8 %** der Zelle stehen (gemessen, §8c). Ohne Ausgleich hieße
> `grid_opacity` bei jeder Rundung etwas anderes und man müsste beide Regler
> gemeinsam nachziehen. Die Tinte wird deshalb durch `mix(1, 0.53, round)`
> geteilt — in `background.frag` **und** in `star.frag`. Läuft die gemessene
> Zahl weg, wandert die Helligkeit mit; §8c pinnt sie gegen die Konstante.
- **Palette aus `ui/theme.py`.** Grund um `#04070c`, Linien in der
  HUD-Kantenfarbe `#7fb4cc`, Sterne in der Textfarbe `#dceaf4`, Knoten im
  Akzent (Vorgabe HUD-Cyan `#17b2c4` = `SCHEME[0]`).

> **Der Rasterschleier ist empfindlich.** Der Grund ist fast schwarz, eine
> Aufhellung von 0.010 ist dort schon **18 % relative** Modulation und
> streift den Schirm sichtbar (gemessen an einer ruhigen Spalte: 57 gegen 47
> als RGB-Summe). Er steht deshalb auf der Hälfte. Wer `pixel_size` erhöht,
> vergrößert zugleich seine Periode und macht ihn erneut auffällig.

## Ein Schaltersatz, drei Orte

Die Attribute von `BackgroundLayer`, die Schlüssel des `background`-Abschnitts
in `config.json` und die Regler im Dev-UI-Header `Background` sind **dieselbe
Menge** (14): `enabled`, `grid_enabled`, `stars_enabled`, `accent_color`,
`grid_opacity`, `grid_anchor`, `grid_max_speed_px`, `idle_fade_delay`,
`pixel_size`, `pixel_round`, `star_density`, `star_opacity`,
`star_motion_scale`, `star_zoom_influence`. §7 prüft alle drei gegeneinander.
Laufzeitzustand (`star_pan_px`, `grid_anchor_m`, `grid_lag_px`, `grid_fade`,
`time_s`, `star_zoom`) gehört ausdrücklich nicht dazu — und §7 hat genau das schon
**zweimal** gefangen, als ein neues Zustandsfeld die Ausnahmeliste nicht
mitbekam. Wer hier ein Feld ergänzt, ergänzt auch dort.

`idle_fade_delay` zählt Sekunden ohne **Zoom**; Schwenken holt das Gitter
bewusst nicht zurück (Raten 7 /s ein gegen 1.1 /s aus).

## `background_ms` ist kein Kostenposten, sondern ein Stau

`_draw_background` ist der erste GL-Aufruf des Bildes und schluckt die
Wartezeit auf GPU/VSync des vorigen. Schaltet man die Ebene ab, wandern
dieselben ~10 ms nach `reference_trails_ms`, während `frame` sich nicht rührt.
Real kostet sie **~0.3–0.45 ms** (A/B über `frame`: 7.85 gegen 7.40 ms
Median). Optimiere gegen `frame`, nie gegen `background_ms` — dasselbe Prinzip
wie in `.claude/rules/rendering.md` unter „Frame timing — read the labels
literally".

## Fallstricke

> **`u_level_phase` ist ein `vec2`-Array — moderngl will eine Liste von
> PAAREN.** Eine flache Liste wirft `Value after * must be an iterable, not
> float`. Der Fehler war einmal drin und sah aus wie ein Shader-Bug: der
> Schreibversuch schlug still fehl, die Phasen blieben null, das Gitter klebte
> am Bildschirm statt an der Welt, und die Knoten schienen zu fehlen (einer
> lag genau unter dem Schiff in der Bildmitte). Deshalb verschluckt
> `Renderer._write_uniform` einen Fehlschlag **nicht** mehr, sondern vermerkt
> ihn einmal je Uniform in `debug_info` und druckt ihn.

> **Die Uniform-Arrays werden immer voll geschrieben** (`MAX_LEVELS` Einträge,
> hinten mit Nullen aufgefüllt). Sonst zeichnet ein Rest aus dem letzten Bild
> mit, sobald `u_level_count` wieder steigt.

> **Die Ebene liegt INNERHALB des FXAA-Passes.** Sie ist die unterste Schicht;
> sie herauszuziehen hieße alles andere mitzuziehen.

> **Tests, die absolute Tinte gegen einen dunklen Grund zählen, müssen
> `renderer.background.enabled = False` setzen.** Die Ebene färbt **jeden**
> Pixel. `tests/ship_scale_test.py` misst die Silhouette über „hell und wenig
> gesättigt" — und genau so sehen die Gitterlinien aus; ohne Abschalten meldet
> er auf jeder Zoomstufe dieselben 705 px. `body_style_test.py` schaltet sie
> aus demselben Grund ab. `selection_camera_test.py` bildet Differenzen zweier
> Bilder und ist unbetroffen.

> **Sterne sind instanzierte QUADS, keine Punkt-Sprites — und das ist kein
> Geschmack.** Der NVIDIA-Treiber dieses Rechners (4.6.0, 595.71) liefert in
> `gl_PointCoord` in **jedem** Fragment exakt `(0, 0)`; die Punkte decken ihre
> 9×9 Fragmente korrekt ab, nur die Koordinate ist tot. Damit war
> `q = |0 − 0.5|·2 = 1`, die Zellmaske verwarf jedes Fragment, und das
> **gesamte Sternenfeld war unsichtbar, sobald `pixel_round > 0`** — also in
> der Vorgabe. Der Vertex-Shader führt sein `v_uv` deshalb selbst;
> `gl_PointSize` und `PROGRAM_POINT_SIZE` fallen gleich mit weg.
>
> Zwei Puffer im VAO: das geteilte Einheitsquadrat (`'2f'`, vier Ecken als
> `TRIANGLE_STRIP`) und die Sterntabelle **je Instanz** (`'2f 4f 1f/i'` — das
> `/i` ist der ganze Unterschied). Gezeichnet mit `vertices=4,
> instances=n`. Wer eine achte Spalte anhängt, muss Layout *und*
> `build_star_table` *und* §3 anfassen.

> **Ein rein rechnender Test kann so etwas nicht finden.**
> `background_test.py` war grün, während nichts zu sehen war. Dafür gibt es
> `tests/background_gl_test.py`: er zieht Pixel und prüft, dass die
> **geschifften** Werte aus `config.json` Tinte aufs Bild setzen — bei jeder
> Rundung, nicht nur bei `pixel_round = 0`.
