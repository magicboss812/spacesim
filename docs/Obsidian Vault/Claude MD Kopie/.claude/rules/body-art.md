---
paths:
  - "spacesim/bodies/style.py"
  - "spacesim/bodies/icon.py"
  - "spacesim/ship/art.py"
  - "spacesim/render/bodies.py"
  - "spacesim/render/ship.py"
  - "spacesim/render/gl/body_icon.vert"
  - "spacesim/render/gl/body_icon.frag"
  - "spacesim/render/gl/body_surface.vert"
  - "spacesim/render/gl/body_surface.frag"
---

# Procedural vector art — planets, their marker, and the ship

- `bodies/style.py` — **the procedural vector look of the celestial bodies**
  (D2). Pure numpy, no GL: `build_planet_style(seed, color, …)` turns one
  32-bit seed into an icosphere, a noise field over it, that field quantised
  into tiers ("tiles"), the contour lines between tiers, an inner figure on
  the bright tiles and a few great-circle rings. Output is two flat float32
  tables — `tri` (n, 10) and `seg` (m, 13) — laid out so they go into a GL
  buffer without repacking. `expand_segments()` turns segments into quads.
  Ported from the Claude-Design mockup (`Planet Mockup.dc.html`), including
  its 32-bit `Math.imul` hashes bit for bit; a different hash still looks like
  a planet, just not like *that* planet.

> **It is geometry, and it must stay geometry.** The drawing lives in the
> **unit circle** and the renderer only scales it to the screen radius, so it
> is sharp at every zoom and a body costs the same vertices whatever size it
> is drawn at. Three things follow, and each is load-bearing:
>
> 1. **The bodies do not rotate**, so the random orientation is baked in once
>    and the far side can be thrown away **at build time** — half the geometry
>    gone, and every z-test moved out of the frame.
> 2. **Line width is a SCREEN quantity.** `body_line.vert` expands each
>    segment into a quad *after* scaling, offsetting by `a_half` pixels along
>    the screen-space normal. A contour therefore stays 2 px while the planet
>    grows. `tests/body_style_test.py` §5 measures exactly that: one probe
>    segment drawn at radius 80 and 320 comes out the same number of pixels
>    thick. (Measuring ink over the whole drawing instead does **not** work —
>    the tier fills dominate it and scale with the body, so the ratio comes
>    out 0.88 and proves nothing.)
> 3. **Lighting stays dynamic.** Every vertex carries its facet's 3D normal
>    and the light direction is a uniform, so the terminator travels with the
>    orbit without a single vertex being rebuilt. Per vertex there are two
>    alphas — `a_alpha` at full light and `a_dark`, what is left of it at
>    none — which is enough to reproduce all five of the mockup's layers.
>
> The direction is measured **in screen space**, from the body toward
> `_find_light_source()` (highest `light_intensity`, else the most massive
> body). That way it automatically follows a rotating plotting frame, and the
> star itself is drawn emissive.

> **The light is tilted toward the viewer on purpose, and that is a choice
> about the picture, not a mistake about the physics.** `body_light_tilt`
> (0.55) is the z-component of the light vector. At 0.0 you get the exactly
> correct phase for a top-down view of the orbital plane — and it is
> unusable: the sub-solar point then sits *on the limb* forever, the centre
> of the disc is always on the terminator, and every planet is eternally
> half-lit with its brightest region under the strongest foreshortening. It
> reads as a dark smudge. 0.55 is the mockup's own value and tips it into a
> three-quarter light: one side clearly brighter than the other, which is the
> whole point. Set it to 0 in `config.json` for the strict version.

> **The detail level is chosen so a FACET stays about the same size in
> PIXELS.** `body_vector_facet_px` (14) drives `_body_detail_levels()`, which
> picks among `coarse`/`medium`/`fine` (80 / 320 / 1280 facets) and
> **cross-fades** across the switch, in log-radius, over
> `body_vector_detail_blend`. Both halves matter and both were seen:
> pinned to `fine`, a body of 40 px radius has 3 px facets and turns into a
> golf ball; pinned to `coarse`, zooming in loses exactly the drawing the
> whole thing exists for. Without the cross-fade the pattern would jump
> mid-gesture. Measured, the facet stays between 10 and 28 px for screen radii
> from 40 to 320 px; outside that the ladder is at its end (4.6 px at 15,
> 69 px at 900) and `tests/body_style_test.py` §9 asserts it clamps **there**
> and not earlier — otherwise the ladder would secretly be one rung.

> **Built off the main thread, once per body and level.** The build is pure
> computation (no GL), so it runs in a one-worker `ThreadPoolExecutor` and
> only the upload happens on the main thread. Measured on the 28-body system:
> the build costs **2.0 / 4.0 / 13.0 ms** and holds **0.12 / 0.32 / 1.13 MB**
> for coarse / medium / fine. Synchronously, the first frame of a body coming
> into range cost **18.5 ms** — a hitch precisely while zooming, i.e. where it
> shows. Threaded, the worst frame in that window is **3.96 ms** and the
> drawing appears about 6 frames later. Steady state costs **+0.05 to
> +0.10 ms** per frame with a detailed body on screen and **0.00 ms** zoomed
> out, because below `body_vector_min_radius_px` (11) nothing is built and
> nothing is drawn — `tests/body_style_test.py` §7 asserts the frame is then
> **bit-identical** to the old flat disc, with a counter-check at 200 px so it
> cannot pass vacuously. The cache is keyed on (seed, mode, shape, colour,
> level, density) and is therefore bounded by 3 × the body count, ~42 MB in
> the worst case; it is never evicted and never needs to be.

> **Every figure must stay inside the triangle, or it leaves the planet.**
> The `dot` variant sized its circle by *half the distance to a vertex*. Near
> the limb the facets are foreshortened into slivers, and there that radius
> reaches outside the triangle and past the disc — measured 1.0045, i.e.
> 0.45 % beyond an edge where the body has already ended. It is sized by the
> **incircle** now (`2·A / perimeter`), and §2 checks the whole drawing
> against the unit circle.

- `render/gl/body_surface.{vert,frag}`, `render/gl/body_line.{vert,frag}` — the
  two programs that draw `body_style`'s output. Same uniforms, same lighting
  law; the line pair additionally expands the quad and carries a 0.5 px seam
  for antialiasing (`v_core`/`v_dist`), because at 0.6 px the grid lines
  would otherwise crawl while zooming. There is no shared VAO — each body
  owns its buffer (`Renderer._upload_body_style`), and the three draw calls
  per body are grid → fills → contours/figures/rings, in that order, because
  alphas add.

## Die Marke — `bodies/icon.py`

Sinkt ein Körper unter `renderer.body_icon_min_radius_px`, wird er de-gerendert und
gegen eine **Positions-Marke** getauscht (`render/renderer.py::_draw_body_icon`). Das
war eine flache Scheibe in Körperfarbe — bei Systemzoom also 27 gleiche Punkte.
Seit 2026-08-30 trägt sie ein gesätes Zellmuster in zwei Varianten:
`rosette` (voller Kern, gewürfelter Außenring) und `signature` (Kern plus 2–4
Zacken auf dem Ring). Beide aus **einem** 32-bit-Seed, `body_icon_variant`
wählt.

> **`body_icon_seed_offset`** (Vorgabe `0`) verändert diesen Seed für **jeden**
> Körper auf einmal — die ganze Serie neu würfeln, ohne `style_seed` in
> `solar_system.json` anzufassen. `body_icon.seed_for(body, offset)` mischt den
> Versatz durch dieselbe `_Rng`-Formel wie ein Körper-Seed selbst
> (`_imul(offset, _SEED_MIX) + _SEED_ADD`, dann XOR) — eine einfache Addition
> hätte benachbarte Versätze zu fast identischen Mustern gemacht, weil `_Rng`
> seinen ersten Wurf stark vom niedrigsten Bit des Seeds abhängen lässt.
> Gemessen unterscheiden sich Versatz 0 und 1 in 19 von 32 Bit. Im Dev-UI
> (`icon seed-versatz`) verstellt ein Klick auf **würfeln** ihn um eins und
> leert `_body_icon_cache`, damit der neue Wert sofort zu sehen ist.

> **Warum das überhaupt nötig ist.** In `solar_system.json` tragen **Ganymed und
> Oberon dieselbe Farbe** (`#9c8f7c`), und fünf Monde liegen in praktisch
> demselben Grau (`#c0c0c0`, `#d3d3d3`, `#c9c9c9`, `#c8c8c8`, `#b0b0b0`). Über
> die Farbe sind sie als Marke nicht zu trennen; über das Muster schon.
> `tests/body_icon_test.py` §1 und §4 prüfen genau dieses Paar.

> **Das Zellmuster liegt im ICON-Raum, nicht im Bildschirmraum — daran hängt
> die Bewegung.** Der Hintergrund rastet bewusst auf den Schirm
> (`background.frag`: `frag = (floor(gl_FragCoord.xy/px)+0.5)*px`, jede Kante
> `step`); sein Muster rückt deshalb pixelweise. Für eine Marke, die einem
> Körper folgt, wäre das genau die stockende Bewegung, die nicht gewollt ist.
> Der Fragment-Shader rechnet die Zellen stattdessen aus `v_local`, der
> interpolierten Icon-Koordinate, und die hängt an der **Gleitkomma**-Position
> der Marke. Das Muster kann daher nicht über die Marke wandern — es *ist* die
> Marke.

> **Eine Marke ist EIN Quad, nicht ein Quad je Zelle.** Der Canvas-Entwurf, an
> dem die Optik abgenommen wurde, zeichnete jede Zelle als eigenes `fillRect`.
> Beide Seiten einer gemeinsamen Kante schrieben dann ihre eigene
> Kantenglättung in dieselbe Pixelspalte, und die Naht wanderte beim Bewegen
> sichtbar mit — es sah nach Flimmern aus und wurde fälschlich der Icon-Raum-
> Rasterung angelastet. Mit einem Quad gehört ein Fragment nicht einem
> Primitiv, sondern einem Flächenstück: es gibt keine aneinanderstoßenden
> Primitive und damit keine Naht.

> **Diese Naht trug aber zugleich das Raster.** Das war zuerst
> `body_icon_cell_gap` — und der Spalt war die falsche Antwort auf die
> richtige Beobachtung; die eigentliche steht im nächsten Absatz. Er ist
> als Regler geblieben, steht aber auf **0**. Die Naht senkte die gemeinsame Pixelspalte zweier
> voll deckender Zellen auf `1 - a/4`, also **75 %**. Genau diese dunklen
> Linien ließen die Marke im Entwurf als Gitter aus Pixeln lesen. Die erste
> gebaute Fassung beseitigte sie sauber — und lieferte damit einen **flachen
> Klecks**: gemessen hatte die Kernzeile der Rosette *null* Einschnitte, drei
> gleich helle Zellen nebeneinander sind eine Fläche. Im Spiel fiel genau das
> auf, während alle Tests grün waren.
>
> Der Spalt zieht jede Zelle um `1 - u_cell_gap` ein, dieselbe Rolle wie
> `1 - 0.18*pixel_round` beim Hintergrund-Gitter. Gemessen (tiefste Trennlinie,
> relativ zur Zelle, bei Halo 0.30): `0.00 → 100 %` (keine Linie),
> `0.12 → 74 %` (die Naht des Entwurfs), **`0.22 → 58 %`**, `0.32 → 50 %`.
> Der Halo liegt hinter den Zellen und füllt die Spalten mit: 44 % ohne Halo,
> 58 % bei 0.30, 67 % bei 0.53.

> **Nacheinander überblenden, nicht mitteln — daher kommt die Tiefe.** Der
> Canvas-Entwurf zeichnete jede Zelle als eigenes `fillRect` mit Deckkraft
> **über** die schon gezeichneten. Der Shader mittelte statt dessen die
> Flächenanteile der überdeckten Zellen — der mathematisch saubere Box-Filter,
> und genau deshalb sah er *flach* aus: eine Mittelung zieht alles zum
> Mittelwert, so dass Zellgrenzen verschwinden und zwischen zwei Stufen ein
> glatter Übergang steht statt vieler Zwischentöne.
>
> Die Schleife in `body_icon.frag` benutzt jetzt den **over-Operator** in
> Zeichenreihenfolge (`acc = acc·(1−ca) + col·ca`). Damit dunkeln Zellgrenzen
> nach — das sind die „Umrisse" der Zellen — und es entstehen viele
> Zwischenstufen. Der Spalt wird dadurch überflüssig.

> **Der Umriss jeder Zelle wird GEZEICHNET, und seine Breite ist ein
> Bildschirmmaß.** Erst entstand er als Nebenwirkung des over-Operators, also
> aus der Reihenfolge der Shader-Schleife — willkürlich, und vom Halo wieder
> aufgefüllt: gemessen lagen bei Radius 16 nur 2 von 28 senkrechten und 0 von
> 26 waagerechten Zellgrenzen überhaupt unter 90 % Helligkeit. Jede Zelle
> dunkelt jetzt zu ihrem eigenen Rand hin nach (`max(|qx|,|qy|)`, also ein
> Quadrat — in beiden Achsen gleich). Das ist zugleich die *gemalte*
> Kantenglättung: der dunkle Saum liest sich als Schatten der Zelle.
>
> **`body_icon_cell_rim` ist eine Breite in PIXELN, kein Anteil der Zelle** —
> dieselbe Regel wie für die Linienbreiten in `body_line.vert`. Als Anteil
> (0.34) war der Saum bei Radius 16 nur **0.55 px** breit, und ob er überhaupt
> abgetastet wurde, hing an der Bruchteil-Position der Marke. Weil x und y
> verschiedene Phasen haben, zeigte dann *eine* Achse ihre Umrisse und die
> andere nicht — genau das war im Spiel zu sehen. Ab rund einem Pixel Breite
> kommen beide an; `tests/body_icon_test.py` §6 zählt die dunklen Linien in
> Spalten- und Zeilenmittel und verlangt, dass keine Achse ausfällt.

> **Falle beim Nachmessen: `fbo.read()` liefert die Zeilen von UNTEN nach
> oben.** Ohne ein `[::-1]` tastet jede y-Messung die falschen Zeilen ab. Eine
> Sonde ohne den Flip „bewies" hier eine Achsen-Asymmetrie, die es gar nicht
> gab — die Zeilen- und Spaltenmittel des rohen Bildes zeigten das Gitter in
> beiden Achsen. Wer eine Achse gegen die andere misst, dreht das Bild zuerst
> um.

> **Jede Zelle hat ihre EIGENE Helligkeit (`body_icon_shade_jitter`, 0.30).**
> Drei Stufen sind zu wenig Tiefe: gleich eingestufte Nachbarn verschmelzen
> sonst zu einer Fläche, und die Marke sieht gedruckt aus statt texturiert.
> `cell_shade()` hasht `(Zelle, Seed)` zu einem Faktor und legt ihn auf Farbe
> **und** Deckkraft — nur die Farbe zu variieren gibt bunte Flecken, nur die
> Deckkraft einen Schleier. Der Wert hängt an nichts als Zelle und Seed, kann
> also nicht flimmern, wenn die Marke sich bewegt. Gemessen: **28**
> unterscheidbare Helligkeiten gegen **24** ohne Streuung
> (`tests/body_icon_test.py` §6).

> **Der Radius trägt die Zellen mit.** Bei 9×9 und Radius 8 ist eine Zelle
> 1.6 px breit — genug für ein Muster, zu wenig für Zellumrisse. Bei Radius 16
> sind es 3.2 px, und erst dann liest sich jede Zelle als eigenes Feld mit
> Rand. Radius und Raster gehören deshalb zusammen betrachtet: das Raster
> bestimmt, WIE VIELE Zellen es gibt, der Radius, ob man sie einzeln sieht.

> **Gemittelt wird über das PIXEL-QUADRAT, nicht nur über den Umriss.** Eine
> erste Fassung glättete nur freie Außenkanten und ließ die Grenzen zwischen
> zwei Stufen hart. Gemessen schwankte die Helligkeit über einen Pixel Drift um
> **±5.4 %** — die inneren Grenzen kippen beim Wandern Pixel für Pixel zwischen
> zwei Stufenfarben, und bei Deckkräften von 0.55 bis 1.0 fällt das ins
> Gewicht. Der exakte Box-Filter über die 3×3-Nachbarschaft bringt dieselbe
> Messung auf **±0.10 %**, bei ±7.46 % in der Gegenprobe mit harter Kante.
> `tests/body_icon_test.py` §5 pinnt beides.
>
> `body_icon_edge_px` ist die Filterbreite. Gemessen (Ruhe über einen Pixel
> Drift / Anteil Pixel auf einem Stufen-Plateau):
> `0.0 → 7.44 % / 42.4 %`, `0.6 → 1.63 % / 35.8 %`, **`1.0 → 0.11 % / 29.7 %`**,
> `1.4 → 0.73 % / 27.4 %`. **1.0 ist nicht bloß ein runder Wert, sondern das
> Optimum**: genau ein Bildschirmpixel ist die richtige Breite für einen
> Box-Filter, und breiter wird es wieder unruhiger. 56 Pixel liegen dabei noch
> auf voller Stufenhelligkeit, die Zellen bleiben also innen satt.
>
> Der Filter begrenzt zugleich, wie fein ein Zellspalt noch ankommt: schmaler
> als etwa eine halbe Pixelbreite wird er weggemittelt. Beide Regler hängen
> also zusammen — wer `edge_px` hochzieht, braucht mehr `cell_gap`.

> **8 px Radius, nicht 4.** `body_icon_min_radius_px` ist die Mindestgröße —
> unterhalb bleibt die Marke immer bei diesem Wert, egal was
> `body_icon_size_influence` sagt — und zugleich der **Greifradius**
> (`_pick_radius_px`, über `_body_icon_draw_radius_px`) — das Klickziel wächst
> also mit, was kleine Monde erst brauchbar anklickbar macht.

> **Die Marken-Größe skaliert optional mit dem PHYSISCHEN Körper-Radius —
> nicht mit dem Bildschirmradius, und das war ein echter Fehlgriff.** Die
> erste Fassung nahm `true_radius_px` (`body.radius * camera.scale`) und
> mischte `scaled = min + (true - min) * einfluss`, geklemmt auf `[min,
> max]`. Das hatte im Spiel **keine sichtbare Wirkung, bei jedem
> Einfluss-Wert**: eine Marke existiert per Definition nur, solange der
> Körper kleiner als `min` ist, also ist `true - min` in genau der Situation,
> in der man die Marke sieht, fast immer negativ — `scaled` blieb unter
> `min` und wurde sofort wieder daraufhin geklemmt. Ein Test mit absichtlich
> großen, über `min` liegenden Werten bestand trotzdem, weil er die
> Situation prüfte, in der es *fast* funktioniert (das schmale
> Überblend-Band), nicht die, in der man das Feature tatsächlich benutzt
> (Systemzoom, jeder Körper eine winzige Marke).
>
> `_body_icon_draw_radius_px(body_radius_m)` hängt jetzt am **physischen**
> Radius des Körpers (Meter, `body.radius`) — der ändert sich nie mit dem
> Zoom. `_update_icon_radius_range(bodies)` bestimmt einmal je Frame die
> Spanne aller geladenen Körper-Radien (das Schiff zählt nicht mit, sein
> `radius` ist ein technischer Platzhalter), `_body_icon_size_factor`
> ordnet einen Radius **log-skaliert** (nicht linear — die Radien liegen
> über 3,5 Dekaden, von Mimas' 2·10⁵ m bis zur Sonne 7·10⁸ m; linear würde
> alles außer der Sonne auf denselben Punkt drücken) in `[0, 1]` innerhalb
> dieser Spanne ein, und `body_icon_size_influence` mischt zwischen „immer
> `min`" (`= 0`) und „voll nach dem log-Faktor, bis `max`" (`= 1`). Ein
> Jupiter-großer Körper ist damit **bei jedem Zoom** sichtbar größer als ein
> kleiner Mond, nicht nur kurz während der Überblendung. `_pick_radius_px`
> ruft dieselbe Funktion mit demselben Argument auf, damit das Klickziel
> eines weit entfernten Riesenkörpers nicht auf einen verschwindenden
> Bildschirmradius zusammenschrumpft. `tests/body_icon_test.py` §7 pinnt
> alle Randfälle (0, 1, 0.5, außerhalb der Spanne) gegen eine von Hand
> gesetzte Radius-Spanne und verlangt ausdrücklich, dass ein „Mond" und ein
> „Planet" bei `influence = 1` unterschiedlich groß bleiben — unabhängig vom
> (in diesem Test bewusst winzigen) Bildschirmradius.

> **Das RASTER ist der Detailgrad, nicht die Bildschirmgröße.**
> `body_icon_grid` (Vorgabe **9**, Höchstwert `MAX_GRID = 15`) bestimmt allein,
> wie viele Zellen eine Marke hat. Eine größere Marke zeigt dasselbe Muster
> größer, nie ein feineres — das ist die Zusage, und §3b prüft sie.
> `MAX_GRID` ist 15 und nicht 16, weil die Entwürfe **radial um eine
> Mittelzelle** gebaut sind; ein gerades Raster hat keine. Ein gerader Wert
> wird deshalb auf den nächstkleineren ungeraden gezogen.
>
> Bei 5×5 gibt es zu wenige Zellen für Textur — die Marke ist ein Klecks mit
> Rand. Bei 9×9 sind es 41 Zellen, und das ist der Charakter des abgenommenen
> Entwurfs.

> **Es gibt eine harte Auflösungsgrenze, und sie ist kein Filterproblem.**
> Unter rund **1.5 px je Zelle** kann der Schirm das Raster nicht mehr
> auflösen; sechzehn Pixel tragen keine fünfzehn abwechselnden Zellen. Gemessen
> an der Zahl sichtbarer Helligkeitswechsel quer durch die Marke:
>
> | | 5×5 | 9×9 | 13×13 | 15×15 |
> |---|---|---|---|---|
> | Radius 8 px | 80 | **86** | 84 | 75 |
> | Radius 24 px | 108 | 139 | — | **265** |
>
> Bei Radius 8 sättigt es also um 9 herum und fällt danach wieder — bei Radius
> 24 trägt jedes feinere Raster auch mehr Muster. Wer 15×15 wirklich sehen
> will, braucht eine größere Marke; ein feinerer Filter hilft nicht. Beide
> Zeilen stehen in `tests/body_icon_test.py` §6.
>
> Der Box-Filter ist deshalb auf eine **halbe Zelle** gedeckelt
> (`body_icon.frag`). Ohne den Deckel mittelte ein feineres Raster sich selbst
> weg, und mehr Raster hieße *weniger* Muster — genau verkehrt herum. Der Preis
> steht in §5: die Laufruhe geht von ±0.10 % auf ±1.00 %, gegen ±13.4 % in der
> Gegenprobe mit harter Kante.

> **Der Tausch wird überblendet.** Früher galt Icon-Radius == Umschaltschwelle,
> und weil beide Seiten dieselbe flache Scheibe zeichneten, war der Tausch exakt
> nahtlos. Eine Pixelmarke sieht anders aus als eine schattierte Scheibe mit
> Limbus — bei gleichem Radius poppt es trotzdem. Zwischen
> `body_icon_min_radius_px` und `body_icon_min_radius_px * body_icon_fade_factor`
> wird der Körper deshalb ganz normal gezeichnet und die Marke **darüber**
> ausgeblendet (`_body_icon_fade`). Das kostet den Körper-Zeichenweg keine Zeile.
>
> **Das Bandende ist ein FAKTOR, kein absoluter Pixelwert — das war zweimal ein
> Bug.** Mit einem festen `body_icon_fade_radius_px` verlor die Grenze zweimal
> den Anschluss ans Minimum: einmal stand sie bei `min = 32` unter dem alten
> `fade = 13` verkehrt herum (das Band hatte negative Breite), einmal musste
> sie bei `min = 16` von Hand auf `25.6` nachgerechnet werden. `body_icon_fade_factor`
> (Vorgabe `1.6`) bemisst sich am jeweils aktuellen `min` und kann diese Klasse
> von Fehler nicht mehr reproduzieren.

> **Der Label-Modus `"zoom"` misst den ECHTEN Radius, nicht die Marke.**
> `_queue_body_label` nimmt dafür ein eigenes `size_radius_px`. Beides zu
> vermengen war lange folgenlos, weil die Marke mit 4 px unter
> `body_label_min_radius_px` (5) lag — mit 8 px lag sie darüber, und plötzlich
> trug im Zoom-Modus jeder winzige Mond seinen Namen. Der Anker hängt an der
> *Zeichnung*, die Entscheidung am *Körper*.

> **Gewürfelt wird je Zelle, radial gewichtet — kein Rauschfeld, kein fester
> Kern.** Drei Fassungen, zwei davon falsch:
>
> 1. *Fester Kern von fünf Zellen* (`d² ≤ 1`) — bei 5×5 richtig, bei 15×15 ein
>    Punkt in der Mitte; die Marke zerfiel in Spitze.
> 2. *Mitwachsender Kern* (`radius · 0.42`) — eine glatte helle Scheibe, die
>    genau die Textur auffrisst, um die es geht.
> 3. *FBM-Rauschfeld* statt Einzelwürfen, um zusammenhängende Flecken statt
>    Körnung zu bekommen. Klingt richtig und war es nicht: zusammen mit dem
>    radialen Abfall, den die Marke braucht, damit sie eine Scheibe bleibt,
>    wurde aus **jedem** Körper dieselbe Scheibe mit hellem Kern. Die
>    Eigenschaft, um die es hier überhaupt geht — dass man die Körper
>    auseinanderhält —, ging dabei verloren.
>
> Es bleibt beim Einzelwurf je Zelle, aber die **Wahrscheinlichkeiten hängen am
> Radius**: `p_leer = 0.02 + 0.58·t²`, `p_hell = 0.74 − 0.66·t` mit `t` als
> Abstand vom Zentrum. Das gibt einen dichten hellen Kern mit unregelmäßigem
> Rand und einen ausfransenden Saum, und weil es Anteile sind und keine
> Zellzahlen, bleibt der Charakter bei jedem Raster erhalten. `_sym()` hält die
> Punktsymmetrie und damit den Schwerpunkt.

Das Muster steckt als **2 bit je Zelle in sechzehn uint32**
(`uniform uint u_cells[16]`), also 256 Plätze für die 225 Zellen von 15×15.
Dynamische Indizierung eines uniform-*Arrays* ist in GLSL 330 erlaubt; die
frühere `uvec4` musste ausgeschrieben werden und trug nur 64 Zellen. Alle
Marken teilen sich das statische Einheits-Quad aus `_ensure_quad_vbo`; es gibt
keinen Puffer je Körper und nichts, was pro Frame belegt würde. Pro Körper sind
es sechs Uniform-Schreibvorgänge, der Rest läuft über `_set_uniform`.

- `ship/art.py` — the player ship's vector artwork, transcribed from the
  Claude Design mockup `Ship Mockup.dc.html`
  (project `7c1401de-e967-41a5-818a-742e9a6015b3`). The shape lists are kept
  in the mockup's own **SVG coordinates** (y down, viewBox `0 0 300 170`) so
  they can be diffed against it line by line; `build(accent)` flips them into
  local ship space (+x = nose, +y up, origin at the visual centre), flattens
  the beziers/arcs/dashes, ear-clips every fill into triangles and merges the
  result into ~37 `(mode, rgba, width, start, count)` batches over one shared
  vertex array. Pure numbers — no pygame, no GL, no numba. **Order is the
  mockup's paint order and must not be sorted by colour.** Drawn by
  `Renderer._draw_ship_sprite`; its on-screen size is set by
  `_ship_length_px()` — see the ship-size note in `.claude/rules/rendering.md`.
