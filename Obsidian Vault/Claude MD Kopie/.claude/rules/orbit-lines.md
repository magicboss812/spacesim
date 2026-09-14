---
paths:
  - "spacesim/bodies/orbit_lines.py"
  - "spacesim/render/orbits.py"
---

# Bahnlinien der Körper (`bodies/orbit_lines.py`, Phase C)

- `bodies/orbit_lines.py` — **die bahnlinien der körper** (Phase C). Pure numpy, kein
  GL, headless testbar wie `bodies/style.py`. Eine kurve je körper:
  `future_tracks()` löst die weltpositionen **aller** körper (und ihrer
  eltern) über das **vorhersage-fenster** auf EINEM zeitgitter,
  `future_track()` ist nur noch der dünne mantel für einen einzelnen davon —
  ein rechenweg, damit einzel- und stapelaufruf nicht auseinanderlaufen
  können. `FrameAffineTable` projiziert sie in den plot-frame, `closest_approach()`
  misst die annäherung, `approach_alpha()` und `reveal_fraction()` machen
  daraus helligkeit und länge. `OrbitLineSet` hält das über die frames.
  Gezeichnet in `Renderer._draw_orbit_lines`, vor den körpern in den
  FXAA-puffer. Toggle `O` / `renderer.orbit_lines_enabled`.

> **Die linie ist eine ZEITKURVE im plot-frame, keine ellipse.** Das war der
> erste entwurf und er war schlicht falsch: eine feste kepler-ellipse, starr
> zur aktuellen frame-zeit transformiert. Ein plot-frame ist aber eine
> **zeitabhängige** abbildung — jeder punkt gehört in den frame, den seine
> EIGENE zeit aufspannt. Im Erd-rahmen zeichnete die alte fassung der Erde
> eine bahn um die Sonne, obwohl die Erde dort per definition im ursprung
> steht; im Mond-rahmen legte sie dem Mond eine schleife um sich selbst.
>
> `frame_origin_body()` gibt es deshalb: der ursprungskörper bekommt **gar
> keine linie**. Er bewegt sich in seinem eigenen rahmen nicht, eine linie
> für ihn zeigt nur noch den unterschied zweier kepler-modelle (siehe unten).
> Gemessen im Mond-rahmen über 3.8 tage: die Erde ist zeitrichtig eine kurze
> kurve auf konstantem Mond-abstand, starr transformiert dagegen **40x
> länger** und einmal um die Sonne herum.

> **Das fenster ist das des PRÄDIKTORS, und die endkappen sind der messwert.**
> Die schiffslinie endet beim schiff zur zeit `t_end`; die linie eines körpers
> endet bei diesem körper zur **selben absoluten zeit**. Fallen die beiden
> endkappen zusammen, ist das schiff dort, wo der körper dann steht. Das ist
> der ganze zweck der funktion, und es funktioniert nur, weil beide linien
> durch dieselbe transformation gehen.
>
> **Die körper-endkappe ist ein kreis mit dem ECHTEN radius**
> (`_draw_body_disc_outline`, `body.radius * camera.scale`) — kein fester
> pixelwert mehr, keine raute. Sie schrumpft mit heraus-zoomen unter ein pixel
> und verschwindet dann, genau wie die körperscheibe selbst; darunter (< 0.75 px)
> wird sie gar nicht gezeichnet. Die **schiffskappe** bleibt die kleine weisse
> raute (`_draw_end_cap`, `orbit_line_end_cap_px`) — das schiff hat `radius = 0`,
> und raute-in-kreis liest sich sauber als treffer.
>
> Zwei dinge folgen daraus und sind nicht offensichtlich. **Die kappen treffen
> sich nur bei einer begegnung GENAU am horizont-ende** — passiert die
> annäherung mitten im fenster, stehen die kappen auseinander, obwohl es eine
> begegnung gibt (gemessen 411 px kappenabstand bei 186 px dichtester
> annäherung). `+`/`-` verschiebt `t_end` und ist damit das einstellrad. Und
> **ein gemeinsames fenster für alle körper ist zugleich das, was die sache
> bezahlbar macht** — eine einzige knotentabelle für alle statt 26.

> **Zwei bänder auf EINER messung: helligkeit und LÄNGE.**
> `approach_alpha()` (1..3 SOI) setzt die deckkraft, `reveal_fraction()`
> (10..30 SOI) setzt, wie viel der linie überhaupt gezeichnet wird — sie
> rollt sich vom körper aus ab. Das enthüllungs-band ist bewusst **weiter**
> als das helligkeits-band: die endkappe ist das, womit gezielt wird, und
> sie muss schon da sein, während man noch steuert. Näherkommen ändert dann
> nur noch die helligkeit. Ein körper ohne annäherung ist gar kein kandidat
> — es gibt keinen boden und keine dauerhafte „karte" mehr; typisch stehen
> 0–3 linien auf dem schirm, und jede davon ist etwas, das man treffen kann.
> Referenz- und auswahlkörper sind immer kandidaten.

> **Der frame wird je ZEITFENSTER ausgewertet, nicht je punkt — und schon gar
> nicht je körper.** `FrameAffineTable` bestimmt die starre transformation auf
> einem knotengitter und interpoliert kubisch (`_cubic_4pt`, wortgleich zu
> `reference_frames`). Gespeichert wird der **winkel**, nicht die matrix:
> interpoliert man `cos` und `sin` getrennt, verlieren die spalten ihre länge
> und das ergebnis ist keine drehung mehr.
>
> **Und die kepler-lösung läuft für das ganze system auf einmal.** Die
> Newton-iteration konvergiert nach drei bis vier schritten — der aufwand
> liegt also nicht in der mathematik, sondern in numpys aufruf-overhead auf
> 192 kurzen arrays: gemessen **172 us je körper einzeln gegen 34 us im
> stapel**. Alle körper teilen sich ohnehin EIN zeitgitter (das des
> prädiktors), also wird EINE `(k, t)`-iteration für alle zusammen gelöst,
> und die elternkette fällt beim aufsummieren gleich mit ab. Die mittlere
> anomalie wird dabei auf `[-pi, pi)` gefaltet — ohne das läuft Newton bei
> grossem `dt` auf einer weit entfernten wurzel an.

> Die knotenzahl kommt aus dem, was sich im fenster wirklich dreht, und **es
> sind zwei winkel, nicht einer**: der rahmen selbst (richtungs-frame) *und*
> sein ursprung. Ein körperzentrierter, nicht rotierender rahmen hat
> drehwinkel 0 und trotzdem eine gekrümmte verschiebung — nur den ersten zu
> messen gäbe dort 8 knoten für einen halben umlauf. Gemessen bei 0.05 rad je
> knoten: 1x horizont 8 knoten, 64x horizont 84. Der fehler gegen die
> punktweise schleife bleibt unter **1e-6 der szene** (77 km auf 1e11 m,
> 0.04 px) — eine größenordnung besser als die knoten-interpolation des
> rahmens selbst, die laut der notiz bei `physics/reference_frames.py` mit 0.54 px
> angesetzt ist.
>
> **Nicht** den weg über `frame.set_origin_interp_window()` nehmen, so
> verlockend er ist: `_origin_interp_q` wird auch vom **skalaren** pfad
> gelesen (`_body_world_position_at_time`), das fenster zu setzen verschiebt
> also still jeden körper, den `_draw_body` in diesem bild zeichnet.

> **ES GIBT NUR NOCH EIN KÖRPERMODELL, UND DAS IST KEPLER (2026-08-27).**
> Bis dahin waren es drei, und die notiz hier hat zweimal die falsche
> zwillingsfunktion benannt, bevor klar war, dass die aufteilung selbst der
> fehler ist:
>
> - **modell A** — `bodies.position_at_time` fror die winkelrate auf der
>   epoche ein (`theta_t = theta_ref + omega_ref·dt`, `omega_ref = v/r` aus
>   vis-viva). Das war die schwerkraft des **welt**-integrators,
>   `world_kernels._body_pos_at_time` baute die näherung nach.
> - **modell B** — `reference_frames._body_world_position_exact` und
>   `predictor._body_scripted_relative_xy_numba` lösen kepler richtig
>   (mittlere anomalie + Newton).
> - **modell C** — `world.update_planets` integrierte `dθ/dt = v/r` als
>   **Euler-schritt**, dessen schrittweite die chunk-grösse war.
>
> Bei `t = now` stimmten alle überein (gemessen exakt 0.0 m), danach liefen
> sie auseinander — 3.8e8 m beim Mond nach 30 tagen. **Modell A und C sind
> weg**: `bodies.kepler_relative_xy` ist jetzt die eine quelle, und
> `orbit_position`, `position_at_time` und `world_kernels._body_pos_at_time`
> gehen alle drei durch sie. Welt, prädiktor, frame-ursprung und bahnlinien
> rechnen damit dieselben körper.
>
> **Warum das nicht bloss aufräumen war.** `omega = v/r` ist nur an den
> apsiden die wahre winkelrate — dazwischen steckt in `v` die
> radialkomponente, die sich gar nicht dreht, der körper lief also
> systematisch zu schnell (~e²/4 je umlauf). Und die position wurde aus dem
> radius zum ALTEN winkel mit dem NEUEN winkel gebildet. Vor allem aber war
> modell C ein verfahren **erster ordnung mit der chunk-grösse als
> schrittweite**, und die hängt an der raffungsstufe: `step_simulation`
> zerlegt in `max(max_substep_seconds, warp-decke)`, also 1000 s in echtzeit
> und 4375 s bei 1 y/s. Gemessen wanderte der Mond über 20 tage dadurch um
> 7.5e4 m (chunk 1000) bis 1.3e6 m (chunk 16000), und auf einer
> mondtransferbahn kam über 25 tage ein perigäum von **9.05e6 m (chunk 4375)
> gegen 6.76e6 m (chunk 30)** heraus. **Der zeitraffer rechnete eine andere
> welt als die echtzeit.** Eine gegenprobe mit verfeinerter integrator-decke
> (3000 → 30 s) bewegte dieselbe zahl um 0 m — der fehler sass also nicht im
> schiff, sondern in den körpern.
>
> Exakt gelöst ist die fortschreibung schrittweiten-**unabhängig**, weil die
> zusammensetzung exakter schritte wieder der exakte schritt ist: gemessen
> steht der Mond nach 20 tagen bei jeder chunk-grösse von 100 bis 16000 s auf
> **6.5e-4 m** genau an derselben stelle (vorher 1.33e6 m).
> `tests/warp_predictor_test.py` §22 misst genau das.
>
> **Für die bahnlinien ändert sich dabei nichts** — sie benutzten schon
> modell B, aus dem folgenden grund, und der gilt unverändert: gezeichnet
> wird `spur(t) − ursprung(t)`, und der ursprung kommt aus
> `reference_frames`. Der fehler steckte ausschliesslich in der
> **elternkette**: die eigene ellipse ist unter beiden modellen dieselbe
> kurve — die konstante rate verschiebt nur die phase darauf —, aber der
> *elter* wanderte unter modell A vom rahmen-ursprung weg und zog das kind
> mit. Deshalb war die Erde im Sonnen-rahmen unauffällig (die Sonne steht
> fest, es gibt keine kette) und der Mond im Erd-rahmen nicht: gemessen bei
> 90 tagen horizont lief sein Erd-abstand von 5.9e7 bis 3.1e9 m, statt in
> seiner bahnschale 3.633e8..4.055e8 m zu bleiben — **der Mond flog
> scheinbar davon.** `tests/orbit_lines_test.py` §2 baut modell A dafür
> eigens nach (`_constant_rate_position_at_time`), weil
> `position_at_time` als gegenprobe nicht mehr taugt — es deckt sich jetzt
> auf 1 mm mit dem rahmen-modell.

> **Die spur braucht einen WINKEL-boden, und das fenster ist nicht ihres.**
> `track_samples` (192) stichproben liegen gleichmässig auf dem fenster des
> **prädiktors**, und die umlaufzeit eines mondes weiss davon nichts. Gemessen
> über ein fenster von 30 jahren (transfer Erde → Neptun) bekommt Triton
> **3527° je stichprobe** — fast zehn umläufe zwischen zwei benachbarten
> punkten. Gezeichnet wird daraus ein sternpolygon aus sehnen quer durch die
> bahn; die linie ist dort nicht grob, sondern eine andere kurve. Io kommt auf
> 11 717°, Mimas auf 21 969°.
>
> `OrbitLineSet._build_draw_track` legt deshalb ein **zeichen-gitter** neben
> das mess-gitter: `samples_per_period` (64, also 5.6° je stichprobe —
> pfeilhöhe 0.0012 R, bei 230 px bahnradius 0.28 px und damit gerade
> `orbit_line_tolerance_px`) und `max_track_samples` (1024). Gemessen über
> alle 26 kandidaten und drei fensterlängen: **worst case 3527° → 5.63°**.
>
> **DIE MASSGEBENDE PERIODE IST DIE IM PLOT-FRAME, NICHT DIE UM DEN EIGENEN
> ELTER.** Gezeichnet wird `körper(t) − ursprung(t)`, und diese differenz
> trägt die frequenzen **beider** elternketten — bis auf die glieder, die
> beide gemeinsam haben, denn die heben sich exakt weg. Im Titania-rahmen
> läuft *Uranus* einmal je 8.71 tagen um den bildmittelpunkt (es ist Titanias
> bahn mit umgekehrtem vorzeichen), nicht einmal je 84 jahren. Nach seiner
> eigenen periode bemessen sah das fenster harmlos aus, die spur aliaste
> trotzdem — gemessen **70.2° je stichprobe** für Uranus im Titania-rahmen,
> 90.4° für Neptun im Triton-rahmen, 120.1° für die Erde im Mond-rahmen.
> `relative_min_period(körper, ursprung)` ist das minimum über die
> nicht-geteilten glieder beider ketten; damit fallen dieselben fälle auf
> 5.64° / 5.63° / 6.29°. Die reste über 5.63° sind die exzentrizität: der
> boden setzt gleiche ZEIT-schritte, und die winkelrate ist im periapsis
> höher (Merkur, e = 0.21, kommt auf 8.7° — pfeilhöhe 0.67 px, weit weg vom
> aliasing).
>
> Vier dinge, die daran hängen.
>
> **Der brennpunkt gehört in den neuberechnungs-schlüssel.** Welche körper ein
> eigenes gitter bekommen, hängt an referenz-, auswahl- und ursprungskörper —
> und `_needs_recompute` kannte keinen davon. Eine frisch angeklickte linie
> zeigte deshalb erst das grobe gemeinsame gitter und sprang erst bei der
> nächsten neuberechnung glatt: das „low poly beim anwählen, dann sofort
> glatt". Ein klick ist ein seltenes ereignis, die neuberechnung kostet
> 0.67 ms und läuft dann einmal.
>
> **Aber `id(points)` gehört NICHT hinein.** Der prädiktor baut sein punkt-array
> im halt fast jedes bild neu — vorn verbraucht, hinten verlängert —, die kurve
> ist dabei dieselbe, nur fortgeschritten. Mit der array-adresse im schlüssel
> galt jedes dieser bilder als neue geometrie: **284 von 300 bildern** voll neu
> gerechnet statt der ~10, die die `_sample_step`-drossel darunter zulässt —
> **rend_calc 22.9 ms statt 13.3 ms**, und das war der ganze unterschied.
> Die zweite hälfte ist die schlimmere: `id()` ist eine ADRESSE, die CPython
> sofort wiederverwendet. Vor der ordner-umstellung fiel der schlüssel in
> **129 von 300 bildern** zufällig auf die adresse des gerade freigegebenen
> arrays und ließ die neuberechnung aus, obwohl der inhalt ein anderer war.
> Die drossel war damit nie schnell, sondern nichtdeterministisch; die
> umstellung hat bloß das allokationsmuster verschoben und den zufall
> weggenommen. `generation` ist das richtige signal — der prädiktor zählt ihn
> genau dann hoch, wenn wirklich neue geometrie vorliegt
> (`_invalidate_derived_caches(soft=False)` in `ship/predictor/compute.py` und
> `jobs.py`), und lässt ihn beim bloßen fortschreiben stehen (`soft=True` im
> halt). Zugelassene staleness gemessen: **max 0.26 % des fensters**
> (847 s auf 328 000 s), median 0.06 %.
>
> Gegenprobe: `tests/orbit_lines_test.py`, „50 ruhige frames → 0
> neuberechnungen" und „je generation genau eine neuberechnung". Beide gehen
> durch, sobald die adresse draußen ist — mit ihr drin bestanden sie nur, weil
> der test ein stabiles array übergibt und den fall gar nicht erzeugt.
>
> **Gemessen wird weiter auf dem gemeinsamen gitter.** `closest_approach`
> braucht den körper zu den zeiten der SCHIFFSpunkte — ein eigenes gitter
> hätte dort keine schiffsposition daneben. `miss`/`t_min` sind deshalb
> unverändert, und mit ihnen deckkraft und enthüllung.
>
> **Gekappt wird erst, wenn verfeinern nicht mehr reicht.** Über
> `max_periods_drawn` (16 = `max_track_samples / samples_per_period`) hinaus
> helfen stichproben nicht mehr: 1871 übereinanderliegende umläufe sind auch
> sauber abgetastet nur eine gefüllte scheibe, und sie kosten 1.5 mio
> Kepler-lösungen. Dann wird nur das **ende** des fensters gezeichnet — die
> letzten umläufe vor der ankunft. Zehn Mars-umläufe (vorher schon brauchbare
> 10°) bleiben damit vollständig und bekommen 649 statt 192 stichproben.
>
> **Die endkappe überlebt das, und sie ist der grund für die richtung der
> kappung.** Gemessen bleibt der letzte punkt der gezeichneten spur bei
> **0.000e+00 m** von der exakten position des körpers zur endzeit. Preis: die
> enthüllung rollt bei einem gekappten körper nicht mehr von seiner HEUTIGEN
> position ab, sondern vom anfang des gezeichneten fensters.
>
> Für jeden körper, dessen periode das fenster überdauert, ändert sich gar
> nichts — gegengeprüft auf **dieselben array-objekte**, also auch dieselbe
> gemeinsame knotentabelle. Und ein eigenes gitter wird nur für körper gebaut,
> die überhaupt eine linie bekommen (`reveal_fraction > 0`, referenz- und
> auswahlkörper); sonst wären es ~27 000 Kepler-lösungen je neuberechnung für
> 20+ linien, die nie erscheinen.

> **`track[::stride]` verliert das ende — und das ende ist hier der messwert.**
> Bei 192 punkten und stride 64 endet die kurve auf index 128 von 191, es
> fehlen also **33 %**. Auf dem schirm sah das aus wie eine mondlinie, deren
> endpunkt zu nichts gehört. `stride_indices()` nimmt den letzten punkt
> immer mit; §12 prüft das über sechs punktzahlen und die ganze stride-leiter,
> mit gegenprobe auf der nackten scheibe.

> **Die faint volllinie: EIN ganzer umlauf, hinter der enthüllten spur.**
> Zusätzlich zur enthüllten zukunfts-spur bekommt jeder kandidat eine blasse
> linie über **genau eine umlaufperiode** des körpers um seinen elter
> (`orbital_period()` = `2π√(a³/μ)`). Sie ist dasselbe `körper(t) − ursprung(t)`
> durch dieselbe pipeline — also im plot-frame automatisch richtig: im
> nicht-rotierenden Sonnen-rahmen schliesst sich die Erdbahn zur ellipse, im
> richtungs-frame nicht (der rahmen dreht sich über die periode mit, und das
> ist die wahrheit).
>
> Gebaut in `OrbitLineSet._recompute` (`full_track`, `full_track_t`,
> `full_track_len`) auf einem **eigenen zeitgitter je körper** — jeder hat eine
> andere periode. Gezeichnet in `_draw_orbit_lines` **vor** der hellen spur, mit
> `alpha = spur_alpha · orbit_line_full_alpha_mult`, **alle stichproben
> projiziert** (kein stride: `full_track_len` ist die welt-bogenlänge, über eine
> ganze periode trägt die eltern-heliozentrik das ~100-fache der plot-frame-
> länge hinein — es sind ohnehin nur 0–3 linien).
>
> **Eine `FrameAffineTable` JE KÖRPER, über SEINE periode** — kein gemeinsames
> gitter wie bei der spur-tabelle. Die perioden liegen um größenordnungen
> auseinander (Mond 27 d, Mars 687 d); ein über die längste gespanntes gitter
> ließe der kurzen periode zu wenige knoten, und die kubische ursprungs-
> interpolation explodiert dann (gemessen: die Mondlinie schoss mit ±1e6 px aus
> dem bild). Gecacht über die frames auf `(id(frame), fensteranfang,
> fensterende)` — **nicht `id(full_track_t)`**, dessen freigegebene id
> wiederverwendet würde; der cache wird bei jedem frame-wechsel ganz verworfen.
> Gröberer `orbit_line_full_knot_angle` (0.12 gegen 0.05), die linie ist blass.
> `orbit_line_full_max_span_s` (7.5e7 s ≈ 810 d) kappt ab Jupiter: dort ist die
> periode so lang, dass die 3-punkt-schätzung der knotenzahl im rotierenden
> frame aliast.
>
> **Die WELTbahn eines mondes schliesst sich nicht** — die Erde trägt ihn über
> 27 tage ~4.5e10 m um die Sonne. Nur der elternrelative offset ist die
> geschlossene Kepler-ellipse; im plot-frame des elters wird daraus wieder eine
> schleife. `tests/orbit_lines_test.py` §16 prüft periode, fensteranfang/-spanne,
> den deckel und die Erd-schleife im Sonnen-rahmen (lücke 0.0 m).

> **Kosten: 0.6 ms je bild**, ganzes bild 11.3 gegen 10.5 ms median (volles
> sonnensystem, 180 fps, eine sichtbare linie). Der löwenanteil ist die
> knotentabelle (~42 sondierungen); die eigentliche projektion und das
> zeichnen sind reines numpy. Eine volle neuberechnung der spuren kostet
> **0.67 ms**, ein ruhiges bild **0.021 ms** — die neuberechnung läuft nur,
> wenn der prädiktor eine neue linie geliefert hat oder die simzeit um mehr
> als einen stichprobenschritt vorgerückt ist.
