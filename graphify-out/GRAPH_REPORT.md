# Graph Report - spacesim  (2026-09-21)

## Corpus Check
- 144 files · ~269,744 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2612 nodes · 4513 edges · 176 communities (137 shown, 34 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 84 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c75053f7`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- njit
- layout.py
- SystemLoader
- Camera
- Vec2
- Stand 11.03.2026
- ._role
- .apply_to_renderer
- ui_horizon_slider_test.py
- NavballCluster
- SystemMap
- units.py
- warp_predictor_test.py
- with_alpha
- ShaderPipelineMixin
- world
- Widget
- UIRoot
- ViewMixin
- ShipDrawMixin
- _BodyEphemerisMixin
- PredictionDrawMixin
- TextMixin
- Rect
- UIDraw
- DrawMixin
- ManeuverExecutor
- HoldMixin
- ._runs_from_screen_points
- ManeuverPreview
- UIState
- BodyDrawMixin
- preview.py
- BodyBrowser
- bootstrap.py
- TimingHistory
- Telemetry
- Palette
- style.py
- Renderer
- ConfigLoader
- selection_camera_test.py
- UIContext
- ImguiLayer
- art.py
- render_budget_test.py
- BurnProfile
- JobsMixin
- orbit_lines_test.py
- prediction_detail_test.py
- Stack
- BodyCentredBodyDirectionReferenceFrame
- horizon_targets_test.py
- ManeuverDrawMixin
- Manöverknoten — planen, vorschauen, brennen
- ManeuverAxesBlock
- ReferenceFrameSelector
- reference_frames.py
- BackgroundLayer
- GLDeviceMixin
- _ManeuverPlate
- Toggle
- ManeuverPlan
- ApsisTooltip
- Slider
- icon.py
- ComputeMixin
- Gliederung
- OrbitDrawMixin
- InputRouter
- ManeuverNode
- ui_hud_test.py
- devui.py
- SegmentBar
- background_test.py
- body_icon_test.py
- ShipBadge
- Readout
- orbit_lines.py
- Ausgangslage im April 2026
- HorizonPolicy
- maneuver_hud_test.py
- Hud
- TextRenderer
- .sample
- advance
- OrbitLineSet
- Entwicklung nach dem April-Stand
- ._render_tracked
- .draw
- frame_affine_at
- BodyCentredNonRotatingReferenceFrame
- HorizonSlider
- devui_timing_test.py
- maneuver_render_test.py
- _burn_arc_numba
- compass_from_frame_direction
- ._recompute
- .update
- .levels
- Window
- renderer.py
- _Rng
- 09.05.2026 April-Mai Vergleich.md
- Predictor und Integratoren
- 11.04.2026 Finales Update für die Osterferien:
- ._origin_xy_arrays
- Hintergrund-Ebene — Sternenfeld und rekursives Dreiecksgitter
- parse_hex_color
- _w17_frame
- ._build_draw_track
- Renderer — draw path, GL state, shaders, screen conventions
- world_kernels.py
- _earth_velocity
- PlanetStyle
- DevContext
- _hermite23
- _line_with_head
- ui_render_test.py
- _imul
- Rendering der Predictor-Linie
- HudPanel
- spacesim
- _empty_points
- .orbital_speed_scale
- WarpBar
- SnapRosette
- Test map
- OrbitalElements
- Role
- frame_project
- 01 Einleitung.md
- Quellen.md
- Predictor RKN
- ZoomButtons
- .star_table
- ._texture_for
- Camera, input and the in-app controls
- IconRail
- .enable_epicycles
- Config and system loading
- The player-facing HUD (`ui/`)
- compass_to_screen
- _emit_shape
- Project-Entwicklung.md
- Integrator.md
- Predictor
- maneuver_preview_test.py
- _hermite_at
- render/__init__.py
- render/draw.py
- arc_ruler
- _kosten
- OrbitLineEntry
- Procedural vector art — planets, their marker, and the ship
- Physics engine — world, kernels, body model, time warp
- Trajectory predictor — the drawn look-ahead line
- 2. Physikalische Grundlagen
- .warp_step_allowed
- .speed_at_radius
- .text_mission_time
- .view_mode_label
- .gauge_speed
- .gauge_altitude
- .radial_fraction
- .set_thrust_level
- ui_units_test.py
- devui.md
- orbit-lines.md
- reference-frames.md
- Projekt Stand 08.09.2026

## God Nodes (most connected - your core abstractions)
1. `Vec2` - 71 edges
2. `Widget` - 57 edges
3. `ConfigLoader` - 46 edges
4. `world` - 45 edges
5. `Camera` - 45 edges
6. `Renderer` - 43 edges
7. `Rect` - 41 edges
8. `Telemetry` - 41 edges
9. `Predictor` - 39 edges
10. `with_alpha()` - 37 edges

## Surprising Connections (you probably didn't know these)
- `body` --uses--> `Vec2`  [INFERRED]
  bodies/body.py → physics/vec.py
- `frame()` --uses--> `body`  [INFERRED]
  tests/prediction_detail_test.py → bodies/body.py
- `orbital_period()` --indirect_call--> `mass()`  [INFERRED]
  bodies/orbit_lines.py → ui/units.py
- `soi_radius()` --indirect_call--> `mass()`  [INFERRED]
  bodies/orbit_lines.py → ui/units.py
- `ShipDrawMixin` --uses--> `Vec2`  [INFERRED]
  render/ship.py → physics/vec.py

## Import Cycles
- None detected.

## Communities (176 total, 34 thin omitted)

### Community 0 - "njit"
Cohesion: 0.18
Nodes (27): _compute_acc_nearest_numba(), _compute_acc_numba(), _compute_acc_time_numba(), _leapfrog_step_numba(), _local_timescale_numba(), Beschleunigung und schrittverfahren. Zwei familien, und der unterschied ist…, `min_i sqrt(r_i^3 / (G m_i))` am ORT (x, y) zur zeit local_t. Wort fuer wort…, _rk4_step_numba() (+19 more)

### Community 1 - "layout.py"
Cohesion: 0.06
Nodes (52): ease(), Kern der UI-schicht: rechtecke, verankerung, widget-basis, eingabe-routing.…, Framerate-unabhaengiges exponentielles easing (1 - exp(-rate * dt)), dieselbe…, Der schwebezettel an einer Ap/Pe-raute auf der bahnlinie. Die raute selbst und…, Ausklappbare koerperliste zur wahl des BEZUGSKOERPERS. Die taste R blaettert…, frame(), plate(), Die formsprache der instrumententafel -- an EINER stelle. Vier bauteile, aus… (+44 more)

### Community 2 - "SystemLoader"
Cohesion: 0.06
Nodes (28): body, kepler_relative_xy(), Return this body's Kepler position at simulation time t without modifying…, Position nach `dt` auf der eigenen bahn -- exakt, siehe kepler_relative_xy().…, Bahnposition relativ zum mutterkoerper nach `dt`, EXAKT geloest. Rueckgabe `(x,…, schiff, bequemlichkeitsfunktion zum erstellen von Vec2., vec() (+20 more)

### Community 3 - "Camera"
Cohesion: 0.05
Nodes (28): Camera, Setzt ein Objekt zur Verfolgung (SOFORT, ohne anflug)., Körper, auf den `Home` die ansicht zurückholt (das schiff). Die kamera kennt…, Fährt die ansicht GEGLÄTTET auf `body` und heftet sie dort an. Der trick ist,…, Taste Home: geglättet zurück zum heimatkörper (dem schiff)., Beendet die Objektverfolgung. Die aktuelle ansicht wird als freies ziel…, Die tatsaechliche zoom-obergrenze (px je meter). `max_scale` ist die harte,…, Weltposition, auf die die kamera gerade zuläuft. (+20 more)

### Community 4 - "Vec2"
Cohesion: 0.06
Nodes (20): skalare multiplikation (self * scalar)., in-place vektor-addition., in-place vektor-subtraktion., in-place skalare multiplikation., skalare multiplikation (scalar * self)., quadratische länge für performance-kritischen code., betrag (länge) des vektors., erstellt eine kopie dieses vektors. (+12 more)

### Community 5 - "Stand 11.03.2026"
Cohesion: 0.05
Nodes (40): bodies.py, camera.py, loader.py, main.py, predictor_mp.py, predictor.py, rendering.py, schiff.py (+32 more)

### Community 6 - "._role"
Cohesion: 0.25
Nodes (4): Rollengroesse -> pixelgroesse, fuer die pixelschrift GERASTET. SB Liquid ist…, Die pixelschrift wird OHNE kantenglaettung gerastert. Mit glaettung traegt SB…, Die instrumentenschrift setzt ZIFFERN AUF FESTER BREITE. SB Liquid ist nicht…, Laufweite dieser rolle in pixeln (em-wert * schriftgroesse).

### Community 7 - ".apply_to_renderer"
Cohesion: 0.07
Nodes (23): Any, _float_list(), Liest die konfiguration ein und gibt sie zurueck. `utf-8-sig` statt `utf-8`:…, Gibt einen abschnitt (z. B. 'camera') als dict zurueck, notfalls leer., Liest einen wert ueber einen punkt-pfad, z. B. get('camera.zoom_factor').…, Wandelt `value` mit `caster` um; bei unpassendem wert warnung + default. so…, Liest einen kommazahl-parameter., Liest einen ganzzahl-parameter. (+15 more)

### Community 8 - "ui_horizon_slider_test.py"
Cohesion: 0.11
Nodes (11): _Dummy, make(), _Motion, _Predictor, _PredLen, Kopftest fuer ui/widgets/rate_slider.py -- die federphysik und die…, value_box ist eine 1-elementige liste, damit on_change zurueckschreibt., _Text (+3 more)

### Community 9 - "NavballCluster"
Cohesion: 0.09
Nodes (15): NavballCluster, Punkt auf dem linken bogen -> schubstufe, oder None. Getroffen wird ein RING-…, Das rad stellt den schub -- aber NUR ueber dem bogen und dem streifen. Ueberall…, Das ORB- (links) bzw. ALT-kaestchen (rechts)., Der THR- (links) bzw. V/S-streifen (rechts) darunter., Der ORBITAL.INFO-block am unteren rand. Er ist nur um eine halbe flankenbreite…, Eine der beiden messwert-flanken. Aufbau der vorlage: winzige gesperrte…, Der schmale streifen unter einer flanke. (+7 more)

### Community 10 - "SystemMap"
Cohesion: 0.09
Nodes (20): _body_color(), _log_spread(), ``body.color`` (0..255 aus dem JSON) -> zeichenfarbe. Unveraendert uebernommen,…, Werte auf [low, high] verteilen -- logarithmisch, optional nach rang.…, Der ECHTE bahnwinkel des koerpers um seinen mutterkoerper. Aus den momentanen…, Die karte als widget: kachel, ausfahren, treffer, mond-ansicht., (zentralkoerper, planeten, monde je planet), einmal aufgebaut. Die koerperliste…, Groesse aus dem ausfahr-fortschritt, nicht aus dem schalter. Zwischen kachel… (+12 more)

### Community 11 - "units.py"
Cohesion: 0.09
Nodes (34): altitude(), angle(), countdown(), countdown_compact(), delta_v(), distance(), duration(), _finite() (+26 more)

### Community 12 - "warp_predictor_test.py"
Cohesion: 0.11
Nodes (16): _advance_chunked22(), build(), _burn(), _bvel24(), _clear_of_bodies21(), _departure24(), _leo_world(), _line() (+8 more)

### Community 13 - "with_alpha"
Cohesion: 0.10
Nodes (15): AttitudeRing, _polar(), Der Attitude-Ring -- das Herzstueck des HUDs. Ein 2D-lagemesser statt einer…, RUND, nicht quadratisch. Das widget ist ein quadrat, der ring darin ein kreis.…, Loslassen gibt das schiff SOFORT wieder frei. _manual_heading faellt weg, damit…, Das schwache polargitter auf der ringflaeche. Es macht die flaeche zu einem…, Die vier himmelsrichtungen, immer aufrecht. Eine richtung wird WEGGELASSEN,…, Die vier bahnsymbole als VEKTOREN, nicht als schriftzeichen. Der entwurf setzt… (+7 more)

### Community 14 - "ShaderPipelineMixin"
Cohesion: 0.10
Nodes (16): Die OpenGL-pipelines des Renderers und der GL-zustandscache. Die GLSL-quellen…, Programm der positions-marke. Teilt sich das statische einheits-quad mit der…, Shader uebersetzen, VAOs/VBOs anlegen, GL-zustand cachen. Die…, Programme fuer die vektor-zeichnung der koerper. Anders als die uebrigen…, Texturierte quads (labels, HUD) in der ortho-konvention (y nach oben)., Hintergrund-ebene: vollbild-quad (gitter) + punkt-sprites (sterne). Beide…, Erstellt wiederverwendbare puffer, programme und VAOs für kritische render-…, Geteilter dynamischer vertex-puffer für polylines und ortho-geometrie. (+8 more)

### Community 15 - "world"
Cohesion: 0.12
Nodes (11): Decke fuer die integrator-schrittweite aus der raffung ableiten. Die kosten von…, Tatsaechlich benutzte decke -- nie kleiner als die konfigurierte., Rueckt die welt um `sim_seconds` vor, aufgeteilt in stuecke. Die stueckgroesse…, Weltposition des koerpers zur simulationszeit `time_s`…, One explicit RKN4-style step for r'' = a(r, t). p0: initial position v0:…, Störmer-Verlet (KDK leapfrog) — 2nd-order symplectic RKN., Adaptive embedded RKN-style step by step-doubling. Compares: - one full step h…, Koerperzustand in flache arrays. None = fuer den kernel ungeeignet. Abgelehnt… (+3 more)

### Community 16 - "Widget"
Cohesion: 0.07
Nodes (14): Basisklasse. Haelt verankerung, groesse, kinder und hover-zustand. Groessen und…, Eigengroesse in PIXELN fuer size-komponenten, die None sind. Ueberschreiben, wo…, Loest die groessen-angabe auf. Pro achse sind drei angaben erlaubt: eine zahl…, Effektive groesse in pixeln: explizite vorgabe, sonst measure(). Container…, Berechnet self.rect aus anker, abstand und groesse., Flaeche, in der kinder verankert werden. Panels ziehen hier ihr padding ab., Eigene darstellung. Basisklasse zeichnet nichts., Trefferflaeche. Bekommt ctx, weil sie nicht immer gleich self.rect ist -- ein… (+6 more)

### Community 17 - "UIRoot"
Cohesion: 0.12
Nodes (9): Sich selbst und alle nachkommen, eltern zuerst., Wurzel des widget-baums. Verteilt eingaben und meldet, ob sie verbraucht…, Solange ein widget gedrueckt wird, gehoert die maus ihm -- auch wenn der zeiger…, Alle sichtbaren widgets in ZEICHENREIHENFOLGE, hinten zuerst. Sortiert nach…, Oberstes widget unter dem zeiger, das die maus beansprucht. Genau die…, True = das ereignis gehoert der UI und darf NICHT weitergereicht werden., UIRoot, build_hierarchy() (+1 more)

### Community 18 - "ViewMixin"
Cohesion: 0.09
Nodes (11): Von der punkteliste abgeleitete zwischenergebnisse verwerfen. soft=True heisst:…, Feinste punktdichte, die den HORIZONT noch traegt -- oder None. Die kernel…, Was HERAUSKOMMT: punkte, Ap/Pe-marker, laenge, abstand, zoom. Horizont…, Wie viel von der gerechneten kurve GEZEICHNET wird, in metern. None oder >= der…, Zahl der zu zeichnenden fuehrenden punkte, oder None fuer alle. Ueber den…, Die zu ZEICHNENDE kurve (siehe set_display_length). Der ausschnitt wird gemerkt…, Traegt diese punkteliste brauchbare geschwindigkeits-spalten? Alles oder…, Apoapsis/Periapsis-Marker der aktuellen Prädiktionslinie. Rückgabe: ndarray (m,… (+3 more)

### Community 19 - "ShipDrawMixin"
Cohesion: 0.11
Nodes (12): Returns the ship's apparent speed in the active plotting frame. This respects…, Massstabs-faktor des schiffs fuer die aktuelle zoomstufe. 1.0 bei…, Gezeichnete schiffslaenge in echten bildschirm-pixeln. Basislaenge (design-…, Das schiff: sprite, pfeil, fahne und die orientierung.…, Halbe hoehe der gezeichneten schiffs-grafik in bildschirm-pixeln. Bezugsgroesse…, Die gebaute schiffs-grafik, gecacht bis die akzentfarbe wechselt., Helligkeit der abgasfahne, weich zwischen leerlauf und schub.…, Das schiff aus `ship_art` zeichnen -- in festen bildschirm-pixeln. Die grafik… (+4 more)

### Community 20 - "_BodyEphemerisMixin"
Cohesion: 0.17
Nodes (12): _body_arg_periapsis(), _body_true_anomaly(), _BodyEphemerisMixin, _build_kepler_elements(), _has_scripted_orbit_data(), Kuerzeste umlaufzeit in der elternkette von `body`, oder None. Die WELT-…, Stapelfassung von _body_world_position_exact ueber ein zeit-gitter. `qt` ist…, Vektorisierte fassung von _scripted_top_level_position_at_time. None = die… (+4 more)

### Community 21 - "PredictionDrawMixin"
Cohesion: 0.12
Nodes (8): PredictionDrawMixin, Alle stichproben-punkte in EINEM rutsch projizieren. Gibt ``(None,…, Erlaubte abweichung der gezeichneten linie -- in metern und pixeln. Der…, Kubische zwischenpunkte setzen -- nur sichtbar, nur so fein wie noetig.…, Die vorhersagelinie: abtastung, Hermite-verfeinerung, Ap/Pe-marker. EINE…, Das ROH-scan-budget dorthin legen, wo die linie im BILD liegt.…, Gleichmaessige stichprobe der rohpunkte -- GEMERKT, nicht neu gebaut. Das…, Zeichnet apoapsis/periapsis-marker des predictors auf die linie. Marker kommen…

### Community 22 - "TextMixin"
Cohesion: 0.11
Nodes (13): Schriften und der label-textur-cache des Renderers. Das spieler-HUD hat mit…, Setzt die DESIGN-schriftgrößen und baut die fonts neu auf., Benutzer-skalenfaktor (multiplikativ auf die automatische skala)., Beschriftungs-textur besorgen -- moeglichst eine wiederverwendete. Wie in…, Schriften, der label-textur-cache und getoentes blitten. UI-groessen sind…, Rastert eine beschriftung -- bei bedarf HART und GESPERRT. Zwei zugestaendnisse…, Leitet ui_scale aus der fensterhöhe ab. Gibt True bei änderung zurück. Skaliert…, Text an TOP-DOWN koordinaten zeichnen (x = links, y = oberkante). Nimmt dem… (+5 more)

### Community 23 - "Rect"
Cohesion: 0.13
Nodes (4): _Ctx, Das Nötigste, das der Widget-Code berührt., Achsenparalleles rechteck in top-down pixeln., Rect

### Community 24 - "UIDraw"
Cohesion: 0.11
Nodes (11): Zeichen-primitive der UI-schicht, alle auf EINEM SDF-shader.…, Abgerundetes rechteck. (x, y) = obere linke ecke, top-down. radius: skalar oder…, Kreis um (cx, cy), top-down., Kreisring. Umgesetzt als kreis OHNE fuellung mit rahmen der gewuenschten…, Kreisbogen. Winkel gegen den uhrzeigersinn, 0 = nach rechts., Beliebig gedrehte linie -- ein um ihre achse rotiertes rechteck. cap='round'…, Instanz in den stapel legen -- gezeichnet wird erst in flush().…, Alle seit dem letzten flush gesammelten formen zeichnen. (+3 more)

### Community 25 - "DrawMixin"
Cohesion: 0.10
Nodes (13): DrawMixin, Liang-Barsky clipping for screen-space line segments. Returns (cx0, cy0, cx1,…, Die zeichen-primitive: linien, ortho-formen, texturen, clipping. Alles hier…, Converts one logical predictor polyline into multiple visible screen-space…, Lädt ein (N,2)-float32-array in den geteilten dynamischen VBO. orphan()…, Kernel-weg von `_build_clipped_polyline_runs`. Ein numba-aufruf statt einer…, Zeichnet eine textur als quad in der ortho-konvention (y nach oben). (x, y) ist…, Zeichnet eine bildschirm-space polyline (top-down-konvention) via GLSL+VBO. (+5 more)

### Community 26 - "ManeuverExecutor"
Cohesion: 0.11
Nodes (9): ManeuverExecutor, Den naechsten knoten scharfschalten. False, wenn das nicht geht., Von der hauptschleife gerufen, sobald der spieler selbst steuert. Handeingabe…, Die schrittklemme fuer eine gegebene sim-zeit (testbar ohne welt)., Wie weit die welt in DIESEM frame hoechstens vorruecken darf. Scharf:…, Muss die raffung jetzt auf echtzeit herunter? Ja, sobald die zuendung naeher…, VOR `world.step(sim_seconds)` aufrufen. Legt das delta-v des bevorstehenden…, Scharfschalten, ausrichten, zuenden, brennen, aufraeumen. (+1 more)

### Community 27 - "HoldMixin"
Cohesion: 0.10
Nodes (11): HoldMixin, Der halt, das verbrauchen der kurve und das umschalten auf einen neuen bahnast.…, Kurve VERBRAUCHEN statt starr verschieben. Rueckgabe: die zahl der vorn…, Setzt den kurvenanfang auf das schiff. DER REGELFALL IST DAS VERBRAUCHEN, NICHT…, Gibt die Anzahl der Einträge in `new_points` zurück, die sich von `old_points`…, Zeitraffer-halt ein/aus. Ausschalten erzwingt eine neuberechnung. Die beiden…, Die gehaltene kurve ist ueberholt (schub, rahmenwechsel, ...). `soft=True`…, Kurve VERBRAUCHEN statt neu rechnen. True = frame ist erledigt. Die vorhersage… (+3 more)

### Community 28 - "._runs_from_screen_points"
Cohesion: 0.15
Nodes (11): _compact_min_step_numba(), _densify_numba(), _max_gap_refine_numba(), Die Numba-fassungen der reinen zahlenschleifen im linien-zeichenweg. Min-step-…, Zu weit auseinanderliegende RDP-punkte wieder auffuellen. Dieselbe schleife wie…, Segmente laenger als `max_segment` linear unterteilen. Zwei durchgaenge…, _rdp_keep_numba(), Das zeichnen der vorhersagelinie. Der teure teil ist nicht das zeichnen,… (+3 more)

### Community 29 - "ManeuverPreview"
Cohesion: 0.12
Nodes (10): _Done, ManeuverPreview, Die kette, ihre marker, und der weg, auf dem sie NEBENHER laeuft. NEBENLAEUFIG,…, Reichweite setzen. Der naechste neuaufbau uebernimmt sie., Ein fertiges ergebnis einwechseln. True, wenn eines kam., Auf einen laufenden auftrag warten und ihn einwechseln. Nur fuer tests und…, Nur rechnen, wenn sich etwas bewegt hat. True, wenn gerechnet wurde.…, SYNCHRON rechnen und sofort einwechseln (tests, screenshots). (+2 more)

### Community 30 - "UIState"
Cohesion: 0.11
Nodes (8): Beobachtbarer ansichts-zustand, den HUD und tastatur gemeinsam bedienen.…, Setzt die auswahl. Gibt True zurueck, wenn sie sich geaendert hat. LOEST…, Die drei modi der HUD-rahmenwahl in einem schritt. 'surface' -> mitrotierender…, Aktueller modus als index fuer die HUD-rahmenwahl., Erzwingt ein neuanwenden ohne aenderung (start, system-neuladen)., Bezugsrahmen-auswahl, referenzkoerper und overlay-schalter., Zweiter koerper fuer den body-direction-rahmen. Bevorzugt den mutterkoerper des…, UIState

### Community 31 - "BodyDrawMixin"
Cohesion: 0.05
Nodes (27): BodyDrawMixin, Positions-marke eines körpers, konstanter bildschirmgröße. `radius` ist der…, Die koerper: scheibe, prozedurale marke, vektor-look, beschriftung,…, Die spanne der PHYSISCHEN koerper-radien im geladenen system. Einmal je frame…, 0..1: wo dieser koerper-radius innerhalb der GELADENEN spanne liegt. LOG-…, Der GEZEICHNETE radius der marke -- ein je koerper KONSTANTER wert aus seinem…, Deckkraft der marke bei diesem echten bildschirmradius. 1.0 unterhalb der…, Cache-schluessel: alles, was die zeichnung bestimmt. Bewusst NICHT `id(body)`:… (+19 more)

### Community 32 - "preview.py"
Cohesion: 0.14
Nodes (19): Der autopilot, der einen knoten wirklich fliegt. IDLE --arm()--> ARMED…, Manoeverknoten: planen, vorschau zeichnen, automatisch brennen. profile.py das…, burn_direction_world(), orbital_basis(), Manoeverknoten und der plan, der sie haelt. EIN KNOTEN SPEICHERT ZWEI ZAHLEN…, Prograde und einwaerts-normal am ort des knotens, in WELTkoordinaten. `rel_*`…, Weltrichtung und betrag des geplanten schubs. Zurueck kommt `(dx, dy,…, body_state_at() (+11 more)

### Community 33 - "BodyBrowser"
Cohesion: 0.17
Nodes (7): _body_dot_color(), BodyBrowser, Symbolknopf plus die dahinter liegende koerperliste. EIN widget statt knopf +…, Gecachte gliederung. Die koerperliste aendert sich zur laufzeit nicht, der…, Das panel in seiner MOMENTANEN aufklapp-hoehe. Es waechst aus der unterkante…, Symbol, beschriftung und der aufklapp-winkel. Das systemsymbol (zentralkoerper,…, ``body.color`` (0..255-tripel aus dem JSON) -> zeichenfarbe. Bewusst…

### Community 34 - "bootstrap.py"
Cohesion: 0.10
Nodes (14): App, build_app(), Den ganzen apparat zusammenbauen: welt, kamera, predictor, renderer, UI. DIE…, Alles, was die hauptschleife braucht -- ein sack voll verweise. Bewusst keine…, rotation mit echtem (wanduhr-)delta behandeln damit das drehen sich glatt…, Manual nose thrust as acceleration per real frame. dies stellt sicher, dass der…, Latch/unlatch an orientation-hold. Tapping the active mode clears it., Hold the ship nose on a world-space heading supplied by the renderer. The… (+6 more)

### Community 35 - "TimingHistory"
Cohesion: 0.12
Nodes (10): Ringpuffer der per-frame zeitmessung, direkt fuer imgui.plot_lines. Ein…, Eine probe. Heisser pfad -- keine allokation, kein dict, kein try. `ui_calc`…, Laenge umstellen und dabei die JUENGSTEN proben behalten., `(feld, offset)` fuer plot_lines. Offset = aelteste probe., Kopie in chronologischer reihenfolge (alt -> neu). Nur fuer tests und ausgaben…, `(cur, avg, max)` ueber den GEFUELLTEN teil des puffers. Der ungefuellte rest…, Zerfallender spitzenwert der serie. Nicht gerastert. Je frame HOECHSTENS EINMAL…, Gerasterter achsen-maximalwert einer serie. (+2 more)

### Community 37 - "Palette"
Cohesion: 0.11
Nodes (7): Palette, Die vier bedeutungsfarben plus die feste, dunkle grundierung. Der grund (panel,…, Setzt die vier farben und leitet alle rollen neu ab., Der farbige schein hinter einem block -- ein schlagschatten mit versatz null…, Buendelt die stufen. Erreichbar ueber UIContext.theme., Der EINE farbsatz (siehe modulkopf)., Theme

### Community 38 - "style.py"
Cohesion: 0.16
Nodes (20): build_planet_style(), _color_basis(), _emit_ring(), _euler_matrix(), expand_segments(), _fbm(), _hash3(), _hsl() (+12 more)

### Community 39 - "Renderer"
Cohesion: 0.15
Nodes (7): IdentityReferenceFrame, Der Renderer -- zusammengesetzt aus mixins, ein zustand. Die klasse ist ueber…, Konvertiert einen Welt-Punkt zu einer bestimmten Sim-Zeit in…, Die debug-textwand unten links -- ein schneller rohwert-blick. Standardmaessig…, Renderer, Knobs, Nur die felder, die `_ship_zoom_shrink_factor` anfasst. Die kennlinie wird OHNE…

### Community 40 - "ConfigLoader"
Cohesion: 0.09
Nodes (14): ConfigLoader, Laedt `config.json` und verteilt die parameter auf die einzelnen module. die…, Path, Initialisiert den Loader. falls `filepath` weggelassen oder relativ angegeben…, Predictor, Die vorausberechnete bahnlinie -- zusammengesetzt aus mixins. Ueber…, build(), Welt + schiff auf einer kreisbahn um die Erde + predictor. (+6 more)

### Community 41 - "selection_camera_test.py"
Cohesion: 0.11
Nodes (14): blob_centroid(), draw(), FakeBody, look_at(), make_camera(), Regressionstest fuer auswahl per mausklick und den kamera-anflug. Geprueft…, Schnelles rein/raus-zoomen ueber 240 frames, koerper faehrt mit. Gibt die…, Einen frame zeichnen. Die koerper-optik wird NEBENLAEUFIG gebaut, ein einzelner… (+6 more)

### Community 42 - "UIContext"
Cohesion: 0.25
Nodes (3): Alles, was widgets zum zeichnen und messen brauchen. Bewusst ein explizites…, Design-einheiten -> pixel. Das gegenstueck zu Renderer.ui_px(). Nimmt auch eine…, UIContext

### Community 43 - "ImguiLayer"
Cohesion: 0.14
Nodes (4): ImguiLayer, ImGui-overlay auf dem geteilten moderngl-context., pygame-ereignis nach imgui uebersetzen., imgui-frame abschliessen und ueber moderngl zeichnen.

### Community 44 - "art.py"
Cohesion: 0.13
Nodes (23): _arc_half(), build(), _circle(), _cross3(), _dash(), _hex_rgb(), _in_triangle(), _plume_polygons() (+15 more)

### Community 45 - "render_budget_test.py"
Cohesion: 0.25
Nodes (6): busy_wait(), churn(), draw_frame(), Regressionstest fuer die zeitmessung des zeichenwegs und fuer lecks. Anlass:…, `n` frames mit wechselndem zoom, zeit und auswahl. Der wechsel ist absicht: ein…, Aktives warten. `time.sleep` waere hier zu grob (windows: ~15 ms).

### Community 46 - "BurnProfile"
Cohesion: 0.11
Nodes (9): BurnProfile, Hebelstellung 0..1 -- was der schubbogen des HUDs anzeigt., Aufsummiertes delta-v von der zuendung bis `tau`., Delta-v ueber ein zeitfenster, GESCHLOSSEN gerechnet. Der ausfuehrer benutzt…, Trapez- oder dreiecksprofil fuer ein gegebenes delta-v., Zeit von der zuendung bis zum knoten -- die haelfte der dauer. Gilt, weil beide…, Absolute sim-zeit, zu der der schub einsetzen muss., Schubbeschleunigung `tau` sekunden nach der zuendung. ACHTUNG:… (+1 more)

### Community 47 - "JobsMixin"
Cohesion: 0.15
Nodes (8): JobsMixin, Die asynchrone rechen-pipeline des predictors. Hier wird nicht gerechnet, hier…, Obergrenze: konfiguration und verfuegbare kerne., So viele gleichzeitige laeufe, dass je BILD eines fertig wird. Die dauer einer…, Schub-neuberechnung ANFORDERN statt sie im hauptthread zu erzwingen. Waehrend…, Die asynchrone rechen-pipeline. Mehrere rechnungen laufen VERSETZT…, Neue horizont-/abstands-kurve ANFORDERN, ohne den halt aufzugeben. Dasselbe…, Wie viele auftraege rechnen gerade? `_pending_futures` enthaelt auch bereits…

### Community 48 - "orbit_lines_test.py"
Cohesion: 0.09
Nodes (13): check(), close(), _constant_rate_position_at_time(), CountingFrame, _drawn_step_deg(), line_missing_moon_by(), Regressionstest fuer orbit_lines.py -- die bahn-linien der koerper. Reine…, Winkelschritt der kurve koerper(t)-ursprung(t) um den bildmittelpunkt. (+5 more)

### Community 49 - "prediction_detail_test.py"
Cohesion: 0.12
Nodes (16): _drawn7(), drawn_arc_length(), drawn_radius_error(), _FakeCam, frame(), _hermite7(), _miss7(), Die vorhersagelinie wird so fein gezeichnet, wie der schirm es zeigt. Gemessen… (+8 more)

### Community 50 - "Stack"
Cohesion: 0.21
Nodes (4): Panel, Unsichtbarer container, der seine kinder aneinanderreiht. Die verankerung aus…, Abgerundete flaeche mit rahmen, schatten und innenabstand. blocks_mouse ist…, Stack

### Community 51 - "BodyCentredBodyDirectionReferenceFrame"
Cohesion: 0.12
Nodes (4): BodyCentredBodyDirectionReferenceFrame, Stapelfassung von to_this_frame_xy. None = nicht moeglich. Der renderer…, ReferenceFrame, TargetBodyDirectionReferenceFrame

### Community 52 - "horizon_targets_test.py"
Cohesion: 0.15
Nodes (13): horizon_compute_rung(), horizon_targets(), predictor_horizon_lengths(), Die laengenregel des vorhersage-horizonts. Die modulfunktionen sind rein, damit…, (gezeichnete laenge, gerechnete laenge) fuer den vorhersage-horizont.…, Naechsthoehere sprosse einer groben leiter ueber `wanted`. Die leiter haengt an…, Wie `predictor_horizon_lengths`, aber mit dem slider-griff. DIE GERECHNETE…, apply() (+5 more)

### Community 53 - "ManeuverDrawMixin"
Cohesion: 0.17
Nodes (9): ManeuverDrawMixin, Eine (n,5)-linie in schirmpunkte -- grob abgetastet, dann VERFEINERT. NUR…, Den plan zeichnen und seine schirmpositionen melden. Die trefferliste wird bei…, Wo die koerper stehen, wenn das schiff am ENDE DES PLANS ankommt. Dieselbe…, Ein sechseck -- die form, die sonst nichts im bild traegt. Die raute gehoert…, Ein stiel vom marker weg und eine pfeilspitze am ende. `deflection` ist der…, Geplante bahn, knotenmarker und ziehgriffe., Weltrichtung -> SCHIRMrichtung (einheitsvektor), oder None. Zwei schritte: der… (+1 more)

### Community 54 - "Manöverknoten — planen, vorschauen, brennen"
Cohesion: 0.09
Nodes (22): `a_max` ist eine SIM-Zeit-Beschleunigung, Bekannte Grenze, Der Countdown wird ROT, sobald er auf `T+` springt, Der Gizmo fängt die Maus nur über einem Griff, Der njit-Zwilling ist absichtlich doppelt — und wird bewacht, Der Zeitraffer zieht sich vor der Zündung selbst herunter, Die Endkappen stehen am Ende des PLANS, Die gezeichnete Linie wird VERFEINERT, nicht nur ausgedünnt (+14 more)

### Community 55 - "ManeuverAxesBlock"
Cohesion: 0.05
Nodes (23): setter, _edit_text(), ManeuverAxesBlock, ManeuverGizmo, ManeuverPlanBlock, Die ziehgriffe AN DER LINIE -- kein rechteck, nur treffer. Er zeichnet nichts:…, Auslenkung [-1, 1] -> ratenfaktor [-1, 1]. Null in der totzone, sonst `sign *…, Den knueppel INTEGRIEREN -- hier laeuft der wert, nicht im ereignis. Deshalb… (+15 more)

### Community 56 - "ReferenceFrameSelector"
Cohesion: 0.19
Nodes (5): FrameChangeCallback, PlottingFrameAdapter, ReferenceFrameSelector, FrameController, Bezugsrahmen und bezugskoerper anwenden. Haengt an `UIState.on_change`, wird…

### Community 57 - "reference_frames.py"
Cohesion: 0.18
Nodes (18): apparent_orbital_directions(), describe_plotting_frame(), _fallback_secondary_index(), _kepler_true_anomaly_from_mean(), _mean_anomaly_from_true(), new_plotting_frame(), PlottingFrameParameters, _prograde_from_line() (+10 more)

### Community 58 - "BackgroundLayer"
Cohesion: 0.13
Nodes (10): BackgroundLayer, Zustand und rechnung der hintergrund-ebene. Kein GL. `update()` einmal je bild…, True (einmalig), wenn der VBO neu geschrieben werden muss., Staerke des atmenden feldes, auf [0, 1] geklemmt., Der WAHRE gitteranker -- wo das lattice stehen muesste. "frame" (Vorgabe): die…, Der GEZEIGTE anker -- was `levels()` als phase bekommt., drift(), orbiting() (+2 more)

### Community 59 - "GLDeviceMixin"
Cohesion: 0.15
Nodes (8): GLDeviceMixin, Die GL-geraeteschicht des Renderers. Ein MIXIN von `Renderer`, kein eigenes…, Context, FXAA-ziele, present und resize -- die geraeteschicht. Liegt in…, Wendet FXAA Post-Processing an. Erwartet, dass der ziel-framebuffer (screen)…, Fuehrt den buffer-swap aus und schreibt die swap-zeit in die timings. Von der…, Initialisiert OpenGL-Einstellungen (moderngl-state)., Erstellt FBO-textur und framebuffer in aktueller fenstergröße., Initialisiert FXAA Framebuffer und Shader.

### Community 60 - "_ManeuverPlate"
Cohesion: 0.12
Nodes (9): ManeuverBurnBlock, ManeuverNodesBar, _ManeuverPlate, Ein plaettchen im navball-raster. `SIDE` waehlt die flanke, `PLACE` das feld…, Die zur kugel zeigende seite bleibt SCHARF -- wie bei den flanken., Eine zeile: beschriftung links, wert rechts. DREI STUFEN, genau wie…, DV, brenndauer, countdown, EXECUTE -- ueber der ORB-flanke., Darf EXECUTE gedrueckt werden? Nur mit einem knoten, der wirklich delta-v… (+1 more)

### Community 61 - "Toggle"
Cohesion: 0.18
Nodes (4): Button, Klickbare schaltflaeche. Der klick loest beim LOSLASSEN aus, und nur wenn der…, Rastender schalter mit schiebeknopf. value darf ein aufrufbares objekt sein…, Toggle

### Community 62 - "ManeuverPlan"
Cohesion: 0.17
Nodes (4): ManeuverPlan, Bis zu `max_nodes` knoten, nach zeit sortiert., Der zeitlich naechste knoten -- den fliegt EXECUTE., Nach einer direkten feldaenderung aufrufen. Sortiert nach (ein ziehgriff darf…

### Community 63 - "ApsisTooltip"
Cohesion: 0.19
Nodes (6): ApsisTooltip, Die beiden zeilen als (schluessel, wert). ETA ist SIMULATIONSZEIT bis zur…, Hoehe der abstands-fahne, die der renderer unter die raute setzt., Zwei zeilen -- ankunftszeit und bahntempo -- unter der raute., Trefferkreis um die raute, in echten pixeln. Grosszuegiger als die raute…, Der marker unter dem zeiger, oder None. Bei zwei dicht beieinander liegenden…

### Community 65 - "icon.py"
Cohesion: 0.17
Nodes (14): cells_array(), icon_palette(), _mix(), Entwurf A -- die Scheibe wird durchgehend gewuerfelt, radial gewichtet. Kein…, Entwurf D -- Kern plus 2-4 gesaete Zacken. Die Zacken sitzen auf dem…, Die POSITIONS-MARKE eines koerpers -- das icon beim herauszoomen. Nicht zu…, Die drei stufen-farben aus `body.color` (RGB 0..255). Der farbton bleibt…, Das feld als `(grid, grid)` int8 -- fuer tests und fehlersuche. Zeile 0 ist die… (+6 more)

### Community 66 - "ComputeMixin"
Cohesion: 0.16
Nodes (5): ComputeMixin, Vom weltzustand zur punktreihe. Ein SCHNAPPSCHUSS friert alles ein, was die…, sqrt(r_dominant / |g_total|) am schiff, in sekunden -- oder None. Dieselbe…, Ein schnappschuss fuer die manoever-vorschau. Die vorschau…, Mittlere inverse geschwindigkeit ueber den horizont mitschreiben. Einzige…

### Community 67 - "Gliederung"
Cohesion: 0.13
Nodes (14): 1 Einleitung, 2 Physikalische Grundlagen, 3 Näherung und Navigation: Apollo bis heute, 4 Bahnmanöver: Transfers und Swing-by, 5 Entwicklung der Simulation, 6 Validierung: Wie genau ist die Simulation?, 7 Fazit und Ausblick, Dateiorganisation (+6 more)

### Community 68 - "OrbitDrawMixin"
Cohesion: 0.17
Nodes (7): OrbitDrawMixin, Der zustandsbehaftete teil, ueber frames hinweg gehalten. Die konfiguration…, Die bahnlinien der koerper und die aufgezeichneten referenzspuren., Fertig projizierte bildschirmpunkte als polylinie, mit culling., Kleine raute auf dem endpunkt einer linie. Die endkappen sind der eigentliche…, Kreis-umriss mit dem ECHTEN radius des koerpers auf dem linienende. Das ist der…, Wo jeder koerper waehrend des VORHERSAGE-FENSTERS entlanglaeuft. Dieselbe…

### Community 69 - "InputRouter"
Cohesion: 0.06
Nodes (29): main(), spacesim -- 2D-N-Koerper-Bahnmechanik mit spielbarem raumschiff. DAS IST DER…, Zeitskala der bahnbewegung, in sekunden -- die KLEINSTE, die ein koerper dem…, load_config(), Zentrale Konfiguration laden. ALLE spielbaren parameter stehen in config.json;…, InputRouter, Zeit auf der vorhersagelinie, die dem mauszeiger am naechsten liegt. Der…, +'/'-' verstellen den MANUELLEN faktor, nicht die laenge direkt. Sonst wuerde… (+21 more)

### Community 70 - "ManeuverNode"
Cohesion: 0.17
Nodes (4): ManeuverNode, Das schubprofil dieses knotens -- IMMER frisch gerechnet., Ein geplanter impuls: wann, und wieviel prograde/normal., Der manoeverplan: knoten, reihenfolge, und die orbitale basis. Was hier…

### Community 71 - "ui_hud_test.py"
Cohesion: 0.14
Nodes (8): cursor_at(), Cyc, frame(), _open_progress(), Regressionstest des spieler-HUDs (Phase 4). Drei ebenen, bewusst getrennt: 1.…, Ganzzahlige fensterkoordinate auf einem kompasswinkel am ring. GANZZAHLIG, weil…, Bildet NavballCluster._strip nach: drei stufen, wert gewinnt., _strip_width()

### Community 72 - "devui.py"
Cohesion: 0.18
Nodes (13): _checkbox(), draw_dev_panels(), _draw_timing_graphs(), _fmt_si(), _nice_ceiling(), Dear ImGui entwickler-oberflaeche (moderngl-nativ). Bewusst NUR fuer…, Zeichnet die entwickler-panels. `ctxobj` ist ein DevContext., Kompakte SI-darstellung fuer die readouts. (+5 more)

### Community 73 - "SegmentBar"
Cohesion: 0.25
Nodes (4): Platz, den der notch-tab AUSSERHALB des rahmens braucht. Er wird in measure()…, Die flaeche des eigentlichen rahmens, ohne das tab-band., Waagerechte zellenleiste mit sich ausschliessenden optionen. Traegt den…, SegmentBar

### Community 74 - "background_test.py"
Cohesion: 0.16
Nodes (9): pan_drift(), pattern_origin_px(), Regressionstest fuer background.py -- sternenfeld und dreiecksgitter. Reine…, Sterndrift, wenn die kamera um `screens` bildbreiten schwenkt., `path` abfahren, jeweils die anker zurueckgeben., run(), _ss(), visible() (+1 more)

### Community 75 - "body_icon_test.py"
Cohesion: 0.16
Nodes (11): edge_count(), FakeBody, ink(), line_dips(), object, Die POSITIONS-MARKE der koerper -- rechnung und echte pixel. Das icon hatte…, Ein koerper, wie er aus solar_system.json faellt., Gesamte helligkeit -- das mass, an dem sich kriechen zeigt. (+3 more)

### Community 77 - "Readout"
Cohesion: 0.20
Nodes (5): Label, Text-widgets: einfaches label und die zweispaltige messwert-zeile., Beschriftung links, wert rechts -- die standardzeile der info-panels. Der wert…, Text. Die groesse folgt standardmaessig dem inhalt (size=(None, None)). text…, Readout

### Community 78 - "orbit_lines.py"
Cohesion: 0.18
Nodes (12): _constant_track(), frame_origin_body(), future_track(), future_tracks(), polyline_stride(), Bahn-linien der himmelskoerper -- reine geometrie, kein GL. Kurven je koerper…, Der koerper, der im ursprung dieses plot-frames sitzt, oder None. Er bekommt…, Wo `body` zu den zeiten `times` stehen wird, (k, 2) in weltkoordinaten. Duenner… (+4 more)

### Community 79 - "Ausgangslage im April 2026"
Cohesion: 0.15
Nodes (13): Ausgangslage im April 2026, Beispiel, Bereits vorhandene Funktionen, Performance-Problem, Problem, Problem, Probleme, Probleme im April-Stand (+5 more)

### Community 80 - "HorizonPolicy"
Cohesion: 0.15
Nodes (7): Tastenbelegung und die klick-geste. Die vorfahrt (custom-UI -> ImGui -> welt)…, HorizonPolicy, Haelt den horizont-zustand und setzt ihn am predictor durch. Der manuelle…, +' / '-': den MANUELLEN faktor verstellen, nicht die laenge direkt. Sonst…, Horizont neu setzen, wenn sich basis*manuell*raffung geaendert hat., Horizont-faktor aus der raffung -- zweierpotenz, gedeckelt. `rate` ist die…, warp_length_mult()

### Community 81 - "maneuver_hud_test.py"
Cohesion: 0.17
Nodes (9): click(), frame(), key(), press(), Die vier manoever-plaettchen im navball-raster und der ziehgriff. Fuenf ebenen:…, Ein voller HUD-frame -- MIT zeichnen. Das `root.render()` ist keine zierde:…, Einen vollstaendigen klick auf den widget-baum schicken., Zeiger hin, dann drueckn -- ueber die ECHTE ereigniskette. Die bewegung gehoert… (+1 more)

### Community 82 - "Hud"
Cohesion: 0.19
Nodes (5): Hud, Einmal pro frame VOR ui_root.begin_frame() aufrufen. Erst abtasten, dann…, Naechstliegende zeitraffer-stufe zum AKTUELLEN sim_dt. Zurueckgelesen statt…, Ist diese stufe momentan erlaubt? (bahn-zeitskala, siehe…, Baut den widget-baum und haelt ihn pro frame aktuell. Besitzt die telemetrie…

### Community 83 - "TextRenderer"
Cohesion: 0.26
Nodes (5): Sucht die schriftdateien EINMAL, je familie und strichstaerke. Reihenfolge:…, Rastert alle rollen in der aktuellen skalierten pixelgroesse NEU. Fertige…, Font-verwaltung, label-textur-cache und texturiertes blitten., Eigene texquad-pipeline. Bewusst NICHT die des Renderers geteilt: die UI-…, TextRenderer

### Community 84 - ".sample"
Cohesion: 0.18
Nodes (5): Alles, was der MANEUVER-block je frame zeigt -- einmal gelesen. Brenndauer und…, Obergrenze der raffung aus der bahn-zeitskala., mu und spezifische bahnenergie -- die grundlage von vis-viva., Geschwindigkeit eines koerpers -- auch eines skriptgefuehrten. ACHTUNG, das ist…, v . r_dach gegen den bezugskoerper -- steig- bzw. sinkrate. Die geschwindigkeit…

### Community 85 - "advance"
Cohesion: 0.18
Nodes (11): advance(), _advance_pattern(), _at_periapsis(), _fires(), _measure(), _place_on_ellipse(), probe(), Wie oft fordert die schub-erkennung in `frames` bildern eine neuberechnung an?… (+3 more)

### Community 86 - "OrbitLineSet"
Cohesion: 0.16
Nodes (4): approach_alpha(), OrbitLineSet, Deckkraft aus der dichtesten annaeherung, in SOI-vielfachen. `miss <= soi_full…, Deckkraft und zukunfts-spur aller koerper, ueber die frames gehalten. Teilt die…

### Community 87 - "Entwicklung nach dem April-Stand"
Cohesion: 0.17
Nodes (12): Entwicklung nach dem April-Stand, Orbit- und Bahndaten, Reference-Frame-System, Schiffskontrolle, Verbesserung, Warum ist das wichtig?, Warum war das wichtig?, Weltphysik (+4 more)

### Community 88 - "._render_tracked"
Cohesion: 0.33
Nodes (3): Breite der breitesten ziffer dieser schrift, einmal gemessen., Rastert text, bei bedarf mit LAUFWEITE (letter-spacing). Die oberflaeche sperrt…, Ein render-aufruf, mit alphakanal auch im hart gerasterten fall. Ohne…

### Community 90 - "frame_affine_at"
Cohesion: 0.22
Nodes (8): _cubic_4pt(), frame_affine_at(), FrameAffineTable, Die starre transformation eines plot-frames ueber ein ZEITFENSTER. Alle koerper…, Knotenzahl aus dem, was sich im fenster WIRKLICH dreht. Zwei winkel, und beide…, (fx, fy) fuer beliebige zeiten im fenster, vollstaendig in numpy. Das gitter…, Die starre transformation des frames zur zeit `t`. Gibt `(r00, r01, r10, r11,…, Kubik DURCH ALLE VIER knoten (Lagrange), ausgewertet auf [p1, p2]. Wortgleich…

### Community 91 - "BodyCentredNonRotatingReferenceFrame"
Cohesion: 0.15
Nodes (7): BodyCentredNonRotatingReferenceFrame, linear_reference(), make_frame(), _origin_run(), Der ursprung eines plot-rahmens wird KUBISCH interpoliert, nicht linear.…, (groesster abstand zum exakten wert, groesste bewegung je frame) in m., Der ALTE weg: sehne zwischen den beiden umgebenden knoten.

### Community 92 - "HorizonSlider"
Cohesion: 0.20
Nodes (3): HorizonSlider, Die GEZEICHNETE horizontlaenge, aus dem predictor zurueckgelesen.…, Waagerechter raten-regler mit federrueckstellung in die mitte.

### Community 93 - "devui_timing_test.py"
Cohesion: 0.18
Nodes (5): FakePredictor, FakeRenderer, push_ramp(), Regressionstest fuer die timing-ringpuffer der entwickler-oberflaeche. Prueft…, n proben, in denen jede serie einen eindeutigen wert bekommt.

### Community 94 - "maneuver_render_test.py"
Cohesion: 0.18
Nodes (6): Die fuenfte zeichen-domaene: der manoeverplan. Sie zeichnet drei dinge und…, _empty(), max_deviation_px(), Die manoever-zeichnung auf einem echten GL-context. Geprueft wird NICHT, wie es…, Groesster abstand eines WAHREN kurvenpunkts vom gezeichneten zug. Punkt-zu-…, Leer, egal ob None oder ein array der laenge 0. Die schirmkurve ist seit der…

### Community 95 - "_burn_arc_numba"
Cohesion: 0.16
Nodes (13): _burn_arc_numba(), _profile_accel_numba(), Der brennbogen: schwerkraft plus konstant gerichteter schub.…, RK4 ueber den brennbogen: schwerkraft plus konstant gerichteter schub. FESTE…, _no_body_memo(), Die @njit-kerne der bahnvorhersage -- reine funktionen, kein `self`.…, Leerer notizblock fuer aufrufer, bei denen sich das merken nicht lohnt.…, Punkte auf POINT_COLUMNS bringen; fehlende tangenten werden NaN. (+5 more)

### Community 97 - "._recompute"
Cohesion: 0.25
Nodes (7): closest_approach(), Radius der einflusssphaere, `a * (m / m_elter)^0.4`, oder None. Der massstab,…, Kleinster abstand zwischen schiff und koerper ZUR GLEICHEN ZEIT. Alle drei…, Ob dieser koerper in DIESEM bild ueberhaupt eine linie bekommt. Der gemeinsame…, Wie VIEL der linie gezeichnet wird, 0..1 -- neben der deckkraft. Zweites,…, reveal_fraction(), soi_radius()

### Community 98 - ".update"
Cohesion: 0.18
Nodes (10): fold(), fold_spans(), Die kleinste GITTERTRANSLATION, in der `(x, y)` gefaltet werden darf. Der anker…, `value` in `(-span/2, +span/2]` zurueckholen., Die geschwindigkeit des verfolgten koerpers, aus seiner POSITION.…, Einen bildschritt weiterdrehen. `cam_world_xy` ist die kameraposition in…, jump(), Geradeausflug; gibt (layer, ZURUECKGELEGTE strecke in px) zurueck. ACHTUNG:… (+2 more)

### Community 99 - ".levels"
Cohesion: 0.20
Nodes (6): Level, Wie GLSL `smoothstep` -- C1-stetig, damit nichts poppt., Eine sichtbare gitter-dekade., Die sichtbaren dekaden, hellste zuerst, hoechstens `MAX_LEVELS`. Eine dekade…, Gitterphase in `(a, b)`-koordinaten, modulo 2. `a = x*sqrt(3)/ws`, `b = y/ws`;…, _smoothstep()

### Community 100 - "Window"
Cohesion: 0.25
Nodes (4): Fenster, GL-context und der frame-takt. Einmalige einrichtung, die mit der…, Das SDL/OpenGL-fenster und die uhr, die den frame-takt vorgibt., Einen frame abwarten und das ECHTE delta in sekunden liefern., Window

### Community 101 - "renderer.py"
Cohesion: 0.08
Nodes (22): Zentrale konfiguration: `config.json` ->…, BackgroundDrawMixin, Das zeichnen der hintergrund-ebene. Die geometrie (sterne, gitter) rechnet…, Sternenfeld und dekaden-gitter. NICHTS HIER SKALIERT MIT camera.scale -- der…, Laedt die sterntabelle in den instanz-VBO, wenn die dichte wechselt. Der puffer…, Zeichnet sternenfeld und gitter -- die unterste schicht. Laeuft VOR allem…, family_normals(), lattice_vertices() (+14 more)

### Community 102 - "_Rng"
Cohesion: 0.22
Nodes (7): build_icon(), IconCells, object, Das gepackte zellfeld einer marke., Eine marke aus einem seed. Rein rechnerisch, kein GL, kein pygame., Der `rng`-generator des mockups (xorshift-multiply, 32 bit)., _Rng

### Community 103 - "09.05.2026 April-Mai Vergleich.md"
Cohesion: 0.22
Nodes (8): Aktuelle Probleme, Aktueller Entwicklungsstand, Inhaltsverzeichnis, Integration, Noch bestehendes Problem, Performance, Rendering-Culling, Spielstatus

### Community 104 - "Predictor und Integratoren"
Cohesion: 0.22
Nodes (9): ASPI-Idee, Ergebnis, Gedanke, Predictor und Integratoren, Problem, Rolling-System, Vorteile, Wechsel auf RK45 (+1 more)

### Community 105 - "11.04.2026 Finales Update für die Osterferien:"
Cohesion: 0.22
Nodes (8): 11.04.2026 Finales Update für die Osterferien:, BIGGEST CHANGES:, Evaluation / Probleme, OSTERUPDATE: Was gehört alles zu diesem Update?, Poliastro & Astropy, Reference Körper:, Small Changes:, Was kann behoben werden und wie?

### Community 106 - "._origin_xy_arrays"
Cohesion: 0.22
Nodes (4): Wie _prepare_cache + to_this_frame_xy, nur fuer ein ganzes feld. Winkel und…, Wie _prepare_cache + to_this_frame_xy, nur fuer ein ganzes feld. Winkel und…, Kubik DURCH ALLE VIER knoten (Lagrange), ausgewertet auf [p1, p2]. Rein…, Stapelfassung von _body_world_position_at_time. None = nicht moeglich. Nutzt…

### Community 107 - "Hintergrund-Ebene — Sternenfeld und rekursives Dreiecksgitter"
Cohesion: 0.12
Nodes (15): `background_ms` ist kein Kostenposten, sondern ein Stau, Das atmende Feld, Der Frühausstieg trägt den ganzen Shader, Die eine Regel, die alles andere erklärt, Die Faltung trägt ein √3, Die Geschwindigkeitsgrenze, Die Gittergeometrie — und der Fehler, den das Mockup hatte, Ein Schaltersatz, drei Orte (+7 more)

### Community 109 - "_w17_frame"
Cohesion: 0.22
Nodes (9): arc_length(), predictor_warp_length_mult() aus test.py., step_simulation() aus test.py., Ein frame der hauptschleife -- nur der zeitraffer-relevante teil., Auf from_rate einschwingen, dann auf to_rate wechseln. Rueckgabe: (ms auf dem…, _w17_frame(), _w17_mult(), _w17_step() (+1 more)

### Community 110 - "._build_draw_track"
Cohesion: 0.25
Nodes (7): chain_periods(), orbital_period(), Umlaufzeit des koerpers um seinen elter, `2*pi*sqrt(a^3/mu)`, oder None. Fuer…, `{id(glied): umlaufzeit}` fuer jedes glied der elternkette von `body`. Ein…, Schnellste umlaufzeit, die in `koerper(t) - ursprung(t)` steckt. DIE UMLAUFZEIT…, Das gitter, auf dem die spur GEZEICHNET wird. Standardfall ist das gemeinsame:…, relative_min_period()

### Community 111 - "Renderer — draw path, GL state, shaders, screen conventions"
Cohesion: 0.15
Nodes (12): Body names are driven by SELECTION, not by zoom, Frame timing — read the labels literally, `render/gl/`, `render/renderer.py`, Renderer — draw path, GL state, shaders, screen conventions, Resize and dynamic resolution, Text must be pixel-snapped and must not go through FXAA, The background layer draws first, and absorbs the frame's stall (+4 more)

### Community 112 - "world_kernels.py"
Cohesion: 0.23
Nodes (12): _acceleration_at(), advance_dynamics(), _body_pos_at_time(), Numba-fassung des welt-integrators. WORTGLEICH zur python-fassung.…, Nachbau von world.acceleration_at. Reihenfolge = koerper-reihenfolge., world._rkn4_step_body_state., world._verlet_step_body_state (Stoermer-Verlet, KDK)., world.update_dynamics -- die aeussere schrittsteuerung, 1:1. `bx`/`by` werden… (+4 more)

### Community 113 - "_earth_velocity"
Cohesion: 0.20
Nodes (6): _earth_velocity(), _first_chord_direction(), _first_index(), Wie weit vorn sitzt die erste fahne, in punkten?, Die alte regel: erste sehne ueber 1e-12 m, ohne mindestlaenge., Skriptgefuehrte koerper haben velocity == 0 -- zentraldifferenz noetig.

### Community 114 - "PlanetStyle"
Cohesion: 0.29
Nodes (3): PlanetStyle, object, Fertige zeichnung eines koerpers im einheitskreis. `tri` sind die flaechen…

### Community 115 - "DevContext"
Cohesion: 0.29
Nodes (5): DevContext, _ms(), Robuste ms-zahl. Fehlende/kaputte werte werden zu 0, nicht zu NaN. Ein NaN in…, Sammelt die objekte, die die entwickler-panels verstellen duerfen. Bewusst ein…, Eine probe der vier zeitreihen. Je frame einmal, aus der hauptschleife. BEWUSST…

### Community 118 - "ui_render_test.py"
Cohesion: 0.19
Nodes (10): corner_probe(), draw_frame(), Regressionstest der UI-zeichenschicht -- gegen echte PIXEL. Die lehre aus Phase…, Zeichnet ein rechteck bei (100, 100) und tastet seine linke obere ecke ab., Framebuffer als (h, w, 3) uint8 in TOP-DOWN reihenfolge. viewport= wird…, read_pixels(), cut_corners(), Eckwert-tupel, bei dem NUR die gewaehlten ecken gefast sind. Die vorlage fast… (+2 more)

### Community 119 - "_imul"
Cohesion: 0.40
Nodes (5): Der seed einer marke: `style_seed`, sonst der name -- plus ein globaler…, seed_for(), _imul(), Stabiler 32-bit-seed aus dem koerpernamen. Damit bekommt jeder koerper ohne…, seed_from_name()

### Community 120 - "Rendering der Predictor-Linie"
Cohesion: 0.33
Nodes (6): Problem: zu viele Punkte im Bildschirmraum, Rendering der Predictor-Linie, Verbesserung: begrenzte Punktanzahl beim Rendering, Verbesserung: sichtbare Abschnitte statt einer Punktliste, Verbesserung: Zwischenspeicherung der gerenderten Linie, Weiterhin bestehendes Problem

### Community 121 - "HudPanel"
Cohesion: 0.26
Nodes (5): HudPanel, Gefaster doppelrahmen mit notch-tab auf der unterkante. Stapelt seine kinder…, Platz fuer den notch-tab UNTERHALB des rahmens. Er sitzt ausserhalb der kante,…, Fiktive inhaltsflaeche fuer die messung. Die zeilen sind (FILL, None) breit --…, Hoehe aus den kindern, breite aus der vorgabe. Haengt bewusst NICHT von…

### Community 122 - "spacesim"
Cohesion: 0.17
Nodes (11): Controls, Conventions, Git — never on your own, graphify, Invariants — don't break these, Keeping these files current, Module map, Run (+3 more)

### Community 123 - "_empty_points"
Cohesion: 0.27
Nodes (8): _find_apsis_markers_numba(), Ap/Pe-suche auf einer fertigen punktreihe. Die marker sitzen auf den EXTREMA…, _refine_apsis_numba(), _empty_points(), Leere punkteliste in der kanonischen breite., Die vorausberechnete bahnlinie des schiffs. Die REINE ZAHLENARBEIT liegt in…, Die vorausberechnete bahnlinie des schiffs. core.py zustand, integrator-guete,…, Die ausgabeseite des predictors. `get_points()` liefert die GEZEICHNETE kurve…

### Community 124 - ".orbital_speed_scale"
Cohesion: 0.33
Nodes (3): (bahntempo, kreisbahn-tempo, fluchttempo) am aktuellen ort. Der MASSSTAB der…, Nadellaenge auf [0, 1]: 0 = ruhe, ~0.71 = kreisbahn, 1 = flucht., Wo auf derselben skala die kreisbahn liegt -- immer 1/sqrt(2). Steht hier…

### Community 125 - "WarpBar"
Cohesion: 0.20
Nodes (3): Der zeitraffer -- die zellenleiste plus die laufende missionszeit. Die uhr…, Der rahmen endet ueber dem uhrstreifen., WarpBar

### Community 126 - "SnapRosette"
Cohesion: 0.33
Nodes (3): Der orientierungs-autopilot als rosette -- vier knoepfe um das schiff. Bildet…, Ein schlanker pfeil nach oben -- dieselbe silhouette wie die schiffsnase am…, SnapRosette

### Community 127 - "Test map"
Cohesion: 0.25
Nodes (7): Frames, lines, camera, Known pre-existing failures — not regressions, Manoeverknoten, Physics, Predictor and time warp, Rendering and UI, Test map

### Community 130 - "frame_project"
Cohesion: 0.50
Nodes (4): frame_project(), _frame_project_scalar(), Weltpunkte in den plot-frame, je ZEIT statt je PUNKT. Die zukunfts-spuren aller…, Punktweise rueckfallebene -- langsam, aber immer richtig.

### Community 131 - "01 Einleitung.md"
Cohesion: 0.50
Nodes (3): 1.1 Motivation, 1. Einleitung:, Abstrakt:

### Community 132 - "Quellen.md"
Cohesion: 0.50
Nodes (3): Beitragsquellen, Bildquellen, Literatur / Papers

### Community 133 - "Predictor RKN"
Cohesion: 0.50
Nodes (4): Bedeutung für den Predictor, Predictor RKN, Vorteile, Warum RKN?

### Community 135 - ".star_table"
Cohesion: 0.50
Nodes (3): build_star_table(), `(n, 7)`-float32: x, y, radius, alpha, parallaxe, funkelphase, zoomphase.…, Die sterntabelle, bei bedarf neu erzeugt.

### Community 136 - "._texture_for"
Cohesion: 0.29
Nodes (3): Textur dieser groesse besorgen -- moeglichst eine wiederverwendete. Der teure…, Verdraengte textur einsammeln statt freigeben (bis zum deckel)., Groesse des gerenderten textes in pixeln, ohne zu zeichnen.

### Community 137 - "Camera, input and the in-app controls"
Cohesion: 0.29
Nodes (6): Camera, input and the in-app controls, Controls (in-app), Input priority is custom UI → ImGui → world, Modules, Sim rate is decoupled from framerate, Zoom has a physical ceiling, not just a numeric one

### Community 139 - ".enable_epicycles"
Cohesion: 0.33
Nodes (3): konvertiert position/geschwindigkeit (relativ zum parent) in orbitale…, epizykel-modus aktivieren mit wurzel in `center`. speichert den aktuellen…, gespeicherten zustand der körper wiederherstellen und epizykel-modus…

### Community 140 - "Config and system loading"
Cohesion: 0.40
Nodes (4): Config and system loading, `config.json` — all user/player-tunable parameters, `config/loader.py`, `solar_system.json`

### Community 141 - "The player-facing HUD (`ui/`)"
Cohesion: 0.40
Nodes (4): Body names in the world are set in the house font, Conventions that bind every widget, The player-facing HUD (`ui/`), Two HUD elements read the renderer instead of the world

### Community 142 - "compass_to_screen"
Cohesion: 0.50
Nodes (4): compass_to_screen(), Ein bogen aus EINZELNEN zellen -- die anzeigeform der vorlage. Ein…, Kompasswinkel -> der winkel, den UIDraw.arc erwartet. UIDraw rechnet…, segment_arc()

### Community 146 - "Predictor"
Cohesion: 0.67
Nodes (3): Nachteile, Predictor, Vorteile

### Community 147 - "maneuver_preview_test.py"
Cohesion: 0.40
Nodes (3): radius_range(), Die manoever-vorschau: was die geplanten knoten aus der bahn machen. Die…, Kleinster und groesster abstand der linie zur Erde.

### Community 148 - "_hermite_at"
Cohesion: 0.67
Nodes (3): _hermite_at(), Die linie KUBISCH auswerten -- so zeichnet der renderer sie auch. Linear…, _walk18()

### Community 150 - "render/draw.py"
Cohesion: 0.50
Nodes (3): Zeichen-primitive: polylinien, ortho-formen, texturen, clipping. Die clipping-…, _clip_runs_numba(), Liang-Barsky ueber die GANZE polylinie, laufweise zerlegt. Wort-fuer-wort…

### Community 151 - "arc_ruler"
Cohesion: 0.50
Nodes (4): arc_ruler(), polar(), Teilung entlang eines kreisbogens, in kompassgrad. inward=True zieht die…, Kompasswinkel -> punkt in TOP-DOWN pixeln (0 = oben, im uhrzeigersinn).

### Community 174 - "Projekt Stand 08.09.2026"
Cohesion: 0.40
Nodes (4): Neuerungen bis zum letzten Commit:, Projekt Stand 08.09.2026, Repo-Struktur, Spiel starten

## Knowledge Gaps
- **176 isolated node(s):** `_Motion`, `Die eine Regel, die alles andere erklärt`, `Sterndrift — an der Eigengeschwindigkeit, nicht am Bild`, `Das atmende Feld`, `Die Geschwindigkeitsgrenze` (+171 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1241 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **34 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Predictor` connect `ConfigLoader` to `bootstrap.py`, `ComputeMixin`, `SystemLoader`, `ui_hud_test.py`, `_empty_points`, `warp_predictor_test.py`, `JobsMixin`, `maneuver_hud_test.py`, `ViewMixin`, `maneuver_preview_test.py`, `horizon_targets_test.py`, `prediction_detail_test.py`, `HoldMixin`, `maneuver_render_test.py`?**
  _High betweenness centrality (0.117) - this node is a cross-community bridge._
- **Why does `Renderer` connect `Renderer` to `SystemLoader`, `Camera`, `ShaderPipelineMixin`, `ShipDrawMixin`, `PredictionDrawMixin`, `TextMixin`, `DrawMixin`, `BodyDrawMixin`, `bootstrap.py`, `selection_camera_test.py`, `render_budget_test.py`, `prediction_detail_test.py`, `ManeuverDrawMixin`, `BackgroundLayer`, `GLDeviceMixin`, `OrbitDrawMixin`, `ui_hud_test.py`, `body_icon_test.py`, `maneuver_hud_test.py`, `maneuver_render_test.py`, `renderer.py`?**
  _High betweenness centrality (0.101) - this node is a cross-community bridge._
- **Why does `Rect` connect `Rect` to `Slider`, `layout.py`, `ui_horizon_slider_test.py`, `UIContext`, `SystemMap`, `background_test.py`, `_emit_shape`, `Widget`, `maneuver_hud_test.py`, `Stack`, `HorizonSlider`, `HudPanel`, `_ManeuverPlate`?**
  _High betweenness centrality (0.101) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `Vec2` (e.g. with `body` and `world`) actually correct?**
  _`Vec2` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `world` (e.g. with `Vec2` and `FrameController`) actually correct?**
  _`world` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `_Motion`, `Die eine Regel, die alles andere erklärt`, `Sterndrift — an der Eigengeschwindigkeit, nicht am Bild` to the rest of the system?**
  _176 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `layout.py` be split into smaller, more focused modules?**
  _Cohesion score 0.055533199195171024 - nodes in this community are weakly interconnected._