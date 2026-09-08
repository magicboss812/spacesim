# Graph Report - spacesim  (2026-09-08)

## Corpus Check
- 126 files · ~237,223 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2628 nodes · 4710 edges · 156 communities (124 shown, 28 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 89 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9e8d1db9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- predictor/core.py
- with_alpha
- SystemLoader
- Camera
- Vec2
- Stand 11.03.2026
- ._texture_for
- .apply_to_renderer
- HorizonSlider
- NavballCluster
- SystemMap
- units.py
- warp_predictor_test.py
- AttitudeRing
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
- renderer.py
- ManeuverPreview
- UIState
- BodyDrawMixin
- preview.py
- BodyBrowser
- schiffcontrol
- TimingHistory
- Telemetry
- Palette
- style.py
- Renderer
- Predictor
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
- InputRouter
- ManeuverAxesBlock
- ReferenceFrameSelector
- reference_frames.py
- BackgroundLayer
- GLDeviceMixin
- _ManeuverPlate
- Toggle
- ManeuverPlan
- ApsisTooltip
- Dropdown
- icon.py
- build_app
- Gliederung
- OrbitDrawMixin
- loop.py
- ManeuverNode
- ui_hud_test.py
- devui.py
- SegmentBar
- background_test.py
- body_icon_test.py
- TargetName
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
- _bvel24
- devui_timing_test.py
- ConfigLoader
- _fires
- compass_from_frame_direction
- ._recompute
- .update
- .levels
- Window
- body_style_test.py
- _Rng
- 09.05.2026 April-Mai Vergleich.md
- Predictor und Integratoren
- 11.04.2026 Finales Update für die Osterferien:
- ._origin_xy_arrays
- ReferenceFrame
- background.py
- _w17_frame
- mass
- VirtualBodyCentredNonRotatingReferenceFrame
- BackgroundDrawMixin
- _first_index
- PlanetStyle
- DevContext
- _hermite23
- _line_with_head
- _imul
- Rendering der Predictor-Linie
- KeplerScriptedOrbit
- .orbital_speed_scale
- ._retarget
- OrbitalElements
- Role
- frame_project
- 01 Einleitung.md
- Quellen.md
- Predictor RKN
- .star_table
- Knobs
- _advance_chunked22
- compass_to_screen
- _emit_shape
- Project-Entwicklung.md
- Integrator.md
- Predictor
- _clear_of_bodies21
- _hermite_at
- render/__init__.py
- _earth_velocity
- _first_chord_direction
- _kosten
- .adjust_node_dv
- .warp_step_allowed
- .speed_at_radius
- .text_mission_time
- .view_mode_label
- .gauge_speed
- .gauge_altitude
- .radial_fraction
- .set_thrust_level

## God Nodes (most connected - your core abstractions)
1. `Vec2` - 88 edges
2. `Widget` - 65 edges
3. `Telemetry` - 48 edges
4. `ConfigLoader` - 47 edges
5. `world` - 46 edges
6. `Camera` - 46 edges
7. `Rect` - 46 edges
8. `Renderer` - 43 edges
9. `with_alpha()` - 42 edges
10. `Predictor` - 40 edges

## Surprising Connections (you probably didn't know these)
- `body` --uses--> `Vec2`  [INFERRED]
  bodies/body.py → physics/vec.py
- `world` --uses--> `body`  [INFERRED]
  physics/world.py → bodies/body.py
- `frame()` --uses--> `body`  [INFERRED]
  tests/prediction_detail_test.py → bodies/body.py
- `soi_radius()` --indirect_call--> `mass()`  [INFERRED]
  bodies/orbit_lines.py → ui/units.py
- `FrameController` --uses--> `ReferenceFrameSelector`  [INFERRED]
  runtime/bootstrap.py → physics/reference_frames.py

## Import Cycles
- None detected.

## Communities (156 total, 28 thin omitted)

### Community 0 - "predictor/core.py"
Cohesion: 0.05
Nodes (67): _find_apsis_markers_numba(), Ap/Pe-suche auf einer fertigen punktreihe. Die marker sitzen auf den EXTREMA…, _refine_apsis_numba(), _burn_arc_numba(), _profile_accel_numba(), Der brennbogen: schwerkraft plus konstant gerichteter schub.…, RK4 ueber den brennbogen: schwerkraft plus konstant gerichteter schub. FESTE…, _empty_points() (+59 more)

### Community 1 - "with_alpha"
Cohesion: 0.05
Nodes (69): ease(), Kern der UI-schicht: rechtecke, verankerung, widget-basis, eingabe-routing.…, Framerate-unabhaengiges exponentielles easing -- dieselbe formel wie das…, Der schwebezettel an einer Ap/Pe-raute auf der bahnlinie. Die raute selbst und…, Der Attitude-Ring -- das Herzstueck des HUDs. Ein 2D-lagemesser statt einer…, Ausklappbare koerperliste zur wahl des BEZUGSKOERPERS. Bisher liess sich der…, arc_ruler(), bar_cells() (+61 more)

### Community 2 - "SystemLoader"
Cohesion: 0.08
Nodes (25): body, kepler_relative_xy(), Return this body's Kepler position at simulation time t without modifying…, Position nach `dt` auf der eigenen bahn -- exakt, siehe kepler_relative_xy().…, Bahnposition relativ zum mutterkoerper nach `dt`, EXAKT geloest. Rueckgabe `(x,…, schiff, bequemlichkeitsfunktion zum erstellen von Vec2., vec() (+17 more)

### Community 3 - "Camera"
Cohesion: 0.06
Nodes (23): Camera, Wandelt Bildschirmkoordinaten in Weltkoordinaten um., screen->welt gegen eine BELIEBIGE position/skala. Wird für das zoom-ankern…, Setzt ein Objekt zur Verfolgung (SOFORT, ohne anflug)., Körper, auf den `Home` die ansicht zurückholt (das schiff). Die kamera kennt…, Fährt die ansicht GEGLÄTTET auf `body` und heftet sie dort an. Der trick ist,…, Taste Home: geglättet zurück zum heimatkörper (dem schiff)., Beendet die Objektverfolgung. Die aktuelle ansicht wird als freies ziel… (+15 more)

### Community 4 - "Vec2"
Cohesion: 0.05
Nodes (23): erstellt eine kopie dieses vektors., in (x, y)-tuple umwandeln., erstellt Vec2 aus tuple oder liste., quadratische distanz zu einem anderen vektor., vektor auf null zurücksetzen., vektor-komponenten setzen., skalare multiplikation (self * scalar)., in-place vektor-addition. (+15 more)

### Community 5 - "Stand 11.03.2026"
Cohesion: 0.05
Nodes (40): bodies.py, camera.py, loader.py, main.py, predictor_mp.py, predictor.py, rendering.py, schiff.py (+32 more)

### Community 6 - "._texture_for"
Cohesion: 0.14
Nodes (7): Rollengroesse -> pixelgroesse, fuer die pixelschrift GERASTET. SB Liquid ist…, Die pixelschrift wird OHNE kantenglaettung gerastert. Gemessen ueber…, Die instrumentenschrift setzt ZIFFERN AUF FESTER BREITE. SB Liquid ist nicht…, Laufweite dieser rolle in pixeln (em-wert * schriftgroesse)., Textur dieser groesse besorgen -- moeglichst eine wiederverwendete. Der teure…, Verdraengte textur einsammeln statt freigeben (bis zum deckel)., Groesse des gerenderten textes in pixeln, ohne zu zeichnen.

### Community 7 - ".apply_to_renderer"
Cohesion: 0.08
Nodes (24): Any, _float_list(), Liest die konfiguration ein und gibt sie zurueck. `utf-8-sig` statt `utf-8`:…, Gibt einen abschnitt (z. B. 'camera') als dict zurueck, notfalls leer., Liest einen wert ueber einen punkt-pfad, z. B. get('camera.zoom_factor').…, Wandelt `value` mit `caster` um; bei unpassendem wert warnung + default. so…, Liest einen kommazahl-parameter., Liest einen ganzzahl-parameter. (+16 more)

### Community 8 - "HorizonSlider"
Cohesion: 0.07
Nodes (16): _Ctx, _Dummy, make(), _Motion, _Predictor, _PredLen, Kopftest fuer ui/widgets/rate_slider.py -- die federphysik und die…, Das Nötigste, das der Widget-Code berührt. (+8 more)

### Community 9 - "NavballCluster"
Cohesion: 0.09
Nodes (15): NavballCluster, Punkt auf dem linken bogen -> schubstufe, oder None. Getroffen wird ein RING-…, Das rad stellt den schub -- aber NUR ueber dem bogen und dem streifen. Ueberall…, Das ORB- (links) bzw. ALT-kaestchen (rechts)., Der THR- (links) bzw. V/S-streifen (rechts) darunter., Der ORBITAL.INFO-block am unteren rand. Er ist nur um eine halbe flankenbreite…, Eine der beiden messwert-flanken. Aufbau der vorlage: winzige gesperrte…, Der schmale streifen unter einer flanke. (+7 more)

### Community 10 - "SystemMap"
Cohesion: 0.09
Nodes (20): _body_color(), _log_spread(), ``body.color`` (0..255 aus dem JSON) -> zeichenfarbe. Unveraendert uebernommen,…, Werte auf [low, high] verteilen -- logarithmisch, optional nach rang.…, Der ECHTE bahnwinkel des koerpers um seinen mutterkoerper. Aus den momentanen…, Die karte als widget: kachel, ausfahren, treffer, mond-ansicht., (zentralkoerper, planeten, monde je planet), einmal aufgebaut. Die koerperliste…, Groesse aus dem ausfahr-fortschritt, nicht aus dem schalter. Zwischen kachel… (+12 more)

### Community 11 - "units.py"
Cohesion: 0.08
Nodes (33): Regressionstest fuer ui/units.py. Reine funktionen -- laeuft ohne fenster, ohne…, altitude(), angle(), countdown(), countdown_compact(), delta_v(), distance(), duration() (+25 more)

### Community 12 - "warp_predictor_test.py"
Cohesion: 0.14
Nodes (6): _advance_pattern(), _at_periapsis(), _leo_world(), _line(), _long_horizon_points(), Regressionen fuer horizont, welt-integrator und zeitraffer-halt. Alle drei…

### Community 13 - "AttitudeRing"
Cohesion: 0.09
Nodes (13): AttitudeRing, _polar(), RUND, nicht quadratisch. Das widget ist ein quadrat, der ring darin ein kreis…, Loslassen gibt das schiff SOFORT wieder frei. Ohne das blieb _manual_heading…, Das schwache polargitter auf der ringflaeche. Es macht die flaeche zu einem…, Die vier himmelsrichtungen, immer aufrecht. Eine richtung wird WEGGELASSEN,…, Die vier bahnsymbole als VEKTOREN, nicht als schriftzeichen. Der entwurf setzt…, Nur noch die kurs-plakette, mittig als nabe des rings. Der GESCHWINDIGKEITSWERT… (+5 more)

### Community 14 - "ShaderPipelineMixin"
Cohesion: 0.10
Nodes (16): Die OpenGL-pipelines des Renderers und der GL-zustandscache. Die GLSL-quellen…, Programm der positions-marke. Teilt sich das statische einheits-quad mit der…, Shader uebersetzen, VAOs/VBOs anlegen, GL-zustand cachen. Die…, Programme fuer die vektor-zeichnung der koerper. Anders als die uebrigen…, Texturierte quads (labels, HUD) in der ortho-konvention (y nach oben)., Hintergrund-ebene: vollbild-quad (gitter) + punkt-sprites (sterne). Beide…, Erstellt wiederverwendbare puffer, programme und VAOs für kritische render-…, Geteilter dynamischer vertex-puffer für polylines und ortho-geometrie. (+8 more)

### Community 15 - "world"
Cohesion: 0.09
Nodes (14): Decke fuer die integrator-schrittweite aus der raffung ableiten. Die kosten von…, Tatsaechlich benutzte decke -- nie kleiner als die konfigurierte., Rueckt die welt um `sim_seconds` vor, aufgeteilt in stuecke. Stand frueher als…, Return the body's world position at a given simulation time. For now: - if the…, One explicit RKN4-style step for r'' = a(r, t). p0: initial position v0:…, Störmer-Verlet (KDK leapfrog) — 2nd-order symplectic RKN., Adaptive embedded RKN-style step by step-doubling. Compares: - one full step h…, konvertiert position/geschwindigkeit (relativ zum parent) in orbitale… (+6 more)

### Community 16 - "Widget"
Cohesion: 0.09
Nodes (10): Basisklasse. Haelt verankerung, groesse, kinder und hover-zustand. Groessen und…, Eigengroesse in PIXELN fuer size-komponenten, die None sind. Ueberschreiben, wo…, Loest die groessen-angabe auf. Pro achse sind drei angaben erlaubt: eine zahl…, Effektive groesse in pixeln: explizite vorgabe, sonst measure(). Container…, Berechnet self.rect aus anker, abstand und groesse., Flaeche, in der kinder verankert werden. Panels ziehen hier ihr padding ab., Eigene darstellung. Basisklasse zeichnet nichts., Trefferflaeche. Bekommt ctx, weil sie nicht immer gleich self.rect ist -- ein… (+2 more)

### Community 17 - "UIRoot"
Cohesion: 0.08
Nodes (16): corner_probe(), draw_frame(), Regressionstest der UI-zeichenschicht -- gegen echte PIXEL. Die lehre aus Phase…, Zeichnet ein rechteck bei (100, 100) und tastet seine linke obere ecke ab., Framebuffer als (h, w, 3) uint8 in TOP-DOWN reihenfolge. viewport= wird…, read_pixels(), Wurzel des widget-baums. Verteilt eingaben und meldet, ob sie verbraucht…, Solange ein widget gedrueckt wird, gehoert die maus ihm -- auch wenn der zeiger… (+8 more)

### Community 18 - "ViewMixin"
Cohesion: 0.09
Nodes (11): Von der punkteliste abgeleitete zwischenergebnisse verwerfen. soft=True heisst:…, Feinste punktdichte, die den HORIZONT noch traegt -- oder None. Die kernel…, Was HERAUSKOMMT: punkte, Ap/Pe-marker, laenge, abstand, zoom. Horizont…, Wie viel von der gerechneten kurve GEZEICHNET wird, in metern. None oder >= der…, Zahl der zu zeichnenden fuehrenden punkte, oder None fuer alle. Ueber den…, Die zu ZEICHNENDE kurve (siehe set_display_length). Der ausschnitt wird gemerkt…, Traegt diese punkteliste brauchbare geschwindigkeits-spalten? Alles oder…, Apoapsis/Periapsis-Marker der aktuellen Prädiktionslinie. Rückgabe: ndarray (m,… (+3 more)

### Community 19 - "ShipDrawMixin"
Cohesion: 0.08
Nodes (15): Tie the ship nose to the drawn orbital vector for the latched snap. Computes…, Debug overlay: always draws prograde (green) + normal-inward (magenta).…, Env-guarded (SPACESIM_DEBUG_ORIENT=1) screen-space angle report. Prints, in one…, Das schiff: sprite, pfeil, fahne, schubvektor und die orientierung.…, Returns the ship's apparent speed in the active plotting frame. This respects…, Massstabs-faktor des schiffs fuer die aktuelle zoomstufe. 1.0 bei…, Gezeichnete schiffslaenge in echten bildschirm-pixeln. Basislaenge (design-…, Halbe hoehe der gezeichneten schiffs-grafik in bildschirm-pixeln. Bezugsgroesse… (+7 more)

### Community 20 - "_BodyEphemerisMixin"
Cohesion: 0.16
Nodes (14): _body_arg_periapsis(), _body_true_anomaly(), _BodyEphemerisMixin, _build_kepler_elements(), _has_scripted_orbit_data(), _mean_anomaly_from_true(), _orbit_model_from_body(), Vektorisierte fassung von _relative_position_to_parent_at_time. (+6 more)

### Community 21 - "PredictionDrawMixin"
Cohesion: 0.12
Nodes (8): PredictionDrawMixin, Alle stichproben-punkte in EINEM rutsch projizieren. Gibt ``(None,…, Erlaubte abweichung der gezeichneten linie -- in metern und pixeln. Der…, Kubische zwischenpunkte setzen -- nur sichtbar, nur so fein wie noetig.…, Die vorhersagelinie: abtastung, Hermite-verfeinerung, Ap/Pe-marker. EINE…, Das ROH-scan-budget dorthin legen, wo die linie im BILD liegt.…, Gleichmaessige stichprobe der rohpunkte -- GEMERKT, nicht neu gebaut. Das…, Zeichnet apoapsis/periapsis-marker des predictors auf die linie. Marker kommen…

### Community 22 - "TextMixin"
Cohesion: 0.11
Nodes (13): Schriften und der label-textur-cache des Renderers. Das spieler-HUD hat mit…, Setzt die DESIGN-schriftgrößen und baut die fonts neu auf., Benutzer-skalenfaktor (multiplikativ auf die automatische skala)., Beschriftungs-textur besorgen -- moeglichst eine wiederverwendete. Wie in…, Schriften, der label-textur-cache und getoentes blitten. UI-groessen sind…, Rastert eine beschriftung -- bei bedarf HART und GESPERRT. Zwei zugestaendnisse…, Leitet ui_scale aus der fensterhöhe ab. Gibt True bei änderung zurück. Skaliert…, Text an TOP-DOWN koordinaten zeichnen (x = links, y = oberkante). Nimmt dem… (+5 more)

### Community 23 - "Rect"
Cohesion: 0.11
Nodes (7): Achsenparalleles rechteck in top-down pixeln., Rect, HudPanel, Gefaster doppelrahmen mit notch-tab auf der unterkante. Stapelt seine kinder…, Platz fuer den notch-tab UNTERHALB des rahmens. Er sitzt ausserhalb der kante,…, Fiktive inhaltsflaeche fuer die messung. Die zeilen sind (FILL, None) breit --…, Hoehe aus den kindern, breite aus der vorgabe. Haengt bewusst NICHT von…

### Community 24 - "UIDraw"
Cohesion: 0.10
Nodes (12): Zeichen-primitive der UI-schicht, alle auf EINEM SDF-shader.…, Abgerundetes rechteck. (x, y) = obere linke ecke, top-down. radius: skalar oder…, Kreis um (cx, cy), top-down., Kreisring. Umgesetzt als kreis OHNE fuellung mit rahmen der gewuenschten…, Kreisbogen. Winkel gegen den uhrzeigersinn, 0 = nach rechts., Beliebig gedrehte linie -- ein um ihre achse rotiertes rechteck. cap='round'…, Trennlinie. Achsenparallel, deshalb ohne rotation und damit ohne rasterungs-…, Instanz in den stapel legen -- gezeichnet wird erst in flush(). Frueher war… (+4 more)

### Community 25 - "DrawMixin"
Cohesion: 0.10
Nodes (13): DrawMixin, Liang-Barsky clipping for screen-space line segments. Returns (cx0, cy0, cx1,…, Die zeichen-primitive: linien, ortho-formen, texturen, clipping. Alles hier…, Converts one logical predictor polyline into multiple visible screen-space…, Lädt ein (N,2)-float32-array in den geteilten dynamischen VBO. orphan()…, Kernel-weg von `_build_clipped_polyline_runs`. Ein numba-aufruf statt einer…, Zeichnet eine textur als quad in der ortho-konvention (y nach oben). Ersatz für…, Zeichnet eine bildschirm-space polyline (top-down-konvention) via GLSL+VBO. (+5 more)

### Community 26 - "ManeuverExecutor"
Cohesion: 0.10
Nodes (9): ManeuverExecutor, Den naechsten knoten scharfschalten. False, wenn das nicht geht., Von der hauptschleife gerufen, sobald der spieler selbst steuert. Handeingabe…, Die schrittklemme fuer eine gegebene sim-zeit (testbar ohne welt)., Wie weit die welt in DIESEM frame hoechstens vorruecken darf. Scharf:…, Muss die raffung jetzt auf echtzeit herunter? Ja, sobald die zuendung naeher…, VOR `world.step(sim_seconds)` aufrufen. Legt das delta-v des bevorstehenden…, Scharfschalten, ausrichten, zuenden, brennen, aufraeumen. (+1 more)

### Community 27 - "HoldMixin"
Cohesion: 0.10
Nodes (11): HoldMixin, Der halt, das verbrauchen der kurve und das umschalten auf einen neuen bahnast.…, Kurve VERBRAUCHEN statt starr verschieben. Rueckgabe: die zahl der vorn…, Setzt den kurvenanfang auf das schiff. DER REGELFALL IST DAS VERBRAUCHEN, NICHT…, Gibt die Anzahl der Einträge in `new_points` zurück, die sich von `old_points`…, Zeitraffer-halt ein/aus. Ausschalten erzwingt eine neuberechnung. Die beiden…, Die gehaltene kurve ist ueberholt (schub, rahmenwechsel, ...). `soft=True`…, Kurve VERBRAUCHEN statt neu rechnen. True = frame ist erledigt. WARUM. Ohne… (+3 more)

### Community 28 - "renderer.py"
Cohesion: 0.13
Nodes (15): Zeichen-primitive: polylinien, ortho-formen, texturen, clipping. Die clipping-…, _clip_runs_numba(), _compact_min_step_numba(), _densify_numba(), _max_gap_refine_numba(), Die Numba-fassungen der reinen zahlenschleifen im linien-zeichenweg. Min-step-…, Liang-Barsky ueber die GANZE polylinie, laufweise zerlegt. Wort-fuer-wort…, Zu weit auseinanderliegende RDP-punkte wieder auffuellen. Dieselbe schleife wie… (+7 more)

### Community 29 - "ManeuverPreview"
Cohesion: 0.11
Nodes (10): _Done, ManeuverPreview, Die kette, ihre marker, und der weg, auf dem sie NEBENHER laeuft. NEBENLAEUFIG,…, Reichweite setzen. Der naechste neuaufbau uebernimmt sie., Ein fertiges ergebnis einwechseln. True, wenn eines kam., Auf einen laufenden auftrag warten und ihn einwechseln. Nur fuer tests und…, Nur rechnen, wenn sich etwas bewegt hat. True, wenn gerechnet wurde.…, SYNCHRON rechnen und sofort einwechseln (tests, screenshots). (+2 more)

### Community 30 - "UIState"
Cohesion: 0.11
Nodes (8): Setzt die auswahl. Gibt True zurueck, wenn sie sich geaendert hat. LOEST…, Overlay direkt setzen (HUD-knopf), statt zu kippen (taste T)., Die drei modi der HUD-rahmenwahl in einem schritt. 'surface' -> mitrotierender…, Aktueller modus als index fuer die HUD-rahmenwahl., Erzwingt ein neuanwenden ohne aenderung (start, system-neuladen)., Bezugsrahmen-auswahl, referenzkoerper und overlay-schalter., Zweiter koerper fuer den body-direction-rahmen. Bevorzugt den mutterkoerper des…, UIState

### Community 31 - "BodyDrawMixin"
Cohesion: 0.05
Nodes (27): BodyDrawMixin, Positions-marke eines körpers, konstanter bildschirmgröße. `radius` ist der…, Die koerper: scheibe, prozedurale marke, vektor-look, beschriftung,…, Die spanne der PHYSISCHEN koerper-radien im geladenen system. Einmal je frame…, 0..1: wo dieser koerper-radius innerhalb der GELADENEN spanne liegt. LOG-…, Der GEZEICHNETE radius der marke -- ein je koerper KONSTANTER wert aus seinem…, Deckkraft der marke bei diesem echten bildschirmradius. 1.0 unterhalb der…, Zeichnet einen körper als shader-gesteuertes quad (scheibe + optional… (+19 more)

### Community 32 - "preview.py"
Cohesion: 0.14
Nodes (19): Der autopilot, der einen knoten wirklich fliegt. IDLE --arm()--> ARMED…, Manoeverknoten: planen, vorschau zeichnen, automatisch brennen. profile.py das…, burn_direction_world(), orbital_basis(), Manoeverknoten und der plan, der sie haelt. EIN KNOTEN SPEICHERT ZWEI ZAHLEN…, Prograde und einwaerts-normal am ort des knotens, in WELTkoordinaten. `rel_*`…, Weltrichtung und betrag des geplanten schubs. Zurueck kommt `(dx, dy,…, body_state_at() (+11 more)

### Community 33 - "BodyBrowser"
Cohesion: 0.13
Nodes (10): Sich selbst und alle nachkommen, eltern zuerst., _body_dot_color(), BodyBrowser, build_hierarchy(), Symbolknopf plus die dahinter liegende koerperliste. EIN widget statt knopf +…, Gecachte gliederung. Die koerperliste aendert sich zur laufzeit nicht, der…, Das panel in seiner MOMENTANEN aufklapp-hoehe. Es waechst aus der unterkante…, Symbol, beschriftung und der aufklapp-winkel. Das systemsymbol (zentralkoerper,… (+2 more)

### Community 34 - "schiffcontrol"
Cohesion: 0.15
Nodes (6): Latch/unlatch an orientation-hold. Tapping the active mode clears it., Hold the ship nose on a world-space heading supplied by the renderer. The…, Smoothly rotate toward a world-space direction vector (see above)., rotation mit echtem (wanduhr-)delta behandeln damit das drehen sich glatt…, Manual nose thrust as acceleration per real frame. dies stellt sicher, dass der…, schiffcontrol

### Community 35 - "TimingHistory"
Cohesion: 0.12
Nodes (10): Ringpuffer der per-frame zeitmessung, direkt fuer imgui.plot_lines. Ein…, Eine probe. Heisser pfad -- keine allokation, kein dict, kein try. `ui_calc`…, Laenge umstellen und dabei die JUENGSTEN proben behalten., `(feld, offset)` fuer plot_lines. Offset = aelteste probe., Kopie in chronologischer reihenfolge (alt -> neu). Nur fuer tests und ausgaben…, `(cur, avg, max)` ueber den GEFUELLTEN teil des puffers. Der ungefuellte rest…, Zerfallender spitzenwert der serie. Nicht gerastert. Je frame HOECHSTENS EINMAL…, Gerasterter achsen-maximalwert einer serie. (+2 more)

### Community 37 - "Palette"
Cohesion: 0.10
Nodes (8): Palette, Die vier bedeutungsfarben plus die feste, dunkle grundierung. Der grund (panel,…, Setzt die vier farben und leitet alle rollen neu ab., Ungehellte farbe -- fuer flaechen, nie fuer schrift., Der farbige schein hinter einem block -- ein schlagschatten mit versatz null…, Buendelt die stufen. Erreichbar ueber UIContext.theme., Nur noch der EINE satz -- der wechselknopf ist entfallen., Theme

### Community 38 - "style.py"
Cohesion: 0.16
Nodes (20): build_planet_style(), _color_basis(), _emit_ring(), _euler_matrix(), expand_segments(), _fbm(), _hash3(), _hsl() (+12 more)

### Community 39 - "Renderer"
Cohesion: 0.19
Nodes (5): IdentityReferenceFrame, Die alte debug-textwand unten links. Seit Phase 4 standardmaessig AUS: ihre…, Der Renderer -- zusammengesetzt aus mixins, ein zustand. Die klasse war 5900…, Konvertiert einen Welt-Punkt zu einer bestimmten Sim-Zeit in…, Renderer

### Community 40 - "Predictor"
Cohesion: 0.10
Nodes (11): Predictor, Die vorausberechnete bahnlinie -- zusammengesetzt aus mixins. Die klasse war…, build(), radius_range(), Die manoever-vorschau: was die geplanten knoten aus der bahn machen. Die…, Welt + schiff auf einer kreisbahn um die Erde + predictor., Kleinster und groesster abstand der linie zur Erde., _earth_orbit_scene() (+3 more)

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
Cohesion: 0.11
Nodes (13): FakeBody, ink(), Der hintergrund auf dem ECHTEN framebuffer -- was numpy nicht sehen kann.…, Ein koerper wie aus solar_system.json: position, velocity IMMER null., Einen hintergrund aus definiertem zustand zeichnen und auslesen., Zahl der pixel, die die eine schicht gegenueber der anderen aendert., shot(), busy_wait() (+5 more)

### Community 46 - "BurnProfile"
Cohesion: 0.11
Nodes (9): BurnProfile, Hebelstellung 0..1 -- was der schubbogen des HUDs anzeigt., Aufsummiertes delta-v von der zuendung bis `tau`., Delta-v ueber ein zeitfenster, GESCHLOSSEN gerechnet. Der ausfuehrer benutzt…, Trapez- oder dreiecksprofil fuer ein gegebenes delta-v., Zeit von der zuendung bis zum knoten -- die haelfte der dauer. Gilt, weil beide…, Absolute sim-zeit, zu der der schub einsetzen muss., Schubbeschleunigung `tau` sekunden nach der zuendung. ACHTUNG:… (+1 more)

### Community 47 - "JobsMixin"
Cohesion: 0.16
Nodes (8): JobsMixin, Rechnet ueberhaupt ein auftrag? (bequemlichkeit fuer altes verhalten), Obergrenze: konfiguration und verfuegbare kerne., So viele gleichzeitige laeufe, dass je BILD eines fertig wird. Die dauer einer…, Schub-neuberechnung ANFORDERN statt sie im hauptthread zu erzwingen. Waehrend…, Die asynchrone rechen-pipeline. Mehrere rechnungen laufen VERSETZT…, Neue horizont-/abstands-kurve ANFORDERN, ohne den halt aufzugeben. Dasselbe…, Wie viele auftraege rechnen gerade? `_pending_futures` enthaelt auch bereits…

### Community 48 - "orbit_lines_test.py"
Cohesion: 0.09
Nodes (13): check(), close(), _constant_rate_position_at_time(), CountingFrame, _drawn_step_deg(), line_missing_moon_by(), Regressionstest fuer orbit_lines.py -- die bahn-linien der koerper. Reine…, Winkelschritt der kurve koerper(t)-ursprung(t) um den bildmittelpunkt. (+5 more)

### Community 49 - "prediction_detail_test.py"
Cohesion: 0.12
Nodes (16): _drawn7(), drawn_arc_length(), drawn_radius_error(), _FakeCam, frame(), _hermite7(), _miss7(), Die vorhersagelinie wird so fein gezeichnet, wie der schirm es zeigt. Gemessen… (+8 more)

### Community 50 - "Stack"
Cohesion: 0.21
Nodes (4): Group, Unsichtbarer container. Nur zum verankern -- verbraucht keine maus., Unsichtbarer container, der seine kinder aneinanderreiht. Die verankerung aus…, Stack

### Community 52 - "horizon_targets_test.py"
Cohesion: 0.11
Nodes (17): Tastenbelegung und die klick-geste. Stand als eine 100-zeilige if/elif-kette…, horizon_compute_rung(), horizon_targets(), predictor_horizon_lengths(), Die laengenregel des vorhersage-horizonts. Lag als lose funktionen und closures…, Horizont-faktor aus der raffung -- zweierpotenz, gedeckelt. `rate` ist die…, (gezeichnete laenge, gerechnete laenge) fuer den vorhersage-horizont.…, Horizont neu setzen, wenn sich basis*manuell*raffung geaendert hat. (+9 more)

### Community 53 - "ManeuverDrawMixin"
Cohesion: 0.15
Nodes (10): ManeuverDrawMixin, Die fuenfte zeichen-domaene: der manoeverplan. Sie zeichnet drei dinge und…, Eine (n,5)-linie in schirmpunkte -- grob abgetastet, dann VERFEINERT. NUR…, Den plan zeichnen und seine schirmpositionen melden. Die trefferliste wird bei…, Wo die koerper stehen, wenn das schiff am ENDE DES PLANS ankommt. Dieselbe…, Ein sechseck -- die form, die sonst nichts im bild traegt. Die raute gehoert…, Ein stiel vom marker weg und eine pfeilspitze am ende. `deflection` ist der…, Geplante bahn, knotenmarker und ziehgriffe. (+2 more)

### Community 54 - "InputRouter"
Cohesion: 0.16
Nodes (7): InputRouter, Zeit auf der vorhersagelinie, die dem mauszeiger am naechsten liegt. Der…, +'/'-' verstellen den MANUELLEN faktor, nicht die laenge direkt. Sonst wuerde…, Uebersetzt ereignisse in aenderungen an welt, kamera und predictor., Auswahl / anflug. Aendert WEDER bezugskoerper NOCH bezugsrahmen., Linke maustaste: koerper auswaehlen / anfliegen. Die kamera zieht mit der…, Eine taste anwenden. Liefert False, wenn das spiel enden soll.

### Community 55 - "ManeuverAxesBlock"
Cohesion: 0.05
Nodes (23): setter, _edit_text(), ManeuverAxesBlock, ManeuverGizmo, ManeuverPlanBlock, Die ziehgriffe AN DER LINIE -- kein rechteck, nur treffer. Er zeichnet nichts:…, Auslenkung [-1, 1] -> ratenfaktor [-1, 1]. Null in der totzone, sonst `sign *…, Den knueppel INTEGRIEREN -- hier laeuft der wert, nicht im ereignis. Deshalb… (+15 more)

### Community 56 - "ReferenceFrameSelector"
Cohesion: 0.26
Nodes (5): FrameChangeCallback, PlottingFrameParameters, Bestimmt, welchem körper die kamera für den ausgewählten plotting-frame folgen…, ReferenceFrameSelector, resolve_plotting_camera_target_index()

### Community 57 - "reference_frames.py"
Cohesion: 0.18
Nodes (16): apparent_orbital_directions(), describe_plotting_frame(), _fallback_secondary_index(), _kepler_true_anomaly_from_mean(), new_plotting_frame(), _prograde_from_line(), reference-frame-primitiven und selector/adapter-verkabelung für spacesim. dies…, Frame-space tangent of the *actual* predictor polyline at its start. ``points``… (+8 more)

### Community 58 - "BackgroundLayer"
Cohesion: 0.12
Nodes (12): BackgroundLayer, Zustand und rechnung der hintergrund-ebene. Kein GL. `update()` einmal je bild…, True (einmalig), wenn der VBO neu geschrieben werden muss., Staerke des atmenden feldes, auf [0, 1] geklemmt., Der WAHRE gitteranker -- wo das lattice stehen muesste. "frame" (Vorgabe): die…, Der GEZEIGTE anker -- was `levels()` als phase bekommt., drift(), orbiting() (+4 more)

### Community 59 - "GLDeviceMixin"
Cohesion: 0.15
Nodes (8): GLDeviceMixin, Die GL-geraeteschicht des Renderers. War teil der 5900-zeiligen…, Wendet FXAA Post-Processing an. Erwartet, dass der ziel-framebuffer (screen)…, Context, FXAA-ziele, present und resize -- die geraeteschicht. Liegt in…, Fuehrt den buffer-swap aus und schreibt die swap-zeit in die timings. Von der…, Initialisiert OpenGL-Einstellungen (moderngl-state)., Erstellt FBO-textur und framebuffer in aktueller fenstergröße., Initialisiert FXAA Framebuffer und Shader.

### Community 60 - "_ManeuverPlate"
Cohesion: 0.12
Nodes (9): ManeuverBurnBlock, ManeuverNodesBar, _ManeuverPlate, Ein plaettchen im navball-raster. `SIDE` waehlt die flanke, `PLACE` das feld…, Die zur kugel zeigende seite bleibt SCHARF -- wie bei den flanken., Eine zeile: beschriftung links, wert rechts. DREI STUFEN, genau wie…, DV, brenndauer, countdown, EXECUTE -- ueber der ORB-flanke., Darf EXECUTE gedrueckt werden? Nur mit einem knoten, der wirklich delta-v… (+1 more)

### Community 61 - "Toggle"
Cohesion: 0.12
Nodes (6): Button, Mehrere sich gegenseitig ausschliessende optionen in einer leiste. Die form…, Klickbare schaltflaeche. Der klick loest beim LOSLASSEN aus, und nur wenn der…, Rastender schalter mit schiebeknopf. value darf ein aufrufbares objekt sein…, SegmentedControl, Toggle

### Community 62 - "ManeuverPlan"
Cohesion: 0.17
Nodes (4): ManeuverPlan, Bis zu `max_nodes` knoten, nach zeit sortiert., Der zeitlich naechste knoten -- den fliegt EXECUTE., Nach einer direkten feldaenderung aufrufen. Sortiert nach (ein ziehgriff darf…

### Community 63 - "ApsisTooltip"
Cohesion: 0.19
Nodes (6): ApsisTooltip, Die beiden zeilen als (schluessel, wert). ETA ist SIMULATIONSZEIT bis zur…, Hoehe der abstands-fahne, die der renderer unter die raute setzt., Zwei zeilen -- ankunftszeit und bahntempo -- unter der raute., Trefferkreis um die raute, in echten pixeln. Grosszuegiger als die raute…, Der marker unter dem zeiger, oder None. Bei zwei dicht beieinander liegenden…

### Community 64 - "Dropdown"
Cohesion: 0.14
Nodes (4): Dropdown, Auswahlfeld mit aufklappliste. Die liste wird vom widget SELBST gezeichnet,…, Horizontaler regler. log=True fuer groessen, die ueber mehrere zehnerpotenzen…, Slider

### Community 65 - "icon.py"
Cohesion: 0.17
Nodes (14): cells_array(), icon_palette(), _mix(), Entwurf A -- die Scheibe wird durchgehend gewuerfelt, radial gewichtet. Kein…, Entwurf D -- Kern plus 2-4 gesaete Zacken. Die Zacken sitzen auf DEMSELBEN…, Die POSITIONS-MARKE eines koerpers -- das icon beim herauszoomen. Nicht zu…, Die drei stufen-farben aus `body.color` (RGB 0..255). Der farbton bleibt…, Das feld als `(grid, grid)` int8 -- fuer tests und fehlersuche. Zeile 0 ist die… (+6 more)

### Community 66 - "build_app"
Cohesion: 0.19
Nodes (8): main(), spacesim -- 2D-N-Koerper-Bahnmechanik mit spielbarem raumschiff. DAS IST DER…, App, build_app(), load_config(), Alles, was die hauptschleife braucht -- ein sack voll verweise. Bewusst keine…, Zentrale Konfiguration laden. ALLE spielbaren parameter stehen in config.json;…, Ein echter frame des SPIELS (welt + HUD) als PNG. `hud_shot.py` zeichnet nur…

### Community 67 - "Gliederung"
Cohesion: 0.13
Nodes (14): 1 Einleitung, 2 Physikalische Grundlagen, 3 Näherung und Navigation: Apollo bis heute, 4 Bahnmanöver: Transfers und Swing-by, 5 Entwicklung der Simulation, 6 Validierung: Wie genau ist die Simulation?, 7 Fazit und Ausblick, Dateiorganisation (+6 more)

### Community 68 - "OrbitDrawMixin"
Cohesion: 0.17
Nodes (7): OrbitDrawMixin, Der zustandsbehaftete teil, ueber frames hinweg gehalten. Die konfiguration…, Die bahnlinien der koerper und die aufgezeichneten referenzspuren., Fertig projizierte bildschirmpunkte als polylinie, mit culling., Kleine raute auf dem endpunkt einer linie. Die endkappen sind der eigentliche…, Kreis-umriss mit dem ECHTEN radius des koerpers auf dem linienende. Das ist der…, Wo jeder koerper waehrend des VORHERSAGE-FENSTERS entlanglaeuft. Dieselbe…

### Community 69 - "loop.py"
Cohesion: 0.15
Nodes (16): _apply_horizon(), _apply_maneuver(), _clamp_warp(), FrameTimingPrinter, _handle_resize(), Die hauptschleife. Die REIHENFOLGE in `run()` ist an mehreren stellen…, Die `TIMING:`-zeile je frame. Zerlegt den frame in vorhersage-rechnung gegen…, Raffung auf das begrenzen, was die BAHN noch aufloest. (+8 more)

### Community 70 - "ManeuverNode"
Cohesion: 0.13
Nodes (7): ManeuverNode, Das schubprofil dieses knotens -- IMMER frisch gerechnet., Ein geplanter impuls: wann, und wieviel prograde/normal., armed_setup(), make_executor(), Welt + plan + scharfer ausfuehrer, an einem knoten weit vorn., Der manoeverplan: knoten, reihenfolge, und die orbitale basis. Was hier…

### Community 71 - "ui_hud_test.py"
Cohesion: 0.14
Nodes (8): cursor_at(), Cyc, frame(), _open_progress(), Regressionstest des spieler-HUDs (Phase 4). Drei ebenen, bewusst getrennt: 1.…, Ganzzahlige fensterkoordinate auf einem kompasswinkel am ring. GANZZAHLIG, weil…, Bildet NavballCluster._strip nach: drei stufen, wert gewinnt., _strip_width()

### Community 72 - "devui.py"
Cohesion: 0.18
Nodes (13): _checkbox(), draw_dev_panels(), _draw_timing_graphs(), _fmt_si(), _nice_ceiling(), Dear ImGui entwickler-oberflaeche (moderngl-nativ). Bewusst NUR fuer…, Zeichnet die entwickler-panels. `ctxobj` ist ein DevContext., Kompakte SI-darstellung fuer die readouts. (+5 more)

### Community 73 - "SegmentBar"
Cohesion: 0.08
Nodes (12): Platz, den der notch-tab AUSSERHALB des rahmens braucht. Er wird in measure()…, Die flaeche des eigentlichen rahmens, ohne das tab-band., Der zeitraffer -- die zellenleiste plus die laufende missionszeit. Die uhr…, Der rahmen endet ueber dem uhrstreifen., Der orientierungs-autopilot als rosette -- vier knoepfe um das schiff. Bildet…, Ein schlanker pfeil nach oben -- dieselbe silhouette wie die schiffsnase am…, SYSTEM / LOCAL -- zwei zoomstufen mit einem klick. Die stufen werden aus den…, Waagerechte zellenleiste mit sich ausschliessenden optionen. Traegt den… (+4 more)

### Community 74 - "background_test.py"
Cohesion: 0.18
Nodes (9): jump(), pattern_origin_px(), Regressionstest fuer background.py -- sternenfeld und dreiecksgitter. Reine…, `path` abfahren, jeweils die anker zurueckgeben., run(), _ss(), tracked(), visible() (+1 more)

### Community 75 - "body_icon_test.py"
Cohesion: 0.12
Nodes (13): Das zeichnen der himmelskoerper. Die detail-leiter (`_body_detail_levels`)…, Bahnlinien und referenzspuren. Die kurven selbst rechnet…, edge_count(), FakeBody, ink(), line_dips(), object, Die POSITIONS-MARKE der koerper -- rechnung und echte pixel. Das icon hatte… (+5 more)

### Community 76 - "TargetName"
Cohesion: 0.12
Nodes (6): Die gesperrte versal-ueberschrift eines panels., TARGET' links, 'LOCKED'-marke rechts., Der ziel-name in der zielfarbe., SectionLabel, TargetHeader, TargetName

### Community 77 - "Readout"
Cohesion: 0.20
Nodes (5): Label, Text-widgets: einfaches label und die zweispaltige messwert-zeile., Beschriftung links, wert rechts -- die standardzeile der info-panels. Der wert…, Text. Die groesse folgt standardmaessig dem inhalt (size=(None, None)). text…, Readout

### Community 78 - "orbit_lines.py"
Cohesion: 0.18
Nodes (12): _constant_track(), frame_origin_body(), future_track(), future_tracks(), polyline_stride(), Bahn-linien der himmelskoerper -- reine geometrie, kein GL. Zwei kurven je…, Der koerper, der im ursprung dieses plot-frames sitzt, oder None. Er bekommt…, Wo `body` zu den zeiten `times` stehen wird, (k, 2) in weltkoordinaten. Duenner… (+4 more)

### Community 79 - "Ausgangslage im April 2026"
Cohesion: 0.15
Nodes (13): Ausgangslage im April 2026, Beispiel, Bereits vorhandene Funktionen, Performance-Problem, Problem, Problem, Probleme, Probleme im April-Stand (+5 more)

### Community 80 - "HorizonPolicy"
Cohesion: 0.29
Nodes (3): HorizonPolicy, Haelt den horizont-zustand und setzt ihn am predictor durch. Der manuelle…, +' / '-': den MANUELLEN faktor verstellen, nicht die laenge direkt. Sonst…

### Community 81 - "maneuver_hud_test.py"
Cohesion: 0.17
Nodes (9): click(), frame(), key(), press(), Die vier manoever-plaettchen im navball-raster und der ziehgriff. Fuenf ebenen:…, Ein voller HUD-frame -- MIT zeichnen. Das `root.render()` ist keine zierde:…, Einen vollstaendigen klick auf den widget-baum schicken., Zeiger hin, dann drueckn -- ueber die ECHTE ereigniskette. Die bewegung gehoert… (+1 more)

### Community 82 - "Hud"
Cohesion: 0.11
Nodes (9): Hud, Einmal pro frame VOR ui_root.begin_frame() aufrufen. Erst abtasten, dann…, Naechstliegende zeitraffer-stufe zum AKTUELLEN sim_dt. Zurueckgelesen statt…, Ist diese stufe momentan erlaubt? (bahn-zeitskala, siehe…, Baut den widget-baum und haelt ihn pro frame aktuell. Besitzt die telemetrie…, IconRail, Die pille oben links: leuchtpunkt, schiffsname, bezugskoerper. Im entwurf steht…, Schmale senkrechte leiste fuer das kompakte layout. Ersetzt unter der… (+1 more)

### Community 83 - "TextRenderer"
Cohesion: 0.21
Nodes (6): Sucht die schriftdateien EINMAL, je familie und strichstaerke. Reihenfolge:…, Rastert alle rollen in der aktuellen skalierten pixelgroesse NEU. Fertige…, Font-verwaltung, label-textur-cache und texturiertes blitten., Wie draw(), aber erst beim naechsten flush() ausgefuehrt. Der einzige zweck:…, Eigene texquad-pipeline. Bewusst NICHT die des Renderers geteilt: die UI-…, TextRenderer

### Community 84 - ".sample"
Cohesion: 0.18
Nodes (5): Alles, was der MANEUVER-block je frame zeigt -- einmal gelesen. Brenndauer und…, Obergrenze der raffung aus der bahn-zeitskala., mu und spezifische bahnenergie -- die grundlage von vis-viva., Geschwindigkeit eines koerpers -- auch eines skriptgefuehrten. ACHTUNG, das ist…, v . r_dach gegen den bezugskoerper -- steig- bzw. sinkrate. Die geschwindigkeit…

### Community 85 - "advance"
Cohesion: 0.22
Nodes (9): advance(), _burn(), _measure(), probe(), Kurvenpunkt zur ABSOLUTEN sim-zeit t (linear interpoliert)., Ein weltschritt -- IN DER REIHENFOLGE DES SPIELS. `test.py::update` ruft erst…, _run24(), run_warp() (+1 more)

### Community 87 - "Entwicklung nach dem April-Stand"
Cohesion: 0.17
Nodes (12): Entwicklung nach dem April-Stand, Orbit- und Bahndaten, Reference-Frame-System, Schiffskontrolle, Verbesserung, Warum ist das wichtig?, Warum war das wichtig?, Weltphysik (+4 more)

### Community 88 - "._render_tracked"
Cohesion: 0.33
Nodes (3): Breite der breitesten ziffer dieser schrift, einmal gemessen., Rastert text, bei bedarf mit LAUFWEITE (letter-spacing). Die oberflaeche sperrt…, Ein render-aufruf, mit alphakanal auch im hart gerasterten fall. Ohne…

### Community 89 - ".draw"
Cohesion: 0.33
Nodes (3): Zeichnet text an TOP-DOWN koordinaten. align: 'left' | 'center' | 'right' --…, Zeichnet und leert die aufgeschobene warteschlange., Top-down -> ortho und auf das pixelraster rasten.

### Community 90 - "frame_affine_at"
Cohesion: 0.22
Nodes (8): _cubic_4pt(), frame_affine_at(), FrameAffineTable, Kubik DURCH ALLE VIER knoten (Lagrange), ausgewertet auf [p1, p2]. Wortgleich…, Die starre transformation eines plot-frames ueber ein ZEITFENSTER. Alle koerper…, Knotenzahl aus dem, was sich im fenster WIRKLICH dreht. Zwei winkel, und beide…, (fx, fy) fuer beliebige zeiten im fenster, vollstaendig in numpy. Das gitter…, Die starre transformation des frames zur zeit `t`. Gibt `(r00, r01, r10, r11,…

### Community 91 - "BodyCentredNonRotatingReferenceFrame"
Cohesion: 0.20
Nodes (7): BodyCentredNonRotatingReferenceFrame, linear_reference(), make_frame(), _origin_run(), Der ursprung eines plot-rahmens wird KUBISCH interpoliert, nicht linear.…, (groesster abstand zum exakten wert, groesste bewegung je frame) in m., Der ALTE weg: sehne zwischen den beiden umgebenden knoten.

### Community 92 - "_bvel24"
Cohesion: 0.50
Nodes (4): _bvel24(), _departure24(), _orbit24(), Bahngeschwindigkeit eines geskripteten koerpers. `body.velocity` ist bei allen…

### Community 93 - "devui_timing_test.py"
Cohesion: 0.18
Nodes (5): FakePredictor, FakeRenderer, push_ramp(), Regressionstest fuer die timing-ringpuffer der entwickler-oberflaeche. Prueft…, n proben, in denen jede serie einen eindeutigen wert bekommt.

### Community 94 - "ConfigLoader"
Cohesion: 0.06
Nodes (22): ConfigLoader, Zentrale konfiguration: `config.json` ->…, Laedt `config.json` und verteilt die parameter auf die einzelnen module. die…, Path, PlottingFrameAdapter, FrameController, Den ganzen apparat zusammenbauen: welt, kamera, predictor, renderer, UI. Das…, Bezugsrahmen und bezugskoerper anwenden. Haengt an `UIState.on_change`, wird… (+14 more)

### Community 95 - "_fires"
Cohesion: 0.67
Nodes (3): _fires(), _place_on_ellipse(), Wie oft fordert die schub-erkennung in `frames` bildern eine neuberechnung an?…

### Community 97 - "._recompute"
Cohesion: 0.22
Nodes (7): closest_approach(), OrbitLineEntry, Radius der einflusssphaere, `a * (m / m_elter)^0.4`, oder None. Der massstab,…, Kleinster abstand zwischen schiff und koerper ZUR GLEICHEN ZEIT. Alle drei…, Was zu EINEM koerper je bild bekannt ist., Das gitter, auf dem die spur GEZEICHNET wird. Standardfall ist das gemeinsame:…, soi_radius()

### Community 98 - ".update"
Cohesion: 0.22
Nodes (8): fold(), fold_spans(), Die kleinste GITTERTRANSLATION, in der `(x, y)` gefaltet werden darf. Der anker…, `value` in `(-span/2, +span/2]` zurueckholen., Die geschwindigkeit des verfolgten koerpers, aus seiner POSITION. > **Ein…, Einen bildschritt weiterdrehen. `cam_world_xy` ist die kameraposition in…, Geradeausflug; gibt (layer, ZURUECKGELEGTE strecke in px) zurueck. ACHTUNG:…, slide()

### Community 99 - ".levels"
Cohesion: 0.20
Nodes (6): Level, Wie GLSL `smoothstep` -- C1-stetig, damit nichts poppt., Eine sichtbare gitter-dekade., Die sichtbaren dekaden, hellste zuerst, hoechstens `MAX_LEVELS`. Eine dekade…, Gitterphase in `(a, b)`-koordinaten, modulo 2. `a = x*sqrt(3)/ws`, `b = y/ws`;…, _smoothstep()

### Community 100 - "Window"
Cohesion: 0.25
Nodes (4): Fenster, GL-context und der frame-takt. Stand als erste 70 zeilen von `main()`…, Das SDL/OpenGL-fenster und die uhr, die den frame-takt vorgibt., Einen frame abwarten und das ECHTE delta in sekunden liefern., Window

### Community 101 - "body_style_test.py"
Cohesion: 0.29
Nodes (7): draw(), half_brightness(), look_at(), probe_thickness(), Regressionstest der prozeduralen koerper-optik (D2). Die zeichnung eines…, Einen frame zeichnen -- und, wenn noetig, auf den bau warten. Der bau laeuft…, read_pixels()

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

### Community 108 - "background.py"
Cohesion: 0.22
Nodes (7): family_normals(), lattice_vertices(), parse_hex_color(), Hintergrund-ebene -- sternenfeld und rekursives Dreiecksgitter. Zwei schichten,…, `"#17b2c4"` -> `(r, g, b)` in 0..1. Bei unsinn: `default` (HUD-cyan)., Die knoten des dreiecksgitters als `(n, 2)`-array in weltkoordinaten. `x =…, Die drei linien-normalen.

### Community 109 - "_w17_frame"
Cohesion: 0.22
Nodes (9): arc_length(), predictor_warp_length_mult() aus test.py., step_simulation() aus test.py., Ein frame der hauptschleife -- nur der zeitraffer-relevante teil., Auf from_rate einschwingen, dann auf to_rate wechseln. Rueckgabe: (ms auf dem…, _w17_frame(), _w17_mult(), _w17_step() (+1 more)

### Community 110 - "mass"
Cohesion: 0.25
Nodes (8): chain_periods(), orbital_period(), Umlaufzeit des koerpers um seinen elter, `2*pi*sqrt(a^3/mu)`, oder None. Fuer…, `{id(glied): umlaufzeit}` fuer jedes glied der elternkette von `body`. Ein…, Schnellste umlaufzeit, die in `koerper(t) - ursprung(t)` steckt. DIE UMLAUFZEIT…, relative_min_period(), mass(), Masse in kg / t / kt / Mt / Gt, darueber wissenschaftlich in kg. Bewusst KEINE…

### Community 112 - "BackgroundDrawMixin"
Cohesion: 0.29
Nodes (5): BackgroundDrawMixin, Das zeichnen der hintergrund-ebene. Die geometrie (sterne, gitter) rechnet…, Sternenfeld und dekaden-gitter. NICHTS HIER SKALIERT MIT camera.scale -- der…, Laedt die sterntabelle in den instanz-VBO, wenn die dichte wechselt. Der puffer…, Zeichnet sternenfeld und gitter -- die unterste schicht. Laeuft VOR allem…

### Community 114 - "PlanetStyle"
Cohesion: 0.29
Nodes (3): PlanetStyle, object, Fertige zeichnung eines koerpers im einheitskreis. `tri` sind die flaechen…

### Community 115 - "DevContext"
Cohesion: 0.29
Nodes (5): DevContext, _ms(), Robuste ms-zahl. Fehlende/kaputte werte werden zu 0, nicht zu NaN. Ein NaN in…, Sammelt die objekte, die die entwickler-panels verstellen duerfen. Bewusst ein…, Eine probe der vier zeitreihen. Je frame einmal, aus test.py. BEWUSST…

### Community 119 - "_imul"
Cohesion: 0.40
Nodes (5): Der seed einer marke: `style_seed`, sonst der name -- plus ein globaler…, seed_for(), _imul(), Stabiler 32-bit-seed aus dem koerpernamen. Damit bekommt jeder koerper ohne…, seed_from_name()

### Community 120 - "Rendering der Predictor-Linie"
Cohesion: 0.33
Nodes (6): Problem: zu viele Punkte im Bildschirmraum, Rendering der Predictor-Linie, Verbesserung: begrenzte Punktanzahl beim Rendering, Verbesserung: sichtbare Abschnitte statt einer Punktliste, Verbesserung: Zwischenspeicherung der gerenderten Linie, Weiterhin bestehendes Problem

### Community 121 - "KeplerScriptedOrbit"
Cohesion: 0.47
Nodes (3): KeplerScriptedOrbit, Hilfs-Kepler-Orbit, nur für die Visualisierungs-Frame-Logik., _rotate_xy()

### Community 124 - ".orbital_speed_scale"
Cohesion: 0.33
Nodes (3): (bahntempo, kreisbahn-tempo, fluchttempo) am aktuellen ort. Der MASSSTAB der…, Nadellaenge auf [0, 1]: 0 = ruhe, ~0.71 = kreisbahn, 1 = flucht., Wo auf derselben skala die kreisbahn liegt -- immer 1/sqrt(2). Steht hier…

### Community 125 - "._retarget"
Cohesion: 0.40
Nodes (4): approach_alpha(), Deckkraft aus der dichtesten annaeherung, in SOI-vielfachen. `miss <= soi_full…, Wie VIEL der linie gezeichnet wird, 0..1 -- neben der deckkraft. Zweites,…, reveal_fraction()

### Community 129 - "Role"
Cohesion: 0.40
Nodes (3): Eine typo-rolle: groesse, laufweite, strichstaerke, schriftfamilie. FAMILIE ist…, Rueckwaertskompatibler alias -- die pixelschrift IST dicktengleich., Role

### Community 130 - "frame_project"
Cohesion: 0.50
Nodes (4): frame_project(), _frame_project_scalar(), Weltpunkte in den plot-frame, je ZEIT statt je PUNKT. Die ellipse besteht aus…, Punktweise rueckfallebene -- langsam, aber immer richtig.

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

### Community 142 - "compass_to_screen"
Cohesion: 0.50
Nodes (4): compass_to_screen(), Ein bogen aus EINZELNEN zellen -- die anzeigeform der vorlage. Ein…, Kompasswinkel -> der winkel, den UIDraw.arc erwartet. UIDraw rechnet…, segment_arc()

### Community 146 - "Predictor"
Cohesion: 0.67
Nodes (3): Nachteile, Predictor, Vorteile

### Community 147 - "_clear_of_bodies21"
Cohesion: 0.67
Nodes (3): _clear_of_bodies21(), t_char, wenn das schiff bei diesem Erd-abstand auf der transferbahn steht., _tchar_at()

### Community 148 - "_hermite_at"
Cohesion: 0.67
Nodes (3): _hermite_at(), Die linie KUBISCH auswerten -- so zeichnet der renderer sie auch. Linear…, _walk18()

## Knowledge Gaps
- **94 isolated node(s):** `_Motion`, `Warum gibt es Integrators?`, `Funktionen des Spiels:`, `Leitfrage`, `Gewichtung` (+89 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1196 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **28 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Rect` connect `Rect` to `Dropdown`, `with_alpha`, `HorizonSlider`, `UIContext`, `SystemMap`, `background_test.py`, `_emit_shape`, `Widget`, `maneuver_hud_test.py`, `Stack`, `_ManeuverPlate`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Why does `Predictor` connect `Predictor` to `predictor/core.py`, `build_app`, `SystemLoader`, `ui_hud_test.py`, `warp_predictor_test.py`, `JobsMixin`, `maneuver_hud_test.py`, `ViewMixin`, `prediction_detail_test.py`, `horizon_targets_test.py`, `HoldMixin`, `ConfigLoader`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `Renderer` connect `Renderer` to `SystemLoader`, `Knobs`, `ShaderPipelineMixin`, `ShipDrawMixin`, `PredictionDrawMixin`, `TextMixin`, `DrawMixin`, `renderer.py`, `BodyDrawMixin`, `selection_camera_test.py`, `render_budget_test.py`, `prediction_detail_test.py`, `ManeuverDrawMixin`, `BackgroundLayer`, `GLDeviceMixin`, `build_app`, `OrbitDrawMixin`, `ui_hud_test.py`, `body_icon_test.py`, `maneuver_hud_test.py`, `ConfigLoader`, `body_style_test.py`, `BackgroundDrawMixin`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `Vec2` (e.g. with `body` and `ReferenceFrame`) actually correct?**
  _`Vec2` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `world` (e.g. with `body` and `Vec2`) actually correct?**
  _`world` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `_Motion`, `Warum gibt es Integrators?`, `Funktionen des Spiels:` to the rest of the system?**
  _94 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `predictor/core.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05319355464958261 - nodes in this community are weakly interconnected._