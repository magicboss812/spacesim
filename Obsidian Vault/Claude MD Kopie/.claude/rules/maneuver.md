---
paths:
  - "spacesim/ship/maneuver/**"
  - "spacesim/physics/kernels/burn.py"
  - "spacesim/render/maneuver.py"
  - "spacesim/ui/hud/maneuver.py"
---

# Manöverknoten — planen, vorschauen, brennen

Bis zu fünf Knoten auf der Vorhersagelinie. Jeder trägt zwei Zahlen
(prograde, normal) und eine Zeit; alles andere ist abgeleitet.

| Datei | Rolle |
| --- | --- |
| `ship/maneuver/profile.py` | `BurnProfile` — **die** Quelle für Brenndauer und Zündzeitpunkt |
| `ship/maneuver/plan.py` | `ManeuverNode`, `ManeuverPlan`, `orbital_basis`, `burn_direction_world` |
| `ship/maneuver/preview.py` | die Kette coast → burn → coast als gezeichnete Linie |
| `ship/maneuver/executor.py` | der Autopilot: scharfschalten, zünden, brennen, aufräumen |
| `physics/kernels/burn.py` | `_burn_arc_numba` (RK4 mit Schub) + der njit-Zwilling des Profils |
| `render/maneuver.py` | fünfte Zeichen-Domäne: Linie, Marker, Ziehgriffe |
| `ui/hud/maneuver.py` | die vier Plättchen im Navball-Raster und `ManeuverGizmo` |

## Es gibt genau EINE Brenndauer-Rechnung, und das ist `BurnProfile`

Vorschau und Ausführung rufen dieselbe Klasse. Stünde die Dauer an zwei
Stellen, zeigte die gezeichnete Linie ein anderes Manöver als geflogen wird
— und zwar erst dann, wenn es zu spät ist, es zu bemerken.

**Erhalten wird die RATE, nicht die Rampenzeit**: `r = a_max / ramp_seconds`
in m/s³. Bei kleinem Δv wird `a_max` nie erreicht; mit fester Rampen*zeit*
gäbe es dort einen Sprung in der Dauer. Mit fester Rate gehen Trapez
(`T = ramp + Δv/a_max`) und Dreieck (`T = 2·√(Δv/r)`) stetig ineinander über
— `tests/maneuver_profile_test.py` §3 misst 0.99999833 gegen 1.00000167 s
über die Nahtstelle.

Beide Formen sind symmetrisch, und **daraus folgt der Zündzeitpunkt**: bei
`total_time/2` ist genau die Hälfte des Δv geliefert, der Knoten sitzt also
in der MITTE des Brennvorgangs. `lead_time` und `ignition_time()` sind
nichts als diese Symmetrie. §5 prüft sie an vier Profilen.

## Der njit-Zwilling ist absichtlich doppelt — und wird bewacht

`physics/kernels/burn.py::_profile_accel_numba` schreibt
`BurnProfile.accel_at` ein zweites Mal, weil Numba keine Python-Objekte
annimmt. §8 des Profiltests vergleicht beide an 2004 Stützstellen auf
**exakte** Gleichheit (gemessen: größte Abweichung 0). Ohne diese Prüfung
könnten Vorschau und Ausführung unbemerkt auseinanderlaufen.

## Die Schritte des Brennbogens liegen auf den PHASENGRENZEN

Das Profil ist an den beiden Rampenknicken stetig, aber nicht
differenzierbar. RK4 wiegt drei Auswertungen wie Simpson — über einen Knick
hinweg ist das nur erster Ordnung. Gemessen bei 400 gleichverteilten
Schritten über ein 12.6-s-Profil: **119.999775 statt 120.000000 m/s**.
Innerhalb einer Phase ist die Beschleunigung linear oder konstant, und
Simpson integriert beides exakt — mit phasenweiser Schrittung landet das Δv
auf der letzten Stelle. Jede Phase bekommt deshalb ihre eigene Schrittweite.

Nebenwirkung: der mittlere *Index* ist nicht mehr die halbe *Zeit* — wer im
Bogen nach der Hälfte sucht, sucht über Spalte 2.

## `a_max` ist eine SIM-Zeit-Beschleunigung

`schiffcontrol.apply_thrust` addiert `thrust_acc * real_dt` (600 m/s² ×
Wandsekunden), während die Welt um `sim_dt * tick_rate * real_dt` SIM-Sekunden
vorrückt — auf der untersten Raffungsstufe 60 pro Echtsekunde. In Sim-Zeit
beträgt die Beschleunigung der Pfeiltaste also **600 / 60 = 10 m/s²**. Die
Vorschau integriert in Sim-Zeit, folglich:

    a_max_sim = thrust_acc / realtime_warp_max

Mit `thrust_acc` direkt bränne der Autopilot **sechzigmal** so hart wie die
Pfeiltaste — und die Vorschau zeigte diesen Brennvorgang auch noch korrekt
an, so dass nichts auf dem Schirm den Fehler verriete.
`config.maneuver.max_accel` überschreibt die Ableitung.
`tests/maneuver_execute_test.py` §1 rechnet beide Wege gegeneinander.

## Die Schubrichtung wird EINMAL aufgelöst und festgehalten

`(prograde, normal)` werden an der Knotenzeit gegen die Bahn aufgelöst
(`orbital_basis`) und ergeben eine feste Weltrichtung für den ganzen
Brennvorgang. Genau das macht Vorschau und Ausführung zur selben
Integration; eine mitdrehende Richtung kann die Vorschau nicht im Voraus
kennen.

**Normal ist positiv NACH INNEN** — dieselbe Vorzeichenregel wie
`reference_frames.apparent_orbital_directions::normal_in`. Daran hängen die
Snap-Rosette, die Marker am Kompassring und die Ziehgriffe gemeinsam; zwei
Regeln hieße, der Griff zeigt nach oben und das Schiff brennt nach unten.

## Die Vorschau rechnet NEBENHER, und das ist keine Kür

Ein Neuaufbau kostet gemessen 7–15 ms. Im Hauptthread gerechnet bezahlt er
genau die Eingabe, die ihn auslöst: beim Ziehen eines Griffs fiel die
Bildrate von 100 auf 40. `ManeuverPreview` stellt deshalb **Aufträge** an
einen Arbeitsthread — erlaubt, weil alle beteiligten Kernel `nogil=True`
sind, dieselbe Voraussetzung wie bei `ship/predictor/jobs.py`.

**Die Arbeitsteilung ist die ganze Schwierigkeit.** Im Hauptthread entsteht
der Auftrag (`_make_job`): Schnappschuss, eine **Kopie** der Basislinie und
je Knoten Zeit, Δv und der Zustand des Bezugskörpers zu dieser Zeit. Das
alles liest Python-Objekte, die der Hauptthread im selben Moment
weiterschreibt. Der Arbeiter (`_run_preview_job`) sieht danach nur noch
Arrays und Zahlen. Ein Auftrag zur Zeit; eingewechselt wird mit **einer**
Zuweisung, also nie halb.

Gemessen bei 1920×1080 ohne VSync: Hauptthread **0.12 ms Median, 0.35 ms
Maximum** je Frame während laufender Änderung. Bildzeit ohne Knoten 7.2 ms
(140 fps), mit stehendem Knoten 9.1 ms (110 fps), mit laufendem Δv 11.3 ms
(88 fps) — ein Aufschlag von **2.2 ms**, wo vorher die Bildrate auf 40
einbrach. `tests/maneuver_preview_test.py` §8b prüft zusätzlich, dass der
nebenläufige Weg **exakt** dieselben Zahlen liefert wie der synchrone
(größte Abweichung 0.0).

Der Riegel ist damit nicht mehr die Uhr, sondern der Auftrag: solange einer
läuft, wird kein zweiter gestellt. Der feste Mindestabstand von vorher
(0.15 s) rastete gegen die unregelmäßigen Versionssprünge des Predictors und
ließ die Linie ungleichmäßig nachziehen — er steht jetzt auf 0.

`wait()` blockiert bis zum Ergebnis. Nur zwei Stellen dürfen das:
`tools/game_shot.py` (ein Abzug darf nicht einen Auftrag alt sein) und
`InputRouter.toggle_execute` — die Schubrichtung kommt aus den Markern der
Vorschau, und wer gleich nach `N` die `X` drückt, träfe sonst auf eine
Kette, die den Knoten noch nicht kennt.

## Reichweite und Auflösung sind ZWEI Regler

Die Vorhersagelinie beantwortet, wo das Schiff *gleich* ist — da will man
Auflösung. Der Plan beantwortet, wo ein Knoten den vierten Umlauf hinlegt —
da will man Weite. An einem Regler hängend müsste man für Weite immer
Auflösung mitkaufen, und die Vorschau reichte in der Vorgabe nur ein Drittel
so weit wie die Linie, gegen die sie gesetzt wird.

`ManeuverPreview.length_mult` (HUD: `PLAN`, über der ALT-Flanke) wirkt auf
den **Punktabstand** (`precision`, eine Bogenlänge), nicht auf die
Punktzahl: Reichweite = Abstand × Punkte, und nur der Abstand ist ohne
Kosten je Punkt zu haben. `max_iters` wächst mit, sonst bräche der adaptive
Integrator auf halber Strecke ab statt weiter zu reichen. Gemessen bei 1500
Punkten: ×1 → 13.4 h, ×2 → 26.3 h, ×4 → 52.8 h, bei **gleichbleibender**
Punktzahl. Die Rechenzeit wächst mit dem Bogen, nicht mit den Punkten —
7.5 ms bei ×1, 80 ms bei ×16; deshalb steht die Obergrenze bei 32.

## Die Vorschau ist eine Kette, und sie rechnet nur bei Änderung

Jeder Knoten liest seinen Zustand von der Linie des vorigen
(`state_on_curve`, kubische Hermite über die Geschwindigkeitsspalten) —
**das** macht sie zur Kette. Der Abschnitt vor dem ersten Knoten wird nicht
neu propagiert, sondern von der vorhandenen Vorhersagelinie gelesen.

Drei Dinge daran waren beim Bauen falsch und sind es jetzt nicht mehr:

1. **Der Brennbogen wird zum Zeichnen ausgedünnt** (`burn_draw_points`, 24).
   Er wird mit Hunderten Schritten integriert — das ist Genauigkeit —, ist
   aber ein paar Sekunden lang. Alle Schritte zu zeichnen verbrannte
   gemessen **400 von 1200 Punkten** für einen 12-Sekunden-Bogen, worauf die
   anschließende Gleitphase zu kurz wurde, um den nächsten Knoten zu
   erreichen: die Kette brach nach dem ersten Glied ab.
2. **Das Budget wird NICHT vorab aufgeteilt.** Eine halbierte Gleitphase
   reicht zeitlich kürzer, und ein Knoten wenige Sekunden hinter ihrem Ende
   fällt aus der Kette — gemessen bei 65 062 s Abstand, während die
   ungeteilte Phase 244 434 s weit reichte. Jede Phase rechnet mit dem
   vollen Rest und wird hinterher abgeschnitten; verbraucht ist nur, was
   übrig bleibt.
3. **Abgeschnitten wird an der ZÜNDUNG des nächsten Knotens**, nicht an
   dessen Knotenzeit. Die Zündung liegt eine halbe Brenndauer davor; am
   Knoten geschnitten überlappen sich die Abschnitte um genau diese Spanne
   und die zusammengefügte Linie läuft an der Nahtstelle rückwärts. Und der
   Budget-Abbruch steht deshalb **hinter** dem Abschneiden — davor sah er
   immer null.

Ein Knoten jenseits des gezeichneten Horizonts hat keinen ablesbaren
Zustand. Die Kette bricht dort ab, statt zu raten
(`tests/maneuver_preview_test.py` §9).

**Neu gerechnet wird nur**, wenn `plan.version`,
`predictor._trajectory_version` oder `length_mult` sich bewegt haben. Wer
`plan.nodes[i].dv_*` direkt schreibt, **muss** `plan.touch()` rufen — sonst
bleibt die Linie stehen.

## Kick-then-drift, und warum der Schritt gedeckelt ist

`ManeuverExecutor.update()` legt das Δv eines Schrittes an, BEVOR
`world.step()` ihn geht; die Vorschau integriert stetig. Der Unterschied ist
erster Ordnung in der Schrittweite, deshalb `burn_step_max_s` (0.5 s).
Gemessen nach einem 120 m/s-Brennvorgang plus 15 Minuten Flug:
**1.72 m/s auf 25 433 m/s** (0.007 %) und **3944 m auf r = 1.04e7 m**
(0.038 %) — `tests/maneuver_execute_test.py` §8.

Der Δv-Betrag selbst ist exakt, weil `dv_between()` geschlossen rechnet
statt `a·dt`: gemessen 120.000000000 von 120.0 geplanten.

## Der Zeitraffer zieht sich vor der Zündung selbst herunter

Schub ist Echtzeit-only. Scharf geschaltet deckelt `max_sim_seconds()` den
Schritt auf die verbleibende Zeit bis zur Zündung — sonst rückt ein
Zeitraffer-Frame um Stunden vor und der ganze Brennvorgang fällt zwischen
zwei Bilder (§4 prüft das mit 1-Stunden-Frames: größter Überschuss 0.000e+00
s). Innerhalb von `orient_lead_seconds` zwingt `_apply_maneuver()` in
`runtime/loop.py` `camera.sim_dt` auf Echtzeit.

**Handeingabe bricht immer ab.** Alle vier Steuertasten, nicht nur der
Schub: wer im Brennvorgang dreht, macht die restliche Brenndauer ohnehin
ungültig. Der Knoten bleibt dabei im Plan.

## Die Griffe sind KNÜPPEL, keine Schieberegler

Der Wert war die absolute Zeigerstrecke mal einem Faktor: für 500 m/s musste
man 660 px ziehen, also quer über den Schirm, und am Bildrand war Schluss.
Jetzt ist die Auslenkung eine **Rate** (`handle_dv_rate`, 260 m/s je Sekunde
bei Vollausschlag über `handle_travel_px` = 96 px), die läuft, solange
gehalten wird — dieselbe Bauart wie der Horizontregler
(`ui/widgets/rate_slider.py`) und aus demselben Grund: eine Größe ohne
natürliche Obergrenze braucht ein Steuer, keinen Weg.

Das ist auch der Grund, warum die Eingabe weich ist. Am absoluten Regler
sprang der Wert mit jedem Maus-Ereignis; jetzt läuft er **zeitintegriert** in
`ManeuverGizmo.update()`, also mit der Bildrate geglättet und unabhängig
davon, wie oft das Betriebssystem die Maus meldet. Kennlinie quadratisch mit
0.08 Totzone: nahe der Ruhelage trifft man einzelne m/s, am Anschlag
hunderte in der Sekunde. Der gezogene Griff wird ausgelenkt **gezeichnet**
(`renderer.maneuver_drag_handle`) — ohne sichtbaren Ausschlag gäbe es keine
Rückmeldung darüber, wie schnell der Wert gerade läuft.

## Schirmpositionen werden VERÖFFENTLICHT, nicht nachgerechnet

`render/maneuver.py` legt `renderer.maneuver_node_hits` ab (Marker + vier
Griffe, je Zeichendurchgang geleert und neu gefüllt) — dasselbe Muster wie
`apsis_marker_hits` und aus demselben Grund: nur der Renderer kennt die
zeitabhängige Frame-Transformation, die den Marker auf der gezeichneten
Linie hält. `ManeuverGizmo` trifft gegen genau diese Zahlen.

Daraus folgt **ein Frame Versatz**, absichtlich: die Liste entsteht in
`render()`, also nach der Ereignisschleife. Bei 60 fps unsichtbar, und die
Alternative wäre die zweite Transformation, die diese Aufteilung vermeidet.

Die **Griffrichtungen** kommen aus `to_this_frame_vector_xy`, nicht aus
einer Differenz zweier transformierter Punkte: ein Epsilon-Schritt neben
einem Ort bei 1.5e11 m verliert sechs Stellen, bevor er gerechnet ist.

## Die gezeichnete Linie wird VERFEINERT, nicht nur ausgedünnt

Die Vorschau setzt ihre Punkte in gleichem **Bogenabstand**, und dieser
Abstand wächst mit der eingestellten Reichweite (`length_mult`). Ein fester
Stride darauf macht die Linie mit jeder Verlängerung kantiger, bis sie
sichtbar aus geraden Stücken besteht — gemessen **0.53 px** Abweichung bei
×4 und **0.50 px** bei ×16 gegen 0.11 px bei ×1, und am Bildschirm als
Polygonzug zu sehen. Das ist kein Nebeneffekt der Nebenläufigkeit; es ist
die Abtastung selbst.

Der Ausweg ist derselbe, den die Vorhersagelinie längst nimmt: grob
abtasten (`maneuver.path_coarse_points`, 160) und die Segmente per
**kubischer Hermite** so weit unterteilen, wie eine Flachheitsschranke in
*Pixeln* es verlangt — `_hermite_refine_world` mit
`_prediction_error_budget`, dieselben zwei Funktionen, die
`render/prediction.py` benutzt. Die Auflösung hängt damit am **Bildschirm**
statt an der Linienlänge.

Gemessen nach dem Umbau: **0.025 px** bei ×1, **0.22 px** bei ×4, **0.32
px** bei ×16 — bei 384 / 444 / 479 gezeichneten Punkten gegen ein Budget
von 480. Länger heißt also nicht mehr gröber, und die Kosten bleiben
gedeckelt: unsichtbare Segmente werden gar nicht erst unterteilt, und das
Budget wird gleichmäßig gedrückt statt am Ende abgeschnitten. Bildzeit
1920×1080: 7.0 ms ohne Knoten, 9.8 ms mit gezeichneter Linie, 12.5 ms
während laufender Δv-Änderung.

Die Zwischenpunkte sind **kubische Näherungen**, keine integrierten
Zustände — bei sehr großem Punktabstand weicht die Näherung selbst um
Bruchteile eines Pixels ab. `tests/maneuver_render_test.py` §3c lässt der
Gegenprobe deshalb ein Zehntel Pixel Spielraum und prüft den Faktor
zwischen grob und fein über alle drei Stufen gemeinsam (größter Faktor
4.17).

**Der Ziehweg wird NICHT verfeinert** (`refine=False`). Er wird nicht
gezeichnet, sondern nur durchsucht — Glattheit kauft dort nichts und
kostet Projektionen genau in dem Moment, in dem gezogen wird.

## Die Endkappen stehen am Ende des PLANS

Dieselbe Aussage wie bei den Bahnlinien (`render/orbits.py`), nur eine Bahn
weiter: dort steht die Kappe für das Ende der *Vorhersage*linie, hier für
das Ende der *geplanten*. Der Plan reicht weiter, und ohne eigene Kappen
ließe sich gar nicht ablesen, wo ein Knoten das Schiff gegenüber den
Körpern hinlegt.

`_draw_maneuver_end_caps` benutzt dieselben zwei Zeichner wie die
Bahnlinien: `_draw_body_disc_outline` für den Körper zur Planendzeit und
`_draw_end_cap` für das Linienende. Der Kreis trägt den **echten** Radius
(`body.radius * camera.scale`), keinen festen Pixelwert — liegt die grüne
Raute in ihm, steckt das Schiff zur Planendzeit im Körper. Beides in Grün,
der Farbe der geplanten Bahn: die weißen und körperfarbenen Kappen gehören
der Vorhersage, und zwei Bedeutungen in einer Farbe wären keine.

Die Auswahl regelt sich selbst: unter 0.75 px Radius wird nichts gezeichnet
(derselbe Boden wie bei den Bahnlinien), und was neben dem Bild liegt,
fällt in `_draw_body_disc_outline` heraus. Bei Sonnensystem-Zoom bleibt
damit von allein nichts übrig. Abschaltbar über `maneuver.end_caps`.

**Projiziert wird in EINEM Rutsch** (`to_this_frame_xy_arrays`). Punktweise
war genau der Fehler, den `render/prediction.py` für die Vorhersagelinie
schon einmal behoben hat — dort lagen 3000 einzelne Aufrufe bei 5.6 ms je
Frame, praktisch alles davon Python-Aufruf-Overhead. Hier waren es 900
Punkte je Frame plus 400 weitere während eines Zugs, und *das* war neben der
Rechnung im Hauptthread die zweite Hälfte des Bildraten-Einbruchs.
`tests/maneuver_render_test.py` §3b vergleicht beide Wege: Abweichung
0.000e+00 px. `maneuver.path_draw_points` (480) ist seit der Verfeinerung
das **Budget**, nicht mehr ein Stride: die Punkte landen dort, wo die Kurve
biegt, statt gleichmäßig über eine meist gerade Linie.

`maneuver_curve_screen` (die Basislinie in Schirmkoordinaten, für das
Verschieben des Markers) wird **nur während eines MARKER-zugs** gebaut —
`maneuver_drag_curve`, nicht `maneuver_drag_active`. Ein Griff verschiebt
den Knoten gar nicht, er ändert nur sein Δv; die zusätzlichen Projektionen
liefen also ausgerechnet während der Eingabe, bei der es auf Bildrate
ankommt.

## Der Gizmo fängt die Maus nur über einem Griff

`hit_test` prüft Radien, kein Rechteck. Das Widget sitzt mitten im Bild über
der Bahn; eines, das dort flächig Klicks schluckt, blockierte Körperauswahl
und Kameraschwenk — derselbe Fehler, den `AttitudeRing.hit_test` für den
Kompassring behebt.

## Vier Plättchen IM Navball-Raster, nicht ein Kasten daneben

Der Navball-Block hat vier freie Felder, und sie sind bereits gerastert:
über der ORB-Flanke, unter dem THR-Streifen, über der ALT-Flanke, unter dem
V/S-Streifen. Alles andere *daneben* zu stellen — erst ein 200×206-Klotz,
dann drei Plättchen mit eigenem Dock-Abstand — lässt den Block als Navball
*mit Anhängsel* lesen und lässt gleichzeitig diese vier Felder leer.

| Plättchen | Feld | Inhalt |
| --- | --- | --- |
| `ManeuverBurnBlock` (76 hoch) | über ORB | DV, Brenndauer, Countdown, `EXECUTE` |
| `ManeuverAxesBlock` (40 hoch) | unter THR | die beiden Δv-Achsen als **Eingabefelder** |
| `ManeuverPlanBlock` (48 hoch) | über ALT | Reichweite der Vorschau |
| `ManeuverNodesBar` (40 hoch) | unter V/S | Knotenwahl, `+ NODE` / `DEL` |

**Breite und x-Lage kommen aus dem Navball, nicht von hier.** `layout()`
liest `NavballCluster.flank_rect()` / `strip_rect()`; eigene Zahlen wären
ein zweites Layout für dieselbe Flanke, und schon eine geänderte `BOX_H`
oder UI-Skala ließe die Plättchen danebenstehen.
`tests/maneuver_hud_test.py` §1 vergleicht x und Breite gegen genau diese
Rechtecke.

Oben sehen sie aus wie eine **Flanke** (Doppelrahmen, Beschriftung im
Kasten — die Flanken tragen ihr `ORB` auch innen, keinen Notch-Tab), unten
wie ein **Streifen** (flache, versenkte Platte). Nach außen zeigende Ecken
gefast, die zur Kugel zeigenden scharf — dieselbe Regel, nach der schon die
Flanken angesetzt statt danebengestellt aussehen.

**Die untere Höhe ist gedeckelt, und zwar gerechnet.** Der
`ORBITAL.INFO`-Block ist nur um eine halbe Flankenbreite eingerückt, ragt
also unter *beide* Flanken und beginnt 47 Einheiten unter dem Streifen. Bei
42 Einheiten Höhe lag die Platte 1 px darauf — `layout()` klemmt deshalb
gegen `info_rect()`, statt sich auf die 40 zu verlassen.

## Die Δv-Zeilen sind Eingabefelder, keine Schrittknöpfe

Vier Pfeilknöpfe je Achse fraßen 60 der 124 Einheiten Breite, und wer
1900 m/s einstellen will, klickt 190-mal. Angeklickt nimmt die Zeile die
Tastatur und man tippt die Zahl hinein — **Ziffern und Punkt, sonst
nichts**, an Ort und Stelle statt in einem Dialog woanders.

Vier Entscheidungen daran sind nicht offensichtlich:

1. **Die Richtung ist die Beschriftung, kein tippbares Minus.** `PRO`
   schaltet auf `RET`, `NRM` auf `ANM`. Deshalb ist der kleinste tippbare
   Wert 0.0 — ein Minus gäbe es sonst zweimal, einmal als Zeichen und einmal
   als Knopf. Das Vorzeichen des gespeicherten Werts bleibt die Wahrheit;
   `_sign` ist nur das Gedächtnis für den Fall 0.0, wo es keines gibt.
2. **Das Feld öffnet MIT dem stehenden Wert, aber als Ganzes markiert.**
   Beide reinen Formen sind falsch. Leer geöffnet zeigte die Zeile `0`,
   während der Knoten noch 250 trug — das liest sich als *zurückgesetzt*,
   und wer nur eine Stelle ändern wollte, muss die Zahl neu tippen. Bloß
   vorbelegt hängt man dagegen an: aus `0.0` und getippten `250.5` wurde
   `0.02505`, und der Punkt, den man tippt, war schon vergeben.

   `_select_all` trägt beides zugleich: die **erste Ziffer ersetzt** den
   Wert als Ganzes (angehängt wäre aus `250` und getippter `3` das `2503`),
   **Pfeil oder Klick steigen ein** und setzen den Caret. `_edit_text()`
   schneidet die tote `.0` ab — eine Stelle, die man erst wegzulöschen hat,
   um eine ganze Zahl zu ändern, wäre im Weg.
3. **Es ist ein richtiges Textfeld: Caret, Pfeile, Einfügen an der Stelle.**
   `←` `→` `Pos1` `Ende` bewegen die Schreibstelle (auf der Markierung
   setzen sie den Caret an deren Rand), `Rückschritt` löscht davor, `Entf`
   darunter, getippt wird **an** der Stelle statt am Ende. Ein Klick ins
   schon offene Feld setzt den Caret dorthin, wo er hinzeigt —
   `_caret_from_x` misst dafür die **Präfixe desselben Textes**, denn der
   Anzeigegrad ist tabellarisch, der Punkt aber nicht.

   **Die Längenprüfung auf `event.unicode` ist load-bearing.**
   `'' in '0123456789'` ist **wahr** — ein Teilstring-Test, kein
   Zeichentest — und jede Taste ohne Zeichen (Pfeile, Umschalt, F-Tasten)
   liefert `''`. Ohne `len(char) != 1` zählte der Pfeil als Ziffer, warf die
   Markierung weg und hängte nichts an: das Feld stand nach dem ersten
   Pfeildruck auf leer und schrieb beim Schließen 0. `tests/maneuver_hud_test.py`
   §4 hält genau das fest.
4. **Zwei Zustände, zwei Zeichen — und beide sind gemessen.** Markiert: ein
   Band in der Achsenfarbe unter dem ganzen Wert (ohne es sah es aus, als
   lösche das Feld die Zahl von selbst). Sonst: ein blinkender Caret
   (`_CARET_BLINK_S`, 60 % hell; jeder Anschlag setzt die Phase auf 0) in
   derselben Achsenfarbe, **mittig auf der Zeichengrenze**. In
   `palette.text`, einen Pixel breit und auf der Schreibstelle beginnend,
   unterschied er sich vom Bild ohne ihn in genau **vier Pixeln** — er lag
   unter dem weißen Stamm der nächsten Ziffer und war dieselbe Farbe. In der
   Achsenfarbe, 1.5 Einheiten breit und in die Lücke gerückt: **30 Pixel**,
   dunkler Grund auf beiden Seiten. Keine neue Farbe kommt dazu; beide
   Zeichen tragen die Farbe ihrer Achse.
3. **Geschrieben wird beim ABSCHLUSS, nicht bei jedem Anschlag.** Jeder
   Anschlag ist ein `plan.touch()`, und wer `25000` tippt, lässt die
   Vorschaukette auch die Zwischenstufen `2`, `25`, `250` und `2500`
   rechnen — die teuerste davon ist die letzte vor der gewollten, und ein
   vertipptes `250000` hängt die Eingabe an einem Brennbogen fest, den
   niemand sehen wollte. `_commit()` ist deshalb der **einzige** Schreibweg
   (Enter, Tab, Klick woanders, Fokusverlust); `Escape` verwirft, ohne dass
   irgendetwas zurückzunehmen wäre.

   Der Riegel ist `_touched`, **nicht** der Puffer: ein leerer Puffer ohne
   jeden Anschlag heißt *nichts gesagt* — wer anklickt und weggeht, schreibt
   nicht, und ein Feld, das dabei auf 0.0 spränge, wäre eine Falle. Ein
   leerer Puffer *nach* dem Weglöschen der Stellen ist dagegen die Eingabe
   `0.0`. `tests/maneuver_hud_test.py` §4 prüft beide Hälften, samt der
   Gegenprobe, dass `plan.version` während des Tippens **still steht**.

   Folge fürs Mausrad: `on_wheel` schließt **erst** und liest **dann** den
   Wert. Andersherum schriebe es den alten Wert plus einen Schritt zurück
   und der getippte wäre weg.
5. **Die Tastatur wird nur genommen, wenn sie gebraucht wird.**
   `takes_keyboard` ist eine *Eigenschaft* (Hover über einem Wertfeld oder
   offenes Feld), keine Konstante: `UIRoot` fragt sie im Moment des Klicks
   ab und setzt danach den Fokus. Ein festes `True` hieße, dass ein Klick
   irgendwohin auf das Plättchen `N` / `X` / `WASD` verschluckt, bis man
   woanders hinklickt. Buchstaben werden im offenen Feld **verschluckt**,
   nicht durchgereicht — sonst setzte ein `n` beim Tippen einen Knoten.

Das Gesamt-Δv ist **nicht** tippbar: es ist die Wurzel aus beiden Achsen,
also ein Ergebnis. Es steht darum im BURN-Kasten, bei Brenndauer und
Countdown, wo alles andere auch abgeleitet ist. Das Mausrad über einer Zeile
stellt weiterhin einen Feinschritt — der schnelle Griff, den die Pfeilknöpfe
hatten, ohne ihre Breite.

## Was in 124 Einheiten Breite passt, ist gemessen

Innenbreite: 102 im gerahmten Kasten, 108 in der flachen Platte. Dagegen
(im `value`-Grad): `1907.6m/s` = 91, `T-1d 02:55:56` = **125**,
`00:03:11` = 79, `PRO` = 24, `m/s` = 25.

Zwei Folgen daraus:

- **`units.countdown_compact`** — die volle Form passt nur ohne ihre
  Beschriftung, also ausgerechnet ohne das, was `NODE` von `IGN`
  unterscheidet. Die Genauigkeit wandert deshalb mit der Größenordnung:
  `T-1y 5d`, `T-1d 02h`, `T-02:55`, `T-02:05`. Über einem Tag sagen
  Sekunden nichts, unter einer Stunde sind sie das Einzige, worauf es
  ankommt.
- **Die Δv-Zeilen tragen gar keine Einheit.** `m/s` kostet 25 der 70 px
  Wertspalte und passte damit nur, wenn *beide* Zahlen gerade klein sind —
  eine Einheit, die je nach Wert erscheint und verschwindet, ist schlechter
  als keine. Die `DV`-Zeile zwei Plättchen weiter oben, in derselben
  Spalte, trägt sie. Der Schriftgrad bleibt aus demselben Grund in beiden
  Zeilen fest, während die abgeleiteten Zeilen im BURN-Kasten die übliche
  dreistufige Verkürzung von `NavballCluster._strip` benutzen.

## Der Countdown wird ROT, sobald er auf `T+` springt

Ein Knoten in der Vergangenheit ist nicht mehr anfliegbar — die Kette
rechnet ihn noch, aber das Schiff ist vorbei. In Amber, der Farbe jedes
anderen Brennwertes, unterscheidet sich dieser Zustand nur durch ein
Vorzeichen zwei Zeichen weit links. `palette.danger` ist dafür da und ist
**keine fünfte Bedeutungsfarbe**: die vier Akzente tragen Bedeutung, `danger`
trägt einen Zustand — wie `disabled`, das direkt daneben steht.

## Farben: keine fünfte

`ROLE_INDEX` bekommt `'node': 2` (Amber, Energie — ein Brennvorgang) und
`'node_path': 3` (Grün, das Versprechen des Autopiloten). Die Griffe tragen
Grün für ±prograde und Magenta für ±normal — dieselben, die die Rosette den
beiden Achsen gibt. `render/maneuver.py` hält die drei RGB-Tripel als Zahlen
(der Renderer kennt die HUD-Palette nicht) und sie sind **wortgleich** zu
`ui/theme.py::SCHEME`.

## Bekannte Grenze

Knoten 5 sitzt auf einer Bahn, die schon vier Integrationen tief ist. Die
Kette wird bei jeder Änderung der Grundbahn neu aufgebaut, was den Drift
begrenzt — aber das Ende eines fünfgliedrigen Mehrjahresplans ist eine
Näherung. Das gehört zur Sache, nicht zu den Fehlern.
