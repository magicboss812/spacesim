Ich lade Plan, Konventionen und die Kapitel-5-Datei.

## Check 5.1 Zielsetzung

**Urteil:** Inhaltlich trägt 5.1 das Ziel, die Reihenfolge aber nicht. Der Zweck für die Arbeit steht erst im dritten Absatz, Z. 13 („Aus beiden Zwecken“) springt über den Konzeptkatalog in Z. 11 zurück auf Z. 9, und Z. 15 gehört zur Architektur. Am schwersten wiegt, dass Z. 5 Principia als Lösung des KSP-Problems nennt und offenlässt, warum trotzdem ein eigenes Spiel nötig war.

### Umfang
436 Wörter, ca. 1,1 Seiten. Dazu kommt die Kapiteleinleitung in Z. 1 mit 77 Wörtern (ca. 0,2 Seiten). Keine Gleichungen, keine Abbildungen. Der Plan legt kein Budget für 5.1 fest. Kapitel 5 hat insgesamt rund 2960 Wörter (ca. 8 Seiten), das sind derzeit etwa 23 % von Kapitel 2, 3 und 5 zusammen. Das entspricht dem Orientierungswert aus dem Plan, und mit Kapitel 4 sinkt der Anteil noch. Kürzen ist also nicht nötig.

### Struktur von oben
- Z. 1: Vorschau auf das Kapitel. Die Überschrift `# 5. Entwicklung der Simulation` fehlt, obwohl `02` und `03` mit H1 beginnen. „verzweigten Problemen“ ist unklar.
- Z. 5: KSP scheidet aus, Entscheidung für das eigene Spiel, persönlicher Erkenntnisgewinn, poliastro/astropy. Der letzte Punkt ist Entwicklungshistorie und hier ein Einschub.
- Z. 7: KI-Prototyp und Vorbild Spaceflight Simulator. Mit „Am Anfang des Konzepts“ springt die Chronologie zurück.
- Z. 9: Zweck für die Arbeit (Sätze 1–3) und Zweck als Spiel (Sätze 4–5).
- Z. 11: Katalog der Schwerpunkte mit fünf Gliedern.
- Z. 13: Zielkonflikt. Bezieht sich auf Z. 9, nicht auf Z. 11 (Rücksprung).
- Z. 15: Python und eigene Engine. Das ist eine Architekturentscheidung (Einschub).
- Z. 17: Vorschau auf 5.3 und 5.4, die Z. 1 schon enthält.

Formal gibt es keine Probleme.

### Roter Faden
Eine Zielsetzung beginnt mit dem Zweck. Hier steht zuerst die Werkzeugwahl, und der Zweck kommt erst danach. Vorgeschlagene Reihenfolge:
1. Z. 9, Sätze 1–3 (Zweck für die Leitfrage)
2. Z. 5 ohne den poliastro-Satz, ergänzt um die Begründung gegen Principia
3. Z. 7
4. Z. 11
5. Z. 9, Sätze 4–5, direkt gefolgt von Z. 13 (beide Zwecke, dann der Konflikt zwischen ihnen)
6. Z. 15 nach 5.2 verschieben, den poliastro-Satz nach 5.3 (Z. 50)
7. Z. 17 durch eine Überleitung zu 5.2 ersetzen

### Inhalt
Fehler:
- Z. 13, „von der exakten Lösung abweicht“: Nach 2.4 gibt es keine exakte Lösung. Gemeint ist der geschätzte Fehler pro Schritt, also die Toleranz aus Z. 60. Vorschlag: „… so genau integrieren, dass der geschätzte Fehler jedes Rechenschritts in der Position unter einem Meter bleibt.“ Eine Schrittvorgabe sagt außerdem nichts über den aufsummierten Fehler über einen ganzen Transfer. Den braucht die Problemfrage (siehe 5.3, Fehlende Inhalte).

Schwächen:
- Z. 5: Die Entscheidung folgt so nicht aus dem vorigen Satz, denn mit Principia integriert KSP ebenfalls numerisch.
- Z. 9, Satz 2: Die Aussage zu Gravitationsmanövern ist unbelegt. Entweder auf Kapitel 4 verweisen oder [28]/[36] zitieren, falls diese Seiten die Grenzen der Näherung nennen (vorher prüfen).
- Z. 7, „nach eigener Beobachtung“: grenzwertig nach der Regel, sich selbst nicht zu erwähnen. Zudem unbelegt und vage („recht genau“).
- Z. 5, „brachte zudem deutlich mehr Einsicht“, und Z. 15, „auch wenn dieses Nötigste bereits sehr umfangreich ist“: persönliche Wertungen ohne Funktion im Argument.
- Quellen: [44] wird in 5.1 zuerst genannt, danach folgen [40], [43], [42], [41], [45]. Falls das Verzeichnis nach erster Nennung nummeriert ist, stimmt die Reihenfolge nicht.

### Fehlende Inhalte
Es fehlt die Begründung, warum Principia nicht reicht. Das stärkste Argument steht schon im eigenen Text: Die Einleitung (letzter Absatz) und 5.4 (Z. 90) nennen die beliebig wiederholbare Mission unter gleichen Bedingungen. Dieses Argument nach Z. 5 übernehmen und ergänzen, was sonst zutrifft (2D, volle Kontrolle über Konfiguration und Messwerte). Das kann nur der Autor beurteilen, eine externe Quelle braucht es dafür nicht.

### Ende
5.1 endet mit einer doppelten Vorschau auf 5.3/5.4 und überspringt 5.2. Ein Satz, der zur Architektur überleitet, reicht als Abschluss.

### Prioritäten
1. Begründung gegen Principia in Z. 5 ergänzen.
2. Absätze umstellen wie oben vorgeschlagen, Z. 15 nach 5.2 und den poliastro-Satz nach 5.3 verschieben.
3. „exakte Lösung“ in Z. 13 korrigieren.
4. Z. 17 durch eine Überleitung zu 5.2 ersetzen und die H1 setzen.
5. Z. 9, Satz 2 belegen oder auf Kapitel 4 verweisen.

---

## Check 5.2 Architektur und Hierarchie

**Urteil:** In der jetzigen Form trägt 5.2 keinen eigenen Abschnitt. Der Titel verspricht eine „Hierarchie“, die nirgends erklärt wird. Der Verweis auf 5.3 widerspricht dem, was 5.3 tatsächlich sagt.

### Umfang
100 Wörter, ca. 0,3 Seiten. Mit Code 1 (19 Zeilen, ca. 0,35 Seiten) kommt der Abschnitt auf etwa 0,6 Seiten. Kein Budget im Plan.

### Struktur von oben
- Z. 21: Einleitungssatz zu Code 1.
- Z. 23–42: Code 1 mit Unterschrift.
- Z. 44: Ordner einzeln beschrieben, aber nur `physics`, `ship`, `config`, `render`, `ui`, `runtime`.

Formal sauber.

### Roter Faden
5.2 hat keinen Einstieg aus 5.1 und keinen Ausgang nach 5.3. Inhaltlich gehören zwei Stellen hierher: Z. 15 aus 5.1 (Python, eigene Engine) und der Absatz „Eigene Sternsysteme“ (Z. 76) aus 5.3, der `config` und `bodies` beschreibt und mit Entwicklungsverlauf nichts zu tun hat.

### Inhalt
Fehler:
- Z. 44, „früher war sie auf wenige große Dateien verteilt (Abschnitt 5.3)“: 5.3 behandelt das nicht. Z. 50 sagt sogar, das Spiel habe wie der Prototyp von Anfang an aus vielen Dateien bestehen sollen. Entweder den Umbau in 5.3 ergänzen oder den Halbsatz streichen.

Schwächen:
- Z. 44, „einzelne Python-Dateien, die jeweils genau einen Zweck erfüllen“: Das gilt nicht für `ui/assets`.
- Gerade die Ordner, die die Arbeit braucht, bleiben unerklärt: `physics/kernels` (Numba, Z. 58), `ship/predictor` und `ship/maneuver` (5.3/5.4), `bodies` (Kepler-Bahnen) sowie `tests` und `tools`.

### Fehlende Inhalte
- Der Ablauf eines Bilds durch die Module, also Schiffsphysik, Vorhersage und Darstellung. Das wird gebraucht, weil 5.3 (Z. 56, Z. 62) Schiffsphysik und Vorhersage als getrennte Rechnungen mit eigenen Integratoren behandelt, ohne dass der Leser weiß, dass es zwei Komponenten sind. Beleg ist der eigene Code.
- Die Bedeutung von „Hierarchie“. Entweder die Modulabhängigkeiten erklären oder die Körperhierarchie Sonne → Planet → Mond (Z. 76/86). Sonst den Titel auf „Architektur“ kürzen.

### Ende
Der Abschnitt bricht nach der Beschreibung von `runtime` ab.

**Entscheidung:** 5.2 ausbauen (ca. +0,3 bis 0,5 Seiten, mit Ablauf eines Bilds, Z. 15 und Z. 76) oder ganz auflösen (Code 1 an den Anfang von 5.3, Z. 44 auf zwei Sätze kürzen). Ausbauen macht 5.3 verständlicher. Auflösen spart einen Abschnitt, aber die Trennung zwischen Physik und Vorhersage muss dann in 5.3 erklärt werden.

### Prioritäten
1. Den falschen Verweis auf 5.3 in Z. 44 beheben.
2. Entscheiden, ob 5.2 ausgebaut oder aufgelöst wird.
3. Beim Ausbau den Ablauf eines Bilds ergänzen und Z. 15 sowie Z. 76 hierher verschieben.
4. „Hierarchie“ erklären oder aus dem Titel streichen.

---

## Check 5.3 Entwicklungsverlauf

**Urteil:** Die Chronologie Verlet → RK4 → Fehlerkontrolle → RKN4 trägt den Abschnitt. Am Ende springt sie aber zweimal zurück (Yoshida, fortlaufende Vorhersage), und „Eigene Sternsysteme“ ist ein Fremdkörper. Größtes Problem: Die Energiedrift ist falsch definiert, und die Genauigkeit wird nur qualitativ angegeben. 2.4 verspricht aber „wie genau sie rechnet“, und die Problemfrage hängt daran.

### Umfang
1697 Wörter, ca. 4,5 Seiten, ohne Gleichungen und Abbildungen. Das sind 57 % von Kapitel 5. Kein Budget im Plan.

### Struktur von oben
- Z. 48: KI-Prototyp, seine Fehler, Arbeitsablauf.
- Z. 50: Entscheidung für die Dateistruktur, Stand am 1. März, Gleitkommagenauigkeit, Überleitung zu den Integratoren.
- Z. 54 (211 W): Theorie der numerischen Integration und Energiedrift. Rücksprung in die Theorie, und vieles davon steht schon in 2.4.
- Z. 56: Verlet, scheinbare Drift, Vorhersage mit eigener Schrittweite.
- Z. 58: RK4, Ordnung, Rechenlast, Numba.
- Z. 60 (210 W): Principia, RK45, Schrittverdopplung, Toleranzen.
- Z. 62 (222 W): RKN4, Umsetzungsfehler, Euler-Satz (Einschub), Yoshida (zeitlicher Rücksprung).
- Z. 66: Obergrenze der Schrittweite im Zeitraffer.
- Z. 70: Fehler im Bezugssystem und seine Behebung.
- Z. 72: fortlaufende Vorhersage. Zeitlich liegt sie vor Z. 70 (Rücksprung).
- Z. 76: Konfigurationsdatei (gehört nach 5.2).

Drei Absätze liegen über 180 Wörtern (Z. 54, 60, 62).

### Roter Faden
- Z. 50 endet mit dem Problem der Integratoren, Z. 54 beginnt danach mit Theorie. Z. 54 wiederholt 2.4 (Grund für die numerische Integration, Schrittweite, Abb. 6). Neu ist dort nur die Fehlerfortpflanzung mit der Energiedrift. Kürzen auf einen Verweis auf 2.4 und diesen neuen Teil.
- Z. 62, „Das Verfahren, das die Simulation dafür verwendet“: Das „dafür“ hat keinen Bezug. Außerdem bleibt offen, ob die Schrittverdopplung aus Z. 60 seit Mai mit RKN4 läuft.
- Z. 66 löst den Zielkonflikt aus 5.1 Z. 13, sagt das aber nicht. Er knüpft an das Ende von Z. 60 an und gehört direkt dahinter.
- Z. 76 wiederholt zum zweiten Mal, dass die Körper auf Kepler-Bahnen laufen (Z. 54, Z. 86).

Vorgeschlagene Reihenfolge:
1. Z. 48
2. Z. 50
3. Z. 54, gekürzt
4. Z. 56
5. Z. 58
6. Z. 60
7. Z. 66 mit Rückbezug auf 5.1
8. Yoshida
9. RKN4 mit Umsetzungsfehler
10. Neuer Absatz zur heutigen Genauigkeit
11. Z. 72
12. Z. 70
13. Schluss mit Überleitung zu 5.4

Z. 76 wandert nach 5.2.

### Inhalt
Fehler:
- **Z. 54, Energiedrift:** Der Text definiert sie über die Größe der Abweichung („so stark … dass man von einer Energiedrift spricht“). Gemeint ist aber die systematische Zu- oder Abnahme der Bahnenergie. Auch „selbst im Zweikörpersystem“ verdreht die Logik. Energieerhaltung gilt *nur* dort. Im Feld bewegter Körper ändert sich die Energie des Schiffs tatsächlich, und genau darauf beruht das Gravitationsmanöver in Kapitel 4. Vorschlag: „Da der Fehler jedes Schritts in den nächsten eingeht, kann sich die berechnete Bahn über viele Umläufe immer weiter von der tatsächlichen entfernen. Im Zweikörperproblem zeigt sich das als Energiedrift, bei der die Bahnenergie nach Gl. (9) stetig zu- oder abnimmt, obwohl sie nach Abschnitt 2.2 erhalten bleibt. Im Feld mehrerer bewegter Körper ist die Energie des Schiffs dagegen nicht erhalten, weshalb sich die Drift nur in einem Zweikörper-Testlauf als Genauigkeitsmaß eignet.“
- **Z. 62, Yoshida:** „deutlich mehr Auswertungen pro Schritt“ stimmt für die vierte Ordnung nicht. Das Verfahren setzt drei Verlet-Schritte zusammen und braucht damit drei Auswertungen, genauso viele wie RKN4. Mehr werden es erst ab Ordnung 6 (sieben Teilschritte) [45]. Entweder die Ordnung nennen oder die Begründung allein auf den Verlust der Symplektizität bei angepasster Schrittweite stützen [40].
- Z. 56, „wo sich die Beschleunigung bei fester Schrittweite am schnellsten ändert“: Der Bezug ist falsch. Die Beschleunigung ändert sich an der Periapsis unabhängig von der Schrittweite am schnellsten. Die feste Schrittweite erklärt nur, warum der Fehler dort groß wird.
- Z. 58: „Er ist pro Schritt deutlich kleiner“ und „viermal so hohe Rechenlast“ gelten jeweils nur bei gleicher Schrittweite. Das sollte dastehen.

Schwächen:
- Z. 60, „Ihr wichtigster Gedanke ist die Fehlerkontrolle“: Diese Wertung deckt [44] nicht. Soweit bekannt rechnet Principia die Bahnen von Raumfahrzeugen mit einem eingebetteten Runge-Kutta-Nyström-Verfahren (Dormand, El-Mikkawy, Prince 1986) und anpassbarer Schrittweite. Das wäre ein direkter Beleg für die Wahl von RKN4 in Z. 62, muss aber vorher im Quellcode [44] geprüft werden.
- Z. 60, „mit dem ungenaueren Ergebnis wird der Fehler kontrolliert“: unscharf. Präziser ist, dass die Differenz den Fehler des Ergebnisses vierter Ordnung schätzt und mit dem Ergebnis fünfter Ordnung weitergerechnet wird. Für eine Verfahrensaussage ist die SciPy-Dokumentation [41] schwach. [42] oder [43, Abschn. II.5] sind besser.
- Z. 50, „Voraussetzung für eine möglichst fehlerfreie KI-gestützte Implementierung“: unbelegt und zu stark formuliert. „erleichtert“ trifft es.
- Z. 70, „im absoluten Raum“: Der Begriff ist physikalisch vorbelastet. „im festen, sonnenzentrierten Bezugssystem der Simulation“ ist genauer.
- Z. 70: Die Langsamkeit im heliozentrischen System wird genannt, aber nie aufgelöst.
- Z. 72: Die Ursache („da keine neuen Zwischenstände mehr entstanden“) ist nicht nachvollziehbar.
- Z. 66: „drastisch“ ist umgangssprachlich. Außerdem wechselt das Subjekt von der Simulation zur Vorhersage, ohne dass klar wird, ob die Obergrenze für beide gilt.

Geprüft und korrekt:
- Gleitkommaabstände: $2^{14}\ \mathrm{m} \approx 16\ \mathrm{km}$ bei einfacher und $2^{-15}\ \mathrm{m} \approx 0{,}03\ \mathrm{mm}$ bei doppelter Genauigkeit.
- Faktor 16 und Division durch 15 bei der Schrittverdopplung: $2^4$ beziehungsweise $2^4 - 1$.
- Zeitskala $\sqrt{r^3/(Gm)} = T/2\pi$.
- Zwei Auswertungen von RKN4 am selben Ort.

Die Codeangaben selbst (Toleranzen, Neuberechnung in jedem Bild, Datumsangaben) konnten nicht gegen das Repo geprüft werden, weil der Zugriff auf GitHub hier gesperrt war.

### Fehlende Inhalte
- **Eine Genauigkeitsangabe mit Zahl.** Zum Beispiel ein Zweikörper-Testlauf mit der relativen Energieabweichung nach einer festen Zahl von Umläufen, oder die Abweichung der Periapsis gegenüber der Kepler-Bahn. Das wird dreimal gebraucht. 2.4 verweist für „wie genau sie rechnet“ auf Kapitel 5, die Problemfrage fragt nach „wie genau“, und 3.4 führt Abweichungen von $0{,}07$ bis $0{,}09\ \mathrm{km/s}$ auf Sonne und Mond zurück. Diese Zuordnung hält nur, wenn der Integrationsfehler nachweislich kleiner ist. Die Energiedrift als Maß liegt schon als %%-Material in 2.4 bereit, die Theorie belegt [40]. Falls `tests/` einen solchen Lauf enthält, reicht dessen Ergebnis.
- Ob Schiffsphysik und Vorhersage heute mit demselben Verfahren und derselben Toleranz rechnen. Z. 56 nennt die abweichende Vorhersage als Problem, die Lösung wird nur über Z. 70 angedeutet. Das braucht 3.4, wo die Linie die Vorhersage zeigt und der Flug von der Schiffsphysik gerechnet wird.

**Entscheidung:** Die Kapiteleinleitung nennt Kapitel 5 „vorletztes Kapitel“. Damit gibt es kein eigenes Validierungskapitel wie in der verworfenen Gliederung. Der Genauigkeitsnachweis gehört dann nach 5.3. Mit einem eigenen Kapitel 6 stimmt „vorletztes“ nicht mehr.

### Ende
5.3 endet mit einem Konfigurationsdetail. Es fehlt ein Schlussabsatz, der den heutigen Stand zusammenfasst (RKN4, anpassbare Schrittweite, Toleranz, gemessene Genauigkeit), damit den Zielkonflikt aus 5.1 schließt und zu 5.4 überleitet.

### Prioritäten
1. Die Energiedrift in Z. 54 richtig definieren und auf den Zweikörperfall einschränken.
2. Die Genauigkeit mit einem Messwert belegen.
3. Die Yoshida-Begründung in Z. 62 korrigieren.
4. Umstellen wie vorgeschlagen, Z. 54 auf das Neue gegenüber 2.4 kürzen und das „dafür“ in Z. 62 klären.
5. Die Principia-Aussage in Z. 60 am Quellcode prüfen und RK45 mit [42] oder [43] belegen. Z. 76 verschieben und einen Schluss ergänzen.

---

## Check 5.4 Manöverplanung und Vereinfachungen des Modells

**Urteil:** Das ist der am saubersten gebaute Abschnitt des Kapitels. Erst kommt die Manöverplanung, dann das Modell, und der Schluss bindet beides an den Vergleich der Missionsvarianten. Die Probleme sind eine sachlich falsche Begründung in Z. 82, eine zu absolute Aussage in Z. 90 und die Doppelung mit 5.3.

### Umfang
653 Wörter, ca. 1,7 Seiten, ohne Gleichungen und Abbildungen. Kein Budget im Plan.

### Struktur von oben
- Z. 80: Manöverknoten, Anteile, Gl. (15) und (16), bis zu fünf Knoten.
- Z. 82 (192 W): endliche Brenndauer, Schubprofil, Zündzeitpunkt, feste Richtung, Zahlenbeispiel.
- Z. 84: Masse null, Begründung über Gl. (19) und (18).
- Z. 86: Kepler-Bahnen, nur ein Teil der N-Körper-Physik (Doppelung mit Z. 54 und Z. 76).
- Z. 88: Ebene, Punktmassen, keine Relativität.
- Z. 90: ausgelassene Spielelemente und Schlussgedanke.

Z. 82 lässt sich nach „am nächsten kommt.“ teilen.

### Roter Faden
Die Übergänge stimmen. Auf Kapitelebene gilt: 3.4 nutzt die Simulation vor diesem Abschnitt. Brenndauer und die nur teilweise abgebildete N-Körper-Physik sollten dort einmal auf 5.4 vorverweisen.

### Inhalt
Fehler:
- **Z. 82, „da die Vorhersage eine mitdrehende Richtung nicht im Voraus berechnen könnte“:** Das stimmt so nicht. Eine Richtung, die stets in Flugrichtung zeigt, ergibt sich in jedem Schritt aus Ort und Geschwindigkeit und lässt sich genauso integrieren. Principia bietet solche Brennvorgänge in mitgeführten Bezugssystemen an. Möglicherweise ist der eigentliche Grund ein anderer: Bei fest im Raum stehender Richtung hängt die Beschleunigung weiterhin nur von Ort und Zeit ab. Damit bleibt RKN4 nach der Voraussetzung aus Z. 62 auch während des Brennens anwendbar, während eine mitdrehende Richtung die Geschwindigkeit in die Beschleunigung bringt. Diese Begründung nur übernehmen, wenn sie dem Code entspricht.
- **Z. 90, „vereinfacht die Simulation die Physik an keiner Stelle“:** zu absolut. Die feste Schubrichtung und die vorgegebene Beschleunigung von $10\ \mathrm{m/s^2}$ sind Entwurfsentscheidungen für die Bedienung. Haltbar ist die engere Aussage, dass die Bewegungsgleichung selbst nirgends zugunsten der Spielbarkeit verändert wird.

Schwächen:
- Z. 86 wiederholt Z. 54 und Z. 76. Hier steht die Aussage am richtigen Ort, in 5.3 kann sie auf einen Halbsatz schrumpfen.
- Z. 80 und 5.1 Z. 11: 5.1 verspricht eine „globale Vorausplanung interplanetarer Reisen“, 5.4 beschreibt nur Manöverknoten. Entweder fehlt eine Planungsfunktion (etwa für Phasenwinkel), oder 5.1 verspricht zu viel.

Geprüft und korrekt:
- $3\ \mathrm{km/s} \,/\, 10\ \mathrm{m/s^2} = 300\ \mathrm{s}$ und $300/5400 \cdot 360° = 20°$.
- Die Summe der 27 Körper.
- Die Gleichungsverweise (15), (16), (18), (19) und (1).

### Fehlende Inhalte
Nichts Zwingendes. Die Vereinfachungen decken ab, was 3.4 und Kapitel 4 zur Einordnung brauchen.

### Ende
Z. 90 schließt mit der Wiederholbarkeit der Mission, und das trägt als Abschluss des Abschnitts und des Kapitels. Es fehlt noch ein halber Satz, der zum nächsten Kapitel überleitet.

### Prioritäten
1. Die Begründung für die feste Schubrichtung in Z. 82 korrigieren und am Code prüfen.
2. Die Aussage in Z. 90 auf die Bewegungsgleichung einschränken.
3. Z. 82 teilen.
4. Das Versprechen der „globalen Vorausplanung“ in 5.1 mit 5.4 abgleichen.
5. Eine Überleitung zum Folgekapitel und einen Vorverweis aus 3.4 setzen.