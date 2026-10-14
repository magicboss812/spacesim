In diesem vorletzten Kapitel wird in die Entwicklung der Simulation genauer vertieft, und die konkreten physikalischen sowie teilweise technischen Aspekte hervorgehoben. Wofür und wie die Simulation zuerst als Konzept entstand, welche Hierarchie die Funktionen verfolgen, wie mehrere Teilschritte zu mehreren verzweigten Problemen mit der Genauigkeit und der numerischen Integration führten und wie ein wechselbares Bezugssystem erstmals umgesetzt wurde und nun aussieht, wird in den folgenden Unterkapiteln ausführlich erläutert.

## 5.1 Zielsetzung

Die Konzeption ist mit einem erstmals groben KI-Prototyp etabliert worden. Kernkonzept war ein n-köperphysikalisches 2D Simulationsspiel mit starker Anlehnung an dem Smartphone sowie Computerspiel, "SpaceFlight Simulator", welches in vereinfachter Physik ganz akkurat die Navigation und das Reisen in einem Zweikörpersystem nachbauen lässt. Die Simulation rechnet dementsprechend in N-Körperschleifen zwischen dem Schiff und den Planeten und folgte am Anfang genau diesem Kerngedanke, um unnötige Prozessorresourcen zu sparen. Ebenso ist das Spiel im Zweidimensionalen und nicht im Dreidimensionalen, da von oben betrachtet das gesamte Sonnensystem weitestgehend auf einer Ebene um die Sonne kreist. Die weitere Vektorkomponente würde weitere irrelevante Berechnungen zur Simulation beifügen, die bei der theoretischen Rechnung in Kapitel 3 und 4 sowieso aus der schon beschriebenen Geometriekonstellation herausgenommen wurde. 

Abschließend sind folgende Schwerpunkte für das Konzept konkret relevant geworden: N-Körperschleifen zwischen Planeten und dem Raumschiff; umfangreiche Weltraumerkundung mit realen Maßstäben sowie freier Benutzerfestlegung von eigenen oder realen Sternsystemen; präzise Vorhersage der Schiffsposition; Orbitlinien von Planeten; flexibles & komplexes Set an festgelegten Bezugssystemen einzelner Planeten & Monde; globale Vorausplanung interplanetarer Reisen. 

Auf diese einzelnen Schwerpunkte wird in 5.3 intensiver eingegangen. Das Spiel hat absichtlich keine Landungen, Errungenschaften und Vereinfachungen der Physik, sodass die vollständige simulative Erfahrung problemlos übermittelt werden kann. Landungen benötigen eine Kollisionsgeometrie von jedem Planeten und weiteren speziellen Bedingungen, die für den rechnerischen Zweck der Simulation erstmals nicht notwendig waren und noch sind.

## 5.2 Architektur und Hierarchie

Die Simulation ist nun endgültig in Python implementiert worden. Die Ordnerstruktur ist in dieser Einteilung aufgebaut:
spacesim/
├── bodies/
├── config/
├── physics/
│   └── kernels/
├── render/
│   └── gl/
├── runtime/
├── ship/
│   ├── maneuver/
│   └── predictor/
├── tests/
├── tools/
└── ui/
    ├── assets/
    ├── hud/
    └── widgets/
*Ordnerhierarchie von "/spacesim/", Stammordner des gesamten Spiels*

Jeder Ordner beinhaltet einzelne Python-Dateien die ihren eigenen Zweck gezielt erfüllen und keinen weiteren. Die Hauptphysik ist überwiegend nur im `physics` und `ship` Ordner, wobei sie vorher in einzelne Dateien geclustert waren, mehr dazu in 5.3. Im Konfigurationsordner `config` liegen die Konfigurationen für das Sonnensystem und für alle relevanten Parameter aus dem Spielcode, wie z.B. Physiktoleranzen, Planetenrendering, usw.. `render` und `ui` übernehmen die visuelle Arbeit im Spiel und tragen kaum einen wichtigen Beitrag zur Seminararbeit, außer der visuellen Ausgabe.  `runtime` baut das Spielfenster von Grund auf und erzeugt so die grundlegende Engine des Spiels bei jedem Start

## 5.3 Entwicklungsverlauf

Die KI `gpt-[unbekanntes Modell]` hat aus einer übermäßigen Liste an Antworten zu über 100 relevanten Fragen für den Prototyp einen ersten Entwurf erstellt. Dieser Entwurf ist in der Programmiersprache Python geschrieben und wies zuerst viele sichtbare und technische Probleme auf. Planeten- sowie Schiffspositionen litten an einem sogenannten "Floating Point Precision Error", sodass sie in bestimmten Zeitabständen blockig zu anderen Orten in der Nähe sprangen und kein vernünftiger Orbit zu sehen war. Zudem hatte das Schiff keine korrekte realistische Anziehung des Planeten, sondern wurde chaotisch in eine beliebige Richtung geschleudert. Das physikalische Konzept wurde verfolgt, aber erwies sich als unbrauchbar und größtenteils fehlerhaft. Somit wurde der Prototyp später nicht im Kontext der KI jemals wieder verwendet, um die präzise Konzeption nicht zu beeinträchtigen. Anschließend ist die menschliche Konzeption und eine ki-gestützte Weiterführung der Implementation als ein globales Workflow festgesetzt worden. 

Im technischen Konzept war Python ideal wegen seiner möglichen Komplexität aber weiterhin lesbaren Struktur. Das Spiel solle, wie im KI-Prototyp, aus mehreren einzelnen Dateien aufgebaut werden, sodass die Architektur und Fehler klarer durchschaubar sind. Daneben ist eine aufgeteilte Codebase-Struktur eine Voraussetzung für fehlerfreie ki-gestützte Implementierung. Bis zum ersten richtigen Upload im März wurden viele relevante Vektoroperatoren implementiert, die Weltphysik, die Keplerbahnen der Planeten, das Schiff als einziger Punkt, die erste Rendering Pipeline, die dynamische Bildschirmkamera und der Konfigurationsinitiator aufgebaut, und die Schiffvorhersage als eine teilweise funktionierende Erstversion konzipiert und implementiert. Die ersten Hauptprobleme waren die Integratoren von jeweils der Weltphysik und der Spielervorhersage.

Die Physik in der Simulation wird aus festen zwei-Körperschleifen, dem Schiff und ein Planet n, für jeden Planeten berechnet. Es ergeben sich also 27 Paare aus Schiff und Planet, die alle gleichzeitig jederzeit auf das Schiff wirken. Die Simulation muss anschließend die Gravitationsbeschleunigung und die resultierende Position numerisch integrieren, weil es keine eindeutige Lösung in einem N-Körpersystem gibt. Bei der Simulation wird es partial abgezogen, da nur das Schiff numerisch integriert wird. Um so ein System zu lösen, fehlen bei den Differentialgleichungen relevante Erhaltungsgrößen, die nicht bewiesen werden können, da die Nichtlinearität der sogenannten Differentialgleichungen zweiter Ordnung keine analytische Lösung beinhalten. Somit wird auf numerisches Integrieren zurückgewiesen, bei der pro Zeitschritt die momentane Position und Geschwindigkeit erfasst wird und für den nächsten Schritt neu integriert wird. So häufen sich die Berechnungen irgendwann zu sichtbaren signifikanten Lösungen die stark der eigentlichen Lösung entsprechen. Ein großes Problem dabei ist, wie bei jedem Zeitschritt integriert wird und wie groß diese Zeitschritte (jeweils) sind. Je kleiner die Zeitschritte sind, desto genauer ergibt sich die numerische Lösung die der richtigen immer näher kommt. Pro Zeitschritt können sich aber diese Fehler häufen und werden zum nächsten Zeitschritt weiter getragen. Nach mehreren Orbits kann sich also die numerische Lösung von der tatsächlichen so stark unterscheiden, dass von einem "Energiedrift" gesprochen wird, weil die Umlaufbahn selbst bei einem Zweikörpersystem aus unbekannter Herkunft Energie dazu bekommt oder abgibt, obwohl es eigentlich gar nicht soll.