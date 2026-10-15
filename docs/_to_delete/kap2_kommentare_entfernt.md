%%
Am 08.10.2026 aus 2.4 herausgenommen (Kürzung auf Abb. 5, Begründung fehlender geschlossener Lösung, Patched Conics und Einflusssphäre, numerische Integration mit Abb. 6, Spielbezug). Gleichungs- und Abbildungsnummern im Folgenden sind die alten.

Poincaré: Henri Poincaré zeigte in einer 1890 veröffentlichten Arbeit, dass sich die Gleichungen des Dreikörperproblems nicht integrieren lassen, und beschrieb darin erstmals mathematisch eine chaotische Bewegung [29, Abschn. 1.1] [32].

Eingeschränktes Dreikörperproblem mit alter Gl. (20):

Für ein Raumfahrzeug vereinfacht sich Gl. [[#^eq-nkoerper|(19)]], da seine Masse gegenüber der von Sonne und Planeten verschwindend klein ist. Es wird von allen Himmelskörpern angezogen, wirkt aber selbst nicht merklich auf deren Bewegung zurück. Im einfachsten Fall umlaufen zwei Himmelskörper ihren gemeinsamen Schwerpunkt auf Kreisbahnen, und ein dritter Körper mit vernachlässigbarer Masse bewegt sich in ihrem Feld, etwa ein Raumfahrzeug im System aus Erde und Mond oder aus Sonne und einem Planeten. Selbst dieses eingeschränkte Dreikörperproblem besitzt keine allgemeine geschlossene Lösung [26]. Gibt man die Bahnen $\vec{R}_k(t)$ aller Himmelskörper vor, bleibt allein die Bewegungsgleichung des Raumfahrzeugs mit dem Ortsvektor $\vec{R}$ zu lösen, in der jeder Himmelskörper wie in Abschnitt 2.1 über seinen Gravitationsparameter $\mu_k$ eingeht:

$$\ddot{\vec{R}} = -\sum_{k} \mu_k\,\frac{\vec{R} - \vec{R}_k(t)}{\left|\vec{R} - \vec{R}_k(t)\right|^3}, \qquad \mu_k = G\,m_k\tag{20}$$
^eq-eingeschraenkt

Gl. [[#^eq-eingeschraenkt|(20)]] folgt aus Gl. [[#^eq-nkoerper|(19)]], wenn die Masse des Raumfahrzeugs gegen null geht.

v∞ an der Einflusssphäre: Beim Verlassen der Einflusssphäre wird die Geschwindigkeit des Raumfahrzeugs zur Anfangsgeschwindigkeit seiner Bahn um die Sonne [28]. Ihr Unterschied zur Geschwindigkeit des Planeten ist die hyperbolische Exzessgeschwindigkeit $v_\infty$ nach Gl. [[#^eq-exzess|(14)]], mit der sich das Raumfahrzeug vom Planeten entfernt [33].

Energiedrift als Genauigkeitsmaß (für Kapitel 5): Die grobe Bahn in Abb. 6 spiralt nach außen, sodass ihre Bahnenergie nach Gl. [[#^eq-energie|(9)]] zunimmt, obwohl sie nach Abschnitt 2.2 konstant bleiben müsste. Wie weit die Energie einer berechneten Bahn von ihrem Anfangswert abweicht, ist deshalb ein Maß für die Genauigkeit einer numerischen Integration. Welches Verfahren die Simulation verwendet und wie genau sie nach diesem Maß rechnet, beschreibt Kapitel 5.

Alter Spielbezug: In der verwendeten Simulation wird das Raumfahrzeug als Körper ohne Masse nach Gl. [[#^eq-eingeschraenkt|(20)]] numerisch integriert, und die vorausberechnete Bahn in der Anzeige ist das Ergebnis dieser Integration.

![[Obsidian Vault/Evaluation/Grafiken & Bilder/SCREENSHOT.png|700]]
*Abb. 7, Raumfahrzeug auf hoher Erdumlaufbahn (weiß) mit Manöverknoten nahe der Periapsis und geplanter Bahn nach dem Manöver (grün), außen die Mondbahn. An der Bahn Apoapsis und Periapsis der numerischen Integration, am unteren Rand $\Delta v$, Brenndauer sowie Apoapsis und Periapsis nach der Zweikörpernäherung. Das Symbol des Raumfahrzeugs ist nicht maßstäblich – eigener Screenshot*

Abb. 7 zeigt ein Raumfahrzeug auf einer hohen Erdumlaufbahn (weiß), außen verläuft als schwacher Kreis die Bahn des Mondes. Die Markierungen Ap und Pe auf der weißen Linie geben den größten und kleinsten Abstand zum Erdmittelpunkt auf der numerisch integrierten Bahn an. Die Anzeige am unteren Bildrand berechnet Apoapsis und Periapsis dagegen nach Abschnitt 2.1 allein aus dem momentanen Ort und der Geschwindigkeit, als gäbe es nur die Erde als Zentralkörper. Die beiden Apoapsiswerte unterscheiden sich um rund $3\ \mathrm{Mm}$ ($150{,}95$ gegenüber $147{,}87\ \mathrm{Mm}$), da die übrigen Himmelskörper, vor allem der Mond, die tatsächliche Bahn verändern. An diesem Unterschied wird unmittelbar sichtbar, wie weit die Zweikörpernäherung von der numerisch berechneten Bahn abweicht.

Alter Schlussabsatz zur Leitfrage: Für die Leitfrage werden beide Wege gebraucht. Die Patched-Conics-Näherung liefert den Geschwindigkeitsaufwand eines Transfers oder Gravitationsmanövers als Rechenwert, die numerische Integration spielt dieselbe Mission unter dem gleichzeitigen Einfluss aller Körper durch. Da nach Abschnitt 2.3 der Geschwindigkeitsaufwand $\Delta v$ das Maß ist, an dem die Missionsvarianten verglichen werden, zeigt der Vergleich beider Wege, wie weit die Näherung trägt und welchen Geschwindigkeitsaufwand die Missionsvarianten im Modell der Simulation erfordern.
%%

%%
Material für Kapitel 5, am 06.10.2026 aus 2.4 herausgenommen (Aufbau der Simulation, Manöverknoten, Zündzeitpunkt, fehlendes Massenmodell):

In der verwendeten Simulation bewegt sich das Raumfahrzeug nach Gl. [[#^eq-eingeschraenkt|(20)]] im Feld von 27 Himmelskörpern, der Sonne, acht Planeten, Pluto und 17 Monden. Diese folgen vorgegebenen Kepler-Bahnen, Monde relativ zu ihrem Planeten, und stören einander nicht, während das Raumfahrzeug selbst die Masse null hat. Seine Beschleunigung wird in jedem Zeitschritt aus den Orten aller Himmelskörper zu genau diesem Zeitpunkt aufsummiert, und die vorausberechnete Bahn in der Anzeige ist das Ergebnis dieser numerischen Integration.

Brennvorgänge berechnet die Simulation wie in Abschnitt 2.3 beschrieben numerisch mit Schub. Manöver werden über Manöverknoten auf der vorausberechneten Bahn geplant, von denen jeder eine Geschwindigkeitsänderung in Flugrichtung und eine senkrecht dazu trägt. In Abb. 7 liegt ein Knoten nahe der Periapsis, wo ein Schub nach dem Oberth-Effekt am meisten Bahnenergie liefert, und trägt $\Delta v = 301{,}1\ \mathrm{m/s}$ in Flugrichtung. Die grüne Linie zeigt die geplante Bahn nach dem Manöver, deren Apoapsis wie in Abb. 4 nach außen rückt. Sie reicht bis in die Nähe des Mondes, dessen Ort zum Ende der Planung als grüner Kreis markiert ist, und biegt dort sichtbar ab.

Die Anzeige gibt zum Manöver den Betrag $\Delta v$ und eine Brenndauer von $30\ \mathrm{s}$ an. Das Triebwerk zündet um die halbe Brenndauer vor dem Knoten, sodass der Knoten in der Mitte des Brennvorgangs liegt und dem Ort des impulsiven Manövers entspricht. Masse und Treibstoff bildet die Simulation nicht ab. Für den Vergleich der Missionsvarianten genügt das, da der Geschwindigkeitsaufwand nach Gl. [[#^eq-deltav|(15)]] und [[#^eq-deltav-ges|(16)]] nicht von der Masse abhängt. Der Schub wird direkt als Beschleunigung vorgegeben, und der Aufwand eines Manövers wird unmittelbar in $\Delta v$ gemessen.
%%
