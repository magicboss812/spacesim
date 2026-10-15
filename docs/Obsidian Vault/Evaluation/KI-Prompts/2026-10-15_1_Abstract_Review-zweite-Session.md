## Check Abstract (01 Einleitung, Z. 2–4)

**Urteil:** Die Struktur trägt. Der Abstract stellt die Frage, nennt die Methode, gibt die Ergebnisse beider Teilfragen und der Simulation und zieht einen Schluss. Alle Zahlen stimmen mit 3.3, 3.4, 4.2, 4.4 und dem Fazit überein. Das größte Problem ist, dass er bei der Simulation mehr behauptet als Kapitel 4.4 belegt. Die Ursache wird als sicher dargestellt, die Anpassung des Periapsisabstands in Variante 1 fehlt, und die Modellgrenze aus dem Fazit fehlt ebenfalls. Dadurch klingt der Schlusssatz allgemeingültig.

### Umfang
207 Wörter Fließtext in 9 Sätzen, ohne Gleichungen, Abbildungen und Quellenangaben. Für einen Abstract ist das passend (üblich sind 150–250 Wörter). Im Plan steht kein eigenes Budget. Der Abstract ist ein einziger Absatz mit über 180 Wörtern, was bei einem Abstract üblich ist und nicht aufgeteilt werden muss.

### Struktur von oben
- Z. 1: Problemfrage als fett gesetzte Zeile über dem Abstract
- Z. 2: Überschrift `## Abstrakt:`
- Z. 4, S1: Einordnung (Bahnen kosten Geschwindigkeitsaufwand)
- S2: Fragestellung, fast wörtlich die Problemfrage aus Z. 1
- S3: Methode der Rechnung (Patched Conics umschrieben, Kreisbahnen, eine Ebene)
- S4: Ergebnis Teilfrage 1 (31 / 44 %)
- S5: Ergebnis Teilfrage 2 (7,3 / 2,9 %, Variante 2 schneller)
- S6: Methode der Simulation, dazu „Einsparungen bleiben bestehen“
- S7: Abweichungen Simulation gegen Rechnung
- S8: Ursache der größeren Abweichung
- S9: Schlussfolgerung zur Näherung

Formal gibt es zwei Probleme:
- „Abstrakt“ ist im Deutschen ein Adjektiv. Als Überschrift passt „Abstract“ oder „Kurzfassung“, ohne Doppelpunkt. Die gleiche Bezeichnung steht auch in `KI-Nutzung.md` und im Prompt-Dateinamen.
- Die Problemfrage steht in Z. 1, in S2 und in 1.1 (Z. 17), also dreimal fast gleich lautend.

### Roter Faden
- **Methode aufgeteilt:** Die Methode der Simulation steht in S2 („eigens entwickelte … Simulation“) und in S6 („die das Raumfahrzeug im Feld aller Himmelskörper numerisch integriert“). Die Methode der Rechnung steht dazwischen in S3. Wer die Methodik sucht, muss sie aus drei Sätzen zusammensetzen.
- **Unklarer Satzanfang in S7:** „Die der Bahntransfers …“ setzt voraus, dass der Leser „Einsparungen“ aus S6 im Kopf behält. Das liest sich holprig.
- **Kein Bruch:** Ansonsten führt jeder Satz den vorigen fort, und es gibt keine Rücksprünge.

Vorgeschlagene Reihenfolge:
1. Einordnung (S1)
2. Frage (S2, gekürzt)
3. Methode von Rechnung und Simulation zusammen (S3 und der Relativsatz aus S6)
4. Ergebnisse der Rechnung (S4, S5)
5. Ergebnisse der Simulation einschließlich der Anpassung in Variante 1 (S6, S7)
6. Ursache, vorsichtig formuliert (S8)
7. Schluss mit Modellgrenze (S9)

### Inhalt
**Fehler**
- **S8, Ursache zu bestimmt:** „Die größere Abweichung geht auf die Anziehung der Sonne … zurück.“ In 4.4 (Z. 132, 155) steht nur, dass die Messungen darauf „hindeuten“. Das Fazit schreibt „vor allem“. Besser: „lässt sich vor allem … zuschreiben“.
- **S7, fehlende Bedingung bei Variante 1:** Die 1,6 % gelten erst, nachdem der Periapsisabstand um 8,8 % vergrößert wurde (4.4, Z. 120). Ohne diese Angabe wirkt Variante 1 fast deckungsgleich mit der Rechnung, obwohl dort die gleiche Ursache wirkt wie in Variante 2.
- **S7, Richtung fehlt:** „weichen … um rund 15 % ab“ sagt nicht, dass die Einsparung in Variante 2 *kleiner* ausfällt. 4.4 und das Fazit sagen das ausdrücklich.

**Schwächen**
- **S4/S5, Zuordnung unklar:** Bei „31 beziehungsweise 44 %“ und „7,3 beziehungsweise 2,9 %“ ist nicht erkennbar, welcher Wert zu welchem Fall gehört, weil die Parameter nicht genannt werden. Die Apoapsiden 1,75 und 2,0 AE und die beiden Varianten (tangentiale Ankunft am Saturn, Periapsis bei fünf Jupiterradien) sollten kurz genannt werden.
- **S6, Simulation zu weit gefasst:** Die Beschreibung ist richtig, wirkt aber wie eine volle N-Körper-Rechnung. Laut 3.4 (Z. 292) laufen die Himmelskörper auf vorgegebenen Kepler-Bahnen, und integriert wird nur das Raumfahrzeug. Ein Zusatz von zwei Wörtern reicht, etwa „auf vorgegebenen Bahnen“.
- **S3, Begriff fehlt:** Die Näherung wird umschrieben, aber nicht benannt, und S9 greift sie als „die Näherung“ wieder auf. „Patched-Conics-Näherung“ einmal zu nennen macht den Rückbezug eindeutig.
- **S9 fast wortgleich mit dem Fazit:** Der Satz entspricht dem letzten Satz des vierten Fazit-Absatzes. In einem Abstract ist das zulässig, eine eigene Formulierung wirkt aber weniger kopiert.
- **Prüfvermerk offen:** Der Eintrag in `KI-Nutzung.md` (Z. 56) hat in der Spalte „Prüfung des Autors“ noch den Platzhalter „(vom Autor zu ergänzen)“.

### Fehlende Inhalte
- **Modellgrenze:** Das Fazit (fünfter Absatz) zeigt, dass Rechnung und Simulation dieselben kreisförmigen Bahnen in einer Ebene annehmen. Allein die Exzentrizität der Marsbahn verschiebt den Hohmann-Aufwand um bis zu 0,17 km/s, also um mehr als jede gemessene Abweichung. Ohne diesen Satz verallgemeinert S9 aus je einem Beispiel. Eine neue Quelle ist dafür nicht nötig, der Inhalt steht in 3.4 und im Fazit.

### Ende
Der Abstract endet mit einer klaren Schlussfolgerung (S9), aber ohne Einschränkung. Ein besserer Schluss formuliert S9 als Aussage über das untersuchte Modell und hängt einen Satz zur Grenze an. Beispiel:

„Im untersuchten Modell genügt die Näherung damit für Transfers zwischen zwei Planeten, während sich ein Gravitationsmanöver erst mit numerischer Integration genau planen lässt. Da Rechnung und Simulation dieselben kreisförmigen Planetenbahnen annehmen, beschreiben die Einsparungen das Verhältnis der Missionsvarianten zueinander und noch nicht den Aufwand einer realen Mission.“

Ersatz für S6 bis S8:

„In der Simulation bleiben alle Einsparungen bestehen, bei den Bahntransfers mit einer Abweichung von weniger als $2\ \%$ gegenüber der Rechnung. Beim Gravitationsmanöver stimmt die Einsparung in der ersten Variante auf $1{,}6\ \%$ überein, setzt dort aber einen um $8{,}8\ \%$ größeren Periapsisabstand voraus, während sie in der zweiten Variante um rund $15\ \%$ kleiner ausfällt. Beides lässt sich vor allem der Anziehung der Sonne innerhalb der großen Einflusssphäre Jupiters zuschreiben, die die Näherung vernachlässigt.“

### Prioritäten
1. S7/S8 ersetzen: Ursache vorsichtiger formulieren, die Anpassung in Variante 1 nennen und angeben, dass die Einsparung in Variante 2 kleiner ausfällt.
2. S9 auf das Modell einschränken und den Satz zur Modellgrenze anhängen.
3. In S4/S5 die Parameter nennen (1,75 / 2,0 AE, die beiden Varianten am Jupiter).
4. Überschrift in „Abstract“ oder „Kurzfassung“ ändern, ohne Doppelpunkt, und die Bezeichnung in `KI-Nutzung.md` angleichen.
5. Methode in S3 bündeln, „Patched-Conics-Näherung“ einmal nennen und S2 kürzen.

**Deine Entscheidung:** Wo steht die Problemfrage? Wenn die Zeile in Z. 1 bleibt, kann S2 auf einen Halbsatz schrumpfen. Allerdings steht der Abstract dann nicht mehr für sich, falls er später allein vor dem Inhaltsverzeichnis steht. Fällt Z. 1 weg (die Frage steht ohnehin in 1.1, Z. 17), bleibt S2 vollständig, und der Abstract kann ohne Umbau unnummeriert vor das Inhaltsverzeichnis wandern.