**Problemfrage**: Wie stark senken energieoptimale Bahntransfers gegenüber schnelleren Transfers sowie Gravitationsmanöver gegenüber dem direkten Transfer den Geschwindigkeitsaufwand interplanetarer Missionen, und wie genau lassen sich die berechneten Einsparungen in einer eigenen Simulation reproduzieren?
## Abstrakt:

Leer

___
## 1. Einleitung:

### 1.1 Motivation

Die Apollo-11-Mission im Juli 1969 markiert den Punkt, an dem erstmals Menschen einen anderen Himmelskörper betraten [2]. Auffällig am Flugprofil dieser und der vorangegangenen Missionen ist, dass die Raumfahrzeuge sich nicht auf direktem Weg zum Mond bewegten, sondern einer weit ausholenden Bahn folgten, die die Anziehung von Erde und Mond ausnutzte [2]. Diese Bahnform ging auf eine bewusste Entscheidung der Missionsplanung zurück.

Der Grund dafür liegt in einer Eigenheit der Raumfahrt, die der Alltagserfahrung widerspricht. Auf der Erde kostet der kürzeste Weg am wenigsten Treibstoff, im Weltraum gilt das nicht [1]. Wegen der enormen Entfernungen und der ständigen Einwirkung planetarer Gravitationsfelder ist eine gerade Verbindung zwischen zwei Planeten energetisch außerordentlich teuer und mit heutiger Antriebstechnik in vielen Fällen unerreichbar. Raumfahrzeuge bewegen sich stattdessen antriebslos auf gekrümmten Bahnen, die sie durch kurze, gezielte Triebwerksmanöver verändern [1]. Die dafür nötige Geschwindigkeitsänderung bestimmt unmittelbar die mitzuführende Treibstoffmasse und damit, ob eine Mission überhaupt startbar ist [5].

Aus diesem Zusammenhang entsteht der zentrale Kompromiss der Astrodynamik zwischen Flugdauer und Treibstoffbedarf [1]. Ein energieoptimaler Transfer, der den Zielplaneten auf einer langgezogenen Bahn erreicht, dauert länger als ein schnellerer Transfer, verlangt aber deutlich weniger Geschwindigkeitsänderung [1]. Weiter senken lässt sich der Aufwand durch Gravitationsmanöver, bei denen ein Raumfahrzeug im Vorbeiflug an einem Planeten Geschwindigkeit gewinnt, ohne dafür Treibstoff aufzuwenden [1]. Die Voyager-Sonden erreichten auf diesem Weg die äußeren Planeten mit einem Vorrat, der für einen direkten Transfer ohne Vorbeiflug bei weitem nicht gereicht hätte [1] [4], und auch die für die kommenden Jahre geplanten Marsmissionen rechnen mit Flugzeiten von mehreren Monaten statt mit dem kürzesten Weg [1].

Vor diesem Hintergrund geht die vorliegende Seminararbeit der Frage nach, wie stark energieoptimale Bahntransfers gegenüber schnelleren Transfers sowie Gravitationsmanöver gegenüber dem direkten Transfer den Geschwindigkeitsaufwand interplanetarer Missionen senken und wie genau sich die berechneten Einsparungen in einer eigenen Simulation reproduzieren lassen. Die rechnerische Behandlung dieser Frage beruht üblicherweise auf Formeln, die nur zwei Körper berücksichtigen und damit die Störungen durch alle übrigen Massen des Systems ausklammern. Historische Missionsdaten wiederum belegen jeweils nur die tatsächlich geflogene Bahn und erlauben keinen Vergleich mit den Alternativen, die dieselbe Mission auch hätte fliegen können.

Zur Untersuchung wurde daher eine eigene zweidimensionale N-Körper-Simulation entwickelt, in der sich dieselbe Mission wiederholt unter veränderten Bedingungen durchführen lässt, etwa mit und ohne Vorbeiflug oder mit unterschiedlich starken Manövern. Anhand dieses Modells werden konkrete Navigationsbeispiele simuliert und die berechneten Einsparungen mit den in der Simulation gemessenen Werten verglichen.

