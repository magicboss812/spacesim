# 3. Bahntransfers
## 3.1 Hohmann-Transfer
Nach Abschnitt 2.3 bleibt der Ort eines Raumfahrzeugs bei einem impulsiven Manöver unverändert, sodass die neue Bahn stets durch den Punkt des Manövers verläuft. Zwei Bahnen ohne gemeinsamen Punkt lassen sich deshalb mit einem einzelnen Impuls nicht verbinden. Es braucht mindestens zwei Impulse, von denen der erste auf eine Zwischenbahn führt, die beide Bahnen erreicht, und der zweite von dieser auf die Zielbahn. Walter Hohmann zeigte 1925, dass bei zwei Kreisbahnen eine Ellipse, die beide Bahnen an gegenüberliegenden Punkten berührt, der sparsamste Weg mit zwei Impulsen ist [34]. Dieser **Hohmann-Transfer** wird hier für zwei Kreisbahnen um denselben Zentralkörper berechnet, die in einer gemeinsamen Ebene liegen, was der in Abschnitt 2.1 begründeten Beschränkung auf zwei Dimensionen entspricht. Beide Manöver gelten als impulsiv im Sinne von Abschnitt 2.3.

Die Startbahn hat den Radius $r_1$, die Zielbahn den Radius $r_2$, und zunächst liegt die Zielbahn außen ($r_1 < r_2$). Die Transferbahn berührt dann die Startbahn in ihrer Periapsis und die Zielbahn in ihrer Apoapsis, sodass $r_p = r_1$ und $r_a = r_2$ gilt [34]. Ihre große Halbachse $a_t$ folgt damit unmittelbar aus Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-halbachse|(7)]]:

$$a_t = \frac{r_1 + r_2}{2}\tag{21}$$
^eq-hohmann-halbachse

Abb. 8 zeigt beide Kreisbahnen, die Transferellipse und die Punkte der beiden Impulse. Geflogen wird nur die Hälfte der Ellipse zwischen den beiden Berührpunkten.

![[Obsidian Vault/Evaluation/Grafiken & Bilder/hohmann_transfer.png|450]]
*Abb. 8, Hohmann-Transfer von einer inneren Kreisbahn mit dem Radius $r_1$ auf eine äußere mit dem Radius $r_2$. Der erste Impuls liegt in der Periapsis, der zweite in der Apoapsis der Transferellipse – KI-erstellt (Claude)*

In welche Richtung die beiden Impulse wirken, ergibt sich aus der Bahnenergie. Für eine Kreisbahn ist die große Halbachse gleich dem Radius, und nach Gl. [[#^eq-hohmann-halbachse|(21)]] liegt $a_t$ zwischen $r_1$ und $r_2$. Nach Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-energie-halbachse|(10)]] liegt deshalb auch die Energie der Transferbahn zwischen den Energien der beiden Kreisbahnen, für $r_1 < r_2$ also $\varepsilon_1 < \varepsilon_t < \varepsilon_2$ [34]. Das Raumfahrzeug muss demnach zweimal Energie gewinnen, was an einem festen Ort nach Abschnitt 2.2 nur über eine höhere Geschwindigkeit möglich ist. Der erste Impuls beschleunigt es auf der Startbahn und hebt die gegenüberliegende Apside bis auf den Radius der Zielbahn an. Nach einem halben Umlauf erreicht es dort die Apoapsis der Transferbahn, wo der zweite Impuls erneut beschleunigt und die Periapsis von $r_1$ auf $r_2$ anhebt, sodass die Bahn kreisförmig wird [34]. Führt der Transfer nach innen, kehrt sich die Reihenfolge der Energien um, und das Raumfahrzeug bremst an beiden Punkten [34].

Sparsam ist der Hohmann-Transfer, weil die Geschwindigkeiten von Kreisbahn und Transferbahn an beiden Berührpunkten dieselbe Richtung haben. Jeder Impuls ändert dadurch nur den Betrag der Geschwindigkeit, ohne zusätzlich ihre Richtung drehen zu müssen [34]. Der Geschwindigkeitsaufwand ergibt sich deshalb aus vier Beträgen. Die Geschwindigkeiten $v_{k,1}$ und $v_{k,2}$ auf den beiden Kreisbahnen liefert Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-kreisbahn|(12)]] mit $r_1$ und $r_2$. Die Geschwindigkeiten $v_{t,1}$ und $v_{t,2}$, die das Raumfahrzeug auf der Transferbahn an denselben beiden Orten hat, folgen aus der Vis-viva-Gleichung [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-visviva|(11)]] mit der großen Halbachse $a_t$ [34]:

$$v_{t,1} = \sqrt{\mu\left(\frac{2}{r_1} - \frac{1}{a_t}\right)}, \qquad v_{t,2} = \sqrt{\mu\left(\frac{2}{r_2} - \frac{1}{a_t}\right)}\tag{22}$$
^eq-hohmann-vt

Die beiden Impulse sind die Unterschiede zwischen Transfer- und Kreisbahngeschwindigkeit am jeweiligen Ort:

$$\Delta v_1 = v_{t,1} - v_{k,1}, \qquad \Delta v_2 = v_{k,2} - v_{t,2}\tag{23}$$
^eq-hohmann-impulse

Beim Transfer nach außen sind beide Werte positiv, beim Transfer nach innen beide negativ. Da ein Bremsmanöver ebenso Treibstoff verbraucht wie eine Beschleunigung, zählen nach Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-deltav-ges|(16)]] die Beträge, und der Geschwindigkeitsaufwand des Hohmann-Transfers beträgt [34]:

$$\Delta v = \left|\Delta v_1\right| + \left|\Delta v_2\right|\tag{24}$$
^eq-hohmann-deltav

Zwischen den beiden Impulsen durchläuft das Raumfahrzeug die halbe Transferellipse. Die Flugzeit $t_H$ ist deshalb die halbe Umlaufzeit der Transferbahn nach Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-umlaufzeit|(8)]] [34]:

$$t_H = \pi\sqrt{\frac{a_t^3}{\mu}}\tag{25}$$
^eq-hohmann-flugzeit

Mit den Radien der beiden Bahnen sind damit Geschwindigkeitsaufwand und Flugzeit festgelegt. Für einen Flug zwischen zwei Planeten, bei dem die Sonne der Zentralkörper ist, kommt eine weitere Bedingung hinzu. Der Zielplanet bewegt sich während der Flugzeit auf seiner Bahn weiter, sodass der Start zu einem Zeitpunkt erfolgen muss, bei dem Raumfahrzeug und Zielplanet gleichzeitig am Ankunftspunkt eintreffen [35]. Beschrieben wird die Stellung der beiden Planeten durch den **Phasenwinkel** $\gamma$, den Winkel an der Sonne zwischen den Richtungen zum Startplaneten und zum Zielplaneten [35]. Er wird hier in Umlaufrichtung vom Startplaneten aus gemessen, sodass der Zielplanet bei positivem $\gamma$ vorausläuft.

Wie weit ein Planet in einer bestimmten Zeit vorrückt, gibt seine mittlere Winkelgeschwindigkeit $n$ an, die sich aus der Umlaufzeit $T$ nach Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-umlaufzeit|(8)]] ergibt [35]:

$$n = \frac{2\pi}{T}\tag{26}$$
^eq-winkelgeschwindigkeit

Auf einer Kreisbahn ist die Winkelgeschwindigkeit konstant, und der Planet legt in der Zeit $t$ den Winkel $n\,t$ zurück. Auf einer Ellipse gilt Gl. [[#^eq-winkelgeschwindigkeit|(26)]] nur im Mittel über einen Umlauf, weshalb die Annahme kreisförmiger Planetenbahnen es erlaubt, auf die Keplergleichung für den Zusammenhang zwischen Zeit und Ort zu verzichten.

Das Raumfahrzeug legt auf der halben Transferellipse den Winkel $\pi$ um die Sonne zurück, der Zielplanet in derselben Zeit den Winkel $n_2\,t_H$ (Abb. 9). Beide treffen genau dann zusammen, wenn der Vorsprung des Zielplaneten beim Start und sein Weg während des Flugs zusammen $\pi$ ergeben. Für den Phasenwinkel beim Start gilt deshalb [35]:

$$\gamma_1 = \pi - n_2\,t_H\tag{27}$$
^eq-phasenwinkel

![[Obsidian Vault/Evaluation/Grafiken & Bilder/phasenwinkel_start.png|450]]
*Abb. 9, Phasenwinkel $\gamma_1$ zwischen Start- und Zielplanet beim Start eines Hohmann-Transfers nach außen. Der Zielplanet legt während des Flugs den Winkel $n_2\,t_H$ zurück und erreicht den Ankunftspunkt gleichzeitig mit dem Raumfahrzeug – KI-erstellt (Claude)*

Bei einem Transfer nach außen läuft der Zielplanet langsamer um als das Raumfahrzeug und muss ihm beim Start vorauslaufen. Bei einem Transfer nach innen kann $n_2\,t_H$ größer als $\pi$ werden, und ein negativer Wert von $\gamma_1$ bedeutet dann, dass der Zielplanet beim Start hinter dem Startplaneten steht.

Da beide Planeten mit unterschiedlicher Winkelgeschwindigkeit umlaufen, ändert sich der Phasenwinkel fortlaufend, und der Wert aus Gl. [[#^eq-phasenwinkel|(27)]] liegt nur zu bestimmten Zeitpunkten vor. Die Zeit, nach der dieselbe Stellung der beiden Planeten wiederkehrt, heißt **synodische Periode** und ergibt sich aus den beiden Umlaufzeiten [35]:

$$T_{syn} = \frac{T_1\,T_2}{\left|T_1 - T_2\right|}\tag{28}$$
^eq-synodisch

Sie ist der Abstand zwischen zwei aufeinanderfolgenden Startfenstern. Auch der Rückflug verlangt eine bestimmte Phasenlage, sodass nach der Ankunft in der Regel eine Wartezeit am Ziel entsteht [35].

Der Hohmann-Transfer ist damit der sparsamste Weg zwischen zwei Kreisbahnen mit zwei Impulsen, legt aber zugleich die Flugzeit nach Gl. [[#^eq-hohmann-flugzeit|(25)]] und über Gl. [[#^eq-phasenwinkel|(27)]] den Startzeitpunkt fest. Welchen Geschwindigkeitsaufwand ein schnellerer Weg erfordert, untersucht Abschnitt 3.2.
