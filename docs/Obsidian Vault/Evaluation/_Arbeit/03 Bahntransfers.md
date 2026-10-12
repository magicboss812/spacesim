# 3. Bahntransfers
## 3.1 Hohmann-Transfer
Nach Abschnitt 2.3 bleibt der Ort eines Raumfahrzeugs bei einem impulsiven Manöver unverändert, sodass die neue Bahn stets durch den Punkt des Manövers verläuft. Zwei Bahnen ohne gemeinsamen Punkt lassen sich deshalb mit einem einzelnen Impuls nicht verbinden. Es braucht mindestens zwei Impulse, von denen der erste auf eine Zwischenbahn führt, die beide Bahnen erreicht, und der zweite von dieser auf die Zielbahn. Bei zwei Kreisbahnen ist eine Ellipse, die beide Bahnen an gegenüberliegenden Punkten berührt, der sparsamste Weg mit zwei Impulsen [34]. Dieser **Hohmann-Transfer** wird hier für zwei Kreisbahnen um denselben Zentralkörper berechnet, die in einer gemeinsamen Ebene liegen, was der in Abschnitt 2.1 begründeten Beschränkung auf zwei Dimensionen entspricht. Beide Manöver gelten als impulsiv im Sinne von Abschnitt 2.3.

Die Startbahn hat den Radius $r_1$, die Zielbahn den Radius $r_2$, und zunächst liegt die Zielbahn außen ($r_1 < r_2$). Die Transferbahn berührt dann die Startbahn in ihrer Periapsis und die Zielbahn in ihrer Apoapsis, sodass $r_p = r_1$ und $r_a = r_2$ gilt [34]. Ihre große Halbachse $a_t$ folgt damit unmittelbar aus Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-halbachse|(7)]]:

$$a_t = \frac{r_1 + r_2}{2}\tag{20}$$
^eq-hohmann-halbachse

Abb. 7 zeigt beide Kreisbahnen, die Transferellipse und die Punkte der beiden Impulse. Geflogen wird nur die Hälfte der Ellipse zwischen den beiden Berührpunkten.

![[Obsidian Vault/Evaluation/Grafiken & Bilder/hohmann_transfer.png|450]]
*Abb. 7, Hohmann-Transfer von einer inneren Kreisbahn mit dem Radius $r_1$ auf eine äußere mit dem Radius $r_2$. Der erste Impuls liegt in der Periapsis, der zweite in der Apoapsis der Transferellipse – KI-erstellt (Claude)*

In welche Richtung die beiden Impulse wirken, ergibt sich aus der Bahnenergie. Für eine Kreisbahn ist die große Halbachse gleich dem Radius, und nach Gl. [[#^eq-hohmann-halbachse|(20)]] liegt $a_t$ zwischen $r_1$ und $r_2$. Nach Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-energie-halbachse|(10)]] liegt deshalb auch die Energie der Transferbahn zwischen den Energien der beiden Kreisbahnen, für $r_1 < r_2$ also $\varepsilon_1 < \varepsilon_t < \varepsilon_2$ [34]. Das Raumfahrzeug muss demnach zweimal Energie gewinnen, was an einem festen Ort nach Abschnitt 2.2 nur über eine höhere Geschwindigkeit möglich ist. Der erste Impuls beschleunigt es auf der Startbahn und hebt die gegenüberliegende Apside bis auf den Radius der Zielbahn an. Nach einem halben Umlauf erreicht es dort die Apoapsis der Transferbahn, wo der zweite Impuls erneut beschleunigt und die Periapsis von $r_1$ auf $r_2$ anhebt, sodass die Bahn kreisförmig wird [34]. Führt der Transfer nach innen, kehrt sich die Reihenfolge der Energien um, und das Raumfahrzeug bremst an beiden Punkten [34].

Sparsam ist der Hohmann-Transfer, weil die Geschwindigkeiten von Kreisbahn und Transferbahn an beiden Berührpunkten dieselbe Richtung haben. Jeder Impuls ändert dadurch nur den Betrag der Geschwindigkeit, ohne zusätzlich ihre Richtung drehen zu müssen [34]. Der Geschwindigkeitsaufwand ergibt sich deshalb aus vier Beträgen. Die Geschwindigkeiten $v_{k,1}$ und $v_{k,2}$ auf den beiden Kreisbahnen liefert Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-kreisbahn|(12)]] mit $r_1$ und $r_2$. Die Geschwindigkeiten $v_{t,1}$ und $v_{t,2}$, die das Raumfahrzeug auf der Transferbahn an denselben beiden Orten hat, folgen aus der Vis-viva-Gleichung [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-visviva|(11)]] mit der großen Halbachse $a_t$ [34]:

$$v_{t,1} = \sqrt{\mu\left(\frac{2}{r_1} - \frac{1}{a_t}\right)}, \qquad v_{t,2} = \sqrt{\mu\left(\frac{2}{r_2} - \frac{1}{a_t}\right)}\tag{21}$$
^eq-hohmann-vt

Die beiden Impulse sind die Unterschiede zwischen Transfer- und Kreisbahngeschwindigkeit am jeweiligen Ort:

$$\Delta v_1 = v_{t,1} - v_{k,1}, \qquad \Delta v_2 = v_{k,2} - v_{t,2}\tag{22}$$
^eq-hohmann-impulse

Beim Transfer nach außen sind beide Werte positiv, beim Transfer nach innen beide negativ. Da ein Bremsmanöver ebenso Treibstoff verbraucht wie eine Beschleunigung, zählen nach Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-deltav-ges|(16)]] die Beträge, und der Geschwindigkeitsaufwand des Hohmann-Transfers beträgt [34]:

$$\Delta v = \left|\Delta v_1\right| + \left|\Delta v_2\right|\tag{23}$$
^eq-hohmann-deltav

Zwischen den beiden Impulsen durchläuft das Raumfahrzeug die halbe Transferellipse. Die Flugzeit $t_H$ ist deshalb die halbe Umlaufzeit der Transferbahn nach Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-umlaufzeit|(8)]] [34]:

$$t_H = \pi\sqrt{\frac{a_t^3}{\mu}}\tag{24}$$
^eq-hohmann-flugzeit

Mit den Radien der beiden Bahnen sind damit Geschwindigkeitsaufwand und Flugzeit festgelegt. Für einen Flug zwischen zwei Planeten, bei dem die Sonne der Zentralkörper ist, kommt eine weitere Bedingung hinzu. Der Zielplanet bewegt sich während der Flugzeit auf seiner Bahn weiter, sodass der Start zu einem Zeitpunkt erfolgen muss, bei dem Raumfahrzeug und Zielplanet gleichzeitig am Ankunftspunkt eintreffen [35]. Beschrieben wird die Stellung der beiden Planeten durch den **Phasenwinkel** $\gamma$, den Winkel an der Sonne zwischen den Richtungen zum Startplaneten und zum Zielplaneten [35]. Er wird hier in Umlaufrichtung vom Startplaneten aus gemessen, sodass der Zielplanet bei positivem $\gamma$ vorausläuft.

Wie weit ein Planet in einer bestimmten Zeit vorrückt, gibt seine mittlere Winkelgeschwindigkeit $n$ an, die sich aus der Umlaufzeit $T$ nach Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-umlaufzeit|(8)]] ergibt [35]:

$$n = \frac{2\pi}{T}\tag{25}$$
^eq-winkelgeschwindigkeit

Auf einer Kreisbahn ist die Winkelgeschwindigkeit konstant, und der Planet legt in der Zeit $t$ den Winkel $n\,t$ zurück. Auf einer Ellipse gilt Gl. [[#^eq-winkelgeschwindigkeit|(25)]] nur im Mittel über einen Umlauf, weshalb die Annahme kreisförmiger Planetenbahnen es erlaubt, auf die Keplergleichung für den Zusammenhang zwischen Zeit und Ort zu verzichten.

Das Raumfahrzeug legt auf der halben Transferellipse den Winkel $\pi$ um die Sonne zurück, der Zielplanet in derselben Zeit den Winkel $n_2\,t_H$ (Abb. 8). Beide treffen genau dann zusammen, wenn der Vorsprung des Zielplaneten beim Start und sein Weg während des Flugs zusammen $\pi$ ergeben. Für den Phasenwinkel beim Start gilt deshalb [35]:

$$\gamma_1 = \pi - n_2\,t_H\tag{26}$$
^eq-phasenwinkel

![[Obsidian Vault/Evaluation/Grafiken & Bilder/phasenwinkel_start.png|450]]
*Abb. 8, Phasenwinkel $\gamma_1$ zwischen Start- und Zielplanet beim Start eines Hohmann-Transfers nach außen. Der Zielplanet legt während des Flugs den Winkel $n_2\,t_H$ zurück und erreicht den Ankunftspunkt gleichzeitig mit dem Raumfahrzeug – KI-erstellt (Claude)*

Bei einem Transfer nach außen läuft der Zielplanet langsamer um als das Raumfahrzeug und muss ihm beim Start vorauslaufen. Bei einem Transfer nach innen kann $n_2\,t_H$ größer als $\pi$ werden, und ein negativer Wert von $\gamma_1$ bedeutet dann, dass der Zielplanet beim Start hinter dem Startplaneten steht.

Da beide Planeten mit unterschiedlicher Winkelgeschwindigkeit umlaufen, ändert sich der Phasenwinkel fortlaufend, und der Wert aus Gl. [[#^eq-phasenwinkel|(26)]] liegt nur zu bestimmten Zeitpunkten vor. Die Zeit, nach der dieselbe Stellung der beiden Planeten wiederkehrt, heißt **synodische Periode** und ergibt sich aus den beiden Umlaufzeiten [35]:

$$T_{syn} = \frac{T_1\,T_2}{\left|T_1 - T_2\right|}\tag{27}$$
^eq-synodisch

Sie ist der Abstand zwischen zwei aufeinanderfolgenden Startfenstern. Auch der Rückflug verlangt eine bestimmte Phasenlage, sodass nach der Ankunft in der Regel eine Wartezeit am Ziel entsteht [35].

Wie groß diese Werte zwischen zwei Planetenbahnen ausfallen, zeigt ein Hohmann-Transfer von der Erdbahn zur Marsbahn mit der Sonne als Zentralkörper. Betrachtet werden dabei allein die Bahnen der beiden Planeten. Das Raumfahrzeug befindet sich zu Beginn auf der Erdbahn und bewegt sich mit deren Kreisbahngeschwindigkeit um die Sonne, eine Umlaufbahn um die Erde wird nicht berücksichtigt. Ebenso endet der Transfer auf der Marsbahn, und die Anziehung von Erde und Mars bleibt außer Acht. Die Planeten selbst gehen nur in den Phasenwinkel ein, weil der Mars bei der Ankunft am Zielpunkt stehen muss. Der Gravitationsparameter der Sonne beträgt $\mu = 1{,}32712 \cdot 10^{20}\ \mathrm{m^3/s^2}$ [17, Tab. 8]. Als Radien der beiden Kreisbahnen dienen die mittleren Sonnenabstände der Planeten, also die großen Halbachsen ihrer Bahnen, mit $r_1 = 1{,}49598 \cdot 10^{11}\ \mathrm{m}$ für die Erde [17, Tab. 10] und $r_2 = 2{,}27939 \cdot 10^{11}\ \mathrm{m}$ für den Mars [17, Tab. 10]. Wie im übrigen Abschnitt liegen beide Bahnen in einer gemeinsamen Ebene, und beide Manöver gelten als impulsiv. Alle Größen werden in SI-Einheiten eingesetzt. Zwischenwerte sind gekürzt angegeben, weitergerechnet wird aber stets mit dem ungerundeten Wert, sodass erst das Ergebnis gerundet wird.

Nach Gl. [[#^eq-hohmann-halbachse|(20)]] hat die Transferbahn die große Halbachse

$$a_t = \frac{1{,}49598 \cdot 10^{11} + 2{,}27939 \cdot 10^{11}}{2}\ \mathrm{m} = 1{,}887685 \cdot 10^{11}\ \mathrm{m}$$

Die Kreisbahngeschwindigkeiten auf Erd- und Marsbahn folgen aus Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-kreisbahn|(12)]]:

$$v_{k,1} = \sqrt{\frac{1{,}32712 \cdot 10^{20}}{1{,}49598 \cdot 10^{11}}}\ \mathrm{m/s} \approx 29\,784{,}6\ \mathrm{m/s}, \qquad v_{k,2} = \sqrt{\frac{1{,}32712 \cdot 10^{20}}{2{,}27939 \cdot 10^{11}}}\ \mathrm{m/s} \approx 24\,129{,}4\ \mathrm{m/s}$$

Auf der Transferbahn hat das Raumfahrzeug an denselben beiden Orten nach Gl. [[#^eq-hohmann-vt|(21)]] die Geschwindigkeiten

$$v_{t,1} = \sqrt{1{,}32712 \cdot 10^{20}\left(\frac{2}{1{,}49598 \cdot 10^{11}} - \frac{1}{1{,}887685 \cdot 10^{11}}\right)}\ \mathrm{m/s} \approx 32\,729{,}3\ \mathrm{m/s}$$

$$v_{t,2} = \sqrt{1{,}32712 \cdot 10^{20}\left(\frac{2}{2{,}27939 \cdot 10^{11}} - \frac{1}{1{,}887685 \cdot 10^{11}}\right)}\ \mathrm{m/s} \approx 21\,480{,}5\ \mathrm{m/s}$$

Das Raumfahrzeug ist auf der Transferbahn demnach an der Erdbahn schneller und an der Marsbahn langsamer als auf der jeweiligen Kreisbahn, sodass beide Impulse nach Gl. [[#^eq-hohmann-impulse|(22)]] positiv sind und beschleunigen:

$$\Delta v_1 = \left(32\,729{,}3 - 29\,784{,}6\right)\ \mathrm{m/s} \approx 2\,944{,}7\ \mathrm{m/s}, \qquad \Delta v_2 = \left(24\,129{,}4 - 21\,480{,}5\right)\ \mathrm{m/s} \approx 2\,648{,}9\ \mathrm{m/s}$$

Der Geschwindigkeitsaufwand nach Gl. [[#^eq-hohmann-deltav|(23)]] beträgt damit

$$\Delta v = \left|\Delta v_1\right| + \left|\Delta v_2\right| \approx \left(2\,944{,}7 + 2\,648{,}9\right)\ \mathrm{m/s} \approx 5\,593{,}6\ \mathrm{m/s}$$

Für die Flugzeit liefert Gl. [[#^eq-hohmann-flugzeit|(24)]] mit $1\ \text{Tag} = 86\,400\ \mathrm{s}$

$$t_H = \pi\sqrt{\frac{\left(1{,}887685 \cdot 10^{11}\right)^3}{1{,}32712 \cdot 10^{20}}}\ \mathrm{s} \approx 2{,}23660 \cdot 10^{7}\ \mathrm{s} \approx 258{,}866\ \text{Tage}$$

Phasenwinkel und synodische Periode setzen die Umlaufzeiten der beiden Planeten voraus, die Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-umlaufzeit|(8)]] mit $a = r_1$ und $a = r_2$ liefert:

$$T_1 = \frac{2\pi}{\sqrt{1{,}32712 \cdot 10^{20}}}\left(1{,}49598 \cdot 10^{11}\right)^{3/2}\ \mathrm{s} \approx 3{,}15583 \cdot 10^{7}\ \mathrm{s}$$

$$T_2 = \frac{2\pi}{\sqrt{1{,}32712 \cdot 10^{20}}}\left(2{,}27939 \cdot 10^{11}\right)^{3/2}\ \mathrm{s} \approx 5{,}93544 \cdot 10^{7}\ \mathrm{s}$$

Nach Gl. [[#^eq-winkelgeschwindigkeit|(25)]] rückt der Mars mit der mittleren Winkelgeschwindigkeit

$$n_2 = \frac{2\pi}{5{,}93544 \cdot 10^{7}\ \mathrm{s}} \approx 1{,}05859 \cdot 10^{-7}\ \mathrm{rad/s}$$

vor und legt während der Flugzeit den Winkel $n_2\,t_H \approx 2{,}36764\ \mathrm{rad}$ zurück. Der Phasenwinkel beim Start beträgt nach Gl. [[#^eq-phasenwinkel|(26)]]

$$\gamma_1 = \pi - 2{,}36764 \approx 0{,}773950\ \mathrm{rad} \approx 44{,}3441°$$

Die synodische Periode folgt aus Gl. [[#^eq-synodisch|(27)]]:

$$T_{syn} = \frac{3{,}15583 \cdot 10^{7} \cdot 5{,}93544 \cdot 10^{7}}{\left|3{,}15583 \cdot 10^{7} - 5{,}93544 \cdot 10^{7}\right|}\ \mathrm{s} \approx 6{,}73880 \cdot 10^{7}\ \mathrm{s} \approx 779{,}954\ \text{Tage}$$

Gerundet erfordert der Transfer an der Erdbahn einen Impuls von $\Delta v_1 \approx 2{,}94\ \mathrm{km/s}$ und an der Marsbahn einen von $\Delta v_2 \approx 2{,}65\ \mathrm{km/s}$, zusammen einen Geschwindigkeitsaufwand von $\Delta v \approx 5{,}59\ \mathrm{km/s}$. Das Raumfahrzeug ist dafür $t_H \approx 259$ Tage unterwegs. Beim Start muss der Mars dem Raumfahrzeug um $\gamma_1 \approx 44{,}3°$ vorauslaufen, und diese Stellung kehrt nach $T_{syn} \approx 780$ Tagen wieder, sodass sich ein solches Startfenster etwa alle 2,1 Jahre öffnet.

Das Rechenbeispiel beschränkt sich auf die Bahnen der beiden Planeten. Wie das Raumfahrzeug von einer Umlaufbahn um die Erde auf die Transferbahn gelangt und am Ende in eine Umlaufbahn um den Mars einschwenkt, untersucht Abschnitt 3.2.

## 3.2 Abflug und Ankunft
Das Rechenbeispiel in Abschnitt 3.1 setzt voraus, dass sich das Raumfahrzeug bereits mit der Kreisbahngeschwindigkeit der Erdbahn um die Sonne bewegt und am Ende ebenso auf der Marsbahn verbleibt. Tatsächlich startet es aus einer Umlaufbahn um die Erde, der **Parkbahn**, und soll am Ziel in eine Umlaufbahn um den Mars einschwenken. Nach der Patched-Conics-Näherung aus Abschnitt 2.4 wird der Flug dafür in Abschnitte mit jeweils einem einzigen Zentralkörper zerlegt [36]. Innerhalb der Einflusssphäre der Erde verlässt das Raumfahrzeug den Planeten auf einer **Abflughyperbel**, in deren Brennpunkt die Erde steht. Außerhalb der Einflusssphären folgt es der Transferellipse aus Abschnitt 3.1 um die Sonne, bis es in die Einflusssphäre des Mars eintritt und sich ihm auf einer **Ankunftshyperbel** nähert [36]. Abb. 9 zeigt beide Hyperbeln. Berechnet wird zuerst die Transferellipse, da sie festlegt, mit welcher Geschwindigkeit das Raumfahrzeug die Grenzen der beiden Einflusssphären überquert, und erst daraus folgen die beiden Hyperbeln [36].

Eine scharfe Grenze besitzt die Einflusssphäre nicht, da sich die Anziehungen von Sonne und Planet entlang der Bahn stetig ändern. Sie dient als Trennlinie, an der die Rechnung den Zentralkörper wechselt [36]. Festgelegt wird sie über einen Vergleich der Beschleunigungen. Beschreibt man die Bewegung des Raumfahrzeugs einmal mit der Sonne und einmal mit dem Planeten als Zentralkörper, wirkt der jeweils andere Körper als Störung. Innerhalb der Einflusssphäre stört die Sonne die Bewegung um den Planeten verhältnismäßig weniger als der Planet die Bewegung um die Sonne [28]. Unter der Näherung, dass das Raumfahrzeug ungefähr so weit von der Sonne entfernt ist wie der Planet, folgt aus diesem Kriterium ein Radius der Einflusssphäre, der vom Bahnradius und vom Massenverhältnis zwischen Planet und Sonne abhängt [28]. Wegen $\mu = G\,m$ aus Abschnitt 2.1 lässt sich das Massenverhältnis durch das Verhältnis der Gravitationsparameter ersetzen. Im Folgenden kennzeichnen die Indizes 1 und 2 wie in Abschnitt 3.1 Größen an der Erd- und Marsbahn, die Indizes $E$ und $M$ Größen an den Planeten selbst und $\odot$ die Sonne. Für Erde und Mars lautet das Ergebnis [28]:

$$r_{SOI,E} = r_1\left(\frac{\mu_E}{\mu_\odot}\right)^{2/5}, \qquad r_{SOI,M} = r_2\left(\frac{\mu_M}{\mu_\odot}\right)^{2/5}\tag{28}$$
^eq-einflusssphaere

Mit $\mu_\odot = 1{,}32712 \cdot 10^{20}\ \mathrm{m^3/s^2}$ aus Abschnitt 3.1, $\mu_E = 3{,}98600 \cdot 10^{14}\ \mathrm{m^3/s^2}$ und $\mu_M = 4{,}28284 \cdot 10^{13}\ \mathrm{m^3/s^2}$ [17, Tab. 8] reicht die Einflusssphäre der Erde bis $r_{SOI,E} \approx 9{,}24648 \cdot 10^{8}\ \mathrm{m}$ und die des Mars bis $r_{SOI,M} \approx 5{,}77228 \cdot 10^{8}\ \mathrm{m}$.

Gemessen an den Bahnradien sind beide Einflusssphären klein, bei der Erde etwa $0{,}62\ \%$ von $r_1$ und beim Mars etwa $0{,}25\ \%$ von $r_2$. Im Maßstab der Transferellipse schrumpfen sie damit nahezu zu Punkten, und aus Sicht der Sonne beginnt und endet die Transferbahn praktisch am Ort des jeweiligen Planeten. Die heliozentrische Rechnung aus Abschnitt 3.1 zwischen den Bahnradien $r_1$ und $r_2$ bleibt deshalb gültig.

Gegenüber den Planeten selbst sind die Einflusssphären dagegen groß. Mit den Äquatorradien $R_E = 6\,378{,}1366\ \mathrm{km}$ und $R_M = 3\,396{,}19\ \mathrm{km}$ [17, Tab. 9] reicht die der Erde etwa 145 Erdradien weit und die des Mars etwa 170 Marsradien. Eine Umlaufbahn wenige hundert Kilometer über der Oberfläche liegt somit tief im Inneren der Einflusssphäre, und bis zu deren Grenze ist der Betrag der potentiellen Energie $-\mu/r$ aus Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-energie|(9)]] auf weniger als ein Prozent seines Wertes in einer solchen Bahn gesunken. Aus Sicht des Planeten liegt die Grenze der Einflusssphäre damit praktisch im Unendlichen. In dieser Näherung ist die Geschwindigkeit, mit der das Raumfahrzeug sie relativ zum Planeten überquert, die hyperbolische Exzessgeschwindigkeit $v_\infty$ aus Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-exzess|(14)]], also die Geschwindigkeit, die nach dem Verlassen des Gravitationsfelds übrig bleibt [33].

Mit dem Zentralkörper wechselt an dieser Grenze auch der Körper, auf den die Geschwindigkeiten bezogen werden. Die Geschwindigkeiten in Abschnitt 3.1 gelten relativ zur Sonne, $v_\infty$ dagegen relativ zum Planeten. So wie der relative Ortsvektor $\vec{r} = \vec{R}_2 - \vec{R}_1$ in Abschnitt 2.1 die Differenz zweier Ortsvektoren ist, ist eine Relativgeschwindigkeit die Differenz zweier Geschwindigkeiten. Die Geschwindigkeit $\vec{v}$ des Raumfahrzeugs relativ zur Sonne ist deshalb die Summe aus der Geschwindigkeit $\vec{v}_P$, mit der ein Planet um die Sonne läuft, und der Geschwindigkeit relativ zu diesem Planeten [33]:

$$\vec{v} = \vec{v}_P + \vec{v}_\infty\tag{29}$$
^eq-geschwindigkeitsaddition

Die beiden Vektoren werden dabei wie in Abb. 4 aneinandergesetzt. Für Erde und Mars auf ihren Kreisbahnen ist $\vec{v}_P$ die Kreisbahngeschwindigkeit $\vec{v}_{k,1}$ beziehungsweise $\vec{v}_{k,2}$, und $\vec{v}$ ist an der Grenze der Einflusssphäre die Geschwindigkeit $\vec{v}_{t,1}$ beziehungsweise $\vec{v}_{t,2}$ auf der Transferbahn (Abb. 9, jeweils rechts).

Beim Hohmann-Transfer sind die Geschwindigkeiten von Raumfahrzeug und Planet an den Berührpunkten parallel. Am Mars ergibt sich das von selbst, weil die Transferellipse die Marsbahn in ihrer Apoapsis berührt. An der Erde muss der Impuls in der Parkbahn dafür so gesetzt werden, dass das Raumfahrzeug die Einflusssphäre in Bewegungsrichtung der Erde verlässt. Gl. [[#^eq-geschwindigkeitsaddition|(29)]] geht damit in eine Gleichung der Beträge über. An der Erde gilt $v_{\infty,1} = v_{t,1} - v_{k,1}$ [33] und am Mars $v_{\infty,2} = v_{k,2} - v_{t,2}$ [37], nach Gl. [[#^eq-hohmann-impulse|(22)]] sind das genau die beiden Impulse des Hohmann-Transfers:

$$v_{\infty,1} = \Delta v_1, \qquad v_{\infty,2} = \Delta v_2\tag{30}$$
^eq-exzess-impulse

Die Impulse aus Abschnitt 3.1 sind demnach Geschwindigkeiten relativ zum Planeten. Vor dem ersten Impuls bewegt sich das Raumfahrzeug mit der Kreisbahngeschwindigkeit der Erdbahn und ruht damit relativ zur Erde, und $\Delta v_1$ ist die Geschwindigkeit, mit der es sich danach von ihr entfernt. Da Abschnitt 3.1 die Anziehung der Erde vernachlässigt, liefert der Impuls diese Relativgeschwindigkeit dort unmittelbar. Mit der Anziehung der Erde ist sie erst das, was nach dem Herausklettern aus ihrem Gravitationsfeld übrig bleiben muss, sodass der Impuls in der Parkbahn einen anderen Betrag hat. Am Mars gilt dasselbe in umgekehrter Richtung.

An der Erde ist das Raumfahrzeug schneller als der Planet und entfernt sich in dessen Bewegungsrichtung. Am Mars ist es langsamer, sodass der Planet das Raumfahrzeug einholt und es die Einflusssphäre vor dem Planeten erreicht [37]. Was in Abschnitt 3.1 als Beschleunigung auf die Kreisbahngeschwindigkeit der Marsbahn erscheint, ist relativ zum Mars deshalb ein Abbremsen, da das Raumfahrzeug am Ende dieselbe Geschwindigkeit um die Sonne haben soll wie der Planet. Beim Überqueren der Grenze erhält das Raumfahrzeug selbst keinen Impuls, es wechselt allein die Beschreibung seiner Bewegung [33]. Welcher Impuls in der Umlaufbahn um den Planeten nötig ist, damit es die Einflusssphäre mit $v_\infty$ verlässt oder bei der Ankunft in eine Umlaufbahn übergeht, folgt aus der Hyperbel.

Innerhalb der Einflusssphäre eines Planeten mit dem Gravitationsparameter $\mu_P$ bewegt sich das Raumfahrzeug auf einer Hyperbel, deren Periapsis auf einer kreisförmigen Umlaufbahn um den Planeten liegt, beim Abflug auf der Parkbahn und bei der Ankunft auf der Zielbahn. Diese Kreisbahn hat den Radius $r_0$ und nach Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-kreisbahn|(12)]] die Kreisbahngeschwindigkeit $v_{k,0} = \sqrt{\mu_P/r_0}$, die Fluchtgeschwindigkeit am selben Ort beträgt nach Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-flucht|(13)]] $v_{esc} = \sqrt{2\mu_P/r_0}$. Der Impuls wird wie beim Hohmann-Transfer tangential ausgeführt, sodass er allein den Betrag der Geschwindigkeit ändert und sein Ort zur Periapsis der Hyperbel wird, $r_p = r_0$ [33]. Entlang der Hyperbel bleibt nach Abschnitt 2.2 die Bahnenergie erhalten, die nach Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-exzess|(14)]] $\varepsilon = v_\infty^2/2$ beträgt. In der Periapsis folgt aus Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-energie|(9)]]

$$\frac{v_p^2}{2} - \frac{\mu_P}{r_0} = \frac{v_\infty^2}{2}$$

und mit $v_{esc}$ die Geschwindigkeit in der Periapsis [33]:

$$v_p^2 = v_{esc}^2 + v_\infty^2\tag{31}$$
^eq-periapsis-hyperbel

Das Quadrat der Periapsisgeschwindigkeit setzt sich aus einem Anteil, mit dem das Raumfahrzeug den Planeten gerade verlassen könnte, und dem Anteil der Exzessgeschwindigkeit zusammen. Addiert werden dabei die Quadrate der Geschwindigkeiten und damit die Energien. Gl. [[#^eq-periapsis-hyperbel|(31)]] überträgt so das nach Gl. [[#^eq-exzess-impulse|(30)]] bekannte $v_\infty$ auf den Ort des Impulses. Beim Abflug gibt sie an, wie schnell das Raumfahrzeug unmittelbar nach dem Impuls sein muss, damit es beim Aufstieg bis zur Grenze der Einflusssphäre von $v_p$ genau auf $v_\infty$ verlangsamt wird. Bei der Ankunft gibt sie an, auf welche Geschwindigkeit es beim Fall von der Grenze bis zur Periapsis anwächst. Da Kreisbahn und Hyperbel in der Periapsis dieselbe Richtung haben, ist der Impuls am Planeten die Differenz der Beträge [33], [37]:

$$\Delta v_P = v_p - v_{k,0} = \sqrt{\frac{2\mu_P}{r_0} + v_\infty^2} - \sqrt{\frac{\mu_P}{r_0}}\tag{32}$$
^eq-parkbahn-impuls

Beim Abflug beschleunigt dieser Impuls das Raumfahrzeug in der Parkbahn von $v_{k,0}$ auf $v_p$. Gezündet wird an der Stelle der Parkbahn, von der aus die Asymptote der Hyperbel parallel zur Bahngeschwindigkeit des Planeten verläuft, also die Gerade, der sich die Hyperbel in großer Entfernung annähert (Abb. 9, links) [33]. Die Ankunft verläuft spiegelbildlich. Das Raumfahrzeug tritt mit $v_{\infty,2}$ in die Einflusssphäre des Mars ein, und über den seitlichen Abstand seiner Anflugrichtung vom Planeten wird festgelegt, dass die Periapsis der Ankunftshyperbel auf dem Radius der Zielbahn liegt (Abb. 9, rechts) [37]. Ohne Manöver verließe es die Einflusssphäre entlang der Hyperbel wieder, mit unverändertem Betrag von $v_\infty$ [23]. Ein tangentialer Bremsimpuls in der Periapsis, wo er am kleinsten ausfällt, senkt die Geschwindigkeit von $v_p$ auf $v_{k,0}$, und das Raumfahrzeug schwenkt in die Zielbahn ein [37]. Sein Betrag folgt ebenfalls aus Gl. [[#^eq-parkbahn-impuls|(32)]].

![[Obsidian Vault/Evaluation/Grafiken & Bilder/abflug_ankunft_hyperbel.png|700]]
*Abb. 9, Abflughyperbel an der Erde (links) und Ankunftshyperbel am Mars (rechts) innerhalb der Einflusssphäre, nicht maßstäblich. Rechts daneben jeweils die Zusammensetzung der Geschwindigkeit an der Grenze der Einflusssphäre nach Gl. (29) – KI-erstellt (Claude)*

Für das Rechenbeispiel Erde–Mars werden die ungerundeten Impulse aus Abschnitt 3.1 nach Gl. [[#^eq-exzess-impulse|(30)]] als Exzessgeschwindigkeiten übernommen, also $v_{\infty,1} \approx 2\,944{,}7\ \mathrm{m/s}$ an der Erde und $v_{\infty,2} \approx 2\,648{,}9\ \mathrm{m/s}$ am Mars. Park- und Zielbahn sind Kreisbahnen in $300\ \mathrm{km}$ Höhe über dem Äquatorradius des jeweiligen Planeten, und wie in Abschnitt 3.1 liegen alle Bahnen in einer gemeinsamen Ebene. Die Größen aus Gl. [[#^eq-periapsis-hyperbel|(31)]] und [[#^eq-parkbahn-impuls|(32)]] erhalten im Beispiel zusätzlich den Index $E$ oder $M$ des jeweiligen Planeten. Wie in Abschnitt 3.1 wird mit ungerundeten Zwischenwerten weitergerechnet und erst das Ergebnis gerundet.

Die Parkbahn um die Erde hat den Radius $r_{0,E} = R_E + 300\ \mathrm{km} = 6{,}6781366 \cdot 10^{6}\ \mathrm{m}$. Nach Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-kreisbahn|(12)]] und [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-flucht|(13)]] betragen dort Kreisbahn- und Fluchtgeschwindigkeit

$$v_{k,0,E} = \sqrt{\frac{3{,}98600 \cdot 10^{14}}{6{,}6781366 \cdot 10^{6}}}\ \mathrm{m/s} \approx 7\,725{,}8\ \mathrm{m/s}, \qquad v_{esc,E} = \sqrt{\frac{2 \cdot 3{,}98600 \cdot 10^{14}}{6{,}6781366 \cdot 10^{6}}}\ \mathrm{m/s} \approx 10\,925{,}9\ \mathrm{m/s}$$

In der Periapsis der Abflughyperbel erreicht das Raumfahrzeug nach Gl. [[#^eq-periapsis-hyperbel|(31)]]

$$v_{p,E} = \sqrt{10\,925{,}9^2 + 2\,944{,}7^2}\ \mathrm{m/s} \approx 11\,315{,}7\ \mathrm{m/s}$$

und der Impuls in der Parkbahn beträgt nach Gl. [[#^eq-parkbahn-impuls|(32)]] $\Delta v_E = v_{p,E} - v_{k,0,E} \approx 3\,590{,}0\ \mathrm{m/s}$.

Die Zielbahn um den Mars hat den Radius $r_{0,M} = R_M + 300\ \mathrm{km} = 3{,}69619 \cdot 10^{6}\ \mathrm{m}$, und Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-kreisbahn|(12)]] und [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-flucht|(13)]] liefern

$$v_{k,0,M} = \sqrt{\frac{4{,}28284 \cdot 10^{13}}{3{,}69619 \cdot 10^{6}}}\ \mathrm{m/s} \approx 3\,404{,}0\ \mathrm{m/s}, \qquad v_{esc,M} = \sqrt{\frac{2 \cdot 4{,}28284 \cdot 10^{13}}{3{,}69619 \cdot 10^{6}}}\ \mathrm{m/s} \approx 4\,814{,}0\ \mathrm{m/s}$$

In der Periapsis der Ankunftshyperbel ist das Raumfahrzeug nach Gl. [[#^eq-periapsis-hyperbel|(31)]]

$$v_{p,M} = \sqrt{4\,814{,}0^2 + 2\,648{,}9^2}\ \mathrm{m/s} \approx 5\,494{,}6\ \mathrm{m/s}$$

schnell, und der Bremsimpuls beträgt nach Gl. [[#^eq-parkbahn-impuls|(32)]] $\Delta v_M = v_{p,M} - v_{k,0,M} \approx 2\,090{,}6\ \mathrm{m/s}$.

Zusammen ergibt sich nach Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-deltav-ges|(16)]] der Geschwindigkeitsaufwand

$$\Delta v = \left|\Delta v_E\right| + \left|\Delta v_M\right| \approx 5\,680{,}6\ \mathrm{m/s}$$

| Hohmann-Transfer Erde–Mars            | Impuls an der Erde      | Impuls am Mars          | Summe                   |
| ------------------------------------- | ----------------------- | ----------------------- | ----------------------- |
| Erdbahn zur Marsbahn (Abschnitt 3.1)  | $2{,}94\ \mathrm{km/s}$ | $2{,}65\ \mathrm{km/s}$ | $5{,}59\ \mathrm{km/s}$ |
| Parkbahn zur Zielbahn (Abschnitt 3.2) | $3{,}59\ \mathrm{km/s}$ | $2{,}09\ \mathrm{km/s}$ | $5{,}68\ \mathrm{km/s}$ |

*Tab. 1, Impulse und Geschwindigkeitsaufwand des Hohmann-Transfers Erde–Mars, heliozentrisch zwischen den Planetenbahnen und zwischen Kreisbahnen in $300\ \mathrm{km}$ Höhe um beide Planeten, gerundet*

Gerundet erfordert der Abflug aus der Parkbahn um die Erde einen Impuls von $\Delta v_E \approx 3{,}59\ \mathrm{km/s}$ und das Einschwenken in die Zielbahn um den Mars einen von $\Delta v_M \approx 2{,}09\ \mathrm{km/s}$, zusammen $\Delta v \approx 5{,}68\ \mathrm{km/s}$ (Tab. 1). Das sind nur rund $0{,}09\ \mathrm{km/s}$ mehr als die heliozentrische Summe aus Abschnitt 3.1, obwohl das Raumfahrzeug nun zusätzlich die Erde verlässt und vom Mars eingefangen wird. Allein das Verlassen der Erde aus der Parkbahn würde $v_{esc,E} - v_{k,0,E} \approx 3{,}20\ \mathrm{km/s}$ kosten, der Abflugimpuls liegt aber nur etwa $0{,}65\ \mathrm{km/s}$ über $v_{\infty,1}$. Am Mars fällt der Bremsimpuls sogar etwa $0{,}56\ \mathrm{km/s}$ kleiner aus als $v_{\infty,2}$.

Die Ursache ist der Oberth-Effekt aus Abschnitt 2.3 [22]. Nach Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-oberth|(17)]] bringt ein Impuls $\Delta v$ in Flugrichtung bei der Geschwindigkeit $v$ den Energiezuwachs $v\,\Delta v + \Delta v^2/2$. Würde das Raumfahrzeug die Erde zunächst mit dem Impuls $v_{esc,E} - v_{k,0,E}$ gerade verlassen, käme es nach Abschnitt 2.2 in großer Entfernung praktisch zur Ruhe. Ein zweiter Impuls brächte dort mit $v \approx 0$ nur noch $\Delta v^2/2$ und müsste deshalb selbst $v_{\infty,1}$ betragen, sodass beide zusammen etwa $6{,}14\ \mathrm{km/s}$ kosteten. In der Parkbahn wirkt der gesamte Impuls bei hoher Geschwindigkeit, und der Term $v\,\Delta v$ vergrößert seinen Energiebeitrag, weshalb dort $3{,}59\ \mathrm{km/s}$ genügen.

Ob der Impuls am Planeten größer oder kleiner als $v_\infty$ ausfällt, hängt allein vom Verhältnis $v_\infty/v_{k,0}$ ab. Mit $v_{esc} = \sqrt{2}\,v_{k,0}$ nach Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-flucht|(13)]] folgt aus Gl. [[#^eq-parkbahn-impuls|(32)]] durch Quadrieren

$$\Delta v_P < v_\infty \quad\Leftrightarrow\quad \sqrt{2\,v_{k,0}^2 + v_\infty^2} < v_{k,0} + v_\infty \quad\Leftrightarrow\quad v_\infty > \frac{v_{k,0}}{2}$$

An der Erde beträgt $v_{\infty,1}$ nur etwa $0{,}38\,v_{k,0,E}$ und liegt unter dieser Schwelle, am Mars beträgt $v_{\infty,2}$ etwa $0{,}78\,v_{k,0,M}$ und liegt darüber. Für den Bremsimpuls bei der Ankunft gilt dieselbe Rechnung, denn rückwärts betrachtet ist das Einschwenken in die Zielbahn ein Abflug aus ihr.

Mit beiden Impulsen ist der Geschwindigkeitsaufwand des Hohmann-Transfers zwischen den Umlaufbahnen um Erde und Mars bestimmt. Abschnitt 3.3 stellt ihm einen schnelleren Transfer gegenüber, bei dem die Transferellipse die Marsbahn schneidet. Dort sind die Geschwindigkeiten von Raumfahrzeug und Mars bei der Ankunft nicht mehr parallel, sodass sich $v_\infty$ nur noch als Vektordifferenz nach Gl. [[#^eq-geschwindigkeitsaddition|(29)]] bestimmen lässt.

## 3.3 Vergleich mit einem schnelleren Transfer
Nach Abschnitt 3.1 verbindet der Hohmann-Transfer zwei Kreisbahnen mit dem kleinsten Geschwindigkeitsaufwand, der mit zwei Impulsen möglich ist [34]. Mit der Transferellipse legt er nach Gl. [[#^eq-hohmann-flugzeit|(24)]] zugleich die Flugzeit fest, da das Raumfahrzeug stets die halbe Umlaufzeit dieser Ellipse unterwegs ist, zwischen Erd- und Marsbahn also rund 259 Tage. Da jede andere Transferbahn mit zwei Impulsen mehr Geschwindigkeitsaufwand erfordert, ist ein kürzerer Flug nur gegen einen höheren Aufwand möglich. Bei jedem Transfer stellt sich damit die Frage „günstiger oder schneller?“, also wie viel zusätzlicher Geschwindigkeitsaufwand ein bestimmter Zeitgewinn kostet.

Betrachtet wird hier ein schnellerer Transfer, der sich so wenig wie möglich vom Hohmann-Transfer unterscheidet. Das Raumfahrzeug startet wie dort tangential an der Erdbahn, sodass der Abflugpunkt die Periapsis der Transferellipse bleibt. Der erste Impuls fällt jedoch stärker aus und hebt die Apoapsis $r_a$ über den Radius der Marsbahn hinaus. Die Transferellipse erreicht die Marsbahn dadurch vor ihrer Apoapsis und schneidet sie im Punkt S bei der wahren Anomalie $\nu_2 < 180°$, statt sie zu berühren (Abb. 10). Um Start- und Zielbahn zu verbinden, genügt es, dass die Transferbahn beide schneidet [38]. Geflogen wird nur der Bogen von der Periapsis bis S. Die Apoapsis ist dabei der einzige Parameter, der sich gegenüber dem Hohmann-Transfer ändert. Liegt sie genau auf der Marsbahn, fällt S mit der Apoapsis zusammen, und es ergibt sich wieder der Hohmann-Transfer, der damit als Grenzfall in der folgenden Rechnung enthalten ist.

Für den Abflug ändert sich gegenüber den Abschnitten 3.1 und 3.2 allein die große Halbachse. Sie folgt aus Gl. [[#^eq-hohmann-halbachse|(20)]], wenn dort $r_2$ durch $r_a$ ersetzt wird, und die Exzentrizität der Transferellipse aus Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-exzentrizitaet|(6)]] mit $r_p = r_1$. Gl. [[#^eq-hohmann-vt|(21)]] liefert mit diesem $a_t$ die Geschwindigkeit $v_{t,1}$ an der Erdbahn. Da der Start tangential erfolgt, bewegen sich Raumfahrzeug und Erde wie beim Hohmann-Transfer parallel, und es gilt weiterhin $v_{\infty,1} = v_{t,1} - v_{k,1}$ [33]. Den Impuls in der Parkbahn liefert damit Gl. [[#^eq-parkbahn-impuls|(32)]].

Den Schwerpunkt bildet die Ankunft. Im Schnittpunkt S hat das Raumfahrzeug die Apoapsis noch nicht erreicht und ist weiterhin auf dem Weg nach außen. Seine Geschwindigkeit besitzt deshalb neben einem seitlichen Anteil senkrecht zur Verbindungslinie zur Sonne einen radialen Anteil entlang dieser Linie und ist gegen die Bahnrichtung des Mars geneigt, die auf dessen Kreisbahn rein seitlich verläuft. Den Betrag $v_2$ dieser Geschwindigkeit liefert die Vis-viva-Gleichung [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-visviva|(11)]] mit $r = r_2$ und der großen Halbachse $a_t$ der Transferellipse. Ihre Richtung folgt aus der Erhaltung des Drehimpulses. Nach Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-drehimpuls|(4)]] ist $h$ der Betrag eines Kreuzprodukts mit $\vec{r}$, in das allein der seitliche Anteil $v_\perp$ der Geschwindigkeit eingeht, sodass $h = r\,v_\perp$ gilt [11]. In der Periapsis ist die Geschwindigkeit rein seitlich, und es gilt $h = r_1\,v_{t,1}$. Da $h$ entlang der gesamten Bahn erhalten bleibt [15], beträgt der seitliche Anteil in S

$$v_\perp = \frac{h}{r_2} = \frac{r_1\,v_{t,1}}{r_2}\tag{33}$$
^eq-v-seitlich

Der radiale Anteil $v_r$ steht senkrecht auf $v_\perp$, sodass sich beide nach dem Satz des Pythagoras zu $v_2$ zusammensetzen. Der Winkel $\varphi$ zwischen der Geschwindigkeit und der Bahnrichtung des Mars folgt aus ihrem Verhältnis [11]:

$$v_r = \sqrt{v_2^2 - v_\perp^2}, \qquad \tan\varphi = \frac{v_r}{v_\perp}\tag{34}$$
^eq-v-radial

Wo S auf der Transferellipse liegt, ergibt sich aus der Bahngleichung [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-bahngleichung|(5)]]. Mit $r = r_2$ und $h^2/\mu = r_1(1+e)$ aus Abschnitt 2.1 folgt $\cos\nu_2 = \left(r_1(1+e)/r_2 - 1\right)/e$ [16].

Die Geschwindigkeit relativ zum Mars ist nach Gl. [[#^eq-geschwindigkeitsaddition|(29)]] die Vektordifferenz $\vec{v}_{\infty,2} = \vec{v} - \vec{v}_{k,2}$ an der Grenze der Einflusssphäre, die nach Abschnitt 3.2 im Maßstab der Transferellipse praktisch mit S zusammenfällt. Da $\vec{v}_{k,2}$ rein seitlich zeigt, bleibt vom seitlichen Anteil nur die Differenz $v_\perp - v_{k,2}$ übrig, während der radiale Anteil vollständig erhalten bleibt (Abb. 10, rechts). Gezählt wird der Betrag der Vektordifferenz, in den beide Anteile eingehen [38]:

$$v_{\infty,2} = \sqrt{\left(v_{k,2} - v_\perp\right)^2 + v_r^2}\tag{35}$$
^eq-exzess-schnitt

Für $r_a = r_2$ liegt S in der Apoapsis. Dort verschwindet nach Gl. [[#^eq-v-radial|(34)]] der radiale Anteil, $v_\perp$ stimmt mit $v_{t,2}$ überein, und Gl. [[#^eq-exzess-schnitt|(35)]] geht in $v_{\infty,2} = v_{k,2} - v_{t,2}$ aus Gl. [[#^eq-exzess-impulse|(30)]] über.

Der radiale Anteil ist genau die Bewegung nach außen, die den Flug verkürzt. Das Raumfahrzeug erreicht die Marsbahn, während es sich noch von der Sonne entfernt, und spart so den langsamen Abschnitt um die Apoapsis ein. Beim Hohmann-Transfer verschwindet dieser Anteil, weil die Marsbahn dort genau am größten Sonnenabstand der Transferellipse erreicht wird. Der Mars bewegt sich auf seiner Kreisbahn dagegen nicht nach außen und gleicht den radialen Anteil deshalb an keiner Stelle aus, sodass dieser unvermindert in $v_{\infty,2}$ eingeht.

Wie in Abschnitt 3.2 ist $v_{\infty,2}$ die Geschwindigkeit relativ zum Planeten an der Grenze seiner Einflusssphäre und kein Impuls. Den Bremsimpuls in der Zielbahn liefert erst Gl. [[#^eq-parkbahn-impuls|(32)]], in die nur der Betrag von $v_\infty$ eingeht. Die Richtung, aus der das Raumfahrzeug in die Einflusssphäre eintritt, legt lediglich die Lage der Ankunftshyperbel um den Mars fest, deren Periapsis wie in Abschnitt 3.2 über den seitlichen Abstand der Anflugrichtung auf den Radius der Zielbahn gelegt wird [37].

Für das Rechenbeispiel Erde–Mars gelten dieselben Werte wie in den Abschnitten 3.1 und 3.2, also die Gravitationsparameter und Bahnradien aus [17] sowie Park- und Zielbahn in $300\ \mathrm{km}$ Höhe über dem Äquatorradius. Die Apoapsis wird in astronomischen Einheiten (AE) angegeben. Eine AE entspricht nahezu genau dem mittleren Sonnenabstand der Erde, sodass hier $1\ \text{AE} = r_1$ gesetzt wird und die Marsbahn bei etwa $1{,}52\ \text{AE}$ liegt. Gerechnet wird für $r_a = 1{,}75\ \text{AE}$ und $r_a = 2{,}0\ \text{AE}$, wobei der Rechenweg für $1{,}75\ \text{AE}$ ausgeschrieben ist. Wie bisher wird mit ungerundeten Zwischenwerten weitergerechnet.

Mit $r_a = 1{,}75\,r_1 = 2{,}617965 \cdot 10^{11}\ \mathrm{m}$ hat die Transferellipse nach Gl. [[#^eq-hohmann-halbachse|(20)]] und [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-exzentrizitaet|(6)]] die große Halbachse und die Exzentrizität

$$a_t = \frac{1{,}49598 \cdot 10^{11} + 2{,}617965 \cdot 10^{11}}{2}\ \mathrm{m} = 2{,}0569725 \cdot 10^{11}\ \mathrm{m}, \qquad e = \frac{2{,}617965 - 1{,}49598}{2{,}617965 + 1{,}49598} \approx 0{,}272727$$

An der Erdbahn bewegt sich das Raumfahrzeug nach Gl. [[#^eq-hohmann-vt|(21)]] mit

$$v_{t,1} = \sqrt{1{,}32712 \cdot 10^{20}\left(\frac{2}{1{,}49598 \cdot 10^{11}} - \frac{1}{2{,}0569725 \cdot 10^{11}}\right)}\ \mathrm{m/s} \approx 33\,601{,}6\ \mathrm{m/s}$$

und entfernt sich von der Erde mit $v_{\infty,1} = v_{t,1} - v_{k,1} \approx 3\,817{,}0\ \mathrm{m/s}$. Mit $v_{esc,E}$ und $v_{k,0,E}$ aus Abschnitt 3.2 ergeben Gl. [[#^eq-periapsis-hyperbel|(31)]] die Periapsisgeschwindigkeit $v_{p,E} \approx 11\,573{,}4\ \mathrm{m/s}$ und Gl. [[#^eq-parkbahn-impuls|(32)]] den Impuls in der Parkbahn $\Delta v_E \approx 3\,847{,}7\ \mathrm{m/s}$.

An der Marsbahn liefert die Vis-viva-Gleichung [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-visviva|(11)]] den Betrag der Geschwindigkeit

$$v_2 = \sqrt{1{,}32712 \cdot 10^{20}\left(\frac{2}{2{,}27939 \cdot 10^{11}} - \frac{1}{2{,}0569725 \cdot 10^{11}}\right)}\ \mathrm{m/s} \approx 22\,787{,}5\ \mathrm{m/s}$$

und Gl. [[#^eq-v-seitlich|(33)]] und [[#^eq-v-radial|(34)]] seine beiden Anteile

$$v_\perp = \frac{1{,}49598 \cdot 10^{11} \cdot 33\,601{,}6}{2{,}27939 \cdot 10^{11}}\ \mathrm{m/s} \approx 22\,053{,}0\ \mathrm{m/s}, \qquad v_r = \sqrt{22\,787{,}5^2 - 22\,053{,}0^2}\ \mathrm{m/s} \approx 5\,739{,}1\ \mathrm{m/s}$$

Die Geschwindigkeit ist damit um $\varphi \approx 14{,}6°$ gegen die Bahnrichtung des Mars geneigt. Der Schnittpunkt liegt bei $\cos\nu_2 \approx -0{,}603900$, also $\nu_2 \approx 127{,}1°$. Nach Gl. [[#^eq-exzess-schnitt|(35)]] tritt das Raumfahrzeug mit

$$v_{\infty,2} = \sqrt{\left(24\,129{,}4 - 22\,053{,}0\right)^2 + 5\,739{,}1^2}\ \mathrm{m/s} \approx 6\,103{,}2\ \mathrm{m/s}$$

in die Einflusssphäre des Mars ein. Mit $v_{esc,M}$ und $v_{k,0,M}$ aus Abschnitt 3.2 folgen aus Gl. [[#^eq-periapsis-hyperbel|(31)]] $v_{p,M} \approx 7\,773{,}2\ \mathrm{m/s}$ und aus Gl. [[#^eq-parkbahn-impuls|(32)]] der Bremsimpuls $\Delta v_M \approx 4\,369{,}2\ \mathrm{m/s}$. Zusammen beträgt der Geschwindigkeitsaufwand $\Delta v = \left|\Delta v_E\right| + \left|\Delta v_M\right| \approx 8\,216{,}9\ \mathrm{m/s}$.

Für $r_a = 2{,}0\ \text{AE}$ liefert derselbe Rechenweg eine Transferellipse mit $a_t = 2{,}24397 \cdot 10^{11}\ \mathrm{m}$ und $e \approx 0{,}333333$. An der Erdbahn ergeben sich $v_{t,1} \approx 34\,392{,}3\ \mathrm{m/s}$, $v_{\infty,1} \approx 4\,607{,}7\ \mathrm{m/s}$ und $\Delta v_E \approx 4\,132{,}0\ \mathrm{m/s}$. An der Marsbahn betragen $v_2 \approx 23\,938{,}2\ \mathrm{m/s}$, $v_\perp \approx 22\,571{,}9\ \mathrm{m/s}$ und $v_r \approx 7\,971{,}4\ \mathrm{m/s}$, sodass die Geschwindigkeit um $\varphi \approx 19{,}5°$ geneigt ist und der Schnittpunkt bei $\nu_2 \approx 112{,}0°$ liegt. Daraus folgen $v_{\infty,2} \approx 8\,122{,}1\ \mathrm{m/s}$, $\Delta v_M \approx 6\,037{,}6\ \mathrm{m/s}$ und ein Geschwindigkeitsaufwand von $\Delta v \approx 10\,169{,}6\ \mathrm{m/s}$.

Die Flugzeit bis S lässt sich nicht wie in Gl. [[#^eq-hohmann-flugzeit|(24)]] als Bruchteil der Umlaufzeit angeben, da nur ein Teil der Ellipse durchlaufen wird und das Raumfahrzeug auf ihr nicht gleichmäßig schnell ist. Sie folgt aus der Keplergleichung, die die Zeit seit dem Durchgang durch die Periapsis mit der wahren Anomalie verknüpft und hier ohne Herleitung verwendet wird [39]. Da der Abflug in der Periapsis erfolgt, ist diese Zeit gleich der Flugzeit. Für $\nu_2 \approx 127{,}1°$ bei $1{,}75\ \text{AE}$ ergibt sie etwa 162 Tage, für $\nu_2 \approx 112{,}0°$ bei $2{,}0\ \text{AE}$ etwa 137 Tage. Ob die Simulation diese Flugzeiten bestätigt, prüft Abschnitt 3.4.

| Transfer Erde–Mars        | Apoapsis              | Flugzeit  | $v_{\infty,1}$ in km/s | $v_{\infty,2}$ in km/s | Impuls an der Erde in km/s | Impuls am Mars in km/s | Summe in km/s |
| ------------------------- | --------------------- | --------- | ---------------------- | ---------------------- | -------------------------- | ---------------------- | ------------- |
| Hohmann-Transfer (Tab. 1) | $1{,}52\ \text{AE}$   | 259 Tage  | $2{,}94$               | $2{,}65$               | $3{,}59$                   | $2{,}09$               | $5{,}68$      |
| schnellerer Transfer      | $1{,}75\ \text{AE}$   | 162 Tage  | $3{,}82$               | $6{,}10$               | $3{,}85$                   | $4{,}37$               | $8{,}22$      |
| schnellerer Transfer      | $2{,}0\ \text{AE}$    | 137 Tage  | $4{,}61$               | $8{,}12$               | $4{,}13$                   | $6{,}04$               | $10{,}17$     |

*Tab. 2, Flugzeit, Exzessgeschwindigkeiten und Impulse des Transfers Erde–Mars zwischen Kreisbahnen in $300\ \mathrm{km}$ Höhe um beide Planeten für drei Apoapsiden der Transferellipse, gerundet*

Gegenüber dem Hohmann-Transfer verkürzt sich die Flugzeit auf 162 beziehungsweise 137 Tage, während der Geschwindigkeitsaufwand von $5{,}68$ auf $8{,}22$ beziehungsweise $10{,}17\ \mathrm{km/s}$ steigt (Tab. 2). Der größte Teil dieses Anstiegs entsteht bei der Ankunft. An der Erde wächst $v_{\infty,1}$ nur von $2{,}94$ auf $3{,}82$ beziehungsweise $4{,}61\ \mathrm{km/s}$, am Mars wächst $v_{\infty,2}$ dagegen von $2{,}65$ auf $6{,}10$ beziehungsweise $8{,}12\ \mathrm{km/s}$.

Dieser Zuwachs geht vollständig auf den radialen Anteil zurück. Der seitliche Unterschied $v_{k,2} - v_\perp$ sinkt sogar von $2{,}65$ auf $2{,}08$ beziehungsweise $1{,}56\ \mathrm{km/s}$, da der stärkere erste Impuls den Drehimpuls und nach Gl. [[#^eq-v-seitlich|(33)]] auch $v_\perp$ vergrößert. Der radiale Anteil steigt dagegen von null auf $5{,}74$ beziehungsweise $7{,}97\ \mathrm{km/s}$. Da beide Anteile in Gl. [[#^eq-exzess-schnitt|(35)]] quadratisch eingehen, stammen bei $1{,}75\ \text{AE}$ rund $88\ \%$ und bei $2{,}0\ \text{AE}$ rund $96\ \%$ von $v_{\infty,2}^2$ aus dem radialen Anteil. Der Bremsimpuls wächst schwächer als $v_{\infty,2}$, von $2{,}09$ auf $4{,}37$ beziehungsweise $6{,}04\ \mathrm{km/s}$, da $v_{\infty,2}$ die Schwelle $v_{k,0,M}/2 \approx 1{,}70\ \mathrm{km/s}$ aus Abschnitt 3.2 weit überschreitet und der Oberth-Effekt den Impuls unter $v_{\infty,2}$ drückt. Den Mehraufwand gleicht das nur zum Teil aus.

![[Obsidian Vault/Evaluation/Grafiken & Bilder/schnittbahn_transfer.png|700]]
*Abb. 10, Hohmann-Transfer und schnellerer Transfer mit der Apoapsis $1{,}75\ \text{AE}$ zwischen Erd- und Marsbahn, maßstäblich (links). Die Transferellipse des schnelleren Transfers schneidet die Marsbahn im Punkt S bei der wahren Anomalie $\nu_2$. Rechts die Zerlegung der Geschwindigkeiten in S nach Gl. (29) und (33) bis (35) – KI-erstellt (Claude)*

Abb. 11 zeigt den Geschwindigkeitsaufwand über der Flugzeit für Apoapsiden von der Marsbahn bis $3{,}2\ \text{AE}$. Die Kurve beginnt beim Hohmann-Transfer waagerecht. Rückt die Apoapsis nur auf $1{,}53\ \text{AE}$ und damit knapp über die Marsbahn hinaus, verkürzt sich die Flugzeit bereits um rund 21 Tage, während der Geschwindigkeitsaufwand um weniger als $0{,}1\ \mathrm{km/s}$ steigt. Um die Apoapsis ändert sich der Abstand zur Sonne nur langsam, sodass schon eine geringfügig größere Apoapsis diesen zeitraubenden Abschnitt abschneidet. Der dabei entstehende radiale Anteil ist noch klein und erhöht $v_{\infty,2}$ wegen der quadratischen Addition in Gl. [[#^eq-exzess-schnitt|(35)]] kaum. Mit wachsender Apoapsis rückt der Schnittpunkt dagegen nur noch wenig nach vorn, während der radiale Anteil weiter zunimmt und $v_{\infty,2}$ nahezu allein bestimmt. In den ersten drei Wochen Zeitgewinn kostet jeder Tag im Mittel etwa $4\ \mathrm{m/s}$, zwischen $1{,}75$ und $2{,}0\ \text{AE}$ bereits rund $79\ \mathrm{m/s}$.

![[Obsidian Vault/Evaluation/Grafiken & Bilder/aufwand_flugzeit.png|700]]
*Abb. 11, Geschwindigkeitsaufwand des Transfers zwischen Kreisbahnen in $300\ \mathrm{km}$ Höhe um Erde und Mars über der Flugzeit für Apoapsiden von der Marsbahn bis $3{,}2\ \text{AE}$, mit den Impulsen an beiden Planeten – Rechnung und Grafik KI-erstellt (Claude)*

In der Nähe des Hohmann-Transfers lässt sich Flugzeit demnach fast ohne Mehraufwand gewinnen, weiter von ihm entfernt wird jeder zusätzliche Tag deutlich teurer. Daraus ergibt sich, wann welcher Transfer geeignet ist. Bestimmt allein der Geschwindigkeitsaufwand die Kosten eines Flugs, weil er nach Gl. [[Obsidian Vault/Evaluation/_Arbeit/02 Physikalische Grundlagen#^eq-raketengleichung|(18)]] die mitzuführende Masse festlegt, und verursacht die Flugzeit selbst keine Kosten, ist der Hohmann-Transfer die beste Wahl. Verursacht auch die Flugzeit Kosten, lohnt sich ein schnellerer Transfer, solange der Zeitgewinn den zusätzlichen Geschwindigkeitsaufwand aufwiegt. Für eine geringe Abweichung vom Hohmann-Transfer gilt das schon, wenn die Flugzeit nur wenig ins Gewicht fällt. Eine starke Verkürzung ist dagegen nur gerechtfertigt, wenn die Flugzeit sehr viel schwerer wiegt als der Geschwindigkeitsaufwand.

Damit ist die erste Teilfrage als Rechenwert beantwortet. Zwischen den Umlaufbahnen um Erde und Mars senkt der Hohmann-Transfer den Geschwindigkeitsaufwand gegenüber den schnelleren Transfers mit einer Apoapsis von $1{,}75$ und $2{,}0\ \text{AE}$ um $2{,}54$ beziehungsweise $4{,}49\ \mathrm{km/s}$, also um rund $31$ beziehungsweise $44\ \%$, und verlängert die Flugzeit dafür um 97 beziehungsweise 122 Tage. Wie genau sich diese Werte in einer eigenen Simulation reproduzieren lassen, untersucht Abschnitt 3.4.
