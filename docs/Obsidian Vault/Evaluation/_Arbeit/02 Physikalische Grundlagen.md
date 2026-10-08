# 2. Physikalische Grundlagen
Die Himmelsmechanik beschreibt die Bewegung von Körpern unter gegenseitiger Gravitation; die Astrodynamik wendet diese Beschreibung auf Raumfahrzeuge an und ergänzt sie um die gezielte Bahnänderung durch Antrieb. Beide gehen auf Keplers Beschreibung der Planetenbahnen und Newtons Gravitationsgesetz zurück [Curtis, Kap. 2]. Dieses Kapitel führt die Begriffe ein, mit denen die Transferrechnungen in Kapitel 3 und die Gravitationsmanöver in Kapitel 4 arbeiten, und benennt an welcher Stelle die verwendete Simulation vereinfacht.
## 2.1 Gravitation und das Zweikörperproblem
Isaac Newton (1643–1727) legte 1687 in seiner *Philosophiae Naturalis Principia Mathematica* die Grundlagen der klassischen Mechanik fest. Für die Bahnmechanik sind zwei seiner drei Bewegungsgesetze entscheidend [1, S. 2]. Nach dem zweiten Gesetz ist die Impulsänderung eines Körpers der einwirkenden Kraft proportional, was für konstante Masse $\vec{F} = m\vec{a}$ ergibt. Nach dem dritten Gesetz wirken zwei Körper stets mit gleich großen, entgegengesetzt gerichteten Kräften aufeinander.

Außerdem formulierte Newton das Gravitationsgesetz, das die gegenseitige Anziehung zweier Massen beschreibt [8]:

$$F = G\,\frac{m_1 m_2}{r^2}\tag{1}$$
^eq-gravitation

$F$ ... Betrag der Gravitationskraft, $G$ ... Gravitationskonstante, $m_1, m_2$ ... Massen der beiden Körper, $r$ ... Abstand zwischen den Körpern

Nach Gl. [[#^eq-gravitation|(1)]] wächst die Kraft mit dem Produkt der Massen und nimmt mit dem Quadrat des Abstands ab [8]. Die Gravitationskonstante $G$ beträgt rund $6{,}674 \cdot 10^{-11}\ \mathrm{m^3/(kg\,s^2)}$ und legt die Stärke der Gravitation fest [8]. Gl. [[#^eq-gravitation|(1)]] gilt streng für Punktmassen. Für kugelsymmetrische Körper, wie es die Erde näherungsweise ist, gilt sie außerhalb des Körpers ebenso, da ihre Gravitation dort wirkt, als säße die gesamte Masse im Mittelpunkt [18].

Nach dem ersten Keplerschen Gesetz ist die Bahn eines Planeten eine Ellipse, in deren einem Brennpunkt die Sonne steht [1, S. 3]. Die Punkte einer Bahn mit dem kleinsten und dem größten Abstand zum Zentralkörper heißen **Periapsis** und **Apoapsis**, zusammen **Apsiden** [1, S. 4]. Keplers Beschreibung behandelt den Zentralkörper als ruhend. Nach dem dritten Newtonschen Gesetz wirkt die Gravitationskraft jedoch mit gleichem Betrag und entgegengesetzter Richtung auf beide Körper, sodass sich auch der Zentralkörper bewegt. Um beide Bewegungen zu erfassen, werden die Körper durch ihre Ortsvektoren $\vec{R}_1$ und $\vec{R}_2$ in einem Inertialsystem beschrieben, da das zweite Newtonsche Gesetz nur dort gilt. Der relative Ortsvektor $\vec{r} = \vec{R}_2 - \vec{R}_1$ zeigt von Körper 1 zu Körper 2 und hat die Länge $r$.

Gl. [[#^eq-gravitation|(1)]] liefert nur den Betrag der Kraft. Teilt man $\vec{r}$ durch seine Länge $r$, erhält man ihre Richtung, und die Kraft auf Körper 1 lautet in Vektorschreibweise [Curtis, Abschn. 2.2, Gl. 2.9, S. 57]:

$$\vec{F}_1 = \frac{G\,m_1 m_2}{r^2}\cdot\frac{\vec{r}}{r} = \frac{G\,m_1 m_2}{r^3}\,\vec{r}\tag{2}$$
^eq-kraftvektor

Wendet man Gl. [[#^eq-kraftvektor|(2)]] und die Gegenkraft $-\vec{F}_1$ nach dem zweiten Newtonschen Gesetz auf beide Körper an und bildet die Differenz ihrer Beschleunigungen, folgt die Beschleunigung von Körper 2 relativ zu Körper 1 [Curtis, Abschn. 2.3, Gl. 2.20–2.22, S. 63] [14]:

$$\ddot{\vec{r}} = -G\,(m_1 + m_2)\,\frac{\vec{r}}{r^3} = -\frac{\mu}{r^3}\,\vec{r}, \qquad \mu = G\,(m_1 + m_2)\tag{3}$$
^eq-zweikoerper

Das Minuszeichen richtet die Beschleunigung auf Körper 1 aus. Die Massen treten nur noch als Summe auf und bilden zusammen mit $G$ den **Gravitationsparameter** $\mu$. Da ein Raumfahrzeug gegenüber einem Himmelskörper eine verschwindend kleine Masse besitzt, gilt $\mu \approx G\,m_1$, und $\mu$ wird zu einer Eigenschaft des Zentralkörpers allein [14]. Für die Erde beträgt $\mu = 3{,}98600 \cdot 10^{14}\ \mathrm{m^3/s^2}$ [17]. Dieser Wert ist genauer bekannt als $G$ und die Erdmasse einzeln, weil er sich direkt aus beobachteten Bahnen bestimmen lässt [14]. Gl. [[#^eq-zweikoerper|(3)]] ist die Bewegungsgleichung des Zweikörperproblems, und die Gravitation geht ab hier nur noch über das $\mu$ des jeweiligen Zentralkörpers ein.

Kepler leitete die Ellipsenform aus Beobachtungen ab. Dass sie aus Gl. [[#^eq-zweikoerper|(3)]] folgt, zeigt sich über den spezifischen Drehimpuls:

$$\vec{h} = \vec{r} \times \dot{\vec{r}}\tag{4}$$
^eq-drehimpuls

Da die Gravitation stets entlang von $\vec{r}$ wirkt, kann sie $\vec{h}$ nicht verändern, sodass Richtung und Betrag $h = |\vec{h}|$ konstant bleiben [15]. Der Vektor $\vec{h}$ steht senkrecht auf $\vec{r}$ und $\dot{\vec{r}}$, deshalb bleibt die gesamte Bahn in einer festen Ebene [15]. Für das Zweikörperproblem ist eine Beschreibung in der Ebene damit vollständig. Abb. 1 zeigt die Größen, mit denen eine Ellipsenbahn in dieser Ebene beschrieben wird.

![[Obsidian Vault/Evaluation/Grafiken & Bilder/elliptical-orbit-definitions.svg|450]]
*Abb. 1, Geometrie und Bezeichnungen einer Ellipsenbahn – [10]*

In einem der Brennpunkte, in Abb. 1 dem Punkt $F$, steht der Zentralkörper $m_1$, und $m_2$ umläuft ihn auf der Ellipse [11]. Der Abstand zwischen einer Apside und dem Mittelpunkt $C$ ist die **große Halbachse** $a$ [10]. Die **wahre Anomalie** $\nu$ ist der Winkel am Brennpunkt $F$ zwischen der Richtung zur Periapsis und der momentanen Position von $m_2$, in Abb. 1 für den Punkt $B$ als $\beta$ eingezeichnet [10]. Mit diesen Größen lässt sich die Bahngleichung angeben. Sie folgt durch Integration aus Gl. [[#^eq-zweikoerper|(3)]], wenn man die Erhaltung von $\vec{h}$ nutzt [16]:

$$r = \frac{h^2}{\mu}\,\frac{1}{1 + e\cos\nu}\tag{5}$$
^eq-bahngleichung

Gl. [[#^eq-bahngleichung|(5)]] gibt den Abstand $r$ zum Zentralkörper in Abhängigkeit von der wahren Anomalie $\nu$ an. Dabei bestimmen $h$ und $\mu$ die Größe der Bahn, die **Exzentrizität** $e$ ihre Form [16]. An der Periapsis ist $\nu = 0°$ und $r$ am kleinsten, an der Apoapsis ist $\nu = 180°$ und $r$ am größten. Das ergibt $r_p = \frac{h^2}{\mu}\frac{1}{1+e}$ und $r_a = \frac{h^2}{\mu}\frac{1}{1-e}$. Setzt man $h^2/\mu = r_p(1+e) = r_a(1-e)$ gleich, fällt $h$ heraus, und die Exzentrizität folgt allein aus den beiden Apsidenabständen [10]:

$$e = \frac{r_a - r_p}{r_a + r_p}\tag{6}$$
^eq-exzentrizitaet

Die Exzentrizität legt fest, welcher Kegelschnitt vorliegt. Für $e = 0$ ist die Bahn ein Kreis [12], für $0 < e < 1$ eine Ellipse [10], für $e = 1$ eine Parabel [13] und für $e > 1$ eine Hyperbel [16].

Bei einer Ellipse entspricht die Strecke von der Periapsis zur Apoapsis entlang der Apsidenlinie der doppelten großen Halbachse [10]:

$$2a = r_p + r_a\tag{7}$$
^eq-halbachse

Die große Halbachse bestimmt außerdem die Umlaufzeit. Nach dem dritten Keplerschen Gesetz hängt sie allein von $a$ (und $\mu$ bei gegebenem Zentralkörper) ab und ist von der Exzentrizität unabhängig [10]:

$$T = \frac{2\pi}{\sqrt{\mu}}\,a^{3/2}\tag{8}$$
^eq-umlaufzeit

Vollständig beschrieben ist eine Bahn im Raum durch sechs klassische Bahnelemente (Abb. 2): große Halbachse $a$, Exzentrizität $e$, Inklination $i$, Rektaszension des aufsteigenden Knotens $\Omega$, Argument der Periapsis $\omega$ und Zeitpunkt des Periapsisdurchgangs $t_p$ [19]. Liegt die Bahn in der Bezugsebene, ist die Inklination null und der aufsteigende Knoten nicht definiert [19]. In einer ebenen Beschreibung genügen deshalb vier Elemente. Das begründet die Beschränkung der Simulation auf zwei Dimensionen und schließt Manöver zur Änderung der Bahnebene aus dieser Untersuchung aus.

![[Obsidian Vault/Evaluation/Grafiken & Bilder/bahnelemente_3d.png|450]]
*Abb. 2, Bahnelemente einer Bahn im Raum mit Knotenlinie, Inklination $i$, Rektaszension des aufsteigenden Knotens $\Omega$, Argument der Periapsis $\omega$ und wahrer Anomalie $\nu$ – KI-erstellt (Claude)*

Verändern lässt sich eine Bahn gezielt, indem ein Raumfahrzeug an einer Apside seine Energie ändert. Mehr Energie an der Apoapsis hebt die Periapsis an, weniger Energie senkt sie [1, S. 4]. An der Periapsis wirkt eine Energieänderung entsprechend auf die Apoapsis [1, S. 4]. Wie die Energie einer Bahn mit ihrer großen Halbachse zusammenhängt, zeigt Abschnitt 2.2.

## 2.2 Bahnenergie und große Halbachse
Neben dem Drehimpuls bleibt bei der Relativbewegung eine zweite Größe erhalten, die Energie. Da die Masse des Raumfahrzeugs für seine Bahn keine Rolle spielt, betrachtet man die Energie pro Kilogramm, die **spezifische Bahnenergie** $\varepsilon$. Sie setzt sich aus der kinetischen Energie $v^2/2$ mit $v = |\dot{\vec{r}}|$ und der potentiellen Energie $-\mu/r$ zusammen, deren Nullpunkt in unendlicher Entfernung liegt [20]:

$$\varepsilon = \frac{v^2}{2} - \frac{\mu}{r}\tag{9}$$
^eq-energie

Der Nullpunkt der potentiellen Energie ist frei wählbar, da physikalisch nur Energieänderungen eine Rolle spielen [20]. Liegt er in unendlicher Entfernung, ist die potentielle Energie überall sonst negativ, weshalb auch $\varepsilon$ negativ werden kann, obwohl die kinetische Energie selbst nie negativ ist.

Multipliziert man Gl. [[#^eq-zweikoerper|(3)]] skalar mit $\dot{\vec{r}}$, lässt sich die linke Seite als zeitliche Ableitung von $v^2/2$ und die rechte als zeitliche Ableitung von $\mu/r$ schreiben. Daraus folgt $\mathrm{d}\varepsilon/\mathrm{d}t = 0$, sodass $\varepsilon$ entlang der gesamten Bahn konstant bleibt [20]. Mit wachsendem Abstand nimmt die potentielle Energie $-\mu/r$ zu, sodass die kinetische Energie im selben Maß abnimmt. Das Raumfahrzeug ist deshalb an der Periapsis am schnellsten und an der Apoapsis am langsamsten.

Da $\varepsilon$ überall denselben Wert hat, genügt es, sie an einem einzigen Punkt der Bahn auszuwerten. An der Periapsis steht die Geschwindigkeit senkrecht auf $\vec{r}$, sodass sich Gl. [[#^eq-drehimpuls|(4)]] zu $h = r_p v_p$ vereinfacht. Aus Gl. [[#^eq-bahngleichung|(5)]] folgt für $\nu = 0°$ zudem $h^2 = \mu\,r_p(1+e)$. Eingesetzt in Gl. [[#^eq-energie|(9)]] ergibt sich

$$\varepsilon = \frac{h^2}{2r_p^2} - \frac{\mu}{r_p} = \frac{\mu\,(e-1)}{2r_p}$$

Aus Gl. [[#^eq-bahngleichung|(5)]] und [[#^eq-halbachse|(7)]] folgt $r_p = a(1-e)$, womit sich die Exzentrizität herauskürzt [10]:

$$\varepsilon = -\frac{\mu}{2a}\tag{10}$$
^eq-energie-halbachse

Die Energie einer Bahn hängt damit allein von ihrer großen Halbachse ab und ist von der Exzentrizität unabhängig [10]. Eine Kreisbahn und eine stark gestreckte Ellipse mit gleichem $a$ besitzen also dieselbe Energie, so wie sie nach Gl. [[#^eq-umlaufzeit|(8)]] auch dieselbe Umlaufzeit haben. Umgekehrt lässt sich $a$ nur verändern, indem die Energie der Bahn verändert wird. Jede Vergrößerung oder Verkleinerung einer Bahn ist somit ein Energiewechsel, und auf diesem Zusammenhang bauen die Transferrechnungen in Kapitel 3 auf.

Die Energiegleichung [[#^eq-energie|(9)]] wird auch als Vis-viva-Gleichung bezeichnet [20]. Meist ist damit die Form gemeint, die sich ergibt, wenn man sie mit Gl. [[#^eq-energie-halbachse|(10)]] gleichsetzt und nach $v$ auflöst. Sie liefert die Geschwindigkeit an jedem Punkt einer Bahn aus dem Abstand zum Zentralkörper und der großen Halbachse:

$$v^2 = \mu\left(\frac{2}{r} - \frac{1}{a}\right)\tag{11}$$
^eq-visviva

Mit Gl. [[#^eq-visviva|(11)]] werden in Kapitel 3 die Geschwindigkeiten vor und nach einem Manöver berechnet, wobei zwei Sonderfälle wiederholt gebraucht werden. Auf einer Kreisbahn gilt $r = a$, und Gl. [[#^eq-visviva|(11)]] vereinfacht sich zur **Kreisbahngeschwindigkeit** [12]:

$$v_k = \sqrt{\frac{\mu}{r}}\tag{12}$$
^eq-kreisbahn

Lässt man $a$ gegen unendlich gehen, strebt $\varepsilon$ gegen null, und das Raumfahrzeug erreicht gerade noch unendliche Entfernung. Die dafür nötige **Fluchtgeschwindigkeit** liegt um den Faktor $\sqrt{2}$ über der Kreisbahngeschwindigkeit am selben Ort [13]:

$$v_{esc} = \sqrt{\frac{2\mu}{r}} = \sqrt{2}\,v_k\tag{13}$$
^eq-flucht

![[Obsidian Vault/Evaluation/Grafiken & Bilder/energiediagramm.png|450]]
*Abb. 3, Energiediagramm mit der potentiellen Energie $-\mu/r$ und der Bahnenergie $\varepsilon$ für die drei Bahnformen. Der senkrechte Abstand zur Kurve ist die kinetische Energie. Der Umkehrpunkt gibt nur eine Obergrenze des Abstands an, da die Bewegung quer zum Radius vernachlässigt ist – KI-erstellt (Claude)*

In Abschnitt 2.1 wurden die Kegelschnitte über die Exzentrizität unterschieden. Dieselbe Einteilung ergibt sich über das Vorzeichen von $\varepsilon$ (Abb. 3), das zusätzlich zeigt, ob der Körper dem Zentralkörper entkommen kann. In unendlicher Entfernung wäre die potentielle Energie null, und da die kinetische Energie nicht negativ sein kann, hätte ein Körper, der dort ankommt, stets $\varepsilon \geq 0$. Ist $\varepsilon$ negativ, reicht die Bewegungsenergie deshalb an keinem Punkt der Bahn aus, um das Unendliche zu erreichen. Der Körper bleibt gebunden und bewegt sich auf einer Ellipse oder einem Kreis, was mit Gl. [[#^eq-energie-halbachse|(10)]] übereinstimmt, die für jedes endliche $a$ einen negativen Wert liefert. Für $\varepsilon = 0$ kommt er dort gerade zur Ruhe, und die Bahn ist eine Parabel [13]. Für $\varepsilon > 0$ bleibt ihm auch im Unendlichen Bewegungsenergie übrig, und die Bahn verläuft als Hyperbel [21]. Zählt man $a$ bei der Hyperbel als positive Länge, kehrt sich in Gl. [[#^eq-energie-halbachse|(10)]] und [[#^eq-visviva|(11)]] das Vorzeichen des Terms mit $a$ um, sodass dort $\varepsilon = \mu/(2a)$ gilt [21]. Die Geschwindigkeit, die im Unendlichen übrig bleibt, heißt **hyperbolische Exzessgeschwindigkeit** $v_\infty$. Da dort der Term $\mu/r$ verschwindet, folgt aus Gl. [[#^eq-energie|(9)]] [21]:

$$v_\infty^2 = 2\,\varepsilon\tag{14}$$
^eq-exzess

Bei einem Vorbeiflug ohne Schub bleibt der Betrag von $v_\infty$ relativ zum Planeten erhalten, lediglich ihre Richtung wird gedreht [23]. Damit wird $v_\infty$ zur entscheidenden Größe für die Gravitationsmanöver in Kapitel 4.

Die Bahn eines Raumfahrzeugs lässt sich folglich nur verändern, indem sich seine Energie ändert. Nach Gl. [[#^eq-energie|(9)]] ist das an einem festen Ort allein über die Geschwindigkeit möglich. Wie eine solche Geschwindigkeitsänderung beschrieben und als Aufwand gemessen wird, behandelt Abschnitt 2.3.

## 2.3 Geschwindigkeitsaufwand als Maß
Um die Geschwindigkeit eines Raumfahrzeugs zu ändern, werden seine Triebwerke gezündet. Dauert der Brennvorgang nur kurz im Vergleich zur Zeit, in der das Raumfahrzeug antriebslos fliegt, spricht man von einem **impulsiven Manöver**. Dabei wird angenommen, dass sich der Geschwindigkeitsvektor in Betrag und Richtung schlagartig ändert, während der Ort des Raumfahrzeugs unverändert bleibt [24]. Diese Idealisierung erspart es, die Bewegungsgleichung mit dem Schub der Triebwerke zu lösen, und ist zulässig, solange sich das Raumfahrzeug während des Brennvorgangs kaum weiterbewegt, wie bei Triebwerken mit hohem Schub und kurzer Brenndauer [Curtis, Abschn. 6.2, S. 287]. Die Änderung der Geschwindigkeit wird als Vektor angegeben [24]:

$$\Delta\vec{v} = \vec{v}_2 - \vec{v}_1\tag{15}$$
^eq-deltav

Dabei ist $\vec{v}_1$ die Geschwindigkeit unmittelbar vor und $\vec{v}_2$ unmittelbar nach dem Manöver. Abb. 4 zeigt links einen Schub in Flugrichtung, der eine Kreisbahn in eine Ellipse überführt, und rechts den allgemeinen Fall eines Schubs schräg zur Flugrichtung.

![[Obsidian Vault/Evaluation/Grafiken & Bilder/impulsives_manoever.png|700]]
*Abb. 4, Impulsives Manöver: Übergang von einer Kreisbahn auf eine Ellipse durch Schub in Flugrichtung (links) und Zusammensetzung der Geschwindigkeiten bei beliebiger Schubrichtung (rechts) – KI-erstellt (Claude)*

Besteht eine Mission aus mehreren Manövern, werden die Beträge der einzelnen Geschwindigkeitsänderungen addiert [24]:

$$\Delta v_{ges} = \sum_i \left|\Delta\vec{v}_i\right|\tag{16}$$
^eq-deltav-ges

Diese Summe wird als **Geschwindigkeitsaufwand** einer Mission bezeichnet und im Folgenden kurz $\Delta v$ genannt. Im Rahmen des impulsiven Manövers ergibt sie sich allein aus den Bahnen vor und nach jedem Manöver und hängt deshalb nicht von Masse oder Triebwerk des Raumfahrzeugs ab.

Welche Energieänderung ein Manöver bewirkt, hängt vom Ort ab, an dem es ausgeführt wird. Erfolgt ein einzelner Schub mit dem Betrag $\Delta v$ in Flugrichtung, wächst die kinetische Energie von $v^2/2$ auf $(v+\Delta v)^2/2$, während $r$ und damit die potentielle Energie beim impulsiven Manöver gleich bleiben. Nach Gl. [[#^eq-energie|(9)]] steigt die Bahnenergie also um

$$\Delta\varepsilon = v\,\Delta v + \frac{\Delta v^2}{2}\tag{17}$$
^eq-oberth

Derselbe Geschwindigkeitszuwachs bringt demnach umso mehr Energie, je schneller das Raumfahrzeug bereits ist. Ein Schub ist deshalb dort am wirksamsten, wo die Geschwindigkeit am größten ist, auf einer Ellipse also an der Periapsis. Dieser Zusammenhang wird als **Oberth-Effekt** bezeichnet [22]. Mit $\Delta\varepsilon$ ändert sich nach Gl. [[#^eq-energie-halbachse|(10)]] die große Halbachse.

Erfolgt der Schub tangential an einer Apside, bleibt dieser Ort eine Apside, und es verschiebt sich wie in Abschnitt 2.1 beschrieben allein die gegenüberliegende, in Abb. 4 die Apoapsis nach außen. Zwei solche tangentialen Schübe, einer an der Ausgangs- und einer an der Zielbahn, bilden die **Hohmann-Transferbahn** zwischen zwei Kreisbahnen [Curtis, Abschn. 6.3, S. 289]. Bei einem Transfer nach außen, etwa von der Erd- zur Marsbahn, kommt sie mit besonders wenig Treibstoff aus [6]. Ihre Berechnung folgt in Kapitel 3.

Wie viel Treibstoff ein bestimmter Geschwindigkeitsaufwand erfordert, beschreibt die Raketengrundgleichung. Ein Triebwerk stößt Masse mit der effektiven Austrittsgeschwindigkeit $v_e$ aus. Sinkt die Masse des Raumfahrzeugs dabei von der Masse $m_0$ vor dem Brennvorgang, die den Treibstoff einschließt, auf die Masse $m_1$ nach dem Brennvorgang, gewinnt es ohne äußere Kräfte in diesem Brennvorgang die Geschwindigkeit [5]

$$\Delta v = v_e \ln\frac{m_0}{m_1}\tag{18}$$
^eq-raketengleichung

Statt $v_e$ wird meist der spezifische Impuls $I_{sp}$ eines Triebwerks angegeben, der über $v_e = I_{sp}\,g_0$ mit der Normfallbeschleunigung $g_0 = 9{,}81\ \mathrm{m/s^2}$ zusammenhängt [5]. Für Triebwerke mit flüssigem Sauerstoff und Wasserstoff liegt er bei etwa $455\ \mathrm{s}$ [Curtis, Abschn. 6.2, Tab. 6.1, S. 288]. Nach Gl. [[#^eq-raketengleichung|(18)]] wächst das Massenverhältnis $m_0/m_1$ exponentiell mit $\Delta v$. Die verbrauchte Treibstoffmasse ist dabei $m_0 - m_1$. Mit einem solchen Triebwerk beträgt sie bei $\Delta v = 4\ \mathrm{km/s}$ etwa das 1,5-Fache der Masse nach dem Brennvorgang, bei $8\ \mathrm{km/s}$ bereits etwa das 5-Fache.

Bei aufeinanderfolgenden Manövern addieren sich die Geschwindigkeitsänderungen, während sich die Massenverhältnisse nach Gl. [[#^eq-raketengleichung|(18)]] multiplizieren. Der Geschwindigkeitsaufwand einer Mission muss deshalb sorgfältig geplant werden, um die mitgeführte Treibstoffmasse zugunsten der Nutzlast gering zu halten [Curtis, Abschn. 6.2, S. 288]. In dieser Arbeit dient $\Delta v$ daher als Maß, an dem die Missionsvarianten verglichen werden.

Dauert ein Brennvorgang länger, bewegt sich das Raumfahrzeug währenddessen merklich weiter, und die Annahme des impulsiven Manövers trifft nicht mehr zu. Die Bahn lässt sich dann nicht mehr geschlossen angeben und wird durch numerische Integration der Bewegungsgleichung mit Schub bestimmt [Curtis, Abschn. 6.1, S. 287].

Alle bisherigen Beziehungen gelten für einen einzelnen Zentralkörper. Auf einer interplanetaren Mission wirken jedoch Sonne und Planeten gleichzeitig auf das Raumfahrzeug, womit sich Abschnitt 2.4 befasst.

## 2.4 N-Körperproblem und numerische Näherung
Wirken $N$ Körper gleichzeitig aufeinander, zieht jeder von ihnen an jedem anderen. Mit den Ortsvektoren $\vec{R}_i$ im Inertialsystem aus Abschnitt 2.1 ergibt sich die Beschleunigung eines Körpers als Summe der Anziehungen aller übrigen [29, Abschn. 2.1]:


$$\ddot{\vec{R}}_i = -G\sum_{j \neq i} m_j\,\frac{\vec{R}_i - \vec{R}_j}{\left|\vec{R}_i - \vec{R}_j\right|^3}, \qquad i = 1, \dots, N\tag{19}$$
^eq-nkoerper

Da die Beschleunigung jedes Körpers von den Orten aller anderen abhängt, die sich zur selben Zeit verändern, lassen sich die Gleichungen nicht einzeln lösen. Beim Zweikörperproblem ließ sich die Bewegung über Erhaltungsgrößen bestimmen, über den Drehimpuls die Bahngleichung [[#^eq-bahngleichung|(5)]] und über die Energie nach Gl. [[#^eq-energie-halbachse|(10)]] die große Halbachse. Gl. [[#^eq-nkoerper|(19)]] besitzt zwar ebenfalls zehn Erhaltungsgrößen, darunter Gesamtenergie und Gesamtdrehimpuls, schon bei drei Körpern im Raum stehen ihnen jedoch 18 Orts- und Geschwindigkeitskomponenten gegenüber, sodass sie für eine geschlossene Lösung nicht ausreichen [29, Abschn. 2.2]. Heinrich Bruns bewies 1887, dass keine weiteren Erhaltungsgrößen existieren, die sich als algebraische Ausdrücke der Orte und Geschwindigkeiten schreiben lassen [31]. Für mehr als zwei Körper gibt es deshalb keine allgemeine geschlossene Lösung [25].

Abb. 5 stellt eine Ellipse aus dem Zweikörperproblem den Bahnen dreier Körper gegenüber, die in der Anordnung des pythagoreischen Dreikörperproblems aus der Ruhe starten [29, Abschn. 3.4]. Während die Ellipse nach jedem Umlauf in sich zurückläuft, wiederholen sich die Bahnen der drei Körper nicht.

![[Obsidian Vault/Evaluation/Grafiken & Bilder/zwei_drei_koerper.png|700]]
*Abb. 5, Ellipsenbahn im Zweikörperproblem (links) und Bahnen dreier Körper mit den Massen 3, 4 und 5, die an den Ecken eines rechtwinkligen Dreiecks mit den Seitenlängen 3, 4 und 5 aus der Ruhe starten (rechts), Anordnung nach [29, Abschn. 3.4] – Rechnung und Grafik KI-erstellt (Claude)*

Gl. [[#^eq-nkoerper|(19)]] wird deshalb auf zwei Wegen näherungsweise behandelt. Der erste baut auf der Zweikörperlösung auf, da in der Umgebung eines Himmelskörpers meist dessen Anziehung überwiegt [25]. Die **Patched-Conics-Näherung** vernachlässigt die schwächeren Körper deshalb abschnittsweise ganz. Innerhalb der **Einflusssphäre** (engl. sphere of influence, SOI) eines Planeten gilt dieser als Zentralkörper, außerhalb die Sonne, und die Kegelschnitte der einzelnen Abschnitte werden an deren Grenze zu einer Gesamtbahn zusammengesetzt [28]. Mit dieser Näherung werden die Transfers und Gravitationsmanöver in den Kapiteln 3 und 4 berechnet.

Der zweite Weg verzichtet auf eine Bahnformel und berechnet die Bewegung schrittweise. Dazu wird die Zeit in kleine Schritte $\Delta t$ zerlegt, in jedem Schritt aus den aktuellen Orten nach Gl. [[#^eq-nkoerper|(19)]] die Beschleunigung bestimmt und daraus Ort und Geschwindigkeit zum nächsten Zeitpunkt fortgeschrieben. Dieses Vorgehen heißt **numerische Integration**. Ihr Ergebnis ist eine Folge einzelner Bahnpunkte, deren Genauigkeit mit kleinerer Schrittweite steigt, wofür mehr Rechenzeit nötig ist [27]. Für Abb. 6 wurde eine Kreisbahn mit dem einfachsten Verfahren berechnet, das in jedem Schritt die Beschleunigung vom Anfang des Schrittes verwendet (Euler-Verfahren). Mit 36 Schritten je Umlauf entfernt sich die berechnete Bahn deutlich von der exakten, mit 360 Schritten bleibt sie weit näher an ihr.

![[Obsidian Vault/Evaluation/Grafiken & Bilder/numerische_integration.png|500]]
*Abb. 6, Numerische Integration einer Kreisbahn über einen Umlauf mit dem Euler-Verfahren bei 36 und 360 Zeitschritten – Rechnung und Grafik KI-erstellt (Claude)*

Die verwendete Simulation geht den zweiten Weg. Da die Masse des Raumfahrzeugs gegenüber der von Sonne und Planeten verschwindend klein ist, wirkt es nicht merklich auf deren Bewegung zurück, und allein seine Bahn wird im Feld aller Himmelskörper numerisch integriert. Die vorausberechnete Bahn in der Anzeige ist das Ergebnis dieser Integration. Welches Verfahren die Simulation dabei verwendet und wie genau sie rechnet, beschreibt Kapitel 5.

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
