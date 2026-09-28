# 2. Physikalische Grundlagen
Die Himmelsmechanik beschreibt die Bewegung von Körpern unter gegenseitiger Gravitation; die Astrodynamik wendet diese Beschreibung auf Raumfahrzeuge an und ergänzt sie um die gezielte Bahnänderung durch Antrieb. Beide gehen auf Keplers Beschreibung der Planetenbahnen und Newtons Gravitationsgesetz zurück [Curtis, Kap. 2]. Dieses Kapitel führt die Begriffe ein, mit denen die Transferrechnungen in Kapitel 3 und die Gravitationsmanöver in Kapitel 4 arbeiten, und benennt an welcher Stelle die verwendete Simulation vereinfacht.
## 2.1 Gravitation und das Zweikörperproblem
Isaac Newton (1643–1727) legte 1687 in seiner *Philosophiae Naturalis Principia Mathematica* die Grundlagen der klassischen Mechanik fest. Für die Bahnmechanik sind zwei seiner drei Bewegungsgesetze entscheidend [1 S.2]. Nach dem zweiten Gesetz ist die Impulsänderung eines Körpers der einwirkenden Kraft proportional, was für konstante Masse $\vec{F} = m\vec{a}$ ergibt. Nach dem dritten Gesetz wirken zwei Körper stets mit gleich großen, entgegengesetzt gerichteten Kräften aufeinander.

Außerdem formulierte Newton das Gravitationsgesetz, das die gegenseitige Anziehung zweier Massen beschreibt [8]:

$$F = G\,\frac{m_1 m_2}{r^2}\tag{1}$$
^eq-gravitation

$F$ ... Betrag der Gravitationskraft, $G$ ... Gravitationskonstante, $m_1, m_2$ ... Massen der beiden Körper, $r$ ... Abstand zwischen den Körpern

Nach Gl. [[#^eq-gravitation|(1)]] wächst die Kraft mit dem Produkt der Massen und nimmt mit dem Quadrat des Abstands ab [8]. Die Gravitationskonstante $G$ beträgt rund $6{,}674 \cdot 10^{-11}\ \mathrm{m^3/(kg\,s^2)}$ und legt die Stärke der Gravitation fest [8].

Bezeichnet man die Masse des Zentralkörpers mit $M$ und die des umlaufenden Körpers mit $m$, wirkt auf diesen die Kraft $F = m \cdot GM/r^2$. Da nach dem zweiten Newtonschen Gesetz zugleich $F = m\,a$ gilt, lassen sich beide Ausdrücke gleichsetzen [8] [Curtis S.22-23]. Die Masse $m$ kürzt sich heraus, und es bleibt die **Gravitationsbeschleunigung** [Curtis S.23]:

$$g = \frac{G M}{r^2}\tag{2}$$
^eq-gravibeschleunigung

Gl. [[#^eq-gravitation|(1)]] gilt streng für Punktmassen. Für kugelsymmetrische Körper, wie es die Erde näherungsweise ist, gilt sie außerhalb des Körpers ebenso, da ihre Gravitation dort wirkt, als säße die gesamte Masse im Mittelpunkt [18]. Die Gravitationsbeschleunigung zeigt deshalb stets zum Erdmittelpunkt, und ihr Betrag hängt nach Gl. [[#^eq-gravibeschleunigung|(2)]] allein vom Abstand ab [Curtis S.23]. Was das für die Bahn eines Körpers bedeutet, zeigt Newtons Gedankenexperiment: Eine Patrone wird in $1000\ \mathrm{m}$ Höhe waagerecht abgefeuert, Luftwiderstand wird vernachlässigt [1 S.4].

![[Obsidian Vault/Evaluation/Grafiken & Bilder/newton_kanonenkugel_FHD.png|450]]
*Abb. 1, Waagerechter Abschuss mit steigender Geschwindigkeit nach Newtons Gedankenexperiment – KI-erstellt (Claude)*

Da die Höhe in allen drei Fällen von Abb. 1 gleich ist, wirkt dieselbe Kraft, und allein die Geschwindigkeit entscheidet über die Bahn. Die Patrone befindet sich ab dem Abschuss im freien Fall und schlägt umso weiter entfernt auf, je schneller sie ist, da sich die Erdoberfläche unter ihr wegkrümmt [1 S.4]. Fällt sie genauso stark, wie sich die Erde krümmt, trifft sie nie auf und hat einen Orbit erreicht [1 S.4]. Auf einer Kreisbahn steht die Geschwindigkeit überall senkrecht auf der Kraft, auf jeder anderen Bahn nur an den Punkten mit dem kleinsten und dem größten Abstand zum Zentralkörper. Diese Punkte heißen **Periapsis** und **Apoapsis**, zusammen **Apsiden** [1 S.4]. Nach dem ersten Keplerschen Gesetz ist die Bahn eines Planeten eine Ellipse, in deren einem Brennpunkt die Sonne steht [1 S.3].

Keplers Beschreibung und Gl. [[#^eq-gravibeschleunigung|(2)]] behandeln den Zentralkörper als ruhend. Nach dem dritten Newtonschen Gesetz wirkt die Gravitationskraft jedoch mit gleichem Betrag und entgegengesetzter Richtung auf beide Körper, sodass sich auch der Zentralkörper bewegt. Um beide Bewegungen zu erfassen, werden die Körper durch ihre Ortsvektoren $\vec{R}_1$ und $\vec{R}_2$ in einem Inertialsystem beschrieben, da das zweite Newtonsche Gesetz nur dort gilt. Der relative Ortsvektor $\vec{r} = \vec{R}_2 - \vec{R}_1$ zeigt von Körper 1 zu Körper 2 und hat die Länge $r$.

Gl. [[#^eq-gravitation|(1)]] liefert nur den Betrag der Kraft. Teilt man $\vec{r}$ durch seine Länge $r$, erhält man ihre Richtung, und die Kraft auf Körper 1 lautet in Vektorschreibweise [Curtis 2.2, Gl. 2.9, S.57]:

$$\vec{F}_1 = \frac{G\,m_1 m_2}{r^2}\cdot\frac{\vec{r}}{r} = \frac{G\,m_1 m_2}{r^3}\,\vec{r}\tag{3}$$
^eq-kraftvektor

Die dritte Potenz setzt sich aus dem $r^2$ des Gravitationsgesetzes und dem $r$ der Normierung zusammen, der Betrag bleibt $\frac{G\,m_1 m_2}{r^2}$. Wendet man Gl. [[#^eq-kraftvektor|(3)]] und die Gegenkraft $-\vec{F}_1$ nach dem zweiten Newtonschen Gesetz auf beide Körper an und bildet die Differenz ihrer Beschleunigungen, folgt die Beschleunigung von Körper 2 relativ zu Körper 1 [Curtis 2.3, Gl. 2.20-2.22, S.63] [14]:

$$\ddot{\vec{r}} = -G\,(m_1 + m_2)\,\frac{\vec{r}}{r^3} = -\frac{\mu}{r^3}\,\vec{r}, \qquad \mu = G\,(m_1 + m_2)\tag{4}$$
^eq-zweikoerper

Das Minuszeichen richtet die Beschleunigung auf Körper 1 aus. Die Massen treten nur noch als Summe auf und bilden zusammen mit $G$ den **Gravitationsparameter** $\mu$. Da ein Raumfahrzeug gegenüber einem Himmelskörper eine verschwindend kleine Masse besitzt, gilt $\mu \approx G\,m_1$, und $\mu$ wird zu einer Eigenschaft des Zentralkörpers allein [14]. Für die Erde beträgt $\mu = 3{,}98600 \cdot 10^{14}\ \mathrm{m^3/s^2}$ [17]. Dieser Wert ist genauer bekannt als $G$ und die Erdmasse einzeln, weil er sich direkt aus beobachteten Bahnen bestimmen lässt [14]. Gl. [[#^eq-zweikoerper|(4)]] ist die Bewegungsgleichung des Zweikörperproblems, und die Gravitation geht ab hier nur noch über das $\mu$ des jeweiligen Zentralkörpers ein.

Kepler leitete die Ellipsenform aus Beobachtungen ab. Dass sie aus Gl. [[#^eq-zweikoerper|(4)]] folgt, zeigt sich über den spezifischen Drehimpuls:

$$\vec{h} = \vec{r} \times \dot{\vec{r}}\tag{5}$$
^eq-drehimpuls

Da die Gravitation stets entlang von $\vec{r}$ wirkt, kann sie $\vec{h}$ nicht verändern, sodass Richtung und Betrag $h = |\vec{h}|$ konstant bleiben [15]. Der Vektor $\vec{h}$ steht senkrecht auf $\vec{r}$ und $\dot{\vec{r}}$, deshalb bleibt die gesamte Bahn in einer festen Ebene [15]. Für das Zweikörperproblem ist eine Beschreibung in der Ebene damit vollständig. Abb. 2 zeigt die Größen, mit denen eine Ellipsenbahn in dieser Ebene beschrieben wird.

![[Obsidian Vault/Evaluation/Grafiken & Bilder/elliptical-orbit-definitions.svg|450]]
*Abb. 2, Geometrie und Bezeichnungen einer Ellipsenbahn – [10]*

In einem der Brennpunkte, in Abb. 2 dem Punkt $F$, steht der Zentralkörper $m_1$, und $m_2$ umläuft ihn auf der Ellipse [11]. Der Abstand zwischen einer Apside und dem Mittelpunkt $C$ ist die **große Halbachse** $a$ [10]. Die **wahre Anomalie** $\nu$ ist der Winkel am Brennpunkt $F$ zwischen der Richtung zur Periapsis und der momentanen Position von $m_2$, in Abb. 2 für den Punkt $B$ als $\beta$ eingezeichnet [10]. Mit diesen Größen lässt sich die Bahngleichung angeben. Sie folgt durch Integration aus Gl. [[#^eq-zweikoerper|(4)]], wenn man die Erhaltung von $\vec{h}$ nutzt [16]:

$$r = \frac{h^2}{\mu}\,\frac{1}{1 + e\cos\nu}\tag{6}$$
^eq-bahngleichung

Gl. [[#^eq-bahngleichung|(6)]] gibt den Abstand $r$ zum Zentralkörper in Abhängigkeit von der wahren Anomalie $\nu$ an. Dabei bestimmen $h$ und $\mu$ die Größe der Bahn, die **Exzentrizität** $e$ ihre Form [16]. An der Periapsis ist $\nu = 0°$ und $r$ am kleinsten, an der Apoapsis ist $\nu = 180°$ und $r$ am größten. Das ergibt $r_p = \frac{h^2}{\mu}\frac{1}{1+e}$ und $r_a = \frac{h^2}{\mu}\frac{1}{1-e}$. Setzt man $h^2/\mu = r_p(1+e) = r_a(1-e)$ gleich, fällt $h$ heraus, und die Exzentrizität folgt allein aus den beiden Apsidenabständen [10]:

$$e = \frac{r_a - r_p}{r_a + r_p}\tag{7}$$
^eq-exzentrizitaet

![[Obsidian Vault/Evaluation/Grafiken & Bilder/exzentrizitaet_erdnah_8k.png]]
*Abb. 3, Exzentrizität bei verschiedenen Abstandsverhältnissen – KI-erstellt (Claude)*

Abb. 3 zeigt, wie $e$ die Form der Bahn bestimmt. Für $e = 0$ ist der Nenner in Gl. [[#^eq-bahngleichung|(6)]] konstant, $r$ hängt nicht von $\nu$ ab, und die Bahn ist ein Kreis [12]. Für $0 < e < 1$ ist die Bahn eine Ellipse, deren Apoapsis weiter vom Brennpunkt entfernt liegt als die Periapsis [10]. Für $e = 1$ wird der Nenner bei $\nu = 180°$ null, sodass $r$ ohne Grenze wächst. Die Bahn ist dann eine Parabel ohne Apoapsis, auf der sich der Körper unbegrenzt entfernt und nicht zurückkehrt [13]. Für $e > 1$ ergibt sich eine Hyperbel [16].

Bei einer Ellipse entspricht die Strecke von der Periapsis zur Apoapsis entlang der Apsidenlinie der doppelten großen Halbachse [10]:

$$2a = r_p + r_a\tag{8}$$
^eq-halbachse

Die große Halbachse bestimmt außerdem die Umlaufzeit. Nach dem dritten Keplerschen Gesetz hängt sie allein von $a$ ab und ist von der Exzentrizität unabhängig [10]:

$$T = \frac{2\pi}{\sqrt{\mu}}\,a^{3/2}\tag{9}$$
^eq-umlaufzeit

Vollständig beschrieben ist eine Bahn im Raum durch sechs klassische Bahnelemente (Abb. 4): große Halbachse $a$, Exzentrizität $e$, Inklination $i$, Rektaszension des aufsteigenden Knotens $\Omega$, Argument der Periapsis $\omega$ und Zeitpunkt des Periapsisdurchgangs $t_p$ [19]. Liegt die Bahn in der Bezugsebene, ist die Inklination null und der aufsteigende Knoten nicht definiert [19]. In einer ebenen Beschreibung genügen deshalb vier Elemente. Das begründet die Beschränkung der Simulation auf zwei Dimensionen und schließt Manöver zur Änderung der Bahnebene aus dieser Untersuchung aus.

![[Obsidian Vault/Evaluation/Grafiken & Bilder/bahnelemente_3d.png|450]]
*Abb. 4, Bahnelemente einer Bahn im Raum mit Knotenlinie, Inklination $i$, Rektaszension des aufsteigenden Knotens $\Omega$, Argument der Periapsis $\omega$ und wahrer Anomalie $\nu$ – KI-erstellt (Claude)*

Verändern lässt sich eine Bahn gezielt, indem ein Raumfahrzeug an einer Apside seine Energie ändert. Mehr Energie an der Apoapsis hebt die Periapsis an, weniger Energie senkt sie [1 S.4]. An der Periapsis wirkt eine Energieänderung entsprechend auf die Apoapsis [1 S.4]. Wie die Energie einer Bahn mit ihrer großen Halbachse zusammenhängt, zeigt Abschnitt 2.2.

## 2.2 Bahnenergie und große Halbachse
Neben dem Drehimpuls bleibt bei der Relativbewegung eine zweite Größe erhalten, die Energie. Da die Masse des Raumfahrzeugs für seine Bahn keine Rolle spielt, betrachtet man die Energie pro Kilogramm, die **spezifische Bahnenergie** $\varepsilon$. Sie setzt sich aus der kinetischen Energie $v^2/2$ mit $v = |\dot{\vec{r}}|$ und der potentiellen Energie $-\mu/r$ zusammen, deren Nullpunkt in unendlicher Entfernung liegt [20]:

$$\varepsilon = \frac{v^2}{2} - \frac{\mu}{r}\tag{10}$$
^eq-energie

Multipliziert man Gl. [[#^eq-zweikoerper|(4)]] skalar mit $\dot{\vec{r}}$, heben sich die Beiträge gegenseitig auf, sodass $\varepsilon$ entlang der gesamten Bahn konstant bleibt [20]. Wegen des negativen Vorzeichens der potentiellen Energie muss Energie zugeführt werden, damit sich ein Körper vom Zentralkörper entfernt. Auf einer Ellipse wandelt sich deshalb fortlaufend Lageenergie in Bewegungsenergie um und wieder zurück. Das Raumfahrzeug ist nahe der Periapsis schnell und nahe der Apoapsis langsam, während die Summe beider Anteile gleich bleibt.

Da $\varepsilon$ überall denselben Wert hat, genügt es, sie an einem einzigen Punkt der Bahn auszuwerten. An der Periapsis steht die Geschwindigkeit senkrecht auf $\vec{r}$, sodass sich Gl. [[#^eq-drehimpuls|(5)]] zu $h = r_p v_p$ vereinfacht. Aus Gl. [[#^eq-bahngleichung|(6)]] folgt für $\nu = 0°$ zudem $h^2 = \mu\,r_p(1+e)$. Eingesetzt in Gl. [[#^eq-energie|(10)]] ergibt sich

$$\varepsilon = \frac{h^2}{2r_p^2} - \frac{\mu}{r_p} = \frac{\mu\,(e-1)}{2r_p}$$

Aus Gl. [[#^eq-bahngleichung|(6)]] und [[#^eq-halbachse|(8)]] folgt $r_p = a(1-e)$, womit sich die Exzentrizität herauskürzt [10]:

$$\varepsilon = -\frac{\mu}{2a}\tag{11}$$
^eq-energie-halbachse

Die Energie einer Bahn hängt damit allein von ihrer großen Halbachse ab und ist von der Exzentrizität unabhängig [10]. Eine Kreisbahn und eine stark gestreckte Ellipse mit gleichem $a$ besitzen also dieselbe Energie, so wie sie nach Gl. [[#^eq-umlaufzeit|(9)]] auch dieselbe Umlaufzeit haben. Umgekehrt lässt sich $a$ nur verändern, indem die Energie der Bahn verändert wird. Jede Vergrößerung oder Verkleinerung einer Bahn ist somit ein Energiewechsel, und auf diesem Zusammenhang bauen die Transferrechnungen in Kapitel 3 auf.

Gl. [[#^eq-energie|(10)]] wird auch als Vis-viva-Gleichung bezeichnet [20]. Setzt man sie mit Gl. [[#^eq-energie-halbachse|(11)]] gleich und löst nach $v$ auf, erhält man die Geschwindigkeit an jedem Punkt einer Bahn aus dem Abstand zum Zentralkörper und der großen Halbachse:

$$v^2 = \mu\left(\frac{2}{r} - \frac{1}{a}\right)\tag{12}$$
^eq-visviva

Mit Gl. [[#^eq-visviva|(12)]] werden in Kapitel 3 die Geschwindigkeiten vor und nach einem Manöver berechnet, wobei zwei Sonderfälle wiederholt gebraucht werden. Auf einer Kreisbahn gilt $r = a$, und Gl. [[#^eq-visviva|(12)]] vereinfacht sich zur **Kreisbahngeschwindigkeit** [12]:

$$v_k = \sqrt{\frac{\mu}{r}}\tag{13}$$
^eq-kreisbahn

Lässt man $a$ gegen unendlich gehen, strebt $\varepsilon$ gegen null, und das Raumfahrzeug erreicht gerade noch unendliche Entfernung. Die dafür nötige **Fluchtgeschwindigkeit** liegt um den Faktor $\sqrt{2}$ über der Kreisbahngeschwindigkeit am selben Ort [13]:

$$v_{esc} = \sqrt{\frac{2\mu}{r}} = \sqrt{2}\,v_k\tag{14}$$
^eq-flucht

Über das Vorzeichen von $\varepsilon$ lassen sich auch die Kegelschnitte aus Abschnitt 2.1 einordnen. Für $\varepsilon < 0$ ist der Körper gebunden und bewegt sich auf einer Ellipse oder einem Kreis, $\varepsilon = 0$ entspricht der Parabel [13], und für $\varepsilon > 0$ verläuft die Bahn als Hyperbel [21]. Auf einer Hyperbel besitzt der Körper auch in unendlicher Entfernung noch eine Restgeschwindigkeit, die **hyperbolische Exzessgeschwindigkeit** $v_\infty$. Da dort der Term $\mu/r$ verschwindet, folgt aus Gl. [[#^eq-energie|(10)]] [21]:

$$v_\infty^2 = 2\,\varepsilon\tag{15}$$
^eq-exzess

Für die Gravitationsmanöver in Kapitel 4 ist $v_\infty$ die entscheidende Größe, da ein Vorbeiflug an einem Planeten ihren Betrag relativ zum Planeten unverändert lässt und lediglich ihre Richtung dreht.

Offen bleibt, wie sich die Energie einer Bahn verändern lässt. Nach Gl. [[#^eq-energie|(10)]] ist das an einem festen Ort nur über die Geschwindigkeit möglich. Erfolgt ein Schub in Flugrichtung so kurz, dass sich $r$ währenddessen praktisch nicht ändert, wächst die kinetische Energie von $v^2/2$ auf $(v+\Delta v)^2/2$, die Bahnenergie also um

$$\Delta\varepsilon = v\,\Delta v + \frac{\Delta v^2}{2}\tag{16}$$
^eq-oberth

Derselbe Geschwindigkeitszuwachs $\Delta v$ bringt demnach umso mehr Energie, je schneller das Raumfahrzeug bereits ist. Ein Schub ist deshalb dort am wirksamsten, wo die Geschwindigkeit am größten ist, auf einer Ellipse also an der Periapsis. Dieser Zusammenhang wird als **Oberth-Effekt** bezeichnet [22]. Erfolgt der Schub tangential an einer Apside, bleibt der Brennort eine Apside, und nach Gl. [[#^eq-halbachse|(8)]] verschiebt sich allein die gegenüberliegende, wie am Ende von Abschnitt 2.1 beschrieben. Wie groß der dafür nötige Geschwindigkeitsaufwand ist und wie er mit dem Treibstoffverbrauch zusammenhängt, behandelt Abschnitt 2.3.
