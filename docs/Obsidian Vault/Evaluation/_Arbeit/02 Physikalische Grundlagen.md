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

Der Nullpunkt der potentiellen Energie ist frei wählbar, da physikalisch nur Energieänderungen eine Rolle spielen [20]. Legt man ihn in unendliche Entfernung, ist die potentielle Energie überall sonst negativ. Nach Gl. [[#^eq-energie|(10)]] gewinnt ein Körper, der aus großer Entfernung auf den Zentralkörper zufällt, Bewegungsenergie und verliert im selben Maß Lageenergie, die deshalb unter null sinkt. Aus diesem Grund kann auch $\varepsilon$ negativ werden, obwohl die kinetische Energie selbst nie negativ ist.

Multipliziert man Gl. [[#^eq-zweikoerper|(4)]] skalar mit $\dot{\vec{r}}$, lässt sich die linke Seite als zeitliche Ableitung von $v^2/2$ und die rechte als zeitliche Ableitung von $\mu/r$ schreiben. Daraus folgt $\mathrm{d}\varepsilon/\mathrm{d}t = 0$, sodass $\varepsilon$ entlang der gesamten Bahn konstant bleibt [20]. Mit wachsendem Abstand nimmt die potentielle Energie $-\mu/r$ zu, sodass die kinetische Energie im selben Maß abnimmt. Das Raumfahrzeug ist deshalb an der Periapsis am schnellsten und an der Apoapsis am langsamsten.

Da $\varepsilon$ überall denselben Wert hat, genügt es, sie an einem einzigen Punkt der Bahn auszuwerten. An der Periapsis steht die Geschwindigkeit senkrecht auf $\vec{r}$, sodass sich Gl. [[#^eq-drehimpuls|(5)]] zu $h = r_p v_p$ vereinfacht. Aus Gl. [[#^eq-bahngleichung|(6)]] folgt für $\nu = 0°$ zudem $h^2 = \mu\,r_p(1+e)$. Eingesetzt in Gl. [[#^eq-energie|(10)]] ergibt sich

$$\varepsilon = \frac{h^2}{2r_p^2} - \frac{\mu}{r_p} = \frac{\mu\,(e-1)}{2r_p}$$

Aus Gl. [[#^eq-bahngleichung|(6)]] und [[#^eq-halbachse|(8)]] folgt $r_p = a(1-e)$, womit sich die Exzentrizität herauskürzt [10]:

$$\varepsilon = -\frac{\mu}{2a}\tag{11}$$
^eq-energie-halbachse

Die Energie einer Bahn hängt damit allein von ihrer großen Halbachse ab und ist von der Exzentrizität unabhängig [10]. Eine Kreisbahn und eine stark gestreckte Ellipse mit gleichem $a$ besitzen also dieselbe Energie, so wie sie nach Gl. [[#^eq-umlaufzeit|(9)]] auch dieselbe Umlaufzeit haben. Umgekehrt lässt sich $a$ nur verändern, indem die Energie der Bahn verändert wird. Jede Vergrößerung oder Verkleinerung einer Bahn ist somit ein Energiewechsel, und auf diesem Zusammenhang bauen die Transferrechnungen in Kapitel 3 auf.

Die Energiegleichung [[#^eq-energie|(10)]] wird auch als Vis-viva-Gleichung bezeichnet [20]. Meist ist damit die Form gemeint, die sich ergibt, wenn man sie mit Gl. [[#^eq-energie-halbachse|(11)]] gleichsetzt und nach $v$ auflöst. Sie liefert die Geschwindigkeit an jedem Punkt einer Bahn aus dem Abstand zum Zentralkörper und der großen Halbachse:

$$v^2 = \mu\left(\frac{2}{r} - \frac{1}{a}\right)\tag{12}$$
^eq-visviva

Mit Gl. [[#^eq-visviva|(12)]] werden in Kapitel 3 die Geschwindigkeiten vor und nach einem Manöver berechnet, wobei zwei Sonderfälle wiederholt gebraucht werden. Auf einer Kreisbahn gilt $r = a$, und Gl. [[#^eq-visviva|(12)]] vereinfacht sich zur **Kreisbahngeschwindigkeit** [12]:

$$v_k = \sqrt{\frac{\mu}{r}}\tag{13}$$
^eq-kreisbahn

Lässt man $a$ gegen unendlich gehen, strebt $\varepsilon$ gegen null, und das Raumfahrzeug erreicht gerade noch unendliche Entfernung. Die dafür nötige **Fluchtgeschwindigkeit** liegt um den Faktor $\sqrt{2}$ über der Kreisbahngeschwindigkeit am selben Ort [13]:

$$v_{esc} = \sqrt{\frac{2\mu}{r}} = \sqrt{2}\,v_k\tag{14}$$
^eq-flucht

![[Obsidian Vault/Evaluation/Grafiken & Bilder/energiediagramm.png|450]]
*Abb. 5, Energiediagramm mit der potentiellen Energie $-\mu/r$ und der Bahnenergie $\varepsilon$ für die drei Bahnformen. Der senkrechte Abstand zur Kurve ist die kinetische Energie. Der Umkehrpunkt gibt nur eine Obergrenze des Abstands an, da die Bewegung quer zum Radius vernachlässigt ist – KI-erstellt (Claude)*

In Abschnitt 2.1 wurden die Kegelschnitte über die Exzentrizität unterschieden. Dieselbe Einteilung ergibt sich über das Vorzeichen von $\varepsilon$ (Abb. 5), das zusätzlich zeigt, ob der Körper dem Zentralkörper entkommen kann. In unendlicher Entfernung wäre die potentielle Energie null, und da die kinetische Energie nicht negativ sein kann, hätte ein Körper, der dort ankommt, stets $\varepsilon \geq 0$. In Abb. 5 entspricht die kinetische Energie an jedem Ort dem senkrechten Abstand zwischen der waagerechten Linie für $\varepsilon$ und der Kurve $-\mu/r$, der nie negativ werden darf. Ist $\varepsilon$ negativ, reicht die Bewegungsenergie deshalb an keinem Punkt der Bahn aus, um das Unendliche zu erreichen. Der Körper bleibt gebunden und bewegt sich auf einer Ellipse oder einem Kreis, was mit Gl. [[#^eq-energie-halbachse|(11)]] übereinstimmt, die für jedes endliche $a$ einen negativen Wert liefert. Für $\varepsilon = 0$ erreicht er, wie bei der Fluchtgeschwindigkeit, gerade noch unendliche Entfernung und kommt dort zur Ruhe, die Bahn ist eine Parabel [13]. Für $\varepsilon > 0$ bleibt ihm auch im Unendlichen Bewegungsenergie übrig, und die Bahn verläuft als Hyperbel [21]. Zählt man $a$ bei der Hyperbel als positive Länge, kehrt sich in Gl. [[#^eq-energie-halbachse|(11)]] und [[#^eq-visviva|(12)]] das Vorzeichen des Terms mit $a$ um, sodass dort $\varepsilon = \mu/(2a)$ gilt [21]. Die Geschwindigkeit, die im Unendlichen übrig bleibt, heißt **hyperbolische Exzessgeschwindigkeit** $v_\infty$. Da dort der Term $\mu/r$ verschwindet, folgt aus Gl. [[#^eq-energie|(10)]] [21]:

$$v_\infty^2 = 2\,\varepsilon\tag{15}$$
^eq-exzess

Bei einem Vorbeiflug ohne Schub bleibt der Betrag von $v_\infty$ relativ zum Planeten erhalten, lediglich ihre Richtung wird gedreht [23]. Damit wird $v_\infty$ zur entscheidenden Größe für die Gravitationsmanöver in Kapitel 4.

Die Bahn eines Raumfahrzeugs lässt sich folglich nur verändern, indem sich seine Energie ändert. Nach Gl. [[#^eq-energie|(10)]] ist das an einem festen Ort allein über die Geschwindigkeit möglich. Wie eine solche Geschwindigkeitsänderung beschrieben und als Aufwand gemessen wird, behandelt Abschnitt 2.3.

## 2.3 Geschwindigkeitsaufwand als Maß
Um die Geschwindigkeit eines Raumfahrzeugs zu ändern, werden seine Triebwerke gezündet. Dauert der Brennvorgang nur kurz im Vergleich zur Zeit, in der das Raumfahrzeug antriebslos fliegt, spricht man von einem **impulsiven Manöver**. Dabei wird angenommen, dass sich der Geschwindigkeitsvektor in Betrag und Richtung schlagartig ändert, während der Ort des Raumfahrzeugs unverändert bleibt [24]. Diese Idealisierung erspart es, die Bewegungsgleichung mit dem Schub der Triebwerke zu lösen, und ist zulässig, solange sich das Raumfahrzeug während des Brennvorgangs kaum weiterbewegt, wie bei Triebwerken mit hohem Schub und kurzer Brenndauer [Curtis, Abschn. 6.2, S. 287]. Die Änderung der Geschwindigkeit wird als Vektor angegeben [24]:

$$\Delta\vec{v} = \vec{v}_2 - \vec{v}_1\tag{16}$$
^eq-deltav

Dabei ist $\vec{v}_1$ die Geschwindigkeit unmittelbar vor und $\vec{v}_2$ unmittelbar nach dem Manöver. Abb. 6 zeigt links einen Schub in Flugrichtung, der eine Kreisbahn in eine Ellipse überführt, und rechts den allgemeinen Fall eines Schubs schräg zur Flugrichtung.

![[Obsidian Vault/Evaluation/Grafiken & Bilder/impulsives_manoever.png|700]]
*Abb. 6, Impulsives Manöver: Übergang von einer Kreisbahn auf eine Ellipse durch Schub in Flugrichtung (links) und Zusammensetzung der Geschwindigkeiten bei beliebiger Schubrichtung (rechts) – KI-erstellt (Claude)*

Besteht eine Mission aus mehreren Manövern, werden die Beträge der einzelnen Geschwindigkeitsänderungen addiert [24]:

$$\Delta v_{ges} = \sum_i \left|\Delta\vec{v}_i\right|\tag{17}$$
^eq-deltav-ges

Diese Summe wird als **Geschwindigkeitsaufwand** $\Delta v$ einer Mission bezeichnet. Sie ergibt sich allein aus den Bahnen vor und nach jedem Manöver und hängt deshalb nicht von Masse oder Triebwerk des Raumfahrzeugs ab. Ein Beispiel für eine Überführung aus zwei Manövern ist die **Hohmann-Transferbahn**, eine Ellipse, deren Periapsis auf der Ausgangsbahn und deren Apoapsis auf der Zielbahn liegt und die mit besonders wenig Treibstoff auskommt [6]. Ihre Berechnung folgt in Kapitel 3.

Welche Energieänderung ein Manöver bewirkt, hängt vom Ort ab, an dem es ausgeführt wird. Erfolgt der Schub in Flugrichtung, wächst die kinetische Energie von $v^2/2$ auf $(v+\Delta v)^2/2$, während $r$ und damit die potentielle Energie beim impulsiven Manöver gleich bleiben. Nach Gl. [[#^eq-energie|(10)]] steigt die Bahnenergie also um

$$\Delta\varepsilon = v\,\Delta v + \frac{\Delta v^2}{2}\tag{18}$$
^eq-oberth

Derselbe Geschwindigkeitszuwachs bringt demnach umso mehr Energie, je schneller das Raumfahrzeug bereits ist. Ein Schub ist deshalb dort am wirksamsten, wo die Geschwindigkeit am größten ist, auf einer Ellipse also an der Periapsis. Dieser Zusammenhang wird als **Oberth-Effekt** bezeichnet [22]. Mit $\Delta\varepsilon$ ändert sich nach Gl. [[#^eq-energie-halbachse|(11)]] die große Halbachse. Erfolgt der Schub tangential an einer Apside, bleibt der Ort des Schubs eine Apside, und nach Gl. [[#^eq-halbachse|(8)]] verschiebt sich allein die gegenüberliegende. In Abb. 6 wird der Ort des Schubs so zur neuen Periapsis, während die Apoapsis nach außen rückt.

Wie viel Treibstoff ein bestimmter Geschwindigkeitsaufwand erfordert, beschreibt die Raketengrundgleichung. Ein Triebwerk stößt Masse mit der effektiven Austrittsgeschwindigkeit $v_e$ aus. Sinkt die Masse des Raumfahrzeugs dabei von der Masse $m_0$ vor dem Brennvorgang, die den Treibstoff einschließt, auf die Masse $m_1$ nach dem Brennvorgang, gewinnt es ohne äußere Kräfte die Geschwindigkeit [5]

$$\Delta v = v_e \ln\frac{m_0}{m_1}\tag{19}$$
^eq-raketengleichung

Statt $v_e$ wird meist der spezifische Impuls $I_{sp}$ eines Triebwerks angegeben, der über $v_e = I_{sp}\,g_0$ mit der Normfallbeschleunigung $g_0 = 9{,}81\ \mathrm{m/s^2}$ zusammenhängt [5]. Für Triebwerke mit flüssigem Sauerstoff und Wasserstoff liegt er bei etwa $455\ \mathrm{s}$ [Curtis, Abschn. 6.2, S. 288]. Nach Gl. [[#^eq-raketengleichung|(19)]] wächst das Massenverhältnis $m_0/m_1$ exponentiell mit $\Delta v$ (Abb. 7). Mit einem solchen Triebwerk beträgt die benötigte Treibstoffmasse bei $\Delta v = 4\ \mathrm{km/s}$ etwa das 1,5-Fache der Masse nach dem Brennvorgang, bei $8\ \mathrm{km/s}$ bereits etwa das 5-Fache. Die verbrauchte Treibstoffmasse ist dabei jeweils $m_0 - m_1$.

![[Obsidian Vault/Evaluation/Grafiken & Bilder/massenverhaeltnis_raketengleichung.png|462]]
*Abb. 7, Massenverhältnis nach der Raketengrundgleichung für einen spezifischen Impuls von 455 s – KI-erstellt (Claude)*

Bei aufeinanderfolgenden Manövern addieren sich die Geschwindigkeitsänderungen, während sich die Massenverhältnisse nach Gl. [[#^eq-raketengleichung|(19)]] multiplizieren. Der Geschwindigkeitsaufwand einer Mission muss deshalb sorgfältig geplant werden, um die mitgeführte Treibstoffmasse zugunsten der Nutzlast gering zu halten [Curtis, Abschn. 6.2, S. 288]. In dieser Arbeit dient $\Delta v$ daher als Maß, an dem die Missionsvarianten verglichen werden.

Dauert ein Brennvorgang länger, bewegt sich das Raumfahrzeug währenddessen merklich weiter, und die Annahme des impulsiven Manövers trifft nicht mehr zu. Die Bahn lässt sich dann nicht mehr geschlossen angeben und wird durch numerische Integration der Bewegungsgleichung mit Schub bestimmt [Curtis, Abschn. 6.1, S. 287]. Auf diese Weise berechnet auch die verwendete Simulation ihre Brennvorgänge. Manöver werden dort über Manöverknoten auf der vorausberechneten Bahn geplant, von denen jeder eine Geschwindigkeitsänderung in Flugrichtung und eine senkrecht dazu trägt. Die Anzeige gibt daraus den Betrag $\Delta v$ und die Brenndauer an. Das Triebwerk zündet um die halbe Brenndauer vor dem Knoten, sodass der Knoten in der Mitte des Brennvorgangs liegt und dem Ort des impulsiven Manövers entspricht. Masse und Treibstoff bildet die Simulation nicht ab. Der Schub wird direkt als Beschleunigung vorgegeben, und der Aufwand eines Manövers wird unmittelbar in $\Delta v$ gemessen.

%%
Screenshot (Abb. 8) folgt: Manöverknoten an der Periapsis einer Erdbahn, rein prograde Δv, Vorschaulinie mit angehobener Apoapsis, MANEUVER-Block mit DV und Brenndauer lesbar, ohne Ausschnitt. Bildunterschrift und Eintrag unter Bildquellen danach ergänzen.
%%

Alle bisherigen Beziehungen gelten für einen einzelnen Zentralkörper. Auf einer interplanetaren Mission wirken jedoch Sonne und Planeten gleichzeitig auf das Raumfahrzeug, womit sich Abschnitt 2.4 befasst.
