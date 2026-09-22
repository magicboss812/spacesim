# 2. Physikalische Grundlagen
Die Himmelsmechanik beschreibt die Bewegung von Körpern unter gegenseitiger Gravitation; die Astrodynamik wendet diese Beschreibung auf Raumfahrzeuge an und ergänzt sie um die gezielte Bahnänderung durch Antrieb. Beide gehen auf Keplers Beschreibung der Planetenbahnen und Newtons Gravitationsgesetz zurück [Curtis, Kap. 2]. Dieses Kapitel führt die Begriffe ein, mit denen die Transferrechnungen in Kapitel 3 und die Gravitationsmanöver in Kapitel 4 arbeiten, und benennt an welcher Stelle die verwendete Simulation vereinfacht. Es werden oft Vektoren in den Formeln erwähnt, da die Geschwindigkeit, Beschleunigung und vereinheitlicht Kraft in einem drei dimensionalen System jeweils in verschiedene Richtungen und Magnituden wirken.
## 2.1 Gravitation und das Zweikörperproblem
Isaac Newton (1643-1727) legt in seiner "_Mathematical Principles of Natural Philosophy_", veröffentlicht in 1685, die Grundlagen der klassischen Mechanik fest. Sie umfasst den Einbezug der Keplerschen Gesetzen der Bahnbewegung und vereinheitlicht so das Gesetz der Schwerkraft, das Newtonsches Gravitationsgesetz [9]. 

$$\vec{F} = G \frac{m_1 m_2}{r^2}\tag{1}$$
^eq-gravitation
$\vec{F}$...Kraft mit Vektorrichtung, $G$...Newtonsche Gravitationskonstante, $m_{1/2}$...Masse der Objekte, $r$...Abstand zwischen beider Objekte

Wie in Gl. [[#^eq-gravitation|(1)]] abgebildet, beschreibt das Gesetz nach Newton, wie sich zwei schwere Objekte in einem geschlossenen System verhalten [8]. $m_1$ und $m_2$ sind jeweils massereiche Objekte, die sich gegenseitig in dem Verhältnis des Abstands $r^2$ zum Quadrat antiproportional anziehen, wobei ihre Masse ungleich null ist [8]. Die Newtonsche Gravitationskonstante $G$ liegt bei $6.6742*10^{-11}m^3/{kg*s^2}$ und bestimmt die Stärke der Gravitation zwischen zwei Objekten [8]. Da sie sehr klein ist, hat die Gravitation erst einen spürbaren Effekt, wenn beide Objekte nahbeieinander oder sehr massereich sind, sprich Objekte auf astronomischer Skala [8]. Wenn einer der Massen viel kleiner ist als die andere (z.B. Raumschiffe in Betracht gezogen), ändert sich die Schreibweise zu

$$\vec{F} =m (\frac{GM}{r^2})\tag{2}$$

und da auch $\vec{F} = m*\vec{a}$  eine Kraft beschreibt, lassen sich beide Formeln gleichsetzen und auf die Beschleunigung umformen [8] [Curtis S.22-23]

$$m*\vec{a}=m (\frac{GM}{r^2})\tag{3}$$
$$\vec{g}= \frac{GM} {r^2}\tag{4}$$
^eq-gravibeschleunigung

In Gl. [[#^eq-gravibeschleunigung|(3)]] wird $m$ eliminiert und die Beschleunigung steht allein rechts [[#^eq-gravibeschleunigung|(4)]]. $\vec{g}$ ergibt die **Gravitationsbeschleunigung**, die durch den jeweiligen Körper auf einen anderen einwirkt [Curtis S.23]. Sie ist vor allem in der Astrodynamik relevant, da menschengemachte Satelliten im Verhältnis zu deren Bezugskörper eine insignifikante Masse haben:

$$\vec{F} = G \cdot \frac{M \cdot m}{r^2} = \frac{3{,}986004418 \cdot 10^{14}\ \mathrm{m^3/s^2} \cdot 100\ \mathrm{kg}}{4{,}58464 \cdot 10^{13}\ \mathrm{m^2}}\tag{5}$$
$$\vec{F} = 869{,}4\ \mathrm{N}\tag{6}$$
$$\vec{F} = \frac{G \cdot M}{r^2} = \frac{\mu}{r^2} = \frac{3{,}986004418 \cdot 10^{14}\ \mathrm{m^3/s^2}}{4{,}58464 \cdot 10^{13}\ \mathrm{m^2}}\tag{7}$$
$$\vec{a} = 8{,}694\ \mathrm{m/s^2}\tag{8}$$
Durch $\vec{a}=\frac{\vec{F}}{m}$ ist $869,4\  \mathrm{N}$ gleich $\vec{a} = 8{,}694\ \mathrm{m/s^2}\tag{8}$, da bei der Rundung der Unterschied wegfällt. Newton beschreibt außerdem die 3 Grundgesetze der Bewegung: "1. Jeder Körper verharrt im Zustand der Ruhe oder der gleichförmigen geradlinigen Bewegung, solange er nicht durch einwirkende Kräfte gezwungen wird, diesen Zustand zu ändern. 2. Die Änderung des Bewegungszustands eines Körpers ist proportional zur einwirkenden Kraft und erfolgt entlang der Geraden, in deren Richtung die Kraft wirkt. 3. Die wechselseitigen Einwirkungen zweier Körper aufeinander sind immer gleich groß und entgegengesetzt gerichtet" [1 S.2]. Aus diesen Bedingungen lässt sich praktisch feststellen, wie Körper andere Körper umkreisen können: Es wird angenommen, für die Gravitation gehen die Massen punktförmig vom Massenmittelpunkt aus, der bei der Erde bei $r=0$ liegen würde. Für diese Annahme gilt dann, die Gravitation verläuft immer zum Mittelpunkt jeder Masse, wobei die Richtung als Vektor und die Stärke/Kraft als Magnitude des Vektors dargestellt wird [Curtis S.23]. Da die Kraft quadratisch mit dem Abstand in Gl.(5) abnimmt, ändert sich auch die Magnitude des Vektors. Der Kraftvektor, daraus dann $\vec{g}$, beeinflusst den Geschwindigkeitsvektor des Schiffes. Er steht genau senkrecht zur Gravitation, entlang der Bewegung des Schiffes. Ebenso wird angenommen, eine Patrone eines Scharfschützengewehrs wird bei einer Höhe von 1000m geschossen und es gäbe keinen Luftwiderstand.
![[Obsidian Vault/Evaluation/Grafiken & Bilder/newton_kanonenkugel_FHD.png]]
*Abb. 1, Visualisierung der Erde des Verhältnisses zwischen Geschwindigkeit und Kraft - Claude*

Die Flugbahn der Patrone entspricht einem waagerechten Wurf, bzw. die Höhe in einem freien Fall, dadurch dass die Beschleunigung des freien Falls die Gravitationsbeschleunigung $\vec{g}$ ist [Curtis S.23]. $\vec{g}$ ist immer nach den Mittelpunkt der Kraft ausgerichtet [Curtis S.23]. Die Grafiken in Abb. 1 zeigen, wie die Geschwindigkeit die Flugbahn der Patrone beeinflusst. Die Patrone ist immer in dem freien Fall, fällt durch die Krümmung der Erde auf einem anderen Punkt signifikant weiter, wenn die Geschwindigkeit höher ist [1 S.4]. Gibt das Gewehr einen genug starken Impuls, so wird die Patrone permanent auf die Erde fallen, aber ist schnell genug, sodass sie nicht die Erde trifft. Die Patrone ist erreicht einen Orbit [1 S.4]. Ein Ansatz wäre, dass sie so viel fällt, wie sich die Erde krümmt, sodass sie nicht mehr auf die Erde fallen könnte [1 S.4]. Von diesem Konzept leiten sich die Grundlagen der Himmelsmechanik ab: 
