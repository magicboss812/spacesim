# Gleichungsverzeichnis

## Größen

Indizes: $1$/$2$ Start-/Zielbahn, $E$ Erde, $M$ Mars, $\mathrm{J}$ Jupiter, $\mathrm{Sa}$ Saturn, $\odot$ Sonne, $P$ Planet, $0$ Park-/Zielbahn um einen Planeten, $k$ Kreisbahn, $t$ Transferbahn.

| Symbol | Bezeichnung |
| --- | --- |
| $G$ | Gravitationskonstante |
| $m_1, m_2$ | Massen der beiden Körper; in Gl. (18) $m_1$ Masse nach dem Brennvorgang |
| $m_0$ | Masse vor dem Brennvorgang einschließlich Treibstoff |
| $m_j$ | Masse von Körper $j$ |
| $N$; $i, j$ | Anzahl der Körper; Indizes der Körper |
| $F$, $\vec F_1$ | Betrag der Gravitationskraft; Kraft auf Körper 1 |
| $\vec R_i$ | Ortsvektor von Körper $i$ im Inertialsystem |
| $\vec r$, $r$ | relativer Ortsvektor $\vec R_2 - \vec R_1$; Abstand zum Zentralkörper bzw. Bahnradius |
| $\dot{\vec r}$, $\ddot{\vec r}$ | Relativgeschwindigkeit; Relativbeschleunigung |
| $\mu$ | Gravitationsparameter |
| $\vec h$, $h$ | spezifischer Drehimpuls; dessen Betrag |
| $e$ | Exzentrizität (in Gl. (37), (38), (40) der Vorbeiflughyperbel) |
| $\nu$, $\nu_2$ | wahre Anomalie; wahre Anomalie des Schnittpunkts S |
| $a$, $a_t$ | große Halbachse; große Halbachse der Transferellipse |
| $r_p$, $r_a$ | Periapsis- und Apoapsisabstand |
| $r_0$ | Radius der Park- bzw. Zielbahn |
| $r_{SOI}$ | Radius der Einflusssphäre |
| $T$, $T_{syn}$ | Umlaufzeit; synodische Periode |
| $n$ | mittlere Winkelgeschwindigkeit |
| $\gamma_1$ | Phasenwinkel beim Start |
| $t_H$ | Flugzeit des Hohmann-Transfers |
| $\varepsilon$, $\Delta\varepsilon$ | spezifische Bahnenergie; deren Änderung |
| $v$ | Betrag der Geschwindigkeit (in Gl. (39) relativ zur Sonne) |
| $\vec v_1$, $\vec v_2$ | Geschwindigkeit unmittelbar vor und nach einem Manöver (Gl. (15)) |
| $v_k$ | Kreisbahngeschwindigkeit |
| $v_{esc}$ | Fluchtgeschwindigkeit |
| $v_\infty$ | hyperbolische Exzessgeschwindigkeit, Geschwindigkeit relativ zum Planeten |
| $v_p$ | Geschwindigkeit in der Periapsis der Hyperbel |
| $v_{t,1}$, $v_{t,2}$ | Geschwindigkeit auf der Transferbahn an Start- und Zielbahn |
| $\vec v_P$, $v_P$ | Bahngeschwindigkeit des Planeten |
| $v_\perp$, $v_r$ | seitlicher und radialer Anteil der Geschwindigkeit relativ zur Sonne |
| $\Delta\vec v$, $\Delta v$ | Geschwindigkeitsänderung; Geschwindigkeitsaufwand bzw. Impuls |
| $\Delta v_{ges}$ | Geschwindigkeitsaufwand einer Mission |
| $\Delta v_1$, $\Delta v_2$ | erster und zweiter Impuls des Hohmann-Transfers |
| $\Delta v_P$ | Impuls in der Park- bzw. Zielbahn |
| $\Delta v_\mathrm{VB}$ | Geschwindigkeitsänderung durch den Vorbeiflug |
| $v_e$ | effektive Austrittsgeschwindigkeit |
| $\delta$ | Umlenkwinkel |
| $\varphi$ | Winkel zwischen $\vec v_\infty$ und $\vec v_P$ |

## Gleichungen

**(1) Gravitationsgesetz**
$$F = G\,\frac{m_1 m_2}{r^2}\tag{1}$$

**(2) Gravitationskraft in Vektorschreibweise**
$$\vec{F}_1 = \frac{G\,m_1 m_2}{r^2}\cdot\frac{\vec{r}}{r} = \frac{G\,m_1 m_2}{r^3}\,\vec{r}\tag{2}$$

**(3) Bewegungsgleichung des Zweikörperproblems**
$$\ddot{\vec{r}} = -G\,(m_1 + m_2)\,\frac{\vec{r}}{r^3} = -\frac{\mu}{r^3}\,\vec{r}, \qquad \mu = G\,(m_1 + m_2)\tag{3}$$

**(4) Spezifischer Drehimpuls**
$$\vec{h} = \vec{r} \times \dot{\vec{r}}\tag{4}$$

**(5) Bahngleichung**
$$r = \frac{h^2}{\mu}\,\frac{1}{1 + e\cos\nu}\tag{5}$$

**(6) Exzentrizität aus den Apsidenabständen**
$$e = \frac{r_a - r_p}{r_a + r_p}\tag{6}$$

**(7) Große Halbachse**
$$2a = r_p + r_a\tag{7}$$

**(8) Umlaufzeit (drittes Keplersches Gesetz)**
$$T = \frac{2\pi}{\sqrt{\mu}}\,a^{3/2}\tag{8}$$

**(9) Spezifische Bahnenergie**
$$\varepsilon = \frac{v^2}{2} - \frac{\mu}{r}\tag{9}$$

**(10) Bahnenergie und große Halbachse**
$$\varepsilon = -\frac{\mu}{2a}\tag{10}$$

**(11) Vis-viva-Gleichung**
$$v^2 = \mu\left(\frac{2}{r} - \frac{1}{a}\right)\tag{11}$$

**(12) Kreisbahngeschwindigkeit**
$$v_k = \sqrt{\frac{\mu}{r}}\tag{12}$$

**(13) Fluchtgeschwindigkeit**
$$v_{esc} = \sqrt{\frac{2\mu}{r}} = \sqrt{2}\,v_k\tag{13}$$

**(14) Hyperbolische Exzessgeschwindigkeit**
$$v_\infty^2 = 2\,\varepsilon\tag{14}$$

**(15) Geschwindigkeitsänderung beim impulsiven Manöver**
$$\Delta\vec{v} = \vec{v}_2 - \vec{v}_1\tag{15}$$

**(16) Geschwindigkeitsaufwand einer Mission**
$$\Delta v_{ges} = \sum_i \left|\Delta\vec{v}_i\right|\tag{16}$$

**(17) Energieänderung durch einen Schub in Flugrichtung (Oberth-Effekt)**
$$\Delta\varepsilon = v\,\Delta v + \frac{\Delta v^2}{2}\tag{17}$$

**(18) Raketengrundgleichung**
$$\Delta v = v_e \ln\frac{m_0}{m_1}\tag{18}$$

**(19) Bewegungsgleichungen des N-Körperproblems**
$$\ddot{\vec{R}}_i = -G\sum_{j \neq i} m_j\,\frac{\vec{R}_i - \vec{R}_j}{\left|\vec{R}_i - \vec{R}_j\right|^3}, \qquad i = 1, \dots, N\tag{19}$$

**(20) Große Halbachse der Hohmann-Transferellipse**
$$a_t = \frac{r_1 + r_2}{2}\tag{20}$$

**(21) Geschwindigkeiten auf der Transferbahn**
$$v_{t,1} = \sqrt{\mu\left(\frac{2}{r_1} - \frac{1}{a_t}\right)}, \qquad v_{t,2} = \sqrt{\mu\left(\frac{2}{r_2} - \frac{1}{a_t}\right)}\tag{21}$$

**(22) Impulse des Hohmann-Transfers**
$$\Delta v_1 = v_{t,1} - v_{k,1}, \qquad \Delta v_2 = v_{k,2} - v_{t,2}\tag{22}$$

**(23) Geschwindigkeitsaufwand des Hohmann-Transfers**
$$\Delta v = \left|\Delta v_1\right| + \left|\Delta v_2\right|\tag{23}$$

**(24) Flugzeit des Hohmann-Transfers**
$$t_H = \pi\sqrt{\frac{a_t^3}{\mu}}\tag{24}$$

**(25) Mittlere Winkelgeschwindigkeit**
$$n = \frac{2\pi}{T}\tag{25}$$

**(26) Phasenwinkel beim Start**
$$\gamma_1 = \pi - n_2\,t_H\tag{26}$$

**(27) Synodische Periode**
$$T_{syn} = \frac{T_1\,T_2}{\left|T_1 - T_2\right|}\tag{27}$$

**(28) Radius der Einflusssphäre**
$$r_{SOI,E} = r_1\left(\frac{\mu_E}{\mu_\odot}\right)^{2/5}, \qquad r_{SOI,M} = r_2\left(\frac{\mu_M}{\mu_\odot}\right)^{2/5}\tag{28}$$

**(29) Geschwindigkeitsaddition an der Grenze der Einflusssphäre**
$$\vec{v} = \vec{v}_P + \vec{v}_\infty\tag{29}$$

**(30) Exzessgeschwindigkeiten beim Hohmann-Transfer**
$$v_{\infty,1} = \Delta v_1, \qquad v_{\infty,2} = \Delta v_2\tag{30}$$

**(31) Periapsisgeschwindigkeit der Hyperbel**
$$v_p^2 = v_{esc}^2 + v_\infty^2\tag{31}$$

**(32) Impuls in der Park- bzw. Zielbahn**
$$\Delta v_P = v_p - v_{k,0} = \sqrt{\frac{2\mu_P}{r_0} + v_\infty^2} - \sqrt{\frac{\mu_P}{r_0}}\tag{32}$$

**(33) Seitlicher Geschwindigkeitsanteil im Schnittpunkt S**
$$v_\perp = \frac{h}{r_2} = \frac{r_1\,v_{t,1}}{r_2}\tag{33}$$

**(34) Radialer Geschwindigkeitsanteil im Schnittpunkt S**
$$v_r = \sqrt{v_{t,2}^2 - v_\perp^2}\tag{34}$$

**(35) Exzessgeschwindigkeit am Mars beim schnelleren Transfer**
$$v_{\infty,2} = \sqrt{\left(v_{k,2} - v_\perp\right)^2 + v_r^2}\tag{35}$$

**(36) Wahre Anomalie des Schnittpunkts**
$$\cos\nu_2 = \frac{1}{e}\left(\frac{r_1\,(1+e)}{r_2} - 1\right)\tag{36}$$

**(37) Exzentrizität der Vorbeiflughyperbel**
$$e = 1 + \frac{r_p\,v_\infty^2}{\mu_P}\tag{37}$$

**(38) Umlenkwinkel**
$$\sin\frac{\delta}{2} = \frac{1}{e}\tag{38}$$

**(39) Geschwindigkeit relativ zur Sonne nach dem Kosinussatz**
$$v^2 = v_P^2 + v_\infty^2 + 2\,v_P\,v_\infty\cos\varphi\tag{39}$$

**(40) Geschwindigkeitsänderung durch den Vorbeiflug**
$$\Delta v_\mathrm{VB} = 2\,v_\infty\sin\frac{\delta}{2} = \frac{2\,v_\infty}{e}\tag{40}$$

**(41) Geschwindigkeitsanteile nach dem Vorbeiflug am Jupiter**
$$v_{\perp,\mathrm{J}} = v_{k,\mathrm{J}} - v_{\infty,\mathrm{J}}\cos\delta, \qquad v_{r,\mathrm{J}} = v_{\infty,\mathrm{J}}\sin\delta\tag{41}$$

**(42) Energiebedingung für die tangentiale Ankunft an der Saturnbahn**
$$\frac{v_{\perp,\mathrm{J}}^2 + v_{r,\mathrm{J}}^2}{2} - \frac{\mu_\odot}{r_\mathrm{J}} = \frac{1}{2}\left(\frac{r_\mathrm{J}\,v_{\perp,\mathrm{J}}}{r_\mathrm{Sa}}\right)^2 - \frac{\mu_\odot}{r_\mathrm{Sa}}\tag{42}$$
