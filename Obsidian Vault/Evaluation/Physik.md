$$F = G \frac{m_1 \times m_2}{r^2}$$

- $F$ — Gravitationskraft
- $G$ — Gravitationskonstante
- $m_1$ — Masse des ersten Körpers
- $m_2$ — Masse des zweiten Körpers
- $r$ — Abstand zwischen den Körpern

[[world.py]]
Die Weltphysik besteht größtenteils nur aus dem Kernelement mit der Newtonschen Graviationsformel. Die Formel läuft in einem speziellen Loop, wo jeder Körper mit dem Schiff in Verbindung gebracht wird. Das Schiff hat dazu die Beschleunigung, die nur von der anderen Masse abhängig ist. So ergibt sich die Formel, wo nur $m_1$ vom Körper, z.B. die Erde, abhängig ist: 
$$a = \frac{G \cdot m_1}{r^2}$$
```python title="pseudo-beispiel.py"
import math
from vec import Vec2
from bodies import body

G = 6.6730831e-11

for jeden Körper in Welt:
    r = Abstand(Schiff, Körper)
    a = G * Körper.masse / r²
    Richtung = normalisiert(Körper.position - Schiff.position)
    Schiff.geschwindigkeit += a * Richtung * dt
```

Jede Beschleunigung ist nach Planeten geteilt und wird dann kombiniert. Und ggf. entscheidet auch die Differenz, wie sich das Raumschiff verhält. Wird es genauso viel vom Mond angezogen wie die Erde in einem weiten Mond-Orbit, so **eliminieren** sich beide Kräfte/Beschleunigungen.

Mit einem [[Integrator]] (Näherungsverfahren) wird dann grob-gesagt die Beschleunigung angewendet. 
