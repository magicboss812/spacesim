"""Die abgasfahne als partikel -- RELATIV zum schiff, nie in weltkoordinaten.

Jedes partikel lebt in einem raum, der in der schiffsmitte haengt, aber
NICHT mitdreht (achsen wie der bildschirm, y nach oben, einheit wie
`ship/builds.py`: das schiff ist 230 lang). Ausgestossen wird es an einer
duese in der aktuellen nasenrichtung; dreht das schiff, fliegen die alten
partikel in ihrer alten richtung weiter und die fahne kruemmt sich. Die
bewegung des schiffs selbst traegt die ganze fahne mit -- sie existiert nur
relativ zu ihm.

Dazu kommen je frame drei feste lichtpunkte, die nicht simuliert werden:
das gluehen an der duese und, unter vollem schub, die schockknoten
(machsche rauten) auf der achse hinter jeder duese.

`render_points()` liefert (m, 5)-float32-zeilen (x, y, groesse, hitze,
rauch); `render/ship_pixel.py` streut sie als punkte in einen
hitze-puffer im zellraster des schiffs, und `ship_pixel.frag` faerbt die
hitze ueber eine geditherte glut-rampe aus der akzentfarbe.

Rein numerisch -- kein pygame, kein GL.
"""
import math

import numpy as np

MAX_PARTICLES = 1400
# Zuendung: so viele partikel auf einmal, wenn der schub hochfaehrt.
IGNITION_BURST = 70
IGNITION_LEVEL = 0.45


class PlumeParticles:

    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        n = MAX_PARTICLES
        self.pos = np.zeros((n, 2))
        self.vel = np.zeros((n, 2))
        self.age = np.zeros(n)
        self.life = np.ones(n)
        self.size0 = np.zeros(n)
        self.heat0 = np.zeros(n)
        self.alive = np.zeros(n, dtype=bool)
        self._carry = 0.0
        self._prev_level = 0.0
        self.level = 0.0
        self.head = (1.0, 0.0)
        self.nozzles = ()
        self._flicker = 1.0

    def clear(self):
        self.alive[:] = False
        self._carry = 0.0

    def step(self, dt, level, head, nozzles, rate, speed, life, spread):
        """Ein zeitschritt (echtzeit, s). level 0..1 = schub (leerlauf ~0.2)."""
        self.level = float(level)
        self.head = (float(head[0]), float(head[1]))
        self.nozzles = tuple(nozzles)
        dt = max(0.0, min(float(dt), 0.05))
        if dt <= 0.0:
            return
        rng = self.rng
        self._flicker = float(rng.uniform(0.86, 1.08))

        a = self.alive
        self.age[a] += dt
        self.alive &= self.age < self.life
        a = self.alive
        self.vel[a] *= math.exp(-1.8 * dt)
        self.pos[a] += self.vel[a] * dt

        lvl = max(0.0, float(level))
        want = float(rate) * lvl ** 2.5 * dt + self._carry
        count = int(want)
        self._carry = want - count
        burst = IGNITION_BURST if (lvl > IGNITION_LEVEL >= self._prev_level) else 0
        self._prev_level = lvl
        if self.nozzles:
            if count:
                self._emit(count, lvl, speed, life, spread, 1.0)
            if burst:
                self._emit(burst, 1.0, speed * 1.35, life * 0.8, spread * 2.2, 1.5)

    def _emit(self, count, lvl, speed, life, spread, boost):
        rng = self.rng
        free = np.flatnonzero(~self.alive)[:count]
        m = len(free)
        if m == 0:
            return
        noz = np.asarray(self.nozzles, dtype=np.float64)
        weights = noz[:, 2] ** 2
        pick = rng.choice(len(noz), size=m, p=weights / weights.sum())
        nx, ny, nr = noz[pick, 0], noz[pick, 1], noz[pick, 2]
        u = rng.uniform(-0.7, 0.7, m)                      # quer ueber die duese
        lx = nx + rng.uniform(-1.5, 0.5, m)
        ly = ny + u * nr
        sp = speed * rng.uniform(0.75, 1.25, m)
        vx_l = -sp
        vy_l = sp * rng.normal(0.0, spread, m) + u * sp * 0.10
        hx, hy = self.head
        self.pos[free, 0] = hx * lx - hy * ly
        self.pos[free, 1] = hy * lx + hx * ly
        self.vel[free, 0] = hx * vx_l - hy * vy_l
        self.vel[free, 1] = hy * vx_l + hx * vy_l
        self.age[free] = 0.0
        self.life[free] = life * rng.uniform(0.6, 1.2, m)
        self.size0[free] = nr * rng.uniform(0.30, 0.55, m) * boost
        # Kleine duesen geben kleinere partikel und damit weniger
        # ueberlagerung; ohne ausgleich bliebe ihre fahne rot statt weiss.
        comp = np.minimum(3.2, 15.0 / np.maximum(nr, 1.0))
        self.heat0[free] = rng.uniform(0.20, 0.34, m) * (0.25 + 0.75 * lvl) * boost * comp
        self.alive[free] = True

    def render_points(self):
        a = self.alive
        t = np.clip(self.age[a] / self.life[a], 0.0, 1.0)
        size = self.size0[a] * (1.0 + 1.7 * t)
        heat = self.heat0[a] * (1.0 - t) ** 1.7
        smoke = 0.45 * t * (1.0 - t) * 4.0 * np.clip(1.0 - self.heat0[a] * (1.0 - t), 0.0, 1.0)
        parts = [np.column_stack((self.pos[a], size, heat, smoke))]

        # Feste lichtpunkte: gluehen an der duese, schockknoten bei schub.
        hx, hy = self.head
        lvl = self.level
        extra = []
        for x, y, r in self.nozzles:
            def put(lx, ly, sz, h):
                extra.append((hx * lx - hy * ly, hy * lx + hx * ly, sz, h, 0.0))
            put(x - 0.25 * r, y, 1.10 * r, (0.30 + 1.10 * lvl) * self._flicker)
            core = max(0.0, (lvl - 0.40) / 0.60)
            if core > 0.0:
                for i in range(4):
                    d = r * (1.6 + 2.0 * i)
                    put(x - d, y, r * (0.62 - 0.10 * i),
                        (1.9 - 0.38 * i) * core * self._flicker)
        if extra:
            parts.append(np.asarray(extra))
        out = np.concatenate(parts, axis=0) if parts else np.zeros((0, 5))
        return np.ascontiguousarray(out, dtype=np.float32)

    def light_sources(self, max_lights=8):
        """Die fahne als punktlichter fuer den rumpf: je duese das gluehen am
        austritt, dazu bis zu vier hitzegewichtete schwerpunkte der partikel
        (nach alter gebuendelt, alter ~ abstand). Zeilen (x, y, staerke,
        radius) im selben raum wie die partikel."""
        out = []
        hx, hy = self.head
        lvl = self.level
        for x, y, r in self.nozzles:
            lx, ly = x - 0.25 * r, y
            out.append((hx * lx - hy * ly, hy * lx + hx * ly,
                        (0.25 + 1.0 * lvl) * self._flicker, 1.6 * r))
        a = self.alive
        if a.any():
            t = np.clip(self.age[a] / self.life[a], 0.0, 1.0)
            heat = self.heat0[a] * (1.0 - t) ** 1.7
            pos = self.pos[a]
            size = self.size0[a] * (1.0 + 1.7 * t)
            for lo, hi in ((0.0, 0.2), (0.2, 0.45), (0.45, 0.7), (0.7, 1.01)):
                sel = (t >= lo) & (t < hi)
                w = heat[sel]
                total = float(w.sum())
                if total < 0.05:
                    continue
                cx = float((pos[sel, 0] * w).sum() / total)
                cy = float((pos[sel, 1] * w).sum() / total)
                out.append((cx, cy, 0.035 * total, float(size[sel].mean()) * 2.0))
        return out[:max_lights]

    def extent(self):
        """Groesster abstand eines partikels von der schiffsmitte (+ groesse)."""
        a = self.alive
        if not a.any():
            return 0.0
        d = np.hypot(self.pos[a, 0], self.pos[a, 1]) + self.size0[a] * 2.7
        return float(d.max())
