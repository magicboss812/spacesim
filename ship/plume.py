"""Die abgasfahne als partikel -- RELATIV zum schiff, nie in weltkoordinaten.

Jedes partikel lebt in einem raum, der in der schiffsmitte haengt, aber
NICHT mitdreht (achsen wie der bildschirm, y nach oben, einheit wie
`ship/builds.py`: das schiff ist 230 lang). Ausgestossen wird es an einer
duese in der aktuellen nasenrichtung; dreht das schiff, fliegen die alten
partikel in ihrer alten richtung weiter und die fahne kruemmt sich. Die
bewegung des schiffs selbst traegt die ganze fahne mit -- sie existiert nur
relativ zu ihm.

Das triebwerk ist ein kleines modell, nicht ein skalierungsfaktor:

- **Hochlaufen.** `spool` folgt der hebelstellung mit einer zeitkonstante
  (hoch 0.22 s, runter 0.12 s). Die geschwindigkeit des schiffs aendert
  sich sofort (das macht die physik); die FLAMME braucht ihre zeit.
- **Jedes partikel traegt den zustand seines ausstosses.** Austrittstempo,
  groesse, streuung, hitze und lebensdauer haengen am spool im moment der
  emission. Eine aenderung am hebel wandert deshalb mit der stroemung die
  fahne hinunter, statt die ganze fahne auf einmal umzuskalieren: wenig
  schub gibt eine kurze, breite, flackernde flamme, voller schub eine
  lange, enge mit weissem kern.
- **Vakuum.** Kein umgebungsdruck: also KEINE schockknoten (machsche
  rauten brauchen eine atmosphaere, gegen die der strahl anprallt), kein
  bremsen der partikel und kein quellender rauch. Der strahl ist
  unterexpandiert und weitet sich frei auf -- je hoeher der kammerdruck
  (schub), desto weiter der kegel. An seine stelle tritt ein heisser
  KERN direkt hinter der duese, dessen laenge mit dem schub waechst und
  den die stroemungsfront nach der zuendung freilegt. Was am ende bleibt,
  ist duenner kalter niederschlag (kondensat), kein rauch.
- **Temperatur je partikel.** Sie haengt am spool beim ausstoss (plus
  streuung) und sinkt mit dem alter; der zusammenbau waehlt daraus den
  FARBTON (wenig schub: dunkles rot, voller schub: gelb mit weiss-blauem
  kern), die hitze nur die helligkeit.
- **Zuendung und abschalten.** Beim zuenden ein kleiner, langsamer puff an
  der duese; danach waechst die fahne mit ihren partikeln nach aussen. Beim
  abschalten endet nur der nachschub -- der rest treibt ab und kuehlt aus,
  und die duese glueht langsam nach (`nozzle_temp`, abkuehlung 0.8 s).

`render_points()` liefert (m, 6)-float32-zeilen (x, y, groesse, hitze,
kalt, temperatur) fuer den hitze-puffer, `light_sources()` die fahne als punktlichter
fuer den rumpf.

Rein numerisch -- kein pygame, kein GL.
"""
import math

import numpy as np

MAX_PARTICLES = 1400

SPOOL_UP_S = 0.22
SPOOL_DOWN_S = 0.12
NOZZLE_HEAT_S = 0.10
NOZZLE_COOL_S = 0.80
FIRING = 0.04          # ab diesem spool gilt das triebwerk als laufend
# Flackern: ein neuer zielwert alle FLICKER_HOLD_S, weich angelaufen. Ein
# frischer zufallswert JE FRAME war ein stroboskop bei der bildrate.
FLICKER_HOLD_S = 0.08
FLICKER_TAU_S = 0.04
# Rumpflicht: zeitlich geglaettet. Die schwerpunkte zufaelliger partikel
# springen von frame zu frame, und jede aenderung schaltet halbton-pixel um.
LIGHT_TAU_S = 0.20
N_LIGHT_BUCKETS = 4


def _approach(value, target, dt, tau):
    return value + (target - value) * (1.0 - math.exp(-dt / max(tau, 1e-4)))


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
        self.temp0 = np.zeros(n)
        self.alive = np.zeros(n, dtype=bool)
        self._carry = 0.0
        self.spool = 0.0           # triebwerksleistung 0..1
        self.nozzle_temp = 0.0     # gluehen der duese 0..1 (traege)
        self.front = 0.0           # stroemungsfront seit der zuendung, einheiten
        self.idle = 0.0
        self.head = (1.0, 0.0)
        self.nozzles = ()
        self._firing = False
        self._speed = 0.0
        self._flicker = 1.0
        self._flicker_target = 1.0
        self._flicker_hold = 0.0
        self._lights = None        # (n, 4) geglaettet: x, y, staerke, radius

    # Altname: der renderer las frueher `level` (0..1) -- jetzt der spool.
    @property
    def level(self):
        return max(self.spool, self.idle)

    def clear(self):
        self.alive[:] = False
        self._carry = 0.0
        self.spool = 0.0
        self.nozzle_temp = 0.0
        self.front = 0.0
        self._firing = False
        self._flicker = self._flicker_target = 1.0
        self._flicker_hold = 0.0
        self._lights = None

    def step(self, dt, throttle, idle, head, nozzles, rate, speed, life, spread):
        """Ein echtzeit-schritt. throttle 0..1 = hebelstellung, mit der die
        hauptduese gerade brennt (0 = aus); idle = zuendflamme 0..1."""
        self.head = (float(head[0]), float(head[1]))
        self.nozzles = tuple(nozzles)
        self.idle = max(0.0, float(idle))
        dt = max(0.0, min(float(dt), 0.05))
        if dt <= 0.0:
            return
        rng = self.rng
        target = max(0.0, min(1.0, float(throttle)))
        # Flackern: bei wenig schub (niedriger kammerdruck, unruhige
        # verbrennung) deutlich staerker als bei vollem.
        self._flicker_hold -= dt
        if self._flicker_hold <= 0.0:
            amp = 0.10 + 0.22 * (1.0 - self.spool)
            self._flicker_target = float(1.0 + rng.uniform(-amp, amp * 0.5))
            self._flicker_hold += FLICKER_HOLD_S * rng.uniform(0.6, 1.4)
            self._flicker_hold = max(self._flicker_hold, 0.0)
        self._flicker = _approach(self._flicker, self._flicker_target, dt, FLICKER_TAU_S)

        # --- triebwerk -----------------------------------------------------
        self.spool = _approach(self.spool, target, dt,
                               SPOOL_UP_S if target > self.spool else SPOOL_DOWN_S)
        glow_target = max(self.spool, self.idle * 0.6)
        self.nozzle_temp = _approach(self.nozzle_temp, glow_target, dt,
                                     NOZZLE_HEAT_S if glow_target > self.nozzle_temp
                                     else NOZZLE_COOL_S)
        s = self.spool
        ex_speed = float(speed) * (0.40 + 0.60 * s)
        self._speed = ex_speed
        firing = target > 0.0 or s > FIRING
        if firing and not self._firing and target > 0.0:
            self.front = 0.0
            self._ignite(speed)
        self._firing = firing
        self.front = self.front + ex_speed * dt if firing else 0.0

        # --- bestehende partikel -----------------------------------------
        a = self.alive
        self.age[a] += dt
        self.alive &= self.age < self.life
        a = self.alive
        # Vakuum: nichts bremst. Die partikel fliegen geradeaus weiter und
        # verblassen durch ausdehnung, nicht durch abbremsen.
        self.pos[a] += self.vel[a] * dt

        # --- nachschub: hauptstrom nach spool, dazu die zuendflamme -------
        out = s ** 1.2 + self.idle ** 2.5
        want = float(rate) * out * dt + self._carry
        count = int(want)
        self._carry = want - count
        if count and self.nozzles:
            self._emit(count, s, ex_speed, life, spread)
        self._smooth_lights(dt)

    def _ignite(self, speed):
        """Zuendpuff: wenige, langsame, weit gestreute partikel an der duese
        -- ein aufflackern, kein voller strahl."""
        if not self.nozzles:
            return
        self._emit(24, 0.35, float(speed) * 0.22, 0.30, 0.65, boost=1.2)

    def _emit(self, count, s, ex_speed, life, spread, boost=1.0):
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
        # Vakuum: der strahl ist unterexpandiert und weitet sich frei auf;
        # mehr kammerdruck (schub) heisst weiterer kegel. Wenig schub: eng
        # und kurz.
        cone = float(spread) * (0.55 + 1.25 * s)
        sp = ex_speed * rng.uniform(0.75, 1.25, m)
        vx_l = -sp
        vy_l = sp * rng.normal(0.0, cone, m) + u * sp * 0.10
        hx, hy = self.head
        self.pos[free, 0] = hx * lx - hy * ly
        self.pos[free, 1] = hy * lx + hx * ly
        self.vel[free, 0] = hx * vx_l - hy * vy_l
        self.vel[free, 1] = hy * vx_l + hx * vy_l
        self.age[free] = 0.0
        self.life[free] = float(life) * (0.70 + 0.30 * s) * rng.uniform(0.6, 1.2, m)
        self.size0[free] = nr * rng.uniform(0.30, 0.55, m) * (0.70 + 0.40 * s) * boost
        # Kleine duesen geben kleinere partikel und damit weniger
        # ueberlagerung; ohne ausgleich bliebe ihre fahne rot statt weiss.
        comp = np.minimum(3.2, 15.0 / np.maximum(nr, 1.0))
        self.heat0[free] = rng.uniform(0.20, 0.34, m) * (0.25 + 0.75 * s) * boost * comp
        # Temperatur beim ausstoss: aus dem spool, je partikel gestreut --
        # daher die lebendigen farbwechsel im strahl.
        self.temp0[free] = np.clip((0.22 + 0.78 * s) * rng.uniform(0.85, 1.12, m), 0.0, 1.1)
        self.alive[free] = True

    def _core(self, x, y, r):
        """Heisser kern hinter einer duese: punkte (abstand, groesse, hitze,
        temperatur) entlang der achse, zur spitze hin duenner und kuehler.
        Seine laenge waechst mit dem schub; nach der zuendung legt ihn die
        stroemungsfront frei. (Keine schockknoten -- im vakuum gibt es
        keinen umgebungsdruck, gegen den sie sich bilden koennten.)"""
        s = self.spool
        core = max(0.0, min(1.0, (s - 0.15) / 0.70))
        if core <= 0.0:
            return []
        length = r * (1.2 + 5.0 * core)
        n = 7
        out = []
        for i in range(n):
            f = i / (n - 1)
            d = r * 0.4 + length * f
            reach = min(1.0, max(0.0, (self.front - d) / max(r, 1.0)))
            h = (1.75 - 1.1 * f) * core * reach * self._flicker
            if h > 0.0:
                out.append((d, r * (0.58 - 0.30 * f), h, 1.0 - 0.25 * f))
        return out

    def render_points(self):
        a = self.alive
        t = np.clip(self.age[a] / self.life[a], 0.0, 1.0)
        # Freie ausdehnung: bei vollem schub blaeht sich der strahl staerker
        # auf (ballistisch, kein quellen wie rauch).
        size = self.size0[a] * (1.0 + (1.2 + 1.6 * self.temp0[a]) * t)
        # Langsamer als in luft: im vakuum kuehlt das gas nur durch
        # ausdehnung, nicht durch mischen mit umgebung.
        heat = self.heat0[a] * (1.0 - t) ** 1.4
        temp = self.temp0[a] * (1.0 - 0.55 * t)
        # Kondensat: am ende, wenn das gas ausgekuehlt ist, ein duenner
        # kalter rest -- viel weniger als rauch in einer atmosphaere.
        cold = 0.22 * np.clip((t - 0.45) / 0.55, 0.0, 1.0) * (1.0 - t) * 4.0
        parts = [np.column_stack((self.pos[a], size, heat, cold, temp))]

        # Feste lichtpunkte: gluehen der duese, heisser kern.
        hx, hy = self.head
        extra = []
        for x, y, r in self.nozzles:
            def put(lx, ly, sz, h, tp):
                extra.append((hx * lx - hy * ly, hy * lx + hx * ly, sz, h, 0.0, tp))
            put(x - 0.25 * r, y, 1.10 * r,
                (0.30 * min(1.0, self.idle / 0.22) + 1.10 * self.nozzle_temp) * self._flicker,
                0.25 + 0.70 * self.nozzle_temp)
            for d, sz, h, tp in self._core(x, y, r):
                put(x - d, y, sz, h, tp)
        if extra:
            parts.append(np.asarray(extra))
        out = np.concatenate(parts, axis=0)
        return np.ascontiguousarray(out, dtype=np.float32)

    def _raw_lights(self):
        """Feste plaetze: je duese ein licht, dann N_LIGHT_BUCKETS
        alters-eimer. Ein leerer eimer bleibt mit staerke 0 stehen (statt zu
        verschwinden), damit die glaettung je platz weiterlaeuft. Ohne
        flackern: das gehoert der flamme, nicht dem rumpf -- auf dem halbton
        schaltete es jedes frame pixel um, auch im leerlauf."""
        hx, hy = self.head
        rows = []
        for x, y, r in self.nozzles:
            lx, ly = x - 0.25 * r, y
            strength = 0.25 * min(1.0, self.idle / 0.22) + 1.0 * self.nozzle_temp
            rows.append((hx * lx - hy * ly, hy * lx + hx * ly, strength, 1.6 * r))
        a = self.alive
        t = np.clip(self.age[a] / self.life[a], 0.0, 1.0)
        heat = self.heat0[a] * (1.0 - t) ** 1.7
        pos = self.pos[a]
        size = self.size0[a] * (1.0 + 1.7 * t)
        for lo, hi in ((0.0, 0.2), (0.2, 0.45), (0.45, 0.7), (0.7, 1.01)):
            sel = (t >= lo) & (t < hi)
            w = heat[sel]
            total = float(w.sum())
            if total <= 1e-6:
                rows.append((math.nan, math.nan, 0.0, 0.0))
                continue
            cx = float((pos[sel, 0] * w).sum() / total)
            cy = float((pos[sel, 1] * w).sum() / total)
            # Nur der hauptstrom leuchtet den rumpf an; die paar partikel der
            # zuendflamme kommen und gehen zufaellig und liessen den halbton
            # im leerlauf kriechen. Dort bleibt allein das (ruhige) duesenlicht.
            gate = min(1.0, self.spool / 0.15)
            rows.append((cx, cy, 0.035 * total * gate, float(size[sel].mean()) * 2.0))
        return np.asarray(rows, dtype=float).reshape(-1, 4)

    def _smooth_lights(self, dt):
        raw = self._raw_lights()
        cur = self._lights
        if cur is None or cur.shape != raw.shape:
            self._lights = np.where(np.isnan(raw), 0.0, raw)
            return
        k = 1.0 - math.exp(-dt / LIGHT_TAU_S)
        empty = np.isnan(raw[:, 0])
        fresh = cur[:, 2] <= 1e-4          # war aus: ort sofort uebernehmen
        for i in range(raw.shape[0]):
            if empty[i]:
                cur[i, 2] += (0.0 - cur[i, 2]) * k
            elif fresh[i]:
                cur[i, :2] = raw[i, :2]
                cur[i, 3] = raw[i, 3]
                cur[i, 2] += (raw[i, 2] - cur[i, 2]) * k
            else:
                cur[i] += (raw[i] - cur[i]) * k

    def light_sources(self, max_lights=8):
        """Die fahne als punktlichter fuer den rumpf: je duese das gluehen am
        austritt, dazu bis zu vier hitzegewichtete schwerpunkte der partikel
        (nach alter gebuendelt, alter ~ abstand), zeitlich geglaettet
        (`LIGHT_TAU_S`). Zeilen (x, y, staerke, radius) im selben raum wie
        die partikel."""
        if self._lights is None:
            return []
        out = [tuple(float(v) for v in row) for row in self._lights if row[2] > 1e-3]
        return out[:max_lights]

    def extent(self):
        """Groesster abstand eines partikels von der schiffsmitte (+ groesse)."""
        a = self.alive
        if not a.any():
            return 0.0
        d = np.hypot(self.pos[a, 0], self.pos[a, 1]) + self.size0[a] * 2.7
        return float(d.max())
