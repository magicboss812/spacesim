"""Regressionstest fuer background.py -- sternenfeld und dreiecksgitter.

Reine funktionen: kein fenster, kein GL, kein pygame.
Aufruf: python tests/background_test.py

Die beiden tragenden pruefungen sind §1 und §2:

* §1 pinnt die **gittergeometrie**. Der entwurf, aus dem diese ebene stammt,
  hatte hier einen fehler: er setzte `y = (j+m)*ws/2`, womit die haelfte
  aller knoten auf HALBEN vielfachen von `ws` landete -- also mitten in den
  dreiecken statt auf den kreuzungen. Der test rechnet jeden knoten gegen
  alle drei geradengleichungen und misst die nachbarabstaende.
* §2 pinnt die **stetigkeit**. Es darf nirgends eine harte schwelle geben,
  sonst poppt eine dekade beim zoomen weg.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np

import background
from background import BackgroundLayer

FAILURES = []


def check(condition, label, detail=''):
    if condition:
        print(f"  ok   {label}" + (f": {detail}" if detail else ''))
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label}: {detail}")


# ---------------------------------------------------------------------- §1
print("\n§1  gittergeometrie -- knoten liegen auf den kreuzungen")

WS = 1000.0
verts = background.lattice_vertices(WS, (-6, 6), (-6, 6))
check(verts.shape[0] > 40, "knoten erzeugt", f"{verts.shape[0]} stueck")

# Jeder knoten muss auf einer linie JEDER der drei scharen liegen, d.h.
# dot(P, n)/ws muss fuer alle drei normalen ganzzahlig sein.
worst = 0.0
for nx, ny in background.family_normals():
    t = (verts[:, 0] * nx + verts[:, 1] * ny) / WS
    err = np.abs(t - np.round(t)).max()
    worst = max(worst, float(err))
check(worst < 1e-12, "jeder knoten liegt auf allen drei linienscharen",
      f"max |t - round(t)| = {worst:.3e}")

# Gegenprobe: die fehlerhafte formel des entwurfs. Sie erzeugt knoten auf
# halben vielfachen von ws, die auf schar 1 (y = i*ws) NICHT liegen.
bad = []
S3 = math.sqrt(3.0) / 2.0
for j in range(-4, 5):
    for m in range(-4, 5):
        bad.append(((m - j) * WS / (2 * S3), (j + m) * WS * 0.5))
bad = np.asarray(bad)
t_bad = bad[:, 1] / WS
bad_off = float(np.abs(t_bad - np.round(t_bad)).max())
check(bad_off > 0.4, "gegenprobe: die entwurfs-formel verfehlt schar 1",
      f"max |t - round(t)| = {bad_off:.3f} (halbe zelle)")

# Gleichseitigkeit: der kuerzeste nachbarabstand ist 2/sqrt(3) * ws und tritt
# genau sechsmal je knoten auf.
centre = verts[(np.abs(verts[:, 0]) < 1e-9) & (np.abs(verts[:, 1]) < 1e-9)]
check(centre.shape[0] == 1, "ursprung ist ein knoten")
d = np.hypot(verts[:, 0], verts[:, 1])
d = np.sort(d[d > 1e-9])
expected = 2.0 / math.sqrt(3.0) * WS
check(abs(d[0] - expected) < 1e-9, "kuerzester nachbarabstand = 2/sqrt(3)*ws",
      f"{d[0]:.6f} vs {expected:.6f}")
check(int(np.sum(np.abs(d - expected) < 1e-9)) == 6,
      "und er tritt sechsmal auf (gleichseitig)",
      f"{int(np.sum(np.abs(d - expected) < 1e-9))} nachbarn")

# Paritaet: p + q ungerade darf NICHT vorkommen.
q = verts[:, 0] * math.sqrt(3.0) / WS
p = verts[:, 1] / WS
parity = np.abs(np.round(p) + np.round(q)) % 2
check(float(parity.max()) < 1e-9, "alle knoten erfuellen p + q gerade")


# ---------------------------------------------------------------------- §2
print("\n§2  stetigkeit -- keine dekade darf springen")

bg = BackgroundLayer()
bg.grid_fade = 1.0

# Zwoelf dekaden zoom in feinen schritten. Wir verfolgen die summe der
# deckkraefte und die je dekade -- beides muss stetig sein.
steps = 4000
scales = np.logspace(-8.0, 4.0, steps)
# Den ersten schritt vorbelegen: sonst vergleicht die schleife gegen ein
# leeres dict und meldet den start des feldes als sprung.
prev_by_k = {level.k: level.alpha for level in bg.levels(float(scales[0]))}
max_jump = 0.0
max_jump_at = 0.0
max_total_jump = 0.0
prev_total = None
n_levels_seen = set()

for s in scales:
    lv = bg.levels(float(s))
    n_levels_seen.add(len(lv))
    by_k = {level.k: level.alpha for level in lv}
    total = sum(by_k.values())

    for k in set(by_k) | set(prev_by_k):
        jump = abs(by_k.get(k, 0.0) - prev_by_k.get(k, 0.0))
        if jump > max_jump:
            max_jump = jump
            max_jump_at = float(s)
    if prev_total is not None:
        max_total_jump = max(max_total_jump, abs(total - prev_total))
    prev_by_k = by_k
    prev_total = total

check(max_jump < 0.02, "keine dekade aendert ihre deckkraft sprunghaft",
      f"max |d alpha| = {max_jump:.5f} bei scale {max_jump_at:.3e}")
check(max_total_jump < 0.02, "auch die summe laeuft stetig",
      f"max |d sum| = {max_total_jump:.5f}")
check(max(n_levels_seen) <= background.MAX_LEVELS,
      f"nie mehr als MAX_LEVELS={background.MAX_LEVELS} dekaden",
      f"gesehen: {sorted(n_levels_seen)}")
check(min(n_levels_seen) >= 1, "und nie weniger als eine",
      f"gesehen: {sorted(n_levels_seen)}")

# Eine dekade betritt/verlaesst die liste genau bei deckkraft null.
entering = []
for s in np.logspace(-8.0, 4.0, 1500):
    for level in bg.levels(float(s)):
        entering.append(level.alpha)
check(min(entering) >= 0.0 and max(entering) <= 1.0,
      "jede gelistete deckkraft liegt in [0, 1]",
      f"[{min(entering):.4f}, {max(entering):.4f}]")

# Knoten kommen spaeter als die linien und nie ohne sie.
bad_nodes = 0
for s in np.logspace(-8.0, 4.0, 1500):
    for level in bg.levels(float(s)):
        if level.node_alpha > level.alpha + 1e-12:
            bad_nodes += 1
check(bad_nodes == 0, "knoten sind nie heller als die linien ihrer dekade",
      f"{bad_nodes} verstoesse")


# ---------------------------------------------------------------------- §3
print("\n§3  sternenfeld -- determinismus")

a1 = background.build_star_table(260)
a2 = background.build_star_table(260)
check(np.array_equal(a1, a2), "gleicher seed -> bitgleiche tabelle")
check(a1.shape == (260, 7), "form stimmt (7 spalten inkl. zoomphase)",
      f"{a1.shape}")

wide = background.build_star_table(400)
check(np.array_equal(wide[:260], a1),
      "groessere dichte haengt nur hinten an, die ersten 260 bleiben stehen")

check(float(a1[:, 4].min()) >= 0.05 and float(a1[:, 4].max()) <= 0.55,
      "parallaxen-tiefe im entwurfsbereich 0.05..0.55",
      f"[{a1[:, 4].min():.3f}, {a1[:, 4].max():.3f}]")
check(float(a1[:, 2].min()) >= 0.4 and float(a1[:, 2].max()) <= 1.9,
      "radius im entwurfsbereich 0.4..1.9",
      f"[{a1[:, 2].min():.3f}, {a1[:, 2].max():.3f}]")
check(float(a1[:, 6].min()) >= 0.0 and float(a1[:, 6].max()) < 1.0,
      "zoomphase gleichverteilt in [0, 1)",
      f"[{a1[:, 6].min():.3f}, {a1[:, 6].max():.3f}]")

bg2 = BackgroundLayer()
bg2.star_density = 100
_ = bg2.star_table()
check(bg2.take_stars_dirty() is True, "erste tabelle meldet sich als dirty")
check(bg2.take_stars_dirty() is False, "und danach genau einmal nicht mehr")
bg2.star_density = 120
_ = bg2.star_table()
check(bg2.take_stars_dirty() is True, "dichteaenderung meldet sich erneut")


# ---------------------------------------------------------------------- §4
print("\n§4  sterndrift -- an der EIGENGESCHWINDIGKEIT, nicht am zoom")

VP = (1280.0, 800.0)
ORIGIN = (0.0, 0.0)
import io as _io
import json as _json


def drift(scale, velocity, steps=60, dt=1 / 60, motion=0.5, cam=ORIGIN):
    """Sterndrift nach `steps` bildern bei fester geschwindigkeit."""
    layer = BackgroundLayer()
    layer.star_motion_scale = motion
    for _ in range(steps):
        layer.update(dt, scale, scale, cam, focus_velocity=velocity,
                     viewport=VP)
    return layer.star_pan_px.copy()


# DAS ist die regression, die den nutzer gestoert hat: derselbe koerper,
# derselbe flug, vierzehn zehnerpotenzen zoom dazwischen -- die sterne
# muessen sich identisch bewegen. Vorher ging die drift ueber
# `delta_welt * camera.scale`, war also proportional zum zoom.
V = (1.0e3, 0.0)                       # 1 km/s
runs = {sc: drift(sc, V) for sc in (1e-9, 1e-6, 1e-3, 1e0, 1e5)}
spread = max(abs(float(r[0]) - float(runs[1e-6][0])) for r in runs.values())
check(spread < 1e-9,
      "ueber 14 zehnerpotenzen zoom ist die drift bit-identisch",
      f"max abweichung {spread:.3e} px")

# star_motion_scale ist "px je sekunde bei 1 km/s" -- also nachrechenbar.
check(abs(float(runs[1e-6][0]) - 0.5) < 1e-9,
      "1 km/s, 1 s, scale 0.5 -> genau 0.5 px",
      f"{float(runs[1e-6][0]):.9f} px")

# Zoomen allein bewegt nichts.
bg3 = BackgroundLayer()
sc = 1e-6
bg3.update(1 / 60, sc, sc, ORIGIN, focus_velocity=(0.0, 0.0), viewport=VP)
before = bg3.star_pan_px.copy()
for _ in range(30):
    sc *= 1.1
    bg3.update(1 / 60, sc, sc, ORIGIN, focus_velocity=(0.0, 0.0), viewport=VP)
check(np.allclose(bg3.star_pan_px, before),
      "zoom allein bewegt die sterne nicht",
      f"d = {np.abs(bg3.star_pan_px - before).max():.3e} px")

# Vorzeichen: nach +x fliegen laesst die sterne nach -x wandern. Der shader
# negiert star_pan_px, also muss die x-komponente hier POSITIV sein.
check(float(runs[1e-6][0]) > 0.0 and abs(float(runs[1e-6][1])) < 1e-12,
      "flug nach +x treibt nur die x-komponente, mit richtigem vorzeichen",
      f"({runs[1e-6][0]:+.4f}, {runs[1e-6][1]:+.4f})")
south = drift(1e-6, (0.0, 1.0e3))
check(float(south[1]) < 0.0,
      "flug nach +y (welt-y nach oben) treibt top-down-y negativ",
      f"{float(south[1]):+.4f} px")

# FREIE KAMERA. Ohne verfolgten koerper gibt es keine eigengeschwindigkeit,
# und der schwenk treibt die sterne -- als BILDSCHIRM-bewegung gelesen, nicht
# als weltgeschwindigkeit. Das ist die zweite regression dieser art: in
# weltmetern gerechnet sind 0.8 schirme je sekunde bei 1e-9 px/m rund
# 1e12 m/s, also tausendfach ueber der notbremse. Die sterne rasten dann bei
# JEDEM schwenk mit voller klammergeschwindigkeit davon, egal wie langsam man
# schwenkt -- und umgekehrt bei 1e5 px/m gar nicht.
def pan_drift(scale, screens=1.0, steps=60):
    """Sterndrift, wenn die kamera um `screens` bildbreiten schwenkt."""
    layer = BackgroundLayer()
    layer.star_motion_scale = 0.5
    per = VP[0] * screens / scale / steps          # meter je schritt
    layer.update(1 / 60, scale, scale, (0.0, 0.0), viewport=VP)
    for i in range(steps):
        layer.update(1 / 60, scale, scale, ((i + 1) * per, 0.0), viewport=VP)
    return float(layer.star_pan_px[0])


pans = {sc: pan_drift(sc) for sc in (1e-9, 1e-6, 1e-3, 1.0, 1e5)}
pan_spread = max(pans.values()) - min(pans.values())
check(pan_spread < 1e-6,
      "ein schwenk um EINE bildbreite treibt die sterne ueber 14 "
      "zehnerpotenzen zoom gleich weit",
      f"{min(pans.values()):.4f}..{max(pans.values()):.4f} px "
      f"(spanne {pan_spread:.3e})")
pan_expect = VP[0] * 0.5 * background.FREE_PAN_GAIN
check(abs(pans[1e-6] - pan_expect) < 1e-6,
      "und zwar um bildbreite * star_motion_scale * FREE_PAN_GAIN",
      f"{pans[1e-6]:.4f} px gegen {pan_expect:.4f}")

# Gegenprobe: als weltgeschwindigkeit gelesen liefe derselbe schwenk in die
# notbremse -- und zwar bei jedem zoom unter etwa 1e-4 px/m.
naive = VP[0] / 1e-9 / (0.5 / background.STAR_SPEED_UNIT) ** -1 / 60.0
check(naive > background.STAR_PAN_CLAMP_FRAC * math.hypot(*VP) * 1e3,
      "gegenprobe: in weltmetern gerechnet laege der schwenk weit ueber "
      "der klammer",
      f"{naive:.3e} px/bild gegen klammer "
      f"{background.STAR_PAN_CLAMP_FRAC * math.hypot(*VP):.1f}")

# Ein bezugsrahmen-wechsel kann gar nichts mehr anrichten: das modell rechnet
# in ABSOLUTEN groessen, die der wechsel nicht anfasst.
plain = drift(1e-6, V, steps=10)
check(abs(float(plain[0]) - 10.0 / 60.0 * 0.5) < 1e-9,
      "gegenprobe: die drift haengt nur an v und dt, an nichts sonst",
      f"{float(plain[0]):.9f} px")

# Notbremse.
bg6 = BackgroundLayer()
bg6.star_motion_scale = 5.0
bg6.update(1 / 60, 1e-6, 1e-6, ORIGIN, focus_velocity=(1.0e12, 0.0),
           viewport=VP)
limit = background.STAR_PAN_CLAMP_FRAC * math.hypot(*VP)
check(abs(float(bg6.star_pan_px[0])) <= limit + 1e-9,
      "eine absurde geschwindigkeit wird auf die klammer gekappt",
      f"{abs(float(bg6.star_pan_px[0])):.2f} <= {limit:.2f} px")


# --------------------------------------------------------------------- §4c
print("\n§4c  die eigengeschwindigkeit kommt aus der POSITION")

# DAS ist die zweite regression, die den nutzer gestoert hat, und sie sass
# nicht in dieser datei: `body.velocity` ist fuer JEDEN himmelskoerper exakt
# (0, 0). solar_system.json setzt es so, und `world.update_planets` schreibt
# nur die kepler-POSITION zurueck, nie die geschwindigkeit. Wer sich auf das
# feld verliess, bekam ein stillstehendes sternenfeld, sobald irgendetwas
# ausser dem (integrierten) Schiff angeschaut wurde.
#
# Gegenprobe zuerst -- ohne sie ist der rest dieses abschnitts wertlos:
_sys = _json.loads(_io.open(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'solar_system.json'), encoding='utf-8-sig').read())
_entries = _sys['bodies'] if isinstance(_sys, dict) and 'bodies' in _sys else _sys
_scripted = [b for b in _entries if b.get('fixed')]
check(_scripted and all(
    tuple(b.get('velocity') or (0, 0)) == (0, 0) for b in _scripted),
    "gegenprobe: JEDER geskriptete koerper hat velocity = [0, 0] -- das feld "
    "taugt nicht als quelle",
    f"{len(_scripted)} koerper geprueft")

ORBIT_V = 2.98e4        # Erde, m/s


def orbiting(steps=60, dt_sim=1.0, key="Erde", motion=0.5):
    """Ein koerper bewegt sich; velocity wird NICHT uebergeben."""
    layer = BackgroundLayer()
    layer.star_motion_scale = motion
    for i in range(steps + 1):
        layer.update(1 / 60, 1e-6, 1e-6, ORIGIN,
                     focus_world_xy=(ORBIT_V * i * dt_sim, 0.0),
                     focus_key=key, sim_time=i * dt_sim, viewport=VP)
    return layer


moved = orbiting()
check(float(moved.star_pan_px[0]) > 0.0,
      "ein koerper ohne velocity-feld bewegt die sterne trotzdem",
      f"{float(moved.star_pan_px[0]):+.4f} px in 1 s")
check(abs(float(moved.star_pan_px[0]) - ORBIT_V / 1000.0 * 0.5) < 1e-6,
      "und zwar mit genau der abgeleiteten geschwindigkeit",
      f"{float(moved.star_pan_px[0]):.6f} px gegen "
      f"{ORBIT_V / 1000.0 * 0.5:.6f}")

# ZEITRAFFER darf sie nicht beschleunigen: geteilt wird durch die SIM-zeit,
# geschritten wird mit der ECHTEN. Sonst stroben die sterne bei 1 y/s.
warped = orbiting(dt_sim=86400.0)          # 1 d je bild statt 1 s
check(abs(float(warped.star_pan_px[0]) - float(moved.star_pan_px[0])) < 1e-9,
      "zeitraffer beschleunigt die sterne nicht -- 1 d/bild wie 1 s/bild",
      f"{float(warped.star_pan_px[0]):.6f} gegen "
      f"{float(moved.star_pan_px[0]):.6f} px")

# Ein KOERPERWECHSEL ist keine bewegung. 1e11 m in einem bild waeren als
# geschwindigkeit gelesen sofort die notbremse -- das feld risse quer ueber
# den schirm.
switch = BackgroundLayer()
switch.star_motion_scale = 0.5
switch.update(1 / 60, 1e-6, 1e-6, ORIGIN, focus_world_xy=(0.0, 0.0),
              focus_key="Schiff", sim_time=0.0, viewport=VP)
switch.update(1 / 60, 1e-6, 1e-6, ORIGIN, focus_world_xy=(0.0, 0.0),
              focus_key="Schiff", sim_time=1.0, viewport=VP)
switch.update(1 / 60, 1e-6, 1e-6, ORIGIN, focus_world_xy=(2.3e11, 0.0),
              focus_key="Mars", sim_time=2.0, viewport=VP)
check(abs(float(switch.star_pan_px[0])) < 1e-12,
      "ein koerperwechsel setzt neu an, statt 1e11 m als flug zu lesen",
      f"{float(switch.star_pan_px[0]):+.3e} px")

# Gegenprobe: ohne den schluessel waere derselbe wechsel voll in der klammer.
naive = BackgroundLayer()
naive.star_motion_scale = 0.5
naive.update(1 / 60, 1e-6, 1e-6, ORIGIN,
             focus_velocity=(2.3e11 / 1.0, 0.0), viewport=VP)
check(abs(float(naive.star_pan_px[0])) >= limit - 1e-9,
      "gegenprobe: als flug gelesen ginge er direkt in die notbremse",
      f"{abs(float(naive.star_pan_px[0])):.1f} px = klammer {limit:.1f}")

# Eine echte geschwindigkeit hat weiter vorrang -- das Schiff traegt eine.
override = BackgroundLayer()
override.star_motion_scale = 0.5
override.update(1 / 60, 1e-6, 1e-6, ORIGIN, focus_velocity=(1.0e3, 0.0),
                focus_world_xy=(9.9e9, 0.0), focus_key="Schiff",
                sim_time=1.0, viewport=VP)
check(abs(float(override.star_pan_px[0]) - 0.5 / 60.0) < 1e-9,
      "focus_velocity hat vorrang vor der ableitung",
      f"{float(override.star_pan_px[0]):.9f} px")


# --------------------------------------------------------------------- §4b
print("\n§4b  atmendes sternenfeld -- dichte bleibt beim zoomen konstant")

# Der shader (star.vert) rechnet je stern:
#   f = fract(star_zoom + zoomphase);  e = mix(1, 2^f, amount)
#   w = mix(1, fenster(f), amount)
# Ein stern ist mit wahrscheinlichkeit 1/e^2 im bild (seine kachel ist
# viewport*e gross), gewichtet mit w. Die summe ist die erwartete sichtbare
# sternzahl -- und die muss vom zoom unabhaengig sein, sonst verklumpt das
# feld beim herauszoomen.
phases = background.build_star_table(4000)[:, 6].astype(np.float64)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def visible(star_zoom, amount):
    f = np.mod(star_zoom + phases, 1.0)
    e = 1.0 + (np.exp2(f) - 1.0) * amount
    w = 1.0 + (_ss(0.0, 0.18, f) * (1.0 - _ss(0.82, 1.0, f)) - 1.0) * amount
    return float(np.sum(w / (e * e)))


for amount in (0.35, 1.0):
    counts = [visible(z, amount) for z in np.linspace(0.0, 8.0, 400)]
    lo, hi = min(counts), max(counts)
    rel = (hi - lo) / (0.5 * (hi + lo))
    check(rel < 0.03,
          f"amount={amount}: sichtbare sternzahl schwankt ueber 8 oktaven "
          f"unter 3 %", f"{lo:.1f}..{hi:.1f} ({rel * 100:.2f} %)")

# Stetigkeit: kein stern darf beim oktavwechsel aufblitzen. Am rand des
# fensters ist w null, dort springt e von 2 auf 1 -- unsichtbar.
edge = _ss(0.0, 0.18, 0.0) * (1.0 - _ss(0.82, 1.0, 0.0))
edge_hi = _ss(0.0, 0.18, 1.0) * (1.0 - _ss(0.82, 1.0, 1.0))
check(abs(edge) < 1e-12 and abs(edge_hi) < 1e-12,
      "das ausblendfenster ist an BEIDEN oktavraendern exakt null",
      f"w(0)={edge:.2e}, w(1)={edge_hi:.2e}")

# amount = 0 muss EXAKT das starre feld ergeben (e = 1, w = 1).
f0 = np.mod(0.0 + phases, 1.0)
e0 = 1.0 + (np.exp2(f0) - 1.0) * 0.0
w0 = 1.0 + (_ss(0.0, 0.18, f0) * (1.0 - _ss(0.82, 1.0, f0)) - 1.0) * 0.0
check(np.allclose(e0, 1.0) and np.allclose(w0, 1.0),
      "amount=0 ergibt exakt das starre feld (e=1, w=1)")

layer = BackgroundLayer()
layer.star_zoom_influence = 2.5
check(abs(layer.zoom_amount() - 1.0) < 1e-12,
      "zoom_amount klemmt nach oben auf 1")
layer.star_zoom_influence = -3.0
check(abs(layer.zoom_amount()) < 1e-12, "und nach unten auf 0")

zl = BackgroundLayer()
zl.update(1 / 60, 4.0, 4.0, ORIGIN, focus_velocity=(0.0, 0.0), viewport=VP)
check(abs(zl.star_zoom - 2.0 * background.STAR_ZOOM_RATE) < 1e-12,
      "star_zoom = log2(scale) * STAR_ZOOM_RATE", f"{zl.star_zoom:.6f}")


# ---------------------------------------------------------------------- §5
print("\n§5  leerlauf-ausblenden")

bg7 = BackgroundLayer()
bg7.idle_fade_delay = 0.5
scale = 1e-6
# Zoomen -> gitter kommt.
for _ in range(60):
    scale *= 1.02
    bg7.update(1 / 60, scale, scale, ORIGIN, viewport=VP)
check(bg7.grid_fade > 0.9, "zoomen blendet das gitter ein",
      f"grid_fade = {bg7.grid_fade:.3f}")

# Stillstand -> nach der verzoegerung geht es weg.
for _ in range(240):
    bg7.update(1 / 60, scale, scale, ORIGIN, viewport=VP)
check(bg7.grid_fade < 0.1, "stillstand blendet es wieder aus",
      f"grid_fade = {bg7.grid_fade:.3f}")

# Schwenken allein darf es NICHT zurueckholen.
bg7.update(1 / 60, scale, scale, ORIGIN, viewport=VP)
fade_before = bg7.grid_fade
for i in range(60):
    bg7.update(1 / 60, scale, scale, (i * 1.0e9, 0.0), viewport=VP)
check(bg7.grid_fade <= fade_before + 1e-9,
      "schwenken allein holt das gitter nicht zurueck",
      f"{fade_before:.4f} -> {bg7.grid_fade:.4f}")

# Ausgeschaltet -> keine dekaden.
bg8 = BackgroundLayer()
bg8.grid_fade = 1.0
bg8.grid_enabled = False
check(bg8.levels(1e-6) == [], "grid_enabled=False liefert keine dekaden")


# ---------------------------------------------------------------------- §6
print("\n§6  gitterphase -- praezise trotz 1e11 m kameraposition")

# Ein knoten muss auch bei riesigem offset auf ganzzahliger phase landen.
ws = 1.0e6
cam_x = 4.0e11
cam_y = -1.5e11
pa, pb = BackgroundLayer._phases(ws, cam_x, cam_y)
check(abs(pa) <= 2.0 and abs(pb) <= 2.0, "phasen liegen in (-2, 2)",
      f"a={pa:.6f}, b={pb:.6f}")
# Um ws verschoben muss die phase b sich um genau 1 aendern (modulo 2).
_, pb2 = BackgroundLayer._phases(ws, cam_x, cam_y + ws)
delta = (pb2 - pb) % 2.0
check(abs(delta - 1.0) < 1e-6,
      "verschiebung um eine zellweite dreht phase b um genau 1",
      f"d = {delta:.9f}")

# Farbe.
check(background.parse_hex_color("#ec3013") == (236 / 255, 48 / 255, 19 / 255),
      "hex-farbe wird korrekt zerlegt")
# Der ausweichwert ist das HUD-cyan -- und zwar BITGLEICH mit der geparsten
# farbe, nicht eine gerundete abschrift davon.
check(background.parse_hex_color("nonsense")
      == background.parse_hex_color(background.DEFAULT_ACCENT_HEX),
      "unsinn faellt exakt auf die geparste vorgabefarbe zurueck",
      f"{background.parse_hex_color('nonsense')}")
check(background.DEFAULT_ACCENT_HEX == "#17b2c4",
      "und die vorgabe ist das HUD-cyan aus ui/theme.py SCHEME[0]")
check(background.parse_hex_color("#abc") == (0xAA / 255, 0xBB / 255, 0xCC / 255),
      "kurzform #abc wird expandiert")


# ---------------------------------------------------------------------- §8
print("\n§8  gitteranker -- ein festes lattice IM PLOT-FRAME")

# Das gitter haengt an keiner geschwindigkeit mehr, sondern an der
# kameraposition im aktiven bezugsrahmen. Daraus folgt alles, was der nutzer
# sehen will, ohne dass es eigens gebaut werden muesste.
FAST = 1.0e9        # grenze praktisch aus: reine positionstreue pruefen


def tracked(scale=1e-6, anchor="frame", limit=FAST):
    layer = BackgroundLayer()
    layer.grid_anchor = anchor
    layer.grid_max_speed_px = limit
    layer.grid_fade = 1.0
    return layer


def run(layer, path, focus=None, scale=1e-6, dt=1 / 60):
    """`path` abfahren, jeweils die anker zurueckgeben."""
    out = []
    for i, cam in enumerate(path):
        f = None if focus is None else focus[i]
        layer.update(dt, scale, scale, ORIGIN,
                     grid_target=layer.grid_target_xy(cam, f), viewport=VP)
        out.append(layer.anchor_xy())
    return out


check(set(background.GRID_ANCHORS) == {"frame", "focus"},
      "GRID_ANCHORS ist die vollstaendige liste",
      f"{background.GRID_ANCHORS}")

# 1. Der bezugskoerper steht still. Im koerperfesten plot-frame liegt er im
#    ursprung; sitzt die kamera auf ihm, ist cam_frame konstant null.
still = run(tracked(), [(0.0, 0.0)] * 120)
check(max(abs(a[0]) + abs(a[1]) for a in still) < 1e-12,
      "kamera auf dem bezugskoerper -> das gitter steht exakt still",
      f"max |anker| {max(abs(a[0]) + abs(a[1]) for a in still):.3e} m")

# 2. Ein SCHWENK verschiebt das gitter um genau die strecke, um die sich die
#    welt verschiebt -- nicht mehr und nicht weniger. Das war der zweite
#    beschwerdepunkt: vorher trieb eine abgeleitete "geschwindigkeit" das
#    gitter in die klammer, und es raste beim schwenken davon.
pan = [(i * 500.0, 0.0) for i in range(61)]
pan_anchor = run(tracked(), pan)
worst = max(abs(a[0] - c[0]) + abs(a[1] - c[1])
            for a, c in zip(pan_anchor, pan))
check(worst < 1e-9,
      "ein schwenk schiebt das gitter um exakt die schwenkstrecke",
      f"groesste abweichung {worst:.3e} m auf 30 km")

# 3. Eine KREISBAHN zeichnet einen kreis. Der anker ist die position im
#    bezugsrahmen, also faehrt er die bahn selbst ab -- die schleife schliesst
#    sich, weil sie dieselbe schleife IST.
R_ORB, TURNS = 7.0e6, 720
circle = [(R_ORB * math.cos(2 * math.pi * i / TURNS),
           R_ORB * math.sin(2 * math.pi * i / TURNS))
          for i in range(TURNS + 1)]
orbit = run(tracked(scale=1e-5), circle, scale=1e-5)
xs = [a[0] for a in orbit]
ys = [a[1] for a in orbit]
check(min(xs) < 0.0 < max(xs) and min(ys) < 0.0 < max(ys),
      "auf einer kreisbahn laeuft das gitter in BEIDE richtungen je achse",
      f"x [{min(xs):+.3e}, {max(xs):+.3e}], y [{min(ys):+.3e}, {max(ys):+.3e}]")
closing = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
check(closing < 1e-6 * R_ORB,
      "und schliesst sich nach einem umlauf exakt -- die bahn liest sich "
      "als bahn", f"restversatz {closing:.3e} m gegen radius {R_ORB:.3e} m")
ratio = (max(xs) - min(xs)) / (max(ys) - min(ys))
check(abs(ratio - 1.0) < 1e-6,
      "der weg ist ein kreis, keine gerade",
      f"achsverhaeltnis {ratio:.6f}")

# 4. anchor="focus" nimmt dem gitter genau diese bewegung wieder -- es klebt
#    dann am blickziel und zeigt nur noch den massstab.
glued = run(tracked(anchor="focus", scale=1e-5), circle, focus=circle,
            scale=1e-5)
check(max(abs(a[0]) + abs(a[1]) for a in glued) < 1e-9,
      'anchor="focus": dieselbe kreisbahn laesst das gitter still stehen',
      f"max |anker| {max(abs(a[0]) + abs(a[1]) for a in glued):.3e} m")

# Und der anker ist wirklich die DIFFERENZ, nicht die kameraposition.
gt = BackgroundLayer()
gt.grid_anchor = "focus"
ax, ay = gt.grid_target_xy((1.0e11, 2.0e11), (1.0e11, 2.0e11))
check(abs(ax) < 1e-9 and abs(ay) < 1e-9,
      'anchor="focus": kamera auf dem koerper -> ziel im ursprung',
      f"({ax:.3e}, {ay:.3e})")
ax, _ = gt.grid_target_xy((1.0e11 + 500.0, 2.0e11), (1.0e11, 2.0e11))
check(abs(ax - 500.0) < 1e-6,
      "ein schwenk von 500 m verschiebt auch dort das ziel um 500 m",
      f"{ax:.6f} m")
ax, ay = gt.grid_target_xy((123.0, 456.0), None)
check(abs(ax - 123.0) < 1e-9 and abs(ay - 456.0) < 1e-9,
      'ohne verfolgten koerper faellt "focus" auf die kamera zurueck',
      f"({ax}, {ay})")
gf = BackgroundLayer()
fx, fy = gf.grid_target_xy((123.0, 456.0), (7.0, 8.0))
check(abs(fx - 123.0) < 1e-9 and abs(fy - 456.0) < 1e-9,
      '"frame" ignoriert den verfolgten koerper',
      f"({fx}, {fy})")


# --------------------------------------------------------------------- §8b
print("\n§8b  die geschwindigkeitsgrenze -- lesbar statt schmierstreifen")

# Weltfest ist ehrlich, aber bei 1e2 px/m rast ein schiff mit 7.7 km/s um
# 8e5 px/s vorbei. Deshalb laeuft der ANKER dem wahren wert nur mit
# `grid_max_speed_px` nach. Zwei zusagen, und beide sind noetig:
#
#   1. UNTER der grenze ist das gitter EXAKT weltfest -- sonst logen die
#      zellen ueber den abstand, gerade wenn man genau hinsieht.
#   2. UEBER der grenze gleitet es mit genau dieser rate, nicht schneller.
LIMIT = 600.0


def slide(scale, speed_m_s, steps=120, dt=1 / 60, limit=LIMIT):
    """Geradeausflug; gibt (layer, ZURUECKGELEGTE strecke in px) zurueck.

    ACHTUNG: `grid_anchor_m` wird modulo `fold_spans()` gefaltet -- bei
    1e2 px/m ist die x-periode nur 115 m, ein flug von 15 km wickelt sich
    also 133-mal um. Der rohe endwert taugt darum nicht als weg; die
    schritte muessen einzeln entfaltet werden.
    """
    layer = tracked(scale=scale, limit=limit)
    span = background.fold_spans(scale)[0]
    prev = None
    total = 0.0
    for i in range(steps + 1):
        cam = (speed_m_s * i * dt, 0.0)
        layer.update(dt, scale, scale, ORIGIN,
                     grid_target=layer.grid_target_xy(cam), viewport=VP)
        cur = layer.anchor_xy()[0]
        if prev is not None:
            total += background.fold(cur - prev, span)
        prev = cur
    return layer, total * scale


# Langsam genug: die grenze greift nicht, der rueckstand bleibt exakt null.
lay, moved = slide(1e-6, 1.0e6)                # 1e6 m/s * 1e-6 px/m = 1 px/s
check(lay.grid_lag_px < 1e-9,
      "unter der grenze haengt das gitter um NULL pixel hinterher",
      f"{lay.grid_lag_px:.3e} px")
check(abs(moved - 2.0) < 1e-6,
      "und legt in 2 s exakt die wahren 2 px zurueck",
      f"{moved:.6f} px")

# Zu schnell: es gleitet mit genau der grenze, unabhaengig davon, wie absurd
# die wahre geschwindigkeit ist.
rates = {}
for v in (7.7e3, 7.7e5, 7.7e9):
    lay, moved = slide(1e2, v)                 # 1e2 px/m: der extremzoom
    rates[v] = moved / 2.0
spread = max(rates.values()) - min(rates.values())
check(spread < 1e-6 and abs(rates[7.7e3] - LIMIT) < 1e-3,
      "ueber der grenze gleitet es mit exakt grid_max_speed_px, egal wie "
      "schnell die welt ist",
      f"{min(rates.values()):.3f}..{max(rates.values()):.3f} px/s "
      f"gegen {LIMIT:.0f}")

# Gegenprobe: OHNE grenze waere derselbe fall ein schmierstreifen.
# (Fein abgetastet, sonst legt der flug je schritt mehr als eine halbe
# faltungsperiode zurueck und laesst sich gar nicht mehr entfalten -- was
# fuer sich schon zeigt, wie absurd schnell das ist.)
lay, moved = slide(1e2, 7.7e3, steps=1200, dt=1 / 600, limit=FAST)
free_rate = moved / 2.0
check(free_rate > 500.0 * LIMIT,
      "gegenprobe: ungebremst raest derselbe flug ueber den schirm",
      f"{free_rate:.3e} px/s gegen grenze {LIMIT:.0f}")

# Ein RAHMENWECHSEL ist kein flug. Bleibt der schluessel gleich, gliten die
# 3.8e8 m von Erde zu Mond bei 1500 px/s ueber acht minuten durchs bild;
# wechselt er, sind sie sofort da. Eine sprunghoehen-schwelle taugt dafuer
# nicht: bei 1e-4 px/m misst dieser wechsel 3.8e4 px je bild, ein vorbeiflug
# am zoomanschlag 8.3e4 -- jede schwelle dazwischen trifft einmal falsch.
MOON = 3.84e8


def jump(key_a, key_b, scale=1e-4, limit=1500.0):
    layer = tracked(scale=scale, limit=limit)
    layer.update(1 / 60, scale, scale, ORIGIN, grid_target=(0.0, 0.0),
                 grid_key=key_a, viewport=VP)
    layer.update(1 / 60, scale, scale, ORIGIN, grid_target=(MOON, 0.0),
                 grid_key=key_b, viewport=VP)
    return layer.anchor_xy()[0]


same = jump("Erde", "Erde")
other = jump("Erde", "Mond")
check(abs(same * 1e-4 - 1500.0 / 60.0) < 1e-6,
      "gleicher bezug: der sprung wird abgefahren, nicht uebernommen",
      f"{same * 1e-4:.3f} px in einem bild (grenze {1500.0 / 60.0:.1f})")
check(abs(background.fold(other - MOON,
                          background.fold_spans(1e-4)[0])) < 1e-6,
      "neuer bezug: er wird sofort uebernommen",
      f"{other:.6e} m gegen ziel {MOON:.6e} m (modulo faltung)")

# Der schwenk muss UNTER der grenze liegen, sonst haengt das gitter beim
# schwenken hinterher -- genau der eindruck, der behoben werden sollte.
PAN_PX_S = 800.0        # camera.move_speed = 1.0 schirmhoehen/s bei 800 px
defaults = BackgroundLayer()
check(defaults.grid_max_speed_px > PAN_PX_S,
      "die Vorgabe liegt ueber der schwenkrate der kamera",
      f"{defaults.grid_max_speed_px:.0f} > {PAN_PX_S:.0f} px/s")

# 0 friert das gitter ein, statt es unbegrenzt laufen zu lassen.
frozen = tracked(limit=0.0)
frozen_path = run(frozen, [(i * 1.0e6, 0.0) for i in range(30)])
check(max(abs(a[0]) for a in frozen_path) < 1e-9,
      "grid_max_speed_px = 0 friert das gitter ein",
      f"max |anker| {max(abs(a[0]) for a in frozen_path):.3e} m")

# Der ZOOM-FIXPUNKT des musters ist die bildmitte. Der anker steht in metern,
# die phase ist also skalenfrei; ein in PIXELN gefuehrter versatz haengt
# dagegen als phase += sqrt(3)*D/sp an der zoomstufe und legt den fixpunkt auf
# -D. Gemessen lag er einmal 4226 px daneben, und ein zoomschritt um Faktor
# 1.5 schob das muster um 2113 px -- mehr als eine bildbreite.
def pattern_origin_px(layer, scale):
    lv = layer.levels(scale, *layer.anchor_xy())
    if not lv:
        return None, None
    top = lv[0]
    return -top.phase_a * top.spacing_px / math.sqrt(3.0), top


def zoom_fixed_point(layer, scale, k=1.02):
    o1, l1 = pattern_origin_px(layer, scale)
    o2, l2 = pattern_origin_px(layer, scale * k)
    if o1 is None or o2 is None or l1.k != l2.k:
        return None
    return (o2 - k * o1) / (1.0 - k)


worst_fp = 0.0
for sc in (1e-6, 1e-3, 1e-2, 1.0):
    lay, _ = slide(sc, 3.0e3, steps=60 * 30, limit=FAST)
    fp = zoom_fixed_point(lay, sc)
    if fp is not None:
        worst_fp = max(worst_fp, abs(fp))
check(worst_fp < 1e-6,
      "der zoom-fixpunkt des musters IST die bildmitte, auf jeder zoomstufe",
      f"groesster versatz {worst_fp:.3e} px")

D = 4226.0
sp = 441.6
o1 = -(math.sqrt(3.0) * D / sp) * sp / math.sqrt(3.0)
o2 = -(math.sqrt(3.0) * D / (sp * 1.02)) * (sp * 1.02) / math.sqrt(3.0)
fp_old = (o2 - 1.02 * o1) / (1.0 - 1.02)
check(abs(fp_old + D) < 1e-6,
      "gegenprobe: mit einem pixel-versatz laege der fixpunkt bei -D",
      f"{fp_old:.1f} px statt 0")


# --------------------------------------------------------------------- §8d
print("\n§8d  die faltung ist eine echte GITTERTRANSLATION")

# Der aufholfehler wird modulo `fold_spans()` gefaltet, damit ein
# rahmenwechsel (bis 1e11 m sprung) nicht ewig braucht. Das darf nur, wenn
# die faltung fuer JEDE sichtbare dekade ein gittervielfaches ist -- sonst
# springt das muster um einen bruchteil einer zelle.
#
# Die x-periode ist `2*ws/sqrt(3)`, nicht `ws`: die knoten stehen bei
# x = q*ws/sqrt(3), und q muss um eine GERADE zahl springen, sonst kippt die
# paritaet (§1). Genau dieses sqrt(3) hat gefehlt.
def phase_set(layer, scale, ax, ay):
    return [(lv.k, lv.phase_a, lv.phase_b)
            for lv in layer.levels(scale, ax, ay)]


shift_ok = True
worst_shift = 0.0
probe = BackgroundLayer()
probe.grid_fade = 1.0
for scale in (1e-7, 1e-6, 3e-5, 1e-3, 1e-1):
    spans = background.fold_spans(scale)
    base = phase_set(probe, scale, 1.234e7, -5.678e7)
    for n, (dx, dy) in ((1, (spans[0], 0.0)), (-2, (0.0, spans[1])),
                        (3, (spans[0], spans[1]))):
        moved = phase_set(probe, scale,
                          1.234e7 + n * dx, -5.678e7 + n * dy)
        if len(moved) != len(base):
            shift_ok = False
            continue
        for (k0, a0, b0), (k1, a1, b1) in zip(base, moved):
            worst_shift = max(worst_shift,
                              abs(background.fold(a1 - a0, 2.0)),
                              abs(background.fold(b1 - b0, 2.0)))
check(shift_ok and worst_shift < 1e-6,
      "eine faltung um fold_spans() aendert auf KEINER sichtbaren dekade "
      "die phase", f"groesste phasenaenderung {worst_shift:.3e} zellen")

# Gegenprobe: die naive faltung modulo 10^k. In x verschiebt sie das muster um
# sqrt(3)*n zellen -- eine irrationale zahl, also nie ein gitterpunkt.
naive_ws = math.pow(background.LEVEL_BASE,
                    math.ceil(math.log10(background.LEVEL_FADE_OUT[1] / 1e-6)))
base = phase_set(probe, 1e-6, 1.234e7, -5.678e7)
naive = phase_set(probe, 1e-6, 1.234e7 + naive_ws, -5.678e7)
worst_naive = max(abs(background.fold(a1 - a0, 2.0))
                  for (_, a0, _), (_, a1, _) in zip(base, naive))
check(worst_naive > 0.1,
      "gegenprobe: modulo 10^k gefaltet springt das muster um einen "
      "bruchteil einer zelle",
      f"{worst_naive:.4f} zellen versatz")


# --------------------------------------------------------------------- §8c
print("\n§8c  rasterzelle -- form und neutralwert")

# Die zellmaske im shader:
#   q = |fract(frag/px) - 0.5| * 2 ;  d = mix(max(q.x,q.y), |q|, round)
#   maske = step(d, 1 - 0.18*round)
# Bei round = 0 muss sie fuer JEDE lage in der zelle 1 sein, sonst bekaeme
# der reine pixelraster loecher.
grid = np.linspace(0.0, 1.0, 41)[:-1]
gx, gy = np.meshgrid(grid, grid)
q = np.stack([np.abs(gx - 0.5) * 2.0, np.abs(gy - 0.5) * 2.0])


def mask(round_amount):
    cheb = np.maximum(q[0], q[1])
    circ = np.hypot(q[0], q[1])
    d = cheb + (circ - cheb) * round_amount
    return (d <= 1.0 - 0.18 * round_amount).astype(float)


m0 = mask(0.0)
check(float(m0.min()) == 1.0, "round=0: die zelle ist vollstaendig gefuellt",
      f"fuellgrad {m0.mean() * 100:.1f} %")
m1 = mask(1.0)
check(0.3 < float(m1.mean()) < 0.85,
      "round=1: ein punkt mit spalt, weder voll noch fast leer",
      f"fuellgrad {m1.mean() * 100:.1f} %")
mid = mask(0.5)
check(m1.mean() < mid.mean() < m0.mean(),
      "der fuellgrad faellt monoton mit der rundung",
      f"{m0.mean() * 100:.0f} % > {mid.mean() * 100:.0f} % > "
      f"{m1.mean() * 100:.0f} %")

bgr = BackgroundLayer()
check(bgr.pixel_round == 1.0,
      "Vorgabe ist die leuchtpunkt-matrix (round = 1)")
# Der fuellgrad-ausgleich im shader haengt an dieser zahl -- laeuft sie weg,
# wird das gitter beim drehen an pixel_round heller oder dunkler.
check(abs(m1.mean() - 0.53) < 0.03,
      "der gemessene fuellgrad deckt sich mit CELL_FILL_ROUND = 0.53",
      f"{m1.mean():.4f}")


# ---------------------------------------------------------------------- §7
print("\n§7  konfiguration -- config.json, layer und ImGui-panel decken sich")

import re as _re

from loader import ConfigLoader

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cl = ConfigLoader()
cl.load()
fresh = BackgroundLayer()
cl.apply_to_background(fresh)
check(cl.unknown_keys == [],
      "kein schluessel des abschnitts bleibt unverbraucht",
      f"{cl.unknown_keys}")

cfg_keys = set(_json.loads(
    _io.open(os.path.join(_here, 'config.json'), encoding='utf-8-sig').read()
)['background'])

# Die laufzeit-zustaende sind bewusst KEINE config-schluessel.
runtime_only = {'star_pan_px', 'grid_anchor_m', 'grid_lag_px',
                'grid_fade', 'time_s', 'star_zoom'}
layer_keys = {k for k in vars(BackgroundLayer())
              if not k.startswith('_')} - runtime_only

check(cfg_keys == layer_keys,
      "config.json und BackgroundLayer haben dieselbe schluesselmenge",
      f"nur config: {sorted(cfg_keys - layer_keys)}, "
      f"nur layer: {sorted(layer_keys - cfg_keys)}")

# Und das ImGui-panel: der Background-block muss jeden schluessel anfassen.
src = _io.open(os.path.join(_here, 'devui.py'), encoding='utf-8').read()
start = src.find('collapsing_header("Background")')
check(start > 0, "devui.py hat einen Background-block")
end = src.find('collapsing_header(', start + 10)
block = src[start:end if end > start else len(src)]
touched = set(_re.findall(r"bg\.([a-z_]+)", block))
touched |= set(_re.findall(r"bg,\s*'([a-z_]+)'", block))
missing = cfg_keys - touched
check(not missing,
      "das ImGui-panel bedient jeden config-schluessel",
      f"fehlend: {sorted(missing)}" if missing else f"{len(cfg_keys)} regler")

# Gegenprobe: der block darf auch nichts anfassen, was es nicht gibt.
unknown = {a for a in touched
           if not hasattr(BackgroundLayer(), a) and a not in ('levels',)}
check(not unknown, "und greift auf kein unbekanntes attribut zu",
      f"{sorted(unknown)}")


print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for failure in FAILURES:
        print(f"  {failure}")
    sys.exit(1)
print("background.py: alle pruefungen bestanden")
