"""Regressionstest der prozeduralen koerper-optik (D2).

Die zeichnung eines koerpers ist GEOMETRIE, keine textur -- und genau diese
zusage ist es, die hier gemessen wird, nicht nur die abwesenheit von
abstuerzen. Wie in `ui_render_test.py` laeuft alles durch den echten
zeichenpfad (`Renderer.render`) und wird am framebuffer nachgemessen.

Geprueft wird:
  1. Der bau ist deterministisch und haengt am seed
  2. Alles liegt im einheitskreis, die rueckseite ist schon weggeworfen
  3. Zeichenreihenfolge: gitternetz unter den fuellungen
  4. `expand_segments`: sechs vertices je segment, richtung normiert
  5. Linienbreite ist eine BILDSCHIRM-groesse, keine weltgroesse
  6. Die beleuchtung ist dynamisch und folgt der lichtquelle
  7. Unter der schwelle aendert sich kein einziges pixel
  8. Gebaut wird einmal, nicht pro frame
  9. Die detailleiter haelt die facettengroesse in pixeln fest
 10. Seed aus dem namen, ueberschreibbar per `style_seed`

Aufruf: python tests/body_style_test.py
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SDL_WINDOWS_DPI_AWARENESS', 'permonitorv2')

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import moderngl
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL

import body_style

W, H = 900, 700
FAILURES = []


def check(condition, label, detail=''):
    if condition:
        print(f"  ok   {label}{(' -- ' + detail) if detail else ''}")
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label} -- {detail}")


# =====================================================================
print("\n1. Der bau ist deterministisch und haengt am seed")
# =====================================================================

BLAU = (0x44, 0x88, 0xff)
a = body_style.build_planet_style(4711, color=BLAU, mode='bands', shape='nested')
b = body_style.build_planet_style(4711, color=BLAU, mode='bands', shape='nested')
c = body_style.build_planet_style(4712, color=BLAU, mode='bands', shape='nested')

check(np.array_equal(a.tri, b.tri) and np.array_equal(a.seg, b.seg),
      "derselbe seed liefert bit-identische geometrie",
      f"{a.tri.shape} / {a.seg.shape}")
check(a.seg.shape != c.seg.shape or not np.array_equal(a.seg, c.seg),
      "ein anderer seed liefert eine andere zeichnung",
      f"{a.segment_count} gegen {c.segment_count} segmente")

rot = body_style.build_planet_style(4711, color=(0xff, 0x40, 0x20),
                                    mode='bands', shape='nested')
check(np.array_equal(a.seg[:, :7], rot.seg[:, :7]),
      "die farbe aendert die geometrie nicht, nur die farbspalten")
check(not np.array_equal(a.seg[:, 7:10], rot.seg[:, 7:10]),
      "...und die farbspalten aendert sie wirklich",
      f"{a.seg[0, 7:10]} gegen {rot.seg[0, 7:10]}")

# Ohne diese gegenprobe koennte oben alles bestehen, weil beide leer sind.
check(a.segment_count > 200 and a.triangle_count > 50,
      "die zeichnung ist nicht leer",
      f"{a.triangle_count} dreiecke, {a.segment_count} segmente")

# =====================================================================
print("\n2. Alles liegt im einheitskreis, die rueckseite ist schon weg")
# =====================================================================

radius_tri = np.hypot(a.tri[:, 0], a.tri[:, 1]).max()
radius_seg = max(np.hypot(a.seg[:, 0], a.seg[:, 1]).max(),
                 np.hypot(a.seg[:, 2], a.seg[:, 3]).max())
check(radius_tri <= 1.0 + 1e-5 and radius_seg <= 1.0 + 1e-5,
      "kein punkt ragt aus dem einheitskreis",
      f"dreiecke {radius_tri:.6f}, segmente {radius_seg:.6f}")

normals = np.linalg.norm(a.tri[:, 2:5], axis=1)
check(abs(normals.max() - 1.0) < 1e-4 and abs(normals.min() - 1.0) < 1e-4,
      "die normalen sind normiert", f"{normals.min():.6f} .. {normals.max():.6f}")
check(a.tri[:, 4].min() > 0.0,
      "keine fuellung auf der rueckseite -- die wird beim BAUEN verworfen, "
      "nicht pro frame", f"kleinstes nz {a.tri[:, 4].min():.4f}")

# =====================================================================
print("\n3. Zeichenreihenfolge: gitternetz unter den fuellungen")
# =====================================================================

under = a.seg[:a.under_segments]
over = a.seg[a.under_segments:]
check(a.under_segments > 0 and over.shape[0] > 0,
      "beide gruppen sind belegt",
      f"{a.under_segments} darunter, {over.shape[0]} darueber")
check(float(under[:, 12].max()) <= 0.6 + 1e-6,
      "unter den fuellungen liegt nur das duenne gitternetz",
      f"breiteste linie {under[:, 12].max():.2f} px")
check(float(over[:, 12].max()) >= 2.0 - 1e-6,
      "die konturen (2.0 px) liegen darueber",
      f"breiteste linie {over[:, 12].max():.2f} px")

# =====================================================================
print("\n4. expand_segments: sechs vertices je segment, richtung normiert")
# =====================================================================

verts = body_style.expand_segments(a.seg)
check(verts.shape[0] == a.segment_count * 6
      and verts.shape[1] == body_style.VERT_COLUMNS,
      "sechs vertices je segment", f"{verts.shape}")
dir_len = np.hypot(verts[:, 10], verts[:, 11])
check(abs(dir_len.max() - 1.0) < 1e-5 and abs(dir_len.min() - 1.0) < 1e-5,
      "die segmentrichtung ist normiert (der shader rechnet damit weiter)",
      f"{dir_len.min():.6f} .. {dir_len.max():.6f}")
check(set(np.unique(verts[:, 12]).tolist()) == {-1.0, 1.0}
      and set(np.unique(verts[:, 13]).tolist()) == {-1.0, 1.0},
      "seite und kappe sind reine vorzeichen")
entartet = np.vstack([a.seg[:1].copy(), a.seg[:1].copy()])
entartet[0, 2:4] = entartet[0, 0:2]          # laenge exakt null
check(body_style.expand_segments(entartet).shape[0] == 6,
      "entartete segmente fallen weg statt NaN zu erzeugen")

# =====================================================================
print("\n5.-10. gegen echte pixel")
# =====================================================================

pygame.display.init()
pygame.font.init()
pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL, vsync=0)
gl = moderngl.create_context()

from camera import Camera
from loader import ConfigLoader, SystemLoader
from rendering import Renderer
from vec import Vec2
from world import world as World

config = ConfigLoader()
world = World(float(config.get('physics.gravitational_constant', 6.6730831e-11)))
world.body = SystemLoader(config.get('simulation.system_file',
                                     'solar_system.json')).load()
config.apply_to_world(world)
world.update_planets(0.0)

renderer = Renderer(W, H, enable_fxaa=False, ctx=gl)
config.apply_to_renderer(renderer)
camera = Camera(None, W, H)
config.apply_to_camera(camera)

ERDE = next(b for b in world.body if b.name == 'Erde')
SONNE = next(b for b in world.body if b.name == 'Sonne')
SONNE_HOME = Vec2(float(SONNE.position.x), float(SONNE.position.y))


def look_at(subject, radius_px):
    camera.scale = float(radius_px) / float(subject.radius)
    camera.target_scale = camera.scale
    camera.target = None
    camera.position = Vec2(float(subject.position.x), float(subject.position.y))
    camera.target_position = Vec2(camera.position.x, camera.position.y)


def read_pixels():
    data = gl.screen.read(viewport=(0, 0, W, H), components=3, dtype='f1')
    return np.frombuffer(data, dtype=np.uint8).reshape(H, W, 3)[::-1]


def draw(settle=True):
    """Einen frame zeichnen -- und, wenn noetig, auf den bau warten.

    Der bau laeuft nebenlaeufig, deshalb waere ein einzelner frame ein
    wettlauf. Gewartet wird auf das ERGEBNIS, nicht auf eine feste zeit.
    """
    for _ in range(600):
        renderer.render(world.body, camera, sim_time=0.0)
        if not settle or not renderer.body_vector_style:
            break
        if not renderer._body_style_jobs and renderer.debug_info.get('bodies_vector'):
            break
        time.sleep(0.002)
    return read_pixels()


# ---------------------------------------------------------------------
print("\n5. Linienbreite ist eine BILDSCHIRM-groesse, keine weltgroesse")
# ---------------------------------------------------------------------
# Gemessen an EINEM segment, das durch denselben puffer und denselben
# shader laeuft wie die echte zeichnung. Ueber die volle zeichnung waere
# die messung wertlos: dort tragen die flaechenfuellungen den groessten
# teil der tinte, und die skalieren natuerlich mit dem koerper.
#
# Waere die breite ein weltmass, verdoppelte sie sich mit dem radius.

def probe_thickness(radius_px, width_px):
    seg = np.asarray([[-0.6, 0.0, 0.6, 0.0,
                       0.0, 0.0, 1.0,
                       1.0, 1.0, 1.0,
                       1.0, 1.0, float(width_px)]], dtype=np.float32)
    probe = body_style.PlanetStyle(
        np.zeros((0, body_style.TRI_COLUMNS), dtype=np.float32),
        seg, 0, 0, 'bands', 'nested', 'nested', 2, 2)
    entry = renderer._upload_body_style(probe)
    gl.screen.use()
    gl.clear(0.0, 0.0, 0.0, 1.0)
    renderer._draw_body_vector(entry, W / 2.0, H / 2.0, float(radius_px),
                               (0.0, 0.0, 1.0), 1.0, 1.0)
    column = read_pixels()[:, W // 2].max(axis=1)
    for buffer in entry['buffers']:
        buffer.release()
    return int((column > 40).sum())


for width in (0.6, 3.0):
    thin = probe_thickness(80.0, width)
    thick = probe_thickness(320.0, width)
    check(thin == thick and thin > 0,
          f"eine {width} px breite linie bleibt bei vierfachem radius gleich dick",
          f"{thin} px bei radius 80, {thick} px bei radius 320")

check(probe_thickness(80.0, 3.0) > probe_thickness(80.0, 0.6),
      "...und eine breitere linie ist wirklich breiter (sonst misst das nichts)",
      f"{probe_thickness(80.0, 3.0)} px gegen {probe_thickness(80.0, 0.6)} px")

renderer.body_vector_detail = 'medium'   # eine feste stufe fuer den rest

# ---------------------------------------------------------------------
print("\n6. Die beleuchtung ist dynamisch und folgt der lichtquelle")
# ---------------------------------------------------------------------


def half_brightness(sun_offset_x):
    SONNE.position = Vec2(float(ERDE.position.x) + sun_offset_x,
                          float(ERDE.position.y))
    look_at(ERDE, 220.0)
    frame = draw().astype(np.float64).mean(axis=2)
    ys, xs = np.mgrid[0:H, 0:W]
    disc = ((xs - W / 2.0) ** 2 + (ys - H / 2.0) ** 2) < (220.0 * 0.9) ** 2
    left = disc & (xs < W / 2.0)
    right = disc & (xs > W / 2.0)
    return frame[left].mean(), frame[right].mean()


links_l, links_r = half_brightness(-2.0e11)
rechts_l, rechts_r = half_brightness(+2.0e11)
check(links_l > links_r * 1.6,
      "steht der stern links, ist die linke haelfte klar heller",
      f"{links_l:.1f} gegen {links_r:.1f}")
check(rechts_r > rechts_l * 1.6,
      "steht er rechts, kehrt es sich um",
      f"{rechts_l:.1f} gegen {rechts_r:.1f}")
check(abs(links_l - rechts_r) < 0.25 * max(links_l, rechts_r),
      "die beiden faelle sind spiegelbilder, nicht zwei verschiedene sachen",
      f"{links_l:.1f} gegen {rechts_r:.1f}")
SONNE.position = Vec2(SONNE_HOME.x, SONNE_HOME.y)

# ---------------------------------------------------------------------
print("\n7. Unter der schwelle aendert sich kein einziges pixel")
# ---------------------------------------------------------------------
# Die zusage des einblendens: unterhalb von `body_vector_min_radius_px`
# gibt es weder zeichnung noch glimmen, der koerper sieht aus wie vorher.

klein = float(renderer.body_vector_min_radius_px) - 1.0
look_at(ERDE, klein)
renderer.body_vector_style = True
mit = draw(settle=False)
renderer.body_vector_style = False
ohne = draw(settle=False)
renderer.body_vector_style = True
check(np.array_equal(mit, ohne),
      f"bei {klein:.0f} px radius sind beide bilder bit-identisch",
      f"groesster unterschied {int(np.abs(mit.astype(int) - ohne.astype(int)).max())}")

look_at(ERDE, 200.0)
mit = draw()
renderer.body_vector_style = False
ohne = draw(settle=False)
renderer.body_vector_style = True
check(not np.array_equal(mit, ohne),
      "...bei 200 px dagegen nicht -- sonst pruefte der test oben nichts",
      f"groesster unterschied {int(np.abs(mit.astype(int) - ohne.astype(int)).max())}")

# ---------------------------------------------------------------------
print("\n8. Gebaut wird einmal, nicht pro frame")
# ---------------------------------------------------------------------

look_at(ERDE, 200.0)
draw()
belegt = len(renderer._body_style_gpu)
t0 = time.perf_counter()
for _ in range(30):
    renderer.render(world.body, camera, sim_time=0.0)
je_frame = (time.perf_counter() - t0) / 30.0 * 1000.0
check(len(renderer._body_style_gpu) == belegt and not renderer._body_style_jobs,
      "30 weitere frames bauen nichts nach",
      f"{belegt} zeichnungen im speicher")
check(je_frame < 4.0,
      "und kosten entsprechend wenig", f"{je_frame:.2f} ms je frame")
check(renderer.debug_info.get('bodies_vector', 0) >= 1,
      "der koerper wird dabei wirklich als vektor gezeichnet")

# ---------------------------------------------------------------------
print("\n9. Die detailleiter haelt die facettengroesse in pixeln fest")
# ---------------------------------------------------------------------

renderer.body_vector_detail = None


def facet_px(radius):
    layers = renderer._body_detail_levels(radius)
    total = sum(weight for _level, weight in layers)
    facet = sum(body_style.FACET_FRACTION[level] * radius * weight
                for level, weight in layers)
    return layers, total, facet


# Innerhalb der leiter bleibt die facette in der naehe des ziels.
for radius in (40.0, 80.0, 160.0, 320.0):
    layers, total, facet = facet_px(radius)
    ok = abs(total - 1.0) < 1e-6 and 1 <= len(layers) <= 2
    check(ok and 10.0 <= facet <= 28.0,
          f"bei {radius:4.0f} px radius: {[(l, round(wgt, 2)) for l, wgt in layers]}",
          f"facette ~{facet:.1f} px, summe der gewichte {total:.3f}")

# Ausserhalb ist die leiter zu ende, und das ist kein fehler, sondern die
# aussage: unter der groebsten und ueber der feinsten stufe gibt es nichts
# mehr zu waehlen. Geprueft wird, dass sie DORT anschlaegt und nicht schon
# vorher -- sonst waere die leiter in wahrheit einstufig.
for radius, expected in ((15.0, 'coarse'), (900.0, 'fine')):
    layers, total, facet = facet_px(radius)
    check(layers == ((expected, 1.0),),
          f"bei {radius:4.0f} px schlaegt die leiter an ihrem ende an",
          f"{expected}, facette ~{facet:.1f} px statt 14")

renderer.body_vector_detail = 'medium'
check(renderer._body_detail_levels(15.0) == (('medium', 1.0),),
      "eine erzwungene stufe schaltet die leiter ab")
renderer.body_vector_detail = None

# ---------------------------------------------------------------------
print("\n10. Seed aus dem namen, ueberschreibbar per style_seed")
# ---------------------------------------------------------------------

check(body_style.seed_from_name('Erde') == body_style.seed_from_name('Erde')
      and body_style.seed_from_name('Erde') != body_style.seed_from_name('Mars'),
      "der name-seed ist stabil und unterscheidet koerper",
      f"Erde {body_style.seed_from_name('Erde')}, "
      f"Mars {body_style.seed_from_name('Mars')}")

key_default = renderer._body_style_key(ERDE, 'medium')
ERDE.style_seed = 123456
key_forced = renderer._body_style_key(ERDE, 'medium')
ERDE.style_seed = None
check(key_default[0] == body_style.seed_from_name('Erde')
      and key_forced[0] == 123456,
      "style_seed aus der JSON schlaegt den namen",
      f"{key_default[0]} -> {key_forced[0]}")

ERDE.style_shape = 'diamond'
key_shape = renderer._body_style_key(ERDE, 'medium')
ERDE.style_shape = None
check(key_shape[2] == 'diamond' and key_default[2] == body_style.DEFAULT_SHAPE,
      "style_shape ebenso -- und der standard ist 'nested'",
      f"{key_default[2]!r} -> {key_shape[2]!r}")

check(body_style.DEFAULT_MODE == 'bands'
      and body_style.DEFAULT_SHAPE == 'nested',
      "der gewuenschte standard ist 'bands - nested'")

# =====================================================================
pygame.quit()
if FAILURES:
    print(f"\nkoerper-optik: {len(FAILURES)} pruefung(en) fehlgeschlagen")
    for line in FAILURES:
        print('  -', line)
    sys.exit(1)
print("\nkoerper-optik: alle pruefungen bestanden")
