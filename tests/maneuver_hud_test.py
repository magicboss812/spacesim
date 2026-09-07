"""Die vier manoever-plaettchen im navball-raster und der ziehgriff.

Fuenf ebenen:

1. **Lage** -- die plaettchen sitzen im RASTER des navball-blocks: sie
   teilen breite und x-lage mit ORB/ALT/THR/V-S und ueberlappen weder den
   block noch die apsiden-leiste noch einander, in jeder geprueften
   fenstergroesse. Rechneten sie ihre breite selbst, waere es ein zweites
   layout fuer dieselbe flanke.
2. **Sperre** -- der EXECUTE-knopf ist abgeblendet, solange der knoten kein
   delta-v traegt. Ein knoten ohne delta-v ist ein PLATZHALTER, und ein
   knopf, der dann etwas ausloest, waere eine luege.
3. **Wirkung** -- ein klick auf einen schrittknopf und ein zug am griff
   aendern wirklich das delta-v des knotens UND zaehlen plan.version hoch.
   Ohne den zaehler bliebe die gezeichnete linie auf dem alten stand.
4. **Knueppel** -- der griff ist ein RATEN-steuer: die auslenkung bestimmt
   die geschwindigkeit, `update(dt)` integriert sie. Gehalten, ohne die
   maus weiter zu bewegen, laeuft der wert weiter; losgelassen steht er.
5. **Eingabefeld** -- die beiden delta-v-zeilen nehmen ZIFFERN UND PUNKT
   und sonst nichts, schreiben bei jedem anschlag durch, und die tastatur
   gehoert ihnen nur, solange sie gebraucht wird -- sonst verschluckte ein
   klick auf das plaettchen die tasten N / X.

Aufruf: python tests/maneuver_hud_test.py
"""

import math
import os
import sys

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

W, H = 1600, 900
FAILURES = []


def check(condition, label, detail=''):
    status = 'OK  ' if condition else 'FEHL'
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ''))
    if not condition:
        FAILURES.append(label)


pygame.display.init()
pygame.font.init()
pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
gl = moderngl.create_context()
gl.enable(moderngl.BLEND)
gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

from config.loader import ConfigLoader
from physics.vec import G
from physics.world import world as World
from render.renderer import Renderer
from runtime.system_loader import SystemLoader
from ship.camera import Camera
from ship.control import schiffcontrol
from ship.maneuver import (
    ManeuverExecutor,
    ManeuverNode,
    ManeuverPlan,
    ManeuverPreview,
)
from ship.predictor import Predictor
from ui import UIContext, UIRoot, UIState
from ui.core import Rect
from ui import units
from ui.hud import Hud

config = ConfigLoader(None)
config.load()
bodies = SystemLoader("solar_system.json").load()
world = World(G)
world.body = bodies
config.apply_to_world(world)
world.update_planets(0.0)
sim_ship = next(b for b in bodies if b.is_ship)
earth = next(b for b in bodies if b.name == "Erde")
control = schiffcontrol(sim_ship)
config.apply_to_ship_control(control)

mu = world.G * earth.mass
r = float(getattr(earth, 'radius', 6.371e6)) + 4.0e6
ahead = earth.position_at_time(1.0)
behind = earth.position_at_time(-1.0)
sim_ship.position.x = earth.position.x + r
sim_ship.position.y = earth.position.y
sim_ship.velocity.x = (float(ahead.x) - float(behind.x)) / 2.0
sim_ship.velocity.y = (float(ahead.y) - float(behind.y)) / 2.0 + math.sqrt(mu / r)

camera = Camera(None, W, H)
config.apply_to_camera(camera)
camera.follow(sim_ship)
camera.snap_to_targets()
renderer = Renderer(W, H, enable_fxaa=False, ctx=gl)
config.apply_to_renderer(renderer)
predictor = Predictor(recompute_every_update=True, **config.predictor_kwargs())
config.apply_to_predictor(predictor)
predictor.async_compute = False
predictor.initialize(sim_ship, world)
predictor.update(sim_ship, world)

mc = config.maneuver_kwargs()
plan = ManeuverPlan(max_nodes=mc['max_nodes'],
                    min_executable_dv=mc['min_executable_dv'])
preview = ManeuverPreview(max_points=mc['preview_max_points'])
executor = ManeuverExecutor(plan, sim_ship, control,
                            thrust_acc_max=600.0, realtime_warp_max=60.0)
selected = [0]

state = UIState(world.body, initial_reference_index=world.body.index(earth))
ui = UIContext(gl, W, H, ui_scale=renderer.ui_scale)
root = UIRoot(ui)
hud = Hud(root, world, sim_ship, control, camera, renderer, predictor, state,
          maneuver_plan=plan, maneuver_preview=preview,
          maneuver_executor=executor,
          maneuver_selected_get=lambda: selected[0],
          maneuver_selected_set=lambda i: selected.__setitem__(0, int(i)),
          maneuver_dv_step_fine=mc['dv_step_fine'],
          maneuver_dv_step_coarse=mc['dv_step_coarse'],
          maneuver_dv_rate=mc['handle_dv_rate'],
          maneuver_handle_travel_px=mc['handle_travel_px'],
          maneuver_length_get=lambda: preview.length_mult,
          maneuver_length_set=preview.set_length_mult,
          maneuver_length_min=mc['preview_length_mult_min'],
          maneuver_length_max=mc['preview_length_mult_max'])


def frame():
    """Ein voller HUD-frame -- MIT zeichnen.

    Das `root.render()` ist keine zierde: ohne es laeuft `draw()` in diesem
    test nie, und ein undefinierter name dort faellt erst im spiel auf. Ist
    genau einmal passiert (`with_unit`, uebriggeblieben aus einem
    entfernten zweig).
    """
    hud.update()
    root.begin_frame(1.0 / 60.0)
    gl.screen.use()
    root.render()


def click(x, y):
    """Einen vollstaendigen klick auf den widget-baum schicken."""
    press(x, y)
    root.handle_event(pygame.event.Event(
        pygame.MOUSEBUTTONUP, {'pos': (x, y), 'button': 1}))


def press(x, y):
    """Zeiger hin, dann drueckn -- ueber die ECHTE ereigniskette.

    Die bewegung gehoert dazu: `UIRoot` fragt `takes_keyboard` im moment des
    klicks ab, und das plaettchen beantwortet die frage aus seinem
    hover-schluessel. Ohne bewegung bekaeme es nie den fokus, und `update()`
    schloesse das eingabefeld im naechsten frame wieder.
    """
    root.handle_event(pygame.event.Event(
        pygame.MOUSEMOTION, {'pos': (x, y), 'rel': (0, 0), 'buttons': (0, 0, 0)}))
    root.handle_event(pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, {'pos': (x, y), 'button': 1}))


def key(name, char='', mod=0):
    """Einen tastendruck durch den widget-baum schicken.

    Rueckgabe: ob der baum ihn VERBRAUCHT hat. Genau daran haengt, ob ein
    'n' waehrend des tippens einen knoten setzt.
    """
    code = name if isinstance(name, int) else ord(name)
    return root.handle_event(pygame.event.Event(
        pygame.KEYDOWN, {'key': code, 'unicode': char, 'mod': mod}))


frame()

def overlaps(a, b):
    return not (a.right <= b.left or a.left >= b.right
                or a.bottom <= b.top or a.top >= b.bottom)


print("\n1) vier plaettchen, ausgerichtet am navball-raster")
axes = getattr(hud, 'maneuver_axes', None)
nodes = getattr(hud, 'maneuver_nodes', None)
burn = getattr(hud, 'maneuver_burn', None)
plan_block = getattr(hud, 'maneuver_length', None)
plates = [('burn', burn), ('axes', axes), ('plan', plan_block),
          ('nodes', nodes)]
for name, plate in plates:
    check(plate is not None, f"hud.maneuver_* : {name} existiert", "")
    check(plate.rect.w > 0 and plate.rect.h > 0, f"{name}: hat ein rechteck",
          f"{plate.rect.w:.0f}x{plate.rect.h:.0f}")

# DAS IST DIE EIGENTLICHE PRUEFUNG DIESES ABSCHNITTS: breite und x-lage
# kommen aus dem navball, nicht aus eigenen zahlen. Waeren es eigene, stuende
# das plaettchen bei jeder anderen UI-skala oder BOX_H daneben.
left_flank = hud.navball.flank_rect(ui, 'left')
right_flank = hud.navball.flank_rect(ui, 'right')
left_strip = hud.navball.strip_rect(ui, 'left')
right_strip = hud.navball.strip_rect(ui, 'right')
for name, plate, ref in (('burn', burn, left_flank), ('axes', axes, left_strip),
                         ('plan', plan_block, right_flank),
                         ('nodes', nodes, right_strip)):
    check(abs(plate.rect.x - ref[0]) < 0.5 and abs(plate.rect.w - ref[2]) < 0.5,
          f"{name}: teilt x-lage und breite mit seiner flanke",
          f"x {plate.rect.x:.0f}/{ref[0]:.0f}, w {plate.rect.w:.0f}/{ref[2]:.0f}")

check(burn.rect.bottom <= left_flank[1] + 1.0,
      "BURN sitzt UEBER der ORB-flanke",
      f"bottom={burn.rect.bottom:.0f}, flanke={left_flank[1]:.0f}")
check(plan_block.rect.bottom <= right_flank[1] + 1.0,
      "PLAN sitzt UEBER der ALT-flanke", "")
check(axes.rect.top >= left_strip[1] + left_strip[3] - 1.0,
      "die achsen sitzen UNTER dem THR-streifen",
      f"top={axes.rect.top:.0f}, streifen={left_strip[1] + left_strip[3]:.0f}")
check(nodes.rect.top >= right_strip[1] + right_strip[3] - 1.0,
      "die knotenleiste UNTER dem V/S-streifen", "")

print("\n2) lage in drei fenstergroessen -- und KEINE ueberlappung")
# Der eigentliche punkt der aufteilung: drei plaetten um EIN instrument
# herum duerfen weder einander noch navball und rosette beruehren, sonst
# waere die aufteilung nur eine umverteilung des gedraenges.
for width, height in ((1280, 800), (1600, 900), (1920, 1080)):
    root.resize(width, height, ui_scale=renderer.ui_scale)
    frame()
    boxes = [(name, p) for name, p in plates if p.visible]
    if not boxes:
        check(True, f"{width}x{height}: plaetten ausgeblendet (schmale fassung)", "")
        continue
    for name, b in boxes:
        inside = (b.rect.left >= 0 and b.rect.top >= 0
                  and b.rect.right <= width and b.rect.bottom <= height)
        check(inside, f"{width}x{height} {name}: vollstaendig im bild",
              f"({b.rect.left:.0f},{b.rect.top:.0f})-"
              f"({b.rect.right:.0f},{b.rect.bottom:.0f})")
    # Gegen die apsiden-leiste UND die rosette. Die leiste ist nur um eine
    # halbe flankenbreite eingerueckt, ragt also unter beide flanken -- sie
    # ist die schranke, die die hoehe der unteren plaettchen festlegt.
    ix, iy, iw, ih = hud.navball.info_rect(ui)
    info = Rect(ix, iy, iw, ih)
    for name, b in boxes:
        for other_name, other in (('ORBITAL.INFO', info),
                                  ('snaps', hud.snaps.rect)):
            check(not overlaps(b.rect, other),
                  f"{width}x{height} {name}: frei von {other_name}", "")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            check(not overlaps(boxes[i][1].rect, boxes[j][1].rect),
                  f"{width}x{height}: {boxes[i][0]} frei von {boxes[j][0]}", "")
root.resize(W, H, ui_scale=renderer.ui_scale)
frame()
block = burn

print("\n3) SPERRE -- ohne delta-v ist EXECUTE abgeblendet")
check(block.execute_enabled() is False, "ohne knoten: gesperrt", "")
node = ManeuverNode(world.time + 1800.0)
plan.add(node)
frame()
check(len(plan) == 1, "ein knoten liegt vor", "")
check(block.execute_enabled() is False,
      "knoten OHNE delta-v: weiterhin gesperrt", f"dv={node.dv_total}")
node.dv_prograde = 50.0
plan.touch()
frame()
check(block.execute_enabled() is True, "mit delta-v: frei", f"dv={node.dv_total}")
node.dv_prograde = 0.0
plan.touch()
frame()
check(block.execute_enabled() is False, "und wieder gesperrt, wenn es weg ist", "")

print("\n4) WIRKUNG -- die zeile nimmt eine getippte zahl an")
# Die vier pfeilknoepfe je achse sind weg: sie frassen 60 der 124 einheiten
# breite, und wer 1900 m/s einstellen will, klickt 190-mal. Gepruef wird
# ueber die ECHTE ereigniskette, weil genau daran haengt, wem die tastatur
# gerade gehoert.
regions = axes._regions(ui)
before_version = plan.version
x, y, w, h = regions['pro_value']
press(x + w * 0.5, y + h * 0.5)
check(axes.takes_keyboard is True,
      "ueber dem wertfeld nimmt das plaettchen die tastatur", "")
check(root.wants_keyboard is True, "und der baum meldet das nach aussen", "")
check(axes._editing == 'prograde', "der klick oeffnet das feld",
      f"{axes._editing}")
frame()
check(axes._editing == 'prograde', "und es bleibt ueber den frame hinaus offen", "")

for ch in '250.5':
    check(key(ch, ch) is True, f"'{ch}' landet im feld", "")
frame()
check(abs(node.dv_prograde - 250.5) < 1e-9,
      "waehrend des tippens steht der wert schon im knoten",
      f"{node.dv_prograde}")
check(plan.version > before_version, "und plan.version zaehlt hoch", "")

# Buchstaben werden VERSCHLUCKT, nicht durchgereicht -- sonst setzte ein 'n'
# beim tippen einen knoten.
check(key(pygame.K_n, 'n') is True, "ein buchstabe wird verschluckt", "")
check(abs(node.dv_prograde - 250.5) < 1e-9, "und aendert nichts", "")

# Ein ZWEITER punkt ebenso.
key('.', '.')
check(axes._buffer == '250.5', "ein zweiter punkt wird abgelehnt",
      f"{axes._buffer!r}")

key(pygame.K_BACKSPACE)
check(abs(node.dv_prograde - 250.0) < 1e-9,
      "rueckschritt loescht die letzte stelle", f"{node.dv_prograde}")

key(pygame.K_RETURN, '\r')
check(axes._editing is None, "enter schliesst das feld", "")
frame()
check(abs(node.dv_prograde - 250.0) < 1e-9, "der wert steht", "")

print("\n4b) angeklickt und NICHT getippt aendert nichts")
# Ein feld, das beim blossen anklicken auf 0.0 springt, waere eine falle.
x, y, w, h = regions['pro_value']
press(x + w * 0.5, y + h * 0.5)
click(hud.navball.rect.center_x, hud.navball.rect.bottom - 4.0)
frame()
check(abs(node.dv_prograde - 250.0) < 1e-9,
      "der alte wert steht unveraendert", f"{node.dv_prograde}")
check(axes.takes_keyboard is False,
      "und die tastatur gehoert wieder dem spiel", "")

print("\n4c) die RICHTUNG ist die beschriftung, kein tippbares minus")
# Der kleinste tippbare wert ist 0.0. Ein minus gaebe es sonst zweimal --
# einmal als zeichen, einmal als knopf.
sx, sy, sw, sh = regions['pro_sign']
click(sx + sw * 0.5, sy + sh * 0.5)
frame()
check(node.dv_prograde < 0.0, "RET dreht das vorzeichen um",
      f"{node.dv_prograde}")
check(axes.takes_keyboard is False,
      "und ueber dem richtungsknopf bleibt die tastatur beim spiel", "")
click(sx + sw * 0.5, sy + sh * 0.5)
frame()
check(node.dv_prograde > 0.0, "noch einmal gedrueckt: wieder prograde", "")

x, y, w, h = regions['nrm_value']
axes.on_mouse_move(ui, x + w * 0.5, y + h * 0.5)
axes.on_wheel(ui, 0, 1)
check(abs(node.dv_normal - mc['dv_step_fine']) < 1e-9,
      "das rad ueber der zeile stellt einen feinschritt",
      f"{node.dv_normal}")
axes.on_mouse_move(ui, axes.rect.left - 100.0, axes.rect.top - 100.0)

print("\n5) die pips waehlen den knoten")
plan.add(ManeuverNode(world.time + 3600.0, dv_prograde=10.0))
frame()
regions = nodes._regions(ui)
x, y, w, h = regions['pip_1']
click(x + w * 0.5, y + h * 0.5)
frame()
check(selected[0] == 1, "der zweite pip waehlt knoten 1", f"{selected[0]}")
check(hud.telemetry.maneuver_node is plan.nodes[1],
      "die telemetrie zeigt jetzt auf ihn", "")

print("\n6) jede platte verbraucht die maus nur ueber sich selbst")
for name, plate in plates:
    check(plate.hit_test(ui, plate.rect.center_x, plate.rect.center_y) is True,
          f"{name}: ein klick darauf trifft sie", "")
    check(root._pick(plate.rect.center_x, plate.rect.top - 60.0) is not plate,
          f"{name}: ein klick darueber nicht", "")

print("\n7) der ring bekommt einen fuenften marker, sobald scharf ist")
from ui.hud.attitude import _MARKER_ORDER
check(any(entry[0] == 'node' for entry in _MARKER_ORDER),
      "'node' steht in _MARKER_ORDER", f"{[e[0] for e in _MARKER_ORDER]}")
selected[0] = 0
plan.nodes[0].dv_prograde = 60.0
plan.touch()
preview.rebuild(plan, sim_ship, world, predictor, earth,
                executor.a_max_sim(), mc['ramp_seconds'])
armed = executor.arm(world, earth, preview)
check(armed, "der knoten laesst sich scharfschalten", "")
renderer.maneuver_burn_direction = (executor.dir_x, executor.dir_y)
frame()
check('node' in hud.telemetry.marker_headings,
      "die telemetrie liefert seinen kurs",
      f"{sorted(hud.telemetry.marker_headings)}")
check(block.execute_enabled() is True, "und EXECUTE heisst jetzt ABORT", "")
executor.abort('test')
renderer.maneuver_burn_direction = None
frame()

print("\n8) unter dem umbruch verschwinden ALLE VIER plaettchen")
root.resize(700, 640, ui_scale=renderer.ui_scale)
frame()
for name, plate in plates:
    check(plate.visible is False,
          f"{name}: schmale fassung ausgeblendet (die tasten N/X bleiben)",
          f"visible={plate.visible}")
root.resize(W, H, ui_scale=renderer.ui_scale)
frame()
for name, plate in plates:
    check(plate.visible is True, f"{name}: breite fassung wieder da", "")

print("\n9) der ziehgriff faengt NUR seine eigenen griffe")
gizmo = hud.gizmo
check(gizmo is not None, "hud.gizmo existiert", "")
renderer.maneuver_node_hits = [{
    'index': 0, 'sx': 500.0, 'sy': 400.0, 'radius_px': 8.0,
    'handles': [
        {'kind': 'prograde', 'sx': 530.0, 'sy': 400.0, 'radius_px': 7.0,
         'dir_sx': 1.0, 'dir_sy': 0.0},
        {'kind': 'retrograde', 'sx': 470.0, 'sy': 400.0, 'radius_px': 7.0,
         'dir_sx': -1.0, 'dir_sy': 0.0},
        {'kind': 'normal', 'sx': 500.0, 'sy': 370.0, 'radius_px': 7.0,
         'dir_sx': 0.0, 'dir_sy': -1.0},
        {'kind': 'antinormal', 'sx': 500.0, 'sy': 430.0, 'radius_px': 7.0,
         'dir_sx': 0.0, 'dir_sy': 1.0},
    ],
}]
check(gizmo.hit_test(ui, 530.0, 400.0) is True,
      "der prograde-griff wird getroffen", "")
check(gizmo.hit_test(ui, 500.0, 400.0) is True, "der marker ebenso", "")
check(gizmo.hit_test(ui, 500.0, 200.0) is False,
      "eine leere stelle daneben NICHT (sonst blockiert er den kameraschwenk)", "")

print("\n10) der griff ist ein KNUEPPEL: auslenkung = rate, gehalten laeuft er")
# Der eigentliche unterschied zum alten schieberegler. Frueher war der wert
# die absolute zeigerstrecke mal einem faktor -- 500 m/s brauchten 660 px,
# also den halben schirm, und am bildrand war schluss. Jetzt bestimmt die
# auslenkung, WIE SCHNELL der wert laeuft, und `update(dt)` integriert das.
selected[0] = 0
frame()
node0 = plan.nodes[0]
start = float(node0.dv_prograde)
before = plan.version
dt = 1.0 / 60.0

gizmo.on_mouse_down(ui, 530.0, 400.0, 1)
check(renderer.maneuver_drag_active is True,
      "der zug meldet sich beim renderer an", "")
check(renderer.maneuver_drag_curve is False,
      "ein GRIFF-zug baut die schirmkurve NICHT (die kostet je punkt)", "")
handle = renderer.maneuver_drag_handle
check(handle is not None and handle[1] == 'prograde',
      "der renderer weiss, welcher griff gehalten wird", f"{handle}")

# Noch nicht bewegt: totzone, es darf sich nichts aendern.
gizmo.update(ui, dt)
check(abs(float(node0.dv_prograde) - start) < 1e-12,
      "in der ruhelage laeuft NICHTS", f"{float(node0.dv_prograde) - start:.6f}")

# Vollausschlag nach aussen, dann eine sekunde halten.
gizmo.on_mouse_move(ui, 530.0 + mc['handle_travel_px'], 400.0)
check(abs(renderer.maneuver_drag_handle[2] - mc['handle_travel_px']) < 1e-6,
      "der ausschlag wird zum zeichnen gemeldet",
      f"{renderer.maneuver_drag_handle[2]:.1f} px")
for _ in range(60):
    gizmo.update(ui, dt)
moved = float(node0.dv_prograde) - start
check(abs(moved - mc['handle_dv_rate']) < mc['handle_dv_rate'] * 0.02,
      "voller ausschlag, eine sekunde = handle_dv_rate mehr prograde",
      f"{moved:.2f} von {mc['handle_dv_rate']:.0f}")
check(plan.version > before, "und plan.version zaehlt hoch", "")

# Halb ausgelenkt ist deutlich langsamer als halb so schnell -- die
# quadratische kennlinie ist der grund, warum man nahe der ruhelage
# einzelne m/s trifft.
start = float(node0.dv_prograde)
gizmo.on_mouse_move(ui, 530.0 + mc['handle_travel_px'] * 0.5, 400.0)
for _ in range(60):
    gizmo.update(ui, dt)
half = float(node0.dv_prograde) - start
check(0.0 < half < moved * 0.35,
      "halber ausschlag laeuft klar unter der haelfte (quadratisch)",
      f"{half:.2f} gegen {moved * 0.5:.2f} bei linear")

gizmo.on_mouse_up(ui, 530.0, 400.0, 1)
check(renderer.maneuver_drag_active is False, "loslassen meldet ab", "")
check(renderer.maneuver_drag_handle is None, "und nimmt den ausschlag zurueck", "")
stopped = float(node0.dv_prograde)
for _ in range(60):
    gizmo.update(ui, dt)
check(abs(float(node0.dv_prograde) - stopped) < 1e-12,
      "losgelassen steht der wert still", "")

print("\n11) der retrograde-griff zieht dieselbe achse zurueck")
start = float(node0.dv_prograde)
gizmo.on_mouse_down(ui, 470.0, 400.0, 1)
gizmo.on_mouse_move(ui, 470.0 - mc['handle_travel_px'], 400.0)
for _ in range(60):
    gizmo.update(ui, dt)
moved = float(node0.dv_prograde) - start
check(moved < 0.0 and abs(moved + mc['handle_dv_rate']) < mc['handle_dv_rate'] * 0.02,
      "nach aussen ziehen VERMINDERT prograde", f"{moved:.2f}")
gizmo.on_mouse_up(ui, 470.0, 400.0, 1)

print("\n12) den marker verschieben setzt die knotenzeit neu")
renderer.maneuver_curve_screen = np.array(
    [(500.0 + i * 4.0, 400.0, world.time + 100.0 * i) for i in range(20)],
    dtype=np.float64)
before_t = float(node0.t_node)
gizmo.on_mouse_down(ui, 500.0, 400.0, 1)
check(renderer.maneuver_drag_curve is True,
      "DIESER zug baut die schirmkurve schon", "")
gizmo.on_mouse_move(ui, 540.0, 402.0)
check(abs(float(node0.t_node) - (world.time + 100.0 * 10)) < 1e-6,
      "der knoten springt auf die naechste stuetzstelle der linie",
      f"{before_t:.1f} -> {float(node0.t_node):.1f}")
gizmo.on_mouse_up(ui, 540.0, 402.0, 1)
renderer.maneuver_node_hits = []
renderer.maneuver_curve_screen = None

print("\n13) ein knoten in der VERGANGENHEIT faerbt seinen countdown rot")
# In amber -- der farbe jedes anderen brennwertes -- unterscheidet sich
# 'T+00:04:11' von 'T-00:04:11' nur durch ein zeichen. Der zustand hat
# nichts mit den vier bedeutungsfarben zu tun, deshalb palette.danger.
past = ManeuverNode(world.time - 600.0, dv_prograde=25.0)
plan.add(past)
selected[0] = plan.nodes.index(past)
frame()
check(hud.telemetry.maneuver_time_to_node < 0.0,
      "die telemetrie meldet eine negative restzeit",
      f"{hud.telemetry.maneuver_time_to_node:.0f} s")
check(units.countdown(hud.telemetry.maneuver_time_to_node).startswith('T+'),
      "und der text traegt 'T+'",
      units.countdown(hud.telemetry.maneuver_time_to_node))
recorded = []
real_draw = ui.text.draw


def spy(text, x, y, **kwargs):
    recorded.append((text, kwargs.get('color')))
    return real_draw(text, x, y, **kwargs)


ui.text.draw = spy
burn.draw(ui)
ui.text.draw = real_draw
colors = [c for t, c in recorded if str(t).startswith('T+')]
check(bool(colors), "der countdown wird gezeichnet", f"{len(colors)}")
if colors:
    check(colors[0] == ui.theme.palette.danger,
          "und zwar in palette.danger", f"{colors[0]}")
    check(colors[0] != ui.theme.palette.node,
          "also NICHT im amber der uebrigen brennwerte", "")
plan.remove(past)
selected[0] = 0
frame()

print("\n14) die vorschau hat ihren EIGENEN reichweiten-regler")
# Zwei fragen, zwei regler: die vorhersagelinie will aufloesung (man sieht,
# wo das schiff GLEICH ist), der plan will weite (wo ein knoten den vierten
# umlauf hinlegt). An einem regler haengend muesste man fuer weite immer
# aufloesung mitkaufen.
slider = getattr(hud, 'maneuver_length', None)
check(slider is not None, "hud.maneuver_length existiert", "")
check(slider is not hud.horizon,
      "und ist NICHT der vorhersage-horizont unten links", "")
check(slider.rect.w == right_flank[2],
      "er ist so breit wie die ALT-flanke, unter der er sitzt",
      f"{slider.rect.w:.0f} gegen {right_flank[2]:.0f}")
start_mult = preview.length_mult
slider.on_wheel(ui, 0, 1)
check(preview.length_mult > start_mult,
      "das rad verlaengert die vorschau",
      f"{start_mult:g} -> {preview.length_mult:g}")
for _ in range(400):
    slider.on_wheel(ui, 0, 1)
check(abs(preview.length_mult - mc['preview_length_mult_max']) < 1e-9,
      "und klemmt an der obergrenze", f"{preview.length_mult:g}")
for _ in range(400):
    slider.on_wheel(ui, 0, -1)
check(abs(preview.length_mult - mc['preview_length_mult_min']) < 1e-9,
      "wie an der untergrenze", f"{preview.length_mult:g}")
preview.set_length_mult(start_mult)

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for failure in FAILURES:
        print(f"  {failure}")
    pygame.quit()
    sys.exit(1)
print("ui/hud/maneuver: alle pruefungen bestanden")
pygame.quit()
