"""Ein echter frame des SPIELS (welt + HUD) als PNG.

Faehrt `runtime/bootstrap.py` hoch, laeuft N frames ohne ereignisse und liest
den framebuffer VOR present(), also genau das, was im fenster stuende.

    python tools/game_shot.py [breite hoehe] [-o datei.png] [--frames N]
                              [--node PRO,NRM] [--zoom SCALE]
                              [--hide all|KEY,KEY,...] [--dv PRO]
                              [--horizon METRES] [--focus NAME]
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SPACESIM = os.path.dirname(HERE)
sys.path.insert(0, SPACESIM)
os.chdir(SPACESIM)
os.environ.setdefault('SDL_WINDOWS_DPI_AWARENESS', 'permonitorv2')

parser = argparse.ArgumentParser()
parser.add_argument('width', nargs='?', type=int, default=1920)
parser.add_argument('height', nargs='?', type=int, default=1080)
parser.add_argument('-o', '--out', default=None)
parser.add_argument('--frames', type=int, default=90)
parser.add_argument('--node', default=None,
                    help='manoeverknoten setzen: "prograde,normal" in m/s')
parser.add_argument('--node-at', type=float, default=0.25,
                    help='wo auf dem horizont der knoten sitzt (0..1)')
parser.add_argument('--zoom', type=float, default=None,
                    help='kamera-massstab in px/m')
parser.add_argument('--hide', default=None,
                    help='nicht zeichnen (dev-UI "HUD visibility"): "all" '
                         'oder schluessel aus ui/hud/visibility.py, z. B. '
                         '"navball,apsis_block"')
parser.add_argument('--dv', type=float, default=None,
                    help='schiff vor dem ersten frame um PRO m/s prograde '
                         '(relativ zum bezugskoerper) anstossen -- fuer eine '
                         'exzentrische bahn mit einem Ap und einem Pe')
parser.add_argument('--horizon', type=float, default=None,
                    help='gezeichnete laenge der vorhersagelinie in metern '
                         '(predictor.set_display_length) -- z. B. genau ein '
                         'umlauf, damit sich die apsiden nicht stapeln')
parser.add_argument('--focus', default=None,
                    help='kamera auf diesen koerper (name aus '
                         'solar_system.json), z. B. "Jupiter"')
args = parser.parse_args()

W, H = args.width, args.height

with open(os.path.join(SPACESIM, 'config', 'config.json'), 'r',
          encoding='utf-8-sig') as fh:
    cfg = json.load(fh)
cfg['window']['width'] = W
cfg['window']['height'] = H
cfg['window']['vsync'] = False
cfg['debug']['print_frame_timings'] = False
cfg['debug']['print_loader_info'] = False
tmp_cfg = os.path.join(tempfile.gettempdir(), 'spacesim_shot_config.json')
with open(tmp_cfg, 'w', encoding='utf-8') as fh:
    json.dump(cfg, fh)
os.environ['SPACESIM_CONFIG'] = tmp_cfg

import numpy as np
import pygame

from runtime.bootstrap import build_app, load_config
from ship.maneuver import ManeuverNode

app = build_app(load_config())
gl = app.window.ctx
frame_dt = 1.0 / 60.0
_wall = [0.0]

if args.hide:
    vis = app.hud_visibility
    if args.hide.strip().lower() == 'all':
        vis.set_all(False)
    else:
        for key in args.hide.split(','):
            vis.set(key.strip(), False)

if args.dv and app.ship is not None:
    import math
    ref = app.ui_state.reference_body
    rvx = rvy = 0.0
    if ref is not None and hasattr(ref, 'position_at_time'):
        t = app.world.time
        a, b = ref.position_at_time(t - 1.0), ref.position_at_time(t + 1.0)
        rvx, rvy = (b.x - a.x) * 0.5, (b.y - a.y) * 0.5
    vx = app.ship.velocity.x - rvx
    vy = app.ship.velocity.y - rvy
    n = math.hypot(vx, vy) or 1.0
    app.ship.velocity.x += vx / n * args.dv
    app.ship.velocity.y += vy / n * args.dv

if args.horizon:
    app.predictor.set_display_length(float(args.horizon))

if args.focus:
    _target = next((b for b in app.world.body
                    if getattr(b, 'name', '') == args.focus), None)
    if _target is None:
        raise SystemExit(f'--focus: kein koerper {args.focus!r}')
    app.camera.follow(_target)
    app.camera.position = _target.position.copy()
    app.camera.target_position = _target.position.copy()

if args.zoom is not None:
    app.camera.scale = float(args.zoom)
    app.camera.target_scale = float(args.zoom)


def one_frame():
    pygame.event.pump()
    if app.hud is not None:
        app.hud.update()
    app.ui_root.begin_frame(frame_dt)
    # devui wird BEWUSST uebersprungen: ohne die passende
    # new_frame/render-paarung wirft ImGui beim zweiten frame ein IM_ASSERT,
    # und fuer einen abzug wird es nicht gebraucht.
    app.world.step(app.camera.sim_dt * app.tick_rate * frame_dt, app.max_substep)
    app.camera.update(frame_dt, ui_wants_keyboard=False)
    if app.predictor.num_points > 0 and app.ship is not None:
        app.predictor.set_hold(not app.thrust_allowed())
        app.predictor.set_view_scale(app.camera.target_scale)
        app.predictor.update(app.ship, app.world)
    if app.maneuver_preview is not None and app.ship is not None:
        app.maneuver_preview.maybe_rebuild(
            app.maneuver_plan, app.ship, app.world, app.predictor,
            app.ui_state.reference_body, app.maneuver_executor.a_max_sim(),
            app.maneuver_config['ramp_seconds'], now_wall=_wall[0])
        # Eine ECHT laufende uhr -- mit einer festen zahl greift der
        # mindestabstand der vorschau (preview_min_interval_s) und sie
        # rechnet nach dem ersten mal nie wieder neu.
        _wall[0] += frame_dt
        # Fuer einen ABZUG wird gewartet: die vorschau rechnet nebenher und
        # waere im letzten bild sonst je nach zufall einen auftrag alt.
        app.maneuver_preview.wait(5.0)
    app.renderer.render(
        app.world.body, app.camera, app.predictor.get_points(),
        predictor=app.predictor, sim_time=app.world.time,
        reference_body=app.ui_state.reference_body,
        ship_control=app.ship_control, real_dt=frame_dt,
        selected_body=app.ui_state.selected_body,
    )
    app.ui_root.render()


# Ein paar frames, damit die vorhersagelinie steht, dann erst den knoten
# setzen -- ohne linie hat er keine zeitachse.
for _ in range(20):
    one_frame()
    app.renderer.present()

if args.node:
    pro, _, nrm = args.node.partition(',')
    points = app.predictor.get_points()
    t0 = float(points[0, 2])
    t_end = float(points[len(points) - 1, 2])
    app.maneuver_plan.add(ManeuverNode(
        t0 + (t_end - t0) * args.node_at,
        dv_prograde=float(pro or 0.0),
        dv_normal=float(nrm or 0.0),
    ))

for i in range(max(1, args.frames)):
    one_frame()
    if i < args.frames - 1:
        app.renderer.present()

data = gl.screen.read(viewport=(0, 0, W, H), components=3, dtype='f1')
frame = np.frombuffer(data, dtype=np.uint8).reshape(H, W, 3)[::-1]
out = args.out or os.path.join(
    os.path.dirname(SPACESIM), 'screenshots for debugging', 'game_current.png')
pygame.image.save(pygame.surfarray.make_surface(frame.transpose(1, 0, 2)), out)
print('geschrieben:', out, f'({W}x{H})')
app.maneuver_preview.shutdown()
app.window.close()
