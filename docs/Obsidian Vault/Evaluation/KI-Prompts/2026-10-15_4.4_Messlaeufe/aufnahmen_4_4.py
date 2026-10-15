"""Aufnahmen zu Abschnitt 4.4 (KI-erstellt, 15.10.2026).

Stellt das Spiel auf einen gespeicherten Zustand aus den Messläufen
(Zeit, Startwinkel von Erde/Jupiter/Saturn, Ort und Geschwindigkeit des
Schiffs), wählt Bezugskörper, Kamera und Länge der Vorhersagelinie, blendet
das HUD bis auf die Apsis-Marken und deren angehefteten Zettel aus und zeichnet
den Frame über runtime/screenshot.py in 8K neu (7680x4320, Zuschnitt in der
Breite wie im Spiel).

Aufruf (Linux ohne Bildschirm):
    xvfb-run -a -s "-screen 0 1920x1080x24" \\
        python aufnahmen_4_4.py <spacesim-repo> <auftrag.json>
auftrag.json: liste von aufnahmen, siehe zustaende_4_4.py.
"""
import json
import math
import os
import sys
import tempfile

SP = os.path.abspath(sys.argv[1])
JOBS = json.load(open(sys.argv[2]))
sys.path.insert(0, SP)
os.chdir(SP)

W, H = 2560, 1440
with open(os.path.join(SP, 'config', 'config.json'), 'r', encoding='utf-8-sig') as fh:
    cfg = json.load(fh)
cfg['window']['width'] = W
cfg['window']['height'] = H
cfg['window']['vsync'] = False
cfg['debug']['print_frame_timings'] = False
cfg['debug']['print_loader_info'] = False
tmp_cfg = os.path.join(tempfile.gettempdir(), 'spacesim_4_4_config.json')
with open(tmp_cfg, 'w', encoding='utf-8') as fh:
    json.dump(cfg, fh)
os.environ['SPACESIM_CONFIG'] = tmp_cfg

import pygame                                        # noqa: E402
from runtime.bootstrap import build_app, load_config  # noqa: E402
from physics.vec import Vec2                         # noqa: E402

app = build_app(load_config())
frame_dt = 1.0 / 60.0
_wall = [0.0]
bodies = {b.name: b for b in app.world.body}
ship = app.ship
vis = app.hud_visibility
vis.set_all(False)
for key in ('apsis_tooltip', 'apsis_markers', 'body_labels'):
    try:
        vis.set(key, True)
    except Exception as exc:          # noqa: BLE001
        print('sichtbarkeit', key, exc)


def set_state(job):
    w = app.world
    for n in ('Erde', 'Jupiter', 'Saturn'):
        bodies[n].eccentricity = 0.0
    for n, th in job['theta0'].items():
        bodies[n].theta = th
        bodies[n]._kepler_ref_theta = th
        bodies[n]._kepler_ref_time = 0.0
    # alle anderen koerper auf ihre startwerte aus solar_system.json
    for b in w.body:
        if b.is_ship or not b.scripted_orbit or b.name in job['theta0']:
            continue
        b.theta = b._theta0_saved
        b._kepler_ref_theta = b._theta0_saved
        b._kepler_ref_time = 0.0
    w.time = 0.0
    w.update_planets(0.0)
    t = float(job['time'])
    w.time = t
    w.update_planets(t)
    ship.position = Vec2(*job['ship_pos'])
    ship.velocity = Vec2(*job['ship_vel'])


for b in app.world.body:
    b._theta0_saved = b.theta


def one_frame():
    pygame.event.pump()
    if app.hud is not None:
        app.hud.update()
    app.ui_root.begin_frame(frame_dt)
    app.camera.update(frame_dt, ui_wants_keyboard=False)
    if app.predictor.num_points > 0 and ship is not None:
        app.predictor.set_hold(False)
        app.predictor.set_view_scale(app.camera.target_scale)
        app.predictor.update(ship, app.world)
    app.renderer.render(
        app.world.body, app.camera, app.predictor.get_points(),
        predictor=app.predictor, sim_time=app.world.time,
        reference_body=app.ui_state.reference_body,
        ship_control=app.ship_control, real_dt=frame_dt,
        selected_body=app.ui_state.selected_body,
    )
    app.ui_root.render()


def draw_scaled():
    app.renderer.render(
        app.world.body, app.camera, app.predictor.get_points(),
        predictor=app.predictor, sim_time=app.world.time,
        reference_body=app.ui_state.reference_body,
        ship_control=app.ship_control, real_dt=0.0,
        selected_body=app.ui_state.selected_body,
    )
    app.ui_root.render()


app.camera.sim_dt = 0.0
results = []
for job in JOBS:
    set_state(job)
    ref = bodies[job['ref']]
    app.ui_state.set_reference_index(app.world.body.index(ref))
    focus = bodies[job['focus']] if job.get('focus') else ship
    app.camera.follow(focus)
    app.camera.position = focus.position.copy()
    app.camera.target_position = focus.position.copy()
    app.camera.scale = float(job['zoom'])
    app.camera.target_scale = float(job['zoom'])
    app.predictor.reset()
    if job.get('horizon'):
        app.horizon.set_mult(float(job['horizon']) / app.horizon.base_length)
        app.horizon.apply(app.predictor, 0.0)
    for _ in range(int(job.get('frames', 90))):
        if job.get('horizon'):
            app.horizon.apply(app.predictor, 0.0)
        one_frame()
        app.renderer.present()
    pinned = None
    want = job.get('pin')                     # 'Pe' oder 'Ap'
    if want:
        hits = list(getattr(app.renderer, 'apsis_marker_hits', []) or [])
        cands = []
        for hit in hits:
            d = hit if isinstance(hit, dict) else getattr(hit, '__dict__', {})
            cands.append(d)
        from ui.hud.apsis_tooltip import _hit_dict
        ms = [m for m in (_hit_dict(h) for h in hits) if m is not None]
        ms = [m for m in ms if (m['apoapsis'] if want == 'Ap' else not m['apoapsis'])]
        if ms:
            ms.sort(key=lambda m: m['t_abs'])
            m = ms[0]
            app.ui_root._mouse_pos = (m['x'], m['y'])
            ok = 'hover'
            pinned = dict(ok=ok, distance=m['distance'], t_abs=m['t_abs'],
                          eta_s=m['t_abs'] - app.world.time,
                          speed_tooltip=app.hud.telemetry.speed_at_radius(m['distance']))
        for _ in range(10):
            app.ui_root._mouse_pos = (m['x'], m['y']) if ms else app.ui_root._mouse_pos
            one_frame()
            app.renderer.present()
    app.screenshot.directory = os.path.dirname(os.path.abspath(job['out']))
    app.screenshot.set_resolution(job.get('res', '8K'))
    app.screenshot.set_crop(*job.get('crop', (0.0, 1.0)))
    app.screenshot.request()
    one_frame()
    path = app.screenshot.render_scaled(app, draw_scaled)
    app.screenshot.wait()
    if path:
        os.replace(path, job['out'])
    print('aufnahme', job['out'], pinned, flush=True)
    results.append(dict(out=job['out'], pinned=pinned))

json.dump(results, open(os.path.splitext(sys.argv[2])[0] + '_ergebnis.json', 'w'), indent=1, default=float)
app.screenshot.shutdown()
app.maneuver_preview.shutdown()
app.window.close()
