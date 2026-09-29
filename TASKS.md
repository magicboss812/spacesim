# TASKS

Implementation queue for harness sessions. **One session works exactly one
task, then stops.** The next session picks up the next open task. Each task
carries the context a fresh session needs; everything else lives in
`CLAUDE.md` and the path-scoped rules in `.claude/rules/`.

Prompt to start a session:

> Work the next open task in `TASKS.md`. Follow its session protocol.

Status marks: `[ ]` open, `[x]` done, `[!]` blocked (reason in its Result).

## Session protocol

1. **Pick** the first task marked `[ ]`. Tasks run in order and later ones
   build on earlier ones. If a task above it is still `[ ]` in your checkout,
   your branch does not contain the previous session's work: say so and stop.
2. **Read** `CLAUDE.md`, then the rule files the task lists, before touching
   code.
3. **Set up** the environment (below). If the task changes anything visible,
   take a baseline screenshot first.
4. **Implement** only the task. Anything worth doing outside it becomes a
   one-line note in the Result, not part of the diff.
5. **Verify** as the task says. UI work is verified by looking at
   screenshots. Run the existing tests of the files you touched (map in
   `.claude/rules/tests.md`, each runs as `python tests/<file>.py` from the
   repo root; pre-existing failures are listed there). From Task 2 on, do not
   write new test files unless the task asks for one.
6. **Document** per `CLAUDE.md` → "Keeping these files current": edit the
   owning rule file in place; new tunables go into `config/config.json`, read
   through the typed accessors in `config/loader.py`, never hardcoded.
7. **Close out**: flip the task to `[x]` and fill its Result (what changed,
   the measurement that proves it, what the next task must know; 3 to 8
   lines). Blocked: mark `[!]` and write why. A task too large for one
   session (Task 1 may be): commit the finished parts, leave it `[ ]`, write
   in its Result what is done and what remains, and stop. The next session
   continues it.
8. **Commit and push** to the branch this session was given. Working a task
   from this file is the user's request to commit and push that task.
   Screenshots never go into the repo.
9. **Stop.** Do not start the next task.

## Environment (cloud container)

The container starts bare. Once per session:

```
pip install moderngl pygame numpy numba imgui-bundle
ln -sf /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
```

`astropy` and `poliastro` from the `CLAUDE.md` install line are imported
nowhere and fail to build on this image's Python 3.11; skip them. The symlink
exists because moderngl `dlopen`s the unversioned `libGL.so`, which the image
lacks. On the user's Windows machine none of this applies.

There is no display. Everything GL runs under `xvfb-run` on Mesa llvmpipe, a
CPU rasterizer. **llvmpipe timings are not the user's GPU** (NVIDIA,
2560×1440 borderless, 180 fps target): use them for before/after comparisons
inside the same container, never as absolute numbers.

## Screenshots

`tools/game_shot.py` boots the real app, runs N frames and saves what the
window would show (world + player HUD, never the ImGui dev UI):

```
xvfb-run -a -s "-screen 0 2560x1440x24" \
  python tools/game_shot.py 2560 1440 -o "$SCRATCH/shot.png" --frames 90
```

- **The Xvfb screen size must equal the capture size.** `config.json` has
  `window.borderless: true`, so the window takes the whole virtual screen. A
  1600×900 capture of a 1920×1080 screen silently loses the top and right
  edge of the HUD (observed: time-warp bar and system map missing).
- Always pass `-o` into the scratchpad or `/tmp`. The default goes to
  `../screenshots for debugging/`, outside the repo.
- `--node PRO,NRM` places a maneuver node (m/s) at `--node-at` (0..1 along the
  horizon, default 0.25). `--zoom SCALE` sets px/m (`1e-9` shows the inner
  solar system). When a task needs a state the script cannot reach (selected
  body, warp step, pinned tooltip, pause), add a generic flag to the script.
- Check the user's resolution **2560×1440** and the tight **1280×800**.
- View the PNG with the Read tool. For detail, crop and upscale:
  ```
  python -c "import pygame as p; s=p.image.load('shot.png'); c=s.subsurface((X,Y,W,H)); p.image.save(p.transform.scale(c,(W*2,H*2)),'crop.png')"
  ```
- **The HUD's visual language** (read before any UI work,
  `.claude/rules/hud-ui.md` has the full rules): four palette colours with one
  meaning each (`ui/theme.py`, `ROLE_INDEX`), chamfered panels with double
  frames and notch tabs (`ui/hud/chrome.py`), pixel-style type
  (`ui/text.py`), dark teal ground with the triangle grid. New elements sit in
  that language: no fifth colour role, nothing rounded, glossy or glowing.

## Measuring performance

- `debug.print_frame_timings` (on in `config.json`) prints one `TIMING:` line
  per frame. Read the labels exactly as `.claude/rules/rendering.md` →
  "Frame timing" defines them; `debug.render_benchmark_debug` prints the
  renderer's full split.
- Fixed-length run:
  `SPACESIM_MAX_FRAMES=900 xvfb-run -a -s "-screen 0 1280x800x24" python main.py > run.log`
- Profile: `python -m cProfile -o prof.out main.py` in the same command; read
  with `pstats`, sorted by `tottime` and by `cumtime`.
- Drop the first ~150 frames (Numba JIT and cache warm-up) from any average.

---

## Task 1: Performance overhaul `[ ]`

**Goal.** Find and remove every CPU and GPU cost that changes neither what
the player sees nor what the physics computes. The accuracy bar is absolute:
no loss of visual detail, precision, line smoothness, marker placement or
physical correctness. Two parts: **A**, the per-frame cost of the game loop;
**B**, the predictor's cost on long transfer horizons and its lag during
burns. Do A first; B has its own acceptance check below.

**Read first:** `.claude/rules/rendering.md`, `predictor.md`,
`orbit-lines.md`, `reference-frames.md`, `hud-ui.md`, `background.md`,
`physics-world.md` (kernel bit-identity, warp ceilings), `tests.md`.

**Hard rules.**
- World physics is never skipped, thinned, delayed or culled, on screen or
  off. `world.step` advances every body every frame. The Numba world kernel
  stays bit-identical to the Python reference (`tests/warp_predictor_test.py`).
- Work may leave the main thread only if its result is ready for the frame
  that shows it. No result arrives a frame late, nothing drawn lags the state
  it depicts, and the displayed output is identical to the synchronous path.
  The one existing exception is the predictor's async pipeline, which is one
  compute behind by design; Part B is about shrinking that lag, not adding
  more of it elsewhere.
- Culling off-screen **drawing** is allowed. Culling off-screen **state** that
  anything else reads is not: HUD telemetry, `renderer.apsis_marker_hits`, the
  tooltip, orbit-line reveal/fade state, predictor caches.
- The invariants in `CLAUDE.md` hold (curve consumed not translated, one
  Kepler model, one `BurnProfile`, SI units, the two Y conventions).
- Every change is backed by a before/after measurement (median and p95 of
  `frame` or of the specific sub-timing) and a visual A/B.

### Part A: per-frame cost

**Baseline** (this container, 2026-09-29, 1280×800, default start, 900
frames; TIMING means over the last 300 frames, call counts per frame over
all 900; llvmpipe inflates everything GPU-side):

| per frame | value |
|---|---|
| `frame` | 43.9 ms, of which `display.flip` ≈ 12.7 ms (llvmpipe rasterising) |
| `rend_calc` / `ui_calc` / `pred_draw` | 18.0 / 9.2 / 4.6 ms |
| `Predictor.update` (warp hold, steady state) | ≈ 1.4 ms (`pred_calc` 41.4 ms is the one startup compute, not per frame) |
| GL draw calls (`VertexArray.render`) | ~190 |
| HUD `UIDraw.rect` calls | ~245, ≈ 2.0 ms Python time |
| HUD text blits (`ui/text.py::_blit`) | ~77, ≈ 2.5 ms |
| `_draw_apsis_markers` (2 markers) | ≈ 2.0 ms |
| `_draw_orbit_lines` | ≈ 4.2 ms |
| uniform writes through `.value` | ~260 |

**Leads, not conclusions.** Verify each before acting on it.
1. **HUD rebuild.** `ui/draw.py` already batches rects into one instanced
   draw, but every text blit flushes the batch first (`_submit` docstring), so
   ~77 labels split the HUD into many draws. And all ~245 instances are
   rebuilt in Python every frame although most panels do not change between
   frames. Candidates: caching instance data per widget while its inputs are
   unchanged, batching labels (atlas or one textured instanced draw) without
   breaking the z-order or the pixel snapping (`rendering.md` → "Text must be
   pixel-snapped").
2. **Text.** `ui/text.py::_texture_for` hit rates; values re-formatted every
   frame although unchanged; world labels in `render/text.py`.
3. **Apsis markers** cost ~1 ms each for a diamond and a label. Find out why
   (label texture churn from a changing distance string, the per-marker
   time-dependent frame transform, `_draw_line_segments` setup).
4. **Orbit lines** (`bodies/orbit_lines.py`, `render/orbits.py`):
   `future_tracks`, `_recompute` and its trigger conditions, `_cubic_4pt`,
   `FrameAffineTable.project`; pure-Python `kepler_relative_xy` called ~70
   times per frame although `physics/kernels/kepler.py` has the njit path
   (results must stay identical).
5. **Reference frames:** `_body_world_position_exact` recursion and
   `_knot_positions_batch` per frame; memoise per `(body, t)` within a frame.
6. **Uniforms and GL state:** skip writes whose value did not change
   (`render/pipelines.py::_set_uniform`; some caches exist, see
   `rendering.md`).
7. **Python overhead in hot paths:** `getattr` with defaults (~2 M calls per
   900 frames), `max`/`round` in `ui/core.py::px`.
8. **The per-frame `TIMING:` print** with `flush=True` is on in
   `config.json`; on a Windows console that is not free. Measure it; if it
   matters, throttle it (every N frames or ~2 Hz). Keep the feature.
9. **GPU side:** overdraw of full-screen passes (background early-out, FXAA),
   body-style triangle counts at small screen radii (detail ladder in
   `body-art.md`), line tessellation budgets. Only where pixels stay the same.
10. **Predictor:** steady state is the hold path (`hold.py`:
    `_handle_trajectory_branch_change`, `_hold_advance`, `_hold_extend_tail`,
    `view.py::interpolation_error_floor`, `get_apsis_markers` twice per
    frame). Look for redundant per-frame work there; async route and warp
    hold keep their semantics (`predictor.md`). Compute cost and burn lag
    belong to Part B.
11. Whatever else the profiles show. Also profile the zoomed-out solar system
    (`--zoom 1e-9`: many orbit lines and body icons), a maneuver node
    (`--node 800,0`) and a high warp step.

### Part B: long-horizon predictor cost and burn responsiveness

**The report.** With the horizon long enough for transfers to Saturn and
beyond, one prediction compute takes **70 to 90 ms** on the user's machine
(about 17 ms at the default horizon). During a transfer burn the async line
then refreshes less often than the frame rate and trails the ship by about
one compute, so it feels laggy. Both are to be solved as far as the physics
allows.

**The accuracy requirement comes first.** The Ap/Pe markers of a transfer
(position, time, the HUD distance and countdown) must stay where the ship
actually arrives. Warping to the apoapsis at Saturn must show the ship
reaching the point the marker showed right after burnout, and the markers
must not drift on screen or in their numbers during the warp. No speed-up
may make this measurably worse. The world is never approximated to make the
prediction agree with it; the prediction has to agree with the exact world.
`CLAUDE.md` → "Predictor and World share kernels" applies to any integrator
change.

**Acceptance check** (build it first, run it before and after every change;
a headless script that builds the app like `tools/game_shot.py` does, may be
committed as `tools/transfer_bench.py`):
1. Ship in the default Erde parking orbit; burn prograde to a Hohmann
   transfer toward Saturn, once as an executed maneuver node and once as
   manual full throttle; repeat toward Neptun. Raise the horizon until the Ap
   marker at the target's distance exists.
2. Right after burnout record the predicted Ap: world position, `t_abs`,
   distance to Sonne, and the compute time and step count.
3. Warp the world to that time through the real step pattern (`world.step`
   with the loop's per-frame `sim_step` at the warp steps the HUD offers) and
   find the ship's actual extremum of distance to Sonne near `t_abs`. Record
   the position and time error, and the marker's screen position and
   displayed numbers sampled along the warp (drift).
4. Pass: the Ap error and the drift are not larger than before (within the
   noise of two baseline runs). Report metres, seconds, and px at the zoom
   where the Ap region fills the screen.

**Where the time goes today.** Measure before choosing a lead: step count and
where the steps concentrate (parking orbit at departure, heliocentric
cruise, the target's SOI with its moons; `_local_timescale_numba` binds
there), body placement against the ship integration, emitted point count
(`apply_predictor_horizon` scales `num_points` up to `max_num_points` 40 000),
apsis scan, main-thread swap, and the per-compute slowdown when six run at
once (`predictor.md` measured 67 → 87 ms from memory bandwidth). Read
`predictor.md` first: body memo, step ceiling sized by the horizon's time
span, local orbit clamp, point budget, pipeline depth, paced consumption.
Several obvious ideas were already done and measured there.

**Leads for the compute cost.**
1. Steps that buy no apsis accuracy (e.g. a cruise segment pinned by a
   ceiling rather than by the error control) are candidates; steps near the
   departure, the apsis and flybys are not.
2. All 28 bodies are placed at every stage. Grouping a far planet with its
   moons into one source, or a tabulated ephemeris over the horizon instead
   of per-stage Kepler solves, changes the numbers against the world: allowed
   only if the acceptance check shows no measurable change.
3. Full recomputes that an extension of the held line could replace
   (horizon slider, warp step changes, burnout).
4. Output cost: 40 000 points, the apsis scan over all of them, the copy on
   swap.

**Leads for the burn lag.** Today (`predictor.md` → "Under thrust the line
is refreshed by THROUGHPUT" and "Results are consumed at a PACE"): at most
`thrust_pipeline_depth` (6) computes run at once, one started per frame, one
swapped per frame, each result one compute old. At 80 ms per compute and
180 fps that cannot be fresh every frame, and the lag stays one compute.
1. **Latency compensation.** Start each job from the state the ship will
   have when its result is displayed (now plus the expected latency), carried
   through the known thrust: exact for an executor burn (`BurnProfile` is
   deterministic), the held input for manual thrust (when the input changes,
   the error is what today's lag already is). The line then meets the ship
   instead of trailing it. The short arc between the live ship and the job's
   start state must still join the head, and the curve stays consumed, never
   translated.
2. **Executor burns already know their result.** The preview chain
   (`ship/maneuver/preview.py`) computes coast, burn and coast with the same
   `BurnProfile` the executor flies. During an autopilot burn its post-burn
   line and apsides could be shown and only corrected by the predictor when
   the flown state leaves a tolerance. Measure preview against flown at
   burnout first (kick-then-drift vs the RK4 arc, `maneuver.md`).
3. **Near field fresh, far field at pipeline rate**: only if both come from
   the same snapshot and the seam is invisible; otherwise drop the idea.
4. Everything that cuts the compute cost raises the refresh rate directly.
   The frame rate must not drop (the pool leaves a core free,
   `_ensure_executor`).

**Burn acceptance**, at the Saturn-transfer horizon under full throttle and
an executor burn: share of frames with a fresh line, displayed snapshot age
in ms, deviation of the drawn line from a synchronous reference in px (the
`predictor.md` tables are the format), fps unchanged; at burnout the line
settles on the exact coasting line with an Ap marker jump below 1 px, and
the acceptance check above still passes.

### Verification (both parts)
- Run every test in `tests/` before the first change (record the failures)
  and after the last. No new failure.
- Visual A/B with `tools/game_shot.py` at 2560×1440 for at least: default
  start, `--zoom 1e-9`, `--node 800,0`. First diff two baseline runs against
  each other to learn the noise floor, then baseline against optimised:
  ```
  python - <<'EOF'
  import numpy as np, pygame as p
  a = p.surfarray.array3d(p.image.load('before.png')).astype(int)
  b = p.surfarray.array3d(p.image.load('after.png')).astype(int)
  d = np.abs(a - b).max(axis=2)
  print('px changed >8:', int((d > 8).sum()), 'max diff:', int(d.max()))
  EOF
  ```
  Explain or revert any difference above the noise floor, and look at the
  images.
- The Result holds a before/after table per scenario: `frame`, `rend_calc`,
  `ui_calc`, `pred_draw`, draw calls; median and p95. For Part B: compute
  time and steps at the Saturn and Neptun horizons, the burn metrics, and
  the acceptance check's Ap error and drift, before and after.

**Result:** _open_

---

## Task 2: Screenshot key, time pause, HUD visibility toggles `[ ]`

Clean captures for the paper's figures.

**Read first:** `.claude/rules/hud-ui.md`, `devui.md`, `camera-input.md`,
`physics-world.md` (time warp), `runtime/loop.py`.

1. **Screenshot.** Key `F12` (free; `F1` is the dev UI) plus a button in the
   dev UI. Captures world + player HUD **without** the ImGui dev UI: read the
   framebuffer after `ui_root.render()` and before `devui.render()` in
   `runtime/loop.py`. Full window resolution, PNG, into a folder set in
   `config.json` (default `screenshots/` at the repo root, added to
   `.gitignore`), named `spacesim_YYYY-MM-DD_HHMMSS.png`. Encode and write on a
   worker thread so the frame does not hitch; print the path.
2. **Pause.** A new leftmost segment of the time-warp bar
   (`ui/hud/controls.py::WarpBar`, steps in `ui/hud/layout.py::WARP_STEPS`) and
   a key (`Space` is unbound in the game keymap; confirm in `runtime/input.py`
   and `ship/camera.py`). Paused means `world.step` receives 0 s: no sim time
   passes, the mission clock stops. Do **not** set `sim_dt = 0` (there is a
   `min_sim_dt` floor and several divisions by it); gate the step in the loop
   with an explicit flag. While paused, rendering, camera, HUD and the
   prediction display keep working; thrust is refused as under warp
   (`thrust_allowed()`), rotation stays allowed, an armed maneuver does not
   progress. Unpausing restores the previous warp step. Paused state shows in
   the bar's existing active-segment style, the pause glyph drawn as vectors
   like the other glyphs.
3. **HUD visibility.** A dev-UI section "HUD visibility": one checkbox per
   top-level HUD element (ship badge, body browser, target panel, time-warp
   bar, system map, navball cluster incl. the maneuver tiles, AP/PE block,
   snap rosette, frame/zoom/predict group, apsis tooltip) plus "hide all";
   and toggles for the renderer's world overlays (apsis markers and labels,
   body name labels, maneuver node markers and handles). Hiding removes the
   element from **display only**: its update, telemetry and every screen
   position it publishes (maneuver gizmo, `renderer.apsis_marker_hits`) keep
   being computed each frame. Check whether `Widget.visible` (`ui/core.py`)
   also skips update or layout; if it does, add a draw-only flag instead. A
   hidden element takes no mouse input.

**Verify:** screenshots with everything hidden and with only the navball
cluster hidden; one F12 capture opened and checked; a few seconds paused with
`world.time` unchanged and the view still responsive.

**Result:** _open_

---

## Task 3: Ap/Pe markers on the maneuver preview orbit `[ ]`

The planned orbit (`ship/maneuver/preview.py`, drawn in `render/maneuver.py`)
has no apsis markers. The main line has them:
`render/prediction.py::_draw_apsis_markers`, fed by
`physics/kernels/apsis.py` (`_find_apsis_markers_numba`,
`_refine_apsis_numba`) through `ship/predictor/view.py::get_apsis_markers`.

**Read first:** `.claude/rules/maneuver.md`, `predictor.md` (apsis markers),
`hud-ui.md`.

- Scan the preview's coast segment after the last burn for apsides relative
  to the reference body with the same kernels (same refinement onto the
  cubic). Compute when the preview rebuilds, never per frame.
- Draw them with the main line's marker routine, distinguishable from the
  current-orbit markers. `maneuver.md` → "Farben: keine fünfte": no new
  colour; use the node-path colour family or a dimmer Ap/Pe hue, decided by
  screenshot.
- Publish their hits like `apsis_marker_hits`, flagged as preview markers,
  so the tooltip and Task 4 can use them.

**Verify:** screenshots with `--node 800,0` and a retrograde node.

**Result:** _open_

---

## Task 4: Pinned apsis tooltip with warp-to-apsis `[ ]`

**Read first:** `.claude/rules/hud-ui.md` (tooltip), `camera-input.md` and
`physics-world.md` (warp steps and ceilings), `maneuver.md` → "Der
Zeitraffer zieht sich vor der Zündung selbst herunter".

- Clicking an Ap/Pe marker pins its tooltip (`ui/hud/apsis_tooltip.py`) until
  the user clicks anywhere else. Hover stays as is. The pinned tooltip follows
  its marker; if the marker disappears (passed, orbit changed) it closes.
- Below its rows, a HUD-style "WARP TO" button. Target: apsis time minus a
  margin from `config.json` (new key, default 1800 s = 30 min).
- Smooth and failsafe: step down through `WARP_STEPS` as the remaining time
  shrinks (highest allowed step that still leaves a few real seconds at that
  rate), respect the existing clamps (`_clamp_warp`, greyed steps), and
  **never overshoot**: cap the per-frame sim step so `world.time` cannot pass
  the stop time (same pattern as `maneuver_executor.max_sim_seconds` in
  `runtime/loop.py`). On arrival drop to the lowest step, or pause if that
  reads better; document the choice. A manual warp change, thrust or any new
  click cancels it. Not offered on preview-orbit markers or when the stop
  time has already passed.

**Verify:** one warp to a distant apoapsis, one attempt on a periapsis less
than 30 min away; print `world.time` at stop against the target.

**Result:** _open_

---

## Task 5: Apsis marker redesign `[ ]`

The marker is a rotated square from four line segments
(`render/prediction.py::_draw_apsis_markers`; colours duplicated in
`ui/hud/apsis_tooltip.py`). Replace it with a glyph in the HUD's language
(chamfers, pixel grid, palette), Ap and Pe distinguishable at a glance,
about today's footprint (radius 5 design units, scaled like other UI sizes):
neither ornate nor tiny. Applies to the main line and the preview markers of
Task 3. Hit radius and fade-by-orbit-size logic stay. Tooltip colours stay in
sync.

**Verify:** crops at 2560×1440 and 1280×800, on empty ground and over grid
and orbit lines.

**Result:** _open_

---

## Task 6: Star and gas-giant body styles `[ ]`

**Read first:** `.claude/rules/body-art.md`, `config-loader.md`.

- New flags `is_star` and `is_gas` per body in `config/solar_system.json`,
  both `false` when missing (loader `runtime/system_loader.py`, stored on
  `bodies/body.py::body`). Set `is_star: true` on Sonne and `is_gas: true` on
  Jupiter, Saturn, Uranus, Neptun.
- A star variant in `bodies/style.py` / `render/bodies.py`: bright emissive
  disc, limb darkening, restrained granulation or corona, unmistakably a star
  at every zoom. A gas-giant variant: latitude bands with turbulence at the
  band edges, no terrain tiers. Stay geometry and seeded like the existing
  styles (`body-art.md`); keep the icon marker (`bodies/icon.py`) consistent.

**Verify:** Sonne, Jupiter and Saturn close up and at icon size.

**Result:** _open_

---

## Task 7: Ship art to match the HUD `[ ]`

**Read first:** `.claude/rules/body-art.md` (ship), `rendering.md` → "The
ship is drawn in SCREEN pixels".

The ship (`ship/art.py`, `render/ship.py`) looks like it belongs to another
game next to the pixel-style HUD. Nudge it toward that look: pixel-snapped
outline, HUD palette, flatter shading, a hint of the chamfer language. Same
silhouette and size, no heavy restyle.

**Verify:** before/after crops at two zoom levels.

**Result:** _open_

---

## Task 8: Overlaps around the navball `[ ]`

**Read first:** `.claude/rules/hud-ui.md`, `maneuver.md` → "Vier Plättchen
IM Navball-Raster".

Elements overlap in the bottom-centre cluster (navball ring and flank tiles,
AP/PE block, maneuver tiles, snap rosette). Capture at 2560×1440 and
1280×800, with and without `--node 800,0` and with a burn armed if
reachable; crop, name each overlap, fix it in `ui/hud/layout.py`,
`ui/hud/navball.py` or `ui/hud/maneuver.py` without changing the cluster's
overall arrangement.

**Result:** _open_

---

## Task 9: Full-period faint orbit for the selected body `[ ]`

**Read first:** `.claude/rules/orbit-lines.md` → "Die faint volllinie".

When the user selects (clicks) a body, show a low-opacity line over exactly
one orbital period around its parent (Mars: around Sonne), for every body.
The faint full line exists, but `orbit_line_full_max_span_s` (7.5e7 s ≈
810 d) cuts it off from Jupiter outward, because the 3-point knot-count
estimate aliases in rotating frames over long periods. Lift the cap for the
selected body and fix the knot count at its cause (e.g. derive it from the
frame's angular rate and the period) instead of raising the number. Only the
selected body gets the long line, built when selection or frame changes,
never per frame.

**Verify:** Neptun and Pluto selected, in the Sonne frame (closed ellipse)
and in a rotating frame.

**Result:** _open_
