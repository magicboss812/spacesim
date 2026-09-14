---
paths:
  - "spacesim/ship/camera.py"
  - "spacesim/ship/control.py"
  - "spacesim/runtime/input.py"
  - "spacesim/runtime/loop.py"
  - "spacesim/main.py"
---

# Camera, input and the in-app controls

## Input priority is custom UI → ImGui → world

`ui_wants_mouse` / `ui_wants_keyboard`. Note `schiffcontrol` and the camera's
WASD pan read the keyboard by **polling** (`pygame.key.get_pressed()`), not
by events, so they must check the flags explicitly — an event-level guard
alone is not enough.

## Modules

- `ship/camera.py` — `Camera`: world↔screen, zoom, target follow, sim_dt control,
  and the smoothed fly-to (`focus_on` / `recentre` / `set_home_body`).
- `ship/control.py` — `schiffcontrol`: keyboard → ship rotation, nose thrust, and
  latched orientation-hold autopilot (`toggle_snap` sets `snap_mode`;
  `orient_towards_angle` smoothly rotates `theta`). The target heading is
  computed by `Renderer._apply_orientation_snap` (render-time, tied to the drawn
  orbital vectors), not by `schiffcontrol` itself.

## Sim rate is decoupled from framerate

**Sim rate is decoupled from framerate.** `runtime/loop.py` advances the world
**time-proportionally**: `camera.sim_dt * window.fps * frame_dt` per frame, so the
rate is a constant `sim_dt * fps` sim-seconds per real second whatever the rendered
framerate. `simulation.max_frame_dt` clamps the real delta fed to the sim and to
camera/ship input, so a stall can't inject one huge step.

> **Do not "fix" this into a fixed-tick accumulator.** That was tried and reverted.
> The tick rate equals the frame rate here, so the accumulator quantises against
> vsync jitter: measured over 600 frames, **9 frames advanced nothing and 7
> advanced double** (16.4% variation vs 4.2%). Visibly, the ship stutters against
> the predictor line, which is redrawn every frame regardless. The integrator is
> adaptive and `world.step()` already splits into `max_substep_seconds`
> chunks, so a variable outer `dt` is harmless.

## Zoom has a physical ceiling, not just a numeric one

`camera.max_scale` is a raw px-per-metre number, so on a wider window it
allows zooming *further in* — the same setting shows a different amount of
world. `Camera._scale_ceiling()` therefore takes the smaller of `max_scale`
and `width / min_visible_span_m`, and every clamp site goes through it
(zoom step, easing, `set_scale`, `snap_to_targets`).

`camera.min_visible_span_m` (default **12.0**) is the real limit: the screen
never shows less than that many metres across, whatever the resolution.
Verified: `target_scale = 1e10` clamps to 106.67 px/m on a 1280 px window,
i.e. exactly 12.0 m. Set it to 0 to disable and fall back to `max_scale`
alone.

## Controls (in-app)


- `WASD` — pan camera. **Arrow keys no longer pan** — they drive the ship only.
  (They used to do both; `camera.update` and `schiffcontrol` consumed the same
  keys simultaneously.) `move_speed` is now **screen-heights per second**, not
  pixels per second.
- **Mouse: wheel = zoom, middle/right-drag = pan.** Zoom is eased in log space
  toward `camera.target_scale` and anchored on the **screen centre**; panning
  eases toward `camera.target_position` and glides on release
  (`camera.pan_inertia_*`). `camera.scale`/`position` are the *drawn* values —
  never write them directly, write the `target_*` ones and let
  `Camera.update(dt)` ease. Easing is framerate-independent
  (`1 - exp(-rate * dt)`). Panning **breaks the follow**; `Home` restores it.

> **The zoom anchors on the SCREEN CENTRE, and the follow is all-or-nothing.**
> Both halves settled 2026-08-24 after two earlier attempts said otherwise;
> don't "restore" either of them.
>
> `zoom_by()` now takes no anchor at all: it changes `target_scale` and moves
> the camera by exactly nothing. Cursor anchoring (the map convention, and
> what this note described for a day) shifts the camera target on *every*
> notch, and the eased position then chases that target — during fast in-and-out
> zooming that reads as the camera trying to catch up to the ship. Measured
> over a 240-frame sweep with the body under time warp:
> **0.0 px of movement now, up to 4726 px off-centre with cursor anchoring.**
> Anchoring on the followed body instead (the version before that) keeps the
> body still but drags the picture back toward it whenever you look elsewhere.
> Centre anchoring is the only one of the three that both keeps the followed
> body nailed down and leaves a free camera alone.
>
> The follow itself lost its offset. `camera.follow_offset` is **gone**: there
> is no "following, but shifted" state any more. It made time warp unusable —
> a carried offset means the camera keeps flying at the body's full orbital
> speed after the player merely looked aside, and it is what the zoom lag hung
> off. `Camera._shift_target_position` (WASD, drag, inertia) therefore calls
> `unfollow()` on its **first** invocation and shifts `target_position`
> afterwards. Two consequences: a panned camera stands still in world space,
> at world velocity exactly 0 — it does not inherit the body's motion — and
> the detach must be once-only, since `unfollow()` resets `target_position` to
> the current (still easing) position and would otherwise defeat the easing
> every frame.
>
> What survives is `Camera._focus_offset` — the residual of a running
> `focus_on` fly-to, eased to a constant zero. Outside a fly-to it is exactly
> `(0, 0)`, so `position` is bit-for-bit `target.position` and there is
> nothing that *could* lag during a zoom. Re-attaching is only ever the
> player's doing: a click on a body, `Home` (→ `Camera.recentre`, which flies
> to `home_body`), or the one startup `camera.follow(ship)` in `runtime/bootstrap.py`.
> That startup call must sit **before** the first `FrameController.apply()`,
> or its `FRAME:` line reports `camera_follow=frei`.
>
> `tests/selection_camera_test.py` §4 and §5 cover all of it, each with a
> counter-check: 0.0 px vs 4726 px on the zoom sweep, and a body that flies
> 75 px out of frame in the 5 s during which the panned camera does not move.
- **Feed `camera.target_scale`, not `camera.scale`, to
  `predictor.set_view_scale()`.** Any change beyond `snapshot_view_rel_tol`
  (1e-6) *used to* flag `_view_scale_changed` unconditionally, which makes
  `Predictor.update()` run `_compute_full()` synchronously on the main
  thread. The animating scale causes that every frame of a zoom (measured:
  37 rebuilds per gesture vs 1). Since 2026-08-17 the flag is only set when
  the zoom changes the **effective point spacing** (`_effective_precision()`)
  — with the horizon pinned (`length = num_points × precision`, `ship/horizon.py`) the
  spacing is floored to `precision` at every zoom, the rebuild would be
  bit-identical, and it is skipped entirely. Measured: 30 wheel notches =
  **0 sync recomputes, worst update 0.2 ms** (was 62 ms *per notch* with the
  full solar system — the zoom fps collapse). The old guarantee "after a
  zoom, update() builds a line synchronously if none exists" is kept: the
  skip only happens when there are points to keep. Async snapshot staleness
  compares `eff_precision` the same way. Note this removed the side effect
  `tests/prediction_projection_test.py` silently relied on (a sync rebuild
  freezing the line for each batch/scalar render pair) — its `draw_runs` now
  takes `refresh=False` for the second render of a pair instead.
- **Left-click selects a body; a second click on it flies there.** `Home`
  returns to the ship, a click on empty space deselects. The ship is in the
  pick list like any other object.

> **Selection is a VIEW state and must never move the reference body.**
> `UIState.selected_index` sits beside `reference_index` but is a different
> thing: the reference body decides what is computed and drawn against, and
> that stays the player's decision (`R`, the body drawer). A click only says
> "this one". Two consequences that are load-bearing:
> `UIState.select_body()` deliberately does **not** fire `on_change` — that
> callback rebuilds the plotting frame and calls `predictor.invalidate_hold()`,
> i.e. a full trajectory recompute per mouse click, for a marker. And
> `FrameController.apply()` (`runtime/bootstrap.py`) **no longer touches the camera at
> all** — it used to call `camera.follow(ship)` unconditionally, so the next
> `R` or `1`/`2` would yank the view off the planet you just flew to, or undo
> a free pan. It was there to stop a frame change jumping the view, which it
> never needed to be: screen centre is `frame(camera.position)` and camera and
> content go through the same rigid transform, so a frame change moves both
> equally. That also made the previously dead `camera.follow(earth)` at the
> top of `main()` live again — it had always been overwritten a moment later —
> so it was removed in favour of one explicit startup follow.
>
> **Picking goes through `_world_to_screen_xy`, not `camera.world_to_screen`.**
> The frame-aware path is the one that draws, so the grab area is the drawn
> area by construction. Measured in a Sonne–Erde direction frame half a year
> in, the frame-blind transform puts Erde **340 px** away from where it is
> drawn. `Renderer.pick_body` runs on mouse-up only — 35.6 µs for 28 bodies —
> and takes the **nearest centre**, not the largest hit, so a moon in front of
> a screen-filling Sonne still wins. The DOWN/UP pair is required: reacting to
> the press alone would let every pan that happens to start on a body reset the
> selection.
>
> **The camera flies by easing the OFFSET, never the absolute position.**
> `Camera.focus_on(body)` retargets immediately and sets
> `_render_follow_offset` to exactly the difference that leaves the picture
> unchanged, so the eased quantity converges on a constant zero. Easing the
> absolute position onto a moving body instead keeps the `v/k` residual
> documented at `ship/camera.py:43` — the test measures 0 m against that
> alternative's steady lag. `focus_smoothing` (4.5) covers 95 % of the way in
> ~0.7 s; `pan_smoothing` (20) is back the moment it lands, so dragging stays
> direct.
>
> **Reference frames need no inverse transform here.** Screen centre is
> `frame(camera.position)` and the body goes through the same rigid transform,
> so a world-space offset is a straight-line pan on screen in a non-rotating
> frame, and in a body-direction frame additionally rotates at the reference
> body's orbital rate — ~1e-7 rad over the transition. Interpolating in frame
> space would be more code for a sub-pixel difference.
>
> The marker itself (`Renderer._draw_selection_marker`) is four filled
> triangles in **one** `_draw_ortho_shape` call, drawn after the FXAA resolve
> like the body labels, with sizes in design units so it is the same on screen
> at any zoom. Measured **+0.023 ms per frame** with a selection and exactly
> zero without. Its spin and pulse phases advance off `real_dt`, so they are
> framerate-independent. Because the top arrow lands precisely where
> `_draw_body` anchors the body label, `selection_label_lift_px()` pushes that
> label up — computed **without** the pulse on purpose, so the text does not
> breathe.

- `Camera.handle_event` / `update` take `ui_wants_mouse` / `ui_wants_keyboard`
  — the input-priority seam for the UI layers (custom UI → ImGui → world).
- Arrow `Left`/`Right` — rotate ship. Arrow `Up`/`Down` — nose thrust
  forward/backward.

> **Thrust is real-time only; rotation always works.** Above
> `simulation.realtime_warp_max` (60 sim-s/s — the HUD's lowest warp step)
> `runtime/loop.py` skips `apply_thrust`. A per-frame impulse is neither dosable nor
> reproducible when one frame is hours of flight (it would scale with the
> frame rate), and it invalidates the held prediction every frame. Rotation
> is untouched because it does not change the orbit. The throttle reads
> `Telemetry.thrust_locked` and shows `HOLD` in place of the percentage —
> without that the player presses `Up`, nothing happens, and nothing on
> screen says why. The bar still shows the level it will use back in
> real time.
>
> **`min_sim_dt` is per *tick*; the warp steps are per *real second*.** Mixing
> the two locked thrust out permanently, fixed 2026-08-16. `config.json` sets
> `min_sim_dt = 1.0`, which was exactly the lowest warp step back when the
> game ran at 60 fps. At the current `window.fps = 180` that floor pins the
> slowest reachable rate to `1.0 × 180 = 180` sim-s/s — three times
> `realtime_warp_max`. So `thrust_allowed()` was false at *every* warp step
> including the lowest, the throttle read `HOLD` forever, and the predictor
> never left the warp hold. `Camera.allow_warp_rate(rate, tick_rate)` lowers
> the floor so the real-time step stays reachable at any frame rate;
> `runtime/bootstrap.py` calls it right after the tick rate is known.
> `tests/warp_predictor_test.py` §6 checks 30–240 fps, both the HUD button
> and `PageDown`, and the throttle label itself.
- `I` / `K` / `J` / `L` — toggle an orientation-hold autopilot (latched: tap to
  engage, tap again to release). `I`=prograde, `K`=retrograde, `J`=normal-inward,
  `L`=antinormal-outward. The HUD mirror is the **snap rosette**
  (`controls.SnapRosette`): four buttons on a cross around a ship glyph rather
  than a 2×2 grid, because the four directions are an axis pair, not a list —
  prograde faces retrograde, normal faces antinormal, and the grid hid that.
  The colour follows the *axis*, so both ends of a pair share it. On the tap the nose rotates smoothly to the target
  (acquire), then `schiffcontrol._snap_locked` pins it exactly on the vector
  every frame — so a velocity change that swings the vector faster than
  `rotation_speed` can't make the nose lag or flip the wrong way.
  Directions are tied to the **predictor line** — `apparent_orbital_directions`
  takes the predictor's own polyline (`predictor.get_points()`) and uses the
  tangent of that drawn line in the active plotting frame, so the directions
  track the line as it changes shape (thrust, precession) and follow
  rotating/translating frames correctly. (A straight-coast finite-difference is
  the fallback only when no line is available.) The **drawn prograde/normal
  vectors are the single source of truth**: `Renderer._apply_orientation_snap`
  runs inside `render()` right before the ship arrow is drawn and ties the nose
  onto the selected vector via `frame.heading_from_this_frame` at the *same*
  `_frame_time_s` used to draw the arrow — so the nose lands exactly on the
  vector regardless of `sim_dt` or frame-rotation rate. `retrograde`/
  `antinormal_out` are just the negated `prograde`/`normal_in`.
  Only world-space `theta` is stored, so physics stays absolute. Debug overlay
  always draws prograde (green) + normal-inward (magenta) vectors from the ship.
  (`Up`/`Down` nose thrust still applies manual delta-v.)

> **The tangent chord needs a minimum LENGTH, not just a non-zero one.** Fixed
> 2026-08-17. `_prograde_from_line` took the chord from `points[0]` to the first
> sample more than `1e-12 m` away. Under the warp hold `points[0]` *is* the ship
> and the sample ahead of it stands still, so that chord runs continuously from
> a full point spacing down to **zero** between every two consumed samples. The
> ship is never exactly on the held curve either — world and predictor propagate
> the planets slightly differently, leaving a lateral offset `d` of a few metres.
> The angle error is `~ d/c`, so as `c → 0` it diverges: measured on a straight
> line with `d = 37 m`, the old rule gave −0.002° at a full spacing but
> **−20° at c/spacing = 1e-4 and −88° at 1e-6**. That is the navball and the
> prograde/normal vectors flying into random directions after a few seconds of
> warp — and why zooming "fixed" it, since a zoom forces a full recompute that
> resets `d` to zero. It got worse the longer you warped only because more
> consumed samples mean more chances to catch a near-zero `c`.
>
> The chord must therefore clear `_MIN_TANGENT_CHORD_FRACTION` (0.25) of a
> **real sample-to-sample spacing**, measured with the head excluded — the head
> is the very thing that shrinks. A chord of 1000 m is not degenerate, it is
> merely too short *against the noise*, which an absolute epsilon can never
> express. With uniform spacing (everywhere outside the hold) the `i = 1` chord
> already clears the bar, so that path stays **bit-identical** and is tested
> that way. Measured after: worst case 0.011° instead of 0.388° over 3000
> frames, and flat instead of growing.
- **The HUD is mouse-driven and mirrors the keybinds.** Warp buttons write
  `camera.sim_dt`; SRF/ORB/TGT writes `UIState` (same state as
  `1`/`2`/`T`); the snap rosette calls `ship_control.toggle_snap` (same as
  `I`/`K`/`J`/`L`); the throttle arc scales `ship_control.thrust_acc`; SYSTEM /
  LOCAL set `camera.target_scale` from the real body distances; the body
  drawer under the ship badge writes `UIState.set_reference_index` (same
  state as `R`, but shown rather than cycled blind); dragging the attitude
  ring steers via `orient_towards_angle` (rate-limited, and it releases the
  latched autopilot first). Every control **reads its value back from the
  simulation** rather than holding its own copy, so keyboard and HUD cannot
  drift apart. `renderer.hud_enabled=false` skips building it entirely.
  (A `tools/hud_shot.py` for offscreen HUD renders is referenced in older
  notes but does not exist — `tools/` holds only `run_and_log.py`.)

> **Ring drag: the ring is grabbed, not the nose — and it must let go.** Two
> bugs lived here, both fixed 2026-08-16. (1) The ticks are drawn at
> `deg - heading`, so heading rises as the ring turns *counter*-clockwise;
> setting `heading = cursor + offset` therefore ran the ring backwards under
> the cursor. The grabbed ring point is what has to stay put, so the offset
> is `heading + cursor` and the drag reads `heading = offset - cursor`.
> (2) `_manual_heading` was never cleared on release, so `update()` kept
> calling `orient_towards_angle`, which sets `_snap_locked` on arrival and
> from then on pins `theta` **every frame**. The arrow keys still wrote
> `theta` and were overwritten in the same frame — after one drag the ship
> could only be steered from the ring. `on_mouse_up` now clears both.
> Steering is live only while the button is held.
- **The old debug text wall is off by default** (`renderer.show_debug_hud`).
  Its numbers live in the HUD and in the F1 dev panel now.
- `F1` — toggle the ImGui developer tools (hidden by default;
  `debug.devui_visible` starts them open). `F1` is handled *before* the
  input-priority check so the panel can always be closed, even while a text
  field has focus.
- `O` — toggle the bodies' orbit lines (`bodies/orbit_lines.py`).
- `P` — toggle predictor. `E` — toggle epicycle (Ptolemaic) mode.
- `R` — cycle reference body. `1` — non-rotating frame.
  `2` — body-direction frame. `T` — target overlay.
- `+` / `-` — predictor length ×2 / ÷2 (instant, full recompute). `9` / `0` —
  predictor precision (finer / coarser). The **HUD horizon slider**
  (bottom-left, under FRAME) is the smooth alternative to `+` / `-`: a
  centre-sprung rate control that changes only the *drawn* length per frame
  (`predictor.set_display_length`), with the computed horizon trailing
  asynchronously up a coarse ratchet. Its range is `0.25×`–`256×` of the base
  horizon (`predictor.horizon_slider_min_mult` / `_max_mult`), i.e. 2.5 Gm to
  2.56 Tm; past `4×` the point spacing coarsens exactly as it does under `+`.
  See `.claude/rules/hud-ui.md`.
- `PageUp` / `PageDown` — sim_dt up / down. `Esc` — quit.
- **Warp steps are `1m/s, 10m/s, 1h/s, 1d/s, 7d/s, 30d/s, 100d/s, 1y/s`**
  (`ui/hud/layout.py::WARP_STEPS`, sim-seconds per real second). The top
  three were added 2026-08-18; steps that the current orbit cannot resolve
  are greyed out and refuse clicks, and `_clamp_warp` in `runtime/loop.py`
  backstops the keyboard. See the time-warp note in `.claude/rules/physics-world.md`.
