---
paths:
  - "spacesim/render/**"
  - "spacesim/render/gl/**"
  - "spacesim/runtime/gl_device.py"
  - "spacesim/runtime/window.py"
---

# Renderer — draw path, GL state, shaders, screen conventions

## The Renderer is ONE class, assembled from mixins

`render/renderer.py` holds `__init__`, `render()`, the frame projection and the
HUD. Everything else lives in sibling files as a mixin and is pulled in by the
class statement:

    class Renderer(GLDeviceMixin, ShaderPipelineMixin, DrawMixin, TextMixin,
                   BackgroundDrawMixin, BodyDrawMixin, ShipDrawMixin,
                   OrbitDrawMixin, PredictionDrawMixin):

**They share one `self`.** A method in `render/prediction.py` reads
`self._line_program`, which `__init__` created and `render/pipelines.py`
compiled. That is deliberate and is not to be "cleaned up" into composition
(`self.gl.line_program`): it would rewrite several hundred attribute accesses
across the largest files in the project for no behavioural gain.

Two consequences when adding a method: put it in the file that owns its
*topic*, not the one that owns the fields it touches; and check the whole class
for the name first — a duplicate silently shadows by MRO order, which is the
declaration order above.

> **`render/line_kernels.py` must define its five kernel names even when numba
> is missing.** While the kernels sat in the same file as the Renderer, the
> `_LINE_KERNELS_OK` guard at each call site was enough and an undefined name
> was never evaluated. As a separate module they arrive through
> `from render.line_kernels import ...`, and a missing name fails the **import**
> — taking the entire renderer down instead of just the fast path. The
> `except` branch therefore binds all five to `None`.

## Two Y conventions — convert at the ortho boundary

The world draws **top-down** (`line.vert`/`body.vert` flip y; `_world_to_screen_xy` returns
y-down). Text and the ship arrow draw through the **ortho** pipeline
(`ortho.vert`, `texquad.vert`: y-up, origin bottom-left). Anything derived
from world coordinates must be converted before it reaches the ortho pipeline
— use **`Renderer._ortho_y(y)`** for shapes and **`_blit_text_topdown(text,
x_left, y_top, font)`** for text. Never pass a `_world_to_screen_xy` result
straight to `_blit_cached_text` or `_draw_ortho_shape`.

> This was a real bug, fixed 2026-08-15. The ship arrow and every label were
> drawn at `height - y`, i.e. **mirrored about the screen centre**. It was
> invisible for years because the camera hard-snapped the ship to the exact
> centre, where the mirror is the identity. As soon as the ship sat
> off-centre the arrow flew hundreds of px off its own trajectory. The HUD
> block is the one legitimate ortho-native caller and is left alone.

**Only the position was wrong — directions were always right.** A translation
cannot rotate a vector, so the mirror never affected headings. `theta` is
measured **clockwise**: `schiff.apply_thrust` accelerates along
`Vec2(cos θ, −sin θ)`, so that *is* the world nose direction. Hence
`_draw_ship_arrow` uses `(cos θ, −sin θ)` and `_apply_orientation_snap`
measures `atan2(−d.y, d.x)`. Changing either to `+sin` inverts steering.

## Text must be pixel-snapped and must not go through FXAA

Two independent causes of soft, ghosted labels, both fixed 2026-08-15:
1. World-derived label anchors are subpixel (`Erde` at `y=113.7048`). With
   `LINEAR` filtering that spreads every glyph row across two pixel rows —
   fully-opaque pixels drop 19.5% → 9% and a faint copy appears under the
   text. `_draw_texture_ortho` now rounds the quad corner. The HUD was never
   affected because its origin is integer.
2. `_draw_body` runs inside the FXAA FBO pass, so body labels were being
   edge-filtered: fully-opaque pixels 34.7% → 5.3%, glyphs smeared over 55%
   more pixels. Body labels are therefore collected into
   `Renderer._deferred_labels` and drawn in `render()` **after**
   `_apply_fxaa`, where the ship and apsis labels already were.

## The renderer no longer swaps buffers

`Renderer.render()` draws the world and stops; `Renderer.present()` does the `pygame.display.flip()` and fills in
`swap_or_present_ms` / `frame_ms`. The main loop calls `render()` → overlays →
`present()`, so anything that must draw last has a slot. Don't put `flip()`
back inside `render()`.

## Resize and dynamic resolution

**Dynamic resolution.** The window can be freely resized or maximised.
`WINDOWSIZECHANGED` in `runtime/loop.py` calls `Renderer.resize()` (viewport, FXAA target
rebuild, text-cache invalidation) and updates `camera.width/height`.

> **moderngl `ctx.screen` never learns the new size.** It detects its size once,
> at context creation, and every `ctx.screen.use()` restores *viewport and
> scissor* from that stale value. `render()` calls `ctx.screen.use()` after the
> FXAA pass, so without an explicit fix the scissor clips the prediction line,
> ship and HUD to the **old** window rect — maximising showed only a corner of the
> scene. `Renderer.resize()` therefore reassigns **both** `ctx.screen.viewport`
> **and** `ctx.screen.scissor`. `ctx.screen.scissor = None` does *not* work: it
> resets to the framebuffer's own (still stale) size. `ctx.screen.read()` is
> affected the same way — pass `viewport=` explicitly.

**UI scale.** UI is authored
in **design units** and multiplied by `Renderer.ui_scale` — derived from window
height against `renderer.ui_scale_reference_height`, clamped to
`[ui_scale_min, ui_scale_max]` and multiplied by the user's `renderer.ui_scale`.
The lower clamp is 1.0 on purpose: below the reference height nothing shrinks, it
only grows on larger displays. Use `Renderer.ui_px(design_units)` to convert.
Fonts are **re-rasterised** at the new pixel size on every scale change
(`_rebuild_fonts`) rather than stretched, so text stays sharp at any resolution.

## `render/renderer.py`

- `render/renderer.py` — moderngl `Renderer`: bodies, trajectories, HUD, FXAA.
  Attaches to the pygame-created GL context (`moderngl.create_context()`,
  shared wrapper passed in from `runtime/window.py`). Everything is shader-based —
  there is no fixed-function fallback path anymore.
  Screen-space culling for predictor points (`prediction_sampling_tolerance_px`).
  Resolution-driven line detail: `_prediction_error_budget` (tolerance ladder)
  and `_hermite_refine_world` (cubic sub-points, visible segments only).

> **The prediction line is projected in one batch, not point by point.**
> `_adaptive_prediction_screen_points` used to loop over ~3000 sampled points
> calling `_world_to_screen_xy_at_time` each time — measured **19.5 ms per
> frame** with a body-centred frame, almost all of it Python call overhead,
> to draw ~400 points. Three paths now run in numpy:
> `_project_prediction_batch` (points → screen at once, via the frame's
> `to_this_frame_xy_arrays`), `_build_prediction_indices` (the even
> subsample), and a vectorised trivial-reject pre-pass in
> `_build_clipped_polyline_runs`. Result: **draw_prediction 15.5 ms →
> 3.6 ms, whole frame 22 ms → 9 ms**, with the drawn polyline
> **bit-identical** — `tests/prediction_projection_test.py` renders each
> scene twice, once with the batch path disabled, and asserts a maximum
> deviation of exactly 0 px.
>
> Two rules if you touch this. **`ReferenceFrame.to_this_frame_xy_arrays`
> returns `None` by default, and must stay that way** — returning identity
> would let a subclass that overrides `to_this_frame_xy` but forgets the
> array version silently draw its line untransformed. Not being able to
> batch is harmless; computing the wrong thing is not. And **the batched
> maths must be operation-for-operation the same as the scalar version**,
> including the knot interpolation (`_origin_xy_arrays` reproduces
> `xlo + (xhi - xlo) * frac` on the same uniform grid), which is why the
> equality is exact rather than approximate. Note
> `TargetBodyDirectionReferenceFrame` names its bodies `target`/`reference`,
> not `primary`/`secondary`, and its origin is the *target* — the two
> direction frames look identical but are not.

> **The raw scan budget follows the VIEW; an even subsample of a long line is
> not a subsample, it is a different curve (2026-08-31).**
> `_prediction_scan_indices` spreads `prediction_render_max_raw_scan` (3000)
> evenly over the whole points array, and everything downstream — the cubic
> refinement included — can only interpolate *between* what it picked. Short
> lines are unaffected (`raw_count <= max_scan` returns every index), so this
> is invisible until the horizon is long. On an Erde → Neptun transfer
> (horizon 4.5e12 m, 40 000 stored points at 1.125e8 m) it is the whole bug:
> stride 13.3 makes the **drawn** spacing 1.5e9 m — 1590 px at that zoom —
> and of the 6 stored points crossing the Neptune flyby the even subsample
> picks **0**. Measured against the true arc, the drawn line misses it by
> **278 px**, which is the line shooting straight past the planet in the
> screenshot.
>
> Two symptoms, one cause, and the second is the one that reads as a bug
> rather than as coarseness. `step = (count-1)/(max_scan-1)` depends on
> `count`, and `count` drops every frame under the warp hold as
> `Predictor._hold_advance` consumes the head — so the *absolute* points
> chosen walk, up to a full stored point (119 px) per frame, while the stored
> samples behind the head stand bit-identical. The polyline's vertices hop
> sideways: measured **median 23.6 px, worst 287 px of line movement per
> frame**. That is the "wobbles like a swinging rope at every curve and close
> flyby" report.
>
> `_refocus_scan_indices` re-spends the same budget instead of enlarging it:
> segments whose bounding box touches the padded view (a **segment** test, not
> a point test — at stride 13 a chord is 1590 px and the screen 1280, so the
> chord that crosses the encounter often has neither endpoint on screen) are
> resampled at **stride 1**, and a quarter of the budget keeps the off-screen
> remainder coarse so a re-entry still lands in the right place. At stride 1
> there is no phase left to walk. Measured on the same case: support points on
> the flyby arc **0 → 6**, deviation **278 → 2.0 px (138×)**, per-frame
> movement **23.6 → 0.00 px**, and the total scanned actually *falls*
> 3000 → 804.
>
> It returns `None` — leaving the old path bit-identical — for a line shorter
> than the cap and for a fully zoomed-out view where everything is visible;
> there the even spread already is the right answer and no finer stride fits.
> The residual 2.0 px is not the renderer's: it is the predictor emitting only
> 6 points across that arc (uniform arc-length spacing, `length / num_points`),
> and it is the next thing to fix if that is not good enough.

> **The whole line path is COLUMNS now, and never becomes a list of tuples
> (2026-08-27).** The batch note above stopped at the projection; everything
> after it — clipping, min-step compaction, RDP, gap refinement, densification
> — still handed the curve through `list(zip(xs.tolist(), ys.tolist()))` and
> back through `np.asarray` at every stage. At ~4000 points a frame that was
> the second-largest single item in the draw path after the clipper itself.
> `_project_prediction_batch` therefore returns `(None, visible, (sx, sy))` —
> the first slot is deliberately `None`, the columns *are* the points — and
> `_build_clipped_polyline_runs` returns `(n, 2)` float64 arrays rather than
> lists of tuples. Callers that only had tuples (orbit lines, reference
> trails) get columns built for them once at the top, so they take the same
> fast path instead of quietly falling into the Python clipper.
>
> Three more kernels came with it, each **word-for-word** the state machine it
> replaces: `_clip_runs_numba` (the whole Liang-Barsky run splitter in one
> call — the previous single most expensive function of the frame, measured
> **~15 ms at 4000 segments**, essentially all Python loop overhead),
> `_max_gap_refine_numba` and `_densify_numba`. The gap refiner reimplements
> Python's **banker's rounding** on purpose: a support index off by one is a
> different line. The kernels sit at the top of the file behind
> **`_LINE_KERNELS_OK`**; the Python fallbacks (`_compact_min_step_indices`,
> `_max_gap_refine_indices`, `_densify_screen_columns`) stay for the no-numba
> case, **must stay behaviourally identical**, and now also return *indices*,
> so even that path works on columns. `tests/prediction_projection_test.py` §1
> pins the whole chain at a `0.000e+00 px` difference — run it after touching
> any of them.
>
> One micro-optimisation is worth knowing because it looks like a shortcut and
> is not: merging the RDP keeps with the forced first 32 points is **not** a
> general set union — the second operand is a gapless prefix and
> `keep_indices` is already sorted and duplicate-free, so it is one
> `searchsorted` plus a concatenation instead of the hash table `np.union1d`
> builds (measured **0.44 ms per call, two calls per frame**).

> **Uniform writes and GL allocations are cached, because they are calls, not
> maths (2026-08-27).** Two separate costs, same shape of fix.
>
> 1. **GL state cache.** Every `program['u_x'].value = …` and every
>    `ctx.line_width = …` is its own driver call, and the line path rewrote
>    both on *every* draw — measured **~300 uniform writes per frame**, of
>    which almost none differed from the previous call (`u_viewport` is
>    constant over the whole frame, `u_color` over whole groups).
>    `_set_uniform` / `_set_line_width` hold the last **written** value and
>    still write every change, so the output is unchanged. `resize()` calls
>    `_invalidate_gl_state_cache()` — `u_viewport` hangs off the window size,
>    and that is the one value the cache would otherwise be stale about.
>    `ui/text.py::_blit` does the same for its two uniforms and drops the
>    cache in `resize()`.
> 2. **Texture pooling.** The expensive part of a *new* label is not the
>    rasterising but the GL allocation — measured **~0.3 ms per
>    `ctx.texture(...)`**, and the labels that carry a number (speed,
>    altitude, timer, the apsis distances) change every frame. Evicted
>    textures almost always fit the next one, since a digit more or less does
>    not change the line height, so they are collected by size and refilled
>    with `write()` instead of released: `Renderer._acquire_label_texture` /
>    `_retire_label_texture` (cap 64) and the same pair in `ui/text.py`
>    (cap 96). **A font rebuild must drop the pool** — those textures carry
>    the old size and none of them fits any more.
>
> The Ap/Pe diamonds are four unconnected strokes drawn through
> `_draw_line_segments` (`GL_LINES`). They used to be batched one draw per
> colour, but **each marker now carries its own alpha** (`_draw_apsis_markers`
> fades it by the orbit's on-screen size — `dist·camera.scale`, the apsis radius
> in pixels — between `apsis_marker_fade_min_px` and `_full_px`, so Pe/Ap don't
> stack onto the ship/Erde markers when the conic is small on screen). That is
> one draw per marker again, which is fine: real orbits show 1 Pe + 1 Ap, rarely
> two each. The label fades with it via `_blit_text_topdown(..., color=)` →
> `texquad.frag`'s `u_color` alpha.

## `render/gl/`

- `render/gl/` — GLSL 330: `body.{vert,frag}`, `line.{vert,frag}` (top-down,
y flipped in the vert shader), `ortho.vert` (bottom-up ortho convention —
ship arrow, debug crosses; replicates the old fixed-function
`gluOrtho2D(0,w,0,h)` mapping), `texquad.{vert,frag}` (textured quads for
labels/HUD, ortho convention), `ui_rect.{vert,frag}` (the UI SDF shader).
`body_surface.{vert,frag}` + `body_line.{vert,frag}` draw the procedural
body art (see `bodies/style.py`). FXAA lives inline in `render/renderer.py`.

> `texquad.frag` now has a **`u_color` tint uniform**. Text is rasterised
> white and tinted in the shader, so one white texture serves every colour —
> baking the colour into `font.render` would put it in the cache key and
> give every hover state its own GL texture. **GL initialises uniforms to 0,
> so a caller that skips `u_color` draws nothing.** Both call sites
> (`Renderer._draw_texture_ortho`, `ui/text.py`) always set it.

> `ui_rect.frag` is one SDF that covers ~90% of the UI: rounded rects with
> per-corner radii, borders, drop shadows, vertical gradients — plus
> circles, rings and arcs, because a circle is just a rect whose radius is
> half its edge, **and chamfered rects, because a negative radius selects
> `chamfer_box()` instead** (see the chamfer note in `.claude/rules/hud-ui.md`). Rects are spanned **in pixels**, so the SDF is a pixel
> distance and `smoothstep(±0.5)` gives an exactly 1px antialiased edge
> without `fwidth` and without depending on viewport size.

## The ship is drawn in SCREEN pixels, not world geometry

`_draw_ship_sprite` draws `ship/art.py` geometry — at a true scale the ship
would be sub-pixel at every playable zoom. `_ship_length_px()` is the single
source for that size (`renderer.ship_length_px` design units ×
`renderer.ship_render_scale` × the zoom shrink) and every path — sprite, arrow
fallback, label offset — goes through it.

The size is therefore *nearly* zoom-independent: flat down to
`ship_zoom_shrink_start_scale` (camera px per metre), then ramping to
`ship_zoom_shrink_min` at `ship_zoom_shrink_end_scale`, so a system-wide view
doesn't get a 78-px silhouette pasted over the orbit. The ramp is a
**smoothstep in log(scale)**, computed once per frame in `render()` into
`_ship_zoom_factor` — log because zoom is multiplicative (same reason
`camera._ease_scale` interpolates that way), smoothstep so both ends are
kink-free. Regression: `tests/ship_scale_test.py`.

**The ship carries no floating text.** The always-on name (above) and speed
(below) labels were removed from the `body.is_ship` branch of `_draw_body` —
both values live in the player HUD's navball cluster, and the name still
appears through the selection-label system when the ship is clicked.
`_ship_half_height_px` / `_ship_frame_speed_m_s` stay (selection marker, HUD
telemetry).

The old triangle `_draw_ship_arrow` survives as the fallback when
`ship_sprite_enabled` is off or the artwork fails to build; it scales with the
same factor. `_ship_half_height_px` keeps the ship's name/speed labels clear of
the silhouette at any scale.

## Body names are driven by SELECTION, not by zoom

`_queue_body_label` (called from both `_draw_body` paths, full disc *and* icon)
asks `_wants_body_label`, which reads `renderer.body_label_mode`:

- `"selected"` (default) — labels only `selected_body`, but **always**,
  including when the body has shrunk to a 4-px position icon.
- `"zoom"` — the old rule: any body over `body_label_min_radius_px` screen
  radius.
- `"both"` — the union.

Labels are collected into `_deferred_labels` and drawn after the FXAA resolve.
`selection_label_lift_px` raises the text over the selection marker's top
arrow — that lift is the only thing tying the label to the marker, so
`selection_marker_enabled = False` removes the arrows and keeps the name.
§8/§10 of `tests/selection_camera_test.py` pin both.

## The background layer draws first, and absorbs the frame's stall

`_draw_background` (see `.claude/rules/background.md`) runs right after
`target_fbo.clear()` and paints every pixel, so it is effectively the clear.
It stays **inside** the FXAA pass — it is the bottom layer, and pulling it out
would mean pulling everything else out with it.

Being the frame's first GL call, it inherits the wait for the previous frame's
GPU/VSync. Measured: `background_ms` reads ~10 ms with the layer on, but
turning the layer off moves the same ~10 ms into `reference_trails_ms` and
leaves `frame` unchanged (16.60 ms vs 16.40 ms median). Its real cost is
~0.2-0.35 ms. Optimise against `frame`, never against `background_ms`.

## Frame timing — read the labels literally

`last_frame_timings['frame_ms']` is `render()` **itself**; `present()` no longer
rewrites it. The main loop draws the player HUD (`ui_root.render()`) and the
dev-UI *between* the two calls, and that span is reported separately as
`overlay_ms` (`ui_calc` in the `TIMING:` line and in the dev-UI graphs).

It used to be folded into `rend_calc`, which made the renderer look about twice
as expensive as it is: measured on this machine `render()` runs ~11 ms median
while the player HUD alone costs ~8 ms. So a high `rend_calc` now really is the
renderer, and inside it the order is `orbit_lines_ms` (~4 ms) > predictor
prepare (~3 ms) > `bodies_ms` (~1.4 ms). `_emit_render_benchmark`
(`debug.render_benchmark_debug`) prints the full split.
`tests/render_budget_test.py` §1 pins the attribution with a counter-check that
the old arithmetic swallowed the gap, and §3 pins that nothing is allocated per
frame.
