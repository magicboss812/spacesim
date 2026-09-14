---
paths:
  - "spacesim/ship/predictor/**"
  - "spacesim/ship/horizon.py"
  - "spacesim/physics/kernels/**"
  - "spacesim/render/prediction.py"
---

# Trajectory predictor — the drawn look-ahead line

## The Predictor is ONE class, assembled from mixins

`ship/predictor/` is a package, not a module. `core.py` keeps the state, the
integrator quality and `update()` — the frame entry point that picks which of
the four routes (hold, rolling, async, full sync) this frame takes. The rest:

    class Predictor(HoldMixin, ComputeMixin, JobsMixin, ViewMixin):

    hold.py     the warp hold, consuming the curve, branch changes
    compute.py  snapshot -> integration -> point list
    jobs.py     the async pipeline: submit, cancel, depth, swap
    view.py     what comes out: points, Ap/Pe markers, length, spacing

As with the Renderer these share one `self` — see `.claude/rules/rendering.md`.

**The numerics are not here.** Integrators, the Kepler body model, the apsis
search and the propagation kernels live in `physics/kernels/`, because they are
physics and are shared with the world. `ship/predictor/__init__.py` re-exports
`_find_apsis_markers_numba` and friends, since the predictor was their address
for years and `tests/warp_predictor_test.py` still imports them that way.

**Invariant: `_anchor_first_point(ship, world)` CONSUMES the curve; the rigid
shift is only the fallback.** Changed 2026-08-25 — see the apsis-stability note
below.
First choice is `_advance_points_along_curve`, the
same mechanism the warp hold has always used: drop the leading samples whose
absolute time has passed, leave the rest exactly where it is, prepend the
ship as a synthetic head. The rigid translation survives for the rolling
mode and for a curve whose time has run out.
**If it does shift rigidly, it must rebase position AND time — and record
the offset.** The `(n,5)` points array's third column is absolute sim time,
and the renderer uses it via `_world_to_screen_xy_at_time` to pick the
plotting frame's *epoch*. If only x/y are dragged forward, the time base
falls behind by one `sim_dt` per frame (measured 900–2700 s) and, with a
moving frame origin (body-centred non-rotating), the whole line is displaced
by the origin's drift over that interval — measured 54.5 px at 2e-6 px/m,
thousands of px zoomed in, and twitching as the async latency varies. The
shifted times then no longer match `snapshot["sim_time"]`, so the amount
goes into **`_points_time_offset`**; anything converting a point time back
into a *local* time (the apsis scan propagates the reference body with it)
must subtract it, or it reads the bodies that far ahead. Invariant: **after
anchoring, `points[0]` is the ship's position at `world.time`.**

---

- `ship/predictor/` — trajectory lookahead. Default mode is `"rkn"` (adaptive
  RKN with step-doubling). Primary `@njit` kernels: `_compute_distance_points_rkn_numba`
  (main hot path), `_rkn_acc_numba`, `_rkn4_step_numba`, `_rkn_adaptive_step_numba`,
  and their `_time_` variants for reference-frame–aware integration.
  `_rk4_step_numba` is kept only as an ASPI/rolling-mode fallback — it is
  not the default integrator. Quality presets (`fast`/`balanced`/`accurate`/`rk4`)
  adjust `rkn_*` tolerances. Async via `ThreadPoolExecutor`. Largest file in
  the repo (**6500 lines**, with `render/renderer.py` at 4900 right behind it) —
  edit with care. Apsis markers:
  `_find_apsis_markers_numba` scans the prediction line for local min/max
  distance to the reference body; `Predictor.get_apsis_markers()` returns
  them lazily (cached per points-array identity),
  `Renderer._draw_apsis_markers` draws Pe/Ap diamonds + distance labels.
  Toggles: `predictor.apsis_markers_enabled`, `renderer.show_apsis_markers`.
  **`_refine_apsis_numba` places the marker on the CUBIC, not on its chord**
  (2026-08-31). The parabola fit on `d²` gives the vertex's *parameter*; the
  position then comes from the same Bézier-form Hermite the renderer draws
  (`b1 = p0 + v0·dt/3`, `b2 = p1 - v1·dt/3`), word-for-word
  `rendering._hermite_refine_world`. Interpolating linearly along the segment
  — the old code, whose comment claimed it put the marker "exactly on the
  drawn line" — puts it on the chord instead, off by the sagitta. That is
  fractions of a pixel while the spacing is small against the curvature
  radius, and not at all on a long horizon: at spacing 1.125e8 m around a
  1.69e8 m periapsis (Erde → Neptun) it measures **6.03e6 m = 6.4 px**,
  flipping side with the sampling phase — the "line sits under the Pe marker,
  next frame above it" report. On the cubic: **22.7 m = 0.00 px**.
  **The `use_tangents` flag is passed in from Python and must stay that way.**
  The kernel cannot check its own velocity columns: it is `fastmath=True`, so
  LLVM may assume `nnan` — measured here, *both* `math.isfinite(nan)` and
  `nan == nan` return **True** inside an `@njit(fastmath=True)` function. A
  guard there would have fed NaN into the cubic and the markers would have
  vanished, exactly as the `body_memo` `valid` column note below describes.
  `Predictor._points_have_tangents` does it in numpy instead; with the flag
  clear the result is the old linear form to **3.7e-9 m**, which is what the
  chord kernels (ASPI, plain RK4, `NaN` tangents on purpose) need, since their
  stretches are drawn straight anyway.
  Under the warp hold the per-frame trim/extend used to invalidate that cache
  every frame (and HUD + renderer each scanned all 10 000 points, twice per
  frame). Hold-path mutations are now **soft** invalidations
  (`_invalidate_derived_caches(soft=True)`): the retained samples are
  bit-identical, so cached markers are served with expired ones filtered out,
  and a full rescan runs at most every `apsis_hold_rescan_s` (0.25 s). A new
  marker at the far end of the horizon appears at most that much later.
  Points are `(n, POINT_COLUMNS=5)` — `[x, y, t_abs, vx, vy]`; kernels that
  place their samples linearly on a step chord (ASPI, plain RK4) write `NaN`
  tangents on purpose, and the renderer draws those stretches straight rather
  than inventing a curvature the samples do not have.
  `interpolation_error_floor()` reports the finest drawn accuracy the current
  point spacing can support.

> **Horizon beats point density — never the other way round.** The kernel
> emits points at a fixed spacing and at most `num_points` of them, so the
> drawn arc is `num_points × spacing`. `auto_precision_from_zoom` *refines*
> the spacing as you zoom in (`target_screen_step_px / view_scale`), which
> silently **shortened the prediction**: measured 10 % of the horizon left at
> `view_scale` 2e-5 and 1 % at 2e-4. On screen that reads as the line being
> cut off at the first screen edge and never coming back — with the Ap/Pe
> markers gone and `CLOSEST`/`T-CA` blank, because all three read the same
> points array. (The renderer was **not** at fault:
> `_build_clipped_polyline_runs` already splits a path that leaves and
> re-enters the view into separate runs and never joins them across the gap.)
> `_horizon_spacing_floor()` = `length / num_points` now floors the spacing,
> so a full budget draws coarser rather than shorter. That is the right side
> of the trade: a chord's sagitta grows as `c²/8R`, so the extra coarseness
> is sub-pixel wherever the orbit is nearly straight, while a missing horizon
> makes the display useless. More near-field detail comes from a bigger
> `predictor.num_points`, never from a shorter horizon.

> **Near-field detail comes from INTERPOLATING, not from more points.** The
> points array is `(n,5)` — `[x, y, t_abs, vx, vy]` (`predictor.POINT_COLUMNS`).
> The two velocity columns cost nothing: the kernel already evaluates a cubic
> Hermite inside every integrator step to place its samples, and already
> computes the consistent tangent at each one (`resume_vx/vy`) — it just threw
> all but the last away. With them stored, the line stops being a sequence of
> positions and becomes a **piecewise cubic curve** the renderer can evaluate at
> any density at draw time.
>
> That matters because at the default 1000 km spacing a straight chord cuts a
> LEO orbit's corner by `c²/8R` = **17.8 km**. Cubic interpolation of the same
> points leaves **7.6 m** — measured, 2350×, with zero extra integration. (Note
> `auto_precision_from_zoom` cannot help here: `ship/horizon.py` sets
> `length = num_points × precision`, so `_horizon_spacing_floor()` equals
> `precision` exactly and the zoom refinement is floored out at every zoom.)
>
> **Drawn error has two independent terms, and only one of them is free:**
>
>     error = |Hermite − truth|  +  |polyline − Hermite|
>             (set by SPACING)      (set by SUBDIVISION)
>
> Subdivision only touches the second. The first is the textbook cubic bound
> `h⁴/384·|f⁗|`, which for an orbit is `c⁴/(384 R³)` —
> `Predictor.interpolation_error_floor()` computes it from the circumradius of
> three consecutive points and it matches numerically integrated arcs to 0.4 %
> over three decades. **So the ladder's 1 mm and 1 cm rungs are not reachable by
> drawing harder**; they need a finer `precision`, i.e. new integration. The
> renderer therefore reports target *and* achieved tolerance (`F1` → prediction
> detail) rather than claiming a precision the line does not have.
>
> Two traps if you touch `interpolation_error_floor()` or
> `_hermite_refine_world()`. **Thinning the point list must not change the point
> spacing** — a `linspace` over the array spreads the triples apart and, since
> the floor goes as `c⁴`, reports a multiple (measured 120 m instead of 7.6 m);
> sample *starting indices* and keep each triple adjacent. And **the
> subdivision count is `n = ceil(sqrt(0.75·M/tol))`, not `sqrt(√3·M/(8·tol))`** —
> the latter circulates in rasterisers and under-subdivides by 1.86×, which
> showed up as exactly `sagitta/n²` = 1122 m against a 1000 m promise.

> **One error budget for the whole draw chain.** `renderer.prediction_error_ladder_m`
> (`[1mm, 1cm, 1m, 100m, 1km]`) is a **maximum deviation of the drawn line from
> the true trajectory**, not a point spacing. Per frame,
> `_prediction_error_budget()` takes `0.3 px / prediction_detail_scale`,
> converts to metres via `camera.scale`, and snaps **down** to the largest rung
> that still honours it, clamped to the ladder's ends.
>
> The quantisation is load-bearing, not cosmetic: a continuously varying target
> would re-derive the line on every frame of a zoom — the same failure
> `snapshot_view_rel_tol` already had (37 rebuilds per gesture vs 1).
>
> **All four stages must share that one budget** (0.5 subdivision / 0.25 RDP /
> 0.1 min-step, summing under 1 because the errors add). They were independent
> before, and the simplification simply undid the refinement: RDP's own
> tolerance may reach 0.25 px, which at 4.4e-5 px/m is 5700 m, so a 1000 m
> promise measured 1990 m. Also **the point budget degrades uniformly, never
> truncates** — `counts` is scaled by `room/total`, because spending it
> front-to-back would shorten the horizon, which is the one failure
> `_horizon_spacing_floor()` exists to prevent.
>
> **Flatness is estimated in world space, then scaled to pixels** — every
> plotting frame is a rigid transform, and a second difference is a length, so
> rotation and translation leave it alone. Projecting the two inner Bézier
> control points instead is the obvious implementation and costs **1.9 ms/frame**
> over ~3000 segments to choose a subdivision level that added five points. What
> the world-space form does *not* see is the curvature a rotating frame adds over
> one segment (~0.14 m against 0.83 m of real bulge for the Erde–Sonne direction
> frame); `prediction_sampling_max_segment_px` covers that gap.
>
> Segments whose chord-expanded bounding box misses the view get **zero**
> sub-points. `tests/prediction_detail_test.py` measures all of this against an
> analytic circle: 17 850 m → 503 m at default detail, → 11.8 m with the budget
> raised, which is exactly the interpolation floor plus the integrator's own
> error.

> **Under time warp the line is held, not recomputed.** `Predictor.set_hold()`
> (driven from `runtime/loop.py` above `simulation.realtime_warp_max`, default
> 60 sim-s/s) switches `update()` to `_hold_advance()`: leading points whose
> absolute time has passed are dropped and the rest stays exactly where it is,
> because **a prediction is a property of the orbit, not of the instant** —
> without thrust the ship simply slides along it. Refresh is one *synchronous*
> `_compute_full` when the supply falls below `hold_refresh_fraction` (0.25),
> about every 38 frames at 7d/s, so the line can never run dry.
>
> The bug it fixes: `_anchor_first_point` translates the **whole** curve
> rigidly so its head sits on the ship. At 7d/s the ship advances ~1e8 m per
> frame, so a stale async result got dragged that far sideways and snapped
> back on the next swap — measured **3.9 px median and 77 px p99 of movement
> per frame** at 2e-6 px/m, which is the shaking. Held, that is **0**. Two
> things had to follow: the trim must be **committed before** any early
> return (otherwise the curve freezes while the ship flies on, the head
> offset grows every frame and the hold latches off), and while holding,
> `_anchor_first_point` attaches the head with a **taper** over
> `hold_taper_points` (64) instead of translating everything — otherwise it
> corrupts the very curve the hold is trying to keep still, leaving the time
> column inconsistent with the geometry. Real world-vs-predictor drift is
> only ~37 m per frame, so the taper has almost nothing to absorb.

> **Changing the warp STEP requests a new line, it does not force one.**
> Fixed 2026-08-18, and it is the same fix as the thrust one two notes down —
> the same mistake at a different trigger. The step sets the horizon factor
> (1x/4x/16x/64x from 7 d/s up, `predictor_warp_length_mult`), so every change
> calls `set_length()`, which called `invalidate_hold()`, which made
> `_hold_advance` return False, which made `update()` run `_compute_full`
> **synchronously on the main thread**. Measured with the full solar system at
> 180 fps, against 0.3–0.5 ms in the neighbouring frames:
>
> | up | | down | |
> |---|---|---|---|
> | 7d/s → 30d/s | 47.6 ms | 1y/s → 100d/s | 30.6 ms |
> | 30d/s → 100d/s | 31.1 ms | 100d/s → 30d/s | 48.2 ms |
> | 100d/s → 1y/s | 40.6 ms | 30d/s → 7d/s | 14.9 ms |
>
> That is the hitch on switching, and it is unnecessary: **a held curve that
> is merely the wrong LENGTH is not wrong.** Too short only means it needs
> refreshing sooner; too long means nothing at all, since
> `set_display_length()` draws only the un-warped part either way. So
> `invalidate_hold(soft=True)` marks that case, `_request_hold_recompute()`
> submits exactly one job, and the hold keeps consuming the old curve until it
> lands. Result: worst frame across the whole ladder **82.0 → 9.4 ms**, medians
> unchanged. `set_precision` is soft for the same reason (spacing is cosmetic).
>
> **The two directions of `set_hold` are not symmetric.** Switching the hold
> *off* stays a hard invalidation — the player may thrust in the very next
> frame. Switching it *on* adopts a curve the async path refreshed every frame
> right up to the previous one, so it is soft; hard cost 14.1 ms at the
> 10m/s → 1h/s step, where the hold engages, against 0.2 ms either side.
>
> Three things had to follow, each found by measuring.
>
> 1. **The swap must not rebase.** `_swap_ready_result` normally translates the
>    new curve rigidly onto the current ship position. Under the hold that is
>    the exact artefact the hold exists to remove: at 30 d/s the ship covers
>    ~1.3 days of orbit while the job runs, and shifting the curve by that
>    chord puts it beside the trajectory. It is swapped in at its absolute
>    position and time (`allow_rebase=False`) and `_hold_advance` then drops
>    the points whose time has passed — the hold's own mechanism, unchanged.
> 2. **The resume context must be kept, and kept separately.** Dropping it
>    stops `_hold_extend_tail` topping the tail up, so the line is only
>    consumed: measured 10 000 → 6 075 points over 16 frames on 7d/s → 30d/s,
>    a 39 % shorter line that snaps back on the swap. But it cannot simply be
>    left in `_resume_context` either, because **the worker overwrites that
>    the moment it finishes** — one or two frames before the result is swapped
>    in — and extending with the new spacing while the rest of the curve
>    carries the old one breaks both `_display_point_count`'s index fraction
>    and the tangent's minimum chord, which each assume uniform spacing. Hence
>    `_hold_resume_context`, owned solely by `_request_hold_recompute`.
> 3. **`_display_point_count` must measure the curve, not trust `length`.** It
>    derived the drawn fraction from `self.length`, i.e. the last *requested*
>    horizon. During the transition that is already 4x while `points` still
>    holds the 1x curve, so it drew a quarter of it — the line collapsed to
>    **25.0 % / 24.9 % / 24.8 %** on the three upward steps and sprang back on
>    the swap. It now reads the spacing off two adjacent stored samples (still
>    O(1)), taken **mid-curve**, because `points[0]` is the ship and carries a
>    deliberately short first chord.

> **The HUD horizon slider rides `set_display_length` every frame.** It is a
> rate control (`ui/widgets/rate_slider.py::HorizonSlider`): the drawn length
> tracks the knob through `set_display_length` (O(1), no recompute,
> independent of the hold / async swap / thrust), while the *computed* length
> trails via the already-soft `set_length`. No `reset()` on this path — that
> (still on `+` / `-`) is the synchronous `_compute_full` the slider exists to
> avoid.
>
> **The clip is set to `drawn` in EVERY frame, never to `None`.** It belongs to
> the length the player asked for, not to the one currently requested of the
> integrator. The old condition `drawn if wanted > drawn else None` switched it
> off exactly when `wanted` fell back to `drawn` — i.e. on slider *release*,
> while the longer curve was still in `self.points`. For the few frames until
> the shorter job landed, that curve was drawn unclipped: the line jumped to
> the ceiling length and back. `_display_point_count()` returns `None` for a
> curve that is not longer than the clip anyway, so the permanent clip costs
> nothing. Its `limit >= spacing * (n - 1 - q)` test carries **one quantum of
> slack** for the same reason — `spacing` is a single mid-curve chord, i.e. an
> estimate, and without slack the comparison tips on an exactly-fitting curve
> and the quantum rounding shaves up to `q` points (8 × 1 Mm) off the end.
>
> **While grabbing, the computed length climbs a coarse RATCHET — it never
> pins to the slider ceiling.** `ship/horizon.py::horizon_targets()` takes
> `current_length` and only ever raises it, to the next rung of a ladder
> anchored at `PREDICTOR_BASE_LENGTH` with factor
> `predictor.horizon_grab_step_factor` (4.0); dragging inward changes it not at
> all, because a too-long curve is exactly what the clip removes. Two things
> forbid the alternatives. Per-frame `set_length` is out because it
> `_cancel_pending_job()`s, so no curve ever lands and the line freezes for the
> whole drag. Pinning to the ceiling — the original design — is out because it
> does not scale: `predictor.horizon_slider_max_mult` is now **256.0**, and a
> pin would make every touch of the knob integrate a 2.56e12 m horizon even for
> a nudge from 1× to 1.2×; it was also why the readout sat on the ceiling for
> the whole drag. Measured with the ratchet: a full 1×→256× sweep costs **4–5**
> `set_length` calls, an inward sweep **0**, and steady-state cost tracks the
> horizon actually dialled in. Guarded by `tests/horizon_targets_test.py` §2–4.
>
> **The slider max is the player's ceiling, not the point budget's.**
> `HORIZON_MULT_MAX` is read straight from config; past `max_num_points ×
> base_spacing / base_length` (4×) the spacing coarsens, which is the same
> coarsening `+` has always allowed. What §23 forbids is coarsening from the
> *warp* factor, and that clamp still lives in `predictor_horizon_lengths()`,
> independent of this number.
>
> **The drawn point count is quantised so a slow drag does not churn
> `id(points)`.** `get_points()` returns `self.points[:count]`; every frame of a
> drag `set_display_length` moves the raw metres a little, so an un-quantised
> `count` shifts ~every frame → a fresh view object → the renderer's
> line-subdivision cache and `get_apsis_markers()` (both keyed on `id(pts)`)
> miss every frame. `_display_point_count` rounds `count` to a multiple of
> `self._display_quantum` (`predictor.display_length_quantum_points`, default 8)
> at **both** return sites, *after* the `limit is None` / `limit >= full` early
> returns (those still mean "draw everything" and are untouched). Paired with
> this, `set_display_length()` no longer calls `_clear_display_view()` — that
> hard reset forced a rebuild every frame regardless of the rounded count;
> `get_points()`'s own three-way check (base identity, `_display_view_limit`,
> `None`) already invalidates correctly.
>
> §17 is the only warp section that both clips (`_w17_frame`) and draws via
> `get_points()` (its `min_drawn` fraction), and it moves by ≤0.08 % — under
> its threshold, SERIAL baseline still `FEHLGESCHLAGEN: 2`. §23 clips too but
> scans `p.points` directly. §18/§19/§21 set no `display_length` at all.

> **Real time now uses the hold's mechanism too — the Ap/Pe markers were
> jittering because it did not.** Fixed 2026-08-25. Outside the warp hold
> `_anchor_first_point` translated the whole curve rigidly so its head sat on
> the ship. The shift is not one frame of motion but the **whole age of the
> snapshot** (`max_async_wall_age` allows 1.5 s of real time, i.e. up to 90
> sim-seconds of orbital motion at 60 s/s), and the reference body does *not*
> move with it. What is left over is the ship↔body **relative** motion: the
> entire conic sits that far to one side of the body, which is precisely the
> periapsis height. Because the age tracks the compute latency, the displayed
> Pe/Ap distance moved with it — reported as **500 against 510 km** on a lunar
> flyby, flickering frame to frame. Under time warp it never appeared, for the
> one reason that the hold already consumed the curve instead of shifting it.
>
> `_advance_points_along_curve` is that mechanism, now shared by both paths: a
> `searchsorted` over the time column drops the samples whose time has passed,
> the rest stays put in **position and time**, and the ship is prepended as the
> new head. The search uses `side='right'`, not `'left'` — a sample landing
> exactly on `now` is the present, and the present is the head we are about to
> prepend; leaving it would put two points on top of each other and give the
> first segment length zero, tangent included, which is what the navball hangs
> off. Exact equality is the *normal* case here, not an edge case: a freshly
> computed curve begins at `ship@world.time` by construction and
> `_anchor_first_point` runs immediately after.
>
> The head carries the **ship's own** velocity as its tangent. It used to
> inherit the next sample's, so the continuously shortening first segment
> carried a tangent belonging to a different point of the orbit.
> `tests/apsis_stability_test.py` measures the marker's standstill against the
> distance the ship covers in the same time — that difference *is* what the
> rigid shift was off by.

> **The head is prepended, not trimmed to.** Samples can only be dropped
> whole — there is no half a sample. Leaving the next sample *ahead* of the
> ship as `points[0]` and pulling the front of the curve back onto the ship
> made that pull-back run from 0 to a **full point spacing** and reset each
> time a sample was consumed: a sawtooth whose amplitude is a *world* length,
> so zoom magnified artefact and segments alike and the line appeared to
> advance in steps at **every** zoom level. `_hold_advance` now leaves the
> remaining samples completely untouched and prepends the ship's own position
> (at `world.time`) as the new head, so the first segment is a genuine partial
> chord that shortens continuously until the next sample is consumed.
> Measured: head-to-ship distance **exactly 0**, and every sample behind the
> head **bit-identical** frame to frame. `_synthetic_head` (renamed from
> `_hold_synthetic_head` when real time adopted the same mechanism) marks that
> head so the next frame strips it before searching, otherwise the array would
> grow by one point per frame.
>
> **The no-op case must not copy.** In real time the samples sit hundreds of
> kilometres apart while a frame advances a fraction of that, so for many
> frames in a row *no* sample falls due. Then only the existing head is
> updated in place — which not only saves the copy but keeps the **identity**
> of the array, and two caches hang off that: the apsis scan (keyed on
> `id(points)`) and the renderer's line sampling
> (`_make_prediction_line_cache_key`). Both would run dry every frame
> otherwise. The markers may stand still through it because the scan skips the
> head anyway and nothing behind it moved.

> **The hold needs an upper bound on how far the ship may drift off the curve.**
> The failsafe measured the gap **along** the path and allowed four point
> spacings, so a *lateral* offset walked straight past it — and the supply
> never runs low either, because `_hold_extend_tail` keeps topping the tail up
> (measured 0 full recomputes in 3000 frames). A held curve could therefore be
> computed once and never checked against the world again.
> `_hold_advance` now measures the perpendicular distance from the ship to the
> first real chord and, past `hold_drift_max_px` (0.5 px, converted through
> `_view_scale` — a world length means nothing without a zoom), asks for a new
> curve the same way a warp-step change does: **asynchronously**, one job at a
> time, old curve stays on screen. `hold_drift_max_px = 0` switches it off.
> With the RKN4 order fixed above it almost never fires (measured worst lateral
> offset 1589 m = 0.016 px over 1.5 revolutions at 1 h/s), which is where a
> safety net belongs.

> **The synthetic head is not a sample of the curve, and the apsis scan must
> not read it as one.** `_hold_advance` prepends the ship's *world* position as
> `points[0]`; it is off the held curve by whatever the two disagree about —
> measured 37 km against a 1.3 km regular point step. Seeded as the trend
> scan's starting value, that jump flips the direction and the scan reports an
> extremum at index 1: **an Ap/Pe flag sitting on the ship**, blinking on and
> off from frame to frame as the jump crosses the hysteresis or not.
> `_find_apsis_markers_numba` therefore takes `skip_head`, starts its trend at
> that index and suppresses `best_idx <= skip_head`; `get_apsis_markers()`
> passes 1 exactly while `_synthetic_head` is set, so **without a synthetic head
> the scan is bit-identical to before**.
>
> A second, independent source of the same flicker: the soft-stale path served
> cached markers for up to `apsis_hold_rescan_s` **across a swap**, i.e. on a
> curve that no longer existed (measured a flag at r = 3.71e7 m while the ship
> stood at 3.79e7 m and still climbing). Every replacement of `self.points` by
> a freshly computed array now hard-invalidates, and the soft path additionally
> checks `_points_generation` — filtering is only allowed on the very curve the
> markers were scanned from. §18/§19 cover both.

> **`reset()` must clear `_last_swapped_snapshot`, or the line never comes
> back.** Fixed 2026-08-16. `predictor.reset()` hangs off `9`/`0`/`+`/`-` in
> `runtime/input.py`. It emptied the points but left behind the record of *which ship
> state produced them* — a record only a successful swap ever refreshes.
> `update()` compares the live ship velocity against that record to detect
> thrust, so with it frozen the difference grew every frame (~24 m/s per
> frame in warp), and every frame it bumped `_trajectory_version`, cancelled
> the in-flight job and resubmitted — while the job needs more than one frame
> to finish. Measured: **20 frames, 20 jobs submitted, 0 swapped, the line
> never returned.** With the record cleared it is back in **2 frames** and the
> version stays put. This is what made the attitude ring jump: with no line,
> `apparent_orbital_directions` falls back from the drawn-polyline tangent to
> the straight-coast finite difference, so every marker moves to a different
> source. `set_precision`/`set_length` additionally call `invalidate_hold()` —
> without it the warp hold swallowed the change completely (measured: point
> count and heading both moved by exactly 0).

> **Thrust REQUESTS a recompute, it does not force one on the main thread.**
> Fixed 2026-08-17. Thrust pushes the ship's velocity past
> `snapshot_velocity_*_tol` in *every* frame, and both thrust detectors
> (`_handle_trajectory_branch_change`, and the `_last_swapped_snapshot` check
> in `update()`) responded by bumping `_trajectory_version`, cancelling the
> in-flight job, clearing the points and running `_compute_full`
> **synchronously**. With the full solar system that is **0.12 ms coasting
> against 59 ms under thrust** — ~14 fps for as long as the arrow key is held,
> which is the "game lags very badly when I apply thrust" report. It was also
> self-defeating: the version bumped again before any job could finish, so
> under sustained thrust **0 async results were ever swapped in** — the exact
> pattern the `reset()` note above describes.
>
> `_request_thrust_recompute()` replaces both hard paths with a **coalesced**
> one: if a job is already running, do nothing (it is already fresher than the
> drawn line); otherwise submit exactly one. The old line stays on screen and
> keeps being glued to the ship by `_anchor_first_point`. Measured: main-thread
> `update()` **59 ms → 0.22 ms**, ~19 results swapped per second during the
> burn (was 0), `_trajectory_version` constant, point count never 0.
>
> **What this trades away, measured.** The drawn line now lags the burn by one
> compute. Against a synchronous reference at full throttle (600 m/s²) the
> worst deviation is 7.8e6 m at the far end of a 1e10 m horizon — but it grows
> *quadratically* along the arc, so in screen terms it is **0.001 px over the
> first 1 % of the line, 0.14 px over the first 25 %, 0.57 px with the whole
> horizon in frame**: sub-pixel at every zoom, and below the renderer's own
> `_prediction_error_budget` (0.3 px). Coasting is unaffected (the async path
> already carried ~70 km of latency there). After burnout the line is back to
> the coasting noise floor in **1–2 frames**, because the second detector keeps
> requesting until `_last_swapped_snapshot` matches the live velocity again.
>
> Two things must stay. A **position** discontinuity (teleport, reparenting)
> is still a hard invalidation — that one really does make the old line
> garbage. And the coalesce only applies **when a line exists**: with no
> points it falls back to the hard/sync path, preserving the "update() builds
> one synchronously if none exists" guarantee. `tests/warp_predictor_test.py`
> §9 asserts all of it and fails on 4 of 5 checks against the old behaviour.

> **96 % of a prediction is placing the 28 bodies, not integrating the ship.**
> Measured 2026-08-17: the same compute is **61.7 ms with moving bodies and
> 0.6 ms with them frozen**. Every acceleration evaluation asked each body for
> its position at time `t`, and each of those ran a full Kepler solve. Three
> kinds of pure repetition sat inside that, and `body_memo` — a `(n,10)`
> scratch array threaded down the `_time_` kernel chain — removes all three
> **bit-identically**:
>
> 1. **Same body, same time, asked again.** Columns `[t, x, y, valid]` per
>    body; a hit needs an exact time match, so the returned value is the one
>    that same code path just produced. Step-doubling alone evaluates
>    `{t, t+h/4, t+h/2, t+3h/4, t+h}` across three RKN4 steps — 12 stage
>    evaluations over 5 distinct times.
> 2. **Every moon re-solved its parent.** Saturn was solved six times per
>    evaluation, once for itself and once per moon. The descent now stores each
>    chain link, and the ascent stops at any ancestor already known for that
>    time.
> 3. **Time-independent orbit constants, recomputed every call.** `M₀`, mean
>    motion, `√(1-e²)`, `cos/sin(arg)` depend only on the elements —
>    8 of ~19 transcendental ops. `_body_kepler_constants_numba` computes them
>    **word for word** as the inline path did, once per run, into columns 4–9.
>
> A fourth fix was not about repetition: the function allocated
> `np.empty(n)` for the parent chain **on every call** — over 200 000 heap
> allocations per prediction. The chain is at most three links, so the descent
> re-walks the parent pointers instead. That one change alone was 30 ms → 17 ms.
>
> Together: **61.7 ms → 17.0 ms (3.6×)**, and the line's refresh rate during a
> burn goes from 12.7 to **48 per second**. `Predictor.use_body_memo = False`
> restores the old path for A/B checks (same role as
> `world.use_fast_integrator`); `tests/warp_predictor_test.py` §10 asserts
> `array_equal` across five configurations including a two-link moon chain.
>
> **Never hand a numba kernel a module-level numpy array.** Numba types a
> global array as a compile-time constant, i.e. **readonly** — and then any
> store into it, *even one that is unreachable at runtime*, fails type
> inference and takes the whole calling kernel down with it. A module-level
> `_NO_BODY_MEMO` was enough to make `_find_apsis_markers_numba` raise
> `NumbaTypeError`; `get_apsis_markers()` caught the exception and returned
> zero markers, so **the Ap/Pe diamonds and their HUD readouts silently
> disappeared** while everything else kept working. Empty memos now come from
> `_no_body_memo()` (Python callers) or a locally allocated
> `np.zeros((0, 10))` (inside a kernel, which cannot call Python). The
> `except` around the scan now reports once instead of swallowing, and
> `tests/warp_predictor_test.py` §11 checks Pe/Ap against an analytic
> e = 0.5 ellipse around Erde (measured 8.0002e6 m vs 8.0e6, 2.4002e7 vs
> 2.4e7).

> **Never use a NaN sentinel in these kernels.** The obvious design — leave the
> time column `NaN` and rely on `NaN == t` being false — is silently broken:
> every kernel here is `@njit(fastmath=True)`, which enables LLVM's `nnan`,
> i.e. the promise that no NaN occurs, so the comparison may be folded to
> true. Measured: the very first lookup reported a hit and returned the
> uninitialised zeros, putting every body at the origin and moving the
> trajectory by 1.8e6 m. Hence the explicit `valid` column. Note this fails
> *quietly* — the line still looks like a trajectory.
>
> **Cost is linear in the horizon**, because it is set by the number of
> integration steps: 17 / 37 / 74 / 109 ms at 1× / 2× / 4× / 8×. Each `+`
> press doubles the horizon and therefore halves the burn refresh rate.
> `rkn_adaptive_far_maxdt` is what stops that being unbounded — see the next
> note for how it decides.

> **The step ceiling must be sized by the horizon's TIME SPAN, and the
> instantaneous speed is not a measure of it.** Fixed 2026-08-18. The far-field
> step cap exists so a long look-ahead costs a bounded number of steps:
> `desired = (horizon_arc / speed) / rkn_far_field_target_steps`. On a circular
> orbit that is right. On an **eccentric** one `speed` is the orbit's *maximum*
> at periapsis and its *minimum* at apoapsis, so the same arc is estimated to
> take wildly different times depending only on where the ship happens to
> stand — and at periapsis the estimate is too **short**, so the cap is too
> tight and the run costs a multiple. Measured on Pe 29 Gm / Ap 129 Gm at a
> 3.2e11 m horizon (~5 `+` presses, the whole ellipse in frame):
>
> | | steps | compute | max dt used |
> |---|---|---|---|
> | periapsis | 6663 | **295 ms** | 1500 (pinned at the floor) |
> | apoapsis | 1160 | 43 ms | 6562 |
>
> Same orbit, same arc, **6×** — and the budget of 2500 was hit by neither.
> End-to-end at 55 fps under full throttle, fraction of frames that get a
> freshly computed line: **apoapsis 98 %, periapsis 27.5 %**, with gaps of
> median 2 but **up to 18 frames**. That irregularity is the whole report — a
> uniformly low rate reads as low fps, an occasional 0.3 s freeze reads as lag.
> It is also why the same burn at apoapsis felt fine.
>
> The honest quantity is the **mean inverse speed over the arc**, and the last
> run measured it exactly: its time span over its arc length.
> `_record_horizon_time_per_arc` stores that ratio (not the span, so a `+`/`-`
> on the horizon carries over), `_make_snapshot` uses it, and the old estimator
> survives only as the cold-start fallback. **There is no feedback loop**: the
> span is a property of the orbit, not of the step size, so a bigger cap does
> not change it — it is a fixed point, reached in one compute.
>
> After: **1252 steps at periapsis, 1251 at apoapsis** — compute flat at ~51 ms
> either side, 295 → 51 ms (5.8×) where it hurt, and 97.5 % of frames get a
> fresh line (was 27.5 %). Apoapsis is unchanged at 98 %.
>
> Two things to keep. `rkn_far_field_target_steps` was **re-tuned 2500 → 1250
> at the same time**, and that is not a separate accuracy cut: with the honest
> estimator, 2500 would have *doubled* apoapsis cost (1160 → 2500 steps) to buy
> accuracy nobody was missing. 1250 reproduces the apoapsis behaviour that was
> already shipping and makes periapsis match it. The accuracy that buys, at
> that horizon, against a `max_dt = 300` reference: 56 m over the first 1 % of
> the arc, 78 km over the whole 320 Gm of it — **2.7e-4 px** at the zoom in the
> screenshot. And **short horizons are untouched**, bit-identically: the floor
> `rkn_max_dt` (1500 s) still binds at 1× and 8×, so the default configuration
> computes the same floats it did before. `tests/warp_predictor_test.py` §15
> asserts the periapsis/apoapsis step ratio (fails at 5.80 against the old
> code), that both land on the budget, that the fed-back number really is the
> span, that `reset()` clears it, and the bit-identity at the default horizon.

> **The horizon-scaled step ceiling must not step over the orbit — and the
> orbit clamp is LOCAL, evaluated per step, not once at the snapshot.**
> `rkn_adaptive_far_maxdt` raises `max_dt` with the *horizon* so a long
> look-ahead costs a bounded number of steps. It knows nothing about the orbit,
> though, so after a few `+` presses one step covers a real fraction of the
> orbital period, and from there the ceiling — not the error control — sets the
> step. Measured in an Earth orbit (rp 2e7 m, e = 0.6, T = 97 h) at 64x
> horizon: the line moved up to **6.0e7 m** against the same run with the fixed
> 1500 s ceiling — further than the orbit is wide, i.e. it was drawing a
> different trajectory. The clamp is
> `characteristic_timescale / rkn_max_dt_timescale_divisor` (30), the same
> `sqrt(r/|g|)` = T/2pi the warp limiter already uses, floored at the preset
> `rkn_max_dt` so the near field is never made *stricter* than it was.
>
> **`_make_snapshot` used to compute that clamp once, from where the ship stood
> when the snapshot was taken, and apply it to the whole run.** That is right
> only for a trajectory that stays in one gravitational regime, and every
> interplanetary departure leaves one. On a Hohmann transfer Earth → Jupiter
> the ship is in a parking orbit at t = 0, so `t_char/30` = **31 s**, floored to
> 1500 s — and that 1500 s then governed the **2.85 years of heliocentric
> cruise** as well, where the error control happily takes tens of thousands of
> seconds. Measured at a 128x horizon (1.28e12 m, the arc that just reaches
> Jupiter), 28 bodies:
>
> | | steps | compute |
> |---|---|---|
> | global clamp (was) | 56 423 | **1936 ms** |
> | local clamp (is) | 1 280 | **50 ms** |
>
> Same trajectory: **5.5e5 m apart on a 1.28e12 m arc = 0.0006 px** with the
> whole arc in frame, against the renderer's own 0.3 px budget. That 1936 ms is
> the report "the prediction takes 2000–3000 ms and the line can't be drawn or
> keep up with a burn" — and note the cost was **entirely in the far field**,
> where it bought nothing.
>
> `_local_timescale_numba` is the same formula in numba form (minimum over all
> gravity sources of `sqrt(r³/(G m)`) — **never `argmax(g)`**, for the reason
> `.claude/rules/physics-world.md` sets out), and
> `_rkn_adaptive_step_time_numba` calls it once per step before k1. **The order
> is what makes it free**: it evaluates the bodies at `local_t`, exactly the
> time k1 is about to need them, so it *warms* `body_memo` rather than adding
> Kepler solves — what is left is 28 square roots per step against 9×28 Kepler
> solves. Measured overhead where the clamp changes nothing: **≤ 2 %**.
>
> Both directions are asserted, and both matter. On any orbit that stays in its
> regime the local path is **bit-identical** — LEO at 1x and 64x, e = 0.6 at
> 64x, and a lunar transfer at 4x all give the same step count and 0.000e+00
> deviation, because the clamp there sits below the `rkn_max_dt` floor and
> never binds. `Predictor.use_local_step_ceiling = False` restores the global
> form for A/B checks (same role as `use_body_memo`). §20 and §24.

> **`rkn_max_dt_ceiling` was 30000 → 120000 s once the clamp went local**, and
> that is not a loosening. With the clamp global it was the *only* guard left
> on a departure trajectory; local, two **physical** bounds already bind —
> `desired` (the horizon's own step budget) and `t_char_local/30` (the orbit
> you are actually near) — and the absolute number was merely the tightest of
> the three. Removing that redundancy at the Jupiter horizon: **2848 → 1280
> steps, 103 → 50 ms**, where 1280 *is* `rkn_far_field_target_steps`, so the
> value saturates — 300000 s measures identically. Accuracy against a run with
> `max_dt` pinned to 300 s: **2.501e6 → 2.665e6 m** over 1.28e12 m, i.e.
> 0.0025 → 0.0027 px. The near field is untouched and bit-identical (LEO, e=0.6
> and the lunar transfer all measure 0.000e+00 between 30 000 s and 300 000 s),
> because the orbit clamp there is two orders of magnitude lower than either.
>
> **The point budget therefore grows with the horizon** (2026-08-21).
> `_horizon_spacing_floor()` is `length / num_points`, so with a fixed budget
> every `+` doubled the arc *and* the spacing — samples per revolution halved
> on every press. In a 2e7 m Earth orbit that is 180 per revolution at base,
> 22 at 8x, **5.6 at 32x**, and at 5.6 the cubic Hermite between two samples
> is simply not the orbit any more: the line reads as bulges with corners
> between them. `apply_predictor_horizon` now scales `num_points` to hold the
> spacing at its base value, capped by `predictor.max_num_points` (40 000).
> Measured at 64x in that orbit: **207 samples per revolution instead of 52**,
> one compute 561 → 651 ms (+16 %, on a worker), array 0.4 → 1.6 MB.
>
> Three things had to follow, each of them a hitch on the *main* thread that
> only appears because the warp step moves the horizon on every change:
>
> 1. **`set_num_points` needs a soft form.** Its `reset()` empties the line,
>    and `update()` then rebuilds one synchronously — the very cost
>    `set_length(soft)` already removed. `soft=True` marks the hold instead
>    (same reasoning: a curve with the wrong point *count* is not wrong, only
>    over- or under-supplied).
> 2. **The tail may not grow by the whole jump in one frame.**
>    `_hold_extend_tail(30000)` on the 7d/s → 30d/s step measured **40.3 ms**
>    against 0.3 ms neighbours. `hold_extend_max_points` (1000) spreads it;
>    the normal case tops up ~170 points and never sees the cap.
> 3. **The hold's refresh threshold is measured against the budget**, so a
>    budget jump alone pushed a still-full curve under it and forced a
>    synchronous recompute. While a replacement is already in flight
>    (`_hold_pending_swap`) the threshold drops to an absolute emergency
>    value — the line cannot run dry in that state anyway.
>
> Worst switch frame across the whole ladder after all three: **2.0 ms**.

> **Thrust is detected by SUBTRACTING gravity, not by out-shouting it.**
> Fixed 2026-08-17. `_handle_trajectory_branch_change` used to compare the
> whole velocity jump against a threshold sized like gravity itself
> (`max(tol, 4·|g|·dt)`). Far from a planet that works; **near periapsis it
> goes blind**: on an e = 0.7 orbit around Erde, `|g| = 8.1 m/s²` gives a
> 65 m/s threshold over a 2 s step, while full throttle adds only 6.7 m/s per
> frame. So no recompute was requested exactly where the trajectory changes
> fastest. The line was then kept alive only by the generic
> `recompute_every_update` resubmit, which is **single-flight** — the thrust
> pipeline switched off. Measured at 4× horizon: **66 → 18 refreshes/s on
> approach to periapsis**, snapping back on the way out. That is the "jumpy,
> laggy near periapsis" report; it is a detector bug, not an integrator cost
> (compute time is flat 27–29 ms all the way round the orbit).
>
> The test now measures the **residual**: `|Δv − g·dt|`, i.e. what is left
> after gravity explains what it can. Per frame on that orbit:
>
> | | coasting | thrusting |
> |---|---|---|
> | periapsis (7.0 Mm) | 0.023 m/s | 6.69 m/s |
> | apoapsis (39.7 Mm) | 0.000 m/s | 6.67 m/s |
>
> The 1 m/s tolerance now sits ~290× above the coasting noise and ~7× below
> the thrust signal, everywhere. It also catches a case the old rule missed
> even without the gravity margin: thrust opposing gravity, where the *total*
> jump is 0.98 m/s (under tolerance) but the residual is 6.66 m/s.
>
> **The threshold must scale with the CURVATURE of g, not with g.** Over a
> warp step `g·dt` no longer explains the change — at 7 d/s the residual
> reaches 28 000 m/s. `4·|g_now − g_prev|·dt` grows with exactly that error
> (3.4e6 m/s at the same step), so the detector stays quiet from 0.5 s to
> 100 800 s steps and the warp hold is never torn up. `g_prev` is last
> frame's value, so this costs nothing; when it is missing the code falls
> back to the old, permissive `4·|g|·dt`. §13 checks both directions —
> thrust detected 12/12 frames and coasting 0/12 at four points around the
> orbit, and 0/6 under a 7 d/s warp step.

> **Results are consumed at a PACE, not as fast as they arrive.** The
> pipeline's runs are *started* evenly (one per frame) but do not *finish*
> evenly — compute time jitters. Taking the newest finished result each frame
> turns that into stall-then-double-step: measured under full throttle at
> periapsis, one frame in ~70 advances two jobs at once, its shape change is
> twice its neighbours', and the displayed snapshot's age drops 6 s → 4 s in
> that frame. A uniformly low refresh rate is invisible; one outlier per
> second is not — this is the "feels like high network ping, not like low
> fps" report, and it is the same problem a network jitter buffer solves.
>
> `_swap_ready_result` therefore takes the **oldest** finished result and lets
> up to `predictor.swap_backlog_max` wait, skipping ahead only past that.
> The trade is explicit and was measured at periapsis, per 300 frames, with
> the drawn line compared against a synchronous reference:
>
> | backlog | double-steps | age | deviation |
> |---|---|---|---|
> | 0 | 4 | 2 s | 8.4 px |
> | **1 (default)** | **1** | **4 s** | **10.3 px** |
> | 2 | 0 | 6 s | 17.5 px |
>
> Note the deviation baseline: near periapsis one compute of latency is
> already ~8 px, not the sub-pixel figure that holds far from a planet — the
> orbit is far more Δv-sensitive there. So the buffer is not free, and 1 is
> the chosen compromise (three quarters of the outliers for ~2 px).
>
> **Stalls cannot be removed at all**, only doubles. Arrivals can never exceed
> one per frame, because a job is only worth starting once the ship state has
> advanced — submitting twice in one frame would just compute the same
> snapshot twice. Total arrivals = frames − stalls + doubles, so the two are
> the same phenomenon seen from either side. A one-frame hold is barely
> visible; a double-step is.

> **Under thrust the line is refreshed by THROUGHPUT, not by lower latency.**
> One prediction takes ~17 ms; one frame takes ~11 ms. Computed one after
> another the line can therefore be new at best every other frame — that is
> the chunkiness during a burn, and no amount of shaving the single compute
> fixes it, because the two numbers are within 2× of each other.
> `predictor.thrust_pipeline_depth` (6) caps how many computes may run at
> once. Since at most one is submitted per frame they *start* one frame apart
> and so *finish* one frame apart. Latency is untouched; each individual line
> is still one compute old.
>
> **The depth is derived, not chosen.** `_target_pipeline_depth()` =
> `ceil(last_compute_ms / frame_ms) + 1`, capped. A fixed number cannot work,
> because the compute cost scales with the horizon while the frame time does
> not: 3 was right at 1× and far too small at 4× (3 runs / 74 ms = 40
> refreshes against 90 frames). The frame time is measured by the predictor
> itself from the gap between its own `update()` calls (EMA) — nothing hands
> it a frame dt. Measured at 84 fps with an 8 ms main-thread load, fraction of
> frames that get a fresh line:
>
> | horizon | depth 1 | fixed 3 | adaptive (cap 6) |
> |---|---|---|---|
> | 1× | 50 % | 99 % | 99 % (depth 3) |
> | 2× | 30 % | 72 % | 92 % (depth 5) |
> | 4× | 16 % | 38 % | 63 % (depth 6) |
> | 8× | 11 % | 26 % | 39 % (depth 6) |
>
> Frame rate was flat at 83.5–84.6 fps across every row — the cost lands on
> idle cores, not on the frame. Two caveats: concurrency makes each *single*
> compute slower (67 → 87 ms at 4× depth 6, memory bandwidth), so throughput
> scales sub-linearly and raising the cap past ~6 buys little; and at 4× and
> beyond the honest fix is a shorter horizon, not more threads.
>
> This is only allowed because every kernel is `nogil=True` — the workers use
> other cores instead of taking the GIL. `_ensure_executor` sizes the pool
> `min(depth, cpu_count-1)` to leave a core for the main thread. Main-thread
> `update()` goes 0.33 → 0.81 ms (the extra is swapping the line in more
> often, not building snapshots — those are 0.046 ms). Coasting is
> deliberately left at one job in flight; the depth applies to thrust only.
>
> **Out-of-order completion is the real hazard here**, and it is guarded, not
> hoped for: `_swap_ready_result` takes the **newest** finished job, discards
> finished ones older than it, leaves running ones alone, and refuses any job
> id below `_last_swapped_job_id`. Without that a late-finishing older result
> would drag the line back to a pre-burn shape — the same visual failure the
> warp hold exists to prevent. §12 asserts strictly ascending job ids *and*
> strictly increasing snapshot velocity over a 2 s burn (measured 356 swaps,
> 0 backward steps, queue never above the cap).
>
> Two counting rules matter. The in-flight check must count **running** jobs,
> not `len(_pending_futures)` — that list also holds finished-but-not-yet-swapped
> results, which occupy no worker, and counting them blocked new work in
> exactly the frames where one had just finished. And `_swap_ready_result`
> takes at most one result per frame by construction, so the refresh rate can
> never exceed the frame rate — which is the point: `pred_hz` in the TIMING
> line should read close to the fps, and `pipe` shows the depth in use.

> **The held line rolls: consumed at the front, extended at the back.**
> Consuming alone makes the line *shrink* until the refresh snaps it back to
> full length — it pulses in step with the refresh. `_hold_extend_tail()`
> therefore appends exactly as many points as were consumed, so the horizon
> stays constant (measured **1.000 .. 1.000** of the horizon at 1h/s, 1d/s
> and 7d/s) and full recomputes drop to **1 in 240 frames**. Cost falls with
> it: 0.08–0.16 ms median instead of periodic 8 ms spikes.
>
> This is a genuine continuation, not a new run: `_resume_context` keeps the
> **original snapshot** (its body arrays are propagated analytically from
> *its* epoch — continuing against a fresher snapshot would be a different
> integration), plus the integrator state, step size and time origin. The
> kernel gained `init_t` / `init_accumulated` / `init_proposed_dt` and returns
> the resume state in `stats[7:14]`; passing zeros reproduces the old
> behaviour exactly, which is why the golden line comparison still matches.
>
> **The resume point is the last EMITTED point, not the end of the last
> integration step.** Two bugs live here, both found by measuring the joint.
> (1) When the emission loop stopped because the point budget was full, the
> segment's leftover arc was never added to `accumulated` — harmless for a
> one-shot run (nothing reads it afterwards) but it put the resume point up to
> a whole step behind. (2) Fixing that alone let `accumulated` exceed
> `precision`, which makes `distance_to_place` negative on resume and collapses
> the placement maths. Resuming from the emitted point makes the leftover zero
> by definition. Its velocity comes from differentiating the *same* Hermite
> polynomial used to interpolate its position, so it is consistent rather than
> approximated. Result: the largest segment equals the median spacing exactly
> (**factor 1.00**) — no seam.
> **`update_dynamics` comes BEFORE `update_planets`, and that is not a
> stylistic choice.** `update_planets(dt)` winds `body.theta` forward by a full
> chunk and bookmarks `self.time` as the epoch of that new angle. That is only
> true if `update_dynamics(dt)` already moved the clock to the end of the
> chunk — the order `physics/world.py::step` uses. Swap the two calls and the bookmark
> is one chunk off, so `position_at_time(t)` hands back the position at
> `t + dt`: **every scripted body sits one chunk in the future for the ship's
> force evaluation.** The error is first order in the chunk, and the chunk is
> `max_substep_seconds` = 1000 s (more under warp). Measured in an Earth orbit
> (rp 2e7 m, e = 0.3), distance between the world and the analytically
> propagated prediction line after 4800 s:
>
> | chunk | 1000 s | 300 s | 5 s |
> |---|---|---|---|
> | `dynamics, planets` (the game) | 5.2e1 m | 5.2e1 m | 5.2e1 m |
> | `planets, dynamics` | 9.4e6 m | 3.9e6 m | 7.4e4 m |
>
> The game has always had it right; `tests/warp_predictor_test.py::advance`
> had it backwards and was therefore measuring the hold against a world the
> game does not compute. Fixed 2026-08-21, and §18 now asserts both the
> chunk-independence and that the swapped order fails.

