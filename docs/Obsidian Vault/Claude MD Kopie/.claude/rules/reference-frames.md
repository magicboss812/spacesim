---
paths:
  - "spacesim/physics/reference_frames.py"
---

# Plotting frames (`physics/reference_frames.py`)

- `physics/reference_frames.py` — Principia-inspired frame system.
  `ReferenceFrameSelector`, `PlottingFrameAdapter`, Kepler helpers. Uses
  astropy + poliastro.

> **The plotting frame's ORIGIN is interpolated cubically, and that is what
> made a long prediction line wavy.** Fixed 2026-08-23. The drawn line is
> `ship(t) − origin(t)`, and the origin — the reference body's position — is
> not propagated per point but interpolated between at most
> `frame_origin_interp_max_knots` (256) exact knots spread over the
> prediction's **time window**. That interpolation used to be a straight
> chord, so the line carried the curvature the *reference body's own orbit*
> was missing: zero at every knot, maximal between them. On screen that is
> evenly spaced bulges with hard **corners** at the knots — and none of it
> comes from the predicted points, which is why neither a bigger draw budget
> nor finer subdivision helped (measured: identical to a draw with the budget
> raised 15×; both drew the same bent curve, just more smoothly).
>
> The size follows the chord formula `R·θ²/8` with `θ = 2π·q / T_reference`.
> The window is the prediction's time span, so `q` — and the error — grows
> with **every `+` press**. For Erde (`R` = 1.496e11 m, `T` = 1 y) at 256
> knots, against a practically exact interpolation:
>
> | horizon | window | predicted | measured |
> |---|---|---|---|
> | 1×   | 3.8 d  | 0.00 px  | 0.00 px |
> | 32×  | 0.33 y | 0.17 px  | 0.19 px |
> | 128× | 1.33 y | 2.66 px  | 6.13 px |
> | 512× | 5.33 y | 42.51 px | **40.68 px** |
>
> `_cubic_4pt` (Lagrange through four knots, evaluated on the middle
> interval) drops that to `~0.023·R·θ⁴`: **40.68 px → 0.54 px at 512×**, and
> 0.00 px at 128× and below. It is **free** — the same knots, two more reads
> from the same `_position_cache`, measured 5.14/5.56/4.35/4.26 ms per frame
> against 5.32/5.17/5.05/4.03 ms before, i.e. inside the noise. Raising the
> knot count instead was measured at **+2.5 to +4.4 ms**, which is why the
> interpolation order, not the knot count, is the lever.
>
> Three things must hold. **Not Catmull-Rom** — its `(p2−p0)/2` slope is a
> chord and therefore short by `sin θ/θ`, making the scheme 3rd order; it
> measured 5.4e6 m against Lagrange's 1.0e6 m at θ = 0.131, i.e. 2.44 px
> against 0.54 px on screen. The Lagrange form reproduces any cubic exactly.
> **The scalar and batch paths share the one arithmetic expression**
> (`_cubic_4pt` is pure arithmetic, so it takes floats and numpy arrays
> alike) — the house rule for every vectorised path here, and
> `_origin_xy_arrays` therefore had to widen its knot grid by one on each
> side, since four knots are now needed per evaluation instead of two.
> **The knot values themselves stay bit-exact**: the basis is exactly
> `(0,1,0,0)` at `s=0` and `(0,0,1,0)` at `s=1`. The price is C0 rather than
> C1 at the knots — a kink is still possible in principle, it is merely 4th
> order small (0.14 px against 42 px). `tests/frame_origin_interp_test.py`
> covers all of it, each check with a counter-check that the old linear path
> fails the same bound (285×–73608× worse).
>
> **A grid that cannot resolve the origin's own orbit is not a grid — then
> stop interpolating (2026-08-31).** The note above raises the *order*; this
> one is about the case where no order helps. `q = span/256` depends only on
> the predictor's horizon and knows nothing about the body sitting in the
> origin. For a planet that is harmless (Neptune moves 0.0044 rad in 42 days).
> For a **moon** it is not: at a 3650 d horizon `q` = 14 d against an 8.7 d
> period, so the cubic is interpolating a curve it never sampled. Measured as
> the displacement of the drawn line from the exact value — and therefore
> from the Ap/Pe markers, which are drawn *after* `draw_prediction` clears
> the window and are exact:
>
> | origin | horizon | q/period | line↔marker | per frame |
> |---|---|---|---|---|
> | Neptun | 3650 d | 0.00 | 0.0 px | 0.0 px |
> | Mond | 3650 d | 0.52 | 462.5 px | 480.1 px |
> | Triton | 365 d | 0.24 | 39.2 px | 32.2 px |
>
> The "per frame" column is the report: the window starts at the points
> array's **head time**, which advances every frame under warp, so the knot
> grid slides and the whole line slides with it — at the same absolute time,
> on a bit-identical curve.
>
> More knots cannot fix it: Titania over 3650 d at 16 knots per orbit needs
> 6704 of them, more than the ~4000 points that get projected at all. So when
> `frame_origin_interp_min_knots_per_period` (16) cannot be met,
> `set_origin_interp_window` sets `q = 0` **and** `_origin_exact_batch`, and
> `_origin_xy_arrays` evaluates `_knot_positions_batch` on the point times
> directly. Result **0.000e+00 m** residual in every failing case, planets
> untouched (Neptun 0.29 ms before and after).
>
> **The exact batch must not reconcile against `_position_cache`.** That loop
> exists for the knot grid, where the same ~260 times are asked for again by
> both paths and across frames. Point times are 4000 different ones every
> frame: the loop measured **2.4 ms of the 3.3 ms** and would add 12 000
> cache entries per frame that nothing ever reads back. With
> `reconcile=False` it is **0.85 ms** — and the moon frames that previously
> fell through to the *scalar* loop got faster too (Triton 2.93 → 0.55 ms,
> Titania 3.04 → 0.69 ms), because that fallback was the slowest path of the
> three.
>
> **Measured end to end**, on the real pipeline (full solar system, real
> `Renderer`, real warp hold, ship on a Neptune flyby with periapsis 1.69e8 m,
> 1 d per frame, the *drawn* polyline read back out of
> `_prediction_line_cache_points` and compared against the Pe marker in the
> same image):
>
> | plot frame | horizon | Pe↔line (median/max) | line movement per frame |
> |---|---|---|---|
> | Neptun (planet) | 8× | 0.32 / 0.50 px → **0.32 px** | 0.26 → **0.00 px** |
> | Triton (moon) | 8× | 1.21 / 1.84 px → **0.50 px** | 0.81 → **0.00 px** |
> | Triton (moon) | 64× | 25.0 / 362.8 px → **1.33 px** | 77.6 / 417.5 → **0.00 px** |
>
> Two things to take from the table. The damage scales with the **horizon**
> (the window sets `q`), which is why the report said "especially at a long
> predictor line". And a **planet** in the origin was never affected — a
> future session chasing a wobble in a Sonne- or planet-centred frame should
> look elsewhere, this is not it.
>
> **What this is NOT.** Three plausible suspects were measured and cleared,
> so don't re-investigate them: the predicted *points* are fine (at 512× the
> near field is off by 3.4 km over a 1e10 m arc, 3e-7 relative, 0.00 px);
> the far-field step ceiling `rkn_adaptive_far_maxdt` is irrelevant (turning
> it off changes the drawn error by exactly 0); and the draw budget is not
> binding in the shipped configuration (a 15× budget changes the line by
> 0.00 px). The one regime where the *sampling* does bind is a predictor
> configured with few points — with `num_points = 40` a fixed-zoom sweep
> measured the visible error going 0.10 px → 698 px purely from horizon
> length, because `_prediction_scan_indices` subsamples uniformly over the
> whole array regardless of what is on screen, and
> `_hermite_refine_world`'s `room = budget − len(indices)` then charges the
> subdivision budget for points that the clipper throws away. With the
> shipped `num_points = 10000` neither ever binds (measured 0.00 px and
> +0 ms), so a view-aware scan was built, measured at **zero accuracy gain
> for +2.9 ms per frame**, and deliberately **not** kept.

> **Full-solar-system per-frame costs were killed in four places (2026-08-17),
> all bit-identical to the paths they replace.** With 28 bodies the game sat
> at ~30 fps; these took the frame from ~20 ms to ~6.5 ms real:
> (1) `reference_frames._knot_positions_batch` evaluates the origin-knot grid
> (`_origin_xy_arrays`) as a numpy-vectorised Kepler chain instead of ~260
> pure-Python solves per frame. It reconciles with `_position_cache` (existing
> entries win, new ones are seeded), so scalar and batch paths read literally
> the same floats — measured max diff 0.0 m. **That reconciliation runs over
> Python lists, not over the arrays** (2026-08-27): `qt[j]` and `wx[j] = …` on
> a numpy array cost about an order of magnitude more than list access, and
> the loop covers all ~260 knots for every body of every parent chain and
> frame. Only the positions where the cache actually held a *different* value
> are written back, in one fancy-index store. Nothing is computed differently
> — same values, same order, same cache entries. If a branch can't be batched
> (mixed per-time fallbacks, `debug_ephemeris`) it returns None and the old
> scalar loop runs. The vectorised Newton solve freezes each element the
> iteration its |dE| drops below 1e-10 — the scalar loop's exact sequence.
> (2) `rendering._compact_min_step_numba` / `_rdp_keep_numba` port the
> min-step compaction and RDP simplification to numba, word-for-word (same
> stack order, same comparisons); the Python methods stay as fallback.
> (3) Reference trails are numpy ring buffers (`{'buf', 'n'}`), not deques of
> tuples — drawing slices a view instead of converting 27 lists per frame.
> (4) `world._serialize_for_kernel` caches the structure-static arrays
> (masses, Kepler elements, parent indices) keyed on the body-id and
> parent-id tuples; positions and the `_kepler_ref_*` bookmark are refilled
> per call. Any reparenting (epicycles) changes the key and rebuilds.
> `tests/energy_test.py` still lands on exactly `4.2230e-10`.
