---
paths:
  - "spacesim/physics/world.py"
  - "spacesim/physics/world_kernels.py"
  - "spacesim/bodies/body.py"
  - "spacesim/physics/vec.py"
  - "spacesim/physics/kernels/**"
---

# Physics engine — world, kernels, body model, time warp

- `physics/world.py` — physics engine. `world` class, adaptive RKN4 integrator,
  `update_planets` (scripted bodies) + `update_dynamics` (free bodies),
  `should_release` / `release_body` for orbit handoff.
- `physics/world_kernels.py` — the **Numba fast path for `update_dynamics`**, a
  word-for-word transcription of the Python integrator (same coefficients,
  same step-doubling, same tolerances, same summation order). `physics/world.py`
  keeps the Python version as the reference and falls back to it whenever
  the kernel can't run; `world.use_fast_integrator = False` forces the
  fallback for A/B checks.

> **The fast path must stay bit-identical, and it is tested that way.**
> `update_dynamics` steps in bounded chunks, so its cost grows *linearly* with
> time warp. The kernel is a ~40x speedup over the Python reference (measured
> 2026-08-18 on the 28-body system: **25.5 us per substep against 1014 us**)
> while `tests/warp_predictor_test.py` asserts `|Δpos| = |Δvel| = 0.0` against
> the Python path and `tests/energy_test.py` stays on `4.2571e-07` /
> `4.2230e-10`. Two traps. **`_body_pos_at_time` must stay word-for-word
> `bodies.kepler_relative_xy`** — since 2026-08-27 world and predictor share
> the *same* Kepler model, and this kernel is the world's copy of it. (Until
> then this paragraph said the opposite: "do not reuse the predictor's
> kernels, they solve Kepler while `bodies.position_at_time` uses a
> constant-angular-rate approximation." That difference *was* the bug — see
> the body-model note in `.claude/rules/orbit-lines.md`.) Change one, change
> the other; §2 measures the bit-identity and catches it immediately. And
> **keep the sum in body order** (float addition is not associative; a
> different order changes the energy drift).

> **Per-substep cost is linear in the number of gravity sources — so old
> measurements do not survive a bigger system file.** This paragraph used to
> quote 336 steps at 47.4 ms Python / 0.88 ms kernel. Those were taken when
> `solar_system.json` held **4** bodies; it now holds **28**, and re-measuring
> gave figures **7x larger** across the board — exactly the body-count ratio.
> `_body_pos_at_time` (`physics/world_kernels.py:63`) is called once per body per
> acceleration evaluation, and there are 12 evaluations per substep
> (step-doubling = 3 RKN4 steps x 4 stages), i.e. **324 body placements per
> substep**, each a Kepler-ish solve with ~7 transcendentals. Any timing claim
> here must name the body count it was taken at.
>
> That also means the predictor's `body_memo` win (61.7 -> 17.0 ms, documented
> below) has a **direct unclaimed analogue here**: the same three redundancies
> are present — the same body at the same time re-asked across the 12
> evaluations (only 5 distinct times), every moon re-solving its parent, and
> the time-independent orbit constants recomputed on every call.

> **Time warp is bounded by TWO ceilings and one physical limit.** Added
> 2026-08-18 to make interplanetary transfers reachable (a Hohmann
> Earth-Pluto transfer is ~45 years — 75 minutes of real time at the old
> 7 d/s ceiling, 45 seconds at 1 y/s).
>
> **The two numbers below moved in `config.json` on 2026-08-27**, and every
> measurement in this note (and in the notes above it) was taken at the old
> pair: `integrator_max_step`
> **30 → 3000 s** and `integrator_position_tolerance` **1.0 → 0.10 m**. The
> mechanism is unchanged — a 100x tighter tolerance buys back most of what a
> 100x looser ceiling would have cost, and point 3 below is exactly the
> reason the ceiling may be raised at all (the error control, not the
> ceiling, sets the step near a body). Re-measure before quoting a substep
> count or a millisecond figure from this note against the current config.
> `tests/energy_test.py` is **not** affected — it sets
> `integrator_position_tolerance = 1e99` itself and drives fixed steps, so its
> `4.2571e-07` / `4.2230e-10` still stand.
>
> 1. **`integrator_max_step`** (was 30 s, now 3000 s) caps each substep. At 365 d/s and
>    180 fps the world advances 175 000 sim-s per frame = **5 984 substeps,
>    measured 168.7 ms**. `world.set_warp_step_ceiling(sim_s_per_frame)`
>    raises the ceiling to hit `integrator_warp_substep_target` (40) steps per
>    frame; `effective_max_step()` never returns less than the configured
>    value, so **at real time the integrator computes bit-identical floats**
>    and `energy_test` is untouched by construction.
> 2. **`max_substep_seconds`** (1000 s, `physics/world.py::step`) chunks the
>    frame, and every chunk costs at least one substep — so it silently
>    becomes the binding ceiling. Sweeping `integrator_max_step` alone
>    flattens at 176 substeps however high it goes. The chunk is now
>    `max(MAX_SUBSTEP, ceiling)`. **Raising either alone buys nothing.**
> 3. **The ceiling is safe because the error control, not the ceiling, sets
>    the step near a body.** Measured in a 400 km orbit over 5 orbits: raising
>    the ceiling 30 s -> 100 000 s (3 300x) moves the *actual* step only
>    27.7 s -> 69.3 s, and altitude after 5 orbits is 400.335 km either way —
>    identical, i.e. the raised ceiling does not move the trajectory at all.
>    (Was 34.7 s / +0.374 -> +0.411 km while RKN4 was accidentally 3rd order;
>    4th order earns a larger step at the same tolerance — 1 m at the time of
>    the measurement, 0.10 m since 2026-08-27.) That is why no
>    far-field geometric heuristic (a la `rkn_adaptive_far_maxdt`) is needed
>    here — it would only add a way to get it wrong.
> 4. **Step-size memory.** The loop restarted at the ceiling on every call and
>    re-derived its way down by rejection — measured 1 000 rejections and 1.5x
>    the wall time in LEO. `advance_dynamics` now takes and returns an
>    `h_hint`. It is **active only when the ceiling is above the configured
>    floor**, so the default path stays bit-identical. Implemented in the
>    kernel *and* the Python reference, as the bit-identity rule requires.
> 5. **Some warp is impossible, not merely expensive.** One frame at 1 y/s is
>    48 hours — ~24 orbits of a 2 h orbit. No ceiling integrates that in 40
>    steps (measured 5 120 substeps, 270 ms). So warp is *capped by
>    proximity*, the way KSP does it: `world.characteristic_timescale(ship)` =
>    the **minimum over all bodies** of `sqrt(r_i³ / (G m_i))`, which for a
>    circular orbit around any one of them is exactly `T/2pi` (exact in LEO
>    and at 1 AU). A frame may advance at most
>    `t_char / simulation.warp_timescale_divisor` (3.0). **Never the HUD's
>    orbital period** — that one is solved against the player-selected
>    *reference body*, so someone in Earth orbit with the Sun selected would
>    get a year instead of two hours, and a physics limit would depend on a
>    display setting. Enforced in three places, because `PageUp`/`PageDown`
>    and the dev panel bypass the HUD: `Telemetry.warp_step_allowed` (greys
>    the step out), `Hud._set_warp` (ignores the click),
>    `runtime/loop.py::_clamp_warp` (the backstop).
>
>    > **And never `argmax(g)` — that was the bug, fixed 2026-08-27.** The
>    > body with the largest `G m / r²` is not the body you are orbiting.
>    > Beyond `r ≈ 2.6e8 m` from Earth the **Sun** wins (6.1e-3 against
>    > 4.4e-3 m/s²), while Earth's SOI reaches 9.2e8 m and the **Moon sits at
>    > 3.8e8 m, right inside that band** — so on every lunar transfer the
>    > selection flipped to the Sun and reported *its* timescale. Measured on
>    > `rp 7e6 / ra 4.05e8 m`: the value jumped **1.4e5 → 3.4e6 s (25×)** at
>    > that crossing, fell back to 1.9e4 s at the Moon and rose again behind
>    > it — over a 2 % step in radius a **97× jump**.
>    >
>    > Both consumers broke with it. The warp limiter then permitted
>    > `t_char/3` = 1.18e6 s per frame on an orbit of `T` = 9.3e5 s — **1.57
>    > orbital periods in ONE frame**; same start, 25 days, perigee
>    > **1.107e7 m in real time against 6.76e6 m under warp**. And the
>    > predictor's step ceiling (`rkn_max_dt_timescale_divisor`) went 1500 →
>    > 17254 s and back on the next flip, so the drawn line showed a
>    > different trajectory each time.
>    >
>    > The minimum has no such jump, because it is a minimum of **continuous**
>    > functions — the same step now measures at most 1.03×. It is also
>    > simply more correct: the old `total_g` in the denominator put LEO
>    > 0.03 % and 1 AU 0.5 % off, the minimum hits `T/2pi` exactly.
>    > `tests/warp_predictor_test.py` §21 measures continuity and the
>    > permitted frame step, each with a counter-check that the far field and
>    > LEO are unchanged.
>
> Result at 1 y/s, 28 bodies, 180 fps: world+predictor **172.9 ms -> 4.14 ms
> median** (42x), worst frame 7.64 ms, zero full predictor recomputes.
> `tests/warp_predictor_test.py` §16 covers all of it.

> **At high warp a LONGER prediction horizon is cheaper, not dearer.**
> Counter-intuitive, and measured. The per-frame cost under warp is dominated
> by the **synchronous `_compute_full`** that fires when the held line runs dry
> (`hold_refresh_fraction`), not by the length of the line. At 1 y/s over 600
> frames: 1x horizon -> 277 full recomputes, 4.20 ms median; 64x -> **zero**,
> 0.70 ms. But 256x -> one recompute costing **54 ms**, a visible hitch. So the
> horizon is a product `base x manual x warp`
> (`ship/horizon.py::HorizonPolicy.apply`), the warp factor being a power of two
> **rounded** (flooring puts 1 y/s on 32x, measured 14.7 ms worst frame against
> 4.8 ms at 64x) and **capped at 64**. Powers of two also supply the
> hysteresis: `set_length()` discards the hold, so the value must only move
> when the player changes warp step. `+`/`-` move the *manual* factor, never
> the length directly — otherwise the per-frame rescale overwrites the
> player's input on the very next frame.

> **The warp extension is a SUPPLY, not a picture — and it must not touch the
> drawn line (2026-08-27).** Only the un-warped part is drawn
> (`set_display_length`), so the extension is invisible by construction and
> may therefore change nothing that is visible. It changed two things, both
> through `wanted`:
>
> 1. **The point budget is capped** at `predictor.max_num_points` (40 000).
>    Once the cap binds, every further extension coarsens the **spacing**
>    instead of adding points — measured at manual 8× and 64× warp, the
>    40 000 points on the drawn stretch became **626**, and the drawn curve
>    then missed the same trajectory by **2.3e6 m even with the cubic Hermite
>    evaluation** (6.8e6 m with plain chords).
> 2. **`horizon_arc` in `_make_snapshot` is `points × spacing`**, so the
>    extension raised the far-field step ceiling — 2163 → 8676 s in the same
>    situation — and the *integrated* trajectory moved by **2.3e6 m** (with
>    the ceiling pinned: 8.4e4 m).
>
> Together that is the report "the prediction looks completely different under
> time warp": ~4.6e6 m on a line whose periapsis is 1e7 m, from pressing the
> warp key alone. `ship/horizon.py::predictor_horizon_lengths` therefore caps the
> warp factor at what the **point budget still carries at base spacing**, so
> `wanted ≤ max_num_points × base_spacing`; the spacing — and with it
> `horizon_arc` and the ceiling — stays exactly the real-time one. If the
> player has already gone past the budget with `+`, the warp factor drops to
> 1: *their* coarsening stands (it is intended and documented), warp's is not
> added on top.
>
> Measured after, against the real-time line from the same ship state: **0.0
> m** at manual 1×/4×/8× and 16×/64× warp alike. And it is **cheaper**, not
> dearer — one compute at manual 8×, 1 y/s goes **847.6 → 61.2 ms**, and over
> 200 held frames the main-thread `update()` median goes 0.254 → 0.218 ms
> with **zero** full recomputes either way. `tests/warp_predictor_test.py`
> §23, with a counter-check that the uncapped rule moves the line by 2.3e6 m.

> **RKN4 was silently 3rd order until 2026-08-18 — one coefficient.** The third
> stage read `p3 = p0 + v0*h/2 + a2*h²/8`. In classical RKN4 it is `a1`, which
> makes k₂ and k₃ share an argument — that identity *is* the method's whole
> advantage (order 4 from 3 force evaluations, where RK4 needs 4). With `a2` the
> stages differ, the 4th-order condition breaks, and you pay the 4th evaluation
> for 3rd-order accuracy — strictly worse than plain RK4 on both counts.
>
> Measured against an analytic circular orbit, one revolution, fixed step,
> halving h (order p ⇒ error falls 2^p):
>
> | steps | before (`a2`) | ord | after (`a1`) | ord |
> |---|---|---|---|---|
> | 200 | 3.87e+03 m | 2.98 | 2.74e+01 m | 4.03 |
> | 800 | 6.09e+01 m | **3.00** | 1.05e-01 m | **4.01** |
> | 1600 | 7.63e+00 m | **3.00** | 6.56e-03 m | **4.00** |
>
> At equal cost (4800 force evaluations) that is 1.81e+01 m before against
> 6.56e-03 m after. `energy_test`'s RKN4 fixed-step drift went
> `6.4718e-04` → `4.2571e-07`; **Verlet's `4.2230e-10` is unchanged**, which is
> the check that only the RKN4 path moved.
>
> **The fix must land in `physics/world_kernels.py:147` as well** — it did not, for six
> hours, and since the kernel is the default path the game kept running 3rd
> order while the Python reference was 4th. That is exactly the divergence the
> bit-identity rule exists to catch.
>
> **And it must land in `ship/predictor/` too — it did not, for two days.**
> `_rkn4_step_numba` and `_rkn4_step_time_numba` kept the `k2` form, so the
> *world* integrated 4th order while the *drawn prediction* integrated 3rd.
> Measured the same way (circular orbit, one revolution, fixed step, error at
> the closing point): the predictor's kernel came out at order **3.00** and now
> comes out at **4.01**; at 800 steps that is `4.266 m` → `7.365e-03 m`, a
> factor of 580 at identical cost. Nothing in the game *displayed* the world's
> integrator, so the only visible symptom was the drawn line disagreeing with
> where the ship actually went — which is invisible in real time (the line is
> recomputed and re-anchored every frame) and only shows under the time-warp
> hold, where the line stands still and the ship slides along it.
>
> Note `p3` now equals `p2` bit-for-bit, so `a3 == a2` and the third evaluation
> is pure waste — collapsing it to the classical 3-stage form cuts it
> bit-identically.
>
> **Done in the predictor (2026-08-30), still open in the world.**
> `predictor._rkn4_step_time_numba` and `_rkn4_step_numba` now read `k3 = k2`
> instead of re-evaluating. Bit-identical across all nine measurement cases in
> `tests/warp_predictor_test.py` §24 (same step count, 0.000e+00 deviation).
> **But it is worth 1.02x–1.15x there, not the ~25 % this note promises, and
> the reason is `body_memo`**: k3 ran at the *same time* as k2, so every body
> hit the memo and only the 28 lookups plus the force sum were ever paid — the
> Kepler solves were already saved. The **world kernel has no memo**, so the
> full ~25 % is still on the table in `physics/world_kernels.py:147` / `physics/world.py`, and
> that is where this note's figure still applies.

## The body model (`bodies/body.py`, `physics/vec.py`)

- `bodies/body.py` — `body` (celestial) and `schiff` (ship subclass) data classes,
  plus **`kepler_relative_xy()` — the one scripted-orbit model in the
  project**. `orbit_position` (used by `world.update_planets`) and
  `position_at_time` (used by the integrator's force loops) both go through
  it, and `world_kernels._body_pos_at_time` is its word-for-word numba twin.
  Exact Kepler, so propagating in one step or in a hundred gives the same
  answer — which is what stops the time-warp chunking moving the planets.
  See the body-model note in `.claude/rules/orbit-lines.md`.
- `physics/vec.py` — `Vec2` with `__slots__`. Also exports `G = 6.6730831e-11`.
