---
paths:
  - "spacesim/config/loader.py"
  - "spacesim/config/config.json"
  - "spacesim/config/solar_system.json"
  - "spacesim/runtime/system_loader.py"
---

# Config and system loading

## `config/loader.py`

`SystemLoader` — two-pass JSON load, resolves `is_moon_of` to object
references.

`ConfigLoader` — reads `config.json` and distributes the values via
`apply_globals` / `apply_to_world` / `apply_to_camera` / `apply_to_ship_control`
/ `apply_to_predictor` / `apply_to_renderer` / `apply_to_background` /
`predictor_kwargs`. `apply_to_background` is reached **from**
`apply_to_renderer` (the layer hangs off `renderer.background`), so
`apply_all` needs no extra argument.

> **Read values through `get_float` / `get_int` / `get_bool` / `get_str`, not
> bare `get()` + `float()`.** The typed accessors warn and fall back on a
> malformed value instead of raising, and `get()` returns `Any` — a config
> value can be a nested section. `_strip_jsonc` is leniency for hand-edits (a
> stray `//` note, a trailing comma), **not a file format**.

`async_compute` / `rolling_mode` are constructor-only (executor setup) and are
passed through `predictor_kwargs()`, not re-assigned afterwards.

## `config.json` — all user/player-tunable parameters

One per line, strict JSON with no comments. Sections: `window`, `simulation`,
`physics`, `camera`, `ship`, `predictor`, `renderer`, `background`, `debug`.

**New tunables belong here, not hardcoded in `main.py`.** Missing keys fall
back to the in-code default; unknown keys are reported at startup
(`ConfigLoader.unknown_keys`).

`renderer.orbit_line_full_*` (faint one-revolution loop) and
`renderer.apsis_marker_fade_{min,full}_px` (size-based apsis fade) follow the
same `_assign` pattern — plain typed keys, no dev-UI parity requirement (only
`background` is pinned three-ways). The dev panel carries sliders for the fade
range and `orbit_line_full_alpha_mult` under **Renderer › visuals**.

Ownership follows the *drawer*, not the subject: the ship's own look is under
`renderer` (`ship_sprite_enabled`, `ship_length_px`, `ship_render_scale`,
`ship_zoom_shrink_*`, `ship_accent_color`, `ship_plume_idle`) because the
renderer owns it — `ship` holds the flight-model knobs. Same for what gets
written next to a body: `body_label_mode` / `body_label_min_radius_px`.

`background` is the one section that is **not** just a bag of values: its key
set, the attributes of `background.BackgroundLayer` and the sliders in the
dev-UI's `Background` header are pinned to be identical, and
`tests/background_test.py` §7 checks all three against each other. Adding a
knob means adding it in all three places — and a new piece of *runtime state*
on the layer means adding it to that test's `runtime_only` exclusion, or §7
reports it as a missing config key. See `.claude/rules/background.md`.

`background.grid_anchor` is the section's only enum (`"focus"` | `"world"`);
`apply_to_background` warns and falls back to `"focus"` on anything else,
because a typo must not silently select different behaviour.

`physics.gravitational_constant` overrides `vec.G` at startup via
`ConfigLoader.apply_globals()`, which rebinds the `from vec import G` copies in
`bodies`, `world` and `reference_frames` — rebinding `vec.G` alone would not
reach them. `world.G`, the value the predictor snapshots, is set by
`apply_to_world`.

## `solar_system.json`

Default system: **28 entries** = 27 gravitational bodies (Sonne, 9 planets, 17
moons) + the ship `SaturnV`. Every planet and moon is `fixed: true` with Kepler
elements, so only the ship is integrated — but per-substep cost still scales
with the *count*, because every body is placed on every acceleration
evaluation.

Three optional appearance keys, all defaulted so no entry needs them:
`style_seed` (else derived from the **name**, so every body already has its own
stable pattern), `style_mode` (default `bands`) and `style_shape` (default
`nested`). That pairing is the intended house look; the keys exist so one body
can deliberately differ. Within a body a seeded ~14 % of the figures switch to
an accent shape, which is what stops a planet reading as one repeated stamp.
