---
paths:
  - "config/loader.py"
  - "config/config.json"
  - "config/solar_system.json"
  - "runtime/system_loader.py"
  - "runtime/paths.py"
  - "spacesim.spec"
  - ".github/workflows/build.yml"
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
`physics`, `camera`, `ship`, `predictor`, `renderer`, `background`,
`warp_to` (`apsis_margin_s`, `point_margin_s`, `lead_real_seconds`, via
`warp_to_kwargs()`), `screenshot` (`directory`, read by `ConfigLoader.screenshot_dir()`; relative
paths resolve from the repo root, since `SPACESIM_CONFIG` may point into a
temp folder; `resolution`: `window`/`2K`/`4K`/`6K`/`8K`, the preselected
capture height, an unknown value warns and falls back to `window`), `debug`.

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
`ship_zoom_shrink_*`, `ship_build`, `ship_pixel_style`, `ship_build_colors`,
`ship_plume_idle`, `ship_plume_*`) because the
renderer owns it — `ship` holds the flight-model knobs. Same for what gets
written next to a body: `body_label_mode` / `body_label_min_radius_px`, and for
the prediction line's look (`prediction_line_style`, `_color`, `_mark_color`,
`_fade_steps`, `_fade_min`, and `prediction_line_styles`: {style: {setting:
value}}, merged over `render/prediction_pixel.py::DEFAULT_STYLE_SETTINGS`, so
a missing setting takes the default and an unknown one is ignored).

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
can deliberately differ. `is_star` / `is_gas` (both `false` when missing)
switch a body to the star or gas-giant drawing and marker
(`.claude/rules/body-art.md`); Sonne carries `is_star`, the four giants
`is_gas`. Within a body a seeded ~14 % of the figures switch to
an accent shape, which is what stops a planet reading as one repeated stamp.

## Where `config/` lives — `runtime/paths.py` and the frozen build

Both loaders resolve relative paths from `runtime/paths.py::config_dir()`, the
screenshot folder from `user_dir()`. Never from `__file__` again: in a
PyInstaller build `__file__` points into the bundled, read-only part, so an
edited `config.json` next to the exe would be silently ignored.

- **Repo:** the repo root, as always.
- **Frozen Windows/Linux:** the exe's folder (`SpaceSim/config/`).
- **Frozen macOS:** `~/Library/Application Support/spacesim/`. Not next to the
  `.app`: Gatekeeper runs a downloaded app from a random read-only path (app
  translocation), and a file changed inside the `.app` breaks its signature.

`prepare_frozen()` (first call in `main.py`) copies a missing file from the
bundled template and installs an excepthook writing `crash.log` there, since a
`console=False` app has no stderr anyone sees. Numba needs nothing: frozen, it
uses its own user-wide cache. Measured in a Linux build: first start 54 s,
second 17 s, `NUMBA_CACHE_DIR` not involved.

**Build:** `pyinstaller spacesim.spec` on the target OS (no cross-compiling).
`.github/workflows/build.yml` builds both; its macOS job also runs the frozen
app 150 frames twice and `tools/game_shot.py --check-gl-errors` on the
runner's Apple software renderer, so a Mac build is checked without a Mac.
Verified locally in a Linux build: an edited external `config.json` takes
effect, a missing `solar_system.json` is restored, a broken one ends in
`crash.log` with exit 1.
