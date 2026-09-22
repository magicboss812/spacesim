# spacesim

2D N-body orbital mechanics simulator with a playable spacecraft, built for a
Seminararbeit on Bahnmechanik. Real-time pygame + moderngl (OpenGL) rendering,
Numba-JIT trajectory prediction, Principia-style reference frame transforms.

## Run

```
pip install moderngl pygame numpy numba astropy poliastro imgui-bundle
cd spacesim && python main.py
```

Main folder is `spacesim/`, and it is a **source root, not a package** — every
module imports absolutely from it (`from physics.world import world`), and the
test files put it on `sys.path` themselves. **The entry point is `main.py`**;
it only wires `runtime/bootstrap.py` to `runtime/loop.py`. Window is resizable
(default 1280×800, VSync on); every player-facing knob lives in
`spacesim/config/config.json`. Env flags: `SPACESIM_PREDICTOR_ASYNC=1`,
`SPACESIM_MAX_FRAMES=<n>`, `SPACESIM_CONFIG=<path>`.

> **Never call `pygame.init()`.** It starts *every* submodule, and both `mixer`
> and `joystick` enumerate the machine's devices while doing so — measured
> **25.2 s + 20.1 s = 45.3 s** before the window even exists, which reads as a
> hang, not a load. The game uses only `display`, `event`, `font`, `image`,
> `key`, `mouse` and `time`; of those only `display` and `font` need
> initialising, so `runtime/window.py` and every GL test call exactly those two.
> The cost lives in audio/HID drivers, not this code — it can reappear on any
> machine at any time.

## Module map

Seven topic folders under the source root. **`Renderer` and `Predictor` are each
ONE class assembled from mixins** spread over their folder — the files below are
parts of one object sharing one `self`, not separate components.

| folder | what lives there |
|---|---|
| `main.py` | the entry point — 27 lines: config → `bootstrap` → `loop` |
| `runtime/` | `window` (display/GL/DPI/clock), `gl_device` (FXAA, present, resize), `bootstrap` (builds everything — **the construction order is load-bearing**), `loop` (the frame loop + `TIMING:`), `input` (keymap, click gesture), `system_loader` |
| `physics/` | `world` (adaptive RKN4, `update_planets`/`update_dynamics`, `step`), `world_kernels` (the Numba fast path, **bit-identical** to the Python reference), `vec` (`Vec2`, `G`), `reference_frames` (Principia-style plot frames; **largest file, 1733**), `kernels/` (the predictor's `@njit` maths: `kepler`, `integrators`, `apsis`, `propagate`) |
| `bodies/` | `body` (`body`/`schiff` + `kepler_relative_xy()`, the **one** scripted-orbit model), `style`, `icon`, `orbit_lines`. Pure numpy — **no GL here** |
| `ship/` | `control` (`schiffcontrol`), `camera` (world↔screen, zoom, follow, `sim_dt`/warp), `art`, `horizon` (`HorizonPolicy`), `predictor/` (`core`, `hold`, `compute`, `jobs`, `view`), `maneuver/` (`profile` — **die** brenndauer, `plan`, `preview`, `executor`) |
| `render/` | `renderer` (the class itself), `pipelines`, `draw`, `text`, and the five draw domains `bodies`/`ship`/`orbits`/`prediction`/`maneuver`; `background{,_draw}`, `line_kernels`, and `gl/` (GLSL 330, reached via `render.GL_DIR`) |
| `config/` | `loader` (`ConfigLoader`), `config.json` (all tunables), `solar_system.json` (27 bodies + ship `SaturnV`) |
| `ui/` | the player-facing HUD, plus `devui.py` — Dear ImGui **developer** tools (`F1`), a separate system |

## Where the detailed notes live

Each file below is a **path-scoped rule** — it loads automatically when you open
the source it covers. Read one directly when a question enters its topic without
touching its files.

| rule file | covers |
|---|---|
| `.claude/rules/physics-world.md` | integrator, RKN4 order, kernel bit-identity, **time warp ceilings**, the Kepler body model |
| `.claude/rules/predictor.md` | the drawn line: horizon vs detail, the warp hold, thrust pipeline, apsis markers, body memo |
| `.claude/rules/rendering.md` | draw path & line kernels, GL state caches, Y conventions, text/FXAA, ship size, frame timing |
| `.claude/rules/body-art.md` | `bodies/style.py` + `ship/art.py` — geometry, lighting, detail ladder |
| `.claude/rules/orbit-lines.md` | die Bahnlinien: Zeitkurve im Plot-Frame, SOI-Bänder, `FrameAffineTable`, faint Volllinie über einen Umlauf, Kreis-Endkappe mit echtem Radius |
| `.claude/rules/background.md` | Sternenfeld + Gitter: nichts skaliert mit `camera.scale`, Lattice-Parität, Pixelraster & HUD-Palette |
| `.claude/rules/reference-frames.md` | cubic origin interpolation, batched Kepler knots, per-frame cost work |
| `.claude/rules/camera-input.md` | zoom/follow/selection, sim-rate decoupling, **the full controls reference** |
| `.claude/rules/maneuver.md` | manoeverknoten: das rampenprofil, die vorschau-kette, der autopilot, die griffe |
| `.claude/rules/hud-ui.md` | the whole `ui/` layer — theme, chamfers, typefaces, navball, widgets |
| `.claude/rules/devui.md` | the ImGui dev panel and its timing ring buffer |
| `.claude/rules/config-loader.md` | `config.json` sections, typed accessors, `solar_system.json` |
| `.claude/rules/tests.md` | what every test file asserts, and the known pre-existing failures |
| `.claude/rules/paper.md` | die Seminararbeit in `spacesim/docs/`: Ordnerrollen, Quellen lesen (Curtis-Seitenversatz, Apollo-OCR), Zitierweise, deutscher Stil, KI-Protokoll |

## The paper — `spacesim/docs/`

The Seminararbeit, an **Obsidian vault**: chapters in `Obsidian Vault/Evaluation/_Arbeit/`,
sources in `_Arbeit/Quellen/`, raw notes in `Evaluation/*.md`. **Everything written
for the paper is German**; every AI-assisted text change gets a row in
`Evaluation/KI-Nutzung.md`. Full ruleset: `.claude/rules/paper.md`.

## Keeping these files current

Update the docs in the same pass as the code, and **match the size of the note
to the size of the change**. A note earns its place only by stopping a future
session redoing a mistake: state the rule, the measurement proving it, and the
counter-check that would fail without it — nothing else.

| what changed | what to write |
|---|---|
| **a new mechanic** — a new system, mode, or way the sim behaves | one row in the module map or routing table above, **plus one section** in the rule file that owns it |
| **a new source file** | a module-map row here; a new rule file only if no existing one owns it |
| **a bugfix** | edit the affected note *in place*. Do **not** append a second note describing the fix — these files record the current state, not a changelog |
| **a rebalance or tuned constant** | usually nothing — `config.json` is the record, and new tunables belong there, never hardcoded in `main.py`. Write here only if the value is load-bearing and a plausible "improvement" would break it, and then change the number in the existing note, not the prose around it |

A **new rule file** is worth creating when a topic owns its own source files
*and* its notes pass ~50 lines. Give it `paths:` frontmatter covering those
files, add a routing-table row, and write cross-file references as explicit
paths (`` `.claude/rules/x.md` ``) — never "the note above", which breaks the
moment a note moves.

**This file stays under 200 lines.** It loads in full every session, so
anything only true while working inside one module belongs in that module's
rule file, not here.

## Codebase navigation — graphify first

A pre-built knowledge graph lives in `graphify-out/` (`GRAPH_REPORT.md` first,
then `graph.html` / `graph.json`) — for **cross-module** questions; read code
directly for narrow lookups. **STALE since the 2026-09-03 restructure** —
re-run `/graphify` first.

## Git — never on your own

`spacesim/` is its own repo; `Werk/` is not, so `CLAUDE.md` and
`.claude/rules/` are tracked by nothing. **Never `commit`, `add`, branch, push,
stash or reset unless the user asks in that same turn** — inspecting is fine.

## Screenshots — "check the screenshot(s)"

They live in `screenshots for debugging/`, named `Screenshot YYYY-MM-DD
HHMMSS.png`, so **newest = last by name and by mtime**. *Check the
screenshot(s)* in any wording means go read the newest N (plain singular = 1) —
the image is not attached (`ls -t "screenshots for debugging" | head -N`). They
are usually **crops**, so the HUD is often missing and the scale unknown: read
them for *shape*, and get every number from a measurement instead.

## Invariants — don't break these

- **Physics stays in absolute (barycentric) space.** Reference frames are a
  render-time transform only. Never bake a frame into stored body state.
- **Numba kernels must stay pure.** `@njit` functions cannot take Python
  objects or call into pygame / OpenGL. Pass plain arrays and scalars.
- **There is exactly ONE scripted-body model, and it is Kepler** —
  `bodies.body.kepler_relative_xy()`, reached by `orbit_position`,
  `position_at_time`, `physics/world_kernels.py::_body_pos_at_time` and
  `physics/kernels/kepler.py` alike. Exact
  propagation means one step and a hundred give the same answer, which is what
  stops time warp moving the planets. Never reintroduce a constant-rate or
  Euler variant; that split *was* the bug (see `physics-world.md`).
- **Scripted orbits are not integrated.** `fixed=True` bodies with an
  `is_moon_of` parent follow Kepler elements; only the ship and other
  non-`fixed` bodies go through `update_dynamics`. (There is **no**
  `world.should_release()` / `MIN_GRAVITY_THRESHOLD` handover, in this tree or
  any commit checked — `body.released` is `False` in `bodies/body.py:117` and
  never set `True`. Writing one is new work.)
- **A prediction curve is CONSUMED, never translated.** The look-ahead belongs
  to the *orbit*, not the moment: `_advance_points_along_curve` drops the
  points whose time has passed and prepends the ship as the new head; nothing
  behind it moves. Never "reattach" it by shifting `points[:, 0/1]` — the
  reference body does not shift with it, so the conic ends up displaced by the
  *relative* motion and the periapsis is wrong by exactly that. Guarded by
  `tests/apsis_stability_test.py`.
- **ONE burn-duration calculation: `BurnProfile`** (`ship/maneuver/profile.py`).
  Preview and executor both call it; ignition is `t_node - total_time/2`,
  derived on every read. Its `a_max` is a **sim-time** acceleration
  (`thrust_acc / realtime_warp_max` = 10 m/s², not 600) — `maneuver.md`.
- **Predictor and World share kernels** — change an integrator constant in one,
  change it in the other.
- **`Vec2` semantics.** `a + b`, `a - b`, `a * k` return new instances; only
  `+=` / `-=` / `*=` mutate — use `magnitude_squared()` in inner loops.
- **SI units only.** Metres, seconds, kilograms, m/s, m/s². 2D only. `G` lives
  in `physics/vec.py`, overridable via `physics.gravitational_constant`. Never
  introduce AU, days, or Earth-masses.
- **Two Y conventions — convert at the ortho boundary.** The world draws
  top-down (`line.vert`/`body.vert` flip y); text and the ship draw ortho (y-up,
  origin bottom-left). Use `Renderer._ortho_y(y)` and `_blit_text_topdown(...)`;
  never pass a `_world_to_screen_xy` result straight to `_blit_cached_text` or
  `_draw_ortho_shape`. See `rendering.md`.
- **UI sizes are design units, never pixels** (`UIContext.px()` /
  `Renderer.ui_px()` scale them) — a literal pixel in widget code is a bug at
  any resolution but the reference one.
- **Input priority is custom UI → ImGui → world.** Every consumer takes
  `ui_wants_mouse` / `ui_wants_keyboard`. `schiffcontrol` and the camera's WASD
  pan **poll** the keyboard, so they must check the flags explicitly — an
  event-level guard is not enough.
- **No dependency manifest** — a new import goes in the install line *and* `README.md`.
- **Debug flags exist** — flip them instead of adding `print`s:
  `predictor.debug_moving_sources`, `renderer.debug_predictor`,
  `world.integrator_debug`, `renderer.maneuver_enabled`.

## Conventions

- **German + English mix is intentional.** `Schiff`, `Erde`, `Mond`, `Sonne`,
  `arg_periapsis`, `is_moon_of` stay as they are — don't translate them.
  Comments are mostly German in older modules (`physics/world.py`, `ship/camera.py`), mostly
  English in newer ones (`physics/reference_frames.py`).
- **Class naming is inconsistent on purpose.** `Vec2`, `Renderer`, `Camera`,
  `Predictor` are PascalCase; `world`, `body`, `schiff`, `schiffcontrol` are
  lowercase. Match the surrounding file rather than renaming.

## Controls

The full key table lives in `.claude/rules/camera-input.md`. The essentials:
`WASD` pans (**arrow keys drive the ship, not the camera**), `←` `→` rotate,
`↑` `↓` thrust, `I` `K` `J` `L` orientation hold, `R` / `1` / `2` reference
frame, `N` / `X` maneuver node / arm burn, `PageUp` `PageDown` sim_dt, `F1` devui.

Warp steps run `1m/s … 1y/s`; those the orbit cannot resolve are greyed out.
**Thrust is real-time only** (an armed node drops the warp itself); rotation
always works. The HUD mirrors every keybind and reads its values back from the
simulation. Steps and camera rules: `.claude/rules/camera-input.md`.
