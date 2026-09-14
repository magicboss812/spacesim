---
paths:
  - "spacesim/ui/**"
  - "spacesim/ui/**/*.py"
  - "spacesim/render/gl/ui_rect.vert"
  - "spacesim/render/gl/ui_rect.frag"
  - "spacesim/render/gl/texquad.vert"
  - "spacesim/render/gl/texquad.frag"
---

# The player-facing HUD (`ui/`)

| module | role |
| --- | --- |
| `units.py` | pure number formatting, testable without GL |
| `theme.py` | palette, type scale, spacing, corner shapes |
| `text.py` | fonts, label-texture cache, tinted blitting |
| `draw.py` | drawing primitives on ONE SDF shader |
| `core.py` | rects, anchoring, widget base, input routing |
| `state.py` | observable view state (reference frame, overlays) |
| `widgets/` | button, dropdown, label, panel, slider |
| `hud/` | the concrete HUD elements (attitude, navball, telemetry, body_browser, system_map, apsis_tooltip, controls, panels, chrome, layout) |

## Conventions that bind every widget

- **`ui/` picks the top-down convention and converts at its own boundary.**
  Every public API in `ui/` takes top-down screen pixels (origin top-left) —
  the same convention as pygame's mouse events, so hit-testing needs no
  conversion at all. The flip to ortho happens in exactly two places:
  `UIDraw._submit` and `TextRenderer._blit`. Nothing else in `ui/` may touch
  `self.height - y`.
- **UI sizes are design units, never pixels.** Widgets store anchor, offset
  and size in design units; `UIContext.px()` scales them (the counterpart to
  `Renderer.ui_px()`). Anchoring — not stretching — is what keeps elements
  glued to their corner on resize. A literal pixel value in widget code is a
  bug at any resolution but the reference one.
- **Retained mode with anchored layout, not immediate mode.** The HUD
  structure is static and soft hover/panel transitions are painful in pure
  immediate mode — every widget would need externally held animation state.
  Widgets hold their own state and get layout → update → draw once per frame;
  a widget remembers a corner/edge, an offset and a size, so on resize it
  stays stuck to its corner instead of being redragged from scratch.

---

- `ui/` — **the player-facing HUD** (Phase 3). Separate from `ui/devui.py` in
  purpose: `devui` is the developer toolbox, `ui/` is the designed surface.
  They share nothing but the input-priority chain (custom UI → ImGui → world).
  - `ui/units.py` — compact formatters (distance, speed, `1y 23d 04:05:06`,
    mass, Δv, angle). Pure, no GL, covered by `tests/ui_units_test.py`.
  - `ui/theme.py` — palette, type scale, spacing, corner shapes, motion
    rates. Restyled 2026-08-19 onto the **KSP2 instrument-panel idiom**; the
    whole look is reachable from this one file.

> **A negative corner radius is a CHAMFER, not a rounding.** That single sign
> carries the visual identity. Uniform heavy roundness (`radius 16` panels,
> `radius 999` pills everywhere) was the strongest "a machine designed this"
> tell in the previous HUD, and the house typeface — SB Liquid — cuts its own
> corners at 45°, so rounding fought the type on every element. `v_radius`
> already had four free floats, so `shape_distance()` in `ui_rect.frag`
> branches on the sign: `chamfer_box()` intersects the sharp rect with the
> 45°-rotated half-plane `|x| + |y| <= hx + hy - c`. Both halves are convex
> and 1-Lipschitz, so `max()` is still a pixel distance and the existing
> `smoothstep(±0.5)` antialiases the diagonal exactly like the straights —
> **no second shader, no extra attribute, no extra draw call.**
>
> `theme.cut_corners(size, top_left=False, …)` builds the per-corner tuple.
> Leaving *some* corners sharp is the point: the asymmetry is what reads as
> engineered rather than decorated, and it is how blocks dock to each other
> (the ship badge's bottom-left corner is sharp because the reference-body
> selector sits under it). `tests/ui_render_test.py` §9 discriminates the two
> forms at the one point where they differ most — for corner size `c`, the
> point `(0.35c, 0.35c)` is **outside** a chamfer (`x + y < c`) and **inside**
> a rounding (distance `0.92c` from the arc centre at `(c, c)`). Do not move
> that probe to `0.3c`: there the rounded distance is `0.99c`, which lands in
> the antialiased edge and measures a half value.
>
> `UIContext.px()` accepts a tuple as well as a scalar, because a corner
> radius is now per-corner. `Panel` passed one through unconverted once and
> died on `float() argument must be … not 'tuple'`.

> **Two typefaces, and the pixel one must be rastered differently.**
> `ui/assets/` holds `ui-display.ttf` (SB Liquid W01 Solid — the user's own
> licensed copy, a chamfered pixel face) and `ui-text{,-bold}.ttf`
> (Oxanium 500/700, OFL, squared letterforms with softly rounded corners;
> `OFL-Oxanium.txt` sits beside them). `theme.Role.family` picks between
> `'display'` and `'text'` — the old `mono` bool survives only as a property.
>
> The pairing works because the two faces measure almost the same: at nominal
> size 10 SB Liquid gives 7 px cap height and 68 px advance for a test string,
> Oxanium 8 px and 64 px. They mix in one line without size juggling.
>
> Two things in `ui/text.py` are load-bearing and were measured:
>
> 1. **The pixel face is rastered with `antialias=False`.** Measured over
>    `ABCDEFG0123456789`, `font.render(..., True, ...)` leaves a
>    half-covering fringe at *every* size (6–22 % of pixels partly covered);
>    with antialiasing off there are exactly **two** alpha values, 0 and 255.
>    That is the whole difference between "pixelated" and "blurry pixelated".
>    pygame returns a palettised surface in that mode, so `_render_plain()`
>    goes through a colour-keyed `SRCALPHA` blit — `image.tostring('RGBA')`
>    on the raw surface would bake an opaque black box around every glyph.
> 2. **Its pixel size is snapped to a multiple of 5, floor 10.** Off the
>    ladder the stems come out uneven: at 11–13 px the same string carries
>    1 px *and* 2 px stems side by side, which reads as unrest even though
>    every edge is hard. On 10/15/20/25/30/35/40/45/50 the stem width is
>    uniform. Below that, 9 px leaves a capital only 6 px of ink. The
>    **design sizes in `TypeScale` therefore sit on the five-ladder too** —
>    put 11 or 13 on a `display` role and it snaps onto its neighbour's pixel
>    size, losing the step. The text face is *not* snapped: it has no raster
>    to hit, and a 5 px jump there would just be a coarse type jump.
>    Textures for the pixel face use `NEAREST`.
>
> `tests/ui_render_test.py` §10 asserts the family split, the antialias flag
> per role, the five-ladder at four `ui_scale` values, zero partly-covered
> pixels in the display face and a non-zero fringe in the text face (that
> counter-check is what stops the whole section passing vacuously).

> **The display face needs TABULAR FIGURES, because it is not monospaced.**
> Measured at 15 px, SB Liquid's `1` is 9 px wide and every other digit is
> 10. A right-aligned countdown therefore changes total width whenever a `1`
> enters or leaves it, and the *left* edge of `T-01:55:21` twitched once a
> second. `_role_tabular()` is true for the whole `display` family and
> `_render_tracked()` then lays digits on a fixed cell equal to the widest
> digit, **centring** each glyph in its cell (left-aligning puts a visible
> gap after a narrow `1`). Only digits — other characters keep their natural
> width, since within one format they always sit in the same place. The text
> face stays proportional. Measured after: every second and hour value of the
> countdown renders at exactly 99.0 px in `value`, 159.0 px in `gauge`.
> `tests/ui_hud_test.py` §11 checks that, and also that the raw font is
> *not* monospaced — otherwise the section would prove nothing.
  - `ui/text.py` — `TextRenderer`: role-based fonts (`match_font`, or a TTF
    dropped into `ui/assets/`), label-texture cache with FIFO eviction **over
    a pool of retired textures** (see the GL-allocation note in
    `.claude/rules/rendering.md`), tinted blitting, `defer()`/`flush()` for anything that must land
    after FXAA.

> **Text is rasterised white and tinted via `u_color`** (`texquad.frag`). One
> white texture per string then serves any colour — were colour part of the
> rasterisation it would have to enter the cache key and every hover state
> would occupy its own GL texture. The two failure modes that make text soft
> or ghosted (sub-pixel anchors under `LINEAR` filtering, and drawing before
> the FXAA resolve) are the same ones fixed for world-space labels — measured
> numbers and the fix live in `.claude/rules/rendering.md` ("Text must be
> pixel-snapped and must not go through FXAA"); `defer()`/`flush()` above is
> `ui/text.py`'s side of that same rule.

  - `ui/draw.py` — `UIDraw`: rounded rect, circle, ring, arc, line, divider —
    **all on one SDF shader** (angles are degrees, counter-clockwise, 0 =
    right). A circle is a rect with radius = half the edge,
    a ring is that with a border and no fill, an arc is a ring with the arc
    params. **Draws are batched and instanced** (2026-08-17): `_submit` packs
    the 33 per-shape floats into an instance buffer; `flush()` issues ONE
    instanced draw. Each shape used to be its own draw call with ~14 uniform
    writes — at ~160 calls per HUD frame (48 attitude-ring ticks alone) that
    was the single largest UI cost. Instance order == call order, so blending
    is unchanged; `TextRenderer._blit` calls `rect_flush` (wired in
    `UIContext`) before drawing so text still layers exactly by call order.
    The shader takes per-instance attributes (`i_*` → flat varyings) instead
    of uniforms; only `u_viewport` remains a uniform.

> **Filling one instance row is ONE assignment, not thirty-three (2026-08-27).**
> Every single store into a numpy row is a full ufunc dispatch, and over the
> ~200 shapes of a HUD frame that was measured as the largest single item
> inside `_submit`. `UIDraw` keeps a flat view on the same memory
> (`_instance_flat`, rebuilt in `_ensure_capacity` alongside the array) and
> writes the 33 floats as one slice assignment — same order, same layout, it
> must still match `_INSTANCE_FORMAT`. And `flush()` hands the rows straight
> to `write()` instead of `tobytes()`: rows `0..count` are already contiguous,
> so the copy bought nothing — **~1.5 MB per frame** at ~200 shapes and the
> nearly 60 flushes a frame takes.
  - `ui/core.py` — `Rect`, anchor constants, `UIContext`, `Widget`, `UIRoot`
    (hit-testing, hover/press/focus, z-order, `wants_mouse`/`wants_keyboard`).

> **`z` is global, not per-level.** `UIRoot.paint_order()` flattens the tree
> into one list sorted by `(effective_z, depth-first index)`, where effective
> z = parent's + own. Both drawing and hit-testing walk that list (picking
> walks it reversed). Sorting only *siblings* — which is what it did first —
> means a `z=200` popup still loses to any panel that sits later in the tree:
> the palette popup was drawn *under* the system-map panel and that panel
> swallowed its clicks. A `z` that only holds within one level is not a `z`.
  - `ui/state.py` — `UIState`: reference body, frame extension, target overlay,
    and the clicked-body selection (`selected_index`, which is deliberately
    *not* the reference body — see the selection note in `.claude/rules/camera-input.md`).
    These used to be *local variables* in `test.py::main()` (now
    `runtime/bootstrap.py`), unreachable for
    any HUD control; they now live here with a change notification, so
    keyboard (`R` / `1` / `2` / `T`) and HUD controls write the same state and
    cannot diverge. **Invariant: view state only, no physics** — the reference
    frame is a pure display transform.
  - `ui/widgets/` — Panel, Group, Label, Readout, Button, Toggle,
    SegmentedControl, Slider (linear/log), HorizonSlider (centre-sprung rate),
    Dropdown.
  - `ui/hud/` — the concrete HUD elements, restyled 2026-08-19 onto the KSP2
    instrument panel. `chrome.py` (**the form vocabulary — chamfer, double
    frame, notch tab, ruler, segmented arc; every surface goes through it**),
    `navball.py` (the instrument cluster), `attitude.py` (the 2D compass ring
    inside it), `telemetry.py` (all displayed values, sampled once per frame),
    `panels.py` (ship badge, target block, compact rails), `body_browser.py`
    (the reference-body drawer), `controls.py` (warp bar, frame selector,
    snap rosette, zoom), `layout.py` (anchoring + the responsive swap). Entry
    point is `Hud(...)`; the main loop calls `hud.update()` once per frame
    before `ui_root.begin_frame()`.

> **The HUD is four blocks in four corners, and the middle stays empty.**
> Top-left ship badge → reference selector → target block (one column, equal
> widths, docked by sharp corners); top-right the warp bar with the UT clock;
> bottom-centre the navball cluster with the snap rosette docked to its right;
> bottom-left frame selector, zoom and the predictor-horizon slider. The previous layout scattered eight
> separate lozenges and built the bottom-centre column tall enough to sit on
> top of the orbit — the one thing the screen exists to show.
>
> `NavballCluster` (474 × 282 design units) is **one widget**, not a stack:
> compass ring, throttle arc, radial-speed arc, the ORB and ALT flanks, their
> THR / V/S strips, the heading badge and the `ORBITAL.INFO` block with AP/PE.
> The ring is its only child, because it owns hard-won drag logic. Everything
> else is drawn by the cluster itself — as children they would be separate
> boxes again, which is exactly what the densification removes. `ORBITAL
> ELEMENTS` and the standalone throttle panel are **gone**; those values live
> here now. The throttle is set by dragging the left arc or its strip, and the
> wheel only acts over those two — anywhere else it must fall through to the
> camera, or the zoom stops working with the cursor over the instrument.
>
> **A widget that draws a circle must hit-test a circle.** `AttitudeRing`
> is a square widget containing a round instrument, and its four empty
> corners used to belong to it. The throttle arc sits *outside* the ring
> (radius 109 vs 92) but its upper and lower ends fall *inside* the ring's
> square, so the ring swallowed exactly those clicks — the throttle could
> not be set in its top third even though the arc was visibly clear.
> `AttitudeRing.hit_test` is now a radius check. `tests/ui_hud_test.py` §10
> asserts all four corners are rejected, that the arc's 4 %, 50 % and 96 %
> points route to `NavballCluster`, **and** that the 96 % point really does
> lie inside the ring's rect — without that counter-check the section would
> pass vacuously if the arc ever moved outward.
>
> **A widget's rect must reserve space for its notch tab.** The tab rides
> *outside* the edge (`chrome.tab`, `edge='top'` draws above `y`). Drawn
> inside it covers the block's first row — on the warp bar that hid the two
> lowest steps. `SegmentBar.tab_height()` / `HudPanel.tab_height()` add it in
> `measure()` and subtract it again in `_bar_rect()` / `_frame_rect()`.
>
> **AP and PE are distances from the CENTRE, ALT is a height above ground.**
> They used to disagree: `Telemetry.text_periapsis` printed
> `periapsis - reference.radius` while the Pe/Ap flags on the line print
> `markers[:, 4]`, the distance to the reference body's *centre*. Two numbers
> for the same point, a whole body radius apart (6371 km at Erde), with nothing
> on screen saying which was which. Both are centre distances now; the height
> above the surface is what `ALT` is for. `text_closest` follows, since it
> reads the very same marker.

> The right flank shows **radial speed** (`v · r̂`, `Telemetry.radial_speed`),
> not KSP2's ground-relative vertical speed — there is no surface normal here,
> and `v · r̂` is exactly zero at both apsides, which makes the arc an apsis
> indicator for free. The arc and the `V/S` strip under it are the *same*
> number: the arc fills outward from its middle (up = climbing, down =
> falling), the strip prints it. Its full scale is the orbital speed, not a
> fixed number: real rates run from m/s in LEO to km/s on a transfer, so any
> fixed full scale is useless at one end — full deflection therefore means
> "all of the motion is straight up or straight down". **Subtract the body's
> velocity via `Telemetry.body_velocity()`**, never the stored `velocity`
> field — scripted bodies keep their load-time zero, which would put a
> planet's 30 km/s into the readout.
>
> Its denominator is `OrbitalElements.speed` — reference-relative — **not
> `frame_speed`**. `frame_speed` depends on the selected plotting frame, so
> the gauge would rescale because someone changed the *view*; numerator and
> denominator have to come from the same motion.
>
> **The flank strips degrade in three stages, because their content scales
> with the value's magnitude.** `+14.41km/s` at 15 px filled the strip's whole
> 100 px and ran straight over the `V/S` label. `NavballCluster._strip()`
> tries big-value-with-label, then small-value-with-label, then
> small-value-alone — the value always wins, the label is the extra. One
> fallback is not enough: the pixel face snaps its size to a multiple of 5, so
> the "small" role is not proportionally smaller everywhere, and at 250 km/s
> it overflows too. Radial speed is also printed to **one** decimal; the
> second says nothing next to an arc showing the same thing.
> `tests/ui_hud_test.py` §13 sweeps eleven magnitudes × both signs × four
> `ui_scale` values and asserts nothing overflows — plus that the label
> survives in most cases, or the check would pass by always dropping it.
>
> **The velocity needle is scaled by the ORBIT, not by a constant.** It used
> to run `min(speed / 2600.0, 1.0)` — a number belonging to nothing. In LEO
> you fly 7.7 km/s and the needle pins; around a small moon it barely moves.
> `Telemetry.orbital_speed_scale()` returns `(v, v_circ, v_esc)` at the
> current radius from the reference body's own `mu`: `v_circ = sqrt(mu/r)`,
> `v_esc = sqrt(2)·v_circ`, and the needle is `v / v_esc`. The same needle
> length then means the same thing at every body — measured, a circular orbit
> lands on **0.7071 at Erde, Mond, Jupiter and Sonne alike**, and escape is
> full scale by construction. The ring also draws a faint reference circle at
> the circular-speed length, because a scaled needle without a graduation is
> just a wobbling line. `v` comes from `OrbitalElements.speed`, i.e. relative
> to the reference body — barycentric speed would carry Earth's 30 km/s and
> pin the needle in any Earth orbit. `tests/ui_hud_test.py` §12 covers the
> four bodies, both ends of the scale and the no-reference-body path (which
> `UIState` cannot reach — `set_reference_index(None)` is a no-op — but
> `Telemetry` accepts `ui_state=None`, so the test goes through that door).

> **The horizon slider is a RATE control, not a position one**
> (`ui/widgets/rate_slider.py::HorizonSlider`, bottom-left stack under FRAME +
> the SYSTEM/LOCAL zoom buttons). The knob rests at centre and springs back
> there (`ease` at `theme.motion.fast`); its displacement is the *speed* at
> which the drawn prediction length changes, integrated in `update()` as
> `mult *= exp(k·f(offset)·dt)`, `k = ln(max/min) / sweep` sized so a full
> deflection sweeps `predictor.horizon_slider_min_mult` (0.25) →
> `horizon_slider_max_mult` (256.0) in `horizon_slider_sweep_seconds` (3.5). A
> ±0.06 deadzone and an `x^1.8` response on
> the deadzone-shifted magnitude keep it still at rest and fine near centre.
>
> It is smooth because moving the knob only writes a clamped multiplier
> (`set_predictor_horizon_mult`); `HorizonPolicy.apply()` in `ship/horizon.py`
> turns that into one `predictor.set_display_length(drawn)` per frame — an O(1)
> clip of the already-computed curve. While the knob is held that function
> reads `hud.horizon.is_grabbing` and, via `ship/horizon.py::horizon_targets()`, walks
> the *computed* horizon up a coarse ratchet instead of pinning it to the
> ceiling; the cost details and the release-flash the pin caused are in
> `.claude/rules/predictor.md`. The `+` / `-` keys are unchanged (they still
> `predictor.reset()`).
>
> **The readout shows the DRAWN length, so it must not read `length` back.**
> `_horizon_metres()` takes `min(get_display_length(), predictor.display_length)`.
> `get_display_length()` alone is the *computed* horizon, which now runs ahead
> of the knob by up to one ratchet rung — and under the old ceiling pin sat on
> the maximum for the whole drag, so the number never moved while the player
> dragged.

> **The reference list unfolds DOWNWARD out of its button.** `BodyBrowser`
> keeps `open` as the *target* and `_open_t` as the drawn state, eased with
> the project's usual framerate-independent `1 - exp(-rate·dt)` at
> `motion.normal` (rate 22 made the first frame jump 30 % of the height,
> which reads as a snap rather than an unfold). Only the **height** animates:
> x, y and width are fixed, so it grows out of the button's bottom edge
> instead of flying in from the side. Each row fades in exactly as the
> growing bottom edge sweeps past it (`reveal = (bottom - row_y) / row_h`),
> which is what stops entries popping into existence.
>
> Two consequences. Drawing is driven by `_open_t`, not `open`, or closing
> would just vanish; and `hit_test` uses the *animated* rect and only while
> `open` is true, so a row that is not yet drawn cannot be clicked and a
> closing list stops swallowing clicks. That last part changed an existing
> test: `ui_hud_test` §8 used to open the list and click a row in the next
> frame, which now legitimately misses — it runs the unfold out first.
>
> **The body list is derived, never authored.** `body_browser.build_hierarchy`
> reads the `is_moon_of` links the loader has already resolved to object
> references: bodies without a parent are roots (sorted by mass, so the star
> leads), everything else nests under its parent, and siblings sort by
> `semi_major_axis` — distance from the body they orbit. Extending
> `solar_system.json` therefore needs no second list to keep in sync. The
> walk is cycle-guarded and appends anything unvisited as a root, because a
> bad `is_moon_of` link would otherwise recurse forever or silently drop a
> body. Ships are excluded for the same reason as
> `UIState.celestial_indices`: a frame sitting on the ship collapses its own
> orbit to a point.

> **Hierarchy is carried by type contrast, not colour**: letterspaced 10 px
> labels against 25–30 px readouts. That is also why the palette below has
> exactly one meaning per colour — a value can't lean on colour to stand out
> from its label if colour is already spoken for.

> **The palette is four colours, and it is now FIXED.** `theme.ROLE_INDEX`
> maps every role onto one of four values, so the whole HUD is recoloured
> from one place without touching a widget. The three swappable sets and the
> shuffle button are **gone** (2026-08-19): a colour that can be reassigned
> cannot carry meaning. The one scheme is cyan = data (orbit, ring, velocity,
> frame), magenta = the second axis (normal/antinormal, target), amber =
> energy and caution (throttle, altitude, warp), green = engaged/ready
> (autopilot, ship) — the same semantics KSP2 uses. `theme.palette_sets()`
> still exists and returns that single set.
>
> Surfaces use the raw colour; text and thin strokes go through
> `theme.readable()`, which lifts a colour toward white until it clears
> ~4.5:1 on the near-black ground. Without that lift `#22577a` sits at ~1.6:1
> and is unusable as a label. `theme.Palette.glow()` was cut from 0.28 to
> 0.13 alpha at the same time — the glow was what made every element read as
> a separately floating object.

> **Scripted bodies have no valid `velocity`.** `world.update_planets()`
> advances `body.position` for `fixed=True` Kepler bodies and never touches
> `body.velocity`, which keeps its load-time value — `0.0` for every body in
> `solar_system.json`. Reading it directly computes orbital elements against
> a *stationary* planet: a clean circular orbit around Erde came out as a
> hyperbola with **e = 23.5** instead of **e ≈ 0**. Use
> `Telemetry.body_velocity()`, which central-differences
> `position_at_time()` — the same function the integrator uses for moving
> gravity sources. Measured: Erde's stored field `0.0 m/s`, derived
> `30 287.8 m/s`. `Renderer._ship_relative_speed_m_s` still reads the raw
> field and has the same flaw.

> **Das manoever-werkzeug sitzt IM navball-raster** (`ui/hud/maneuver.py`),
> nicht daneben. Der block hat vier freie felder, und sie sind bereits
> gerastert: ueber der ORB-flanke, unter dem THR-streifen, ueber der
> ALT-flanke, unter dem V/S-streifen. Alles andere daneben zu stellen --
> erst ein 200 × 206-klotz, dann drei plaettchen mit eigenem dock-abstand --
> laesst den block als navball MIT ANHAENGSEL lesen und laesst gleichzeitig
> diese vier felder leer.
>
> | plaettchen | feld | inhalt |
> | --- | --- | --- |
> | `ManeuverBurnBlock` (76 hoch) | ueber ORB | DV, brenndauer, countdown, `EXECUTE` |
> | `ManeuverAxesBlock` (40 hoch) | unter THR | die beiden delta-v-achsen als EINGABEFELDER |
> | `ManeuverPlanBlock` (48 hoch) | ueber ALT | reichweite der vorschau |
> | `ManeuverNodesBar` (40 hoch) | unter V/S | knotenwahl, `+ NODE` / `DEL` |
>
> **Breite und x-lage kommen aus dem navball, nicht von hier.** `layout()`
> liest `NavballCluster.flank_rect()` / `strip_rect()` -- eigene zahlen
> waeren ein zweites layout fuer dieselbe flanke, und schon eine geaenderte
> `BOX_H` oder UI-skala liesse die plaettchen danebenstehen. Die untere
> hoehe ist zusaetzlich gegen `info_rect()` GEKLEMMT: die apsiden-leiste ist
> nur um eine halbe flankenbreite eingerueckt und ragt unter beide flanken.
>
> Oben sehen sie aus wie eine FLANKE (doppelrahmen, beschriftung im kasten
> -- die flanken tragen ihr `ORB` auch innen, keinen notch-tab), unten wie
> ein STREIFEN (flache, versenkte platte). Nach aussen zeigende ecken
> gefast, die zur kugel zeigenden scharf -- dieselbe regel, nach der schon
> die flanken angesetzt statt danebengestellt aussehen.
>
> `_regions()` ist die EINZIGE geometriequelle jedes plaettchens --
> zeichnen und treffertest lesen dasselbe woerterbuch.
>
> **Die beiden delta-v-zeilen sind eingabefelder.** Vier pfeilknoepfe je
> achse frassen 60 der 124 einheiten breite, und wer 1900 m/s einstellen
> will, klickt 190-mal. `takes_keyboard` ist deshalb eine EIGENSCHAFT
> (hover ueber einem wertfeld oder offenes feld), keine konstante: `UIRoot`
> fragt sie im moment des klicks ab, und ein festes `True` verschluckte
> `N` / `X` / `WASD`, bis man woanders hinklickt. Volle begruendung --
> auch, warum die richtung ein knopf und kein tippbares minus ist --
> `.claude/rules/maneuver.md`.
>
> **EXECUTE ist abgeblendet, solange der knoten kein Delta-v traegt.** Ein
> frischer knoten ist ein PLATZHALTER; ein knopf, der dann etwas ausloeste,
> waere eine luege. `ManeuverExecutor.arm()` lehnt ohnehin ab -- die sperre ist
> die SICHTBARE haelfte einer regel, die zweimal gilt. Unter dem umbruch
> fallen alle vier plaettchen weg (planungs-werkzeug, kein fluginstrument);
> die tasten `N` / `X` und der ziehgriff an der linie bleiben.
>
> Der kompassring bekommt einen **fuenften marker** (`'node'`, sechseck in
> amber), sobald ein knoten scharf ist, und die rosette beschriftet sich dann
> mit `SNAP.NODE`. Die palette bekommt **keine fuenfte farbe**: `'node'` →
> amber, `'node_path'` → gruen. Der countdown wird `palette.danger`, sobald
> er auf `T+` springt — das ist ein ZUSTAND wie `disabled`, keine fuenfte
> bedeutungsfarbe.

## Two HUD elements read the renderer instead of the world

`hud/apsis_tooltip.py`, `hud/system_map.py` and `ManeuverGizmo` are the only
widgets that take their geometry from outside `ui/`, and each for one reason:

- **The apsis tooltip** hangs on the Ap/Pe diamond, whose screen position only
  `Renderer._draw_apsis_markers` knows — it applies the *time-dependent* frame
  transform that keeps the marker on the drawn line. The renderer therefore
  publishes `renderer.apsis_marker_hits` (`sx, sy, radius_px, is_apoapsis,
  distance_m, t_abs, alpha`, cleared and refilled every call) and the widget
  hit-tests against it. The tooltip sets `blocks_mouse = False` and picks in
  `update()` from `ctx.mouse_x/y`: it sits over the world, and a widget that
  swallowed clicks there would block body selection and camera panning for a
  pure readout. Its speed comes from `Telemetry.speed_at_radius()` — vis-viva
  on the same osculating conic AP/PE and the countdowns come from, *not* the
  predictor's velocity columns (the chord kernels write NaN there).
- **The system map** hangs under the warp bar, whose height follows font size
  and notch tab. Its `layout()` therefore reads the warp widget's
  `rect.bottom` rather than carrying a fixed y offset. Body angles are
  `atan2` of the real position relative to the parent, so the map runs on the
  world's clock and is time-warp compatible with no state of its own. Orbits
  are `ctx.draw.ring` — one SDF instance, exactly round at any size, and
  standing still (a trail behind the planet would be a second, contradicting
  clock). Clicks route through the same select-then-focus rule as
  `runtime/input.py::InputRouter.handle_world_click`.

  **The tile's width follows the CIRCLE, and the dots grow faster than the
  frame.** The plot radius is `min(w, h)/2 - _PAD`, so every pixel by which
  the tile is wider than tall is empty margin and nothing else — at 196x150
  that was 46 px per flank, more than half the map radius. Both sizes are
  now barely wider than the frame is tall (134x138 collapsed, 292x300
  expanded). What the expansion buys is not a bigger box but *readable
  bodies*: `_DOT_COLLAPSED` scales every dot with `_expand_t`, so the tile
  keeps its 2–5 px planets while the open map draws them at 3–7 px, the star
  at 11 and moons at 3.4. `_HIT_RADIUS` is a floor under the drawn radius,
  not the reach itself — an 11 px sun with an 11 px hit circle drops clicks
  on its own edge. Moon systems are clamped to the frame rect
  (`_draw_moons`), because Pluto and Neptun sit on the outermost ring and a
  full-size halo there hangs half out of the map.

## Body names in the world are set in the house font

`Renderer._body_label_style()` puts the selected body's name through SB Liquid,
uppercase, letterspaced and hard-rastered (`_build_body_label_font` rounds to
the same multiple of five as `ui/text.py`). It is the only caption *inside* the
picture, and in the system grotesque it read as foreign to the surface.
Consequence for the renderer's cache: `_get_label_texture` keys on
`(text, id(font), height, antialias, tracking)` — two fonts can report the same
height, and the old key would have served one font's glyphs under the other's
name.
