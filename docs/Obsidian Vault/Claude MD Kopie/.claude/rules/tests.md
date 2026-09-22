---
paths:
  - "spacesim/tests/**"
  - "spacesim/tests/*.py"
---

# Test map

**No pytest setup and no `requirements.txt`** — run each file directly
(`python tests/energy_test.py`), from inside `spacesim/`.

## Physics

- **`energy_test.py`** — energy-conservation regression. Pinned values: RKN4
  fixed-step `4.2571e-07`, Verlet `4.2230e-10`. It sets
  `integrator_position_tolerance = 1e99` itself and drives fixed steps, so
  config changes to the integrator ceiling/tolerance do **not** move it.
  Re-run it after any change to release logic or to `_serialize_for_kernel`.
- **`hamiltonian_diag.py`** — a **diagnostic, not a test**. Separates numerical
  from physical energy drift across three scenarios (fixed Earth / scripted
  Earth / plus Moon) and answers whether a symplectic integrator could help
  here at all.

## Predictor and time warp

- **`warp_predictor_test.py`** — the big one. Covers the prediction horizon
  surviving zoom; the Numba integrator being bit-identical to the Python one;
  the warp hold neither jittering nor running dry; real time being reachable at
  any frame rate (§6, 30–240 fps, both the HUD button and `PageDown`); the
  prediction returning after a `reset()`; and the orbital-direction tangent
  surviving the warp hold's shrinking head chord. Numbered sections:

  | § | asserts |
  |---|---|
  | 9 | thrust does not block the main thread — `update()` inside the frame budget under sustained thrust, the line neither emptied nor frozen, settling back to the coasting noise floor after burnout |
  | 10 | the body-placement memo is bit-identical to the un-memoised path (5 configurations, incl. a two-link moon chain) |
  | 11 | Ap/Pe markers land on an analytic e=0.5 ellipse's radii (8.0002e6 vs 8.0e6, 2.4002e7 vs 2.4e7) |
  | 12 | the thrust pipeline never swaps a result backwards — strictly ascending job ids *and* strictly increasing snapshot velocity |
  | 13 | thrust is detected near periapsis but not while coasting or under warp (12/12 and 0/12 at four points; 0/6 under a 7 d/s step) |
  | 14 | swap pacing produces no double-steps |
  | 15 | the far-field step ceiling costs the same at periapsis as at apoapsis (fails at 5.80 against the old code) |
  | 16 | the whole time-warp cost ladder |
  | 17 | a warp-step change does not stall the main thread |
  | 18 | the world's step order matches the prediction's physics, and the held line does not walk away from the ship |
  | 19 | the synthetic head does not fake an apsis |
  | 20 | a long horizon does not bend the orbit |
  | 21 | the orbital timescale is continuous along a lunar transfer and never permits more than a quarter period per frame |
  | 22 | world state does not depend on how the frame is chunked |
  | 23 | time warp leaves the drawn line untouched (counter-check: the uncapped rule moves it 2.3e6 m) |
  | 24 | the step ceiling is **local**: bit-identical to the old global clamp on four orbits that stay in their regime (LEO 1x/64x, e=0.6 64x, lunar transfer 4x — same step count, 0.000e+00), and 40–44× fewer steps on two Earth→Jupiter departures while drawing the same line to 0.0006 px |

- **`prediction_projection_test.py`** — §1 pins the whole line-drawing chain by
  drawing the same line through the batch and the scalar path and demanding a
  `0.000e+00 px` difference. Run it after touching any line kernel. Its
  `draw_runs` takes `refresh=False` for the second render of a pair, since the
  sync-rebuild side effect it used to rely on is gone.
- **`prediction_detail_test.py`** — measures the **drawn** line against an
  analytically known circle: the tolerance ladder snapping onto the right rung,
  the reported interpolation floor matching `c⁴/384R³`, the drawn error staying
  under the promised rung and converging to that floor when the budget is
  raised, detail never shortening the horizon, a tight point budget coarsening
  rather than truncating, and off-screen segments getting no sub-points.
  Measured 17 850 m → 503 m at default detail, → 11.8 m with the budget raised.
  §7 (2026-08-31) covers the **raw** scan budget on a long horizon: a 40 000
  point Erde → Neptun line with a flyby at 70 % of its length, where the even
  subsample picks **0 of the 6** stored points crossing the encounter and
  misses the true arc by **278 px**, and lets the line wander **295 px per
  frame** as the warp hold consumes the head. View-aware: **6 of 6**, 2.02 px,
  and **0.000e+00 px** of movement. Both no-op cases (line shorter than the
  cap, everything on screen) are asserted to leave the old selection alone.
- **`apsis_stability_test.py`** (2026-08-25) — the Ap/Pe markers standing still
  **in real time**: the marker not moving while the orbit does not (measured
  against the distance the ship covers in the same time — exactly the amount
  the old rigid shift was off by), the point list neither growing nor shrinking
  as its synthetic head is prepended and stripped, a shifted time column not
  displacing the reference body, and the warp hold not getting worse from
  sharing the mechanism.
- **`horizon_targets_test.py`** — the pure length rule of the HUD horizon
  slider: §1–3 `test.horizon_targets()` against `predictor_horizon_lengths`
  (grabbing pins the computed length at the ceiling, `drawn` follows the knob),
  §4 the `set_length`-once-per-grip contract against a real `Predictor`, and
  §5 (2026-09-02) that `get_points()` returns the **same array object** when
  `set_display_length` moves less than one `_display_quantum` (8 points) and a
  new one past that — the view-identity churn fix.

## Frames, lines, camera

- **`orbit_lines_test.py`** — pure, headless. The batched future track landing
  on the **plotting frame's own Kepler model** over a two-link moon chain (§2,
  with a counter-check that the world model misses it by >1e9 m, and that a
  moon never leaves its orbital shell at four horizons); the single call being
  bit-identical to the batch; the SOI alpha and reveal bands; the parabolic
  refinement of the closest approach; the per-time frame projection and its
  shared-timeline cache; the `FrameAffineTable` against the scalar loop (with a
  2-knot counter-check); the origin body collapsing to a point in its own frame
  (counter-check: the old rigid transform draws a 40× longer curve); the end
  caps landing on one point for a constructed intercept and separating under a
  30-minute phase shift; and (§12) the stride never dropping the final sample,
  over six point counts and the whole stride ladder. §17 (2026-08-31) is the
  **angular floor** of the future track: over a 30-year window the shared grid
  gives the Mond 752° per sample (asserted as the counter-check), the drawn
  grid 5.63°; the end cap still lands **exactly** on the body at the horizon's
  end time; a window short enough keeps the *same array objects*; and a body
  with no approach gets no private grid at all. It also pins the **plot-frame**
  period: Erde in the Mond frame runs at 27.45 d, not 365 d, and the shared
  grid there violates Nyquist at 2.09 revolutions per sample (the counter-check
  is stated as that ratio, not as a drawn angle — past 180° per sample `unwrap`
  cannot recover the step, and 2.09 revolutions reads as a harmless 36°).
- **`frame_origin_interp_test.py`** — pure, headless. Knot values staying
  bit-exact, a cubic reproduced exactly, the cubic beating the old linear chord
  at three horizon lengths (each with a counter-check that linear fails the
  same bound, 285×–73608× worse), batch equalling scalar bit-for-bit, and
  `q <= 0` still taking the exact path. §5 (2026-08-31) covers the case where
  no interpolation order helps: a **moon** in the origin over a 3650 d horizon,
  where the knot spacing exceeds half its period. Counter-check with the floor
  disabled measures **4.33e8 m** off the exact value and **4.53e8 m of drift
  per frame** as the window slides; with the floor both are **0.000e+00 m**.
  A planet in the origin is asserted to stay on the interpolated fast path.
- **`selection_camera_test.py`** — the centre-anchored zoom moving the followed
  body by exactly 0 px under warp (§4/§5, counter-check 4726 px with cursor
  anchoring); a pan detaching the follow and leaving the camera at world
  velocity 0 (counter-check: a body flies 75 px out of frame in 5 s);
  `focus_on` not jumping and landing exactly on a *moving* body; the selection
  leaving the reference body and the change notification alone; and — against
  real pixels — `pick_body` hitting the drawn disc in a rotating frame, the
  marker sitting as four arrows around it, and (§8/§10) the body-label modes.

## Rendering and UI

- **`render_budget_test.py`** — §1 pins the frame-timing attribution
  (`frame_ms` = `render()` itself, `overlay_ms` separate) with a counter-check
  that the old arithmetic swallowed the gap; §3 pins that nothing is allocated
  per frame.
- **`background_test.py`** — the backdrop layer, headless (no GL). §1 is the
  load-bearing one: every lattice node satisfies **all three** line equations
  to 0, all six neighbours sit at `2/sqrt(3)*ws`, and a counter-check convicts
  the design mockup's formula of missing by half a cell. §2 sweeps the zoom
  over twelve decades and pins that no decade's alpha ever jumps (max 0.0096
  per step) — the "no hard threshold anywhere" rule. §3 star determinism (7
  columns). §4 the drift model, which is the other load-bearing one: the same
  flight at five zoom levels 14 decades apart must move the stars
  **bit-identically** (`0.000e+00 px`), plus the sign convention on both axes,
  the clamp, and the free-camera path — a pan of one screen width must move the
  stars equally far at every zoom, because read as a *world* velocity the same
  pan is 1e12 m/s at 1e-9 px/m and saturates the clamp. §4c the velocity
  **source**: it first convicts `solar_system.json` — all 27 scripted bodies
  carry `velocity: [0, 0]`, so the field is not a usable source and the star
  layer read zero for everything but the ship — then pins that a body with no
  velocity field still moves the stars, that dividing by *sim* time keeps
  1 d/frame identical to 1 s/frame under warp, and that a focus change re-seeds
  instead of reading 1e11 m as a flight (counter-check: read as a flight it
  saturates the clamp at 90.6 px). §4b the breathing
  field — expected visible star count varies under 3 % over eight octaves, the
  fade window is exactly zero at *both* octave edges, and `amount = 0`
  reproduces the rigid field exactly. §5 the idle fade, §6 grid phase at 1e11 m
  offsets and the accent fallback colour, §7 that `config.json`,
  `BackgroundLayer` and the dev-UI header carry the identical key set (it
  catches a new runtime field that was not added to the runtime-only exclusion
  — it has, three times now). §8 the grid anchor as a **fixed lattice in the
  plot frame**: the reference body standing still, a pan shifting it by exactly
  the pan distance (0.000e+00 m over 30 km), a circular orbit tracing a closed
  circle (axis ratio 1.000000), and `"focus"` taking that motion back away.
  §8b the rate limit: below it the motion is exact and `grid_lag_px` is 0;
  above it the slide is exactly `grid_max_speed_px` whatever the true speed
  (counter-check: ungoverned, 500× faster); the default exceeds the camera's
  pan rate; `0` freezes it; the zoom fixed point is the screen centre
  (7.1e-13 px, counter-check convicting a pixel-drift at −D); and a `grid_key`
  change is taken over rather than driven across. Its slide measurements
  **unwrap the anchor's fold** first — the raw value is not a distance, and one
  case is deliberately sampled ten times finer because at 7.7e5 px/s a frame
  covers more than half a fold period and cannot be unwrapped at all.
  §8d the fold is a real **lattice translation**: shifting by `fold_spans()`
  leaves every visible decade's phase untouched (5.3e-14 cells), with a
  counter-check that the naive `modulo 10^k` moves it **0.79 cells** — the x
  period carries a `√3` because the nodes sit at `q·ws/√3`.
  §8c the raster cell: `pixel_round = 0` fills the cell completely (a partial
  fill would put holes in a plain pixel grid), `1` leaves 52.8 %, fill falls
  monotonically, and that 52.8 % matches the `CELL_FILL_ROUND` constant the
  shaders divide by.
- **`background_gl_test.py`** — the same layer on a **real framebuffer**, and
  the only test that could have caught the bug it exists for: `star.frag` read
  `gl_PointCoord`, which this machine's NVIDIA driver fills with `(0, 0)` in
  every fragment, so the cell mask discarded every star and the field was
  **completely invisible at the shipped defaults**. `background_test.py` was
  green throughout. It asserts that the values from `config.json` — not some
  convenient setting — put ink on screen, at every `pixel_round` and not just
  at 0; that the round cell sets fewer pixels than the square one (so the mask
  is doing anything at all); that `grid_enabled` and `enabled` switch exactly
  their own ink; and that a pan changes pixels, which is what would have caught
  the `u_level_phase` write failing silently. §6 drives the real
  `_draw_background` with a body whose `velocity` is zero — the plumbing half
  of §4c — and demands 14.9 px/s of star drift out of it. Every comparison pins
  `grid_fade` and `time_s` first — the fade ramp and the twinkle otherwise
  drift between the two shots and get counted as the layer under test.
- **`ship_scale_test.py`** — the ship's zoom-dependent screen size: the
  smoothstep-in-log ramp between `ship_zoom_shrink_start_scale` and
  `ship_zoom_shrink_end_scale`, and that every path goes through
  `_ship_length_px()`. **Switches the background layer off** — its ink test
  ("bright and low-saturation") otherwise matches the grid lines and reports
  the full frame width at every zoom. `body_style_test.py` does the same.
- **`body_style_test.py`** — the build is deterministic in its seed; nothing
  leaves the unit circle (§2); the draw order; **line width measured on screen
  at two zoom levels** (§5 — the whole point of staying vector; measuring ink
  over the whole drawing instead gives 0.88 and proves nothing); the lighting
  following the star; pixel-identity below the detail threshold (§7, with a
  counter-check at 200 px so it cannot pass vacuously); one build rather than
  one per frame; and the detail ladder clamping at its ends and not earlier
  (§9).
- **`ui_units_test.py`** — the compact formatters. Pure, runs headless.
- **`ui_render_test.py`** — opens a GL window, renders real frames and
  **asserts on pixels read back from the framebuffer**. §9: a negative corner
  radius really is a chamfer and not a rounding (probe at `(0.35c, 0.35c)` —
  do **not** move it to `0.3c`, that lands in the antialiased edge and measures
  a half value), and `cut_corners` leaves the named corner sharp. §10: the
  pixel face rasters to exactly two alpha values and snaps onto the five-pixel
  ladder at every `ui_scale`, with a non-zero-fringe counter-check on the text
  face.
- **`ui_hud_test.py`** — orbital elements against analytically known orbits,
  layout/overlap at real window sizes, and that controls actually move
  simulation state. §8 opens the body drawer and runs the unfold out before
  clicking a row. §10: the attitude ring rejects all four corners of its square
  and the throttle arc's 4/50/96 % points route to `NavballCluster` (with a
  counter-check that the 96 % point really lies inside the ring's rect). §11:
  tabular figures — the countdown renders at a constant width, and the raw font
  is *not* monospaced. §12: the velocity needle scale at four bodies, both ends,
  and the no-reference-body path. §13: the flank strips sweep eleven magnitudes
  × both signs × four `ui_scale` values without overflow, and the label
  survives in most cases.
- **`devui_timing_test.py`** — pure, headless (no GL, no imgui frame). The
  ring's wraparound order (imgui reads `values[(i + offset) % n]`, so `offset`
  must point at the **oldest** sample — point it at the newest and the graph
  runs backwards while still looking plausible), stats over a partially filled
  buffer, `resize` keeping the newest samples, and the 20 µs ceiling on
  `sample_timings`.

`tests/body_icon_test.py` — die **Positions-Marke** (`bodies/icon.py`). §1 der
Seed (gleicher Seed → gleiche Packung, zwei Namen → zwei Marken, und
namentlich **Ganymed gegen Oberon**, die sich eine Farbe teilen). §2 nichts ragt
aus dem Einheitskreis, sonst läge sichtbares Muster außerhalb des Greifradius.
§3 die Bit-Packung ist verlustfrei und 7×7 passt noch in vier uint32. §4 echte
Pixel: die geschifften Vorgaben setzen Tinte, `body_icon_style = "disc"`
zeichnet nachweislich etwas anderes, und die beiden farbgleichen Monde
unterscheiden sich um 102 Pixel. **§5 ist der Kern**: über einen Pixel Drift
bleibt die Gesamthelligkeit auf **±0.10 %** stabil, während sich das Bild sehr
wohl ändert — mit einer Gegenprobe bei `body_icon_edge_px = 0`, die auf ±7.46 %
springt. Ohne die Gegenprobe bestünde den Test auch eine Marke, die gar nicht
glättet. §6 die Überblendung fällt monoton und ohne Sprung > 0.25 — **und die
Tiefe**: die Marke muss mindestens 14 unterscheidbare Helligkeiten tragen
(gemessen 28), mit einer Gegenprobe bei `shade_jitter = 0`, die darunter
fällt. Drei Stufen allein gaben zu wenig, und der flache Klecks der ersten
Fassungen wäre sonst durchgegangen; an einer Zahl hätte es niemand gesehen.
Dazu die **Achsen-Symmetrie** der Zell-Umrisse: dunkle Linien im Spalten- und
im Zeilenmittel, beide mindestens zwei und keine unter der Hälfte der anderen,
mit einer Gegenprobe bei `cell_rim = 0`. Solange die Umrissbreite ein Anteil
der Zelle war, lag sie unter einem Pixel und fiel je nach Bruchteil-Position
auf einer Achse ganz aus. §4 prüft die Marken-Größe nicht mehr
gegen die Zahl 8, sondern gegen das, was sie bedeutet: ein Raster ab 9 und ein
Überblend-Band, das nicht verkehrt herum steht. §3b pinnt, dass allein das
**Raster** den Detailgrad bestimmt (streng mehr Zellen je Stufe), und §6 die
Auflösungsgrenze: bei der geschifften Größe muss 9×9 mehr Kanten zeigen als
5×5, bei Radius 24 muss 15×15 mehr zeigen als beide (108 → 139 → 265), und die
geschiffte Kombination muss über 1.5 px je Zelle bleiben. Die zweite Messung
ist der Beleg für den Halbzellen-Deckel im Box-Filter — ohne ihn frisst ein
feineres Raster sich selbst auf. §7 die **Größen-Skalierung nach dem
PHYSISCHEN Radius** — die erste Fassung skalierte mit dem Bildschirmradius
und hatte deshalb im Spiel bei jedem `size_influence`-Wert keine sichtbare
Wirkung (eine Marke existiert nur, solange der Körper kleiner als `min` ist,
also blieb die Mischung immer unter `min` und wurde zurückgeklemmt — ein Test
mit absichtlich großen Werten bestand trotzdem, weil er nicht die Situation
prüfte, in der das Feature benutzt wird). Jetzt gegen eine von Hand gesetzte
Radius-Spanne (`renderer._icon_radius_range_m`) gemessen: bei
`size_influence = 0` bleibt jede Marke bei `min`, unabhängig vom Radius; bei
`1` bekommt der kleinste geladene Körper genau `min`, der größte genau `max`,
die log-Mitte der Spanne genau die Mitte von `[min, max]`, und ein Radius
außerhalb der Spanne wird geklemmt statt extrapoliert; bei `0.5` trifft es
den linear gemischten Wert; ein „Mond" (2·10⁶ m) und ein „Planet" (7·10⁷ m)
bleiben bei `influence = 1` **unterschiedlich groß, obwohl ihr
Bildschirmradius in diesem Test bewusst identisch winzig ist** — genau das
ist der Gegenbeweis zum alten Fehler; und der Greifradius
(`_pick_radius_px`) folgt in jedem Fall genau dem, was
`_body_icon_draw_radius_px` zeichnet, auch wenn der Bildschirmradius (durch
eine winzige `camera.scale`) fast null ist.

## Manoeverknoten

Sieben dateien, von rein rechnend nach GL geordnet. Alle **gruen** gemessen
2026-09-05.

- **`maneuver_profile_test.py`** — das rampenprofil. Beide zweige gegen von
  hand gerechnete werte, die stetigkeit an der nahtstelle (0.99999833 gegen
  1.00000167 s), die SYMMETRIE, aus der der zuendzeitpunkt folgt (halbes Delta-v
  bei halber zeit, an vier profilen), bildratenunabhaengigkeit von
  `dv_between` (7 / 61 / 1000 schritte, alle 1200.000000000), und §8: der
  **njit-zwilling** `_profile_accel_numba` gegen `BurnProfile.accel_at` an
  2004 stuetzstellen auf EXAKTE gleichheit (gemessen 0). Ohne §8 koennten
  vorschau und ausfuehrung unbemerkt auseinanderlaufen.
- **`maneuver_plan_test.py`** — pur. Die orbitale basis (normal zeigt an 16
  bahnpunkten nach innen), die entartungen (v = 0 → None; radialflug →
  trotzdem orthonormal), die richtungsmischung, die zeitliche sortierung des
  plans, der fuenfer-deckel, und dass `plan.version` **jede** aenderung zaehlt
  — eine direkte feldaenderung ausdruecklich NICHT, bis `touch()` faellt.
- **`maneuver_burn_kernel_test.py`** — der RK4-brennbogen. Ohne schwerkraft
  ist er reine kinematik: geliefertes Delta-v == geplantes auf 1e-6 relativ
  (gemessen 120.000000000) und exakt in schubrichtung. §4 ist die gegenprobe,
  ohne die §1 auch bestuende, wenn die schwerkraft fehlte: der gleitbogen
  gegen die **analytische** kreisbahn nach 600 s (0.000000 m). §5 dieselbe
  bahn gegen `_compute_distance_points_numba_state`. **`precision` dort ist
  ein BOGENABSTAND, kein zeitschritt** — mit 1 m je punkt decken 3000 punkte
  0.5 s ab, nicht 600.
- **`maneuver_preview_test.py`** — die kette. Ein knoten ohne Delta-v laesst
  die bahn stehen; prograde hebt das apoapsis und retrograde senkt das
  periapsis (das findet einen vertauschten basisvektor); normal kippt beide
  apsiden; §6 der ZWEITE knoten sitzt nachweislich nicht mehr auf der
  ursprungsbahn (1.2e7 m daneben); §9 ein knoten jenseits der reichweite wird
  ehrlich fallengelassen; §8 `maybe_rebuild` rechnet ohne aenderung NICHT
  (synchron geprueft, sonst haengt die antwort daran, wie schnell der
  arbeiter fertig wird); **§8b der nebenlaeufige weg liefert EXAKT dieselben
  zahlen wie der synchrone** (abweichung 0.0) und kostet den hauptthread
  einen bruchteil; §8c verdoppelter `length_mult` = rund verdoppelte
  reichweite bei GLEICHER punktzahl — das ist der beleg, dass die reichweite
  im punktabstand sitzt und nicht in der punktzahl.
- **`maneuver_execute_test.py`** — der autopilot. §1 die einheitenfrage
  (`a_max = thrust_acc/realtime_warp_max`, gegengerechnet gegen eine
  echtsekunde pfeiltaste); §4 die schrittklemme springt bei 1-stunden-frames
  **nie** ueber die zuendung (ueberschuss 0.000e+00 s); §5 der volle
  brennvorgang liefert 120.000000000 von 120.0; §7 handeingabe bricht ab und
  laesst den knoten im plan; **§8 ist die hauptsache** — geflogen gegen
  vorgeschaut, 1.72 m/s auf 25 433 m/s und 3944 m auf r = 1.04e7 m.
- **`maneuver_render_test.py`** — GL. Nicht wie es aussieht, sondern was das
  HUD anfassen kann: `maneuver_node_hits` mit vier griffen, paarweise
  gegenueber (skalarprodukt −1.000000) und senkrecht (0.000e+00); die liste
  wird jeden durchgang geleert; `maneuver_curve_screen` entsteht nur waehrend
  eines MARKER-zugs (ein griff-zug baut sie nicht — er verschiebt den knoten
  gar nicht, und die kurve kostet eine projektion je punkt). **§3b vergleicht
  den array-weg gegen den punktweisen**: abweichung 0.000e+00 px. Ohne diese
  gegenprobe waere die schnelle projektion eine unbewiesene abkuerzung.
  **§3c die GLATTHEIT ueber drei reichweiten** (x1/x4/x16): gemessen wird
  der groesste abstand eines gerechneten kurvenpunkts von der GEZEICHNETEN
  strecke — 0.025 / 0.220 / 0.318 px bei 384 / 444 / 479 punkten gegen ein
  budget von 480. Die gegenprobe faehrt dieselbe linie ohne verfeinerung
  und verlangt, dass sie nicht besser ist (mit einem zehntel pixel
  spielraum, weil die zwischenpunkte kubische naeherungen sind) und dass
  der faktor ueber die stufen mindestens einmal ueber 2 liegt (4.17).
  §3d die endkappen: genau eine je gezeichneter linie, koerperkreise mit
  dem ECHTEN radius (gegen `body.radius * scale` gerechnet), alle in gruen,
  und `maneuver_end_caps = False` zeichnet keine.
  **Er setzt den plot-frame ueber `FrameController` auf** — ohne
  ihn rechnet der renderer barycentrisch, der knoten liegt Gigameter neben dem
  bild und der treffertest meldet null marker bei voellig gesunder kette.
  `frames.apply()` macht die vorhersage ungueltig, danach muss
  `predictor.update()` einmal laufen.
- **`maneuver_hud_test.py`** — GL. Sein `frame()` ruft `root.render()` MIT:
  ohne das laeuft `draw()` im test nie, und ein undefinierter name dort
  faellt erst im spiel auf (einmal passiert). §1 die vier plaettchen teilen
  x-lage und breite mit ihrer flanke (gegen `flank_rect()` / `strip_rect()` gerechnet,
  nicht gegen zahlen im test) und sitzen ueber bzw. unter ihr; §2 dasselbe
  in drei fenstergroessen plus **jede flaeche gegen jede** — apsiden-leiste,
  rosette und die vier untereinander; eine aufteilung, die sich selbst
  ueberlappt, ist nur eine umverteilung des gedraenges. §3 die SPERRE
  (EXECUTE abgeblendet ohne Delta-v, in beide richtungen). **§4 das
  eingabefeld, ueber die ECHTE ereigniskette** (`root.handle_event`, weil
  genau daran haengt, wem die tastatur gehoert): der klick oeffnet es und es
  bleibt ueber den frame hinaus offen, die ziffern landen im PUFFER und
  waehrend des tippens steht `plan.version` still (die gegenprobe zum
  schreiben je anschlag), erst enter schreibt in den knoten, escape
  verwirft, leergeloescht-und-abgeschlossen schreibt 0.0, ein buchstabe
  wird VERSCHLUCKT statt durchgereicht (sonst setzte ein 'n' beim tippen
  einen knoten), ein zweiter punkt abgelehnt; das feld oeffnet MIT dem
  stehenden wert (ohne tote '.0') und markiert, die erste ziffer ersetzt ihn
  als ganzes (gegenprobe: angehaengt waere es '2503'); die PFEILE
  navigieren, statt zu loeschen -- die gegenprobe zu `'' in '0123456789'`,
  das fuer jede taste ohne zeichen wahr ist und das feld beim ersten
  pfeildruck leerte --, entf loescht unter dem caret, rueckschritt davor,
  getippt wird an der schreibstelle ('250' -> pfeil links -> '9250'), und
  ein klick ins offene feld setzt den caret ans linke bzw. rechte ende; §4b angeklickt und nicht getippt aendert
  NICHTS; §4c der richtungsknopf dreht das vorzeichen und nimmt die tastatur
  NICHT, das rad stellt einen feinschritt. §5 pips, §6 mausverbrauch, §7 der
  fuenfte ringmarker, §8 der umbruch; §9 der ziehgriff faengt seine radien
  und eine leere stelle daneben **nicht**; **§10 der griff als KNUEPPEL** —
  in der ruhelage laeuft nichts, voller ausschlag eine sekunde liefert
  `handle_dv_rate`, halber ausschlag bleibt unter einem drittel davon (die
  quadratische kennlinie), losgelassen steht der wert still; §11 dasselbe
  rueckwaerts am retrograde-griff; §12 den marker verschieben rastet auf die
  linie; §13 ein knoten in der VERGANGENHEIT faerbt seinen countdown
  `palette.danger` statt amber (der gegencheck, dass es NICHT die node-farbe
  ist, gehoert dazu); §14 der reichweiten-regler ist ein eigener, so breit
  wie die ALT-flanke, und klemmt an beiden enden.

## Known pre-existing failures — not regressions

**Drei dateien laufen seit der umstrukturierung (2026-09-03) gar nicht mehr
an** — sie tragen die alten flachen pfade:

- `energy_test.py` — `from vec import Vec2`; das modul heisst jetzt
  `physics.vec`.
- `hamiltonian_diag.py` — `from vec import Vec2, vec`, dieselbe ursache. Es
  ist ohnehin ein diagnose-skript, kein test mit pruefungen.
- `background_test.py` — liest `spacesim/solar_system.json`; die datei liegt
  jetzt unter `config/solar_system.json`.

Alle drei brechen mit `ModuleNotFoundError` bzw. `FileNotFoundError` ab, bevor
irgendetwas geprueft wird. Das ist kein befund ueber den code, den sie
pruefen sollten — wer sie braucht, muss zuerst die pfade nachziehen.

`body_icon_test.py` §5 misst hier **±2.24 % gegen eine grenze von 2 %**,
reproduzierbar ueber mehrere laeufe (die notiz oben nennt ±0.10 %). Der test
ruft `renderer._draw_body_icon` direkt auf und geht **nie** durch
`renderer.render()`, haengt also an keiner der spaeteren zeichen-domaenen.


`orbit_lines_test.py` has three **timing budgets** that this machine misses:
"eine volle neuberechnung unter 5 ms", "ruhiger frame unter 0.20 ms" and "jeder
weitere koerper unter 25 us". They are not correctness checks.

> **They drift with the machine's thermal state, not with the code.** Measured
> 2026-09-03 across one session: the *same* pre-restructure baseline tree gave
> **median 9.955 ms and 1 failure** early on, and **19.365 ms and 3 failures**
> after ~45 minutes of continuous Numba/GL testing. The restructured tree
> measured 7.1-8.2 ms early and 20.2-25.8 ms late -- i.e. **identical to the
> baseline at each point in time**.
>
> So: never compare a timing budget against a number recorded in a different
> session. If one of these fails, re-run the same test on a known-good tree
> *back to back* before believing it means anything. The first run after any
> file move is worse again, because moving a source file invalidates the
> on-disk `cache=True` Numba caches ("erster 6327 us" is that JIT, not a leak).

`warp_predictor_test.py` **does not produce a clean sheet on this machine, and
has not since before the 2026-08-24 work.** Diff against a known-good run.

Reliably failing:

- "die decke senkt die teilschritte bei 1 y/s um mehr als das 20-fache" —
  measured factor 1 (1408 without the ceiling, 1280 with it). Failed in every
  run ever recorded here.
- "kaum doppelschritte beim einwechseln (puffer 1)" — 9–12 of 240 frames.

Flaky — these come and go between runs of **identical code**:

- "wechsel-frame bleibt im hauptthread billig" (all four warp transitions)
- "jede neue linie gehoert zu einem NEUEREN schiffszustand"
- "der puffer haelt die linie nicht auf"
- "nach brennschluss rastet die linie wieder ein"
- "tiefe 3 erneuert deutlich oefter als tiefe 1"

> **Measured 2026-09-03, three runs across the restructure.** Pre-restructure
> baseline: **8** failures. Restructured, cold Numba cache: **6**.
> Restructured, warm cache: **8** — and the *sets differ in both directions*.
> "nach brennschluss rastet die linie wieder ein" failed once and passed once
> on byte-identical code; the baseline uniquely failed two that the
> restructured tree passed.
>
> All of them sit in the asynchronous compute pipeline, which races a
> `time.sleep(FRAME)` loop against a `ThreadPoolExecutor`. **A changed failure
> set here is not by itself evidence of a regression.** Look instead at the
> deterministic tests: `prediction_projection_test.py` (0.000e+00 px),
> `energy_test.py` (pinned to the digit) and `apsis_stability_test.py`.
>
> A moved source file also invalidates the on-disk `cache=True` Numba caches,
> so the first run after any restructure is slow and fails *more* of these
> timing checks. Re-run before drawing conclusions.
