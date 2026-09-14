---
paths:
  - "spacesim/ui/devui.py"
---

# Developer tools (`ui/devui.py`, F1)

- `ui/devui.py` — **Dear ImGui developer tools only** (`F1`). `ImguiLayer` is a
  **moderngl-native** ImGui backend: it renders draw data through the shared
  `gl_ctx` and translates pygame events itself. It deliberately does *not* use
  `imgui_bundle.python_backends.pygame_backend`, because that subclasses
  `ProgrammablePipelineRenderer` and drags **PyOpenGL** back in — this project
  is intentionally PyOpenGL-free. `renderer_has_vtx_offset` is deliberately
  **not** set: moderngl's `VertexArray.render()` has no `base_vertex`, so imgui
  must not emit a non-zero `vtx_offset`. `draw_dev_panels(DevContext)` holds the
  panels. The player-facing HUD is a separate system — do not grow it here.
  The **Timing** header graphs the same four series as the `TIMING:` print line
  — `TimingHistory` is a `(5, capacity)` float32 ring buffer, one contiguous row
  per series, handed straight to `imgui.plot_lines` with `values_offset`.

> **The timing buffer is filled every frame, not only while the panel is open.**
> `DevContext.sample_timings()` is called from `runtime/loop.py` unconditionally, right
> **after `renderer.present()`** — `render()` sets `swap_or_present_ms = 0.0` and
> only `present()` fills it in, so sampling any earlier records `render draw` as
> a constant zero. Sampling inside `draw_dev_panels` instead would leave the
> history empty at exactly the moment you press `F1` to look at it. The price is
> per-frame work in the main loop, so `push()` does five scalar stores into
> pre-allocated rows — no `append`, no dict, no `np.roll`. Measured **1.0 µs per
> frame** (0.018 % of a 5.6 ms frame); the whole-app A/B is 6.10 → 6.00 ms
> median, i.e. below the 0.1 ms resolution of that measurement. With the panel
> open the four plots cost ~0.1 ms.
>
> Two things the graphs must keep saying out loud. **`predictor compute` is a
> worker thread** — it is the duration of one job, not main-thread load, so it
> is the one series with *no* frame-budget line; drawing one there would invite
> exactly the misreading the label exists to prevent. And **`render draw` is
> mostly the VSync wait**, so a high value there means slack, not a problem.
>
> The axis snaps to a 1/2/5 ladder (`_nice_ceiling`) over a peak that decays
> with `tau = 0.5 s`. Both halves are load-bearing: an axis that tracks the
> running maximum continuously keeps the curve the same height forever, so you
> see every change of *shape* and none of *magnitude*; without the decay a
> single spike pins the axis and the normal signal is unreadable underneath it.
> `tests/devui_timing_test.py` covers the ring's wraparound order (imgui reads
> `values[(i + offset) % n]`, so `offset` must point at the **oldest** sample —
> point it at the newest and the graph runs backwards while still looking
> plausible), stats over a partially filled buffer, `resize` keeping the newest
> samples, and the 20 µs ceiling on `sample_timings`.
