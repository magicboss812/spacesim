"""Regressionstest fuer die timing-ringpuffer der entwickler-oberflaeche.

Prueft `devui.TimingHistory` und `devui._nice_ceiling` -- reine daten, kein
GL, kein fenster, kein imgui-frame. Der test deckt genau die drei stellen ab,
an denen ein ringpuffer typischerweise falsch ist:

* die REIHENFOLGE nach dem umlauf (imgui liest `values[(i + offset) % n]`,
  also muss `offset` auf die AELTESTE probe zeigen -- nicht auf die neueste),
* die STATISTIK auf einem erst teilweise gefuellten puffer (der nullrest darf
  weder in den mittelwert noch in das maximum einfliessen),
* das UMSTELLEN der laenge (die juengsten proben muessen erhalten bleiben).

Dazu kommt die messung, die den eigentlichen zweck der uebung absichert: das
abtasten laeuft in JEDEM frame, auch wenn das panel zu ist, und darf deshalb
im bildbudget nicht auffallen.

Aufruf: python tests/devui_timing_test.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np

from ui import devui
from ui.devui import DevContext, TimingHistory


FAILURES = []


def check(actual, expected, label):
    if actual != expected:
        FAILURES.append(f"{label}: {actual!r} != {expected!r}")
        print(f"  FAIL {label}: {actual!r} != {expected!r}")
    else:
        print(f"  ok   {label}: {actual!r}")


def check_close(actual, expected, label, tol=1e-5):
    if abs(float(actual) - float(expected)) > tol:
        FAILURES.append(f"{label}: {actual!r} != {expected!r}")
        print(f"  FAIL {label}: {actual!r} != {expected!r}")
    else:
        print(f"  ok   {label}: {actual:.6g}")


def check_true(cond, label, detail=""):
    if not cond:
        FAILURES.append(f"{label}{(' -- ' + detail) if detail else ''}")
        print(f"  FAIL {label} {detail}")
    else:
        print(f"  ok   {label} {detail}")


class FakePredictor:
    def __init__(self):
        self.last_compute_ms = 0.0


class FakeRenderer:
    def __init__(self):
        self.last_frame_timings = {}
        self._last_prediction_render_stats = {}


def push_ramp(hist, n, base=0.0):
    """n proben, in denen jede serie einen eindeutigen wert bekommt."""
    for i in range(n):
        v = base + i + 1
        hist.push(pred_compute=v, pred_draw=v * 10.0,
                  rend_calc=v * 100.0, rend_draw=v * 1000.0,
                  frame=v * 10000.0)


# ---------------------------------------------------------------- nice ceiling
print("_nice_ceiling -- die leiter 1/2/5 je dekade")
check(devui._nice_ceiling(0.4), 0.5, 'unter eins')
check(devui._nice_ceiling(0.9), 1.0, 'knapp unter eins')
check(devui._nice_ceiling(1.0), 1.0, 'genau eins')
check(devui._nice_ceiling(1.2), 2.0, 'auf zwei')
check(devui._nice_ceiling(2.0), 2.0, 'genau zwei')
check(devui._nice_ceiling(2.1), 5.0, 'auf fuenf')
check(devui._nice_ceiling(5.0), 5.0, 'genau fuenf')
check(devui._nice_ceiling(6.0), 10.0, 'auf zehn')
check(devui._nice_ceiling(11.0), 20.0, 'zwanzig')
check(devui._nice_ceiling(23.0), 50.0, 'fuenfzig')
check(devui._nice_ceiling(51.0), 100.0, 'hundert')
# Ein leerer puffer darf keine 0-hohe achse liefern (division durch null im
# budget-strich) und kein NaN.
check_true(devui._nice_ceiling(0.0) > 0.0, 'null ergibt positive achse',
           f"{devui._nice_ceiling(0.0)}")
check_true(devui._nice_ceiling(float('nan')) > 0.0, 'NaN ergibt positive achse',
           f"{devui._nice_ceiling(float('nan'))}")
check_true(devui._nice_ceiling(-3.0) > 0.0, 'negativ ergibt positive achse',
           f"{devui._nice_ceiling(-3.0)}")

# ------------------------------------------------------------------ teilfuellung
print()
print("teilweise gefuellter puffer")
hist = TimingHistory(capacity=8)
check(hist.count, 0, 'frisch leer')
window, offset = hist.window('pred_compute')
check(len(window), 0, 'leeres fenster hat laenge 0')
check(offset, 0, 'leeres fenster hat offset 0')
cur, avg, peak = hist.stats('pred_compute')
check((cur, avg, peak), (0.0, 0.0, 0.0), 'leere statistik ist null')

push_ramp(hist, 3)                     # 1, 2, 3
check(hist.count, 3, 'drei proben gezaehlt')
window, offset = hist.window('pred_compute')
check(offset, 0, 'unter der kapazitaet ist der offset 0')
check(list(window), [1.0, 2.0, 3.0], 'fenster ist der gefuellte praefix')
cur, avg, peak = hist.stats('pred_compute')
check(cur, 3.0, 'cur ist die JUENGSTE probe')
check_close(avg, 2.0, 'avg ignoriert den nullrest')
check(peak, 3.0, 'max ignoriert den nullrest')
# Der nullrest darf das maximum auch dann nicht bestimmen, wenn alle werte
# negativ waeren -- hier positiv gehalten, aber der praefix muss exakt sein.
check(list(hist.series('pred_draw')), [10.0, 20.0, 30.0], 'zweite serie parallel')

# ---------------------------------------------------------------------- umlauf
print()
print("umlauf -- offset zeigt auf die AELTESTE probe")
hist = TimingHistory(capacity=4)
push_ramp(hist, 6)                     # 1..6, es bleiben 3,4,5,6
check(hist.count, 4, 'auf kapazitaet begrenzt')
window, offset = hist.window('pred_compute')
check(len(window), 4, 'volles fenster')
check(offset, 2, 'offset = schreibmarke = aelteste probe')
# Genau das, was imgui.plot_lines rechnet.
read_back = [float(window[(i + offset) % len(window)]) for i in range(len(window))]
check(read_back, [3.0, 4.0, 5.0, 6.0], 'imgui-lesereihenfolge ist chronologisch')
check(list(hist.series('pred_compute')), [3.0, 4.0, 5.0, 6.0], 'series() alt -> neu')
cur, avg, peak = hist.stats('pred_compute')
check(cur, 6.0, 'cur nach umlauf')
check_close(avg, 4.5, 'avg nach umlauf')
check(peak, 6.0, 'max nach umlauf')
# Das fenster muss ein zusammenhaengender speicherblock sein -- imgui liest
# ueber einen rohen zeiger, eine kopie waere still falsch bzw. teuer.
check_true(window.flags['C_CONTIGUOUS'], 'fenster ist zusammenhaengend')
check(window.dtype, np.dtype('float32'), 'float32 fuer plot_lines')

# ------------------------------------------------------------------ groesse
print()
print("laenge umstellen erhaelt die juengsten proben")
hist = TimingHistory(capacity=8)
push_ramp(hist, 8)                     # 1..8
hist.resize(4)
check(hist.count, 4, 'verkleinern behaelt vier')
check(list(hist.series('pred_compute')), [5.0, 6.0, 7.0, 8.0], 'die JUENGSTEN vier')
hist.resize(6)
check(hist.count, 4, 'vergroessern erfindet nichts dazu')
check(list(hist.series('pred_compute')), [5.0, 6.0, 7.0, 8.0], 'inhalt unveraendert')
hist.push(pred_compute=9.0, pred_draw=0.0, rend_calc=0.0, rend_draw=0.0, frame=0.0)
check(list(hist.series('pred_compute')), [5.0, 6.0, 7.0, 8.0, 9.0],
      'nach dem vergroessern weiterschreiben')
hist.resize(4)   # gleiche kapazitaet zweimal -> darf nicht neu allozieren
before = hist.series('pred_compute')
hist.resize(4)
check(list(hist.series('pred_compute')), list(before), 'resize auf gleiche laenge ist stabil')

# -------------------------------------------------------------------- pause
print()
print("pause friert ein, reset leert")
hist = TimingHistory(capacity=8)
push_ramp(hist, 3)
hist.paused = True
push_ramp(hist, 5, base=100.0)
check(hist.count, 3, 'pause nimmt nichts an')
check(list(hist.series('pred_compute')), [1.0, 2.0, 3.0], 'inhalt unter pause unveraendert')
hist.paused = False
hist.push(pred_compute=4.0, pred_draw=0.0, rend_calc=0.0, rend_draw=0.0, frame=0.0)
check(list(hist.series('pred_compute')), [1.0, 2.0, 3.0, 4.0], 'nach pause weiter')
hist.reset()
check(hist.count, 0, 'reset leert')
check(hist.stats('pred_compute'), (0.0, 0.0, 0.0), 'statistik nach reset')

# ------------------------------------------------------- achse / spitzenwert
print()
print("achsenskala -- spitzenwert zerfaellt zeitabhaengig")
hist = TimingHistory(capacity=16)
for _ in range(16):
    hist.push(pred_compute=1.4, pred_draw=0.0, rend_calc=0.0, rend_draw=0.0, frame=0.0)
scale = hist.axis_max('pred_compute', dt=1.0 / 180.0)
check(scale, 2.0, 'achse rastet auf die naechste sprosse')
# Ein einzelner ausreisser hebt die achse ...
hist.push(pred_compute=40.0, pred_draw=0.0, rend_calc=0.0, rend_draw=0.0, frame=0.0)
check(hist.axis_max('pred_compute', dt=1.0 / 180.0), 50.0, 'ausreisser hebt die achse')
# ... und sie faellt wieder, wenn er aus dem fenster gelaufen ist. Ohne den
# zerfall bliebe sie auf 50 stehen und die 1.4 ms waeren nicht mehr ablesbar.
for _ in range(16):
    hist.push(pred_compute=1.4, pred_draw=0.0, rend_calc=0.0, rend_draw=0.0, frame=0.0)
for _ in range(400):                    # ~2.2 s bei 180 fps
    axis = hist.axis_max('pred_compute', dt=1.0 / 180.0)
check(axis, 2.0, 'achse faellt zurueck')
check_true(hist.axis_max('pred_compute', dt=0.0) >= 2.0,
           'dt=0 ist kein sonderfall', f"{hist.axis_max('pred_compute', dt=0.0)}")

# Die gemeinsame achse ist das maximum ueber die GEZEICHNETEN serien -- und
# darf die (nicht gezeichnete) frame-serie NICHT enthalten.
hist = TimingHistory(capacity=8)
for _ in range(8):
    hist.push(pred_compute=1.0, pred_draw=3.0, rend_calc=7.0, rend_draw=2.0,
              frame=900.0, ui_calc=4.0)
check(hist.shared_axis_max(dt=1.0 / 180.0), 10.0,
      'gemeinsame achse ueber die gezeichneten serien')
check(list(hist.series('ui_calc')), [4.0] * 8, 'ui_calc ist eine eigene serie')

# ------------------------------------------------------------ DevContext-pfad
print()
print("DevContext.sample_timings liest die richtigen felder")
pred = FakePredictor()
rend = FakeRenderer()
ctx = DevContext(predictor=pred, renderer=rend, tick_rate=180.0)
check_true(isinstance(ctx.timings, TimingHistory), 'DevContext hat einen puffer')

pred.last_compute_ms = 17.0
rend.last_frame_timings = {'frame_ms': 9.0, 'swap_or_present_ms': 3.5,
                           'overlay_ms': 4.25}
rend._last_prediction_render_stats = {'prepare_ms': 1.25, 'draw_ms': 0.75}
ctx.sample_timings(11.0)

check(ctx.timings.count, 1, 'eine probe')
check(ctx.timings.stats('pred_compute')[0], 17.0, 'pred_compute = last_compute_ms')
check(ctx.timings.stats('pred_draw')[0], 2.0, 'pred_draw = prepare + draw')
check(ctx.timings.stats('rend_draw')[0], 3.5, 'rend_draw = swap_or_present_ms')
# frame_ms IST render() selbst -- present() schreibt es nicht mehr um, also
# wird hier nichts mehr abgezogen. Alles zwischen render() und present()
# (spieler-HUD, devtools) steht getrennt in overlay_ms -> ui_calc; frueher
# lief es unsichtbar in rend_calc mit.
check(ctx.timings.stats('rend_calc')[0], 9.0, 'rend_calc = frame_ms')
check(ctx.timings.stats('ui_calc')[0], 4.25, 'ui_calc = overlay_ms')
check(ctx.timings.stats('frame')[0], 11.0, 'frame = schleifenzeit')

# Ein renderer, dessen present() nie gelaufen ist (die GL-tests rufen nur
# render()), hat gar kein overlay_ms. Das darf keine luecke reissen.
rend.last_frame_timings = {'frame_ms': 2.0, 'swap_or_present_ms': 0.0}
ctx.sample_timings(2.0)
check(ctx.timings.stats('rend_calc')[0], 2.0, 'ohne present() bleibt rend_calc stehen')
check(ctx.timings.stats('ui_calc')[0], 0.0, 'ohne overlay_ms ist ui_calc null')

# Fehlende objekte duerfen die hauptschleife nicht sprengen.
empty = DevContext(tick_rate=180.0)
empty.sample_timings(4.0)
check(empty.timings.count, 1, 'ohne predictor/renderer trotzdem eine probe')
check(empty.timings.stats('frame')[0], 4.0, 'frame-zeit auch ohne renderer')
# Muell in den timings-dicts (None, strings) darf nicht durchschlagen.
rend.last_frame_timings = {'frame_ms': None, 'swap_or_present_ms': 'x',
                           'overlay_ms': object()}
rend._last_prediction_render_stats = None
pred.last_compute_ms = None
ctx.sample_timings(None)
check(ctx.timings.stats('frame')[0], 0.0, 'unbrauchbare werte werden zu 0')

# ------------------------------------------------------------------- kosten
print()
print("kosten des abtastens im bildbudget")
pred = FakePredictor()
pred.last_compute_ms = 17.0
rend = FakeRenderer()
rend.last_frame_timings = {'frame_ms': 9.0, 'swap_or_present_ms': 3.5,
                           'overlay_ms': 4.25}
rend._last_prediction_render_stats = {'prepare_ms': 1.25, 'draw_ms': 0.75}
ctx = DevContext(predictor=pred, renderer=rend, tick_rate=180.0)

N = 20000
for _ in range(2000):                   # aufwaermen (numpy, attribut-cache)
    ctx.sample_timings(11.0)
t0 = time.perf_counter()
for _ in range(N):
    ctx.sample_timings(11.0)
per_call_us = (time.perf_counter() - t0) / N * 1e6
budget_us = 1000.0 / 180.0 * 1000.0     # ein frame bei 180 fps
share = per_call_us / budget_us * 100.0
print(f"       {per_call_us:.2f} us je frame = {share:.4f} % eines "
      f"{budget_us / 1000.0:.2f} ms frames")
# Grosszuegig: die messung laeuft auf einem beliebigen rechner unter last.
# Alles unter 20 us ist im rauschen; die referenzmessung liegt bei ~2 us.
check_true(per_call_us < 20.0, 'abtasten kostet unter 20 us',
           f"{per_call_us:.2f} us")
# Und es darf pro frame NICHTS anwachsen (kein append, kein dict-neubau).
check(ctx.timings.count, ctx.timings.capacity, 'der puffer waechst nicht')

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for failure in FAILURES:
        print(f"  {failure}")
    sys.exit(1)
print("devui timing: alle pruefungen bestanden")
