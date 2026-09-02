"""Kopftest fuer ui/widgets/rate_slider.py -- die federphysik und die
ratensteuerung. Laeuft ohne fenster, ohne GL: der widget-code fasst in
update()/on_*() nur ctx.px, ctx.theme und die callables an, alles davon
laesst sich stellen.

Aufruf: python tests/ui_horizon_slider_test.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import ui.core as _core
from ui.core import Rect, UIRoot
from ui.widgets.rate_slider import DEADZONE, HorizonSlider, _response

FAILURES = []


def check(condition, label, detail=''):
    if condition:
        print(f"  ok   {label}{(' -- ' + detail) if detail else ''}")
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label} -- {detail}")


class _Motion:
    fast = 22.0
    normal = 14.0
    slow = 8.0


class _Theme:
    motion = _Motion()
    control_height = 24.0


class _Text:
    def measure(self, s, role):
        return (8.0 * len(str(s)), 12.0)


class _Ctx:
    """Das Nötigste, das der Widget-Code berührt."""
    def __init__(self):
        self.theme = _Theme()
        self.text = _Text()
        self.dt = 0.0

    def px(self, v):
        if isinstance(v, (tuple, list)):
            return tuple(float(x) for x in v)
        return float(v)

    @property
    def screen_rect(self):
        return Rect(0.0, 0.0, 1280.0, 800.0)


class _Predictor:
    def __init__(self, num_points=10000):
        self.num_points = num_points


def make(value_box, minimum=0.25, maximum=4.0, sweep=2.5, predictor=None):
    """value_box ist eine 1-elementige liste, damit on_change zurueckschreibt."""
    def getter():
        return value_box[0]

    def setter(m):
        value_box[0] = m

    w = HorizonSlider(getter, minimum, maximum, setter,
                      predictor=predictor, sweep_seconds=sweep)
    w.rect = Rect(0.0, 0.0, 168.0, 60.0)
    w.enabled = True
    return w


def drive(widget, ctx, seconds, dt=1.0 / 60.0):
    steps = int(round(seconds / dt))
    for _ in range(steps):
        widget.update(ctx, dt)


ctx = _Ctx()

# --- 1. Totzone: kleine auslenkung aendert nichts -----------------------
print("1. Totzone")
box = [1.0]
w = make(box)
w._offset = DEADZONE * 0.5
w.pressed = True
drive(w, ctx, 2.0)
check(box[0] == 1.0, "innerhalb der totzone bleibt der wert stehen",
      f"wert {box[0]}")
w._offset = 0.5
w.pressed = True
drive(w, ctx, 0.2)
check(box[0] > 1.0, "ausserhalb der totzone steigt der wert", f"wert {box[0]:.3f}")

# --- 2. Raten-proportionalitaet ---------------------------------------
print("2. Raten-proportionalitaet")
box = [0.25]
w = make(box)
w._offset = 1.0
w.pressed = True
drive(w, ctx, 2.5)            # sweep_seconds bei voller auslenkung
check(abs(box[0] - 4.0) <= 4.0 * 0.05,
      "volle auslenkung fuer sweep_seconds -> min..max", f"wert {box[0]:.3f}")

box = [0.25]
w = make(box)
w._offset = 0.5
w.pressed = True
drive(w, ctx, 2.5)
check(box[0] < 0.25 + (4.0 - 0.25) * 0.6,
      "halbe auslenkung ist deutlich langsamer", f"wert {box[0]:.3f}")

# --- 3. Rueckweg ----------------------------------------------------------
print("3. Rueckweg zur mitte")
box = [4.0]
w = make(box)
w._offset = -1.0
w.pressed = True
drive(w, ctx, 2.5)
check(abs(box[0] - 0.25) <= 0.25 * 0.10 + 0.02,
      "voll nach links -> zurueck auf min", f"wert {box[0]:.4f}")

# --- 4. Feder laeuft in die mitte, aenderung stoppt --------------------
print("4. Federweg")
box = [1.0]
w = make(box)
w._offset = 0.8
w.pressed = False
drive(w, ctx, 0.4)
check(abs(w._offset) < DEADZONE, "offset zerfaellt in die totzone",
      f"offset {w._offset:.4f}")
frozen = box[0]
drive(w, ctx, 2.0)
check(box[0] == frozen, "nach dem federweg keine weitere aenderung",
      f"{box[0]} == {frozen}")

# gedrueckt gehalten: offset bleibt stehen
w2 = make([1.0])
w2._offset = 0.7
w2.pressed = True
drive(w2, ctx, 0.5)
check(abs(w2._offset - 0.7) < 1e-9, "gedrueckt haelt der offset",
      f"offset {w2._offset}")

# --- 5. Klemmung --------------------------------------------------------
print("5. Klemmung auf [min, max]")
import random
random.seed(1)
box = [1.0]
w = make(box)
w.pressed = True
ok = True
for _ in range(600):
    w._offset = random.uniform(-1.0, 1.0)
    w.update(ctx, 1.0 / 60.0)
    if not (0.25 - 1e-9 <= box[0] <= 4.0 + 1e-9):
        ok = False
        break
check(ok, "der wert verlaesst [0.25, 4.0] nie", f"wert {box[0]:.4f}")

# --- 6. is_grabbing ---------------------------------------------------
print("6. is_grabbing")
w = make([1.0])
w.pressed = False
w._offset = 1.0
check(not w.is_grabbing, "nicht gedrueckt -> kein grabbing", "")
w.pressed = True
w._offset = 0.0
check(not w.is_grabbing, "gedrueckt in der mitte -> kein grabbing", "")
w._offset = 0.5
check(w.is_grabbing, "gedrueckt und ausgelenkt -> grabbing", "")

# --- 7. Mausrad -------------------------------------------------------
print("7. Mausrad-raste")
box = [1.0]
w = make(box)
w.on_wheel(ctx, 0, 1)
check(abs(box[0] - 2.0) < 1e-9, "eine raste hoch = x wheel_step", f"wert {box[0]}")
w.on_wheel(ctx, 0, -1)
check(abs(box[0] - 1.0) < 1e-9, "eine raste runter = / wheel_step", f"wert {box[0]}")
for _ in range(10):
    w.on_wheel(ctx, 0, 1)
check(box[0] == 4.0, "das rad klemmt auch", f"wert {box[0]}")

# --- 8. _response-kurve --------------------------------------------------
print("8. _response")
check(_response(0.0) == 0.0, "_response(0) == 0", "")
check(_response(DEADZONE * 0.9) == 0.0, "_response in der totzone == 0", "")
check(_response(1.0) == 1.0, "_response(1) == 1", f"{_response(1.0)}")
check(_response(-1.0) == -1.0, "_response(-1) == -1", f"{_response(-1.0)}")
check(0.0 < _response(0.5) < 0.5, "_response(0.5) unter der geraden",
      f"{_response(0.5):.3f}")

# --- 9. abgeschaltet bei predictor.num_points == 0 --------------------
print("9. Abgeschaltet ohne vorhersage")
box = [1.0]
pred = _Predictor(num_points=0)
w = make(box, predictor=pred)
w._offset = 1.0
w.pressed = True
w.update(ctx, 1.0 / 60.0)
check(w.enabled is False, "num_points == 0 -> enabled False", "")
check(box[0] == 1.0, "abgeschaltet aendert der regler nichts", f"wert {box[0]}")
w.on_mouse_down(ctx, 80.0, 30.0, 1)
check(box[0] == 1.0, "abgeschaltet ignoriert er klicks", f"wert {box[0]}")
pred.num_points = 10000
w.update(ctx, 1.0 / 60.0)
check(w.enabled is True, "num_points zurueck -> enabled True", "")

# --- 10. _horizon_metres liest aus dem predictor zurueck --------------
print("10. _horizon_metres")


class _PredLen:
    def __init__(self, length):
        self.num_points = 10000
        self._length = length

    def get_display_length(self):
        return self._length


w = make([1.0], predictor=_PredLen(2.4e10))
check(w._horizon_metres() == 2.4e10, "gibt die predictor-laenge zurueck",
      f"{w._horizon_metres()}")
w = make([1.0], predictor=_PredLen(None))
check(w._horizon_metres() is None, "None bleibt None", "")
w = make([1.0], predictor=_PredLen(float('inf')))
check(w._horizon_metres() is None, "inf wird zu None", "")

# --- 11. UIRoot-ausfallsicherung: verlorenes MOUSEBUTTONUP -------------
# Geht das up-ereignis verloren (fokuswechsel mitten im ziehen), bleibt
# `pressed` sonst fuer immer stehen und HorizonSlider.update() integriert
# ungebremst weiter. begin_frame() muss den griff loesen, sobald keine
# maustaste mehr ansteht.
print("11. UIRoot loest verlorenen griff")


class _Dummy:
    def __init__(self):
        self.pressed = True
        self.visible = True
        self.hovered = False
        self.children = []
        self.z = 0
        self.blocks_mouse = False


_orig_get_pressed = _core.pygame.mouse.get_pressed
try:
    _core.pygame.mouse.get_pressed = lambda *a, **k: (0, 0, 0)
    root = UIRoot(ctx)
    dummy = _Dummy()
    root._active_widget = dummy
    root.begin_frame(1.0 / 60.0)
    check(root._active_widget is None,
          "keine maustaste an -> _active_widget geloest", "")
    check(dummy.pressed is False, "und pressed zurueckgesetzt",
          f"pressed {dummy.pressed}")

    # Gegenprobe: solange eine taste ansteht, bleibt der griff.
    _core.pygame.mouse.get_pressed = lambda *a, **k: (1, 0, 0)
    root2 = UIRoot(ctx)
    dummy2 = _Dummy()
    root2._active_widget = dummy2
    root2.begin_frame(1.0 / 60.0)
    check(root2._active_widget is dummy2 and dummy2.pressed is True,
          "taste steht an -> griff bleibt", "")
finally:
    _core.pygame.mouse.get_pressed = _orig_get_pressed

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for f in FAILURES:
        print(f"  {f}")
    sys.exit(1)
print("ui/widgets/rate_slider.py: alle pruefungen bestanden")
