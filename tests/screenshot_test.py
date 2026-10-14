"""Regressionstest fuer die einstellungen von runtime/screenshot.py.

Nur die rechnung: zuschnitt, aufloesungsstufen, ausgabegroesse. Das
offscreen-neuzeichnen selbst braucht einen GL-kontext und ist ueber
runtime.loop.run geprueft (.claude/rules/camera-input.md, F12-notiz).
Aufruf: python tests/screenshot_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.screenshot import CROP_MIN_FRACTION, ScreenshotWriter

FAILURES = []


def check(ok, label):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    if not ok:
        FAILURES.append(label)


shot = ScreenshotWriter('unused')

print("1. vorgabe: fensteraufloesung, voller zuschnitt")
check(shot.resolution_label == 'window', "aufloesung 'window'")
check(shot.crop_is_full, "zuschnitt voll")
check(shot.output_size(1280, 800) == (1280, 800), "ausgabe = fenster")
check(not shot.needs_rerender(800), "kein neuzeichnen")

print("2. stufen sind feste HOEHEN, die breite folgt dem fenster")
for label, height in (('2K', 1440), ('4K', 2160), ('6K', 3240), ('8K', 4320)):
    shot.set_resolution(label)
    w, h = shot.output_size(1280, 800)
    check(h == height and w == round(1280 * height / 800),
          f"{label}: {w}x{h}")
    check(shot.needs_rerender(800), f"{label}: neuzeichnen")
shot.set_resolution('2K')
check(not shot.needs_rerender(1440),
      "2K bei einem 1440 px hohen fenster liest einfach das fenster")
try:
    shot.set_resolution('5K')
    check(False, "unbekannte stufe wirft")
except ValueError:
    check(True, "unbekannte stufe wirft")

print("3. zuschnitt: nur die breite, geklemmt")
shot.set_resolution('4K')
shot.set_crop(0.25, 0.75)
check(shot.output_size(1280, 800) == (1728, 2160),
      f"halbe breite in 4K: {shot.output_size(1280, 800)}")
check(shot.crop_pixels(1280) == (320, 640), "x0/breite im fenster")
shot.set_crop(-0.5, 2.0)
check(shot.crop_left == 0.0 and shot.crop_right == 1.0, "auf 0..1 geklemmt")
shot.set_crop(0.6, 0.4)
check(abs((shot.crop_right - shot.crop_left) - CROP_MIN_FRACTION) < 1e-12,
      "vertauschte raender -> mindestbreite")

print("4. mittig schnappen und symmetrisch ziehen")
shot.set_crop(0.1, 0.5)
shot.centre_crop()
check(abs(shot.crop_left - 0.3) < 1e-12 and abs(shot.crop_right - 0.7) < 1e-12,
      f"mitte bei gleicher breite: {shot.crop_left:.3f}..{shot.crop_right:.3f}")
shot.crop_symmetric = True
shot.set_crop_edge('left', 0.2)
check(abs(shot.crop_right - 0.8) < 1e-12, "links 0.2 -> rechts 0.8")
shot.set_crop_edge('right', 0.9)
check(abs(shot.crop_left - 0.1) < 1e-12, "rechts 0.9 -> links 0.1")
shot.set_crop_edge('left', 0.7)
check(shot.crop_left < 0.5 < shot.crop_right
      and abs(shot.crop_left + shot.crop_right - 1.0) < 1e-12,
      "symmetrisch ueber die mitte gezogen bleibt um die mitte")
shot.reset_crop()
check(shot.crop_is_full, "full width")

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)}")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("ALL PASSED")
