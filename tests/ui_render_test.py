"""Regressionstest der UI-zeichenschicht -- gegen echte PIXEL.

Die lehre aus Phase 0-2 (siehe plans/ui_overhaul_plan.md, abschnitt 4.8):
eine messung ist nur etwas wert, wenn sie durch GENAU DIE funktionen laeuft,
die auch der zeichenpfad benutzt. Deshalb rendert dieser test echte frames
ueber UIRoot.render() und liest den framebuffer zurueck, statt geometrie
nachzurechnen.

Geprueft wird:
  1. Fuellung, rahmen und eckradius des SDF-shaders
  2. Farbtoenung von text (u_color in texquad.frag)
  3. Textschaerfe -- anteil voll deckender pixel
  4. Verankerung ueber einen resize hinweg
  5. Trefferflaeche == gezeichnete flaeche
  6. Maus-vorfahrt (wants_mouse) und klick-weiterleitung
  7. Regler und schalter unter der maus
  8. ui_scale skaliert layout und schrift
  9. Fase statt rundung -- das vorzeichen im eckradius
 10. Die pixelschrift bleibt hart gerastert und auf der fuenfer-leiter

Aufruf: python tests/ui_render_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SDL_WINDOWS_DPI_AWARENESS', 'permonitorv2')

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import moderngl
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE

from ui import TOP_LEFT, TOP_RIGHT, UIContext, UIRoot
from ui.theme import DEFAULT_THEME, cut_corners, rgba
from ui.widgets import Button, Label, Panel, Slider, Toggle

W, H = 800, 500
FAILURES = []


def check(condition, label, detail=''):
    if condition:
        print(f"  ok   {label}{(' -- ' + detail) if detail else ''}")
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label} -- {detail}")


# Nur display+font -- pygame.init() zaehlt mixer- und joystick-geraete auf
# und kostet dabei ~45 s. Siehe runtime/window.py.
pygame.display.init()
pygame.font.init()
pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL | RESIZABLE, vsync=0)
gl = moderngl.create_context()
gl.enable(moderngl.BLEND)
gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)


def read_pixels(width=W, height=H):
    """Framebuffer als (h, w, 3) uint8 in TOP-DOWN reihenfolge.

    viewport= wird explizit uebergeben: ctx.screen kennt seine groesse nur
    vom zeitpunkt der context-erstellung (siehe CLAUDE.md, stale-scissor).
    """
    data = gl.screen.read(viewport=(0, 0, width, height), components=3, dtype='f1')
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)[::-1]


def draw_frame(root, width=W, height=H, dt=1.0):
    gl.screen.use()
    gl.clear(0.0, 0.0, 0.0, 1.0)
    root.begin_frame(dt)
    root.render()
    return read_pixels(width, height)


# --------------------------------------------------------------------- aufbau

ui = UIContext(gl, W, H, ui_scale=1.0, theme=DEFAULT_THEME)
root = UIRoot(ui)

FILL_COLOR = rgba('#204060', 1.0)
BORDER_COLOR = rgba('#ff0000', 1.0)
RADIUS = 16.0

panel = Panel(
    anchor=TOP_LEFT, offset=(40, 40), size=(300, 200),
    fill=FILL_COLOR, border=BORDER_COLOR, radius=RADIUS, shadow=False,
    padding=16,
)
root.add(panel)

clicks = []
button = Button(
    text='ABORT', anchor=TOP_LEFT, offset=(0, 0), size=(120, 30),
    on_click=lambda w: clicks.append(w),
)
panel.add(button)

corner_panel = Panel(
    anchor=TOP_RIGHT, offset=(24, 24), size=(160, 80),
    fill=rgba('#00ff00', 1.0), border=None, radius=0, shadow=False,
)
root.add(corner_panel)

print("1. SDF-rechteck: fuellung, rahmen, eckradius")
frame = draw_frame(root)

px = frame[40 + 100, 40 + 150]
check(tuple(px) == (32, 64, 96), 'fuellung trifft die angeforderte farbe',
      f"gemessen {tuple(px)} erwartet (32, 64, 96)")

edge = frame[40 + 100, 40]
check(edge[0] > 200 and edge[1] < 60, 'rahmen liegt auf der linken kante',
      f"gemessen {tuple(edge)}")

# Die aeussere ecke MUSS hintergrund sein -- das ist der beweis, dass der
# radius wirklich rundet und nicht nur ein rechteck gezeichnet wird.
outer = frame[41, 41]
check(int(outer.sum()) < 30, 'eckpunkt (1,1) ist ausgerundet',
      f"gemessen {tuple(outer)}")
# Auf der diagonale INNERHALB des radius muss dagegen flaeche liegen.
inner = frame[40 + 30, 40 + 30]
check(int(inner.sum()) > 60, 'punkt innerhalb des radius ist gefuellt',
      f"gemessen {tuple(inner)}")

print()
print("2. Verankerung: rechts oben verankertes panel klebt an seiner ecke")
right_col = W - 24 - 80
check(tuple(frame[24 + 40, right_col]) == (0, 255, 0),
      'panel sitzt vor dem resize an der rechten oberen ecke',
      f"gemessen {tuple(frame[24 + 40, right_col])}")

NEW_W, NEW_H = 1100, 620
pygame.display.set_mode((NEW_W, NEW_H), DOUBLEBUF | OPENGL | RESIZABLE, vsync=0)
gl.screen.viewport = (0, 0, NEW_W, NEW_H)
gl.screen.scissor = (0, 0, NEW_W, NEW_H)
root.resize(NEW_W, NEW_H, ui_scale=1.0)
frame2 = draw_frame(root, NEW_W, NEW_H)

right_col2 = NEW_W - 24 - 80
check(tuple(frame2[24 + 40, right_col2]) == (0, 255, 0),
      'panel klebt nach dem resize weiter an der rechten oberen ecke',
      f"gemessen {tuple(frame2[24 + 40, right_col2])}")
check(tuple(frame2[40 + 100, 40 + 150]) == (32, 64, 96),
      'links verankertes panel bleibt unveraendert stehen',
      f"gemessen {tuple(frame2[40 + 100, 40 + 150])}")

root.resize(W, H, ui_scale=1.0)
pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL | RESIZABLE, vsync=0)
gl.screen.viewport = (0, 0, W, H)
gl.screen.scissor = (0, 0, W, H)

print()
print("3. Text: toenung ueber u_color")
corner_panel.visible = False
panel.visible = False
tint_root = UIRoot(ui)
tint_label = Label(text='PROGRADE', anchor=TOP_LEFT, offset=(100, 100),
                   role='heading', color=rgba('#ff8000', 1.0))
tint_root.add(tint_label)
frame3 = draw_frame(tint_root)

band = frame3[95:130, 95:260].reshape(-1, 3).astype(np.int32)
lit = band[band.sum(axis=1) > 40]
check(lit.shape[0] > 0, 'getoenter text erzeugt pixel', f"{lit.shape[0]} pixel")
if lit.shape[0]:
    brightest = lit[lit[:, 0].argmax()]
    check(brightest[0] > 200 and 100 < brightest[1] < 160 and brightest[2] < 40,
          'text traegt die angeforderte farbe (#ff8000)',
          f"hellstes pixel {tuple(brightest)}")

print()
print("4. Textschaerfe: anteil voll deckender pixel")
white_root = UIRoot(ui)
# REINWEISS erzwingen. Die theme-textfarbe ist #e6edf5 -- ein test, der
# gegen 255 prueft, koennte dort NIE bestehen und wuerde eine unschaerfe
# melden, die es nicht gibt. Genau diese art scheinbar sauberer, aber
# bedeutungsloser messung ist in abschnitt 4.8 des plans dokumentiert.
white_root.add(Label(text='ALTITUDE 384.40Mm', anchor=TOP_LEFT,
                     offset=(100, 100), role='mono_readout',
                     color=(1.0, 1.0, 1.0, 1.0)))
frame4 = draw_frame(white_root)
gray = frame4[:, :, 0].astype(np.float32)
w_px, h_px = ui.text.measure('ALTITUDE 384.40Mm', 'mono_readout')
box = gray[100 - 2:100 + int(h_px) + 2, 100 - 2:100 + int(w_px) + 2]
text_lit = box[box > 8]
crisp = float((text_lit > 250).sum()) / max(text_lit.size, 1) * 100.0
# Direkt auf den bildschirm gezeichnet und pixelgerastet lag der wert in
# Phase 2 bei ~28 %; durch FXAA fiel er auf 5.3 %. Alles ueber 15 % zeigt,
# dass weder unschaerfe-ursache aktiv ist.
check(crisp > 15.0, 'text ist pixelscharf',
      f"{crisp:.1f} % voll deckend von {text_lit.size} pixeln")

print()
print("5. Trefferflaeche == gezeichnete flaeche")
panel.visible = True
corner_panel.visible = True
draw_frame(root)

btn = button.rect
check(abs(btn.x - (40 + 16)) < 0.51 and abs(btn.y - (40 + 16)) < 0.51,
      'button sitzt im inhaltsbereich des panels (padding beruecksichtigt)',
      f"rect {btn}")
check(button.hit_test(ui, btn.center_x, btn.center_y),
      'mitte des buttons trifft')
check(not button.hit_test(ui, btn.right + 4, btn.center_y),
      'knapp rechts daneben trifft nicht')

print()
print("6. Eingabe-vorfahrt und klick-weiterleitung")
root._mouse_pos = (5.0, 5.0)
root.begin_frame(0.016)
check(not root.wants_mouse, 'leerer bereich beansprucht die maus nicht')

root._mouse_pos = (btn.center_x, btn.center_y)
root.begin_frame(0.016)
check(root.wants_mouse, 'ueber dem button beansprucht die UI die maus')
check(root.hovered_widget is button, 'der button ist das getroffene widget',
      f"getroffen: {getattr(root.hovered_widget, 'name', None)}")

down = pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                          {'pos': (int(btn.center_x), int(btn.center_y)), 'button': 1})
up = pygame.event.Event(pygame.MOUSEBUTTONUP,
                        {'pos': (int(btn.center_x), int(btn.center_y)), 'button': 1})
check(root.handle_event(down), 'mousedown wird von der UI verbraucht')
check(root.handle_event(up), 'mouseup wird von der UI verbraucht')
check(len(clicks) == 1, 'on_click genau einmal ausgeloest', f"{len(clicks)} aufrufe")

# Loslassen ausserhalb darf NICHT ausloesen -- der zurueckgezogene klick.
clicks.clear()
root.handle_event(down)
away = pygame.event.Event(pygame.MOUSEBUTTONUP,
                          {'pos': (int(btn.right + 60), int(btn.center_y)), 'button': 1})
root.handle_event(away)
check(len(clicks) == 0, 'ausserhalb losgelassener klick loest nicht aus',
      f"{len(clicks)} aufrufe")

print()
print("7. Regler und schalter reagieren auf ziehen")
slider_root = UIRoot(ui)
values = []
slider = Slider(value=0.0, minimum=0.0, maximum=100.0, anchor=TOP_LEFT,
                offset=(50, 300), size=(200, 30),
                on_change=lambda v: values.append(v))
slider_root.add(slider)
toggle_state = {'on': False}
toggle = Toggle(text='FXAA', value=lambda: toggle_state['on'],
                on_change=lambda v: toggle_state.update(on=v),
                anchor=TOP_LEFT, offset=(50, 350), size=(120, 30))
slider_root.add(toggle)
draw_frame(slider_root)

track_x = slider.rect.x + slider.rect.w * 0.5
slider_root._mouse_pos = (track_x, slider.rect.center_y)
slider_root.begin_frame(0.016)
slider_root.handle_event(pygame.event.Event(
    pygame.MOUSEBUTTONDOWN, {'pos': (int(track_x), int(slider.rect.center_y)), 'button': 1}))
check(len(values) == 1 and 45.0 < values[-1] < 55.0,
      'klick auf die mitte der schiene ergibt ~50',
      f"werte {values}")

toggle_root_pos = (toggle.rect.center_x, toggle.rect.center_y)
slider_root._mouse_pos = toggle_root_pos
slider_root.begin_frame(0.016)
slider_root.handle_event(pygame.event.Event(
    pygame.MOUSEBUTTONDOWN, {'pos': (int(toggle_root_pos[0]), int(toggle_root_pos[1])), 'button': 1}))
slider_root.handle_event(pygame.event.Event(
    pygame.MOUSEBUTTONUP, {'pos': (int(toggle_root_pos[0]), int(toggle_root_pos[1])), 'button': 1}))
check(toggle_state['on'] is True, 'schalter kippt auf True',
      f"zustand {toggle_state}")

print()
print("8. ui_scale skaliert layout und schrift")
small_w, small_h = ui.text.measure('ALTITUDE', 'mono_readout')
root.resize(W, H, ui_scale=2.0)
ui.text.resize(W, H, ui_scale=2.0)
big_w, big_h = ui.text.measure('ALTITUDE', 'mono_readout')
check(big_h > small_h * 1.6, 'schrift wird bei ui_scale=2 neu gerastert',
      f"{small_h:.0f}px -> {big_h:.0f}px")
draw_frame(root)
check(abs(panel.rect.w - 600.0) < 1.5, 'panelbreite verdoppelt sich',
      f"{panel.rect.w:.1f}px")
root.resize(W, H, ui_scale=1.0)

print()
print("9. Fase statt rundung (negativer eckradius)")

# Die FORMSPRACHE der oberflaeche haengt an einem vorzeichen: ein negativer
# eckwert ist eine 45-grad-fase statt einer rundung (shaders/ui_rect.frag,
# chamfer_box). Geprueft wird an dem punkt, an dem sich beide am staerksten
# unterscheiden.
#
# Fase der groesse c: die schnittkante laeuft von (c, 0) nach (0, c), alles
# mit x + y < c liegt DRAUSSEN. Rundung mit radius c: der bogenmittelpunkt
# liegt bei (c, c), und (0.35c, 0.35c) hat davon den abstand
# 0.65c*sqrt(2) = 0.92c < c, liegt also DRINNEN. Derselbe punkt entscheidet
# damit zwischen beiden formen. (0.3c waere zu knapp: 0.99c faellt genau in
# die geglaettete kante und misst einen halbwert.)
CUT = 24
FILL_RGB = (255, 200, 0)


def corner_probe(radius):
    """Zeichnet ein rechteck bei (100, 100) und tastet seine linke obere ecke ab."""
    root2 = UIRoot(UIContext(gl, W, H, ui_scale=1.0))
    panel2 = root2.add(Panel(anchor=TOP_LEFT, offset=(100, 100), size=(200, 160),
                             radius=radius, fill=rgba('#ffc800'), border=None,
                             shadow=False))
    panel2.blocks_mouse = False
    pixels = draw_frame(root2)
    # (0.3c, 0.3c) und (0.8c, 0.8c) relativ zur ecke bei (100, 100)
    near = pixels[100 + int(CUT * 0.35), 100 + int(CUT * 0.35)]
    far = pixels[100 + int(CUT * 0.8), 100 + int(CUT * 0.8)]
    middle = pixels[180, 180]
    return near, far, middle


def lit(pixel):
    return int(pixel[0]) > 100


near_cut, far_cut, mid_cut = corner_probe(-CUT)
near_round, far_round, mid_round = corner_probe(CUT)

check(lit(mid_cut) and lit(mid_round), 'beide formen sind in der mitte gefuellt',
      f"fase {tuple(int(v) for v in mid_cut)}, rundung {tuple(int(v) for v in mid_round)}")
check(not lit(near_cut), 'FASE: der punkt bei 0.35c liegt ausserhalb',
      f"gemessen {tuple(int(v) for v in near_cut)}")
check(lit(near_round), 'RUNDUNG: derselbe punkt liegt innerhalb',
      f"gemessen {tuple(int(v) for v in near_round)} -- das ist der unterschied, "
      f"den das vorzeichen macht")
check(lit(far_cut), 'FASE: der punkt bei 0.8c liegt innerhalb',
      f"gemessen {tuple(int(v) for v in far_cut)}")

# ASYMMETRIE: theme.cut_corners() laesst einzelne ecken scharf. Genau das --
# nicht die fase an sich -- laesst einen block konstruiert statt dekoriert
# wirken, und es ist der grund, warum die vier eckwerte einzeln gesetzt werden.
root3 = UIRoot(UIContext(gl, W, H, ui_scale=1.0))
panel3 = root3.add(Panel(anchor=TOP_LEFT, offset=(100, 100), size=(200, 160),
                         radius=cut_corners(CUT, top_left=False),
                         fill=rgba('#ffc800'), border=None, shadow=False))
panel3.blocks_mouse = False
px3 = draw_frame(root3)
tl = px3[102, 102]                     # oben links: soll SCHARF sein
tr = px3[102, 100 + 200 - 3]           # oben rechts: soll gefast sein
check(lit(tl), 'cut_corners: die ausgenommene ecke bleibt scharf',
      f"oben links {tuple(int(v) for v in tl)}")
check(not lit(tr), 'cut_corners: die uebrigen ecken sind gefast',
      f"oben rechts {tuple(int(v) for v in tr)}")


print()
print("10. Die pixelschrift bleibt pixelig, nicht unscharf")

# SB Liquid ist auf einem pixelraster gezeichnet. Zwei dinge muessen stimmen,
# damit sie so aussieht, wie sie gedacht ist -- und beide sind unsichtbar,
# wenn man nur hinschaut, aber messbar:
#
# 1. OHNE kantenglaettung gerastert. Mit glaettung traegt jede glyphe einen
#    halbdeckenden saum; das ist der unterschied zwischen "pixelig" und
#    "unscharf pixelig", den diese oberflaeche gerade nicht haben soll.
# 2. Auf ein VIELFACHES VON FUENF gerundet. Dazwischen faellt die stegbreite
#    unterschiedlich aus (gemessen bei 11-13 px: stege von 1 UND 2 px
#    nebeneinander), was die schrift trotz harter kanten unruhig macht.
text_ctx = UIContext(gl, W, H, ui_scale=1.0)
scale = text_ctx.theme.type_scale

check(scale.section.family == 'display' and scale.key.family == 'text',
      'die beiden familien sind getrennt vergeben',
      f"section={scale.section.family}, key={scale.key.family}")

for role in ('tab', 'section', 'value', 'gauge'):
    check(not text_ctx.text._role_antialias(role),
          f"'{role}' wird ohne kantenglaettung gerastert")
for role in ('key', 'title', 'body'):
    check(text_ctx.text._role_antialias(role),
          f"'{role}' wird mit kantenglaettung gerastert")

for ui_scale in (1.0, 1.25, 1.44, 2.0):
    probe = UIContext(gl, W, H, ui_scale=ui_scale)
    sizes = [probe.text._role_pixel_size(r)
             for r in ('tab', 'section', 'value', 'gauge')]
    check(all(size % 5 == 0 and size >= 10 for size in sizes),
          f"ui_scale {ui_scale}: pixelgroessen liegen auf der fuenfer-leiter",
          f"{sizes}")

# Der eigentliche beweis: die gerasterte flaeche darf nur ZWEI alphawerte
# tragen. Alles dazwischen ist saum.
probe = UIContext(gl, W, H, ui_scale=1.0)
font = probe.text.font('value')
surface = probe.text._render_tracked(
    'ORBITAL.INFO 0123', font, probe.text.tracking_px('value'), False
)
alpha = pygame.surfarray.pixels_alpha(surface).astype(int)
partial = float(((alpha > 8) & (alpha < 247)).sum()) / alpha.size * 100.0
check(partial == 0.0, 'kein einziger teildeckender pixel in der pixelschrift',
      f"{partial:.2f} % teildeckend, {len(np.unique(alpha))} alphawerte")

# Gegenprobe an der leseschrift: die MUSS einen saum haben, sonst waere sie
# ebenfalls hart gerastert und die unterscheidung liefe ins leere.
font = probe.text.font('title')
surface = probe.text._render_tracked(
    'Erde', font, probe.text.tracking_px('title'), True
)
alpha = pygame.surfarray.pixels_alpha(surface).astype(int)
soft = float(((alpha > 8) & (alpha < 247)).sum()) / alpha.size * 100.0
check(soft > 3.0, 'die leseschrift wird dagegen geglaettet',
      f"{soft:.2f} % teildeckend")

print()
if FAILURES:
    print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
    for failure in FAILURES:
        print(f"  {failure}")
    pygame.quit()
    sys.exit(1)
print("ui-render: alle pruefungen bestanden")
pygame.quit()
