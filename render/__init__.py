"""Der zeichen-pfad: OpenGL-pipelines, primitive, der Renderer.

    gl/             die GLSL-quellen (.vert/.frag)
    renderer.py     die Renderer-klasse selbst
    pipelines.py    shader uebersetzen, VBOs, GL-zustandscache
    draw.py         linien-, ortho- und textur-primitive
    text.py         schriften und der label-textur-cache
    bodies.py       koerper, marken, beschriftungen, auswahlmarke
    ship.py         schiffs-sprite, pfeil, fahne, orientierungsvektoren
    orbits.py       bahnlinien und referenzspuren
    prediction.py   die vorhersagelinie: abtastung, Hermite, Ap/Pe-marker
    background.py   sternenfeld und dekaden-gitter

ABSICHTLICH IMPORT-ARM: dieses `__init__` zieht KEIN moderngl herein, damit
`from render import GL_DIR` auch dort billig ist, wo gar nicht gezeichnet wird
(`config/loader.py`, `ui/draw.py`, `ui/text.py`).
"""
import os

# Der EINE ort, an dem der pfad zu den GLSL-quellen steht. Vorher rechneten
# ihn drei module unabhaengig voneinander aus (rendering.py, ui/draw.py,
# ui/text.py) -- beim verschieben der shader war das drei mal dieselbe
# aenderung, und eine davon wurde erfahrungsgemaess vergessen.
GL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gl')

__all__ = ['GL_DIR']
