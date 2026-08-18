"""spacesim/ui -- das SPIELER-HUD.

Abgrenzung zu devui.py: dort liegen die entwicklerwerkzeuge (Dear ImGui,
F1), die bewusst wie ein werkzeugkasten aussehen duerfen. Hier liegt die
gestaltete oberflaeche, die der spieler sieht. Die beiden schichten teilen
sich weder darstellung noch zustand, nur die eingabe-vorfahrt:

    custom-UI (diese schicht)  ->  ImGui  ->  welt (kamera / schiff)

Aufbau:

    units.py    reine zahlenformatierung, ohne GL testbar
    theme.py    palette, typo-stufen, abstaende, radien
    text.py     schriften, label-textur-cache, getoentes blitten
    draw.py     zeichen-primitive auf EINEM SDF-shader
    core.py     rechtecke, verankerung, widget-basis, eingabe-routing
    state.py    beobachtbarer ansichts-zustand (bezugsrahmen, overlays)
    widgets/    schaltflaechen, schalter, regler, aufklappmenues, panels
    hud/        die konkreten HUD-elemente

ALLE groessen in widget-code sind DESIGN-EINHEITEN. Die umrechnung auf echte
pixel passiert ueber UIContext.px(), abgeleitet aus der fensterhoehe -- das
ist, was aufloesungsunabhaengigkeit erreicht.
"""

from . import units
from .core import (
    BOTTOM_CENTER,
    BOTTOM_LEFT,
    BOTTOM_RIGHT,
    CENTER,
    CENTER_LEFT,
    CENTER_RIGHT,
    FILL,
    Rect,
    TOP_CENTER,
    TOP_LEFT,
    TOP_RIGHT,
    UIContext,
    UIRoot,
    Widget,
    ease,
)
from .state import UIState
from .theme import DEFAULT_THEME, Theme, mix, rgba, with_alpha

__all__ = [
    'BOTTOM_CENTER', 'BOTTOM_LEFT', 'BOTTOM_RIGHT', 'CENTER', 'CENTER_LEFT',
    'CENTER_RIGHT', 'DEFAULT_THEME', 'FILL', 'Rect', 'TOP_CENTER', 'TOP_LEFT',
    'TOP_RIGHT', 'Theme', 'UIContext', 'UIRoot', 'UIState', 'Widget', 'ease',
    'mix', 'rgba', 'units', 'with_alpha',
]
