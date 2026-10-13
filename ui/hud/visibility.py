"""Was vom HUD und von den welt-overlays GEZEICHNET wird -- fuer saubere abzuege.

Die dev-oberflaeche (F1 -> "HUD visibility") schaltet hier einzelne elemente
aus, etwa fuer die abbildungen der seminararbeit. AUSBLENDEN HEISST NUR NICHT
ZEICHNEN: jedes element rechnet weiter, die telemetrie laeuft, und jede
schirmposition, die es veroeffentlicht (`renderer.apsis_marker_hits`,
`renderer.maneuver_node_hits`), wird weiter jeden frame gefuellt. Deshalb
`Widget.drawn` und nicht `Widget.visible` -- letzteres ueberspringt auch
update() (ui/core.py). Ein nicht gezeichnetes widget nimmt keine maus an:
`UIRoot.paint_order` laesst es weg, und `_pick` liest dieselbe liste.
"""

#: (schluessel, beschriftung im panel, attribute am Hud). Ein schluessel fuer
#: mehrere widgets, wo die responsive umschaltung zwei fassungen desselben
#: elements haelt (breit / schmal).
HUD_ELEMENTS = (
    ('badge', 'ship badge', ('badge',)),
    ('body_browser', 'body browser', ('body_browser',)),
    ('target', 'target panel', ('target', 'target_rail')),
    ('warp', 'time-warp bar', ('warp',)),
    ('system_map', 'system map', ('system_map',)),
    ('navball', 'navball cluster + maneuver tiles',
     ('navball', 'maneuver_burn', 'maneuver_axes', 'maneuver_nodes',
      'maneuver_length')),
    ('apsis_block', 'AP/PE block', ()),
    ('snaps', 'snap rosette', ('snaps', 'snaps_compact')),
    ('left_stack', 'frame / zoom / predict', ('left_stack',)),
    ('apsis_tooltip', 'apsis tooltip', ('apsis_tooltip',)),
)

#: (schluessel, beschriftung, attribut am renderer). Der renderer liest die
#: flags beim zeichnen; seine trefferlisten fuellt er unabhaengig davon.
WORLD_OVERLAYS = (
    ('apsis_markers', 'apsis markers + labels', 'draw_apsis_markers'),
    ('body_labels', 'body name labels', 'draw_body_labels'),
    ('maneuver_markers', 'maneuver node markers + handles',
     'draw_maneuver_markers'),
)


class HudVisibility:
    """Die schalter und ihre anwendung auf HUD und renderer.

    Haelt den zustand selbst (ein dict schluessel -> bool) und schreibt ihn
    in `apply()` auf die widgets und den renderer. Die dev-oberflaeche ruft
    `set()` / `set_all()`; beide wenden sofort an.
    """

    def __init__(self, hud=None, renderer=None):
        self.hud = hud
        self.renderer = renderer
        self.shown = {key: True for key, _label, _attrs in HUD_ELEMENTS}
        self.shown.update({key: True for key, _label, _attr in WORLD_OVERLAYS})
        self.apply()

    def keys(self):
        return list(self.shown)

    def is_shown(self, key):
        return bool(self.shown.get(key, True))

    def set(self, key, shown):
        if key not in self.shown:
            raise KeyError(key)
        self.shown[key] = bool(shown)
        self.apply()

    def set_all(self, shown):
        for key in self.shown:
            self.shown[key] = bool(shown)
        self.apply()

    @property
    def all_hidden(self):
        return not any(self.shown.values())

    def apply(self):
        hud = self.hud
        if hud is not None:
            for key, _label, attrs in HUD_ELEMENTS:
                for attr in attrs:
                    widget = getattr(hud, attr, None)
                    if widget is not None:
                        widget.drawn = self.is_shown(key)
            navball = getattr(hud, 'navball', None)
            if navball is not None:
                navball.show_info = self.is_shown('apsis_block')
            tooltip = getattr(hud, 'apsis_tooltip', None)
            if tooltip is not None:
                # Ein zettel an einer raute, die nicht gezeichnet ist, haenge
                # sichtbar im leeren.
                tooltip.drawn = (self.is_shown('apsis_tooltip')
                                 and self.is_shown('apsis_markers'))
            gizmo = getattr(hud, 'gizmo', None)
            if gizmo is not None:
                # Der ziehgriff zeichnet nichts selbst, er IST die eingabe zu
                # den markern des renderers. Sind die aus, faengt er auch
                # keine klicks mehr.
                gizmo.drawn = self.is_shown('maneuver_markers')
        renderer = self.renderer
        if renderer is not None:
            for key, _label, attr in WORLD_OVERLAYS:
                setattr(renderer, attr, self.is_shown(key))
