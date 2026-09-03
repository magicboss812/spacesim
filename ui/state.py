"""Beobachtbarer ansichts-zustand, den HUD und tastatur gemeinsam bedienen.

Warum es dieses modul gibt: bezugsrahmen, referenzkoerper und das
ziel-overlay lagen bisher als LOKALE VARIABLEN in test.py::main(). Kein
objekt kam an sie heran -- ein HUD-bedienelement haette sie nicht lesen und
erst recht nicht setzen koennen, ohne die hauptschleife umzubauen.

Hier liegen sie an einer stelle, mit einer aenderungs-benachrichtigung.
Tastatur (R / 1 / 2 / T) und HUD-bedienelemente schreiben denselben zustand
und koennen deshalb nicht auseinanderlaufen.

INVARIANTE: hier steht ANSICHTS-zustand, keine physik. Der bezugsrahmen ist
eine reine darstellungs-transformation; gespeicherter koerper-zustand bleibt
absolut (barycentrisch).
"""

from physics.reference_frames import (
    BODY_CENTRED_BODY_DIRECTION,
    BODY_CENTRED_NON_ROTATING,
)


class UIState:
    """Bezugsrahmen-auswahl, referenzkoerper und overlay-schalter."""

    def __init__(self, bodies, initial_reference_index=None, on_change=None):
        self.bodies = bodies
        self.on_change = on_change

        # Nur himmelskoerper sind als bezugsrahmen sinnvoll -- ein rahmen,
        # der auf dem schiff sitzt, macht die eigene bahn zum punkt.
        self.celestial_indices = [
            i for i, b in enumerate(bodies) if not getattr(b, 'is_ship', False)
        ] or list(range(len(bodies)))

        self.ship_index = next(
            (i for i, b in enumerate(bodies) if getattr(b, 'is_ship', False)), None
        )

        if initial_reference_index is None or initial_reference_index >= len(bodies):
            initial_reference_index = self.celestial_indices[0]
        self.reference_index = int(initial_reference_index)
        self.reference_cursor = (
            self.celestial_indices.index(self.reference_index)
            if self.reference_index in self.celestial_indices else 0
        )

        self.frame_extension = BODY_CENTRED_NON_ROTATING
        self.target_overlay_enabled = False

        # Angeklickter koerper (index in `bodies`) oder None.
        #
        # ABSICHTLICH ETWAS ANDERES ALS `reference_index`. Der bezugskoerper
        # bestimmt, gegen was gerechnet und gezeichnet wird -- das ist eine
        # entscheidung des spielers (taste R, die koerperliste im HUD) und
        # darf nicht als nebenwirkung eines mausklicks umspringen. Die
        # auswahl sagt nur "diesen koerper meine ich gerade".
        self.selected_index = None

    # ------------------------------------------------------------- ableitung

    @property
    def reference_body(self):
        if self.reference_index is None:
            return None
        return self.bodies[self.reference_index]

    @property
    def reference_name(self):
        body = self.reference_body
        return getattr(body, 'name', '--') if body is not None else '--'

    @property
    def selected_body(self):
        if self.selected_index is None:
            return None
        if not (0 <= self.selected_index < len(self.bodies)):
            return None
        return self.bodies[self.selected_index]

    @property
    def frame_mode_label(self):
        if self.frame_extension == BODY_CENTRED_BODY_DIRECTION:
            return 'body-direction'
        return 'non-rotating'

    def secondary_index(self):
        """Zweiter koerper fuer den body-direction-rahmen.

        Bevorzugt den mutterkoerper des referenzkoerpers (Mond -> Erde): die
        richtung mond-erde ist die, die man beim mond sehen will. Sonst der
        erstbeste andere himmelskoerper.
        """
        primary = self.bodies[self.reference_index]
        parent = getattr(primary, 'is_moon_of', None)
        if parent is not None:
            for idx, candidate in enumerate(self.bodies):
                if candidate is parent and not getattr(candidate, 'is_ship', False):
                    return idx
        for idx in self.celestial_indices:
            if idx != self.reference_index:
                return idx
        return self.reference_index

    # ------------------------------------------------------------- mutation

    def _changed(self):
        if self.on_change is not None:
            self.on_change(self)

    def cycle_reference(self, step=1):
        if not self.celestial_indices:
            return
        self.reference_cursor = (
            (self.reference_cursor + int(step)) % len(self.celestial_indices)
        )
        self.reference_index = self.celestial_indices[self.reference_cursor]
        self._changed()

    def set_reference_index(self, index):
        if index is None or index == self.reference_index:
            return
        self.reference_index = int(index)
        if self.reference_index in self.celestial_indices:
            self.reference_cursor = self.celestial_indices.index(self.reference_index)
        self._changed()

    def select_body(self, index):
        """Setzt die auswahl. Gibt True zurueck, wenn sie sich geaendert hat.

        LOEST BEWUSST KEIN on_change AUS. Die benachrichtigung baut in
        test.py den plotting-frame neu auf und macht die gehaltene vorhersage
        ungueltig (`predictor.invalidate_hold`) -- ein voller
        neuaufbau der trajektorie, pro mausklick, fuer eine reine
        darstellungs-markierung. Der renderer liest `selected_index`
        stattdessen je frame direkt.
        """
        if index is not None:
            index = int(index)
            if not (0 <= index < len(self.bodies)):
                index = None
        if index == self.selected_index:
            return False
        self.selected_index = index
        return True

    def clear_selection(self):
        return self.select_body(None)

    def set_frame_extension(self, extension):
        if extension == self.frame_extension:
            return
        self.frame_extension = extension
        self._changed()

    def toggle_target_overlay(self):
        if self.ship_index is None:
            return False
        self.target_overlay_enabled = not self.target_overlay_enabled
        self._changed()
        return True

    def set_target_overlay(self, enabled):
        """Overlay direkt setzen (HUD-knopf), statt zu kippen (taste T)."""
        enabled = bool(enabled) and self.ship_index is not None
        if enabled == self.target_overlay_enabled:
            return False
        self.target_overlay_enabled = enabled
        self._changed()
        return True

    def apply_view_mode(self, mode):
        """Die drei modi der HUD-rahmenwahl in einem schritt.

        'surface'  -> mitrotierender rahmen: die bahn erscheint relativ zur
                      drehenden umgebung des bezugskoerpers
        'orbital'  -> koerperzentriert, nicht rotierend (der normalfall)
        'target'   -> zusaetzlich das ziel-overlay

        In EINEM schritt, weil jede teil-aenderung on_change ausloest und
        damit einen frame-neuaufbau -- zwei aufrufe wuerden den bezugsrahmen
        kurzzeitig in einen zwischenzustand versetzen.
        """
        from physics.reference_frames import (
            BODY_CENTRED_BODY_DIRECTION,
            BODY_CENTRED_NON_ROTATING,
        )
        extension = (BODY_CENTRED_BODY_DIRECTION if mode == 'surface'
                     else BODY_CENTRED_NON_ROTATING)
        overlay = (mode == 'target') and self.ship_index is not None
        if extension == self.frame_extension and overlay == self.target_overlay_enabled:
            return
        self.frame_extension = extension
        self.target_overlay_enabled = overlay
        self._changed()

    def view_mode(self):
        """Aktueller modus als index fuer die HUD-rahmenwahl."""
        from physics.reference_frames import BODY_CENTRED_BODY_DIRECTION
        if self.target_overlay_enabled:
            return 2
        return 0 if self.frame_extension == BODY_CENTRED_BODY_DIRECTION else 1

    def refresh(self):
        """Erzwingt ein neuanwenden ohne aenderung (start, system-neuladen)."""
        self._changed()
