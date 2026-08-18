"""Die konkreten HUD-elemente.

Umgesetzt nach dem entwurf "Spacesim 2D gameplay GUI mockup"
(claude.ai/design, projekt 8790e76e, datei "Spacesim HUD.dc.html").

    telemetry.py     datenschicht -- rechnet alle anzeigewerte einmal pro frame
    attitude.py      der lagemesser (2D-ring, keine navball-kugel)
    panels.py        bahnelemente, ziel, schiffs-plakette
    body_browser.py  ausklappliste der koerper -> wahl des bezugskoerpers
    controls.py      zeitraffer, rahmenwahl, autopilot, schub, zoom, palette
    layout.py        verankerung des ganzen und die responsive umschaltung

Einstieg ist Hud(...) aus layout.py: es baut den widget-baum in eine
bestehende UIRoot und wird pro frame mit update() aktuell gehalten.

WAS DER ENTWURF NICHT VORGIBT und hier entschieden wurde:

- **Zeitraffer-stufen.** Der entwurf zeigt 1x/5x/50x/1k/10k. Fuer bahn-
  mechanik ist das zu fein -- bei 50x dauert ein erdumlauf noch anderthalb
  stunden. Hier stehen stattdessen ACHT stufen, beschriftet als sim-zeit je
  echtsekunde: 1m/s, 10m/s, 1h/s, 1d/s, 7d/s, 30d/s, 100d/s, 1y/s.
  Die genaue liste steht in WARP_STEPS (layout.py), nicht hier.
- **Ziel = bezugskoerper.** Die simulation kennt keine eigene zielauswahl.
  Der bezugskoerper ist der koerper, auf den sich ohnehin alle bahnwerte
  beziehen, und damit die ehrliche entsprechung.
- **Schubregler = schubstufe.** Es gibt keinen dauerschub, nur impulse pro
  frame. Der regler skaliert schiffcontrol.thrust_acc und ist damit wirksam,
  keine attrappe.
- **Symbole gezeichnet, nicht gesetzt.** ◉ und ⊗ (U+25C9 / U+2297) fehlen in
  vielen oberflaechen-schriften; als vektoren stimmen sie immer.
- **Keine palette-verwuerfelung.** Der entwurf verteilt die vier farben per
  zufallsseed neu ("SHUFFLE DISTRIBUTION"). Das ist ein erkundungswerkzeug --
  im spiel muss dieselbe farbe dauerhaft dasselbe bedeuten, deshalb liegt
  die rollenzuordnung in theme.ROLE_INDEX fest.
"""

from .attitude import AttitudeRing
from .layout import WARP_STEPS, Hud
from .telemetry import Telemetry, compass_from_theta

__all__ = ['AttitudeRing', 'Hud', 'Telemetry', 'WARP_STEPS', 'compass_from_theta']
