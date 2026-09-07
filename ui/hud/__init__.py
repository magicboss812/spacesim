"""Die konkreten HUD-elemente.

Die formsprache ist die instrumententafel aus Kerbal Space Program 2
(referenzbilder in "screenshots for debugging/ksp2 original.png" und
"krb09rzxst661.jpg"). Uebernommen wurden davon fuenf dinge, und zwar genau
die fuenf, die den eindruck tragen:

    chrome.py        die bauteile: FASE statt rundung, doppelter rahmen,
                     notch-tab auf der kante, teilung, zellenbogen
    navball.py       der navball-block -- kurs, geschwindigkeit, hoehe,
                     schub, steigrate und AP/PE in EINEM instrument
    attitude.py      der kompassring darin (2D, keine navball-kugel)
    controls.py      zeitraffer als pegel, snap-rosette, zoom
    panels.py        schiffs-plakette und ziel-block
    body_browser.py  ausklappliste der koerper -> wahl des bezugskoerpers
    system_map.py    die system-karte rechts unter dem zeitraffer
    apsis_tooltip.py der schwebezettel an einer Ap/Pe-raute
    telemetry.py     datenschicht -- rechnet alle anzeigewerte einmal pro frame
    maneuver.py      die vier manoever-plaettchen IM navball-raster --
                     ueber ORB die brenndaten, unter THR die delta-v-
                     eingabefelder, ueber ALT die vorschau-reichweite,
                     unter V/S die knotenwahl; dazu der ziehgriff an
                     der bahnlinie
    layout.py        verankerung des ganzen und die responsive umschaltung

Einstieg ist Hud(...) aus layout.py: es baut den widget-baum in eine
bestehende UIRoot und wird pro frame mit update() aktuell gehalten.

WAS DIE VORLAGE NICHT VORGIBT und hier entschieden wurde:

- **Kein horizont, keine kugel.** Die simulation kennt GENAU EINEN
  orientierungswinkel (schiff.theta) -- nick und roll gibt es in dieser
  physik nicht. Eine schattierte kugel wuerde also achsen anzeigen, die
  nicht existieren; in einer Seminararbeit waeren das erfundene daten.
- **Kein tank, kein RCS, keine SAS-modi, keine dritte raumachse.** Nichts
  davon hat eine entsprechung in der simulation. Die snap-rosette hat vier
  richtungen statt sechs, weil es in 2D vier gibt.
- **Zeitraffer-stufen als sim-zeit je echtsekunde.** Ein vielfaches
  ("10000x") sagt bei bahnmechanik nichts; "1h/s" beantwortet die frage
  sofort. ACHT stufen von 1m/s bis 1y/s, liste in WARP_STEPS (layout.py).
  Gezeichnet als PEGEL -- alle stufen bis zur aktuellen sind gefuellt --,
  weil die vorlage ihre raffung ebenso als reihe von winkeln zeigt.
- **Die rechte flanke zeigt die RADIALGESCHWINDIGKEIT.** In der vorlage
  steht dort die senkrechtgeschwindigkeit gegen den boden. Ohne
  oberflaechennormale ist v . r_dach die ehrliche entsprechung, und sie ist
  an Ap und Pe genau null.
- **Ziel = bezugskoerper.** Die simulation kennt keine eigene zielauswahl.
  Der bezugskoerper ist der koerper, auf den sich ohnehin alle bahnwerte
  beziehen, und damit die ehrliche entsprechung.
- **Schub = schubstufe -- ausser beim manoever-autopiloten.** VON HAND gibt es
  nur impulse je frame; der bogen skaliert schiffcontrol.thrust_acc und ist
  damit wirksam, keine attrappe. Ein ausgefuehrter manoeverknoten ist der eine
  DAUERSCHUB in diesem spiel: `ship/maneuver/` faehrt ein rampenprofil ab und
  schreibt seine hebelstellung je frame in set_thrust_level, der bogen zeigt
  also die rampe. Im zeitraffer bleibt der schub gesperrt und der bogen
  beschriftet sich mit HOLD -- ohne das drueckt der spieler 'Up', nichts
  passiert, und nichts auf dem schirm sagt warum. Ein scharfer knoten zieht
  die raffung vor der zuendung selbst auf echtzeit herunter.
- **Symbole gezeichnet, nicht gesetzt.** ◉ und ⊗ (U+25C9 / U+2297) fehlen in
  vielen oberflaechen-schriften; als vektoren stimmen sie immer.
- **Eine festgelegte palette.** Vier farben mit je EINER bedeutung, kein
  wechselknopf: eine farbe, die sich neu verteilen laesst, kann nichts
  bedeuten. Die zuordnung liegt in theme.ROLE_INDEX.
"""

from .apsis_tooltip import ApsisTooltip
from .attitude import AttitudeRing
from .layout import WARP_STEPS, Hud
from .navball import NavballCluster
from .system_map import SystemMap
from .telemetry import Telemetry, compass_from_theta

__all__ = ['ApsisTooltip', 'AttitudeRing', 'Hud', 'NavballCluster', 'SystemMap',
           'Telemetry', 'WARP_STEPS', 'compass_from_theta']
