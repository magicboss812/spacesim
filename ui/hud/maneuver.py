"""Das manoever-werkzeug -- VIER plaettchen IM navball-raster.

WARUM DORT UND NIRGENDWO SONST. Der navball-block hat vier freie felder,
und sie sind bereits gerastert: ueber der ORB-flanke, unter dem
THR-streifen, ueber der ALT-flanke, unter dem V/S-streifen. Alles andere
daneben zu stellen -- erst ein 200x206-klotz, dann drei plaettchen mit
eigenem dock-abstand -- laesst den block als navball MIT ANHAENGSEL lesen
und laesst gleichzeitig diese vier felder leer. Die plaettchen sind
deshalb keine nachbarn des blocks mehr, sie sind teil seines rasters:

    +--------+                        +--------+
    |  BURN  |                        |  PLAN  |   <- ueber den flanken
    +--------+   /--------------\\      +--------+
    |  ORB   |  |   kompassring  |     |  ALT   |
    +--------+   \\--------------/      +--------+
    | THR    |                         |   V/S  |
    +--------+                        +--------+
    | PRO    |                        | o o o  |   <- unter den streifen
    | NRM    |                        | +   -  |
    +--------+                        +--------+

DIE MASSE KOMMEN AUS DEM NAVBALL, NICHT VON HIER. `layout()` liest
`NavballCluster.flank_rect()` / `strip_rect()`; breite und x-lage sind
damit dieselben wie bei ORB/ALT/THR/V-S, in jeder UI-skala. Eigene zahlen
waeren ein zweites layout, und schon eine geaenderte `BOX_H` liesse die
plaettchen danebenstehen.

    OBEN wie eine FLANKE: doppelrahmen, beschriftung im kasten (die
    flanken tragen ihr 'ORB' auch innen, keinen notch-tab).
    UNTEN wie ein STREIFEN: flache, versenkte platte.

Nach aussen zeigende ecken gefast, die zur kugel zeigenden SCHARF -- genau
die regel, nach der schon die flanken angesetzt statt danebengestellt
aussehen.

DIE BEIDEN DELTA-V-ZEILEN SIND EINGABEFELDER, keine schrittknoepfe. Vier
pfeilknoepfe je achse frassen 60 der 124 einheiten breite, und wer 1900 m/s
einstellen will, klickt 190-mal. Angeklickt nimmt die zeile die tastatur
und man tippt die zahl hinein -- ZIFFERN UND PUNKT, sonst nichts, und das
feld waechst an ort und stelle statt in einem dialog woanders. Es startet
MIT DEM STEHENDEN WERT, aber als ganzes markiert: die erste ziffer ERSETZT
ihn (bloss vorbelegt hiesse anhaengen -- aus '0.0' und getippten '250.5'
wurde '0.02505'), ein RUECKSCHRITT steigt in ihn ein und aendert ihn
weiter. Leer zu starten war die andere haelfte desselben fehlers: die zeile
zeigte '0', waehrend der knoten noch 250 trug. GESCHRIEBEN WIRD ERST BEIM
ABSCHLUSS -- Enter, Tab oder ein klick woanders --, nicht bei jedem
anschlag: jeder anschlag ist sonst ein `plan.touch()`, und wer '25000'
tippt, laesst die kette auch die ZWISCHENSTUFEN 2, 25, 250 und 2500
rechnen. Die teuerste davon ist die letzte vor der gewollten, und ein
vertipptes '250000' haengt die eingabe an einem brennbogen fest, den
niemand sehen wollte. Die
RICHTUNG ist kein vorzeichen zum tippen, sondern die beschriftung selbst:
`PRO` schaltet auf `RET`, `NRM` auf `ANM`. Deshalb ist der kleinste
tippbare wert 0.0 -- ein minus gaebe es zweimal, einmal als zeichen und
einmal als knopf.

Das GESAMT-Delta-v ist NICHT tippbar: es ist die wurzel aus beiden achsen,
also ein ergebnis. Es steht darum im BURN-kasten, bei brenndauer und
countdown, wo alles andere auch abgeleitet ist.

DER EXECUTE-KNOPF IST GESPERRT, SOLANGE DER KNOTEN KEIN DELTA-V TRAEGT. Ein
frischer knoten ist ein PLATZHALTER: man setzt ihn an eine stelle der bahn
und entscheidet erst danach, was dort geschehen soll. Ein knopf, der in
diesem zustand etwas ausloeste, waere eine luege -- und
`ManeuverExecutor.arm()` lehnt ohnehin ab, die sperre ist also die SICHTBARE
haelfte einer regel, die zweimal gilt.

DER COUNTDOWN WIRD ROT, SOBALD ER AUF 'T+' SPRINGT. Ein knoten in der
vergangenheit ist nicht mehr anfliegbar -- die kette rechnet ihn noch, aber
das schiff ist vorbei. In amber, der farbe jedes anderen brennwertes,
unterscheidet sich dieser zustand nur durch ein vorzeichen zwei zeichen
weit links. `palette.danger` ist dafuer da und ist KEINE fuenfte
bedeutungsfarbe: die vier akzente tragen bedeutung, danger traegt einen
zustand (wie `disabled`, das direkt daneben steht).

Farben: prograde gruen, normal magenta -- dieselben, die die rosette den
beiden achsen gibt (`controls.SnapRosette.MODES`) und die griffe an der
linie tragen (`render/maneuver.py`). Delta-v, brenndauer und countdown in
amber, weil ein brennvorgang energie ist. Nichts davon ist eine neue farbe.

KOORDINATEN top-down, MASSE in design-einheiten.
"""

import math

import numpy as np
import pygame

from .. import units
from ..core import Rect, Widget, ease
from ..theme import mix, with_alpha
from . import chrome
from .controls import _button_colors

# Alle masse in DESIGN-EINHEITEN (siehe .claude/rules/hud-ui.md). Breite und
# x-lage stehen hier NICHT -- die kommen aus dem navball.

#: Abstand zur flanke bzw. zum streifen. Derselbe wert, den der navball
#: zwischen flanke und streifen benutzt (`navball.FLANK_GAP`).
DOCK_GAP = 6.0

#: Die beiden oberen kaesten. BURN traegt drei werte und einen knopf, PLAN
#: einen ablesewert und eine spur -- deshalb verschieden hoch. Gemeinsam
#: ist ihre UNTERkante: sie sitzen auf ihrer flanke.
BURN_HEIGHT = 76.0
PLAN_HEIGHT = 48.0

#: Die beiden unteren platten. HOEHER GEHT NICHT: der ORBITAL.INFO-block ist
#: nur um eine halbe flankenbreite eingerueckt, ragt also unter beide
#: flanken, und beginnt 47 einheiten unter dem streifen -- was tiefer
#: haengt, liegt darauf. `layout()` klemmt zusaetzlich gegen seine oberkante,
#: damit eine geaenderte BOX_H oder INFO_H das nicht stillschweigend kippt.
AXES_HEIGHT = 40.0
NODES_HEIGHT = 40.0

_PAD = 8.0
_LINE_H = 15.0
_EXEC_H = 18.0
_PIP_H = 11.0
_PIP_GAP = 3.0
_FOOT_H = 15.0

#: Die beiden achsen: (schluessel, feldname, positive/negative beschriftung).
_AXES = (
    ('pro', 'prograde', 'PRO', 'RET'),
    ('nrm', 'normal', 'NRM', 'ANM'),
)

#: Was ein eingabefeld annimmt. Ziffern und EIN punkt -- ein minus waere die
#: zweite art, die richtung zu sagen (siehe modulkopf).
_DIGITS = '0123456789'

#: Blinkdauer des carets in sekunden (60 % davon hell). Ein stehender
#: strich in einem HUD voller ruhiger anzeigen liest sich als teil der
#: zeichnung; das blinken ist, was ihn als schreibstelle ausweist.
_CARET_BLINK_S = 1.06


def _edit_text(value):
    """Ein delta-v als das, was man beim OEFFNEN vorfindet.

    Wie die anzeige auf eine nachkommastelle, aber OHNE die tote '.0':
    vorbelegt ist der text der weiterzutippende, und eine stelle, die man
    erst wieder wegloeschen muss, um eine ganze zahl zu aendern, waere im
    weg. Das vorzeichen bleibt draussen -- es ist der richtungsknopf.
    """
    text = f"{abs(float(value or 0.0)):.1f}"
    return text[:-2] if text.endswith('.0') else text


class _ManeuverPlate(Widget):
    """Ein plaettchen im navball-raster.

    `SIDE` waehlt die flanke, `PLACE` das feld darueber oder darunter, und
    `layout()` holt sich beides aus dem navball -- siehe modulkopf.
    """

    #: 'left' (ORB/THR) oder 'right' (ALT/V-S).
    SIDE = 'left'
    #: 'above' = ueber der flanke, 'below' = unter dem streifen.
    PLACE = 'above'
    PLATE_HEIGHT = 48.0

    def __init__(self, telemetry, navball=None, **kwargs):
        kwargs.setdefault('size', (124.0, None))
        super().__init__(**kwargs)
        self.telemetry = telemetry
        self.navball = navball
        # Das plaettchen faengt die maus ab: es sitzt ueber der bahn, und ein
        # klick, der zwischen zwei knoepfen durchfaellt, wuerde die kamera
        # darunter schwenken.
        self.blocks_mouse = True
        self._hover_key = None

    # ------------------------------------------------------------ geometrie

    def measure(self, ctx):
        navball = self.navball
        width = 124.0
        if navball is not None and navball.rect.w > 0.0:
            width = navball.flank_rect(ctx, self.SIDE)[2]
        else:
            width = ctx.px(width)
        return (width, ctx.px(self.PLATE_HEIGHT))

    def layout(self, ctx, parent_rect):
        navball = self.navball
        if navball is None or navball.rect.w <= 0.0:
            return super().layout(ctx, parent_rect)
        height = ctx.px(self.PLATE_HEIGHT)
        gap = ctx.px(DOCK_GAP)
        if self.PLACE == 'above':
            fx, fy, fw, _fh = navball.flank_rect(ctx, self.SIDE)
            self.rect = Rect(fx, fy - gap - height, fw, height)
        else:
            sx, sy, sw, sh = navball.strip_rect(ctx, self.SIDE)
            top = sy + sh + gap
            # GEKLEMMT gegen die apsiden-leiste: sie ragt unter beide
            # flanken, und ein plaettchen, das auf ihr liegt, ist der
            # fehler, den die aufteilung gerade beseitigen soll.
            limit = navball.info_rect(ctx)[1] - ctx.px(1.0)
            self.rect = Rect(sx, top, sw, max(ctx.px(12.0),
                                              min(height, limit - top)))
        self.layout_children(ctx)
        return self.rect

    def _corners(self, ctx):
        """Die zur kugel zeigende seite bleibt SCHARF -- wie bei den flanken."""
        if self.SIDE == 'left':
            return (True, False, False, True)
        return (False, True, True, False)

    # -------------------------------------------------------------- treffer

    def _regions(self, ctx):
        raise NotImplementedError

    def _hit(self, ctx, x, y):
        for key, (rx, ry, rw, rh) in self._regions(ctx).items():
            if key.startswith('_'):
                continue
            if rx <= x < rx + rw and ry <= y < ry + rh:
                return key
        return None

    def on_mouse_move(self, ctx, x, y):
        self._hover_key = self._hit(ctx, x, y)
        return True

    def _active(self):
        return getattr(self.telemetry, 'maneuver_state', 'idle') in (
            'armed', 'burning')

    # -------------------------------------------------------------- zeichnen

    def _draw_row(self, ctx, rect, key, value, color, key_color=None):
        """Eine zeile: beschriftung links, wert rechts.

        DREI STUFEN, genau wie `NavballCluster._strip`, und aus demselben
        anlass: der inhalt haengt an der groessenordnung des werts. Bei 124
        einheiten breite belegt 'T-1d 07:29:43' die ganze zeile und schoebe
        sich sonst ueber sein eigenes 'NODE'. Der wert gewinnt immer, die
        beschriftung ist die zugabe.
        """
        palette = ctx.theme.palette
        x, y, w, h = rect
        middle = y + h * 0.5
        gap = ctx.px(5.0)
        caption_w = ctx.text.measure(key, 'caption')[0]

        role = None
        for candidate in ('value', 'throttle_value', 'caption'):
            if caption_w + gap + ctx.text.measure(value, candidate)[0] <= w:
                role = candidate
                break
        if role is not None:
            ctx.text.draw(key, x, middle, role='caption',
                          color=key_color or palette.text_dim,
                          align='left', valign='middle')
        else:
            role = 'caption'
        ctx.text.draw(value, x + w, middle, role=role, color=color,
                      align='right', valign='middle')


class ManeuverBurnBlock(_ManeuverPlate):
    """DV, brenndauer, countdown, EXECUTE -- ueber der ORB-flanke."""

    SIDE = 'left'
    PLACE = 'above'
    PLATE_HEIGHT = BURN_HEIGHT

    def _regions(self, ctx):
        gap = ctx.px(ctx.theme.frame_gap)
        pad = ctx.px(_PAD)
        x = self.rect.x + gap + pad
        w = self.rect.w - (gap + pad) * 2.0
        y = self.rect.y + gap + ctx.px(4.0)

        regions = {}
        line_h = ctx.px(_LINE_H)
        for key in ('_dv', '_burn', '_countdown'):
            regions[key] = (x, y, w, line_h)
            y += line_h
        y += ctx.px(2.0)
        regions['execute'] = (x, y, w, ctx.px(_EXEC_H))
        return regions

    def execute_enabled(self):
        """Darf EXECUTE gedrueckt werden?

        Nur mit einem knoten, der wirklich delta-v traegt. Waehrend eines
        laufenden manoevers ist der knopf ebenfalls aktiv -- er heisst dann
        ABORT.
        """
        executor = getattr(self.telemetry, 'maneuver_executor', None)
        if executor is not None and executor.is_active:
            return True
        return bool(getattr(self.telemetry, 'maneuver_can_arm', False))

    def on_mouse_up(self, ctx, x, y, button):
        if button != 1:
            return True
        if self._hit(ctx, x, y) == 'execute' and self.execute_enabled():
            self.telemetry.toggle_execute()
        return True

    def draw(self, ctx):
        palette = ctx.theme.palette
        telemetry = self.telemetry
        regions = self._regions(ctx)
        active = self._active()

        chrome.frame(ctx, self.rect.x, self.rect.y, self.rect.w, self.rect.h,
                     cut=14.0, corners=self._corners(ctx),
                     glow_role='node' if active else None)

        self._draw_row(ctx, regions['_dv'], 'DV',
                       units.delta_v(telemetry.maneuver_dv_total),
                       palette.node, key_color=with_alpha(palette.node, 0.9))
        self._draw_row(ctx, regions['_burn'], 'BURN',
                       units.duration(telemetry.maneuver_burn_seconds),
                       palette.node)
        # Solange nichts scharf ist, zaehlt der countdown auf den KNOTEN;
        # scharf zaehlt er auf die ZUENDUNG -- das ist dann der zeitpunkt,
        # auf den es ankommt, eine halbe brenndauer frueher.
        countdown = (telemetry.maneuver_time_to_ignition if active
                     else telemetry.maneuver_time_to_node)
        # ROT, SOBALD ES 'T+' HEISST (siehe modulkopf).
        past = countdown is not None and float(countdown) < 0.0
        # KOMPAKT gesetzt: die volle form ist 125 px breit und passt in
        # die 102 px innenbreite nur ohne ihre beschriftung -- also
        # ausgerechnet ohne das, was NODE von IGN unterscheidet.
        self._draw_row(ctx, regions['_countdown'],
                       'IGN' if active else 'NODE',
                       units.countdown_compact(countdown),
                       palette.danger if past else palette.node)

        x, y, w, h = regions['execute']
        enabled = self.execute_enabled()
        hover = 1.0 if self._hover_key == 'execute' else 0.0
        if not enabled:
            # DIE SICHTBARE HAELFTE DER SPERRE (siehe modulkopf).
            fill, text_color, border = (
                palette.disabled, palette.text_dimmer, palette.edge_inner)
        else:
            fill, text_color, border = _button_colors(
                ctx, active, palette.snap, hover)
        chrome.plate(ctx, x, y, w, h, fill=fill, line=border, cut=5.0)
        ctx.text.draw('ABORT' if active else 'EXECUTE',
                      x + w * 0.5, y + h * 0.5, role='button_sm',
                      color=text_color, align='center', valign='middle')


class ManeuverAxesBlock(_ManeuverPlate):
    """Die beiden delta-v-achsen als EINGABEFELDER -- unter dem THR-streifen.

    Die beschriftung ist der richtungsknopf (PRO/RET, NRM/ANM), der wert
    rechts ist tippbar. Warum so und nicht mit vorzeichen: siehe modulkopf.
    """

    SIDE = 'left'
    PLACE = 'below'
    PLATE_HEIGHT = AXES_HEIGHT

    def __init__(self, telemetry, navball=None, **kwargs):
        super().__init__(telemetry, navball=navball, **kwargs)
        #: Welche achse gerade getippt wird, und was bisher dasteht.
        self._editing = None
        self._buffer = ''
        #: Die SCHREIBSTELLE im puffer, 0..len. Pfeile, pos1/ende und ein
        #: klick ins offene feld setzen sie; eingefuegt wird genau hier.
        self._caret = 0
        #: Ist der ganze wert MARKIERT (der zustand beim oeffnen)? Dann
        #: ersetzt die naechste ziffer ihn, und ein pfeil hebt die markierung
        #: auf, statt sie wegzuwerfen. Gezeichnet wird sie als band.
        self._select_all = False
        #: Blinkphase des carets, in sekunden. Jeder anschlag setzt sie auf
        #: 0 zurueck -- ein caret, der beim tippen gerade dunkel ist, sieht
        #: aus wie ein haenger.
        self._caret_phase = 0.0
        #: Ob am offenen feld ueberhaupt eine taste war. Trennt 'nichts
        #: gesagt' (klick, dann weg -- schreibt nicht) von 'auf null
        #: geloescht' (getippt und wieder weggeloescht -- schreibt 0.0).
        self._touched = False
        #: Die zuletzt gewaehlte richtung je achse. Die WAHRHEIT ist das
        #: vorzeichen des werts; dies hier ist nur das gedaechtnis fuer den
        #: fall 0.0, wo es keines gibt.
        self._sign = {'prograde': 1.0, 'normal': 1.0}

    # -- Die tastatur nur DANN nehmen, wenn sie auch gebraucht wird -------
    #
    # `UIRoot` fragt `takes_keyboard` im moment des klicks ab und setzt
    # danach den fokus; ein festes True hiesse, dass ein klick irgendwohin
    # auf dieses plaettchen die tasten N / X / WASD verschluckt, bis man
    # woanders hinklickt. Der hover-schluessel steht zu diesem zeitpunkt
    # bereits (die maus bewegt sich vor jedem klick).

    @property
    def takes_keyboard(self):
        return (self._editing is not None
                or self._hover_key in ('pro_value', 'nrm_value'))

    @takes_keyboard.setter
    def takes_keyboard(self, value):
        # Widget.__init__ schreibt hier False hinein. Der wert wird
        # abgeleitet, also verfaellt die zuweisung.
        pass

    # ------------------------------------------------------------ geometrie

    def _regions(self, ctx):
        pad = ctx.px(_PAD)
        x = self.rect.x + pad
        w = self.rect.w - pad * 2.0
        y = self.rect.y + ctx.px(4.0)

        regions = {}
        row_h = ctx.px(_LINE_H) + ctx.px(2.0)
        label_w = ctx.text.measure('PRO', 'caption')[0] + ctx.px(6.0)
        for prefix, _field, _pos, _neg in _AXES:
            regions[f'{prefix}_sign'] = (x, y, label_w, row_h)
            regions[f'{prefix}_value'] = (
                x + label_w, y, w - label_w, row_h)
            y += row_h
        return regions

    # -------------------------------------------------------------- zustand

    def _node(self):
        return getattr(self.telemetry, 'maneuver_node', None)

    def _value_of(self, field):
        node = self._node()
        if node is None:
            return None
        return float(node.dv_prograde if field == 'prograde' else node.dv_normal)

    def _write(self, field, magnitude):
        """Betrag schreiben, richtung aus `_sign`. Zaehlt plan.version hoch."""
        node = self._node()
        plan = getattr(self.telemetry, 'maneuver_plan', None)
        if node is None or plan is None:
            return False
        value = max(0.0, float(magnitude)) * self._sign[field]
        if field == 'prograde':
            node.dv_prograde = value
        else:
            node.dv_normal = value
        # OHNE touch() bliebe die gezeichnete linie auf dem alten stand --
        # die vorschau rechnet nur bei geaenderter version neu.
        plan.touch()
        return True

    def _sync_signs(self):
        """Das gedaechtnis dem echten vorzeichen nachfuehren."""
        for _prefix, field, _pos, _neg in _AXES:
            value = self._value_of(field)
            if value is None or value == 0.0:
                continue
            self._sign[field] = -1.0 if value < 0.0 else 1.0

    def _commit(self):
        """Das feld schliessen UND DABEI SCHREIBEN -- der einzige schreibweg.

        Wer anklickt und wieder weggeht, ohne zu tippen, hat nichts gesagt
        -- und ein feld, das dabei auf 0.0 springt, waere eine falle.
        Deshalb `_touched` und nicht der puffer: ein LEERER puffer nach dem
        wegloeschen der stellen IST die eingabe 0.0, ein leerer puffer ohne
        jeden anschlag ist keine.
        """
        if self._editing is None:
            return
        if self._touched:
            self._apply_buffer()
        self._close()

    def _open(self, field):
        """Das feld oeffnen: vorbelegt, und der wert als GANZES markiert."""
        self._commit()
        self._editing = field
        self._buffer = _edit_text(self._value_of(field))
        self._caret = len(self._buffer)
        self._select_all = True
        self._caret_phase = 0.0
        self._touched = False

    def _close(self):
        """Das feld schliessen OHNE zu schreiben -- der escape-weg."""
        self._editing = None
        self._buffer = ''
        self._caret = 0
        self._select_all = False
        self._touched = False

    def dismiss(self):
        # Woanders geklickt: die eingabe steht, das feld schliesst.
        self._commit()

    # -------------------------------------------------------------- eingabe

    def on_mouse_down(self, ctx, x, y, button):
        if button != 1:
            return True
        key = self._hit(ctx, x, y)
        if key is None or key.endswith('_sign'):
            self._commit()
        if key is None:
            return True

        for prefix, field, _pos, _neg in _AXES:
            if key == f'{prefix}_sign':
                # Richtung umschalten -- das ersetzt das tippbare minus.
                self._sign[field] = -self._sign[field]
                value = self._value_of(field)
                if value is not None:
                    self._write(field, abs(value))
                return True
            if key == f'{prefix}_value':
                if self._node() is None:
                    return True
                if self._editing != field:
                    # MIT DEM STEHENDEN WERT VORBELEGT, aber als GANZES
                    # MARKIERT. Ein leeres feld log ueber den knoten: der
                    # wert stand noch drin, die zeile zeigte '0'. Blosses
                    # vorbelegen dagegen hiesse ANHAENGEN -- aus '0.0' und
                    # getippten '250.5' wurde '0.02505', und der punkt, den
                    # man tippt, war schon vergeben. Markiert kann beides:
                    # tippen ersetzt, pfeil oder klick steigen ein.
                    self._open(field)
                else:
                    # SCHON OFFEN: der klick setzt den caret dorthin, wo er
                    # hinzeigt. Ohne das waere der pfeil der einzige weg in
                    # eine stehende zahl hinein.
                    self._caret = self._caret_from_x(ctx, x)
                    self._select_all = False
                    self._caret_phase = 0.0
                return True
        return True

    def on_wheel(self, ctx, dx, dy):
        """Das rad verstellt die zeile unter dem zeiger um einen feinschritt.

        Der schnelle griff, den die vier pfeilknoepfe frueher hatten -- nur
        ohne die 60 einheiten breite, die sie dafuer brauchten.
        """
        if not dy or self._node() is None:
            return False
        for prefix, field, _pos, _neg in _AXES:
            if self._hover_key != f'{prefix}_value':
                continue
            step = float(getattr(self.telemetry, 'maneuver_dv_step_fine', 1.0))
            # ERST schliessen, DANN lesen: der getippte wert steht bis zum
            # `_commit` nur im puffer, und vorher gelesen schriebe das rad
            # den alten wert plus einen schritt zurueck.
            if self._editing == field:
                self._commit()
            current = abs(self._value_of(field) or 0.0)
            self._write(field, current + step * (1.0 if dy > 0 else -1.0))
            return True
        return False

    def on_key(self, ctx, event):
        """Ein richtiges textfeld: caret, pfeile, ZIFFERN UND PUNKT."""
        if self._editing is None:
            return False
        key = event.key
        # Jeder anschlag zeigt den caret sofort wieder.
        self._caret_phase = 0.0

        if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_TAB):
            self._commit()
            return True
        if key == pygame.K_ESCAPE:
            # VERWERFEN: geschrieben wird erst beim schliessen, also steht
            # noch der alte wert im knoten und nichts ist rueckgaengig zu
            # machen.
            self._close()
            return True

        if key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_HOME, pygame.K_END):
            # Die pfeile HEBEN DIE MARKIERUNG AUF und setzen den caret an
            # ihren rand -- das ist der weg in eine stehende zahl hinein.
            # Sie duerfen NICHT als eingabe zaehlen: `event.unicode` ist
            # fuer sie leer, und `'' in _DIGITS` ist wahr (siehe unten).
            length = len(self._buffer)
            if key == pygame.K_LEFT:
                self._caret = 0 if self._select_all else max(0, self._caret - 1)
            elif key == pygame.K_RIGHT:
                self._caret = (length if self._select_all
                               else min(length, self._caret + 1))
            elif key == pygame.K_HOME:
                self._caret = 0
            else:
                self._caret = length
            self._select_all = False
            return True

        if key in (pygame.K_BACKSPACE, pygame.K_DELETE):
            if self._select_all:
                # Auf der markierung loescht er sie GANZ -- wie in jedem
                # textfeld. In die zahl hinein kommt man mit pfeil oder
                # klick, nicht mit dem rueckschritt.
                self._buffer = ''
                self._caret = 0
            elif key == pygame.K_BACKSPACE and self._caret > 0:
                self._buffer = (self._buffer[:self._caret - 1]
                                + self._buffer[self._caret:])
                self._caret -= 1
            elif key == pygame.K_DELETE and self._caret < len(self._buffer):
                self._buffer = (self._buffer[:self._caret]
                                + self._buffer[self._caret + 1:])
            self._select_all = False
            self._touched = True
            return True

        char = event.unicode
        # DIE LAENGE ZUERST PRUEFEN. `'' in '0123456789'` ist WAHR -- ein
        # teilstring-test, kein zeichentest. Ohne diese zeile zaehlte jede
        # taste OHNE zeichen (pfeile, umschalt, F-tasten) als ziffer: sie
        # warf die markierung weg und haengte nichts an, das feld sprang
        # also beim ersten pfeildruck auf leer und beim schliessen auf 0.
        if len(char) != 1:
            return True
        # Komma wie punkt: auf einer deutschen tastatur liegt auf dem
        # ziffernblock ein komma, und beides meint hier dieselbe stelle.
        if char == ',':
            char = '.'
        if char not in _DIGITS + '.':
            # Alles andere wird VERSCHLUCKT, nicht durchgereicht: sonst
            # setzte ein 'n' waehrend des tippens einen knoten.
            return True
        if self._select_all:
            # Die erste ziffer ERSETZT den markierten wert als ganzes.
            self._buffer = ''
            self._caret = 0
            self._select_all = False
        if char == '.' and '.' in self._buffer:
            return True
        self._buffer = (self._buffer[:self._caret] + char
                        + self._buffer[self._caret:])
        self._caret += 1
        # NICHT schreiben -- der puffer ist die anzeige, geschrieben wird
        # in `_commit`. Warum: siehe den block ueber `_commit`.
        self._touched = True
        return True

    # ------------------------------------------------- caret und schriftgrad

    def _value_role(self, ctx, text):
        """Der grad, in dem eine wertspalte gezeichnet wird.

        Muss in `draw` und im treffertest DERSELBE sein, sonst zeigt der
        caret auf eine stelle, die anders breit gesetzt ist.
        """
        room = self._regions(ctx)['pro_value'][2] - ctx.px(4.0) * 2.0
        return ('value' if ctx.text.measure(text, 'value')[0] <= room
                else 'caption')

    def _text_left(self, ctx, region, text, role):
        """Linke kante des RECHTSBUENDIG gesetzten textes."""
        return region[0] + region[2] - ctx.px(4.0) \
            - ctx.text.measure(text, role)[0]

    def _caret_from_x(self, ctx, x):
        """Die schreibstelle, auf die ein klick bei `x` zeigt.

        Gemessen wird ueber die PRAEFIXE desselben textes, nicht ueber eine
        angenommene zeichenbreite: der anzeigegrad ist tabellarisch, der
        punkt aber nicht, und ein '.' waere sonst eine ziffer breit.
        """
        text = self._buffer
        if not text:
            return 0
        prefix = 'pro' if self._editing == 'prograde' else 'nrm'
        region = self._regions(ctx)[f'{prefix}_value']
        role = self._value_role(ctx, text)
        left = self._text_left(ctx, region, text, role)
        best, best_d = 0, abs(x - left)
        for i in range(1, len(text) + 1):
            d = abs(x - (left + ctx.text.measure(text[:i], role)[0]))
            if d < best_d:
                best, best_d = i, d
        return best

    def _apply_buffer(self):
        """Den puffer in den knoten schreiben. NUR aus `_commit` heraus.

        Ein leerer puffer heisst hier 0.0 und nicht 'nichts' -- `_commit`
        ruft nur, wenn ueberhaupt getippt wurde, und dann ist das
        wegloeschen der stellen die eingabe null.
        """
        if self._editing is None:
            return
        try:
            magnitude = float(self._buffer) if self._buffer.strip('.') else 0.0
        except ValueError:
            return
        self._write(self._editing, magnitude)

    # -------------------------------------------------------------- zeichnen

    def update(self, ctx, dt):
        super().update(ctx, dt)
        # Der fokus kann anderswo landen (klick ins leere loest ihn ganz).
        # Dann ist die eingabe vorbei, auch ohne dass jemand dismiss ruft.
        if self._editing is not None and not self.focused:
            self._commit()
        if self._editing is None:
            self._sync_signs()
        else:
            self._caret_phase += float(dt)

    def draw(self, ctx):
        palette = ctx.theme.palette
        regions = self._regions(ctx)
        enabled = self._node() is not None

        chrome.plate(ctx, self.rect.x, self.rect.y, self.rect.w, self.rect.h,
                     fill=palette.panel_sunken, cut=8.0,
                     corners=self._corners(ctx))

        pad = ctx.px(4.0)
        room = regions['pro_value'][2] - pad * 2.0

        for prefix, field, positive, negative in _AXES:
            color = palette.snap if field == 'prograde' else palette.normal
            value = self._value_of(field)
            label = positive if self._sign[field] >= 0.0 else negative

            sx, sy, sw, sh = regions[f'{prefix}_sign']
            hover = 1.0 if self._hover_key == f'{prefix}_sign' else 0.0
            ctx.text.draw(
                label, sx, sy + sh * 0.5, role='caption',
                color=mix(color, palette.text, hover * 0.5) if enabled
                else palette.text_dimmer,
                align='left', valign='middle')

            vx, vy, vw, vh = regions[f'{prefix}_value']
            editing = self._editing == field
            if editing:
                text = self._buffer
                text_color = palette.text
                ctx.draw.rect(vx, vy, vw, vh, fill=palette.active,
                              radius=-ctx.px(3.0))
                # ZWEI ZUSTAENDE, ZWEI ZEICHEN. Markiert: ein band unter
                # dem ganzen wert, denn die naechste ziffer ersetzt ihn --
                # ohne das sah es aus, als loesche das feld die zahl von
                # selbst. Sonst: ein caret GENAU an der schreibstelle, denn
                # eingefuegt wird dort und nicht am ende.
                #
                # Das band traegt die ACHSENFARBE, keine neue: die vier
                # akzente tragen bedeutung (siehe modulkopf).
                erole = self._value_role(ctx, text)
                ewidth = ctx.text.measure(text, erole)[0] if text else 0.0
                eleft = vx + vw - pad - ewidth
                if self._select_all and text:
                    ctx.draw.rect(eleft - ctx.px(2.0), vy + ctx.px(2.0),
                                  ewidth + ctx.px(4.0), vh - ctx.px(4.0),
                                  fill=with_alpha(color, 0.5),
                                  radius=-ctx.px(2.0))
                elif (self._caret_phase % _CARET_BLINK_S) < _CARET_BLINK_S * 0.6:
                    # IN DER ACHSENFARBE und ueber der ganzen zeilenhoehe.
                    # Gemessen: in `palette.text`, einen pixel breit und auf
                    # der schreibstelle beginnend, unterschied er sich vom
                    # bild ohne ihn in genau VIER pixeln -- er lag unter dem
                    # weissen stamm der naechsten ziffer und war dieselbe
                    # farbe. Er sitzt deshalb MITTIG auf der zeichengrenze,
                    # also in der luecke zwischen zwei ziffern.
                    cw = max(2.0, ctx.px(1.5))
                    cx = eleft + (ctx.text.measure(text[:self._caret], erole)[0]
                                  if self._caret else 0.0)
                    ctx.draw.rect(cx - cw * 0.5, vy + ctx.px(1.0),
                                  cw, vh - ctx.px(2.0), fill=color)
            elif not enabled:
                text = units.delta_v(None)
                text_color = palette.text_dimmer
            else:
                text = f"{abs(value):.1f}"
                text_color = color
                if self._hover_key == f'{prefix}_value':
                    ctx.draw.rect(vx, vy, vw, vh, fill=palette.hover,
                                  radius=-ctx.px(3.0))

            # FESTER schriftgrad in beiden zeilen. Ihn je zeile mitwandern
            # zu lassen -- wie es `_draw_row` fuer die abgeleiteten werte
            # tut -- liesse '0.0' gross und '1907.6' klein nebeneinander
            # stehen, und das sind zwei felder DESSELBEN eingabepaars.
            #
            # OHNE EINHEIT, und zwar immer: 'm/s' kostet 25 der 70 px
            # wert-spalte und passte damit nur, wenn BEIDE zahlen gerade
            # klein sind. Eine einheit, die je nach wert erscheint und
            # verschwindet, ist schlechter als gar keine -- und die
            # DV-zeile zwei plaettchen weiter oben, in derselben spalte,
            # traegt sie.
            role = 'value' if ctx.text.measure(text, 'value')[0] <= room \
                else 'caption'
            ctx.text.draw(text, vx + vw - pad, vy + vh * 0.5, role=role,
                          color=text_color, align='right', valign='middle')



class ManeuverPlanBlock(_ManeuverPlate):
    """Die REICHWEITE der vorschau -- ueber der ALT-flanke.

    Ein mittenzentrierter RATEN-regler wie `ui/widgets/rate_slider.py`: die
    auslenkung ist die geschwindigkeit der aenderung, nicht ihr zielwert,
    und der knauf federt in die mitte zurueck. Er steckt hier im plaettchen
    statt als eigenes widget daneben, weil er ins raster gehoert -- und weil
    ein 168 einheiten breiter regler neben einem 124er kasten die flanke
    krumm macht.

    ZWEI FRAGEN, ZWEI REGLER: die vorhersagelinie (`PREDICT`, unten links)
    will aufloesung, der plan will weite. An einem regler haengend muesste
    man fuer weite immer aufloesung mitkaufen -- volle begruendung in
    `.claude/rules/maneuver.md`.
    """

    SIDE = 'right'
    PLACE = 'above'
    PLATE_HEIGHT = PLAN_HEIGHT

    #: Totzone um die mitte, als anteil des halben wegs.
    DEADZONE = 0.06

    def __init__(self, telemetry, navball=None, value=None, on_change=None,
                 minimum=0.25, maximum=32.0, sweep_seconds=2.5,
                 wheel_step=2.0, **kwargs):
        super().__init__(telemetry, navball=navball, **kwargs)
        self.value = value
        self.on_change = on_change
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.sweep_seconds = max(1e-3, float(sweep_seconds))
        self.wheel_step = float(wheel_step)
        self._offset = 0.0

    def _regions(self, ctx):
        gap = ctx.px(ctx.theme.frame_gap)
        pad = ctx.px(_PAD)
        x = self.rect.x + gap + pad
        w = self.rect.w - (gap + pad) * 2.0
        top = self.rect.y + gap + ctx.px(3.0)
        return {
            '_head': (x, top, w, ctx.px(11.0)),
            '_span': (x, top + ctx.px(11.0), w, ctx.px(_LINE_H)),
            'track': (x, top + ctx.px(11.0 + _LINE_H + 1.0), w, ctx.px(8.0)),
        }

    # -------------------------------------------------------------- zustand

    def _resolve(self):
        try:
            return float(self.value() if callable(self.value) else self.value)
        except (TypeError, ValueError):
            return 1.0

    def _clamp(self, mult):
        return max(self.minimum, min(self.maximum, float(mult)))

    def _span_text(self):
        """Die ZEITSPANNE der gezeichneten kette, zurueckgelesen.

        Nicht der multiplikator: der sagt nur 'viermal so viel wie sonst'.
        Die frage am regler ist, wie weit der plan reicht -- und das misst
        sich in zeit, weil knoten zeiten sind.
        """
        preview = getattr(self.telemetry, 'maneuver_preview', None)
        span = getattr(preview, 'span_seconds', None) if preview else None
        if span is None or not math.isfinite(float(span)):
            return '--'
        return units.duration(float(span))

    # -------------------------------------------------------------- eingabe

    def _offset_from_x(self, ctx, x):
        tx, _ty, tw, _th = self._regions(ctx)['track']
        half = tw * 0.5
        if half <= 0.0:
            return 0.0
        return max(-1.0, min(1.0, (float(x) - (tx + half)) / half))

    def on_mouse_down(self, ctx, x, y, button):
        if button == 1:
            self._offset = self._offset_from_x(ctx, x)
        return True

    def on_mouse_move(self, ctx, x, y):
        super().on_mouse_move(ctx, x, y)
        if self.pressed:
            self._offset = self._offset_from_x(ctx, x)
        return True

    def on_wheel(self, ctx, dx, dy):
        if not dy or self.on_change is None:
            return False
        factor = self.wheel_step ** (1.0 if dy > 0 else -1.0)
        self.on_change(self._clamp(self._resolve() * factor))
        return True

    def update(self, ctx, dt):
        super().update(ctx, dt)
        if not self.pressed:
            self._offset = ease(self._offset, 0.0, ctx.theme.motion.fast, dt)
        magnitude = abs(self._offset)
        if magnitude <= self.DEADZONE or self.on_change is None:
            return
        scaled = (magnitude - self.DEADZONE) / (1.0 - self.DEADZONE)
        factor = math.copysign(scaled ** 1.8, self._offset)
        # Exponentiell (linear im logarithmus): der plan wird in dekaden
        # wahrgenommen, nicht in stunden.
        k = math.log(self.maximum / self.minimum) / self.sweep_seconds
        self.on_change(self._clamp(self._resolve() * math.exp(k * factor * dt)))

    # -------------------------------------------------------------- zeichnen

    def draw(self, ctx):
        palette = ctx.theme.palette
        regions = self._regions(ctx)
        chrome.frame(ctx, self.rect.x, self.rect.y, self.rect.w, self.rect.h,
                     cut=14.0, corners=self._corners(ctx))

        hx, hy, hw, hh = regions['_head']
        ctx.text.draw('PLAN', hx, hy + hh * 0.5, role='caption',
                      color=with_alpha(palette.node, 0.9), valign='middle')
        ctx.text.draw(f"x{self._resolve():g}", hx + hw, hy + hh * 0.5,
                      role='unit', color=palette.text_dim,
                      align='right', valign='middle')

        sx, sy, sw, sh = regions['_span']
        ctx.text.draw(self._span_text(), sx + sw, sy + sh * 0.5,
                      role='value', color=palette.text,
                      align='right', valign='middle')

        tx, ty, tw, th = regions['track']
        track_h = max(1.0, ctx.px(4.0))
        track_y = ty + (th - track_h) * 0.5
        ctx.draw.rect(tx, track_y, tw, track_h, fill=palette.panel_sunken,
                      radius=track_h * 0.5)
        tick_w = max(1.0, ctx.px(1.5))
        ctx.draw.rect(tx + tw * 0.5 - tick_w * 0.5, track_y - ctx.px(2.0),
                      tick_w, track_h + ctx.px(4.0), fill=palette.text_dim)
        knob_x = tx + tw * 0.5 + tw * 0.5 * self._offset
        if abs(self._offset) > self.DEADZONE:
            lo, hi = sorted((tx + tw * 0.5, knob_x))
            ctx.draw.rect(lo, track_y, max(track_h, hi - lo), track_h,
                          fill=palette.node if self._offset > 0.0
                          else palette.text_dim,
                          radius=track_h * 0.5)
        ctx.draw.circle(
            knob_x, track_y + track_h * 0.5,
            ctx.px(5.0) + ctx.px(1.0) * self._hover_t,
            fill=mix(palette.text, palette.node, self._press_t),
            border_color=palette.panel_sunken,
            border_width=ctx.theme.border_width)


class ManeuverNodesBar(_ManeuverPlate):
    """Knotenwahl und die beiden knoepfe -- unter dem V/S-streifen."""

    SIDE = 'right'
    PLACE = 'below'
    PLATE_HEIGHT = NODES_HEIGHT

    def _regions(self, ctx):
        pad = ctx.px(_PAD)
        x = self.rect.x + pad
        w = self.rect.w - pad * 2.0
        y = self.rect.y + ctx.px(5.0)

        regions = {}
        count = max(1, int(getattr(self.telemetry, 'maneuver_max', 5)))
        pip_gap = ctx.px(_PIP_GAP)
        pip_w = (w - pip_gap * (count - 1)) / float(count)
        pip_h = ctx.px(_PIP_H)
        for i in range(count):
            regions[f'pip_{i}'] = (x + i * (pip_w + pip_gap), y, pip_w, pip_h)
        y += pip_h + ctx.px(4.0)

        foot_h = ctx.px(_FOOT_H)
        half = (w - ctx.px(4.0)) * 0.5
        regions['add'] = (x, y, half, foot_h)
        regions['delete'] = (x + half + ctx.px(4.0), y, half, foot_h)
        return regions

    def on_mouse_up(self, ctx, x, y, button):
        if button != 1:
            return True
        key = self._hit(ctx, x, y)
        if key is None:
            return True
        if key.startswith('pip_'):
            self.telemetry.select_node(int(key.split('_')[1]))
        elif key == 'add':
            self.telemetry.add_node()
        elif key == 'delete':
            self.telemetry.delete_node()
        return True

    def draw(self, ctx):
        palette = ctx.theme.palette
        telemetry = self.telemetry
        regions = self._regions(ctx)

        chrome.plate(ctx, self.rect.x, self.rect.y, self.rect.w, self.rect.h,
                     fill=palette.panel_sunken, cut=8.0,
                     corners=self._corners(ctx))

        count = int(getattr(telemetry, 'maneuver_count', 0))
        selected = int(getattr(telemetry, 'maneuver_selected', 0))
        total = int(getattr(telemetry, 'maneuver_max', 5))
        for i in range(total):
            rect = regions.get(f'pip_{i}')
            if rect is None:
                continue
            x, y, w, h = rect
            filled = i < count
            hover = 1.0 if self._hover_key == f'pip_{i}' else 0.0
            if filled:
                fill, _text, border = _button_colors(
                    ctx, filled and i == selected, palette.node, hover)
            else:
                fill, border = palette.disabled, palette.edge_inner
            chrome.plate(ctx, x, y, w, h, fill=fill, line=border, cut=3.0)

        full = count >= total
        has_node = getattr(telemetry, 'maneuver_node', None) is not None
        for key, label, usable in (('add', '+ NODE', not full),
                                   ('delete', 'DEL', has_node)):
            x, y, w, h = regions[key]
            hover = 1.0 if self._hover_key == key else 0.0
            if usable:
                fill, text_color, border = _button_colors(
                    ctx, False, palette.frame, hover)
            else:
                fill, text_color, border = (
                    palette.disabled, palette.text_dimmer, palette.edge_inner)
            chrome.plate(ctx, x, y, w, h, fill=fill, line=border, cut=3.0)
            ctx.text.draw(label, x + w * 0.5, y + h * 0.5, role='button_sm',
                          color=text_color, align='center', valign='middle')


class ManeuverGizmo(Widget):
    """Die ziehgriffe AN DER LINIE -- kein rechteck, nur treffer.

    Er zeichnet nichts: marker, stiele und pfeilspitzen malt der renderer
    (`render/maneuver.py`), weil nur der die zeitabhaengige
    frame-transformation kennt, die sie auf der bahn haelt. Dieses widget ist
    ausschliesslich der EINGABEweg dazu und trifft gegen genau die
    schirmpositionen, die dort gezeichnet wurden
    (`renderer.maneuver_node_hits`) -- dieselbe arbeitsteilung wie beim
    schwebezettel an den Ap/Pe-rauten.

    DIE GRIFFE SIND KNUEPPEL, KEINE SCHIEBEREGLER. Frueher war der wert die
    absolute zeigerstrecke mal einem faktor: fuer 500 m/s musste man den
    zeiger 660 px weit ziehen, also quer ueber den schirm, und am bildrand
    war schluss. Jetzt ist die AUSLENKUNG eine RATE (m/s je sekunde), die
    laeuft, solange gehalten wird -- dieselbe bauart wie der horizontregler
    (`ui/widgets/rate_slider.py`) und aus demselben grund: eine groesse ohne
    natuerliche obergrenze braucht ein steuer, keinen weg.

    Es ist auch der grund, warum die eingabe jetzt weich ist. Am absoluten
    regler sprang der wert mit jedem maus-ereignis; hier laeuft er
    ZEITINTEGRIERT in `update()` weiter, also mit der bildrate geglaettet
    und unabhaengig davon, wie oft das betriebssystem die maus meldet.

    ER FAENGT DIE MAUS NUR UEBER EINEM GRIFF. `hit_test` prueft radien, kein
    rechteck: das widget sitzt mitten im bild ueber der bahn, und eines, das
    dort flaechig klicks schluckt, blockierte koerperauswahl und
    kameraschwenk -- derselbe fehler, den `AttitudeRing.hit_test` fuer den
    kompassring behebt.

    EIN FRAME VERSATZ, UND ZWAR ABSICHTLICH. Die trefferliste entsteht in
    `renderer.render()`, also NACH der ereignisschleife; ein klick prueft
    damit gegen die positionen des vorbildes. Bei 60 bildern je sekunde ist
    das unsichtbar, und die alternative -- die schirmposition hier noch
    einmal auszurechnen -- waere genau die zweite transformation, die diese
    aufteilung vermeidet.
    """

    #: Totzone um die ruhelage, als anteil des vollen wegs. Darunter laeuft
    #: nichts -- ein knueppel, der im ruhezustand nicht ruht, verstellt den
    #: knoten schon beim anfassen.
    DEADZONE = 0.08

    def __init__(self, telemetry, dv_rate=260.0, travel_px=96.0, **kwargs):
        kwargs.setdefault('size', (0.0, 0.0))
        kwargs.setdefault('z', 55)
        super().__init__(**kwargs)
        self.telemetry = telemetry
        #: Volle auslenkung in delta-v je sekunde.
        self.dv_rate = float(dv_rate)
        #: Weg von der ruhelage bis zum vollausschlag, in bildschirmpixeln.
        self.travel_px = max(8.0, float(travel_px))
        self.blocks_mouse = True
        self._grab = None
        self._press = (0.0, 0.0)
        self._offset = 0.0

    # ------------------------------------------------------------- treffer

    def _renderer(self):
        return getattr(self.telemetry, 'renderer', None)

    def _pick(self, x, y):
        renderer = self._renderer()
        for hit in list(getattr(renderer, 'maneuver_node_hits', ()) or ()):
            for handle in hit['handles']:
                r = float(handle['radius_px']) + 6.0
                if math.hypot(x - handle['sx'], y - handle['sy']) <= r:
                    return ('handle', int(hit['index']), handle['kind'], handle)
            r = float(hit['radius_px']) + 6.0
            if math.hypot(x - hit['sx'], y - hit['sy']) <= r:
                return ('marker', int(hit['index']), None, hit)
        return None

    def hit_test(self, ctx, x, y):
        if not self.visible:
            return False
        return self._pick(float(x), float(y)) is not None

    # ------------------------------------------------------------- eingabe

    def _node(self, index):
        plan = getattr(self.telemetry, 'maneuver_plan', None)
        if plan is None or not (0 <= index < len(plan)):
            return None
        return plan.nodes[index]

    @staticmethod
    def _response(offset):
        """Auslenkung [-1, 1] -> ratenfaktor [-1, 1].

        Null in der totzone, sonst `sign * ((|x| - dz) / (1 - dz)) ** 2`.
        Dieselbe form wie `rate_slider._response`, nur mit ganzzahligem
        exponenten: quadratisch heisst, dass man nahe der ruhelage einzelne
        m/s trifft und am anschlag hunderte in der sekunde bekommt.
        """
        magnitude = abs(float(offset))
        if magnitude <= ManeuverGizmo.DEADZONE:
            return 0.0
        scaled = (magnitude - ManeuverGizmo.DEADZONE) / (1.0 - ManeuverGizmo.DEADZONE)
        return math.copysign(scaled * scaled, offset)

    def _publish_drag(self, curve=False, handle=None):
        renderer = self._renderer()
        if renderer is None:
            return
        renderer.maneuver_drag_active = bool(curve or handle is not None)
        # Erst DIESES flag laesst den renderer die BASISlinie in
        # schirmkoordinaten ablegen -- sie kostet eine projektion je punkt
        # und wird deshalb nur waehrend eines MARKER-zugs gebaut.
        renderer.maneuver_drag_curve = bool(curve)
        renderer.maneuver_drag_handle = handle

    def on_mouse_down(self, ctx, x, y, button):
        if button != 1:
            return True
        picked = self._pick(float(x), float(y))
        if picked is None:
            return True
        kind, index, handle_kind, payload = picked
        node = self._node(index)
        if node is None:
            return True

        self.telemetry.select_node(index)
        self._press = (float(x), float(y))
        self._offset = 0.0
        if kind == 'handle':
            axis = ('prograde' if handle_kind in ('prograde', 'retrograde')
                    else 'normal')
            self._grab = ('handle', index, handle_kind, axis,
                          float(payload['dir_sx']), float(payload['dir_sy']))
            self._publish_drag(handle=(index, handle_kind, 0.0))
        else:
            self._grab = ('marker', index, None, None, 0.0, 0.0)
            self._publish_drag(curve=True)
        return True

    def on_mouse_move(self, ctx, x, y):
        if self._grab is None:
            return True
        kind, index, handle_kind, _axis, dir_sx, dir_sy = self._grab

        if kind == 'handle':
            # Die zeigerbewegung auf die GRIFFRICHTUNG projizieren: nur der
            # anteil laengs des pfeils zaehlt, seitliches wackeln nicht.
            # Nach aussen ziehen heisst immer MEHR in richtung des pfeils --
            # am retrograde-pfeil ist das weniger prograde.
            dx = float(x) - self._press[0]
            dy = float(y) - self._press[1]
            along = dx * dir_sx + dy * dir_sy
            self._offset = max(-1.0, min(1.0, along / self.travel_px))
            self._publish_drag(handle=(index, handle_kind,
                                       self._offset * self.travel_px))
            return True

        # Marker: auf die naechste stuetzstelle der gezeichneten linie
        # rasten. Die zeit kommt damit VON DER LINIE und nicht aus einer
        # eigenen umrechnung -- der knoten kann gar nicht neben ihr landen.
        node = self._node(index)
        plan = getattr(self.telemetry, 'maneuver_plan', None)
        if node is None or plan is None:
            return True
        renderer = self._renderer()
        curve = getattr(renderer, 'maneuver_curve_screen', None)
        if curve is None or len(curve) < 2:
            return True
        # argmin statt einer schleife: die schirmkurve traegt einige hundert
        # punkte und wird bei JEDER mausbewegung durchsucht.
        try:
            d2 = ((curve[:, 0] - float(x)) ** 2 + (curve[:, 1] - float(y)) ** 2)
            node.t_node = float(curve[int(np.argmin(d2)), 2])
        except Exception:
            best_t = None
            best_d2 = None
            for point in curve:
                d = (float(x) - point[0]) ** 2 + (float(y) - point[1]) ** 2
                if best_d2 is None or d < best_d2:
                    best_d2, best_t = d, float(point[2])
            if best_t is None:
                return True
            node.t_node = best_t
        plan.touch()
        return True

    def update(self, ctx, dt):
        """Den knueppel INTEGRIEREN -- hier laeuft der wert, nicht im ereignis.

        Deshalb ist die eingabe weich: die aenderung haengt an der
        vergangenen zeit, nicht an der zahl der maus-meldungen.
        """
        super().update(ctx, dt)
        if self._grab is None or self._grab[0] != 'handle':
            return
        factor = self._response(self._offset)
        if factor == 0.0:
            return
        _kind, index, handle_kind, axis, _dsx, _dsy = self._grab
        node = self._node(index)
        plan = getattr(self.telemetry, 'maneuver_plan', None)
        if node is None or plan is None:
            return
        sign = -1.0 if handle_kind in ('retrograde', 'antinormal') else 1.0
        amount = sign * factor * self.dv_rate * float(dt)
        if axis == 'prograde':
            node.dv_prograde = float(node.dv_prograde) + amount
        else:
            node.dv_normal = float(node.dv_normal) + amount
        # OHNE touch() bliebe die gezeichnete linie auf dem alten stand.
        plan.touch()

    def on_mouse_up(self, ctx, x, y, button):
        self._grab = None
        self._offset = 0.0
        self._publish_drag()
        return True
