"""Der zettel an einem ANGEKLICKTEN punkt der vorhersagelinie.

Ein klick auf die linie (nicht auf einen koerper, nicht auf eine raute)
heftet ihn an: `InputRouter.handle_world_click` fragt
`Renderer.pick_prediction_time` nach der zeit des punktes und ruft `pin()`.
Ueberfahren zeigt NICHTS -- die linie liegt mitten im bild, und ein zettel,
der beim blossen ueberstreichen aufspringt, waere laerm.

Er zeigt, was man an einem beliebigen bahnpunkt wissen will: den ABSTAND zum
mittelpunkt des bezugskoerpers (dieselbe groesse wie AP/PE, siehe
`.claude/rules/hud-ui.md`), das TEMPO relativ zu ihm und die ZEIT bis zur
ankunft -- und darunter "WARP TO" (runtime/warp_to.py), ziel der punkt minus
`warp_to.point_margin_s` (vorgabe 0: man will dort ankommen).

Gespeichert wird nur die ZEIT des punktes. Der renderer wertet sie jedes bild
auf der aktuellen linie aus (`Renderer._draw_trajectory_pin`) und meldet die
schirmposition samt zustand zurueck. Liegt die zeit hinter dem schiff
(angekommen) oder jenseits der linie, schliesst der zettel.

ANDERE FORM ALS DIE APSIDEN: eine quadratische marke mit vier strichen statt
einer raute, und die DATENFARBE cyan (`palette.orbit`) statt Ap-blau/Pe-orange.
"""

import math

from ..theme import with_alpha
from .. import units
from . import chrome
from .apsis_tooltip import ApsisTooltip
from ship.maneuver.preview import body_state_at


class PointTooltip(ApsisTooltip):
    """Abstand, tempo und ankunftszeit eines angeklickten bahnpunkts."""

    #: Unter der marke steht keine abstands-fahne -- aber der notch-tab des
    #: zettels reitet auf seiner oberkante und braucht denselben platz.
    HAS_FLAG = False

    def __init__(self, telemetry, **kwargs):
        super().__init__(telemetry, **kwargs)
        self._pin_t = None
        self._pin_state = None

    # ------------------------------------------------------------ anheften

    def pin(self, t_abs):
        """Den punkt zur zeit `t_abs` anheften. False, wenn er vorbei ist."""
        if t_abs is None or not self.drawn or float(t_abs) <= self._now():
            return False
        self._pin_t = float(t_abs)
        self.pinned = True
        self._missing_s = 0.0
        self._marker = None
        self._push_pin()
        return True

    def pin_at(self, x, y):
        # Rauten gehoeren dem apsiden-zettel; dieser hier wird ueber pin()
        # angeheftet.
        return False

    def unpin(self):
        super().unpin()
        self._pin_t = None
        self._pin_state = None
        self._push_pin()

    def _push_pin(self):
        renderer = self._renderer()
        if renderer is not None:
            renderer.trajectory_pin_t = self._pin_t

    def _pick(self, ctx):
        return None

    def _follow(self, dt):
        if self._pin_t is None or self._pin_t <= self._now():
            self.unpin()
            return
        hit = getattr(self._renderer(), 'trajectory_pin_hit', None)
        if hit is None or abs(float(hit[6]) - self._pin_t) > 1e-6:
            # Im frame des anheftens hat der renderer noch nicht gezeichnet;
            # erst ein laengeres fehlen heisst: die linie reicht nicht mehr
            # bis dort.
            self._missing_s += dt
            if self._missing_s > 0.5:
                self.unpin()
            return
        self._missing_s = 0.0
        sx, sy, x, y, vx, vy, t_abs = hit
        renderer = self._renderer()
        radius = float(getattr(renderer, 'apsis_marker_radius_px', 5.0))
        self._marker = {
            'x': float(sx), 'y': float(sy), 'radius': radius * 2.0,
            'apoapsis': False, 'preview': False, 't_abs': float(t_abs),
            'distance': math.nan, 'speed': math.nan, 'alpha': 1.0,
        }
        reference = getattr(getattr(self.telemetry, 'ui_state', None),
                            'reference_body', None)
        if reference is not None:
            rx, ry, rvx, rvy = body_state_at(reference, float(t_abs))
            self._marker['distance'] = math.hypot(x - rx, y - ry)
            self._marker['speed'] = math.hypot(vx - rvx, vy - rvy)

    def _refresh_position(self):
        """Wie beim apsiden-zettel: die position dieses bildes."""
        hit = getattr(self._renderer(), 'trajectory_pin_hit', None)
        if (hit is not None and self._pin_t is not None
                and abs(float(hit[6]) - self._pin_t) <= 1e-6):
            self._marker['x'] = float(hit[0])
            self._marker['y'] = float(hit[1])

    # ------------------------------------------------------------ warp to

    def warp_offered(self):
        return (self.pinned and self._marker is not None
                and self._warp_to() is not None)

    def warp_target(self):
        warp = self._warp_to()
        margin = float(getattr(warp, 'point_margin_s', 0.0)) if warp else 0.0
        return self._marker['t_abs'] - margin

    def warp_label(self):
        return 'punkt'

    # ------------------------------------------------------------ anzeige

    def _flag_height(self):
        ctx = self._ctx
        if ctx is None:
            return 0.0
        return ctx.text.measure('X', 'tab')[1] + ctx.px(4)

    def _rows(self):
        marker = self._marker
        if marker is None:
            return ()
        eta = marker['t_abs'] - self._now()
        return (
            ('R', units.distance(marker['distance'])
             if math.isfinite(marker['distance']) else '--'),
            ('REL V', units.speed(marker['speed'])
             if math.isfinite(marker['speed']) else '--'),
            ('ETA', units.duration(eta if eta > 0.0 else 0.0)),
        )

    def _color(self, ctx):
        return ctx.theme.palette.accent_for('orbit')

    def draw(self, ctx):
        super().draw(ctx)
        # Kennung oben am zettel: ein notch-tab in der datenfarbe, damit er
        # auch ohne die marke nicht mit einem apsiden-zettel zu verwechseln
        # ist.
        if self._t <= 0.004 or self._marker is None:
            return
        rows = self._rows()
        if not rows:
            return
        width, height = self._measure(ctx, rows)
        left, top = self._anchor(ctx, width, height)
        top += (1.0 - max(0.0, min(1.0, self._t))) * ctx.px(4.0)
        chrome.tab(ctx, chrome.tab_text('ORBIT', 'POINT'),
                   left + ctx.px(10.0), top,
                   color=with_alpha(self._color(ctx), max(0.0, min(1.0, self._t))),
                   edge='top')
