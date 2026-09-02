"""Mittenzentrierter RATEN-regler.

Zieht man ihn aus der mitte, aendert sich die gesteuerte groesse WEITER,
solange man haelt -- die auslenkung ist die GESCHWINDIGKEIT der aenderung,
nicht ihr zielwert. Loslassen federt weich in die mitte zurueck.

Getrennt von ui/widgets/slider.py (das ist ein LAGE-regler), weil die
interaktion eine voellig andere ist -- ein `log=`-schalter reichte nicht.

Benutzt fuer den vorhersage-horizont: die gezeichnete linienlaenge folgt
der auslenkung jeden frame ueber predictor.set_display_length() (O(1), kein
neuaufbau). Siehe plans/predictor_horizon_slider_design.md.
"""

import math

from ..core import Rect, Widget, ease
from ..theme import mix

#: Totzone um die mitte. Darunter zaehlt die auslenkung als null -- sonst
#: zittert der federweg beim zuruecklaufen noch winzige aenderungen ein,
#: und ein regler, der im ruhezustand nicht ruht, ist unbrauchbar.
DEADZONE = 0.06


def _response(offset):
    """Auslenkung [-1, 1] -> raten-faktor [-1, 1].

    Null in der totzone, sonst `sign * ((|x| - dz) / (1 - dz)) ** 1.8`. Die
    potenz gibt feines gefuehl nahe der mitte und volle geschwindigkeit am
    anschlag; ohne sie ist der regler bei kleiner auslenkung schon zu schnell.
    """
    magnitude = abs(float(offset))
    if magnitude <= DEADZONE:
        return 0.0
    scaled = (magnitude - DEADZONE) / (1.0 - DEADZONE)
    return math.copysign(scaled ** 1.8, offset)


class HorizonSlider(Widget):
    """Waagerechter raten-regler mit federrueckstellung in die mitte."""

    def __init__(self, value, minimum, maximum, on_change, predictor=None,
                 sweep_seconds=2.5, wheel_step=2.0, role='frame',
                 size=(168.0, None), **kwargs):
        super().__init__(size=size, **kwargs)
        self.value = value              # callable -> aktueller multiplikator
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.on_change = on_change
        self.predictor = predictor
        self.sweep_seconds = max(float(sweep_seconds), 1e-3)
        self.wheel_step = float(wheel_step)
        self.role = role
        self.blocks_mouse = True
        self._offset = 0.0

    # ------------------------------------------------------------- werte
    def resolve_value(self):
        return float(self.value() if callable(self.value) else self.value)

    @property
    def is_grabbing(self):
        return bool(self.pressed) and abs(self._offset) > DEADZONE

    def _clamp_mult(self, mult):
        return max(self.minimum, min(self.maximum, float(mult)))

    # ------------------------------------------------------------ layout
    def _tab_h(self, ctx):
        return ctx.text.measure('X', 'tab')[1] + ctx.px(4.0)

    def _readout_h(self, ctx):
        return ctx.text.measure('X', 'mono_readout')[1] + ctx.px(4.0)

    def measure(self, ctx):
        track = ctx.px(5.0)
        knob = ctx.px(18.0)
        height = self._tab_h(ctx) + self._readout_h(ctx) + max(track, knob) + ctx.px(6.0)
        return (ctx.px(168.0), height)

    def _track_rect(self, ctx):
        # Spur einwaerts gesetzt: die '-'/'+'-endglyphen sitzen px(9)+px(3)
        # AUSSERHALB der spur, und ohne diesen einzug ragte das minus ueber
        # die widget-kante hinaus fast bis an den bildschirmrand.
        inset = ctx.px(12.0)
        h = ctx.px(5.0)
        top_band = self._tab_h(ctx) + self._readout_h(ctx)
        free = self.rect.h - top_band - h
        top = self.rect.y + top_band + max(0.0, free) * 0.5
        return Rect(self.rect.x + inset, top, self.rect.w - inset * 2.0, h)

    # ------------------------------------------------------------ eingabe
    def _offset_from_x(self, ctx, x):
        track = self._track_rect(ctx)
        half = track.w * 0.5
        if half <= 0.0:
            return 0.0
        return max(-1.0, min(1.0, (float(x) - track.center_x) / half))

    def on_mouse_down(self, ctx, x, y, button):
        if button == 1 and self.enabled:
            self._offset = self._offset_from_x(ctx, x)
        return True

    def on_mouse_move(self, ctx, x, y):
        if self.pressed and self.enabled:
            self._offset = self._offset_from_x(ctx, x)
            return True
        return False

    def on_wheel(self, ctx, dx, dy):
        if not self.enabled or not dy:
            return False
        factor = self.wheel_step ** (1.0 if dy > 0 else -1.0)
        if self.on_change is not None:
            self.on_change(self._clamp_mult(self.resolve_value() * factor))
        return True

    # ------------------------------------------------------------ update
    def update(self, ctx, dt):
        super().update(ctx, dt)          # _hover_t / _press_t
        if self.predictor is not None:
            self.enabled = int(getattr(self.predictor, 'num_points', 1)) > 0
        if not self.pressed:
            self._offset = ease(self._offset, 0.0, ctx.theme.motion.fast, dt)
        f = _response(self._offset)
        if f != 0.0 and self.enabled and self.on_change is not None:
            # Exponentiell (linear im logarithmus): so fuehlt sich die
            # bewegung ueber die ganze spanne gleich an -- der horizont wird
            # in dekaden wahrgenommen, nicht in metern.
            k = math.log(self.maximum / self.minimum) / self.sweep_seconds
            new = self.resolve_value() * math.exp(k * f * float(dt))
            self.on_change(self._clamp_mult(new))

    # ------------------------------------------------------------ zeichnen
    def _horizon_metres(self):
        pred = self.predictor
        if pred is not None:
            try:
                length = pred.get_display_length()
            except Exception:
                length = None
            if length and math.isfinite(length):
                return float(length)
        return None

    def draw(self, ctx):
        # Lazy, sonst zieht ui/widgets/__init__ das ganze ui/hud-paket beim
        # import herein (layout.py importiert wiederum ui/widgets -> zyklus).
        from ..hud import chrome
        from .. import units

        palette = ctx.theme.palette
        color = palette.accent_for(self.role)          # cyan = daten/rahmen
        track = self._track_rect(ctx)
        enabled = self.enabled

        # Notch-tab auf der oberkante.
        chrome.tab(ctx, chrome.tab_text('PREDICT'),
                   track.x + ctx.px(12.0), self.rect.y + self._tab_h(ctx),
                   color=color if enabled else palette.text_dim, edge='top')

        # Ablesewert: die GEZEICHNETE horizontlaenge, aus dem predictor
        # zurueckgelesen -- nie lokal gehalten.
        metres = self._horizon_metres()
        ctx.text.draw(
            units.distance(metres) if metres is not None else '--',
            self.rect.right, self.rect.y + self._tab_h(ctx),
            role='mono_readout',
            color=palette.text if enabled else palette.text_dim,
            align='right',
        )

        # Spur.
        ctx.draw.rect(track.x, track.y, track.w, track.h,
                      fill=palette.panel_sunken, radius=track.h * 0.5)

        # Mittenkerbe.
        tick_w = max(1.0, ctx.px(2.0))
        ctx.draw.rect(track.center_x - tick_w * 0.5, track.y - ctx.px(2.0),
                      tick_w, track.h + ctx.px(4.0),
                      fill=palette.text if enabled else palette.text_dim)

        # Fuellbalken von der MITTE zum knauf -- richtung, nicht lage.
        knob_x = track.center_x + track.w * 0.5 * self._offset
        if enabled and abs(self._offset) > DEADZONE:
            lo, hi = sorted((track.center_x, knob_x))
            fill = color if self._offset > 0.0 else palette.text_dim
            ctx.draw.rect(lo, track.y, max(track.h, hi - lo), track.h,
                          fill=fill, radius=track.h * 0.5)

        # Endkappen-glyphen.
        gy = track.center_y
        gr = ctx.px(3.0)
        gcol = palette.text_muted if enabled else palette.text_dim
        gw = max(1.0, ctx.px(1.4))
        ctx.draw.line(track.x - ctx.px(9.0) - gr, gy, track.x - ctx.px(9.0) + gr,
                      gy, gcol, width=gw)                      # minus links
        ctx.draw.line(track.right + ctx.px(9.0) - gr, gy,
                      track.right + ctx.px(9.0) + gr, gy, gcol, width=gw)
        ctx.draw.line(track.right + ctx.px(9.0), gy - gr,
                      track.right + ctx.px(9.0), gy + gr, gcol, width=gw)  # plus

        # Knauf.
        knob_r = ctx.px(7.0) + ctx.px(1.5) * self._hover_t
        ctx.draw.circle(
            knob_x, track.center_y, knob_r,
            fill=mix(palette.text, palette.accent_strong, self._press_t)
            if enabled else palette.text_dim,
            border_color=palette.panel_sunken, border_width=ctx.theme.border_width,
        )
