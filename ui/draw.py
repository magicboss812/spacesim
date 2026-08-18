"""Zeichen-primitive der UI-schicht, alle auf EINEM SDF-shader.

shaders/ui_rect.{vert,frag} kann abgerundete rechtecke mit rahmen, schatten
und verlauf -- und, weil ein kreis nur ein rechteck mit radius = halbe kante
ist, auch kreise, ringe und kreisboegen. Damit deckt eine einzige pipeline
praktisch die gesamte HUD-flaeche ab; das ist der grund, warum es hier keine
zweite geometrie-pipeline gibt.

KOORDINATEN: alle oeffentlichen methoden nehmen TOP-DOWN bildschirmpixel
(ursprung oben links, y nach unten) -- dieselbe konvention wie pygames
maus-ereignisse. Die umrechnung in die ortho-konvention (y nach oben) des
shaders passiert ausschliesslich in _submit(). Das ist die in CLAUDE.md
geforderte "eine konvention, umrechnung an der grenze".

WINKEL: grad, gegen den uhrzeigersinn, 0 = nach rechts -- so, wie man es
auf dem bildschirm sieht.
"""

import math
import os

import moderngl
import numpy as np

_SHADER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shaders'
)

_TAU = 6.28318530718
_TRANSPARENT = (0.0, 0.0, 0.0, 0.0)


class UIDraw:
    """Duenne huelle um den SDF-shader. Haelt keinen zeichen-zustand."""

    def __init__(self, ctx, width, height):
        self.ctx = ctx
        self.width = int(width)
        self.height = int(height)
        self._program = None
        self._vao = None
        self._quad_vbo = None
        # Zuletzt gesetzte uniform-werte, siehe _uniform().
        self._last = {}
        self._init_pipeline()

    # Per-instanz-layout, muss zu den i_*-attributen in ui_rect.vert passen:
    # rect(4) expand(1) rotation(1) radius(4) fill(4) fill2(4) gradient(1)
    # border_color(4) border_width(1) shadow_color(4) shadow_offset(2)
    # shadow_softness(1) arc(2) = 33 floats.
    _INSTANCE_FLOATS = 33
    _INSTANCE_FORMAT = '4f 1f 1f 4f 4f 4f 1f 4f 1f 4f 2f 1f 2f/i'
    _INSTANCE_NAMES = (
        'i_rect', 'i_expand', 'i_rotation', 'i_radius', 'i_fill', 'i_fill2',
        'i_gradient', 'i_border_color', 'i_border_width', 'i_shadow_color',
        'i_shadow_offset', 'i_shadow_softness', 'i_arc',
    )

    def _init_pipeline(self):
        try:
            with open(os.path.join(_SHADER_DIR, 'ui_rect.vert'), 'r', encoding='utf-8') as f:
                vertex_source = f.read()
            with open(os.path.join(_SHADER_DIR, 'ui_rect.frag'), 'r', encoding='utf-8') as f:
                fragment_source = f.read()
            program = self.ctx.program(
                vertex_shader=vertex_source, fragment_shader=fragment_source
            )
            quad = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4')
            self._quad_vbo = self.ctx.buffer(quad.tobytes())
            self._program = program
            self._instance_capacity = 0
            self._instance_count = 0
            self._instance_data = np.empty((0, self._INSTANCE_FLOATS), dtype='f4')
            self._inst_vbo = None
            self._vao = None
            self._ensure_capacity(256)
            self._last = {}
        except Exception as exc:
            print(f"UI DRAW WARNING: ui_rect-pipeline nicht verfuegbar ({exc})")
            self._program = None
            self._vao = None

    def _ensure_capacity(self, wanted):
        """Instanz-puffer (CPU + GL) auf mindestens `wanted` zeilen bringen."""
        if wanted <= self._instance_capacity:
            return
        cap = max(256, self._instance_capacity * 2, int(wanted))
        data = np.empty((cap, self._INSTANCE_FLOATS), dtype='f4')
        if self._instance_count:
            data[:self._instance_count] = self._instance_data[:self._instance_count]
        self._instance_data = data
        # GL-puffer und VAO neu aufbauen -- der VAO haelt den alten puffer.
        for obj in (self._vao, self._inst_vbo):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass
        self._inst_vbo = self.ctx.buffer(
            reserve=cap * self._INSTANCE_FLOATS * 4, dynamic=True
        )
        self._vao = self.ctx.vertex_array(
            self._program,
            [
                (self._quad_vbo, '2f', 'a_corner'),
                (self._inst_vbo, self._INSTANCE_FORMAT, *self._INSTANCE_NAMES),
            ],
        )
        self._instance_capacity = cap

    @property
    def available(self):
        return self._vao is not None

    # ------------------------------------------------------------ primitive

    def rect(self, x, y, w, h, fill=None, radius=0.0, border_color=None,
             border_width=0.0, shadow=None, shadow_offset=(0.0, -3.0),
             shadow_softness=8.0, gradient_to=None, rotation_deg=0.0,
             arc=None):
        """Abgerundetes rechteck. (x, y) = obere linke ecke, top-down.

        radius:       skalar oder (oben-links, oben-rechts, unten-rechts,
                      unten-links) in pixeln; wird im shader auf die halbe
                      kante geklemmt, 'pill' erreicht man mit einem grossen wert.
        gradient_to:  zweite farbe -> vertikaler verlauf von fill (oben) dorthin.
        arc:          (startwinkel_grad, ueberstrichener_winkel_grad) oder None.
        """
        if self._vao is None:
            return
        if w <= 0.0 or h <= 0.0:
            return

        fill = fill or _TRANSPARENT
        border_color = border_color or _TRANSPARENT
        shadow_color = shadow or _TRANSPARENT

        if isinstance(radius, (tuple, list)):
            radii = tuple(float(r) for r in radius[:4])
            if len(radii) < 4:
                radii = radii + (0.0,) * (4 - len(radii))
        else:
            radii = (float(radius),) * 4

        # Auf ganze pixel rasten: eine 1px-rahmenlinie auf einer halben
        # pixelgrenze wird sonst zu zwei halbdeckenden zeilen.
        left = round(float(x))
        top = round(float(y))
        width = max(1.0, round(float(w)))
        height = max(1.0, round(float(h)))

        # Das quad muss gross genug sein fuer schatten und kantenglaettung,
        # sonst wird der schatten am quad-rand abgeschnitten.
        expand = 2.0
        if shadow_color[3] > 0.0:
            expand += float(shadow_softness) + max(
                abs(float(shadow_offset[0])), abs(float(shadow_offset[1]))
            )

        if arc is None:
            arc_params = (0.0, _TAU)
        else:
            start_deg, sweep_deg = arc
            sweep = math.radians(float(sweep_deg))
            if sweep >= _TAU or sweep <= 0.0:
                arc_params = (0.0, _TAU)
            else:
                arc_params = (math.radians(float(start_deg)) % _TAU, sweep)

        self._submit(
            left, top, width, height, radii, fill,
            gradient_to if gradient_to is not None else fill,
            1.0 if gradient_to is not None else 0.0,
            border_color, float(border_width),
            shadow_color, shadow_offset, float(shadow_softness),
            expand, math.radians(float(rotation_deg)), arc_params,
        )

    def circle(self, cx, cy, radius, fill=None, border_color=None,
               border_width=0.0, shadow=None, **kwargs):
        """Kreis um (cx, cy), top-down."""
        d = float(radius) * 2.0
        self.rect(
            float(cx) - float(radius), float(cy) - float(radius), d, d,
            fill=fill, radius=float(radius), border_color=border_color,
            border_width=border_width, shadow=shadow, **kwargs
        )

    def ring(self, cx, cy, radius, thickness, color, **kwargs):
        """Kreisring. Umgesetzt als kreis OHNE fuellung mit rahmen der
        gewuenschten staerke -- derselbe SDF, kein zweiter shader."""
        self.circle(
            cx, cy, radius, fill=_TRANSPARENT, border_color=color,
            border_width=float(thickness), **kwargs
        )

    def arc(self, cx, cy, radius, thickness, color, start_deg, sweep_deg, **kwargs):
        """Kreisbogen. Winkel gegen den uhrzeigersinn, 0 = nach rechts."""
        self.circle(
            cx, cy, radius, fill=_TRANSPARENT, border_color=color,
            border_width=float(thickness), arc=(start_deg, sweep_deg), **kwargs
        )

    def line(self, x0, y0, x1, y1, color, width=1.0, cap='butt'):
        """Beliebig gedrehte linie -- ein um ihre achse rotiertes rechteck.

        cap='round' rundet die enden ueber den eckradius ab.
        """
        dx = float(x1) - float(x0)
        # In ortho-koordinaten zeigt +y nach OBEN, im top-down-eingang nach
        # unten -- deshalb das vorzeichen beim winkel.
        dy = -(float(y1) - float(y0))
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        angle_deg = math.degrees(math.atan2(dy, dx))
        cx = (float(x0) + float(x1)) * 0.5
        cy = (float(y0) + float(y1)) * 0.5
        radius = float(width) * 0.5 if cap == 'round' else 0.0
        self.rect(
            cx - length * 0.5, cy - float(width) * 0.5, length, float(width),
            fill=color, radius=radius, rotation_deg=angle_deg,
        )

    def divider(self, x, y, length, color, thickness=1.0, vertical=False):
        """Trennlinie. Achsenparallel, deshalb ohne rotation und damit ohne
        rasterungs-unschaerfe an den enden."""
        if vertical:
            self.rect(x, y, thickness, length, fill=color)
        else:
            self.rect(x, y, length, thickness, fill=color)

    # --------------------------------------------------------------- intern

    def _submit(self, left, top, width, height, radii, fill, fill2, gradient,
                border_color, border_width, shadow_color, shadow_offset,
                shadow_softness, expand, rotation_rad, arc_params):
        """Instanz in den stapel legen -- gezeichnet wird erst in flush().

        Frueher war jeder aufruf ein eigener draw mit 14 uniform-schreib-
        vorgaengen; bei gut 160 aufrufen pro HUD-frame war das der groesste
        einzelposten der UI-zeit. Jetzt sammeln sich aufeinanderfolgende
        formen in einem per-instanz-puffer und gehen als EIN instanzierter
        draw an die GPU. Die instanz-reihenfolge ist die aufruf-reihenfolge,
        das blending bleibt also exakt gleich; text (eigene pipeline) stoesst
        vor seinem eigenen draw einen flush an, damit die schichtung stimmt.
        """
        # TOP-DOWN -> ORTHO: die untere linke ecke liegt bei
        # height - (top + hoehe). Das ist die EINZIGE stelle der UI-schicht,
        # an der die konvention wechselt.
        ortho_y = float(self.height) - (top + height)

        n = self._instance_count
        if n >= self._instance_capacity:
            self._ensure_capacity(n + 1)
        row = self._instance_data[n]
        row[0] = left
        row[1] = ortho_y
        row[2] = width
        row[3] = height
        row[4] = expand
        row[5] = rotation_rad
        row[6:10] = radii
        row[10:14] = fill
        row[14:18] = fill2
        row[18] = gradient
        row[19:23] = border_color
        row[23] = border_width
        row[24:28] = shadow_color
        row[28] = shadow_offset[0]
        row[29] = shadow_offset[1]
        row[30] = shadow_softness
        row[31] = arc_params[0]
        row[32] = arc_params[1]
        self._instance_count = n + 1

    def flush(self):
        """Alle seit dem letzten flush gesammelten formen zeichnen."""
        count = self._instance_count
        if count <= 0 or self._vao is None:
            return
        self._instance_count = 0
        viewport = (float(self.width), float(self.height))
        if self._last.get('u_viewport') != viewport:
            self._last['u_viewport'] = viewport
            self._program['u_viewport'].value = viewport
        self._inst_vbo.write(self._instance_data[:count].tobytes())
        self._vao.render(moderngl.TRIANGLE_STRIP, instances=count)

    def resize(self, width, height):
        self.width = int(width)
        self.height = int(height)
        # u_viewport haengt an der fenstergroesse -- der cache waere sonst
        # genau ueber diesen einen wert veraltet und die UI landete im
        # falschen massstab.
        self._last = {}

    def release(self):
        for obj in (self._vao, self._quad_vbo, self._inst_vbo, self._program):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass
        self._vao = None
        self._quad_vbo = None
        self._inst_vbo = None
        self._program = None
