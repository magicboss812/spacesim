"""Das zeichnen des schiffs und seiner orientierungsvektoren.

Die vektor-geometrie selbst liegt in `ship/art.py` (reines numpy); hier steht
nur der weg auf den schirm. `orbital_frame_directions` ist die EINZIGE
quelle der wahrheit fuer den rast-autopiloten -- siehe
.claude/rules/camera-input.md.
"""
import math

import moderngl

import numpy as np

from physics.reference_frames import apparent_orbital_directions
from physics.vec import Vec2
from ship import art as ship_art


class ShipDrawMixin:
    """Das schiff: sprite, pfeil, fahne und die orientierung.

    `_apply_orientation_snap` laeuft INNERHALB von render(), unmittelbar bevor
    der pfeil gezeichnet wird -- mit demselben rahmen und derselben rahmenzeit
    wie die gezeichneten prograde/normal-vektoren. Nur so landet die nase exakt
    auf dem vektor, unabhaengig von sim_dt und drehrate des rahmens."""

    def orbital_frame_directions(self, ship, reference_body=None, prediction_points=None):
        """The frame-space orbital directions of the ship.

        Single source of truth for the orientation snap and the HUD
        (`ui/hud/telemetry.py`): prograde/normal_in are the tangent/inward of
        the *drawn* predictor line in the active plotting frame;
        retrograde/antinormal are their opposites. Evaluated at the renderer's current ``_frame_time_s`` — the
        same instant the ship arrow is drawn.
        """
        frame = self._active_frame()
        if reference_body is None:
            reference_body = getattr(self, "current_reference_body", None)
        ref_pos = getattr(reference_body, "position", None)
        directions = apparent_orbital_directions(
            frame, self._frame_time_s, ship.position, ship.velocity, ref_pos,
            points=prediction_points,
        )
        # Die FUENFTE richtung: die schubrichtung eines scharfgeschalteten
        # manoeverknotens. Sie kommt als WELTvektor herein (der ausfuehrer
        # loest sie einmal auf und haelt sie fest) und wird hier in
        # frame-koordinaten gedreht, damit _apply_orientation_snap sie wie
        # jede andere behandeln kann -- ohne sonderweg fuer den autopiloten.
        burn = getattr(self, "maneuver_burn_direction", None)
        if burn is not None:
            try:
                fx, fy = frame.to_this_frame_vector_xy(
                    self._frame_time_s, float(burn[0]), float(burn[1]))
                length = math.hypot(fx, fy)
                if length > 1e-30:
                    directions["node"] = Vec2(fx / length, fy / length)
            except Exception:
                pass
        return frame, directions

    def _apply_orientation_snap(self, ship, ship_control, reference_body,
                                prediction_points, real_dt):
        """Tie the ship nose to the drawn orbital vector for the latched snap.

        Computes the world heading whose *drawn* arrow coincides with the
        frame-space snap vector, using ``heading_from_this_frame`` at the SAME
        ``_frame_time_s`` that ``_draw_body`` uses to draw the arrow. This makes
        the nose lock onto the on-screen prograde/normal vector exactly, with no
        dependence on sim_dt or frame rotation rate. The ship's stored ``theta``
        stays in world space (physics remains absolute); only the render-time
        transform is inverted here.
        """
        if ship is None or ship_control is None:
            return
        mode = getattr(ship_control, "snap_mode", None)
        if not mode:
            return
        try:
            frame, directions = self.orbital_frame_directions(
                ship, reference_body, prediction_points
            )
            d = directions.get(mode)
            if d is None:
                return
            # `theta` ist im uhrzeigersinn gemessen (siehe _draw_ship_sprite und
            # schiff.apply_thrust: nasenrichtung = (cos theta, -sin theta)).
            # Damit die nase auf der frame-richtung d landet, muss also
            # (cos theta_f, -sin theta_f) == d gelten -> d.y negiert messen.
            ang_frame = math.atan2(-float(d.y), float(d.x))
            try:
                theta_target = frame.heading_from_this_frame(self._frame_time_s, ang_frame)
            except Exception:
                theta_target = ang_frame
            ship_control.orient_towards_angle(theta_target, real_dt)
        except Exception:
            return

    def _ship_relative_speed_m_s(self, ship, reference_body=None):
        if ship is None:
            return None

        try:
            vx = float(ship.velocity.x)
            vy = float(ship.velocity.y)
        except Exception:
            return None

        if reference_body is not None:
            try:
                vx -= float(reference_body.velocity.x)
                vy -= float(reference_body.velocity.y)
            except Exception:
                pass

        return math.hypot(vx, vy)

    def _ship_frame_speed_m_s(self, ship, dt_s=1.0):
        """
        Returns the ship's apparent speed in the active plotting frame.

        This respects translated, rotating, target-overlay, and time-dependent
        frames by finite-differencing the active frame transform. It does not
        use the clamped visual velocity vector length.
        """
        if ship is None:
            return None

        try:
            t0 = float(self._frame_time_s)
            dt = max(1e-3, float(dt_s))

            x0 = float(ship.position.x)
            y0 = float(ship.position.y)
            vx = float(ship.velocity.x)
            vy = float(ship.velocity.y)

            frame = self._active_frame()

            fx0, fy0 = frame.to_this_frame_xy(t0, x0, y0)
            fx1, fy1 = frame.to_this_frame_xy(
                t0 + dt,
                x0 + vx * dt,
                y0 + vy * dt,
            )

            dvx = float(fx1) - float(fx0)
            dvy = float(fy1) - float(fy0)

            return math.hypot(dvx, dvy) / dt
        except Exception:
            return None

    def _ship_zoom_shrink_factor(self, camera_scale):
        """Massstabs-faktor des schiffs fuer die aktuelle zoomstufe.

        1.0 bei `ship_zoom_shrink_start_scale` und darueber,
        `ship_zoom_shrink_min` bei `ship_zoom_shrink_end_scale` und darunter,
        dazwischen ein smoothstep im LOG-raum der skala. Log, weil zoom
        multiplikativ ist (`camera._ease_scale` interpoliert aus demselben
        grund logarithmisch): linear in `scale` gerechnet waere die ganze
        ueberblendung in der obersten dekade verbraucht und der rest ein
        sprung. Smoothstep statt gerade, damit auch die ENDEN der rampe
        knickfrei sind -- ein linearer verlauf springt am start- und
        endpunkt sichtbar in der aenderungsrate.

        Reine rechnung, kein GL -- damit sie ohne kontext pruefbar ist.
        """
        if not bool(getattr(self, 'ship_zoom_shrink_enabled', True)):
            return 1.0
        try:
            scale = float(camera_scale)
            start = float(self.ship_zoom_shrink_start_scale)
            end = float(self.ship_zoom_shrink_end_scale)
            floor = float(self.ship_zoom_shrink_min)
        except (TypeError, ValueError):
            return 1.0
        floor = max(0.05, min(1.0, floor))
        if not (math.isfinite(scale) and scale > 0.0):
            return 1.0
        if not (start > 0.0 and end > 0.0 and end < start):
            # Unbrauchbar konfiguriert (vertauscht oder gleich): lieber die
            # volle feste groesse als eine division durch null.
            return 1.0
        if scale >= start:
            return 1.0
        if scale <= end:
            return floor
        t = math.log(start / scale) / math.log(start / end)
        t = t * t * (3.0 - 2.0 * t)
        return 1.0 + (floor - 1.0) * t

    def _ship_length_px(self):
        """Gezeichnete schiffslaenge in echten bildschirm-pixeln.

        Basislaenge (design-einheiten -> `ui_px`) x spieler-regler
        `ship_render_scale` x zoom-schrumpfung. EIN weg fuer alle
        zeichenpfade, damit grafik, pfeil-fallback und label-abstand nicht
        auseinanderlaufen.
        """
        return (self.ui_px(self.ship_length_px)
                * max(0.01, float(self.ship_render_scale))
                * max(0.05, float(getattr(self, '_ship_zoom_factor', 1.0))))

    def _ship_half_height_px(self):
        """Halbe hoehe der gezeichneten schiffs-grafik in bildschirm-pixeln.

        Bezugsgroesse fuer alles, was NEBEN dem schiff sitzt (labels). Faellt
        auf die halbe breite des fallback-pfeils zurueck, wenn die grafik aus
        ist.
        """
        geo = self._ship_geometry() if self.ship_sprite_enabled else None
        if geo is None:
            return 7.0 * max(0.05, float(getattr(self, '_ship_zoom_factor', 1.0)))
        return self._ship_length_px() * 0.5 * geo.height / geo.length

    def _ship_geometry(self):
        """Die gebaute schiffs-grafik, gecacht bis die akzentfarbe wechselt."""
        cache = self._ship_geometry_cache
        if cache is not None and cache.accent == self.ship_accent_color:
            return cache
        try:
            cache = ship_art.build(self.ship_accent_color)
        except Exception as exc:
            print(f"RENDERER WARNING: schiffs-grafik konnte nicht gebaut werden ({exc})")
            self.ship_sprite_enabled = False
            return None
        self._ship_geometry_cache = cache
        return cache

    def _ship_plume_intensity(self, body, real_dt):
        """Helligkeit der abgasfahne, weich zwischen leerlauf und schub.

        `body.last_thrust_direction` wird je frame geleert und von
        `schiffcontrol` gesetzt, sobald schub anliegt -- es ist also ein
        echtes "brennt gerade"-signal. Nur schub NACH VORN zuendet die
        hauptduese: beim rueckwaerts-schub (pfeil ab) sitzen die duesen an
        der nase, hinten glimmt dann nur der leerlauf.
        """
        idle = max(0.0, min(1.0, float(self.ship_plume_idle)))
        target = idle
        thrust = getattr(body, 'last_thrust_direction', None)
        if thrust is not None:
            try:
                # Der vergleich laeuft in WELTkoordinaten: theta und der
                # schubvektor sind beide absolut, die frame-transformierte
                # zeichenrichtung waere hier der falsche massstab.
                theta_world = float(getattr(body, 'theta', 0.0))
                dot = (float(thrust.x) * math.cos(theta_world)
                       - float(thrust.y) * math.sin(theta_world))
                if dot > 0.0:
                    target = 1.0
            except Exception:
                target = 1.0
        # Zeitkonstante ~80 ms, mit dem ECHTEN frame-delta gerechnet, damit
        # das aufflammen bei 30 wie bei 240 fps gleich schnell ist.
        dt = max(0.0, float(real_dt))
        k = 1.0 if dt <= 0.0 else min(1.0, dt / 0.08)
        self._ship_plume_level += (target - self._ship_plume_level) * k
        return self._ship_plume_level

    def _draw_ship_sprite(self, body, x, y, r, g, b, theta_override=None):
        """Das schiff aus `ship_art` zeichnen -- in festen bildschirm-pixeln.

        Die grafik liegt im lokalen schiffsraum vor (+x = nase, +y nach oben,
        einheit "SVG-pixel"). Hier wird sie einmal je frame gedreht, auf die
        gewuenschte bildschirmlaenge skaliert und an die schiffsposition
        geschoben; die batches aus `ship_art` sind nur slices in dieses eine
        transformierte array.
        """
        geo = self._ship_geometry() if self.ship_sprite_enabled else None
        if geo is None:
            self._draw_ship_arrow(body, x, y, r, g, b, theta_override=theta_override)
            return

        theta = float(theta_override) if theta_override is not None else float(getattr(body, 'theta', 0.0))

        # Die grafik laeuft ueber die ORTHO-pipeline (y nach oben), die
        # uebergebene position kommt aber aus _world_to_screen_xy (top-down).
        y = self._ortho_y(y)

        # `theta` ist im UHRZEIGERSINN gemessen: schiff.apply_thrust schiebt
        # entlang Vec2(cos theta, -sin theta), das ist die weltrichtung der
        # nase. Die grafik muss also ebenfalls (cos, -sin) zeigen.
        hx = math.cos(theta)
        hy = -math.sin(theta)

        scale = self._ship_length_px() / geo.length

        # Eine drehmatrix fuer das GANZE array: (x', y') = (hx*x - hy*y,
        # hy*x + hx*y). Rechtshaendig, y zeigt in der ortho-konvention nach
        # oben -- die grafik wird also nicht gespiegelt.
        rot = np.array(((hx, hy), (-hy, hx)), dtype=np.float64)
        pts = geo.verts @ rot
        pts *= scale
        pts[:, 0] += x
        pts[:, 1] += y

        def draw(ops, alpha_gain):
            for mode, rgba, width, start, count in ops:
                alpha = float(rgba[3]) * alpha_gain
                if alpha <= 0.002:
                    continue
                # Die koerperfarbe des schiffs wirkt als tint: bei dem weissen
                # standard-schiff ist das die identitaet, ein eingefaerbtes
                # schiff behaelt aber seine kennfarbe.
                color = (rgba[0] * r, rgba[1] * g, rgba[2] * b, alpha)
                if mode == 'lines':
                    self._draw_ortho_shape(
                        pts[start:start + count], color, moderngl.LINES,
                        width=min(4.0, max(1.0, width * scale)),
                    )
                else:
                    self._draw_ortho_shape(
                        pts[start:start + count], color, moderngl.TRIANGLES,
                    )

        plume = self._ship_plume_intensity(body, getattr(self, '_frame_real_dt', 0.0))
        if plume > 0.0:
            draw(geo.plume_ops, plume)
        draw(geo.ops, 1.0)

        if self.debug_predictor:
            # cyan cross = uebergebene screen-position (= der ursprung der grafik)
            size = 3.0
            self._draw_ortho_shape(
                [(x - size, y), (x + size, y), (x, y - size), (x, y + size)],
                color=(0.0, 1.0, 1.0, 1.0),
                mode=moderngl.LINES,
            )

    def _draw_ship_arrow(self, body, x, y, r, g, b, theta_override=None):
        """Der dreiecks-pfeil.

        Rueckfallweg, wenn `ship_sprite_enabled` aus ist oder `ship_art` sich
        nicht bauen liess -- bis auf die grafik identisch zu
        `_draw_ship_sprite` (gleiche pixel-groesse, gleiche nasenrichtung).
        """
        # in bildschirm-pixeln zeichnen, damit die schiffgröße nicht mit der
        # welt-geometrie skaliert. Die zoom-schrumpfung (siehe
        # _ship_zoom_shrink_factor) gilt hier genauso wie fuer die grafik --
        # sonst waere der fallback weit herausgezoomt ploetzlich der groessere
        # von beiden.
        zoom = max(0.05, float(getattr(self, '_ship_zoom_factor', 1.0)))
        arrow_length = 18.0 * zoom
        arrow_half_width = 7.0 * zoom
        tail_offset = 6.0 * zoom

        theta = float(theta_override) if theta_override is not None else float(getattr(body, 'theta', 0.0))

        # Der pfeil laeuft ueber die ORTHO-pipeline (y nach oben), die
        # uebergebene position kommt aber aus _world_to_screen_xy (top-down).
        y = self._ortho_y(y)

        # `theta` ist im UHRZEIGERSINN gemessen: schiff.apply_thrust schiebt
        # entlang Vec2(cos theta, -sin theta), das ist die weltrichtung der
        # nase. Der pfeil muss also ebenfalls (cos, -sin) zeigen.
        hx = math.cos(theta)
        hy = -math.sin(theta)
        nx = -hy
        ny = hx

        # ursprung anpassen damit der dreiecks-schwerpunkt an (x, y) liegt.
        # der schwerpunkt des dreiecks aus nase und schwanz-ecken liegt
        # entlang der richtung versetzt um (arrow_length - 2*tail_offset)/3
        # in bildschirm-pixeln. verschiebe den lokalen ursprung zurück um diesen
        # betrag damit die welt-position des schiffs dem visuellen mittelpunkt des pfeils entspricht.
        centroid_offset = (arrow_length - 2.0 * tail_offset) / 3.0
        origin_x = x - hx * centroid_offset
        origin_y = y - hy * centroid_offset

        nose_x = origin_x + hx * arrow_length
        nose_y = origin_y + hy * arrow_length
        tail_x = origin_x - hx * tail_offset
        tail_y = origin_y - hy * tail_offset

        left_x = tail_x + nx * arrow_half_width
        left_y = tail_y + ny * arrow_half_width
        right_x = tail_x - nx * arrow_half_width
        right_y = tail_y - ny * arrow_half_width

        self._draw_ortho_shape(
            [(nose_x, nose_y), (left_x, left_y), (right_x, right_y)],
            color=(r, g, b, 1.0),
            mode=moderngl.TRIANGLES,
        )
        # debug: kleine marker zeichnen und einzeilige info ausgeben die
        # den dreiecks-schwerpunkt mit der übergebenen screen-position vergleicht.
        try:
            if self.debug_predictor:
                centroid_x = (nose_x + left_x + right_x) / 3.0
                centroid_y = (nose_y + left_y + right_y) / 3.0
                print(f"PRED_DBG_DRAW: centroid=({centroid_x:.6f},{centroid_y:.6f}) screen_pos=({x:.6f},{y:.6f})")
                # magenta cross = centroid, cyan cross = passed screen pos
                size = 3.0
                self._draw_ortho_shape(
                    [(centroid_x - size, centroid_y), (centroid_x + size, centroid_y),
                     (centroid_x, centroid_y - size), (centroid_x, centroid_y + size)],
                    color=(1.0, 0.0, 1.0, 1.0),
                    mode=moderngl.LINES,
                )
                self._draw_ortho_shape(
                    [(x - size, y), (x + size, y),
                     (x, y - size), (x, y + size)],
                    color=(0.0, 1.0, 1.0, 1.0),
                    mode=moderngl.LINES,
                )
        except Exception:
            pass
