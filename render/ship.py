"""Das zeichnen des schiffs und seiner orientierungsvektoren.

Die vektor-geometrie selbst liegt in `ship/art.py` (reines numpy); hier steht
nur der weg auf den schirm. Die gezeichneten orbital-vektoren sind die EINZIGE
quelle der wahrheit fuer den rast-autopiloten -- siehe
.claude/rules/camera-input.md.
"""
import math
import os

import moderngl

import numpy as np

from physics.reference_frames import apparent_orbital_directions
from ship import art as ship_art


class ShipDrawMixin:
    """Das schiff: sprite, pfeil, fahne, schubvektor und die orientierung.

    `_apply_orientation_snap` laeuft INNERHALB von render(), unmittelbar bevor
    der pfeil gezeichnet wird -- mit demselben rahmen und derselben rahmenzeit
    wie die gezeichneten prograde/normal-vektoren. Nur so landet die nase exakt
    auf dem vektor, unabhaengig von sim_dt und drehrate des rahmens."""

    def draw_ship_thrust_vector(self, ship, camera):
        if ship is None:
            return

        try:
            direction = getattr(ship, "last_thrust_direction", None)
            if direction is None:
                return

            vx = float(direction.x)
            vy = float(direction.y)

            frame = self._active_frame()
            try:
                vx, vy = frame.to_this_frame_vector_xy(self._frame_time_s, vx, vy)
            except Exception:
                pass

            mag = math.hypot(vx, vy)
            if mag <= 1e-12:
                return

            vx /= mag
            vy /= mag

            sx, sy = self._world_to_screen_xy(float(ship.position.x), float(ship.position.y), camera)
            length_px = 45.0
            ex = sx + vx * length_px
            ey = sy - vy * length_px

            self._draw_polyline([(sx, sy), (ex, ey)], color=(1.0, 0.5, 0.1, 0.95), width=2.0)
        except Exception:
            return

    def active_plotting_frame(self):
        """Public accessor for the frame the ship control uses to hold a snap."""
        return self._active_frame()

    def orbital_frame_directions(self, ship, reference_body=None, prediction_points=None):
        """The frame-space orbital directions used to draw the overlay vectors.

        Single source of truth for both the debug vectors and the orientation
        snap: prograde/normal_in are the tangent/inward of the *drawn* predictor
        line in the active plotting frame; retrograde/antinormal are their
        opposites. Evaluated at the renderer's current ``_frame_time_s`` — the
        same instant the ship arrow is drawn.
        """
        frame = self._active_frame()
        if reference_body is None:
            reference_body = getattr(self, "current_reference_body", None)
        ref_pos = getattr(reference_body, "position", None)
        return frame, apparent_orbital_directions(
            frame, self._frame_time_s, ship.position, ship.velocity, ref_pos,
            points=prediction_points,
        )

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

    def draw_ship_orientation_debug_vectors(self, ship, camera, reference_body=None,
                                            prediction_points=None):
        """Debug overlay: always draws prograde (green) + normal-inward (magenta).

        Directions come from ``apparent_orbital_directions`` fed the actual
        predictor polyline (``prediction_points``) — i.e. the tangent of the
        drawn line as it appears in the active plotting frame — so they already
        live in frame space and only need the screen y-flip, exactly like the
        drawn trajectory. This keeps them glued to the predictor line as it
        changes shape, including in rotating/translating frames.
        """
        if ship is None:
            return

        try:
            frame, directions = self.orbital_frame_directions(
                ship, reference_body, prediction_points
            )

            sx, sy = self._world_to_screen_xy(float(ship.position.x), float(ship.position.y), camera)
            length_px = 55.0

            for key, color in (
                ("prograde", (0.2, 1.0, 0.35, 0.95)),
                ("normal_in", (0.9, 0.3, 1.0, 0.95)),
            ):
                d = directions.get(key)
                if d is None:
                    continue
                ex = sx + float(d.x) * length_px
                ey = sy - float(d.y) * length_px
                # Stash the ACTUAL drawn pixel direction of each vector so the
                # diagnostic can compare raw screen geometry (non-derived).
                if key == "normal_in":
                    self._last_normal_screen_dir = (ex - sx, ey - sy)
                elif key == "prograde":
                    self._last_prograde_screen_dir = (ex - sx, ey - sy)
                self._draw_polyline([(sx, sy), (ex, ey)], color=color, width=2.0)

            self._debug_orientation_angles(ship, camera, frame, directions, sx, sy)
        except Exception:
            return

    def _debug_orientation_angles(self, ship, camera, frame, directions, sx, sy):
        """Env-guarded (SPACESIM_DEBUG_ORIENT=1) screen-space angle report.

        Prints, in one common screen convention, the heading of: the actual
        predictor orbit line, my green prograde, the blue velocity vector, and
        the ship nose. Whichever one disagrees is the culprit for the reported
        45 deg offset. Behavior-neutral: only prints, throttled.
        """
        if os.environ.get("SPACESIM_DEBUG_ORIENT", "0").strip().lower() in ("0", "", "false", "off", "no"):
            return
        self._debug_orient_counter = getattr(self, "_debug_orient_counter", 0) + 1
        if self._debug_orient_counter % 30 != 1:
            return

        def sdeg(dx, dy):
            # Screen-space heading: vectors are drawn as (dx, -dy), so the
            # on-screen angle of a frame-space direction is atan2(-dy, dx).
            return math.degrees(math.atan2(-dy, dx))

        parts = []

        # RAW drawn pixel angles (from the actual vertices, y-down screen space).
        def rawdeg(v):
            if v is None:
                return None
            return math.degrees(math.atan2(v[1], v[0]))

        norm_v = getattr(self, "_last_normal_screen_dir", None)
        arrow_v = getattr(self, "_last_arrow_screen_dir", None)
        norm_deg = rawdeg(norm_v)
        # DISPLAYED arrow angle: the arrow renders under gluOrtho2D bottom-up
        # while vectors render via the line shader top-down, so the arrow's
        # on-screen y is the negation of its input y.
        arrow_deg = None if arrow_v is None else math.degrees(math.atan2(-arrow_v[1], arrow_v[0]))

        if norm_deg is not None:
            parts.append(f"magenta_raw={norm_deg:8.3f}")
        if arrow_deg is not None:
            parts.append(f"arrow_raw={arrow_deg:8.3f}")

        # Per-sample rotation of each, so co-rotation vs counter-rotation is
        # directly visible (this is the user's actual complaint).
        prev = getattr(self, "_dbg_prev_raw", None)
        if prev is not None and norm_deg is not None and arrow_deg is not None:
            dmag = (norm_deg - prev[0] + 180.0) % 360.0 - 180.0
            darr = (arrow_deg - prev[1] + 180.0) % 360.0 - 180.0
            sense = "SAME" if (dmag * darr) >= 0 else "OPPOSITE"
            parts.append(f"d_mag={dmag:+7.3f} d_arrow={darr:+7.3f} [{sense}]")
        if norm_deg is not None and arrow_deg is not None:
            self._dbg_prev_raw = (norm_deg, arrow_deg)
            gap = (arrow_deg - norm_deg + 180.0) % 360.0 - 180.0
            parts.append(f"gap={gap:+7.3f}")

        mode = getattr(getattr(self, "_dbg_ship_control", None), "snap_mode", None)
        frame_label = getattr(frame, "label", frame.__class__.__name__)
        print(f"ORIENT_DBG: snap={mode} " + "  ".join(parts) + f"  frame='{frame_label}'")

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

    def _format_speed_label(self, speed_m_s):
        if speed_m_s is None:
            return ""

        speed_m_s = float(speed_m_s)
        if speed_m_s >= 1000.0:
            return f"{speed_m_s / 1000.0:.2f} km/s"

        return f"{speed_m_s:.1f} m/s"

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
            # alte feste groesse als eine division durch null.
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
        auf die halbe breite des alten pfeils zurueck, wenn die grafik aus ist.
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

        `body.last_thrust_direction` wird in test.py je frame geleert und von
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
        # Ohne diese umrechnung landet das schiff an der ueber die
        # bildschirmmitte gespiegelten stelle -- exakt mittig faellt das nicht
        # auf, abseits der mitte steht es weit neben seiner bahn.
        y = self._ortho_y(y)

        # `theta` ist im UHRZEIGERSINN gemessen: schiff.apply_thrust schiebt
        # entlang Vec2(cos theta, -sin theta), das ist die weltrichtung der
        # nase. Die grafik muss also ebenfalls (cos, -sin) zeigen.
        hx = math.cos(theta)
        hy = -math.sin(theta)
        # Stash the ACTUAL drawn nose screen-direction so diagnostics can compare
        # the real ship pixels against the drawn vectors (non-circular check).
        self._last_arrow_screen_dir = (hx, hy)

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
        """Der alte dreiecks-pfeil.

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
        # Ohne diese umrechnung wird der pfeil an der ueber die bildschirmmitte
        # gespiegelten stelle gezeichnet: exakt mittig faellt das nicht auf,
        # abseits der mitte steht das schiff weit neben seiner bahn.
        y = self._ortho_y(y)

        # `theta` ist im UHRZEIGERSINN gemessen: schiff.apply_thrust schiebt
        # entlang Vec2(cos theta, -sin theta), das ist die weltrichtung der
        # nase. Der pfeil muss also ebenfalls (cos, -sin) zeigen.
        # Die positions-korrektur oben aendert daran nichts -- eine
        # verschiebung dreht keine richtung.
        hx = math.cos(theta)
        hy = -math.sin(theta)
        nx = -hy
        ny = hx
        # Stash the ACTUAL drawn nose screen-direction so diagnostics can compare
        # the real arrow pixels against the drawn vectors (non-circular check).
        self._last_arrow_screen_dir = (hx, hy)

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
