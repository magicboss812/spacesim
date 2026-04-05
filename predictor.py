# predictor.py

from vec import Vec2
import math
import time
import poliastro
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numba import njit

# Predictor is intentionally Numba-only in this workspace.
NUMBA_AVAILABLE = True


if NUMBA_AVAILABLE:
    @njit(cache=True, nogil=True, fastmath=True)
    def _compute_acc_numba(x, y, body_x, body_y, body_m, body_fixed, G):
        ax = 0.0
        ay = 0.0
        for i in range(body_x.shape[0]):
            if body_fixed[i] == 0:
                continue
            dx = body_x[i] - x
            dy = body_y[i] - y
            dist2 = dx * dx + dy * dy
            if dist2 < 1e-12:
                continue
            invd = 1.0 / math.sqrt(dist2)
            accm = G * body_m[i] / dist2
            ax += dx * invd * accm
            ay += dy * invd * accm
        return ax, ay


    @njit(cache=True, nogil=True, fastmath=True)
    def _compute_distance_points_numba(
        init_px,
        init_py,
        init_vx,
        init_vy,
        body_x,
        body_y,
        body_m,
        body_fixed,
        G,
        dt,
        precision,
        max_points,
        max_iters,
    ):
        out = np.empty((max_points, 3), dtype=np.float64)
        out[0, 0] = init_px
        out[0, 1] = init_py
        out[0, 2] = 0.0

        count = 1
        px = init_px
        py = init_py
        vx = init_vx
        vy = init_vy
        accumulated = 0.0

        t = 0.0
        for _ in range(max_iters):
            if count >= max_points:
                break

            k1_ax, k1_ay = _compute_acc_numba(px, py, body_x, body_y, body_m, body_fixed, G)
            k1_vx, k1_vy = vx, vy

            p2x = px + k1_vx * (dt / 2.0)
            p2y = py + k1_vy * (dt / 2.0)
            v2x = vx + k1_ax * (dt / 2.0)
            v2y = vy + k1_ay * (dt / 2.0)
            k2_ax, k2_ay = _compute_acc_numba(p2x, p2y, body_x, body_y, body_m, body_fixed, G)
            k2_vx, k2_vy = v2x, v2y

            p3x = px + k2_vx * (dt / 2.0)
            p3y = py + k2_vy * (dt / 2.0)
            v3x = vx + k2_ax * (dt / 2.0)
            v3y = vy + k2_ay * (dt / 2.0)
            k3_ax, k3_ay = _compute_acc_numba(p3x, p3y, body_x, body_y, body_m, body_fixed, G)
            k3_vx, k3_vy = v3x, v3y

            p4x = px + k3_vx * dt
            p4y = py + k3_vy * dt
            v4x = vx + k3_ax * dt
            v4y = vy + k3_ay * dt
            k4_ax, k4_ay = _compute_acc_numba(p4x, p4y, body_x, body_y, body_m, body_fixed, G)
            k4_vx, k4_vy = v4x, v4y

            next_px = px + (k1_vx + 2.0 * k2_vx + 2.0 * k3_vx + k4_vx) * (dt / 6.0)
            next_py = py + (k1_vy + 2.0 * k2_vy + 2.0 * k3_vy + k4_vy) * (dt / 6.0)
            next_vx = vx + (k1_ax + 2.0 * k2_ax + 2.0 * k3_ax + k4_ax) * (dt / 6.0)
            next_vy = vy + (k1_ay + 2.0 * k2_ay + 2.0 * k3_ay + k4_ay) * (dt / 6.0)

            seg_dx = next_px - px
            seg_dy = next_py - py
            seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)

            if seg_len <= 0.0:
                px = next_px
                py = next_py
                vx = next_vx
                vy = next_vy
                continue

            local_px = px
            local_py = py
            local_vx = vx
            local_vy = vy
            rem_dx = seg_dx
            rem_dy = seg_dy
            rem_len = seg_len

            while rem_len + accumulated >= precision and count < max_points:
                if rem_len <= 0.0:
                    break

                distance_to_place = precision - accumulated
                frac = distance_to_place / rem_len

                sample_px = local_px + rem_dx * frac
                sample_py = local_py + rem_dy * frac
                sample_t = t + frac * dt

                out[count, 0] = sample_px
                out[count, 1] = sample_py
                out[count, 2] = sample_t
                count += 1

                local_px = sample_px
                local_py = sample_py

                rem_dx = next_px - local_px
                rem_dy = next_py - local_py
                rem_len = math.sqrt(rem_dx * rem_dx + rem_dy * rem_dy)
                accumulated = 0.0

            if rem_len + accumulated < precision:
                accumulated += rem_len

            px = next_px
            py = next_py
            vx = next_vx
            vy = next_vy
            t += dt

        return out, count
class Predictor:
    def __init__(
        self,
        num_points=5000,
        dt=60.0,
        workers=None,
        debug=True,
        recompute_every_update=True,
        precision=1e6,
        length=None,
        use_numba=True,
        async_compute=True,
    ):
        # num_points: hard cap on returned points
        # dt: internal integration timestep (seconds)
        # precision: desired spacing between successive predictor points (meters)
        # length: optional total distance horizon (meters). If None, num_points is used as cap.
        self.num_points = int(num_points)
        self.dt = float(dt)
        self.precision = float(precision)
        # base_precision is the reference spacing used to compute point budget when length is set
        self.base_precision = float(precision)
        self.length = None if length is None else float(length)

        self.points = np.empty((0, 3), dtype=np.float64) if np is not None else []
        self.debug = debug
        self.initialized = False
        # Keep recompute cadence frame-based so it is independent from sim timestep.
        self.recompute_every_update = recompute_every_update

        self.workers = 7 if workers is None else int(workers)
        self.use_numba = True

        # Optional zoom-aware precision: when zooming in, use finer spacing
        # so rendering can show more local detail without manual key presses.
        self.auto_precision_from_zoom = True
        self.target_screen_step_px = 2.0
        self.min_precision = 1.0
        self._view_scale = None

        self.async_compute = bool(async_compute)
        self._executor = None
        self._pending_future = None
        self._pending_job_id = 0
        self._next_job_id = 1
        self._last_swapped_job_id = 0
        self._jobs_submitted = 0
        self._jobs_swapped = 0
        
        # Metadata about the last swapped async result (snapshot dict)
        self._last_swapped_snapshot = None

        # Tolerances for detecting stale async snapshots. If the speed
        # difference between the snapshot and the current ship exceeds
        # max(abs_tol, rel_tol * current_speed) the snapshot is considered stale.
        self.snapshot_velocity_rel_tol = 0.01
        self.snapshot_velocity_abs_tol = 100.0
        # Position-based staleness threshold (meters). If the distance
        # between snapshot ship position and current ship position exceeds
        # this (or the velocity*age allowance below), the snapshot is stale.
        self.snapshot_position_abs_tol = 1000.0

        # If True, force a synchronous recompute when a stale snapshot
        # is detected (may cause frame hitching). Default: False.
        self.force_sync_on_stale = False

        # View-change cooldown: while zoom is changing, avoid submitting or
        # swapping predictor results to prevent visual jumps. Seconds.
        # Default to 0.0 so we don't artificially delay submitting new
        # computations after a zoom; use view-scale checks at swap time
        # instead to reject mismatched results.
        self.view_change_cooldown = 0.0
        self._view_change_cooldown_until = 0.0
        # Relative tolerance for considering a snapshot computed at a
        # different view scale as stale. Tighten this to avoid accepting
        # results computed for a slightly different zoom level.
        self.snapshot_view_rel_tol = 1e-6

        # Internal flag set when view-scale changes significantly. When True
        # `update()` will perform a synchronous recompute immediately so the
        # predictor matches the new zoom on the same frame.
        self._view_scale_changed = False

        if self.async_compute:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="predictor")

    def reset(self):
        self._cancel_pending_job()
        self.points = np.empty((0, 3), dtype=np.float64) if np is not None else []
        self.initialized = False

    def _points_count(self):
        if np is not None and isinstance(self.points, np.ndarray):
            return int(self.points.shape[0])
        return len(self.points)

    def _anchor_first_point(self, ship):
        if self._points_count() == 0:
            return
        sx = float(ship.position.x)
        sy = float(ship.position.y)
        if np is not None and isinstance(self.points, np.ndarray):
            self.points[0, 0] = sx
            self.points[0, 1] = sy
        else:
            # Preserve timestamp if present on the first point
            try:
                t0 = float(self.points[0][2])
            except Exception:
                t0 = 0.0
            self.points[0] = (sx, sy, t0)

    def set_view_scale(self, scale: float):
        try:
            scale = float(scale)
        except Exception:
            return
        if scale > 0.0:
            old = self._view_scale

            # If the scale hasn't changed meaningfully, do nothing.
            # This avoids resetting the short view-change cooldown every frame
            # when `set_view_scale` is called repeatedly with the same value.
            if old is not None:
                try:
                    rel_change = abs(scale - old) / max(abs(old), 1e-30)
                except Exception:
                    rel_change = 0.0
                if rel_change <= self.snapshot_view_rel_tol:
                    return

            # Apply the new scale and emit debug info for real changes.
            self._view_scale = scale
            # Flag that the view scale changed so the next update() call
            # can synchronously recompute immediately (same-frame fix).
            try:
                self._view_scale_changed = True
                if self.debug:
                    print("PRED_DBG_VIEW_CHANGED: flagged for sync recompute")
            except Exception:
                pass
            if self.debug:
                try:
                    eff = self._effective_precision()
                except Exception:
                    eff = self.precision
                print(f"PRED_DBG_VIEW_SCALE: old={old} new={self._view_scale} eff_precision={eff}")

            # If zoom changed significantly, cancel any pending async job so
            # we don't accept results computed with the previous precision.
            try:
                if old is not None and self.async_compute and self._pending_future is not None:
                    rel = abs(self._view_scale - old) / max(abs(old), 1e-30)
                    if rel > 0.02:
                        if self.debug:
                            print(f"PRED_DBG_CANCEL_PENDING: zoom rel_change={rel:.3f} canceling pending job")
                        self._cancel_pending_job()
            except Exception:
                pass

            # Start a short cooldown during which submissions/swaps are skipped
            # to avoid visual jumps while the user is actively zooming.
            try:
                self._view_change_cooldown_until = time.time() + float(self.view_change_cooldown)
                if self.debug:
                    print(f"PRED_DBG_VIEW_COOLDOWN: until={self._view_change_cooldown_until:.6f}")
            except Exception:
                pass

    def _effective_precision(self):
        effective = float(self.precision)
        if self.auto_precision_from_zoom and self._view_scale is not None:
            zoom_precision = self.target_screen_step_px / max(self._view_scale, 1e-30)
            effective = min(effective, max(self.min_precision, zoom_precision))
        return effective

    def _serialize_bodies(self, world):
        lst = []
        for b in world.body:
            # Include all bodies as gravitational sources (use current positions)
            lst.append((b.position.x, b.position.y, b.mass, True))
        return lst

    def _serialize_bodies_numba(self, world):
        count = len(world.body)
        body_x = np.empty(count, dtype=np.float64)
        body_y = np.empty(count, dtype=np.float64)
        body_m = np.empty(count, dtype=np.float64)
        body_fixed = np.empty(count, dtype=np.uint8)
        for i, b in enumerate(world.body):
            body_x[i] = float(b.position.x)
            body_y[i] = float(b.position.y)
            body_m[i] = float(b.mass)
            # Treat all bodies as gravitational sources for predictor
            body_fixed[i] = 1
        return body_x, body_y, body_m, body_fixed

    def _compute_acc(self, position, bodies, G):
        total = Vec2(0.0, 0.0)
        for bx, by, mass, fixed in bodies:
            if not fixed:
                continue
            dirv = Vec2(bx, by) - position
            dist2 = dirv.magnitude_squared()
            if dist2 < 1e-12:
                continue
            invd = 1.0 / math.sqrt(dist2)
            fdir = dirv * invd
            accm = G * mass / dist2
            total += fdir * accm
        return total

    def _rk4_step(self, p, v, bodies, G):
        dt = self.dt

        k1_a = self._compute_acc(p, bodies, G)
        k1_v = v

        p2 = p + k1_v * (dt / 2)
        v2 = v + k1_a * (dt / 2)
        k2_a = self._compute_acc(p2, bodies, G)
        k2_v = v2

        p3 = p + k2_v * (dt / 2)
        v3 = v + k2_a * (dt / 2)
        k3_a = self._compute_acc(p3, bodies, G)
        k3_v = v3

        p4 = p + k3_v * dt
        v4 = v + k3_a * dt
        k4_a = self._compute_acc(p4, bodies, G)
        k4_v = v4

        new_p = p + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * (dt / 6)
        new_v = v + (k1_a + 2 * k2_a + 2 * k3_a + k4_a) * (dt / 6)

        return new_p, new_v

    def _compute_full_python(self, ship, world, max_points):
        bodies = self._serialize_bodies(world)
        G = world.G

        pos = ship.position.copy()
        vel = ship.velocity.copy()
        pts = [pos.copy()]

        accumulated = 0.0

        safety_iters = max(100000, max_points * 100)
        iters = 0

        while len(pts) < max_points and iters < safety_iters:
            iters += 1
            next_pos, next_vel = self._rk4_step(pos, vel, bodies, G)

            seg_vec = next_pos - pos
            seg_len = seg_vec.magnitude()

            if seg_len <= 0:
                pos = next_pos
                vel = next_vel
                continue

            local_pos = pos
            local_vel = vel
            remaining_vec = seg_vec
            remaining_len = seg_len

            while remaining_len + accumulated >= self.precision and len(pts) < max_points:
                if remaining_len <= 0:
                    break

                distance_to_place = self.precision - accumulated
                fraction = distance_to_place / remaining_len

                sample_point = local_pos + remaining_vec * fraction
                sample_vel = local_vel + (next_vel - local_vel) * fraction

                pts.append(sample_point.copy())

                local_pos = sample_point
                local_vel = sample_vel

                remaining_vec = next_pos - local_pos
                remaining_len = remaining_vec.magnitude()
                accumulated = 0.0

            if remaining_len + accumulated < self.precision:
                accumulated += remaining_len

            pos = next_pos
            vel = next_vel

        return pts

    def _cancel_pending_job(self):
        if self._pending_future is None:
            return
        if not self._pending_future.done():
            self._pending_future.cancel()
        self._pending_future = None
        self._pending_job_id = 0

    def _make_snapshot(self, ship, world, max_points):
        effective_precision = self._effective_precision()
        snapshot = {
            "ship_px": float(ship.position.x),
            "ship_py": float(ship.position.y),
            "ship_vx": float(ship.velocity.x),
            "ship_vy": float(ship.velocity.y),
            "G": float(world.G),
            "dt": float(self.dt),
            "precision": float(effective_precision),
            "max_points": int(max_points),
            "max_iters": int(max(10000, max_points * 100)),
            "numba": True,
        }
        # include simulation time when available for staleness diagnostics
        # Record both simulation time and wall-clock submit timestamp.
        try:
            snapshot["sim_time"] = float(world.time)
        except Exception:
            snapshot["sim_time"] = 0.0
        try:
            snapshot["submit_ts"] = float(time.time())
        except Exception:
            snapshot["submit_ts"] = 0.0
        # Record the view scale used when creating this snapshot so that
        # results computed for a different zoom level can be considered stale.
        try:
            snapshot["view_scale"] = float(self._view_scale) if self._view_scale is not None else None
        except Exception:
            snapshot["view_scale"] = None
        body_x, body_y, body_m, body_fixed = self._serialize_bodies_numba(world)
        snapshot["body_x"] = body_x
        snapshot["body_y"] = body_y
        snapshot["body_m"] = body_m
        snapshot["body_fixed"] = body_fixed
        return snapshot

    def _compute_from_snapshot(self, snapshot):
        out, used = _compute_distance_points_numba(
            snapshot["ship_px"],
            snapshot["ship_py"],
            snapshot["ship_vx"],
            snapshot["ship_vy"],
            snapshot["body_x"],
            snapshot["body_y"],
            snapshot["body_m"],
            snapshot["body_fixed"],
            snapshot["G"],
            snapshot["dt"],
            snapshot["precision"],
            snapshot["max_points"],
            snapshot["max_iters"],
        )
        points = out[:int(used)].copy()

        # Convert per-sample times (relative to snapshot) to absolute sim time
        try:
            base_sim_time = float(snapshot.get("sim_time", 0.0)) if snapshot is not None else 0.0
        except Exception:
            base_sim_time = 0.0

        try:
            if np is not None and isinstance(points, np.ndarray) and points.shape[1] >= 3:
                points = points.copy()
                points[:, 2] = points[:, 2] + base_sim_time
            else:
                # list of triples
                pts = []
                for p in points:
                    try:
                        pts.append((float(p[0]), float(p[1]), float(p[2]) + base_sim_time))
                    except Exception:
                        pts.append((float(p[0]), float(p[1]), base_sim_time))
                points = pts
        except Exception:
            pass

        return {"points": points, "snapshot": snapshot}

    def _submit_async_compute(self, ship, world, max_points):
        if self._executor is None:
            return
        # Don't submit new compute jobs while view-change cooldown is active.
        try:
            if time.time() < getattr(self, '_view_change_cooldown_until', 0.0):
                if self.debug:
                    print("PRED_DBG_SUBMIT_SKIPPED: view-change cooldown active")
                return
        except Exception:
            pass
        snapshot = self._make_snapshot(ship, world, max_points)
        if self.debug:
            print(
                f"PRED_DBG_SUBMIT: job={self._next_job_id} submit_ts={snapshot.get('submit_ts',0.0):.6f} "
                f"view_scale={snapshot.get('view_scale',None)} "
                f"ship_v=({snapshot['ship_vx']:.3f},{snapshot['ship_vy']:.3f}) max_pts={snapshot['max_points']}"
            )
        self._pending_future = self._executor.submit(self._compute_from_snapshot, snapshot)
        self._pending_job_id = self._next_job_id
        self._next_job_id += 1
        self._jobs_submitted += 1

    def _swap_ready_result(self, current_ship=None, current_world=None):
        if self._pending_future is None or not self._pending_future.done():
            return False

        finished_future = self._pending_future
        finished_job_id = self._pending_job_id
        self._pending_future = None
        self._pending_job_id = 0

        try:
            result = finished_future.result()

            # Normalize result: {"points": ..., "snapshot": ...}
            if isinstance(result, dict):
                points = result.get("points")
                snapshot = result.get("snapshot")
            else:
                points = result
                snapshot = None

            # If we have both a snapshot and the current ship, check
            # whether the snapshot's velocity or position deviates
            # significantly from the current ship velocity/position.
            # Additionally treat snapshots computed for a different
            # view scale as stale.
            if snapshot is not None and current_ship is not None:
                svx = float(snapshot.get("ship_vx", 0.0))
                svy = float(snapshot.get("ship_vy", 0.0))
                cur_vx = float(current_ship.velocity.x)
                cur_vy = float(current_ship.velocity.y)
                # Use vector difference (magnitude) to capture direction+speed changes
                dvx = cur_vx - svx
                dvy = cur_vy - svy
                delta_speed = math.hypot(dvx, dvy)
                cur_speed = math.hypot(cur_vx, cur_vy)
                allowed_speed = max(self.snapshot_velocity_abs_tol, self.snapshot_velocity_rel_tol * max(cur_speed, 1.0))

                # Position-based staleness: compare snapshot ship pos vs current
                spx = float(snapshot.get("ship_px", 0.0))
                spy = float(snapshot.get("ship_py", 0.0))
                cur_px = float(current_ship.position.x)
                cur_py = float(current_ship.position.y)
                pos_delta = math.hypot(cur_px - spx, cur_py - spy)

                # Age in wall-clock seconds since submission
                submit_ts = float(snapshot.get("submit_ts", 0.0))
                age = max(0.0, time.time() - submit_ts)
                allowed_pos = max(self.snapshot_position_abs_tol, self.snapshot_velocity_abs_tol * max(0.1, age))

                # View-scale staleness: if snapshot was computed for a
                # different view scale and the difference exceeds tol,
                # consider it stale due to zoom change.
                snap_view = snapshot.get("view_scale", None)
                is_stale_view = False
                try:
                    if snap_view is not None and self._view_scale is not None:
                        rel_view = abs(snap_view - self._view_scale) / max(abs(self._view_scale), 1e-30)
                        if rel_view > float(self.snapshot_view_rel_tol):
                            is_stale_view = True
                except Exception:
                    is_stale_view = False

                is_stale_speed = delta_speed > allowed_speed
                is_stale_pos = pos_delta > allowed_pos

                if is_stale_view or is_stale_speed or is_stale_pos:
                    if self.debug:
                        print(
                            f"PRED_DBG_STALE: job={finished_job_id} submit_ts={submit_ts:.6f} "
                            f"dv=({dvx:.3f},{dvy:.3f}) |dv|={delta_speed:.3f} allowed_speed={allowed_speed:.3f} "
                            f"pos_delta={pos_delta:.3f} allowed_pos={allowed_pos:.3f} age={age:.3f} view_stale={is_stale_view}"
                        )
                    # If staleness is caused by a view/zoom change, just discard
                    # the result and do NOT force a synchronous recompute; a
                    # fresh job will be (re)submitted after the view cooldown.
                    if is_stale_view:
                        if self.debug:
                            print(f"PRED_DBG_STALE_VIEW: job={finished_job_id} snap_view={snap_view} cur_view={self._view_scale}")
                        return False

                    # For dynamics-based staleness (velocity/position), optionally
                    # force a synchronous recompute to get an up-to-date curve.
                    if self.force_sync_on_stale and current_world is not None:
                        if self.debug:
                            print("PREDICTOR: forcing synchronous recompute due to stale snapshot (dynamics)")
                        self._compute_full(current_ship, current_world)
                        self._anchor_first_point(current_ship)
                        self._last_swapped_job_id = finished_job_id
                        self._jobs_swapped += 1
                        return True
                    else:
                        if self.debug:
                            print("PREDICTOR: discarding stale async snapshot (dynamics)")
                        return False

            # Accept result
            self.points = points
            self.initialized = True
            self._last_swapped_job_id = finished_job_id
            self._jobs_swapped += 1
            self._last_swapped_snapshot = snapshot
            if self.debug:
                try:
                    cnt = points.shape[0] if (np is not None and hasattr(points, "shape")) else len(points)
                except Exception:
                    cnt = 0
                if snapshot is not None:
                    svx = float(snapshot.get("ship_vx", 0.0))
                    svy = float(snapshot.get("ship_vy", 0.0))
                    stime = snapshot.get("time", 0.0)
                    print(
                        f"PRED_DBG_SWAP: job={finished_job_id} time={stime:.6f} ship_v=({svx:.3f},{svy:.3f}) pts={cnt}"
                    )
            return True
        except Exception as exc:
            if self.debug:
                print(f"PREDICTOR: async compute failed ({exc})")
            return False

    def _get_target_point_cap(self):
        """Return desired maximum number of points determined by `length`.

        Important: this cap is computed from `self.length` and the initial
        `base_precision` so that changing `precision` later does not change
        the target number of points while `length` is set.
        """
        if self.num_points <= 0:
            return 0

        if self.length is None:
            return self.num_points

        spacing_for_cap = self.base_precision if self.base_precision > 0.0 else self.precision
        max_by_length = max(1, int(self.length / spacing_for_cap) + 1)
        return min(self.num_points, max_by_length)

    def _compute_full(self, ship, world):
        if self.num_points <= 0:
            self.points = np.empty((0, 2), dtype=np.float64) if np is not None else []
            self.initialized = True
            return

        if self.precision <= 0.0:
            raise ValueError("Predictor precision must be > 0")

        max_points = self._get_target_point_cap()

        snapshot = self._make_snapshot(ship, world, max_points)
        result = self._compute_from_snapshot(snapshot)
        if isinstance(result, dict):
            self.points = result["points"]
            self._last_swapped_snapshot = result.get("snapshot")
        else:
            self.points = result
        self.initialized = True

    def initialize(self, ship, world):
        self.reset()
        self._compute_full(ship, world)
        self._anchor_first_point(ship)

    def update(self, ship, world):
        if self.num_points <= 0:
            self.reset()
            return

        if self.precision <= 0.0:
            raise ValueError("Predictor precision must be > 0")

        if not self.async_compute:
            if not self.initialized:
                self.initialize(ship, world)
                return

            if self.recompute_every_update:
                if self.debug:
                    print(
                        f"PREDICTOR_UPDATE: recomputing full predictor "
                        f"num_points={self.num_points} current_points={self._points_count()} removed=0 mode=frame"
                    )
                self._compute_full(ship, world)
                self._anchor_first_point(ship)
                return

            removed = self.remove_passed_points(ship)
            target_points = self._get_target_point_cap()
            if self._points_count() < target_points:
                if self.debug:
                    print(
                        f"PREDICTOR_UPDATE: recomputing full predictor "
                        f"target_points={target_points} current_points={self._points_count()} removed={removed}"
                    )
                self._compute_full(ship, world)
            self._anchor_first_point(ship)
            return

        # If the view-scale just changed, perform a synchronous recompute
        # now so the predictor matches the new zoom on the same frame.
        if getattr(self, '_view_scale_changed', False):
            if ship is not None and world is not None:
                if self.debug:
                    print("PRED_DBG_SYNC_RECOMPUTE: view-scale changed, computing synchronously now")
                # Cancel any pending async job and compute immediately.
                self._cancel_pending_job()
                self._compute_full(ship, world)
                self._anchor_first_point(ship)
                self._view_scale_changed = False
                return

        swapped = self._swap_ready_result(ship, world)
        target_points = self._get_target_point_cap()

        if not self.initialized and self._pending_future is None:
            self._submit_async_compute(ship, world, target_points)
            return

        if not self.recompute_every_update:
            self.remove_passed_points(ship)

        if self._pending_future is None:
            if self.recompute_every_update or self._points_count() < target_points or swapped:
                self._submit_async_compute(ship, world, target_points)

        if self.initialized:
            self._anchor_first_point(ship)

    def get_points(self):
        return self.points

    def get_precision_factor(self):
        if self.base_precision <= 0.0:
            return 1.0
        return self.precision / self.base_precision

    def get_display_length(self):
        """HUD-only display length: scale `length` by precision factor for user feedback.

        This value is not used in computation; it just communicates how spacing
        affects the visualized predictor length.
        """
        if self.length is None:
            return None
        return self.length * self.get_precision_factor()

    def set_precision(self, meters: float):
        meters = float(meters)
        if meters <= 0.0:
            raise ValueError("precision must be > 0")
        self.precision = meters
        if self.async_compute:
            self._cancel_pending_job()

    def set_length(self, meters: float | None):
        if meters is None:
            self.length = None
            if self.async_compute:
                self._cancel_pending_job()
            return
        meters = float(meters)
        if meters <= 0.0:
            raise ValueError("length must be > 0")
        self.length = meters
        if self.async_compute:
            self._cancel_pending_job()

    def set_num_points(self, count: int):
        self.num_points = max(0, int(count))
        self.reset()

    def advance_state(self, world=None):
        # Compatibility hook: in async mode this opportunistically swaps in
        # finished results without forcing a new computation.
        if self.async_compute:
            self._swap_ready_result(None, world)

    def get_async_status(self):
        return {
            "enabled": self.async_compute,
            "pending": self._pending_future is not None,
            "submitted_jobs": self._jobs_submitted,
            "swapped_jobs": self._jobs_swapped,
            "last_swapped_job_id": self._last_swapped_job_id,
            "effective_precision": self._effective_precision(),
        }

    def close(self):
        self._cancel_pending_job()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def remove_passed_points(self, ship):
        if self._points_count() < 2:
            return 0

        removed = 0
        sx = float(ship.position.x)
        sy = float(ship.position.y)

        if np is not None and isinstance(self.points, np.ndarray):
            while self.points.shape[0] > 1:
                dx0 = self.points[0, 0] - sx
                dy0 = self.points[0, 1] - sy
                d0 = dx0 * dx0 + dy0 * dy0
                dx1 = self.points[1, 0] - sx
                dy1 = self.points[1, 1] - sy
                d1 = dx1 * dx1 + dy1 * dy1
                if d1 < d0:
                    self.points = self.points[1:]
                    removed += 1
                else:
                    break
        else:
            while len(self.points) > 1:
                try:
                    p0 = self.points[0]
                    p1 = self.points[1]
                    # handle triples (x,y,t) or Vec2
                    if hasattr(p0, 'distance_squared_to'):
                        d0 = p0.distance_squared_to(ship.position)
                        d1 = p1.distance_squared_to(ship.position)
                    else:
                        x0 = float(p0[0]); y0 = float(p0[1])
                        x1 = float(p1[0]); y1 = float(p1[1])
                        dx0 = x0 - sx
                        dy0 = y0 - sy
                        dx1 = x1 - sx
                        dy1 = y1 - sy
                        d0 = dx0 * dx0 + dy0 * dy0
                        d1 = dx1 * dx1 + dy1 * dy1

                    if d1 < d0:
                        self.points.pop(0)
                        removed += 1
                    else:
                        break
                except Exception:
                    break

        return removed
