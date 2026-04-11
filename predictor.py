# predictor.py

from vec import Vec2
import math
import time
import poliastro
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numba import njit
# Predictor ist absichtlich Numba-only
NUMBA_AVAILABLE = True


if NUMBA_AVAILABLE:
    @njit(cache=True, nogil=True, fastmath=True, parallel=True)
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
        ref_enabled,
        ref_px,
        ref_py,
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

            ref_ax = 0.0
            ref_ay = 0.0
            if ref_enabled != 0:
                ref_ax, ref_ay = _compute_acc_numba(ref_px, ref_py, body_x, body_y, body_m, body_fixed, G)

            k1_ax_raw, k1_ay_raw = _compute_acc_numba(px, py, body_x, body_y, body_m, body_fixed, G)
            k1_ax = k1_ax_raw - ref_ax
            k1_ay = k1_ay_raw - ref_ay
            k1_vx, k1_vy = vx, vy

            p2x = px + k1_vx * (dt / 2.0)
            p2y = py + k1_vy * (dt / 2.0)
            v2x = vx + k1_ax * (dt / 2.0)
            v2y = vy + k1_ay * (dt / 2.0)
            k2_ax_raw, k2_ay_raw = _compute_acc_numba(p2x, p2y, body_x, body_y, body_m, body_fixed, G)
            k2_ax = k2_ax_raw - ref_ax
            k2_ay = k2_ay_raw - ref_ay
            k2_vx, k2_vy = v2x, v2y

            p3x = px + k2_vx * (dt / 2.0)
            p3y = py + k2_vy * (dt / 2.0)
            v3x = vx + k2_ax * (dt / 2.0)
            v3y = vy + k2_ay * (dt / 2.0)
            k3_ax_raw, k3_ay_raw = _compute_acc_numba(p3x, p3y, body_x, body_y, body_m, body_fixed, G)
            k3_ax = k3_ax_raw - ref_ax
            k3_ay = k3_ay_raw - ref_ay
            k3_vx, k3_vy = v3x, v3y

            p4x = px + k3_vx * dt
            p4y = py + k3_vy * dt
            v4x = vx + k3_ax * dt
            v4y = vy + k3_ay * dt
            k4_ax_raw, k4_ay_raw = _compute_acc_numba(p4x, p4y, body_x, body_y, body_m, body_fixed, G)
            k4_ax = k4_ax_raw - ref_ax
            k4_ay = k4_ay_raw - ref_ay
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


    @njit(cache=True, nogil=True, fastmath=True)
    def _compute_distance_points_numba_state(
        init_px,
        init_py,
        init_vx,
        init_vy,
        init_t,
        ref_enabled,
        ref_px,
        ref_py,
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
        # spalten: x, y, t, vx, vy
        out = np.empty((max_points, 5), dtype=np.float64)
        out[0, 0] = init_px
        out[0, 1] = init_py
        out[0, 2] = init_t
        out[0, 3] = init_vx
        out[0, 4] = init_vy

        count = 1
        px = init_px
        py = init_py
        vx = init_vx
        vy = init_vy
        accumulated = 0.0

        t = init_t

        for _ in range(max_iters):
            if count >= max_points:
                break

            ref_ax = 0.0
            ref_ay = 0.0
            if ref_enabled != 0:
                ref_ax, ref_ay = _compute_acc_numba(ref_px, ref_py, body_x, body_y, body_m, body_fixed, G)

            k1_ax_raw, k1_ay_raw = _compute_acc_numba(px, py, body_x, body_y, body_m, body_fixed, G)
            k1_ax = k1_ax_raw - ref_ax
            k1_ay = k1_ay_raw - ref_ay
            k1_vx, k1_vy = vx, vy

            p2x = px + k1_vx * (dt / 2.0)
            p2y = py + k1_vy * (dt / 2.0)
            v2x = vx + k1_ax * (dt / 2.0)
            v2y = vy + k1_ay * (dt / 2.0)
            k2_ax_raw, k2_ay_raw = _compute_acc_numba(p2x, p2y, body_x, body_y, body_m, body_fixed, G)
            k2_ax = k2_ax_raw - ref_ax
            k2_ay = k2_ay_raw - ref_ay
            k2_vx, k2_vy = v2x, v2y

            p3x = px + k2_vx * (dt / 2.0)
            p3y = py + k2_vy * (dt / 2.0)
            v3x = vx + k2_ax * (dt / 2.0)
            v3y = vy + k2_ay * (dt / 2.0)
            k3_ax_raw, k3_ay_raw = _compute_acc_numba(p3x, p3y, body_x, body_y, body_m, body_fixed, G)
            k3_ax = k3_ax_raw - ref_ax
            k3_ay = k3_ay_raw - ref_ay
            k3_vx, k3_vy = v3x, v3y

            p4x = px + k3_vx * dt
            p4y = py + k3_vy * dt
            v4x = vx + k3_ax * dt
            v4y = vy + k3_ay * dt
            k4_ax_raw, k4_ay_raw = _compute_acc_numba(p4x, p4y, body_x, body_y, body_m, body_fixed, G)
            k4_ax = k4_ax_raw - ref_ax
            k4_ay = k4_ay_raw - ref_ay
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
                t += dt
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
                sample_vx = local_vx + (next_vx - local_vx) * frac
                sample_vy = local_vy + (next_vy - local_vy) * frac

                out[count, 0] = sample_px
                out[count, 1] = sample_py
                out[count, 2] = sample_t
                out[count, 3] = sample_vx
                out[count, 4] = sample_vy
                count += 1

                local_px = sample_px
                local_py = sample_py
                local_vx = sample_vx
                local_vy = sample_vy

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
        
        self.num_points = int(num_points)
        self.dt = float(dt)
        self.precision = float(precision)
        self.base_precision = float(precision)
        self.length = None if length is None else float(length)

        self.points = np.empty((0, 3), dtype=np.float64) if np is not None else []
        self.debug = debug
        self.initialized = False
        self.recompute_every_update = recompute_every_update

        self.workers = 12 if workers is None else int(workers)
        self.use_numba = True

        self.auto_precision_from_zoom = True
        self.target_screen_step_px = 2.0
        self.min_precision = 1.0
        self._view_scale = None

        self.async_compute = bool(async_compute)

        self.rolling_mode = True
        self._roll_states = np.empty((0, 5), dtype=np.float64) if np is not None else []
        self._executor = None
        self._pending_future = None
        self._pending_futures = []
        self._pending_job_id = 0
        self._next_job_id = 1
        self._last_swapped_job_id = 0
        self._jobs_submitted = 0
        self._jobs_swapped = 0
        self._single_flight = True

        self._computed_since_last_update = 0
        

        self._last_swapped_snapshot = None

        self.snapshot_velocity_rel_tol = 0.01
        self.snapshot_velocity_abs_tol = 100.0

        self.snapshot_position_abs_tol = 1000.0

        self.force_sync_on_stale = False


        self.view_change_cooldown = 0.0
        self._view_change_cooldown_until = 0.0

        self.snapshot_view_rel_tol = 1e-6

        self._view_scale_changed = False

        # optionale übersetzung des referenzrahmens. wenn gesetzt, berechnet predictor
        # bewegung in einem körper-zentrierten nicht-rotierenden rahmen durch subtraktion
        # der referenzkörper-beschleunigung.
        self.reference_body_index = None

        if self.async_compute and not self.rolling_mode:
            max_workers = max(1, int(self.workers))
            self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="predictor")

            self._pending_futures = []

    def reset(self):
        self._cancel_pending_job()
        self.points = np.empty((0, 3), dtype=np.float64) if np is not None else []
        self._roll_states = np.empty((0, 5), dtype=np.float64) if np is not None else []
        self.initialized = False

    def set_reference_body_index(self, index: int | None):
        if index is None:
            new_index = None
        else:
            new_index = int(index)

        if new_index == self.reference_body_index:
            return

        self.reference_body_index = new_index
            # Frame-Änderung macht aktuell gespeicherte Prädiktor-Punkte ungültig.
        self.reset()

    def _resolve_reference_body(self, world):
        idx = self.reference_body_index
        if idx is None:
            return 0, 0.0, 0.0

        try:
            idx = int(idx)
        except Exception:
            return 0, 0.0, 0.0

        try:
            if idx < 0 or idx >= len(world.body):
                return 0, 0.0, 0.0
            ref = world.body[idx]
            return 1, float(ref.position.x), float(ref.position.y)
        except Exception:
            return 0, 0.0, 0.0

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
            # timestamp beibehalten falls beim ersten punkt vorhanden
            try:
                t0 = float(self.points[0][2])
            except Exception:
                t0 = 0.0
            self.points[0] = (sx, sy, t0)

    def _count_recomputed_points(self, old_points, new_points, tol=1e-6):
        """Gibt die Anzahl der Einträge in `new_points` zurück, die sich von `old_points` unterscheiden.

        Der vergleich überspringt den ersten punkt (anker) und behandelt
        einen zusätzlichen "tail" in `new_points` gegenüber `old_points`
        als neu berechnet.
        """
        try:
            if old_points is None:
                old_len = 0
            else:
                if np is not None and isinstance(old_points, np.ndarray):
                    old_len = int(old_points.shape[0])
                else:
                    old_len = len(old_points)
        except Exception:
            old_len = 0

        try:
            if new_points is None:
                return 0
            if np is not None and isinstance(new_points, np.ndarray):
                new_len = int(new_points.shape[0])
            else:
                new_len = len(new_points)
        except Exception:
            return 0

        if old_len <= 0:
            return max(0, new_len)

        try:
            if np is not None and isinstance(new_points, np.ndarray) and isinstance(old_points, np.ndarray):
                old_arr = old_points
                new_arr = new_points
            else:
                old_arr = np.array(old_points, dtype=np.float64)
                new_arr = np.array(new_points, dtype=np.float64)
        except Exception:
            try:
                old_arr = np.array(old_points, dtype=np.float64)
                new_arr = np.array(new_points, dtype=np.float64)
            except Exception:
                return max(0, new_len)

        min_len = min(int(old_arr.shape[0]), int(new_arr.shape[0]))

        if min_len <= 1:
            changed_in_overlap = 0
        else:
            a = old_arr[1:min_len, :2]
            b = new_arr[1:min_len, :2]
            diffs = np.abs(a - b) > float(tol)
            rows_changed = np.any(diffs, axis=1)
            changed_in_overlap = int(np.count_nonzero(rows_changed))

        added_tail = max(0, int(new_arr.shape[0]) - int(old_arr.shape[0]))

        return changed_in_overlap + added_tail

    def set_view_scale(self, scale: float):
        try:
            scale = float(scale)
        except Exception:
            return
        if scale > 0.0:
            old = self._view_scale

            if old is not None:
                try:
                    rel_change = abs(scale - old) / max(abs(old), 1e-30)
                except Exception:
                    rel_change = 0.0
                if rel_change <= self.snapshot_view_rel_tol:
                    return


            self._view_scale = scale
 
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

  
            try:
                if old is not None and self.async_compute and len(getattr(self, "_pending_futures", [])) > 0:
                    rel = abs(self._view_scale - old) / max(abs(old), 1e-30)
                    if rel > 0.02:
                        if self.debug:
                            print(f"PRED_DBG_CANCEL_PENDING: zoom rel_change={rel:.3f} canceling pending job")
                        self._cancel_pending_job()
            except Exception:
                pass

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
    # alle wartenden futures abbrechen (unterstützt multi-worker-modus).
        pending = getattr(self, "_pending_futures", None)
        if pending is None:

            if self._pending_future is None:
                return
            if not self._pending_future.done():
                self._pending_future.cancel()
            self._pending_future = None
            self._pending_job_id = 0
            return

        for job_id, fut in list(pending):
            try:
                if not fut.done():
                    fut.cancel()
            except Exception:
                pass
        pending.clear()
        self._pending_job_id = 0

    def _make_snapshot(self, ship, world, max_points):
        effective_precision = self._effective_precision()
        ref_enabled, ref_px, ref_py = self._resolve_reference_body(world)
        snapshot = {
            "ship_px": float(ship.position.x),
            "ship_py": float(ship.position.y),
            "ship_vx": float(ship.velocity.x),
            "ship_vy": float(ship.velocity.y),
            "ref_enabled": int(ref_enabled),
            "ref_px": float(ref_px),
            "ref_py": float(ref_py),
            "G": float(world.G),
            "dt": float(self.dt),
            "precision": float(effective_precision),
            "max_points": int(max_points),
            "max_iters": int(max(10000, max_points * 100)),
            "numba": True,
        }

        try:
            snapshot["sim_time"] = float(world.time)
        except Exception:
            snapshot["sim_time"] = 0.0
        try:
            snapshot["submit_ts"] = float(time.time())
        except Exception:
            snapshot["submit_ts"] = 0.0

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
            int(snapshot.get("ref_enabled", 0)),
            float(snapshot.get("ref_px", 0.0)),
            float(snapshot.get("ref_py", 0.0)),
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
        computed_count = int(used)

        try:
            base_sim_time = float(snapshot.get("sim_time", 0.0)) if snapshot is not None else 0.0
        except Exception:
            base_sim_time = 0.0

        try:
            if np is not None and isinstance(points, np.ndarray) and points.shape[1] >= 3:
                points = points.copy()
                points[:, 2] = points[:, 2] + base_sim_time
            else:

                pts = []
                for p in points:
                    try:
                        pts.append((float(p[0]), float(p[1]), float(p[2]) + base_sim_time))
                    except Exception:
                        pts.append((float(p[0]), float(p[1]), base_sim_time))
                points = pts
        except Exception:
            pass

        return {"points": points, "snapshot": snapshot, "computed": computed_count}

    def _compute_full_rolling(self, ship, world):
        if self.num_points <= 0:
            self.points = np.empty((0, 3), dtype=np.float64) if np is not None else []
            self._roll_states = np.empty((0, 5), dtype=np.float64) if np is not None else []
            self.initialized = True
            return

        if self.precision <= 0.0:
            raise ValueError("Predictor precision must be > 0")

        max_points = self._get_target_point_cap()
        snapshot = self._make_snapshot(ship, world, max_points)
        base_t = float(snapshot.get("sim_time", 0.0))

        out, used = _compute_distance_points_numba_state(
            snapshot["ship_px"],
            snapshot["ship_py"],
            snapshot["ship_vx"],
            snapshot["ship_vy"],
            base_t,
            int(snapshot.get("ref_enabled", 0)),
            float(snapshot.get("ref_px", 0.0)),
            float(snapshot.get("ref_py", 0.0)),
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

        states = out[:int(used)].copy()
        new_points = states[:, :3].copy() if (np is not None and isinstance(states, np.ndarray) and states.shape[0] > 0) else np.empty((0, 3), dtype=np.float64)

        try:
            old_points = self.points if (np is not None and isinstance(self.points, np.ndarray)) else np.array(self.points, dtype=np.float64) if self.points is not None else None
        except Exception:
            old_points = None
        try:
            changed = int(self._count_recomputed_points(old_points, new_points))
        except Exception:
            changed = int(new_points.shape[0]) if (hasattr(new_points, 'shape')) else 0
        try:
            self._computed_since_last_update += changed
        except Exception:
            pass
        self._roll_states = states
        if np is not None and isinstance(states, np.ndarray) and states.shape[0] > 0:
            self.points = new_points.copy()
        else:
            self.points = np.empty((0, 3), dtype=np.float64) if np is not None else []
        self.initialized = True
        self._last_swapped_snapshot = snapshot

    def _append_rolling_tail(self, world, missing_points):
        if missing_points <= 0:
            return 0
        if np is None or not isinstance(self._roll_states, np.ndarray) or self._roll_states.shape[0] == 0:
            return 0

        tail = self._roll_states[-1]
        init_px = float(tail[0])
        init_py = float(tail[1])
        init_t = float(tail[2])
        init_vx = float(tail[3])
        init_vy = float(tail[4])

        body_x, body_y, body_m, body_fixed = self._serialize_bodies_numba(world)
        ref_enabled, ref_px, ref_py = self._resolve_reference_body(world)
        max_new_points = int(missing_points) + 1  # include seed sample at index 0
        max_iters = int(max(10000, max_new_points * 100))

        out, used = _compute_distance_points_numba_state(
            init_px,
            init_py,
            init_vx,
            init_vy,
            init_t,
            int(ref_enabled),
            float(ref_px),
            float(ref_py),
            body_x,
            body_y,
            body_m,
            body_fixed,
            float(world.G),
            float(self.dt),
            float(self._effective_precision()),
            max_new_points,
            max_iters,
        )

        if int(used) <= 1:
            return 0

        to_add = out[1:int(used)].copy()
        if to_add.shape[0] > missing_points:
            to_add = to_add[:missing_points]
        if to_add.shape[0] <= 0:
            return 0

        self._roll_states = np.concatenate((self._roll_states, to_add), axis=0)
        self.points = self._roll_states[:, :3].copy()
        added = int(to_add.shape[0])
        try:
            self._computed_since_last_update += added
        except Exception:
            pass
        return added

    def _update_rolling(self, ship, world):
        # On first run or when zoom changed (auto precision), rebuild once.
        if (not self.initialized) or ( np is None or not isinstance(self._roll_states, np.ndarray) or self._roll_states.shape[0] == 0) or getattr(self, "_view_scale_changed", False):
            self._compute_full_rolling(ship, world)
            self._view_scale_changed = False
        else:
            removed = self.remove_passed_points(ship)
            if removed > 0 and np is not None and isinstance(self._roll_states, np.ndarray) and self._roll_states.shape[0] > 0:
                cut = min(int(removed), max(0, self._roll_states.shape[0] - 1))
                if cut > 0:
                    self._roll_states = self._roll_states[cut:]
                self.points = self._roll_states[:, :3].copy()

            target_points = self._get_target_point_cap()
            missing = target_points - self._points_count()
            if missing > 0:
                self._append_rolling_tail(world, missing)

        self._anchor_first_point(ship)
        if np is not None and isinstance(self._roll_states, np.ndarray) and self._roll_states.shape[0] > 0:
            self._roll_states[0, 0] = float(ship.position.x)
            self._roll_states[0, 1] = float(ship.position.y)
            try:
                self._roll_states[0, 2] = float(world.time)
            except Exception:
                pass
            self._roll_states[0, 3] = float(ship.velocity.x)
            self._roll_states[0, 4] = float(ship.velocity.y)

    def _submit_async_compute(self, ship, world, max_points):
        pending = getattr(self, "_pending_futures", [])

        if self._single_flight:
            if len(pending) > 0:
                # DEBUG-AGENT-ADD BEGIN - debug only, remove after verification
                try:
                    print(f"PRED_DBG_SUBMIT: blocked single_flight pending_len={len(pending)} _next_job_id={self._next_job_id}", flush=True)
                except Exception:
                    pass
                # DEBUG-AGENT-ADD END
                return

        snapshot = self._make_snapshot(ship, world, max_points)
        # DEBUG-AGENT-ADD BEGIN - debug only, remove after verification
        try:
            exec_info = 'has_executor' if getattr(self, '_executor', None) is not None else 'no_executor'
            print(f"PRED_DBG_SUBMIT: submitting job_id={self._next_job_id} max_points={max_points} {exec_info}", flush=True)
        except Exception:
            pass
        # DEBUG-AGENT-ADD END

        fut = self._executor.submit(self._compute_from_snapshot, snapshot)
        job_id = self._next_job_id

            # Ersetze Queue statt endlos anzuhängen
        if self._single_flight:
            self._pending_futures = [(job_id, fut)]
        else:
            self._pending_futures.append((job_id, fut))

        self._next_job_id += 1
        self._jobs_submitted += 1

    def _swap_ready_result(self, current_ship=None, current_world=None):
        pending = getattr(self, "_pending_futures", None)

        # DEBUG-AGENT-ADD BEGIN - debug only, remove after verification
        try:
            pend_len = len(pending) if pending is not None else None
            print(f"PRED_DBG_SWAP: entering swap pending_len={pend_len} _pending_job_id={getattr(self,'_pending_job_id',0)} _pending_future_set={getattr(self,'_pending_future',None) is not None}", flush=True)
        except Exception:
            pass
        # DEBUG-AGENT-ADD END

        if pending is None:
            pending = []

        if not pending:
            if self._pending_future is None or not self._pending_future.done():
                # DEBUG-AGENT-ADD BEGIN - debug only
                try:
                    print(f"PRED_DBG_SWAP: no pending list and no single pending future ready (pending_future={self._pending_future})", flush=True)
                except Exception:
                    pass
                # DEBUG-AGENT-ADD END
                return False
            finished_future = self._pending_future
            finished_job_id = self._pending_job_id
            self._pending_future = None
            self._pending_job_id = 0
        else:
            finished_future = None
            finished_job_id = None

            if len(pending) > 2:

                pending[:] = pending[-2:]

            # find first completed future
            for idx, (jid, fut) in enumerate(pending):
                try:
                    done = fut.done()
                except Exception:
                    done = False
                if done:
                    # DEBUG-AGENT-ADD BEGIN - debug only
                    try:
                        print(f"PRED_DBG_SWAP: found finished future jid={jid} idx={idx} done={done}", flush=True)
                    except Exception:
                        pass
                    # DEBUG-AGENT-ADD END
                    finished_future = fut
                    finished_job_id = jid
                    pending.pop(idx)
                    break

            if finished_future is None:
                # DEBUG-AGENT-ADD BEGIN - debug only
                try:
                    print(f"PRED_DBG_SWAP: no finished future found in pending", flush=True)
                except Exception:
                    pass
                # DEBUG-AGENT-ADD END
                return False

        try:
            result = finished_future.result()


            if isinstance(result, dict):
                points = result.get("points")
                snapshot = result.get("snapshot")
            else:
                points = result
                snapshot = None


            if snapshot is not None and current_ship is not None:
                svx = float(snapshot.get("ship_vx", 0.0))
                svy = float(snapshot.get("ship_vy", 0.0))
                cur_vx = float(current_ship.velocity.x)
                cur_vy = float(current_ship.velocity.y)

                dvx = cur_vx - svx
                dvy = cur_vy - svy
                delta_speed = math.hypot(dvx, dvy)
                cur_speed = math.hypot(cur_vx, cur_vy)
                allowed_speed = max(self.snapshot_velocity_abs_tol, self.snapshot_velocity_rel_tol * max(cur_speed, 1.0))


                spx = float(snapshot.get("ship_px", 0.0))
                spy = float(snapshot.get("ship_py", 0.0))
                cur_px = float(current_ship.position.x)
                cur_py = float(current_ship.position.y)
                pos_delta = math.hypot(cur_px - spx, cur_py - spy)

                submit_ts = float(snapshot.get("submit_ts", 0.0))
                age = max(0.0, time.time() - submit_ts)
                allowed_pos = max(self.snapshot_position_abs_tol, self.snapshot_velocity_abs_tol * max(0.1, age))


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

                    if is_stale_view:
                            # DEBUG-AGENT-ADD BEGIN - debug only
                            try:
                                print(f"PRED_DBG_SWAP: rejecting result - stale_view (snap_view={snap_view} cur_view={self._view_scale})", flush=True)
                            except Exception:
                                pass
                            # DEBUG-AGENT-ADD END
                            return False


                    if self.force_sync_on_stale and current_world is not None:
                        self._compute_full(current_ship, current_world)
                        self._anchor_first_point(current_ship)
                        self._last_swapped_job_id = finished_job_id
                        self._jobs_swapped += 1
                        return True
                    else:
                        return False


            try:
                old_points = self.points if (np is not None and isinstance(self.points, np.ndarray)) else np.array(self.points, dtype=np.float64) if self.points is not None else None
            except Exception:
                old_points = None

         
            try:
                changed = int(self._count_recomputed_points(old_points, points))
            except Exception:
           
                changed = None
                if isinstance(result, dict):
                    changed = result.get('computed', None)
                if changed is None:
                    try:
                        changed = int(points.shape[0]) if (np is not None and hasattr(points, 'shape')) else int(len(points))
                    except Exception:
                        changed = 0
            try:
                self._computed_since_last_update += int(changed)
            except Exception:
                pass

            # DEBUG-AGENT-ADD: log swap result
            try:
                print(f"PRED_DBG_SWAP: swapped job_id={finished_job_id} changed={changed} points_len={(points.shape[0] if (np is not None and hasattr(points, 'shape')) else len(points))}", flush=True)
            except Exception:
                pass

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
            return True
        except Exception as exc:
            return False

    def _get_target_point_cap(self):

        if self.num_points <= 0:
            return 0

        if self.length is None:
            return self.num_points

        spacing_for_cap = self.base_precision if self.base_precision > 0.0 else self.precision
        max_by_length = max(1, int(self.length / spacing_for_cap) + 1)
        return min(self.num_points, max_by_length)

    def _compute_full(self, ship, world):
        if self.rolling_mode:
            self._compute_full_rolling(ship, world)
            return

        if self.num_points <= 0:
            self.points = np.empty((0, 2), dtype=np.float64) if np is not None else []
            self.initialized = True
            return

        if self.precision <= 0.0:
            raise ValueError("Predictor precision must be > 0")

        max_points = self._get_target_point_cap()

        snapshot = self._make_snapshot(ship, world, max_points)

        try:
            old_points = self.points if (np is not None and isinstance(self.points, np.ndarray)) else np.array(self.points, dtype=np.float64) if self.points is not None else None
        except Exception:
            old_points = None

        result = self._compute_from_snapshot(snapshot)
        if isinstance(result, dict):
            new_points = result["points"]
            self.points = new_points
            self._last_swapped_snapshot = result.get("snapshot")
        else:
            new_points = result
            self.points = new_points

        self.initialized = True
 
        try:
            changed = int(self._count_recomputed_points(old_points, new_points))
        except Exception:
  
            changed = None
            if isinstance(result, dict):
                changed = result.get('computed', None)
            if changed is None:
                try:
                    changed = int(self.points.shape[0]) if (np is not None and hasattr(self.points, 'shape')) else int(len(self.points))
                except Exception:
                    changed = 0
        try:
            self._computed_since_last_update += int(changed)
        except Exception:
            pass

    def initialize(self, ship, world):
        self.reset()
        if self.rolling_mode:
            self._compute_full_rolling(ship, world)
            self._anchor_first_point(ship)
            if np is not None and isinstance(self._roll_states, np.ndarray) and self._roll_states.shape[0] > 0:
                self._roll_states[0, 0] = float(ship.position.x)
                self._roll_states[0, 1] = float(ship.position.y)
                try:
                    self._roll_states[0, 2] = float(world.time)
                except Exception:
                    pass
                self._roll_states[0, 3] = float(ship.velocity.x)
                self._roll_states[0, 4] = float(ship.velocity.y)
            return
        self._compute_full(ship, world)
        self._anchor_first_point(ship)

    def update(self, ship, world):
        try:
            self._computed_since_last_update = 0
        except Exception:
            pass

        if self.num_points <= 0:
            self.reset()
            if self.debug:
                try:
                    print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                except Exception:
                    pass
            self._computed_since_last_update = 0
            return

        if self.precision <= 0.0:
            raise ValueError("Predictor precision must be > 0")

        if self.rolling_mode:
            # Detect sudden ship velocity changes (thrust) even in rolling
            # mode by tracking the last observed ship velocity. If a large
            # delta is detected, rebuild the full rolling state so stored
            # points don't remain stale.
            try:
                cur_vx = float(ship.velocity.x)
                cur_vy = float(ship.velocity.y)
                last_vx = getattr(self, '_last_ship_vx', None)
                last_vy = getattr(self, '_last_ship_vy', None)
                if last_vx is not None and last_vy is not None:
                    dvx = cur_vx - float(last_vx)
                    dvy = cur_vy - float(last_vy)
                    delta_speed = math.hypot(dvx, dvy)
                    cur_speed = math.hypot(cur_vx, cur_vy)
                    allowed_speed = max(self.snapshot_velocity_abs_tol, self.snapshot_velocity_rel_tol * max(cur_speed, 1.0))
                    if delta_speed >= allowed_speed:
                        if self.debug:
                            try:
                                print(f"PRED_DBG_VEL_CHANGE: dv={delta_speed:.6e} allowed={allowed_speed:.6e}", flush=True)
                            except Exception:
                                pass
                        try:
                            # Rebuild entire rolling prediction synchronously.
                            self._compute_full_rolling(ship, world)
                            self._anchor_first_point(ship)
                        except Exception:
                            pass
                        # Update remembered velocity and report
                        try:
                            self._last_ship_vx = cur_vx
                            self._last_ship_vy = cur_vy
                        except Exception:
                            pass
                        if self.debug:
                            try:
                                print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                            except Exception:
                                pass
                        self._computed_since_last_update = 0
                        return
                # remember velocity for next update
                self._last_ship_vx = cur_vx
                self._last_ship_vy = cur_vy
            except Exception:
                pass

            self._update_rolling(ship, world)
            if self.debug:
                try:
                    print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                except Exception:
                    pass
            self._computed_since_last_update = 0
            return

        if not self.async_compute:
            if not self.initialized:
                self.initialize(ship, world)
                if self.debug:
                    try:
                        print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                    except Exception:
                        pass
                self._computed_since_last_update = 0
                return

            if self.recompute_every_update:
                self._compute_full(ship, world)
                self._anchor_first_point(ship)
                if self.debug:
                    try:
                        print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                    except Exception:
                        pass
                self._computed_since_last_update = 0
                return

            removed = self.remove_passed_points(ship)
            target_points = self._get_target_point_cap()
            if self._points_count() < target_points:
                self._compute_full(ship, world)
            self._anchor_first_point(ship)
            if self.debug:
                try:
                    print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                except Exception:
                    pass
            self._computed_since_last_update = 0
            return


        if getattr(self, '_view_scale_changed', False):
            if ship is not None and world is not None:
                self._cancel_pending_job()
                self._compute_full(ship, world)
                self._anchor_first_point(ship)
                self._view_scale_changed = False
                if self.debug:
                    try:
                        print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                    except Exception:
                        pass
                self._computed_since_last_update = 0
                return

        # Detect large ship state changes (e.g. player thrust) and force
        # a recompute so stored predictor points don't remain stale.
        try:
            if ship is not None and self._last_swapped_snapshot is not None:
                svx = float(self._last_swapped_snapshot.get("ship_vx", 0.0))
                svy = float(self._last_swapped_snapshot.get("ship_vy", 0.0))
                cur_vx = float(ship.velocity.x)
                cur_vy = float(ship.velocity.y)

                dvx = cur_vx - svx
                dvy = cur_vy - svy
                delta_speed = math.hypot(dvx, dvy)
                cur_speed = math.hypot(cur_vx, cur_vy)
                allowed_speed = max(self.snapshot_velocity_abs_tol, self.snapshot_velocity_rel_tol * max(cur_speed, 1.0))

                if delta_speed >= allowed_speed:
                    if self.debug:
                        try:
                            print(f"PRED_DBG_VEL_CHANGE: dv={delta_speed:.6e} allowed={allowed_speed:.6e}", flush=True)
                        except Exception:
                            pass

                    # Cancel pending work and either recompute synchronously
                    # (rolling mode / non-async) or submit a fresh async job.
                    try:
                        self._cancel_pending_job()
                    except Exception:
                        pass

                    if self.rolling_mode:
                        self._compute_full_rolling(ship, world)
                        self._anchor_first_point(ship)
                        if self.debug:
                            try:
                                print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                            except Exception:
                                pass
                        self._computed_since_last_update = 0
                        return

                    target_points = self._get_target_point_cap()
                    if self.async_compute:
                        try:
                            self._submit_async_compute(ship, world, target_points)
                        except Exception:
                            pass
                        if self.debug:
                            try:
                                print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                            except Exception:
                                pass
                        self._computed_since_last_update = 0
                        return
                    else:
                        self._compute_full(ship, world)
                        self._anchor_first_point(ship)
                        if self.debug:
                            try:
                                print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                            except Exception:
                                pass
                        self._computed_since_last_update = 0
                        return
        except Exception:
            pass

        swapped = self._swap_ready_result(ship, world)
        target_points = self._get_target_point_cap()

        if not self.initialized:
            self._submit_async_compute(ship, world, target_points)
            if self.debug:
                try:
                    print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
                except Exception:
                    pass
            self._computed_since_last_update = 0
            return

        if not self.recompute_every_update:
            self.remove_passed_points(ship)

        if self.recompute_every_update or self._points_count() < target_points or swapped:
            self._submit_async_compute(ship, world, target_points)

        if self.initialized:
            self._anchor_first_point(ship)
        if self.debug:
            try:
                print(f"PRED_DBG_COMPUTED: computed={self._computed_since_last_update}")
            except Exception:
                pass
        self._computed_since_last_update = 0

    def get_points(self):
        return self.points

    def get_precision_factor(self):
        if self.base_precision <= 0.0:
            return 1.0
        return self.precision / self.base_precision

    def get_display_length(self):
        if self.length is None:
            return None
        return self.length * self.get_precision_factor()

    def set_precision(self, meters: float):
        meters = float(meters)
        if meters <= 0.0:
            raise ValueError("precision must be > 0")
        self.precision = meters
        if self.rolling_mode:
            self.reset()
        elif self.async_compute:
            self._cancel_pending_job()

    def set_length(self, meters: float | None):
        if meters is None:
            self.length = None
            if self.rolling_mode:
                self.reset()
            elif self.async_compute:
                self._cancel_pending_job()
            return
        meters = float(meters)
        if meters <= 0.0:
            raise ValueError("length must be > 0")
        self.length = meters
        if self.rolling_mode:
            self.reset()
        elif self.async_compute:
            self._cancel_pending_job()

    def set_num_points(self, count: int):
        self.num_points = max(0, int(count))
        self.reset()

    def advance_state(self, world=None):

        if self.async_compute:
            self._swap_ready_result(None, world)

    def get_async_status(self):
        return {
            "enabled": self.async_compute,
            "pending": len(getattr(self, "_pending_futures", [])) > 0,
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
            if self.points.shape[0] <= 1:
                return 0
            coords = self.points[:, :2]
            dx = coords[:, 0] - sx
            dy = coords[:, 1] - sy
            dsq = dx * dx + dy * dy
    
            diffs = dsq[1:] - dsq[:-1]
            idxs = np.nonzero(diffs >= 0)[0]
            if idxs.size == 0:

                remove_count = max(0, self.points.shape[0] - 1)
            else:
                remove_count = int(idxs[0])
            if remove_count > 0:
                self.points = self.points[remove_count:]
                removed += remove_count
        else:

            n = len(self.points)
            if n <= 1:
                return 0
            remove_count = 0
            try:
                for i in range(n - 1):
                    p0 = self.points[i]
                    p1 = self.points[i + 1]
                    if hasattr(p0, 'distance_squared_to'):
                        d0 = p0.distance_squared_to(ship.position)
                        d1 = p1.distance_squared_to(ship.position)
                    else:
                        x0 = float(p0[0]); y0 = float(p0[1])
                        x1 = float(p1[0]); y1 = float(p1[1])
                        dx0 = x0 - sx; dy0 = y0 - sy
                        dx1 = x1 - sx; dy1 = y1 - sy
                        d0 = dx0 * dx0 + dy0 * dy0
                        d1 = dx1 * dx1 + dy1 * dy1
                    if d1 < d0:
                        remove_count += 1
                    else:
                        break
            except Exception:
                remove_count = 0
            if remove_count > 0:
                try:
                    del self.points[:remove_count]
                except Exception:

                    for _ in range(remove_count):
                        try:
                            self.points.pop(0)
                        except Exception:
                            break
                removed += remove_count

        return removed
