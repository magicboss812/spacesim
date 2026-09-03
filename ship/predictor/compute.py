"""Die rechnung: schnappschuss, integration, punktreihe.

Die kerne selbst liegen in `physics/kernels/` -- hier steht, WAS ihnen
uebergeben wird und wie das ergebnis zusammengesetzt wird.
"""
import math
import time

import numpy as np

from physics.vec import Vec2
from physics.kernels import (
    BODY_MEMO_COLUMNS,
    POINT_COLUMNS,
    _empty_points,
    _no_body_memo,
    _widen_points,
)
from physics.kernels.propagate import (
    _compute_distance_points_aspi_numba,
    _compute_distance_points_numba,
    _compute_distance_points_numba_state,
    _compute_distance_points_rkn_numba,
)
from physics.kernels.kepler import _body_position_at_time_numba


class ComputeMixin:
    """Vom weltzustand zur punktreihe.

    Ein SCHNAPPSCHUSS friert alles ein, was die rechnung braucht (koerper,
    bahnelemente, schiffszustand, horizont), damit sie in einem hintergrund-
    faden laufen kann, ohne die welt zu lesen, waehrend der hauptfaden sie
    weiterbewegt. Die eigentliche zahlenarbeit steht in `physics/kernels/`."""

    def _snapshot_body_index_by_names(self, snapshot, names):
        body_names = snapshot.get("body_names", None) if snapshot is not None else None
        if body_names is None:
            return -1
        wanted = set(names)
        for i, name in enumerate(body_names):
            key = str(name).strip().lower()
            if key in wanted:
                return int(i)
        return -1

    def _snapshot_body_position_at_local_t(self, snapshot, index, local_t):
        if snapshot is None or index < 0:
            return None
        try:
            if bool(snapshot.get("use_time_dependent_bodies", True)):
                return _body_position_at_time_numba(
                    int(index),
                    float(local_t),
                    snapshot["body_x"],
                    snapshot["body_y"],
                    snapshot["body_m"],
                    snapshot["body_scripted"],
                    snapshot["body_a"],
                    snapshot["body_e"],
                    snapshot["body_theta"],
                    snapshot["body_arg"],
                    snapshot["body_parent"],
                    float(snapshot["G"]),
                    _no_body_memo(),
                )
            return float(snapshot["body_x"][index]), float(snapshot["body_y"][index])
        except Exception:
            return None

    def _snapshot_body_velocity_at_local_t(self, snapshot, index, local_t):
        dt = 1.0
        p0 = self._snapshot_body_position_at_local_t(snapshot, index, float(local_t) - dt)
        p1 = self._snapshot_body_position_at_local_t(snapshot, index, float(local_t) + dt)
        if p0 is None or p1 is None:
            return None
        return (float(p1[0]) - float(p0[0])) / (2.0 * dt), (float(p1[1]) - float(p0[1])) / (2.0 * dt)

    def _debug_moving_source_snapshot(self, snapshot):
        if not getattr(self, "debug_moving_sources", False):
            return
        try:
            labels = [
                ("Earth", ("earth", "erde")),
                ("Mun", ("mun", "moon", "mond")),
            ]
            parts = []
            for label, names in labels:
                idx = self._snapshot_body_index_by_names(snapshot, names)
                if idx < 0:
                    continue
                samples = []
                for t in (0.0, 3600.0, 7200.0):
                    pos = self._snapshot_body_position_at_local_t(snapshot, idx, t)
                    if pos is not None:
                        samples.append(f"t=+{t:.0f} {label}=({pos[0]:.6e},{pos[1]:.6e})")
                if samples:
                    parts.append(" ".join(samples))
            if parts:
                print("PRED_SOURCE_DBG: " + " | ".join(parts), flush=True)
        except Exception:
            pass

    def _debug_predictor_energy(self, snapshot, points):
        if not getattr(self, "debug_moving_sources", False):
            return
        try:
            if points is None:
                return
            if np is not None and isinstance(points, np.ndarray):
                n = int(points.shape[0])
                if n < 3:
                    return
                get_point = lambda i: (float(points[i, 0]), float(points[i, 1]), float(points[i, 2]))
            else:
                n = len(points)
                if n < 3:
                    return
                get_point = lambda i: (float(points[i][0]), float(points[i][1]), float(points[i][2]))

            earth_idx = self._snapshot_body_index_by_names(snapshot, ("earth", "erde"))
            if earth_idx < 0:
                return

            base_t = float(snapshot.get("sim_time", 0.0))
            earth_mass = float(snapshot["body_m"][earth_idx])
            G = float(snapshot["G"])
            indices = [0, n // 2, n - 1]
            parts = []
            for idx in indices:
                px, py, abs_t = get_point(idx)
                local_t = abs_t - base_t
                earth_pos = self._snapshot_body_position_at_local_t(snapshot, earth_idx, local_t)
                earth_vel = self._snapshot_body_velocity_at_local_t(snapshot, earth_idx, local_t)
                if earth_pos is None or earth_vel is None:
                    continue

                if idx <= 0:
                    px2, py2, t2 = get_point(1)
                    dt = max(1e-9, t2 - abs_t)
                    ship_vx = (px2 - px) / dt
                    ship_vy = (py2 - py) / dt
                elif idx >= n - 1:
                    px0, py0, t0 = get_point(n - 2)
                    dt = max(1e-9, abs_t - t0)
                    ship_vx = (px - px0) / dt
                    ship_vy = (py - py0) / dt
                else:
                    px0, py0, t0 = get_point(idx - 1)
                    px2, py2, t2 = get_point(idx + 1)
                    dt = max(1e-9, t2 - t0)
                    ship_vx = (px2 - px0) / dt
                    ship_vy = (py2 - py0) / dt

                rel_x = px - float(earth_pos[0])
                rel_y = py - float(earth_pos[1])
                rel_vx = ship_vx - float(earth_vel[0])
                rel_vy = ship_vy - float(earth_vel[1])
                r = math.hypot(rel_x, rel_y)
                if r <= 1e-9:
                    continue
                energy = 0.5 * (rel_vx * rel_vx + rel_vy * rel_vy) - G * earth_mass / r
                parts.append(f"i={idx} t=+{local_t:.3f}s E={energy:.6e}")
            if parts:
                print("PRED_ENERGY_DBG: " + " | ".join(parts), flush=True)
        except Exception:
            pass

    def _warn_rolling_rkn_once(self):
        if not self.debug:
            return
        if not self.rolling_mode or self.integrator_mode != "rkn":
            return
        if getattr(self, "_rolling_rkn_warning_printed", False):
            return
        self._rolling_rkn_warning_printed = True
        try:
            print("PRED_DBG_WARNING: rolling_mode uses RK4 state helper, not adaptive RKN", flush=True)
        except Exception:
            pass

    def _log_snapshot_result(self, accepted, reason, snapshot, cur_sim_time, sim_age, pos_delta, vel_delta):
        if not self.debug:
            return
        try:
            snap_sim_time = float(snapshot.get("sim_time", 0.0)) if snapshot is not None else 0.0
        except Exception:
            snap_sim_time = 0.0
        try:
            cur_time = float(cur_sim_time) if cur_sim_time is not None else float("nan")
        except Exception:
            cur_time = float("nan")
        try:
            age = float(sim_age) if sim_age is not None else float("nan")
        except Exception:
            age = float("nan")
        try:
            pd = float(pos_delta)
        except Exception:
            pd = float("nan")
        try:
            vd = float(vel_delta)
        except Exception:
            vd = float("nan")
        try:
            snapshot_version = int(snapshot.get("trajectory_version", -1)) if snapshot is not None else -1
        except Exception:
            snapshot_version = -1
        try:
            current_version = int(self._trajectory_version)
        except Exception:
            current_version = -1

        if accepted:
            print(
                "PRED_DBG_ACCEPT_SNAPSHOT: "
                f"reason={reason} "
                f"version={current_version} "
                f"sim_age={age:.6e} "
                f"pos_delta={pd:.6e} "
                f"vel_delta={vd:.6e} "
                f"snapshot_sim_time={snap_sim_time:.6f} "
                f"current_world_time={cur_time:.6f}",
                flush=True,
            )
        else:
            print(
                "PRED_DBG_REJECT_SNAPSHOT: "
                f"reason={reason} "
                f"snapshot_version={snapshot_version} "
                f"current_version={current_version} "
                f"sim_age={age:.6e} "
                f"pos_delta={pd:.6e} "
                f"vel_delta={vd:.6e} "
                f"snapshot_sim_time={snap_sim_time:.6f} "
                f"current_world_time={cur_time:.6f}",
                flush=True,
            )

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
            body_fixed[i] = 1 if getattr(b, "fixed", True) else 0
        return body_x, body_y, body_m, body_fixed

    def _serialize_body_orbits_numba(self, world):
        count = len(world.body)
        body_scripted = np.empty(count, dtype=np.uint8)
        body_a = np.empty(count, dtype=np.float64)
        body_e = np.empty(count, dtype=np.float64)
        body_theta = np.empty(count, dtype=np.float64)
        body_arg = np.empty(count, dtype=np.float64)
        body_parent = np.empty(count, dtype=np.int64)

        body_to_index = {}
        for i, b in enumerate(world.body):
            body_to_index[b] = int(i)

        for i, b in enumerate(world.body):
            try:
                a = float(getattr(b, "semi_major_axis", 0.0) or 0.0)
            except Exception:
                a = 0.0
            try:
                e = float(getattr(b, "eccentricity", 0.0) or 0.0)
            except Exception:
                e = 0.0
            try:
                theta = float(getattr(b, "theta", 0.0) or 0.0)
            except Exception:
                theta = 0.0
            try:
                arg = float(getattr(b, "arg_periapsis", 0.0) or 0.0)
            except Exception:
                arg = 0.0

            parent = getattr(b, "is_moon_of", None)
            parent_index = body_to_index.get(parent, -1)
            scripted = bool(getattr(b, "scripted_orbit", False)) or (a > 0.0 and parent_index >= 0)

            body_scripted[i] = 1 if scripted else 0
            body_a[i] = a
            body_e[i] = e
            body_theta[i] = theta
            body_arg[i] = arg
            body_parent[i] = int(parent_index)

        return body_scripted, body_a, body_e, body_theta, body_arg, body_parent

    def _characteristic_timescale(self, world, ship):
        """sqrt(r_dominant / |g_total|) am schiff, in sekunden -- oder None.

        Dieselbe groesse, die `world.characteristic_timescale` fuer die
        zeitraffer-obergrenze benutzt; sie wird hier durchgereicht, damit es
        nur EINE definition davon gibt. Faellt die welt aus (tests reichen
        manchmal nur ein objekt herein), gibt es None und die decke bleibt,
        wie sie war.
        """
        if world is None or ship is None:
            return None
        fn = getattr(world, 'characteristic_timescale', None)
        if fn is None:
            return None
        try:
            value = fn(ship)
        except Exception:
            return None
        if value is None:
            return None
        try:
            value = float(value)
        except Exception:
            return None
        return value if math.isfinite(value) and value > 0.0 else None

    def _make_snapshot(self, ship, world, max_points):
        effective_precision = self._effective_precision()
        ref_enabled, ref_px, ref_py = self._resolve_reference_body(world)
        physics_ref_enabled = 0
        ref_index = self._current_reference_body_index()

        # Horizon-scaled far-field step ceiling. A long look-ahead over a smooth
        # arc is otherwise integrated at the fixed max_dt cap, so cost grows
        # ~arc/max_dt. Raise max_dt for long horizons to target a bounded step
        # budget (roughly constant compute cost); the adaptive tolerance +
        # step-doubling still refine near planets, so only the smooth far field
        # coarsens. Floored at the preset max_dt (short horizons unchanged) and
        # capped by the ceiling (close-approach safety). Tied to the HORIZON
        # (arc = max_points × precision), not to `precision`, so the spacing
        # decouple holds.
        eff_max_dt = float(self.rkn_max_dt)
        if self.rkn_adaptive_far_maxdt and float(self.rkn_far_field_target_steps) > 0.0:
            horizon_arc = float(max_points) * float(effective_precision)
            # Wieviel ZEIT deckt dieser bogen ab? Genau das braucht die
            # schrittzahl-schaetzung -- und genau das darf NICHT aus der
            # momentangeschwindigkeit kommen. Auf einer exzentrischen bahn ist
            # sie im perihel das MAXIMUM und im aphel das MINIMUM der ganzen
            # bahn, der fehler geht also in beide richtungen und ausgerechnet
            # im perihel nach unten: die schaetzung faellt zu kurz aus, die
            # schrittweite wird zu klein gedeckelt und der lauf kostet ein
            # vielfaches. Gemessen auf Pe 29 Gm / Ap 129 Gm bei 32x horizont:
            # 6663 schritte / 256 ms im perihel gegen 1160 / 43 ms im aphel --
            # dieselbe bahn, derselbe bogen, 6x. Genau das ist das stocken der
            # linie am perihel (die auffrischung faellt unter die bildrate)
            # und genau deshalb ist am aphel nichts davon zu merken.
            #
            # Die ehrliche groesse ist die MITTLERE inverse geschwindigkeit
            # ueber den bogen, und die kennt der letzte lauf bereits exakt:
            # seine zeitspanne durch seine bogenlaenge. Als verhaeltnis
            # gespeichert ueberlebt sie auch ein '+'/'-' auf den horizont.
            # Rueckkopplung ohne ruecklauf: die zeitspanne ist eine eigenschaft
            # der bahn, nicht der schrittweite -- ein groesseres max_dt
            # verandert sie nicht, es gibt also keinen regelkreis.
            time_per_arc = float(getattr(self, "_horizon_time_per_arc", 0.0) or 0.0)
            if time_per_arc <= 0.0:
                # Erster lauf: nichts gemessen, also der alte schaetzer.
                speed = math.hypot(float(ship.velocity.x), float(ship.velocity.y))
                if speed > 1.0:
                    time_per_arc = 1.0 / speed
            if time_per_arc > 0.0 and horizon_arc > 0.0:
                desired = (horizon_arc * time_per_arc) / float(self.rkn_far_field_target_steps)
                ceiling = float(self.rkn_max_dt_ceiling)
                # DIE DECKE DARF DIE BAHN NICHT UEBERSPRINGEN.
                #
                # `desired` kennt nur den horizont, nicht die bahn. Bei vielen
                # '+'-druecken wird sie deshalb groesser als ein nennenswerter
                # bruchteil der umlaufzeit -- und dann liegt die schrittweite
                # nicht mehr an der fehlerkontrolle, sondern an der decke.
                # Gemessen in einer erdumlaufbahn (rp 2e7 m, e = 0.6, T = 97 h)
                # bei 64x horizont: die linie weicht gegen dieselbe rechnung
                # mit fester decke (1500 s) um bis zu **6.0e7 m** ab, mehr als
                # die bahn selbst gross ist -- die vorhersage zeigt dann
                # schlicht eine andere bahn.
                #
                # Dieselbe schranke, die schon der zeitraffer benutzt:
                # `sqrt(r_dominant/|g|)`, fuer eine kreisbahn genau T/2pi.
                # Im FERNFELD (heliozentrisch, t_char ~ 5e6 s) ist sie um
                # groessenordnungen groesser als die decke und aendert nichts
                # -- der fernfeld-gewinn bleibt also unangetastet.
                #
                # SIE WIRD HIER NICHT MEHR EINGERECHNET, SONDERN IM KERNEL JE
                # SCHRITT. Hier war sie EINE zahl fuer den ganzen lauf, gemessen
                # am schiff, wie es beim anlegen des schnappschusses stand --
                # und damit falsch fuer jede bahn, die ihr regime verlaesst. Auf
                # einer abflugbahn (Erdorbit -> Jupiter) galt die zeitskala der
                # ERDE fuer die ganzen 2.85 jahre reiseflug: 24 633 schritte /
                # 899 ms statt 1 276 / 56 ms. Der kernel wertet dieselbe formel
                # jetzt am jeweiligen ORT aus (`_local_timescale_numba`), womit
                # die klammer im nahfeld unveraendert greift und sich erst
                # oeffnet, wenn das schiff den koerper wirklich verlassen hat.
                if not self.use_local_step_ceiling:
                    t_char = self._characteristic_timescale(world, ship)
                    if t_char is not None and t_char > 0.0:
                        orbit_cap = t_char / max(1e-9, float(self.rkn_max_dt_timescale_divisor))
                        if orbit_cap < ceiling:
                            ceiling = orbit_cap
                eff_max_dt = max(eff_max_dt, min(desired, ceiling))

        snapshot = {
            "ship_px": float(ship.position.x),
            "ship_py": float(ship.position.y),
            "ship_vx": float(ship.velocity.x),
            "ship_vy": float(ship.velocity.y),
            "ref_enabled": int(physics_ref_enabled),
            "reference_body_index": int(ref_index),
            "trajectory_version": int(self._trajectory_version),
            "ref_px": float(ref_px),
            "ref_py": float(ref_py),
            "G": float(world.G),
            "dt": float(self.dt),
            "precision": float(effective_precision),
            "max_points": int(max_points),
            "max_iters": int(max(10000, max_points * 100)),
            "numba": True,
            "integrator_mode": str(self.integrator_mode),
            "aspi_min_dt": float(self.aspi_min_dt),
            "aspi_max_dt": float(self.aspi_max_dt),
            "aspi_safety_g": float(self.aspi_safety_g),
            "aspi_safety_m": float(self.aspi_safety_m),
            "aspi_close_acc_threshold": float(self.aspi_close_acc_threshold),
            "aspi_use_rk4_fallback": bool(self.aspi_use_rk4_fallback),
            "rkn_min_dt": float(self.rkn_min_dt),
            "rkn_max_dt": float(eff_max_dt),
            # Boden und teiler der ORTLICHEN decke. Der boden ist die
            # schrittdecke der qualitaetsstufe -- die ortliche klammer darf nie
            # darunter, sonst wuerde sie das nahfeld strenger rechnen als der
            # alte globale weg. Teiler 0 = klammer aus.
            "rkn_max_dt_floor": float(self.rkn_max_dt),
            "rkn_max_dt_timescale_divisor": (
                float(self.rkn_max_dt_timescale_divisor)
                if (self.use_local_step_ceiling
                    and self.rkn_adaptive_far_maxdt
                    and float(self.rkn_max_dt_timescale_divisor) > 0.0)
                else 0.0
            ),
            "rkn_rtol": float(self.rkn_rtol),
            "rkn_atol_pos": float(self.rkn_atol_pos),
            "rkn_atol_vel": float(self.rkn_atol_vel),
            "rkn_safety": float(self.rkn_safety),
            "rkn_min_factor": float(self.rkn_min_factor),
            "rkn_max_factor": float(self.rkn_max_factor),
            "rkn_max_rejects": int(self.rkn_max_rejects),
            "base_precision": float(self.base_precision),
            "rkn_interval_coupling": bool(self.rkn_interval_coupling),
            "rkn_interval_tol_exponent": float(self.rkn_interval_tol_exponent),
            "strict_snapshot_matching": bool(self.strict_snapshot_matching),
            "use_time_dependent_bodies": bool(self.use_time_dependent_bodies),
            "use_reference_acceleration_correction": False,
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
        try:
            snapshot["eff_precision"] = float(self._effective_precision())
        except Exception:
            snapshot["eff_precision"] = None
        # Muss ueber den schnappschuss laufen, nicht ueber self: der kernel
        # laeuft im worker-thread und darf den schalter nicht mitten im lauf
        # wechseln sehen.
        snapshot["use_body_memo"] = bool(getattr(self, "use_body_memo", True))
        body_x, body_y, body_m, body_fixed = self._serialize_bodies_numba(world)
        snapshot["body_x"] = body_x
        snapshot["body_y"] = body_y
        snapshot["body_m"] = body_m
        snapshot["body_fixed"] = body_fixed
        (
            body_scripted,
            body_a,
            body_e,
            body_theta,
            body_arg,
            body_parent,
        ) = self._serialize_body_orbits_numba(world)
        snapshot["body_scripted"] = body_scripted
        snapshot["body_a"] = body_a
        snapshot["body_e"] = body_e
        snapshot["body_theta"] = body_theta
        snapshot["body_arg"] = body_arg
        snapshot["body_parent"] = body_parent
        snapshot["body_names"] = [str(getattr(b, "name", "")) for b in world.body]
        if getattr(self, "debug_moving_sources", False):
            self._debug_moving_source_snapshot(snapshot)
        return snapshot

    def _compute_from_snapshot(self, snapshot):
        # Thin timing shim: record the wall-clock cost of the actual trajectory
        # compute into self.last_compute_ms (single choke point for both the
        # async worker and the synchronous paths). See last_compute_ms in __init__.
        _t0 = time.perf_counter()
        try:
            result = self._compute_from_snapshot_impl(snapshot)
            self._record_horizon_time_per_arc(result, snapshot)
            return result
        finally:
            self.last_compute_ms = (time.perf_counter() - _t0) * 1000.0

    def _record_horizon_time_per_arc(self, result, snapshot):
        """Mittlere inverse geschwindigkeit ueber den horizont mitschreiben.

        Einzige quelle fuer die schrittweiten-deckelung in `_make_snapshot`
        (dort steht, warum die momentangeschwindigkeit dafuer untauglich ist).
        Laeuft auf dem worker-thread; es ist eine einzelne float-zuweisung,
        also unter der GIL atomar -- der hauptthread liest nie einen halben
        wert. Nur volle laeufe zaehlen: eine kurze fortsetzung (der
        schwanz-anbau im zeitraffer) misst nur ihr eigenes stueck bahn und
        wuerde die mittelung wieder auf einen momentanwert zusammenziehen.
        """
        try:
            points = result.get("points") if isinstance(result, dict) else None
            if points is None or len(points) < 3:
                return
            precision = float(snapshot.get("precision", 0.0) or 0.0)
            max_points = int(snapshot.get("max_points", 0) or 0)
            n = int(len(points))
            if precision <= 0.0 or max_points <= 0 or n < max(3, max_points // 2):
                return
            arc = float(n - 1) * precision
            span = float(points[-1, 2]) - float(points[0, 2])
            if arc > 0.0 and math.isfinite(span) and span > 0.0:
                self._horizon_time_per_arc = span / arc
        except Exception:
            pass

    def _compute_from_snapshot_impl(self, snapshot):
        mode = self._normalize_integrator_mode(snapshot.get("integrator_mode", "rkn"))
        self._debug_integrator_mode("compute", snapshot)
        rkn_stats = None

        if mode == "rkn":
            min_dt = float(snapshot.get("rkn_min_dt", 0.1))
            max_dt = float(snapshot.get("rkn_max_dt", 1500.0))
            base_dt = float(snapshot.get("dt", 60.0))
            rtol = float(snapshot.get("rkn_rtol", 1e-7))
            atol_pos = float(snapshot.get("rkn_atol_pos", 10.0))
            atol_vel = float(snapshot.get("rkn_atol_vel", 1e-4))
            safety = float(snapshot.get("rkn_safety", 0.9))
            min_factor = float(snapshot.get("rkn_min_factor", 0.2))
            max_factor = float(snapshot.get("rkn_max_factor", 5.0))
            max_rejects = int(snapshot.get("rkn_max_rejects", 32))

            if (not math.isfinite(min_dt)) or min_dt <= 0.0:
                min_dt = 0.1
            if (not math.isfinite(max_dt)) or max_dt <= 0.0:
                max_dt = 1500.0
            if max_dt < min_dt:
                max_dt = min_dt
            if (not math.isfinite(base_dt)) or base_dt <= 0.0:
                base_dt = max_dt
            if (not math.isfinite(rtol)) or rtol < 0.0:
                rtol = 1e-7
            if (not math.isfinite(atol_pos)) or atol_pos <= 0.0:
                atol_pos = 10.0
            if (not math.isfinite(atol_vel)) or atol_vel <= 0.0:
                atol_vel = 1e-4
            if (not math.isfinite(safety)) or safety <= 0.0:
                safety = 0.9
            if (not math.isfinite(min_factor)) or min_factor <= 0.0:
                min_factor = 0.2
            if (not math.isfinite(max_factor)) or max_factor < min_factor:
                max_factor = max(min_factor, 5.0)
            if max_rejects < 0:
                max_rejects = 0

            # --- Option A: intervall-gekoppelte schrittweite + toleranz -------
            # Koppelt die schrittzahl an die punktzahl statt an die bogenlänge.
            # Die max_dt-decke begrenzt schritte auf ~ein abtast-intervall pro
            # schritt (kosten-obergrenze ~ num_points auf glatten bögen); die
            # toleranz-lockerung sorgt dafür, dass diese decke auf glatten bögen
            # tatsächlich bindet, statt unnötig fein zu unterteilen. Nahe
            # vorbeiflügen übersteigt der fehler auch die gelockerte toleranz
            # weiterhin → unterteilung bis min_dt bleibt erhalten (sicherheit).
            # base_precision >= effektive precision → coarsen==1 → identität.
            if bool(snapshot.get("rkn_interval_coupling", False)):
                base_precision = float(snapshot.get("base_precision", 0.0))
                precision_val = float(snapshot.get("precision", 0.0))
                if base_precision > 0.0 and precision_val > base_precision:
                    coarsen = precision_val / base_precision
                    speed = math.hypot(
                        float(snapshot.get("ship_vx", 0.0)),
                        float(snapshot.get("ship_vy", 0.0)),
                    )
                    # zielschrittweite: ~ein abtast-intervall arc pro schritt
                    if speed > 1e-9:
                        dt_target = precision_val / speed
                    else:
                        dt_target = max_dt * coarsen
                    # decke nur anheben, nie senken; gegen absurde werte kappen
                    eff_max_dt = max(max_dt, min(dt_target, max_dt * coarsen))
                    if math.isfinite(eff_max_dt) and eff_max_dt > max_dt:
                        max_dt = eff_max_dt
                        base_dt = max_dt
                    # toleranz mit der vergröberung lockern (RKN4: fehler ~ dt^p)
                    exponent = float(snapshot.get("rkn_interval_tol_exponent", 4.0))
                    tol_scale = coarsen ** exponent
                    if math.isfinite(tol_scale) and tol_scale > 1.0:
                        rtol = rtol * tol_scale
                        atol_pos = atol_pos * tol_scale
                        atol_vel = atol_vel * tol_scale

            body_scripted = snapshot.get("body_scripted", None)
            body_a = snapshot.get("body_a", None)
            body_e = snapshot.get("body_e", None)
            body_theta = snapshot.get("body_theta", None)
            body_arg = snapshot.get("body_arg", None)
            body_parent = snapshot.get("body_parent", None)
            body_count = snapshot["body_x"].shape[0]
            if body_scripted is None:
                body_scripted = np.zeros(body_count, dtype=np.uint8)
            if body_a is None:
                body_a = np.zeros(body_count, dtype=np.float64)
            if body_e is None:
                body_e = np.zeros(body_count, dtype=np.float64)
            if body_theta is None:
                body_theta = np.zeros(body_count, dtype=np.float64)
            if body_arg is None:
                body_arg = np.zeros(body_count, dtype=np.float64)
            if body_parent is None:
                body_parent = np.full(body_count, -1, dtype=np.int64)

            use_time_dependent_bodies = 1 if bool(snapshot.get("use_time_dependent_bodies", True)) else 0
            ref_index = int(snapshot.get("reference_body_index", -1))

            # Die ORTLICHE schrittdecke (siehe _rkn_adaptive_step_time_numba).
            # `max_dt_floor` ist die schrittdecke der qualitaetsstufe -- unter
            # sie darf die ortliche rechnung nie gehen, damit das nahfeld exakt
            # so teuer bleibt wie zuvor. `timescale_divisor` = 0 schaltet die
            # ganze ortliche klammer ab (der A/B-schalter fuer den bit-vergleich
            # und der zustand, in dem `rkn_adaptive_far_maxdt` aus ist).
            max_dt_floor = float(snapshot.get("rkn_max_dt_floor", max_dt))
            timescale_divisor = float(snapshot.get("rkn_max_dt_timescale_divisor", 0.0))
            if not math.isfinite(max_dt_floor) or max_dt_floor <= 0.0:
                max_dt_floor = max_dt
            if not math.isfinite(timescale_divisor) or timescale_divisor <= 0.0:
                timescale_divisor = 0.0

            out, used, rkn_stats = _compute_distance_points_rkn_numba(
                snapshot["ship_px"],
                snapshot["ship_py"],
                snapshot["ship_vx"],
                snapshot["ship_vy"],
                0,
                float(snapshot.get("ref_px", 0.0)),
                float(snapshot.get("ref_py", 0.0)),
                snapshot["body_x"],
                snapshot["body_y"],
                snapshot["body_m"],
                snapshot["body_fixed"],
                body_scripted,
                body_a,
                body_e,
                body_theta,
                body_arg,
                body_parent,
                snapshot["G"],
                base_dt,
                snapshot["precision"],
                snapshot["max_points"],
                snapshot["max_iters"],
                min_dt,
                max_dt,
                rtol,
                atol_pos,
                atol_vel,
                safety,
                min_factor,
                max_factor,
                max_rejects,
                use_time_dependent_bodies,
                ref_index,
                float(snapshot.get("resume_t", 0.0)),
                float(snapshot.get("resume_accumulated", 0.0)),
                float(snapshot.get("resume_proposed_dt", 0.0)),
                1 if snapshot.get("use_body_memo", True) else 0,
                max_dt_floor,
                timescale_divisor,
            )
            # Alles aufheben, was noetig ist, um GENAU HIER weiterzurechnen.
            # Entscheidend ist, dass der SCHNAPPSCHUSS mitgehalten wird: die
            # koerper-arrays sind auf seine epoche bezogen und werden im
            # kernel analytisch fortgeschrieben. Mit einem frischeren
            # schnappschuss weiterzurechnen waere ein anderer lauf.
            self._resume_context = {
                'snapshot': snapshot,
                'base_dt': base_dt,
                'min_dt': min_dt,
                'max_dt': max_dt,
                # Die ortliche decke muss mit fortgesetzt werden, sonst rechnet
                # `_hold_extend_tail` den angehaengten schwanz nach einer
                # anderen regel als den rest der kurve -- genau die naht, die
                # der fortsetzbare kernel vermeiden soll.
                'max_dt_floor': max_dt_floor,
                'timescale_divisor': timescale_divisor,
                'rtol': rtol,
                'atol_pos': atol_pos,
                'atol_vel': atol_vel,
                'safety': safety,
                'min_factor': min_factor,
                'max_factor': max_factor,
                'max_rejects': max_rejects,
                'body_scripted': body_scripted,
                'body_a': body_a,
                'body_e': body_e,
                'body_theta': body_theta,
                'body_arg': body_arg,
                'body_parent': body_parent,
                'use_time_dependent_bodies': use_time_dependent_bodies,
                'ref_index': ref_index,
                'state': (float(rkn_stats[7]), float(rkn_stats[8]),
                          float(rkn_stats[9]), float(rkn_stats[10])),
                'accumulated': float(rkn_stats[11]),
                'proposed_dt': float(rkn_stats[12]),
                'kernel_t': float(rkn_stats[13]),
            }
        elif mode == "aspi" or mode == "aspi_rk4_fallback":
            min_dt = float(snapshot.get("aspi_min_dt", 1.0))
            max_dt = float(snapshot.get("aspi_max_dt", 120.0))
            base_dt = float(snapshot.get("dt", 60.0))
            safety_g = float(snapshot.get("aspi_safety_g", 0.05))
            safety_m = float(snapshot.get("aspi_safety_m", 0.5))
            close_acc_threshold = float(snapshot.get("aspi_close_acc_threshold", 0.02))

            if (not math.isfinite(min_dt)) or min_dt <= 0.0:
                min_dt = 1.0
            if (not math.isfinite(max_dt)) or max_dt <= 0.0:
                max_dt = 120.0
            if max_dt < min_dt:
                max_dt = min_dt
            if (not math.isfinite(base_dt)) or base_dt <= 0.0:
                base_dt = min_dt
            if (not math.isfinite(safety_g)) or safety_g <= 0.0:
                safety_g = 0.05
            if (not math.isfinite(safety_m)) or safety_m <= 0.0:
                safety_m = 0.5
            if (not math.isfinite(close_acc_threshold)) or close_acc_threshold < 0.0:
                close_acc_threshold = 0.02

            out, used = _compute_distance_points_aspi_numba(
                snapshot["ship_px"],
                snapshot["ship_py"],
                snapshot["ship_vx"],
                snapshot["ship_vy"],
                0,
                float(snapshot.get("ref_px", 0.0)),
                float(snapshot.get("ref_py", 0.0)),
                snapshot["body_x"],
                snapshot["body_y"],
                snapshot["body_m"],
                snapshot["body_fixed"],
                snapshot["G"],
                base_dt,
                snapshot["precision"],
                snapshot["max_points"],
                snapshot["max_iters"],
                min_dt,
                max_dt,
                safety_g,
                safety_m,
                close_acc_threshold,
                bool(snapshot.get("aspi_use_rk4_fallback", True)),
            )
        else:
            out, used = _compute_distance_points_numba(
                snapshot["ship_px"],
                snapshot["ship_py"],
                snapshot["ship_vx"],
                snapshot["ship_vy"],
                0,
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

        if getattr(self, "debug_moving_sources", False):
            self._debug_predictor_energy(snapshot, points)

        return {"points": points, "snapshot": snapshot, "computed": computed_count, "rkn_stats": rkn_stats}

    def _compute_full_rolling(self, ship, world):
        start_ts = time.time()
        try:
            if self.num_points <= 0:
                self.points = _empty_points()
                self._roll_states = np.empty((0, 5), dtype=np.float64) if np is not None else []
                self.initialized = True
                return

            if self.precision <= 0.0:
                raise ValueError("Predictor precision must be > 0")

            max_points = self._get_target_point_cap()
            snapshot = self._make_snapshot(ship, world, max_points)
            base_t = float(snapshot.get("sim_time", 0.0))

            # Rolling mode keeps the existing RK4 state path for now.
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
            # Alle fuenf spalten uebernehmen (frueher [:, :3]): die
            # geschwindigkeiten sind echte RK4-werte an den stuetzstellen und
            # taugen als tangente fuer die zeichenzeit-verfeinerung.
            new_points = states.copy() if (np is not None and isinstance(states, np.ndarray) and states.shape[0] > 0) else _empty_points()

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
                self.points = _empty_points()
            self.initialized = True
            self._last_swapped_snapshot = snapshot
        finally:
            try:
                if self.debug:
                    dur = time.time() - start_ts
                    try:
                        rsn = self._roll_states.shape[0] if (isinstance(getattr(self, '_roll_states', None), np.ndarray)) else 'n/a'
                    except Exception:
                        rsn = 'n/a'
                    print(f"PRED_DBG_COMPUTE_FULL_ROLLING: took {dur:.3f}s roll_states={rsn}", flush=True)
            except Exception:
                pass

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
        ref_enabled = 0
        max_new_points = int(missing_points) + 1  # include seed sample at index 0
        max_iters = int(max(10000, max_new_points * 100))

        # Rolling tail extension intentionally stays on the RK4 state helper.
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
        self.points = self._roll_states.copy()
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

            target_points = self._get_target_point_cap()
            missing = target_points - self._points_count()
            if missing > 0:
                self._append_rolling_tail(world, missing)

        self._anchor_first_point(ship, world)
        if np is not None and isinstance(self._roll_states, np.ndarray) and self._roll_states.shape[0] > 0:
            self._roll_states[0, 0] = float(ship.position.x)
            self._roll_states[0, 1] = float(ship.position.y)
            try:
                self._roll_states[0, 2] = float(world.time)
            except Exception:
                pass
            self._roll_states[0, 3] = float(ship.velocity.x)
            self._roll_states[0, 4] = float(ship.velocity.y)

    def _compute_full(self, ship, world):
        if self.rolling_mode:
            self._compute_full_rolling(ship, world)
            return

        if self.num_points <= 0:
            self.points = _empty_points()
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
            self._apply_rkn_stats(result.get("rkn_stats"))
        else:
            new_points = result
            self.points = new_points

        # Siehe _swap_ready_result: neue kurve, neue marker -- und eine
        # zeitspalte, die wieder auf ihrem eigenen schnappschuss sitzt.
        self._points_time_offset = 0.0
        self._synthetic_head = False
        self._invalidate_derived_caches()
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
