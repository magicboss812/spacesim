# predictor.py

from vec import Vec2   # <-- hier dein Dateiname einsetzen
import math

class Predictor:
    def __init__(self, num_points=5000, dt=60.0, spacing=100000.0):
        self.num_points = num_points
        self.dt = dt
        self.spacing = spacing  # gewünschter Abstand zwischen Punkten (Distanzbasiert)

        self.points = []
        self.sim_pos = None
        self.sim_vel = None

        # incremental fill state
        self.accumulated_distance = 0.0
        self.tail_extend_threshold = self.spacing * 2.0
        self.tail_extend_count = 2
        self.min_points = 3
        # enable debug prints to observe behavior
        self.debug = True

        self.initialized = False

    def reset(self):
        self.points.clear()
        self.sim_pos = None
        self.sim_vel = None
        self.initialized = False

    def initialize(self, ship, world):
        """Initial komplette Vorhersage berechnen (distanzbasiert)."""
        self.points.clear()

        self.sim_pos = ship.position.copy()
        self.sim_vel = ship.velocity.copy()

        self.points.append(self.sim_pos.copy())

        # reset incremental accumulator and (re)fill as before
        self.accumulated_distance = 0.0

        if self.debug:
            print(f"[Predictor DEBUG] initialize: start full fill num_points={self.num_points} spacing={self.spacing}")

        sim_steps = 0
        appended = len(self.points)

        while len(self.points) < self.num_points:
            prev_pos = self.sim_pos.copy()
            self._verlet_step(world)
            sim_steps += 1

            step2 = self.sim_pos.distance_squared_to(prev_pos)
            step_distance = math.sqrt(step2)
            self.accumulated_distance += step_distance

            if self.accumulated_distance >= self.spacing:
                self.points.append(self.sim_pos.copy())
                self.accumulated_distance = 0.0
                appended += 1

        if self.debug:
            print(f"[Predictor DEBUG] initialize: simulated_steps={sim_steps} appended_points={appended}")

        self.initialized = True

    def update(self, ship, world):
        """
        Wird jedes Frame aufgerufen.
        """

        if not self.initialized:
            self.initialize(ship, world)
            return

        # remove points the ship already passed
        removed = self.remove_passed_points(ship)
        if self.debug:
            print(f"[Predictor DEBUG] update: points_after_removal={len(self.points)} removed_passed={removed}")

        # Falls Schiff stark vom Predictor abweicht → neu initialisieren
        if self.points and math.sqrt(self.points[0].distance_squared_to(ship.position)) > self.spacing * 2:
            if self.debug:
                print("[Predictor DEBUG] update: ship deviated — reinitializing")
            self.initialize(ship, world)
            return

        # ensure a small window of points exists
        sim_steps_min = 0
        appended_min = 0
        while len(self.points) < self.min_points and len(self.points) < self.num_points:
            prev_pos = self.sim_pos.copy()
            self._verlet_step(world)
            sim_steps_min += 1
            step2 = self.sim_pos.distance_squared_to(prev_pos)
            step_distance = math.sqrt(step2)
            self.accumulated_distance += step_distance
            if self.accumulated_distance >= self.spacing:
                self.points.append(self.sim_pos.copy())
                self.accumulated_distance = 0.0
                appended_min += 1

        if self.debug and appended_min:
            print(f"[Predictor DEBUG] update: min_fill simulated_steps={sim_steps_min} appended={appended_min}")

        # only extend the tail when the ship approaches the last point
        if self.points:
            last = self.points[-1]
            dist_to_last = math.sqrt(last.distance_squared_to(ship.position))
            if dist_to_last < self.tail_extend_threshold:
                if self.debug:
                    print(f"[Predictor DEBUG] update: close to tail dist_to_last={dist_to_last} threshold={self.tail_extend_threshold}")
                appended = 0
                sim_steps_tail = 0
                while appended < self.tail_extend_count and len(self.points) < self.num_points:
                    prev_pos = self.sim_pos.copy()
                    self._verlet_step(world)
                    sim_steps_tail += 1
                    step2 = self.sim_pos.distance_squared_to(prev_pos)
                    step_distance = math.sqrt(step2)
                    self.accumulated_distance += step_distance
                    if self.accumulated_distance >= self.spacing:
                        self.points.append(self.sim_pos.copy())
                        self.accumulated_distance = 0.0
                        appended += 1
                if self.debug:
                    print(f"[Predictor DEBUG] update: tail_extend simulated_steps={sim_steps_tail} appended={appended}")
            else:
                if self.debug:
                    print(f"[Predictor DEBUG] update: not close to tail (dist_to_last={dist_to_last}) — no extension")

    def get_points(self):
        return self.points

    def remove_passed_points(self, ship):
        if len(self.points) < 2:
            return 0

        removed = 0
        while len(self.points) > 1:
            d0 = self.points[0].distance_squared_to(ship.position)
            d1 = self.points[1].distance_squared_to(ship.position)

            if d1 < d0:
                self.points.pop(0)
                removed += 1
            else:
                break

        if self.debug and removed:
            print(f"[Predictor DEBUG] remove_passed_points: removed={removed} remaining={len(self.points)}")

        return removed

    def _verlet_step(self, world):
        dt = self.dt

        # RK4 Stage 1
        k1_a = self._compute_acceleration(self.sim_pos, world)
        k1_v = self.sim_vel.copy()

        # RK4 Stage 2
        pos_2 = self.sim_pos + k1_v * (dt / 2)
        vel_2 = self.sim_vel + k1_a * (dt / 2)
        k2_a = self._compute_acceleration(pos_2, world)
        k2_v = vel_2.copy()

        # RK4 Stage 3
        pos_3 = self.sim_pos + k2_v * (dt / 2)
        vel_3 = self.sim_vel + k2_a * (dt / 2)
        k3_a = self._compute_acceleration(pos_3, world)
        k3_v = vel_3.copy()

        # RK4 Stage 4
        pos_4 = self.sim_pos + k3_v * dt
        vel_4 = self.sim_vel + k3_a * dt
        k4_a = self._compute_acceleration(pos_4, world)
        k4_v = vel_4.copy()

        # Combine all stages (weighted average)
        self.sim_pos += (k1_v + 2*k2_v + 2*k3_v + k4_v) * (dt / 6)
        self.sim_vel += (k1_a + 2*k2_a + 2*k3_a + k4_a) * (dt / 6)

    def _compute_acceleration(self, position, world):
        total_acc = Vec2(0.0, 0.0)

        G = world.G

        for body in world.body:
            if not body.fixed:
                continue

            direction = body.position - position
            distance_sq = direction.magnitude_squared()

            if distance_sq < 1e-12:
                continue

            inv_dist = 1.0 / (distance_sq ** 0.5)
            force_dir = direction * inv_dist

            acc_mag = G * body.mass / distance_sq

            total_acc += force_dir * acc_mag

        return total_acc
