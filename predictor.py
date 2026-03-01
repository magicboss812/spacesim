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

        accumulated_distance = 0.0
        last_point = self.sim_pos.copy()

        while len(self.points) < self.num_points:
            prev_pos = self.sim_pos.copy()
            self._verlet_step(world)

            step2 = self.sim_pos.distance_squared_to(prev_pos)
            step_distance = math.sqrt(step2) 
            accumulated_distance += step_distance

            if accumulated_distance >= self.spacing:
                self.points.append(self.sim_pos.copy())
                accumulated_distance = 0.0
                last_point = self.sim_pos.copy()

        self.initialized = True

    def update(self, ship, world):
        """
        Wird jedes Frame aufgerufen.
        """

        if not self.initialized:
            self.initialize(ship, world)
            return

        # Falls Schiff stark vom Predictor abweicht → neu initialisieren
        if self.points and math.sqrt(self.points[0].distance_squared_to(ship.position)) > self.spacing * 2:
            self.initialize(ship, world)
            return

        self._remove_passed_points(ship)

        accumulated_distance = 0.0
        while len(self.points) < self.num_points:
            prev_pos = self.sim_pos.copy()
            self._verlet_step(world)

            step2 = self.sim_pos.distance_squared_to(prev_pos)
            step_distance = math.sqrt(step2)
            accumulated_distance += step_distance

            if accumulated_distance >= self.spacing:
                self.points.append(self.sim_pos.copy())
                accumulated_distance = 0.0

    def get_points(self):
        return self.points

    def _remove_passed_points(self, ship):
        if len(self.points) < 2:
            return

        while len(self.points) > 1:
            d0 = self.points[0].distance_squared_to(ship.position)
            d1 = self.points[1].distance_squared_to(ship.position)

            if d1 < d0:
                self.points.pop(0)
            else:
                break

    def _verlet_step(self, world):
        dt = self.dt

        # Alte Beschleunigung
        acc = self._compute_acceleration(self.sim_pos, world)

        # Position
        self.sim_pos += self.sim_vel * dt + acc * (0.5 * dt * dt)

        # Neue Beschleunigung
        new_acc = self._compute_acceleration(self.sim_pos, world)

        # Velocity
        self.sim_vel += (acc + new_acc) * (0.5 * dt)

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
