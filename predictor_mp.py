import math
from multiprocessing import Process, Queue, Event
from vec import Vec2


def predictor_worker(input_queue, output_queue, stop_event):

    while not stop_event.is_set():

        if input_queue.empty():
            continue

        data = input_queue.get()

        if data is None:
            break

        (
            request_id,
            ship_pos,
            ship_vel,
            grav_bodies,
            G,
            num_points,
            distance_interval
        ) = data

        pos_x, pos_y = ship_pos
        vel_x, vel_y = ship_vel

        path_points = []
        path_points.append((pos_x, pos_y))

        accumulated_distance = 0.0
        last_x = pos_x
        last_y = pos_y
        dt = 60.0
        def compute_acc(px, py, grav_bodies, G):
            acc_x = 0.0
            acc_y = 0.0

            for (bx, by, mass, _, _) in grav_bodies:
                dx = bx - px
                dy = by - py

                r2 = dx * dx + dy * dy
                if r2 < 1e10:
                    continue

                r = math.sqrt(r2)
                inv_r3 = 1.0 / (r2 * r)
                factor = G * mass * inv_r3

                acc_x += dx * factor
                acc_y += dy * factor

            return acc_x, acc_y
        acc_x, acc_y = compute_acc(pos_x, pos_y, grav_bodies, G)
        stored_points = 1  # Startpunkt zählt
        max_iterations = 200000  # Sicherheitslimit

        iterations = 0

        while stored_points < num_points and iterations < max_iterations:

            iterations += 1

            if stop_event.is_set():
                return

            # --- Velocity Verlet ---

            new_x = pos_x + vel_x * dt + 0.5 * acc_x * dt * dt
            new_y = pos_y + vel_y * dt + 0.5 * acc_y * dt * dt

            new_acc_x, new_acc_y = compute_acc(new_x, new_y, grav_bodies, G)

            vel_x += 0.5 * (acc_x + new_acc_x) * dt
            vel_y += 0.5 * (acc_y + new_acc_y) * dt

            pos_x = new_x
            pos_y = new_y
            acc_x = new_acc_x
            acc_y = new_acc_y

            # --- Distanz prüfen ---

            dx_move = pos_x - last_x
            dy_move = pos_y - last_y
            dist = math.sqrt(dx_move * dx_move + dy_move * dy_move)

            accumulated_distance += dist

            if accumulated_distance >= distance_interval:
                path_points.append((pos_x, pos_y))
                stored_points += 1
                accumulated_distance = 0.0
                last_x = pos_x
                last_y = pos_y

        output_queue.put((request_id, path_points))


class PredictorMP:

    def __init__(self, num_points=50, distance_interval=100000000.0):
        self.num_points = num_points
        self.distance_interval = distance_interval

        self.input_queue = Queue()
        self.output_queue = Queue()
        self.request_id = 0
        self.latest_request_id = -1
        self.busy = False
        self.stop_event = Event()

        self.process = Process(
            target=predictor_worker,
            args=(self.input_queue, self.output_queue, self.stop_event),
            daemon=True
        )
        self.process.start()

        self.latest_points = []

    def predict_async(self, ship, world):
        if self.busy:
            return
        self.busy = True
        self.request_id += 1
        current_id = self.request_id
        grav_bodies = []
        for b in world.body:
            if b is ship:
                continue
            grav_bodies.append((
                b.position.x,
                b.position.y,
                b.mass,
                b.velocity.x,
                b.velocity.y
            ))

        self.input_queue.put((
            current_id,
            (ship.position.x, ship.position.y),
            (ship.velocity.x, ship.velocity.y),
            grav_bodies,
            world.G,
            self.num_points,
            self.distance_interval
        ))

    def update(self):
        while not self.output_queue.empty():
            request_id, raw_points = self.output_queue.get()
            if request_id >= self.request_id:
                self.latest_request_id = request_id
                self.latest_points = [Vec2(x, y) for (x, y) in raw_points]
            self.busy = False

    def get_points(self):
        return self.latest_points

    def shutdown(self):
        self.stop_event.set()
        self.input_queue.put(None)
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()