import math
from vec import Vec2
from bodies import body

try:
    import numpy as _np
    import world_kernels as _wk
    _KERNELS_OK = bool(_wk.NUMBA_AVAILABLE)
except Exception:                                    # pragma: no cover
    _np = None
    _wk = None
    _KERNELS_OK = False

G = 6.6730831e-11

class world:

    def __init__(self, G):
        self.G = G
        self.body = []
        self.time = 0.0
        self.integrator_max_step = 30.0
        self.integrator_min_step = 0.01
        # Zeitraffer-decke fuer die schrittweite. `integrator_max_step` ist die
        # UNTERGRENZE dieser decke -- in echtzeit bleibt sie exakt 30 s, der
        # integrator rechnet dort also dieselben floats wie bisher. Erst wenn
        # set_warp_step_ceiling() eine hoehere decke setzt, darf der schritt
        # groesser werden. Siehe dort fuer die messung, die das rechtfertigt.
        self.integrator_max_step_effective = 0.0
        self.integrator_warp_substep_target = 40.0
        self.integrator_max_step_ceiling = 1.0e6
        # Zuletzt angenommene schrittweite. Ohne dieses gedaechtnis faengt
        # jeder aufruf wieder bei der decke an und arbeitet sich per ablehnung
        # herunter -- mit angehobener decke ist das der groesste einzelposten
        # (gemessen 2000-km-orbit bei 1 y/s: 19810 ablehnungen, 629 ms/frame).
        # Nur aktiv, wenn die decke ueber der konfigurierten liegt, damit der
        # standardfall bit-identisch bleibt.
        self._integrator_h_hint = 0.0
        self.integrator_position_tolerance = 1.0
        self.integrator_velocity_tolerance = 0.001
        self.integrator_debug = False
        self.integrator_mode = "rkn4"  # "rkn4" | "verlet"
        self.integrator_last_substeps = 0
        self.integrator_last_rejections = 0
        self.integrator_last_min_step_forced = 0
        self.integrator_last_worst_pos_error = 0.0
        self.integrator_last_worst_vel_error = 0.0
        # Numba-fassung des integrators benutzen, wenn verfuegbar. Gleiche
        # formeln, gleiche toleranzen, gleiche reihenfolge -- nur ohne
        # Python-objekte (siehe world_kernels.py). Zum vergleichen der
        # beiden pfade auf False setzen; die energiedrift muss identisch
        # bleiben.
        self.use_fast_integrator = True
        # Epicycle (Ptolemaic) mode state. When enabled, top-level bodies
        # (those with no parent) will be reparented to the chosen center
        # body so that the resulting motion produces epicycles relative
        # to that center.
        self._epicycle_enabled = False
        self._epicycle_center = None
        self._epicycle_saved = {}

    def characteristic_timescale(self, ship):
        """Zeitskala der bahnbewegung um den DOMINIERENDEN koerper, in sekunden.

        Fuer eine kreisbahn ist sqrt(r/|g|) genau T/2pi -- die zeit, in der sich
        der geschwindigkeitsvektor nennenswert dreht. Sie ist die ehrliche
        obergrenze fuer einen zeitraffer-schritt: rueckt ein frame um mehr als
        einen bruchteil davon vor, ist die bahn nicht mehr aufgeloest, ganz
        gleich wie fein der integrator rechnet.

        Warum nicht die umlaufzeit aus dem HUD? Weil die gegen den vom SPIELER
        gewaehlten bezugskoerper gerechnet wird. Wer im erdorbit die Sonne als
        bezug einstellt, bekaeme dort ein jahr statt zwei stunden -- die grenze
        haenge dann an einer anzeigeeinstellung statt an der physik. Der
        dominierende koerper ist der mit dem groessten G*m/r^2, also der, der
        die bahn tatsaechlich bestimmt.

        Rueckgabe None, wenn es keinen dominierenden koerper gibt.
        """
        if ship is None:
            return None
        px = ship.position.x
        py = ship.position.y
        best_g = 0.0
        best_r2 = 0.0
        total_g = 0.0
        for b in self.body:
            if b is ship or getattr(b, 'is_ship', False):
                continue
            mass = float(getattr(b, 'mass', 0.0) or 0.0)
            if mass <= 0.0:
                continue
            dx = b.position.x - px
            dy = b.position.y - py
            r2 = dx * dx + dy * dy
            if r2 < 1e-6:
                continue
            g = self.G * mass / r2
            total_g += g
            if g > best_g:
                best_g = g
                best_r2 = r2
        if best_g <= 0.0 or total_g <= 0.0:
            return None
        return math.sqrt(math.sqrt(best_r2) / total_g)

    def set_warp_step_ceiling(self, sim_seconds_per_frame):
        """Decke fuer die integrator-schrittweite aus der raffung ableiten.

        Die kosten von update_dynamics sind LINEAR in der zahl der teilschritte,
        und die ist sim-sekunden-pro-frame / schrittweite. Eine feste decke von
        30 s heisst deshalb: bei 365 d/s 5984 teilschritte je frame, gemessen
        168.7 ms -- das 30-fache des frame-budgets.

        Statt die decke an der geometrie auszurichten (wie es der predictor mit
        rkn_adaptive_far_maxdt tut) wird hier direkt eine ZAHL VON TEILSCHRITTEN
        angepeilt. Das ist zulaessig, weil die schrittweite ohnehin nicht von der
        decke bestimmt wird, sondern von der fehlerkontrolle: gemessen in einem
        400-km-orbit aendert eine anhebung der decke von 30 s auf 100 000 s die
        TATSAECHLICHE schrittweite nur von 27.7 s auf 34.7 s (hoehendrift ueber
        5 umlaeufe: +0.374 km gegen +0.411 km). Nahe an einem koerper haelt also
        die toleranz die zuegel, die decke greift nur im fernfeld -- und dort
        kostet sie das 500-fache.

        Die decke ist damit eine UNTERGRENZE der schrittweite, keine obergrenze
        der genauigkeit: reicht sie nicht, lehnt die fehlerkontrolle ab und
        halbiert, es entstehen also mehr teilschritte als angepeilt. Genau so
        soll es sein.

        Gemessen bei 365 d/s (28 koerper, 180 fps): 168.7 ms -> 0.31 ms.
        """
        base = max(float(getattr(self, "integrator_max_step", 30.0)), 1e-9)
        span = abs(float(sim_seconds_per_frame))
        target = max(float(getattr(self, "integrator_warp_substep_target", 40.0)), 1.0)
        cap = max(float(getattr(self, "integrator_max_step_ceiling", 1.0e6)), base)
        self.integrator_max_step_effective = min(max(base, span / target), cap)
        return self.integrator_max_step_effective

    def effective_max_step(self):
        """Tatsaechlich benutzte decke -- nie kleiner als die konfigurierte."""
        base = max(float(getattr(self, "integrator_max_step", 30.0)), 1e-9)
        return max(base, float(getattr(self, "integrator_max_step_effective", 0.0)))

    def update_planets(self, dt):
        for body in self.body:
            # Überspringe Schiffe komplett - sie haben keine orbit_position
            if body.is_ship:
                continue
            if not body.scripted_orbit:
                continue
            parent_pos = body.is_moon_of.position if body.is_moon_of else None
            mu = self.G * body.is_moon_of.mass if body.is_moon_of else None

            # DIE EPOCHE GEHOERT ZUR REIHENFOLGE, UND DIE IST
            # `update_dynamics(dt)` DANN `update_planets(dt)`
            # (test.py::update). In dieser reihenfolge hat `self.time` bereits
            # das ende des chunks erreicht, wenn hier `body.theta` um genau
            # diesen chunk vorgeschrieben wird -- bookmark und winkel gehoeren
            # also zusammen, und der naechste `update_dynamics`-aufruf liest
            # sie richtig.
            #
            # DREHT MAN DIE REIHENFOLGE UM, IST ES FALSCH, und zwar
            # erster ordnung im chunk: `position_at_time(tau)` liefert dann
            # systematisch die position bei `tau + dt`, jeder geskriptete
            # koerper steht fuer die kraftrechnung des schiffs einen chunk in
            # der zukunft. Gemessen (erdumlaufbahn rp 2e7 m, e = 0.3, abstand
            # der welt von der analytisch propagierten predictor-linie nach
            # 4800 s):
            #
            #   chunk                       1000 s    300 s     5 s
            #   dynamics, planets (spiel)   5.2e1 m   5.2e1 m   5.2e1 m
            #   planets, dynamics           9.4e6 m   3.9e6 m   7.4e4 m
            #
            # Wer die aufrufe also vertauscht, verschiebt die bahn um
            # kilometer -- und weil der fehler mit dem chunk waechst, um so
            # mehr, je hoeher die raffung. `tests/warp_predictor_test.py`
            # tut genau das in seinem `advance()`-helfer; §18 misst es.
            body.position = body.orbit_position(dt, parent_pos, mu)
            body._kepler_ref_theta = body.theta
            body._kepler_ref_time = self.time

            # DANN prüfen ob Release nötig

        # Hinweis: Der Epizykel-Modus wird durch Umparenting der Top-Level-
        # Körper zum gewählten Zentrum via `enable_epicycles()` aktiviert; 
        # update_planets folgt einfach den aktuell gesetzten Elternbeziehungen.

    def _body_position_at_time(self, body, time_s):
        """
        Return the body's world position at a given simulation time.

        For now:
        - if the body has a time-aware orbit method later, use it here
        - otherwise fall back to current body.position

        This keeps the integrator ready for scripted moving planets at intermediate stages.
        """
        try:
            if hasattr(body, "position_at_time"):
                return body.position_at_time(time_s)
        except Exception:
            pass

        return body.position

    def acceleration_at(self, target_body, position, time_s=None):
        acc = Vec2(0.0, 0.0)

        if time_s is None:
            time_s = self.time

        for other in self.body:
            if other is target_body:
                continue

            try:
                other_pos = self._body_position_at_time(other, time_s)
            except Exception:
                other_pos = other.position

            delta = other_pos - position
            r2 = delta.magnitude_squared()
            if r2 < 1e-10:
                continue

            r = math.sqrt(r2)
            acc += delta * (self.G * other.mass / (r2 * r))

        return acc

    def _rkn4_step_body_state(self, body, p0, v0, t0, h):
        """
        One explicit RKN4-style step for r'' = a(r, t).

        p0: initial position
        v0: initial velocity
        t0: initial simulation time
        h: step size in seconds

        Returns:
            new_position, new_velocity
        """
        a1 = self.acceleration_at(body, p0, t0)

        p2 = p0 + v0 * (h * 0.5) + a1 * (h * h * 0.125)
        a2 = self.acceleration_at(body, p2, t0 + h * 0.5)

        p3 = p0 + v0 * (h * 0.5) + a1 * (h * h * 0.125)
        a3 = self.acceleration_at(body, p3, t0 + h * 0.5)

        p4 = p0 + v0 * h + a3 * (h * h * 0.5)
        a4 = self.acceleration_at(body, p4, t0 + h)

        new_p = p0 + v0 * h + (a1 + a2 + a3) * (h * h / 6.0)
        new_v = v0 + (a1 + 2.0 * a2 + 2.0 * a3 + a4) * (h / 6.0)

        return new_p, new_v

    def _verlet_step_body_state(self, body, p0, v0, t0, h):
        """Störmer-Verlet (KDK leapfrog) — 2nd-order symplectic RKN."""
        a0 = self.acceleration_at(body, p0, t0)
        p1 = p0 + v0 * h + a0 * (0.5 * h * h)
        a1 = self.acceleration_at(body, p1, t0 + h)
        v1 = v0 + (a0 + a1) * (0.5 * h)
        return p1, v1

    def _adaptive_rkn_step_body_state(
        self,
        body,
        p0,
        v0,
        t0,
        h,
        pos_tol=1.0,
        vel_tol=0.001,
    ):
        """
        Adaptive embedded RKN-style step by step-doubling.

        Compares:
        - one full step h
        - two half steps h/2

        Returns:
            accepted, new_position, new_velocity, pos_error, vel_error
        """
        step_fn = self._verlet_step_body_state if getattr(self, "integrator_mode", "rkn4") == "verlet" \
            else self._rkn4_step_body_state

        p_full, v_full = step_fn(body, p0, v0, t0, h)

        half = h * 0.5
        p_half, v_half = step_fn(body, p0, v0, t0, half)
        p_two, v_two = step_fn(body, p_half, v_half, t0 + half, half)

        pos_err = (p_two - p_full).magnitude()
        vel_err = (v_two - v_full).magnitude()

        accepted = pos_err <= pos_tol and vel_err <= vel_tol
        return accepted, p_two, v_two, pos_err, vel_err

    def _rv_to_orbital(self, r_vec, v_vec, mu):
        """konvertiert position/geschwindigkeit (relativ zum parent) in orbitale elementen.

        gibt (a, e, theta, arg_peri) zurück wobei theta die wahre anomalie ist gemessen
        vom periapsis und arg_peri das periapsis-argument (radian) ist.
        """
        r = r_vec.magnitude()
        v = v_vec.magnitude()
        if r <= 0.0 or mu is None or mu <= 0.0:
            return None

        # specific angular momentum (scalar z-component)
        h = r_vec.x * v_vec.y - r_vec.y * v_vec.x

        # specific energy
        eps = 0.5 * v * v - mu / r
        if abs(eps) < 1e-20:
            # Division durch Null vermeiden; Fallback: kreisförmige Bahn
            a = r
        else:
            a = -mu / (2.0 * eps)

        # eccentricity vector: (v x h)/mu - r_hat
        # v x h_vec (2D) => (h * v_y, -h * v_x)
        vxh_x = h * v_vec.y
        vxh_y = -h * v_vec.x
        evec_x = vxh_x / mu - r_vec.x / r
        evec_y = vxh_y / mu - r_vec.y / r
        e = math.sqrt(evec_x * evec_x + evec_y * evec_y)

        # argument of periapsis (direction of e-vector)
        arg_peri = math.atan2(evec_y, evec_x) if e > 1e-12 else 0.0

        # true anomaly measured from periapsis: angle(r) - arg_peri
        theta_world = math.atan2(r_vec.y, r_vec.x)
        theta = theta_world - arg_peri
        # normalize to 0..2pi
        theta = (theta + 2.0 * math.pi) % (2.0 * math.pi)

        return a, e, theta, arg_peri
    def update_dynamics(self, dt):
        self.integrator_last_substeps = 0
        self.integrator_last_rejections = 0
        self.integrator_last_min_step_forced = 0
        self.integrator_last_worst_pos_error = 0.0
        self.integrator_last_worst_vel_error = 0.0

        dynamic_bodies = [
            b for b in self.body
            if not getattr(b, "scripted_orbit", False) and not getattr(b, "fixed", False)
        ]

        total_dt = float(dt)
        if not dynamic_bodies:
            self.time += total_dt
            return

        direction = 1.0 if total_dt >= 0.0 else -1.0
        remaining = abs(total_dt)

        max_step = self.effective_max_step()
        min_step = max(float(getattr(self, "integrator_min_step", 0.01)), 1e-12)
        pos_tol = max(float(getattr(self, "integrator_position_tolerance", 1.0)), 1e-12)
        vel_tol = max(float(getattr(self, "integrator_velocity_tolerance", 0.001)), 1e-12)

        if self._advance_dynamics_fast(
            dynamic_bodies, total_dt, max_step, min_step, pos_tol, vel_tol
        ):
            self._log_integrator_debug(total_dt)
            self.time += total_dt
            return

        t = float(self.time)

        hint = self._step_hint(max_step)
        hint = max_step if hint <= 0.0 else min(hint, max_step)

        while remaining > 1e-12:
            h = min(hint, remaining) * direction

            while True:
                saved_states = [
                    (b, b.position.copy(), b.velocity.copy())
                    for b in dynamic_bodies
                ]

                accepted_all = True
                worst_pos_err = 0.0
                worst_vel_err = 0.0
                new_states = []

                for b, p0, v0 in saved_states:
                    accepted, p_new, v_new, pos_err, vel_err = self._adaptive_rkn_step_body_state(
                        b,
                        p0,
                        v0,
                        t,
                        h,
                        pos_tol=pos_tol,
                        vel_tol=vel_tol,
                    )

                    worst_pos_err = max(worst_pos_err, pos_err)
                    worst_vel_err = max(worst_vel_err, vel_err)

                    if not accepted and abs(h) > min_step:
                        accepted_all = False
                        break

                    new_states.append((b, p_new, v_new))

                self.integrator_last_worst_pos_error = max(self.integrator_last_worst_pos_error, worst_pos_err)
                self.integrator_last_worst_vel_error = max(self.integrator_last_worst_vel_error, worst_vel_err)

                if accepted_all:
                    for b, p_new, v_new in new_states:
                        b.position = p_new
                        b.velocity = v_new

                    self.integrator_last_substeps += 1
                    t += h
                    remaining -= abs(h)
                    hint = min(abs(h) * 2.0, max_step)
                    break

                for b, p_old, v_old in saved_states:
                    b.position = p_old
                    b.velocity = v_old

                self.integrator_last_rejections += 1
                h *= 0.5

                if abs(h) <= min_step:
                    step_fn = self._verlet_step_body_state if getattr(self, "integrator_mode", "rkn4") == "verlet" \
                        else self._rkn4_step_body_state
                    new_states = []
                    for b, p0, v0 in saved_states:
                        p_new, v_new = step_fn(b, p0, v0, t, h)
                        new_states.append((b, p_new, v_new))

                    for b, p_new, v_new in new_states:
                        b.position = p_new
                        b.velocity = v_new

                    self.integrator_last_substeps += 1
                    self.integrator_last_min_step_forced += 1
                    t += h
                    remaining -= abs(h)
                    hint = min(abs(h) * 2.0, max_step)
                    break

        self._store_step_hint(max_step, hint)
        self._log_integrator_debug(total_dt)

        self.time += total_dt

    def _log_integrator_debug(self, total_dt):
        if getattr(self, "integrator_debug", False):
            print(
                "INTEGRATOR_DBG: "
                f"dt={total_dt:.6g} "
                f"substeps={self.integrator_last_substeps} "
                f"rejections={self.integrator_last_rejections} "
                f"forced={self.integrator_last_min_step_forced} "
                f"worst_pos={self.integrator_last_worst_pos_error:.6e} "
                f"worst_vel={self.integrator_last_worst_vel_error:.6e}"
            )

    # ------------------------------------------------------ schneller pfad

    def _serialize_for_kernel(self, dynamic_bodies):
        """Koerperzustand in flache arrays. None = fuer den kernel ungeeignet.

        Abgelehnt wird, sobald ein `is_moon_of` auf einen koerper zeigt, der
        gar nicht in self.body steht: die python-fassung wuerde dessen
        position trotzdem addieren, der kernel kann sie nicht indizieren.
        Lieber den langsamen, aber sicher gleichwertigen weg gehen, als in
        so einem fall still etwas anderes zu rechnen.
        """
        bodies = self.body
        count = len(bodies)

        # Struktur-cache: massen, bahnelemente und eltern-verknuepfungen
        # aendern sich nur, wenn die KOERPERLISTE selbst umgebaut wird
        # (release_body, epizykel an/aus) -- und jede dieser aenderungen
        # aendert die id-tupel unten. Positionen und der Kepler-epoch-
        # bookmark wandern dagegen mit jedem weltschritt und werden pro
        # aufruf neu eingetragen. Vorher wurden ALLE elf arrays bei jedem
        # teilschritt neu gebaut (5-6x pro frame bei hohem zeitraffer).
        structure_key = (
            tuple(id(b) for b in bodies),
            tuple(id(getattr(b, "is_moon_of", None)) for b in bodies),
        )
        cached = getattr(self, "_kernel_static_cache", None)
        if cached is not None and cached[0] == structure_key:
            (_, index_of, bx, by, bm, k_has, k_a, k_e, k_arg, k_parent,
             k_ref_theta, k_ref_time, k_mu) = cached
        else:
            index_of = {id(b): i for i, b in enumerate(bodies)}

            bx = _np.empty(count, dtype=_np.float64)
            by = _np.empty(count, dtype=_np.float64)
            bm = _np.empty(count, dtype=_np.float64)
            k_has = _np.zeros(count, dtype=_np.int64)
            k_a = _np.zeros(count, dtype=_np.float64)
            k_e = _np.zeros(count, dtype=_np.float64)
            k_arg = _np.zeros(count, dtype=_np.float64)
            k_parent = _np.full(count, -1, dtype=_np.int64)
            k_ref_theta = _np.zeros(count, dtype=_np.float64)
            k_ref_time = _np.zeros(count, dtype=_np.float64)
            k_mu = _np.zeros(count, dtype=_np.float64)

            for i, b in enumerate(bodies):
                bm[i] = float(b.mass)

                parent = getattr(b, "is_moon_of", None)
                if parent is None:
                    continue
                parent_index = index_of.get(id(parent))
                if parent_index is None:
                    return None

                a = getattr(b, "semi_major_axis", None)
                if a is None or float(a) == 0.0:
                    continue
                mu = self.G * float(getattr(parent, "mass", 0.0) or 0.0)
                if mu <= 0.0:
                    continue

                # Genau die bedingungen, unter denen position_at_time rechnet
                # statt self.position zurueckzugeben.
                k_has[i] = 1
                k_a[i] = float(a)
                k_e[i] = float(getattr(b, "eccentricity", 0.0) or 0.0)
                k_arg[i] = float(getattr(b, "arg_periapsis", 0.0) or 0.0)
                k_parent[i] = parent_index
                k_mu[i] = mu

            self._kernel_static_cache = (
                structure_key, index_of, bx, by, bm, k_has, k_a, k_e, k_arg,
                k_parent, k_ref_theta, k_ref_time, k_mu,
            )

        for i, b in enumerate(bodies):
            bx[i] = float(b.position.x)
            by[i] = float(b.position.y)
            if k_has[i]:
                k_ref_theta[i] = float(getattr(b, "_kepler_ref_theta", 0.0) or 0.0)
                k_ref_time[i] = float(getattr(b, "_kepler_ref_time", 0.0) or 0.0)

        dyn = _np.array([index_of[id(b)] for b in dynamic_bodies],
                        dtype=_np.int64)
        dyn_px = _np.array([float(b.position.x) for b in dynamic_bodies])
        dyn_py = _np.array([float(b.position.y) for b in dynamic_bodies])
        dyn_vx = _np.array([float(b.velocity.x) for b in dynamic_bodies])
        dyn_vy = _np.array([float(b.velocity.y) for b in dynamic_bodies])

        return (bx, by, bm, k_has, k_a, k_e, k_arg, k_parent, k_ref_theta,
                k_ref_time, k_mu, dyn, dyn_px, dyn_py, dyn_vx, dyn_vy)

    def _step_hint(self, max_step):
        """Startweite fuer den naechsten aufruf -- 0 heisst 'bei der decke'.

        Das gedaechtnis wird NUR benutzt, wenn die decke ueber der
        konfigurierten liegt, also ausschliesslich im zeitraffer. Im standard-
        fall gibt es nichts zu gewinnen (dort wurden 0 ablehnungen gemessen),
        und so bleibt die schrittfolge dort garantiert bit-identisch --
        tests/energy_test.py muss weiter auf 6.4718e-04 landen.
        """
        base = max(float(getattr(self, "integrator_max_step", 30.0)), 1e-9)
        if max_step <= base * (1.0 + 1e-12):
            return 0.0
        return float(getattr(self, "_integrator_h_hint", 0.0))

    def _store_step_hint(self, max_step, hint):
        base = max(float(getattr(self, "integrator_max_step", 30.0)), 1e-9)
        if max_step <= base * (1.0 + 1e-12):
            self._integrator_h_hint = 0.0
        else:
            self._integrator_h_hint = float(hint)

    def _advance_dynamics_fast(self, dynamic_bodies, total_dt, max_step,
                               min_step, pos_tol, vel_tol):
        """True, wenn der kernel den schritt uebernommen hat."""
        if not (_KERNELS_OK and getattr(self, "use_fast_integrator", True)):
            return False
        try:
            packed = self._serialize_for_kernel(dynamic_bodies)
            if packed is None:
                return False
            (bx, by, bm, k_has, k_a, k_e, k_arg, k_parent, k_ref_theta,
             k_ref_time, k_mu, dyn, dyn_px, dyn_py, dyn_vx, dyn_vy) = packed

            mode = 1 if getattr(self, "integrator_mode", "rkn4") == "verlet" else 0
            hint = self._step_hint(max_step)
            (substeps, rejections, forced, worst_pos, worst_vel,
             hint_out) = _wk.advance_dynamics(
                dyn, dyn_px, dyn_py, dyn_vx, dyn_vy,
                float(self.time), float(total_dt),
                bx, by, bm, k_has, k_a, k_e, k_arg, k_parent, k_ref_theta,
                k_ref_time, k_mu, float(self.G), mode,
                float(max_step), float(min_step), float(pos_tol), float(vel_tol),
                float(hint),
            )
            self._store_step_hint(max_step, hint_out)
        except Exception:
            # Jeder fehler faellt auf die python-fassung zurueck; sie ist die
            # referenz, der kernel nur die schnelle uebersetzung davon.
            return False

        for i, b in enumerate(dynamic_bodies):
            b.position = Vec2(float(dyn_px[i]), float(dyn_py[i]))
            b.velocity = Vec2(float(dyn_vx[i]), float(dyn_vy[i]))

        self.integrator_last_substeps = int(substeps)
        self.integrator_last_rejections = int(rejections)
        self.integrator_last_min_step_forced = int(forced)
        self.integrator_last_worst_pos_error = float(worst_pos)
        self.integrator_last_worst_vel_error = float(worst_vel)
        return True

    def enable_epicycles(self, center):
        """epizykel-modus aktivieren mit wurzel in `center`.

        speichert den aktuellen eltern/orbit-zustand für alle körper und setzt dann
        jedes top-level körper (deren `is_moon_of` None ist) als child von `center`.
        der gespeicherte zustand wird so abgelegt, dass `disable_epicycles()`
        die ursprüngliche konfiguration wiederherstellen kann.
        """
        if center is None:
            return False

        # If already enabled with same center, do nothing
        if self._epicycle_enabled and self._epicycle_center is center:
            return True

        # If enabled with another center, restore first
        if self._epicycle_enabled:
            self.disable_epicycles()

        saved = {}
        for b in self.body:
            saved[b] = {
                'is_moon_of': getattr(b, 'is_moon_of', None),
                'semi_major_axis': getattr(b, 'semi_major_axis', None),
                'eccentricity': getattr(b, 'eccentricity', None),
                'scripted_orbit': getattr(b, 'scripted_orbit', False),
                'released': getattr(b, 'released', False),
                'theta': getattr(b, 'theta', 0.0),
                'arg_periapsis': getattr(b, 'arg_periapsis', 0.0),
                'position': b.position.copy() if hasattr(b, 'position') else None,
                'velocity': b.velocity.copy() if hasattr(b, 'velocity') else None,
            }

        # Apply epicycle reparenting
        for b in self.body:
            if b is center:
                # keep center stationary under scripted orbit
                b.is_moon_of = None
                b.semi_major_axis = 0.0
                b.eccentricity = 0.0
                b.scripted_orbit = True
                b.released = False
                continue

            orig_parent = saved[b]['is_moon_of']
            # Skip ships entirely (they should remain dynamic)
            if getattr(b, 'is_ship', False):
                continue

            # Only reparent bodies that were scripted_orbit originally (planetary
            # bodies defined with orbital elements). Do not change purely
            # dynamic bodies.
            if orig_parent is None and saved[b]['scripted_orbit']:
                # Berechne relatives r/v zum Zentrum und leite neue orbitale Elemente ab
                
                    rel_r = b.position - center.position
                    # If center has velocity attribute, use relative velocity, else assume 0
                    center_v = getattr(center, 'velocity', Vec2(0.0, 0.0))
                    rel_v = b.velocity - center_v if hasattr(b, 'velocity') else Vec2(0.0, 0.0)
                    mu = self.G * getattr(center, 'mass', 0.0)
                    elems = self._rv_to_orbital(rel_r, rel_v, mu)
                    if elems is not None:
                        a, e, theta_rel, arg_peri = elems
                        b.is_moon_of = center
                        b.semi_major_axis = float(max(0.0, a))
                        b.eccentricity = float(max(0.0, min(0.999999, e)))
                        b.theta = float(theta_rel)
                        b.arg_periapsis = float(arg_peri)
                        b.scripted_orbit = True
                        b.released = False
                    else:
                        # Fallback: setze kreisförmige Bahn mit dem aktuellen Abstand
                        try:
                            r = (b.position - center.position).magnitude()
                        except Exception:
                            r = float(getattr(b, 'semi_major_axis', 0.0) or 0.0)
                        b.is_moon_of = center
                        b.semi_major_axis = float(max(0.0, r))
                        b.eccentricity = 0.0
                        b.theta = math.atan2((b.position - center.position).y, (b.position - center.position).x)
                        b.arg_periapsis = 0.0
                        b.scripted_orbit = True
                        b.released = False
            else:
                # preserve moons' parent relationships and dynamic bodies; ensure
                # scripted bodies stay scripted if they were originally.
                if saved[b]['scripted_orbit']:
                    b.scripted_orbit = True

        self._epicycle_saved = saved
        self._epicycle_enabled = True
        self._epicycle_center = center
        return True

    def disable_epicycles(self):
        """gespeicherten zustand der körper wiederherstellen und epizykel-modus deaktivieren."""
        if not self._epicycle_enabled:
            return False

        for b in self.body:
            saved = self._epicycle_saved.get(b)
            if saved is None:
                continue
            b.is_moon_of = saved['is_moon_of']
            b.semi_major_axis = saved['semi_major_axis']
            b.eccentricity = saved['eccentricity']
            b.scripted_orbit = saved['scripted_orbit']
            b.released = saved['released']
            # restore angular state
            if 'theta' in saved:
                try:
                    b.theta = float(saved['theta'])
                except Exception:
                    pass
            if 'arg_periapsis' in saved:
                try:
                    b.arg_periapsis = float(saved['arg_periapsis'])
                except Exception:
                    pass

        self._epicycle_saved = {}
        self._epicycle_enabled = False
        self._epicycle_center = None
        return True

