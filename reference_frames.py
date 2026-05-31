"""
reference-frame-primitiven und selector/adapter-verkabelung für spacesim.

dies folgt derselben high-level aufteilung wie bei Principia:
- frame-parameter durch UI-logik ausgewählt,
- adapter wandelt parameter in konkrete frame-objekte um,
- renderer wendet transformationen an, physik bleibt im absoluten raum.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

from vec import Vec2, G as NEWTONIAN_G


BODY_CENTRED_NON_ROTATING = 6000
BODY_CENTRED_BODY_DIRECTION = 6002


@dataclass(frozen=True)
class PlottingFrameParameters:
    extension: int
    primary_index: int
    secondary_index: int | None = None


@dataclass(frozen=True)
class KeplerScriptedOrbit:
    """Hilfs-Kepler-Orbit, nur für die Visualisierungs-Frame-Logik."""

    semi_major_axis_m: float
    eccentricity: float
    argument_of_periapsis_rad: float

    def radius_m(self, true_anomaly_rad: float) -> float:
        a = float(self.semi_major_axis_m)
        e = float(self.eccentricity)
        nu = float(true_anomaly_rad)
        denom = 1.0 + e * math.cos(nu)
        if abs(denom) < 1e-12:
            denom = 1e-12 if denom >= 0.0 else -1e-12
        return a * (1.0 - e * e) / denom

    def perifocal_xy(self, true_anomaly_rad: float) -> tuple[float, float]:
        nu = float(true_anomaly_rad)
        r = self.radius_m(nu)
        return r * math.cos(nu), r * math.sin(nu)

    def inertial_xy(self, true_anomaly_rad: float) -> tuple[float, float]:
        x_p, y_p = self.perifocal_xy(true_anomaly_rad)
        return _rotate_xy(x_p, y_p, -float(self.argument_of_periapsis_rad))


def _rotate_xy(x_m: float, y_m: float, angle_rad: float) -> tuple[float, float]:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return c * x_m + s * y_m, -s * x_m + c * y_m


def _world_to_frame_xy(
    world_x: float,
    world_y: float,
    origin_x: float,
    origin_y: float,
    frame_x_axis_angle_rad: float,
) -> tuple[float, float]:
    rel_x = float(world_x) - float(origin_x)
    rel_y = float(world_y) - float(origin_y)
    return _rotate_xy(rel_x, rel_y, float(frame_x_axis_angle_rad))


def _heading_world_to_frame(theta_world: float, frame_x_axis_angle_rad: float) -> float:
    hx = math.cos(float(theta_world))
    hy = math.sin(float(theta_world))
    fx, fy = _rotate_xy(hx, hy, float(frame_x_axis_angle_rad))
    return math.atan2(fy, fx)


def _has_scripted_orbit_data(body) -> bool:
    try:
        a = float(getattr(body, "semi_major_axis", 0.0) or 0.0)
        e = float(getattr(body, "eccentricity", 0.0) or 0.0)
    except Exception:
        return False
    return a > 0.0 and 0.0 <= e < 1.0


def _body_true_anomaly(body) -> float:
    # Loader/code versions use theta, theta0, or true_anomaly. Prefer live theta.
    for attr in ("theta", "true_anomaly", "theta0"):
        try:
            value = getattr(body, attr, None)
            if value is not None:
                return float(value)
        except Exception:
            pass
    return 0.0


def _body_arg_periapsis(body) -> float:
    for attr in ("arg_periapsis", "argument_of_periapsis"):
        try:
            value = getattr(body, attr, None)
            if value is not None:
                return float(value)
        except Exception:
            pass
    return 0.0


def _solve_eccentric_anomaly(M: float, e: float) -> float:
    E = M
    for _ in range(50):
        dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < 1e-10:
            break
    return E


def _kepler_true_anomaly_from_mean(M: float, e: float) -> float:
    E = _solve_eccentric_anomaly(M, e)
    return math.atan2(math.sqrt(max(0.0, 1.0 - e * e)) * math.sin(E), math.cos(E) - e)


def _mean_anomaly_from_true(nu: float, e: float) -> float:
    cos_nu = math.cos(nu)
    cos_E = (e + cos_nu) / (1.0 + e * cos_nu)
    cos_E = max(-1.0, min(1.0, cos_E))
    sin_E = math.sqrt(max(0.0, 1.0 - cos_E * cos_E)) * (1.0 if math.sin(nu) >= 0.0 else -1.0)
    E = math.atan2(sin_E, cos_E)
    return E - e * math.sin(E)


def _build_kepler_elements(body, mu: float) -> dict | None:
    try:
        a_m = float(getattr(body, "semi_major_axis", 0.0) or 0.0)
        e = float(getattr(body, "eccentricity", 0.0) or 0.0)
        if a_m <= 0.0 or e < 0.0 or e >= 1.0:
            return None
        nu0 = _body_true_anomaly(body)
        arg = _body_arg_periapsis(body)
        n = math.sqrt(float(mu) / (a_m ** 3))
        M0 = _mean_anomaly_from_true(nu0, e)
        return {"a": a_m, "e": e, "arg": arg, "n": n, "M0": M0}
    except Exception:
        return None


def _orbit_model_from_body(body) -> KeplerScriptedOrbit | None:
    try:
        a = float(getattr(body, "semi_major_axis", 0.0) or 0.0)
        e = float(getattr(body, "eccentricity", 0.0) or 0.0)
        arg = _body_arg_periapsis(body)
    except Exception:
        return None
    if a <= 0.0 or e < 0.0 or e >= 1.0:
        return None
    return KeplerScriptedOrbit(
        semi_major_axis_m=a,
        eccentricity=e,
        argument_of_periapsis_rad=arg,
    )


class ReferenceFrame:
    label = "Barycentric"

    def set_epoch_time(self, time_s: float) -> None:
        return

    def set_origin_interp_window(self, t0: float, t1: float, sample_count: int = 0) -> None:
        # No-op auf dem basis-frame (Identity ist ohnehin O(1)). Bewegte
        # origin-frames überschreiben dies, um die origin-position über das
        # zeitfenster zu interpolieren statt pro punkt zu propagieren.
        return

    def to_this_frame_xy(self, time_s: float, x: float, y: float) -> tuple[float, float]:
        return float(x), float(y)

    def to_this_frame_at_time(self, time_s: float, position: Vec2) -> Vec2:
        px, py = self.to_this_frame_xy(time_s, position.x, position.y)
        return Vec2(px, py)

    def to_this_frame_vector_xy(self, time_s: float, vx: float, vy: float) -> tuple[float, float]:
        return float(vx), float(vy)

    def from_this_frame_vector_xy(self, time_s: float, vx: float, vy: float) -> tuple[float, float]:
        return float(vx), float(vy)

    def transform_heading(self, time_s: float, theta_world: float) -> float:
        return float(theta_world)

    def heading_from_this_frame(self, time_s: float, theta_frame: float) -> float:
        vx = math.cos(float(theta_frame))
        vy = math.sin(float(theta_frame))
        wx, wy = self.from_this_frame_vector_xy(time_s, vx, vy)
        return math.atan2(wy, wx)


class IdentityReferenceFrame(ReferenceFrame):
    label = "Barycentric"


class _BodyEphemerisMixin:
    # zeitabfragen für gecachte ephemeris-positionen quantisieren, um predictor-rendering
    # glatt zu halten und teure pro-punkt-propagationsaufrufe zu vermeiden.
    frame_time_quantization_s = 0

    # Max. anzahl exakter origin-stützstellen (knots) pro render, zwischen denen
    # die origin-position linear interpoliert wird. Begrenzt teure propagations-
    # aufrufe bei bewegten origin-körpern (z.B. Erde) auf O(knots) statt
    # O(predictor-punkte). Nur aktiv, wenn der renderer ein zeitfenster mit mehr
    # punkten als knots setzt — sonst exakt → nie langsamer als zuvor.
    frame_origin_interp_max_knots = 256

    def _init_ephemeris(self) -> None:
        self._epoch_time_s = 0.0
        self._epoch_initialized = False
        self._position_cache = {}
        self._relative_state_cache = {}
        self._angle_cache = {}
        self._virtual_pos_cache = {}
        self.debug_ephemeris = False
        self._debug_ephemeris_counter = 0
        # origin-interpolation: q<=0 bedeutet "exakt" (deaktiviert).
        self._origin_interp_q = 0.0
        self._origin_interp_t0 = 0.0

    def set_epoch_time(self, time_s: float) -> None:
        try:
            epoch = float(time_s)
        except Exception:
            epoch = 0.0

        if self._epoch_initialized and abs(epoch - self._epoch_time_s) <= 1e-12:
            return

        self._epoch_time_s = epoch
        self._epoch_initialized = True
        self._position_cache = {}
        self._relative_state_cache = {}
        self._angle_cache = {}
        self._virtual_pos_cache = {}

    def set_origin_interp_window(self, t0: float, t1: float, sample_count: int = 0) -> None:
        # Aktiviert lineare interpolation der origin-position zwischen gleichmäßig
        # über [t0, t1] verteilten knots — aber nur, wenn mehr punkte als knots
        # projiziert werden (sonst wäre exakt günstiger). q<=0 => exakt.
        try:
            a = float(t0)
            b = float(t1)
        except Exception:
            self._origin_interp_q = 0.0
            return
        span = b - a
        knots = max(1, int(self.frame_origin_interp_max_knots))
        if (not math.isfinite(span)) or span <= 0.0 or int(sample_count) <= knots:
            self._origin_interp_q = 0.0
            return
        self._origin_interp_t0 = a
        self._origin_interp_q = span / knots

    def _quantized_time(self, time_s: float) -> float:
        try:
            t = float(time_s)
        except Exception:
            return 0.0

        quantum = float(getattr(self, "frame_time_quantization_s", 0.0) or 0.0)
        if quantum <= 0.0:
            return t
        return round(t / quantum) * quantum

    def _body_world_position_at_time(self, body, time_s: float, stack: set[int] | None = None) -> tuple[float, float]:
        # Interpolierender wrapper: bei aktivem zeitfenster (q>0) wird die origin-
        # position zwischen zwei exakten knots linear interpoliert statt pro punkt
        # propagiert. stack gesetzt => rekursiver elternaufruf, der exakt bleiben
        # muss. q<=0 => exakt (identisches verhalten wie zuvor, nie langsamer).
        q = self._origin_interp_q
        if q <= 0.0 or body is None or stack is not None:
            return self._body_world_position_exact(body, time_s, stack)
        t = float(time_s)
        n = math.floor((t - self._origin_interp_t0) / q)
        klo = self._origin_interp_t0 + n * q
        khi = klo + q
        xlo, ylo = self._body_world_position_exact(body, klo, None)
        xhi, yhi = self._body_world_position_exact(body, khi, None)
        frac = (t - klo) / q
        if frac <= 0.0:
            return xlo, ylo
        if frac >= 1.0:
            return xhi, yhi
        return (xlo + (xhi - xlo) * frac, ylo + (yhi - ylo) * frac)

    def _body_world_position_exact(self, body, time_s: float, stack: set[int] | None = None) -> tuple[float, float]:
        if body is None:
            return 0.0, 0.0

        qt = self._quantized_time(time_s)
        cache_key = (id(body), qt)
        if cache_key in self._position_cache:
            return self._position_cache[cache_key]

        if stack is None:
            stack = set()
        body_id = id(body)
        if body_id in stack:
            return float(body.position.x), float(body.position.y)

        stack.add(body_id)

        parent = getattr(body, "is_moon_of", None)
        dt = qt - float(self._epoch_time_s)

        # Predictor samples are future absolute positions. In a body-centred plotting
        # frame, the origin body must also be evaluated at the same future sample time.
        # Freezing a scripted body at the current epoch causes geocentric predictor artifacts.
        if parent is None:
            px = float(body.position.x)
            py = float(body.position.y)

            scripted_pos = None
            if getattr(body, "scripted_orbit", False) or _has_scripted_orbit_data(body):
                scripted_pos = self._scripted_top_level_position_at_time(body, dt)

            if scripted_pos is not None:
                wx, wy = scripted_pos
            elif not getattr(body, "scripted_orbit", False) and not getattr(body, "fixed", False):
                try:
                    vx = float(body.velocity.x)
                    vy = float(body.velocity.y)
                except Exception:
                    vx = 0.0
                    vy = 0.0
                wx = px + vx * dt
                wy = py + vy * dt
            else:
                wx = px
                wy = py
        else:
            parent_x, parent_y = self._body_world_position_exact(parent, qt, stack)
            rel_x, rel_y = self._relative_position_to_parent_at_time(body, parent, dt)
            wx = parent_x + rel_x
            wy = parent_y + rel_y

        stack.remove(body_id)
        self._position_cache[cache_key] = (float(wx), float(wy))
        return float(wx), float(wy)

    def _scripted_top_level_position_at_time(self, body, dt_s: float) -> tuple[float, float] | None:
        """Visual-only propagation for parentless scripted bodies around world origin.

        This is a fallback for top-level orbital elements. It does not mutate the body and
        does not affect physics. Child orbits still use `_relative_position_to_parent_at_time`.
        """
        if not _has_scripted_orbit_data(body):
            return None

        try:
            a = float(getattr(body, "semi_major_axis", 0.0) or 0.0)
            e = float(getattr(body, "eccentricity", 0.0) or 0.0)
            nu0 = _body_true_anomaly(body)
            arg = _body_arg_periapsis(body)
        except Exception:
            return None

        if a <= 0.0 or e < 0.0 or e >= 1.0:
            return None

        n = None
        for attr in ("mean_motion", "angular_velocity", "orbit_angular_velocity"):
            try:
                value = getattr(body, attr, None)
                if value is not None:
                    n = float(value)
                    break
            except Exception:
                pass

        if n is None:
            for attr in ("orbital_period", "period", "orbit_period"):
                try:
                    value = getattr(body, attr, None)
                    if value is not None and float(value) > 0.0:
                        n = 2.0 * math.pi / float(value)
                        break
                except Exception:
                    pass

        if n is None:
            central_mass = None
            for attr in ("central_mass", "parent_mass", "primary_mass"):
                try:
                    value = getattr(body, attr, None)
                    if value is not None and float(value) > 0.0:
                        central_mass = float(value)
                        break
                except Exception:
                    pass
            if central_mass is None:
                # In this project, top-level scripted planets are usually intended to orbit
                # the world-origin Sun. Without access to the body list from the frame object,
                # use solar mass as a visual fallback only.
                central_mass = 1.989e30
            try:
                n = math.sqrt(NEWTONIAN_G * central_mass / (a * a * a))
            except Exception:
                n = None

        if n is None or not math.isfinite(n):
            return None

        nu = nu0 + float(n) * float(dt_s)
        denom = 1.0 + e * math.cos(nu)
        if abs(denom) < 1e-12:
            return None
        r = a * (1.0 - e * e) / denom
        x_orb = r * math.cos(nu)
        y_orb = r * math.sin(nu)
        c = math.cos(arg)
        s = math.sin(arg)
        wx = x_orb * c - y_orb * s
        wy = x_orb * s + y_orb * c

        try:
            if getattr(self, "debug_ephemeris", False):
                self._debug_ephemeris_counter += 1
                if self._debug_ephemeris_counter <= 5 or self._debug_ephemeris_counter % 250 == 0:
                    print(
                        f"FRAME_EPHEMERIS_DBG: body={getattr(body, 'name', '?')} "
                        f"dt={float(dt_s):.3f} pos=({wx:.6e},{wy:.6e}) mode=top_level_scripted"
                    )
        except Exception:
            pass

        return float(wx), float(wy)

    def _relative_position_to_parent_at_time(self, body, parent, dt_s: float) -> tuple[float, float]:
        state = self._relative_epoch_state(body, parent)
        rel0_x, rel0_y = state["rel0_m"]
        relv_x, relv_y = state["relv_m_s"]

        if state["use_kepler"]:
            try:
                kep = state["kepler_elements"]
                M = kep["M0"] + kep["n"] * float(dt_s)
                nu = _kepler_true_anomaly_from_mean(M, kep["e"])
                p = kep["a"] * (1.0 - kep["e"] * kep["e"])
                denom = 1.0 + kep["e"] * math.cos(nu)
                if abs(denom) > 1e-12 and p > 0.0:
                    r = p / denom
                    c = math.cos(kep["arg"])
                    s = math.sin(kep["arg"])
                    x_orb = r * math.cos(nu)
                    y_orb = r * math.sin(nu)
                    return x_orb * c - y_orb * s, x_orb * s + y_orb * c
            except Exception:
                pass

        return rel0_x + relv_x * dt_s, rel0_y + relv_y * dt_s

    def _relative_epoch_state(self, body, parent):
        state_key = (id(body), id(parent))
        cached = self._relative_state_cache.get(state_key)
        if cached is not None:
            return cached

        rel0_x = float(body.position.x) - float(parent.position.x)
        rel0_y = float(body.position.y) - float(parent.position.y)

        try:
            relv_x = float(body.velocity.x) - float(parent.velocity.x)
            relv_y = float(body.velocity.y) - float(parent.velocity.y)
        except Exception:
            relv_x = 0.0
            relv_y = 0.0

        state = {
            "rel0_m": (rel0_x, rel0_y),
            "relv_m_s": (relv_x, relv_y),
            "use_kepler": False,
            "kepler_elements": None,
        }

        scripted_state = self._scripted_relative_state_from_elements(body, parent)
        if scripted_state is not None:
            s_rel_x, s_rel_y, s_rel_vx, s_rel_vy, mu = scripted_state
            if float(mu) > 0.0:
                kep = _build_kepler_elements(body, float(mu))
                if kep is not None:
                    state = {
                        "rel0_m": (s_rel_x, s_rel_y),
                        "relv_m_s": (s_rel_vx, s_rel_vy),
                        "use_kepler": True,
                        "kepler_elements": kep,
                    }

        self._relative_state_cache[state_key] = state
        return state

    def _scripted_relative_state_from_elements(self, body, parent):
        # Do not rely only on `scripted_orbit`. Some loader versions mark orbital
        # bodies through semi_major_axis/eccentricity/is_moon_of without setting that flag.
        if not (getattr(body, "scripted_orbit", False) or _has_scripted_orbit_data(body)):
            return None

        try:
            a = float(getattr(body, "semi_major_axis", 0.0) or 0.0)
            e = float(getattr(body, "eccentricity", 0.0) or 0.0)
            nu = _body_true_anomaly(body)
            arg = _body_arg_periapsis(body)
            parent_mass = float(getattr(parent, "mass", 0.0) or 0.0)
        except Exception:
            return None

        if a <= 0.0 or parent_mass <= 0.0 or e < 0.0 or e >= 1.0:
            return None

        mu = NEWTONIAN_G * parent_mass
        if mu <= 0.0:
            return None

        p = a * (1.0 - e * e)
        if p <= 0.0:
            return None

        denom = 1.0 + e * math.cos(nu)
        if abs(denom) < 1e-12:
            return None

        r = p / denom
        x_orb = r * math.cos(nu)
        y_orb = r * math.sin(nu)

        h = math.sqrt(mu * p)
        if h <= 0.0:
            return None

        v_r = (mu / h) * e * math.sin(nu)
        v_t = (mu / h) * (1.0 + e * math.cos(nu))

        vx_orb = v_r * math.cos(nu) - v_t * math.sin(nu)
        vy_orb = v_r * math.sin(nu) + v_t * math.cos(nu)

        c = math.cos(arg)
        s = math.sin(arg)

        rel_x = x_orb * c - y_orb * s
        rel_y = x_orb * s + y_orb * c
        rel_vx = vx_orb * c - vy_orb * s
        rel_vy = vx_orb * s + vy_orb * c

        return rel_x, rel_y, rel_vx, rel_vy, mu


class BodyCentredNonRotatingReferenceFrame(_BodyEphemerisMixin, ReferenceFrame):
    def __init__(self, primary_body):
        self._init_ephemeris()
        self.primary_body = primary_body
        self.label = f"Body-centred non-rotating ({getattr(primary_body, 'name', '?')})"

    def to_this_frame_xy(self, time_s: float, x: float, y: float) -> tuple[float, float]:
        origin_x, origin_y = self._body_world_position_at_time(self.primary_body, time_s)
        return (float(x) - origin_x, float(y) - origin_y)


class VirtualBodyCentredNonRotatingReferenceFrame(_BodyEphemerisMixin, ReferenceFrame):
    """Ein nicht-rotierender Rahmen, dessen Primärposition virtuell
    aus einem scripted child (Mond) berechnet wird. Dies implementiert
    einen rein visuellen "orbit-swap", bei dem ein oberer fixer Körper
    so dargestellt wird, als würde er seinen scripted-Mond umkreisen,
    ohne den Physikzustand zu verändern.
    """

    def __init__(self, primary_body, child_body):
        self._init_ephemeris()
        self.primary_body = primary_body
        self.child_body = child_body
        self.label = f"Virtual-swap ({getattr(primary_body, 'name', '?')} <- {getattr(child_body, 'name', '?')})"

    def _virtual_primary_pos(self, time_s: float):
        t = float(time_s)
        cached = self._virtual_pos_cache.get(t)
        if cached is not None:
            return cached

        orbit = _orbit_model_from_body(self.child_body)
        if orbit is None:
            p_x, p_y = self._body_world_position_at_time(self.primary_body, time_s)
            vp = Vec2(float(p_x), float(p_y))
            self._virtual_pos_cache[t] = vp
            return vp

        theta_child = _body_true_anomaly(self.child_body)
        try:
            parent = getattr(self.child_body, "is_moon_of", None)
            if parent is not None:
                child_x, child_y = self._body_world_position_at_time(self.child_body, time_s)
                parent_x, parent_y = self._body_world_position_at_time(parent, time_s)
                rel_x = child_x - parent_x
                rel_y = child_y - parent_y
                arg = _body_arg_periapsis(self.child_body)
                theta_child = math.atan2(rel_y, rel_x) - arg
        except Exception:
            pass

        rel_x, rel_y = orbit.inertial_xy(theta_child + math.pi)
        child_x, child_y = self._body_world_position_at_time(self.child_body, time_s)
        vp = Vec2(float(child_x) + rel_x, float(child_y) + rel_y)
        self._virtual_pos_cache[t] = vp
        return vp

    def to_this_frame_xy(self, time_s: float, x: float, y: float) -> tuple[float, float]:
        vp = self._virtual_primary_pos(time_s)
        return float(x) - float(vp.x), float(y) - float(vp.y)


class BodyCentredBodyDirectionReferenceFrame(_BodyEphemerisMixin, ReferenceFrame):
    def __init__(self, primary_body, secondary_body):
        self._init_ephemeris()
        self.primary_body = primary_body
        self.secondary_body = secondary_body
        self.label = f"Body-direction ({getattr(primary_body, 'name', '?')} -> {getattr(secondary_body, 'name', '?')})"

    def _x_axis_angle(self, time_s: float) -> float:
        primary_x, primary_y = self._body_world_position_at_time(self.primary_body, time_s)
        secondary_x, secondary_y = self._body_world_position_at_time(self.secondary_body, time_s)
        dx = secondary_x - primary_x
        dy = secondary_y - primary_y
        norm2 = dx * dx + dy * dy
        if norm2 <= 1e-30:
            return 0.0
        return math.atan2(dy, dx)

    def _prepare_cache(self, time_s: float) -> None:
        cache_time = self._quantized_time(time_s)
        entry = self._angle_cache.get(cache_time)
        if entry is not None:
            self._cache_cos, self._cache_sin, self._cache_origin_x, self._cache_origin_y = entry
            self._cache_time = cache_time
            return
        angle = self._x_axis_angle(cache_time)
        origin_x, origin_y = self._body_world_position_at_time(self.primary_body, cache_time)
        self._cache_cos = math.cos(angle)
        self._cache_sin = math.sin(angle)
        self._cache_origin_x = origin_x
        self._cache_origin_y = origin_y
        self._cache_time = cache_time
        self._angle_cache[cache_time] = (self._cache_cos, self._cache_sin, origin_x, origin_y)

    def to_this_frame_xy(self, time_s: float, x: float, y: float) -> tuple[float, float]:
        self._prepare_cache(time_s)
        rel_x = float(x) - self._cache_origin_x
        rel_y = float(y) - self._cache_origin_y
        rx = self._cache_cos * rel_x - self._cache_sin * rel_y
        ry = self._cache_sin * rel_x + self._cache_cos * rel_y
        return rx, ry

    def transform_heading(self, time_s: float, theta_world: float) -> float:
        hx = math.cos(float(theta_world))
        hy = math.sin(float(theta_world))
        fx, fy = self.to_this_frame_vector_xy(time_s, hx, hy)
        return math.atan2(fy, fx)

    def to_this_frame_vector_xy(self, time_s: float, vx: float, vy: float) -> tuple[float, float]:
        self._prepare_cache(time_s)
        rx = self._cache_cos * float(vx) - self._cache_sin * float(vy)
        ry = self._cache_sin * float(vx) + self._cache_cos * float(vy)
        return rx, ry

    def from_this_frame_vector_xy(self, time_s: float, vx: float, vy: float) -> tuple[float, float]:
        self._prepare_cache(time_s)
        wx = self._cache_cos * float(vx) + self._cache_sin * float(vy)
        wy = -self._cache_sin * float(vx) + self._cache_cos * float(vy)
        return wx, wy


class TargetBodyDirectionReferenceFrame(_BodyEphemerisMixin, ReferenceFrame):
    def __init__(self, target_body, reference_body):
        self._init_ephemeris()
        self.target_body = target_body
        self.reference_body = reference_body
        self.label = f"Target overlay ({getattr(target_body, 'name', '?')} vs {getattr(reference_body, 'name', '?')})"

    def _x_axis_angle(self, time_s: float) -> float:
        target_x, target_y = self._body_world_position_at_time(self.target_body, time_s)
        reference_x, reference_y = self._body_world_position_at_time(self.reference_body, time_s)
        dx = reference_x - target_x
        dy = reference_y - target_y
        norm2 = dx * dx + dy * dy
        if norm2 <= 1e-30:
            return 0.0
        return math.atan2(dy, dx)

    def _prepare_cache(self, time_s: float) -> None:
        cache_time = self._quantized_time(time_s)
        entry = self._angle_cache.get(cache_time)
        if entry is not None:
            self._cache_cos, self._cache_sin, self._cache_origin_x, self._cache_origin_y = entry
            self._cache_time = cache_time
            return
        angle = self._x_axis_angle(cache_time)
        origin_x, origin_y = self._body_world_position_at_time(self.target_body, cache_time)
        self._cache_cos = math.cos(angle)
        self._cache_sin = math.sin(angle)
        self._cache_origin_x = origin_x
        self._cache_origin_y = origin_y
        self._cache_time = cache_time
        self._angle_cache[cache_time] = (self._cache_cos, self._cache_sin, origin_x, origin_y)

    def to_this_frame_xy(self, time_s: float, x: float, y: float) -> tuple[float, float]:
        self._prepare_cache(time_s)
        rel_x = float(x) - self._cache_origin_x
        rel_y = float(y) - self._cache_origin_y
        rx = self._cache_cos * rel_x - self._cache_sin * rel_y
        ry = self._cache_sin * rel_x + self._cache_cos * rel_y
        return rx, ry

    def transform_heading(self, time_s: float, theta_world: float) -> float:
        hx = math.cos(float(theta_world))
        hy = math.sin(float(theta_world))
        fx, fy = self.to_this_frame_vector_xy(time_s, hx, hy)
        return math.atan2(fy, fx)

    def to_this_frame_vector_xy(self, time_s: float, vx: float, vy: float) -> tuple[float, float]:
        self._prepare_cache(time_s)
        rx = self._cache_cos * float(vx) - self._cache_sin * float(vy)
        ry = self._cache_sin * float(vx) + self._cache_cos * float(vy)
        return rx, ry

    def from_this_frame_vector_xy(self, time_s: float, vx: float, vy: float) -> tuple[float, float]:
        self._prepare_cache(time_s)
        wx = self._cache_cos * float(vx) + self._cache_sin * float(vy)
        wy = -self._cache_sin * float(vx) + self._cache_cos * float(vy)
        return wx, wy


def _resolve_body(index: int, bodies: Sequence[object]):
    idx = int(index)
    if idx < 0 or idx >= len(bodies):
        raise IndexError(f"Body index out of range: {idx}")
    return bodies[idx]


def _fallback_secondary_index(primary_index: int, bodies: Sequence[object]) -> int:
    if len(bodies) <= 1:
        return int(primary_index)
    for idx in range(len(bodies)):
        if idx != int(primary_index):
            return idx
    return int(primary_index)


def _find_virtual_swap_child(primary_body, bodies: Sequence[object]):
    """Gibt das scripted-kind zurück, das für einen rein visuellen orbit-swap verwendet wird, oder None."""
    try:
        has_orbit = getattr(primary_body, 'semi_major_axis', None) is not None and float(getattr(primary_body, 'semi_major_axis', 0.0)) > 0.0
    except Exception:
        has_orbit = False

    if has_orbit or (not getattr(primary_body, 'fixed', False)):
        return None

    candidate = None
    for child in bodies:
        if getattr(child, 'is_moon_of', None) is primary_body and (getattr(child, 'scripted_orbit', False) or _has_scripted_orbit_data(child)):
            try:
                if float(getattr(child, 'semi_major_axis', 0.0) or 0.0) > 0.0:
                    return child
            except Exception:
                pass
            if candidate is None:
                candidate = child
    return candidate


def resolve_plotting_camera_target_index(frame_parameters: PlottingFrameParameters, bodies: Sequence[object]) -> int:
    """Bestimmt, welchem körper die kamera für den ausgewählten plotting-frame folgen soll."""
    primary_index = int(frame_parameters.primary_index)
    return primary_index


def new_plotting_frame(frame_parameters: PlottingFrameParameters, bodies: Sequence[object]) -> ReferenceFrame:
    extension = int(frame_parameters.extension)
    primary = _resolve_body(frame_parameters.primary_index, bodies)

    if extension == BODY_CENTRED_NON_ROTATING:
        return BodyCentredNonRotatingReferenceFrame(primary)

    if extension == BODY_CENTRED_BODY_DIRECTION:
        secondary_index = frame_parameters.secondary_index
        if secondary_index is None:
            secondary_index = _fallback_secondary_index(frame_parameters.primary_index, bodies)
        secondary = _resolve_body(secondary_index, bodies)
        return BodyCentredBodyDirectionReferenceFrame(primary, secondary)

    return IdentityReferenceFrame()


def describe_plotting_frame(frame_parameters: PlottingFrameParameters, bodies: Sequence[object]) -> str:
    extension = int(frame_parameters.extension)
    primary = _resolve_body(frame_parameters.primary_index, bodies)
    primary_name = getattr(primary, "name", f"#{frame_parameters.primary_index}")

    if extension == BODY_CENTRED_NON_ROTATING:
        return f"Body-centred non-rotating ({primary_name})"

    if extension == BODY_CENTRED_BODY_DIRECTION:
        secondary_index = frame_parameters.secondary_index
        if secondary_index is None:
            secondary_index = _fallback_secondary_index(frame_parameters.primary_index, bodies)
        secondary = _resolve_body(secondary_index, bodies)
        secondary_name = getattr(secondary, "name", f"#{secondary_index}")
        return f"Body-direction ({primary_name} -> {secondary_name})"

    return "Barycentric"


FrameChangeCallback = Callable[[PlottingFrameParameters, int | None, int | None], None]


class ReferenceFrameSelector:
    def __init__(self, on_change: FrameChangeCallback | None = None):
        self._on_change = on_change
        self._frame_parameters = PlottingFrameParameters(
            extension=BODY_CENTRED_NON_ROTATING,
            primary_index=0,
            secondary_index=None,
        )
        self._target_body_index: int | None = None
        self._target_reference_index: int | None = None

    def set_frame_parameters(self, frame_parameters: PlottingFrameParameters) -> None:
        self._frame_parameters = frame_parameters

    def frame_parameters(self) -> PlottingFrameParameters:
        return self._frame_parameters

    def set_to_body_non_rotating(self, primary_index: int) -> None:
        self._target_body_index = None
        self._target_reference_index = None
        self._frame_parameters = PlottingFrameParameters(
            extension=BODY_CENTRED_NON_ROTATING,
            primary_index=int(primary_index),
            secondary_index=None,
        )
        self.effect_change()

    def set_to_body_direction(self, primary_index: int, secondary_index: int) -> None:
        self._target_body_index = None
        self._target_reference_index = None
        self._frame_parameters = PlottingFrameParameters(
            extension=BODY_CENTRED_BODY_DIRECTION,
            primary_index=int(primary_index),
            secondary_index=int(secondary_index),
        )
        self.effect_change()

    def set_target_frame(self, target_body_index: int, reference_body_index: int) -> None:
        self._target_body_index = int(target_body_index)
        self._target_reference_index = int(reference_body_index)
        self.effect_change()

    def clear_target_frame(self) -> None:
        self._target_body_index = None
        self._target_reference_index = None
        self.effect_change()

    def effect_change(self) -> None:
        if self._on_change is not None:
            self._on_change(self._frame_parameters, self._target_body_index, self._target_reference_index)


class PlottingFrameAdapter:
    def __init__(self, renderer, bodies: Sequence[object]):
        self._renderer = renderer
        self._bodies = bodies

    def update_plotting_frame(
        self,
        frame_parameters: PlottingFrameParameters,
        target_body_index: int | None = None,
        target_reference_index: int | None = None,
    ) -> None:
        base_frame = new_plotting_frame(frame_parameters, self._bodies)
        base_label = describe_plotting_frame(frame_parameters, self._bodies)
        self._renderer.set_plotting_frame(base_frame, label=base_label)

        if target_body_index is None:
            self._renderer.clear_target_frame()
            return

        reference_index = int(target_reference_index) if target_reference_index is not None else int(frame_parameters.primary_index)
        target_body = _resolve_body(int(target_body_index), self._bodies)
        reference_body = _resolve_body(reference_index, self._bodies)

        target_frame = TargetBodyDirectionReferenceFrame(target_body, reference_body)
        target_label = f"Target overlay ({getattr(target_body, 'name', '?')} vs {getattr(reference_body, 'name', '?')})"
        self._renderer.set_target_frame(target_frame, label=target_label)
