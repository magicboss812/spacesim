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

import numpy as np
from astropy import units as u
from astropy.coordinates import CartesianRepresentation
from astropy.coordinates.matrix_utilities import rotation_matrix
from poliastro.twobody.propagation import kepler as poliastro_kepler

from vec import Vec2, G as NEWTONIAN_G


BODY_CENTRED_NON_ROTATING = 6000
BODY_CENTRED_BODY_DIRECTION = 6002
_M_TO_KM = 1e-3
_KM_TO_M = 1e3


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
        a_q = float(self.semi_major_axis_m) * u.m
        e = float(self.eccentricity)
        nu = float(true_anomaly_rad)
        denom = 1.0 + e * math.cos(nu)
        if abs(denom) < 1e-12:
            denom = 1e-12 if denom >= 0.0 else -1e-12
        r_q = a_q * (1.0 - e * e) / denom
        return float(r_q.to_value(u.m))

    def perifocal_xy(self, true_anomaly_rad: float) -> tuple[float, float]:
        nu = float(true_anomaly_rad)
        r = self.radius_m(nu)
        return r * math.cos(nu), r * math.sin(nu)

    def inertial_xy(self, true_anomaly_rad: float) -> tuple[float, float]:
        x_p, y_p = self.perifocal_xy(true_anomaly_rad)
        # astropy rotation_matrix() mit axis='z' folgt einer left-hand-rule konvention;
        # nutze -arg_periapsis um die in-plane +arg_periapsis-rotation unserer
        # scripted-orbit-gleichungen abzugleichen.
        return _rotate_xy_with_astropy(x_p, y_p, -float(self.argument_of_periapsis_rad))


def _cartesian_xy_m(x_m: float, y_m: float) -> CartesianRepresentation:
    return CartesianRepresentation(float(x_m) * u.m, float(y_m) * u.m, 0.0 * u.m)


def _rotate_xy_with_astropy(x_m: float, y_m: float, angle_rad: float) -> tuple[float, float]:
    rep = _cartesian_xy_m(x_m, y_m)
    rot = rotation_matrix(float(angle_rad), "z", unit=u.rad)
    transformed = rep.transform(rot)
    return float(transformed.x.to_value(u.m)), float(transformed.y.to_value(u.m))


def _world_to_frame_xy(
    world_x: float,
    world_y: float,
    origin_x: float,
    origin_y: float,
    frame_x_axis_angle_rad: float,
) -> tuple[float, float]:
    rel_x = float(world_x) - float(origin_x)
    rel_y = float(world_y) - float(origin_y)
    return _rotate_xy_with_astropy(rel_x, rel_y, float(frame_x_axis_angle_rad))


def _heading_world_to_frame(theta_world: float, frame_x_axis_angle_rad: float) -> float:
    hx = math.cos(float(theta_world))
    hy = math.sin(float(theta_world))
    fx, fy = _rotate_xy_with_astropy(hx, hy, float(frame_x_axis_angle_rad))
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

    def to_this_frame_xy(self, time_s: float, x: float, y: float) -> tuple[float, float]:
        return float(x), float(y)

    def to_this_frame_at_time(self, time_s: float, position: Vec2) -> Vec2:
        px, py = self.to_this_frame_xy(time_s, position.x, position.y)
        return Vec2(px, py)

    def transform_heading(self, time_s: float, theta_world: float) -> float:
        return float(theta_world)


class IdentityReferenceFrame(ReferenceFrame):
    label = "Barycentric"


class _BodyEphemerisMixin:
    # zeitabfragen für gecachte ephemeris-positionen quantisieren, um predictor-rendering
    # glatt zu halten und teure pro-punkt-propagationsaufrufe zu vermeiden.
    frame_time_quantization_s = 600.0

    def _init_ephemeris(self) -> None:
        self._epoch_time_s = 0.0
        self._epoch_initialized = False
        self._position_cache = {}
        self._relative_state_cache = {}
        self.debug_ephemeris = False
        self._debug_ephemeris_counter = 0

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
            parent_x, parent_y = self._body_world_position_at_time(parent, qt, stack)
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
                r_km, _ = poliastro_kepler(
                    state["k_km3_s2"],
                    state["r0_km"],
                    state["v0_km_s"],
                    float(dt_s),
                )
                return float(r_km[0]) * _KM_TO_M, float(r_km[1]) * _KM_TO_M
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
            "k_km3_s2": 0.0,
            "r0_km": np.array([rel0_x * _M_TO_KM, rel0_y * _M_TO_KM, 0.0], dtype=np.float64),
            "v0_km_s": np.array([relv_x * _M_TO_KM, relv_y * _M_TO_KM, 0.0], dtype=np.float64),
        }

        scripted_state = self._scripted_relative_state_from_elements(body, parent)
        if scripted_state is not None:
            s_rel_x, s_rel_y, s_rel_vx, s_rel_vy, mu = scripted_state
            k_km3_s2 = float(mu) * 1e-9
            if k_km3_s2 > 0.0:
                state = {
                    "rel0_m": (s_rel_x, s_rel_y),
                    "relv_m_s": (s_rel_vx, s_rel_vy),
                    "use_kepler": True,
                    "k_km3_s2": k_km3_s2,
                    "r0_km": np.array([s_rel_x * _M_TO_KM, s_rel_y * _M_TO_KM, 0.0], dtype=np.float64),
                    "v0_km_s": np.array([s_rel_vx * _M_TO_KM, s_rel_vy * _M_TO_KM, 0.0], dtype=np.float64),
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
        try:
            if getattr(self, "_cache_vp_time", None) == float(time_s):
                return Vec2(self._cache_vp_x, self._cache_vp_y)
        except Exception:
            pass

        orbit = _orbit_model_from_body(self.child_body)
        if orbit is None:
            p_x, p_y = self._body_world_position_at_time(self.primary_body, time_s)
            vp = Vec2(float(p_x), float(p_y))
            self._cache_vp_time = float(time_s)
            self._cache_vp_x = float(vp.x)
            self._cache_vp_y = float(vp.y)
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
        self._cache_vp_time = float(time_s)
        self._cache_vp_x = float(vp.x)
        self._cache_vp_y = float(vp.y)
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
        try:
            if getattr(self, "_cache_time", None) == cache_time:
                return
        except Exception:
            pass
        angle = self._x_axis_angle(cache_time)
        origin_x, origin_y = self._body_world_position_at_time(self.primary_body, cache_time)
        self._cache_cos = math.cos(angle)
        self._cache_sin = math.sin(angle)
        self._cache_origin_x = origin_x
        self._cache_origin_y = origin_y
        self._cache_time = cache_time

    def to_this_frame_xy(self, time_s: float, x: float, y: float) -> tuple[float, float]:
        self._prepare_cache(time_s)
        rel_x = float(x) - self._cache_origin_x
        rel_y = float(y) - self._cache_origin_y
        rx = self._cache_cos * rel_x - self._cache_sin * rel_y
        ry = self._cache_sin * rel_x + self._cache_cos * rel_y
        return rx, ry

    def transform_heading(self, time_s: float, theta_world: float) -> float:
        self._prepare_cache(time_s)
        hx = math.cos(float(theta_world))
        hy = math.sin(float(theta_world))
        fx = self._cache_cos * hx - self._cache_sin * hy
        fy = self._cache_sin * hx + self._cache_cos * hy
        return math.atan2(fy, fx)


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
        try:
            if getattr(self, "_cache_time", None) == cache_time:
                return
        except Exception:
            pass
        angle = self._x_axis_angle(cache_time)
        origin_x, origin_y = self._body_world_position_at_time(self.target_body, cache_time)
        self._cache_cos = math.cos(angle)
        self._cache_sin = math.sin(angle)
        self._cache_origin_x = origin_x
        self._cache_origin_y = origin_y
        self._cache_time = cache_time

    def to_this_frame_xy(self, time_s: float, x: float, y: float) -> tuple[float, float]:
        self._prepare_cache(time_s)
        rel_x = float(x) - self._cache_origin_x
        rel_y = float(y) - self._cache_origin_y
        rx = self._cache_cos * rel_x - self._cache_sin * rel_y
        ry = self._cache_sin * rel_x + self._cache_cos * rel_y
        return rx, ry

    def transform_heading(self, time_s: float, theta_world: float) -> float:
        self._prepare_cache(time_s)
        hx = math.cos(float(theta_world))
        hy = math.sin(float(theta_world))
        fx = self._cache_cos * hx - self._cache_sin * hy
        fy = self._cache_sin * hx + self._cache_cos * hy
        return math.atan2(fy, fx)


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
    extension = int(frame_parameters.extension)
    if extension != BODY_CENTRED_NON_ROTATING:
        return primary_index

    primary = _resolve_body(primary_index, bodies)
    child = _find_virtual_swap_child(primary, bodies)
    if child is None:
        return primary_index

    for idx, body in enumerate(bodies):
        if body is child:
            return idx
    return primary_index


def new_plotting_frame(frame_parameters: PlottingFrameParameters, bodies: Sequence[object]) -> ReferenceFrame:
    extension = int(frame_parameters.extension)
    primary = _resolve_body(frame_parameters.primary_index, bodies)

    if extension == BODY_CENTRED_NON_ROTATING:
        candidate = _find_virtual_swap_child(primary, bodies)
        if candidate is not None:
            return VirtualBodyCentredNonRotatingReferenceFrame(primary, candidate)
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
