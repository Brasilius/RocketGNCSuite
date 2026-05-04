"""6-DOF rocket dynamics with ZYX Euler attitude representation.

Frame convention
----------------
World  : ENU  — x = East, y = North, z = Up
Body   : z = nose direction (up at launch), x = East, y = North at launch
         → identity quaternion / zero Euler angles = rocket pointing straight up

Euler  : ZYX order — ψ (yaw, around z), θ (pitch, around y), φ (roll, around x)
         Singularity at θ = ±90° (horizontal flight, never reached sub-orbitally).

TVC    : two nozzle deflection angles
         d1 — tilts nozzle in body-xz plane  →  M_y pitch torque
         d2 — tilts nozzle in body-yz plane  →  M_x yaw/roll torque
         r_nozzle = [0, 0, -L_arm] (aft of CG in body frame)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.environment import atmosphere, gravity as grav_fn


# ---------------------------------------------------------------------------
# Helpers shared by dynamics, sensors, and filters
# ---------------------------------------------------------------------------

def rot_matrix(euler: np.ndarray) -> np.ndarray:
    """ZYX rotation matrix R such that v_ENU = R @ v_body."""
    phi, theta, psi = euler
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    return np.array([
        [ct*cy,  sp*st*cy - cp*sy,  cp*st*cy + sp*sy],
        [ct*sy,  sp*st*sy + cp*cy,  cp*st*sy - sp*cy],
        [-st,    sp*ct,              cp*ct],
    ])


def euler_kin(euler: np.ndarray) -> np.ndarray:
    """Matrix W s.t. euler_dot = W @ omega_body (ZYX kinematics).

    Singular at θ = ±90°.  Use quat_kinematics for propagation instead.
    """
    phi, theta, _ = euler
    cp, sp = np.cos(phi), np.sin(phi)
    ct = np.cos(theta)
    if abs(ct) < 1e-8:
        ct = np.copysign(1e-8, ct)
    return np.array([
        [1.0,  sp * np.sin(theta) / ct,  cp * np.sin(theta) / ct],
        [0.0,  cp,                        -sp],
        [0.0,  sp / ct,                   cp / ct],
    ])


# ---------------------------------------------------------------------------
# Quaternion helpers (singularity-free attitude representation)
# ---------------------------------------------------------------------------

def euler_to_quat(euler: np.ndarray) -> np.ndarray:
    """ZYX Euler [φ, θ, ψ] → unit quaternion [w, x, y, z]."""
    phi, theta, psi = 0.5 * euler
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi),   np.sin(psi)
    return np.array([
        cp*ct*cy + sp*st*sy,
        sp*ct*cy - cp*st*sy,
        cp*st*cy + sp*ct*sy,
        cp*ct*sy - sp*st*cy,
    ])


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] → ZYX Euler [φ, θ, ψ] (rad).

    Near |θ| = 90° the φ/ψ split is non-unique; arctan2(0,0) → 0 gives
    a canonical representative that yields the correct rotation matrix.
    """
    w, x, y, z = q / np.linalg.norm(q)
    phi   = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    theta = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    psi   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([phi, theta, psi])


def rot_from_quat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from unit quaternion [w, x, y, z].

    v_ENU = R @ v_body  (same convention as rot_matrix, but no Euler singularity).
    """
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def quat_kinematics(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Quaternion derivative: q̇ = 0.5 · q ⊗ [0, p, qr, r]."""
    w, x, y, z = q
    p, qr, r = omega
    return 0.5 * np.array([
        -x*p  - y*qr - z*r,
         w*p  + y*r  - z*qr,
         w*qr - x*r  + z*p,
         w*r  + x*qr - y*p,
    ])


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class RocketState:
    pos:   np.ndarray   # [x, y, z] ENU metres
    vel:   np.ndarray   # [vx, vy, vz] ENU m/s
    euler: np.ndarray   # [φ, θ, ψ] radians
    omega: np.ndarray   # [p, q, r] body angular rates rad/s
    mass:  float        # total mass kg
    time:  float = 0.0

    def nav_state(self) -> np.ndarray:
        """Flat 9-vector [pos, vel, euler] used by EKF / particle filter."""
        return np.concatenate([self.pos, self.vel, self.euler])


# ---------------------------------------------------------------------------
# Rocket configuration
# ---------------------------------------------------------------------------

@dataclass
class RocketConfig:
    thrust:   float = 8_000.0                            # N
    isp:      float = 220.0                              # s
    mass_dry: float = 100.0                              # kg
    mass_prop:float = 150.0                              # kg  → m0 = 250 kg
    area_ref: float = float(np.pi * 0.15 ** 2)          # m²  (d = 0.30 m)
    cd:       float = 0.40                               # drag coefficient
    inertia:  np.ndarray = field(
        default_factory=lambda: np.array([4.0, 120.0, 120.0]))  # [Ixx, Iyy, Izz] kg·m²
    tvc_arm:  float = 1.2                                # m  (nozzle aft of CG)
    # Aerodynamic stability derivatives (fin-stabilised, dimensionless)
    cm_alpha: float = -0.40   # pitch stability (negative = stable)
    cn_beta:  float = -0.40   # yaw/sideslip stability


# ---------------------------------------------------------------------------
# Dynamics engine
# ---------------------------------------------------------------------------

class RocketDynamics:
    G0 = 9.80665

    def __init__(self, cfg: RocketConfig):
        self.cfg  = cfg
        self.mdot = cfg.thrust / (cfg.isp * self.G0)   # kg/s propellant flow

    # ------------------------------------------------------------------
    def specific_force_body(self, state: RocketState, tvc: np.ndarray) -> np.ndarray:
        """Non-gravitational acceleration in body frame — what the IMU measures."""
        alt       = state.pos[2]
        rho       = atmosphere(alt)[0]
        R         = rot_matrix(state.euler)
        burning   = state.mass > self.cfg.mass_dry + 0.5
        d1, d2    = tvc

        # Thrust (aligned along body z, tilted by TVC)
        T = self.cfg.thrust if burning else 0.0
        F_thr_body = T * np.array([np.sin(d1), np.sin(d2),
                                    np.cos(d1) * np.cos(d2)])

        # Aerodynamic drag (in body frame, opposes velocity)
        v_body = R.T @ state.vel
        v_mag  = np.linalg.norm(v_body)
        if v_mag > 0.5:
            F_drag_body = (-0.5 * rho * self.cfg.cd * self.cfg.area_ref
                           * v_mag ** 2 * v_body / v_mag)
        else:
            F_drag_body = np.zeros(3)

        return (F_thr_body + F_drag_body) / state.mass

    # ------------------------------------------------------------------
    def _tvc_torque(self, state: RocketState, tvc: np.ndarray) -> np.ndarray:
        burning = state.mass > self.cfg.mass_dry + 0.5
        T = self.cfg.thrust if burning else 0.0
        d1, d2 = tvc
        L = self.cfg.tvc_arm
        # r_nozzle = [0, 0, -L] in body frame
        # F_body ≈ T * [d1, d2, 1]  (small-angle)
        # tau = r × F:   tau_x = L*T*d2,   tau_y = -L*T*d1,   tau_z = 0
        return np.array([L * T * d2, -L * T * d1, 0.0])

    def _aero_torque(self, state: RocketState) -> np.ndarray:
        """Restoring aerodynamic moments (fin stabilisation)."""
        alt   = state.pos[2]
        rho   = atmosphere(alt)[0]
        R     = rot_matrix(state.euler)
        v_body = R.T @ state.vel
        v_mag  = np.linalg.norm(v_body)
        if v_mag < 5.0:
            return np.zeros(3)

        alpha = np.arctan2(v_body[0], v_body[2])   # pitch angle-of-attack
        beta  = np.arctan2(v_body[1], v_body[2])   # sideslip

        q_bar = 0.5 * rho * v_mag ** 2
        d_ref = 0.30   # reference diameter
        A     = self.cfg.area_ref

        M_y = self.cfg.cm_alpha * q_bar * A * d_ref * alpha   # pitch restoring
        M_x = self.cfg.cn_beta  * q_bar * A * d_ref * beta    # yaw  restoring
        return np.array([M_x, M_y, 0.0])

    # ------------------------------------------------------------------
    def step(self, state: RocketState, tvc: np.ndarray, dt: float) -> RocketState:
        """RK4 step using quaternion attitude — no Euler singularity at θ = ±90°."""
        q0 = euler_to_quat(state.euler)

        def derivs(pos, vel, q, omega, mass):
            alt   = float(pos[2])
            rho   = atmosphere(alt)[0]
            R     = rot_from_quat(q)
            g_enu = np.array([0.0, 0.0, -grav_fn(alt)])
            burning = mass > self.cfg.mass_dry + 0.5
            d1, d2  = tvc
            T = self.cfg.thrust if burning else 0.0

            F_thr  = T * np.array([np.sin(d1), np.sin(d2), np.cos(d1)*np.cos(d2)])
            v_body = R.T @ vel
            v_mag  = np.linalg.norm(v_body)
            F_drag = ((-0.5*rho*self.cfg.cd*self.cfg.area_ref*v_mag**2*v_body/v_mag)
                      if v_mag > 0.5 else np.zeros(3))
            f_b = (F_thr + F_drag) / mass

            L = self.cfg.tvc_arm
            tau_tvc = np.array([L*T*d2, -L*T*d1, 0.0])

            if v_mag > 5.0:
                alpha    = np.arctan2(v_body[0], v_body[2])
                beta     = np.arctan2(v_body[1], v_body[2])
                q_bar    = 0.5 * rho * v_mag**2
                A, d_ref = self.cfg.area_ref, 0.30
                tau_aero = np.array([
                    self.cfg.cn_beta  * q_bar * A * d_ref * beta,
                    self.cfg.cm_alpha * q_bar * A * d_ref * alpha,
                    0.0,
                ])
            else:
                tau_aero = np.zeros(3)

            tau = tau_tvc + tau_aero
            p_, qr, r_ = omega
            Ix, Iy, Iz = self.cfg.inertia
            domega = np.array([
                (tau[0] - (Iz - Iy)*qr*r_) / Ix,
                (tau[1] - (Ix - Iz)*p_*r_) / Iy,
                (tau[2] - (Iy - Ix)*p_*qr) / Iz,
            ])
            dmass = -self.mdot if burning else 0.0
            return vel.copy(), R @ f_b + g_enu, quat_kinematics(q, omega), domega, dmass

        def _nq(q_in, dq, h):
            q_new = q_in + h * dq
            return q_new / np.linalg.norm(q_new)

        dp1, dv1, dq1, do1, dm1 = derivs(state.pos, state.vel, q0, state.omega, state.mass)
        q2 = _nq(q0, dq1, 0.5*dt)
        dp2, dv2, dq2, do2, dm2 = derivs(
            state.pos + 0.5*dt*dp1, state.vel + 0.5*dt*dv1, q2,
            state.omega + 0.5*dt*do1, max(self.cfg.mass_dry, state.mass + 0.5*dt*dm1))
        q3 = _nq(q0, dq2, 0.5*dt)
        dp3, dv3, dq3, do3, dm3 = derivs(
            state.pos + 0.5*dt*dp2, state.vel + 0.5*dt*dv2, q3,
            state.omega + 0.5*dt*do2, max(self.cfg.mass_dry, state.mass + 0.5*dt*dm2))
        q4 = _nq(q0, dq3, dt)
        dp4, dv4, dq4, do4, dm4 = derivs(
            state.pos + dt*dp3, state.vel + dt*dv3, q4,
            state.omega + dt*do3, max(self.cfg.mass_dry, state.mass + dt*dm3))

        c   = dt / 6.0
        q_f = q0 + c * (dq1 + 2*dq2 + 2*dq3 + dq4)
        q_f /= np.linalg.norm(q_f)
        return RocketState(
            pos   = state.pos   + c*(dp1 + 2*dp2 + 2*dp3 + dp4),
            vel   = state.vel   + c*(dv1 + 2*dv2 + 2*dv3 + dv4),
            euler = quat_to_euler(q_f),
            omega = state.omega + c*(do1 + 2*do2 + 2*do3 + do4),
            mass  = max(self.cfg.mass_dry,
                        state.mass + c*(dm1 + 2*dm2 + 2*dm3 + dm4)),
            time  = state.time + dt,
        )
