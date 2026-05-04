"""Extended Kalman Filter for rocket navigation.

State  x = [px, py, pz,  vx, vy, vz,  φ, θ, ψ]   (9 × 1)

Process model driven by IMU measurements:
  p_new  = p + v * dt
  v_new  = v + (R(euler) @ f_body + g_ENU) * dt
  e_new  = euler + W(euler) @ omega * dt

Process Jacobian computed numerically (central differences) for robustness.

Measurement models:
  Altimeter     : z_alt = pz                         (scalar)
  Horizon tracker: z_hor = [φ, θ]                   (2-vector)
"""
from __future__ import annotations

import numpy as np

from sim.dynamics import rot_matrix, euler_to_quat, quat_to_euler, quat_kinematics
from sim.environment import gravity


# ---------------------------------------------------------------------------
# Process model (Euler integration — filter doesn't need RK4 accuracy)
# Quaternion kinematics replace euler_kin to avoid singularity at θ = ±90°.
# ---------------------------------------------------------------------------

def _process(x: np.ndarray, f_body: np.ndarray, omega: np.ndarray,
             dt: float) -> np.ndarray:
    pos   = x[0:3]
    vel   = x[3:6]
    euler = x[6:9]

    alt   = max(0.0, pos[2])
    g_enu = np.array([0.0, 0.0, -gravity(alt)])
    R     = rot_matrix(euler)

    q     = euler_to_quat(euler)
    q_new = q + dt * quat_kinematics(q, omega)   # quat_kinematics returns q̇ = 0.5·q⊗ω
    q_new /= np.linalg.norm(q_new)

    new_pos   = pos + vel * dt
    new_vel   = vel + (R @ f_body + g_enu) * dt
    new_euler = quat_to_euler(q_new)

    return np.concatenate([new_pos, new_vel, new_euler])


def _jacobian_process(x: np.ndarray, f_body: np.ndarray, omega: np.ndarray,
                      dt: float, eps: float = 1e-5) -> np.ndarray:
    """Numerical central-difference Jacobian of the process function."""
    n  = len(x)
    fx = _process(x, f_body, omega, dt)
    F  = np.zeros((n, n))
    for i in range(n):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        F[:, i] = (_process(xp, f_body, omega, dt)
                   - _process(xm, f_body, omega, dt)) / (2 * eps)
    return F


# ---------------------------------------------------------------------------
# EKF class
# ---------------------------------------------------------------------------

class EKF:
    """9-state extended Kalman filter for rocket navigation."""

    def __init__(
        self,
        x0:     np.ndarray,          # initial state [9]
        P0:     np.ndarray,          # initial covariance [9×9]
        sigma_acc:   float = 0.30,   # m/s²  — used to build Q
        sigma_gyro:  float = 0.004,  # rad/s — used to build Q
        sigma_alt:   float = 5.0,    # m     — altimeter noise std
        sigma_att:   float = 0.009,  # rad   — horizon tracker noise std (~0.5°)
    ):
        self.x = x0.copy()
        self.P = P0.copy()

        self._sa   = sigma_acc
        self._sg   = sigma_gyro
        self.R_alt = np.array([[sigma_alt ** 2]])
        self.R_hor = np.diag([sigma_att ** 2, sigma_att ** 2])

        # Altimeter measurement matrix (pz = x[2])
        self.H_alt      = np.zeros((1, 9))
        self.H_alt[0, 2] = 1.0

        # Horizon tracker measurement matrix ([φ, θ] = x[6:8])
        self.H_hor        = np.zeros((2, 9))
        self.H_hor[0, 6]  = 1.0
        self.H_hor[1, 7]  = 1.0

    # ------------------------------------------------------------------
    def predict(self, f_body: np.ndarray, omega: np.ndarray, dt: float) -> None:
        """IMU-driven prediction step."""
        Q = self._build_Q(dt)

        F      = _jacobian_process(self.x, f_body, omega, dt)
        self.x = _process(self.x, f_body, omega, dt)
        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------------
    def update_altimeter(self, z_alt: float) -> None:
        """Measurement update from altimeter."""
        z = np.array([z_alt])
        y = z - self.H_alt @ self.x
        S = self.H_alt @ self.P @ self.H_alt.T + self.R_alt
        K = self.P @ self.H_alt.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self._joseph_update(K, self.H_alt, self.R_alt)

    def update_horizon(self, z_hor: np.ndarray) -> None:
        """Measurement update from horizon tracker [φ, θ]."""
        y = z_hor - self.H_hor @ self.x
        S = self.H_hor @ self.P @ self.H_hor.T + self.R_hor
        K = self.P @ self.H_hor.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self._joseph_update(K, self.H_hor, self.R_hor)

    def _joseph_update(self, K: np.ndarray, H: np.ndarray,
                        R: np.ndarray) -> np.ndarray:
        """Joseph form: P = (I-KH) P (I-KH)' + K R K'  (numerically stable)."""
        I_KH = np.eye(9) - K @ H
        P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return (P + P.T) * 0.5   # enforce symmetry

    # ------------------------------------------------------------------
    @property
    def state(self) -> np.ndarray:
        return self.x.copy()

    @property
    def std(self) -> np.ndarray:
        """1-sigma standard deviations for each state element."""
        return np.sqrt(np.diag(self.P))

    # ------------------------------------------------------------------
    def _build_Q(self, dt: float) -> np.ndarray:
        """Discrete-time process noise covariance."""
        sa2 = self._sa  ** 2
        sg2 = self._sg  ** 2
        # Position noise is dominated by velocity uncertainty integrated
        qp = sa2 * dt ** 3 / 3.0
        qv = sa2 * dt
        qe = sg2 * dt
        # Small floor to keep P from collapsing
        return np.diag([qp, qp, qp, qv, qv, qv, qe, qe, qe]) + np.eye(9) * 1e-10
