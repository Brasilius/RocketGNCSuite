"""Sequential Importance Resampling (SIR) particle filter — vectorised.

State per particle: x = [px, py, pz,  vx, vy, vz,  φ, θ, ψ]   (9 × 1)

Key design notes
----------------
* All particle operations are fully NumPy-vectorised (no Python loops).
* Log-weights prevent floating-point underflow.
* Each particle samples its OWN IMU noise each step, so the cloud spreads
  in altitude independently and the altimeter can discriminate.
  (Using the same IMU sample for all particles collapses the cloud to a
  single trajectory, making measurement updates ineffective.)
* Systematic resampling when ESS < N/2.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from sim.environment import gravity


# ---------------------------------------------------------------------------
# Vectorised helpers
# ---------------------------------------------------------------------------

def _batch_rot(euler: np.ndarray) -> np.ndarray:
    """ZYX rotation matrices for N particles — (N, 3, 3)."""
    phi, theta, psi = euler[:, 0], euler[:, 1], euler[:, 2]
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    N = len(phi)
    R = np.empty((N, 3, 3))
    R[:, 0, 0] =  ct*cy;  R[:, 0, 1] =  sp*st*cy - cp*sy;  R[:, 0, 2] =  cp*st*cy + sp*sy
    R[:, 1, 0] =  ct*sy;  R[:, 1, 1] =  sp*st*sy + cp*cy;  R[:, 1, 2] =  cp*st*sy - sp*cy
    R[:, 2, 0] = -st;     R[:, 2, 1] =  sp*ct;             R[:, 2, 2] =  cp*ct
    return R


def _batch_euler_to_quat(euler: np.ndarray) -> np.ndarray:
    """ZYX Euler (N,3) → quaternion [w,x,y,z] (N,4)."""
    h = 0.5 * euler
    cp, sp = np.cos(h[:, 0]), np.sin(h[:, 0])
    ct, st = np.cos(h[:, 1]), np.sin(h[:, 1])
    cy, sy = np.cos(h[:, 2]), np.sin(h[:, 2])
    q = np.empty((len(euler), 4))
    q[:, 0] = cp*ct*cy + sp*st*sy
    q[:, 1] = sp*ct*cy - cp*st*sy
    q[:, 2] = cp*st*cy + sp*ct*sy
    q[:, 3] = cp*ct*sy - sp*st*cy
    return q


def _batch_quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (N,4) → ZYX Euler (N,3). Singularity-safe."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    phi   = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    theta = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    psi   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.column_stack([phi, theta, psi])


def _batch_quat_dot(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Quaternion kinematics for N particles — q̇ = 0.5·q⊗[0,ω].

    q     : (N, 4)
    omega : (N, 3)
    Returns (N, 4)
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    p, qr, r = omega[:, 0], omega[:, 1], omega[:, 2]
    dq = np.empty_like(q)
    dq[:, 0] = 0.5 * (-x*p  - y*qr - z*r)
    dq[:, 1] = 0.5 * ( w*p  + y*r  - z*qr)
    dq[:, 2] = 0.5 * ( w*qr - x*r  + z*p)
    dq[:, 3] = 0.5 * ( w*r  + x*qr - y*p)
    return dq


# ---------------------------------------------------------------------------
# SIR Particle Filter
# ---------------------------------------------------------------------------

class ParticleFilter:
    """SIR particle filter — fully vectorised prediction and updates."""

    def __init__(
        self,
        x0:     np.ndarray,          # initial mean state [9]
        P0:     np.ndarray,          # initial covariance [9×9]
        N:      int   = 500,
        sigma_acc:   float = 0.30,   # m/s²  IMU-like process noise for vel
        sigma_gyro:  float = 0.004,  # rad/s IMU-like process noise for euler
        sigma_pos:   float = 1.0,    # m/√s  additional position noise
        sigma_alt:   float = 5.0,    # m     altimeter noise std
        sigma_att:   float = 0.009,  # rad   horizon tracker noise std
        rng: np.random.Generator | None = None,
    ):
        self.N    = N
        self.rng  = rng or np.random.default_rng(42)

        # Process noise std vector (applied per √s, so multiply by √dt each step)
        self._noise_std = np.array([
            sigma_pos,   sigma_pos,   sigma_pos,    # position (small, velocity drift dominates)
            sigma_acc,   sigma_acc,   sigma_acc,    # velocity
            sigma_gyro,  sigma_gyro,  sigma_gyro,   # euler
        ])

        # Measurement sigmas (scalar or vector)
        self._sig_alt = float(sigma_alt)
        self._sig_att = np.array([sigma_att, sigma_att])

        # Initialise particles from Gaussian
        L = np.linalg.cholesky(P0)
        draws = self.rng.standard_normal((9, N))
        self.particles = (x0[:, None] + L @ draws).T           # (N, 9)
        self.log_w     = np.full(N, -np.log(N), dtype=float)   # uniform

    # ------------------------------------------------------------------
    def predict(self, f_body: np.ndarray, omega_body: np.ndarray,
                dt: float) -> None:
        """Propagate all particles (vectorised).

        Each particle draws its own IMU noise so the cloud spreads
        independently — essential for measurement updates to work.
        Quaternion kinematics replace Euler kinematics to avoid the θ=±90° singularity.
        """
        N     = self.N
        pos   = self.particles[:, 0:3]    # (N, 3)
        vel   = self.particles[:, 3:6]
        euler = self.particles[:, 6:9]

        # Per-particle noisy IMU
        noise_v = self.rng.standard_normal((N, 3)) * (self._noise_std[3] * np.sqrt(dt))
        noise_e = self.rng.standard_normal((N, 3)) * (self._noise_std[6] * np.sqrt(dt))
        noise_p = self.rng.standard_normal((N, 3)) * (self._noise_std[0] * np.sqrt(dt))

        omega_N = omega_body + noise_e / dt    # (N, 3) per-particle gyro
        f_N     = f_body     + noise_v / dt    # (N, 3) per-particle accel

        R     = _batch_rot(euler)              # (N, 3, 3)
        a_enu = np.einsum('nij,nj->ni', R, f_N)
        alt_mean = float(np.mean(np.maximum(0.0, pos[:, 2])))
        a_enu[:, 2] += -gravity(alt_mean)

        # Quaternion attitude propagation — singularity-free
        q     = _batch_euler_to_quat(euler)    # (N, 4)
        dq    = _batch_quat_dot(q, omega_N)    # (N, 4)
        q_new = q + dt * dq
        q_new /= np.linalg.norm(q_new, axis=1, keepdims=True)

        self.particles[:, 0:3] = pos + vel   * dt + noise_p
        self.particles[:, 3:6] = vel + a_enu * dt
        self.particles[:, 6:9] = _batch_quat_to_euler(q_new)

    # ------------------------------------------------------------------
    def update_altimeter(self, z_alt: float) -> None:
        """Vectorised log-likelihood update from altimeter."""
        h = self.particles[:, 2]                         # predicted pz (N,)
        self.log_w += -0.5 * ((z_alt - h) / self._sig_alt) ** 2
        self._normalise()
        self._maybe_resample()

    def update_horizon(self, z_hor: np.ndarray) -> None:
        """Vectorised log-likelihood update from horizon tracker [φ, θ]."""
        h = self.particles[:, 6:8]                       # (N, 2)
        diff = z_hor - h                                 # (N, 2)
        self.log_w += -0.5 * np.sum((diff / self._sig_att) ** 2, axis=1)
        self._normalise()
        self._maybe_resample()

    # ------------------------------------------------------------------
    @property
    def state(self) -> np.ndarray:
        """Weighted mean estimate."""
        w = np.exp(self.log_w)
        return w @ self.particles                        # (9,)

    @property
    def covariance(self) -> np.ndarray:
        w    = np.exp(self.log_w)
        mu   = self.state
        diff = self.particles - mu                      # (N, 9)
        return (w[:, None] * diff).T @ diff             # (9, 9)

    @property
    def ess(self) -> float:
        w = np.exp(self.log_w)
        return float(1.0 / np.sum(w ** 2))

    # ------------------------------------------------------------------
    def _normalise(self) -> None:
        self.log_w -= logsumexp(self.log_w)

    def _maybe_resample(self) -> None:
        if self.ess < self.N / 2:
            self._systematic_resample()

    def _systematic_resample(self) -> None:
        w   = np.exp(self.log_w)
        cdf = np.cumsum(w)
        u0  = self.rng.uniform(0.0, 1.0 / self.N)
        pos = u0 + np.arange(self.N) / self.N
        idx = np.searchsorted(cdf, pos)
        self.particles = self.particles[idx].copy()
        self.log_w     = np.full(self.N, -np.log(self.N))
