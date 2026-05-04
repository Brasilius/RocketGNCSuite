"""TVC attitude controller — PD on pitch and yaw Euler errors.

Sign conventions (from dynamics.py):
    d1  tilts nozzle in body-xz plane → tau_y = -L*T*d1  (pitch moment)
    d2  tilts nozzle in body-yz plane → tau_x =  L*T*d2  (roll moment)

Pitch stability analysis (θ̈ = tau_y/Iyy, θ̇ ≈ q):
    For stability the characteristic equation must have all-positive coefficients.
    Correct law: d1 = -(Kp*err - Kd*q)  →  tau_y = L*T*(Kp*err - Kd*q)
    →  Iyy·θ̈ + L·T·Kd·θ̇ + L·T·Kp·θ = L·T·Kp·θ_t   ✓

    Incorrect (positive-feedback): d1 = -(Kp*err + Kd*q) makes Kd term destabilise.

Roll is controlled by d2 to keep φ ≈ 0 using the same sign logic.
Yaw has no direct TVC authority; aerodynamic restoring moment provides passive stability.
"""
from __future__ import annotations

import numpy as np

from sim.dynamics import RocketState


class TVCController:
    def __init__(
        self,
        kp_pitch: float = 0.25,     # rad/rad
        kd_pitch: float = 0.80,     # rad·s/rad
        kp_roll:  float = 0.20,
        kd_roll:  float = 0.60,
        max_deflect_deg: float = 5.0,
    ):
        self.kp_pitch    = kp_pitch
        self.kd_pitch    = kd_pitch
        self.kp_roll     = kp_roll
        self.kd_roll     = kd_roll
        self.max_deflect = np.deg2rad(max_deflect_deg)

    def compute(
        self,
        target: np.ndarray,   # [φ_t, θ_t, ψ_t]
        state: RocketState,
        burning: bool,
    ) -> np.ndarray:
        """Return TVC deflections [d1, d2] in radians."""
        if not burning:
            return np.zeros(2)

        phi,   theta,  _    = state.euler
        phi_t, theta_t, _   = target
        p, q, _              = state.omega

        # Pitch: d1 = -(Kp*err - Kd*q)  — stable PD (see header for derivation)
        d1 = -(self.kp_pitch * (theta_t - theta) - self.kd_pitch * q)

        # Roll: d2 = -(Kp*err - Kd*p)   — same sign logic, controls φ → 0
        d2 = -(self.kp_roll  * (phi_t   - phi)   - self.kd_roll  * p)

        d1 = np.clip(d1, -self.max_deflect, self.max_deflect)
        d2 = np.clip(d2, -self.max_deflect, self.max_deflect)
        return np.array([d1, d2])
