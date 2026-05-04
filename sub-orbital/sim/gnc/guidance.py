"""Gravity-turn pitch programme.

Phase 0  (0  → t_kick):   vertical hold, θ_target = 0
Phase 1  (t_kick → t_end): sinusoidal pitch-up to theta_max, then hold
Phase 2  (post-burnout):   ballistic — target attitude = current velocity direction

The initial 'kick' displaces the vehicle slightly off-vertical so that
aerodynamic and gravity forces can drive the natural gravity turn.
"""
from __future__ import annotations

import numpy as np

from sim.dynamics import RocketState


class GravityTurnGuidance:
    def __init__(
        self,
        t_kick:     float = 3.0,    # s   — start pitching over
        t_burnout:  float = 41.0,   # s   — end of thrust (approx)
        theta_max:  float = 20.0,   # deg — maximum pitch angle from vertical
        yaw_target: float = 0.0,    # deg — heading direction (0 = East)
    ):
        self.t_kick    = t_kick
        self.t_end     = t_burnout
        self.theta_max = np.deg2rad(theta_max)
        self.psi_tgt   = np.deg2rad(yaw_target)

    def target_attitude(self, state: RocketState) -> np.ndarray:
        """Return target Euler angles [φ, θ, ψ] at the current time."""
        t = state.time

        phi_tgt = 0.0
        psi_tgt = self.psi_tgt

        if t < self.t_kick:
            theta_tgt = 0.0
        elif t < self.t_end:
            frac      = (t - self.t_kick) / (self.t_end - self.t_kick)
            theta_tgt = self.theta_max * np.sin(np.pi / 2 * frac) ** 2
        else:
            # After burnout track current attitude (no active guidance)
            theta_tgt = state.euler[1]

        return np.array([phi_tgt, theta_tgt, psi_tgt])
