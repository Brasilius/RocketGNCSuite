"""Horizon tracker — measures roll and pitch angles relative to local horizontal.

For a sub-orbital vehicle this is modelled as a star-tracker / Earth-limb
sensor that directly observes the vehicle's roll (φ) and pitch (θ) angles
in the ENU frame.  Yaw (ψ) is not observable from Earth's horizon alone.
"""
from __future__ import annotations

import numpy as np

from sim.dynamics import RocketState


class HorizonTracker:
    """Returns [φ_meas, θ_meas] — noisy observations of roll and pitch."""

    def __init__(
        self,
        sigma_deg: float = 0.5,      # deg  noise std per axis
        rng: np.random.Generator | None = None,
    ):
        self.sigma = np.deg2rad(sigma_deg)
        self.rng   = rng or np.random.default_rng(2)

    def measure(self, state: RocketState) -> np.ndarray:
        phi, theta = state.euler[0], state.euler[1]
        noise = self.sigma * self.rng.standard_normal(2)
        return np.array([phi + noise[0], theta + noise[1]])
