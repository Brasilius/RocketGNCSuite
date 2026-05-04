"""Radar/laser altimeter — measures height above launch site."""
from __future__ import annotations

import numpy as np

from sim.dynamics import RocketState


class Altimeter:
    """Returns a scalar altitude measurement z ~ N(z_true, sigma²)."""

    def __init__(
        self,
        sigma: float = 5.0,          # m  measurement noise std
        bias:  float = 1.0,          # m  constant bias
        rng: np.random.Generator | None = None,
    ):
        self.sigma = sigma
        self.bias  = bias
        self.rng   = rng or np.random.default_rng(1)

    def measure(self, state: RocketState) -> float:
        return float(state.pos[2] + self.bias + self.sigma * self.rng.standard_normal())
