"""IMU sensor model: accelerometer + gyroscope with bias and white noise."""
from __future__ import annotations

import numpy as np

from sim.dynamics import RocketState, RocketDynamics, rot_matrix


class IMU:
    """Strapdown inertial measurement unit.

    Accelerometer output: specific force in body frame (non-gravitational accel).
    Gyroscope output    : angular rate in body frame.

    Both channels carry a slowly-varying bias (Gauss-Markov) plus zero-mean
    white noise at each sample.
    """

    def __init__(
        self,
        sigma_acc:  float = 0.30,   # m/s²  accelerometer white noise std
        sigma_gyro: float = 0.004,  # rad/s gyro white noise std
        tau_acc:    float = 300.0,  # s     bias correlation time
        tau_gyro:   float = 300.0,  # s     bias correlation time
        bias_acc_init:  float = 0.05,  # m/s²  initial bias magnitude (each axis)
        bias_gyro_init: float = 5e-4,  # rad/s initial bias magnitude
        rng: np.random.Generator | None = None,
    ):
        self.sigma_acc  = sigma_acc
        self.sigma_gyro = sigma_gyro
        self.tau_acc    = tau_acc
        self.tau_gyro   = tau_gyro
        self.rng = rng or np.random.default_rng(0)

        self.bias_acc  = self.rng.uniform(-bias_acc_init,  bias_acc_init,  3)
        self.bias_gyro = self.rng.uniform(-bias_gyro_init, bias_gyro_init, 3)

    # ------------------------------------------------------------------
    def step_bias(self, dt: float) -> None:
        """Propagate bias via first-order Gauss-Markov process."""
        qa = np.sqrt(2 * self.sigma_acc ** 2 / self.tau_acc  * dt)
        qg = np.sqrt(2 * self.sigma_gyro** 2 / self.tau_gyro * dt)
        self.bias_acc  = (np.exp(-dt / self.tau_acc)  * self.bias_acc
                          + qa * self.rng.standard_normal(3))
        self.bias_gyro = (np.exp(-dt / self.tau_gyro) * self.bias_gyro
                          + qg * self.rng.standard_normal(3))

    # ------------------------------------------------------------------
    def measure(
        self,
        state: RocketState,
        dynamics: RocketDynamics,
        tvc: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (accel_body, omega_body) noisy measurements."""
        self.step_bias(dt)

        f_true    = dynamics.specific_force_body(state, tvc)
        omega_true = state.omega.copy()

        f_meas = (f_true
                  + self.bias_acc
                  + self.sigma_acc * self.rng.standard_normal(3))
        omega_meas = (omega_true
                      + self.bias_gyro
                      + self.sigma_gyro * self.rng.standard_normal(3))
        return f_meas, omega_meas
