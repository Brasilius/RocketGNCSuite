"""Tests for IMU, altimeter, and horizon tracker sensors."""
import numpy as np
import pytest

from sim.dynamics import RocketConfig, RocketDynamics, RocketState
from sim.sensors.altimeter import Altimeter
from sim.sensors.horizon_tracker import HorizonTracker
from sim.sensors.imu import IMU


def _state(alt=1000.0, vel=None, euler=None, omega=None, mass=200.0):
    return RocketState(
        pos   = np.array([0.0, 0.0, alt]),
        vel   = vel   if vel   is not None else np.zeros(3),
        euler = euler if euler is not None else np.zeros(3),
        omega = omega if omega is not None else np.zeros(3),
        mass  = mass,
        time  = 0.0,
    )


# ---------------------------------------------------------------------------
# Altimeter
# ---------------------------------------------------------------------------

def test_altimeter_mean_near_truth():
    alt_sensor = Altimeter(sigma=5.0, bias=0.0, rng=np.random.default_rng(0))
    st = _state(alt=5000.0)
    measurements = [alt_sensor.measure(st) for _ in range(2000)]
    assert abs(np.mean(measurements) - 5000.0) < 1.0     # mean within 1 m


def test_altimeter_std_matches_sigma():
    sigma = 7.0
    alt_sensor = Altimeter(sigma=sigma, bias=0.0, rng=np.random.default_rng(1))
    st = _state(alt=3000.0)
    measurements = [alt_sensor.measure(st) for _ in range(2000)]
    assert abs(np.std(measurements) - sigma) < 0.5


def test_altimeter_bias():
    bias = 3.0
    alt_sensor = Altimeter(sigma=0.001, bias=bias, rng=np.random.default_rng(2))
    st = _state(alt=2000.0)
    measurements = [alt_sensor.measure(st) for _ in range(50)]
    assert abs(np.mean(measurements) - (2000.0 + bias)) < 0.1


# ---------------------------------------------------------------------------
# Horizon tracker
# ---------------------------------------------------------------------------

def test_horizon_tracker_mean_near_truth():
    ht = HorizonTracker(sigma_deg=0.5, rng=np.random.default_rng(3))
    phi, theta = 0.1, 0.2
    st = _state(euler=np.array([phi, theta, 0.3]))
    measurements = np.array([ht.measure(st) for _ in range(2000)])
    np.testing.assert_allclose(measurements.mean(axis=0), [phi, theta], atol=0.02)


def test_horizon_tracker_std():
    sigma_deg = 0.5
    ht = HorizonTracker(sigma_deg=sigma_deg, rng=np.random.default_rng(4))
    st = _state()
    measurements = np.array([ht.measure(st) for _ in range(2000)])
    expected_sigma = np.deg2rad(sigma_deg)
    np.testing.assert_allclose(measurements.std(axis=0), [expected_sigma, expected_sigma],
                               atol=0.002)


def test_horizon_tracker_output_shape():
    ht = HorizonTracker(rng=np.random.default_rng(5))
    st = _state()
    z = ht.measure(st)
    assert z.shape == (2,)


# ---------------------------------------------------------------------------
# IMU
# ---------------------------------------------------------------------------

def _setup_imu():
    cfg = RocketConfig(cd=0.0)
    dyn = RocketDynamics(cfg)
    return IMU(rng=np.random.default_rng(10)), dyn, cfg


def test_imu_output_shapes():
    imu, dyn, cfg = _setup_imu()
    st = _state(mass=cfg.mass_dry + cfg.mass_prop)
    f_meas, omega_meas = imu.measure(st, dyn, np.zeros(2), 0.01)
    assert f_meas.shape == (3,)
    assert omega_meas.shape == (3,)


def test_imu_accel_near_truth_no_bias():
    """With no gyro bias, many IMU samples should average near the truth."""
    cfg = RocketConfig(cd=0.0)
    dyn = RocketDynamics(cfg)
    imu = IMU(sigma_acc=0.01, sigma_gyro=0.0, bias_acc_init=0.0, bias_gyro_init=0.0,
              rng=np.random.default_rng(11))
    st = _state(mass=cfg.mass_dry + cfg.mass_prop)
    f_true = dyn.specific_force_body(st, np.zeros(2))
    measurements = np.array([imu.measure(st, dyn, np.zeros(2), 0.01)[0]
                              for _ in range(500)])
    np.testing.assert_allclose(measurements.mean(axis=0), f_true, atol=0.05)


def test_imu_omega_near_truth():
    """IMU gyro should reflect true omega with small noise."""
    cfg = RocketConfig(cd=0.0)
    dyn = RocketDynamics(cfg)
    imu = IMU(sigma_gyro=0.001, bias_gyro_init=0.0, rng=np.random.default_rng(12))
    omega_true = np.array([0.01, -0.02, 0.005])
    st = _state(omega=omega_true, mass=cfg.mass_dry)
    _, omega_meas = imu.measure(st, dyn, np.zeros(2), 0.01)
    # Should be within a few sigma
    np.testing.assert_allclose(omega_meas, omega_true, atol=0.05)
