"""Tests for EKF and particle filter."""
import numpy as np
import pytest

from sim.dynamics import RocketConfig, RocketDynamics, RocketState
from sim.filters.ekf import EKF
from sim.filters.particle import ParticleFilter, _batch_euler_to_quat, _batch_quat_to_euler


# ---------------------------------------------------------------------------
# EKF
# ---------------------------------------------------------------------------

def _make_ekf(pos=None, alt=500.0, alt_std=5.0):
    x0 = np.zeros(9)
    if pos is not None:
        x0[:3] = pos
    x0[2] = alt
    P0 = np.diag([10.0]*3 + [1.0]*3 + [0.01]*3)
    return EKF(x0, P0, sigma_alt=alt_std)


def test_ekf_predict_state_shape():
    ekf = _make_ekf()
    ekf.predict(np.zeros(3), np.zeros(3), 0.01)
    assert ekf.state.shape == (9,)
    assert ekf.std.shape == (9,)


def test_ekf_static_prediction():
    """Zero force, zero omega: position should not change (only gravity acts)."""
    ekf = _make_ekf(alt=1000.0)
    state_before = ekf.state.copy()
    # Pure gravity: no position change over tiny dt (first-order)
    ekf.predict(np.zeros(3), np.zeros(3), 0.001)
    # Horizontal pos unchanged
    np.testing.assert_allclose(ekf.state[0], state_before[0], atol=1e-4)
    np.testing.assert_allclose(ekf.state[1], state_before[1], atol=1e-4)


def test_ekf_covariance_grows_without_updates():
    ekf = _make_ekf()
    P_before = ekf.P.copy()
    for _ in range(100):
        ekf.predict(np.zeros(3), np.zeros(3), 0.01)
    assert np.trace(ekf.P) > np.trace(P_before)


def test_ekf_altimeter_update_reduces_alt_uncertainty():
    ekf = _make_ekf(alt=1000.0)
    # Inflate altitude uncertainty
    ekf.P[2, 2] = 1000.0
    std_before = ekf.std[2]
    ekf.update_altimeter(1000.0)
    assert ekf.std[2] < std_before


def test_ekf_altimeter_pulls_estimate():
    ekf = _make_ekf(alt=1000.0)
    ekf.update_altimeter(900.0)
    assert ekf.state[2] < 1000.0     # pulled down toward measurement


def test_ekf_horizon_update_reduces_attitude_uncertainty():
    ekf = _make_ekf()
    ekf.P[6, 6] = 1.0
    ekf.P[7, 7] = 1.0
    std6_before = ekf.std[6]
    std7_before = ekf.std[7]
    ekf.update_horizon(np.array([0.0, 0.0]))
    assert ekf.std[6] < std6_before
    assert ekf.std[7] < std7_before


def test_ekf_covariance_symmetric():
    ekf = _make_ekf()
    for _ in range(50):
        ekf.predict(np.random.randn(3) * 0.1, np.random.randn(3) * 0.01, 0.01)
        ekf.update_altimeter(float(np.random.randn() * 5))
    np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-10)


# ---------------------------------------------------------------------------
# Particle filter
# ---------------------------------------------------------------------------

def _make_pf(N=50, alt=500.0, rng_seed=42):
    x0 = np.zeros(9)
    x0[2] = alt
    P0 = np.diag([10.0]*3 + [1.0]*3 + [0.01]*3)
    return ParticleFilter(x0, P0, N=N, rng=np.random.default_rng(rng_seed))


def test_pf_state_shape():
    pf = _make_pf()
    assert pf.state.shape == (9,)


def test_pf_initial_mean_close_to_x0():
    x0 = np.zeros(9)
    x0[2] = 500.0
    P0 = np.diag([1.0]*9)
    pf = ParticleFilter(x0, P0, N=1000, rng=np.random.default_rng(0))
    np.testing.assert_allclose(pf.state, x0, atol=0.5)


def test_pf_weights_sum_to_one():
    pf = _make_pf()
    w = np.exp(pf.log_w)
    assert abs(w.sum() - 1.0) < 1e-10


def test_pf_ess_uniform_weights():
    pf = _make_pf(N=100)
    assert abs(pf.ess - 100.0) < 0.1


def test_pf_predict_no_crash():
    pf = _make_pf()
    pf.predict(np.array([0.0, 0.0, 30.0]), np.zeros(3), 0.01)
    assert not np.any(np.isnan(pf.particles))


def test_pf_altimeter_update_pulls_estimate():
    """Altimeter measurement should pull the PF altitude estimate toward it."""
    x0 = np.zeros(9)
    x0[2] = 1000.0
    # Large altitude uncertainty so particles span the measurement region
    P0 = np.diag([1.0, 1.0, 2500.0] + [1.0]*6)
    pf = ParticleFilter(x0, P0, N=500, rng=np.random.default_rng(99))
    alt_before = pf.state[2]
    z_meas = 800.0          # 200 m below mean
    pf.update_altimeter(z_meas)
    alt_after = pf.state[2]
    # Estimate must have moved toward the measurement
    assert abs(alt_after - z_meas) < abs(alt_before - z_meas)


def test_pf_ess_after_many_updates():
    """After many informative updates, ESS may drop — systematic resampling restores it."""
    pf = _make_pf(N=200, alt=1000.0)
    for _ in range(30):
        pf.update_altimeter(1000.0)   # measurement matches mean → weights stay uniform
    assert pf.ess > 50   # should not have collapsed


# ---------------------------------------------------------------------------
# Batch quaternion helpers in particle filter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("euler_batch", [
    [[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]],
    [[0.3, 0.5, -0.2], [-0.1, 0.4, 0.1]],
])
def test_batch_quat_round_trip(euler_batch):
    e = np.array(euler_batch)
    q = _batch_euler_to_quat(e)
    e_back = _batch_quat_to_euler(q)
    np.testing.assert_allclose(e_back, e, atol=1e-10)


def test_batch_euler_to_quat_unit_norm():
    rng = np.random.default_rng(1)
    euler = rng.uniform(-0.8, 0.8, (100, 3))
    q = _batch_euler_to_quat(euler)
    norms = np.linalg.norm(q, axis=1)
    np.testing.assert_allclose(norms, np.ones(100), atol=1e-12)
