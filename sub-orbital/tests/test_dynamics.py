"""Tests for dynamics, environment, and quaternion helpers."""
import numpy as np
import pytest

from sim.dynamics import (
    RocketConfig, RocketDynamics, RocketState,
    rot_matrix, euler_to_quat, quat_to_euler, rot_from_quat, quat_kinematics,
)
from sim.environment import atmosphere, gravity


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def test_atmosphere_sea_level():
    rho, P, T = atmosphere(0.0)
    assert abs(T - 288.15)   < 0.01
    assert abs(P - 101_325)  < 1.0
    assert abs(rho - 1.225)  < 0.001


def test_atmosphere_11km():
    _, _, T = atmosphere(11_000.0)
    assert abs(T - 216.65) < 0.01


def test_atmosphere_50km():
    rho, _, _ = atmosphere(50_000.0)
    assert rho < 0.002          # very thin


def test_gravity_sea_level():
    assert abs(gravity(0.0) - 9.80665) < 1e-5


def test_gravity_decreases_with_altitude():
    assert gravity(100_000.0) < gravity(0.0)


# ---------------------------------------------------------------------------
# Euler helpers
# ---------------------------------------------------------------------------

def test_rot_matrix_identity():
    R = rot_matrix(np.zeros(3))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("euler", [
    [0.0,  0.0, 0.0],
    [0.1, -0.2, 0.3],
    [0.5,  0.7, -0.4],
])
def test_rot_matrix_orthogonal(euler):
    R = rot_matrix(np.array(euler))
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("euler", [
    [0.0,   0.0,  0.0],
    [0.1,  -0.2,  0.3],
    [0.5,   0.7, -0.4],
    [-0.3,  0.85, 0.2],   # close to but not at θ=90°
])
def test_quat_round_trip(euler):
    e = np.array(euler)
    q = euler_to_quat(e)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-12
    np.testing.assert_allclose(quat_to_euler(q), e, atol=1e-10)


def test_euler_to_quat_unit_norm():
    for _ in range(20):
        e = np.random.uniform(-0.8, 0.8, 3)
        q = euler_to_quat(e)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-12


@pytest.mark.parametrize("euler", [
    [0.0,  0.0,  0.0],
    [0.2, -0.3,  0.4],
    [0.5,  0.7, -0.4],
])
def test_rot_from_quat_matches_rot_matrix(euler):
    e = np.array(euler)
    R_euler = rot_matrix(e)
    R_quat  = rot_from_quat(euler_to_quat(e))
    np.testing.assert_allclose(R_quat, R_euler, atol=1e-10)


def test_quat_kinematics_identity():
    """At identity quaternion pure-y rotation, q̇ encodes rotation about y."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.0, 1.0, 0.0])   # 1 rad/s around body-y (pitch)
    dq = quat_kinematics(q, omega)
    # Expected: dq = 0.5 * [0, 0, 1, 0]
    np.testing.assert_allclose(dq, [0.0, 0.0, 0.5, 0.0], atol=1e-12)


def test_quat_kinematics_zero_omega():
    q = euler_to_quat(np.array([0.1, 0.2, 0.3]))
    dq = quat_kinematics(q, np.zeros(3))
    np.testing.assert_allclose(dq, np.zeros(4), atol=1e-12)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def _make_state(cfg, mass=None, pos=None, vel=None, euler=None):
    return RocketState(
        pos   = pos   if pos   is not None else np.zeros(3),
        vel   = vel   if vel   is not None else np.zeros(3),
        euler = euler if euler is not None else np.zeros(3),
        omega = np.zeros(3),
        mass  = mass  if mass  is not None else cfg.mass_dry,
        time  = 0.0,
    )


def test_no_thrust_no_drag_ballistic():
    """In vacuum (cd=0) without thrust, only gravity acts: Δv = g·dt vertically."""
    cfg = RocketConfig(cd=0.0, thrust=0.0)
    dyn = RocketDynamics(cfg)
    state = _make_state(cfg, pos=np.array([0.0, 0.0, 1000.0]),
                        vel=np.array([0.0, 0.0, 100.0]))
    dt = 0.01
    new_state = dyn.step(state, np.zeros(2), dt)
    g0 = gravity(1000.0)
    expected_vz = 100.0 - g0 * dt
    assert abs(new_state.vel[2] - expected_vz) < 0.01     # <1 cm/s error
    # Horizontal velocity unchanged
    np.testing.assert_allclose(new_state.vel[:2], [0.0, 0.0], atol=1e-8)


def test_mass_decreases_during_burn():
    cfg = RocketConfig()
    dyn = RocketDynamics(cfg)
    state = _make_state(cfg, mass=cfg.mass_dry + cfg.mass_prop)
    new_state = dyn.step(state, np.zeros(2), 1.0)
    assert new_state.mass < state.mass


def test_mass_clamps_at_dry():
    cfg = RocketConfig()
    dyn = RocketDynamics(cfg)
    state = _make_state(cfg, mass=cfg.mass_dry)    # no propellant
    new_state = dyn.step(state, np.zeros(2), 1.0)
    assert new_state.mass == cfg.mass_dry


def test_step_time_advances():
    cfg = RocketConfig()
    dyn = RocketDynamics(cfg)
    state = _make_state(cfg)
    dt = 0.05
    new_state = dyn.step(state, np.zeros(2), dt)
    assert abs(new_state.time - dt) < 1e-12


def test_specific_force_no_drag_no_tvc():
    """Without drag or TVC, specific force equals thrust/mass upward."""
    cfg = RocketConfig(cd=0.0)
    dyn = RocketDynamics(cfg)
    state = _make_state(cfg, mass=cfg.mass_dry + cfg.mass_prop)
    f = dyn.specific_force_body(state, np.zeros(2))
    expected_fz = cfg.thrust / state.mass
    assert abs(f[2] - expected_fz) < 1e-6


def test_vertical_launch_stays_in_plane():
    """Symmetric launch with zero TVC should stay in the North-East plane."""
    cfg = RocketConfig()
    dyn = RocketDynamics(cfg)
    state = _make_state(cfg, mass=cfg.mass_dry + cfg.mass_prop)
    for _ in range(100):
        state = dyn.step(state, np.zeros(2), 0.01)
    np.testing.assert_allclose(state.pos[1], 0.0, atol=1e-6)   # no North drift


def test_quaternion_step_no_singularity():
    """Simulate through horizontal flight (θ near 90°) without NaN."""
    cfg = RocketConfig(thrust=0.0, cd=0.0)
    dyn = RocketDynamics(cfg)
    # Start pitched at 80°, omega tilting further
    state = RocketState(
        pos=np.zeros(3), vel=np.array([100.0, 0.0, 10.0]),
        euler=np.array([0.0, np.deg2rad(80.0), 0.0]),
        omega=np.array([0.0, 0.05, 0.0]),   # pitching toward 90°
        mass=cfg.mass_dry, time=0.0,
    )
    for _ in range(500):        # integrate through θ=90° region
        state = dyn.step(state, np.zeros(2), 0.01)
    assert not np.any(np.isnan(state.euler))
    assert not np.any(np.isnan(state.vel))
    assert not np.any(np.isnan(state.pos))
