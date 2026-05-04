"""Sub-orbital rocket GNC simulator.

Run:   uv run python main.py
       uv run python main.py --dt 0.02 --n-particles 200 --no-plot

Physics  : 6-DOF ENU, ZYX Euler, RK4 integration, ISA atmosphere, TVC
Sensors  : IMU (100 Hz), Altimeter (10 Hz), Horizon tracker (10 Hz)
Filters  : EKF (9-state, numerical Jacobians)
           Particle filter (SIR, 500 particles, log-weights)
"""
from __future__ import annotations

import argparse
import time as _time

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

from sim.dynamics import RocketConfig, RocketDynamics, RocketState
from sim.environment import gravity
from sim.filters.ekf import EKF
from sim.filters.particle import ParticleFilter
from sim.gnc.control import TVCController
from sim.gnc.guidance import GravityTurnGuidance
from sim.sensors.altimeter import Altimeter
from sim.sensors.horizon_tracker import HorizonTracker
from sim.sensors.imu import IMU

console = Console()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sub-orbital GNC simulator")
    p.add_argument("--dt",           type=float, default=0.01,  help="Simulation step (s)")
    p.add_argument("--t-max",        type=float, default=360.0, help="Max simulation time (s)")
    p.add_argument("--meas-rate",    type=int,   default=10,    help="Altimeter/horizon rate (Hz)")
    p.add_argument("--n-particles",  type=int,   default=500,   help="Particle filter count")
    p.add_argument("--seed",         type=int,   default=7,     help="RNG seed")
    p.add_argument("--no-plot",      action="store_true",       help="Skip matplotlib output")
    p.add_argument("--save-plot",    type=str,   default="results.png")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def make_initial_state(cfg: RocketConfig) -> RocketState:
    return RocketState(
        pos=np.array([0.0, 0.0, 0.0]),
        vel=np.zeros(3),
        euler=np.zeros(3),         # vertical, nose up, heading East
        omega=np.zeros(3),
        mass=cfg.mass_dry + cfg.mass_prop,
        time=0.0,
    )


def initial_filter_state(true_state: RocketState, rng: np.random.Generator,
                          pos_std: float = 5.0, vel_std: float = 1.0,
                          att_std_deg: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Start filter with small random offset from truth."""
    x0 = true_state.nav_state()
    noise = rng.standard_normal(9)
    stds  = np.array([pos_std]*3 + [vel_std]*3 + [np.deg2rad(att_std_deg)]*3)
    x0   += noise * stds
    P0    = np.diag(stds ** 2)
    return x0, P0


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    cfg = RocketConfig()

    dynamics  = RocketDynamics(cfg)
    guidance  = GravityTurnGuidance(t_burnout=cfg.mass_prop / dynamics.mdot + 1.0)
    control   = TVCController()
    imu       = IMU(rng=np.random.default_rng(args.seed + 1))
    altimeter = Altimeter(rng=np.random.default_rng(args.seed + 2))
    horizon   = HorizonTracker(rng=np.random.default_rng(args.seed + 3))

    state = make_initial_state(cfg)
    x0, P0 = initial_filter_state(state, rng)
    ekf = EKF(x0, P0)
    pf  = ParticleFilter(x0, P0, N=args.n_particles,
                         rng=np.random.default_rng(args.seed + 4))

    dt       = args.dt
    meas_dt  = 1.0 / args.meas_rate      # s between slow-sensor updates
    next_meas = meas_dt                   # time of next slow-sensor update

    # Pre-allocate history arrays (guessed max steps)
    max_steps = int(args.t_max / dt) + 10
    t_hist    = np.empty(max_steps)
    truth_h   = np.empty((max_steps, 9))
    ekf_h     = np.empty((max_steps, 9))
    ekf_std_h = np.empty((max_steps, 9))
    pf_h      = np.empty((max_steps, 9))
    z_alt_list: list[float] = []
    t_alt_list: list[float] = []

    tvc = np.zeros(2)

    console.rule("[bold cyan]Sub-orbital GNC Simulator")
    console.print(f"  Rocket: {cfg.mass_dry+cfg.mass_prop:.0f} kg  |  "
                  f"T={cfg.thrust/1e3:.1f} kN  |  Isp={cfg.isp:.0f} s  |  "
                  f"Burn {cfg.mass_prop/dynamics.mdot:.1f} s")
    console.print(f"  dt={dt*1000:.0f} ms  |  slow-sensors @ {args.meas_rate} Hz  |  "
                  f"PF N={args.n_particles}")
    console.print()

    step = 0
    wall_start = _time.perf_counter()

    with Progress(SpinnerColumn(), "[progress.description]{task.description}",
                  TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Simulating…", total=None)

        while state.time < args.t_max and state.pos[2] >= 0.0:
            burning = state.mass > cfg.mass_dry + 0.5

            # ── GNC ─────────────────────────────────────────────────────────
            target = guidance.target_attitude(state)
            tvc    = control.compute(target, state, burning)

            # ── True dynamics (RK4) ─────────────────────────────────────────
            state = dynamics.step(state, tvc, dt)

            # ── IMU measurement (every step) ─────────────────────────────────
            f_meas, omega_meas = imu.measure(state, dynamics, tvc, dt)

            # ── Filter prediction ────────────────────────────────────────────
            ekf.predict(f_meas, omega_meas, dt)
            pf.predict(f_meas, omega_meas, dt)

            # ── Slow-sensor measurement update ───────────────────────────────
            if state.time >= next_meas:
                z_a = altimeter.measure(state)
                z_h = horizon.measure(state)

                ekf.update_altimeter(z_a)
                ekf.update_horizon(z_h)
                pf.update_altimeter(z_a)
                pf.update_horizon(z_h)

                z_alt_list.append(z_a)
                t_alt_list.append(state.time)
                next_meas += meas_dt

            # ── Record history ───────────────────────────────────────────────
            t_hist[step]    = state.time
            truth_h[step]   = state.nav_state()
            ekf_h[step]     = ekf.state
            ekf_std_h[step] = ekf.std
            pf_h[step]      = pf.state
            step += 1

            if step % 500 == 0:
                prog.update(task, description=(
                    f"t={state.time:.0f}s  alt={state.pos[2]/1e3:.1f} km  "
                    f"|v|={np.linalg.norm(state.vel):.0f} m/s  "
                    f"θ={np.rad2deg(state.euler[1]):.1f}°  "
                    f"ESS={pf.ess:.0f}"))

    wall_time = _time.perf_counter() - wall_start

    # Trim arrays
    t_hist    = t_hist[:step]
    truth_h   = truth_h[:step]
    ekf_h     = ekf_h[:step]
    ekf_std_h = ekf_std_h[:step]
    pf_h      = pf_h[:step]

    apogee    = float(np.max(truth_h[:, 2]))
    t_apogee  = float(t_hist[int(np.argmax(truth_h[:, 2]))])
    pos_err_final_ekf = float(np.linalg.norm(ekf_h[-1, :3] - truth_h[-1, :3]))
    pos_err_final_pf  = float(np.linalg.norm(pf_h[-1, :3]  - truth_h[-1, :3]))
    rms_alt_ekf = float(np.sqrt(np.mean((ekf_h[:, 2] - truth_h[:, 2]) ** 2)))
    rms_alt_pf  = float(np.sqrt(np.mean((pf_h[:, 2]  - truth_h[:, 2]) ** 2)))

    # Summary table
    tbl = Table(title="Simulation Summary", show_header=True)
    tbl.add_column("Metric",         style="cyan")
    tbl.add_column("Value",          style="white")
    tbl.add_row("Steps simulated",   f"{step:,}")
    tbl.add_row("Wall time",         f"{wall_time:.2f} s")
    tbl.add_row("Apogee",            f"{apogee/1e3:.2f} km  at t={t_apogee:.1f} s")
    tbl.add_row("Final pos err EKF", f"{pos_err_final_ekf:.1f} m")
    tbl.add_row("Final pos err PF",  f"{pos_err_final_pf:.1f} m")
    tbl.add_row("RMS altitude EKF",  f"{rms_alt_ekf:.2f} m")
    tbl.add_row("RMS altitude PF",   f"{rms_alt_pf:.2f} m")
    console.print(tbl)

    t_burnout = cfg.mass_prop / dynamics.mdot
    history = dict(
        t=t_hist,
        truth=truth_h,
        ekf=ekf_h,
        ekf_std=ekf_std_h,
        pf=pf_h,
        z_alt=np.array(z_alt_list),
        t_alt=np.array(t_alt_list),
        t_burnout=t_burnout,
    )
    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    history = run(args)

    if not args.no_plot:
        from sim.visualization import plot_results
        import matplotlib
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass

        out = plot_results(history, save_path=args.save_plot)
        console.print(f"\n[green]Plot saved → {args.save_plot}[/green]")

        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
