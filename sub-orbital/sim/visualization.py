"""Multi-panel simulation results plotter."""
from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless safe; caller may switch to TkAgg/Qt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 – registers 3D projection


def plot_results(history: dict[str, Any], save_path: str = "results.png") -> None:
    """Generate a 3×3 panel figure and save it to *save_path*.

    Parameters
    ----------
    history : dict with keys
        't'           – 1-D time array (s)
        'truth'       – (N,9) true nav state [pos, vel, euler]
        'ekf'         – (N,9) EKF state estimate
        'ekf_std'     – (N,9) EKF 1-sigma standard deviations
        'pf'          – (N,9) particle filter state estimate
        'z_alt'       – (M,) altimeter measurements
        't_alt'       – (M,) altimeter measurement times
    """
    t      = np.asarray(history['t'])
    truth  = np.asarray(history['truth'])
    ekf    = np.asarray(history['ekf'])
    estd   = np.asarray(history['ekf_std'])
    pf     = np.asarray(history['pf'])
    z_alt  = np.asarray(history['z_alt'])
    t_alt  = np.asarray(history['t_alt'])

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Sub-orbital GNC Simulation — Sensor Fusion Results", fontsize=14,
                 fontweight='bold')
    gs  = GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

    # ── 3-D trajectory ──────────────────────────────────────────────────────
    ax3d = fig.add_subplot(gs[0:2, 0], projection='3d')
    ax3d.plot(truth[:,0]/1e3, truth[:,1]/1e3, truth[:,2]/1e3,
              'b',  lw=1.2, label='Truth')
    ax3d.plot(ekf[:,0]/1e3,   ekf[:,1]/1e3,   ekf[:,2]/1e3,
              'r--', lw=0.9, label='EKF')
    ax3d.plot(pf[:,0]/1e3,    pf[:,1]/1e3,    pf[:,2]/1e3,
              'g:',  lw=0.9, label='PF')
    # launch / apogee markers
    imax = int(np.argmax(truth[:,2]))
    ax3d.scatter(*truth[0, :3]/1e3,    color='blue',  s=40, zorder=5)
    ax3d.scatter(*truth[imax,:3]/1e3,  color='orange',s=60, marker='^', zorder=5)
    ax3d.set_xlabel('East (km)')
    ax3d.set_ylabel('North (km)')
    ax3d.set_zlabel('Alt (km)')
    ax3d.legend(fontsize=8)
    ax3d.set_title('3-D Trajectory', fontsize=10)

    # ── Altitude vs time ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, truth[:,2]/1e3, 'b', lw=1.2, label='Truth')
    ax.plot(t, ekf[:,2]/1e3,   'r--', lw=0.9, label='EKF')
    ax.plot(t, pf[:,2]/1e3,    'g:',  lw=0.9, label='PF')
    ax.scatter(t_alt, z_alt/1e3, s=2, c='gray', alpha=0.4, label='Altimeter')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Altitude')
    ax.legend(fontsize=7)
    ax.grid(True, ls='--', alpha=0.4)

    # ── Speed vs time ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    spd_truth = np.linalg.norm(truth[:,3:6], axis=1)
    spd_ekf   = np.linalg.norm(ekf[:,3:6],   axis=1)
    spd_pf    = np.linalg.norm(pf[:,3:6],    axis=1)
    ax.plot(t, spd_truth, 'b', lw=1.2, label='Truth')
    ax.plot(t, spd_ekf,   'r--', lw=0.9, label='EKF')
    ax.plot(t, spd_pf,    'g:',  lw=0.9, label='PF')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed')
    ax.legend(fontsize=7)
    ax.grid(True, ls='--', alpha=0.4)

    # ── Position error ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    pos_err_ekf = np.linalg.norm(ekf[:,0:3] - truth[:,0:3], axis=1)
    pos_err_pf  = np.linalg.norm(pf[:,0:3]  - truth[:,0:3], axis=1)
    ax.plot(t, pos_err_ekf, 'r', lw=1.0, label='EKF error')
    ax.plot(t, pos_err_pf,  'g', lw=1.0, label='PF error')
    ax.fill_between(t,
                    pos_err_ekf - 3*estd[:,2],
                    pos_err_ekf + 3*estd[:,2],
                    alpha=0.15, color='red', label='EKF 3σ (alt)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position error (m)')
    ax.set_title('3-D Position Error')
    ax.legend(fontsize=7)
    ax.grid(True, ls='--', alpha=0.4)

    # ── Velocity error ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    vel_err_ekf = np.linalg.norm(ekf[:,3:6] - truth[:,3:6], axis=1)
    vel_err_pf  = np.linalg.norm(pf[:,3:6]  - truth[:,3:6], axis=1)
    ax.plot(t, vel_err_ekf, 'r', lw=1.0, label='EKF')
    ax.plot(t, vel_err_pf,  'g', lw=1.0, label='PF')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity error (m/s)')
    ax.set_title('Velocity Error')
    ax.legend(fontsize=7)
    ax.grid(True, ls='--', alpha=0.4)

    # ── Pitch angle (clipped to boost+early-coast to avoid Euler singularity display issues)
    ax   = fig.add_subplot(gs[2, 0])
    mask = t <= 80.0
    ax.plot(t[mask], np.rad2deg(truth[mask, 7]), 'b', lw=1.2, label='Truth')
    ax.plot(t[mask], np.rad2deg(ekf[mask,   7]), 'r--', lw=0.9, label='EKF')
    ax.plot(t[mask], np.rad2deg(pf[mask,    7]), 'g:',  lw=0.9, label='PF')
    t_burnout = history.get('t_burnout', 41.0)
    ax.axvline(t_burnout, color='orange', ls='--', lw=0.8, label=f'Burnout {t_burnout:.0f}s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch θ (deg)')
    ax.set_title('Pitch Angle (boost phase)')
    ax.legend(fontsize=7)
    ax.grid(True, ls='--', alpha=0.4)

    # ── Attitude error (per-axis, boost phase only) ──────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    att_err_ekf = np.rad2deg(np.abs(
        np.arctan2(np.sin(ekf[mask,6:9] - truth[mask,6:9]),
                   np.cos(ekf[mask,6:9] - truth[mask,6:9]))))
    att_err_pf  = np.rad2deg(np.abs(
        np.arctan2(np.sin(pf[mask, 6:9] - truth[mask,6:9]),
                   np.cos(pf[mask, 6:9] - truth[mask,6:9]))))
    ax.plot(t[mask], att_err_ekf[:, 1], 'r', lw=1.0, label='EKF pitch')
    ax.plot(t[mask], att_err_pf[:,  1], 'g', lw=1.0, label='PF pitch')
    ax.axvline(t_burnout, color='orange', ls='--', lw=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch error (deg)')
    ax.set_title('Pitch Estimation Error (boost phase)')
    ax.legend(fontsize=7)
    ax.grid(True, ls='--', alpha=0.4)

    # ── EKF 3-sigma bounds for altitude ─────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    alt_err_ekf = ekf[:,2] - truth[:,2]
    sig_alt     = estd[:,2]
    ax.plot(t, alt_err_ekf,      'r',  lw=1.0, label='EKF alt error')
    ax.plot(t,  3*sig_alt,        'r--', lw=0.7, label='±3σ')
    ax.plot(t, -3*sig_alt,        'r--', lw=0.7)
    ax.fill_between(t, -3*sig_alt, 3*sig_alt, alpha=0.15, color='red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (m)')
    ax.set_title('EKF Altitude Error vs 3σ')
    ax.legend(fontsize=7)
    ax.grid(True, ls='--', alpha=0.4)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path
