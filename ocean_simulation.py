"""
ocean_simulation.py
===================
Full ocean simulation integrating all modules.

Produces:
  1. Animated ocean surface (random sea state from JONSWAP spectrum)
  2. Subsurface velocity and pressure fields
  3. Particle tracking (Lagrangian drift)
  4. Wave-obstacle interaction (island scattering)
  5. Summary statistics panel

Run with: python ocean_simulation.py
Toggle scenarios via SCENARIO variable below.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Our modules
from wave_theory import (
    LinearWave, WaveField, waves_from_jonswap, stokes_elevation,
    jonswap_spectrum, dispersion_curves, G
)
from complex_potential import compute_fields, cylinder_flow, corner_flow
from coastal_scattering import (
    island_wave_field, headland_flow, two_island_flow,
    compute_wave_runup, resonance_frequencies
)
ocean_colors = [
    '#0a1628',   # deep midnight blue
    '#0d2b52',   # deep ocean
    '#1a4f7e',   # mid ocean
    '#2878b5',   # surface blue
    '#5ab4d9',   # light surface
    '#a8ddf0',   # crest
    '#e8f6fd',   # whitecap
]
OCEAN_CMAP = LinearSegmentedColormap.from_list('ocean', ocean_colors, N=256)

pressure_colors = ['#1a3a5c', '#2878b5', '#7fbddb', '#f0f0f0', '#f5a27b', '#d45a2a', '#8b1a0a']
PRESSURE_CMAP = LinearSegmentedColormap.from_list('pressure', pressure_colors, N=256)

class OceanConfig:
    Lx: float = 200.0         # domain width (m)
    Lz: float = 60.0          # domain depth (m)
    Nx: int = 300              # horizontal resolution
    Nz: int = 100              # vertical resolution
    surface_Nx: int = 600      # surface resolution
    Hs: float = 3.5            # significant wave height (m)
    Tp: float = 12.0           # peak period (s)
    gamma: float = 3.3         # JONSWAP peak enhancement
    h: float = 50.0            # water depth (m)
    n_waves: int = 60          # number of spectral components
    island_radius: float = 15.0
    island_x: float = 0.0
    island_y: float = 0.0
    dt: float = 0.4            # time step per frame (s)
    fps: int = 25
    n_frames: int = 200
    n_particles: int = 30


def build_random_sea(cfg: OceanConfig, seed: int = 42) -> WaveField:
    """Build a JONSWAP random sea state."""
    print(f"  Building sea state: Hs={cfg.Hs}m, Tp={cfg.Tp}s, h={cfg.h}m, "
          f"n_waves={cfg.n_waves}")
    wf = waves_from_jonswap(
        Hs=cfg.Hs, Tp=cfg.Tp, gamma=cfg.gamma,
        h=cfg.h, n_waves=cfg.n_waves, seed=seed
    )
    print(f"  Significant wave height (computed): {wf.significant_wave_height():.2f} m")
    return wf
class ParticleTracker:
    """
    Track Lagrangian particles using the wave velocity field.
    Each particle follows: dx/dt = u(x,y,t), dy/dt = v(x,y,t).
    Integrated with 4th-order Runge-Kutta.
    This reveals the Stokes drift — particles drift slowly in the wave direction.
    """

    def __init__(self, x0: np.ndarray, y0: np.ndarray, wf: WaveField):
        self.x = x0.copy()
        self.y = y0.copy()
        self.wf = wf
        self.history_x = [x0.copy()]
        self.history_y = [y0.copy()]

    def velocity(self, x, y, t):
        u = self.wf.velocity_u(x, y, t)
        v = self.wf.velocity_v(x, y, t)
        return u, v

    def step(self, t: float, dt: float):
        """RK4 time step."""
        u1, v1 = self.velocity(self.x, self.y, t)
        u2, v2 = self.velocity(self.x + 0.5*dt*u1, self.y + 0.5*dt*v1, t + 0.5*dt)
        u3, v3 = self.velocity(self.x + 0.5*dt*u2, self.y + 0.5*dt*v2, t + 0.5*dt)
        u4, v4 = self.velocity(self.x + dt*u3, self.y + dt*v3, t + dt)

        self.x += dt * (u1 + 2*u2 + 2*u3 + u4) / 6
        self.y += dt * (v1 + 2*v2 + 2*v3 + v4) / 6

        # Reflect off bottom and free surface (approximate)
        self.y = np.clip(self.y, -45, 0)
        self.history_x.append(self.x.copy())
        self.history_y.append(self.y.copy())

def run_full_simulation(cfg: OceanConfig = None, save_path: str = None):
    if cfg is None:
        cfg = OceanConfig()

    print("=== Ocean Simulation ===\n")
    print("Phase 1: Building wave field from JONSWAP spectrum...")
    wf = build_random_sea(cfg)
    x_surf = np.linspace(-cfg.Lx/2, cfg.Lx/2, cfg.surface_Nx)
    x_sub = np.linspace(-cfg.Lx/2, cfg.Lx/2, cfg.Nx)
    z_sub = np.linspace(-cfg.Lz, 0, cfg.Nz)
    X_sub, Z_sub = np.meshgrid(x_sub, z_sub)
    times = [0, 2, 5, 10]
    t_anim = 8.0 
    eta = wf.surface_elevation(x_surf, t_anim)
    eta_stokes = stokes_elevation(x_surf, t_anim,
                                   wf.waves[np.argmax([w.amplitude for w in wf.waves])],
                                   stokes_order=3)
    print("Phase 2: Computing subsurface velocity field...")
    u_field = wf.velocity_u(X_sub, Z_sub, t_anim)
    v_field = wf.velocity_v(X_sub, Z_sub, t_anim)
    speed_field = np.sqrt(u_field**2 + v_field**2)
    pressure_field = -0.5 * speed_field**2
    pressure_field += G * (-Z_sub)
    print("Phase 3: Computing island wave scattering...")
    F_island = island_wave_field(radius=1.0, wave_speed=1.0)
    xg = np.linspace(-4, 4, 400)
    yg = np.linspace(-4, 4, 400)
    Xg, Yg = np.meshgrid(xg, yg)
    scatter_fields = compute_fields(F_island, Xg, Yg)
    mask_island = Xg**2 + Yg**2 < 1.0
    print("Phase 4: Particle tracking (Stokes drift)...")
    rng = np.random.default_rng(0)
    x0 = rng.uniform(-cfg.Lx/2 + 10, -20, cfg.n_particles)
    y0 = rng.uniform(-30, -2, cfg.n_particles)
    tracker = ParticleTracker(x0, y0, wf)
    n_track_steps = 60
    for i in range(n_track_steps):
        tracker.step(t=i * cfg.dt, dt=cfg.dt)
    theta_runup, runup = compute_wave_runup(radius=1.0)
    f_arr = np.linspace(0.02, 0.5, 400)
    S_arr = jonswap_spectrum(f_arr, Hs=cfg.Hs, Tp=cfg.Tp, gamma=cfg.gamma)
    k_arr, disp_curves = dispersion_curves([10, 50, 200, np.inf])
    print("Phase 5: Rendering figure...")
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#0a1628')

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.42, wspace=0.38,
                           left=0.06, right=0.97,
                           top=0.94, bottom=0.05)

    title_kw = dict(fontsize=11, color='#e0eaf5', pad=8)
    label_kw = dict(fontsize=9, color='#8aafc8')
    tick_kw = dict(colors='#8aafc8', labelsize=8)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#0d1f3c')
    for ti, alpha_val in zip([0, 3, 6], [0.15, 0.25, 0.4]):
        eta_t = wf.surface_elevation(x_surf, ti)
        ax1.plot(x_surf, eta_t, color='#5ab4d9', alpha=alpha_val, linewidth=0.8)


    ax1.fill_between(x_surf, eta, -8, where=(eta > -8),
                      color='#1a4f7e', alpha=0.4)
    ax1.plot(x_surf, eta, color='#5ab4d9', linewidth=1.8, label='η(x,t)')

    dominant_idx = np.argmax([w.amplitude for w in wf.waves])
    dominant = wf.waves[dominant_idx]
    eta_dom = wf.surface_elevation(x_surf * 0 + 0, t_anim)  # full field
    ax1.axhline(y=0, color='#3a5a7c', linewidth=0.7, linestyle='--', alpha=0.5)
    ax1.axhline(y=cfg.Hs/2, color='#e06c1f', linewidth=1, linestyle=':',
                alpha=0.7, label=f'Hs/2 = {cfg.Hs/2:.1f} m')
    ax1.axhline(y=-cfg.Hs/2, color='#e06c1f', linewidth=1, linestyle=':', alpha=0.7)

    ax1.set_xlim(-cfg.Lx/2, cfg.Lx/2)
    ax1.set_ylim(-8, 8)
    ax1.set_title(f'Ocean surface η(x,t) — JONSWAP random sea (Hs={cfg.Hs}m, Tp={cfg.Tp}s)',
                  **title_kw)
    ax1.set_xlabel('x (m)', **label_kw)
    ax1.set_ylabel('η (m)', **label_kw)
    ax1.tick_params(**tick_kw)
    ax1.legend(fontsize=8, loc='upper right', facecolor='#0d2b52', labelcolor='#e0eaf5')
    ax1.spines[:].set_color('#3a5a7c')
    ax1.annotate(f'Hs = {cfg.Hs:.1f} m', xy=(80, cfg.Hs/2 + 0.3),
                 fontsize=9, color='#e06c1f')
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.set_facecolor('#0d1f3c')

    im2 = ax2.pcolormesh(X_sub, Z_sub, speed_field,
                          cmap='YlOrRd', shading='auto', alpha=0.9,
                          vmin=0, vmax=speed_field.max() * 0.8)
    skip = 12
    ax2.quiver(X_sub[::skip, ::skip], Z_sub[::skip, ::skip],
               u_field[::skip, ::skip], v_field[::skip, ::skip],
               color='white', alpha=0.6, scale=6, width=0.002)
    ax2.fill_between(x_surf, eta, 5, color='#1a4f7e', alpha=0.3)
    ax2.plot(x_surf, eta, color='#5ab4d9', linewidth=1.5)
    ax2.axhline(y=0, color='#5ab4d9', linewidth=0.5, linestyle='--', alpha=0.4)
    ax2.fill_between(x_sub, -cfg.Lz, -cfg.Lz + 2, color='#4a3520', alpha=0.8)

    cb2 = plt.colorbar(im2, ax=ax2, shrink=0.8, label='Speed |V| (m/s)')
    cb2.ax.yaxis.label.set_color('#8aafc8')
    cb2.ax.tick_params(colors='#8aafc8', labelsize=7)

    ax2.set_xlim(-cfg.Lx/2, cfg.Lx/2)
    ax2.set_ylim(-cfg.Lz, 6)
    ax2.set_title('Subsurface velocity field', **title_kw)
    ax2.set_xlabel('x (m)', **label_kw)
    ax2.set_ylabel('depth z (m)', **label_kw)
    ax2.tick_params(**tick_kw)
    ax2.spines[:].set_color('#3a5a7c')
    ax2.text(-cfg.Lx/2 + 5, -cfg.h + 2, f'h = {cfg.h:.0f} m',
             fontsize=8, color='#8aafc8', va='top')
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_facecolor('#0d1f3c')

    psi = np.ma.masked_where(mask_island, scatter_fields['psi'])
    pressure_sc = np.ma.masked_where(mask_island, scatter_fields['pressure'])

    im3 = ax3.pcolormesh(Xg, Yg, pressure_sc, cmap=PRESSURE_CMAP,
                          shading='auto', alpha=0.85)
    levels = np.linspace(np.percentile(psi.compressed(), 5),
                          np.percentile(psi.compressed(), 95), 22)
    ax3.contour(Xg, Yg, psi, levels=levels, colors='white',
                linewidths=0.6, alpha=0.55)

    island_circle = Circle((0, 0), 1.0, color='#2c4a1e', zorder=10,
                             linewidth=2, edgecolor='#5a8a3e')
    ax3.add_patch(island_circle)

    cb3 = plt.colorbar(im3, ax=ax3, shrink=0.8, label='Pressure')
    cb3.ax.yaxis.label.set_color('#8aafc8')
    cb3.ax.tick_params(colors='#8aafc8', labelsize=7)

    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    ax3.set_aspect('equal')
    ax3.set_title('Island scattering', **title_kw)
    ax3.set_xlabel('x (m)', **label_kw)
    ax3.set_ylabel('y (m)', **label_kw)
    ax3.tick_params(**tick_kw)
    ax3.spines[:].set_color('#3a5a7c')

    # Arrow
    ax3.annotate('', xy=(-3.5, 0), xytext=(-3.9, 0),
                 arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    ax4 = fig.add_subplot(gs[2, :2])
    ax4.set_facecolor('#0d1f3c')
    im4 = ax4.pcolormesh(X_sub, Z_sub, speed_field, cmap=OCEAN_CMAP,
                          shading='auto', alpha=0.7, vmin=0)

    trail_len = min(20, len(tracker.history_x))
    n_show = min(n_track_steps, len(tracker.history_x))
    for i in range(cfg.n_particles):
        px = [tracker.history_x[j][i] for j in range(n_show)]
        py = [tracker.history_y[j][i] for j in range(n_show)]
        # Fade trail
        for j in range(len(px) - 1):
            alpha = 0.3 + 0.7 * j / (len(px) - 1)
            ax4.plot(px[j:j+2], py[j:j+2], color='#f0a040', alpha=alpha,
                     linewidth=0.8)
        ax4.plot(px[-1], py[-1], 'o', color='#f0d060', markersize=3, zorder=5)
    ax4.plot(x0, y0, 's', color='#40d0f0', markersize=3, alpha=0.7,
             label='Start', zorder=6)

    ax4.fill_between(x_surf, eta, 5, color='#1a4f7e', alpha=0.25)
    ax4.plot(x_surf, eta, color='#5ab4d9', linewidth=1.2, alpha=0.8)
    ax4.fill_between(x_sub, -cfg.Lz, -cfg.Lz+2, color='#4a3520', alpha=0.8)

    ax4.set_xlim(-cfg.Lx/2, cfg.Lx/2)
    ax4.set_ylim(-cfg.Lz, 6)
    ax4.set_title('Lagrangian particle tracking (Stokes Drift)\n'
                  'RK4 integration of V = F′(z)', **title_kw)
    ax4.set_xlabel('x (m)', **label_kw)
    ax4.set_ylabel('depth z (m)', **label_kw)
    ax4.tick_params(**tick_kw)
    ax4.legend(fontsize=8, loc='upper right', facecolor='#0d2b52', labelcolor='#e0eaf5')
    ax4.spines[:].set_color('#3a5a7c')

    drift_dx = np.mean([tracker.history_x[-1][i] - tracker.history_x[0][i]
                         for i in range(cfg.n_particles)])
    ax4.text(0, -cfg.Lz + 5, f'Mean Stokes drift: {drift_dx:.2f} m',
             fontsize=8, color='#f0a040', ha='center')

    ax5 = fig.add_subplot(gs[2, 2])
    ax5.set_facecolor('#0d1f3c')
    for w in wf.waves:
        fi = w.omega / (2 * np.pi)
        ax5.bar(fi, 0.5 * w.amplitude**2 / 0.01, width=0.008,
                color='#5ab4d9', alpha=0.4, linewidth=0)
    f_fine = np.linspace(0.02, 0.5, 400)
    S_fine = jonswap_spectrum(f_fine, Hs=cfg.Hs, Tp=cfg.Tp, gamma=cfg.gamma)
    ax5.plot(f_fine, S_fine, color='#e06c1f', linewidth=2.5, label='JONSWAP S(f)')
    ax5.axvline(x=1/cfg.Tp, color='#f0d060', linestyle='--', linewidth=1.5,
                label=f'fp = 1/Tp = {1/cfg.Tp:.3f} Hz')

    ax5.set_xlim(0.02, 0.45)
    ax5.set_ylim(bottom=0)
    ax5.set_title(f'Wave spectrum S(f)\nγ={cfg.gamma}, Hs={cfg.Hs}m, Tp={cfg.Tp}s',
                  **title_kw)
    ax5.set_xlabel('Frequency f (Hz)', **label_kw)
    ax5.set_ylabel('S(f) (m²/Hz)', **label_kw)
    ax5.tick_params(**tick_kw)
    ax5.legend(fontsize=8, facecolor='#0d2b52', labelcolor='#e0eaf5')
    ax5.spines[:].set_color('#3a5a7c')
    ax5.set_facecolor('#0d1f3c')
    fig.text(0.5, 0.975,
             'Ocean Wave Simulation',
             ha='center', va='top', fontsize=14, color='#e0eaf5', fontweight='bold')
    fig.text(0.5, 0.958,
             'F(z) = φ + iψ  |  ∇²φ = 0  |  ω² = gk tanh(kh)  |  V = F′(z)',
             ha='center', va='top', fontsize=10, color='#5ab4d9', style='italic')

    out_path = save_path or './ocean_simulation.png'
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\nSaved → {out_path}")
    return fig
def run_animation(cfg: OceanConfig = None, save_path: str = None, n_frames: int = 80):
    if cfg is None:
        cfg = OceanConfig()

    print("=== Ocean Animation ===")
    wf = build_random_sea(cfg, seed=7)

    x_surf = np.linspace(-cfg.Lx/2, cfg.Lx/2, 500)
    x_sub = np.linspace(-cfg.Lx/2, cfg.Lx/2, 200)
    z_sub = np.linspace(-cfg.Lz, 0, 80)
    X_sub, Z_sub = np.meshgrid(x_sub, z_sub)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                              gridspec_kw={'height_ratios': [1.2, 1.8]})
    fig.patch.set_facecolor('#0a1628')
    for ax in axes:
        ax.set_facecolor('#0d1f3c')
        ax.spines[:].set_color('#3a5a7c')
        ax.tick_params(colors='#8aafc8', labelsize=8)
    t0 = 0.0
    eta0 = wf.surface_elevation(x_surf, t0)
    u0 = wf.velocity_u(X_sub, Z_sub, t0)
    v0 = wf.velocity_v(X_sub, Z_sub, t0)
    speed0 = np.sqrt(u0**2 + v0**2)
    ax = axes[0]
    ax.set_xlim(-cfg.Lx/2, cfg.Lx/2)
    ax.set_ylim(-8, 8)
    surf_fill = ax.fill_between(x_surf, eta0, -8, color='#1a4f7e', alpha=0.4)
    surf_line, = ax.plot(x_surf, eta0, color='#5ab4d9', linewidth=1.8)
    ax.axhline(0, color='#3a5a7c', lw=0.7, linestyle='--', alpha=0.5)
    ax.axhline(cfg.Hs/2, color='#e06c1f', lw=1, linestyle=':', alpha=0.7)
    ax.axhline(-cfg.Hs/2, color='#e06c1f', lw=1, linestyle=':', alpha=0.7)
    time_text = ax.text(0.02, 0.92, 't = 0.0 s', transform=ax.transAxes,
                         fontsize=10, color='#e0eaf5')
    ax.set_ylabel('η (m)', color='#8aafc8', fontsize=9)
    ax.set_title('Ocean Surface η(x,t)', color='#e0eaf5', fontsize=11)
    ax2 = axes[1]
    im = ax2.pcolormesh(X_sub, Z_sub, speed0, cmap='YlOrRd', shading='auto',
                         vmin=0, vmax=speed0.max() * 1.2)
    skip = 8
    qv = ax2.quiver(X_sub[::skip, ::skip], Z_sub[::skip, ::skip],
                     u0[::skip, ::skip], v0[::skip, ::skip],
                     color='white', alpha=0.55, scale=5, width=0.003)
    ax2.fill_between(x_surf, eta0, 5, color='#1a4f7e', alpha=0.3)
    surf_top, = ax2.plot(x_surf, eta0, color='#5ab4d9', linewidth=1.2)
    ax2.fill_between(x_sub, -cfg.Lz, -cfg.Lz+2, color='#4a3520', alpha=0.8)
    ax2.set_xlim(-cfg.Lx/2, cfg.Lx/2)
    ax2.set_ylim(-cfg.Lz, 6)
    ax2.set_xlabel('x (m)', color='#8aafc8', fontsize=9)
    ax2.set_ylabel('depth z (m)', color='#8aafc8', fontsize=9)
    ax2.set_title('Subsurface velocity field — V = F′(z)', color='#e0eaf5', fontsize=11)

    fig.colorbar(im, ax=ax2, shrink=0.7, label='Speed |V| (m/s)')

    fig.suptitle('Ocean Wave Simulation  |  ∇²φ = 0  |  ω² = gk·tanh(kh)',
                 color='#e0eaf5', fontsize=12, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fill_state = {
        "top": axes[0].fill_between(x_surf, eta0, -8, color="#1a4f7e", alpha=0.4),
        "bot": axes[1].fill_between(x_surf, eta0, 5, color="#1a4f7e", alpha=0.3),
    }

    def update(frame):
        t = frame * cfg.dt
        eta = wf.surface_elevation(x_surf, t)
        u = wf.velocity_u(X_sub, Z_sub, t)
        v = wf.velocity_v(X_sub, Z_sub, t)
        speed = np.sqrt(u**2 + v**2)

        surf_line.set_ydata(eta)
        surf_top.set_ydata(eta)
        time_text.set_text(f"t = {t:.1f} s")
        im.set_array(speed.ravel())
        qv.set_UVC(u[::skip, ::skip], v[::skip, ::skip])

        fill_state["top"].remove()
        fill_state["bot"].remove()
        fill_state["top"] = axes[0].fill_between(x_surf, eta, -8, color="#1a4f7e", alpha=0.4)
        fill_state["bot"] = axes[1].fill_between(x_surf, eta, 5, color="#1a4f7e", alpha=0.3)

        return [surf_line, surf_top, time_text, im, qv]

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                    interval=1000 // cfg.fps, blit=False)

    out_path = save_path or './ocean_animation.gif'
    print(f"  Saving animation ({n_frames} frames)...")
    writer = animation.PillowWriter(fps=cfg.fps)
    anim.save(out_path, writer=writer, dpi=100)
    print(f"  Saved → {out_path}")
    plt.close(fig)
    return out_path

if __name__ == '__main__':
    import sys

    cfg = OceanConfig()
    cfg.Hs = 3.5
    cfg.Tp = 12.0
    cfg.h = 50.0
    cfg.n_waves = 60

    print("Running full ocean simulation...\n")
    fig = run_full_simulation(cfg, save_path='./ocean_simulation.png')
    print("\nRunning animation (this takes ~60 seconds)...")
    run_animation(cfg, save_path='./ocean_animation.gif', n_frames=60)
    print("\nDone.")
