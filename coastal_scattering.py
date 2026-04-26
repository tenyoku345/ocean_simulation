"""
coastal_scattering.py
=====================
Wave scattering and diffraction around coastal obstacles using
conformal mapping — the direct application of Sec 126.

The strategy (from the textbook):
  1. Choose a conformal map w = f(z) that transforms your obstacle
     (island, headland, harbour) into a simple domain in the w-plane.
  2. Write the complex potential in the w-plane (uniform flow + corrections).
  3. Pull back to the z-plane: F_z(z) = F_w(f(z)).
  4. Stagnation points of F_z give wave runup maxima on the coastline.

Obstacles modelled here:
  - Circular island         : w = z + R²/z  (Sec 126, Example 2)
  - Headland (wedge)        : w = z^{π/α}   (Sec 126, Example 1, generalised)
  - Harbour entrance (slit) : Schwarz-Christoffel (Sec 131 analogy)
  - Multiple islands        : method of images
"""

import numpy as np
from complex_potential import (
    uniform_flow, cylinder_flow, corner_flow,
    vortex, superpose, compute_fields
)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def island_wave_field(radius: float = 1.0, wave_speed: float = 1.0,
                       circulation: float = 0.0):
    """
    Waves scattering around a circular island of given radius.
    Directly uses Sec 126 Example 2: F(z) = A(z + R²/z).

    The streamlines of this flow ARE the wave crests in the far field
    and the diffraction pattern near the island.

    With circulation: models island with tidal vortex (e.g. wake eddies).
    Returns the complex potential function.
    """
    return cylinder_flow(A=wave_speed, radius=radius, circulation=circulation)


def compute_wave_runup(radius: float = 1.0, wave_speed: float = 1.0,
                        n_theta: int = 360):
    """
    Wave pressure (runup) distribution around the island circumference.
    From Bernoulli (Sec 124): P ∝ -½|V|²

    On the cylinder |z|=R, the speed is:
      |V| = 2A|sin θ|  (Sec 126, Exercise 4)

    So pressure is greatest at the stagnation points θ=0, π (front and back)
    and least at the flanks θ=±π/2.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta)
    speed_on_surface = 2 * wave_speed * np.abs(np.sin(theta))
    pressure = -0.5 * speed_on_surface**2
    pressure -= pressure.min()
    pressure /= pressure.max() + 1e-12
    return theta, pressure

def headland_flow(alpha_deg: float = 90.0, A: float = 1.0):
    """
    Flow around a headland with interior angle alpha (degrees).
    Uses the corner flow of Sec 126, Example 1: F(z) = A·z^n
    where n = π / alpha (in radians).

    alpha = 90°: right-angle headland, n=2 (classic corner flow)
    alpha = 45°: sharper headland, n=4
    alpha = 180°: flat coastline, n=1 (uniform flow)

    The stagnation point is at z=0 (the tip of the headland).
    This is where wave runup is maximum — waves pile up at the tip.
    """
    alpha_rad = np.radians(alpha_deg)
    n = np.pi / alpha_rad
    return corner_flow(A=A, n=n)

def harbour_flow(entrance_width: float = 0.5, basin_depth: float = 2.0,
                  wave_amplitude: float = 1.0):
    """
    Simplified harbour model using a source-at-entrance approach.

    The harbour is modelled as a rectangular basin open at one end.
    Waves entering through the narrow mouth create a potential:
      F(z) ≈ (Q/2π) · Log(z - z_entrance)  [source at entrance]
            + uniform flow representing the incoming wave

    This is the flow-through-a-slit analogy from Sec 131.
    The resonance frequency satisfies: L = (2n-1)λ/4, n=1,2,...
    (quarter-wavelength condition, analogous to an organ pipe).
    """
    source_strength = wave_amplitude * entrance_width * 2 * np.pi
    F_source = lambda z: (source_strength / (2 * np.pi)) * np.log(z + 1j * 0.01)
    F_uniform = uniform_flow(A=wave_amplitude * 0.3, angle=np.pi / 2)

    def F(z):
        return F_source(z) + F_uniform(z)
    return F


def resonance_frequencies(L: float, h: float, n_modes: int = 5) -> np.ndarray:
    """
    Resonance frequencies of a closed rectangular harbour of length L, depth h.
    Quarter-wave condition: L = (2n-1) * lambda/4
    => lambda_n = 4L/(2n-1)
    => k_n = 2pi/lambda_n = pi(2n-1)/(2L)
    => omega_n from dispersion relation.
    """
    from wave_theory import dispersion
    modes = np.arange(1, n_modes + 1)
    k_n = np.pi * (2 * modes - 1) / (2 * L)
    omega_n = np.array([dispersion(k, h) for k in k_n])
    T_n = 2 * np.pi / omega_n
    return T_n

def two_island_flow(d: float = 3.0, R: float = 0.8, A: float = 1.0):
    """
    Flow past two circular islands of radius R, centres at ±d on x-axis.
    Uses superposition (method of images):
      F = A·z + A·R²/z  (island 1 at origin)
          + image terms for island 2 (approximate for d >> R)

    For exact solution one would use elliptic functions; this
    first-order approximation is accurate when d >> R.
    """
    F1 = cylinder_flow(A=A, radius=R)
    doublet_strength = A * R**2
    def F2_approx(z):
        return doublet_strength / (z - d)

    def F(z):
        return (A * (z) + A * R**2 / (z + d)
                + A * R**2 / (z - d))
    return F

def plot_scattering_scenario(title, F, x_range, y_range,
                              obstacles=None, nx=500, ny=500,
                              n_streamlines=25, ax=None, cmap='RdBu_r'):
    """
    Plot the wave scattering field for a given complex potential.
    obstacles: list of (type, params) for masking and drawing.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.get_figure()

    x = np.linspace(*x_range, nx)
    y = np.linspace(*y_range, ny)
    X, Y = np.meshgrid(x, y)

    fields = compute_fields(F, X, Y)
    speed = fields['speed']
    psi = fields['psi']
    pressure = fields['pressure']
    mask = np.zeros_like(X, dtype=bool)
    if obstacles:
        for obs in obstacles:
            if obs['type'] == 'circle':
                cx, cy, r = obs['cx'], obs['cy'], obs['r']
                mask |= ((X - cx)**2 + (Y - cy)**2) < r**2
            elif obs['type'] == 'wedge':
                angle = obs.get('angle', np.pi / 2)
                theta = np.arctan2(Y, X)
                r_dist = np.sqrt(X**2 + Y**2)
                mask |= (r_dist < 0.3) | ((np.abs(theta) > angle / 2) & (r_dist < 0.5))

    pressure_masked = np.ma.masked_where(mask, pressure)
    speed_masked = np.ma.masked_where(mask, speed)
    im = ax.pcolormesh(X, Y, pressure_masked, cmap=cmap,
                        shading='auto', alpha=0.85,
                        vmin=np.percentile(pressure_masked.compressed(), 2),
                        vmax=np.percentile(pressure_masked.compressed(), 98))
    psi_masked = np.ma.masked_where(mask, psi)
    psi_vals = np.linspace(np.percentile(psi_masked.compressed(), 5),
                            np.percentile(psi_masked.compressed(), 95),
                            n_streamlines)
    ax.contour(X, Y, psi_masked, levels=psi_vals,
               colors='white', linewidths=0.7, alpha=0.6)
    if obstacles:
        for obs in obstacles:
            if obs['type'] == 'circle':
                circle = plt.Circle((obs['cx'], obs['cy']), obs['r'],
                                     color='#2c2c2c', zorder=10)
                ax.add_patch(circle)
                ax.plot(obs['cx'], obs['cy'], 'w+', markersize=8, zorder=11)
            elif obs['type'] == 'wedge':
                angle = obs.get('angle', np.pi / 2)
                half = angle / 2
                verts = np.array([
                    [0, 0],
                    [x_range[0], x_range[0] * np.tan(half)],
                    [x_range[0], y_range[0]],
                    [0, 0]
                ])
                from matplotlib.patches import Polygon
                poly = Polygon(verts, closed=True, color='#2c2c2c', zorder=10)
                ax.add_patch(poly)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Pressure (Bernoulli)')
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.annotate('', xy=(x_range[0] + 0.8, y_range[0] + 0.3),
                xytext=(x_range[0] + 0.1, y_range[0] + 0.3),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.text(x_range[0] + 0.5, y_range[0] + 0.55, 'wave',
            color='white', fontsize=9, ha='center')

    return fig, ax


if __name__ == '__main__':
    from wave_theory import LinearWave

    print("=== Coastal Scattering Demo ===\n")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Wave Scattering — Conformal Mapping (Sec 126)\n"
                 "White lines = streamlines (wave paths), colour = pressure",
                 fontsize=14, y=0.98)
    F1 = island_wave_field(radius=1.0, wave_speed=1.0)
    plot_scattering_scenario(
        "Circular island (Sec 126, Example 2)\nF(z) = A(z + R²/z)",
        F1, (-4, 4), (-4, 4),
        obstacles=[{'type': 'circle', 'cx': 0, 'cy': 0, 'r': 1.0}],
        ax=axes[0, 0], cmap='RdBu_r')
    F2 = island_wave_field(radius=1.0, wave_speed=1.0, circulation=4.0)
    plot_scattering_scenario(
        "Island with tidal vortex (Γ=4)\nMagnus effect: asymmetric wave field",
        F2, (-4, 4), (-4, 4),
        obstacles=[{'type': 'circle', 'cx': 0, 'cy': 0, 'r': 1.0}],
        ax=axes[0, 1], cmap='PuOr_r')
    F3 = headland_flow(alpha_deg=90.0, A=1.0)
    plot_scattering_scenario(
        "Headland — 90° wedge (Sec 126, Example 1)\nF(z) = A·z², stagnation at tip",
        F3, (0.01, 4), (0.01, 4),
        obstacles=None,
        ax=axes[1, 0], cmap='RdBu_r')
    axes[1, 0].axhline(y=0, color='#2c2c2c', lw=3, xmin=0)
    axes[1, 0].axvline(x=0, color='#2c2c2c', lw=3, ymin=0)
    axes[1, 0].text(0.15, 0.15, 'Stagnation\npoint', fontsize=8,
                    color='white', ha='center')
    F4 = two_island_flow(d=2.5, R=0.7, A=1.0)
    plot_scattering_scenario(
        "Two islands — method of images\nWave channelling between obstacles",
        F4, (-5, 5), (-4, 4),
        obstacles=[
            {'type': 'circle', 'cx': -2.5, 'cy': 0, 'r': 0.7},
            {'type': 'circle', 'cx':  2.5, 'cy': 0, 'r': 0.7},
        ],
        ax=axes[1, 1], cmap='RdBu_r')
    theta, runup = compute_wave_runup()
    ax_inset = axes[0, 0].inset_axes([0.72, 0.72, 0.26, 0.26],
                                       transform=axes[0, 0].transAxes)
    ax_inset.fill_between(np.degrees(theta), runup, alpha=0.7, color='#e06c1f')
    ax_inset.set_title('Runup', fontsize=7)
    ax_inset.set_xticks([0, 180, 360])
    ax_inset.set_xticklabels(['0°', '180°', '360°'], fontsize=6)
    ax_inset.tick_params(labelsize=6)

    plt.tight_layout()
    plt.savefig('/tmp/coastal_scattering.png', dpi=120, bbox_inches='tight')
    print("Saved coastal_scattering.png")
    print("\nHarbour resonance periods (L=500m, h=10m):")
    T_res = resonance_frequencies(L=500, h=10, n_modes=4)
    for i, T in enumerate(T_res):
        print(f"  Mode {i+1}: T = {T:.1f} s  (f = {1/T:.4f} Hz)")
