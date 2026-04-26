"""
complex_potential.py
====================
Core engine for 2D irrotational, incompressible fluid flow.

Mathematical foundation: Brown & Churchill Secs 124-125.
Every flow is encoded as an analytic function F(z) = phi + i*psi.
  - phi(x,y): velocity potential  (equipotentials phi=c are perpendicular to flow)
  - psi(x,y): stream function     (streamlines psi=c are the flow paths)
  - V(z) = F'(z)_conjugate        velocity field (Sec 125, eq 3)

The key insight: Laplace's equation is automatically satisfied by the real
and imaginary parts of ANY analytic function. So we build complex potentials
by composing elementary analytic functions, and fluid dynamics falls out for free.
"""

import numpy as np
from typing import Callable
ComplexPotential = Callable[[np.ndarray], np.ndarray]

def uniform_flow(A: float = 1.0, angle: float = 0.0) -> ComplexPotential:
    """
    Uniform flow of speed A at angle (radians) to the x-axis.
    F(z) = A * e^{-i*angle} * z
    V = A * e^{i*angle}  — constant everywhere.
    """
    direction = A * np.exp(-1j * angle)
    def F(z):
        return direction * z
    F.__doc__ = f"Uniform flow: speed={A:.3f}, angle={np.degrees(angle):.1f}°"
    return F


def source_sink(strength: float, z0: complex = 0.0) -> ComplexPotential:
    """
    Source (strength > 0) or sink (strength < 0) at z0.
    F(z) = (strength / 2pi) * Log(z - z0)
    Streamlines: radial lines from z0. Equipotentials: circles around z0.
    """
    def F(z):
        return (strength / (2 * np.pi)) * np.log(z - z0)
    return F


def vortex(circulation: float, z0: complex = 0.0) -> ComplexPotential:
    """
    Point vortex with given circulation Gamma at z0.
    F(z) = -i * (Gamma / 2pi) * Log(z - z0)
    A vortex is a source rotated by 90°: streamlines become circles.
    """
    def F(z):
        return -1j * (circulation / (2 * np.pi)) * np.log(z - z0)
    return F


def doublet(strength: float, z0: complex = 0.0) -> ComplexPotential:
    """
    Doublet (source-sink pair in the limit of zero separation).
    F(z) = strength / (2pi * (z - z0))
    Used internally to construct the cylinder flow.
    """
    def F(z):
        return strength / (2 * np.pi * (z - z0))
    return F


def corner_flow(A: float = 1.0, n: float = 2.0) -> ComplexPotential:
    """
    Flow in a wedge of half-angle pi/(2n).
    F(z) = A * z^n
    n=2: flow around a 90-degree corner (Sec 126, Example 1)
    n=1: uniform flow
    n=0.5: flow around a flat plate edge
    Stream function: psi = A * r^n * sin(n*theta)
    """
    def F(z):
        return A * z**n
    return F


def cylinder_flow(A: float = 1.0, radius: float = 1.0,
                  circulation: float = 0.0) -> ComplexPotential:
    """
    Flow past a cylinder of given radius (Sec 126, Example 2).
    F(z) = A*(z + radius^2/z) - i*(Gamma/2pi)*Log(z)

    The Joukowski-like map w = z + R^2/z sends the circle |z|=R
    to the segment [-2R, 2R] on the real axis, mapping the exterior
    of the cylinder to the upper half-plane.

    With circulation != 0: Magnus effect — the cylinder experiences lift.
    Kutta-Joukowski theorem: L = rho * A * Gamma (per unit span).

    Stagnation points (V=0): z = (iGamma/4piA) +/- sqrt(R^2 - (Gamma/4piA)^2)
    """
    R2 = radius**2
    def F(z):
        return A * (z + R2 / z) - 1j * (circulation / (2 * np.pi)) * np.log(z)
    return F


def superpose(*potentials: ComplexPotential) -> ComplexPotential:
    """
    Superpose multiple complex potentials.
    Since Laplace's equation is linear, any sum of solutions is a solution.
    This is the mathematical basis of the wave superposition in Phase 3.
    """
    def F(z):
        return sum(f(z) for f in potentials)
    return F

def compute_fields(F: ComplexPotential, x: np.ndarray, y: np.ndarray,
                   eps: float = 1e-8):
    """
    Given a complex potential F and grid arrays x, y, compute:
      phi   : velocity potential
      psi   : stream function
      u, v  : x and y velocity components
      speed : |V|
      pressure : relative pressure from Bernoulli (Sec 124)
                 P/rho = C - 0.5*|V|^2

    Velocity is computed via complex differentiation: V = conj(F'(z))
    We use a small finite difference for F' to handle any F.
    """
    Z = x + 1j * y
    FZ = F(Z)
    phi = np.real(FZ)
    psi = np.imag(FZ)
    dz = eps * (1 + 1j)
    Fprime = (F(Z + dz) - F(Z - dz)) / (2 * dz)
    V = np.conj(Fprime)
    u = np.real(V)
    v = np.imag(V)
    speed = np.abs(V)
    pressure = -0.5 * speed**2
    pressure -= pressure.mean()

    return {
        'phi': phi,
        'psi': psi,
        'u': u,
        'v': v,
        'speed': speed,
        'pressure': pressure,
    }

def joukowski_map(z: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Joukowski transformation: w = z + c^2/z
    Maps a circle of radius R > c to an airfoil-like curve.
    At R = c: maps the circle to the flat plate [-2c, 2c].
    The cylinder flow F(w) pulls back to the z-plane as F(z + c^2/z).
    """
    return z + c**2 / z


def apply_conformal_map(F_w: ComplexPotential,
                        map_fn: Callable) -> ComplexPotential:
    """
    If F_w is the complex potential in the w-plane and w = map_fn(z),
    then F_z(z) = F_w(map_fn(z)) is the potential in the z-plane.
    This is the core of Sec 126: solve in a simple domain, pull back.
    """
    def F_z(z):
        return F_w(map_fn(z))
    return F_z

def find_stagnation_points(F: ComplexPotential, x_range, y_range,
                           nx=400, ny=400, threshold=0.05):
    """
    Find points where |V| < threshold (approximate stagnation points).
    These are the points of maximum pressure (Bernoulli: Sec 124).
    """
    x = np.linspace(*x_range, nx)
    y = np.linspace(*y_range, ny)
    X, Y = np.meshgrid(x, y)
    fields = compute_fields(F, X, Y)
    speed = fields['speed']
    mask = speed < threshold * speed.max()
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    pts = [(x[xi], y[yi]) for xi, yi in zip(xs[::10], ys[::10])]
    return pts


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.linspace(-3, 3, 400)
    y = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, y)
    F = cylinder_flow(A=1.0, radius=1.0, circulation=2.0)
    fields = compute_fields(F, X, Y)
    R = np.sqrt(X**2 + Y**2)
    mask = R < 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Cylinder flow with circulation (Magnus effect)\n"
                 "F(z) = A(z + 1/z) - i(Γ/2π)Log(z)", fontsize=13)

    for ax, key, title, cmap in zip(
        axes,
        ['psi', 'speed', 'pressure'],
        ['Stream function ψ (flow paths)', 'Speed |V|', 'Pressure (Bernoulli)'],
        ['RdBu', 'viridis', 'RdBu_r']
    ):
        data = np.ma.masked_where(mask, fields[key])
        im = ax.pcolormesh(X, Y, data, cmap=cmap, shading='auto')
        if key == 'psi':
            ax.contour(X, Y, np.ma.masked_where(mask, fields['psi']),
                       levels=30, colors='k', linewidths=0.5, alpha=0.6)
        circle = plt.Circle((0, 0), 1.0, color='gray', zorder=5)
        ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig('/tmp/cylinder_test.png', dpi=120, bbox_inches='tight')
    print("Test passed — saved to /tmp/cylinder_test.png")
