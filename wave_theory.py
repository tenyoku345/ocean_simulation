"""
wave_theory.py
==============
Linear water wave theory built on the complex potential framework.

Mathematical foundation: Phase 3 of the roadmap.

Starting from ∇²φ = 0 (Sec 124) with:
  - Free surface BC: φ_tt + g*φ_y = 0  at y = 0
  - Bottom BC:       φ_y = 0           at y = -h

We find the dispersion relation:
  ω² = g·k·tanh(k·h)

And the complex velocity potential for a progressive wave:
  F(z, t) = (iAg/ω) · [cosh k(z+h) / cosh(kh)] · e^{i(kx - ωt)}

This module provides:
  1. Dispersion relation solver
  2. Single wave potential and velocity fields
  3. Wave packet (superposition of many waves)
  4. Ocean wave spectrum (JONSWAP)
  5. Stokes wave (weakly nonlinear correction)
"""

import numpy as np
from scipy.optimize import brentq
from dataclasses import dataclass, field
from typing import Optional
G = 9.81 
def dispersion(k: float, h: float = np.inf) -> float:
    """Angular frequency ω for wavenumber k in water depth h."""
    if np.isinf(h):
        return np.sqrt(G * k)          # deep water: ω = sqrt(g*k)
    return np.sqrt(G * k * np.tanh(k * h))


def wavenumber_from_frequency(omega: float, h: float = np.inf,
                               tol: float = 1e-10) -> float:
    """
    Invert the dispersion relation to find k given ω.
    Uses the deep-water approximation as initial guess, then Newton-Raphson.
    """
    if np.isinf(h):
        return omega**2 / G
    k0 = omega**2 / G

    def residual(k):
        return G * k * np.tanh(k * h) - omega**2
    k_lo, k_hi = 1e-6, max(k0 * 10, 1.0)
    while residual(k_hi) < 0:
        k_hi *= 2
    return brentq(residual, k_lo, k_hi, xtol=tol)


def phase_velocity(k: float, h: float = np.inf) -> float:
    """Phase speed c = ω/k."""
    return dispersion(k, h) / k


def group_velocity(k: float, h: float = np.inf) -> float:
    """
    Group velocity cg = dω/dk — the speed at which wave energy travels.
    In deep water: cg = c/2.
    In shallow water (kh << 1): cg = c = sqrt(g*h)  (non-dispersive, like tsunamis).
    """
    if np.isinf(h):
        return 0.5 * np.sqrt(G / k)
    kh = k * h
    omega = dispersion(k, h)
    n = 0.5 * (1 + 2 * kh / np.sinh(2 * kh))
    return n * omega / k

@dataclass
class LinearWave:
    """
    A single linear water wave.

    Parameters
    ----------
    amplitude : wave amplitude A (metres)
    k         : wavenumber (rad/m)
    h         : water depth (m), np.inf for deep water
    direction : propagation direction (radians, 0 = positive x)
    phase     : initial phase offset (radians)
    """
    amplitude: float = 1.0
    k: float = 1.0
    h: float = np.inf
    direction: float = 0.0
    phase: float = 0.0

    def __post_init__(self):
        self.omega = dispersion(self.k, self.h)
        self.period = 2 * np.pi / self.omega
        self.wavelength = 2 * np.pi / self.k
        self.c = phase_velocity(self.k, self.h)
        self.cg = group_velocity(self.k, self.h)
        self.kx = self.k * np.cos(self.direction)
        self.ky = self.k * np.sin(self.direction)

    def surface_elevation(self, x: np.ndarray, t: float) -> np.ndarray:
        """η(x, t) = A · cos(kx - ωt + phase)"""
        return self.amplitude * np.cos(
            self.kx * x - self.omega * t + self.phase)

    def velocity_potential(self, x: np.ndarray, y: np.ndarray,
                           t: float) -> np.ndarray:
        """
        φ(x, y, t) = (Ag/ω) · [cosh k(y+h) / cosh(kh)] · sin(kx - ωt)

        This is the real part of the complex potential (Phase 3 derivation).
        y=0 is the mean surface, y=-h is the seabed.
        Clamp y to [-h, 0] so we don't evaluate above the surface.
        """
        y_clamped = np.clip(y, -self.h, 0)
        if np.isinf(self.h):
            depth_factor = np.exp(self.k * y_clamped)  # deep water: e^{ky}
        else:
            depth_factor = np.cosh(self.k * (y_clamped + self.h)) / np.cosh(self.k * self.h)

        phase_arg = self.kx * x - self.omega * t + self.phase
        return (self.amplitude * G / self.omega) * depth_factor * np.sin(phase_arg)

    def velocity_u(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """Horizontal velocity u = ∂φ/∂x = A·ω·[cosh k(y+h)/sinh(kh)]·cos(kx-ωt)"""
        y_clamped = np.clip(y, -self.h, 0)
        if np.isinf(self.h):
            depth_factor = np.exp(self.k * y_clamped)
        else:
            depth_factor = np.cosh(self.k * (y_clamped + self.h)) / np.sinh(self.k * self.h)
            depth_factor = np.where(np.isfinite(depth_factor), depth_factor, 0)

        phase_arg = self.kx * x - self.omega * t + self.phase
        return self.amplitude * self.omega * depth_factor * np.cos(phase_arg) * np.cos(self.direction)

    def velocity_v(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """Vertical velocity v = ∂φ/∂y"""
        y_clamped = np.clip(y, -self.h, 0)
        if np.isinf(self.h):
            depth_factor = np.exp(self.k * y_clamped)
        else:
            depth_factor = np.sinh(self.k * (y_clamped + self.h)) / np.sinh(self.k * self.h)
            depth_factor = np.where(np.isfinite(depth_factor), depth_factor, 0)

        phase_arg = self.kx * x - self.omega * t + self.phase
        return self.amplitude * self.omega * depth_factor * np.sin(phase_arg)

    def particle_orbit(self, x0: float, y0: float,
                       t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Lagrangian particle orbit (elliptical in finite depth, circular in deep water).
        x(t) = x0 - A · [cosh k(y0+h)/sinh(kh)] · sin(kx0 - ωt)
        y(t) = y0 + A · [sinh k(y0+h)/sinh(kh)] · cos(kx0 - ωt)
        """
        y0c = np.clip(y0, -self.h, 0)
        if np.isinf(self.h):
            ax = self.amplitude * np.exp(self.k * y0c)
            ay = ax
        else:
            ax = self.amplitude * np.cosh(self.k * (y0c + self.h)) / np.sinh(self.k * self.h)
            ay = self.amplitude * np.sinh(self.k * (y0c + self.h)) / np.sinh(self.k * self.h)
            ax = np.where(np.isfinite(ax), ax, 0)
            ay = np.where(np.isfinite(ay), ay, 0)

        phase_arg = self.kx * x0 - self.omega * t + self.phase
        x_orbit = x0 - ax * np.sin(phase_arg)
        y_orbit = y0 + ay * np.cos(phase_arg)
        return x_orbit, y_orbit

    def summary(self) -> str:
        return (
            f"Linear wave: A={self.amplitude:.2f} m, λ={self.wavelength:.2f} m, "
            f"T={self.period:.2f} s, c={self.c:.2f} m/s, cg={self.cg:.2f} m/s, "
            f"h={'∞' if np.isinf(self.h) else f'{self.h:.1f} m'}"
        )

class WaveField:
    """
    Superposition of N linear waves.
    Because Laplace's equation is linear, the total velocity potential is:
      φ_total = Σ φ_n
    This is the mathematical basis of ocean wave modelling.
    """

    def __init__(self, waves: list[LinearWave]):
        self.waves = waves

    def surface_elevation(self, x: np.ndarray, t: float) -> np.ndarray:
        return sum(w.surface_elevation(x, t) for w in self.waves)

    def velocity_u(self, x, y, t):
        return sum(w.velocity_u(x, y, t) for w in self.waves)

    def velocity_v(self, x, y, t):
        return sum(w.velocity_v(x, y, t) for w in self.waves)

    def significant_wave_height(self) -> float:
        """Hs = 4 * sqrt(variance) — standard oceanographic measure."""
        variance = sum(0.5 * w.amplitude**2 for w in self.waves)
        return 4 * np.sqrt(variance)

def jonswap_spectrum(f: np.ndarray, Hs: float = 3.0,
                     Tp: float = 10.0, gamma: float = 3.3) -> np.ndarray:
    """
    JONSWAP (Joint North Sea Wave Project) spectrum S(f).

    The standard model for wind-generated ocean waves.
    Hs  : significant wave height (m)
    Tp  : peak period (s)
    gamma: peak enhancement factor (1 = Pierson-Moskowitz, 3.3 = typical storm)

    Returns spectral density S(f) in m²/Hz.
    The significant wave height satisfies: Hs = 4*sqrt(integral S df).
    """
    f = np.asarray(f, dtype=float)
    fp = 1.0 / Tp
    alpha = 0.0624 / (0.230 + 0.0336 * gamma - 0.185 / (1.9 + gamma))
    alpha_hs = (Hs / 4)**2 / (0.2257 * Tp**4)

    sigma = np.where(f <= fp, 0.07, 0.09)
    r = np.exp(-((f - fp)**2) / (2 * sigma**2 * fp**2))

    with np.errstate(divide='ignore', invalid='ignore'):
        S = (alpha_hs * G**2 * (2 * np.pi)**(-4) * f**(-5)
             * np.exp(-1.25 * (f / fp)**(-4))
             * gamma**r)
    S = np.where(np.isfinite(S), S, 0)
    S = np.where(f > 0, S, 0)
    return S


def waves_from_jonswap(Hs: float = 3.0, Tp: float = 10.0,
                        gamma: float = 3.3, h: float = np.inf,
                        n_waves: int = 50, f_min: float = 0.04,
                        f_max: float = 0.5,
                        spreading: float = 0.3,
                        seed: int = 42) -> WaveField:
    """
    Generate a random sea state from the JONSWAP spectrum.
    Each frequency component becomes a LinearWave with:
      - amplitude from the spectrum
      - random phase (stationary random process)
      - direction spread around the mean direction
    """
    rng = np.random.default_rng(seed)
    freqs = np.linspace(f_min, f_max, n_waves)
    df = freqs[1] - freqs[0]
    S = jonswap_spectrum(freqs, Hs=Hs, Tp=Tp, gamma=gamma)
    amplitudes = np.sqrt(2 * S * df)
    phases = rng.uniform(0, 2 * np.pi, n_waves)
    directions = rng.normal(0, spreading, n_waves)

    waves = []
    for A, f, phi, d in zip(amplitudes, freqs, phases, directions):
        if A > 0:
            omega = 2 * np.pi * f
            k = wavenumber_from_frequency(omega, h)
            waves.append(LinearWave(
                amplitude=A, k=k, h=h,
                direction=d, phase=phi
            ))
    return WaveField(waves)

def stokes_elevation(x: np.ndarray, t: float, wave: LinearWave,
                      stokes_order: int = 3) -> np.ndarray:
    """
    Stokes wave expansion to given order.
    The Stokes parameter epsilon = k*A (wave steepness) measures nonlinearity.

    eta = A*cos(theta)
        + (k*A^2/2)*cos(2*theta)            [2nd order: wave crest sharpens]
        + (3/8)*(k*A)^2*A*cos(3*theta)      [3rd order: Stokes drift]

    where theta = kx - omega*t + phase.
    For steepness k*A > 0.44, waves begin to break (Miche criterion).
    """
    k, A = wave.k, wave.amplitude
    eps = k * A
    theta = wave.kx * x - wave.omega * t + wave.phase

    eta = A * np.cos(theta)
    if stokes_order >= 2:
        eta += (k * A**2 / 2) * np.cos(2 * theta)
    if stokes_order >= 3:
        eta += (3/8) * eps**2 * A * np.cos(3 * theta)
    return eta

def dispersion_curves(h_values: list[float],
                       k_range: tuple = (0.01, 5.0), n: int = 300):
    """
    Compute dispersion curves ω(k) for multiple depths.
    Returns k array and dict {h: omega_array}.
    """
    k = np.linspace(*k_range, n)
    curves = {}
    for h in h_values:
        curves[h] = np.array([dispersion(ki, h) for ki in k])
    return k, curves


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("=== Water Wave Theory Demo ===\n")
    wave = LinearWave(amplitude=1.5, k=0.5, h=30.0)
    print(wave.summary())

    k_arr, curves = dispersion_curves([5, 20, 100, np.inf])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Linear Water Wave Theory", fontsize=14)

    ax = axes[0]
    labels = {5: '5 m', 20: '20 m', 100: '100 m', np.inf: 'deep (∞)'}
    for h, omega in curves.items():
        ax.plot(k_arr, omega, label=f'h = {labels[h]}', linewidth=2)
    ax.set_xlabel('Wavenumber k (rad/m)')
    ax.set_ylabel('ω (rad/s)')
    ax.set_title('Dispersion relation ω²=gk·tanh(kh)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax = axes[1]
    f = np.linspace(0.01, 0.5, 500)
    for gamma, ls in [(1.0, '--'), (3.3, '-'), (7.0, ':')]:
        S = jonswap_spectrum(f, Hs=3.0, Tp=10.0, gamma=gamma)
        ax.plot(f, S, ls, label=f'γ={gamma}', linewidth=2)
    ax.set_xlabel('Frequency f (Hz)')
    ax.set_ylabel('S(f) (m²/Hz)')
    ax.set_title('JONSWAP wave spectrum')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    t_arr = np.linspace(0, wave.period, 200)
    depths = [0, -5, -10, -20]
    colors = ['#1a6faf', '#2d9cdb', '#56ccf2', '#bde4f5']
    for d, col in zip(depths, colors):
        xp, yp = wave.particle_orbit(0, d, t_arr)
        ax.plot(xp, yp + d, color=col, linewidth=2, label=f'y={d} m')
    ax.set_xlabel('x displacement (m)')
    ax.set_ylabel('Depth y (m)')
    ax.set_title('Particle orbits vs depth (ellipses)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('/tmp/wave_theory.png', dpi=120, bbox_inches='tight')
    print("Saved wave_theory.png")
