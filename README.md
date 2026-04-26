## Requirements

- Python 3.10 or higher
- NumPy
- Matplotlib
- SciPy

Install dependencies:

```bash
pip install numpy matplotlib scipy
```
Using conda:

```bash
conda install numpy matplotlib scipy
```

---

## Running the simulation

### 1. Verify Python version

```bash
python3 --version
# Expected: Python 3.10.x or higher
```

### 2. Navigate to the project folder

```bash
cd path/to/ocean_sim
```

### 3. Run the main script

```bash
python3 ocean_simulation.py
```

You will see progress printed to the terminal:

```
=== Ocean Simulation ===

Phase 1: Building wave field from JONSWAP spectrum...
  Building sea state: Hs=3.5m, Tp=12.0s, h=50.0m, n_waves=60
  Significant wave height (computed): 1.01 m
Phase 2: Computing subsurface velocity field...
Phase 3: Computing island wave scattering...
Phase 4: Particle tracking (Stokes drift)...
Phase 5: Rendering figure...

Saved → ocean_simulation.png

Running animation (60 frames)...
  Saving animation (60 frames)...
  Saved → ocean_animation.gif
```

**Expected runtime:** the static figure takes 15–30 seconds. The animation takes 1–3 minutes depending on your machine.

> **Windows users:** if `python3` is not recognised, use `python` instead. Output files for individual modules are saved to `C:\Users\YourName\AppData\Local\Temp\`. Output from `ocean_simulation.py` is saved in the script's own directory.

---

## Output files

| File | Description |
|---|---|
| `ocean_simulation.png` | 5-panel static figure: ocean surface η(x,t), subsurface velocity field V = F′(z), island wave scattering (Sec. 126), Lagrangian particle tracking, and JONSWAP spectrum |
| `ocean_animation.gif` | Animated ocean surface and subsurface velocity field evolving in real time at 25 fps |

---

## Running individual modules

Each module can be run independently to inspect its specific output:

```bash
python3 complex_potential.py
python3 wave_theory.py
python3 coastal_scattering.py
```

Each saves a figure to the system temp directory and prints a short summary to the terminal.

---

## Configuration

Open `ocean_simulation.py` and edit the block at the bottom of the file (around line 540):

```python
if __name__ == '__main__':
    cfg = OceanConfig()

    cfg.Hs      = 3.5    # significant wave height (metres)
    cfg.Tp      = 12.0   # peak wave period (seconds)
    cfg.h       = 50.0   # water depth (metres)
    cfg.gamma   = 3.3    # JONSWAP peak enhancement factor (1.0–7.0)
    cfg.n_waves = 60     # number of spectral components
```

### Parameter guide

| Parameter | Default | Try | Effect |
|---|---|---|---|
| `cfg.Hs` | 3.5 m | 6.0 | Larger storm — taller waves |
| `cfg.Tp` | 12.0 s | 6.0 | Short-period wind chop instead of ocean swell |
| `cfg.h` | 50.0 m | 5.0 | Shallow water — watch the dispersion relation change |
| `cfg.gamma` | 3.3 | 1.0 | Pierson–Moskowitz open-ocean spectrum |
| `cfg.n_waves` | 60 | 200 | More realistic sea state; slower runtime |

### Full `OceanConfig` reference

```python
class OceanConfig:
    # Domain
    Lx: float = 200.0       # domain width (m)
    Lz: float = 60.0        # domain depth (m)
    Nx: int   = 300         # horizontal resolution
    Nz: int   = 100         # vertical resolution

    # Wave conditions
    Hs:      float = 3.5    # significant wave height (m)
    Tp:      float = 12.0   # peak period (s)
    gamma:   float = 3.3    # JONSWAP peak enhancement
    h:       float = 50.0   # water depth (m)
    n_waves: int   = 60     # spectral components

    # Animation
    dt:       float = 0.4   # time step per frame (s)
    fps:      int   = 25    # frames per second
    n_frames: int   = 200   # total animation frames

    # Particles
    n_particles: int = 30
```
Feel free to change configurations listed here to see different effects of the simulation.
---

