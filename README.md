# torchspin

**Simulate and fit EPR/ESR spectra in Python — differentiable, GPU-capable, and validated against [EasySpin](https://easyspin.org).**

[![PyPI](https://img.shields.io/pypi/v/torchspin)](https://pypi.org/project/torchspin/)
[![Python](https://img.shields.io/pypi/pyversions/torchspin)](https://pypi.org/project/torchspin/)
[![Tests](https://github.com/follmerlab/torchspin/actions/workflows/python-tests.yml/badge.svg)](https://github.com/follmerlab/torchspin/actions/workflows/python-tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

torchspin reimplements the core physics engine of EasySpin (MATLAB) in
Python/PyTorch. You get the simulators you already know — `pepper`, `garlic`,
`chili`, `salt`, `saffron`, `curry`, `cardamom`, `esfit` — with the same units
and conventions, plus three things MATLAB cannot give you:

- **Gradients.** Spectra are differentiable end to end, so you can fit with
  gradient descent, propagate uncertainty, or put a simulator inside a neural
  network.
- **A Python workflow.** NumPy and torch tensors in and out, no license server,
  installable with `pip`.
- **Optional GPU.** `Options(device='cuda')` for batched and gradient-based
  work.

torchspin is a reimplementation, not a drop-in replacement for every EasySpin
feature. Before relying on it for published results, read
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) — it states per simulator what is
validated against MATLAB and what is not.

---

## Installation

```bash
pip install torchspin                 # core
pip install "torchspin[plot]"         # + matplotlib for the plotting helpers
pip install "torchspin[gui]"          # + the interactive fitting panel (torchspin.fitgui)
```

CPU-only PyTorch, to avoid the large CUDA wheels:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torchspin
```

Requires Python 3.10+, PyTorch 2.0+, NumPy 1.23+, SciPy 1.10+.

---

## Quick start

### CW powder spectrum (`pepper`)

```python
from torchspin import SpinSystem, Experiment, Options, pepper

sys = SpinSystem(
    S=[0.5],
    g=[[2.009, 2.006, 2.002]],
    Nucs=['14N'],
    A=[[10.0, 10.0, 95.0]],   # MHz
    lw=[1.0, 0.0],            # Gaussian FWHM (mT)
)
exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=1024, Harmonic=1)
B, spec = pepper(sys, exp, Options(GridSize=50))
```

### Solution EPR (`garlic`)

```python
from torchspin import SpinSystem, Experiment, garlic

sys = SpinSystem(S=[0.5], g=2.003, Nucs='1H', A=10.0, lw=[0.2, 0.0])
exp = Experiment(mwFreq=9.5, Range=[336, 342], nPoints=512, Harmonic=1)
B, spec = garlic(sys, exp)
```

### Fitting (`esfit`)

```python
import numpy as np
from torchspin import esfit, FitOptions

def model(p):
    sys_mod = SpinSystem(S=[0.5], g=[[p[0], p[1], p[2]]], lw=[1.0, 0.0])
    _, spc = pepper(sys_mod, exp, Options(GridSize=31, Verbosity=0))
    return spc.numpy()

result = esfit(
    data_measured, model,
    p0=np.array([2.00, 2.10, 2.20]),
    lb=np.array([1.95, 2.05, 2.15]),
    ub=np.array([2.05, 2.15, 2.25]),
    options=FitOptions(method='global', progress='text'),
)
print('Best-fit g:', result.pfit)
```

`progress='text'` prints RMSD, evaluation count and ETA as it runs, and
interrupting the kernel returns the best fit so far.

Population methods (swarm, genetic, grid, montecarlo, global) can evaluate each
generation in parallel worker processes with `FitOptions(n_workers='auto')`. This
is off by default. Each worker is a separate Python process with its own torch
(about 0.3–0.5 GB), so `'auto'` uses the cores you were allocated, at most 32, and
no more workers than fit in half the free memory. A script that uses it needs the
usual `if __name__ == '__main__':` guard around the code that runs the fit.

### Interactive fitting panel (`torchspin.fitgui`)

In JupyterLab (`pip install "torchspin[gui]"`), the same kind of fit runs in a panel
with parameter sliders, fix boxes, method controls, Start/Stop and a live plot of the
fit, residual and RMSD:

```python
%matplotlib widget
import numpy as np
from torchspin import SpinSystem, Experiment, Options, pepper
from torchspin.esfit import FitOptions
from torchspin.fitgui import FitSession, FitPanel

exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=1024, Harmonic=1)

def model(p):  # p = [gx, gy, gz, Az (MHz), lwpp (mT)]
    sys = SpinSystem(S=[0.5], g=p[:3], Nucs='14N', A=[[18, 18, p[3]]], lwpp=[p[4], 0])
    return pepper(sys, exp, Options(GridSize=[31, 4], Verbosity=0))[1].numpy()

B = np.linspace(330, 350, 1024)
data = model([2.0085, 2.0060, 2.0025, 92.0, 0.35])          # or your measured spectrum
data = data + 0.02 * np.abs(data).max() * np.random.default_rng(1).standard_normal(data.size)

session = FitSession(data, model, p0=[2.011, 2.0085, 2.005, 101.0, 0.5],
                     vary=[0.004, 0.004, 0.004, 15.0, 0.25],
                     names=['gx', 'gy', 'gz', 'Az', 'lwpp'],
                     options=FitOptions(method='global', max_iter=100))
FitPanel(session, x=B, xlabel='B (mT)').show()
```

Press **Start** to fit; **Stop** returns the best fit so far, and `session.result`
holds the result afterwards. The panel's `n_workers` box accepts `auto` for parallel
population fits. The full walkthrough is
[`examples/notebooks/09_fit_gui.ipynb`](examples/notebooks/09_fit_gui.ipynb).

### Differentiable spectrum

```python
import torch
from torchspin import differentiable_spectrum

g = torch.tensor([2.0, 2.1, 2.2], dtype=torch.float64, requires_grad=True)
B, spec = differentiable_spectrum(g, mwFreq_GHz=9.5, B_range=(300, 380),
                                  nPoints=512, lw_mT=1.0)
loss = ((spec - target_spec) ** 2).sum()
loss.backward()
print('dL/dg:', g.grad)
```

---

## Documentation

Documentation is in the `docs/` folder. Some is as standalone markdown files,
but the functions within torchspin are documented by ``sphinx``. To build this
documentation, install torchspin with the `doc` group of optional dependencies,
and run ``make clean html`` in the `docs/` directory. The docs can then be
accessed by opening ``docs/build/html/index.html`` in a browser.

---

## Units and conventions

These match EasySpin, so parameters transfer directly:

| | |
|---|---|
| Energy | MHz |
| Magnetic field | mT |
| Euler angles | radians, z-y'-z'' **passive** rotation |
| Basis ordering | m = S, S−1, …, −S (descending) |
| Default dtype | `torch.complex128` |

---

## What's included

| Category | Modules |
|----------|---------|
| CW EPR | `pepper` (powder; `matrix`, `perturb` and `hybrid` resonance solvers, `grid_convergence` to check `GridSize`), `garlic` (solution), `chili` (slow-motion), `salt` (ENDOR) |
| Pulse EPR | `saffron` (ESEEM/HYSCORE), `spidyan` (arbitrary sequences), `saffron_thyme` (real pulses) |
| Trajectory | `cardamom` (MD/diffusion/jump), `mdload`, `mdhmm` |
| Magnetometry | `curry` (susceptibility, magnetization) |
| Fitting | `esfit` — 10 methods (simplex, L-BFGS-B, Powell, Levenberg–Marquardt, trust-region reflective, grid, Monte Carlo, genetic, swarm, and `global` = swarm + simplex polish); `torchspin.fitgui` interactive panel |
| Autograd | `differentiable_spectrum`; `pepper`, `garlic`, `salt`, `saffron` and `curry` are differentiable |
| Machine learning | `torchspin.ml` — parameter packing, batched simulation, datasets, differentiable losses |
| Data I/O | `eprload` (14 vendor formats), `eprsave` (BES3T), `orca2torchspin` |
| Batch / GPU | `batch_pepper`, `batch_simulate`; `Options(device='cuda')` |

---

## Tutorials

Start with `01_torchspin_quickstart.ipynb`, then branch to autograd or fitting
depending on what you need. Full index: [docs/index.md](docs/index.md).

| Notebook | What it covers |
|---|---|
| `01_torchspin_quickstart` | Core API: spin systems, Hamiltonians, powder and solution spectra |
| `02_gpu_and_autograd` | GPU execution and differentiable spectra |
| `03_spectral_fitting` | Parameter recovery with `esfit` |
| `04_benchmarks` | Timings against MATLAB/EasySpin |
| `05_fitting_real_world` | Bad starting guesses, loss landscapes, global vs local search |
| `06_blue_copper_fitting` | End-to-end Cu(II) case study |
| `07_easyspin_parity` | Overlays against stored EasySpin references |
| `08_pepper_playground` | Simulate, add noise, fit with every `esfit` method |
| `09_fit_gui` | Interactive fitting panel: sliders, live fit, RMSD trace |

They live in `examples/notebooks/`, with scripted equivalents of EasySpin's
example collection in `examples/{solidstate,liquids,slowmotion,endor,fitting,magnetometry}`.

---

## Accuracy and performance

Every simulator with an EasySpin counterpart is checked against stored MATLAB
reference output on each commit. Agreement is cosine ≥ 0.999 (most
0.9999–1.0000) for `pepper`, `garlic`, `salt` and `curry`, with absolute
intensities matched where EasySpin defines them. The pulse-EPR routes
(`saffron`, `saffron_thyme`, `spidyan`) are held to a stricter metric — the
relative L2 error on the raw *and* the mean-centered signal, 1e-16..1e-7 —
because a cosine is scale-invariant and dominated by a pulse-EPR trace's DC
offset.
`chili` and the stochastic `cardamom` have documented caveats. Details per
simulator, with the evidence: [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

On speed, torchspin is competitive rather than uniformly faster. On an audited
nine-workload comparison it beat EasySpin on six — most strongly on
slow-motion `chili` (7.3×) and trajectory `cardamom` (2.0×) — and was slower on
three, including matrix-method Cu(II) powders (0.34×). The GPU path pays off
for batched and gradient-based work, not for single simulations. Numbers,
replicates and a per-workload device audit:
[BENCHMARK_VERIFICATION.md](benchmarks/results/workstation_20260904_verified/BENCHMARK_VERIFICATION.md).

---

## Citation

If you use torchspin in published work, please cite both:

- **TorchSpin** — Follmer, A.H. *arXiv:2610.18924* (2026) https://arxiv.org/abs/2610.18924.
- **EasySpin**, the MATLAB toolbox this port is based on —
  Stoll, S. & Schweiger, A. *J. Magn. Reson.* **178** (2006) 42–55.


---

## Contributing

Bug reports, documentation fixes and pull requests are welcome. Development
install and the test layout are in [CONTRIBUTING.md](CONTRIBUTING.md):

```bash
git clone https://github.com/follmerlab/torchspin.git
cd torchspin
pip install -e ".[dev]"
pytest -q -m "not slow" -k "not matlab_validation"   # fast pass, ~4 min
```

The MATLAB reference data ships in the repository but not in the PyPI package,
so the cross-validation suite only runs from a checkout.

---

## Related

- [EasySpin](https://easyspin.org) — the MATLAB reference implementation
- [PyTorch](https://pytorch.org) — the autograd engine

## License

MIT — see [LICENSE.md](LICENSE.md). torchspin is derived from EasySpin, which is
also MIT licensed; both copyrights are reproduced there.
