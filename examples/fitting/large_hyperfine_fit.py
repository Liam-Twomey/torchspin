"""Staged fit of a large-hyperfine solution spectrum (natural-abundance Zr)

Natural zirconium is 11 % 91Zr (I = 5/2) and 89 % nuclear-spin-free isotopes.
At X-band with A(91Zr) of about 780 MHz, nearly 90 % of the intensity is
in the central I = 0 line. That line depends on g and the line widths but not
on A. The six 91Zr satellites are weak and spread over about 140 mT, so the
RMSD as a function of A has a false minimum wherever a shifted satellite comb
still overlaps part of the data. Here, a one-shot Levenberg-Marquardt fit of
all four parameters started at A = 740 MHz stops at A = 750 MHz (true 780).
method='global' finds 780 MHz on this data set, but takes about 10 times as
long as the staged fit. It has also been seen to lock onto a wrong satellite
assignment on similar data.

The staged recipe avoids that:

  1. Fit g and the widths with A fixed. The central line does not depend on A.
  2. Scan A on a 1-D grid with g and widths fixed, and keep the best value.
     Each satellite moves by about m_I·A/(g·mu_B), so the RMSD has one sharp
     minimum per satellite assignment.
  3. Polish all parameters together with a local optimizer (Levenberg-Marquardt).

Two further points for large hyperfine couplings (see docs/pitfalls.md):

* garlic's default ``Method='exact'`` (Breit-Rabi) is the right choice here.
  Low-order perturbation theory is badly wrong at A/nu of about 0.08. For
  isotropic 91Zr with A = 780 MHz at 9.65 GHz, the relative L2 error against
  the exact spectrum is 0.82 for perturb2 and 0.11 for perturb3.
* The model passes ``Options(AccumMethod='linear')``. The default nearest-bin
  accumulation is piecewise-constant in g and A, which makes finite-difference
  optimizers and uncertainties blind to the line positions.

The data are synthetic.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchspin import SpinSystem, garlic
from torchspin.experiment import Experiment, Options
from torchspin.esfit import esfit, FitOptions

EXP = Experiment(mwFreq=9.65, Range=[290.0, 450.0], nPoints=2048, Harmonic=1)
OPT = Options(AccumMethod='linear')
TRUE = dict(g=1.885, A=780.0, lwG=0.8, lwL=1.2)


def simulate(g, A, lwG, lwL):
    sys = SpinSystem(S=[0.5], g=g, Nucs=['Zr'], A=A, lw=[abs(lwG), abs(lwL)])
    return np.asarray(garlic(sys, EXP, OPT)[1], dtype=float)


def run(seed: int = 0):
    clean = simulate(**TRUE)
    rng = np.random.default_rng(seed)
    data = clean + 0.004 * np.ptp(clean) * rng.standard_normal(clean.size)

    # Starting guess: g close, A off by 40 MHz (more than a line width at every satellite)
    g0, A0, lwG0, lwL0 = 1.8856, 740.0, 1.2, 0.8
    quiet = dict(verbosity=0, target='fcn')

    # Stage 1: g and widths, A fixed
    r1 = esfit(data, lambda p: simulate(p[0], A0, p[1], p[2]),
               np.array([g0, lwG0, lwL0]), vary=np.array([0.003, 1.0, 1.0]),
               options=FitOptions(method='levmar', max_iter=200,
                                  compute_uncertainties=False, **quiet))
    g1, lwG1, lwL1 = r1.pfit
    print(f"Stage 1  g = {g1:.5f}, lw = [{lwG1:.3f}, {lwL1:.3f}] mT  (A fixed at {A0:.0f} MHz)")

    # Stage 2: 1-D scan of A with the stage-1 values fixed
    A_grid = np.arange(600.0, 950.0, 2.0)
    rmsd = []
    for A in A_grid:
        sim = simulate(g1, A, lwG1, lwL1)
        scale = max(float(sim @ data) / float(sim @ sim), 0.0)
        rmsd.append(np.sqrt(np.mean((scale * sim - data) ** 2)))
    rmsd = np.asarray(rmsd)
    A2 = float(A_grid[np.argmin(rmsd)])
    print(f"Stage 2  A = {A2:.0f} MHz from a {A_grid.size}-point scan")

    # Stage 3: local polish of everything
    r3 = esfit(data, lambda p: simulate(*p), np.array([g1, A2, lwG1, lwL1]),
               vary=np.array([0.002, 10.0, 0.5, 0.5]),
               options=FitOptions(method='levmar', max_iter=300, **quiet))
    for name, val, err, true in zip(['g', 'A', 'lwG', 'lwL'], r3.pfit, r3.pstd, TRUE.values()):
        print(f"Stage 3  {name:4s} = {val:10.5f} ± {err:.2g}   (true {true})")

    return EXP, data, A_grid, rmsd, r1, r3


def _plot(exp, data, A_grid, rmsd, r1, r3):
    B = np.linspace(exp.Range[0], exp.Range[1], exp.nPoints)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    axes[0].plot(B, data, 'k', lw=0.6, label='data (synthetic)')
    axes[0].plot(B, r1.fit, 'g--', lw=1.0, label='stage 1 (A fixed, wrong)')
    axes[0].plot(B, r3.fit, 'r', lw=1.0, label='stage 3 (polished)')
    axes[0].set_xlabel('Magnetic field (mT)')
    axes[0].legend(fontsize=8)
    axes[1].plot(A_grid, rmsd, 'b.-', ms=3)
    axes[1].set_xlabel('A(91Zr) (MHz)')
    axes[1].set_ylabel('RMSD (stage 2 scan)')
    axes[2].plot(B, data - r3.fit, 'b', lw=0.6)
    axes[2].axhline(0, color='k', lw=0.5)
    axes[2].set_xlabel('Magnetic field (mT)')
    axes[2].set_ylabel('Residual')
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    _plot(*run())
