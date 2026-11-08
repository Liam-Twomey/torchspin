"""Heavy, realistic workloads for performance campaigns (docs/v1_acceleration/ROADMAP.md).

Each workload is a zero-argument callable returning the simulation result, built
from parameters chosen so that a single call takes seconds to minutes on a CPU
core.  ``python benchmarks/workloads/run_workloads.py --list`` prints them.
"""
from __future__ import annotations

import numpy as np

from torchspin import (SpinSystem, Experiment, Options, pepper, garlic, chili, saffron,
                       cardamom, CardamomPar, CardamomOptions, esfit, FitOptions)
from torchspin.saffron import PulseExperiment, SaffronOptions


def _pepper_cu_2n(device='cpu', grid=(19, 4)):
    sys = SpinSystem(S=[0.5], g=[2.05, 2.05, 2.25], Nucs='63Cu,14N,14N',
                     A=[[30, 30, 500], [40, 40, 40], [40, 40, 40]], lwpp=[0.5, 0])
    exp = Experiment(mwFreq=9.5, Range=[260, 360], nPoints=2048)
    return lambda: pepper(sys, exp, Options(Verbosity=0, GridSize=list(grid), device=device))


def _pepper_mn(device='cpu', grid=(19, 4)):
    sys = SpinSystem(S=[2.5], g=2.0, D=[[600, 100]], Nucs='55Mn', A=[[250, 250, 250]], lwpp=[1, 0])
    exp = Experiment(mwFreq=9.5, Range=[260, 360], nPoints=2048)
    return lambda: pepper(sys, exp, Options(Verbosity=0, GridSize=list(grid), device=device))


def _pepper_triplet_perturb_dense(device='cpu', grid=(91, 1)):
    sys = SpinSystem(S=[0.5], g=[2.0023, 2.0025, 2.0035], Nucs='14N,1H,1H', A=[[20, 20, 90], [8, 8, 12], [5, 5, 7]], lwpp=[0.2, 0])
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=4096)
    return lambda: pepper(sys, exp, Options(Verbosity=0, GridSize=list(grid), Method='perturb2', device=device))


def _pepper_strain(device='cpu', grid=(61, 4)):
    sys = SpinSystem(S=[0.5], g=[2.008, 2.006, 2.003], Nucs='14N', A=[[20, 20, 85]],
                     gStrain=[[0.003, 0.002, 0.001]], HStrain=[10, 10, 30], lwpp=[0.3, 0])
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=2048)
    return lambda: pepper(sys, exp, Options(Verbosity=0, GridSize=list(grid), device=device))


def _pepper_batch_fit_loop(device='cpu', n=20):
    """Parameter sweep: the pattern behind esfit / autograd — n forward calls."""
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=1024)
    gs = np.linspace(2.000, 2.010, n)

    def run():
        out = []
        for g in gs:
            sys = SpinSystem(S=[0.5], g=[float(g), 2.006, 2.003], Nucs='14N', A=[[20, 20, 85]], lwpp=[0.3, 0])
            out.append(pepper(sys, exp, Options(Verbosity=0, GridSize=[31, 4], device=device))[1])
        return out
    return run


def _chili_nitroxide_2nuc(device='cpu'):
    # chili is CPU-only (scipy.sparse SLE solver); ``device`` is accepted for a
    # uniform factory signature but has no effect (see WORKLOAD_DEVICES).
    # tcorr 30 ns needs a large LLMK basis (EasySpin auto-sizing)
    sys = SpinSystem(S=[0.5], g=[2.008, 2.006, 2.003], Nucs='14N,1H', A=[[20, 20, 85], [5, 5, 8]], tcorr=3e-8, lw=[0.1, 0.1])
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=1024)
    return lambda: chili(sys, exp)


def _chili_powder_potential(device='cpu'):
    sys = SpinSystem(S=[0.5], g=[2.008, 2.006, 2.003], Nucs='14N', A=[[20, 20, 85]], tcorr=2e-8, lw=[0.1, 0.1],
                     Potential=[[2, 0, 0, 1.5]])
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=1024)
    return lambda: chili(sys, exp)


def _saffron_hyscore(device='cpu', n=512):
    sys = SpinSystem(S=[0.5], g=[2.0023] * 3, Nucs='14N,1H', A=[[3, 3, 6], [2, 2, 8]], Q=[[-0.5, -0.5, 1.0], [0, 0, 0]])
    exp = PulseExperiment(Sequence='HYSCORE', Field=350, dt=0.016, nPoints=n, tau=0.1)
    return lambda: saffron(sys, exp, SaffronOptions(GridSize=91, device=device))


def _cardamom_diffusion(device='cpu', ntraj=200, nsteps=1000):
    # cardamom is CPU-only (NumPy trajectory propagation); ``device`` has no
    # effect.  The ensemble is seeded, so every call propagates the same
    # trajectories and outputs are comparable across runs and thread counts.
    sys = SpinSystem(S=[0.5], g=[[2.008, 2.006, 2.003]], Nucs='14N', A=[[20, 20, 85]], tcorr=1e-9)
    exp = Experiment(mwFreq=9.5, Range=[332, 352], nPoints=256, Harmonic=0)
    return lambda: cardamom(sys, exp, CardamomPar(Model='diffusion', nTraj=ntraj, nSteps=nsteps, dtSpin=1e-10,
                                                  dtSpatial=1e-10, seed=2026),
                            CardamomOptions(Method='fast', Verbosity=0))


# --------------------------------------------------------------------------- esfit
# End-to-end fits.  A fixed iteration budget keeps the work per call constant, so
# the timing measures the evaluation machinery (finite-difference Jacobians,
# population pools, nested thread pools), not how fast a run happens to converge.
_FIT_EXP = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=1024)
_FIT_TRUE = np.array([2.0080, 2.0060, 2.0030, 0.30])        # gx, gy, gz, lwpp (mT)
_FIT_START = np.array([2.0070, 2.0068, 2.0024, 0.40])


class _FitModel:
    """Nitroxide pepper model of (gx, gy, gz, lwpp); picklable for process pools."""

    def __init__(self, device='cpu', grid=(31, 4)):
        self.device, self.grid = device, list(grid)

    def __call__(self, p):
        sys = SpinSystem(S=[0.5], g=[float(p[0]), float(p[1]), float(p[2])], Nucs='14N',
                         A=[[20, 20, 85]], lwpp=[float(p[3]), 0])
        return pepper(sys, _FIT_EXP, Options(Verbosity=0, GridSize=self.grid, device=self.device))[1].cpu().numpy()


def _fit_data(model):
    return model(_FIT_TRUE)


def _esfit_levmar(device='cpu'):
    """levmar, 4 parameters: serial finite-difference Jacobian (5 calls per J) today."""
    model = _FitModel(device)
    data = _fit_data(model)
    opt = FitOptions(method='levmar', max_iter=8, compute_uncertainties=True, verbosity=0)
    return lambda: esfit(data, model, p0=_FIT_START, vary=np.array([0.003, 0.003, 0.003, 0.2]), options=opt)


def _esfit_swarm(device='cpu', workers='processes'):
    """Particle swarm, 4 parameters, population evaluated with n_workers='auto'."""
    model = _FitModel(device)
    data = _fit_data(model)
    opt = FitOptions(method='swarm', swarm_size=32, max_iter=4, compute_uncertainties=False,
                     n_workers='auto', workers=workers, verbosity=0)
    return lambda: esfit(data, model, p0=_FIT_START, vary=np.array([0.003, 0.003, 0.003, 0.2]), options=opt)


def _esfit_swarm_threads_matrix(device='cpu'):
    """Nested parallelism stress case: esfit thread backend over the matrix method
    (Cu/2N, 72x72), whose eigh thread pool runs inside every worker."""
    sys0 = dict(S=[0.5], Nucs='63Cu,14N,14N', A=[[30, 30, 500], [40, 40, 40], [40, 40, 40]], lwpp=[0.5, 0])
    exp = Experiment(mwFreq=9.5, Range=[260, 360], nPoints=1024)

    def model(p):
        sys = SpinSystem(g=[float(p[0]), float(p[0]), float(p[1])], **sys0)
        return pepper(sys, exp, Options(Verbosity=0, GridSize=[10, 4], device=device))[1].cpu().numpy()

    data = model(np.array([2.05, 2.25]))
    opt = FitOptions(method='swarm', swarm_size=8, max_iter=2, compute_uncertainties=False,
                     n_workers='auto', workers='threads', verbosity=0)
    return lambda: esfit(data, model, p0=np.array([2.06, 2.24]), vary=np.array([0.02, 0.02]), options=opt)


# Devices each workload can actually execute on.  ``chili`` (scipy.sparse
# stochastic-Liouville solver) and ``cardamom`` (NumPy trajectory propagation)
# never touch a CUDA device: their ``device`` argument is ignored, so they must
# not be timed as CUDA runs.  ``pepper`` and ``saffron`` accept
# ``Options.device`` / ``SaffronOptions.device``; on CUDA parts of the work stay
# on the host (grid interpolation, triangle projection, and the eigh CPU pool
# for matrices larger than 32x32 -- torchspin/_linalg.py), which the runner
# reports as ``hybrid`` execution.
WORKLOAD_DEVICES = {
    'pepper_cu_2n_matrix': ('cpu', 'cuda'),
    'pepper_mn_S52_matrix': ('cpu', 'cuda'),
    'pepper_perturb_dense_grid': ('cpu', 'cuda'),
    'pepper_strain_summation': ('cpu', 'cuda'),
    'pepper_fit_loop_20': ('cpu', 'cuda'),
    'chili_nitroxide_2nuc': ('cpu',),
    'chili_powder_potential': ('cpu',),
    'saffron_hyscore_512': ('cpu', 'cuda'),
    'cardamom_diffusion_200x1000': ('cpu',),
    'esfit_levmar_4p': ('cpu', 'cuda'),
    'esfit_swarm_4p_procs': ('cpu',),
    'esfit_swarm_4p_threads': ('cpu', 'cuda'),
    'esfit_swarm_threads_matrix': ('cpu',),
}

# Workloads whose output differs between calls; CPU/CUDA consistency checks and
# cross-run comparisons must allow for it.  (cardamom is seeded now; the swarm fits
# draw their initial population from an unseeded generator.)
STOCHASTIC = {'esfit_swarm_4p_procs', 'esfit_swarm_4p_threads', 'esfit_swarm_threads_matrix'}


def supports_device(name: str, device: str) -> bool:
    """True if ``WORKLOADS[name]`` genuinely executes on ``device``."""
    key = 'cuda' if str(device).startswith('cuda') else str(device)
    return key in WORKLOAD_DEVICES.get(name, ('cpu',))


WORKLOADS = {
    'pepper_cu_2n_matrix': _pepper_cu_2n,
    'pepper_mn_S52_matrix': _pepper_mn,
    'pepper_perturb_dense_grid': _pepper_triplet_perturb_dense,
    'pepper_strain_summation': _pepper_strain,
    'pepper_fit_loop_20': _pepper_batch_fit_loop,
    'chili_nitroxide_2nuc': _chili_nitroxide_2nuc,
    'chili_powder_potential': _chili_powder_potential,
    'saffron_hyscore_512': _saffron_hyscore,
    'cardamom_diffusion_200x1000': _cardamom_diffusion,
    'esfit_levmar_4p': _esfit_levmar,
    'esfit_swarm_4p_procs': lambda device='cpu': _esfit_swarm(device, workers='processes'),
    'esfit_swarm_4p_threads': lambda device='cpu': _esfit_swarm(device, workers='threads'),
    'esfit_swarm_threads_matrix': _esfit_swarm_threads_matrix,
}
