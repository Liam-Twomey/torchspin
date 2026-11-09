"""Finite-difference probes evaluated concurrently give the sequential fit.

levmar's initial Jacobian, the trf Jacobians, the L-BFGS-B gradients and the
uncertainty Jacobian each make several independent model calls. With
FitOptions(n_workers > 1) they run in a worker pool. The probe points, the order
in which results are recorded and the number of evaluations must not change, so
for a deterministic model the fitted parameters and uncertainties are the same.
"""
import inspect
import warnings

import numpy as np
import pytest
from scipy.optimize import least_squares

from torchspin.esfit import FitOptions, _jacobian_finite_diff, esfit

_X = np.linspace(-5.0, 5.0, 201)


def _two_lines(p):
    """Module-level (picklable) model: two Gaussians, centers p[0], p[1], widths p[2], p[3]."""
    return (np.exp(-0.5 * ((_X - p[0]) / p[2]) ** 2)
            + 0.6 * np.exp(-0.5 * ((_X - p[1]) / p[3]) ** 2))


_TRUE = np.array([-1.2, 1.5, 0.8, 0.5])
_START = np.array([-1.0, 1.3, 0.9, 0.6])
_VARY = np.array([0.5, 0.5, 0.3, 0.3])
_DATA = _two_lines(_TRUE) + 0.01 * np.random.default_rng(1).standard_normal(_X.size)

_SCIPY_WORKERS = 'workers' in inspect.signature(least_squares).parameters


def _fit(method, n_workers, workers='threads'):
    opt = FitOptions(method=method, max_iter=30, compute_uncertainties=True, verbosity=0,
                     n_workers=n_workers, workers=workers)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return esfit(_DATA, _two_lines, p0=_START, vary=_VARY, options=opt)


@pytest.mark.parametrize('method', ['levmar', 'trf', 'lbfgsb'])
def test_parallel_probes_match_sequential_threads(method):
    if method in ('trf', 'lbfgsb') and not _SCIPY_WORKERS:
        pytest.skip('SciPy < 1.16 has no workers= for finite differences')
    seq, par = _fit(method, 1), _fit(method, 4, 'threads')
    np.testing.assert_array_equal(par.pfit, seq.pfit)
    np.testing.assert_array_equal(par.pstd, seq.pstd)
    assert par.n_evaluations == seq.n_evaluations


@pytest.mark.parametrize('method', ['levmar', 'trf'])
def test_parallel_probes_match_sequential_processes(method):
    if method == 'trf' and not _SCIPY_WORKERS:
        pytest.skip('SciPy < 1.16 has no workers= for finite differences')
    seq, par = _fit(method, 1), _fit(method, 3, 'processes')
    # a pure-NumPy model gives the same numbers in every process
    np.testing.assert_array_equal(par.pfit, seq.pfit)
    np.testing.assert_array_equal(par.pstd, seq.pstd)
    assert par.n_evaluations == seq.n_evaluations


def test_uncertainty_jacobian_parallel_equals_sequential():
    x = _TRUE.copy()
    J1 = _jacobian_finite_diff(_two_lines, x)
    J4 = _jacobian_finite_diff(_two_lines, x, n_workers=4, workers='threads')
    np.testing.assert_array_equal(J4, J1)
    # and it is the central difference it claims to be
    i, step = 2, 1e-7 * max(abs(x[2]), 1.0)
    xp, xm = x.copy(), x.copy(); xp[i] += step; xm[i] -= step
    np.testing.assert_array_equal(J1[:, i], (_two_lines(xp) - _two_lines(xm)) / (2 * step))


def test_parallel_probes_reach_the_workers():
    """With n_workers > 1 the probes really run on pool threads."""
    import threading
    seen = set()
    main = threading.get_ident()

    def model(p):
        seen.add(threading.get_ident())
        return _two_lines(p)

    opt = FitOptions(method='levmar', max_iter=3, compute_uncertainties=True, verbosity=0,
                     n_workers=4, workers='threads')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        esfit(_DATA, model, p0=_START, vary=_VARY, options=opt)
    assert seen - {main}, 'no model call left the main thread'


def _visible_devices(_x):
    """Module-level (picklable): what a worker sees."""
    import os
    return os.environ.get('CUDA_VISIBLE_DEVICES')


def test_devices_pin_workers_round_robin():
    """FitOptions.devices gives each spawned worker one GPU, round-robin."""
    import importlib
    from torchspin.esfit import _FIT_DEVICES, _map_points
    E = importlib.import_module('torchspin.esfit')  # the module, not the esfit() function
    token = _FIT_DEVICES.set(['cuda:0', 'cuda:1'])
    try:
        E._POOL = None if E._POOL is None else (E._POOL[2].terminate() or None)
        seen = _map_points(_visible_devices, list(range(16)), 4, 'processes')
    finally:
        _FIT_DEVICES.reset(token)
        if E._POOL is not None:
            E._POOL[2].terminate(); E._POOL = None
    # the devices exist only as strings here: no GPU is needed for the assignment
    assert set(seen) == {'0', '1'}


def test_devices_reject_non_cuda():
    from torchspin.esfit import _cuda_index
    assert _cuda_index('cuda:1') == '1' and _cuda_index('cuda') == '0' and _cuda_index(2) == '2'
    with pytest.raises(ValueError):
        _cuda_index('cpu')
