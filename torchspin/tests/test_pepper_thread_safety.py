"""pepper's slot bookkeeping must be private to each thread.

esfit's thread backend (FitOptions(workers='threads')) runs several pepper calls at
once. The per-call slot keys used to live in a function attribute shared by the
whole process, so one call could read another call's keys. That raised IndexError
in _filter_slots, or mislabeled transitions with separate='transitions'.
"""
import importlib
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from torchspin import Experiment, Options, SpinSystem, pepper
# the module, not the pepper() function that torchspin/__init__ re-exports under that name
_pm = importlib.import_module('torchspin.pepper')


def _slots(n_pairs: int, n_orient: int = 3):
    """Synthetic resonances: n_pairs distinct level pairs at every orientation."""
    B = [torch.linspace(300.0, 310.0, n_pairs, dtype=torch.float64) + k for k in range(n_orient)]
    I = [torch.linspace(1.0, 0.1, n_pairs, dtype=torch.float64) for _ in range(n_orient)]
    P = [torch.tensor([[0, 1 + j] for j in range(n_pairs)]) for _ in range(n_orient)]
    return B, I, P, n_orient


def test_slot_keys_are_per_thread():
    """Thread A's keys survive thread B building its own slots in between."""
    a_built, b_done = threading.Event(), threading.Event()
    out = {}

    def thread_a():
        B, I, P, n = _slots(4)
        Bs, Is, _ = _pm._transition_slots(B, I, None, P, n)
        a_built.set()
        b_done.wait(5)
        # filter with a threshold that keeps the 2 strongest of A's 4 slots
        _pm._filter_slots(Bs, Is, None, 0.6)
        out['a'] = list(_pm._slot_state.keys)

    def thread_b():
        a_built.wait(5)
        B, I, P, n = _slots(1)
        _pm._transition_slots(B, I, None, P, n)
        out['b'] = list(_pm._slot_state.keys)
        b_done.set()

    ta, tb = threading.Thread(target=thread_a), threading.Thread(target=thread_b)
    ta.start(); tb.start(); ta.join(); tb.join()
    assert out['b'] == [(0, 1, 0)]
    assert out['a'] == [(0, 1, 0), (0, 2, 0)], out['a']


def test_concurrent_pepper_matches_serial():
    """Matrix-method pepper in a thread pool gives the serial spectra exactly."""
    exp = Experiment(mwFreq=9.5, Range=[300, 360], nPoints=512)
    kw = dict(S=[0.5], Nucs='63Cu', A=[[30, 30, 500]], lwpp=[0.8, 0])

    def run(g):
        sys = SpinSystem(g=[g, g, 2.25 + (g - 2.05)], **kw)
        return pepper(sys, exp, Options(Verbosity=0, GridSize=[7, 4], Method='matrix'))[1].numpy()

    gs = list(np.linspace(2.03, 2.07, 8))
    ref = [run(g) for g in gs]
    with ThreadPoolExecutor(4) as ex:
        got = list(ex.map(run, gs))
    for r, o in zip(ref, got):
        np.testing.assert_allclose(o, r, rtol=0, atol=1e-12 * np.abs(r).max())
