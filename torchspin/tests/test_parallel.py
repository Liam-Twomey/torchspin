"""Tests for torchspin.parallel: the shared CPU thread budget."""
import importlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from torchspin import _linalg, parallel
from torchspin.parallel import cpu_budget, inner_threads, native_thread_env, thread_budget

# the module, not the esfit() function that torchspin/__init__ re-exports under the same name
_esfit = importlib.import_module('torchspin.esfit')


def test_cpu_budget_honors_affinity(monkeypatch):
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {0, 1, 2}, raising=False)
    monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    assert cpu_budget() == 3


def test_cpu_budget_honors_slurm_without_affinity(monkeypatch):
    monkeypatch.delattr(os, 'sched_getaffinity', raising=False)
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '5')
    assert cpu_budget() == 5


def test_cpu_budget_affinity_caps_slurm(monkeypatch):
    # the affinity mask is what the kernel enforces; a larger SLURM value is ignored
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {0, 1}, raising=False)
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '16')
    assert cpu_budget() == 2


def test_cpu_budget_ignores_bad_slurm_value(monkeypatch):
    monkeypatch.delattr(os, 'sched_getaffinity', raising=False)
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', 'not-a-number')
    assert cpu_budget() == (os.cpu_count() or 1)


def test_thread_budget_nests_and_restores():
    base = inner_threads()
    assert base == torch.get_num_threads()
    with thread_budget(3):
        assert inner_threads() == min(3, base)
        with thread_budget(8):           # nesting can only shrink the allowance
            assert inner_threads() == min(3, base)
        with thread_budget(1):
            assert inner_threads() == 1
        assert inner_threads() == min(3, base)
    assert inner_threads() == base


def test_thread_budget_is_per_thread():
    """Concurrent workers hold their own budgets; the caller's is untouched."""
    seen = {}
    barrier = threading.Barrier(2)

    def work(n):
        with thread_budget(n):
            barrier.wait()               # both budgets are active at the same time
            seen[n] = inner_threads()

    prev = torch.get_num_threads()
    torch.set_num_threads(max(prev, 4))
    try:
        with ThreadPoolExecutor(2) as ex:
            list(ex.map(work, [1, 2]))
        assert seen == {1: 1, 2: 2}
        assert inner_threads() == torch.get_num_threads()
    finally:
        torch.set_num_threads(prev)


def test_pool_size_follows_budget():
    prev = torch.get_num_threads()
    torch.set_num_threads(max(prev, 4))
    try:
        assert _linalg._pool_size(10_000) == torch.get_num_threads()
        with thread_budget(2):
            assert _linalg._pool_size(10_000) == 2
        with thread_budget(1):
            assert _linalg._pool_size(10_000) == 1
    finally:
        torch.set_num_threads(prev)


def test_pool_map_stays_inside_budget():
    """pool_map runs at most `budget` items concurrently."""
    active, peak = 0, 0
    lock = threading.Lock()

    def item(_):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        threading.Event().wait(0.01)
        with lock:
            active -= 1
        return 0

    prev = torch.get_num_threads()
    torch.set_num_threads(max(prev, 4))
    try:
        with thread_budget(2):
            _linalg.pool_map(item, list(range(16)))
        assert peak <= 2
    finally:
        torch.set_num_threads(prev)


def test_budgeted_eigh_matches_serial():
    torch.manual_seed(3)
    A = torch.randn(96, 12, 12, dtype=torch.complex128)
    H = A + A.conj().transpose(-1, -2)
    ref = torch.linalg.eigvalsh(H)
    for n in (1, 2, 4):
        with thread_budget(n):
            w = _linalg.eigvalsh(H)
        assert torch.allclose(w, ref, atol=1e-12)


def test_native_thread_env_sets_and_restores(monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '7')
    monkeypatch.delenv('VECLIB_MAXIMUM_THREADS', raising=False)
    with native_thread_env(2):
        for k in parallel.NATIVE_THREAD_VARS:
            assert os.environ[k] == '2'
    assert os.environ['OMP_NUM_THREADS'] == '7'
    assert 'VECLIB_MAXIMUM_THREADS' not in os.environ


def test_thread_budget_blas_cap_optional():
    tpc = pytest.importorskip('threadpoolctl')
    with thread_budget(1, blas=True):
        infos = [i for i in tpc.threadpool_info() if i.get('user_api') == 'blas']
        assert all(i['num_threads'] == 1 for i in infos)


def test_esfit_auto_workers_use_cpu_budget(monkeypatch):
    import importlib
    res = importlib.import_module('torchspin._resources')
    monkeypatch.setattr(res, 'cpu_budget', lambda: 3)
    monkeypatch.setattr(res, 'available_memory', lambda: None)
    assert _esfit._resolve_workers('auto', 100) == 3
    assert _esfit._resolve_workers('auto', 2) == 2
    assert _esfit._resolve_workers(5, 100) == 5


def test_esfit_threads_per_worker_fits_budget(monkeypatch):
    monkeypatch.setattr(parallel, 'cpu_budget', lambda: 8)
    prev = torch.get_num_threads()
    torch.set_num_threads(max(prev, 8))
    try:
        assert _esfit._threads_per_worker(4) == 2
        assert _esfit._threads_per_worker(3) == 2
        assert _esfit._threads_per_worker(16) == 1
    finally:
        torch.set_num_threads(prev)


def test_esfit_thread_backend_keeps_full_inner_budget(monkeypatch):
    """The thread backend must not shrink the inner budget per worker. Native BLAS
    threads cannot be capped per Python thread, so a smaller budget only removes
    _linalg's serialized single-threaded pool and oversubscribes the cores."""
    monkeypatch.setattr(parallel, 'cpu_budget', lambda: 4)
    prev = torch.get_num_threads()
    torch.set_num_threads(max(prev, 4))
    seen = []

    def objective(x):
        seen.append(inner_threads())
        return float(np.sum(x ** 2))

    try:
        X = np.arange(8, dtype=float).reshape(4, 2)
        F = _esfit._eval_population(objective, X, n_workers=2, workers='threads')
        assert np.allclose(F, (X ** 2).sum(axis=1))
        assert seen and all(s == torch.get_num_threads() for s in seen)
    finally:
        torch.set_num_threads(prev)


def test_native_thread_env_passes_passive_wait_policy_to_children(monkeypatch):
    from torchspin.parallel import native_thread_env
    monkeypatch.delenv('OMP_WAIT_POLICY', raising=False)
    with native_thread_env(2):
        assert os.environ['OMP_WAIT_POLICY'] == 'PASSIVE'
    assert 'OMP_WAIT_POLICY' not in os.environ
    monkeypatch.setenv('OMP_WAIT_POLICY', 'ACTIVE')
    with native_thread_env(2):
        assert os.environ['OMP_WAIT_POLICY'] == 'ACTIVE'
