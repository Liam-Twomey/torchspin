"""CPU thread budget shared by every level of parallelism in torchspin.

torchspin can run three nested levels of CPU parallelism:

1. an outer pool of whole simulations (esfit population methods,
   :func:`torchspin.ml.simulate_batch`);
2. inner thread pools inside one simulation (the batched eigendecomposition and
   chunked accumulation in :mod:`torchspin._linalg`);
3. the native BLAS/OpenMP pools of torch, NumPy and SciPy.

If each level sizes itself to the whole machine, they multiply and oversubscribe
the cores. A small fit on a 32-core node can then run several times slower than
on 8. This module gives the levels one shared budget:

* :func:`cpu_budget` is the number of cores this process may use. It honors the
  CPU affinity mask (``taskset``, cgroups, batch schedulers), then
  ``SLURM_CPUS_PER_TASK``, then ``os.cpu_count()``.
* :func:`thread_budget` is a context manager that caps the inner pools for the
  enclosed code. It works per thread (a :class:`contextvars.ContextVar`), so
  concurrent workers of an outer thread pool can each hold their own share. With
  ``blas=True`` and ``threadpoolctl`` installed (the ``[hpc]`` extra) it also caps
  the NumPy/SciPy BLAS pools. That cap is process-wide, so use it only from a
  single thread.
* :func:`native_thread_env` sets the environment variables that size the native
  pools of a *child* process. They are read at library load, so they must be in
  place before a spawned worker starts.

The budget only limits how many workers torchspin's own pools start. It never
changes results: each matrix or chunk is still processed by the same routine.
"""
from __future__ import annotations

import contextlib
import contextvars
import os

__all__ = ['cpu_budget', 'inner_threads', 'thread_budget', 'native_thread_env',
           'NATIVE_THREAD_VARS']

#: Environment variables that size the native thread pools of torch, NumPy and SciPy
#: (OpenMP, OpenBLAS, MKL, numexpr, Apple Accelerate).
NATIVE_THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                      'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')

_inner: contextvars.ContextVar = contextvars.ContextVar('torchspin_inner_threads', default=None)


def cpu_budget() -> int:
    """Number of CPU cores this process may use.

    In order of precedence: the CPU affinity mask (Linux ``sched_getaffinity``,
    which reflects ``taskset`` and scheduler cgroups), ``SLURM_CPUS_PER_TASK``,
    and ``os.cpu_count()``. A SLURM value larger than the affinity mask is ignored,
    because the mask is what the kernel enforces.
    """
    n = None
    getaff = getattr(os, 'sched_getaffinity', None)
    if getaff is not None:
        try:
            n = len(getaff(0))
        except OSError:
            n = None
    slurm = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm:
        try:
            s = int(slurm)
        except ValueError:
            s = None
        if s and s > 0:
            n = s if n is None else min(n, s)
    if n is None:
        n = os.cpu_count() or 1
    return max(1, int(n))


def inner_threads() -> int:
    """Worker count available to torchspin's inner pools in the current context.

    This is the value set by the innermost enclosing :func:`thread_budget`, or
    ``torch.get_num_threads()`` when no budget is active. It never exceeds
    ``torch.get_num_threads()``, so an explicit ``torch.set_num_threads`` still caps it.
    """
    import torch
    n = torch.get_num_threads()
    b = _inner.get()
    return max(1, min(n, b)) if b is not None else max(1, n)


@contextlib.contextmanager
def thread_budget(n: int, blas: bool = False):
    """Cap torchspin's inner pools at ``n`` threads (and BLAS too, with ``blas=True``).

    Example::

        with torchspin.parallel.thread_budget(4):
            x, y = pepper(sys, exp, opt)   # eigh pool and accumulation use <= 4 threads

    Nested budgets can only shrink the allowance. The inner-pool setting is local
    to the calling thread's context, so concurrent workers can hold different
    budgets.

    ``blas=True`` also applies ``threadpoolctl.threadpool_limits`` to the BLAS
    libraries loaded in the process (a no-op if ``threadpoolctl`` is not installed).
    That limit is process-wide and not thread-safe, so do not use ``blas=True``
    inside concurrent workers. Give worker *processes* their caps with
    :func:`native_thread_env` instead.
    """
    n = max(1, int(n))
    outer = _inner.get()
    if outer is not None:
        n = min(n, outer)
    token = _inner.set(n)
    try:
        limiter = contextlib.nullcontext()
        if blas:
            try:
                from threadpoolctl import threadpool_limits
            except ImportError:
                pass
            else:
                limiter = threadpool_limits(limits=n, user_api='blas')
        with limiter:
            yield n
    finally:
        _inner.reset(token)


@contextlib.contextmanager
def native_thread_env(n: int):
    """Temporarily set :data:`NATIVE_THREAD_VARS` to ``n`` (for spawning workers).

    Child processes inherit the environment at spawn and size their native pools
    from it at import. Without these caps every spawned worker starts one thread
    per core. The previous values are restored on exit.
    """
    n = max(1, int(n))
    saved = {k: os.environ.get(k) for k in NATIVE_THREAD_VARS}
    for k in NATIVE_THREAD_VARS:
        os.environ[k] = str(n)
    try:
        yield n
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
