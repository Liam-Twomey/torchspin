"""OpenMP wait policy: idle torch threads should sleep, not spin.

torch's OpenMP threads spin-wait after every parallel region by default. A fit
evaluates many small spectra, so the Python thread and torchspin's own worker
threads end up competing with spinning threads for the cores, and on many-core
Linux machines a fit ran several times slower with 8 threads than with 1.
``OMP_WAIT_POLICY=PASSIVE`` makes idle threads sleep. The OpenMP runtime reads it
once, when torch loads it, so it must be in the environment before torch is
imported. This module must not import torch at load time.
"""
from __future__ import annotations

import contextlib
import os
import sys
import warnings

VAR = 'OMP_WAIT_POLICY'

_HINT = ("torch was imported before torchspin, so torchspin could not set {var}=PASSIVE; "
         "idle OpenMP threads will spin-wait, which can make fits several times slower "
         "with more than one thread. Import torchspin first, or set the variable before "
         "Python starts: export {var}=PASSIVE (any explicit value silences this warning).")


def configure_at_import() -> None:
    """Called by ``torchspin/__init__`` before anything imports torch."""
    if 'torch' not in sys.modules:
        os.environ.setdefault(VAR, 'PASSIVE')
        return
    import torch
    if should_warn(True, os.environ.get(VAR), sys.platform, torch.get_num_threads()):
        warnings.warn(_HINT.format(var=VAR), RuntimeWarning, stacklevel=3)


def should_warn(torch_loaded: bool, value, platform: str, n_threads: int) -> bool:
    """Warn only where it matters: torch already loaded, nothing set, Linux, threads > 1."""
    return bool(torch_loaded and value is None and platform.startswith('linux') and n_threads > 1)


@contextlib.contextmanager
def passive_for_children():
    """Make processes spawned inside this block start with ``OMP_WAIT_POLICY=PASSIVE``.

    A spawned worker loads its own OpenMP runtime, so it gets the policy even when
    the parent imported torch too early. An explicit user value is passed on unchanged.
    """
    had = VAR in os.environ
    os.environ.setdefault(VAR, 'PASSIVE')
    try:
        yield
    finally:
        if not had:
            os.environ.pop(VAR, None)
