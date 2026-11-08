"""Machine-learning helpers: batched simulation over parameter sets, parameter
packing, on-the-fly datasets and differentiable losses.

Every simulator in torchspin that is differentiable (``pepper`` via
:func:`torchspin.pepper_autograd.pepper_autograd`, ``garlic``, ``salt``,
``saffron``, ``curry``) accepts a :class:`~torchspin.spinsystem.SpinSystem`
whose parameter tensors may ``requires_grad``.  This module packages the
plumbing around that for training and inference workflows:

* :class:`ParamSpec` — which spin-system fields are free, their shapes, ranges
  and transforms (``'linear'``, ``'log'``, ``'softplus'``), with
  :meth:`ParamSpec.pack` / :meth:`ParamSpec.unpack` between a flat tensor and a
  :class:`SpinSystem` built from a template (on the autograd graph when the
  flat tensor requires grad).
* :func:`simulate` / :func:`simulate_batch` — one call per parameter row
  (``pepper``, ``garlic``, ``salt``, ``saffron``, ``curry`` or a callable),
  ``(N, nPoints)`` output, optional process pool for no-grad data generation.
* :class:`SpectrumDataset` — an :class:`torch.utils.data.IterableDataset`
  sampling parameters from ranges (or a callable), simulating, normalizing,
  resampling to a fixed axis and adding noise, deterministic for a given seed.
* :func:`cosine_loss`, :func:`rmsd`, :func:`normalize`, :func:`resample` —
  torch operations, so a network's parameter prediction can be refined through
  the simulator with a physics loss.

Example::

    from torchspin import SpinSystem, Experiment, Options
    from torchspin.ml import ParamSpec, simulate_batch, SpectrumDataset

    template = SpinSystem(S=[0.5], g=[[2.05, 2.05, 2.25]], Nucs='63Cu', A=[[30, 30, 500]], lw=[0.5, 0])
    spec = ParamSpec({'g': [(2.0, 2.1), (2.0, 2.1), (2.15, 2.35)],
                      'A': [(20, 60), (20, 60), (350, 650)],
                      'lw': [(0.2, 1.5), (0.0, 0.0)]})
    exp = Experiment(mwFreq=9.5, Range=[260, 360], nPoints=1024, Harmonic=1)
    opt = Options(GridSize=[31, 4])
    ds = SpectrumDataset(spec, template, exp, opt, simulator='pepper', n=1000, seed=0,
                         normalize='maxabs', noise=0.01)
    x, params = next(iter(ds))          # x: (1024,), params: flat tensor
"""
from __future__ import annotations

import copy
import dataclasses
import math
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import torch

from torchspin.experiment import Experiment, Options
from torchspin.spinsystem import SpinSystem

__all__ = ['ParamSpec', 'simulate', 'simulate_batch', 'SpectrumDataset', 'cosine_loss', 'rmsd',
           'normalize', 'resample']

_F64 = torch.float64


# =============================================================================
# Parameter packing
# =============================================================================

class ParamSpec:
    """Free parameters of a spin system: ``{field: [(low, high), ...]}``.

    ``field`` is a :class:`SpinSystem` attribute (``'g'``, ``'A'``, ``'D'``,
    ``'Q'``, ``'lw'``, ``'lwpp'``, ``'tcorr'``, ``'HStrain'``, ``'gStrain'``,
    ``'AStrain'``, ``'ee'`` ...).  The list gives one ``(low, high)`` per element
    of the *flattened* field as stored in the template (``g`` is ``(1, 3)`` for
    one electron, ``A`` is ``(nNuclei, 3)`` ...); equal bounds fix an element.
    ``transforms`` maps a field to ``'linear'`` (default), ``'log'`` (positive
    quantities spanning decades, e.g. ``tcorr``) or ``'softplus'`` (positive
    widths).  Packed vectors hold the *transformed* free elements in the order of
    :attr:`names`.
    """

    def __init__(self, ranges: dict, transforms: Optional[dict] = None):
        self.ranges = {k: [tuple(map(float, r)) for r in v] for k, v in ranges.items()}
        self.transforms = dict(transforms or {})
        self.names: list[str] = []
        self.index: list[tuple[str, int]] = []
        for field, rng in self.ranges.items():
            for i, (lo, hi) in enumerate(rng):
                if hi > lo:
                    self.names.append(f'{field}[{i}]')
                    self.index.append((field, i))
        self.n_free = len(self.index)

    # ---- transforms ------------------------------------------------------
    def _fwd(self, field, v):
        t = self.transforms.get(field, 'linear')
        if t == 'linear':
            return v
        if t == 'log':
            return torch.log10(v)
        if t == 'softplus':
            return torch.log(torch.expm1(v))
        raise ValueError(f'unknown transform {t!r}')

    def _inv(self, field, u):
        t = self.transforms.get(field, 'linear')
        if t == 'linear':
            return u
        if t == 'log':
            return 10.0 ** u
        if t == 'softplus':
            return torch.nn.functional.softplus(u)
        raise ValueError(f'unknown transform {t!r}')

    # ---- pack / unpack -----------------------------------------------------
    def _field_tensor(self, sys: SpinSystem, field: str) -> torch.Tensor:
        v = getattr(sys, field)
        if v is None:
            raise ValueError(f'template SpinSystem has no {field!r}')
        if isinstance(v, (list, tuple)):
            return torch.stack([torch.as_tensor(e, dtype=_F64).reshape(()) for e in v])
        return torch.as_tensor(v, dtype=_F64)

    def pack(self, sys: SpinSystem) -> torch.Tensor:
        """Flat (transformed) vector of the free elements of ``sys``."""
        vals = []
        for field, i in self.index:
            vals.append(self._fwd(field, self._field_tensor(sys, field).reshape(-1)[i]))
        return torch.stack(vals) if vals else torch.zeros(0, dtype=_F64)

    def unpack(self, template: SpinSystem, vec: torch.Tensor) -> SpinSystem:
        """SpinSystem from ``template`` with the free elements replaced by ``vec``
        (kept on the autograd graph when ``vec`` requires grad)."""
        vec = torch.as_tensor(vec, dtype=_F64).reshape(-1)
        if vec.numel() != self.n_free:
            raise ValueError(f'expected {self.n_free} parameters, got {vec.numel()}')
        fields: dict[str, torch.Tensor] = {}
        k = 0
        for field, i in self.index:
            if field not in fields:
                fields[field] = self._field_tensor(template, field).clone()
            flat = fields[field].reshape(-1)
            # functional update keeps the graph: build the new flat tensor by masks
            one_hot = torch.zeros_like(flat); one_hot[i] = 1.0
            flat = flat * (1.0 - one_hot) + one_hot * self._inv(field, vec[k])
            fields[field] = flat.reshape(fields[field].shape)
            k += 1
        kwargs = _template_kwargs(template)
        on_graph = vec.requires_grad
        for field, t in fields.items():
            if field in ('lw', 'lwpp'):
                kwargs[field] = [t.reshape(-1)[j] if on_graph else float(t.reshape(-1)[j]) for j in range(t.numel())]
            elif field in ('tcorr', 'logtcorr'):
                kwargs[field] = t.reshape(-1)[0] if on_graph else float(t.reshape(-1)[0])
            else:
                kwargs[field] = t
        return SpinSystem(**kwargs)

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """``(n, n_free)`` packed vectors drawn uniformly from the ranges (in the
        untransformed space; ``'log'`` fields uniform in log10)."""
        rows = []
        for field, i in self.index:
            lo, hi = self.ranges[field][i]
            lo_t, hi_t = self._fwd(field, torch.tensor(lo, dtype=_F64)), self._fwd(field, torch.tensor(hi, dtype=_F64))
            u = torch.rand(n, dtype=_F64, generator=generator)
            rows.append(lo_t + (hi_t - lo_t) * u)
        return torch.stack(rows, dim=1) if rows else torch.zeros(n, 0, dtype=_F64)


_SYS_FIELDS = ('S', 'g', 'gFrame', 'Nucs', 'n', 'A', 'AFrame', 'Q', 'QFrame', 'D', 'DFrame', 'ee', 'eeFrame',
               'lw', 'lwpp', 'lwEndor', 'tcorr', 'logtcorr', 'HStrain', 'gStrain', 'AStrain', 'DStrain',
               'Potential', 'T1', 'T2', 'weight')


def _template_kwargs(template: SpinSystem) -> dict:
    """Constructor arguments reproducing ``template`` (fields torchspin parses)."""
    kw = {}
    for f in _SYS_FIELDS:
        v = getattr(template, f, None)
        if v is None:
            continue
        if f == 'Nucs':
            kw[f] = v if isinstance(v, str) else ','.join(v) if isinstance(v, (list, tuple)) else v
        elif f == 'n' and all(int(x) == 1 for x in v):
            continue
        elif torch.is_tensor(v):
            kw[f] = v.detach().clone()
        else:
            kw[f] = copy.deepcopy(v)
    return kw


# =============================================================================
# Simulation
# =============================================================================

def _resolve_simulator(simulator):
    if callable(simulator):
        return simulator
    name = str(simulator)
    if name == 'pepper':
        from torchspin.pepper_autograd import pepper_autograd
        return pepper_autograd
    if name == 'pepper_nograd':
        from torchspin.pepper import pepper
        return pepper
    if name == 'garlic':
        from torchspin.garlic import garlic
        return garlic
    if name == 'salt':
        from torchspin.salt import salt
        return salt
    if name == 'saffron':
        from torchspin.saffron import saffron
        return lambda s, e, o: saffron(s, e, o)[:2]
    if name == 'curry':
        from torchspin.curry import curry
        return curry
    raise ValueError(f'unknown simulator {simulator!r}')


def simulate(simulator, sys: SpinSystem, exp, opt=None, **kw):
    """``(x, y)`` from a simulator name or callable; ``y`` as a real float64 tensor."""
    fn = _resolve_simulator(simulator)
    out = fn(sys, exp, opt, **kw) if opt is not None else fn(sys, exp, **kw)
    x, y = out[0], out[1]
    y = torch.as_tensor(y)
    if torch.is_complex(y):
        y = y.real
    return x, y.to(_F64)


def _worker(args):
    simulator, template_kwargs, spec, exp, opt, vec, sim_kwargs = args
    torch.set_num_threads(1)
    template = SpinSystem(**template_kwargs)
    with torch.no_grad():
        sys = spec.unpack(template, torch.as_tensor(vec, dtype=_F64))
        x, y = simulate(simulator, sys, exp, opt, **sim_kwargs)
    return np.asarray(x), y.numpy()


def simulate_batch(simulator, template: SpinSystem, exp, opt, spec: ParamSpec, params: torch.Tensor, *,
                   grad: bool = False, workers: int = 0, **sim_kwargs):
    """Simulate every row of ``params`` (``(N, n_free)`` packed vectors).

    Returns ``(x, Y)`` with ``Y`` of shape ``(N, nPoints)``.  With ``grad=True``
    the rows stay on the autograd graph (``params`` must ``requires_grad``);
    with ``workers > 1`` (no-grad only) the rows are spread over a spawned
    process pool with one torch thread per worker.
    """
    params = torch.as_tensor(params, dtype=_F64)
    if params.ndim == 1:
        params = params.unsqueeze(0)
    n = params.shape[0]
    if workers and workers > 1 and not grad:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        tk = _template_kwargs(template)
        jobs = [(simulator, tk, spec, exp, opt, params[i].detach().numpy(), sim_kwargs) for i in range(n)]
        from torchspin.parallel import native_thread_env
        # cap the native BLAS/OpenMP pools of the spawned workers (one thread each,
        # matching the torch.set_num_threads(1) in _worker); uncapped, every worker
        # would start one thread per core
        with native_thread_env(1):
            pool = ctx.Pool(workers)
        with pool:
            res = pool.map(_worker, jobs)
        x = torch.as_tensor(res[0][0])
        return x, torch.stack([torch.as_tensor(r[1]) for r in res])
    ys, x = [], None
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i in range(n):
            sys = spec.unpack(template, params[i])
            x, y = simulate(simulator, sys, exp, opt, **sim_kwargs)
            ys.append(y)
    return torch.as_tensor(x), torch.stack(ys)


# =============================================================================
# Dataset
# =============================================================================

def resample(x_new: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of ``y(x)`` (uniform ``x``) at ``x_new``; differentiable in ``y``.
    ``y`` may be ``(..., nX)``."""
    x = torch.as_tensor(x, dtype=_F64); x_new = torch.as_tensor(x_new, dtype=_F64)
    n = x.numel()
    dx = float(x[1] - x[0])
    f = (x_new - x[0]) / dx
    lo = torch.clamp(torch.floor(f).long(), 0, n - 2)
    w = (f - lo.to(_F64)).clamp(0.0, 1.0)
    return y[..., lo] * (1.0 - w) + y[..., lo + 1] * w


def normalize(y: torch.Tensor, mode: Optional[str] = 'maxabs', x: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``'maxabs'`` (max |y| = 1), ``'area'`` (∫|y| dx = 1, needs ``x``), ``'zscore'`` or ``None``;
    along the last axis, differentiable."""
    if mode is None:
        return y
    if mode == 'maxabs':
        return y / y.abs().amax(dim=-1, keepdim=True).clamp(min=1e-300)
    if mode == 'area':
        dx = float(x[1] - x[0]) if x is not None else 1.0
        return y / (y.abs().sum(dim=-1, keepdim=True) * dx).clamp(min=1e-300)
    if mode == 'zscore':
        return (y - y.mean(dim=-1, keepdim=True)) / y.std(dim=-1, keepdim=True).clamp(min=1e-300)
    raise ValueError(f'unknown normalization {mode!r}')


class SpectrumDataset(torch.utils.data.IterableDataset):
    """On-the-fly ``(spectrum, packed parameters)`` pairs.

    Parameters are drawn with :meth:`ParamSpec.sample` (or ``sampler(n, generator)``
    when given), simulated with ``simulator`` under ``torch.no_grad``, resampled to
    ``x_out`` if given, normalized (``normalize``), and corrupted with Gaussian
    noise of standard deviation ``noise`` (relative to the normalized scale) plus
    an optional random polynomial baseline of degree ``baseline_degree`` and
    amplitude ``baseline``.  Deterministic for a given ``seed`` (per worker of a
    ``DataLoader``, the worker id is folded into the seed).  ``n`` items per
    epoch (``None`` = infinite).
    """

    def __init__(self, spec: ParamSpec, template: SpinSystem, exp, opt, *, simulator='pepper',
                 n: Optional[int] = 1000, seed: int = 0, normalize: Optional[str] = 'maxabs',
                 x_out: Optional[torch.Tensor] = None, noise: float = 0.0, baseline: float = 0.0,
                 baseline_degree: int = 1, sampler: Optional[Callable] = None, dtype=torch.float32,
                 sim_kwargs: Optional[dict] = None):
        super().__init__()
        self.spec, self.template, self.exp, self.opt = spec, template, exp, opt
        self.simulator, self.n, self.seed = simulator, n, seed
        self.normalize_mode, self.x_out, self.noise = normalize, x_out, noise
        self.baseline, self.baseline_degree, self.sampler, self.dtype = baseline, baseline_degree, sampler, dtype
        self.sim_kwargs = dict(sim_kwargs or {})

    def _generator(self) -> torch.Generator:
        info = torch.utils.data.get_worker_info()
        seed = self.seed + (info.id * 1_000_003 if info is not None else 0)
        return torch.Generator().manual_seed(seed)

    def make(self, vec: torch.Tensor, generator: Optional[torch.Generator] = None):
        """One sample: simulate ``vec``, post-process, return ``(spectrum, vec)``."""
        with torch.no_grad():
            sys = self.spec.unpack(self.template, vec)
            x, y = simulate(self.simulator, sys, self.exp, self.opt, **self.sim_kwargs)
            x = torch.as_tensor(x, dtype=_F64)
            if self.x_out is not None:
                y = resample(self.x_out, x, y); x = torch.as_tensor(self.x_out, dtype=_F64)
            y = normalize(y, self.normalize_mode, x)
            if self.noise > 0:
                y = y + self.noise * torch.randn(y.shape, dtype=_F64, generator=generator)
            if self.baseline > 0:
                t = torch.linspace(-1.0, 1.0, y.shape[-1], dtype=_F64)
                coef = self.baseline * (2.0 * torch.rand(self.baseline_degree + 1, dtype=_F64, generator=generator) - 1.0)
                y = y + sum(c * t ** k for k, c in enumerate(coef))
        return y.to(self.dtype), vec.to(self.dtype)

    def __iter__(self):
        gen = self._generator()
        info = torch.utils.data.get_worker_info()
        count = self.n if info is None or self.n is None else math.ceil(self.n / info.num_workers)
        i = 0
        while count is None or i < count:
            vec = (self.sampler(1, gen) if self.sampler is not None else self.spec.sample(1, gen))[0]
            yield self.make(vec, gen)
            i += 1


# =============================================================================
# Losses
# =============================================================================

def cosine_loss(y: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
    """``1 − cos(y, y_ref)`` along the last axis (shape-only, scale-free)."""
    num = (y * y_ref).sum(dim=-1)
    den = y.norm(dim=-1) * y_ref.norm(dim=-1)
    return 1.0 - num / den.clamp(min=1e-300)


def rmsd(y: torch.Tensor, y_ref: torch.Tensor, scale: bool = True) -> torch.Tensor:
    """Root-mean-square deviation along the last axis, after a least-squares
    amplitude fit of ``y`` to ``y_ref`` when ``scale`` is set."""
    if scale:
        a = (y * y_ref).sum(dim=-1, keepdim=True) / (y * y).sum(dim=-1, keepdim=True).clamp(min=1e-300)
        y = a * y
    return torch.sqrt(((y - y_ref) ** 2).mean(dim=-1))
