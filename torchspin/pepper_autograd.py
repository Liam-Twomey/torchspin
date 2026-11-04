"""Differentiable ``pepper``: pepper's forward path on the autograd graph.

``pepper_autograd(sys, exp, opt)`` computes the same rigid-limit powder spectrum as
:func:`torchspin.pepper.pepper` for the matrix-diagonalization path without strain,
using the *same* stages -- resonance-field search with transition tracking
(:func:`torchspin.resfields_batch.resfields_batch`), EasySpin grid interpolation,
SOPHE triangle/zone projection, and EasySpin's sampled-kernel convolution and
harmonic -- but with every stage after the search written in torch, so that
``torch.autograd`` propagates gradients from the spectrum to the spin-system
tensors (``sys.g``, ``sys.A``, ``sys.D``, ``sys.lw`` ...) that ``requires_grad``.

Gradients through the resonance-field search need no custom backward: the search
is pure torch and its final safeguarded Newton step at a converged root is the
implicit-function derivative dB_res/dθ = −(∂f/∂θ)/(∂f/∂B).

Differences to ``torchspin.autograd.differentiable_spectrum``'s broadband path (a
separate forward model that sums a Gaussian per orientation and transition): the
powder average here is smooth at the same GridSize as pepper, the forward output
equals pepper's to rounding, and the harmonic/line shape are EasySpin's.

Supported: field sweeps, ``Opt.Method='matrix'`` and ``'perturb*'`` (pepper's
perturbation route: single electron, no strain), every grid symmetry, ``GridSize`` int or ``[N, Ninterp]``,
``Harmonic`` 0-2, global ``lw``/``lwpp`` (Gaussian and/or Lorentzian, tensors
allowed), ``Temperature``, strains.  Strains (HStrain, gStrain, AStrain on the graph; DStrain magnitudes are
constants) use the summation branch.  ``ModAmp`` (pseudo-modulation, torch).  Not yet: ``mwPhase``, ordering,
photoselection, crystals, frequency sweeps (these raise ``NotImplementedError``).
"""
from __future__ import annotations

import math
from dataclasses import replace as _dc_replace
from functools import lru_cache
from typing import Optional

import numpy as np
import torch

from torchspin.experiment import Experiment, Options, auto_batch_size
from torchspin.ham import ham
from torchspin.hamsymm import hamsymm
from torchspin.resfields_batch import resfields_batch
from torchspin.sphgrid import sphgrid
from torchspin.spinsystem import SpinSystem
import importlib as _importlib
_pep = _importlib.import_module("torchspin.pepper")   # the module, not the exported function

__all__ = ['pepper_autograd', 'convspec_t', 'projecttriangles_t', 'projectzones_t']

_F64 = torch.float64


# =============================================================================
# Line shapes and EasySpin convolution in torch
# =============================================================================

def _gaussian_t(x: torch.Tensor, x0: float, fwhm, deriv: int) -> torch.Tensor:
    """Area-normalized Gaussian (absorption) and its derivatives, torch port of
    :func:`torchspin.lineshape.gaussian` (phase 0)."""
    fwhm = torch.as_tensor(fwhm, dtype=x.dtype, device=x.device)
    sig = fwhm / math.sqrt(8.0 * math.log(2.0))
    k = (x - x0) / (sig * math.sqrt(2.0))
    n = int(deriv)
    if n == 0:
        herm = torch.ones_like(k)
    elif n == 1:
        herm = 2.0 * k
    elif n == 2:
        herm = 4.0 * k * k - 2.0
    else:
        raise ValueError('deriv must be 0, 1 or 2')
    prefactor = math.sqrt(2.0 / math.pi) / (2.0 * sig)
    deriv_factor = (-1.0 / sig) ** n * 2.0 ** (-n / 2.0)
    return prefactor * deriv_factor * herm * torch.exp(-k * k)


def _lorentzian_t(x: torch.Tensor, x0: float, fwhm, deriv: int) -> torch.Tensor:
    """Area-normalized Lorentzian (absorption) and derivatives, torch port of
    :func:`torchspin.lineshape.lorentzian` (phase 0)."""
    fwhm = torch.as_tensor(fwhm, dtype=x.dtype, device=x.device)
    gamma = fwhm / math.sqrt(3.0)
    pre = 2.0 / (math.pi * math.sqrt(3.0))
    z = (x - x0) / gamma
    denom = 1.0 + (4.0 / 3.0) * z * z
    if deriv == 0:
        return pre / gamma / denom
    if deriv == 1:
        return -8.0 / 3.0 * pre / gamma ** 2 * z / denom ** 2
    if deriv == 2:
        return 8.0 / 3.0 * pre / gamma ** 3 * (4.0 * z * z - 1.0) / denom ** 3
    raise ValueError('deriv must be 0, 1 or 2')


def _es_convolve_t(spec: torch.Tensor, dx: float, fwhm, deriv: int, alpha: float) -> torch.Tensor:
    """Torch port of ``convspec._es_convolve`` (EasySpin ``convspec``, one line shape):
    the kernel is *sampled* on the 2N+1 extended grid in units of the increment,
    then applied by FFT.  ``fwhm`` may be a tensor that requires grad."""
    n = spec.shape[0]
    NN = 2 * n + 1
    mid = int(math.floor(NN / 2 + 0.5)) + 1
    x = torch.arange(1, NN + 1, dtype=_F64, device=spec.device)
    fw = torch.as_tensor(fwhm, dtype=_F64, device=spec.device) / dx
    line = torch.zeros(NN, dtype=_F64, device=spec.device)
    if alpha != 0:
        line = line + alpha * _gaussian_t(x, float(mid), fw, deriv)
    if alpha != 1:
        line = line + (1.0 - alpha) * _lorentzian_t(x, float(mid), fw, deriv)
    line_shift = torch.cat([line[mid - 1:], line[:mid - 1]])
    decay = torch.fft.ifft(line_shift.to(torch.complex128))
    out = torch.fft.fft(torch.fft.ifft(spec.to(torch.complex128), n=NN) * decay)
    out = (NN / dx ** deriv) * out
    return out[:n].real


def _pos(v) -> bool:
    return float(torch.as_tensor(v).detach()) > 0.0


def convspec_t(spec: torch.Tensor, dx: float, fwhm_g=0.0, fwhm_l=0.0, deriv: int = 0) -> torch.Tensor:
    """Differentiable :func:`torchspin.convspec.convspec` (phase 0): Gaussian then
    Lorentzian sampled kernels; with both widths zero the derivative is taken
    spectrally.  Same numbers as the numpy implementation to rounding."""
    from torchspin.convspec import _fft_convolve
    N = spec.shape[0]
    if N < 2:
        return spec
    spec = spec.to(_F64)
    if not _pos(fwhm_g) and not _pos(fwhm_l):
        return _fft_convolve(spec, N, dx, 0.0, 0.0, deriv) if deriv > 0 else spec
    deriv_left = deriv
    if _pos(fwhm_g):
        spec = _es_convolve_t(spec, dx, fwhm_g, deriv, 1.0)
        deriv_left = 0
    if _pos(fwhm_l):
        spec = _es_convolve_t(spec, dx, fwhm_l, deriv_left, 0.0)
    return spec


# =============================================================================
# SOPHE projection in torch (EasySpin projecttriangles.c / projectzones.c)
# =============================================================================

def _ramp_bins_t(n: int, base: torch.Tensor, a: torch.Tensor, b: torch.Tensor, delta: float, rising: bool) -> torch.Tensor:
    """Accumulate the linear ramps of the tent functions on ``[a, b]`` (fractional bin
    coordinates, ``a ≤ b``) into a length-``n`` spectrum.  Torch port of
    ``pepper._ramp_bins`` with ``f0 = base / ((b − a)·delta)``.

    Differentiable in ``base``, ``a`` and ``b``; the bin indices are detached.  A
    ramp of zero width (tied vertices, generic for axial systems where the
    resonance field does not depend on φ) contributes nothing to the value but
    its weight ``base·(b − a)/(2·delta)`` grows linearly with the width, so that
    term is kept on the graph instead of being masked away — otherwise the
    gradient with respect to parameters that lift the tie (A_x vs A_y) is wrong.
    """
    dev = base.device
    spec = torch.zeros(n, dtype=_F64, device=dev)
    width = b - a
    first = torch.trunc(a).detach().long()
    last = torch.trunc(b).detach().long()
    ok = (width.detach() >= 0.0) & (first < n) & (last >= 0)
    if not bool(ok.any()):
        return spec
    base, a, b, width, first, last = base[ok], a[ok], b[ok], width[ok], first[ok], last[ok]
    one = first == last
    if bool(one.any()):
        idx = first[one]
        inr = (idx >= 0) & (idx < n)
        # f0·width²/2 with f0 = base/(width·delta): linear in the width, defined at width = 0
        spec = spec.index_add(0, idx[inr], (base[one] * width[one] / (2.0 * delta))[inr])
    multi = ~one
    if not bool(multi.any()):
        return spec
    base, a, b, width, first, last = base[multi], a[multi], b[multi], width[multi], first[multi], last[multi]
    f0 = base / (width * delta)                      # width > 0 here (different bins)
    firstf = first.to(_F64); lastf = last.to(_F64)
    if rising:
        v_first = f0 * (firstf + 1.0 - a) ** 2 / 2.0
        v_last = f0 * ((lastf + b) / 2.0 - a) * (b - lastf)
    else:
        v_first = f0 * (firstf + 1.0 - a) * (b - (a + firstf + 1.0) / 2.0)
        v_last = f0 * (b - lastf) ** 2 / 2.0
    m = first >= 0
    spec = spec.index_add(0, first[m], v_first[m])
    m = last < n
    spec = spec.index_add(0, last[m], v_last[m])
    # interior bins lo .. hi-1: value f0·(k − (a − ½)) rising, f0·((b − ½) − k) falling
    lo = torch.clamp(first + 1, min=0)
    hi = torch.clamp(last, max=n)
    has_int = lo < hi
    if bool(has_int.any()):
        f0i, ai, bi, lo, hi = f0[has_int], a[has_int], b[has_int], lo[has_int], hi[has_int]
        if rising:
            s1, s0 = f0i, -f0i * (ai - 0.5)
        else:
            s1, s0 = -f0i, f0i * (bi - 0.5)
        S1 = torch.zeros(n + 1, dtype=_F64, device=dev).index_add(0, lo, s1).index_add(0, hi, -s1)
        S0 = torch.zeros(n + 1, dtype=_F64, device=dev).index_add(0, lo, s0).index_add(0, hi, -s0)
        c1 = torch.cumsum(S1, 0)[:n]; c0 = torch.cumsum(S0, 0)[:n]
        spec = spec + c1 * torch.arange(n, dtype=_F64, device=dev) + c0
    return spec


def projecttriangles_t(tri_idx: torch.Tensor, areas: torch.Tensor, fun: torch.Tensor, amp: torch.Tensor,
                       x0: float, delta: float, n: int) -> torch.Tensor:
    """Differentiable SOPHE triangle projection (EasySpin ``projecttriangles``).

    ``fun``/``amp``: ``(n_slots, n_knots)`` positions and amplitudes (NaN = no
    resonance); ``tri_idx`` ``(nTri, 3)`` knot indices; ``areas`` ``(nTri,)`` solid
    angles.  Returns the spectral density on the ``n``-point axis starting at
    ``x0`` with step ``delta``, summed over slots and triangles.  Every
    triangle's tent function is piecewise-linear in its three vertex positions
    and linear in the amplitudes, so gradients exist almost everywhere."""
    dev = fun.device
    P = fun[:, tri_idx].reshape(-1, 3)                       # (n_slots·nTri, 3)
    A = amp[:, tri_idx].reshape(-1, 3)
    area = areas.to(_F64).repeat(fun.shape[0])
    ok = ~torch.isnan(P.detach()).any(dim=1)
    spec = torch.zeros(n, dtype=_F64, device=dev)
    if not bool(ok.any()):
        return spec
    P, A, area = P[ok], A[ok], area[ok]
    P, _ = torch.sort(P, dim=1)
    amplitude = A.mean(dim=1)
    left_b = (P[:, 0] - x0) / delta
    middle_b = (P[:, 1] - x0) / delta
    right_b = (P[:, 2] - x0) / delta
    width = right_b - left_b
    w1 = middle_b - left_b
    w2 = right_b - middle_b
    zero = width.detach() == 0.0
    if bool(zero.any()):
        fi = torch.trunc(left_b[zero]).detach().long()
        inr = (fi >= 0) & (fi < n)
        spec = spec.index_add(0, fi[inr], (amplitude[zero] * area[zero] / delta)[inr])
    nz = ~zero
    if not bool(nz.any()):
        return spec
    amplitude, area, left_b, middle_b, right_b, width, w1, w2 = (
        amplitude[nz], area[nz], left_b[nz], middle_b[nz], right_b[nz], width[nz], w1[nz], w2[nz])
    base = 2.0 * amplitude * area / width
    spec = spec + _ramp_bins_t(n, base, left_b, middle_b, delta, rising=True)
    spec = spec + _ramp_bins_t(n, base, middle_b, right_b, delta, rising=False)
    return spec


def projectzones_t(pos: torch.Tensor, amp: torch.Tensor, seg_weights: torch.Tensor,
                   x0: float, delta: float, n: int) -> torch.Tensor:
    """Differentiable axial-grid projection (EasySpin ``projectzones``): each segment
    between consecutive knots spreads a uniform density over its field interval.
    ``pos``/``amp``: ``(n_slots, n_knots)``; ``seg_weights``: ``(n_knots-1,)``."""
    dev = pos.device
    nSeg = seg_weights.shape[0]
    p0, p1 = pos[:, :nSeg], pos[:, 1:nSeg + 1]
    left = torch.minimum(p0, p1).reshape(-1); right = torch.maximum(p0, p1).reshape(-1)
    meanAmp = ((amp[:, :nSeg] + amp[:, 1:nSeg + 1]) / 2.0 / delta).reshape(-1)
    sw = seg_weights.to(_F64).repeat(pos.shape[0])
    spec = torch.zeros(n, dtype=_F64, device=dev)
    ok = ~(torch.isnan(left.detach()) | torch.isnan(right.detach()))
    left, right, meanAmp, sw = left[ok], right[ok], meanAmp[ok], sw[ok]
    left_b = (left - x0) / delta; right_b = (right - x0) / delta
    first = torch.trunc(left_b).detach().long(); last = torch.trunc(right_b).detach().long()
    ok = (first < n) & (last >= 0)
    left_b, right_b, meanAmp, sw, first, last = left_b[ok], right_b[ok], meanAmp[ok], sw[ok], first[ok], last[ok]
    if left_b.numel() == 0:
        return spec
    one = first == last
    if bool(one.any()):
        idx = first[one]; inr = (idx >= 0) & (idx < n)
        spec = spec.index_add(0, idx[inr], (meanAmp[one] * sw[one])[inr])
    multi = ~one
    if not bool(multi.any()):
        return spec
    left_b, right_b, meanAmp, sw, first, last = left_b[multi], right_b[multi], meanAmp[multi], sw[multi], first[multi], last[multi]
    height = meanAmp * sw / (right_b - left_b)
    m = first >= 0
    spec = spec.index_add(0, first[m], (height * (first.to(_F64) + 1.0 - left_b))[m])
    m = last < n
    spec = spec.index_add(0, last[m], (height * (right_b - last.to(_F64)))[m])
    lo = torch.clamp(first + 1, min=0); hi = torch.clamp(last, max=n)
    has_int = lo < hi
    if bool(has_int.any()):
        S0 = torch.zeros(n + 1, dtype=_F64, device=dev).index_add(0, lo[has_int], height[has_int]).index_add(0, hi[has_int], -height[has_int])
        spec = spec + torch.cumsum(S0, 0)[:n]
    return spec


# =============================================================================
# Grid interpolation as linear maps (+ torch L3 for axial intensities)
# =============================================================================

def _numpy_interp(vals: np.ndarray, symmetry: str, N_c: int, N_f: int, mode: str, linear: bool) -> np.ndarray:
    """The numpy interpolation pepper applies to one slot (positions ``mode='pos'``
    with EasySpin G3, or values/linear)."""
    _, _, _, vecs_c = sphgrid(symmetry, N_c)
    _, _, _, vecs_f = sphgrid(symmetry, N_f)
    vecs_c = vecs_c if vecs_c.shape[0] == 3 else vecs_c.T
    vecs_f = vecs_f if vecs_f.shape[0] == 3 else vecs_f.T
    if linear or mode == 'val':
        return _pep._interp_grid(vals, symmetry, N_c, N_f, linear=True)
    if symmetry in _pep._CLOSEDPHI_1OCT or symmetry == 'Dinfh':
        return _pep._interp_sph(vecs_c, vecs_f, vals)
    return _pep._interp_grid(vals, symmetry, N_c, N_f)


@lru_cache(maxsize=32)
def _interp_matrix(symmetry: str, N_c: int, N_f: int, mode: str, linear: bool) -> np.ndarray:
    """``(n_fine, n_coarse)`` matrix of the (linear) interpolation scheme, built by
    interpolating the columns of the identity."""
    n_c = sphgrid(symmetry, N_c)[0].numel()
    n_f = sphgrid(symmetry, N_f)[0].numel()
    M = np.zeros((n_f, n_c))
    eye = np.eye(n_c)
    for j in range(n_c):
        M[:, j] = _numpy_interp(eye[:, j], symmetry, N_c, N_f, mode, linear)
    return M


def _interp_l3_t(y: torch.Tensor, N_f: int) -> torch.Tensor:
    """Torch port of ``pepper._interp_l3`` (EasySpin gridinterp 'L3', axial grids)
    for a batch of slots ``y`` ``(n_slots, n)``; NaNs propagate."""
    n = y.shape[1]
    factor = int(round((N_f - 1) / (n - 1)))
    x = torch.arange(factor, dtype=_F64, device=y.device) / factor
    X = torch.stack([x ** 3, x ** 2, x, torch.ones_like(x)], dim=1)          # (factor, 4)
    H = torch.tensor([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=_F64, device=y.device)
    d = y[:, 1:] - y[:, :-1]                                                 # (S, n-1)
    T = torch.zeros_like(y)
    if n > 2:
        d0, d1 = d[:, :n - 2], d[:, 1:n - 1]
        same = (torch.sign(d0.detach()) * torch.sign(d1.detach())) > 0
        # Replace the unselected differences (which may be NaN for slots missing
        # at some orientations) before any arithmetic: torch.where's backward
        # multiplies the unselected branch's local gradient by 0, and 0·NaN = NaN.
        one = torch.ones_like(d0)
        d0, d1 = torch.where(same, d0, one), torch.where(same, d1, one)
        dmax = torch.maximum(d0.abs(), d1.abs()); dmin = torch.minimum(d0.abs(), d1.abs())
        Tin = torch.where(same, 2 * dmin * dmax / (d0 + d1), torch.zeros_like(d0))
        T = torch.cat([torch.zeros_like(y[:, :1]), Tin, torch.zeros_like(y[:, :1])], dim=1)
    C = torch.stack([y[:, :-1], y[:, 1:], T[:, :-1], T[:, 1:]], dim=1)      # (S, 4, n-1)
    yii = torch.einsum('fk,kl,slm->sfm', X, H, C)                            # (S, factor, n-1)
    fine = yii.permute(0, 2, 1).reshape(y.shape[0], -1)                      # column-major flatten
    return torch.cat([fine, y[:, -1:]], dim=1)


def _interp_slots_t(vals: torch.Tensor, symmetry: str, N_c: int, N_f: int, mode: str, any_nan: bool) -> torch.Tensor:
    """Interpolate ``(n_slots, n_coarse)`` slot values to the fine grid exactly as
    ``pepper._interp_slot`` does per slot, on the autograd graph.  NaN patterns of
    the fine knots are taken from the numpy implementation (detached)."""
    dev = vals.device
    if symmetry == 'Dinfh' and (mode == 'val' or any_nan):
        return _interp_l3_t(vals, N_f)
    linear = (mode == 'val') or any_nan
    M = torch.as_tensor(_interp_matrix(symmetry, N_c, N_f, mode, linear), dtype=_F64, device=dev)
    nan_c = torch.isnan(vals.detach())
    if not bool(nan_c.any()):
        return vals @ M.T
    out = torch.nan_to_num(vals, nan=0.0) @ M.T
    vnp = vals.detach().cpu().numpy()
    mask = np.stack([np.isnan(_numpy_interp(vnp[s], symmetry, N_c, N_f, mode, linear)) for s in range(vals.shape[0])])
    return torch.where(torch.as_tensor(mask, device=dev), torch.full_like(out, float('nan')), out)


# =============================================================================
# Transition slots on the graph
# =============================================================================

def _slots_t(all_B: list, all_I: list, all_pairs: list, n_orient: int, threshold: float, all_W: Optional[list] = None):
    """Gather per-orientation resonances into ``(n_slots, n_orient)`` tensors of
    positions and intensities (NaN = no resonance), following
    ``pepper._transition_slots`` (slot = level pair + occurrence index) and
    ``pepper._filter_slots`` (EasySpin transition pre-selection)."""
    B_flat = torch.cat([b for b in all_B if b is not None and b.numel()])
    I_flat = torch.cat([i for i in all_I if i is not None and i.numel()])
    have_W = all_W is not None and any(w is not None and w.numel() for w in all_W)
    W_flat = torch.cat([w for w in all_W if w is not None and w.numel()]) if have_W else None
    dev = B_flat.device
    offsets = np.cumsum([0] + [(b.numel() if b is not None else 0) for b in all_B])
    slots: dict = {}
    s_idx, o_idx, f_idx = [], [], []
    for k in range(n_orient):
        B = all_B[k]
        if B is None or B.numel() == 0:
            continue
        Bn = B.detach().cpu().numpy()
        pn = all_pairs[k].detach().cpu().numpy() if all_pairs is not None and all_pairs[k] is not None else None
        if pn is None:
            for i in range(Bn.shape[0]):
                key = ('idx', i)
                slots.setdefault(key, len(slots))
                s_idx.append(slots[key]); o_idx.append(k); f_idx.append(offsets[k] + i)
            continue
        order = sorted(range(Bn.shape[0]), key=lambda i: (int(pn[i, 0]), int(pn[i, 1]), float(Bn[i])))
        occ: dict = {}
        for i in order:
            uv = (int(pn[i, 0]), int(pn[i, 1]))
            c = occ.get(uv, 0); occ[uv] = c + 1
            key = (uv[0], uv[1], c)
            slots.setdefault(key, len(slots))
            s_idx.append(slots[key]); o_idx.append(k); f_idx.append(offsets[k] + i)
    n_slots = len(slots)
    s_t = torch.tensor(s_idx, dtype=torch.long, device=dev)
    o_t = torch.tensor(o_idx, dtype=torch.long, device=dev)
    f_t = torch.tensor(f_idx, dtype=torch.long, device=dev)
    nan = torch.full((n_slots, n_orient), float('nan'), dtype=_F64, device=dev)
    B_s = nan.index_put((s_t, o_t), B_flat.to(_F64)[f_t])
    I_s = nan.index_put((s_t, o_t), I_flat.to(_F64)[f_t])
    W_s = nan.index_put((s_t, o_t), W_flat.to(_F64)[f_t]) if have_W else None
    if threshold > 0 and n_slots:
        with torch.no_grad():
            smax = torch.nan_to_num(I_s.abs(), nan=0.0).max(dim=1).values
            keep = smax >= threshold * smax.max()
        B_s, I_s = B_s[keep], I_s[keep]
        W_s = W_s[keep] if W_s is not None else None
    if all_W is not None:
        return B_s, I_s, W_s
    return B_s, I_s


# =============================================================================
# Strain accumulation: bin-integrated Gaussians (EasySpin lisum1i, Gaussian template)
# =============================================================================

def gaussian_bins_t(x: torch.Tensor, pos: torch.Tensor, fwhm: torch.Tensor, amp: torch.Tensor,
                    min_fwhm: float, chunk: int = 256) -> torch.Tensor:
    """Sum of bin-integrated Gaussians, torch port of ``pepper._gaussian_bins``:
    each line contributes ``amp × [Φ(upper bin edge) − Φ(lower bin edge)]`` (its
    sum over the axis equals ``amp``); widths below ``min_fwhm`` are clipped
    (lisum1i).  Differentiable in ``pos``, ``fwhm`` and ``amp``; the lines are
    evaluated densely in chunks (the ±7σ√2 window of the NumPy version only
    skips bins that are zero to double precision)."""
    n = x.shape[0]
    dev = pos.device
    if pos.numel() == 0:
        return torch.zeros(n, dtype=_F64, device=dev)
    dx = float(x[1] - x[0])
    xr = x.to(dev)
    sig = torch.clamp(fwhm, min=min_fwhm) / math.sqrt(8.0 * math.log(2.0))
    sg = sig * math.sqrt(2.0)
    out = torch.zeros(n, dtype=_F64, device=dev)
    for k0 in range(0, pos.numel(), chunk):
        p = pos[k0:k0 + chunk, None]; g = sg[k0:k0 + chunk, None]; a = amp[k0:k0 + chunk, None]
        hi = torch.erf((xr[None, :] + 0.5 * dx - p) / g)
        lo = torch.erf((xr[None, :] - 0.5 * dx - p) / g)
        out = out + (0.5 * a * (hi - lo)).sum(dim=0)
    return out


# =============================================================================
# Main
# =============================================================================

def _check_supported(sys: SpinSystem, exp: Experiment, opt: Options) -> None:
    def _nz(t):
        return t is not None and bool(torch.any(torch.as_tensor(t) != 0))
    if getattr(exp, 'Range', None) is None or getattr(exp, 'mwFreq', None) is None:
        raise NotImplementedError('pepper_autograd: field sweeps only (Experiment.mwFreq and Range).')
    if getattr(exp, 'Field', None) is not None and getattr(exp, 'mwRange', None) is not None:
        raise NotImplementedError('pepper_autograd: frequency sweeps are not supported.')
    mod_amp = getattr(exp, 'ModAmp', 0.0)
    if mod_amp is not None and float(torch.as_tensor(mod_amp).detach()) > 0 and int(exp.Harmonic) < 1:
        raise ValueError('With field modulation (Exp.ModAmp), Exp.Harmonic=0 does not work.')
    if float(getattr(exp, 'mwPhase', 0.0) or 0.0) != 0.0:
        raise NotImplementedError('pepper_autograd: Exp.mwPhase (dispersion) is not supported.')
    for name in ('Ordering', 'lightBeam', 'SampleFrame', 'CrystalSymmetry', 'MolFrame'):
        v = getattr(exp, name, None)
        if v is not None and v != '' and not (isinstance(v, (list, tuple)) and len(v) == 0):
            raise NotImplementedError(f'pepper_autograd: Experiment.{name} is not supported.')
    if getattr(sys, 'initState', None) is not None:
        raise NotImplementedError('pepper_autograd: Sys.initState is not supported.')
    # Without this the matrix branch would silently run below, giving the exact
    # spectrum of the whole system instead of the hybrid one — a different
    # forward model from pepper(Method='hybrid'), not just a missing gradient.
    if str(getattr(opt, 'Method', 'matrix')) == 'hybrid':
        raise NotImplementedError("pepper_autograd: Options.Method='hybrid' is not "
                                  "differentiable yet; use 'matrix' or 'perturb'.")
    if any(v > 1 for v in (getattr(sys, 'n', None) or [])):
        raise NotImplementedError('pepper_autograd: sets of equivalent nuclei '
                                  '(SpinSystem.n > 1) are not supported; list each '
                                  'nucleus separately in Nucs.')
    if int(exp.Harmonic) not in (0, 1, 2):
        raise ValueError('Harmonic must be 0, 1 or 2.')


def pepper_autograd(sys_or_list, exp: Experiment, opt: Optional[Options] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Rigid-limit powder CW EPR spectrum, identical to :func:`torchspin.pepper.pepper`
    (matrix method, no strain) and differentiable with respect to every spin-system
    tensor that ``requires_grad`` (``g``, ``A``, ``D``, ``lw``/``lwpp`` ...).

    Accepts a single :class:`SpinSystem` or a list of components. As in
    ``pepper``, every component is expanded into its isotopologues (natural
    abundance for element names such as ``'Zr'``, or ``Sys.Abund``) and the
    spectra are summed with weight ``Sys.weight`` × abundance. The abundances
    are constants, so the sum stays on the graph. :func:`torchspin.pepper.pepper`
    calls this function itself when any input requires grad.

    Returns ``(B, spec)``: field axis (mT) and spectrum on the autograd graph.
    """
    from torchspin.isotopologues import expand_components
    opt = opt if opt is not None else Options()
    if str(getattr(opt, 'separate', '') or ''):
        raise NotImplementedError("pepper_autograd: Options.separate is not supported.")
    components = expand_components(sys_or_list, getattr(opt, 'IsoCutoff', 1e-4))
    for c in components:
        _check_supported(c, exp, opt)
    x, total = None, None
    for c in components:
        x, spec = _pepper_autograd_single(c, exp, opt)
        w = getattr(c, 'weight', 1.0)
        spec = spec * (w if torch.is_tensor(w) else float(w))
        total = spec if total is None else total + spec
    return x, total


def _pepper_autograd_single(sys: SpinSystem, exp: Experiment, opt: Options) -> tuple[torch.Tensor, torch.Tensor]:
    """pepper_autograd for one isotopologue (no weight applied)."""
    device = torch.device(opt.device)
    nPoints = int(exp.nPoints)
    B_lo, B_hi = float(exp.Range[0]), float(exp.Range[1])
    x = torch.linspace(B_lo, B_hi, nPoints, dtype=_F64)
    dx = (B_hi - B_lo) / (nPoints - 1)

    # Hamiltonian operators (on the graph when sys tensors require grad)
    H0, mux, muy, muz = ham(sys)
    H0, mux, muy, muz = H0.to(device), mux.to(device), muy.to(device), muz.to(device)

    # Grid symmetry and knots (discrete; detached)
    grid_sym = opt.GridSymmetry
    R_sym2mol = torch.eye(3, dtype=_F64)
    if grid_sym in ('auto', '', None):
        grid_sym, R_sym2mol = hamsymm(sys)
    N_coarse, N_interp = opt.grid_size_coarse, opt.grid_size_interp
    nfKnots = (N_coarse - 1) * N_interp + 1 if N_interp > 1 else N_coarse
    phi_arr, theta_arr, _, _ = sphgrid(grid_sym, N_coarse)
    theta_sym = theta_arr.clone()
    if not torch.allclose(R_sym2mol, torch.eye(3, dtype=_F64), atol=1e-10):
        sin_t = torch.sin(theta_arr)
        v_sym = torch.stack([sin_t * torch.cos(phi_arr), sin_t * torch.sin(phi_arr), torch.cos(theta_arr)], dim=1)
        v_mol = (R_sym2mol.T @ v_sym.T).T
        theta_arr = torch.acos(v_mol[:, 2].clamp(-1.0, 1.0))
        phi_arr = torch.atan2(v_mol[:, 1], v_mol[:, 0])
    phi_arr, theta_arr = phi_arr.to(device), theta_arr.to(device)
    n_orient = phi_arr.shape[0]

    # Resonances: pepper's perturbation route (single electron, no strain, no
    # ZFS+Boltzmann) when Opt.Method='perturb*', otherwise the matrix search with
    # pepper's transition pre-selection (detached) and Threshold 0 in the search
    def _nz(t):
        return t is not None and bool(torch.any(torch.as_tensor(t).detach() != 0))
    strain_present = any(_nz(getattr(sys, k, None)) for k in ('HStrain', 'gStrain', 'AStrain', 'DStrain'))
    needs_exact_boltz = (float(sys.S[0]) > 0.5 and sys.D is not None)
    use_perturb = (str(opt.Method).startswith('perturb') and sys.nElectrons == 1
                   and not strain_present and not needs_exact_boltz)
    all_B, all_I, all_P, all_Wl = [None] * n_orient, [None] * n_orient, [None] * n_orient, [None] * n_orient
    if use_perturb:
        from torchspin.resfields_perturb import resfields_perturb_batch
        _w = B_hi - B_lo
        exp_search = _dc_replace(exp, Range=[max(0.0, B_lo - 0.2 * _w), B_hi + 0.2 * _w], CenterSweep=None)
        B_list, I_list = resfields_perturb_batch(sys, phi_arr, theta_arr, exp_search, opt, return_full=True)
        for j in range(n_orient):
            all_B[j], all_I[j] = B_list[j], I_list[j]          # index-matched slots, no level pairs
    else:
        with torch.no_grad():
            pairs = _pep._preselect_pairs(H0.detach().cpu(), mux.detach().cpu(), muy.detach().cpu(), muz.detach().cpu(),
                                          sys, exp, opt, 0.5 * (B_lo + B_hi))
        opt_rf = _dc_replace(opt, Threshold=0.0)
        batch = auto_batch_size(opt.BatchSize, H0.shape[0], n_orient)
        for s in range(0, n_orient, batch):
            e = min(s + batch, n_orient)
            B_list, I_list, W_list, P_list = resfields_batch(H0, mux, muy, muz, phi_arr[s:e], theta_arr[s:e], exp, opt_rf,
                                                             sys=sys, return_pairs=True, pairs=pairs)
            for j in range(e - s):
                all_B[s + j], all_I[s + j], all_P[s + j], all_Wl[s + j] = B_list[j], I_list[j], P_list[j], W_list[j]
    if all(b is None or b.numel() == 0 for b in all_B):
        return x, torch.zeros(nPoints, dtype=_F64)
    has_strain = any(w is not None and w.numel() > 0 for w in all_Wl)
    if has_strain:
        B_c, I_c, W_c = _slots_t(all_B, all_I, all_P, n_orient, float(opt.Threshold), all_W=all_Wl)
    else:
        B_c, I_c = _slots_t(all_B, all_I, all_P, n_orient, float(opt.Threshold))
        W_c = None
    if opt.Verbosity >= 1:
        print(f'pepper_autograd: {n_orient} orientations, {B_c.shape[0]} transition slots, grid {grid_sym}')

    # Interpolation to the fine grid
    do_interp = N_interp > 1 and n_orient > 1
    if do_interp:
        any_nan = bool(torch.isnan(B_c.detach()).any())
        B_f = _interp_slots_t(B_c, grid_sym, N_coarse, nfKnots, 'pos', any_nan)
        I_f = _interp_slots_t(I_c, grid_sym, N_coarse, nfKnots, 'val', any_nan)
        W_f = _interp_slots_t(W_c, grid_sym, N_coarse, nfKnots, 'val', any_nan) if W_c is not None else None
        _, theta_fine, _, _ = sphgrid(grid_sym, nfKnots)
        theta_final, N_final = theta_fine, nfKnots
    else:
        B_f, I_f, W_f, theta_final, N_final = B_c, I_c, W_c, theta_sym, N_coarse

    tri, facet_w = _pep._facets(grid_sym, N_final, theta_final)
    facet_w_t = torch.as_tensor(np.asarray(facet_w, dtype=float), dtype=_F64, device=device)
    if not has_strain:
        # Projection (density) and χ integral
        if tri is None:
            dens = projectzones_t(B_f, I_f, facet_w_t, B_lo, dx, nPoints)
        else:
            dens = projecttriangles_t(torch.as_tensor(np.asarray(tri), dtype=torch.long, device=device), facet_w_t,
                                      B_f, I_f, B_lo, dx, nPoints)
        spec = dens * (2.0 * math.pi)
    else:
        # Strain: per-facet Gaussian lines (EasySpin pepper.m summation branch):
        # facet center position, mean intensity × solid angle, mean width smoothed
        # by the facet's field spread (Opt.Smoothing = 2), accumulated as
        # bin-integrated Gaussians (lisum1i).
        if tri is None:
            fPosC = 0.5 * (B_f[:, :-1] + B_f[:, 1:])
            fSpread = (B_f[:, 1:] - B_f[:, :-1]).abs()
            fIntC = facet_w_t.unsqueeze(0) * 0.5 * (I_f[:, :-1] + I_f[:, 1:])
            fWidM = 0.5 * (W_f[:, :-1] + W_f[:, 1:])
            c1, c2 = 1.57246, 18.6348
        else:
            tri_t = torch.as_tensor(np.asarray(tri), dtype=torch.long, device=device)
            Bv = B_f[:, tri_t]                                   # (S, nTri, 3)
            fPosC = Bv.mean(dim=2)
            fSpread = Bv.max(dim=2).values - Bv.min(dim=2).values
            fIntC = facet_w_t.unsqueeze(0) * I_f[:, tri_t].mean(dim=2)
            fWidM = W_f[:, tri_t].mean(dim=2)
            c1, c2 = 2.8269, 42.6843
        fPosC, fSpread, fIntC, fWidM = fPosC.reshape(-1), fSpread.reshape(-1), fIntC.reshape(-1), fWidM.reshape(-1)
        with torch.no_grad():
            ok = torch.isfinite(fPosC) & torch.isfinite(fIntC) & torch.isfinite(fWidM) & (fIntC > 0)
        fPosC, fSpread, fIntC, fWidM = fPosC[ok], fSpread[ok], fIntC[ok], fWidM[ok]
        pos_spread = fSpread.detach() > 0
        Lambda = torch.where(pos_spread, fWidM / torch.where(pos_spread, fSpread, torch.ones_like(fSpread)),
                             torch.full_like(fSpread, float('inf')))
        gam = torch.where(pos_spread, 1.0 / torch.sqrt(c1 * Lambda ** 2 + c2 * Lambda ** 4), torch.zeros_like(Lambda))
        fWidC = fWidM * (1.0 + 2.0 * gam)                        # Opt.Smoothing = 2
        spec = gaussian_bins_t(x, fPosC, fWidC, fIntC, dx / 100.0)
        spec = spec / dx * (2.0 * math.pi)

    # Global line width and harmonic (EasySpin convspec; with strain the harmonic
    # of an unbroadened spectrum is taken by finite differences, as pepper does)
    lw = sys.get_lw()
    fwhm_g, fwhm_l = lw[0], lw[1]
    mod_amp = getattr(exp, 'ModAmp', 0.0)
    use_mod = mod_amp is not None and float(torch.as_tensor(mod_amp).detach()) > 0
    H = 0 if use_mod else int(exp.Harmonic)       # with modulation: convolve the absorption
    if has_strain and not _pos(fwhm_g) and not _pos(fwhm_l):
        for _ in range(H):
            spec = _pep._central_difference(spec, dx)
    else:
        spec = convspec_t(spec, dx, fwhm_g, fwhm_l, deriv=H)
    if use_mod:
        from torchspin.dataproc import fieldmod_t
        spec = fieldmod_t(x, spec, mod_amp, int(exp.Harmonic))   # pseudo-modulation (Exp.ModAmp)
    return x, spec.cpu() if spec.device.type != 'cpu' else spec
