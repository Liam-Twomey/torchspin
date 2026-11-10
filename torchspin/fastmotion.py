"""fastmotion — Kivelson/Freed fast-motion Lorentzian linewidths for CW EPR.

Port of EasySpin's ``fastmotion.m``.

Computes FWHM Lorentzian linewidths in the fast-motion regime using the
perturbation theory formulae of Freed, Bruno & Polnaszek (1971) / Atherton
(1993), pp. 331–332:

    FWHM(mI) = A + B·mI + C·mI² + Σ D_ij · mI_i · mI_j  (single/multi nucleus)

where A, B, C, D are combinations of spectral densities j₀, j₁ and the
anisotropy of the g and A tensors.

Example
-------
>>> from torchspin import SpinSystem
>>> from torchspin.fastmotion import fastmotion
>>> sys = SpinSystem(S=[0.5], g=[[2.0088, 2.0064, 2.0027]], Nucs='14N',
...                 A=[[20.0, 20.0, 85.0]])
>>> lw, mI_all = fastmotion(sys, field_mT=350.0, tcorr=1e-10, domain='field')
>>> lw.shape  # one width per EPR line
(3,)
"""
from __future__ import annotations

import itertools
import math
from typing import Optional, Tuple

import numpy as np
import torch

from torchspin.constants import BMAGN, PLANCK


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fastmotion(
    sys,
    field_mT: float,
    tcorr: float,
    domain: str = "field",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-line Lorentzian FWHM linewidths for fast-motion CW EPR.

    NumPy front end of :func:`fastmotion_t` (same algorithm in torch, on the
    autograd graph when ``sys`` carries tensors that require grad).

    Parameters
    ----------
    sys : SpinSystem
        Spin system.  Must have ``S = [0.5]``, and at least one anisotropic
        tensor (g or A).  If ``sys.A`` is given, it should contain principal
        values in MHz.  If ``sys.Q`` is given (quadrupole), it should be in
        MHz.
    field_mT : float
        Static magnetic field [mT] used to evaluate the Larmor angular
        frequency ω₀ = g₀ μ_B B / ℏ.
    tcorr : float
        Isotropic rotational correlation time [seconds].
    domain : str
        ``'field'`` (default) → return FWHM linewidths in **mT**;
        ``'freq'`` → MHz.

    Returns
    -------
    lw : ndarray, shape ``(nLines,)``
    mI_all : ndarray, shape ``(nLines, nNuclei)``
    """
    lw, mI_all = fastmotion_t(sys, field_mT, tcorr, domain)
    return lw.detach().cpu().numpy(), mI_all


def fastmotion_t(sys, field_mT, tcorr, domain: str = "field") -> Tuple[torch.Tensor, np.ndarray]:
    """Kivelson–Freed fast-motion line widths (EasySpin ``fastmotion.m``) in torch.

    ``field_mT`` and ``tcorr`` may be floats or scalar tensors; g, A and Q are
    taken from ``sys`` as tensors.  Returns ``(lw, mI_all)`` with ``lw`` a
    ``(nLines,)`` tensor (mT for ``domain='field'``, MHz for ``'freq'``) and the
    mI combinations as a NumPy array (constants).
    """
    from torchspin.spinsystem import _reject_h0extra
    _reject_h0extra(sys, 'fastmotion')
    if domain not in ("field", "freq"):
        raise ValueError(f"domain must be 'field' or 'freq', got {domain!r}")
    dt = torch.float64
    hbar = PLANCK / (2.0 * math.pi)
    g_flat = torch.as_tensor(sys.g, dtype=dt).reshape(-1)
    eye3 = torch.eye(3, dtype=dt)
    if g_flat.numel() == 1:
        g_mat = eye3 * g_flat[0]
    elif g_flat.numel() == 3:
        g_mat = torch.diag(g_flat)
    elif g_flat.numel() == 9:
        g_mat = g_flat.reshape(3, 3)
    else:
        raise ValueError(f"Invalid g shape: {sys.g.shape}")
    g0 = torch.trace(g_mat) / 3.0
    g1 = (g_mat - eye3 * g0).reshape(9)
    B_T = torch.as_tensor(field_mT, dtype=dt) * 1e-3
    tcorr = torch.as_tensor(tcorr, dtype=dt)
    cc = BMAGN * B_T / hbar
    omega0 = g0 * cc
    j0 = tcorr
    j1 = tcorr / (1.0 + omega0 ** 2 * tcorr ** 2)
    nNuclei = sys.nNuclei
    I_arr = np.array(sys.I, dtype=float) if nNuclei > 0 else np.array([], dtype=float)
    if nNuclei > 0:
        A_t = torch.as_tensor(sys.A, dtype=dt)
        cols = []
        for iNuc in range(nNuclei):
            A_nuc = torch.diag(A_t[iNuc, :3]) if A_t.shape[0] == nNuclei else A_t[3 * iNuc: 3 * (iNuc + 1), :3]
            A0_nuc = torch.trace(A_nuc) / 3.0
            cols.append(((A_nuc - eye3 * A0_nuc) * 1e6 * (2.0 * math.pi)).reshape(9))
        A1_mat = torch.stack(cols, dim=1)          # (9, nNuclei)
        AA = A1_mat.T @ A1_mat                       # (nNuclei, nNuclei)
        gA = cc * (g1 @ A1_mat)                      # (nNuclei,)
    else:
        AA = torch.zeros(0, 0, dtype=dt)
        gA = torch.zeros(0, dtype=dt)
    gg = cc ** 2 * (g1 @ g1)
    aniso_g = bool(gg.detach().abs() > 0.0)
    aniso_A = nNuclei > 0 and bool(torch.any(AA.detach() != 0.0))
    has_Q = nNuclei > 0 and getattr(sys, "Q", None) is not None
    if not aniso_g and not aniso_A and not has_Q:
        raise ValueError("Either g or A must be anisotropic, or Q must be present.")
    if nNuclei > 0:
        II = torch.as_tensor(I_arr * (I_arr + 1), dtype=dt)
        A_coeff = gg * (2 / 15 * j0 + 1 / 10 * j1) + (II * torch.diagonal(AA)).sum() * (1 / 20 * j0 + 7 / 60 * j1)
        B_coeff = gA * (4 / 15 * j0 + 1 / 5 * j1)
        C_coeff = torch.diagonal(AA) * (1 / 12 * j0 - 1 / 60 * j1)
        D_coeff = AA * (4 / 15 * j0 + 1 / 10 * j1)
        D_coeff = D_coeff - torch.tril(D_coeff)
    else:
        A_coeff = gg * (2 / 15 * j0 + 1 / 10 * j1)
        B_coeff = torch.zeros(0, dtype=dt); C_coeff = torch.zeros(0, dtype=dt); D_coeff = torch.zeros(0, 0, dtype=dt)
    to_mhz = 1.0 / (math.pi * 1e6)
    A_MHz, B_MHz, C_MHz, D_MHz = A_coeff * to_mhz, B_coeff * to_mhz, C_coeff * to_mhz, D_coeff * to_mhz
    if nNuclei > 0 and has_Q:
        Q_t = torch.as_tensor(sys.Q, dtype=dt)
        if Q_t.ndim == 1:
            Q_t = Q_t.reshape(nNuclei, -1)
        PP = (Q_t ** 2).sum(dim=1) * (1e6 * 2.0 * math.pi) ** 2
        II2 = torch.as_tensor(I_arr * (I_arr + 1), dtype=dt)
        I_t = torch.as_tensor(I_arr, dtype=dt)
        qA_MHz = (j0 / 20 * PP * II2 * (II2 - 1)).sum() * to_mhz
        qC_MHz = j0 / 20 * PP * I_t * (I_t + 1) * 2.0 * to_mhz
        qE_MHz = j0 / 20 * PP * (-3.0) * to_mhz
    else:
        qA_MHz = torch.zeros((), dtype=dt); qC_MHz = torch.zeros(nNuclei, dtype=dt); qE_MHz = torch.zeros(nNuclei, dtype=dt)
    mI_all = _all_mI(I_arr)
    mI_t = torch.as_tensor(mI_all, dtype=dt)                      # (nLines, nNuclei)
    lw = A_MHz.expand(mI_t.shape[0]).clone()
    if nNuclei > 0:
        lw = (lw + mI_t @ B_MHz + (mI_t ** 2) @ C_MHz
              + torch.einsum('li,ij,lj->l', mI_t, D_MHz, mI_t)
              + qA_MHz + (mI_t ** 2) @ qC_MHz + (mI_t ** 4) @ qE_MHz)
    if domain == "field":
        lw = lw * (1e6 * PLANCK / (g0 * BMAGN) * 1e3)
    return lw, mI_all


def _all_mI(I_arr: np.ndarray) -> np.ndarray:
    """Return all mI combinations as rows, ordered like ``resfields_perturb``.

    For each nucleus with spin Iₖ, mI ranges over −Iₖ, −Iₖ+1, …, +Iₖ.
    The Cartesian product of all nuclei is taken in that order.

    Parameters
    ----------
    I_arr : ndarray, shape ``(nNuclei,)``
        Nuclear spin quantum numbers.

    Returns
    -------
    ndarray, shape ``(nLines, nNuclei)`` where ``nLines = Π(2Iₖ+1)``.
    """
    if len(I_arr) == 0:
        return np.zeros((1, 0))
    mI_ranges = [np.arange(-I, I + 1) for I in I_arr]
    combos = list(itertools.product(*mI_ranges))
    return np.array(combos, dtype=float)
