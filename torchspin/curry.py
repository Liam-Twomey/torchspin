"""Magnetic susceptibility and magnetization calculator.

Port of EasySpin's ``curry.m`` (core physics, powder average).

Computes the thermal expectation value of the magnetic moment ⟨μ_z⟩ and
the static magnetic susceptibility χ = d⟨μ_z⟩/dB for a powder sample.

Physics
-------
For each orientation (φ, θ) with weight w:

    H(B) = H0 - B · μ_zL          [MHz]  (B in mT, μ in MHz/mT)

    Boltzmann: ρ_k = exp(-(E_k - E_0) · β)   β = h·10⁶ / (k_B·T)  [1/MHz]

    ⟨μ_z⟩ [MHz/mT] = Σ_k ρ_k <k|μ_zL|k> / Σ ρ_k

χ per molecule [J/T²] = d⟨μ_z⟩/dB via forward difference with step δB.

χ_mol [m³/mol] = χ_per_molecule · μ₀ · N_A

μ_BM [dimensionless] = ⟨μ_z⟩[MHz/mT] · h · 10⁹ / μ_B

Example
-------
>>> import torch

>>> from torchspin import SpinSystem
>>> from torchspin.curry import curry
>>> sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
>>> B = torch.tensor([0.0, 340.0])         # mT
>>> T = torch.tensor([1.0, 10.0, 100.0, 300.0])  # K
>>> mu_BM, chi_mol = curry(sys, B, T)
>>> # chi_mol at 300 K should be ~1.57e-8 m³/mol (Curie law)
"""
from __future__ import annotations

import math
from typing import Union

import torch
from torchspin._linalg import eigh as _eigh

from torchspin.constants import BMAGN, BOLTZMANN, PLANCK
from torchspin.ham import ham
from torchspin.sphgrid import sphgrid
from torchspin.spinsystem import SpinSystem

# Physical constants not yet in torchspin/constants.py
_MU0 = 4.0 * math.pi * 1e-7   # T·m/A  (vacuum permeability)
_N_A = 6.02214076e23           # mol⁻¹  (Avogadro)

# Unit conversion: MHz/mT → J/T
# 1 MHz/mT = 1e6 Hz / 1e-3 T = 1e9 Hz/T = 1e9 * h J/T
_MHZ_PER_MT_TO_J_PER_T = 1e9 * PLANCK   # J/T per (MHz/mT)


def curry(
    sys: SpinSystem,
    B_fields: Union[torch.Tensor, list, float],
    temperatures: Union[torch.Tensor, list, float],
    *,
    grid_size: int = 10,
    delta_B: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute magnetization and molar susceptibility vs field and temperature.

    Parameters
    ----------
    sys:
        Spin system.
    B_fields:
        Magnetic field values (mT).  Scalar, list, or 1-D tensor.
    temperatures:
        Temperature values (K).  Scalar, list, or 1-D tensor.
        Must be positive (> 0 K).
    grid_size:
        SOPHE grid parameter for powder averaging (default 10).
        Larger values give a finer angular grid and more accurate results.
    delta_B:
        Field step (mT) used for the numerical derivative to compute
        susceptibility (default 1.0 mT).

    Returns
    -------
    mu_BM:
        Magnetic moment along the field direction, in Bohr magnetons.
        Shape ``(nB, nT)``.
    chi_mol:
        Molar magnetic susceptibility (SI), in m³/mol.
        Shape ``(nB, nT)``.

    Notes
    -----
    The susceptibility is the *parallel* component χ_∥ = ∂⟨μ_z⟩/∂H, powder-
    averaged.  This matches the quantity accessible in a DC SQUID or VSM
    experiment.

    Unit conventions:
        * 1 cm³/mol (CGS-Gaussian, "emu/mol") = 4π × 10⁻⁶ m³/mol (SI)
        * To convert to cm³/mol: multiply by 10⁶ / (4π) ≈ 79 577.5

    Curie law check (S=1/2, g=2.0, T=300 K):
        χ_mol ≈ 1.57 × 10⁻⁸ m³/mol  = 1.25 × 10⁻³ cm³/mol  (χT = 0.375 cm³·K/mol)
    """
    # -----------------------------------------------------------------------
    # Parse inputs
    # -----------------------------------------------------------------------
    B_arr = _to_1d_float64(B_fields)
    T_arr = _to_1d_float64(temperatures)
    nB = B_arr.numel()
    nT = T_arr.numel()

    if (T_arr <= 0).any():
        raise ValueError("All temperatures must be positive (> 0 K).")
    if delta_B <= 0:
        raise ValueError("delta_B must be positive.")

    # -----------------------------------------------------------------------
    # Build field-independent Hamiltonian and moment operators
    # -----------------------------------------------------------------------
    H0, mux, muy, muz = ham(sys, B0=None)

    # -----------------------------------------------------------------------
    # Powder grid (Ci symmetry = upper hemisphere, appropriate for magnetics)
    # -----------------------------------------------------------------------
    phi_arr, theta_arr, weights_arr, _ = sphgrid("Ci", grid_size)
    nOri = phi_arr.shape[0]
    total_weight = weights_arr.sum().item()

    # -----------------------------------------------------------------------
    # Allocate output
    # -----------------------------------------------------------------------
    mu_BM_out = torch.zeros(nB, nT, dtype=torch.float64)
    chi_mol_out = torch.zeros(nB, nT, dtype=torch.float64)

    # -----------------------------------------------------------------------
    # Pre-compute beta values for all temperatures: shape (nT,)
    # -----------------------------------------------------------------------
    beta_arr = (1e6 * PLANCK / (BOLTZMANN * T_arr)).to(dtype=torch.float64)  # 1/MHz

    # -----------------------------------------------------------------------
    # Orientation loop — B-field and temperature loops are vectorized
    # -----------------------------------------------------------------------
    # Stack B and B+delta_B into a single (2*nB,) vector for batched eigh
    B_both = torch.cat([B_arr, B_arr + delta_B])   # (2*nB,)

    for iOri in range(nOri):
        phi = phi_arr[iOri].item()
        theta = theta_arr[iOri].item()
        w = weights_arr[iOri].item() / total_weight

        sin_th = math.sin(theta)
        cos_th = math.cos(theta)
        cos_ph = math.cos(phi)
        sin_ph = math.sin(phi)

        muzL = sin_th * cos_ph * mux + sin_th * sin_ph * muy + cos_th * muz

        # ------------------------------------------------------------------
        # Batch all 2*nB Hamiltonians and diagonalize in one call
        # H_batch shape: (2*nB, n, n)
        # ------------------------------------------------------------------
        H_batch = H0.unsqueeze(0) - B_both[:, None, None] * muzL.unsqueeze(0)
        E_batch, V_batch = _eigh(H_batch)   # (2*nB, n), (2*nB, n, n)

        E_B  = E_batch[:nB]    # (nB, n)
        V_B  = V_batch[:nB]    # (nB, n, n)
        E_Bd = E_batch[nB:]    # (nB, n)
        V_Bd = V_batch[nB:]    # (nB, n, n)

        # muzL diagonal in each eigenbasis: (nB, n)
        # diag_k(V^† muzL V) = einsum('...ij,...jk,...ki->...i', V†, muzL, V)
        muzL_diag_B  = _muzL_diag_batch(muzL, V_B)   # (nB, n)  real
        muzL_diag_Bd = _muzL_diag_batch(muzL, V_Bd)  # (nB, n)  real

        # ------------------------------------------------------------------
        # Vectorized Boltzmann + moment accumulation over T
        # E_B shape (nB, n); beta_arr shape (nT,)
        # ------------------------------------------------------------------
        # Shift eigenvalues by ground state: (nB, n)
        E_B_shift  = E_B  - E_B[:, 0:1]
        E_Bd_shift = E_Bd - E_Bd[:, 0:1]

        # Unnormalized populations: (nB, nT, n)
        # beta_arr (nT,) × E_B_shift (nB, n) → broadcast over (nB, nT, n)
        pop_B_raw  = torch.exp(-E_B_shift.unsqueeze(1)  * beta_arr[None, :, None])
        pop_Bd_raw = torch.exp(-E_Bd_shift.unsqueeze(1) * beta_arr[None, :, None])

        Z_B  = pop_B_raw.sum(dim=2, keepdim=True)    # (nB, nT, 1)
        Z_Bd = pop_Bd_raw.sum(dim=2, keepdim=True)

        pop_B  = pop_B_raw  / Z_B    # (nB, nT, n)
        pop_Bd = pop_Bd_raw / Z_Bd

        # ⟨μ_z⟩ = Σ_k pop_k * diag_k: (nB, nT)
        mu_z  = (pop_B  * muzL_diag_B.unsqueeze(1)).sum(dim=2)   # (nB, nT)
        mu_z2 = (pop_Bd * muzL_diag_Bd.unsqueeze(1)).sum(dim=2)  # (nB, nT)

        # Accumulate moment (MHz/mT → J/T → Bohr magnetons)
        mu_BM_out  += w * (mu_z  * _MHZ_PER_MT_TO_J_PER_T / BMAGN)

        # Susceptibility via forward difference (MHz/mT² → J/T² → m³/mol)
        deriv = (mu_z2 - mu_z) / delta_B             # (nB, nT) MHz/mT²
        chi_mol_out += w * deriv * _MHZ_PER_MT_TO_J_PER_T * 1e3 * _MU0 * _N_A

    return mu_BM_out, chi_mol_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_1d_float64(x) -> torch.Tensor:
    """Convert scalar / list / tensor to 1-D float64 tensor."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    return x.to(dtype=torch.float64).reshape(-1)


def _muzL_diag_batch(muzL: torch.Tensor, V_batch: torch.Tensor) -> torch.Tensor:
    """Batched diagonal ⟨k|muzL|k⟩ for a batch of eigenbases.

    Parameters
    ----------
    muzL : (n, n) complex128
    V_batch : (nB, n, n) complex128 — columns are eigenvectors

    Returns
    -------
    (nB, n) float64 — real diagonal elements
    """
    # diag(V^† muzL V) = (V^† muzL V)_kk = Σ_i Σ_j V*_ik (muzL)_ij V_jk
    # Using einsum: '...ij,...jk,...ki->...i' (but 'i' and 'k' swapped below for clarity)
    # Equivalent: for each b: diag(V_b^† muzL V_b)
    # = einsum 'bni,nj,bjk->bk' but simpler: compute V^† muzL V then take diagonal
    # tmp[b, i, k] = Σ_j muzL[i,j] * V_batch[b, j, k]
    tmp = muzL.unsqueeze(0) @ V_batch                    # (nB, n, n)
    # diag[b, k] = Σ_i V*[b, i, k] * tmp[b, i, k] — sum over i (dim=1)
    diag = torch.real((V_batch.conj() * tmp).sum(dim=1))  # (nB, n)
    return diag
