"""Thermal equilibrium density matrix.

Port of EasySpin's ``sigeq.m``.

Computes the Boltzmann-weighted density matrix for a spin Hamiltonian at a
given temperature.

Example
-------
>>> import torch
>>> from torchspin.sigeq import sigeq
>>> H = torch.diag(torch.tensor([0.0, 100.0, 200.0]))  # MHz
>>> sigma = sigeq(H, 300.0)
>>> sigma.shape
torch.Size([3, 3])
"""
from __future__ import annotations

import torch

from torchspin.constants import PLANCK, BOLTZMANN


def sigeq(
    Ham: torch.Tensor,
    Temperature: float,
    *,
    polarization: bool = False,
) -> torch.Tensor:
    """Compute thermal equilibrium density matrix.

    Parameters
    ----------
    Ham:
        Hamiltonian matrix (in MHz), shape ``(N, N)``.
    Temperature:
        Temperature in Kelvin.  Must be > 0.
    polarization:
        If ``True``, return only the polarization part (deviation from
        the maximally mixed state ``I/N``).

    Returns
    -------
    sigma:
        Hermitian density matrix, shape ``(N, N)``.
        If ``polarization=True``, trace is zero.
    """
    if not isinstance(Temperature, (int, float)) or Temperature <= 0:
        raise ValueError(f"Temperature must be a positive real number, got {Temperature}")

    Ham = torch.as_tensor(Ham, dtype=torch.complex128)
    if Ham.ndim != 2 or Ham.shape[0] != Ham.shape[1]:
        raise ValueError(f"Ham must be a square matrix, got shape {Ham.shape}")

    N = Ham.shape[0]

    # beta = h / (k_B * T), with h in J·s and Ham in MHz (factor 1e6 for Hz→MHz)
    beta = (1e6 * PLANCK) / (BOLTZMANN * Temperature)

    # Boltzmann density matrix: sigma = expm(-beta * H) / Z
    sigma = torch.matrix_exp(-beta * Ham)
    Z = torch.trace(sigma).real
    sigma = sigma / Z

    # Hermitianize to suppress numerical artifacts
    sigma = (sigma + sigma.conj().T) / 2

    if polarization:
        sigma = sigma - torch.eye(N, dtype=sigma.dtype, device=sigma.device) / N

    return sigma
