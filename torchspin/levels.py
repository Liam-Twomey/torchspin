"""Energy levels of a spin system as a function of magnetic field.

Port of EasySpin's ``levels.m``.

Computes eigenvalues (and optionally eigenvectors) of H(B) = H0 - B·muzL
across a sweep of field values for one or more orientations.

Example
-------
>>> from torchspin import SpinSystem
>>> from torchspin.levels import levels
>>> import torch
>>> sys = SpinSystem(S=[1], g=[[2.0, 2.0, 2.0]], D=[[0, 0, 1000.0]])
>>> B = torch.linspace(0, 600, 301)
>>> B_axis, E = levels(sys, 0.0, 0.0, B)
>>> E.shape  # (nStates=3, nB=301)
torch.Size([3, 301])
"""
from __future__ import annotations

import math
from typing import Optional, Union

import torch

from torchspin.ham import ham
from torchspin.spinsystem import SpinSystem


def levels(
    sys: SpinSystem,
    phi: Union[float, torch.Tensor],
    theta: Union[float, torch.Tensor],
    B: Union[torch.Tensor, list, tuple],
    *,
    return_vectors: bool = False,
    quadratic_zeeman: bool = False,
) -> Union[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Compute energy levels of a spin system vs magnetic field.

    Parameters
    ----------
    sys:
        Spin system specification.
    phi:
        Azimuthal angle(s) of the static field in the molecular frame (radians).
        Scalar for a single orientation.
    theta:
        Polar angle(s) of the static field in the molecular frame (radians).
        Scalar for a single orientation.
    B:
        Magnetic field values (mT).  Can be:

        * A 1-D tensor / list of field values, used directly.
        * A 2-element ``[B_min, B_max]`` range, expanded to 101 points (matching
          MATLAB ``levels.m`` default).
        * A scalar tensor or float giving a single field value.
    return_vectors:
        If ``True``, also return eigenvector array ``V``.
    quadratic_zeeman:
        If ``True``, include the field-quadratic and field-cubic higher-order
        Zeeman terms of ``sys.Ham`` (orders lB = 2, 3), so that
        H(B) = H0 − B·muzL + B²·n·G2·n + B³·G3[n,n,n] equals ``ham(sys, B·n)``
        (same meaning as ``Options.QuadraticZeeman``).  Default ``False``: these
        terms are dropped, as before.

    Returns
    -------
    B_axis:
        Field axis, shape ``(nB,)`` (mT).
    E:
        Eigenvalues (MHz), sorted ascending, shape ``(nStates, nB)`` for a single
        orientation, or ``(nOri, nStates, nB)`` for multiple orientations.
    V: *(only when* ``return_vectors=True`` *)*
        Eigenvectors, shape ``(nStates, nStates, nB)`` for single orientation,
        or ``(nOri, nStates, nStates, nB)`` for multiple orientations.

    Notes
    -----
    The Hamiltonian is:

    .. code-block:: text

        H(B) = H0 - B * muzL
        muzL = sin(theta)*cos(phi)*mux + sin(theta)*sin(phi)*muy + cos(theta)*muz

    where ``(phi, theta)`` define the static field direction in the molecular frame
    (same convention as :func:`resfields`).
    """
    # -----------------------------------------------------------------------
    # Build field axis
    # -----------------------------------------------------------------------
    B_tensor = _parse_field(B)
    nB = B_tensor.shape[0]

    # -----------------------------------------------------------------------
    # Build field-independent Hamiltonian and moment operators
    # -----------------------------------------------------------------------
    H0, mux, muy, muz = ham(sys, B0=None)
    nStates = H0.shape[0]
    zho = None
    if quadratic_zeeman:
        from torchspin.ham_ezho import field_nonlinear_zeeman_ops, project_field_nonlinear_zeeman
        zho = field_nonlinear_zeeman_ops(sys, dtype=H0.dtype, device=H0.device)

    # -----------------------------------------------------------------------
    # Normalize phi / theta to arrays so we can handle both scalar and vector
    # -----------------------------------------------------------------------
    phi_arr = torch.as_tensor(phi, dtype=torch.float64).reshape(-1)
    theta_arr = torch.as_tensor(theta, dtype=torch.float64).reshape(-1)

    # For scalar phi/theta the user likely passes two scalars; broadcast to
    # the same length (must be equal length or one is scalar).
    if phi_arr.numel() == 1 and theta_arr.numel() > 1:
        phi_arr = phi_arr.expand(theta_arr.numel())
    elif theta_arr.numel() == 1 and phi_arr.numel() > 1:
        theta_arr = theta_arr.expand(phi_arr.numel())
    elif phi_arr.numel() != theta_arr.numel():
        raise ValueError(
            f"phi and theta must have the same length or one must be scalar; "
            f"got {phi_arr.numel()} and {theta_arr.numel()}."
        )

    nOri = phi_arr.numel()
    single_ori = nOri == 1

    # -----------------------------------------------------------------------
    # Allocate output
    # -----------------------------------------------------------------------
    if single_ori:
        E_out = torch.zeros(nStates, nB, dtype=torch.float64)
        if return_vectors:
            V_out = torch.zeros(nStates, nStates, nB, dtype=torch.complex128)
    else:
        E_out = torch.zeros(nOri, nStates, nB, dtype=torch.float64)
        if return_vectors:
            V_out = torch.zeros(nOri, nStates, nStates, nB, dtype=torch.complex128)

    # -----------------------------------------------------------------------
    # Loop over orientations
    # -----------------------------------------------------------------------
    for iOri in range(nOri):
        ph = phi_arr[iOri].item()
        th = theta_arr[iOri].item()

        sin_th = math.sin(th)
        cos_th = math.cos(th)
        cos_ph = math.cos(ph)
        sin_ph = math.sin(ph)

        # muzL: projection of total moment on field direction
        muzL = sin_th * cos_ph * mux + sin_th * sin_ph * muy + cos_th * muz
        if zho is not None:
            Q2, Q3 = project_field_nonlinear_zeeman(
                zho, torch.tensor([[sin_th * cos_ph, sin_th * sin_ph, cos_th]], dtype=torch.float64))

        # Loop over field values
        for iB in range(nB):
            B_val = B_tensor[iB].item()
            H = H0 - B_val * muzL
            if zho is not None:
                if Q2 is not None:
                    H = H + B_val ** 2 * Q2[0]
                if Q3 is not None:
                    H = H + B_val ** 3 * Q3[0]

            if return_vectors:
                E_i, V_i = torch.linalg.eigh(H)
                if single_ori:
                    E_out[:, iB] = E_i
                    V_out[:, :, iB] = V_i
                else:
                    E_out[iOri, :, iB] = E_i
                    V_out[iOri, :, :, iB] = V_i
            else:
                E_i = torch.linalg.eigvalsh(H)
                if single_ori:
                    E_out[:, iB] = E_i
                else:
                    E_out[iOri, :, iB] = E_i

    if return_vectors:
        return B_tensor, E_out, V_out
    return B_tensor, E_out


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_field(B) -> torch.Tensor:
    """Convert B input to a 1-D float64 tensor of field values."""
    if not isinstance(B, torch.Tensor):
        B = torch.tensor(B, dtype=torch.float64)
    else:
        B = B.to(dtype=torch.float64)

    B = B.reshape(-1)

    if B.numel() == 2:
        # Treat as [B_min, B_max] range (101 points, matching MATLAB default)
        B = torch.linspace(B[0].item(), B[1].item(), 101, dtype=torch.float64)
    # else: use as-is (scalar or array)

    return B
