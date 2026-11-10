"""Top-level spin Hamiltonian assembler for torchspin.

Port of EasySpin's ``ham.m``.

The full spin Hamiltonian is::

    H = H0 - B0 · μ

where::

    H0  = H_ZF + H_EE + H_HF + H_NQ + H_NN   (field-independent)
    μ   = mux, muy, muz                        (total magnetic-dipole-moment operators)
    muzL = (B0/|B0|) · (mux, muy, muz)

All energies are in **MHz** (matching EasySpin's convention).
Magnetic field vectors are in **mT**.
"""
from __future__ import annotations

from typing import Optional

import torch

from torchspin.ham_cf import ham_cf
from torchspin.ham_ee import ham_ee
from torchspin.ham_ez import ham_ez
from torchspin.ham_hf import ham_hf
from torchspin.ham_nn import ham_nn
from torchspin.ham_nq import ham_nq
from torchspin.ham_nz import ham_nz
from torchspin.ham_oz import ham_oz
from torchspin.ham_so import ham_so
from torchspin.ham_zf import ham_zf
from torchspin.spinsystem import SpinSystem

__all__ = ['ham']


def ham(
    sys: SpinSystem,
    B0: Optional[list | torch.Tensor] = None,
    dtype: torch.dtype = torch.complex128,
    device: str = 'cpu',
) -> torch.Tensor | tuple:
    """Spin Hamiltonian matrix (or field-independent parts + moment operators).

    Parameters
    ----------
    sys:
        Spin system.
    B0:
        Static magnetic field vector in mT, shape ``(3,)``.  If given,
        returns the full Hamiltonian ``H = H0 - |B0| * muzL``.
        If ``None``, returns ``(H0, mux, muy, muz)``.
    dtype:
        Output dtype.  Default ``torch.complex128``.
    device:
        PyTorch device string.

    Returns
    -------
    H:
        If *B0* is given — full Hamiltonian (MHz), shape
        ``(nStates, nStates)``.
    (H0, mux, muy, muz):
        If *B0* is ``None`` — field-independent Hamiltonian and the three
        magnetic-dipole-moment operators (MHz and MHz/mT respectively),
        each of shape ``(nStates, nStates)``.

    Notes
    -----
    The sign convention follows EasySpin::

        H = H0 - B0[0]*mux - B0[1]*muy - B0[2]*muz
          = H0 - norm(B0) * muzL

    where ``muzL`` is the magnetic-moment operator projected along ``B0``.
    The nuclear Zeeman has a POSITIVE contribution to mux/muy/muz (opposite
    sign to electron Zeeman internally, but combined here with the same
    convention: H = H0 - B · μ_total).
    """
    # Field-independent part
    H0 = (
        ham_zf(sys, dtype=dtype, device=device)
        + ham_ee(sys, dtype=dtype, device=device)
        + ham_hf(sys, dtype=dtype, device=device)
        + ham_nq(sys, dtype=dtype, device=device)
        + ham_nn(sys, dtype=dtype, device=device)
    )

    # Add crystal-field and spin-orbit coupling when OAMs are present
    if sys.nL > 0:
        H0 = H0 + ham_cf(sys, dtype=dtype, device=device)
        H0 = H0 + ham_so(sys, dtype=dtype, device=device)

    # Generic extra field-independent terms (SpinSystem.H0Extra): sum coeff*op,
    # kept on the autograd graph in coeff and op. In H0, so both the (H0, mu)
    # and the ham(sys, B0) path include them.
    if getattr(sys, 'H0Extra', None) is not None:
        for coeff, op in sys.H0Extra:
            if dtype.is_complex:
                op_ = op.to(dtype=dtype, device=device)
            else:
                if bool(torch.any(op.detach().imag != 0)):
                    raise ValueError('ham: SpinSystem.H0Extra has a complex operator; '
                                     'use a complex dtype.')
                op_ = op.real.to(dtype=dtype, device=device)
            H0 = H0 + coeff.to(device) * op_

    # Electron magnetic-dipole-moment operators (negative prefactor baked in)
    mux, muy, muz = ham_ez(sys, B0=None, dtype=dtype, device=device)

    # Add nuclear magnetic-dipole-moment operators when nuclei are present.
    # ham_nz returns (mux_n, muy_n, muz_n) with POSITIVE prefactor.
    # mux_e already carries a negative prefactor, so the total moment is
    # mux = mux_e + mux_n and H = H0 - (mux*B) gives:
    #   electron contribution: -mux_e*B  = +BMAGN/PLANCK * g * B * Sx   (positive)
    #   nuclear contribution:  -mux_n*B  = -NMAGN/PLANCK * gn * B * Iz  (negative for gn>0)
    if sys.nNuclei > 0:
        mux_n, muy_n, muz_n = ham_nz(sys, B0=None, dtype=dtype, device=device)
        mux = mux + mux_n
        muy = muy + muy_n
        muz = muz + muz_n

    # Add orbital Zeeman moment operators when OAMs are present
    if sys.nL > 0:
        mux_oz, muy_oz, muz_oz = ham_oz(sys, B0=None, dtype=dtype, device=device)
        mux = mux + mux_oz
        muy = muy + muy_oz
        muz = muz + muz_oz

    # Higher-order Zeeman terms (Sys.Ham*): the zero-field part goes into H0 and
    # the linear-in-B part into the moment operators; at a given field the
    # complete ham_ezho(Sys,B0) is added instead. ham_ezho's G1 is dH/dB, and
    # H = H0 - mu.B, so the linear part enters the moments as -G1. EasySpin's
    # ham.m adds +G1, which flips the sign of these terms relative to
    # ham(Sys,B0) and to every other Zeeman term; torchspin deliberately does
    # not follow it (KNOWN_LIMITATIONS.md, "Higher-order Zeeman terms").
    # Quadratic and cubic terms (G2, G3) are not representable as H0 - mu.B and
    # are included only by ham(Sys, B0).
    ho_zeeman = sys.Ham is not None and any(
        bool(torch.any(v != 0)) for v in sys.Ham.values()
    )

    if B0 is None:
        if ho_zeeman:
            from torchspin.ham_ezho import ham_ezho
            G0, G1, _, _ = ham_ezho(sys, B0=None, dtype=dtype, device=device)
            H0 = H0 + G0
            mux = mux - G1[0]
            muy = muy - G1[1]
            muz = muz - G1[2]
        return H0, mux, muy, muz

    # Full Hamiltonian
    if isinstance(B0, torch.Tensor):
        Bvec = B0.double().tolist()
    else:
        Bvec = [float(b) for b in B0]

    H = H0 - (mux * Bvec[0] + muy * Bvec[1] + muz * Bvec[2])
    if ho_zeeman:
        from torchspin.ham_ezho import ham_ezho
        H = H + ham_ezho(sys, B0=Bvec, dtype=dtype, device=device)
    return H
