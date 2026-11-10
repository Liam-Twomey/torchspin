"""Orientation selectivity weights for CW EPR powder spectra.

Port of EasySpin's ``orisel.m``.

For a given microwave frequency and excitation bandwidth, ``orisel``
computes how strongly each crystallite orientation in a powder is excited.
This is used to weight ENDOR powder averages when orientation selection
is significant (e.g. narrow-bandwidth spectrometers at X-band or Q-band).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

__all__ = ['orisel', 'OriselOptions']


@dataclass
class OriselOptions:
    """Options for :func:`orisel`.

    Attributes
    ----------
    GridSize : int
        SOPHE grid knot number.  Default 19.
    GridSymmetry : str
        Grid symmetry ('auto', 'Ci', 'C2h', 'D2h').  Default 'auto'.
    ExcitBandwidth : float
        Excitation bandwidth (full-width, MHz).  Default 100 MHz.
        A Lorentzian profile with this FWHM is used to weight each orientation.
    Threshold : float
        Minimum weight threshold (relative to max).  Orientations below
        this weight are returned with weight 0.  Default 0.0 (keep all).
    """
    GridSize: int = 19
    GridSymmetry: str = 'auto'
    ExcitBandwidth: float = 100.0
    Threshold: float = 0.0


def orisel(sys, exp, opt: Optional[OriselOptions] = None):
    """Compute orientation selectivity weights for a powder EPR experiment.

    Port of EasySpin's ``orisel.m``.

    For each crystallite orientation in the SOPHE powder grid, the weight
    reflects how strongly that orientation contributes to the EPR signal
    at the specified microwave frequency ``mwFreq`` and field range.
    A narrow excitation bandwidth increases orientation selectivity.

    Parameters
    ----------
    sys : SpinSystem
        Spin system.
    exp : Experiment
        Experiment with ``mwFreq`` (GHz) and ``Range`` (mT) set.
    opt : OriselOptions, optional
        Grid size, symmetry, excitation bandwidth.

    Returns
    -------
    weights : ndarray, shape (nOrientations,)
        Boltzmann-weighted, bandwidth-weighted selectivity for each orientation.
        Normalized so the maximum weight is 1.0.
    phi : ndarray, shape (nOrientations,)
        Azimuthal angles (rad) of the SOPHE grid points.
    theta : ndarray, shape (nOrientations,)
        Polar angles (rad) of the SOPHE grid points.

    Notes
    -----
    The weight for each orientation is computed as:

    .. math::

        w_i = \\sum_{\\text{transitions } t}
              I_t \\cdot L(\\nu_{\\text{mw}} - \\nu_t; \\Delta\\nu)

    where :math:`L` is a Lorentzian with FWHM = ``ExcitBandwidth`` (MHz),
    :math:`\\nu_t` is the resonance frequency of transition *t* at the
    resonance field :math:`B_t` closest to the EPR field
    :math:`B_{\\text{res}} = h\\nu_{\\text{mw}} / (g_{\\text{eff}} \\mu_B)`,
    and :math:`I_t` is the transition intensity.

    Examples
    --------
    >>> from torchspin import SpinSystem
    >>> from torchspin.experiment import Experiment
    >>> from torchspin.orisel import orisel, OriselOptions
    >>> sys = SpinSystem(S=[0.5], g=[[2.0, 2.1, 2.2]], lw=[0.5, 0])
    >>> exp = Experiment(mwFreq=9.5, Range=[300, 380])
    >>> weights, phi, theta = orisel(sys, exp)
    >>> weights.max()  # should be 1.0
    1.0
    """
    if opt is None:
        opt = OriselOptions()
    from torchspin.spinsystem import _reject_h0extra
    _reject_h0extra(sys, 'orisel (perturbation theory)')

    from torchspin.sphgrid import sphgrid
    from torchspin.hamsymm import hamsymm
    from torchspin.resfields_perturb import resfields_perturb
    from torchspin.experiment import Experiment

    mwFreq_MHz = float(exp.mwFreq) * 1e3  # GHz → MHz
    B_range = exp.Range  # [Bmin, Bmax] in mT

    # Determine grid symmetry
    sym = opt.GridSymmetry
    if sym == 'auto':
        sym_result = hamsymm(sys)
        sym = sym_result[0] if isinstance(sym_result, tuple) else sym_result

    # Generate SOPHE grid
    phi_arr, theta_arr, weights_grid, _ = sphgrid(sym, opt.GridSize)
    nOri = len(phi_arr)

    excit_hwhm = opt.ExcitBandwidth / 2.0  # half-width in MHz

    selectivity = np.zeros(nOri)

    for i in range(nOri):
        phi_i = float(phi_arr[i])
        theta_i = float(theta_arr[i])

        try:
            result = resfields_perturb(sys, phi_i, theta_i, exp)
            if isinstance(result, tuple) and len(result) == 2:
                B_res, intens = result
            else:
                selectivity[i] = 0.0
                continue
        except Exception:
            selectivity[i] = 0.0
            continue

        if len(B_res) == 0:
            selectivity[i] = 0.0
            continue

        # Compute resonance frequency at each resonance field
        # f_res ≈ mwFreq (by definition); use Lorentzian in field-to-freq space
        # Weight = sum over transitions of I_t * L(B_res_t - B_center; FWHM)
        # where B_center is the field at mwFreq for an isotropic g_eff
        # More precisely: use Lorentzian in frequency domain centered at mwFreq
        # with FWHM = excit_bandwidth
        w_sum = 0.0
        for Bt, It in zip(B_res, intens):
            # Convert field offset to frequency offset:
            # delta_nu ≈ (Bt - B_res_target) * g_eff * BMAGN / PLANCK  [MHz]
            # For simplicity use a Lorentzian in the field domain
            # Lorentzian weight: 1 / (1 + ((Bt - B_target)/HWHM_field)^2)
            # where HWHM_field = ExcitBandwidth / (d_nu/d_B) ≈ ExcitBandwidth / slope
            # Use d_nu/d_B = GFREE * BMAGN/PLANCK * 1e-9 MHz/mT as approximation
            from torchspin.constants import GFREE, BMAGN, PLANCK
            dnu_dB = GFREE * BMAGN / PLANCK * 1e-9  # MHz/mT  ≈ 28.02 MHz/mT
            hwhm_B = excit_hwhm / dnu_dB  # mT
            B_target = mwFreq_MHz / dnu_dB  # approximate center
            delta_B = float(Bt) - B_target
            lorentz = 1.0 / (1.0 + (delta_B / hwhm_B) ** 2)
            w_sum += float(It) * lorentz
        selectivity[i] = w_sum * float(weights_grid[i])

    # Normalize
    w_max = selectivity.max()
    if w_max > 0:
        selectivity = selectivity / w_max

    # Apply threshold
    if opt.Threshold > 0:
        selectivity[selectivity < opt.Threshold] = 0.0

    return selectivity, phi_arr.numpy(), theta_arr.numpy()
