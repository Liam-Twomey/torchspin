"""Differentiable EPR spectrum functions for gradient-based optimization.

:func:`differentiable_spectrum` builds a :class:`~torchspin.spinsystem.SpinSystem`
from parameter tensors and, by default (``method='pepper'``), evaluates it with
:func:`torchspin.pepper_autograd.pepper_autograd` -- pepper's own forward path
(resonance-field search with transition tracking, EasySpin grid interpolation,
SOPHE triangle/zone projection, EasySpin convolution and harmonic) written on
the autograd graph.  The forward output equals :func:`torchspin.pepper` to
rounding and ``torch.autograd`` propagates gradients to ``g``, ``A``, ``D`` and
``lw_mT``.

The two earlier stand-alone models are kept as ``method='broadband'`` (N-spin:
diagonalize H(B) at every field point, soft Gaussian per transition and
orientation) and ``method='analytical'`` (S=1/2 closed-form resonance fields,
Gaussian per orientation).  Both sum discrete orientations without
interpolation or projection and therefore carry orientation-grid ripple at the
GridSize where pepper is already converged (Cu(II) hyperfine, GridSize 31: cosine
to EasySpin 0.985-0.999 in absorption and 0.73-0.88 in first derivative, versus
0.9998-1.0000 / 0.999 for pepper); they are deprecated.

Typical use::

    import torch
    from torchspin.autograd import differentiable_spectrum

    g = torch.tensor([2.45, 2.20, 1.95], dtype=torch.float64, requires_grad=True)
    A = torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64, requires_grad=True)
    B, spec = differentiable_spectrum(g, lw_mT=0.8, mwFreq_GHz=9.65,
                                      B_range=(280.0, 380.0), Nucs=['1H'], A=A)
    grad_g, grad_A = torch.autograd.grad(spec.sum(), [g, A])
"""
from __future__ import annotations

import math
from typing import Optional

import torch

__all__ = ['differentiable_spectrum', 'inputs_require_grad']


# ---------------------------------------------------------------------------
# N-spin broadband helpers
# ---------------------------------------------------------------------------

def inputs_require_grad(*objs) -> bool:
    """True if any tensor field of the given SpinSystem/Experiment objects (or
    lists of them) requires grad. Lists and tuples of tensors (e.g. ``lw``) are
    searched one level deep."""
    import dataclasses

    def _rg(v) -> bool:
        if torch.is_tensor(v):
            return bool(v.requires_grad)
        if isinstance(v, (list, tuple)):
            return any(torch.is_tensor(u) and u.requires_grad for u in v)
        return False

    def _rg_deep(v) -> bool:
        # SpinSystem.H0Extra is a list of (coeff, op) pairs: two levels deep
        if torch.is_tensor(v):
            return bool(v.requires_grad)
        if isinstance(v, (list, tuple)):
            return any(_rg_deep(u) for u in v)
        return False

    for obj in objs:
        items = obj if isinstance(obj, (list, tuple)) else [obj]
        for it in items:
            if it is None or not dataclasses.is_dataclass(it):
                continue
            if any(_rg(getattr(it, f.name, None)) for f in dataclasses.fields(it)):
                return True
            if _rg_deep(getattr(it, 'H0Extra', None)):
                return True
    return False


def _broadband_spectrum(
    sys: 'SpinSystem',
    mwFreq_GHz: float,
    B_range: tuple[float, float],
    nPoints: int,
    Harmonic: int,
    lw_mT: float | torch.Tensor,
    GridSize: int,
    GridSymmetry: str,
    Temperature: Optional[float | torch.Tensor],
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadband differentiable EPR spectrum for arbitrary spin systems.

    Diagonalizes H(B) at every field point for each powder orientation and
    accumulates transition intensities via a soft Gaussian kernel in frequency
    space.  The entire pipeline is on the PyTorch computation graph.
    """
    from torchspin.ham import ham
    from torchspin.sphgrid import sphgrid
    from torchspin.hamsymm import hamsymm
    from torchspin.constants import BMAGN, PLANCK, BOLTZMANN

    _dev = device
    mwFreq_MHz = mwFreq_GHz * 1e3  # GHz → MHz

    # ---- orientation grid ----
    if GridSymmetry == 'auto':
        try:
            result = hamsymm(sys)
            grid_sym = result[0] if isinstance(result, tuple) else result
        except Exception:
            grid_sym = 'C1'
    else:
        grid_sym = GridSymmetry

    phi_arr, theta_arr, weights_arr, _ = sphgrid(grid_sym, GridSize)
    phi = phi_arr.to(device=_dev, dtype=dtype).detach()
    theta = theta_arr.to(device=_dev, dtype=dtype).detach()
    wt = weights_arr.to(device=_dev, dtype=dtype).detach()
    wt = wt * (4.0 * math.pi / wt.sum())  # normalize to 4π
    nOri = phi.shape[0]

    # ---- field axis ----
    B_min, B_max = float(B_range[0]), float(B_range[1])
    B_axis = torch.linspace(B_min, B_max, nPoints, dtype=dtype, device=_dev)

    # ---- Hamiltonian operators (on the computation graph) ----
    H0, mux, muy, muz = ham(sys, dtype=torch.complex128, device=_dev)
    nStates = H0.shape[0]

    # ---- frequency-domain linewidth ----
    # Convert field-domain FWHM (mT) to frequency FWHM (MHz) using an
    # approximate average conversion: σ_freq ≈ lw_mT × g_iso × BMAGN/(PLANCK·1e9)
    # We use g_iso ≈ 2.0 as a representative value for the conversion.
    # The exact per-transition correction is applied below.
    mhz_per_mt = BMAGN / PLANCK * 1e-9  # MHz/mT per unit g ≈ 13.996
    sigma_B = lw_mT / math.sqrt(8.0 * math.log(2.0))  # mT (σ from FWHM)

    # ---- accumulate spectrum ----
    spec = torch.zeros(nPoints, dtype=dtype, device=_dev)

    for iOri in range(nOri):
        sin_th = torch.sin(theta[iOri])
        cos_th = torch.cos(theta[iOri])
        cos_ph = torch.cos(phi[iOri])
        sin_ph = torch.sin(phi[iOri])

        # lab-z projection of magnetic moment operator
        muzL = sin_th * cos_ph * mux + sin_th * sin_ph * muy + cos_th * muz

        # Batched Hamiltonian: H(B_k) = H0 - B_k * muzL, shape (nB, n, n)
        H_batch = H0.unsqueeze(0) - B_axis.reshape(-1, 1, 1).to(torch.complex128) * muzL.unsqueeze(0)

        # Hermitianise (guards against floating-point asymmetry)
        H_batch = (H_batch + H_batch.conj().transpose(-2, -1)) / 2

        # Batch eigendecompose
        E, V = torch.linalg.eigh(H_batch)  # E: (nB, n), V: (nB, n, n)

        # ---- transition frequencies and intensities ----
        # Build all i<j pairs
        idx_i = []
        idx_j = []
        for i in range(nStates):
            for j in range(i + 1, nStates):
                idx_i.append(i)
                idx_j.append(j)
        nTrans = len(idx_i)
        if nTrans == 0:
            continue

        # Transition frequencies: ΔE[b, t] = E[b, j] - E[b, i]  (MHz)
        dE = E[:, idx_j] - E[:, idx_i]  # (nB, nTrans)

        # Transition intensities: perpendicular-mode transition rate
        # TR = (|μ_L|² − |n·μ_L|²) / 2 where μ_L = (<i|μ|j>) vector
        # Compute matrix elements of mux, muy, muz in eigenbasis at each B
        Vi = V[:, :, idx_i]  # (nB, n, nTrans)
        Vj = V[:, :, idx_j]  # (nB, n, nTrans)

        # <i|Op|j> = Vi†·Op·Vj for each B and each transition
        # Use einsum: Vi†[b,:,t] @ Op @ Vj[b,:,t]
        mux_me = torch.einsum('bnt,nm,bmt->bt', Vi.conj(), mux, Vj)  # (nB, nTrans)
        muy_me = torch.einsum('bnt,nm,bmt->bt', Vi.conj(), muy, Vj)
        muz_me = torch.einsum('bnt,nm,bmt->bt', Vi.conj(), muz, Vj)
        muzL_me = torch.einsum('bnt,nm,bmt->bt', Vi.conj(), muzL, Vj)

        mu_sq = (mux_me.abs()**2 + muy_me.abs()**2 + muz_me.abs()**2).real  # (nB, nTrans)
        muzL_sq = muzL_me.abs().pow(2).real
        TR = (mu_sq - muzL_sq) / 2.0  # (nB, nTrans)

        # Boltzmann population factor
        if Temperature is not None and Temperature > 0:
            kT_MHz = BOLTZMANN * Temperature / (PLANCK * 1e6)  # kT in MHz
            # populations: exp(-E/kT), normalized per B-point
            E_shifted = E - E[:, 0:1]  # shift so lowest = 0
            pops = torch.exp(-E_shifted / kT_MHz)  # (nB, n)
            pops = pops / pops.sum(dim=1, keepdim=True)
            # polarisation = pop_i - pop_j for each transition
            pol = pops[:, idx_i] - pops[:, idx_j]  # (nB, nTrans)
        else:
            pol = torch.ones(1, 1, dtype=dtype, device=_dev)

        intensity = (TR * pol).real  # (nB, nTrans)

        # ---- Per-transition frequency-domain linewidth ----
        # dΔE/dB = <j|muzL|j> - <i|muzL|i>  (slope of transition frequency vs B)
        muzL_diag_i = torch.einsum('bnt,nm,bmt->bt', Vi.conj(), muzL, Vi).real  # (nB, nTrans)
        muzL_diag_j = torch.einsum('bnt,nm,bmt->bt', Vj.conj(), muzL, Vj).real
        dEdB = (muzL_diag_j - muzL_diag_i).abs().clamp(min=1e-6)  # (nB, nTrans)

        sigma_freq = dEdB * sigma_B  # (nB, nTrans) MHz

        # ---- Gaussian kernel in frequency domain ----
        # Resonance deviation: ΔE(B) - mwFreq  (MHz)
        deviation = dE.real - mwFreq_MHz  # (nB, nTrans)

        # Normalized Gaussian: G(x, σ) = exp(-x²/(2σ²)) / (σ√(2π))
        gauss_norm = 1.0 / (sigma_freq * math.sqrt(2.0 * math.pi))
        gauss = torch.exp(-0.5 * (deviation / sigma_freq) ** 2) * gauss_norm

        # To match field-domain pepper output, multiply by dEdB to convert
        # from frequency-domain density to field-domain density:
        # S_field(B) = S_freq(ν) × |dν/dB|
        # Combined with the 1/σ_freq in the Gaussian normalization, the dEdB
        # factors cancel, giving an effective field-domain Gaussian.
        contribution = intensity * gauss * dEdB  # (nB, nTrans)

        # Sum over transitions, weight by orientation
        spec = spec + wt[iOri] * contribution.sum(dim=1).real

    # ---- Harmonic differentiation ----
    if Harmonic >= 1:
        # Numerical pseudo-derivative via spectral differentiation
        dB = (B_max - B_min) / (nPoints - 1) if nPoints > 1 else 1.0
        for _ in range(Harmonic):
            spec = torch.gradient(spec, spacing=(dB,), dim=0)[0]

    return B_axis, spec


def _legacy_differentiable_spectrum(
    g: torch.Tensor,
    lw_mT: float | torch.Tensor,
    mwFreq_GHz: float,
    B_range: tuple[float, float],
    nPoints: int = 1024,
    Harmonic: int = 1,
    GridSize: int = 31,
    GridSymmetry: str = 'D2h',
    device: Optional[str] = None,
    # ---- N-spin parameters (trigger broadband path when any is set) ----
    S: float = 0.5,
    Nucs: Optional[list[str]] = None,
    A: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None,
    Temperature: Optional[float | torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fully differentiable powder EPR spectrum.

    Computes a CW EPR powder spectrum as a differentiable function of the
    spin-Hamiltonian parameters.  The entire computation is a sequence of
    PyTorch tensor operations, so ``torch.autograd.grad`` / ``.backward()``
    propagate gradients to ``g`` (and to ``A``, ``D`` when provided).

    Two internal paths:

    * **Analytical** (S=1/2 only, no nuclei/ZFS): fast closed-form B_res.
    * **Broadband** (general): diagonalizes H(B) at every field point,
      accumulates via soft Gaussian kernels — no root-finding.

    Parameters
    ----------
    g:
        Principal g-values ``[gx, gy, gz]``, shape ``(3,)``.
        Set ``requires_grad=True`` before calling to enable autograd.
    lw_mT:
        Gaussian linewidth (FWHM) in mT.  Must be positive.
        A scalar tensor with ``requires_grad=True`` enables linewidth gradients.
    mwFreq_GHz:
        Microwave frequency in GHz.
    B_range:
        ``(B_min, B_max)`` sweep range in mT.
    nPoints:
        Number of spectral points.  Default 1024.
    Harmonic:
        Detection harmonic: 0 = absorption, 1 = first-derivative (default),
        2 = second-derivative.
    GridSize:
        Number of SOPHE grid knots along the quarter meridian.  Default 31.
    GridSymmetry:
        SOPHE grid symmetry: ``'D2h'`` (default), ``'Dinfh'``, ``'Ci'``,
        or ``'auto'`` (auto-detect from Hamiltonian symmetry).
    device:
        PyTorch device string.  Defaults to the device of ``g``.
    S:
        Electron spin quantum number.  Default 0.5.
    Nucs:
        Nuclear isotope labels, e.g. ``['1H']`` or ``['14N', '1H']``.
        When provided (with ``A``), activates the N-spin broadband path.
    A:
        Hyperfine tensor(s) in MHz.  Shape ``(nNuclei, 3)`` for diagonal
        tensors or ``(nNuclei*3, 3)`` for full 3×3 tensors.
        Set ``requires_grad=True`` to enable gradient propagation.
    D:
        Zero-field splitting parameter(s) in MHz.  Shape ``(3,)`` for
        principal values ``[Dx, Dy, Dz]``, or shape ``(1,)``/``(2,)`` for
        ``D`` or ``[D, E]``.  Requires ``S > 0.5``.
    Temperature:
        Temperature in K for Boltzmann populations.  ``None`` = infinite T.
        In the N-spin path, a positive scalar tensor with
        ``requires_grad=True`` enables temperature gradients.

    Returns
    -------
    B_axis : torch.Tensor, shape (nPoints,)
        Field axis in mT.
    spec : torch.Tensor, shape (nPoints,)
        Spectrum (on the computation graph when input tensors require grad).

    Notes
    -----
    **Linewidth / gradient precision (N-spin broadband path only).**
    The broadband path accumulates per-transition contributions via a soft
    Gaussian kernel whose width is set by ``lw_mT`` (and per-orientation
    strain widths, if any). For gradients to remain well-conditioned, the
    effective kernel width at each field point must span at least a few
    grid samples. As a rule of thumb:

    * ``lw_mT >= 0.5 * (B_range[1] - B_range[0]) / nPoints`` gives reliable
      gradient magnitudes.
    * Below that, gradients can approach zero for large Hilbert spaces
      because kernel samples alias between grid points. If you need very
      narrow lines with accurate gradients, increase ``nPoints``.
    * For the analytical S=1/2 path (``Nucs=None``, ``D=None``, ``S=0.5``)
      this constraint does not apply — there's no kernel discretization.

    Examples
    --------
    S=1/2 analytical (unchanged from original):

    >>> import torch
    >>> from torchspin.autograd import differentiable_spectrum
    >>> g = torch.tensor([2.0, 2.1, 2.2], dtype=torch.float64, requires_grad=True)
    >>> B, spec = differentiable_spectrum(g, lw_mT=1.0, mwFreq_GHz=9.5,
    ...                                   B_range=(300.0, 380.0), nPoints=512)
    >>> grad, = torch.autograd.grad(spec.sum(), g)
    >>> grad.shape
    torch.Size([3])

    N-spin with hyperfine:

    >>> A = torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64, requires_grad=True)
    >>> B, spec = differentiable_spectrum(g, lw_mT=1.0, mwFreq_GHz=9.5,
    ...     B_range=(300.0, 380.0), Nucs=['1H'], A=A)
    >>> grad_g, grad_A = torch.autograd.grad(spec.sum(), [g, A])
    """
    from torchspin.sphgrid import sphgrid
    from torchspin.constants import BMAGN, PLANCK

    if g.dim() != 1 or g.shape[0] != 3:
        raise ValueError("g must be a 1-D tensor of shape (3,) = [gx, gy, gz].")
    if lw_mT <= 0:
        raise ValueError("lw_mT must be positive.")
    if Harmonic not in (0, 1, 2):
        raise ValueError("Harmonic must be 0, 1, or 2.")

    # Device consistency: A and D must live on the same device as g to avoid
    # silent device-mismatch errors deep in eigh/matmul.
    for _name, _t in (('A', A), ('D', D)):
        if _t is not None and hasattr(_t, 'device') and str(_t.device) != str(g.device):
            raise ValueError(
                f"{_name} is on device '{_t.device}' but g is on '{g.device}'. "
                f"Move all tensors to the same device before calling "
                f"differentiable_spectrum()."
            )

    # ---- Route to N-spin broadband path when nuclei, ZFS, or S>1/2 ----
    _use_broadband = (Nucs is not None) or (D is not None) or (S != 0.5)
    if _use_broadband:
        from torchspin.spinsystem import SpinSystem
        _device = device if device is not None else str(g.device)
        dtype = g.dtype

        # Build SpinSystem from parameter tensors
        # g must be reshaped to (1, 3) for SpinSystem
        g_2d = g.unsqueeze(0) if g.dim() == 1 else g
        sys_kwargs: dict = dict(S=[S], g=g_2d)
        if Nucs is not None:
            sys_kwargs['Nucs'] = Nucs
        if A is not None:
            A_2d = A.unsqueeze(0) if A.dim() == 1 else A
            sys_kwargs['A'] = A_2d
        if D is not None:
            # D can be scalar, [D,E], or [Dx,Dy,Dz]
            if D.dim() == 0:
                D_2d = D.unsqueeze(0).unsqueeze(0)
            elif D.dim() == 1:
                D_2d = D.unsqueeze(0)
            else:
                D_2d = D
            sys_kwargs['D'] = D_2d

        sys = SpinSystem(**sys_kwargs)
        return _broadband_spectrum(
            sys, mwFreq_GHz, B_range, nPoints, Harmonic,
            lw_mT, GridSize, GridSymmetry, Temperature,
            _device, dtype,
        )

    _device = device if device is not None else g.device
    dtype = g.dtype

    # ------------------------------------------------------------------ grid
    # Grid angles and weights are constants (detached from the graph).
    phi_arr, theta_arr, weights, _ = sphgrid(GridSymmetry, GridSize)
    phi     = phi_arr.to(device=_device, dtype=dtype).detach()
    theta   = theta_arr.to(device=_device, dtype=dtype).detach()
    weights = weights.to(device=_device, dtype=dtype).detach()
    # Normalize weights to sum 4π (full-sphere equivalent)
    weights = weights * (4.0 * math.pi / weights.sum())

    # --------------------------------------------------------- resonance field
    gx, gy, gz = g[0], g[1], g[2]
    sin_th = torch.sin(theta)
    cos_th = torch.cos(theta)
    cos_ph = torch.cos(phi)
    sin_ph = torch.sin(phi)

    # g_eff² for each orientation
    g_eff_sq = (
        gx ** 2 * (sin_th * cos_ph) ** 2
        + gy ** 2 * (sin_th * sin_ph) ** 2
        + gz ** 2 * cos_th ** 2
    )
    g_eff = torch.sqrt(g_eff_sq)                          # (nOri,)

    # B_res [mT] = mwFreq_MHz / (g_eff * BMAGN/PLANCK * 1e-9)
    # BMAGN/PLANCK * 1e-9 converts from MHz/mT (≈ 13.996 MHz/mT at g=2)
    mhz_per_mt = BMAGN / PLANCK * 1e-9                    # MHz/mT at g=1
    B_res = (mwFreq_GHz * 1e3) / (g_eff * mhz_per_mt)    # mT, (nOri,)

    # --------------------------------------------------------- field axis
    B_min, B_max = float(B_range[0]), float(B_range[1])
    B_axis = torch.linspace(B_min, B_max, nPoints,
                             dtype=dtype, device=_device)  # (nPts,)

    # ------------------------------------------------- Gaussian kernel sum
    sigma = lw_mT / math.sqrt(8.0 * math.log(2.0))        # σ from FWHM

    # Pairwise field difference: shape (nOri, nPts)
    dB = B_axis.unsqueeze(0) - B_res.unsqueeze(1)

    # Normalized Gaussian: G(dB) = exp(-0.5*(dB/σ)²) / (σ√(2π))
    gauss_norm = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    gauss = torch.exp(-0.5 * (dB / sigma) ** 2) * gauss_norm

    if Harmonic == 0:
        kernel = gauss
    elif Harmonic == 1:
        # First derivative of Gaussian: −(dB/σ²) · G(dB)
        kernel = -(dB / (sigma ** 2)) * gauss
    else:  # Harmonic == 2
        # Second derivative: (dB²/σ⁴ − 1/σ²) · G(dB)
        kernel = ((dB ** 2) / (sigma ** 4) - 1.0 / (sigma ** 2)) * gauss

    # Weighted sum over orientations → spectrum  (nPts,)
    spec = (weights.unsqueeze(1) * kernel).sum(0)

    return B_axis, spec


def _pepper_autograd_spectrum(g, lw_mT, mwFreq_GHz, B_range, nPoints, Harmonic, GridSize, GridSymmetry,
                              device, S, Nucs, A, D, Temperature):
    """``method='pepper'``: SpinSystem from the parameter tensors → pepper_autograd."""
    from torchspin.spinsystem import SpinSystem
    from torchspin.experiment import Experiment, Options
    from torchspin.pepper_autograd import pepper_autograd
    g_2d = g.unsqueeze(0) if g.dim() == 1 else g
    kw: dict = dict(S=[S], g=g_2d)
    if Nucs is not None:
        kw['Nucs'] = Nucs
        if A is not None:
            kw['A'] = A.unsqueeze(0) if A.dim() == 1 else A
    if D is not None:
        kw['D'] = D.reshape(1, -1) if D.dim() <= 1 else D
    lw_t = lw_mT if isinstance(lw_mT, torch.Tensor) else torch.tensor(float(lw_mT), dtype=torch.float64)
    kw['lw'] = [lw_t.to(torch.float64), 0.0]
    sys = SpinSystem(**kw)
    exp = Experiment(mwFreq=float(mwFreq_GHz), Range=[float(B_range[0]), float(B_range[1])], nPoints=int(nPoints),
                     Harmonic=int(Harmonic),
                     Temperature=(Temperature if isinstance(Temperature, torch.Tensor) else
                                  (float(Temperature) if Temperature is not None else None)))
    dev = device if device is not None else str(g.device)
    grid_sym = 'auto' if GridSymmetry in (None, '', 'auto') else GridSymmetry
    opt = Options(Verbosity=0, GridSize=list(GridSize) if isinstance(GridSize, (list, tuple)) else int(GridSize),
                  GridSymmetry=grid_sym, device=dev)
    B, spec = pepper_autograd(sys, exp, opt)
    return B.to(g.device), spec.to(device=g.device, dtype=g.dtype)


def differentiable_spectrum(
    g: torch.Tensor,
    lw_mT: float | torch.Tensor,
    mwFreq_GHz: float,
    B_range: tuple[float, float],
    nPoints: int = 1024,
    Harmonic: int = 1,
    GridSize: int | list[int] = 31,
    GridSymmetry: str = 'auto',
    device: Optional[str] = None,
    S: float = 0.5,
    Nucs: Optional[list[str]] = None,
    A: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None,
    Temperature: Optional[float | torch.Tensor] = None,
    method: str = 'pepper',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fully differentiable powder CW EPR spectrum.

    Parameters
    ----------
    g:
        Principal g-values ``[gx, gy, gz]`` (shape ``(3,)``); ``requires_grad=True``
        enables gradients.
    lw_mT:
        Gaussian line width (FWHM, mT); a scalar tensor with ``requires_grad``
        enables line-width gradients.
    mwFreq_GHz, B_range, nPoints, Harmonic:
        Spectrometer frequency (GHz), sweep ``(B_min, B_max)`` in mT, number of
        points, detection harmonic (0, 1, 2).
    GridSize:
        pepper's ``Options.GridSize``: an int ``N`` (knots along the quarter meridian,
        projection at the knots) or ``[N, Ninterp]`` (EasySpin grid interpolation).
        Default 31.  For ``method='broadband'``/``'analytical'`` only an int is used.
    GridSymmetry:
        ``'auto'`` (from the Hamiltonian, default) or an explicit symmetry
        (``'D2h'``, ``'Dinfh'``, ``'Ci'`` ...).
    device:
        Torch device; defaults to ``g.device``.
    S, Nucs, A, D:
        Electron spin, nuclear isotopes, hyperfine tensors (``(nNuclei, 3)`` or
        ``(3*nNuclei, 3)``, MHz) and zero-field splitting (``[D, E]``, ``D`` or
        ``[Dx, Dy, Dz]``, MHz).  ``A`` and ``D`` may ``requires_grad``.
    Temperature:
        Temperature (K) for Boltzmann populations; ``None`` = infinite.  A scalar
        tensor with ``requires_grad`` enables temperature gradients.
    method:
        ``'pepper'`` (default): :func:`torchspin.pepper_autograd.pepper_autograd`,
        identical to :func:`torchspin.pepper` in the forward direction.
        ``'broadband'`` / ``'analytical'``: the earlier stand-alone models
        (deprecated; orientation-grid ripple, see the module docstring).

    Returns
    -------
    B_axis, spec : torch.Tensor
        Field axis (mT) and spectrum on the autograd graph.
    """
    import warnings
    if g.dim() != 1 or g.shape[0] != 3:
        raise ValueError("g must be a 1-D tensor of shape (3,) = [gx, gy, gz].")
    if float(torch.as_tensor(lw_mT).detach()) <= 0:
        raise ValueError("lw_mT must be positive.")
    if Harmonic not in (0, 1, 2):
        raise ValueError("Harmonic must be 0, 1, or 2.")
    for _name, _t in (('A', A), ('D', D)):
        if _t is not None and hasattr(_t, 'device') and str(_t.device) != str(g.device):
            raise ValueError(
                f"{_name} is on device '{_t.device}' but g is on '{g.device}'. "
                f"Move all tensors to the same device before calling differentiable_spectrum().")
    if method == 'pepper':
        return _pepper_autograd_spectrum(g, lw_mT, mwFreq_GHz, B_range, nPoints, Harmonic, GridSize, GridSymmetry,
                                         device, S, Nucs, A, D, Temperature)
    if method not in ('broadband', 'analytical'):
        raise ValueError("method must be 'pepper', 'broadband' or 'analytical'.")
    warnings.warn(f"differentiable_spectrum(method='{method}') is deprecated: it is a separate forward model with "
                  "orientation-grid ripple; use method='pepper' (default).", DeprecationWarning, stacklevel=2)
    gs = int(GridSize[0]) if isinstance(GridSize, (list, tuple)) else int(GridSize)
    gsym = 'D2h' if GridSymmetry in (None, '', 'auto') and method == 'analytical' else GridSymmetry
    if method == 'broadband' and Nucs is None and D is None and S == 0.5:
        # force the N-spin model for a bare S=1/2 system
        return _broadband_spectrum(_bare_system(g), mwFreq_GHz, B_range, nPoints, Harmonic, lw_mT, gs,
                                   gsym, Temperature, device if device is not None else str(g.device), g.dtype)
    return _legacy_differentiable_spectrum(g, lw_mT, mwFreq_GHz, B_range, nPoints, Harmonic, gs, gsym, device,
                                           S, Nucs, A, D, Temperature)


def _bare_system(g: torch.Tensor):
    from torchspin.spinsystem import SpinSystem
    return SpinSystem(S=[0.5], g=g.unsqueeze(0) if g.dim() == 1 else g)
