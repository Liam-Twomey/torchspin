"""
garlic - Fast-motion and isotropic solution EPR simulation

MATLAB equivalent: easyspin/garlic.m

Simulates CW-EPR spectra for S=1/2 systems in solution:
- Isotropic mode (tcorr=0): isotropic g and A, sharp lines broadened by
  ``sys.lw`` / ``sys.lwpp``
- Fast-motion mode (tcorr>0): Kivelson/Freed motional-narrowing linewidths
  per hyperfine line

Line positions follow EasySpin's per-nuclear-group algorithm (garlic.m):
each entry of ``Nucs`` (optionally a set of ``n`` equivalent nuclei, reduced
with ``equivcouple``) contributes a list of resonance shifts from either the
exact Breit–Rabi solution (``Options.Method='exact'``, default; fixed-point
iteration after J. A. Weil, J. Magn. Reson. 4, 394 (1971)) or a Taylor
expansion of it in the hyperfine coupling (``'perturb1'``–``'perturb5'``,
``'perturb'`` = 5th order, Newton–Raphson root finding).  The shift lists of
all groups are then combined additively.

Perturbation theory converges slowly when the hyperfine coupling is a sizable
fraction of the microwave frequency. For isotropic 91Zr with A = 780 MHz at
9.65 GHz (A/nu = 0.08), the relative L2 error against ``'exact'`` is 0.82
(perturb2), 0.11 (perturb3), 0.018 (perturb4) and 0.0032 (perturb5). Use
``'exact'`` for large A/nu. ``'exact'`` is also the default.

The default ``AccumMethod`` for the isotropic limit is ``'binning'`` (nearest
field bin, as in EasySpin). It is piecewise-constant in the line positions,
so it has no finite-difference derivative with respect to g or A. Use
``'linear'`` (two-bin split) or ``'explicit'`` in a model that is fitted with
a gradient-based optimizer. On the autograd graph, garlic switches to
``'linear'`` itself.

Usage:
    B, spec = garlic(sys, exp)
    B, spec = garlic(sys, exp, opt)
    nu, spec = garlic(sys, exp)   # frequency sweep (exp.Field + exp.mwRange)

Parameters:
    sys : SpinSystem
        S = [1/2] (one electron spin, must be 1/2)
        g : isotropic value, principal values or full matrix (averaged)
        Nucs : str / list — nuclear isotopes, e.g. '14N' or '1H,1H'
        n : list[int], optional — number of equivalent nuclei per Nucs entry
        A : hyperfine couplings [MHz] (only the isotropic part is used)
        lw / lwpp : [GaussianFWHM, LorentzianFWHM]; mT for field sweeps,
            MHz for frequency sweeps
        tcorr / logtcorr : rotational correlation time (fast-motion regime)

    exp : Experiment
        Field sweep: mwFreq [GHz] + Range or CenterSweep [mT]
        Frequency sweep: Field [mT] + mwRange or mwCenterSweep [GHz]
        nPoints, Harmonic (0/1/2), mwPhase [rad], Temperature [K],
        ModAmp [mT] (field modulation; requires Harmonic ≥ 1),
        Mode ('perpendicular' / 'parallel')

    opt : Options, optional
        Method : 'exact' (default) | 'perturb' | 'perturb1'..'perturb5'
        AccumMethod : None (auto) | 'binning' | 'linear' | 'explicit'
        Accuracy, MaxIterations : solver controls
        Verbosity : 0 (silent), 1 (normal)

Returns:
    x : torch.Tensor, shape (nPoints,)
        Field axis [mT] (field sweep) or frequency axis [GHz] (frequency sweep)
    spec : torch.Tensor, shape (nPoints,)
        Simulated spectrum (EasySpin intensity convention)

Reference:
    EasySpin documentation: https://easyspin.org/easyspin/documentation/garlic.html
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.constants import BMAGN, PLANCK, NMAGN, BOLTZMANN
from torchspin.convspec import convspec
from torchspin.lineshape import lorentzian as _lorentzian
from torchspin.dataproc import fieldmod as _fieldmod
from torchspin.utils import equivcouple
from torchspin.fastmotion import fastmotion_t as _fastmotion_t

__all__ = ['garlic']


_PERTURB_ORDER = {
    'exact': 0, 'perturb': 5,
    'perturb1': 1, 'perturb2': 2, 'perturb3': 3, 'perturb4': 4, 'perturb5': 5,
}


def garlic(
    sys_or_list,
    exp: Experiment,
    opt: Optional[Options] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solution (fast-motion or isotropic) CW EPR simulation.

    Accepts a single :class:`SpinSystem` or a list of components (EasySpin
    ``compisoloop``): every component is expanded into its isotopologues
    (``'B'``, ``'(63,65)Cu'`` + ``Abund``), simulated separately and summed with
    ``weight × abundance``; ``Options.separate='components'`` returns one row per
    component/isotopologue.  The sweep range may be omitted (``Range``/
    ``CenterSweep`` for field sweeps, ``mwRange`` for frequency sweeps with
    ``Field``): it is then determined automatically from the line positions and
    widths as in EasySpin, which is only possible for a single (isotopologue)
    component.  See :func:`_garlic_single` for the line engine.
    """
    opt_ = opt if opt is not None else Options()
    from torchspin.experiment import require_positive_field
    require_positive_field(exp)          # Field = 0 is a pepper/salt-only option
    from torchspin.isotopologues import expand_components
    components = expand_components(sys_or_list, getattr(opt_, 'IsoCutoff', 1e-4))
    freq_sweep = exp.is_freq_swept or (exp.Field is not None and exp.mwFreq is None)
    auto_range = (exp.mwRange is None) if freq_sweep else (exp.Range is None)
    if auto_range and len(components) > 1:
        raise ValueError('For multiple components, the sweep range cannot be determined automatically. '
                         + ('Specify Exp.mwRange or Exp.mwCenterSweep.' if freq_sweep
                            else 'Specify Exp.Range or Exp.CenterSweep.'))
    rows, x_out = [], None
    for s_ in components:
        x_out, y_ = _garlic_single(s_, exp, opt_)
        rows.append(y_ * float(getattr(s_, 'weight', 1.0)))
    if str(getattr(opt_, 'separate', '')) == 'components':
        return x_out, torch.stack(rows)
    return x_out, sum(rows)


def _garlic_single(
    sys: SpinSystem,
    exp: Experiment,
    opt: Optional[Options] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solution (fast-motion or isotropic) CW EPR simulation.

    Computes the isotropic-averaged EPR spectrum for an S=1/2 spin system in
    solution. When ``sys.tcorr`` (or ``sys.logtcorr``) is set, Kivelson/Freed
    fast-motion linewidth theory is applied per hyperfine line.

    Parameters
    ----------
    sys : SpinSystem
        Spin system. Only the isotropic parts of g and A enter the line
        positions; the anisotropic parts enter the fast-motion linewidths.
        ``sys.n`` gives the number of equivalent nuclei per ``Nucs`` entry.
    exp : Experiment
        Field sweep: ``mwFreq`` (GHz) and ``Range``/``CenterSweep`` (mT).
        Frequency sweep: ``Field`` (mT) and ``mwRange``/``mwCenterSweep``
        (GHz). Also ``nPoints``, ``Harmonic``, ``mwPhase``, ``Temperature``,
        ``ModAmp`` (field modulation, mT) and ``Mode``.
    opt : Options, optional
        ``Method`` (``'exact'`` default, or ``'perturbN'``), ``AccumMethod``,
        ``Accuracy``, ``MaxIterations``, ``Verbosity``.

    Returns
    -------
    x : torch.Tensor, shape (nPoints,)
        Field axis in mT (field sweep) or frequency axis in GHz (frequency
        sweep).
    spec : torch.Tensor, shape (nPoints,)
        Simulated spectrum.

    Examples
    --------
    Nitroxide triplet in solution::

        >>> from torchspin import SpinSystem, Experiment, garlic
        >>> sys = SpinSystem(S=[0.5], g=2.006, Nucs='14N', A=[16, 16, 95],
        ...                  tcorr=1e-10, lw=[0.1, 0.0])
        >>> exp = Experiment(mwFreq=9.5, Range=[336, 342], nPoints=1024, Harmonic=1)
        >>> B, spc = garlic(sys, exp)

    Methyl radical with three equivalent protons, second-order perturbation::

        >>> sys = SpinSystem(S=[0.5], g=2.0026, Nucs='1H', n=[3], A=65, lw=[0, 0.1])
        >>> exp = Experiment(mwFreq=9.7, CenterSweep=[346, 12])
        >>> B, spc = garlic(sys, exp, Options(Method='perturb2'))

    Notes
    -----
    * For slow-motion regimes (tcorr > ~1e-8 s), use :func:`chili` instead.
    * For rigid-limit / powder spectra, use :func:`pepper`.
    * Line positions are the exact Breit–Rabi solution by default (EasySpin
      default); MATLAB-validated to cosine similarity > 0.999.
    """
    if opt is None:
        opt = Options()
    verbose = opt.Verbosity >= 1
    dt = torch.float64

    # -----------------------------------------------------------------------
    # Step 1: Validate spin system and dynamics regime
    # -----------------------------------------------------------------------
    _validate_system(sys)

    if sys.logtcorr is not None:
        tcorr = 10.0 ** torch.as_tensor(sys.logtcorr, dtype=dt).reshape(-1)[0]
    elif sys.tcorr is not None:
        tcorr = torch.as_tensor(sys.tcorr, dtype=dt).reshape(-1)[0]
    else:
        tcorr = torch.zeros((), dtype=dt)
    tcorr_f = float(tcorr.detach())
    fast_motion = tcorr_f > 0
    if fast_motion:
        if tcorr_f < 1e-13:
            raise ValueError(f"Correlation time too small: {tcorr_f} s")
        if tcorr_f > 1e-3:
            raise ValueError(f"Correlation time too large for fast-motion regime: {tcorr_f} s")
        if any(v > 1 for v in sys.n):
            raise ValueError(
                "Cannot treat equivalent nuclei in fast-motion regime! "
                "Please specify each nucleus separately (e.g., Nucs='1H,1H' instead of n=[2])"
            )
    on_graph = any(t is not None and torch.is_tensor(t) and t.requires_grad
                   for t in (sys.g, sys.A, sys.Q, sys.tcorr, sys.logtcorr,
                             *(sys.lw if isinstance(sys.lw, (list, tuple)) else [sys.lw]),
                             *(sys.lwpp if isinstance(sys.lwpp, (list, tuple)) else [sys.lwpp])))
    # NB: a differentiable spectrum uses AccumMethod 'linear' (or 'explicit') below

    # -----------------------------------------------------------------------
    # Step 2: Experiment parameters
    # -----------------------------------------------------------------------
    field_sweep = not (exp.is_freq_swept or (exp.Field is not None and exp.mwFreq is None))
    if field_sweep:
        if exp.mwFreq is None:
            raise ValueError("exp.mwFreq is required for field sweeps")
        auto_range = exp.Range is None
        sweep_range = None if auto_range else [float(exp.Range[0]), float(exp.Range[1])]
    else:
        auto_range = exp.mwRange is None
        sweep_range = None if auto_range else [float(exp.mwRange[0]), float(exp.mwRange[1])]
    n_points = int(exp.nPoints)
    harmonic = int(exp.Harmonic) if exp.Harmonic is not None else (1 if field_sweep else 0)
    if harmonic not in (0, 1, 2):
        raise ValueError("exp.Harmonic must be 0, 1 or 2.")
    mod_amp = float(exp.ModAmp or 0.0)
    if mod_amp > 0:
        if not field_sweep:
            raise ValueError("exp.ModAmp cannot be used with frequency sweeps.")
        if harmonic < 1:
            raise ValueError("With field modulation (exp.ModAmp), exp.Harmonic=0 does not work.")
        mod_harmonic = harmonic
        conv_harmonic = 0
    else:
        mod_harmonic = 0
        conv_harmonic = harmonic
    mode = getattr(exp, 'Mode', 'perpendicular') or 'perpendicular'
    if mode not in ('perpendicular', 'parallel'):
        raise ValueError("exp.Mode must be 'perpendicular' or 'parallel'.")
    parallel_mode = mode == 'parallel'
    method = opt.Method if opt.Method and opt.Method != 'matrix' else 'exact'
    if method not in _PERTURB_ORDER:
        raise ValueError(f"Unknown method '{method}'. Use 'exact', 'perturb' or 'perturb1'..'perturb5'.")
    perturb_order = _PERTURB_ORDER[method]
    accum_method = opt.AccumMethod or ('explicit' if fast_motion else 'binning')
    if accum_method == 'binning' and on_graph:
        # nearest-bin sticks cannot be differentiated in the line positions; the
        # two-bin linear split (EasySpin makespec) is the differentiable form
        accum_method = 'linear'

    # -----------------------------------------------------------------------
    # Step 3: Isotropic g and central resonance
    # -----------------------------------------------------------------------
    giso = _isotropic_g(sys)                       # 0-d tensor
    if field_sweep:
        mw_freq_hz = float(exp.mwFreq) * 1e9
        central = mw_freq_hz * PLANCK / (BMAGN * giso) * 1e3      # mT
    else:
        mw_freq_hz = None
        central = BMAGN * giso * float(exp.Field) * 1e-3 / PLANCK  # Hz
    if verbose:
        print(f"garlic: isotropic g = {float(giso):.6f}")
        if field_sweep:
            print(f"garlic: g={float(giso):g} resonance at {float(central):.4f} mT")
        else:
            print(f"garlic: g={float(giso):g} resonance at {float(central)/1e9:.6f} GHz")
        print(f"garlic: method '{method}', harmonic {harmonic}, {mode} mode")

    # -----------------------------------------------------------------------
    # Step 4: Fast-motion linewidths (per hyperfine line)
    # -----------------------------------------------------------------------
    fm_lw = None
    fm_lookup = None
    if fast_motion:
        if field_sweep:
            fm_lw, mI_all = _fastmotion_t(sys, central, tcorr, domain='field')            # mT
        else:
            fm_lw, mI_all = _fastmotion_t(sys, float(exp.Field), tcorr, domain='freq')    # MHz
            fm_lw = fm_lw / 1e3                                                            # GHz
        fm_d = fm_lw.detach()
        if bool(torch.all(fm_d == 0)):
            raise ValueError(
                "All calculated fast-motion linewidths for rotational diffusion are zero. "
                "Please provide full tensors, and not just isotropic averages."
            )
        if bool(torch.any(fm_d <= 0)):
            raise ValueError("Some calculated fast-motion linewidths are negative. They must be positive.")
        mI_all = np.asarray(mI_all, dtype=float).reshape(fm_lw.shape[0], -1)
        fm_lookup = {tuple(np.round(row * 2).astype(int)): k for k, row in enumerate(mI_all)}
        if verbose:
            print(f"garlic: fast-motion linewidths min={float(fm_d.min()):.4g}, max={float(fm_d.max()):.4g}")

    # -----------------------------------------------------------------------
    # Step 5: Resonance line positions and multiplicities
    # -----------------------------------------------------------------------
    positions, multiplicities, mI_labels, (pos_min, pos_max) = _resonance_lines(
        sys, giso, central, field_sweep, mw_freq_hz,
        float(exp.Field) if not field_sweep else None,
        perturb_order, float(opt.Accuracy), int(opt.MaxIterations), verbose,
    )
    if not field_sweep:
        positions = positions / 1e9  # Hz -> GHz
    n_lines = positions.numel()
    if verbose:
        print(f"garlic: {n_lines} lines, spread {float(positions.max()-positions.min()):.4g}")

    # -----------------------------------------------------------------------
    # Step 6: Line intensities
    # -----------------------------------------------------------------------
    if parallel_mode:
        intensities = torch.zeros_like(multiplicities)
    else:
        transition_rate = (8 * math.pi ** 2) * giso ** 2 * (BMAGN / PLANCK / 1e9 / 2) ** 2
        intensities = multiplicities * transition_rate
        if field_sweep:
            dBdE = PLANCK / (giso * BMAGN) * 1e9  # mT/MHz
            intensities = intensities * dBdE
        if exp.Temperature is not None and np.isfinite(float(exp.Temperature)):
            if float(exp.Temperature) <= 0:
                raise ValueError("exp.Temperature must be positive (K).")
            if field_sweep:
                delta_e = PLANCK * mw_freq_hz
            else:
                delta_e = BMAGN * giso * float(exp.Field) * 1e-3
            e = torch.exp(-torch.as_tensor(delta_e, dtype=dt) / BOLTZMANN / float(exp.Temperature))
            polarization = (1.0 - e) / (1.0 + e)
            intensities = intensities * polarization

    # -----------------------------------------------------------------------
    # Step 7: Linewidths
    # -----------------------------------------------------------------------
    lw_list = sys.get_lw()
    if torch.is_tensor(lw_list):
        lw_arr = [v.to(dt) for v in lw_list.reshape(-1)]
    else:
        lw_arr = [torch.as_tensor(v, dtype=dt).reshape(-1)[0] for v in (lw_list if isinstance(lw_list, (list, tuple)) else [lw_list])]
    fwhm_g = lw_arr[0] if len(lw_arr) > 0 else torch.zeros((), dtype=dt)
    fwhm_l0 = lw_arr[1] if len(lw_arr) > 1 else torch.zeros((), dtype=dt)
    if float(fwhm_g.detach()) < 0 or float(fwhm_l0.detach()) < 0:
        raise ValueError("sys.lw entries must be non-negative.")
    if not field_sweep:
        fwhm_g = fwhm_g / 1e3   # MHz -> GHz
        fwhm_l0 = fwhm_l0 / 1e3
    if fast_motion:
        # Per-line Lorentzian: fast-motion width + residual homogeneous width
        idx = [fm_lookup[tuple(np.round(mI_labels[k] * 2).astype(int))] for k in range(n_lines)]
        lorentz_lw = fm_lw[torch.tensor(idx, dtype=torch.long)] + fwhm_l0
    else:
        lorentz_lw = fwhm_l0
    fg_f, fl_f = float(fwhm_g.detach()), float(fwhm_l0.detach())
    no_broadening = (not fast_motion) and fg_f == 0 and fl_f == 0
    if no_broadening and harmonic != 0:
        # EasySpin auto-selects Harmonic=0 when no broadening is given (and
        # errors if a derivative is requested explicitly). torchspin's
        # Experiment cannot tell "not given" from the default 1, so fall back
        # to the absorption stick spectrum like EasySpin's auto-harmonic.
        if mod_harmonic > 0:
            raise ValueError("No broadening given. Cannot apply field modulation to a stick spectrum.")
        if verbose:
            print(f"garlic: no broadening given; using Harmonic=0 instead of {harmonic}")
        harmonic = 0
        conv_harmonic = 0

    # -----------------------------------------------------------------------
    # Step 8: Spectrum construction
    # -----------------------------------------------------------------------
    if auto_range:
        # EasySpin garlic.m SweepAutoRange: line extremes ± max(5·maxLw, Stretch·spread, minrange)
        stretch = float(getattr(opt, 'Stretch', 0.25))
        lw_max_f = max(float(v.detach()) for v in lw_arr) if lw_arr else 0.0
        if fast_motion:
            fm_max = float(fm_lw.detach().max()) * (1e3 if not field_sweep else 1.0)   # MHz for frequency sweeps
            max_lw = max(fm_max, float(lw_arr[0].detach()))
        else:
            max_lw = lw_max_f
        if field_sweep:
            pad = max(5.0 * max_lw, (pos_max - pos_min) * stretch, 1.0)      # mT
            sweep_range = [max(pos_min - pad, 0.0), pos_max + pad]
        else:
            pad = max(5.0 * max_lw / 1e3, (pos_max - pos_min) * stretch, 1e6)  # Hz (as EasySpin)
            sweep_range = [max(pos_min - pad, 0.0) / 1e9, (pos_max + pad) / 1e9]
        if verbose:
            print(f"garlic: automatic sweep range {sweep_range[0]:.6g} to {sweep_range[1]:.6g}")
    x_axis = torch.linspace(sweep_range[0], sweep_range[1], n_points, dtype=dt)
    dx = float(x_axis[1] - x_axis[0])
    mw_phase = float(getattr(exp, 'mwPhase', 0.0) or 0.0)
    from torchspin.pepper_autograd import convspec_t
    if accum_method == 'explicit':
        lor = torch.as_tensor(lorentz_lw, dtype=dt).reshape(-1)
        if bool(torch.any(lor.detach() <= 0)):
            raise ValueError("Cannot use explicit accumulation with zero Lorentzian linewidth.")
        dx_fine = min(dx, float(lor.detach().min()) / 5.0)
        n_fine = int(round((sweep_range[1] - sweep_range[0]) / dx_fine + 1))
        x_fine = torch.linspace(sweep_range[0], sweep_range[1], n_fine, dtype=dt)
        dx_fine = float(x_fine[1] - x_fine[0])
        if lor.numel() == 1:
            lor = lor.expand(n_lines)
        spec_fine = torch.zeros(n_fine, dtype=dt)
        nz = intensities.detach() != 0
        if bool(nz.any()):
            spec_fine = spec_fine + (intensities[nz].unsqueeze(1)
                                     * _lorentzian_lines_t(x_fine, positions[nz], lor[nz], conv_harmonic, mw_phase)).sum(dim=0)
        # Remaining Gaussian broadening is an absorption-shape convolution
        if fg_f > 0 and fg_f > 2 * dx_fine:
            spec_fine = convspec_t(spec_fine, dx_fine, fwhm_g, 0.0, deriv=0)
        spec = _interp1_t(x_axis, x_fine, spec_fine) if n_fine != n_points else spec_fine
    elif accum_method in ('binning', 'linear'):
        if fast_motion:
            raise ValueError("Cannot use delta binning (Options.AccumMethod='binning') in the fast-motion regime.")
        spec_t = _stick_spectrum(positions, intensities, sweep_range, n_points, verbose,
                                 linear=(accum_method == 'linear')) / dx
        fwhm_l = fwhm_l0
        # EasySpin skips convolution with a lineshape narrower than 2 increments
        # (it is delta-like); with a derivative harmonic it errors, torchspin's
        # FFT convolution copes, so the width is kept in that case.
        if fl_f > 0 and fl_f <= 2 * dx and conv_harmonic == 0:
            fwhm_l = 0.0
        g_conv = fwhm_g
        if fg_f > 0 and fg_f <= 2 * dx and conv_harmonic == 0:
            g_conv = 0.0
        if float(torch.as_tensor(g_conv).detach()) > 0 or float(torch.as_tensor(fwhm_l).detach()) > 0 or conv_harmonic > 0:
            spec_t = convspec_t(spec_t, dx, g_conv, fwhm_l, deriv=conv_harmonic)
        spec = spec_t
        if mw_phase != 0.0:
            raise NotImplementedError("exp.mwPhase != 0 requires Options.AccumMethod='explicit' in garlic.")
    else:
        raise ValueError(f"Unknown Options.AccumMethod '{accum_method}'.")

    # -----------------------------------------------------------------------
    # Step 9: Field modulation
    # -----------------------------------------------------------------------
    if mod_harmonic > 0:
        from torchspin.dataproc import fieldmod_t
        spec = fieldmod_t(x_axis, spec, mod_amp, mod_harmonic)
    return x_axis, spec


# ---------------------------------------------------------------------------
# Resonance line engine (EasySpin garlic.m, lines 520–760) — torch
# ---------------------------------------------------------------------------

def _resonance_lines(
    sys: SpinSystem,
    giso: torch.Tensor,
    central: torch.Tensor,
    field_sweep: bool,
    mw_freq_hz: Optional[float],
    field_mT: Optional[float],
    perturb_order: int,
    accuracy: float,
    max_iter: int,
    verbose: bool,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, tuple]:
    """Line positions (mT or Hz), relative multiplicities, mI labels and the
    (posmin, posmax) extremes used by EasySpin's automatic sweep range.

    Returns
    -------
    positions : (nLines,) tensor
        Field sweep: mT.  Frequency sweep: Hz.  On the autograd graph.
    multiplicities : (nLines,) tensor
        Product over nuclear groups of (multiplicity of F)/(2I+1)^n; sums to 1.
    mI_labels : (nLines, nNuclei) ndarray
        mI of each group for each line (NaN for n>1 groups, where the label
        is the effective-spin projection and not a single-nucleus mI).
    """
    dt = torch.float64
    n_nuc = sys.nNuclei
    if n_nuc == 0:
        c = torch.as_tensor(central, dtype=dt).reshape(1)
        return (c, torch.ones(1, dtype=dt), np.zeros((1, 0)), (float(c.detach()), float(c.detach())))
    I_all = np.array(sys.I, dtype=float)
    gn_all = np.array(sys.gn, dtype=float)
    if sys.gnscale is not None:
        gn_all = gn_all * np.asarray(sys.gnscale, dtype=float).reshape(-1)[:n_nuc]
    a_all = _isotropic_A_hz(sys)   # Hz, one per nucleus (tensor)
    n_all = np.array(sys.n, dtype=int)
    if np.any(I_all == 0):
        raise ValueError("Nuclei with spin 0 present. Cannot compute spectrum.")
    if bool(torch.any(a_all.detach() == 0)):
        raise ValueError("Nuclei with hyperfine coupling 0 present. Cannot compute spectrum.")
    if field_sweep:
        zf_splitting = float((a_all.detach().abs() * torch.as_tensor(I_all + 0.5, dtype=dt)).sum())
        if mw_freq_hz < zf_splitting and verbose:
            print(f"garlic: microwave frequency ({mw_freq_hz/1e9:.4f} GHz) is smaller than the "
                  f"largest zero-field splitting estimate ({zf_splitting/1e9:.4f} GHz); "
                  "spectrum might be inaccurate. Consider pepper().")
    a_all_J = a_all * PLANCK
    shifts_per_group: list[torch.Tensor] = []
    amps_per_group: list[torch.Tensor] = []
    labels_per_group: list[np.ndarray] = []
    for i_grp in range(n_nuc):
        I_nuc = float(I_all[i_grp])
        n_eq = int(n_all[i_grp])
        aiso = a_all_J[i_grp].abs()               # J; sign is irrelevant for the spectrum
        gn = float(gn_all[i_grp])
        F_list, mult_list = equivcouple(I_nuc, n_eq)
        n_lines_grp = (2 * I_nuc + 1) ** n_eq
        pos_list, amp_list, lab_list = [], [], []
        for F, mult in zip(F_list, mult_list):
            if mult == 0:
                continue
            if field_sweep:
                if perturb_order == 0:
                    pos = _breit_rabi_field(F, aiso, giso, gn, mw_freq_hz, accuracy, max_iter)
                else:
                    pos = _perturb_field(F, aiso, giso, gn, mw_freq_hz, perturb_order, accuracy, max_iter)
                pos = pos * 1e3  # T -> mT
            else:
                B_T = field_mT * 1e-3
                if perturb_order == 0:
                    pos = _breit_rabi_freq(F, aiso, giso, gn, B_T)
                else:
                    pos = _perturb_freq(F, aiso, giso, gn, B_T, perturb_order)
            mI = np.arange(-F, F + 0.5, 1.0)
            pos_list.append(pos)
            amp_list.append(torch.full((pos.numel(),), float(mult) / n_lines_grp, dtype=dt))
            lab_list.append(mI if n_eq == 1 else np.full(pos.numel(), np.nan))
        shifts_per_group.append(torch.cat(pos_list) - central)
        amps_per_group.append(torch.cat(amp_list))
        labels_per_group.append(np.concatenate(lab_list))
        if verbose:
            print(f"garlic: spin group {i_grp+1}: {len(F_list)} F spins, {shifts_per_group[-1].numel()} lines")

    # Combine all groups (EasySpin allcombinations '+' / '*')
    # EasySpin auto-range extremes: central + sum over groups of min/max shift
    c_f = float(central.detach())
    pos_min = c_f + sum(float(sh.detach().min()) for sh in shifts_per_group)
    pos_max = c_f + sum(float(sh.detach().max()) for sh in shifts_per_group)
    positions = torch.as_tensor(central, dtype=dt).reshape(1)
    amps = torch.ones(1, dtype=dt)
    labels = np.zeros((1, 0))
    for sh, am, lb in zip(shifts_per_group, amps_per_group, labels_per_group):
        positions = (positions[:, None] + sh[None, :]).reshape(-1)
        amps = (amps[:, None] * am[None, :]).reshape(-1)
        n_old, n_new = labels.shape[0], sh.numel()
        labels = np.concatenate(
            [np.repeat(labels, n_new, axis=0), np.tile(lb, n_old)[:, None]], axis=1
        )
    return positions, amps, labels, (pos_min, pos_max)


def _breit_rabi_field(F, aiso, giso, gn, mw_freq_hz, accuracy, max_iter):
    """Resonance fields (T) of an effective spin F: fixed-point Breit–Rabi iteration
    in torch (the converged iteration carries the implicit gradient)."""
    dt = torch.float64
    gammae = giso * BMAGN
    h = PLANCK
    mI = torch.arange(-F, F + 0.5, 1.0, dtype=dt)
    B_ = torch.zeros_like(mI)
    gamman = 0.0  # neglect nuclear Zeeman for the start field
    first = True
    converged = False
    for _ in range(max_iter):
        nu = mw_freq_hz + gamman / h * B_
        q = 1.0 - (aiso / (2 * h * nu)) ** 2
        arg = mI ** 2 + q * ((h * nu / aiso) ** 2 - (F + 0.5) ** 2)
        if bool(torch.any(arg.detach() < 0)):
            raise ValueError(
                "Hyperfine coupling too large compared to microwave frequency. "
                "Cannot compute resonance field positions. Use pepper() instead."
            )
        B_new = aiso / (gammae + gamman) / q * (-mI + torch.sqrt(arg))
        if not first:
            rel_change = 1.0 - B_new.detach() / B_.detach()
            converged = bool(torch.all(rel_change.abs() < accuracy))
        B_ = B_new
        gamman = gn * NMAGN  # re-include nuclear Zeeman
        first = False
        if converged:
            break
    if not converged:
        raise ValueError(f"Breit-Rabi solver didn't converge after {max_iter} iterations!")
    return B_


def _perturb_coeffs(F, mI, aiso, pre):
    """Taylor-expansion coefficients c3..c6 (J) of the Breit–Rabi energy difference."""
    I = F
    c3 = pre[0] * (I * (I + 1) - mI ** 2)
    c4 = pre[1] * mI * (1 - 2 * I * (I + 1) + 2 * mI ** 2)
    c5 = pre[2] * (I - 2 * I ** 3 - I ** 4 + 6 * (I ** 2 + I - 1) * mI ** 2 - 5 * mI ** 4)
    c6 = pre[3] * mI * (1 + 6 * (I - 1) * I * (I + 1) * (I + 2)
                        - 10 * (-3 + 2 * I * (I + 1)) * mI ** 2 + 14 * mI ** 4)
    return c3, c4, c5, c6


def _polyval_t(c, x):
    out = c[0] * torch.ones_like(x) if torch.is_tensor(x) else c[0]
    for ck in c[1:]:
        out = out * x + ck
    return out


def _perturb_field(F, aiso, giso, gn, mw_freq_hz, order, accuracy, max_iter):
    """Resonance fields (T) from an order-N expansion in aiso, Newton–Raphson (torch)."""
    dt = torch.float64
    pre = [aiso * (aiso / (giso * BMAGN + gn * NMAGN) / 2.0) ** k for k in range(1, 5)]
    out = []
    for mI in np.arange(-F, F + 0.5, 1.0):
        c3, c4, c5, c6 = _perturb_coeffs(F, float(mI), aiso, pre)
        c = [giso * BMAGN, aiso * float(mI) - mw_freq_hz * PLANCK, c3, c4, c5, c6]
        n = order + 1
        c = c[:n]
        dc = [(n - 1 - k) * c[k] for k in range(n - 1)]
        Bb = -c[1] / c[0]  # first-order start guess
        dB = Bb
        converged = False
        for _ in range(max_iter):
            if abs(float((dB / Bb).detach())) < accuracy:
                converged = True
                break
            dB = _polyval_t(c, Bb) / _polyval_t(dc, Bb)
            Bb = Bb - dB
        if not converged:
            raise ValueError("Newton-Raphson has convergence problem!")
        out.append(torch.as_tensor(Bb, dtype=dt).reshape(()))
    return torch.stack(out)


def _breit_rabi_freq(F, aiso, giso, gn, B_T):
    """Resonance frequencies (Hz) of an effective spin F at fixed field: Breit–Rabi formula."""
    dt = torch.float64
    gebB = giso * BMAGN * B_T
    gnbB = gn * NMAGN * B_T
    mI = torch.arange(-F, F + 0.5, 1.0, dtype=dt)
    alpha = (gebB + gnbB) / aiso / (F + 0.5)
    E1 = (-aiso / 4 - gnbB * (mI + 0.5)
          + (F + 0.5) * aiso / 2 * torch.sqrt(1 + 2 * (mI + 0.5) / (F + 0.5) * alpha + alpha ** 2))
    E2 = (-aiso / 4 - gnbB * (mI - 0.5)
          - (F + 0.5) * aiso / 2 * torch.sqrt(1 + 2 * (mI - 0.5) / (F + 0.5) * alpha + alpha ** 2))
    return (E1 - E2) / PLANCK


def _perturb_freq(F, aiso, giso, gn, B_T, order):
    """Resonance frequencies (Hz) from an order-N expansion in aiso at fixed field."""
    dt = torch.float64
    pre = [aiso * (aiso / (giso * BMAGN * B_T + gn * NMAGN * B_T) / 2.0) ** k for k in range(1, 5)]
    out = []
    for mI in np.arange(-F, F + 0.5, 1.0):
        c3, c4, c5, c6 = _perturb_coeffs(F, float(mI), aiso, pre)
        c = [giso * BMAGN * B_T, aiso * float(mI), c3, c4, c5, c6]
        out.append(sum(c[:order + 1]) / PLANCK)
    return torch.stack([torch.as_tensor(v, dtype=dt).reshape(()) for v in out])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stick_spectrum(positions, amplitudes, sweep_range, n_points, verbose=False, linear=False) -> torch.Tensor:
    """EasySpin ``constructstickspectrum``: each line goes into its nearest bin
    (``linear=False``, bit-comparable with EasySpin's garlic output) or is split
    linearly between the two neighboring bins (``linear=True``, EasySpin
    ``makespec``; differentiable in the line positions).
    """
    dt = torch.float64
    positions = torch.as_tensor(positions, dtype=dt).reshape(-1)
    amplitudes = torch.as_tensor(amplitudes, dtype=dt).reshape(-1)
    frac = (positions - sweep_range[0]) / (sweep_range[1] - sweep_range[0]) * (n_points - 1)
    spec = torch.zeros(n_points, dtype=dt)
    if not linear:
        idx = torch.round(frac.detach()).long()
        in_range = (idx >= 0) & (idx < n_points)
        if verbose and bool(torch.any(~in_range)):
            print("garlic: ** Spectrum exceeds sweep range. Artifacts at the limits possible.")
        return spec.index_add(0, idx[in_range], amplitudes[in_range])
    lo = torch.floor(frac.detach()).long()
    w_hi = frac - lo.to(dt)
    hi = lo + 1
    m_lo = (lo >= 0) & (lo < n_points)
    m_hi = (hi >= 0) & (hi < n_points)
    if verbose and bool(torch.any(~(m_lo & m_hi))):
        print("garlic: ** Spectrum exceeds sweep range. Artifacts at the limits possible.")
    spec = spec.index_add(0, lo[m_lo], (amplitudes * (1.0 - w_hi))[m_lo])
    return spec.index_add(0, hi[m_hi], (amplitudes * w_hi)[m_hi])


def _lorentzian_lines_t(x: torch.Tensor, x0: torch.Tensor, fwhm: torch.Tensor, diff: int, phase: float) -> torch.Tensor:
    """Area-normalized Lorentzian absorption (phase-rotated with dispersion) for a
    batch of lines: ``(nLines, nX)``.  Torch port of ``lineshape.lorentzian``."""
    gamma = (fwhm / math.sqrt(3.0)).unsqueeze(1)
    pre = 2.0 / (math.pi * math.sqrt(3.0))
    z = (x.unsqueeze(0) - x0.unsqueeze(1)) / gamma
    denom = 1.0 + (4.0 / 3.0) * z ** 2
    if diff == 0:
        yabs = pre / gamma / denom
        ydisp = pre ** 2 * math.pi / gamma * z / denom
    elif diff == 1:
        yabs = -8.0 / 3.0 * pre / gamma ** 2 * z / denom ** 2
        ydisp = pre ** 2 * math.pi / gamma ** 2 * (1.0 - 4.0 / 3.0 * z ** 2) / denom ** 2
    elif diff == 2:
        yabs = 8.0 / 3.0 * pre / gamma ** 3 * (4.0 * z ** 2 - 1.0) / denom ** 3
        ydisp = (pre ** 2 * math.pi / gamma ** 3 * 2.0 * (4.0 / 3.0) * z * (4.0 / 3.0 * z ** 2 - 3.0) / denom ** 3)
    else:
        raise ValueError("diff must be 0, 1 or 2")
    if phase != 0.0:
        return yabs * math.cos(phase) + ydisp * math.sin(phase)
    return yabs


def _interp1_t(x_new: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``np.interp`` for a uniformly spaced ``x`` (torch, differentiable in ``y``)."""
    n = x.numel()
    dxf = float(x[1] - x[0])
    f = (x_new - x[0]) / dxf
    lo = torch.clamp(torch.floor(f).long(), 0, n - 2)
    w = (f - lo.to(y.dtype)).clamp(0.0, 1.0)
    return y[lo] * (1.0 - w) + y[lo + 1] * w


def _validate_system(sys: SpinSystem):
    """Validate that system is compatible with garlic (S=1/2 only)."""
    if sys.nElectrons != 1 or abs(sys.Spins[0] - 0.5) > 1e-10:
        raise ValueError("garlic only supports systems with one electron spin S=1/2")

    if sys.ee is not None:
        ee_tensor = torch.as_tensor(sys.ee)
        if torch.any(ee_tensor != 0):
            raise ValueError("garlic does not support electron-electron coupling (Sys.ee)")

    if sys.nn is not None:
        nn_tensor = torch.as_tensor(sys.nn)
        if torch.any(nn_tensor != 0):
            raise ValueError("garlic does not support nucleus-nucleus coupling (Sys.nn)")

    if getattr(sys, 'Ham', None):
        raise ValueError("garlic does not support Sys.Ham* parameters!")


def _isotropic_g(sys: SpinSystem) -> torch.Tensor:
    """Isotropic g: mean of principal values (or of the eigenvalues of a full matrix); 0-d tensor."""
    g = torch.as_tensor(sys.g, dtype=torch.float64).reshape(-1)
    if g.numel() == 1:
        return g[0]
    if g.numel() == 3:
        return g.mean()
    if g.numel() == 9:
        return torch.linalg.eigvals(g.reshape(3, 3)).real.mean()
    raise ValueError(f"Invalid g shape: {tuple(sys.g.shape)}")


def _isotropic_A_hz(sys: SpinSystem) -> torch.Tensor:
    """Isotropic hyperfine coupling per nucleus in Hz (mean of principal values / eigenvalues)."""
    n_n = sys.nNuclei
    A = torch.as_tensor(sys.A, dtype=torch.float64)
    if A.ndim == 1:
        A = A.reshape(n_n, -1)
    if A.shape[0] == n_n:
        # Principal values (n_n, 3*nElectrons); single electron → first 3 columns
        a_iso = A[:, :3].mean(dim=1)
    elif A.shape[0] == 3 * n_n:
        a_iso = torch.stack([
            torch.linalg.eigvals(A[3 * i:3 * (i + 1), :3]).real.mean() for i in range(n_n)
        ])
    else:
        raise ValueError(f"Invalid A shape: {tuple(A.shape)}")
    return a_iso * 1e6
