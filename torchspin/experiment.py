"""Experiment and Options dataclasses for torchspin.

Mirrors EasySpin's ``Exp`` and ``Opt`` structs for CW EPR powder simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class Experiment:
    """CW EPR experimental parameters.

    Field-swept mode (default)
    --------------------------
    Set ``mwFreq`` (GHz) and ``Range`` ([Bmin, Bmax] in mT).
    Returns field axis in mT.

    Frequency-swept mode
    --------------------
    Set ``Field`` (fixed field in mT) and ``mwRange`` ([nu_lo, nu_hi] in GHz).
    ``Range`` is not used; ``mwFreq`` is optional.
    Returns frequency axis in GHz.

    Parameters
    ----------
    mwFreq:
        Microwave frequency in GHz (field-swept mode).
    Range:
        Field sweep range ``[Bmin, Bmax]`` in mT (field-swept mode).
        Not required when ``mwRange`` or ``CenterSweep`` is set.
    CenterSweep:
        Alternative to ``Range``: ``[center, sweep_width]`` in mT
        (EasySpin ``Exp.CenterSweep``).  Converted to ``Range`` on
        construction.
    mwCenterSweep:
        Alternative to ``mwRange`` for frequency sweeps:
        ``[center, sweep_width]`` in GHz.  Converted to ``mwRange``.
    SampleFrame:
        Euler angles ``[alpha, beta, gamma]`` (rad) of the sample frame for a
        single-crystal simulation (EasySpin ``Exp.SampleFrame``). ``None`` =
        powder/isotropic sample. Used by :func:`chili` (and pepper crystals,
        PORT_SPEC_2 Phase 3).
    ModAmp:
        Peak-to-peak field-modulation amplitude in mT (EasySpin
        ``Exp.ModAmp``).  ``0`` (default) = no modulation; the detection
        harmonic is then produced by differentiating the lineshape.  When
        ``> 0``, :func:`garlic` convolves the absorption lineshape and
        applies pseudo-modulation (:func:`torchspin.dataproc.fieldmod`)
        at ``Harmonic``, which must be ≥ 1.  Field sweeps only.
    nPoints:
        Number of points in the spectrum.  Default 1024.
    Harmonic:
        Detection harmonic: 0 = absorption, 1 = first-derivative (default),
        2 = second-derivative.
    mwPhase:
        Detection phase in radians: 0 = pure absorption, pi/2 = dispersion.
    Temperature:
        Sample temperature in K.  ``None`` (default) = infinite temperature
        (equal Boltzmann populations for all levels).
    Mode:
        Microwave excitation geometry.

        - ``'perpendicular'`` (default): B1 ⊥ B0. Standard CW EPR — selects
          allowed transitions with Δm_S = ±1.
        - ``'parallel'``: B1 ∥ B0. Selects forbidden transitions with
          Δm_S = 0, useful for integer-spin half-field lines (e.g.
          triplets, S=1 systems).
    Field:
        Fixed static field in mT (frequency-swept mode).  ``None`` = field-swept.
    mwRange:
        Microwave frequency sweep range ``[nu_lo, nu_hi]`` in GHz
        (frequency-swept mode).  ``None`` = field-swept.
    ExciteWidth:
        Microwave excitation bandwidth (FWHM, MHz) for orientation
        selection in fixed-field experiments (e.g. ENDOR via ``salt``).
        ``None`` (default) = no orientation selection.

    Fixed-field mode (ENDOR)
    ------------------------
    Set ``Field`` only (``Range`` and ``mwRange`` both ``None``).  Used by
    ``salt`` for fixed-field ENDOR; ``mwFreq`` + ``ExciteWidth`` optionally
    enable orientation selection.
    """

    mwFreq: Optional[float] = None
    Range: Optional[list] = None
    CenterSweep: Optional[list] = None
    mwCenterSweep: Optional[list] = None
    ModAmp: float = 0.0
    SampleFrame: Optional[list] = None
    CrystalSymmetry: Optional[object] = None   # space group number/symbol or point group (EasySpin Exp.CrystalSymmetry)
    MolFrame: Optional[list] = None            # Euler angles sample->molecule (EasySpin Exp.MolFrame)
    SampleRotation: Optional[tuple] = None     # (axis letter or 3-vector, rho or list of rho) (EasySpin Exp.SampleRotation)
    nPoints: int = 1024
    Harmonic: Optional[int] = None  # detection harmonic; default 1 for field sweeps, 0 for frequency sweeps (EasySpin)
    mwPhase: float = 0.0
    Temperature: Optional[float] = None
    lightBeam: Optional[object] = None   # photoexcitation: '', 'perpendicular', 'parallel', 'unpolarized' or (k, alpha) (EasySpin Exp.lightBeam)
    lightScatter: float = 0.0            # isotropic (scattered-light) contribution to the photoselection weights
    Ordering: Optional[object] = None    # partial ordering: scalar lambda or callable f(alpha,beta,gamma) / f(beta) (EasySpin Exp.Ordering)
    Mode: str = 'perpendicular'
    mwMode: Optional[object] = None      # EasySpin Exp.mwMode: 'perpendicular', 'parallel', or (k, alpha|'unpolarized'|'circular+'|'circular-'); None → Mode
    Field: Optional[float] = None
    mwRange: Optional[list] = None
    ExciteWidth: Optional[float] = None

    @property
    def is_freq_swept(self) -> bool:
        """True when Field and mwRange are both set (frequency-swept mode)."""
        return self.Field is not None and self.mwRange is not None

    @property
    def is_fixed_field(self) -> bool:
        """True when only Field is set (fixed-field mode, e.g. ENDOR)."""
        return (self.Field is not None and self.mwRange is None
                and self.Range is None)

    def __post_init__(self) -> None:
        # CenterSweep / mwCenterSweep → Range / mwRange (EasySpin semantics)
        if self.CenterSweep is not None:
            if self.Range is not None:
                raise ValueError("Give either Experiment.Range or Experiment.CenterSweep, not both.")
            if len(self.CenterSweep) != 2 or self.CenterSweep[1] <= 0:
                raise ValueError("Experiment.CenterSweep must be [center, sweep_width] with width > 0.")
            c, w = float(self.CenterSweep[0]), float(self.CenterSweep[1])
            self.Range = [c - w / 2.0, c + w / 2.0]
        if self.mwCenterSweep is not None:
            if self.mwRange is not None:
                raise ValueError("Give either Experiment.mwRange or Experiment.mwCenterSweep, not both.")
            if len(self.mwCenterSweep) != 2 or self.mwCenterSweep[1] <= 0:
                raise ValueError("Experiment.mwCenterSweep must be [center, sweep_width] with width > 0.")
            c, w = float(self.mwCenterSweep[0]), float(self.mwCenterSweep[1])
            self.mwRange = [c - w / 2.0, c + w / 2.0]
        if self.ModAmp is None:
            self.ModAmp = 0.0
        if self.ModAmp < 0:
            raise ValueError("Experiment.ModAmp must be zero or a positive number (mT).")
        if self.is_freq_swept:
            # Frequency-swept mode validation
            if len(self.mwRange) != 2:
                raise ValueError("Experiment.mwRange must be [nu_lo, nu_hi] in GHz.")
            if self.mwRange[0] >= self.mwRange[1]:
                raise ValueError("Experiment.mwRange[1] must be larger than mwRange[0].")
            if self.Field <= 0:
                raise ValueError("Experiment.Field must be positive (mT).")
        elif self.is_fixed_field:
            # Fixed-field mode validation (ENDOR)
            if self.Field <= 0:
                raise ValueError("Experiment.Field must be positive (mT).")
        else:
            # Field-swept mode validation.  Range may be omitted (automatic sweep
            # range, EasySpin garlic/pepper SweepAutoRange); simulation functions
            # that cannot auto-range raise their own error.
            if self.Range is not None and len(self.Range) != 2:
                raise ValueError("Experiment.Range must be [Bmin, Bmax].")
            if self.Range is not None and self.Range[0] >= self.Range[1]:
                raise ValueError("Experiment.Range[1] must be larger than Range[0].")
            if self.mwFreq is None or self.mwFreq <= 0:
                raise ValueError("mwFreq must be positive (GHz) for field-swept mode.")
        if self.Harmonic is None:
            self.Harmonic = 0 if (self.Field is not None and self.mwFreq is None) else 1
        self.Harmonic = int(self.Harmonic)
        if self.Harmonic not in (0, 1, 2):
            raise ValueError("Harmonic must be 0, 1, or 2.")
        if self.nPoints < 2:
            raise ValueError("nPoints must be at least 2.")


VALID_METHODS = frozenset({
    'matrix', 'exact', 'hybrid', 'perturb',
    'perturb1', 'perturb2', 'perturb3', 'perturb4', 'perturb5',
})

# Perturbation order per method name.  The bare name 'perturb' means different
# orders in different EasySpin simulators: garlic's Breit-Rabi expansion goes to
# fifth order, while pepper's resonance-field solver is second order.  Anything
# pepper cannot deliver is rejected rather than silently run as second order.
GARLIC_PERTURB_ORDER = {
    'perturb': 5, 'perturb1': 1, 'perturb2': 2,
    'perturb3': 3, 'perturb4': 4, 'perturb5': 5,
}
PEPPER_PERTURB_ORDER = {'perturb': 2, 'perturb1': 1, 'perturb2': 2}


@dataclass
class Options:
    """Computational options for CW EPR powder simulation.

    Parameters
    ----------
    Method:
        Computational method.  Which names are accepted depends on the
        simulator:

        :func:`pepper` — ``'matrix'`` (default) exact diagonalization;
        ``'perturb'``/``'perturb2'`` second-order perturbation theory
        (``'perturb1'`` first order); ``'hybrid'`` exact core plus
        perturbational ligand nuclei, selected with ``HybridCoreNuclei``.
        ``'perturb3'``–``'perturb5'`` are not implemented here and raise rather
        than quietly running second order.

        :func:`garlic` — ``'exact'`` (Breit–Rabi fixed-point solver, EasySpin
        default) / ``'perturb'`` / ``'perturb1'``–``'perturb5'`` (perturbation
        theory of the given order; ``'perturb'`` = 5th order).  Note that the
        bare name means different orders in the two simulators, as in EasySpin.

        :func:`salt` ignores ``Method``: it always finds resonance fields by
        diagonalization and the ENDOR frequencies by first-order perturbation
        theory for one electron without quadrupole couplings, and by matrix
        diagonalization otherwise or when ``Enhancement=True``.
    HybridCoreNuclei:
        ``Method='hybrid'`` only.  1-based indices into ``Sys.Nucs`` of the
        nuclei to keep in the exactly diagonalized core; all electron spins are
        always in the core.  Empty (the default) makes every nucleus
        perturbational.  For Cu(II) with nitrogen ligands, ``[1]`` keeps the
        copper exact and treats the nitrogens perturbationally.
    HybridIntThreshold:
        ``Method='hybrid'`` only.  Nuclear sub-lines whose amplitude summed over
        the grid is below this fraction of the strongest are dropped
        (EasySpin default 0.005).
    AccumMethod:
        garlic only.  Spectrum accumulation: ``'binning'`` (nearest-bin stick
        spectrum + convolution; default in the isotropic regime), ``'linear'``
        (line split between its two neighboring bins, EasySpin ``makespec``;
        differentiable in the line positions and selected automatically when the
        spin system carries grad tensors) or ``'explicit'`` (per-line
        Lorentzians; default in the fast-motion regime).  ``None`` = auto.
    Accuracy:
        garlic only.  Relative convergence criterion of the Breit–Rabi
        fixed-point / Newton iterations.  Default ``1e-12`` (EasySpin).
    MaxIterations:
        garlic only.  Iteration cap for the resonance-position solvers.
        Default ``15`` (EasySpin).
    GridSize:
        SOPHE grid size, either an integer ``N`` or a two-element list
        ``[N_coarse, N_interp]``.

        * Integer form: ``N`` — compute at N knots along the quarter
          meridian, no interpolation.
        * List form: ``[N_coarse, N_interp]`` — compute resonance fields
          at a coarse ``N_coarse``-knot grid, then spline-interpolate to a
          fine grid of ``nfKnots = (N_coarse - 1) * N_interp + 1`` knots
          before accumulating the spectrum.  Matches MATLAB EasySpin's
          default ``Opt.GridSize = [19 4]`` (→ nfKnots = 73).

        Default ``[19, 4]`` gives smooth powder averages without computing
        resonance fields at every fine-grid orientation.  For isotropic or
        very fast simulations, pass an integer (e.g. ``GridSize=31``) to
        skip interpolation.
    GridSymmetry:
        Symmetry used to select the powder grid fraction:

        - ``'auto'`` (default): detect the Hamiltonian symmetry via
          :func:`torchspin.hamsymm` and select the matching grid fraction
          automatically. Falls back to ``'Ci'`` only when detection fails
          (e.g. scalar Q tensors or other pathological inputs).
        - ``'Dinfh'``: axially symmetric — single meridian, phi=0.
        - ``'D2h'``: rhombic — one octant.
        - ``'Ci'``: triclinic — full hemisphere (phi 0..2π, theta 0..π/2).
        - ``'C1'``: no symmetry — full sphere.

    Threshold:
        Relative intensity cutoff for transitions.  Transitions with
        intensity below ``Threshold * max_intensity`` are discarded.
    Verbosity:
        Output verbosity: 0 = silent, 1 = progress, 2 = detailed.
    BatchSize:
        Number of orientations per batched resonance search. Default ``None``
        = automatic: all orientations at once, capped so that the batched
        Hamiltonians (orientations × field knots × dim²) stay below ~400 MB.
        Set to 1 for sequential processing (debugging/low-memory systems).
    Enhancement:
        :func:`~torchspin.endorfrq.endorfrq` and :func:`~torchspin.salt.salt`
        only (EasySpin ``Opt.Enhancement``).  ``False`` (default, as in
        EasySpin): the rf transition operator is the nuclear magnetic moment
        only.  ``True``: it is the total magnetic moment (electron + nuclear,
        the moment operators of :func:`~torchspin.ham.ham`), projected on the
        same rf directions, which includes hyperfine enhancement.  Transition
        selection (``Threshold``) still uses the nuclear-only rates, and
        electronic (EPR) level pairs are excluded, as in EasySpin.  ``salt``
        then always uses the matrix ENDOR path (its automatic first-order
        perturbation shortcut has no enhancement); ``endorfrq_perturb``
        raises.
    """

    Method: str = 'matrix'
    HybridCoreNuclei: Union[int, List[int], None] = None   # EasySpin Opt.HybridCoreNuclei (1-based)
    HybridIntThreshold: float = 0.005                      # EasySpin Opt.HybridIntThreshold
    AccumMethod: Optional[str] = None
    Accuracy: float = 1e-12
    MaxIterations: int = 15
    GridSize: Union[int, List[int]] = field(default_factory=lambda: [19, 4])
    GridSymmetry: str = 'auto'
    Threshold: float = 1e-4
    Verbosity: int = 0
    BatchSize: Optional[int] = None
    Sites: Optional[list] = None      # crystal sites to include (1-based), EasySpin Opt.Sites
    separate: str = ''                # '', 'orientations', 'sites' (crystals), 'components' (list input)
    IsoCutoff: float = 1e-4           # relative abundance cutoff for isotopologues (EasySpin Opt.IsoCutoff)
    Stretch: float = 0.25             # automatic sweep range padding factor (EasySpin garlic Opt.Stretch)
    ModellingAccuracy: float = 2e-6   # resonance search: segment-model accuracy relative to mwFreq (EasySpin Opt.ModellingAccuracy)
    device: str = 'cpu'
    Enhancement: bool = False         # ENDOR hyperfine enhancement (EasySpin Opt.Enhancement), endorfrq/salt matrix paths

    @property
    def grid_size_coarse(self) -> int:
        """Coarse grid knot count (N_coarse from GridSize)."""
        if isinstance(self.GridSize, (list, tuple)):
            return int(self.GridSize[0])
        return int(self.GridSize)

    @property
    def grid_size_interp(self) -> int:
        """Interpolation factor (N_interp from GridSize, 1 = no interpolation)."""
        if isinstance(self.GridSize, (list, tuple)):
            return int(self.GridSize[1]) if len(self.GridSize) > 1 else 1
        return 1

    def __post_init__(self) -> None:
        if self.Method not in VALID_METHODS:
            raise ValueError(
                f"Unknown Options.Method '{self.Method}'. "
                f"Valid: {', '.join(sorted(VALID_METHODS))}."
            )
        if self.AccumMethod not in (None, 'binning', 'linear', 'explicit'):
            raise ValueError(
                f"Options.AccumMethod must be None, 'binning', 'linear' or 'explicit'; got '{self.AccumMethod}'."
            )
        # Validate GridSize — accept int or [N_coarse, N_interp]
        if isinstance(self.GridSize, (list, tuple)):
            if len(self.GridSize) < 1 or len(self.GridSize) > 2:
                raise ValueError(
                    "GridSize must be an int N or a two-element list [N_coarse, N_interp]."
                )
            if self.GridSize[0] < 2:
                raise ValueError("GridSize[0] (coarse knots) must be at least 2.")
            if len(self.GridSize) > 1 and self.GridSize[1] < 1:
                raise ValueError("GridSize[1] (interpolation factor) must be at least 1.")
        else:
            if self.GridSize < 2:
                raise ValueError("GridSize must be at least 2.")
        valid_sym = {'Dinfh', 'D2h', 'C2h', 'Ci', 'C1', 'O3', 'auto'}
        if self.GridSymmetry not in valid_sym:
            raise ValueError(
                f"GridSymmetry must be one of {valid_sym}; got '{self.GridSymmetry}'."
            )
        # Graceful CUDA fallback: warn and revert to CPU if CUDA is requested
        # but not available (e.g. CPU-only PyTorch install).
        if self.device and "cuda" in str(self.device):
            import torch as _torch
            import warnings as _warnings
            if not _torch.cuda.is_available():
                _warnings.warn(
                    f"CUDA device '{self.device}' requested but CUDA is not available; "
                    "falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.device = 'cpu'


def auto_batch_size(batch_size, dim: int, n_orient: int) -> int:
    """Orientations per batched resonance-search call (``Options.BatchSize``).

    ``None`` = automatic: all orientations at once, capped so that the batched
    Hamiltonians (orientations × ~10 field knots × dim², complex128) stay
    below ~400 MB (PERF_PLAN §2.1 item 2).
    """
    if batch_size:
        return int(batch_size)
    return max(10, min(int(n_orient), int(2.5e7 // (10 * dim * dim))))
