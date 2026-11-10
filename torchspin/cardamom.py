"""cardamom — Trajectory-based simulation of CW-EPR spectra.

Port of MATLAB EasySpin's ``cardamom.m``.  Simulates CW-EPR spectra of
S=1/2 spin labels using stochastic or molecular dynamics trajectories.

Workflow:
1. Generate/receive orientation trajectories (diffusion, jump, or MD)
2. Propagate density matrices along trajectories
3. Compute FID from expectation value of S+
4. FFT → frequency-domain → field-domain spectrum
5. Interpolate onto user sweep range

References
----------
[1] Sezer et al., J. Chem. Phys. 128, 165106 (2008)
[2] Oganesyan, Phys. Chem. Chem. Phys. 13, 4724 (2011)

Typical usage::

    from torchspin import SpinSystem, Experiment, cardamom, CardamomPar

    sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                     Nucs=['14N'], A=[[10.0, 10.0, 95.0]], tcorr=5e-9)
    exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
    par = CardamomPar(Model='diffusion', dtSpin=1e-10, nSteps=2000,
                      dtSpatial=1e-11, nTraj=50, nOrients=20)
    B, spc, td, t = cardamom(sys, exp, par)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

from torchspin._cardamom_propagatedm import propagate_fast, propagate_istos
from torchspin._cardamom_utils import spiral_grid
from torchspin._rng import child_seeds as _child_seeds
from torchspin.constants import BMAGN, GFREE, PLANCK
from torchspin.stochtraj_diffusion import (
    stochtraj_diffusion, DiffusionPar,
    _euler2quat_active, _quat_to_rotmat_batch,
)
from torchspin.stochtraj_jump import stochtraj_jump, JumpPar
from torchspin.utils import unitconvert


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CardamomPar:
    """Simulation parameters for ``cardamom()``."""
    Model: str = 'diffusion'     # 'diffusion', 'jump', 'MD-direct'
    dtSpin: float = 1e-10        # spin propagation time step (s)
    nSteps: int = 2000           # number of spin propagation steps
    dtSpatial: float | None = None  # spatial dynamics time step (s)
    nTraj: int = 100             # number of trajectories
    nOrients: int = 100          # number of powder-average orientations
    Orients: np.ndarray | None = None  # (nOrients, 2) custom (phi, theta) grid
    OriStart: np.ndarray | None = None  # starting orientations for trajectories
    BlockLength: int = 1         # block averaging length for MD tensors
    seed: int | None = None      # RNG seed for the stochastic trajectories (None = fresh ensemble each call)


@dataclass
class CardamomOptions:
    """Options for ``cardamom()``."""
    Method: str = 'fast'          # 'fast' or 'ISTOs'
    FFTWindow: bool = True        # apply Hann window before FFT
    Verbosity: int = 0            # 0: silent, 1: progress, 2: debug
    chkCon: bool = False          # convergence check (Gelman-Rubin)
    convTolerance: float = 2e-2   # convergence threshold


@dataclass
class MDInput:
    """Molecular dynamics trajectory input."""
    FrameTraj: np.ndarray = None         # (3, 3, nSteps, nTraj) rotation matrices
    FrameTrajwrtProt: np.ndarray = None  # protein-frame rotations
    dt: float = None                     # MD time step (s)
    removeGlobal: bool = True
    DiffGlobal: float | None = None
    tauR: float | None = None            # rotational correlation time (s)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def cardamom(
    sys,
    exp,
    par: CardamomPar,
    opt: CardamomOptions | None = None,
    md: MDInput | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trajectory-based simulation of CW-EPR spectra.

    Parameters
    ----------
    sys:
        SpinSystem with S=1/2, g-tensor, and dynamics parameters
        (tcorr, Diff, TransRates, etc.).
    exp:
        Experiment parameters (mwFreq in GHz, Range in mT, nPoints,
        Harmonic).
    par:
        Simulation parameters (Model, time steps, trajectories).
    opt:
        Optional simulation options (Method, FFTWindow, etc.).
    md:
        Optional MD trajectory input (for 'MD-direct' model).

    Returns
    -------
    B:
        Magnetic field axis in mT, shape ``(nPoints,)``.
    spc:
        EPR spectrum, shape ``(nPoints,)``.
    TDSignal:
        Time-domain FID signal, shape ``(nSteps,)``.
    t:
        Time axis for FID, shape ``(nSteps,)``.
    """
    from torchspin.spinsystem import _reject_h0extra
    _reject_h0extra(sys, 'cardamom')
    if opt is None:
        opt = CardamomOptions()

    # --- Validate spin system ---
    if opt.Method == 'fast':
        if len(sys.S) != 1 or sys.S[0] != 0.5:
            raise ValueError(
                "cardamom Method='fast' requires S = 1/2. "
                "Use Method='ISTOs' for other spin systems."
            )

    # --- Extract g-tensor ---
    g = np.asarray(sys.g.detach().cpu() if hasattr(sys.g, 'detach') else sys.g,
                   dtype=np.float64).ravel()
    if g.size == 1:
        g = np.full(3, g[0])
    elif g.size != 3:
        raise ValueError("g must have 1 or 3 elements for cardamom.")

    # --- Extract A-tensor (optional) ---
    A = None
    if sys.Nucs:
        A_raw = sys.A
        if A_raw is not None:
            A = np.asarray(
                A_raw.detach().cpu() if hasattr(A_raw, 'detach') else A_raw,
                dtype=np.float64
            ).ravel()[:3]

    # --- Extract dynamics ---
    # jump and MD models carry their own dynamics (Sys.TransRates / MD input)
    Diff = _get_diffusion(sys) if par.Model == 'diffusion' else None

    # --- Experimental parameters ---
    mwFreq_Hz = exp.mwFreq * 1e9  # GHz → Hz
    omega = 2.0 * np.pi * mwFreq_Hz
    nPoints = getattr(exp, 'nPoints', 1024)
    Range = np.asarray(exp.Range, dtype=np.float64)
    CenterField = np.mean(Range)
    Harmonic = getattr(exp, 'Harmonic', 1)

    # --- Spatial dynamics parameters ---
    nStepsSpin = par.nSteps
    dtSpin = par.dtSpin
    dtSpatial = par.dtSpatial if par.dtSpatial is not None else dtSpin

    # Number of spatial steps to cover spin propagation time
    nStepsSpatial = int(np.ceil(nStepsSpin * dtSpin / dtSpatial))

    # --- Linewidth broadening ---
    lw = _get_linewidth(sys)

    # --- Orientation grid ---
    nOrients = par.nOrients
    if par.Orients is not None:
        gridPhi = par.Orients[:, 0]
        gridTheta = par.Orients[:, 1]
        nOrients = len(gridPhi)
    else:
        gridPhi, gridTheta = spiral_grid(nOrients)

    # Uniform weights for spiral grid
    weights = np.ones(nOrients) / nOrients

    # --- Time axis ---
    t = np.linspace(0, nStepsSpin * dtSpin, nStepsSpin)

    # --- Global dynamics ---
    DiffGlobal = getattr(sys, 'DiffGlobal', None)
    if DiffGlobal is not None and not isinstance(DiffGlobal, (int, float)):
        DiffGlobal = np.atleast_1d(np.asarray(DiffGlobal, dtype=np.float64))
    includeGlobalDynamics = DiffGlobal is not None

    # --- Random streams ---
    # One stream per orientation (local dynamics) and one per orientation (global
    # dynamics), derived from par.seed: an orientation draws the same trajectories
    # whether it is simulated alone or in a group, so seeded results do not depend
    # on the group size, the thread count, or how orientations are distributed.
    loc_seeds = _child_seeds(par.seed, nOrients, stream=0)
    glob_seeds = _child_seeds(par.seed, nOrients, stream=1)

    # --- Main orientation loop ---
    TDSignals = []

    if opt.Method != 'ISTOs' and par.Model in ('diffusion', 'jump'):
        # Fast method: the orientations are independent, so their trajectories
        # are generated and propagated in groups (one block of nTraj
        # trajectories per orientation) — the per-step Python loops then run
        # once per group instead of once per orientation (PERF_PLAN §2.4).
        # group size: amortises the per-step Python overhead, but large arrays
        # fall out of cache — scale the budget with the available threads
        import torch as _torch
        # the group size may depend on the thread count: each orientation draws from
        # its own seed stream (loc_seeds / glob_seeds), so seeded results do not
        budget = min(2e6, 2e5 * _torch.get_num_threads())
        G = max(1, min(nOrients, int(budget // max(1, par.nTraj * nStepsSpin))))
        for g0 in range(0, nOrients, G):
            ids = list(range(g0, min(g0 + G, nOrients)))
            Gk = len(ids)
            if opt.Verbosity >= 1:
                print(f"  Orientations {ids[0] + 1}-{ids[-1] + 1}/{nOrients}")
            RTrajLocal, _ = _generate_local_trajectory(par, sys, Diff, dtSpatial, nStepsSpatial, md, repeats=Gk,
                                                       seeds=None if loc_seeds is None else [loc_seeds[i] for i in ids])
            qLab = np.stack([_euler2quat_active(0.0, gridTheta[i], gridPhi[i]) for i in ids], axis=1)   # (4, Gk)
            qLab_traj = np.repeat(np.repeat(qLab[:, np.newaxis, :], nStepsSpin, axis=1), par.nTraj, axis=2)  # (4, nSteps, Gk*nTraj)
            if includeGlobalDynamics:
                # start every orientation's block from the spiral grid over nTraj, as the
                # per-orientation path does (a spiral over Gk*nTraj would make the
                # result depend on the group size)
                diff_par = DiffusionPar(dt=dtSpin, nSteps=nStepsSpin, nTraj=Gk * par.nTraj,
                                        OriStart=np.tile(_spiral_start(par.nTraj), (1, Gk)),
                                        seed=None if glob_seeds is None else [glob_seeds[i] for i in ids])
                _, _, qGlobal = stochtraj_diffusion(DiffGlobal, diff_par)
                qLab_traj = _quatmult(qLab_traj, qGlobal)
            RLab = _quat_to_rotmat_batch(qLab_traj)
            sig = propagate_fast(g, RTrajLocal, omega, dtSpin, nStepsSpin, Gk * par.nTraj, A=A, RLab=RLab, groups=Gk)
            sig = np.atleast_2d(sig)
            for j, i in enumerate(ids):
                TDSignals.append(weights[i] * sig[j])
        nOrients_done = nOrients
    else:
        nOrients_done = 0

    for iOri in range(nOrients_done, nOrients):
        if opt.Verbosity >= 1 and iOri % max(1, nOrients // 10) == 0:
            print(f"  Orientation {iOri+1}/{nOrients}")

        # Generate local trajectory
        RTrajLocal, qTrajLocal = _generate_local_trajectory(
            par, sys, Diff, dtSpatial, nStepsSpatial, md,
            seeds=None if loc_seeds is None else [loc_seeds[iOri]],
        )

        # Lab-frame rotation for this orientation
        qLab = _euler2quat_active(0.0, gridTheta[iOri], gridPhi[iOri])
        qLab_traj = np.tile(qLab[:, np.newaxis, np.newaxis],
                            (1, nStepsSpin, par.nTraj))  # (4, nSteps, nTraj)

        # Global dynamics
        if includeGlobalDynamics:
            diff_par = DiffusionPar(
                dt=dtSpin, nSteps=nStepsSpin, nTraj=par.nTraj,
                seed=None if glob_seeds is None else [glob_seeds[iOri]],
            )
            _, _, qGlobal = stochtraj_diffusion(DiffGlobal, diff_par)
            # Combine: qLab = qLab * qGlobal (Hamilton product, vectorised)
            qLab_traj = _quatmult(qLab_traj, qGlobal)

        # Propagate density matrix
        if opt.Method == 'ISTOs':
            # Extract nuclear spin info for magint
            nuc_spins_list = None
            gn_list = None
            D_arr = None
            if sys.Nucs:
                from torchspin.nucdata import nucspin, nucgval
                nuc_spins_list = [float(nucspin(n)) for n in sys.Nucs]
                gn_list = [float(nucgval(n)) for n in sys.Nucs]
            if hasattr(sys, 'D') and sys.D is not None:
                D_arr = np.asarray(
                    sys.D.detach().cpu() if hasattr(sys.D, 'detach') else sys.D,
                    dtype=float
                )
                if D_arr.ndim == 1:
                    D_arr = D_arr.reshape(1, -1)

            signal = propagate_istos(
                list(sys.Spins), g, A,
                qTrajLocal, omega, dtSpin, nStepsSpin, par.nTraj,
                CenterField,
                D=D_arr, gn=gn_list, nuc_spins=nuc_spins_list,
                qLab=qLab_traj,
                block_length=par.BlockLength,
            )
        else:
            # Fast method
            RLab = _quat_to_rotmat_batch(qLab_traj)  # (3,3,nSteps,nTraj)
            signal = propagate_fast(
                g, RTrajLocal, omega, dtSpin, nStepsSpin, par.nTraj,
                A=A, RLab=RLab,
            )

        TDSignals.append(weights[iOri] * signal)

    # --- Average over orientations ---
    # EasySpin cardamom.m: each orientation's FID carries weight(iOri) = 1/nOrients
    # and the columns are then averaged again (mean(TDSignal,2), mean(spcArray,2)),
    # so the output scale is 1/nOrients of the weighted sum — reproduced here.
    TDSignal = np.sum(TDSignals, axis=0) / len(TDSignals)  # (nSteps,)

    # --- Update time axis if block averaging changed signal length ---
    nActual = TDSignal.shape[0]
    if nActual != len(t):
        dtEffective = dtSpin * par.BlockLength
        t = np.linspace(0, nActual * dtEffective, nActual)

    # --- FFT pipeline ---
    spc, B_axis = _fft_pipeline(
        TDSignal, t, dtSpin, omega, g, lw, exp, nPoints, Range,
        Harmonic, opt.FFTWindow,
    )

    return B_axis, spc, TDSignal, t


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_diffusion(sys) -> np.ndarray:
    """Extract diffusion tensor from SpinSystem."""
    if hasattr(sys, 'logtcorr') and sys.logtcorr is not None:
        tcorr = 10.0 ** sys.logtcorr
        return np.atleast_1d(1.0 / (6.0 * tcorr))
    elif hasattr(sys, 'tcorr') and sys.tcorr is not None:
        return np.atleast_1d(1.0 / (6.0 * sys.tcorr))
    elif hasattr(sys, 'logDiff') and sys.logDiff is not None:
        return np.atleast_1d(10.0 ** sys.logDiff)
    elif hasattr(sys, 'Diff') and sys.Diff is not None:
        return np.atleast_1d(np.asarray(sys.Diff, dtype=np.float64))
    else:
        raise ValueError(
            "SpinSystem must have tcorr, logtcorr, Diff, or logDiff "
            "for cardamom simulation."
        )


def _get_linewidth(sys) -> np.ndarray:
    """Extract linewidth [Gaussian_FWHM, Lorentzian_FWHM] in mT."""
    lw = getattr(sys, 'lw', None)
    if lw is None:
        return np.array([0.0, 0.0])
    lw = np.atleast_1d(np.asarray(lw, dtype=np.float64)).ravel()
    if lw.size == 1:
        return np.array([lw[0], 0.0])
    return lw[:2]


def _spiral_start(n_traj: int) -> np.ndarray:
    """(3, n_traj) Euler angles of the spiral starting grid stochtraj_diffusion
    builds for ``n_traj`` trajectories when no OriStart or potential is given."""
    pts = np.linspace(-1, 1, n_traj)
    return np.stack([np.sqrt(np.pi * n_traj) * np.arcsin(pts), np.arccos(pts),
                     np.sqrt(np.pi * n_traj) * np.arcsin(pts)], axis=0)


def _generate_local_trajectory(par, sys, Diff, dtSpatial, nStepsSpatial, md, repeats: int = 1,
                               seeds=None):
    """Local trajectories for one orientation, or for ``repeats`` orientations at
    once (``repeats`` consecutive blocks of ``par.nTraj`` trajectories with the
    same starting orientations).

    ``seeds`` holds one seed per block (``repeats`` entries), or is ``None`` for an
    unseeded run.
    """
    if par.Model == 'diffusion':
        Potential = getattr(sys, 'Potential', None)
        if Potential is not None:
            Potential = np.asarray(Potential)
        OriStart = par.OriStart
        if repeats > 1:
            if OriStart is not None:
                OriStart = np.asarray(OriStart, dtype=np.float64)
                if OriStart.ndim == 1:
                    OriStart = OriStart[:, np.newaxis]
                if OriStart.shape[1] == 1:
                    OriStart = np.tile(OriStart, (1, par.nTraj))
            elif Potential is None:
                # the spiral starting grid stochtraj_diffusion would build for nTraj
                OriStart = _spiral_start(par.nTraj)
            if OriStart is not None:
                OriStart = np.tile(OriStart, (1, repeats))
        diff_par = DiffusionPar(
            dt=dtSpatial, nSteps=nStepsSpatial, nTraj=par.nTraj * repeats,
            OriStart=OriStart, seed=seeds,
        )
        _, RTraj, qTraj = stochtraj_diffusion(Diff, diff_par, Potential=Potential)
        return RTraj, qTraj

    elif par.Model == 'jump':
        TransRates = getattr(sys, 'TransRates', None)
        TransProb = getattr(sys, 'TransProb', None)
        Orientations = getattr(sys, 'Orientations', None)

        if TransRates is not None:
            TransRates = np.asarray(TransRates, dtype=np.float64)
        if TransProb is not None:
            TransProb = np.asarray(TransProb, dtype=np.float64)
        if Orientations is not None:
            Orientations = np.asarray(Orientations, dtype=np.float64)

        jump_par = JumpPar(dt=dtSpatial, nSteps=nStepsSpatial, nTraj=par.nTraj * repeats, seed=seeds)
        _, RTraj, qTraj, _ = stochtraj_jump(
            TransRates=TransRates, TransProb=TransProb,
            Orientations=Orientations, par=jump_par,
        )
        return RTraj, qTraj

    elif par.Model == 'MD-direct':
        if md is None or md.FrameTraj is None:
            raise ValueError("MD trajectory (md.FrameTraj) required for MD-direct model.")
        RTraj = md.FrameTraj
        # Build quaternion trajectory from rotation matrices
        nS = RTraj.shape[2]
        nT = RTraj.shape[3]
        qTraj = np.zeros((4, nS, nT), dtype=np.float64)
        # Just use identity quaternion — will use RTraj directly
        qTraj[0, :, :] = 1.0
        return RTraj, qTraj

    else:
        raise ValueError(
            f"Unknown model '{par.Model}'. "
            f"Choose from: 'diffusion', 'jump', 'MD-direct'."
        )


def _quatmult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternion arrays (leading axis of length 4)."""
    a1, b1, c1, d1 = q1[0], q1[1], q1[2], q1[3]
    a2, b2, c2, d2 = q2[0], q2[1], q2[2], q2[3]
    return np.stack([
        a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
        a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
        a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
        a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
    ], axis=0)


def _quatmult_single(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (Hamilton convention)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _fft_pipeline(
    TDSignal: np.ndarray,
    t: np.ndarray,
    dtSpin: float,
    omega: float,
    g: np.ndarray,
    lw: np.ndarray,
    exp,
    nPoints: int,
    Range: np.ndarray,
    Harmonic: int,
    FFTWindow: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """FFT pipeline: window → zero-pad → broaden → FFT → interpolate.

    Returns (spc, B_axis) in mT.
    """
    nSteps = len(TDSignal)
    tLong = t.copy()

    # --- Apply Hann window ---
    if FFTWindow:
        tMax = t[-1]
        window = 0.5 * (1.0 + np.cos(np.pi * t / tMax))
        TDSignal = TDSignal * window

    # --- Zero-padding for field resolution ---
    Bres_mT = 0.01  # 0.1 G = 0.01 mT target resolution
    mwFreq_MHz = exp.mwFreq * 1e3  # GHz → MHz
    Bres_MHz = Bres_mT * abs(BMAGN / PLANCK) * np.mean(g) * 1e-9  # mT → MHz
    tReq = 1.0 / (Bres_MHz * 1e6)  # s

    tMaxActual = t[-1]
    if tMaxActual < tReq:
        M = int(np.ceil(tReq / dtSpin))
    else:
        M = int(np.ceil(tMaxActual / dtSpin))
    M = max(M, nSteps)

    # Zero-pad
    if M > nSteps:
        TDSignal_padded = np.zeros(M, dtype=np.complex128)
        TDSignal_padded[:nSteps] = TDSignal
    else:
        TDSignal_padded = TDSignal.copy()
        M = nSteps

    tLong = np.linspace(0, M * dtSpin, M)

    # --- Broadening in time domain ---
    if lw[0] > 0:
        # Gaussian: FWHM in mT → Hz
        fwhm_Hz = _mT_to_Hz(lw[0], np.mean(g))
        alpha = np.pi**2 * fwhm_Hz**2 / (4.0 * np.log(2.0))
        TDSignal_padded *= np.exp(-alpha * tLong**2)

    if lw[1] > 0:
        # Lorentzian: FWHM in mT → T2 in s
        fwhm_Hz = _mT_to_Hz(lw[1], np.mean(g))
        T2 = 1.0 / (np.pi * fwhm_Hz)
        TDSignal_padded *= np.exp(-tLong / T2)

    # --- Multiply by t for differentiation (first derivative = absorption) ---
    TDSignal_padded *= tLong

    # --- FFT ---
    spcFFT = np.imag(np.fft.fftshift(np.fft.fft(TDSignal_padded)))

    # --- Frequency axis ---
    freq = np.fft.fftshift(np.fft.fftfreq(M, d=dtSpin))  # Hz

    # --- Convert to field axis ---
    gMean = np.mean(g)
    fftAxis_mT = _freq_to_field(freq, mwFreq_MHz, gMean)

    # --- Interpolate onto sweep range ---
    B_sweep = np.linspace(Range[0], Range[1], nPoints)

    # Sort for interpolation (fftAxis may not be monotonic after conversion)
    sort_idx = np.argsort(fftAxis_mT)
    interp_fn = interp1d(
        fftAxis_mT[sort_idx], spcFFT[sort_idx],
        kind='linear', bounds_error=False, fill_value=0.0
    )
    spc = interp_fn(B_sweep)

    # EasySpin cardamom.m (field sweeps): the spectrum is imag(FFT(t·FID))
    # interpolated onto the field axis — Exp.Harmonic is not applied.  (Frequency
    # sweeps integrate to the absorption with cumtrapz, see cardamom.m.)
    return spc, B_sweep


def _mT_to_Hz(lw_mT: float, gMean: float) -> float:
    """Convert linewidth from mT to Hz."""
    # lw(MHz) = lw(mT) * |g * BMAGN / PLANCK| * 1e-9 (mT→T)
    return abs(lw_mT * gMean * BMAGN / PLANCK * 1e-9) * 1e6  # → Hz


def _freq_to_field(freq_Hz: np.ndarray, mwFreq_MHz: float, gMean: float) -> np.ndarray:
    """Convert frequency offset (Hz) to field (mT).

    field = (freq_offset + mwFreq) / (gMean * BMAGN/PLANCK * 1e-9)
    """
    # Total frequency in MHz
    total_MHz = freq_Hz / 1e6 + mwFreq_MHz
    # Convert MHz → mT using g-factor
    # B(mT) = freq(MHz) / (g * BMAGN/PLANCK * 1e-9) / 1e6 * 1e3
    # = freq(MHz) / (g * 13.9962 GHz/T) * 1e3
    gamma = gMean * abs(BMAGN / PLANCK) * 1e-9  # MHz/mT
    B_mT = total_MHz / gamma
    return B_mT
