"""saffron — Simulate pulse EPR signals and spectra.

Port of EasySpin's ``saffron.m``. Supports predefined experiments
(2pESEEM, 3pESEEM, 4pESEEM, HYSCORE, MimsENDOR) and custom pulse
sequences using the fast (ideal-pulse) algorithm.

All energies in **MHz**, magnetic fields in **mT**, times in **µs**.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import torch

from torchspin.constants import BMAGN, NMAGN, PLANCK, GFREE
from torchspin.ham import ham as _ham_full
from torchspin.hamsymm import hamsymm
from torchspin.lineshape import apowin
from torchspin.nucdata import nucgval
from torchspin.rotations import erot
from torchspin.saffron_peaks import sf_peaks, sf_evolve
from torchspin.sphgrid import sphgrid
from torchspin.spinops import sop
from torchspin.spinsystem import SpinSystem, nucspinrmv


# =====================================================================
# Public data structures
# =====================================================================

@dataclass
class PulseExperiment:
    """Pulse EPR experiment specification."""
    Field: float                         # mT — static B0
    Sequence: Union[str, list] = '2pESEEM'
    dt: Union[float, list, None] = None  # µs, time step per dimension
    nPoints: Union[int, list, None] = None
    tau: float = 0.0                     # µs — fixed tau
    T: float = 0.0                       # µs — mixing time
    t1: float = 0.0                      # µs — HYSCORE initial t1
    t2: float = 0.0                      # µs — HYSCORE initial t2
    mwFreq: Optional[float] = None       # GHz — orientation selection;
                                         # required for real pulses
    ExciteWidth: Optional[float] = None  # MHz — excitation bandwidth
    Range: Optional[list] = None         # [lo, hi] MHz — ENDOR range
    tprf: float = 20.0                   # µs — ENDOR RF pulse length
    Temperature: Optional[float] = None  # K
    # Crystal fields (any set → single-crystal simulation)
    SampleFrame: Optional[list] = None       # (N,3) Euler angles, lab→sample (rad)
    CrystalSymmetry: Optional[object] = None # space group number/symbol
    MolFrame: Optional[list] = None          # 3 Euler angles, sample→mol (rad)
    # Custom sequence fields
    Flip: Optional[list] = None          # flip angles (units of π/2)
    Phase: Optional[list] = None
    Inc: Optional[list] = None           # incrementation scheme
    t: Optional[list] = None             # initial delays (µs)
    tp: Optional[list] = None            # pulse durations (µs, 0=ideal)
    Filter: Optional[str] = None
    # Real-pulse ("thyme") detection fields — EasySpin s_sequencer.m 255-325.
    # Ignored by the fast (ideal-pulse) algorithm.
    DetWindow: Union[float, list, None] = None   # µs, scalar or [t1, t2]
    DetSequence: Union[int, list, str, None] = None  # per-event detection flags
    DetPhase: Optional[float] = None     # rad, detection phase (default 0)
    DetFrequency: Optional[float] = None # GHz, down-conversion (default mwFreq)
    DetIntegrate: bool = False           # integrate the echo transient
    DetOperator: Optional[object] = None # spidyan-only; rejected by saffron
    # Real-pulse ("thyme") sequence fields. Exp.Sequence may be a list of
    # pulse dicts and delays, as in EasySpin's general method.
    Dim1: Optional[list] = None          # e.g. ['d1,d2', 0.5]
    Dim2: Optional[list] = None
    PhaseCycle: Optional[object] = None  # dict {pulse_index: [[phase, w], ...]}


@dataclass
class SaffronOptions:
    """Options for saffron simulation."""
    GridSize: int = 31
    GridSymmetry: str = 'auto'
    ProductRule: bool = False
    TimeDomain: bool = False
    # EasySpin's default is conditional: 0 (adjacent-level population swap)
    # for a single SHF nucleus or under ProductRule, else 1. ``None`` selects
    # that rule; an explicit int overrides it.
    EndorMethod: int | None = None       # 0=pop swap, 1=sum transitions, 2=sweep
    # EasySpin default is dimension-dependent: 4 for 1D, 2 for 2D sequences.
    # ``None`` selects that rule; an explicit int overrides it.
    Expand: int | None = None            # frequency-domain zero-fill 2^Expand
    Window: str = 'ham+'                 # apodization window
    ZeroFillFactor: int = 2
    OriThreshold: float = 0.005
    # Engine selection, as in EasySpin's Opt.SimulationMode: 'fast' is the
    # ideal-pulse algorithm, 'thyme' the real-pulse density-matrix engine.
    # ``None`` keeps torchspin's historical behavior of inferring the engine
    # from nonzero Exp.tp / a list-valued Exp.Sequence / Exp.DetWindow.
    SimulationMode: Optional[str] = None  # 'fast' | 'thyme'
    # Real-pulse engine only (EasySpin Opt.IntTimeStep / Opt.SimFreq).
    IntTimeStep: Optional[float] = None  # µs, propagation time step
    SimFreq: Optional[float] = None      # GHz, simulation-frame frequency
    Nuclei: Optional[list] = None        # subset of nuclei (1-based)
    Transitions: Optional[list] = None
    separate: str = ''                   # '' or 'components'
    Verbosity: int = 0
    device: str = 'cpu'


# =====================================================================
# Internal data structures
# =====================================================================

@dataclass
class _NuclearSubspace:
    """Nuclear spin Hamiltonian components in nuclear-only Hilbert space."""
    Hnzx: torch.Tensor
    Hnzy: torch.Tensor
    Hnzz: torch.Tensor
    Hhfx: torch.Tensor
    Hhfy: torch.Tensor
    Hhfz: torch.Tensor
    Hnq: torch.Tensor
    # ENDOR operators (optional)
    Ix: Optional[torch.Tensor] = None
    Iy: Optional[torch.Tensor] = None
    Iz: Optional[torch.Tensor] = None


# =====================================================================
# Experiment name → ID mapping
# =====================================================================

_EXP_NAMES = {
    '2pESEEM': 1,
    '3pESEEM': 2,
    '4pESEEM': 3,
    'HYSCORE': 4,
    'MimsENDOR': 5,
}

_EXP_PARAMS = {
    # id: (nIntervals, nDimensions, IncSchemeID, nPathways, pathwayprefactors)
    1: (2, 1, 2, 1, [+1/2]),
    2: (3, 1, 1, 2, [+1/8, +1/8]),
    3: (4, 1, 2, 2, [-1/8, -1/8]),
    4: (4, 2, 11, 2, [-1/8, -1/8]),
    5: (3, 1, 0, 2, [+1/8, +1/8]),
}

# Incrementation scheme mapping (MATLAB saffron.m lines 420-432)
_INC_SCHEME_MAP = {
    (1,):          1,
    (1, 1):        2,
    (1, -1):       3,
    (1, 2):        11,
    (1, 2, 1):     12,
    (1, 2, 2):     13,
    (1, 1, 2):     14,
    (1, 2, 2, 1):  15,
    (1, 2, -2, 1): 16,
    (1, 1, 2, 2):  17,
}


# =====================================================================
# Pathway parser (port of MATLAB pathwayparser)
# =====================================================================

def _pathwayparser(ctp: np.ndarray):
    """Parse coherence transfer pathways into propagator indices.

    Port of MATLAB ``pathwayparser()`` (saffron.m lines 2157-2217).

    Parameters
    ----------
    ctp : ndarray of int, shape (nPathways, nIntervals)
        Pathway codes: 1=alpha, 2=beta, 3=+, 4=-

    Returns
    -------
    free_l, free_r : ndarray, shape (nPathways, nIntervals)
        Manifold indices (1=alpha, 2=beta) for left/right free evolution.
    pulse_l, pulse_r : ndarray, shape (nPathways, nIntervals)
        Propagator indices for left/right pulse action.
        1=alpha-block, 2=beta-block, 3=cross(a→b), 4=cross(b→a)
    """
    # Superpropagator table (MATLAB lines 2179-2180)
    # Row = "after" state (1=a,2=b,3=+,4=-), Col = "before" state
    PropL = np.array([
        [1, 3, 1, 3],
        [4, 2, 4, 2],
        [1, 3, 1, 3],
        [4, 2, 4, 2],
    ], dtype=np.int64)
    PropR = np.array([
        [1, 3, 3, 1],
        [4, 2, 2, 4],
        [4, 2, 2, 4],
        [1, 3, 3, 1],
    ], dtype=np.int64)

    n_pathways, n_intervals = ctp.shape

    # Free evolution indices: diagonal entries of superpropagator
    # idx = 5*ctp - 4 (MATLAB line 2199)
    # CRITICAL: MATLAB uses column-major linear indexing. In Python/numpy,
    # .flat uses row-major. Use .T.flat to match MATLAB column-major order.
    idx_free = 5 * ctp - 4  # maps 1→1, 2→6, 3→11, 4→16
    free_l = PropL.T.flat[idx_free - 1]
    free_r = PropR.T.flat[idx_free - 1]

    # Pulse propagator indices
    # "before" state: 2 (beta) before first pulse, then ctp of previous interval
    before = np.column_stack([
        np.full(n_pathways, 2, dtype=np.int64),
        ctp[:, :-1]
    ])

    # idx = ctp + (before-1)*4  (MATLAB line 2193, column-major indexing)
    idx_pulse = ctp + (before - 1) * 4
    pulse_l = PropL.T.flat[idx_pulse - 1]
    pulse_r = PropR.T.flat[idx_pulse - 1]

    return free_l, free_r, pulse_l, pulse_r


def _compute_pathway_prefactors(
    pathways: np.ndarray,
    flip: list[float],
    phase: list[float],
    pulse_l: np.ndarray,
    pulse_r: np.ndarray,
    ideal_pulse: list[bool],
) -> np.ndarray:
    """Compute amplitude prefactors for each pathway from ideal pulse theory.

    Port of MATLAB saffron.m lines 573-596.

    Parameters
    ----------
    pathways : ndarray (nPathways, nIntervals)
    flip : list of float
        Flip angles in units of π/2.
    phase : list of float
        Pulse phases in units of π/2.
    pulse_l, pulse_r : ndarray (nPathways, nIntervals)
    ideal_pulse : list of bool

    Returns
    -------
    ndarray of complex128, shape (nPathways,)
    """
    n_pathways, n_intervals = pathways.shape
    prefactors = np.ones(n_pathways, dtype=np.complex128)

    for i_pulse in range(n_intervals):
        if not ideal_pulse[i_pulse]:
            continue
        theta = flip[i_pulse] * math.pi / 2
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        ph = phase[i_pulse]

        for ip in range(n_pathways):
            # Left factor
            pl_code = pulse_l[ip, i_pulse]
            if pl_code in (1, 2):
                pL = c
            elif pl_code == 3:
                pL = -1j * s * (-1j) ** ph
            else:  # 4
                pL = -1j * s * (1j) ** ph
            # Right factor
            pr_code = pulse_r[ip, i_pulse]
            if pr_code in (1, 2):
                pR = c
            elif pr_code == 3:
                pR = 1j * s * (1j) ** ph
            else:  # 4
                pR = 1j * s * (-1j) ** ph
            prefactors[ip] *= pL * pR

    return prefactors


def _apply_coherence_filter(pathways: np.ndarray, filter_str: str) -> np.ndarray:
    """Apply coherence filter to pathway list.

    Port of MATLAB saffron.m lines 543-558.

    Parameters
    ----------
    pathways : ndarray (nPathways, nIntervals)
    filter_str : str
        Filter characters: '0'=zero-order, '1'=single-quantum,
        'a'=alpha, 'b'=beta, '+'=plus, '-'=minus, '.'/'*'=any

    Returns
    -------
    ndarray — filtered pathways
    """
    keep = np.ones(len(pathways), dtype=bool)
    for i_int, ch in enumerate(filter_str):
        col = pathways[:, i_int]
        if ch == '0':
            keep &= (col == 1) | (col == 2)
        elif ch == '1':
            keep &= (col == 3) | (col == 4)
        elif ch == 'a':
            keep &= (col == 1)
        elif ch == 'b':
            keep &= (col == 2)
        elif ch == '+':
            keep &= (col == 3)
        elif ch == '-':
            keep &= (col == 4)
        elif ch in ('.', '*'):
            pass  # keep all
        else:
            raise ValueError(
                f"Invalid filter character '{ch}' at position {i_int}. "
                "Use '0','1','a','b','+','-','.' or '*'."
            )
    return pathways[keep]


# =====================================================================
# Nuclear subspace builder
# =====================================================================

def _build_nuclear_subspaces(
    sys: SpinSystem,
    shf_nuclei: list[int],
    product_rule: bool,
    is_endor: bool,
    device: str,
) -> list[_NuclearSubspace]:
    """Build nuclear-only Hamiltonian operators for saffron fast algorithm.

    Parameters
    ----------
    sys : SpinSystem
    shf_nuclei : list of int
        0-based indices of nuclei participating in ESEEM/ENDOR.
    product_rule : bool
        If True, each nucleus gets its own subspace.
    is_endor : bool
        If True, also build Ix/Iy/Iz operators.
    device : str
        Torch device.

    Returns
    -------
    list of _NuclearSubspace
    """
    dtype = torch.complex128
    I_all = sys.I  # list of nuclear spins for shf nuclei (0-based indexing)
    I_shf = [I_all[i] for i in shf_nuclei]

    if product_rule:
        # Each nucleus in its own subspace
        subspaces = []
        for ii, inuc in enumerate(shf_nuclei):
            I_nuc = I_all[inuc]
            spins = [I_nuc]
            Ix = sop(spins, [[1, 1]], dtype=dtype, device=device)
            Iy = sop(spins, [[1, 2]], dtype=dtype, device=device)
            Iz = sop(spins, [[1, 3]], dtype=dtype, device=device)
            sub = _build_single_subspace(sys, inuc, spins, Ix, Iy, Iz,
                                          is_endor, dtype, device)
            subspaces.append(sub)
        return subspaces
    else:
        # All nuclei in one subspace
        if not I_shf:
            return []
        spins = I_shf
        n_nuc = len(spins)
        sub = _NuclearSubspace(
            Hnzx=torch.zeros(1, dtype=dtype, device=device),
            Hnzy=torch.zeros(1, dtype=dtype, device=device),
            Hnzz=torch.zeros(1, dtype=dtype, device=device),
            Hhfx=torch.zeros(1, dtype=dtype, device=device),
            Hhfy=torch.zeros(1, dtype=dtype, device=device),
            Hhfz=torch.zeros(1, dtype=dtype, device=device),
            Hnq=torch.zeros(1, dtype=dtype, device=device),
        )
        # Initialize with correct size
        dim = int(np.prod([int(2*I + 1) for I in spins]))
        zero = torch.zeros(dim, dim, dtype=dtype, device=device)
        sub.Hnzx = zero.clone()
        sub.Hnzy = zero.clone()
        sub.Hnzz = zero.clone()
        sub.Hhfx = zero.clone()
        sub.Hhfy = zero.clone()
        sub.Hhfz = zero.clone()
        sub.Hnq = zero.clone()
        if is_endor:
            sub.Ix = zero.clone()
            sub.Iy = zero.clone()
            sub.Iz = zero.clone()

        for ii, inuc in enumerate(shf_nuclei):
            spin_idx = ii + 1  # 1-based index within the subspace
            Ix = sop(spins, [[spin_idx, 1]], dtype=dtype, device=device)
            Iy = sop(spins, [[spin_idx, 2]], dtype=dtype, device=device)
            Iz = sop(spins, [[spin_idx, 3]], dtype=dtype, device=device)

            # Nuclear Zeeman
            gn = nucgval(sys.Nucs[inuc])
            pre = -gn * NMAGN / (1e3 * PLANCK * 1e6)  # MHz/mT
            sub.Hnzx = sub.Hnzx + pre * Ix
            sub.Hnzy = sub.Hnzy + pre * Iy
            sub.Hnzz = sub.Hnzz + pre * Iz

            # Hyperfine
            A_mol = _get_A_matrix_mol(sys, inuc, dtype, device)
            sub.Hhfx = sub.Hhfx + A_mol[0, 0]*Ix + A_mol[0, 1]*Iy + A_mol[0, 2]*Iz
            sub.Hhfy = sub.Hhfy + A_mol[1, 0]*Ix + A_mol[1, 1]*Iy + A_mol[1, 2]*Iz
            sub.Hhfz = sub.Hhfz + A_mol[2, 0]*Ix + A_mol[2, 1]*Iy + A_mol[2, 2]*Iz

            # Quadrupole
            if sys.I[inuc] >= 1:
                Q_mol = _get_Q_matrix_mol(sys, inuc, dtype, device)
                Hnq_ = (Ix @ (Q_mol[0, 0]*Ix + Q_mol[0, 1]*Iy + Q_mol[0, 2]*Iz) +
                         Iy @ (Q_mol[1, 0]*Ix + Q_mol[1, 1]*Iy + Q_mol[1, 2]*Iz) +
                         Iz @ (Q_mol[2, 0]*Ix + Q_mol[2, 1]*Iy + Q_mol[2, 2]*Iz))
                sub.Hnq = sub.Hnq + Hnq_

            # ENDOR
            if is_endor:
                sub.Ix = sub.Ix + Ix
                sub.Iy = sub.Iy + Iy
                sub.Iz = sub.Iz + Iz

        return [sub]


def _build_single_subspace(sys, inuc, spins, Ix, Iy, Iz,
                            is_endor, dtype, device):
    """Build a single-nucleus _NuclearSubspace."""
    gn = nucgval(sys.Nucs[inuc])
    pre = -gn * NMAGN / (1e3 * PLANCK * 1e6)  # MHz/mT

    A_mol = _get_A_matrix_mol(sys, inuc, dtype, device)
    Hhfx = A_mol[0, 0]*Ix + A_mol[0, 1]*Iy + A_mol[0, 2]*Iz
    Hhfy = A_mol[1, 0]*Ix + A_mol[1, 1]*Iy + A_mol[1, 2]*Iz
    Hhfz = A_mol[2, 0]*Ix + A_mol[2, 1]*Iy + A_mol[2, 2]*Iz

    Hnq = torch.zeros_like(Ix)
    if sys.I[inuc] >= 1:
        Q_mol = _get_Q_matrix_mol(sys, inuc, dtype, device)
        Hnq = (Ix @ (Q_mol[0, 0]*Ix + Q_mol[0, 1]*Iy + Q_mol[0, 2]*Iz) +
               Iy @ (Q_mol[1, 0]*Ix + Q_mol[1, 1]*Iy + Q_mol[1, 2]*Iz) +
               Iz @ (Q_mol[2, 0]*Ix + Q_mol[2, 1]*Iy + Q_mol[2, 2]*Iz))

    return _NuclearSubspace(
        Hnzx=pre * Ix, Hnzy=pre * Iy, Hnzz=pre * Iz,
        Hhfx=Hhfx, Hhfy=Hhfy, Hhfz=Hhfz,
        Hnq=Hnq,
        Ix=Ix if is_endor else None,
        Iy=Iy if is_endor else None,
        Iz=Iz if is_endor else None,
    )


def _get_A_matrix_mol(sys, inuc, dtype, device):
    """Get 3x3 A-tensor in molecular frame for nucleus inuc (0-based)."""
    if sys.A is None:
        return torch.zeros(3, 3, dtype=torch.float64, device=device)

    A_tensor = sys.A
    if sys.fullA:
        # Full 3×3 blocks stacked: rows (inuc*3):(inuc*3+3)
        A_pv = A_tensor[inuc*3:(inuc*3+3), :].to(
            device=device, dtype=torch.float64
        )
        return A_pv
    else:
        # Principal values: A[inuc, :] = [Ax, Ay, Az]
        A_pv = A_tensor[inuc, :].to(device=device, dtype=torch.float64)
        if sys.AFrame is not None:
            af = sys.AFrame[inuc, :].to(device=device, dtype=torch.float64)
            R = erot(af, device=device).T  # AFrame → molecular frame
        else:
            R = torch.eye(3, dtype=torch.float64, device=device)
        A_mol = R @ torch.diag(A_pv) @ R.T
        A_mol = (A_mol + A_mol.T) / 2
        return A_mol


def _get_Q_matrix_mol(sys, inuc, dtype, device):
    """Get 3x3 Q-tensor in molecular frame for nucleus inuc (0-based)."""
    if sys.Q is None:
        return torch.zeros(3, 3, dtype=torch.float64, device=device)

    Q_tensor = sys.Q
    if sys.fullQ:
        Q_pv = Q_tensor[inuc*3:(inuc*3+3), :].to(
            device=device, dtype=torch.float64
        )
        return Q_pv
    else:
        Q_pv = Q_tensor[inuc, :].to(device=device, dtype=torch.float64)
        # Handle scalar Q (single value = e²qQ/[4I(2I-1)])
        # EasySpin expands to [-Q, -Q, 2Q] principal values
        if Q_pv.numel() == 1:
            q = Q_pv.item()
            Q_pv = torch.tensor([-q, -q, 2*q], dtype=torch.float64, device=device)
        elif Q_pv.numel() == 2:
            # EasySpin [e2qQ/h, eta] two-element format
            e2qQ_h = Q_pv[0].item()
            eta = Q_pv[1].item()
            I_nuc = float(sys.I[inuc])
            q = e2qQ_h / (4.0 * I_nuc * (2.0 * I_nuc - 1.0))
            Q_pv = torch.tensor(
                [q * (-1.0 + eta), q * (-1.0 - eta), 2.0 * q],
                dtype=torch.float64, device=device,
            )
        if sys.QFrame is not None:
            qf = sys.QFrame[inuc, :].to(device=device, dtype=torch.float64)
            R = erot(qf, device=device).T
        else:
            R = torch.eye(3, dtype=torch.float64, device=device)
        Q_mol = R @ torch.diag(Q_pv) @ R.T
        Q_mol = (Q_mol + Q_mol.T) / 2
        return Q_mol


# =====================================================================
# Main public function
# =====================================================================

def saffron(
    sys: SpinSystem,
    exp: PulseExperiment,
    opt: Optional[SaffronOptions] = None,
) -> tuple:
    """Simulate pulse EPR signals and spectra.

    Parameters
    ----------
    sys : SpinSystem
    exp : PulseExperiment
    opt : SaffronOptions, optional

    Returns
    -------
    tuple of (x, signal, info)
        x : ndarray or list of ndarray — time or frequency axes
        signal : ndarray — simulated signal (time domain; complex) or ENDOR
            spectrum.  When the spin system carries tensors that require grad,
            ``signal`` and ``info['td']``/``info['fd']`` are torch tensors on the
            autograd graph instead.
        info : dict — contains 'td' and 'fd' fields

    Notes
    -----
    ``Opt.SimulationMode`` selects the engine, as in EasySpin: ``'fast'`` is
    the ideal-pulse algorithm implemented here, ``'thyme'`` the real-pulse
    density-matrix engine in :func:`torchspin.saffron_thyme.saffron_thyme`.
    Left unset, the engine is inferred: real pulses (nonzero ``Exp.tp``), a
    list-valued ``Exp.Sequence`` or an ``Exp.DetWindow`` select ``'thyme'``.

    The real-pulse engine has a different output contract -- one detected
    point per increment for single-point detection, with the transient axis
    appended to the indirect axes otherwise -- and requires ``Exp.mwFreq``.
    See its docstring.
    """
    if opt is None:
        opt = SaffronOptions()

    # --- Multi-component input: saffron([Sys1, Sys2], exp, opt) ---
    # Each component is simulated separately and combined with Sys.weight.
    # With opt.separate == 'components', the per-component signals are
    # returned stacked along the first axis instead of summed.
    if isinstance(sys, (list, tuple)):
        x_out = None
        signals = []
        infos = []
        for s in sys:
            x_i, y_i, info_i = saffron(s, exp, opt)
            w = float(getattr(s, 'weight', 1.0) or 1.0)
            signals.append(w * torch.as_tensor(y_i))
            infos.append(info_i)
            x_out = x_i
        if getattr(opt, 'separate', '') == 'components':
            signal = torch.stack(signals, dim=0)
            info = {'components': infos}
        else:
            signal = signals[0]
            for y_i in signals[1:]:
                signal = signal + y_i
            info = {'components': infos}
            # Combine accumulable fields where shapes agree
            for key in ('td', 'fd'):
                parts = [i.get(key) for i in infos]
                if all(p is not None for p in parts):
                    try:
                        combined = sum(
                            float(getattr(s, 'weight', 1.0) or 1.0) * torch.as_tensor(p)
                            for s, p in zip(sys, parts))
                        info[key] = combined
                    except (ValueError, RuntimeError):
                        pass
        if not (torch.is_tensor(signal) and signal.requires_grad):
            signal = signal.detach().cpu().numpy()
            for key in ('td', 'fd'):
                if torch.is_tensor(info.get(key)):
                    info[key] = info[key].detach().cpu().numpy()
        return x_out, signal, info

    # Deep-copy exp so defaults/mutations in this function do not leak back to the
    # caller (critical for esfit loops and for reproducible benchmarks). Completes
    # audit item A10, which previously only addressed sys.HStrain mutation.
    import copy as _copy
    exp = _copy.deepcopy(exp)

    device = opt.device
    dtype = torch.complex128
    on_graph = any(t is not None and torch.is_tensor(t) and t.requires_grad
                   for t in (sys.g, sys.A, sys.Q, sys.D, getattr(sys, 'ee', None), getattr(sys, 'nn', None)))
    if on_graph and not opt.TimeDomain and not opt.ProductRule:
        # The default frequency-domain accumulation bins every peak into a
        # frequency bin before the inverse FFT (EasySpin sf_peaks), which is not
        # differentiable in the peak frequencies; the time-domain evolution
        # (EasySpin Opt.TimeDomain) is the exact form and is used for gradients.
        import dataclasses as _dc
        opt = _dc.replace(opt, TimeDomain=True)

    # --- Engine selection (EasySpin Opt.SimulationMode, saffron.m 246-261) ---
    # EasySpin selects the engine explicitly and defaults to 'fast'; torchspin
    # historically inferred it from nonzero Exp.tp. Opt.SimulationMode=None
    # keeps that inference, an explicit value overrides it.
    predefined = isinstance(exp.Sequence, str) and exp.Sequence in _EXP_NAMES
    real_pulses = (exp.tp is not None
                   and any(float(v) != 0.0 for v in np.atleast_1d(exp.tp)))
    # EasySpin saffron.m:351 rejects a predefined sequence with real pulses in
    # every mode. Checked before engine selection because nonzero Exp.tp would
    # otherwise infer the real-pulse engine, whose predefined templates
    # (s_predefinedexperiments) use fixed 16/32 ns pulses and never read Exp.tp,
    # so the requested pulse lengths would be silently ignored.
    if predefined and real_pulses:
        raise ValueError(
            "You cannot use predefined sequences (Exp.Sequence) with real "
            "pulses (Exp.tp). Remove Exp.tp for the ideal-pulse "
            f"{exp.Sequence} simulation, or write the sequence explicitly "
            "for real pulses, e.g. Sequence=[P90, tau, P180, tau] with pulse "
            "dicts {'tp': ..., 'Flip': ...}, Dim1=['d1,d2', dt], DetWindow=0 "
            "and Opt.SimulationMode='thyme'.")
    sim_mode = opt.SimulationMode
    if sim_mode is not None:
        sim_mode = str(sim_mode).lower()
        if sim_mode not in ('fast', 'thyme'):
            raise ValueError(
                f"Opt.SimulationMode must be 'fast' or 'thyme', got "
                f"{opt.SimulationMode!r}.")
    else:
        sequence_is_list = (not isinstance(exp.Sequence, str)
                            and exp.Sequence is not None)
        if exp.DetWindow is not None or sequence_is_list or real_pulses:
            sim_mode = 'thyme'
        else:
            sim_mode = 'fast'

    if sim_mode == 'thyme':
        from torchspin.saffron_thyme import saffron_thyme
        return saffron_thyme(sys, exp, opt)

    # --- Fast (ideal-pulse) algorithm from here on ---
    # EasySpin saffron.m:303 rejects the spidyan-only detection fields
    # (predefined sequences with real pulses were rejected above, saffron.m:351).
    for _f in ('DetSequence', 'DetOperator'):
        if getattr(exp, _f, None) is not None:
            raise ValueError(
                f"Exp.{_f} is specific to spidyan and cannot be used with "
                f"saffron's fast algorithm. Remove it, or select the "
                f"real-pulse engine with Opt.SimulationMode='thyme'.")
    if exp.DetIntegrate:
        raise ValueError(
            "Exp.DetIntegrate only applies to the real-pulse engine. Set "
            "Opt.SimulationMode='thyme' to use it.")
    if not isinstance(exp.Sequence, str):
        raise ValueError(
            "The fast algorithm needs Exp.Sequence to be a predefined "
            f"experiment name ({', '.join(sorted(_EXP_NAMES))}) with "
            "Exp.Flip/Exp.t/Exp.Inc for a custom sequence. A list-valued "
            "Exp.Sequence is the real-pulse syntax; it requires "
            "Opt.SimulationMode='thyme'.")

    # --- Parse experiment ---
    if predefined:
        exp_id = _EXP_NAMES[exp.Sequence]
        n_intervals, n_dims, inc_scheme_id, n_pathways, pf_doc = _EXP_PARAMS[exp_id]
        # MATLAB saffron stores nominal pathway prefactors (+1/2, +1/8, -1/8, ...)
        # but does NOT multiply them into the time-domain output for predefined
        # sequences (see saffron.m comment near line 1660:
        # "No need to normalize out the experiment prefactors ... since they
        # were not included above"). The sign and amplitude of the predefined
        # output are determined by sf_peaks/sf_evolve internally. Apply 1.0
        # as the per-pathway scale so we match MATLAB.
        # Audit: P3A-saffron-amplitude-bug, fixed 2026-04-19.
        pf = [1.0] * n_pathways
        is_endor = (exp_id == 5)
        custom_pathways = None
        # Only custom sequences are renormalized by their pathway prefactors.
        pathway_pf_max = None
    else:
        # Custom pulse sequence
        exp_id = -1
        is_endor = False
        if exp.Flip is None:
            raise ValueError("Custom sequence requires Exp.Flip (flip angles in units of π/2).")
        if exp.Inc is None:
            raise ValueError("Custom sequence requires Exp.Inc (incrementation scheme).")

        n_intervals = len(exp.Flip)
        if len(exp.Inc) != n_intervals:
            raise ValueError("Exp.Inc must have same length as Exp.Flip.")

        # Determine IncSchemeID from Inc pattern
        inc_scheme = tuple(v for v in exp.Inc if v != 0)
        if inc_scheme not in _INC_SCHEME_MAP:
            raise ValueError(
                f"Unsupported incrementation scheme {list(inc_scheme)}. "
                f"Supported: {[list(k) for k in _INC_SCHEME_MAP]}"
            )
        inc_scheme_id = _INC_SCHEME_MAP[inc_scheme]
        n_dims = max(abs(v) for v in exp.Inc) if any(v != 0 for v in exp.Inc) else 1

        # Default initial delays
        if exp.t is None:
            exp.t = [0.0] * n_intervals
        if len(exp.t) != n_intervals:
            raise ValueError("Exp.t must have same length as Exp.Flip.")

        # Default phases (y-phase = 1 in units of π/2)
        if exp.Phase is None:
            exp.Phase = [1.0] * n_intervals

        # Default pulse durations (0 = ideal)
        if exp.tp is None:
            exp.tp = [0.0] * n_intervals
        # Real (finite-duration) pulses were dispatched to the thyme engine
        # above, so every pulse is ideal here.
        ideal_pulse = [tp == 0.0 for tp in exp.tp]

        # Determine refocusing pathways
        from torchspin.saffron_pathways import find_refocusing_pathways
        pathway_list = find_refocusing_pathways(exp.t, exp.Inc)

        # Apply coherence filter
        if exp.Filter is not None:
            if len(exp.Filter) != n_intervals:
                raise ValueError(
                    f"Exp.Filter must have {n_intervals} characters, got {len(exp.Filter)}."
                )
            pathway_list = _apply_coherence_filter(pathway_list, exp.Filter)
            if len(pathway_list) == 0:
                raise ValueError("Exp.Filter is too restrictive: no echo pathways remain.")

        # Parse pathways into propagator indices
        free_l_all, free_r_all, pulse_l_all, pulse_r_all = _pathwayparser(pathway_list)

        # Compute ideal-pulse prefactors
        pf_array = _compute_pathway_prefactors(
            pathway_list, exp.Flip, exp.Phase,
            pulse_l_all, pulse_r_all, ideal_pulse,
        )

        # Remove zero-amplitude pathways
        keep = np.abs(pf_array) > 1e-6
        pathway_list = pathway_list[keep]
        free_l_all = free_l_all[keep]
        free_r_all = free_r_all[keep]
        pulse_l_all = pulse_l_all[keep]
        pulse_r_all = pulse_r_all[keep]
        pf_array = pf_array[keep]

        n_pathways = len(pathway_list)
        if n_pathways == 0:
            raise ValueError("No pathways with nonzero amplitude for this sequence.")
        pf = pf_array.tolist()
        # saffron.m:1669 divides a custom sequence's signal by the largest
        # surviving pathway prefactor (after the <1e-6 pathways are dropped,
        # saffron.m:605). Predefined sequences are not renormalized this way.
        pathway_pf_max = float(np.max(np.abs(pf_array)))

        # Extract incremented-interval indices for sf_peaks/sf_evolve
        inc_mask = [v != 0 for v in exp.Inc]
        idx_inc_l = free_l_all[:, inc_mask]  # (nPathways, nIncIntervals)
        idx_inc_r = free_r_all[:, inc_mask]

        custom_pathways = {
            'pathway_list': pathway_list,
            'free_l': free_l_all,
            'free_r': free_r_all,
            'pulse_l': pulse_l_all,
            'pulse_r': pulse_r_all,
            'idx_inc_l': idx_inc_l,
            'idx_inc_r': idx_inc_r,
        }

    # --- Defaults for dt and nPoints ---
    if exp.dt is None:
        if is_endor:
            pass  # ENDOR doesn't need dt
        else:
            exp.dt = 0.008 if n_dims == 1 else [0.008, 0.008]
    if isinstance(exp.dt, (int, float)):
        exp.dt = [float(exp.dt)] * n_dims
    elif isinstance(exp.dt, (list, tuple)):
        exp.dt = [float(d) for d in exp.dt]

    if exp.nPoints is None:
        if is_endor:
            exp.nPoints = 1001
        elif n_dims == 1:
            exp.nPoints = 512
        else:
            exp.nPoints = [256, 256]
    if isinstance(exp.nPoints, int):
        exp.nPoints = [exp.nPoints] * n_dims

    # --- Relaxation ---
    T1 = sys.T1 if sys.T1 is not None else float('inf')
    T2 = sys.T2 if sys.T2 is not None else float('inf')

    # --- ENDOR range / linewidth ---
    if is_endor:
        if exp.Range is None:
            raise ValueError("Exp.Range is required for MimsENDOR.")
        if sys.lwEndor is None:
            raise ValueError("Sys.lwEndor is required for MimsENDOR.")

    # --- Identify SHF nuclei ---
    n_nuclei = sys.nNuclei
    if opt.Nuclei is not None:
        shf_nuclei = [n - 1 for n in opt.Nuclei]  # 1-based to 0-based
    else:
        shf_nuclei = list(range(n_nuclei))

    # EasySpin: Opt.EndorMethod defaults to 0 for a single SHF nucleus (or when
    # ProductRule splits the problem into one-nucleus subspaces), else 1.
    endor_method = opt.EndorMethod
    if endor_method is None:
        endor_method = 0 if (len(shf_nuclei) == 1 or opt.ProductRule) else 1

    # --- Determine algorithm path ---
    n_electrons = sys.nElectrons
    two_manifolds = (n_electrons == 1 and float(sys.S[0]) == 0.5)
    if n_electrons > 1:
        raise NotImplementedError(
            "saffron's fast algorithm requires a single electron spin "
            "(EasySpin: 'Fast mode does not work for more than one electron "
            "spin'). For multi-electron systems use the real-pulse engine: "
            "set Opt.SimulationMode='thyme' and give every pulse a finite "
            "length (Exp.tp, or a pulse dict with 'tp' in Exp.Sequence)."
        )

    # --- Build nuclear subspaces ---
    nuc_subspaces = _build_nuclear_subspaces(
        sys, shf_nuclei, opt.ProductRule, is_endor, device
    )
    n_subspaces = len(nuc_subspaces)

    # --- Get nuclear spins for normalization ---
    I_active = [sys.I[i] for i in shf_nuclei]
    eq_density_trace = float(np.prod([2*I + 1 for I in I_active])) if I_active else 1.0

    # --- g-tensor in molecular frame (S=1/2 fast path only) ---
    g = sys.g[0].to(torch.float64).to(device)  # (3,) or (3,3)
    if g.ndim == 1:
        if sys.gFrame is not None:
            gf = sys.gFrame[0].to(torch.float64).to(device)
            Rg = erot(gf, device=device).T  # gFrame → mol
            g_mol = Rg @ torch.diag(g) @ Rg.T
        else:
            g_mol = torch.diag(g)
    else:
        g_mol = g

    # --- For general S>1/2: pre-build electronic Hamiltonian ---
    if not two_manifolds:
        # Strip all nuclei to get the core electronic system
        core_sys = nucspinrmv(sys, list(range(sys.nNuclei))) if sys.nNuclei > 0 else sys
        H0_e, mux_e, muy_e, muz_e = _ham_full(core_sys, device=device, dtype=dtype)
        n_el_dim = H0_e.shape[0]
        el_spins = [float(s) for s in (sys.S if hasattr(sys.S, '__iter__') else [sys.S])]
        # Total electron spin operators summed over all spins (here nElectrons==1)
        Sx_e = sop(el_spins, [[1, 1]], dtype=dtype, device=device)
        Sy_e = sop(el_spins, [[1, 2]], dtype=dtype, device=device)
        Sz_e = sop(el_spins, [[1, 3]], dtype=dtype, device=device)

    # --- Orientation selection setup ---
    orientation_selection = (exp.mwFreq is not None and exp.ExciteWidth is not None)
    hstrain_local = sys.HStrain if sys.HStrain is not None else torch.zeros(3, dtype=torch.float64)

    # --- Orientation grid ---
    # EasySpin (p_sampletype.m) decides sample type from CrystalSymmetry or
    # MolFrame only: Exp.SampleFrame alone rotates the sample but leaves a
    # disordered powder a powder. This matches pepper()'s crystal_sample rule.
    crystal_mode = (exp.CrystalSymmetry is not None
                    or exp.MolFrame is not None)
    if exp.SampleFrame is not None and not crystal_mode:
        # torchspin <= 0.4.0 treated SampleFrame alone as a single crystal,
        # which EasySpin does not. Warn rather than change results silently.
        warnings.warn(
            "Exp.SampleFrame was given without Exp.CrystalSymmetry or "
            "Exp.MolFrame, so this is a powder simulation and SampleFrame has "
            "no effect on it (EasySpin p_sampletype.m). torchspin 0.4.0 and "
            "earlier simulated a single crystal here. For a single crystal, "
            "set Exp.CrystalSymmetry (e.g. 'P1') or Exp.MolFrame.",
            UserWarning,
            stacklevel=2,
        )
    if crystal_mode:
        # Single crystal: enumerate symmetry-related sites (EasySpin
        # p_crystalorientations). Each orientation gets weight 4*pi.
        from torchspin.sitetransforms import crystal_orientations
        ang_M2L = crystal_orientations(exp.SampleFrame, exp.CrystalSymmetry,
                                       exp.MolFrame)
        phi_arr = torch.tensor(ang_M2L[:, 0], dtype=torch.float64)
        theta_arr = torch.tensor(ang_M2L[:, 1], dtype=torch.float64)
        chi_arr = torch.tensor(ang_M2L[:, 2], dtype=torch.float64)
        weights_arr = torch.full((ang_M2L.shape[0],), 4.0 * math.pi,
                                 dtype=torch.float64)
    else:
        if opt.GridSymmetry == 'auto':
            try:
                result = hamsymm(sys)
                grid_sym = result[0] if isinstance(result, tuple) else result
            except (RuntimeError, ValueError):
                grid_sym = 'C1'
        else:
            grid_sym = opt.GridSymmetry
        phi_arr, theta_arr, weights_arr, _ = sphgrid(grid_sym, opt.GridSize)
        chi_arr = None
    n_orient = phi_arr.shape[0]

    # --- Prepare output buffers ---
    # EasySpin: Opt.Expand defaults to 4 for 1D and 2 for 2D sequences.
    expand = opt.Expand if opt.Expand is not None else (4 if n_dims == 1 else 2)
    expansion_factor = 2 ** expand
    n_points_exp = [n * expansion_factor for n in exp.nPoints]

    if is_endor:
        rf = np.linspace(exp.Range[0], exp.Range[1], exp.nPoints[0])
        endorspc = np.zeros(exp.nPoints[0])
    else:
        if n_dims == 2:
            total_td = torch.zeros(exp.nPoints[0], exp.nPoints[1],
                                   dtype=dtype, device=device)
        else:
            total_td = torch.zeros(exp.nPoints[0], dtype=dtype, device=device)

        # Per-(pathway, subspace) time-domain storage for
        # ProductRule + TimeDomain (values overwritten each orientation)
        pw_td = {}

        if not opt.TimeDomain:
            if opt.ProductRule:
                pw_buff_re = {}
                pw_buff_im = {}
                for ip in range(n_pathways):
                    for isp in range(n_subspaces):
                        if n_dims == 2:
                            pw_buff_re[(ip, isp)] = torch.zeros(
                                n_points_exp[0], n_points_exp[1],
                                dtype=torch.float64, device=device)
                            pw_buff_im[(ip, isp)] = torch.zeros(
                                n_points_exp[0], n_points_exp[1],
                                dtype=torch.float64, device=device)
                        else:
                            pw_buff_re[(ip, isp)] = torch.zeros(
                                n_points_exp[0], dtype=torch.float64, device=device)
                            pw_buff_im[(ip, isp)] = torch.zeros(
                                n_points_exp[0], dtype=torch.float64, device=device)
            else:
                if n_dims == 2:
                    buff_re = torch.zeros(n_points_exp[0], n_points_exp[1],
                                         dtype=torch.float64, device=device)
                    buff_im = torch.zeros(n_points_exp[0], n_points_exp[1],
                                         dtype=torch.float64, device=device)
                else:
                    buff_re = torch.zeros(n_points_exp[0],
                                         dtype=torch.float64, device=device)
                    buff_im = torch.zeros(n_points_exp[0],
                                         dtype=torch.float64, device=device)

    # ==================================================================
    # Orientation loop
    # ==================================================================
    if is_endor and on_graph:
        raise NotImplementedError('saffron: the Mims-ENDOR path accumulates in NumPy and is not differentiable yet.')
    n_skipped = 0

    for i_ori in range(n_orient):
        phi = float(phi_arr[i_ori])
        theta = float(theta_arr[i_ori])
        chi = float(chi_arr[i_ori]) if chi_arr is not None else 0.0
        ori_weight = float(weights_arr[i_ori])

        # Lab frame axes in molecular frame
        R = erot(torch.tensor([phi, theta, chi], dtype=torch.float64, device=device),
                 device=device)
        zLab_M = R[2]  # (3,)
        yLab_M = R[1]  # (3,)

        # --- Manifold setup and EPR transition finding ---
        if two_manifolds:
            # S=1/2: analytical g-tensor quantization axis
            quant_axis = g_mol.T @ zLab_M
            quant_axis = quant_axis / quant_axis.norm()
            manifold_S = [
                +0.5 * quant_axis,  # EasySpin Ea = upper manifold, i_m=0
                -0.5 * quant_axis,  # EasySpin Eb = lower manifold, i_m=1
            ]
            if orientation_selection:
                if g.ndim == 1:
                    geff_vec = torch.diag(g) @ zLab_M if sys.gFrame is None else g_mol @ zLab_M
                else:
                    geff_vec = g_mol @ zLab_M
                geff = geff_vec.norm()
                nu_mw = geff * BMAGN * exp.Field / (1e3 * PLANCK * 1e6)  # MHz
                hstrain = hstrain_local.to(torch.float64).to(device)
                if hstrain.ndim == 1:
                    lw_vec = torch.diag(hstrain) @ zLab_M
                else:
                    lw_vec = hstrain @ zLab_M
                lw2 = lw_vec.dot(lw_vec) + exp.ExciteWidth**2
                orisel_weight = torch.exp(-(nu_mw - exp.mwFreq * 1e3)**2 / lw2)   # on the graph (g, HStrain)
                if float(orisel_weight.detach()) < opt.OriThreshold:
                    n_skipped += 1
                    continue
            else:
                orisel_weight = 1.0
            # Single transition: a_m=0 (alpha), b_m=1 (beta)
            transitions_list = [(0, 1, orisel_weight)]

        else:
            # S>1/2: diagonalize electronic Hamiltonian, compute <S> per eigenstate
            muzL_e = zLab_M[0]*mux_e + zLab_M[1]*muy_e + zLab_M[2]*muz_e
            H_e = H0_e - exp.Field * muzL_e
            H_e = (H_e + H_e.conj().T) / 2
            E_eig, V_eig = torch.linalg.eigh(H_e)
            manifold_S = []
            for iM in range(n_el_dim):
                vec = V_eig[:, iM]
                manifold_S.append(torch.stack([
                    (vec.conj() @ Sx_e @ vec).real,
                    (vec.conj() @ Sy_e @ vec).real,
                    (vec.conj() @ Sz_e @ vec).real,
                ]))
            # Transition matrix |<i|SyLab|j>| in eigenbasis, optionally excitation-weighted
            SyLab_e = yLab_M[0]*Sx_e + yLab_M[1]*Sy_e + yLab_M[2]*Sz_e
            SyLab_eig = (V_eig.conj().T @ SyLab_e @ V_eig).abs()
            # EasySpin takes the OriThreshold reference level from the
            # *unweighted* |<i|SyLab|j>| matrix, before excitation weighting.
            maxSy = SyLab_eig.max().item()
            if orientation_selection:
                # E_eig ascending; lower triangle (row a > col b) has E_a > E_b
                dE = E_eig.unsqueeze(1) - E_eig.unsqueeze(0)
                excit_amp = torch.exp(-((dE - exp.mwFreq * 1e3) / exp.ExciteWidth)**2)
                SyLab_eig = SyLab_eig * excit_amp
            if maxSy < 1e-30:
                n_skipped += 1
                continue
            # Lower triangle: row a > col b → a is higher-energy (alpha-like, i_m=0)
            lower_tri = torch.tril(SyLab_eig.clone(), -1)
            lower_tri[lower_tri < opt.OriThreshold * maxSy] = 0.0
            a_idx_t, b_idx_t = torch.where(lower_tri > 0)
            if a_idx_t.numel() == 0:
                n_skipped += 1
                continue
            transitions_list = list(zip(
                a_idx_t.tolist(), b_idx_t.tolist(),
                lower_tri[a_idx_t, b_idx_t].tolist(),
            ))

        # --- Loop over EPR transitions (one for S=1/2, multiple for S>1/2) ---
        for a_m, b_m, tr_w in transitions_list:
            # a_m: higher-energy electron eigenstate (alpha-like, maps to i_m=0)
            # b_m: lower-energy electron eigenstate  (beta-like,  maps to i_m=1)
            manifold_S_pair = [manifold_S[a_m], manifold_S[b_m]]
            # EasySpin: under the product rule (non-ENDOR) the per-subspace
            # signals are computed with unit weight; ori/transition weights
            # are applied once when the subspace product is accumulated.
            prefactor_w = ori_weight * tr_w
            if opt.ProductRule and not is_endor:
                prefactor = 1.0
            else:
                prefactor = prefactor_w

            # --- Diagonalize nuclear Hamiltonians per manifold ---
            for i_space in range(n_subspaces):
                sub = nuc_subspaces[i_space]
                manifold_E = [None, None]
                manifold_V = [None, None]

                for i_m in range(2):  # 0=alpha-like, 1=beta-like
                    S_exp = manifold_S_pair[i_m]
                    Hnuc = (exp.Field * (zLab_M[0] * sub.Hnzx +
                                     zLab_M[1] * sub.Hnzy +
                                     zLab_M[2] * sub.Hnzz) +
                        sub.Hnq +
                        S_exp[0] * sub.Hhfx +
                        S_exp[1] * sub.Hhfy +
                        S_exp[2] * sub.Hhfz)
                    Hnuc = (Hnuc + Hnuc.conj().T) / 2  # Hermitianize
                    E_vals, V_mat = torch.linalg.eigh(Hnuc)
                    manifold_E[i_m] = E_vals.real
                    manifold_V[i_m] = V_mat

                # Overlap matrix: M = Va^H @ Vb
                Va = manifold_V[0]  # alpha
                Vb = manifold_V[1]  # beta
                Ea = manifold_E[0]
                Eb = manifold_E[1]
                M = Va.conj().T @ Vb
                Mt = M.conj().T
                n_nuc_states = len(Ea)

                # --- Predefined experiment dispatch ---

                if exp_id == 1:
                    # 2pESEEM: pathway +-, IncSchemeID=2
                    G = (pf[0] * prefactor) * M
                    D = M.clone()
                    T1l = Mt.clone()
                    T1r = Mt.clone()
                    if exp.tau > 0:
                        Q_ = (torch.exp(-2j * math.pi * Ea * exp.tau).unsqueeze(1) *
                              torch.exp(+2j * math.pi * Eb * exp.tau).unsqueeze(0))
                        G = Q_ * G
                        D = Q_.conj() * D

                    if is_endor:
                        pass
                    elif opt.TimeDomain:
                        td_ = sf_evolve(inc_scheme_id, exp.nPoints, exp.dt,
                                        [1, 2], [2, 1], Ea, Eb, G, D, T1l, T1r)
                        if opt.ProductRule:
                            pw_td[(0, i_space)] = td_
                        else:
                            total_td = total_td + td_
                    else:
                        if opt.ProductRule:
                            sf_peaks(inc_scheme_id, pw_buff_re[(0, i_space)],
                                     pw_buff_im[(0, i_space)], exp.dt,
                                     [1, 2], [2, 1], Ea, Eb, G, D, T1l, T1r)
                        else:
                            sf_peaks(inc_scheme_id, buff_re, buff_im, exp.dt,
                                     [1, 2], [2, 1], Ea, Eb, G, D, T1l, T1r)

                elif exp_id == 2:
                    # 3pESEEM: two pathways +alpha- and +beta-, IncSchemeID=1
                    Q_ = (torch.exp(-2j * math.pi * Ea * exp.tau).unsqueeze(1) *
                          torch.exp(+2j * math.pi * Eb * exp.tau).unsqueeze(0))
                    G_ = (pf[0] * prefactor) * Q_ * M
                    D_ = Q_.conj() * M
                    G1 = G_ @ Mt
                    D1 = D_ @ Mt
                    G2 = Mt @ G_
                    D2 = Mt @ D_

                    if exp.T != 0:
                        q_ = torch.exp(-2j * math.pi * Ea * exp.T)
                        G1 = (q_.unsqueeze(1) * q_.unsqueeze(0).conj()) * G1
                        q_ = torch.exp(-2j * math.pi * Eb * exp.T)
                        G2 = (q_.unsqueeze(1) * q_.unsqueeze(0).conj()) * G2

                    for ip, (Gi, Di, fl, fr) in enumerate([
                        (G1, D1, [1], [1]),
                        (G2, D2, [2], [2]),
                    ]):
                        if opt.TimeDomain:
                            td_ = sf_evolve(inc_scheme_id, exp.nPoints, exp.dt,
                                            fl, fr, Ea, Eb, Gi, Di)
                            if opt.ProductRule:
                                pw_td[(ip, i_space)] = td_
                            else:
                                total_td = total_td + td_
                        else:
                            if opt.ProductRule:
                                sf_peaks(inc_scheme_id,
                                         pw_buff_re[(ip, i_space)],
                                         pw_buff_im[(ip, i_space)],
                                         exp.dt, fl, fr, Ea, Eb, Gi, Di)
                            else:
                                sf_peaks(inc_scheme_id, buff_re, buff_im,
                                         exp.dt, fl, fr, Ea, Eb, Gi, Di)

                elif exp_id == 3:
                    # 4pESEEM: two pathways, IncSchemeID=2
                    Q_ = (torch.exp(-2j * math.pi * Ea * exp.tau).unsqueeze(1) *
                          torch.exp(+2j * math.pi * Eb * exp.tau).unsqueeze(0))
                    G_ = (pf[0] * prefactor) * Q_ * M
                    D_ = Q_.conj() * M
                    G1 = G_ @ Mt
                    D1 = Mt @ D_
                    G2 = Mt @ G_
                    D2 = D_ @ Mt
                    Tl1, Tr1 = Mt.clone(), M.clone()
                    Tl2, Tr2 = M.clone(), Mt.clone()

                    if exp.T != 0:
                        q_ = torch.exp(-2j * math.pi * Ea * exp.T)
                        G1 = (q_.unsqueeze(1) * q_.unsqueeze(0).conj()) * G1
                        q_ = torch.exp(-2j * math.pi * Eb * exp.T)
                        G2 = (q_.unsqueeze(1) * q_.unsqueeze(0).conj()) * G2
                        q_ = torch.exp(+2j * math.pi * Eb * exp.T)
                        D1 = (q_.unsqueeze(1) * q_.unsqueeze(0).conj()) * D1
                        q_ = torch.exp(+2j * math.pi * Ea * exp.T)
                        D2 = (q_.unsqueeze(1) * q_.unsqueeze(0).conj()) * D2

                    for ip, (Gi, Di, Tli, Tri, fl, fr) in enumerate([
                        (G1, D1, Tl1, Tr1, [1, 2], [1, 2]),
                        (G2, D2, Tl2, Tr2, [2, 1], [2, 1]),
                    ]):
                        if opt.TimeDomain:
                            td_ = sf_evolve(inc_scheme_id, exp.nPoints, exp.dt,
                                            fl, fr, Ea, Eb, Gi, Di, Tli, Tri)
                            if opt.ProductRule:
                                pw_td[(ip, i_space)] = td_
                            else:
                                total_td = total_td + td_
                        else:
                            if opt.ProductRule:
                                sf_peaks(inc_scheme_id,
                                         pw_buff_re[(ip, i_space)],
                                         pw_buff_im[(ip, i_space)],
                                         exp.dt, fl, fr, Ea, Eb, Gi, Di, Tli, Tri)
                            else:
                                sf_peaks(inc_scheme_id, buff_re, buff_im,
                                         exp.dt, fl, fr, Ea, Eb, Gi, Di, Tli, Tri)

                elif exp_id == 4:
                    # HYSCORE: two pathways, IncSchemeID=11 (2D)
                    Q_ = (torch.exp(-2j * math.pi * Ea * exp.tau).unsqueeze(1) *
                          torch.exp(+2j * math.pi * Eb * exp.tau).unsqueeze(0))
                    G_ = (pf[0] * prefactor) * Q_ * M
                    D_ = Q_.conj() * M
                    G1 = G_ @ Mt
                    G2 = Mt @ G_
                    D1 = Mt @ D_
                    D2 = D_ @ Mt
                    Tl1, Tr1 = Mt.clone(), M.clone()
                    Tl2, Tr2 = M.clone(), Mt.clone()

                    if exp.t1 != 0:
                        q_ = torch.exp(-2j * math.pi * exp.t1 * Ea)
                        G1 = (q_.unsqueeze(1) * q_.unsqueeze(0).conj()) * G1
                        q_ = torch.exp(-2j * math.pi * exp.t1 * Eb)
                        G2 = (q_.unsqueeze(1) * q_.unsqueeze(0).conj()) * G2
                    if exp.t2 != 0:
                        q_ = torch.exp(+2j * math.pi * exp.t2 * Eb)
                        D1 = (q_.unsqueeze(1) * q_.unsqueeze(0).conj()) * D1
                        q_ = torch.exp(+2j * math.pi * exp.t2 * Ea)
                        D2 = (q_.unsqueeze(1) * q_.unsqueeze(0).conj()) * D2

                    for ip, (Gi, Di, Tli, Tri, fl, fr) in enumerate([
                        (G1, D1, Tl1, Tr1, [1, 2], [1, 2]),
                        (G2, D2, Tl2, Tr2, [2, 1], [2, 1]),
                    ]):
                        if opt.TimeDomain:
                            td_ = sf_evolve(inc_scheme_id, exp.nPoints, exp.dt,
                                            fl, fr, Ea, Eb, Gi, Di, Tli, Tri)
                            if opt.ProductRule:
                                pw_td[(ip, i_space)] = td_
                            else:
                                total_td = total_td + td_
                        else:
                            if opt.ProductRule:
                                sf_peaks(inc_scheme_id,
                                         pw_buff_re[(ip, i_space)],
                                         pw_buff_im[(ip, i_space)],
                                         exp.dt, fl, fr, Ea, Eb, Gi, Di, Tli, Tri)
                            else:
                                sf_peaks(inc_scheme_id, buff_re, buff_im,
                                         exp.dt, fl, fr, Ea, Eb, Gi, Di, Tli, Tri)

                elif exp_id == 5:
                    # MimsENDOR
                    # EasySpin convention: Q_[a,b] = exp(-2j*pi*Ea_upper[a]*tau)
                    # * conj(exp(-2j*pi*Eb_lower[b]*tau))  (MATLAB ' = conj transpose)
                    # = exp(-2j*pi*(Eu[a] - El[b])*tau)
                    # In Python naming: Ea=lower(ms=-1/2), Eb=upper(ms=+1/2)
                    # So: Q_[a,b] = exp(-2j*pi*Eb[a]*tau) * exp(+2j*pi*Ea[b]*tau)
                    # and M_eff = Vb†@Va = Mt (V_upper† @ V_lower = EasySpin's M)
                    Q_ = (torch.exp(-2j * math.pi * Eb * exp.tau).unsqueeze(1) *
                          torch.exp(+2j * math.pi * Ea * exp.tau).unsqueeze(0))
                    G_ = prefactor * Q_ * Mt
                    D_ = Q_.conj() * Mt
                    G1 = G_ @ M
                    D1 = D_ @ M
                    G2 = M @ G_
                    D2 = M @ D_

                    if exp.T != 0:
                        # G1 uses upper (Eb) manifold for T evolution
                        q_ = torch.exp(-2j * math.pi * Eb * exp.T)
                        G1 = (q_.unsqueeze(1) * q_.unsqueeze(0).conj()) * G1
                        # G2 uses lower (Ea) manifold for T evolution
                        q_ = torch.exp(-2j * math.pi * Ea * exp.T)
                        G2 = (q_.unsqueeze(1) * q_.unsqueeze(0).conj()) * G2

                    # Remove nuclear coherences
                    G1 = torch.diag(torch.diag(G1))
                    G2 = torch.diag(torch.diag(G2))

                    trG1D1 = torch.trace(G1 @ D1)
                    trG2D2 = torch.trace(G2 @ D2)

                    Gam = sys.lwEndor / math.sqrt(2 * math.log(2))
                    pre_endor = math.sqrt(2 / math.pi) / Gam
                    rf_t = torch.tensor(rf, dtype=torch.float64, device=device)

                    if endor_method == 0:
                        # Population swap (adjacent levels)
                        # G1 = upper (Eb) manifold, G2 = lower (Ea) manifold
                        for j in range(n_nuc_states - 1):
                            i = j + 1
                            G1_ = G1.clone()
                            tmp = G1_[i, i].clone()
                            G1_[i, i] = G1_[j, j]
                            G1_[j, j] = tmp
                            G2_ = G2.clone()
                            tmp = G2_[i, i].clone()
                            G2_[i, i] = G2_[j, j]
                            G2_[j, j] = tmp
                            ampl1 = (trG1D1 - torch.trace(G1_ @ D1)).real.item()
                            ampl2 = (trG2D2 - torch.trace(G2_ @ D2)).real.item()
                            freq1 = (Eb[i] - Eb[j]).item()  # upper manifold
                            freq2 = (Ea[i] - Ea[j]).item()  # lower manifold
                            endorspc += (ampl1 * pre_endor *
                                         np.exp(-2 * ((rf - freq1) / Gam)**2))
                            endorspc += (ampl2 * pre_endor *
                                         np.exp(-2 * ((rf - freq2) / Gam)**2))

                    elif endor_method == 1:
                        # Sum-over-transitions with bandwidth-limited Iy pulse
                        # G1 = upper (Vb) manifold, G2 = lower (Va) manifold
                        IyLab = (yLab_M[0] * sub.Ix +
                                 yLab_M[1] * sub.Iy +
                                 yLab_M[2] * sub.Iz)
                        Iy1 = Vb.conj().T @ IyLab @ Vb
                        Iy2 = Va.conj().T @ IyLab @ Va

                        nu1 = torch.abs(Eb.unsqueeze(1) - Eb.unsqueeze(0))
                        nu2 = torch.abs(Ea.unsqueeze(1) - Ea.unsqueeze(0))

                        fwhm = 1.0 / exp.tprf
                        Gamma = fwhm / (2 * math.sqrt(math.log(2)))
                        theta_rf = math.pi

                        for j in range(n_nuc_states):
                            for i in range(j + 1, n_nuc_states):
                                freq1 = nu1[i, j].item()
                                freq2 = nu2[i, j].item()
                                BW1 = torch.exp(-((nu1 - freq1) / Gamma)**2)
                                BW2 = torch.exp(-((nu2 - freq2) / Gamma)**2)
                                Prfa = torch.linalg.matrix_exp(-1j * theta_rf * (Iy1 * BW1))
                                Prfb = torch.linalg.matrix_exp(-1j * theta_rf * (Iy2 * BW2))
                                G1_ = torch.diag(torch.diag(Prfa @ G1 @ Prfa.conj().T))
                                G2_ = torch.diag(torch.diag(Prfb @ G2 @ Prfb.conj().T))
                                sig1 = (trG1D1 - torch.trace(G1_ @ D1)).real.item()
                                sig2 = (trG2D2 - torch.trace(G2_ @ D2)).real.item()
                                endorspc += (sig1 * pre_endor *
                                             np.exp(-((rf - freq1) / Gam)**2))
                                endorspc += (sig2 * pre_endor *
                                             np.exp(-((rf - freq2) / Gam)**2))

                    elif endor_method == 2:
                        # Sweep method
                        # G1 = upper (Vb) manifold, G2 = lower (Va) manifold
                        IyLab = (yLab_M[0] * sub.Ix +
                                 yLab_M[1] * sub.Iy +
                                 yLab_M[2] * sub.Iz)
                        Iy1 = Vb.conj().T @ IyLab @ Vb
                        Iy2 = Va.conj().T @ IyLab @ Va

                        nu1 = torch.abs(Eb.unsqueeze(1) - Eb.unsqueeze(0))
                        nu2 = torch.abs(Ea.unsqueeze(1) - Ea.unsqueeze(0))

                        fwhm = 1.0 / exp.tprf
                        Gamma = fwhm / (2 * math.sqrt(math.log(2)))
                        theta_rf = math.pi
                        BW_threshold = 0.01

                        for irf in range(len(rf)):
                            BW1 = torch.exp(-((nu1 - rf[irf]) / Gamma)**2)
                            BW2 = torch.exp(-((nu2 - rf[irf]) / Gamma)**2)
                            if BW1.max().item() > BW_threshold:
                                Prfa = torch.linalg.matrix_exp(
                                    -1j * theta_rf * (Iy1 * BW1))
                                G1_ = torch.diag(torch.diag(
                                    Prfa @ G1 @ Prfa.conj().T))
                                endorspc[irf] += (
                                    trG1D1 - torch.trace(G1_ @ D1)).real.item()
                            if BW2.max().item() > BW_threshold:
                                Prfb = torch.linalg.matrix_exp(
                                    -1j * theta_rf * (Iy2 * BW2))
                                G2_ = torch.diag(torch.diag(
                                    Prfb @ G2 @ Prfb.conj().T))
                                endorspc[irf] += (
                                    trG2D2 - torch.trace(G2_ @ D2)).real.item()

                elif exp_id == -1:
                    # Custom sequence — general CTP-based method
                    # Port of MATLAB saffron.m lines 1202-1283
                    eyeN = torch.eye(n_nuc_states, dtype=dtype, device=device)
                    E_manifold = [Ea, Eb]  # index 1=alpha, 2=beta
                    increments = exp.Inc

                    for ip in range(n_pathways):
                        # Build BlockL/BlockR by walking through intervals
                        blocks_l = []
                        blocks_r = []
                        Left = torch.tensor(pf[ip] * prefactor, dtype=dtype, device=device)
                        Right = torch.tensor(1.0, dtype=dtype, device=device)
                        i_block = 0

                        for i_int in range(n_intervals):
                            # Pulse propagator (ideal only)
                            pl = custom_pathways['pulse_l'][ip, i_int]
                            pr = custom_pathways['pulse_r'][ip, i_int]
                            if pl == 3:
                                Left = M @ Left if Left.ndim == 2 else M * Left
                            elif pl == 4:
                                Left = Mt @ Left if Left.ndim == 2 else Mt * Left
                            # pl in (1,2): multiply by identity (no-op for matrix, keep scalar)
                            elif pl in (1, 2) and Left.ndim < 2:
                                pass  # scalar stays scalar

                            if pr == 3:
                                Right = Right @ Mt if Right.ndim == 2 else Right * Mt
                            elif pr == 4:
                                Right = Right @ M if Right.ndim == 2 else Right * M
                            elif pr in (1, 2) and Right.ndim < 2:
                                pass

                            # Free evolution propagator
                            t_delay = exp.t[i_int]
                            if t_delay > 0:
                                fl = custom_pathways['free_l'][ip, i_int]
                                fr = custom_pathways['free_r'][ip, i_int]
                                E_l = E_manifold[fl - 1]
                                E_r = E_manifold[fr - 1]
                                evol_l = torch.exp(-2j * math.pi * E_l * t_delay)
                                evol_r = torch.exp(+2j * math.pi * E_r * t_delay)
                                if Left.ndim == 2:
                                    Left = torch.diag(evol_l) @ Left
                                else:
                                    Left = torch.diag(evol_l) * Left
                                if Right.ndim == 2:
                                    Right = Right @ torch.diag(evol_r)
                                else:
                                    Right = Right * torch.diag(evol_r)

                            # Block boundary at incremented intervals
                            if increments[i_int] != 0:
                                if i_block == 0:
                                    if Left.ndim == 2 and Right.ndim == 2:
                                        G_cust = Left @ Right
                                    elif Left.ndim < 2 and Right.ndim == 2:
                                        G_cust = Left * Right
                                    elif Left.ndim == 2 and Right.ndim < 2:
                                        G_cust = Left * Right
                                    else:
                                        G_cust = Left * Right * eyeN
                                else:
                                    bl = Left if Left.ndim == 2 else eyeN
                                    br = Right if Right.ndim == 2 else eyeN
                                    blocks_l.append(bl)
                                    blocks_r.append(br)
                                i_block += 1
                                Left = torch.tensor(1.0, dtype=dtype, device=device)
                                Right = torch.tensor(1.0, dtype=dtype, device=device)

                        # Detection matrix
                        if increments[-1] != 0:
                            D_cust = M.clone()
                        else:
                            if Right.ndim == 2 and Left.ndim == 2:
                                D_cust = Right @ M @ Left
                            elif Right.ndim < 2 and Left.ndim == 2:
                                D_cust = Right * (M @ Left)
                            elif Right.ndim == 2 and Left.ndim < 2:
                                D_cust = (Right @ M) * Left
                            else:
                                D_cust = Right * Left * M

                        # Build mixing matrix arguments (BlockL interleaved with BlockR)
                        mixing_args = []
                        for bl, br in zip(blocks_l, blocks_r):
                            mixing_args.append(bl)
                            mixing_args.append(br)

                        # Get free_l/free_r for incremented intervals
                        fl_inc = custom_pathways['idx_inc_l'][ip].tolist()
                        fr_inc = custom_pathways['idx_inc_r'][ip].tolist()

                        if opt.TimeDomain:
                            td_ = sf_evolve(inc_scheme_id, exp.nPoints, exp.dt,
                                            fl_inc, fr_inc, Ea, Eb, G_cust, D_cust,
                                            *mixing_args)
                            total_td = total_td + td_
                        else:
                            sf_peaks(inc_scheme_id, buff_re, buff_im, exp.dt,
                                     fl_inc, fr_inc, Ea, Eb, G_cust, D_cust,
                                     *mixing_args)

            # --- Product rule: multiply subspace td signals ---
            if opt.ProductRule and not is_endor:
                for ip in range(n_pathways):
                    td_ = torch.ones(exp.nPoints[0] if n_dims == 1
                                     else (exp.nPoints[0], exp.nPoints[1]),
                                     dtype=dtype, device=device)
                    for isp in range(n_subspaces):
                        if opt.TimeDomain:
                            this_td = pw_td[(ip, isp)]
                        else:
                            buf = (pw_buff_re[(ip, isp)] +
                                   1j * pw_buff_im[(ip, isp)])
                            if n_dims == 1:
                                this_td = torch.fft.ifft(buf) * len(buf)
                                this_td = this_td[:exp.nPoints[0]]
                            else:
                                this_td = torch.fft.ifft2(buf) * buf.numel()
                                this_td = this_td[:exp.nPoints[0], :exp.nPoints[1]]
                            # Reset buffers for next orientation
                            pw_buff_re[(ip, isp)].zero_()
                            pw_buff_im[(ip, isp)].zero_()
                        td_ = td_ * this_td
                    total_td = total_td + prefactor_w * td_

    # ==================================================================
    # Postprocessing
    # ==================================================================
    info = {}

    if is_endor:
        endorspc = np.real(endorspc)
        endorspc /= eq_density_trace
        endorspc /= n_pathways

        if endor_method == 2:
            from torchspin.convspec import convspec
            endorspc_t = torch.tensor(endorspc, dtype=torch.float64)
            dx = rf[1] - rf[0]
            endorspc_t = convspec(endorspc_t, dx, sys.lwEndor)
            endorspc = endorspc_t.numpy()

        x = rf
        signal = endorspc
        info['fd'] = endorspc
        info['td'] = None
        return x, signal, info

    # --- IFFT for frequency-domain accumulation (non-product-rule) ---
    if not opt.ProductRule and not opt.TimeDomain:
        buf = buff_re + 1j * buff_im
        if n_dims == 1:
            td = torch.fft.ifft(buf) * len(buf)
            td = td[:exp.nPoints[0]]
        else:
            td = torch.fft.ifft2(buf) * buf.numel()
            td = td[:exp.nPoints[0], :exp.nPoints[1]]
    else:
        td = total_td

    # --- Normalize ---
    td = td / eq_density_trace
    td = td / n_pathways
    if pathway_pf_max is not None:
        # EasySpin saffron.m:1669, custom (non-predefined) sequences only.
        td = td / pathway_pf_max

    # --- Time axes ---
    if exp_id == 1:
        t1 = np.arange(exp.nPoints[0]) * exp.dt[0] + exp.tau
    elif exp_id == 2:
        t1 = np.arange(exp.nPoints[0]) * exp.dt[0] + exp.tau + exp.T
    elif exp_id == 3:
        t1 = np.arange(exp.nPoints[0]) * exp.dt[0] + exp.T
    elif exp_id == 4:
        t1 = np.arange(exp.nPoints[0]) * exp.dt[0] + exp.t1
        t2 = np.arange(exp.nPoints[1]) * exp.dt[1] + exp.t2
    else:
        t1 = np.arange(exp.nPoints[0]) * exp.dt[0]
        if n_dims == 2:
            t2 = np.arange(exp.nPoints[1]) * exp.dt[1]

    # --- Relaxation decay (torch) ---
    td = torch.as_tensor(td)
    if not torch.is_complex(td):
        td = td.to(torch.complex128)
    rdt = torch.float64
    t1_t = torch.as_tensor(t1, dtype=rdt, device=td.device)
    if T1 != float('inf') or T2 != float('inf'):
        if exp_id == 1 and T2 != float('inf'):
            td = td * torch.exp(-2 * t1_t / T2)
        elif exp_id == 2 and T1 != float('inf'):
            td = td * math.exp(-2 * exp.tau / T2) * torch.exp(-t1_t / T1)
        elif exp_id == 4 and T1 != float('inf'):
            t2_t = torch.as_tensor(t2, dtype=rdt, device=td.device)
            decay = torch.exp(-t1_t / T1).reshape(-1, 1) * torch.exp(-t2_t / T1).reshape(1, -1)
            if T2 != float('inf'):
                decay = decay * math.exp(-2 * exp.tau / T2)
            td = td * decay
    info['td'] = td
    # --- Apodization + FFT (torch) ---
    if n_dims == 1:
        tdx = td - td.mean()
        win = torch.as_tensor(np.asarray(apowin(opt.Window, tdx.numel()), dtype=float), dtype=rdt, device=td.device)
        tdx = tdx * win
        n_fft = opt.ZeroFillFactor * tdx.numel()
        fd = torch.fft.fftshift(torch.fft.fft(tdx, n=n_fft))
        f1 = _fdaxis(exp.dt[0], fd.numel())
        info['fd'] = fd
        info['f'] = f1
        x = t1
        signal = td
    else:
        w1 = torch.as_tensor(np.asarray(apowin(opt.Window, exp.nPoints[0]), dtype=float), dtype=rdt, device=td.device)
        w2 = torch.as_tensor(np.asarray(apowin(opt.Window, exp.nPoints[1]), dtype=float), dtype=rdt, device=td.device)
        tdx = td - td.mean()
        n_fft = [opt.ZeroFillFactor * exp.nPoints[0],
                 opt.ZeroFillFactor * exp.nPoints[1]]
        fd = torch.fft.fftshift(torch.fft.fftn(tdx * (w1.reshape(-1, 1) * w2.reshape(1, -1)), s=n_fft))
        f1 = _fdaxis(exp.dt[0], fd.shape[0])
        f2 = _fdaxis(exp.dt[1], fd.shape[1])
        info['fd'] = fd
        info['f1'] = f1
        info['f2'] = f2
        x = [t1, t2]
        signal = td
    if not on_graph:
        # API unchanged for plain simulations: NumPy outputs. Tensors on the
        # autograd graph are returned only when the spin system requires grad.
        signal = signal.detach().cpu().numpy()
        info['td'] = info['td'].detach().cpu().numpy()
        info['fd'] = info['fd'].detach().cpu().numpy()
    return x, signal, info


# =====================================================================
# Utilities
# =====================================================================

def _fdaxis(dt: float, n: int) -> np.ndarray:
    """Frequency axis for FFT output (after fftshift), in MHz."""
    return np.linspace(-0.5 / dt, 0.5 / dt, n, endpoint=False)
