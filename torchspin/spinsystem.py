"""SpinSystem dataclass for torchspin.

Mirrors EasySpin's ``Sys`` struct.  A ``SpinSystem`` collects all spin quantum
numbers, interaction tensors, and Euler frame angles needed to build the full
spin Hamiltonian.

Minimal usage::

    import torch
    from torchspin import SpinSystem

    # S=1/2 radical with g anisotropy
    sys = SpinSystem(S=[0.5], g=[[2.005, 2.004, 2.002]])

    # S=1/2 with one 14N nucleus and isotropic hyperfine
    sys = SpinSystem(
        S=[0.5],
        g=[[2.006, 2.006, 2.002]],
        Nucs=['14N'],
        A=[[10.0, 10.0, 50.0]],   # MHz
    )

Units
-----
* All fields involving energies / splittings are in **MHz** (matching EasySpin).
* Magnetic fields passed to Hamiltonian functions are in **mT**.
* Euler angles are in **radians**.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from torchspin.nucdata import nucspin, nucgval, nucabund, nucdata
from torchspin.isotopologues import is_isotope_mixture


def _quadrupole_spin(nuc: str) -> float:
    """Nuclear spin for the Q shorthand conversion; for isotope mixtures the
    most abundant isotope with I >= 1 (0 if none)."""
    if not is_isotope_mixture(nuc):
        return float(_nuclear_spin(nuc))
    from torchspin.isotopologues import _parse_nucleus
    element, mass, abund = _parse_nucleus(nuc, None) if '(' not in nuc else (None, None, None)
    if element is None:   # custom mixture: use the listed isotopes
        inner = nuc[nuc.index('(') + 1:nuc.index(')')]
        element = nuc[nuc.index(')') + 1:]
        mass = [int(x) for x in inner.replace(' ', '').split(',') if x]
        abund = [1.0] * len(mass)
    best, best_ab = 0.0, -1.0
    for m, ab in zip(mass, abund):
        I = float(_nuclear_spin(f'{m}{element}'))
        if I >= 1 and ab > best_ab:
            best, best_ab = I, ab
    return best


def _nuclear_spin(isotope: str) -> float:
    """Return the nuclear spin quantum number for *isotope*."""
    return nucspin(isotope)


def _nuclear_gn(isotope: str) -> float:
    """Return the nuclear g-factor for *isotope*."""
    return nucgval(isotope)


# ---------------------------------------------------------------------------
# SpinSystem
# ---------------------------------------------------------------------------

@dataclass
class SpinSystem:
    """Spin system specification, analogous to EasySpin's ``Sys`` struct.

    Parameters
    ----------
    S:
        List of electron spin quantum numbers, e.g. ``[0.5]`` or
        ``[0.5, 1.0]``.
    g:
        Electron g-tensors.  Shape ``(nElectrons, 3)`` for diagonal tensors
        (principal values) or ``(3*nElectrons, 3)`` for full 3×3 tensors.
        Defaults to ``gfree`` for all electrons if not given.
    gFrame:
        Euler angles (alpha, beta, gamma) in radians rotating the molecular
        frame to the g-tensor principal frame.  Shape ``(nElectrons, 3)``.
        Defaults to zero (no rotation).
    D:
        Zero-field splitting tensors (MHz).  Shape ``(nElectrons, 3)`` for
        diagonal (Dxx, Dyy, Dzz) or ``(3*nElectrons, 3)`` for full tensors.
        Pass ``None`` (default) to omit ZFS entirely.
    DFrame:
        Euler angles for D-tensor frame.  Shape ``(nElectrons, 3)``.
    B:
        Stevens ZFS coefficients (MHz) for high-order terms.  List of tensors
        indexed by rank k starting at 2: ``B[0]`` = rank-2, ``B[1]`` = rank-4, etc.
        Each ``B[k]`` has shape ``(nElectrons, 2*k_actual+1)`` where the columns
        correspond to q = k_actual, k_actual-1, ..., -k_actual.
        If D is present, rank-2 Stevens terms are skipped.
    BFrame:
        Euler angles for each B tensor.  Same list structure as B.
    Nucs:
        List of nuclear isotope strings, e.g. ``['14N', '1H']``.
    A:
        Hyperfine coupling tensors (MHz).  Shape ``(nNuclei, 3*nElectrons)``
        for diagonal entries (Axx, Ayy, Azz per electron), or
        ``(3*nNuclei, 3*nElectrons)`` for full tensors.  Defaults to zero.
    AFrame:
        Euler angles for each A-tensor.  Shape ``(nNuclei, 3*nElectrons)``.
    ee:
        Electron-electron coupling tensors (MHz).  Shape ``(nPairs, 3)`` for
        diagonal or ``(3*nPairs, 3)`` for full.
    eeFrame:
        Euler angles for each ee coupling tensor.  Shape ``(nPairs, 3)``.
    gnscale:
        Per-nucleus scaling of the nuclear g-factor (dimensionless).  Shape
        ``(nNuclei,)``.  Defaults to 1.0 for each nucleus.
    sigma:
        Chemical shielding tensor (dimensionless) per nucleus.  Shape
        ``(nNuclei, 3)`` for diagonal or ``(3*nNuclei, 3)`` for full.
        Defaults to identity (no shielding effect).
    sigmaFrame:
        Euler angles for chemical shielding frame.  Shape ``(nNuclei, 3)``.
    Q:
        Nuclear quadrupole coupling tensors (MHz).  Shape ``(nNuclei, 3)``
        for diagonal or ``(3*nNuclei, 3)`` for full.  ``None`` means no
        quadrupole interaction.
    QFrame:
        Euler angles for Q-tensor frame.  Shape ``(nNuclei, 3)``.
    nn:
        Nuclear-nuclear coupling tensors (MHz).  Shape ``(nPairs_nuc, 3)``
        for diagonal or ``(3*nPairs_nuc, 3)`` for full.
    nnFrame:
        Euler angles for nn coupling tensor frame.
    lw:
        Linewidth ``[GaussianFWHM, LorentzianFWHM]`` in mT, used by
        ``pepper``.  Defaults to ``[0.0, 0.0]`` (no broadening, as in EasySpin).
    H0Extra:
        Extra field-independent Hamiltonian terms, a list of ``(coeff, op)``
        pairs added to H0 as ``sum(coeff * op)`` (MHz) by :func:`torchspin.ham`
        on both the ``(H0, mu)`` and the ``ham(sys, B0)`` path.  ``coeff`` is a
        real scalar (float or 0-d tensor, may require grad); ``op`` is an
        ``(nStates, nStates)`` Hermitian matrix (tensor or array, real or
        complex) in the full spin-system basis, ordered as :func:`torchspin.sop`
        builds it (electrons, then nuclei, then OAMs).  Not an EasySpin field.
        Solvers that build the Hamiltonian without ``ham()`` (perturbation
        theory, hybrid, garlic, chili, fastmotion, cardamom, saffron's fast
        algorithm, nucspinrmv/nucspinkeep/nucspinadd) raise when it is set.
        Default ``None`` (no extra term).
    """

    S: list[float]
    g: Optional[list | torch.Tensor] = None
    gFrame: Optional[list | torch.Tensor] = None
    D: Optional[list | torch.Tensor] = None
    D_: Optional[list | torch.Tensor] = None   # zero-field splitting as [D, E/D] per electron (EasySpin Sys.D_)
    DFrame: Optional[list | torch.Tensor] = None
    B: Optional[list[Optional[torch.Tensor]]] = None
    BFrame: Optional[list[Optional[torch.Tensor]]] = None
    Nucs: list[str] = field(default_factory=list)
    n: Optional[list[int]] = None  # Number of equivalent nuclei per entry in Nucs (EasySpin Sys.n); default 1
    A: Optional[list | torch.Tensor] = None
    A_: Optional[list | torch.Tensor] = None  # spherical hyperfine form per nucleus/electron: [aiso], [aiso T] or [aiso T rho] (EasySpin Sys.A_)
    AFrame: Optional[list | torch.Tensor] = None
    ee: Optional[list | torch.Tensor] = None
    ee2: Optional[list | torch.Tensor] = None  # isotropic biquadratic exchange per electron pair, +ee2·(S1·S2)² (MHz)
    eeFrame: Optional[list | torch.Tensor] = None
    # Nuclear interaction fields
    gnscale: Optional[list | torch.Tensor] = None
    sigma: Optional[list | torch.Tensor] = None
    sigmaFrame: Optional[list | torch.Tensor] = None
    Q: Optional[list | torch.Tensor] = None
    QFrame: Optional[list | torch.Tensor] = None
    nn: Optional[list | torch.Tensor] = None
    nnFrame: Optional[list | torch.Tensor] = None
    # Linewidth for spectrum simulation
    lw: Optional[list[float]] = None
    lwpp: Optional[list[float]] = None  # Peak-to-peak linewidth (alternative to lw)
    # Strain parameters for anisotropic broadening
    HStrain: Optional[list | torch.Tensor] = None  # [3] MHz, molecular frame
    gStrain: Optional[list | torch.Tensor] = None  # [nElectrons, 3]
    AStrain: Optional[list | torch.Tensor] = None  # [3] MHz (first nucleus only)
    DStrain: Optional[list | torch.Tensor] = None  # MHz, [FWHM_D, FWHM_E] (one row per electron)
    gAStrainCorr: float = 1.0  # +1 (correlated) or -1 (anticorrelated)
    DStrainCorr: float | list[float] = 0.0  # Correlation coefficient between D and E (per electron)
    # Multi-component spectra (mirrors MATLAB Sys.weight)
    weight: float = 1.0  # Relative intensity weight for multi-component simulations
    Abund: Optional[list] = None  # isotope abundances for custom mixtures in Nucs, e.g. '(63,65)Cu' (EasySpin Sys.Abund)
    initState: Optional[object] = None  # non-equilibrium state: (pops_or_rho, basis), density matrix, 'T0', 'singlet' (EasySpin Sys.initState)
    tdm: Optional[object] = None        # transition dipole moment direction in the molecular frame: 'x', [phi theta] or 3-vector (EasySpin Sys.tdm)
    # Slow-motion dynamics (chili)
    tcorr: Optional[float] = None    # Isotropic rotational correlation time (s)
    logtcorr: Optional[float] = None # log10(tcorr) — alternative to tcorr
    Diff: Optional[float] = None     # Isotropic rotational diffusion coefficient (s^-1)
    logDiff: Optional[float] = None  # log10(Diff) — alternative to Diff
    # Pulse EPR (saffron)
    T1: Optional[float] = None       # Longitudinal relaxation time (µs)
    T2: Optional[float] = None       # Transverse relaxation time (µs)
    lwEndor: Optional[float] = None  # ENDOR linewidth FWHM (MHz)
    # Trajectory-based simulation (cardamom)
    Potential: Optional[list | torch.Tensor | np.ndarray] = None  # Orienting potential: [L,M,K,λ] rows or 3D grid
    DiffGlobal: Optional[float | list] = None  # Global rotational diffusion rate (s⁻¹)
    TransRates: Optional[list | torch.Tensor | np.ndarray] = None  # Transition rate matrix (nStates×nStates)
    TransProb: Optional[list | torch.Tensor | np.ndarray] = None   # Transition probability matrix (nStates×nStates)
    Orientations: Optional[list | torch.Tensor | np.ndarray] = None  # Euler angles per state (nStates×3 or 3×nStates)
    # Orbital angular momentum (OAM)
    L: Optional[list[float]] = None       # OAM quantum numbers, one per OAM
    gL: Optional[list[float]] = None      # Orbital g-factors (default 1.0 per OAM)
    soc: Optional[list | torch.Tensor] = None  # Spin-orbit coupling [nElectrons, nOrders] MHz
    # Crystal-field parameters (Stevens operator coefficients, MHz)
    CF1: Optional[list | torch.Tensor] = None   # rank 1, shape (nL, 3)
    CF2: Optional[list | torch.Tensor] = None   # rank 2, shape (nL, 5)
    CF3: Optional[list | torch.Tensor] = None   # rank 3, shape (nL, 7)
    CF4: Optional[list | torch.Tensor] = None   # rank 4, shape (nL, 9)
    CF5: Optional[list | torch.Tensor] = None   # rank 5, shape (nL, 11)
    CF6: Optional[list | torch.Tensor] = None   # rank 6, shape (nL, 13)
    CF7: Optional[list | torch.Tensor] = None   # rank 7, shape (nL, 15)
    CF8: Optional[list | torch.Tensor] = None   # rank 8, shape (nL, 17)
    CF9: Optional[list | torch.Tensor] = None   # rank 9, shape (nL, 19)
    CF10: Optional[list | torch.Tensor] = None  # rank 10, shape (nL, 21)
    CF11: Optional[list | torch.Tensor] = None  # rank 11, shape (nL, 23)
    CF12: Optional[list | torch.Tensor] = None  # rank 12, shape (nL, 25)
    # Higher-order Zeeman (McGavin-Tennant-Weil formalism)
    # Keys: '{lB}{lS}{l}' strings (e.g., '110', '112', '222')
    # Values: (nElectrons, 2*l+1) arrays of coefficients
    Ham: Optional[dict] = None
    # Generic extra field-independent terms: H0 += sum_k coeff_k * op_k (MHz).
    # coeff: real scalar (float or 0-d tensor, may require grad); op: (n, n)
    # Hermitian matrix in the full spin-system basis (torchspin order, as sop
    # builds it). Default None: no extra term.
    H0Extra: Optional[list] = None

    def __post_init__(self) -> None:
        from torchspin.constants import GFREE

        # Normalize S
        if isinstance(self.S, (int, float)):
            self.S = [float(self.S)]
        else:
            self.S = [float(s) for s in self.S]

        # Normalize Nucs
        if isinstance(self.Nucs, str):
            if self.Nucs:
                # Split comma-separated nuclei, strip whitespace
                # split on commas outside parentheses ('(63,65)Cu,N' → ['(63,65)Cu', 'N'])
                self.Nucs = [nuc.strip() for nuc in re.split(r',(?![^()]*\))', self.Nucs) if nuc.strip()]
            else:
                self.Nucs = []

        # Normalize L (orbital angular momenta)
        if self.L is not None:
            if isinstance(self.L, (int, float)):
                self.L = [float(self.L)]
            else:
                self.L = [float(l) for l in self.L]
            # Default gL to 1.0 per OAM
            if self.gL is None:
                self.gL = [1.0] * len(self.L)
            elif isinstance(self.gL, (int, float)):
                self.gL = [float(self.gL)] * len(self.L)
            else:
                self.gL = [float(gl) for gl in self.gL]
            # Normalize soc to tensor
            if self.soc is not None:
                self.soc = _to_tensor(self.soc, torch.float64)
                if self.soc.ndim == 1:
                    self.soc = self.soc.unsqueeze(0)

        # --- Ham (higher-order Zeeman) ---
        if self.Ham is not None:
            ham_dict = {}
            for key, val in self.Ham.items():
                ham_dict[key] = _to_tensor(val, torch.float64)
                if ham_dict[key].ndim == 1:
                    ham_dict[key] = ham_dict[key].unsqueeze(0)
            self.Ham = ham_dict

        n_e = self.nElectrons
        n_n = self.nNuclei
        from itertools import combinations
        n_pairs = sum(1 for _ in combinations(range(n_e), 2))

        # --- g tensor ---
        # EasySpin: Sys.g may not be combined with Ham110/Ham112 (the linear
        # Zeeman term is then given entirely by the Ham parameters, g = 0).
        _ham_linear = self.Ham is not None and any(
            k in ('110', '112', 'Ham110', 'Ham112') for k in self.Ham.keys()
        )
        if self.g is None:
            if _ham_linear:
                self.g = torch.zeros(n_e, 3, dtype=torch.float64)
            else:
                # Default: isotropic gfree for each electron
                self.g = torch.tensor(
                    [[GFREE, GFREE, GFREE]] * n_e, dtype=torch.float64
                )
        else:
            self.g = _to_tensor(self.g, torch.float64)
            if _ham_linear and bool(torch.any(self.g != 0)):
                raise ValueError(
                    "Cannot use g and Ham112 or Ham110 simultaneously. Remove one of them."
                )
            # Normalize g to shape (nElectrons, 3) or (3*nElectrons, 3)
            if self.g.ndim == 0:
                # Scalar: isotropic g for all electrons
                self.g = self.g.expand(n_e, 3).clone()
            elif self.g.ndim == 1:
                if self.g.shape[0] == n_e:
                    # [g1, g2, ...]: one isotropic g per electron (EasySpin column form)
                    self.g = self.g.unsqueeze(1).expand(n_e, 3).clone()
                elif self.g.shape[0] == 3:
                    # [gx, gy, gz] for one electron (replicated for all)
                    self.g = self.g.unsqueeze(0).expand(n_e, 3).clone()
                elif self.g.shape[0] == 3 * n_e:
                    # Full 3×3 matrices stacked as vector
                    self.g = self.g  # keep as is
            elif self.g.ndim == 2 and self.g.shape == (n_e, 1):
                self.g = self.g.expand(n_e, 3).clone()
            # Otherwise assume correct 2D shape

        # --- gFrame ---
        if self.gFrame is None:
            self.gFrame = torch.zeros(n_e, 3, dtype=torch.float64)
        else:
            self.gFrame = _to_tensor(self.gFrame, torch.float64)
            if self.gFrame.ndim == 1:
                self.gFrame = self.gFrame.unsqueeze(0)

        # --- D_ = [D, E/D] per electron (EasySpin Sys.D_) → D = [D, E] ---
        if self.D_ is not None:
            if self.D is not None:
                raise ValueError('Zero-field splitting given in both D and D_. Please remove one of them.')
            D_ = _to_tensor(self.D_, torch.float64).reshape(-1, 2)
            if D_.shape[0] != n_e:
                raise ValueError('D_ has wrong size: expected one [D, E/D] row per electron.')
            self.D = torch.stack([D_[:, 0], D_[:, 0] * D_[:, 1]], dim=1)
            self.D_ = D_
        # --- D tensor ---
        if self.D is not None:
            self.D = _to_tensor(self.D, torch.float64)
            # Normalize D: scalar -> (n_e, 3), vector of length 3 -> (n_e, 3)
            if self.D.ndim == 0:
                # Scalar D (EasySpin: D with E = 0) -> [-1 -1 2]/3 * D for all electrons
                self.D = (self.D * torch.tensor([-1.0, -1.0, 2.0], dtype=torch.float64) / 3).unsqueeze(0).expand(n_e, 3).clone()
            elif self.D.ndim == 1 and n_e > 1 and self.D.shape[0] == n_e and n_e not in (2, 3):
                # [D1, D2, ...]: one scalar D (E = 0) per electron (EasySpin column form)
                self.D = (self.D.unsqueeze(1) * torch.tensor([-1.0, -1.0, 2.0], dtype=torch.float64) / 3).clone()
            elif self.D.ndim == 2 and self.D.shape == (n_e, 1):
                # (n_e, 1) column: one scalar D (E = 0) per electron
                self.D = (self.D * torch.tensor([-1.0, -1.0, 2.0], dtype=torch.float64) / 3).clone()
            elif self.D.ndim == 1:
                if self.D.shape[0] == 2:
                    # EasySpin [D, E] shorthand: convert to [Dxx, Dyy, Dzz]
                    D_val, E_val = self.D[0], self.D[1]
                    Dxx = -D_val / 3 + E_val
                    Dyy = -D_val / 3 - E_val
                    Dzz = 2 * D_val / 3
                    principal = torch.stack([Dxx, Dyy, Dzz])
                    self.D = principal.unsqueeze(0).expand(n_e, 3).clone()
                elif self.D.shape[0] == 3:
                    # [Dxx, Dyy, Dzz] for one electron
                    self.D = self.D.unsqueeze(0).expand(n_e, 3).clone()
                elif self.D.shape[0] == 3 * n_e:
                    # Full 3×3 matrices stacked as vector
                    self.D = self.D  # keep as is
            elif self.D.ndim == 2 and self.D.shape[1] == 2:
                # (n_e, 2) EasySpin [D, E] per electron: convert each row
                D_vals = self.D[:, 0]
                E_vals = self.D[:, 1]
                Dxx = -D_vals / 3 + E_vals
                Dyy = -D_vals / 3 - E_vals
                Dzz = 2 * D_vals / 3
                self.D = torch.stack([Dxx, Dyy, Dzz], dim=1).clone()
            # Otherwise assume correct 2D shape

        # --- DFrame ---
        if self.DFrame is None:
            self.DFrame = torch.zeros(n_e, 3, dtype=torch.float64)
        else:
            self.DFrame = _to_tensor(self.DFrame, torch.float64)
            if self.DFrame.ndim == 1:
                self.DFrame = self.DFrame.unsqueeze(0)

        # --- B (Stevens ZFS coefficients) ---
        # B is a list indexed by rank offset: B[0] = k=2, B[1] = k=4, etc.
        # Each B[i] is (nElectrons, 2*k+1) for the corresponding k
        if self.B is not None:
            # Convert each element to tensor if present
            for i, Bk in enumerate(self.B):
                if Bk is not None:
                    self.B[i] = _to_tensor(Bk, torch.float64)

        # --- BFrame ---
        if self.BFrame is not None:
            for i, BFk in enumerate(self.BFrame):
                if BFk is not None:
                    self.BFrame[i] = _to_tensor(BFk, torch.float64)

        # --- n (equivalent nuclei, EasySpin Sys.n) ---
        # Each entry of Nucs may stand for a group of n identical, equivalent
        # nuclei (same isotope, same isotropic coupling).  Only garlic uses
        # this (via equivcouple); pepper/chili reject n>1 like EasySpin.
        if self.n is None:
            self.n = [1] * n_n
        else:
            if isinstance(self.n, torch.Tensor):
                self.n = self.n.flatten().tolist()
            elif isinstance(self.n, (int, float)):
                self.n = [self.n]
            self.n = [int(round(float(v))) for v in self.n]
            if len(self.n) != n_n:
                raise ValueError(
                    f"SpinSystem.n has {len(self.n)} entries, expected one per nucleus ({n_n})."
                )
            if any(v < 1 for v in self.n):
                raise ValueError("SpinSystem.n entries must be positive integers.")

        # --- A_ (spherical hyperfine form, EasySpin validatespinsys) → A principal values
        #     A = aiso + T·[-1 -1 2] + rho·[-1 +1 0], per (nucleus, electron) block ---
        if n_n > 0 and self.A_ is not None:
            if self.A is not None:
                raise ValueError('Hyperfine data given in both A and A_. Please remove one of them.')
            A_ = _to_tensor(self.A_, torch.float64)
            if A_.ndim == 1:
                A_ = A_.reshape(n_n, -1) if (A_.numel() == n_n and n_n > 1) else A_.reshape(1, -1)
            if A_.shape[0] != n_n:
                raise ValueError(f'A_ must have one row per nucleus ({n_n}).')
            per_e = A_.shape[1] // n_e
            if A_.shape[1] != per_e * n_e or per_e not in (1, 2, 3):
                raise ValueError('A_ needs 1, 2 or 3 values per nucleus and electron: [aiso], [aiso T] or [aiso T rho].')
            cols = []
            for e in range(n_e):
                blk = A_[:, e * per_e:(e + 1) * per_e]
                aiso = blk[:, 0]
                T = blk[:, 1] if per_e >= 2 else torch.zeros(n_n, dtype=torch.float64)
                rho = blk[:, 2] if per_e == 3 else torch.zeros(n_n, dtype=torch.float64)
                cols.append(torch.stack([aiso - T - rho, aiso - T + rho, aiso + 2 * T], dim=1))
            self.A = torch.cat(cols, dim=1)
            self.A_ = A_
        # --- A tensor ---
        if n_n > 0 and self.A is not None:
            self.A = _to_tensor(self.A, torch.float64)
            # Normalize A: 
            # scalar -> (n_n, 3*n_e)
            # 1D vector handling: must check length == nNuclei BEFORE length == 3
            # to match MATLAB's issize(Sys.A,[1 nNuclei]) precedence
            if self.A.ndim == 0:
                # Scalar A: isotropic for all nuclei
                self.A = self.A.expand(n_n, 3 * n_e).clone()
            elif self.A.ndim == 1:
                if n_e == 1 and self.A.shape[0] == n_n:
                    # [A1, A2, ..., An] for n nuclei, 1 electron
                    # Expand isotropic values: [[A1,A1,A1], [A2,A2,A2], ...]
                    # Matches MATLAB: Sys.A = Sys.A(:)*[1 1 1]
                    self.A = self.A.unsqueeze(1).expand(n_n, 3).clone()
                elif self.A.shape[0] == 3:
                    # [Ax, Ay, Az] for one nucleus, one electron
                    # Expand to all nuclei (principal values)
                    self.A = self.A.unsqueeze(0).expand(n_n, 3).clone()
            # Otherwise assume correct 2D shape (n_n, 3*n_e)
        elif n_n > 0 and self.A is None:
            # zero hyperfine by default
            self.A = torch.zeros(n_n, 3 * n_e, dtype=torch.float64)

        # --- AFrame ---
        if self.AFrame is None:
            self.AFrame = torch.zeros(n_n, 3 * n_e, dtype=torch.float64)
        elif n_n > 0:
            self.AFrame = _to_tensor(self.AFrame, torch.float64)
            # Normalize AFrame: 1D [α, β, γ] for one nucleus -> (n_n, 3*n_e)
            if self.AFrame.ndim == 1:
                if self.AFrame.shape[0] == 3:
                    # Euler angles for one nucleus, one electron
                    self.AFrame = self.AFrame.unsqueeze(0).expand(n_n, 3).clone()
            # Otherwise assume correct 2D shape

        # --- ee2 (biquadratic exchange, one value per electron pair) ---
        if n_pairs > 0:
            if self.ee2 is None:
                self.ee2 = torch.zeros(n_pairs, dtype=torch.float64)
            else:
                self.ee2 = _to_tensor(self.ee2, torch.float64).reshape(-1)
                if self.ee2.numel() != n_pairs:
                    raise ValueError(f'ee2 must have one value per electron pair ({n_pairs}).')
        # --- ee tensor ---
        if n_pairs > 0 and self.ee is not None:
            self.ee = _to_tensor(self.ee, torch.float64)
            # Normalize ee: scalar -> (n_pairs, 3), vector of length 3 -> (n_pairs, 3)
            if self.ee.ndim == 0:
                # Scalar ee: replicate for all pairs
                self.ee = self.ee.expand(n_pairs, 3).clone()
            elif self.ee.ndim == 1:
                if self.ee.shape[0] == 3 and n_pairs != 3:
                    # [Jxx, Jyy, Jzz] for one pair
                    self.ee = self.ee.unsqueeze(0).expand(n_pairs, 3).clone()
                elif self.ee.shape[0] == n_pairs:
                    # One isotropic J per pair (EasySpin column form)
                    self.ee = self.ee.unsqueeze(1).expand(n_pairs, 3).clone()
                elif self.ee.shape[0] == 3:
                    # Ambiguous n_pairs == 3 with 3 values: treat as
                    # principal values for every pair (legacy behavior)
                    self.ee = self.ee.unsqueeze(0).expand(n_pairs, 3).clone()
                elif self.ee.shape[0] == 3 * n_pairs:
                    # Full 3×3 matrices stacked as vector
                    self.ee = self.ee  # keep as is
            # Otherwise assume correct 2D shape
        elif n_pairs > 0 and self.ee is None:
            self.ee = torch.zeros(n_pairs, 3, dtype=torch.float64)

        # --- eeFrame ---
        if self.eeFrame is None:
            self.eeFrame = torch.zeros(max(n_pairs, 1), 3, dtype=torch.float64)
        else:
            self.eeFrame = _to_tensor(self.eeFrame, torch.float64)
            # Normalize eeFrame: 1D [α, β, γ] for one pair -> (n_pairs, 3)
            if self.eeFrame.ndim == 1 and self.eeFrame.shape[0] == 3:
                self.eeFrame = self.eeFrame.unsqueeze(0).expand(n_pairs, 3).clone()
            # Otherwise assume correct 2D shape

        # --- gnscale ---
        if self.gnscale is None:
            self.gnscale = torch.ones(max(n_n, 1), dtype=torch.float64)
        else:
            self.gnscale = _to_tensor(self.gnscale, torch.float64).reshape(-1)
            if self.gnscale.numel() == 1 and n_n > 1:
                self.gnscale = self.gnscale.expand(n_n).clone()

        # --- sigma (chemical shielding) ---
        # Convention matches MATLAB EasySpin: sigma defaults to ones(nNuclei, 3),
        # and ham_nz uses pre * sigma * Ik directly (no (I-sigma) transformation).
        def _per_nucleus_rows(T: torch.Tensor, n_rows: int, isotropic_scalar: bool) -> torch.Tensor:
            """Normalize 0-D / 1-D per-nucleus (or per-pair) input to (n_rows, 3)."""
            if T.ndim == 0:
                return T.expand(n_rows, 3).clone()
            if T.ndim == 1:
                if n_rows > 1 and T.shape[0] == n_rows and (n_rows != 3 or isotropic_scalar):
                    # one isotropic value per row (EasySpin column form)
                    return T.unsqueeze(1).expand(n_rows, 3).clone()
                if T.shape[0] == 3:
                    return T.unsqueeze(0).expand(n_rows, 3).clone()
                if T.shape[0] == 3 * n_rows:
                    return T.reshape(n_rows, 3).clone()
            if T.ndim == 2 and T.shape == (n_rows, 1):
                return T.expand(n_rows, 3).clone()
            return T

        if self.sigma is None:
            self.sigma = torch.ones(max(n_n, 1), 3, dtype=torch.float64)
        else:
            self.sigma = _per_nucleus_rows(_to_tensor(self.sigma, torch.float64), max(n_n, 1), True)

        # --- sigmaFrame ---
        if self.sigmaFrame is None:
            self.sigmaFrame = torch.zeros(max(n_n, 1), 3, dtype=torch.float64)
        else:
            self.sigmaFrame = _per_nucleus_rows(_to_tensor(self.sigmaFrame, torch.float64), max(n_n, 1), False)

        # --- Q tensor ---
        # EasySpin forms (validatespinsys): one value per nucleus (e2qQ/h, eta=0),
        # [e2qQ/h eta] per nucleus, principal values [Qx Qy Qz] per nucleus, or a
        # full (3·nNuclei, 3) tensor.  The first two are converted to principal
        # values e2qQ/h/(4I(2I-1))·[-1+eta, -1-eta, 2] (zero for I < 1).
        if self.Q is not None:
            self.Q = _to_tensor(self.Q, torch.float64)
            Q = self.Q
            if Q.ndim == 0:
                Q = Q.reshape(1, 1)
            elif Q.ndim == 1:
                if Q.shape[0] == n_n:
                    Q = Q.reshape(n_n, 1)
                elif Q.shape[0] == 2 * n_n:
                    Q = Q.reshape(n_n, 2)
                elif Q.shape[0] == 3 * n_n:
                    Q = Q.reshape(n_n, 3)
            if Q.ndim == 2 and Q.shape[0] == n_n and Q.shape[1] in (1, 2):
                rows = []
                for i in range(n_n):
                    nuc = self.Nucs[i]
                    I_q = _quadrupole_spin(nuc)
                    # torch ops only, so a Q that requires grad stays on the graph
                    e2qQh = Q[i, 0]
                    eta = Q[i, 1] if Q.shape[1] == 2 else torch.zeros_like(e2qQh)
                    if I_q < 1:
                        rows.append(torch.zeros(3, dtype=torch.float64))
                    else:
                        rows.append(e2qQh / (4 * I_q * (2 * I_q - 1))
                                    * torch.stack([-1 + eta, -1 - eta, 2 * torch.ones_like(eta)]))
                Q = torch.stack(rows)
            self.Q = Q

        # --- QFrame ---
        if self.QFrame is None:
            self.QFrame = torch.zeros(max(n_n, 1), 3, dtype=torch.float64)
        else:
            self.QFrame = _per_nucleus_rows(_to_tensor(self.QFrame, torch.float64), max(n_n, 1), False)

        # --- nn tensor (nuclear-nuclear coupling) ---
        from itertools import combinations as _comb
        n_nuc_pairs = sum(1 for _ in _comb(range(n_n), 2))
        if self.nn is not None:
            self.nn = _to_tensor(self.nn, torch.float64)
            if n_nuc_pairs > 0:
                if self.nn.ndim == 0:
                    self.nn = self.nn.expand(n_nuc_pairs, 3).clone()
                elif self.nn.ndim == 1:
                    if self.nn.shape[0] == n_nuc_pairs and n_nuc_pairs != 3:
                        # one isotropic coupling per pair (EasySpin column form)
                        self.nn = self.nn.unsqueeze(1).expand(n_nuc_pairs, 3).clone()
                    elif self.nn.shape[0] == 3:
                        self.nn = self.nn.unsqueeze(0).expand(n_nuc_pairs, 3).clone()
                elif self.nn.ndim == 2 and self.nn.shape == (n_nuc_pairs, 1):
                    self.nn = self.nn.expand(n_nuc_pairs, 3).clone()

        # --- nnFrame ---
        if self.nnFrame is None:
            self.nnFrame = torch.zeros(max(n_nuc_pairs, 1), 3, dtype=torch.float64)
        else:
            self.nnFrame = _per_nucleus_rows(_to_tensor(self.nnFrame, torch.float64), max(n_nuc_pairs, 1), False)

        # --- lw and lwpp ---
        # Handle linewidth: lw (FWHM) and lwpp (peak-to-peak) are mutually exclusive
        # Keep both fields to track which one the user specified (important for fitting)
        # Conversion: lw = lwpp * sqrt(3) for both Gaussian and Lorentzian
        if self.lw is not None and self.lwpp is not None:
            raise ValueError("Specify either lw or lwpp, not both.")
        
        # Normalize to [Gaussian, Lorentzian] format
        if self.lwpp is not None:
            if isinstance(self.lwpp, (int, float)):
                self.lwpp = [float(self.lwpp), 0.0]
            elif len(self.lwpp) == 1:
                self.lwpp = [float(self.lwpp[0]), 0.0]
        
        if self.lw is not None:
            if isinstance(self.lw, (int, float)):
                self.lw = [float(self.lw), 0.0]
            elif len(self.lw) == 1:
                self.lw = [float(self.lw[0]), 0.0]
        
        # Set default if neither specified
        if self.lw is None and self.lwpp is None:
            self.lw = [0.0, 0.0]  # Default: no broadening (EasySpin Sys.lw = 0)

        # --- Strain parameters ---
        # HStrain: [3] MHz in molecular frame
        if self.HStrain is not None:
            self.HStrain = _to_tensor(self.HStrain, torch.float64)
            if self.HStrain.ndim == 0:
                self.HStrain = self.HStrain.expand(3).clone()
            elif self.HStrain.shape[0] != 3:
                raise ValueError("HStrain must have shape [3].")

        # gStrain: [nElectrons, 3] unitless
        if self.gStrain is not None:
            self.gStrain = _to_tensor(self.gStrain, torch.float64)
            if self.gStrain.ndim == 0:
                # Scalar: isotropic for all electrons
                self.gStrain = self.gStrain.expand(n_e, 3).clone()
            elif self.gStrain.ndim == 1:
                if self.gStrain.shape[0] == 3:
                    # [3] -> expand to all electrons
                    self.gStrain = self.gStrain.unsqueeze(0).expand(n_e, 3).clone()
                else:
                    raise ValueError("gStrain must have shape [3] or [nElectrons, 3].")

        # AStrain: [3] MHz (first nucleus only)
        if self.AStrain is not None:
            self.AStrain = _to_tensor(self.AStrain, torch.float64)
            if self.AStrain.ndim == 0:
                self.AStrain = self.AStrain.expand(3).clone()
            elif self.AStrain.shape[0] != 3:
                raise ValueError("AStrain must have shape [3].")

        # DStrain: normalized to (nElectrons, 2) = [FWHM_D, FWHM_E] per electron
        # (EasySpin Sys.DStrain: one row per electron spin, 1-3 columns).
        if self.DStrain is not None:
            self.DStrain = _to_tensor(self.DStrain, torch.float64)
            if self.DStrain.ndim == 0:
                self.DStrain = self.DStrain.reshape(1, 1)
            elif self.DStrain.ndim == 1:
                if n_e > 1 and self.DStrain.shape[0] == n_e and n_e > 3:
                    self.DStrain = self.DStrain.unsqueeze(1)          # one FWHM_D per electron
                else:
                    self.DStrain = self.DStrain.unsqueeze(0)          # one row, all electrons
            if self.DStrain.shape[1] > 3 or self.DStrain.shape[1] < 1:
                raise ValueError("DStrain must have 1, 2 or 3 columns ([FWHM_D, FWHM_E, ...]).")
            if self.DStrain.shape[0] == 1 and n_e > 1:
                self.DStrain = self.DStrain.expand(n_e, self.DStrain.shape[1]).clone()
            if self.DStrain.shape[0] != n_e:
                raise ValueError(f"DStrain must have {n_e} rows, one per electron spin.")
            if self.DStrain.shape[1] == 1:
                self.DStrain = torch.cat([self.DStrain, torch.zeros(n_e, 1, dtype=torch.float64)], dim=1)
            else:
                self.DStrain = self.DStrain[:, :2].clone()
        # DStrainCorr: one D/E correlation coefficient per electron
        if isinstance(self.DStrainCorr, torch.Tensor):
            self.DStrainCorr = self.DStrainCorr.flatten().tolist()
        if isinstance(self.DStrainCorr, (int, float)):
            self.DStrainCorr = [float(self.DStrainCorr)] * n_e
        else:
            self.DStrainCorr = [float(v) for v in self.DStrainCorr]
            if len(self.DStrainCorr) == 1 and n_e > 1:
                self.DStrainCorr = self.DStrainCorr * n_e
            if len(self.DStrainCorr) != n_e:
                raise ValueError(f"DStrainCorr must have {n_e} entries, one per electron spin.")

        # --- H0Extra (generic extra field-independent operators) ---
        if self.H0Extra is not None:
            self.H0Extra = _normalize_h0extra(self)

    # ------------------------------------------------------------------
    # Read-only derived properties
    # ------------------------------------------------------------------

    @property
    def nElectrons(self) -> int:
        return len(self.S)

    @property
    def nNuclei(self) -> int:
        return len(self.Nucs)

    @property
    def I(self) -> list[float]:
        """Nuclear spin quantum numbers (derived from Nucs)."""
        return [_nuclear_spin(iso) for iso in self.Nucs]

    @property
    def gn(self) -> list[float]:
        """Nuclear g-factors (derived from Nucs)."""
        return [_nuclear_gn(iso) for iso in self.Nucs]

    @property
    def nL(self) -> int:
        """Number of orbital angular momenta."""
        return len(self.L) if self.L is not None else 0

    @property
    def Spins(self) -> list[float]:
        """All spin quantum numbers: electrons first, then nuclei, then OAMs."""
        oam = [float(l) for l in self.L] if self.L is not None else []
        return self.S + self.I + oam

    @property
    def nStates(self) -> int:
        """Total Hilbert space dimension."""
        return math.prod(int(round(2 * s + 1)) for s in self.Spins)

    @property
    def fullg(self) -> bool:
        """True if g is given as full 3×3 matrices."""
        if self.g.ndim < 2:
            return False
        return self.g.shape[0] == 3 * self.nElectrons

    @property
    def fullD(self) -> bool:
        """True if D is given as full 3×3 matrices."""
        if self.D is None:
            return False
        if self.D.ndim < 2:
            return False
        return self.D.shape[0] == 3 * self.nElectrons

    @property
    def fullA(self) -> bool:
        """True if A is given as full 3×3 matrices per nucleus per electron."""
        if self.A is None or self.nNuclei == 0:
            return False
        return self.A.shape[0] == 3 * self.nNuclei

    @property
    def fullee(self) -> bool:
        """True if ee is given as full 3×3 matrices."""
        if self.ee is None:
            return False
        from itertools import combinations
        n_pairs = sum(1 for _ in combinations(range(self.nElectrons), 2))
        return n_pairs > 0 and self.ee.shape[0] == 3 * n_pairs

    @property
    def fullsigma(self) -> bool:
        """True if sigma is given as full 3×3 matrices."""
        return self.sigma.shape[0] == 3 * max(self.nNuclei, 1)

    @property
    def fullQ(self) -> bool:
        """True if Q is given as full 3×3 matrices."""
        if self.Q is None:
            return False
        return self.Q.shape[0] == 3 * self.nNuclei

    @property
    def fullnn(self) -> bool:
        """True if nn is given as full 3×3 matrices."""
        if self.nn is None:
            return False
        from itertools import combinations
        n_pairs = sum(1 for _ in combinations(range(self.nNuclei), 2))
        return n_pairs > 0 and self.nn.shape[0] == 3 * n_pairs

    def get_lw(self) -> list[float]:
        """Get effective linewidth as [Gaussian_FWHM, Lorentzian_FWHM] in mT.
        
        If lwpp is specified, converts to lw using:
        - Gaussian: lw = lwpp * sqrt(2*ln(2)) ≈ 1.177 * lwpp
        - Lorentzian: lw = lwpp * sqrt(3) ≈ 1.732 * lwpp
        
        This matches MATLAB EasySpin convention (see validatespinsys.m).
        
        Returns
        -------
        list[float]
            [Gaussian_FWHM, Lorentzian_FWHM] in mT
        """
        import math
        
        if self.lw is not None:
            return self.lw
        elif self.lwpp is not None:
            # Convert lwpp to lw using correct factors for each lineshape
            # Gaussian: sqrt(2*log(2)) for derivative peak-to-peak to FWHM
            # Lorentzian: sqrt(3) for derivative peak-to-peak to FWHM
            gauss_factor = math.sqrt(2.0 * math.log(2.0))  # ≈ 1.177
            lorentz_factor = math.sqrt(3.0)  # ≈ 1.732
            return [self.lwpp[0] * gauss_factor, self.lwpp[1] * lorentz_factor]
        else:
            # Should not happen due to __post_init__, but handle gracefully
            return [0.0, 0.0]

    def validate(self) -> None:
        """Comprehensive spin system validation.

        Checks quantum numbers, tensor shapes, frame angles, interaction
        consistency, and physical plausibility.  Raises ``ValueError`` on
        the first problem found.

        Port of EasySpin's ``private/validatespinsys.m``.
        """
        from itertools import combinations

        n_e = self.nElectrons
        n_n = self.nNuclei
        n_pairs = sum(1 for _ in combinations(range(n_e), 2))
        n_nuc_pairs = sum(1 for _ in combinations(range(n_n), 2))

        # --- Electron spins ---
        if len(self.S) == 0:
            raise ValueError("At least one electron spin is required (S list is empty).")
        for s in self.S:
            if s <= 0:
                raise ValueError(f"Electron spin must be positive, got S={s}.")
            if (2 * s) % 1 != 0:
                raise ValueError(
                    f"Electron spin must be a positive multiple of 1/2, got S={s}."
                )

        # --- g tensor ---
        if self.g.shape[0] not in (n_e, 3 * n_e):
            raise ValueError(
                f"g tensor has {self.g.shape[0]} rows, expected {n_e} or {3*n_e} "
                f"for {n_e} electron(s)."
            )
        if self.g.ndim == 2 and self.g.shape[1] != 3:
            raise ValueError(
                f"g tensor has {self.g.shape[1]} columns, expected 3."
            )

        # --- gFrame ---
        if self.gFrame.shape != (n_e, 3):
            raise ValueError(
                f"gFrame shape {tuple(self.gFrame.shape)} does not match "
                f"expected ({n_e}, 3)."
            )

        # --- D tensor ---
        if self.D is not None:
            if self.D.shape[0] not in (n_e, 3 * n_e):
                raise ValueError(
                    f"D tensor has {self.D.shape[0]} rows, expected {n_e} or "
                    f"{3*n_e} for {n_e} electron(s)."
                )
            # ZFS requires S >= 1
            for i, s in enumerate(self.S):
                if s < 1:
                    # Check if D is actually nonzero for this electron
                    if not self.fullD:
                        row = self.D[i]
                        if torch.any(row != 0):
                            raise ValueError(
                                f"D tensor is nonzero for electron {i} with S={s}, "
                                f"but ZFS requires S >= 1."
                            )

        # --- DFrame ---
        if self.DFrame.shape != (n_e, 3):
            raise ValueError(
                f"DFrame shape {tuple(self.DFrame.shape)} does not match "
                f"expected ({n_e}, 3)."
            )

        # --- A tensor ---
        if n_n > 0 and self.A is not None:
            expected_cols = 3 * n_e
            if self.A.shape[0] not in (n_n, 3 * n_n):
                raise ValueError(
                    f"A tensor has {self.A.shape[0]} rows, expected {n_n} or "
                    f"{3*n_n} for {n_n} nucleus/nuclei."
                )
            if self.A.ndim == 2 and self.A.shape[1] != expected_cols:
                raise ValueError(
                    f"A tensor has {self.A.shape[1]} columns, expected "
                    f"{expected_cols} (3 × {n_e} electrons)."
                )

        # --- AFrame ---
        if n_n > 0 and self.AFrame is not None:
            if self.AFrame.shape[0] not in (n_n, 3 * n_n):
                raise ValueError(
                    f"AFrame has {self.AFrame.shape[0]} rows, expected {n_n}."
                )

        # --- Nuclei ---
        for nuc in self.Nucs:
            if is_isotope_mixture(nuc):
                # 'Cu' / '(63,65)Cu': expanded by torchspin.isotopologues before simulation
                continue
            try:
                nucdata(nuc)
            except (ValueError, KeyError):
                raise ValueError(
                    f"Unknown nucleus '{nuc}'. Use isotope notation like '14N', '1H'."
                )
        if n_n > 0 and self.A is None:
            raise ValueError(
                "Nuclei are specified (Nucs) but no hyperfine coupling (A) is given."
            )

        # --- Q tensor ---
        if self.Q is not None:
            if self.Q.shape[0] not in (n_n, 3 * n_n):
                raise ValueError(
                    f"Q tensor has {self.Q.shape[0]} rows, expected {n_n} or "
                    f"{3*n_n} for {n_n} nucleus/nuclei."
                )
            # Quadrupole requires I >= 1
            for i, nuc in enumerate(self.Nucs):
                if is_isotope_mixture(nuc):
                    continue
                I_val = _nuclear_spin(nuc)
                if I_val < 1 and not self.fullQ:
                    row = self.Q[i]
                    if torch.any(row != 0):
                        raise ValueError(
                            f"Q tensor is nonzero for nucleus '{nuc}' (I={I_val}), "
                            f"but quadrupole interaction requires I >= 1."
                        )

        # --- ee coupling ---
        if n_e > 1:
            if self.ee is not None:
                if self.ee.shape[0] not in (n_pairs, 3 * n_pairs):
                    raise ValueError(
                        f"ee tensor has {self.ee.shape[0]} rows, expected {n_pairs} "
                        f"or {3*n_pairs} for {n_pairs} electron pair(s)."
                    )

        # --- nn coupling ---
        if self.nn is not None:
            if self.nn.shape[0] not in (n_nuc_pairs, 3 * n_nuc_pairs):
                raise ValueError(
                    f"nn tensor has {self.nn.shape[0]} rows, expected {n_nuc_pairs} "
                    f"or {3*n_nuc_pairs} for {n_nuc_pairs} nuclear pair(s)."
                )

        # --- Strain parameters ---
        if self.HStrain is not None and self.HStrain.shape[0] != 3:
            raise ValueError("HStrain must have 3 elements.")
        if self.gStrain is not None:
            if self.gStrain.shape != (n_e, 3):
                raise ValueError(
                    f"gStrain shape {tuple(self.gStrain.shape)} does not match "
                    f"expected ({n_e}, 3)."
                )
        if self.AStrain is not None and self.AStrain.shape[0] != 3:
            raise ValueError("AStrain must have 3 elements.")
        if self.DStrain is not None and tuple(self.DStrain.shape) != (self.nElectrons, 2):
            raise ValueError("DStrain must have shape (nElectrons, 2) = [FWHM_D, FWHM_E] per electron.")

        # --- Linewidth ---
        if self.lw is not None and self.lwpp is not None:
            raise ValueError("Specify either lw or lwpp, not both.")
        if self.lw is not None:
            if any(v < 0 for v in self.lw):
                raise ValueError("Linewidth (lw) values must be non-negative.")
        if self.lwpp is not None:
            if any(v < 0 for v in self.lwpp):
                raise ValueError("Linewidth (lwpp) values must be non-negative.")

        # --- OAM ---
        if self.L is not None:
            for l_val in self.L:
                if l_val < 0 or (2 * l_val) % 1 != 0:
                    raise ValueError(
                        f"OAM quantum number must be a non-negative multiple of "
                        f"1/2, got L={l_val}."
                    )
            if self.gL is not None and len(self.gL) != len(self.L):
                raise ValueError(
                    f"gL has {len(self.gL)} elements but L has {len(self.L)}."
                )

        # --- Dynamics ---
        if self.tcorr is not None and self.tcorr <= 0:
            raise ValueError("tcorr must be positive.")
        if self.Diff is not None and self.Diff <= 0:
            raise ValueError("Diff must be positive.")
        if self.T1 is not None and self.T1 <= 0:
            raise ValueError("T1 must be positive.")
        if self.T2 is not None and self.T2 <= 0:
            raise ValueError("T2 must be positive.")

        # --- Mutual exclusivity: dynamics parameters ---
        if self.tcorr is not None and self.logtcorr is not None:
            raise ValueError(
                "Specify either tcorr or logtcorr, not both."
            )
        if self.Diff is not None and self.logDiff is not None:
            raise ValueError(
                "Specify either Diff or logDiff, not both."
            )

        # --- Mutual exclusivity: D vs B2 (Stevens rank-2) ---
        if self.D is not None and self.B is not None:
            has_D = torch.any(self.D != 0).item()
            has_B2 = (len(self.B) > 0 and self.B[0] is not None
                      and torch.any(self.B[0] != 0).item())
            if has_D and has_B2:
                raise ValueError(
                    "Cannot use both D and B2 (Stevens rank-2 coefficients) "
                    "simultaneously. Use one or the other."
                )

        # --- Strain correlation bounds ---
        if any(r < -1 or r > 1 for r in (self.DStrainCorr if isinstance(self.DStrainCorr, (list, tuple)) else [self.DStrainCorr])):
            raise ValueError(
                f"DStrainCorr must be between -1 and +1, got {self.DStrainCorr}."
            )
        if self.gAStrainCorr not in (1.0, -1.0):
            raise ValueError(
                f"gAStrainCorr must be +1 (correlated) or -1 (anticorrelated), "
                f"got {self.gAStrainCorr}."
            )

        # --- Strain non-negativity ---
        if self.HStrain is not None and torch.any(self.HStrain < 0):
            raise ValueError("HStrain values must be non-negative.")
        if self.gStrain is not None and torch.any(self.gStrain < 0):
            raise ValueError("gStrain values must be non-negative.")
        if self.AStrain is not None and torch.any(self.AStrain < 0):
            raise ValueError("AStrain values must be non-negative.")
        if self.DStrain is not None and torch.any(self.DStrain < 0):
            raise ValueError("DStrain values must be non-negative.")

        # --- Q matrix symmetry (when full 3x3 blocks) ---
        if self.Q is not None and self.fullQ:
            for i in range(n_n):
                Q_block = self.Q[3 * i:3 * i + 3, :]
                if not torch.allclose(Q_block, Q_block.T, atol=1e-10):
                    raise ValueError(
                        f"Q tensor for nucleus {i} ('{self.Nucs[i]}') is not "
                        f"symmetric. Full Q matrices must be symmetric."
                    )

        # --- Crystal field coherence with L ---
        for k in range(1, 13):
            cf_field = getattr(self, f'CF{k}', None)
            if cf_field is not None:
                if self.L is None:
                    raise ValueError(
                        f"CF{k} requires orbital angular momentum L to be specified."
                    )
                for j, l_val in enumerate(self.L):
                    if k > 2 * l_val:
                        raise ValueError(
                            f"CF{k} rank {k} exceeds 2*L={2*l_val} for "
                            f"orbital {j}."
                        )


# ---------------------------------------------------------------------------
# Nuclear spin manipulation functions
# ---------------------------------------------------------------------------

def _nucleus_indices(sys: 'SpinSystem', idx) -> list[int]:
    """Normalize nucleus index/indices to a sorted list of 0-based ints."""
    n = sys.nNuclei
    if isinstance(idx, (int, np.integer)):
        idx = [int(idx)]
    else:
        idx = [int(i) for i in idx]
    for i in idx:
        if i < 0 or i >= n:
            raise ValueError(f"Nucleus index {i} out of range (0-based, nNuclei={n}).")
    return sorted(set(idx))


def nucspinkeep(sys: 'SpinSystem', idx) -> 'SpinSystem':
    """Return a new SpinSystem with only the specified nuclei kept.

    Parameters
    ----------
    sys : SpinSystem
        Input spin system.
    idx : int or sequence of int
        0-based nucleus index/indices to keep.

    Returns
    -------
    SpinSystem
        New spin system with only the specified nuclei.

    Examples
    --------
    >>> sys2 = nucspinkeep(sys, [0, 2])   # keep nuclei 0 and 2
    """
    _reject_h0extra(sys, 'nucspinkeep')
    keep = _nucleus_indices(sys, idx)
    return _filter_nuclei(sys, keep)


def nucspinrmv(sys: 'SpinSystem', idx) -> 'SpinSystem':
    """Return a new SpinSystem with the specified nuclei removed.

    Parameters
    ----------
    sys : SpinSystem
        Input spin system.
    idx : int or sequence of int
        0-based nucleus index/indices to remove.

    Returns
    -------
    SpinSystem
        New spin system without the specified nuclei.

    Examples
    --------
    >>> sys2 = nucspinrmv(sys, 1)   # remove nucleus 1
    """
    _reject_h0extra(sys, 'nucspinrmv')
    rmv = _nucleus_indices(sys, idx)
    keep = [i for i in range(sys.nNuclei) if i not in rmv]
    return _filter_nuclei(sys, keep)


def nucspinadd(sys: 'SpinSystem', nuc: str, A=None, Q=None,
               AFrame=None, QFrame=None, gnscale=1.0) -> 'SpinSystem':
    """Return a new SpinSystem with one nucleus appended.

    Parameters
    ----------
    sys : SpinSystem
        Input spin system.
    nuc : str
        Isotope string, e.g. ``'1H'`` or ``'14N'``.
    A : array_like or None
        Hyperfine coupling tensor (MHz).  Shape ``(3,)`` for diagonal
        [Ax, Ay, Az] or ``(3, 3)`` for a full matrix.  ``None`` → zero.
    Q : array_like or None
        Nuclear quadrupole coupling tensor (MHz).  Only for I ≥ 1.
    AFrame : array_like or None
        Euler angles for A-tensor frame.  Shape ``(3,)`` or ``(3*nElectrons,)``.
    QFrame : array_like or None
        Euler angles for Q-tensor frame.  Shape ``(3,)``.
    gnscale : float
        Nuclear g-factor scale factor (dimensionless).  Default 1.0.

    Returns
    -------
    SpinSystem
        New spin system with the nucleus appended.

    Examples
    --------
    >>> sys2 = nucspinadd(sys, '1H', A=[5.0, 5.0, 10.0])
    """
    _reject_h0extra(sys, 'nucspinadd')
    import copy, numpy as np

    n_e = sys.nElectrons
    n_n = sys.nNuclei

    new_Nucs = list(sys.Nucs) + [nuc]

    # Build new A tensor: existing rows + new row
    if A is None:
        A_new_row = torch.zeros(1, 3 * n_e, dtype=torch.float64)
    else:
        A_arr = torch.tensor(np.asarray(A, dtype=float), dtype=torch.float64)
        if A_arr.ndim == 0:
            A_new_row = A_arr.expand(3).unsqueeze(0).expand(1, 3 * n_e).clone()
        elif A_arr.ndim == 1 and A_arr.shape[0] == 3:
            # [Ax, Ay, Az] — replicate for all electrons
            A_new_row = A_arr.repeat(n_e).unsqueeze(0)  # (1, 3*n_e)
        else:
            A_new_row = A_arr.reshape(1, -1)

    if n_n == 0 or sys.A is None:
        new_A = A_new_row
    else:
        # sys.A may have shape (n_n, 3*n_e)
        existing_A = sys.A.reshape(n_n, 3 * n_e) if sys.A.ndim == 1 else sys.A
        new_A = torch.cat([existing_A, A_new_row], dim=0)

    # AFrame: new row
    if AFrame is None:
        AF_new_row = torch.zeros(1, 3 * n_e, dtype=torch.float64)
    else:
        AF_new_row = torch.tensor(np.asarray(AFrame, dtype=float), dtype=torch.float64).reshape(1, -1)
    existing_AF = sys.AFrame.reshape(n_n, 3 * n_e) if n_n > 0 else torch.zeros(0, 3 * n_e, dtype=torch.float64)
    new_AFrame = torch.cat([existing_AF, AF_new_row], dim=0) if n_n > 0 else AF_new_row

    # gnscale: append
    existing_gs = sys.gnscale[:n_n] if n_n > 0 else torch.zeros(0, dtype=torch.float64)
    new_gnscale = torch.cat([existing_gs, torch.tensor([gnscale], dtype=torch.float64)])

    # Q: append if provided
    new_Q = None
    if Q is not None:
        Q_new = torch.tensor(np.asarray(Q, dtype=float), dtype=torch.float64)
        if Q_new.ndim == 1:
            Q_new = Q_new.unsqueeze(0)
        if sys.Q is not None:
            new_Q = torch.cat([sys.Q.reshape(n_n, 3) if sys.Q.ndim > 1 else sys.Q.reshape(-1, 3),
                                Q_new], dim=0)
        else:
            # Pad existing nuclei with zeros
            new_Q = torch.cat([torch.zeros(n_n, 3, dtype=torch.float64), Q_new], dim=0)

    # QFrame: append
    QF_new = torch.tensor(np.asarray(QFrame, dtype=float), dtype=torch.float64).reshape(3) if QFrame else torch.zeros(3, dtype=torch.float64)
    existing_QF = sys.QFrame[:n_n] if n_n > 0 and sys.QFrame is not None else torch.zeros(0, 3, dtype=torch.float64)
    if n_n > 0:
        new_QFrame = torch.cat([existing_QF.reshape(n_n, 3), QF_new.unsqueeze(0)], dim=0)
    else:
        new_QFrame = QF_new.unsqueeze(0)

    return SpinSystem(
        S=list(sys.S),
        g=sys.g.clone(),
        gFrame=sys.gFrame.clone(),
        D=sys.D.clone() if sys.D is not None else None,
        DFrame=sys.DFrame.clone() if sys.DFrame is not None else None,
        B=sys.B,
        BFrame=sys.BFrame,
        Nucs=new_Nucs,
        A=new_A,
        AFrame=new_AFrame,
        Q=new_Q,
        QFrame=new_QFrame,
        gnscale=new_gnscale,
        lw=list(sys.lw) if sys.lw else None,
        lwpp=list(sys.lwpp) if sys.lwpp else None,
        HStrain=sys.HStrain.clone() if sys.HStrain is not None else None,
        gStrain=sys.gStrain.clone() if sys.gStrain is not None else None,
        AStrain=sys.AStrain.clone() if sys.AStrain is not None else None,
        DStrain=sys.DStrain.clone() if sys.DStrain is not None else None,
        weight=sys.weight,
    )


def isotopologues(sys: 'SpinSystem') -> list[tuple[float, 'SpinSystem']]:
    """Generate all isotopologues of a spin system with natural-abundance weights.

    For each nucleus in *sys*, queries ``nucabund`` to find the natural
    abundance of its isotope.  Returns a list of ``(weight, SpinSystem)``
    tuples where *weight* is the product of abundances for all nuclei.

    Parameters
    ----------
    sys : SpinSystem
        Input spin system (typically with specific isotopes, e.g. ``'14N'``).

    Returns
    -------
    list of (float, SpinSystem)
        Each entry is ``(weight, sys_variant)`` where *weight* is the natural
        abundance fraction and *sys_variant* is a new SpinSystem identical to
        *sys* but with *weight* set.  The sum of weights ≤ 1.0.

    Notes
    -----
    Only a single variant is returned per isotope (the specified isotope in
    *sys.Nucs* with its natural abundance).  To generate variants for *all*
    naturally occurring isotopes of each element, use a more specialized
    function.

    Examples
    --------
    >>> variants = isotopologues(sys)
    >>> for w, s in variants:
    ...     print(f"weight={w:.4f}, Nucs={s.Nucs}")
    """
    if sys.nNuclei == 0:
        return [(1.0, sys)]

    # Get abundance for each nucleus
    abundances = [float(nucabund(nuc)) for nuc in sys.Nucs]

    # Total weight = product of all abundances
    weight = 1.0
    for a in abundances:
        weight *= a

    if weight <= 0:
        return []

    # Return single variant (the exact isotope combination specified)
    import copy as _copy
    new_sys = _copy.copy(sys)
    object.__setattr__(new_sys, 'weight', weight) if hasattr(new_sys, '__dataclass_fields__') else setattr(new_sys, 'weight', weight)
    return [(weight, new_sys)]


def _filter_nuclei(sys: 'SpinSystem', keep: list[int]) -> 'SpinSystem':
    """Build a new SpinSystem keeping only nuclei at indices *keep*."""
    import copy as _copy
    n_e = sys.nElectrons
    n_n = sys.nNuclei

    new_Nucs = [sys.Nucs[i] for i in keep]

    # A tensor: rows correspond to nuclei
    if sys.A is not None:
        A_full = sys.A.reshape(n_n, 3 * n_e)
        new_A = A_full[keep, :] if keep else torch.zeros(0, 3 * n_e, dtype=torch.float64)
    else:
        new_A = None

    # AFrame
    if sys.AFrame is not None and n_n > 0:
        AF_full = sys.AFrame.reshape(n_n, 3 * n_e)
        new_AF = AF_full[keep, :] if keep else torch.zeros(0, 3 * n_e, dtype=torch.float64)
    else:
        new_AF = None

    # Q tensor
    if sys.Q is not None:
        Q_full = sys.Q.reshape(n_n, 3)
        new_Q = Q_full[keep, :] if keep else None
    else:
        new_Q = None

    # QFrame
    if sys.QFrame is not None and n_n > 0:
        QF_full = sys.QFrame.reshape(n_n, 3)
        new_QF = QF_full[keep, :] if keep else None
    else:
        new_QF = None

    # gnscale
    if sys.gnscale is not None and n_n > 0:
        new_gs = sys.gnscale[keep] if keep else torch.zeros(0, dtype=torch.float64)
    else:
        new_gs = None

    return SpinSystem(
        S=list(sys.S),
        g=sys.g.clone(),
        gFrame=sys.gFrame.clone(),
        D=sys.D.clone() if sys.D is not None else None,
        DFrame=sys.DFrame.clone() if sys.DFrame is not None else None,
        B=sys.B,
        BFrame=sys.BFrame,
        Nucs=new_Nucs,
        A=new_A,
        AFrame=new_AF,
        Q=new_Q,
        QFrame=new_QF,
        gnscale=new_gs,
        lw=list(sys.lw) if sys.lw else None,
        lwpp=list(sys.lwpp) if sys.lwpp else None,
        HStrain=sys.HStrain.clone() if sys.HStrain is not None else None,
        gStrain=sys.gStrain.clone() if sys.gStrain is not None else None,
        AStrain=sys.AStrain.clone() if sys.AStrain is not None else None,
        DStrain=sys.DStrain.clone() if sys.DStrain is not None else None,
        weight=sys.weight,
    )


# ---------------------------------------------------------------------------
# spinvec
# ---------------------------------------------------------------------------

def spinvec(sys: 'SpinSystem') -> list:
    """Return all spin quantum numbers (electrons + nuclei) as a flat list.

    Port of EasySpin's ``spinvec.m``.

    Parameters
    ----------
    sys : SpinSystem

    Returns
    -------
    list of float
        Spin quantum numbers [S1, S2, ..., I1, I2, ...] in SpinSystem order.

    Examples
    --------
    >>> from torchspin import SpinSystem
    >>> spinvec(SpinSystem(S=[0.5], Nucs=['14N']))
    [0.5, 1.0]
    """
    return list(sys.Spins.tolist()) if hasattr(sys.Spins, 'tolist') else list(sys.Spins)


# ---------------------------------------------------------------------------
# H0Extra
# ---------------------------------------------------------------------------

_H0EXTRA_HERM_TOL = 1e-10


def _normalize_h0extra(sys: 'SpinSystem') -> Optional[list]:
    """Validate and normalize ``sys.H0Extra`` to a list of
    ``(coeff, op)`` = (0-d float64 tensor, (n, n) complex128 tensor) pairs.

    Idempotent (``dataclasses.replace`` re-runs ``__post_init__``) and graph
    preserving: tensors are converted with ``.to``/``.reshape``, never
    detached, so a coefficient or operator that requires grad stays on the
    autograd graph.  An empty list normalizes to ``None``.
    """
    terms = sys.H0Extra
    if isinstance(terms, tuple) and len(terms) == 2 and not isinstance(terms[0], (tuple, list)):
        raise TypeError("SpinSystem.H0Extra must be a list of (coeff, op) pairs, "
                        "e.g. H0Extra=[(coeff, op)].")
    if not isinstance(terms, (list, tuple)):
        raise TypeError("SpinSystem.H0Extra must be a list of (coeff, op) pairs.")
    if len(terms) == 0:
        return None
    mixtures = [nc for nc in sys.Nucs if is_isotope_mixture(nc)]
    if mixtures:
        raise ValueError(
            f"SpinSystem.H0Extra cannot be combined with isotope mixtures in Nucs "
            f"({mixtures}): the isotopologue expansion changes the nuclei, so one "
            f"operator cannot fit every isotopologue. Give explicit isotopes "
            f"(e.g. '63Cu') and build one component per isotopologue.")
    if any(v > 1 for v in (sys.n or [])):
        raise ValueError("SpinSystem.H0Extra cannot be combined with sets of equivalent "
                         "nuclei (SpinSystem.n > 1): the operator is defined on the full "
                         "Hilbert space; list each nucleus separately in Nucs.")
    dim = sys.nStates
    out = []
    for k, term in enumerate(terms):
        if not isinstance(term, (list, tuple)) or len(term) != 2:
            raise TypeError(f"SpinSystem.H0Extra[{k}] must be a (coeff, op) pair.")
        coeff, op = term
        c = coeff if torch.is_tensor(coeff) else torch.as_tensor(np.asarray(coeff))
        if c.is_complex():
            raise ValueError(f"SpinSystem.H0Extra[{k}]: coeff must be real (a Hermitian "
                             f"term needs a real coefficient), got a complex value.")
        if c.numel() != 1:
            raise ValueError(f"SpinSystem.H0Extra[{k}]: coeff must be a scalar, got "
                             f"shape {tuple(c.shape)}.")
        c = c.to(torch.float64).reshape(())
        O = op if torch.is_tensor(op) else torch.as_tensor(np.asarray(op))
        if O.ndim != 2 or tuple(O.shape) != (dim, dim):
            raise ValueError(
                f"SpinSystem.H0Extra[{k}]: op has shape {tuple(O.shape)}, expected "
                f"({dim}, {dim}) for spins {sys.Spins} (full spin-system basis).")
        O = O.to(torch.complex128)
        Od = O.detach()
        scale = max(1.0, float(Od.abs().max()))
        if float((Od - Od.conj().T).abs().max()) > _H0EXTRA_HERM_TOL * scale:
            raise ValueError(f"SpinSystem.H0Extra[{k}]: op is not Hermitian "
                             f"(tolerance {_H0EXTRA_HERM_TOL:g} relative).")
        out.append((c, O))
    return out


def _reject_h0extra(sys, where: str) -> None:
    """Raise if *sys* carries ``H0Extra`` and *where* cannot include it."""
    if getattr(sys, 'H0Extra', None) is not None:
        raise ValueError(
            f"{where} builds the spin Hamiltonian from the interaction tensors "
            f"without torchspin.ham() and cannot include SpinSystem.H0Extra. Use a "
            f"solver that diagonalizes the full Hamiltonian (e.g. pepper with "
            f"Options(Method='matrix')), or remove H0Extra.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_tensor(x, dtype: torch.dtype) -> torch.Tensor:
    """Convert *x* to a ``torch.Tensor`` with the given dtype.

    Tensors are converted with ``.to`` (no copy, autograd graph preserved), so
    parameters that ``requires_grad`` stay differentiable inside the spin system
    (``pepper_autograd``, ``differentiable_spectrum``).  Lists or tuples whose
    elements are tensors (e.g. ``g=[g_tensor]``, ``A=[[ax, ay, az]]`` with
    tensor entries) are stacked instead of copied through ``torch.tensor``,
    which would detach them (and fails for multi-element tensors).
    """
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    if isinstance(x, (list, tuple)) and _contains_tensor(x):
        return torch.stack([_to_tensor(e, dtype) for e in x]) if len(x) else torch.zeros(0, dtype=dtype)
    return torch.tensor(x, dtype=dtype)


def _contains_tensor(x) -> bool:
    if isinstance(x, torch.Tensor):
        return True
    if isinstance(x, (list, tuple)):
        return any(_contains_tensor(e) for e in x)
    return False
