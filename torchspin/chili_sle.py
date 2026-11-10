"""
Stochastic Liouville equation solver for slow-motion CW EPR — port of
EasySpin's ``chili`` (general Liouvillian method).

Mirrors, function by function, EasySpin's ``chili.m`` and its private helpers
``generateoribasis``, ``jjjsymbol``, ``magint``, ``rbos``, ``hil2liouv``,
``liouvhamiltonian``, ``diffsuperop``, ``chili_xlmk``, ``chili_eqpopvec``,
``chili_lanczos``/``chili_contfracspec`` and the surrounding field-sweep and
output-scaling logic.  The spin Liouville space uses MATLAB's column-major
``vec`` convention throughout so that indices, ``pq`` ordering and the
detection vector coincide with EasySpin's.

Public entry point: :func:`chili_sle` (called by :func:`torchspin.chili.chili`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from torchspin.constants import BMAGN, PLANCK, NMAGN, BOLTZMANN
from torchspin.angmom import wigner3j, wignerd
from torchspin.spinops import sop
from torchspin.rotations import erot
from torchspin.sphgrid import sphgrid
from torchspin.convspec import convspec
from torchspin.dataproc import fieldmod

__all__ = ['ChiliSLEOptions', 'chili_sle']


# =============================================================================
# Options
# =============================================================================

@dataclass
class ChiliSLEOptions:
    """Options for :func:`chili_sle` (EasySpin ``Opt`` fields of chili)."""
    LLMK: list = field(default_factory=lambda: [14, 7, 2, 6])
    jKmin: Optional[int] = None          # -1 or +1; auto from tensor collinearity
    evenK: Optional[bool] = None         # auto: no beta tilts
    highField: bool = False              # pSmin = +1
    pImax: Optional[list] = None
    pImaxall: Optional[int] = None
    MpSymm: Optional[bool] = None        # None = auto (EasySpin's effective default, see chili_sle)
    Solver: str = ''                     # 'L' (Lanczos), '\\' (direct), 'E' (eigen); '' = auto
    FieldSweepMethod: str = ''           # 'approxlin' (default), 'approxinv', 'explicit'
    GridSize: int = 19                   # powder grid (only with an orienting potential)
    GridSymmetry: str = 'Dinfh'
    Rescale: bool = True
    Threshold: float = 1e-6
    Lentz: bool = True
    IncludeNZI: bool = True
    PostConvNucs: list = field(default_factory=list)
    useStartvecSelectionRules: bool = True
    PeqTol: Optional[list] = None
    Verbosity: int = 0


# =============================================================================
# Orientational basis (generateoribasis.m, processbasis in chili.m)
# =============================================================================

@dataclass
class Basis:
    LLMK: list
    evenLmax: int = 0
    oddLmax: int = 0
    Mmax: int = 0
    Kmax: int = 0
    jKmin: Optional[int] = None
    evenK: Optional[bool] = None
    pSmin: int = -1
    pImax: Optional[np.ndarray] = None
    pImaxall: float = 0.0
    MpSymm: bool = False
    L: Optional[np.ndarray] = None
    jK: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None
    M: Optional[np.ndarray] = None
    DirTilt: bool = False


def processbasis(basis: Basis, max_potential_K: Optional[int], I: list, symmetry: dict) -> Basis:
    """EasySpin chili.m ``processbasis``."""
    nNuclei = len(I)
    basis.evenLmax, basis.oddLmax, basis.Mmax, basis.Kmax = [int(v) for v in basis.LLMK]
    if basis.jKmin is None:
        basis.jKmin = +1 if symmetry['tensorsCollinear'] else -1
    if basis.evenK is None:
        basis.evenK = bool(symmetry['nobetatilts'])
    # NOTE: EasySpin's chili.m contains a block (commented out in the shipped
    # version) that would set oddLmax = Kmax = 0 for axial systems; it is not
    # active, so it is not reproduced here.
    I = np.asarray(I, dtype=float)
    if nNuclei == 0:
        basis.pImax = np.zeros(0)
        basis.pImaxall = 0.0
    else:
        if basis.pImax is None:
            basis.pImax = 2 * I
        pI = np.asarray(basis.pImax, dtype=float)
        if pI.size == 1 and nNuclei > 1:
            pI = np.full(nNuclei, float(pI))
        basis.pImax = np.minimum(pI, 2 * I)
        if basis.pImaxall is None:
            basis.pImaxall = float(np.sum(basis.pImax))
        basis.pImaxall = min(float(basis.pImaxall), float(np.sum(2 * I)))
    return basis


def generateoribasis(basis: Basis) -> Basis:
    """EasySpin ``generateoribasis`` — LjKKM basis (L-ordered)."""
    Leven = list(range(0, basis.evenLmax + 1, 2))
    Lodd = list(range(1, basis.oddLmax + 1, 2))
    Llist = sorted(Leven + Lodd)
    Kmax = basis.Kmax
    if basis.evenK:
        Kmax = (Kmax // 2) * 2
    Mmax = basis.Mmax
    deltaK = 2 if basis.evenK else 1
    rows = []
    for L in Llist:
        Lparity = +1 if L % 2 == 0 else -1
        for jK in range(basis.jKmin, 2, 2):
            Kmx = min(L, Kmax)
            for K in range(0, Kmx + 1, deltaK):
                if K == 0 and Lparity != jK:
                    continue
                Mmx = min(L, Mmax)
                for M in range(-Mmx, Mmx + 1):
                    rows.append((L, jK, K, M))
    arr = np.array(rows, dtype=int).reshape(-1, 4)
    basis.L, basis.jK, basis.K, basis.M = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    return basis


# =============================================================================
# 3j-symbol tables (jjjsymbol.m)
# =============================================================================

def jjjsymbol(evenLmax: int, oddLmax: int, compute_rank1: bool):
    """Tables indexed by ``L*L + (L - MK)`` (0-based), MK = L..-L."""
    Ls = sorted(list(range(0, evenLmax + 1, 2)) + list(range(1, oddLmax + 1, 2)))
    nBasis = (max(Ls) + 1) ** 2
    jjj0 = np.zeros((nBasis, nBasis))
    jjj1 = np.zeros((nBasis, nBasis)) if compute_rank1 else None
    jjj2 = np.zeros((nBasis, nBasis))
    for L1 in Ls:
        for MK1 in range(L1, -L1 - 1, -1):
            i1 = L1 * L1 + (L1 - MK1)
            for L2 in Ls:
                for MK2 in range(L2, -L2 - 1, -1):
                    i2 = L2 * L2 + (L2 - MK2)
                    if L1 == L2 or MK1 == -MK2:
                        jjj0[i1, i2] = wigner3j(L1, 0, L2, MK1, 0, MK2)
                    if compute_rank1 and abs(-MK1 - MK2) <= 1 and abs(L1 - L2) <= 1:
                        jjj1[i1, i2] = wigner3j(L1, 1, L2, MK1, -MK2 - MK1, MK2)
                    if abs(-MK1 - MK2) <= 2 and abs(L1 - L2) <= 2:
                        jjj2[i1, i2] = wigner3j(L1, 2, L2, MK1, -MK2 - MK1, MK2)
    return jjj0, jjj1, jjj2


# =============================================================================
# Spin part: ISTO coefficients (istotensor.m, tensor_cart2sph.m, magint.m)
# =============================================================================

def istotensor(a, b):
    """EasySpin ``istotensor``: a, b are 3-vectors of numbers or of matrices."""
    ax, ay, az = a
    bx, by, bz = b
    T0 = -(1 / math.sqrt(3)) * (_mul(ax, bx) + _mul(ay, by) + _mul(az, bz))
    T1 = [
        -0.5 * ((_mul(ax, bz) - _mul(az, bx)) + 1j * (_mul(ay, bz) - _mul(az, by))),
        -(1j / math.sqrt(2)) * (_mul(ay, bx) - _mul(ax, by)),
        -0.5 * ((_mul(ax, bz) - _mul(az, bx)) - 1j * (_mul(ay, bz) - _mul(az, by))),
    ]
    T2 = [
        0.5 * ((_mul(ax, bx) - _mul(ay, by)) + 1j * (_mul(ax, by) + _mul(ay, bx))),
        -0.5 * ((_mul(ax, bz) + _mul(az, bx)) + 1j * (_mul(ay, bz) + _mul(az, by))),
        math.sqrt(2 / 3) * (_mul(az, bz) - 0.5 * (_mul(ax, bx) + _mul(ay, by))),
        0.5 * ((_mul(ax, bz) + _mul(az, bx)) - 1j * (_mul(ay, bz) + _mul(az, by))),
        0.5 * ((_mul(ax, bx) - _mul(ay, by)) - 1j * (_mul(ax, by) + _mul(ay, bx))),
    ]
    return T0, T1, T2


def _mul(a, b):
    """Product of two factors that may be scalars or sparse matrices."""
    if sp.issparse(a) and sp.issparse(b):
        return a @ b
    return a * b


def tensor_cart2sph(Tc: np.ndarray):
    """EasySpin ``tensor_cart2sph``: (T0, T1[3], T2[5]) with q = +k..-k ordering."""
    Tc = np.asarray(Tc, dtype=float)
    T1 = np.zeros(3, dtype=complex)
    T2 = np.zeros(5, dtype=complex)
    if Tc.size == 1:
        T0 = -math.sqrt(1 / 3) * 3 * float(Tc)
    elif Tc.size == 3:
        T0 = -math.sqrt(1 / 3) * (Tc[0] + Tc[1] + Tc[2])
        T2[0] = 0.5 * (Tc[0] - Tc[1]); T2[4] = T2[0]
        T2[2] = math.sqrt(2 / 3) * (Tc[2] - 0.5 * (Tc[0] + Tc[1]))
    else:
        x, y, z = 0, 1, 2
        T0 = -math.sqrt(1 / 3) * (Tc[x, x] + Tc[y, y] + Tc[z, z])
        T1[0] = -0.5 * (Tc[z, x] - Tc[x, z] + 1j * (Tc[z, y] - Tc[y, z]))
        T1[2] = -0.5 * (Tc[z, x] - Tc[x, z] - 1j * (Tc[z, y] - Tc[y, z]))
        T1[1] = -(1j / math.sqrt(2)) * (Tc[x, y] - Tc[y, x])
        T2[0] = 0.5 * ((Tc[x, x] - Tc[y, y]) + 1j * (Tc[x, y] + Tc[y, x]))
        T2[4] = 0.5 * ((Tc[x, x] - Tc[y, y]) - 1j * (Tc[x, y] + Tc[y, x]))
        T2[1] = -0.5 * ((Tc[x, z] + Tc[z, x]) + 1j * (Tc[y, z] + Tc[z, y]))
        T2[3] = +0.5 * ((Tc[x, z] + Tc[z, x]) - 1j * (Tc[y, z] + Tc[z, y]))
        T2[2] = math.sqrt(2 / 3) * (Tc[z, z] - 0.5 * (Tc[x, x] + Tc[y, y]))
    return T0, T1, T2


def _spin_ops(spins):
    """Sparse Sx, Sy, Sz for every spin in the full product space."""
    ops = []
    for i in range(len(spins)):
        ops.append([sp.csr_matrix(sop(spins, [i + 1, c]).numpy()) for c in (1, 2, 3)])
    return ops


def _frame_rotate(T: np.ndarray, angles) -> np.ndarray:
    """R_X2M T R_X2M' with R_X2M = erot(angles).' (frame X -> molecular)."""
    angles = np.asarray(angles, dtype=float)
    if not np.any(angles):
        return T
    R = erot(angles.tolist()).numpy().T
    return R @ T @ R.T


def magint(sys, spin_ops, center_field_mT: float, include_NZI: bool, explicit_field_sweep: bool):
    """EasySpin ``magint`` (general-Liouvillian branch)."""
    nEl = sys.nElectrons
    nNuc = sys.nNuclei
    S = list(sys.S)
    T0, T1, T2, F0, F1, F2, fielddep = [], [], [], [], [], [], []
    B0 = [0.0, 0.0, 1.0] if explicit_field_sweep else [0.0, 0.0, center_field_mT / 1e3]

    def add(a, b, Tc, isfd):
        t0, t1, t2 = istotensor(a, b)
        f0, f1, f2 = tensor_cart2sph(Tc)
        T0.append(t0); T1.append(t1); T2.append(t2)
        F0.append(f0); F1.append(f1); F2.append(f2); fielddep.append(isfd)

    g_all = sys.g.numpy()
    for e in range(nEl):
        g = g_all[3 * e:3 * e + 3, :] if sys.fullg else np.diag(g_all[e, :])
        if not sys.fullg and sys.gFrame is not None:
            g = _frame_rotate(g, sys.gFrame[e].numpy())
        add(B0, spin_ops[e], g * BMAGN / PLANCK, True)
    if nNuc > 0:
        A_all = sys.A.numpy()
        for e in range(nEl):
            for n in range(nNuc):
                if sys.fullA:
                    A_ = A_all[3 * n:3 * n + 3, 3 * e:3 * e + 3] * 1e6
                else:
                    A_ = np.diag(A_all[n, 3 * e:3 * e + 3]) * 1e6
                    if sys.AFrame is not None:
                        A_ = _frame_rotate(A_, sys.AFrame[n, 3 * e:3 * e + 3].numpy())
                add(spin_ops[e], spin_ops[nEl + n], A_, False)
    if any(s > 0.5 for s in S) and sys.D is not None:
        D_all = sys.D.numpy()
        for e in range(nEl):
            if S[e] < 1:
                continue
            D_ = D_all[3 * e:3 * e + 3, :] if sys.fullD else np.diag(D_all[e, :])
            D_ = D_ * 1e6
            if not sys.fullD and sys.DFrame is not None:
                D_ = _frame_rotate(D_, sys.DFrame[e].numpy())
            add(spin_ops[e], spin_ops[e], D_, False)
    if nEl > 1 and sys.ee is not None:
        ee = sys.ee.numpy()
        c = 0
        for e1 in range(nEl):
            for e2 in range(e1 + 1, nEl):
                J_ = ee[3 * c:3 * c + 3, :] if sys.fullee else np.diag(ee[c, :])
                if not sys.fullee and sys.eeFrame is not None:
                    J_ = _frame_rotate(J_, sys.eeFrame[c].numpy())
                add(spin_ops[e1], spin_ops[e2], J_ * 1e6, False)
                c += 1
    if include_NZI:
        gn = np.asarray(sys.gn, dtype=float)
        gnscale = np.asarray(sys.gnscale, dtype=float).reshape(-1)
        for n in range(nNuc):
            add(B0, spin_ops[nEl + n], np.array(-gn[n] * gnscale[n] * NMAGN / PLANCK), True)
    F0 = np.array(F0, dtype=complex); F1 = np.array(F1, dtype=complex).reshape(-1, 3); F2 = np.array(F2, dtype=complex).reshape(-1, 5)
    symmetry = {
        'nobetatilts': bool(np.allclose(F2[:, [1, 3]], 0)),
        'tensorsCollinear': bool(np.all(np.isreal(F2))) and bool(np.allclose(F2.imag, 0)),
        'axialSystem': bool(np.allclose(F2[:, 0], 0)),
    }
    return {'T0': T0, 'T1': T1, 'T2': T2}, {'F0': F0, 'F1': F1, 'F2': F2}, symmetry, np.array(fielddep, dtype=bool)


# =============================================================================
# Rotational basis operators (rbos.m) and Hilbert -> Liouville (hil2liouv.m)
# =============================================================================

def hil2liouv(H):
    """kron(I, H) - kron(H.', I)  (MATLAB column-major vec convention)."""
    n = H.shape[0]
    I = sp.identity(n, format='csr')
    return (sp.kron(I, H, format='csr') - sp.kron(H.T, I, format='csr')).tocsr()


def rbos(T, F, angles, fielddep):
    """EasySpin ``rbos``: (Q0B, Q1B, Q2B, Q0G, Q1G, Q2G) in Liouville space.

    ``B`` = field-dependent terms, ``G`` = field-independent terms.
    """
    D1 = wignerd(1, angles[0], angles[1], angles[2])
    D2 = wignerd(2, angles[0], angles[1], angles[2])
    F0, F1, F2 = F['F0'], F['F1'], F['F2']
    n = T['T0'][0].shape[0]
    Z = sp.csr_matrix((n, n), dtype=complex)
    nTerms = F0.size
    Q0B, Q0G = Z.copy(), Z.copy()
    for t in range(nTerms):
        if fielddep[t]:
            Q0B = Q0B + np.conj(F0[t]) * T['T0'][t]
        else:
            Q0G = Q0G + np.conj(F0[t]) * T['T0'][t]
    have1 = bool(np.any(F1))
    Q1B = Q1G = None
    if have1:
        Q1B = [[Z.copy() for _ in range(3)] for _ in range(3)]
        Q1G = [[Z.copy() for _ in range(3)] for _ in range(3)]
        for mp in range(3):
            for mq in range(3):
                for m in range(3):
                    for t in range(nTerms):
                        term = D1[m, mp] * (-1) * np.conj(F1[t, mq]) * T['T1'][t][m]
                        if fielddep[t]:
                            Q1B[mp][mq] = Q1B[mp][mq] + term
                        else:
                            Q1G[mp][mq] = Q1G[mp][mq] + term
    Q2B = [[Z.copy() for _ in range(5)] for _ in range(5)]
    Q2G = [[Z.copy() for _ in range(5)] for _ in range(5)]
    for mp in range(5):
        for mq in range(5):
            for m in range(5):
                for t in range(nTerms):
                    if F2[t, mq] == 0:
                        continue
                    term = D2[m, mp] * np.conj(F2[t, mq]) * T['T2'][t][m]
                    if fielddep[t]:
                        Q2B[mp][mq] = Q2B[mp][mq] + term
                    else:
                        Q2G[mp][mq] = Q2G[mp][mq] + term
    conv = lambda M: hil2liouv(sp.csr_matrix(M))
    Q0B, Q0G = conv(Q0B), conv(Q0G)
    if have1:
        Q1B = [[conv(Q1B[i][j]) for j in range(3)] for i in range(3)]
        Q1G = [[conv(Q1G[i][j]) for j in range(3)] for i in range(3)]
    Q2B = [[conv(Q2B[i][j]) for j in range(5)] for i in range(5)]
    Q2G = [[conv(Q2G[i][j]) for j in range(5)] for i in range(5)]
    return Q0B, Q1B, Q2B, Q0G, Q1G, Q2G


# =============================================================================
# Liouville Hamiltonian in the LjKKM basis (liouvhamiltonian.m)
# =============================================================================

def _liouvhamiltonian_loop(basis: Basis, Q0, Q1, Q2, jjj0, jjj1, jjj2):
    """Reference port of ``liouvhamiltonian.m`` (Python loops over basis pairs)."""
    L, M, K, jK = basis.L, basis.M, basis.K, basis.jK
    nSpin = Q0.shape[0]
    nOri = L.size
    nTot = nOri * nSpin
    rows, cols, vals = [], [], []

    # --- rank 0 ---
    Q0c = Q0.tocoo()
    r0, c0, v0 = Q0c.row, Q0c.col, Q0c.data
    for ib in range(nOri):
        L_, M_, K_ = int(L[ib]), int(M[ib]), int(K[ib])
        idx0 = ib * nSpin
        jjjM = jjj0[L_ * L_ + L_ + M_, L_ * L_ + L_ - M_]
        jjjK = jjj0[L_ * L_ + L_ + K_, L_ * L_ + L_ - K_]
        f = (-1) ** (K_ - M_) * (2 * L_ + 1) * jjjM * jjjK
        rows.append(idx0 + r0); cols.append(idx0 + c0); vals.append(f * v0)
    H0 = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nTot, nTot))

    # --- rank 1 ---
    H1 = sp.csr_matrix((nTot, nTot), dtype=complex)
    if Q1 is not None:
        rows, cols, vals = [], [], []
        for ib in range(nOri):
            L1, M1, K1, jK1 = int(L[ib]), int(M[ib]), int(K[ib]), int(jK[ib])
            idx1 = ib * nSpin
            K1zero = K1 == 0
            for jb in range(nOri):
                L2 = int(L[jb])
                if abs(L1 - L2) > 1:
                    continue
                M2 = int(M[jb])
                if abs(M1 - M2) > 1:
                    continue
                K2 = int(K[jb])
                if abs(K1 - K2) > 1:
                    continue
                jK2 = int(jK[jb])
                idx2 = jb * nSpin
                NL = math.sqrt((2 * L1 + 1) * (2 * L2 + 1))
                jjjM = jjj1[L1 * L1 + L1 - M1, L2 * L2 + L2 - M2]
                jjjKa = jjj1[L1 * L1 + L1 - K1, L2 * L2 + L2 - K2]
                K2zero = K2 == 0
                idxM = 1 - (M1 - M2)
                idxKa = 1 - (K1 - K2)
                idxKb = 1 - (-K1 + K2)
                pref = (1 / (2 * math.sqrt((1 + K1zero) * (1 + K2zero)))) * np.conj(np.sqrt(complex(jK1))) * np.sqrt(complex(jK2)) * NL * jjjM
                block = pref * jjjKa * ((-1) ** (K1 - M1) * Q1[idxM][idxKa] + jK1 * jK2 * (-1) ** (K2 - M1) * Q1[idxM][idxKb])
                if K1 + K2 <= 1:
                    jjjKb = jjj1[L1 * L1 + L1 - K1, L2 * L2 + L2 + K2]
                    idxKc = 1 - (-K1 - K2)
                    idxKd = 1 - (K1 + K2)
                    block = block + pref * jjjKb * (jK1 * (-1) ** (L2 - M1) * Q1[idxM][idxKc] + jK2 * (-1) ** (L2 + K1 + K2 - M1) * Q1[idxM][idxKd])
                bc = block.tocoo()
                rows.append(bc.row + idx1); cols.append(bc.col + idx2); vals.append(bc.data)
        if rows:
            H1 = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nTot, nTot))
            H1 = H1 + sp.triu(H1, 1).T

    # --- rank 2 ---
    rows, cols, vals = [], [], []
    for ib in range(nOri):
        L1, M1, K1, jK1 = int(L[ib]), int(M[ib]), int(K[ib]), int(jK[ib])
        idx1 = ib * nSpin
        K1zero = K1 == 0
        for jb in range(ib, nOri):
            L2 = int(L[jb])
            if abs(L1 - L2) > 2:
                break
            M2 = int(M[jb])
            if abs(M1 - M2) > 2:
                continue
            K2 = int(K[jb])
            if abs(K1 - K2) > 2:
                continue
            jK2 = int(jK[jb])
            idx2 = jb * nSpin
            NL = math.sqrt((2 * L1 + 1) * (2 * L2 + 1))
            jjjM = jjj2[L1 * L1 + L1 - M1, L2 * L2 + L2 + M2]
            jjjKa = jjj2[L1 * L1 + L1 - K1, L2 * L2 + L2 + K2]
            K2zero = K2 == 0
            idxM = 2 - (M1 - M2)
            idxKa = 2 - (K1 - K2)
            idxKb = 2 - (-K1 + K2)
            pref = (1 / (2 * math.sqrt((1 + K1zero) * (1 + K2zero)))) * np.conj(np.sqrt(complex(jK1))) * np.sqrt(complex(jK2)) * NL * jjjM
            block = pref * jjjKa * ((-1) ** (K1 - M1) * Q2[idxM][idxKa] + jK1 * jK2 * (-1) ** (K2 - M1) * Q2[idxM][idxKb])
            if K1 + K2 <= 2:
                jjjKb = jjj2[L1 * L1 + L1 - K1, L2 * L2 + L2 - K2]
                idxKc = 2 - (-K1 - K2)
                idxKd = 2 - (K1 + K2)
                block = block + pref * jjjKb * (jK1 * (-1) ** (L2 - M1) * Q2[idxM][idxKc] + jK2 * (-1) ** (L2 + K1 + K2 - M1) * Q2[idxM][idxKd])
            bc = block.tocoo()
            rows.append(bc.row + idx1); cols.append(bc.col + idx2); vals.append(bc.data)
    H2 = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nTot, nTot))
    H2 = sp.triu(H2)
    H2 = H2 + sp.triu(H2, 1).T
    return (H0 + H1 + H2).tocsr()


def _sign(n: np.ndarray) -> np.ndarray:
    """(-1)**n for integer arrays."""
    return np.where(np.asarray(n) % 2 == 0, 1.0, -1.0)


def _pairs_within(L, M, K, dmax: int, upper: bool, chunk: int = 512):
    """Index pairs (ib, jb) with |ΔL|, |ΔM|, |ΔK| ≤ dmax; ``upper`` restricts to jb ≥ ib.

    Vectorised in row chunks; the basis is L-ordered, so the loop version's
    ``break`` on |ΔL| > dmax (rank 2) selects exactly the same pairs.
    """
    n = L.size
    ibs, jbs = [], []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        ok = (np.abs(L[s:e, None] - L[None, :]) <= dmax)
        ok &= (np.abs(M[s:e, None] - M[None, :]) <= dmax)
        ok &= (np.abs(K[s:e, None] - K[None, :]) <= dmax)
        if upper:
            ok &= (np.arange(s, e)[:, None] <= np.arange(n)[None, :])
        i, j = np.nonzero(ok)
        ibs.append(i + s); jbs.append(j)
    return np.concatenate(ibs), np.concatenate(jbs)


def _kron_blocks(nOri, nSpin, ib, jb, idxM, idxK, coef, Q):
    """Σ_{m,k} kron(C_mk, Q[m][k]) with C_mk[ib, jb] = Σ coef over the entries tagged (m, k)."""
    nTot = nOri * nSpin
    H = sp.csr_matrix((nTot, nTot), dtype=complex)
    nb = len(Q)
    tag = idxM * nb + idxK
    for m in range(nb):
        for k in range(nb):
            blk = Q[m][k]
            if blk is None or (hasattr(blk, 'nnz') and blk.nnz == 0):
                continue
            sel = tag == m * nb + k
            if not sel.any():
                continue
            C = sp.csr_matrix((coef[sel], (ib[sel], jb[sel])), shape=(nOri, nOri))
            H = H + sp.kron(C, sp.csr_matrix(blk), format='csr')
    return H


def liouvhamiltonian(basis: Basis, Q0, Q1, Q2, jjj0, jjj1, jjj2):
    """EasySpin ``liouvhamiltonian.m`` — vectorised over basis-function pairs.

    Every (ib, jb) pair contributes ``coef × Q[idxM][idxK]`` blocks; the
    Hamiltonian is assembled as Σ kron(C_mk, Q[m][k]) over the 5×5 (rank 2)
    and 3×3 (rank 1) block types, which reproduces
    :func:`_liouvhamiltonian_loop` entry by entry (PERF_PLAN §2.5).
    """
    L = np.asarray(basis.L, dtype=int); M = np.asarray(basis.M, dtype=int)
    K = np.asarray(basis.K, dtype=int); jK = np.asarray(basis.jK, dtype=int)
    nOri = L.size
    if nOri > 1 and np.any(np.diff(L) < 0):
        return _liouvhamiltonian_loop(basis, Q0, Q1, Q2, jjj0, jjj1, jjj2)
    nSpin = Q0.shape[0]
    nTot = nOri * nSpin
    iL = L * L + L
    sq_jK = np.sqrt(jK.astype(complex))

    # --- rank 0 ---
    f0 = _sign(K - M) * (2 * L + 1) * jjj0[iL + M, iL - M] * jjj0[iL + K, iL - K]
    H0 = sp.kron(sp.diags(f0, 0, format='csr'), sp.csr_matrix(Q0), format='csr')

    # --- rank 1 ---
    H1 = sp.csr_matrix((nTot, nTot), dtype=complex)
    if Q1 is not None:
        ib, jb = _pairs_within(L, M, K, 1, upper=False)
        if ib.size:
            L1, M1, K1, jK1 = L[ib], M[ib], K[ib], jK[ib]
            L2, M2, K2, jK2 = L[jb], M[jb], K[jb], jK[jb]
            i1, i2 = iL[ib], iL[jb]
            NL = np.sqrt((2 * L1 + 1) * (2 * L2 + 1))
            jjjM = jjj1[i1 - M1, i2 - M2]
            jjjKa = jjj1[i1 - K1, i2 - K2]
            pref = (1 / (2 * np.sqrt((1 + (K1 == 0)) * (1 + (K2 == 0))))) * np.conj(sq_jK[ib]) * sq_jK[jb] * NL * jjjM
            idxM = 1 - (M1 - M2)
            ibs = [ib, ib]; jbs = [jb, jb]; ms = [idxM, idxM]
            ks = [1 - (K1 - K2), 1 - (-K1 + K2)]
            cs = [pref * jjjKa * _sign(K1 - M1), pref * jjjKa * jK1 * jK2 * _sign(K2 - M1)]
            c = K1 + K2 <= 1
            if c.any():
                jjjKb = jjj1[i1[c] - K1[c], i2[c] + K2[c]]
                ibs += [ib[c], ib[c]]; jbs += [jb[c], jb[c]]; ms += [idxM[c], idxM[c]]
                ks += [1 - (-K1[c] - K2[c]), 1 - (K1[c] + K2[c])]
                cs += [pref[c] * jjjKb * jK1[c] * _sign(L2[c] - M1[c]),
                       pref[c] * jjjKb * jK2[c] * _sign(L2[c] + K1[c] + K2[c] - M1[c])]
            H1 = _kron_blocks(nOri, nSpin, np.concatenate(ibs), np.concatenate(jbs), np.concatenate(ms),
                              np.concatenate(ks), np.concatenate(cs), Q1)
            H1 = H1 + sp.triu(H1, 1).T

    # --- rank 2 ---
    ib, jb = _pairs_within(L, M, K, 2, upper=True)
    H2 = sp.csr_matrix((nTot, nTot), dtype=complex)
    if ib.size:
        L1, M1, K1, jK1 = L[ib], M[ib], K[ib], jK[ib]
        L2, M2, K2, jK2 = L[jb], M[jb], K[jb], jK[jb]
        i1, i2 = iL[ib], iL[jb]
        NL = np.sqrt((2 * L1 + 1) * (2 * L2 + 1))
        jjjM = jjj2[i1 - M1, i2 + M2]
        jjjKa = jjj2[i1 - K1, i2 + K2]
        pref = (1 / (2 * np.sqrt((1 + (K1 == 0)) * (1 + (K2 == 0))))) * np.conj(sq_jK[ib]) * sq_jK[jb] * NL * jjjM
        idxM = 2 - (M1 - M2)
        ibs = [ib, ib]; jbs = [jb, jb]; ms = [idxM, idxM]
        ks = [2 - (K1 - K2), 2 - (-K1 + K2)]
        cs = [pref * jjjKa * _sign(K1 - M1), pref * jjjKa * jK1 * jK2 * _sign(K2 - M1)]
        c = K1 + K2 <= 2
        if c.any():
            jjjKb = jjj2[i1[c] - K1[c], i2[c] - K2[c]]
            ibs += [ib[c], ib[c]]; jbs += [jb[c], jb[c]]; ms += [idxM[c], idxM[c]]
            ks += [2 - (-K1[c] - K2[c]), 2 - (K1[c] + K2[c])]
            cs += [pref[c] * jjjKb * jK1[c] * _sign(L2[c] - M1[c]),
                   pref[c] * jjjKb * jK2[c] * _sign(L2[c] + K1[c] + K2[c] - M1[c])]
        H2 = _kron_blocks(nOri, nSpin, np.concatenate(ibs), np.concatenate(jbs), np.concatenate(ms),
                          np.concatenate(ks), np.concatenate(cs), Q2)
        H2 = sp.triu(H2)
        H2 = H2 + sp.triu(H2, 1).T
    return (H0 + H1 + H2).tocsr()


# =============================================================================
# Diffusion superoperator (diffsuperop.m) and potential expansion (chili_xlmk.m)
# =============================================================================

def chili_xlmk(potential: dict, R):
    """X^L_{MK} coefficients of the potential-dependent part of the diffusion operator."""
    lam = np.asarray(potential['lambda'], dtype=complex).reshape(-1)
    Lp = np.asarray(potential['L'], dtype=int).reshape(-1)
    Mp = np.asarray(potential['M'], dtype=int).reshape(-1)
    Kp = np.asarray(potential['K'], dtype=int).reshape(-1)
    if lam.size == 0 or np.all(lam == 0):
        return []
    keep = lam != 0
    lam, Lp, Mp, Kp = lam[keep], Lp[keep], Mp[keep], Kp[keep]
    idx = (Mp != 0) | (Kp != 0)
    lam = np.concatenate([lam, np.conj(lam[idx]) * (-1.0) ** (Kp[idx] - Mp[idx])])
    Lp = np.concatenate([Lp, Lp[idx]]); Mp = np.concatenate([Mp, -Mp[idx]]); Kp = np.concatenate([Kp, -Kp[idx]])
    nP = lam.size
    maxLpot = int(Lp.max())
    allMzero = bool(np.all(Mp == 0))
    lam_ = [np.zeros((2 * L + 1, 2 * L + 1), dtype=complex) for L in range(maxLpot + 1)]
    for p in range(nP):
        lam_[Lp[p]][Mp[p] + Lp[p], Kp[p] + Lp[p]] = lam[p]

    def lamf(L, M, K):
        return lam_[L][M + L, K + L]

    R = np.asarray(R, dtype=float).reshape(-1)
    if R.size == 1:
        R = np.repeat(R, 3)
    Rz = R[2]; Rp = (R[0] + R[1]) / 2; Rd = (R[0] - R[1]) / 4
    maxLx = 2 * maxLpot
    X = [np.zeros((2 * L + 1, 2 * L + 1), dtype=complex) for L in range(maxLx + 1)]
    cm = lambda L, K: math.sqrt(max(L * (L + 1) - K * (K - 1), 0.0))
    cp = lambda L, K: math.sqrt(max(L * (L + 1) - K * (K + 1), 0.0))
    for L in range(maxLx + 1):
        Mrange = [0] if allMzero else range(-L, L + 1)
        for M in Mrange:
            for K in range(-L, L + 1):
                A = 0.0
                if L <= maxLpot:
                    if Rd != 0:
                        if K + 2 <= L:
                            A += Rd * cm(L, K + 1) * cm(L, K + 2) * lamf(L, M, K + 2)
                        if K - 2 >= -L:
                            A += Rd * cp(L, K - 1) * cp(L, K - 2) * lamf(L, M, K - 2)
                    A += lamf(L, M, K) * (Rp * (L * (L + 1) - K * K) + Rz * K * K)
                B = 0.0
                for p1 in range(nP):
                    lam1 = lam[p1]
                    if lam1 == 0:
                        continue
                    L1, M1, K1 = int(Lp[p1]), int(Mp[p1]), int(Kp[p1])
                    for p2 in range(nP):
                        lam2 = lam[p2]
                        if lam2 == 0:
                            continue
                        L2, M2, K2 = int(Lp[p2]), int(Mp[p2]), int(Kp[p2])
                        if M1 - M + M2 != 0:
                            continue
                        if abs(L1 - L2) > L or L > L1 + L2:
                            continue
                        if M1 == 0 and M2 == 0 and M == 0 and (L1 + L2 + L) % 2:
                            continue
                        B_ = 0.0
                        if Rd != 0:
                            if K1 - K + K2 + 2 == 0 and abs(K1 + 1) <= L1 and abs(K2 + 1) <= L2:
                                cpp = cp(L1, K1) * cp(L2, K2)
                                if cpp != 0:
                                    B_ += Rd * cpp * wigner3j(L1, L, L2, K1 + 1, -K, K2 + 1)
                            if K1 - K + K2 - 2 == 0 and abs(K1 - 1) <= L1 and abs(K2 - 1) <= L2:
                                cmm = cm(L1, K1) * cm(L2, K2)
                                if cmm != 0:
                                    B_ += Rd * cmm * wigner3j(L1, L, L2, K1 - 1, -K, K2 - 1)
                        if K1 - K + K2 == 0:
                            if Rz != 0 and K1 != 0 and K2 != 0:
                                B_ += Rz * K1 * K2 * wigner3j(L1, L, L2, K1, -K, K2)
                            if Rp != 0 and abs(K1 + 1) <= L1 and abs(K2 - 1) <= L2:
                                cpm = cp(L1, K1) * cm(L2, K2)
                                if cpm != 0:
                                    B_ += Rp * cpm * wigner3j(L1, L, L2, K1 + 1, -K, K2 - 1)
                        if B_ == 0:
                            continue
                        B += lam1 * lam2 * wigner3j(L1, L, L2, M1, -M, M2) * B_
                X[L][M + L, K + L] = -0.5 * A - (2 * L + 1) / 4 * (-1.0) ** (K - M) * B
    return X


def diffsuperop(basis: Basis, R, XLMK, potential: Optional[dict]):
    """EasySpin ``diffsuperop`` (K-symmetrized LjKKM basis)."""
    L, M, K, jK = basis.L, basis.M, basis.K, basis.jK
    n = L.size
    R = np.asarray(R, dtype=float).reshape(-1)
    if R.size == 1:
        R = np.repeat(R, 3)
    Rz = R[2]; Rp = (R[0] + R[1]) / 2; Rd = (R[0] - R[1]) / 4
    include_pot = XLMK is not None and len(XLMK) > 0
    if not include_pot and Rd == 0:
        diag = Rp * (L * (L + 1) - K ** 2) + Rz * K ** 2
        return sp.diags(diag.astype(float), 0, format='csr')
    Np = lambda L_, K_: math.sqrt(max((L_ * (L_ + 1) - K_ * (K_ + 1)) * (L_ * (L_ + 1) - (K_ + 1) * (K_ + 2)), 0.0))
    Nm = lambda L_, K_: math.sqrt(max((L_ * (L_ + 1) - K_ * (K_ - 1)) * (L_ * (L_ + 1) - (K_ - 1) * (K_ - 2)), 0.0))
    rows, cols, vals = [], [], []
    for b1 in range(n):
        L1, M1, K1, jK1 = int(L[b1]), int(M[b1]), int(K[b1]), int(jK[b1])
        for b2 in range(b1, n):
            if int(L[b2]) != L1 or int(M[b2]) != M1 or int(jK[b2]) != jK1:
                continue
            K2 = int(K[b2])
            if K1 != K2 and K1 != K2 + 2 and K1 != K2 - 2:
                continue
            val = Rp * L1 * (L1 + 1) + (Rz - Rp) * K2 * K2 if K1 == K2 else 0.0
            val2 = Np(L1, K2) * (K1 == K2 + 2) + Nm(L1, K2) * ((K1 == K2 - 2) + jK1 * (-1) ** (L1 + K1) * (-K1 == K2 - 2))
            val += Rd * val2 / math.sqrt((1 + (K1 == 0)) * (1 + (K2 == 0)))
            if val == 0:
                continue
            rows.append(b1); cols.append(b2); vals.append(val)
            if b1 != b2:
                rows.append(b2); cols.append(b1); vals.append(np.conj(val))
    Gamma = sp.csr_matrix((np.array(vals, dtype=complex), (rows, cols)), shape=(n, n))
    if not include_pot:
        return Gamma
    Lxmax = len(XLMK) - 1
    allMzero = bool(np.all(np.asarray(potential['M']) == 0))
    X = lambda L_, M_, K_: XLMK[L_][M_ + L_, K_ + L_]
    rows, cols, vals = [], [], []
    for b1 in range(n):
        L1, M1, K1, jK1 = int(L[b1]), int(M[b1]), int(K[b1]), int(jK[b1])
        for b2 in range(b1, n):
            L2, M2, K2, jK2 = int(L[b2]), int(M[b2]), int(K[b2]), int(jK[b2])
            if allMzero and M1 != M2:
                continue
            val = 0.0
            for Lx in range(abs(L1 - L2), min(Lxmax, L1 + L2) + 1):
                if abs(M1 - M2) > Lx:
                    continue
                v = 0.0
                if abs(K1 - K2) <= Lx:
                    X1 = X(Lx, M1 - M2, K1 - K2) + X(Lx, M1 - M2, -K1 + K2) * jK1 * jK2 * (-1) ** (Lx + K1 + K2)
                    if X1 != 0:
                        v += X1 * wigner3j(L1, Lx, L2, -K1, K1 - K2, K2)
                if abs(K1 + K2) <= Lx:
                    X2 = X(Lx, M1 - M2, -K1 - K2) * jK1 * (-1) ** (K1 + Lx + L2) + X(Lx, M1 - M2, K1 + K2) * jK2 * (-1) ** (L2 + K2)
                    if X2 != 0:
                        v += X2 * wigner3j(L1, Lx, L2, -K1, K1 + K2, -K2)
                if v == 0:
                    continue
                val += wigner3j(L1, Lx, L2, -M1, M1 - M2, M2) * v
            if val == 0:
                continue
            prefjK = np.conj(np.sqrt(complex(jK1))) * np.sqrt(complex(jK2)) / 2 / math.sqrt((1 + (K1 == 0)) * (1 + (K2 == 0)))
            prefL = math.sqrt((2 * L1 + 1) * (2 * L2 + 1))
            val = (-1) ** (K1 - M1) * prefL * prefjK * val
            rows.append(b1); cols.append(b2); vals.append(val)
            if b1 != b2:
                rows.append(b2); cols.append(b1); vals.append(np.conj(val))
    return (Gamma + sp.csr_matrix((np.array(vals, dtype=complex), (rows, cols)), shape=(n, n))).tocsr()


# =============================================================================
# Equilibrium population vector (chili_eqpopvec.m)
# =============================================================================

def chili_eqpopvec(basis: Basis, potential: Optional[dict], use_selection_rules=True, tol=None):
    from scipy import integrate
    tol = tol or [1e-6, 1e-4, 1e-4]
    thr, abstol, reltol = tol
    L, M, K, jK = basis.L, basis.M, basis.K, basis.jK
    n = L.size
    if potential is None or np.asarray(potential['lambda']).size == 0 or not np.any(np.asarray(potential['lambda'])):
        idx0 = np.where((L == 0) & (M == 0) & (K == 0))[0]
        if idx0.size != 1:
            raise ValueError("Exactly one orientational basis function with L=M=K=0 is allowed.")
        v = np.zeros(n); v[idx0[0]] = 1.0
        return v
    Lp = np.asarray(potential['L'], dtype=int); Mp = np.asarray(potential['M'], dtype=int)
    Kp = np.asarray(potential['K'], dtype=int); lam = np.asarray(potential['lambda'], dtype=complex)
    keep = (lam != 0) & ~((Lp == 0) & (Mp == 0) & (Kp == 0))
    Lp, Mp, Kp, lam = Lp[keep], Mp[keep], Kp[keep], lam[keep]

    def U(a, b, c):
        u = 0.0
        for p in range(lam.size):
            if Kp[p] == 0 and Mp[p] == 0:
                u = u - wignerd([Lp[p], Mp[p], Kp[p]], b) * lam[p].real
            else:
                u = u - 2 * (wignerd([Lp[p], Mp[p], Kp[p]], a, b, c) * lam[p]).real
        return u

    zeroMp, zeroKp = bool(np.all(Mp == 0)), bool(np.all(Kp == 0))
    evenLp = bool(np.all(Lp % 2 == 0)); evenMp = bool(np.all(Mp % 2 == 0)); evenKp = bool(np.all(Kp % 2 == 0))
    q1 = lambda f: integrate.quad(f, 0, math.pi, epsabs=abstol, epsrel=reltol, limit=200)[0]
    q_bc = lambda f: integrate.dblquad(lambda c, b: f(b, c), 0, math.pi, 0, 2 * math.pi, epsabs=abstol, epsrel=reltol)[0]
    q_ab = lambda f: integrate.dblquad(lambda b, a: f(a, b), 0, 2 * math.pi, 0, math.pi, epsabs=abstol, epsrel=reltol)[0]
    q_abc = lambda f: integrate.tplquad(lambda c, b, a: f(a, b, c), 0, 2 * math.pi, 0, math.pi, 0, 2 * math.pi, epsabs=abstol, epsrel=reltol)[0]
    if zeroMp and zeroKp:
        Z = (2 * math.pi) ** 2 * q1(lambda b: math.exp(-U(0, b, 0)) * math.sin(b))
    elif zeroMp:
        Z = 2 * math.pi * q_bc(lambda b, c: math.exp(-U(0, b, c)) * math.sin(b))
    elif zeroKp:
        Z = 2 * math.pi * q_ab(lambda a, b: math.exp(-U(a, b, 0)) * math.sin(b))
    else:
        Z = q_abc(lambda a, b, c: math.exp(-U(a, b, c)) * math.sin(b))
    sqrtZ = math.sqrt(Z)
    out = np.zeros(n, dtype=complex)
    for b_ in range(n):
        L_, M_, K_, jK_ = int(L[b_]), int(M[b_]), int(K[b_]), int(jK[b_])
        if use_selection_rules:
            if zeroMp:
                if M_ != 0 or (evenLp and L_ % 2) or (evenKp and K_ % 2) or jK_ != 1:
                    continue
                if zeroKp:
                    if K_ != 0:
                        continue
                    Int = (2 * math.pi) ** 2 * q1(lambda b: wignerd([L_, 0, 0], b) * math.exp(-U(0, b, 0) / 2) / sqrtZ * math.sin(b))
                else:
                    Int = 2 * math.pi * q_bc(lambda b, c: math.cos(K_ * c) * wignerd([L_, 0, K_], b) * math.exp(-U(0, b, c) / 2) / sqrtZ * math.sin(b))
            elif zeroKp:
                if K_ != 0 or (evenLp and L_ % 2) or (evenMp and M_ % 2):
                    continue
                Int = 2 * math.pi * q_ab(lambda a, b: math.cos(M_ * a) * wignerd([L_, M_, 0], b) * math.exp(-U(a, b, 0) / 2) / sqrtZ * math.sin(b))
            else:
                Int = _complex_tplquad(lambda a, b, c: np.conj(wignerd([L_, M_, K_], a, b, c)) * math.exp(-U(a, b, c) / 2) / sqrtZ * math.sin(b), abstol, reltol)
        else:
            Int = _complex_tplquad(lambda a, b, c: np.conj(wignerd([L_, M_, K_], a, b, c)) * math.exp(-U(a, b, c) / 2) / sqrtZ * math.sin(b), abstol, reltol)
        Int = complex(Int)
        if Int.real != 0 and abs(Int.imag / Int.real) < 1e-5:
            Int = Int.real
        Int = math.sqrt((2 * L_ + 1) / (8 * math.pi ** 2)) * Int
        Int = math.sqrt(2 / (1 + (K_ == 0))) * Int
        if abs(Int) >= thr:
            out[b_] = Int
    return out


def _complex_tplquad(f, abstol, reltol):
    from scipy import integrate
    re = integrate.tplquad(lambda c, b, a: f(a, b, c).real, 0, 2 * math.pi, 0, math.pi, 0, 2 * math.pi, epsabs=abstol, epsrel=reltol)[0]
    im = integrate.tplquad(lambda c, b, a: f(a, b, c).imag, 0, 2 * math.pi, 0, math.pi, 0, 2 * math.pi, epsabs=abstol, epsrel=reltol)[0]
    return re + 1j * im


# =============================================================================
# pq ordering of the spin Liouville basis (pqorder.m)
# =============================================================================

def pqorder(spins):
    spins = [float(s) for s in spins]
    nStates = [int(round(2 * s + 1)) for s in spins]
    N = int(np.prod(nStates))
    n1, n2 = N, 1
    m = np.zeros((N, len(spins)))
    for i, s in enumerate(spins):
        n1 //= nStates[i]
        m_ = np.tile(np.arange(s, -s - 1, -1.0).reshape(1, -1), (n1, n2))
        m[:, i] = m_.flatten(order='F')
        n2 *= nStates[i]
    pq = np.zeros((N * N, 2 * len(spins)))
    for i in range(len(spins)):
        m1 = np.tile(m[:, i:i + 1], (1, N))
        m2 = m1.T
        pq[:, 2 * i] = (m1 - m2).flatten(order='F')
        pq[:, 2 * i + 1] = (m1 + m2).flatten(order='F')
    return pq


# =============================================================================
# Lanczos tridiagonalisation with Lentz continued fraction (chili_lanczos.m)
# =============================================================================

def chili_lanczos(A, b, z, threshold: float, max_steps: Optional[int] = None):
    """Spectrum s(z) = <b|(A + z)^-1|b> for a complex-symmetric A (pseudo-norm Lanczos)."""
    N = b.size
    if max_steps is None:
        max_steps = N
    interval = min(10, int(math.ceil(N / 20)))
    tiny = 1e-30
    spec = np.full(z.shape, tiny, dtype=complex)
    C = spec.copy(); D = np.zeros_like(spec)
    q = b / np.sqrt(b @ b)
    bq = 0.0
    alpha = np.zeros(max_steps, dtype=complex); beta = np.zeros(max_steps, dtype=complex)
    converged = False
    k_done = 0
    for k in range(max_steps):
        y = A @ q
        alpha[k] = q @ y
        y = y - alpha[k] * q - bq
        beta[k] = np.sqrt(y @ y)
        k_done = k + 1
        bb = alpha[k] + z
        a = 1.0 if k == 0 else -beta[k - 1] ** 2
        D = 1.0 / (bb + a * D)
        C = bb + a / C
        Delta = C * D
        spec = spec * Delta
        if (k + 1) % interval == 0:
            change = np.max(np.abs(Delta - 1))
            converged = change < threshold
            if converged:
                break
        if beta[k] == 0:
            break
        bq = beta[k] * q
        q = y / beta[k]
    return spec, converged, k_done


# =============================================================================
# Main
# =============================================================================

def chili_sle(sys, exp, opt: Optional[ChiliSLEOptions] = None):
    """Slow-motion CW EPR spectrum by the stochastic Liouville equation (EasySpin chili)."""
    if opt is None:
        opt = ChiliSLEOptions()
    verbose = opt.Verbosity >= 1

    # ---- system checks ------------------------------------------------------
    from torchspin.spinsystem import _reject_h0extra
    _reject_h0extra(sys, 'chili')
    if getattr(sys, 'n', None) is not None and any(v > 1 for v in sys.n) and not opt.PostConvNucs:
        raise ValueError("chili cannot handle sets of equivalent nuclei (Sys.n > 1) unless treated by post-convolution (Opt.PostConvNucs).")
    if getattr(sys, 'nn', None) is not None and bool(torch.any(torch.as_tensor(sys.nn) != 0)):
        raise ValueError("chili does not support nuclear-nuclear couplings (Sys.nn).")
    for name in ('HStrain', 'gStrain', 'AStrain', 'DStrain'):
        v = getattr(sys, name, None)
        if v is not None and bool(torch.any(torch.as_tensor(v) != 0)):
            raise ValueError("chili does not support strains (HStrain, gStrain, AStrain, DStrain).")

    # post-convolution nuclei: remove them from the SLE system
    full_sys = sys
    pc_nucs = sorted(set(int(k) for k in opt.PostConvNucs)) if opt.PostConvNucs else []
    if pc_nucs:
        from torchspin.spinsystem import SpinSystem
        keep_nuc = [i for i in range(sys.nNuclei) if (i + 1) not in pc_nucs]
        kw = {'S': list(sys.S), 'g': sys.g.tolist(), 'gFrame': sys.gFrame.tolist(), 'lw': list(sys.get_lw())}
        if keep_nuc:
            kw['Nucs'] = ','.join(sys.Nucs[i] for i in keep_nuc)
            kw['A'] = sys.A[keep_nuc].tolist(); kw['AFrame'] = sys.AFrame[keep_nuc].tolist()
        for name in ('D', 'DFrame', 'ee', 'eeFrame', 'tcorr', 'logtcorr', 'Diff', 'logDiff', 'Potential'):
            v = getattr(sys, name, None)
            if v is not None:
                kw[name] = v.tolist() if isinstance(v, torch.Tensor) else v
        sys = SpinSystem(**kw)

    lw = list(sys.get_lw())
    convolution_broadening = lw[0] > 0

    # ---- potential ----------------------------------------------------------
    pot = getattr(sys, 'Potential', None)
    potential = None
    if pot is not None:
        P = np.asarray(pot.numpy() if isinstance(pot, torch.Tensor) else pot, dtype=complex).reshape(-1, 4)
        rmv = (P[:, 3] == 0) | ((P[:, 0] == 0) & (P[:, 1] == 0) & (P[:, 2] == 0))
        P = P[~rmv]
        if P.shape[0] > 0:
            potential = {'L': P[:, 0].real.astype(int), 'M': P[:, 1].real.astype(int), 'K': P[:, 2].real.astype(int), 'lambda': P[:, 3]}
            if np.any(potential['K'] < 0) or np.any(potential['M'][potential['K'] == 0] < 0):
                raise ValueError("Potential terms must have K >= 0 (and M >= 0 for K = 0).")
    use_potential = potential is not None

    # ---- experiment ---------------------------------------------------------
    field_sweep = not exp.is_freq_swept
    nPoints = int(exp.nPoints)
    if field_sweep:
        if exp.mwFreq is None or exp.Range is None:
            raise ValueError("chili: field sweep needs exp.mwFreq and exp.Range/CenterSweep.")
        center_field = 0.5 * (exp.Range[0] + exp.Range[1])
        x_axis = np.linspace(exp.Range[0], exp.Range[1], nPoints)
        harmonic = int(exp.Harmonic) if exp.Harmonic is not None else 1
    else:
        center_field = float(exp.Field)
        x_axis = np.linspace(exp.mwRange[0], exp.mwRange[1], nPoints)
        harmonic = int(exp.Harmonic) if exp.Harmonic is not None else 0
    mod_amp = float(getattr(exp, 'ModAmp', 0.0) or 0.0)
    if mod_amp > 0:
        if not field_sweep:
            raise ValueError("Exp.ModAmp cannot be used with frequency sweeps.")
        if harmonic < 1:
            raise ValueError("With field modulation (Exp.ModAmp), Exp.Harmonic=0 does not work.")
        mod_harmonic, conv_harmonic, deriv_harmonic = harmonic, 0, 0
    else:
        mod_harmonic = 0
        conv_harmonic, deriv_harmonic = (harmonic, 0) if convolution_broadening else (0, harmonic)
    mw_phase = float(getattr(exp, 'mwPhase', 0.0) or 0.0)
    sample_frame = getattr(exp, 'SampleFrame', None)
    integrate_over_grid = (sample_frame is None) and use_potential

    # ---- dynamics (processdynamics) ------------------------------------------
    if getattr(sys, 'Diff', None) is not None:
        R = np.asarray(sys.Diff, dtype=float).reshape(-1)
    elif getattr(sys, 'logDiff', None) is not None:
        R = 10.0 ** np.asarray(sys.logDiff, dtype=float).reshape(-1)
    elif getattr(sys, 'tcorr', None) is not None:
        R = 1.0 / 6.0 / np.asarray(sys.tcorr, dtype=float).reshape(-1)
    elif getattr(sys, 'logtcorr', None) is not None:
        R = 1.0 / 6.0 / 10.0 ** np.asarray(sys.logtcorr, dtype=float).reshape(-1)
    else:
        raise ValueError("You must specify a rotational correlation time or a diffusion tensor (tcorr, logtcorr, Diff or logDiff).")
    if R.size == 1:
        R = np.repeat(R, 3)
    elif R.size == 2:
        R = np.array([R[0], R[0], R[1]])
    if lw[1] > 0:
        lorentz_fwhm = lw[1] * 28 * 1e6 if field_sweep else lw[1] * 1e6   # Hz (EasySpin: g ≈ 2.0006 for field sweeps)
        T2 = 1.0 / lorentz_fwhm / math.pi
    else:
        T2 = math.inf

    # ---- method / solver ------------------------------------------------------
    fsm = opt.FieldSweepMethod or 'approxlin'
    explicit = field_sweep and fsm == 'explicit'
    solver = opt.Solver or ('\\' if explicit else 'L')
    if explicit and solver != '\\':
        raise ValueError("For an explicit field sweep, use Opt.Solver='\\\\'.")

    # ---- spin part -------------------------------------------------------------
    spins = list(sys.Spins)
    spin_ops = _spin_ops(spins)
    nStates = int(np.prod([int(round(2 * s + 1)) for s in spins]))
    Sdet = sp.csr_matrix((nStates, nStates), dtype=complex)
    for e in range(sys.nElectrons):
        Sdet = Sdet + spin_ops[e][0]
    T, F, symmetry, fielddep = magint(sys, spin_ops, center_field, opt.IncludeNZI, explicit)
    if np.all(F['F1'] == 0) and np.all(F['F2'] == 0):
        raise ValueError("This is an isotropic spin system. chili cannot calculate a slow-motion spectrum.")

    # EasySpin selects its compiled 'fast' Liouvillian builder for one S=1/2 electron
    # with <= 2 nuclei and no potential; that builder uses the M-pS-pI (Meirovitch)
    # symmetry basis, which the general method only applies with Opt.MpSymm=true.
    # Reproducing EasySpin's default output therefore means MpSymm on for those
    # systems (verified in MATLAB: general+MpSymm vs fast, cosine 0.99999 at
    # tcorr = 100 ns where the plain general basis gives 0.937).
    mp_symm = opt.MpSymm
    if mp_symm is None:
        # Empirical rule verified against EasySpin (MATLAB, 2026-09-02): with the
        # rhombic basis (Mmax > 0 and Kmax > 0) fast == general+MpSymm at half
        # amplitude for every stored case; with an axial basis (Mmax = 0 or
        # Kmax = 0) fast == plain general.  The fast builder itself is not ported.
        Mmax, Kmax = int(opt.LLMK[2]), int(opt.LLMK[3])
        mp_symm = (sys.nElectrons == 1 and float(sys.S[0]) == 0.5 and sys.nNuclei <= 2 and not use_potential
                   and Mmax > 0 and Kmax > 0)
    basis = Basis(LLMK=list(opt.LLMK), jKmin=opt.jKmin, evenK=opt.evenK, pSmin=+1 if opt.highField else -1,
                  pImax=None if opt.pImax is None else np.asarray(opt.pImax, dtype=float),
                  pImaxall=opt.pImaxall, MpSymm=mp_symm)
    basis = processbasis(basis, int(np.max(potential['K'])) if use_potential else None, list(sys.I), symmetry)

    if sys.fullg:
        g_np = sys.g.numpy()
        gavg = float(np.mean([np.mean(np.linalg.eigvals(g_np[3 * e:3 * e + 3, :]).real) for e in range(sys.nElectrons)]))
    else:
        gavg = float(np.mean(sys.g.numpy()))

    # ---- field / frequency axis, omega --------------------------------------
    if field_sweep:
        B_ = x_axis
        if fsm == 'explicit':
            B0 = B_; nu = np.full(nPoints, float(exp.mwFreq))
        elif fsm == 'approxinv':
            B0 = np.array([center_field]); nu = float(exp.mwFreq) * center_field / B_
        else:
            B0 = np.array([center_field]); nu = float(exp.mwFreq) - (B_ - center_field) * gavg * BMAGN / PLANCK * 1e-3 / 1e9
    else:
        B0 = np.array([center_field]); nu = x_axis
    B0_T = B0 / 1e3
    omega0 = 2 * math.pi * nu * 1e9 + 1j / T2

    # ---- orientations -----------------------------------------------------------
    if integrate_over_grid:
        if int(opt.GridSize) == 1:
            phi = np.array([0.0]); theta = np.array([0.0]); weights = np.array([4 * math.pi])
        else:
            p_, t_, w_, _ = sphgrid(opt.GridSymmetry, int(opt.GridSize))
            phi, theta, weights = p_.numpy(), t_.numpy(), w_.numpy()
    else:
        if sample_frame is not None:
            sf = np.asarray(sample_frame, dtype=float).reshape(-1)
            phi = np.array([-sf[2]]); theta = np.array([-sf[1]])
        else:
            phi = np.array([0.0]); theta = np.array([0.0])
        weights = np.array([4 * math.pi])
    weights = 4 * math.pi * weights / weights.sum()
    basis.DirTilt = bool(np.any(theta != 0))

    # ---- orientational basis and spin-basis pruning -----------------------------
    basis = generateoribasis(basis)
    nOri = basis.L.size
    nSpin = nStates ** 2
    pq = pqorder(spins)
    keep = np.ones(nSpin, dtype=bool)
    for e in range(sys.nElectrons):
        keep &= pq[:, 2 * e] >= basis.pSmin
    pIsum = np.zeros(nSpin)
    for n in range(sys.nNuclei):
        pI = pq[:, 2 * sys.nElectrons + 2 * n]
        pIsum += pI
        keep &= np.abs(pI) <= basis.pImax[n]
    keep &= np.abs(pIsum) <= basis.pImaxall
    keep_full = np.tile(keep, nOri)
    if mp_symm:
        psum = pq[:, 0::2].sum(axis=1)
        keep_full &= (psum[None, :] - basis.M[:, None] == 1).reshape(-1)
    if verbose:
        print(f"chili: basis {nOri} orientational × {nSpin} spin functions, {int(keep_full.sum())} kept")

    compute_rank1 = bool(np.any(F['F1']))
    jjj0, jjj1, jjj2 = jjjsymbol(basis.evenLmax, basis.oddLmax, compute_rank1)
    XLMK = chili_xlmk(potential, R) if use_potential else []
    Gamma = diffsuperop(basis, R, XLMK, potential)
    Gamma = sp.kron(Gamma, sp.identity(nSpin, format='csr'), format='csr')
    Gamma = Gamma[keep_full][:, keep_full]

    # ---- starting vector --------------------------------------------------------
    sqrtPeq = chili_eqpopvec(basis, potential, opt.useStartvecSelectionRules, opt.PeqTol)
    sdet_vec = Sdet.toarray().flatten(order='F')
    sdet_vec = sdet_vec / np.linalg.norm(sdet_vec)
    start = np.kron(sqrtPeq, sdet_vec)[keep_full]
    norm_peq = float(np.linalg.norm(sqrtPeq) ** 2)
    if use_potential and norm_peq < 0.99 and verbose:
        print(f"chili: norm of equilibrium population vector {norm_peq:.3g} (<1: basis may be too small)")
    start = start / np.linalg.norm(start)
    basis_size = start.size

    # ---- orientation loop --------------------------------------------------------
    spec = np.zeros(nPoints, dtype=complex)
    for iOri in range(phi.size):
        Q0B, Q1B, Q2B, Q0G, Q1G, Q2G = rbos(T, F, [phi[iOri], theta[iOri], 0.0], fielddep)
        if explicit:
            HB = liouvhamiltonian(basis, Q0B, Q1B, Q2B, jjj0, jjj1, jjj2)[keep_full][:, keep_full]
            HG = liouvhamiltonian(basis, Q0G, Q1G, Q2G, jjj0, jjj1, jjj2)[keep_full][:, keep_full]
        else:
            Q0 = Q0B + Q0G
            Q1 = None if Q1B is None else [[Q1B[i][j] + Q1G[i][j] for j in range(3)] for i in range(3)]
            Q2 = [[Q2B[i][j] + Q2G[i][j] for j in range(5)] for i in range(5)]
            H = liouvhamiltonian(basis, Q0, Q1, Q2, jjj0, jjj1, jjj2)[keep_full][:, keep_full]
        thisspec = np.zeros(nPoints, dtype=complex)
        for iB in range(B0_T.size):
            if explicit:
                H = B0_T[iB] * HB + HG
            Lm = (2j * math.pi * H + Gamma).tocsr()
            if Lm.shape[0] != basis_size:
                raise RuntimeError("Liouvillian size inconsistent with basis size.")
            if opt.Rescale:
                scale = float(np.abs(Lm.data).max()) if Lm.nnz else 1.0
                Lm = Lm / scale
                omega = omega0 / scale
            else:
                scale = 1.0
                omega = omega0
            if solver == 'L':
                s_, converged, steps = chili_lanczos(Lm, start, -1j * omega, opt.Threshold)
                if not converged:
                    if verbose:
                        print(f"chili: Lanczos did not converge (basis {basis_size}, LLMK {opt.LLMK}); increase the basis.")
                    s_ = np.ones(nPoints, dtype=complex)
                elif verbose:
                    print(f"chili: Lanczos converged after {steps} steps (basis {basis_size})")
                thisspec_i = s_
            elif solver == '\\':
                I = sp.identity(basis_size, format='csc')
                Lc = Lm.tocsc()
                if explicit:
                    u = spla.spsolve(Lc - 1j * omega[iB] * I, start)
                    thisspec[iB] = np.vdot(start, u)
                    continue
                thisspec_i = np.empty(nPoints, dtype=complex)
                for k in range(nPoints):
                    u = spla.spsolve(Lc - 1j * omega[k] * I, start)
                    thisspec_i[k] = np.vdot(start, u)
            elif solver == 'E':
                Ld = Lm.toarray()
                Lam, U = np.linalg.eig(Ld)
                amp = (np.conj(start) @ U) * np.linalg.solve(U, start)
                thisspec_i = np.sum(amp[:, None] / (Lam[:, None] - 1j * omega[None, :]), axis=0)
            else:
                raise ValueError(f"Unknown Opt.Solver '{solver}'. Use 'L', '\\\\' or 'E'.")
            if not explicit:
                thisspec = thisspec_i
        spec = spec + thisspec * weights[iOri]

    # ---- output scaling (chili.m) -------------------------------------------------
    spec = spec / scale * 1e10 / 2
    if mp_symm and opt.MpSymm is None:
        # EasySpin's fast builder (its default for these systems) yields exactly half
        # the amplitude of general+MpSymm (MATLAB: ratio 2.0000 for 1 nucleus, 2.0024
        # for 2); the automatic selection reproduces the default (fast) output.
        spec = spec / 2
    if opt.highField:
        spec = spec / 2
    if not field_sweep:
        spec = spec * 1e3
        mw_phase = -mw_phase
    spec = np.real(np.exp(1j * mw_phase) * spec)

    # ---- post-convolution nuclei (garlic) ----------------------------------------
    if pc_nucs:
        from torchspin.garlic import garlic
        from torchspin.spinsystem import SpinSystem
        from torchspin.experiment import Experiment
        idx = [k - 1 for k in pc_nucs]
        A_full = full_sys.A.numpy()
        pc_kw = {'S': [0.5], 'g': float(np.mean(full_sys.g.numpy())),
                 'Nucs': ','.join(full_sys.Nucs[i] for i in idx),
                 'A': [float(np.mean(A_full[i, :3])) for i in idx]}
        if getattr(full_sys, 'n', None) is not None:
            pc_kw['n'] = [int(full_sys.n[i]) for i in idx]
        dx = (x_axis[-1] - x_axis[0]) / (nPoints - 1) * (1.0 if field_sweep else 1e3)
        pc_kw['lw'] = [dx / 12, 0.0]
        if field_sweep:
            pc_exp = Experiment(mwFreq=float(np.mean(x_axis)) * pc_kw['g'] * BMAGN / PLANCK * 1e-3 / 1e9,
                                Range=[float(x_axis[0]), float(x_axis[-1])], Harmonic=0, nPoints=nPoints)
        else:
            pc_exp = Experiment(Field=float(exp.Field), mwRange=[float(x_axis[0]), float(x_axis[-1])], Harmonic=0, nPoints=nPoints)
        _, spc_pc = garlic(SpinSystem(**pc_kw), pc_exp)
        spc_pc = spc_pc.numpy()
        spc_pc = spc_pc / spc_pc.sum()
        spec = np.convolve(spec, spc_pc, mode='same')

    # ---- temperature ------------------------------------------------------------
    if exp.Temperature is not None and np.isfinite(exp.Temperature):
        dE = PLANCK * float(exp.mwFreq) * 1e9 if field_sweep else BMAGN * gavg * float(exp.Field) * 1e-3
        e_ = math.exp(-dE / BOLTZMANN / float(exp.Temperature))
        spec = spec * (1 - e_) / (1 + e_)

    # ---- Gaussian convolution, derivative, modulation ------------------------------
    dx_axis = x_axis[1] - x_axis[0]
    fwhmG = lw[0] if field_sweep else lw[0] / 1e3
    spec_t = torch.tensor(spec, dtype=torch.float64)
    if fwhmG > 0 and convolution_broadening:
        spec_t = convspec(spec_t, dx_axis, fwhm_g=fwhmG, fwhm_l=0.0, deriv=conv_harmonic)
    for _ in range(deriv_harmonic):
        d = torch.zeros_like(spec_t)
        d[1:-1] = (spec_t[2:] - spec_t[:-2]) / (2 * dx_axis); d[0] = (spec_t[1] - spec_t[0]) / dx_axis; d[-1] = (spec_t[-1] - spec_t[-2]) / dx_axis
        spec_t = d
    if mod_harmonic > 0:
        spec_t = torch.tensor(fieldmod(x_axis, spec_t.numpy(), mod_amp, mod_harmonic), dtype=torch.float64)
    return torch.tensor(x_axis, dtype=torch.float64), spec_t
