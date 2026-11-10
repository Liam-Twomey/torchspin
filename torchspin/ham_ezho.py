"""Higher-order Zeeman interaction Hamiltonian.

Port of EasySpin's ``ham_ezho.m``.

Uses the formalism from:
  McGavin, Tennant, and Weil
  J. Magn. Reson. 87, 92-109 (1990)

The spin Hamiltonian is expanded in powers of the magnetic field::

    H = sum_{lB,lS,l}  G_{lB,lS,l}  (spherical tensor coupling)

where lB is the order in B, lS is the spin-operator rank, and l is the
coupling rank (|lB-lS| <= l <= lB+lS, l even).

Parameters are stored in ``SpinSystem.Ham`` as a dict mapping
``'{lB}{lS}{l}'`` strings to ``(nElectrons, 2*l+1)`` arrays.

Example
-------
>>> from torchspin import SpinSystem
>>> from torchspin.ham_ezho import ham_ezho
>>> sys = SpinSystem(S=[0.5], Ham={'112': [[1.0, 0, 0, 0, 0.5]]})
>>> H = ham_ezho(sys, B0=[0.0, 0.0, 340.0])
>>> H.shape
torch.Size([2, 2])
"""
from __future__ import annotations

import math
from itertools import permutations
from typing import Optional

import numpy as np
import torch
from scipy.special import lpmv

from torchspin.angmom import wigner3j
from torchspin.rotutils import vec2ang
from torchspin.stev import stev
from torchspin.spinsystem import SpinSystem


# ---------------------------------------------------------------------------
# Conversion tables (from McGavin-Tennant-Weil / EasySpin ham_ezho.m)
# ---------------------------------------------------------------------------

# alphapm1(lB+1) = 1/sqrt(c_{lB})  — spherical harmonic normalization
_ALPHAPM1_SQ = [1, 1, 1.5, 2.5, 35/8, 63/8, 231/16, 429/16, 6435/128]
_ALPHAPM1 = [1.0 / math.sqrt(a) for a in _ALPHAPM1_SQ]

# Alm[lS][|mS|] — conversion between Stevens operators and normalized
# spherical tensor operators.  Index by (lS, abs_mS).
_ALM = {
    1: [1.0, 1.0],
    2: [math.sqrt(6), 1/math.sqrt(2), math.sqrt(2)],
    3: [math.sqrt(10), 2*math.sqrt(5/3), math.sqrt(2/3), 2.0],
    4: [2*math.sqrt(70), math.sqrt(7), math.sqrt(14), 1.0, 2*math.sqrt(2)],
    5: [6*math.sqrt(14), 2*math.sqrt(42/5), math.sqrt(6/5),
        12/math.sqrt(5), 2*math.sqrt(2/5), 4.0],
    6: [4*math.sqrt(231), 2*math.sqrt(11), 4*math.sqrt(22/5),
        2*math.sqrt(22/5), 4*math.sqrt(11/3), 2*math.sqrt(2/3),
        4*math.sqrt(2)],
    7: [4*math.sqrt(429), 8*math.sqrt(429/7), 4*math.sqrt(286/7),
        8*math.sqrt(143/7), 4*math.sqrt(13/7), 8*math.sqrt(13/7),
        4*math.sqrt(2/7), 8.0],
    8: [24*math.sqrt(1430), 2*math.sqrt(1430), 4*math.sqrt(143/7),
        2*math.sqrt(78/7), 4*math.sqrt(130/7), 2*math.sqrt(10/7),
        4*math.sqrt(15), 2*math.sqrt(2), 8*math.sqrt(2)],
}


# ---------------------------------------------------------------------------
# Helper: Stevens operator → normalized spherical tensor operator
# ---------------------------------------------------------------------------

def _stev_to_sph(spins, lS, mS, iSpin, dtype, device):
    """Convert Stevens operator to normalized spherical tensor T_{lS,mS}."""
    if lS == 0:
        # T_{0,0} = identity (normalized)
        n = math.prod(int(round(2 * s + 1)) for s in spins)
        return torch.eye(n, dtype=dtype, device=device)

    alm = _ALM[lS]
    abs_mS = abs(mS)

    if mS == 0:
        return stev(spins, lS, 0, iSpin=iSpin, dtype=dtype) / alm[0]
    elif mS > 0:
        O_pos = stev(spins, lS, mS, iSpin=iSpin, dtype=dtype)
        O_neg = stev(spins, lS, -mS, iSpin=iSpin, dtype=dtype)
        return (-1)**mS / (alm[abs_mS] * math.sqrt(2)) * (O_pos + 1j * O_neg)
    else:
        O_pos = stev(spins, lS, abs_mS, iSpin=iSpin, dtype=dtype)
        O_neg = stev(spins, lS, mS, iSpin=iSpin, dtype=dtype)
        return 1.0 / (alm[abs_mS] * math.sqrt(2)) * (O_pos - 1j * O_neg)


# ---------------------------------------------------------------------------
# Core: compute H for a given field and selected lB orders
# ---------------------------------------------------------------------------

def _ham_ezho_at_field(sys, B, eSpins, lB_list, dtype, device):
    """Evaluate the higher-order Zeeman Hamiltonian at a specific field."""
    spins = sys.Spins
    n_states = sys.nStates
    n_electrons = sys.nElectrons

    H = torch.zeros(n_states, n_states, dtype=dtype, device=device)

    if sys.Ham is None or len(sys.Ham) == 0:
        return H

    if eSpins is None:
        eSpins = list(range(1, n_electrons + 1))

    # Convert B to spherical coordinates
    B_arr = np.array(B, dtype=np.float64)
    rB = float(np.linalg.norm(B_arr))

    if rB > 0:
        phi, theta = vec2ang(B_arr)
        cos_theta = math.cos(float(theta))
    else:
        phi = 0.0
        cos_theta = 1.0

    ham_dict = sys.Ham

    for iSpin in eSpins:
        i0 = iSpin - 1  # 0-based for parameter indexing

        for lB in lB_list:
            # Collect lS values for this lB from Ham dict keys
            lS_values = set()
            for key in ham_dict:
                if len(key) >= 2 and key[0] == str(lB):
                    try:
                        lS_val = int(key[1])
                        lS_values.add(lS_val)
                    except ValueError:
                        pass

            for lS in sorted(lS_values):
                # Time-reversal: lB + lS must be even
                if (lB + lS) % 2 != 0:
                    continue

                strlBlS = f'{lB}{lS}'

                # Precompute field tensor components T_{lB,mB}
                TlBmB = {}
                for mB in range(lB, -lB - 1, -1):
                    abs_mB = abs(mB)
                    # Associated Legendre (MATLAB convention, no CS phase)
                    # scipy lpmv *includes* CS phase → multiply by (-1)^m to remove
                    P_lm = float((-1)**abs_mB * lpmv(abs_mB, lB, cos_theta))

                    fac_ratio = math.sqrt(
                        math.factorial(lB - abs_mB) /
                        math.factorial(lB + abs_mB)
                    )

                    # Sign: 1 for mB <= 0, (-1)^mB for mB > 0
                    sign = (-1.0)**mB if mB > 0 else 1.0

                    pre = fac_ratio * sign
                    TlBmB[mB] = _ALPHAPM1[lB] * pre * P_lm * np.exp(1j * mB * float(phi))

                Glb = rB**lB / math.sqrt(2) if lB > 0 else 1.0 / math.sqrt(2)

                min_l = abs(lB - lS)
                max_l = lB + lS

                # Find all coupling ranks l for this (lB, lS) pair
                for key in ham_dict:
                    if not key.startswith(strlBlS):
                        continue
                    suffix = key[len(strlBlS):]
                    if not suffix:
                        continue
                    try:
                        l_val = int(suffix)
                    except ValueError:
                        continue

                    if l_val < min_l or l_val > max_l or l_val % 2 != 0:
                        continue

                    params = ham_dict[key]
                    if params.ndim == 1:
                        params = params.unsqueeze(0)
                    if i0 >= params.shape[0]:
                        continue

                    Z_row = params[i0].detach().cpu().numpy()
                    if not np.any(Z_row):
                        continue

                    l = l_val

                    # Convert Z (Stevens convention) to spherical tensor coefficients a
                    # Z has 2l+1 elements for q = l, l-1, ..., -l
                    a = np.zeros(2 * l + 1, dtype=np.complex128)

                    for m_pos in range(l, 0, -1):
                        lp = l - m_pos   # index for q = +m_pos
                        lm = l + m_pos   # index for q = -m_pos
                        a[lp] = Glb * (-1)**m_pos * (Z_row[lp] - 1j * Z_row[lm])
                        a[lm] = Glb * (Z_row[lp] + 1j * Z_row[lm])

                    # m = 0
                    a[l] = Glb * math.sqrt(2) * Z_row[l]

                    # Sum contributions
                    for iq in range(2 * l + 1):
                        if abs(a[iq]) < 1e-15:
                            continue
                        m = l - iq

                        pre_m = (-1)**m * math.sqrt(2 * l + 1)

                        for mB in range(lB, -lB - 1, -1):
                            if abs(TlBmB[mB]) < 1e-10:
                                continue

                            mS = m - mB
                            if abs(mS) > lS:
                                continue

                            w3j = wigner3j(lB, lS, l, mB, mS, -m)
                            if abs(w3j) < 1e-15:
                                continue

                            TlSmS = _stev_to_sph(
                                spins, lS, mS, iSpin, dtype, device
                            )

                            coeff = complex(a[iq] * pre_m * w3j * TlBmB[mB])
                            H = H + coeff * TlSmS

    return H


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ham_ezho(
    sys: SpinSystem,
    B0: Optional[list | torch.Tensor] = None,
    eSpins: Optional[list[int]] = None,
    lB_select: Optional[int | list[int]] = None,
    dtype: torch.dtype = torch.complex128,
    device: str = 'cpu',
) -> torch.Tensor | tuple:
    """Higher-order Zeeman interaction Hamiltonian.

    Parameters
    ----------
    sys:
        Spin system with ``Ham`` dict containing higher-order Zeeman
        parameters.
    B0:
        Magnetic field vector in mT, shape ``(3,)``.  If given, returns
        the full Hamiltonian.  If ``None``, returns the tensor
        decomposition ``(G0, G1, G2, G3)``.
    eSpins:
        1-based electron spin indices (default: all).
    lB_select:
        Select specific orders in B (0, 1, 2, 3).  Default: all orders
        present in ``sys.Ham``.
    dtype:
        Output dtype.
    device:
        PyTorch device.

    Returns
    -------
    H : torch.Tensor
        If *B0* is given — Hamiltonian (MHz), shape ``(nStates, nStates)``.
    (G0, G1, G2, G3) : tuple
        If *B0* is ``None`` — field-derivative tensors:

        - **G0**: zero-field Hamiltonian (MHz)
        - **G1**: ``[dH/dBx, dH/dBy, dH/dBz]`` (MHz/mT)
        - **G2**: ``[[d²H/dBidBj]]`` 3×3 list (MHz/mT²)
        - **G3**: ``[[[d³H/dBidBjdBk]]]`` 3×3×3 list (MHz/mT³)
    """
    n = sys.nStates
    zero = lambda: torch.zeros(n, n, dtype=dtype, device=device)

    if sys.Ham is None or len(sys.Ham) == 0:
        if B0 is not None:
            return zero()
        return (zero(),
                [zero() for _ in range(3)],
                [[zero() for _ in range(3)] for _ in range(3)],
                [[[zero() for _ in range(3)] for _ in range(3)] for _ in range(3)])

    # Find highest order in B
    highest = 0
    for key in sys.Ham:
        lB_val = int(key[0])
        if lB_val > highest:
            highest = lB_val

    # ---- Full Hamiltonian at B0 ----
    if B0 is not None:
        if lB_select is not None:
            if isinstance(lB_select, int):
                lB_list = [lB_select]
            else:
                lB_list = list(lB_select)
        else:
            lB_list = list(range(highest + 1))

        H = _ham_ezho_at_field(sys, B0, eSpins, lB_list, dtype, device)
        H = (H + H.conj().T) / 2
        return H

    # ---- Tensor decomposition ----

    def _hermit(H):
        return (H + H.conj().T) / 2

    # G0: lB=0 at zero field
    G0 = _hermit(_ham_ezho_at_field(sys, [0, 0, 0], eSpins, [0], dtype, device))

    # G1: evaluate at unit field along each axis, lB=1 only
    G1 = [zero() for _ in range(3)]
    if highest >= 1:
        for ax in range(3):
            field = [0.0, 0.0, 0.0]
            field[ax] = 1.0
            G1[ax] = _hermit(_ham_ezho_at_field(
                sys, field, eSpins, [1], dtype, device))

    # G2: numerical differentiation via combination
    G2 = [[zero() for _ in range(3)] for _ in range(3)]
    if highest >= 2:
        # Diagonal: G2[n][n] = H(e_n, lB=2)
        for ax in range(3):
            field = [0.0, 0.0, 0.0]
            field[ax] = 1.0
            G2[ax][ax] = _hermit(_ham_ezho_at_field(
                sys, field, eSpins, [2], dtype, device))

        # Off-diagonal
        xyz = [0, 1, 2]
        for n_ax in range(3):
            others = [i for i in xyz if i != n_ax]
            a, b = others
            field = [1.0, 1.0, 1.0]
            field[n_ax] = 0.0
            H_mixed = _ham_ezho_at_field(sys, field, eSpins, [2], dtype, device)
            G2[a][b] = _hermit(0.5 * (H_mixed - G2[a][a] - G2[b][b]))
            G2[b][a] = G2[a][b]

    # G3: third-order tensor via polarisation identities
    G3 = [[[zero() for _ in range(3)] for _ in range(3)] for _ in range(3)]
    if highest >= 3:
        # Diagonal
        for ax in range(3):
            field = [0.0, 0.0, 0.0]
            field[ax] = 1.0
            G3[ax][ax][ax] = _hermit(_ham_ezho_at_field(
                sys, field, eSpins, [3], dtype, device))

        # Two-index terms (e.g., G3[a][b][b])
        xyz = [0, 1, 2]
        for n_ax in range(3):
            others = [i for i in xyz if i != n_ax]
            field_p = [1.0, 1.0, 1.0]
            field_p[n_ax] = 0.0
            lp = _ham_ezho_at_field(sys, field_p, eSpins, [3], dtype, device)

            for perm in permutations(others):
                a_idx, b_idx = perm
                field_m = list(field_p)
                field_m[b_idx] = -1.0
                lm = _ham_ezho_at_field(
                    sys, field_m, eSpins, [3], dtype, device)
                me = _hermit(
                    (1.0 / 6.0) * (lp + lm - 2 * G3[a_idx][a_idx][a_idx]))
                for idx_perm in permutations([a_idx, b_idx, b_idx]):
                    i, j, k = idx_perm
                    G3[i][j][k] = me

        # Three-index term (all distinct: G3[0][1][2] and permutations)
        H_all = _ham_ezho_at_field(
            sys, [1, 1, 1], eSpins, [3], dtype, device)
        dif = zero()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    dif = dif + G3[a][b][c]
        me = _hermit((1.0 / 6.0) * (H_all - dif))
        for idx_perm in permutations([0, 1, 2]):
            i, j, k = idx_perm
            G3[i][j][k] = me

    return G0, G1, G2, G3


# ---------------------------------------------------------------------------
# Field-nonlinear (quadratic and cubic) Zeeman terms for the simulators
# (Options.QuadraticZeeman)
# ---------------------------------------------------------------------------

def _ham_orders_present(sys: SpinSystem) -> set:
    """Orders lB of the nonzero entries of ``sys.Ham``."""
    out = set()
    if sys.Ham is None:
        return out
    for key, val in sys.Ham.items():
        if val is not None and bool(torch.any(torch.as_tensor(val) != 0)):
            out.add(int(key[0]))
    return out


def has_field_nonlinear_zeeman(sys: SpinSystem) -> bool:
    """True if ``sys.Ham`` has a nonzero term of order lB >= 2 in the field."""
    return any(lB >= 2 for lB in _ham_orders_present(sys))


def field_nonlinear_zeeman_ops(
    sys: SpinSystem,
    dtype: torch.dtype = torch.complex128,
    device: str = 'cpu',
) -> Optional[tuple]:
    """Quadratic and cubic field-derivative tensors of ``sys.Ham`` as stacked tensors.

    Returns ``None`` when ``sys.Ham`` has no nonzero lB >= 2 term, otherwise
    ``(G2, G3)`` with ``G2`` of shape ``(3, 3, n, n)`` and ``G3`` of shape
    ``(3, 3, 3, n, n)`` (``None`` when that order is absent), such that

        H(B) = H0 - B·μ + Σ_ij B_i B_j G2[i, j] + Σ_ijk B_i B_j B_k G3[i, j, k]

    equals ``ham(sys, B)``.  For a field ``B n`` along the unit vector ``n`` the
    nonlinear part is ``B² n·G2·n + B³ G3[n, n, n]``.  Raises
    ``NotImplementedError`` for terms of order lB >= 4, which the decomposition
    of :func:`ham_ezho` does not provide.  The ``Ham`` coefficients are constants
    (not on the autograd graph), as in :func:`ham_ezho`.
    """
    orders = _ham_orders_present(sys)
    if not any(lB >= 2 for lB in orders):
        return None
    if any(lB >= 4 for lB in orders):
        raise NotImplementedError(
            "Options.QuadraticZeeman: Sys.Ham terms of order lB >= 4 in the field are "
            "not supported (only lB = 2 and lB = 3)."
        )
    _, _, G2l, G3l = ham_ezho(sys, B0=None, dtype=dtype, device=device)
    G2 = torch.stack([torch.stack(row) for row in G2l]) if 2 in orders else None
    G3 = (torch.stack([torch.stack([torch.stack(r) for r in plane]) for plane in G3l])
          if 3 in orders else None)
    return G2, G3


def project_field_nonlinear_zeeman(ops: Optional[tuple], nvec: torch.Tensor) -> tuple:
    """Per-orientation ``(Q2, Q3)`` = ``(n·G2·n, G3[n,n,n])``, each ``(M, n, n)`` or None.

    ``nvec`` holds the unit field directions, shape ``(M, 3)``.
    """
    if ops is None:
        return None, None
    G2, G3 = ops
    Q2 = Q3 = None
    if G2 is not None:
        nv = nvec.to(device=G2.device, dtype=G2.dtype)
        Q2 = torch.einsum('mi,mj,ijab->mab', nv, nv, G2)
    if G3 is not None:
        nv = nvec.to(device=G3.device, dtype=G3.dtype)
        Q3 = torch.einsum('mi,mj,mk,ijkab->mab', nv, nv, nv, G3)
    return Q2, Q3


def check_quadratic_zeeman_unsupported(sys: SpinSystem, opt, where: str) -> None:
    """Raise when ``opt.QuadraticZeeman`` is on, ``sys.Ham`` has lB >= 2 terms and
    the calling solver (*where*) cannot include them."""
    if opt is not None and bool(getattr(opt, 'QuadraticZeeman', False)) and has_field_nonlinear_zeeman(sys):
        raise NotImplementedError(
            f"{where} does not support Options.QuadraticZeeman=True with field-quadratic/cubic "
            f"Sys.Ham terms (lB >= 2). Use pepper with Options.Method='matrix', or set "
            f"QuadraticZeeman=False to drop these terms as before."
        )
