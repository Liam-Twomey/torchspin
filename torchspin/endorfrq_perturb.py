"""ENDOR transition frequencies via perturbation theory (Iwasaki 1974).

Fast alternative to the matrix-diagonalization-based :func:`endorfrq`
for powder averaging.  Restricted to single-electron systems.

Reference
---------
  M. Iwasaki, J. Magn. Reson. 16, 417-423 (1974).
  "Second-order perturbation treatment of the general spin Hamiltonian
   in an arbitrary coordinate system"

Restrictions
------------
- Single electron spin (nElectrons == 1)
- No nuclear-nuclear couplings (Sys.nn)
- No A-strain or D-strain
- No parallel-mode ENDOR
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from torchspin.spinsystem import SpinSystem
from torchspin.constants import PLANCK, BMAGN, NMAGN
from torchspin.rotations import erot


def endorfrq_perturb(
    sys: SpinSystem,
    B_field_mT: float,
    phi: float = 0.0,
    theta: float = 0.0,
    order: int = 2,
    nuclei: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ENDOR transition frequencies via perturbation theory (Iwasaki 1974).

    Parameters
    ----------
    sys:
        Spin system.  Must have ``nElectrons == 1``.
    B_field_mT:
        Static magnetic field in mT.
    phi, theta:
        Orientation angles (radians).  The static field direction in the
        molecular frame is ``n = (sin θ cos φ, sin θ sin φ, cos θ)``.
    order:
        Perturbation order: 1 or 2 (default 2).
    nuclei:
        List of 0-based nucleus indices to include.  Default: all nuclei.

    Returns
    -------
    freqs : Tensor, shape ``(nTrans,)``
        ENDOR frequencies in MHz (positive).
    intensities : Tensor, shape ``(nTrans,)``
        Relative intensities (proportional to nuclear transition probability).

    Raises
    ------
    ValueError
        If system has more than one electron or other unsupported features.

    Notes
    -----
    The nuclear energy in state (mS, mI) is (Iwasaki 1974):

        E = mI · |K(mS)|  +  E2A  +  E1Q  +  E2Q  +  E2DA

    where K(mS) = mS · A·u + ν_I · h is the effective field at the nucleus,
    u = g·h/|g·h| is the effective g direction, and ν_I = -gₙ·μₙ·B₀/h
    is the nuclear Larmor frequency in MHz.

    ENDOR transitions connect states mI and mI+1 within the same mS manifold.
    """
    from torchspin.spinsystem import _reject_h0extra
    _reject_h0extra(sys, 'endorfrq_perturb (perturbation theory)')
    if sys.nElectrons != 1:
        raise ValueError(
            f"endorfrq_perturb: nElectrons must be 1, got {sys.nElectrons}."
        )
    if sys.nNuclei == 0:
        return torch.empty(0), torch.empty(0)

    device = sys.g.device
    dt = torch.float64
    B0_T = float(B_field_mT) * 1e-3  # mT → T

    # --- g tensor (3×3) in molecular frame ---------------------------------
    if sys.fullg:
        g = sys.g[:3, :].to(dt)
    else:
        g_diag = torch.diag(sys.g[0, :].to(dt))
        R_g2M = erot(sys.gFrame[0, :].detach().tolist()).T
        Rg = torch.as_tensor(R_g2M, dtype=dt, device=device)
        g = Rg @ g_diag @ Rg.T

    S = float(sys.S[0])
    SS1 = S * (S + 1)
    mS_arr = torch.arange(S, -S - 0.5, -1, dtype=dt, device=device)  # S,...,-S
    n_mS = int(2 * S + 1)

    # --- D tensor for high-spin --------------------------------------------
    high_spin = S > 0.5
    D_mat = None
    if high_spin and sys.D is not None:
        if sys.fullD:
            D_mat = sys.D[:3, :].to(dt)
        else:
            D_diag = torch.diag(sys.D[0, :].to(dt))
            R_D2M = erot(sys.DFrame[0, :].tolist()).T
            RD = torch.as_tensor(R_D2M, dtype=dt, device=device)
            D_mat = RD @ D_diag @ RD.T
        # Traceless (Iwasaki requirement)
        D_mat = D_mat - torch.eye(3, dtype=dt, device=device) * D_mat.trace() / 3

    if high_spin and D_mat is not None:
        tr_DD = float((D_mat @ D_mat).trace())
    else:
        tr_DD = 0.0

    # --- Nuclei to process -------------------------------------------------
    if nuclei is None:
        nuclei = list(range(sys.nNuclei))

    # --- Field direction ---------------------------------------------------
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    h = torch.tensor(
        [sin_t * math.cos(phi), sin_t * math.sin(phi), cos_t],
        dtype=dt, device=device,
    )

    gh = g.T @ h
    geff = gh.norm()
    u = gh / geff

    # Zero-order Zeeman energy at fixed B0 (MHz)
    E0 = float(geff) * BMAGN * B0_T / PLANCK * 1e-6  # Hz → MHz

    # --- ZFS precomputations -----------------------------------------------
    if high_spin and D_mat is not None:
        Du = D_mat @ u
        u_Du = float(u @ Du)
        u_DDu = float(Du @ Du)
        D1_sq = u_DDu - u_Du ** 2
        D2_sq = 2 * tr_DD + u_Du ** 2 - 4 * u_DDu
    else:
        Du = None
        u_Du = D1_sq = D2_sq = 0.0

    # Nuclear Larmor frequencies (MHz): nuI = -gn * nmagn * B0 / planck / 1e6
    nuI = torch.tensor(
        [-float(sys.gn[n]) * NMAGN * B0_T / PLANCK * 1e-6 for n in range(sys.nNuclei)],
        dtype=dt, device=device,
    )

    # --- Accumulate output ------------------------------------------------
    all_freqs = []
    all_ints = []

    for iNuc in nuclei:
        I_nuc = float(sys.I[iNuc])
        II1 = I_nuc * (I_nuc + 1)
        mI_arr = torch.arange(-I_nuc, I_nuc + 0.5, dtype=dt, device=device)  # -I..+I
        n_mI = int(2 * I_nuc + 1)

        # A tensor
        if sys.fullA:
            A_mat = sys.A[3 * iNuc : 3 * iNuc + 3, :].to(dt)
        else:
            A_diag = torch.diag(sys.A[iNuc, :].to(dt))
            R_A2M = erot(sys.AFrame[iNuc, :].tolist()).T
            RA = torch.as_tensor(R_A2M, dtype=dt, device=device)
            A_mat = RA @ A_diag @ RA.T

        # A @ u (hyperfine field direction per unit mS)
        Ah = A_mat @ u  # (3,)

        # Quadrupole
        has_quad = I_nuc > 0.5 and sys.Q is not None and iNuc < sys.Q.shape[0]
        if has_quad:
            Q_diag = torch.diag(sys.Q[iNuc, :].to(dt))
            if hasattr(sys, 'QFrame') and sys.QFrame is not None:
                R_Q2M = erot(sys.QFrame[iNuc, :].tolist()).T
                RQ = torch.as_tensor(R_Q2M, dtype=dt, device=device)
                Q_mat = RQ @ Q_diag @ RQ.T
            else:
                Q_mat = Q_diag
            Q_mat = Q_mat - torch.eye(3, dtype=dt, device=device) * Q_mat.trace() / 3
        else:
            Q_mat = None

        # Second-order A invariants (independent of mS and mI)
        if order >= 2:
            try:
                invA = torch.linalg.inv(A_mat)
            except Exception:
                invA = torch.zeros(3, 3, dtype=dt, device=device)
            trAA = float((A_mat.T @ A_mat).trace())
            detA = float(torch.linalg.det(A_mat))
            A_u_norm_sq = float((A_mat @ u).dot(A_mat @ u))

        # Nuclear energy for each (mS, mI): shape (n_mS, n_mI)
        NucEnergy = torch.zeros(n_mS, n_mI, dtype=dt, device=device)

        for imS in range(n_mS):
            mS_val = float(mS_arr[imS])

            # Effective field at nucleus
            HFfield = mS_val * Ah          # (3,)
            Larmor = nuI[iNuc] * h         # (3,)
            Kh = HFfield + Larmor          # (3,)
            normKh = Kh.norm()

            if normKh < 1e-30:
                k_vec = h.clone()
            else:
                k_vec = Kh / normKh

            # First-order nuclear energy
            E1A = mI_arr * float(normKh)

            # Strong coupling sign flip (Iwasaki convention)
            strong_coup = HFfield.norm() > Larmor.norm()
            if strong_coup and mS_val < 0:
                E1A = -E1A

            # First-order quadrupole
            if Q_mat is not None:
                kPk = float(k_vec @ Q_mat @ k_vec)
                E1Q = -kPk / 2 * (II1 - 3 * mI_arr ** 2)
            else:
                E1Q = torch.zeros(n_mI, dtype=dt, device=device)

            if order >= 2:
                # Second-order hyperfine
                Ak = A_mat.T @ k_vec          # A^T k, (3,)
                k_Au = float(Ak @ u)
                k_AAk = float(Ak @ Ak)

                A1_sq = k_AAk - k_Au ** 2
                A2 = detA * float(u @ invA @ k_vec)
                A3 = trAA - A_u_norm_sq - k_AAk + k_Au ** 2

                E2A = (
                    A1_sq * mS_val * mI_arr ** 2
                    - A2 * (SS1 - mS_val ** 2) * mI_arr
                    + A3 / 2 * mS_val * (II1 - mI_arr ** 2)
                ) / (2 * E0)
                if strong_coup and mS_val < 0:
                    E2A = -E2A

                # ZFS-hyperfine cross-term (high-spin only)
                if high_spin and Du is not None:
                    DA = float(Du @ Ak) - u_Du * k_Au
                    E2DA = -DA * (SS1 - 3 * mS_val ** 2) * mI_arr / E0
                else:
                    E2DA = torch.zeros(n_mI, dtype=dt, device=device)

                # Second-order quadrupole
                if Q_mat is not None:
                    kPPk = float(k_vec @ Q_mat @ Q_mat @ k_vec)
                    P1_sq = kPPk - kPk ** 2
                    P2_sq = 2 * float(Q_mat.trace()) ** 2 + kPk ** 2 - 4 * kPPk
                    denom = 2 * float(normKh) + 1e-30
                    E2Q = -(
                        P1_sq * mI_arr * (4 * II1 - 8 * mI_arr ** 2 - 1)
                        - P2_sq / 4 * mI_arr * (2 * II1 - 2 * mI_arr ** 2 - 1)
                    ) / denom
                else:
                    E2Q = torch.zeros(n_mI, dtype=dt, device=device)

                NucEnergy[imS] = E1A + E1Q + E2A + E2DA + E2Q
            else:
                NucEnergy[imS] = E1A + E1Q

        # ENDOR frequencies: |E(mI+1) - E(mI)| for each mS manifold
        dE = (NucEnergy[:, 1:] - NucEnergy[:, :-1]).abs()  # (n_mS, n_mI-1)
        freqs = dE.reshape(-1)

        # Intensities ∝ |<mI|I±|mI∓1>|² / (2I+1)
        # = (I+mI)(I-mI+1)/4 / (2I+1)  for mI = lower level
        gn = float(sys.gn[iNuc])
        pre = (gn * NMAGN / PLANCK * 1e-9) ** 2
        mat_sq = torch.tensor(
            [
                (I_nuc + float(mI_arr[im])) * (I_nuc - float(mI_arr[im]) + 1) / 4
                for im in range(n_mI - 1)
            ],
            dtype=dt, device=device,
        )
        # Same intensity for each mS manifold
        ints = mat_sq.repeat(n_mS) * (pre / n_mI)

        all_freqs.append(freqs)
        all_ints.append(ints)

    if not all_freqs:
        return torch.empty(0, device=device), torch.empty(0, device=device)

    return torch.cat(all_freqs), torch.cat(all_ints)
