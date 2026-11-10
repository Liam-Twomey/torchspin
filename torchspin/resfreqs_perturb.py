"""Frequency-swept EPR resonance frequencies via perturbation theory.

Computes EPR resonance frequencies (in GHz) for a fixed static field B₀
using the Iwasaki second-order perturbation expansion.  This is the
frequency-domain analogue of :func:`resfields_perturb`.

Reference
---------
  M. Iwasaki, J. Magn. Reson. 16, 417-423 (1974).

Restrictions
------------
- Single electron spin (nElectrons == 1)
- No orbital angular momentum (Sys.L)
- No nuclear-nuclear couplings (Sys.nn)
- No tilted D tensor with D-strain simultaneously
"""
from __future__ import annotations

import itertools
import math
from typing import Optional

import torch

from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.constants import PLANCK, BMAGN
from torchspin.rotations import erot


def resfreqs_perturb(
    sys: SpinSystem,
    phi: float,
    theta: float,
    exp: Experiment,
    opt: Optional[Options] = None,
    order: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find resonance frequencies using perturbation theory.

    Parameters
    ----------
    sys:
        Spin system.  Must have ``nElectrons == 1``.
    phi, theta:
        Orientation angles (radians).  The static field direction in the
        molecular frame is ``n = (sin θ cos φ, sin θ sin φ, cos θ)``.
    exp:
        Experiment parameters.  Must set ``exp.Field`` (in mT) and
        optionally ``exp.Range`` ([ν_min, ν_max] in GHz).
    opt:
        Options (Threshold for intensity cutoff).  If None, uses defaults.
    order:
        Perturbation order: 1 or 2.  Default is 2 (second-order).

    Returns
    -------
    nu_res : Tensor, shape ``(nTransitions,)``
        Resonance frequencies in GHz.
    intensities : Tensor, shape ``(nTransitions,)``
        Transition intensities (a.u.).

    Notes
    -----
    At fixed field B₀, the EPR transition frequency for (mS) → (mS−1) is:

        ν = E₀ + δE₁_D + δE₂_D + Σₙ (δE₁_A + δE₂_A + δE₂_DA)

    where E₀ = g_eff · μ_B · B₀ / h is the Zeeman frequency and the δE
    terms are the Iwasaki first- and second-order perturbation corrections.
    """
    if opt is None:
        opt = Options()

    from torchspin.spinsystem import _reject_h0extra
    _reject_h0extra(sys, 'resfreqs_perturb (perturbation theory)')
    if sys.nElectrons != 1:
        raise ValueError(
            f"resfreqs_perturb: nElectrons must be 1, got {sys.nElectrons}."
        )

    device = sys.g.device
    dt = torch.float64

    B0_T = float(exp.Field) * 1e-3  # mT → T

    # --- g tensor (3×3) in molecular frame --------------------------------
    if sys.fullg:
        g = sys.g[:3, :].to(dt)
    else:
        g_diag = torch.diag(sys.g[0, :].to(dt))
        R_g2M = erot(sys.gFrame[0, :].tolist()).T
        Rg = torch.as_tensor(R_g2M, dtype=dt, device=device)
        g = Rg @ g_diag @ Rg.T

    S = float(sys.S[0])
    n_transitions = int(2 * S)

    # --- D tensor for high-spin -------------------------------------------
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
        # Make traceless (Iwasaki requirement)
        D_mat = D_mat - torch.eye(3, dtype=dt, device=device) * D_mat.trace() / 3

    if high_spin and D_mat is not None:
        tr_DD = float((D_mat @ D_mat).trace())
    else:
        tr_DD = 0.0

    # --- Nuclear spins and hyperfine tensors ------------------------------
    n_nuclei = sys.nNuclei
    I_list = []
    A_list = []

    if n_nuclei > 0:
        I_list = [float(sys.I[n]) for n in range(n_nuclei)]
        for iNuc in range(n_nuclei):
            if sys.fullA:
                A_mat = sys.A[3 * iNuc : 3 * iNuc + 3, :].to(dt)
            else:
                A_diag = torch.diag(sys.A[iNuc, :].to(dt))
                R_A2M = erot(sys.AFrame[iNuc, :].tolist()).T
                RA = torch.as_tensor(R_A2M, dtype=dt, device=device)
                A_mat = RA @ A_diag @ RA.T
            A_list.append(A_mat)

        # Enumerate all nuclear mI combinations
        mI_ranges = [
            torch.arange(-I, I + 1, 1, dtype=dt, device=device)
            for I in I_list
        ]
        mI_combos = torch.tensor(
            list(itertools.product(*[r.tolist() for r in mI_ranges])),
            dtype=dt, device=device,
        )  # (n_nuc_states, n_nuclei)
        n_nuc_states_total = mI_combos.shape[0]
    else:
        mI_combos = None
        n_nuc_states_total = 1

    # --- Field direction in molecular frame --------------------------------
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    n0 = torch.tensor(
        [sin_t * math.cos(phi), sin_t * math.sin(phi), cos_t],
        dtype=dt, device=device,
    )

    # Effective g along field direction
    g_n = g.T @ n0   # (3,)
    geff = g_n.norm()
    u = g_n / geff    # unit vector in g·n direction

    # Zero-order Zeeman energy (MHz)
    E0 = float(geff) * BMAGN * B0_T / PLANCK * 1e-6  # Hz → MHz

    # --- Transition intensity prefactor (perpendicular mode) ---------------
    gg = g @ g.T
    tr_gg = float(gg.trace())
    g_u_norm_sq = float((g @ u).dot(g @ u))

    mS_trans = torch.arange(S, -S, -1, dtype=dt, device=device)  # S,...,-S+1
    c_sq = (BMAGN / 2) ** 2 * (S * (S + 1) - mS_trans * (mS_trans - 1))
    c_sq = c_sq / (PLANCK * 1e9) ** 2
    intensity_factor = c_sq * (tr_gg - g_u_norm_sq) / 2  # avg over chi, per transition

    # --- ZFS precomputations ----------------------------------------------
    if high_spin and D_mat is not None:
        Du = D_mat @ u
        u_Du = float(u @ Du)
        u_DDu = float(Du @ Du)
        D1_sq = u_DDu - u_Du ** 2
        D2_sq = 2 * tr_DD + u_Du ** 2 - 4 * u_DDu
    else:
        Du = None
        u_Du = D1_sq = D2_sq = 0.0

    # --- Hyperfine precomputations ----------------------------------------
    k_list = []       # unit vectors k_n = A_n @ u / |A_n @ u|
    nK_list = []      # |A_n @ u|
    invA_list = []
    trAA_list = []
    detA_list = []
    A_u_norm_sq_list = []

    if n_nuclei > 0:
        for iNuc in range(n_nuclei):
            A_mat = A_list[iNuc]
            K = A_mat @ u
            nK = K.norm()
            k = K / (nK + 1e-30)
            k_list.append(k)
            nK_list.append(nK)
            if order >= 2:
                try:
                    invA_list.append(torch.linalg.inv(A_mat))
                except Exception:
                    invA_list.append(torch.zeros(3, 3, dtype=dt, device=device))
                trAA_list.append(float((A_mat.T @ A_mat).trace()))
                detA_list.append(float(torch.linalg.det(A_mat)))
                A_u_norm_sq_list.append(float((A_mat @ u).dot(A_mat @ u)))

    # --- Compute resonance frequencies for each EPR transition (mS→mS-1) --
    nu_res_list = []
    int_list = []

    for idx_mS in range(n_transitions):
        mS = S - idx_mS  # S, S-1, ..., -S+1

        # First-order ZFS
        if high_spin and D_mat is not None:
            dE1D = -u_Du / 2 * (3 - 6 * mS)
        else:
            dE1D = 0.0

        # First-order hyperfine: each nuclear combo contributes Σ_n mI_n * |A_n @ u|
        if n_nuclei > 0:
            dE1A = torch.zeros(n_nuc_states_total, dtype=dt, device=device)
            for iNuc in range(n_nuclei):
                dE1A = dE1A + mI_combos[:, iNuc] * float(nK_list[iNuc])
        else:
            dE1A = torch.tensor([0.0], dtype=dt, device=device)

        # Second-order corrections
        if order >= 2:
            # ZFS second-order
            if high_spin and D_mat is not None:
                x = D1_sq * (
                    4 * S * (S + 1) - 3 * (8 * mS ** 2 - 8 * mS + 3)
                ) - D2_sq / 4 * (2 * S * (S + 1) - 3 * (2 * mS ** 2 - 2 * mS + 1))
                dE2D = -x / (2 * E0)
            else:
                dE2D = 0.0

            # Hyperfine second-order
            if n_nuclei > 0:
                dE2A = torch.zeros(n_nuc_states_total, dtype=dt, device=device)
                dE2DA = torch.zeros(n_nuc_states_total, dtype=dt, device=device)
                for iNuc in range(n_nuclei):
                    k_vec = k_list[iNuc]
                    A_mat = A_list[iNuc]
                    mI_vals = mI_combos[:, iNuc]
                    I_nuc = I_list[iNuc]
                    II1 = I_nuc * (I_nuc + 1)

                    Ak = A_mat.T @ k_vec
                    k_Au = float(Ak @ u)
                    k_AAk = float(Ak @ Ak)

                    A1_sq = k_AAk - k_Au ** 2
                    A2 = detA_list[iNuc] * float(u @ invA_list[iNuc] @ k_vec)
                    A3 = (trAA_list[iNuc] - A_u_norm_sq_list[iNuc]
                          - k_AAk + k_Au ** 2)

                    x = (A1_sq * mI_vals ** 2
                         - A2 * (1 - 2 * mS) * mI_vals
                         + A3 / 2 * (II1 - mI_vals ** 2))
                    dE2A = dE2A + x / (2 * E0)

                    # ZFS-hyperfine cross-term (high-spin only)
                    if high_spin and Du is not None:
                        DA = float(Du @ Ak) - u_Du * k_Au
                        dE2DA = dE2DA + DA * (3 - 6 * mS) * mI_vals / E0
            else:
                dE2A = torch.tensor([0.0], dtype=dt, device=device)
                dE2DA = torch.tensor([0.0], dtype=dt, device=device)

            nu = (E0 + dE1D + dE2D + (dE1A + dE2A + dE2DA)) * 1e-3  # MHz → GHz
        else:
            nu = (E0 + dE1D + dE1A) * 1e-3  # GHz

        # Intensity: same for all nuclear substates (first-order spin matrix elements)
        n_nuc = dE1A.shape[0]
        int_vals = intensity_factor[idx_mS].expand(n_nuc) / n_nuc_states_total

        nu_res_list.append(nu)
        int_list.append(int_vals)

    if not nu_res_list:
        return torch.empty(0, device=device), torch.empty(0, device=device)

    nu_all = torch.cat(nu_res_list)
    int_all = torch.cat(int_list)

    # Filter by frequency range if specified
    if exp.Range is not None and len(exp.Range) == 2:
        nu_min = float(exp.Range[0])
        nu_max = float(exp.Range[1])
        mask = (nu_all >= nu_min) & (nu_all <= nu_max)
        nu_all = nu_all[mask]
        int_all = int_all[mask]

    return nu_all, int_all
