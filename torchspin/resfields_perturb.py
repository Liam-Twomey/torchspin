"""Resonance field calculation using perturbation theory (Iwasaki formulas).

Fast alternative to matrix diagonalization for single-electron systems.
Based on:
  M. Iwasaki, J. Magn. Reson. 16, 417-423 (1974)
  "Second-order perturbation treatment of the general spin hamiltonian
   in an arbitrary coordinate system"
  https://doi.org/10.1016/0022-2364(74)90223-6

Restrictions:
- Single electron spin only (S = 1/2, 1, 3/2, etc.)
- No orbital angular momentum
- No nuclear-nuclear coupling

Accuracy:
- S=1/2 with hyperfine: ~1e-8 relative error (~0.000001 mT)
- S=1 with ZFS: ~1e-4 relative error (~0.04 mT at X-band)
- S=3/2 with ZFS: ~1e-3 relative error (~0.5 mT at X-band)
- Higher-order terms become important for strong ZFS (D > 100 MHz)
"""
from __future__ import annotations

import itertools
import math
from typing import Optional

import torch

from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.constants import PLANCK, BMAGN, BOLTZMANN
from torchspin.rotations import erot


def resfields_perturb_batch(
    sys: SpinSystem,
    phi_batch: torch.Tensor,
    theta_batch: torch.Tensor,
    exp: Experiment,
    opt: Optional[Options] = None,
    order: int = 2,
    return_full: bool = False,
    photo_weights: Optional[torch.Tensor] = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Vectorized resfields_perturb over a batch of orientations.

    Parameters
    ----------
    sys:
        Spin system (nElectrons == 1).
    phi_batch, theta_batch:
        Orientation angles (radians), shape ``(M,)``.
    exp:
        Experiment parameters.
    opt:
        Options.
    order:
        Perturbation order: 1 or 2.
    return_full:
        If True, return a tensor of shape ``(n_combos * n_mS_transitions,)``
        per orientation where out-of-range transitions are NaN (intensities 0).
        This preserves a consistent transition index across all orientations,
        needed for pepper's transition-by-transition interpolation.  If False
        (default), only in-range transitions are returned (variable length).

    phi_batch, theta_batch:
        Orientation angles (radians), shape ``(M,)``.
    exp:
        Experiment parameters.
    opt:
        Options.
    order:
        Perturbation order: 1 or 2.

    Returns
    -------
    B_res_list:
        List of M tensors, one per orientation.
    intens_list:
        List of M tensors, one per orientation.
    """
    if opt is None:
        opt = Options()

    from torchspin.spinsystem import _reject_h0extra
    _reject_h0extra(sys, 'resfields_perturb (perturbation theory)')

    from torchspin.ham_ezho import check_quadratic_zeeman_unsupported
    check_quadratic_zeeman_unsupported(sys, opt, 'resfields_perturb_batch')

    if sys.nElectrons != 1:
        raise ValueError(
            f"Perturbation theory requires exactly 1 electron, system has {sys.nElectrons}."
        )

    M = phi_batch.shape[0]
    S = sys.S[0]
    n_transitions = int(2 * S)
    device = phi_batch.device  # Use caller's device (pepper moves angles to opt.device)
    dtype = torch.float64

    # ── Build orientation-independent g, A, D tensors ─────────────────────
    if sys.fullg:
        g = sys.g[:3, :].to(dtype=dtype, device=device)
    else:
        g_diag = torch.diag(sys.g[0, :].to(dtype=dtype, device=device))
        R_g2M = erot(sys.gFrame[0, :], device=device).T
        g = R_g2M @ g_diag @ R_g2M.T

    n_nuclei = sys.nNuclei
    A_list = []
    if n_nuclei > 0:
        for iNuc in range(n_nuclei):
            if sys.fullA:
                A_mat = sys.A[3 * iNuc: 3 * iNuc + 3, :].to(dtype=dtype, device=device)
            else:
                A_diag = torch.diag(sys.A[iNuc, :].to(dtype=dtype, device=device))
                R_A2M = erot(sys.AFrame[iNuc, :], device=device).T
                A_mat = R_A2M @ A_diag @ R_A2M.T
            A_list.append(A_mat)

        mI_ranges = [
            torch.arange(-I, I + 1, 1, dtype=dtype, device=device)
            for I in sys.I
        ]
        mI_combos = torch.tensor(
            list(itertools.product(*[r.tolist() for r in mI_ranges])),
            dtype=dtype, device=device,
        )  # (n_combos, n_nuclei)
    else:
        mI_combos = None

    high_spin = S > 0.5
    if high_spin and sys.D is not None:
        if sys.fullD:
            D = sys.D[:3, :].to(dtype=dtype, device=device)
        else:
            D_diag = torch.diag(sys.D[0, :].to(dtype=dtype, device=device))
            R_D2M = erot(sys.DFrame[0, :], device=device).T
            D = R_D2M @ D_diag @ R_D2M.T
        D = D - torch.eye(3, dtype=dtype, device=device) * D.trace() / 3
    else:
        D = None

    E0 = exp.mwFreq * 1e3  # GHz → MHz
    B_lo, B_hi = float(exp.Range[0]), float(exp.Range[1])

    # ── Batch orientation vectors: (M, 3) ─────────────────────────────────
    sin_th = torch.sin(theta_batch).to(dtype)
    cos_th = torch.cos(theta_batch).to(dtype)
    cos_ph = torch.cos(phi_batch).to(dtype)
    sin_ph = torch.sin(phi_batch).to(dtype)
    n0_batch = torch.stack([sin_th * cos_ph, sin_th * sin_ph, cos_th], dim=1)  # (M, 3)

    # g_n_batch = (g.T @ n0.T).T = n0 @ g  ... actually g.T @ n0 per vector:
    # g.T[i,j]=g[j,i], so (g.T @ n0)[i] = sum_j g[j,i]*n0[j] = (n0 @ g)[i] when treating row@mat
    g_n_batch = n0_batch @ g                         # (M, 3)
    g_eff_batch = torch.linalg.norm(g_n_batch, dim=1)  # (M,)
    u_batch = g_n_batch / g_eff_batch.unsqueeze(1)     # (M, 3)
    pre_batch = 1e9 * PLANCK / (g_eff_batch * BMAGN)   # (M,) MHz→mT

    # Intensity factor (same for all orientations within an mS transition)
    mS_transitions = torch.arange(S, -S, -1, dtype=dtype, device=device)
    c_sq = (BMAGN / 2) ** 2 * (S * (S + 1) - mS_transitions * (mS_transitions - 1))
    c_sq = c_sq / (1e9 * PLANCK) ** 2

    gg = g @ g.T
    tr_gg = gg.trace()                      # tensor: keeps the g dependence on the autograd graph
    # |g @ u|² per orientation: (u @ g.T)·(g @ u) = ||g @ u||² = ||(u @ g.T)||²
    g_u_batch = u_batch @ g.T                           # (M, 3)
    g_u_norm_sq_batch = (g_u_batch * g_u_batch).sum(dim=1)  # (M,)
    # χ-averaged transition rate for the excitation mode (EasySpin resfields_perturb.m /
    # p_excitationgeometry): linear (1-ξ1²)/2, unpolarized (1+ξk²)/4, circular
    # (1+ξk²)/2 ± 2ξk²·det(g)/|gᵀu| times (tr(ggᵀ) - |gu|²)
    from torchspin.excitation import excitation_geometry, exp_mw_mode
    geom = excitation_geometry(exp_mw_mode(exp))
    aniso = (tr_gg - g_u_norm_sq_batch)                                   # (M,)
    if geom.kind == 'linear':
        mode_factor = (1.0 - geom.xi1 ** 2) / 2.0 * aniso
    elif geom.kind == 'unpolarized':
        mode_factor = (1.0 + geom.xik ** 2) / 4.0 * aniso
    else:
        gT_u_norm = torch.linalg.norm(u_batch @ g, dim=1)                 # |gᵀu|
        mode_factor = ((1.0 + geom.xik ** 2) / 2.0 * aniso
                       + geom.sense * 2.0 * geom.xik ** 2 * torch.linalg.det(g) / gT_u_norm)
    intens_factor_batch = c_sq.unsqueeze(1) * mode_factor.unsqueeze(0)
    # × dBdE = (h/μB·1e9)/g_eff (mT/MHz; EasySpin resfields_perturb.m), so the
    # intensities are consistent with the matrix-diagonalization path.
    intens_factor_batch = intens_factor_batch * pre_batch.unsqueeze(0)
    # intens_factor_batch: (n_transitions, M)

    # ZFS quantities
    if high_spin and D is not None:
        # u_D_u: (M,)
        u_Du_batch = (u_batch @ D * u_batch).sum(dim=1)     # (M,)
        Du_batch = u_batch @ D                               # (M, 3)
        DDu_batch = Du_batch @ D                             # (M, 3)  D symmetric
        u_D_Du_batch = (u_batch * DDu_batch).sum(dim=1)     # (M,)
        tr_DD = (D @ D).trace()
        D1_sq_batch = tr_DD - u_Du_batch ** 2               # (M,)
        D2_sq_batch = 2 * tr_DD + u_Du_batch ** 2 - 4 * u_D_Du_batch  # (M,)

    # Hyperfine: precompute per-nucleus quantities, vectorized over orientations
    nK_batch_list = []    # (M,) per nucleus
    k_batch_list = []     # (M, 3) per nucleus
    invA_list_prep = []
    det_A_list_prep = []
    tr_AA_list_prep = []

    if n_nuclei > 0:
        for iNuc in range(n_nuclei):
            A_mat = A_list[iNuc]
            K_batch = u_batch @ A_mat.T                     # (M, 3)
            nK_b = torch.linalg.norm(K_batch, dim=1)        # (M,)
            k_b = K_batch / (nK_b.unsqueeze(1) + 1e-30)     # (M, 3)
            nK_batch_list.append(nK_b)
            k_batch_list.append(k_b)
            if order >= 2:
                invA_list_prep.append(torch.linalg.inv(A_mat))   # once
                tr_AA_list_prep.append((A_mat @ A_mat).trace())  # once (tensor: on the graph)
                det_A_list_prep.append(torch.linalg.det(A_mat))  # once (tensor: on the graph)

        # nK_all: (M, n_nuclei)
        nK_all_batch = torch.stack(nK_batch_list, dim=1)

    # ── Boltzmann factor precomputation ────────────────────────────────────
    # pf_mhz: h / (kT) in units of 1/MHz (so that pf_mhz * E_MHz is dimensionless)
    do_boltzmann = exp.Temperature is not None and exp.Temperature > 0
    if do_boltzmann:
        pf_mhz = PLANCK * 1e6 / (BOLTZMANN * exp.Temperature)
        n_states = int(2 * S + 1)
        k_vals = torch.arange(n_states, dtype=dtype, device=device)  # (n_states,)
        # mS values: mS_k[k] = S - k  (k=0 → mS=S highest, k=n-1 → mS=-S lowest)
        mS_k_vals = S - k_vals  # (n_states,)
        # ZFS first-order coefficient: (u_Du/2) * (3*mS_k² - S*(S+1))
        # Only non-zero when high_spin and D is set (u_Du_batch already computed above)
        _use_zfs_boltz = high_spin and D is not None
        if _use_zfs_boltz:
            _S_S1 = S * (S + 1)
            _mS_k_sq = mS_k_vals ** 2  # (n_states,)

    # ── Loop over electron spin transitions (usually just 1 for S=1/2) ─────
    B_res_M = [[] for _ in range(M)]
    I_res_M  = [[] for _ in range(M)]
    B_full_parts: list = []
    I_full_parts: list = []

    for idx_mS in range(n_transitions):
        mS = S - idx_mS

        # Intensity: (M,)
        intens_M = intens_factor_batch[idx_mS]  # (M,)
        if photo_weights is not None:
            intens_M = intens_M * photo_weights.to(device=intens_M.device, dtype=intens_M.dtype)
        # Each hyperfine line carries 1/nNucSublevels of the electron transition
        # intensity (EasySpin: Int = repelem(Intensity,nNucSublevels)/nNucSublevels)
        if mI_combos is not None:
            intens_M = intens_M / mI_combos.shape[0]

        # First-order ZFS: (M,)
        if high_spin and D is not None:
            E1D_batch = -u_Du_batch / 2 * (3 - 6 * mS)
        else:
            E1D_batch = torch.zeros(M, dtype=dtype, device=device)

        # First-order hyperfine: (M, n_combos)
        if n_nuclei > 0:
            E1A_batch = nK_all_batch @ mI_combos.T   # (M, n_combos)
        else:
            E1A_batch = torch.zeros(M, 1, dtype=dtype, device=device)

        # Second-order corrections
        if order >= 2:
            # ZFS second-order: (M,)
            if high_spin and D is not None:
                x_batch = (
                    D1_sq_batch * (4 * S * (S + 1) - 3 * (8 * mS ** 2 - 8 * mS + 3))
                    - D2_sq_batch / 4 * (2 * S * (S + 1) - 3 * (2 * mS ** 2 - 2 * mS + 1))
                )
                E2D_batch = -x_batch / (2 * E0)
            else:
                E2D_batch = torch.zeros(M, dtype=dtype, device=device)

            # Hyperfine second-order: (M, n_combos)
            if n_nuclei > 0:
                E2A_batch = torch.zeros(M, mI_combos.shape[0], dtype=dtype, device=device)
                E2DA_batch = torch.zeros(M, mI_combos.shape[0], dtype=dtype, device=device)

                for iNuc in range(n_nuclei):
                    A_mat = A_list[iNuc]
                    k_b = k_batch_list[iNuc]          # (M, 3)
                    invA = invA_list_prep[iNuc]        # (3, 3)
                    tr_AA = tr_AA_list_prep[iNuc]
                    det_A = det_A_list_prep[iNuc]
                    mI_vals = mI_combos[:, iNuc]       # (n_combos,)
                    I_nuc = sys.I[iNuc]

                    Ak_b = k_b @ A_mat               # (M, 3)  = A.T @ k per orientation
                    k_A_u_b = (Ak_b * u_batch).sum(dim=1)   # (M,)
                    k_AA_k_b = (Ak_b * Ak_b).sum(dim=1)     # (M,)

                    A1_sq_b = k_AA_k_b - k_A_u_b ** 2       # (M,)
                    # A2: det_A * (u @ invA @ k) per orientation
                    invA_k_b = k_b @ invA.T                  # (M, 3)
                    A2_b = det_A * (u_batch * invA_k_b).sum(dim=1)  # (M,)

                    Au_b = u_batch @ A_mat.T                 # (M, 3)
                    Au_norm_sq_b = (Au_b * Au_b).sum(dim=1)  # (M,)
                    A3_b = tr_AA - Au_norm_sq_b - k_AA_k_b + k_A_u_b ** 2  # (M,)

                    II1 = I_nuc * (I_nuc + 1)
                    # x[m, c] = A1_sq[m]*mI[c]^2 - A2[m]*(1-2*mS)*mI[c] + A3[m]/2*(II1-mI[c]^2)
                    # Broadcast (M,) × (n_combos,) → (M, n_combos)
                    mI2 = mI_vals ** 2                        # (n_combos,)
                    x_batch_nuc = (
                        A1_sq_b.unsqueeze(1) * mI2.unsqueeze(0)
                        - A2_b.unsqueeze(1) * (1 - 2 * mS) * mI_vals.unsqueeze(0)
                        + A3_b.unsqueeze(1) / 2 * (II1 - mI2.unsqueeze(0))
                    )
                    E2A_batch += x_batch_nuc / (2 * E0)

                    if high_spin and D is not None:
                        DA_b = (Du_batch * Ak_b).sum(dim=1) - u_Du_batch * k_A_u_b  # (M,)
                        y_batch = DA_b.unsqueeze(1) * (3 - 6 * mS) * mI_vals.unsqueeze(0)
                        E2DA_batch -= y_batch / E0
            else:
                E2A_batch = torch.zeros(M, 1, dtype=dtype, device=device)
                E2DA_batch = torch.zeros(M, 1, dtype=dtype, device=device)
        else:
            E2D_batch = torch.zeros(M, dtype=dtype, device=device)
            n_c = mI_combos.shape[0] if n_nuclei > 0 else 1
            E2A_batch = torch.zeros(M, n_c, dtype=dtype, device=device)
            E2DA_batch = torch.zeros(M, n_c, dtype=dtype, device=device)

        # Resonance fields: (M, n_combos)
        E_corr = (E0
                  - E1D_batch.unsqueeze(1)
                  - E2D_batch.unsqueeze(1)
                  - E1A_batch
                  - E2A_batch
                  - E2DA_batch)                              # (M, n_combos)
        B_res_batch = E_corr * pre_batch.unsqueeze(1)        # (M, n_combos)

        # Field range filter
        in_range = (B_res_batch >= B_lo) & (B_res_batch <= B_hi)  # (M, n_combos)

        # Intensity threshold
        if opt.Threshold > 0:
            above_thresh = intens_M >= opt.Threshold          # (M,)
        else:
            above_thresh = torch.ones(M, dtype=torch.bool, device=device)

        # Boltzmann population weighting: pol[m,c] = pop(mS) - pop(mS-1)
        if do_boltzmann:
            # State energies: Zeeman + 1st-order ZFS [MHz]
            # E[m,c,k] = -mS_k * B_res[m,c] / pre_batch[m]  (Zeeman)
            #           + (u_Du[m]/2) * (3*mS_k² - S*(S+1))  (1st-order ZFS)
            B_over_pre = B_res_batch / pre_batch.unsqueeze(1)  # (M, n_combos)
            E_state = (-mS_k_vals.unsqueeze(0).unsqueeze(0)
                       * B_over_pre.unsqueeze(2))              # (M, n_combos, n_states)
            if _use_zfs_boltz:
                # ZFS correction to state energies (orientation-dependent)
                E_zfs = (u_Du_batch.unsqueeze(1).unsqueeze(2) / 2
                         * (3 * _mS_k_sq - _S_S1))            # (M, 1, n_states)
                E_state = E_state + E_zfs
            E_min = E_state.amin(dim=2, keepdim=True)          # (M, n_combos, 1)
            boltz = torch.exp(-pf_mhz * (E_state - E_min))    # (M, n_combos, n_states)
            Z = boltz.sum(dim=2, keepdim=True)
            pop = boltz / Z                                    # (M, n_combos, n_states)
            # k_vals ordering: k=0 → mS=S, k=n-1 → mS=-S
            # Transition mS→mS-1: upper state k=idx_mS (mS), lower state k=idx_mS+1 (mS-1)
            # With Zeeman alone mS is lower energy, so absorption ∝ pop[idx_mS] - pop[idx_mS+1]
            # With large ZFS the ordering can flip — but pop difference remains the right sign
            # if computed from actual state energies (hence amin normalization above)
            pol = pop[..., idx_mS] - pop[..., idx_mS + 1]    # (M, n_combos)
            intens_weighted = intens_M.unsqueeze(1) * pol
        else:
            intens_weighted = intens_M.unsqueeze(1).expand_as(B_res_batch)

        if return_full:
            # Keep all combos with consistent indexing; out-of-range → NaN / 0
            B_part = B_res_batch.clone()
            keep = in_range & above_thresh.unsqueeze(1)
            B_part[~keep] = float('nan')
            I_part = intens_weighted.clone()
            I_part[~keep] = 0.0
            B_full_parts.append(B_part)   # (M, n_combos)
            I_full_parts.append(I_part)   # (M, n_combos)
        else:
            # Collect per orientation (variable length — only in-range)
            for m in range(M):
                if not above_thresh[m]:
                    continue
                mask_m = in_range[m]
                if not mask_m.any():
                    continue
                B_res_M[m].append(B_res_batch[m][mask_m])
                I_res_M[m].append(intens_weighted[m][mask_m])

    if return_full:
        B_full = torch.cat(B_full_parts, dim=1)  # (M, n_combos * n_transitions)
        I_full = torch.cat(I_full_parts, dim=1)
        return [B_full[m] for m in range(M)], [I_full[m] for m in range(M)]

    # Assemble compressed output lists
    B_res_list = []
    intens_list = []
    for m in range(M):
        if B_res_M[m]:
            B_res_list.append(torch.cat(B_res_M[m]))
            intens_list.append(torch.cat(I_res_M[m]))
        else:
            B_res_list.append(torch.zeros(0, dtype=dtype, device=device))
            intens_list.append(torch.zeros(0, dtype=dtype, device=device))

    return B_res_list, intens_list


def resfields_perturb(
    sys: SpinSystem,
    phi: float,
    theta: float,
    exp: Experiment,
    opt: Optional[Options] = None,
    order: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find resonance fields using perturbation theory.

    Much faster than matrix diagonalization for systems with many nuclear
    states, but restricted to single-electron systems.

    Parameters
    ----------
    sys:
        Spin system.  Must have ``nElectrons == 1``.
    phi, theta:
        Orientation angles (radians).  Static field direction is:
        n = (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)).
    exp:
        Experiment parameters (mwFreq in GHz, Range in mT).
    opt:
        Options (Threshold for intensity cutoff).  If None, uses defaults.
    order:
        Perturbation order: 1 or 2.  Default is 2 (second-order).

    Returns
    -------
    B_res:
        Resonance fields in mT, shape ``(nTransitions,)``.
    intensities:
        Transition intensities (a.u.), shape ``(nTransitions,)``.

    Raises
    ------
    ValueError:
        If system has more than one electron or uses unsupported features.
    """
    if opt is None:
        opt = Options()

    from torchspin.ham_ezho import check_quadratic_zeeman_unsupported
    check_quadratic_zeeman_unsupported(sys, opt, 'resfields_perturb')

    # Validate system
    from torchspin.spinsystem import _reject_h0extra
    _reject_h0extra(sys, 'resfields_perturb (perturbation theory)')
    if sys.nElectrons != 1:
        raise ValueError(
            f"Perturbation theory requires exactly 1 electron, system has {sys.nElectrons}."
        )

    S = sys.S[0]  # electron spin
    n_transitions = int(2 * S)  # number of allowed EPR transitions

    # Extract system parameters
    device = torch.device(opt.device)
    dtype_real = torch.float64

    # g tensor (3x3) in molecular frame
    if sys.fullg:
        g = sys.g[:3, :].to(dtype=dtype_real, device=device)  # (3, 3)
    else:
        # Rotate from g-frame to molecular frame
        g_diag = torch.diag(sys.g[0, :].to(dtype=dtype_real, device=device))  # (3, 3)
        R_g2M = erot(sys.gFrame[0, :], device=device).T  # g → molecular
        g = R_g2M @ g_diag @ R_g2M.T

    # D tensor for high-spin (3x3, traceless)
    high_spin = S > 0.5
    if high_spin and sys.D is not None:
        if sys.fullD:
            D = sys.D[:3, :].to(dtype=dtype_real, device=device)  # (3, 3)
        else:
            D_diag = torch.diag(sys.D[0, :].to(dtype=dtype_real, device=device))
            R_D2M = erot(sys.DFrame[0, :], device=device).T
            D = R_D2M @ D_diag @ R_D2M.T
        # Make traceless (required for Iwasaki)
        D = D - torch.eye(3, dtype=dtype_real, device=device) * D.trace() / 3
    else:
        D = None

    # Nuclear spins and hyperfine
    n_nuclei = sys.nNuclei
    I_list = []
    A_list = []
    mI_combos = None

    if n_nuclei > 0:
        I_list = sys.I
        # Build list of A tensors (each 3x3)
        for iNuc in range(n_nuclei):
            if sys.fullA:
                A_mat = sys.A[3 * iNuc : 3 * iNuc + 3, :].to(dtype=dtype_real, device=device)
            else:
                A_diag = torch.diag(sys.A[iNuc, :].to(dtype=dtype_real, device=device))
                R_A2M = erot(sys.AFrame[iNuc, :], device=device).T
                A_mat = R_A2M @ A_diag @ R_A2M.T
            A_list.append(A_mat)

        # Enumerate all nuclear spin combinations: mI = -I, -I+1, ..., I
        # For N nuclei with spins I_k, this gives prod(2*I_k + 1) combinations
        mI_ranges = [
            torch.arange(-I, I + 1, 1, dtype=dtype_real, device=device)
            for I in I_list
        ]
        # Cartesian product
        mI_combos = torch.tensor(
            list(itertools.product(*[r.tolist() for r in mI_ranges])),
            dtype=dtype_real,
            device=device,
        )  # (n_nuc_states, n_nuclei)

    # Field direction in molecular frame
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    n0 = torch.tensor(
        [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta],
        dtype=dtype_real,
        device=device,
    )

    # Effective g along field direction
    g_n = g.T @ n0  # (3,)
    g_eff = torch.linalg.norm(g_n)
    u = g_n / g_eff  # unit vector in g·n direction

    # Microwave frequency in MHz
    E0 = exp.mwFreq * 1e3  # GHz → MHz

    # Frequency-to-field conversion factor (MHz → mT)
    # B = E / (g_eff * β), where β = BMAGN in J/T
    pre = 1e9 * PLANCK / (g_eff * BMAGN)  # MHz → mT

    # Transition intensity prefactor (for perpendicular mode)
    # I ∝ |<i|μ_perp|j>|² ∝ c² * (Tr[g·g^T] - |g·u|²)
    # where c² = (β/2)² * [S(S+1) - m_S(m_S-1)]
    # Generate mS values for each transition: mS = S, S-1, ..., -S+1
    mS_transitions = torch.arange(S, -S, -1, dtype=dtype_real, device=device)  # n_transitions values
    c_sq = (BMAGN / 2) ** 2 * (S * (S + 1) - mS_transitions * (mS_transitions - 1))
    c_sq = c_sq / (PLANCK * 1e9) ** 2  # normalize

    gg = g @ g.T
    tr_gg = gg.trace()
    g_u_norm_sq = torch.linalg.norm(g @ u) ** 2
    intensity_factor = c_sq * (tr_gg - g_u_norm_sq) / 2  # average over chi
    # × dBdE = (h/μB·1e9)/g_eff (mT/MHz), as in EasySpin resfields_perturb.m
    intensity_factor = intensity_factor * pre

    # Compute resonance fields for each transition
    B_res_list = []
    Int_list = []

    # Precompute ZFS-related quantities if needed
    if high_spin and D is not None:
        u_D_u = u @ D @ u
        Du = D @ u
        u_D_Du = u @ D @ Du
        tr_DD = (D @ D).trace()
        # Iwasaki D invariants
        D1_sq = tr_DD - u_D_u ** 2
        D2_sq = 2 * tr_DD + u_D_u ** 2 - 4 * u_D_Du

    # Precompute hyperfine quantities
    k_list = []  # unit vectors k for each nucleus
    nK_list = []  # |A·u| for each nucleus
    invA_list = []
    tr_AA_list = []
    det_A_list = []

    if n_nuclei > 0:
        for iNuc in range(n_nuclei):
            A_mat = A_list[iNuc]
            K = A_mat @ u  # (3,)
            nK = torch.linalg.norm(K)
            k = K / (nK + 1e-30)  # avoid division by zero
            nK_list.append(nK)
            k_list.append(k)

            if order >= 2:
                invA_list.append(torch.linalg.inv(A_mat))
                tr_AA_list.append((A_mat @ A_mat).trace())
                det_A_list.append(torch.linalg.det(A_mat))

    # Loop over electron spin transitions (mS → mS-1)
    for idx_mS in range(n_transitions):
        mS = S - idx_mS  # S, S-1, ..., -S+1

        # First-order ZFS correction
        if high_spin and D is not None:
            E1D = -u_D_u / 2 * (3 - 6 * mS)
        else:
            E1D = 0.0

        # First-order hyperfine correction (depends on mI)
        if n_nuclei > 0:
            # E1A[i] = mI[combos[i], nuc] * nK[nuc] summed over nuclei
            E1A = torch.zeros(
                mI_combos.shape[0], dtype=dtype_real, device=device
            ) # (n_nuc_states,)
            for iNuc in range(n_nuclei):
                E1A += mI_combos[:, iNuc] * nK_list[iNuc]
        else:
            E1A = torch.tensor([0.0], dtype=dtype_real, device=device)

        # Second-order corrections
        if order >= 2:
            # ZFS second-order
            if high_spin and D is not None:
                x = D1_sq * (
                    4 * S * (S + 1) - 3 * (8 * mS ** 2 - 8 * mS + 3)
                ) - D2_sq / 4 * (2 * S * (S + 1) - 3 * (2 * mS ** 2 - 2 * mS + 1))
                E2D = -x / (2 * E0)
            else:
                E2D = 0.0

            # Hyperfine second-order
            if n_nuclei > 0:
                E2A = torch.zeros(mI_combos.shape[0], dtype=dtype_real, device=device)
                E2DA = torch.zeros(mI_combos.shape[0], dtype=dtype_real, device=device)

                for iNuc in range(n_nuclei):
                    k_vec = k_list[iNuc]
                    A_mat = A_list[iNuc]
                    mI_vals = mI_combos[:, iNuc]  # (n_nuc_states,)
                    I_nuc = I_list[iNuc]

                    # A·k and related quantities
                    Ak = A_mat.T @ k_vec  # (3,)
                    k_A_u = Ak @ u
                    k_AA_k = torch.linalg.norm(Ak) ** 2

                    # Iwasaki A invariants
                    A1_sq = k_AA_k - k_A_u ** 2
                    A2 = det_A_list[iNuc] * (u @ invA_list[iNuc] @ k_vec)
                    A3 = (
                        tr_AA_list[iNuc]
                        - torch.linalg.norm(A_mat @ u) ** 2
                        - k_AA_k
                        + k_A_u ** 2
                    )

                    II1 = I_nuc * (I_nuc + 1)
                    x = (
                        A1_sq * mI_vals ** 2
                        - A2 * (1 - 2 * mS) * mI_vals
                        + A3 / 2 * (II1 - mI_vals ** 2)
                    )
                    E2A += x / (2 * E0)

                    # ZFS-hyperfine cross term
                    if high_spin and D is not None:
                        DA = Du @ Ak - u_D_u * k_A_u
                        y = DA * (3 - 6 * mS) * mI_vals
                        E2DA -= y / E0
            else:
                E2A = torch.tensor([0.0], dtype=dtype_real, device=device)
                E2DA = torch.tensor([0.0], dtype=dtype_real, device=device)
        else:
            E2D = 0.0
            E2A = 0.0 if n_nuclei == 0 else torch.zeros(mI_combos.shape[0], dtype=dtype_real, device=device)
            E2DA = 0.0 if n_nuclei == 0 else torch.zeros(mI_combos.shape[0], dtype=dtype_real, device=device)

        # Total energy correction and resonance field
        # B_res = (E0 - E1D - E2D - E1A - E2A - E2DA) * pre
        E_total = E0 - E1D - E2D
        if isinstance(E1A, torch.Tensor):
            B_res = (E_total - E1A - E2A - E2DA) * pre  # (n_nuc_states,)
        else:
            B_res = torch.tensor([E_total * pre], dtype=dtype_real, device=device)

        # Filter by field range
        in_range = (B_res >= exp.Range[0]) & (B_res <= exp.Range[1])
        B_res_valid = B_res[in_range]

        # Intensity is same for all nuclear transitions of this mS transition
        intensity = intensity_factor[idx_mS].item()
        if n_nuclei > 0:
            # 1/nNucSublevels per hyperfine line (EasySpin resfields_perturb.m)
            intensity = intensity / mI_combos.shape[0]
        intensities = torch.full_like(B_res_valid, intensity)

        # Apply threshold
        if intensity >= opt.Threshold:
            B_res_list.append(B_res_valid)
            Int_list.append(intensities)

    # Concatenate results
    if len(B_res_list) > 0:
        B_res_all = torch.cat(B_res_list)
        Int_all = torch.cat(Int_list)
    else:
        B_res_all = torch.tensor([], dtype=dtype_real, device=device)
        Int_all = torch.tensor([], dtype=dtype_real, device=device)

    return B_res_all, Int_all
