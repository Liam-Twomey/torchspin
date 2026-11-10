"""Strain width calculation for orientation-dependent line broadening.

Implements the EasySpin strain algorithm (resfields.m lines 715-1320).
Strain models Gaussian distributions of spin Hamiltonian parameters, resulting
in transition-specific, orientation-dependent Gaussian FWHMs.

References
----------
EasySpin ``resfields.m``, ``resfields_perturb.m``, ``getdstrainops.m``
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torchspin._linalg import eigh as _eigh


from torchspin.rotations import erot
from torchspin.spinsystem import SpinSystem


def compute_strain_widths(
    sys: SpinSystem,
    H0: torch.Tensor,        # field-independent Hamiltonian
    mux: torch.Tensor,       # x magnetic moment operator
    muy: torch.Tensor,       # y magnetic moment operator
    muz: torch.Tensor,       # z magnetic moment operator  
    phi: float,              # orientation azimuthal angle (rad)
    theta: float,            # orientation polar angle (rad)
    transitions: torch.Tensor,  # [nTrans, 2] level pairs (u, v) with u < v
    mwFreq: float,           # microwave frequency (GHz)
    B0: float,               # resonance field (mT) for this orientation
    dtype: torch.dtype = torch.complex128,
) -> Optional[torch.Tensor]:
    """
    Compute orientation-dependent Gaussian FWHM (mT) for each transition.

    This function implements the strain width algorithm from EasySpin's
    ``resfields.m`` (lines 1283-1314). Each strain type contributes to the
    total linewidth-squared, which is then converted from MHz^2 to mT using
    the ``dBdE`` conversion factor (inverse slope of the Zeeman term).

    Parameters
    ----------
    sys:
        Spin system with optional strain fields (HStrain, gStrain, DStrain, AStrain).
    H0:
        Field-independent Hamiltonian (MHz), shape ``[nStates, nStates]``.
    mux, muy, muz:
        Magnetic moment operators (MHz/mT), shape ``[nStates, nStates]``.
    phi:
        Azimuthal angle of B0 direction in molecular frame (radians).
    theta:
        Polar angle of B0 direction in molecular frame (radians).
    transitions:
        Transition level pairs ``[u, v]`` where ``u < v``, shape ``[nTrans, 2]``.
    mwFreq:
        Microwave frequency in GHz (used for gStrain conversion).
    B0:
        Resonance field  value in mT (used for dBdE estimate if needed).
    dtype:
        Data type for complex  arithmetic.

    Returns
    -------
    widths:
        Gaussian FWHM (mT) for each transition, shape ``[nTrans]``. Returns
        ``None`` if no strain parameters are present.

    Notes
    -----
    Total line width squared (MHz^2) is computed as:

    .. code-block:: python

        LineWidth2 = HStrain_contribution
                   + gStrain_contribution
                   + AStrain_contribution
                   + DStrain_contribution

    Then converted to field domain:

    .. code-block:: python

        width_mT = dBdE * sqrt(LineWidth2)

    where ``dBdE = 1 / |<v|muz|v> - <u|muz|u>|`` is the field-to-energy
    conversion factor for this transition.
    """
    # Check if any strain is present
    has_strain = _has_any_strain(sys)
    if not has_strain:
        return None

    device = H0.device
    transitions = transitions.to(device=device)
    nTrans = transitions.shape[0]
    
    # Compute orientation vector in molecular frame
    zLab_M = _orientation_vector(phi, theta, device=device)

    # Compute muzL: projection of total magnetic moment onto lab-z (field direction)
    sin_th = math.sin(theta)
    cos_th = math.cos(theta)
    cos_ph = math.cos(phi)
    sin_ph = math.sin(phi)
    muzL = sin_th * cos_ph * mux + sin_th * sin_ph * muy + cos_th * muz

    # Diagonalize Hamiltonian at resonance field to get eigenvectors
    H_total = H0 - B0 * muzL  # Total Hamiltonian at B0 for this orientation
    energies, vectors = torch.linalg.eigh(H_total)
    # Sort by energy (should already be sorted from eigh)
    idx = torch.argsort(energies.real)
    energies = energies[idx]
    vectors = vectors[:, idx]

    # Pre-compute transition matrix elements helper
    def transition_element(Op: torch.Tensor, iTrans: int) -> torch.complex128:
        """
        Compute <v|Op|v> - <u|Op|u> for transition iTrans.
        
        Equivalent to MATLAB: m = @(Op) real((V'-U')*Op*(V+U))
        where U = vectors[:, u], V = vectors[:, v].
        """
        u_idx = transitions[iTrans, 0].long()
        v_idx = transitions[iTrans, 1].long()
        U = vectors[:, u_idx]
        V = vectors[:, v_idx]
        # <v|Op|v> - <u|Op|u> = (V'*Op*V - U'*Op*U)
        # Simplification: (V'-U')*Op*(V+U) equivalent for real Op
        # NOTE: Must use manual computation because torch's @ operator
        # doesn't properly handle complex conjugation for 1D tensors
        temp = Op @ (V + U)
        val = torch.sum(torch.conj(V - U) * temp)
        return val

    # Allocate output
    widths_squared = torch.zeros(nTrans, dtype=torch.float64, device=device)

    # --- 1. HStrain contribution ---
    if sys.HStrain is not None:
        lw2_H = _compute_hstrain_width_squared(sys, zLab_M)
        widths_squared += lw2_H

    # --- 2. gStrain + AStrain contribution (combined) ---
    if sys.gStrain is not None or sys.AStrain is not None:
        lw2_gA = _compute_gstrain_astrain_width_squared(
            sys, zLab_M, vectors, transitions, mwFreq
        )
        widths_squared += lw2_gA

    # --- 3. DStrain contribution ---
    if sys.DStrain is not None:
        lw2_D = _compute_dstrain_width_squared(
            sys, zLab_M, vectors, transitions, transition_element
        )
        widths_squared += lw2_D

    # --- 4. Convert from MHz^2 to mT ---
    # dBdE = 1 / |<v|muzL|v> - <u|muzL|u>| converts frequency -> field
    # muzL is the Hamiltonian derivative w.r.t. B for this orientation.
    # Take real part of the diagonal element (cross terms are purely imaginary).
    dBdE = torch.zeros(nTrans, dtype=torch.float64, device=device)
    for iTrans in range(nTrans):
        muzL_diff = transition_element(muzL, iTrans)
        muzL_diff_abs = abs(torch.real(muzL_diff).item())
        if muzL_diff_abs > 1e-12:  # Avoid division by zero for forbidden transitions
            dBdE[iTrans] = 1.0 / muzL_diff_abs
        else:
            # Forbidden transition or near-zero slope: set large dBdE (small width)
            dBdE[iTrans] = 0.0  # Will result in zero width

    widths_mT = dBdE * torch.sqrt(widths_squared)

    return widths_mT


# ---------------------------------------------------------------------------
# Batched strain width computation (vectorized over all orientations)
# ---------------------------------------------------------------------------

def compute_strain_widths_batch(
    sys: SpinSystem,
    H0: torch.Tensor,           # (n, n)  field-independent Hamiltonian
    muzL_batch: torch.Tensor,   # (M, n, n)  per-orientation muzL
    phi_batch: torch.Tensor,    # (M,)  azimuthal angles (rad)
    theta_batch: torch.Tensor,  # (M,)  polar angles (rad)
    B0_batch: torch.Tensor,     # (M,)  representative resonance field per orientation (mT)
    pairs_list: Optional[list], # M items, each (nTrans_m, 2) int tensor or None
    mwFreq: float,              # GHz
    pairs_flat: Optional[torch.Tensor] = None,   # (M, 2): exactly one transition per orientation
    ho_batch: Optional[tuple] = None,            # (Q2, Q3), each (M, n, n) or None: Options.QuadraticZeeman
) -> list:
    """Vectorized strain widths for a batch of M orientations.

    Replaces M individual calls to :func:`compute_strain_widths` with one
    batched ``torch.linalg.eigh`` call, giving a large speed-up on
    orientation-dense grids (Ci, C2h) with HStrain / gStrain.

    Handles HStrain, gStrain (single- and multi-electron), gStrain+AStrain and
    DStrain (per electron, tilted D frames, D/E correlation) in vectorised form.

    With ``pairs_flat`` (one transition per orientation, the resfields_batch
    call pattern) the widths are returned as a single ``(M,)`` tensor.

    With ``ho_batch = (Q2, Q3)`` (``Options.QuadraticZeeman``) the Hamiltonian at
    B0 is H0 − B0 muzL + B0² Q2 + B0³ Q3 and dB/dE uses the slope operator
    muzL − 2 B0 Q2 − 3 B0² Q3.

    Returns
    -------
    list of M tensors (nTrans_m,) in mT, or None if orientation had no resonances.
    """
    if not _has_any_strain(sys):
        return torch.zeros(pairs_flat.shape[0], dtype=torch.float64, device=H0.device) if pairs_flat is not None else [None] * len(pairs_list)

    device = H0.device
    M = phi_batch.shape[0]
    ph = phi_batch.to(device=device, dtype=torch.float64)
    th = theta_batch.to(device=device, dtype=torch.float64)

    # --- Orientation unit vectors: (M, 3) ---
    sin_th = torch.sin(th);  cos_th = torch.cos(th)
    sin_ph = torch.sin(ph);  cos_ph = torch.cos(ph)
    zLab_M = torch.stack([sin_th * cos_ph, sin_th * sin_ph, cos_th], dim=1)

    # --- Batched eigh at B0 for all M orientations ---
    H_b = H0.unsqueeze(0) - B0_batch[:, None, None] * muzL_batch  # (M, n, n)
    slope_batch = muzL_batch
    if ho_batch is not None:
        Bb = B0_batch.to(torch.float64)[:, None, None]
        Q2, Q3 = ho_batch
        if Q2 is not None:
            H_b = H_b + Bb ** 2 * Q2
            slope_batch = slope_batch - 2.0 * Bb * Q2
        if Q3 is not None:
            H_b = H_b + Bb ** 3 * Q3
            slope_batch = slope_batch - 3.0 * Bb ** 2 * Q3
    _, V_b = _eigh(H_b)  # (M, n, n)

    # --- muzL diagonal in eigenbasis: (M, n) ---
    # diag_k(V^† muzL V) = (V.conj() * (muzL @ V)).sum(dim=-2)
    # (with QuadraticZeeman: the slope operator -dH/dB in place of muzL)
    tmp = slope_batch @ V_b                             # (M, n, n)
    muzL_diag = torch.real((V_b.conj() * tmp).sum(dim=1))  # (M, n)

    # --- Precompute orientation-independent strain matrices ---
    # HStrain: (3,)  →  lw2_H[m] = sum(H^2 * z_m^2)
    lw2_H_vec = torch.zeros(M, dtype=torch.float64, device=device)
    if sys.HStrain is not None:
        H2 = sys.HStrain.to(device=device, dtype=torch.float64) ** 2        # (3,)
        lw2_H_vec = (H2.unsqueeze(0) * zLab_M ** 2).sum(dim=1)   # (M,)

    # gStrain: per-electron gStrainMatrix (3, 3) in molecular frame
    gmats = _gstrain_matrices(sys, mwFreq, device) if sys.gStrain is not None else \
        [torch.zeros(3, 3, dtype=torch.float64, device=device)] * sys.nElectrons
    gStrainMatrix = gmats[0]
    multi_g = sys.gStrain is not None and sys.nElectrons > 1
    if multi_g and sys.AStrain is not None and bool(torch.any(sys.AStrain != 0)):
        raise ValueError("AStrain is not supported in spin systems with more than one electron spin.")

    # lw2_g[m] = z_m^T * (G^2) * z_m   — vectorised over M (single-electron case)
    gStrainMatrix2 = gStrainMatrix ** 2                 # element-wise square
    tmp_g  = zLab_M @ gStrainMatrix2                    # (M, 3)
    lw2_g_vec = (tmp_g * zLab_M).sum(dim=1)             # (M,)

    # Multi-electron g strain: per-electron z_m^T G_e^2 z_m (M,) and the
    # lab-z spin projection diagonals |<k|kSzL_e|k>| in the eigenbasis (M, n)
    gterm_e: list = []
    kSzL_diag_e: list = []
    if multi_g:
        for e in range(sys.nElectrons):
            tmp_e = zLab_M @ (gmats[e] ** 2)
            gterm_e.append((tmp_e * zLab_M).sum(dim=1))                     # (M,)
            Sx, Sy, Sz = _electron_spin_ops(sys, e, device, V_b.dtype)
            zc = zLab_M.to(dtype=V_b.dtype)
            kSzL_b = (zc[:, 0, None, None] * Sx + zc[:, 1, None, None] * Sy
                      + zc[:, 2, None, None] * Sz)                          # (M, n, n)
            kSzL_diag_e.append(torch.real((V_b.conj() * (kSzL_b @ V_b)).sum(dim=1)))  # (M, n)

    # AStrain: build AStrainMatrix (3, 3) in molecular frame; compute mI_eff batch
    AStrainMatrix: Optional[torch.Tensor] = None
    mI_eff_batch: Optional[torch.Tensor] = None
    if sys.AStrain is not None and sys.nNuclei > 0 and sys.nElectrons == 1:
        from torchspin.spinops import sop
        AStrainMatrix = torch.diag(
            sys.AStrain.to(device=device, dtype=torch.float64)
        )
        if sys.AFrame is not None and torch.any(sys.AFrame[0] != 0):
            R_A2M = erot(sys.AFrame[0].tolist(), device=device).T.to(dtype=torch.float64)
            AStrainMatrix = R_A2M @ AStrainMatrix @ R_A2M.T
        mIz_op = sop(sys.Spins, [[2, 3]], device=device).to(dtype=H0.dtype)
        tmp_mi = mIz_op.unsqueeze(0) @ V_b              # (M, n, n)
        mI_eff_batch = torch.real((V_b.conj() * tmp_mi).sum(dim=1))  # (M, n)

    # DStrain: per-electron dH/dD and dH/dE operators (EasySpin getdstrainops;
    # DFrame and DStrainCorr included), diagonals in the eigenbasis: (M, n)
    dstrain_diags: list = []
    if sys.DStrain is not None:
        V_c = V_b.to(torch.complex128)
        for op_pair in _dstrain_operators(sys, device, torch.complex128):
            if op_pair is None:
                continue
            dHdD_op, dHdE_op = op_pair
            dD_diag = torch.real((V_c.conj() * (dHdD_op.unsqueeze(0) @ V_c)).sum(dim=1))
            dE_diag = torch.real((V_c.conj() * (dHdE_op.unsqueeze(0) @ V_c)).sum(dim=1))
            dstrain_diags.append((dD_diag, dE_diag))

    # --- Build per-orientation results (flat over all transitions) ---
    corr = getattr(sys, 'gAStrainCorr', 1.0)
    if pairs_flat is not None:
        counts = None
        pairs_all = pairs_flat.to(device=device)
        m_of = torch.arange(M, device=device)
    else:
        counts = [0 if (p is None or p.numel() == 0) else int(p.shape[0]) for p in pairs_list]
        if sum(counts) == 0:
            return [None] * M
        pairs_all = torch.cat([p.reshape(-1, 2) for p, c in zip(pairs_list, counts) if c > 0], dim=0).to(device=device)
        m_of = torch.repeat_interleave(torch.arange(M, device=device), torch.tensor(counts, device=device))
    u_idx = pairs_all[:, 0].long()
    v_idx = pairs_all[:, 1].long()

    # HStrain (transition-independent)
    lw2 = lw2_H_vec[m_of].clone()

    # gStrain (+AStrain combined)
    if multi_g:
        for e in range(sys.nElectrons):
            w_e = torch.abs(kSzL_diag_e[e][m_of, v_idx] - kSzL_diag_e[e][m_of, u_idx])
            lw2 += w_e * gterm_e[e][m_of]
    elif sys.gStrain is not None and AStrainMatrix is None:
        lw2 += lw2_g_vec[m_of]
    elif AStrainMatrix is not None:
        # SM = G (if gStrain) + corr * mI_avg * A, lw2 += z^T (SM ∘ SM) z per transition
        mI_avg = 0.5 * (mI_eff_batch[m_of, u_idx] + mI_eff_batch[m_of, v_idx])          # (K,)
        G = gStrainMatrix if sys.gStrain is not None else torch.zeros_like(AStrainMatrix)
        SM = G.unsqueeze(0) + (corr * mI_avg).view(-1, 1, 1) * AStrainMatrix.unsqueeze(0)  # (K, 3, 3)
        z = zLab_M[m_of]                                                                   # (K, 3)
        lw2 += torch.einsum('ki,kij,kj->k', z, SM ** 2, z)

    # DStrain (sum over electrons)
    for dD_diag, dE_diag in dstrain_diags:
        mD = dD_diag[m_of, v_idx] - dD_diag[m_of, u_idx]
        mE = dE_diag[m_of, v_idx] - dE_diag[m_of, u_idx]
        lw2 += mD ** 2 + mE ** 2

    # dBdE: |<v|muzL|v> - <u|muzL|u>|^{-1}  (already computed)
    muzL_diff = muzL_diag[m_of, v_idx] - muzL_diag[m_of, u_idx]
    dBdE = torch.where(torch.abs(muzL_diff) > 1e-12, 1.0 / torch.abs(muzL_diff), torch.zeros_like(muzL_diff))
    widths = dBdE * torch.sqrt(torch.clamp(lw2, min=0.0))
    if counts is None:
        return widths
    chunks = torch.split(widths, counts)
    return [None if c == 0 else w for w, c in zip(chunks, counts)]


# ---------------------------------------------------------------------------
# Helper functions for each strain type
# ---------------------------------------------------------------------------

def _has_any_strain(sys: SpinSystem) -> bool:
    """Check if any strain parameter is present."""
    return (
        (sys.HStrain is not None and torch.any(sys.HStrain > 0))
        or (sys.gStrain is not None and torch.any(sys.gStrain > 0))
        or (sys.AStrain is not None and torch.any(sys.AStrain > 0))
        or (sys.DStrain is not None and torch.any(sys.DStrain > 0))
    )


def _orientation_vector(
    phi: float,
    theta: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Compute unit vector in molecular frame given (phi, theta).
    
    Returns zLab_M: [3] tensor (x, y, z components in molecular frame).
    """
    theta_t = torch.as_tensor(theta, dtype=torch.float64, device=device)
    phi_t = torch.as_tensor(phi, dtype=torch.float64, device=device)
    sin_theta = torch.sin(theta_t)
    cos_theta = torch.cos(theta_t)
    sin_phi = torch.sin(phi_t)
    cos_phi = torch.cos(phi_t)
    
    # Spherical -> Cartesian
    x = sin_theta * cos_phi
    y = sin_theta * sin_phi
    z = cos_theta
    
    return torch.stack([x, y, z])



def _gstrain_matrices(sys: SpinSystem, mwFreq: float, device) -> list:
    """Per-electron g-strain matrices in the molecular frame (EasySpin ``gStrainMatrix{iEl}``, MHz).

    ``diag(gStrain(iEl,:)./g(iEl,:))*mwFreq`` rotated from the g frame to the
    molecular frame with ``R_g2M = erot(gFrame).'``.  Electrons without strain
    get a zero matrix.
    """
    mats = []
    for e in range(sys.nElectrons):
        M = torch.zeros(3, 3, dtype=torch.float64, device=device)
        if sys.gStrain is not None and bool(torch.any(sys.gStrain[e] != 0)):
            if sys.fullg:
                raise ValueError("gStrain and AStrain are not supported when full g matrices are given!")
            g_strain = sys.gStrain[e].to(device=device, dtype=torch.float64)
            g_values = sys.g[e].to(device=device, dtype=torch.float64)
            M = torch.diag((g_strain / g_values) * (mwFreq * 1e3))
            if bool(torch.any(sys.gFrame[e] != 0)):
                R_g2M = erot(sys.gFrame[e].tolist(), device=device).T.to(dtype=torch.float64)
                M = R_g2M @ M @ R_g2M.T
        mats.append(M)
    return mats


def _electron_spin_ops(sys: SpinSystem, e: int, device, dtype):
    """(Sx, Sy, Sz) of electron ``e`` (0-based) in the full spin space."""
    from torchspin.spinops import sop
    Sx = sop(sys.Spins, [[e + 1, 1]], device=device).to(dtype=dtype)
    Sy = sop(sys.Spins, [[e + 1, 2]], device=device).to(dtype=dtype)
    Sz = sop(sys.Spins, [[e + 1, 3]], device=device).to(dtype=dtype)
    return Sx, Sy, Sz


def _dstrain_operators(sys: SpinSystem, device, dtype) -> list:
    """EasySpin ``getdstrainops``: per-electron ``(dH/dD, dH/dE)`` pre-multiplied by the strain FWHMs.

    Handles a tilted D frame (``DFrame``) and a D/E correlation coefficient
    (``DStrainCorr``).  Electrons without D strain (or with S < 1) get ``None``.
    """
    ops = []
    if sys.DStrain is None:
        return [None] * sys.nElectrons
    for e in range(sys.nElectrons):
        dD = float(sys.DStrain[e, 0])
        dE = float(sys.DStrain[e, 1]) if sys.DStrain.shape[1] > 1 else 0.0
        if (dD == 0 and dE == 0) or sys.S[e] < 1.0:
            ops.append(None)
            continue
        Sx, Sy, Sz = _electron_spin_ops(sys, e, device, dtype)
        if sys.DFrame is not None and bool(torch.any(sys.DFrame[e] != 0)):
            R = erot(sys.DFrame[e].tolist(), device=device).to(dtype=dtype)  # molecular -> D frame
            SxD = R[0, 0] * Sx + R[0, 1] * Sy + R[0, 2] * Sz
            SyD = R[1, 0] * Sx + R[1, 1] * Sy + R[1, 2] * Sz
            SzD = R[2, 0] * Sx + R[2, 1] * Sy + R[2, 2] * Sz
            SxD2, SyD2, SzD2 = SxD @ SxD, SyD @ SyD, SzD @ SzD
        else:
            SxD2, SyD2, SzD2 = Sx @ Sx, Sy @ Sy, Sz @ Sz
        dHdD_ = (2 * SzD2 - SxD2 - SyD2) / 3
        dHdE_ = SxD2 - SyD2
        r = float(sys.DStrainCorr[e]) if isinstance(sys.DStrainCorr, (list, tuple)) else float(sys.DStrainCorr)
        if r != 0:
            R12 = r * dD * dE
            cov = torch.tensor([[dD ** 2, R12], [R12, dE ** 2]], dtype=torch.float64)
            L, V = torch.linalg.eigh(cov)          # ascending, like MATLAB eig for symmetric input
            L = torch.sqrt(torch.clamp(L, min=0.0))
            dHdD = float(L[0]) * (float(V[0, 0]) * dHdD_ + float(V[0, 1]) * dHdE_)
            dHdE = float(L[1]) * (float(V[1, 0]) * dHdD_ + float(V[1, 1]) * dHdE_)
        else:
            dHdD = dD * dHdD_
            dHdE = dE * dHdE_
        ops.append((dHdD, dHdE))
    return ops


def _compute_hstrain_width_squared(
    sys: SpinSystem,
    zLab_M: torch.Tensor,
) -> float:
    """
    HStrain contribution: sum(HStrain^2 .* zLab_M^2).
    
    HStrain is [3] MHz in molecular frame. This is a simple quadratic form.
    
    Returns linewidth^2 in MHz^2.
    """
    HStrain2 = sys.HStrain.to(
        device=zLab_M.device, dtype=torch.float64
    ) ** 2
    lw2 = torch.sum(HStrain2 * zLab_M ** 2).item()
    return lw2


def _compute_gstrain_astrain_width_squared(
    sys: SpinSystem,
    zLab_M: torch.Tensor,
    vectors: torch.Tensor,
    transitions: torch.Tensor,
    mwFreq: float,
) -> torch.Tensor:
    """
    g-strain and A-strain contribution (correlated if AStrain present).
    
    For single-electron systems:
        gStrainMatrix = diag(gStrain ./ g) * mwFreq  (MHz)
        (rotated to molecular frame if gFrame given)
        
        If AStrain present:
            AStrainMatrix = diag(AStrain) (MHz)
            (rotated to molecular frame if AFrame given)
            effective mI for each transition computed
            StrainMatrix = gStrainMatrix + corr * mI * AStrainMatrix
        
        lw2_gA[iTrans] = zLab_M.' * StrainMatrix[iTrans]^2 * zLab_M
    
    For multi-electron systems the per-electron matrices are weighted by
    |<v|kSzL_e|v> - <u|kSzL_e|u>| (EasySpin's non-simple g strain branch).
    
    Returns tensor of shape [nTrans] with MHz^2 contribution for each transition.
    """
    nTrans = transitions.shape[0]
    device = vectors.device
    lw2_gA = torch.zeros(nTrans, dtype=torch.float64, device=device)

    # Multi-electron g strain (EasySpin resfields.m 1299-1309, ~simplegStrain):
    #   gA2 = sum_e |m(kSzL{e})| * gStrainMatrix{e}.^2 ;  lw2 += z' gA2 z
    # where kSzL{e} = zLab_M . (Sx_e, Sy_e, Sz_e) and m(Op) = <v|Op|v> - <u|Op|u>.
    if sys.nElectrons > 1:
        if sys.AStrain is not None and bool(torch.any(sys.AStrain != 0)):
            raise ValueError("AStrain is not supported in spin systems with more than one electron spin.")
        gmats = _gstrain_matrices(sys, mwFreq, device)
        zc = zLab_M.to(dtype=vectors.dtype)
        kSzL = []
        for e in range(sys.nElectrons):
            Sx, Sy, Sz = _electron_spin_ops(sys, e, device, vectors.dtype)
            kSzL.append(zc[0] * Sx + zc[1] * Sy + zc[2] * Sz)
        for iTrans in range(nTrans):
            u_idx = transitions[iTrans, 0].long()
            v_idx = transitions[iTrans, 1].long()
            U = vectors[:, u_idx]
            V = vectors[:, v_idx]
            gA2 = torch.zeros(3, 3, dtype=torch.float64, device=device)
            for e in range(sys.nElectrons):
                m_e = torch.real(torch.sum(torch.conj(V - U) * (kSzL[e] @ (V + U))))
                gA2 = gA2 + abs(float(m_e)) * gmats[e] ** 2
            lw2_gA[iTrans] = (zLab_M @ gA2 @ zLab_M).item()
        return lw2_gA

    # gStrain contribution.
    # MATLAB formula (resfields.m line 787-793):
    #   gStrainMatrix_mol = R_g2M * diag(gStrain/g * mwFreq) * R_g2M'  (in molecular frame)
    #   gAslw2 = (gStrainMatrix_mol + corr * mI * AStrainMatrix_mol).^2  (element-wise square)
    #   lw2 += z' * gAslw2 * z
    # NOTE: element-wise square differs from matrix square (||M@z||^2) for non-diagonal M.
    # For diagonal matrices (no frame rotation) both are identical; for rotated frames they differ.
    if sys.gStrain is not None:
        g_strain = sys.gStrain[0, :].to(device=device, dtype=torch.float64)
        g_values = sys.g[0, :].to(device=device, dtype=torch.float64)
        gStrain_freq = (g_strain / g_values) * (mwFreq * 1e3)  # [3] MHz in g-frame

        # Build gStrain matrix in molecular frame (diagonal in g-frame, then rotate)
        gStrainMatrix = torch.diag(gStrain_freq)  # [3, 3] diagonal in g-frame
        if torch.any(sys.gFrame[0, :] != 0):
            R_g2M = erot(sys.gFrame[0, :].tolist(), device=device).T.to(dtype=torch.float64)  # g→mol
            gStrainMatrix = R_g2M @ gStrainMatrix @ R_g2M.T  # now in molecular frame
    else:
        gStrainMatrix = torch.zeros(3, 3, dtype=torch.float64, device=device)

    # AStrain handling
    if sys.AStrain is not None and sys.nNuclei > 0:
        # AStrain only for first nucleus
        if sys.nElectrons > 1:
            raise ValueError(
                "AStrain not supported for multi-electron systems (EasySpin limitation)."
            )

        # AStrainMatrix in molecular frame (diagonal or rotated)
        AStrainMatrix = torch.diag(
            sys.AStrain.to(device=device, dtype=torch.float64)
        )  # [3, 3] in A-frame

        # Rotate to molecular frame if AFrame given
        if sys.AFrame is not None and torch.any(sys.AFrame[0, :] != 0):
            R_A2M = erot(sys.AFrame[0, :].tolist(), device=device).T.to(dtype=torch.float64)
            AStrainMatrix = R_A2M @ AStrainMatrix @ R_A2M.T

        # Compute effective mI for each transition using eigenvectors at B0
        from torchspin.spinops import sop
        mIz_op = sop(sys.Spins, [[2, 3]], device=device)  # mIz for nucleus 1 (index 2)
        mI_eff = torch.real(torch.diag(torch.conj(vectors).T @ mIz_op @ vectors))

        # For each transition, get average mI
        corr = sys.gAStrainCorr
        for iTrans in range(nTrans):
            u_idx = transitions[iTrans, 0].long()
            v_idx = transitions[iTrans, 1].long()
            mI_avg = 0.5 * (mI_eff[u_idx] + mI_eff[v_idx])

            # Combined strain matrix for this transition (both in molecular frame)
            StrainMatrix_iTrans = gStrainMatrix + corr * mI_avg * AStrainMatrix

            # MATLAB formula: z' * (M .^ 2) * z  (element-wise square, NOT matrix square)
            lw2_gA[iTrans] = (zLab_M @ (StrainMatrix_iTrans ** 2) @ zLab_M).item()
    else:
        # No AStrain: just gStrain.
        # MATLAB formula: z' * (gStrainMatrix_mol .^ 2) * z  (element-wise square)
        lw2_single = (zLab_M @ (gStrainMatrix ** 2) @ zLab_M).item()
        lw2_gA[:] = lw2_single

    return lw2_gA


def _compute_dstrain_width_squared(
    sys: SpinSystem,
    zLab_M: torch.Tensor,
    vectors: torch.Tensor,
    transitions: torch.Tensor,
    transition_element,
) -> torch.Tensor:
    """
    D-strain contribution: sum over electrons of
    ``|<v|dH/dD|v> - <u|dH/dD|u>|^2 + |<v|dH/dE|v> - <u|dH/dE|u>|^2``
    with the operators from EasySpin's ``getdstrainops`` (tilted D frames and
    D/E correlation included; see :func:`_dstrain_operators`).

    Returns tensor of shape [nTrans] with MHz^2 contribution.
    """
    nTrans = transitions.shape[0]
    device = vectors.device
    lw2_D = torch.zeros(nTrans, dtype=torch.float64, device=device)
    for op_pair in _dstrain_operators(sys, device, vectors.dtype):
        if op_pair is None:
            continue
        dHdD, dHdE = op_pair
        for iTrans in range(nTrans):
            mD = transition_element(dHdD, iTrans)
            mE = transition_element(dHdE, iTrans)
            lw2_D[iTrans] += torch.real(mD) ** 2 + torch.real(mE) ** 2
    return lw2_D

    # Compute dH/dD and dH/dE operators
    # From getdstrainops.m logic (simplified for diagonal D in molecular frame)
    from torchspin.spinops import sop

    Sx = sop(sys.Spins, [[1, 1]], device=device)  # Sx for electron 1
    Sy = sop(sys.Spins, [[1, 2]], device=device)  # Sy
    Sz = sop(sys.Spins, [[1, 3]], device=device)  # Sz

    # Stevens operator derivatives (assuming D-frame aligned with molecular frame)
    # dH/dD ~ (3*Sz^2 - S(S+1)*I) / 3
    # dH/dE ~ (Sx^2 - Sy^2)
    I_mat = torch.eye(Sx.shape[0], dtype=Sx.dtype, device=device)
    dHdD = (3.0 * Sz @ Sz - S * (S + 1) * I_mat) / 3.0
    dHdE = Sx @ Sx - Sy @ Sy

    # Pre-multiply by strain FWHMs
    DeltaD = sys.DStrain[0].item()  # FWHM_D
    DeltaE = sys.DStrain[1].item() if sys.DStrain.shape[0] > 1 else 0.0  # FWHM_E

    dHdD = DeltaD * dHdD
    dHdE = DeltaE * dHdE

    # For each transition, compute |<v|dHdD|v> - <u|dHdD|u>|^2
    for iTrans in range(nTrans):
        mD = transition_element(dHdD, iTrans)
        mE = transition_element(dHdE, iTrans)
        lw2_D[iTrans] = torch.real(mD) ** 2 + torch.real(mE) ** 2

    return lw2_D


def compute_strain_widths_freq_batch(
    sys: SpinSystem,
    H0: torch.Tensor,           # (n, n)
    muzL_batch: torch.Tensor,   # (M, n, n)
    phi_batch: torch.Tensor,    # (M,)
    theta_batch: torch.Tensor,  # (M,)
    B0: float,                  # static field (mT), the same for all orientations
    pairs_list: list,           # M items, each (nTrans_m, 2) int tensor or None
    ho_batch: Optional[tuple] = None,   # (Q2, Q3), each (M, n, n) or None: Options.QuadraticZeeman
) -> list:
    """Frequency-domain strain widths (MHz) for frequency-swept spectra.

    Port of the strain block of EasySpin ``resfreqs_matrix.m``: with
    ``m(op) = <v|op|v> - <u|op|u>`` the squared width is

        HStrain.^2·zL.^2 + Σ_e |m(dHdD_e)|² + |m(dHdE_e)|²
        + |m(dHdAx)|² + |m(dHdAy)|² + |m(dHdAz)|²
        + (m(μzL)·B0·zLᵀ·G·zL)²,   G = diag(gStrain./g) in the molecular frame,

    without the field conversion factor (no dBdE).  Returns a list of M
    tensors ``(nTrans_m,)`` in MHz (None where the orientation has no pairs).
    """
    if not _has_any_strain(sys):
        return [None] * len(pairs_list)
    device = H0.device
    M = phi_batch.shape[0]
    ph = phi_batch.to(device=device, dtype=torch.float64)
    th = theta_batch.to(device=device, dtype=torch.float64)
    zLab_M = torch.stack([torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)], dim=1)
    H_b = H0.unsqueeze(0) - float(B0) * muzL_batch
    if ho_batch is not None:
        # QuadraticZeeman: eigenvectors of the full H(B0); the g-strain term below
        # keeps the linear moment (g strain perturbs the linear Zeeman term only)
        if ho_batch[0] is not None:
            H_b = H_b + float(B0) ** 2 * ho_batch[0]
        if ho_batch[1] is not None:
            H_b = H_b + float(B0) ** 3 * ho_batch[1]
    _, V_b = _eigh(H_b)
    V_c = V_b.to(torch.complex128)

    def diag_in_eigbasis(op):
        return torch.real((V_c.conj() * (op.to(torch.complex128).unsqueeze(0) @ V_c)).sum(dim=1))  # (M, n)

    muzL_diag = torch.real((V_b.conj() * (muzL_batch @ V_b)).sum(dim=1))
    lw2_H = torch.zeros(M, dtype=torch.float64, device=device)
    if sys.HStrain is not None:
        lw2_H = ((sys.HStrain.to(device=device, dtype=torch.float64) ** 2).unsqueeze(0) * zLab_M ** 2).sum(dim=1)
    # g strain: dimensionless G = diag(gStrain/g) (rotated); only the first electron, as EasySpin
    zGz = None
    if sys.gStrain is not None and bool(torch.any(sys.gStrain != 0)):
        G = _gstrain_matrices(sys, 1e-3, device)[0]          # mwFreq=1e-3 GHz → factor 1 MHz
        zGz = ((zLab_M @ G) * zLab_M).sum(dim=1)              # (M,)
    # A strain: separate x, y, z terms, first electron / first nucleus
    a_diags: list = []
    if sys.AStrain is not None and sys.nNuclei > 0 and bool(torch.any(sys.AStrain != 0)):
        from torchspin.spinops import sop
        n_e = sys.nElectrons
        for c in (1, 2, 3):
            S_op = sop(sys.Spins, [[1, c]], device=device).to(torch.complex128)
            I_op = sop(sys.Spins, [[n_e + 1, c]], device=device).to(torch.complex128)
            a_diags.append(diag_in_eigbasis(float(sys.AStrain[c - 1]) * (I_op @ S_op)))
    d_diags: list = []
    if sys.DStrain is not None:
        for op_pair in _dstrain_operators(sys, device, torch.complex128):
            if op_pair is None:
                continue
            d_diags.append((diag_in_eigbasis(op_pair[0]), diag_in_eigbasis(op_pair[1])))
    results: list = []
    for m in range(M):
        pairs_m = pairs_list[m]
        if pairs_m is None or pairs_m.numel() == 0:
            results.append(None)
            continue
        u_idx = pairs_m[:, 0].long(); v_idx = pairs_m[:, 1].long()
        lw2 = torch.zeros(pairs_m.shape[0], dtype=torch.float64, device=device) + lw2_H[m]
        for dD, dE in d_diags:
            lw2 += (dD[m, v_idx] - dD[m, u_idx]) ** 2 + (dE[m, v_idx] - dE[m, u_idx]) ** 2
        for ad in a_diags:
            lw2 += (ad[m, v_idx] - ad[m, u_idx]) ** 2
        if zGz is not None:
            lw2 += ((muzL_diag[m, v_idx] - muzL_diag[m, u_idx]) * float(B0) * zGz[m]) ** 2
        results.append(torch.sqrt(torch.clamp(lw2, min=0.0)))
    return results
