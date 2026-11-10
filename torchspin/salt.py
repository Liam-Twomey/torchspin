"""ENDOR powder spectrum simulator for torchspin.

Port of EasySpin's ``salt.m``.

For each SOPHE powder-grid orientation, finds the EPR resonance field(s)
via ``resfields_batch``, then calls ``endorfrq`` to get the nuclear transition
frequencies.  Accumulates a frequency-domain ENDOR spectrum.

GPU acceleration
----------------
Set ``opt.device = 'cuda'`` to move all Hamiltonians to GPU.  The EPR
resonance-field search (``resfields_batch``) is fully vectorized over
orientations; each ``endorfrq`` call uses the device'd operators for its
eigh diagonalization.  Operators are built once and reused across the loop.

Pipeline::

    ham(sys)           → H0, mux, muy, muz          (built once, on device)
    ham_nz(sys)        → snmux, snmuy, snmuz         (built once, on device)
    sphgrid(...)       → phi, theta, weights
    resfields_batch(…) → B_res[k], intens_epr[k]     (batched over all k)
    for each orientation k:
        for each B_res:
            endorfrq(…, _H0=H0, …)  → freqs, intens  (reuses device operators)
            bin freqs into freq_axis
    convspec(...)       → Gaussian broadening
    return freq_axis, spec
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torchspin._linalg import eigh as _eigh


from torchspin.convspec import convspec
from torchspin.endorfrq import endorfrq
from torchspin.experiment import Experiment, Options
from torchspin.ham import ham
from torchspin.ham_nz import ham_nz
from torchspin.hamsymm import hamsymm
from torchspin.resfields_perturb import resfields_perturb, resfields_perturb_batch
from torchspin.sphgrid import sphgrid
from torchspin.spinsystem import SpinSystem


def _fixed_field_transitions(
    sys: SpinSystem,
    ops: tuple,
    phi_arr: torch.Tensor,
    theta_arr: torch.Tensor,
    field_mT: float,
    orisel_w: torch.Tensor,
    f_lo: float,
    f_hi: float,
    threshold: float,
    use_perturb: bool,
    enhancement: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ENDOR transition positions/intensities on the orientation grid.

    Returns ``(Pdat, Idat)`` as ``(n_trans, n_orient)`` tensors on the autograd
    graph when the spin system carries tensors that require grad — the
    layout of EasySpin's ``endorfrq`` output that ``salt.m`` feeds into grid
    interpolation and triangle/zone projection.  Each row is one transition
    followed smoothly across orientations: a ``(nucleus, mS)`` pair for the
    first-order perturbation path, an energy-level pair for the matrix path.
    ``Idat`` already carries the orientation-selection weight ``orisel_w``.
    ``enhancement`` (matrix path only; the caller never combines it with
    ``use_perturb``): intensities from the total moment ``mux/muy/muz``
    (hyperfine enhancement, EasySpin ``Opt.Enhancement``), selection from the
    nuclear-only rates, electronic level pairs excluded.
    """
    H0, mux, muy, muz, snmux, snmuy, snmuz = ops
    device = H0.device
    dt64 = torch.float64
    n_orient = phi_arr.shape[0]
    sin_t, cos_t = torch.sin(theta_arr), torch.cos(theta_arr)
    sin_p, cos_p = torch.sin(phi_arr), torch.cos(phi_arr)
    h_all = torch.stack([sin_t * cos_p, sin_t * sin_p, cos_t], dim=1)  # (M, 3)

    if use_perturb:
        # First-order perturbation theory (EasySpin Opt.Method='perturb1'):
        # freq = |mS * A@u + nuI * h|, unit ENDOR intensity.
        from torchspin.rotations import erot as _erot
        from torchspin.constants import NMAGN as _NMAGN, PLANCK as _PLANCK
        if sys.fullg:
            g_mat = sys.g[:3, :].to(dtype=dt64, device=device)
        else:
            g_diag = torch.diag(sys.g[0, :].to(dtype=dt64, device=device))
            Rg = torch.as_tensor(_erot(sys.gFrame[0, :].tolist()).T,
                                 dtype=dt64, device=device)
            g_mat = Rg @ g_diag @ Rg.T
        gn_all = h_all @ g_mat
        u_all = gn_all / gn_all.norm(dim=1, keepdim=True).clamp(min=1e-30)
        S_val = float(sys.S[0])
        mS_vals = torch.arange(S_val, -S_val - 0.5, -1.0, dtype=dt64, device=device)
        rows = []
        for iNuc in range(sys.nNuclei):
            gn_i = float(sys.gn[iNuc])
            nuI = -gn_i * _NMAGN * (field_mT * 1e-3) / _PLANCK * 1e-6  # MHz
            if sys.fullA:
                A_i = sys.A[3 * iNuc: 3 * iNuc + 3, :].to(dtype=dt64, device=device)
            else:
                A_diag = torch.diag(sys.A[iNuc, :].to(dtype=dt64, device=device))
                R_A = torch.as_tensor(_erot(sys.AFrame[iNuc, :].tolist()).T,
                                      dtype=dt64, device=device)
                A_i = R_A @ A_diag @ R_A.T
            Ah = u_all @ A_i.T                                     # (M, 3)
            for mS_val in mS_vals.tolist():
                K = mS_val * Ah + nuI * h_all
                rows.append(K.norm(dim=1))
        Pdat = torch.stack(rows, dim=0)                            # (T, M)
        Idat = orisel_w.to(dt64).unsqueeze(0).expand_as(Pdat).clone()
    else:
        # Matrix diagonalization: every level pair is a transition slot.
        dim = H0.shape[0]
        iu = torch.triu_indices(dim, dim, offset=1, device=device)
        pi_all, pj_all = iu[0], iu[1]
        if enhancement:
            same = _same_nuclear_block(sys, pi_all, pj_all)
            pi_all, pj_all = pi_all[same], pj_all[same]
        n_pairs = pi_all.numel()
        bytes_per_entry = dim * dim * dim * 16 * 3
        batch = max(1, min(512, (100 * 1024 * 1024) // bytes_per_entry))
        P_rows = torch.empty((n_orient, n_pairs), dtype=dt64, device=device)
        I_rows = torch.empty((n_orient, n_pairs), dtype=dt64, device=device)
        if enhancement:
            S_rows = torch.empty((n_orient, n_pairs), dtype=dt64, device=device)
        for s in range(0, n_orient, batch):
            e = min(s + batch, n_orient)
            hz = h_all[s:e]
            nx = torch.stack([cos_t[s:e] * cos_p[s:e], cos_t[s:e] * sin_p[s:e], -sin_t[s:e]], dim=1)
            ny = torch.stack([-sin_p[s:e], cos_p[s:e], torch.zeros_like(cos_p[s:e])], dim=1)
            muzL = (hz[:, 0].view(-1, 1, 1) * mux + hz[:, 1].view(-1, 1, 1) * muy
                    + hz[:, 2].view(-1, 1, 1) * muz)
            snx = (nx[:, 0].view(-1, 1, 1) * snmux + nx[:, 1].view(-1, 1, 1) * snmuy
                   + nx[:, 2].view(-1, 1, 1) * snmuz)
            sny = ny[:, 0].view(-1, 1, 1) * snmux + ny[:, 1].view(-1, 1, 1) * snmuy
            H_B = H0.unsqueeze(0) - field_mT * muzL
            E_b, V_b = _eigh(H_B)
            P_rows[s:e] = (E_b[:, pj_all] - E_b[:, pi_all]).abs()
            Vh = V_b.conj().transpose(1, 2)
            mx = torch.bmm(Vh, torch.bmm(snx, V_b))
            my = torch.bmm(Vh, torch.bmm(sny, V_b))
            rate = (mx.abs() ** 2 + my.abs() ** 2) / 2.0
            if enhancement:
                # Total moment perpendicular to the field (hyperfine enhancement)
                ex = (nx[:, 0].view(-1, 1, 1) * mux + nx[:, 1].view(-1, 1, 1) * muy
                      + nx[:, 2].view(-1, 1, 1) * muz)
                ey = ny[:, 0].view(-1, 1, 1) * mux + ny[:, 1].view(-1, 1, 1) * muy
                mex = torch.bmm(Vh, torch.bmm(ex, V_b))
                mey = torch.bmm(Vh, torch.bmm(ey, V_b))
                S_rows[s:e] = rate[:, pi_all, pj_all].real.to(dt64).detach()   # selection only
                I_rows[s:e] = ((mex.abs() ** 2 + mey.abs() ** 2) / 2.0)[:, pi_all, pj_all].real.to(dt64)
            else:
                I_rows[s:e] = rate[:, pi_all, pj_all].real.to(dt64)
        Pdat = P_rows.T.contiguous()
        Idat = I_rows.T.contiguous() * orisel_w.to(dt64).unsqueeze(0)
        # EasySpin endorfrq transition selection: drop pairs that are weak
        # everywhere (relative to the strongest) or never enter the window.
        # With enhancement the selection uses the nuclear-only rates (EasySpin).
        sel_rows = S_rows if enhancement else I_rows
        max_all = float(sel_rows.max()) if sel_rows.numel() else 0.0
        strong = sel_rows.max(dim=0).values >= threshold * max_all if max_all > 0 \
            else torch.ones(n_pairs, dtype=torch.bool, device=device)
        inwin = ((Pdat >= f_lo) & (Pdat <= f_hi)).any(dim=1)
        keep = strong & inwin
        Pdat, Idat = Pdat[keep], Idat[keep]

    return Pdat, Idat


def _same_nuclear_block(sys: SpinSystem, pi: torch.Tensor, pj: torch.Tensor) -> torch.Tensor:
    """Mask of level pairs inside one block of ``prod(2I+1)`` energy-sorted levels.

    EasySpin endorfrq's exclusion of electronic transitions (used here only
    with ``Options.Enhancement``, where EPR transitions would otherwise carry
    the full electron moment).
    """
    n_nuc = int(round(math.prod(2.0 * float(i) + 1.0 for i in sys.I)))
    return (pi // n_nuc) == (pj // n_nuc)


def _project_powder(
    Pdat: torch.Tensor,
    Idat: torch.Tensor,
    grid_sym: str,
    N_coarse: int,
    N_interp: int,
    theta_sym: torch.Tensor,
    x_lo: float,
    dx: float,
    n_points: int,
    verbosity: int = 0,
) -> torch.Tensor:
    """Powder average by grid interpolation and triangle/zone projection.

    Port of the disordered-sample branch of EasySpin ``salt.m``: each
    transition slot is (optionally) interpolated from the ``N_coarse`` grid
    to ``(N_coarse-1)*N_interp+1`` knots, then projected onto the frequency
    axis with ``projecttriangles`` (``projectzones`` for axial grids).  As in
    EasySpin, interpolation is switched off when any slot has a missing
    orientation (NaN).  All stages are the torch implementations shared with
    :mod:`torchspin.pepper_autograd`, so the result stays on the autograd
    graph.  Returns a spectral density (per MHz).
    """
    from torchspin.pepper import _facets
    from torchspin.pepper_autograd import _interp_slots_t, projecttriangles_t, projectzones_t

    n_trans, n_orient = Pdat.shape
    dev = Pdat.device
    any_nan = bool(torch.isnan(Pdat.detach()).any())
    do_interp = (N_interp > 1) and (not any_nan) and (n_orient > 1)
    if do_interp:
        nfKnots = (N_coarse - 1) * N_interp + 1
        _, theta_f, _, _ = sphgrid(grid_sym, nfKnots)
        N_final, theta_final = nfKnots, theta_f
        P_f = _interp_slots_t(Pdat, grid_sym, N_coarse, nfKnots, 'pos', False)
        I_f = _interp_slots_t(Idat, grid_sym, N_coarse, nfKnots, 'val', False)
    else:
        N_final, theta_final = N_coarse, theta_sym
        P_f, I_f = Pdat, Idat
    I_f = torch.where(I_f < 0.0, torch.zeros_like(I_f), I_f)   # interpolation underswing (EasySpin)
    tri_idx, facet_w = _facets(grid_sym, N_final, theta_final)
    if verbosity >= 1:
        kind = 'segments' if tri_idx is None else 'triangles'
        n_fac = len(facet_w) if tri_idx is None else len(tri_idx)
        print(f"  salt: {kind} projection, {n_fac} facets, {n_trans} transitions"
              f"{f', interpolation factor {N_interp}' if do_interp else ''}")
    facet_w_t = torch.as_tensor(np.asarray(facet_w, dtype=float), dtype=torch.float64, device=dev)
    if tri_idx is None:
        return projectzones_t(P_f, I_f, facet_w_t, x_lo, dx, n_points)
    return projecttriangles_t(torch.as_tensor(np.asarray(tri_idx), dtype=torch.long, device=dev), facet_w_t,
                              P_f, I_f, x_lo, dx, n_points)


def salt(
    sys: SpinSystem,
    exp: Experiment,
    opt: Optional[Options] = None,
    freq_range: Optional[tuple[float, float]] = None,
    n_points: int = 512,
    lw_mhz: float | torch.Tensor = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Field-swept ENDOR powder spectrum (frequency domain).

    Parameters
    ----------
    sys:
        Spin system.  Must contain at least one nucleus.
    exp:
        Experimental parameters.  ``exp.mwFreq`` (GHz) and ``exp.Range``
        (mT, EPR sweep range) are used to locate EPR resonance fields.
    opt:
        Computational options.  Defaults to ``Options()``.
        ``GridSize`` is an int ``N`` or ``[N, Ninterp]`` as in EasySpin: in
        fixed-field mode the transitions are followed across the ``N``-knot
        grid, interpolated ``Ninterp``-fold and projected onto the frequency
        axis (``projecttriangles``/``projectzones``), so coarse grids give
        smooth powder patterns.  Set ``opt.device='cuda'`` for GPU acceleration.
    freq_range:
        ``(f_lo, f_hi)`` in MHz for the ENDOR frequency axis.  If ``None``,
        the range is set to ``[0, 2 * mwFreq * 1e3]`` (i.e. 0 to 2× the
        microwave frequency).
    n_points:
        Number of points in the frequency axis.  Default 512.
    lw_mhz:
        Gaussian FWHM (MHz) for the ENDOR lineshape convolution.  Default 0.1.

    Returns
    -------
    freq_axis:
        Frequency axis in MHz, shape ``(n_points,)``.
    spec:
        ENDOR spectrum (a.u.), shape ``(n_points,)``.

    Raises
    ------
    ValueError
        If the spin system contains no nuclei.
    """
    if sys.nNuclei == 0:
        raise ValueError("salt: spin system must contain at least one nucleus.")
    # salt treats single-electron systems without quadrupole coupling by
    # perturbation theory (EPR fields and ENDOR), built from the tensors.
    if (getattr(sys, 'H0Extra', None) is not None and sys.nElectrons == 1):
        from torchspin.spinsystem import _reject_h0extra
        _reject_h0extra(sys, 'salt (single-electron perturbation path)')

    if opt is None:
        opt = Options()
    # Hyperfine enhancement needs the matrix ENDOR path: it bypasses the
    # automatic first-order perturbation shortcut below.
    enhancement = bool(getattr(opt, 'Enhancement', False))

    from torchspin.ham_ezho import check_quadratic_zeeman_unsupported
    check_quadratic_zeeman_unsupported(sys, opt, 'salt')

    # ── Frequency axis ─────────────────────────────────────────────────────
    if freq_range is None:
        f_lo, f_hi = 0.0, 2.0 * exp.mwFreq * 1e3  # MHz
    else:
        f_lo, f_hi = float(freq_range[0]), float(freq_range[1])

    device = torch.device(opt.device)
    freq_axis = torch.linspace(f_lo, f_hi, n_points, dtype=torch.float64, device=device)
    spec = torch.zeros(n_points, dtype=torch.float64, device=device)
    df = (f_hi - f_lo) / (n_points - 1)  # MHz per point

    # ── Grid symmetry ──────────────────────────────────────────────────────
    grid_sym = opt.GridSymmetry
    R_sym2mol = torch.eye(3, dtype=torch.float64)
    if grid_sym in ('auto', ''):
        grid_sym, R_sym2mol = hamsymm(sys)
    # GridSize: int N → compute + project at N knots; [N1, Ni] → compute at
    # N1 knots, interpolate to (N1-1)*Ni+1 knots before projection (EasySpin).
    N_coarse = opt.grid_size_coarse
    N_interp = opt.grid_size_interp
    phi_arr, theta_arr, weights_arr, vecs_sym_c = sphgrid(grid_sym, N_coarse)
    theta_sym = theta_arr.clone()          # symmetry-frame polar angles
    if vecs_sym_c.shape[0] != 3:
        vecs_sym_c = vecs_sym_c.T          # (3, n_orient), symmetry frame

    # Apply symmetry-frame rotation if needed
    if not torch.allclose(R_sym2mol, torch.eye(3, dtype=torch.float64), atol=1e-10):
        sin_t = torch.sin(theta_arr)
        v_sym = torch.stack(
            [sin_t * torch.cos(phi_arr), sin_t * torch.sin(phi_arr), torch.cos(theta_arr)],
            dim=1,
        )
        v_mol = (R_sym2mol.T @ v_sym.T).T
        theta_arr = torch.acos(v_mol[:, 2].clamp(-1.0, 1.0))
        phi_arr   = torch.atan2(v_mol[:, 1], v_mol[:, 0])

    n_orient = phi_arr.shape[0]
    total_weight = weights_arr.sum().item()

    # ── Build operators once and move to target device ─────────────────────
    H0, mux, muy, muz = ham(sys, B0=None)
    snmux, snmuy, snmuz = ham_nz(sys, B0=None)

    H0    = H0.to(device)
    mux   = mux.to(device)
    muy   = muy.to(device)
    muz   = muz.to(device)
    snmux = snmux.to(device)
    snmuy = snmuy.to(device)
    snmuz = snmuz.to(device)

    phi_arr     = phi_arr.to(device)
    theta_arr   = theta_arr.to(device)
    weights_arr = weights_arr.to(device)

    if opt.Verbosity >= 1:
        print(f"salt: {n_orient} orientations, device={opt.device}, "
              f"freq=[{f_lo:.1f},{f_hi:.1f}] MHz")

    # ── Phase 1: EPR resonance fields via vectorized perturbation theory ────
    # Fixed-field mode (EasySpin salt semantics): every orientation
    # contributes at exp.Field; optional orientation selection via
    # exp.ExciteWidth + exp.mwFreq (Gaussian weight on the offset between
    # the effective EPR frequency and the spectrometer frequency).
    fixed_field_mode = getattr(exp, 'is_fixed_field', False)
    if fixed_field_mode:
        B_fixed = torch.tensor([float(exp.Field)], dtype=torch.float64,
                               device=device)
        excite_w = getattr(exp, 'ExciteWidth', None)
        if excite_w is not None and exp.mwFreq is not None:
            # Exact orientation selection (EasySpin orisel.m): diagonalize
            # H(Field) per orientation, weight each EPR transition by
            # TransitionRate * exp(-2*xi^2) with
            # xi = (freq - mwFreq) / GammaWidth,
            # GammaWidth^2 = (HStrain_proj^2 + ExciteWidth^2) / (2 log2).
            dt64 = torch.float64
            sin_t = torch.sin(theta_arr)
            cos_t = torch.cos(theta_arr)
            sin_p = torch.sin(phi_arr)
            cos_p = torch.cos(phi_arr)
            zx, zy, zz = sin_t * cos_p, sin_t * sin_p, cos_t
            xx, xy, xz = cos_t * cos_p, cos_t * sin_p, -sin_t
            yx, yy = -sin_p, cos_p  # yLab_z = 0

            muzL = (zx.view(-1, 1, 1) * mux + zy.view(-1, 1, 1) * muy
                    + zz.view(-1, 1, 1) * muz)
            muxL = (xx.view(-1, 1, 1) * mux + xy.view(-1, 1, 1) * muy
                    + xz.view(-1, 1, 1) * muz)
            muyL = yx.view(-1, 1, 1) * mux + yy.view(-1, 1, 1) * muy

            H_B = H0.unsqueeze(0) - float(exp.Field) * muzL
            E_b, V_b = _eigh(H_B)          # (M, dim), (M, dim, dim)
            dim_os = H0.shape[0]
            iu = torch.triu_indices(dim_os, dim_os, offset=1, device=device)
            u_idx, v_idx = iu[0], iu[1]
            freqs_os = (E_b[:, v_idx] - E_b[:, u_idx]).abs()   # (M, P) MHz

            mxe = torch.bmm(V_b.conj().transpose(1, 2), torch.bmm(muxL, V_b))
            mye = torch.bmm(V_b.conj().transpose(1, 2), torch.bmm(muyL, V_b))
            rate = (mxe.abs() ** 2 + mye.abs() ** 2) / 2.0
            rate_uv = rate[:, u_idx, v_idx]                     # (M, P)

            lw2 = torch.full((n_orient,), float(excite_w) ** 2,
                             dtype=dt64, device=device)
            if sys.HStrain is not None:
                hs = sys.HStrain.to(dtype=dt64, device=device)
                h_os = torch.stack([zx, zy, zz], dim=1)
                if hs.ndim == 1:
                    lw2 = lw2 + ((h_os ** 2) * (hs ** 2).unsqueeze(0)).sum(dim=1)
                else:
                    lw_vec = h_os @ hs.T
                    lw2 = lw2 + (lw_vec * lw_vec).sum(dim=1)
            gamma2 = lw2 / (2.0 * math.log(2.0))                # GammaWidth^2
            xi2 = ((freqs_os - exp.mwFreq * 1e3) ** 2
                   / gamma2.unsqueeze(1))
            orisel_w = (rate_uv * torch.exp(-2.0 * xi2)).sum(dim=1)  # (M,)
        else:
            orisel_w = torch.ones(n_orient, dtype=torch.float64,
                                  device=device)
        # ── Powder average by interpolation + projection (EasySpin salt.m) ──
        # Every orientation contributes at the same field, so transitions can
        # be followed across the grid and projected as tents/segments instead
        # of being binned as sticks.  Stick binning is kept for isotropic
        # (single-orientation) systems and for the field-swept mode below.
        if grid_sym != 'O3' and n_orient > 2:
            _perturb_ok = (sys.nElectrons == 1
                           and (sys.Q is None or not torch.any(sys.Q.abs() > 0))
                           and not enhancement)
            Pdat, Idat = _fixed_field_transitions(
                sys, (H0, mux, muy, muz, snmux, snmuy, snmuz),
                phi_arr, theta_arr, float(exp.Field), orisel_w,
                f_lo, f_hi, float(opt.Threshold), _perturb_ok,
                enhancement=enhancement,
            )
            # EasySpin Opt.OriThreshold: orientations outside the excitation
            # window are not computed (NaN), which also disables interpolation.
            if excite_w is not None and exp.mwFreq is not None:
                w_d = orisel_w.detach()
                ori_thr = float(getattr(opt, 'OriThreshold', 1e-4))
                if float(w_d.max()) > 0 and ori_thr > 0:
                    drop = ((w_d / w_d.max()) < ori_thr).unsqueeze(0).expand_as(Pdat)
                    Pdat = torch.where(drop, torch.full_like(Pdat, float('nan')), Pdat)
            dens = _project_powder(
                Pdat, Idat, grid_sym, N_coarse, N_interp, theta_sym,
                f_lo, df, n_points, verbosity=opt.Verbosity,
            )
            # Density (per MHz, solid angle 4π) → per-bin, orientation-averaged
            # amplitude, the same scale as the stick-binning path.
            spec = dens * (df / (4.0 * math.pi))
            if float(torch.as_tensor(lw_mhz).detach()) > 0.0:
                from torchspin.pepper_autograd import convspec_t
                spec = convspec_t(spec, df, lw_mhz, 0.0, deriv=0)
            return freq_axis.cpu(), spec.cpu()

        B_res_all  = [B_fixed] * n_orient
        intens_all = [orisel_w[k:k + 1] for k in range(n_orient)]
    elif any(t is not None and torch.is_tensor(t) and t.requires_grad
             for t in (sys.g, sys.A, sys.D, sys.Q, getattr(sys, 'ee', None))):
        raise NotImplementedError('salt: the field-swept (Experiment.Range) ENDOR mode bins sticks and is not '
                                  'differentiable; use the fixed-field mode (Experiment.Field) for gradients.')
    elif sys.nElectrons == 1:
        B_res_all, intens_all = resfields_perturb_batch(
            sys, phi_arr, theta_arr, exp, opt
        )
    else:
        from torchspin.resfields_batch import resfields_batch
        from torchspin.experiment import auto_batch_size
        batch_size = auto_batch_size(opt.BatchSize, H0.shape[0], n_orient)
        B_res_all  = [None] * n_orient
        intens_all = [None] * n_orient
        for batch_idx in range((n_orient + batch_size - 1) // batch_size):
            s = batch_idx * batch_size
            e = min(s + batch_size, n_orient)
            B_list, I_list, _ = resfields_batch(
                H0, mux, muy, muz, phi_arr[s:e], theta_arr[s:e], exp, opt, sys=None
            )
            for j, (B_res, intens_epr) in enumerate(zip(B_list, I_list)):
                B_res_all[s + j]  = B_res
                intens_all[s + j] = intens_epr

    # ── Phase 2: ENDOR computation ─────────────────────────────────────────
    # For single-electron systems without quadrupole coupling, use fast
    # first-order perturbation theory (matching MATLAB Opt.Method='perturb1'):
    #   freq = |K_i(mS)| where K_i = mS * A_i@u + nuI_i * h   [vectorized]
    #   intensity = 1 (uniform over mI manifold)
    # This replaces batched eigh on the full Hilbert space.
    _use_perturb_endor = (sys.nElectrons == 1
                          and (sys.Q is None
                               or not torch.any(sys.Q.abs() > 0))
                          and not enhancement)

    if _use_perturb_endor:
        # Fully vectorized first-order perturbation theory ENDOR.
        # Matches MATLAB Opt.Method='perturb1': freq = |K_i(mS)|, intensity=1.
        # No eigh — all ops are O(N_flat × n_nuc × n_mS).
        from torchspin.rotations import erot as _erot
        from torchspin.constants import NMAGN as _NMAGN, PLANCK as _PLANCK
        dt64 = torch.float64
        dev = device

        # g matrix (3×3)
        if sys.fullg:
            g_mat = sys.g[:3, :].to(dtype=dt64, device=dev)
        else:
            g_diag = torch.diag(sys.g[0, :].to(dtype=dt64, device=dev))
            Rg = torch.as_tensor(
                _erot(sys.gFrame[0, :].tolist()).T, dtype=dt64, device=dev
            )
            g_mat = Rg @ g_diag @ Rg.T

        # Field direction vectors for all orientations: (M, 3)
        sin_t = torch.sin(theta_arr.to(dtype=dt64, device=dev))
        cos_t = torch.cos(theta_arr.to(dtype=dt64, device=dev))
        h_all = torch.stack([
            sin_t * torch.cos(phi_arr.to(dtype=dt64, device=dev)),
            sin_t * torch.sin(phi_arr.to(dtype=dt64, device=dev)),
            cos_t,
        ], dim=1)
        # Effective g direction: u = g^T h / |g^T h|
        gn_all  = h_all @ g_mat
        u_all   = gn_all / gn_all.norm(dim=1, keepdim=True).clamp(min=1e-30)

        S_val = float(sys.S[0])
        mS_vals = torch.arange(
            S_val, -S_val - 0.5, -1.0, dtype=dt64, device=dev
        )  # (n_mS,)

        # Flatten valid (orientation, B_res) pairs into arrays
        _flat_idx  = []   # orientation index
        _flat_B    = []   # EPR field (mT)
        _flat_wamp = []   # w_k * amp_epr
        for k in range(n_orient):
            B_res = B_res_all[k]
            if B_res is None or B_res.numel() == 0:
                continue
            w_k   = weights_arr[k].item()
            I_epr = intens_all[k]
            for b_idx in range(B_res.numel()):
                amp = float(I_epr[b_idx].item())
                if amp > 0.0:
                    _flat_idx.append(k)
                    _flat_B.append(float(B_res[b_idx].item()))
                    _flat_wamp.append(w_k * amp)

        if _flat_B:
            idx_t  = torch.tensor(_flat_idx, dtype=torch.long, device=dev)
            B_flat = torch.tensor(_flat_B, dtype=dt64, device=dev)       # (N,)
            w_flat = torch.tensor(_flat_wamp, dtype=dt64, device=dev)    # (N,)
            N_flat = idx_t.shape[0]
            h_flat = h_all[idx_t]   # (N, 3) — field direction
            u_flat = u_all[idx_t]   # (N, 3) — effective g direction

            # Per-nucleus vectorized accumulation
            for iNuc in range(sys.nNuclei):
                gn_i = float(sys.gn[iNuc])
                # Nuclear Larmor (MHz): nuI = -gn * NMAGN * B[T] / PLANCK * 1e-6
                nuI_flat = -gn_i * _NMAGN * (B_flat * 1e-3) / _PLANCK * 1e-6  # (N,)

                # A tensor for nucleus i
                if sys.fullA:
                    A_i = sys.A[3*iNuc: 3*iNuc+3, :].to(dtype=dt64, device=dev)
                else:
                    A_diag = torch.diag(sys.A[iNuc, :].to(dtype=dt64, device=dev))
                    R_A = torch.as_tensor(
                        _erot(sys.AFrame[iNuc, :].tolist()).T,
                        dtype=dt64,
                        device=dev,
                    )
                    A_i = R_A @ A_diag @ R_A.T

                # Hyperfine field vector: Ah[n] = A_i @ u[n]  (N, 3)
                Ah_flat = u_flat @ A_i.T   # (N, 3)

                for mS_val in mS_vals.tolist():
                    # K = mS * A@u + nuI * h  (N, 3)
                    K_flat = (mS_val * Ah_flat
                              + nuI_flat.unsqueeze(1) * h_flat)  # (N, 3)
                    freq_flat = K_flat.norm(dim=1)               # (N,) MHz

                    # Scatter into frequency axis
                    mask = (freq_flat >= f_lo) & (freq_flat <= f_hi)
                    if mask.any():
                        bins = ((freq_flat[mask] - f_lo) / df
                                ).round().long().clamp(0, n_points - 1)
                        spec.scatter_add_(0, bins, w_flat[mask])

        if opt.Verbosity >= 2:
            print(f"  salt: {N_flat} (orientation, B) pairs, perturb1 ENDOR")
    else:
        # Matrix diagonalization path: batched eigh on full Hilbert space.
        # Adaptive batch size: target ~100 MB per bmm
        dim = H0.shape[0]
        n_pairs_est = dim * (dim - 1) // 2
        bytes_per_entry = dim * max(n_pairs_est, 1) * 16 * 2
        _BATCH_ENDOR = max(1, min(512, (100 * 1024 * 1024) // bytes_per_entry))

    # Build flat lists (used only by the matrix-diagonalization path below)
    flat_phi   = []
    flat_theta = []
    flat_B     = []
    flat_w_amp = []   # w_k * amp_epr (orientation weight × EPR intensity)

    if not _use_perturb_endor:
        for k in range(n_orient):
            B_res = B_res_all[k]
            if B_res is None or B_res.numel() == 0:
                continue
            w_k   = weights_arr[k].item()
            phi_k = phi_arr[k].item()
            th_k  = theta_arr[k].item()
            I_epr = intens_all[k]
            for b_idx in range(B_res.numel()):
                amp = I_epr[b_idx].item()
                if amp > 0.0:
                    flat_phi.append(phi_k)
                    flat_theta.append(th_k)
                    flat_B.append(B_res[b_idx].item())
                    flat_w_amp.append(w_k * amp)

    if flat_B and not _use_perturb_endor:
        flat_phi_t = torch.tensor(flat_phi, dtype=torch.float64, device=device)
        flat_theta_t = torch.tensor(flat_theta, dtype=torch.float64, device=device)
        flat_B_t = torch.tensor(flat_B, dtype=torch.float64, device=device)
        flat_w_amp_t = torch.tensor(flat_w_amp, dtype=torch.float64, device=device)
        N_flat = flat_B_t.shape[0]

        # Precompute all-pairs indices (same for every entry)
        all_i = [i for i in range(dim) for j in range(i + 1, dim)]
        all_j = [j for i in range(dim) for j in range(i + 1, dim)]
        pi_all = torch.tensor(all_i, dtype=torch.long, device=device)
        pj_all = torch.tensor(all_j, dtype=torch.long, device=device)
        if enhancement:
            same = _same_nuclear_block(sys, pi_all, pj_all)
            pi_all, pj_all = pi_all[same], pj_all[same]

        for b_start in range(0, N_flat, _BATCH_ENDOR):
            b_end = min(b_start + _BATCH_ENDOR, N_flat)
            Mb = b_end - b_start

            phi_b   = flat_phi_t[b_start:b_end]
            theta_b = flat_theta_t[b_start:b_end]
            B_b     = flat_B_t[b_start:b_end]
            w_b     = flat_w_amp_t[b_start:b_end]

            # Build muzL[Mb, dim, dim] and snm_perp operators
            sin_th = torch.sin(theta_b)
            cos_th = torch.cos(theta_b)
            cos_ph = torch.cos(phi_b)
            sin_ph = torch.sin(phi_b)
            nz_x = (sin_th * cos_ph).view(Mb, 1, 1)
            nz_y = (sin_th * sin_ph).view(Mb, 1, 1)
            nz_z = cos_th.view(Mb, 1, 1)
            nx_x = (cos_th * cos_ph).view(Mb, 1, 1)
            nx_y = (cos_th * sin_ph).view(Mb, 1, 1)
            nx_z = (-sin_th).view(Mb, 1, 1)
            ny_x = (-sin_ph).view(Mb, 1, 1)
            ny_y = cos_ph.view(Mb, 1, 1)

            muzL_b = nz_x * mux + nz_y * muy + nz_z * muz          # (Mb, dim, dim)
            snmux_perp = nx_x * snmux + nx_y * snmuy + nx_z * snmuz
            snmuy_perp = ny_x * snmux + ny_y * snmuy                  # ny_z=0

            # Batched Hamiltonian and eigensystem
            H_B = H0.unsqueeze(0) - B_b.view(Mb, 1, 1) * muzL_b     # (Mb, dim, dim)
            E_b, V_b = _eigh(H_B)                         # (Mb, dim), (Mb, dim, dim)

            # All transition frequencies: (Mb, n_pairs) — cheap, just eigenvalue diffs
            freqs_all = (E_b[:, pj_all] - E_b[:, pi_all]).abs()

            # Frequency range filter FIRST (before expensive matrix elements)
            mask_f = (freqs_all >= f_lo) & (freqs_all <= f_hi)
            if not mask_f.any():
                continue
            mb_idx, pair_idx = torch.where(mask_f)  # (n_surv,)

            # Compute matrix elements only for surviving combos.
            # snmux_perp_m @ V_b[m] → (Mb, dim, dim); then index surviving pairs.
            snmux_V = torch.bmm(snmux_perp, V_b)  # (Mb, dim, dim)
            snmuy_V = torch.bmm(snmuy_perp, V_b)  # (Mb, dim, dim)

            pi_surv = pi_all[pair_idx]   # (n_surv,)
            pj_surv = pj_all[pair_idx]   # (n_surv,)
            # psi_i[k] = V_b[mb_idx[k], :, pi_surv[k]] → (n_surv, dim)
            psi_i_s = V_b[mb_idx, :, pi_surv]           # (n_surv, dim)
            # snmux_V[mb_idx[k], :, pj_surv[k]] → (n_surv, dim)
            op_x_s = snmux_V[mb_idx, :, pj_surv]        # (n_surv, dim)
            op_y_s = snmuy_V[mb_idx, :, pj_surv]        # (n_surv, dim)
            me_x = (psi_i_s.conj() * op_x_s).sum(dim=1) # (n_surv,)
            me_y = (psi_i_s.conj() * op_y_s).sum(dim=1) # (n_surv,)

            intens_surv = ((me_x.abs() ** 2 + me_y.abs() ** 2) / 2.0).real  # (n_surv,)
            if enhancement:
                # Total moment perpendicular to the field (hyperfine
                # enhancement); the threshold below still uses the
                # nuclear-only rates (EasySpin).
                ex_V = torch.bmm(nx_x * mux + nx_y * muy + nx_z * muz, V_b)
                ey_V = torch.bmm(ny_x * mux + ny_y * muy, V_b)
                me_ex = (psi_i_s.conj() * ex_V[mb_idx, :, pj_surv]).sum(dim=1)
                me_ey = (psi_i_s.conj() * ey_V[mb_idx, :, pj_surv]).sum(dim=1)
                intens_enh = ((me_ex.abs() ** 2 + me_ey.abs() ** 2) / 2.0).real

            # Apply intensity threshold
            if opt.Threshold > 0:
                max_i = intens_surv.max()
                if max_i > 0:
                    thresh_mask = intens_surv >= opt.Threshold * max_i
                    mb_idx      = mb_idx[thresh_mask]
                    pair_idx    = pair_idx[thresh_mask]
                    freqs_all_s = freqs_all[mb_idx, pair_idx]
                    intens_surv = intens_surv[thresh_mask]
                    if enhancement:
                        intens_enh = intens_enh[thresh_mask]
                else:
                    continue
            else:
                freqs_all_s = freqs_all[mb_idx, pair_idx]

            if mb_idx.numel() == 0:
                continue

            freqs_flat = freqs_all_s
            if enhancement:
                intens_surv = intens_enh
            amps_flat  = w_b[mb_idx] * intens_surv

            bins = ((freqs_flat - f_lo) / df).round().long()
            mask_bins = (bins >= 0) & (bins < n_points)
            spec.scatter_add_(0, bins[mask_bins], amps_flat[mask_bins])

        if opt.Verbosity >= 2:
            print(f"  salt: {N_flat} total (orientation, B_epr) ENDOR computations")

    # ── Normalize ──────────────────────────────────────────────────────────
    if total_weight > 0:
        spec = spec / total_weight

    # ── Lineshape broadening ───────────────────────────────────────────────
    if lw_mhz > 0.0:
        spec = convspec(spec, df, fwhm_g=lw_mhz, fwhm_l=0.0, deriv=0)

    return freq_axis.cpu(), spec.cpu()
