"""Batched orientation processing for GPU acceleration.

Replaces the previous Python loop with fully vectorized matrix operations:

1. Build muzL[M, dim, dim] for all M orientations simultaneously.
2. Sweep a coarse B-grid (n_coarse points) across ALL orientations in one
   batched eigvalsh call to locate resonance-field brackets.
3. Refine each bracket with n_bisect bisection steps — all brackets in parallel.
4. Compute intensities via one final batched eigh (also handles Boltzmann).
5. Reassemble per-orientation result lists; compute strain widths sequentially
   when requested (unchanged from the original per-orientation path).

Expected speedup over the old Python loop:
  CPU:  5–20×  (loop overhead eliminated)
  GPU: 50–200× (full SIMD parallelism for batched eigh)
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from torchspin._linalg import eigh
from torchspin.experiment import Experiment, Options
from torchspin.spinsystem import SpinSystem


# Number of coarse B-grid points used for bracket detection.
# 20 points ≈ (B_hi−B_lo)/19 interval, typically 3 mT for a 60 mT range —
# finer than the original 2-point bracket and catches multiple crossings.
_N_COARSE = 20

# Bisection iterations.  25 steps reduce a 3 mT bracket to < 1e-7 mT,
# well below the Brent tolerance of 1e-5 mT used in the original code.
_N_BISECT = 25


def resfields_batch(
    H0: torch.Tensor,
    mux: torch.Tensor,
    muy: torch.Tensor,
    muz: torch.Tensor,
    phi_batch: torch.Tensor,
    theta_batch: torch.Tensor,
    exp: Experiment,
    opt: Options,
    sys: Optional[SpinSystem] = None,
    return_pairs: bool = False,
    xlab_batch: Optional[torch.Tensor] = None,
    init_state: Optional[tuple] = None,
    photo_weights: Optional[torch.Tensor] = None,
    R_batch: Optional[torch.Tensor] = None,
    pairs: Optional[list] = None,
    hybrid: Optional[dict] = None,
    zeeman_ho: Optional[tuple] = None,
) -> tuple:
    """Find resonance fields for a batch of orientations (vectorized).

    Parameters
    ----------
    H0, mux, muy, muz:
        Hamiltonian and moment operators (MHz, MHz/mT), each ``(dim, dim)``.
    phi_batch, theta_batch:
        Orientation angles (radians), shape ``(M,)``.
    exp, opt, sys:
        Experiment, options, and spin system.  With *hybrid*, ``sys`` is the
        core system (electrons plus the exactly treated nuclei).
    hybrid:
        Enables EasySpin's ``Opt.Method='hybrid'``.  A dict with ``full_sys``
        (the complete spin system), ``perturb_idx`` (0-based indices of the
        nuclei treated perturbationally), ``n_equiv`` (``SpinSystem.n``) and
        ``int_threshold``.  Each core line is replaced by the nuclear sub-lines
        of :func:`torchspin.resfields_hybrid.hybrid_sublines`.
    zeeman_ho:
        ``Options.QuadraticZeeman``: ``(G2, G3)`` from
        :func:`torchspin.ham_ezho.field_nonlinear_zeeman_ops` (either may be
        None).  The Hamiltonian becomes H(B) = H0 − B muzL + B² Q2 + B³ Q3 with
        Q2 = n·G2·n and Q3 = G3[n,n,n]; level slopes, Newton polish, dB/dE and
        strain widths use dH/dB = −muzL + 2B Q2 + 3B² Q3.  ``None`` (default):
        H(B) = H0 − B muzL exactly as before.

    Returns
    -------
    B_res_list, intens_list, widths_list:
        Lists of tensors (or None), one per orientation.  Same interface as
        the previous sequential implementation.
    """
    M = phi_batch.shape[0]
    if M == 0:
        return ([], [], [], []) if return_pairs else ([], [], [])

    mwFreq_MHz = exp.mwFreq * 1e3       # GHz → MHz
    B_lo = float(exp.Range[0])
    B_hi = float(exp.Range[1])
    rdtype = torch.float64
    dim = H0.shape[0]
    nStates = dim
    dev = phi_batch.device

    # ── Step 1: muzL[M, dim, dim] ─────────────────────────────────────────────
    phi_b = phi_batch.to(rdtype)
    th_b  = theta_batch.to(rdtype)

    sin_th = torch.sin(th_b)
    cos_th = torch.cos(th_b)
    cos_ph = torch.cos(phi_b)
    sin_ph = torch.sin(phi_b)

    # Broadcast: (M,) → (M, 1, 1) for matrix dimensions
    n_x = (sin_th * cos_ph).view(M, 1, 1)
    n_y = (sin_th * sin_ph).view(M, 1, 1)
    n_z = cos_th.view(M, 1, 1)

    # muzL: (M, dim, dim)  — H(B) = H0 - B * muzL[m]
    muzL = n_x * mux.unsqueeze(0) + n_y * muy.unsqueeze(0) + n_z * muz.unsqueeze(0)

    # Field-quadratic/cubic Zeeman terms (Options.QuadraticZeeman): per-orientation
    # Q2 = n·G2·n and Q3 = G3[n,n,n], H(B) = H0 - B muzL + B² Q2 + B³ Q3.
    Q2 = Q3 = None
    if zeeman_ho is not None:
        if hybrid is not None:
            raise NotImplementedError("Options.Method='hybrid' does not support "
                                      "Options.QuadraticZeeman=True with lB >= 2 Sys.Ham terms.")
        from torchspin.ham_ezho import project_field_nonlinear_zeeman
        nvec = torch.stack([n_x.view(M), n_y.view(M), n_z.view(M)], dim=1)
        Q2, Q3 = project_field_nonlinear_zeeman(zeeman_ho, nvec)
        Q2 = Q2.to(device=dev, dtype=muzL.dtype) if Q2 is not None else None
        Q3 = Q3.to(device=dev, dtype=muzL.dtype) if Q3 is not None else None
    has_ho = Q2 is not None or Q3 is not None

    def _ho_H(Bb: torch.Tensor, idx=None):
        """Nonlinear part B² Q2 + B³ Q3 for fields Bb broadcast as (..., 1, 1)."""
        out = 0.0
        if Q2 is not None:
            out = out + Bb ** 2 * (Q2 if idx is None else Q2[idx])
        if Q3 is not None:
            out = out + Bb ** 3 * (Q3 if idx is None else Q3[idx])
        return out

    def _ho_dH(Bb: torch.Tensor, idx=None):
        """Field derivative of the nonlinear part, 2B Q2 + 3B² Q3."""
        out = 0.0
        if Q2 is not None:
            out = out + 2.0 * Bb * (Q2 if idx is None else Q2[idx])
        if Q3 is not None:
            out = out + 3.0 * Bb ** 2 * (Q3 if idx is None else Q3[idx])
        return out

    # ── Step 2: Segment model of the level diagram (EasySpin resfields.m) ──────
    # Energies E(B) and slopes dE/dB = -<n|μzL|n> at a set of field knots define a
    # cubic Hermite model of every transition energy on each segment.  Knots are
    # doubled (midpoints diagonalized) until the model reproduces the transition
    # energies at the midpoints to mwFreq·ModellingAccuracy.  Resonances are the
    # roots of the local cubics, polished by two safeguarded Newton steps with
    # exact diagonalization.  This replaces a dense coarse grid plus bisection
    # (~45 diagonalizations per resonance) by ~10-20 per orientation.
    all_pairs = list(pairs) if pairs is not None else [(i, j) for i in range(nStates) for j in range(i + 1, nStates)]
    pair_i_arr = torch.tensor([p[0] for p in all_pairs], dtype=torch.long, device=dev)
    pair_j_arr = torch.tensor([p[1] for p in all_pairs], dtype=torch.long, device=dev)
    nP = pair_i_arr.numel()
    tol = mwFreq_MHz * float(getattr(opt, 'ModellingAccuracy', 2e-6))
    muzL_c = muzL.to(torch.complex128) if not muzL.is_complex() else muzL

    _chunk_elems = 2.5e7          # complex128 elements per batched eigh (~400 MB)

    def _diag_at(Bk: torch.Tensor):
        """Energies (M, nB, n) and slopes dE/dB (M, nB, n) at fields Bk (nB,)."""
        per = max(1, int(_chunk_elems // (M * dim * dim)))
        Es, Ds = [], []
        for b0 in range(0, Bk.numel(), per):
            Bc = Bk[b0:b0 + per]
            H = H0.unsqueeze(0).unsqueeze(0) - Bc.view(1, -1, 1, 1) * muzL.unsqueeze(1)
            if has_ho:
                Bb = Bc.view(1, -1, 1, 1)
                Q2u = Q2.unsqueeze(1) if Q2 is not None else None
                Q3u = Q3.unsqueeze(1) if Q3 is not None else None
                if Q2u is not None:
                    H = H + Bb ** 2 * Q2u
                if Q3u is not None:
                    H = H + Bb ** 3 * Q3u
            E, V = eigh(H)
            Es.append(E)
            if has_ho:
                dH = -muzL_c.unsqueeze(1)
                if Q2u is not None:
                    dH = dH + 2.0 * Bb * Q2u
                if Q3u is not None:
                    dH = dH + 3.0 * Bb ** 2 * Q3u
                Ds.append((V.conj() * (dH @ V)).sum(dim=-2).real)
            else:
                Ds.append(-(V.conj() * (muzL_c.unsqueeze(1) @ V)).sum(dim=-2).real)
            del H, V
        return torch.cat(Es, dim=1), torch.cat(Ds, dim=1)

    def _pair_f(E, D):
        f = E[..., pair_j_arr] - E[..., pair_i_arr] - mwFreq_MHz     # (M, nB, nP)
        fd = D[..., pair_j_arr] - D[..., pair_i_arr]                 # (M, nB, nP)
        return f, fd

    def _relevant(fa, fb, da, db, h):
        """Segments × pairs in which the transition energy can reach ν
        (endpoint values within the slope-bounded excursion of the cubic)."""
        bound = 1.5 * h * torch.maximum(da.abs(), db.abs()) + tol
        return (fa * fb <= 0) | (torch.minimum(fa.abs(), fb.abs()) <= bound)

    n_seg = 4
    Bk = torch.linspace(B_lo, B_hi, n_seg + 1, dtype=rdtype, device=dev)
    E_k, D_k = _diag_at(Bk)
    max_rounds, max_knots = 8, 400
    for _round in range(max_rounds):
        f_k, fd_k = _pair_f(E_k, D_k)
        h = (Bk[1:] - Bk[:-1]).view(1, -1, 1)
        fa, fb = f_k[:, :-1, :], f_k[:, 1:, :]
        da, db = fd_k[:, :-1, :], fd_k[:, 1:, :]
        rel = _relevant(fa, fb, da, db, h)
        if not rel.any():
            break
        # check the Hermite model at the midpoints of segments that host candidate transitions
        seg_ids = torch.where(rel.any(dim=(0, 2)))[0]
        Bm = 0.5 * (Bk[seg_ids] + Bk[seg_ids + 1])
        E_m, D_m = _diag_at(Bm)
        f_m, _ = _pair_f(E_m, D_m)
        pred = 0.5 * (fa[:, seg_ids] + fb[:, seg_ids]) + h[:, seg_ids] * (da[:, seg_ids] - db[:, seg_ids]) / 8.0
        dev_seg = torch.where(rel[:, seg_ids], (pred - f_m).abs(), torch.zeros_like(pred))
        bad = dev_seg.amax(dim=(0, 2)) > tol                          # per checked segment
        # merge the checked midpoints into the knot set (computed anyway)
        Bk = torch.cat([Bk, Bm]); order = torch.argsort(Bk); Bk = Bk[order]
        E_k = torch.cat([E_k, E_m], dim=1)[:, order]; D_k = torch.cat([D_k, D_m], dim=1)[:, order]
        if not bad.any() or Bk.numel() >= max_knots:
            break
    f_k, fd_k = _pair_f(E_k, D_k)
    nSeg = Bk.numel() - 1
    hseg = (Bk[1:] - Bk[:-1])                                      # (nSeg,)
    h = hseg.view(1, -1, 1)
    fa, fb = f_k[:, :-1, :], f_k[:, 1:, :]
    da, db = fd_k[:, :-1, :] * h, fd_k[:, 1:, :] * h                # slopes in t units
    cand = _relevant(fa, fb, fd_k[:, :-1, :], fd_k[:, 1:, :], h)
    mm, ss, pp = torch.where(cand)
    B_list, m_list, p_list, E_list, psi_i_list, psi_j_list = [], [], [], [], [], []
    if mm.numel() > 0:
        c3 = 2 * (fa - fb) + da + db
        c2 = 3 * (fb - fa) - 2 * da - db
        c1 = da
        c0 = fa
        C3, C2, C1, C0 = c3[mm, ss, pp], c2[mm, ss, pp], c1[mm, ss, pp], c0[mm, ss, pp]
        K = mm.numel()
        # roots of the cubics via companion matrices (quadratic/linear when c3 ≈ 0)
        scale = torch.stack([C3.abs(), C2.abs(), C1.abs(), C0.abs()], dim=1).max(dim=1).values.clamp(min=1e-300)
        C3n, C2n, C1n, C0n = C3 / scale, C2 / scale, C1 / scale, C0 / scale
        roots = torch.full((K, 3), float('nan'), dtype=rdtype, device=dev)
        cubic = C3n.abs() > 1e-10
        if cubic.any():
            comp = torch.zeros(int(cubic.sum()), 3, 3, dtype=rdtype, device=dev)
            comp[:, 1, 0] = 1.0; comp[:, 2, 1] = 1.0
            comp[:, 0, 0] = -C2n[cubic] / C3n[cubic]; comp[:, 0, 1] = -C1n[cubic] / C3n[cubic]; comp[:, 0, 2] = -C0n[cubic] / C3n[cubic]
            # torch computes general (non-Hermitian) eigenvalues on CUDA one matrix at a
            # time with a host sync each (hundreds of thousands per pepper call); one
            # batched CPU call runs the same LAPACK routine as the CPU path
            ev = torch.linalg.eigvals(comp.cpu()).to(dev) if comp.is_cuda else torch.linalg.eigvals(comp)
            real = ev.imag.abs() < 1e-7 * (1 + ev.real.abs())
            roots[cubic] = torch.where(real, ev.real, torch.full_like(ev.real, float('nan')))
        quad = (~cubic) & (C2n.abs() > 1e-10)
        if quad.any():
            disc = C1n[quad] ** 2 - 4 * C2n[quad] * C0n[quad]
            ok = disc >= 0
            sq = torch.sqrt(disc.clamp(min=0))
            r1 = (-C1n[quad] + sq) / (2 * C2n[quad]); r2 = (-C1n[quad] - sq) / (2 * C2n[quad])
            rq = torch.stack([torch.where(ok, r1, torch.full_like(r1, float('nan'))),
                              torch.where(ok, r2, torch.full_like(r2, float('nan'))),
                              torch.full_like(r1, float('nan'))], dim=1)
            roots[quad] = rq
        lin = (~cubic) & (~quad) & (C1n.abs() > 1e-14)
        if lin.any():
            roots[lin, 0] = -C0n[lin] / C1n[lin]
        # keep roots in [0, 1) (last segment: [0, 1]); a sign-change segment with no
        # model root falls back to the secant estimate
        last = (ss == nSeg - 1).unsqueeze(1)
        inside = (roots >= 0) & ((roots < 1) | (last & (roots <= 1)))
        roots = torch.where(inside, roots, torch.full_like(roots, float('nan')))
        sign_ch = (C0 * (C3 + C2 + C1 + C0)) < 0
        none = torch.isnan(roots).all(dim=1) & sign_ch
        if none.any():
            fa_k = C0[none]; fb_k = (C3 + C2 + C1 + C0)[none]
            roots[none, 0] = (fa_k / (fa_k - fb_k)).clamp(0, 1)
        kk, rr = torch.where(~torch.isnan(roots))
        if kk.numel() > 0:
            t = roots[kk, rr]
            B_c = Bk[ss[kk]] + t * hseg[ss[kk]]
            m_c, p_c = mm[kk], pp[kk]
            lo = Bk[ss[kk]]; hi = Bk[ss[kk] + 1]
            # One exact diagonalization per candidate at the model root (EasySpin
            # evaluates the intensity there).  The exact energies and
            # Hellmann-Feynman slopes give a Newton correction of the field for free;
            # the two eigenvectors of each transition are kept for the intensities.
            # Chunked so that at most ~400 MB of eigenvectors exist at a time.
            per = max(1, int(_chunk_elems // (dim * dim)))
            for c0 in range(0, B_c.numel(), per):
                sl = slice(c0, c0 + per)
                Bv, mv, pv = B_c[sl], m_c[sl], p_c[sl]
                H = H0.unsqueeze(0) - Bv.view(-1, 1, 1) * muzL[mv]
                if has_ho:
                    H = H + _ho_H(Bv.view(-1, 1, 1), mv)
                E_v, V_v = eigh(H)
                if has_ho:
                    dE = (V_v.conj() * ((_ho_dH(Bv.view(-1, 1, 1), mv) - muzL_c[mv]) @ V_v)).sum(dim=-2).real
                else:
                    dE = -(V_v.conj() * (muzL_c[mv] @ V_v)).sum(dim=-2).real
                ii = pair_i_arr[pv]; jj = pair_j_arr[pv]
                r = torch.arange(Bv.numel(), device=dev)
                f_v = E_v[r, jj] - E_v[r, ii] - mwFreq_MHz
                fd_v = dE[r, jj] - dE[r, ii]
                step = torch.where(fd_v.abs() > 1e-12, f_v / fd_v, torch.zeros_like(f_v))
                # apply only corrections that are small on the segment scale (looping
                # transitions: the slope vanishes and Newton is unreliable)
                small = step.abs() <= 0.25 * (hi[sl] - lo[sl]).abs()
                B_new = torch.where(small, Bv - step, Bv)
                # accept: model root within the tolerance window and inside the range
                keep = (f_v.abs() <= max(100 * tol, 1e-3)) & (B_new >= B_lo) & (B_new <= B_hi)
                Vp = V_v.transpose(-1, -2)                    # rows = eigenvectors
                B_list.append(B_new[keep]); m_list.append(mv[keep]); p_list.append(pv[keep])
                E_list.append(E_v[keep]); psi_i_list.append(Vp[r, ii][keep]); psi_j_list.append(Vp[r, jj][keep])
                del H, V_v, Vp, dE
    if B_list and sum(b.numel() for b in B_list) > 0:
        B_res_all = torch.cat(B_list); m_idx = torch.cat(m_list); p_idx = torch.cat(p_list)
        E_res = torch.cat(E_list); psi_i = torch.cat(psi_i_list); psi_j = torch.cat(psi_j_list)
        # remove duplicates (roots found from two adjacent segments)
        order = torch.argsort(m_idx * (nP + 1) * 1e6 + p_idx * 1e6 + (B_res_all - B_lo) / max(B_hi - B_lo, 1e-30) * 1e5)
        B_res_all, m_idx, p_idx = B_res_all[order], m_idx[order], p_idx[order]
        E_res, psi_i, psi_j = E_res[order], psi_i[order], psi_j[order]
        same = torch.zeros_like(B_res_all, dtype=torch.bool)
        same[1:] = (m_idx[1:] == m_idx[:-1]) & (p_idx[1:] == p_idx[:-1]) & ((B_res_all[1:] - B_res_all[:-1]).abs() < 1e-6 * max(B_hi - B_lo, 1e-30))
        B_res_all, m_idx, p_idx = B_res_all[~same], m_idx[~same], p_idx[~same]
        E_res, psi_i, psi_j = E_res[~same], psi_i[~same], psi_j[~same]
    else:
        empty = torch.zeros(0, dtype=rdtype, device=dev)
        if return_pairs:
            return [empty] * M, [empty] * M, [None] * M, [None] * M
        return [empty] * M, [empty] * M, [None] * M
    nBrackets = m_idx.shape[0]
    bk = torch.arange(nBrackets, dtype=torch.long, device=dev)
    pi_bk = pair_i_arr[p_idx]
    pj_bk = pair_j_arr[p_idx]
    muzL_bk = muzL[m_idx]

    # ── Step 4: Intensities from the eigenpairs at the resonance fields ───────
    # E_res (nBrackets, nStates) and the transition eigenvectors psi_i, psi_j
    # (nBrackets, dim) were gathered at the candidate diagonalization above.

    # Transition moment matrix elements: <ψ_i|Op|ψ_j>
    # mux/muy/muz are (dim, dim); psi_j is (nBrackets, dim).
    # mux[None] @ psi_j[:,:,None] → (nBrackets, dim, 1) → squeeze
    def _me(op: torch.Tensor) -> torch.Tensor:
        op_pj = (op.unsqueeze(0) @ psi_j.unsqueeze(-1)).squeeze(-1)  # (nBrackets, dim)
        return (psi_i.conj() * op_pj).sum(dim=-1)                    # (nBrackets,)

    mux_me = _me(mux)
    muy_me = _me(muy)
    muz_me = _me(muz)
    mu_sq  = (mux_me.abs() ** 2 + muy_me.abs() ** 2 + muz_me.abs() ** 2).real

    # muzL varies per bracket → use bmm
    muzL_pj = torch.bmm(muzL_bk, psi_j.unsqueeze(-1)).squeeze(-1)  # (nBrackets, dim)
    muzL_me = (psi_i.conj() * muzL_pj).sum(dim=-1)
    muzL_sq = muzL_me.abs() ** 2

    # Transition rate for the excitation geometry (EasySpin resfields.m /
    # p_excitationgeometry).  Powder (χ average) uses |μ|² and |nB0·μ|²; single
    # crystals (R_batch: rows = lab axes in the molecular frame) the lab-frame
    # transition moment μ_L = R μ_M.
    from torchspin.excitation import excitation_geometry, exp_mw_mode
    geom = excitation_geometry(exp_mw_mode(exp))
    if xlab_batch is not None:
        # legacy: linearly polarised B1 along the given lab direction (mol-frame rep.)
        xl = xlab_batch.to(device=mux_me.device, dtype=torch.float64)[m_idx]   # (nBrackets, 3)
        mu_x = xl[:, 0] * mux_me + xl[:, 1] * muy_me + xl[:, 2] * muz_me
        intensities_all = (mu_x.abs() ** 2).real
    elif R_batch is not None:
        Rb = R_batch.to(device=mux_me.device, dtype=torch.float64)[m_idx]      # (nBrackets, 3, 3)
        mu_M = torch.stack([mux_me, muy_me, muz_me], dim=-1)                 # (nBrackets, 3)
        mu_L = (Rb.to(mu_M.dtype) @ mu_M.unsqueeze(-1)).squeeze(-1)           # (nBrackets, 3)
        nB1 = torch.tensor(geom.nB1, dtype=mu_L.dtype, device=mu_L.device)
        nk = torch.tensor(geom.nk, dtype=mu_L.dtype, device=mu_L.device)
        if geom.kind == 'linear':
            intensities_all = ((mu_L @ nB1).abs() ** 2).real
        else:
            nk_mu_sq = ((mu_L @ nk).abs() ** 2).real
            if geom.kind == 'unpolarized':
                intensities_all = (mu_sq - nk_mu_sq) / 2.0
            else:
                cross = torch.linalg.cross(1j * mu_L.conj(), mu_L)            # i μ × μ* with μ = <v|μ|u> = conj(<u|μ|v>)
                intensities_all = (mu_sq - nk_mu_sq) - geom.sense * (cross @ nk).real
    else:
        if geom.kind == 'linear':
            intensities_all = ((1.0 - geom.xi1 ** 2) * mu_sq + (3.0 * geom.xi1 ** 2 - 1.0) * muzL_sq.real) / 2.0
        elif geom.kind == 'unpolarized':
            intensities_all = ((1.0 + geom.xik ** 2) * mu_sq + (1.0 - 3.0 * geom.xik ** 2) * muzL_sq.real) / 4.0
        else:
            # z-component of i μ_L × μ_L* is χ-invariant: use the χ=0 lab axes
            ph = phi_batch.to(device=mux_me.device, dtype=torch.float64)[m_idx]
            th = theta_batch.to(device=mux_me.device, dtype=torch.float64)[m_idx]
            xL = torch.stack([torch.cos(th) * torch.cos(ph), torch.cos(th) * torch.sin(ph), -torch.sin(th)], dim=1)
            yL = torch.stack([-torch.sin(ph), torch.cos(ph), torch.zeros_like(ph)], dim=1)
            mu_M = torch.stack([mux_me, muy_me, muz_me], dim=-1)
            mux_L = (mu_M * xL.to(mu_M.dtype)).sum(dim=-1)
            muy_L = (mu_M * yL.to(mu_M.dtype)).sum(dim=-1)
            cross_z = 2.0 * (mux_L * muy_L.conj()).imag   # nB0·(iμ×μ*) with μ = <v|μ|u> = conj(<u|μ|v>)
            intensities_all = (((1.0 + geom.xik ** 2) * mu_sq + (1.0 - 3.0 * geom.xik ** 2) * muzL_sq.real) / 2.0
                               - geom.sense * geom.xik * cross_z)

    # Aasa-Vänngård frequency-to-field factor dBdE = 1/|<j|μzL|j> - <i|μzL|i>|
    # (mT/MHz; EasySpin resfields.m "general form of the famous 1/g factor")
    # With QuadraticZeeman the slope operator is -dH/dB = muzL - 2B Q2 - 3B² Q3 at B_res.
    slope_op = muzL_bk - _ho_dH(B_res_all.view(-1, 1, 1).to(rdtype), m_idx) if has_ho else muzL_bk
    muzL_ii = (psi_i.conj() * torch.bmm(slope_op, psi_i.unsqueeze(-1)).squeeze(-1)).sum(dim=-1).real
    muzL_jj = (psi_j.conj() * torch.bmm(slope_op, psi_j.unsqueeze(-1)).squeeze(-1)).sum(dim=-1).real
    slope = (muzL_jj - muzL_ii).abs()
    dBdE = torch.where(slope > 1e-5, 1.0 / slope, torch.ones_like(slope))
    intensities_all = intensities_all * dBdE

    # Non-equilibrium populations (EasySpin Sys.initState): polarization
    # <u|rho|u> - <v|rho|v> (or rho[u,u]-rho[v,v] in the eigenbasis); nuclear
    # sublevels are already accounted for in rho (no 1/prod(2I+1)).
    if init_state is not None:
        rho, basis = init_state
        rho = rho.to(device=psi_i.device, dtype=psi_i.dtype)
        if basis == 'eigen':
            pops_d = torch.diagonal(rho).real
            polar = (pops_d[pi_bk] - pops_d[pj_bk]).real
        else:
            pu = (psi_i.conj() * (psi_i @ rho.T)).sum(dim=-1).real
            pv = (psi_j.conj() * (psi_j @ rho.T)).sum(dim=-1).real
            polar = pu - pv
        intensities_all = intensities_all * polar
    # Boltzmann populations
    elif exp.Temperature is not None and exp.Temperature > 0:
        from torchspin.constants import PLANCK, BOLTZMANN
        kT = BOLTZMANN * exp.Temperature
        pf = 1e6 * PLANCK / kT                          # MHz⁻¹
        rel_E  = E_res - E_res[:, 0:1]                  # (nBrackets, nStates)
        pops   = torch.exp(-pf * rel_E)
        pops   = pops / pops.sum(dim=-1, keepdim=True)   # out of place: keeps autograd through Exp
        polar  = (pops[bk, pi_bk] - pops[bk, pj_bk]).real
        intensities_all = intensities_all * polar
    else:
        # Infinite temperature (EasySpin resfields.m): polarization 1 shared
        # among the nuclear sublevels → 1/prod(2I+1).  With hybrid the count is
        # taken over the full system, not just the exactly treated core, which
        # is what EasySpin divides by.
        pol_sys = hybrid['full_sys'] if hybrid is not None else sys
        if pol_sys is not None and pol_sys.nNuclei > 0:
            n_nuc_states = 1.0
            for I, mult in zip(pol_sys.I, pol_sys.n or [1] * pol_sys.nNuclei):
                n_nuc_states *= (2 * I + 1) ** int(mult)
            intensities_all = intensities_all / n_nuc_states

    # Photoselection weights per orientation (EasySpin Idat = ... * photoWeight)
    if photo_weights is not None:
        pw = photo_weights.to(device=intensities_all.device, dtype=intensities_all.dtype)
        intensities_all = intensities_all * pw[m_idx]

    # ── Strain widths of the core transitions ────────────────────────────────
    # EasySpin evaluates the strain width of every transition at its own
    # resonance field (eigenvectors there): batch over all (orientation,
    # resonance) brackets, one "orientation" per resonance.  Computed before any
    # hybrid expansion so the sub-lines of one core transition inherit its width
    # unchanged, as EasySpin does.
    widths_flat: Optional[torch.Tensor] = None
    if sys is not None:
        from torchspin.strainwidth import compute_strain_widths_batch, _has_any_strain
        if _has_any_strain(sys) and B_res_all.numel() > 0:
            widths_flat = compute_strain_widths_batch(
                sys=sys,
                H0=H0,
                muzL_batch=muzL[m_idx],
                phi_batch=phi_batch[m_idx],
                theta_batch=theta_batch[m_idx],
                B0_batch=B_res_all.to(torch.float64),
                pairs_list=None,
                mwFreq=exp.mwFreq,
                pairs_flat=torch.stack([pi_bk, pj_bk], dim=1),
                **({'ho_batch': (Q2[m_idx] if Q2 is not None else None,
                                 Q3[m_idx] if Q3 is not None else None)} if has_ho else {}),
            ).reshape(-1)
        elif _has_any_strain(sys):
            widths_flat = torch.zeros(0, dtype=rdtype, device=dev)

    # ── Hybrid: replace every core line by its nuclear sub-lines ─────────────
    if hybrid is not None and nBrackets > 0:
        from torchspin.resfields_hybrid import hybrid_sublines
        nvec = torch.stack([torch.sin(theta_batch) * torch.cos(phi_batch),
                            torch.sin(theta_batch) * torch.sin(phi_batch),
                            torch.cos(theta_batch)], dim=1).to(rdtype)
        shift, amp = hybrid_sublines(
            sys=hybrid['full_sys'],
            perturb_idx=hybrid['perturb_idx'],
            S_core=hybrid['S_core'],
            psi_u=psi_i, psi_v=psi_j,
            B_res=B_res_all.to(rdtype), dBdE=dBdE.to(rdtype),
            nvec=nvec[m_idx],
            int_threshold=hybrid.get('int_threshold', 0.005),
            n_equiv=hybrid.get('n_equiv'),
        )
        n_sub = shift.shape[1]
        # Positions add, intensities multiply, everything else is replicated.
        # Each sub-line keeps its own slot: the pair index is widened so that the
        # sort below orders by (orientation, core transition, sub-line) and the
        # slot a sub-line occupies is the same for every orientation.
        B_res_all = (B_res_all.unsqueeze(-1) + shift).reshape(-1)
        intensities_all = (intensities_all.unsqueeze(-1) * amp).reshape(-1)
        sub = torch.arange(n_sub, device=dev, dtype=p_idx.dtype)
        p_idx = (p_idx.unsqueeze(-1) * n_sub + sub).reshape(-1)
        nP = nP * n_sub
        m_idx = m_idx.repeat_interleave(n_sub)
        pi_bk = pi_bk.repeat_interleave(n_sub)
        pj_bk = pj_bk.repeat_interleave(n_sub)
        if widths_flat is not None and widths_flat.numel() > 0:
            widths_flat = widths_flat.repeat_interleave(n_sub)
        nBrackets = B_res_all.shape[0]

    # ── Step 5: Reassemble per-orientation result lists (tensor ops; one host sync) ──
    # Intensity threshold per orientation (absolute values: emissive lines of
    # non-equilibrium states are negative), then sort by (orientation, pair
    # index, B_res).  The pair-index order is the natural loop order (0,1) <
    # (0,2) < (1,2) ...; it is what keeps transition slot i_t consistent between
    # neighboring orientations in the triangle projection (ascending-B order
    # breaks at crossover angles).
    if opt.Threshold > 0 and nBrackets > 0:
        absI = intensities_all.abs()
        max_per_m = torch.zeros(M, dtype=absI.dtype, device=dev).scatter_reduce(0, m_idx, absI, reduce='amax', include_self=True)
        keep = absI >= opt.Threshold * max_per_m[m_idx]
        keep |= max_per_m[m_idx] <= 0
        B_res_all, intensities_all, pi_bk, pj_bk, p_idx, m_idx = (
            B_res_all[keep], intensities_all[keep], pi_bk[keep], pj_bk[keep], p_idx[keep], m_idx[keep])
        if widths_flat is not None and widths_flat.numel() > 0:
            widths_flat = widths_flat[keep]
        nBrackets = B_res_all.shape[0]
    span = float(B_hi) - float(B_lo) + 1.0
    sort_key = (m_idx.to(rdtype) * (nP + 1) + p_idx.to(rdtype)) * span + (B_res_all - float(B_lo)).clamp(min=0.0, max=span)
    order = torch.argsort(sort_key)
    B_res_all, intensities_all, pi_bk, pj_bk, p_idx, m_idx = (
        B_res_all[order], intensities_all[order], pi_bk[order], pj_bk[order], p_idx[order], m_idx[order])
    if widths_flat is not None and widths_flat.numel() > 0:
        widths_flat = widths_flat[order]
    counts = torch.bincount(m_idx, minlength=M).tolist()          # the single device→host sync
    empty = torch.zeros(0, dtype=rdtype, device=dev)
    B_split = torch.split(B_res_all, counts)
    I_split = torch.split(intensities_all, counts)
    pi_split = torch.split(pi_bk, counts)
    pj_split = torch.split(pj_bk, counts)
    B_res_list:   list[torch.Tensor]           = [b if c else empty for b, c in zip(B_split, counts)]
    intens_list:  list[torch.Tensor]           = [v if c else empty for v, c in zip(I_split, counts)]
    widths_list:  list[Optional[torch.Tensor]] = [None] * M
    _pairs_saved: list                         = [(a, b) if c else None for a, b, c in zip(pi_split, pj_split, counts)]

    # Strain widths, split like the rest (computed above, before any expansion).
    if widths_flat is not None:
        if widths_flat.numel() > 0:
            widths_list = [w if c else torch.zeros(0, dtype=rdtype, device=dev)
                           for w, c in zip(torch.split(widths_flat, counts), counts)]
        else:
            widths_list = [torch.zeros(0, dtype=rdtype, device=dev)] * M

    if return_pairs:
        pairs_out = [
            None if p is None else torch.stack([p[0], p[1]], dim=1) for p in _pairs_saved
        ]
        return B_res_list, intens_list, widths_list, pairs_out
    return B_res_list, intens_list, widths_list
