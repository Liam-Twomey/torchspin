"""Resonance field finder for CW EPR (matrix method) in torchspin.

For each orientation specified by (phi, theta), finds all magnetic field
values B_res at which an allowed EPR transition occurs at the given
microwave frequency, and computes the corresponding transition intensities.

Algorithm (matrix diagonalization method):
1. Build muzL = projection of total magnetic moment onto lab-z (= field direction)
2. H(B) = H0 - B * muzL  is linear in B → eigenvalues are linear in B
3. Bracket transitions: find pairs (i,j) where E_j(B) - E_i(B) crosses mwFreq
   as B sweeps from Range[0] to Range[1]
4. Locate exact crossing with Brent's method
5. Intensity = |<ψ_i|muzperp|ψ_j>|²  (perpendicular excitation)
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from torchspin.experiment import Experiment, Options
from torchspin.spinsystem import SpinSystem


def resfields(
    H0: torch.Tensor,
    mux: torch.Tensor,
    muy: torch.Tensor,
    muz: torch.Tensor,
    phi: float,
    theta: float,
    exp: Experiment,
    opt: Options,
    sys: Optional[SpinSystem] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Find resonance fields and intensities for one orientation.

    Parameters
    ----------
    H0:
        Field-independent Hamiltonian (MHz), shape ``(nStates, nStates)``.
    mux, muy, muz:
        Magnetic-moment operators (MHz/mT), each ``(nStates, nStates)``.
        Convention: H(B) = H0 - (mux*B[0] + muy*B[1] + muz*B[2]).
    phi, theta:
        Orientation angles (radians).  The static field lies along:
        n = (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)).
    exp:
        Experiment parameters (mwFreq in GHz, Range in mT).
    opt:
        Options (Threshold).
    sys:
        Spin system (optional). Required for strain width calculation.

    Returns
    -------
    B_res:
        Resonance fields in mT, shape ``(nTransitions,)``.
    intensities:
        Transition intensities (a.u.), shape ``(nTransitions,)``.
    widths:
        Gaussian FWHM widths (mT) for each transition, shape ``(nTransitions,)``.
        Returns ``None`` if ``sys`` is not provided or no strain parameters.
    """
    if sys is not None:
        from torchspin.ham_ezho import check_quadratic_zeeman_unsupported
        check_quadratic_zeeman_unsupported(sys, opt, 'resfields (single orientation)')
    mwFreq_MHz = exp.mwFreq * 1e3  # GHz → MHz

    # Projection of moment onto field direction (lab-z = along B)
    sin_th = math.sin(theta)
    cos_th = math.cos(theta)
    cos_ph = math.cos(phi)
    sin_ph = math.sin(phi)

    muzL = sin_th * cos_ph * mux + sin_th * sin_ph * muy + cos_th * muz

    # Build H at two endpoints
    B_lo, B_hi = float(exp.Range[0]), float(exp.Range[1])
    H_lo = H0 - B_lo * muzL
    H_hi = H0 - B_hi * muzL

    # Diagonalize at both endpoints
    E_lo, V_lo = torch.linalg.eigh(H_lo)
    E_hi, V_hi = torch.linalg.eigh(H_hi)

    # Eigenvector-based level tracking: pair eigenstates at B_lo and B_hi
    # by maximum overlap to handle level crossings correctly.
    # Without tracking, eigh's sorted eigenvalues can swap indices at crossings,
    # causing spurious or missed resonances for high-spin systems.
    nStates = E_lo.shape[0]

    # Overlap matrix: |<V_lo[:,i] | V_hi[:,j]>|^2
    overlap = (V_lo.conj().T @ V_hi).abs() ** 2  # (nStates, nStates)

    # Greedy assignment: for each state at B_lo, find best match at B_hi
    perm_hi = torch.zeros(nStates, dtype=torch.long)
    used = set()
    for i in range(nStates):
        row = overlap[i].clone()
        for u in used:
            row[u] = -1.0  # exclude already-assigned
        j = int(row.argmax())
        perm_hi[i] = j
        used.add(j)

    # Reorder B_hi eigenvalues to match B_lo state labeling
    E_hi_tracked = E_hi[perm_hi]

    B_res_list: list[float] = []
    intensity_list: list[float] = []
    transition_pairs: list[tuple[int, int]] = []

    for i in range(nStates):
        for j in range(i + 1, nStates):
            diff_lo = (E_lo[j] - E_lo[i] - mwFreq_MHz).item()
            diff_hi = (E_hi_tracked[j] - E_hi_tracked[i] - mwFreq_MHz).item()

            if diff_lo * diff_hi > 0:
                continue  # same sign → no crossing

            # Brent's method to find B_res in [B_lo, B_hi]
            B_root = _brent(H0, muzL, i, j, mwFreq_MHz, B_lo, B_hi,
                            diff_lo, diff_hi)
            if B_root is None:
                continue

            # Diagonalize at B_res to get intensities
            H_res = H0 - B_root * muzL
            E_res, V_res = torch.linalg.eigh(H_res)

            # Boltzmann population factors (if Temperature is provided)
            if exp.Temperature is not None and exp.Temperature > 0:
                # Boltzmann prefactor: 1e6*h/(k_B*T) in MHz^-1
                from torchspin.constants import PLANCK, BOLTZMANN
                kT = BOLTZMANN * exp.Temperature  # J/K * K = J
                prefactor = 1e6 * PLANCK / kT  # (J·s) / J = s, then *1e6 → MHz^-1
                
                # Populations: exp(-E/kT) normalized
                #  Use E_res - E_res[0] to shift so lowest level has pop=1 before normalization
                rel_energies = E_res - E_res[0]  # MHz
                populations = torch.exp(-prefactor * rel_energies)
                populations = populations / populations.sum()
                
                # Polarization: population difference (lower - upper)
                # EPR: absorb photon to go i→j, so i is lower and j is upper
                polarization = (populations[i] - populations[j]).item()
            else:
                # Infinite temperature: equal populations → no weighting
                # EasySpin resfields.m (infinite temperature): polarization 1 per
                # electron transition, shared among the nuclear sublevels →
                # 1/prod(2I+1). (Finite-temperature populations over the full
                # state space already carry that factor.) Needs `sys`; without
                # it the bare 1 is used.
                polarization = 1.0 / _nuc_states(sys) if sys is not None else 1.0

            # Intensity: chi-averaged perpendicular mode transition rate.
            # MATLAB formula (resfields.m): TR = (|mu_L|² - |nB0·mu_L|²) / 2
            # where mu_L = (<i|mux|j>, <i|muy|j>, <i|muz|j>) is the transition
            # moment vector, and nB0 is the unit vector along the static field.
            # This averages over all azimuthal angles chi around the field axis.
            #
            # NOTE: torch's 1D @ 1D uses BLAS dotc which conjugates the left
            # operand for complex tensors.  Use element-wise multiply + sum
            # to get the correct NON-conjugated matrix elements.

            psi_i = V_res[:, i]
            psi_j = V_res[:, j]

            mux_me = (psi_i.conj() * (mux @ psi_j)).sum()
            muy_me = (psi_i.conj() * (muy @ psi_j)).sum()
            muz_me = (psi_i.conj() * (muz @ psi_j)).sum()
            mu_sq = (mux_me.abs()**2 + muy_me.abs()**2 + muz_me.abs()**2).real.item()

            muzL_me = (psi_i.conj() * (muzL @ psi_j)).sum()
            muzL_sq = muzL_me.abs().item()**2

            transition_rate = (mu_sq - muzL_sq) / 2.0
            
            # Aasa-Vänngård frequency-to-field factor (EasySpin resfields.m):
            # dBdE = 1/|<v|dH/dB|v> - <u|dH/dB|u>| (mT/MHz), the general 1/g factor
            muzL_ii = (psi_i.conj() * (muzL @ psi_i)).sum().real.item()
            muzL_jj = (psi_j.conj() * (muzL @ psi_j)).sum().real.item()
            slope = abs(muzL_jj - muzL_ii)
            dBdE = 1.0 / slope if slope > 1e-5 else 1.0
            intens = transition_rate * polarization * dBdE

            B_res_list.append(B_root)
            intensity_list.append(intens)
            transition_pairs.append((i, j))

    if not B_res_list:
        empty = torch.zeros(0, dtype=torch.float64)
        return empty, empty, None

    B_res_t = torch.tensor(B_res_list, dtype=torch.float64)
    intensity_t = torch.tensor(intensity_list, dtype=torch.float64)

    # Apply intensity threshold
    kept_indices = None
    if opt.Threshold > 0 and intensity_t.numel() > 0:
        max_i = intensity_t.max().item()
        if max_i > 0:
            keep = intensity_t >= opt.Threshold * max_i
            B_res_t = B_res_t[keep]
            intensity_t = intensity_t[keep]
            kept_indices = torch.where(keep)[0]
            # Filter transition pairs too
            transition_pairs = [transition_pairs[idx] for idx in kept_indices.tolist()]

    # Compute strain widths if sys is provided and any strain present
    widths_t = None
    if sys is not None and len(transition_pairs) > 0:
        from torchspin.strainwidth import compute_strain_widths
        
        # Convert transition pairs to tensor [nTrans, 2]
        transitions_tensor = torch.tensor(transition_pairs, dtype=torch.long)
        
        # Compute strain widths at each transition's actual B_res.
        # For S=1/2 there's only one transition so the distinction is moot.
        # For S>=1, using per-transition B_res gives correct eigenvectors
        # and avoids 10-30% errors for large ZFS (D > 1000 MHz).
        n_trans = len(transition_pairs)
        if n_trans > 0 and B_res_t.numel() > 0:
            widths_list = []
            for t_idx in range(n_trans):
                B0_t = B_res_t[t_idx].item() if t_idx < B_res_t.numel() else 0.5 * (exp.Range[0] + exp.Range[1])
                trans_single = transitions_tensor[t_idx:t_idx + 1]
                w_t = compute_strain_widths(
                    sys=sys, H0=H0, mux=mux, muy=muy, muz=muz,
                    phi=phi, theta=theta, transitions=trans_single,
                    mwFreq=exp.mwFreq, B0=B0_t,
                )
                widths_list.append(w_t)
            widths_t = torch.cat(widths_list, dim=0) if all(w is not None for w in widths_list) else None
        else:
            B0_fallback = 0.5 * (exp.Range[0] + exp.Range[1])
            widths_t = compute_strain_widths(
                sys=sys, H0=H0, mux=mux, muy=muy, muz=muz,
                phi=phi, theta=theta, transitions=transitions_tensor,
                mwFreq=exp.mwFreq, B0=B0_fallback,
            )

    return B_res_t, intensity_t, widths_t


# ---------------------------------------------------------------------------
# Brent's root-finding method
# ---------------------------------------------------------------------------

def _nuc_states(sys) -> float:
    """Number of nuclear sublevels prod(2I+1) (1 if no nuclei)."""
    n = 1.0
    for I in getattr(sys, 'I', []):
        n *= (2 * I + 1)
    return n


def _brent(
    H0: torch.Tensor,
    muzL: torch.Tensor,
    i: int,
    j: int,
    mwFreq_MHz: float,
    a: float,
    b: float,
    fa: float,
    fb: float,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> Optional[float]:
    """Find B where E_j(B) - E_i(B) = mwFreq_MHz using Brent's method.

    Parameters
    ----------
    fa, fb:
        Pre-computed function values at a and b (diff_lo, diff_hi).
    """
    if abs(fa) < 1e-12:
        return a
    if abs(fb) < 1e-12:
        return b

    c, fc = a, fa
    d = e = b - a

    for _ in range(max_iter):
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa
            d = e = b - a

        if abs(fc) < abs(fb):
            a, fa = b, fb
            b, fb = c, fc
            c, fc = a, fa

        tol1 = 2 * 1e-15 * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)

        if abs(xm) <= tol1 or abs(fb) < 1e-12:
            return b

        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2 * xm * s
                q = 1 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            if p > 0:
                q = -q
            else:
                p = -p
            s = e
            e = d
            if (2 * p < 3 * xm * q - abs(tol1 * q)) and (p < abs(0.5 * s * q)):
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d

        a, fa = b, fb
        if abs(d) > tol1:
            b += d
        else:
            b += math.copysign(tol1, xm)

        # Evaluate function at new b
        H_b = H0 - b * muzL
        E_b = torch.linalg.eigvalsh(H_b)
        fb = (E_b[j] - E_b[i] - mwFreq_MHz).item()

    return b  # best estimate after max_iter
