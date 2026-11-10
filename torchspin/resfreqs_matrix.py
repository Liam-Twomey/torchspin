"""Resonance frequencies for fixed-field frequency-swept EPR (matrix method).

Port of EasySpin's ``resfreqs_matrix.m`` (core physics, single-orientation).

In field-swept EPR (``resfields.py``), the microwave frequency is fixed and the
static field is swept.  In frequency-swept EPR, the static field is fixed and
the resonance frequency is computed directly from the eigenvalue differences:

    freq_ij = E_j - E_i   (MHz)

No root-finding is required because the Hamiltonian at fixed B is diagonalized
once, and all transition energies are available immediately.

Usage
-----
>>> from torchspin import SpinSystem
>>> from torchspin.experiment import Experiment, Options
>>> from torchspin.resfreqs_matrix import resfreqs_matrix
>>> sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
>>> freq, intens = resfreqs_matrix(sys, 0.0, 0.0, B=340.0)
>>> freq      # MHz — should be near 9524 MHz for g=2, B=340 mT
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torchspin._linalg import eigh as _eigh


from torchspin.experiment import Experiment, Options
from torchspin.ham import ham
from torchspin.spinsystem import SpinSystem


def resfreqs_matrix(
    sys: SpinSystem,
    phi: float,
    theta: float,
    B: float,
    freq_range: Optional[tuple[float, float]] = None,
    threshold: float = 1e-4,
    exp: Optional[Experiment] = None,
    opt: Optional[Options] = None,
    init_state: Optional[tuple] = None,
    photo_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute resonance frequencies and intensities for one orientation.

    Diagonalizes ``H = H0 - B * muzL`` at fixed static field ``B`` and
    returns all transition frequencies ``E_j - E_i`` (MHz) together with
    their chi-averaged perpendicular transition rates.

    Parameters
    ----------
    sys:
        Spin system.
    phi:
        Azimuthal angle of the static field in the molecular frame (radians).
    theta:
        Polar angle of the static field in the molecular frame (radians).
    B:
        Static magnetic field (mT).
    freq_range:
        Optional ``(nu_min, nu_max)`` in GHz.  Transitions outside this range
        are removed from the output.  If ``None``, all transitions are returned.
    threshold:
        Relative intensity threshold for discarding weak transitions
        (default ``1e-4``).  Set to ``0`` to keep all.
    exp:
        Optional :class:`~torchspin.experiment.Experiment` object.
        If provided, ``exp.Temperature`` is used for Boltzmann weighting.
    opt:
        Optional :class:`~torchspin.experiment.Options` object.

    Returns
    -------
    freq:
        Resonance frequencies in MHz, shape ``(nTransitions,)``.
    intensities:
        Transition intensities (a.u.), shape ``(nTransitions,)``.

    Notes
    -----
    The transition rate formula is identical to :func:`resfields`:

    .. code-block:: text

        TR = (|mu_L|² - |nB0·mu_L|²) / 2

    which chi-averages over all orientations of the oscillating field B1
    perpendicular to the static field B0.
    """
    if opt is None:
        opt = Options()

    # -----------------------------------------------------------------------
    # Build field-independent Hamiltonian and moment operators
    # -----------------------------------------------------------------------
    H0, mux, muy, muz = ham(sys, B0=None)
    nStates = H0.shape[0]

    # -----------------------------------------------------------------------
    # Build muzL (moment projection onto static field direction)
    # -----------------------------------------------------------------------
    sin_th = math.sin(theta)
    cos_th = math.cos(theta)
    cos_ph = math.cos(phi)
    sin_ph = math.sin(phi)

    muzL = sin_th * cos_ph * mux + sin_th * sin_ph * muy + cos_th * muz

    # -----------------------------------------------------------------------
    # Diagonalize H at fixed B
    # -----------------------------------------------------------------------
    H = H0 - B * muzL
    if opt is not None and bool(getattr(opt, 'QuadraticZeeman', False)):
        # Options.QuadraticZeeman: add the field-quadratic/cubic Sys.Ham terms at B·n
        from torchspin.ham_ezho import field_nonlinear_zeeman_ops, project_field_nonlinear_zeeman
        _zho = field_nonlinear_zeeman_ops(sys, dtype=H.dtype, device=H.device)
        if _zho is not None:
            nv = torch.tensor([[sin_th * cos_ph, sin_th * sin_ph, cos_th]], dtype=torch.float64)
            Q2, Q3 = project_field_nonlinear_zeeman(_zho, nv)
            if Q2 is not None:
                H = H + float(B) ** 2 * Q2[0]
            if Q3 is not None:
                H = H + float(B) ** 3 * Q3[0]
    E, V = _eigh(H)  # E sorted ascending, V[:,k] = eigenvector k

    # -----------------------------------------------------------------------
    # Boltzmann populations (if Temperature given)
    # -----------------------------------------------------------------------
    Temperature = getattr(exp, "Temperature", None) if exp is not None else None
    if Temperature is not None and Temperature > 0:
        from torchspin.constants import PLANCK, BOLTZMANN
        kT = BOLTZMANN * Temperature
        prefactor = 1e6 * PLANCK / kT
        rel_E = E - E[0]
        pops = torch.exp(-prefactor * rel_E)
        pops = pops / pops.sum()
    else:
        pops = None

    # -----------------------------------------------------------------------
    # Collect all transition pairs (i < j)
    # -----------------------------------------------------------------------
    freq_list: list[float] = []
    intens_list: list[float] = []

    freq_range_MHz: Optional[tuple[float, float]] = None
    if freq_range is not None:
        freq_range_MHz = (freq_range[0] * 1e3, freq_range[1] * 1e3)

    for i in range(nStates):
        for j in range(i + 1, nStates):
            freq_ij = (E[j] - E[i]).item()  # MHz

            # Filter by frequency range
            if freq_range_MHz is not None:
                if freq_ij < freq_range_MHz[0] or freq_ij > freq_range_MHz[1]:
                    continue

            # Transition intensity: chi-averaged perpendicular mode
            psi_i = V[:, i]
            psi_j = V[:, j]

            mux_me = (psi_i.conj() * (mux @ psi_j)).sum()
            muy_me = (psi_i.conj() * (muy @ psi_j)).sum()
            muz_me = (psi_i.conj() * (muz @ psi_j)).sum()
            mu_sq = (mux_me.abs()**2 + muy_me.abs()**2 + muz_me.abs()**2).real.item()

            muzL_me = (psi_i.conj() * (muzL @ psi_j)).sum()
            muzL_sq = muzL_me.abs().item() ** 2

            transition_rate = (mu_sq - muzL_sq) / 2.0

            # Boltzmann polarization. Matches MATLAB's resfreqs_matrix.m:
            # Polarization = Polarization / prod(2*Sys.I+1), which gives the
            # per-transition intensity scaling consistent with pepper/garlic.
            # For purely electronic systems (no nuclei) the nuc_factor is 1.
            if init_state is not None:
                # non-equilibrium populations (EasySpin Sys.initState); nuclear
                # sublevels already included in rho -> no 1/prod(2I+1)
                rho, basis = init_state
                rho = rho.to(dtype=psi_i.dtype)
                if basis == 'eigen':
                    polarization = (rho[i, i] - rho[j, j]).real.item()
                else:
                    pu = (psi_i.conj() * (rho @ psi_i)).sum().real.item()
                    pv = (psi_j.conj() * (rho @ psi_j)).sum().real.item()
                    polarization = pu - pv
            elif pops is not None:
                polarization = (pops[i] - pops[j]).item()
            else:
                polarization = 1.0
            # Divide by prod(2I+1) over nuclei in the spin system
            if sys.nNuclei > 0 and init_state is None:
                nuc_factor = 1.0
                for I_nuc in sys.Spins[sys.nElectrons:]:
                    nuc_factor *= (2.0 * I_nuc + 1.0)
                polarization = polarization / nuc_factor

            intens_ij = transition_rate * polarization * photo_weight

            freq_list.append(freq_ij)
            intens_list.append(intens_ij)

    if not freq_list:
        empty = torch.zeros(0, dtype=torch.float64)
        return empty, empty

    freq_t = torch.tensor(freq_list, dtype=torch.float64)
    intens_t = torch.tensor(intens_list, dtype=torch.float64)

    # -----------------------------------------------------------------------
    # Intensity threshold (relative)
    # -----------------------------------------------------------------------
    if threshold > 0 and intens_t.numel() > 0:
        max_i = intens_t.abs().max().item()
        if max_i > 0:
            keep = intens_t.abs() >= threshold * max_i
            freq_t = freq_t[keep]
            intens_t = intens_t[keep]

    return freq_t, intens_t


def resfreqs_batch(
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
    zeeman_ho: Optional[tuple] = None,
    zero_field: bool = False,
) -> tuple:
    """Resonance frequencies for a batch of orientations at fixed ``exp.Field``
    (frequency-swept spectra), with the same interface and conventions as
    :func:`torchspin.resfields_batch.resfields_batch` (EasySpin resfreqs_matrix):

    * positions in **GHz**, all level pairs ``i<j`` (no range filter, so that
      transitions can be tracked across orientations; the accumulation drops
      out-of-window contributions), intensity ``TransitionRate·Polarization·photoWeight``
      with the χ-averaged perpendicular rate (or ``|<v|n_B1·μ|u>|²`` for
      ``xlab_batch`` / parallel mode), Boltzmann, ``init_state`` or
      infinite-temperature (``1/∏(2I+1)``) polarization;
    * strain widths in GHz from :func:`compute_strain_widths_freq_batch`;
    * a relative intensity threshold ``opt.Threshold`` on |intensity|.

    With ``zeeman_ho = (G2, G3)`` (``Options.QuadraticZeeman``, from
    :func:`torchspin.ham_ezho.field_nonlinear_zeeman_ops`) the Hamiltonian at the
    fixed field is H0 − B muzL + B² n·G2·n + B³ G3[n,n,n].

    Returns ``(pos_list, intens_list, widths_list[, pairs_list])`` with one
    tensor per orientation.

    ``zero_field=True`` (``Exp.Field = 0`` with ``Options.ZeroField``): the
    Hamiltonian is ``H0`` for every orientation and is diagonalized once; the
    levels are grouped into degenerate clusters
    (:func:`torchspin._zerofield.level_clusters`) and each returned transition
    is a pair of clusters, with the rate·polarization summed over all its level
    pairs, the position the difference of the cluster mean energies, and the
    level pair ``(first level of the lower cluster, first level of the upper
    cluster)`` as its label (also the form ``pairs`` must take).  Transitions
    inside a cluster (zero frequency) are not returned.
    """
    dev = H0.device
    M = phi_batch.shape[0]
    B0 = float(exp.Field)
    ph = phi_batch.to(device=dev, dtype=torch.float64)
    th = theta_batch.to(device=dev, dtype=torch.float64)
    zL = torch.stack([torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)], dim=1)  # (M,3)
    zc = zL.to(dtype=mux.dtype)
    muzL_b = zc[:, 0, None, None] * mux + zc[:, 1, None, None] * muy + zc[:, 2, None, None] * muz  # (M,n,n)
    ho_b = None
    n = H0.shape[0]
    zf = None
    if zero_field:
        if init_state is not None:
            raise ValueError('Options.ZeroField cannot be combined with non-equilibrium populations (Sys.initState).')
        # B = 0: one Hamiltonian for all orientations, diagonalized once
        from torchspin._zerofield import level_clusters, cluster_means, inter_cluster_pairs
        E0, V0 = _eigh(H0)
        E_b, V_b = E0.unsqueeze(0).expand(M, n), V0.unsqueeze(0).expand(M, n, n)
        labels, first, onehot = level_clusters(E0)
        K = first.numel()
        iu, ju, ca, cb = inter_cluster_pairs(labels)
        key = ca * K + cb
        if pairs is not None:
            want = {(int(a_), int(b_)) for a_, b_ in pairs}
            ok = torch.tensor([(int(first[a_]), int(first[b_])) in want for a_, b_ in zip(ca.tolist(), cb.tolist())],
                              dtype=torch.bool, device=dev)
            iu, ju, key = iu[ok], ju[ok], key[ok]
        ukey, inv = torch.unique(key, return_inverse=True)
        ua, ub = ukey // K, ukey % K
        Ebar = cluster_means(E0, onehot)
        zf = dict(labels=labels, inv=inv, n_cp=ukey.numel(),
                  freq=(Ebar[ub] - Ebar[ua]) * 1e-3,                       # GHz
                  iu_c=first[ua], ju_c=first[ub])
    else:
        H_b = H0.unsqueeze(0) - B0 * muzL_b
        if zeeman_ho is not None:
            from torchspin.ham_ezho import project_field_nonlinear_zeeman
            Q2, Q3 = project_field_nonlinear_zeeman(zeeman_ho, zL)
            Q2 = Q2.to(device=dev, dtype=H_b.dtype) if Q2 is not None else None
            Q3 = Q3.to(device=dev, dtype=H_b.dtype) if Q3 is not None else None
            if Q2 is not None:
                H_b = H_b + B0 ** 2 * Q2
            if Q3 is not None:
                H_b = H_b + B0 ** 3 * Q3
            ho_b = (Q2, Q3)
        E_b, V_b = _eigh(H_b)
        if pairs is not None:
            iu = torch.tensor([p_[0] for p_ in pairs], dtype=torch.long, device=dev)
            ju = torch.tensor([p_[1] for p_ in pairs], dtype=torch.long, device=dev)
        else:
            iu, ju = torch.triu_indices(n, n, offset=1, device=dev)
    from torchspin.excitation import excitation_geometry, exp_mw_mode
    geom = excitation_geometry(exp_mw_mode(exp))
    pos_list, int_list, wid_list, pairs_list = [], [], [], []
    for m in range(M):
        V = V_b[m]; E = E_b[m].real
        psi_i = V[:, iu].T; psi_j = V[:, ju].T                       # (nPairs, n)
        def me(op):
            return (psi_i.conj() * (psi_j @ op.T)).sum(dim=-1)
        mx, my, mz = me(mux), me(muy), me(muz)
        mzL = zc[m, 0] * mx + zc[m, 1] * my + zc[m, 2] * mz
        mu_sq = (mx.abs() ** 2 + my.abs() ** 2 + mz.abs() ** 2).real
        mzL_sq = (mzL.abs() ** 2).real
        if xlab_batch is not None:
            xl = xlab_batch[m].to(device=dev, dtype=torch.float64)
            rate = ((xl[0] * mx + xl[1] * my + xl[2] * mz).abs() ** 2).real
        elif R_batch is not None:
            Rm = R_batch[m].to(device=dev, dtype=mx.dtype)
            mu_L = torch.stack([mx, my, mz], dim=-1) @ Rm.T                    # (nPairs, 3) lab frame
            nB1 = torch.tensor(geom.nB1, dtype=mu_L.dtype, device=dev)
            nk = torch.tensor(geom.nk, dtype=mu_L.dtype, device=dev)
            if geom.kind == 'linear':
                rate = ((mu_L @ nB1).abs() ** 2).real
            elif geom.kind == 'unpolarized':
                rate = (mu_sq - ((mu_L @ nk).abs() ** 2).real) / 2.0
            else:
                cross = torch.linalg.cross(1j * mu_L.conj(), mu_L)   # μ = <v|μ|u> = conj(<u|μ|v>)
                rate = (mu_sq - ((mu_L @ nk).abs() ** 2).real) - geom.sense * (cross @ nk).real
        elif geom.kind == 'linear':
            rate = ((1.0 - geom.xi1 ** 2) * mu_sq + (3.0 * geom.xi1 ** 2 - 1.0) * mzL_sq) / 2.0
        elif geom.kind == 'unpolarized':
            rate = ((1.0 + geom.xik ** 2) * mu_sq + (1.0 - 3.0 * geom.xik ** 2) * mzL_sq) / 4.0
        else:
            xLv = torch.tensor([math.cos(th[m]) * math.cos(ph[m]), math.cos(th[m]) * math.sin(ph[m]), -math.sin(th[m])], dtype=mx.dtype, device=dev)
            yLv = torch.tensor([-math.sin(ph[m]), math.cos(ph[m]), 0.0], dtype=mx.dtype, device=dev)
            mu_M = torch.stack([mx, my, mz], dim=-1)
            cross_z = 2.0 * ((mu_M @ xLv) * (mu_M @ yLv).conj()).imag   # μ = <v|μ|u> = conj(<u|μ|v>)
            rate = (((1.0 + geom.xik ** 2) * mu_sq + (1.0 - 3.0 * geom.xik ** 2) * mzL_sq) / 2.0
                    - geom.sense * geom.xik * cross_z)
        if init_state is not None:
            rho, basis = init_state
            rho = rho.to(device=dev, dtype=V.dtype)
            if basis == 'eigen':
                pd = torch.diagonal(rho).real
                polar = pd[iu] - pd[ju]
            else:
                pu = (psi_i.conj() * (psi_i @ rho.T)).sum(dim=-1).real
                pv = (psi_j.conj() * (psi_j @ rho.T)).sum(dim=-1).real
                polar = pu - pv
        elif exp.Temperature is not None and exp.Temperature > 0:
            from torchspin.constants import PLANCK, BOLTZMANN
            pf = 1e6 * PLANCK / (BOLTZMANN * exp.Temperature)
            pops = torch.exp(-pf * (E - E[0])); pops = pops / pops.sum()
            polar = pops[iu] - pops[ju]
        else:
            polar = torch.ones_like(rate)
            if sys is not None and sys.nNuclei > 0:
                for I in sys.I:
                    polar = polar / (2 * I + 1)
        intens = rate * polar
        if photo_weights is not None:
            intens = intens * float(photo_weights[m])
        if zf is not None:
            # sum over the level pairs of every cluster pair (basis invariant)
            intens = torch.zeros(zf['n_cp'], dtype=intens.dtype, device=dev).index_add(0, zf['inv'], intens)
            freq = zf['freq']
            iu_o, ju_o = zf['iu_c'], zf['ju_c']
        else:
            freq = (E[ju] - E[iu]) * 1e-3                                # GHz
            iu_o, ju_o = iu, ju
        keep = torch.ones_like(intens, dtype=torch.bool)
        if opt.Threshold > 0 and intens.numel() > 0:
            mx_i = intens.abs().max().item()
            if mx_i > 0:
                keep = intens.abs() >= opt.Threshold * mx_i
        pos_list.append(freq[keep]); int_list.append(intens[keep])
        pairs_list.append(torch.stack([iu_o[keep], ju_o[keep]], dim=1))
        wid_list.append(None)
    if sys is not None:
        from torchspin.strainwidth import compute_strain_widths_freq_batch, _has_any_strain
        if _has_any_strain(sys):
            if zf is not None:
                w = compute_strain_widths_freq_batch(sys, H0, muzL_b, ph, th, B0, pairs_list,
                                                     clusters=zf['labels'])
            else:
                w = compute_strain_widths_freq_batch(sys, H0, muzL_b, ph, th, B0, pairs_list,
                                                     **({'ho_batch': ho_b} if ho_b is not None else {}))
            wid_list = [None if x is None else x * 1e-3 for x in w]
    if return_pairs:
        return pos_list, int_list, wid_list, pairs_list
    return pos_list, int_list, wid_list
