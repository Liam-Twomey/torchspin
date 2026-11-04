"""Smoke tests for all torchspin_examples.

Each test:
  - Imports the example module
  - Calls run() — asserts it finishes without exception
  - Asserts outputs are finite and non-zero
  - Asserts output shape matches nPoints where applicable

Key examples also check spectral properties (peak positions, line counts, etc.).
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add torchspin_examples to sys.path so imports work
EXAMPLES_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(EXAMPLES_ROOT.parent))


def _import(subdir, name):
    """Import a torchspin_examples module by subdirectory and filename stem."""
    spec = importlib.util.spec_from_file_location(
        name, EXAMPLES_ROOT / subdir / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_spectrum(result, nPoints=None, label=""):
    """Assert the first two elements of result are a valid B/freq axis and spectrum."""
    B, spec = result[0], result[1]
    # Allow numpy arrays or torch tensors
    B_np = B.numpy() if isinstance(B, torch.Tensor) else np.asarray(B)
    s_np = spec.numpy() if isinstance(spec, torch.Tensor) else np.asarray(spec)
    assert np.all(np.isfinite(B_np)), f"{label}: B axis contains non-finite values"
    assert np.all(np.isfinite(s_np)), f"{label}: spectrum contains non-finite values"
    assert np.any(s_np != 0), f"{label}: spectrum is all zeros"
    if nPoints is not None:
        assert len(s_np) == nPoints, f"{label}: expected {nPoints} points, got {len(s_np)}"


# ---------------------------------------------------------------------------
# Solidstate
# ---------------------------------------------------------------------------

def test_solidstate_broaden():
    mod = _import("solidstate", "broaden")
    result = mod.run()
    B, spec = result[0], result[1]
    _check_spectrum(result, label="broaden")


def test_solidstate_gstrain():
    mod = _import("solidstate", "gstrain")
    results = mod.run()[0]
    # results is a list of (B, spc_no, spc_gs) tuples
    assert len(results) > 0
    for B_np, spc_no, spc_gs in results:
        assert np.all(np.isfinite(B_np))
        assert np.all(np.isfinite(spc_no))
        assert np.all(np.isfinite(spc_gs))


@pytest.mark.slow
def test_solidstate_iron_highspin():
    mod = _import("solidstate", "iron_highspin")
    result = mod.run()
    _check_spectrum(result, label="iron_highspin")


def test_solidstate_triplet_naphthalene():
    mod = _import("solidstate", "triplet_naphthalene")
    result = mod.run()
    _check_spectrum(result, label="triplet_naphthalene")


def test_solidstate_triplet_c60():
    mod = _import("solidstate", "triplet_c60")
    result = mod.run()
    _check_spectrum(result, label="triplet_c60")


@pytest.mark.slow
def test_solidstate_temperature():
    mod = _import("solidstate", "temperature")
    res = mod.run()
    B_ref, spectra = res[0], res[1]
    B_np = B_ref.numpy() if isinstance(B_ref, torch.Tensor) else np.asarray(B_ref)
    assert np.all(np.isfinite(B_np))
    for s in spectra:
        s_np = np.asarray(s)
        assert np.all(np.isfinite(s_np)) and np.any(s_np != 0)


@pytest.mark.slow
def test_solidstate_copper_nitrogens():
    mod = _import("solidstate", "copper_nitrogens")
    result = mod.run()
    _check_spectrum(result, label="copper_nitrogens")
    # Cu + 4x14N: spectrum should be broad (range > 5 mT)
    B_np = result[0].numpy() if isinstance(result[0], torch.Tensor) else np.asarray(result[0])
    assert B_np[-1] - B_np[0] > 5.0


@pytest.mark.slow
def test_solidstate_eespins():
    mod = _import("solidstate", "eespins")
    result = mod.run()
    _check_spectrum(result, label="eespins")


def test_solidstate_freqsweep():
    mod = _import("solidstate", "freqsweep")
    # returns (B, spc_field, freq_GHz, spc_freq, B_fixed_mT)
    result = mod.run()
    assert result is not None and len(result) >= 4
    for s in (result[1], result[3]):
        assert np.all(np.isfinite(np.asarray(s))) and np.any(np.asarray(s) != 0)


@pytest.mark.slow
def test_solidstate_chromium_iii():
    mod = _import("solidstate", "chromium_iii")
    result = mod.run()
    _check_spectrum(result, label="chromium_iii")


@pytest.mark.slow
def test_solidstate_cuedta():
    mod = _import("solidstate", "cuedta")
    result = mod.run()
    _check_spectrum(result, label="cuedta")


@pytest.mark.slow
def test_solidstate_copper_sxq():
    mod = _import("solidstate", "copper_sxq")
    Bs, specS, Bx, specX, Bq, specQ = mod.run()
    for B, s in [(Bs, specS), (Bx, specX), (Bq, specQ)]:
        assert np.all(np.isfinite(B)) and np.all(np.isfinite(s)) and np.any(s != 0)


@pytest.mark.slow
def test_solidstate_matrixperturb():
    mod = _import("solidstate", "matrixperturb")
    # returns (B, spc_matrix, spc_perturb, t_matrix, t_perturb)
    B, spc_m, spc_p, t_m, t_p = mod.run()
    _check_spectrum((B, spc_m), label="matrixperturb_matrix")
    _check_spectrum((B, spc_p), label="matrixperturb_perturb")
    assert t_m > 0 and t_p > 0


@pytest.mark.slow
def test_solidstate_inhomogeneous():
    mod = _import("solidstate", "inhomogeneous")
    result = mod.run()
    _check_spectrum(result, label="inhomogeneous")


@pytest.mark.slow
def test_solidstate_triplet_halffieldintensity():
    mod = _import("solidstate", "triplet_halffieldintensity")
    result = mod.run()
    _check_spectrum(result, label="triplet_halffieldintensity")


def test_solidstate_triplet_triphenylbenzene():
    mod = _import("solidstate", "triplet_triphenylbenzene")
    result = mod.run()
    _check_spectrum(result, label="triplet_triphenylbenzene")


@pytest.mark.slow
def test_solidstate_gd_transitions():
    mod = _import("solidstate", "gd_transitions")
    result = mod.run()
    _check_spectrum(result, label="gd_transitions")


@pytest.mark.slow
def test_solidstate_trifluormethyl():
    mod = _import("solidstate", "trifluormethyl")
    result = mod.run()
    _check_spectrum(result, label="trifluormethyl")


@pytest.mark.slow
def test_solidstate_mn_strain():
    mod = _import("solidstate", "mn_strain")
    result = mod.run()
    _check_spectrum(result, label="mn_strain")


# ---------------------------------------------------------------------------
# Liquids
# ---------------------------------------------------------------------------

def test_liquids_biphenyl():
    mod = _import("liquids", "biphenyl")
    result = mod.run()
    _check_spectrum(result, nPoints=2048, label="biphenyl")
    # biphenyl radical: spectrum should be centered near g=2.003 (~336-340 mT at X-band)
    B_np = result[0].numpy() if isinstance(result[0], torch.Tensor) else np.asarray(result[0])
    assert 330 < B_np.mean() < 345


def test_liquids_fremysalt():
    mod = _import("liquids", "fremysalt")
    result = mod.run()
    _check_spectrum(result, label="fremysalt")
    # 14N nitroxide: 3 lines
    s_np = result[1].numpy() if isinstance(result[1], torch.Tensor) else np.asarray(result[1])
    # Check there are at least 2 sign changes (3-line pattern in derivative mode)
    signs = np.sign(s_np[s_np != 0])
    n_changes = np.sum(np.diff(signs) != 0)
    assert n_changes >= 4, f"fremysalt: expected ≥4 sign changes for 3-line derivative, got {n_changes}"


def test_liquids_fast_fnb():
    mod = _import("liquids", "fast_fnb")
    # returns (B_axis, spectra, logtcorr_list)
    res = mod.run()
    spectra, logtcorr_list = res[1], res[2]
    assert len(spectra) == len(logtcorr_list)
    for k, s in enumerate(spectra):
        s_np = s.numpy() if isinstance(s, torch.Tensor) else np.asarray(s)
        assert np.all(np.isfinite(s_np)) and np.any(s_np != 0), f"fast_fnb[{k}]: bad spectrum"


def test_liquids_naphthalene():
    mod = _import("liquids", "naphthalene")
    result = mod.run()
    B, spec_anion, spec_cation = result
    _check_spectrum((B, spec_anion), nPoints=8192, label="naphthalene_anion")
    _check_spectrum((B, spec_cation), nPoints=8192, label="naphthalene_cation")
    # Anion and cation should be different
    a = spec_anion.numpy() if isinstance(spec_anion, torch.Tensor) else np.asarray(spec_anion)
    c = spec_cation.numpy() if isinstance(spec_cation, torch.Tensor) else np.asarray(spec_cation)
    cosine = float(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c) + 1e-30))
    assert cosine < 0.99, "naphthalene: anion and cation spectra should differ"


def test_liquids_isotopemix():
    mod = _import("liquids", "isotopemix")
    result = mod.run()
    B, spec1, spec2 = result
    _check_spectrum((np.asarray(B), np.asarray(spec1)), label="isotopemix_1")
    _check_spectrum((np.asarray(B), np.asarray(spec2)), label="isotopemix_2")


def test_liquids_biarylradical():
    mod = _import("liquids", "biarylradical")
    result = mod.run()
    _check_spectrum(result, nPoints=8192, label="biarylradical")


def test_liquids_cobalttrimer():
    mod = _import("liquids", "cobalttrimer")
    result = mod.run()
    _check_spectrum(result, nPoints=4096, label="cobalttrimer")


def test_liquids_nitroxide_ftcorr():
    mod = _import("liquids", "nitroxide_ftcorr")
    result = mod.run()
    B, spectra, logtcorr_vals = result
    assert len(spectra) == len(logtcorr_vals)
    for k, s in enumerate(spectra):
        s_np = np.asarray(s)
        assert np.all(np.isfinite(s_np)), f"nitroxide_ftcorr[{k}]: non-finite"
        assert np.any(s_np != 0), f"nitroxide_ftcorr[{k}]: all zeros"


# ---------------------------------------------------------------------------
# Magnetometry
# ---------------------------------------------------------------------------

def test_magnetometry_curie_law():
    mod = _import("magnetometry", "curie_law")
    # returns (T_arr, chi_cm3, chi_curie, chiT, C_cm3)
    result = mod.run()
    assert result is not None and len(result) >= 4
    chi_np = np.asarray(result[1])
    chiT_np = np.asarray(result[3])
    assert np.all(np.isfinite(chi_np)) and np.all(chi_np > 0)
    # chi*T should be roughly constant (Curie law)
    assert chiT_np.std() / chiT_np.mean() < 0.05, "curie_law: chi*T should be nearly constant"


def test_magnetometry_squid_basic():
    mod = _import("magnetometry", "squid_basic")
    # returns (B_fields, curves, T_K) where curves is list of (M, label, S_val) tuples
    B_fields, curves, T_K = mod.run()
    assert len(curves) > 0
    for entry in curves:
        M_np = np.asarray(entry[0])   # first element is M array
        assert np.all(np.isfinite(M_np)) and np.any(M_np > 0)


def test_magnetometry_magnetization_tempdep():
    mod = _import("magnetometry", "magnetization_tempdep")
    # returns (T_arr, results) where results is list of M arrays
    T_arr, results = mod.run()
    assert np.all(np.isfinite(np.asarray(T_arr)))
    assert len(results) > 0


def test_magnetometry_susceptibility():
    mod = _import("magnetometry", "susceptibility")
    # returns (T_arr, chi, chiT, D_cm1)
    res = mod.run()
    assert np.all(np.isfinite(np.asarray(res[1])))
    assert np.all(np.isfinite(np.asarray(res[2])))


def test_magnetometry_effectivemagneticmoment():
    mod = _import("magnetometry", "effectivemagneticmoment")
    # returns (T_arr, mu, mu_eff, mu_eff_lim, g_val, D_cm1)
    res = mod.run()
    assert res is not None and len(res) >= 3
    assert np.all(np.isfinite(np.asarray(res[1])))
    assert np.all(np.isfinite(np.asarray(res[2])))


# ---------------------------------------------------------------------------
# Slow motion
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_slowmotion_nitroxide_basic():
    mod = _import("slowmotion", "nitroxide_basic")
    result = mod.run()
    _check_spectrum(result, label="nitroxide_basic")


@pytest.mark.slow
def test_slowmotion_nitroxide_tcorr():
    mod = _import("slowmotion", "nitroxide_tcorr")
    # returns list of (B, spec, tcorr) tuples
    results = mod.run()
    assert isinstance(results, list) and len(results) > 0
    for k, entry in enumerate(results):
        s_np = entry[1].numpy() if isinstance(entry[1], torch.Tensor) else np.asarray(entry[1])
        assert np.all(np.isfinite(s_np)), f"nitroxide_tcorr[{k}]: non-finite"
        assert np.any(s_np != 0), f"nitroxide_tcorr[{k}]: all zeros"


@pytest.mark.slow
def test_slowmotion_nitroxide_frq():
    mod = _import("slowmotion", "nitroxide_frq")
    # returns list of (B, spec, mwFreq) tuples
    results = mod.run()
    assert isinstance(results, list) and len(results) > 0
    for entry in results:
        s_np = entry[1].numpy() if isinstance(entry[1], torch.Tensor) else np.asarray(entry[1])
        assert np.all(np.isfinite(s_np)) and np.any(s_np != 0)


@pytest.mark.slow
def test_slowmotion_tempone():
    mod = _import("slowmotion", "tempone")
    result = mod.run()
    _check_spectrum(result, label="tempone")


@pytest.mark.slow
def test_slowmotion_tumbling():
    mod = _import("slowmotion", "tumbling")
    # returns list of (B, spec, tcorr) tuples
    results = mod.run()
    assert isinstance(results, list) and len(results) > 0
    for entry in results:
        s_np = entry[1].numpy() if isinstance(entry[1], torch.Tensor) else np.asarray(entry[1])
        assert np.all(np.isfinite(s_np)) and np.any(s_np != 0)


@pytest.mark.slow
def test_slowmotion_slow_fast():
    mod = _import("slowmotion", "slow_fast")
    # returns (B_chili, spec_chili, B_garlic, spec_garlic, tcorr)
    result = mod.run()
    assert result is not None and len(result) >= 4
    for i in (1, 3):
        s_np = np.asarray(result[i])
        assert np.all(np.isfinite(s_np)) and np.any(s_np != 0)


# ---------------------------------------------------------------------------
# ENDOR
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_endor_endorsimple():
    mod = _import("endor", "endorsimple")
    result = mod.run()
    assert result is not None and len(result) >= 2
    freq, spec = result[0], result[1]
    f_np = np.asarray(freq); s_np = np.asarray(spec)
    assert np.all(np.isfinite(f_np))
    assert np.all(np.isfinite(s_np))
    assert np.any(s_np != 0)


@pytest.mark.slow
def test_endor_endorseparate():
    mod = _import("endor", "endorseparate")
    # returns (freq_np, spectra, spec_sum, nu_H)
    result = mod.run()
    assert result is not None and len(result) >= 3
    s_np = np.asarray(result[2])  # spec_sum
    assert np.all(np.isfinite(s_np)) and np.any(s_np != 0)


@pytest.mark.slow
def test_endor_excitew():
    mod = _import("endor", "excitew")
    # returns list of (freq, spec, ExciteWidth) tuples
    results = mod.run()
    assert results is not None and len(results) > 0
    for entry in results:
        s_np = np.asarray(entry[1])
        assert np.all(np.isfinite(s_np)) and np.any(s_np != 0)


@pytest.mark.slow
def test_endor_manynuclei():
    mod = _import("endor", "manynuclei")
    result = mod.run()
    assert result is not None and len(result) >= 2
    freq, spec = result[0], result[1]
    assert np.all(np.isfinite(np.asarray(freq)))
    assert np.all(np.isfinite(np.asarray(spec)))
    assert np.any(np.asarray(spec) != 0)


@pytest.mark.slow
def test_endor_endorperturb():
    mod = _import("endor", "endorperturb")
    result = mod.run()
    assert result is not None and len(result) >= 2


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_fitting_basicfit():
    mod = _import("fitting", "basicfit")
    result = mod.run()
    assert result is not None


@pytest.mark.slow
def test_fitting_multicomponents():
    mod = _import("fitting", "multicomponents")
    result = mod.run()
    assert result is not None


@pytest.mark.slow
def test_fitting_twocompfit():
    mod = _import("fitting", "twocompfit")
    result = mod.run()
    assert result is not None


@pytest.mark.slow
def test_fitting_pentacenefit():
    mod = _import("fitting", "pentacenefit")
    result = mod.run()
    assert result is not None


@pytest.mark.slow
def test_fitting_globallocal():
    mod = _import("fitting", "globallocal")
    result = mod.run()
    assert result is not None


def test_fitting_large_hyperfine_fit():
    """The staged recipe recovers A(91Zr) where a one-shot local fit does not."""
    mod = _import("fitting", "large_hyperfine_fit")
    _, _, _, _, _, r3 = mod.run()
    assert abs(r3.pfit[1] - 780.0) < 2.0
    assert abs(r3.pfit[0] - 1.885) < 1e-4


@pytest.mark.slow
def test_fitting_fit_multifreq():
    mod = _import("fitting", "fit_multifreq")
    result = mod.run()
    assert result is not None


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def test_analysis_denoise():
    mod = _import("analysis", "denoise")
    # returns (x, y_pure, y_noisy, y_flat, y_binom, y_sg, y_rc)
    result = mod.run()
    assert result is not None and len(result) >= 3
    x_np = np.asarray(result[0])
    assert np.all(np.isfinite(x_np))
    for s in result[1:]:
        assert np.all(np.isfinite(np.asarray(s)))
