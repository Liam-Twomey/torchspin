"""GPU vs CPU consistency tests.

These tests are skipped when CUDA is unavailable. On a CUDA-enabled system
they verify that representative simulators produce numerically identical
output on CPU and GPU within float64 tolerance.

Per audit recommendation (Phase 2 deployment review): GPU vs CPU bit-
identical results are not expected for FFT or eigh paths, but max relative
difference should be < 1e-6 for eigh-dominated paths and < 1e-4 for FFT-
dominated paths.

Run on a CUDA workstation::

    pytest torchspin/tests/test_gpu_consistency.py -v

These are the primary smoke tests when first installing torchspin on a new
GPU machine.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from torchspin import (SpinSystem, Experiment, Options,
                       pepper, differentiable_spectrum)
from torchspin.saffron import saffron, PulseExperiment, SaffronOptions


CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_REASON = "CUDA not available"


def _to_numpy(x):
    """Move to CPU + numpy regardless of input device/type."""
    if hasattr(x, 'detach'):
        x = x.detach()
    if hasattr(x, 'cpu'):
        x = x.cpu()
    return np.asarray(x)


def _max_rel_diff(a, b) -> float:
    """Max relative difference; returns 0 if both arrays are zero."""
    a = _to_numpy(a).ravel()
    b = _to_numpy(b).ravel()
    dtype = np.complex128 if np.iscomplexobj(a) or np.iscomplexobj(b) else np.float64
    a = a.astype(dtype)
    b = b.astype(dtype)
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: {a.shape} vs {b.shape}")
    denom = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-30)
    return float(np.max(np.abs(a - b)) / denom)


# ---------------------------------------------------------------------------
# Smoke test: CUDA actually works at all
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
def test_cuda_visible():
    """CUDA detected and a basic torch operation runs on GPU."""
    assert torch.cuda.device_count() >= 1
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda', dtype=torch.float64)
    y = torch.linalg.eigvalsh(torch.diag(x))
    assert torch.allclose(y, x.cpu().to('cuda'))


# ---------------------------------------------------------------------------
# pepper CPU vs GPU
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
def test_pepper_cpu_vs_cuda_nitroxide():
    """pepper output identical between CPU and CUDA for a nitroxide.

    Tolerance: 1e-6 (eigh-dominated path is essentially deterministic in
    float64; FFT in convspec adds ~1e-10 noise).
    """
    sys = SpinSystem(
        S=[0.5], g=[[2.009, 2.006, 2.002]],
        Nucs='14N', A=[[10.0, 10.0, 95.0]],
        lw=[1.0, 0.0],
    )
    exp = Experiment(mwFreq=9.5, Range=[330.0, 350.0],
                     nPoints=512, Harmonic=1)

    opt_cpu = Options(GridSize=31, Verbosity=0, device='cpu')
    opt_gpu = Options(GridSize=31, Verbosity=0, device='cuda')

    _, spc_cpu = pepper(sys, exp, opt_cpu)
    _, spc_gpu = pepper(sys, exp, opt_gpu)

    rel = _max_rel_diff(spc_cpu, spc_gpu)
    assert rel < 1e-6, (
        f"pepper CPU vs CUDA max rel diff = {rel:.2e}; expected < 1e-6"
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
def test_pepper_cpu_vs_cuda_triplet_zfs():
    """pepper for S=1 ZFS triplet: CPU vs CUDA float64 parity."""
    sys = SpinSystem(
        S=[1.0], g=[[2.0, 2.0, 2.0]],
        D=[200.0, 200.0, -400.0],
        lw=[1.0, 0.0],
    )
    exp = Experiment(mwFreq=9.5, Range=[300.0, 380.0],
                     nPoints=512, Harmonic=0)

    opt_cpu = Options(GridSize=31, Verbosity=0, device='cpu')
    opt_gpu = Options(GridSize=31, Verbosity=0, device='cuda')

    _, spc_cpu = pepper(sys, exp, opt_cpu)
    _, spc_gpu = pepper(sys, exp, opt_gpu)

    rel = _max_rel_diff(spc_cpu, spc_gpu)
    assert rel < 1e-6, f"pepper triplet CPU vs CUDA rel diff = {rel:.2e}"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
def test_pepper_cpu_vs_cuda_gstrain():
    """pepper g-strain broadening keeps all intermediate tensors on CUDA."""
    sys = SpinSystem(
        S=[0.5], g=[[2.0104, 2.0074, 2.0026]],
        gStrain=[[0.001, 0.0008, 0.0005]], lw=[0.0, 0.0],
    )
    exp = Experiment(mwFreq=9.5, Range=[332.0, 342.0],
                     nPoints=512, Harmonic=1)

    _, spc_cpu = pepper(
        sys, exp, Options(GridSize=21, Verbosity=0, device='cpu')
    )
    _, spc_gpu = pepper(
        sys, exp, Options(GridSize=21, Verbosity=0, device='cuda')
    )

    rel = _max_rel_diff(spc_cpu, spc_gpu)
    assert rel < 1e-6, f"pepper g-strain CPU vs CUDA rel diff = {rel:.2e}"


# garlic: no CUDA path
# ---------------------------------------------------------------------------
# garlic ignores Options.device: it is closed-form torch on the CPU. A CPU-vs-CUDA
# comparison would compare the CPU result with itself, so there is no garlic test
# here. Add one when garlic gains a device path.


# ---------------------------------------------------------------------------
# salt CPU vs GPU
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
def test_salt_cpu_vs_cuda():
    """salt CPU vs CUDA: powder ENDOR (batched eigh of a 4x4 Hamiltonian)."""
    from torchspin.salt import salt
    sys = SpinSystem(S=0.5, g=2.0, Nucs='1H', A=[[10.0, 10.0, 10.0]], lw=[0.5, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[330.0, 355.0], nPoints=64, Harmonic=0)

    f_cpu, spc_cpu = salt(sys, exp, Options(GridSize=7, GridSymmetry='Ci', Threshold=1e-3,
                                            device='cpu'), n_points=256)
    f_gpu, spc_gpu = salt(sys, exp, Options(GridSize=7, GridSymmetry='Ci', Threshold=1e-3,
                                            device='cuda'), n_points=256)

    # salt computes on the device and returns host tensors (salt.py returns .cpu())
    assert _max_rel_diff(f_cpu, f_gpu) < 1e-12
    rel = _max_rel_diff(spc_cpu, spc_gpu)
    assert rel < 1e-6, f"salt CPU vs CUDA max rel diff = {rel:.2e}; expected < 1e-6"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
def test_sigeq_polarization_on_cuda():
    """sigeq with polarization keeps the input's device (it used to build the
    identity on the CPU and fail with a device mismatch)."""
    from torchspin.sigeq import sigeq
    H = torch.diag(torch.tensor([-1.0e3, 1.0e3], dtype=torch.complex128))
    ref = sigeq(H, 10.0, polarization=True)
    out = sigeq(H.cuda(), 10.0, polarization=True)
    assert out.device.type == 'cuda'
    assert _max_rel_diff(ref, out) < 1e-12


# ---------------------------------------------------------------------------
# saffron CPU vs GPU
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
def test_saffron_cpu_vs_cuda_2peseem():
    """saffron 2pESEEM keeps tensor construction and output conversion safe."""
    sys = SpinSystem(
        S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
        Nucs='1H', A=[[3.0, 3.0, 9.0]],
    )
    exp = PulseExperiment(
        Sequence='2pESEEM', Field=350.0, dt=0.010,
        tau=0.0, nPoints=64,
    )

    _, signal_cpu, _ = saffron(
        sys, exp, SaffronOptions(GridSize=10, TimeDomain=True, device='cpu')
    )
    _, signal_gpu, _ = saffron(
        sys, exp, SaffronOptions(GridSize=10, TimeDomain=True, device='cuda')
    )

    rel = _max_rel_diff(signal_cpu, signal_gpu)
    assert rel < 1e-6, f"saffron CPU vs CUDA rel diff = {rel:.2e}"


# ---------------------------------------------------------------------------
# autograd CPU vs GPU
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
def test_autograd_spectrum_cpu_vs_cuda():
    """differentiable_spectrum CPU vs CUDA forward output."""
    g_cpu = torch.tensor([2.009, 2.006, 2.002], dtype=torch.float64,
                         device='cpu')
    g_gpu = g_cpu.cuda()

    B_cpu, spec_cpu = differentiable_spectrum(
        g_cpu, mwFreq_GHz=9.5, B_range=(330, 350),
        nPoints=256, lw_mT=1.0, GridSize=21, device='cpu',
    )
    B_gpu, spec_gpu = differentiable_spectrum(
        g_gpu, mwFreq_GHz=9.5, B_range=(330, 350),
        nPoints=256, lw_mT=1.0, GridSize=21, device='cuda',
    )

    rel_B = _max_rel_diff(B_cpu, B_gpu)
    rel_spec = _max_rel_diff(spec_cpu, spec_gpu)
    assert rel_B < 1e-10, f"B axis CPU vs CUDA: {rel_B:.2e}"
    assert rel_spec < 1e-6, f"spec CPU vs CUDA: {rel_spec:.2e}"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
def test_autograd_gradient_cpu_vs_cuda():
    """Gradients of g should match between CPU and CUDA."""
    def _grad(device):
        g = torch.tensor([2.009, 2.006, 2.002], dtype=torch.float64,
                         device=device, requires_grad=True)
        _, spec = differentiable_spectrum(
            g, mwFreq_GHz=9.5, B_range=(330, 350),
            nPoints=256, lw_mT=1.0, GridSize=21, device=device,
        )
        loss = (spec ** 2).sum()
        loss.backward()
        return g.grad.detach().cpu().clone()

    grad_cpu = _grad('cpu')
    grad_gpu = _grad('cuda')

    rel = _max_rel_diff(grad_cpu, grad_gpu)
    assert rel < 1e-5, (
        f"autograd gradient CPU vs CUDA max rel diff = {rel:.2e}; "
        f"expected < 1e-5"
    )


# ---------------------------------------------------------------------------
# Graceful CUDA fallback (always runs, even without CUDA)
# ---------------------------------------------------------------------------


def test_cuda_fallback_on_cpu_only_install():
    """Options(device='cuda') on a CPU-only install warns + reverts.

    Audit D20 fix verification: previously this would crash with a cryptic
    low-level torch error; now it warns once and falls back to CPU.
    """
    import warnings
    if CUDA_AVAILABLE:
        pytest.skip("This test only validates fallback when CUDA is missing")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        opt = Options(device='cuda')
        assert opt.device == 'cpu', (
            f"CUDA-unavailable fallback should set device='cpu', got {opt.device!r}"
        )
        assert any('CUDA' in str(x.message) for x in w), (
            "Expected RuntimeWarning mentioning CUDA"
        )
