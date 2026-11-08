"""torchspin._linalg: the thread-pooled batched eigh/eigvalsh must equal the serial
torch call (same LAPACK routine per matrix) for every shape it dispatches on."""
import pytest
import torch

from torchspin import _linalg


@pytest.mark.parametrize('shape', [(40, 6, 6), (5, 9, 36, 36), (70, 72, 72), (3, 8, 8), (12, 12)])
def test_pooled_eigh_matches_serial(shape):
    torch.manual_seed(0)
    A = torch.randn(*shape, dtype=torch.complex128)
    H = A + A.conj().transpose(-1, -2)
    E0, V0 = torch.linalg.eigh(H)
    prev = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        E1, V1 = _linalg.eigh(H)
        w1 = _linalg.eigvalsh(H)
    finally:
        torch.set_num_threads(prev)
    assert E1.shape == E0.shape and V1.shape == V0.shape
    assert torch.allclose(E1, E0, atol=1e-12, rtol=1e-12)
    assert torch.allclose(w1, torch.linalg.eigvalsh(H), atol=1e-12, rtol=1e-12)
    # eigenvectors agree up to a phase per column
    ph = (V0.conj() * V1).sum(dim=-2)
    assert torch.allclose(ph.abs(), torch.ones_like(ph.abs()), atol=1e-10)
    assert torch.get_num_threads() == prev


def test_pooled_eigh_keeps_autograd_path():
    A = torch.randn(64, 4, 4, dtype=torch.float64, requires_grad=True)
    H = A + A.transpose(-1, -2)
    E, V = _linalg.eigh(H)
    E.sum().backward()
    assert A.grad is not None and torch.isfinite(A.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_cuda_large_matrices_routed_to_cpu():
    A = torch.randn(40, 48, 48, dtype=torch.complex128, device='cuda')
    H = A + A.conj().transpose(-1, -2)
    E, V = _linalg.eigh(H)
    assert E.is_cuda and V.is_cuda
    E0 = torch.linalg.eigvalsh(H)
    assert torch.allclose(E, E0, atol=1e-9, rtol=1e-9)


# --- LAPACK driver failures on degenerate input -----------------------------
# zheevd (divide and conquer) can fail to converge on Hermitian matrices with
# many exactly equal eigenvalue gaps, which is what a spin system with several
# identical nuclei produces: pepper with Cu + 4x14N raised
# "The algorithm failed to converge ... too many repeated eigenvalues" from one
# matrix out of a 59-matrix batch.  eigh/eigvalsh repair such matrices
# individually instead of letting the whole orientation batch abort.

def _degenerate_hermitian(n=12, dtype=torch.complex128):
    """A Hermitian matrix with heavily repeated eigenvalues."""
    torch.manual_seed(3)
    A = torch.randn(n, n, dtype=dtype)
    Q, _ = torch.linalg.qr(A)
    lam = torch.tensor([1.0] * (n // 3) + [2.0] * (n // 3) + [3.0] * (n - 2 * (n // 3)),
                       dtype=Q.real.dtype)
    H = Q @ torch.diag(lam).to(Q.dtype) @ Q.conj().transpose(-1, -2)
    return 0.5 * (H + H.conj().transpose(-1, -2))


def _failing(real_fn, fail_on):
    """Wrap a torch routine so it raises _LinAlgError for selected batch sizes."""
    def wrapped(H, *a, **k):
        if H.ndim >= 3 and H.shape[0] in fail_on:
            raise torch._C._LinAlgError('linalg.eigh: (Batch element 0): The algorithm '
                                        'failed to converge (test stub)')
        if H.ndim < 3 and 0 in fail_on:
            raise torch._C._LinAlgError('linalg.eigh: failed to converge (test stub)')
        return real_fn(H, *a, **k)
    return wrapped


@pytest.mark.parametrize('batch', [1, 4, 40])
def test_eigh_repairs_a_failing_batch(monkeypatch, batch):
    """A batch that the torch driver rejects is still diagonalized correctly."""
    H = _degenerate_hermitian().expand(batch, -1, -1).contiguous()
    expected = torch.linalg.eigvalsh(H)
    # Fail for the full batch and for every chunk size the pool might produce.
    monkeypatch.setattr(torch.linalg, 'eigh',
                        _failing(torch.linalg.eigh, set(range(batch + 1))))
    monkeypatch.setattr(_linalg, '_safe_eigh',
                        _linalg._safe(torch.linalg.eigh, __import__('numpy').linalg.eigh))
    E, V = _linalg.eigh(H)
    assert E.shape == expected.shape and V.shape == H.shape
    assert torch.allclose(E, expected, atol=1e-10, rtol=1e-10)
    # V must still diagonalize H
    recon = V @ torch.diag_embed(E.to(V.dtype)) @ V.conj().transpose(-1, -2)
    assert torch.allclose(recon, H, atol=1e-9)


def test_eigvalsh_repairs_a_failing_batch(monkeypatch):
    H = _degenerate_hermitian().expand(40, -1, -1).contiguous()
    expected = torch.linalg.eigvalsh(H)
    monkeypatch.setattr(torch.linalg, 'eigvalsh',
                        _failing(torch.linalg.eigvalsh, set(range(41))))
    monkeypatch.setattr(_linalg, '_safe_eigvalsh',
                        _linalg._safe(torch.linalg.eigvalsh, __import__('numpy').linalg.eigvalsh))
    w = _linalg.eigvalsh(H)
    assert torch.allclose(w, expected, atol=1e-10, rtol=1e-10)


def test_jitter_fallback_when_numpy_also_fails():
    """With both LAPACK routes refusing, a tiny diagonal perturbation is the last resort."""
    import numpy as np
    H = _degenerate_hermitian()
    expected = torch.linalg.eigvalsh(H)
    real = torch.linalg.eigh
    calls = {'n': 0}

    def torch_fails_once(A, *a, **k):
        # Fail on the unperturbed matrix, succeed on the jittered retry.
        calls['n'] += 1
        if calls['n'] == 1:
            raise torch._C._LinAlgError('failed to converge (test stub)')
        return real(A, *a, **k)

    def np_fails(a):
        raise np.linalg.LinAlgError('Eigenvalues did not converge (test stub)')

    E, V = _linalg._one(torch_fails_once, np_fails, H)
    assert calls['n'] == 2, 'the jittered retry should have been used'
    # The perturbation is 1e-14 relative, so eigenvalues move well below 1e-9.
    assert torch.allclose(E, expected, atol=1e-9)


def test_grad_path_failure_does_not_detach():
    """On the autograd graph the NumPy detour is skipped so gradients survive."""
    import numpy as np
    A = torch.randn(6, 6, dtype=torch.float64, requires_grad=True)
    H = A + A.transpose(-1, -2)
    calls = {'np': 0}

    def np_spy(a):
        calls['np'] += 1
        return np.linalg.eigvalsh(a)

    real = torch.linalg.eigvalsh
    first = {'done': False}

    def fails_first(X, *a, **k):
        if not first['done']:
            first['done'] = True
            raise torch._C._LinAlgError('failed to converge (test stub)')
        return real(X, *a, **k)

    w = _linalg._one(fails_first, np_spy, H, allow_numpy=not H.requires_grad)
    assert calls['np'] == 0, 'NumPy must not be used for a tensor carrying gradients'
    assert w.requires_grad
    w.sum().backward()
    assert A.grad is not None and torch.isfinite(A.grad).all()


def test_thread_count_restored_under_concurrency():
    """_pooled mutates a process-global setting; concurrent callers must not corrupt it."""
    from concurrent.futures import ThreadPoolExecutor
    torch.manual_seed(1)
    A = torch.randn(80, 16, 16, dtype=torch.complex128)
    H = A + A.conj().transpose(-1, -2)
    prev = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        with ThreadPoolExecutor(4) as ex:
            outs = list(ex.map(lambda _: _linalg.eigvalsh(H), range(8)))
        assert torch.get_num_threads() == 4, 'global thread count was left modified'
    finally:
        torch.set_num_threads(prev)
    ref = torch.linalg.eigvalsh(H)
    for w in outs:
        assert torch.allclose(w, ref, atol=1e-12)


def test_cuda_chunked_reassembles_batches(monkeypatch):
    """The memory-bounded chunking used for CUDA batches returns the unchunked result
    (checked on CPU tensors with a forced small chunk)."""
    monkeypatch.setattr(_linalg, '_cuda_chunk', lambda nb, device: 7)
    torch.manual_seed(5)
    A = torch.randn(3, 10, 6, 6, dtype=torch.complex128)
    H = A + A.conj().transpose(-1, -2)
    E, V = _linalg._cuda_chunked(_linalg._safe_eigh, H)
    E0, V0 = torch.linalg.eigh(H)
    assert E.shape == E0.shape and V.shape == V0.shape
    assert torch.allclose(E, E0, atol=1e-12)
    # eigenvectors up to phase: compare the reconstructed matrices
    assert torch.allclose(V @ torch.diag_embed(E).to(V.dtype) @ V.conj().transpose(-1, -2), H, atol=1e-10)
    w = _linalg._cuda_chunked(_linalg._safe_eigvalsh, H)
    assert torch.allclose(w, E0, atol=1e-12)
