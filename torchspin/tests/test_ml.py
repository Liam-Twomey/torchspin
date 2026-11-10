"""torchspin.ml: parameter packing, batched simulation, dataset determinism, losses."""
import torch

from torchspin import SpinSystem, Experiment, Options
from torchspin.ml import ParamSpec, simulate, simulate_batch, SpectrumDataset, cosine_loss, rmsd, normalize, resample
from torchspin.pepper import pepper

torch.set_default_dtype(torch.float64)

TEMPLATE = SpinSystem(S=[0.5], g=[[2.05, 2.05, 2.25]], Nucs='63Cu', A=[[30, 30, 500]], lw=[0.5, 0.0])
SPEC = ParamSpec({'g': [(2.0, 2.1), (2.0, 2.1), (2.15, 2.35)], 'A': [(20, 60), (20, 60), (350, 650)], 'lw': [(0.2, 1.5), (0, 0)]},
                 transforms={'lw': 'softplus'})
EXP = Experiment(mwFreq=9.5, Range=[260, 360], nPoints=256, Harmonic=1)
OPT = Options(GridSize=7, Verbosity=0)


def test_pack_unpack_roundtrip():
    vec = SPEC.pack(TEMPLATE)
    assert vec.numel() == SPEC.n_free == 7
    sys = SPEC.unpack(TEMPLATE, vec)
    assert torch.allclose(sys.g, TEMPLATE.g) and torch.allclose(sys.A, TEMPLATE.A)
    assert abs(float(sys.lw[0]) - 0.5) < 1e-12 and float(sys.lw[1]) == 0.0
    v2 = vec.clone(); v2[5] = SPEC._fwd('A', torch.tensor(600.0))
    assert abs(float(SPEC.unpack(TEMPLATE, v2).A[0, 2]) - 600.0) < 1e-12


def test_simulate_batch_matches_pepper_rows():
    gen = torch.Generator().manual_seed(3)
    P = SPEC.sample(3, gen)
    x, Y = simulate_batch('pepper', TEMPLATE, EXP, OPT, SPEC, P)
    assert Y.shape == (3, 256)
    for i in range(3):
        _, y = pepper(SPEC.unpack(TEMPLATE, P[i]), EXP, OPT)
        assert float((Y[i] - y).abs().max() / y.abs().max()) < 1e-9


def test_gradient_through_unpack_and_loss():
    gen = torch.Generator().manual_seed(1)
    target = SPEC.sample(1, gen)[0]
    _, y_ref = simulate('pepper', SPEC.unpack(TEMPLATE, target), EXP, OPT)
    theta = (target + 0.01 * torch.randn(SPEC.n_free, generator=gen)).requires_grad_(True)
    _, y = simulate('pepper', SPEC.unpack(TEMPLATE, theta), EXP, OPT)
    loss = cosine_loss(normalize(y), normalize(y_ref)) + rmsd(y, y_ref)
    (g,) = torch.autograd.grad(loss, theta)
    assert g.shape == theta.shape and torch.all(torch.isfinite(g)) and float(g.abs().sum()) > 0


def test_dataset_deterministic_and_shapes():
    x_out = torch.linspace(262.0, 358.0, 128)
    ds = SpectrumDataset(SPEC, TEMPLATE, EXP, OPT, n=2, seed=5, noise=0.02, baseline=0.05, x_out=x_out)
    a = list(iter(ds)); b = list(iter(ds))
    assert len(a) == 2 and a[0][0].shape == (128,) and a[0][1].shape == (SPEC.n_free,)
    assert torch.equal(a[0][0], b[0][0]) and torch.equal(a[1][1], b[1][1])
    assert a[0][0].dtype == torch.float32


def test_resample_and_normalize():
    x = torch.linspace(0.0, 1.0, 11); y = x ** 2
    xn = torch.tensor([0.05, 0.5, 0.95])
    assert torch.allclose(resample(xn, x, y), torch.tensor([0.005, 0.25, 0.905]), atol=1e-12)
    assert float(normalize(torch.tensor([1.0, -4.0, 2.0])).abs().max()) == 1.0


def test_simulate_batch_workers():
    gen = torch.Generator().manual_seed(7)
    P = SPEC.sample(2, gen)
    x1, Y1 = simulate_batch('pepper', TEMPLATE, EXP, OPT, SPEC, P)
    x2, Y2 = simulate_batch('pepper', TEMPLATE, EXP, OPT, SPEC, P, workers=2)
    assert torch.allclose(Y1, Y2) and torch.allclose(torch.as_tensor(x1), torch.as_tensor(x2))


def test_simulate_batch_reuses_its_worker_pool():
    """A data-generation loop starts the worker processes once, not on every call."""
    import importlib
    E = importlib.import_module('torchspin.esfit')
    gen = torch.Generator().manual_seed(11)
    P = SPEC.sample(4, gen)
    _, Y1 = simulate_batch('pepper', TEMPLATE, EXP, OPT, SPEC, P, workers=2)
    pool = E._POOL[2]
    pids = {p.pid for p in pool._pool}
    _, Y2 = simulate_batch('pepper', TEMPLATE, EXP, OPT, SPEC, P, workers=2)
    assert E._POOL[2] is pool and {p.pid for p in pool._pool} == pids
    assert torch.equal(Y1, Y2)
    _, Y3 = simulate_batch('pepper', TEMPLATE, EXP, OPT, SPEC, P, workers='auto')
    assert torch.allclose(Y1, Y3)
