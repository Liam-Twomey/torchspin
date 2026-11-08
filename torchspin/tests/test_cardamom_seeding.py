"""Seeded cardamom results must not depend on how orientations are grouped.

The fast path processes orientations in groups whose size scales with
``torch.get_num_threads()``. Before per-orientation seed streams, every group was
regenerated from the same ``par.seed``. A seeded spectrum therefore changed with the
thread count, and orientations in different groups shared identical trajectories.
"""
import numpy as np
import pytest
import torch

from torchspin import Experiment, SpinSystem
from torchspin._rng import child_seeds, traj_generators
from torchspin.cardamom import CardamomPar, _spiral_start, cardamom
from torchspin.stochtraj_diffusion import DiffusionPar, stochtraj_diffusion
from torchspin.stochtraj_jump import JumpPar, stochtraj_jump

N_TRAJ, N_STEPS, N_ORI = 20, 2000, 12


def _group_size(threads: int) -> int:
    # mirrors the grouping rule of cardamom's fast path
    budget = min(2e6, 2e5 * threads)
    return max(1, min(N_ORI, int(budget // (N_TRAJ * N_STEPS))))


def _run(threads: int, seed=7, **sys_kw):
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]], tcorr=5e-9, lw=[0.5], **sys_kw)
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        model = 'jump' if 'TransRates' in sys_kw else 'diffusion'
        extra = {'dtSpatial': 1e-11} if model == 'jump' else {}
        par = CardamomPar(Model=model, dtSpin=1e-10, nSteps=N_STEPS, nTraj=N_TRAJ,
                          nOrients=N_ORI, seed=seed, **extra)
        _, spc, td, _ = cardamom(sys, exp, par)
        return np.asarray(spc), np.asarray(td)
    finally:
        torch.set_num_threads(prev)


def test_grouping_actually_changes_between_thread_counts():
    # guards the tests below: they are only meaningful if the groups differ
    assert _group_size(1) != _group_size(8)


@pytest.mark.parametrize('sys_kw', [
    {},
    {'DiffGlobal': 1e7},
    {'TransRates': np.array([[-1e9, 1e9], [1e9, -1e9]]),
     'Orientations': np.array([[0.0, 0.0, 0.0], [0.0, np.pi / 4, 0.0]])},
], ids=['diffusion', 'global-dynamics', 'jump'])
def test_seeded_spectrum_independent_of_thread_count(sys_kw):
    spc1, td1 = _run(1, **sys_kw)
    spc8, td8 = _run(8, **sys_kw)
    np.testing.assert_allclose(td8, td1, rtol=0, atol=1e-12 * np.max(np.abs(td1)))
    np.testing.assert_allclose(spc8, spc1, rtol=0, atol=1e-12 * np.max(np.abs(spc1)))


def test_orientation_blocks_get_distinct_trajectories():
    seeds = child_seeds(5, 3)
    par = DiffusionPar(dt=1e-10, nSteps=50, nTraj=3 * 4, seed=seeds)
    _, _, q = stochtraj_diffusion(np.array([1e8, 1e8, 1e8]), par)    # (4, nSteps, nTraj)
    blocks = [q[:, :, 4 * k:4 * (k + 1)] for k in range(3)]
    assert not np.allclose(blocks[0], blocks[1])
    assert not np.allclose(blocks[1], blocks[2])


def test_block_drawn_alone_equals_block_drawn_in_group():
    seeds = child_seeds(5, 3)
    D = np.array([1e8, 2e8, 3e8])
    start = _spiral_start(4)        # the default spiral grid depends on nTraj: fix it per block
    _, _, q_all = stochtraj_diffusion(D, DiffusionPar(dt=1e-10, nSteps=50, nTraj=12,
                                                      OriStart=np.tile(start, (1, 3)), seed=seeds))
    _, _, q_one = stochtraj_diffusion(D, DiffusionPar(dt=1e-10, nSteps=50, nTraj=4,
                                                      OriStart=start, seed=[seeds[1]]))
    np.testing.assert_array_equal(q_all[:, :, 4:8], q_one)


def test_jump_block_drawn_alone_equals_block_drawn_in_group():
    seeds = child_seeds(9, 2)
    kw = dict(TransRates=np.array([[-1e9, 1e9], [1e9, -1e9]]),
              Orientations=np.array([[0.0, 0.0, 0.0], [0.0, np.pi / 4, 0.0]]))
    *_, s_all = stochtraj_jump(par=JumpPar(dt=1e-11, nSteps=80, nTraj=6, seed=seeds), **kw)
    *_, s_one = stochtraj_jump(par=JumpPar(dt=1e-11, nSteps=80, nTraj=3, seed=[seeds[1]]), **kw)
    np.testing.assert_array_equal(s_all[:, 3:], s_one)


def test_scalar_seed_keeps_single_stream():
    gens = traj_generators(42, 10)
    assert len(gens) == 1 and gens[0][1] == 10
    np.testing.assert_array_equal(gens[0][0].random(5), np.random.default_rng(42).random(5))


def test_uneven_seed_blocks_rejected():
    with pytest.raises(ValueError):
        traj_generators([1, 2, 3], 10)


def test_unseeded_child_seeds_stay_none():
    assert child_seeds(None, 4) is None
