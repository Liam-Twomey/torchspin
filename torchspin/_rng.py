"""Reproducible random streams for blocks of stochastic trajectories.

A seeded stochastic simulation must give the same result however its work is split:
across thread counts, orientation groups, processes or cluster tasks. To guarantee
that, every independent unit of work (one orientation's block of trajectories) gets
its own random stream, derived from the user's seed with
:class:`numpy.random.SeedSequence`. A unit then draws the same numbers whether it is
simulated alone or batched together with others.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

SeedLike = Union[None, int, np.integer, np.random.SeedSequence, np.random.Generator]


def child_seeds(seed: Optional[int], n: int, stream: int = 0) -> Optional[List[np.random.SeedSequence]]:
    """``n`` independent child seeds of ``seed`` for the given ``stream``.

    Different ``stream`` values (e.g. 0 = local dynamics, 1 = global dynamics) give
    unrelated families of seeds. Returns ``None`` when ``seed`` is ``None``, which keeps
    unseeded runs non-deterministic.
    """
    if seed is None:
        return None
    return np.random.SeedSequence([int(seed), int(stream)]).spawn(int(n))


def traj_generators(seed: Union[SeedLike, Sequence[SeedLike]], n_traj: int) -> List[Tuple[np.random.Generator, int]]:
    """Generators and their trajectory counts for ``n_traj`` trajectories.

    A scalar seed (or ``None``) gives one generator for all trajectories, the
    long-standing behavior. A sequence of ``B`` seeds splits the trajectories into
    ``B`` consecutive equal blocks, each drawn from its own generator.
    """
    if seed is None or isinstance(seed, (int, np.integer, np.random.SeedSequence, np.random.Generator)):
        return [(np.random.default_rng(seed), int(n_traj))]
    seeds = list(seed)
    if not seeds or n_traj % len(seeds):
        raise ValueError(f"{len(seeds)} seed blocks do not divide {n_traj} trajectories evenly")
    per = n_traj // len(seeds)
    return [(np.random.default_rng(s), per) for s in seeds]


def draw(gens: List[Tuple[np.random.Generator, int]], fn: Callable[[np.random.Generator, int], np.ndarray],
         axis: int) -> np.ndarray:
    """``fn(generator, n)`` for every block, concatenated along the trajectory ``axis``."""
    if len(gens) == 1:
        g, n = gens[0]
        return fn(g, n)
    return np.concatenate([fn(g, n) for g, n in gens], axis=axis)
