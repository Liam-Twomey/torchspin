"""Degenerate-level bookkeeping for zero-field simulations (``Options.ZeroField``).

At ``B = 0`` the Hamiltonian is the same for every orientation and its levels
can be exactly degenerate (Kramers doublets, accidental degeneracies of
isotropic couplings).  Inside a degenerate subspace the eigenbasis returned by
the eigensolver is arbitrary, so the rate of a single level pair is not
defined; only the rate summed over all pairs that connect two degenerate
clusters is (``tr(P_a O P_b O†)``).  These helpers group the levels into
clusters and sum level-pair quantities over cluster pairs.  Everything stays on
the autograd graph: the cluster assignment uses detached eigenvalues, the sums
are linear maps.
"""
from __future__ import annotations

import torch

#: Relative tolerance (of the level spread) below which two levels count as degenerate.
CLUSTER_RTOL = 1e-9
#: Absolute floor of the tolerance, in the energy unit of the eigenvalues (MHz).
CLUSTER_ATOL = 1e-9


def level_clusters(E: torch.Tensor):
    """Group ascending eigenvalues ``E`` (n,) into degenerate clusters.

    Two neighboring levels belong to the same cluster when they differ by less
    than ``max(CLUSTER_RTOL * (E_max - E_min), CLUSTER_ATOL)``.  Returns
    ``(labels, first, onehot)``: the cluster index of every level (n,), the
    first (lowest) level of every cluster (K,), and the (n, K) float64 one-hot
    membership matrix.
    """
    Ed = E.detach().real.to(torch.float64)
    n = Ed.shape[0]
    spread = float(Ed.max() - Ed.min()) if n else 0.0
    tol = max(CLUSTER_RTOL * spread, CLUSTER_ATOL)
    gaps = Ed[1:] - Ed[:-1]
    new = torch.cat([torch.zeros(1, dtype=torch.long, device=Ed.device), (gaps > tol).long()])
    labels = torch.cumsum(new, dim=0)
    K = int(labels[-1]) + 1 if n else 0
    starts = torch.cat([torch.ones(1, dtype=torch.bool, device=Ed.device), gaps > tol])
    first = torch.nonzero(starts, as_tuple=True)[0]                  # labels are sorted
    onehot = torch.zeros(n, K, dtype=torch.float64, device=Ed.device)
    onehot[torch.arange(n, device=Ed.device), labels] = 1.0
    return labels, first, onehot


def cluster_means(E: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
    """Mean eigenvalue of every cluster, (..., K) from (..., n)."""
    return (E.real.to(torch.float64) @ onehot) / onehot.sum(dim=0)


def inter_cluster_pairs(labels: torch.Tensor):
    """All level pairs ``u < v`` in different clusters and their cluster pairs.

    Returns ``(iu, ju, ca, cb)``: level indices and cluster indices (``ca < cb``).
    """
    n = labels.shape[0]
    iu, ju = torch.triu_indices(n, n, offset=1, device=labels.device)
    keep = labels[iu] != labels[ju]
    iu, ju = iu[keep], ju[keep]
    return iu, ju, labels[iu], labels[ju]
