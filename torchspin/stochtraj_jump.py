"""Markovian jump trajectory generator.

Port of MATLAB EasySpin's ``stochtraj_jump.m``.  Generates stochastic
trajectories of discrete state jumps via kinetic Monte Carlo, with
optional conversion to quaternion/rotation-matrix orientations.

Each state has an associated orientation (Euler angles), and the system
jumps between states according to a transition rate matrix or transition
probability matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import expm

from torchspin._rng import draw, traj_generators
from torchspin.stochtraj_diffusion import _euler2quat_active, _quat_to_rotmat_batch


# ---------------------------------------------------------------------------
# Dataclass for jump parameters
# ---------------------------------------------------------------------------

@dataclass
class JumpPar:
    """Parameters for ``stochtraj_jump``.

    Attributes
    ----------
    seed:
        Optional seed for the RNG.  When set, the generated trajectory is
        bit-reproducible across runs.  A sequence of ``B`` seeds splits the
        ``nTraj`` trajectories into ``B`` equal consecutive blocks with
        independent streams (see ``DiffusionPar.seed``).
    """
    dt: float                              # time step (s) — required
    nSteps: Optional[int] = None           # number of steps
    nTraj: int = 1                         # number of trajectories
    tMax: Optional[float] = None           # total simulation time (s)
    StatesStart: Optional[np.ndarray] = None  # (nTraj,) starting states (1-based)
    seed: object = None                    # int, SeedSequence, or a sequence of per-block seeds (None = non-deterministic)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def stochtraj_jump(
    TransRates: Optional[np.ndarray] = None,
    TransProb: Optional[np.ndarray] = None,
    Orientations: Optional[np.ndarray] = None,
    par: JumpPar = None,
    *,
    statesOnly: bool = False,
) -> tuple[np.ndarray, ...]:
    """Generate Markovian jump trajectories via kinetic Monte Carlo.

    Either ``TransRates`` or ``TransProb`` must be provided (not both).
    ``TransRates`` takes precedence if both are given.

    Parameters
    ----------
    TransRates:
        Transition rate matrix, shape ``(nStates, nStates)``.
        Off-diagonal elements are positive rates; diagonal elements
        are negative (row sums = 0).
    TransProb:
        Transition probability matrix, shape ``(nStates, nStates)``.
        Rows sum to 1.  Ignored if ``TransRates`` is given.
    Orientations:
        Euler angles for each state, shape ``(nStates, 3)`` or
        ``(3, nStates)``.  Required unless ``statesOnly=True``.
    par:
        Simulation parameters.
    statesOnly:
        If True, only return ``(t, stateTraj)`` without computing
        quaternion/rotation-matrix trajectories.

    Returns
    -------
    t:
        Time axis, shape ``(nSteps,)``.
    RTraj:
        Rotation matrix trajectories, shape ``(3, 3, nSteps, nTraj)``.
        Only returned if ``statesOnly=False``.
    qTraj:
        Quaternion trajectories, shape ``(4, nSteps, nTraj)``.
        Only returned if ``statesOnly=False``.
    stateTraj:
        State trajectories, shape ``(nSteps, nTraj)``.  1-based state
        indices.
    """
    if par is None:
        raise ValueError("JumpPar must be provided.")

    dt = par.dt
    nTraj = par.nTraj

    # --- Build transition probability matrix ---
    if TransRates is not None:
        TRM = np.asarray(TransRates, dtype=np.float64)
        if TRM.ndim != 2 or TRM.shape[0] != TRM.shape[1]:
            raise ValueError("TransRates must be a square matrix.")
        # Validate: positive off-diag, negative diag, rows sum to 0
        if np.any(np.diag(TRM) > 0):
            raise ValueError("TransRates diagonal must be non-positive.")
        off_diag = TRM - np.diag(np.diag(TRM))
        if np.any(off_diag < 0):
            raise ValueError("TransRates off-diagonal elements must be non-negative.")
        if np.any(np.abs(TRM.sum(axis=1)) / max(np.abs(TRM).max(), 1e-30) > 1e-10):
            raise ValueError("TransRates rows must sum to zero.")
        TPM = expm(dt * TRM)
    elif TransProb is not None:
        TPM = np.asarray(TransProb, dtype=np.float64)
        if TPM.ndim != 2 or TPM.shape[0] != TPM.shape[1]:
            raise ValueError("TransProb must be a square matrix.")
        if np.any(TPM < 0):
            raise ValueError("TransProb elements must be non-negative.")
        if np.any(np.abs(TPM.sum(axis=1) - 1.0) > 1e-10):
            raise ValueError("TransProb rows must sum to 1.")
    else:
        raise ValueError("Either TransRates or TransProb must be provided.")

    nStates = TPM.shape[0]

    # --- Orientations ---
    if not statesOnly:
        if Orientations is None:
            raise ValueError(
                f"Orientations for {nStates} states are required. "
                f"Provide a ({nStates}, 3) array, or set statesOnly=True."
            )
        Oris = np.asarray(Orientations, dtype=np.float64)
        if Oris.shape == (3, nStates):
            Oris = Oris.T  # → (nStates, 3)
        if Oris.shape != (nStates, 3):
            raise ValueError(
                f"Orientations must be (nStates, 3) or (3, nStates), "
                f"got {Oris.shape}."
            )
        # Pre-compute quaternions for each state
        qStates = np.zeros((4, nStates), dtype=np.float64)
        for i in range(nStates):
            qStates[:, i] = _euler2quat_active(*Oris[i])

    # --- Time parameters ---
    nSteps = par.nSteps
    if nSteps is None:
        if par.tMax is not None:
            nSteps = int(np.ceil(par.tMax / dt))
        else:
            nSteps = 500

    # --- Seeded RNG (reproducibility per JumpPar.seed) ---
    _gens = traj_generators(par.seed, nTraj)

    # --- Starting states ---
    if par.StatesStart is not None:
        starts = np.asarray(par.StatesStart, dtype=int).ravel()
        if starts.size == 1 and nTraj > 1:
            starts = np.full(nTraj, starts[0], dtype=int)
    else:
        starts = draw(_gens, lambda g, n: g.integers(1, nStates + 1, size=n), axis=0)

    # --- Cumulative transition probability matrix ---
    cumulTPM = np.cumsum(TPM, axis=1)

    # --- Run kinetic Monte Carlo ---
    stateTraj = np.zeros((nSteps, nTraj), dtype=int)
    stateTraj[0, :] = starts

    # each trajectory draws its uniform numbers from its block's generator, in order
    traj_rng = [g for g, n in _gens for _ in range(n)]
    for iTraj in range(nTraj):
        u = traj_rng[iTraj].random(nSteps)
        state = stateTraj[0, iTraj]
        for iStep in range(1, nSteps):
            # Find first state where cumulative prob exceeds random number
            state = np.searchsorted(cumulTPM[state - 1], u[iStep - 1]) + 1
            state = min(state, nStates)  # clamp
            stateTraj[iStep, iTraj] = state

    t = np.linspace(0, nSteps * dt, nSteps)

    if statesOnly:
        return t, stateTraj

    # --- Convert state trajectories to quaternion/rotation trajectories ---
    qTraj = np.zeros((4, nSteps, nTraj), dtype=np.float64)
    for iTraj in range(nTraj):
        for iStep in range(nSteps):
            state = stateTraj[iStep, iTraj]
            qTraj[:, iStep, iTraj] = qStates[:, state - 1]  # 1-based → 0-based

    RTraj = _quat_to_rotmat_batch(qTraj)

    return t, RTraj, qTraj, stateTraj
