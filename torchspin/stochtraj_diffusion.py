"""Brownian rotational diffusion trajectory generator.

Port of MATLAB EasySpin's ``stochtraj_diffusion.m`` and
``stochtraj_proprottraj.m``.  Generates stochastic rotational trajectories
via Euler-Maruyama integration of the Langevin equation in quaternion space.

References
----------
[1] Sezer et al., J. Chem. Phys. 128, 165106 (2008)
    https://doi.org/10.1063/1.2908075
[2] Leimkuhler, Appl. Math. Res. Express 2013, 34 (2013)
    https://doi.org/10.1093/amrx/abs010
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from torchspin._rng import draw, traj_generators
from torchspin.rotutils import euler2quat, quat2rotmat, quatmult


# ---------------------------------------------------------------------------
# Dataclass for diffusion parameters
# ---------------------------------------------------------------------------

@dataclass
class DiffusionPar:
    """Parameters for ``stochtraj_diffusion``.

    Attributes
    ----------
    seed:
        Optional seed for the random number generator.  When set, the
        generated trajectory is bit-reproducible across runs.  When ``None``
        (default), a fresh non-deterministic generator is used.  A sequence
        of ``B`` seeds splits the ``nTraj`` trajectories into ``B`` equal
        consecutive blocks with independent streams, so a block draws the
        same numbers whether it is generated alone or together with others
        (used by cardamom to keep seeded results independent of how
        orientations are grouped).
    """
    dt: Optional[float] = None        # time step (s)
    nSteps: Optional[int] = None      # number of steps
    nTraj: int = 1                     # number of trajectories
    tMax: Optional[float] = None      # total simulation time (s)
    OriStart: Optional[np.ndarray] = None  # (3, nTraj) Euler angles
    Integrator: str = 'Euler-Maruyama'     # or 'Leimkuhler-Matthews'
    seed: object = None                    # int, SeedSequence, or a sequence of per-block seeds (None = non-deterministic)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def stochtraj_diffusion(
    Diff: np.ndarray,
    par: DiffusionPar,
    *,
    Potential: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Brownian rotational diffusion trajectories.

    Parameters
    ----------
    Diff:
        Rotational diffusion tensor principal values (s⁻¹).
        Shape ``(3,)`` (rhombic), ``(2,)`` (axial → [Dperp, Dpar]),
        or scalar (isotropic).
    par:
        Simulation parameters (time step, number of steps, trajectories).
    Potential:
        Optional orienting potential.  If a 2D array with 4 columns,
        interpreted as ``[L, M, K, lambda]`` rows (Wigner function
        expansion).  If a 3D array, interpreted as a numerical potential
        on an (alpha, beta, gamma) grid.

    Returns
    -------
    t:
        Time axis, shape ``(nSteps,)``.
    RTraj:
        Rotation matrix trajectories, shape ``(3, 3, nSteps, nTraj)``.
    qTraj:
        Quaternion trajectories, shape ``(4, nSteps, nTraj)``.
    """
    # --- Normalize diffusion tensor to (3,) ---
    Diff = np.atleast_1d(np.asarray(Diff, dtype=np.float64)).ravel()
    if Diff.size == 1:
        Diff = np.full(3, Diff[0])
    elif Diff.size == 2:
        Diff = np.array([Diff[0], Diff[0], Diff[1]])
    elif Diff.size != 3:
        raise ValueError("Diff must have 1, 2, or 3 elements.")

    tcorr = 1.0 / (6.0 * Diff)

    nSteps = par.nSteps
    dt = par.dt
    nTraj = par.nTraj

    # --- Seeded RNG (reproducibility per DiffusionPar.seed) ---
    _gens = traj_generators(par.seed, nTraj)

    # --- Resolve time parameters ---
    if dt is None and par.tMax is not None and nSteps is not None:
        dt = par.tMax / nSteps
    elif dt is None:
        dt = float(np.min(tcorr)) / 10.0
    if nSteps is None:
        if par.tMax is not None:
            nSteps = int(np.ceil(par.tMax / dt))
        else:
            nSteps = int(np.ceil(200.0 * np.max(tcorr) / dt))

    # --- Parse potential ---
    LMK = None
    lam = None
    interpGrad = None

    if Potential is not None:
        Potential = np.asarray(Potential)
        if Potential.ndim == 2 and Potential.shape[1] == 4:
            # Wigner function expansion [L, M, K, lambda]
            LMK = Potential[:, :3].astype(int)
            lam = Potential[:, 3].astype(complex)
        elif Potential.ndim == 3:
            # Numerical potential on grid — compute gradient interpolant
            interpGrad = _build_numeric_gradient(Potential)
        else:
            raise ValueError(
                "Potential must be an (N,4) array of [L,M,K,lambda] rows "
                "or a 3D grid array."
            )

    # --- Starting orientations ---
    if par.OriStart is not None:
        OriStart = np.asarray(par.OriStart, dtype=np.float64)
        if OriStart.ndim == 1:
            OriStart = OriStart[:, np.newaxis]
        if OriStart.shape[1] == 1 and nTraj > 1:
            OriStart = np.tile(OriStart, (1, nTraj))
    else:
        if Potential is None:
            # Spiral grid starting orientations
            pts = np.linspace(-1, 1, nTraj)
            gridTheta = np.arccos(pts)
            gridPhi = np.sqrt(np.pi * nTraj) * np.arcsin(pts)
            gridPsi = np.sqrt(np.pi * nTraj) * np.arcsin(pts)
            OriStart = np.stack([gridPhi, gridTheta, gridPsi], axis=0)
        else:
            # Random starting orientations for potential case
            OriStart = draw(_gens, lambda g, n: np.stack([
                g.uniform(0, 2 * np.pi, n),
                np.arccos(g.uniform(-1, 1, n)),
                g.uniform(0, 2 * np.pi, n),
            ], axis=0), axis=1)

    # Convert starting Euler angles → quaternions
    q0 = np.zeros((4, nTraj), dtype=np.float64)
    for k in range(nTraj):
        alpha, beta, gamma = OriStart[:, k]
        q0[:, k] = _euler2quat_active(alpha, beta, gamma)

    # --- Initialize quaternion trajectories ---
    qTraj = np.zeros((4, nTraj, nSteps), dtype=np.float64)
    qTraj[:, :, 0] = q0

    # --- Propagate ---
    qTraj = _propagate(qTraj, Diff, dt, nSteps, nTraj,
                       par.Integrator, LMK, lam, interpGrad, gens=_gens)

    # Permute to (4, nSteps, nTraj) — matches MATLAB output convention
    qTraj = np.transpose(qTraj, (0, 2, 1))  # (4, nSteps, nTraj)

    # Convert to rotation matrices
    t = np.linspace(0, nSteps * dt, nSteps)
    RTraj = _quat_to_rotmat_batch(qTraj)  # (3, 3, nSteps, nTraj)

    return t, RTraj, qTraj


# ---------------------------------------------------------------------------
# Core propagation kernel  (port of stochtraj_proprottraj.m)
# ---------------------------------------------------------------------------

def _propagate(
    q: np.ndarray,        # (4, nTraj, nSteps)
    Diff: np.ndarray,     # (3,)
    dt: float,
    nSteps: int,
    nTraj: int,
    integrator: str,
    LMK: np.ndarray | None,
    lam: np.ndarray | None,
    interpGrad: object | None,
    rng: Optional[np.random.Generator] = None,
    gens: Optional[list] = None,
) -> np.ndarray:
    """Euler-Maruyama quaternion propagation (Eq. 61 in Sezer 2008)."""
    # Pre-compute random angular steps (seeded via rng if provided)
    if gens is None:
        gens = [(rng if rng is not None else np.random.default_rng(), nTraj)]
    randns = draw(gens, lambda g, n: g.standard_normal((3, n, nSteps)), axis=1)
    scale = np.sqrt(2.0 * Diff * dt)[:, np.newaxis, np.newaxis]

    if integrator == 'Leimkuhler-Matthews':
        rand_steps = scale * (randns[:, :, :-1] + randns[:, :, 1:]) / 2.0
        # Pad last step (not used — loop ends at nSteps-1)
        rand_steps = np.concatenate(
            [rand_steps, np.zeros((3, nTraj, 1))], axis=2
        )
    else:
        rand_steps = scale * randns

    has_wigner_pot = LMK is not None and lam is not None
    has_numeric_pot = interpGrad is not None

    for iStep in range(1, nSteps):
        qLast = q[:, :, iStep - 1]  # (4, nTraj)
        curr_rand = rand_steps[:, :, iStep]  # (3, nTraj)

        if has_wigner_pot:
            torque = _calc_aniso_torque(LMK, lam, qLast)
            ang_step = torque * (Diff[:, np.newaxis] * dt) + curr_rand
        elif has_numeric_pot:
            alpha, beta, gamma = _quat2euler_active(qLast)
            grad = interpGrad(alpha, beta, gamma)  # (3, nTraj)
            torque = -grad
            ang_step = torque * (Diff[:, np.newaxis] * dt) + curr_rand
        else:
            ang_step = curr_rand

        # Angular step → quaternion increment (Eq. 48 in Sezer 2008)
        theta = np.sqrt(np.sum(ang_step**2, axis=0, keepdims=True))  # (1, nTraj)
        theta_safe = np.where(theta < 1e-30, 1e-30, theta)
        ux = ang_step[0:1] / theta_safe
        uy = ang_step[1:2] / theta_safe
        uz = ang_step[2:3] / theta_safe

        st = np.sin(theta / 2.0)
        ct = np.cos(theta / 2.0)

        q1, q2, q3, q4 = qLast[0:1], qLast[1:2], qLast[2:3], qLast[3:4]

        q[0, :, iStep] = (q1 * ct - q2 * ux * st - q3 * uy * st - q4 * uz * st).ravel()
        q[1, :, iStep] = (q2 * ct + q1 * ux * st - q4 * uy * st + q3 * uz * st).ravel()
        q[2, :, iStep] = (q3 * ct + q4 * ux * st + q1 * uy * st - q2 * uz * st).ravel()
        q[3, :, iStep] = (q4 * ct - q3 * ux * st + q2 * uy * st + q1 * uz * st).ravel()

    return q


# ---------------------------------------------------------------------------
# Wigner-function torque (port of stochtraj_calcanistorque)
# ---------------------------------------------------------------------------

def _calc_aniso_torque(
    LMK: np.ndarray,
    lam: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Compute anisotropic torque from Wigner potential on quaternions.

    This is a simplified implementation for the most common case:
    axial potential with L=2, M=0, K=0.  For general LMK terms,
    the torque is computed via numerical differentiation of the
    Wigner D-function expansion.

    Parameters
    ----------
    LMK:
        Array of shape ``(nTerms, 3)`` with L, M, K values.
    lam:
        Complex lambda coefficients, shape ``(nTerms,)``.
    q:
        Quaternions, shape ``(4, nTraj)``.

    Returns
    -------
    torque:
        Torque vector, shape ``(3, nTraj)``.
    """
    nTraj = q.shape[1]
    torque = np.zeros((3, nTraj), dtype=np.float64)

    # Convert quaternions to Euler angles for gradient
    alpha, beta, gamma = _quat2euler_active(q)

    eps = 1e-6
    for dim, angles_name in enumerate(['alpha', 'beta', 'gamma']):
        # Numerical gradient of the potential w.r.t. each Euler angle
        U_plus = np.zeros(nTraj, dtype=np.float64)
        U_minus = np.zeros(nTraj, dtype=np.float64)

        a_p, b_p, g_p = alpha.copy(), beta.copy(), gamma.copy()
        a_m, b_m, g_m = alpha.copy(), beta.copy(), gamma.copy()

        if dim == 0:
            a_p += eps; a_m -= eps
        elif dim == 1:
            b_p += eps; b_m -= eps
        else:
            g_p += eps; g_m -= eps

        for i, (L, M, K) in enumerate(LMK):
            L, M, K = int(L), int(M), int(K)
            lam_i = lam[i]
            U_plus += np.real(-lam_i * _wignerd_small(L, M, K, a_p, b_p, g_p))
            U_minus += np.real(-lam_i * _wignerd_small(L, M, K, a_m, b_m, g_m))

        torque[dim] = -(U_plus - U_minus) / (2.0 * eps)

    return torque


def _wignerd_small(L: int, M: int, K: int,
                   alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray
                   ) -> np.ndarray:
    """Compute Wigner D^L_{MK}(alpha, beta, gamma).

    Uses the small-d formula with explicit phase factors.
    For M=0, K=0 this reduces to the Legendre polynomial P_L(cos(beta)).
    """
    # D^L_{MK} = exp(-i*M*alpha) * d^L_{MK}(beta) * exp(-i*K*gamma)
    d = _small_d(L, M, K, beta)
    return np.exp(-1j * M * alpha) * d * np.exp(-1j * K * gamma)


def _small_d(L: int, M: int, K: int, beta: np.ndarray) -> np.ndarray:
    """Wigner small-d matrix element d^L_{MK}(beta).

    Uses the explicit sum formula:
    d^L_{MK}(beta) = sum_s (-1)^s * sqrt(...) * cos^a * sin^b / factorials
    """
    from math import factorial

    cb2 = np.cos(beta / 2.0)
    sb2 = np.sin(beta / 2.0)

    s_min = max(0, K - M)
    s_max = min(L + K, L - M)

    result = np.zeros_like(beta, dtype=np.float64)
    prefactor = np.sqrt(
        factorial(L + M) * factorial(L - M) *
        factorial(L + K) * factorial(L - K)
    )

    for s in range(s_min, s_max + 1):
        denom = (factorial(L + K - s) * factorial(s) *
                 factorial(M - K + s) * factorial(L - M - s))
        exp_c = 2 * L + K - M - 2 * s
        exp_s = M - K + 2 * s
        sign = (-1) ** (M - K + s)
        result += sign * cb2**exp_c * sb2**exp_s / denom

    return prefactor * result


# ---------------------------------------------------------------------------
# Numerical potential gradient
# ---------------------------------------------------------------------------

def _build_numeric_gradient(potential_grid: np.ndarray):
    """Build gradient interpolant from 3D potential grid.

    Parameters
    ----------
    potential_grid:
        3D array of shape ``(n_alpha, n_beta, n_gamma)`` defining the
        orienting potential over Euler angles.

    Returns
    -------
    interp_fn:
        Callable ``(alpha, beta, gamma) -> (3, N)`` gradient array.
    """
    from scipy.interpolate import RegularGridInterpolator

    na, nb, ng = potential_grid.shape
    alpha_grid = np.linspace(0, 2 * np.pi, na)
    beta_grid_full = np.linspace(0, np.pi, nb + 2)
    beta_grid = beta_grid_full[1:-1]  # avoid poles
    gamma_grid = np.linspace(0, 2 * np.pi, ng)

    # Compute gradient via np.gradient
    da = np.gradient(potential_grid, alpha_grid, axis=0)
    db = np.gradient(potential_grid, beta_grid, axis=1)
    dg = np.gradient(potential_grid, gamma_grid, axis=2)

    interp_a = RegularGridInterpolator(
        (alpha_grid, beta_grid, gamma_grid), da, method='linear',
        bounds_error=False, fill_value=0.0
    )
    interp_b = RegularGridInterpolator(
        (alpha_grid, beta_grid, gamma_grid), db, method='linear',
        bounds_error=False, fill_value=0.0
    )
    interp_g = RegularGridInterpolator(
        (alpha_grid, beta_grid, gamma_grid), dg, method='linear',
        bounds_error=False, fill_value=0.0
    )

    def grad_fn(alpha, beta, gamma):
        pts = np.stack([alpha, beta, gamma], axis=-1)
        return np.stack([interp_a(pts), interp_b(pts), interp_g(pts)], axis=0)

    return grad_fn


# ---------------------------------------------------------------------------
# Quaternion ↔ Euler conversion (active convention, matching MATLAB)
# ---------------------------------------------------------------------------

def _euler2quat_active(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Convert Euler angles (z-y'-z'') to quaternion (active convention)."""
    ca, sa = np.cos(alpha / 2), np.sin(alpha / 2)
    cb, sb = np.cos(beta / 2), np.sin(beta / 2)
    cg, sg = np.cos(gamma / 2), np.sin(gamma / 2)

    q = np.array([
        ca * cb * cg - sa * cb * sg,
        ca * sb * sg - sa * sb * cg,
        ca * sb * cg + sa * sb * sg,
        ca * cb * sg + sa * cb * cg,
    ])
    return q


def _quat2euler_active(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert quaternion(s) to Euler angles (active, z-y'-z'').

    Parameters
    ----------
    q:
        Quaternions, shape ``(4,)`` or ``(4, N)``.

    Returns
    -------
    alpha, beta, gamma:
        Euler angles in radians.
    """
    if q.ndim == 1:
        q = q[:, np.newaxis]

    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

    beta = np.arccos(np.clip(q0**2 - q1**2 - q2**2 + q3**2, -1, 1))

    sb = np.sin(beta)
    safe = np.abs(sb) > 1e-10

    alpha = np.where(
        safe,
        np.arctan2(q2 * q3 + q0 * q1, -q1 * q3 + q0 * q2),
        np.arctan2(2 * (q0 * q3 + q1 * q2), q0**2 + q1**2 - q2**2 - q3**2),
    )
    gamma = np.where(
        safe,
        np.arctan2(q2 * q3 - q0 * q1, q1 * q3 + q0 * q2),
        0.0,
    )

    return alpha, beta, gamma


# ---------------------------------------------------------------------------
# Batch quaternion → rotation matrix
# ---------------------------------------------------------------------------

def _quat_to_rotmat_batch(q: np.ndarray) -> np.ndarray:
    """Convert quaternion trajectories to rotation matrices.

    Parameters
    ----------
    q:
        Quaternion array, shape ``(4, nSteps, nTraj)``.

    Returns
    -------
    R:
        Rotation matrices, shape ``(3, 3, nSteps, nTraj)``.
    """
    import torch
    qt = torch.from_numpy(np.ascontiguousarray(q, dtype=np.float64))
    q0, q1, q2, q3 = qt[0], qt[1], qt[2], qt[3]
    R = torch.empty((3, 3) + tuple(q0.shape), dtype=torch.float64)
    R[0, 0] = 1 - 2 * (q2 * q2 + q3 * q3)
    R[0, 1] = 2 * (q1 * q2 - q0 * q3)
    R[0, 2] = 2 * (q1 * q3 + q0 * q2)
    R[1, 0] = 2 * (q1 * q2 + q0 * q3)
    R[1, 1] = 1 - 2 * (q1 * q1 + q3 * q3)
    R[1, 2] = 2 * (q2 * q3 - q0 * q1)
    R[2, 0] = 2 * (q1 * q3 - q0 * q2)
    R[2, 1] = 2 * (q2 * q3 + q0 * q1)
    R[2, 2] = 1 - 2 * (q1 * q1 + q2 * q2)
    return R.numpy()
