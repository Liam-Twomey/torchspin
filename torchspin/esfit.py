"""
Least-squares fitting of EPR spectra.

This module provides functionality for fitting experimental EPR data by
optimizing simulation parameters to minimize residuals.
"""

import dataclasses
import time
import warnings

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, Any
from scipy.optimize import minimize, least_squares, OptimizeResult


__all__ = ['esfit', 'FitOptions', 'FitResult', 'TorchSpinParameterHandler']


@dataclass
class FitOptions:
    """Options for esfit optimization.
    
    Attributes
    ----------
    method : str
        Optimization algorithm. Options:
          'global'    — particle swarm (global) → simplex (local polish);
                        recommended for derivative EPR with rugged loss
          'simplex'   — Nelder-Mead (local; fast but easy to get trapped)
          'lbfgsb'    — L-BFGS-B (local, gradient via finite diff)
          'powell'    — Powell's method (local, derivative-free)
          'grid'      — systematic grid search
          'montecarlo'— random sampling + simulated annealing
          'genetic'   — genetic algorithm
          'swarm'     — particle swarm only (no polish)
          'trf'       — bounded trust-region reflective least squares on the
                        residual vector (SciPy least_squares; see trf_* below)
          'levmar'    — Levenberg-Marquardt
        Default: 'simplex'
    autoscale : str
        Scaling method for fit. Options: 'lsq' (least squares),
        'maxabs' (maximum absolute value), 'none' (no scaling).
        Default: 'lsq'. The 'lsq' scale is constrained to be non-negative,
        as in EasySpin, so the model must carry the sign of the signal: fit
        an inversion-recovery curve with ``-exp(-t/T)``, not ``exp(-t/T)``.
    lsq_negative_scale : str
        What 'lsq' autoscaling does when the unconstrained scale comes out
        negative (model anti-correlated with the data). 'clip' (default)
        returns the constrained least-squares optimum: scale 0 with the
        baseline refitted alone. 'abs' reproduces EasySpin's
        ``coeffs(1) = abs(coeffs(1))``, which mirrors the model but keeps the
        baseline solved for the opposite sign, so the objective is not a
        least-squares fit of anything. esfit warns if the final fit has a
        negative unconstrained scale.
    baseline : int
        Baseline correction polynomial order. -1 for no baseline,
        0 for constant, 1 for linear, 2 for quadratic, 3 for cubic.
        Default: -1
    target : str
        Target transformation applied to residuals before computing RMSD.
        Options:
          'auto' — pick 'int' for derivative-like data (mean << std, e.g.
                   field-modulated CW EPR with Harmonic=1), else 'fcn'.
                   Matches MATLAB EasySpin's pepper/garlic auto-default.
          'fcn'  — raw residuals (sim - data).
          'int'  — cumulative-integral residuals (smooths derivative-EPR
                   loss landscape and removes oscillating local minima).
          'diff' — first-difference residuals.
        Default: 'auto'
    max_iter : int
        Maximum number of iterations. Default: 2000
    tol_fun : float
        Function tolerance for convergence. Default: 1e-6
    tol_x : float
        Parameter tolerance for convergence. Default: 1e-6
    verbosity : int
        Verbosity level. 0=silent (default), 1=phase headers (swarm
        start/end, polish start), 2=per-iteration RMSD printed to stdout.
    callback : Optional[Callable]
        If not None, called each iteration of population-based methods
        (swarm, genetic, montecarlo, global) with a dict payload
        ``{"method": str, "iter": int, "max_iter": int, "best_f": float,
        "best_x": np.ndarray}``. Use to drive live plots or progress
        bars from the caller. Exceptions raised in the callback are
        swallowed so they can never crash the fit. Default: None.
    compute_uncertainties : bool
        Whether to compute parameter uncertainties. Default: True
    trf_diff_step, trf_x_scale, trf_gradient_tol
        Options for ``method='trf'`` (SciPy ``least_squares(method='trf')``,
        bounded trust-region reflective least squares). The solver minimizes
        the same target-transformed residual vector whose RMSD the other
        methods minimize, so esfit's parameter mapping (dict/array ``p0``,
        fixed parameters, ``lb``/``ub``/``vary``), amplitude and baseline
        handling (``autoscale``/``baseline``), ``target``, progress reporting
        and stopping (``max_time``, ``stop_when``, interrupt) apply unchanged.
          trf_diff_step    — relative finite-difference step for the Jacobian
                             (SciPy ``diff_step``; None = SciPy's default,
                             about 1.5e-8). Relative to the physical parameter
                             values, unlike ``lm_delta``.
          trf_x_scale      — SciPy ``x_scale``: 'jac' (default; scale from the
                             Jacobian columns), a float, or an array with one
                             entry per *active* (non-fixed) parameter.
          trf_gradient_tol — SciPy ``gtol`` (default 1e-7). ``tol_fun`` and
                             ``tol_x`` are passed as ``ftol`` and ``xtol``.
        Evaluation budget: for 'trf' ``max_iter`` is SciPy's ``max_nfev``, the
        number of *step* residual evaluations. The evaluations SciPy spends on
        the finite-difference Jacobian (about one per active parameter per
        iteration) do not count against that budget, but they are counted in
        ``FitResult.n_evaluations`` and drive the progress/stop checks. An
        exhausted budget returns ``success=False`` (SciPy's message "The
        maximum number of function evaluations is exceeded.") with
        ``interrupted=False``; a stop or interrupt returns the best point so
        far with ``interrupted=True``. ``FitResult.n_iterations`` is SciPy's
        Jacobian-evaluation count (one per trust-region iteration).
        TRF is a local optimization method, not a CPU-acceleration feature:
        it runs sequentially in the calling process. Simulation threading
        (``torch.set_num_threads``) and the population-worker parallelism of
        ``n_workers`` (population methods only) are separate and unaffected.
    """
    method: str = 'simplex'
    autoscale: str = 'lsq'
    lsq_negative_scale: str = 'clip'
    baseline: int = -1
    target: str = 'auto'
    max_iter: int = 2000
    tol_fun: float = 1e-6
    tol_x: float = 1e-6
    verbosity: int = 0
    callback: Optional[Callable] = None
    compute_uncertainties: bool = True
    # Grid search options
    grid_points: int = 10
    # Monte Carlo / simulated annealing options
    mc_samples: int = 1000
    mc_temp_start: float = 1.0
    mc_temp_end: float = 1e-4
    # Genetic algorithm options
    ga_population: int = 50
    ga_mutation: float = 0.1
    ga_crossover: float = 0.7
    ga_elite: int = 5
    # Particle swarm options
    swarm_size: int = 30
    swarm_inertia: float = 0.7
    swarm_cognitive: float = 1.4
    swarm_social: float = 1.4
    # Levenberg-Marquardt options
    lm_lambda: float = 1e-3
    lm_delta: float = 1e-3        # EasySpin FitOpt.delta (finite-difference step in [-1,1] units)
    lm_gradient_tol: float = 1e-5   # EasySpin FitOpt.Gradient (= TolFun)
    lm_step_tol: float = 1e-8
    # Trust-region reflective least-squares options. max_iter is the solver's
    # max_nfev budget for this method (finite-difference probes are additional).
    # diff_step is relative to the physical parameters, as in scipy.least_squares.
    trf_diff_step: Optional[float] = None
    trf_x_scale: Union[str, float, np.ndarray] = 'jac'
    trf_gradient_tol: float = 1e-7
    # Parallelism for population-based methods (swarm, genetic, montecarlo,
    # grid, global). n_workers=1 is sequential; an int > 1 or 'auto' (one per
    # core) evaluates the population in a spawned process pool with one torch
    # thread per worker (workers='auto'|'processes'; 'threads' forces a thread
    # pool, which gains little because the forward model holds the GIL).
    # Models defined in a notebook cell need cloudpickle (pip install
    # cloudpickle) to reach the workers; otherwise a thread pool is used.
    n_workers: Union[int, str] = 1
    workers: str = 'auto'
    # Seed for numpy RNGs in population-based optimizers. None (default)
    # = non-deterministic (each run produces a different particle/sample
    # realization). An int (e.g. seed=42) makes the fit bit-reproducible
    # across runs — recommended for tutorials, regression tests, and
    # manuscript figures where the numbers must be citable.
    seed: Optional[int] = None
    # Real-time progress and early stop (all methods).
    #   progress: None (silent) | 'text' (one flushed line per interval/every N
    #     evaluations: iteration, evaluations, current and best RMSD, elapsed,
    #     evaluations/s, ETA where the budget is known) | 'bar' (tqdm bar when tqdm
    #     is importable, else text) | callable(info: dict) — return False to stop.
    #   A resource summary (torch threads, worker processes, CPUs, device, RSS)
    #   is printed once at the start and a totals line at the end.
    #   With n_workers > 1 updates arrive per evaluation chunk, not per evaluation.
    #   max_time: wall-clock budget in seconds; stop_when: predicate(info) -> True
    #   stops.  A stop (or a KeyboardInterrupt / kernel interrupt) returns the
    #   best-so-far FitResult with success=False and interrupted=True.
    progress: Union[None, str, Callable] = None
    progress_every: Optional[int] = None
    progress_interval: float = 1.0
    max_time: Optional[float] = None
    stop_when: Optional[Callable] = None


@dataclass
class FitResult:
    """Results from esfit optimization.
    
    Attributes
    ----------
    pfit : np.ndarray
        Fitted parameter values (active parameters only)
    pnames : list[str]
        Parameter names
    p_start : np.ndarray
        Starting parameter values
    p_fixed : np.ndarray
        Boolean mask indicating fixed parameters
    fit : np.ndarray
        Fitted spectrum (scaled + baseline corrected)
    fit_raw : np.ndarray
        Raw simulation output (before scaling)
    residuals : np.ndarray
        Residuals (fit - data, before the ``target`` transform)
    scale : float
        Scale factor applied to simulation
    baseline : np.ndarray
        Baseline vector
    rmsd : float
        Root mean square deviation
    ssr : float
        Sum of squared residuals
    n_params : int
        Number of fitted parameters
    n_data : int
        Number of data points
    algorithm : str
        Algorithm used
    n_iterations : int
        Number of iterations
    success : bool
        Whether fit converged successfully
    message : str
        Convergence message
    pstd : Optional[np.ndarray]
        Parameter standard deviations
    ci95 : Optional[np.ndarray]
        95% confidence intervals (n_params x 2)
    cov : Optional[np.ndarray]
        Covariance matrix
    corr : Optional[np.ndarray]
        Correlation matrix
    r_squared : Optional[float]
        Coefficient of determination (R²)
    reduced_chi_square : Optional[float]
        Reduced chi-square statistic
    noise_std : Optional[float]
        Estimated noise standard deviation
    """
    pfit: np.ndarray
    pnames: list[str]
    p_start: np.ndarray
    p_fixed: np.ndarray
    fit: np.ndarray
    fit_raw: np.ndarray
    residuals: np.ndarray
    scale: float
    baseline: np.ndarray
    rmsd: float
    ssr: float
    n_params: int
    n_data: int
    algorithm: str
    n_iterations: int
    success: bool
    message: str
    pstd: Optional[np.ndarray] = None
    ci95: Optional[np.ndarray] = None
    cov: Optional[np.ndarray] = None
    corr: Optional[np.ndarray] = None
    r_squared: Optional[float] = None
    reduced_chi_square: Optional[float] = None
    noise_std: Optional[float] = None
    n_evaluations: Optional[int] = None      # objective evaluations (all methods, all phases)
    elapsed_s: Optional[float] = None        # optimizer wall time (excludes the initial RMSD evaluation)
    interrupted: bool = False                # True when stopped early (interrupt, max_time, stop_when, callback)


class TorchSpinParameterHandler:
    """Handle dict-style parameters for torchspin simulators.
    
    Supports fitting parameters in SpinSystem and Experiment objects by
    specifying which fields to vary.
    
    Examples
    --------
    >>> from torchspin import SpinSystem, Experiment
    >>> sys0 = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
    >>> exp = Experiment(mwFreq=9.5, Range=[330, 350])
    >>> p0 = {'Sys': sys0, 'Exp': exp}
    >>> vary = {'Sys': {'g': [[0.01, 0.01, 0.01]]}}
    >>> handler = TorchSpinParameterHandler(p0, vary)
    """
    
    def __init__(
        self,
        p0: dict,
        vary: Optional[dict] = None,
        lb: Optional[dict] = None,
        ub: Optional[dict] = None
    ):
        """
        Initialize handler for dict-style parameters.
        
        Parameters
        ----------
        p0 : dict
            Initial parameter values, e.g. {'Sys': SpinSystem(...), 'Exp': Experiment(...)}
        vary : dict, optional
            Variation specification, e.g. {'Sys': {'g': [[0.01, 0.01, 0.01]]}}
        lb, ub : dict, optional
            Explicit bounds (mutually exclusive with vary)
        """
        from torchspin import SpinSystem, Experiment
        
        self.p0_dict = p0
        self.vary_dict = vary
        self.lb_dict = lb
        self.ub_dict = ub
        
        # Parse vary/bounds specification
        if vary is not None and (lb is not None or ub is not None):
            raise ValueError("Cannot specify both 'vary' and 'lb'/'ub'")
        
        # Build parameter map: list of (key, field, indices, value, lb, ub)
        self.param_map = []
        self.pnames = []
        
        if vary is not None:
            self._parse_vary()
        elif lb is not None or ub is not None:
            self._parse_bounds()
        else:
            raise ValueError("Must specify either 'vary' or 'lb'/'ub'")
        
        # Create flat parameter vector
        self.n_params = len(self.param_map)
        self.p0_vec = np.array([entry['value'] for entry in self.param_map])
        self.lb = np.array([entry['lb'] for entry in self.param_map])
        self.ub = np.array([entry['ub'] for entry in self.param_map])
        
        # Check for fixed parameters
        self.fixed_mask = self.lb == self.ub
        self.active_mask = ~self.fixed_mask
        self.n_active = np.sum(self.active_mask)
        
        if self.n_active == 0:
            raise ValueError("No variable parameters - all parameters are fixed (lb == ub)")
        
        self.p0_active = self.p0_vec[self.active_mask]
        self.lb_active = self.lb[self.active_mask]
        self.ub_active = self.ub[self.active_mask]
    
    # Fields that cannot be negative, so that p0 - vary is clipped at zero for
    # them rather than handing the optimizer an unphysical lower bound (EasySpin
    # esfit.m: nonnegFieldNames).  Only reachable in this dict style, where the
    # field name is known; the plain-vector style has no names to go on, which
    # is also true of EasySpin.
    _NONNEGATIVE_FIELDS = frozenset({
        'lw', 'lwpp', 'lwEndor', 'HStrain', 'gStrain', 'DStrain', 'AStrain',
        'weight', 'initState', 'tcorr', 'Diff', 'T1', 'T2',
    })

    def _lower_bound(self, field: str, value: float, vary: float) -> float:
        lb = float(value) - float(vary)
        if field in self._NONNEGATIVE_FIELDS and lb < 0.0:
            return 0.0
        return lb

    def _parse_vary(self):
        """Parse vary specification to build parameter map."""
        for key, vary_spec in self.vary_dict.items():
            if key not in self.p0_dict:
                raise ValueError(f"Key '{key}' in vary not found in p0")
            
            obj = self.p0_dict[key]
            
            # Iterate over fields to vary
            for field, vary_val in vary_spec.items():
                if not hasattr(obj, field):
                    raise ValueError(f"Object '{key}' has no field '{field}'")
                
                p0_val = getattr(obj, field)
                
                # Convert torch tensors to numpy
                if isinstance(p0_val, torch.Tensor):
                    p0_val = p0_val.cpu().numpy()
                if isinstance(vary_val, torch.Tensor):
                    vary_val = vary_val.cpu().numpy()
                
                # Handle different value types
                if isinstance(p0_val, (list, np.ndarray)):
                    p0_array = np.asarray(p0_val)
                    vary_array = np.asarray(vary_val)
                    
                    if p0_array.shape != vary_array.shape:
                        raise ValueError(
                            f"{key}.{field}: p0 shape {p0_array.shape} != vary shape {vary_array.shape}"
                        )
                    
                    # Flatten and add each element
                    p0_flat = p0_array.ravel()
                    vary_flat = vary_array.ravel()
                    
                    for i, (p0_elem, vary_elem) in enumerate(zip(p0_flat, vary_flat)):
                        # Compute multi-dimensional index
                        idx = np.unravel_index(i, p0_array.shape)
                        idx_str = ','.join(map(str, idx))
                        
                        self.param_map.append({
                            'key': key,
                            'field': field,
                            'index': idx,
                            'value': float(p0_elem),
                            'lb': self._lower_bound(field, p0_elem, vary_elem),
                            'ub': float(p0_elem + vary_elem)
                        })
                        self.pnames.append(f"{key}.{field}[{idx_str}]")
                
                elif isinstance(p0_val, (int, float, np.number)):
                    # Scalar parameter
                    vary_scalar = float(vary_val)
                    self.param_map.append({
                        'key': key,
                        'field': field,
                        'index': None,
                        'value': float(p0_val),
                        'lb': self._lower_bound(field, p0_val, vary_scalar),
                        'ub': float(p0_val + vary_scalar)
                    })
                    self.pnames.append(f"{key}.{field}")
                
                else:
                    raise TypeError(
                        f"{key}.{field}: unsupported type {type(p0_val)}"
                    )
    
    def _parse_bounds(self):
        """Parse explicit lb/ub bound dictionaries.

        Both ``lb`` and ``ub`` must be given, mirror each other in structure,
        and reference fields present in ``p0`` — same layout as ``vary``,
        e.g. ``lb={'Sys': {'g': [[1.9, 1.9, 1.9]]}}``.
        """
        if self.lb_dict is None or self.ub_dict is None:
            raise ValueError(
                "Dict-style bounds require both 'lb' and 'ub'.")
        if set(self.lb_dict.keys()) != set(self.ub_dict.keys()):
            raise ValueError("'lb' and 'ub' must have the same keys.")

        for key, lb_spec in self.lb_dict.items():
            if key not in self.p0_dict:
                raise ValueError(f"Key '{key}' in lb/ub not found in p0")
            ub_spec = self.ub_dict[key]
            if set(lb_spec.keys()) != set(ub_spec.keys()):
                raise ValueError(
                    f"'{key}': lb and ub must specify the same fields.")
            obj = self.p0_dict[key]

            for field, lb_val in lb_spec.items():
                if not hasattr(obj, field):
                    raise ValueError(f"Object '{key}' has no field '{field}'")
                ub_val = ub_spec[field]
                p0_val = getattr(obj, field)

                if isinstance(p0_val, torch.Tensor):
                    p0_val = p0_val.cpu().numpy()
                if isinstance(lb_val, torch.Tensor):
                    lb_val = lb_val.cpu().numpy()
                if isinstance(ub_val, torch.Tensor):
                    ub_val = ub_val.cpu().numpy()

                if isinstance(p0_val, (list, np.ndarray)):
                    p0_array = np.asarray(p0_val, dtype=float)
                    lb_array = np.asarray(lb_val, dtype=float)
                    ub_array = np.asarray(ub_val, dtype=float)
                    if (p0_array.shape != lb_array.shape
                            or p0_array.shape != ub_array.shape):
                        raise ValueError(
                            f"{key}.{field}: p0 shape {p0_array.shape} != "
                            f"lb shape {lb_array.shape} / ub shape "
                            f"{ub_array.shape}")
                    p0_flat = p0_array.ravel()
                    lb_flat = lb_array.ravel()
                    ub_flat = ub_array.ravel()
                    for i, (p0_e, lb_e, ub_e) in enumerate(
                            zip(p0_flat, lb_flat, ub_flat)):
                        if lb_e > ub_e:
                            raise ValueError(
                                f"{key}.{field}: lb > ub at element {i}")
                        if not (lb_e <= p0_e <= ub_e):
                            raise ValueError(
                                f"{key}.{field}: p0 value {p0_e} outside "
                                f"[{lb_e}, {ub_e}] at element {i}")
                        idx = np.unravel_index(i, p0_array.shape)
                        idx_str = ','.join(map(str, idx))
                        self.param_map.append({
                            'key': key,
                            'field': field,
                            'index': idx,
                            'value': float(p0_e),
                            'lb': float(lb_e),
                            'ub': float(ub_e),
                        })
                        self.pnames.append(f"{key}.{field}[{idx_str}]")

                elif isinstance(p0_val, (int, float, np.number)):
                    lb_s, ub_s = float(lb_val), float(ub_val)
                    if lb_s > ub_s:
                        raise ValueError(f"{key}.{field}: lb > ub")
                    if not (lb_s <= float(p0_val) <= ub_s):
                        raise ValueError(
                            f"{key}.{field}: p0 value {p0_val} outside "
                            f"[{lb_s}, {ub_s}]")
                    self.param_map.append({
                        'key': key,
                        'field': field,
                        'index': None,
                        'value': float(p0_val),
                        'lb': lb_s,
                        'ub': ub_s,
                    })
                    self.pnames.append(f"{key}.{field}")

                else:
                    raise TypeError(
                        f"{key}.{field}: unsupported type {type(p0_val)}")
    
    def to_vector(self, params: Union[np.ndarray, dict]) -> np.ndarray:
        """Convert parameters to active parameter vector."""
        if isinstance(params, np.ndarray):
            return params[self.active_mask]
        
        # Extract from dict
        pvec = np.zeros(self.n_params)
        for i, entry in enumerate(self.param_map):
            obj = params[entry['key']]
            val = getattr(obj, entry['field'])
            
            if entry['index'] is not None:
                # Convert to numpy array to handle tensor/list indexing
                val_array = np.asarray(val)
                pvec[i] = val_array[entry['index']]
            else:
                pvec[i] = val
        
        return pvec[self.active_mask]
    
    def to_dict(self, pvec_active: np.ndarray) -> dict:
        """Convert active parameter vector to dict structure."""
        import copy
        import torch
        from torchspin import SpinSystem, Experiment
        
        # Expand active vector to full vector
        pvec_full = self.p0_vec.copy()
        pvec_full[self.active_mask] = pvec_active
        
        # Deep copy p0 dict
        result = copy.deepcopy(self.p0_dict)
        
        # Update fields
        for i, entry in enumerate(self.param_map):
            obj = result[entry['key']]
            field = entry['field']
            new_val = pvec_full[i]
            
            # Current value in the copy being built (type preserved: tensor,
            # list, float). It must be the copy, not p0: reading p0 for every
            # element made each element of an array field overwrite the ones
            # set before it, so only its last element ever reached the model.
            original_val = getattr(obj, field)

            if entry['index'] is not None:
                # Array parameter - update single element
                if isinstance(original_val, torch.Tensor):
                    # Convert to numpy, modify, convert back to tensor
                    arr = original_val.detach().cpu().numpy().copy()
                    arr[entry['index']] = new_val
                    arr_tensor = torch.tensor(arr, dtype=original_val.dtype, device=original_val.device)
                    setattr(obj, field, arr_tensor)
                else:
                    arr = np.array(original_val, dtype=float)
                    arr[entry['index']] = new_val
                    setattr(obj, field, arr.tolist() if isinstance(original_val, list) else arr)
            else:
                # Scalar parameter
                if isinstance(original_val, torch.Tensor):
                    # Convert scalar to tensor with original dtype/device
                    setattr(obj, field, torch.tensor(float(new_val), dtype=original_val.dtype, device=original_val.device))
                else:
                    setattr(obj, field, float(new_val))
        
        return result
    
    def to_full(self, pvec_active: np.ndarray) -> np.ndarray:
        """Convert active parameter vector to full parameter vector."""
        pvec_full = self.p0_vec.copy()
        pvec_full[self.active_mask] = pvec_active
        return pvec_full
    
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return bounds for active parameters."""
        return self.lb_active, self.ub_active
    
    def get_names(self) -> list[str]:
        """Return names of active parameters."""
        return [name for i, name in enumerate(self.pnames) if self.active_mask[i]]


class ParameterHandler:
    """Manages parameter conversions and bounds for optimization.
    
    Handles conversion between dict-style and vector-style parameters,
    and enforces parameter bounds during optimization.
    """
    
    def __init__(
        self,
        p0: Union[np.ndarray, dict],
        vary: Optional[Union[np.ndarray, dict]] = None,
        lb: Optional[Union[np.ndarray, dict]] = None,
        ub: Optional[Union[np.ndarray, dict]] = None
    ):
        """
        Initialize parameter handler.
        
        Parameters
        ----------
        p0 : array or dict
            Initial parameter values
        vary : array or dict, optional
            Allowed variation (p0 ± vary). Mutually exclusive with lb/ub.
        lb, ub : array or dict, optional
            Lower and upper bounds. Mutually exclusive with vary.
        """
        self.is_dict_style = isinstance(p0, dict)
        
        if self.is_dict_style:
            # Use TorchSpinParameterHandler for dict-style
            from torchspin.esfit import TorchSpinParameterHandler
            self._handler = TorchSpinParameterHandler(p0, vary, lb, ub)
            # Expose interface
            self.p0 = self._handler.p0_vec
            self.n_params = self._handler.n_params
            self.n_active = self._handler.n_active
            self.lb = self._handler.lb
            self.ub = self._handler.ub
            self.fixed_mask = self._handler.fixed_mask
            self.active_mask = self._handler.active_mask
            self.p0_active = self._handler.p0_active
            self.lb_active = self._handler.lb_active
            self.ub_active = self._handler.ub_active
            self.pnames = self._handler.pnames
            return
        
        # Array style
        self.p0 = np.asarray(p0, dtype=float)
        self.n_params = len(self.p0)
        
        # Parse bounds
        if vary is not None and (lb is not None or ub is not None):
            raise ValueError("Cannot specify both 'vary' and 'lb'/'ub'")
        
        if vary is not None:
            vary = np.asarray(vary, dtype=float)
            if len(vary) != self.n_params:
                raise ValueError(f"vary length {len(vary)} != p0 length {self.n_params}")
            self.lb = self.p0 - vary
            self.ub = self.p0 + vary
        elif lb is not None or ub is not None:
            if lb is None:
                lb = np.full(self.n_params, -np.inf)
            if ub is None:
                ub = np.full(self.n_params, np.inf)
            self.lb = np.asarray(lb, dtype=float)
            self.ub = np.asarray(ub, dtype=float)
            if len(self.lb) != self.n_params or len(self.ub) != self.n_params:
                raise ValueError("Bounds length must match p0 length")
        else:
            raise ValueError(
                "No bounds specified. Provide 'vary' (±variation), or "
                "'lb'/'ub' (explicit bounds). Implicit ±50% defaults "
                "are unreliable for parameters near zero or negative."
            )
        
        # Identify fixed parameters (lb == ub)
        self.fixed_mask = self.lb == self.ub
        self.active_mask = ~self.fixed_mask
        self.n_active = np.sum(self.active_mask)
        
        if self.n_active == 0:
            raise ValueError("No variable parameters - all parameters are fixed (lb == ub)")
        
        # Extract active parameters
        self.p0_active = self.p0[self.active_mask]
        self.lb_active = self.lb[self.active_mask]
        self.ub_active = self.ub[self.active_mask]
        
        # Parameter names
        self.pnames = [f'p{i}' for i in range(self.n_params)]
    
    def to_vector(self, params: Union[np.ndarray, dict]) -> np.ndarray:
        """Convert parameters to active parameter vector."""
        if self.is_dict_style:
            return self._handler.to_vector(params)
        p = np.asarray(params, dtype=float)
        return p[self.active_mask]
    
    def to_full(self, pvec_active: np.ndarray) -> np.ndarray:
        """Convert active parameter vector to full parameter vector."""
        if self.is_dict_style:
            return self._handler.to_full(pvec_active)
        p_full = self.p0.copy()
        p_full[self.active_mask] = pvec_active
        return p_full
    
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return bounds for active parameters."""
        return self.lb_active, self.ub_active
    
    def get_names(self) -> list[str]:
        """Return names of active parameters."""
        return [name for i, name in enumerate(self.pnames) if self.active_mask[i]]


class ModelWrapper:
    """Wraps model function for consistent interface.
    
    Handles both custom functions (taking parameter vector) and
    torchspin simulators (taking Sys, Exp structures).
    """
    
    def __init__(
        self,
        model_fn: Callable,
        p0: Union[np.ndarray, dict],
        exp_data_len: int
    ):
        """
        Initialize model wrapper.
        
        Parameters
        ----------
        model_fn : callable
            Simulation function
        p0 : array or dict
            Initial parameters (determines calling style)
        exp_data_len : int
            Expected output length for validation
        """
        self.model_fn = model_fn
        self.data_len = exp_data_len
        self.is_dict_style = isinstance(p0, dict)
        
        # Detect if this is a torchspin simulator
        if self.is_dict_style:
            self.is_torchspin = self._detect_torchspin_simulator(model_fn)
        else:
            self.is_torchspin = False
    
    def _detect_torchspin_simulator(self, fn: Callable) -> bool:
        """Check if function is a torchspin simulator (pepper, garlic, etc.)."""
        # Check function name
        if hasattr(fn, '__name__'):
            name = fn.__name__
            if name in ['pepper', 'garlic', 'chili', 'salt']:
                return True
        
        # Check module
        if hasattr(fn, '__module__'):
            if fn.__module__ and 'torchspin' in fn.__module__:
                return True
        
        return False
    
    @staticmethod
    def _garlic_uses_binning(sys, exp) -> bool:
        """Whether garlic's default accumulation for ``sys``/``exp`` is
        'binning': not in the fast-motion regime (any component with a
        positive correlation time) and no mwPhase (which needs 'explicit')."""
        if exp is not None and float(getattr(exp, 'mwPhase', 0.0) or 0.0) != 0.0:
            return False
        for c in (sys if isinstance(sys, (list, tuple)) else [sys]):
            for name, tf in (('logtcorr', lambda v: 10.0 ** v), ('tcorr', lambda v: v)):
                v = getattr(c, name, None)
                if v is not None:
                    v0 = float(torch.as_tensor(v, dtype=torch.float64).detach().reshape(-1)[0])
                    if tf(v0) > 0:
                        return False
                    break
        return True

    def _fd_friendly_options(self, opt, sys=None, exp=None):
        """For garlic, replace the default nearest-bin accumulation by the
        two-bin linear split when the user left ``Opt.AccumMethod`` unset.

        Nearest-bin sticks make the spectrum piecewise-constant in g and A, so
        every finite-difference Jacobian (levmar, trf, uncertainties) is zero
        for them. 'linear' is the form garlic itself uses on the autograd graph.
        Fast-motion and mwPhase runs default to 'explicit' and are left alone.
        """
        if getattr(self.model_fn, '__name__', '') != 'garlic':
            return opt
        if not self._garlic_uses_binning(sys, exp):
            return opt
        from torchspin.experiment import Options
        if opt is None:
            return Options(AccumMethod='linear')
        if getattr(opt, 'AccumMethod', None) is None:
            return dataclasses.replace(opt, AccumMethod='linear')
        return opt

    def __call__(self, pvec_active: np.ndarray, param_handler: ParameterHandler) -> np.ndarray:
        """
        Evaluate model with active parameter vector.
        
        Parameters
        ----------
        pvec_active : np.ndarray
            Active parameter values
        param_handler : ParameterHandler
            Parameter handler for conversion
        
        Returns
        -------
        sim : np.ndarray
            Simulated data vector (1D)
        """
        if self.is_dict_style:
            # Convert to dict and call torchspin simulator
            params_dict = param_handler._handler.to_dict(pvec_active)
            
            if self.is_torchspin:
                # Torchspin simulator: call with (Sys, Exp) or (Sys, Exp, Opt)
                sys = params_dict.get('Sys')
                exp = params_dict.get('Exp')
                opt = params_dict.get('Opt', None)
                opt = self._fd_friendly_options(opt, sys, exp)

                if opt is not None:
                    B, spec = self.model_fn(sys, exp, opt)
                else:
                    B, spec = self.model_fn(sys, exp)
                
                # Extract spectrum
                output = spec
            else:
                # Custom function expecting dict
                output = self.model_fn(params_dict)
        else:
            # Convert to full parameter vector
            p_full = param_handler.to_full(pvec_active)
            
            # Call model function
            output = self.model_fn(p_full)
        
        # Convert output to numpy array
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        output = np.asarray(output, dtype=float).ravel()
        
        # Validate output length
        if len(output) != self.data_len:
            raise ValueError(
                f"Model output length {len(output)} != data length {self.data_len}"
            )
        
        return output


def _baseline_design(n: int, baseline_order: int) -> np.ndarray:
    """Polynomial baseline columns ``[1, x, x², ...]`` on ``x = linspace(0, 1, n)``."""
    x = np.linspace(0, 1, n)
    return np.column_stack([x**k for k in range(baseline_order + 1)])


def _fit_polynomial_baseline(resid: np.ndarray, baseline_order: int) -> np.ndarray:
    """Least-squares polynomial baseline of order ``baseline_order`` fitted to
    ``resid`` (zero vector for ``baseline_order < 0``)."""
    n = len(resid)
    if baseline_order < 0:
        return np.zeros(n)
    P = _baseline_design(n, baseline_order)
    coeffs, _, _, _ = np.linalg.lstsq(P, resid, rcond=None)
    return P @ coeffs


def _fit_scale_baseline_lsq(
    sim: np.ndarray,
    data: np.ndarray,
    baseline_order: int,
    negative_scale: str = 'clip',
    return_negative: bool = False,
):
    """
    Fit a non-negative scale and a polynomial baseline by least squares.

    Solves ``min ||scale·sim + P·b − data||`` subject to ``scale ≥ 0``, where
    ``P = [1, x, x², ...]``. The scale is forced non-negative, as in EasySpin,
    because a free sign lets the optimizer "fit" by mirroring the model, which
    creates spurious minima on derivative spectra.

    Parameters
    ----------
    sim : np.ndarray
        Simulated data
    data : np.ndarray
        Experimental data
    baseline_order : int
        Polynomial order for baseline (-1 for none)
    negative_scale : {'clip', 'abs'}
        What to do when the unconstrained solve gives a negative scale.
        ``'clip'`` (default) returns the constrained optimum: ``scale = 0``
        with the baseline refitted alone. ``'abs'`` is EasySpin's
        ``coeffs(1) = abs(coeffs(1))``: it mirrors the scale but keeps the
        baseline solved for the opposite sign, which is not a least-squares
        fit of anything.
    return_negative : bool
        Also return whether the unconstrained scale was negative.

    Returns
    -------
    scale : float
        Optimal non-negative scale factor
    baseline : np.ndarray
        Baseline vector
    negative : bool
        Only with ``return_negative=True``.
    """
    if negative_scale not in ('clip', 'abs'):
        raise ValueError(f"lsq_negative_scale must be 'clip' or 'abs', got {negative_scale!r}")
    n = len(sim)

    if baseline_order < 0:
        scale = float((sim @ data) / (sim @ sim + 1e-12))
        baseline = np.zeros(n)
        negative = scale < 0
        if negative:
            scale = abs(scale) if negative_scale == 'abs' else 0.0
    else:
        P = _baseline_design(n, baseline_order)
        D = np.column_stack([sim, P])
        coeffs, _, _, _ = np.linalg.lstsq(D, data, rcond=None)
        scale = float(coeffs[0])
        negative = scale < 0
        if not negative:
            baseline = P @ coeffs[1:]
        elif negative_scale == 'abs':
            # EasySpin parity (esfit.m: coeffs(1) = abs(coeffs(1)))
            scale = abs(scale)
            baseline = P @ coeffs[1:]
        else:
            # One sign-constrained coefficient in a convex problem: the
            # constrained optimum sits on the boundary scale = 0, with the
            # baseline refitted on its own.
            scale = 0.0
            baseline = _fit_polynomial_baseline(data, baseline_order)

    if return_negative:
        return scale, baseline, negative
    return scale, baseline


def _apply_target_transform(residuals: np.ndarray, target: str) -> np.ndarray:
    """Apply target transformation to residuals."""
    if target == 'fcn':
        return residuals
    elif target == 'int':
        return np.cumsum(residuals)
    elif target == 'diff':
        return np.diff(residuals, prepend=0)
    else:
        raise ValueError(f"Unknown target: '{target}'. Use 'fcn', 'int', or 'diff'")


def compute_residuals(
    pvec_active: np.ndarray,
    data: np.ndarray,
    model_wrapper: ModelWrapper,
    param_handler: ParameterHandler,
    options: FitOptions
) -> tuple[np.ndarray, float, dict]:
    """
    Compute residuals and RMSD for given parameters.
    
    Parameters
    ----------
    pvec_active : np.ndarray
        Active parameter values
    data : np.ndarray
        Experimental data
    model_wrapper : ModelWrapper
        Model function wrapper
    param_handler : ParameterHandler
        Parameter handler
    options : FitOptions
        Fitting options
    
    Returns
    -------
    residuals : np.ndarray
        Residuals after scaling/baseline/target transform
    rmsd : float
        Root mean square deviation
    info : dict
        Additional info (sim, scale, baseline, fit, etc.)
    """
    # 1. Evaluate model
    sim = model_wrapper(pvec_active, param_handler)
    
    # 2. Apply autoscaling
    negative_scale = False
    if options.autoscale == 'lsq':
        scale, baseline, negative_scale = _fit_scale_baseline_lsq(
            sim, data, options.baseline,
            negative_scale=options.lsq_negative_scale, return_negative=True)
        fit = scale * sim + baseline
    elif options.autoscale == 'maxabs':
        scale = np.max(np.abs(data)) / (np.max(np.abs(sim)) + 1e-12)
        # polynomial baseline fitted to what the scaled model leaves over
        baseline = _fit_polynomial_baseline(data - scale * sim, options.baseline)
        fit = scale * sim + baseline
    else:  # 'none'
        scale = 1.0
        baseline = _fit_polynomial_baseline(data - sim, options.baseline)
        fit = sim + baseline
    
    # 3. Compute residuals
    residuals_raw = fit - data
    
    # 4. Apply target transformation
    residuals = _apply_target_transform(residuals_raw, options.target)
    
    # 5. Compute RMSD
    rmsd = np.sqrt(np.mean(residuals**2))
    
    info = {
        'sim': sim,
        'fit': fit,
        'scale': scale,
        'baseline': baseline,
        'residuals_raw': residuals_raw,
        'negative_scale': negative_scale,
    }
    
    return residuals, rmsd, info


def _objective_function(
    pvec_active: np.ndarray,
    data: np.ndarray,
    model_wrapper: ModelWrapper,
    param_handler: ParameterHandler,
    options: FitOptions
) -> float:
    """Objective function for optimization (returns scalar RMSD)."""
    try:
        # Clip parameters to bounds
        lb, ub = param_handler.get_bounds()
        pvec_active_clipped = np.clip(pvec_active, lb, ub)
        
        _, rmsd, _ = compute_residuals(
            pvec_active_clipped, data, model_wrapper, param_handler, options
        )
        return rmsd
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        # Recoverable errors (out-of-range parameters, convergence failures)
        if options.verbosity >= 1:
            print(f"Warning: simulation error at current parameters: {e}")
        return 1e10


def _optimize_simplex(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions
) -> OptimizeResult:
    """
    Run Nelder-Mead simplex optimization.

    Matches MATLAB EasySpin's ``esfit_simplex.m`` behavior:
      * Initial simplex edge length = ``delta * (ub - lb)`` with delta=0.1
        in original space, which corresponds to 0.2 in the transformed
        [-1, 1] space (the previous 0.05 was 4x too small and caused
        premature convergence to local minima).
      * For n <= 2 parameters, classical NM coefficients [1, 2, 0.5, 0.5]
        (scipy's ``adaptive=True`` reduces to these for n=2 anyway, so we
        leave it on for n>=3 where Gao-Han adaptive params help).
      * Termination: scipy's NM requires BOTH ``xatol`` and ``fatol`` to
        be satisfied (matches EasySpin's dual-criterion stop), so we map
        ``xatol = TolEdgeLength = 1e-4`` and ``fatol = options.tol_fun``.
    """
    lb, ub = bounds
    n_params = len(p0)

    def transform(x):
        return 2 * (x - lb) / (ub - lb + 1e-12) - 1

    def untransform(x_scaled):
        return lb + (ub - lb) * (x_scaled + 1) / 2

    p0_scaled = transform(p0)

    def obj_scaled(x_scaled):
        x = untransform(x_scaled)
        return objective_fn(x)

    # Initial simplex: vertex i (for i=1..n) perturbs parameter i-1 by
    # delta_t = 0.2 in [-1, 1] space (= EasySpin delta=0.1 * (ub-lb)
    # in original space, since transformed (ub-lb)=2). MATLAB:
    #   v(nParams+1:nParams+1:end) = ... + delta.*(ub-lb).'
    delta_t = 0.2
    initial_simplex = np.zeros((n_params + 1, n_params))
    initial_simplex[0] = p0_scaled
    for i in range(n_params):
        vertex = p0_scaled.copy()
        # Step toward the interior of [-1, 1] so we don't immediately
        # land on the bound when p0 is near ub
        step = delta_t if vertex[i] + delta_t <= 1.0 else -delta_t
        vertex[i] += step
        initial_simplex[i + 1] = vertex

    # iteration counter only — stopping always comes from _StopFit raised inside the
    # objective, so scipy's per-version callback-halting semantics do not matter
    tracker = _tracker_of(objective_fn)
    result = minimize(
        obj_scaled,
        p0_scaled,
        method='Nelder-Mead',
        callback=lambda *a, **k: tracker.iteration(tracker.iter, None),
        options={
            'maxiter': options.max_iter,
            'xatol': 1e-4,
            'fatol': options.tol_fun,
            'disp': options.verbosity > 0,
            'adaptive': True,
            'initial_simplex': initial_simplex,
        }
    )

    result.x = untransform(result.x)
    result.x = np.clip(result.x, lb, ub)

    return result


def _optimize_lbfgsb(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions
) -> OptimizeResult:
    """Run L-BFGS-B optimization (supports bounds natively)."""
    from scipy.optimize import Bounds
    
    lb, ub = bounds
    tracker = _tracker_of(objective_fn)
    result = minimize(
        objective_fn,
        p0,
        method='L-BFGS-B',
        bounds=Bounds(lb, ub),
        callback=lambda *a, **k: tracker.iteration(tracker.iter, None),
        options={
            'maxiter': options.max_iter,
            'ftol': options.tol_fun,
            'gtol': 1e-5,
            'disp': options.verbosity > 0
        }
    )
    return result


def _optimize_powell(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions
) -> OptimizeResult:
    """Run Powell optimization (derivative-free)."""
    from scipy.optimize import Bounds
    
    lb, ub = bounds
    tracker = _tracker_of(objective_fn)
    result = minimize(
        objective_fn,
        p0,
        method='Powell',
        bounds=Bounds(lb, ub),
        callback=lambda *a, **k: tracker.iteration(tracker.iter, None),
        options={
            'maxiter': options.max_iter,
            'ftol': options.tol_fun,
            'xtol': options.tol_x,
            'disp': options.verbosity > 0
        }
    )
    return result


def _emit_progress(
    options: FitOptions,
    method: str,
    it: int,
    max_iter: int,
    best_f: float,
    best_x: np.ndarray,
) -> None:
    """Emit per-iteration progress when verbosity>=2 or a callback is set.

    Prints a flushed one-line update (safe to run inside Jupyter cells:
    ``flush=True`` makes the message visible while the cell is still
    executing) and invokes ``options.callback(info)`` if provided.
    """
    if options.verbosity >= 2:
        print(f"  [{method}] iter {it+1:4d}/{max_iter}  best RMSD = {best_f:.5e}",
              flush=True)
    if options.callback is not None:
        try:
            options.callback({
                "method": method,
                "iter": it + 1,
                "max_iter": max_iter,
                "best_f": best_f,
                "best_x": best_x.copy(),
            })
        except Exception:
            # Never let a user-supplied callback crash the fit
            pass


class _StopFit(Exception):
    """Raised from the progress tracker to stop a fit early (max_time, stop_when,
    a progress callback returning False).  An Exception subclass on purpose:
    scipy swallows StopIteration, and nothing between the raise site and esfit
    catches plain Exceptions."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _rss_mb() -> Optional[float]:
    """Resident set size in MB (psutil if available, else getrusage)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        try:
            import resource, sys as _sys
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return r / 1e6 if _sys.platform == 'darwin' else r / 1e3
        except Exception:
            return None


def _population_size(options: 'FitOptions', n_params: int) -> Optional[int]:
    m = options.method
    if m in ('swarm', 'global'):
        return int(options.swarm_size)
    if m == 'genetic':
        return int(options.ga_population)
    if m == 'montecarlo':
        return int(options.mc_samples)
    if m == 'grid':
        return int(min(options.max_iter, max(2, options.grid_points) ** max(1, n_params)))
    return None


class _FitProgress:
    """Per-fit progress tracker: evaluation count, best-so-far, timing, rate-limited
    reporting (text / tqdm bar / callable) and the early-stop checks.  Lives in the
    calling process only (never pickled; worker results are merged in the parent)."""

    def __init__(self, options: 'FitOptions', method: str, n_params: int = 0, n_data: int = 0, enabled: bool = True):
        self.options = options
        self.method = method
        self.phase = method
        self.n_params, self.n_data = int(n_params), int(n_data)
        self.n_evaluations = 0
        self.best_f = float('inf')
        self.best_x: Optional[np.ndarray] = None
        self.last_f = float('nan')
        self.iter = 0
        self.iter_total: Optional[int] = None
        self.eval_total: Optional[int] = None
        self.t0: Optional[float] = None
        self.t_last_emit = float('-inf')
        self.n_last_emit = 0
        prog = options.progress if enabled else None
        self.mode = 'callable' if callable(prog) else (prog if prog in ('text', 'bar') else None)
        self.enabled = self.mode is not None
        self.active = self.enabled or options.max_time is not None or options.stop_when is not None
        self._bar = None
        self.stop_reason: Optional[str] = None
        self.interrupted = False
        self.rss0: Optional[float] = None
        self.elapsed_s: Optional[float] = None

    # -- timing --------------------------------------------------------------
    def elapsed(self) -> float:
        return 0.0 if self.t0 is None else time.perf_counter() - self.t0

    def start(self) -> None:
        self.t0 = time.perf_counter()
        self.t_last_emit = self.t0
        if not self.enabled:
            return
        self.rss0 = _rss_mb()
        if self.mode == 'bar':
            try:
                from tqdm.auto import tqdm
                self._bar = tqdm(total=self.iter_total or self.eval_total, desc=f'esfit[{self.phase}]', leave=True)
            except Exception:
                self._bar = None
                self.mode = 'text'
                print('esfit: progress bar unavailable (no tqdm); using text', flush=True)
        if self.mode in ('text', 'bar'):
            print(self._resource_line(), flush=True)

    def _resource_line(self) -> str:
        import os
        import torch as _torch
        o = self.options
        pop = _population_size(o, self.n_params)
        if pop is None:
            workers = 'serial'
        else:
            nw = _resolve_workers(o.n_workers, pop)
            kind = 'threads' if o.workers == 'threads' else 'procs (spawn)'
            workers = f"{nw} {kind}" if nw > 1 else 'serial'
            workers += f" (n_workers={o.n_workers!r})"
        dev = []
        try:
            dev.append('cuda ' + ('yes' if _torch.cuda.is_available() else 'no'))
            if hasattr(_torch.backends, 'mps'):
                dev.append('mps ' + ('yes' if _torch.backends.mps.is_available() else 'no'))
        except Exception:
            pass
        rss = f" | rss {self.rss0:.0f} MB" if self.rss0 is not None else ''
        return (f"esfit[{self.method}] {self.n_params} params, {self.n_data} points | torch threads "
                f"{_torch.get_num_threads()} | workers {workers} | cpus {os.cpu_count()} | {', '.join(dev)}{rss}")

    # -- recording -----------------------------------------------------------
    def record(self, x, f: float, emit: bool = True) -> None:
        self.n_evaluations += 1
        f = float(f)
        self.last_f = f
        if f < self.best_f:
            self.best_f = f
            self.best_x = np.array(x, dtype=float, copy=True)
        if emit:
            self.maybe_emit()
            self.check()

    def record_batch(self, X, F) -> None:
        F = np.asarray(F, dtype=float).ravel()
        if F.size == 0:
            return
        self.n_evaluations += int(F.size)
        i = int(np.argmin(F))
        self.last_f = float(F[i])
        if F[i] < self.best_f:
            self.best_f = float(F[i])
            self.best_x = np.array(np.asarray(X)[i], dtype=float, copy=True)
        self.maybe_emit()
        self.check()

    def iteration(self, it: int, total: Optional[int] = None) -> None:
        self.iter = int(it) + 1
        self.iter_total = None if total is None else int(total)
        if self._bar is not None and self.iter_total is not None:
            self._bar.total = self.iter_total
            self._bar.n = self.iter
            self._bar.refresh()
        self.maybe_emit()
        self.check()

    def set_phase(self, label: str) -> None:
        self.phase = label
        self.iter = 0
        self.iter_total = None
        if self.mode == 'text':
            print(f"  [{label}] phase start at {self.n_evaluations} evaluations, {self.elapsed():.1f} s", flush=True)
        elif self._bar is not None:
            self._bar.set_description(f'esfit[{label}]')

    # -- reporting -----------------------------------------------------------
    def info(self) -> dict:
        el = self.elapsed()
        rate = self.n_evaluations / el if el > 0 else None
        eta = None
        if self.iter_total and self.iter > 0:
            eta = el / self.iter * max(0, self.iter_total - self.iter)
        elif self.eval_total and self.n_evaluations > 0:
            eta = el / self.n_evaluations * max(0, self.eval_total - self.n_evaluations)
        return {'method': self.method, 'phase': self.phase, 'iter': self.iter, 'iter_total': self.iter_total,
                'n_evaluations': self.n_evaluations, 'eval_total': self.eval_total, 'rmsd': self.last_f,
                'best_f': self.best_f, 'best_x': None if self.best_x is None else self.best_x.copy(),
                'elapsed_s': el, 'evals_per_s': rate, 'eta_s': eta}

    def _text_line(self, info: dict) -> str:
        parts = [f"[{info['phase']}]"]
        if info['iter_total']:
            parts.append(f"iter {info['iter']}/{info['iter_total']}")
        elif info['iter']:
            parts.append(f"iter {info['iter']}")
        parts.append(f"eval {info['n_evaluations']}" + (f"/{info['eval_total']}" if info['eval_total'] else ''))
        parts.append(f"rmsd {info['rmsd']:.3e}  best {info['best_f']:.3e}")
        parts.append(f"{info['elapsed_s']:.1f} s")
        if info['evals_per_s']:
            parts.append(f"{info['evals_per_s']:.1f} ev/s")
        if info['eta_s'] is not None:
            parts.append(f"ETA {info['eta_s']:.0f} s")
        return '  ' + '  '.join(parts)

    def maybe_emit(self, force: bool = False) -> None:
        if not self.enabled:
            return
        o = self.options
        now = time.perf_counter()
        due = force
        if o.progress_every and self.n_evaluations - self.n_last_emit >= int(o.progress_every):
            due = True
        if now - self.t_last_emit >= float(o.progress_interval):
            due = True
        if not due:
            return
        self.t_last_emit = now
        self.n_last_emit = self.n_evaluations
        info = self.info()
        if self.mode == 'text':
            print(self._text_line(info), flush=True)
        elif self.mode == 'bar':
            if self._bar is not None:
                if self.iter_total is None:
                    self._bar.total = self.eval_total
                    self._bar.n = self.n_evaluations
                self._bar.set_postfix_str(f"rmsd {info['rmsd']:.3e} best {info['best_f']:.3e}"
                                          + (f" {info['evals_per_s']:.1f} ev/s" if info['evals_per_s'] else ''))
                self._bar.refresh()
        elif self.mode == 'callable':
            try:
                ret = o.progress(info)
            except Exception:
                ret = None
            if ret is False and self.stop_reason is None:
                self.stop_reason = 'progress callback returned False'

    # -- early stop ----------------------------------------------------------
    def check(self) -> None:
        o = self.options
        if self.stop_reason is None:
            if o.max_time is not None and self.t0 is not None and self.elapsed() >= float(o.max_time):
                self.stop_reason = f'max_time {o.max_time} s reached'
            elif o.stop_when is not None:
                try:
                    if o.stop_when(self.info()):
                        self.stop_reason = 'stop_when condition met'
                except Exception:
                    pass
        if self.stop_reason is not None:
            self.interrupted = True
            raise _StopFit(self.stop_reason)

    def synthesize_result(self, reason: str) -> OptimizeResult:
        self.interrupted = True
        self.stop_reason = self.stop_reason or reason
        x = None if self.best_x is None else self.best_x.copy()
        return OptimizeResult(x=x, fun=self.best_f, success=False,
                              message=(f"interrupted after {self.n_evaluations} evaluations, {self.elapsed():.1f} s "
                                       f"({self.stop_reason}); returning best so far"),
                              nit=self.iter, nfev=self.n_evaluations)

    def finish(self) -> None:
        self.elapsed_s = self.elapsed()
        if self._bar is not None:
            try:
                self._bar.close()
            except Exception:
                pass
        if not self.enabled or self.mode == 'callable':
            return
        rate = self.n_evaluations / self.elapsed_s if self.elapsed_s > 0 else 0.0
        rss = _rss_mb()
        rss_s = '' if rss is None or self.rss0 is None else f", rss {rss:.0f} MB ({rss - self.rss0:+.0f} MB)"
        if self.interrupted:
            print(f"esfit[{self.method}] interrupted ({self.stop_reason}) after {self.n_evaluations} evals, "
                  f"{self.elapsed_s:.1f} s: returning best so far, rmsd {self.best_f:.3e}{rss_s}", flush=True)
        else:
            print(f"esfit[{self.method}] done: {self.n_evaluations} evals, {self.elapsed_s:.1f} s, {rate:.1f} ev/s, "
                  f"best rmsd {self.best_f:.3e}{rss_s}", flush=True)


class _TrackedObjective:
    """Objective wrapper used on the serial path: records every evaluation in the
    tracker (count, best-so-far, progress, early stop).  Never pickled — the
    population evaluator unwraps ``.raw`` before shipping work to processes."""

    def __init__(self, raw: Callable, tracker: _FitProgress):
        self.raw = raw
        self.tracker = tracker

    def __call__(self, x):
        f = self.raw(x)
        self.tracker.record(x, f)
        return f

    def __reduce__(self):
        raise TypeError('_TrackedObjective is not picklable; pass .raw to worker processes')


def _tracker_of(fn) -> _FitProgress:
    """The tracker behind an objective, or a disabled one for bare callables."""
    t = getattr(fn, 'tracker', None)
    return t if t is not None else _FitProgress(FitOptions(), 'none', enabled=False)


_POOL = None                # cached spawn pool: (n_proc, threads_per_worker, pool)


def _pool_init(n_threads: int):
    import torch as _torch
    _torch.set_num_threads(n_threads)


def _get_pool(n_proc: int, per: int):
    """Process pool with the 'spawn' start method (fork is not safe once torch's
    thread pool exists), created once and reused across generations/calls."""
    global _POOL
    import multiprocessing as mp
    if _POOL is not None and _POOL[0] == n_proc and _POOL[1] == per:
        return _POOL[2]
    if _POOL is not None:
        _POOL[2].terminate()
    # the children inherit the environment at spawn: cap the BLAS/OpenMP thread
    # pools of numpy/scipy/torch per worker (otherwise every worker starts one
    # thread per core and the pool thrashes)
    from torchspin.parallel import native_thread_env
    with native_thread_env(per):
        pool = mp.get_context('spawn').Pool(n_proc, initializer=_pool_init, initargs=(per,))
    _POOL = (n_proc, per, pool)
    return pool


class _CloudPickled:
    """Carries a non-picklable callable (notebook cell function, closure) to spawned
    workers by value via cloudpickle."""

    def __init__(self, fn):
        import cloudpickle
        self._blob = cloudpickle.dumps(fn)
        self._fn = fn

    def __call__(self, x):
        if self._fn is None:
            import cloudpickle
            self._fn = cloudpickle.loads(self._blob)
        return self._fn(x)

    def __getstate__(self):
        return {'_blob': self._blob, '_fn': None}


def _threads_per_worker(n_workers: int) -> int:
    """Inner threads per outer worker so that workers × threads fits the core budget."""
    import torch as _torch
    from torchspin.parallel import cpu_budget
    total = min(_torch.get_num_threads(), cpu_budget())
    return max(1, total // max(1, n_workers))


def _resolve_workers(n_workers, n: int) -> int:
    """'auto' → every core this process may use (affinity / SLURM aware, see
    torchspin.parallel.cpu_budget), capped by the population size; ints pass through."""
    if isinstance(n_workers, str):
        from torchspin.parallel import cpu_budget
        return max(1, min(cpu_budget(), n))
    return int(n_workers or 1)


def _eval_population(
    objective_fn: Callable,
    X: np.ndarray,
    n_workers=1,
    workers: str = 'auto',
) -> np.ndarray:
    """Evaluate ``objective_fn`` on each row of ``X``.

    ``n_workers > 1`` (or ``'auto'`` = one per core) evaluates the rows
    concurrently.  ``workers='processes'`` uses a spawned process pool with one
    torch thread per worker — the effective option for torchspin forward models,
    whose Python-level work holds the GIL.  The objective is pickled by
    value through ``cloudpickle`` (so functions defined in a notebook cell and
    closures work); without cloudpickle, only objectives from importable modules
    can be sent by reference and the rest fall back to a thread pool.
    ``workers='threads'`` forces the thread pool.
    """
    global _POOL
    n = X.shape[0]
    n_workers = _resolve_workers(n_workers, n)
    if n_workers <= 1 or n <= 1:
        return np.array([objective_fn(x) for x in X])
    # the progress tracker stays in this process: workers get the raw objective and
    # the parent merges each chunk of results (count, best-so-far, stop checks)
    raw = objective_fn.raw if isinstance(objective_fn, _TrackedObjective) else objective_fn
    tracker = _tracker_of(objective_fn)
    if workers in ('auto', 'processes'):
        # Ship the objective by value (cloudpickle) whenever possible: pickling by
        # reference succeeds for functions defined in a notebook's __main__ but the
        # spawned worker cannot resolve them, dies before registering the task and
        # the map hangs forever.  Reference pickling only for importable modules.
        fn = None
        try:
            fn = _CloudPickled(raw)
        except Exception:
            import pickle
            mod = getattr(raw, '__module__', None)
            if mod not in (None, '__main__', '__mp_main__'):
                try:
                    pickle.dumps(raw); fn = raw
                except Exception:
                    fn = None
        if fn is not None:
            n_proc = min(n_workers, n)
            per = _threads_per_worker(n_proc)
            pool = _get_pool(n_proc, per)
            step = n if not tracker.active else max(2 * n_proc, int(tracker.options.progress_every or 0))
            out = []
            try:
                for s in range(0, n, step):
                    fc = pool.map(fn, list(X[s:s + step]), chunksize=1)
                    out.extend(fc)
                    tracker.record_batch(X[s:s + step], fc)
            except (KeyboardInterrupt, _StopFit):
                # do not leave the cached pool grinding the interrupted generation
                try:
                    pool.terminate()
                finally:
                    _POOL = None
                raise
            return np.array(out)
        if workers == 'processes':
            import warnings
            warnings.warn("esfit: objective is not picklable and cloudpickle is not installed; using a thread pool")
    from concurrent.futures import ThreadPoolExecutor
    # No per-worker thread_budget here. A smaller inner budget makes _linalg skip its
    # serialized single-threaded LAPACK pool, but the native BLAS/OpenMP thread count
    # is process-wide and cannot be capped per Python thread: each worker then calls
    # LAPACK at the full process width, and n_workers x native threads oversubscribe
    # the cores (measured 2.1x slower at 8 threads for the matrix method). The process
    # backend is the one that partitions cores.
    with ThreadPoolExecutor(max_workers=min(n_workers, n)) as ex:
        F = np.array(list(ex.map(raw, list(X))))
    tracker.record_batch(X, F)
    return F


def _optimize_grid(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions,
) -> OptimizeResult:
    """Systematic grid search over parameter bounds.

    Evaluates the objective on a regular ``grid_points``-point grid per
    dimension, then returns the best point found.  For N parameters the
    grid has ``grid_points^N`` evaluations.  Intended for small N (≤4) or
    low ``grid_points``.
    """
    lb, ub = bounds
    n = len(p0)
    npts = max(2, options.grid_points)

    # Build per-axis grids
    grids = [np.linspace(lb[i], ub[i], npts) for i in range(n)]

    best_x = p0.copy()
    best_f = objective_fn(p0)

    import itertools
    combos = list(itertools.product(*grids))
    if len(combos) > options.max_iter:
        combos = combos[:options.max_iter]
    grid_pts = np.array(combos)
    tracker = _tracker_of(objective_fn)
    tracker.eval_total = len(grid_pts) + 1
    grid_f = _eval_population(objective_fn, grid_pts, options.n_workers, options.workers)
    tracker.eval_total = None
    n_eval = len(grid_pts)

    idx = int(np.argmin(grid_f))
    if grid_f[idx] < best_f:
        best_f = float(grid_f[idx])
        best_x = grid_pts[idx].copy()

    return OptimizeResult(
        x=best_x,
        fun=best_f,
        success=True,
        message=f"Grid search: {n_eval} evaluations, best RMSD={best_f:.4e}",
        nit=n_eval,
        nfev=n_eval,
    )


def _optimize_montecarlo(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions,
) -> OptimizeResult:
    """Monte Carlo / simulated annealing optimizer.

    Draws ``mc_samples`` random parameter sets uniformly within bounds,
    then applies a simple simulated annealing refinement around the best
    point found.  Temperature decreases geometrically from ``mc_temp_start``
    to ``mc_temp_end`` over ``max_iter`` SA steps.
    """
    lb, ub = bounds
    rng = np.random.default_rng(options.seed)

    # Phase 1: random sampling (parallelisable)
    n_random = min(options.mc_samples, options.max_iter)
    best_x = p0.copy()
    best_f = objective_fn(p0)

    samples = rng.uniform(lb, ub, size=(n_random, len(p0)))
    tracker = _tracker_of(objective_fn)
    tracker.eval_total = n_random + 1
    sample_f = _eval_population(objective_fn, samples, options.n_workers, options.workers)
    tracker.eval_total = None
    idx = int(np.argmin(sample_f))
    if sample_f[idx] < best_f:
        best_f = float(sample_f[idx])
        best_x = samples[idx].copy()

    n_eval = n_random + 1

    # Phase 2: simulated annealing refinement around best point
    n_sa = max(1, options.max_iter - n_random)
    T_start = options.mc_temp_start
    T_end = options.mc_temp_end
    T_ratio = (T_end / T_start) ** (1.0 / n_sa) if n_sa > 1 else 1.0

    current_x = best_x.copy()
    current_f = best_f
    T = T_start
    span = (ub - lb)

    for step in range(n_sa):
        # Propose random step proportional to temperature
        delta = rng.normal(0, T * span)
        proposal = np.clip(current_x + delta, lb, ub)
        f = objective_fn(proposal)
        n_eval += 1

        # Accept if better, or probabilistically if worse
        dE = f - current_f
        if dE < 0 or (T > 0 and rng.random() < np.exp(-dE / T)):
            current_x = proposal
            current_f = f
            if f < best_f:
                best_f = f
                best_x = proposal.copy()

        T *= T_ratio

        # Emit per-step progress (stride to keep output readable)
        stride = max(1, n_sa // 20)
        if step % stride == 0 or step == n_sa - 1:
            _emit_progress(options, 'montecarlo', step, n_sa, best_f, best_x)
        tracker.iteration(step, n_sa)

    return OptimizeResult(
        x=best_x,
        fun=best_f,
        success=True,
        message=f"Monte Carlo: {n_eval} evaluations, best RMSD={best_f:.4e}",
        nit=n_eval,
        nfev=n_eval,
    )


def _optimize_genetic(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions,
) -> OptimizeResult:
    """Genetic algorithm with tournament selection and Gaussian mutation.

    Maintains a population of ``ga_population`` candidates.  Each generation:

    1. Evaluate all candidates.
    2. Preserve the top ``ga_elite`` individuals unchanged (elitism).
    3. Fill remaining slots via tournament selection (size 2) + crossover.
    4. Apply Gaussian mutation with probability ``ga_mutation``.

    Continues for ``max_iter`` generations or until ``tol_fun`` is met.
    """
    lb, ub = bounds
    n = len(p0)
    pop_size = max(options.ga_elite + 2, options.ga_population)
    elite = max(1, options.ga_elite)
    mut_rate = options.ga_mutation
    cross_rate = options.ga_crossover
    rng = np.random.default_rng(options.seed)

    # Initialize population: elite[0] = p0, rest = random
    pop = np.empty((pop_size, n))
    pop[0] = np.clip(p0, lb, ub)
    pop[1:] = rng.uniform(lb, ub, size=(pop_size - 1, n))

    fitness = _eval_population(objective_fn, pop, options.n_workers, options.workers)
    n_eval = pop_size
    best_f = float('inf')
    best_x = p0.copy()

    span = ub - lb

    for gen in range(options.max_iter):
        # Sort by fitness
        order = np.argsort(fitness)
        pop = pop[order]
        fitness = fitness[order]

        if fitness[0] < best_f:
            best_f = fitness[0]
            best_x = pop[0].copy()

        _emit_progress(options, 'genetic', gen, options.max_iter,
                       best_f, best_x)
        _tracker_of(objective_fn).iteration(gen, options.max_iter)

        if best_f < options.tol_fun:
            break

        # Build next generation
        new_pop = np.empty_like(pop)
        new_pop[:elite] = pop[:elite]  # elitism

        for i in range(elite, pop_size):
            # Tournament selection (size 2) for two parents
            idx_a = rng.integers(0, pop_size, 2)
            idx_b = rng.integers(0, pop_size, 2)
            pa = pop[idx_a[np.argmin(fitness[idx_a])]]
            pb = pop[idx_b[np.argmin(fitness[idx_b])]]

            # Uniform crossover
            if rng.random() < cross_rate:
                mask = rng.random(n) < 0.5
                child = np.where(mask, pa, pb)
            else:
                child = pa.copy()

            # Gaussian mutation
            for j in range(n):
                if rng.random() < mut_rate:
                    child[j] += rng.normal(0, 0.05 * span[j])
            child = np.clip(child, lb, ub)
            new_pop[i] = child

        # Evaluate new individuals (population eval, possibly parallel)
        new_fit = _eval_population(objective_fn, new_pop[elite:], options.n_workers, options.workers)
        fitness[elite:] = new_fit
        n_eval += pop_size - elite

        pop = new_pop

        if n_eval >= options.max_iter * pop_size:
            break

    return OptimizeResult(
        x=best_x,
        fun=best_f,
        success=True,
        message=f"Genetic: {gen+1} generations, {n_eval} evaluations, best RMSD={best_f:.4e}",
        nit=gen + 1,
        nfev=n_eval,
    )


def _optimize_swarm(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions,
) -> OptimizeResult:
    """Particle Swarm Optimization (PSO).

    Port of EasySpin's ``esfit_swarm.m``.

    Maintains a swarm of ``swarm_size`` particles.  Each iteration updates
    velocities via cognitive and social terms, then moves particles.

    Parameters
    ----------
    w  : ``swarm_inertia`` (default 0.7)  — inertia weight
    c1 : ``swarm_cognitive`` (default 1.4) — personal-best attraction
    c2 : ``swarm_social``   (default 1.4) — global-best attraction
    """
    lb, ub = bounds
    n = len(p0)
    swarm_n = max(4, options.swarm_size)
    w = options.swarm_inertia
    c1 = options.swarm_cognitive
    c2 = options.swarm_social
    rng = np.random.default_rng(options.seed)
    span = ub - lb

    # Initialize positions and velocities
    pos = rng.uniform(lb, ub, size=(swarm_n, n))
    pos[0] = np.clip(p0, lb, ub)  # seed with initial guess
    vel = rng.uniform(-span, span, size=(swarm_n, n)) * 0.1

    fitness = _eval_population(objective_fn, pos, options.n_workers, options.workers)
    n_eval = swarm_n

    pbest_pos = pos.copy()
    pbest_fit = fitness.copy()

    gbest_idx = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = float(pbest_fit[gbest_idx])

    for it in range(options.max_iter):
        r1 = rng.uniform(0, 1, (swarm_n, n))
        r2 = rng.uniform(0, 1, (swarm_n, n))

        vel = (w * vel +
               c1 * r1 * (pbest_pos - pos) +
               c2 * r2 * (gbest_pos - pos))

        # Clamp velocity to span to prevent explosion
        vel = np.clip(vel, -span, span)
        pos = np.clip(pos + vel, lb, ub)

        fitness = _eval_population(objective_fn, pos, options.n_workers, options.workers)
        n_eval += swarm_n

        # Update personal bests
        improved = fitness < pbest_fit
        pbest_pos[improved] = pos[improved]
        pbest_fit[improved] = fitness[improved]

        # Update global best
        cur_best_idx = int(np.argmin(pbest_fit))
        if pbest_fit[cur_best_idx] < gbest_fit:
            gbest_fit = float(pbest_fit[cur_best_idx])
            gbest_pos = pbest_pos[cur_best_idx].copy()

        _emit_progress(options, 'swarm', it, options.max_iter,
                       gbest_fit, gbest_pos)
        _tracker_of(objective_fn).iteration(it, options.max_iter)

        if gbest_fit < options.tol_fun:
            break

    return OptimizeResult(
        x=gbest_pos,
        fun=gbest_fit,
        success=True,
        message=f"Swarm: {it+1} iterations, {n_eval} evaluations, best RMSD={gbest_fit:.4e}",
        nit=it + 1,
        nfev=n_eval,
    )


def _optimize_global(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions,
) -> OptimizeResult:
    """Global + local pipeline.

    Phase 1: particle swarm to escape local minima and locate the basin
             of the global optimum (population eval is parallelisable
             via ``options.n_workers``).
    Phase 2: Nelder-Mead simplex polish from the swarm's best point.

    This is the recommended default for derivative-EPR fitting where the
    raw loss landscape has many narrow local minima of width ~ linewidth.
    The global phase finds the right basin; the local phase refines.
    """
    from dataclasses import replace

    # Phase 1: swarm — budget half the iterations and reuse swarm options.
    swarm_iter = max(20, options.max_iter // 2)
    swarm_opts = replace(options, max_iter=swarm_iter)
    if options.verbosity >= 1:
        print(f"  [global] phase 1 — swarm ({swarm_iter} iterations, "
              f"{options.swarm_size} particles)", flush=True)
    tracker = _tracker_of(objective_fn)
    tracker.set_phase('global/swarm')
    swarm_res = _optimize_swarm(objective_fn, p0, bounds, swarm_opts)

    # Phase 2: simplex polish from the swarm's best point.
    polish_iter = max(200, options.max_iter - swarm_res.nfev)
    polish_opts = replace(options, max_iter=polish_iter)
    if options.verbosity >= 1:
        print(f"  [global] phase 2 — simplex polish (best after swarm: "
              f"{swarm_res.fun:.5e})", flush=True)
    tracker.set_phase('global/simplex')
    polish_res = _optimize_simplex(objective_fn, swarm_res.x, bounds, polish_opts)

    final_x = polish_res.x if polish_res.fun < swarm_res.fun else swarm_res.x
    final_f = min(polish_res.fun, swarm_res.fun)
    n_eval = swarm_res.nfev + polish_res.nfev

    return OptimizeResult(
        x=final_x,
        fun=final_f,
        success=True,
        message=(f"Global: swarm {swarm_res.nfev} evals -> {swarm_res.fun:.4e}, "
                 f"polish {polish_res.nfev} evals -> {polish_res.fun:.4e}"),
        nit=swarm_res.nit + polish_res.nit,
        nfev=n_eval,
    )


def _optimize_levmar(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions,
    resid_fn: Optional[Callable] = None,
) -> OptimizeResult:
    """Levenberg-Marquardt least squares — port of EasySpin ``esfit_levmar.m``.

    Parameters are rescaled to [-1, 1] (``ScaleParams``), the residual Jacobian
    is estimated by forward differences with step ``lm_delta`` (EasySpin
    ``FitOpt.delta``, 1e-3) and then updated by Broyden rank-1 updates, with one
    coordinate re-estimated by finite differences per iteration when the step
    in that coordinate is small (``gamma = 0.8``).  Damping follows Nielsen's
    update; the loop stops when the gradient (``lm_gradient_tol``, EasySpin
    ``TolFun``) or the step (``tol_x``, EasySpin ``TolStep``) is small.
    ``resid_fn(p)`` must return the residual vector; ``objective_fn`` is only
    used to report the RMSD.
    """
    if resid_fn is None:
        raise ValueError("_optimize_levmar needs resid_fn (residual vector).")
    lb, ub = (np.asarray(b, dtype=float) for b in bounds)
    n = len(p0)
    span = np.where(ub > lb, ub - lb, 1.0)
    transform = lambda p: 2.0 * (p - lb) / span - 1.0
    untransform = lambda x: lb + span * (x / 2.0 + 0.5)
    xlb, xub = transform(lb), transform(ub)
    delta = float(options.lm_delta)
    tol_step = float(options.tol_x)
    tol_grad = float(options.lm_gradient_tol)

    n_eval = [0]
    tracker = _tracker_of(objective_fn)

    def funeval(x):
        f = np.asarray(resid_fn(untransform(x)), dtype=float).ravel()
        n_eval[0] += 1
        if not np.all(np.isfinite(f)):
            raise ValueError('residuals contain NaN or Inf')
        F = float(f @ f)
        # F is the sum of squares of the target-transformed residuals: sqrt(F/n) is
        # exactly the RMSD compute_residuals reports
        tracker.record(untransform(x), np.sqrt(F / max(1, f.size)))
        return F, f

    x = np.clip(transform(np.asarray(p0, dtype=float)), xlb, xub)
    F, f = funeval(x)
    J = np.zeros((f.size, n))
    for i in range(n):
        x1 = x.copy(); x1[i] += delta
        _, f1 = funeval(x1)
        J[:, i] = (f1 - f) / delta
    _warn_dead_jacobian_columns(J, 'levmar')
    g = J.T @ f
    norm_g = np.linalg.norm(g, ord=np.inf)
    A = J.T @ J
    mu = float(options.lm_lambda) * max(float(np.max(np.diag(A))), 1e-300)
    nu = 2.0
    j = 0
    it = 0
    message = 'max_iter reached'
    for it in range(1, int(options.max_iter) + 1):
        tracker.iteration(it - 1, None)
        if norm_g <= tol_grad:
            message = 'gradient below threshold'
            break
        # LM step: (A + mu I) h = -g, increasing mu until positive definite
        while True:
            try:
                R = np.linalg.cholesky(A + mu * np.eye(n))
                if 1.0 / np.linalg.cond(R) < 1e-15:
                    raise np.linalg.LinAlgError
                break
            except np.linalg.LinAlgError:
                mu *= 10.0
        h = np.linalg.solve(R.T, np.linalg.solve(R, -g))
        norm_h = np.linalg.norm(h)
        if norm_h <= tol_step * (tol_step + np.linalg.norm(x)):
            message = 'parameter step below threshold'
            break
        xnew = np.clip(x + h, xlb, xub)
        Fnew, fnew = funeval(xnew)
        # refresh one Jacobian column by finite differences when its step was small
        j = (j % n) + 1
        if abs(h[j - 1]) < 0.8 * norm_h:
            xu = x.copy(); xu[j - 1] += delta
            _, fu = funeval(xu)
            hu = xu - x
            J = J + np.outer((fu - f - J @ hu) / (hu @ hu), hu)
        J = J + np.outer((fnew - f - J @ h) / (h @ h), h)
        rho = (F - Fnew) / (0.5 * (h @ (mu * h - g)))
        if rho > 0:
            x, F, f = xnew, Fnew, fnew
            mu *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            nu = 2.0
        else:
            mu *= nu
            nu *= 2.0
        g = J.T @ f
        norm_g = np.linalg.norm(g, ord=np.inf)
        A = J.T @ J
        if not np.isfinite(norm_g) or not np.isfinite(A).all():
            message = 'non-finite gradient'
            break
    p_best = untransform(x)
    return OptimizeResult(x=p_best, fun=float(objective_fn(p_best)), nfev=n_eval[0],
                          nit=it, success=True, message=message)


def _optimize_trf(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions,
    resid_fn: Optional[Callable] = None,
) -> OptimizeResult:
    """Bounded trust-region reflective least squares (SciPy ``least_squares``).

    Minimizes the target-transformed residual vector returned by ``resid_fn``
    within ``bounds``, so esfit's autoscale/baseline/target handling is
    retained.  Every residual evaluation, including SciPy's finite-difference
    Jacobian probes, is recorded in the progress tracker: progress reporting,
    ``max_time``/``stop_when`` and interrupts behave as for the other methods
    (a stop raises ``_StopFit`` out of SciPy and esfit returns the best point
    so far).  ``options.max_iter`` is SciPy's ``max_nfev`` (step evaluations;
    Jacobian probes are not counted against it).  SciPy's success flag is kept,
    so an exhausted budget reports ``success=False``.  ``objective_fn`` only
    supplies the tracker.
    """
    if resid_fn is None:
        raise ValueError("_optimize_trf needs resid_fn (residual vector).")
    lb, ub = (np.asarray(b, dtype=float) for b in bounds)
    tracker = _tracker_of(objective_fn)

    def residual(p):
        r = np.asarray(resid_fn(p), dtype=float).ravel()
        if not np.all(np.isfinite(r)):
            raise ValueError('residuals contain NaN or Inf')
        # sqrt(mean(r^2)) is exactly the RMSD compute_residuals reports
        tracker.record(p, float(np.sqrt(np.mean(r * r))))
        return r

    x0 = np.clip(np.asarray(p0, dtype=float), lb, ub)
    fit = least_squares(residual, x0, bounds=(lb, ub), method='trf',
                        x_scale=options.trf_x_scale, diff_step=options.trf_diff_step,
                        max_nfev=int(options.max_iter), ftol=options.tol_fun, xtol=options.tol_x,
                        gtol=options.trf_gradient_tol)
    # SciPy reports no iteration count for 'trf'; each trust-region iteration
    # evaluates the Jacobian once, so njev is the closest measure.
    n_iter = fit.njev if getattr(fit, 'njev', None) is not None else fit.nfev
    if getattr(fit, 'jac', None) is not None:
        _warn_dead_jacobian_columns(fit.jac, 'trf')
    return OptimizeResult(x=fit.x, fun=float(np.sqrt(np.mean(fit.fun ** 2))), success=bool(fit.success),
                          message=str(fit.message), nfev=int(fit.nfev), nit=int(n_iter))


_DEAD_COLUMN_HINT = (
    "A piecewise-constant model has no finite-difference derivative: garlic's "
    "default Options.AccumMethod='binning' places each line in its nearest field "
    "bin, so small shifts of g or A do not change the spectrum. Use "
    "AccumMethod='linear' or 'explicit' in the model.")


def _dead_jacobian_columns(J: np.ndarray, rel_tol: float = 1e-12) -> np.ndarray:
    """Boolean mask of Jacobian columns that are zero, or negligible relative to
    the largest column (the parameter has no effect at the finite-difference step)."""
    J = np.asarray(J, dtype=float)
    if J.ndim != 2 or J.shape[1] == 0:
        return np.zeros(0, dtype=bool)
    norms = np.linalg.norm(J, axis=0)
    ref = float(np.max(norms)) if np.all(np.isfinite(norms)) else 0.0
    return ~(norms > rel_tol * ref) if ref > 0 else np.ones(J.shape[1], dtype=bool)


def _warn_dead_jacobian_columns(J: np.ndarray, where: str, names=None) -> np.ndarray:
    """Warn if any Jacobian column is dead (see _dead_jacobian_columns); return the mask."""
    dead = _dead_jacobian_columns(J)
    if dead.any():
        idx = np.flatnonzero(dead)
        label = ', '.join(str(names[i]) if names is not None else f'#{i}' for i in idx)
        warnings.warn(
            f"esfit ({where}): the residuals do not change with parameter(s) {label} at "
            f"the finite-difference step, so they are invisible to the "
            f"{'optimizer' if where != 'uncertainties' else 'error estimate'}. "
            + _DEAD_COLUMN_HINT, UserWarning, stacklevel=3)
    return dead


def _estimate_uncertainties(
    pfit: np.ndarray,
    data: np.ndarray,
    model_wrapper: ModelWrapper,
    param_handler: ParameterHandler,
    options: FitOptions
) -> dict:
    """
    Estimate parameter uncertainties via Jacobian + covariance.
    
    Returns
    -------
    uncertainties : dict
        'pstd': standard deviations
        'ci95': 95% confidence intervals (n_params x 2)
        'cov': covariance matrix
        'corr': correlation matrix
    """
    # Define residual function
    def residuals_fn(pvec):
        residuals, _, _ = compute_residuals(
            pvec, data, model_wrapper, param_handler, options
        )
        return residuals
    
    # Compute Jacobian via finite differences
    J_all = _jacobian_finite_diff(residuals_fn, pfit)
    # A parameter the residuals do not depend on would get sigma = 0 from pinv,
    # which reads as perfectly determined. Report it as undetermined (nan)
    # instead and leave it out of the covariance of the others.
    dead = _warn_dead_jacobian_columns(J_all, 'uncertainties', param_handler.get_names())
    live = ~dead
    if not live.any():
        nan = np.full(len(pfit), np.nan)
        return {'pstd': nan, 'ci95': np.column_stack([nan, nan]),
                'cov': np.full((len(pfit),) * 2, np.nan),
                'corr': np.full((len(pfit),) * 2, np.nan)}
    J = J_all[:, live]

    # Compute residuals at optimum
    residuals = residuals_fn(pfit)
    n_data = len(residuals)
    n_params = len(pfit)

    def _embed(m_live):
        m = np.full((n_params, n_params), np.nan)
        m[np.ix_(live, live)] = m_live
        return m

    # Compute covariance matrix (HC1 estimator)
    try:
        JtJ_inv = np.linalg.pinv(J.T @ J)
        # HC1 sandwich estimator: avoid building O(n²) diagonal matrix
        # (J.T @ diag(r²) @ J) == (J.T * r²) @ J
        W   = residuals ** 2                          # (n_data,)
        JtWJ = (J.T * W) @ J                          # (n_params, n_params)
        cov  = JtJ_inv @ JtWJ @ JtJ_inv
        cov *= n_data / max(n_data - n_params, 1)     # degrees-of-freedom correction

        # Standard deviations — clamp to avoid sqrt(negative) from float noise
        diag_cov = np.maximum(np.diag(cov), 0.0)

        # Correlation matrix — use same clamped diagonal to avoid NaN
        D_inv = np.diag(1.0 / (np.sqrt(diag_cov) + 1e-12))
        corr = _embed(D_inv @ cov @ D_inv)
        cov = _embed(cov)
        pstd = np.full(n_params, np.nan)
        pstd[live] = np.sqrt(diag_cov)

        # 95% confidence intervals (1.96 = norm.ppf(0.975))
        ci95 = np.column_stack([pfit - 1.96 * pstd, pfit + 1.96 * pstd])
        
        return {
            'pstd': pstd,
            'ci95': ci95,
            'cov': cov,
            'corr': corr
        }
    except np.linalg.LinAlgError as e:
        if options.verbosity > 0:
            print(f"Warning: Could not compute uncertainties: {e}")
        return {
            'pstd': None,
            'ci95': None,
            'cov': None,
            'corr': None
        }


def _jacobian_finite_diff(
    fn: Callable,
    x: np.ndarray,
    eps: float = 1e-7
) -> np.ndarray:
    """Compute Jacobian via central finite differences."""
    n_params = len(x)
    f0 = fn(x)
    n_data = len(f0)
    J = np.zeros((n_data, n_params))
    
    for i in range(n_params):
        x_plus = x.copy()
        x_minus = x.copy()
        step = eps * max(abs(x[i]), 1.0)
        x_plus[i] += step
        x_minus[i] -= step
        J[:, i] = (fn(x_plus) - fn(x_minus)) / (2*step)
    
    return J


def _compute_fit_diagnostics(
    residuals: np.ndarray,
    fit: np.ndarray,
    n_params: int
) -> dict:
    """
    Compute additional fit quality metrics.
    
    Parameters
    ----------
    residuals : np.ndarray
        Fit residuals
    fit : np.ndarray
        Fitted values
    n_params : int
        Number of fitted parameters
    
    Returns
    -------
    diagnostics : dict
        R²: coefficient of determination
        reduced_chi_square: χ²/(n_data - n_params)
        noise_std: estimated noise level
    """
    data = fit + residuals  # Reconstruct data
    n_data = len(residuals)
    
    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data - np.mean(data))**2)
    r_squared = 1 - ss_res / (ss_tot + 1e-12)
    
    # Reduced chi-square: ss_res / dof
    # Without an independent noise estimate, this equals the estimated noise
    # variance.  It is NOT a true chi-square test (which requires known sigma).
    # Values >> 1 indicate a poor fit; values near 0 indicate overfitting.
    dof = max(n_data - n_params, 1)
    reduced_chi_square = ss_res / dof
    noise_std = np.sqrt(reduced_chi_square)

    return {
        'r_squared': r_squared,
        'reduced_chi_square': reduced_chi_square,
        'noise_std': noise_std
    }


def esfit(
    data: Union[np.ndarray, torch.Tensor],
    model_fn: Callable,
    p0: Union[np.ndarray, dict],
    vary: Optional[Union[np.ndarray, dict]] = None,
    lb: Optional[Union[np.ndarray, dict]] = None,
    ub: Optional[Union[np.ndarray, dict]] = None,
    options: Optional[FitOptions] = None
) -> FitResult:
    """
    Fit experimental data using least-squares optimization.
    
    Parameters
    ----------
    data : array_like
        Experimental data vector (1D)
    model_fn : callable
        Simulation function. For custom functions: ``datasim = model_fn(p)``
        where ``p`` is a parameter vector. For torchspin simulators (pepper,
        garlic, chili, etc.), use dict-style ``p0``/``vary`` to fit spin-system
        and experiment parameters by name (see examples below).

        The local optimizers ('levmar', 'trf', 'lbfgsb') and the uncertainty
        estimate use finite differences, so the model must be smooth in its
        parameters. garlic's default ``Options.AccumMethod='binning'`` is not:
        it puts each line in its nearest field bin, so the spectrum is
        piecewise-constant in g and A. When esfit calls garlic itself
        (dict style), it switches an unset AccumMethod to 'linear'. A custom
        model that wraps garlic should pass ``Options(AccumMethod='linear')``
        or ``'explicit'``. esfit warns when a parameter has no effect on the
        residuals and reports its ``pstd`` as nan, not 0.
    p0 : array or dict
        Initial parameters. Array for custom model functions; dict-style
        (mirroring ``SpinSystem``/``Experiment`` fields) for torchspin
        simulators.
    vary : array or dict, optional
        Allowed variation (p0 ± vary). Mutually exclusive with lb/ub.

        In the dict style the field names are known, so fields that cannot be
        negative (``lw``, ``lwpp``, ``HStrain``, ``gStrain``, ``AStrain``,
        ``DStrain``, ``weight``, ``tcorr``, ``T1``, ``T2``, ...) get their lower
        bound clipped at zero when ``p0 - vary`` would go below it, as EasySpin's
        ``esfit`` does.  A plain parameter vector carries no such information and
        is used as given, so ``vary`` larger than ``p0`` there means a negative
        lower bound; pass explicit ``lb``/``ub`` if that matters.
    lb, ub : array or dict, optional
        Lower/upper bounds, used exactly as given. Mutually exclusive with vary.
    options : FitOptions, optional
        Fitting options (algorithm, scaling, baseline, etc.)
    
    Returns
    -------
    result : FitResult
        Fit results with parameters, uncertainties, quality metrics
    
    Examples
    --------
    Fit a simple function:
    
    >>> def model(p):
    ...     x = np.linspace(0, 10, 100)
    ...     return p[0] * x + p[1]
    >>> result = esfit(data, model, p0=[1.0, 0.0], vary=[2.0, 2.0])
    
    Fit with explicit bounds:
    
    >>> result = esfit(data, model, p0=[1.0, 0.0], lb=[0.0, -1.0], ub=[5.0, 1.0])
    """
    # 1. Validate inputs
    data = np.asarray(data).ravel()
    if options is None:
        options = FitOptions()

    # Resolve target='auto': pick 'int' for derivative-like data
    # (mean << std, characteriztic of field-modulated CW EPR with
    # Harmonic=1) so the loss landscape is smooth enough for local
    # optimizers. Matches MATLAB EasySpin esfit.m:382-390 which sets
    # TargetID=2 (integral) for pepper/garlic with Harmonic>0.
    if options.target == 'auto':
        data_mean = float(np.abs(np.mean(data)))
        data_std = float(np.std(data))
        is_derivative_like = data_std > 0 and data_mean < 0.1 * data_std
        from dataclasses import replace
        options = replace(options, target='int' if is_derivative_like else 'fcn')
        if options.verbosity > 0:
            print(f"  Target (auto-detected): {options.target} "
                  f"(|mean|/std = {data_mean / max(data_std, 1e-30):.3g})")

    if options.verbosity > 0:
        print("Starting esfit...")
        print(f"  Algorithm: {options.method}")
        print(f"  Autoscale: {options.autoscale}")
        print(f"  Baseline:  {options.baseline}")
        print(f"  Target:    {options.target}")
        print(f"  Data points: {len(data)}")
    
    # 2. Parse parameters and bounds
    param_handler = ParameterHandler(p0, vary=vary, lb=lb, ub=ub)
    pvec_0 = param_handler.to_vector(p0)
    lb_vec, ub_vec = param_handler.get_bounds()
    pnames = param_handler.get_names()
    
    if options.verbosity > 0:
        print(f"  Parameters: {param_handler.n_active} variable, {param_handler.n_params - param_handler.n_active} fixed")
    
    # 3. Wrap model function
    model_wrapper = ModelWrapper(model_fn, p0, len(data))
    
    # 4. Define objective function (a partial of a worker-safe options copy, so it
    #    pickles for the process pool; progress/stop callables stay in this process)
    import functools
    from dataclasses import replace as _replace
    worker_options = _replace(options, progress=None, stop_when=None, callback=None)
    obj_fn = functools.partial(_objective_function, data=data, model_wrapper=model_wrapper,
                               param_handler=param_handler, options=worker_options)
    tracker = _FitProgress(options, options.method, n_params=param_handler.n_active, n_data=len(data))
    tracked = _TrackedObjective(obj_fn, tracker)
    
    # 5. Evaluate initial RMSD (recorded, so an early interrupt still returns p0)
    rmsd_0 = obj_fn(pvec_0)
    tracker.record(pvec_0, rmsd_0, emit=False)     # before the guarded block: no reporting/stop yet
    if options.verbosity > 0:
        print(f"  Initial RMSD: {rmsd_0:.6e}")
    
    # 6. Run optimization
    if options.verbosity > 0:
        print("\nOptimizing...")
    if options.method not in ('simplex', 'lbfgsb', 'powell', 'grid', 'montecarlo', 'genetic', 'swarm', 'global', 'levmar', 'trf'):
        raise ValueError(f"Unknown method: '{options.method}'")

    tracker.start()
    try:
        if options.method == 'simplex':
            opt_result = _optimize_simplex(tracked, pvec_0, (lb_vec, ub_vec), options)
        elif options.method == 'lbfgsb':
            opt_result = _optimize_lbfgsb(tracked, pvec_0, (lb_vec, ub_vec), options)
        elif options.method == 'powell':
            opt_result = _optimize_powell(tracked, pvec_0, (lb_vec, ub_vec), options)
        elif options.method == 'grid':
            opt_result = _optimize_grid(tracked, pvec_0, (lb_vec, ub_vec), options)
        elif options.method == 'montecarlo':
            opt_result = _optimize_montecarlo(tracked, pvec_0, (lb_vec, ub_vec), options)
        elif options.method == 'genetic':
            opt_result = _optimize_genetic(tracked, pvec_0, (lb_vec, ub_vec), options)
        elif options.method == 'swarm':
            opt_result = _optimize_swarm(tracked, pvec_0, (lb_vec, ub_vec), options)
        elif options.method == 'global':
            opt_result = _optimize_global(tracked, pvec_0, (lb_vec, ub_vec), options)
        else:
            def resid_fn(pvec):
                return compute_residuals(np.clip(pvec, lb_vec, ub_vec), data, model_wrapper,
                                         param_handler, worker_options)[0]
            local_solver = _optimize_trf if options.method == 'trf' else _optimize_levmar
            opt_result = local_solver(tracked, pvec_0, (lb_vec, ub_vec), options, resid_fn=resid_fn)
    except KeyboardInterrupt:
        opt_result = tracker.synthesize_result('KeyboardInterrupt')
    except _StopFit as e:
        opt_result = tracker.synthesize_result(e.reason)
    finally:
        tracker.finish()
    
    pfit_vec = opt_result.x
    
    # 7. Compute final fit and residuals
    residuals, rmsd, info = compute_residuals(
        pfit_vec, data, model_wrapper, param_handler, options
    )
    if info.get('negative_scale'):
        warnings.warn(
            "esfit: lsq autoscale found a negative scale at the fitted parameters "
            "(the model is anti-correlated with the data), so the scale was "
            f"{'clipped to 0' if options.lsq_negative_scale == 'clip' else 'replaced by its absolute value'}. "
            "Check the sign of the model: with autoscale='lsq' it must carry the "
            "sign of the signal.", UserWarning, stacklevel=2)
    
    if options.verbosity > 0:
        print(f"\nFinal RMSD: {rmsd:.6e}")
        if rmsd_0 > 0:
            print(f"Improvement: {(rmsd_0 - rmsd)/rmsd_0 * 100:.1f}%")
        print(f"Success: {opt_result.success}")
        print(f"Message: {opt_result.message}")
    
    # 8. Compute uncertainties if requested
    if options.compute_uncertainties and opt_result.success:
        if options.verbosity > 0:
            print("\nComputing uncertainties...")
        uncertainties = _estimate_uncertainties(
            pfit_vec, data, model_wrapper, param_handler, options
        )
    else:
        uncertainties = {
            'pstd': None,
            'ci95': None,
            'cov': None,
            'corr': None
        }
    
    # 9. Compute fit diagnostics
    diagnostics = _compute_fit_diagnostics(info['residuals_raw'], info['fit'], len(pfit_vec))
    
    # 10. Construct result object
    result = FitResult(
        pfit=pfit_vec,
        pnames=pnames,
        p_start=pvec_0,
        p_fixed=param_handler.fixed_mask,
        fit=info['fit'],
        fit_raw=info['sim'],
        residuals=info['residuals_raw'],
        scale=info['scale'],
        baseline=info['baseline'],
        rmsd=rmsd,
        ssr=np.sum(residuals**2),
        n_params=len(pfit_vec),
        n_data=len(data),
        algorithm=options.method,
        n_iterations=opt_result.nit if hasattr(opt_result, 'nit') else opt_result.nfev,
        success=opt_result.success,
        message=opt_result.message,
        pstd=uncertainties['pstd'],
        ci95=uncertainties['ci95'],
        cov=uncertainties['cov'],
        corr=uncertainties['corr'],
        r_squared=diagnostics['r_squared'],
        reduced_chi_square=diagnostics['reduced_chi_square'],
        noise_std=diagnostics['noise_std'],
        n_evaluations=tracker.n_evaluations,
        elapsed_s=tracker.elapsed_s,
        interrupted=tracker.interrupted,
    )
    
    if options.verbosity > 0:
        print("\nFitted parameters:")
        for i, (name, val, std) in enumerate(zip(pnames, pfit_vec, uncertainties['pstd'] if uncertainties['pstd'] is not None else [None]*len(pfit_vec))):
            if std is not None:
                print(f"  {name}: {val:.6e} ± {std:.6e}")
            else:
                print(f"  {name}: {val:.6e}")
        print(f"\nFit quality:")
        print(f"  RMSD:      {result.rmsd:.6e}")
        print(f"  R²:        {result.r_squared:.6f}")
        print(f"  Red. χ²:   {result.reduced_chi_square:.6f}")
        print(f"  Noise std: {result.noise_std:.6e}")
        print()
    
    return result
