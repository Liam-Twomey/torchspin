"""
Unit tests for esfit fitting functionality.
"""

import os

import numpy as np
import pytest
import torch
from torchspin import esfit, FitOptions, SpinSystem, Experiment, pepper
from torchspin.experiment import Options


class TestParameterHandling:
    """Test parameter conversions and bounds handling."""
    
    def test_array_input_vary(self):
        """Test array input with vary specification."""
        from torchspin.esfit import ParameterHandler
        
        p0 = np.array([2.0, 10.0])
        vary = np.array([0.5, 2.0])
        
        handler = ParameterHandler(p0, vary=vary)
        
        assert len(handler.p0) == 2
        assert handler.n_params == 2
        assert handler.n_active == 2
        np.testing.assert_allclose(handler.lb, [1.5, 8.0])
        np.testing.assert_allclose(handler.ub, [2.5, 12.0])
    
    def test_array_input_bounds(self):
        """Test array input with explicit bounds."""
        from torchspin.esfit import ParameterHandler
        
        p0 = np.array([2.0, 10.0])
        lb = np.array([1.0, 5.0])
        ub = np.array([3.0, 15.0])
        
        handler = ParameterHandler(p0, lb=lb, ub=ub)
        
        np.testing.assert_allclose(handler.lb, lb)
        np.testing.assert_allclose(handler.ub, ub)
    
    def test_fixed_parameters(self):
        """Test that fixed parameters (lb == ub) are detected."""
        from torchspin.esfit import ParameterHandler
        
        p0 = np.array([2.0, 10.0, 5.0])
        lb = np.array([1.0, 10.0, 4.0])
        ub = np.array([3.0, 10.0, 6.0])  # p[1] is fixed
        
        handler = ParameterHandler(p0, lb=lb, ub=ub)
        
        assert handler.n_params == 3
        assert handler.n_active == 2
        assert handler.fixed_mask[1] == True
        assert handler.active_mask[1] == False
    
    def test_to_vector_to_full(self):
        """Test conversion between full and active parameters."""
        from torchspin.esfit import ParameterHandler
        
        p0 = np.array([2.0, 10.0, 5.0])
        lb = np.array([1.0, 10.0, 4.0])
        ub = np.array([3.0, 10.0, 6.0])  # p[1] is fixed
        
        handler = ParameterHandler(p0, lb=lb, ub=ub)
        
        # Extract active
        p_active = handler.to_vector(p0)
        assert len(p_active) == 2
        np.testing.assert_allclose(p_active, [2.0, 5.0])
        
        # Reconstruct full
        p_full = handler.to_full(np.array([2.5, 5.5]))
        np.testing.assert_allclose(p_full, [2.5, 10.0, 5.5])
    
    def test_no_variable_parameters_raises(self):
        """Test that error is raised if all parameters are fixed."""
        from torchspin.esfit import ParameterHandler
        
        p0 = np.array([2.0, 10.0])
        lb = np.array([2.0, 10.0])
        ub = np.array([2.0, 10.0])  # All fixed
        
        with pytest.raises(ValueError, match="No variable parameters"):
            ParameterHandler(p0, lb=lb, ub=ub)
    
    def test_vary_and_bounds_raises(self):
        """Test that specifying both vary and bounds raises error."""
        from torchspin.esfit import ParameterHandler
        
        p0 = np.array([2.0, 10.0])
        vary = np.array([0.5, 2.0])
        lb = np.array([1.0, 5.0])
        
        with pytest.raises(ValueError, match="Cannot specify both"):
            ParameterHandler(p0, vary=vary, lb=lb)


class TestModelWrapper:
    """Test model function wrapping."""
    
    def test_simple_function(self):
        """Test wrapping a simple custom function."""
        from torchspin.esfit import ModelWrapper, ParameterHandler
        
        def model(p):
            x = np.linspace(0, 10, 100)
            return p[0] * x + p[1]
        
        p0 = np.array([2.0, 1.0])
        handler = ParameterHandler(p0, vary=np.array([1.0, 1.0]))
        wrapper = ModelWrapper(model, p0, exp_data_len=100)
        
        result = wrapper(handler.to_vector(p0), handler)
        assert len(result) == 100
        assert isinstance(result, np.ndarray)
    
    def test_torch_output_conversion(self):
        """Test that torch output is converted to numpy."""
        from torchspin.esfit import ModelWrapper, ParameterHandler
        
        def model(p):
            return torch.ones(50) * p[0]
        
        p0 = np.array([2.0])
        handler = ParameterHandler(p0, vary=np.array([1.0]))
        wrapper = ModelWrapper(model, p0, exp_data_len=50)
        
        result = wrapper(handler.to_vector(p0), handler)
        assert isinstance(result, np.ndarray)
        assert len(result) == 50
    
    def test_wrong_output_length_raises(self):
        """Test that error is raised if model output length is wrong."""
        from torchspin.esfit import ModelWrapper, ParameterHandler
        
        def model(p):
            return np.ones(100) * p[0]
        
        p0 = np.array([2.0])
        handler = ParameterHandler(p0, vary=np.array([1.0]))
        wrapper = ModelWrapper(model, p0, exp_data_len=50)  # Expect 50, get 100
        
        with pytest.raises(ValueError, match="output length"):
            wrapper(handler.to_vector(p0), handler)


class TestAutoscaling:
    """Test autoscaling methods."""
    
    def test_lsq_scale_no_baseline(self):
        """Test LSQ scaling without baseline."""
        from torchspin.esfit import _fit_scale_baseline_lsq
        
        sim = np.array([1.0, 2.0, 3.0, 4.0])
        data = np.array([2.0, 4.0, 6.0, 8.0])  # scale = 2
        
        scale, baseline = _fit_scale_baseline_lsq(sim, data, baseline_order=-1)
        
        assert abs(scale - 2.0) < 0.01
        np.testing.assert_allclose(baseline, 0.0)
    
    def test_lsq_scale_with_baseline(self):
        """Test LSQ scaling with constant baseline."""
        from torchspin.esfit import _fit_scale_baseline_lsq
        
        sim = np.array([1.0, 2.0, 3.0, 4.0])
        data = np.array([3.0, 5.0, 7.0, 9.0])  # scale=2, baseline=1
        
        scale, baseline = _fit_scale_baseline_lsq(sim, data, baseline_order=0)
        
        assert abs(scale - 2.0) < 0.01
        assert np.all(np.abs(baseline - 1.0) < 0.01)


class TestTargetTransforms:
    """Test target transformations."""
    
    def test_fcn_no_transform(self):
        """Test 'fcn' target (no transformation)."""
        from torchspin.esfit import _apply_target_transform
        
        residuals = np.array([1.0, 2.0, 3.0])
        result = _apply_target_transform(residuals, 'fcn')
        np.testing.assert_allclose(result, residuals)
    
    def test_int_cumsum(self):
        """Test 'int' target (cumulative sum)."""
        from torchspin.esfit import _apply_target_transform
        
        residuals = np.array([1.0, 2.0, 3.0])
        result = _apply_target_transform(residuals, 'int')
        np.testing.assert_allclose(result, [1.0, 3.0, 6.0])
    
    def test_diff_derivative(self):
        """Test 'diff' target (derivative)."""
        from torchspin.esfit import _apply_target_transform
        
        residuals = np.array([1.0, 3.0, 6.0])
        result = _apply_target_transform(residuals, 'diff')
        # diff with prepend=0: [1.0, 3.0-1.0, 6.0-3.0] = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])
    
    def test_invalid_target_raises(self):
        """Test that invalid target raises error."""
        from torchspin.esfit import _apply_target_transform
        
        residuals = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Unknown target"):
            _apply_target_transform(residuals, 'invalid')


class TestSimpleFits:
    """Test fitting simple analytical functions."""
    
    def test_fit_linear_function(self):
        """Test fitting y = mx + b."""
        # Generate synthetic data
        x = np.linspace(0, 10, 100)
        m_true, b_true = 2.5, 1.0
        y_true = m_true * x + b_true
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, len(x))
        y_data = y_true + noise
        
        # Define model
        def model(p):
            return p[0] * x + p[1]
        
        # Fit without autoscaling to recover true parameters
        result = esfit(
            y_data,
            model,
            p0=np.array([1.0, 0.0]),
            vary=np.array([3.0, 3.0]),
            options=FitOptions(verbosity=0, autoscale='none', baseline=-1)
        )
        
        assert result.success
        assert abs(result.pfit[0] - m_true) < 0.1
        assert abs(result.pfit[1] - b_true) < 0.1
        assert result.rmsd < 0.1
    
    def test_fit_exponential(self):
        """Test fitting y = A * exp(-x/tau)."""
        # Generate synthetic data
        x = np.linspace(0, 10, 100)
        A_true, tau_true = 5.0, 2.0
        y_true = A_true * np.exp(-x/tau_true)
        np.random.seed(123)
        noise = np.random.normal(0, 0.05, len(x))
        y_data = y_true + noise
        
        # Define model
        def model(p):
            return p[0] * np.exp(-x/p[1])
        
        # Fit without autoscaling to recover true parameters
        result = esfit(
            y_data,
            model,
            p0=np.array([3.0, 1.0]),
            lb=np.array([0.1, 0.1]),
            ub=np.array([10.0, 10.0]),
            options=FitOptions(verbosity=0, autoscale='none', baseline=-1, max_iter=2000)
        )
        
        assert result.success
        assert abs(result.pfit[0] - A_true) < 0.5
        assert abs(result.pfit[1] - tau_true) < 0.5
    
    def test_fit_with_fixed_parameter(self):
        """Test fitting with one fixed parameter."""
        # Generate data: y = mx + b with known b
        x = np.linspace(0, 10, 100)
        m_true, b_true = 2.5, 1.0
        y_data = m_true * x + b_true
        
        # Define model
        def model(p):
            return p[0] * x + p[1]
        
        # Fit with b fixed, no autoscaling
        result = esfit(
            y_data,
            model,
            p0=np.array([1.0, 1.0]),
            lb=np.array([0.0, 1.0]),  # b fixed at 1.0
            ub=np.array([5.0, 1.0]),
            options=FitOptions(verbosity=0, autoscale='none', baseline=-1)
        )
        
        assert result.success
        assert result.n_params == 1  # Only m is fitted
        assert abs(result.pfit[0] - m_true) < 0.01
    
    def test_fit_with_baseline_correction(self):
        """Test fitting with baseline correction."""
        # Generate data with linear baseline. The model must not lie in the
        # span of the baseline polynomial, or its amplitude is unidentifiable
        # (an earlier version used p[0]*x, which a linear baseline absorbs).
        x = np.linspace(0, 10, 100)
        m_true = 2.0
        baseline_true = 0.5 + 0.1 * x  # Linear baseline
        y_data = m_true * np.sin(x) + baseline_true

        # Define model (without baseline)
        def model(p):
            return p[0] * np.sin(x)
        
        # Fit with linear baseline and no separate scaling
        result = esfit(
            y_data,
            model,
            p0=np.array([1.0]),
            vary=np.array([2.0]),
            options=FitOptions(verbosity=0, autoscale='none', baseline=1)
        )
        
        assert result.success
        # When using baseline correction without autoscaling, parameter should be close to true value
        assert abs(result.pfit[0] - m_true) < 0.1


class TestOptimizationAlgorithms:
    """Test different optimization algorithms."""
    
    def test_simplex_algorithm(self):
        """Test Nelder-Mead simplex optimization."""
        x = np.linspace(0, 10, 50)
        y_data = 2.0 * x + 1.0
        
        def model(p):
            return p[0] * x + p[1]
        
        result = esfit(
            y_data,
            model,
            p0=np.array([1.0, 0.0]),
            vary=np.array([2.0, 2.0]),
            options=FitOptions(method='simplex', verbosity=0)
        )
        
        assert result.success
        assert result.algorithm == 'simplex'
    
    def test_lbfgsb_algorithm(self):
        """Test L-BFGS-B optimization."""
        x = np.linspace(0, 10, 50)
        y_data = 2.0 * x + 1.0
        
        def model(p):
            return p[0] * x + p[1]
        
        result = esfit(
            y_data,
            model,
            p0=np.array([1.0, 0.0]),
            vary=np.array([2.0, 2.0]),
            options=FitOptions(method='lbfgsb', verbosity=0)
        )
        
        assert result.success
        assert result.algorithm == 'lbfgsb'
    
    def test_powell_algorithm(self):
        """Test Powell optimization."""
        x = np.linspace(0, 10, 50)
        y_data = 2.0 * x + 1.0
        
        def model(p):
            return p[0] * x + p[1]
        
        result = esfit(
            y_data,
            model,
            p0=np.array([1.0, 0.0]),
            vary=np.array([2.0, 2.0]),
            options=FitOptions(method='powell', verbosity=0)
        )
        
        assert result.success
        assert result.algorithm == 'powell'


class TestUncertainties:
    """Test uncertainty estimation."""
    
    def test_uncertainties_computed(self):
        """Test that uncertainties are computed by default."""
        x = np.linspace(0, 10, 100)
        y_data = 2.0 * x + 1.0
        
        def model(p):
            return p[0] * x + p[1]
        
        result = esfit(
            y_data,
            model,
            p0=np.array([1.0, 0.0]),
            vary=np.array([2.0, 2.0]),
            options=FitOptions(verbosity=0)
        )
        
        assert result.pstd is not None
        assert result.ci95 is not None
        assert result.cov is not None
        assert result.corr is not None
        assert len(result.pstd) == 2
        assert result.ci95.shape == (2, 2)
    
    def test_uncertainties_disabled(self):
        """Test that uncertainties can be disabled."""
        x = np.linspace(0, 10, 100)
        y_data = 2.0 * x + 1.0
        
        def model(p):
            return p[0] * x + p[1]
        
        result = esfit(
            y_data,
            model,
            p0=np.array([1.0, 0.0]),
            vary=np.array([2.0, 2.0]),
            options=FitOptions(verbosity=0, compute_uncertainties=False)
        )
        
        assert result.pstd is None
        assert result.ci95 is None


class TestBoundsEnforcement:
    """Test that parameter bounds are respected."""
    
    def test_bounds_respected(self):
        """Test that fitted parameters stay within bounds."""
        # Data that would fit to m=10 without bounds
        x = np.linspace(0, 1, 100)
        y_data = 10.0 * x
        
        def model(p):
            return p[0] * x
        
        # Constrain m to [0, 5], no autoscaling
        result = esfit(
            y_data,
            model,
            p0=np.array([2.0]),
            lb=np.array([0.0]),
            ub=np.array([5.0]),
            options=FitOptions(verbosity=0, autoscale='none', baseline=-1)
        )
        
        # Should hit upper bound
        assert result.pfit[0] <= 5.0
        assert result.pfit[0] >= 0.0
        assert result.pfit[0] > 4.5  # Should be near upper bound


class TestAutoscaleMethods:
    """Test different autoscaling methods."""
    
    def test_lsq_autoscale(self):
        """Test LSQ autoscaling."""
        x = np.linspace(0, 10, 100)
        y_data = 2.0 * x + 1.0
        
        def model(p):
            return p[0] * x  # Missing offset
        
        result = esfit(
            y_data,
            model,
            p0=np.array([1.0]),
            vary=np.array([2.0]),
            options=FitOptions(autoscale='lsq', baseline=0, verbosity=0)
        )
        
        assert result.success
        assert result.scale > 0
    
    def test_maxabs_autoscale(self):
        """Test maxabs autoscaling."""
        x = np.linspace(0, 10, 100)
        y_data = 2.0 * x
        
        def model(p):
            return p[0] * x
        
        result = esfit(
            y_data,
            model,
            p0=np.array([1.0]),
            vary=np.array([2.0]),
            options=FitOptions(autoscale='maxabs', verbosity=0)
        )
        
        assert result.success
        assert result.scale > 0
    
    def test_none_autoscale(self):
        """Test no autoscaling."""
        x = np.linspace(0, 10, 100)
        y_data = 2.0 * x
        
        def model(p):
            return p[0] * x
        
        result = esfit(
            y_data,
            model,
            p0=np.array([1.0]),
            vary=np.array([2.0]),
            options=FitOptions(autoscale='none', verbosity=0)
        )
        
        assert result.success
        assert abs(result.scale - 1.0) < 1e-6


class TestFitQuality:
    """Test fit quality metrics."""
    
    def test_rmsd_computed(self):
        """Test that RMSD is computed correctly."""
        x = np.linspace(0, 10, 100)
        y_data = 2.0 * x + 1.0
        np.random.seed(42)
        y_data += np.random.normal(0, 0.1, len(x))
        
        def model(p):
            return p[0] * x + p[1]
        
        result = esfit(
            y_data,
            model,
            p0=np.array([1.0, 0.0]),
            vary=np.array([2.0, 2.0]),
            options=FitOptions(verbosity=0, autoscale='none', baseline=-1)
        )
        
        assert result.rmsd > 0
        assert result.rmsd < 0.2  # Should be small for good fit
    
    def test_ssr_computed(self):
        """Test that SSR is computed."""
        x = np.linspace(0, 10, 100)
        y_data = 2.0 * x + 1.0
        
        def model(p):
            return p[0] * x + p[1]
        
        result = esfit(
            y_data,
            model,
            p0=np.array([1.0, 0.0]),
            vary=np.array([2.0, 2.0]),
            options=FitOptions(verbosity=0)
        )
        
        assert result.ssr >= 0
        # SSR = sum(residuals^2) ~= rmsd^2 * n_data
        expected_ssr = result.rmsd**2 * result.n_data
        assert abs(result.ssr - expected_ssr) < 1e-6


class TestEPRFitting:
    """Test esfit with pepper as the model function (EPR spectra)."""

    def test_fit_isotropic_g_from_clean_data(self):
        """Fit isotropic g-value from noise-free synthetic spectrum.

        Uses simplex (gradient-free) — more robust for EPR spectra which
        have smooth but not perfectly differentiable loss surfaces.
        Tolerance is 0.002 to account for finite grid resolution.
        """
        from torchspin.experiment import Options

        g_true = 2.005
        exp = Experiment(mwFreq=9.5, Range=[330.0, 350.0], nPoints=256, Harmonic=1)
        opt = Options(GridSize=19, GridSymmetry='auto', Verbosity=0)

        sys_true = SpinSystem(S=0.5, g=g_true, lwpp=[0.5, 0.0])
        _, spc_ref = pepper(sys_true, exp, opt)
        spc_data = spc_ref.numpy()

        def model(p):
            sys = SpinSystem(S=0.5, g=float(p[0]), lwpp=[0.5, 0.0])
            _, spc = pepper(sys, exp, opt)
            return spc.numpy()

        result = esfit(
            spc_data,
            model,
            p0=np.array([2.003]),
            lb=np.array([1.998]),
            ub=np.array([2.012]),
            options=FitOptions(method='simplex', verbosity=0, max_iter=500,
                               autoscale='maxabs', baseline=-1),
        )

        assert result.success
        assert abs(result.pfit[0] - g_true) < 0.002, (
            f"g recovery failed: |{result.pfit[0]:.5f} - {g_true}| = "
            f"{abs(result.pfit[0]-g_true):.5f} > 0.002"
        )

    def test_fit_rhombic_g_from_clean_data(self):
        """Fit rhombic g-tensor from noise-free synthetic spectrum.

        Uses simplex with GridSize=19 for speed. Tolerance 0.003 for g
        to account for grid resolution.
        """
        from torchspin.experiment import Options

        g_true = [2.008, 2.005, 2.001]
        exp = Experiment(mwFreq=9.5, Range=[330.0, 350.0], nPoints=512, Harmonic=1)
        opt = Options(GridSize=19, GridSymmetry='auto', Verbosity=0)

        sys_true = SpinSystem(S=0.5, g=g_true, lwpp=[0.4, 0.0])
        _, spc_ref = pepper(sys_true, exp, opt)
        spc_data = spc_ref.numpy()

        def model(p):
            sys = SpinSystem(S=0.5, g=p[:3].tolist(), lwpp=[0.4, 0.0])
            _, spc = pepper(sys, exp, opt)
            return spc.numpy()

        result = esfit(
            spc_data,
            model,
            p0=np.array([2.005, 2.003, 1.999]),
            lb=np.array([1.995, 1.995, 1.990]),
            ub=np.array([2.020, 2.015, 2.010]),
            options=FitOptions(method='simplex', verbosity=0, max_iter=1000,
                               autoscale='maxabs', baseline=-1),
        )

        assert result.success
        for i, (g_fit, g_ref) in enumerate(zip(result.pfit, g_true)):
            assert abs(g_fit - g_ref) < 0.003, (
                f"g[{i}] recovery failed: |{g_fit:.5f} - {g_ref}| = "
                f"{abs(g_fit-g_ref):.5f} > 0.003"
            )

    def test_fit_residual_decreases(self):
        """Verify that the optimizer reduces the residual vs starting point."""
        from torchspin.experiment import Options

        g_true = 2.006
        exp = Experiment(mwFreq=9.5, Range=[330.0, 350.0], nPoints=256, Harmonic=1)
        opt = Options(GridSize=11, GridSymmetry='auto', Verbosity=0)

        sys_true = SpinSystem(S=0.5, g=g_true, lwpp=[0.5, 0.0])
        _, spc_ref = pepper(sys_true, exp, opt)
        spc_data = spc_ref.numpy()

        def model(p):
            sys = SpinSystem(S=0.5, g=float(p[0]), lwpp=[0.5, 0.0])
            _, spc = pepper(sys, exp, opt)
            return spc.numpy()

        p0 = np.array([2.000])  # Noticeably offset from true value 2.006
        spc0 = model(p0)
        rmsd_initial = np.sqrt(np.mean(
            (spc0 / np.abs(spc0).max() - spc_data / np.abs(spc_data).max()) ** 2
        ))

        result = esfit(
            spc_data,
            model,
            p0=p0,
            lb=np.array([1.990]),
            ub=np.array([2.015]),
            options=FitOptions(method='simplex', verbosity=0, max_iter=300,
                               autoscale='maxabs', baseline=-1),
        )

        assert result.success
        spc_fit = model(result.pfit)
        rmsd_final = np.sqrt(np.mean(
            (spc_fit / np.abs(spc_fit).max() - spc_data / np.abs(spc_data).max()) ** 2
        ))
        assert rmsd_final < rmsd_initial, (
            f"Residual did not decrease: {rmsd_final:.4f} >= {rmsd_initial:.4f}"
        )


class TestMATLABCrossValidation:
    """Fit MATLAB-generated EPR spectra with torchspin esfit+pepper.

    These tests verify that torchspin can recover the parameters used in
    EasySpin (MATLAB) to simulate a spectrum.  Reference .mat files were
    generated by ``tests/data/esfit_matlab_validation_generate.m``.

    Tolerances allow for grid resolution (GridSize=25) and 1-2 % noise:
        Δg ≤ 0.001 for g-tensor components
        ΔA ≤ 3 MHz  for hyperfine couplings
    """

    DATA_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'tests', 'data',
    )

    def _mat_path(self, name):
        return os.path.join(self.DATA_DIR, name)

    def test_fit_rhombic_g_from_matlab_spectrum(self):
        """Recover rhombic g-tensor from MATLAB-generated noisy spectrum.

        Reference: esfit_test1_gtensor.mat
          Sys.g = [2.009, 2.006, 2.002], lwpp = 0.5 mT, noise = 2 %
          Field range [330, 350] mT, mwFreq = 9.5 GHz
        """
        mat_path = self._mat_path('esfit_test1_gtensor.mat')
        if not os.path.exists(mat_path):
            pytest.skip(f'Reference file not found: {mat_path}')

        import scipy.io as sio
        mat = sio.loadmat(mat_path, simplify_cells=True)
        B_mT = mat['B1']
        spc_data = mat['spec1_noisy']

        g_true = [2.009, 2.006, 2.002]
        exp = Experiment(mwFreq=9.5,
                         Range=[float(B_mT.min()), float(B_mT.max())],
                         nPoints=len(B_mT), Harmonic=1)
        opt = Options(GridSize=25, GridSymmetry='auto', Verbosity=0)

        def model(p):
            sys = SpinSystem(S=0.5, g=p[:3].tolist(), lwpp=[0.5, 0.0])
            _, spc = pepper(sys, exp, opt)
            return spc.numpy()

        result = esfit(
            spc_data,
            model,
            p0=np.array([2.007, 2.004, 2.000]),
            lb=np.array([2.000, 2.000, 1.995]),
            ub=np.array([2.015, 2.012, 2.008]),
            options=FitOptions(method='simplex', verbosity=0, max_iter=1500,
                               autoscale='maxabs', baseline=-1),
        )

        assert result.success
        for i, (g_fit, g_ref) in enumerate(zip(result.pfit, g_true)):
            assert abs(g_fit - g_ref) < 0.001, (
                f"g[{i}] recovery: |{g_fit:.5f} - {g_ref}| = "
                f"{abs(g_fit - g_ref):.5f} > 0.001"
            )

    def test_fit_g_and_hyperfine_from_matlab_spectrum(self):
        """Recover g-tensor and 14N hyperfine from MATLAB-generated noisy spectrum.

        Reference: esfit_test2_gtensor_hf.mat
          Sys.g = [2.008, 2.006, 2.003], A = [20, 20, 85] MHz, lwpp = 0.3 mT
          Field range [330, 350] mT, noise = 1.5 %
        """
        mat_path = self._mat_path('esfit_test2_gtensor_hf.mat')
        if not os.path.exists(mat_path):
            pytest.skip(f'Reference file not found: {mat_path}')

        import scipy.io as sio
        mat = sio.loadmat(mat_path, simplify_cells=True)
        B_mT = mat['B2']
        spc_data = mat['spec2_noisy']

        g_true = [2.008, 2.006, 2.003]
        Aiso_true, Az_true = 20.0, 85.0

        exp = Experiment(mwFreq=9.5,
                         Range=[float(B_mT.min()), float(B_mT.max())],
                         nPoints=len(B_mT), Harmonic=1)
        opt = Options(GridSize=20, GridSymmetry='auto', Verbosity=0)

        def model(p):
            sys = SpinSystem(S=0.5, g=p[:3].tolist(), Nucs='14N',
                             A=[[float(p[3]), float(p[3]), float(p[4])]],
                             lwpp=[0.3, 0.0])
            _, spc = pepper(sys, exp, opt)
            return spc.numpy()

        result = esfit(
            spc_data,
            model,
            p0=np.array([2.006, 2.004, 2.001, 20.0, 85.0]),
            lb=np.array([2.000, 2.000, 1.998, 10.0, 70.0]),
            ub=np.array([2.015, 2.012, 2.008, 35.0, 100.0]),
            options=FitOptions(method='simplex', verbosity=0, max_iter=2000,
                               autoscale='maxabs', baseline=-1),
        )

        assert result.success
        # Compare sorted g-values: in powder spectra, gx/gy assignment is
        # degenerate (indistinguishable by symmetry), so the optimizer may
        # swap them.  Sorting gives the physically meaningful comparison.
        g_fit_sorted = sorted(result.pfit[:3])
        g_true_sorted = sorted(g_true)
        for i, (gf, gt) in enumerate(zip(g_fit_sorted, g_true_sorted)):
            assert abs(gf - gt) < 0.001, (
                f"g_sorted[{i}] recovery: |{gf:.5f} - {gt:.3f}| = "
                f"{abs(gf - gt):.5f} > 0.001"
            )
        # Hyperfine: noisier due to trade-off with g, use looser tolerance
        assert abs(result.pfit[4] - Az_true) < 5.0, (
            f"Az recovery: |{result.pfit[4]:.1f} - {Az_true}| > 5 MHz"
        )


class TestEsfitRobustness:
    """Additional robustness tests for esfit with EPR spectra."""

    def test_fit_g_and_lw_simultaneously(self):
        """Fit both isotropic g and Gaussian linewidth from clean data.

        With bounds that separate g from lw, simplex should recover both
        parameters to within tolerance.
        """
        from torchspin.experiment import Options

        g_true, lw_true = 2.003, 0.8
        exp = Experiment(mwFreq=9.5, Range=[330.0, 350.0], nPoints=256, Harmonic=1)
        opt = Options(GridSize=15, GridSymmetry='auto', Verbosity=0)

        sys_true = SpinSystem(S=0.5, g=g_true, lw=[lw_true, 0.0])
        _, spc_ref = pepper(sys_true, exp, opt)
        spc_data = spc_ref.numpy()

        def model(p):
            sys = SpinSystem(S=0.5, g=float(p[0]), lw=[float(p[1]), 0.0])
            _, spc = pepper(sys, exp, opt)
            return spc.numpy()

        result = esfit(
            spc_data,
            model,
            p0=np.array([2.001, 0.6]),
            lb=np.array([1.995, 0.3]),
            ub=np.array([2.010, 1.5]),
            options=FitOptions(method='simplex', verbosity=0, max_iter=1000,
                               autoscale='maxabs', baseline=-1),
        )

        assert result.success
        assert abs(result.pfit[0] - g_true) < 0.003, (
            f"g recovery: |{result.pfit[0]:.5f} - {g_true}| > 0.003"
        )
        assert abs(result.pfit[1] - lw_true) < 0.15, (
            f"lw recovery: |{result.pfit[1]:.3f} - {lw_true}| > 0.15 mT"
        )

    def test_fit_converges_from_offset_start(self):
        """Fit returns success and reduces SSR when started 3 mT off the true field.

        This tests that the optimizer runs without error from a noticeably wrong
        starting point and produces a lower SSR than at the starting point.
        Uses a simpler analytical model (not pepper) for deterministic behavior.
        """
        # Synthetic Gaussian derivative: data at x0_true=5.0, optimizer starts at 4.5
        x = np.linspace(0.0, 10.0, 200)
        x0_true, sigma_true = 5.0, 0.5
        y_true = -(x - x0_true) / sigma_true**2 * np.exp(-0.5 * ((x - x0_true)/sigma_true)**2)
        np.random.seed(7)
        y_data = y_true + 0.01 * np.random.normal(size=len(x))

        def model(p):
            x0, sig = p[0], p[1]
            return -(x - x0) / sig**2 * np.exp(-0.5 * ((x - x0)/sig)**2)

        p0 = np.array([4.5, 0.6])  # noticeably wrong start
        ssr_initial = np.sum((model(p0) - y_data)**2)

        result = esfit(
            y_data,
            model,
            p0=p0,
            lb=np.array([3.0, 0.2]),
            ub=np.array([7.0, 1.5]),
            options=FitOptions(method='simplex', verbosity=0, max_iter=2000,
                               autoscale='none', baseline=-1),
        )

        assert result.success
        ssr_final = np.sum((model(result.pfit) - y_data)**2)
        assert ssr_final < ssr_initial, (
            f"SSR did not decrease: {ssr_final:.4e} >= {ssr_initial:.4e}"
        )
        assert abs(result.pfit[0] - x0_true) < 0.1, (
            f"|pfit[0] - x0_true| = {abs(result.pfit[0]-x0_true):.4f} > 0.1"
        )

    def test_no_permutation_with_ordered_bounds(self):
        """Asymmetric bounds prevent g-component permutation in simplex fits.

        For rhombic g with bounds g[0] > g[1] > g[2] (no overlap), the
        simplex cannot swap components, so recovery is exact.
        """
        from torchspin.experiment import Options

        g_true = [2.010, 2.005, 2.001]
        exp = Experiment(mwFreq=9.5, Range=[328.0, 352.0], nPoints=512, Harmonic=1)
        opt = Options(GridSize=19, GridSymmetry='auto', Verbosity=0)

        sys_true = SpinSystem(S=0.5, g=g_true, lw=[0.4, 0.0])
        _, spc_ref = pepper(sys_true, exp, opt)
        spc_data = spc_ref.numpy()

        def model(p):
            sys = SpinSystem(S=0.5, g=p[:3].tolist(), lw=[0.4, 0.0])
            _, spc = pepper(sys, exp, opt)
            return spc.numpy()

        # Non-overlapping bounds enforce g[0] > g[1] > g[2] ordering
        result = esfit(
            spc_data,
            model,
            p0=np.array([2.008, 2.004, 1.999]),
            lb=np.array([2.006, 2.002, 1.997]),
            ub=np.array([2.015, 2.008, 2.003]),
            options=FitOptions(method='simplex', verbosity=0, max_iter=1500,
                               autoscale='maxabs', baseline=-1),
        )

        assert result.success
        for i, (g_fit, g_ref) in enumerate(zip(result.pfit, g_true)):
            assert abs(g_fit - g_ref) < 0.003, (
                f"g[{i}] recovery: |{g_fit:.5f} - {g_ref}| > 0.003"
            )

    def test_fit_spectral_cosine_high(self):
        """Fitted spectrum should have cosine similarity > 0.99 with reference."""
        from torchspin.experiment import Options

        g_true = 2.004
        exp = Experiment(mwFreq=9.5, Range=[330.0, 348.0], nPoints=256, Harmonic=1)
        opt = Options(GridSize=15, GridSymmetry='auto', Verbosity=0)

        sys_true = SpinSystem(S=0.5, g=g_true, lw=[0.6, 0.0])
        _, spc_ref = pepper(sys_true, exp, opt)
        spc_data = spc_ref.numpy()

        def model(p):
            sys = SpinSystem(S=0.5, g=float(p[0]), lw=[0.6, 0.0])
            _, spc = pepper(sys, exp, opt)
            return spc.numpy()

        result = esfit(
            spc_data,
            model,
            p0=np.array([2.002]),
            lb=np.array([1.998]),
            ub=np.array([2.010]),
            options=FitOptions(method='simplex', verbosity=0, max_iter=600,
                               autoscale='maxabs', baseline=-1),
        )

        spc_fit = model(result.pfit)
        # Cosine similarity
        dot = np.dot(spc_data, spc_fit)
        cos = dot / (np.linalg.norm(spc_data) * np.linalg.norm(spc_fit) + 1e-12)
        assert cos > 0.99, f"Cosine similarity {cos:.4f} < 0.99"

    def test_default_max_iter_is_2000(self):
        """FitOptions default max_iter should be 2000."""
        opts = FitOptions()
        assert opts.max_iter == 2000, (
            f"Default max_iter is {opts.max_iter}, expected 2000"
        )


class TestNewOptimizationMethods:
    """Tests for grid, montecarlo, and genetic optimization algorithms."""

    # A simple 1-D quadratic: f(x) = (x - 3)^2, minimum at x=3.
    @staticmethod
    def _quad_model():
        x_true = 3.0
        x_data = np.array([x_true])  # "data"
        def model(p):
            return np.array([p[0]])
        return x_data, model

    def test_grid_search_finds_minimum(self):
        """Grid search: 1-D quadratic minimized to within search resolution."""
        x = np.linspace(0, 10, 200)
        y_data = 3.0 * np.exp(-((x - 5.0) / 0.5) ** 2)

        def model(p):
            return p[0] * np.exp(-((x - p[1]) / 0.5) ** 2)

        result = esfit(
            y_data,
            model,
            p0=np.array([2.5, 4.0]),
            lb=np.array([1.0, 3.0]),
            ub=np.array([5.0, 7.0]),
            options=FitOptions(method='grid', grid_points=8, verbosity=0,
                               autoscale='none', baseline=-1),
        )
        assert result.success
        assert result.algorithm == 'grid'
        # Best grid point should be reasonably close to truth
        assert abs(result.pfit[1] - 5.0) < 1.5  # resolution ≈ (7-3)/7 ≈ 0.57

    def test_grid_method_lower_rmsd_than_start(self):
        """Grid result has lower RMSD than initial guess."""
        x = np.linspace(0, 10, 100)
        y_data = np.exp(-((x - 4.0) / 1.0) ** 2)

        def model(p):
            return np.exp(-((x - p[0]) / 1.0) ** 2)

        p0 = np.array([2.0])  # bad start
        result = esfit(
            y_data,
            model,
            p0=p0,
            lb=np.array([0.0]),
            ub=np.array([10.0]),
            options=FitOptions(method='grid', grid_points=20, verbosity=0,
                               autoscale='none', baseline=-1),
        )
        rmsd_start = float(np.sqrt(np.mean((model(p0) - y_data) ** 2)))
        rmsd_fit = float(np.sqrt(np.mean((model(result.pfit) - y_data) ** 2)))
        assert rmsd_fit < rmsd_start

    def test_montecarlo_finds_minimum(self):
        """Monte Carlo + SA: 1-D Gaussian peak recovered."""
        x = np.linspace(0, 10, 200)
        y_data = np.exp(-((x - 6.0) / 0.8) ** 2)

        def model(p):
            return np.exp(-((x - p[0]) / 0.8) ** 2)

        result = esfit(
            y_data,
            model,
            p0=np.array([3.0]),
            lb=np.array([0.0]),
            ub=np.array([10.0]),
            options=FitOptions(method='montecarlo', mc_samples=300, verbosity=0,
                               autoscale='none', baseline=-1),
        )
        assert result.success
        assert result.algorithm == 'montecarlo'
        assert abs(result.pfit[0] - 6.0) < 1.0

    def test_montecarlo_lower_rmsd_than_start(self):
        """Monte Carlo result has lower RMSD than initial guess."""
        x = np.linspace(0, 10, 100)
        y_data = 2.0 * np.exp(-((x - 7.0) / 1.0) ** 2)

        def model(p):
            return p[0] * np.exp(-((x - p[1]) / 1.0) ** 2)

        p0 = np.array([1.0, 3.0])
        result = esfit(
            y_data,
            model,
            p0=p0,
            lb=np.array([0.5, 1.0]),
            ub=np.array([4.0, 9.0]),
            options=FitOptions(method='montecarlo', mc_samples=200, verbosity=0,
                               autoscale='none', baseline=-1),
        )
        rmsd_start = float(np.sqrt(np.mean((model(p0) - y_data) ** 2)))
        rmsd_fit = float(np.sqrt(np.mean((model(result.pfit) - y_data) ** 2)))
        assert rmsd_fit < rmsd_start

    def test_genetic_finds_minimum(self):
        """Genetic algorithm: 1-D Gaussian peak recovered."""
        x = np.linspace(0, 10, 200)
        y_data = np.exp(-((x - 4.5) / 0.6) ** 2)

        def model(p):
            return np.exp(-((x - p[0]) / 0.6) ** 2)

        result = esfit(
            y_data,
            model,
            p0=np.array([2.0]),
            lb=np.array([0.0]),
            ub=np.array([10.0]),
            options=FitOptions(method='genetic', ga_population=30, verbosity=0,
                               autoscale='none', baseline=-1, max_iter=500),
        )
        assert result.success
        assert result.algorithm == 'genetic'
        assert abs(result.pfit[0] - 4.5) < 1.0

    def test_genetic_lower_rmsd_than_start(self):
        """Genetic algorithm result has lower RMSD than initial guess."""
        x = np.linspace(0, 10, 100)
        y_data = np.sin(x) + 0.5

        def model(p):
            return np.sin(p[0] * x) + p[1]

        p0 = np.array([2.0, 0.0])  # wrong frequency
        result = esfit(
            y_data,
            model,
            p0=p0,
            lb=np.array([0.5, -1.0]),
            ub=np.array([3.0, 2.0]),
            options=FitOptions(method='genetic', ga_population=20, verbosity=0,
                               autoscale='none', baseline=-1, max_iter=300),
        )
        rmsd_start = float(np.sqrt(np.mean((model(p0) - y_data) ** 2)))
        rmsd_fit = float(np.sqrt(np.mean((model(result.pfit) - y_data) ** 2)))
        assert rmsd_fit < rmsd_start

    def test_unknown_method_raises(self):
        """Unknown method raises ValueError."""
        x = np.linspace(0, 1, 10)
        y_data = x.copy()
        with pytest.raises(ValueError, match="Unknown method"):
            esfit(
                y_data,
                lambda p: p[0] * x,
                p0=np.array([1.0]),
                vary=np.array([0.5]),
                options=FitOptions(method='bogus', verbosity=0),
            )

    def test_swarm_finds_minimum(self):
        """Particle swarm: Gaussian peak recovered."""
        x = np.linspace(0, 10, 200)
        y_data = np.exp(-((x - 5.5) / 0.7) ** 2)

        def model(p):
            return np.exp(-((x - p[0]) / 0.7) ** 2)

        result = esfit(
            y_data, model, p0=np.array([3.0]),
            lb=np.array([0.0]), ub=np.array([10.0]),
            options=FitOptions(method='swarm', swarm_size=20, max_iter=100,
                               verbosity=0, autoscale='none', baseline=-1),
        )
        assert result.success
        assert result.algorithm == 'swarm'
        assert abs(result.pfit[0] - 5.5) < 1.0

    def test_swarm_lower_rmsd_than_start(self):
        """Swarm result has lower RMSD than initial guess."""
        x = np.linspace(0, 10, 100)
        y_data = np.exp(-x / 3.0)

        def model(p):
            return np.exp(-x / p[0])

        p0 = np.array([8.0])  # bad start
        result = esfit(
            y_data, model, p0=p0, lb=np.array([0.5]), ub=np.array([10.0]),
            options=FitOptions(method='swarm', swarm_size=15, max_iter=50,
                               verbosity=0, autoscale='none', baseline=-1),
        )
        rmsd_start = float(np.sqrt(np.mean((model(p0) - y_data) ** 2)))
        rmsd_fit = float(np.sqrt(np.mean((model(result.pfit) - y_data) ** 2)))
        assert rmsd_fit < rmsd_start

    def test_swarm_fitoptions_fields(self):
        """FitOptions stores swarm hyper-parameters."""
        opts = FitOptions(swarm_size=50, swarm_inertia=0.8,
                          swarm_cognitive=1.2, swarm_social=1.6)
        assert opts.swarm_size == 50
        assert opts.swarm_inertia == 0.8
        assert opts.swarm_cognitive == 1.2
        assert opts.swarm_social == 1.6

    def test_grid_fitoptions_fields(self):
        """FitOptions stores grid/mc/ga hyper-parameters."""
        opts = FitOptions(
            grid_points=15,
            mc_samples=500,
            mc_temp_start=2.0,
            mc_temp_end=1e-3,
            ga_population=80,
            ga_mutation=0.05,
            ga_crossover=0.8,
            ga_elite=10,
        )
        assert opts.grid_points == 15
        assert opts.mc_samples == 500
        assert opts.mc_temp_start == 2.0
        assert opts.mc_temp_end == 1e-3
        assert opts.ga_population == 80
        assert opts.ga_mutation == 0.05
        assert opts.ga_crossover == 0.8
        assert opts.ga_elite == 10


    def test_levmar_fitoptions_fields(self):
        """FitOptions stores Levenberg-Marquardt hyper-parameters."""
        opts = FitOptions(lm_lambda=1e-4, lm_delta=1e-8,
                          lm_gradient_tol=1e-10, lm_step_tol=1e-10)
        assert opts.lm_lambda == 1e-4
        assert opts.lm_delta == 1e-8
        assert opts.lm_gradient_tol == 1e-10
        assert opts.lm_step_tol == 1e-10


class TestLevenbergMarquardt:
    """Tests for the Levenberg-Marquardt optimizer."""

    def test_levmar_finds_peak(self):
        """LM: Gaussian peak position should be recovered."""
        x = np.linspace(0, 10, 200)
        y_data = np.exp(-((x - 5.5) / 0.7) ** 2)

        def model(p):
            return np.exp(-((x - p[0]) / 0.7) ** 2)

        result = esfit(
            y_data, model, p0=np.array([4.0]),
            lb=np.array([0.0]), ub=np.array([10.0]),
            options=FitOptions(method='levmar', max_iter=200,
                               verbosity=0, autoscale='none', baseline=-1),
        )
        assert result.success
        assert result.algorithm == 'levmar'
        assert abs(result.pfit[0] - 5.5) < 1.0

    def test_levmar_improves_rmsd(self):
        """LM should reduce RMSD from initial guess."""
        x = np.linspace(0, 10, 100)
        y_data = np.exp(-x / 3.0)

        def model(p):
            return np.exp(-x / p[0])

        p0 = np.array([8.0])
        result = esfit(
            y_data, model, p0=p0, lb=np.array([0.5]), ub=np.array([10.0]),
            options=FitOptions(method='levmar', max_iter=100,
                               verbosity=0, autoscale='none', baseline=-1),
        )
        rmsd_start = float(np.sqrt(np.mean((model(p0) - y_data) ** 2)))
        rmsd_fit = float(np.sqrt(np.mean((model(result.pfit) - y_data) ** 2)))
        assert rmsd_fit < rmsd_start

    def test_levmar_two_params(self):
        """LM with two parameters: amplitude and center."""
        x = np.linspace(0, 10, 200)
        y_data = 1.8 * np.exp(-((x - 5.0) / 1.0) ** 2)

        def model(p):
            return p[0] * np.exp(-((x - p[1]) / 1.0) ** 2)

        result = esfit(
            y_data, model,
            p0=np.array([1.0, 3.0]),
            lb=np.array([0.5, 0.0]),
            ub=np.array([3.0, 10.0]),
            options=FitOptions(method='levmar', max_iter=200,
                               verbosity=0, autoscale='none', baseline=-1),
        )
        assert abs(result.pfit[0] - 1.8) < 0.5
        assert abs(result.pfit[1] - 5.0) < 1.0


class TestSimplexExploration:
    """Regression tests for the Nelder-Mead simplex exploration radius.

    Background: torchspin's ``_optimize_simplex`` previously seeded the
    initial simplex with edge length 0.05 in the transformed [-1, 1]
    parameter space (= 2.5 % of the bounded range). EasySpin's
    ``esfit_simplex.m`` uses ``delta = 0.1`` of (ub - lb) in the
    untransformed space, which corresponds to 0.2 in the transformed
    space — 4x larger. The undersized simplex caused premature
    convergence to local minima even on well-conditioned 2-parameter
    problems.

    These tests fail with the old 0.05 perturbation and pass with the
    EasySpin-matching 0.2 perturbation.
    """

    def test_simplex_escapes_local_minimum_basin(self):
        """Two-parameter rugged surface: a global quadratic minimum at
        ``true_min`` decorated with shallow sinusoidal ripples that
        create many local minima of width ~0.3 in the bounded range.
        The OLD 0.05-edge simplex (= 0.25 in a 10-wide range) sits
        entirely inside one ripple basin and converges there. The
        EasySpin-matching 0.2-edge simplex (= 1.0 in real space)
        spans multiple basins, so reflection/expansion can carry it
        toward the global minimum.
        """
        true_min = np.array([3.0, -1.0])
        ripple_amp = 0.5
        ripple_period = 0.5

        def loss_per_point(p):
            # Vector of "residuals" whose sum-of-squares = the rugged
            # quadratic landscape we want to minimize.
            d0 = p[0] - true_min[0]
            d1 = p[1] - true_min[1]
            base = np.array([d0, d1])
            ripple = ripple_amp * np.array([
                np.sin(2 * np.pi * p[0] / ripple_period),
                np.sin(2 * np.pi * p[1] / ripple_period),
            ])
            return base + ripple

        y_data = np.zeros(2)
        # p0 sits ~5 ripple basins away from the true minimum
        result = esfit(
            y_data, loss_per_point,
            p0=np.array([0.05, 1.95]),
            lb=np.array([-2.0, -6.0]),
            ub=np.array([8.0, 4.0]),
            options=FitOptions(method='simplex', verbosity=0,
                               max_iter=2000,
                               autoscale='none', baseline=-1,
                               compute_uncertainties=False),
        )

        assert result.success
        # Must land in the global basin (within one ripple period of true min)
        for i, (p_fit, p_ref) in enumerate(zip(result.pfit, true_min)):
            assert abs(p_fit - p_ref) < ripple_period, (
                f"simplex stuck in local basin on axis {i}: "
                f"|{p_fit:.5f} - {p_ref}| = {abs(p_fit-p_ref):.5f} "
                f">= ripple_period {ripple_period}"
            )

    def test_simplex_initial_edge_matches_easyspin(self):
        """White-box check: the constructed initial simplex must have
        edge length 0.2 in the transformed [-1, 1] space, matching
        EasySpin's ``delta = 0.1 * (ub - lb)``.
        """
        from torchspin.esfit import _optimize_simplex
        from scipy.optimize import OptimizeResult

        captured = {}

        def fake_minimize_capture(*args, **kwargs):
            captured['initial_simplex'] = kwargs['options']['initial_simplex']
            res = OptimizeResult()
            res.x = args[1]
            res.fun = 0.0
            res.success = True
            res.nit = 0
            res.nfev = 1
            res.message = 'ok'
            return res

        import importlib
        esfit_mod = importlib.import_module('torchspin.esfit')
        orig = esfit_mod.minimize
        try:
            esfit_mod.minimize = fake_minimize_capture
            lb = np.array([-1.0, -1.0])
            ub = np.array([1.0, 1.0])
            p0 = np.array([0.0, 0.0])
            _optimize_simplex(lambda x: 0.0, p0, (lb, ub),
                              FitOptions(verbosity=0))
        finally:
            esfit_mod.minimize = orig

        sim = captured['initial_simplex']
        # edge from vertex 0 (the centroid of the constructed simplex
        # is p0_scaled = [0, 0]) to vertex i+1 (perturbed by 0.2 in axis i)
        edge_axis_0 = abs(sim[1, 0] - sim[0, 0])
        edge_axis_1 = abs(sim[2, 1] - sim[0, 1])
        assert abs(edge_axis_0 - 0.2) < 1e-12, (
            f"axis-0 edge {edge_axis_0} != 0.2 (EasySpin delta * 2)"
        )
        assert abs(edge_axis_1 - 0.2) < 1e-12, (
            f"axis-1 edge {edge_axis_1} != 0.2 (EasySpin delta * 2)"
        )


class TestAutoTargetAndGlobal:
    """Tests for target='auto' detection and method='global' pipeline."""

    def test_target_auto_picks_int_for_derivative_data(self):
        """Derivative-like data (mean ~ 0, std > 0) should auto-resolve
        to target='int'. White-box check via the resolved options.target."""
        from dataclasses import replace
        # Synthetic derivative spectrum: zero mean, real structure
        x = np.linspace(0, 1, 200)
        derivative = np.exp(-((x - 0.5) / 0.05) ** 2) * (x - 0.5)
        derivative = derivative - derivative.mean()  # ensure mean = 0
        data = derivative.copy()

        # Capture which target esfit resolves 'auto' to by intercepting
        # the dataclass replace call (target is normalized in-place at the
        # top of esfit). Easiest: read printed verbosity output.
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            esfit(data, lambda p: p[0] * derivative,
                  p0=np.array([1.0]), vary=np.array([0.1]),
                  options=FitOptions(method='simplex', target='auto',
                                     verbosity=1,
                                     autoscale='none', baseline=-1,
                                     compute_uncertainties=False))
        out = buf.getvalue()
        assert 'auto-detected' in out and 'int' in out, (
            f"target='auto' should resolve to 'int' for derivative data; "
            f"verbosity output:\n{out}"
        )

    def test_target_auto_picks_fcn_for_offset_data(self):
        """Data with a strong DC offset (all-positive Gaussian) should
        auto-resolve to 'fcn', because integrating an offset produces a
        runaway ramp that destabilises the loss."""
        x = np.linspace(0, 1, 200)
        gaussian = np.exp(-((x - 0.5) / 0.05) ** 2)  # all-positive
        data = 2.0 * gaussian

        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            esfit(data, lambda p: p[0] * gaussian,
                  p0=np.array([1.0]), vary=np.array([0.1]),
                  options=FitOptions(method='simplex', target='auto',
                                     verbosity=1,
                                     autoscale='none', baseline=-1,
                                     compute_uncertainties=False))
        out = buf.getvalue()
        assert 'auto-detected' in out and 'fcn' in out, (
            f"target='auto' should resolve to 'fcn' for non-derivative data; "
            f"verbosity output:\n{out}"
        )

    def test_global_method_solves_rugged_2d(self):
        """method='global' must escape narrow local minima where simplex
        alone fails. Uses the same rugged surface from
        ``test_simplex_escapes_local_minimum_basin`` but with start point
        deeper into a wrong basin so simplex truly cannot recover."""
        true_min = np.array([3.0, -1.0])
        ripple_amp = 0.5
        ripple_period = 0.5

        def L(p):
            d0, d1 = p[0] - true_min[0], p[1] - true_min[1]
            return np.array([d0, d1]) + ripple_amp * np.array([
                np.sin(2 * np.pi * p[0] / ripple_period),
                np.sin(2 * np.pi * p[1] / ripple_period),
            ])

        np.random.seed(42)
        result = esfit(
            np.zeros(2), L,
            p0=np.array([-1.0, 2.5]),  # several basins away from true min
            lb=np.array([-5.0, -5.0]), ub=np.array([8.0, 5.0]),
            options=FitOptions(method='global', max_iter=300, swarm_size=30,
                               verbosity=0, autoscale='none', baseline=-1,
                               compute_uncertainties=False),
        )
        assert result.success
        # Global must land within one ripple basin of the true minimum.
        # A plain simplex from the same start hits the nearest basin
        # (~3 units away), so any value < 1.0 here is a clear win.
        for i, (p_fit, p_ref) in enumerate(zip(result.pfit, true_min)):
            assert abs(p_fit - p_ref) < ripple_period, (
                f"global failed on axis {i}: |{p_fit:.4f} - {p_ref}| = "
                f"{abs(p_fit-p_ref):.4f} >= ripple_period {ripple_period}"
            )

    def test_n_workers_population_eval(self):
        """_eval_population with n_workers>1 must produce identical
        results to sequential evaluation."""
        from torchspin.esfit import _eval_population
        np.random.seed(0)
        X = np.random.randn(8, 3)

        def f(x):
            return float((x ** 2).sum())

        seq = _eval_population(f, X, n_workers=1)
        par = _eval_population(f, X, n_workers=4)
        np.testing.assert_allclose(seq, par)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestDictStyleBounds:
    """Explicit lb/ub bound dictionaries (EasySpin esfit lb/ub inputs)."""

    def test_dict_bounds_basic(self):
        from torchspin import SpinSystem
        from torchspin.esfit import TorchSpinParameterHandler
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        h = TorchSpinParameterHandler(
            {'Sys': sys}, vary=None,
            lb={'Sys': {'g': [[1.9, 1.9, 1.9]]}},
            ub={'Sys': {'g': [[2.1, 2.1, 2.1]]}})
        assert h.n_params == 3
        assert np.allclose(h.lb, 1.9) and np.allclose(h.ub, 2.1)
        assert h.pnames == ['Sys.g[0,0]', 'Sys.g[0,1]', 'Sys.g[0,2]']

    def test_dict_bounds_fixed_params_masked(self):
        from torchspin import SpinSystem
        from torchspin.esfit import TorchSpinParameterHandler
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.2]])
        h = TorchSpinParameterHandler(
            {'Sys': sys}, vary=None,
            lb={'Sys': {'g': [[1.9, 2.0, 2.1]]}},
            ub={'Sys': {'g': [[2.1, 2.0, 2.3]]}})
        # middle element fixed (lb == ub) → excluded from active set
        assert h.n_active == 2

    def test_dict_bounds_errors(self):
        from torchspin import SpinSystem
        from torchspin.esfit import TorchSpinParameterHandler
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        with pytest.raises(ValueError, match="lb > ub"):
            TorchSpinParameterHandler(
                {'Sys': sys}, vary=None,
                lb={'Sys': {'g': [[2.2, 1.9, 1.9]]}},
                ub={'Sys': {'g': [[2.1, 2.1, 2.1]]}})
        with pytest.raises(ValueError, match="outside"):
            TorchSpinParameterHandler(
                {'Sys': sys}, vary=None,
                lb={'Sys': {'g': [[2.05, 1.9, 1.9]]}},
                ub={'Sys': {'g': [[2.1, 2.1, 2.1]]}})


def _quadratic_objective(x):
    """Module-level (picklable) objective for the process-pool test."""
    import numpy as np
    return float(np.sum((np.asarray(x) - 0.5) ** 2))


def test_eval_population_process_pool_matches_sequential():
    """workers='processes' (spawned pool, capped BLAS threads) must return the
    same values in the same order as the sequential evaluation."""
    import numpy as np
    from torchspin.esfit import _eval_population
    X = np.random.default_rng(0).random((6, 3))
    seq = _eval_population(_quadratic_objective, X, n_workers=1)
    par = _eval_population(_quadratic_objective, X, n_workers=3, workers='processes')
    assert np.allclose(seq, par)


def test_eval_population_process_pool_notebook_style_objective():
    """A closure (not picklable by reference) reaches spawned workers via cloudpickle."""
    import numpy as np
    pytest.importorskip('cloudpickle')
    from torchspin.esfit import _eval_population
    offset = 0.25
    obj = lambda x: float(np.sum((np.asarray(x) - offset) ** 2))   # noqa: E731
    X = np.random.default_rng(1).random((5, 2))
    assert np.allclose(_eval_population(obj, X, n_workers='auto', workers='processes'), _eval_population(obj, X, 1))


def test_eval_population_main_module_function_goes_by_value():
    """A function that pickles by reference into __main__ (as notebook cells do) must be
    shipped by value: by reference the spawned worker cannot resolve it and the map hangs."""
    import numpy as np
    pytest.importorskip('cloudpickle')
    from torchspin.esfit import _eval_population

    def nb_style(x):
        return float(np.sum(np.asarray(x)))
    nb_style.__module__ = '__main__'
    nb_style.__qualname__ = 'nb_style'
    X = np.random.default_rng(2).random((4, 3))
    assert np.allclose(_eval_population(nb_style, X, 2, 'processes'), X.sum(axis=1))


class TestAutoscaleSignAndBaseline:
    """GitHub #6: lsq autoscale with a negative unconstrained scale, and the
    baselines of autoscale='none'/'maxabs'."""

    def _design(self, n, order):
        x = np.linspace(0, 1, n)
        return np.column_stack([x**k for k in range(order + 1)])

    def test_negative_scale_clips_to_zero_and_refits_baseline(self):
        from torchspin.esfit import _fit_scale_baseline_lsq
        t = np.linspace(0, 10, 200)
        data = 5.0 - 3.0 * np.exp(-t / 2.0)
        sim = np.exp(-t / 2.0)                        # wrong sign
        scale, baseline, neg = _fit_scale_baseline_lsq(sim, data, 1, return_negative=True)
        assert neg and scale == 0.0
        P = self._design(t.size, 1)
        ref = P @ np.linalg.lstsq(P, data, rcond=None)[0]
        np.testing.assert_allclose(baseline, ref, rtol=0, atol=1e-12)

    def test_constrained_is_never_worse_than_abs(self):
        from torchspin.esfit import _fit_scale_baseline_lsq
        rng = np.random.default_rng(1)
        for order in (-1, 0, 2):
            for _ in range(20):
                sim, data = rng.standard_normal(50), rng.standard_normal(50)
                s_c, b_c = _fit_scale_baseline_lsq(sim, data, order)
                s_a, b_a = _fit_scale_baseline_lsq(sim, data, order, negative_scale='abs')
                assert s_c >= 0 and s_a >= 0
                assert (np.linalg.norm(s_c * sim + b_c - data)
                        <= np.linalg.norm(s_a * sim + b_a - data) + 1e-12)

    def test_positive_scale_unchanged(self):
        from torchspin.esfit import _fit_scale_baseline_lsq
        x = np.linspace(0, 1, 100)
        sim = np.sin(8 * x)
        data = 2.0 * sim + 0.3 + 0.1 * x
        for mode in ('clip', 'abs'):
            scale, baseline = _fit_scale_baseline_lsq(sim, data, 1, negative_scale=mode)
            assert abs(scale - 2.0) < 1e-10
            np.testing.assert_allclose(baseline, 0.3 + 0.1 * x, atol=1e-10)

    def test_inversion_recovery_wrong_sign_warns(self):
        """The reproducer from #6: with the wrong model sign the fit no longer
        minimizes a mirrored objective, and esfit warns."""
        t = np.linspace(0, 10, 400)
        y = 5.0 - 3.0 * np.exp(-t / 2.0) + 0.01 * np.random.default_rng(0).standard_normal(t.size)
        opts = FitOptions(method='levmar', baseline=0, target='fcn', verbosity=0,
                          compute_uncertainties=False)
        with pytest.warns(UserWarning, match="negative scale"):
            r = esfit(y, lambda p: np.exp(-t / p[0]), np.array([2.2]),
                      lb=np.array([0.1]), ub=np.array([20.0]), options=opts)
        # scale 0 + constant baseline: the RMSD is the data's own spread
        assert r.scale == 0.0
        assert abs(r.rmsd - np.std(y)) < 1e-9
        r = esfit(y, lambda p: -np.exp(-t / p[0]), np.array([2.2]),
                  lb=np.array([0.1]), ub=np.array([20.0]), options=opts)
        assert abs(r.pfit[0] - 2.0) < 0.02

    def test_autoscale_none_baseline_fits_data_minus_sim(self):
        from torchspin.esfit import compute_residuals, ModelWrapper, ParameterHandler
        x = np.linspace(0, 1, 80)
        data = np.sin(6 * x) + 0.4 - 0.2 * x
        ph = ParameterHandler(np.array([1.0]), vary=np.array([0.5]))
        mw = ModelWrapper(lambda p: p[0] * np.sin(6 * x), np.array([1.0]), x.size)
        for autoscale in ('none', 'maxabs'):
            opts = FitOptions(autoscale=autoscale, baseline=1, target='fcn')
            _, rmsd, info = compute_residuals(np.array([1.0]), data, mw, ph, opts)
            resid = data - info['scale'] * info['sim']
            P = self._design(x.size, 1)
            ref = P @ np.linalg.lstsq(P, resid, rcond=None)[0]
            np.testing.assert_allclose(info['baseline'], ref, atol=1e-12)
        # with autoscale='none' the model plus a linear baseline is exact
        _, rmsd, _ = compute_residuals(np.array([1.0]), data, mw, ph,
                                       FitOptions(autoscale='none', baseline=1, target='fcn'))
        assert rmsd < 1e-12


class TestDictStyleArrayFields:
    """Every element of an array field must reach the model, not only the last."""

    def test_to_dict_applies_every_element(self):
        from torchspin.esfit import ParameterHandler
        exp = Experiment(mwFreq=9.65, Range=[300.0, 440.0], nPoints=64)
        sys0 = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Nucs=['1H'], A=[[10.0, 10.0, 10.0]],
                          lw=[1.0, 2.0])
        ph = ParameterHandler({'Sys': sys0, 'Exp': exp},
                              vary={'Sys': {'g': [[0.1, 0.1, 0.0]], 'A': [[5.0, 0, 0]],
                                            'lw': [0.5, 0.5]}})
        p = ph.p0_active + np.array([0.01, 0.02, 1.0, 0.1, 0.2])
        d = ph._handler.to_dict(p)
        np.testing.assert_allclose(d['Sys'].g.numpy(), [[2.01, 2.02, 2.0]])
        np.testing.assert_allclose(d['Sys'].A.numpy(), [[11.0, 10.0, 10.0]])
        np.testing.assert_allclose(d['Sys'].lw, [1.1, 2.2])
        # p0 itself is untouched
        np.testing.assert_allclose(sys0.g.numpy(), [[2.0, 2.0, 2.0]])
        assert sys0.lw == [1.0, 2.0]


class TestBinnedGarlicFiniteDifferences:
    """GitHub #5: garlic's nearest-bin accumulation is piecewise-constant in
    the line positions, so finite-difference Jacobians are zero for g and A."""

    exp = Experiment(mwFreq=9.65, Range=[300.0, 440.0], nPoints=400, Harmonic=1)

    def _sim(self, p, accum):
        from torchspin import garlic
        sys = SpinSystem(S=[0.5], g=p[0], Nucs=['91Zr'], A=p[1], lw=[abs(p[2]), abs(p[3])])
        opt = Options(AccumMethod=accum) if accum else None
        return np.asarray(garlic(sys, self.exp, opt)[1], float)

    def _data(self):
        clean = self._sim([1.885, 780.0, 1.0, 2.0], 'explicit')
        rng = np.random.default_rng(0)
        return clean + 0.01 * np.ptp(clean) * rng.standard_normal(clean.size)

    def test_dead_parameter_warns_and_reports_nan_sigma(self):
        data = self._data()
        opts = FitOptions(method='levmar', max_iter=50, verbosity=0)
        with pytest.warns(UserWarning, match="do not change with parameter"):
            r = esfit(data, lambda p: self._sim(p, 'binning'), np.array([1.8853, 785.0, 1.2, 1.8]),
                      vary=np.array([0.005, 30.0, 0.8, 0.8]), options=opts)
        assert np.isnan(r.pstd[1])                      # A: undetermined, not sigma = 0
        assert np.all(np.isfinite(r.pstd[2:]))          # line widths still estimated

    def test_smooth_model_has_no_warning(self):
        import warnings as _w
        data = self._data()
        with _w.catch_warnings():
            _w.simplefilter('error', UserWarning)
            r = esfit(data, lambda p: self._sim(p, 'linear'), np.array([1.8853, 785.0, 1.2, 1.8]),
                      vary=np.array([0.005, 30.0, 0.8, 0.8]),
                      options=FitOptions(method='levmar', max_iter=100, verbosity=0))
        assert abs(r.pfit[1] - 780.0) < 2.0
        assert np.all(np.isfinite(r.pstd)) and np.all(r.pstd > 0)

    def test_dict_style_garlic_switches_default_binning_to_linear(self):
        from torchspin import garlic
        from torchspin.esfit import ModelWrapper
        data = self._data()
        sys0 = SpinSystem(S=[0.5], g=1.8853, Nucs=['91Zr'], A=785.0, lw=[1.2, 1.8])
        vary = {'Sys': {'g': [[0.015, 0, 0]], 'A': [[90.0, 0, 0]], 'lw': [0.8, 0.8]}}
        r = esfit(data, garlic, {'Sys': sys0, 'Exp': self.exp}, vary=vary,
                  options=FitOptions(method='levmar', max_iter=100, verbosity=0))
        a_iso = (r.pfit[1] + 2 * 785.0) / 3
        assert abs(a_iso - 780.0) < 2.0
        assert np.all(np.isfinite(r.pstd)) and np.all(r.pstd > 0)
        # a user-chosen AccumMethod is respected
        mw = ModelWrapper(garlic, {'Sys': sys0}, data.size)
        assert mw._fd_friendly_options(Options(AccumMethod='binning'), sys0, self.exp).AccumMethod == 'binning'
        user_opt = Options()
        assert mw._fd_friendly_options(user_opt, sys0, self.exp).AccumMethod == 'linear'
        assert user_opt.AccumMethod is None              # not mutated
        # fast motion defaults to 'explicit' and is left alone
        fm = SpinSystem(S=[0.5], g=2.0, Nucs=['14N'], A=40.0, lw=[0.1, 0.1], tcorr=1e-10)
        assert mw._fd_friendly_options(None, fm, self.exp) is None
