.. torchspin documentation master file, created by
   sphinx-quickstart on Mon Nov  9 15:53:09 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

torchspin documentation
=======================

torchspin reimplements the core physics engine of EasySpin (MATLAB) in
Python/PyTorch. You get the simulators you already know — ``pepper``, ``garlic``,
``chili``, ``salt``, ``saffron``, ``curry``, ``cardamom``, ``esfit`` — with the same units
and conventions, plus three things MATLAB cannot give you:

* **Gradients.** Spectra are differentiable end to end, so you can fit with
  gradient descent, propagate uncertainty, or put a simulator inside a neural
  network.
* **A Python workflow.** NumPy and torch tensors in and out, no license server,
  installable with ``pip``.
* **Optional GPU.** ``Options(device='cuda')`` for batched and gradient-based
  work.

torchspin is a reimplementation, not a drop-in replacement for every EasySpin
feature. Before relying on it for published results, read
``KNOWN_LIMITATIONS.md`` — it states per simulator what is
validated against MATLAB and what is not.


Installation
------------

.. code:: sh

    pip install torchspin                 # core
    pip install "torchspin[plot]"         # + matplotlib for the plotting helpers
    pip install "torchspin[gui]"          # + the interactive fitting panel (torchspin.fitgui)

CPU-only PyTorch, to avoid the large CUDA wheels:

.. code:: sh

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install torchspin

Requires Python 3.10+, PyTorch 2.0+, NumPy 1.23+, SciPy 1.10+.


Quick start
-----------

CW powder spectrum (`pepper`)
*****************************

.. code:: python

    from torchspin import SpinSystem, Experiment, Options, pepper
    
    sys = SpinSystem(
        S=[0.5],
        g=[[2.009, 2.006, 2.002]],
        Nucs=['14N'],
        A=[[10.0, 10.0, 95.0]],   # MHz
        lw=[1.0, 0.0],            # Gaussian FWHM (mT)
    )
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=1024, Harmonic=1)
    B, spec = pepper(sys, exp, Options(GridSize=50))

Solution EPR (`garlic`)
***********************

.. code:: python

    from torchspin import SpinSystem, Experiment, garlic
    
    sys = SpinSystem(S=[0.5], g=2.003, Nucs='1H', A=10.0, lw=[0.2, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[336, 342], nPoints=512, Harmonic=1)
    B, spec = garlic(sys, exp)

Fitting (`esfit`)
*****************

.. code:: python

    import numpy as np
    from torchspin import esfit, FitOptions
    
    def model(p):
        sys_mod = SpinSystem(S=[0.5], g=[[p[0], p[1], p[2]]], lw=[1.0, 0.0])
        _, spc = pepper(sys_mod, exp, Options(GridSize=31, Verbosity=0))
        return spc.numpy()
    
    result = esfit(
        data_measured, model,
        p0=np.array([2.00, 2.10, 2.20]),
        lb=np.array([1.95, 2.05, 2.15]),
        ub=np.array([2.05, 2.15, 2.25]),
        options=FitOptions(method='global', progress='text'),
    )
    print('Best-fit g:', result.pfit)

``progress='text'`` prints RMSD, evaluation count and ETA as it runs, and
interrupting the kernel returns the best fit so far.

Population methods (swarm, genetic, grid, montecarlo, global) can evaluate each
generation in parallel worker processes with ``FitOptions(n_workers='auto')``. This
is off by default. Each worker is a separate Python process with its own torch
(about 0.3–0.5 GB), so ``'auto'`` uses the cores you were allocated, at most 32, and
no more workers than fit in half the free memory. A script that uses it needs the
usual ``if __name__ == '__main__':`` guard around the code that runs the fit.

Interactive fitting panel (`torchspin.fitgui`)
**********************************************

In JupyterLab (``pip install "torchspin[gui]"``), the same kind of fit runs in a panel
with parameter sliders, fix boxes, method controls, Start/Stop and a live plot of the
fit, residual and RMSD:

.. code:: python

    %matplotlib widget
    import numpy as np
    from torchspin import SpinSystem, Experiment, Options, pepper
    from torchspin.esfit import FitOptions
    from torchspin.fitgui import FitSession, FitPanel
    
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=1024, Harmonic=1)
    
    def model(p):  # p = [gx, gy, gz, Az (MHz), lwpp (mT)]
        sys = SpinSystem(S=[0.5], g=p[:3], Nucs='14N', A=[[18, 18, p[3]]], lwpp=[p[4], 0])
        return pepper(sys, exp, Options(GridSize=[31, 4], Verbosity=0))[1].numpy()
    
    B = np.linspace(330, 350, 1024)
    data = model([2.0085, 2.0060, 2.0025, 92.0, 0.35])          # or your measured spectrum
    data = data + 0.02 * np.abs(data).max() * np.random.default_rng(1).standard_normal(data.size)
    
    session = FitSession(data, model, p0=[2.011, 2.0085, 2.005, 101.0, 0.5],
                         vary=[0.004, 0.004, 0.004, 15.0, 0.25],
                         names=['gx', 'gy', 'gz', 'Az', 'lwpp'],
                         options=FitOptions(method='global', max_iter=100))
    FitPanel(session, x=B, xlabel='B (mT)').show()

Press **Start** to fit; **Stop** returns the best fit so far, and ``session.result``
holds the result afterwards. The panel's ``n_workers`` box accepts ``auto`` for parallel
population fits. The full walkthrough is
[``examples/notebooks/09_fit_gui.ipynb``](examples/notebooks/09_fit_gui.ipynb).

Differentiable spectrum
***********************

.. code:: python

    import torch
    from torchspin import differentiable_spectrum
    
    g = torch.tensor([2.0, 2.1, 2.2], dtype=torch.float64, requires_grad=True)
    B, spec = differentiable_spectrum(g, mwFreq_GHz=9.5, B_range=(300, 380),
                                      nPoints=512, lw_mT=1.0)
    loss = ((spec - target_spec) ** 2).sum()
    loss.backward()
    print('dL/dg:', g.grad)

Citation
********

If you use torchspin in published work, please cite both:

* **TorchSpin** — Follmer, A.H. *arXiv:2610.18924* (2026) https://arxiv.org/abs/2610.18924.
* **EasySpin**, the MATLAB toolbox this port is based on —
  Stoll, S. & Schweiger, A. *J. Magn. Reson.* **178** (2006) 42–55.


The MATLAB reference data ships in the repository but not in the PyPI package,
so the cross-validation suite only runs from a checkout.

Related
*******

- [EasySpin](https://easyspin.org) — the MATLAB reference implementation
- [PyTorch](https://pytorch.org) — the autograd engine

License
*******

MIT — see [LICENSE.md](LICENSE.md). torchspin is derived from EasySpin, which is
also MIT licensed; both copyrights are reproduced there.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   modules

