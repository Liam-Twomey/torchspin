# torchspin documentation

torchspin is a differentiable PyTorch port of EasySpin. The API mirrors EasySpin's (`SpinSystem`,
`Experiment`, `Options`, `pepper`, `garlic`, `chili`, `salt`, `saffron`, `cardamom`, `esfit`, …), so the
[EasySpin documentation](https://easyspin.org/easyspin/documentation/) is the reference for the physics
and the meaning of parameters; this page collects torchspin's own material.

## Tutorials (executed notebooks, `examples/notebooks/`)

| Notebook | What it shows |
|---|---|
| [01 quickstart](../examples/notebooks/01_torchspin_quickstart.ipynb) | Spin systems, Hamiltonians, powder (`pepper`) and solution (`garlic`) spectra, level diagrams |
| [02 GPU and autograd](../examples/notebooks/02_gpu_and_autograd.ipynb) | `device='cuda'`, `differentiable_spectrum`, gradients with respect to spin-Hamiltonian parameters, gradient-based fitting |
| [03 spectral fitting](../examples/notebooks/03_spectral_fitting.ipynb) | `esfit` on synthetic powder and solution spectra with `method='global'` |
| [04 benchmarks](../examples/notebooks/04_benchmarks.ipynb) | MATLAB/EasySpin vs torchspin timings (threads, GPU, worker processes) from the 2026-09 campaign, plus live timing |
| [05 real-world fitting](../examples/notebooks/05_fitting_real_world.ipynb) | Bad starting guesses, loss-landscape slices, simplex traps vs global search, 5-parameter stress test |
| [06 blue copper](../examples/notebooks/06_blue_copper_fitting.ipynb) | Rhombic Cu(II) case study: simplex, Levenberg–Marquardt and global head to head |
| [07 EasySpin parity](../examples/notebooks/07_easyspin_parity.ipynb) | Overlays against stored EasySpin outputs (pepper, garlic, chili, saffron HYSCORE, cardamom) with cosine similarities |
| [08 pepper playground](../examples/notebooks/08_pepper_playground.ipynb) | Simulate, add noise, fit with every `esfit` method with live progress; compare recovery, residual and time |
| [09 fit GUI](../examples/notebooks/09_fit_gui.ipynb) | Interactive panel (`torchspin.fitgui`): sliders, fix boxes, Start/Stop, live fit/residual/RMSD trace |

Scripts with the same coverage as EasySpin's example collection live in `examples/{solidstate,liquids,
slowmotion,endor,fitting,magnetometry}` (`examples/tests/compare_with_matlab_refs.py` checks them
against MATLAB references). `examples/fitting/fit_bruker_cw.py` loads a Bruker file with `eprload`,
reads the acquisition parameters and fits it.

## Machine learning

`torchspin.ml` packages the plumbing for training and inference on top of the
differentiable simulators: `ParamSpec` (which spin-system fields are free,
their ranges and transforms, with `pack`/`unpack` to and from a flat tensor),
`simulate`/`simulate_batch`, `SpectrumDataset` (an `IterableDataset` that
samples parameters, simulates, normalizes, resamples and adds noise
deterministically for a given seed) and the differentiable losses
`cosine_loss` / `rmsd`.

## Reference material in the repository

- [README](../README.md) — installation, quick start, feature list, validation status.
- [KNOWN_LIMITATIONS](../KNOWN_LIMITATIONS.md) — what is and is not ported, per simulator, with the evidence.
- [Pitfalls](pitfalls.md) — valid settings that quietly degrade a result: perturbation theory at large A/ν,
  binned garlic spectra in fits, the lsq autoscale sign, satellite-assignment minima, spidyan step and frame limits.
- [CHANGELOG](../CHANGELOG.md) — what changed when, including the parity and performance rounds.
- [Benchmark report](../benchmarks/results/BENCHMARK_REPORT.md) — MATLAB/EasySpin vs torchspin on nine
  workloads, before and after the 2026-09 optimizations, with profiles and the measurement procedure.
- [Verified matched-host campaign](https://github.com/follmerlab/torchspin/blob/main/benchmarks/results/workstation_20260904_verified/BENCHMARK_VERIFICATION.md)
  — the authoritative performance measurement: five replicates per point, medians with IQRs, and a
  per-workload device audit showing which runs actually used CUDA. Supersedes the benchmark report
  wherever the two disagree.

## Running on a workstation

- CPU: torchspin uses all cores through thread pools inside the simulators (`torch.set_num_threads(n)`
  caps it). Population-based fits add `FitOptions(n_workers='auto')` (one process per core;
  `pip install "torchspin[fit]"` for models defined in notebooks).
- GPU: `Options(device='cuda')`; worthwhile for autograd and batched fitting, not for single simulations.
- Progress and early stop: `FitOptions(progress='text', max_time=…)`; a kernel interrupt returns the best
  fit so far.

## Tests

`python -m pytest` runs the full suite (2659 tests, including the MATLAB-validation modules that
compare against the stored references in `tests/data`; no MATLAB installation needed).
