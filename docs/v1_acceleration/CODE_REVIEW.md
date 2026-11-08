# torchspin Code Review — GPU and Parallel-CPU Acceleration Audit
**Version:** 0.6.1 (main @ 1f03559), reviewed on branch `dev/v1-acceleration`
**Last updated:** 2026-11-08
**Status:** Exploration only — no source changes. Input for the v1.0 acceleration plan.

---

## What It Is

torchspin is a PyTorch port of EasySpin: CW EPR (pepper, garlic, chili, salt), pulse
EPR (saffron, saffron_thyme, spidyan), stochastic dynamics (cardamom), and fitting
(esfit). Several simulators are differentiable. This review asks where the code can be
made faster on three targets: Apple-silicon laptops, a 64-core CUDA reference
workstation (2× RTX 4090), and SLURM HPC clusters. Every claim cites the file and line
in the 0.6.1 tree where it was checked.

---

## Baseline: what is measured today

The latest controlled campaign is `benchmarks/results/workstation_20260904_verified/`.
It ran at v0.2.2, so the code has changed since, but the hot spots listed below are
still present in 0.6.1. It used one warm-up and five replicates, reports the median,
pinned 32 cores, and capped the native thread pools.

| Workload | EasySpin (s) | torchspin best CPU (s) | torchspin CUDA (s) | CUDA mode |
|---|---:|---:|---:|---|
| pepper Cu/2N matrix (72×72) | 0.292 | 0.847 (32 t) | 0.674 | hybrid, eigh 100 % CPU |
| pepper Mn S=5/2 matrix (36×36) | 3.03 | 1.07 (32 t) | 0.954 | hybrid, eigh 100 % CPU |
| pepper perturbation, dense grid | 0.0818 | 0.0616 (1 t) | 0.162 | hybrid, 0.2 % kernels |
| pepper strain summation | 0.814 | 0.994 (8 t) | 1.23 | hybrid, 7610 D→H copies |
| pepper 20-call fit loop | 2.5 | 1.29 (8 t); **5.28 at 32 t** | 1.77 | hybrid, 30660 D→H copies |
| chili nitroxide, 2 nuclei | 1.15 | 0.157 (flat 1–32 t) | n/a | CPU only |
| chili powder + potential | 4.53 | 3.08 (flat 1–32 t) | n/a | CPU only |
| saffron HYSCORE 512 | 0.66 | 0.752 (32 t) | 0.228 | hybrid, 16.6 % kernels |
| cardamom 200 traj × 1000 steps | 20.4 | 9.97 (8 t) | n/a | CPU only |

**Main point:** none of the CUDA runs is GPU-bound. GPU kernels never take more than 17 %
of wall time. The rest goes to host-side NumPy/SciPy work, eigendecompositions sent back
to the CPU, and per-orientation Python loops. Three workloads have no CUDA path at all.

---

## Architecture as it affects acceleration

### Device and precision model
- Device is chosen per call through `Options.device`, which defaults to `'cpu'`
  (`experiment.py:291`). If CUDA is unavailable it falls back to CPU (`:337-347`).
  There is no global default device and no `'auto'`.
- Simulators that honor `device`: pepper (`pepper.py:1269`), salt (`salt.py:265`),
  saffron through its own `SaffronOptions.device` (`saffron.py:108, 621`), and the
  autograd helpers (`autograd.py:372, 401, 484`).
- No device support: garlic, chili, curry, cardamom, spidyan and saffron_thyme contain
  no device references.
- Nearly all physics paths are float64/complex128. Resonance searches (tolerance
  mwFreq·2e-6, `resfields_batch.py:122`), Lanczos (`chili_sle.py:868-907`) and
  long propagator products all need double precision.

### Linear-algebra layer (`torchspin/_linalg.py`)
- **CPU:** batched `eigh` is split into chunks and run on a thread pool of
  single-threaded LAPACK calls (`:111-125`). This is the one CPU-parallel primitive.
- **CUDA:** batches of matrices larger than 32×32 are copied to the host, solved by the
  CPU pool, and copied back (`:174-178`), because of cuSOLVER's `syevjBatched` limit.
- **Autograd path:** `_EighDegenerateSafe.forward` calls the *unpooled* `_safe_eigh`
  (`:145`), so CPU gradient batches run serially.
- **Global lock:** `torch.set_num_threads` is changed and restored under a global
  `_threads_lock` (`:35, 118-124, 213-219`).
- **Callers:** curry, salt, strainwidth, resfreqs_matrix and resfields_batch/hybrid use
  it. **saffron does not**; it calls `torch.linalg.eigh` directly (`saffron.py:1048, 1113`).

### Apple MPS (torch 2.13)
- MPS has no float64, no `linalg.eigh`, and no `linalg.matrix_exp`. FFT and complex64
  do work.
- MPS is therefore not a realistic target for the eigen-based simulators. On Apple silicon
  the gains have to come from the CPU cores. The only
  MPS-plausible stages are those with low precision needs (cardamom 3×3 propagation,
  exciteprofile, spectrum accumulation), and each needs a complex64 parity check first.

---

## Findings by simulator (ranked within each section)

### pepper / resfields (CW powder)
1. **The spectrum axis is always on the CPU.** `x` and `spec` are created without a device
   (`pepper.py:1442-1443`). Every stage after the resonance search therefore runs on the
   host, whatever `device` is set to.
2. **Slot assembly copies each orientation to the host separately.**
   - `_transition_slots` (`pepper.py:219-277`) loops over orientations in Python and
     calls `.detach().cpu().numpy()` several times per orientation (`:243-244, 257-259`).
   - `resfields_batch` already holds flat sorted tensors and splits them into lists
     (`resfields_batch.py:477-498`).
   - Fix: build the slot matrix on the device with `unique` + scatter. Low risk, as long
     as the `last_keys` slot ordering is preserved.
3. **Interpolation and strain summation run per slot in NumPy/SciPy.**
   - Per slot: `_interp_slot` ×3 (`pepper.py:1693-1699`), NumPy facet statistics
     (`:1702-1719`), and one `_gaussian_bins` call (`:1720`).
   - `_gaussian_bins` is torch internally but converts from NumPy and syncs on every call
     (`:319-323`).
   - The no-strain projection uses `np.add.at` (`_projecttriangles` `:930-980`).
   - Torch versions of these kernels already exist in `pepper_autograd.py`:
     `gaussian_bins_t` (`:422`), `projecttriangles_t` (`:202`), `_interp_slots_t` (`:344`).
     Unifying the two copies removes duplicated code and moves the stage to the device.
4. **Strain widths repeat a diagonalization.**
   - `compute_strain_widths_batch` re-diagonalizes every resonance at B0
     (`strainwidth.py:224-226`), but only uses ⟨u|Op|u⟩ and ⟨v|Op|v⟩ (`:313-334`).
   - The eigenvectors already exist at `resfields_batch.py:255`.
   - Removing the repeat roughly halves the eigh calls in strain simulations. It needs a
     parity test, because the stored vectors were computed before the Newton step.
5. **Eigensolver for large Hilbert spaces on GPU.**
   - Matrices larger than 32×32 make a round trip to the host per knot or candidate chunk
     (`resfields_batch.py:126-185, 239-243`).
   - Options, cheapest first:
     - (a) Overlap the CPU-pool eigh with GPU post-processing.
     - (b) Per-matrix `zheevd` on several CUDA streams, cuSOLVER `Xsyevd` batched, or
       MAGMA batched.
     - (c) A warm-started batched Jacobi kernel. H(B) varies smoothly along B, so
       neighboring-knot eigenvectors are a good start.
     - (d) Rayleigh–Ritz or perturbative Newton steps, with an exact solve only on failure.
   - (c) and (d) carry high correctness risk near level crossings.
6. **Smaller items.**
   - `_preselect_pairs` runs 10 serial eighs (`pepper.py:600-612`).
   - The site/frame orientation loop copies per orientation (`:1558-1572`).
   - `convspec` uses NumPy FFT (`convspec.py:135`).
   - `torch.compile` is off unless `TORCHSPIN_COMPILE=1` (`_compile.py:35`).
7. **Process-level parallelism.**
   - Components × isotopologues run serially (`pepper.py:1092-1095`; `batch.py:103-107`).
   - They are independent, so they map directly onto multiple GPUs, a process pool, or
     SLURM array tasks.

### saffron (pulse EPR, fast path)
1. **One Python iteration per orientation, with tiny kernels inside.**
   - The loop is `saffron.py:1000`. Inside it:
     - nuclear-subspace `eigh` on N×N matrices with N ≈ 2–6 (`:1113`);
     - per-pathway Python dispatch;
     - `scatter_add_` per pathway (`saffron_peaks.py:99-100, 191-192`);
     - host syncs every orientation (`float(...)` `:1001-1004, 1035`; `.item()` `:1062`;
       `.tolist()` `:1078-1081`).
   - This explains the 16.6 % kernel share.
   - Fix: make orientation a leading batch dimension. The (nOri, N, N) eigh stays on
     cuSOLVER's batched path, and the `OriThreshold` `continue` becomes a mask.
   - Expected to be the largest saffron gain.
   - Moderate risk: ProductRule products must stay per orientation before summing
     (`:1516-1535`).
2. **The time-domain path loops over every time point.**
   - It is forced whenever parameters require gradients (`saffron.py:624-631`).
   - HYSCORE is an nP1×nP2 scalar double loop (`saffron_peaks.py:443-447`), so every
     element becomes an autograd node. The same pattern appears at `:391, 416, 473,
     513-515, 558-575`.
   - Fix: the closed form `UUt^k = exp(-2πiνk·dt)` turns this into a batched einsum. The
     algebra is exact, so risk is low.
   - This is the main gradient-path speedup.
3. **Precision is hard-coded** to complex128 (`saffron.py:621, 972-991`). Making the dtype
   configurable, then validating complex64, is the only route to MPS for saffron.

### spidyan / saffron_thyme (real pulses)
- Both are entirely NumPy/SciPy with no device argument.
- Serial loops:
  - per orientation (`saffron_thyme.py:239`), with `.cpu().numpy()` at `:243-245`;
  - per acquisition point (`spidyan.py:997`), re-propagating from the initial state;
  - one `scipy.linalg.expm` per waveform step (`:912-918, 1240-1280`);
  - sequential propagator products (`:1232-1234`) and step-by-step U·σ·U† (`:1351-1356,
    1455-1460`).
- Proposed rewrite:
  - (a) port to torch with a device argument;
  - (b) eigen-propagators for free evolution (diagonalize once; each detected point becomes
    a phase multiply);
  - (c) build pulse tables with one batched `matrix_exp`;
  - (d) tree or prefix-scan products. Relaxation updates are affine, so they can be scanned
    in a homogeneous (n²+1) form;
  - (e) batch orientations × acquisition points × phase cycles.
- This is the largest rewrite, with moderate-to-high risk because of event reordering
  (`spidyan.py:1048-1059`).
- complex128 is required (GHz Zeeman phases over 10³–10⁵ steps), so CPU and CUDA only.

### chili (slow-motion SLE)
Profiled on an Apple-silicon development machine for `chili_powder_potential`: 3.37 s.
1. **Serial orientation loop** (`chili_sle.py:1122-1167`), 19 orientations when a potential
   is set (`:983, :53`).
   - Lanczos takes about 50 % of the time: each of 100–210 steps is a single-threaded scipy
     `csr_matvec` (N=5616, nnz≈846k).
   - Rebuilding the Liouvillian per orientation takes about 37 %: `rbos` `:318-370`, 988
     `hil2liouv` calls, about 2300 sparse kron+add calls (`:501-517`).
   - Fix (a): L(Ω) is linear in the Wigner D²(Ω) (`:349-358`). Build the 25 basis
     Liouvillians once; each orientation then needs only a dense coefficient product.
   - Fix (b): a process pool over orientations.
   - Fix (c), GPU: batched block-diagonal SpMM Lanczos in complex128 on cuSPARSE. A single
     5k SpMV is launch-bound, so batching is what makes the GPU worthwhile.
   - Risk: the output scale comes from the last orientation (`:1139-1141, 1170`), copied
     from EasySpin, so a parallel version has to keep it.
2. **Direct/explicit solver** does one `spsolve` per field point per orientation
   (`:1155-1166`). Field points are independent, so they suit a pool or array job and can
   reuse the symbolic factorization.
3. `diffsuperop` and `chili_eqpopvec` are O(n²) Python loops (`:701-753`). They are cheap
   at the default LLMK but grow with it.
4. `_chili_legacy` (`chili.py:567`) is unreachable.

### cardamom / stochastic trajectories
1. **The ISTOs method calls `scipy.linalg.expm` once per (step, trajectory)**
   (`_cardamom_propagatedm.py:380-385`) inside Python double loops (`:361-364, 393-399`).
   Batched `matrix_exp` or eigh should give the largest per-call gain in the package.
2. **The fast path has per-step Python loops** (`_cardamom_propagatedm.py:108-114`) and
   NumPy↔torch layout churn (`:97, 142, 185-187`).
   - rho₀ = ½I (`:105`), so rho_t can be written with associative propagator products and
     computed by a parallel prefix scan.
   - With no potential, the quaternion increments are independent and can be scanned too.
   - With a potential, the trajectory is serial in time but parallel over about 20k
     trajectories, which suits a GPU.
   - Finite-difference potential gradients (`stochtraj_diffusion.py:287-304`) could become
     analytic or autograd.
   - 3×3 unitaries might tolerate complex64, which would make MPS possible after a parity
     check.
3. **`stochtraj_jump`** fills trajectories element by element in nested Python loops
   (`:172-178, 187-190`). Vectorizing it is low risk.

### esfit (fitting)
1. **Only population methods run in parallel.**
   - They use `_eval_population` (`esfit.py:1583-1652`): a cached spawn pool that caps BLAS
     threads before `Pool()` (`:1539-1552`) and ships objectives with cloudpickle.
   - Serial today:
     - simplex, powell;
     - the lbfgsb and trf finite-difference gradients;
     - the levmar finite-difference Jacobian (`:2063-2068`, `:2103-2107`);
     - the uncertainty central differences (`:2297-2316`).
   - Sending those columns through `_eval_population` is a near-free P- to 2P-fold gain
     for the local methods.
2. **No autograd Jacobian.** pepper, salt, saffron and garlic are differentiable, and
   `torch.autograd.functional.jacobian` over `differentiable_spectrum` is already tested
   (`tests/test_autograd.py:207-213`), but esfit always uses finite differences.
3. **`torch.func.vmap` is not feasible yet.** `_EighDegenerateSafe` is an old-style
   `autograd.Function` with no vmap/jvp rules, and pepper has NumPy detours. Batched
   population evaluation would need a `simulate_population` API that stacks parameter
   sets on the orientation axis (`pepper.py:1269-1415`).
4. **Every evaluation forces a host sync** (`.detach().cpu().numpy()`, `esfit.py:860-861`).
5. **Process workers do not pin a GPU** (`_pool_init` `:1523-1525`), so a second GPU is
   never used.

---

## Cross-cutting: thread budget and nested parallelism

- `n_workers='auto'` uses `os.cpu_count()` (`esfit.py:1576-1579`). It ignores CPU affinity
  and `SLURM_CPUS_PER_TASK`, so it oversubscribes inside SLURM allocations and `taskset`.
- Inner pools size themselves from the process-wide torch thread count
  (`_linalg.py:52-53, 210`), whatever the caller is doing. Small problems therefore fan out
  across all cores. This is the most likely cause of the fit-loop result (5.3 s at 32
  threads vs 1.3 s at 8).
- In esfit's thread backend, `_threads_lock` makes concurrent evaluations queue for the
  pool. While the lock is held, `set_num_threads(1)` silently drops the other threads to
  one thread.
- The package does not use `threadpoolctl` anywhere, so NumPy/SciPy BLAS threading inside
  pepper interpolation is uncontrolled. `ml.simulate_batch` spawns a fresh pool on every
  call without capping BLAS (`ml.py:229-258`).
- **Recommendation:** one `torchspin.parallel` module that:
  - resolves the core budget from affinity, then SLURM variables, then `cpu_count`;
  - hands out outer workers × inner threads within that budget;
  - carries the inner budget in a `contextvar` instead of a global set/restore;
  - applies the env/threadpoolctl caps everywhere.

## Cluster / SLURM readiness

- `benchmarks/cluster/slurm_cpu.sh` exports `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`
  (`:13`). `run_workloads._apply_thread_env` then uses `os.environ.setdefault`
  (`run_workloads.py:56-61`), so the "1-thread" and "4-thread" sweep points actually run
  with 16 native threads. OPENBLAS is never capped, and the script points to a missing
  PERF_PLAN.md (`:8`).
- `slurm_gpu.sh` uses a single GPU.
- Missing: no array jobs, no multi-node runs, no fit checkpoint/restart, no distributed
  executor (no submitit, dask-jobqueue, Ray or mpi4py in the repo).
- **Naturally independent work units:**
  - orientation shards (the summed spectra/FIDs are linear, so the reduction is exact);
  - components and isotopologues;
  - chili field points (explicit mode);
  - spidyan acquisition points;
  - esfit population members and multistarts;
  - parameter sweeps.

---

## Known Issues and Observations

Correctness and test integrity:

1. **cardamom seeding depends on thread count (correctness).**
   - The orientation group size is `G = f(torch.get_num_threads())` (`cardamom.py:211-212`).
   - Every group calls `_generate_local_trajectory` with the same `par.seed` (`:218, 375,
     392`), and global dynamics use `seed+1` (`:223, 253`).
   - Result: a seeded run gives different spectra on hosts with different thread counts,
     orientations in different groups get identical trajectories, and on the serial path
     every orientation shares one ensemble.
   - Fix: `np.random.SeedSequence(seed).spawn(nOrients)`, one stream per orientation. This
     must be fixed before any sharding.
2. **The garlic GPU-consistency test checks nothing.** `test_garlic_cpu_vs_cuda` passes
   `device='cuda'` (`tests/test_gpu_consistency.py:160-164`), but garlic has no device
   handling, so it compares CPU against CPU.
3. **`sigeq.py:70` creates `torch.eye(N, dtype=complex128)` with no device.** This will
   raise a device-mismatch error on GPU when polarization is used.
4. **The SLURM thread sweep is mislabeled** (see Cluster above).
5. **The CPU autograd eigh is unpooled** (`_linalg.py:145`), so gradient fits do not use
   the multi-core path that forward simulations use.

Performance infrastructure:

6. **No regression gating.** CI runs `pytest -m "not slow"` only
   (`.github/workflows/python-tests.yml:54`). There are no stored baselines, no comparator,
   and no esfit end-to-end workload in `benchmarks/workloads/workloads.py:97-130`.
7. **The `--procs` benchmark includes pool spawn time in each replicate**
   (`run_workloads.py:309-311`).
8. **GPU consistency coverage** is pepper, saffron and autograd only. salt and curry are
   not covered.

---

## What It Can Do (Current State)

- **Apple silicon:** CPU only, as a correctness and development host.
- **CPU workstation:** the thread-pooled eigh scales well for large batched matrix
  problems. Fit loops and small problems lose speed at high thread counts.
- **CUDA:** pepper, salt and saffron run, but as hybrid CPU/GPU in every benchmarked
  workload. chili, cardamom, garlic, spidyan and curry have no GPU path.
- **HPC:** one-node scripts only, and the CPU sweep is mislabeled.
- **Compared with EasySpin:** torchspin's best CPU configuration is faster in 6 of 9
  workloads under the warm protocol.
- **GPU benefit today** is real only for large orientation batches of small matrices
  (saffron HYSCORE 0.23 s vs 0.66 s in EasySpin) and for autograd workloads.

---

## Immediate Improvement Priorities (ordered by expected impact ÷ risk)

1. **Thread-budget module and affinity/SLURM-aware `'auto'`.** Fixes the 4× regression
   at high thread counts and is a prerequisite for every nested-parallel item. Low risk.
2. **Fix cardamom seeding and the vacuous garlic GPU test.** These are correctness items,
   and both are prerequisites for sharding and for trusting GPU parity.
3. **Benchmark harness:** stored baselines, a regression comparator, esfit end-to-end
   workloads, and a fixed SLURM sweep. Everything after this point needs measurement.
4. **Parallel finite-difference Jacobians in esfit** (levmar, trf, lbfgsb, uncertainties),
   then an opt-in autograd Jacobian.
5. **Make pepper device-resident after the resonance search:**
   - device-side slot assembly;
   - batched interpolation and strain accumulation;
   - unify with the torch kernels already in `pepper_autograd.py`;
   - remove the duplicate strain-width eigh.
6. **Batch saffron over orientations, and replace the per-element `sf_evolve` loops with a
   closed-form time domain.**
7. **chili:**
   - precompute the D²-basis Liouvillians;
   - add an orientation process pool;
   - later, batched cuSPARSE Lanczos on CUDA.
8. **cardamom:**
   - batched `matrix_exp` for ISTOs;
   - device torch plus a prefix scan for the fast path;
   - vectorize the jump model.
9. **Multi-GPU and cluster execution:**
   - per-worker GPU pinning;
   - an `executor=` hook (concurrent.futures interface, submitit or dask-jobqueue on SLURM);
   - orientation and component sharding with an exact reduction;
   - fit checkpointing.
10. **GPU eigensolver for matrices larger than 32×32** (multi-stream `zheevd`, MAGMA, or
    warm-started Jacobi) and the spidyan torch rewrite. These are the largest and riskiest
    items and belong late in v1.0.
