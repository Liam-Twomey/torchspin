# torchspin v1.0 acceleration roadmap

Working branch: `dev/v1-acceleration`. This branch will run for a long time; it is
merged only once the measured gains are confirmed. The audit behind this roadmap is in
[CODE_REVIEW.md](CODE_REVIEW.md).

## Principles

- **Measure first.** No simulator rewrite lands without a before/after campaign run under
  the protocol in `benchmarks/results/workstation_20260904_verified/BENCHMARK_VERIFICATION.md`:
  - one warm-up and five replicates, median and IQR;
  - native thread pools capped together with torch;
  - CPU/CUDA output consistency checked on every run.
- **float64 stays the default.** Reduced precision (float32/complex64) is opt-in, and only
  on paths with a parity test against the float64 result at a documented tolerance.
- **The core dependencies stay torch + numpy + scipy.** Acceleration libraries go into
  optional extras (`[hpc]`, later `[gpu]`), and every one of them has a pure-torch fallback.
- **Results do not depend on the thread count or on how work is sharded.** Seeded
  stochastic simulations must give identical output however the work is grouped.

## Phases

| Phase | Scope | Status |
|---|---|---|
| 1 | Foundations: thread budget (`torchspin.parallel`), correctness fixes found in the audit (cardamom seeding, `sigeq` device, an empty GPU test), benchmark baselines and regression comparator, SLURM template | done (speed-neutral, as intended) |
| 2 | esfit: parallel finite-difference Jacobians (levmar, trf, lbfgsb, uncertainties), per-worker GPU pinning (`FitOptions.devices`), `simulate_batch` pool reuse; OpenMP wait policy and memory-aware `n_workers='auto'` (released as 0.6.3–0.6.5) | done; opt-in autograd Jacobian shelved (slower than finite differences) |
| 3 | pepper: device-resident post-processing (slot assembly, interpolation, strain accumulation, shared torch kernels with `pepper_autograd`), drop the duplicate strain-width diagonalization | **dropped**: CPU-first (see Decision) |
| 4 | saffron orientation batching and closed-form time domain; chili basis Liouvillians and an orientation pool; cardamom batched propagators and prefix scans | open (CPU work) |
| 5 | Cluster execution (`executor=` hook, sharding with exact reductions, fit checkpointing); opt-in float32 paths | open (CPU/cluster parts) |
| 6 | GPU eigensolver for matrices larger than 32×32; spidyan torch port | **dropped** (GPU part): CPU-first |

**Decision (2026-11-09): CPU first.** Measured on a cluster node, one 64-core CPU node with
`simulate_batch` workers produced 1473 Cu(II) spectra/s against 76 on one A100, and the
CUDA path leaves the GPU at 8–16 % utilization because most of the work is host-side.
For fits on laptops and workstations, worker processes are the effective route as well.
CPU nodes are also what most users have the most access to. GPU work (Phases 3 and 6) is
therefore dropped; `device='cuda'` stays supported and tested, but is not optimized
further. Future acceleration work targets CPU parallelism (Phases 4–5).

## Results log

Each entry: date, commit, host class ("CUDA workstation", "SLURM CPU node",
"SLURM GPU node", "Apple silicon"), the workloads, and the median change against the
stored baseline. Raw campaign data are not committed.

| Date | Commit | Host class | Change | Notes |
|---|---|---|---|---|
| 2026-11-08 | 4c9a009 vs v0.6.1 | 64-core workstation, 2× RTX 3090, idle, 32-core pinned set | CPU 1/8/32 threads: all 13 workloads within ±7 % of v0.6.1 | Phase 1 is neutral on speed, as intended: it fixes correctness and measurement. The thread-backend per-worker budget was 2.1× slower at 8 threads and was reverted (4c9a009); A/B rerun 11.34 s vs 11.40 s. |
| 2026-11-08 | 0.6.2 fixes | same host, CUDA | Mn(II) matrix 34.7 → 1.16 s, Cu/2N 8.6 → 0.83 s, dense strain OOM → 1.86 s; CPU/CUDA NRMSE ≤ 3e-13 | CUDA is still hybrid (≤ 20 % of wall time in kernels), the Phase 3 target. Released as 0.6.2. |
| 2026-11-08 | 72a7809 vs v0.6.1 | SLURM CPU nodes (shared, 224-core) | not usable: unchanged code varied up to 2.8× between runs | Shared-node timing needs a paired A/B design on one whole node. |
| 2026-11-08 | 6c03935 vs v0.6.1 | SLURM CPU node, whole 224-core node, paired A B A B, taskset-pinned | CPU 1/8/32/64 threads: dev/v0.6.1 median ratio 0.88–1.23; every ratio whose replicate ranges are disjoint lies in 0.90–1.01 | Confirms the workstation result: Phase 1 is speed-neutral. The 8-thread 1.19–1.23 ratios (fit loop, strain) have overlapping ranges, i.e. node noise. v0.6.1 crashed twice in esfit_swarm_threads_matrix on the pepper thread race fixed in 0.6.2. |
| 2026-11-08 | b17faab, default vs `OMP_WAIT_POLICY=PASSIVE` | SLURM CPU node (whole 224-core node) and 64-core workstation, paired A B A B | 8 threads: fit loop 9.46 → 2.37 s / 3.30 → 2.15 s, levmar 13.16 → 4.08 s / 5.81 → 3.41 s; no disjoint-range regression above 1.01× at 1–64 threads | Root cause of "more threads = slower fits": OpenMP spin-wait. Shipped in 0.6.3. Neutral at 32/64 threads, where fits are still slower than at 1 thread (separate cause, open). |
| 2026-11-08 | b17faab | Apple-silicon laptop, 8 threads | parallel finite-difference probes (`n_workers='auto'`): levmar 1.47 → 1.02 s, trf 1.55 → 0.84 s | Phase 2b. Cluster A/B pending. |
| 2026-11-09 | f085e6c | SLURM CPU node (whole 224-core node), 6 runs per arm | parallel finite-difference probes: levmar 1.56–1.67×, trf 2.6–3.2× faster at 8/32/64 threads (8 threads: levmar 9.70 → 5.81 s, trf 10.47 → 3.26 s) | Phase 2b confirmed. These runs imported torch before torchspin, so the parent process still spin-waited (baselines slow); found the gap that 0.6.4 closes. The harness now imports torchspin first and records `OMP_WAIT_POLICY`. |
| 2026-11-09 | 49ee809 | 64-core workstation, 2× RTX 3090, 16-core pinned set | `FitOptions.devices`: Cu/2N matrix swarm fit, 8 CUDA workers 13.41 s on one GPU → 11.86 s on two; 16 CPU workers 14.53 s | Phase 2d. GPU utilization 8–16 %: the CUDA path is host-bound until Phase 3. On the small nitroxide model CUDA workers lose to CPU workers (4.8 s vs 2.0 s). |
| 2026-11-09 | — | Apple-silicon laptop | autograd Jacobian, nitroxide 4 parameters × 1024 points: reverse mode 46 s, forward mode fails (internal assert), double-backward jvp 1.06 s, scalar gradient 0.11 s; finite differences 0.17 s | Phase 2c **shelved**: a backward pass through pepper costs several forward simulations, so autograd Jacobians are slower than finite differences at these sizes. Revisit only for lbfgsb with many parameters. |
| 2026-11-09 | — | 64-core workstation | one fit-model call with PASSIVE: 105 ms (1 thread), 119 ms (8), 130 ms (32) | The remaining 32-thread slowdown is thread wake-up overhead on a model too small to parallelize. **Accepted**: fits take seconds; parallel probes in worker processes sidestep it. |
| 2026-11-09 | 69ab1d6 | SLURM CPU node (64 cores) vs one A100 | `ml.simulate_batch`, Cu(II): 1 process 42 spectra/s; 64 workers 197 → **1473** spectra/s once the pool is reused (was restarted on every call); one A100 76 spectra/s (host-bound, 16 % GPU utilization) | ML data generation belongs on CPU nodes. Phase 3 (GPU post-processing) stays parked. |
| 2026-11-09 | — | laptop, 1 thread | Cu(II) + 2 ¹⁴N with random rhombic parameters: 14–20× slower than the axial template. 5.5× = 55 vs 10 orientations (symmetry); 2.9× = more forbidden transitions above `Threshold`. `Threshold=1e-2`: 3.6× faster, NRMSE 0.4 % | Documented in pitfalls ("Training sets of large spin systems"). The rest is eigh of 72×72 matrices, CPU-only on GPUs (Phase 6). |

**Open from Phase 1:**
- **Fixed in 0.6.3/0.6.4 at 8 threads (OpenMP spin-wait).** At 32–64 threads serial fits stay about 25 % slower than at 1 thread: wake-up overhead on small models, accepted. Earlier note:
- Serial fits get slower when threads are added: on the cluster node, v0.6.1's 20-call fit loop takes 2.2 s at 1 thread, 6.6 s at 8 and 4.0 s at 32; esfit levmar takes 3.6 s, 11.0 s and 6.4 s (the workstation showed the same pattern, 1.5 s at 8 vs 2.8 s at 32). Oversubscription inside one process: the first Phase 2 target.
- esfit's thread backend is about 15× slower than its process backend on the same swarm fit, at every thread count.
