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
| 1 | Foundations: thread budget (`torchspin.parallel`), correctness fixes found in the audit (cardamom seeding, `sigeq` device, an empty GPU test), benchmark baselines and regression comparator, SLURM template | in progress |
| 2 | esfit: parallel finite-difference Jacobians (levmar, trf, lbfgsb, uncertainties), opt-in autograd Jacobian, per-worker GPU pinning | planned |
| 3 | pepper: device-resident post-processing (slot assembly, interpolation, strain accumulation, shared torch kernels with `pepper_autograd`), drop the duplicate strain-width diagonalization | planned |
| 4 | saffron orientation batching and closed-form time domain; chili basis Liouvillians and an orientation pool; cardamom batched propagators and prefix scans | planned |
| 5 | Cluster and multi-GPU execution (`executor=` hook, sharding with exact reductions, fit checkpointing); opt-in float32 paths | planned |
| 6 | GPU eigensolver for matrices larger than 32×32; spidyan torch port | planned |

## Results log

Each entry: date, commit, host class ("CUDA workstation", "SLURM CPU node",
"SLURM GPU node", "Apple silicon"), the workloads, and the median change against the
stored baseline. Raw campaign data are not committed.

| Date | Commit | Host class | Change | Notes |
|---|---|---|---|---|
| 2026-11-08 | 4c9a009 vs v0.6.1 | 64-core workstation, 2× RTX 3090, idle, 32-core pinned set | CPU 1/8/32 threads: all 13 workloads within ±7 % of v0.6.1 | Phase 1 is neutral on speed, as intended: it fixes correctness and measurement. The thread-backend per-worker budget was 2.1× slower at 8 threads and was reverted (4c9a009); A/B rerun 11.34 s vs 11.40 s. |
| 2026-11-08 | 0.6.2 fixes | same host, CUDA | Mn(II) matrix 34.7 → 1.16 s, Cu/2N 8.6 → 0.83 s, dense strain OOM → 1.86 s; CPU/CUDA NRMSE ≤ 3e-13 | CUDA is still hybrid (≤ 20 % of wall time in kernels), the Phase 3 target. Released as 0.6.2. |
| 2026-11-08 | 72a7809 vs v0.6.1 | SLURM CPU nodes (shared, 224-core) | not usable: unchanged code varied up to 2.8× between runs | Shared-node timing needs a paired A/B design on one whole node. |

**Open from Phase 1:**
- The serial 20-call fit loop is still slower at 32 threads than at 8 (2.8 s vs 1.5 s): oversubscription inside one process, not addressed yet.
- esfit's thread backend is about 15× slower than its process backend on the same swarm fit, at every thread count.
