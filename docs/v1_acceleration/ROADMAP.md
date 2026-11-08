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
| – | – | – | – | Phase 1 baseline pending |
