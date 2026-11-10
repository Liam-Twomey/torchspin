# Benchmark baselines

Each JSON file here is a reduced benchmark campaign: for every workload, the median
wall time and interquartile range, plus the torchspin commit, torch version, device
and thread count it was measured with. Host names, paths and CPU affinity lists are
stripped, and files are named by a generic host class
(`cuda-workstation-cpu32.json`, `slurm-cpu-node-cpu64.json`, ...).

Baselines are only comparable on the same host class with the same thread count and
device.

## Refresh a baseline

1. Run a campaign under the replicated protocol (one warm-up, five replicates):

   ```bash
   python benchmarks/workloads/run_workloads.py --threads 32 --warmup 1 --repeat 5 \
       --out /path/outside/the/repo/campaign_cpu32.json
   ```

   Keep the raw campaign JSON outside the repository. It records host metadata.

2. Reduce it to a baseline:

   ```bash
   python benchmarks/analysis/compare_baseline.py save /path/outside/the/repo/campaign_cpu32.json \
       --label cuda-workstation-cpu32 --out benchmarks/baselines/cuda-workstation-cpu32.json
   ```

## Check a change against a baseline

```bash
python benchmarks/analysis/compare_baseline.py compare new_campaign.json \
    benchmarks/baselines/cuda-workstation-cpu32.json
```

A workload is reported as a regression when its median is more than 20 % slower
(`--tolerance`) *and* the interquartile ranges of the two runs do not overlap. If any
workload regresses, the exit status is 1.
