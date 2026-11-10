#!/bin/bash
# Verified benchmark campaign on a single workstation (no scheduler), one job at a time.
#
#   bash benchmarks/cluster/run_workstation_verified.sh                 # all stages
#   STAGES="env cpu matlab" bash benchmarks/cluster/run_workstation_verified.sh
#   STAGES="gpu" GPU_INDEX=1 bash benchmarks/cluster/run_workstation_verified.sh   # later, when a GPU is idle
#
# Protocol (BENCHMARK_VERIFICATION.md):
#   * every process is pinned to the same CPU set (CPUSET, default 0-31: 32 physical
#     cores of the 5995WX; SMT siblings are 64-95) with taskset;
#   * torchspin CPU: separate processes for 1, 8 and 32 threads, with torch and the
#     native pools (OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS) set to the same count;
#   * torchspin CUDA: one process, host side capped at 32 threads, one profiled
#     verification call and a CPU/CUDA consistency check per workload, unsupported
#     (CPU-only) workloads skipped;
#   * MATLAB/EasySpin: same CPU set, MATLAB's own threading (recorded in the JSON);
#   * one untimed warm-up and WARMUP/NREP replicates everywhere; every replicate saved.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
source ~/miniconda3/etc/profile.d/conda.sh && conda activate torchspin
ES=${EASYSPIN:-$HOME/code/EPR/EasySpin/easyspin}
OUT=${OUT:-benchmarks/results/workstation_$(date +%Y%m%d)_verified}; mkdir -p "$OUT"
STAGES=${STAGES:-"env cpu matlab gpu"}
CPUSET=${CPUSET:-0-31}
WARMUP=${WARMUP:-1}; NREP=${NREP:-5}
GPU_INDEX=${GPU_INDEX:-0}
LOG=$OUT/campaign.log
unset DISPLAY   # MATLAB -batch aborts on an unreachable X display
PIN="taskset -c $CPUSET"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
snapshot() {  # system state before/after a stage: load, GPU utilisation, foreign processes
  { echo "== $1 $(date '+%F %T')"; uptime; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    # NOTE: system_state.log lists every user's processes -- do not redistribute it.
    ps -eo user,pid,pcpu,etime,comm --sort=-pcpu | awk 'NR==1 || $3>5.0' | head -12; } >> "$OUT/system_state.log"
}

if [[ " $STAGES " == *" env "* ]]; then
  log "stage env -> $OUT/env.txt"
  {
    echo "date: $(date '+%F %T %Z')"; echo "host: $(hostname)"; echo "os: $(lsb_release -ds 2>/dev/null) kernel $(uname -r)"
    lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)"
    echo "memory: $(free -g | awk '/Mem:/{print $2" GB"}')"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
    nvidia-smi | grep -o "CUDA Version: [0-9.]*"
    echo "cpuset: $CPUSET ($(taskset -c $CPUSET python -c 'import os;print(len(os.sched_getaffinity(0)))') cpus)"
    python -c "import sys,torch,numpy,scipy; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version()); print('numpy', numpy.__version__, 'scipy', scipy.__version__); print('torch default threads', torch.get_num_threads())"
    python -c "import threadpoolctl; print('threadpoolctl', threadpoolctl.__version__)"
    echo "torchspin commit: $(git rev-parse HEAD)"; echo "torchspin branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "torchspin working tree:"; git status --short | sed 's/^/  /'
    echo "easyspin path: $ES"; echo "easyspin commit: $(git -C "$ES" rev-parse HEAD 2>/dev/null) $(git -C "$ES" log -1 --format=%ci 2>/dev/null)"
    echo "easyspin working tree: $(git -C "$ES" status --short 2>/dev/null | wc -l) modified files"
    echo "matlab: $(matlab -batch "disp(version)" 2>/dev/null | tail -1)"
  } | tee "$OUT/env.txt"
fi

if [[ " $STAGES " == *" cpu "* ]]; then
  snapshot "cpu-start"
  log "stage gpu-consistency tests"
  # recorded, not gating: an older checkout under comparison may fail these tests,
  # and that must not abort the timing stages
  $PIN python -m pytest torchspin/tests/test_gpu_consistency.py -q -p no:cacheprovider 2>&1 | tail -2 | tee "$OUT/gpu_consistency.txt" || true
  for th in 1 8 32; do
    log "stage cpu threads=$th"
    env OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th NUMEXPR_NUM_THREADS=$th \
      $PIN python benchmarks/workloads/run_workloads.py --device cpu --threads $th --warmup $WARMUP --repeat $NREP \
      --out "$OUT/cpu_threads${th}.json" 2>&1 | tee -a "$LOG"
  done
  snapshot "cpu-end"
fi

if [[ " $STAGES " == *" matlab "* ]]; then
  snapshot "matlab-start"
  log "stage matlab (warmup $WARMUP, nrep $NREP)"
  (cd benchmarks/workloads && env BENCH_WARMUP=$WARMUP BENCH_NREP=$NREP BENCH_OUT=matlab_workloads_verified.json \
     $PIN matlab -batch "addpath(genpath('$ES')); matlab_workloads" 2>&1 | tee -a "../../$LOG"
   mv matlab_workloads_verified.json "../../$OUT/matlab_workloads.json")
  snapshot "matlab-end"
fi

if [[ " $STAGES " == *" gpu "* ]]; then
  snapshot "gpu-start"
  log "stage cuda (GPU $GPU_INDEX, host threads 32)"
  env CUDA_VISIBLE_DEVICES=$GPU_INDEX OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 NUMEXPR_NUM_THREADS=32 \
    $PIN python benchmarks/workloads/run_workloads.py --device cuda --threads 32 --warmup $WARMUP --repeat $NREP \
    --out "$OUT/gpu.json" 2>&1 | tee -a "$LOG"
  snapshot "gpu-end"
fi
log "campaign stages [$STAGES] done: $OUT"
