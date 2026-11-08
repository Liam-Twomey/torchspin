#!/bin/bash
# Generic SLURM job-array template for torchspin benchmark campaigns.
#
# Each array task runs the workload suite at one CPU thread count (or, with
# DEVICE=cuda, on one GPU). Fill in the site-specific lines marked EDIT before
# submitting; nothing cluster-specific is assumed here.
#
#   sbatch --array=0-3 benchmarks/cluster/slurm_array_template.sbatch
#   DEVICE=cuda sbatch --array=0 --gres=gpu:1 --cpus-per-task=8 benchmarks/cluster/slurm_array_template.sbatch
#
# Outputs go to $OUT_ROOT/<array job id>/ (default: the submit directory). Point
# OUT_ROOT at fast scratch storage for large campaigns, and copy the JSON files
# you want to keep afterwards. They record host metadata, so keep them outside
# the repository. Reduce them with benchmarks/analysis/compare_baseline.py.
#
#SBATCH --job-name=torchspin-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=%x-%A_%a.out
# EDIT: account / partition / constraint / QOS for your site, for example
##SBATCH --account=<account>
##SBATCH --partition=<partition>
##SBATCH --constraint=<node-type>     # pin one node generation so tasks are comparable
set -euo pipefail

# EDIT: environment activation for your site, or pass ENV_SETUP=<script> to have
# a site script sourced here (module loads, conda/venv activation)
# module load <python-or-conda-module>
# source activate <env>    or    source <venv>/bin/activate
if [ -n "${ENV_SETUP:-}" ]; then
  # shellcheck disable=SC1090
  source "$ENV_SETUP"
fi

THREADS_LIST=(${THREADS_LIST:-1 8 32 64})   # one array task per entry
DEVICE=${DEVICE:-cpu}
REPEAT=${REPEAT:-5}
WARMUP=${WARMUP:-1}
ONLY=${ONLY:-}                               # comma-separated workload names, empty = all
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
OUT_ROOT=${OUT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}

idx=${SLURM_ARRAY_TASK_ID:-0}
th=${THREADS_LIST[$idx]}
if [ "$th" -gt "${SLURM_CPUS_PER_TASK:-$th}" ]; then
  echo "thread count $th exceeds the allocation (${SLURM_CPUS_PER_TASK} CPUs)" >&2
  exit 2
fi
out_dir="$OUT_ROOT/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
mkdir -p "$out_dir"
tag="${DEVICE}_threads${th}"

{
  echo "date: $(date -Is)"
  echo "array task: $idx  threads: $th  device: $DEVICE  cpus allocated: ${SLURM_CPUS_PER_TASK:-?}"
  python - <<'PY'
import torch, numpy, scipy, torchspin
from torchspin.parallel import cpu_budget
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'torchspin', torchspin.__version__)
print('cpu_budget', cpu_budget())
PY
  git -C "$REPO" rev-parse HEAD 2>/dev/null || true
  [ "$DEVICE" = cpu ] || nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
} > "$out_dir/env_${tag}.txt" 2>&1

cd "$REPO"
if [ "$DEVICE" != cpu ]; then
  python -m pytest torchspin/tests/test_gpu_consistency.py -q -p no:cacheprovider \
    > "$out_dir/gpu_consistency_${tag}.txt" 2>&1 || true
fi

# run_workloads.py overwrites OMP/MKL/OPENBLAS/NUMEXPR/VECLIB thread variables with --threads
python benchmarks/workloads/run_workloads.py --device "$DEVICE" --threads "$th" \
  --warmup "$WARMUP" --repeat "$REPEAT" ${ONLY:+--only "$ONLY"} \
  --out "$out_dir/${tag}.json" --save-outputs "$out_dir/${tag}_outputs.npz"
echo "DONE ${tag}" >> "$out_dir/markers.txt"
