#!/usr/bin/env bash
# Profiling runs for Qwen3-32B (target) + Qwen3-0.6B (draft), matching the team's
# ar_qwen32b_tp2 / sd_qwen32b_k5_tp2 / ssd_qwen32b_tp2_3gpu configs, but short.
# Runs sequentially so GPUs are never shared. Logs -> logs/, traces -> nsys/.
set -u
source "$HOME/ssd-prof/prof_env.sh"
cd "$HOME/ssd-prof"
mkdir -p logs nsys
rm -f logs/ALL_DONE

COMMON="--qwen --size 32 --b 1 --temp 0 --all --output_len 128"
GROUP="--wandb --group profiling"
NSYS="nsys profile -t cuda,nvtx --trace-fork-before-exec=true --cuda-graph-trace=node \
      --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true"
SSD="--gpus 3 --spec --async --k 7 --f 3"
SD="--gpus 2 --spec --k 5"
AR="--gpus 2"

run() {  # name, cmd...
  local name=$1; shift
  echo "=== $(date) START $name" | tee -a logs/run_all.log
  "$@" > "logs/$name.log" 2>&1
  echo "=== $(date) END $name exit=$?" | tee -a logs/run_all.log
  pkill -9 -f "bench/bench.py" 2>/dev/null; sleep 5
}

# 1) Per-step timers (SSD_PROFILE adds syncs: use for breakdowns, not throughput)
run prof_ssd_qwen32b env SSD_PROFILE=1 SSD_PROFILE_DRAFT=1 python -O bench/bench.py $COMMON $SSD --numseqs 4 $GROUP --name prof_ssd_qwen32b
python profiling/log_profile_to_wandb.py logs/prof_ssd_qwen32b.log >> logs/run_all.log 2>&1
run prof_sd_qwen32b env SSD_PROFILE=1 python -O bench/bench.py $COMMON $SD --numseqs 4 $GROUP --name prof_sd_qwen32b
python profiling/log_profile_to_wandb.py logs/prof_sd_qwen32b.log >> logs/run_all.log 2>&1

# 2) Nsight Systems timelines (no SSD_PROFILE, so overlap is not distorted)
run nsys_ssd_qwen32b env SSD_NVTX=1 SSD_PROFILE_WINDOW=50:110 $NSYS -o nsys/nsys_ssd_qwen32b python -O bench/bench.py $COMMON $SSD --numseqs 1 $GROUP --name nsys_ssd_qwen32b
python profiling/nsys_to_wandb.py nsys/nsys_ssd_qwen32b.nsys-rep --log logs/nsys_ssd_qwen32b.log --mode async >> logs/run_all.log 2>&1
run nsys_sd_qwen32b env SSD_NVTX=1 SSD_PROFILE_WINDOW=50:110 $NSYS -o nsys/nsys_sd_qwen32b python -O bench/bench.py $COMMON $SD --numseqs 1 $GROUP --name nsys_sd_qwen32b
python profiling/nsys_to_wandb.py nsys/nsys_sd_qwen32b.nsys-rep --log logs/nsys_sd_qwen32b.log --mode sync >> logs/run_all.log 2>&1
run nsys_ar_qwen32b env SSD_NVTX=1 SSD_PROFILE_WINDOW=50:110 $NSYS -o nsys/nsys_ar_qwen32b python -O bench/bench.py $COMMON $AR --numseqs 1 $GROUP --name nsys_ar_qwen32b
python profiling/nsys_to_wandb.py nsys/nsys_ar_qwen32b.nsys-rep --log logs/nsys_ar_qwen32b.log --mode ar >> logs/run_all.log 2>&1

touch logs/ALL_DONE
