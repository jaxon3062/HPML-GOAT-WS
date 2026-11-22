# Qwen3-32B profiling: AR vs SD vs SSD (2026-11-22)

Target Qwen3-32B (TP=2, GPUs 0-1), draft Qwen3-0.6B. Same settings as the team's
`ar_qwen32b_tp2`, `sd_qwen32b_k5_tp2` (k=5) and `ssd_qwen32b_tp2_3gpu` (k=7, f=3, draft on GPU 2),
b=1, temp=0, `--all` datasets, but short runs (128 output tokens) for profiling.

W&B (HPML-GOAT/ssd, group `profiling`):

| Run | What | W&B id |
|---|---|---|
| prof_ssd_qwen32b | SSD_PROFILE per-step timers, 550 steps | 43qt6o4m |
| prof_sd_qwen32b | SSD_PROFILE per-step timers, 606 steps | gf9gu2k1 |
| nsys_ssd_qwen32b | Nsight timeline, 59 steps | bwhcawyi |
| nsys_sd_qwen32b | Nsight timeline, 59 steps | yiw9fcoy |
| nsys_ar_qwen32b | Nsight timeline, 59 steps | 5gxfsqbs |

## Results

Per decode step, from the Nsight runs (no extra syncs):

| | AR | SD (k=5) | SSD (k=7, f=3) |
|---|---|---|---|
| Step time | 26.9 ms | 51.7 ms | 40.0 ms |
| Verify (target forward) | 26.9 ms | 29.3 ms | 29.7 ms |
| Handshake (getting draft tokens) | - | 22.3 ms | 10.2 ms |
| Tokens per step (from timer runs) | 1 | 3.37 | 3.73 |
| Implied throughput | 37 tok/s | 65 tok/s | 93 tok/s |
| Team's measured throughput (full run) | 37.9 | 73.6 | 104.7 |

SSD detail (timer run): cache hit rate 75%. Handshake is 5.0 ms on a hit vs 23.1 ms on a miss;
tokens/step 3.94 on a hit vs 3.09 on a miss. The draft's work per round (37.9 ms: 28.0 ms tree decode
+ 4.1 build + 5.7 service) is 1.34x the verify time (28.4 ms), so the target waits a few ms even on hits.
Nsight: 70% of draft tree time overlaps target verification; target GPUs spend 23% of time waiting for the draft.

SD detail: the draft runs only on GPU 0, so GPU 1 sits idle ~50% of the time while GPU 0 drafts.

## Files

- `*_timeline.png`: per-GPU timelines (12 decode steps from the middle of the captured window).
- `ssd/profiling/`: `nsys_to_wandb.py`, `log_profile_to_wandb.py`, `run_qwen32b_profiling.sh`, `prof_env.sh`.
- Code changes (all opt-in): `ssd/ssd/utils/nvtx.py` plus NVTX ranges in `engine/step.py`,
  `engine/draft_runner.py` and `engine/llm_engine.py`. `SSD_NVTX=1` turns on the ranges;
  `SSD_PROFILE_WINDOW=a:b` calls cudaProfilerStart/Stop at decode steps a and b.
- Raw `.nsys-rep` traces: W&B artifacts on the nsys runs, and `~jack200336/ssd-prof/nsys/` on gpubox.

## Reproduce (on gpubox)

The scripts expect the `ssd/` directory checked out at `~/ssd-prof` with its own `uv sync` environment
(plus `uv pip install -U wandb matplotlib`), and `prof_env.sh` copied to `~/ssd-prof/prof_env.sh`.

```bash
cd ~/ssd-prof && source prof_env.sh
bash profiling/run_qwen32b_profiling.sh   # ~20 min once weights are in page cache
```
Note: `/etc/nccl.conf` sets a 4-GPU NCCL graph file; 2-/3-GPU runs need
`NCCL_GRAPH_FILE=/data/nccl_empty_graph.xml` (set in prof_env.sh).
