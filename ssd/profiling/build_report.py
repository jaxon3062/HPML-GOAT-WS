"""Rebuild the 'SSD Experiment Tracking Dashboard' W&B report (HPML-GOAT/ssd).

Usage: python profiling/build_report.py [--dry-run]
Backs up the current report spec to profiling/report_backup_<ts>.json before saving.
"""
import argparse
import json
import time

import wandb
import wandb_workspaces.reports.v2 as wr

ENTITY, PROJECT = "HPML-GOAT", "ssd"
URL = "https://forge.coreweave.com/HPML-GOAT/ssd/reports/SSD-Experiment-Tracking-Dashboard--VmlldzoxODM0Njg4OA"

BENCH = ["ar_qwen32b_tp2", "sd_qwen32b_k5_tp2", "ssd_qwen32b_tp2_3gpu"]
PROF = ["prof_sd_qwen32b", "prof_ssd_qwen32b"]
NSYS = ["nsys_ar_qwen32b", "nsys_sd_qwen32b", "nsys_ssd_qwen32b"]
COLOR = {"ar": "#888780", "sd": "#1D9E75", "ssd": "#7F77DD"}


def run_ids(names):
    api = wandb.Api()
    out = {}
    for r in api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": {"$in": names}}):
        out[r.name] = r.id  # latest wins if duplicated
    missing = set(names) - set(out)
    assert not missing, f"runs not found: {missing}"
    return out


def runset(title, names):
    ids = run_ids(names)
    settings = {}
    for name, rid in ids.items():
        kind = "ssd" if "ssd" in name else ("sd" if "sd_" in name else "ar")
        settings[rid] = wr.RunSettings(color=COLOR[kind])
    quoted = ", ".join(f"'{n}'" for n in names)
    return wr.Runset(entity=ENTITY, project=PROJECT, name=title,
                     filters=f"Metric('displayName') in [{quoted}]", run_settings=settings)


def non_smoke_runset():
    """Benchmark runs only: drop smoke tests and profiling runs (their timers distort throughput)."""
    api = wandb.Api()
    smoke = sorted({r.name for r in api.runs(f"{ENTITY}/{PROJECT}")
                    if r.name.startswith(("smoke", "prof_", "nsys_"))})
    quoted = ", ".join(f"'{n}'" for n in smoke)
    return wr.Runset(entity=ENTITY, project=PROJECT, name="Benchmark runs (no smoke tests or profiling runs)",
                     filters=f"Metric('displayName') not in [{quoted}]" if smoke else "")


def L(x, y, w=8, h=6):
    return wr.Layout(x=x, y=y, w=w, h=h)


def build_blocks():
    return [
        wr.P("Qwen3-32B target + Qwen3-0.6B draft on 4×A100 80GB PCIe. "
             "AR = autoregressive (2 GPUs) · SD = speculative decoding, k=5 (2 GPUs) · "
             "SSD = async speculative speculative decoding, k=7, fan-out 3 (2 target GPUs + 1 draft GPU)."),
        wr.TableOfContents(),

        wr.H2("1. Throughput"),
        wr.PanelGrid(
            runsets=[runset("Full benchmark (512 prompts × 512 tokens)", BENCH)],
            panels=[
                wr.BarPlot(title="End-to-end throughput (tok/s)", metrics=["official_end_to_end_throughput"],
                           orientation="v", layout=L(0, 0)),
                wr.BarPlot(title="Tokens gained per decode step", metrics=["metrics_avg_accepted_suffix_lens_with_recovery"],
                           orientation="v", layout=L(8, 0)),
                wr.BarPlot(title="Time per decode step (ms)", metrics=["metrics_avg_target_step_time_ms"],
                           orientation="v", layout=L(16, 0)),
            ],
        ),

        wr.H2("2. Speculation quality"),
        wr.PanelGrid(
            runsets=[runset("Full benchmark: SD vs SSD", BENCH[1:])],
            panels=[
                wr.ScalarChart(title="SSD cache hit rate", metric="metrics_avg_cache_hits",
                               font_size="large", layout=L(0, 0)),
                wr.BarPlot(title="Tokens per step: all steps vs cache hits",
                           metrics=["metrics_avg_accepted_suffix_lens_with_recovery", "metrics_avg_accepted_suffix_lens_on_hit"],
                           orientation="v", layout=L(8, 0, 16)),
                wr.LinePlot(title="Accepted tokens per step (histogram)",
                            y=["metrics_accepted_suffix_lens_with_recovery_histogram"], layout=L(0, 6, 12)),
                wr.LinePlot(title="Accepted tokens per step on cache hit (histogram)",
                            y=["metrics_accepted_suffix_lens_on_hit_histogram"], layout=L(12, 6, 12)),
            ],
        ),

        wr.H2("3. GPU timelines (Nsight Systems)"),
        wr.P("Purple = target forward · green = draft · orange = draft redo on cache miss · "
             "light gray = target waiting for draft · dark gray = NCCL."),
        wr.PanelGrid(
            runsets=[runset("Nsight Systems traces (60 decode steps each)", NSYS)],
            panels=[
                wr.MediaBrowser(title="Timeline: 12 decode steps", media_keys=["nsys/timeline"],
                                num_columns=1, mode="gallery", gallery_axis="run", layout=L(0, 0, 24, 12)),
                wr.BarPlot(title="Mean decode step (ms)", metrics=["nsys_mean_step_ms"],
                           orientation="v", layout=L(0, 12)),
                wr.BarPlot(title="Target GPU 0: busy vs waiting for draft",
                           metrics=["nsys_gpu0_target_busy_frac", "nsys_gpu0_target_wait_for_draft_frac"],
                           orientation="v", layout=L(8, 12)),
                wr.BarPlot(title="Draft work hidden behind verify (SSD)",
                           metrics=["nsys_draft_tree_overlapped_with_verify_frac"],
                           orientation="v", layout=L(16, 12)),
            ],
        ),

        wr.H2("4. Per-step breakdown"),
        wr.P("Short runs with per-step timers on (timers add syncs, so use these for ratios, not throughput)."),
        wr.PanelGrid(
            runsets=[runset("Per-step profiling (16 prompts × 128 tokens)", PROF)],
            panels=[
                wr.BarPlot(title="Mean step breakdown (ms)",
                           metrics=["profile_mean_handshake_ms", "profile_mean_verify_ms", "profile_mean_postprocess_ms"],
                           orientation="v", layout=L(0, 0, 12)),
                wr.BarPlot(title="SSD: cache hit vs miss step (ms)",
                           metrics=["profile_hit_total_ms", "profile_miss_total_ms"],
                           orientation="v", layout=L(12, 0, 12)),
                wr.LinePlot(title="Handshake time per step (ms)", x="profile/step",
                            y=["profile/target/handshake"], smoothing_factor=0.6, layout=L(0, 6, 12)),
                wr.LinePlot(title="Verify time per step (ms)", x="profile/step",
                            y=["profile/target/verify"], smoothing_factor=0.6, layout=L(12, 6, 12)),
                wr.LinePlot(title="SSD draft: tree decode time per round (ms)", x="profile/draft_step",
                            y=["profile/draft/decode_tree"], smoothing_factor=0.6, layout=L(0, 12, 12)),
                wr.LinePlot(title="Tokens gained per step", x="profile/step",
                            y=["profile/target/toks"], smoothing_factor=0.8, layout=L(12, 12, 12)),
            ],
        ),

        wr.H2("5. All benchmark runs"),
        wr.PanelGrid(
            runsets=[non_smoke_runset()],
            panels=[
                wr.BarPlot(title="Throughput, all runs (tok/s)", metrics=["official_end_to_end_throughput"],
                           orientation="h", max_runs_to_show=40, layout=L(0, 0, 24, 8)),
            ],
        ),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    report = wr.Report.from_url(URL)
    backup = f"profiling/report_backup_{int(time.time())}.json"
    with open(backup, "w") as f:
        json.dump(report._to_model().model_dump(mode="json", by_alias=True), f)
    print("backed up current report to", backup)

    report.title = "Speculative Speculative Decoding: throughput and profiling"
    report.description = "AR vs SD vs SSD on Qwen3-32B: throughput, speculation quality, GPU timelines, per-step profiling."
    report.width = "fixed"
    report.blocks = build_blocks()
    if args.dry_run:
        print("dry run: built", len(report.blocks), "blocks; not saved")
        return
    report.save()
    print("saved:", report.url)


if __name__ == "__main__":
    main()
