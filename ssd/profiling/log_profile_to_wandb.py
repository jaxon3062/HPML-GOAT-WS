"""Parse an SSD_PROFILE=1 bench log and attach per-step timings to its W&B run.

Usage:
    python profiling/log_profile_to_wandb.py LOGFILE [--skip 10] [--run-id ID]

The W&B run id is read from the log ("wandb.ai/.../runs/<id>") unless given.
Logs, under the "profile/" prefix:
  - per-step line charts: handshake, verify, postprocess, total, toks, hit
  - per-step draft charts (async only): service, build_tree, decode_tree, populate, total
  - summary means, hit/miss split, and a tokens-per-ms efficiency figure
  - tables with every parsed row, and histograms
"""
import argparse
import re
import statistics as st

import wandb

KV = re.compile(r"(\w+)=([\d.]+)(?:ms)?")
HITS = re.compile(r"hits=(\d+)/(\d+)")
RUN_ID = re.compile(r"/runs/([a-z0-9]{6,})")


def parse(path):
    target, draft, run_id = [], [], None
    with open(path, errors="replace") as f:
        for line in f:
            if run_id is None:
                m = RUN_ID.search(line)
                if m:
                    run_id = m.group(1)
            if line.startswith("[PROFILE target]"):
                row = {k: float(v) for k, v in KV.findall(line) if k != "hits"}
                m = HITS.search(line)
                if m:
                    hit, n = int(m.group(1)), int(m.group(2))
                    row["hit_frac"] = hit / n
                target.append(row)
            elif line.startswith("[PROFILE draft] service="):
                draft.append({k: float(v) for k, v in KV.findall(line)})
    return target, draft, run_id


def mean(rows, k):
    vals = [r[k] for r in rows if k in r]
    return st.mean(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("--skip", type=int, default=10, help="warmup steps to drop")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--entity", default="HPML-GOAT")
    ap.add_argument("--project", default="ssd")
    args = ap.parse_args()

    target, draft, run_id = parse(args.logfile)
    run_id = args.run_id or run_id
    assert run_id, "could not find a W&B run id in the log; pass --run-id"
    target, draft = target[args.skip:], draft[args.skip:]
    assert target, "no [PROFILE target] lines found (was SSD_PROFILE=1 set?)"

    run = wandb.init(entity=args.entity, project=args.project, id=run_id, resume="must")
    wandb.define_metric("profile/step")
    wandb.define_metric("profile/target/*", step_metric="profile/step")
    wandb.define_metric("profile/draft_step")
    wandb.define_metric("profile/draft/*", step_metric="profile/draft_step")

    t_keys = ["handshake", "verify", "postprocess", "total", "toks", "hit_frac"]
    for i, r in enumerate(target):
        wandb.log({"profile/step": i, **{f"profile/target/{k}": r[k] for k in t_keys if k in r}})
    d_keys = ["service", "build_tree", "decode_tree", "populate", "total"]
    for i, r in enumerate(draft):
        wandb.log({"profile/draft_step": i, **{f"profile/draft/{k}": r[k] for k in d_keys if k in r}})

    summary = {f"profile_mean_{k}_ms": mean(target, k) for k in ["handshake", "verify", "postprocess", "total"]}
    summary["profile_mean_toks_per_step"] = mean(target, "toks")
    summary["profile_toks_per_ms"] = summary["profile_mean_toks_per_step"] / summary["profile_mean_total_ms"]
    summary["profile_handshake_frac"] = summary["profile_mean_handshake_ms"] / summary["profile_mean_total_ms"]
    summary["profile_num_steps"] = len(target)
    if any("hit_frac" in r for r in target):
        hits = [r for r in target if r.get("hit_frac") == 1.0]
        miss = [r for r in target if r.get("hit_frac") == 0.0]
        summary["profile_hit_rate"] = mean(target, "hit_frac")
        for name, rows in [("hit", hits), ("miss", miss)]:
            summary[f"profile_{name}_handshake_ms"] = mean(rows, "handshake")
            summary[f"profile_{name}_total_ms"] = mean(rows, "total")
            summary[f"profile_{name}_toks"] = mean(rows, "toks")
    if draft:
        for k in d_keys:
            summary[f"profile_draft_mean_{k}_ms"] = mean(draft, k)
        summary["profile_draft_over_verify"] = summary["profile_draft_mean_total_ms"] / summary["profile_mean_verify_ms"]
    for k, v in summary.items():
        if v is not None:
            run.summary[k] = v

    cols = sorted({k for r in target for k in r})
    wandb.log({
        "profile/target_table": wandb.Table(columns=["step"] + cols,
                                            data=[[i] + [r.get(c) for c in cols] for i, r in enumerate(target)]),
        "profile/handshake_hist": wandb.Histogram([r["handshake"] for r in target]),
        "profile/total_hist": wandb.Histogram([r["total"] for r in target]),
    })
    if draft:
        wandb.log({"profile/draft_table": wandb.Table(columns=["draft_step"] + d_keys,
                                                      data=[[i] + [r.get(c) for c in d_keys] for i, r in enumerate(draft)])})
    log_art = wandb.Artifact(f"profile-log-{run_id}", type="profile-log")
    log_art.add_file(args.logfile)
    run.log_artifact(log_art)

    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()})
    run.finish()


if __name__ == "__main__":
    main()
