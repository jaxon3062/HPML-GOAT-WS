"""Turn an Nsight Systems trace of an SSD bench run into a GPU timeline + metrics in W&B.

Usage:
    python profiling/nsys_to_wandb.py REPORT.nsys-rep --log LOGFILE [--steps 12] [--mode async|sync|ar]

Requires the run to have been made with SSD_NVTX=1 (range names target:* / draft:* / "step N").
Exports REPORT.sqlite (if missing), classifies every GPU kernel by the NVTX range that
launched it, draws one lane per GPU, computes overlap/idle metrics, and attaches the
image, metrics and the .nsys-rep (as an artifact) to the run whose id is in LOGFILE.
"""
import argparse
import bisect
import os
import re
import sqlite3
import subprocess
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

COLORS = {
    "verify": "#7F77DD",       # target forward / verification
    "draft_tree": "#1D9E75",   # async draft building its speculation tree
    "draft_jit": "#D85A30",    # draft regenerating on a cache miss
    "draft_sync": "#1D9E75",   # sync SD: draft running on the target GPU
    "comm": "#5F5E5A",         # NCCL kernels doing real communication (e.g. TP all-reduce)
    "wait": "#D3D1C7",         # NCCL recv spinning while target waits for draft (= idle)
    "other": "#B4B2A9",        # postprocess, sampling, unlabelled
}
LABELS = {
    "verify": "Target forward (verify)",
    "draft_tree": "Draft: build tree",
    "draft_jit": "Draft: JIT on cache miss",
    "draft_sync": "Draft (on target GPU)",
    "comm": "NCCL communication",
    "wait": "Waiting for draft (NCCL recv)",
    "other": "Other kernels",
}
RUN_ID = re.compile(r"/runs/([a-z0-9]{6,})")


def export_sqlite(rep):
    db = os.path.splitext(rep)[0] + ".sqlite"
    if not os.path.exists(db):
        subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", db, rep], check=True)
    return db


def load(db):
    con = sqlite3.connect(db)
    q = lambda s: con.execute(s).fetchall()  # noqa: E731
    tables = {r[0] for r in q("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "CUPTI_ACTIVITY_KIND_KERNEL" in tables, "no kernels in trace (capture window empty?)"
    assert "NVTX_EVENTS" in tables, "no NVTX ranges in trace (was SSD_NVTX=1 set?)"
    nvtx = q("""
        SELECT n.start, n.end, n.globalTid, COALESCE(n.text, s.value)
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
        WHERE n.end IS NOT NULL AND n.end > n.start""")
    kernels = q("""
        SELECT k.start, k.end, k.deviceId, s.value, r.globalTid, r.start
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
          ON r.correlationId = k.correlationId AND (r.globalTid >> 24) = (k.globalPid >> 24)""")
    return nvtx, kernels


def build_range_index(nvtx):
    by_tid = defaultdict(list)
    for start, end, tid, name in nvtx:
        if name and (name.startswith("target:") or name.startswith("draft:")):
            by_tid[tid].append((start, end, name))
    idx = {}
    for tid, rs in by_tid.items():
        rs.sort()
        idx[tid] = ([r[0] for r in rs], rs)
    return idx


def innermost(idx, tid, t):
    if tid not in idx or t is None:
        return None
    starts, rs = idx[tid]
    i = bisect.bisect_right(starts, t) - 1
    best = None
    # walk back over candidate ranges that started before t; innermost = latest start containing t
    for j in range(i, max(-1, i - 64), -1):
        s, e, name = rs[j]
        if s <= t <= e:
            if best is None or s > best[0]:
                best = (s, name)
            if name == "draft:jit":  # jit is always nested inside draft:service
                return name
    return best[1] if best else None


def classify(name, rng, mode):
    if "nccl" in name.lower():
        return "wait" if (rng == "target:handshake" and mode == "async") else "comm"
    if rng is None:
        return None  # resolved later (TP worker ranks have no NVTX ranges)
    if rng == "draft:jit":
        return "draft_jit"
    if rng.startswith("draft:"):
        return "draft_tree" if rng != "draft:service" else "other"
    if rng in ("target:verify", "target:ar_decode", "target:prefill"):
        return "verify"
    if rng == "target:handshake":
        return "draft_sync" if mode == "sync" else "comm"
    return "other"


def union_len(iv):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def overlap_len(a, b):
    """Total time where interval sets a and b overlap (both pre-merged not required)."""
    def merge(iv):
        out = []
        for s, e in sorted(iv):
            if out and s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return out
    a, b = merge(a), merge(b)
    i = j = tot = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--log", required=True, help="bench stdout log (to find the W&B run id)")
    ap.add_argument("--mode", choices=["async", "sync", "ar"], required=True)
    ap.add_argument("--steps", type=int, default=12, help="number of decode steps to draw")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--entity", default="HPML-GOAT")
    ap.add_argument("--project", default="ssd")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    db = export_sqlite(args.report)
    nvtx, kernels = load(db)
    idx = build_range_index(nvtx)

    steps = sorted((s, e, n) for s, e, _, n in nvtx if n and n.startswith("step "))
    assert len(steps) >= 2, "no 'step N' NVTX ranges captured"
    win_steps = steps[1:]  # skip first (may be partially captured)
    t0_all, t1_all = win_steps[0][0], win_steps[-1][1]
    mid = max(0, len(win_steps) // 2 - args.steps // 2)
    draw = win_steps[mid:mid + args.steps]
    d0, d1 = draw[0][0], draw[-1][1]

    per_dev = defaultdict(lambda: defaultdict(list))  # dev -> cat -> [(s,e)]
    pids_with_ranges = {tid >> 24 for tid in idx}
    unlabelled = []
    for ks, ke, dev, name, tid, rstart in kernels:
        if ke < t0_all or ks > t1_all:
            continue
        cat = classify(name, innermost(idx, tid, rstart), args.mode)
        iv = (max(ks, t0_all), min(ke, t1_all))
        if cat is None:
            unlabelled.append((dev, tid, iv))
        else:
            per_dev[dev][cat].append(iv)

    draft_devs = [d for d, c in per_dev.items() if c.get("draft_tree") or c.get("draft_jit")]
    # TP worker ranks (rank>0) run without NVTX ranges; they only execute target forwards.
    for dev, tid, iv in unlabelled:
        worker = tid is not None and (tid >> 24) not in pids_with_ranges
        per_dev[dev]["verify" if worker and dev not in draft_devs else "other"].append(iv)
    devs = sorted(per_dev)
    win = t1_all - t0_all

    # ---- metrics over the whole captured window ----
    m = {"nsys_window_ms": win / 1e6, "nsys_steps": len(win_steps),
         "nsys_mean_step_ms": win / 1e6 / len(win_steps)}
    for d in devs:
        role = "draft" if d in draft_devs and args.mode == "async" else "target"
        allk = [iv for c, ivs in per_dev[d].items() if c != "wait" for iv in ivs]
        m[f"nsys_gpu{d}_{role}_busy_frac"] = union_len(allk) / win
        m[f"nsys_gpu{d}_{role}_comm_frac"] = union_len(per_dev[d]["comm"]) / win
        if per_dev[d]["wait"]:
            m[f"nsys_gpu{d}_{role}_wait_for_draft_frac"] = union_len(per_dev[d]["wait"]) / win
        for c in ("verify", "draft_tree", "draft_jit", "draft_sync"):
            if per_dev[d][c]:
                m[f"nsys_gpu{d}_{c}_frac"] = union_len(per_dev[d][c]) / win
    target_devs = [d for d in devs if d not in draft_devs or args.mode != "async"]
    if args.mode == "async" and draft_devs and target_devs:
        tv = per_dev[target_devs[0]]["verify"]
        dt = per_dev[draft_devs[0]]["draft_tree"]
        if dt:
            m["nsys_draft_tree_overlapped_with_verify_frac"] = overlap_len(tv, dt) / max(1, union_len(dt))
    hs = [e - s for s, e, _, n in nvtx if n == "target:handshake" and t0_all <= s <= t1_all]
    vf = [e - s for s, e, _, n in nvtx if n == "target:verify" and t0_all <= s <= t1_all]
    if hs:
        m["nsys_mean_handshake_ms"] = sum(hs) / len(hs) / 1e6
    if vf:
        m["nsys_mean_verify_ms"] = sum(vf) / len(vf) / 1e6

    # ---- timeline image ----
    fig, ax = plt.subplots(figsize=(14, 0.9 * len(devs) + 1.6), dpi=150)
    order = ["wait", "other", "comm", "verify", "draft_sync", "draft_tree", "draft_jit"]
    for row, d in enumerate(reversed(devs)):
        for c in order:
            bars = [((max(s, d0) - d0) / 1e6, (min(e, d1) - max(s, d0)) / 1e6)
                    for s, e in per_dev[d][c] if e > d0 and s < d1]
            if bars:
                ax.broken_barh(bars, (row - 0.35, 0.7), facecolors=COLORS[c], linewidth=0)
    for s, _, _ in draw:
        ax.axvline((s - d0) / 1e6, color="#888780", lw=0.4, ls=":")
    names = []
    for d in reversed(devs):
        role = "draft" if d in draft_devs and args.mode == "async" else "target"
        names.append(f"GPU {d} ({role})")
    ax.set_yticks(range(len(devs)), names)
    ax.set_xlabel("time (ms)  -  dotted lines = decode step boundaries")
    ax.set_xlim(0, (d1 - d0) / 1e6)
    ax.set_title(f"{os.path.basename(args.report)}: {len(draw)} decode steps ({args.mode})")
    used = {c for d in devs for c in per_dev[d] if per_dev[d][c]}
    legend = [Patch(color=COLORS[c], label=LABELS[c]) for c in order if c in used]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=min(4, len(legend)), frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    png = os.path.splitext(args.report)[0] + "_timeline.png"
    fig.savefig(png)
    print("wrote", png)
    print({k: round(v, 4) for k, v in m.items()})

    if args.no_wandb:
        return
    import wandb
    run_id = args.run_id
    if run_id is None:
        with open(args.log, errors="replace") as f:
            for line in f:
                mm = RUN_ID.search(line)
                if mm:
                    run_id = mm.group(1)
                    break
    assert run_id, "no W&B run id found in log; pass --run-id"
    run = wandb.init(entity=args.entity, project=args.project, id=run_id, resume="must")
    run.log({"nsys/timeline": wandb.Image(png)})
    for k, v in m.items():
        run.summary[k] = v
    art = wandb.Artifact(f"nsys-{run_id}", type="nsys-report")
    art.add_file(args.report)
    run.log_artifact(art)
    run.finish()


if __name__ == "__main__":
    main()
