"""Opt-in NVTX ranges for Nsight Systems timelines.

Enabled with SSD_NVTX=1 (inherited by spawned worker/draft processes).
When disabled, nvtx_range() is a no-op context manager.

Optional profile window: SSD_PROFILE_WINDOW="start:end" makes the engine call
cudaProfilerStart() at decode step `start` and cudaProfilerStop() at step `end`,
for use with `nsys profile --capture-range=cudaProfilerApi`.
"""
import os
from contextlib import nullcontext

import torch

NVTX_ENABLED = os.environ.get("SSD_NVTX", "0") == "1"


def nvtx_range(name: str):
    if NVTX_ENABLED:
        return torch.cuda.nvtx.range(name)
    return nullcontext()


def profile_window():
    """Return (start, end) step indices from SSD_PROFILE_WINDOW, or None."""
    spec = os.environ.get("SSD_PROFILE_WINDOW")
    if not spec:
        return None
    start, end = (int(x) for x in spec.split(":"))
    assert 0 <= start < end, f"bad SSD_PROFILE_WINDOW={spec}"
    return start, end
