#!/usr/bin/env python3
"""Is the torch profiler blind inside an mp_tuner-style worker process?

mp_tuner does `mp.set_start_method("spawn", force=True)` then `mp.Pool(...)`.
This isolates the question from anything MoE-specific: run a plain matmul under
torch.profiler in (1) the main process and (2) a spawned Pool worker, and count
the CUDA device rows each one sees.

  main sees CUDA rows, worker sees 0  -> profiler cannot attach in the worker;
                                         that alone explains every us=0.
  both see rows                       -> the blindness is specific to the FlyDSL
                                         launch path, not the process model.

Run inside m3-flydsl:
  python3 /work/probe_profiler_in_child.py
"""
import multiprocessing as mp

import torch
import torch.profiler as tpf


def count_cuda_rows(tag):
    torch.cuda.set_device(0)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()

    with tpf.profile(
        activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
        profile_memory=False,
        with_stack=False,
        with_modules=True,
    ) as prof:
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()

    # exactly what get_trace_perf / post_process_data look at
    events = list(prof.events())
    cuda_events = [e for e in events if "CUDA" in str(getattr(e, "device_type", ""))]
    dev_time = sum(getattr(e, "self_device_time_total", 0) or 0 for e in cuda_events)
    return (
        f"[{tag}] total_events={len(events)} cuda_events={len(cuda_events)} "
        f"self_device_time_total={dev_time:.1f}us"
    )


def _child(_):
    return count_cuda_rows("spawned worker")


if __name__ == "__main__":
    print(count_cuda_rows("main process"), flush=True)
    mp.set_start_method("spawn", force=True)
    with mp.Pool(1) as pool:
        print(pool.map(_child, [0])[0], flush=True)
