#!/usr/bin/env python3
"""Why can't the torch profiler see GPU kernels here?

Established so far:
  - main process alone            -> sees kernels
  - spawned child, parent idle    -> sees kernels
  - spawned child, parent did one matmul (no profiler) -> sees NOTHING
  - under rocprofv3, even the main process sees NOTHING

The last point says the ROCm tracing backend is single-consumer. This script
separates the remaining hypotheses:

  H1 env inheritance  -- the parent's HIP init mutates os.environ, and the
                         spawned child inherits something that breaks it
  H2 parent/child     -- specific to being a descendant of a HIP-init'd process
  H3 concurrent HIP   -- ANY other live process holding a HIP context on the
                         node blocks tracing, regardless of relationship

Modes:
  envdiff   print env vars added/changed by torch.cuda init
  profile [dev]      profile a matmul on `dev`, report CUDA event count
  holder  [dev] [s]  hold a HIP context on `dev` for `s` seconds (no profiling)
  cleanenv           parent inits HIP, then runs `profile` in a subprocess
                     twice: inherited env vs sanitized env
"""
import copy
import os
import subprocess
import sys

SELF = os.path.abspath(__file__)


def _profile(dev=0):
    import torch
    import torch.profiler as tpf

    torch.cuda.set_device(dev)
    a = torch.randn(4096, 4096, device=f"cuda:{dev}", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device=f"cuda:{dev}", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    with tpf.profile(
        activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
    ev = list(prof.events())
    cu = [e for e in ev if "CUDA" in str(getattr(e, "device_type", ""))]
    dt = sum(getattr(e, "self_device_time_total", 0) or 0 for e in cu)
    return len(ev), len(cu), dt


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "profile"

    if mode == "envdiff":
        before = copy.deepcopy(dict(os.environ))
        import torch

        torch.cuda.set_device(0)
        (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"))
        torch.cuda.synchronize()
        after = dict(os.environ)
        added = {k: v for k, v in after.items() if k not in before}
        changed = {
            k: (before[k], v) for k, v in after.items() if k in before and before[k] != v
        }
        print(f"env added by HIP init  : {added or '(none)'}")
        print(f"env changed by HIP init: {changed or '(none)'}")

    elif mode == "profile":
        dev = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        tot, cu, dt = _profile(dev)
        verdict = "SEES KERNELS" if cu else "BLIND"
        print(f"[profile dev={dev}] total={tot} cuda={cu} dev_time={dt:.1f}us -> {verdict}")

    elif mode == "holder":
        dev = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        secs = float(sys.argv[3]) if len(sys.argv) > 3 else 60
        import time

        import torch

        torch.cuda.set_device(dev)
        x = torch.randn(1024, 1024, device=f"cuda:{dev}")
        (x @ x).sum().item()
        torch.cuda.synchronize()
        print(f"[holder] HIP context live on dev={dev} for {secs}s", flush=True)
        time.sleep(secs)

    elif mode == "cleanenv":
        import torch

        torch.cuda.set_device(0)
        (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"))
        torch.cuda.synchronize()
        print("parent initialised HIP; now running child twice", flush=True)

        print("-- child with INHERITED env --", flush=True)
        subprocess.run([sys.executable, SELF, "profile", "0"], check=False)

        # strip everything ROCm/HIP/HSA related, keep only the bare minimum
        clean = {
            k: v
            for k, v in os.environ.items()
            if not any(
                k.startswith(p)
                for p in ("HIP", "HSA", "ROC", "AMD", "GPU_", "PYTORCH_", "TORCH")
            )
        }
        clean.setdefault("PATH", os.environ.get("PATH", ""))
        print("-- child with SANITIZED env --", flush=True)
        subprocess.run([sys.executable, SELF, "profile", "0"], env=clean, check=False)

    elif mode == "bisect":
        # cleanenv proved the breakage is env-mediated even though HIP init
        # changes no env var -- so the culprit is a var that was already in the
        # container env. Drop them one at a time and see which one restores sight.
        import torch

        cands = sorted(
            k
            for k in os.environ
            if any(
                k.startswith(p)
                for p in ("HIP", "HSA", "ROC", "AMD", "GPU_", "PYTORCH_", "TORCH")
            )
        )
        print(f"candidates ({len(cands)}): {cands}\n", flush=True)

        torch.cuda.set_device(0)
        (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"))
        torch.cuda.synchronize()
        print("parent initialised HIP\n", flush=True)

        for k in cands:
            env = {kk: vv for kk, vv in os.environ.items() if kk != k}
            r = subprocess.run(
                [sys.executable, SELF, "profile", "0"],
                env=env,
                capture_output=True,
                text=True,
            )
            line = next(
                (l for l in r.stdout.splitlines() if l.startswith("[profile")), "?"
            )
            verdict = "SEES" if "SEES KERNELS" in line else "blind"
            print(f"  without {k:<28} = {os.environ[k][:28]:<30} -> {verdict}", flush=True)

    elif mode == "dumpenv":
        import json

        print("@@ENV@@" + json.dumps(dict(os.environ)))

    elif mode == "envleak":
        # The bisect showed ANY explicit env= fixes the child, so the content of
        # os.environ is not the issue -- passing env= at all is. That points at a
        # C-level setenv() done by HIP init: invisible to Python's os.environ
        # (hence envdiff saw nothing), but inherited by an env=None child via execv.
        # Dump what a child ACTUALLY receives, before and after HIP init.
        import json

        def child_env():
            r = subprocess.run(
                [sys.executable, SELF, "dumpenv"], capture_output=True, text=True
            )
            line = next(
                l for l in r.stdout.splitlines() if l.startswith("@@ENV@@")
            )
            return json.loads(line[len("@@ENV@@"):])

        before = child_env()
        import torch

        torch.cuda.set_device(0)
        (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"))
        torch.cuda.synchronize()
        after = child_env()

        added = {k: v for k, v in after.items() if k not in before}
        changed = {
            k: (before[k], v) for k, v in after.items() if k in before and before[k] != v
        }
        removed = [k for k in before if k not in after]
        print(f"child env ADDED after parent HIP init  : {added or '(none)'}")
        print(f"child env CHANGED after parent HIP init: {changed or '(none)'}")
        print(f"child env REMOVED after parent HIP init: {removed or '(none)'}")
        print(f"\npython os.environ sees these? "
              f"{ {k: os.environ.get(k, '<ABSENT from os.environ>') for k in added} }")

    elif mode == "pinpoint":
        # envleak found 6 vars that HIP init sets via C setenv() and that an
        # env=None child inherits. Which one actually blinds the child? Start
        # from the FULL leaked env (reproduces the failure) and drop one at a time,
        # so the only difference from the broken case is that single var.
        import json

        def child_env():
            r = subprocess.run(
                [sys.executable, SELF, "dumpenv"], capture_output=True, text=True
            )
            return json.loads(
                next(l for l in r.stdout.splitlines() if l.startswith("@@ENV@@"))[
                    len("@@ENV@@"):
                ]
            )

        before = child_env()
        import torch

        torch.cuda.set_device(0)
        (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"))
        torch.cuda.synchronize()
        after = child_env()
        leaked = [k for k in after if k not in before]

        def run(env, label):
            r = subprocess.run(
                [sys.executable, SELF, "profile", "0"],
                env=env,
                capture_output=True,
                text=True,
            )
            line = next(
                (l for l in r.stdout.splitlines() if l.startswith("[profile")), "?"
            )
            print(f"  {label:<52} -> {'SEES' if 'SEES KERNELS' in line else 'blind'}",
                  flush=True)

        print(f"leaked vars: {leaked}\n", flush=True)
        run(after, "full leaked env (expect: blind = reproduces)")
        for k in leaked:
            run({kk: vv for kk, vv in after.items() if kk != k}, f"leaked env minus {k}")

    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
