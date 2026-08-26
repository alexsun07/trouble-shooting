#!/usr/bin/env python3
"""Characterise exactly when the "profiler sees no GPU kernels" bug triggers.

Established: HIP init makes rocprofiler C-setenv ROCPROFILER_REGISTER_LIBRARY,
and a child inheriting it is blind. That is one instance of a more general rule;
this script pins the rule down.

  T1 trigger    is "var present at process start" sufficient on its own, with no
                parent involved at all?
  T2 when       which torch call actually injects it (import? is_available?
                device_count? set_device? first kernel?), observed through the C
                environ via ctypes.getenv -- os.environ cannot see setenv()
  T3 fork       are fork children affected too, or only exec'd ones?
  T4 depth      does an intermediate non-Python process (bash) still propagate it?
  T5 value      does the path have to be a real library, or is mere presence enough?
  T6 visible    can the CHILD see the var in os.environ (it inherits at exec),
                even though the PARENT cannot?
"""
import ctypes
import os
import subprocess
import sys

SELF = os.path.abspath(__file__)
VAR = "ROCPROFILER_REGISTER_LIBRARY"
_libc = ctypes.CDLL("libc.so.6")
_libc.getenv.restype = ctypes.c_char_p


def c_getenv(name=VAR):
    """Read the REAL process environment, including C-level setenv() writes."""
    v = _libc.getenv(name.encode())
    return v.decode() if v else None


def py_getenv(name=VAR):
    return os.environ.get(name)


def profile_here(dev=0):
    import torch
    import torch.profiler as tpf

    torch.cuda.set_device(dev)
    a = torch.randn(4096, 4096, device=f"cuda:{dev}", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device=f"cuda:{dev}", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    with tpf.profile(activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA]) as p:
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
    cu = [e for e in p.events() if "CUDA" in str(getattr(e, "device_type", ""))]
    return len(cu)


def verdict(n):
    return f"cuda_events={n} -> {'SEES' if n else 'BLIND'}"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "profile"

    if mode == "profile":
        # Report what this process inherited, then whether it can profile.
        print(
            f"  [child] inherited {VAR}="
            f"{'<set>' if c_getenv() else '<unset>'} "
            f"| visible in os.environ: {py_getenv() is not None} "
            f"| {verdict(profile_here())}"
        )

    elif mode == "T2":
        # Watch the C environ across torch initialisation stages.
        def show(stage):
            c, p = c_getenv(), py_getenv()
            print(
                f"  {stage:<34} C-environ={'SET' if c else 'unset':<5} "
                f"os.environ={'SET' if p else 'unset'}"
            )

        show("process start")
        import torch

        show("after import torch")
        torch.cuda.is_available()
        show("after cuda.is_available()")
        torch.cuda.device_count()
        show("after cuda.device_count()")
        torch.cuda.set_device(0)
        show("after cuda.set_device(0)")
        x = torch.randn(64, 64, device="cuda")
        show("after tensor alloc on GPU")
        (x @ x).sum().item()
        torch.cuda.synchronize()
        show("after first kernel + sync")

    elif mode == "T3":
        # fork children share the parent's already-initialised HIP state.
        import torch

        torch.cuda.set_device(0)
        (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"))
        torch.cuda.synchronize()
        print(f"  parent initialised HIP; C-environ {VAR}="
              f"{'SET' if c_getenv() else 'unset'}")
        pid = os.fork()
        if pid == 0:
            try:
                # NOTE: profiling after fork with a live HIP context is not
                # generally supported; a crash here is itself a finding.
                print(f"  [fork child] {verdict(profile_here())}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [fork child] FAILED: {type(exc).__name__}: {exc}")
            finally:
                os._exit(0)
        os.waitpid(pid, 0)

    elif mode == "T4":
        import torch

        torch.cuda.set_device(0)
        (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"))
        torch.cuda.synchronize()
        print("  parent initialised HIP; launching grandchild via bash")
        subprocess.run(["bash", "-c", f"{sys.executable} {SELF} profile"], check=False)

    elif mode == "T7":
        # The realistic launcher pattern: the parent only ASKS whether a GPU
        # exists -- never allocates, never runs a kernel -- then spawns workers.
        import torch

        print(f"  before: C-environ={'SET' if c_getenv() else 'unset'}")
        torch.cuda.is_available()
        print(f"  after is_available(): C-environ={'SET' if c_getenv() else 'unset'}")
        print("  spawning child (parent never ran a kernel):")
        subprocess.run([sys.executable, SELF, "profile"], check=False)

    elif mode == "T8":
        # Is the process that SET the var affected itself?
        import torch

        torch.cuda.is_available()
        print(f"  self set the var (C-environ={'SET' if c_getenv() else 'unset'}); "
              f"own profiler: {verdict(profile_here())}")

    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
