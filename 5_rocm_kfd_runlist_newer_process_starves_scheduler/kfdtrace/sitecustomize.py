# Pairs with kfdtrace.so: install faulthandler on $KFDTRACE_SIG so the shim's raise() dumps every
# thread's Python stack into $KFDTRACE_DIR/trace.<pid>.txt. Re-registers after fork.
import os
if os.environ.get("KFDTRACE_DIR"):
    import faulthandler, signal, sys

    def _arm():
        sig = int(os.environ.get("KFDTRACE_SIG", "40"))
        f = open(os.path.join(os.environ["KFDTRACE_DIR"], f"trace.{os.getpid()}.txt"), "a")
        f.write(f"=== armed pid={os.getpid()} argv={' '.join(sys.argv)[:200]}\n"); f.flush()
        faulthandler.register(sig, file=f, all_threads=True, chain=False)
        globals()["_f"] = f

    _arm()
    os.register_at_fork(after_in_child=_arm)
