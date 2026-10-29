"""Hold idle KFD compute queues on one GPU, launching no work after setup.

Each torch stream gets its own HIP hardware queue while GPU_MAX_HW_QUEUES allows it, so
`python3 queue_hog.py N` (with GPU_MAX_HW_QUEUES > N) ends up with N + 1 compute queues on MI355X /
ROCm 7.2 (N streams plus one more the runtime opens) -- check with kfd_queues.sh. Exits when its parent dies.
`python3 queue_hog.py N <port>` waits until the server on <port> answers /health (plus 60 s, past the
warmup) before touching the GPU, so its queues are created after the server's.
"""
import os, sys, time, urllib.request
import torch

n = int(sys.argv[1])
ppid = os.getppid()
if len(sys.argv) > 2:
    while True:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[2]}/health", timeout=5).status == 200:
                break
        except Exception:
            pass
        if os.getppid() != ppid:
            sys.exit(0)
        time.sleep(5)
    time.sleep(60)
x = torch.zeros(1, device="cuda")
streams = [torch.cuda.Stream() for _ in range(n - 1)]
for s in streams:
    with torch.cuda.stream(s):
        x.add_(1)
torch.cuda.synchronize()
print(f"queue_hog pid={os.getpid()} streams={n} ready", flush=True)
while os.getppid() == ppid:
    time.sleep(2)
