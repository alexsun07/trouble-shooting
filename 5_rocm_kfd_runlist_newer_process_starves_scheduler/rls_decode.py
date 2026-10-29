"""Decode KFD's HWS runlist (sudo cat /sys/kernel/debug/kfd/rls) into per-GPU counts:
processes (MAP_PROCESS), compute / SDMA queues (MAP_QUEUES engine_sel), and whether the runlist
ends in a chained RUN_LIST -- KFD appends that only when the runlist is oversubscribed
(kfd_packet_manager.c pm_create_runlist_ib). Node numbering follows /sys/class/kfd topology:
the first GPU node is GPU 0.   sudo cat /sys/kernel/debug/kfd/rls | python3 rls_decode.py [-v: also each process's queues in runlist order]
"""
import re, sys

nodes, cur = {}, None
for line in sys.stdin:
    m = re.match(r"Node (\d+), gpu_id (\w+):", line)
    if m:
        cur = int(m.group(1)); nodes[cur] = []
        continue
    m = re.match(r"\s*[0-9a-f]{8}: (.*)", line)
    if m and cur is not None:
        nodes[cur] += [int(w, 16) for w in m.group(1).split()]
first = min(nodes) if nodes else 0
for n, dw in sorted(nodes.items()):
    i, procs, cq, sq, chain, quanta, order = 0, 0, 0, 0, False, set(), []
    while i < len(dw):
        h = dw[i]
        if h >> 30 != 3:
            i += 1; continue
        count, op = (h >> 16) & 0x3FFF, (h >> 8) & 0xFF
        if op == 0xA1:
            procs += 1; quanta.add(dw[i + 1] >> 25); order.append([0, 0])
        elif op == 0xA2:
            eng = (dw[i + 1] >> 26) & 7
            if eng == 0: cq += 1; order[-1][0] += 1
            else: sq += 1; order[-1][1] += 1
        elif op == 0xA5:
            chain = True
        i += count + 2
    print(f"GPU{n - first} node{n}: procs={procs} compute_q={cq} sdma_q={sq} chained_runlist(oversubscribed)={chain} process_quantum={sorted(quanta)}"
          + (f" order(compute+sdma per process, runlist order)={' '.join(f'{c}+{d}' for c, d in order)}" if "-v" in sys.argv else ""))
