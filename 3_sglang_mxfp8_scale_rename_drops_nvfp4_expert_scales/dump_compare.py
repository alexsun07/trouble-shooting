"""Compare two sglang debug tensor dumps (rank 0, first pass with >1 token).

  python3 dump_compare.py <dump_dir_a> <dump_dir_b>
Prints, in A's order, per module: shapes, absmax and relative L2 difference.
"""
import glob
import sys

import torch


def load(d):
    for f in sorted(glob.glob(f"{d}/TP0_*/Pass*.pt")):
        t = torch.load(f, map_location="cpu", weights_only=False)
        v = t.get("model.embed_tokens")
        if torch.is_tensor(v) and v.shape[0] > 1:
            return f, t
    raise SystemExit(f"no multi-token pass in {d}")


fa, a = load(sys.argv[1])
fb, b = load(sys.argv[2])
print("A", fa, "\nB", fb)
for k, va in a.items():
    vb = b.get(k)
    if not (torch.is_tensor(va) and va.is_floating_point()):
        continue
    if not (torch.is_tensor(vb) and vb.is_floating_point()):
        print(f"{k:55s} A{tuple(va.shape)} only in A")
        continue
    xa, xb = va.float(), vb.float()
    if xa.shape != xb.shape:
        print(f"{k:55s} shape A{tuple(xa.shape)} B{tuple(xb.shape)}")
        continue
    rel = ((xa - xb).norm() / xb.norm().clamp(min=1e-12)).item()
    flag = "  <<<" if rel > 0.05 else ""
    print(f"{k:55s} {str(tuple(xa.shape)):14s} maxA={xa.abs().max():9.3g} maxB={xb.abs().max():9.3g} rel={rel:.4f}{flag}")
for k in b:
    if k not in a and torch.is_tensor(b[k]):
        print(f"{k:55s} B{tuple(b[k].shape)} only in B")
