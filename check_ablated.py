#!/usr/bin/env python3
"""Decode every edited tensor from an abliterated .hgn and check for
NaN/Inf, plus verify the ablation actually zeroed the direction projection."""
import sys
import numpy as np

sys.path.insert(0, "/home/user/hgn-abliterate")
from hgn_abliterate import (HgnFile, make_codec, ht_sides, select_targets,
                            load_direction, RESIDUAL, HYPER)

BASE = "/home/user/halogen/official/qwen38-flash-next-ht43.hgn"
ABL = "/home/user/halogen/abliterated/ht43-abliterated.hgn"
DIR = "/tmp/dir-embed.f32"

r2560 = load_direction(DIR)
abl = HgnFile(ABL)
targets = select_targets(abl, None, None, None, verbose=False)
print(f"targets: {len(targets)}", flush=True)

# 1) side planes sanity (f16 can carry NaN/Inf too)
for e in abl.entries:
    if e.name.endswith((".suh", ".svh")):
        v = np.frombuffer(bytes(abl.payload(e)), "<f2").astype(np.float32)
        if not np.all(np.isfinite(v)):
            print(f"BAD SIDE PLANE {e.name}: "
                  f"{int((~np.isfinite(v)).sum())} non-finite", flush=True)
        if np.any(v == 0):
            print(f"WARN {e.name}: {int((v == 0).sum())} zero scales", flush=True)
print("side planes checked", flush=True)

# 2) decode each edited tensor, check finite + projection strength
nbad = 0
for i, e in enumerate(targets):
    codec = make_codec(e.store, e.variant, abl.payload(e), e.N, e.K,
                       *(ht_sides(abl, e) if e.store == 16 else (None, None)))
    r = r2560 if e.K == RESIDUAL else np.tile(r2560, 4)
    r = r / np.linalg.norm(r)
    nonfinite = 0
    proj_abs = []
    row_abs = []
    for a in range(0, e.N, 4096):
        b = min(a + 4096, e.N)
        w = codec.decode_rows(a, b)
        nonfinite += int((~np.isfinite(w)).sum())
        proj_abs.append(np.abs(w @ r))
        row_abs.append(np.linalg.norm(w, axis=1))
    proj = np.concatenate(proj_abs)
    rows = np.concatenate(row_abs)
    # fraction of row energy still along the direction
    frac = float(np.median(proj / np.maximum(rows, 1e-12)))
    status = "BAD" if nonfinite else "ok "
    if nonfinite:
        nbad += 1
    print(f"{status} [{i+1}/{len(targets)}] {e.name} "
          f"store={e.store} nonfinite={nonfinite} "
          f"median|proj|/|row|={frac:.4f}", flush=True)

print(f"done: {nbad} tensors with non-finite values", flush=True)
