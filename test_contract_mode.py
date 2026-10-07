#!/usr/bin/env python3
"""Self-test for contract (writer-side) ablation and derive-direction.

Builds a synthetic checkpoint containing the three contract writer shapes
(2-D rows-mode in bf16 and HT, fused-experts mode, embed_tokens last-mode)
plus a reader tensor that must stay untouched, then checks:
  1. contract ablation collapses the OUTPUT-axis projection along r
  2. non-writer tensors are byte-identical
  3. derive-direction recovers r from the (base, ablated) pair
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hgn_abliterate as ha
from test_hgn_abliterate import build_file, q4c_payload

RNG = np.random.default_rng(7)
D = 2560
CB = np.array([-1, -0.69619, -0.52507, -0.39492, -0.28444, -0.18477,
               -0.09105, 0, 0.07958, 0.16093, 0.24611, 0.33792,
               0.44071, 0.56262, 0.72296, 1], np.float32)


def main():
    tmp = tempfile.mkdtemp(prefix="hgn-contract-")
    base = os.path.join(tmp, "base.hgn")
    out = os.path.join(tmp, "abl.hgn")
    dirp = os.path.join(tmp, "dir.f32")

    r = RNG.standard_normal(D).astype(np.float32)
    r /= np.linalg.norm(r)
    r.tofile(dirp)

    def writer(n, k):
        w = RNG.standard_normal((n, k)).astype(np.float32) * 0.05
        v = RNG.standard_normal(k).astype(np.float32)
        w += 3.0 * r[:, None] * v[None, :]  # strong output-axis component
        return w

    sed = writer(2560, 640)                       # rows mode, bf16
    outp = writer(2560, 640)                      # rows mode, HT
    gup = RNG.standard_normal((2, 2560, 640)).astype(np.float32) * 0.05
    gup += 3.0 * r[None, :, None] * RNG.standard_normal(
        (2, 640)).astype(np.float32)[:, None, :]  # experts mode
    embed = RNG.standard_normal((128, D)).astype(np.float32) * 0.05
    embed += 3.0 * RNG.standard_normal(128).astype(np.float32)[:, None] * r
    qproj = RNG.standard_normal((64, D)).astype(np.float32) * 0.05

    su = np.where(RNG.random(640) < 0.5, -1.0, 1.0).astype(np.float32)
    sv = (np.linalg.norm(outp, axis=1) * 0.02).astype(np.float32)
    import hgnht
    ht_blob = hgnht.encode(outp, su, sv, workers=4, beam=128)

    # 3-bit fused experts (variant 0x1206) with per-expert side planes
    sed3 = writer(2560, 640)
    sed3_3d = np.stack([sed3, writer(2560, 640)])  # [2, 2560, 640]
    su3 = np.stack([np.where(RNG.random(640) < 0.5, -1.0, 1.0).astype(np.float32)
                    for _ in range(2)])            # [2, 640]
    sv3 = (np.linalg.norm(sed3_3d, axis=2) * 0.02).astype(np.float32)  # [2, 2560]
    ht3_blob = b"".join(
        hgnht.encode(sed3_3d[e], su3[e], sv3[e], workers=4, beam=128, bits=3)
        for e in range(2))

    tensors = [
        ("embed_tokens.weight", 5, 2, [128, D], q4c_payload(embed, CB)),
        ("layers.0.mlp.shared_expert.down_proj.weight", 0, 0, [2560, 640],
         ha.bf16_encode(sed).tobytes()),
        ("layers.0.mlp.experts.down_proj.weight", 0, 0, [2, 2560, 640],
         ha.bf16_encode(gup.reshape(5120, 640)).tobytes()),
        ("layers.0.linear_attn.out_proj.weight", 16, 0x1208, [2560, 640],
         ht_blob),
        ("layers.0.linear_attn.out_proj.suh", 2, 0, [640],
         su.astype(np.float16).tobytes()),
        ("layers.0.linear_attn.out_proj.svh", 2, 0, [2560],
         sv.astype(np.float16).tobytes()),
        ("layers.1.mlp.experts.down_proj.weight", 16, 0x1206, [2, 2560, 640],
         ht3_blob),
        ("layers.1.mlp.experts.down_proj.suh", 2, 0, [2, 640],
         su3.astype(np.float16).tobytes()),
        ("layers.1.mlp.experts.down_proj.svh", 2, 0, [2, 2560],
         sv3.astype(np.float16).tobytes()),
        ("layers.0.self_attn.q_proj.weight", 0, 0, [64, D],
         ha.bf16_encode(qproj).tobytes()),
    ]
    build_file(base, tensors, "contract-test")

    rc = subprocess.run(
        [sys.executable, os.path.join(HERE, "hgn_abliterate.py"),
         "abliterate", base, "-o", out, "--direction", dirp,
         "--writers", "contract"],
        capture_output=True, text=True, cwd=HERE)
    print(rc.stdout + rc.stderr)
    assert rc.returncode == 0, "contract abliterate failed"
    assert "targets: 5 tensors" in rc.stdout, "expected 5 contract targets"

    h = ha.HgnFile(out)
    hb = ha.HgnFile(base)
    for e in h.entries:
        assert ha.xor_fold_view(h.payload(e)) == e.checksum, e.name
        assert ha.size_rule(e.store, e.variant, e.N, e.K) == e.size, e.name
    print("PASS container: checksums + size rules")

    for name in ("layers.0.self_attn.q_proj.weight",
                 "layers.0.linear_attn.out_proj.svh"):
        a = bytes(hb.payload(hb.by_name[name]))
        b = bytes(h.payload(h.by_name[name]))
        assert a == b, f"untouched tensor changed: {name}"
    print("PASS non-writer tensors byte-identical")

    def dec(hgn, name):
        e = hgn.by_name[name]
        sides = ha.ht_sides(hgn, e) if e.store == 16 else (None, None)
        c = ha.make_codec(e.store, e.variant, hgn.payload(e), e.N, e.K, *sides)
        return e, c.decode_rows(0, e.N)

    def out_proj_ratio(hgn, name):
        e, w = dec(hgn, name)
        if e.rank == 3:
            W = w.reshape(-1, D, e.K)
            proj = np.einsum("o,eok->ek", r, W)
            colnorm = np.linalg.norm(W, axis=1)
        else:
            proj = r @ w
            colnorm = np.linalg.norm(w, axis=0)
        keep = colnorm > 0
        return np.abs(proj[keep]) / colnorm[keep]

    def row_proj_ratio(hgn, name):
        e, w = dec(hgn, name)
        nrm = np.linalg.norm(w, axis=1)
        keep = nrm > 0
        return np.abs(w[keep] @ r) / nrm[keep]

    for name in ("layers.0.mlp.shared_expert.down_proj.weight",
                 "layers.0.mlp.experts.down_proj.weight",
                 "layers.0.linear_attn.out_proj.weight",
                 "layers.1.mlp.experts.down_proj.weight"):
        p0 = out_proj_ratio(hb, name)
        p1 = out_proj_ratio(h, name)
        assert p0.mean() > 0.1, f"{name}: test signal missing in base"
        assert p1.mean() < 0.1 * p0.mean() and p1.max() < 0.3 * p0.mean(), \
            f"{name}: output-axis residual mean {p1.mean():.4f}/max " \
            f"{p1.max():.4f} vs base mean {p0.mean():.4f}"
    p0 = row_proj_ratio(hb, "embed_tokens.weight")
    p1 = row_proj_ratio(h, "embed_tokens.weight")
    assert p0.mean() > 0.1 and p1.mean() < 0.1 * p0.mean(), \
        f"embed_tokens: residual {p1.mean():.4f} vs base {p0.mean():.4f}"
    print("PASS output-axis refusal component collapsed")

    ext = os.path.join(tmp, "derived.f32")
    rc = subprocess.run(
        [sys.executable, os.path.join(HERE, "hgn_abliterate.py"),
         "derive-direction", "--base", base, "--donor", out, "--out", ext],
        capture_output=True, text=True, cwd=HERE)
    print(rc.stdout + rc.stderr)
    assert rc.returncode == 0, "derive-direction failed"
    r2 = np.fromfile(ext, np.float32)
    cos = abs(float(r2 @ r))
    assert cos > 0.99, f"derived direction cos={cos:.4f}"
    print(f"PASS derive-direction round-trip: |cos| = {cos:.5f}")

    hb.close()
    h.close()
    print("\nCONTRACT TESTS PASSED")


if __name__ == "__main__":
    main()
