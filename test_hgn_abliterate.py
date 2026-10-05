#!/usr/bin/env python3
"""Self-test for hgn_abliterate.py using a synthetic HGN file.

Builds a small checkpoint with one tensor per editable store plus pass-through
tensors, abliterates it with a known direction, and checks:
  1. output container parses, size rules and checksums hold
  2. pass-through tensors are byte-identical
  3. abliterated rows have ~zero component along the direction
  4. extract() recovers the direction from a (base, abliterated) pair
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hgn_abliterate as ha

RNG = np.random.default_rng(42)
D = 2560


def q4c_payload(w, cb):
    """Encode float matrix w (N,K) to q4c v2 bytes with codebook cb."""
    N, K = w.shape
    ng = K // 32
    g = w.reshape(N, ng, 32)
    mx = np.abs(g).max(axis=2)
    s = mx / float(np.abs(cb).max())
    safe = np.where(s > 0, s, 1.0)
    codes = ha.quantize_to_codebook((g / safe[:, :, None]).astype(np.float32),
                                    cb, True).reshape(N, K)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    S = 64 + ha.pad(N * K // 2, 64)
    Rr = ha.pad(K // 16, 16)
    out = bytearray(64 + ha.pad(N * K // 2, 64) + N * Rr)
    out[:64] = cb.astype(np.float32).tobytes()
    out[64:64 + N * K // 2] = packed.reshape(-1).tobytes()
    row = np.zeros((N, Rr), np.uint8)
    row[:, : ng * 2] = s.astype(np.float16).view(np.uint8).reshape(N, ng * 2)
    out[S:S + N * Rr] = row.reshape(-1).tobytes()
    return bytes(out)


def build_file(path, tensors, identity):
    """tensors: list of (name, store, variant, shape, payload bytes)."""
    n = len(tensors)
    ps = ha.pad(ha.HEADER_SIZE + ha.ENTRY_SIZE * n, 64)
    with open(path, "wb") as f:
        f.truncate(ps)
        f.seek(ps)
        offs, cks = [], []
        off = ps
        for name, store, variant, shape, blob in tensors:
            offs.append(off)
            f.write(blob)
            cks.append(ha.xor_fold_bytes(blob))
            off += ha.pad(len(blob), 64)
        f.seek(0)
        f.write(struct.pack("<IIQQQQ", ha.MAGIC, 2, n, ha.HEADER_SIZE,
                            ps, off))
        f.write(identity.encode().ljust(64, b"\x00"))
        for i, (name, store, variant, shape, blob) in enumerate(tensors):
            rec = bytearray(ha.ENTRY_SIZE)
            rec[0:96] = name.encode().ljust(96, b"\x00")
            struct.pack_into("<II", rec, 0x60, store, len(shape))
            struct.pack_into("<4q", rec, 0x68,
                             *(list(shape) + [0] * (4 - len(shape))))
            struct.pack_into("<QQ", rec, 0x88, offs[i], len(blob))
            struct.pack_into("<II", rec, 0x98, cks[i], variant)
            f.write(rec)
        f.truncate(off)


def main():
    tmp = tempfile.mkdtemp(prefix="hgn-test-")
    base = os.path.join(tmp, "base.hgn")
    out = os.path.join(tmp, "abl.hgn")
    dirp = os.path.join(tmp, "dir.f32")

    # known direction
    r = RNG.standard_normal(D).astype(np.float32)
    r /= np.linalg.norm(r)
    r.tofile(dirp)

    # tensors (with a strong refusal component along r so the test measures
    # removal against a real signal, not the requant noise floor)
    # NF4-like codebook: real checkpoints use learned codebooks of comparable
    # quality; a random codebook would only measure requant noise.
    cb = np.array([-1, -0.69619, -0.52507, -0.39492, -0.28444, -0.18477,
                   -0.09105, 0, 0.07958, 0.16093, 0.24611, 0.33792,
                   0.44071, 0.56262, 0.72296, 1], np.float32)

    def with_refusal(n, scale, k=1):
        w = (RNG.standard_normal((n, D * k)) * scale).astype(np.float32)
        rr = np.tile(r, k)
        w += (RNG.standard_normal(n).astype(np.float32) * scale * 10)[:, None] * rr[None, :]
        return w

    lm = with_refusal(64, 0.05)
    inproj = with_refusal(256, 0.05)
    gate = with_refusal(8, 0.05)
    mixdown = with_refusal(32, 0.05, k=4)
    qproj = with_refusal(64, 0.05)
    gup = with_refusal(128, 0.05)
    sgate = with_refusal(32, 0.05)
    sup = with_refusal(32, 0.05)
    embed = RNG.standard_normal((128, D)).astype(np.float32) * 0.05
    down = RNG.standard_normal((64, 640)).astype(np.float32) * 0.05
    kproj = with_refusal(128, 0.05)  # store 16 (HT) target

    # HT side planes: su = +/-1 signs, sv = signed row scales (real
    # checkpoints carry |sv| ~ 0.02 * row norms)
    su_ht = np.where(RNG.random(D) < 0.5, -1.0, 1.0).astype(np.float32)
    sv_ht = (np.linalg.norm(kproj, axis=1) * 0.02).astype(np.float32)
    sv_ht[0] = 0.0  # exercise the zero-scale path
    kproj[0] = 0.0
    import hgnht
    ht_blob = hgnht.encode(kproj, su_ht, sv_ht, workers=4, beam=128)

    def enc(store, variant, bits, w2d):
        n_, k_ = w2d.shape
        size = ha.size_rule(store, variant, n_, k_)
        if store == 12:
            c = ha.Q6kCodec(memoryview(bytearray(size)), n_, k_)
        else:
            c = ha.AffineCodec(memoryview(bytearray(size)), n_, k_, bits,
                               variant)
        c.encode_rows(0, n_, w2d)
        return c.finalize()

    # q8g64 payload for mixdown
    N, K = mixdown.shape
    ng = K // 64
    g = mixdown.reshape(N, ng, 64)
    mn, mx = g.min(axis=2), g.max(axis=2)
    s = (mx - mn) / np.float32(255.0)
    safe = np.where(s > 0, s, 1.0)
    code = np.clip(np.rint((g - mn[:, :, None]) / safe[:, :, None]),
                   0, 255).astype(np.uint8)
    row = np.zeros((N, K + K // 16), np.uint8)
    row[:, :K] = code.reshape(N, K)
    rec = np.stack([s.astype(np.float16), mn.astype(np.float16)], axis=2)
    row[:, K:] = rec.view(np.uint8).reshape(N, ng * 4)
    mix_blob = row.reshape(-1).tobytes()

    # q8s32 payload for down
    N2, K2 = down.shape
    ng2 = K2 // 32
    g2 = down.reshape(N2, ng2, 32)
    s2 = np.abs(g2).max(axis=2) / np.float32(127.0)
    safe2 = np.where(s2 > 0, s2, 1.0)
    c2 = np.clip(np.rint(g2 / safe2[:, :, None]), -127, 127).astype(np.int8)
    Rr2 = ha.pad(K2 + K2 // 16, 16)
    row2 = np.zeros((N2, Rr2), np.uint8)
    row2[:, :K2] = c2.view(np.uint8).reshape(N2, K2)
    row2[:, K2:K2 + ng2 * 2] = s2.astype(np.float16).view(np.uint8
                                                          ).reshape(N2, ng2 * 2)
    down_blob = row2.reshape(-1).tobytes()

    tensors = [
        ("lm_head.weight", 5, 2, [64, D], q4c_payload(lm, cb)),
        ("embed_tokens.weight", 5, 2, [128, D], q4c_payload(embed, cb)),
        ("layers.0.linear_attn.in_proj_qkv.weight", 5, 2, [256, D],
         q4c_payload(inproj, cb)),
        ("layers.0.mlp.gate.weight", 0, 0, [8, D],
         ha.bf16_encode(gate).tobytes()),
        ("layers.0.attn_hyper_connection.input_mix_weight_down.weight",
         7, 0, [32, 4 * D], mix_blob),
        ("layers.0.mlp.experts.down_proj.weight", 11, 0, [64, 640], down_blob),
        ("layers.0.self_attn.q_proj.weight", 12, 0, [64, D],
         enc(12, 0, None, qproj)),
        ("layers.0.mlp.experts.gate_up_proj.weight", 14, 1, [4, 32, D],
         enc(14, 1, 4, gup)),
        ("layers.0.mlp.shared_expert.gate_proj.weight", 15, 0, [32, D],
         enc(15, 0, 5, sgate)),
        ("layers.0.mlp.shared_expert.up_proj.weight", 15, 1, [32, D],
         enc(15, 1, 5, sup)),
        ("layers.0.self_attn.k_proj.weight", 16, 0x1208, [128, D], ht_blob),
        ("layers.0.self_attn.k_proj.suh", 2, 0, [D],
         su_ht.astype(np.float16).tobytes()),
        ("layers.0.self_attn.k_proj.svh", 2, 0, [128],
         sv_ht.astype(np.float16).tobytes()),
        ("layers.1.ple.ple_embedding.layer_multipliers", 4, 0, [3],
         np.arange(3, dtype=np.int64).tobytes()),
    ]
    build_file(base, tensors, "ht43-test")

    # 1. abliterate
    rc = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "hgn_abliterate.py"), "abliterate",
         base, "-o", out, "--direction", dirp],
        capture_output=True, text=True)
    print(rc.stdout + rc.stderr)
    assert rc.returncode == 0, "abliterate failed"

    # 2. output validates: parse + checksums + size rules
    h = ha.HgnFile(out)
    assert h.identity == "ht43-test"
    for e in h.entries:
        assert ha.xor_fold_view(h.payload(e)) == e.checksum, e.name
        assert ha.size_rule(e.store, e.variant, e.N, e.K) == e.size, e.name
    print("PASS container: checksums + size rules")

    # 3. pass-through byte-identical
    hb = ha.HgnFile(base)
    for name in ("embed_tokens.weight",
                 "layers.0.mlp.experts.down_proj.weight",
                 "layers.1.ple.ple_embedding.layer_multipliers"):
        a = bytes(hb.payload(hb.by_name[name]))
        b = bytes(h.payload(h.by_name[name]))
        assert a == b, f"pass-through changed: {name}"
    print("PASS pass-through byte-identical")

    # 4. refusal component removed: ablated projection must collapse relative
    # to the projection present in the base file (allowing requant noise)
    def proj_of(hgn, name, rv):
        e = hgn.by_name[name]
        sides = ha.ht_sides(hgn, e) if e.store == 16 else (None, None)
        codec = ha.make_codec(e.store, e.variant, hgn.payload(e), e.N, e.K,
                              *sides)
        w = codec.decode_rows(0, e.N)
        nrm = np.linalg.norm(w, axis=1)
        keep = nrm > 0  # zero-scale HT rows decode to zero; skip them
        return np.abs(w[keep] @ rv) / nrm[keep]

    for name in ("lm_head.weight",
                 "layers.0.linear_attn.in_proj_qkv.weight",
                 "layers.0.mlp.gate.weight",
                 "layers.0.self_attn.q_proj.weight",
                 "layers.0.self_attn.k_proj.weight",
                 "layers.0.mlp.experts.gate_up_proj.weight",
                 "layers.0.mlp.shared_expert.gate_proj.weight",
                 "layers.0.mlp.shared_expert.up_proj.weight"):
        p0 = proj_of(hb, name, r)
        p1 = proj_of(h, name, r)
        assert p0.mean() > 0.1, f"{name}: test signal missing in base"
        assert p1.mean() < 0.1 * p0.mean() and p1.max() < 0.3 * p0.mean(), \
            f"{name}: residual mean {p1.mean():.4f}/max {p1.max():.4f} " \
            f"vs base mean {p0.mean():.4f}"
    name = "layers.0.attn_hyper_connection.input_mix_weight_down.weight"
    rt4 = np.tile(r, 4)
    p0 = proj_of(hb, name, rt4)
    p1 = proj_of(h, name, rt4)
    assert p0.mean() > 0.1 and p1.mean() < 0.1 * p0.mean() \
        and p1.max() < 0.3 * p0.mean(), \
        f"mixdown: residual mean {p1.mean():.4f}/max {p1.max():.4f} " \
        f"vs base mean {p0.mean():.4f}"
    print("PASS refusal component collapsed vs base signal")

    # 5. extract recovers the direction from the (base, abliterated) pair
    ext = os.path.join(tmp, "recovered.f32")
    rc = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "hgn_abliterate.py"), "extract",
         "--base", base, "--abliterated", out, "--out", ext],
        capture_output=True, text=True)
    print(rc.stdout + rc.stderr)
    assert rc.returncode == 0, "extract failed"
    r2 = np.fromfile(ext, np.float32)
    cos = abs(float(r2 @ r))
    assert cos > 0.99, f"extracted direction cos={cos:.4f}"
    print(f"PASS extract round-trip: |cos| = {cos:.5f}")

    hb.close()
    h.close()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
