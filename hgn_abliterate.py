#!/usr/bin/env python3
"""hgn-abliterate: abliterate halogen .hgn checkpoints (ht43 / Qwen3.8 Flash-Next family).

Implements refusal-direction orthogonalization (Arditi et al. 2024,
"Refusal in Language Models Is Mediated by a Single Direction") directly on
HGN weight containers, per the clean-room spec jtsylve/hgn-spec v1.0.1.

For a unit refusal direction r in residual-stream space, every linear weight
matrix W that reads the residual stream is updated as

    W <- W - alpha * (W r) r^T          (per row: w <- w - alpha * (w.r) r)

so the layer stops reading the refusal direction. Quantized tensors are
dequantized row-block by row-block, transformed, and requantized to the same
store code, variant and (for q4c) codebook, so the output file stays
byte-layout compatible with halogen-flash.

Commands:
  inspect     Dump the header and tensor table (optionally verify checksums).
  extract     Estimate the refusal direction from a (base, abliterated) pair:
              dW = W_abl - W_base is rank-1 (-=(W r) r^T), so the top right
              singular vector of dW is r.
  abliterate  Apply the orthogonalization and write a new .hgn file.

Only numpy is required.
"""

import argparse
import mmap
import os
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import hgnht
except ImportError:
    hgnht = None

MAGIC = 0x314E4748  # "HGN1"
HEADER_SIZE = 0x68
ENTRY_SIZE = 0xA0
IDENTITY_LEN = 64

STORE_NAMES = {
    0: "bf16", 1: "f32", 2: "f16", 3: "i32", 4: "i64", 5: "q4c",
    6: "fp8r", 7: "q8g64", 8: "i4l", 10: "fp8g", 11: "q8s32",
    12: "q6k", 13: "iq4nl", 14: "q4a", 15: "q5a",
    16: "ht",  # store 16 / variant 0x1208: cyclic scalar trellis
}

RESIDUAL = 2560          # hidden size
HYPER = 4 * RESIDUAL     # hyper-connection width (4 streams of 2560)

# Tensor-name patterns whose input axis is the residual stream (or the 4-stream
# hyper-connection bundle). These are the matrices abliteration orthogonalizes.
TARGET_RE = re.compile(
    r"(?:^|\.)(?:"
    r"q_proj|k_proj|v_proj"                    # full attention + mtp
    r"|index_qk_proj"                         # sparse-attention indexer
    r"|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b"  # gated deltanet inputs
    r"|gate_proj|up_proj|gate|shared_expert_gate"  # mlp inputs + router
    r"|input_mix_weight_down|block_inject_weight"  # hyper-connections
    r"|key_proj"                              # ple.key_proj
    r"|fc_hidden"                             # mtp.fc_hidden
    r")\.weight$"
)
TARGET_EXTRA = re.compile(r"(?:^|\.)experts\.gate_up_proj\.weight$|^lm_head\.weight$")
EXCLUDE_RE = re.compile(r"^visual\.|^embed_tokens\.weight$|ngram_embedding")

# Stores this tool can decode AND re-encode losslessly-in-place.
# bf16, f32, q4c(v2), q8g64, q8s32, q6k, q4a, q5a
EDITABLE_STORES = {0, 1, 5, 7, 11, 12, 14, 15, 16}


def pad(x, a):
    return (x + a - 1) // a * a


# ---------------------------------------------------------------------------
# Container

class Entry:
    __slots__ = ("name", "store", "rank", "shape", "offset", "size",
                 "checksum", "variant")

    def parse(self, buf, off):
        name_raw = buf[off:off + 96]
        z = name_raw.find(b"\x00")
        self.name = name_raw[: z if z >= 0 else 96].decode("ascii", "replace")
        (self.store, self.rank) = struct.unpack_from("<II", buf, off + 0x60)
        shape = struct.unpack_from("<4q", buf, off + 0x68)
        self.shape = shape[: self.rank]
        (self.offset, self.size) = struct.unpack_from("<QQ", buf, off + 0x88)
        (self.checksum, self.variant) = struct.unpack_from("<II", buf, off + 0x98)
        return self

    @property
    def K(self):
        return self.shape[-1]

    @property
    def N(self):
        n = 1
        for d in self.shape[:-1]:
            n *= d
        return n


class HgnFile:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.size = os.fstat(self.f.fileno()).st_size
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        if self.size < HEADER_SIZE:
            raise ValueError("file shorter than header")
        (magic, version, n, table_off, data_off, fsize) = struct.unpack_from(
            "<IIQQQQ", self.mm, 0)
        if magic != MAGIC:
            raise ValueError(f"bad magic 0x{magic:08X}, expected HGN1")
        if version not in (1, 2):
            raise ValueError(f"bad version {version}")
        if fsize != self.size:
            raise ValueError(f"header file-size {fsize} != actual {self.size}")
        ident = self.mm[0x28:0x28 + IDENTITY_LEN]
        self.identity = ident[: ident.find(b"\x00")].decode("ascii", "replace")
        self.version = version
        self.entries = []
        if table_off + ENTRY_SIZE * n > self.size:
            raise ValueError("tensor table does not fit in file")
        for i in range(n):
            e = Entry().parse(self.mm, table_off + ENTRY_SIZE * i)
            if e.offset + e.size > self.size:
                raise ValueError(f"{e.name}: payload runs past EOF")
            self.entries.append(e)
        names = [e.name for e in self.entries]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tensor names")
        self.by_name = {e.name: e for e in self.entries}

    def payload(self, entry):
        return memoryview(self.mm)[entry.offset: entry.offset + entry.size]

    def checksum(self, entry):
        return xor_fold_view(self.payload(entry))

    def close(self):
        import gc
        gc.collect()
        try:
            self.mm.close()
        except BufferError:
            pass  # a codec still references the mapping; process exit frees it
        self.f.close()


def xor_fold_view(view):
    """32-bit XOR fold of a byte view, zero-filling a short final word."""
    fold = np.uint32(0)
    n = len(view)
    step = 1 << 23  # 8 MiB chunks
    pos = 0
    while pos < n:
        chunk = view[pos: pos + step]
        m = len(chunk)
        words = np.frombuffer(chunk, dtype=np.uint8)
        if m % 4:
            words = np.concatenate([words, np.zeros(4 - m % 4, np.uint8)])
        fold ^= np.bitwise_xor.reduce(words.view(np.uint32), dtype=np.uint32)
        pos += step
    return int(fold)


def xor_fold_bytes(b):
    return xor_fold_view(memoryview(b))


# ---------------------------------------------------------------------------
# Number format helpers

def bf16_decode(bits):
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16_encode(x):
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    lsb = (u >> 16) & np.uint32(1)
    r = u + np.uint32(0x7FFF) + lsb  # round-to-nearest-even
    return (r >> np.uint32(16)).astype(np.uint16)


# ---------------------------------------------------------------------------
# Store codecs (row-block streaming)
#
# Each codec exposes:
#   size_rule(N, K, variant) -> payload bytes
#   decode(block of rows [a,b)) -> float32 array (b-a, K)
#   encode(rows [a,b), w) -> bytes for those rows' region
# For planar formats (q4c v2) decode/encode touch two planes, so the codec
# works on the whole payload buffer instead of a single region.

def size_rule(store, variant, N, K):
    if store in (0, 2):
        return 2 * N * K
    if store in (1, 3):
        return 4 * N * K
    if store == 4:
        return 8 * N * K
    if store == 5:
        if variant == 2:
            return 64 + pad(N * K // 2, 64) + N * pad(K // 16, 16)
        if variant in (0, 1):
            return 64 + N * (K // 2 + K // 16)
    if store == 6:
        return N * K + 2 * N
    if store == 12:
        return 210 * N * K // 256
    if store in (14, 15):
        B = pad(N * K // 2, 64)
        P1 = B if store == 14 else B + pad(N * K // 8, 64)
        if variant == 0:
            return P1 + N * pad(4 * K // 32, 16)
        if variant == 1:
            P2 = pad(P1 + N * pad(2 * K // 32, 16), 64)
            return P2 + N * pad(4 * K // 256, 16)
    if store == 7:
        return N * (K + K // 16)
    if store == 11:
        return N * pad(K + K // 16, 16)
    if store == 16:
        if variant == 0x1208:
            return N * K // 2
        if variant == 0x1206:
            return 3 * N * K // 8
    raise ValueError(f"no size rule for store {store} variant {variant}")


class Q4cV2Codec:
    """q4c variant 2: planar codes + f16 scales per group of 32, f32 codebook."""

    def __init__(self, payload, N, K):
        if K % 32:
            raise ValueError("q4c v2 requires K multiple of 32")
        self.N, self.K = N, K
        self.cb = np.frombuffer(payload[:64], dtype=np.float32).copy()
        self.sorted_cb = bool(np.all(np.diff(self.cb) > 0))
        self.code_base = 64
        self.S = 64 + pad(N * K // 2, 64)
        self.R = pad(K // 16, 16)
        self.out = bytearray(len(payload))
        self.out[:64] = payload[:64]  # keep codebook verbatim

    def decode_rows(self, a, b):
        K = self.K
        codes = np.frombuffer(
            self.payload_view(), np.uint8,
            count=(b - a) * (K // 2), offset=self.code_base + a * (K // 2),
        ).reshape(b - a, K // 2)
        idx = np.empty((b - a, K), np.uint8)
        idx[:, 0::2] = codes & 0x0F
        idx[:, 1::2] = codes >> 4
        sc = np.frombuffer(
            self.payload_view(), np.float16,
            count=(b - a) * (K // 32), offset=self.S + a * self.R,
        ).reshape(b - a, K // 32).astype(np.float32)
        return self.cb[idx] * np.repeat(sc, 32, axis=1)

    def payload_view(self):
        return self._pv

    def set_payload(self, view):
        self._pv = view

    def encode_rows(self, a, b, w):
        K = self.K
        ng = K // 32
        g = w.reshape(b - a, ng, 32)
        mx = np.abs(g).max(axis=2)
        cbmax = float(np.abs(self.cb).max()) or 1.0
        s = mx / cbmax
        safe = np.where(s > 0, s, 1.0)
        ws = (g / safe[:, :, None]).astype(np.float32)
        codes = quantize_to_codebook(ws, self.cb, self.sorted_cb)
        codes = codes.reshape(b - a, K)
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        self.out[self.code_base + a * (K // 2):
                 self.code_base + b * (K // 2)] = packed.reshape(-1).tobytes()
        sf16 = s.astype(np.float16)
        row = np.zeros((b - a, self.R), np.uint8)
        row[:, : ng * 2] = sf16.view(np.uint8).reshape(b - a, ng * 2)
        self.out[self.S + a * self.R: self.S + b * self.R] = row.reshape(-1).tobytes()

    def finalize(self):
        return bytes(self.out)


def quantize_to_codebook(ws, cb, sorted_cb):
    """Nearest-codebook-level index for each value in ws (float32)."""
    if sorted_cb:
        idx = np.searchsorted(cb, ws)
        idx = np.clip(idx, 1, 15)
        left = cb[idx - 1]
        right = cb[idx]
        take_left = np.abs(ws - left) <= np.abs(right - ws)
        return np.where(take_left, idx - 1, idx).astype(np.uint8)
    # unsorted fallback, sub-chunked to bound memory
    out = np.empty(ws.shape, np.uint8)
    flat = ws.reshape(-1)
    oflat = out.reshape(-1)
    step = 1 << 20
    for i in range(0, flat.size, step):
        blk = flat[i: i + step]
        oflat[i: i + step] = np.abs(blk[:, None] - cb[None, :]).argmin(axis=1)
    return out


class Q8g64Codec:
    """q8g64: per-row codes + affine (f16 scale, f16 bias) per group of 64."""

    def __init__(self, payload, N, K):
        if K % 64:
            raise ValueError("q8g64 requires K multiple of 64")
        self.N, self.K = N, K
        self.row = K + K // 16
        self.payload = payload

    def decode_rows(self, a, b):
        K = self.K
        base = a * self.row
        blk = np.frombuffer(self.payload, np.uint8,
                            count=(b - a) * self.row, offset=base)
        codes = blk.reshape(b - a, self.row)[:, :K].astype(np.float32)
        rec = np.frombuffer(
            blk.reshape(b - a, self.row)[:, K:].tobytes(), np.float16
        ).reshape(b - a, K // 64, 2)
        sc = rec[:, :, 0].astype(np.float32)
        bi = rec[:, :, 1].astype(np.float32)
        return codes * np.repeat(sc, 64, axis=1) + np.repeat(bi, 64, axis=1)

    def encode_rows(self, a, b, w):
        K = self.K
        ng = K // 64
        g = w.reshape(b - a, ng, 64)
        mn = g.min(axis=2)
        mx = g.max(axis=2)
        rng = mx - mn
        s = rng / np.float32(255.0)
        safe = np.where(s > 0, s, 1.0)
        code = np.clip(np.rint((g - mn[:, :, None]) / safe[:, :, None]),
                       0, 255).astype(np.uint8)
        out = np.zeros((b - a, self.row), np.uint8)
        out[:, :K] = code.reshape(b - a, K)
        rec = np.stack([s.astype(np.float16), mn.astype(np.float16)], axis=2)
        out[:, K:] = rec.view(np.uint8).reshape(b - a, ng * 4)
        self.out = getattr(self, "out", bytearray())
        self.out += out.reshape(-1).tobytes()

    def finalize(self):
        return bytes(self.out)


class Q8s32Codec:
    """q8s32: signed 8-bit codes, f16 scale per group of 32, padded rows."""

    def __init__(self, payload, N, K):
        if K % 64:
            raise ValueError("q8s32 requires K multiple of 64")
        self.N, self.K = N, K
        self.row = pad(K + K // 16, 16)
        self.payload = payload
        self.out = bytearray()

    def decode_rows(self, a, b):
        K = self.K
        blk = np.frombuffer(self.payload, np.uint8,
                            count=(b - a) * self.row, offset=a * self.row)
        rows = blk.reshape(b - a, self.row)
        codes = rows[:, :K].astype(np.int8).astype(np.float32)
        sc = np.frombuffer(rows[:, K: K + K // 16].tobytes(), np.float16
                           ).reshape(b - a, K // 32).astype(np.float32)
        return codes * np.repeat(sc, 32, axis=1)

    def encode_rows(self, a, b, w):
        K = self.K
        ng = K // 32
        g = w.reshape(b - a, ng, 32)
        mx = np.abs(g).max(axis=2)
        s = mx / np.float32(127.0)
        safe = np.where(s > 0, s, 1.0)
        code = np.clip(np.rint(g / safe[:, :, None]), -127, 127
                       ).astype(np.int8)
        out = np.zeros((b - a, self.row), np.uint8)
        out[:, :K] = code.view(np.uint8).reshape(b - a, K)
        out[:, K: K + ng * 2] = s.astype(np.float16).view(np.uint8
                                                          ).reshape(b - a, ng * 2)
        self.out += out.reshape(-1).tobytes()

    def finalize(self):
        return bytes(self.out)


class DenseCodec:
    """bf16 (store 0) and f32 (store 1)."""

    def __init__(self, payload, N, K, store):
        self.N, self.K, self.store = N, K, store
        self.payload = payload
        self.out = bytearray()

    def decode_rows(self, a, b):
        K = self.K
        if self.store == 0:
            bits = np.frombuffer(self.payload, np.uint16,
                                 count=(b - a) * K, offset=2 * a * K)
            return bf16_decode(bits).reshape(b - a, K)
        return np.frombuffer(self.payload, np.float32,
                             count=(b - a) * K,
                             offset=4 * a * K).reshape(b - a, K).copy()

    def encode_rows(self, a, b, w):
        if self.store == 0:
            self.out += bf16_encode(w).tobytes()
        else:
            self.out += np.ascontiguousarray(w, np.float32).tobytes()

    def finalize(self):
        return bytes(self.out)


class Q6kCodec:
    """q6k: GGUF Q6_K super-blocks, 210 bytes per 256 values."""

    _I = np.arange(256)

    def __init__(self, payload, N, K):
        if K % 256:
            raise ValueError("q6k requires K multiple of 256")
        self.N, self.K = N, K
        self.nb = K // 256
        self.payload = payload
        self.out = bytearray()
        i = self._I
        self.h = i // 128
        self.q = (i % 128) // 32
        self.l = i % 32

    def decode_rows(self, a, b):
        nb = self.nb
        blk = np.frombuffer(self.payload, np.uint8,
                            count=(b - a) * 210 * nb,
                            offset=(b - a) * 0 + a * 210 * nb)
        rows = blk.reshape(b - a, nb, 210)
        ql = rows[:, :, 0:128]
        qh = rows[:, :, 128:192]
        sc = rows[:, :, 192:208].astype(np.int8).astype(np.float32)
        d = np.frombuffer(rows[:, :, 208:210].tobytes(),
                          np.float16).reshape(b - a, nb).astype(np.float32)
        i = self._I
        qli = 64 * self.h + 32 * (self.q % 2) + self.l
        sh = 4 * (self.q // 2)
        qhi = 32 * self.h + self.l
        shi = 2 * self.q
        lo = (ql[:, :, qli] >> sh) & 0x0F
        hi = (qh[:, :, qhi] >> shi) & 0x03
        codes = (lo | (hi << 4)).astype(np.float32) - np.float32(32.0)
        scales = sc[:, :, i // 16]
        return (codes * scales * d[:, :, None]).reshape(b - a, self.K)

    def encode_rows(self, a, b, w):
        nb = self.nb
        g = w.reshape(b - a, nb, 256)
        grp = np.abs(g.reshape(b - a, nb, 16, 16)).max(axis=3)  # (r,nb,16)
        d = np.maximum(grp.max(axis=2) / np.float32(31.0 * 127.0),
                       np.float32(1e-7))
        d16 = d.astype(np.float16)
        d = d16.astype(np.float32)
        sc = np.clip(np.rint(grp / (np.float32(31.0) * d[:, :, None])),
                     0, 127).astype(np.int8)
        sc32 = np.where(sc > 0, sc.astype(np.float32), np.float32(1.0))
        eff = np.repeat(sc32 * d[:, :, None], 16, axis=2)
        eff = np.where(eff > 0, eff, 1.0)
        codes = np.clip(np.rint(g / eff) + 32, 0, 63).astype(np.uint8)
        lo = codes & 0x0F
        hi = codes >> 4
        ql = np.zeros((b - a, nb, 128), np.uint8)
        qh = np.zeros((b - a, nb, 64), np.uint8)
        for qq in range(4):
            sel = self.q == qq
            bidx = 64 * self.h[sel] + 32 * (qq % 2) + self.l[sel]
            ql[:, :, bidx] |= (lo[:, :, sel] << (4 * (qq // 2)))
            qh[:, :, 32 * self.h[sel] + self.l[sel]] |= (hi[:, :, sel]
                                                         << (2 * qq))
        out = np.zeros((b - a, nb, 210), np.uint8)
        out[:, :, 0:128] = ql
        out[:, :, 128:192] = qh
        out[:, :, 192:208] = sc.view(np.uint8)
        out[:, :, 208:210] = d16.view(np.uint8).reshape(b - a, nb, 2)
        self.out += out.reshape(-1).tobytes()

    def finalize(self):
        return bytes(self.out)


class AffineCodec:
    """q4a / q5a: planar affine codes. bits=4 or 5, variant 0 (pair) or 1."""

    def __init__(self, payload, N, K, bits, variant):
        if variant not in (0, 1):
            raise ValueError(f"q{bits}a variant must be 0 or 1")
        if variant == 0 and K % 32:
            raise ValueError("affine variant 0 requires K multiple of 32")
        if variant == 1 and K % 256:
            raise ValueError("affine variant 1 requires K multiple of 256")
        self.N, self.K, self.bits, self.variant = N, K, bits, variant
        self.maxc = (1 << bits) - 1
        self.B = pad(N * K // 2, 64)
        self.H = self.B
        self.P1 = self.B if bits == 4 else self.B + pad(N * K // 8, 64)
        self.S1 = (pad(4 * K // 32, 16) if variant == 0
                   else pad(2 * K // 32, 16))
        self.P2 = pad(self.P1 + N * self.S1, 64)
        self.S2 = pad(4 * K // 256, 16)
        self.payload = payload
        # planar layout: buffer each plane separately, concatenate in order
        self._nib = bytearray()
        self._hi = bytearray()
        self._p1 = bytearray()
        self._p2 = bytearray()

    def _codes(self, a, b):
        K = self.K
        nib = np.frombuffer(self.payload, np.uint8,
                            count=(b - a) * (K // 2),
                            offset=a * (K // 2)).reshape(b - a, K // 2)
        codes = np.empty((b - a, K), np.uint16)
        codes[:, 0::2] = nib & 0x0F
        codes[:, 1::2] = nib >> 4
        if self.bits == 5:
            words = np.frombuffer(self.payload, np.uint32,
                                  count=(b - a) * (K // 32),
                                  offset=self.H + a * (K // 4)
                                  ).reshape(b - a, K // 32)
            hi = (words[:, :, None] >> np.arange(32, dtype=np.uint32)) & 1
            codes += 16 * hi.reshape(b - a, K)
        return codes.astype(np.uint8)

    def decode_rows(self, a, b):
        K = self.K
        ng = K // 32
        codes = self._codes(a, b).reshape(b - a, ng, 32).astype(np.float32)
        plane = np.frombuffer(self.payload, np.uint8,
                              count=(b - a) * self.S1,
                              offset=self.P1 + a * self.S1
                              ).reshape(b - a, self.S1)
        if self.variant == 0:
            rec = np.frombuffer(plane[:, :ng * 4].tobytes(),
                                np.float16).reshape(b - a, ng, 2)
            scale = rec[:, :, 0].astype(np.float32)
            zero = rec[:, :, 1].astype(np.float32)
        else:
            scm = plane[:, :ng * 2]
            sc = scm[:, 0::2].astype(np.float32)
            m = scm[:, 1::2].astype(np.float32)
            nb = K // 256
            p2 = np.frombuffer(self.payload, np.uint8,
                               count=(b - a) * self.S2,
                               offset=self.P2 + a * self.S2
                               ).reshape(b - a, self.S2)
            dd = np.frombuffer(p2[:, :nb * 4].tobytes(),
                               np.float16).reshape(b - a, nb, 2)
            d = dd[:, :, 0].astype(np.float32)
            dmin = dd[:, :, 1].astype(np.float32)
            rep = ng // nb
            scale = np.repeat(d, rep, axis=1) * sc
            zero = -np.repeat(dmin, rep, axis=1) * m
        return (codes * scale[:, :, None]
                + zero[:, :, None]).reshape(b - a, K)

    def encode_rows(self, a, b, w):
        K = self.K
        ng = K // 32
        g = w.reshape(b - a, ng, 32)
        mn = g.min(axis=2)
        mx = g.max(axis=2)
        rng = mx - mn
        if self.variant == 0:
            s = (rng / np.float32(float(self.maxc))).astype(np.float16)
            s32 = s.astype(np.float32)
            zero = mn.astype(np.float16)
            safe = np.where(s32 > 0, s32, 1.0)
            code = np.clip(np.rint((g - mn[:, :, None]) / safe[:, :, None]),
                           0, self.maxc).astype(np.uint8)
            rec = np.stack([s, zero], axis=2).view(np.uint8
                                                   ).reshape(b - a, ng * 4)
            plane = np.zeros((b - a, self.S1), np.uint8)
            plane[:, :ng * 4] = rec
        else:
            nb = K // 256
            rep = ng // nb
            rng_b = rng.reshape(b - a, nb, rep).max(axis=2)
            minv = (-mn).reshape(b - a, nb, rep).max(axis=2)
            d = np.maximum(rng_b / np.float32(float(self.maxc) * 63.0),
                           np.float32(1e-7)).astype(np.float16)
            dmin = np.maximum(minv / np.float32(63.0),
                              np.float32(1e-7)).astype(np.float16)
            d32 = d.astype(np.float32)
            dm32 = dmin.astype(np.float32)
            drep = np.repeat(d32, rep, axis=1)
            dmrep = np.repeat(dm32, rep, axis=1)
            sc = np.clip(np.rint(rng / (np.float32(float(self.maxc)) * drep)),
                         0, 63).astype(np.uint8)
            m = np.clip(np.rint((-mn) / dmrep), 0, 63).astype(np.uint8)
            scale = drep * sc.astype(np.float32)
            zero = -dmrep * m.astype(np.float32)
            safe = np.where(scale > 0, scale, 1.0)
            code = np.clip(np.rint((g - zero[:, :, None])
                                   / safe[:, :, None]), 0, self.maxc
                           ).astype(np.uint8)
            scm = np.empty((b - a, ng * 2), np.uint8)
            scm[:, 0::2] = sc.reshape(b - a, ng)
            scm[:, 1::2] = m.reshape(b - a, ng)
            plane = np.zeros((b - a, self.S1), np.uint8)
            plane[:, :ng * 2] = scm
            rec2 = np.stack([d, dmin], axis=2).view(np.uint8
                                                    ).reshape(b - a, nb * 4)
            p2 = np.zeros((b - a, self.S2), np.uint8)
            p2[:, :nb * 4] = rec2
        codes = code.reshape(b - a, K)
        self._nib += (codes[:, 0::2]
                      | (codes[:, 1::2] << 4)).reshape(-1).tobytes()
        if self.bits == 5:
            hi = (codes.reshape(b - a, ng, 32) >> 4) & 1
            words = (hi.astype(np.uint32)
                     << np.arange(32, dtype=np.uint32)).sum(
                         axis=2, dtype=np.uint32)
            self._hi += words.tobytes()
        self._p1 += plane.reshape(-1).tobytes()
        if self.variant == 1:
            self._p2 += p2.reshape(-1).tobytes()

    def finalize(self):
        return bytes(self._nib) + bytes(self._hi) + bytes(self._p1) \
            + bytes(self._p2)


class HTCodec:
    """HT (store 16, variants 0x1208/4-bit and 0x1206/3-bit): cyclic scalar
    trellis + H128 rotations.

    Side planes come from the tensor's .suh/.svh siblings: su (K,) of +/-1
    signs and sv (N,) signed row scales. Fused rank-3 expert tensors instead
    carry per-expert planes: su (E, K) and sv (E, O) with N = E*O; their rows
    are processed expert-by-expert with that expert's planes. Row rotations
    couple blocks of 128 rows, so decode expands row ranges to 128 boundaries
    and encode requires 128-aligned chunks (expert-aligned for rank-3).
    Encoding uses the native Viterbi encoder (ht_trellis.so).
    """

    def __init__(self, payload, N, K, su, sv, workers=8, beam=128, bits=4):
        self.bits = bits
        self.payload = payload
        self.N, self.K = N, K
        self.workers, self.beam = workers, beam
        self.out = bytearray()
        su = np.ascontiguousarray(su, np.float32)
        sv = np.ascontiguousarray(sv, np.float32)
        if su.ndim == 2:
            self.E, self.O = su.shape[0], sv.shape[1]
            if self.E * self.O != N or su.shape[1] != K:
                raise ValueError("HT per-expert plane shape mismatch")
            if self.O % 128 or K % 128:
                raise ValueError("HT requires O and K divisible by 128")
            self.per_expert = True
        else:
            if N % 128 or K % 128:
                raise ValueError("HT requires N and K divisible by 128")
            self.per_expert = False
        self.su, self.sv = su, sv
        if len(payload) != N * K * bits // 8:
            raise ValueError("HT payload size mismatch")

    def _slice(self, a, b):
        return bytes(self.payload[a * self.K * self.bits // 8:
                                  b * self.K * self.bits // 8])

    def decode_rows(self, a, b):
        if not self.per_expert:
            first = a // 128 * 128
            last = (b + 127) // 128 * 128
            dec = hgnht.decode(self._slice(first, last),
                               (last - first, self.K), self.su,
                               self.sv[first:last], self.bits)
            return np.ascontiguousarray(dec[a - first:b - first])
        parts = []
        for e in range(a // self.O, (b + self.O - 1) // self.O):
            lo, hi = max(a, e * self.O), min(b, (e + 1) * self.O)
            first = (lo - e * self.O) // 128 * 128 + e * self.O
            last = (hi - e * self.O + 127) // 128 * 128 + e * self.O
            dec = hgnht.decode(self._slice(first, last), (last - first,
                               self.K), self.su[e],
                               self.sv[e][first - e * self.O:
                                          last - e * self.O], self.bits)
            parts.append(dec[lo - first:hi - first])
        return np.ascontiguousarray(np.concatenate(parts))

    def _rotate(self, w, su, sv):
        rot = hgnht.rotate(np.ascontiguousarray(w, np.float32), su, sv)
        if not np.all(np.isfinite(rot)):
            bad = int(np.isnan(rot).sum() + np.isinf(rot).sum())
            raise ValueError(
                f"HT encode: {bad} non-finite rotated values — the input "
                f"weights or direction vector contain NaN/inf")
        return rot

    def encode_rows(self, a, b, w):
        if self.per_expert:
            if a % self.O or b % self.O:
                raise ValueError("HT fused experts encode requires "
                                 f"expert-aligned chunks ({self.O} rows)")
            for e in range(a // self.O, b // self.O):
                rot = self._rotate(w[(e * self.O - a):(e + 1) * self.O - a],
                                   self.su[e], self.sv[e])
                self.out += hgnht.encode_rot(rot, self.workers, self.beam,
                                             self.bits)
        else:
            if a % 128 or b % 128:
                raise ValueError("HT encode requires 128-aligned row chunks "
                                 "(use a --chunk-rows that is a multiple "
                                 "of 128)")
            rot = self._rotate(w, self.su, self.sv[a:b])
            self.out += hgnht.encode_rot(rot, self.workers, self.beam,
                                         self.bits)

    def finalize(self):
        return bytes(self.out)


def ht_sides(hgn, entry):
    """Load the .suh/.svh side planes for an HT tensor."""
    root = entry.name[:-7] if entry.name.endswith(".weight") else entry.name
    su_e = hgn.by_name.get(root + ".suh")
    sv_e = hgn.by_name.get(root + ".svh")
    if su_e is None or sv_e is None:
        raise ValueError(f"{entry.name}: missing HT side planes "
                         f"{root}.suh/.svh")
    su = np.frombuffer(bytes(hgn.payload(su_e)), "<f2").astype(np.float32)
    sv = np.frombuffer(bytes(hgn.payload(sv_e)), "<f2").astype(np.float32)
    if entry.rank == 3:
        # fused experts carry per-expert planes: suh [E, K], svh [E, O]
        su = su.reshape(entry.shape[0], entry.K)
        sv = sv.reshape(entry.shape[0], entry.shape[1])
    return su, sv


def make_codec(store, variant, payload, N, K, su=None, sv=None):
    if store == 5:
        if variant != 2:
            raise ValueError("only q4c variant 2 is editable")
        c = Q4cV2Codec(payload, N, K)
        c.set_payload(payload)
        return c
    if store == 7:
        return Q8g64Codec(payload, N, K)
    if store == 11:
        return Q8s32Codec(payload, N, K)
    if store == 12:
        return Q6kCodec(payload, N, K)
    if store in (14, 15):
        return AffineCodec(payload, N, K, 4 if store == 14 else 5, variant)
    if store == 16:
        if hgnht is None:
            raise ValueError("store 16 (HT) needs hgnht.py/hgnenc.py next to "
                             "this script")
        if variant not in (0x1208, 0x1206):
            raise ValueError(f"HT variant 0x{variant:04x} not editable "
                             f"(only 0x1208 / 4-bit and 0x1206 / 3-bit)")
        if su is None or sv is None:
            raise ValueError("HT needs its .suh/.svh side planes")
        return HTCodec(payload, N, K, su, sv,
                       bits=3 if variant == 0x1206 else 4)
    if store in (0, 1):
        return DenseCodec(payload, N, K, store)
    raise ValueError(f"store {store} not editable")


# ---------------------------------------------------------------------------
# Abliteration core

def load_direction(path):
    if path.endswith(".npy"):
        r = np.load(path).astype(np.float32).reshape(-1)
    else:
        r = np.fromfile(path, dtype=np.float32)
    if r.size != RESIDUAL:
        raise ValueError(f"direction has {r.size} values, expected {RESIDUAL}")
    if not np.all(np.isfinite(r)):
        raise ValueError(
            f"direction contains {int(np.isnan(r).sum())} NaN / "
            f"{int(np.isinf(r).sum())} inf values — bad direction file "
            f"(a NaN direction usually means the extraction tensor was "
            f"byte-identical in the pair)")
    nrm = float(np.linalg.norm(r))
    if nrm == 0:
        raise ValueError("direction is all zeros")
    return r / nrm


def layer_of(name):
    m = re.match(r"(?:layers|mtp\.layers)\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    if name.startswith("mtp."):
        return "mtp"
    return None


def in_layer_spec(layer, spec):
    if spec is None:
        return True
    if layer is None:  # global tensors (lm_head, mixer) always included
        return True
    if layer == "mtp":
        return "mtp" in spec
    return layer in spec


def parse_layer_spec(text):
    spec = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if part == "mtp":
            spec.add("mtp")
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            spec.update(range(int(lo), int(hi) + 1))
        else:
            spec.add(int(part))
    return spec


def select_targets(hgn, layer_spec, only_re, skip_re, verbose=True,
                   writers=None):
    if writers == "contract":
        names = set(contract_writers(hgn))
        targets = []
        for e in hgn.entries:
            if e.name not in names:
                continue
            if only_re and not only_re.search(e.name):
                continue
            if skip_re and skip_re.search(e.name):
                continue
            if not in_layer_spec(layer_of(e.name), layer_spec):
                continue
            if e.store not in EDITABLE_STORES:
                if verbose:
                    print(f"  skip {e.name}: store "
                          f"{STORE_NAMES.get(e.store, e.store)} not editable")
                continue
            if e.store == 16 and e.variant not in (0x1208, 0x1206):
                if verbose:
                    print(f"  skip {e.name}: HT variant 0x{e.variant:04x} "
                          f"not editable")
                continue
            if _out_axis_side(e) is None:
                raise ValueError(
                    f"{e.name}: contract writer shape {e.shape} has no "
                    f"residual output axis")
            targets.append(e)
        return targets
    targets = []
    for e in hgn.entries:
        if EXCLUDE_RE.search(e.name):
            continue
        if not (TARGET_RE.search(e.name) or TARGET_EXTRA.search(e.name)):
            continue
        if only_re and not only_re.search(e.name):
            continue
        if skip_re and skip_re.search(e.name):
            continue
        if e.K not in (RESIDUAL, HYPER):
            if verbose:
                print(f"  skip {e.name}: K={e.K} is not a residual axis")
            continue
        if not in_layer_spec(layer_of(e.name), layer_spec):
            continue
        if e.store not in EDITABLE_STORES:
            if verbose:
                print(f"  skip {e.name}: store {STORE_NAMES.get(e.store, e.store)} "
                      f"not editable by this tool")
            continue
        if e.store == 16:
            if e.variant not in (0x1208, 0x1206):
                if verbose:
                    print(f"  skip {e.name}: HT variant 0x{e.variant:04x} "
                          f"not editable")
                continue
            if e.N % 128 or e.K % 128:
                if verbose:
                    print(f"  skip {e.name}: HT needs dims divisible by 128")
                continue
            root = e.name[:-7] if e.name.endswith(".weight") else e.name
            if (hgn.by_name.get(root + ".suh") is None
                    or hgn.by_name.get(root + ".svh") is None):
                if verbose:
                    print(f"  skip {e.name}: missing HT side planes "
                          f"{root}.suh/.svh")
                continue
        try:
            expected = size_rule(e.store, e.variant, e.N, e.K)
        except ValueError as exc:
            if verbose:
                print(f"  skip {e.name}: {exc}")
            continue
        if expected != e.size:
            raise ValueError(f"{e.name}: table size {e.size} != size rule {expected}")
        targets.append(e)
    return targets


def transform_rows(w, r, alpha):
    """w: (rows, K) float32, r: unit vector length K. w <- w - a (w.r) r^T."""
    proj = w @ r
    w -= np.float32(alpha) * proj[:, None] * r[None, :]
    return w


def contract_writers(hgn):
    """The pinned 149-writer abliteration contract (output-side matrices that
    write the residual stream, plus embed_tokens). Returns the subset of
    names present in this file. Matches the official ht43 tooling contract:
    per layer the experts.down_proj, the dense out_proj (o_proj on every 4th
    layer) and shared_expert.down_proj, plus the MTP head, embed_tokens and
    layers.1.ple.value_proj."""
    names = []
    for layer in range(48):
        names.append(f"layers.{layer}.mlp.experts.down_proj.weight")
        dense = (f"layers.{layer}.self_attn.o_proj.weight"
                 if layer % 4 == 3
                 else f"layers.{layer}.linear_attn.out_proj.weight")
        names.append(dense)
        names.append(f"layers.{layer}.mlp.shared_expert.down_proj.weight")
    names.extend([
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        "mtp.layers.0.mlp.experts.down_proj.weight",
        "embed_tokens.weight",
        "layers.1.ple.value_proj.weight",
    ])
    assert len(names) == 149 and len(set(names)) == 149
    return [n for n in names if n in hgn.by_name]


def _out_axis_side(e):
    """Which axis of a contract writer holds the residual output features:
    'last' for embed_tokens (each row IS a residual vector), 'rows' for
    2-D writers (output index = row), 'experts' for fused down_proj
    (output index = row % 2560)."""
    if e.name == "embed_tokens.weight":
        return "last"
    if e.rank == 2 and e.N == RESIDUAL:
        return "rows"
    if e.rank == 3 and e.shape[1] == RESIDUAL:
        return "experts"
    return None


def accumulate_cols(proj, w, r, mode, a):
    """Accumulate proj += r^T W over a decoded row chunk [a:b) of a writer
    whose output axis is the row axis. proj: (K,) for 'rows' mode or
    (n_experts, K) for 'experts' mode; w: (rows, K) float32."""
    out_idx = (np.arange(a, a + w.shape[0]) % RESIDUAL) if mode == "experts" \
        else np.arange(a, a + w.shape[0])
    contrib = r[out_idx][:, None] * w
    if mode == "rows":
        proj += contrib.sum(axis=0)
    else:
        np.add.at(proj, np.arange(a, a + w.shape[0]) // RESIDUAL, contrib)


def ablate_out_axis(codec, e, r, alpha, chunk_rows):
    """Writer-side ablation: remove the r component from the OUTPUT axis of
    a contract writer so the refusal direction can never be written into
    the residual stream. Single pass: chunks are chosen so every row whose
    output index feeds a given accumulation is present in the same chunk
    (2-D writers fit in one chunk; fused experts chunk expert-by-expert)."""
    side = _out_axis_side(e)
    if side == "last":
        # embed_tokens: each row IS a residual vector
        for a in range(0, e.N, chunk_rows):
            b = min(a + chunk_rows, e.N)
            w = codec.decode_rows(a, b)
            transform_rows(w, r, alpha)
            codec.encode_rows(a, b, w)
    elif side == "rows":
        # 2-D [2560, K]: proj = r^T W needs every output row at once
        w = codec.decode_rows(0, e.N)
        proj = r @ w
        w -= np.float32(alpha) * r[:, None] * proj[None, :]
        codec.encode_rows(0, e.N, w)
    else:
        # fused experts: rows are expert-major blocks of RESIDUAL rows
        step = RESIDUAL * max(1, chunk_rows // RESIDUAL)
        for a in range(0, e.N, step):
            b = min(a + step, e.N)
            w = codec.decode_rows(a, b)
            W = w.reshape((b - a) // RESIDUAL, RESIDUAL, e.K)
            proj = np.einsum("o,eok->ek", r, W)
            W -= np.float32(alpha) * r[None, :, None] * proj[:, None, :]
            codec.encode_rows(a, b, w)


def abliterate(in_path, out_path, direction, alpha=1.0, layer_spec=None,
               only=None, skip=None, identity=None, verify=True,
               chunk_rows=4096, writers=None):
    hgn = HgnFile(in_path)
    r2560 = load_direction(direction)
    only_re = re.compile(only) if only else None
    skip_re = re.compile(skip) if skip else None
    print(f"input: {in_path}  identity={hgn.identity!r}  entries={len(hgn.entries)}")
    targets = select_targets(hgn, layer_spec, only_re, skip_re,
                             writers=writers)
    print(f"targets: {len(targets)} tensors")
    if not targets:
        raise ValueError(
            "no editable target tensors: every residual-reading matrix in "
            "this checkpoint uses a store this tool cannot decode (e.g. the "
            "3-bit HT variant 0x1206). Abliteration would be a no-op.")
    target_names = {e.name for e in targets}

    n = len(hgn.entries)
    payload_start = pad(HEADER_SIZE + ENTRY_SIZE * n, 64)
    out = open(out_path, "wb")
    out.truncate(payload_start)  # header+table rewritten at the end
    new_checksums = []
    new_offsets = []
    off = payload_start
    for e in hgn.entries:
        new_offsets.append(off)
        # seek to the aligned offset: tensors are spaced by pad(size, 64),
        # so any size that is not a multiple of 64 leaves a gap that must
        # NOT be skipped by sequential writes (the table points here)
        out.seek(off)
        if e.name not in target_names:
            view = hgn.payload(e)
            write_copy(out, view)
            # bytes are unchanged, so the table checksum still holds
            new_checksums.append(e.checksum)
        else:
            print(f"  ablating {e.name} [{e.store_name()} {e.shape}] "
                  f"store={STORE_NAMES[e.store]}")
            payload = hgn.payload(e)
            if verify:
                got = xor_fold_view(payload)
                if got != e.checksum:
                    raise ValueError(f"{e.name}: checksum mismatch on input "
                                     f"(0x{got:08X} != 0x{e.checksum:08X})")
            codec = make_codec(e.store, e.variant, payload, e.N, e.K,
                               *(ht_sides(hgn, e) if e.store == 16 else (None, None)))
            side = _out_axis_side(e) if writers == "contract" else None
            if side is not None:
                ablate_out_axis(codec, e, r2560, alpha, chunk_rows)
            else:
                r = r2560 if e.K == RESIDUAL else np.tile(r2560, 4)
                r = r / np.linalg.norm(r)  # tiled vector is not unit-norm
                for a in range(0, e.N, chunk_rows):
                    b = min(a + chunk_rows, e.N)
                    w = codec.decode_rows(a, b)
                    transform_rows(w, r, alpha)
                    codec.encode_rows(a, b, w)
            blob = codec.finalize()
            if len(blob) != e.size:
                raise AssertionError(f"{e.name}: re-encoded size {len(blob)} "
                                     f"!= original {e.size}")
            out.write(blob)
            new_checksums.append(xor_fold_bytes(blob))
        off += pad(e.size, 64)
    file_size = off

    # header + table last, per the spec's writer procedure
    out.seek(0)
    ident = (identity or hgn.identity).encode("ascii")
    if len(ident) >= IDENTITY_LEN:
        raise ValueError("identity too long")
    ident = ident.ljust(IDENTITY_LEN, b"\x00")
    out.write(struct.pack("<IIQQQQ", MAGIC, 2, n, HEADER_SIZE,
                          payload_start, file_size))
    out.write(ident)
    for i, e in enumerate(hgn.entries):
        name = e.name.encode("ascii")
        if len(name) > 95:
            raise ValueError(f"name too long: {e.name}")
        rec = bytearray(ENTRY_SIZE)
        rec[0:96] = name.ljust(96, b"\x00")
        struct.pack_into("<II", rec, 0x60, e.store, e.rank)
        struct.pack_into("<4q", rec, 0x68, *(list(e.shape) + [0] * (4 - e.rank)))
        struct.pack_into("<QQ", rec, 0x88, new_offsets[i], e.size)
        struct.pack_into("<II", rec, 0x98, new_checksums[i], e.variant)
        out.write(rec)
    out.truncate(file_size)
    out.flush()
    os.fsync(out.fileno())
    out.close()
    hgn.close()
    print(f"wrote {out_path} ({file_size} bytes, {n} tensors)")


def write_copy(out, view):
    step = 1 << 24
    pos = 0
    while pos < len(view):
        out.write(view[pos: pos + step])
        pos += step


# Entry helper for logging
def _store_name(self):
    return STORE_NAMES.get(self.store, f"store{self.store}")


Entry.store_name = _store_name


# ---------------------------------------------------------------------------
# Direction extraction from a (base, abliterated) pair

def _editable_pair_tensor(base, abl, wanted):
    """Return a tensor name usable for extraction: the wanted one if both
    files store it editably AND its payload actually differs, else the
    first target-pattern tensor that is editable in both and differs.
    A byte-identical tensor has zero dW and yields a NaN direction."""
    eb, ea = base.by_name.get(wanted), abl.by_name.get(wanted)
    if eb is not None and ea is not None and eb.store in EDITABLE_STORES \
            and ea.store in EDITABLE_STORES and eb.K == RESIDUAL \
            and eb.checksum != ea.checksum:
        return wanted
    for relaxed in (False, True):
        for e in base.entries:
            ea = abl.by_name.get(e.name)
            if ea is None or e.shape != ea.shape:
                continue
            if e.K != RESIDUAL or e.N < 32:
                continue
            if e.checksum == ea.checksum:  # zero dW: useless for extraction
                continue
            if not relaxed:
                if not (TARGET_RE.search(e.name) or TARGET_EXTRA.search(e.name)):
                    continue
                if EXCLUDE_RE.search(e.name):
                    continue
            if e.store in EDITABLE_STORES and ea.store in EDITABLE_STORES:
                return e.name
    return None


def extract(base_path, abl_path, tensor_name, out_path, chunk_rows=8192):
    base = HgnFile(base_path)
    abl = HgnFile(abl_path)
    if tensor_name not in base.by_name or tensor_name not in abl.by_name \
            or base.by_name[tensor_name].store not in EDITABLE_STORES \
            or abl.by_name[tensor_name].store not in EDITABLE_STORES \
            or base.by_name[tensor_name].checksum \
            == abl.by_name[tensor_name].checksum:
        alt = _editable_pair_tensor(base, abl, tensor_name)
        if alt is None:
            raise ValueError(
                f"no usable tensor found in the pair: every candidate is "
                f"either undecodable or byte-identical between the two "
                f"files (zero dW -> no direction to extract). "
                f"{tensor_name} is store "
                f"{STORE_NAMES.get(base.by_name[tensor_name].store, '?') if tensor_name in base.by_name else 'missing'}"
                f" in the base file.")
        print(f"note: {tensor_name} is undecodable or byte-identical in the "
              f"pair; using {alt} instead")
        tensor_name = alt
    eb = base.by_name.get(tensor_name)
    ea = abl.by_name.get(tensor_name)
    if eb is None or ea is None:
        raise ValueError(f"{tensor_name} missing from one of the files")
    if eb.shape != ea.shape or eb.K != ea.K or eb.N != ea.N:
        raise ValueError(f"shape mismatch: {eb.shape} vs {ea.shape}")
    K = eb.K
    if K != RESIDUAL:
        print(f"warning: K={K}; direction will live in that space, not the "
              f"{RESIDUAL}-wide residual stream")
    bsides = ht_sides(base, eb) if eb.store == 16 else (None, None)
    asides = ht_sides(abl, ea) if ea.store == 16 else (None, None)
    cb = make_codec(eb.store, eb.variant, base.payload(eb), eb.N, K, *bsides)
    ca = make_codec(ea.store, ea.variant, abl.payload(ea), ea.N, K, *asides)
    G = np.zeros((K, K), np.float64)
    for a in range(0, eb.N, chunk_rows):
        b = min(a + chunk_rows, eb.N)
        d = ca.decode_rows(a, b) - cb.decode_rows(a, b)
        G += d.T @ d
        print(f"\r  gram: rows {b}/{eb.N}", end="", flush=True)
    print()
    trG = float(np.trace(G))
    if not np.isfinite(trG) or trG <= 0:
        raise ValueError(
            f"{tensor_name}: zero difference between base and abliterated "
            f"(trace of Gram = {trG}) — no direction to extract from this "
            f"tensor. Pick one the abliteration actually edited.")
    # top eigenvector of the Gram matrix = right singular vector of dW
    v = np.random.default_rng(0).standard_normal(K).astype(np.float64)
    v /= np.linalg.norm(v)
    for _ in range(500):
        prev = v
        v = G @ v
        v /= np.linalg.norm(v)
        if abs(float(v @ prev)) > 1 - 1e-12:
            break
    r = v.astype(np.float32)
    if K != RESIDUAL:
        raise ValueError("cannot write residual-space direction from this tensor")
    r = r / np.linalg.norm(r)
    eig = float(v @ (G @ v)) / max(float(v @ v), 1e-30)
    fro = float(np.sqrt(np.trace(G)))
    print(f"  top eigenvalue {eig:.3e}, sqrt(tr G) {fro:.3e} "
          f"(rank-1 dominance: {eig / max(fro * fro, 1e-30):.3f})")
    r.tofile(out_path)
    print(f"wrote direction ({K} f32) to {out_path}")
    base.close()
    abl.close()


# ---------------------------------------------------------------------------
# Inspect

def inspect(path, verify=False, name_filter=None):
    hgn = HgnFile(path)
    print(f"{path}")
    print(f"  version={hgn.version} identity={hgn.identity!r} "
          f"entries={len(hgn.entries)} size={hgn.size}")
    bad = 0
    for e in hgn.entries:
        if name_filter and not re.search(name_filter, e.name):
            continue
        line = (f"  {e.name:60s} {e.store_name():>6s} v{e.variant} "
                f"{str(list(e.shape)):>22s} size={e.size}")
        try:
            expected = size_rule(e.store, e.variant, e.N, e.K)
            if expected != e.size:
                line += f"  SIZE-RULE-FAIL expected {expected}"
        except ValueError:
            pass
        if verify:
            got = hgn.checksum(e)
            if got != e.checksum:
                line += f"  CHECKSUM-FAIL got 0x{got:08X}"
                bad += 1
        print(line)
    if verify:
        print(f"  checksum failures: {bad}")
    hgn.close()


# ---------------------------------------------------------------------------
# Direction derivation from a quantized pair

def derive_direction(base_path, donor_path, out_path, chunk_rows=4096):
    """Derive the refusal direction from a (base, already-abliterated) pair
    by aggregating the output-axis Gram of dW across every contract writer.

    The true edit is the same rank-1 r r^T in every writer, while the
    requantization noise between two independently encoded quantized files
    is independent per tensor. Summing dW dW^T over ~149 tensors therefore
    raises the rank-1 dominance by roughly 149x over any single tensor --
    a single quantized tensor's delta is noise-dominated (~0.03 dominance)
    and yields a garbage direction.
    """
    base = HgnFile(base_path)
    donor = HgnFile(donor_path)
    names = [n for n in contract_writers(base) if n in donor.by_name]
    G = np.zeros((RESIDUAL, RESIDUAL), np.float64)
    used = 0
    for name in names:
        eb, ed = base.by_name[name], donor.by_name[name]
        if eb.shape != ed.shape or eb.checksum == ed.checksum:
            continue
        if eb.store not in EDITABLE_STORES or ed.store not in EDITABLE_STORES:
            print(f"  skip {name}: store not decodable")
            continue
        if eb.store == 16 and not (eb.variant in (0x1208, 0x1206)
                                   and ed.variant in (0x1208, 0x1206)):
            print(f"  skip {name}: HT variant not decodable")
            continue
        side = _out_axis_side(eb)
        if side is None:
            print(f"  skip {name}: no residual output axis")
            continue
        cb = make_codec(eb.store, eb.variant, base.payload(eb), eb.N, eb.K,
                        *(ht_sides(base, eb) if eb.store == 16 else (None, None)))
        cd = make_codec(ed.store, ed.variant, donor.payload(ed), ed.N, ed.K,
                        *(ht_sides(donor, ed) if ed.store == 16 else (None, None)))
        step = RESIDUAL * max(1, chunk_rows // RESIDUAL) \
            if side == "experts" else chunk_rows
        for a in range(0, eb.N, step):
            b = min(a + step, eb.N)
            d = cd.decode_rows(a, b) - cb.decode_rows(a, b)
            if side == "last":
                G += d.T @ d
            elif side == "rows":
                G += d @ d.T
            else:
                W = d.reshape((b - a) // RESIDUAL, RESIDUAL, eb.K)
                G += np.einsum("eak,ebk->ab", W, W)
            del d
        used += 1
        print(f"\r  writers: {used}/{len(names)}  {name}", end="", flush=True)
    print()
    trG = float(np.trace(G))
    if not np.isfinite(trG) or trG <= 0:
        raise ValueError("zero difference across the whole writer contract")
    v = np.random.default_rng(0).standard_normal(RESIDUAL)
    v /= np.linalg.norm(v)
    for _ in range(1000):
        prev = v
        v = G @ v
        v /= np.linalg.norm(v)
        if abs(float(v @ prev)) > 1 - 1e-13:
            break
    eig = float(v @ (G @ v))
    print(f"  top eigenvalue {eig:.3e}, trace {trG:.3e} "
          f"(rank-1 dominance: {eig / trG:.3f})")
    r = v.astype(np.float32)
    r /= np.linalg.norm(r)
    if r[int(np.argmax(np.abs(r)))] < 0:
        r = -r
    r.tofile(out_path)
    print(f"wrote direction ({RESIDUAL} f32) to {out_path}")
    base.close()
    donor.close()


# ---------------------------------------------------------------------------
# CLI

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="hgn-abliterate",
        description="Abliterate halogen .hgn checkpoints (ht43 / Flash-Next).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="dump header and tensor table")
    p.add_argument("file")
    p.add_argument("--checksums", action="store_true", help="verify every payload")
    p.add_argument("--grep", help="only show names matching this regex")

    p = sub.add_parser("extract", help="estimate refusal direction from a pair")
    p.add_argument("--base", required=True, help="original .hgn")
    p.add_argument("--abliterated", required=True, help="already-abliterated .hgn")
    p.add_argument("--tensor", default="lm_head.weight")
    p.add_argument("--out", required=True, help="output .bin (f32) or .npy")

    p = sub.add_parser(
        "derive-direction",
        help="derive the refusal direction from a base/abliterated pair by "
             "noise-averaging dW over the whole writer contract (recommended "
             "when both files are quantized)")
    p.add_argument("--base", required=True, help="original .hgn")
    p.add_argument("--donor", required=True, help="already-abliterated .hgn")
    p.add_argument("--out", required=True, help="output .bin (f32)")
    p.add_argument("--chunk-rows", type=int, default=4096)

    p = sub.add_parser("abliterate", help="apply orthogonalization")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--direction", help="f32 binary or .npy, 2560 values")
    g.add_argument("--from-pair", nargs=2, metavar=("BASE", "ABLITERATED"),
                   help="extract the direction from a base/abliterated pair first")
    p.add_argument("--extract-tensor", default="lm_head.weight",
                   help="tensor used for --from-pair extraction")
    p.add_argument("--writers", choices=("readers", "contract"),
                   default="contract",
                   help="contract: ablate the output axis of the official "
                        "149-writer list (recommended); readers: legacy "
                        "input-axis ablation of residual-reading matrices")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="0..1, fraction of the refusal component to remove")
    p.add_argument("--layers", help='e.g. "20-47,mtp" (default: all)')
    p.add_argument("--only", help="regex: restrict target tensor names")
    p.add_argument("--skip", help="regex: exclude target tensor names")
    p.add_argument("--identity", help="override the identity string")
    p.add_argument("--no-verify", action="store_true",
                   help="skip input checksum verification")
    p.add_argument("--chunk-rows", type=int, default=4096)

    args = ap.parse_args(argv)
    if args.cmd == "inspect":
        inspect(args.file, args.checksums, args.grep)
    elif args.cmd == "extract":
        extract(args.base, args.abliterated, args.tensor, args.out)
    elif args.cmd == "derive-direction":
        derive_direction(args.base, args.donor, args.out,
                         chunk_rows=args.chunk_rows)
    elif args.cmd == "abliterate":
        direction = args.direction
        tmp = None
        if args.from_pair:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".f32", delete=False)
            tmp.close()
            direction = tmp.name
            if args.writers == "contract":
                derive_direction(args.from_pair[0], args.from_pair[1],
                                 direction, chunk_rows=args.chunk_rows)
            else:
                extract(args.from_pair[0], args.from_pair[1],
                        args.extract_tensor, direction)
        abliterate(
            args.input, args.output, direction,
            alpha=args.alpha,
            layer_spec=parse_layer_spec(args.layers) if args.layers else None,
            only=args.only, skip=args.skip, identity=args.identity,
            verify=not args.no_verify, chunk_rows=args.chunk_rows,
            writers=args.writers,
        )
        if tmp:
            os.unlink(tmp.name)


if __name__ == "__main__":
    main()
