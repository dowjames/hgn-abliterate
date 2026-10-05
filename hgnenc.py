"""Encoders/decoders for the HGN stores needed to re-encode the abliteration writers.

q4c v2 and q8g64 follow jtsylve/hgn-spec 1.0.1 STORAGE.md. I4R (store 23, variant
128) is reverse-engineered from qwen38-flash-next-v2.hgn and verified against the
base BF16 weights: per expert, codes [O][K/2] (low nibble first) for all experts,
then f16 scales [O][K/128] for all experts; value (q - 8) * scale in the rotated
basis Hk(Ho(diag(svh) W diag(suh))), H = normalized Sylvester Hadamard of 128.
"""
import json, os, struct
import numpy as np

# ---------------------------------------------------------------- BF16 source
class SafeTensors:
    """Validated, indexed safetensors access with row and byte streaming."""
    _DTYPE_BYTES = {
        "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
        "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
        "U32": 4, "I32": 4, "F32": 4,
        "U64": 8, "I64": 8, "F64": 8,
    }

    def __init__(self, d):
        self.d = os.path.realpath(os.fspath(d))
        index_path = os.path.join(self.d, "model.safetensors.index.json")
        if not os.path.isfile(index_path):
            raise ValueError(f"missing indexed safetensors file: {index_path}")
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        wm = index.get("weight_map")
        if not isinstance(wm, dict) or not wm:
            raise ValueError(f"invalid weight_map in {index_path}")
        indexed = {}
        for name, shard in wm.items():
            if not isinstance(name, str) or not name or not isinstance(shard, str):
                raise ValueError(f"invalid weight_map entry in {index_path}: {name!r}")
            path = os.path.realpath(os.path.join(self.d, shard))
            if os.path.commonpath((self.d, path)) != self.d:
                raise ValueError(f"safetensors shard escapes input directory: {shard}")
            indexed.setdefault(path, set()).add(name)

        self.loc = {}
        for path, expected_names in sorted(indexed.items()):
            if not os.path.isfile(path):
                raise ValueError(f"missing safetensors shard: {path}")
            with open(path, "rb") as fh:
                prefix = fh.read(8)
                if len(prefix) != 8:
                    raise ValueError(f"truncated safetensors header: {path}")
                header_size = struct.unpack("<Q", prefix)[0]
                file_size = os.fstat(fh.fileno()).st_size
                data_start = 8 + header_size
                if data_start > file_size:
                    raise ValueError(f"truncated safetensors header: {path}")
                try:
                    header = json.loads(fh.read(header_size))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid safetensors header in {path}: {exc}") from exc
            actual_names = {k for k in header if k != "__metadata__"}
            if actual_names != expected_names:
                missing = sorted(expected_names - actual_names)
                extra = sorted(actual_names - expected_names)
                raise ValueError(
                    f"safetensors index/header mismatch in {path}: "
                    f"missing={missing[:3]} extra={extra[:3]}"
                )
            for name in sorted(actual_names):
                spec = header[name]
                try:
                    shape = tuple(int(n) for n in spec["shape"])
                    dtype = spec["dtype"]
                    start, end = (int(x) for x in spec["data_offsets"])
                    unit = self._DTYPE_BYTES[dtype]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid tensor metadata for {name} in {path}") from exc
                if any(n < 0 for n in shape) or start < 0 or end < start:
                    raise ValueError(f"invalid shape or offsets for {name} in {path}")
                elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
                if end - start != elements * unit or data_start + end > file_size:
                    raise ValueError(f"invalid data range for {name} in {path}")
                if name in self.loc:
                    raise ValueError(f"duplicate tensor in safetensors shards: {name}")
                self.loc[name] = (path, data_start + start, shape, dtype)
        if set(self.loc) != set(wm):
            raise ValueError(f"incomplete safetensors index in {index_path}")

    def raw(self, name, lo=None, hi=None):
        """Read a tensor as uint16; optional first-axis slice [lo:hi]."""
        p, off, shape, dt = self.loc[name]
        if dt != "BF16":
            raise ValueError(f"{name}: expected BF16, found {dt}")
        first = shape[0] if shape else 1
        per = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        lo = 0 if lo is None else lo
        hi = first if hi is None else hi
        if not 0 <= lo <= hi <= first:
            raise ValueError(f"{name}: invalid row slice [{lo}:{hi}] for {shape}")
        return np.fromfile(
            p, dtype="<u2", count=(hi - lo) * per, offset=off + lo * per * 2
        ).reshape(([hi - lo] + list(shape[1:])) if shape else ())

    def f32(self, name, lo=None, hi=None):
        return (self.raw(name, lo, hi).astype(np.uint32) << 16).view(np.float32)

    def read_bytes(self, name, start=0, count=None):
        """Read a byte range from one tensor without materializing its shard."""
        p, off, shape, dtype = self.loc[name]
        size = int(np.prod(shape, dtype=np.int64) if shape else 1) * self._DTYPE_BYTES[dtype]
        count = size - start if count is None else count
        if start < 0 or count < 0 or start + count > size:
            raise ValueError(f"{name}: invalid byte range [{start}:{start + count}]")
        with open(p, "rb") as fh:
            fh.seek(off + start)
            data = fh.read(count)
        if len(data) != count:
            raise ValueError(f"{name}: truncated tensor data")
        return data

    def equal(self, other, name, chunk_bytes=1 << 20):
        """Compare identical-shaped tensors bytewise, stopping at first mismatch."""
        a = self.loc[name]
        b = other.loc[name]
        if a[2:] != b[2:]:
            raise ValueError(f"{name}: safetensors shape/dtype mismatch {a[2:]} != {b[2:]}")
        p1, off1, shape, dtype = a
        p2, off2, _, _ = b
        remaining = int(np.prod(shape, dtype=np.int64) if shape else 1) * self._DTYPE_BYTES[dtype]
        read = 0
        with open(p1, "rb") as f1, open(p2, "rb") as f2:
            f1.seek(off1)
            f2.seek(off2)
            while remaining:
                n = min(chunk_bytes, remaining)
                x, y = f1.read(n), f2.read(n)
                if len(x) != n or len(y) != n:
                    raise ValueError(f"{name}: truncated tensor data")
                read += n
                if x != y:
                    return False, read
                remaining -= n
        return True, read

def f16(a):
    return np.asarray(a, np.float32).astype(np.float16)


# ---------------------------------------------------------------- q4c v2
def q4c_decode(buf, shape):
    N, K = int(np.prod(shape[:-1])), shape[-1]
    b = np.frombuffer(buf, np.uint8)
    cb = b[:64].view("<f4")
    codes = b[64:64 + N * K // 2].reshape(N, K // 2)
    S = 64 + (N * K // 2 + 63) // 64 * 64
    R = (K // 16 + 15) // 16 * 16
    sc = b[S:S + N * R].reshape(N, R)[:, :K // 16].copy().view("<f2").astype(np.float32)
    q = np.stack([codes & 15, codes >> 4], -1).reshape(N, K)
    return (cb[q] * np.repeat(sc, 32, 1)).reshape(shape)


def q4c_encode(W, cb, mults=(1.0, 0.95, 0.9, 0.85, 0.8), chunk=8192):
    """W: [..., K] float32. cb: 16 sorted f32 levels. Per group of 32, try scales
    absmax/max|cb| * m and keep the least squared error with nearest-level codes."""
    shape = W.shape
    K = shape[-1]
    W = W.reshape(-1, K)
    N = W.shape[0]
    cb = np.asarray(cb, np.float32)
    mids = (cb[1:] + cb[:-1]) / 2
    cmax = np.abs(cb).max()
    codes = np.empty((N, K), np.uint8)
    scales = np.empty((N, K // 32), np.float16)
    for r0 in range(0, N, chunk):
        g = W[r0:r0 + chunk].reshape(-1, K // 32, 32)
        am = np.abs(g).max(-1)
        best_e = best_q = best_s = None
        for m in mults:
            s = f16(am / cmax * m)
            sf = s.astype(np.float32)
            safe = np.where(sf == 0, 1, sf)
            q = np.searchsorted(mids, g / safe[..., None]).astype(np.uint8)
            e = ((cb[q] * sf[..., None] - g) ** 2).sum(-1)
            if best_e is None:
                best_e, best_q, best_s = e, q, s
            else:
                w = e < best_e
                best_e = np.where(w, e, best_e)
                best_q = np.where(w[..., None], q, best_q)
                best_s = np.where(w, s, best_s)
        codes[r0:r0 + chunk] = best_q.reshape(-1, K)
        scales[r0:r0 + chunk] = best_s
    R = (K // 16 + 15) // 16 * 16
    cplane = (codes[:, 0::2] | (codes[:, 1::2] << 4)).tobytes()
    splane = np.zeros((N, R), np.uint8)
    splane[:, :K // 16] = scales.view(np.uint8).reshape(N, K // 16)
    pad = (-len(cplane)) % 64
    return cb.astype("<f4").tobytes() + cplane + b"\0" * pad + splane.tobytes()


# ---------------------------------------------------------------- q8g64
def q8g64_decode(buf, shape):
    N, K = int(np.prod(shape[:-1])), shape[-1]
    b = np.frombuffer(buf, np.uint8).reshape(N, K + K // 16)
    rec = b[:, K:].copy().view("<f2").astype(np.float32).reshape(N, K // 64, 2)
    q = b[:, :K].astype(np.float32).reshape(N, K // 64, 64)
    return (q * rec[..., :1] + rec[..., 1:]).reshape(shape)


def q8g64_encode(W):
    shape = W.shape
    K = shape[-1]
    g = W.reshape(-1, K // 64, 64)
    lo, hi = g.min(-1), g.max(-1)
    s = f16((hi - lo) / 255)
    b = f16(lo)
    sf, bf = s.astype(np.float32), b.astype(np.float32)
    q = np.clip(np.round((g - bf[..., None]) / np.where(sf == 0, 1, sf)[..., None]), 0, 255).astype(np.uint8)
    rec = np.stack([s, b], -1).view(np.uint8).reshape(g.shape[0], K // 16)
    return np.concatenate([q.reshape(-1, K), rec], 1).tobytes()


# ---------------------------------------------------------------- I4R
def _had(n):
    H = np.array([[1.0]], np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return (H / np.sqrt(n)).astype(np.float32)


H128 = _had(128)


def i4r_rotate(W, su, sv):
    """W: [E, O, K] -> rotated [E, O, K]: Hadamard-128 on K blocks, then on O blocks."""
    E, O, K = W.shape
    X = W * sv[None, :, None] * su[None, None, :]
    X = (X.reshape(E, O, K // 128, 128) @ H128.T).reshape(E, O // 128, 128, K)
    return np.matmul(H128, X).reshape(E, O, K)


def i4r_unrotate(X, su, sv):
    E, O, K = X.shape
    Y = np.matmul(H128.T, X.reshape(E, O // 128, 128, K)).reshape(E, O, K // 128, 128)
    Y = (Y @ H128).reshape(E, O, K)
    return Y * sv[None, :, None] * su[None, None, :]


def i4r_split(buf, E, O, K):
    b = np.frombuffer(buf, np.uint8)
    nc = E * O * K // 2
    return b[:nc].reshape(E, O, K // 2), b[nc:].view("<f2").reshape(E, O, K // 128)


def i4r_decode_rot(codes, scales):
    E, O, K2 = codes.shape
    q = np.stack([codes & 15, codes >> 4], -1).reshape(E, O, K2 * 2).astype(np.float32) - 8
    return (q.reshape(E, O, -1, 128) * scales.astype(np.float32)[..., None]).reshape(E, O, K2 * 2)


def i4r_encode_rot(X, mults=(1.0, 0.97, 0.94, 0.91, 0.88, 0.85)):
    """X rotated [E, O, K]. Signed scale s = -x_ext/8 * m, so the group's extreme
    maps to code 0 (value -8 s); pick m with least squared error."""
    E, O, K = X.shape
    g = X.reshape(E, O, K // 128, 128)
    idx = np.abs(g).argmax(-1)
    ext = np.take_along_axis(g, idx[..., None], -1)[..., 0]
    best_e = best_q = best_s = None
    for m in mults:
        s = f16(-ext / 8 * m)
        sf = s.astype(np.float32)
        q = np.clip(np.round(g / np.where(sf == 0, 1, sf)[..., None]) + 8, 0, 15)
        e = (((q - 8) * sf[..., None] - g) ** 2).sum(-1)
        if best_e is None:
            best_e, best_q, best_s = e, q, s
        else:
            w = e < best_e
            best_e = np.where(w, e, best_e)
            best_q = np.where(w[..., None], q, best_q)
            best_s = np.where(w, s, best_s)
    q = best_q.reshape(E, O, K).astype(np.uint8)
    return (q[..., 0::2] | (q[..., 1::2] << 4)), best_s
