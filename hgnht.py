"""Halogen v2 HT (16/0x1208): cyclic scalar trellis and H128 side planes.

The codebook and packing are recovered from the 0.15.0 gfx1151 decoder.
Native Viterbi is optional for decoding, required for production encoding.
"""
import ctypes
import functools
import os
from pathlib import Path

import numpy as np
from hgnenc import H128


@functools.lru_cache(maxsize=1)
def codebook():
    state = np.arange(65536, dtype=np.uint32)
    hashed = state * np.uint32(0x83DCD12D)
    sums = sum((hashed >> shift) & 255 for shift in (0, 8, 16, 24))
    values = (sums + 0x6400).astype('<u2').view('<f2').astype(np.float32)
    a, b = np.array([0x1EEE, 0xC931], dtype='<u2').view('<f2').astype(np.float32)
    # The product and sum are exactly representable in fp32 before one fp16
    # rounding, matching v_pk_fma_f16 (not two separate half operations).
    result = (values * a + b).astype(np.float16).astype(np.float32)
    result.flags.writeable = False
    return result


def _shape(shape):
    if len(shape) != 2 or any(n <= 0 or n % 128 for n in shape):
        raise ValueError('HT requires a positive [O,K] matrix with O and K divisible by 128')
    return shape


def _planes(shape, su, sv):
    o, k = _shape(shape)
    su, sv = np.asarray(su, np.float32), np.asarray(sv, np.float32)
    if su.shape != (k,) or sv.shape != (o,):
        raise ValueError('HT side-plane dimensions do not match the weights')
    if not np.all(np.abs(su) == 1) or not np.all(np.isfinite(sv)):
        raise ValueError('HT requires input signs +/-1 and finite signed row scales')
    return su, sv


def rotate(weights, su, sv):
    """Normalize rows before the two orthogonal block-Hadamard transforms."""
    o, k = _shape(weights.shape)
    su, sv = _planes(weights.shape, su, sv)
    # A zero scale forces that decoded row to zero. Its normalized row is
    # unconstrained; choose the minimum-norm inverse, not an arbitrary signal
    # that would contaminate other rows when trellis quantization adds error.
    x = np.asarray(weights, np.float32) / np.where(sv == 0, 1, sv)[:, None]
    x[sv == 0] = 0
    x *= su[None, :]
    x = (x.reshape(o, k // 128, 128) @ H128).reshape(o // 128, 128, k)
    return np.matmul(H128, x).reshape(o, k)


def unrotate(rotated, su, sv):
    o, k = _shape(rotated.shape)
    su, sv = _planes(rotated.shape, su, sv)
    x = np.matmul(H128, rotated.reshape(o // 128, 128, k)).reshape(o, k // 128, 128)
    return (x @ H128).reshape(o, k) * sv[:, None] * su[None, :]


def to_tiles(rotated):
    o, k = rotated.shape
    if o % 16 or k % 16:
        raise ValueError('HT trellis tiles must have both dimensions divisible by 16')
    return np.ascontiguousarray(
        rotated.reshape(o // 16, 16, k // 16, 2, 8).transpose(0, 2, 3, 1, 4).reshape(-1, 256),
        dtype=np.float32,
    )


def from_tiles(tiles, shape):
    o, k = shape
    return tiles.reshape(o // 16, k // 16, 2, 16, 8).transpose(0, 3, 1, 2, 4).reshape(o, k)


def decode_tiles(codes):
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[1] != 256 or np.any(codes > 15) or np.any(codes < 0):
        raise ValueError('HT codes must be [tiles,256] nibbles')
    codes = codes.astype(np.uint32)
    states = sum(np.roll(codes, shift, axis=1) << (4 * shift) for shift in range(4))
    return codebook()[states]


def pack(codes, shape):
    o, k = _shape(shape)
    codes = np.asarray(codes, np.uint32).reshape(o // 16, k // 16, 256)
    if np.any(codes > 15):
        raise ValueError('HT codes must be nibbles')
    # Disk tile order is [O/128, K/16, 8 output tiles], not row-major tiles.
    codes = codes.reshape(o // 128, 8, k // 16, 32, 8).transpose(0, 2, 1, 3, 4)
    words = np.bitwise_or.reduce(codes << np.arange(28, -1, -4, dtype=np.uint32), axis=-1)
    return words.astype('<u4').tobytes()


def unpack(payload, shape):
    o, k = _shape(shape)
    if len(payload) != o * k // 2:
        raise ValueError('HT payload length does not match its matrix shape')
    words = np.frombuffer(payload, '<u4').reshape(o // 128, k // 16, 8, 32)
    codes = ((words[..., None] >> np.arange(28, -1, -4, dtype=np.uint32)) & 15)
    return codes.transpose(0, 2, 1, 3, 4).reshape(-1, 256).astype(np.uint8)


def decode_rot(payload, shape):
    return from_tiles(decode_tiles(unpack(payload, shape)), shape)


def decode(payload, shape, su, sv):
    return unrotate(decode_rot(payload, shape), su, sv)


@functools.lru_cache(maxsize=1)
def native():
    path = os.environ.get('HGN_HT_LIBRARY', str(Path(__file__).with_name('ht_trellis.so')))
    try:
        lib = ctypes.CDLL(path)
    except OSError as exc:
        raise ValueError(
            'HT encoding needs ht_trellis.so: compile ht_trellis.cpp with '
            'c++ -O3 -march=native -fopenmp -shared -fPIC, then set HGN_HT_LIBRARY'
        ) from exc
    fp = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    qp = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
    lib.ht_encode.argtypes = [fp, fp, qp, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.ht_encode.restype = ctypes.c_int
    lib.ht_search_fixed.argtypes = [fp, fp, ctypes.c_int, ctypes.c_int, qp]
    lib.ht_search_fixed.restype = ctypes.c_float
    return lib


def encode_tiles(tiles, workers=8, beam=128):
    """Two Viterbi passes: infer a boundary, then enforce cyclic closure.

    beam=0 searches every state; beam>0 keeps that many histories (M-algorithm).
    Even the full-state mode fixes the boundary after pass one; it is not a
    proof of the global optimum over all 4096 cyclic boundaries.
    """
    tiles = np.ascontiguousarray(tiles, np.float32)
    if tiles.ndim != 2 or tiles.shape[1] != 256 or not np.all(np.isfinite(tiles)):
        raise ValueError('HT encoder expects finite [tiles,256] values')
    if not 0 <= beam <= 4096 or workers < 1:
        raise ValueError('HT beam must be 0..4096 and workers must be positive')
    codes = np.empty(tiles.shape, np.uint8)
    if native().ht_encode(tiles, codebook(), codes, len(tiles), workers, beam):
        raise ValueError('native HT encoding failed')
    return codes


def encode_rot(rotated, workers=8, beam=128):
    _shape(rotated.shape)
    return pack(encode_tiles(to_tiles(rotated), workers, beam), rotated.shape)


def encode(weights, su, sv, workers=8, beam=128):
    return encode_rot(rotate(weights, su, sv), workers, beam)


def search_fixed(values, boundary):
    """Small numpy reference: exact closed Viterbi for one fixed history."""
    values = np.asarray(values, np.float32)
    if values.ndim != 1 or not 3 <= len(values) <= 256 or not 0 <= boundary < 4096:
        raise ValueError('expected 3..256 values and a 12-bit boundary')
    costs = np.full(65536, np.inf, np.float32)
    costs[boundary::4096] = 0
    trace = np.empty((len(values), 4096), np.uint8)
    for i, value in enumerate(values):
        grouped = costs.reshape(16, 4096)
        trace[i] = grouped.argmin(axis=0)
        costs = np.repeat(grouped.min(axis=0), 16) + (codebook() - value) ** 2
    end = boundary + int(costs[boundary::4096].argmin()) * 4096
    loss = float(costs[end])
    codes = np.empty(len(values), np.uint8)
    for i in range(len(values) - 1, -1, -1):
        codes[i] = end & 15
        end = (end >> 4) | (int(trace[i, end >> 4]) << 12)
    return codes, loss
