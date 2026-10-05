# hgn-abliterate

Abliterate halogen `.hgn` checkpoints — `w4b`, `v2`, and `ht43`
(`qwen38-flash-next-ht43.hgn`) — without leaving the HGN container: no GGUF
detour, no torch, no GPU.

**ht43 is not a new container.** It is a checkpoint *variant* of the same
HGN format (per the halogen-flash-server README/QUANT.md: experts stored
in fewer bits, ~53.7 GiB vs 62.1). What changed is the store mix — the
newer checkpoints use `q6k` for the 6-bit mixing layers and `q4a`/`q5a`
affine codes for experts. This tool now decodes and re-encodes every store
the Flash-Next checkpoint family uses: `bf16`, `f32`, `q4c` v2, `q8g64`,
`q8s32`, `q6k`, `q4a` (v0/v1), `q5a` (v0/v1).

Implements refusal-direction orthogonalization (Arditi et al. 2024,
*Refusal in Language Models Is Mediated by a Single Direction*) directly on
HGN weight files, following the clean-room spec
[jtsylve/hgn-spec](https://github.com/jtsylve/hgn-spec) v1.0.1 (copies in
`spec/`).

For a unit refusal direction `r` in residual-stream space, every linear
weight matrix `W` that **reads** the residual stream is updated per row as

```
w <- w - alpha * (w . r) * r
```

so the model stops reading the refusal direction. Quantized tensors
(`q4c` v2, `q8g64`, `q8s32`, `bf16`, `f32`) are dequantized row-block by
row-block, transformed, and requantized to the **same store code, variant,
row layout, and (for q4c) codebook**, so the output stays byte-layout
compatible with halogen-flash. All other tensors are copied verbatim;
checksums and size rules are recomputed per the spec's writer procedure.

## Requirements

Python 3.10+ with numpy. Nothing else.

## Usage

```bash
# 1. Inspect a checkpoint (verify integrity with --checksums)
python3 hgn_abliterate.py inspect qwen38-flash-next-w4b.hgn --checksums

# 2a. Abliterate with a known refusal direction (raw f32 binary or .npy,
#     2560 values, residual-stream space)
python3 hgn_abliterate.py abliterate in.hgn -o out-abliterated.hgn \
    --direction refusal_dir.f32

# 2b. Or bootstrap from an existing abliterated pair (e.g. Quat3rnion's v2):
#     dW = W_abl - W_base is rank-1 (=- (W r) r^T), so the top right
#     singular vector of dW is the direction.
python3 hgn_abliterate.py abliterate in.hgn -o out.hgn \
    --from-pair official.hgn quat3rnion-abliterated-v2.hgn \
    --extract-tensor lm_head.weight

# Standalone direction extraction
python3 hgn_abliterate.py extract --base official.hgn \
    --abliterated v2.hgn --tensor lm_head.weight --out dir.f32
```

Useful flags on `abliterate`:

- `--alpha 0.0..1.0` — fraction of the refusal component to remove
  (default 1.0 = full orthogonalization; lower = gentler, less capability
  damage).
- `--layers "20-47,mtp"` — restrict to a layer range (default: all trunk
  layers + MTP head). `lm_head` and the global mixer are always included.
- `--only REGEX` / `--skip REGEX` — restrict/exclude target tensor names.
- `--identity NAME` — override the 64-byte checkpoint identity string.
- `--no-verify` — skip input checksum verification (faster on huge files).
- `--chunk-rows N` — rows per dequant/transform/requant block (default
  4096; lower it if RAM is tight — peak RAM is roughly
  `chunk_rows × K × 4` bytes per tensor).

## What gets edited

Every matrix whose input axis is the residual stream (K=2560) or the
4-stream hyper-connection bundle (K=10240, direction tiled ×4 and
renormalized): `q/k/v_proj`, the indexer `index_qk_proj`, DeltaNet
`in_proj_{qkv,z,a,b}`, MLP `gate_proj/up_proj/gate/shared_expert_gate`,
fused `experts.gate_up_proj`, hyper-connection `input_mix_weight_down` and
`block_inject_weight`, `ple.key_proj`, `mtp.fc_hidden`, and `lm_head`.
`embed_tokens`, the n-gram table, norms, and everything writing *into* the
residual stream (`o_proj`, `down_proj`) are left untouched.

## Store 16: the v2/ht43 trunk (HT trellis — supported)

The v2 and ht43 checkpoints store nearly every residual-reading matrix
(409 tensors: `q/k/v/o_proj`, `in_proj_*`, experts, shared experts,
indexer, `lm_head`) in **store code 16**, which is not defined in
hgn-spec v1.0.1 (the spec describes the 0.13.5 reader; store 16 arrived
with the 0.15/v2 checkpoint). Quat3rnion's v2 build tooling
(`tooling/hgnht.py` + `ht_trellis.cpp` in the abliterated model repo)
documents it as **HT: a cyclic scalar trellis with H128 rotations**:

- Variant `0x1208` (4616): pure 4-bit code plane, `N*K/2` bytes. Each
  value's codebook entry is indexed by a 16-bit state = the last four
  nibbles (cyclic within each 256-value tile); the 65536-entry codebook
  is generated from a hash (`state * 0x83DCD12D`, byte-summed, then an
  fp16 FMA). Tiles are 16x16 blocks in `[O/128, K/16, 8]` disk order.
- Before quantization the weights are normalized by the row scales and
  rotated by two block-Hadamard passes (H128 along K blocks, then along
  O blocks). Side planes live in companion f16 (store 2) tensors:
  `<name>.svh` (N signed row scales, `|svh|` ~ 0.02 x row norms) and
  `<name>.suh` (K values, exactly +/-1). They are kept verbatim during
  abliteration; only the code plane is requantized.
- Variant `0x1206` (4614, 3-bit experts in ht43) has no public decoder —
  those tensors are skipped with a warning, not corrupted.

Decoding is pure numpy (`hgnht.py`, vendored here). Re-encoding needs the
native Viterbi encoder: build it once with
`c++ -O3 -march=native -fopenmp -shared -fPIC ht_trellis.cpp -o ht_trellis.so`
(already built next to `hgnht.py` in this directory; `HGN_HT_LIBRARY`
overrides the path). Validated against real data: decoding ht43
`k_proj` matches the w4b q4c oracle at corr 0.992, and re-encoding the
decoded weights reproduces 98.1% of the original codes.

Note: HT requantization is Viterbi-optimal in the rotated basis, so the
abliteration residual along `r` after requantizing is smaller than for
q4c (~0.001 vs ~0.01 of the base projection). Encoding is compute-heavy
(beam-128 Viterbi, 8 threads): budget roughly an hour or two for a full
v2/ht43 checkpoint.

## Caveats

- **Requant noise floor.** Requantizing to 4-bit re-adds a small component
  along `r` (measured ~0.01 mean on NF4-quality codebooks vs a 0.16 signal).
  This is inherent to editing quantized weights, not a bug; `--alpha 0.9`
  sometimes lands better than 1.0.
- **Direction quality matters more than anything here.** If you can compute
  `r` properly (PCA of harmful-vs-benign residuals on the fp16 source),
  do that and pass `--direction`. `--from-pair` is a bootstrap for when you
  only have quantized files.
- Stores this tool cannot re-encode (`fp8r`, `i4l` — 27B-only formats) are
  skipped with a warning, not corrupted. `fp8g`/`iq4nl` (n-gram table) and
  `i64` (PLE indices) are never abliteration targets and are copied
  verbatim.
- Expect ~10–20 min and a few GB of peak RAM for a full w4b checkpoint
  (the 51 GB n-gram table is copied verbatim, never decoded).
- Output is written atomically-ish: payloads first, header+table last, then
  fsync — a crash mid-write leaves a file that fails the magic/size checks
  rather than a silently wrong one.

## Tests

```bash
python3 test_hgn_abliterate.py
```

Builds a synthetic HGN file (one tensor per editable store + pass-through
tensors), abliterates with a known direction carrying a strong refusal
signal, and verifies: container checksums + size rules, byte-identical
pass-through, collapse of the refusal component, and direction recovery
from a (base, abliterated) pair (|cos| > 0.99).

## Spec notes

The container rules, store layouts, size rules and checksum come from the
clean-room [hgn-spec](https://github.com/jtsylve/hgn-spec) v1.0.1 (copies
in `spec/`). The tool reads the identity from the header and validates
every tensor against the spec's size rules before touching it, so an
unknown checkpoint shape fails loudly instead of writing garbage.
