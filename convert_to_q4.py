"""
convert_to_q4.py - turn a Hugging Face Llama folder (bf16 .safetensors) into one small 4-bit file
that engine.cpp can load.

Usage:
    pip install numpy
    python convert_to_q4.py models/llama-3.2-3b-instruct model.q4.bin

Why: the bf16 model is ~6.5 GB, which does not fit comfortably in 8 GB of RAM next to Windows.
After this step it is ~2.2 GB, and generation is also faster (less data to read per token).

Memory use stays below ~1 GB because every tensor is read and quantized a few thousand rows at a time.

Output file format ("L3Q1"), all little-endian:
  4 bytes  magic "L3Q1"
  9 x int32: dim, hidden(intermediate), n_layers, n_heads, n_kv_heads, head_dim, vocab, max_seq, group_size
  float32  rms_norm_eps
  float32[head_dim/2]  rope inverse frequencies (llama3 scaling already applied)
  token embedding   (8-bit): int8[vocab*dim], then float32 scales[vocab*dim/32]
  per layer: attn_norm f32[dim], ffn_norm f32[dim], then q,k,v,o,gate,up,down (4-bit):
             packed uint8[out*in/2], then float32 scales[out*in/32]
  final norm f32[dim]
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

GS = 32  # group size: 32 consecutive weights share one scale


# ----------------------------------------------------------------------------- reading safetensors
def index_safetensors(model_dir):
    """Map tensor name -> (file, dtype, shape, absolute byte offset). Works for sharded checkpoints."""
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        sys.exit(f"No .safetensors files in {model_dir}")
    tensors = {}
    for path in files:
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        base = 8 + n
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, _end = info["data_offsets"]
            tensors[name] = (path, info["dtype"], info["shape"], base + start)
    return tensors


def read_rows(tensors, name, r0, r1):
    """Rows r0:r1 of a tensor as float32. bf16 is converted by hand (numpy has no bfloat16)."""
    path, dtype, shape, off = tensors[name]
    if dtype == "BF16":
        a = np.memmap(path, dtype=np.uint16, mode="r", offset=off, shape=tuple(shape))
        part = np.ascontiguousarray(a[r0:r1])
        # bf16 is just the top 16 bits of a float32: shift them back up and reinterpret.
        return (part.astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        a = np.memmap(path, dtype=np.float16, mode="r", offset=off, shape=tuple(shape))
        return np.asarray(a[r0:r1], dtype=np.float32)
    if dtype == "F32":
        a = np.memmap(path, dtype=np.float32, mode="r", offset=off, shape=tuple(shape))
        return np.array(a[r0:r1], dtype=np.float32)
    sys.exit(f"Unsupported dtype {dtype} for {name}")


# ----------------------------------------------------------------------------- quantization
def quantize_q4(w):
    """w: float32 [rows, in].  Returns packed uint8 [rows, in/2] and float32 scales [rows, in/32].

    For every group of 32 weights: scale = max|w| / 7, q = round(w/scale) clipped to [-8, 7], stored as q+8 (0..15).
    Packing: byte j of the group = weight j (low nibble) | weight j+16 (high nibble) << 4.
    """
    rows, n = w.shape
    g = w.reshape(rows, n // GS, GS)
    amax = np.abs(g).max(axis=2, keepdims=True)
    scale = amax / 7.0
    safe = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.rint(g / safe), -8, 7).astype(np.int8) + 8
    q = q.astype(np.uint8)
    packed = q[:, :, :16] | (q[:, :, 16:] << 4)
    return packed.reshape(rows, n // 2), scale.reshape(rows, n // GS).astype(np.float32)


def quantize_q8(w):
    """Same idea with 8 bits: scale = max|w| / 127, q = round(w/scale) in [-127, 127]."""
    rows, n = w.shape
    g = w.reshape(rows, n // GS, GS)
    amax = np.abs(g).max(axis=2, keepdims=True)
    scale = amax / 127.0
    safe = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.rint(g / safe), -127, 127).astype(np.int8)
    return q.reshape(rows, n), scale.reshape(rows, n // GS).astype(np.float32)


def write_quantized(out, tensors, name, bits, chunk_elems=8_000_000):
    """Stream one weight matrix to disk: quantize a block of rows at a time to keep RAM low."""
    _path, _dt, shape, _off = tensors[name]
    rows, cols = shape
    if cols % GS:
        sys.exit(f"{name}: {cols} columns is not a multiple of {GS}")
    step = max(1, chunk_elems // cols)
    scales = []
    for r0 in range(0, rows, step):
        w = read_rows(tensors, name, r0, min(rows, r0 + step))
        q, s = quantize_q4(w) if bits == 4 else quantize_q8(w)
        out.write(q.tobytes())
        scales.append(s.reshape(-1))
    out.write(np.concatenate(scales).tobytes())  # all scales of this matrix come after all its integers


def write_f32(out, tensors, name):
    path, _dt, shape, _off = tensors[name]
    arr = read_rows(tensors, name, 0, shape[0])
    out.write(arr.astype(np.float32).tobytes())


# ----------------------------------------------------------------------------- RoPE frequencies
def rope_inv_freq(cfg, head_dim):
    """inv_freq[i] = theta^(-2i/head_dim), then Llama-3.x 'llama3' scaling for long context."""
    theta = float(cfg.get("rope_theta", 10000.0))
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    rs = cfg.get("rope_scaling")
    if rs:
        kind = rs.get("rope_type", rs.get("type"))
        if kind != "llama3":
            sys.exit(f"Unsupported rope_scaling type: {kind}")
        factor = float(rs["factor"])
        low = float(rs["low_freq_factor"])
        high = float(rs["high_freq_factor"])
        orig = float(rs["original_max_position_embeddings"])
        low_wl, high_wl = orig / low, orig / high
        wavelen = 2 * np.pi / inv
        new = np.where(wavelen > low_wl, inv / factor, inv)        # low frequencies: stretch by `factor`
        smooth = (orig / wavelen - low) / (high - low)
        smoothed = (1 - smooth) * inv / factor + smooth * inv      # in-between: blend
        medium = (wavelen >= high_wl) & (wavelen <= low_wl)
        inv = np.where(medium, smoothed, new)                      # high frequencies: unchanged
    return inv.astype(np.float32)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="folder with config.json and *.safetensors")
    ap.add_argument("out_file", help="output file, e.g. model.q4.bin")
    args = ap.parse_args()

    with open(os.path.join(args.model_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    dim = cfg["hidden_size"]
    hidden = cfg["intermediate_size"]
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    head_dim = cfg.get("head_dim") or dim // n_heads
    vocab = cfg["vocab_size"]
    max_seq = cfg["max_position_embeddings"]
    eps = float(cfg.get("rms_norm_eps", 1e-5))
    print(f"dim={dim} hidden={hidden} layers={n_layers} heads={n_heads} kv_heads={n_kv} "
          f"head_dim={head_dim} vocab={vocab} max_seq={max_seq}")

    t = index_safetensors(args.model_dir)
    if "lm_head.weight" in t and not cfg.get("tie_word_embeddings", True):
        sys.exit("This model has a separate lm_head (untied embeddings); this simple engine only supports tied ones "
                 "(Llama 3.2 1B/3B are tied).")

    with open(args.out_file, "wb") as out:
        out.write(b"L3Q1")
        out.write(struct.pack("<9i", dim, hidden, n_layers, n_heads, n_kv, head_dim, vocab, max_seq, GS))
        out.write(struct.pack("<f", eps))
        out.write(rope_inv_freq(cfg, head_dim).tobytes())

        print("embedding (8-bit) ...")
        write_quantized(out, t, "model.embed_tokens.weight", 8)

        for i in range(n_layers):
            p = f"model.layers.{i}."
            write_f32(out, t, p + "input_layernorm.weight")
            write_f32(out, t, p + "post_attention_layernorm.weight")
            for nm in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                       "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
                write_quantized(out, t, p + nm + ".weight", 4)
            print(f"layer {i + 1}/{n_layers} done", end="\r", flush=True)

        write_f32(out, t, "model.norm.weight")

    size = os.path.getsize(args.out_file)
    print(f"\nWrote {args.out_file}: {size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
