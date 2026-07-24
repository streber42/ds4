#!/usr/bin/env python3
"""Generate a small, real, loadable DeepSeek-V4 Mini Flash GGUF fixture for test harness validation.

Creates a small DeepSeek-architecture GGUF file with 4 layers, 16 experts, hidden dim 512,
and 1000 vocab items that fits in ~20 MB VRAM and completes forward passes instantly.
"""

import os
import struct
import sys
from pathlib import Path

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10

GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q8_0 = 8

def pack_str(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw

def pack_u32(v: int) -> bytes:
    return struct.pack("<I", v)

def pack_u64(v: int) -> bytes:
    return struct.pack("<Q", v)

def pack_f32(v: float) -> bytes:
    return struct.pack("<f", v)

def pack_bool(v: bool) -> bytes:
    return struct.pack("?", v)

def pack_kv_u32(key: str, val: int) -> bytes:
    return pack_str(key) + pack_u32(GGUF_TYPE_UINT32) + pack_u32(val)

def pack_kv_f32(key: str, val: float) -> bytes:
    return pack_str(key) + pack_u32(GGUF_TYPE_FLOAT32) + pack_f32(val)

def pack_kv_bool(key: str, val: bool) -> bytes:
    return pack_str(key) + pack_u32(GGUF_TYPE_BOOL) + pack_bool(val)

def pack_kv_str(key: str, val: str) -> bytes:
    return pack_str(key) + pack_u32(GGUF_TYPE_STRING) + pack_str(val)

def pack_kv_array_u32(key: str, vals: list[int]) -> bytes:
    b = pack_str(key) + pack_u32(GGUF_TYPE_ARRAY) + pack_u32(GGUF_TYPE_UINT32) + pack_u64(len(vals))
    for v in vals:
        b += pack_u32(v)
    return b

def pack_kv_array_f32(key: str, vals: list[float]) -> bytes:
    b = pack_str(key) + pack_u32(GGUF_TYPE_ARRAY) + pack_u32(GGUF_TYPE_FLOAT32) + pack_u64(len(vals))
    for v in vals:
        b += pack_f32(v)
    return b

def pack_kv_array_str(key: str, vals: list[str]) -> bytes:
    b = pack_str(key) + pack_u32(GGUF_TYPE_ARRAY) + pack_u32(GGUF_TYPE_STRING) + pack_u64(len(vals))
    for v in vals:
        b += pack_str(v)
    return b

def calc_tensor_bytes(dims: tuple[int, ...], ggml_type: int) -> int:
    n_elems = 1
    for d in dims:
        n_elems *= d
    if ggml_type == GGML_TYPE_F32:
        return n_elems * 4
    elif ggml_type == GGML_TYPE_F16:
        return n_elems * 2
    elif ggml_type == GGML_TYPE_Q8_0:
        assert n_elems % 32 == 0
        return (n_elems // 32) * 34
    else:
        raise ValueError(f"unsupported ggml_type {ggml_type}")

def generate_tensor_data(n_bytes: int, ggml_type: int) -> bytes:
    if ggml_type == GGML_TYPE_F32:
        # Fill with small deterministic float values (e.g. 0.01)
        count = n_bytes // 4
        return struct.pack(f"<{count}f", *[0.01] * count)
    elif ggml_type == GGML_TYPE_F16:
        # 0.01 in float16 is 0x2440
        count = n_bytes // 2
        return struct.pack(f"<{count}H", *[0x2440] * count)
    elif ggml_type == GGML_TYPE_Q8_0:
        # Q8_0 block: 2 bytes scale (float16 1.0 = 0x3c00), 32 int8 values
        n_blocks = n_bytes // 34
        block = struct.pack("<H", 0x3c00) + b"\x01" * 32
        return block * n_blocks
    else:
        return b"\x00" * n_bytes

def pad_to(size: int, align: int) -> int:
    return ((size + align - 1) // align) * align

def main():
    out_path = Path("tests/mini_ds4flash.gguf")
    if len(sys.argv) > 1:
        out_path = Path(sys.argv[1])

    print(f"Generating DeepSeek-V4 Mini Flash GGUF fixture at {out_path}...")

    # Mini DeepSeek-V4 Flash parameters
    n_layer = 4
    n_embd = 512
    n_vocab = 1000
    n_head = 8
    n_head_kv = 1
    n_head_dim = 64
    n_value_dim = 64
    n_rot = 64
    n_out_group = 1
    n_lora_q = 128
    n_lora_o = 128
    n_expert = 16
    n_expert_used = 2
    n_expert_shared = 1
    n_ff_exp = 256
    n_hash_layer = 0
    n_swa = 128
    n_indexer_head = 8
    n_indexer_head_dim = 32
    n_indexer_top_k = 8
    n_hc = 4
    n_hc_sinkhorn_iter = 20

    # Build Vocabulary
    tokens = [
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "<think>",
        "</think>",
        "｜DSML｜",
    ]
    # Fill remaining tokens up to n_vocab
    for i in range(len(tokens), n_vocab):
        tokens.append(f"tok_{i}")

    merges = ["a b", "c d"]

    # Build Metadata KV list
    kv_items = [
        pack_kv_str("general.architecture", "deepseek4"),
        pack_kv_u32("deepseek4.block_count", n_layer),
        pack_kv_u32("deepseek4.embedding_length", n_embd),
        pack_kv_u32("deepseek4.vocab_size", n_vocab),
        pack_kv_u32("deepseek4.attention.head_count", n_head),
        pack_kv_u32("deepseek4.attention.head_count_kv", n_head_kv),
        pack_kv_u32("deepseek4.attention.key_length", n_head_dim),
        pack_kv_u32("deepseek4.attention.value_length", n_value_dim),
        pack_kv_u32("deepseek4.rope.dimension_count", n_rot),
        pack_kv_u32("deepseek4.attention.q_lora_rank", n_lora_q),
        pack_kv_u32("deepseek4.attention.output_lora_rank", n_lora_o),
        pack_kv_u32("deepseek4.attention.output_group_count", n_out_group),
        pack_kv_u32("deepseek4.expert_count", n_expert),
        pack_kv_u32("deepseek4.expert_used_count", n_expert_used),
        pack_kv_u32("deepseek4.expert_feed_forward_length", n_ff_exp),
        pack_kv_u32("deepseek4.expert_shared_count", n_expert_shared),
        pack_kv_u32("deepseek4.hash_layer_count", n_hash_layer),
        pack_kv_u32("deepseek4.attention.sliding_window", n_swa),
        pack_kv_u32("deepseek4.attention.indexer.head_count", n_indexer_head),
        pack_kv_u32("deepseek4.attention.indexer.key_length", n_indexer_head_dim),
        pack_kv_u32("deepseek4.attention.indexer.top_k", n_indexer_top_k),
        pack_kv_u32("deepseek4.hyper_connection.count", n_hc),
        pack_kv_u32("deepseek4.hyper_connection.sinkhorn_iterations", n_hc_sinkhorn_iter),
        pack_kv_f32("deepseek4.rope.freq_base", 10000.0),
        pack_kv_f32("deepseek4.attention.compress_rope_freq_base", 160000.0),
        pack_kv_f32("deepseek4.expert_weights_scale", 1.5),
        pack_kv_f32("deepseek4.attention.layer_norm_rms_epsilon", 1e-6),
        pack_kv_f32("deepseek4.hyper_connection.epsilon", 1e-6),
        pack_kv_bool("deepseek4.expert_weights_norm", True),
        pack_kv_array_u32("deepseek4.attention.compress_ratios", [0, 0, 4, 128]),
        pack_kv_array_f32("deepseek4.swiglu_clamp_exp", [10.0, 10.0, 10.0, 10.0]),
        pack_kv_array_str("tokenizer.ggml.tokens", tokens),
        pack_kv_array_str("tokenizer.ggml.merges", merges),
    ]

    kv_blob = b"".join(kv_items)

    # Tensor definitions: list of (name, dims, ggml_type)
    hc_dim = n_embd * n_hc
    hc_mix_dim = 2 * n_hc + n_hc * n_hc
    q_dim = n_head * n_head_dim
    out_low_dim = n_out_group * n_lora_o

    tensors_def = [
        ("token_embd.weight", (n_embd, n_vocab), GGML_TYPE_F16),
        ("output_norm.weight", (n_embd,), GGML_TYPE_F32),
        ("output.weight", (n_embd, n_vocab), GGML_TYPE_Q8_0),
        ("output_hc_base.weight", (n_hc,), GGML_TYPE_F32),
        ("output_hc_fn.weight", (hc_dim, n_hc), GGML_TYPE_F16),
        ("output_hc_scale.weight", (1,), GGML_TYPE_F32),
    ]

    for il in range(n_layer):
        ratio = [0, 0, 4, 128][il]
        tensors_def.extend([
            (f"blk.{il}.hc_attn_fn.weight", (hc_dim, hc_mix_dim), GGML_TYPE_F16),
            (f"blk.{il}.hc_attn_scale.weight", (3,), GGML_TYPE_F32),
            (f"blk.{il}.hc_attn_base.weight", (hc_mix_dim,), GGML_TYPE_F32),
            (f"blk.{il}.attn_norm.weight", (n_embd,), GGML_TYPE_F32),
            (f"blk.{il}.attn_q_a.weight", (n_embd, n_lora_q), GGML_TYPE_Q8_0),
            (f"blk.{il}.attn_q_a_norm.weight", (n_lora_q,), GGML_TYPE_F32),
            (f"blk.{il}.attn_q_b.weight", (n_lora_q, q_dim), GGML_TYPE_Q8_0),
            (f"blk.{il}.attn_kv.weight", (n_embd, n_head_dim), GGML_TYPE_Q8_0),
            (f"blk.{il}.attn_kv_a_norm.weight", (n_head_dim,), GGML_TYPE_F32),
            (f"blk.{il}.attn_sinks.weight", (n_head,), GGML_TYPE_F32),
            (f"blk.{il}.attn_output_a.weight", (n_head_dim * (n_head // n_out_group), out_low_dim), GGML_TYPE_Q8_0),
            (f"blk.{il}.attn_output_b.weight", (out_low_dim, n_embd), GGML_TYPE_Q8_0),
        ])

        if ratio != 0:
            coff = 2 if ratio == 4 else 1
            comp_width = coff * n_head_dim
            tensors_def.extend([
                (f"blk.{il}.attn_compressor_ape.weight", (comp_width, ratio), GGML_TYPE_F16),
                (f"blk.{il}.attn_compressor_kv.weight", (n_embd, comp_width), GGML_TYPE_F16),
                (f"blk.{il}.attn_compressor_gate.weight", (n_embd, comp_width), GGML_TYPE_F16),
                (f"blk.{il}.attn_compressor_norm.weight", (n_head_dim,), GGML_TYPE_F32),
            ])
        if ratio == 4:
            index_q_dim = n_indexer_head * n_indexer_head_dim
            index_width = 2 * n_indexer_head_dim
            tensors_def.extend([
                (f"blk.{il}.indexer.attn_q_b.weight", (n_lora_q, index_q_dim), GGML_TYPE_Q8_0),
                (f"blk.{il}.indexer.proj.weight", (n_embd, n_indexer_head), GGML_TYPE_F16),
                (f"blk.{il}.indexer_compressor_ape.weight", (index_width, ratio), GGML_TYPE_F16),
                (f"blk.{il}.indexer_compressor_kv.weight", (n_embd, index_width), GGML_TYPE_F16),
                (f"blk.{il}.indexer_compressor_gate.weight", (n_embd, index_width), GGML_TYPE_F16),
                (f"blk.{il}.indexer_compressor_norm.weight", (n_indexer_head_dim,), GGML_TYPE_F32),
            ])

        tensors_def.extend([
            (f"blk.{il}.hc_ffn_fn.weight", (hc_dim, hc_mix_dim), GGML_TYPE_F16),
            (f"blk.{il}.hc_ffn_scale.weight", (3,), GGML_TYPE_F32),
            (f"blk.{il}.hc_ffn_base.weight", (hc_mix_dim,), GGML_TYPE_F32),
            (f"blk.{il}.ffn_norm.weight", (n_embd,), GGML_TYPE_F32),
            (f"blk.{il}.ffn_gate_inp.weight", (n_embd, n_expert), GGML_TYPE_F16),
            (f"blk.{il}.ffn_exp_probs_b.bias", (n_expert,), GGML_TYPE_F32),
            (f"blk.{il}.ffn_gate_exps.weight", (n_embd, n_ff_exp, n_expert), GGML_TYPE_Q8_0),
            (f"blk.{il}.ffn_up_exps.weight", (n_embd, n_ff_exp, n_expert), GGML_TYPE_Q8_0),
            (f"blk.{il}.ffn_down_exps.weight", (n_ff_exp, n_embd, n_expert), GGML_TYPE_Q8_0),
            (f"blk.{il}.ffn_gate_shexp.weight", (n_embd, n_ff_exp), GGML_TYPE_Q8_0),
            (f"blk.{il}.ffn_up_shexp.weight", (n_embd, n_ff_exp), GGML_TYPE_Q8_0),
            (f"blk.{il}.ffn_down_shexp.weight", (n_ff_exp, n_embd), GGML_TYPE_Q8_0),
        ])

    # Build tensor headers & compute offsets
    tensor_headers = []
    tensor_payloads = []
    rel_offset = 0

    for name, dims, gtype in tensors_def:
        n_bytes = calc_tensor_bytes(dims, gtype)
        header = pack_str(name) + pack_u32(len(dims))
        for d in dims:
            header += pack_u64(d)
        header += pack_u32(gtype) + pack_u64(rel_offset)
        tensor_headers.append(header)

        payload = generate_tensor_data(n_bytes, gtype)
        tensor_payloads.append((payload, n_bytes))

        rel_offset += pad_to(n_bytes, GGUF_ALIGNMENT)

    # Write GGUF File
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        # Header
        f.write(GGUF_MAGIC)
        f.write(pack_u32(GGUF_VERSION))
        f.write(pack_u64(len(tensors_def)))
        f.write(pack_u64(len(kv_items)))
        f.write(kv_blob)

        # Tensor directory
        for th in tensor_headers:
            f.write(th)

        # Padding before data
        hdr_len = f.tell()
        pad_bytes = pad_to(hdr_len, GGUF_ALIGNMENT) - hdr_len
        if pad_bytes:
            f.write(b"\x00" * pad_bytes)

        # Tensor data payloads
        for payload, n_bytes in tensor_payloads:
            f.write(payload)
            pad = pad_to(n_bytes, GGUF_ALIGNMENT) - n_bytes
            if pad:
                f.write(b"\x00" * pad)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Successfully generated {out_path} ({size_mb:.2f} MB, {len(tensors_def)} tensors).")

if __name__ == "__main__":
    main()
