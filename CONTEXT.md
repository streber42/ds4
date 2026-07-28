# Domain Model — ds4 GPU engine

## Terms

### tier sweep
A loop over all 4 TP=4 ranks (tiers 0–3) within a single prefill/decode layer function.
Each tier switches the active GPU device, sets `g->tp_rank`, computes only its owned
shard of a sharded operation (MoE experts, attention heads), and stores the partial
result in a per-tier buffer. After the sweep, an all-reduce combines the partials.

### batch prefill attention function
`metal_graph_encode_layer_attention_batch` (ds4.c:27804–29585). Runs as a single call
per layer during prefill, processing up to `prefill_cap` tokens at once.

Phases (defined by `DS4_METAL_PROFILE_ATTN_STAGE` markers):
1. **hc_pre** — HC mix/split/weighted sum → attn_cur
2. **norm** — RMSNorm → attn_norm
3. **q_path** — Q_a proj (q_a → qr → qr_norm), Q_b proj (q_b → q → head_norm → RoPE)
4. **kv_path** — KV proj (kv → kv_norm → RoPE → fp8 quantize), raw cache store
5. **compressor** — Compressor prefill, indexer setup, compressed cache emit
6. **attention** — Kernel dispatch (raw/mixed/indexed/per-token) reads Q + KV → heads
7. **inv_rope** — Inverse RoPE on attention output
8. **output_proj** — Attention output projection (A+B stages) + TP=2 row swap
9. **hc_post** — HC expand (residual + attn_out → after_attn_hc)

### sharded attention weight
A tensor where `engine_tp4_shard_divisor` returns 4, causing each GPU to cache only
25% of the rows. Currently: `attn_q_b`, `attn_output`, `attn_output_a`.

### cuda_resolve_weight_ptr
(ds4_cuda.cu:728) Resolves a model weight pointer for a given logical tier.
Returns the GPU-cached pointer if the weight range is in the tier's device cache
(checked via `ds4_gpu_lookup_cache_strict`), otherwise returns NULL.

### tp_row_split_attn
A TP=2 optimization in batch prefill attention (ds4.c:27846). Gated on
`g->tp_world == 2`. Splits token rows between two ranks, with each rank computing
full 64-head attention on half the tokens, then swapping row halves. Not applicable
to TP=4 (which splits heads, not tokens).

### MLA (Multi-head Latent Attention)
DeepSeek V4's attention architecture: `DS4_N_HEAD_KV = 1` — one shared K,V latent
per token shared across all 64 heads. Head parallelism is in Q projection and
output projection only. Implication: all tiers produce identical K,V data, so KV
cache can be replicated per-tier rather than sharded.
