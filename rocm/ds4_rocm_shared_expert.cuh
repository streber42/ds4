extern "C" int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!cuda_tensor_has_f32(out, n) || !cuda_tensor_has_f32(gate, n) || !cuda_tensor_has_f32(up, n)) return 0;
    if (n == 0u) return 1;
    swiglu_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight);
    return cuda_ok(cudaGetLastError(), "swiglu launch");
}
extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    if (!gate || !up || !mid || !model_map || !x ||
        in_dim == 0u || out_dim == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t x_bytes = 0;
    uint64_t out_bytes = 0;
    if (!cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !cuda_u64_mul3_checked(in_dim, 1u, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(out_dim, 1u, sizeof(float), &out_bytes) ||
        !cuda_tensor_has_bytes(x, x_bytes) || !cuda_tensor_has_bytes(gate, out_bytes) ||
        !cuda_tensor_has_bytes(up, out_bytes) || !cuda_tensor_has_bytes(mid, out_bytes)) {
        return 0;
    }
    if (in_dim == 4096u && (in_dim & 31u) == 0u &&
        cuda_model_range_fits(model_size, gate_offset, weight_bytes) &&
        cuda_model_range_fits(model_size, up_offset, weight_bytes) &&
        !cuda_runtime_config()->disable_shared_gate_up_fused_w32) {
        const char *wg = cuda_model_range_ptr(model_map, gate_offset, weight_bytes, "shared_gate_q8");
        const char *wu = cuda_model_range_ptr(model_map, up_offset, weight_bytes, "shared_up_q8");
        if (!wg || !wu) return 0;
        const int store_gate_up = (g_quality_mode || cuda_runtime_config()->graph_dump) ? 1 : 0;
        const unsigned rows_per_block = 32u;
        shared_gate_up_swiglu_q8_0_rows_w32_kernel<<<
                (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                rows_per_block * 32u>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                reinterpret_cast<const unsigned char *>(wg),
                reinterpret_cast<const unsigned char *>(wu),
                (const float *)x->ptr,
                (uint32_t)blocks,
                out_dim,
                row_bytes,
                store_gate_up,
                clamp);
        return cuda_ok(cudaGetLastError(), "shared gate/up fused q8 launch");
    }
    return ds4_gpu_matmul_q8_0_pair_tensor(gate, up,
                                             model_map, model_size,
                                             gate_offset, up_offset,
                                             in_dim, out_dim, out_dim,
                                             x, 1) &&
           ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp) {
    (void)gate; (void)up; (void)mid; (void)model_map; (void)model_size;
    (void)gate_offset; (void)up_offset; (void)in_dim; (void)out_dim;
    (void)x; (void)n_tok; (void)clamp;
    return 0;
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_batch_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

extern "C" int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    if (!mid || out_dim == 0u || out_dim > UINT32_MAX) return 0;
    uint64_t tmp_bytes = 0;
    if (!cuda_u64_mul3_checked(2u, out_dim, sizeof(float), &tmp_bytes)) return 0;
    void *tmp = cuda_tmp_alloc(tmp_bytes, "shared gate/up mid wrapper");
    if (!tmp) return 0;
    ds4_gpu_tensor gate_tmp = { tmp, out_dim * sizeof(float), 0 };
    ds4_gpu_tensor up_tmp = { (char *)tmp + out_dim * sizeof(float),
                              out_dim * sizeof(float),
                              0 };
    return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(&gate_tmp,
                                                     &up_tmp,
                                                     mid,
                                                     model_map,
                                                     model_size,
                                                     gate_offset,
                                                     up_offset,
                                                     in_dim,
                                                     out_dim,
                                                     x,
                                                     clamp);
}

/* Two-rank TP load-balancing: whichever rank has fewer of the 6 selected
 * routed experts locally this decode step computes the *entire* shared-mid
 * SwiGLU intermediate while the busier rank is still crunching routed
 * experts; the other rank contributes nothing this call (all its rows
 * stay unwritten -- the caller only ever reads the winning rank's mid).
 * Ties go to home_rank so the home tier never makes an unnecessary peer
 * store. Ported verbatim (same tie-break) from the CUDA TP path
 * (ds4_cuda.cu shared_mid_q8_0_preq_warp8_exact_kernel) since which rank
 * wins must exactly match the reference for logits to line up. */
__global__ static void shared_mid_q8_0_preq_warp8_exact_kernel(
        float *mid,
        const unsigned char *gate_w,
        const unsigned char *up_w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        float clamp,
        const int32_t *selected,
        uint32_t expert_split,
        bool home_rank,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    if (selected) {
        uint32_t home_count = 0u;
        uint32_t peer_count = 0u;
        #pragma unroll
        for (uint32_t i = 0; i < 6u; i++) {
            const int32_t expert = selected[i];
            if (expert >= 0 && (uint32_t)expert < expert_split) {
                home_count++;
            } else if (expert >= 0 && (uint32_t)expert < 2u * expert_split) {
                peer_count++;
            }
        }
        const bool assigned = home_rank
            ? home_count <= peer_count : peer_count < home_count;
        if (!assigned) return;
    }
    const unsigned char *gate_row = gate_w + row * blocks * 34u;
    const unsigned char *up_row = up_w + row * blocks * 34u;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const int8_t *xqb = xq + b * 32u;
        const float xs = xscale[b];
        const unsigned char *gb = gate_row + b * 34u;
        const unsigned char *ub = up_row + b * 34u;
        gate += __half2float(*(const __half *)gb) * xs *
                (float)dot_i8_block((const int8_t *)(gb + 2u), xqb, bn, use_dp4a);
        up += __half2float(*(const __half *)ub) * xs *
              (float)dot_i8_block((const int8_t *)(ub + 2u), xqb, bn, use_dp4a);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            gate = fminf(gate, clamp);
            up = fminf(fmaxf(up, -clamp), clamp);
        }
        const float silu = gate / (1.0f + expf(-gate));
        mid[row] = silu * up * 1.0f;
    }
}

extern "C" int ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *prequant,
        uint32_t                expert_split,
        bool                    home_rank) {
    if (!mid || !x || !model_map || in_dim == 0u || out_dim == 0u ||
        x->bytes < in_dim * sizeof(float) ||
        mid->bytes < out_dim * sizeof(float) ||
        (selected && (selected->bytes < 6u * sizeof(int32_t) ||
                      expert_split == 0u))) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (gate_offset > model_size || up_offset > model_size ||
        out_dim > UINT64_MAX / (blocks * 34u)) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * blocks * 34u;
    if (weight_bytes > model_size - gate_offset ||
        weight_bytes > model_size - up_offset) {
        return 0;
    }
    const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, weight_bytes, "shared_mid_gate_exact");
    const char *up_w = cuda_model_range_ptr(model_map, up_offset, weight_bytes, "shared_mid_up_exact");
    if (!gate_w || !up_w) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    int8_t *xq;
    float *xscale;
    if (prequant) {
        if (prequant->bytes < tmp_bytes) return 0;
        xq = (int8_t *)prequant->ptr;
        xscale = (float *)((char *)prequant->ptr + scale_offset);
    } else {
        void *tmp = cuda_tmp_alloc(tmp_bytes, "shared mid q8 exact prequant");
        if (!tmp) return 0;
        xq = (int8_t *)tmp;
        xscale = (float *)((char *)tmp + scale_offset);
        quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32>>>(
                xq, xscale, (const float *)x->ptr, in_dim, blocks);
        if (!cuda_ok(cudaGetLastError(), "shared mid q8 exact quantize launch")) {
            return 0;
        }
    }
    const int use_dp4a = 1;
    shared_mid_q8_0_preq_warp8_exact_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
            (float *)mid->ptr,
            (const unsigned char *)gate_w,
            (const unsigned char *)up_w,
            xq,
            xscale,
            in_dim,
            out_dim,
            blocks,
            clamp,
            selected ? (const int32_t *)selected->ptr : NULL,
            expert_split,
            home_rank,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "shared mid q8 exact launch");
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(gate,
                                                     up,
                                                     mid,
                                                     model_map,
                                                     model_size,
                                                     gate_offset,
                                                     up_offset,
                                                     in_dim,
                                                     out_dim,
                                                     x,
                                                     clamp);
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp) {
    if (n_tok == 1u) {
        return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(gate,
                                                         up,
                                                         mid,
                                                         model_map,
                                                         model_size,
                                                         gate_offset,
                                                         up_offset,
                                                         in_dim,
                                                         out_dim,
                                                         x,
                                                         clamp);
    }
    if (clamp > 1.0e-6f) return 0;
    return ds4_gpu_shared_gate_up_swiglu_q8_0_batch_tensor(gate,
                                                           up,
                                                           mid,
                                                           model_map,
                                                           model_size,
                                                           gate_offset,
                                                           up_offset,
                                                           in_dim,
                                                           out_dim,
                                                           x,
                                                           n_tok);
}

static cudaStream_t g_shared_gate_up_stream = NULL;
static cudaEvent_t g_shared_gate_up_ready_event = NULL;
static void *g_shared_gate_up_tmp = NULL;
static uint64_t g_shared_gate_up_tmp_bytes = 0;
static int g_shared_gate_up_pending = 0;

static int cuda_shared_gate_up_async_wait_internal(void) {
    if (!g_shared_gate_up_pending) return 1;
    cudaError_t err = cudaStreamSynchronize(g_shared_gate_up_stream);
    g_shared_gate_up_pending = 0;
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "shared gate/up async wait failed: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static void *cuda_shared_gate_up_async_tmp_alloc(uint64_t bytes) {
    if (bytes == 0) return NULL;
    if (g_shared_gate_up_tmp_bytes >= bytes) return g_shared_gate_up_tmp;
    if (g_shared_gate_up_tmp) {
        (void)cuda_shared_gate_up_async_wait_internal();
        (void)cudaFree(g_shared_gate_up_tmp);
        g_shared_gate_up_tmp = NULL;
        g_shared_gate_up_tmp_bytes = 0;
    }
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "shared gate/up async temp alloc failed (%.2f MiB): %s\n",
                (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    g_shared_gate_up_tmp = ptr;
    g_shared_gate_up_tmp_bytes = bytes;
    return g_shared_gate_up_tmp;
}

static void cuda_shared_gate_up_async_cleanup(void) {
    if (g_shared_gate_up_stream) {
        (void)cuda_shared_gate_up_async_wait_internal();
    }
    if (g_shared_gate_up_tmp) {
        (void)cudaFree(g_shared_gate_up_tmp);
        g_shared_gate_up_tmp = NULL;
        g_shared_gate_up_tmp_bytes = 0;
    }
    if (g_shared_gate_up_ready_event) {
        (void)cudaEventDestroy(g_shared_gate_up_ready_event);
        g_shared_gate_up_ready_event = NULL;
    }
    if (g_shared_gate_up_stream) {
        (void)cudaStreamDestroy(g_shared_gate_up_stream);
        g_shared_gate_up_stream = NULL;
    }
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_async_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    if (g_quality_mode || cuda_runtime_config()->graph_dump) return 0;
    if (g_shared_gate_up_pending && !cuda_shared_gate_up_async_wait_internal()) return 0;
    if (!gate || !up || !mid || !model_map || !x ||
        in_dim == 0u || out_dim == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0;
    uint64_t weight_bytes = 0;
    if (!cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes)) {
        return 0;
    }
    if (g_quality_mode ||
        !gate || !up || !mid || !model_map || !x ||
        in_dim == 0u || out_dim == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        gate_offset > model_size || up_offset > model_size ||
        weight_bytes > model_size - gate_offset ||
        weight_bytes > model_size - up_offset ||
        x->bytes < in_dim * sizeof(float) ||
        gate->bytes < out_dim * sizeof(float) ||
        up->bytes < out_dim * sizeof(float) ||
        mid->bytes < out_dim * sizeof(float)) {
        return 0;
    }
    const char *wg = cuda_model_range_ptr(model_map, gate_offset, weight_bytes, "shared_gate_q8_pair_async");
    const char *wu = cuda_model_range_ptr(model_map, up_offset, weight_bytes, "shared_up_q8_pair_async");
    if (!wg || !wu) return 0;
    if (!g_shared_gate_up_stream) {
        int least_priority = 0;
        int greatest_priority = 0;
#ifdef __HIP_PLATFORM_AMD__
        hipError_t err = hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        if (err == hipSuccess) {
            err = hipStreamCreateWithPriority(&g_shared_gate_up_stream, cudaStreamNonBlocking, least_priority);
        } else {
            (void)cudaGetLastError();
            err = hipStreamCreateWithFlags(&g_shared_gate_up_stream, cudaStreamNonBlocking);
        }
        if (err != hipSuccess) return 0;
#else
        cudaError_t err = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        if (err == cudaSuccess) {
            err = cudaStreamCreateWithPriority(&g_shared_gate_up_stream, cudaStreamNonBlocking, least_priority);
        } else {
            (void)cudaGetLastError();
            err = cudaStreamCreateWithFlags(&g_shared_gate_up_stream, cudaStreamNonBlocking);
        }
        if (err != cudaSuccess) return 0;
#endif
    }
    if (!g_shared_gate_up_ready_event) {
        cudaError_t err = cudaEventCreateWithFlags(&g_shared_gate_up_ready_event, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "shared gate/up async event create failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    /*
     * This stream is intentionally non-blocking so it can overlap routed MoE.
     * Non-blocking streams do not inherit default-stream ordering, so explicitly
     * wait until the default-stream producer of x (ffn_norm) has completed before
     * quantizing it here.
     */
    cudaError_t dep_err = cudaEventRecord(g_shared_gate_up_ready_event, 0);
    if (dep_err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "shared gate/up async dependency record failed: %s\n", cudaGetErrorString(dep_err));
        (void)cudaGetLastError();
        return 0;
    }
#ifdef __HIP_PLATFORM_AMD__
    dep_err = hipStreamWaitEvent(g_shared_gate_up_stream, g_shared_gate_up_ready_event, 0);
#else
    dep_err = cudaStreamWaitEvent(g_shared_gate_up_stream, g_shared_gate_up_ready_event, 0);
#endif
    if (dep_err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "shared gate/up async dependency wait failed: %s\n", cudaGetErrorString(dep_err));
        (void)cudaGetLastError();
        return 0;
    }
    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_shared_gate_up_async_tmp_alloc(tmp_bytes);
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = 1;
    dim3 qgrid((unsigned)blocks, 1, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32, 0, g_shared_gate_up_stream>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "shared gate/up async quantize launch")) return 0;
    matmul_q8_0_pair_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256, 0, g_shared_gate_up_stream>>>(
            (float *)gate->ptr,
            (float *)up->ptr,
            reinterpret_cast<const unsigned char *>(wg),
            reinterpret_cast<const unsigned char *>(wu),
            xq,
            xscale,
            in_dim,
            out_dim,
            out_dim,
            blocks,
            use_dp4a);
    if (!cuda_ok(cudaGetLastError(), "shared gate/up async pair launch")) return 0;
    swiglu_kernel<<<((unsigned)out_dim + 255u) / 256u, 256, 0, g_shared_gate_up_stream>>>(
            (float *)mid->ptr,
            (const float *)gate->ptr,
            (const float *)up->ptr,
            (uint32_t)out_dim,
            clamp,
            1.0f);
    if (!cuda_ok(cudaGetLastError(), "shared gate/up async swiglu launch")) return 0;
    g_shared_gate_up_pending = 1;
    return 1;
}

extern "C" int ds4_gpu_shared_gate_up_async_wait(void) {
    return cuda_shared_gate_up_async_wait_internal();
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_batch_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    uint64_t x_bytes = 0, out_bytes = 0;
    if (!gate || !up || !mid || !model_map || !x || n_tok == 0 ||
        (in_dim & 31u) != 0u || in_dim == 0u || out_dim == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || gate->bytes < out_bytes || up->bytes < out_bytes || mid->bytes < out_bytes) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0;
    if (!cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes)) return 0;
    if (gate_offset > model_size || up_offset > model_size ||
        weight_bytes > model_size - gate_offset || weight_bytes > model_size - up_offset) {
        return 0;
    }
    const char *wg = cuda_model_range_ptr(model_map, gate_offset, weight_bytes, "shared_gate_q8_batch");
    const char *wu = cuda_model_range_ptr(model_map, up_offset, weight_bytes, "shared_up_q8_batch");
    if (!wg || !wu) return 0;

    const uint32_t rows_per_block = 32u;
    const uint32_t tile = 16u;
    const uint32_t block_tile = 16u;
    const dim3 grid((uint32_t)((out_dim + rows_per_block - 1u) / rows_per_block),
                    (uint32_t)((n_tok + tile - 1u) / tile),
                    1u);
    const size_t shmem = (size_t)tile * block_tile * 32u * sizeof(float);
    const int store_gate_up = (g_quality_mode || cuda_runtime_config()->graph_dump) ? 1 : 0;
#define DS4_LAUNCH_SHARED_GU_BATCH(TT, BT) \
    shared_gate_up_swiglu_q8_0_batch_sharedx_w32_kernel<TT, BT><<<grid, rows_per_block * 32u, shmem>>>( \
            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr, \
            reinterpret_cast<const unsigned char *>(wg), reinterpret_cast<const unsigned char *>(wu), \
            (const float *)x->ptr, (uint32_t)blocks, (uint32_t)out_dim, (uint32_t)n_tok, row_bytes, store_gate_up)
    DS4_LAUNCH_SHARED_GU_BATCH(16u, 16u);
#undef DS4_LAUNCH_SHARED_GU_BATCH
    return cuda_ok(cudaGetLastError(), "shared gate/up fused q8 batch launch");
}
