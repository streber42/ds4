# AGENTS.md

Working tree for the **ds4 ROCm tensor-parallelism port** — kyuz0's `gfx1201-discrete-gpu`
branch rebased onto `antirez/ds4` upstream (which added CUDA/Metal tensor parallelism,
DSpark, and session batching). Goal: implement the ROCm/gfx1201 TP path so DeepSeek-V4-Flash
can run tensor-parallel across 4× AMD R9700 instead of pipeline layer-split only.

## Agent skills

### GPU access & locking protocol

Agents have direct access to the 4×R9700 GPUs (ROCm/gfx1201). You can run `ds4`, `ds4-bench`, `ds4-eval`, `ds4-server`, and `ds4-agent` with `--rocm --gpu-devices 0,1,2,3` to test on real hardware. Use `rocm-smi` to monitor GPU utilization and VRAM. Do not mark issues as `ready-for-human` solely because they require GPU testing — if you have GPU access, you can implement and verify them.

**GPU Locking Protocol:**
- Before starting GPU work (ROCm compilation, GPU tests, benchmarks), acquire the GPU lock:
  `/home/murphy/src/ralph-loop/scripts/ralph_engine.py gpu-acquire <feature-slug> --agent-id "$$"`
  *(This automatically stops the background `dev-vllm` Docker container to grant exclusive GPU access).*
- Release the GPU lock immediately after GPU work finishes:
  `/home/murphy/src/ralph-loop/scripts/ralph_engine.py gpu-release <feature-slug>`
  *(This restarts `dev-vllm`).*
- The lock auto-releases after 1 hour to prevent deadlocks from crashed agents. Non-GPU work (pure Python, docs, static code edits) does not require the lock.

### Issue tracker

Issues and PRDs live as markdown under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context (CONTEXT.md + docs/adr/ at root). See `docs/agents/domain.md`.
