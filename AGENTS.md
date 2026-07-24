# AGENTS.md

Working tree for the **ds4 ROCm tensor-parallelism port** — kyuz0's `gfx1201-discrete-gpu`
branch rebased onto `antirez/ds4` upstream (which added CUDA/Metal tensor parallelism,
DSpark, and session batching). Goal: implement the ROCm/gfx1201 TP path so DeepSeek-V4-Flash
can run tensor-parallel across 4× AMD R9700 instead of pipeline layer-split only.

## Agent skills

### Issue tracker

Issues and PRDs live as markdown under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context (CONTEXT.md + docs/adr/ at root). See `docs/agents/domain.md`.
