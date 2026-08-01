# Automated local server lifecycle management for evaluation harness

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## Problem

Running `make test-opencode-eval` or `python3 tests/test_opencode_reference_eval.py` fails all local benchmark cases if `ds4-server` is not already running on `http://localhost:8000/v1`. Currently, the harness requires a human or prior script to manually start `ds4-server` in the background.

## Key Requirements

1. **Health Check Probe**: Probe `http://localhost:8000/v1/models` or `/health` at evaluation startup.
2. **Server Auto-Launch**: Add `--spawn-server` flag to automatically start `./ds4-server` with `--rocm-tensor-parallel` if no local server is listening.
3. **Graceful Shutdown**: Automatically terminate spawned `ds4-server` instances upon completion of evaluation.

## Acceptance Criteria

- [ ] `python3 tests/test_opencode_reference_eval.py --spawn-server` launches local server if offline.
- [ ] Harness waits for server readiness before running benchmark suite.
- [ ] Harness cleans up spawned server process on exit.

## Comments

**2026-08-01 — Backfilled during a project-wide issue-tracker lint; closed
with 0/3 ACs checked and no explanation in this file.** The closing commit
(`f0e2f46`) has a one-line rationale that was never copied here: "resolved
by Docker compose workflow already documented in issue 12." Verified
against #12 — its Comments confirm a health check gates readiness and
`docker compose up` brings up the server with a single command, which
covers this issue's underlying need (a server that's ready without a human
manually starting it first).

**Not a literal match, worth knowing if this resurfaces:** #12's solution
is a container-based `docker compose up`, not the `--spawn-server` pytest
flag this issue's ACs describe — someone running the eval harness directly
against a bare-metal build (no compose) still hits the original problem.
Closing was a reasonable call at the time (compose was the project's actual
deployment path), but if bare-metal local dev iteration becomes common
again, the original ACs here may be worth revisiting rather than assuming
they're permanently moot.
