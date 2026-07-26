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
