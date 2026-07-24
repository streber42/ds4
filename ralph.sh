#!/usr/bin/env bash
set -euo pipefail

# Ralph Loop Runner for Claude Code + Matt Pocock issue structure with Parallel Worktree support

FEATURE="rocm-tensor-parallel"
MAX_ITERATIONS=50
CONCURRENCY=1
DRY_RUN=false
SINGLE_STEP=false
EFFORT="high"
MODEL=""
CLAUDE_CMD="$HOME/.local/bin/local_claude.sh"
SKIP_PERMISSIONS="--dangerously-skip-permissions"

# Enable multi-core parallel make compilations by default
export MAKEFLAGS="${MAKEFLAGS:--j8}"


usage() {
  echo "Usage: ./ralph.sh [options]"
  echo ""
  echo "Options:"
  echo "  -f, --feature <slug>      Feature directory under .scratch/ (default: rocm-tensor-parallel)"
  echo "  -c, --concurrency <N>     Parallel worker concurrency limit (default: 1)"
  echo "  -p, --parallel            Enable parallel execution with 4 workers"
  echo "  -n, --max-iterations <N>  Max loop iterations (default: 50)"
  echo "  -s, --single-step         Run a single batch/iteration then exit"
  echo "  -d, --dry-run             Print next ready issues & prompts without running Claude Code"
  echo "  -e, --effort <effort>     Set Claude Code effort (low|medium|high, default: high)"
  echo "  -m, --model <model>       Set Claude Code model"
  echo "  -h, --help                Show this help message"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--feature) FEATURE="$2"; shift 2 ;;
    -c|--concurrency) CONCURRENCY="$2"; shift 2 ;;
    -p|--parallel) CONCURRENCY=4; shift ;;
    -n|--max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
    -s|--single-step) SINGLE_STEP=true; shift ;;
    -d|--dry-run) DRY_RUN=true; shift ;;
    -e,--effort) EFFORT="$2"; shift 2 ;;
    -m,--model) MODEL="$2"; shift 2 ;;
    -h,--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="$SCRIPT_DIR/scripts/ralph_engine.py"
WT_BASE="$SCRIPT_DIR/.worktrees"

mkdir -p "$WT_BASE"

echo "============================================================"
echo " Starting Ralph Loop for feature: $FEATURE"
echo " Concurrency limit: $CONCURRENCY worker(s)"
echo "============================================================"

iteration=0

# Only clean up worktrees on explicit exit after loop finishes, not mid-run
cleanup_worktrees() {
  # Remove any leftover .meta and .log files but leave worktree dirs
  # (they are cleaned per-issue after merge in the loop body)
  rm -f "$WT_BASE"/*.meta "$WT_BASE"/*.log 2>/dev/null || true
}
trap cleanup_worktrees EXIT

run_single_issue() {
  local issue="$1"
  echo "Target issue: $issue"

  local prompt
  prompt=$(python3 "$ENGINE" prompt "$FEATURE" "$issue")

  if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY RUN] Would execute Claude Code for issue $issue with prompt:"
    echo "------------------------------------------------------------"
    echo "$prompt"
    echo "------------------------------------------------------------"
    return 0
  fi

  python3 "$ENGINE" status "$issue" "in-progress"

  local claude_cmd=("$CLAUDE_CMD" "$SKIP_PERMISSIONS" --effort "$EFFORT")
  if [[ -n "$MODEL" ]]; then
    claude_cmd+=(--model "$MODEL")
  fi
  # Use --prompt-interactive so Claude Code runs a full tool-using session (writes files,
  # builds, commits) rather than --print which just echoes a response and exits.
  claude_cmd+=(-i "$prompt")

  set +e
  "${claude_cmd[@]}"
  local claude_ret=$?
  set -e

  local status
  status=$(python3 -c "import scripts.ralph_engine as r; print(r.parse_issue(r.Path('$issue'))['status'])")
  echo "Post-run issue status: $status (exit code $claude_ret)"
}

run_parallel_worktree() {
  local issue="$1"
  local slug
  slug=$(basename "$issue" .md)
  local wt_dir="$WT_BASE/$slug"
  local branch_name="ralph-wt-$slug"
  local meta_file="$WT_BASE/$slug.meta"
  local log_file="$WT_BASE/$slug.log"

  echo "[Parallel Worker] Setting up worktree for $slug in $wt_dir..." >&2

  # Suppress git output so it never leaks into captured stdout
  git worktree remove --force "$wt_dir" 2>/dev/null || true
  git branch -D "$branch_name" 2>/dev/null || true
  git worktree add -b "$branch_name" "$wt_dir" HEAD >/dev/null 2>&1

  # Mark in-progress; redirect python output to stderr
  python3 "$ENGINE" status "$issue" "in-progress" >&2

  mkdir -p "$wt_dir/$(dirname "$issue")"
  cp "$issue" "$wt_dir/$issue"

  local prompt
  prompt=$(python3 "$ENGINE" prompt "$FEATURE" "$issue" 2>/dev/null)

  echo "[Parallel Worker] Launching Claude Code process for $slug (logs: $log_file)..." >&2

  (
    cd "$wt_dir"
    local claude_cmd=("$CLAUDE_CMD" "$SKIP_PERMISSIONS" --effort "$EFFORT")
    if [[ -n "$MODEL" ]]; then
      claude_cmd+=(--model "$MODEL")
    fi
    # Use --prompt-interactive so Claude Code runs a full tool-using session (writes
    # files, builds, commits) rather than --print which just prints and exits.
    claude_cmd+=(-i "$prompt")
    "${claude_cmd[@]}" >"$log_file" 2>&1
  ) &
  local pid=$!

  # Write structured metadata to a sidecar file — never to stdout
  printf '%s\n%s\n%s\n%s\n%s\n' \
    "$pid" "$slug" "$issue" "$wt_dir" "$branch_name" > "$meta_file"

  # Return only the pid on stdout so the caller can wait on it
  echo "$pid"
}

while [[ $iteration -lt $MAX_ITERATIONS ]]; do
  iteration=$((iteration + 1))
  echo ""
  echo "--- Loop Iteration $iteration / $MAX_ITERATIONS ---"

  if [[ "$CONCURRENCY" -eq 1 ]]; then
    set +e
    NEXT_ISSUE=$(python3 "$ENGINE" next "$FEATURE" 2>/dev/null)
    RET=$?
    set -e

    if [[ $RET -ne 0 ]] || [[ -z "$NEXT_ISSUE" ]]; then
      echo "No more ready-for-agent issues found."
      break
    fi

    run_single_issue "$NEXT_ISSUE"

    if [[ "$SINGLE_STEP" == true ]]; then
      break
    fi
  else
    set +e
    UNBLOCKED=$(python3 "$ENGINE" unblocked "$FEATURE" -n "$CONCURRENCY" 2>/dev/null)
    set -e

    if [[ -z "$UNBLOCKED" ]]; then
      echo "No more unblocked ready-for-agent issues found."
      break
    fi

    echo "Found unblocked issue batch:"
    echo "$UNBLOCKED"

    if [[ "$DRY_RUN" == true ]]; then
      echo "[DRY RUN] Parallel batch dry run:"
      for issue in $UNBLOCKED; do
        run_single_issue "$issue"
      done
      break
    fi

    pids=()
    slugs=()

    for issue in $UNBLOCKED; do
      pid=$(run_parallel_worktree "$issue")
      pids+=("$pid")
      slugs+=("$(basename "$issue" .md)")
    done

    echo "Waiting for ${#pids[@]} parallel workers to complete..."

    for pid in "${pids[@]}"; do
      wait "$pid" 2>/dev/null || true
    done

    echo "All parallel workers in batch completed. Merging worktree commits..."

    for slug in "${slugs[@]}"; do
      meta_file="$WT_BASE/$slug.meta"
      if [[ ! -f "$meta_file" ]]; then
        echo "Warning: no metadata file for $slug, skipping." >&2
        continue
      fi
      # Read structured fields from sidecar — safe against any characters
      { read -r _pid; read -r _slug; read -r issue; read -r wt_dir; read -r branch_name; } < "$meta_file"

      echo "Processing results for $slug..."

      wt_issue="$wt_dir/$issue"
      wt_status="unknown"
      if [[ -f "$wt_issue" ]]; then
        wt_status=$(python3 -c "import scripts.ralph_engine as r; print(r.parse_issue(r.Path('$wt_issue'))['status'])")
      fi

      echo "Worker $slug reported status: $wt_status"

      if git log HEAD.."$branch_name" --oneline 2>/dev/null | grep -q .; then
        echo "Merging branch $branch_name into main..."
        git merge --no-ff -m "feat($FEATURE): merge completed issue $slug" "$branch_name" || {
          echo "Warning: Merge conflict when merging $branch_name. Reverting merge."
          git merge --abort 2>/dev/null || true
          python3 "$ENGINE" status "$issue" "ready-for-human"
        }
      fi

      if [[ "$wt_status" == "closed" ]] || [[ "$wt_status" == "done" ]]; then
        python3 "$ENGINE" status "$issue" "closed"
        echo "✅ Issue $issue closed successfully."
      elif [[ "$wt_status" == "ready-for-human" ]]; then
        python3 "$ENGINE" status "$issue" "ready-for-human"
        echo "⚠️ Issue $issue flagged for human attention."
      fi

      git worktree remove --force "$wt_dir" 2>/dev/null || true
      git branch -D "$branch_name" 2>/dev/null || true
    done

    if [[ "$SINGLE_STEP" == true ]]; then
      echo "Single step mode enabled. Exiting loop."
      break
    fi
  fi
done

echo ""
echo "============================================================"
echo " Ralph Loop finished."
echo " Status summary:"
python3 "$ENGINE" list "$FEATURE"
echo "============================================================"