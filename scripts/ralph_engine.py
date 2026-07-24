#!/usr/bin/env python3
"""
Ralph Engine - Helper script to parse, manage, and prompt for Matt Pocock style agent issues.
Supports parallel selection and worktree management.
"""

import sys
import os
import re
import argparse
from pathlib import Path

def parse_issue(issue_path):
    path = Path(issue_path)
    if not path.exists():
        return None
    content = path.read_text(encoding='utf-8')
    lines = content.splitlines()
    
    title = ""
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break
            
    status = "unknown"
    status_match = re.search(r"^Status:\s*([\w-]+)", content, re.MULTILINE)
    if status_match:
        status = status_match.group(1).strip()
        
    blocked_by = []
    in_blocked_by = False
    for line in lines:
        if line.startswith("## Blocked by"):
            in_blocked_by = True
            continue
        elif in_blocked_by and line.startswith("## "):
            in_blocked_by = False
        elif in_blocked_by:
            matches = re.findall(r"\.scratch/[^\s`]+\.md", line)
            for m in matches:
                blocked_by.append(m)
                
    return {
        "path": str(path.resolve()),
        "rel_path": str(issue_path),
        "filename": path.name,
        "slug": path.stem,
        "title": title,
        "status": status,
        "blocked_by": blocked_by
    }

def get_all_issues(feature_slug):
    feature_dir = Path(".scratch") / feature_slug
    issues_dir = feature_dir / "issues"
    if not issues_dir.exists():
        return []
    issue_files = sorted(list(issues_dir.glob("*.md")))
    return [parse_issue(f) for f in issue_files if f.is_file()]

def recover_orphaned_in_progress(feature_slug):
    """Reset in-progress issues whose worktree no longer exists.

    When ralph.sh is killed mid-run (Ctrl-C, crash, dry-run, etc.) it may
    have stamped an issue `in-progress` but never created (or already cleaned
    up) the matching worktree. Those issues are permanently stuck unless we
    reset them. We do this automatically on every engine invocation by checking
    whether the expected worktree directory still exists for each in-progress
    issue.
    """
    wt_base = Path(".worktrees")
    issues = get_all_issues(feature_slug)
    recovered = []
    for issue in issues:
        if issue["status"] == "in-progress":
            expected_wt = wt_base / issue["slug"]
            if not expected_wt.exists():
                update_status(issue["rel_path"], "ready-for-agent")
                recovered.append(issue["rel_path"])
    return recovered

def resolve_blocker_status(blocker_ref, issues_by_rel_path):
    norm_ref = blocker_ref.lstrip("./")
    for rel_path, issue in issues_by_rel_path.items():
        if rel_path.lstrip("./") == norm_ref or issue["filename"] == norm_ref:
            return issue["status"]
    return "unknown"

def get_unblocked_issues(feature_slug, max_count=None):
    # Auto-heal any in-progress issues whose worktree has gone away.
    recovered = recover_orphaned_in_progress(feature_slug)
    if recovered:
        print(f"[ralph-engine] Recovered {len(recovered)} orphaned in-progress issue(s): {recovered}", file=sys.stderr)

    issues = get_all_issues(feature_slug)
    issues_by_rel = {i["rel_path"]: i for i in issues}

    unblocked = []
    for issue in issues:
        if issue["status"] in ["ready-for-agent", "needs-triage"]:
            all_blockers_closed = True
            for blocker in issue["blocked_by"]:
                b_status = resolve_blocker_status(blocker, issues_by_rel)
                if b_status not in ["closed", "done"]:
                    all_blockers_closed = False
                    break
            if all_blockers_closed:
                unblocked.append(issue)
                if max_count and len(unblocked) >= max_count:
                    break
    return unblocked

def get_next_issue(feature_slug):
    unblocked = get_unblocked_issues(feature_slug, max_count=1)
    return unblocked[0] if unblocked else None

def update_status(issue_path, new_status):
    path = Path(issue_path)
    if not path.exists():
        return
    content = path.read_text(encoding='utf-8')
    if re.search(r"^Status:\s*[\w-]+", content, re.MULTILINE):
        new_content = re.sub(r"^Status:\s*[\w-]+", f"Status: {new_status}", content, flags=re.MULTILINE)
    else:
        new_content = f"Status: {new_status}\n" + content
    path.write_text(new_content, encoding='utf-8')

def build_prompt(feature_slug, issue_path):
    issue = parse_issue(Path(issue_path))
    prd_path = Path(".scratch") / feature_slug / "PRD.md"
    
    prompt = f"""You are an autonomous AI coding agent executing a single atomic task in a Ralph Loop worktree.

PROJECT & FEATURE CONTEXT:
- Repository guidelines: `AGENTS.md` and `AGENT.md`
- Feature PRD: `{prd_path}`
- Target Issue: `{issue['rel_path']}` ({issue['title']})

YOUR GOAL:
1. Carefully read `AGENTS.md`, `AGENT.md`, `{prd_path}`, and `{issue['rel_path']}`.
2. Implement the requirements described in `{issue['rel_path']}` cleanly and accurately according to project quality rules.
3. Build and test your implementation using parallel compilation (`make -j8` / `make -j8 test`).
4. Ensure all acceptance criteria in `{issue['rel_path']}` are satisfied.

5. In `{issue['rel_path']}`, check off all completed acceptance criteria items (`- [x]`).
6. If tests pass and all acceptance criteria are met, update `Status: in-progress` to `Status: closed` in `{issue['rel_path']}`.
7. Commit your code changes to git with a clean, descriptive message: `feat({feature_slug}): {issue['title']}`.
8. If you are blocked by missing hardware or need human input, update `{issue['rel_path']}` to `Status: ready-for-human` and record your detailed findings under a `## Comments` section at the bottom of the issue file.

Work autonomously until the issue is either `closed` or marked `ready-for-human`.
"""
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Ralph Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    list_parser = subparsers.add_parser("list", help="List all issues for a feature")
    list_parser.add_argument("feature", nargs="?", default="rocm-tensor-parallel", help="Feature slug")

    # unblocked
    unblocked_parser = subparsers.add_parser("unblocked", help="Get all unblocked ready-for-agent issues")
    unblocked_parser.add_argument("feature", nargs="?", default="rocm-tensor-parallel", help="Feature slug")
    unblocked_parser.add_argument("-n", "--limit", type=int, default=0, help="Max number of issues to return (0 = all)")

    # next
    next_parser = subparsers.add_parser("next", help="Get next ready-for-agent issue")
    next_parser.add_argument("feature", nargs="?", default="rocm-tensor-parallel", help="Feature slug")

    # update-status
    status_parser = subparsers.add_parser("status", help="Update issue status")
    status_parser.add_argument("issue_path", help="Path to issue file")
    status_parser.add_argument("new_status", help="New status (e.g. in-progress, closed, ready-for-human)")

    # build-prompt
    prompt_parser = subparsers.add_parser("prompt", help="Build prompt for issue")
    prompt_parser.add_argument("feature", help="Feature slug")
    prompt_parser.add_argument("issue_path", help="Path to issue file")

    args = parser.parse_args()

    if args.command == "list":
        issues = get_all_issues(args.feature)
        for i in issues:
            blockers_str = f" (Blocked by: {', '.join(i['blocked_by'])})" if i['blocked_by'] else ""
            print(f"[{i['status']:<15}] {i['rel_path']} - {i['title']}{blockers_str}")

    elif args.command == "unblocked":
        issues = get_unblocked_issues(args.feature, max_count=args.limit if args.limit > 0 else None)
        for i in issues:
            print(i["rel_path"])

    elif args.command == "next":
        next_issue = get_next_issue(args.feature)
        if next_issue:
            print(next_issue["rel_path"])
        else:
            sys.exit(1)

    elif args.command == "status":
        update_status(args.issue_path, args.new_status)
        print(f"Updated {args.issue_path} -> {args.new_status}")

    elif args.command == "prompt":
        print(build_prompt(args.feature, args.issue_path))

if __name__ == "__main__":
    main()
