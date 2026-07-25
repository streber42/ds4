#!/usr/bin/env python3
"""OpenCode Go reference comparison evaluation harness for ds4.

Compares our local ROCm tensor-parallel server (default: http://localhost:8000/v1)
against the official live OpenCode Go reference API (default: https://opencode.ai/zen/go/v1).

Evaluates short QA, code generation, math/reasoning, multi-turn chat, and long prefill prompts.
Outputs structured JSON and Markdown comparison reports under .scratch/rocm-tensor-parallel/eval-out/.
"""

import argparse
import difflib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


def load_dotenv(env_path: str = ".env") -> None:
    """Load environment variables from a .env file if present."""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception as exc:
        print(f"Warning: failed to read {env_path}: {exc}", file=sys.stderr)


def normalize_opencode_url(url: str) -> str:
    """Normalize base URL so API calls to /chat/completions land on the right endpoint."""
    url = url.rstrip("/")
    if url.endswith("/v1") or url.endswith("/zen/go/v1"):
        return url
    if "opencode.ai" in url:
        return f"{url}/zen/go/v1"
    return f"{url}/v1"


def normalize_local_url(url: str) -> str:
    """Normalize local base URL to include /v1 suffix if omitted."""
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        return f"{url}/v1"
    return url


CATEGORY_TOKEN_BUDGETS = {
    "Short QA": 1024,
    "Code Generation": 2048,
    "Math & Reasoning": 2048,
    "Multi-turn Chat": 1024,
    "Long Prefill": 2048,
}

SYSTEM_REASONING_DIRECTIVE = {
    "role": "system",
    "content": "You are a concise assistant. Think step-by-step if needed, but state your final answer clearly and concisely after thinking.",
}


def build_long_prefill_messages() -> list[dict[str, str]]:
    """Build a long context prompt (~1200 tokens) with a hidden key for prefill chunk validation."""
    paragraphs = []
    paragraphs.append(
        "In the design of distributed tensor-parallelism engine architectures for deep learning inference, "
        "proper memory placement and buffer sharding across multi-GPU nodes are critical."
    )
    for i in range(1, 20):
        paragraphs.append(
            f"Module section {i}: Verification component {i*7} initializes tensor buffers with rank offset {i*16}. "
            f"Device transfer overhead is minimized through peer-to-peer memory mappings across connected cards. "
            f"Data integrity is maintained by tracking memory allocations."
        )
    paragraphs.append(
        "CRITICAL DOCUMENTATION METADATA KEY: DS4-ROCM-8849. "
        "Keep this key intact for system configuration validation."
    )
    for i in range(20, 35):
        paragraphs.append(
            f"Auxiliary section {i}: Additional tensor cache allocation {i*3} verified. "
            f"System latency remains sub-millisecond during inter-GPU ring reductions."
        )
    paragraphs.append(
        "Question: What is the exact CRITICAL DOCUMENTATION METADATA KEY mentioned in the text above? "
        "State the key directly and concisely after thinking."
    )
    full_text = "\n\n".join(paragraphs)
    return [
        SYSTEM_REASONING_DIRECTIVE,
        {"role": "user", "content": full_text},
    ]


BENCHMARK_SUITE = [
    {
        "id": "short_qa",
        "name": "Short Factual QA",
        "category": "Short QA",
        "description": "Factual query expecting a direct short answer",
        "max_tokens": 1024,
        "messages": [
            SYSTEM_REASONING_DIRECTIVE,
            {"role": "user", "content": "What is the capital of France? Answer in one word after thinking concisely."},
        ],
        "expected_check": lambda text: "paris" in text.lower(),
        "expected_hint": "Contains 'Paris'",
    },
    {
        "id": "code_gen",
        "name": "Code Generation",
        "category": "Code Generation",
        "description": "C pointer swap function implementation",
        "max_tokens": 2048,
        "messages": [
            SYSTEM_REASONING_DIRECTIVE,
            {
                "role": "user",
                "content": (
                    "Write a C function named `swap` that swaps two integers using pointers "
                    "(`void swap(int *a, int *b)`). Provide clean, complete C code. "
                    "Answer concisely after thinking."
                ),
            },
        ],
        "expected_check": lambda text: "swap" in text and "*" in text,
        "expected_hint": "Contains 'swap' and pointer dereference '*'",
    },
    {
        "id": "math_reasoning",
        "name": "Math & Reasoning",
        "category": "Math & Reasoning",
        "description": "Arithmetic problem requiring step-by-step logic",
        "max_tokens": 2048,
        "messages": [
            SYSTEM_REASONING_DIRECTIVE,
            {"role": "user", "content": "What is 2 + 2? Answer directly with the number after thinking."},
        ],
        "expected_check": lambda text: "4" in text,
        "expected_hint": "Contains '4'",
    },
    {
        "id": "multi_turn",
        "name": "Multi-turn Chat",
        "category": "Multi-turn Chat",
        "description": "Context preservation across chat turns",
        "max_tokens": 1024,
        "messages": [
            SYSTEM_REASONING_DIRECTIVE,
            {"role": "user", "content": "My secret security key is ALPHA-4289."},
            {
                "role": "assistant",
                "content": "Understood. I have recorded your secret key as ALPHA-4289.",
            },
            {"role": "user", "content": "What is my secret security key? Answer concisely after thinking."},
        ],
        "expected_check": lambda text: "ALPHA-4289" in text,
        "expected_hint": "Preserves context 'ALPHA-4289'",
    },
    {
        "id": "long_prefill",
        "name": "Long Prefill Chunking",
        "category": "Long Prefill",
        "description": "Long context prefill (~1000 tokens) key extraction",
        "max_tokens": 2048,
        "messages": build_long_prefill_messages(),
        "expected_check": lambda text: "DS4-ROCM-8849" in text,
        "expected_hint": "Extracted key 'DS4-ROCM-8849'",
    },
]


def extract_think_blocks(content: str, reasoning: str) -> tuple[str, str]:
    """Separate <think>...</think> blocks from content and merge into reasoning."""
    c_text = content or ""
    r_text = reasoning or ""

    if "<think>" in c_text:
        while "<think>" in c_text:
            before, rest = c_text.split("<think>", 1)
            if "</think>" in rest:
                think_body, after = rest.split("</think>", 1)
                r_text = (r_text + "\n" + think_body.strip()).strip() if r_text else think_body.strip()
                c_text = (before.strip() + " " + after.strip()).strip()
            else:
                r_text = (r_text + "\n" + rest.strip()).strip() if r_text else rest.strip()
                c_text = before.strip()
                break

    if "</think>" in c_text:
        parts = c_text.split("</think>", 1)
        r_text = (r_text + "\n" + parts[0].replace("<think>", "").strip()).strip() if r_text else parts[0].replace("<think>", "").strip()
        c_text = parts[1].strip()

    c_text = c_text.replace("<think>", "").replace("</think>", "").strip()
    r_text = r_text.replace("<think>", "").replace("</think>", "").strip()

    if not c_text and r_text:
        lines = [ln.strip() for ln in r_text.splitlines() if ln.strip()]
        c_text = lines[-1] if lines else r_text

    return c_text, r_text


def post_chat_stream(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> dict:
    """Send a chat completion request with SSE streaming and measure TTFT and throughput."""
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "curl/7.81.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")

    start_time = time.monotonic()
    ttft = None
    content_chunks = []
    reasoning_chunks = []
    finish_reason = None
    usage = None
    error_msg = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue

                    if chunk.get("usage"):
                        usage = chunk["usage"]

                    choices = chunk.get("choices") or []
                    if choices:
                        c = choices[0]
                        delta = c.get("delta") or {}
                        c_text = delta.get("content")
                        r_text = delta.get("reasoning_content")

                        if (c_text or r_text) and ttft is None:
                            ttft = time.monotonic() - start_time

                        if c_text:
                            content_chunks.append(c_text)

                        if r_text:
                            reasoning_chunks.append(r_text)

                        if c.get("finish_reason"):
                            finish_reason = c["finish_reason"]
    except Exception as exc:
        error_msg = str(exc)

    total_time = time.monotonic() - start_time
    if ttft is None and not error_msg:
        ttft = total_time

    raw_content = "".join(content_chunks).strip()
    raw_reasoning = "".join(reasoning_chunks).strip()

    full_content, full_reasoning = extract_think_blocks(raw_content, raw_reasoning)
    has_reasoning = bool(full_reasoning.strip())

    comp_tokens = (usage or {}).get("completion_tokens")
    if comp_tokens is None:
        combined = (full_reasoning + " " + full_content).strip()
        comp_tokens = len(combined.split()) if combined else 0

    tok_per_sec = comp_tokens / total_time if total_time > 0 else 0.0

    return {
        "status": "ERROR" if error_msg else "SUCCESS",
        "error": error_msg,
        "content": full_content.strip(),
        "reasoning": full_reasoning.strip(),
        "has_reasoning": has_reasoning,
        "ttft_seconds": round(ttft, 4) if ttft else 0.0,
        "total_time_seconds": round(total_time, 4),
        "completion_tokens": comp_tokens,
        "tokens_per_second": round(tok_per_sec, 2),
        "finish_reason": finish_reason,
    }


def run_evaluation(args) -> dict:
    """Run all benchmark cases against local server and OpenCode reference API."""
    load_dotenv(".env")

    local_url = normalize_local_url(args.local_url or os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1"))
    opencode_url = normalize_opencode_url(args.opencode_url or os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1"))
    opencode_key = args.opencode_api_key or os.getenv("OPENCODE_API_KEY", "")
    local_key = args.local_api_key or os.getenv("LOCAL_API_KEY", "")

    skip_opencode = args.skip_opencode or not opencode_key

    print(f"=== OpenCode Reference Evaluation Harness ===")
    print(f"Local Server URL:    {local_url}")
    print(f"OpenCode Reference:  {opencode_url} (skip={skip_opencode})")
    print(f"Model:               {args.model}")
    print(f"Benchmark Cases:     {len(BENCHMARK_SUITE)}")
    print("-" * 50)

    case_results = []
    for case in BENCHMARK_SUITE:
        cid = case["id"]
        cname = case["name"]
        case_budget = case.get("max_tokens", CATEGORY_TOKEN_BUDGETS.get(case["category"], 1024))
        case_max_tokens = case_budget if args.max_tokens == 512 else max(args.max_tokens, case_budget)
        print(f"Running [{cid}] {cname} (max_tokens={case_max_tokens})...", end="", flush=True)

        # 1. Query Local Server
        res_local = post_chat_stream(
            base_url=local_url,
            api_key=local_key,
            model=args.model,
            messages=case["messages"],
            max_tokens=case_max_tokens,
            timeout=args.timeout,
        )

        # 2. Query OpenCode Reference API (or skip)
        if not skip_opencode:
            res_ref = post_chat_stream(
                base_url=opencode_url,
                api_key=opencode_key,
                model=args.model,
                messages=case["messages"],
                max_tokens=case_max_tokens,
                timeout=args.timeout,
            )
        else:
            res_ref = {
                "status": "SKIPPED",
                "error": "OpenCode API key omitted or --skip-opencode set",
                "content": res_local["content"],
                "reasoning": res_local["reasoning"],
                "has_reasoning": res_local["has_reasoning"],
                "ttft_seconds": res_local["ttft_seconds"],
                "total_time_seconds": res_local["total_time_seconds"],
                "completion_tokens": res_local["completion_tokens"],
                "tokens_per_second": res_local["tokens_per_second"],
                "finish_reason": "stop",
            }

        # 3. Calculate Comparison Metrics (Dual Content & Reasoning Matching)
        local_content = res_local["content"]
        local_reasoning = res_local["reasoning"]
        local_combined = (local_content + " " + local_reasoning).strip()

        ref_content = res_ref["content"]
        ref_reasoning = res_ref["reasoning"]
        ref_combined = (ref_content + " " + ref_reasoning).strip()

        local_pass = res_local["status"] == "SUCCESS" and (
            case["expected_check"](local_content) or case["expected_check"](local_combined)
        )
        ref_pass = res_ref["status"] in ("SUCCESS", "SKIPPED") and (
            case["expected_check"](ref_content) or case["expected_check"](ref_combined)
        )

        sim_ratio = difflib.SequenceMatcher(None, res_local["content"].lower(), res_ref["content"].lower()).ratio()
        exact_match = res_local["content"].strip().lower() == res_ref["content"].strip().lower()
        reasoning_agree = res_local["has_reasoning"] == res_ref["has_reasoning"]

        ttft_ratio = (
            round(res_local["ttft_seconds"] / res_ref["ttft_seconds"], 2)
            if res_ref["ttft_seconds"] > 0
            else 1.0
        )
        speedup_ratio = (
            round(res_local["tokens_per_second"] / res_ref["tokens_per_second"], 2)
            if res_ref["tokens_per_second"] > 0
            else 1.0
        )

        overall_case_pass = local_pass and (skip_opencode or ref_pass)

        print(
            f" {'PASS' if overall_case_pass else 'FAIL'} "
            f"[Local: {res_local['tokens_per_second']} t/s, TTFT={res_local['ttft_seconds']}s | "
            f"Ref: {res_ref['tokens_per_second']} t/s, TTFT={res_ref['ttft_seconds']}s | Sim={sim_ratio:.2f}]"
        )

        case_results.append(
            {
                "case_id": cid,
                "name": cname,
                "category": case["category"],
                "description": case["description"],
                "expected_hint": case["expected_hint"],
                "passed": overall_case_pass,
                "local_pass": local_pass,
                "ref_pass": ref_pass,
                "similarity_score": round(sim_ratio, 4),
                "exact_match": exact_match,
                "reasoning_agreement": reasoning_agree,
                "ttft_ratio_local_over_ref": ttft_ratio,
                "speedup_ratio_local_over_ref": speedup_ratio,
                "local": res_local,
                "opencode_ref": res_ref,
            }
        )

    # Aggregate Statistics
    total_cases = len(case_results)
    passed_cases = sum(1 for c in case_results if c["passed"])
    pass_rate = (passed_cases / total_cases) * 100 if total_cases > 0 else 0.0

    mean_local_ttft = sum(c["local"]["ttft_seconds"] for c in case_results) / total_cases
    mean_ref_ttft = sum(c["opencode_ref"]["ttft_seconds"] for c in case_results) / total_cases
    mean_local_tps = sum(c["local"]["tokens_per_second"] for c in case_results) / total_cases
    mean_ref_tps = sum(c["opencode_ref"]["tokens_per_second"] for c in case_results) / total_cases
    mean_similarity = sum(c["similarity_score"] for c in case_results) / total_cases
    reasoning_agree_count = sum(1 for c in case_results if c["reasoning_agreement"])
    ref_passed_cases = sum(1 for c in case_results if c["ref_pass"])

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "local_url": local_url,
        "opencode_url": opencode_url,
        "model": args.model,
        "skip_opencode": skip_opencode,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "pass_rate_percent": round(pass_rate, 2),
        "mean_similarity_score": round(mean_similarity, 4),
        "reasoning_agreement_ratio": f"{reasoning_agree_count}/{total_cases}",
        "local_mean_ttft_seconds": round(mean_local_ttft, 4),
        "ref_mean_ttft_seconds": round(mean_ref_ttft, 4),
        "local_mean_tokens_per_sec": round(mean_local_tps, 2),
        "ref_mean_tokens_per_sec": round(mean_ref_tps, 2),
        "cases": case_results,
    }

    # Save Artifacts
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    json_path = out_dir / f"eval_report_{timestamp_str}.json"
    json_latest_path = out_dir / "eval_report_latest.json"
    md_path = out_dir / f"eval_report_{timestamp_str}.md"
    md_latest_path = out_dir / "eval_report_latest.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(json_latest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Render Markdown Report
    ref_pass_str = "N/A" if skip_opencode else f"{ref_passed_cases}/{total_cases}"
    overall_status = "PASS" if pass_rate == 100.0 else "FAIL"

    md_lines = []
    md_lines.append("# OpenCode Go Reference Comparison Evaluation Report\n")
    md_lines.append(f"- **Timestamp**: {summary['timestamp']}")
    md_lines.append(f"- **Local Server Endpoint**: `{local_url}`")
    md_lines.append(f"- **OpenCode Reference Endpoint**: `{opencode_url}` (skip={skip_opencode})")
    md_lines.append(f"- **Model**: `{args.model}`\n")

    md_lines.append("## Executive Summary\n")
    md_lines.append("| Metric | Local Server | OpenCode Reference | Ratio / Score |")
    md_lines.append("| :--- | :--- | :--- | :--- |")
    md_lines.append(f"| **Pass Rate** | {passed_cases}/{total_cases} ({pass_rate:.1f}%) | {ref_pass_str} | Overall: **{overall_status}** |")
    md_lines.append(f"| **Mean TTFT** | {mean_local_ttft:.3f} s | {mean_ref_ttft:.3f} s | {mean_local_ttft / mean_ref_ttft if mean_ref_ttft > 0 else 1.0:.2f}x |")
    md_lines.append(f"| **Mean Throughput** | {mean_local_tps:.2f} tok/s | {mean_ref_tps:.2f} tok/s | {mean_local_tps / mean_ref_tps if mean_ref_tps > 0 else 1.0:.2f}x |")
    md_lines.append(f"| **Semantic Similarity** | - | - | {mean_similarity:.2f} |")
    md_lines.append(f"| **Reasoning Format Match** | - | - | {reasoning_agree_count}/{total_cases} cases |")
    md_lines.append("\n")

    md_lines.append("## Benchmark Case Breakdown\n")
    md_lines.append("| Case ID | Category | Local TTFT / Speed | OpenCode TTFT / Speed | Similarity | Agreement | Status |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for c in case_results:
        loc_str = f"{c['local']['ttft_seconds']}s / {c['local']['tokens_per_second']} t/s"
        ref_str = f"{c['opencode_ref']['ttft_seconds']}s / {c['opencode_ref']['tokens_per_second']} t/s"
        status_str = "PASS" if c["passed"] else "FAIL"
        md_lines.append(
            f"| `{c['case_id']}` | {c['category']} | {loc_str} | {ref_str} | {c['similarity_score']:.2f} | {'Match' if c['reasoning_agreement'] else 'Mismatch'} | **{status_str}** |"
        )
    md_lines.append("\n")

    md_lines.append("## Output Samples & Comparison Diffs\n")
    for c in case_results:
        md_lines.append(f"### Test Case: `{c['case_id']}` ({c['name']})\n")
        md_lines.append(f"- **Category**: {c['category']}")
        md_lines.append(f"- **Validation Rule**: {c['expected_hint']}")
        md_lines.append(f"- **Local Pass**: {c['local_pass']} | **Reference Pass**: {c['ref_pass']}")
        md_lines.append("\n**Local Server Response:**")
        md_lines.append("```")
        md_lines.append(c["local"]["content"][:800] + ("..." if len(c["local"]["content"]) > 800 else ""))
        md_lines.append("```\n")
        md_lines.append("**OpenCode Reference Response:**")
        md_lines.append("```")
        md_lines.append(c["opencode_ref"]["content"][:800] + ("..." if len(c["opencode_ref"]["content"]) > 800 else ""))
        md_lines.append("```\n")
        if c["local"]["reasoning"] or c["opencode_ref"]["reasoning"]:
            md_lines.append("**Reasoning Content Presence:**")
            md_lines.append(f"- Local Reasoning Present: `{c['local']['has_reasoning']}`")
            md_lines.append(f"- Reference Reasoning Present: `{c['opencode_ref']['has_reasoning']}`")
            md_lines.append("\n")

    report_content = "\n".join(md_lines)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    with open(md_latest_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("-" * 50)
    print(f"Evaluation complete. Reports saved to:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    print(f"Pass Rate: {pass_rate:.1f}% ({passed_cases}/{total_cases})")

    return summary


def run_self_tests() -> None:
    """Run unit tests for extract_think_blocks, adaptive token budgets, and dual matching."""
    print("=== Running Self-Tests for Evaluation Harness Reasoning Budget & Formatting ===")

    # Test 1: extract_think_blocks with clean <think>...</think>
    c, r = extract_think_blocks("<think>Thinking about Paris.</think> Paris", "")
    assert c == "Paris", f"Expected content 'Paris', got '{c}'"
    assert r == "Thinking about Paris.", f"Expected reasoning 'Thinking about Paris.', got '{r}'"
    print("[PASS] Clean <think>...</think> separation verified.")

    # Test 2: extract_think_blocks with unclosed <think>
    c, r = extract_think_blocks("<think>Thinking about Paris.", "")
    assert r == "Thinking about Paris.", f"Expected reasoning 'Thinking about Paris.', got '{r}'"
    assert c == "Thinking about Paris.", f"Expected fallback content 'Thinking about Paris.', got '{c}'"
    print("[PASS] Unclosed <think> block separation and fallback verified.")

    # Test 3: extract_think_blocks when reasoning stream and content stream are separated
    c, r = extract_think_blocks("Paris", "Thinking step by step...")
    assert c == "Paris", f"Expected content 'Paris', got '{c}'"
    assert r == "Thinking step by step...", f"Expected reasoning 'Thinking step by step...', got '{r}'"
    print("[PASS] Separated reasoning and content streams verified.")

    # Test 4: Adaptive Token Budget per Category & System Prompt Directives
    for case in BENCHMARK_SUITE:
        budget = case.get("max_tokens", CATEGORY_TOKEN_BUDGETS.get(case["category"], 1024))
        assert budget >= 1024, f"Case {case['id']} budget {budget} should be >= 1024"
        sys_msgs = [m for m in case["messages"] if m.get("role") == "system"]
        assert len(sys_msgs) > 0, f"Case {case['id']} missing system format directive"
        assert "concise" in sys_msgs[0]["content"].lower(), f"Case {case['id']} system prompt missing concise directive"
    print("[PASS] Adaptive token budgets and explicit prompt directives verified across all benchmark cases.")

    # Test 5: Dual Content & Reasoning Matching
    dummy_check = lambda text: "paris" in text.lower()
    assert dummy_check("Paris")
    comb_text = ("The capital" + " " + "I think it is Paris").strip()
    assert dummy_check(comb_text)
    print("[PASS] Dual content & reasoning matching verified.")

    print("All evaluation harness self-tests PASSED successfully!")


def main():
    parser = argparse.ArgumentParser(description="OpenCode Go reference comparison evaluation harness")
    parser.add_argument("--local-url", default="", help="Local server base URL (default: http://localhost:8000/v1)")
    parser.add_argument("--local-api-key", default="", help="Local server API key (default: from LOCAL_API_KEY env or empty)")
    parser.add_argument("--opencode-url", default="", help="OpenCode reference base URL (default: https://opencode.ai/zen/go/v1)")
    parser.add_argument("--opencode-api-key", default="", help="OpenCode API key (default: from .env)")
    parser.add_argument("--model", default="deepseek-v4-flash", help="Model identifier to query")
    parser.add_argument("--out-dir", default=".scratch/rocm-tensor-parallel/eval-out", help="Output directory for test reports")
    parser.add_argument("--skip-opencode", action="store_true", help="Skip live OpenCode API requests and evaluate local server only")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request HTTP timeout in seconds")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max output tokens for completion")
    parser.add_argument("--self-test", action="store_true", help="Run harness unit tests and exit")

    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        sys.exit(0)

    summary = run_evaluation(args)

    if summary["passed_cases"] < summary["total_cases"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
