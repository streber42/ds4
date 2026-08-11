#!/usr/bin/env python3
"""Issue 04: aggregate the ds4-bench sweep CSVs into the tables the issue asks for.

Reads .scratch/gfx1201-wmma-v2/artifacts/04-csv-<shape>-<build>-r<n>.csv.  The
`pass` column is what makes the cold/warm split possible: pass 1 of each process
carries the once-per-process cost, later passes do not.

  usage: 04-summarize.py short|full
"""

import csv
import glob
import os
import statistics
import sys

ART = os.path.dirname(os.path.abspath(__file__))


def load(shape):
    """(build, pass, frontier) -> {repeat: prefill t/s}."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(ART, f"04-csv-{shape}-*.csv"))):
        tag = os.path.basename(path)[len(f"04-csv-{shape}-"):-len(".csv")]
        build, _, repeat = tag.partition("-r")
        with open(path) as fp:
            for r in csv.DictReader(fp):
                key = (build, int(r["pass"]), int(r["ctx_tokens"]))
                rows.setdefault(key, {})[int(repeat)] = float(r["prefill_tps"])
    return rows


def fmt(vals):
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{statistics.mean(vals):.2f} ±{(max(vals) - min(vals)) / 2:.2f} (n={len(vals)})"


def delta(w, n):
    if not w or not n:
        return "-", "-"
    mw, mn = statistics.mean(w), statistics.mean(n)
    return f"{mw - mn:+.2f}", f"{100.0 * (mw - mn) / mn:+.1f}%"


def main():
    shape = sys.argv[1] if len(sys.argv) > 1 else "short"
    rows = load(shape)
    if not rows:
        sys.exit(f"no CSVs matching 04-csv-{shape}-*.csv")

    passes = sorted({k[1] for k in rows})
    frontiers = sorted({k[2] for k in rows})
    vals = lambda b, p, f: list(rows.get((b, p, f), {}).values())

    print(f"## {shape}: prefill t/s by pass and frontier\n")
    print("| pass | frontier | WMMA | NO_WMMA | delta | delta % |")
    print("|---|---|---|---|---|---|")
    for p in passes:
        for f in frontiers:
            d, dp = delta(vals("wmma", p, f), vals("nowmma", p, f))
            print(f"| {p} | {f} | {fmt(vals('wmma', p, f))} | "
                  f"{fmt(vals('nowmma', p, f))} | {d} | {dp} |")

    # The cold-start contribution: same build, same frontier, pass 1 vs pass 2+.
    if len(passes) > 1:
        print(f"\n## {shape}: cold-start cost (pass 1 vs warm passes, within build)\n")
        print("| build | frontier | pass 1 (cold) | passes 2+ (warm) | warm/cold |")
        print("|---|---|---|---|---|")
        for b in ("wmma", "nowmma"):
            for f in frontiers:
                cold, warm = vals(b, 1, f), [v for p in passes[1:] for v in vals(b, p, f)]
                if not cold or not warm:
                    continue
                print(f"| {b} | {f} | {fmt(cold)} | {fmt(warm)} | "
                      f"{statistics.mean(warm) / statistics.mean(cold):.3f}x |")

    # Restate the cold penalty as seconds rather than t/s.  A ratio scales with
    # whatever throughput the build has and so cannot distinguish "the kernel is
    # slower when cold" from "a fixed cost is charged to the first chunk"; the
    # additive form does, and it is comparable across the two builds directly.
    if len(passes) > 1:
        print(f"\n## {shape}: the same cost in seconds (prefill_tokens / t/s)\n")
        print("| build | frontier | cold sec | warm sec | cold-start overhead |")
        print("|---|---|---|---|---|")
        for b in ("wmma", "nowmma"):
            for f in frontiers:
                cold, warm = vals(b, 1, f), [v for p in passes[1:] for v in vals(b, p, f)]
                if not cold or not warm:
                    continue
                n_tok = f if f == frontiers[0] else f - frontiers[frontiers.index(f) - 1]
                cs = n_tok / statistics.mean(cold)
                ws = n_tok / statistics.mean(warm)
                print(f"| {b} | {f} | {cs:.2f} | {ws:.2f} | {cs - ws:+.2f} s |")

    # Why a single cold sample cannot support a headline: pair the two builds
    # invocation-by-invocation instead of averaging, so the swing is visible.
    print(f"\n## {shape}: per-invocation delta at each frontier, cold (pass 1) vs warm (pass 2)\n")
    print("| frontier | pass | invocation | WMMA | NO_WMMA | delta % |")
    print("|---|---|---|---|---|---|")
    for f in frontiers:
        for p in passes[:2]:
            for rep in sorted(rows.get(("wmma", p, f), {})):
                w = rows[("wmma", p, f)][rep]
                n = rows.get(("nowmma", p, f), {}).get(rep)
                dp = f"{100.0 * (w - n) / n:+.1f}%" if n else "-"
                label = "cold" if p == 1 else "warm"
                print(f"| {f} | {p} ({label}) | r{rep} | {w:.2f} | "
                      f"{n:.2f} | {dp} |" if n else
                      f"| {f} | {p} ({label}) | r{rep} | {w:.2f} | - | - |")


if __name__ == "__main__":
    main()
