#!/usr/bin/env python3
"""Compare two ds4 --dump-logits JSON dumps: argmax, top-5, mean/max abs error."""
import json, sys

def load(p):
    d = json.load(open(p))
    return d, [x if x is not None else float('nan') for x in d['logits']]

a_meta, a = load(sys.argv[1])
b_meta, b = load(sys.argv[2])
assert len(a) == len(b), (len(a), len(b))

top = lambda v: sorted(range(len(v)), key=lambda i: -v[i])[:5]
ta, tb = top(a), top(b)
diffs = [abs(x - y) for x, y in zip(a, b)]
mean = sum(diffs) / len(diffs)
mx = max(diffs)
mxi = diffs.index(mx)

print(f"file A: {sys.argv[1]}  argmax_token={a_meta['argmax_token']!r} prompt_tokens={a_meta['prompt_tokens']}")
print(f"file B: {sys.argv[2]}  argmax_token={b_meta['argmax_token']!r} prompt_tokens={b_meta['prompt_tokens']}")
print(f"vocab={len(a)}")
print(f"argmax A={ta[0]} ({a[ta[0]]:.6f})   argmax B={tb[0]} ({b[tb[0]]:.6f})   match={ta[0]==tb[0]}")
print(f"top5   A={ta}\ntop5   B={tb}\ntop5 match (ordered)={ta==tb}  (as set)={set(ta)==set(tb)}")
print(f"mean_abs_logit_err={mean:.6f}  max_abs_logit_err={mx:.6f} at token {mxi}")
