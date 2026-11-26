# Day 53 — KV Cache from Scratch

Generation that stops recomputing the whole prefix at every step. Each layer stores the keys
and values of every token it has processed; each new step feeds only the newest token.

## Quick start

```bash
python src/kv_gpt.py        # cached vs uncached, 100 new tokens, equivalence check
python src/benchmark.py     # speedup sweep at 16 / 64 / 256 new tokens
```

## Core interface

```python
model = MiniGPT(vocab_size=100, no_of_components=6, hidden_states=128,
                no_of_heads=8, mask=True, context_window=32)

model.generate(prompt, max_new_tokens=100, use_cache=True, drop_cache=16)
```

- `use_cache=False` runs the uncached reference loop with the same window policy
- `drop_cache`: how many of the oldest tokens to drop when the window passes 32

## Results

**Equivalence:** cached and uncached greedy output are token-identical, including across
several window slides.

**Speedup** (random-init model, CPU, batch 2, 6-token prompt, 3 runs, mean ± std):

| New tokens | Uncached | Cached | Speedup |
|---|---|---|---|
| 16 | 43.4 ± 1.0 ms | 34.3 ± 3.2 ms | 1.27× |
| 64 | 256.9 ± 8.5 ms | 147.8 ± 14.2 ms | 1.74× |
| 256 | 2,608.5 ± 60.8 ms | 588.5 ± 22.1 ms | 4.43× |

## The math

Uncached work for a prompt of p tokens and n new tokens:

```
p + (p + 1) + … + (p + n − 1) = n (2p + n − 1) / 2   positions
```

Cached: `p + (n − 1)` positions. At p = 6, n = 256: 34,176 vs 261.

Measured time follows `calls × c + positions × t` with c ≈ 2.1 ms per call and t ≈ 0.05 ms per
position. Both loops make the same number of calls, so the fixed cost caps the speedup on a
small model.

## Key insight

A cache is only valid while nothing it stores can change. The causal mask guarantees that
for new tokens. A sliding window breaks it: kept tokens' deeper-layer values still carry the
dropped tokens' influence, so the cache must be rebuilt, not trimmed.