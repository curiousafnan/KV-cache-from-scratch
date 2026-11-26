# Day 53 — Approach

## Shape contract (per layer, batch B, heads H = 8, head size 16)

```
new token input:  (B, 1, 128)
q_h, k_h, v_h:    (B, H, 1, 16)           # newest token only
cache K, V:       (B, H, past_len, 16)    # → (B, H, past_len + 1, 16) after append
scores:           (B, H, 1, past_len + 1)
output:           (B, 1, 128)
```

Measured cache after a 6-token prefill, batch 2: `(2, 8, 6, 16)` for K and for V, in each of
6 layers.

## Prefill and decode

- **Prefill:** the whole prompt goes in at once. Every prompt position is computed in
  parallel and the cache is filled for all of them. Same as a training forward pass.
- **Decode:** one token per step. Only the newest token is fed; its K and V are appended.

## Causal mask with a cache

The query block has `seq_len` rows; the key block has `past_len + seq_len` columns. A query at
row `i` sits at position `past_len + i` and may see keys up to that position:

```
mask = triu(ones(seq_len, total_len), diagonal = past_len + 1)
```

During decode (`seq_len = 1`) nothing is masked: the newest token may see the whole cache.

## Positions

The new token's position is `past_len`, the number of tokens before it in the window. Feeding
one token with a fresh position encoding would give it position 0 at every step, silently.

## Sliding window

Training used 32-token windows, so positions must stay in 0–31.

**Policy (identical in both loops):** keep one `context` tensor. When it grows past 32, drop
the oldest 16 (`drop_cache = 16`, leaving 17 tokens). Positions restart at 0 for the
first kept token.

**Cached loop:** on a slide, discard the cache and rebuild it with one prefill over the
kept tokens. Then decode with the cache until the window fills again. One rebuild every
16 steps instead of every step.

### Why trimming the cache does not work

Deleting the oldest entries from the cache was tried first. It gives different tokens
from the uncached loop, for two reasons:

1. **Positions:** every kept key carries its old position, baked in before layer 1.
2. **Leftover influence:** a kept token's layer-2+ keys were computed while it could still
   attend to the dropped tokens. The uncached loop recomputes the window from scratch, so
   the dropped tokens leave no trace there.

Rebuilding from token IDs fixes both. (Reusing the stale cache on purpose is a real idea —
Transformer-XL does it — but that model was trained for it. This one was not.)

## Benchmark method

- Warmup call before any timing
- Lengths 16, 64, 256 new tokens; 3 runs each; order of cached/uncached alternated per run
- Tokens checked identical at every run
- Each run appended to `experiments.csv`

## A cost model for the speedup

Every forward call costs a fixed amount `c`, plus `t` per position processed. Both loops make
the same number of calls.

```
uncached = calls × c + Σ positions × t
cached   = calls × c + (prompt + new − 1) × t
```

From one 100-token run: **c ≈ 2.1 ms, t ≈ 0.05 ms**. The fixed cost per call is worth about
42 positions of work, so on this small CPU model the cache cannot remove most of the time.