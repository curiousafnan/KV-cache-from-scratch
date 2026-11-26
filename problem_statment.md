# Day 53 — KV Cache from Scratch: Problem Statement

## The problem

A GPT generates one token at a time. The uncached loop feeds the **whole sequence so far**
back through the model at every step and keeps only the last position's logits. Every
earlier token is pushed through all six layers again, at every step.

With a 6-token prompt and 100 new tokens:

| | Positions processed |
|---|---|
| Uncached (6 + 7 + … + 105) | 5,550 |
| Actually needed (prompt once, then 1 per step) | 105 |

The waste grows with the square of the generation length.

## Why the obvious approach fails

Recomputing is correct but wasteful. Under a causal mask, an earlier token can never see a
later one, so its keys and values in every layer are identical at every step. The uncached
loop recomputes numbers it already had.

## The shape of the fix

Store each layer's K and V for every processed token. At each step, feed only the newest
token, compute its Q, K and V, append its K and V to the cache, and attend over the whole
cache. Q is never stored: only the newest token needs a query.

## How we will know it worked

1. **Equivalence:** cached and uncached greedy generation produce **token-identical**
   output for the same model and prompt. Similar is not enough.
2. **Speed:** the speedup grows with generation length.

## Second problem: running past the context window

The model was trained on 32-token windows, so it has only seen positions 0–31. Generating
past 32 tokens without a limit feeds it positions it never trained on, and nothing errors.
The generation loop needs a sliding window, and the cache must stay correct when the window
slides.

## Requirements

- Per-layer K/V cache, appended each step
- Causal mask that accounts for cached positions
- New token's position = number of tokens before it in the window, not 0
- Sliding window with the same policy in the cached and uncached loops
- Equivalence test across the window boundary
- Speed measured at several lengths, with warmup and 3 runs

## Rule 0 — built vs primitive

| Built today | Used as primitive |
|---|---|
| The cache, the append, the mask offset, the position offset, the eviction/rebuild policy | Day 42 multi-head attention, Day 45 MiniGPT |