# Day 53 — Learnings

## Key concepts

**Why K and V are cached and Q is not.** Future queries reuse every earlier token's K and V.
A query is only needed for the current token. Earlier tokens asked their questions in their
own steps, and those answers are already used.

**Why the cache needs a causal mask.** Under the mask, an earlier token never sees a later one,
so adding a token changes nothing about earlier tokens in any layer. Without the mask (BERT),
token 1 would see the new token, its outputs from layer 2 on would change, and its cached
K and V would be wrong.

**Block size = context window.** The 32 lives in the data, not the model: training windows
were 32 tokens, so the model only learned positions 0–31. Past that, sinusoidal encoding
produces positions without error — a silent failure. (A learned position table, as in GPT-2,
would fail loudly with an index error.)

**The cache is a speed trick, not a capability.** Cached and uncached output are identical,
so the cache cannot let the model go past its context window.

## Results

### Equivalence

Token-identical output, cached vs uncached, at 16, 64, 100 and 256 new tokens, and at
100 new tokens with sliding windows (several slides).

### Speedup sweep (before the window was added)

Random-initialized MiniGPT (6 layers, d = 128, 8 heads, vocab 100), CPU, batch 2, 6-token
prompt. 3 runs each, mean ± standard deviation. Predictions from the c/t model.

| New tokens | Uncached | Cached | Speedup | Predicted |
|---|---|---|---|---|
| 16 | 43.4 ± 1.0 ms | 34.3 ± 3.2 ms | 1.27× | 1.28× |
| 64 | 256.9 ± 8.5 ms | 147.8 ± 14.2 ms | 1.74× | 1.85× |
| 256 | 2,608.5 ± 60.8 ms | 588.5 ± 22.1 ms | 4.43× | 4.1× |

The speedups are far outside the run-to-run spread, so they are real.

### Why 4.4× and not ~131×

The position ratio at 256 tokens is 34,176 ÷ 261 ≈ 131×, but that only counts arithmetic.
Both loops make the same number of forward calls, and each call has a fixed cost (~2.1 ms)
worth about 42 positions of work (~0.05 ms each). The cache removes positions, not calls.
A bigger model makes each position cost more, so the fixed cost matters less and the speedup
moves toward the position ratio.

The model underestimates the uncached time at 256 by about 16%: attention gets more expensive
as the sequence gets longer, and the model treats every position as costing the same.

### With the sliding window (100 new tokens, drop 16)

| Version | Speedup | Same tokens |
|---|---|---|
| Rebuild every step once full | 1.12× | True |
| Rebuild once per slide | 1.84× | True |

Both are **single runs, no warmup, uncached first** — the uncached time includes startup cost.
Not yet measured properly.

## Bugs and how they were found

1. **Comparing against the wrong reference.** The first windowed version: uncached slid by one
   token per step, cached dropped a chunk. Different windows → different tokens. Caught by the
   equivalence test returning `False`.
2. **Trimming instead of rebuilding.** Kept cache entries still carried the influence of the
   dropped tokens in layers 2+. The equivalence test returned `False`; reasoning about which
   tokens each layer could see found the cause.
3. **Positions outside the trained range.** An early version kept every token's absolute index,
   so after 100 steps the newest token was at position 105. Both loops did it, so the test
   passed. Found by tracing the position of the newest token, not by a test.
4. **Rebuilding every step.** The `>= context_window` condition stayed true once the cache was
   full, so the rebuild ran on every step (1.12×). `drop_cache` was accepted but never read —
   Rule 7, item 1.

## Surprises

- The speedup was much smaller than the position ratio, and two numbers (c and t) explained it.
- Removing a token from the cache does not remove its influence on the other tokens' cached
  values.

## Process notes

- Predictions were written for the 256-token case. **Predictions for 16 and 64 were not
  written before running.**
- **Tool-assisted:** the model and the caching experiments were done by me. ChatGPT helped
  with debugging, positional encoding, causal masking, cache reconstruction and the benchmark
  script. Day 53 was flagged as a blind build; with that assistance on the core, it does **not**
  count as a completed blind build.

## Hypothesis for later

Reusing the stale cache after a slide carries information from dropped tokens. It might help
or hurt this model, which was never trained that way. Untested.

## Open items

- Run the trained Day 52 model past 32 tokens and read the output
- Rerun the benchmark sweep with the windowed `generate`
- `tests/`: equivalence at 10 / 40 / 100 new tokens; model input never longer than 32
- `experiments.csv`: match the Rule 16 columns (date, day, git hash, seed)
- Move the cached attention into `deepbuild/` (Rule 0)