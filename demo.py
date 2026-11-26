import torch
import time
from src.kv_gpt import MiniGPT


torch.manual_seed(42)

model = MiniGPT(
    vocab_size=100,
    no_of_components=6,
    hidden_states=128,
    no_of_heads=8,
    mask=True,
    context_window=32
)

model.eval()

x = torch.randint(0, 100, (1, 6))

print("Input:", x.tolist())

# Warmup
for mode in [False, True]:
    model.generate(
        x,
        max_new_tokens=10,
        use_cache=mode,
        drop_cache=16
    )

# Without cache
start = time.perf_counter()

without_cache = model.generate(
    x,
    max_new_tokens=40,
    use_cache=False,
    drop_cache=16
)

without_time = time.perf_counter() - start

# With cache
start = time.perf_counter()

with_cache = model.generate(
    x,
    max_new_tokens=40,
    use_cache=True,
    drop_cache=16
)

with_time = time.perf_counter() - start

print("\nWithout cache:")
print(without_cache.tolist())
print(f"Time: {without_time:.4f}s")

print("\nWith cache:")
print(with_cache.tolist())
print(f"Time: {with_time:.4f}s")

print("\nSpeedup:", without_time / with_time)
print("Same tokens:", torch.equal(without_cache, with_cache))

assert torch.equal(without_cache, with_cache)