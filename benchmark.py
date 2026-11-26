import torch
import time
import csv
import statistics
from pathlib import Path
from src.kv_gpt import MiniGPT
import os

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = MiniGPT(
    vocab_size=100,
    no_of_components=6,
    hidden_states=128,
    no_of_heads=8,
    mask=True
).to(device)

model.eval()

x = torch.randint(0, 100, (2, 6), device=device)

lengths = [16, 64, 256]
runs = 3

results = []


def synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(length, use_cache):

    synchronize()

    start = time.perf_counter()

    output = model.generate(
        x,
        max_new_tokens=length,
        use_cache=use_cache
    )

    synchronize()

    elapsed = time.perf_counter() - start

    return elapsed, output


# Warm up both execution modes
print("Warming up...")

for mode in [False, True]:

    model.generate(
        x,
        max_new_tokens=16,
        use_cache=mode
    )

synchronize()


# Benchmark
for length in lengths:

    timings = {
        False: [],
        True: []
    }

    print(f"\nNew tokens: {length}")

    for run in range(runs):

        # Alternate execution order to reduce bias
        modes = (
            [False, True]
            if run % 2 == 0
            else [True, False]
        )

        outputs = {}

        for mode in modes:

            elapsed, output = benchmark(length, mode)

            outputs[mode] = output
            timings[mode].append(elapsed)

            results.append({
                "device": str(device),
                "batch_size": x.size(0),
                "prompt_length": x.size(1),
                "new_tokens": length,
                "run": run + 1,
                "use_cache": mode,
                "elapsed_seconds": elapsed,
                "tokens_per_second": (
                    length * x.size(0) / elapsed
                )
            })

            print(
                f"Run {run + 1} | "
                f"Cache: {mode} | "
                f"Time: {elapsed:.4f}s"
            )

        print(
            "Same tokens:",
            torch.equal(outputs[False], outputs[True])
        )

    uncached_mean = statistics.mean(timings[False])
    cached_mean = statistics.mean(timings[True])

    uncached_std = statistics.stdev(timings[False])
    cached_std = statistics.stdev(timings[True])

    speedup = uncached_mean / cached_mean

    print(
        f"Without cache: "
        f"{uncached_mean:.4f} ± {uncached_std:.4f}s"
    )

    print(
        f"With cache: "
        f"{cached_mean:.4f} ± {cached_std:.4f}s"
    )

    print(f"Speedup: {speedup:.2f}x")


# Save every individual experiment
csv_path = Path(__file__).resolve().parent / "experiments.csv"

write_header = (
    not csv_path.exists() or csv_path.stat().st_size == 0
)

with open(csv_path, "a", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    if write_header:
        writer.writeheader()

    writer.writerows(results)

print(f"\nExperiments saved to: {csv_path}")