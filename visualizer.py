import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("experiments.csv")

df["use_cache"] = (
    df["use_cache"].astype(str).str.lower() == "true"
)

summary = df.groupby(
    ["new_tokens", "use_cache"]
)["elapsed_seconds"].agg(["mean", "std"]).reset_index()

prompt_length = 6


def positions(n, use_cache):
    if use_cache:
        return prompt_length + n - 1

    return n * prompt_length + n * (n - 1) / 2


# Estimate c and t
X = []
y = []

for _, row in summary.iterrows():

    n = row["new_tokens"]
    p = positions(n, row["use_cache"])

    X.append([n, p])
    y.append(row["mean"])

c, t = np.linalg.lstsq(
    np.array(X),
    np.array(y),
    rcond=None
)[0]

print(f"Overhead per forward pass: {c:.8f}s")
print(f"Estimated cost per token: {t:.8f}s")

lengths = np.linspace(
    summary["new_tokens"].min(),
    summary["new_tokens"].max(),
    200
)

plt.figure(figsize=(10, 6))

for mode in [False, True]:

    data = summary[summary["use_cache"] == mode]

    label = "With cache" if mode else "Without cache"

    # Measured results
    plt.errorbar(
        data["new_tokens"],
        data["mean"],
        yerr=data["std"],
        fmt="o",
        capsize=4,
        label=f"{label} — measured"
    )

    # Predicted results
    prediction = (
        c * lengths + t * positions(lengths, mode)
    )

    plt.plot(
        lengths,
        prediction,
        linestyle="--",
        label=f"{label} — predicted"
    )

plt.xlabel("Number of generated tokens")
plt.ylabel("Execution time (seconds)")
plt.title("KV Cache — Measured vs Predicted Inference Time")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig("kv_cache_benchmark.png", dpi=300)
plt.show()