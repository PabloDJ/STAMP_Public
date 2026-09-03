#!/usr/bin/env python3
"""Plot same-node STAMP/VeraGrid and VeraGrid scaling benchmarks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "STAMP/02_results/comparison"
OUTPUT = RESULTS / "WSCC_SSA_runtime_benchmark.png"


def median_iqr(values):
    values = np.asarray(values)
    return (np.median(values),
            np.quantile(values, 0.75) - np.median(values),
            np.median(values) - np.quantile(values, 0.25))


def main() -> None:
    stamp = pd.read_csv(RESULTS / "benchmark_stamp_multivac.csv")
    veragrid = pd.read_csv(RESULTS / "benchmark_veragrid_multivac.csv")
    scaling = pd.read_csv(RESULTS / "benchmark_veragrid_scaling_multivac.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    labels = ["STAMP\neig(88×88)", "VeraGrid\neig(88×88)",
              "VeraGrid\ndense SSA", "VeraGrid\nsparse SSA\nk=10"]
    samples = [stamp.eig_seconds, veragrid.eig_seconds,
               veragrid.dense_ssa_seconds, veragrid.sparse_ssa_seconds]
    colors = ["#5875b7", "#4b9b82", "#8b6db1", "#dc785d"]
    stats = [median_iqr(values) for values in samples]
    medians = [item[0] * 1000 for item in stats]
    errors = np.asarray([[item[2] * 1000 for item in stats],
                         [item[1] * 1000 for item in stats]])
    axes[0].bar(np.arange(4), medians, yerr=errors, color=colors,
                capsize=4, width=0.68)
    axes[0].set_xticks(np.arange(4), labels)
    axes[0].set_ylabel("Median runtime [ms]")
    axes[0].set_title("WSCC same-node benchmark")
    axes[0].grid(axis="y", alpha=0.3)
    for index, value in enumerate(medians):
        axes[0].text(index, value + max(medians) * 0.025, f"{value:.2f}",
                     ha="center", va="bottom", fontsize=9)

    grouped = scaling.groupby("dynamic_states")
    states = np.asarray(sorted(grouped.groups))
    dense = np.asarray([grouped.get_group(n).dense_seconds.median() for n in states])
    sparse = np.asarray([grouped.get_group(n).sparse_seconds.median() for n in states])
    axes[1].plot(states, dense * 1000, "o-", label="Dense, all modes", color="#8b6db1")
    axes[1].plot(states, sparse * 1000, "x-", label="Sparse, k=10", color="#dc785d")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xticks(states, [str(value) for value in states])
    axes[1].set_xlabel("Dynamic states (replicated WSCC systems)")
    axes[1].set_ylabel("Median runtime [ms, log scale]")
    axes[1].set_title("VeraGrid SSA scaling")
    axes[1].legend(frameon=False)
    axes[1].grid(which="both", alpha=0.3)

    fig.suptitle("Small-signal runtime on one Multivac CPU core", weight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    print(OUTPUT)


if __name__ == "__main__":
    main()
