#!/usr/bin/env python
"""
Visualize Ablation Results
==========================

Reads the JSON files produced by parse_ablation_logs.py from
outputs/ablation_linear_probe/{method_name}/test_metrics.json and generates:

  1. Summary table (console) — macro metrics for all methods
  2. Macro AUPRC bar chart — comparing all methods
  3. Per-class AUPRC heatmap — methods × classes
  4. Grid search heatmap — pooling vs LR for each method
  5. Per-class AUPRC grouped bar chart — all methods side-by-side per class

All plots are saved to outputs/ablation_linear_probe/figures/.

Usage:
  python scripts/visualize_ablation_results.py \
      --results-dir /project/ibi-staff/CT-JEPA/public/outputs/ablation_linear_probe

  # Only print the summary table, no plots
  python scripts/visualize_ablation_results.py --table-only
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation if matplotlib is missing
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Pretty method names for display
# ---------------------------------------------------------------------------
METHOD_DISPLAY_NAMES = {
    "full_resolution":       "Full Resolution",
    "raw_resize_256":        "Raw Resize 256",
    "raw_resize_144":        "Raw Resize 144",
    "cropped_patch_aligned": "Cropped Patch-Aligned",
    "cropped_256":           "Cropped 256",
    "cropped_144":           "Cropped 144",
    "tiled_5cut":            "Tiled 5-Cut",
}


def display_name(method):
    return METHOD_DISPLAY_NAMES.get(method, method.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(results_dir):
    """Load all test_metrics.json files from subdirectories."""
    root = Path(results_dir)
    if not root.is_dir():
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    results = {}
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        json_path = subdir / "test_metrics.json"
        if not json_path.is_file():
            continue
        with open(json_path, "r") as f:
            data = json.load(f)
        method = data.get("method", subdir.name)
        results[method] = data

    if not results:
        print(f"ERROR: No test_metrics.json files found under {results_dir}",
              file=sys.stderr)
        sys.exit(1)

    return results


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def print_summary_table(results):
    """Print a rich console table of macro metrics for all methods."""
    # Collect all class names across all methods for a unified order
    all_classes = set()
    for r in results.values():
        all_classes.update(r.get("test_per_class", {}).keys())
    all_classes = sorted(all_classes)

    header = (
        f"{'Method':<28} "
        f"{'Best Pool':<20} "
        f"{'Best LR':<10} "
        f"{'Val AUPRC':>10} "
        f"{'Test AUPRC':>10} "
        f"{'Test AUROC':>10} "
        f"{'Test F1':>10} "
        f"{'Test BA':>10}"
    )
    sep = "=" * len(header)

    print("\n" + sep)
    print("ABLATION STUDY — MACRO TEST METRICS")
    print(sep)
    print(header)
    print("-" * len(header))

    # Sort by test AUPRC descending
    sorted_methods = sorted(
        results.items(),
        key=lambda kv: kv[1].get("test_macro", {}).get("auprc", 0.0),
        reverse=True,
    )

    for method, data in sorted_methods:
        gs = data.get("grid_search", {})
        tm = data.get("test_macro", {})
        print(
            f"{display_name(method):<28} "
            f"{gs.get('best_pooling', 'N/A'):<20} "
            f"{str(gs.get('best_lr', 'N/A')):<10} "
            f"{gs.get('best_val_auprc', 0.0):>10.4f} "
            f"{tm.get('auprc', 0.0):>10.4f} "
            f"{tm.get('auroc', 0.0):>10.4f} "
            f"{tm.get('macro_f1', 0.0):>10.4f} "
            f"{tm.get('balanced_accuracy', 0.0):>10.4f}"
        )

    print(sep)

    # --- Per-class table ---
    print(f"\n{'=' * 100}")
    print("PER-CLASS AUPRC ACROSS METHODS")
    print(f"{'=' * 100}")

    # Header: Method names
    col_width = 32
    print(f"{'Class':<{col_width}}", end="")
    for method, _ in sorted_methods:
        print(f"{display_name(method):>22}", end="")
    print()
    print("-" * 100)

    for cls_name in all_classes:
        print(f"{cls_name:<{col_width}}", end="")
        for method, data in sorted_methods:
            pc = data.get("test_per_class", {}).get(cls_name, {})
            auprc = pc.get("auprc", float("nan"))
            if np.isnan(auprc):
                print(f"{'N/A':>22}", end="")
            else:
                print(f"{auprc:>22.4f}", end="")
        print()

    print("=" * 100)


# ---------------------------------------------------------------------------
# Plot 1: Macro AUPRC bar chart
# ---------------------------------------------------------------------------

def plot_macro_auprc_bars(results, out_dir):
    """Grouped bar chart: macro AUPRC + AUROC + F1 + BA per method."""
    methods = sorted(results.keys(),
                     key=lambda m: results[m].get("test_macro", {}).get("auprc", 0.0),
                     reverse=True)
    labels = [display_name(m) for m in methods]

    metrics = ["auprc", "auroc", "macro_f1", "balanced_accuracy"]
    metric_labels = ["AUPRC", "AUROC", "Macro F1", "Balanced Acc"]

    data_matrix = []
    for m in methods:
        tm = results[m].get("test_macro", {})
        data_matrix.append([tm.get(k, 0.0) for k in metrics])
    data_matrix = np.array(data_matrix)

    x = np.arange(len(labels))
    n_metrics = len(metrics)
    width = 0.8 / n_metrics
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.6), 6))

    for i in range(n_metrics):
        offset = (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(x + offset, data_matrix[:, i], width,
                      label=metric_labels[i], color=colors[i],
                      edgecolor="white", linewidth=0.5)
        # Annotate values on bars
        for bar, val in zip(bars, data_matrix[:, i]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study — Macro Test Metrics by Method")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, "macro_metrics_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2: Per-class AUPRC heatmap
# ---------------------------------------------------------------------------

def plot_per_class_heatmap(results, out_dir):
    """Heatmap: methods (rows) × classes (columns), colored by AUPRC."""
    # Collect all classes
    all_classes = set()
    for r in results.values():
        all_classes.update(r.get("test_per_class", {}).keys())
    all_classes = sorted(all_classes)

    methods = sorted(results.keys(),
                     key=lambda m: results[m].get("test_macro", {}).get("auprc", 0.0),
                     reverse=True)

    # Build matrix
    matrix = np.full((len(methods), len(all_classes)), np.nan)
    for i, method in enumerate(methods):
        pc = results[method].get("test_per_class", {})
        for j, cls_name in enumerate(all_classes):
            matrix[i, j] = pc.get(cls_name, {}).get("auprc", np.nan)

    fig, ax = plt.subplots(figsize=(max(14, len(all_classes) * 0.9),
                                    max(5, len(methods) * 0.6)))

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(all_classes)))
    ax.set_xticklabels(all_classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([display_name(m) for m in methods], fontsize=9)

    # Annotate each cell
    for i in range(len(methods)):
        for j in range(len(all_classes)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.55 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("AUPRC")

    ax.set_title("Per-Class AUPRC Heatmap — Methods × Findings")
    plt.tight_layout()

    path = os.path.join(out_dir, "per_class_auprc_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3: Grid search heatmap per method
# ---------------------------------------------------------------------------

def plot_grid_search_heatmaps(results, out_dir):
    """For each method, produce a pooling × LR heatmap of validation AUPRC."""
    methods = sorted(results.keys())

    for method in methods:
        data = results[method]
        runs = data.get("grid_search", {}).get("runs", [])
        if not runs:
            continue

        # Collect unique pooling schemes and LRs
        poolings = sorted(set(r["pooling"] for r in runs))
        lrs = sorted(set(r["lr"] for r in runs))

        # Build matrix
        matrix = np.full((len(poolings), len(lrs)), np.nan)
        for run in runs:
            if run["best_auprc"] is not None:
                pi = poolings.index(run["pooling"])
                li = lrs.index(run["lr"])
                matrix[pi, li] = run["best_auprc"]

        fig, ax = plt.subplots(figsize=(max(5, len(lrs) * 1.2),
                                        max(3, len(poolings) * 0.8)))

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)

        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([str(lr) for lr in lrs], fontsize=9)
        ax.set_yticks(range(len(poolings)))
        ax.set_yticklabels(poolings, fontsize=9)
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Pooling Scheme")

        # Annotate
        for i in range(len(poolings)):
            for j in range(len(lrs)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if val < 0.55 else "black"
                    ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                            fontsize=8, color=text_color, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Val AUPRC")

        ax.set_title(f"Grid Search — {display_name(method)}")
        plt.tight_layout()

        safe_name = method.replace("/", "_")
        path = os.path.join(out_dir, f"grid_search_{safe_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 4: Per-class AUPRC grouped bar chart
# ---------------------------------------------------------------------------

def plot_per_class_grouped_bars(results, out_dir):
    """Grouped bar chart: one group per class, one bar per method."""
    all_classes = set()
    for r in results.values():
        all_classes.update(r.get("test_per_class", {}).keys())
    all_classes = sorted(all_classes)

    methods = sorted(results.keys(),
                     key=lambda m: results[m].get("test_macro", {}).get("auprc", 0.0),
                     reverse=True)

    if not all_classes or not methods:
        return

    # Build matrix: classes × methods
    matrix = np.full((len(all_classes), len(methods)), np.nan)
    for j, method in enumerate(methods):
        pc = results[method].get("test_per_class", {})
        for i, cls_name in enumerate(all_classes):
            matrix[i, j] = pc.get(cls_name, {}).get("auprc", np.nan)

    x = np.arange(len(all_classes))
    n_methods = len(methods)
    width = 0.8 / n_methods

    # Color palette
    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(n_methods)]

    fig, ax = plt.subplots(figsize=(max(16, len(all_classes) * 1.4), 7))

    for j, method in enumerate(methods):
        offset = (j - (n_methods - 1) / 2) * width
        ax.bar(x + offset, matrix[:, j], width,
               label=display_name(method), color=colors[j],
               edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AUPRC")
    ax.set_title("Per-Class AUPRC — All Methods Compared")
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, "per_class_auprc_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 5: Validation vs Test AUPRC scatter
# ---------------------------------------------------------------------------

def plot_val_vs_test_auprc(results, out_dir):
    """Scatter plot: validation AUPRC vs test AUPRC per method."""
    methods = []
    val_auprcs = []
    test_auprcs = []

    for method, data in results.items():
        gs = data.get("grid_search", {})
        tm = data.get("test_macro", {})
        val = gs.get("best_val_auprc")
        test = tm.get("auprc")
        if val is not None and test is not None:
            methods.append(method)
            val_auprcs.append(val)
            test_auprcs.append(test)

    if not methods:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    for i, method in enumerate(methods):
        ax.scatter(val_auprcs[i], test_auprcs[i],
                   color=colors[i], s=100, edgecolors="black", linewidth=0.5,
                   zorder=5)
        ax.annotate(display_name(method),
                    (val_auprcs[i], test_auprcs[i]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=7, alpha=0.9)

    # Diagonal line
    all_vals = val_auprcs + test_auprcs
    mn, mx = min(all_vals) - 0.02, max(all_vals) + 0.02
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.3, label="y = x")

    ax.set_xlabel("Validation AUPRC (best grid search)")
    ax.set_ylabel("Test AUPRC")
    ax.set_title("Validation vs Test AUPRC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    plt.tight_layout()

    path = os.path.join(out_dir, "val_vs_test_auprc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize ablation study results from parsed JSON files"
    )
    parser.add_argument(
        "--results-dir", type=str,
        default="/project/ibi-staff/CT-JEPA/public/outputs/ablation_linear_probe",
        help="Directory containing per-method subdirs with test_metrics.json"
    )
    parser.add_argument(
        "--table-only", action="store_true",
        help="Only print the console summary table (no plots)"
    )
    args = parser.parse_args()

    # --- Load data ---
    results = load_all_results(args.results_dir)
    print(f"Loaded results for {len(results)} method(s): {', '.join(sorted(results.keys()))}")

    # --- Always print the summary table ---
    print_summary_table(results)

    if args.table_only:
        return

    # --- Generate plots ---
    if not HAS_MPL:
        print("\n⚠️  matplotlib not installed. Skipping plots.", file=sys.stderr)
        print("   Install with: pip install matplotlib", file=sys.stderr)
        return

    out_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nGenerating plots → {out_dir}/")

    plot_macro_auprc_bars(results, out_dir)
    plot_per_class_heatmap(results, out_dir)
    plot_grid_search_heatmaps(results, out_dir)
    plot_per_class_grouped_bars(results, out_dir)
    plot_val_vs_test_auprc(results, out_dir)

    print(f"\nDone! All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
