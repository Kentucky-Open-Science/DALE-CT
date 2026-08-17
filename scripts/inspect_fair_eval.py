"""Print per-class + macro metrics from a fair_eval_results.npz.

Usage: python scripts/inspect_fair_eval.py <path-to-npz> [npz2 ...]
"""
import sys
import numpy as np

for path in sys.argv[1:]:
    d = np.load(path, allow_pickle=True)
    names = list(d["class_names"])
    pauroc = list(d["per_class_auroc"])
    pauprc = list(d["per_class_auprc"])
    labels = d["labels"]
    macro_auroc = float(d["macro_auroc"])
    macro_auprc = float(d["macro_auprc"])
    n = labels.shape[0]
    n_pos = labels.sum(axis=0)
    prev = n_pos / n

    print(f"\n{'='*78}")
    print(f"{path}")
    print(f"n_volumes={n}  n_classes={labels.shape[1]}")
    print(f"MACRO AUROC={macro_auroc:.4f}  AUPRC={macro_auprc:.4f}")
    print(f"{'='*78}")
    print(f"  {'':>38s} {'prev':>6s} {'nPos':>5s} {'AUROC':>8s} {'AUPRC':>8s}")
    for c, name in enumerate(names):
        print(f"  {str(name):>38s} {prev[c]:6.3f} {int(n_pos[c]):5d} "
              f"{pauroc[c]:8.4f} {pauprc[c]:8.4f}")
    # Sanity: any all-zero / all-one label columns (degenerate)?
    deg = [names[c] for c in range(labels.shape[1])
           if n_pos[c] == 0 or n_pos[c] == n]
    print(f"  degenerate classes: {deg if deg else 'none'}")
