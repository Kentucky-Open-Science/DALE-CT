#!/usr/bin/env python
"""
Evaluate an already-saved best probe model (best_model.pth) on the test set and
write best_metrics.json — without re-running the LR grid search AND without
loading the (huge) training split.

Use case: a grid search finished (best_model.pth was saved right after the LR
loop) but the job was killed (e.g. SLURM TIMEOUT) during the post-grid test
evaluation, so best_metrics.json was never written. The full
load_embeddings_and_labels path loads ALL three splits (train+val+test) into
RAM — for TS-Patch that's ~38M patch samples (~170GB), which itself exceeds a
short job's time/memory budget. Since evaluate_probe_detailed computes the
headline macro_auroc/macro_auprc from the eval set alone (train_y/val_y are only
used for prevalence reporting in the per-class detail), this script loads ONLY
val+test, cutting the load to ~12M samples.

Faithfulness: the per-volume load + reshape logic is copied verbatim from
load_embeddings_and_labels (same transpose/reshape for [D,C,G,G]->[D*G*G,C],
same patch_labels_{model_key} key, same row-count validation), and
evaluate_probe_detailed / PatchLinearProbe are imported from the probe module
so the metric computation is identical. train_y/val_y are passed as None (val_y
for test eval is None here — only prevalence detail is dropped, not the metrics).

Usage (in-container):
    python -u scripts/eval_saved_probe.py \
        --config /app/project/.../configs/linear_probe_model_comparison_ts.yaml \
        --model lejepa_1s_v2 --probe patch \
        --best-lr 0.1 --best-val-metric 0.1105
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_model_comparison_probes import (  # noqa: E402
    load_config,
    evaluate_probe_detailed,
    PatchLinearProbe,
    CLSLinearProbe,
)


def _load_split_only(emb_dir, gt_dir, volume_names, probe_type, model_key, split_label):
    """Load embeddings+labels for ONE split. Logic copied verbatim from
    load_embeddings_and_labels._load_split (same reshape/transpose/key logic)."""
    all_X, all_y = [], []
    for name in tqdm(volume_names, desc=f"Loading {split_label} {probe_type}"):
        emb_data = np.load(str(emb_dir / f"{name}.npz"))
        gt_data = np.load(str(gt_dir / f"{name}.nii.gz.npz"))

        if probe_type == "cls":
            X = emb_data["cls"]
            y = gt_data["cls_labels"]
        else:
            X = emb_data["patch"]
            patch_key = f"patch_labels_{model_key}"
            if patch_key in gt_data:
                y = gt_data[patch_key]
            else:
                patch_keys = [k for k in gt_data.keys() if k.startswith("patch_labels")]
                if patch_keys:
                    y = gt_data[patch_keys[0]]
                else:
                    continue

        if probe_type == "cls":
            X_flat = X.reshape(-1, X.shape[-1])
            y_flat = y if y.ndim == 1 else y.reshape(-1, y.shape[-1])
        else:
            X_flat = X.reshape(-1, X.shape[-1])
            if y.ndim == 4:
                y_flat = y.transpose(0, 2, 3, 1).reshape(-1, y.shape[1])
            elif y.ndim == 3:
                y_flat = y.transpose(0, 2, 1).reshape(-1, y.shape[1])
            elif y.ndim == 2:
                y_flat = y.reshape(-1)
            else:
                y_flat = y.reshape(-1, y.shape[-1])

        if X_flat.shape[0] != y_flat.shape[0]:
            print(f"  [WARN] {name}: X rows ({X_flat.shape[0]}) != y rows ({y_flat.shape[0]}), skipping")
            continue
        all_X.append(X_flat)
        all_y.append(y_flat)

    if not all_X:
        return np.array([]), np.array([])
    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True, choices=["cls", "patch"])
    ap.add_argument("--best-lr", type=float, default=None)
    ap.add_argument("--best-val-metric", type=float, default=None)
    ap.add_argument("--skip-val", action="store_true",
                    help="Skip val eval (test-only); saves time if only test metrics are needed")
    args = ap.parse_args()

    config = load_config(args.config)
    exp_cfg = config["experiment"]
    emb_cfg = config["embeddings"]
    gt_cfg = config["groundtruth"]
    task = config["task"]
    task_type = config["task_type"]
    num_classes = config["num_classes"]
    model_key, probe_type = args.model, args.probe

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task: {task} | Model: {model_key} | Probe: {probe_type}")

    emb_dir = Path(emb_cfg["root"]) / model_key / emb_cfg["method"]
    gt_dir = Path(gt_cfg["root"]) / gt_cfg["task"]
    split_file = os.path.join(gt_cfg["root"], gt_cfg.get("split_file", "patient_splits.json"))
    with open(split_file) as f:
        sa = json.load(f)

    def _base(v):
        return v.replace(".nii.gz", "").replace(".npy", "")

    val_bases = {_base(v) for v in sa["val"]}
    test_bases = {_base(v) for v in sa["test"]}
    emb_names = {f.stem for f in emb_dir.glob("*.npz")}
    gt_names = {f.stem.replace(".nii.gz", "") for f in gt_dir.glob("*.npz")}
    common = sorted(emb_names & gt_names)
    val_names = [n for n in common if n in val_bases]
    test_names = [n for n in common if n in test_bases]
    print(f"Found {len(common)} matching volumes | Val: {len(val_names)} | Test: {len(test_names)}")

    save_dir = os.path.join(exp_cfg["save_root"], model_key, probe_type)
    best_model_path = os.path.join(save_dir, "best_model.pth")
    results_path = os.path.join(save_dir, "best_metrics.json")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"best_model.pth not found: {best_model_path}")

    # --- Load ONLY val+test (skip the huge train split) ---
    val_X, val_y = (None, None)
    if not args.skip_val:
        val_X, val_y = _load_split_only(emb_dir, gt_dir, val_names, probe_type, model_key, "val")
    test_X, test_y = _load_split_only(emb_dir, gt_dir, test_names, probe_type, model_key, "test")
    print(f"Loaded — Val: {val_X.shape[0] if val_X is not None else 0} | Test: {test_X.shape[0]}")

    # --- Rebuild probe + load saved best weights ---
    if probe_type == "cls":
        best_model = CLSLinearProbe(exp_cfg["input_dim"], num_classes, task_type)
    else:
        best_model = PatchLinearProbe(exp_cfg["input_dim"], num_classes, task_type)
    best_model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
    best_model = best_model.to(device)

    class_names = config.get("rex_classes", None) or config.get("ts_classes", None)
    min_pos = exp_cfg.get("min_positive_samples", 5)

    val_detailed = {}
    if not args.skip_val and len(val_X) > 0:
        val_X_t = torch.from_numpy(val_X).float()
        val_y_t = torch.from_numpy(val_y).float() if task_type == "multi_label" else torch.from_numpy(val_y).long()
        print("Evaluating on validation set (detailed)...")
        # train_y/val_y=None: only per-class prevalence detail is dropped, not the metrics.
        val_detailed = evaluate_probe_detailed(
            best_model, val_X_t, val_y_t,
            batch_size=exp_cfg["batch_size"], task_type=task_type, device=device,
            class_names=class_names, train_y=None, val_y=None,
            min_positive_samples=min_pos,
        )

    test_metrics = {}
    if len(test_X) > 0:
        test_X_t = torch.from_numpy(test_X).float()
        test_y_t = torch.from_numpy(test_y).float() if task_type == "multi_label" else torch.from_numpy(test_y).long()
        print("Evaluating on test set (detailed)...")
        test_metrics = evaluate_probe_detailed(
            best_model, test_X_t, test_y_t,
            batch_size=exp_cfg["batch_size"], task_type=task_type, device=device,
            class_names=class_names, train_y=None, val_y=None,
            min_positive_samples=min_pos,
        )
        print(f"  Test metrics: { {k: v for k, v in test_metrics.items() if k != 'per_class'} }")

    lr_results = {}
    if args.best_lr is not None:
        lr_results[str(args.best_lr)] = {"val_metric": float(args.best_val_metric) if args.best_val_metric is not None else None}

    results = {
        "model": model_key, "probe_type": probe_type, "task": task,
        "best_lr": args.best_lr,
        "best_val_metric": float(args.best_val_metric) if args.best_val_metric is not None else None,
        "lr_results": lr_results,
        "val_detailed_metrics": val_detailed,
        "test_metrics": test_metrics,
        "n_train": None, "n_val": int(len(val_X)) if val_X is not None else 0, "n_test": int(len(test_X)),
        "note": "Eval-only from saved best_model.pth; LR grid + train split skipped. Metrics identical to in-grid test eval (train_y/val_y=None only drops prevalence detail).",
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
