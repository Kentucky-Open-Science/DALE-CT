#!/usr/bin/env python
"""
Ablation Study Runner: Linear Probe Grid Search Across All Embedding Methods
===========================================================================

Iterates over each preprocessing method's embedding subdirectory, generates
a per-method linear_probe.yaml config on-the-fly, and runs train_gridsearch.py.

This lets you answer: "Which inference strategy (body crop, resize target,
tiling) produces CLS tokens that yield the best downstream classification
performance?"

Output Structure:
  {save_root}/
    {method_name}/
      average/
        best_model.pth
        best_thresholds.npy
      max/
        best_model.pth
        best_thresholds.npy
      learned_attention/
        best_model.pth
        best_thresholds.npy
      global_best_model.pth

Usage:
  # Sequential (all methods on one GPU):
  python scripts/run_ablation_gridsearch.py \
      --config configs/linear_probe_ablation.yaml

  # Parallel via Slurm job array (recommended for cluster):
  sbatch run_ablation_gridsearch.sh

  # Optional: run only specific methods
  python scripts/run_ablation_gridsearch.py \
      --config configs/linear_probe_ablation.yaml \
      --methods cropped_256 raw_resize_256

  # Optional: skip methods that already have results
  python scripts/run_ablation_gridsearch.py \
      --config configs/linear_probe_ablation.yaml \
      --skip-existing
"""

import argparse
import os
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_per_method_config(ablation_cfg, method_name):
    """
    Build a linear_probe.yaml-compatible config dict for a specific method.

    The embedding directories are:
      train: {train_root}/{method_name}/
      valid: {valid_root}/{method_name}/
    """
    emb_cfg = ablation_cfg["embeddings"]
    exp_cfg = ablation_cfg["experiment"]

    train_emb_dir = os.path.join(emb_cfg["train_root"], method_name)
    valid_emb_dir = os.path.join(emb_cfg["valid_root"], method_name)
    save_dir = os.path.join(exp_cfg["save_root"], method_name)

    per_method = {
        "data": {
            "train_label_path": ablation_cfg["data"]["train_label_path"],
            "val_label_path": ablation_cfg["data"]["val_label_path"],
            "train_embedding_dir": train_emb_dir,
            "val_embedding_dir": valid_emb_dir,
        },
        "experiment": {
            "seed": exp_cfg["seed"],
            "total_steps": exp_cfg["total_steps"],
            "eval_freq": exp_cfg["eval_freq"],
            "batch_size": exp_cfg["batch_size"],
            "input_dim": exp_cfg["input_dim"],
            "num_classes": exp_cfg["num_classes"],
            "save_dir": save_dir,
            "pooling_mode": exp_cfg.get("pooling_mode", "embedding"),
            "pooling_schemes": exp_cfg["pooling_schemes"],
            "learning_rates": exp_cfg["learning_rates"],
            "ct_rate_classes": ablation_cfg.get("ct_rate_classes", []),
        },
        "wandb": {
            "project": ablation_cfg["wandb"]["project"],
            "group": f"{ablation_cfg['wandb']['group']}_{method_name}",
            "mode": ablation_cfg["wandb"].get("mode", "offline"),
        },
    }
    return per_method


def method_has_results(save_dir):
    """Check if a method already has a global_best_model.pth (completed)."""
    marker = os.path.join(save_dir, "global_best_model.pth")
    return os.path.exists(marker)


def main():
    parser = argparse.ArgumentParser(
        description="Run linear probe grid search across all embedding methods"
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/linear_probe_ablation.yaml",
        help="Path to the ablation master config"
    )
    parser.add_argument(
        "--methods", type=str, nargs="*",
        default=None,
        help="Specific methods to run (default: all from config)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip methods that already have a global_best_model.pth"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be run without executing"
    )
    args = parser.parse_args()

    ablation_cfg = load_config(args.config)

    # Determine which methods to run
    if args.methods:
        methods = args.methods
    else:
        methods = ablation_cfg["embeddings"]["methods"]

    print("=" * 70)
    print("ABLATION STUDY: Multi-Method Linear Probe Grid Search")
    print("=" * 70)
    print(f"Train embedding root: {ablation_cfg['embeddings']['train_root']}")
    print(f"Valid embedding root: {ablation_cfg['embeddings']['valid_root']}")
    print(f"Save root:            {ablation_cfg['experiment']['save_root']}")
    print(f"Methods to evaluate:  {methods}")
    print(f"Pooling schemes:      {ablation_cfg['experiment']['pooling_schemes']}")
    print(f"Learning rates:       {ablation_cfg['experiment']['learning_rates']}")
    print(f"Total combos/method:  {len(ablation_cfg['experiment']['pooling_schemes']) * len(ablation_cfg['experiment']['learning_rates'])}")
    print("=" * 70)

    results = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gridsearch_script = os.path.join(script_dir, "..", "train_gridsearch.py")
    gridsearch_script = os.path.abspath(gridsearch_script)

    for method_name in methods:
        print(f"\n{'=' * 70}")
        print(f"METHOD: {method_name}")
        print(f"{'=' * 70}")

        per_method_cfg = build_per_method_config(ablation_cfg, method_name)
        save_dir = per_method_cfg["experiment"]["save_dir"]

        # Check train embedding directory exists
        train_dir = per_method_cfg["data"]["train_embedding_dir"]
        valid_dir = per_method_cfg["data"]["val_embedding_dir"]

        if not os.path.isdir(train_dir):
            print(f"  [SKIP] Train embedding directory not found: {train_dir}")
            results[method_name] = "skipped (missing train dir)"
            continue
        if not os.path.isdir(valid_dir):
            print(f"  [SKIP] Valid embedding directory not found: {valid_dir}")
            results[method_name] = "skipped (missing valid dir)"
            continue

        # Count .npz files
        train_count = len(list(Path(train_dir).glob("*.npz"))) + len(list(Path(train_dir).glob("*.npy")))
        valid_count = len(list(Path(valid_dir).glob("*.npz"))) + len(list(Path(valid_dir).glob("*.npy")))
        print(f"  Train embeddings: {train_count} files in {train_dir}")
        print(f"  Valid embeddings: {valid_count} files in {valid_dir}")

        if args.skip_existing and method_has_results(save_dir):
            print(f"  [SKIP] Already has results at {save_dir}")
            results[method_name] = "skipped (existing results)"
            continue

        # Write per-method config to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix=f"linear_probe_{method_name}_"
        ) as f:
            yaml.dump(per_method_cfg, f, default_flow_style=False, sort_keys=False)
            temp_config_path = f.name

        cmd = [
            sys.executable, gridsearch_script,
            "--config", temp_config_path
        ]

        print(f"  Save dir:  {save_dir}")
        print(f"  Command:   python train_gridsearch.py --config {temp_config_path}")

        if args.dry_run:
            print(f"  [DRY RUN] Would execute: {' '.join(cmd)}")
            os.unlink(temp_config_path)
            results[method_name] = "dry-run"
            continue

        print(f"  Running grid search...")
        try:
            subprocess.run(cmd, check=True)
            results[method_name] = "completed"
            print(f"  [DONE] {method_name} completed successfully.")
        except subprocess.CalledProcessError as e:
            results[method_name] = f"failed (exit code {e.returncode})"
            print(f"  [FAIL] {method_name} failed with exit code {e.returncode}")
        finally:
            # Clean up temp config
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    for method_name, status in results.items():
        print(f"  {method_name:<30s} : {status}")
    print("=" * 70)

    # Print instructions for analyzing results
    print("""
Next steps to analyze results:
  1. Each method's best model is at:
       {save_root}/{method_name}/global_best_model.pth
  2. Per-pooling-scheme best models are at:
       {save_root}/{method_name}/{pooling_scheme}/best_model.pth
  3. Compare AUPRC across methods to find the best inference strategy.
  4. The W&B project "{wandb_project}" contains per-method run logs.
""".format(
        save_root=ablation_cfg["experiment"]["save_root"],
        wandb_project=ablation_cfg["wandb"]["project"]
    ))


if __name__ == "__main__":
    main()
