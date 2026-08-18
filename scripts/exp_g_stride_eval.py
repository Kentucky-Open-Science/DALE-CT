"""Exp G — slice-spacing (stride) robustness of the frozen MIL probe.

Simulates thick-slice / sparse-spacing acquisition at EVALUATION time: each test
bag's per-slice CLS sequence is subsampled with stride k (offset 0) before
pooling, while the probe stays the one trained on native spacing. Degradation
vs k measures how robust each backbone's pooled representation is to protocol
spacing — the invariance depth-aware slab sampling explicitly optimizes for.

Reuses the error-bars protocol verbatim: the saved 5-seed probes + per-seed
thresholds, the same test set, the same compute_metrics, and the same
mean-across-seeds bootstrap CI (shared indices per task seed).

Usage (in-container, GPU or CPU):
  python scripts/exp_g_stride_eval.py --config <error_bars-style yaml> \
      --models vitb_depth_aware,vitb_pure_2d [--strides 1,2,4,8] [--out DIR]

The config's output.root must contain selected_configs/ and probes/ for the
requested models (task ctrate). Results: <out>/{model}_stride{k}.json +
summary.md with seed-mean macro metrics and 95% CIs per stride.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_error_bars import (  # noqa: E402
    load_config, build_config_for_task, build_datasets, label_names_for,
    out_paths, mean_across_seeds_ci, task_seed_for, _read_json, write_json,
)
from train_gridsearch import compute_metrics  # noqa: E402
from models.mil_pooling import build_prober  # noqa: E402
from dataloaders.dataloader_embeddings import collate_mil_bags  # noqa: E402


def strided_batches(test_ds, stride, batch_size=16):
    """Yield collated (features, labels, names, mask) with bags[::stride]."""
    batch = []
    for i in range(len(test_ds)):
        feat, label, name = test_ds[i]
        batch.append((feat[::stride], label, name))
        if len(batch) == batch_size:
            yield collate_mil_bags(batch)
            batch = []
    if batch:
        yield collate_mil_bags(batch)


def infer(model, test_ds, stride, device):
    model.eval()
    probs, trues, names = [], [], []
    with torch.no_grad():
        for features, labels, batch_names, mask in strided_batches(test_ds, stride):
            logits, _ = model(features.to(device), mask=mask.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            trues.append(labels.cpu().numpy())
            names.extend(batch_names)
    return np.vstack(probs), np.vstack(trues), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--strides", default="1,2,4,8")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    op = out_paths(config)
    root = config["output"]["root"]
    if not os.path.isdir(os.path.dirname(root)):   # running on the host, not in-container
        root = root.replace("/app/project", "/project")
    out_dir = args.out or os.path.join(root, "stride_eval")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    strides = [int(s) for s in args.strides.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    names18 = label_names_for(config, "ctrate")
    bs = config["bootstrap"]
    task_seed = task_seed_for(config, "ctrate")

    summary_rows = []
    for model_key in args.models.split(","):
        model_key = model_key.strip()
        sel = _read_json(os.path.join(op["selected_configs"], f"{model_key}_ctrate.json"))
        cfg = build_config_for_task(config, "ctrate", model_key)
        _, _, test_ds = build_datasets(config, model_key, "ctrate")
        print(f"[{model_key}] pooling={sel['pooling']} lr={sel['lr']} "
              f"n_test={len(test_ds)} device={device}")

        # load the 5 seed probes once
        probes, thresholds = {}, {}
        for seed in seeds:
            model = build_prober(input_dim=cfg["experiment"]["input_dim"],
                                 num_classes=cfg["experiment"]["num_classes"],
                                 pooling_scheme=sel["pooling"],
                                 pooling_mode=cfg["experiment"]["pooling_mode"],
                                 config=config).to(device)
            state = torch.load(os.path.join(op["probes"], f"{model_key}_ctrate_seed{seed}.pth"),
                               map_location=device)
            model.load_state_dict(state)
            probes[seed] = model
            thresholds[seed] = np.load(
                os.path.join(op["probes"], f"{model_key}_ctrate_seed{seed}_thresholds.npy"))

        for k in strides:
            seed_arrays, per_seed_macro = [], []
            for seed in seeds:
                y_prob, y_true, _ = infer(probes[seed], test_ds, k, device)
                y_pred = (y_prob >= thresholds[seed]).astype(int)
                m = compute_metrics(y_true, y_prob, y_pred, names18)
                per_seed_macro.append({kk: m[kk] for kk in
                                       ["macro_auprc", "macro_auc", "macro_f1",
                                        "macro_ba"]})
                seed_arrays.append({"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred})
            macro_ci, _ = mean_across_seeds_ci(seed_arrays, names18, bs["n_resamples"],
                                               task_seed, bs["ci_percentiles"])
            mean_macro = {kk: float(np.mean([pm[kk] for pm in per_seed_macro]))
                          for kk in per_seed_macro[0]}
            std_macro = {kk: float(np.std([pm[kk] for pm in per_seed_macro]))
                         for kk in per_seed_macro[0]}
            rec = {"model": model_key, "stride": k, "n_test": int(seed_arrays[0]["y_true"].shape[0]),
                   "pooling": sel["pooling"], "lr": sel["lr"], "seeds": seeds,
                   "macro_mean": mean_macro, "macro_std": std_macro, "macro_ci": macro_ci,
                   "per_seed": per_seed_macro}
            write_json(os.path.join(out_dir, f"{model_key}_stride{k}.json"), rec)
            summary_rows.append(rec)
            print(f"  stride {k}: AUPRC {mean_macro['macro_auprc']:.4f} "
                  f"AUROC {mean_macro['macro_auc']:.4f}")

    with open(os.path.join(out_dir, "summary.md"), "a") as fh:
        fh.write("\n| model | stride | AUPRC (5-seed mean, 95% CI) | AUROC | F1 | BA |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for r in summary_rows:
            ci = r["macro_ci"]
            fh.write(f"| {r['model']} | {r['stride']} "
                     f"| {r['macro_mean']['macro_auprc']:.4f} "
                     f"[{ci['auprc'][0]:.4f},{ci['auprc'][1]:.4f}] "
                     f"| {r['macro_mean']['macro_auc']:.4f} "
                     f"[{ci['auroc'][0]:.4f},{ci['auroc'][1]:.4f}] "
                     f"| {r['macro_mean']['macro_f1']:.4f} "
                     f"| {r['macro_mean']['macro_ba']:.4f} |\n")
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
