#!/usr/bin/env python
"""Exp G (Table-I-anchored): slice-spacing robustness of the ORIGINAL Table-I probes.

Loads the exact gridsearch winners behind the paper's 2D-vs-depth-aware ablation
table (CT-MIL outputs, May 2026; learned_attention scheme for both arms) and
re-evaluates them on the same full CT-RATE valid test set with test bags
subsampled at stride k in {1,2,4,8} (offset 0). k=1 must byte-reproduce the
original logged test metrics (depth-aware 0.4631/0.7838/0.4872/0.6932; pure-2D
0.4647/0.7826/0.4920/0.6930), validating the harness; k>1 measures robustness.

Runs inside the lejepa container with /project mounted at /app/project.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CTMIL = "/app/project/ibi-staff/CT-MIL/src"
sys.path.insert(0, CTMIL)

from dataloaders.dataloader_embeddings import CTScanMILDataset, collate_mil_bags  # noqa: E402
from models.colipri_pooling import ColipriProber  # noqa: E402
from train_gridsearch import evaluate  # noqa: E402

VAL_CSV = "/app/project/ibi-staff/CT-JEPA/Process_CT-RATE/dataset/valid_predicted_labels.csv"
OUT = "/app/project/ibi-staff/CT-JEPA/public/outputs/exp_g_stride_tableI"

CLASSES = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]

ARMS = {
    "depth_aware_25d": {
        "emb": "/app/project/ibi-staff/CT-JEPA/features/ctrate/lejepa_base_pretrained/valid_cls",
        "probe_dir": "/app/project/ibi-staff/CT-MIL/outputs/colipri_gridsearch_lejepa_base_pretrained/learned_attention",
        "expect": {"auprc": 0.4631, "auroc": 0.7838, "f1": 0.4872, "ba": 0.6932},
    },
    "pure_2d": {
        "emb": "/app/project/ibi-staff/CT-JEPA/features/ctrate/lejepa_base_pretrained_2d/valid_cls",
        "probe_dir": "/app/project/ibi-staff/CT-MIL/outputs/colipri_gridsearch_lejepa_base_pretrained_2d_2/learned_attention",
        "expect": {"auprc": 0.4647, "auroc": 0.7826, "f1": 0.4920, "ba": 0.6930},
    },
}
STRIDES = [1, 2, 4, 8]


def make_collate(k):
    def _collate(batch):
        return collate_mil_bags([(f[::k], l, n) for f, l, n in batch])
    return _collate


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv(VAL_CSV)
    criterion = nn.BCEWithLogitsLoss()
    results = {}
    for arm, spec in ARMS.items():
        print(f"\n===== {arm} =====", flush=True)
        ds = CTScanMILDataset(test_df, spec["emb"], CLASSES)
        model = ColipriProber(input_dim=768, num_classes=len(CLASSES),
                              pooling_scheme="learned_attention",
                              pooling_mode="embedding").to(device)
        state = torch.load(os.path.join(spec["probe_dir"], "best_model.pth"),
                           map_location=device)
        model.load_state_dict(state)
        thresholds = np.load(os.path.join(spec["probe_dir"], "best_thresholds.npy"))
        results[arm] = {}
        for k in STRIDES:
            loader = DataLoader(ds, batch_size=16, shuffle=False,
                                collate_fn=make_collate(k), num_workers=2)
            m = evaluate(model, loader, criterion, device, CLASSES,
                         fixed_thresholds=thresholds)
            row = {"auprc": float(m["val_macro_auprc"]), "auroc": float(m["val_macro_auc"]),
                   "f1": float(m["val_macro_f1"]), "ba": float(m["val_macro_ba"])}
            results[arm][f"k{k}"] = row
            print(f"  k={k}: " + " ".join(f"{a}={v:.4f}" for a, v in row.items()), flush=True)
        exp = spec["expect"]
        got = results[arm]["k1"]
        repro = all(abs(got[a] - exp[a]) < 5e-4 for a in exp)
        results[arm]["k1_reproduces_tableI"] = bool(repro)
        print(f"  k=1 reproduces Table I: {repro} (expected {exp})", flush=True)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}/summary.json", flush=True)


if __name__ == "__main__":
    main()
