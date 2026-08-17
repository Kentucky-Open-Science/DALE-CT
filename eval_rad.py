import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
import argparse
import os
from tqdm import tqdm
# Import your dataloaders and model architecture
from dataloaders.dataloader_rad_embeddings import create_datasets, collate_mil_bags
from models.colipri_pooling import ColipriProber
from train_gridsearch import compute_metrics


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def map_ctrate_to_rad(y_prob_18, fixed_thresholds, ct_rate_classes, rad_classes):
    """
    Map 18 CT-RATE-huggingface-downloads probe outputs down to the 16 RAD-ChestCT evaluation classes.

    Calcification = max probability (and OR'd binary prediction) of the two
    CT-RATE-huggingface-downloads calcification labels (Arterial wall calcification, Coronary artery
    wall calcification); all other classes map 1:1. Factored from
    evaluate_transfer so the error-bars bootstrap can reuse the exact mapping.
    Returns (y_prob_16, y_pred_16).
    """
    n = y_prob_18.shape[0]
    y_prob = np.zeros((n, len(rad_classes)), dtype=float)
    y_pred = np.zeros((n, len(rad_classes)), dtype=int)

    for i, rad_cls in enumerate(rad_classes):
        if rad_cls == "Calcification":
            # Paper methodology: use the higher probability between the two calcification labels
            idx_art = ct_rate_classes.index("Arterial wall calcification")
            idx_cor = ct_rate_classes.index("Coronary artery wall calcification")

            # Continuous probability for AUC/AUPRC is the max of the two
            y_prob[:, i] = np.maximum(y_prob_18[:, idx_art], y_prob_18[:, idx_cor])

            # Binary prediction for F1/BA: positive if EITHER threshold is met
            art_pred = y_prob_18[:, idx_art] >= fixed_thresholds[idx_art]
            cor_pred = y_prob_18[:, idx_cor] >= fixed_thresholds[idx_cor]
            y_pred[:, i] = (art_pred | cor_pred).astype(int)
        else:
            # 1-to-1 direct mapping for the remaining classes
            idx = ct_rate_classes.index(rad_cls)
            y_prob[:, i] = y_prob_18[:, idx]
            y_pred[:, i] = (y_prob_18[:, idx] >= fixed_thresholds[idx]).astype(int)

    return y_prob, y_pred


def evaluate_transfer(model, loader, device, ct_rate_classes, rad_classes, fixed_thresholds):
    model.eval()
    all_targets, all_probs = [], []

    with torch.no_grad():
        for features, labels, _, mask in tqdm(loader, desc="Evaluating on RAD-ChestCT", leave=False):
            features, mask = features.to(device), mask.to(device)
            # labels shape: (batch_size, 16)

            logits, _ = model(features, mask=mask)
            probs = torch.sigmoid(logits)  # shape: (batch_size, 18)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    y_prob_18 = np.vstack(all_probs)
    y_true = np.vstack(all_targets)

    y_prob, y_pred = map_ctrate_to_rad(y_prob_18, fixed_thresholds, ct_rate_classes, rad_classes)
    m = compute_metrics(y_true, y_prob, y_pred, rad_classes)

    return {
        "val_macro_auc": m["macro_auc"],
        "val_macro_auprc": m["macro_auprc"],
        "val_macro_f1": m["macro_f1"],
        "val_macro_ba": m["macro_ba"],
        "per_class": m["per_class"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="lejepa_v2.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the RAD-ChestCT dataset
    print("⏳ Loading RAD-ChestCT test dataset...")
    _, _, rad_test_dataset = create_datasets(config)

    rad_test_loader = DataLoader(
        rad_test_dataset,
        batch_size=config['experiment']['batch_size'],
        shuffle=False,
        collate_fn=collate_mil_bags,
        num_workers=2
    )

    ct_rate_classes = config['experiment']['ct_rate_classes']
    rad_chestct_classes = config['experiment']['rad_chestct_classes']
    save_dir = config['experiment']['save_dir']

    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Save directory not found: {save_dir}")

    # 2. Automatically detect all subdirectories (pooling schemes) in the save_dir
    pooling_schemes = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
    pooling_schemes.sort()

    if not pooling_schemes:
        print(f"⚠️ No pooling scheme subdirectories found in {save_dir}")
        exit()

    print(f"🔍 Found {len(pooling_schemes)} pooling schemes to evaluate: {', '.join(pooling_schemes)}")

    # 3. Loop through each discovered pooling scheme
    for scheme in pooling_schemes:
        model_path = os.path.join(save_dir, scheme, "best_model.pth")
        thresholds_path = os.path.join(save_dir, scheme, "best_thresholds.npy")

        # Skip if the necessary files aren't there
        if not os.path.exists(model_path) or not os.path.exists(thresholds_path):
            print(f"\n⏭️ Skipping '{scheme}': Missing best_model.pth or best_thresholds.npy")
            continue

        print("\n" + "🔥" * 30)
        print(f"STARTING TRANSFER EVALUATION: {scheme.upper()}")
        print("🔥" * 30)

        # Initialize Model for this specific scheme
        model = ColipriProber(
            input_dim=config['experiment']['input_dim'],
            num_classes=config['experiment']['num_classes'],
            pooling_scheme=scheme,
            pooling_mode=config['experiment'].get('pooling_mode', 'embedding')
        ).to(device)

        # Load weights and thresholds
        model.load_state_dict(torch.load(model_path, map_location=device))
        fixed_thresholds = np.load(thresholds_path)

        # Evaluate
        test_metrics = evaluate_transfer(
            model,
            rad_test_loader,
            device,
            ct_rate_classes,
            rad_chestct_classes,
            fixed_thresholds
        )

        # Print Macro Averages
        print("\n" + "=" * 70)
        print(f"📊 {scheme.upper()} - TRANSFER METRICS (MACRO AVERAGES) 📊")
        print("=" * 70)
        print(f"{'Metric':<20} | {'Model':<10} | {'Random Guessing Baseline':<30}")
        print("-" * 70)

        avg_prevalence = np.mean([m['prevalence'] for m in test_metrics['per_class'].values()])
        avg_random_f1 = np.mean([p / (p + 0.5) for p in [m['prevalence'] for m in test_metrics['per_class'].values()]])

        print(f"{'AUPRC':<20} | {test_metrics['val_macro_auprc']:.4f}     | ~{avg_prevalence:.4f} (Avg Prevalence)")
        print(f"{'AUROC':<20} | {test_metrics['val_macro_auc']:.4f}     | 0.5000")
        print(f"{'Macro F1':<20} | {test_metrics['val_macro_f1']:.4f}     | ~{avg_random_f1:.4f} (Avg Coin-Flip)")
        print(f"{'Balanced Acc (BA)':<20} | {test_metrics['val_macro_ba']:.4f}     | 0.5000")

        # Print Per-Class Performance
        print("\n" + "=" * 70)
        print(f"🔬 {scheme.upper()} - PER-CLASS TRANSFER METRICS 🔬")
        print("=" * 70)

        for name, m in test_metrics['per_class'].items():
            prev = m['prevalence']
            rand_f1 = prev / (prev + 0.5) if prev > 0 else 0.0

            print(f"🔸 {name.upper()}")
            print(f"   Prevalence: {prev:.4f} ({(prev * 100):.1f}% of test set)")
            print(f"   - AUPRC: {m['auprc']:.4f}  (Random: {prev:.4f})")
            print(f"   - AUROC: {m['auroc']:.4f}  (Random: 0.5000)")
            print(f"   - F1:    {m['f1']:.4f}  (Random: {rand_f1:.4f})")
            print(f"   - BA:    {m['ba']:.4f}  (Random: 0.5000)")
            print("-" * 70)