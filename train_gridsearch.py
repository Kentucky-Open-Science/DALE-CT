import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
import argparse
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
import wandb

# Import your dataloaders
from dataloaders.dataloader_embeddings import create_datasets, collate_mil_bags
from models.colipri_pooling import ColipriProber


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_infinite_iterator(dataloader):
    """Yields batches indefinitely for step-based training."""
    while True:
        for batch in dataloader:
            yield batch


def get_optimal_f1_thresholds(y_true, y_prob):
    """
    Paper: "For each abnormality, this threshold is selected as the value
    that maximises the F1-score on the internal CT-Rate validation split."
    """
    thresholds_out = []
    for i in range(y_true.shape[1]):
        # Handle edge cases where a class might not be present in the batch/split
        if len(np.unique(y_true[:, i])) < 2:
            thresholds_out.append(0.5)
            continue

        prec, rec, thresholds = precision_recall_curve(y_true[:, i], y_prob[:, i])

        # Calculate F1 for all possible thresholds
        f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-6)

        # Select the threshold that yields the maximum F1
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]
        thresholds_out.append(best_thresh)

    return np.array(thresholds_out)


def evaluate(model, loader, criterion, device, label_names, fixed_thresholds=None):
    model.eval()
    total_loss = 0
    all_targets, all_probs = [], []

    with torch.no_grad():
        for features, labels, _, mask in tqdm(loader, desc="Evaluating", leave=False):
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
            logits, _ = model(features, mask=mask)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_targets)

    if fixed_thresholds is None:
        best_thresholds = get_optimal_f1_thresholds(y_true, y_prob)
    else:
        best_thresholds = fixed_thresholds

    y_pred = (y_prob >= best_thresholds).astype(int)
    metrics = {"val_loss": total_loss / len(loader)}

    try:
        metrics["val_macro_auc"] = roc_auc_score(y_true, y_prob, average="macro")
        metrics["val_macro_auprc"] = average_precision_score(y_true, y_prob, average="macro")
    except ValueError:
        metrics["val_macro_auc"], metrics["val_macro_auprc"] = 0.0, 0.0

    macro_scores = {"f1": [], "ba": []}
    per_class_metrics = {}

    for i, name in enumerate(label_names):
        y_t = y_true[:, i]
        y_p_prob = y_prob[:, i]
        y_p_bin = y_pred[:, i]

        prevalence = y_t.mean()

        if len(np.unique(y_t)) < 2:
            per_class_metrics[name] = {"auroc": 0.0, "auprc": 0.0, "f1": 0.0, "ba": 0.0, "prevalence": prevalence}
            continue

        auc = roc_auc_score(y_t, y_p_prob)
        auprc = average_precision_score(y_t, y_p_prob)

        tn, fp, fn, tp = confusion_matrix(y_t, y_p_bin, labels=[0, 1]).ravel()

        sens = tp / (tp + fn + 1e-6)
        spec = tn / (tn + fp + 1e-6)
        ppv = tp / (tp + fp + 1e-6)

        f1 = 2 * (ppv * sens) / (ppv + sens + 1e-6)
        ba = (sens + spec) / 2.0

        macro_scores["f1"].append(f1)
        macro_scores["ba"].append(ba)

        per_class_metrics[name] = {
            "auroc": auc,
            "auprc": auprc,
            "f1": f1,
            "ba": ba,
            "prevalence": prevalence
        }

    metrics["val_macro_f1"] = np.mean(macro_scores["f1"]) if macro_scores["f1"] else 0.0
    metrics["val_macro_ba"] = np.mean(macro_scores["ba"]) if macro_scores["ba"] else 0.0
    metrics["thresholds"] = best_thresholds
    metrics["per_class"] = per_class_metrics

    return metrics


def run_experiment(config, train_dataset, val_dataset, device):
    best_overall_auprc = 0.0
    best_config_str = ""
    best_pool_scheme = ""

    train_loader = DataLoader(train_dataset, batch_size=config['experiment']['batch_size'], shuffle=True,
                              collate_fn=collate_mil_bags, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['experiment']['batch_size'], shuffle=False,
                            collate_fn=collate_mil_bags, num_workers=2)

    total_steps = config['experiment']['total_steps']
    eval_freq = config['experiment'].get('eval_freq', 2500)

    for pool_scheme in config['experiment']['pooling_schemes']:
        # Create a subdirectory for this specific pooling type
        scheme_dir = os.path.join(config['experiment']['save_dir'], pool_scheme)
        os.makedirs(scheme_dir, exist_ok=True)

        # Track the best AUPRC for THIS pooling scheme across different LRs
        best_scheme_auprc = 0.0

        for lr in config['experiment']['learning_rates']:
            print(f"\n🚀 Starting Run: Pooling=[{pool_scheme}], LR=[{lr}]")
            run = wandb.init(
                project=config['wandb']['project'], group=config['wandb']['group'],
                name=f"{pool_scheme}_lr_{lr}",
                config={"pooling": pool_scheme, "lr": lr, "optimizer": "SGD"},
                reinit=True, mode=config['wandb']['mode']
            )

            model = ColipriProber(
                input_dim=config['experiment']['input_dim'],
                num_classes=config['experiment']['num_classes'],
                pooling_scheme=pool_scheme,
                pooling_mode=config['experiment'].get('pooling_mode', 'embedding')
            ).to(device)

            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.95, weight_decay=0)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

            criterion = nn.BCEWithLogitsLoss()
            best_run_auprc = 0.0

            train_iter = get_infinite_iterator(train_loader)
            model.train()

            running_train_loss = 0.0
            train_loss_steps = 0

            for step in tqdm(range(1, total_steps + 1), desc=f"Training {pool_scheme} LR={lr}"):
                features, labels, _, mask = next(train_iter)
                features, labels, mask = features.to(device), labels.to(device), mask.to(device)

                optimizer.zero_grad()
                logits, _ = model(features, mask=mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                running_train_loss += loss.item()
                train_loss_steps += 1

                # Evaluate and log periodically
                if step % eval_freq == 0 or step == total_steps:
                    val_metrics = evaluate(model, val_loader, criterion, device, val_dataset.label_cols)
                    current_auprc = val_metrics["val_macro_auprc"]
                    current_f1 = val_metrics["val_macro_f1"]
                    avg_train_loss = running_train_loss / train_loss_steps

                    print(f"\nStep {step}/{total_steps} | Train Loss: {avg_train_loss:.4f} | Val AUPRC: {current_auprc:.4f} | Val F1: {current_f1:.4f}")

                    # Reset running loss for the next evaluation window
                    running_train_loss = 0.0
                    train_loss_steps = 0

                    if current_auprc > best_run_auprc:
                        best_run_auprc = current_auprc

                    # 1. Save if it's the best for THIS pooling scheme (regardless of LR)
                    if current_auprc > best_scheme_auprc:
                        best_scheme_auprc = current_auprc
                        torch.save(model.state_dict(), os.path.join(scheme_dir, "best_model.pth"))
                        np.save(os.path.join(scheme_dir, "best_thresholds.npy"), val_metrics["thresholds"])

                    # 2. Keep track of the absolute best for the final summary
                    if current_auprc > best_overall_auprc:
                        best_overall_auprc = current_auprc
                        best_config_str = f"Pooling: {pool_scheme}, LR: {lr}"
                        best_pool_scheme = pool_scheme

                        # Keep a copy of the global winner in the root save_dir
                        torch.save(model.state_dict(),
                                   os.path.join(config['experiment']['save_dir'], "global_best_model.pth"))

                    model.train()

            wandb.finish()
            print(f"✅ Finished {pool_scheme} (LR={lr}) - Best AUPRC: {best_run_auprc:.4f}")

    print("\n" + "=" * 50)
    print(f"🏆 GRID SEARCH COMPLETE")
    print(f"Best Config: {best_config_str}")
    print(f"Best Global AUPRC: {best_overall_auprc:.4f}")
    print("=" * 50)

    return best_pool_scheme


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="gridsearch.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    os.makedirs(config['experiment']['save_dir'], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("⏳ Preparing datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(config)

    # 1. Run the grid search and get the winning architecture
    best_pool_scheme = run_experiment(config, train_dataset, val_dataset, device)

    # 2. --- FINAL TEST SET EVALUATION ---
    print("\n" + "🔥" * 25)
    print("STARTING FINAL EVALUATION ON UNSEEN TEST SET")
    print("🔥" * 25)

    test_loader = DataLoader(test_dataset, batch_size=config['experiment']['batch_size'], shuffle=False,
                             collate_fn=collate_mil_bags, num_workers=2)

    best_model = ColipriProber(
        input_dim=config['experiment']['input_dim'],
        num_classes=config['experiment']['num_classes'],
        pooling_scheme=best_pool_scheme,
        pooling_mode=config['experiment'].get('pooling_mode', 'embedding')
    ).to(device)

    # Load from the specific subdirectory of the global winner
    best_model_path = os.path.join(config['experiment']['save_dir'], best_pool_scheme, "best_model.pth")
    best_thresh_path = os.path.join(config['experiment']['save_dir'], best_pool_scheme, "best_thresholds.npy")

    best_model.load_state_dict(torch.load(best_model_path))
    fixed_thresholds = np.load(best_thresh_path)

    # Use standard unweighted BCE for test evaluation to match
    criterion = nn.BCEWithLogitsLoss()
    test_metrics = evaluate(best_model, test_loader, criterion, device, test_dataset.label_cols,
                            fixed_thresholds=fixed_thresholds)

    # --- Print Macro Averages ---
    print("\n" + "=" * 70)
    print("📊 FINAL TEST METRICS (MACRO AVERAGES) 📊")
    print("=" * 70)
    print(f"{'Metric':<20} | {'Model':<10} | {'Random Guessing Baseline':<30}")
    print("-" * 70)

    avg_prevalence = np.mean([m['prevalence'] for m in test_metrics['per_class'].values()])
    avg_random_f1 = np.mean([p / (p + 0.5) for p in [m['prevalence'] for m in test_metrics['per_class'].values()]])

    print(f"{'AUPRC':<20} | {test_metrics['val_macro_auprc']:.4f}     | ~{avg_prevalence:.4f} (Avg Prevalence)")
    print(f"{'AUROC':<20} | {test_metrics['val_macro_auc']:.4f}     | 0.5000")
    print(f"{'Macro F1':<20} | {test_metrics['val_macro_f1']:.4f}     | ~{avg_random_f1:.4f} (Avg Coin-Flip)")
    print(f"{'Balanced Acc (BA)':<20} | {test_metrics['val_macro_ba']:.4f}     | 0.5000")

    # --- Print Per-Class Performance ---
    print("\n" + "=" * 70)
    print("🔬 PER-CLASS METRICS 🔬")
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