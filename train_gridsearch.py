import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
import argparse
import os
from contextlib import nullcontext
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve

# Import your dataloaders
from dataloaders.dataloader_embeddings import create_datasets, collate_mil_bags
from models.colipri_pooling import ColipriProber
from models.mil_pooling import build_prober, cyclical_lambda


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


def compute_metrics(y_true, y_prob, y_pred, label_names):
    """
    Macro + per-class metrics from probability and binary predictions.

    This is the metric block factored out of evaluate() / evaluate_transfer()
    so the error-bars bootstrap can recompute the exact same metrics on
    resampled predictions (no new metric code). Macro AUC/AUPRC are computed
    on the full label matrix (all-or-nothing via try/except, as in the
    original); macro F1/BA average over non-degenerate classes only, using
    the len(np.unique(y_t)) < 2 exclusion rule.
    """
    metrics = {}

    try:
        metrics["macro_auc"] = roc_auc_score(y_true, y_prob, average="macro")
        metrics["macro_auprc"] = average_precision_score(y_true, y_prob, average="macro")
    except ValueError:
        metrics["macro_auc"], metrics["macro_auprc"] = 0.0, 0.0

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

    metrics["macro_f1"] = np.mean(macro_scores["f1"]) if macro_scores["f1"] else 0.0
    metrics["macro_ba"] = np.mean(macro_scores["ba"]) if macro_scores["ba"] else 0.0
    metrics["per_class"] = per_class_metrics

    return metrics


def set_seed(seed):
    """Seed all probe-training randomness (init + shuffle). ColipriProber has no dropout."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate(model, loader, criterion, device, label_names, fixed_thresholds=None, disable_tqdm=False):
    model.eval()
    total_loss = 0
    all_targets, all_probs = [], []

    with torch.no_grad():
        for features, labels, _, mask in tqdm(loader, desc="Evaluating", leave=False, disable=disable_tqdm):
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
    m = compute_metrics(y_true, y_prob, y_pred, label_names)

    metrics = {
        "val_loss": total_loss / len(loader),
        "val_macro_auc": m["macro_auc"],
        "val_macro_auprc": m["macro_auprc"],
        "val_macro_f1": m["macro_f1"],
        "val_macro_ba": m["macro_ba"],
        "thresholds": best_thresholds,
        "per_class": m["per_class"],
    }

    return metrics


def train_one_config(train_loader, val_loader, lr, pooling, config, device, seed, label_names,
                     init_lock=None):
    """
    Train a single probe configuration (one pooling scheme, one LR) with the
    SGD + cosine-annealing protocol from run_experiment, evaluating on
    validation every eval_freq and keeping the best-val-AUPRC step.

    Factored from run_experiment's inner loop so the error-bars pipeline can
    train one fixed config under a controlled seed. Returns the best model
    state (CPU), its validation F1 thresholds (optimized on validation at the
    best step), and the best val macro AUPRC. Performs no saving and no wandb
    logging — the caller decides what to persist.

    init_lock: when training several configs concurrently on one GPU
    (run_select's parallel-LR path), pass a threading.Lock so set_seed +
    ColipriProber construction (which use the *global* RNG) are serialized —
    without it, concurrent configs race on the global generator. The training
    loop itself runs outside the lock. None = sequential (default), unchanged.
    """
    total_steps = config['experiment']['total_steps']
    eval_freq = config['experiment'].get('eval_freq', 2500)
    quiet = init_lock is not None  # concurrent configs: keep logs clean

    # set_seed + model/optimizer construction touch the global RNG (nn.Linear
    # init uses the default generator). Serialize under init_lock when running
    # concurrently so each config sees a clean, un-raced seed.
    with (init_lock or nullcontext()):
        if seed is not None:
            set_seed(seed)

        model = build_prober(
            input_dim=config['experiment']['input_dim'],
            num_classes=config['experiment']['num_classes'],
            pooling_scheme=pooling,
            pooling_mode=config['experiment'].get('pooling_mode', 'embedding'),
            config=config,
        ).to(device)

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.95, weight_decay=0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        criterion = nn.BCEWithLogitsLoss()
        mil_cfg = config.get('mil', {}) if isinstance(config, dict) else {}
        n_cycles = mil_cfg.get('n_cycles', 5)
        max_lambda = mil_cfg.get('max_lambda', 1.0)

    best_val_auprc = -1.0
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_thresholds = None

    train_iter = get_infinite_iterator(train_loader)
    model.train()

    running_train_loss = 0.0
    train_loss_steps = 0

    for step in tqdm(range(1, total_steps + 1), desc=f"Training {pooling} LR={lr}", disable=quiet):
        features, labels, _, mask = next(train_iter)
        features, labels, mask = features.to(device), labels.to(device), mask.to(device)

        optimizer.zero_grad()
        logits, aux = model(features, mask=mask)
        loss = criterion(logits, labels)
        if aux is not None:
            # ProbSA KL term with cyclical annealing (paper: M=5 cycles, 0->1).
            kl_w = cyclical_lambda(step, total_steps, n_cycles, max_lambda)
            loss = loss + kl_w * aux
        loss.backward()
        # Grad clip + non-finite skip: the ProbSA KL (cyclical-annealed, up to
        # weight 1.0) drives the attention logits' gradient hard at high LR ->
        # overflow -> NaN logits -> crash in precision_recall_curve. probsa_diag
        # adds a log(sigma^2) term that can hit log(0) -> +inf. Clip at 1.0
        # (standard for KL/variational training; the ProbSA ref CT-MIL clips too)
        # and skip the step on a non-finite loss so one bad batch can't poison the
        # weights. Stable schemes (average/max/abmil) are unaffected: their grad
        # norms stay well under 1.0, so the clip never engages.
        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_train_loss += loss.item()
        train_loss_steps += 1

        # Evaluate periodically and keep the best-val-AUPRC step
        if step % eval_freq == 0 or step == total_steps:
            val_metrics = evaluate(model, val_loader, criterion, device, label_names, disable_tqdm=quiet)
            current_auprc = val_metrics["val_macro_auprc"]
            current_f1 = val_metrics["val_macro_f1"]
            avg_train_loss = running_train_loss / train_loss_steps

            print(f"\nStep {step}/{total_steps} | Train Loss: {avg_train_loss:.4f} | Val AUPRC: {current_auprc:.4f} | Val F1: {current_f1:.4f}")

            # Reset running loss for the next evaluation window
            running_train_loss = 0.0
            train_loss_steps = 0

            if current_auprc > best_val_auprc:
                best_val_auprc = current_auprc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_thresholds = val_metrics["thresholds"]

            model.train()

    return {
        "best_val_auprc": best_val_auprc,
        "best_model_state": best_model_state,
        "best_thresholds": best_thresholds,
    }


def run_experiment(config, train_dataset, val_dataset, device):
    import wandb  # optional dependency; only needed for the v628 grid-search path

    best_overall_auprc = 0.0
    best_config_str = ""
    best_pool_scheme = ""

    train_loader = DataLoader(train_dataset, batch_size=config['experiment']['batch_size'], shuffle=True,
                              collate_fn=collate_mil_bags, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['experiment']['batch_size'], shuffle=False,
                            collate_fn=collate_mil_bags, num_workers=2)

    label_names = val_dataset.label_cols
    seed = config['experiment'].get('seed', 42)

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

            result = train_one_config(train_loader, val_loader, lr, pool_scheme, config, device, seed, label_names)
            best_run_auprc = result["best_val_auprc"]

            # 1. Save if it's the best for THIS pooling scheme (regardless of LR)
            if best_run_auprc > best_scheme_auprc:
                best_scheme_auprc = best_run_auprc
                torch.save(result["best_model_state"], os.path.join(scheme_dir, "best_model.pth"))
                np.save(os.path.join(scheme_dir, "best_thresholds.npy"), result["best_thresholds"])

            # 2. Keep track of the absolute best for the final summary
            if best_run_auprc > best_overall_auprc:
                best_overall_auprc = best_run_auprc
                best_config_str = f"Pooling: {pool_scheme}, LR: {lr}"
                best_pool_scheme = pool_scheme

                # Keep a copy of the global winner in the root save_dir
                torch.save(result["best_model_state"],
                           os.path.join(config['experiment']['save_dir'], "global_best_model.pth"))

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

    best_model = build_prober(
        input_dim=config['experiment']['input_dim'],
        num_classes=config['experiment']['num_classes'],
        pooling_scheme=best_pool_scheme,
        pooling_mode=config['experiment'].get('pooling_mode', 'embedding'),
        config=config,
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