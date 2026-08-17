import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix
)


def evaluate_model(model, loader, criterion, device, label_names):
    """
    Shared evaluation function for MIL models.
    """
    model.eval()
    total_loss = 0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for features, labels, _, mask in tqdm(loader, desc="Evaluating", leave=False):
            # FIX: Move mask to the device alongside features and labels
            features = features.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            # FIX: Pass the mask to the model!
            logits = model(features, mask=mask)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_targets)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {"val_loss": avg_loss}
    macro_scores = {"sens": [], "spec": [], "ppv": [], "npv": [], "f1": []}

    try:
        metrics["val_macro_auc"] = roc_auc_score(y_true, y_prob, average="macro")
        metrics["val_macro_auprc"] = average_precision_score(y_true, y_prob, average="macro")
    except ValueError:
        metrics["val_macro_auc"] = 0.0
        metrics["val_macro_auprc"] = 0.0

    for i, name in enumerate(label_names):
        y_t, y_p_prob, y_p_bin = y_true[:, i], y_prob[:, i], y_pred[:, i]

        # Calculate AUC/AUPRC per class
        try:
            metrics[f"val_auc/{name}"] = roc_auc_score(y_t, y_p_prob) if len(np.unique(y_t)) > 1 else 0.0
            metrics[f"val_auprc/{name}"] = average_precision_score(y_t, y_p_prob) if len(np.unique(y_t)) > 1 else 0.0
        except:
            metrics[f"val_auc/{name}"] = 0.0

        tn, fp, fn, tp = confusion_matrix(y_t, y_p_bin, labels=[0, 1]).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0.0

        metrics[f"val_f1/{name}"] = f1
        for key, val in zip(["sens", "spec", "ppv", "npv", "f1"], [sens, spec, ppv, npv, f1]):
            macro_scores[key].append(val)

    for key in macro_scores:
        metrics[f"val_macro_{key}"] = np.mean(macro_scores[key])

    return metrics