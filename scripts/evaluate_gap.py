import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve

# --- ADD ACCELERATE ---
from accelerate import Accelerator
from accelerate.utils import gather_object

# --- PATH HACK ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataloaders.datasetloader_ctrate_multiscale import CTMultiScaleDataset, MultiScaleSliceProcessor
from utils.config import load_config, load_model_configs

# Import model architecture and helpers from your training script
from supervised_gap import E2ECTClassifier, variable_depth_collate, center_crop_to_multiple


def get_test_dataset(config):
    """Fetches the test dataset using the new test block in the config."""
    target_config = getattr(config, 'test')
    data_dir = target_config.data_dir
    label_csv = getattr(target_config, 'label_csv', None)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Test data directory not found: {data_dir}")
    return CTMultiScaleDataset(config=config, data_dir=data_dir, label_csv=label_csv)


def get_optimal_f1_thresholds(y_true, y_prob):
    """Calculates best thresholds to maximize F1 per class."""
    thresholds_out = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            thresholds_out.append(0.5)
            continue
        prec, rec, thresholds = precision_recall_curve(y_true[:, i], y_prob[:, i])
        f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-6)
        best_thresh = thresholds[np.argmax(f1_scores)]
        thresholds_out.append(best_thresh)
    return np.array(thresholds_out)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to yaml config")
    args = parser.parse_args()

    config = load_config(args.config_file)

    # 1. Initialization
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    embed_dim = getattr(config.experiment, 'embed_dim', 1024)
    num_classes = getattr(config.experiment, 'num_classes', 18)
    max_slices = getattr(config.experiment, 'max_slices', 256)
    patch_size = load_model_configs(config.train.model_type).patch_size

    if accelerator.is_local_main_process:
        print(f"🚀 Initializing Distributed Evaluation across {accelerator.num_processes} GPUs...")

    # Load Model (Will inherently use the actual accelerator's `is_main_process` inside the init)
    model = E2ECTClassifier(config, accelerator, embed_dim=embed_dim, num_classes=num_classes)

    checkpoint_path = os.path.join(config.output_folders.main_output, "best_loss_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint missing at {checkpoint_path}")

    # Load checkpoint to CPU first to prevent VRAM spikes; accelerator will move it
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))

    # Load Validation Thresholds for real-time printing
    thresholds_path = os.path.join(config.output_folders.main_output, "best_thresholds.npy")
    if os.path.exists(thresholds_path):
        if accelerator.is_local_main_process:
            print("✅ Loaded validation thresholds for live feedback.")
        live_thresholds = np.load(thresholds_path)
    else:
        if accelerator.is_local_main_process:
            print("⚠️ No best_thresholds.npy found. Defaulting to 0.5 for live feedback.")
        live_thresholds = np.full(num_classes, 0.5)

    # 2. Dataset Setup
    test_dataset = get_test_dataset(config)
    class_names = test_dataset.class_names
    test_loader = DataLoader(
        test_dataset,
        batch_size=getattr(config.test, 'batch_size', 4),
        num_workers=getattr(config.test, 'num_workers', 4),
        collate_fn=variable_depth_collate,
        pin_memory=True
    )
    processor = MultiScaleSliceProcessor(config, output_dir=None)

    # Prepare model and dataloader using accelerator (shards the data)
    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()

    # 3. Inference Loop
    all_targets, all_probs = [], []

    if accelerator.is_local_main_process:
        print("\n" + "=" * 60)
        print("STARTING DISTRIBUTED INFERENCE")
        print("=" * 60)

    # Disable tqdm on non-main processes to avoid terminal spam
    for batch in tqdm(test_loader, desc="Testing", disable=not accelerator.is_local_main_process):
        volumes_list, labels, filenames = batch
        labels = labels.to(device)

        primary_views = []
        lengths = []

        for i in range(len(volumes_list)):
            raw_volume = volumes_list[i].squeeze(0).to(device)
            primary_view, _ = processor.process_batch(raw_volume, filename=filenames[i])
            primary_view = center_crop_to_multiple(primary_view, patch_size=patch_size)

            S = primary_view.shape[0]
            lengths.append(min(S, max_slices))

            if S > max_slices:
                indices = torch.linspace(0, S - 1, max_slices).long()
                primary_view = primary_view[indices]
            elif S < max_slices:
                pad_size = max_slices - S
                padding = torch.zeros((pad_size, *primary_view.shape[1:]), dtype=primary_view.dtype, device=device)
                primary_view = torch.cat([primary_view, padding], dim=0)

            primary_views.append(primary_view)

        batched_volumes = torch.stack(primary_views, dim=0)

        with accelerator.autocast():
            logits = model(batched_volumes, lengths=lengths)
            probs = torch.sigmoid(logits)

        # --- DISTRIBUTED GATHER ---
        # `gather_for_metrics` automatically drops dummy padded samples at the end of the dataset
        gathered_probs = accelerator.gather_for_metrics(probs)
        gathered_labels = accelerator.gather_for_metrics(labels)
        gathered_filenames = gather_object(filenames)  # Gather string objects

        # Only the main process tracks the total lists and prints feedback
        if accelerator.is_local_main_process:
            probs_np = gathered_probs.cpu().numpy()
            labels_np = gathered_labels.cpu().numpy()

            all_probs.append(probs_np)
            all_targets.append(labels_np)

            # --- Iteration Level Feedback ---
            # FIX: Iterate over len(probs_np) to ignore the string padding from gather_object
            for b_idx in range(len(probs_np)):
                pred_bin = (probs_np[b_idx] >= live_thresholds).astype(int)
                true_bin = labels_np[b_idx].astype(int)

                correct = [class_names[c] for c in range(num_classes) if pred_bin[c] == 1 and true_bin[c] == 1]
                fp = [class_names[c] for c in range(num_classes) if pred_bin[c] == 1 and true_bin[c] == 0]
                fn = [class_names[c] for c in range(num_classes) if pred_bin[c] == 0 and true_bin[c] == 1]

                tqdm.write(f"📄 Vol: {gathered_filenames[b_idx]}")
                if correct: tqdm.write(f"   ✅ Correct Hits: {', '.join(correct)}")
                if fp:      tqdm.write(f"   ❌ False Positives: {', '.join(fp)}")
                if fn:      tqdm.write(f"   📉 Missed (FN): {', '.join(fn)}")
                tqdm.write("-" * 50)

    # 4. Final Metrics Calculation (Only computed on Main Process)
    if accelerator.is_local_main_process:
        y_prob = np.vstack(all_probs)
        y_true = np.vstack(all_targets)

        print("\n" + "=" * 60)
        print("CALCULATING FINAL METRICS (TEST SET OPTIMIZED)")
        print("=" * 60)

        try:
            macro_auc = roc_auc_score(y_true, y_prob, average="macro")
            macro_auprc = average_precision_score(y_true, y_prob, average="macro")
        except ValueError:
            macro_auc, macro_auprc = 0.0, 0.0

        # Calculate optimal thresholds purely on this test data
        test_optimal_thresholds = get_optimal_f1_thresholds(y_true, y_prob)
        y_pred_opt = (y_prob >= test_optimal_thresholds).astype(int)

        macro_f1, macro_ba = [], []
        for i, name in enumerate(class_names):
            y_t, y_p_bin = y_true[:, i], y_pred_opt[:, i]
            if len(np.unique(y_t)) < 2: continue

            tn, fp, fn, tp = confusion_matrix(y_t, y_p_bin, labels=[0, 1]).ravel()
            sens = tp / (tp + fn + 1e-6)
            spec = tn / (tn + fp + 1e-6)
            ppv = tp / (tp + fp + 1e-6)

            f1 = 2 * (ppv * sens) / (ppv + sens + 1e-6)
            ba = (sens + spec) / 2.0

            macro_f1.append(f1)
            macro_ba.append(ba)

        final_f1 = np.mean(macro_f1) if macro_f1 else 0.0
        final_ba = np.mean(macro_ba) if macro_ba else 0.0

        print(f"📊 Global Macro AUROC: {macro_auc:.4f}")
        print(f"📊 Global Macro AUPRC: {macro_auprc:.4f}")
        print(f"📊 Best Macro F1 (Test-Optimized): {final_f1:.4f}")
        print(f"📊 Balanced Accuracy (at Best F1 Thresholds): {final_ba:.4f}\n")

        print("🧠 Per-Class Optimal Test Thresholds:")
        for i, name in enumerate(class_names):
            print(f"   - {name}: {test_optimal_thresholds[i]:.4f}")


if __name__ == "__main__":
    main()