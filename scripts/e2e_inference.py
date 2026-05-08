"""
End-to-End LoRA + Colipri Inference Script
==========================================
Loads the best model checkpoint from e2e_ft.yaml training and runs inference on:
  1. CT-RATE full validation set (unique patients)
  2. RAD-ChestCT full dataset

For RAD-ChestCT, follows the protocol:
  - Excludes "Mosaic attenuation pattern" (not present in RAD-ChestCT)
  - Maps "Calcification" = max("Arterial wall calcification", "Coronary artery wall calcification")

Evaluation:
  - Computes macro and per-class AUROC, AUPRC
  - Optimizes decision thresholds per class on CT-RATE to maximize F1
  - Reports per-class F1 and Balanced Accuracy at those thresholds
  - Applies CT-RATE-optimized thresholds as-is to RAD-ChestCT

Usage:
  python scripts/e2e_inference.py --config configs/finetune_lora.yaml
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

# --- PATH HACK ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataloaders.datasetloader_ctrate_multiscale import (
    CTMultiScaleDataset,
    MultiScaleSliceProcessor,
    get_npy_validation_dataset,
)
from utils.config import load_config, load_model_configs
from utils.dino_utils import build_model_with_config
from models.e2e_colipri import EndToEndColipri
from peft import LoraConfig, get_peft_model


# ============================================================
# CT-RATE label order (18 classes, from e2e_ft.yaml pos_weights)
# ============================================================
CTRATE_CLASS_NAMES = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]

# ============================================================
# RAD-ChestCT label order (16 classes, from rad_labels.csv)
# ============================================================
RAD_CLASS_NAMES = [
    "Calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Hiatal hernia",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
    "Lymphadenopathy",
    "Medical material",
]


def map_ctrate_to_rad(ctrate_probs, ctrate_class_names, rad_class_names):
    """
    Map CT-RATE model outputs (18 classes) to RAD-ChestCT label space (16 classes).

    Protocol:
      - Exclude "Mosaic attenuation pattern" (index 13 in CT-RATE)
      - "Calcification" = max("Arterial wall calcification" (idx 1),
                              "Coronary artery wall calcification" (idx 4))
    """
    name_to_ctrate_idx = {name: i for i, name in enumerate(ctrate_class_names)}

    rad_probs_list = []
    for rad_name in rad_class_names:
        if rad_name == "Calcification":
            arterial_idx = name_to_ctrate_idx["Arterial wall calcification"]
            coronary_idx = name_to_ctrate_idx["Coronary artery wall calcification"]
            rad_probs_list.append(
                np.maximum(ctrate_probs[:, arterial_idx], ctrate_probs[:, coronary_idx])
            )
        elif rad_name == "Mosaic attenuation pattern":
            continue
        else:
            rad_probs_list.append(ctrate_probs[:, name_to_ctrate_idx[rad_name]])

    return np.stack(rad_probs_list, axis=1)


def map_ctrate_thresholds_to_rad(ctrate_thresholds, ctrate_class_names, rad_class_names):
    """
    Map CT-RATE-optimized thresholds (18 classes) to RAD-ChestCT space (16 classes).

    For "Calcification", we take the max of the two calcification thresholds
    (conservative approach: higher threshold = higher specificity).
    """
    name_to_ctrate_idx = {name: i for i, name in enumerate(ctrate_class_names)}

    rad_thresholds = []
    for rad_name in rad_class_names:
        if rad_name == "Calcification":
            arterial_t = ctrate_thresholds[name_to_ctrate_idx["Arterial wall calcification"]]
            coronary_t = ctrate_thresholds[name_to_ctrate_idx["Coronary artery wall calcification"]]
            rad_thresholds.append(max(arterial_t, coronary_t))
        elif rad_name == "Mosaic attenuation pattern":
            continue
        else:
            rad_thresholds.append(ctrate_thresholds[name_to_ctrate_idx[rad_name]])

    return np.array(rad_thresholds)


def get_optimal_f1_thresholds(y_true, y_prob):
    """
    Calculate per-class thresholds that maximize F1 score using precision-recall curve.

    Returns:
        thresholds: numpy array of shape (n_classes,)
    """
    n_classes = y_true.shape[1]
    thresholds_out = []
    for i in range(n_classes):
        if len(np.unique(y_true[:, i])) < 2:
            thresholds_out.append(0.5)
            continue
        prec, rec, thresholds = precision_recall_curve(y_true[:, i], y_prob[:, i])
        # F1 = 2 * P * R / (P + R), computed at each threshold
        # prec[:-1], rec[:-1] because the last point has undefined F1
        f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-8)
        best_idx = np.argmax(f1_scores)
        thresholds_out.append(thresholds[best_idx])
    return np.array(thresholds_out)


def compute_comprehensive_metrics(probs, labels, class_names, thresholds=None):
    """
    Compute a comprehensive set of metrics for a dataset.

    Args:
        probs: (N, C) sigmoid probabilities
        labels: (N, C) binary labels
        class_names: list of C class name strings
        thresholds: optional (C,) array of decision thresholds.
                    If None, thresholds are optimized on this data.

    Returns:
        metrics_dict with keys: auroc, auprc, thresholds, f1, balanced_acc,
        sensitivity, specificity, ppv, and per-class versions of each.
    """
    n_classes = labels.shape[1]

    # --- AUROC & AUPRC ---
    per_class_auroc = []
    per_class_auprc = []
    for c in range(n_classes):
        if labels[:, c].sum() == 0 or labels[:, c].sum() == labels.shape[0]:
            per_class_auroc.append(0.5)
            per_class_auprc.append(0.0)
        else:
            per_class_auroc.append(roc_auc_score(labels[:, c], probs[:, c]))
            per_class_auprc.append(average_precision_score(labels[:, c], probs[:, c]))

    macro_auroc = float(np.mean(per_class_auroc))
    macro_auprc = float(np.mean(per_class_auprc))

    # --- Thresholds ---
    if thresholds is None:
        thresholds = get_optimal_f1_thresholds(labels, probs)

    # --- Binarize ---
    y_pred = (probs >= thresholds).astype(int)

    # --- Per-class F1, Balanced Accuracy, Sensitivity, Specificity, PPV ---
    per_class_f1 = []
    per_class_ba = []
    per_class_sens = []
    per_class_spec = []
    per_class_ppv = []

    for c in range(n_classes):
        y_t = labels[:, c]
        y_p = y_pred[:, c]

        if len(np.unique(y_t)) < 2:
            per_class_f1.append(0.0)
            per_class_ba.append(0.5)
            per_class_sens.append(0.0)
            per_class_spec.append(1.0)
            per_class_ppv.append(0.0)
            continue

        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        ppv = tp / (tp + fp + 1e-8)
        f1 = 2 * (ppv * sens) / (ppv + sens + 1e-8)
        ba = (sens + spec) / 2.0

        per_class_f1.append(f1)
        per_class_ba.append(ba)
        per_class_sens.append(sens)
        per_class_spec.append(spec)
        per_class_ppv.append(ppv)

    macro_f1 = float(np.mean(per_class_f1))
    macro_ba = float(np.mean(per_class_ba))

    return {
        'macro_auroc': macro_auroc,
        'macro_auprc': macro_auprc,
        'macro_f1': macro_f1,
        'macro_ba': macro_ba,
        'thresholds': thresholds,
        'per_class_auroc': per_class_auroc,
        'per_class_auprc': per_class_auprc,
        'per_class_f1': per_class_f1,
        'per_class_ba': per_class_ba,
        'per_class_sens': per_class_sens,
        'per_class_spec': per_class_spec,
        'per_class_ppv': per_class_ppv,
    }


def print_evaluation_report(metrics, class_names, dataset_name, thresholds_source=None):
    """
    Print a formatted evaluation report.

    Args:
        metrics: dict from compute_comprehensive_metrics
        class_names: list of class name strings
        dataset_name: string name of the dataset
        thresholds_source: if thresholds came from another dataset, name it here
    """
    header = f"EVALUATION REPORT: {dataset_name}"
    print("\n" + "=" * 80)
    print(header.center(80))
    print("=" * 80)

    if thresholds_source:
        print(f"  (Thresholds optimized on: {thresholds_source})")

    print(f"\n  {'':>38s} {'AUROC':>8s} {'AUPRC':>8s} {'F1':>8s} {'BalAcc':>8s} {'Sens':>8s} {'Spec':>8s} {'Thresh':>8s}")
    print(f"  {'-'*38} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for c, name in enumerate(class_names):
        print(f"  {name:>38s} "
              f"{metrics['per_class_auroc'][c]:8.4f} "
              f"{metrics['per_class_auprc'][c]:8.4f} "
              f"{metrics['per_class_f1'][c]:8.4f} "
              f"{metrics['per_class_ba'][c]:8.4f} "
              f"{metrics['per_class_sens'][c]:8.4f} "
              f"{metrics['per_class_spec'][c]:8.4f} "
              f"{metrics['thresholds'][c]:8.4f}")

    print(f"  {'-'*38} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'MACRO AVERAGE':>38s} "
          f"{metrics['macro_auroc']:8.4f} "
          f"{metrics['macro_auprc']:8.4f} "
          f"{metrics['macro_f1']:8.4f} "
          f"{metrics['macro_ba']:8.4f}")
    print("=" * 80)


def load_e2e_model(config, accelerator):
    """
    Load the EndToEndColipri model with the best checkpoint from training,
    and explicitly verify that LoRA matrices are non-zero.
    """
    save_dir = config.experiment.save_dir
    best_model_path = os.path.join(save_dir, "e2e_model_best.pth")

    if not os.path.isfile(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    if accelerator.is_main_process:
        print(f"Loading best model from: {best_model_path}")

    # --- Step 1: Load base ViT backbone from original pretrained checkpoint ---
    model_type = config.train.model_type
    model_params = load_model_configs(model_type)

    vit_ckpt_path = config.train.vit_ckpt_path
    if accelerator.is_main_process:
        print(f"Loading backbone from: {vit_ckpt_path}")
    base_vit, _ = build_model_with_config(
        config=config,
        model_params=model_params,
        accelerator=accelerator,
        checkpoint_path=vit_ckpt_path,
        load_pretrained=False,
        model_type=model_type,
    )

    # --- Step 2: Apply fresh LoRA (weights will be overwritten by checkpoint) ---
    lora_config = LoraConfig(
        r=config.experiment.lora_r,
        lora_alpha=config.experiment.lora_alpha,
        lora_dropout=config.experiment.lora_dropout,
        target_modules=["qkv"],
    )
    base_vit = get_peft_model(base_vit, lora_config, adapter_name="default")
    if accelerator.is_main_process:
        print("Applied fresh LoRA adapter")

    # --- Step 3: Build EndToEndColipri with fresh Colipri head ---
    multi_cfg = getattr(config, 'multi_adapter', None)
    model = EndToEndColipri(
        vit_backbone=base_vit,
        colipri_state_dict_path=None,
        input_dim=config.experiment.input_dim,
        pooling_scheme=config.experiment.best_pooling_scheme,
        multi_adapter_config=multi_cfg,
    )

    # --- Step 4: Load trained weights (LoRA + Colipri head) ---
    state_dict = torch.load(best_model_path, map_location="cpu", weights_only=True)
    model_state = model.state_dict()

    def clean_key(k):
        return k.replace("module.", "").replace("_orig_mod.", "")

    clean_model_state = {clean_key(k): k for k in model_state.keys()}
    matched_state = {}
    used_state_dict_keys = set()

    for k, v in state_dict.items():
        ck = clean_key(k)

        # 1. Direct match
        if ck in clean_model_state:
            matched_state[clean_model_state[ck]] = v
            used_state_dict_keys.add(k)
            continue

        # 2. Legacy Colipri Head mapping
        if "Q" in ck and "colipri" in ck:
            matched_state[clean_model_state["colipri_heads.default.Q"]] = v
            used_state_dict_keys.add(k)
            continue
        if "classifier.weight" in ck and "colipri" in ck:
            matched_state[clean_model_state["colipri_heads.default.classifier.weight"]] = v
            used_state_dict_keys.add(k)
            continue
        if "classifier.bias" in ck and "colipri" in ck:
            matched_state[clean_model_state["colipri_heads.default.classifier.bias"]] = v
            used_state_dict_keys.add(k)
            continue

        # 3. Suffix fallback
        for mk, orig_mk in clean_model_state.items():
            if ck.endswith(mk) or mk.endswith(ck):
                matched_state[orig_mk] = v
                used_state_dict_keys.add(k)
                break

    missing = set(model_state.keys()) - set(matched_state.keys())
    unexpected = set(state_dict.keys()) - used_state_dict_keys

    model_state.update(matched_state)
    model.load_state_dict(model_state, strict=False)

    if accelerator.is_main_process:
        print(f"Loaded trainable weights: {len(matched_state)}/{len(state_dict)} keys matched")

        # --- NEW: VERBOSITY & LORA VALIDATION BLOCK ---
        print("\n" + "=" * 60)
        print("🔍 LORA ADAPTER VALIDATION CHECK")
        print("=" * 60)

        lora_A_tensors = []
        lora_B_tensors = []
        for name, param in model.named_parameters():
            if 'lora_A' in name:
                lora_A_tensors.append((name, param))
            elif 'lora_B' in name:
                lora_B_tensors.append((name, param))

        if not lora_A_tensors and not lora_B_tensors:
            print("❌ CRITICAL ERROR: No LoRA parameters found in the model architecture!")
        else:
            print(f"✅ Architecture: Found {len(lora_A_tensors)} LoRA_A and {len(lora_B_tensors)} LoRA_B matrices.")

            total_b_magnitude = 0.0
            print("\n  Sample LoRA_B Matrix Magnitudes (L1 Norm):")
            for i, (name, param) in enumerate(lora_B_tensors):
                mag = torch.abs(param).sum().item()
                total_b_magnitude += mag
                if i < 5:  # Print the first 5 layers to avoid console spam
                    print(f"    - {name}: {mag:.6f}")
            print("    ...")

            if total_b_magnitude == 0.0:
                print("\n❌ FATAL WARNING: All LoRA_B matrices are EXACTLY ZERO.")
                print("   The adapter is acting as a no-op (frozen backbone).")
                print("   This means the LoRA weights either weren't trained or weren't loaded.")
            else:
                print(f"\n✅ SUCCESS: LoRA matrices are NON-ZERO (Total B-matrix L1 Norm: {total_b_magnitude:.2f}).")
                print("   The adapter has been successfully loaded and holds active weights.")
        print("=" * 60 + "\n")

    # --- Step 5: Prepare model with accelerator for multi-GPU inference ---
    model = accelerator.prepare(model)
    return model

@torch.no_grad()
def run_inference_on_dataset(model, dataloader, processor, accelerator, config, dataset_name="Dataset"):
    """
    Run distributed inference on a dataset and collect probabilities and labels.
    """
    model.eval()

    chunk_size = getattr(config.experiment, 'chunk_size', 128)
    device = accelerator.device

    all_probs = []
    all_labels = []
    all_filenames = []

    total_scans = len(dataloader.dataset) if hasattr(dataloader, 'dataset') else "?"
    processed_count = 0
    first_batch = True

    if accelerator.is_main_process:
        print(f"\nStarting inference on {dataset_name} ({total_scans} total volumes)...")

    for volumes, labels, filenames in dataloader:
        raw_volume = volumes.squeeze(0).to(device)
        labels = labels.to(device)

        processed_slices, _ = processor.process_batch(raw_volume)
        processed_slices = processed_slices.unsqueeze(0)  # (1, S, C, H, W)

        # --- NEW: VERBOSE ADAPTER CHECK ON FIRST BATCH ---
        if first_batch and accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            active_list = unwrapped_model.multi_adapter_config['adapters'].keys() if unwrapped_model.is_multi_adapter else ['default']
            print(f"⏩ Forward Pass Routing Check:")
            print(f"   - Input Shape: {processed_slices.shape}")
            print(f"   - Target Adapters: {active_list}")
            first_batch = False
        # ------------------------------------------------

        logits = model(processed_slices, chunk_size=chunk_size)
        probs = torch.sigmoid(logits)

        # Gather results safely across all GPUs
        gathered_probs, gathered_labels = accelerator.gather_for_metrics((probs, labels))
        gathered_filenames = accelerator.gather_for_metrics(filenames)

        # Move to CPU and accumulate only on the main process
        if accelerator.is_main_process:
            probs_np = gathered_probs.cpu().numpy()
            labels_np = gathered_labels.cpu().numpy()

            # --- Print neatly formatted single-line log for each scan ---
            for idx, fname in enumerate(gathered_filenames):
                processed_count += 1

                # Format to exactly 3 decimal places for visual alignment
                formatted_probs = "[" + ", ".join([f"{p:.3f}" for p in probs_np[idx]]) + "]"
                formatted_labels = "[" + ", ".join([str(int(l)) for l in labels_np[idx]]) + "]"

                # <25 ensures filenames are padded with spaces to perfectly align the columns
                # print(
                #     f"[{processed_count}/{total_scans}] {fname:<25} | Truth: {formatted_labels} | Pred: {formatted_probs}",
                #     flush=True)
            # ------------------------------------------------------------

            all_probs.append(probs_np)
            all_labels.append(labels_np)
            all_filenames.extend(gathered_filenames)

    # Concatenate the accumulated lists into final numpy arrays
    if accelerator.is_main_process:
        final_probs = np.concatenate(all_probs, axis=0)
        final_labels = np.concatenate(all_labels, axis=0)
        return final_probs, final_labels, all_filenames
    else:
        return None, None, None

def main():
    from accelerate import Accelerator

    parser = argparse.ArgumentParser(description="E2E LoRA + Colipri Inference")
    parser.add_argument("--config", type=str, default="configs/finetune_lora.yaml",
                        help="Path to inference config YAML")
    parser.add_argument("--ctrate-only", action="store_true",
                        help="Only run inference on CT-RATE validation set")
    parser.add_argument("--rad-only", action="store_true",
                        help="Only run inference on RAD-ChestCT")
    args = parser.parse_args()

    config = load_config(args.config)

    # Initialize accelerator for multi-GPU distributed inference
    accelerator = Accelerator()

    if accelerator.is_main_process:
        print(f"Using device: {accelerator.device}")
        print(f"Number of processes: {accelerator.num_processes}")

    # ============================================================
    # 1. Load Model (prepared for multi-GPU by load_e2e_model)
    # ============================================================
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("LOADING BEST MODEL")
        print("=" * 60)
    model = load_e2e_model(config, accelerator)

    # ============================================================
    # 2. Setup Processor (same normalization as training)
    # ============================================================
    processor = MultiScaleSliceProcessor(config)

    output_dir = config.output_folders.main_output
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    # Will hold CT-RATE thresholds for transfer to RAD-ChestCT
    ctrate_thresholds = None

    # ============================================================
    # 3. CT-RATE Validation Inference
    # ============================================================
    if not args.rad_only:
        if accelerator.is_main_process:
            print("\n" + "=" * 60)
            print("CT-RATE VALIDATION SET INFERENCE")
            print("=" * 60)

        ctrate_data_dir = config.ctrate.data_dir
        ctrate_label_csv = config.ctrate.label_csv
        ctrate_max_patients = getattr(config.ctrate, 'max_patients', None)

        if ctrate_max_patients is not None and ctrate_max_patients > 0:
            val_dataset = get_npy_validation_dataset(
                config=config,
                data_dir=ctrate_data_dir,
                label_csv=ctrate_label_csv,
                max_patients=ctrate_max_patients,
                seed=config.experiment.seed,
            )
        else:
            val_dataset = CTMultiScaleDataset(
                config=config,
                data_dir=ctrate_data_dir,
                label_csv=ctrate_label_csv,
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=getattr(config.validation, 'num_workers', 4),
            shuffle=False,
            pin_memory=True,
        )

        # Prepare dataloader with accelerator to split across GPUs
        val_loader = accelerator.prepare(val_loader)

        if accelerator.is_main_process:
            print(f"CT-RATE validation dataset: {len(val_dataset)} volumes")

        ctrate_probs, ctrate_labels, ctrate_filenames = run_inference_on_dataset(
            model, val_loader, processor, accelerator, config, dataset_name="CT-RATE Val"
        )

        # Compute metrics and save only on the main process
        if accelerator.is_main_process:
            # Compute comprehensive metrics with F1-optimized thresholds
            ctrate_metrics = compute_comprehensive_metrics(
                ctrate_probs, ctrate_labels, CTRATE_CLASS_NAMES
            )
            ctrate_thresholds = ctrate_metrics['thresholds']

            # Print evaluation report
            print_evaluation_report(ctrate_metrics, CTRATE_CLASS_NAMES, "CT-RATE Validation")

            # Save results
            np.savez(
                os.path.join(output_dir, "ctrate_val_results.npz"),
                probs=ctrate_probs,
                labels=ctrate_labels,
                filenames=np.array(ctrate_filenames),
                class_names=np.array(CTRATE_CLASS_NAMES),
                thresholds=ctrate_thresholds,
                macro_auroc=ctrate_metrics['macro_auroc'],
                macro_auprc=ctrate_metrics['macro_auprc'],
                macro_f1=ctrate_metrics['macro_f1'],
                macro_ba=ctrate_metrics['macro_ba'],
                per_class_auroc=np.array(ctrate_metrics['per_class_auroc']),
                per_class_auprc=np.array(ctrate_metrics['per_class_auprc']),
                per_class_f1=np.array(ctrate_metrics['per_class_f1']),
                per_class_ba=np.array(ctrate_metrics['per_class_ba']),
            )
            print(f"\nCT-RATE results saved to {output_dir}/ctrate_val_results.npz")

        # Wait for all processes before proceeding
        accelerator.wait_for_everyone()

    # ============================================================
    # 4. RAD-ChestCT Inference
    # ============================================================
    if not args.ctrate_only:
        if accelerator.is_main_process:
            print("\n" + "=" * 60)
            print("RAD-ChestCT INFERENCE")
            print("=" * 60)

        rad_data_dir = config.rad.data_dir
        rad_label_csv = config.rad.label_csv

        rad_dataset = CTMultiScaleDataset(
            config=config,
            data_dir=rad_data_dir,
            label_csv=rad_label_csv,
        )

        rad_loader = DataLoader(
            rad_dataset,
            batch_size=1,
            num_workers=getattr(config.validation, 'num_workers', 4),
            shuffle=False,
            pin_memory=True,
        )

        # Prepare dataloader with accelerator to split across GPUs
        rad_loader = accelerator.prepare(rad_loader)

        if accelerator.is_main_process:
            print(f"RAD-ChestCT dataset: {len(rad_dataset)} volumes")
            print(f"RAD-ChestCT CSV classes: {rad_dataset.class_names}")

        rad_probs_ctrate, rad_labels_raw, rad_filenames = run_inference_on_dataset(
            model, rad_loader, processor, accelerator, config, dataset_name="RAD-ChestCT"
        )

        # Compute metrics and save only on the main process
        if accelerator.is_main_process:
            # Map CT-RATE 18-class outputs to RAD-ChestCT 16-class space
            rad_probs = map_ctrate_to_rad(rad_probs_ctrate, CTRATE_CLASS_NAMES, RAD_CLASS_NAMES)

            # Reorder RAD-ChestCT labels to match RAD_CLASS_NAMES
            rad_csv_class_names = rad_dataset.class_names
            csv_to_rad_idx = []
            for rad_name in RAD_CLASS_NAMES:
                if rad_name in rad_csv_class_names:
                    csv_to_rad_idx.append(rad_csv_class_names.index(rad_name))
                else:
                    print(f"WARNING: '{rad_name}' not found in RAD-ChestCT CSV columns, skipping")
                    csv_to_rad_idx.append(-1)

            rad_labels = np.zeros((rad_labels_raw.shape[0], len(RAD_CLASS_NAMES)), dtype=np.float32)
            for i, csv_idx in enumerate(csv_to_rad_idx):
                if csv_idx >= 0:
                    rad_labels[:, i] = rad_labels_raw[:, csv_idx]

            # --- Apply CT-RATE thresholds to RAD-ChestCT ---
            if ctrate_thresholds is not None:
                rad_thresholds = map_ctrate_thresholds_to_rad(
                    ctrate_thresholds, CTRATE_CLASS_NAMES, RAD_CLASS_NAMES
                )
                rad_metrics = compute_comprehensive_metrics(
                    rad_probs, rad_labels, RAD_CLASS_NAMES, thresholds=rad_thresholds
                )
                print_evaluation_report(
                    rad_metrics, RAD_CLASS_NAMES, "RAD-ChestCT",
                    thresholds_source="CT-RATE Validation"
                )
            else:
                # No CT-RATE thresholds available (--rad-only mode), optimize on RAD-ChestCT itself
                rad_metrics = compute_comprehensive_metrics(
                    rad_probs, rad_labels, RAD_CLASS_NAMES
                )
                rad_thresholds = rad_metrics['thresholds']
                print_evaluation_report(rad_metrics, RAD_CLASS_NAMES, "RAD-ChestCT")

            # Save results
            np.savez(
                os.path.join(output_dir, "rad_chestct_results.npz"),
                probs=rad_probs,
                labels=rad_labels,
                filenames=np.array(rad_filenames),
                class_names=np.array(RAD_CLASS_NAMES),
                thresholds=rad_thresholds if ctrate_thresholds is not None else rad_metrics['thresholds'],
                macro_auroc=rad_metrics['macro_auroc'],
                macro_auprc=rad_metrics['macro_auprc'],
                macro_f1=rad_metrics['macro_f1'],
                macro_ba=rad_metrics['macro_ba'],
                per_class_auroc=np.array(rad_metrics['per_class_auroc']),
                per_class_auprc=np.array(rad_metrics['per_class_auprc']),
                per_class_f1=np.array(rad_metrics['per_class_f1']),
                per_class_ba=np.array(rad_metrics['per_class_ba']),
            )
            print(f"\nRAD-ChestCT results saved to {output_dir}/rad_chestct_results.npz")

    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("INFERENCE COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    main()
