"""
Main orchestration script for training lightweight probes on frozen
representations and evaluating on the ReX-GroundingCT dataset.

Supports multi-GPU training via HuggingFace Accelerate (DDP).

Supports 4 backbone variants:
  - Kentucky-Open-Science/DALE-CT-2S  (timm, patch_size=16, no registers)
  - Kentucky-Open-Science/DALE-CT-1S  (timm, patch_size=14, no registers)
  - Kentucky-Open-Science/DALE-CT-0   (timm, patch_size=16, no registers)
  - Kentucky-Open-Science/Finetuned-DINOv2-Chest-CT  (HF transformers, patch_size=14, 4 registers)

Workflow:
  1. Load the frozen ViT backbone from HuggingFace Hub.
  2. Attach RexSliceClassifier and RexDenseProbe heads.
  3. Train on 2D slices with BCEWithLogitsLoss (classification + segmentation).
  4. Evaluate slice-level Macro AUROC/AUPRC.
  5. Reconstruct 3D volumes and compute 3D Dice scores + Hit@5%/Hit@10%.

Usage (single-GPU):
    python scripts/train_eval_rex.py --config configs/rex_evaluation/lejepa_2s.yaml

Usage (multi-GPU via Accelerate):
    accelerate launch --num_processes=4 scripts/train_eval_rex.py --config configs/rex_evaluation/lejepa_2s.yaml
"""

import argparse
import math
import os
import sys
from datetime import timedelta
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator, InitProcessGroupKwargs
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# --- PATH HACK: allow imports from parent directory ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from dataloaders.dataloader_rex_nifti import (
    REX_CLASSES,
    CTPreprocessor,
    create_rex_dataloaders,
)
from models.rex_probes import RexSliceClassifier, RexDenseProbe
from utils.config import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def compute_3d_dice(
    pred_3d: np.ndarray,
    gt_3d: np.ndarray,
    num_classes: int = 14,
    smooth: float = 1e-6,
) -> np.ndarray:
    """
    Compute per-class 3D volumetric Dice scores.

    Returns NaN for classes where the ground truth has zero pixels
    (i.e., the finding is not present in this patient).  These NaNs
    are handled by np.nanmean during aggregation so that only
    patients who actually have each finding contribute to its Dice.

    Parameters
    ----------
    pred_3d : np.ndarray  shape [D, C, H, W], binary
    gt_3d : np.ndarray    shape [D, C, H, W], binary
    num_classes : int
    smooth : float

    Returns
    -------
    dice_scores : np.ndarray  shape [C]
        NaN for classes absent in this patient.
    """
    dice_scores = np.full(num_classes, np.nan)
    for c in range(num_classes):
        pred_c = pred_3d[:, c, :, :].reshape(-1)
        gt_c = gt_3d[:, c, :, :].reshape(-1)
        gt_sum = gt_c.sum()
        if gt_sum == 0:
            continue  # leave as NaN — finding not present
        intersection = (pred_c * gt_c).sum()
        dice_scores[c] = (
            2.0 * intersection + smooth
        ) / (pred_c.sum() + gt_sum + smooth)
    return dice_scores


def compute_pos_weight(
    dataloader,
    num_classes: int = 14,
    accelerator: Accelerator = None,
) -> torch.Tensor:
    """
    Compute per-class positive weights for BCEWithLogitsLoss from the
    training set to handle extreme class imbalance.

    pos_weight = num_negatives / num_positives

    This is computed by scanning the training dataloader once before
    training begins.  The result is broadcast across all DDP ranks.

    Parameters
    ----------
    dataloader : DataLoader
        Training dataloader (yields (volumes, masks, cls_labels, ids)).
    num_classes : int
    accelerator : Accelerator or None

    Returns
    -------
    pos_weight : torch.Tensor  shape [num_classes]
        Weights suitable for ``BCEWithLogitsLoss(pos_weight=...)``.
    """
    total_pos = torch.zeros(num_classes)
    total_neg = torch.zeros(num_classes)

    for volumes, masks, cls_labels, patient_ids in tqdm(
        dataloader,
        desc="Computing class weights",
        disable=accelerator is not None and not accelerator.is_local_main_process,
    ):
        for lbl in cls_labels:
            # lbl shape: [D, num_classes]
            lbl_np = lbl.numpy() if isinstance(lbl, torch.Tensor) else lbl
            pos_mask = lbl_np > 0.5
            neg_mask = ~pos_mask
            total_pos += pos_mask.sum(axis=0)
            total_neg += neg_mask.sum(axis=0)

    # Avoid division by zero: if a class never appears, set weight to 1.0
    pos_weight = total_neg / (total_pos + 1e-8)
    # Clamp to reasonable range to avoid extreme weights
    pos_weight = torch.clamp(pos_weight, min=0.1, max=100.0)

    if accelerator is not None and accelerator.is_local_main_process:
        print(f"[PosWeight] Computed per-class pos_weight: {pos_weight.tolist()}")

    return pos_weight


# ---------------------------------------------------------------------------
# Backbone Loading
# ---------------------------------------------------------------------------

def load_backbone_timm(repo_id: str, in_chans: int, patch_size: int, img_size: int):
    """
    Load a timm-based LeJEPA backbone from HuggingFace Hub.

    Uses the pattern from the model card:
        timm.create_model("vit_large_patch14_dinov2", ...)
        hf_hub_download + safetensors load_file
    """
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    print(f"[Backbone] Loading timm model from {repo_id}")
    print(f"           in_chans={in_chans}, patch_size={patch_size}, img_size={img_size}")

    model = timm.create_model(
        "vit_large_patch14_dinov2",
        pretrained=False,
        num_classes=0,
        in_chans=in_chans,
        patch_size=patch_size,
        img_size=img_size,
        dynamic_img_size=True,
    )

    model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"[Backbone] Loaded successfully.")
    return model


def load_backbone_hf(repo_id: str):
    """
    Load a HuggingFace transformers-based DINOv2 backbone.

    Uses the pattern from the model card:
        AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    """
    from transformers import AutoModel

    print(f"[Backbone] Loading HF transformers model from {repo_id}")

    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    model.eval()

    print(f"[Backbone] Loaded successfully.")
    return model


def load_backbone(cfg) -> nn.Module:
    """
    Load the appropriate backbone based on config.

    Returns a frozen backbone module.
    """
    repo_id = cfg.model.repo_id
    backbone_type = cfg.model.backbone_type
    in_chans = cfg.model.in_chans
    patch_size = cfg.model.patch_size
    img_size = cfg.model.native_img_size

    if backbone_type == "timm":
        backbone = load_backbone_timm(
            repo_id=repo_id,
            in_chans=in_chans,
            patch_size=patch_size,
            img_size=img_size,
        )
    elif backbone_type == "hf":
        backbone = load_backbone_hf(repo_id=repo_id)
    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}")

    # Freeze
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    return backbone


# ---------------------------------------------------------------------------
# Training (one epoch)
# ---------------------------------------------------------------------------

def train_one_epoch(
    cls_model: RexSliceClassifier,
    seg_model: RexDenseProbe,
    dataloader,
    cls_criterion: nn.Module,
    seg_criterion: nn.Module,
    optimizer: optim.Optimizer,
    accelerator: Accelerator,
    mini_batch_size: int,
    epoch: int,
    log_interval: int = 10,
    scheduler: optim.lr_scheduler._LRScheduler = None,
) -> Dict[str, float]:
    """
    Train both probe heads for one epoch.

    The dataloader yields one patient volume at a time.  We iterate over
    the D (depth) dimension in mini-batches to avoid OOM.

    If ``scheduler`` is provided (e.g. OneCycleLR), it is stepped after
    every mini-batch update.
    """
    cls_model.train()
    seg_model.train()

    total_cls_loss = 0.0
    total_seg_loss = 0.0
    total_slices = 0

    # Only show progress bar on main process
    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch} [Train]",
        disable=not accelerator.is_local_main_process,
    )
    for batch_idx, (volumes, masks, cls_labels, patient_ids) in enumerate(pbar):
        for vol, msk, lbl in zip(volumes, masks, cls_labels):
            D = vol.shape[0]

            for start in range(0, D, mini_batch_size):
                end = min(start + mini_batch_size, D)

                vol_chunk = vol[start:end].to(accelerator.device)
                msk_chunk = msk[start:end].to(accelerator.device)
                lbl_chunk = lbl[start:end].to(accelerator.device)

                with accelerator.autocast():
                    # Forward: classification
                    cls_logits = cls_model(vol_chunk)
                    cls_loss = cls_criterion(cls_logits, lbl_chunk)

                    # Forward: segmentation
                    seg_logits = seg_model(vol_chunk)
                    seg_loss = seg_criterion(seg_logits, msk_chunk)

                    # Combined loss
                    loss = cls_loss + seg_loss

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                # Step scheduler per batch (required for OneCycleLR)
                if scheduler is not None:
                    scheduler.step()

                B_actual = end - start
                total_cls_loss += cls_loss.item() * B_actual
                total_seg_loss += seg_loss.item() * B_actual
                total_slices += B_actual

            # Update progress bar once per patient (not per mini-batch)
            if batch_idx % log_interval == 0:
                pbar.set_postfix({
                    'cls_loss': f'{cls_loss.item():.4f}',
                    'seg_loss': f'{seg_loss.item():.4f}',
                })

    # Gather loss across all processes
    cls_loss_tensor = torch.tensor([total_cls_loss], device=accelerator.device)
    seg_loss_tensor = torch.tensor([total_seg_loss], device=accelerator.device)
    slices_tensor = torch.tensor([total_slices], device=accelerator.device)

    cls_loss_tensor = accelerator.reduce(cls_loss_tensor, reduction="sum")
    seg_loss_tensor = accelerator.reduce(seg_loss_tensor, reduction="sum")
    slices_tensor = accelerator.reduce(slices_tensor, reduction="sum")

    return {
        'train_cls_loss': cls_loss_tensor.item() / max(slices_tensor.item(), 1),
        'train_seg_loss': seg_loss_tensor.item() / max(slices_tensor.item(), 1),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    cls_model: RexSliceClassifier,
    seg_model: RexDenseProbe,
    dataloader,
    cls_criterion: nn.Module,
    seg_criterion: nn.Module,
    accelerator: Accelerator,
    mini_batch_size: int,
    num_classes: int = 14,
) -> Dict[str, float]:
    """
    Evaluate both probe heads.

    Computes:
      - Slice-level Macro and per-class AUROC/AUPRC (classification)
      - Slice-level segmentation loss
      - 3D volumetric Dice scores per class
      - Hit@5% and Hit@10%

    Predictions are accumulated per-rank first, then gathered once at the
    end via ``accelerator.gather_for_metrics`` to avoid data duplication
    that would occur if gathering inside the per-patient loop.
    """
    cls_model.eval()
    seg_model.eval()

    total_cls_loss = 0.0
    total_seg_loss = 0.0
    total_slices = 0

    # Per-rank accumulators (not yet gathered)
    local_cls_probs = []
    local_cls_targets = []
    per_class_dice_scores = []

    pbar = tqdm(
        dataloader,
        desc="Evaluating",
        disable=not accelerator.is_local_main_process,
    )
    for volumes, masks, cls_labels, patient_ids in pbar:
        for vol, msk, lbl, pid in zip(volumes, masks, cls_labels, patient_ids):
            D = vol.shape[0]
            pred_3d_slices = []

            for start in range(0, D, mini_batch_size):
                end = min(start + mini_batch_size, D)

                vol_chunk = vol[start:end].to(accelerator.device)
                msk_chunk = msk[start:end].to(accelerator.device)
                lbl_chunk = lbl[start:end].to(accelerator.device)

                with accelerator.autocast():
                    cls_logits = cls_model(vol_chunk)
                    cls_loss = cls_criterion(cls_logits, lbl_chunk)
                    cls_probs = torch.sigmoid(cls_logits)

                    seg_logits = seg_model(vol_chunk)
                    seg_loss = seg_criterion(seg_logits, msk_chunk)
                    seg_probs = torch.sigmoid(seg_logits)

                B_actual = end - start
                total_cls_loss += cls_loss.item() * B_actual
                total_seg_loss += seg_loss.item() * B_actual
                total_slices += B_actual

                # Accumulate locally — NO gathering inside the loop
                local_cls_probs.append(cls_probs.cpu())
                local_cls_targets.append(lbl_chunk.cpu())
                pred_3d_slices.append(seg_probs.cpu().numpy())

            # --- 3D Reconstruction & Dice (per patient, local rank only) ---
            pred_3d = np.concatenate(pred_3d_slices, axis=0)
            gt_3d = msk.cpu().numpy()
            pred_3d_bin = (pred_3d >= 0.5).astype(np.float32)

            dice_scores = compute_3d_dice(
                pred_3d_bin, gt_3d, num_classes=num_classes
            )
            per_class_dice_scores.append(dice_scores)

    # --- Gather classification predictions ONCE across all ranks ---
    local_probs_tensor = torch.cat(local_cls_probs, dim=0).to(accelerator.device)
    local_targets_tensor = torch.cat(local_cls_targets, dim=0).to(accelerator.device)

    gathered_probs = accelerator.gather_for_metrics(local_probs_tensor)
    gathered_targets = accelerator.gather_for_metrics(local_targets_tensor)

    y_prob = gathered_probs.cpu().numpy()
    y_true = gathered_targets.cpu().numpy()

    # Reduce losses across ranks
    cls_loss_tensor = torch.tensor([total_cls_loss], device=accelerator.device)
    seg_loss_tensor = torch.tensor([total_seg_loss], device=accelerator.device)
    slices_tensor = torch.tensor([total_slices], device=accelerator.device)

    cls_loss_tensor = accelerator.reduce(cls_loss_tensor, reduction="sum")
    seg_loss_tensor = accelerator.reduce(seg_loss_tensor, reduction="sum")
    slices_tensor = accelerator.reduce(slices_tensor, reduction="sum")

    metrics = {
        'val_cls_loss': cls_loss_tensor.item() / max(slices_tensor.item(), 1),
        'val_seg_loss': seg_loss_tensor.item() / max(slices_tensor.item(), 1),
    }

    # --- Per-class AUROC and AUPRC ---
    per_class_auc = []
    per_class_auprc = []
    for i in range(num_classes):
        y_t = y_true[:, i]
        y_p = y_prob[:, i]
        if len(np.unique(y_t)) < 2:
            per_class_auc.append(0.0)
            per_class_auprc.append(0.0)
        else:
            try:
                per_class_auc.append(roc_auc_score(y_t, y_p))
            except ValueError:
                per_class_auc.append(0.0)
            try:
                per_class_auprc.append(average_precision_score(y_t, y_p))
            except ValueError:
                per_class_auprc.append(0.0)

    metrics['val_macro_auc'] = np.mean(per_class_auc)
    metrics['val_macro_auprc'] = np.mean(per_class_auprc)

    for i, name in enumerate(REX_CLASSES):
        metrics[f'val_auc/{name}'] = per_class_auc[i]
        metrics[f'val_auprc/{name}'] = per_class_auprc[i]

    # --- Aggregate 3D volume-level metrics ---
    if len(per_class_dice_scores) > 0:
        local_dice = torch.tensor(
            np.stack(per_class_dice_scores, axis=0), device=accelerator.device
        )
        all_dice_list = accelerator.gather_for_metrics(local_dice)
        all_dice = all_dice_list.cpu().numpy()

        mean_dice_per_class = np.nanmean(all_dice, axis=0)
        std_dice_per_class = np.nanstd(all_dice, axis=0)
        metrics['val_mean_dice'] = np.nanmean(mean_dice_per_class)

        hit5_per_class = np.nanmean(all_dice > 0.05, axis=0)
        hit10_per_class = np.nanmean(all_dice > 0.10, axis=0)

        metrics['val_hit_at_5'] = np.nanmean(hit5_per_class)
        metrics['val_hit_at_10'] = np.nanmean(hit10_per_class)

        for i, name in enumerate(REX_CLASSES):
            metrics[f'val_dice/{name}'] = mean_dice_per_class[i]
            metrics[f'val_dice_std/{name}'] = std_dice_per_class[i]
            metrics[f'val_hit5/{name}'] = hit5_per_class[i]
            metrics[f'val_hit10/{name}'] = hit10_per_class[i]
    else:
        metrics['val_mean_dice'] = 0.0
        metrics['val_hit_at_5'] = 0.0
        metrics['val_hit_at_10'] = 0.0
        for name in REX_CLASSES:
            metrics[f'val_dice/{name}'] = 0.0
            metrics[f'val_dice_std/{name}'] = 0.0
            metrics[f'val_hit5/{name}'] = 0.0
            metrics[f'val_hit10/{name}'] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# Results Report
# ---------------------------------------------------------------------------

def generate_report(
    cfg,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    save_dir: str,
    total_epochs: int,
):
    """
    Generate a comprehensive results report for the validation split.

    Writes a formatted text file and a JSON file to save_dir.
    """
    import json
    from datetime import datetime

    report_path = os.path.join(save_dir, 'results_report.txt')
    json_path = os.path.join(save_dir, 'results_metrics.json')

    lines = []
    sep = "=" * 72

    lines.append(sep)
    lines.append("  ReX-GroundingCT Fast-Track Evaluation — Results Report")
    lines.append(sep)
    lines.append(f"  Date:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Model:       {cfg.model.repo_id}")
    lines.append(f"  Backbone:    {cfg.model.backbone_type} | patch_size={cfg.model.patch_size} | registers={cfg.model.num_register_tokens}")
    lines.append(f"  Image size:  {cfg.data.image_size}×{cfg.data.image_size}")
    lines.append(f"  Epochs:      {total_epochs}")
    lines.append(f"  LR:          {cfg.train.lr} | Weight Decay: {cfg.train.weight_decay}")
    lines.append(f"  Mini-batch:  {cfg.data.mini_batch_size} slices")
    lines.append(sep)

    # --- Aggregate Metrics ---
    lines.append("")
    lines.append("  AGGREGATE METRICS")
    lines.append("  " + "-" * 50)
    lines.append(f"  Train CLS Loss:        {train_metrics['train_cls_loss']:.6f}")
    lines.append(f"  Train SEG Loss:        {train_metrics['train_seg_loss']:.6f}")
    lines.append(f"  Val CLS Loss:          {val_metrics['val_cls_loss']:.6f}")
    lines.append(f"  Val SEG Loss:          {val_metrics['val_seg_loss']:.6f}")
    lines.append(f"  Val Macro AUROC:       {val_metrics['val_macro_auc']:.4f}")
    lines.append(f"  Val Macro AUPRC:       {val_metrics['val_macro_auprc']:.4f}")
    lines.append(f"  Val Mean 3D Dice:      {val_metrics['val_mean_dice']:.4f}")
    lines.append(f"  Val Hit@5%:            {val_metrics['val_hit_at_5']:.4f}")
    lines.append(f"  Val Hit@10%:           {val_metrics['val_hit_at_10']:.4f}")

    # --- Per-Class Classification ---
    lines.append("")
    lines.append("  PER-CLASS CLASSIFICATION (Slice-Level)")
    lines.append("  " + "-" * 50)
    lines.append(f"  {'Class':<8} {'AUROC':>8} {'AUPRC':>8}")
    lines.append(f"  {'-'*6:<8} {'-'*6:>8} {'-'*6:>8}")
    for name in REX_CLASSES:
        auc = val_metrics.get(f'val_auc/{name}', 0.0)
        auprc = val_metrics.get(f'val_auprc/{name}', 0.0)
        lines.append(f"  {name:<8} {auc:>8.4f} {auprc:>8.4f}")

    # --- Per-Class Segmentation ---
    lines.append("")
    lines.append("  PER-CLASS SEGMENTATION (3D Volumetric)")
    lines.append("  " + "-" * 50)
    lines.append(f"  {'Class':<8} {'Dice':>8} {'±Std':>8} {'Hit@5%':>8} {'Hit@10%':>8}")
    lines.append(f"  {'-'*6:<8} {'-'*6:>8} {'-'*6:>8} {'-'*6:>8} {'-'*6:>8}")
    for name in REX_CLASSES:
        dice = val_metrics.get(f'val_dice/{name}', 0.0)
        dice_std = val_metrics.get(f'val_dice_std/{name}', 0.0)
        hit5 = val_metrics.get(f'val_hit5/{name}', 0.0)
        hit10 = val_metrics.get(f'val_hit10/{name}', 0.0)
        lines.append(
            f"  {name:<8} {dice:>8.4f} {dice_std:>8.4f} {hit5:>8.4f} {hit10:>8.4f}"
        )

    lines.append("")
    lines.append(sep)
    lines.append("  End of Report")
    lines.append(sep)

    report_text = "\n".join(lines)

    # Write text report
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\n[Report] Saved to {report_path}")
    print(report_text)

    # Write JSON metrics
    json_metrics = {
        'model': cfg.model.repo_id,
        'backbone_type': cfg.model.backbone_type,
        'patch_size': cfg.model.patch_size,
        'num_register_tokens': cfg.model.num_register_tokens,
        'image_size': cfg.data.image_size,
        'epochs': total_epochs,
        'lr': cfg.train.lr,
        'weight_decay': cfg.train.weight_decay,
        'mini_batch_size': cfg.data.mini_batch_size,
        'train_cls_loss': train_metrics['train_cls_loss'],
        'train_seg_loss': train_metrics['train_seg_loss'],
        'val_cls_loss': val_metrics['val_cls_loss'],
        'val_seg_loss': val_metrics['val_seg_loss'],
        'val_macro_auc': val_metrics['val_macro_auc'],
        'val_macro_auprc': val_metrics['val_macro_auprc'],
        'val_mean_dice': val_metrics['val_mean_dice'],
        'val_hit_at_5': val_metrics['val_hit_at_5'],
        'val_hit_at_10': val_metrics['val_hit_at_10'],
        'per_class': {},
    }
    for name in REX_CLASSES:
        json_metrics['per_class'][name] = {
            'auc': float(val_metrics.get(f'val_auc/{name}', 0.0)),
            'auprc': float(val_metrics.get(f'val_auprc/{name}', 0.0)),
            'dice': float(val_metrics.get(f'val_dice/{name}', 0.0)),
            'dice_std': float(val_metrics.get(f'val_dice_std/{name}', 0.0)),
            'hit_at_5': float(val_metrics.get(f'val_hit5/{name}', 0.0)),
            'hit_at_10': float(val_metrics.get(f'val_hit10/{name}', 0.0)),
        }

    with open(json_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    print(f"[Report] JSON metrics saved to {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ReX-GroundingCT Fast-Track Evaluation"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/rex_evaluation/lejepa_2s.yaml',
        help='Path to YAML configuration file.',
    )
    args = parser.parse_args()

    # --- Initialize Accelerator for multi-GPU support ---
    process_group_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=7200)
    )
    accelerator = Accelerator(
        mixed_precision="bf16",
        kwargs_handlers=[process_group_kwargs],
    )

    # Load config
    cfg = load_config(args.config)

    if accelerator.is_local_main_process:
        print(OmegaConf.to_yaml(cfg))

    set_seed(42)

    if accelerator.is_local_main_process:
        print(f"[Accelerator] {accelerator.num_processes} process(es) detected.")

    # --- Build preprocessing pipeline from config ---
    pp_cfg = cfg.data.preprocessing
    preprocessor = CTPreprocessor(
        clip_min=pp_cfg.clip_min,
        clip_max=pp_cfg.clip_max,
        mean_hu=pp_cfg.mean_hu,
        std_hu=pp_cfg.std_hu,
        patch_size=cfg.model.patch_size,
    )
    if accelerator.is_local_main_process:
        print(
            f"[Preprocessing] clip=[{pp_cfg.clip_min}, {pp_cfg.clip_max}], "
            f"mean_hu={pp_cfg.mean_hu}, std_hu={pp_cfg.std_hu}, "
            f"patch_size={cfg.model.patch_size}"
        )

    # --- Load frozen backbone ---
    backbone = load_backbone(cfg)

    # --- Build probe heads ---
    embed_dim = cfg.model.embed_dim
    num_classes = cfg.model.num_classes
    image_size = cfg.data.image_size
    patch_size = cfg.model.patch_size
    num_register_tokens = cfg.model.num_register_tokens
    backbone_type = cfg.model.backbone_type

    cls_model = RexSliceClassifier(
        backbone=backbone,
        embed_dim=embed_dim,
        num_classes=num_classes,
        backbone_type=backbone_type,
    )

    seg_model = RexDenseProbe(
        backbone=backbone,
        embed_dim=embed_dim,
        num_classes=num_classes,
        patch_size=patch_size,
        image_size=image_size,
        num_register_tokens=num_register_tokens,
        backbone_type=backbone_type,
    )

    # --- Optimizer (only probe parameters) ---
    trainable_params = (
        list(cls_model.classifier.parameters())
        + list(seg_model.conv_seg.parameters())
    )
    optimizer = optim.AdamW(
        trainable_params,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # --- DataLoaders (created before prepare so we can scan for pos_weight) ---
    train_metadata_csv = cfg.data.get('train_metadata_csv', None)
    valid_metadata_csv = cfg.data.get('valid_metadata_csv', None)
    train_loader, val_loader = create_rex_dataloaders(
        rex_json_path=cfg.data.rex_json_path,
        masks_dir=cfg.data.masks_dir,
        ct_volumes_dir=cfg.data.ct_volumes_dir,
        image_size=image_size,
        batch_size=1,
        num_workers=4,
        preprocessor=preprocessor,
        train_metadata_csv=train_metadata_csv,
        valid_metadata_csv=valid_metadata_csv,
    )

    # --- Compute per-class positive weights from training set ---
    # This must happen BEFORE accelerator.prepare (dataloader not yet wrapped).
    pos_weight = compute_pos_weight(
        train_loader,
        num_classes=num_classes,
        accelerator=accelerator,
    )

    # --- Loss functions with class-balanced weights ---
    # pos_weight is on CPU now; accelerator.prepare() will move the entire
    # criterion (including its internal pos_weight buffer) to the correct device.
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    seg_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- OneCycleLR scheduler for fast convergence in few epochs ---
    total_epochs = cfg.train.epochs
    # Estimate total steps: patients × avg slices per patient / mini_batch_size
    # We use a rough estimate; OneCycleLR adapts well even if approximate.
    steps_per_epoch = len(train_loader) * 200 // cfg.data.mini_batch_size
    total_steps = steps_per_epoch * total_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.train.lr,
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        final_div_factor=100.0,
    )

    if accelerator.is_local_main_process:
        print(
            f"[Scheduler] OneCycleLR with max_lr={cfg.train.lr}, "
            f"total_steps≈{total_steps}, pct_start=0.1"
        )

    # --- Accelerator prepare ---
    # Include loss criteria so their internal pos_weight buffers are moved
    # to the correct device alongside the models.
    (
        cls_model,
        seg_model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        cls_criterion,
        seg_criterion,
    ) = accelerator.prepare(
        cls_model, seg_model, optimizer, scheduler,
        train_loader, val_loader, cls_criterion, seg_criterion,
    )

    # --- Output directory ---
    save_dir = cfg.output.save_dir
    if accelerator.is_local_main_process:
        os.makedirs(save_dir, exist_ok=True)

    # --- Training loop ---
    best_dice = 0.0
    mini_batch_size = cfg.data.mini_batch_size
    log_interval = cfg.output.log_interval
    final_val_metrics = None

    for epoch in range(1, total_epochs + 1):
        if accelerator.is_local_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{total_epochs}")
            print(f"{'='*60}")

        train_metrics = train_one_epoch(
            cls_model=cls_model,
            seg_model=seg_model,
            dataloader=train_loader,
            cls_criterion=cls_criterion,
            seg_criterion=seg_criterion,
            optimizer=optimizer,
            accelerator=accelerator,
            mini_batch_size=mini_batch_size,
            epoch=epoch,
            log_interval=log_interval,
            scheduler=scheduler,  # OneCycleLR steps per batch
        )

        if accelerator.is_local_main_process:
            print(
                f"Train - CLS Loss: {train_metrics['train_cls_loss']:.4f}, "
                f"SEG Loss: {train_metrics['train_seg_loss']:.4f}"
            )

        if epoch % cfg.train.eval_every_n_epochs == 0:
            val_metrics = evaluate(
                cls_model=cls_model,
                seg_model=seg_model,
                dataloader=val_loader,
                cls_criterion=cls_criterion,
                seg_criterion=seg_criterion,
                accelerator=accelerator,
                mini_batch_size=mini_batch_size,
                num_classes=num_classes,
            )
            final_val_metrics = val_metrics

            if accelerator.is_local_main_process:
                print(
                    f"Val - CLS Loss: {val_metrics['val_cls_loss']:.4f}, "
                    f"SEG Loss: {val_metrics['val_seg_loss']:.4f}"
                )
                print(
                    f"Val - Macro AUC: {val_metrics['val_macro_auc']:.4f}, "
                    f"Macro AUPRC: {val_metrics['val_macro_auprc']:.4f}"
                )
                print(
                    f"Val - Mean 3D Dice: {val_metrics['val_mean_dice']:.4f}, "
                    f"Hit@5%: {val_metrics['val_hit_at_5']:.4f}, "
                    f"Hit@10%: {val_metrics['val_hit_at_10']:.4f}"
                )

                if val_metrics['val_mean_dice'] > best_dice:
                    best_dice = val_metrics['val_mean_dice']
                    unwrapped_cls = accelerator.unwrap_model(cls_model)
                    unwrapped_seg = accelerator.unwrap_model(seg_model)
                    torch.save(
                        {
                            'epoch': epoch,
                            'cls_model_state': unwrapped_cls.state_dict(),
                            'seg_model_state': unwrapped_seg.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'val_metrics': val_metrics,
                        },
                        os.path.join(save_dir, 'best_model.pt'),
                    )
                    print(f"  -> Saved best model (Mean Dice: {best_dice:.4f})")

        accelerator.wait_for_everyone()

    # --- Generate final results report ---
    if accelerator.is_local_main_process and final_val_metrics is not None:
        generate_report(
            cfg=cfg,
            train_metrics=train_metrics,
            val_metrics=final_val_metrics,
            save_dir=save_dir,
            total_epochs=total_epochs,
        )

    if accelerator.is_local_main_process:
        print(f"\nTraining complete. Best Mean 3D Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
