"""Fair-subset CT-RATE evaluation for the e2e unfreeze-ablation checkpoints.

Stage 0 of the fair redo: load e2e_model_best.pth for a given variant config,
rebuild the model EXACTLY as train_e2e_lora.py does, and evaluate on the same
992-patient fair valid set the frozen-backbone linear-probe baselines used
(error_bars_fair_subset). The 992 = base-names in
features/ctrate_fair_subset/manifest_valid.txt, mapped to raw-HU .npy volumes
in CT-RATE_valid_hu/. This is the SAME .npy source the baseline embeddings were
extracted from, so model inputs are byte-identical to the baselines' (modulo the
e2e's own preprocessing, which matches its training). NO WebDataset, NO NIfTI
re-read -- both would diverge from the baseline input path.

Model construction mirrors train_e2e_lora.py:
  - base ViT loaded from config.train.vit_ckpt_path (the 2S iter_50000 backbone)
  - LoRA applied iff finetune_method == 'lora'
  - EndToEndColipri built with use_patch_pooling / use_slice_transformer flags
  - freeze_layers called for partial_unfreeze (matches training; frozen blocks
    stay at vit_ckpt_path values, absent from the checkpoint, loaded strict=False)

Usage (in-container):
  python scripts/e2e_eval_fair.py \
    --config /app/project/ibi-staff/CT-JEPA/public/outputs/e2e_unfreeze_ablation/n8/config.yaml \
    --val-manifest /app/project/ibi-staff/CT-JEPA/features/ctrate_fair_subset/manifest_valid.txt
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# --- PATH HACK ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

from dataloaders.datasetloader_ctrate_multiscale import (
    get_npy_validation_dataset,
    MultiScaleSliceProcessor,
)
from utils.config import load_config
from utils.dino_utils import init_dino_evaluiaton_model, freeze_layers
from models.e2e_colipri import EndToEndColipri


# CT-RATE 18-class label order (matches pos_weights in finetune_unfreeze.yaml).
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


def load_e2e_model(config, accelerator, checkpoint_override=None):
    """Rebuild the e2e model as in train_e2e_lora.py and load e2e_model_best.pth.

    Loads the trainable state_dict (LoRA adapters OR unfrozen backbone blocks +
    arch modules + Colipri head) with strict=False so frozen backbone params
    (absent from the checkpoint) retain their vit_ckpt_path values.
    """
    # The trainer saves e2e_model_best.pth into GlobalState['e2e_save_dir'] =
    # output_folders.main_output / train.model_name / output_folders.e2e_save_dir
    # (e.g. .../e2e_unfreeze_ablation/n8/checkpoints/e2e_model_best.pth) -- NOT
    # config.experiment.save_dir, which is the parent dir without the variant.
    if checkpoint_override:
        best_model_path = checkpoint_override
    else:
        best_model_path = os.path.join(
            config.output_folders.main_output,
            config.train.model_name,
            config.output_folders.e2e_save_dir,
            "e2e_model_best.pth",
        )
    if not os.path.isfile(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
    if accelerator.is_main_process:
        print(f"Loading best model from: {best_model_path}")

    # 1. Base ViT backbone from the 2S pretrained checkpoint.
    base_vit = init_dino_evaluiaton_model(config, accelerator)

    # 2. LoRA ONLY for the lora variant (full / partial_unfreeze train the
    #    backbone directly, no adapters).
    finetune_method = getattr(config.experiment, "finetune_method", "lora")
    if finetune_method == "lora":
        lora_config = LoraConfig(
            r=config.experiment.lora_r,
            lora_alpha=config.experiment.lora_alpha,
            lora_dropout=config.experiment.lora_dropout,
            target_modules=["qkv"],
        )
        base_vit = get_peft_model(base_vit, lora_config, adapter_name="default")
        if accelerator.is_main_process:
            print("Applied LoRA adapter (lora variant)")

    # 3. EndToEndColipri with the new architecture flags from the config.
    use_patch_pooling = getattr(config.experiment, "use_patch_pooling", False)
    use_slice_transformer = getattr(config.experiment, "use_slice_transformer", False)
    slice_transformer_config = getattr(config.experiment, "slice_transformer", None)
    multi_cfg = getattr(config, "multi_adapter", None)
    model = EndToEndColipri(
        vit_backbone=base_vit,
        colipri_state_dict_path=None,
        input_dim=config.experiment.input_dim,
        pooling_scheme=config.experiment.best_pooling_scheme,
        multi_adapter_config=multi_cfg,
        use_patch_pooling=use_patch_pooling,
        use_slice_transformer=use_slice_transformer,
        slice_transformer_config=slice_transformer_config,
    )
    if accelerator.is_main_process:
        print(f"Built EndToEndColipri: patch_pooling={use_patch_pooling} "
              f"slice_transformer={use_slice_transformer}")

    # 4. partial_unfreeze: freeze first N blocks to match training construction
    #    (so the param set is identical; eval doesn't use grads but this keeps
    #    the state_dict key layout consistent with how it was saved).
    if finetune_method == "partial_unfreeze":
        n_freeze = getattr(config.experiment, "unfreeze_layers", 0)
        freeze_layers(model, accelerator, use_lora=False, num_layers_to_freeze=n_freeze)
        if accelerator.is_main_process:
            print(f"partial_unfreeze: froze first {n_freeze} blocks")

    # 5. Load the trained trainable state_dict (strict=False: frozen backbone
    #    params are absent from the checkpoint and keep their pretrained values).
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
        # 2. Suffix fallback (handles PEFT / wrapping prefix differences)
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
        print(f"Loaded {len(matched_state)}/{len(state_dict)} checkpoint tensors; "
              f"{len(missing)} model tensors left at backbone-init, "
              f"{len(unexpected)} checkpoint tensors unused")

    model = accelerator.prepare(model)
    return model


def load_fair_valid_volumes(manifest_path):
    """Read manifest_valid.txt (992 base-names, no extension) -> set of full
    VolumeName strings ('valid_<pid>_<scan>_<recon>.nii.gz') for
    get_npy_validation_dataset's allowed_volume_names filter."""
    with open(manifest_path) as f:
        base_names = [line.strip() for line in f if line.strip()]
    allowed = {f"{bn}.nii.gz" for bn in base_names}
    print(f"Fair valid manifest: {len(allowed)} volumes from {manifest_path}")
    return allowed


@torch.no_grad()
def run_inference(model, dataloader, processor, accelerator, config):
    """Run distributed inference over the 992-volume fair valid set."""
    model.eval()
    chunk_size = getattr(config.experiment, "chunk_size", 32)
    device = accelerator.device

    all_probs, all_labels = [], []
    n = len(dataloader.dataset) if hasattr(dataloader, "dataset") else "?"
    if accelerator.is_main_process:
        print(f"\nInference on {n} fair-valid volumes (chunk_size={chunk_size})...")

    for batch_idx, (volumes, labels, filenames) in enumerate(dataloader):
        raw_volume = volumes.squeeze(0).to(device)
        labels = labels.to(device)
        processed_slices, _ = processor.process_batch(raw_volume)
        processed_slices = processed_slices.unsqueeze(0)  # (1, S, C, H, W)

        logits = model(processed_slices, chunk_size=chunk_size)
        probs = torch.sigmoid(logits)

        gathered_probs, gathered_labels = accelerator.gather_for_metrics((probs, labels))
        if accelerator.is_main_process:
            all_probs.append(gathered_probs.cpu().numpy())
            all_labels.append(gathered_labels.cpu().numpy())
            if (batch_idx + 1) % 50 == 0:
                print(f"  [{batch_idx + 1}/{n}] processed", flush=True)

    if accelerator.is_main_process:
        return (np.concatenate(all_probs, axis=0),
                np.concatenate(all_labels, axis=0))
    return None, None


def compute_metrics(probs, labels, class_names):
    per_auroc, per_auprc = [], []
    for c in range(labels.shape[1]):
        pos = labels[:, c].sum()
        if pos == 0 or pos == labels.shape[0]:
            per_auroc.append(0.5)
            per_auprc.append(0.0)
        else:
            per_auroc.append(roc_auc_score(labels[:, c], probs[:, c]))
            per_auprc.append(average_precision_score(labels[:, c], probs[:, c]))
    macro_auroc = float(np.mean(per_auroc))
    macro_auprc = float(np.mean(per_auprc))

    print("\n" + "=" * 70)
    print(f"  {'':>38s} {'AUROC':>8s} {'AUPRC':>8s}")
    print(f"  {'-'*38} {'-'*8} {'-'*8}")
    for c, name in enumerate(class_names):
        print(f"  {name:>38s} {per_auroc[c]:8.4f} {per_auprc[c]:8.4f}")
    print(f"  {'-'*38} {'-'*8} {'-'*8}")
    print(f"  {'MACRO':>38s} {macro_auroc:8.4f} {macro_auprc:8.4f}")
    print("=" * 70)
    return {
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "per_class_auroc": per_auroc,
        "per_class_auprc": per_auprc,
    }


def main():
    parser = argparse.ArgumentParser(description="Fair-subset e2e eval")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the variant config YAML (the one training used)")
    parser.add_argument("--val-manifest", type=str, required=True,
                        help="Path to manifest_valid.txt (992 fair-valid base-names)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override path to e2e_model_best.pth (else derived from config)")
    # The training config's val_data_dir/val_label_csv point at CT-RATE_train_hu
    # + train_predicted_labels.csv (the 300 ckpt-val). The 992 fair-valid TEST
    # lives in CT-RATE_valid_hu + valid_predicted_labels.csv -- these overrides
    # repoint the eval there without duplicating the config.
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override val_data_dir (e.g. CT-RATE_valid_hu for the 992 test)")
    parser.add_argument("--label-csv", type=str, default=None,
                        help="Override val_label_csv (e.g. valid_predicted_labels.csv for the 992 test)")
    parser.add_argument("--output-tag", type=str, default=None,
                        help="If set, write fair_eval_results_<tag>.npz (else fair_eval_results.npz). "
                             "Disambiguates the 992 vs full-3002 evals, which share an out_dir.")
    args = parser.parse_args()

    config = load_config(args.config)
    accelerator = Accelerator(mixed_precision="bf16")
    if accelerator.is_main_process:
        print(f"Device: {accelerator.device} | processes: {accelerator.num_processes}")
        print(f"finetune_method: {getattr(config.experiment, 'finetune_method', 'lora')}")

    # 1. Model
    model = load_e2e_model(config, accelerator, checkpoint_override=args.checkpoint)

    # 2. Processor (same normalization + cropped_global + max_slices as training)
    processor = MultiScaleSliceProcessor(config)

    # 3. Fair valid set: the exact 992 from the manifest
    allowed_volume_names = load_fair_valid_volumes(args.val_manifest)
    val_data_dir = args.data_dir or config.validation.val_data_dir
    val_label_csv = args.label_csv or getattr(config.validation, "val_label_csv", None)
    val_dataset = get_npy_validation_dataset(
        config=config,
        data_dir=val_data_dir,
        label_csv=val_label_csv,
        max_patients=len(allowed_volume_names),  # ignored when allowed_volume_names set
        seed=config.experiment.seed,
        allowed_volume_names=allowed_volume_names,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1,
        num_workers=getattr(config.validation, "num_workers", 4),
        shuffle=False, pin_memory=True,
    )
    val_loader = accelerator.prepare(val_loader)
    if accelerator.is_main_process:
        print(f"Fair valid dataset: {len(val_dataset)} volumes")

    # 4. Inference + metrics
    probs, labels = run_inference(model, val_loader, processor, accelerator, config)
    if accelerator.is_main_process:
        metrics = compute_metrics(probs, labels, CTRATE_CLASS_NAMES)
        # Write results into the variant's own dir (next to config.yaml), not the
        # shared parent -- otherwise the 3 variants would overwrite each other.
        out_dir = os.path.join(config.output_folders.main_output,
                               config.train.model_name)
        _fname = (f"fair_eval_results_{args.output_tag}.npz"
                  if args.output_tag else "fair_eval_results.npz")
        np.savez(
            os.path.join(out_dir, _fname),
            probs=probs, labels=labels,
            class_names=np.array(CTRATE_CLASS_NAMES),
            macro_auroc=metrics["macro_auroc"],
            macro_auprc=metrics["macro_auprc"],
            per_class_auroc=np.array(metrics["per_class_auroc"]),
            per_class_auprc=np.array(metrics["per_class_auprc"]),
        )
        print(f"\nSaved fair_eval_results.npz to {out_dir}")
        print(f"FAIR VALID 992 | macro AUROC={metrics['macro_auroc']:.4f} "
              f"AUPRC={metrics['macro_auprc']:.4f}")


if __name__ == "__main__":
    main()
