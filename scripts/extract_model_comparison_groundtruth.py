#!/usr/bin/env python
"""
Extract Ground-Truth Labels for Model Comparison Study.
======================================================

Loads ReXGroundingCT and TotalSegmentator .nii.gz masks for the
414 ReXGroundingCT-annotated scans in the validation split of CT-RATE-huggingface-downloads,
and pre-computes:

  For ReXGroundingCT (14 classes, multi-label):
    - Slice-level binary labels: [D, 14] per volume
    - Patch-level binary labels: [D, 14, grid, grid] per volume
      (mask downsampled to ViT patch grid)

  For TotalSegmentator (104 organ classes, multi-class):
    - Slice-level multi-class labels: [D] per volume (organ ID per slice)
    - Patch-level multi-class labels: [D, grid, grid] per volume

Volume selection is driven by the ReXGroundingCT JSON metadata — only
scans with ReX annotations are used. A 70/10/20 patient-level
(train/val/test) split is applied.

Results are saved as a single .npz per volume to:
    {output_dir}/rex/{volume_name}.npz
    {output_dir}/totalseg/{volume_name}.npz

If the output already exists for a volume, it is skipped (resume support).

Usage:
    python scripts/extract_model_comparison_groundtruth.py \
        --output-dir /app/project/ibi-staff/CT-JEPA/public/outputs/model_comparison_groundtruth/
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

# --- PATH HACK ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REX_CLASSES = [
    "1a", "1b", "1c", "1d", "1e", "1f",
    "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h",
]

# TotalSegmentator organ classes (104 total)
# These are the standard TotalSegmentator v2 labels
TOTALSEG_CLASSES = [
    "spleen", "kidney_right", "kidney_left", "gallbladder",
    "liver", "stomach", "pancreas", "adrenal_gland_right",
    "adrenal_gland_left", "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
    "esophagus", "trachea", "thyroid_gland", "small_bowel",
    "duodenum", "colon", "urinary_bladder", "prostate",
    "kidney_cyst_left", "kidney_cyst_right", "sacrum", "vertebrae_S1",
    "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2",
    "vertebrae_L1", "vertebrae_T12", "vertebrae_T11", "vertebrae_T10",
    "vertebrae_T9", "vertebrae_T8", "vertebrae_T7", "vertebrae_T6",
    "vertebrae_T5", "vertebrae_T4", "vertebrae_T3", "vertebrae_T2",
    "vertebrae_T1", "vertebrae_C7", "vertebrae_C6", "vertebrae_C5",
    "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1",
    "heart", "aorta", "pulmonary_vein", "brachiocephalic_trunk",
    "subclavian_artery_right", "subclavian_artery_left", "common_carotid_artery_right",
    "common_carotid_artery_left", "brachiocephalic_vein_left",
    "brachiocephalic_vein_right", "atrial_appendage_left",
    "superior_vena_cava", "inferior_vena_cava", "portal_vein_and_splenic_vein",
    "iliac_artery_left", "iliac_artery_right", "iliac_vena_left",
    "iliac_vena_right", "humerus_left", "humerus_right", "scapula_left",
    "scapula_right", "clavicula_left", "clavicula_right", "femur_left",
    "femur_right", "hip_left", "hip_right", "spinal_cord",
    "gluteus_maximus_left", "gluteus_maximus_right", "gluteus_medius_left",
    "gluteus_medius_right", "gluteus_minimus_left", "gluteus_minimus_right",
    "autochthon_left", "autochthon_right", "iliopsoas_left", "iliopsoas_right",
    "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4",
    "rib_left_5", "rib_left_6", "rib_left_7", "rib_left_8",
    "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12",
    "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4",
    "rib_right_5", "rib_right_6", "rib_right_7", "rib_right_8",
    "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12",
    "sternum", "costal_cartilages",
]

# Patch grid sizes for each model at 256x256 input
PATCH_GRID_SIZES = {
    "lejepa_0": 16,      # patch_size=16, 256/16=16
    "lejepa_1s": 18,     # patch_size=14, 256/14≈18 (actually 18.28, floor to 18)
    "lejepa_2s": 16,     # patch_size=16, 256/16=16
    "lejepa_1s_v2": 16,  # patch_size=16 (2S arch), 256/16=16
    "dinov2_ct": 18,     # patch_size=14, 256/14≈18
}


# ---------------------------------------------------------------------------
# ReX Ground Truth Extraction
# ---------------------------------------------------------------------------

def _build_finding_to_class_map(rex_json_path: str) -> dict:
    """Parse MLHC_dataset_version.json to get category -> class index mapping."""
    with open(rex_json_path, "r") as f:
        metadata = json.load(f)

    cat_set = set()
    for split_key, entries in metadata.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            categories = entry.get("categories", {})
            for cat in categories.values():
                if cat is not None:
                    cat_set.add(cat)

    cat_to_idx = {sc: i for i, sc in enumerate(REX_CLASSES)}
    for cat in cat_set:
        if cat not in cat_to_idx:
            raise ValueError(f"Unknown category '{cat}' in {rex_json_path}")
    return cat_to_idx


def _get_finding_to_class_for_scan(entry: dict, cat_to_idx: dict) -> dict:
    """Map finding indices to class indices for a single scan."""
    mapping = {}
    categories = entry.get("categories", {})
    for finding_idx_str, cat in categories.items():
        if cat is not None and cat in cat_to_idx:
            mapping[int(finding_idx_str)] = cat_to_idx[cat]
    return mapping


def _resolve_volume_path(filename: str, ct_volumes_dir: str) -> str:
    """Convert mask filename to CT volume path."""
    base = filename.replace(".nii.gz", "")
    parts = base.split("_")
    split_prefix = parts[0]
    patient_id = parts[1]
    study_id = "_".join(parts[1:3])
    return os.path.join(
        ct_volumes_dir, split_prefix,
        f"{split_prefix}_{patient_id}",
        f"{split_prefix}_{study_id}",
        filename,
    )


def extract_rex_groundtruth(
    rex_json_path: str,
    masks_dir: str,
    ct_volumes_dir: str,
    volume_names: list,
    output_dir: str,
    patch_grid_sizes: dict,
):
    """
    Extract ReX slice-level and patch-level labels for each volume.

    Saves: {output_dir}/rex/{volume_name}.npz with keys:
        - cls_labels: [D, 14] float32 (slice-level multi-label)
        - patch_labels_{model}: [D, 14, grid, grid] bool (patch-level, per model)
    """
    os.makedirs(os.path.join(output_dir, "rex"), exist_ok=True)

    cat_to_idx = _build_finding_to_class_map(rex_json_path)

    # Load all JSON entries
    with open(rex_json_path, "r") as f:
        metadata = json.load(f)

    all_entries = []
    for key in metadata:
        all_entries.extend(metadata[key])

    # Build lookup: filename -> entry
    entry_lookup = {e["name"]: e for e in all_entries}

    volume_name_set = set(volume_names)
    matched = 0

    for vol_name in tqdm(sorted(volume_names), desc="ReX ground truth"):
        out_path = os.path.join(output_dir, "rex", f"{vol_name}.npz")
        if os.path.exists(out_path):
            matched += 1
            continue

        if vol_name not in entry_lookup:
            continue

        entry = entry_lookup[vol_name]
        finding_to_class = _get_finding_to_class_for_scan(entry, cat_to_idx)

        # Load mask
        mask_path = os.path.join(masks_dir, vol_name)
        if not os.path.exists(mask_path):
            continue

        try:
            img = nib.load(mask_path)
            mask_data = img.get_fdata(dtype=np.float32)  # [F, H, W, D]
        except Exception as e:
            print(f"  WARNING: Could not load {mask_path}: {e}")
            continue

        if mask_data.ndim != 4:
            print(f"  WARNING: Expected 4D mask, got {mask_data.shape} for {vol_name}")
            continue

        F, H, W, D = mask_data.shape
        mask_data = np.transpose(mask_data, (3, 0, 1, 2))  # [D, F, H, W]

        # Aggregate into 14 channels
        mask_14 = np.zeros((D, 14, H, W), dtype=np.bool_)
        for f_idx in range(F):
            class_idx = finding_to_class.get(f_idx)
            if class_idx is not None:
                mask_14[:, class_idx, :, :] = np.logical_or(
                    mask_14[:, class_idx, :, :],
                    mask_data[:, f_idx, :, :] > 0.5,
                )

        # Slice-level labels: any positive pixel in slice -> positive
        cls_labels = (mask_14.max(axis=(2, 3)) > 0).astype(np.float32)  # [D, 14]

        # Patch-level labels per model grid size
        patch_labels = {}
        for model_key, grid_size in patch_grid_sizes.items():
            # Resize mask from [D, 14, H, W] to [D, 14, grid, grid]
            # Use max pooling: if any pixel in the patch cell is positive, label positive
            patch_mask = np.zeros((D, 14, grid_size, grid_size), dtype=np.bool_)
            for d in range(D):
                for c in range(14):
                    if mask_14[d, c].any():
                        # Downsample via block reduce (max)
                        h_ratio = H / grid_size
                        w_ratio = W / grid_size
                        for gi in range(grid_size):
                            for gj in range(grid_size):
                                h_start = int(gi * h_ratio)
                                h_end = int((gi + 1) * h_ratio)
                                w_start = int(gj * w_ratio)
                                w_end = int((gj + 1) * w_ratio)
                                patch_mask[d, c, gi, gj] = mask_14[
                                    d, c, h_start:h_end, w_start:w_end
                                ].any()
            patch_labels[f"patch_labels_{model_key}"] = patch_mask

        save_dict = {"cls_labels": cls_labels}
        save_dict.update(patch_labels)

        np.savez_compressed(out_path, **save_dict)
        matched += 1

    print(f"[ReX] Processed {matched}/{len(volume_names)} volumes")


# ---------------------------------------------------------------------------
# TotalSegmentator Ground Truth Extraction
# ---------------------------------------------------------------------------

def extract_totalseg_groundtruth(
    totalseg_dir: str,
    volume_names: list,
    output_dir: str,
    patch_grid_sizes: dict,
):
    """
    Extract TotalSegmentator slice-level and patch-level labels.

    Produces 118-class multi-label soft labels (fractional coverage per organ)
    that exactly match the auxiliary head formulation in SoftLabelSupervisionHead.
    Each label value is the fraction of pixels in the region (slice or patch cell)
    belonging to that organ class, making this a direct test of the pre-training
    auxiliary objective.

    TotalSegmentator masks follow the directory structure:
        {totalseg_dir}/{patient_id}/{patient_id}_{scan_id}/{patient_id}_{scan_id}_{recon_id}.nii.gz

    Volume names follow the pattern: {split}_{patient_id}_{scan_id}_{recon_id}.nii.gz
    e.g., valid_1_a_1.nii.gz -> valid_1/valid_1_a/valid_1_a_1.nii.gz

    Saves: {output_dir}/totalseg/{volume_name}.npz with keys:
        - cls_labels: [D, 118] float32 (fractional organ coverage per slice)
        - patch_labels_{model}: [D, 118, grid, grid] float32 (fractional coverage per patch cell)
    """
    NUM_TS_CLASSES = 118  # Matches SoftLabelSupervisionHead.num_ts_classes
    os.makedirs(os.path.join(output_dir, "totalseg"), exist_ok=True)

    matched = 0

    for vol_name in tqdm(sorted(volume_names), desc="TotalSegmentator ground truth"):
        out_path = os.path.join(output_dir, "totalseg", f"{vol_name}.npz")
        if os.path.exists(out_path):
            matched += 1
            continue

        # Parse volume name: {split}_{patient_id}_{scan_id}_{recon_id}
        # e.g., "valid_1_a_1.nii.gz" -> split="valid", pid="1", scan="a", recon="1"
        base = vol_name.replace(".nii.gz", "").replace(".npy", "")
        parts = base.split("_")
        if len(parts) < 4:
            print(f"  WARNING: Unexpected volume name format: {vol_name}")
            continue

        split_prefix = parts[0]  # e.g., "valid"
        patient_id = parts[1]    # e.g., "1"
        scan_id = parts[2]       # e.g., "a"
        recon_id = parts[3]      # e.g., "1"

        # Build path: {totalseg_dir}/{patient_id}/{patient_id}_{scan_id}/{patient_id}_{scan_id}_{recon_id}.nii.gz
        # Note: the actual directory uses patient_id without split prefix
        seg_path = os.path.join(
            totalseg_dir,
            f"{split_prefix}_{patient_id}",
            f"{split_prefix}_{patient_id}_{scan_id}",
            f"{split_prefix}_{patient_id}_{scan_id}_{recon_id}.nii.gz",
        )

        if not os.path.exists(seg_path):
            continue

        try:
            img = nib.load(seg_path)
            # Use np.asarray to handle integer NIfTI data directly
            # (get_fdata with dtype=np.int32 fails because it expects float)
            seg_data = np.asarray(img.dataobj, dtype=np.int32)  # [H, W, D]
        except Exception as e:
            print(f"  WARNING: Could not load {seg_path}: {e}")
            continue

        if seg_data.ndim == 3:
            seg_data = np.transpose(seg_data, (2, 0, 1))  # [D, H, W]
        elif seg_data.ndim == 4:
            seg_data = seg_data[..., 0]  # Take first channel if 4D
            seg_data = np.transpose(seg_data, (2, 0, 1))

        D, H, W = seg_data.shape

        # Slice-level: multi-label soft labels (fractional coverage per organ)
        # For each slice, compute what fraction of foreground pixels belong to each organ class.
        # Organ IDs are 1-indexed (0 = background). We produce a [D, 118] float32 array.
        cls_labels = np.zeros((D, NUM_TS_CLASSES), dtype=np.float32)
        for d in range(D):
            slice_data = seg_data[d]
            fg_mask = slice_data > 0
            fg_pixels = slice_data[fg_mask]
            if len(fg_pixels) > 0:
                unique, counts = np.unique(fg_pixels, return_counts=True)
                total_fg = len(fg_pixels)
                for organ_id, count in zip(unique, counts):
                    if 1 <= organ_id <= NUM_TS_CLASSES:
                        cls_labels[d, organ_id - 1] = count / total_fg

        # Patch-level labels per model grid size
        # For each patch cell, compute fractional organ coverage as a [118] vector.
        patch_labels = {}
        for model_key, grid_size in patch_grid_sizes.items():
            patch_mask = np.zeros((D, NUM_TS_CLASSES, grid_size, grid_size), dtype=np.float32)
            for d in range(D):
                h_ratio = H / grid_size
                w_ratio = W / grid_size
                for gi in range(grid_size):
                    for gj in range(grid_size):
                        h_start = int(gi * h_ratio)
                        h_end = int((gi + 1) * h_ratio)
                        w_start = int(gj * w_ratio)
                        w_end = int((gj + 1) * w_ratio)
                        cell = seg_data[d, h_start:h_end, w_start:w_end]
                        fg_cell = cell[cell > 0]
                        if len(fg_cell) > 0:
                            unique, counts = np.unique(fg_cell, return_counts=True)
                            total_fg = len(fg_cell)
                            for organ_id, count in zip(unique, counts):
                                if 1 <= organ_id <= NUM_TS_CLASSES:
                                    patch_mask[d, organ_id - 1, gi, gj] = count / total_fg
            patch_labels[f"patch_labels_{model_key}"] = patch_mask

        save_dict = {"cls_labels": cls_labels}
        save_dict.update(patch_labels)

        np.savez_compressed(out_path, **save_dict)
        matched += 1

    print(f"[TotalSegmentator] Processed {matched}/{len(volume_names)} volumes")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _get_rex_validation_volumes(rex_json_path: str) -> list:
    """
    Extract all volume names from the ReXGroundingCT JSON that belong to
    the 'valid' split. These are the only scans with ReX annotations.

    Returns:
        List of volume name strings (e.g., 'valid_1_a_1.nii.gz').
    """
    with open(rex_json_path, "r") as f:
        metadata = json.load(f)

    volume_names = []
    for split_key, entries in metadata.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            name = entry.get("name", "")
            # Only include validation-split volumes
            if name.startswith("valid_"):
                volume_names.append(name)

    # Deduplicate (same volume may appear under multiple split keys)
    volume_names = sorted(set(volume_names))
    return volume_names


def main():
    parser = argparse.ArgumentParser(
        description="Extract ground-truth labels for model comparison study"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/app/project/ibi-staff/CT-JEPA/public/outputs/model_comparison_groundtruth/",
        help="Output directory for ground truth .npz files"
    )
    parser.add_argument(
        "--rex-json", type=str,
        default="/app/data/ReXGroundingCT/MLHC_dataset_version.json",
        help="Path to ReX MLHC_dataset_version.json"
    )
    parser.add_argument(
        "--rex-masks-dir", type=str,
        default="/app/data/ReXGroundingCT/segmentations",
        help="Directory containing ReX segmentation masks"
    )
    parser.add_argument(
        "--ct-volumes-dir", type=str,
        default="/app/data/CT-RATE-huggingface-downloads/dataset/train_valid",
        help="Root directory for CT-RATE-huggingface-downloads volumes"
    )
    parser.add_argument(
        "--totalseg-dir", type=str,
        default="/app/project/ibi-staff/CT-JEPA/Process_CT-RATE/UPDATES_CT-RATE/ts_seg/ts_total/valid_fixed/",
        help="Directory containing TotalSegmentator segmentations"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val/test split"
    )
    parser.add_argument(
        "--skip-rex", action="store_true",
        help="Skip ReX ground truth extraction"
    )
    parser.add_argument(
        "--skip-totalseg", action="store_true",
        help="Skip TotalSegmentator ground truth extraction"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Get ReX-annotated validation volumes ---
    # These are the only scans with ReXGroundingCT annotations.
    # We use them as the canonical volume list for everything.
    print(f"Loading ReX validation volumes from {args.rex_json}...")
    volume_names = _get_rex_validation_volumes(args.rex_json)
    print(f"Found {len(volume_names)} ReX-annotated validation volumes")

    # Save the volume list for reference (used by embedding generation)
    vol_list_path = os.path.join(args.output_dir, "volume_names.json")
    with open(vol_list_path, "w") as f:
        json.dump(volume_names, f, indent=2)
    print(f"Saved volume list to {vol_list_path}")

    # --- Patient-level train/val/test split (70/10/20) ---
    # Extract patient ID from volume name: e.g., "valid_1_a_1.nii.gz" -> "1"
    def _extract_patient_id(vol_name):
        base = vol_name.replace(".nii.gz", "").replace(".npy", "")
        parts = base.split("_")
        if len(parts) >= 2:
            return parts[1]  # patient ID is the second component
        return base  # fallback: use entire name

    # Group volumes by patient ID
    patient_to_volumes = {}
    for vn in volume_names:
        pid = _extract_patient_id(vn)
        patient_to_volumes.setdefault(pid, []).append(vn)

    unique_patients = sorted(patient_to_volumes.keys())
    print(f"Found {len(unique_patients)} unique patients across {len(volume_names)} volumes")

    # Deterministic split: 70% train, 10% val, 20% test
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(unique_patients))
    n_train = int(len(unique_patients) * 0.7)
    n_val = int(len(unique_patients) * 0.1)

    train_pids = set(unique_patients[i] for i in perm[:n_train])
    val_pids = set(unique_patients[i] for i in perm[n_train:n_train + n_val])
    test_pids = set(unique_patients[i] for i in perm[n_train + n_val:])

    # Build volume-level split assignments
    split_assignments = {"train": [], "val": [], "test": []}
    for pid, vols in patient_to_volumes.items():
        if pid in train_pids:
            split_assignments["train"].extend(vols)
        elif pid in val_pids:
            split_assignments["val"].extend(vols)
        else:
            split_assignments["test"].extend(vols)

    split_path = os.path.join(args.output_dir, "patient_splits.json")
    with open(split_path, "w") as f:
        json.dump(split_assignments, f, indent=2)
    print(f"Saved patient-level splits to {split_path}")
    print(f"  Train: {len(split_assignments['train'])} volumes ({len(train_pids)} patients)")
    print(f"  Val:   {len(split_assignments['val'])} volumes ({len(val_pids)} patients)")
    print(f"  Test:  {len(split_assignments['test'])} volumes ({len(test_pids)} patients)")

    # Extract ReX ground truth
    if not args.skip_rex:
        print("\n" + "=" * 60)
        print("Extracting ReXGroundingCT ground truth...")
        print("=" * 60)
        extract_rex_groundtruth(
            rex_json_path=args.rex_json,
            masks_dir=args.rex_masks_dir,
            ct_volumes_dir=args.ct_volumes_dir,
            volume_names=volume_names,
            output_dir=args.output_dir,
            patch_grid_sizes=PATCH_GRID_SIZES,
        )

    # Extract TotalSegmentator ground truth
    if not args.skip_totalseg:
        print("\n" + "=" * 60)
        print("Extracting TotalSegmentator ground truth...")
        print("=" * 60)
        extract_totalseg_groundtruth(
            totalseg_dir=args.totalseg_dir,
            volume_names=volume_names,
            output_dir=args.output_dir,
            patch_grid_sizes=PATCH_GRID_SIZES,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
