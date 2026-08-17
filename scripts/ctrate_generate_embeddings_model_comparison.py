#!/usr/bin/env python
"""
Model Comparison Study: Multi-Model Embedding Generation (Validation Split).
============================================================================

Generates CLS + patch token embeddings for a single backbone on the
ReXGroundingCT-annotated validation scans (414 volumes) using the
cropped_256 preprocessing.

Unlike ctrate_generate_embeddings_multimethod.py which runs multiple
preprocessing methods on one model, this script runs one preprocessing
method (cropped_256) on one model specified via --model_key.

Each volume's embeddings are saved as a single .npz:
    {output_root}/{model_key}/cropped_256/{volume_name}.npz

Keys in the .npz:
    - 'cls':  [D, embed_dim] float32  (CLS token per slice)
    - 'patch': [D, N_patches, embed_dim] float32  (patch tokens per slice)

Usage:
    python scripts/ctrate_generate_embeddings_model_comparison.py \
        --config configs/generate_embeddings_model_comparison.yaml \
        --model_key lejepa_0
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# --- PATH HACK ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataloaders.datasetloader_ctrate_multiscale import (
    CTMultiScaleDataset,
    get_npy_validation_dataset,
)
from utils.config import load_config
from utils.hf_backbone_loader import (
    load_backbone,
    extract_cls_token,
    extract_patch_tokens,
    BACKBONE_SPECS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing (cropped_256 only)
# ---------------------------------------------------------------------------

class Cropped256Preprocessor:
    """
    Body-cropped square ROI, resized to 256x256, normalized.
    Matches the cropped_256 method from MultiMethodProcessor.
    """

    def __init__(self, clip_min=-997.0, clip_max=888.0, mean_hu=-142.39, std_hu=360.97):
        self.clip_min = clip_min
        self.clip_max = clip_max
        range_val = clip_max - clip_min if (clip_max - clip_min) > 0 else 1.0
        self.norm_mean = (mean_hu - clip_min) / range_val
        self.norm_std = std_hu / range_val

    def normalize(self, volume):
        """Z-score normalization."""
        volume = torch.clamp(volume, self.clip_min, self.clip_max)
        range_val = self.clip_max - self.clip_min
        volume = (volume - self.clip_min) / (range_val if range_val > 0 else 1.0)
        volume = (volume - self.norm_mean) / self.norm_std
        return volume

    def compute_body_mask(self, slices):
        """GPU-batched morphological body masking."""
        solid_tissue = (slices > -500).float()
        kernel_size = 25
        pad = kernel_size // 2
        eroded = 1.0 - torch.nn.functional.max_pool2d(
            1.0 - solid_tissue, kernel_size=kernel_size, stride=1, padding=pad
        )
        body_mask = torch.nn.functional.max_pool2d(
            eroded, kernel_size=kernel_size, stride=1, padding=pad
        ) > 0.5
        return body_mask

    def get_global_bbox(self, body_mask):
        """Find a single global square bounding box from body masks."""
        B, C, H, W = body_mask.shape
        global_body_mask = body_mask.any(dim=0, keepdim=True)

        rows = global_body_mask.any(dim=3).squeeze(1)
        cols = global_body_mask.any(dim=2).squeeze(1)
        valid = rows.any(dim=1)

        rmin = torch.argmax(rows.float(), dim=1)
        rmax = H - 1 - torch.argmax(torch.flip(rows, dims=[1]).float(), dim=1)
        cmin = torch.argmax(cols.float(), dim=1)
        cmax = W - 1 - torch.argmax(torch.flip(cols, dims=[1]).float(), dim=1)

        rmin = torch.where(valid, rmin, torch.zeros_like(rmin))
        rmax = torch.where(valid, rmax, torch.full_like(rmax, H - 1))
        cmin = torch.where(valid, cmin, torch.zeros_like(cmin))
        cmax = torch.where(valid, cmax, torch.full_like(cmax, W - 1))

        return rmin, rmax, cmin, cmax

    def crop_square_roi(self, slices, rmin, rmax, cmin, cmax, target_size):
        """Crop a square ROI and resize to target_size."""
        import torchvision
        B = slices.shape[0]
        H, W = slices.shape[2], slices.shape[3]

        bbox_h = rmax - rmin + 1
        bbox_w = cmax - cmin + 1
        side = torch.max(bbox_h, bbox_w)

        center_r = (rmin + rmax) // 2
        center_c = (cmin + cmax) // 2

        sq_rmin = center_r - side // 2
        sq_cmin = center_c - side // 2

        sq_rmin = torch.clamp(sq_rmin, min=0)
        sq_rmax = sq_rmin + side
        overflow_r = sq_rmax - H
        sq_rmin = torch.where(overflow_r > 0, sq_rmin - overflow_r, sq_rmin)
        sq_rmax = torch.where(overflow_r > 0, torch.full_like(sq_rmax, H), sq_rmax)
        sq_rmin = torch.clamp(sq_rmin, min=0)

        sq_cmin = torch.clamp(sq_cmin, min=0)
        sq_cmax = sq_cmin + side
        overflow_c = sq_cmax - W
        sq_cmin = torch.where(overflow_c > 0, sq_cmin - overflow_c, sq_cmin)
        sq_cmax = torch.where(overflow_c > 0, torch.full_like(sq_cmax, W), sq_cmax)
        sq_cmin = torch.clamp(sq_cmin, min=0)

        batch_idx = torch.arange(B, device=slices.device).unsqueeze(1).float()
        boxes = torch.cat([
            batch_idx,
            sq_cmin.repeat(B, 1).float(),
            sq_rmin.repeat(B, 1).float(),
            sq_cmax.repeat(B, 1).float(),
            sq_rmax.repeat(B, 1).float()
        ], dim=1).to(dtype=slices.dtype)

        resized = torchvision.ops.roi_align(
            slices, boxes,
            output_size=(target_size, target_size),
            spatial_scale=1.0,
            aligned=True
        )
        return resized

    def process(self, slices, target_size=256, patch_aligned_target=None):
        """
        Apply cropped_256 preprocessing to a chunk of slices.

        Args:
            slices: Input tensor [B, C, H, W]
            target_size: Initial resize target (e.g., 256)
            patch_aligned_target: If set, does a second nearest-neighbor
                resize to this size to ensure divisibility by patch_size.
                E.g., 252 for patch_size=14 models (252 = 18*14).
        """
        body_mask = self.compute_body_mask(slices)
        rmin, rmax, cmin, cmax = self.get_global_bbox(body_mask)
        cropped = self.crop_square_roi(slices, rmin, rmax, cmin, cmax, target_size)
        normalized = self.normalize(cropped)

        # Patch-aligned resize for models with patch_size that doesn't divide target_size
        # (e.g., patch_size=14 doesn't divide 256; resize 256→252 via nearest)
        if patch_aligned_target is not None and patch_aligned_target != target_size:
            normalized = torch.nn.functional.interpolate(
                normalized,
                size=(patch_aligned_target, patch_aligned_target),
                mode='nearest',
            )

        return normalized


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate_npz(save_path, expected_keys=('cls', 'patch')):
    """
    Validate that an .npz file is complete and not corrupt.

    Returns True if the file exists, can be loaded, and contains all
    expected keys with non-empty arrays. Returns False (and deletes
    the file) if the file is missing, corrupt, or incomplete.
    """
    if not os.path.exists(save_path):
        return False
    try:
        data = np.load(save_path, allow_pickle=False)
        for key in expected_keys:
            if key not in data:
                logger.warning(
                    "Corrupt .npz (missing key '%s'): %s -- deleting and re-processing",
                    key, save_path
                )
                data.close()
                os.remove(save_path)
                return False
            arr = data[key]
            if not isinstance(arr, np.ndarray) or arr.size == 0:
                logger.warning(
                    "Corrupt .npz (empty/invalid '%s'): %s -- deleting and re-processing",
                    key, save_path
                )
                data.close()
                os.remove(save_path)
                return False
        data.close()
        return True
    except Exception as e:
        logger.warning(
            "Corrupt .npz (%s): %s -- deleting and re-processing",
            e, save_path
        )
        try:
            os.remove(save_path)
        except OSError:
            pass
        return False


@torch.no_grad()
def run_embedding_extraction(model, model_spec, dataloader, output_dir,
                              slice_batch_size=256, patch_aligned_target=None):
    """
    Extract CLS + patch token embeddings for all volumes.

    Saves: {output_dir}/{volume_name}.npz with keys 'cls' and 'patch'.

    Args:
        patch_aligned_target: If set, the preprocessor will do a second
            nearest-neighbor resize to this size to ensure divisibility
            by the model's patch_size (e.g., 252 for patch_size=14).
    """
    model.eval()
    device = next(model.parameters()).device

    backbone_type = model_spec["backbone_type"]
    num_register_tokens = model_spec["num_register_tokens"]
    embed_dim = model_spec["embed_dim"]

    # Norm is per-model: read from the backbone spec (clip_min/clip_max/mean_hu/
    # std_hu) so chest (DALE-CT-0-L, trained with full-pool z-score stats) is
    # normalized with its own stats. Existing CT-RATE-trained models lack these
    # fields and fall back to the CT-RATE-0 defaults below — identical to the
    # prior hardcoded behavior.
    clip_min = model_spec.get("clip_min", -997.0)
    clip_max = model_spec.get("clip_max", 888.0)
    mean_hu = model_spec.get("mean_hu", -142.39)
    std_hu = model_spec.get("std_hu", 360.97)
    preprocessor = Cropped256Preprocessor(
        clip_min=clip_min, clip_max=clip_max, mean_hu=mean_hu, std_hu=std_hu
    )
    logger.info(
        f"Preprocessor norm (model={model_spec.get('display_name', '?')}): "
        f"clip[{clip_min}, {clip_max}] mean_hu={mean_hu} std_hu={std_hu}"
    )

    os.makedirs(output_dir, exist_ok=True)

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        volumes, labels, filenames = batch
        filename = filenames[0]
        base_name = os.path.splitext(filename)[0]

        save_path = os.path.join(output_dir, f"{base_name}.npz")
        if _validate_npz(save_path):
            continue

        raw_volume = volumes.squeeze(0)  # (D, H, W) or (D, 1, H, W)
        if raw_volume.ndim == 3 and raw_volume.shape[1] != raw_volume.shape[2]:
            raw_volume = raw_volume.unsqueeze(1)
        elif raw_volume.ndim == 3:
            raw_volume = raw_volume.unsqueeze(1)

        num_slices = raw_volume.shape[0]

        all_cls = []
        all_patch = []

        for i in range(0, num_slices, slice_batch_size):
            end_idx = min(i + slice_batch_size, num_slices)
            slice_chunk = raw_volume[i:end_idx].to(device)

            # Preprocess: cropped_256 (with optional patch-aligned resize)
            processed = preprocessor.process(
                slice_chunk, target_size=256,
                patch_aligned_target=patch_aligned_target
            )

            # Extract CLS token
            cls_tokens = extract_cls_token(
                model, processed, backbone_type, num_register_tokens
            )  # [B, embed_dim]

            # Extract patch tokens
            patch_tokens = extract_patch_tokens(
                model, processed, backbone_type, num_register_tokens
            )  # [B, N_patches, embed_dim]

            all_cls.append(cls_tokens.cpu().numpy())
            all_patch.append(patch_tokens.cpu().numpy())

        cls_array = np.concatenate(all_cls, axis=0)  # [D, embed_dim]
        patch_array = np.concatenate(all_patch, axis=0)  # [D, N_patches, embed_dim]

        np.savez_compressed(save_path, cls=cls_array, patch=patch_array)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate CLS + patch embeddings for model comparison study"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to generate_embeddings_model_comparison.yaml"
    )
    parser.add_argument(
        "--model_key", type=str, required=True,
        choices=["lejepa_0", "lejepa_1s", "lejepa_2s", "lejepa_1s_v2", "dinov2_ct", "lejepa_0_chest"],
        help="Which backbone to use"
    )
    parser.add_argument(
        "--slice-batch-size", type=int, default=256,
        help="Number of slices to process at once"
    )
    args = parser.parse_args()

    config = load_config(config_name=args.config)
    model_key = args.model_key

    logger.info(f"Model: {model_key}")
    logger.info(f"Config: {args.config}")

    # Load backbone from HuggingFace
    model, model_spec = load_backbone(model_key)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load the ReX-annotated volume list (generated by extract_model_comparison_groundtruth.py)
    volume_names_path = config.volume_list
    logger.info(f"Loading volume list from {volume_names_path}...")
    with open(volume_names_path, "r") as f:
        volume_names = json.load(f)
    logger.info(f"Loaded {len(volume_names)} ReX-annotated validation volumes")

    # Load dataset
    data_dir = config.validation.data_dir
    label_csv = getattr(config.validation, 'label_csv', None)

    ds = get_npy_validation_dataset(
        config, data_dir, label_csv,
        max_patients=len(volume_names),
        allowed_volume_names=volume_names
    )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=getattr(config.validation, 'num_workers', 4),
        pin_memory=True,
    )

    # Output directory
    output_root = config.output_folders.main_output
    output_dir = os.path.join(output_root, model_key, "cropped_256")
    logger.info(f"Output directory: {output_dir}")

    # Read patch_aligned_target from config (if present)
    # For patch_size=14 models, this is 252 (floor(256/14)*14)
    # For patch_size=16 models, this is None (256 is already divisible by 16)
    patch_aligned_target = None
    if hasattr(config, 'methods') and hasattr(config.methods, 'cropped_256'):
        patch_aligned_target = getattr(config.methods.cropped_256, 'patch_aligned_target', None)
    if patch_aligned_target is not None:
        logger.info(f"Patch-aligned resize target: {patch_aligned_target} "
                     f"(model patch_size={model_spec['patch_size']})")

    # Run extraction
    run_embedding_extraction(
        model=model,
        model_spec=model_spec,
        dataloader=dataloader,
        output_dir=output_dir,
        slice_batch_size=args.slice_batch_size,
        patch_aligned_target=patch_aligned_target,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
