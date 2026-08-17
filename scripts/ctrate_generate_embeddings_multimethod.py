import os
import sys
import torch
import numpy as np
import argparse
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
import torchvision

# --- PATH HACK ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataloaders.datasetloader_ctrate_multiscale import (
    CTMultiScaleDataset,
    get_wds_dataset,
    get_npy_validation_dataset,
)
from utils.config import load_config, load_model_configs
from utils.dino_utils import init_dino_evaluiaton_model
from utils.hf_backbone_loader import load_backbone
from utils.balanced_subset import (
    select_balanced_patients,
    build_patient_id_set,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def center_crop_to_multiple(tensor, patch_size):
    """
    Center crops the last two dimensions (H, W) of a tensor
    to the nearest multiple of the given patch size.
    """
    h, w = tensor.shape[-2], tensor.shape[-1]
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    if new_h == h and new_w == w:
        return tensor
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return tensor[..., top:top + new_h, left:left + new_w]


def get_dataset(config, allowed_volume_names=None):
    """Load the appropriate dataset based on config.

    Args:
        config: OmegaConf config object.
        allowed_volume_names: Optional set of full VolumeName strings
                              (e.g., 'train_123_a_1.nii.gz') for balanced
                              subset filtering.
    """
    dataset_format = getattr(config.validation, 'dataset_format', 'npy')
    if dataset_format == 'webdataset':
        return get_wds_dataset(config, allowed_volume_names=allowed_volume_names)
    elif dataset_format == 'multisource_zarr':
        # Multi-source zarr: read raw-HU volumes from per-case zarr stores listed
        # in a prep CSV (scan_label, zarr_path). Returns the same (volume, label,
        # filename) contract as CTMultiScaleDataset so run_multi_method_inference
        # unpacks unchanged. zarr is imported lazily inside __getitem__ for fork
        # safety (see dataloaders/dataloader_multisource_zarr_embed.py docstring).
        # balanced_subset is disabled for this format, so allowed_volume_names is
        # None and ignored.
        from dataloaders.dataloader_multisource_zarr_embed import MultisourceZarrEmbedDataset
        case_csv = config.validation.case_csv
        if not os.path.exists(case_csv):
            raise FileNotFoundError(f"case_csv not found: {case_csv}")
        return MultisourceZarrEmbedDataset(case_csv=case_csv)
    else:
        data_dir = config.validation.data_dir
        label_csv = getattr(config.validation, 'label_csv', None)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        # Use the NPY validation dataset helper which supports volume-name filtering
        if allowed_volume_names is not None:
            return get_npy_validation_dataset(
                config, data_dir, label_csv,
                max_patients=len(allowed_volume_names),
                allowed_volume_names=allowed_volume_names
            )
        ds = CTMultiScaleDataset(config=config, data_dir=data_dir, label_csv=label_csv)
        return ds


class MultiMethodProcessor:
    """
    Applies multiple preprocessing strategies to the same batch of slices.
    Each method is configured via the 'methods' dict in the config.

    Supported method configurations:
      - crop: bool          Whether to apply body masking before resize
      - resize_target: int|null  Target size (null = patch-aligned or original)
      - extract_patches: bool    Whether to extract patch tokens (vs CLS only)
      - tile_mode: str|null  If "5cut", applies 5-cut tiling after body crop
      - tile_size: int       Size for each tile cut
    """

    def __init__(self, config, patch_size, output_dir=None):
        self.config = config
        self.patch_size = patch_size
        self.output_dir = output_dir

        # Normalization mode: 'z_score' (default), 'physical', or 'dinomx'
        # 'dinomx' = v628/DINOMX native pipeline: clip to [-1024, 3071], then
        # z-score with pinned corpus stats (mean=-15.9617, std=220.4706) directly
        # on HU values (no intermediate 0-1 mapping).
        self.norm_mode = getattr(config.dataset, 'normalization_mode', 'z_score').lower()

        # Normalization parameters (shared across all methods)
        self.clip_min = getattr(config.dataset, 'clip_min', -997.0)
        self.clip_max = getattr(config.dataset, 'clip_max', 888.0)
        self.mean_hu = getattr(config.dataset, 'mean_hu', -142.39)
        self.std_hu = getattr(config.dataset, 'std_hu', 360.97)
        range_val = self.clip_max - self.clip_min if (self.clip_max - self.clip_min) > 0 else 1.0
        self.norm_mean = (self.mean_hu - self.clip_min) / range_val
        self.norm_std = self.std_hu / range_val
        self.num_channels = getattr(config.dataset, 'num_channels', 1)

        # Parse enabled methods
        self.methods = {}
        if hasattr(config, 'methods'):
            for name, cfg in config.methods.items():
                if getattr(cfg, 'enabled', False):
                    self.methods[name] = cfg
        logger.info(f"Enabled methods: {list(self.methods.keys())}")

    def normalize(self, volume):
        """Normalization matching the model's pre-training pipeline."""
        if self.norm_mode == 'dinomx':
            # v628/DINOMX native pipeline (dinomx/extract_embeddings.py):
            #   1. Clip to [-1024, 3071] (corpus HU range)
            #   2. Z-score with pinned corpus stats directly on HU values
            #      (mean=-15.9617, std=220.4706 — no intermediate 0-1 mapping)
            volume = torch.clamp(volume, self.clip_min, self.clip_max)
            volume = (volume - self.mean_hu) / self.std_hu
            return volume
        elif self.norm_mode == 'div1000':
            # LeJEPA div1000 variants: clamp(HU/1000, -1, 1).
            # Matches training (dataloaders/datasetloader_web_ctrate.py
            # normalize_slab: slab/1000.0 then clamp(-1,1); GPU augmentor sets
            # final_norm=Identity, input_range='-1_1'). Backbone sees [-1,1].
            # clip_min/max/mean_hu/std_hu are IGNORED in this mode.
            return torch.clamp(volume / 1000.0, -1.0, 1.0)
        else:
            # Original TAP-CT/LeJEPA: clip → 0-1 map → z-score
            volume = torch.clamp(volume, self.clip_min, self.clip_max)
            range_val = self.clip_max - self.clip_min
            volume = (volume - self.clip_min) / (range_val if range_val > 0 else 1.0)
            volume = (volume - self.norm_mean) / self.norm_std
            return volume

    def compute_body_mask(self, slices):
        """
        GPU-batched morphological body masking.
        Args:
            slices: (B, 1, H, W) raw HU values
        Returns:
            body_mask: (B, 1, H, W) boolean mask
        """
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
        """
        Find a single global square bounding box from body masks across all slices.
        Args:
            body_mask: (B, 1, H, W) boolean
        Returns:
            (rmin, rmax, cmin, cmax) as tensors
        """
        B, C, H, W = body_mask.shape
        global_body_mask = body_mask.any(dim=0, keepdim=True)  # (1, 1, H, W)

        rows = global_body_mask.any(dim=3).squeeze(1)  # (1, H)
        cols = global_body_mask.any(dim=2).squeeze(1)  # (1, W)
        valid = rows.any(dim=1)  # (1,)

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
        """
        Crop a square ROI from slices and resize to target_size.
        Uses the same global bounding box for all slices (3D-aligned).
        """
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

    def five_cut(self, image, tile_size):
        """
        Takes a square image (B, C, H, W) and produces 5 overlapping crops:
        four corners + center. Each cut is resized to tile_size.
        The input image should already be square (H == W).
        """
        B, C, H, W_img = image.shape
        half = H // 2

        cuts = []
        # Top-left
        cuts.append(image[:, :, :half, :half])
        # Top-right
        cuts.append(image[:, :, :half, half:])
        # Bottom-left
        cuts.append(image[:, :, half:, :half])
        # Bottom-right
        cuts.append(image[:, :, half:, half:])
        # Center
        ch = H // 4
        cuts.append(image[:, :, ch:ch + half, ch:ch + half])

        # Resize each cut to tile_size
        resized_cuts = []
        for cut in cuts:
            r = torch.nn.functional.interpolate(
                cut, size=(tile_size, tile_size),
                mode='bilinear', align_corners=False
            )
            resized_cuts.append(r)

        return torch.stack(resized_cuts, dim=1)  # (B, 5, C, tile_size, tile_size)

    def _compute_target_size(self, current_side, method_cfg):
        """
        Compute the target size for a method, respecting max_size cap.

        Args:
            current_side: the natural side length (bbox side or image side)
            method_cfg: method configuration
        Returns:
            target: integer target size (square), patch-aligned
        """
        max_size = getattr(method_cfg, 'max_size', None)
        # Align to nearest patch_size multiple
        target = max(self.patch_size, (current_side // self.patch_size) * self.patch_size)
        if max_size is not None and target > max_size:
            target = max_size
        return target

    def process_method(self, slices, method_name, method_cfg, full_res_target=None,
                       cached_bbox=None):
        """
        Apply a single preprocessing method to slices.

        Args:
            slices: (B, 1, H, W) raw HU values
            method_name: name of the method
            method_cfg: config for this method
            full_res_target: (H, W) tuple for consistent full_resolution sizing.
                             If provided and method is full_resolution, all slices
                             are resized to this size instead of per-chunk cropping.
            cached_bbox: (rmin, rmax, cmin, cmax) tuple from a prior body_mask
                         computation. When provided, skips compute_body_mask().
        Returns:
            processed: tensor ready for model input
            metadata: dict with info about the processing
            bbox: (rmin, rmax, cmin, cmax) tuple for reuse by other cropped methods
        """
        B, C_orig, H, W = slices.shape
        do_crop = getattr(method_cfg, 'crop', False)
        resize_target = getattr(method_cfg, 'resize_target', None)
        tile_mode = getattr(method_cfg, 'tile_mode', None)
        tile_size = getattr(method_cfg, 'tile_size', 256)

        metadata = {'method': method_name, 'input_shape': (H, W)}
        bbox = None

        if do_crop:
            # Compute body mask and global bounding box (or reuse cached)
            if cached_bbox is not None:
                rmin, rmax, cmin, cmax = cached_bbox
            else:
                body_mask = self.compute_body_mask(slices)
                rmin, rmax, cmin, cmax = self.get_global_bbox(body_mask)
            bbox = (rmin, rmax, cmin, cmax)

            if tile_mode == '5cut':
                # Crop to square ROI first, then 5-cut
                # Use an intermediate size that's a multiple of 2 for clean halving
                intermediate_size = tile_size * 2  # e.g. 512 for 256 tiles
                cropped = self.crop_square_roi(slices, rmin, rmax, cmin, cmax, intermediate_size)
                # Now 5-cut the cropped result
                cuts = self.five_cut(cropped, tile_size)  # (B, 5, C, tile_size, tile_size)
                # Normalize
                cuts = self.normalize(cuts)
                if self.num_channels == 3 and cuts.shape[2] == 1:
                    cuts = cuts.repeat(1, 1, 3, 1, 1)
                metadata['tile_mode'] = '5cut'
                metadata['tile_size'] = tile_size
                metadata['intermediate_crop'] = intermediate_size
                return cuts, metadata, bbox
            else:
                # Standard crop + resize
                if resize_target is not None:
                    target = resize_target
                elif full_res_target is not None:
                    # Use the pre-computed consistent target (e.g. from first chunk)
                    target = full_res_target[0]  # assume square
                    metadata['locked_target'] = target
                else:
                    # Resize to nearest multiple of patch_size, capped at max_size
                    bbox_h = (rmax - rmin + 1).item()
                    bbox_w = (cmax - cmin + 1).item()
                    side = max(bbox_h, bbox_w)
                    target = self._compute_target_size(side, method_cfg)
                    metadata['auto_target'] = target
                    metadata['bbox_side'] = side

                cropped = self.crop_square_roi(slices, rmin, rmax, cmin, cmax, target)
                normalized = self.normalize(cropped)
                if self.num_channels == 3 and normalized.shape[1] == 1:
                    normalized = normalized.repeat(1, 3, 1, 1)
                metadata['target_size'] = target
                return normalized, metadata, bbox
        else:
            # No crop: just resize (or keep original)
            if resize_target is not None:
                resized = torch.nn.functional.interpolate(
                    slices, size=(resize_target, resize_target),
                    mode='bilinear', align_corners=False
                )
                normalized = self.normalize(resized)
                metadata['target_size'] = resize_target
            else:
                # Full resolution: use consistent target size if provided,
                # otherwise compute patch-aligned size capped at max_size
                if full_res_target is not None:
                    target_h, target_w = full_res_target
                    resized = torch.nn.functional.interpolate(
                        slices, size=(target_h, target_w),
                        mode='bilinear', align_corners=False
                    )
                    normalized = self.normalize(resized)
                    metadata['target_size'] = (target_h, target_w)
                else:
                    # Compute target: patch-aligned, capped at max_size
                    max_side = max(H, W)
                    target = self._compute_target_size(max_side, method_cfg)
                    # Resize to target (square) then normalize
                    resized = torch.nn.functional.interpolate(
                        slices, size=(target, target),
                        mode='bilinear', align_corners=False
                    )
                    normalized = self.normalize(resized)
                    metadata['target_size'] = target

            if self.num_channels == 3 and normalized.shape[1] == 1:
                normalized = normalized.repeat(1, 3, 1, 1)
            return normalized, metadata, None


@torch.no_grad()
def run_multi_method_inference(model, dataloader, accelerator, config, output_dir, slice_batch_size=256):
    """
    Run inference with multiple preprocessing methods on each volume.
    Each volume is loaded once; all methods are applied to the same slices.
    """
    model.eval()

    # patch_size: HF-backed models carry it in config.train.patch_size (mirrors
    # BACKBONE_SPECS); local-ckpt models read it from the model_type JSON registry.
    patch_size = getattr(config.train, 'patch_size', None)
    if patch_size is None:
        model_params = load_model_configs(config.train.model_type)
        patch_size = model_params.patch_size
    logger.info(f"Model patch_size: {patch_size}")

    processor = MultiMethodProcessor(config, patch_size=patch_size, output_dir=output_dir)
    enabled_methods = processor.methods

    if not enabled_methods:
        raise ValueError("No methods enabled in config! Set at least one method's 'enabled: true'.")

    # Identify which methods need patch tokens vs CLS only
    patch_token_methods = set()
    cls_only_methods = set()
    for name, cfg in enabled_methods.items():
        if getattr(cfg, 'extract_patches', False):
            patch_token_methods.add(name)
        else:
            cls_only_methods.add(name)

    logger.info(f"Patch-token methods: {patch_token_methods}")
    logger.info(f"CLS-only methods: {cls_only_methods}")

    # Robustly identify timm models
    unwrapped_model = getattr(model, '_orig_mod', getattr(model, 'module', model))
    is_timm = hasattr(unwrapped_model, 'forward_features')

    # Create output directories per method
    for method_name in enabled_methods:
        method_dir = os.path.join(output_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)

    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        volumes, labels, filenames = batch
        filename = filenames[0]
        base_name = os.path.splitext(filename)[0]

        raw_volume = torch.from_numpy(np.ascontiguousarray(volumes)).squeeze(0)  # (D, H, W) or (D, 1, H, W)
        if raw_volume.ndim == 3 and raw_volume.shape[1] != raw_volume.shape[2]:
            # (D, H, W) -> add channel dim
            raw_volume = raw_volume.unsqueeze(1)  # (D, 1, H, W)
        elif raw_volume.ndim == 3:
            raw_volume = raw_volume.unsqueeze(1)

        num_slices = raw_volume.shape[0]

        # Check if all methods already have output for this volume (resume support)
        all_done = True
        for method_name in enabled_methods:
            save_path = os.path.join(output_dir, method_name, f"{base_name}.npz")
            if not os.path.exists(save_path):
                all_done = False
                break
        if all_done:
            continue

        # Pre-compute consistent target sizes for methods where the output
        # patch count could vary across slice chunks. This is needed for:
        #   - full_resolution: different H/W after center_crop_to_multiple
        #   - cropped_patch_aligned: body bbox size varies across slices
        # We run the first slice chunk through all variable-size methods to
        # determine the target sizes, then lock them for all subsequent chunks.
        # Also caches the body bbox from the first cropped method to avoid
        # redundant morphological operations on subsequent cropped methods.
        # NOTE: Only needed when extract_patches=true. Skipped for CLS-only.
        consistent_targets = {}
        needs_precompute = any(
            getattr(cfg, 'extract_patches', False) and getattr(cfg, 'resize_target', None) is None
            for cfg in enabled_methods.values()
        )
        if needs_precompute:
            first_chunk = raw_volume[0:min(slice_batch_size, num_slices)].to(accelerator.device)
            cached_bbox = None  # Reused across all cropped methods
            for method_name, method_cfg in enabled_methods.items():
                extract_patches = getattr(method_cfg, 'extract_patches', False)
                resize_target = getattr(method_cfg, 'resize_target', None)
                do_crop = getattr(method_cfg, 'crop', False)
                if extract_patches and resize_target is None:
                    # This method has variable output size — pre-compute target
                    processed, _, bbox = processor.process_method(
                        first_chunk, method_name, method_cfg, full_res_target=None,
                        cached_bbox=cached_bbox
                    )
                    # Cache bbox from first cropped method for reuse
                    if do_crop and bbox is not None and cached_bbox is None:
                        cached_bbox = bbox
                    if processed.ndim == 5:
                        # Tiled: target is implicit in the 5-cut pipeline (fixed)
                        pass
                    else:
                        # Standard: record the H, W from the first chunk
                        consistent_targets[method_name] = (processed.shape[2], processed.shape[3])
                        logger.info(
                            f"Volume {base_name}: locked {method_name} target = "
                            f"{consistent_targets[method_name]}"
                        )

        # Initialize accumulators for each method
        method_results = {name: [] for name in enabled_methods}
        method_metadata = {name: None for name in enabled_methods}

        for i in range(0, num_slices, slice_batch_size):
            end_idx = min(i + slice_batch_size, num_slices)
            slice_chunk = raw_volume[i:end_idx].to(accelerator.device)

            # Reset bbox cache per chunk (bbox is computed from this chunk's slices)
            chunk_bbox = None
            for method_name, method_cfg in enabled_methods.items():
                # Determine the consistent target for this method
                locked_target = consistent_targets.get(method_name, None)
                do_crop = getattr(method_cfg, 'crop', False)
                processed, meta, bbox = processor.process_method(
                    slice_chunk, method_name, method_cfg,
                    full_res_target=locked_target,
                    cached_bbox=chunk_bbox if do_crop else None
                )
                # Cache bbox from first cropped method for subsequent cropped methods
                if do_crop and bbox is not None and chunk_bbox is None:
                    chunk_bbox = bbox
                if method_metadata[method_name] is None:
                    method_metadata[method_name] = meta

                extract_patches = getattr(method_cfg, 'extract_patches', False)

                if processed.ndim == 5:
                    # Tiled: (B, N_views, C, H, W)
                    B, N, C, H_img, W_img = processed.shape
                    flat = processed.reshape(B * N, C, H_img, W_img)

                    with accelerator.autocast():
                        if is_timm:
                            out = unwrapped_model.forward_features(flat)
                        else:
                            out = model(flat)
                            if hasattr(out, 'last_hidden_state'):
                                out = out.last_hidden_state

                    # CLS token only for tiled methods
                    if out.ndim == 3:
                        cls_tokens = out[:, 0, :]  # (B*N, D)
                    else:
                        cls_tokens = out
                    cls_tokens = cls_tokens.view(B, N, -1)  # (B, N, D)
                    method_results[method_name].append(cls_tokens.cpu().numpy())

                else:
                    # Standard: (B, C, H, W)
                    with accelerator.autocast():
                        if is_timm and extract_patches:
                            out = unwrapped_model.forward_features(processed)
                            if out.ndim == 4:
                                out = out.flatten(2).transpose(1, 2)
                        elif is_timm and not extract_patches:
                            out = unwrapped_model.forward_features(processed)
                            if out.ndim == 3:
                                out = out[:, 0, :]  # CLS only
                        else:
                            out = model(processed)
                            if hasattr(out, 'last_hidden_state'):
                                if extract_patches:
                                    out = out.last_hidden_state
                                else:
                                    out = out.last_hidden_state[:, 0, :]

                    method_results[method_name].append(out.cpu().numpy())

        # Concatenate and save results for each method
        for method_name in enabled_methods:
            if method_results[method_name]:
                full_embs = np.concatenate(method_results[method_name], axis=0)
                save_path = os.path.join(output_dir, method_name, f"{base_name}.npz")
                meta = method_metadata[method_name] or {}
                # Build a clean metadata dict for saving
                save_kwargs = {'embeddings': full_embs, 'method': method_name}
                for k, v in meta.items():
                    save_kwargs[f'meta_{k}'] = str(v)
                np.savez_compressed(save_path, **save_kwargs)

    logger.info(f"Rank {accelerator.process_index} inference complete.")


def _numpy_volume_collate(batch):
    """Collate that returns volumes as numpy instead of a shm-shared tensor.

    PyTorch's default_collate shares the collated volume tensor across worker→main
    via /dev/shm (`_new_shared`), which hits a probabilistic create/unlink race
    (pytorch#101185) under DataLoader churn — this crashed every num_workers>0
    embed run ("could not unlink"/"unable to open shared memory object ... No such
    file or directory"). Building numpy directly (`v.numpy()` is a zero-copy view of
    the per-sample tensor, `np.stack` copies into a fresh array) never enters the
    shm path, so the numpy is simply pickled (copied) to the main process. This
    keeps num_workers>0 (prefetch → fast) while removing the shm mechanism entirely.
    The main loop converts back to a tensor via torch.from_numpy before GPU forward.
    """
    volumes = [b[0] for b in batch]
    labels = [b[1] for b in batch]          # unused downstream; preserved as-is
    filenames = [b[2] for b in batch]
    vols_np = np.stack([
        np.ascontiguousarray(v.numpy() if isinstance(v, torch.Tensor) else v)
        for v in volumes
    ])
    return vols_np, labels, filenames


def main():
    parser = argparse.ArgumentParser(
        description="Multi-method embedding generation from CT-RATE-huggingface-downloads dataset"
    )
    parser.add_argument(
        "--config_file", type=str,
        default="configs/generate_embeddings_multimethod.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--slice_batch_size", type=int, default=256,
        help="Batch size for slices processed at once on GPU"
    )
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")
    config = load_config(args.config_file)

    output_dir = config.output_folders.main_output
    os.makedirs(output_dir, exist_ok=True)

    # Log which methods are enabled
    if hasattr(config, 'methods'):
        enabled = [n for n, c in config.methods.items() if getattr(c, 'enabled', False)]
        logger.info(f"Running {len(enabled)} methods: {enabled}")
    else:
        logger.warning("No 'methods' section found in config!")

    # --- Balanced Subset Selection ---
    allowed_volume_names = None
    if hasattr(config, 'balanced_subset') and getattr(config.balanced_subset, 'enabled', False):
        bs = config.balanced_subset
        logger.info(f"Balanced subset enabled: selecting {bs.n_patients} patients from {bs.label_csv}")
        selected_volumes = select_balanced_patients(
            csv_path=bs.label_csv,
            n_patients=bs.n_patients,
            seed=getattr(bs, 'seed', 42)
        )
        allowed_volume_names = build_patient_id_set(selected_volumes)
        logger.info(f"Balanced subset: {len(selected_volumes)} volumes selected "
                    f"({len(allowed_volume_names)} unique volume names)")
    else:
        logger.info("Balanced subset disabled — processing full dataset.")

    # Load model once. HF-backed models (lejepa_1s/2s, dinov2_ct) load via
    # load_backbone(model_key); local-ckpt models (chest_z1full, 1s_v2) use the
    # existing init_dino_evaluiaton_model path keyed on model_type + vit_ckpt_path.
    if getattr(config.train, 'model_key', None):
        model, model_spec = load_backbone(config.train.model_key)
        logger.info(f"Loaded HF backbone '{config.train.model_key}' via load_backbone "
                    f"(backbone_type={model_spec['backbone_type']}, "
                    f"patch_size={model_spec['patch_size']}, embed_dim={model_spec['embed_dim']})")
    else:
        model = init_dino_evaluiaton_model(config, accelerator)
    model = model.to(accelerator.device)
    model.eval()

    # NOTE: torch.compile is intentionally disabled here because the multi-method
    # pipeline feeds 6+ different input shapes (512, 256, 144, variable patch-aligned)
    # to the model, causing recompilation on every shape change and massive overhead.
    # For single-shape inference, re-enable with: model = torch.compile(model)

    # Load dataset
    dataset = get_dataset(config, allowed_volume_names=allowed_volume_names)
    loader = DataLoader(
        dataset, batch_size=1,
        num_workers=config.validation.num_workers,
        shuffle=False,
        collate_fn=_numpy_volume_collate,
    )

    # Don't prepare WebDataset with accelerator (handles sharding natively)
    dataset_format = getattr(config.validation, 'dataset_format', 'npy')
    if dataset_format != 'webdataset':
        loader = accelerator.prepare(loader)

    run_multi_method_inference(
        model, loader, accelerator, config, output_dir,
        slice_batch_size=args.slice_batch_size
    )


if __name__ == "__main__":
    main()
