"""
Benchmark embedding generation driver (DDP).

Extracts per-volume token-sequence embeddings from a frozen benchmark CT
foundation model and saves them as <base_name>.npz (key "embeddings", shape
(seq, D), float32) for direct consumption by the error_bars MIL probe
(dataloaders/dataloader_embeddings.py).

Mirrors scripts/ctrate_generate_embeddings_multimethod.py's Accelerate DDP
loop and resume support, and REUSES the existing data loaders verbatim:
  - ctrate_train : get_wds_dataset            (WebDataset tars, sharded by node/worker)
  - ctrate_valid : CTMultiScaleDataset        (.npy volumes, sharded by accelerator.prepare)
  - rad          : CTMultiScaleDataset        (.npy volumes, sharded by accelerator.prepare)

Each volume is loaded once as raw HU (D, H, W); model-native preprocessing is
applied inside extract_volume_features (NO DALE-CT body-crop).

Usage:
    python scripts/generate_benchmark_embeddings.py \\
        --config configs/benchmark_embeddings.yaml \\
        --model_key rad_dino --split ctrate_valid [--limit 2]
"""

import os
import sys
import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from omegaconf import OmegaConf

# --- PATH HACK (mirror the multimethod driver) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataloaders.datasetloader_ctrate_multiscale import (
    CTMultiScaleDataset,
    get_wds_dataset,
)
from utils.benchmark_backbone_loader import load_benchmark_model, extract_volume_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VALID_SPLITS = ("ctrate_train", "ctrate_valid", "rad", "multisource")


def load_config(config_path):
    return OmegaConf.load(config_path)


def build_dataset(split_cfg, manifest_basenames=None):
    """Reuse the existing loaders verbatim. Returns an iterable/map dataset
    yielding (volume (D,H,W) float32, label, filename).

    manifest_basenames: optional set of bare base-names (no extension, e.g.
        'train_10001_a_1'). If provided, only matching volumes are yielded --
        webdataset via get_wds_dataset's allowed_volume_names (as .nii.gz),
        npy by post-filtering CTMultiScaleDataset.files. Used to extract the
        Z1 5k-subset (4,942 train / 992 valid) for the fair comparison."""
    fmt = getattr(split_cfg, "dataset_format", "npy")
    if fmt == "webdataset":
        # get_wds_dataset reads config.validation.tar_pattern and shards natively.
        allowed = None
        if manifest_basenames is not None:
            allowed = {f"{bn}.nii.gz" for bn in manifest_basenames}
        return get_wds_dataset(split_cfg, allowed_volume_names=allowed)
    elif fmt == "multisource_zarr":
        # multi-source zarr per-case stores: yields raw-HU (n_slab, H, W)
        # slab volumes (slab sliced at READ time -- 3D FMs emit spatial token grids
        # whose axes no longer correspond to input z, so the slab cannot be applied
        # post-embedding like 2D per-slice CLS). Same (volume, label, filename)
        # contract as CTMultiScaleDataset; per-FM norm is applied inside
        # extract_volume_features, so the volume is RAW HU here.
        from dataloaders.dataloader_multisource_zarr_embed import MultisourceZarr3DDataset
        case_csv = split_cfg.validation.case_csv
        if not os.path.exists(case_csv):
            raise FileNotFoundError(f"case_csv not found: {case_csv}")
        ds = MultisourceZarr3DDataset(case_csv=case_csv)
        if manifest_basenames is not None:
            keep = {str(b) for b in manifest_basenames}
            ds.scan_labels = [s for s in ds.scan_labels if s in keep]
        return ds
    data_dir = split_cfg.validation.data_dir
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    # label_csv=None -> dummy labels (we only need the volume + filename here).
    ds = CTMultiScaleDataset(config=split_cfg, data_dir=data_dir, label_csv=None)
    if manifest_basenames is not None:
        before = len(ds.files)
        ds.files = [f for f in ds.files if os.path.splitext(f)[0] in manifest_basenames]
        print(f"Manifest filter: {before} -> {len(ds.files)} volumes in {data_dir}")
    return ds


def exists_valid(path, embed_dim):
    """Resume support: a saved .npz counts as done iff it loads with key
    'embeddings' of shape (seq, embed_dim)."""
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as data:
            if "embeddings" not in data:
                return False
            arr = data["embeddings"]
        return arr.ndim == 2 and arr.shape[1] == embed_dim
    except Exception:
        return False


@torch.no_grad()
def run(model, spec, loader, accelerator, output_dir, embed_dim, limit=None):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    n_done = n_skipped = 0

    for batch in tqdm(loader, disable=not accelerator.is_local_main_process):
        volumes, _labels, filenames = batch
        filename = filenames[0] if isinstance(filenames, (list, tuple)) else filenames
        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(output_dir, f"{base_name}.npz")

        if exists_valid(save_path, embed_dim):
            n_skipped += 1
            continue

        raw_volume = volumes.squeeze(0)  # (D, H, W) or (D, 1, H, W)
        if raw_volume.ndim == 4 and raw_volume.shape[1] == 1:
            raw_volume = raw_volume[:, 0]  # (D, H, W)

        try:
            # autocast engages the config's mixed_precision ("bf16" on the full
            # DGX run); a no-op under "no" (fp32 smoke). Each _extract_* casts
            # to .float() before returning, so saved .npz is always fp32.
            with accelerator.autocast():
                feats = extract_volume_features(model, spec, raw_volume, accelerator.device)
        except Exception as e:
            logger.error(f"[rank {accelerator.process_index}] {base_name} FAILED: {e}")
            continue

        np.savez_compressed(save_path, embeddings=feats)
        n_done += 1
        if n_done <= 3 or n_done % 50 == 0:
            logger.info(
                f"[rank {accelerator.process_index}] {base_name}: saved {feats.shape} -> {save_path}"
            )

        if limit is not None and n_done >= limit:
            break

    logger.info(
        f"[rank {accelerator.process_index}] done: {n_done} extracted, {n_skipped} skipped (resume)."
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark CT foundation model embedding generation")
    parser.add_argument("--config", type=str, default="configs/benchmark_embeddings.yaml")
    parser.add_argument("--model_key", type=str, required=True,
                        help="One of: rad_dino, dinov3, colipri, ct_clip, merlin, ct_fm")
    parser.add_argument("--split", type=str, required=True, choices=list(VALID_SPLITS))
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N new volumes per rank (smoke test).")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to a file with one bare base-name per line (no "
                             "extension). Only matching volumes are processed. Used "
                             "to extract the Z1 5k-subset for the fair comparison.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model_key not in config.models:
        raise KeyError(f"model_key '{args.model_key}' not in config.models: {list(config.models.keys())}")
    if args.split not in config.splits:
        raise KeyError(f"split '{args.split}' not in config.splits: {list(config.splits.keys())}")

    mixed = getattr(config, "mixed_precision", "no")
    accelerator = Accelerator(mixed_precision=(None if mixed == "no" else mixed))
    if accelerator.is_main_process:
        logger.info(f"model_key={args.model_key} split={args.split} mixed_precision={mixed} "
                    f"world_size={accelerator.num_processes}")

    # Build spec override (DGX weights paths, native-preprocess knobs) from config.
    spec_override = OmegaConf.to_container(config.models[args.model_key], resolve=True)

    model, spec = load_benchmark_model(
        args.model_key, device=accelerator.device, spec_override=spec_override
    )
    embed_dim = spec["embed_dim"]

    split_cfg = config.splits[args.split]

    manifest_basenames = None
    if args.manifest:
        with open(args.manifest) as mf:
            manifest_basenames = {line.strip() for line in mf if line.strip()}
        if accelerator.is_main_process:
            logger.info(f"Manifest filter: {args.manifest} -> {len(manifest_basenames)} base-names")

    dataset = build_dataset(split_cfg, manifest_basenames=manifest_basenames)
    num_workers = getattr(split_cfg.validation, "num_workers", 4)

    loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False)
    # WebDataset shards natively (nodesplitter/workersplitter inside get_wds_dataset);
    # only the map-style npy loader is prepared by the accelerator for DDP sharding.
    if getattr(split_cfg, "dataset_format", "npy") != "webdataset":
        loader = accelerator.prepare(loader)

    output_dir = os.path.join(config.output_root, args.model_key, args.split)
    run(model, spec, loader, accelerator, output_dir, embed_dim, limit=args.limit)


if __name__ == "__main__":
    main()
