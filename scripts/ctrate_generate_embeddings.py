import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator

# --- PATH HACK ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataloaders.datasetloader_ctrate_multiscale import CTMultiScaleDataset, MultiScaleSliceProcessor, get_wds_dataset
from utils.config import load_config, load_model_configs
from utils.dino_utils import init_dino_evaluiaton_model


def center_crop_to_multiple(tensor, patch_size):
    """
    Center crops the last two dimensions (H, W) of a tensor
    to the nearest multiple of the given patch size.
    """
    h, w = tensor.shape[-2], tensor.shape[-1]

    # Calculate the nearest valid multiple
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size

    # Skip if already perfectly divisible
    if new_h == h and new_w == w:
        return tensor

    # Calculate starting indices for a perfect center crop
    top = (h - new_h) // 2
    left = (w - new_w) // 2

    # Slices using ellipsis to support both (B, H, W) and (B, C, H, W) shapes
    return tensor[..., top:top + new_h, left:left + new_w]


def get_dataset(config):
    dataset_format = getattr(config.validation, 'dataset_format', 'npy')

    if dataset_format == 'webdataset':
        # Load the new webdataset pipeline
        return get_wds_dataset(config)
    else:
        # Fallback to the original implementation
        data_dir = config.validation.data_dir

        # safely get label_csv, defaulting to None if it doesn't exist
        label_csv = getattr(config.validation, 'label_csv', None)

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        ds = CTMultiScaleDataset(config=config, data_dir=data_dir, label_csv=label_csv)
        return ds

@torch.no_grad()
def run_inference_3d(model, dataloader, accelerator, config, output_dir, slice_batch_size=256):
    model.eval()

    emb_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    processor = MultiScaleSliceProcessor(config, output_dir=output_dir)
    inference_mode = getattr(config.inference, 'mode', 'tiled')

    # Check for the new config flag (default to False for backward compatibility)
    extract_patches = getattr(config.inference, 'extract_patches', False)

    # Robustly identify timm models, even if wrapped by torch.compile or DDP
    unwrapped_model = getattr(model, '_orig_mod', getattr(model, 'module', model))
    is_timm = hasattr(unwrapped_model, 'forward_features')

    accelerator.print(
        f"Starting Inference | Mode: {inference_mode} | Patches: {extract_patches} | Batch: {slice_batch_size}...")

    def get_features(x):
        """Helper to safely extract features based on model type and config."""
        if is_timm and extract_patches:
            # Bypass compile wrapper to call specific timm method safely
            out = unwrapped_model.forward_features(x)
            # Handle edge case if a CNN is loaded (returns B, D, H, W instead of sequence)
            if out.ndim == 4:
                out = out.flatten(2).transpose(1, 2)
            return out
        else:
            out = model(x)
            if hasattr(out, 'last_hidden_state'):
                if extract_patches:
                    return out.last_hidden_state  # Keep all patches (B, P, D)
                else:
                    return out.last_hidden_state[:, 0, :]  # Keep CLS only (B, D)
            return out

    patch_size = load_model_configs(config.train.model_type).patch_size
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        volumes, labels, filenames = batch
        filename = filenames[0]
        save_name = os.path.splitext(filename)[0] + ".npy"
        save_path = os.path.join(output_dir, save_name)

        # Validate existing .npy — delete if corrupt/partial (e.g., from interrupted run)
        if os.path.exists(save_path):
            try:
                _ = np.load(save_path, allow_pickle=False)
                continue
            except Exception:
                accelerator.print(
                    f"Corrupt .npy detected, deleting and re-processing: {save_path}"
                )
                os.remove(save_path)

        raw_volume = volumes.squeeze(0)
        num_slices = raw_volume.shape[0]

        volume_embeddings = []

        for i in range(0, num_slices, slice_batch_size):
            end_idx = min(i + slice_batch_size, num_slices)
            slice_chunk = raw_volume[i:end_idx].to(accelerator.device)

            primary_view, secondary_view = processor.process_batch(slice_chunk, filename=filename)
            primary_view = center_crop_to_multiple(primary_view, patch_size=patch_size)

            with accelerator.autocast():
                # 1. Run Primary View
                out_primary = get_features(primary_view)

                # 2. Run Secondary View (Tiles), if applicable
                if secondary_view is not None:
                    B, N, C, H, W = secondary_view.shape
                    flat_local = secondary_view.reshape(-1, C, H, W)

                    out_local = get_features(flat_local)

                    if extract_patches:
                        # Reshape local patches: (B*N, P, D) -> (B, N, P, D)
                        _, P, D_emb = out_local.shape
                        out_local = out_local.view(B, N, P, D_emb)

                        # Align primary view: (B, P, D) -> (B, 1, P, D)
                        out_primary = out_primary.unsqueeze(1)
                    else:
                        # Reshape local CLS tokens: (B*N, D) -> (B, N, D)
                        _, D_emb = out_local.shape
                        out_local = out_local.view(B, N, D_emb)

                        # Align primary view: (B, D) -> (B, 1, D)
                        out_primary = out_primary.unsqueeze(1)

                    # Concatenate along the view dimension
                    batch_embs = torch.cat([out_primary, out_local], dim=1)
                else:
                    batch_embs = out_primary

            volume_embeddings.append(batch_embs.cpu().numpy())

        # Concatenate across the slice dimension
        full_volume_embs = np.concatenate(volume_embeddings, axis=0)
        np.save(save_path, full_volume_embs)

    print(f"✅ Rank {accelerator.process_index} inference complete. Embeddings saved to {emb_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings from CT-RATE-huggingface-downloads dataset")
    parser.add_argument("--config_file", type=str, default="ctrate_embeddings.yaml", help="Path to config")
    parser.add_argument("--slice_batch_size", type=int, default=32, help="Batch size for slices")
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")
    config = load_config(args.config_file)

    output_dir = config.output_folders.main_output
    os.makedirs(output_dir, exist_ok=True)

    model = init_dino_evaluiaton_model(config, accelerator)
    model = model.to(accelerator.device)
    model.eval()

    if hasattr(torch, "compile"):
        # Compile is generally safe for fixed size inputs
        model = torch.compile(model)

    dataset = get_dataset(config)
    loader = DataLoader(dataset, batch_size=1, num_workers=config.validation.num_workers, shuffle=False)

    # --- FIX: Only prepare the loader if we are using the old NPY dataset ---
    # Accelerate tries to dispatch IterableDatasets centrally, which crashes on variable depth CTs.
    # WebDataset already handles multi-GPU sharding natively.
    dataset_format = getattr(config.validation, 'dataset_format', 'npy')
    if dataset_format != 'webdataset':
        loader = accelerator.prepare(loader)

    run_inference_3d(model, loader, accelerator, config, output_dir, slice_batch_size=args.slice_batch_size)

if __name__ == "__main__":
    main()