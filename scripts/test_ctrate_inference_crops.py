import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader
from accelerate import Accelerator

# --- PATH HACK ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataloaders.datasetloader_ctrate_multiscale import CTMultiScaleDataset, MultiScaleSliceProcessor, get_wds_dataset
from utils.config import load_config


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

def test_dataloader(config, num_volumes=10, num_random_slices=5):
    """
    Tests the dataset loader and MultiScaleSliceProcessor by fetching random slices
    and triggering the built-in visualization logic.
    """
    accelerator = Accelerator()

    # Force cropped_global mode for this test just to be safe
    config.inference.mode = 'cropped_global'

    # Set up a dedicated test output directory
    test_output_dir = f"{config.output_folders.main_output}/visualizations"
    os.makedirs(test_output_dir, exist_ok=True)

    # Initialize the processor and artificially inflate max_vis so it covers all random slices
    processor = MultiScaleSliceProcessor(config, output_dir=test_output_dir)
    processor.max_vis = num_volumes * num_random_slices

    dataset = get_dataset(config)
    # Using 0 workers for a quick local test script to avoid multiprocessing overhead
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    print(f"🚀 Starting Test | Visualizations will be saved to: {processor.vis_dir}")

    for vol_idx, batch in enumerate(loader):
        if vol_idx >= num_volumes:
            break

        volumes, labels, filenames = batch
        filename = filenames[0]
        raw_volume = volumes.squeeze(0)
        num_slices = raw_volume.shape[0]

        # Generate random unique indices
        if num_slices <= num_random_slices:
            random_indices = torch.arange(num_slices)
        else:
            random_indices = torch.randperm(num_slices)[:num_random_slices]

        print(f"Processing Volume {vol_idx + 1}/{num_volumes}: {filename} (Random Slices: {random_indices.tolist()})")

        # Process each random slice individually so the processor saves all of them
        for idx in random_indices:
            # Extract single slice and keep the batch dimension (1, H, W)
            single_slice = raw_volume[idx:idx+1].to(accelerator.device)

            # Pass the original index in the filename to make debugging easier
            _ = processor.process_batch(single_slice, filename=f"{filename}_idx_{idx.item()}")

    print("\n✅ Test complete. Check the output directory for your original images, masks, and crops.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # It will default to looking for your provided validation config
    parser.add_argument("--config_file", type=str, default="ctrate_generate_embeddings_valid.yaml")
    args = parser.parse_args()

    config = load_config(args.config_file)
    test_dataloader(config)