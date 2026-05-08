import numpy as np
import os
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import pandas as pd
import yaml
import argparse

def process_and_save(args):
    """
    Worker function: Reads raw (S, P, D), pools it, and saves directly to disk.
    Returns True if successful, None if skipped/failed.
    """
    # Unpack the dynamically passed arguments
    fname, raw_dir, out_base_dir, max_depth = args
    base_name = os.path.splitext(fname.replace('.nii.gz', ''))[0]
    npy_path = os.path.join(raw_dir, base_name + ".npy")

    if not os.path.exists(npy_path):
        return None

    try:
        data = np.load(npy_path)

        # Filter criteria using the config variable
        if data.shape[0] > max_depth or data.ndim != 3:
            return None

        # 1. Compute aggregations
        cls_max = np.max(data[:, 0, :], axis=0)
        cls_mean = np.mean(data[:, 0, :], axis=0)
        patch_max = np.max(data[:, 1:, :], axis=0).flatten()
        patch_mean = np.mean(data[:, 1:, :], axis=0).flatten()

        # 2. Save directly to the respective subdirectories
        np.save(os.path.join(out_base_dir, "cls_max", base_name + ".npy"), cls_max)
        np.save(os.path.join(out_base_dir, "cls_mean", base_name + ".npy"), cls_mean)
        np.save(os.path.join(out_base_dir, "patch_max", base_name + ".npy"), patch_max)
        np.save(os.path.join(out_base_dir, "patch_mean", base_name + ".npy"), patch_mean)

        return True

    except Exception as e:
        return None

def run_precompute(csv_path, raw_emb_dir, out_base_dir, num_workers_req, max_depth):
    df = pd.read_csv(csv_path)

    # Create subdirectories for each strategy
    strategies = ["cls_max", "cls_mean", "patch_max", "patch_mean"]
    for strat in strategies:
        os.makedirs(os.path.join(out_base_dir, strat), exist_ok=True)

    print(f"\n🚀 Precomputing for {out_base_dir}")

    # Pack the tasks with the new max_depth parameter
    tasks = [(row['VolumeName'], raw_emb_dir, out_base_dir, max_depth) for _, row in df.iterrows()]

    # Push it hard, capped safely by the system's limits
    num_workers = min(num_workers_req, multiprocessing.cpu_count())
    print(f"🔥 Using {num_workers} processes. No IPC memory sharing required.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_and_save, tasks),
            total=len(tasks),
            desc="Processing & Saving"
        ))

    successful = sum(1 for r in results if r is True)
    print(f"✅ Successfully precomputed {successful} / {len(df)} volumes.")

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Precompute linear aggregations from raw CT embeddings.")
    parser.add_argument("--config", type=str, default="precompute_config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    # Load parameters
    config = load_config(args.config)
    data_cfg = config['data']
    proc_cfg = config['processing']

    # Process Training Data
    run_precompute(
        csv_path=data_cfg['train_csv'],
        raw_emb_dir=data_cfg['raw_train_dir'],
        out_base_dir=data_cfg['out_train_dir'],
        num_workers_req=proc_cfg['num_workers'],
        max_depth=proc_cfg['max_depth']
    )

    # Process Validation Data
    run_precompute(
        csv_path=data_cfg['val_csv'],
        raw_emb_dir=data_cfg['raw_val_dir'],
        out_base_dir=data_cfg['out_val_dir'],
        num_workers_req=proc_cfg['num_workers'],
        max_depth=proc_cfg['max_depth']
    )

if __name__ == "__main__":
    main()