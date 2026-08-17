import torch
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


class CTScanMILDataset(Dataset):
    def __init__(self, df, embedding_dir, label_cols, id_col='VolumeName', file_ext='.nii.gz', preload_ram=True,
                 only_global=False, num_cache_workers=8):
        self.embedding_dir = embedding_dir
        self.preload_ram = preload_ram
        self.only_global = only_global
        self.label_cols = label_cols
        self.id_col = id_col
        self.file_ext = file_ext
        self.num_cache_workers = num_cache_workers

        valid_indices = []
        self.data_cache = []
        self.labels_cache = []
        self.names_cache = []

        print(f"Loading and verifying files from {embedding_dir}...")

        # First pass: resolve paths and filter to existing files (cheap os.path.exists
        # only). The heavy np.load + zlib decompress is deferred to a thread pool
        # below — numpy releases the GIL during file I/O and decompression, so this
        # turns the single-threaded 19 it/s cache into a parallel load.
        candidates = []  # (orig_idx, row, load_path, is_npz, base_name)
        for idx, row in df.iterrows():
            fname = str(row[self.id_col])

            # Handle filename parsing based on the dataset structure
            if self.file_ext:
                base_name = os.path.splitext(fname.replace(self.file_ext, ''))[0]
            else:
                base_name = fname

            # Support both .npy (single array) and .npz (multi-array, key='embeddings')
            npy_path = os.path.join(self.embedding_dir, base_name + ".npy")
            npz_path = os.path.join(self.embedding_dir, base_name + ".npz")

            if os.path.exists(npy_path):
                candidates.append((idx, row, npy_path, False, base_name))
            elif os.path.exists(npz_path):
                candidates.append((idx, row, npz_path, True, base_name))

        # Second pass: load + decompress in parallel. Each job returns the raw array
        # (sliced for only_global) or None on failure; caching/labeling happens in
        # the main thread below, preserving the original valid_indices ordering.
        def _load(item):
            idx, row, load_path, is_npz, base_name = item
            try:
                load_mode = 'r' if self.only_global else None
                if is_npz:
                    archive = np.load(load_path, allow_pickle=True)
                    # Multi-method .npz files store embeddings under the 'embeddings' key
                    if 'embeddings' in archive:
                        data = archive['embeddings']
                    else:
                        # Fallback: use the first array in the archive
                        data = archive[list(archive.keys())[0]]
                else:
                    data = np.load(load_path, mmap_mode=load_mode)

                if self.only_global and data.ndim == 3:
                    data = data[:, 0, :]
                return (idx, row, data, base_name)
            except Exception as e:
                print(f"Warning: could not read {load_path}: {e}")
                return None

        if self.num_cache_workers and self.num_cache_workers > 1 and len(candidates) > 1:
            with ThreadPoolExecutor(max_workers=self.num_cache_workers) as ex:
                results = list(tqdm(ex.map(_load, candidates), total=len(candidates), desc="Caching Dataset"))
        else:
            results = [_load(c) for c in tqdm(candidates, desc="Caching Dataset")]

        for r in results:
            if r is None:
                continue
            idx, row, data, base_name = r
            valid_indices.append(idx)

            if self.preload_ram:
                bag_features = data.reshape(-1, data.shape[-1]).copy()
                features = torch.from_numpy(bag_features).float()

                labels_np = row[self.label_cols].values.astype(np.float32)
                label_tensor = torch.tensor(labels_np, dtype=torch.float32)

                self.data_cache.append(features)
                self.labels_cache.append(label_tensor)
                self.names_cache.append(base_name)

        print(f"Matched {len(valid_indices)} / {len(df)} volumes.")
        self.df = df.loc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.preload_ram:
            return self.data_cache[idx], self.labels_cache[idx], self.names_cache[idx]
        else:
            row = self.df.iloc[idx]
            fname = str(row[self.id_col])
            base_name = os.path.splitext(fname.replace(self.file_ext, ''))[0] if self.file_ext else fname

            # Support both .npy (single array) and .npz (multi-array, key='embeddings')
            npy_path = os.path.join(self.embedding_dir, base_name + ".npy")
            npz_path = os.path.join(self.embedding_dir, base_name + ".npz")

            try:
                if os.path.exists(npy_path):
                    load_mode = 'r' if self.only_global else None
                    data = np.load(npy_path, mmap_mode=load_mode)
                elif os.path.exists(npz_path):
                    archive = np.load(npz_path, allow_pickle=True)
                    if 'embeddings' in archive:
                        data = archive['embeddings']
                    else:
                        data = archive[list(archive.keys())[0]]
                else:
                    raise FileNotFoundError(f"Neither {npy_path} nor {npz_path} found")

                if self.only_global and data.ndim == 3:
                    data = data[:, 0, :]

                bag_features = data.reshape(-1, data.shape[-1]).copy()
                features = torch.from_numpy(bag_features).float()
            except Exception as e:
                features = torch.zeros((1, 1024), dtype=torch.float)

            labels_np = row[self.label_cols].values.astype(np.float32)
            label_tensor = torch.tensor(labels_np, dtype=torch.float32)

            return features, label_tensor, base_name


def collate_mil_bags(batch):
    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    names_list = [item[2] for item in batch]

    labels = torch.stack(labels_list)
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    batch_size = len(features_list)
    input_dim = features_list[0].shape[1]

    # Build padded tensors on the same device as the input bags. When bags are
    # GPU-resident (cache_bags_on_gpu), this avoids re-creating a CPU padded
    # tensor every batch + a host->GPU copy. CPU inputs are unaffected.
    _dev = features_list[0].device
    padded_features = torch.zeros((batch_size, max_len, input_dim), dtype=torch.float, device=_dev)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=_dev)

    for i, feat in enumerate(features_list):
        end = lengths[i]
        padded_features[i, :end, :] = feat
        mask[i, :end] = True

    return padded_features, labels, names_list, mask


def _create_ctrate_datasets(config):
    data_cfg = config['data']
    only_global = data_cfg.get('only_global', False)
    seed = config['experiment'].get('seed', 42)
    label_cols = config['experiment']['ct_rate_classes']
    num_cache_workers = config['experiment'].get('num_cache_workers', 8)

    full_train_df = pd.read_csv(data_cfg['train_label_path'])
    full_train_df['PatientID'] = full_train_df['VolumeName'].apply(
        lambda x: x.split("_")[1] if len(x.split("_")) >= 2 else x
    )

    unique_patients = full_train_df['PatientID'].unique()
    np.random.seed(seed)
    val_patient_ids = np.random.choice(unique_patients, size=1000, replace=False)

    val_df = full_train_df[full_train_df['PatientID'].isin(val_patient_ids)].reset_index(drop=True)
    train_df = full_train_df[~full_train_df['PatientID'].isin(val_patient_ids)].reset_index(drop=True)

    test_df = pd.read_csv(data_cfg['val_label_path'])

    train_dataset = CTScanMILDataset(train_df, data_cfg['train_embedding_dir'], label_cols, only_global=only_global,
                                     num_cache_workers=num_cache_workers)
    val_dataset = CTScanMILDataset(val_df, data_cfg['train_embedding_dir'], label_cols, only_global=only_global,
                                   num_cache_workers=num_cache_workers)
    test_dataset = CTScanMILDataset(test_df, data_cfg['val_embedding_dir'], label_cols, only_global=only_global,
                                    num_cache_workers=num_cache_workers)

    return train_dataset, val_dataset, test_dataset


def _create_rad_datasets(config):
    data_cfg = config['data']
    only_global = data_cfg.get('only_global', False)
    label_cols = config['experiment']['rad_chestct_classes']
    num_cache_workers = config['experiment'].get('num_cache_workers', 8)

    csv_path = data_cfg['rad_label_path']
    emb_dir = data_cfg['rad_embedding_dir']

    print(f"Reading RAD-ChestCT labels from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Split based on prefix in the NoteAcc_DEID column
    train_df = df[df['NoteAcc_DEID'].str.startswith('trn')].reset_index(drop=True)
    val_df = df[df['NoteAcc_DEID'].str.startswith('val')].reset_index(drop=True)
    test_df = df[df['NoteAcc_DEID'].str.startswith('tst')].reset_index(drop=True)

    print(f"--- RAD-ChestCT Split Summary ---")
    print(f"Train Volumes: {len(train_df)}")
    print(f"Valid Volumes: {len(val_df)}")
    print(f"Test Volumes:  {len(test_df)}")
    print(f"---------------------------------\n")

    # Note the id_col and file_ext kwargs specific to RAD-ChestCT
    train_dataset = CTScanMILDataset(train_df, emb_dir, label_cols, id_col='NoteAcc_DEID', file_ext='',
                                     only_global=only_global, num_cache_workers=num_cache_workers)
    val_dataset = CTScanMILDataset(val_df, emb_dir, label_cols, id_col='NoteAcc_DEID', file_ext='',
                                   only_global=only_global, num_cache_workers=num_cache_workers)
    test_dataset = CTScanMILDataset(test_df, emb_dir, label_cols, id_col='NoteAcc_DEID', file_ext='',
                                    only_global=only_global, num_cache_workers=num_cache_workers)

    return train_dataset, val_dataset, test_dataset


def create_datasets(config):
    """Router function to build datasets based on the yaml config."""
    dataset_type = config['data'].get('dataset_type', 'ct_rate')

    if dataset_type == 'rad_chestct':
        return _create_rad_datasets(config)
    else:
        return _create_ctrate_datasets(config)