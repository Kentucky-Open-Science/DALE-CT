import torch
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class CTScanMILDataset(Dataset):
    def __init__(self, df, embedding_dir, label_cols, id_col='VolumeName', file_ext='.nii.gz', preload_ram=True,
                 only_global=False):
        self.embedding_dir = embedding_dir
        self.preload_ram = preload_ram
        self.only_global = only_global
        self.label_cols = label_cols
        self.id_col = id_col
        self.file_ext = file_ext

        valid_indices = []
        self.data_cache = []
        self.labels_cache = []
        self.names_cache = []

        print(f"Loading and verifying files from {embedding_dir}...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Caching Dataset"):
            fname = str(row[self.id_col])

            # Handle filename parsing based on the dataset structure
            if self.file_ext:
                base_name = os.path.splitext(fname.replace(self.file_ext, ''))[0]
            else:
                base_name = fname

            npy_path = os.path.join(self.embedding_dir, base_name + ".npy")

            if os.path.exists(npy_path):
                try:
                    load_mode = 'r' if self.only_global else None
                    data = np.load(npy_path, mmap_mode=load_mode)

                    if self.only_global and data.ndim == 3:
                        data = data[:, 0, :]

                    valid_indices.append(idx)

                    if self.preload_ram:
                        bag_features = data.reshape(-1, data.shape[-1]).copy()
                        features = torch.from_numpy(bag_features).float()

                        labels_np = row[self.label_cols].values.astype(np.float32)
                        label_tensor = torch.tensor(labels_np, dtype=torch.float32)

                        self.data_cache.append(features)
                        self.labels_cache.append(label_tensor)
                        self.names_cache.append(base_name)

                except Exception as e:
                    print(f"Warning: could not read {npy_path}: {e}")

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
            npy_path = os.path.join(self.embedding_dir, base_name + ".npy")

            try:
                load_mode = 'r' if self.only_global else None
                data = np.load(npy_path, mmap_mode=load_mode)

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

    padded_features = torch.zeros((batch_size, max_len, input_dim), dtype=torch.float)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

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

    train_dataset = CTScanMILDataset(train_df, data_cfg['train_embedding_dir'], label_cols, only_global=only_global)
    val_dataset = CTScanMILDataset(val_df, data_cfg['train_embedding_dir'], label_cols, only_global=only_global)
    test_dataset = CTScanMILDataset(test_df, data_cfg['val_embedding_dir'], label_cols, only_global=only_global)

    return train_dataset, val_dataset, test_dataset


def _create_rad_datasets(config):
    data_cfg = config['data']
    only_global = data_cfg.get('only_global', False)
    label_cols = config['experiment']['rad_chestct_classes']

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
                                     only_global=only_global)
    val_dataset = CTScanMILDataset(val_df, emb_dir, label_cols, id_col='NoteAcc_DEID', file_ext='',
                                   only_global=only_global)
    test_dataset = CTScanMILDataset(test_df, emb_dir, label_cols, id_col='NoteAcc_DEID', file_ext='',
                                    only_global=only_global)

    return train_dataset, val_dataset, test_dataset


def create_datasets(config):
    """Router function to build datasets based on the yaml config."""
    dataset_type = config['data'].get('dataset_type', 'ct_rate')

    if dataset_type == 'rad_chestct':
        return _create_rad_datasets(config)
    else:
        return _create_ctrate_datasets(config)