import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

class CTScanLinearDataset(Dataset):
    def __init__(self, df, base_embedding_dir, pool_strategy="cls_max", label_cols=None):
        # We append the specific strategy to the base directory path
        self.embedding_dir = os.path.join(base_embedding_dir, pool_strategy)
        self.pool_strategy = pool_strategy

        if label_cols is None:
            self.label_cols = [
                'Medical material', 'Arterial wall calcification', 'Cardiomegaly',
                'Pericardial effusion', 'Coronary artery wall calcification',
                'Hiatal hernia', 'Lymphadenopathy', 'Emphysema', 'Atelectasis',
                'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela',
                'Pleural effusion', 'Mosaic attenuation pattern',
                'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
                'Interlobular septal thickening'
            ]
        else:
            self.label_cols = label_cols

        valid_indices = []
        self.data_cache = []
        self.labels_cache = []
        self.names_cache = []

        print(f"Loading precomputed {pool_strategy} features from {self.embedding_dir}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Caching {self.pool_strategy}"):
            fname = row['VolumeName']
            base_name = os.path.splitext(fname.replace('.nii.gz', ''))[0]
            npy_path = os.path.join(self.embedding_dir, base_name + ".npy")

            if os.path.exists(npy_path):
                # Load the tiny precomputed 1D array
                data = np.load(npy_path)

                features = torch.from_numpy(data).float()
                labels_np = row[self.label_cols].values.astype(np.float32)
                label_tensor = torch.tensor(labels_np, dtype=torch.float32)

                self.data_cache.append(features)
                self.labels_cache.append(label_tensor)
                self.names_cache.append(base_name)
                valid_indices.append(idx)

        print(f"✅ Cached {len(valid_indices)} / {len(df)} volumes into RAM.")
        self.df = df.loc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Instant retrieval from RAM
        return self.data_cache[idx], self.labels_cache[idx], self.names_cache[idx]


def create_linear_datasets(config, pool_strategy):
    data_cfg = config['data']

    # The old "val" paths are now your "test" paths
    train_csv = data_cfg['train_label_path']
    test_csv = data_cfg['val_label_path']

    train_emb_dir = data_cfg['train_embedding_dir']
    test_emb_dir = data_cfg['val_embedding_dir']

    full_train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # 1. Extract Patient ID
    # Assuming VolumeName is like 'train_1_a_1.nii.gz', splitting by '_' and taking index 1 gets '1'
    def extract_patient_id(filename):
        parts = filename.split('_')
        if len(parts) > 1:
            return parts[1]
        return filename  # Fallback just in case

    full_train_df['patient_id'] = full_train_df['VolumeName'].apply(extract_patient_id)

    # 2. Patient-Level Split (7.5% for validation)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.075, random_state=42)
    train_idx, val_idx = next(gss.split(full_train_df, groups=full_train_df['patient_id']))

    train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_train_df.iloc[val_idx].reset_index(drop=True)

    print(f"📊 Dataset Split: {len(train_df)} Train scans | {len(val_df)} Val scans | {len(test_df)} Test scans")

    # 3. Create the 3 datasets
    train_dataset = CTScanLinearDataset(train_df, train_emb_dir, pool_strategy=pool_strategy)
    # The new validation set draws from the original train embedding directory
    val_dataset = CTScanLinearDataset(val_df, train_emb_dir, pool_strategy=pool_strategy)
    # The test set draws from the old validation directory
    test_dataset = CTScanLinearDataset(test_df, test_emb_dir, pool_strategy=pool_strategy)

    return train_dataset, val_dataset, test_dataset