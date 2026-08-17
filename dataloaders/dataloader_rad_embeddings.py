import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


class RADChestCTDataset(Dataset):
    def __init__(self, df, embedding_dir, label_cols=None, only_global=False, interpolate_slices=False,
                 target_slices=250):
        self.embedding_dir = embedding_dir
        self.only_global = only_global
        self.interpolate_slices = interpolate_slices
        self.target_slices = target_slices

        # Use provided label_cols or fallback to everything after the first column
        self.label_cols = label_cols if label_cols is not None else df.columns[1:].tolist()

        # Filter out any rows that don't have a matching .npy file
        valid_rows = []
        for _, row in df.iterrows():
            patient_id = row['NoteAcc_DEID']
            file_path = os.path.join(self.embedding_dir, f"{patient_id}.npy")

            if os.path.exists(file_path):
                valid_rows.append(row)

        # Overwrite with only the strictly matched data
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"✅ Securely matched {len(self.df)} volumes by exact Patient ID.")

        if self.interpolate_slices:
            print(f"📏 Forcing all sequences to length {self.target_slices} via 1D interpolation.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row['NoteAcc_DEID']

        # Load the specific feature file tied to that ID
        file_path = os.path.join(self.embedding_dir, f"{patient_id}.npy")
        features = np.load(file_path)
        features = torch.tensor(features, dtype=torch.float32)

        # Apply interpolation if configured
        if self.interpolate_slices and self.target_slices is not None:
            # Current shape: (Seq_Len, Input_Dim)
            # F.interpolate needs: (Batch, Channels, Length) -> (1, Input_Dim, Seq_Len)
            features = features.t().unsqueeze(0)

            features = F.interpolate(
                features,
                size=self.target_slices,
                mode='linear',
                align_corners=False
            )

            # Reshape back to (Target_Slices, Input_Dim)
            features = features.squeeze(0).t()

        # Apply global pooling flag if your pipeline uses it
        if self.only_global:
            # Assuming global features implies a single vector, adjust if your logic differs
            if features.dim() > 1:
                features = features.mean(dim=0, keepdim=True)

        # Extract the labels for this specific patient
        labels = row[self.label_cols].values.astype(np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        # Returning 4 items to safely unpack as: features, labels, _, mask
        return features, labels, patient_id


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

    # Create a boolean mask (True for valid instances, False for padding)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, feat in enumerate(features_list):
        end = lengths[i]
        padded_features[i, :end, :] = feat
        mask[i, :end] = True  # Mark valid data

    return padded_features, labels, names_list, mask


def create_datasets(config):
    """
    Creates just the Unseen Test dataset for RAD-ChestCT evaluation.
    Returns (None, None, test_dataset) to match the unpacking shape.
    """
    data_cfg = config['data']
    exp_cfg = config['experiment']

    only_global = data_cfg.get('only_global', False)
    interpolate_slices = data_cfg.get('interpolate_slices', False)
    target_slices = data_cfg.get('target_slices', 250)

    test_csv = data_cfg['rad_test_label_path']
    test_emb_dir = data_cfg['rad_test_embedding_dir']
    label_cols = exp_cfg['rad_chestct_classes']

    print(f"\nReading RAD-ChestCT labels from {test_csv}...")
    test_df = pd.read_csv(test_csv)
    print(f"Total rows in CSV: {len(test_df)}")
    print(f"---------------------------\n")

    # Create the Dataset object with the new interpolation parameters
    test_dataset = RADChestCTDataset(
        test_df,
        test_emb_dir,
        label_cols=label_cols,
        only_global=only_global,
        interpolate_slices=interpolate_slices,
        target_slices=target_slices
    )

    # Return None for train and val to match `_, _, rad_test_dataset = create_datasets(config)`
    return None, None, test_dataset