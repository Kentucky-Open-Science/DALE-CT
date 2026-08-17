"""
Balanced Subset Selection for CT-RATE-huggingface-downloads Embedding Generation.

Selects a subset of patients that maximizes label balance across 18 disease
findings. Only the first scan per patient is considered (lowest recon number).

Strategy:
  1. Read the label CSV and extract patient-level labels (first scan only).
  2. Compute per-label prevalence.
  3. Greedily select patients to balance label distribution:
     - Prioritize patients with rare positive labels.
     - Maintain diversity by tracking how many times each label has been
       selected and preferring patients that fill underrepresented labels.

Usage:
    from utils.balanced_subset import select_balanced_patients
    selected = select_balanced_patients(
        csv_path='/path/to/train_predicted_labels.csv',
        n_patients=5000,
        seed=42
    )
"""

import pandas as pd
import numpy as np
import re
import random
from collections import Counter


def _extract_patient_id(volume_name):
    """
    Extract patient ID from volume name.
    Handles formats:
      - train_123_a_1.nii.gz  -> patient=123, scan=a, recon=1
      - valid_456_b_2.nii.gz  -> patient=456, scan=b, recon=2
    Returns (patient_id, scan_id, recon_id) or None if no match.
    """
    # Pattern: {prefix}_{patient_id}_{scan}_{recon}.nii.gz
    match = re.match(r'^(train|valid)_(\d+)_([a-z]+)_(\d+)\.nii\.gz$', volume_name)
    if match:
        prefix = match.group(1)
        patient_id = int(match.group(2))
        scan_id = match.group(3)
        recon_id = int(match.group(4))
        return patient_id, scan_id, recon_id
    return None


def _get_first_scan_per_patient(df):
    """
    Filter the label DataFrame to keep only the first scan (lowest recon_id)
    per patient. If multiple scans exist for a patient, keeps the one with
    the lowest recon number.

    Args:
        df: DataFrame with 'VolumeName' column and label columns.

    Returns:
        DataFrame with one row per patient (first scan only).
    """
    records = []
    for _, row in df.iterrows():
        info = _extract_patient_id(row['VolumeName'])
        if info is None:
            continue
        patient_id, scan_id, recon_id = info
        records.append({
            'VolumeName': row['VolumeName'],
            'patient_id': patient_id,
            'scan_id': scan_id,
            'recon_id': recon_id,
            **{col: row[col] for col in df.columns if col != 'VolumeName'}
        })

    if not records:
        raise ValueError("No valid volume names found in CSV. "
                         "Expected format: {prefix}_{patient_id}_{scan}_{recon}.nii.gz")

    patient_df = pd.DataFrame(records)

    # For each patient, keep the row with the lowest recon_id
    # If multiple scans exist, keep the first alphabetically by scan_id then lowest recon
    patient_df = patient_df.sort_values(['patient_id', 'scan_id', 'recon_id'])
    patient_df = patient_df.drop_duplicates(subset=['patient_id'], keep='first')

    return patient_df


def select_balanced_patients(csv_path, n_patients, seed=42, label_columns=None):
    """
    Select a balanced subset of patients from a label CSV.

    Algorithm:
      1. Read CSV, extract patient-level labels (first scan only).
      2. Compute per-label positive counts.
      3. Score each patient by how much they help balance the distribution:
         - Rare positive labels get higher weight.
         - Patients with multiple rare positives score higher.
      4. Greedily select top-scoring patients, updating scores after each
         selection to avoid over-representing any label.

    Args:
        csv_path: Path to the label CSV file.
        n_patients: Number of patients to select.
        seed: Random seed for reproducibility.
        label_columns: List of label column names. If None, auto-detected
                       as all columns except 'VolumeName'.

    Returns:
        List of selected VolumeName strings.
    """
    random.seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(csv_path)

    if label_columns is None:
        label_columns = [c for c in df.columns if c != 'VolumeName']

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Label columns ({len(label_columns)}): {label_columns}")

    # Get first scan per patient
    patient_df = _get_first_scan_per_patient(df)
    print(f"After first-scan filtering: {len(patient_df)} unique patients")

    if len(patient_df) <= n_patients:
        print(f"WARNING: Requested {n_patients} patients but only "
              f"{len(patient_df)} available. Returning all.")
        return patient_df['VolumeName'].tolist()

    # Extract label matrix
    labels = patient_df[label_columns].values.astype(np.float32)  # (N_patients, N_labels)
    volume_names = patient_df['VolumeName'].astype(str).values

    # Compute global prevalence (how rare each label is)
    n_total = len(patient_df)
    prevalence = labels.sum(axis=0) / n_total  # (N_labels,)

    # Weight: rarer labels get higher weight (inverse frequency)
    # Add small epsilon to avoid division by zero
    label_weights = 1.0 / (prevalence + 0.01)  # (N_labels,)

    # Normalize weights to sum to N_labels
    label_weights = label_weights / label_weights.sum() * len(label_columns)

    print(f"Label prevalence range: {prevalence.min():.3f} - {prevalence.max():.3f}")
    print(f"Label weights range: {label_weights.min():.2f} - {label_weights.max():.2f}")

    # Track cumulative selected label counts
    selected_mask = np.zeros(n_total, dtype=bool)
    selected_label_counts = np.zeros(len(label_columns), dtype=np.float32)

    selected_volume_names = []

    # Greedy selection
    for step in range(n_patients):
        # Compute score for each unselected patient:
        # score = sum over labels of (label_value * label_weight * (1 - current_fraction))
        # This rewards patients that have positive labels that are currently
        # underrepresented in the selection.

        # Current fraction of each label in the selection
        if step > 0:
            current_fraction = selected_label_counts / step
        else:
            current_fraction = np.zeros(len(label_columns), dtype=np.float32)

        # Target fraction is the global prevalence (balanced representation)
        # We want to select patients whose labels are below target
        deficit = np.maximum(0, prevalence - current_fraction)  # (N_labels,)

        # Score each unselected patient
        available_indices = np.where(~selected_mask)[0]
        available_labels = labels[available_indices]  # (N_available, N_labels)

        # Weighted score: how much does this patient help fill deficits?
        scores = (available_labels * label_weights[np.newaxis, :] * deficit[np.newaxis, :]).sum(axis=1)

        # Add small random noise to break ties
        scores += np.random.uniform(0, 1e-6, size=len(scores))

        # Select the patient with the highest score
        best_local_idx = np.argmax(scores)
        best_global_idx = available_indices[best_local_idx]

        selected_mask[best_global_idx] = True
        selected_label_counts += labels[best_global_idx]
        selected_volume_names.append(volume_names[best_global_idx])

        if (step + 1) % 1000 == 0:
            selected_fraction = selected_label_counts / (step + 1)
            max_deviation = np.abs(selected_fraction - prevalence).max()
            print(f"  Step {step + 1}/{n_patients}: max label deviation = {max_deviation:.4f}")

    # Final statistics
    final_fraction = selected_label_counts / n_patients
    print(f"\nFinal label distribution:")
    for i, col in enumerate(label_columns):
        print(f"  {col:40s}: target={prevalence[i]:.4f}, selected={final_fraction[i]:.4f}, "
              f"diff={final_fraction[i] - prevalence[i]:+.4f}")

    print(f"\nSelected {len(selected_volume_names)} patients from {n_total} total.")

    return selected_volume_names


def get_patient_id_from_volume_name(volume_name):
    """
    Extract just the patient ID number from a volume name.
    Used for filtering WebDataset streams.

    Examples:
        'train_123_a_1.nii.gz' -> 123
        'valid_456_b_2.nii.gz' -> 456
    """
    info = _extract_patient_id(volume_name)
    if info:
        return info[0]  # patient_id
    return None


def build_patient_id_set(volume_names):
    """
    Convert a list of VolumeName strings to a set of full volume names.
    Used for fast O(1) lookup during WebDataset streaming.

    Returns a set of exact VolumeName strings (e.g., 'train_123_a_1.nii.gz')
    so that filtering matches the specific scan+recon, not just the patient.
    This prevents multiple scans/reconstructions for the same patient from
    leaking through when only the first scan was selected.
    """
    return set(volume_names)


def filter_volume_by_patient_set(volume_name, allowed_volume_names):
    """
    Check if a volume is in the allowed set by exact volume name match.

    This matches the full volume identity (patient + scan + recon), not just
    the patient ID. This prevents multiple scans/reconstructions for the same
    patient from leaking through when only the first scan was selected.

    Args:
        volume_name: e.g. 'train_123_a_1.nii.gz' or 'train_123_a_1.npy'
        allowed_volume_names: set of full VolumeName strings
                              (e.g., 'train_123_a_1.nii.gz')

    Returns:
        True if the volume is in the allowed set.
    """
    # Normalize to .nii.gz format for matching (CSV uses .nii.gz)
    clean_name = volume_name.replace('.npy', '.nii.gz')
    return clean_name in allowed_volume_names
