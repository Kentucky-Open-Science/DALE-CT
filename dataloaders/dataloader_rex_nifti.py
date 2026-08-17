"""
PyTorch Dataset and DataLoader for ReX-GroundingCT .nii.gz volumes and masks.

Loads 3D CT volumes and their corresponding ReX segmentation masks, applies
preprocessing (HU clipping, 0-1 mapping, z-score normalization, resizing),
and maps the multi-finding mask channels to the 14 ReX hierarchical
subcategories using the MLHC_dataset_version.json metadata file.

Normalization parameters are read from the config to match each model's
training distribution (LeJEPA-0, LeJEPA-1S, LeJEPA-2S, DINOv2-CT).
"""

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The 14 ReX hierarchical subcategories in canonical order
REX_CLASSES = [
    "1a", "1b", "1c", "1d", "1e", "1f",
    "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h",
]


# ---------------------------------------------------------------------------
# Helper: load HU conversion parameters from metadata CSV
# ---------------------------------------------------------------------------

def _load_hu_conversion_map(metadata_csv_path: str) -> Dict[str, Tuple[float, float]]:
    """
    Parse a CT-RATE-huggingface-downloads metadata CSV and return a mapping:
        VolumeName -> (RescaleSlope, RescaleIntercept)

    The CSV has columns: VolumeName, ..., RescaleSlope, RescaleIntercept, ...

    HU = raw_pixel_value * slope + intercept

    Parameters
    ----------
    metadata_csv_path : str
        Path to train_metadata.csv or valid_metadata.csv.

    Returns
    -------
    hu_map : dict
        Maps volume filename (e.g. 'train_1_a_1.nii.gz') to (slope, intercept).
    """
    hu_map: Dict[str, Tuple[float, float]] = {}
    with open(metadata_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("VolumeName", "").strip()
            if not name:
                continue
            try:
                slope = float(row.get("RescaleSlope", "1"))
                intercept = float(row.get("RescaleIntercept", "0"))
            except (ValueError, TypeError):
                slope = 1.0
                intercept = 0.0
            hu_map[name] = (slope, intercept)

    print(
        f"[HU Conversion] Loaded {len(hu_map)} entries from "
        f"{os.path.basename(metadata_csv_path)}"
    )
    return hu_map


# ---------------------------------------------------------------------------
# Helper: build a class index from the JSON metadata
# ---------------------------------------------------------------------------

def _build_finding_to_class_map(rex_json_path: str) -> Dict[str, int]:
    """
    Parse MLHC_dataset_version.json (or dataset.json) and return a mapping:
        category_string -> class_index (0..13)

    The JSON has structure:
        {
            "train": [
                {
                    "name": "train_2058_a_1.nii.gz",
                    "findings": {
                        "0": "Mild paraseptal emphysematous changes..."
                    },
                    "categories": {
                        "0": "1c"
                    },
                    ...
                },
                ...
            ]
        }

    Note: The JSON may only have a single "train" key with both train_ and
    valid_ entries mixed together.  We scan ALL top-level keys.

    The ``categories`` dict maps finding index (string) to one of the 14
    ReX hierarchical subcategory codes (e.g., "1c", "2b").  We scan all
    entries to build a global mapping, which is consistent across the
    dataset.
    """
    with open(rex_json_path, "r") as f:
        metadata = json.load(f)

    # Collect all unique category strings from ALL top-level keys
    cat_set: set = set()
    for split_key, entries in metadata.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            categories = entry.get("categories", {})
            for cat in categories.values():
                if cat is not None:
                    cat_set.add(cat)

    # Build mapping: category string -> class index
    cat_to_idx = {sc: i for i, sc in enumerate(REX_CLASSES)}

    # Validate that all categories in the JSON are known
    for cat in cat_set:
        if cat not in cat_to_idx:
            raise ValueError(
                f"Unknown category '{cat}' found in {rex_json_path}. "
                f"Expected one of {REX_CLASSES}."
            )

    return cat_to_idx


def _get_finding_to_class_for_scan(
    entry: dict, cat_to_idx: Dict[str, int]
) -> Dict[int, int]:
    """
    For a single scan entry from the JSON, return a mapping:
        finding_index (int) -> class_index (0..13)

    Uses the top-level ``categories`` dict from the JSON entry.
    """
    mapping: Dict[int, int] = {}
    categories = entry.get("categories", {})
    for finding_idx_str, cat in categories.items():
        if cat is not None and cat in cat_to_idx:
            mapping[int(finding_idx_str)] = cat_to_idx[cat]
    return mapping


# ---------------------------------------------------------------------------
# CT Preprocessing (matches HuggingFace model cards exactly)
# ---------------------------------------------------------------------------

class CTPreprocessor:
    """
    Applies the exact HU windowing and Z-score normalization used during
    LeJEPA / DINOv2-CT training.

    Step 1: clip → [clip_min, clip_max], then map to [0, 1]
    Step 2: z-score = (x - norm_mean) / norm_std

    norm_mean and norm_std are derived from the raw HU statistics:
        range_val = clip_max - clip_min
        norm_mean = (mean_hu - clip_min) / range_val
        norm_std  = std_hu / range_val
    """

    def __init__(
        self,
        clip_min: float = -997.0,
        clip_max: float = 888.0,
        mean_hu: float = -142.39,
        std_hu: float = 360.97,
        patch_size: int = 14,
    ):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mean_hu = mean_hu
        self.std_hu = std_hu
        self.patch_size = patch_size

        # Pre-compute 0-1 scaled mean and std
        range_val = self.clip_max - self.clip_min
        self.norm_mean = (self.mean_hu - self.clip_min) / range_val
        self.norm_std = self.std_hu / range_val

    def __call__(self, image_2d: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to a single 2D slice.

        Parameters
        ----------
        image_2d : np.ndarray  shape [H, W], float32, raw Hounsfield Units

        Returns
        -------
        processed : np.ndarray  shape [H', W'], float32, normalized
            H' and W' are adjusted to be multiples of patch_size.
        """
        # Convert to tensor for processing
        vol = torch.from_numpy(image_2d).float()
        if vol.ndim == 2:
            vol = vol.unsqueeze(0)  # [1, H, W]

        # Step 1: Clamp HU values and map to [0, 1]
        vol = torch.clamp(vol, self.clip_min, self.clip_max)
        range_val = self.clip_max - self.clip_min
        vol = (vol - self.clip_min) / range_val

        # Step 2: Z-score standardization
        vol = (vol - self.norm_mean) / self.norm_std

        # Step 3: Pad/crop to patch-aligned dimensions (nearest interpolation)
        C, H, W = vol.shape
        target_h = int((H // self.patch_size) * self.patch_size)
        target_w = int((W // self.patch_size) * self.patch_size)

        if target_h != H or target_w != W:
            vol = vol.unsqueeze(0)  # [1, 1, H, W]
            vol = F.interpolate(
                vol, size=(target_h, target_w), mode='nearest'
            )
            vol = vol.squeeze(0)  # [1, H', W']

        return vol.squeeze(0).numpy()  # [H', W']


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RexNiftiDataset(Dataset):
    """
    PyTorch Dataset for ReX-GroundingCT NIfTI volumes.

    Each item is a single patient volume with its corresponding segmentation
    mask and classification labels.

    Returns
    -------
    volume_tensor : torch.Tensor  shape [D, 1, H, W]
    mask_tensor : torch.Tensor    shape [D, 14, H, W]
    cls_labels : torch.Tensor     shape [D, 14]
    patient_id : str
    """

    def __init__(
        self,
        rex_json_path: str,
        masks_dir: str,
        ct_volumes_dir: str,
        image_size: int = 256,
        split: str = "train",
        preprocessor: Optional[CTPreprocessor] = None,
        metadata_csv_path: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        rex_json_path : str
            Path to MLHC_dataset_version.json.
        masks_dir : str
            Directory containing .nii.gz segmentation masks.
        ct_volumes_dir : str
            Root directory for CT-RATE-huggingface-downloads volumes (contains train/ and valid/).
        image_size : int
            Target size for axial (H, W) dimensions after resizing.
        split : str
            Which split to load: 'train' or 'valid'.
        preprocessor : CTPreprocessor or None
            If None, a default CTPreprocessor is used.
        metadata_csv_path : str or None
            Path to train_metadata.csv or valid_metadata.csv containing
            RescaleSlope and RescaleIntercept for HU conversion.
        """
        super().__init__()
        self.masks_dir = masks_dir
        self.ct_volumes_dir = ct_volumes_dir
        self.image_size = image_size
        self.split = split
        self.preprocessor = preprocessor or CTPreprocessor()

        # Load HU conversion map from metadata CSV
        self.hu_map: Dict[str, Tuple[float, float]] = {}
        if metadata_csv_path is not None and os.path.exists(metadata_csv_path):
            self.hu_map = _load_hu_conversion_map(metadata_csv_path)
        elif metadata_csv_path is not None:
            print(
                f"[RexNiftiDataset] WARNING: metadata CSV not found at "
                f"{metadata_csv_path} — skipping HU conversion"
            )

        # Load metadata and build class mapping
        with open(rex_json_path, "r") as f:
            self.metadata = json.load(f)

        self.cat_to_idx = _build_finding_to_class_map(rex_json_path)

        # Filter entries for the requested split.
        # The JSON may have a single "train" key with both train_ and valid_
        # entries mixed together, identified by filename prefix.
        available_keys = list(self.metadata.keys())
        print(f"[RexNiftiDataset] Available split keys in JSON: {available_keys}")

        # Collect all entries from all keys
        all_entries = []
        for key in available_keys:
            all_entries.extend(self.metadata[key])

        # Filter by filename prefix: "train_" or "valid_"
        prefix = f"{split}_"
        self.entries = [
            e for e in all_entries
            if e.get("name", "").startswith(prefix)
        ]

        if len(self.entries) == 0:
            raise ValueError(
                f"No entries found for split '{split}' (prefix '{prefix}') "
                f"in {rex_json_path}. Available keys: {available_keys}. "
                f"Total entries scanned: {len(all_entries)}"
            )

        # Pre-compute finding->class mapping per scan
        self.scan_mappings: List[Dict[int, int]] = []
        for entry in self.entries:
            self.scan_mappings.append(
                _get_finding_to_class_for_scan(entry, self.cat_to_idx)
            )

        print(
            f"[RexNiftiDataset] Loaded {len(self.entries)} volumes "
            f"for split='{split}'"
        )

    def __len__(self) -> int:
        return len(self.entries)

    def _resolve_volume_path(self, filename: str) -> str:
        """
        Convert a mask filename like 'train_1_a_1.nii.gz' or
        'valid_1_a_1.nii.gz' into the corresponding CT volume path.

        Pattern:
            train_X_Y_Z.nii.gz → .../train/train_X/train_X_Y/train_X_Y_Z.nii.gz
            valid_X_Y_Z.nii.gz → .../valid/valid_X/valid_X_Y/valid_X_Y_Z.nii.gz
        """
        base = filename.replace(".nii.gz", "")
        parts = base.split("_")
        split_prefix = parts[0]
        patient_id = parts[1]
        study_id = "_".join(parts[1:3])

        volume_path = os.path.join(
            self.ct_volumes_dir,
            split_prefix,
            f"{split_prefix}_{patient_id}",
            f"{split_prefix}_{study_id}",
            filename,
        )
        return volume_path

    def _load_and_preprocess_volume(
        self, volume_path: str, filename: str
    ) -> np.ndarray:
        """
        Load a CT volume, convert to Hounsfield Units (if metadata CSV
        provided), then apply HU clipping, 0-1 mapping, and z-score
        normalization via the CTPreprocessor.

        HU = raw_pixel_value * RescaleSlope + RescaleIntercept

        Returns
        -------
        volume : np.ndarray  shape [D, H', W'], float32
            H' and W' are patch-aligned.
        """
        img = nib.load(volume_path)
        volume = img.get_fdata(dtype=np.float32)  # [H, W, D]

        # --- Convert to Hounsfield Units ---
        if filename in self.hu_map:
            slope, intercept = self.hu_map[filename]
            volume = volume * slope + intercept

        # Transpose to [D, H, W] for slice-wise processing
        volume = np.transpose(volume, (2, 0, 1))  # [D, H, W]

        D = volume.shape[0]
        # Process first slice to determine output dimensions
        first_slice = self.preprocessor(volume[0])
        H_out, W_out = first_slice.shape

        processed = np.zeros((D, H_out, W_out), dtype=np.float32)
        processed[0] = first_slice
        for d in range(1, D):
            processed[d] = self.preprocessor(volume[d])

        return processed

    def _load_and_aggregate_mask(
        self, mask_filename: str, finding_to_class: Dict[int, int]
    ) -> np.ndarray:
        """
        Load a ReX segmentation mask and aggregate findings into the 14
        target classes.

        The mask on disk has shape [F, H, W, D] where F is the number of
        independent findings.  We transpose to [D, F, H, W] for slice-wise
        access, then aggregate F into 14 channels via logical OR.

        Returns
        -------
        mask_14 : np.ndarray  shape [D, 14, H, W], bool
        """
        mask_path = os.path.join(self.masks_dir, mask_filename)
        img = nib.load(mask_path)
        mask_data = img.get_fdata(dtype=np.float32)

        if mask_data.ndim != 4:
            raise ValueError(
                f"Expected 4D mask [F, H, W, D], got shape {mask_data.shape} "
                f"for {mask_filename}"
            )

        F, H, W, D = mask_data.shape

        # Transpose to [D, F, H, W]
        mask_data = np.transpose(mask_data, (3, 0, 1, 2))

        # Aggregate into 14 channels
        mask_14 = np.zeros((D, 14, H, W), dtype=np.bool_)
        for f_idx in range(F):
            class_idx = finding_to_class.get(f_idx)
            if class_idx is not None:
                mask_14[:, class_idx, :, :] = np.logical_or(
                    mask_14[:, class_idx, :, :],
                    mask_data[:, f_idx, :, :] > 0.5,
                )

        return mask_14

    def _resize_slice(
        self, image_2d: np.ndarray, mask_2d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize a single axial slice to (image_size, image_size).

        Image uses bilinear interpolation; mask uses nearest-neighbor.
        Input shapes: [H, W] for image, [14, H, W] for mask.
        """
        import cv2

        h, w = image_2d.shape
        if h == self.image_size and w == self.image_size:
            return image_2d, mask_2d

        image_resized = cv2.resize(
            image_2d,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        mask_resized = np.zeros(
            (14, self.image_size, self.image_size), dtype=np.bool_
        )
        for c in range(14):
            mask_resized[c] = cv2.resize(
                mask_2d[c].astype(np.uint8),
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.bool_)

        return image_resized, mask_resized

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        entry = self.entries[idx]
        filename = entry["name"]
        finding_to_class = self.scan_mappings[idx]

        # --- Load and preprocess volume ---
        volume_path = self._resolve_volume_path(filename)
        if not os.path.exists(volume_path):
            raise FileNotFoundError(f"Volume not found: {volume_path}")

        volume = self._load_and_preprocess_volume(
            volume_path, filename
        )  # [D, H', W']

        # --- Load and aggregate mask ---
        mask_14 = self._load_and_aggregate_mask(
            filename, finding_to_class
        )  # [D, 14, H_orig, W_orig]

        D = volume.shape[0]

        # --- Resize each slice to target image_size ---
        volume_resized = np.zeros(
            (D, self.image_size, self.image_size), dtype=np.float32
        )
        mask_resized = np.zeros(
            (D, 14, self.image_size, self.image_size), dtype=np.float32
        )

        for d in range(D):
            img_slice, msk_slice = self._resize_slice(volume[d], mask_14[d])
            volume_resized[d] = img_slice
            mask_resized[d] = msk_slice.astype(np.float32)

        # --- Build classification labels ---
        cls_labels = (mask_resized.sum(axis=(2, 3)) > 0).astype(np.float32)

        # --- Convert to tensors ---
        volume_tensor = torch.from_numpy(volume_resized).unsqueeze(1)
        mask_tensor = torch.from_numpy(mask_resized)
        cls_labels_tensor = torch.from_numpy(cls_labels)

        patient_id = filename.replace(".nii.gz", "")

        return volume_tensor, mask_tensor, cls_labels_tensor, patient_id


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def rex_collate_fn(batch):
    """
    Custom collate that returns lists of volumes (variable D) rather than
    stacking them, since each patient has a different number of slices.
    """
    volumes, masks, labels, ids = zip(*batch)
    return list(volumes), list(masks), list(labels), list(ids)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_rex_dataloaders(
    rex_json_path: str,
    masks_dir: str,
    ct_volumes_dir: str,
    image_size: int = 256,
    batch_size: int = 1,
    num_workers: int = 4,
    preprocessor: Optional[CTPreprocessor] = None,
    train_metadata_csv: Optional[str] = None,
    valid_metadata_csv: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders for the ReX dataset.

    Parameters
    ----------
    train_metadata_csv : str or None
        Path to train_metadata.csv with RescaleSlope/RescaleIntercept.
    valid_metadata_csv : str or None
        Path to valid_metadata.csv with RescaleSlope/RescaleIntercept.

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    """
    train_dataset = RexNiftiDataset(
        rex_json_path=rex_json_path,
        masks_dir=masks_dir,
        ct_volumes_dir=ct_volumes_dir,
        image_size=image_size,
        split="train",
        preprocessor=preprocessor,
        metadata_csv_path=train_metadata_csv,
    )

    val_dataset = RexNiftiDataset(
        rex_json_path=rex_json_path,
        masks_dir=masks_dir,
        ct_volumes_dir=ct_volumes_dir,
        image_size=image_size,
        split="valid",
        preprocessor=preprocessor,
        metadata_csv_path=valid_metadata_csv,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=rex_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=rex_collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader
