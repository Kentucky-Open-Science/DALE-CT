import os
import torch
import pandas as pd
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset
import torch.nn.functional as F
import warnings
import webdataset as wds
import glob
import matplotlib.pyplot as plt
import torchvision

class MultiScaleSliceProcessor:
    """
    Handles three modes based on config.inference.mode:
    1. 'tiled': Global View (resized to 224) + Local View (patches).
    2. 'full_resolution': Returns the slice at its ORIGINAL resolution.
    3. 'cropped_global': Crops a square containing the body, then resizes to 224.
    """

    def __init__(self, config, output_dir=None):
        # Read the normalization mode, defaulting to the original z_score method if not found
        self.norm_mode = getattr(config.dataset, 'normalization_mode', 'z_score').lower()

        self.clip_min = getattr(config.dataset, 'clip_min', -997.0)
        self.clip_max = getattr(config.dataset, 'clip_max', 888.0)
        self.mean_hu = getattr(config.dataset, 'mean_hu', -142.39)
        self.std_hu = getattr(config.dataset, 'std_hu', 360.97)

        range_val = self.clip_max - self.clip_min if (self.clip_max - self.clip_min) > 0 else 1.0
        self.norm_mean = (self.mean_hu - self.clip_min) / range_val
        self.norm_std = self.std_hu / range_val

        self.num_channels = getattr(config.dataset, 'num_channels', 1)
        self.mode = getattr(config.inference, 'mode', 'tiled')

        self.global_crop_size = getattr(config.crops, 'global_crops_size', 224)
        self.stride = int(self.global_crop_size * 0.85)

        # Optional cap on slice count (z-axis). When set, volumes longer than this
        # are uniformly subsampled along z BEFORE the body-crop/roi_align path. This
        # bounds per-batch compute (and mmap I/O) so DDP ranks don't straggle on
        # long thin-slice volumes — the cause of the gradient-sync hangs. None = off.
        self.max_slices = getattr(config.dataset, 'max_slices', None)

        # Visualization setup
        self.vis_count = 0
        self.max_vis = 10
        self.vis_dir = os.path.join(output_dir, "crop_visualizations") if output_dir else None
        if self.vis_dir and not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir, exist_ok=True)

    def normalize(self, volume):
        """Routes the volume to the selected normalization math."""
        # todo: test
        if self.norm_mode == 'physical':
            # Absolute physical mapping (Baseline methodology)
            # 1. Divide HU by 1000
            volume = volume / 1000.0
            # 2. Clip strictly to [-1.0, 1.0]
            volume = torch.clamp(volume, -1.0, 1.0)
            return volume

        elif self.norm_mode == 'dinomx':
            # v628/DINOMX native pipeline (dinomx/extract_embeddings.py):
            #   1. Clip to corpus HU range (e.g. [-1024, 3071])
            #   2. Z-score with pinned corpus stats directly on HU values
            #      (mean/std are raw HU, not 0-1 scaled — no intermediate mapping)
            volume = torch.clamp(volume, self.clip_min, self.clip_max)
            volume = (volume - self.mean_hu) / self.std_hu
            return volume

        else:
            # Original TAP-CT/LeJEPA dataset-dependent mapping
            volume = torch.clamp(volume, self.clip_min, self.clip_max)
            range_val = self.clip_max - self.clip_min
            volume = (volume - self.clip_min) / (range_val if range_val > 0 else 1.0)
            volume = (volume - self.norm_mean) / self.norm_std
            return volume

    def process_batch(self, slices, filename="unknown"):
        """
        Args:
            slices: Tensor of shape (B, H, W) or (B, 1, H, W) in RAW HU values ON THE GPU.
            filename: String used for saving visualization images.
        """

        if slices.ndim == 3:
            slices = slices.unsqueeze(1)  # (B, 1, H, W)

        B, C, H, W = slices.shape

        # Cap slice count before any per-slice work (body-crop, roi_align, norm) so
        # long thin-slice volumes don't dominate per-batch compute and starve DDP
        # peers. Resample the Z (slice) axis to max_slices via trilinear
        # interpolation -- NOT strided subsampling -- so neighboring slices blend
        # in raw-HU space (HU is linear in attenuation, matching scipy/TorchIO
        # Z-resampling) and no slice is discarded. (B,C,H,W)->(1,C,B,H,W) for
        # trilinear; only the D axis is resampled (same-size trilinear is identity
        # on H/W), so XY is untouched. Runs pre-norm on GPU (process_batch callers
        # move the volume to device first), so the blend is on true HU and is fast.
        if self.max_slices is not None and B > self.max_slices:
            x5 = slices.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # (1, C, B, H, W)
            x5 = F.interpolate(x5, size=(self.max_slices, H, W),
                               mode='trilinear', align_corners=False)
            slices = x5.permute(0, 2, 1, 3, 4).squeeze(0)               # (max_slices, C, H, W)
            B = self.max_slices

        if self.mode == 'cropped_global':
            # 1. GPU Batched Morphology
            solid_tissue = (slices > -500).float()
            kernel_size = 25
            pad = kernel_size // 2

            # Step A: Aggressive Erosion
            eroded = 1.0 - F.max_pool2d(1.0 - solid_tissue, kernel_size=kernel_size, stride=1, padding=pad)

            # Step B: Dilation
            body_mask = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=pad) > 0.5

            # --- STRICT 3D ALIGNMENT FIX ---
            # Collapse the batch (slice) dimension to find the absolute maximum body extent
            # across the entire volume. This guarantees a single, uniform bounding box.
            global_body_mask = body_mask.any(dim=0, keepdim=True)  # Shape: (1, 1, H, W)

            # 2. Find the single global bounding box
            rows = global_body_mask.any(dim=3).squeeze(1)  # Shape: (1, H)
            cols = global_body_mask.any(dim=2).squeeze(1)  # Shape: (1, W)
            valid = rows.any(dim=1)  # Shape: (1,)

            rmin = torch.argmax(rows.float(), dim=1)
            rmax = H - 1 - torch.argmax(torch.flip(rows, dims=[1]).float(), dim=1)
            cmin = torch.argmax(cols.float(), dim=1)
            cmax = W - 1 - torch.argmax(torch.flip(cols, dims=[1]).float(), dim=1)

            # Fallback for empty masks
            rmin = torch.where(valid, rmin, torch.zeros_like(rmin))
            rmax = torch.where(valid, rmax, torch.full_like(rmax, H - 1))
            cmin = torch.where(valid, cmin, torch.zeros_like(cmin))
            cmax = torch.where(valid, cmax, torch.full_like(cmax, W - 1))

            # 3. Calculate Square Coordinates for the SINGLE box
            bbox_h = rmax - rmin + 1
            bbox_w = cmax - cmin + 1
            side = torch.max(bbox_h, bbox_w)

            center_r = (rmin + rmax) // 2
            center_c = (cmin + cmax) // 2

            sq_rmin = center_r - side // 2
            sq_cmin = center_c - side // 2

            # Boundary clamping/shifting
            sq_rmin = torch.clamp(sq_rmin, min=0)
            sq_rmax = sq_rmin + side
            overflow_r = sq_rmax - H
            sq_rmin = torch.where(overflow_r > 0, sq_rmin - overflow_r, sq_rmin)
            sq_rmax = torch.where(overflow_r > 0, torch.full_like(sq_rmax, H), sq_rmax)
            sq_rmin = torch.clamp(sq_rmin, min=0)

            sq_cmin = torch.clamp(sq_cmin, min=0)
            sq_cmax = sq_cmin + side
            overflow_c = sq_cmax - W
            sq_cmin = torch.where(overflow_c > 0, sq_cmin - overflow_c, sq_cmin)
            sq_cmax = torch.where(overflow_c > 0, torch.full_like(sq_cmax, W), sq_cmax)
            sq_cmin = torch.clamp(sq_cmin, min=0)

            # 4. Batched Crop & Resize using RoI Align
            batch_idx = torch.arange(B, device=slices.device).unsqueeze(1).float()

            # Broadcast the SINGLE bounding box coordinates across all B slices
            boxes = torch.cat([
                batch_idx,
                sq_cmin.repeat(B, 1).float(),
                sq_rmin.repeat(B, 1).float(),
                sq_cmax.repeat(B, 1).float(),
                sq_rmax.repeat(B, 1).float()
            ], dim=1).to(dtype=slices.dtype)

            resized_pt = torchvision.ops.roi_align(
                slices,
                boxes,
                output_size=(self.global_crop_size, self.global_crop_size),
                spatial_scale=1.0,
                aligned=True
            )
            # --------------------------------

            # 5. Normalize
            norm_pt = self.normalize(resized_pt)

            # Visualization trigger
            if self.vis_count < self.max_vis and self.vis_dir:
                self._save_visualization(
                    slices[0, 0].cpu().numpy(),
                    global_body_mask[0, 0].cpu().numpy(),  # Visualize the global mask
                    sq_rmin[0].item(), sq_rmax[0].item() - 1,
                    sq_cmin[0].item(), sq_cmax[0].item() - 1,
                    norm_pt[0:1], filename, 0
                )
                self.vis_count += 1

            return norm_pt, None
        else:
            # Normalize the entire batch upfront for tiled/full_res
            norm_slices = self.normalize(slices)

            if self.mode == 'full_resolution':
                if norm_slices.shape[1] == 1 and self.num_channels == 3:
                    norm_slices = norm_slices.repeat(1, self.num_channels, 1, 1)
                return norm_slices, None

            elif self.mode == 'tiled':
                global_crops = F.interpolate(
                    norm_slices, size=(self.global_crop_size, self.global_crop_size),
                    mode='bilinear', align_corners=False
                )
                patches = F.unfold(norm_slices, kernel_size=self.global_crop_size, stride=self.stride)
                patches = patches.transpose(1, 2).reshape(B, -1, C, self.global_crop_size,
                                                          self.global_crop_size).squeeze(2)

                if global_crops.shape[1] == 1:
                    global_crops = global_crops.repeat(1, self.num_channels, 1, 1)
                if patches.ndim == 4:
                    patches = patches.unsqueeze(2).repeat(1, 1, self.num_channels, 1, 1)
                elif patches.shape[2] == 1:
                    patches = patches.repeat(1, 1, self.num_channels, 1, 1)

                return global_crops, patches

    def _save_visualization(self, raw_np, mask_np, rmin, rmax, cmin, cmax, norm_pt, filename, batch_idx):
        """Helper to save debugging plots."""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        view_img = np.clip(raw_np, -1000, 400)

        axes[0].imshow(view_img, cmap='gray')
        axes[0].set_title("Original CT")
        axes[0].axis('off')

        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title("Body Mask")
        axes[1].axis('off')

        # Draw bounding box on overlay
        axes[2].imshow(view_img, cmap='gray')
        axes[2].imshow(mask_np, cmap='Reds', alpha=0.3)
        # rmax and cmax are inclusive indices for plotting
        axes[2].plot([cmin, cmax, cmax, cmin, cmin], [rmin, rmin, rmax, rmax, rmin], color='blue', linewidth=2)
        axes[2].set_title("Overlay + Square Crop Box")
        axes[2].axis('off')

        # Display what the model actually sees (Normalized 224x224)
        model_view = norm_pt.squeeze().cpu().numpy()
        axes[3].imshow(model_view, cmap='gray')
        axes[3].set_title("Model Input (Square Crop & Resized)")
        axes[3].axis('off')

        safe_name = os.path.basename(filename).replace(".npy", "")
        save_path = os.path.join(self.vis_dir, f"vis_{self.vis_count:02d}_{safe_name}_slice{batch_idx}.png")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


class CTMultiScaleDataset(Dataset):
    def __init__(self, config, data_dir, label_csv=None, allowed_volume_names=None):
        self.data_dir = data_dir
        self.config = config
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.npy', '.npz'))])

        # Optional exact-volume filter (Stage 2 fair-train): allowed_volume_names is a
        # set of full VolumeName strings with .nii.gz extension (e.g. 'train_679_a_1.nii.gz'),
        # matching the on-disk .npy by base name. When provided, drop everything not in
        # the set BEFORE label loading so __len__ reflects the subset and no label lookup
        # is attempted for excluded volumes.
        if allowed_volume_names is not None:
            allowed_base = set(str(v).replace('.nii.gz', '') for v in allowed_volume_names)
            n_before = len(self.files)
            self.files = [f for f in self.files
                          if f.replace('.npy', '').replace('.npz', '') in allowed_base]
            print(f"CTMultiScaleDataset: filtered to {len(self.files)}/{n_before} allowed volumes "
                  f"from {data_dir}")

        if len(self.files) == 0:
            print(f"⚠️  No .npy or .npz files found in {data_dir}")

        # Toggle label processing based on whether a CSV was provided
        self.has_labels = label_csv is not None and os.path.exists(label_csv)

        if self.has_labels:
            print(f"Loading labels from {label_csv}...")
            df = pd.read_csv(label_csv)
            id_col = df.columns[0]
            self.class_names = list(df.columns[1:])
            self.labels_map = df.set_index(id_col)
        else:
            print("⚠️ No label_csv provided. Proceeding with dummy labels for feature extraction.")
            self.class_names = ["dummy_label"]
            self.labels_map = None

        print(f"✅ Indexed {len(self.files)} volumes.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        file_path = os.path.join(self.data_dir, filename)

        try:
            if filename.endswith('.npy'):
                volume = np.load(file_path, mmap_mode='r')
            elif filename.endswith('.npz'):
                with np.load(file_path) as data:
                    key = list(data.keys())[0]
                    volume = data[key]
        except Exception as e:
            print(f"❌ Corrupt file {filename}: {e}")
            exit()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            volume = torch.from_numpy(volume.astype(np.float32))

        # Only attempt to match labels if we loaded them
        if self.has_labels:
            # 1. CT-RATE-huggingface-downloads format (e.g., valid_1_a_1.nii.gz)
            base_name = filename.replace('.npy', '.nii.gz').replace('.npz', '.nii.gz')
            # 2. RAD-ChestCT format (e.g., trn24737)
            alt_name = filename.replace('.npy', '').replace('.npz', '')

            if base_name in self.labels_map.index:
                label = self.labels_map.loc[base_name].values.astype(np.float32)
            elif alt_name in self.labels_map.index:
                label = self.labels_map.loc[alt_name].values.astype(np.float32)
            else:
                # Loud failure so it never happens silently again
                raise KeyError(
                    f"Label mapping failed! Could not find '{base_name}' or '{alt_name}' "
                    f"in the CSV index. First 5 keys: {list(self.labels_map.index[:5])}"
                )
        else:
            # Return a dummy label scalar to keep the DataLoader from complaining
            label = np.zeros(1, dtype=np.float32)

        return volume, label, filename

# Code for training TAR dataset
def group_by_volume(data):
    """
    Consumes sequential slices from WebDataset and yields full reconstructed volumes.
    Matches the output signature of CTMultiScaleDataset.
    """
    current_vol_name = None
    current_slices = []
    current_label = None

    for slice_data, meta in data:
        vol_name = meta['original_file']

        # Initialize tracking for the first volume in the stream
        if current_vol_name is None:
            current_vol_name = vol_name
            current_label = meta.get('labels', [])

        # When the volume name changes, yield the completed volume and reset
        if vol_name != current_vol_name:
            current_slices.sort(key=lambda x: x[0])  # Ensure strict z-axis ordering
            vol_array = np.stack([x[1] for x in current_slices], axis=0)
            vol_tensor = torch.from_numpy(vol_array.astype(np.float32))

            # Match the legacy format for the inference save_name downstream
            fake_filename = current_vol_name.replace('.nii.gz', '.npy')
            label_arr = np.array(current_label, dtype=np.float32)

            yield vol_tensor, label_arr, fake_filename

            current_vol_name = vol_name
            current_slices = []
            current_label = meta.get('labels', [])

        current_slices.append((meta['slice_index'], slice_data))

    # Yield the final volume when the stream ends
    if current_vol_name is not None and len(current_slices) > 0:
        current_slices.sort(key=lambda x: x[0])
        vol_array = np.stack([x[1] for x in current_slices], axis=0)
        vol_tensor = torch.from_numpy(vol_array.astype(np.float32))

        fake_filename = current_vol_name.replace('.nii.gz', '.npy')
        label_arr = np.array(current_label, dtype=np.float32)

        yield vol_tensor, label_arr, fake_filename


def get_wds_dataset(config, tar_pattern=None, allowed_volume_names=None):
    """Initializes the WebDataset pipeline.

    Args:
        config: OmegaConf config object.
        tar_pattern: Optional glob pattern for tar files. If None, falls back to
                     config.validation.tar_pattern (legacy) then config.validation.train_tar_pattern.
        allowed_volume_names: Optional set of full VolumeName strings
                              (e.g., 'train_123_a_1.nii.gz'). If provided,
                              only these exact volumes are yielded.
    """
    if tar_pattern is None:
        tar_pattern = getattr(config.validation, 'tar_pattern', None)
        if tar_pattern is None:
            tar_pattern = getattr(config.validation, 'train_tar_pattern', None)
    if not tar_pattern:
        raise ValueError("config.validation.tar_pattern or config.validation.train_tar_pattern must be specified")

    urls = sorted(glob.glob(tar_pattern))
    if not urls:
        raise FileNotFoundError(f"No tar files found matching pattern: {tar_pattern}")

    # Build the volume grouper with optional volume-name filtering baked in
    def make_volume_grouper():
        """Returns a group_by_volume function with volume-name filtering closure."""
        if allowed_volume_names is not None:
            from utils.balanced_subset import filter_volume_by_patient_set
            allowed = allowed_volume_names  # capture in closure
            def filtered_group_by_volume(data):
                for vol_tensor, label_arr, filename in group_by_volume(data):
                    if filter_volume_by_patient_set(filename, allowed):
                        yield vol_tensor, label_arr, filename
            return filtered_group_by_volume
        else:
            return group_by_volume

    # nodesplitter and workersplitter ensure DDP and multi-worker safety
    dataset = (
        wds.WebDataset(urls, nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker)
        .decode()
        .to_tuple("npy", "json")
        .compose(make_volume_grouper())
    )

    return dataset


def get_npy_validation_dataset(config, data_dir, label_csv, max_patients=200, seed=42,
                               allowed_volume_names=None):
    """
    Creates a validation dataset from .npy files, limited to max_patients unique patients.

    Handles the naming convention: valid_{patient_id}_{scan}_{recon}.npy
    Example: valid_373_a_1.npy -> patient_id=373, scan=a, recon=1

    Labels are matched at the patient level (valid_{patient_id}.nii.gz in the CSV).
    All scans/reconstructions for a selected patient are included.

    Args:
        config: OmegaConf config object.
        data_dir: Path to directory containing .npy files.
        label_csv: Path to the label CSV file.
        max_patients: Maximum number of unique patients to include.
        seed: Random seed for patient selection.
        allowed_volume_names: Optional set of full VolumeName strings
                              (e.g., 'valid_123_a_1.nii.gz'). If provided,
                              overrides max_patients random selection and uses
                              exactly these volumes.

    Returns:
        CTMultiScaleDataset filtered to the selected volumes.
    """
    import re
    import random

    # Find all .npy files
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    if not all_files:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")

    # Extract patient, scan, and recon IDs
    scan_pattern = re.compile(r'^valid_(\d+)_([a-z]+)_(\d+)\.npy$')

    # Structure: { patient_id: { scan_id: (recon_id, filename) } }
    patient_to_unique_scans = {}

    for f in all_files:
        match = scan_pattern.match(f)
        if match:
            patient_id = match.group(1)
            scan_id = match.group(2)
            recon_id = int(match.group(3))

            if patient_id not in patient_to_unique_scans:
                patient_to_unique_scans[patient_id] = {}

            # If we haven't seen this scan, or if this recon number is lower than the one we have, save it
            if scan_id not in patient_to_unique_scans[patient_id]:
                patient_to_unique_scans[patient_id][scan_id] = (recon_id, f)
            else:
                if recon_id < patient_to_unique_scans[patient_id][scan_id][0]:
                    patient_to_unique_scans[patient_id][scan_id] = (recon_id, f)

    if not patient_to_unique_scans and allowed_volume_names is None:
        # Fallback: try simpler pattern valid_{id}.npy
        fallback_pattern = re.compile(r'^valid_(\d+)\.npy$')
        for f in all_files:
            match = fallback_pattern.match(f)
            if match:
                patient_id = match.group(1)
                patient_to_unique_scans.setdefault(patient_id, {})['a'] = (1, f)

    # When an explicit allowed_volume_names set is given (e.g. Stage 2 ckpt-val
    # drawn from a train_*.npy dir), do NOT raise on the empty patient map -- the
    # valid_-regex above assumes valid_* names and legitimately matches nothing in
    # a train_* dir; the allowed-name branch below builds selected_files directly.
    if not patient_to_unique_scans and allowed_volume_names is None:
        raise ValueError(
            f"Could not parse patient IDs from filenames in {data_dir}. "
            f"Expected pattern: valid_{{patient_id}}_{{scan}}_{{recon}}.npy"
        )

    all_patient_ids = sorted(patient_to_unique_scans.keys())
    print(f"Found {len(all_patient_ids)} unique patients.")

    # Select files: use allowed_volume_names if provided, otherwise random sample
    if allowed_volume_names is not None:
        # Convert VolumeName strings (e.g., 'valid_123_a_1.nii.gz') to .npy filenames
        # and match against files on disk
        allowed_npy = set()
        for vol_name in allowed_volume_names:
            # Ensure vol_name is a string (defensive against pandas type inference)
            npy_name = str(vol_name).replace('.nii.gz', '.npy')
            if npy_name in all_files:
                allowed_npy.add(npy_name)
        selected_files = sorted(allowed_npy)
        missing = len(allowed_volume_names) - len(selected_files)
        if missing:
            print(f"WARNING: {missing} volumes from balanced subset not found "
                  f"in data directory.")
        print(f"Selected {len(selected_files)} volumes from balanced subset.")
    else:
        random.seed(seed)
        if len(all_patient_ids) > max_patients:
            selected_patients = set(random.sample(all_patient_ids, max_patients))
        else:
            selected_patients = set(all_patient_ids)

        # Collect ONLY the unique scans for the selected patients
        selected_files = []
        for pid in sorted(selected_patients):
            for scan_id, (recon_id, file_name) in patient_to_unique_scans[pid].items():
                selected_files.append(file_name)

        print(f"Selected {len(selected_patients)} patients, {len(selected_files)} unique scans for validation")


    # Create the dataset with label CSV
    dataset = CTMultiScaleDataset(config=config, data_dir=data_dir, label_csv=label_csv)

    # Override the file list to only include selected files
    dataset.files = selected_files

    # Fix label matching: map valid_{pid}_{scan}_{recon}.npy -> valid_{pid}.nii.gz
    # The original CTMultiScaleDataset does: filename.replace('.npy', '.nii.gz')
    # But the CSV keys are patient-level: valid_{pid}.nii.gz
    # We need to override __getitem__ label lookup

    def patched_getitem(idx):
        filename = dataset.files[idx]
        file_path = os.path.join(data_dir, filename)

        try:
            volume = np.load(file_path, mmap_mode='r')
        except Exception as e:
            print(f"Corrupt file {filename}: {e}")
            # Return a zero volume as fallback
            volume = np.zeros((1, 512, 512), dtype=np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            volume = torch.from_numpy(volume.astype(np.float32))

        if dataset.has_labels:
            # CSV keys are volume-level (valid_{pid}_{scan}_{recon}.nii.gz), matching
            # the .npy filename exactly. CT-RATE labels are per-volume, NOT per-patient.
            csv_key = filename.replace('.npy', '.nii.gz')

            if csv_key in dataset.labels_map.index:
                label = dataset.labels_map.loc[csv_key].values.astype(np.float32)
            else:
                label = np.zeros(len(dataset.class_names), dtype=np.float32)
        else:
            label = np.zeros(1, dtype=np.float32)

        return volume, label, filename

    dataset.__getitem__ = patched_getitem
    return dataset