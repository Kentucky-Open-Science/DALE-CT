import time
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from omegaconf import OmegaConf

from dataloaders.datasetloader_web_ctrate import CTWebDatasetLoader
from data.guided_data_augmentation_CT_RATE import BatchedGuidedDataAugmentationDINO_CT

# --- Dictionaries for Visualizations ---
TOTALSEG_CLASSES = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder", 5: "liver",
    6: "stomach", 7: "pancreas", 8: "adrenal_gland_right", 9: "adrenal_gland_left",
    10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left", 12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right", 15: "esophagus",
    16: "trachea", 17: "thyroid_gland", 18: "small_bowel", 19: "duodenum",
    20: "colon", 21: "urinary_bladder", 22: "prostate", 23: "kidney_cyst_left",
    24: "kidney_cyst_right", 25: "sacrum", 26: "vertebrae_S1", 27: "vertebrae_L5",
    28: "vertebrae_L4", 29: "vertebrae_L3", 30: "vertebrae_L2", 31: "vertebrae_L1",
    32: "vertebrae_T12", 33: "vertebrae_T11", 34: "vertebrae_T10", 35: "vertebrae_T9",
    36: "vertebrae_T8", 37: "vertebrae_T7", 38: "vertebrae_T6", 39: "vertebrae_T5",
    40: "vertebrae_T4", 41: "vertebrae_T3", 42: "vertebrae_T2", 43: "vertebrae_T1",
    44: "vertebrae_C7", 45: "vertebrae_C6", 46: "vertebrae_C5", 47: "vertebrae_C4",
    48: "vertebrae_C3", 49: "vertebrae_C2", 50: "vertebrae_C1", 51: "heart",
    52: "aorta", 53: "pulmonary_vein", 54: "brachiocephalic_trunk",
    55: "subclavian_artery_right", 56: "subclavian_artery_left",
    57: "common_carotid_artery_right", 58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left", 60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left", 62: "superior_vena_cava", 63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein", 65: "iliac_artery_left", 66: "iliac_artery_right",
    67: "iliac_vena_left", 68: "iliac_vena_right", 69: "humerus_left", 70: "humerus_right",
    71: "scapula_left", 72: "scapula_right", 73: "clavicula_left", 74: "clavicula_right",
    75: "femur_left", 76: "femur_right", 77: "hip_left", 78: "hip_right",
    79: "spinal_cord", 80: "gluteus_maximus_left", 81: "gluteus_maximus_right",
    82: "gluteus_medius_left", 83: "gluteus_medius_right", 84: "gluteus_minimus_left",
    85: "gluteus_minimus_right", 86: "autochthon_left", 87: "autochthon_right",
    88: "iliopsoas_left", 89: "iliopsoas_right", 90: "brain", 91: "skull",
    92: "rib_left_1", 93: "rib_left_2", 94: "rib_left_3", 95: "rib_left_4",
    96: "rib_left_5", 97: "rib_left_6", 98: "rib_left_7", 99: "rib_left_8",
    100: "rib_left_9", 101: "rib_left_10", 102: "rib_left_11", 103: "rib_left_12",
    104: "rib_right_1", 105: "rib_right_2", 106: "rib_right_3", 107: "rib_right_4",
    108: "rib_right_5", 109: "rib_right_6", 110: "rib_right_7", 111: "rib_right_8",
    112: "rib_right_9", 113: "rib_right_10", 114: "rib_right_11", 115: "rib_right_12",
    116: "sternum", 117: "costal_cartilages"
}
REX_CLASSES = [
    "1a", "1b", "1c", "1d", "1e", "1f",
    "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h"
]


def plot_and_save_single_sample(crops_dict, cfg, batch_idx, save_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Fetch the normalization type from the config
    norm_type = getattr(cfg.dataset, 'norm_type', 'zscore')

    if norm_type != 'div1000':
        clip_min = getattr(cfg.dataset, 'clip_min', -997.0)
        clip_max = getattr(cfg.dataset, 'clip_max', 888.0)
        mean_hu = getattr(cfg.dataset, 'mean_hu', -142.39)
        std_hu = getattr(cfg.dataset, 'std_hu', 360.97)
        range_val = clip_max - clip_min if (clip_max - clip_min) > 0 else 1.0
        norm_mean = (mean_hu - clip_min) / range_val
        norm_std = std_hu / range_val

    def draw_crop_row(axes_row, img_tensor, cls_ts, cls_rex, patch_ts, patch_rex, true_rex_mask, title_prefix):
        img_np = img_tensor.cpu().numpy().squeeze()

        # --- Dynamic Un-Normalization ---
        if norm_type == 'div1000':
            # div1000 outputs [-1, 1]. Map back to [0, 1] for matplotlib.
            img_np = (img_np + 1.0) / 2.0
        else:
            # Reverse the z-score normalization
            img_np = (img_np * norm_std) + norm_mean

        # 1. Plot Base Image
        axes_row[0].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        axes_row[0].set_title(f"{title_prefix} Raw CT")
        axes_row[0].axis('off')

        # 2. Extract Top Labels
        cls_ts_np = cls_ts.cpu().numpy()
        cls_rex_np = cls_rex.cpu().numpy()

        top_ts_indices = np.argsort(cls_ts_np)[::-1][:5]
        top_rex_indices = np.argsort(cls_rex_np)[::-1][:3]

        labels, values, colors = [], [], []

        for idx in top_ts_indices:
            if cls_ts_np[idx] > 0.005:
                name = TOTALSEG_CLASSES.get(idx, f"TS-{idx}")
                labels.append(name)
                values.append(cls_ts_np[idx])
                colors.append('royalblue')

        for idx in top_rex_indices:
            if cls_rex_np[idx] > 0.001:
                labels.append(f"ReX: {REX_CLASSES[idx]}")
                values.append(cls_rex_np[idx])
                colors.append('crimson')

        # 3. Plot CLS Bar Chart
        if labels:
            y_pos = np.arange(len(labels))
            axes_row[1].barh(y_pos, values, color=colors)
            axes_row[1].set_yticks(y_pos)
            axes_row[1].set_yticklabels(labels)
            axes_row[1].invert_yaxis()
            axes_row[1].set_xlim(0, 1.0)
            axes_row[1].set_title("[CLS] Volume Distribution")
        else:
            axes_row[1].text(0.5, 0.5, "Background Only", ha='center', va='center')
            axes_row[1].axis('off')

        # 4. Plot Patch Heatmaps (What the model targets)
        H, W = img_np.shape
        G = int(np.sqrt(patch_ts.shape[0]))

        patch_ts_grid = patch_ts.cpu().numpy().reshape(G, G, -1)
        patch_rex_grid = patch_rex.cpu().numpy().reshape(G, G, -1)

        # TS Heatmap
        axes_row[2].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        if len(values) > 0 and 'royalblue' in colors:
            top_ts = top_ts_indices[0]
            hm_ts = cv2.resize(patch_ts_grid[:, :, top_ts], (W, H), interpolation=cv2.INTER_NEAREST)
            axes_row[2].imshow(hm_ts, cmap='jet', alpha=0.4, vmin=0, vmax=1)
            axes_row[2].set_title(f"Patch Map: {TOTALSEG_CLASSES.get(top_ts)}")
        else:
            axes_row[2].set_title("No Major TS Organ")
        axes_row[2].axis('off')

        # ReX Heatmap
        axes_row[3].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        if any(c == 'crimson' for c in colors):
            top_rex = top_rex_indices[0]
            hm_rex = cv2.resize(patch_rex_grid[:, :, top_rex], (W, H), interpolation=cv2.INTER_NEAREST)
            axes_row[3].imshow(hm_rex, cmap='magma', alpha=0.5, vmin=0, vmax=1)
            axes_row[3].set_title(f"Patch Map: ReX {REX_CLASSES[top_rex]}")
        else:
            axes_row[3].set_title("No ReX Abnormalities")
        axes_row[3].axis('off')

        # 5. Plot True High-Res Ground Truth
        axes_row[4].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        if any(c == 'crimson' for c in colors):
            top_rex = top_rex_indices[0]
            gt_hm = true_rex_mask[top_rex].cpu().numpy()
            axes_row[4].imshow(gt_hm, cmap='spring', alpha=0.5, vmin=0, vmax=1)
            axes_row[4].set_title(f"High-Res GT Mask: ReX {REX_CLASSES[top_rex]}")
        else:
            axes_row[4].set_title("No ReX GT")
        axes_row[4].axis('off')

    # Updated figure to accommodate 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    plt.subplots_adjust(wspace=0.3)

    # Process Global Crop 0
    draw_crop_row(
        axes[0],
        crops_dict['global_crops'][0][batch_idx],
        crops_dict['global_ts_crop_ratios'][0][batch_idx],
        crops_dict['global_rex_crop_ratios'][0][batch_idx],
        crops_dict['global_ts_patch_ratios'][0][batch_idx],
        crops_dict['global_rex_patch_ratios'][0][batch_idx],
        crops_dict['global_rex_masks'][batch_idx],  # High-res GT
        "Global Crop"
    )

    # Process Local Crop 0
    draw_crop_row(
        axes[1],
        crops_dict['local_crops'][0][batch_idx],
        crops_dict['local_ts_crop_ratios'][0][batch_idx],
        crops_dict['local_rex_crop_ratios'][0][batch_idx],
        crops_dict['local_ts_patch_ratios'][0][batch_idx],
        crops_dict['local_rex_patch_ratios'][0][batch_idx],
        crops_dict['local_rex_masks'][batch_idx],  # High-res GT
        "Local Crop"
    )

    plt.suptitle(f"LeJEPA V2 ReX Alignment Validation - Sample {save_idx}", fontsize=16)
    save_path = os.path.join(output_dir, f"v2_rex_abnormality_with_gt_{save_idx:02d}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  -> Saved {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Scan CT-RATE dataloader for ReX abnormalities and plot them.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--batches", type=int, default=50, help="Max number of batches to search through")
    parser.add_argument("--num_save", type=int, default=5, help="Number of ReX positive samples to find and plot")
    parser.add_argument("--save_dir", type=str, default="./visualizations", help="Output directory for plots")
    args = parser.parse_args()

    print(f"\nLoading configuration from: {args.config}")
    cfg = OmegaConf.load(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n[1] Initializing CTWebDatasetLoader...")
    init_start = time.time()
    loader_wrapper = CTWebDatasetLoader(cfg)
    train_loader = loader_wrapper.train_loader
    print(f"Initialization took {time.time() - init_start:.3f} seconds.")

    print("\n[2] Initializing GPU Augmentor...")
    augmentor = BatchedGuidedDataAugmentationDINO_CT(cfg).to(device)

    print(f"\nHunting for {args.num_save} ReX abnormalities across maximum {args.batches} batches...")
    print("=" * 80)

    train_iter = iter(train_loader)
    saved_count = 0

    for i in range(args.batches):
        if saved_count >= args.num_save:
            break

        try:
            batch = next(train_iter)
        except StopIteration:
            print("Reached end of dataloader stream.")
            break

        # The augmentor now handles moving specific slices to the GPU natively.
        # We just pass the nested lists directly from the CPU dataloader.
        volumes = batch['volumes_list']
        ts_masks = batch['ts_masks_list']
        rex_masks = batch['rex_masks_list']

        crops, aug_metrics = augmentor(volumes, ts_masks, rex_masks, profile=True)

        if torch.cuda.is_available(): torch.cuda.synchronize()

        batch_size = crops['global_crops'][0].shape[0]
        found_in_batch = 0

        for b in range(batch_size):
            global_rex_sum = crops['global_rex_crop_ratios'][0][b].sum().item()
            local_rex_sum = crops['local_rex_crop_ratios'][0][b].sum().item()

            if global_rex_sum > 0.0 or local_rex_sum > 0.0:
                print(
                    f"[*] Found ReX abnormality in Batch {i:02d}, Crop Index {b:02d}! (Global %: {global_rex_sum:.3f}, Local %: {local_rex_sum:.3f})")
                plot_and_save_single_sample(crops, cfg, batch_idx=b, save_idx=saved_count, output_dir=args.save_dir)
                saved_count += 1
                found_in_batch += 1

                if saved_count >= args.num_save:
                    break

        if found_in_batch == 0:
            print(f"Batch {i:02d} scanned. No ReX abnormalities found in these {batch_size} crops.")

    print("=" * 80)
    print(f"Finished! Successfully captured {saved_count}/{args.num_save} requested ReX visualizations.")


if __name__ == "__main__":
    main()