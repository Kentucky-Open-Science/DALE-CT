import random
import torch
import torch.nn as nn
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.v2.functional as F_v2

from utils.config import load_model_configs


class GuidedRandomCrop(object):
    def __init__(self, scale=(0.05, 0.4)):
        self.scale = scale

    def get_params(self, image_tensor, pixel_coords=None):
        _, img_h, img_w = image_tensor.shape
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        crop_size = int(min(img_h, img_w) * scale_factor)

        if pixel_coords is not None and len(pixel_coords) > 0:
            idx = torch.randint(0, len(pixel_coords), (1,)).item()
            center_y, center_x = pixel_coords[idx, 0].item(), pixel_coords[idx, 1].item()
        else:
            center_y, center_x = random.randint(0, img_h), random.randint(0, img_w)

        top = max(0, min(center_y - crop_size // 2, img_h - crop_size))
        left = max(0, min(center_x - crop_size // 2, img_w - crop_size))
        return top, left, min(crop_size, img_h - top), min(crop_size, img_w - left)

    def get_params_inside(self, image_shape, global_params, pixel_coords=None):
        _, img_h, img_w = image_shape
        g_top, g_left, g_h, g_w = global_params
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        crop_size = min(int(min(img_h, img_w) * scale_factor), g_h, g_w)

        center_y, center_x = None, None
        if pixel_coords is not None and len(pixel_coords) > 0:
            y_coords, x_coords = pixel_coords[:, 0], pixel_coords[:, 1]
            valid_mask = (y_coords >= g_top) & (y_coords < g_top + g_h) & \
                         (x_coords >= g_left) & (x_coords < g_left + g_w)
            valid_coords = pixel_coords[valid_mask]

            if len(valid_coords) > 0:
                idx = torch.randint(0, len(valid_coords), (1,)).item()
                center_y, center_x = valid_coords[idx, 0].item(), valid_coords[idx, 1].item()

        if center_y is None or center_x is None:
            center_y = random.randint(g_top, g_top + g_h - 1)
            center_x = random.randint(g_left, g_left + g_w - 1)

        top = max(g_top, min(center_y - crop_size // 2, g_top + g_h - crop_size))
        left = max(g_left, min(center_x - crop_size // 2, g_left + g_w - crop_size))
        return top, left, crop_size, crop_size


class BatchedRandomGamma(nn.Module):
    def __init__(self, gamma_range=(0.7, 1.5), p=0.8, input_range='0_1'):
        super().__init__()
        self.gamma_range = gamma_range
        self.p = p
        self.input_range = input_range

    def forward(self, img):
        B = img.shape[0]
        gammas = torch.empty(B, 1, 1, 1, device=img.device).uniform_(self.gamma_range[0], self.gamma_range[1])
        apply_mask = (torch.rand(B, 1, 1, 1, device=img.device) < self.p).float()
        gammas = gammas * apply_mask + (1.0 - apply_mask)

        if self.input_range == '-1_1':
            # Shift [-1, 1] to [0, 1] to avoid NaNs with fractional exponents
            img = (img + 1.0) / 2.0
            # --- THE FIX: Clamp strictly to 0.0 to catch floating-point underflows ---
            img = torch.clamp(img, min=0.0, max=1.0)
            img = torch.pow(img, gammas)
            # Shift back to [-1, 1]
            return img * 2.0 - 1.0

        return torch.pow(img, gammas)


class BatchedRandomGaussianBlur(nn.Module):
    def __init__(self, kernel_size=(3, 3), p=0.5):
        super().__init__()
        self.blur = v2.GaussianBlur(kernel_size=kernel_size)
        self.p = p

    def forward(self, img):
        if self.p == 1.0: return self.blur(img)
        if self.p == 0.0: return img

        B = img.shape[0]
        apply_mask = torch.rand(B, device=img.device) < self.p
        if not apply_mask.any(): return img

        out = img.clone()
        out[apply_mask] = self.blur(img[apply_mask])
        return out


class BatchedGuidedDataAugmentationDINO_CT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.local_crops_number = cfg.crops.local_crops_number
        self.global_crops_number = cfg.crops.global_crops_number
        self.guidance_probability = getattr(cfg.crops, "guidance_probability", 0.5)
        self.local_crops_inside_global = getattr(cfg.crops, "local_crops_inside_global", False)

        self.local_crops_size = cfg.crops.local_crops_size
        self.global_crops_size = cfg.crops.global_crops_size
        self.num_classes = 118
        self.num_rex_classes = 14  # Hardcoded ReX Classes count
        model_params = load_model_configs(cfg.train.model_type)
        self.patch_size = model_params.patch_size

        self.guided_crop_gen_global = GuidedRandomCrop(scale=cfg.crops.global_crops_scale)
        self.guided_crop_gen_local = GuidedRandomCrop(scale=cfg.crops.local_crops_scale)

        self.norm_type = getattr(cfg.dataset, 'norm_type', 'zscore')

        if self.norm_type == 'div1000':
            final_norm = nn.Identity()
            input_range = '-1_1'
        else:
            clip_min = getattr(cfg.dataset, 'clip_min', -997.0)
            clip_max = getattr(cfg.dataset, 'clip_max', 888.0)
            mean_hu = getattr(cfg.dataset, 'mean_hu', -142.39)
            std_hu = getattr(cfg.dataset, 'std_hu', 360.97)

            range_val = clip_max - clip_min if (clip_max - clip_min) > 0 else 1.0
            norm_mean = (mean_hu - clip_min) / range_val
            norm_std = std_hu / range_val
            final_norm = v2.Normalize(mean=[norm_mean], std=[norm_std])
            input_range = '0_1'

        gamma_prob = getattr(cfg.crops, 'gamma_prob', 0.8)
        gamma_range = getattr(cfg.crops, 'gamma_range', [0.7, 1.5])
        self.gamma_aug = BatchedRandomGamma(gamma_range=gamma_range, p=gamma_prob, input_range=input_range)

        self.global_transfo1 = v2.Compose([self.gamma_aug, BatchedRandomGaussianBlur(p=0.5), final_norm])
        self.global_transfo2 = v2.Compose([self.gamma_aug, BatchedRandomGaussianBlur(p=0.1), final_norm])
        self.local_transfo = v2.Compose([self.gamma_aug, final_norm])
        self.register_buffer('_device_indicator', torch.empty(0))

    def apply_crop_and_resize(self, image, ts_mask, rex_mask, crop_params, output_size):
        top, left, h, w = crop_params

        img_crop = F_v2.crop(image, top, left, h, w)
        # Changed to BILINEAR for the continuous image data
        img_resized = F_v2.resize(img_crop, (output_size, output_size), interpolation=InterpolationMode.BILINEAR)

        ts_crop = F_v2.crop(ts_mask.unsqueeze(0), top, left, h, w).squeeze(0)
        ts_resized = F_v2.resize(ts_crop.unsqueeze(0), (output_size, output_size),
                                 interpolation=InterpolationMode.NEAREST).squeeze(0)

        rex_crop = F_v2.crop(rex_mask, top, left, h, w)
        rex_resized = F_v2.resize(rex_crop, (output_size, output_size), interpolation=InterpolationMode.NEAREST)

        if random.random() < 0.5:
            img_resized = F_v2.hflip(img_resized)
            ts_resized = F_v2.hflip(ts_resized)
            rex_resized = F_v2.hflip(rex_resized)

        return img_resized, ts_resized, rex_resized

    def _compute_ratios_vectorized(self, masks_bchw, is_ts=True):
        """
        Calculates crop and patch ratios for an entire batch of views simultaneously.
        """
        if is_ts:
            # TS masks are [B, H, W] containing class indices
            masks_one_hot = torch.nn.functional.one_hot(masks_bchw, num_classes=self.num_classes).float()
            # Permute to [B, C, H, W] for pooling
            masks_float = masks_one_hot.permute(0, 3, 1, 2)
        else:
            # CRITICAL FIX: The PNGs load active pixels as 255.
            # We must binarize them to 1.0 before averaging!
            masks_float = (masks_bchw > 0).float()

        # 1. Global Crop Ratios: Simple mean over spatial dims [B, C]
        crop_ratios = masks_float.mean(dim=(2, 3))

        # 2. Patch Ratios: Vectorized calculation using avg_pool2d
        patch_ratios_2d = torch.nn.functional.avg_pool2d(
            masks_float,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Flatten the spatial grid: [B, C, G, G] -> [B, G*G, C]
        B, C, G1, G2 = patch_ratios_2d.shape
        patch_ratios = patch_ratios_2d.view(B, C, G1 * G2).permute(0, 2, 1).contiguous()

        return crop_ratios, patch_ratios

    def forward(self, volumes_list, ts_masks_list, rex_masks_list, profile=False):
        import time
        metrics = {}
        if profile and torch.cuda.is_available(): torch.cuda.synchronize()
        t_start = time.perf_counter()

        B = len(volumes_list)
        global_crops_1, global_crops_2 = [], []
        local_crops_list = [[] for _ in range(self.local_crops_number)]
        g_ts_1, g_ts_2, g_rex_1, g_rex_2 = [], [], [], []
        l_ts_list = [[] for _ in range(self.local_crops_number)]
        l_rex_list = [[] for _ in range(self.local_crops_number)]

        out_dict = {
            'global_ts_crop_ratios': [], 'global_ts_patch_ratios': [],
            'global_rex_crop_ratios': [], 'global_rex_patch_ratios': [],
            'local_ts_crop_ratios': [], 'local_ts_patch_ratios': [],
            'local_rex_crop_ratios': [], 'local_rex_patch_ratios': [],
        }

        device = self._device_indicator.device

        # --- 1. Python Cropping Loop (Optimized Lazy Loading) ---
        for b in range(B):
            vol_recons = volumes_list[b]
            ts_recons = ts_masks_list[b]
            rex_recons = rex_masks_list[b]
            num_recons = len(vol_recons)

            base_global_params_norm = None

            # Generate Global Crops
            for i in range(self.global_crops_number):
                r_idx = random.randint(0, num_recons - 1)

                # Extract references without moving data to GPU yet
                vol_r = vol_recons[r_idx]
                ts_r = ts_recons[r_idx]
                rex_r = rex_recons[r_idx]
                num_slices, H, W = vol_r.shape

                # Identify the target slice
                mid_idx = num_slices // 2

                # Move ONLY the target slice to the GPU
                mid_img = vol_r[mid_idx].unsqueeze(0).to(device, non_blocking=True)
                mid_ts_128 = ts_r[mid_idx].unsqueeze(0).unsqueeze(0).float().to(device, non_blocking=True)
                mid_rex_128 = rex_r[mid_idx].unsqueeze(0).float().to(device, non_blocking=True)

                # Upscale ONLY the target slice
                mid_ts = torch.nn.functional.interpolate(mid_ts_128, size=(H, W), mode='nearest').squeeze(0).squeeze(
                    0).long()
                mid_rex = torch.nn.functional.interpolate(mid_rex_128, size=(H, W), mode='nearest').squeeze(0).byte()

                params = self.guided_crop_gen_global.get_params(mid_img, None)

                if i == 0:
                    base_global_params_norm = (params[0] / H, params[1] / W, params[2] / H, params[3] / W)

                img_res, ts_res, rex_res = self.apply_crop_and_resize(mid_img, mid_ts, mid_rex, params,
                                                                      self.global_crops_size)

                if i == 0:
                    global_crops_1.append(img_res);
                    g_ts_1.append(ts_res);
                    g_rex_1.append(rex_res)
                else:
                    global_crops_2.append(img_res);
                    g_ts_2.append(ts_res);
                    g_rex_2.append(rex_res)

            # Generate Local Crops
            for i in range(self.local_crops_number):
                r_idx = random.randint(0, num_recons - 1)

                vol_r = vol_recons[r_idx]
                ts_r = ts_recons[r_idx]
                rex_r = rex_recons[r_idx]
                num_slices, H, W = vol_r.shape

                # Identify a random target slice
                z = random.randint(0, num_slices - 1)

                # Move ONLY the target slice to the GPU
                img_z = vol_r[z].unsqueeze(0).to(device, non_blocking=True)
                ts_z_128 = ts_r[z].unsqueeze(0).unsqueeze(0).float().to(device, non_blocking=True)
                rex_z_128 = rex_r[z].unsqueeze(0).float().to(device, non_blocking=True)

                # Upscale ONLY the target slice
                ts_z = torch.nn.functional.interpolate(ts_z_128, size=(H, W), mode='nearest').squeeze(0).squeeze(
                    0).long()
                rex_z = torch.nn.functional.interpolate(rex_z_128, size=(H, W), mode='nearest').squeeze(0).byte()

                coords = None

                if random.random() < self.guidance_probability:
                    indices = torch.nonzero(rex_z.sum(dim=0) > 0, as_tuple=False)
                    if indices.shape[0] > 0: coords = indices

                if coords is None:
                    indices = torch.nonzero(ts_z > 0, as_tuple=False)
                    if indices.shape[0] > 0: coords = indices

                if self.local_crops_inside_global and base_global_params_norm:
                    g_top = int(base_global_params_norm[0] * H)
                    g_left = int(base_global_params_norm[1] * W)
                    g_h = int(base_global_params_norm[2] * H)
                    g_w = int(base_global_params_norm[3] * W)
                    params = self.guided_crop_gen_local.get_params_inside((1, H, W), (g_top, g_left, g_h, g_w), coords)
                else:
                    params = self.guided_crop_gen_local.get_params(img_z, coords)

                img_res, ts_res, rex_res = self.apply_crop_and_resize(img_z, ts_z, rex_z, params, self.local_crops_size)
                local_crops_list[i].append(img_res);
                l_ts_list[i].append(ts_res);
                l_rex_list[i].append(rex_res)

        if profile and torch.cuda.is_available(): torch.cuda.synchronize()
        t_crop = time.perf_counter()
        metrics['crop_loop_time'] = t_crop - t_start

        # --- 2. Batched GPU Transforms ---
        out_dict['global_crops'] = [
            self.global_transfo1(torch.stack(global_crops_1)),
            self.global_transfo2(torch.stack(global_crops_2))
        ]
        out_dict['local_crops'] = [self.local_transfo(torch.stack(lc)) for lc in local_crops_list]

        if profile and torch.cuda.is_available(): torch.cuda.synchronize()
        t_transfo = time.perf_counter()
        metrics['transforms_time'] = t_transfo - t_crop

        # --- 3. Vectorized Label Computation ---
        def append_ratios(ts_stack, rex_stack, is_global=True):
            ts_cr, ts_pr = self._compute_ratios_vectorized(torch.stack(ts_stack), is_ts=True)
            rex_cr, rex_pr = self._compute_ratios_vectorized(torch.stack(rex_stack), is_ts=False)
            prefix = 'global' if is_global else 'local'
            out_dict[f'{prefix}_ts_crop_ratios'].append(ts_cr)
            out_dict[f'{prefix}_ts_patch_ratios'].append(ts_pr)
            out_dict[f'{prefix}_rex_crop_ratios'].append(rex_cr)
            out_dict[f'{prefix}_rex_patch_ratios'].append(rex_pr)

        append_ratios(g_ts_1, g_rex_1, is_global=True)
        if self.global_crops_number > 1: append_ratios(g_ts_2, g_rex_2, is_global=True)
        for i in range(self.local_crops_number): append_ratios(l_ts_list[i], l_rex_list[i], is_global=False)

        if profile and torch.cuda.is_available(): torch.cuda.synchronize()
        metrics['label_gen_time'] = time.perf_counter() - t_transfo
        metrics['total_aug_time'] = time.perf_counter() - t_start

        if profile:
            out_dict['global_rex_masks'] = torch.stack(g_rex_1)
            out_dict['local_rex_masks'] = torch.stack(l_rex_list[0])
            return out_dict, metrics
        return out_dict