import os
import glob
import random
import torch
import numpy as np
import pandas as pd
import webdataset as wds
import io
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import cv2

# Hardcoded ReX Class Mapping (14 Classes)
REX_CLASS_TO_IDX = {
    "1a": 0, "1b": 1, "1c": 2, "1d": 3, "1e": 4, "1f": 5,
    "2a": 6, "2b": 7, "2c": 8, "2d": 9, "2e": 10, "2f": 11, "2g": 12, "2h": 13
}
NUM_REX_CLASSES = len(REX_CLASS_TO_IDX)


def group_into_slabs(slab_size_mm, fixed_slices=None):
    def _grouper(src):
        buffer = []
        current_scan = None
        for sample in src:
            key = sample.get('__key__', '')
            scan_id = key.rsplit('/', 1)[0] if '/' in key else (key.rsplit('_', 1)[0] if '_' in key else key)

            # --- NEW LOGIC ---
            if fixed_slices is not None and fixed_slices > 0:
                target_length = fixed_slices
            else:
                # Fallback to dynamic calculation if fixed_slices isn't set
                meta = sample.get('json', {})
                z_spacing = meta.get('z_spacing', 1.0)
                target_length = max(1, int(round(slab_size_mm / z_spacing))) if slab_size_mm > 0 else 1
            # -----------------

            if current_scan is not None and scan_id != current_scan:
                if len(buffer) > 0: yield buffer
                buffer = []

            current_scan = scan_id
            buffer.append(sample)

            if len(buffer) >= target_length:
                yield buffer
                buffer = []

        if len(buffer) > 0: yield buffer

    return _grouper

def _extract_fields(slab):
    # Dynamically find the available reconstructions from the first sample's metadata
    first_json = slab[0].get("json", {})
    recons = list(first_json.get("reconstructions", {}).keys())

    # --- NEW FALLBACK FOR SINGLE-RESOLUTION DATASETS ---
    if not recons:
        recon_data = {'default': {'npy': [], 'ts': [], 'rex': []}}
        json_list = []
        for sample in slab:
            json_list.append(sample.get("json", {}))
            # Fallback to standard webdataset extension keys
            recon_data['default']['npy'].append(sample.get("npy"))
            # Depending on how the tar was created, masks might map to 'ts.png' or '_ts.png'
            recon_data['default']['ts'].append(sample.get("ts.png") or sample.get("_ts.png"))
            recon_data['default']['rex'].append(sample.get("rex.png") or sample.get("_rex.png"))
        return recon_data, json_list
    # ---------------------------------------------------

    recon_data = {rec: {'npy': [], 'ts': [], 'rex': []} for rec in recons}
    json_list = []

    for sample in slab:
        json_list.append(sample.get("json"))
        for rec in recons:
            recon_data[rec]['npy'].append(sample.get(f"{rec}.npy"))
            recon_data[rec]['ts'].append(sample.get(f"{rec}_ts.png"))
            recon_data[rec]['rex'].append(sample.get(f"{rec}_rex.png"))

    return recon_data, json_list


class CTNormalizeAndAugment:
    def __init__(self, config):
        self.norm_type = getattr(config.dataset, 'norm_type', 'zscore')
        self.clip_min = getattr(config.dataset, 'clip_min', -997.0)
        self.clip_max = getattr(config.dataset, 'clip_max', 888.0)

    def normalize_slab(self, npy_data_list):
        stacked_npy = np.stack(npy_data_list, axis=0)
        slab = torch.from_numpy(stacked_npy).float().unsqueeze(1)

        if self.norm_type == 'div1000':
            # Divide by 1000 and clip to [-1, 1]
            slab = slab / 1000.0
            return torch.clamp(slab, -1.0, 1.0)
        else:
            # Original Z-score preparation (scaled to 0-1)
            slab = torch.clamp(slab, self.clip_min, self.clip_max)
            range_val = self.clip_max - self.clip_min
            return (slab - self.clip_min) / (range_val if range_val > 0 else 1.0)

    def process_ts_mask_bytes(self, mask_bytes):
        # Decode instantly to numpy, stay at native 128x128 to save RAM and CPU cycles
        if mask_bytes is None: return np.zeros((128, 128), dtype=np.uint8)
        try:
            nparr = np.frombuffer(mask_bytes, np.uint8)
            mask = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            return mask if mask is not None else np.zeros((128, 128), dtype=np.uint8)
        except Exception:
            return np.zeros((128, 128), dtype=np.uint8)

    def process_rex_mask_bytes(self, mask_bytes, active_classes):
        mask_np = np.zeros((NUM_REX_CLASSES, 128, 128), dtype=np.uint8)
        if mask_bytes is None or not active_classes: return mask_np

        try:
            nparr = np.frombuffer(mask_bytes, np.uint8)
            mask = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if mask is None or mask.size == 0: return mask_np

            N = mask.shape[0] // 128
            if N > 0 and N == len(active_classes):
                mask_reshaped = mask.reshape(N, 128, 128)
                for i, class_name in enumerate(active_classes):
                    if class_name in REX_CLASS_TO_IDX:
                        idx = REX_CLASS_TO_IDX[class_name]
                        mask_np[idx] = mask_reshaped[i]
            return mask_np
        except Exception:
            return mask_np

    def __call__(self, recon_data, json_list):
        out_vols, out_ts, out_rex = [], [], []

        for rec, data in recon_data.items():
            slab_tensor = self.normalize_slab(data['npy'])
            ts_masks = [self.process_ts_mask_bytes(data['ts'][z]) for z in range(len(data['ts']))]

            # --- UPDATED: Safely handle ReX classes whether it's a dict or flat list ---
            rex_masks = []
            for z in range(len(data['npy'])):
                active_classes = json_list[z].get('rex_active_classes', [])
                if isinstance(active_classes, dict):
                    active_classes = active_classes.get(rec, [])

                rex_masks.append(self.process_rex_mask_bytes(data['rex'][z], active_classes))
            # --------------------------------------------------------------------------

            out_vols.append(slab_tensor.squeeze(1))
            out_ts.append(torch.from_numpy(np.stack(ts_masks)).long())
            out_rex.append(torch.from_numpy(np.stack(rex_masks)).to(torch.uint8))

        # --- UPDATED: Check for ReX samples safely ---
        rex_ac = json_list[0].get('rex_active_classes', [])
        if isinstance(rex_ac, dict):
            is_rex_sample = any(bool(rex_ac.get(r, [])) for r in recon_data.keys())
        else:
            is_rex_sample = bool(rex_ac)

        return {
            'raw_volumes': out_vols,  # List of tensors [D, H, W] for each recon
            'ts_masks': out_ts,  # List of tensors for each recon
            'rex_masks': out_rex,  # List of tensors for each recon
            'is_rex_shard': torch.tensor(is_rex_sample, dtype=torch.bool)
        }

def npy_decoder(data): return np.load(io.BytesIO(data))


def json_decoder(data): return json.loads(data.decode("utf-8"))


def null_decoder(data): return data


class CTWebDatasetLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_path = cfg.dataset.dataset_path
        self.batch_size = getattr(cfg.train, 'batch_size_per_gpu', None)
        self.num_workers = getattr(cfg.dataset, 'num_workers', 4)
        self.shuffle_buffer = getattr(cfg.dataset, 'shuffle_buffer', 1000)
        self.prefetch_factor = getattr(cfg.dataset, 'prefetch_factor', 2)

        if getattr(cfg, 'train', None) is not None:
            self.transform = CTNormalizeAndAugment(cfg)
        self.train_loader = self._build_loader()

    def _transform_with_masks(self, tpl):
        recon_data, json_list = tpl
        output = self.transform(recon_data, json_list)

        # Extract labels from the middle slice's metadata
        mid_idx = len(json_list) // 2
        mid_meta = json_list[mid_idx]

        if mid_meta:
            # --- UPDATED: Support both JSON structures ---
            if 'reconstructions' in mid_meta and mid_meta['reconstructions']:
                first_recon = list(mid_meta['reconstructions'].keys())[0]
                first_recon_data = mid_meta['reconstructions'].get(first_recon, {})
                if 'labels' in first_recon_data:
                    output['dataset_labels'] = torch.tensor(first_recon_data['labels'], dtype=torch.float32)
            elif 'labels' in mid_meta:
                # Single-resolution fallback
                output['dataset_labels'] = torch.tensor(mid_meta['labels'], dtype=torch.float32)
            # -------------------------------------------

        return output

    def _build_single_dataset(self, urls, slab_size):
        if not urls: return None

        # Pull the new config variable (defaults to None if missing from older YAMLs)
        fixed_slices = getattr(self.cfg.dataset, 'fixed_slices', None)

        return (
            wds.WebDataset(urls, resampled=True, shardshuffle=True, nodesplitter=wds.split_by_node)
            .decode(
                wds.handle_extension("npy", npy_decoder),
                wds.handle_extension("json", json_decoder),
                wds.handle_extension("png", null_decoder)
            )
            .compose(group_into_slabs(slab_size, fixed_slices=fixed_slices))
        )

    def _build_loader(self):
        slab_size = getattr(self.cfg.dataset, 'slab_size', 0.0)
        rex_ratio = getattr(self.cfg.dataset, 'rex_ratio', 0.5)
        # When True, stream normal (shards-*.tar) + abnormal (rex-shards-*.tar)
        # together at their natural shard-count prevalence, bypassing the
        # rex_ratio RandomMix entirely (no ReX-driven oversampling). ReX
        # supervision/guidance are disabled separately via auxiliary.use_rex /
        # crops.use_rex_guidance. Used by DALE-CT-1S-v2 (ReX-free).
        natural_rex_mix = getattr(self.cfg.dataset, 'natural_rex_mix', False)

        std_urls = glob.glob(os.path.join(self.dataset_path, "shards-*.tar"))
        rex_urls = glob.glob(os.path.join(self.dataset_path, "rex-shards-*.tar"))

        if std_urls: random.shuffle(std_urls)
        if rex_urls: random.shuffle(rex_urls)

        if natural_rex_mix:
            # Single resampled WebDataset over both shard sets -> samples shards
            # uniformly, giving natural normal:abnormal prevalence by shard count
            # (rex_ratio is ignored on this path).
            all_urls = std_urls + rex_urls
            if not all_urls:
                raise ValueError(f"🚨 No 'shards-*.tar' or 'rex-shards-*.tar' files found in {self.dataset_path}. Check your YAML config!")
            random.shuffle(all_urls)
            dataset = (self._build_single_dataset(all_urls, slab_size)
                       .shuffle(self.shuffle_buffer)
                       .map(_extract_fields)
                       .map(self._transform_with_masks))
        else:
            datasets, probs = [], []
            if std_urls:
                datasets.append(self._build_single_dataset(std_urls, slab_size))
                probs.append(1.0 - rex_ratio if rex_urls else 1.0)
            if rex_urls:
                datasets.append(self._build_single_dataset(rex_urls, slab_size))
                probs.append(rex_ratio if std_urls else 1.0)

            if not datasets:
                raise ValueError(f"🚨 No 'shards-*.tar' files found in {self.dataset_path}. Check your YAML config!")

            if len(datasets) == 2:
                # Pass the operations as sequential arguments to DataPipeline
                dataset = wds.DataPipeline(
                    wds.RandomMix(datasets, probs),
                    wds.shuffle(self.shuffle_buffer),
                    wds.map(_extract_fields),
                    wds.map(self._transform_with_masks)
                )
            else:
                # A single WebDataset object still supports the fluent chaining API
                dataset = datasets[0].shuffle(self.shuffle_buffer).map(_extract_fields).map(self._transform_with_masks)

        def _worker_init_fn(worker_id):
            import cv2, torch
            cv2.setNumThreads(0)
            torch.set_num_threads(1)

        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=False, drop_last=True, collate_fn=self.custom_collate,
            prefetch_factor=self.prefetch_factor, worker_init_fn=_worker_init_fn
        )

    @staticmethod
    def custom_collate(batch):
        out_dict = {
            'volumes_list': [s['raw_volumes'] for s in batch],  # List[List[Tensor]]
            'ts_masks_list': [s['ts_masks'] for s in batch],
            'rex_masks_list': [s['rex_masks'] for s in batch],
            'is_rex_shard': torch.stack([s['is_rex_shard'] for s in batch])
        }
        if 'dataset_labels' in batch[0]:
            out_dict['dataset_labels'] = torch.stack([s['dataset_labels'] for s in batch])
        return out_dict