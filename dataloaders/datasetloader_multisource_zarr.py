"""
Zarr-native dataloader for the multi-source chest-CT zarr pool.

Reads per-case zarr v3 groups (members: image [H,W,Z] int16, organ_mask [H,W,Z]
uint8, body_mask [H,W,Z] uint8) directly and feeds the existing GPU augmentor
(BatchedGuidedDataAugmentationDINO_CT) unchanged.

Array layout is [H, W, Z] with chunks (64, 64, 64): slices are the LAST axis and
the Z-chunk is 64. A slab read is `image[:, :, z0:z0+slab_len]` -> decodes one
64-slice Z-chunk regardless of slab_len, so a 3-slice slab costs the same I/O as
a single slice (depth-awareness for ~free).

Design goals:
  * Fast: precomputed (case_idx, z0) sample table; per-worker LRU of opened zarr
    arrays; cv2/torch single-threaded in workers; pin_memory + prefetch.
  * Deterministic + resumable: fixed sample table + DistributedSampler(seed) +
    set_epoch -> the batch stream b0, b1, ... is reproducible across runs. On
    resume the cycling iterator skips `start_iteration` batches to land at the
    exact position -> eliminates the post-resume SIGReg spike at its source
    (no RNG in __getitem__, so worker RNG is irrelevant).
"""
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# NOTE: `zarr` and `cv2` are imported LAZILY inside _worker_init (per-worker), not
# at module top. Both start background threadpools (numcodecs/Blosc, OpenCV) on
# import; if the main process imports them, DataLoader's default fork inherits
# dead thread state and the workers deadlock. Keeping the main process free of
# these imports (and free of webdataset, which also pulls cv2) makes the default
# fork context safe. The collate is inlined here rather than imported from the
# web_ctrate loader for the same reason.

NUM_REX_CLASSES = 14


def multisource_collate(batch):
    """Match the batch dict that BatchedGuidedDataAugmentationDINO_CT.forward
    consumes (same shape as CTWebDatasetLoader.custom_collate)."""
    out = {
        'volumes_list': [s['raw_volumes'] for s in batch],
        'ts_masks_list': [s['ts_masks'] for s in batch],
        'rex_masks_list': [s['rex_masks'] for s in batch],
        'is_rex_shard': torch.stack([s['is_rex_shard'] for s in batch]),
    }
    if 'dataset_labels' in batch[0]:
        out['dataset_labels'] = torch.stack([s['dataset_labels'] for s in batch])
    return out


class _LRU(OrderedDict):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap

    def __getitem__(self, k):
        v = super().__getitem__(k)
        self.move_to_end(k)
        return v

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if len(self) > self.cap:
            self.popitem(last=False)


class MultisourceZarrDataset(Dataset):
    def __init__(self, cfg):
        d = cfg.dataset
        self.cases_root = d.cases_root                       # pool root (holds cases/<source>/<case>.zarr)
        self.manifest_path = d.manifest_path
        # source selection for the multi-source chest pool:
        #   'all'  -> every train_ssl source (full pool)
        #   list   -> isin(sources)
        #   str    -> exact single source (legacy single-source behaviour)
        # store_paths are built per-case from each row's own `source` column, so
        # 'all' / a list works across the 21-source pool. withhold_sources are
        # license-blocked and dropped (lola11 DUA, covid_ct no license).
        self.sources = getattr(d, 'source', 'all')
        self.withhold_sources = set(getattr(d, 'withhold_sources', ['lola11', 'covid_ct']))
        self.slab_len = int(getattr(d, 'slab_len', 3))       # slices per slab (1 = no slab / single-slice)
        self.stride = int(getattr(d, 'stride', self.slab_len))  # default non-overlapping
        self.mask_size = int(getattr(d, 'mask_size', 128))
        self.norm_type = getattr(d, 'norm_type', 'div1000')
        self.clip_min = float(getattr(d, 'clip_min', -997.0))   # z-score clip bounds (config-driven)
        self.clip_max = float(getattr(d, 'clip_max', 888.0))
        self.cache_cap = int(getattr(d, 'case_cache_cap', 512))
        self.require_organ_mask = bool(getattr(d, 'require_organ_mask', True))
        # On-demand local cache for case zarr files. The pool lives on /project
        # (shared, contended) and zarr v3 sharding means a 3-slice slab reads a
        # full [256,256,128] shard -> ~27x read amplification + /project stalls
        # starve the GPUs (93% idle, 11s/iter). Copying each 25MB case to local
        # node storage on first access makes subsequent reads (the same case is
        # sampled ~50x/epoch) hit fast local disk/RAM. Comma-separated dirs
        # tried in order; first success caches there. Empty/missing -> read
        # directly from /project (legacy slow path).
        _cache_dirs = str(getattr(d, 'local_cache_dirs', '') or '')
        self.cache_dirs = [c.strip() for c in _cache_dirs.split(',') if c.strip()]

        df = pd.read_parquet(self.manifest_path)
        m = df[df.role == 'train_ssl']
        if 'pretrain_eligible' in df.columns:           # guard: some manifests omit eligibility flags
            m = m[m.pretrain_eligible == True]
        if 'qc_pass' in df.columns:
            m = m[m.qc_pass == True]
        if isinstance(self.sources, str) and self.sources == 'all':
            pass                                        # full multi-source pool
        elif isinstance(self.sources, (list, tuple, set)):
            m = m[m.source.isin(list(self.sources))]
        else:
            m = m[m.source == self.sources]             # legacy single-source
        if self.withhold_sources:                       # drop license-blocked sources
            m = m[~m.source.isin(self.withhold_sources)]
        if self.require_organ_mask and 'has_organ_mask' in df.columns:
            m = m[m.has_organ_mask == True]

        self.case_ids = m.case_id.tolist()
        src_list = m.source.tolist()
        self.store_paths = [os.path.join(self.cases_root, 'cases', src, cid + '.zarr')
                            for cid, src in zip(self.case_ids, src_list)]
        n_slices = m.n_slices.to_numpy().astype(np.int64)

        case_idx, z0 = [], []
        for ci, ns in enumerate(n_slices):
            start = 0
            while start + self.slab_len <= ns:
                case_idx.append(ci)
                z0.append(start)
                start += self.stride
        self.sample_case = np.asarray(case_idx, dtype=np.int64)
        self.sample_z0 = np.asarray(z0, dtype=np.int64)

        self._cache = None  # per-worker LRU, created lazily

    def __len__(self):
        return int(self.sample_case.shape[0])

    def _worker_init(self):
        import zarr
        import cv2
        self._zarr = zarr
        self._cv2 = cv2
        self._cache = _LRU(self.cache_cap)
        cv2.setNumThreads(0)
        torch.set_num_threads(1)
        for cd in self.cache_dirs:
            os.makedirs(cd, exist_ok=True)

    def _ensure_local(self, ci):
        """Return the path to read case ci from: the pre-warmed /tmp cache if
        present, else /project directly (served by OS page cache after the
        sbatch read-pass). NO on-demand copytree — 192 concurrent copytrees
        thrash the /project MDT and deadlock the DataLoader (smoke 188937 died
        at iter 23 this way: cold bursts grew 28s->35s->77s->660s hard hang).
        The sbatch pre-warm (tar-pipe to /tmp NVMe) + read-pass (cat to
        /dev/null, warms page cache + dentry cache) populate local storage
        BEFORE training, so workers never storm /project's metadata server.
        A /tmp miss falls back to /project: a zarr slab read is ~4 metadata
        ops (vs copytree's 36) and the data is page-cached, so it is safe."""
        if not self.cache_dirs:
            return self.store_paths[ci]
        cid = self.case_ids[ci]
        for cd in self.cache_dirs:
            lp = os.path.join(cd, cid + '.zarr')
            if os.path.isdir(lp):
                return lp
        return self.store_paths[ci]  # not in /tmp -> /project (page-cached)

    def _get_arrays(self, ci):
        if self._cache is None:
            self._worker_init()
        hit = self._cache.get(ci)
        if hit is not None:
            return hit
        path = self._ensure_local(ci)
        g = self._zarr.open(path, mode='r')
        arrays = (g['image'], g['organ_mask'])
        self._cache[ci] = arrays
        return arrays

    def __getitem__(self, idx):
        ci = int(self.sample_case[idx])
        z0 = int(self.sample_z0[idx])
        z1 = z0 + self.slab_len
        img_arr, msk_arr = self._get_arrays(ci)

        # layout [H, W, Z] -> read slab on the Z axis, transpose to [D, H, W]
        img = np.transpose(img_arr[:, :, z0:z1].astype(np.float32), (2, 0, 1))
        msk = np.transpose(msk_arr[:, :, z0:z1], (2, 0, 1))  # [D, H, W] uint8

        if self.norm_type == 'div1000':
            img = np.clip(img / 1000.0, -1.0, 1.0)
        else:
            img = np.clip(img, self.clip_min, self.clip_max)
            img = (img - self.clip_min) / (self.clip_max - self.clip_min)

        vol = torch.from_numpy(np.ascontiguousarray(img))              # [D, H, W] float32

        D, S = self.slab_len, self.mask_size
        msk_small = np.empty((D, S, S), dtype=np.uint8)
        for z in range(D):
            msk_small[z] = self._cv2.resize(msk[z], (S, S), interpolation=self._cv2.INTER_NEAREST)
        ts = torch.from_numpy(msk_small.astype(np.int64))              # [D, S, S] long
        rex = torch.zeros(D, NUM_REX_CLASSES, S, S, dtype=torch.uint8)

        return {
            'raw_volumes': [vol],
            'ts_masks': [ts],
            'rex_masks': [rex],
            'is_rex_shard': torch.tensor(False, dtype=torch.bool),
        }


class ResumableDistributedSampler(DistributedSampler):
    """DistributedSampler that skips the first `start_iteration` batches on the
    FIRST epoch via an INDEX SLICE (not load+discard), so a resumed run lands at
    the exact stream position of a continuous run WITHOUT loading skipped data.

    The continuous run batches the sampler's full shuffled index list: batch k =
    indices[k*bs:(k+1)*bs]. Slicing indices[start_iteration*bs:] makes the first
    yielded batch == the continuous run's batch `start_iteration` -> identical
    data order (and, with restored RNG, identical training). Subsequent epochs
    yield the full list (no skip). The slice is index-only -> O(1) data work for
    the skip, vs the previous _ResumableCycle which loaded+discarded every
    skipped batch (O(start_iteration) data reads -> hours at production scale:
    a 3500-batch resume-skip at ~9 s/batch cold = ~9 h wasted per job, and the
    skip grows to ~28000 within an epoch -> later jobs deadlock in skip alone)."""

    def __init__(self, *args, start_iteration=0, batch_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_iteration = int(start_iteration)
        self._batch_size = int(batch_size)
        self._first_epoch = True

    def set_resume(self, start_iteration):
        self._start_iteration = int(start_iteration)
        self._first_epoch = True

    def __iter__(self):
        indices = list(super().__iter__())
        if self._first_epoch and self._start_iteration > 0:
            self._first_epoch = False
            skip = self._start_iteration * self._batch_size
            if 0 < skip < len(indices):
                indices = indices[skip:]
        return iter(indices)


class _ResumableCycle:
    """Cycles a DataLoader across epochs (set_epoch per epoch). The per-epoch
    skip on resume is handled by ResumableDistributedSampler (index-slice), so
    this just cycles epochs. The stream b0, b1, ... is reproducible because the
    sample table is fixed and the sampler is seeded."""

    def __init__(self, loader, sampler):
        self.loader = loader
        self.sampler = sampler
        self.epoch = 0

    def set_resume(self, start_iteration):
        self.sampler.set_resume(start_iteration)

    def __iter__(self):
        while True:
            self.sampler.set_epoch(self.epoch)
            for batch in self.loader:
                yield batch
            self.epoch += 1


class CTMultisourceZarrLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = MultisourceZarrDataset(cfg)
        self.batch_size = int(cfg.train.batch_size_per_gpu)
        self.num_workers = int(getattr(cfg.dataset, 'num_workers', 8))
        self.prefetch_factor = int(getattr(cfg.dataset, 'prefetch_factor', 2))
        self.seed = int(getattr(cfg.train, 'seed', 42))

        self.sampler = ResumableDistributedSampler(
            self.dataset, shuffle=True, seed=self.seed, drop_last=True,
            batch_size=self.batch_size)
        # DataLoader timeout (s): if a worker hangs (e.g. blocked on a /project
        # I/O syscall — the failure mode that deadlocked smoke 188937), `next()`
        # raises after this many seconds instead of hanging until the 12 h SLURM
        # wall. The crash lets the afterany chain resume from the last checkpoint
        # (no manual cancel). 0 = no timeout (default). Set ~1800s in production;
        # 188946's worst cold burst was 149 s, so 1800 s gives ~12x margin and
        # matches NCCL_TIMEOUT_S. NOTE: a worker hang drains the prefetch buffer
        # first (~num_workers*prefetch_factor batches), so detection is ~buffer +
        # timeout. Compatible with persistent_workers (PyTorch docs, no warning).
        self.timeout = float(getattr(cfg.dataset, 'dataloader_timeout', 0))
        self.loader = DataLoader(
            self.dataset, batch_size=self.batch_size, sampler=self.sampler,
            num_workers=self.num_workers,
            prefetch_factor=(self.prefetch_factor if self.num_workers > 0 else None),
            pin_memory=False, drop_last=True,
            collate_fn=multisource_collate,
            persistent_workers=(self.num_workers > 0),
            timeout=(self.timeout if self.num_workers > 0 else 0),
        )
        self.train_loader = _ResumableCycle(self.loader, self.sampler)

    def set_resume(self, start_iteration):
        self.train_loader.set_resume(start_iteration)
