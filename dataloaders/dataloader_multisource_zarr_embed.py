"""
Embedding-only zarr dataset for the multi-source chest-CT pool.

Reads per-case zarr v3 groups (member: image [H,W,Z] int16) listed in a prep CSV
(scan_label, zarr_path, ...) and yields raw-HU (D,H,W) volumes for the multimethod
embedder (scripts/ctrate_generate_embeddings_multimethod.py). Returns the same
(volume, label, filename) contract as CTMultiScaleDataset so run_multi_method_inference
unpacks unchanged.

Critical correctness points (see dataloaders/datasetloader_multisource_zarr.py):
  * RAW HU, no normalization. The embedder's MultiMethodProcessor.normalize()
    div1000 branch applies clamp(HU/1000, -1, 1) once; normalizing here too would
    double-normalize and silently corrupt the embeddings. Backbone must see [-1,1]
    exactly as in Z1 training.
  * `zarr` is imported LAZILY inside __getitem__ (per-worker), not at module top.
    zarr starts a Blosc threadpool on import; a module-top import in the main
    process is inherited by forked DataLoader workers as dead thread state and
    deadlocks them. The embedder uses the default fork context with num_workers=8.
  * torch.set_num_threads(1) per worker to avoid oversubscription.
  * np.ascontiguousarray before torch.from_numpy (transpose yields a non-contiguous
    view; roi_align in the processor wants contiguous).
  * HOST->container path translation: the prep CSV stores HOST zarr paths
    (/project/...); the embedder runs inside the Pyxis container (mount
    /project:/app/project) where /project does not exist. _to_container_path
    swaps the prefix to /app/project/... so zarr.open resolves.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _to_container_path(p):
    """Map a HOST path (/project/...) to its in-container form (/app/project/...).

    The upstream prep script stores HOST paths in the CSV (verified with
    os.path.isdir on the login node, where /project exists). The embedder runs
    inside the Pyxis container (mount /project:/app/project), where /project
    does not exist but /app/project does. If the path already resolves as-is
    (run on the host, or a different mount), leave it untouched.
    """
    if p.startswith("/project/") and not os.path.exists(p):
        ap = "/app" + p  # /project/... -> /app/project/...
        if os.path.exists(ap):
            return ap
    return p


class MultisourceZarrEmbedDataset(Dataset):
    """Yields (volume[D,H,W] float32 raw-HU, dummy_label, scan_label_str).

    The dummy label is never consumed by the embedder (it uses volumes + filenames
    only) but is kept collatable to match the (volume, label, filename) contract.
    base_name == scan_label (dot-free) so output is <out>/<method>/<scan_label>.npz,
    which the probe's CTScanMILDataset(id_col='scan_label', file_ext='') reads back.
    """

    def __init__(self, case_csv):
        if not os.path.exists(case_csv):
            raise FileNotFoundError(f"case_csv not found: {case_csv}")
        self.df = pd.read_csv(case_csv)
        # sanity: required columns
        for col in ("scan_label", "zarr_path"):
            if col not in self.df.columns:
                raise ValueError(f"case_csv missing required column '{col}': {case_csv}")
        self.scan_labels = self.df["scan_label"].astype(str).tolist()
        # Resolve HOST -> container paths once (the embedder runs inside the
        # Pyxis container; see _to_container_path).
        self.zarr_paths = [_to_container_path(p)
                           for p in self.df["zarr_path"].astype(str).tolist()]
        self._zarr = None  # lazily imported per-worker
        self._initialized = False

    def __len__(self):
        return len(self.df)

    def _worker_init(self):
        import zarr  # lazy: fork-safe (see module docstring)
        self._zarr = zarr
        torch.set_num_threads(1)
        self._initialized = True

    def __getitem__(self, idx):
        if not self._initialized:
            self._worker_init()
        zarr_path = self.zarr_paths[idx]
        scan_label = self.scan_labels[idx]
        g = self._zarr.open(zarr_path, mode="r")
        # layout [H, W, Z] int16 -> (D, H, W) float32, raw HU
        img = np.transpose(g["image"][:].astype(np.float32), (2, 0, 1))
        vol = torch.from_numpy(np.ascontiguousarray(img))
        dummy_label = np.zeros(1, dtype=np.float32)
        return vol, dummy_label, scan_label


class MultisourceZarr3DDataset(Dataset):
    """Yields (volume[n_slab,H,W] float32 RAW-HU, dummy_label, scan_label_str)
    for the 3D benchmark FMs (ct_clip / ct_fm / merlin / colipri).

    The requested slab is sliced at READ time from the V3/native z1 store (image
    [H,W,Z] int16 true-HU, slice axis = last), so the 3D FM sees only the heart
    sub-volume (strategy C, deterministic organ-mask slab). This is MANDATORY for
    3D FMs: they emit spatial token grids whose axes no longer correspond to
    input z after their internal resize/permute/foreground-crop, so the slab
    cannot be applied post-embedding the way prep_slab_embeddings.py z-slices 2D
    per-slice CLS. Per-FM normalization is applied inside extract_volume_features
    -- DO NOT normalize here (would double-normalize). Same (volume, label,
    filename) contract as MultisourceZarrEmbedDataset / CTMultiScaleDataset so the
    3D embedder's run() unpacks unchanged.

    case_csv columns: scan_label, zarr_path (HOST /project path), slab_z0, slab_z1.
    `zarr` is imported lazily per-worker (fork-safe; see module docstring).
    """

    def __init__(self, case_csv, slab_z0_col="slab_z0", slab_z1_col="slab_z1"):
        if not os.path.exists(case_csv):
            raise FileNotFoundError(f"case_csv not found: {case_csv}")
        self.df = pd.read_csv(case_csv)
        for col in ("scan_label", "zarr_path", slab_z0_col, slab_z1_col):
            if col not in self.df.columns:
                raise ValueError(f"case_csv missing required column '{col}': {case_csv}")
        self.scan_labels = self.df["scan_label"].astype(str).tolist()
        self.zarr_paths = [_to_container_path(p)
                           for p in self.df["zarr_path"].astype(str).tolist()]
        self.slab_z0 = self.df[slab_z0_col].astype(int).tolist()
        self.slab_z1 = self.df[slab_z1_col].astype(int).tolist()
        self._zarr = None  # lazily imported per-worker
        self._initialized = False

    def __len__(self):
        return len(self.df)

    def _worker_init(self):
        import zarr  # lazy: fork-safe (see module docstring)
        self._zarr = zarr
        torch.set_num_threads(1)
        self._initialized = True

    def __getitem__(self, idx):
        if not self._initialized:
            self._worker_init()
        z0, z1 = self.slab_z0[idx], self.slab_z1[idx]
        g = self._zarr.open(self.zarr_paths[idx], mode="r")
        # image [H,W,Z] int16 true-HU -> slab on Z (last axis) -> (D=n_slab, H, W) float32
        img = np.transpose(g["image"][:, :, z0:z1 + 1].astype(np.float32), (2, 0, 1))
        vol = torch.from_numpy(np.ascontiguousarray(img))  # (n_slab, 512, 512) raw HU
        dummy_label = np.zeros(1, dtype=np.float32)
        return vol, dummy_label, self.scan_labels[idx]
