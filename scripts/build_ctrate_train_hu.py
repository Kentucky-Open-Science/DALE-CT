#!/usr/bin/env python3
"""Build CT-RATE_train_hu/*.npy for the 4,942 fair-train volumes.

Replicates Process-CT-Data/1_preprocessing/preprocess_val.py EXACTLY (the script
that built CT-RATE_valid_hu), so the train .npy are byte-for-format-compatible
with the valid .npy the e2e trainer/evaluator already consumes:

    raw = nib.load(nifti).get_fdata(dtype=float32)        # (H,W,Z); header scl_* are nan
    hu  = (raw * RescaleSlope) + RescaleIntercept          # per-volume, from train_metadata.csv
    hu  = hu.transpose(2, 0, 1).astype(float16)            # (Z,H,W), float16
    np.save(out, hu)                                       # NO clip, NO resample, NO reorient

Only the 4,942 manifest_train.txt volumes are converted (not all 47,149). The
per-volume RescaleIntercept is NOT uniform (-1024 for 74%, -8192 for 26%), so
the metadata CSV is mandatory - hardcoding -1024 would corrupt 1,300 volumes.

Idempotent: skips existing outputs (safe to re-run / resume). Multiprocessing.

Self-test: before the bulk run, converts 2 sentinel volumes (one inter=-1024,
one inter=-8192), logs raw+HU ranges, asserts finite/float16/3D, and aborts
otherwise. Run with --self-test-only to just inspect.

Runs IN-CONTAINER: paths are /app/project/... (mount is /project:/app/project).
"""
import os
import sys
import csv
import argparse
import multiprocessing as mp

import numpy as np
import nibabel as nib
from tqdm import tqdm

DEFAULT_MANIFEST = '/app/project/ibi-staff/CT-JEPA/features/ctrate_fair_subset/manifest_train.txt'
DEFAULT_META     = '/app/data/CT-RATE/dataset/metadata/train_metadata.csv'
DEFAULT_RAW_ROOT = '/app/data/CT-RATE/dataset/train_valid/train'
DEFAULT_OUT_DIR  = '/app/project/ibi-staff/CT-JEPA/Process_CT-RATE/dataset/CT-RATE_train_hu'

# Sentinels for the self-test (one of each intercept, confirmed present in manifest).
SENTINELS = ['train_10008_a_1', 'train_10001_a_1']  # inter=-1024, inter=-8192


def load_metadata(csv_path):
    meta = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            meta[row['VolumeName']] = (float(row['RescaleSlope']),
                                       float(row['RescaleIntercept']))
    return meta


def load_manifest(path):
    return [l.strip() for l in open(path) if l.strip()]


def raw_path_for(name, raw_root):
    parts = name.split('_')  # train, pid, scan, recon
    pid, scan = parts[1], parts[2]
    return os.path.join(raw_root, f'train_{pid}', f'train_{pid}_{scan}', f'{name}.nii.gz')


def convert_one(nifti_path, name, slope, inter, out_path):
    """Exact preprocess_val.py transform. Returns (out_path, raw_shape, hu_min, hu_max, hu_mean)."""
    img = nib.load(nifti_path)
    raw = img.get_fdata(dtype=np.float32)
    hu = (raw * slope) + inter
    hu = hu.transpose(2, 0, 1).astype(np.float16)
    np.save(out_path, hu)
    return out_path, raw.shape, float(hu.min()), float(hu.max()), float(hu.mean())


def _worker(args):
    nifti_path, name, slope, inter, out_path = args
    if os.path.exists(out_path):
        return ('skip', name, None)
    try:
        convert_one(nifti_path, name, slope, inter, out_path)
        return ('ok', name, None)
    except Exception as e:
        return ('err', name, repr(e))


def self_test(meta, raw_root, out_dir):
    print('=' * 70, flush=True)
    print('SELF-TEST: 2 sentinel volumes (one inter=-1024, one inter=-8192)', flush=True)
    print('=' * 70, flush=True)
    ok = True
    for name in SENTINELS:
        key = f'{name}.nii.gz'
        if key not in meta:
            print(f'  {name}: MISSING from metadata', flush=True); ok = False; continue
        slope, inter = meta[key]
        nifti_path = raw_path_for(name, raw_root)
        if not os.path.exists(nifti_path):
            print(f'  {name}: raw NIfTI MISSING at {nifti_path}', flush=True); ok = False; continue
        out_path = os.path.join(out_dir, f'{name}.npy')
        _, raw_shape, hu_min, hu_max, hu_mean = convert_one(nifti_path, name, slope, inter, out_path)
        a = np.load(out_path, mmap_mode='r')
        finite = bool(np.isfinite(a.astype(np.float32)).all())
        good = (a.dtype == np.float16) and (a.ndim == 3) and finite
        ok = ok and good
        print(f'  {name}: slope={slope} inter={inter}', flush=True)
        print(f'    raw_shape={raw_shape} -> npy shape={a.shape} dtype={a.dtype} '
              f'ndim={a.ndim} finite={finite}', flush=True)
        print(f'    HU min={hu_min:.1f} max={hu_max:.1f} mean={hu_mean:.1f}', flush=True)
        print(f'    -> {"OK" if good else "FAIL"}', flush=True)
    print('=' * 70, flush=True)
    if not ok:
        print('SELF-TEST FAILED - aborting before bulk run.', flush=True)
        sys.exit(1)
    print('SELF-TEST PASSED - proceeding to bulk conversion.', flush=True)
    print('=' * 70, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=DEFAULT_MANIFEST)
    ap.add_argument('--metadata', default=DEFAULT_META)
    ap.add_argument('--raw-root', default=DEFAULT_RAW_ROOT)
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--num-workers', type=int, default=16)
    ap.add_argument('--self-test-only', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'manifest : {args.manifest}', flush=True)
    print(f'metadata : {args.metadata}', flush=True)
    print(f'raw_root : {args.raw_root}', flush=True)
    print(f'out_dir  : {args.out_dir}', flush=True)
    print(f'workers  : {args.num_workers}', flush=True)

    meta = load_metadata(args.metadata)
    names = load_manifest(args.manifest)
    print(f'metadata rows: {len(meta)}  manifest entries: {len(names)}', flush=True)

    self_test(meta, args.raw_root, args.out_dir)
    if args.self_test_only:
        print('--self-test-only: stopping after self-test.', flush=True)
        return

    # Build task list: manifest volumes with existing raw NIfTI + metadata, skip
    # outputs already present (idempotent resume).
    tasks, missing_meta, missing_raw, already = [], 0, 0, 0
    for name in names:
        key = f'{name}.nii.gz'
        if key not in meta:
            missing_meta += 1
            continue
        nifti_path = raw_path_for(name, args.raw_root)
        if not os.path.exists(nifti_path):
            missing_raw += 1
            continue
        out_path = os.path.join(args.out_dir, f'{name}.npy')
        if os.path.exists(out_path):
            already += 1
            continue
        slope, inter = meta[key]
        tasks.append((nifti_path, name, slope, inter, out_path))

    print(f'to-convert={len(tasks)} already-present={already} '
          f'missing-meta={missing_meta} missing-raw={missing_raw}', flush=True)
    if missing_meta or missing_raw:
        print('ERROR: manifest has missing inputs - aborting.', flush=True)
        sys.exit(1)

    n_ok = n_err = 0
    with mp.Pool(args.num_workers) as pool:
        for status, name, err in tqdm(pool.imap_unordered(_worker, tasks),
                                      total=len(tasks), desc='converting'):
            if status == 'ok':
                n_ok += 1
            elif status == 'skip':
                pass
            else:
                n_err += 1
                print(f'  ERROR {name}: {err}', flush=True)

    total = len([f for f in os.listdir(args.out_dir) if f.endswith('.npy')])
    print(f'DONE. converted_this_run={n_ok} errors={n_err} '
          f'total_npy_in_dir={total}', flush=True)
    if n_err:
        sys.exit(1)


if __name__ == '__main__':
    main()
