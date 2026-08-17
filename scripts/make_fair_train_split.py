#!/usr/bin/env python3
"""Split the 4,942 fair-train volumes into internal train / val for Stage 2 e2e finetune.

Replicates the baseline split methodology in
dataloaders/dataloader_embeddings.py::_create_ctrate_datasets:
    PatientID = VolumeName.split("_")[1]
    np.random.seed(seed)
    val_patient_ids = np.random.choice(unique_patients, size=N, replace=False)
    val_df   = rows whose PatientID in val_patient_ids
    train_df = the rest

The ONE deliberate difference: the baseline draws its internal val from the FULL
CT-RATE train CSV (~47k); here we draw from the 4,942 fair-train (manifest_train.txt).
This kills the prior session's 200-pt -> 992 test leak (ckpt-val drawn from valid_hu
overlapped the 992 by 153/435): ckpt-val now comes from train_hu, not valid_hu.

CAVEAT (verified 2026-07-13): CT-RATE's official train/valid split is by VOLUME,
not patient -- the same patient can have scans in both pools. 245 of the 4,942
fair-train patients also have scans in the 992 fair-valid test (~5% soft
patient-level overlap; a common-mode property of ALL existing DALE-CT CT-RATE
numbers, not a per-experiment bug). PROTOCOL A (see split()): ckpt-VAL is drawn
only from fair-train patients NOT in the 992 test, so checkpoint selection is
patient-disjoint from test (kills the prior-session 200-pt -> 992 ckpt-selection
leak), while TRAIN keeps all 4,942 fair-train volumes (incl. the 245 overlap
patients) for comparability to the baseline / Stage-0 eval protocol. Protocol B
(also exclude the 245 from train; cost 245 vols, 0.3%) is documented in split()
but NOT taken -- it would make Stage 2 a non-comparable "honest-train" metric.

Outputs two files of bare VolumeNames with .nii.gz extension (the form
filter_volume_by_patient_set and the trainer's csv_key lookup both expect):
    fair_train_split_train.txt   (4,942 - N volumes)
    fair_train_split_val.txt     (N volumes)

numpy + stdlib only (runs on the DGX login node, which has numpy but not pandas).
Idempotent: refuses to overwrite existing outputs unless --force.
"""
import os
import argparse
import numpy as np

DEFAULT_MANIFEST = '/project/ibi-staff/CT-JEPA/features/ctrate_fair_subset/manifest_train.txt'
DEFAULT_TEST_MANIFEST = '/project/ibi-staff/CT-JEPA/features/ctrate_fair_subset/manifest_valid.txt'
DEFAULT_OUT_DIR = '/project/ibi-staff/CT-JEPA/features/ctrate_fair_subset'
DEFAULT_VAL_SIZE = 300
DEFAULT_SEED = 42


def load_manifest(path):
    names = [l.strip() for l in open(path) if l.strip()]
    if not names:
        raise SystemExit(f'Empty manifest: {path}')
    return names


def patient_id(name):
    """train_679_a_1 -> '679'. Matches baseline VolumeName.split('_')[1]."""
    parts = name.split('_')
    if len(parts) < 2:
        raise ValueError(f"Cannot parse patient id from '{name}'")
    return parts[1]


def split(names, val_size, seed, test_pids):
    """Patient-based split. 4,942 fair-train = 4,942 unique patients (1 vol each),
    so patient-selection == volume-selection 1:1.

    PROTOCOL A (chosen): ckpt-val is drawn ONLY from fair-train patients NOT in the
    992 test, so checkpoint selection is patient-disjoint from test (kills the
    prior-session 200-pt -> 992 ckpt-selection leak). TRAIN keeps ALL 4,942
    fair-train volumes (including the 245 test-overlap patients) to match the
    baseline / Stage-0 eval protocol -- CT-RATE's volume-level split means ~5% of
    fair-train patients also have scans in the 992 valid test, and ALL existing
    DALE-CT numbers train on these overlap patients. Excluding them only in Stage 2
    would make it a non-comparable "honest-train" metric, not the "fair-train"
    claim. (Switch to Protocol B by also filtering `names` if Evan prefers a fully
    disjoint train; cost = 245 volumes, 0.3% of fair-train.)"""
    pid_to_name = {patient_id(n): n for n in names}
    if len(pid_to_name) != len(names):
        # Should not happen for the fair subset (first-scan-per-patient), but guard.
        raise SystemExit(
            f'Manifest has duplicate patient ids: {len(names)} names but only '
            f'{len(pid_to_name)} unique patients. Fair subset should be 1 vol/patient.')

    overlap_in_train = sorted(p for p in pid_to_name if p in test_pids)  # retained in train
    val_candidates = sorted(p for p in pid_to_name if p not in test_pids)
    val_size = min(val_size, len(val_candidates) - 1)

    # Match the baseline exactly: np.random.seed + np.random.choice (legacy global RNG).
    np.random.seed(seed)
    val_patients = set(np.random.choice(val_candidates, size=val_size, replace=False))
    val_names = sorted(pid_to_name[p] for p in val_patients)
    train_names = sorted(n for n in names if patient_id(n) not in val_patients)
    return train_names, val_names, overlap_in_train


def write_list(path, names):
    with open(path, 'w') as f:
        for n in names:
            f.write(f'{n}.nii.gz\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=DEFAULT_MANIFEST)
    ap.add_argument('--test-manifest', default=DEFAULT_TEST_MANIFEST)
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--val-size', type=int, default=DEFAULT_VAL_SIZE)
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, 'fair_train_split_train.txt')
    val_path = os.path.join(args.out_dir, 'fair_train_split_val.txt')
    for p in (train_path, val_path):
        if os.path.exists(p) and not args.force:
            raise SystemExit(f'{p} exists; pass --force to overwrite')

    names = load_manifest(args.manifest)
    test_names = load_manifest(args.test_manifest)
    test_pids = set(patient_id(n) for n in test_names)
    train_names, val_names, overlap_in_train = split(names, args.val_size, args.seed, test_pids)

    # Sanity (Protocol A): train/val disjoint; ckpt-VAL patient-disjoint from the
    # 992 test (the leak fix); all train_*. Train MAY overlap test (retained
    # 245 overlap patients, matching the baseline eval protocol).
    train_pids = set(patient_id(n) for n in train_names)
    val_pids = set(patient_id(n) for n in val_names)
    assert len(set(train_names) & set(val_names)) == 0, 'train/val volume overlap!'
    assert not (val_pids & test_pids), f'val->test patient leak: {val_pids & test_pids}'
    assert all(n.startswith('train_') for n in train_names + val_names), \
        'fair-train split must contain only train_* volumes (no valid_ test leakage)!'

    write_list(train_path, train_names)
    write_list(val_path, val_names)
    print(f'manifest: {len(names)}  test: {len(test_names)}  seed: {args.seed}  val_size: {args.val_size}', flush=True)
    print(f'  test-overlap patients RETAINED in train (Protocol A): {len(overlap_in_train)}', flush=True)
    print(f'  train -> {train_path} ({len(train_names)} vols, {len(train_pids)} patients; '
          f'{len(train_pids & test_pids)} also in test [Protocol A])', flush=True)
    print(f'  val   -> {val_path} ({len(val_names)} vols, {len(val_pids)} patients; 0 in test)', flush=True)
    print(f'  train/val disjoint: True  val&test pids: 0 (ckpt-select clean)  all train_*: True', flush=True)


if __name__ == '__main__':
    main()
