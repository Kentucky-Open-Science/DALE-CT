#!/usr/bin/env python
"""
Add a `patch_labels_{to_key}` entry to every model-comparison ground-truth .npz
by copying an existing `patch_labels_{from_key}` entry that shares the same
patch grid.

Used when a new backbone is added to the model-comparison study with the same
patch_size as an already-extracted model: the per-patch labels are a function
of (mask, grid_size) only, so two grid-identical models have byte-identical
patch labels. This avoids re-running the slow NIfTI mask extraction in
extract_model_comparison_groundtruth.py (which skips existing files on resume).

Atomic per-file rewrite (temp file + os.replace). Skips files that already
contain the target key (idempotent / resume-safe).

Usage:
    python scripts/add_model_comparison_gt_key.py \
        --gt-root /app/project/ibi-staff/CT-JEPA/public/outputs/model_comparison_groundtruth \
        --from-key lejepa_2s --to-key lejepa_1s_v2 --grid 16
"""
import argparse
import glob
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-root", required=True, help="model_comparison_groundtruth/ root")
    ap.add_argument("--from-key", required=True, help="source model key, e.g. lejepa_2s")
    ap.add_argument("--to-key", required=True, help="target model key, e.g. lejepa_1s_v2")
    ap.add_argument("--grid", type=int, required=True, help="expected patch grid (e.g. 16)")
    ap.add_argument("--tasks", nargs="+", default=["rex", "totalseg"], help="GT subdirs to process")
    args = ap.parse_args()

    from_key = f"patch_labels_{args.from_key}"
    to_key = f"patch_labels_{args.to_key}"

    rewritten = 0
    skipped_existing = 0
    missing_from = 0

    for task in args.tasks:
        gd = os.path.join(args.gt_root, task)
        files = sorted(glob.glob(os.path.join(gd, "*.npz")))
        if not files:
            print(f"[{task}] no .npz in {gd}", file=sys.stderr)
            continue
        for f in files:
            z = np.load(f, allow_pickle=False)
            keys = list(z.keys())
            if to_key in keys:
                z.close()
                skipped_existing += 1
                continue
            if from_key not in keys:
                z.close()
                missing_from += 1
                print(f"[{task}] {os.path.basename(f)}: missing {from_key}", file=sys.stderr)
                continue
            src = z[from_key]
            if tuple(src.shape[-2:]) != (args.grid, args.grid):
                z.close()
                raise SystemExit(
                    f"[{task}] {os.path.basename(f)}: {from_key} grid {src.shape[-2:]} "
                    f"!= expected ({args.grid},{args.grid})"
                )
            out = {k: z[k] for k in keys}
            out[to_key] = src
            z.close()
            # Write via a file object: np.savez_compressed appends ".npz" to a
            # string path that lacks it, which would name the temp file
            # "...npz.tmp.npz" and break os.replace below.
            tmp = f + ".tmp"
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **out)
            os.replace(tmp, f)
            rewritten += 1
        print(f"[{task}] processed {len(files)} files")

    print(f"\nDone. rewrote={rewritten} skipped_existing={skipped_existing} missing_from={missing_from}")


if __name__ == "__main__":
    main()
