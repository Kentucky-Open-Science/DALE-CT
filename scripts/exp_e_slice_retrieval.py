"""Exp E — cross-patient anatomical slice retrieval (world model "in use").

Task: for every slice of a query scan A, retrieve the nearest-neighbor slice
(cosine) in every other test scan B. Score whether the retrieved slice matches
the query's true per-slice content:

  ts_cos  cosine between the [118] TotalSegmentator fractional-coverage vectors
  ts_iou  soft IoU  sum(min)/sum(max)  of the same vectors
  rex_hit for query slices with >=1 positive ReX finding: fraction of the
          query's positive findings also positive in the retrieved slice

Conditions per arm (uniform across arms — no per-arm preprocessing choices):
  raw     retrieval on L2-normalized CLS embeddings as-is
  center  per-scan mean subtracted first (removes patient-identity offset;
          diagnostic 2026-08-17: halves 0-L z-err, no effect on ViT-B arms)
  resid   z-direction (fit on 281-scan train split, exp_c recipe) projected
          out — P4 link; run at frac 0 only (shown ~= raw under truncation)

Label-only baselines (arm-independent):
  norm_z  match by normalized slice index (position-only heuristic)
  random  seeded uniform gallery slice (prevalence floor for all metrics)

Optional --truncate-fracs simulates FOV mismatch: drops that fraction of each
GALLERY scan's slices from one (seeded-random) end before retrieval; norm_z
re-normalizes over the truncated span, i.e. it does not know slices are gone.

Operating set: 414 ReX-annotated scans, 70/10/20 patient split (same as the
semantic probes P4/P5); retrieval pairs are over the 86-scan test split; the
z-direction is fit on train. CI: 2000-resample bootstrap over query scans
(per-scan sufficient stats), matching the paper's protocol. GPU-free.

Outputs: outputs/exp_e_retrieval/{arm}_{cond}[ _t{frac} ].json + summary.txt
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_c_zposition import ARM_DIRS, load_embeddings, boot_ci  # noqa: E402
from exp_c_worldmodel_probes import (  # noqa: E402
    GT_ROOT,
    load_split_414,
    load_ts_slices,
    fit_z_direction,
    project_out_z,
)

REX_DIR = os.path.join(GT_ROOT, "rex")  # {vol}.npz: cls_labels[D,14] float32
OUT_DIR = "/project/ibi-staff/CT-JEPA/public/outputs/exp_e_retrieval"
ARMS = ["depth_aware_25d", "pure_2d", "dale_ct_0_chest"]
SEED = 42


def load_rex_slices(vols):
    """{vol: [D,14] float32} for vols whose rex npz exists."""
    out = {}
    for v in vols:
        p = os.path.join(REX_DIR, v + ".npz")
        if os.path.exists(p):
            out[v] = np.load(p)["cls_labels"].astype(np.float32)
    return out


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def truncate_idx(D, frac, rng):
    """Indices kept after dropping frac*D slices from a random end."""
    k = int(round(frac * D))
    if k <= 0 or k >= D:
        return np.arange(D)
    return np.arange(k, D) if rng.random() < 0.5 else np.arange(0, D - k)


def score_pairs(q_ts, r_ts, q_rex, r_rex):
    """Per-query-slice scores; NaN where undefined (zero vec / no positive)."""
    n = q_ts.shape[0]
    ts_cos = np.full(n, np.nan)
    ts_iou = np.full(n, np.nan)
    rex_hit = np.full(n, np.nan)
    qn = np.linalg.norm(q_ts, axis=1)
    rn = np.linalg.norm(r_ts, axis=1)
    ok = (qn > 0) & (rn > 0)
    ts_cos[ok] = (q_ts[ok] * r_ts[ok]).sum(1) / (qn[ok] * rn[ok])
    mx = np.maximum(q_ts, r_ts).sum(1)
    ok2 = mx > 0
    ts_iou[ok2] = np.minimum(q_ts, r_ts)[ok2].sum(1) / mx[ok2]
    qb, rb = q_rex > 0, r_rex > 0
    pos = qb.any(1)
    if pos.any():
        rex_hit[pos] = (qb & rb)[pos].sum(1) / qb[pos].sum(1)
    return ts_cos, ts_iou, rex_hit


def run_retrieval(embs, test_vols, ts, rex, mode, trunc_frac=0.0, seed=SEED):
    """mode: 'emb' (NN on embs), 'norm_z', or 'random'.

    Returns {metric: [per-query-scan mean]} over test_vols as query scans.
    """
    rng = np.random.default_rng(seed)
    # pre-draw gallery truncations once so every query scan sees the same gallery
    kept = {v: truncate_idx(ts[v].shape[0], trunc_frac, rng) for v in test_vols}
    per_scan = {"ts_cos": [], "ts_iou": [], "rex_hit": []}
    for A in test_vols:
        acc = {k: [] for k in per_scan}
        DA = ts[A].shape[0]
        for B in test_vols:
            if B == A:
                continue
            idx = kept[B]
            DB = len(idx)
            if mode == "emb":
                nn = idx[np.argmax(embs[A] @ embs[B][idx].T, axis=1)]
            elif mode == "norm_z":
                # position heuristic over the (possibly truncated) gallery span
                nn = idx[np.round(np.linspace(0, 1, DA) * (DB - 1)).astype(int)]
            else:  # random
                nn = idx[rng.integers(0, DB, size=DA)]
            c, i, h = score_pairs(ts[A], ts[B][nn], rex[A], rex[B][nn])
            acc["ts_cos"].append(c)
            acc["ts_iou"].append(i)
            acc["rex_hit"].append(h)
        for k in per_scan:
            v = np.concatenate(acc[k])
            per_scan[k].append(float(np.nanmean(v)) if np.isfinite(v).any() else np.nan)
    return per_scan


def agg(per_scan):
    out = {}
    for k, vals in per_scan.items():
        v = [x for x in vals if np.isfinite(x)]
        out[k] = {"mean": float(np.mean(v)), "ci": boot_ci(v), "n_scans": len(v)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--truncate-fracs", default="",
                    help="e.g. '0.2,0.4' — extra gallery-truncation conditions")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--smoke", action="store_true", help="10 test scans only")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    split = load_split_414()
    train_vols, test_vols = split["train"], split["test"]
    all_vols = train_vols + split["val"] + test_vols
    ts = load_ts_slices(all_vols)
    rex = load_rex_slices(all_vols)
    fracs = [0.0] + [float(x) for x in args.truncate_fracs.split(",") if x.strip()]

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    results, valid_test = {}, None
    arm_data = {}
    for arm in arms:
        scans = load_embeddings(arm, all_vols)
        vt = [v for v in test_vols
              if v in scans and v in ts and v in rex
              and scans[v].shape[0] == ts[v].shape[0] == rex[v].shape[0]]
        valid_test = vt if valid_test is None else [v for v in valid_test if v in vt]
        arm_data[arm] = scans
    tr = [v for v in train_vols
          if all(v in arm_data[a] for a in arms)
          and v in ts and all(arm_data[a][v].shape[0] == ts[v].shape[0] for a in arms)]
    print(f"valid test scans: {len(valid_test)}  |  z-fit train scans: {len(tr)}")
    if args.smoke:
        valid_test = valid_test[:10]

    for arm in arms:
        scans = arm_data[arm]
        w, _ = fit_z_direction(scans, tr)
        for cond in ["raw", "center", "resid"]:
            if cond == "raw":
                emb_of = lambda e: e
            elif cond == "center":
                emb_of = lambda e: e - e.mean(0)
            else:
                emb_of = lambda e: project_out_z(e, w)
            embs = {v: unit(emb_of(scans[v])) for v in valid_test}
            for f in fracs:
                if cond == "resid" and f:
                    continue
                tag = f"{arm}_{cond}" + (f"_t{f}" if f else "")
                per_scan = run_retrieval(embs, valid_test, ts, rex, "emb", trunc_frac=f)
                results[tag] = agg(per_scan)
                json.dump({"per_scan": per_scan, "agg": results[tag]},
                          open(os.path.join(args.out, tag + ".json"), "w"), indent=1)
                print(tag, {k: round(v["mean"], 4) for k, v in results[tag].items()})

    for mode in ["norm_z", "random"]:
        for f in fracs:
            tag = mode + (f"_t{f}" if f else "")
            per_scan = run_retrieval(None, valid_test, ts, rex, mode, trunc_frac=f)
            results[tag] = agg(per_scan)
            json.dump({"per_scan": per_scan, "agg": results[tag]},
                      open(os.path.join(args.out, tag + ".json"), "w"), indent=1)
            print(tag, {k: round(v["mean"], 4) for k, v in results[tag].items()})

    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write(f"n test scans: {len(valid_test)}\n")
        fh.write(f"{'condition':38s} {'ts_cos':>22s} {'ts_iou':>22s} {'rex_hit':>22s}\n")
        for tag, r in results.items():
            row = " ".join(
                f"{r[k]['mean']:.4f} [{r[k]['ci'][0]:.4f},{r[k]['ci'][1]:.4f}]"
                for k in ["ts_cos", "ts_iou", "rex_hit"])
            fh.write(f"{tag:38s} {row}\n")
    json.dump(results, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
