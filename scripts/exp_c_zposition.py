#!/usr/bin/env python3
"""
Exp C — Volumetric localization from frozen slice embeddings.

Compares the depth-aware (2.5D 12 mm slab) vs pure-2D ViT-Base LeJEPA encoders
on the CT-RATE valid set (3002 scans). Both encoders are identical except for
slab_size=12.0 vs single_slice=true (clean Table-V ablation arms); per-scan CLS
embeddings (shape [D, 768], D = #slices) are pre-extracted on the DGX.

Two tasks, each run for both arms with 2000-resample bootstrap CIs (scans
resampled; matches the ctrate error-bars protocol):

  1. Z-position regression (supervised linear probe).  Ridge (CV alpha) maps
     a slice's CLS embedding -> normalized anatomical z in [0,1] (0 = first
     stored slice, 1 = last).  Patient-level 80/20 split (seed 42).  Per test
     scan: MAE (normalized), MAE (slices = norm*D), R^2, Spearman rho.
     Aggregate = mean over test scans + bootstrap CI.

  2. Slice-ordering recovery (UNSUPERVISED — no z label).  Given the bag of D
     slice embeddings for a scan, recover anatomical order by (a) PCA-1D
     projection sort and (b) nearest-neighbour + 2-opt shortest Hamiltonian
     path on cosine distance.  Metric: |Kendall tau| vs true stored order
     (path is flip-symmetric, so we take the max of forward/reversed).

Clinical "so what": an encoder that encodes z-position (depth-aware) enables
prior-exam co-registration by anatomical position, coverage/protocol QA, and
volumetric lesion localization.  A pure-2D slice encoder trained on random
slices has no objective reason to learn a smooth anatomical latent trajectory.

Outputs:  outputs/exp_c_zposition/{arm}_{zreg,ordering}.json + summary.{json,txt}
"""

import argparse
import json
import os
import sys
import csv
from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau, spearmanr
from scipy.spatial.distance import cdist
from sklearn.linear_model import RidgeCV

ARM_DIRS = {
    "depth_aware_25d": "/project/ibi-staff/CT-JEPA/features/ctrate/lejepa_base_pretrained/valid_cls",
    "pure_2d":         "/project/ibi-staff/CT-JEPA/features/ctrate/lejepa_base_pretrained_2d/valid_cls",
    # Released from-scratch DALE-CT-2S (ViT-Large, patch16, 50k, TS+ReX aux) — per-slice
    # CLS over the valid set. Closes the §3F gap: world-model evidence on a submitted
    # model, not only the DINOv2-init ViT-Base ablation. Embeddings at
    # features/ctrate/lejepa_v2/ (generated from Guided_Chest_CT_LeJEPA_V2/iter_50000,
    # which has rex_ratio:0.2 -> the 2S run). Identity-verified 2026-07-10: lejepa_v2
    # CLS cosine 0.999 to HF-released DALE-CT-2S, 0.03 (orthogonal) to DALE-CT-0.
    "released_dale_ct_2s": "/project/ibi-staff/CT-JEPA/features/ctrate/lejepa_v2/valid_cls",
    # DALE-CT-0-L ("chest") — variant-0 pure SSL on the full ~296k multi-source
    # pool (LeJEPA_0_chest_z1full, iter_191357, no aux head, ViT-L/patch16).
    # Per-slice CLS over the valid set, identical slice sampling to the arms
    # above. Dir is valid/ (not valid_cls/) — the chest embedder wrote there;
    # load_embeddings looks up {stem}.npy by name so the sibling subdirs
    # (crop_visualizations/, embeddings/) are never traversed. Scale-contrast
    # arm: 296k pool. NOTE: outputs/exp_c_zposition_released0/ is MISLABELED
    # 2S data (generated from lejepa_v2 CLS = cosine 0.999 to DALE-CT-2S,
    # 0.03 to DALE-CT-0, identity-checked 2026-07-10). No released DALE-CT-0
    # full-valid CLS exists, so there is NO released-0 z-reg arm — do not cite
    # released0/ as DALE-CT-0.
    "dale_ct_0_chest": "/project/ibi-staff/CT-JEPA/features/ctrate/lejepa_0_chest_cropped_global/valid",
}
VAL_CSV = "/project/ibi-staff/CT-JEPA/Process_CT-RATE/dataset/valid_predicted_labels.csv"
MIN_SLICES = 16           # match continuity config
SEED = 42
N_BOOT = 2000
CI_PCT = [2.5, 97.5]


def patient_id(volume_name):
    # valid_<pid>_<letter>_<scan>.nii.gz  -> pid
    parts = os.path.basename(volume_name).replace(".nii.gz", "").split("_")
    return parts[1] if len(parts) >= 2 else volume_name


def load_scan_list():
    vols = []
    with open(VAL_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            vols.append(row["VolumeName"])
    return vols


def load_embeddings(arm, vols, max_scans=None):
    d = ARM_DIRS[arm]
    scans = {}  # vol -> np.ndarray [D,768]
    for v in vols:
        if max_scans and len(scans) >= max_scans:
            break
        stem = v.replace(".nii.gz", "")
        p = os.path.join(d, stem + ".npy")
        if not os.path.exists(p):
            continue
        a = np.load(p)
        if a.ndim == 3:           # [D, views, dim] -> [D, dim] (defensive)
            a = a[:, 0, :]
        if a.ndim != 2 or a.shape[0] < MIN_SLICES:
            continue
        scans[v] = a.astype(np.float32)
    return scans


def patient_split(scans, frac=0.2, seed=SEED):
    by_pid = defaultdict(list)
    for v in scans:
        by_pid[patient_id(v)].append(v)
    pids = sorted(by_pid)
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n_test = int(round(len(pids) * frac))
    test_pids = set(pids[:n_test])
    train, test = [], []
    for pid in pids:
        (test if pid in test_pids else train).extend(by_pid[pid])
    return train, test


# ---------------- z-position regression ----------------

def zreg(arm, scans, train_vols, test_vols):
    Xtr, ytr = [], []
    for v in train_vols:
        e = scans[v]
        D = e.shape[0]
        z = np.linspace(0.0, 1.0, D, dtype=np.float32)
        Xtr.append(e); ytr.append(z)
    Xtr = np.concatenate(Xtr, 0); ytr = np.concatenate(ytr, 0)
    model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xtr, ytr)

    per_scan = []
    for v in test_vols:
        e = scans[v]; D = e.shape[0]
        z_true = np.linspace(0.0, 1.0, D, dtype=np.float32)
        z_pred = model.predict(e)
        mae_norm = float(np.mean(np.abs(z_pred - z_true)))
        mae_slc = mae_norm * D
        ss_res = float(np.sum((z_pred - z_true) ** 2))
        ss_tot = float(np.sum((z_true - z_true.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        rho = float(spearmanr(z_pred, z_true).statistic)
        per_scan.append({"vol": v, "D": int(D), "mae_norm": mae_norm,
                         "mae_slices": mae_slc, "r2": r2, "rho": rho})
    return {"alpha": float(model.alpha_), "n_train_slices": int(len(ytr)),
            "n_test_scans": len(test_vols), "per_scan": per_scan}


# ---------------- slice-ordering recovery (unsupervised) ----------------

def _abs_kendall(recovered, true_order):
    tau = kendalltau(recovered, true_order).statistic
    if np.isnan(tau):
        return 0.0
    return float(abs(tau))           # flip-symmetric (path has no orientation)


def pca1d_order(e):
    # project onto 1st PC of the per-scan slice embeddings, sort. If the
    # trajectory is a smooth 1D curve, PC1 ~= the trajectory direction and the
    # sort recovers anatomical order (up to reversal, handled by |tau|).
    e0 = e - e.mean(0, keepdims=True)
    _, _, vh = np.linalg.svd(e0, full_matrices=False)
    pc1 = e0 @ vh[0]
    return np.argsort(pc1)


def greedy_nn_order(e, dmat):
    # greedy nearest-neighbour tour, started from the slice closest to the
    # centroid (position-agnostic start, no bias toward true slice 0).
    D = e.shape[0]
    cent = e.mean(0)
    start = int(np.argmin(np.linalg.norm(e - cent, axis=1)))
    visited = np.zeros(D, dtype=bool)
    tour = [start]; visited[start] = True; cur = start
    for _ in range(D - 1):
        d = dmat[cur].copy(); d[visited] = np.inf
        nxt = int(np.argmin(d)); tour.append(nxt); visited[nxt] = True; cur = nxt
    return np.array(tour)


def path_len(order, dmat):
    return float(np.sum(dmat[order[:-1], order[1:]]))


def ordering(arm, scans, test_vols, max_scans=None, seed=SEED):
    per_scan = []
    vols = list(test_vols)
    if max_scans and len(vols) > max_scans:
        rng = np.random.default_rng(seed)
        vols = [vols[i] for i in rng.choice(len(vols), max_scans, replace=False)]
    for vi, v in enumerate(vols):
        e = scans[v]; D = e.shape[0]
        dmat = cdist(e, e, metric="cosine"); np.fill_diagonal(dmat, 0.0)
        to = np.arange(D)
        pca_ord = pca1d_order(e)
        nn_ord = greedy_nn_order(e, dmat)
        per_scan.append({
            "vol": v, "D": int(D),
            "tau_pca1d": _abs_kendall(pca_ord, to),
            "tau_greedy_nn": _abs_kendall(nn_ord, to),
            "true_path_len": path_len(to, dmat),
            "pca_path_len": path_len(pca_ord, dmat),
        })
        if (vi + 1) % 100 == 0:
            print(f"  [{arm}] ordering {vi+1}/{len(vols)}", flush=True)
    return {"n_scans": len(per_scan), "per_scan": per_scan}


# ---------------- bootstrap ----------------

def boot_ci(vals, n=N_BOOT, pct=CI_PCT, seed=SEED):
    a = np.asarray(vals, dtype=np.float64)
    if len(a) < 2:
        return [float(a.mean()) if len(a) else float("nan")] * 2
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for b in range(n):
        s = rng.integers(0, len(a), len(a))
        means[b] = a[s].mean()
    lo, hi = np.percentile(means, pct)
    return [float(lo), float(hi)]


def agg_zreg(res):
    keys = ["mae_norm", "mae_slices", "r2", "rho"]
    out = {"n_test_scans": res["n_test_scans"], "alpha": res["alpha"],
           "n_train_slices": res["n_train_slices"]}
    for k in keys:
        vals = [s[k] for s in res["per_scan"]]
        out[k + "_mean"] = float(np.mean(vals))
        out[k + "_ci"] = boot_ci(vals)
    return out


def agg_ordering(res):
    keys = ["tau_pca1d", "tau_greedy_nn", "true_path_len", "pca_path_len"]
    out = {"n_scans": res["n_scans"]}
    for k in keys:
        vals = [s[k] for s in res["per_scan"]]
        out[k + "_mean"] = float(np.mean(vals))
        out[k + "_ci"] = boot_ci(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/project/ibi-staff/CT-JEPA/public/outputs/exp_c_zposition")
    ap.add_argument("--smoke", action="store_true", help="first 20 scans only")
    ap.add_argument("--ordering_max", type=int, default=600,
                    help="max test scans for the ordering task (unsupervised)")
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset of arms to run (default: all in ARM_DIRS)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    arms_to_run = list(ARM_DIRS) if not args.arms else [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms_to_run if a not in ARM_DIRS]
    if bad:
        raise SystemExit(f"Unknown --arms {bad}; valid: {list(ARM_DIRS)}")

    print("Loading scan list...", flush=True)
    vols = load_scan_list()
    print(f"  {len(vols)} volumes in val CSV", flush=True)

    summary = {}
    for arm in arms_to_run:
        print(f"\n=== Arm: {arm} ===", flush=True)
        scans = load_embeddings(arm, vols, max_scans=(30 if args.smoke else None))
        print(f"  loaded {len(scans)} scans with >= {MIN_SLICES} slices", flush=True)
        train_v, test_v = patient_split(scans)
        print(f"  split: {len(train_v)} train / {len(test_v)} test scans", flush=True)

        zr = zreg(arm, scans, train_v, test_v)
        zr_agg = agg_zreg(zr)
        with open(os.path.join(args.out, f"{arm}_zreg.json"), "w") as f:
            json.dump({"raw": zr, "agg": zr_agg}, f, indent=2)
        print(f"  z-reg: MAE_norm={zr_agg['mae_norm_mean']:.4f} "
              f"R2={zr_agg['r2_mean']:.4f} rho={zr_agg['rho_mean']:.4f}", flush=True)

        om = ordering(arm, scans, test_v,
                      max_scans=(20 if args.smoke else args.ordering_max))
        om_agg = agg_ordering(om)
        with open(os.path.join(args.out, f"{arm}_ordering.json"), "w") as f:
            json.dump({"raw": om, "agg": om_agg}, f, indent=2)
        print(f"  ordering: tau_pca1d={om_agg['tau_pca1d_mean']:.4f} "
              f"tau_greedy_nn={om_agg['tau_greedy_nn_mean']:.4f}", flush=True)

        summary[arm] = {"zreg": zr_agg, "ordering": om_agg}

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    lines = ["Exp C — Volumetric localization from frozen slice embeddings",
             "=" * 64]
    for arm in arms_to_run:
        s = summary[arm]
        z, o = s["zreg"], s["ordering"]
        lines.append(f"\n{arm}")
        lines.append(f"  z-reg   MAE_norm={z['mae_norm_mean']:.4f} CI{z['mae_norm_ci']}  "
                     f"MAE_slc={z['mae_slices_mean']:.2f}  R2={z['r2_mean']:.4f}  "
                     f"rho={z['rho_mean']:.4f}")
        lines.append(f"  order   tau_pca1d={o['tau_pca1d_mean']:.4f} CI{o['tau_pca1d_ci']}  "
                     f"tau_greedy_nn={o['tau_greedy_nn_mean']:.4f} CI{o['tau_greedy_nn_ci']}")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(txt)
    print("\n" + txt, flush=True)
    print(f"\nWrote {args.out}/summary.{{json,txt}}", flush=True)


if __name__ == "__main__":
    main()
