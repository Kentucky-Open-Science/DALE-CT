#!/usr/bin/env python3
"""
Exp C (world-model probes) — find the true value of Depth-Aware sampling.

Extends Exp C (z-reg / slice-ordering) with a battery of world-model probes on
the SAME frozen per-slice CLS embeddings, run on three arms:

  - depth_aware_25d : ViT-Base LeJEPA, slab_size=12.0 (12mm cross-slice slabs)
  - pure_2d         : ViT-Base LeJEPA, single_slice=true (intra-slice only)
  - released_dale_ct_2s : ViT-Large from-scratch DALE-CT-2S (corroboration;
                          no pure-2D control, DINOv2-caveat-robustness check)

The first two are the clean Table-V ablation arms: identical arch (CardViT-base,
patch14, d=768), LeJEPA+SIGReg, aux OFF, 35k iters, same data/crops — ONLY the
sampling differs. The question these probes answer: does Depth-Aware build a
smoother / more physically-ordered latent z-manifold (a world-model property),
beyond what supervised z-regression already captures?

PROBE ROSTER (adversarially vetted — see TMI_SI_SUBMISSION_PLAN / Atlas):
  P0  effective_rank      VALIDITY GATE. If depth-aware is lower-rank, every
                          geometric win below is confounded by rank collapse.
                          Certified only if PR_depth >= 0.9 * PR_2d.
  P1  slab_scale_invariance  d_cos(z_i, z_{i+k}) vs k. MECHANISM probe: depth-
                          aware's objective spans the 12mm slab, so it should
                          show a flat plateau for k within ~4-6 slices then
                          rise; pure-2D rises immediately. The differential AT
                          THE SLAB SCALE is the causal signature.
  P2  isometry            Pearson/Spearman of physical |z_i-z_j| vs latent L2
                          (proper metric, L2-normalized). Penalizes FOLDING —
                          pure-2D's documented pathology — distinct from
                          consecutive continuity.
  P3  residualized_transition  Project out the linear z-direction, then test
                          residual_t -> residual_{t+1}. Isolates transition
                          structure BEYOND the smooth global trajectory.
                          Honest null possible.
  P4  anatomy_beyond_z    k-means on z-residualized CLS, NMI/ARI vs the raw
                          118-dim TotalSegmentator organ-presence pattern.
                          The only SEMANTIC axis. May return null / favor 2D.
  P5  organ_prob_coherence  Per-organ linear probe; total-variation of predicted
                          organ-probability curves across z. Smoother manifold ->
                          fewer-flip curves. Clinical "so what".
  P6  on_manifold_interp  Midpoint of (z_t, z_{t+k}) vs nearest real slice.
                          Convex manifold: small gap; folded: large. NOT a
                          tautology (unlike linear-decode interpolation).
  P7  continuity_metric_fix  Re-report the Latent Continuity Score with a PROPER
                          metric (L2 on unit-norm, obeys triangle inequality =>
                          ratio <= 1). The headline Table-V value 1.6844 exceeds
                          the script's stated max of 1.0 because cosine DISTANCE
                          violates the triangle inequality; this makes the number
                          defensible. Ranking survives.

DROPPED probes (redundant/tautological per vetting):
  - transition decay vs gap k    : dual of continuity (decay == curvature).
  - linear-decode interpolation  : tautology (linear decode of linear interp
                                    => R^2=1 for both arms, no separation).
  - pairwise ordering classifier : mirrors z-reg parity by construction.

Operating sets (stated explicitly in each JSON):
  - Geometric probes (P0,P1,P2,P3,P6,P7): 593-scan test split (patient-level
    80/20, seed 42 — same as exp_c_zposition).
  - Semantic probes (P4,P5): 414 ReX-annotated scans (TS per-slice labels exist
    only here), 70/10/20 patient split -> 86 test scans.

CI: 2000-resample bootstrap over scans (per-scan sufficient stats; NONE of these
metrics is a pooled ratio). Matches the ctrate error-bars protocol.

GPU-FREE: runs on the login node from pre-extracted CLS .npy. No backbone, no
images. Reuses exp_c_zposition.py helpers (ARM_DIRS, load_embeddings,
patient_split, boot_ci).

Outputs: outputs/exp_c_worldmodel/{arm}_{probe}.json + summary.{json,txt}
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

# Reuse the verified Exp C infrastructure (same dir).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from exp_c_zposition import (  # noqa: E402
    ARM_DIRS,
    boot_ci,
    load_embeddings,
    load_scan_list,
    patient_split,
    SEED,
    N_BOOT,
    CI_PCT,
    MIN_SLICES,
)

# ---- paths / constants -----------------------------------------------------
GT_ROOT = "/project/ibi-staff/CT-JEPA/public/outputs/model_comparison_groundtruth"
TS_DIR = os.path.join(GT_ROOT, "totalseg")          # {vol}.npz: cls_labels[D,118]
SPLIT_414 = os.path.join(GT_ROOT, "patient_splits.json")  # train/val/test 70/10/20

SLAB_K_MAX = 12        # max gap k (slices) for slab-scale + interp probes
CLUSTER_K = 5          # anatomical bands per scan (P4)
INTERP_KS = (2, 4, 8)  # gaps for on-manifold interpolation (P6)


# ---- semantic-set helpers --------------------------------------------------

def load_split_414():
    """Return {train,val,test} volume-name lists from patient_splits.json."""
    with open(SPLIT_414) as f:
        d = json.load(f)
    return {"train": d["train"], "val": d["val"], "test": d["test"]}


def load_ts_slices(vols):
    """Load TotalSegmentator per-slice organ-coverage labels for a volume list.

    Returns {vol: np.ndarray[D,118] float32 fractional coverage}. Only volumes
    whose .npz exists are returned; caller intersects with the CLS dict and
    checks D consistency.
    """
    out = {}
    for v in vols:
        p = os.path.join(TS_DIR, v + ".npz")      # vol already ends with .nii.gz
        if not os.path.exists(p):
            continue
        z = np.load(p)
        cls = z["cls_labels"].astype(np.float32)  # [D,118]
        if cls.ndim == 2 and cls.shape[0] >= MIN_SLICES:
            out[v] = cls
    return out


def fit_z_direction(scans, train_vols):
    """Fit a linear z-regressor on train CLS; return the z-direction unit vector.

    Reuses the exp_c z-reg recipe: z = linspace(0,1,D), RidgeCV. Returns w
    (unit-normalized coefficient, [D_emb]) and the fitted model. w is the
    dominant linear position direction; projecting it out removes the trivial
    z signal that makes naive probes return parity.
    """
    Xtr, ytr = [], []
    for v in train_vols:
        e = scans[v]
        D = e.shape[0]
        Xtr.append(e)
        ytr.append(np.linspace(0.0, 1.0, D, dtype=np.float32))
    Xtr = np.concatenate(Xtr, 0)
    ytr = np.concatenate(ytr, 0)
    model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xtr, ytr)
    w = model.coef_.astype(np.float32)
    w = w / (np.linalg.norm(w) + 1e-12)
    return w, model


def project_out_z(emb, w):
    """Remove the linear z-component: emb - (emb@w)/(w@w) * w. (w is unit.)"""
    coef = (emb @ w)[:, None]          # [D,1]
    return emb - coef * w[None, :]     # [D,D_emb]


# ---- per-probe implementations ---------------------------------------------

def probe_effective_rank(arm, scans, test_vols, **_):
    """P0 — participation ratio + stable rank of per-scan embedding covariance.

    Validity gate: geometric wins are CERTIFIED only if PR_depth >= 0.9*PR_2d.
    """
    pr, sr = [], []
    for v in test_vols:
        e = scans[v].astype(np.float64)
        e = e - e.mean(0, keepdims=True)
        _, S, _ = np.linalg.svd(e, full_matrices=False)
        lam = (S ** 2) / e.shape[0]
        pr.append(float((lam.sum() ** 2) / ((lam ** 2).sum() + 1e-12)))
        sr.append(float((S ** 2).sum() / (S[0] ** 2 + 1e-12)))
    return {
        "n_scans": len(test_vols),
        "raw": {"pr": pr, "sr": sr},
        "agg": {
            "n_scans": len(test_vols),
            "pr_mean": float(np.mean(pr)), "pr_ci": boot_ci(pr),
            "sr_mean": float(np.mean(sr)), "sr_ci": boot_ci(sr),
        },
    }


def probe_slab_scale_invariance(arm, scans, test_vols, **_):
    """P1 — d_cos(z_i, z_{i+k}) vs k. The mechanism probe (slab plateau)."""
    profs = []
    ks = list(range(1, SLAB_K_MAX + 1))
    for v in test_vols:
        e = scans[v].astype(np.float64)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)  # L2-norm rows
        D = e.shape[0]
        prof = []
        for k in ks:
            if k >= D:
                prof.append(np.nan)
                continue
            d = 1.0 - (e[:-k] * e[k:]).sum(1)   # cosine distance per pair
            prof.append(float(np.mean(d)))
        profs.append(prof)
    profs = np.array(profs, dtype=np.float64)            # [n_scans, K]
    means = np.nanmean(profs, axis=0)                    # [K]
    cis = [boot_ci(profs[:, k][~np.isnan(profs[:, k])]) for k in range(len(ks))]
    # plateau half-width: largest k where (mean[k]-mean[0]) < 0.5*(mean[-1]-mean[0])
    span = means[-1] - means[0]
    thr = 0.5 * span
    pw = 0
    for k in range(len(ks)):
        if means[k] - means[0] < thr:
            pw = k + 1
        else:
            break
    return {
        "n_scans": len(test_vols),
        "raw": {"ks": ks, "profiles": profs.tolist()},
        "agg": {
            "n_scans": len(test_vols),
            "ks": ks,
            "dcos_mean": means.tolist(),
            "dcos_ci": cis,
            "plateau_width_slices": int(pw),
        },
    }


def probe_isometry(arm, scans, test_vols, **_):
    """P2 — correlation of physical |z_i-z_j| vs latent L2 (unit-norm). Penalizes folding."""
    r_pear, r_spr = [], []
    iu_idx = None
    for v in test_vols:
        e = scans[v].astype(np.float64)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        D = e.shape[0]
        z = np.linspace(0.0, 1.0, D)
        dmat = cdist(e, e, metric="euclidean")          # L2 on unit-norm (proper metric)
        Z = np.abs(z[:, None] - z[None, :])
        iu = np.triu_indices(D, k=1)
        r_pear.append(float(np.corrcoef(Z[iu], dmat[iu])[0, 1]))
        r_spr.append(float(spearmanr(Z[iu], dmat[iu]).statistic))
    return {
        "n_scans": len(test_vols),
        "raw": {"r_pearson": r_pear, "r_spearman": r_spr},
        "agg": {
            "n_scans": len(test_vols),
            "r_pearson_mean": float(np.mean(r_pear)), "r_pearson_ci": boot_ci(r_pear),
            "r_spearman_mean": float(np.mean(r_spr)), "r_spearman_ci": boot_ci(r_spr),
        },
    }


def probe_residualized_transition(arm, scans, train_vols, test_vols, **_):
    """P3 — project out linear z, then predict residual_t -> residual_{t+1}.

    Fit Ridge (multi-output) ONCE on train-residual transitions; score per test
    scan. r2 = global F-norm variance explained (principled single scalar);
    cos = mean cosine of consecutive residuals (collinearity/smoothness).
    """
    w, _ = fit_z_direction(scans, train_vols)
    # fit on train residuals
    Xtr, Ytr = [], []
    for v in train_vols:
        r = project_out_z(scans[v].astype(np.float32), w)
        Xtr.append(r[:-1]); Ytr.append(r[1:])
    Xtr = np.concatenate(Xtr, 0); Ytr = np.concatenate(Ytr, 0)
    model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xtr, Ytr)
    per_scan = []
    for v in test_vols:
        r = project_out_z(scans[v].astype(np.float32), w)
        X = r[:-1]; Y = r[1:]
        Yp = model.predict(X)
        ss_res = float(np.sum((Y - Yp) ** 2))
        ss_tot = float(np.sum((Y - Y.mean(0, keepdims=True)) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        cos = float(np.mean(
            (X * Y).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1) + 1e-12)
        ))
        per_scan.append({"vol": v, "D": int(r.shape[0]), "r2": r2, "cos": cos})
    r2s = [s["r2"] for s in per_scan]; coss = [s["cos"] for s in per_scan]
    return {
        "n_scans": len(test_vols),
        "alpha": float(model.alpha_),
        "raw": {"per_scan": per_scan},
        "agg": {
            "n_scans": len(test_vols), "alpha": float(model.alpha_),
            "r2_mean": float(np.mean(r2s)), "r2_ci": boot_ci(r2s),
            "cos_mean": float(np.mean(coss)), "cos_ci": boot_ci(coss),
        },
    }


def probe_anatomy_beyond_z(arm, scans_414, train_414, test_414, ts_414, **_):
    """P4 — k-means on z-residualized CLS vs TS organ-presence pattern (NMI/ARI).

    Within-scan: D slices get latent-cluster labels (z-residualized CLS) and
    anatomy-cluster labels (binarized TS organ pattern); NMI/ARI between them.
    Honest probe: pure-2D has sharper per-slice anatomy, so this could be null
    or favor pure-2D. Uses ablation arms (aux OFF, neither saw organ labels).
    """
    w, _ = fit_z_direction(scans_414, train_414)
    nmi_vals, ari_vals = [], []
    for v in test_414:
        e = scans_414[v].astype(np.float32)
        tb = (ts_414[v] > 0).astype(np.float32)        # [D,118] organ presence
        if e.shape[0] != tb.shape[0] or e.shape[0] < CLUSTER_K + 2:
            continue
        r = project_out_z(e, w)
        cls_lab = KMeans(n_clusters=CLUSTER_K, n_init=10, random_state=SEED).fit_predict(r)
        anat_lab = KMeans(n_clusters=CLUSTER_K, n_init=10, random_state=SEED).fit_predict(tb)
        nmi_vals.append(float(normalized_mutual_info_score(anat_lab, cls_lab)))
        ari_vals.append(float(adjusted_rand_score(anat_lab, cls_lab)))
    return {
        "n_scans": len(nmi_vals),
        "raw": {"nmi": nmi_vals, "ari": ari_vals},
        "agg": {
            "n_scans": len(nmi_vals), "k": CLUSTER_K,
            "nmi_mean": float(np.mean(nmi_vals)), "nmi_ci": boot_ci(nmi_vals),
            "ari_mean": float(np.mean(ari_vals)), "ari_ci": boot_ci(ari_vals),
        },
    }


def probe_organ_prob_coherence(arm, scans_414, train_414, test_414, ts_414, **_):
    """P5 — per-organ linear probe; total-variation of predicted organ-prob curves.

    Train multi-output Ridge (CLS -> 118 fractional-organ-coverage) on 414-train;
    per test scan, predict P[D,118], measure mean total-variation (|diff|) and
    mean 2nd-difference across z, over organs present in that scan. Lower =
    smoother = favors depth-aware. Clinical "so what": smoother organ-prob curves.
    """
    Xtr, Ytr = [], []
    for v in train_414:
        Xtr.append(scans_414[v].astype(np.float32))
        Ytr.append(ts_414[v])
    Xtr = np.concatenate(Xtr, 0); Ytr = np.concatenate(Ytr, 0)
    model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xtr, Ytr)
    tv_vals, sd_vals, norg = [], [], []
    for v in test_414:
        e = scans_414[v].astype(np.float32)
        tb = ts_414[v]
        if e.shape[0] != tb.shape[0] or e.shape[0] < 4:
            continue
        P = model.predict(e)                              # [D,118]
        present = np.where(tb.any(0))[0]                  # organs present in scan
        if len(present) == 0:
            continue
        tvs = np.mean(np.abs(np.diff(P[:, present], axis=0)), axis=0)
        sds = np.mean(np.abs(np.diff(P[:, present], n=2, axis=0)), axis=0)
        tv_vals.append(float(np.mean(tvs)))
        sd_vals.append(float(np.mean(sds)))
        norg.append(int(len(present)))
    return {
        "n_scans": len(tv_vals),
        "alpha": float(model.alpha_),
        "raw": {"tv": tv_vals, "second_diff": sd_vals, "n_organs": norg},
        "agg": {
            "n_scans": len(tv_vals), "alpha": float(model.alpha_),
            "n_organs_per_scan_mean": float(np.mean(norg)),
            "tv_mean": float(np.mean(tv_vals)), "tv_ci": boot_ci(tv_vals),
            "second_diff_mean": float(np.mean(sd_vals)), "second_diff_ci": boot_ci(sd_vals),
        },
    }


def probe_on_manifold_interp(arm, scans, test_vols, **_):
    """P6 — midpoint of (z_t,z_{t+k}) vs nearest real slice (off-manifold gap).

    Convex smooth manifold: midpoint stays near a real slice (small gap);
    folded manifold: chord cuts through off-manifold space (large gap).
    Normalized by scan mean pairwise distance for scale-invariance.
    """
    per_k = {k: [] for k in INTERP_KS}
    for v in test_vols:
        e = scans[v].astype(np.float64)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        D = e.shape[0]
        mean_pw = cdist(e, e, metric="euclidean").mean() + 1e-12
        for k in INTERP_KS:
            if k >= D - 1:
                per_k[k].append(np.nan)
                continue
            mids = (e[:-k] + e[k:]) / 2.0                # [D-k, D_emb]
            nn = np.min(cdist(mids, e, metric="euclidean"), axis=1)  # nearest real slice
            per_k[k].append(float(np.mean(nn) / mean_pw))
    agg = {}
    for k in INTERP_KS:
        a = np.array(per_k[k], dtype=np.float64)
        valid = a[~np.isnan(a)]
        agg[k] = {
            "norm_gap_mean": float(np.mean(valid)) if len(valid) else float("nan"),
            "norm_gap_ci": boot_ci(valid) if len(valid) else [float("nan"), float("nan")],
        }
    return {
        "n_scans": len(test_vols),
        "raw": {"ks": list(INTERP_KS), "norm_gap": {str(k): per_k[k] for k in INTERP_KS}},
        "agg": {"n_scans": len(test_vols), "ks": list(INTERP_KS),
                "by_k": {str(k): agg[k] for k in INTERP_KS}},
    }


def probe_continuity_metric_fix(arm, scans, test_vols, **_):
    """P7 — Latent Continuity Score with a PROPER metric (L2 on unit-norm).

    The released Table-V value (1.6844) exceeds the stated max of 1.0 because
    cosine DISTANCE violates the triangle inequality. L2 on unit-norm embeddings
    obeys it, bounding ratio <= 1. Ranking (depth-aware >> pure-2D) survives.
    Also reports the original cosine version + a geodesic (arccos) variant for
    transparency.
    """
    proper, cos_orig, geod = [], [], []
    violations = 0
    for v in test_vols:
        e = scans[v].astype(np.float64)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        # L2 on unit-norm (proper metric)
        consec = np.linalg.norm(e[1:] - e[:-1], axis=1).sum()
        firstlast = np.linalg.norm(e[-1] - e[0])
        r = firstlast / (consec + 1e-12)
        if r > 1.0 + 1e-4:
            violations += 1
        proper.append(float(r))
        # original cosine-distance version (indefensible, for transparency)
        cs = (e @ e.T) / (np.linalg.norm(e, axis=1)[:, None] * np.linalg.norm(e, axis=1)[None, :] + 1e-12)
        np.fill_diagonal(cs, 1.0)
        cos_consec = (1.0 - cs[range(len(e) - 1), range(1, len(e))]).sum()
        cos_fl = 1.0 - cs[0, -1]
        cos_orig.append(float(cos_fl / (cos_consec + 1e-12)))
        # geodesic (arccos) — also proper
        g_consec = np.arccos(np.clip(cs[range(len(e) - 1), range(1, len(e))], -1, 1)).sum()
        g_fl = np.arccos(np.clip(cs[0, -1], -1, 1))
        geod.append(float(g_fl / (g_consec + 1e-12)))
    return {
        "n_scans": len(test_vols),
        "raw": {"l2_proper": proper, "cosine_original": cos_orig, "geodesic": geod},
        "agg": {
            "n_scans": len(test_vols),
            "l2_continuity_mean": float(np.mean(proper)), "l2_continuity_ci": boot_ci(proper),
            "l2_ratio_max_observed": float(np.max(proper)),
            "l2_violations_over_1": int(violations),
            "cosine_orig_mean": float(np.mean(cos_orig)),
            "geodesic_mean": float(np.mean(geod)),
        },
    }


# ---- driver ----------------------------------------------------------------

GEOMETRIC_PROBES = {
    "P0_effective_rank": probe_effective_rank,
    "P1_slab_scale_invariance": probe_slab_scale_invariance,
    "P2_isometry": probe_isometry,
    "P3_residualized_transition": probe_residualized_transition,
    "P6_on_manifold_interp": probe_on_manifold_interp,
}
# P7 runs on ALL scans (not the 593 test split) so it faithfully re-reports the
# released Latent Continuity Score (N=3002) with a proper metric. It is
# unsupervised (no train/test discipline needed).
CONTINUITY_PROBE = ("P7_continuity_metric_fix", probe_continuity_metric_fix)
SEMANTIC_PROBES = {
    "P4_anatomy_beyond_z": probe_anatomy_beyond_z,
    "P5_organ_prob_coherence": probe_organ_prob_coherence,
}


def run_arm(arm, vols, out_dir, smoke=False, max_scans=None):
    print(f"\n=== Arm: {arm} ===", flush=True)
    cap = (30 if smoke else max_scans)
    scans = load_embeddings(arm, vols, max_scans=cap)
    print(f"  loaded {len(scans)} scans with >= {MIN_SLICES} slices", flush=True)
    if not scans:
        print("  [skip] no embeddings", flush=True)
        return {}

    train_v, test_v = patient_split(scans)
    print(f"  geometric split: {len(train_v)} train / {len(test_v)} test", flush=True)

    summary_arm = {}

    # --- geometric probes (593 test) ---
    for name, fn in GEOMETRIC_PROBES.items():
        print(f"  [{name}] ...", flush=True, end=" ")
        if name == "P3_residualized_transition":
            res = fn(arm, scans, train_v, test_v)
        else:
            res = fn(arm, scans, test_v)
        with open(os.path.join(out_dir, f"{arm}_{name}.json"), "w") as f:
            json.dump(res, f, indent=2)
        a = res["agg"]
        _print_geometric(name, a)
        summary_arm[name] = a

    # --- P7 continuity-metric fix on ALL scans (faithful re-report, N ~= 3002) ---
    name, fn = CONTINUITY_PROBE
    print(f"  [{name}] ... (all {len(scans)} scans)", flush=True, end=" ")
    res = fn(arm, scans, list(scans.keys()))
    with open(os.path.join(out_dir, f"{arm}_{name}.json"), "w") as f:
        json.dump(res, f, indent=2)
    a = res["agg"]
    _print_geometric(name, a)
    summary_arm[name] = a

    # --- semantic probes (414 set) ---
    splits = load_split_414()
    vol_414 = [v for v in (splits["train"] + splits["test"]) if v in scans]
    scans_414 = {v: scans[v] for v in vol_414}
    ts_414 = load_ts_slices(vol_414)
    scans_414 = {v: e for v, e in scans_414.items()
                 if v in ts_414 and e.shape[0] == ts_414[v].shape[0]}
    train_414 = [v for v in splits["train"] if v in scans_414]
    test_414 = [v for v in splits["test"] if v in scans_414]
    print(f"  semantic 414-set: {len(train_414)} train / {len(test_414)} test "
          f"(D-matched, TS-labeled)", flush=True)
    for name, fn in SEMANTIC_PROBES.items():
        print(f"  [{name}] ...", flush=True, end=" ")
        res = fn(arm, scans_414, train_414, test_414, ts_414)
        with open(os.path.join(out_dir, f"{arm}_{name}.json"), "w") as f:
            json.dump(res, f, indent=2)
        a = res["agg"]
        _print_semantic(name, a)
        summary_arm[name] = a

    return summary_arm


def _print_geometric(name, a):
    if name == "P0_effective_rank":
        print(f"PR={a['pr_mean']:.2f} sr={a['sr_mean']:.2f}", flush=True)
    elif name == "P1_slab_scale_invariance":
        print(f"plateau={a['plateau_width_slices']}slices "
              f"dcos[1]={a['dcos_mean'][0]:.4f} dcos[{a['ks'][-1]}]="
              f"{a['dcos_mean'][-1]:.4f}", flush=True)
    elif name == "P2_isometry":
        print(f"r_pear={a['r_pearson_mean']:.4f} r_spr={a['r_spearman_mean']:.4f}", flush=True)
    elif name == "P3_residualized_transition":
        print(f"r2={a['r2_mean']:.4f} cos={a['cos_mean']:.4f}", flush=True)
    elif name == "P6_on_manifold_interp":
        g = " ".join(f"k{k}={a['by_k'][str(k)]['norm_gap_mean']:.4f}" for k in a["ks"])
        print(g, flush=True)
    elif name == "P7_continuity_metric_fix":
        print(f"L2_cont={a['l2_continuity_mean']:.4f} (max={a['l2_ratio_max_observed']:.4f}, "
              f"viol={a['l2_violations_over_1']}) cos_orig={a['cosine_orig_mean']:.4f}", flush=True)


def _print_semantic(name, a):
    if name == "P4_anatomy_beyond_z":
        print(f"NMI={a['nmi_mean']:.4f} ARI={a['ari_mean']:.4f} (k={a['k']})", flush=True)
    elif name == "P5_organ_prob_coherence":
        print(f"TV={a['tv_mean']:.4f} 2ndDiff={a['second_diff_mean']:.4f} "
              f"(norg={a['n_organs_per_scan_mean']:.1f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/project/ibi-staff/CT-JEPA/public/outputs/exp_c_worldmodel")
    ap.add_argument("--smoke", action="store_true", help="30 scans only (geometric)")
    ap.add_argument("--max_scans", type=int, default=None,
                    help="cap scans loaded per arm (medium smoke; full run = all)")
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset of arms (default: all in ARM_DIRS)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    arms = list(ARM_DIRS) if not args.arms else [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ARM_DIRS]
    if bad:
        raise SystemExit(f"Unknown --arms {bad}; valid: {list(ARM_DIRS)}")

    print("Loading scan list...", flush=True)
    vols = load_scan_list()
    print(f"  {len(vols)} volumes in val CSV", flush=True)

    summary = {}
    for arm in arms:
        summary[arm] = run_arm(arm, vols, args.out, smoke=args.smoke, max_scans=args.max_scans)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- human-readable summary + P0 gate verdict ----
    lines = ["Exp C — World-model probes (Depth-Aware value)",
             "=" * 64]
    for arm in arms:
        s = summary.get(arm, {})
        if not s:
            continue
        lines.append(f"\n{arm}")
        all_names = list(GEOMETRIC_PROBES) + [CONTINUITY_PROBE[0]] + list(SEMANTIC_PROBES)
        for name in all_names:
            a = s.get(name)
            if not a:
                continue
            if name == "P0_effective_rank":
                lines.append(f"  {name:30s} PR={a['pr_mean']:.3f} sr={a['sr_mean']:.2f}")
            elif name == "P1_slab_scale_invariance":
                lines.append(f"  {name:30s} plateau={a['plateau_width_slices']}sl "
                             f"dcos[1]={a['dcos_mean'][0]:.4f} "
                             f"dcos[{a['ks'][-1]}]={a['dcos_mean'][-1]:.4f}")
            elif name == "P2_isometry":
                lines.append(f"  {name:30s} r_pear={a['r_pearson_mean']:.4f} "
                             f"r_spr={a['r_spearman_mean']:.4f}")
            elif name == "P3_residualized_transition":
                lines.append(f"  {name:30s} r2={a['r2_mean']:.4f} cos={a['cos_mean']:.4f}")
            elif name == "P4_anatomy_beyond_z":
                lines.append(f"  {name:30s} NMI={a['nmi_mean']:.4f} ARI={a['ari_mean']:.4f}")
            elif name == "P5_organ_prob_coherence":
                lines.append(f"  {name:30s} TV={a['tv_mean']:.4f} 2ndDiff={a['second_diff_mean']:.4f}")
            elif name == "P6_on_manifold_interp":
                g = " ".join(f"k{k}={a['by_k'][str(k)]['norm_gap_mean']:.4f}" for k in a["ks"])
                lines.append(f"  {name:30s} {g}")
            elif name == "P7_continuity_metric_fix":
                lines.append(f"  {name:30s} L2_cont={a['l2_continuity_mean']:.4f} "
                             f"(max={a['l2_ratio_max_observed']:.4f}) "
                             f"cos_orig={a['cosine_orig_mean']:.4f}")

    # P0 gate verdict (only meaningful with both ablation arms)
    if "depth_aware_25d" in summary and "pure_2d" in summary:
        pr_da = summary["depth_aware_25d"].get("P0_effective_rank", {}).get("pr_mean")
        pr_2d = summary["pure_2d"].get("P0_effective_rank", {}).get("pr_mean")
        if pr_da is not None and pr_2d is not None:
            ratio = pr_da / (pr_2d + 1e-12)
            verdict = "CERTIFIED" if ratio >= 0.9 else "CONFOUNDED-by-rank-collapse"
            lines.append("")
            lines.append(f"P0 GATE: PR_depth={pr_da:.3f} vs PR_2d={pr_2d:.3f} "
                         f"(ratio {ratio:.3f}) -> geometric wins [{verdict}]")

    txt = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(txt)
    print("\n" + txt, flush=True)
    print(f"\nWrote {args.out}/summary.{{json,txt}}", flush=True)


if __name__ == "__main__":
    main()
