"""Exp F — organ-span localization from frozen per-slice CLS embeddings.

Task: given a scan, predict which contiguous slice range contains each organ
("which slices contain the heart?"), from a linear probe on frozen embeddings —
automatic scan-range selection without running a segmentation model at
inference. This cashes in the within-scan structure that depth-aware slab
sampling induces (smooth organ-presence curves), where cross-patient retrieval
(exp E) does not.

Method, per arm:
  1. Fit one multi-output Ridge probe CLS -> 118 TS fractional-coverage classes
     on the 281-scan train split (all slices pooled; RidgeCV alphas 1e-3..1e3).
  2. Predict per-slice coverage curves on val (47) and test (86) scans. Organ
     groups (heart, lung_left, lung_right) take the max over member classes.
  3. Per organ, tune a threshold tau on VAL (grid) maximizing mean span IoU;
     predicted span = largest connected run of slices above tau.
  4. Score on TEST: span IoU, boundary MAE (start/end, in slices and normalized
     by scan depth), n_components (curve coherence: how many disconnected
     above-tau runs — flicker count), miss rate (no predicted span).

GT span: first..last slice with group coverage > GT_EPS (TS masks are
volumetric and contiguous; EPS kills mask speckle). Organs eligible if GT span
exists in >= MIN_SCANS test scans.

Arms: the ViT-B ablation pair (depth_aware_25d vs pure_2d — the 2.5D
attribution contrast) + released ViT-L DALE-CT-0 (CT-RATE) and DALE-CT-0-L
(dale_ct_0_chest) — the capacity/scale rungs of the ladder.

CI: 2000-resample bootstrap over test scans, shared indices across arms;
paired deltas (depth_aware - pure_2d, dale_ct_0_chest - dale_ct_0) with CIs.

GPU-free; run on the DGX login node with the CT-MIL venv.
Outputs: outputs/exp_f_organ_span/{arm}.json + summary.md
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_c_zposition import ARM_DIRS, load_embeddings  # noqa: E402
from exp_c_worldmodel_probes import GT_ROOT, load_split_414, load_ts_slices  # noqa: E402

OUT_DIR = "/project/ibi-staff/CT-JEPA/public/outputs/exp_f_organ_span_v2"
TS_YAML = "/project/ibi-staff/CT-JEPA/public/configs/linear_probe_model_comparison_ts.yaml"
DALE0_DIR = ("/project/ibi-staff/CT-JEPA/public/outputs/model_comparison_embeddings/"
             "lejepa_0/cropped_256")
ARMS = ["depth_aware_25d", "pure_2d", "dale_ct_0", "dale_ct_0_chest"]
GROUPS = {
    "heart": ["heart_myocardium", "heart_atrium_left", "heart_ventricle_left",
              "heart_atrium_right", "heart_ventricle_right"],
    "lung_left": ["lung_upper_lobe_left", "lung_lower_lobe_left"],
    "lung_right": ["lung_upper_lobe_right", "lung_middle_lobe_right",
                   "lung_lower_lobe_right"],
}
SINGLES = ["liver", "spleen", "stomach", "kidney_left", "kidney_right",
           "esophagus", "trachea", "aorta", "sternum"]
GT_EPS = 1e-4
MIN_SCANS = 40
# Relative threshold: tau is a FRACTION of the per-scan curve max (v2 fix —
# absolute-tau grid saturated at its minimum because Ridge coverage predictions
# are low-amplitude for small organs; relative tau is amplitude-invariant and
# arm-fair). Grid over fractions:
TAUS = np.round(np.arange(0.05, 1.0, 0.05), 3)
N_BOOT, SEED = 2000, 42


def load_dale0(vols):
    """DALE-CT-0 per-slice CLS from the model-comparison npz store."""
    out = {}
    for v in vols:
        p = os.path.join(DALE0_DIR, v.replace(".nii.gz", "") + ".npz")
        if os.path.exists(p):
            out[v] = np.load(p)["cls"].astype(np.float32)
    return out


def gt_span(curve):
    """(start, end) of GT coverage, or None if organ absent."""
    idx = np.where(curve > GT_EPS)[0]
    return (int(idx[0]), int(idx[-1])) if len(idx) else None


def runs_above(curve, tau):
    """List of (start, end) runs where curve > tau."""
    above = curve > tau
    if not above.any():
        return []
    d = np.diff(above.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0])
    if above[0]:
        starts = [0] + starts
    if above[-1]:
        ends = ends + [len(curve) - 1]
    return list(zip(starts, ends))


def span_iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return inter / union


def score_scan(pred_curve, gt, tau_frac):
    """Metrics for one scan/organ; threshold = tau_frac * curve max (relative)."""
    cmax = float(pred_curve.max())
    if cmax <= 0:
        return {"iou": 0.0, "mae": np.nan, "mae_norm": np.nan,
                "n_comp": 0, "miss": 1}
    rr = runs_above(pred_curve, tau_frac * cmax)
    n_comp = len(rr)
    if not rr:
        return {"iou": 0.0, "mae": np.nan, "mae_norm": np.nan,
                "n_comp": 0, "miss": 1}
    best = max(rr, key=lambda r: r[1] - r[0])          # largest connected run
    D = len(pred_curve)
    mae = 0.5 * (abs(best[0] - gt[0]) + abs(best[1] - gt[1]))
    return {"iou": span_iou(best, gt), "mae": float(mae),
            "mae_norm": float(mae / D), "n_comp": n_comp, "miss": 0}


def organ_curves(pred, ts_names, organ):
    """Group max over member classes (or the single class)."""
    members = GROUPS.get(organ, [organ])
    cols = [ts_names.index(m) for m in members if m in ts_names]
    return pred[:, cols].max(axis=1)


def evaluate_arm(scans, vols_tr, vols_va, vols_te, ts, ts_names, organs):
    """Fit probe, tune taus on val, score test. Returns per-organ per-scan rows."""
    Xtr = np.concatenate([scans[v] for v in vols_tr])
    Ytr = np.concatenate([ts[v] for v in vols_tr])
    probe = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xtr, Ytr)
    pred = {v: probe.predict(scans[v]) for v in vols_va + vols_te}

    taus, results = {}, {}
    for organ in organs:
        # tau tuned on val mean IoU
        val_items = []
        for v in vols_va:
            g = gt_span(organ_curves(ts[v], ts_names, organ))
            if g:
                val_items.append((organ_curves(pred[v], ts_names, organ), g))
        best_tau, best_iou = TAUS[0], -1.0
        for tau in TAUS:
            ious = [score_scan(c, g, tau)["iou"] for c, g in val_items]
            m = float(np.mean(ious)) if ious else 0.0
            if m > best_iou:
                best_iou, best_tau = m, float(tau)
        taus[organ] = best_tau
        # test
        rows = {}
        for v in vols_te:
            g = gt_span(organ_curves(ts[v], ts_names, organ))
            if g is None:
                continue
            rows[v] = score_scan(organ_curves(pred[v], ts_names, organ), g, best_tau)
        results[organ] = rows
    return taus, results


def boot_stats(vals, idx_sets):
    a = np.asarray(vals, dtype=np.float64)
    means = np.array([np.nanmean(a[i]) for i in idx_sets])
    return float(np.nanmean(a)), [float(np.percentile(means, 2.5)),
                                  float(np.percentile(means, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ts_names = yaml.safe_load(open(TS_YAML))["ts_classes"]
    split = load_split_414()
    all_vols = split["train"] + split["val"] + split["test"]
    ts = load_ts_slices(all_vols)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    arm_scans = {}
    for arm in arms:
        arm_scans[arm] = (load_dale0(all_vols) if arm == "dale_ct_0"
                          else load_embeddings(arm, all_vols))
    # volumes valid in EVERY arm (D matches labels), per split
    def ok(v):
        return all(v in s and v in ts and s[v].shape[0] == ts[v].shape[0]
                   for s in arm_scans.values())
    tr = [v for v in split["train"] if ok(v)]
    va = [v for v in split["val"] if ok(v)]
    te = [v for v in split["test"] if ok(v)]
    print(f"scans: train {len(tr)} / val {len(va)} / test {len(te)}")

    # eligible organs: GT span in >= MIN_SCANS test scans
    organs = []
    for organ in list(GROUPS) + SINGLES:
        n = sum(1 for v in te if gt_span(organ_curves(ts[v], ts_names, organ)))
        if n >= MIN_SCANS:
            organs.append(organ)
        print(f"  organ {organ}: GT present in {n}/{len(te)} test scans"
              + ("" if n >= MIN_SCANS else "  -> EXCLUDED"))

    all_results = {}
    for arm in arms:
        taus, res = evaluate_arm(arm_scans[arm], tr, va, te, ts, ts_names, organs)
        all_results[arm] = {"taus": taus, "results": res}
        json.dump({"taus": taus,
                   "results": {o: res[o] for o in organs}},
                  open(os.path.join(args.out, f"{arm}.json"), "w"), indent=1)
        print(f"[{arm}] done; taus={taus}")

    # ---- aggregate: per-organ table + macro + paired deltas ----
    rng = np.random.default_rng(SEED)
    lines = [f"# Exp F — organ-span localization (test n={len(te)} scans)\n"]
    metrics = ["iou", "mae", "mae_norm", "n_comp", "miss"]
    summary = {}
    pairs = [("depth_aware_25d", "pure_2d"), ("dale_ct_0_chest", "dale_ct_0")]
    for organ in organs + ["MACRO"]:
        lines.append(f"\n## {organ}\n")
        lines.append("| arm | span IoU | bnd MAE (sl) | MAE/D | n_comp | miss |")
        lines.append("|---|---|---|---|---|---|")
        for arm in arms:
            row = {}
            for m in metrics:
                if organ == "MACRO":
                    per_organ = [np.nanmean([r[m] for r in
                                 all_results[arm]["results"][o].values()])
                                 for o in organs]
                    row[m] = (float(np.nanmean(per_organ)), None)
                else:
                    scans_o = sorted(all_results[arm]["results"][organ])
                    vals = [all_results[arm]["results"][organ][v][m] for v in scans_o]
                    idx = [rng.integers(0, len(vals), len(vals)) for _ in range(200)]
                    row[m] = boot_stats(vals, idx)
            summary.setdefault(organ, {})[arm] = {m: row[m][0] for m in metrics}
            fmt = lambda m: (f"{row[m][0]:.3f}" if row[m][1] is None
                             else f"{row[m][0]:.3f} [{row[m][1][0]:.3f},{row[m][1][1]:.3f}]")
            lines.append(f"| {arm} | {fmt('iou')} | {fmt('mae')} | {fmt('mae_norm')} "
                         f"| {fmt('n_comp')} | {fmt('miss')} |")
        # paired deltas on shared scans (IoU + MAE), shared bootstrap indices
        if organ != "MACRO":
            for a1, a2 in pairs:
                if a1 not in arms or a2 not in arms:
                    continue
                shared = sorted(set(all_results[a1]["results"][organ])
                                & set(all_results[a2]["results"][organ]))
                if len(shared) < 10:
                    continue
                d_iou = np.array([all_results[a1]["results"][organ][v]["iou"]
                                  - all_results[a2]["results"][organ][v]["iou"]
                                  for v in shared])
                bs = rng.integers(0, len(shared), (N_BOOT, len(shared)))
                dm = d_iou[bs].mean(1)
                lines.append(f"  - dIoU {a1}-{a2}: {d_iou.mean():+.3f} "
                             f"[{np.percentile(dm,2.5):+.3f},{np.percentile(dm,97.5):+.3f}]")
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
