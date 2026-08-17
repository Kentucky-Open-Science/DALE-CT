import json, glob, os, numpy as np

AGG = "/project/ibi-staff/CT-JEPA/public/outputs/error_bars_fair_subset/aggregate"
PER = "/project/ibi-staff/CT-JEPA/public/outputs/error_bars_fair_subset/per_run"


def get_ci(d):
    bc = d.get("bootstrap_ci_mean_across_seeds", {}).get("macro", {})
    au = bc.get("auroc", [None, None]); ap = bc.get("auprc", [None, None])
    return au, ap


print("=== models WITH aggregate (bootstrap CI) ===")
print("%-26s %-14s %3s %22s %22s" % ("model", "arm", "n", "AUROC [CI]", "AUPRC [CI]"))
for fp in sorted(glob.glob(AGG + "/*.json")):
    if "metadata" in fp:
        continue
    d = json.load(open(fp))
    ms = d["seed_mean_std"]["macro"]
    au_lo, au_hi = get_ci(d)[0]
    ap_lo, ap_hi = get_ci(d)[1]
    print("%-26s %-14s %3d %.4f [%.4f,%.4f]   %.4f [%.4f,%.4f]" % (
        d["model"], d["arm"], d["n_seeds"],
        ms["auroc"]["mean"], au_lo, au_hi,
        ms["auprc"]["mean"], ap_lo, ap_hi))

print("\n=== comparison models (mean over per_run seeds) ===")
arms = {}
for fp in sorted(glob.glob(PER + "/*.json")):
    d = json.load(open(fp))
    arms.setdefault((d["model"], d["arm"]), []).append(d["macro"]["auroc"])
print("%-26s %-14s %3s %8s %8s" % ("model", "arm", "n", "AUROC", "AUPRC"))
for (mdl, arm), vals in sorted(arms.items()):
    # auprc
    aps = []
    for fp in sorted(glob.glob(PER + f"/{mdl}_{arm}_seed*.json")):
        aps.append(json.load(open(fp))["macro"]["auprc"])
    print("%-26s %-14s %3d %.4f   %.4f" % (mdl, arm, len(vals), np.mean(vals), np.mean(aps)))
