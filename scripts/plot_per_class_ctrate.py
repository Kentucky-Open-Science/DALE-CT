#!/usr/bin/env python
"""Per-class CT-RATE figure (5 models, 5-seed mean + 95% bootstrap CI).

For inspection/iteration before insertion into the paper. Reads the 5 CT-RATE
aggregate JSONs from outputs/error_bars_fair_subset/aggregate/ and plots one point per model
per class with asymmetric bootstrap-CI error bars.

Tweak METRIC / SORT_BY / colors at the top. Run:
    .venv/bin/python scripts/plot_per_class_ctrate.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGG = os.path.join(REPO, "outputs", "error_bars_fair_subset", "aggregate")

# ---- config (iterate here) -------------------------------------------------
METRIC = "auprc"          # auprc | auroc | f1 | ba
SORT_BY = "canonical"     # canonical | prevalence | auprc_mean
YLABEL = {
    "auprc": "Per-class AUPRC",
    "auroc": "Per-class AUROC",
    "f1": "Per-class Macro F1",
    "ba": "Per-class Balanced Accuracy",
}[METRIC]

# (json key, legend label, color, marker). Okabe-Ito (colorblind-friendly).
MODELS = [
    ("dinov2_finetuned", "Finetuned DINOv2", "#999999", "o"),
    ("dale_ct_0",        "DALE-CT-0",        "#0072B2", "s"),
    ("dale_ct_0_chest",  "DALE-CT-0-L",      "#56B4E9", "v"),
    ("dale_ct_1s_v2",    "DALE-CT-1S-v2",    "#E69F00", "^"),
    ("dale_ct_2s",       "DALE-CT-2S",       "#D55E00", "D"),
]
# ---------------------------------------------------------------------------

# CT-RATE-huggingface-downloads fair-subset test-set (n_test=992) class prevalences
# (fraction of positive samples per class). Computed from y_true of the shared
# fair-subset test set (per_run/dale_ct_1s_v2_ctrate_seed0.npz); identical test
# set across all models, so a single shared array is used regardless of model.
PREVALENCE = {
    "Medical material": 0.0867,
    "Arterial wall calcification": 0.2732,
    "Cardiomegaly": 0.1058,
    "Pericardial effusion": 0.0554,
    "Coronary artery wall calcification": 0.2470,
    "Hiatal hernia": 0.1401,
    "Lymphadenopathy": 0.2601,
    "Emphysema": 0.1875,
    "Atelectasis": 0.2319,
    "Lung nodule": 0.4536,
    "Lung opacity": 0.3760,
    "Pulmonary fibrotic sequela": 0.2853,
    "Pleural effusion": 0.1048,
    "Mosaic attenuation pattern": 0.0917,
    "Peribronchial thickening": 0.0998,
    "Consolidation": 0.1734,
    "Bronchiectasis": 0.1109,
    "Interlobular septal thickening": 0.0685,
}

def load_model(key):
    d = json.load(open(os.path.join(AGG, f"{key}_ctrate.json")))
    label_names = d["label_names"]
    pc = d["seed_mean_std"]["per_class"]
    ci = d["bootstrap_ci_mean_across_seeds"]["per_class"]
    means = np.array([pc[c][METRIC]["mean"] for c in label_names])
    lo = np.array([ci[c][METRIC][0] for c in label_names])
    hi = np.array([ci[c][METRIC][1] for c in label_names])
    return label_names, means, lo, hi

# Load all models (label_names identical across models; take from first).
loaded = {key: load_model(key) for key, *_ in MODELS}
label_names = loaded[MODELS[0][0]][0]
prev = np.array([PREVALENCE[c] for c in label_names])

# Class ordering.
if SORT_BY == "prevalence" and prev is not None:
    order = np.argsort(-prev)            # most prevalent first
elif SORT_BY == "auprc_mean":
    avg = np.mean([loaded[k][1] for k, *_ in MODELS], axis=0)
    order = np.argsort(-avg)             # highest AUPRC first
else:
    order = np.arange(len(label_names))
names = [label_names[i] for i in order]

n = len(names)
x = np.arange(n)
offsets = np.array([-0.3, -0.15, 0.0, 0.15, 0.3])  # 5 models within a class slot

fig, ax = plt.subplots(figsize=(12.0, 4.6))

all_lo, all_hi = [], []
for (key, label, color, marker), off in zip(MODELS, offsets):
    _, means, lo, hi = loaded[key]
    m = means[order]; lo = lo[order]; hi = hi[order]
    yerr = np.vstack([m - lo, hi - m])
    ax.errorbar(x + off, m, yerr=yerr, fmt=marker, color=color, ms=5.5,
                capsize=2.5, lw=1.1, elinewidth=1.0, label=label, zorder=3)
    all_lo.append(lo.min()); all_hi.append(hi.max())

# Prevalence = random-guessing baseline for AUPRC (a random ranker scores
# AUPRC ~= prevalence per class). Meaningful for AUPRC only; AUROC/F1/BA
# baselines are 0.5 / prevalence-derived differently, so skip there.
if METRIC == "auprc":
    pv = prev[order]
    ax.plot(x, pv, linestyle="none", color="black", marker="X", ms=6, mew=1.4,
            alpha=0.85, label="Prevalence (random)", zorder=2)
    all_lo.append(float(np.nanmin(pv)))

ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
ax.set_ylabel(YLABEL, fontsize=10)
ax.tick_params(axis="y", labelsize=9)
ax.grid(axis="y", ls=":", alpha=0.5)
ax.set_axisbelow(True)
pad = 0.03
ax.set_ylim(max(0.0, min(all_lo) - pad), min(1.0, max(all_hi) + pad))

ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=6,
          fontsize=8.5, frameon=False, handletextpad=0.4, columnspacing=1.2)

fig.tight_layout()
out_png = os.path.join(REPO, "manuscript", "fig_per_class_ctrate.png")
out_pdf = os.path.join(REPO, "manuscript", "fig_per_class_ctrate.pdf")
fig.savefig(out_png, dpi=220, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"wrote {out_png}\nwrote {out_pdf}")
print(f"metric={METRIC} sort={SORT_BY} classes={n} models={len(MODELS)}")
