#!/usr/bin/env python
"""Internal vs external AUPRC line chart for the three 2D backbones.

Connected line chart: AUPRC on the y-axis, the three 2D models on the x-axis in
the order DALE-CT-0, DALE-CT-1S-v2, DALE-CT-2S. Three series (arms) per model:

  * Internal  — CT-RATE (Table III, n_test=992). Single best probe + 95% bootstrap
                CI shown as asymmetric error bars on each point.
  * External (frozen)     — RAD-ChestCT, CT-RATE classifier applied directly (Table IV).
  * External (retrained)  — RAD-ChestCT, encoder frozen, classifier retrained (Table IV).

RAD arms are point means over the available probe seeds (no CI in Table IV), so no
error bars are drawn there.

Values are transcribed verbatim from manuscript/main.tex Tables III & IV
(tab:ct_rate_results, tab:rad_chestct_results). Run:
    .venv/bin/python scripts/plot_internal_external_auprc.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (model key, x-axis label). Order is exactly 0, 1, 2 as requested.
MODELS = [
    ("DALE-CT-0",     "DALE-CT-0"),
    ("DALE-CT-1S-v2", "DALE-CT-1S-v2"),
    ("DALE-CT-2S",    "DALE-CT-2S"),
]

# AUPRC point estimates. CT-RATE values carry a 95% bootstrap CI [lo, hi].
# Source: manuscript/main.tex Table III (CT-RATE) and Table IV (RAD-ChestCT).
DATA = {
    "DALE-CT-0": {
        "internal":  (0.5112, (0.4930, 0.5375)),
        "frozen":    0.3565,
        "retrained": 0.5414,
    },
    "DALE-CT-1S-v2": {
        "internal":  (0.5074, (0.4898, 0.5323)),
        "frozen":    0.3886,
        "retrained": 0.5207,
    },
    "DALE-CT-2S": {
        "internal":  (0.5312, (0.5134, 0.5556)),
        "frozen":    0.3820,
        "retrained": 0.5377,
    },
}

# Three categorical arms (Okabe-Ito, colorblind-safe). Arms, not models, carry color.
SERIES = [
    ("internal",  "Internal (CT-RATE)",   "#0072B2"),
    ("frozen",    "External — frozen",    "#E69F00"),
    ("retrained", "External — retrained", "#009E73"),
]

INK     = "#222222"   # primary text
MUTED   = "#666666"   # value labels / notes
GRID    = "#d9d9d9"   # recessive hairline grid
SURFACE = "#fcfcfb"
ERR     = "#8a8a8a"   # recessive error-bar ink

def main():
    x = np.arange(len(MODELS))
    # Internal upper-CI per point, so the retrained labels can clear the whiskers.
    internal_upper = [DATA[k]["internal"][1][1] for k, _ in MODELS]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for arm, label, color in SERIES:
        vals, lo_err, hi_err = [], [], []
        for key, _ in MODELS:
            entry = DATA[key][arm]
            if arm == "internal":
                v, (lo, hi) = entry
                vals.append(v); lo_err.append(v - lo); hi_err.append(hi - v)
            else:
                vals.append(entry); lo_err.append(0.0); hi_err.append(0.0)
        vals = np.array(vals); lo_err = np.array(lo_err); hi_err = np.array(hi_err)

        # Connected line + filled markers carrying a 2px surface ring.
        ax.plot(x, vals, color=color, linewidth=2.0, marker="o", markersize=7,
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.2,
                solid_capstyle="round", solid_joinstyle="round", zorder=4, label=label)

        # CT-RATE internal: 95% bootstrap CI as recessive asymmetric error bars.
        if arm == "internal":
            ax.errorbar(x, vals, yerr=[lo_err, hi_err], fmt="none",
                        ecolor=ERR, elinewidth=1.0, capsize=2.5, zorder=3)

        # Direct value labels (print figure — no tooltip, so labels carry the value).
        # Retrained sits above (clearing the internal whisker); internal sits below
        # its lower whisker; frozen sits below its point. This keeps the three labels
        # at each x in separate vertical zones so they never collide.
        for xi, v in zip(x, vals):
            if arm == "retrained":
                ylab = max(v, internal_upper[xi]) + 0.012
                va = "bottom"
            elif arm == "internal":
                lo = DATA[MODELS[xi][0]]["internal"][1][0]
                ylab = lo - 0.010
                va = "top"
            else:  # frozen — lowest line, label below
                ylab = v - 0.012
                va = "top"
            ax.text(xi, ylab, f"{v:.3f}", ha="center", va=va,
                    fontsize=7.5, color=MUTED, zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in MODELS], fontsize=10, color=INK)
    ax.set_xlim(-0.3, len(MODELS) - 0.7)
    ax.set_ylabel("AUPRC", fontsize=11, color=INK)
    ax.set_ylim(0, 0.60)
    ax.set_yticks(np.arange(0, 0.61, 0.1))
    ax.tick_params(axis="y", labelsize=9, colors=MUTED)
    ax.tick_params(axis="x", colors=INK)

    # Recessive hairline grid (solid, never dashed); spines off except the baseline.
    ax.yaxis.grid(True, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              fontsize=9, frameon=False, handletextpad=0.5, columnspacing=1.6)

    fig.text(0.5, -0.02,
             "Error bars: 95% bootstrap CI (internal CT-RATE only). "
             "External RAD-ChestCT arms are point means over probe seeds.",
             ha="center", va="top", fontsize=7.5, color=MUTED, style="italic")

    fig.tight_layout()
    out_png = os.path.join(REPO, "manuscript", "fig_internal_external_auprc.png")
    out_pdf = os.path.join(REPO, "manuscript", "fig_internal_external_auprc.pdf")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}\nwrote {out_pdf}")

if __name__ == "__main__":
    main()
