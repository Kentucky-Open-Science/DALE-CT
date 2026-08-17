# Error Bars for the Frozen-Backbone Linear-Probe Evaluation

Adds **airtight error bars** to the existing COLIPRI linear-probe evaluation for
the DALE paper. Two noise sources, reported across **4 frozen backbones** and
**3 arms**:

- **Seed noise** — retrain the probe under 5 seeds of a fixed (validation-selected)
  config; report **mean ± std**.
- **Sampling noise** — bootstrap the test set on each probe's predictions; report
  **95% CIs**.

Backbones are **frozen** (CLS embeddings precomputed on the DGX). Only the
lightweight probes re-run — no encoder training, no new metric code.

---

## What this wraps (not re-derives)

All probe/eval logic already exists and is reused verbatim:

| Reused function | Source | Role |
|---|---|---|
| `train_one_config` | `train_gridsearch.py` | SGD(momentum=0.95, wd=0) + CosineAnnealingLR, BCEWithLogitsLoss, step-based, best-val-macro-AUPRC selection |
| `get_optimal_f1_thresholds` | `train_gridsearch.py` | Per-class F1 threshold optimization on validation |
| `compute_metrics` | `train_gridsearch.py` | Macro + per-class AUROC/AUPRC/F1/BA with the `len(np.unique)<2` exclusion rule |
| `map_ctrate_to_rad` | `eval_rad.py` | 18→16 RAD-ChestCT class mapping (frozen arm) |
| `create_datasets` / `collate_mil_bags` / `CTScanMILDataset` | `dataloaders/dataloader_embeddings.py` | CT-RATE splits + RAD splits (router `dataset_type`) |
| `ColipriProber` | `models/colipri_pooling.py` | The probe (pooling schemes; **no dropout**) |

`train_gridsearch.py` and `eval_rad.py` were refactored only to **extract**
`compute_metrics`, `train_one_config`, and `map_ctrate_to_rad` — the metric
formulas, exclusion rule, hyperparameters, and v628 saving are unchanged. The
v628 configs still run as before.

---

## Protocol

1. **Select (Stage 1, single seed=42):** run the grid (6 LRs × 3 pooling schemes),
   select on validation macro-AUPRC, **write the winning config** to
   `selected_configs/<model>_<task>.json`. The next stage reads this file — nothing
   is hardcoded.

   The grid is defined in `configs/error_bars.yaml`:
   - **Learning rates:** `0.3, 0.1, 0.03, 0.01, 0.003, 0.001`
   - **Pooling schemes:** `average, max, learned_attention`
   - **Training budget:** fixed `total_steps=15000` with CosineAnnealingLR (`T_max=total_steps`);
     eval every `eval_freq=2500`, keeping the best-val-macro-AUPRC checkpoint
     (best-step selection, **not** early stopping — every run trains the full budget).
   - Stage 2 (variance) runs 5 seeds of the **single selected (LR, pooling)** config —
     the grid is not re-run per seed.
2. **Variance (Stage 2, 5 seeds of the fixed config):** read the selected config
   and run 5 seeds of *that config only* — the grid is **not** re-run per seed.
   Per-class F1 thresholds are **re-optimized on validation per seed** (reuses
   `get_optimal_f1_thresholds`). Report test mean ± std + per-run bootstrap CIs.
3. **Aggregate (Stage 3):** read all `per_run/*.json`, write seed mean ± std and
   the mean-across-seeds bootstrap CI to `aggregate/<model>_<arm>.json` +
   `aggregate/all_metadata.json`.

> **Disclosure (for the writeup):** configuration (LR, pooling) is selected on
> the single-seed (seed-42) validation split; variance is reported over 5 seeds
> of that fixed config, with per-class F1 thresholds re-optimized on validation
> per seed.

Stage 2 launches only after Stage 1 succeeds (`afterok`) as a Slurm **array**,
one single-GPU task per (model × seed × task).

---

## Arms

| Arm | Label | Probe | Classes | Thresholds |
|---|---|---|---|---|
| CT-RATE in-domain | `ctrate` | trained on CT-RATE train | 18 (CT-RATE) | per-seed val-optimized |
| RAD retrained | `rad_retrained` | trained fresh on RAD train | 16 (RAD, direct) | per-seed val-optimized |
| RAD frozen | `rad_frozen` | CT-RATE probe applied as-is | 18→16 mapped | CT-RATE per-seed (frozen) |

The **frozen arm** reuses the 5 CT-RATE-trained probes (one per seed, with that
seed's CT-RATE val thresholds) produced by the `ctrate` variance stage —
`probes/<model>_ctrate_seed<s>.pth` + `probes/<model>_ctrate_seed<s>_thresholds.npy`.
Frozen seed `s` = CT-RATE probe seed `s`.

Both RAD arms build the RAD test set via `dataloader_embeddings`
(`dataset_type: rad_chestct`, splits by `NoteAcc_DEID` prefix trn/val/tst) so the
**test set is identical** across the two arms (this also fixes the prior v628
frozen crash where `dataloader_rad_embeddings` matched 0 volumes).

---

## Models (4 frozen backbones, ViT-L, `input_dim=1024`)

Per model, three CLS-embedding dirs (DGX, container-mounted at `/app/project`):
CT-RATE train, CT-RATE valid, RAD-ChestCT. Embedding format: per-slice CLS
`(n_slices, 1024)` per volume (`.npy`/`.npz`) — so the pooling grid
(average / max / learned_attention) is meaningful as-is.

| Key | CT-RATE train | CT-RATE valid | RAD |
|---|---|---|---|
| `dale_ct_0` | `features/ctrate/lejepa_base/train_cls` | `features/ctrate/lejepa_base/valid_cls` | `features/rad/lejepa_base/cls` |
| `dale_ct_1s` | `features/ctrate/lejepa_3D_50k_swa_cropped_global/train` | `…/valid` | `features/rad/lejepa_3D_50k_swa_cropped_global/cls` |
| `dale_ct_2s` | `features/ctrate/lejepa_v2/train_cls` | `features/ctrate/lejepa_v2/valid_cls` | `features/rad/lejepa_v2/cls_ctrate_zscore` |
| `dinov2_finetuned` | `features/ctrate/dino_nongated/train` | `features/ctrate/dino_nongated/valid` | `features/rad/dinov2_finetuned/cls` |

All under `/project/ibi-staff/CT-JEPA/`. Label CSVs:
`Process_CT-RATE/dataset/train_predicted_labels.csv` (CT-RATE train+val draw),
`…/valid_predicted_labels.csv` (CT-RATE **test**), `…/RAD-ChestCT/rad_labels.csv`.

**Split conventions (discovered from the dataloader, not assumed):**
- CT-RATE test = `valid_predicted_labels.csv`.
- CT-RATE validation = fixed 1000-patient random draw (split seed 42) from
  `train_predicted_labels.csv` — used only for selection/thresholds.
- RAD splits by `NoteAcc_DEID` prefix: `trn`→train, `val`→valid, `tst`→test.

---

## Bootstrap

- `n_resamples = 2000`, 95% CI = 2.5 / 97.5 percentiles.
- **Shared indices:** resample indices are generated deterministically from a
  fixed per-task seed + `n_test` (`generate_bootstrap_indices`), so they are
  **identical across all models and seeds within a task**. This keeps CIs
  comparable and leaves paired differences available for later — **paired
  differences are not computed unless explicitly asked.**
  - CT-RATE task seed: `2024`; RAD task seed: `2025` (shared by frozen + retrained).
- Per resample, the **same `compute_metrics`** used for point eval is called on
  `y_true[idx], y_prob[idx], y_pred[idx]` — no new metric code.
- Per-run JSON records each seed's own CI; the **aggregate** bootstraps the
  **mean-across-5-seeds** metric (per resample: metric per seed, average,
  percentile) — the CI on the reported mean.

> **Baseline caveat (for the writeup):** the 3D baselines (COLIPRI, etc.) are
> taken from their papers as **point estimates with no per-scan predictions**, so
> comparisons to them are "our CI vs. their point estimate," **not paired**.

---

## Class-exclusion rule (reused, not redefined)

A class with `len(np.unique(y_true[:, i])) < 2` (degenerate in the split/resample)
is zeroed and excluded from macro F1/BA; macro AUROC/AUPRC use the all-or-nothing
`try/except` from `compute_metrics`. Each per-run JSON records
`excluded_classes_point` (degenerate in the point test set); the rule text is in
every JSON's `exclusion_rule` field.

---

## Outputs (written incrementally — a late array failure loses only its own run)

Root: `/project/ibi-staff/CT-JEPA/public/outputs/error_bars/`

```
selected_configs/<model>_<task>.json     Stage 1: {lr, pooling, val_auprc, thresholds_seed42 (audit), paths}
per_run/<model>_<arm>_seed<s>.json       Stage 2: point metrics + per-run CI + thresholds + metadata
per_run/<model>_<arm>_seed<s>.npz        Stage 2: raw y_true/y_prob/y_pred (for the aggregate's mean-across-seeds CI)
probes/<model>_ctrate_seed<s>.pth        Stage 2 (ctrate): probe weights feeding the frozen arm
probes/<model>_ctrate_seed<s>_thresholds.npy
aggregate/<model>_<arm>.json             Stage 3: seed mean±std + mean-across-seeds CI + per-seed records
aggregate/all_metadata.json              Stage 3: paths, classes, n per split, alignment, disclosure, caveats
```

**Per-run JSON** fields: `model, arm, seed, selected_config{lr,pooling},
thresholds, n_train/n_val/n_test, test_volume_names, macro{auroc,auprc,f1,ba},
per_class{name→{auroc,auprc,f1,ba,prevalence}}, excluded_classes_point,
exclusion_rule, bootstrap{macro,per_class}, bootstrap_settings, label_names,
embedding_paths, companion_npz`.

**Aggregate JSON** fields: `model, arm, seeds, n_seeds, seed_mean_std{macro,per_class}`,
`bootstrap_ci_mean_across_seeds{macro,per_class}`, `per_seed`, `metadata`.

---

## How to launch (DGX)

From this workspace, the flow is:
**macOS → push to DGX (you do this) → mount into container → `sbatch` array.**

1. Push the repo to the DGX so it lands at `/project/ibi-staff/CT-JEPA/public`.
2. On the DGX, from that directory, run the master submitter directly:
   ```bash
   bash run_error_bars_master.sh
   ```
   The master is a *submitter*, not a job — it has no `#SBATCH` directives of its
   own (no `--account`), so do **not** `sbatch` it (the cluster's account rule will
   reject it). Run it with `bash`; it fires off the six `sbatch` calls, each
   submitting an array that carries `#SBATCH --account=ibi-staff -p defq` (from
   `run_error_bars_array.sh`). It prints all six job IDs and exits in ~1 s — the
   work happens in the array jobs, which survive logout. The 6 dependent arrays
   (select ×2, variance ×2, frozen ×1, aggregate ×1) are wired with `afterok`.

Run a **single stage** manually (e.g. one model/seed) via the array script:
```bash
EB_MODE=variance_ctrate sbatch --array=0-0 run_error_bars_array.sh   # model 0, seed 0
```

`run_error_bars_array.sh` mirrors `run_v628_*.sh`: `srun` inside
`evandamron/lejepa:latest`, mounts `/project:/app/project`, syncs code
`$REPO_DIR → /app`, then runs `PYTHONPATH=/app python -u scripts/run_error_bars.py …`.

### Direct (non-Slurm) invocation
```bash
python scripts/run_error_bars.py --config configs/error_bars.yaml --mode select    --task ctrate        --model dale_ct_0
python scripts/run_error_bars.py --config configs/error_bars.yaml --mode variance  --task ctrate        --model dale_ct_0 --seed 0
python scripts/run_error_bars.py --config configs/error_bars.yaml --mode frozen                                       --model dale_ct_0 --seed 0
python scripts/run_error_bars.py --config configs/error_bars.yaml --mode aggregate
```

---

## Guardrails

- The seed controls all probe-training randomness (init + DataLoader shuffle);
  `ColipriProber` has **no dropout**, so seed = init + shuffle only. The split
  seed (42) is fixed separately so the validation split is identical across the
  5 training seeds. Seed is logged in every per-run JSON.
- Existing metrics + exclusion reused verbatim via `compute_metrics` /
  `map_ctrate_to_rad` — no new metric code.
- Incremental writes: each per-run JSON + `.npz` is written as soon as its
  (model, arm, seed) finishes; the aggregate is written last from per-run files.
- Per-task `n_test` is recorded per run and the aggregate's
  `shared_index_alignment` block flags any model whose test set differs (missing
  embeddings) — shared-index CIs are flagged, not silently paired.

## Resource notes (adjust to your partition)

Per-stage limits set in `run_error_bars_master.sh`:
- select / variance_ctrate: `--mem=128G` (full CT-RATE train preloaded into RAM)
- variance_rad_retrained: `--mem=64G`
- frozen_rad: `--mem=32G` (inference only)
- aggregate: `--mem=16G` (loads 60 `.npz` files + 2000×5 `compute_metrics` calls
  per (model, arm); expect ~20–40 min)

## Superseded v628 files (left in place by default)

The single-model v628 scripts/configs that did this exact eval task are
superseded by this pipeline but **left untouched** (the reused
`train_gridsearch.py` / `eval_rad.py` / dataloaders are shared, not bloat):
`run_v628_linear_probe_ctrate.sh`, `run_v628_rad_transfer_frozen.sh`,
`run_v628_linear_probe_rad_retrained.sh`, `run_v628_master.sh`,
`configs/v628/linear_probe_*.yaml`, `configs/v628/rad_transfer_frozen.yaml`.
Remove only if you confirm nothing else depends on them.
