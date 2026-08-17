"""
Error-bars orchestration for the frozen-backbone linear-probe evaluation.

Wraps the existing COLIPRI probe logic (train_gridsearch.train_one_config /
compute_metrics, eval_rad.map_ctrate_to_rad) — adds only the seed loop,
bootstrap, and JSON persistence the DALE paper's error-bar tables need.

Modes
-----
  select    Stage 1: grid search (1 seed) over pooling × LR, select on val
            AUPRC, write selected_configs/<model>_<task>.json.
  variance  Stage 2: train ONE fixed config under --seed, eval on test,
            bootstrap (shared indices), write per_run/<model>_<arm>_seed<s>.json.
            For task=ctrate also writes the probe + thresholds for the frozen arm.
  frozen    Stage 2: apply the CT-RATE-huggingface-downloads-trained probe (this seed) to RAD test via
            the 18→16 mapping, bootstrap, write per_run/<model>_rad_frozen_seed<s>.json.
  aggregate Stage 3: read per_run/*.json, write seed mean±std + bootstrap CIs to
            aggregate/<model>_<arm>.json + aggregate/all_metadata.json.

See configs/error_bars.yaml and ERROR_BARS_README.md.
"""
import os
import sys
import copy
import json
import glob
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import DataLoader

# scripts/ lives one level below the repo root; put the repo root on sys.path so
# the sibling packages (dataloaders, models, train_gridsearch, eval_rad) import
# whether run as `python scripts/run_error_bars.py` or with PYTHONPATH set.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reusable logic from the existing pipeline (imported, not re-derived).
from dataloaders.dataloader_embeddings import create_datasets, collate_mil_bags
from models.mil_pooling import build_prober
from train_gridsearch import train_one_config, compute_metrics
from eval_rad import map_ctrate_to_rad

# Metric key normalization: compute_metrics reports macro_auc/auprc/f1/ba and
# per_class auroc/auprc/f1/ba. We normalize everything to auroc/auprc/f1/ba.
METRIC_KEYS = ["auroc", "auprc", "f1", "ba"]
MACRO_MAP = {"auroc": "macro_auc", "auprc": "macro_auprc", "f1": "macro_f1", "ba": "macro_ba"}

EXCLUSION_RULE = ("len(np.unique(y_true[:, i])) < 2 -> class zeroed and excluded "
                  "from macro F1/BA (lineage-1 rule, via compute_metrics)")
DISCLOSURE = ("Configuration (LR, pooling) selected on the single-seed (seed-42) "
              "validation split; variance reported over 5 seeds of that fixed "
              "config, with per-class F1 thresholds re-optimized on validation "
              "per seed.")
BASELINE_CAVEAT = ("3D baselines (COLIPRI et al.) are taken from their papers as "
                   "point estimates with no per-scan predictions, so comparisons "
                   "are 'our CI vs. their point estimate', not paired.")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_config(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=json_default)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def generate_bootstrap_indices(n_test, n_resamples, seed):
    """Deterministic resample index matrix (n_resamples, n_test).

    Fixed per-task seed + identical n_test => identical indices across all
    models/seeds within a task, so CIs are comparable and paired differences
    stay available. No shared file needed.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_test, size=(n_resamples, n_test))


def macro_point(m):
    return {k: m[MACRO_MAP[k]] for k in METRIC_KEYS}


def perclass_point(m, label_names):
    out = {}
    for name in label_names:
        pc = m["per_class"][name]
        out[name] = {k: pc[k] for k in METRIC_KEYS}
        out[name]["prevalence"] = pc["prevalence"]
    return out


def excluded_classes(y_true, label_names):
    """Classes degenerate in the POINT test set (len(unique)<2). Reporting only;
    uses the same rule as compute_metrics but does not alter metrics."""
    return [name for i, name in enumerate(label_names)
            if len(np.unique(y_true[:, i])) < 2]


def bootstrap_ci(y_true, y_prob, y_pred, label_names, n_resamples, seed, ci_pct):
    """Per-run 95% CI: resample test set (shared indices), recompute metrics via
    the same compute_metrics used for point eval. Returns (macro_ci, perclass_ci)."""
    n_test = y_true.shape[0]
    indices = generate_bootstrap_indices(n_test, n_resamples, seed)
    macro_dist = {k: np.empty(n_resamples) for k in METRIC_KEYS}
    pc_dist = {name: {k: np.empty(n_resamples) for k in METRIC_KEYS} for name in label_names}
    for r, idx in enumerate(indices):
        m = compute_metrics(y_true[idx], y_prob[idx], y_pred[idx], label_names)
        for k in METRIC_KEYS:
            macro_dist[k][r] = m[MACRO_MAP[k]]
        for name in label_names:
            for k in METRIC_KEYS:
                pc_dist[name][k][r] = m["per_class"][name][k]
    macro_ci = {k: [float(np.percentile(macro_dist[k], ci_pct[0])),
                    float(np.percentile(macro_dist[k], ci_pct[1]))] for k in METRIC_KEYS}
    perclass_ci = {name: {k: [float(np.percentile(pc_dist[name][k], ci_pct[0])),
                              float(np.percentile(pc_dist[name][k], ci_pct[1]))]
                          for k in METRIC_KEYS} for name in label_names}
    return macro_ci, perclass_ci


def mean_across_seeds_ci(seed_arrays, label_names, n_resamples, task_seed, ci_pct):
    """Bootstrap CI on the MEAN-across-5-seeds metric: per resample, compute the
    metric for each seed, average, then percentile. Requires shared indices
    (same task_seed + n_test) and identical y_true across seeds."""
    y_true0 = seed_arrays[0]["y_true"]
    n_test = y_true0.shape[0]
    for sa in seed_arrays:
        assert sa["y_true"].shape == y_true0.shape, "n_test mismatch across seeds"
        assert np.array_equal(sa["y_true"], y_true0), "y_true differs across seeds (test set changed)"
    indices = generate_bootstrap_indices(n_test, n_resamples, task_seed)
    macro_dist = {k: np.empty(n_resamples) for k in METRIC_KEYS}
    pc_dist = {name: {k: np.empty(n_resamples) for k in METRIC_KEYS} for name in label_names}
    for r, idx in enumerate(indices):
        per_seed_macro = {k: [] for k in METRIC_KEYS}
        per_seed_pc = {name: {k: [] for k in METRIC_KEYS} for name in label_names}
        for sa in seed_arrays:
            m = compute_metrics(sa["y_true"][idx], sa["y_prob"][idx], sa["y_pred"][idx], label_names)
            for k in METRIC_KEYS:
                per_seed_macro[k].append(m[MACRO_MAP[k]])
            for name in label_names:
                for k in METRIC_KEYS:
                    per_seed_pc[name][k].append(m["per_class"][name][k])
        for k in METRIC_KEYS:
            macro_dist[k][r] = float(np.mean(per_seed_macro[k]))
        for name in label_names:
            for k in METRIC_KEYS:
                pc_dist[name][k][r] = float(np.mean(per_seed_pc[name][k]))
    macro_ci = {k: [float(np.percentile(macro_dist[k], ci_pct[0])),
                    float(np.percentile(macro_dist[k], ci_pct[1]))] for k in METRIC_KEYS}
    perclass_ci = {name: {k: [float(np.percentile(pc_dist[name][k], ci_pct[0])),
                              float(np.percentile(pc_dist[name][k], ci_pct[1]))]
                          for k in METRIC_KEYS} for name in label_names}
    return macro_ci, perclass_ci


def run_inference(model, loader, device):
    """Forward probe over a loader, collect (y_prob, y_true, names). Thin
    collector around ColipriProber — no metric code."""
    model.eval()
    all_probs, all_targets, all_names = [], [], []
    with torch.no_grad():
        for features, labels, names, mask in loader:
            features, mask = features.to(device), mask.to(device)
            logits, _ = model(features, mask=mask)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            all_names.extend(names)
    return np.vstack(all_probs), np.vstack(all_targets), all_names


# --------------------------------------------------------------------------- #
# Config / dataset wiring
# --------------------------------------------------------------------------- #
def label_names_for(config, task):
    """EVAL label space = columns in y_true. ctrate -> 18 CT-RATE-huggingface-downloads classes; both
    RAD arms -> 16 RAD-ChestCT classes (frozen evals on the 16 mapped classes
    even though the probe outputs 18)."""
    if task == "ctrate":
        return list(config["experiment"]["ct_rate_classes"])
    return list(config["experiment"]["rad_chestct_classes"])


def num_classes_for(config, task):
    """MODEL output space. ctrate + rad_frozen -> 18 (CT-RATE-huggingface-downloads probe); rad_retrained -> 16."""
    if task in ("ctrate", "rad_frozen"):
        return len(config["experiment"]["ct_rate_classes"])
    return len(config["experiment"]["rad_chestct_classes"])


def build_config_for_task(config, task, model_key=None):
    """Deep copy with per-task num_classes (model output space). Keeps split_seed
    for the val draw so the validation split is identical across all 5 seeds.

    Per-model input_dim override: benchmark models have heterogeneous embed dims
    (768/1024/864/512/2048); a model-level `input_dim` in config["models"][key]
    overrides the global 1024 default so the probe (and reloaded frozen probe)
    use the real dimension. DALE-CT models have no override -> fall back to global.
    """
    cfg = copy.deepcopy(config)
    exp = cfg["experiment"]
    exp["num_classes"] = num_classes_for(config, task)
    exp["seed"] = exp["split_seed"]  # CT-RATE-huggingface-downloads val draw sees the same patients every seed
    if model_key is not None:
        exp["input_dim"] = config["models"][model_key].get("input_dim", exp["input_dim"])
    return cfg


def build_datasets(config, model_key, task):
    """Route to dataloader_embeddings.create_datasets with this model's paths.
    RAD arms both use dataset_type=rad_chestct so the RAD test set is identical
    (and the v628 frozen 0-volume crash is avoided)."""
    cfg = copy.deepcopy(config)
    mp = config["models"][model_key]
    base = dict(config["data"])
    if task == "ctrate":
        cfg["data"] = {**base, "dataset_type": "ct_rate",
                       "train_embedding_dir": mp["ctrate_train"],
                       "val_embedding_dir": mp["ctrate_valid"]}
    else:  # rad_retrained or rad_frozen
        cfg["data"] = {**base, "dataset_type": "rad_chestct",
                       "rad_embedding_dir": mp["rad"]}
    cfg["experiment"]["seed"] = config["experiment"]["split_seed"]
    return create_datasets(cfg)


def make_loader(ds, config, shuffle, generator=None):
    return DataLoader(ds, batch_size=config["experiment"]["batch_size"], shuffle=shuffle,
                      collate_fn=collate_mil_bags,
                      num_workers=config["experiment"].get("num_workers", 2),
                      generator=generator)


def task_seed_for(config, arm):
    bs = config["bootstrap"]
    return bs["ctrate_seed"] if arm == "ctrate" else bs["rad_seed"]


def out_paths(config):
    root = config["output"]["root"]
    return {k: os.path.join(root, config["output"][k])
            for k in ("selected_configs", "per_run", "probes", "aggregate")}


# --------------------------------------------------------------------------- #
# Stage 1: select
# --------------------------------------------------------------------------- #
def run_select(config, model_key, task, device):
    cfg = build_config_for_task(config, task, model_key)
    names = label_names_for(config, task)
    train_ds, val_ds, _ = build_datasets(config, model_key, task)
    # val_loader is shared across the concurrent LR threads. Safe under
    # num_workers=0: DataLoader.__iter__ returns a fresh iterator each call, so
    # each thread iterates the read-only preloaded cache independently.
    val_loader = make_loader(val_ds, config, shuffle=False)

    split_seed = config["experiment"]["split_seed"]
    lrs = config["experiment"]["learning_rates"]
    poolings = config["experiment"]["pooling_schemes"]

    # Serializes set_seed + ColipriProber construction (global RNG) across the
    # concurrent LR threads — see train_one_config(init_lock=...).
    init_lock = threading.Lock()
    all_results = []

    for pooling in poolings:
        print(f"\n🚀 select [{model_key}/{task}] pooling={pooling} "
              f"(running {len(lrs)} LRs in parallel)")

        def run_one(lr, _pooling=pooling):
            # Per-config train loader with its OWN generator, seeded to split_seed
            # so the concurrent training loops don't race on the global RNG during
            # shuffle. Every config seeds to the same split_seed, so each sees the
            # identical init weights + shuffle sequence it would sequentially.
            gen = torch.Generator()
            gen.manual_seed(split_seed)
            train_loader = make_loader(train_ds, config, shuffle=True, generator=gen)
            res = train_one_config(train_loader, val_loader, lr, _pooling, cfg, device,
                                   split_seed, names, init_lock=init_lock)
            print(f"   -> [{_pooling}/lr={lr}] val AUPRC={res['best_val_auprc']:.4f}")
            return res

        with ThreadPoolExecutor(max_workers=len(lrs)) as ex:
            results = list(ex.map(run_one, lrs))
        for lr, res in zip(lrs, results):
            all_results.append((pooling, lr, res))

    best = None
    for pooling, lr, res in all_results:
        if best is None or res["best_val_auprc"] > best["best_val_auprc"]:
            best = {**res, "pooling": pooling, "lr": lr}

    op = out_paths(config)
    rec = {
        "model": model_key, "task": task,
        "lr": best["lr"], "pooling": best["pooling"],
        "input_dim": cfg["experiment"]["input_dim"],
        "num_classes": cfg["experiment"]["num_classes"],
        "pooling_mode": cfg["experiment"]["pooling_mode"],
        "val_auprc": best["best_val_auprc"],
        "thresholds_seed42": best["best_thresholds"].tolist(),  # audit only
        "label_names": names,
        "embedding_paths": config["models"][model_key],
        "selection_seed": split_seed,
    }
    path = os.path.join(op["selected_configs"], f"{model_key}_{task}.json")
    write_json(path, rec)
    print(f"\n✅ selected [{model_key}/{task}]: pooling={best['pooling']} lr={best['lr']} "
          f"val_AUPRC={best['best_val_auprc']:.4f} -> {path}")
    return rec


# --------------------------------------------------------------------------- #
# Stage 2: variance (ctrate | rad_retrained)
# --------------------------------------------------------------------------- #
def run_variance(config, model_key, task, seed, device):
    op = out_paths(config)
    sel = _read_json(os.path.join(op["selected_configs"], f"{model_key}_{task}.json"))
    cfg = build_config_for_task(config, task, model_key)
    names = label_names_for(config, task)

    train_ds, val_ds, test_ds = build_datasets(config, model_key, task)
    train_loader = make_loader(train_ds, config, shuffle=True)
    val_loader = make_loader(val_ds, config, shuffle=False)
    test_loader = make_loader(test_ds, config, shuffle=False)

    res = train_one_config(train_loader, val_loader, sel["lr"], sel["pooling"], cfg,
                           device, seed, names)
    thresholds = res["best_thresholds"]  # val-optimized at the best step, per seed

    # Build probe, load best-step weights, infer on test.
    model = build_prober(input_dim=cfg["experiment"]["input_dim"],
                         num_classes=cfg["experiment"]["num_classes"],
                         pooling_scheme=sel["pooling"],
                         pooling_mode=cfg["experiment"]["pooling_mode"],
                         config=config).to(device)
    model.load_state_dict(res["best_model_state"])
    y_prob, y_true, test_names = run_inference(model, test_loader, device)
    y_pred = (y_prob >= thresholds).astype(int)

    m = compute_metrics(y_true, y_prob, y_pred, names)
    arm = task  # ctrate | rad_retrained
    bs = config["bootstrap"]
    macro_ci, pc_ci = bootstrap_ci(y_true, y_prob, y_pred, names, bs["n_resamples"],
                                   task_seed_for(config, arm), bs["ci_percentiles"])

    # Companion .npz (raw preds) so aggregate can bootstrap the mean-across-seeds.
    npz_path = os.path.join(op["per_run"], f"{model_key}_{arm}_seed{seed}.npz")
    ensure_dir(os.path.dirname(npz_path))  # np.savez does not create parent dirs
    np.savez(npz_path, y_true=y_true, y_prob=y_prob, y_pred=y_pred)

    rec = _per_run_record(config, model_key, arm, seed, sel, thresholds, m, names,
                          y_true, test_names, macro_ci, pc_ci, bs, npz_path,
                          n_train=len(train_ds), n_val=len(val_ds), n_test=len(test_ds))
    write_json(os.path.join(op["per_run"], f"{model_key}_{arm}_seed{seed}.json"), rec)

    # ctrate probes feed the frozen arm.
    if task == "ctrate":
        ensure_dir(op["probes"])  # torch.save / np.save do not create parent dirs
        torch.save(res["best_model_state"], os.path.join(op["probes"], f"{model_key}_ctrate_seed{seed}.pth"))
        np.save(os.path.join(op["probes"], f"{model_key}_ctrate_seed{seed}_thresholds.npy"), thresholds)

    print(f"✅ variance [{model_key}/{arm} seed{seed}] "
          f"AUPRC={macro_point(m)['auprc']:.4f} AUROC={macro_point(m)['auroc']:.4f}")
    return rec


# --------------------------------------------------------------------------- #
# Stage 2: frozen (apply CT-RATE-huggingface-downloads probe to RAD via 18->16 mapping)
# --------------------------------------------------------------------------- #
def run_frozen(config, model_key, seed, device):
    op = out_paths(config)
    sel_ctrate = _read_json(os.path.join(op["selected_configs"], f"{model_key}_ctrate.json"))
    rad_names = label_names_for(config, "rad_frozen")
    cfg = build_config_for_task(config, "rad_frozen", model_key)  # num_classes=18 (CT-RATE-huggingface-downloads probe)

    probe_path = os.path.join(op["probes"], f"{model_key}_ctrate_seed{seed}.pth")
    thresh_path = os.path.join(op["probes"], f"{model_key}_ctrate_seed{seed}_thresholds.npy")
    if not os.path.exists(probe_path) or not os.path.exists(thresh_path):
        raise FileNotFoundError(f"Frozen arm needs CT-RATE-huggingface-downloads probe for {model_key} seed{seed}: "
                                f"{probe_path} / {thresh_path}")

    model = build_prober(input_dim=cfg["experiment"]["input_dim"],
                         num_classes=cfg["experiment"]["num_classes"],  # 18
                         pooling_scheme=sel_ctrate["pooling"],
                         pooling_mode=cfg["experiment"]["pooling_mode"],
                         config=config).to(device)
    model.load_state_dict(torch.load(probe_path, map_location=device))
    fixed_thresholds = np.load(thresh_path)  # 18-dim, this seed's val thresholds

    _, _, test_ds = build_datasets(config, model_key, "rad_frozen")
    test_loader = make_loader(test_ds, config, shuffle=False)
    y_prob_18, y_true_16, test_names = run_inference(model, test_loader, device)

    y_prob_16, y_pred_16 = map_ctrate_to_rad(y_prob_18, fixed_thresholds,
                                             cfg["experiment"]["ct_rate_classes"], rad_names)
    m = compute_metrics(y_true_16, y_prob_16, y_pred_16, rad_names)

    arm = "rad_frozen"
    bs = config["bootstrap"]
    macro_ci, pc_ci = bootstrap_ci(y_true_16, y_prob_16, y_pred_16, rad_names, bs["n_resamples"],
                                   task_seed_for(config, arm), bs["ci_percentiles"])
    npz_path = os.path.join(op["per_run"], f"{model_key}_{arm}_seed{seed}.npz")
    ensure_dir(os.path.dirname(npz_path))  # np.savez does not create parent dirs
    np.savez(npz_path, y_true=y_true_16, y_prob=y_prob_16, y_pred=y_pred_16)

    rec = _per_run_record(config, model_key, arm, seed, sel_ctrate, fixed_thresholds, m,
                          rad_names, y_true_16, test_names, macro_ci, pc_ci, bs, npz_path,
                          n_train=None, n_val=None, n_test=len(test_ds))
    rec["frozen_thresholds_source"] = f"{model_key}_ctrate_seed{seed}_thresholds.npy (18-dim, mapped)"
    write_json(os.path.join(op["per_run"], f"{model_key}_{arm}_seed{seed}.json"), rec)
    print(f"✅ frozen [{model_key} seed{seed}] "
          f"AUPRC={macro_point(m)['auprc']:.4f} AUROC={macro_point(m)['auroc']:.4f}")
    return rec


def _per_run_record(config, model_key, arm, seed, sel, thresholds, m, names,
                    y_true, test_names, macro_ci, pc_ci, bs, npz_path,
                    n_train, n_val, n_test):
    return {
        "model": model_key, "arm": arm, "seed": seed,
        "selected_config": {"lr": sel["lr"], "pooling": sel["pooling"]},
        "thresholds": np.asarray(thresholds).tolist(),
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "test_volume_names": test_names,
        "macro": macro_point(m),
        "per_class": perclass_point(m, names),
        "excluded_classes_point": excluded_classes(y_true, names),
        "exclusion_rule": EXCLUSION_RULE,
        "bootstrap": {"macro": macro_ci, "per_class": pc_ci},
        "bootstrap_settings": {
            "n_resamples": bs["n_resamples"],
            "seed": task_seed_for(config, arm),
            "ci_percentiles": bs["ci_percentiles"],
            "shared_indices": "fixed per-task seed + n_test => identical resamples across models/seeds within a task",
        },
        "label_names": names,
        "embedding_paths": config["models"][model_key],
        "companion_npz": os.path.basename(npz_path),
    }


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Stage 3: aggregate
# --------------------------------------------------------------------------- #
def run_aggregate(config):
    op = out_paths(config)
    ensure_dir(op["aggregate"])
    bs = config["bootstrap"]

    runs = {}  # (model, arm) -> [records]
    for f in sorted(glob.glob(os.path.join(op["per_run"], "*.json"))):
        rec = _read_json(f)
        runs.setdefault((rec["model"], rec["arm"]), []).append(rec)

    if not runs:
        print("⚠️  no per_run JSONs found; nothing to aggregate.")
        return

    n_test_by_taskarm = {}  # (model, arm) -> n_test for alignment report
    for (model, arm), recs in sorted(runs.items()):
        recs.sort(key=lambda r: r["seed"])
        names = recs[0]["label_names"]
        n_seeds = len(recs)

        # Seed mean ± std from point metrics (JSON only).
        seed_mean_std = {"macro": {}, "per_class": {}}
        for k in METRIC_KEYS:
            vals = np.array([r["macro"][k] for r in recs], dtype=float)
            seed_mean_std["macro"][k] = {"mean": float(np.mean(vals)),
                                         "std": float(np.std(vals, ddof=1)) if n_seeds > 1 else 0.0}
        for name in names:
            seed_mean_std["per_class"][name] = {}
            for k in METRIC_KEYS:
                vals = np.array([r["per_class"][name][k] for r in recs], dtype=float)
                seed_mean_std["per_class"][name][k] = {"mean": float(np.mean(vals)),
                                                       "std": float(np.std(vals, ddof=1)) if n_seeds > 1 else 0.0}

        # Mean-across-seeds bootstrap CI (needs companion .npz per seed).
        macro_ci_mean, pc_ci_mean = None, None
        seed_arrays = []
        for r in recs:
            npz = os.path.join(op["per_run"], r["companion_npz"])
            if not os.path.exists(npz):
                break
            z = np.load(npz)
            seed_arrays.append({"y_true": z["y_true"], "y_prob": z["y_prob"], "y_pred": z["y_pred"]})
        if len(seed_arrays) == n_seeds:
            macro_ci_mean, pc_ci_mean = mean_across_seeds_ci(
                seed_arrays, names, bs["n_resamples"], task_seed_for(config, arm), bs["ci_percentiles"])
        else:
            print(f"⚠️  [{model}/{arm}] missing companion .npz for some seeds; "
                  f"skipping mean-across-seeds CI (seed mean±std still computed).")

        n_test_by_taskarm[(model, arm)] = recs[0]["n_test"]
        agg = {
            "model": model, "arm": arm,
            "seeds": [r["seed"] for r in recs], "n_seeds": n_seeds,
            "seed_mean_std": seed_mean_std,
            "bootstrap_ci_mean_across_seeds": {"macro": macro_ci_mean, "per_class": pc_ci_mean},
            "per_seed": [{"seed": r["seed"], "macro": r["macro"], "per_class": r["per_class"],
                          "bootstrap_ci": r["bootstrap"]} for r in recs],
            "label_names": names,
            "metadata": _metadata_block(config, model, arm, recs[0]["n_test"]),
        }
        write_json(os.path.join(op["aggregate"], f"{model}_{arm}.json"), agg)
        print(f"✅ aggregate [{model}/{arm}] {n_seeds} seeds -> "
              f"{os.path.join(op['aggregate'], model + '_' + arm + '.json')}")

    _write_all_metadata(config, runs, n_test_by_taskarm)


def _metadata_block(config, model, arm, n_test):
    bs = config["bootstrap"]
    return {
        "embedding_paths": config["models"][model],
        "class_names_order": label_names_for(config, arm),
        "n_test": n_test,
        "n_seeds": len(config["experiment"]["seeds"]),
        "n_bootstrap": bs["n_resamples"],
        "bootstrap_seed": task_seed_for(config, arm),
        "shared_indices_note": "identical resample indices across all models/seeds within a task (fixed seed + n_test)",
        "exclusion_rule": EXCLUSION_RULE,
        "disclosure": DISCLOSURE,
        "baseline_caveat": BASELINE_CAVEAT,
    }


def _write_all_metadata(config, runs, n_test_by_taskarm):
    op = out_paths(config)
    bs = config["bootstrap"]
    # Shared-index alignment: within each task, all models must share n_test.
    by_task = {}  # task -> {model: n_test}
    for (model, arm), n_test in n_test_by_taskarm.items():
        task = "ctrate" if arm == "ctrate" else "rad"
        by_task.setdefault(task, {})[model] = n_test
    alignment = {}
    for task, m in by_task.items():
        vals = set(m.values())
        alignment[task] = {"n_test_per_model": m, "aligned": len(vals) == 1,
                           "note": "RAD arms share the RAD test set; CT-RATE-huggingface-downloads shares the valid set."}

    meta = {
        "models": config["models"],
        "arms": ["ctrate", "rad_frozen", "rad_retrained"],
        "ct_rate_classes": config["experiment"]["ct_rate_classes"],
        "rad_chestct_classes": config["experiment"]["rad_chestct_classes"],
        "grid": {"pooling_schemes": config["experiment"]["pooling_schemes"],
                 "learning_rates": config["experiment"]["learning_rates"]},
        "seeds": config["experiment"]["seeds"],
        "split_seed": config["experiment"]["split_seed"],
        "n_bootstrap": bs["n_resamples"],
        "bootstrap_seeds": {"ctrate": bs["ctrate_seed"], "rad": bs["rad_seed"]},
        "ci_percentiles": bs["ci_percentiles"],
        "shared_index_alignment": alignment,
        "exclusion_rule": EXCLUSION_RULE,
        "disclosure": DISCLOSURE,
        "baseline_caveat": BASELINE_CAVEAT,
        "input_dim": config["experiment"]["input_dim"],
        "num_classes": {"ctrate": len(config["experiment"]["ct_rate_classes"]),
                        "rad_frozen": len(config["experiment"]["ct_rate_classes"]),
                        "rad_retrained": len(config["experiment"]["rad_chestct_classes"])},
    }
    write_json(os.path.join(op["aggregate"], "all_metadata.json"), meta)
    print(f"✅ aggregate metadata -> {os.path.join(op['aggregate'], 'all_metadata.json')}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/error_bars.yaml")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["select", "variance", "frozen", "aggregate"])
    parser.add_argument("--task", type=str, default=None, choices=["ctrate", "rad_retrained"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not args.mode:
        parser.error("--mode is required")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "select":
        if not args.task or not args.model:
            parser.error("select requires --task and --model")
        run_select(config, args.model, args.task, device)

    elif args.mode == "variance":
        if not args.task or not args.model or args.seed is None:
            parser.error("variance requires --task, --model, --seed")
        run_variance(config, args.model, args.task, args.seed, device)

    elif args.mode == "frozen":
        if not args.model or args.seed is None:
            parser.error("frozen requires --model, --seed")
        run_frozen(config, args.model, args.seed, device)

    elif args.mode == "aggregate":
        run_aggregate(config)


if __name__ == "__main__":
    main()
