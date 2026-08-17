#!/usr/bin/env python
"""
Parse Ablation Logs → Structured JSON Results
=============================================

Scans SLURM log files from run_ablation_gridsearch.sh and extracts:
  - Best pooling scheme and learning rate (from grid search)
  - Macro-averaged test metrics (AUPRC, AUROC, F1, Balanced Accuracy)
  - Per-class test metrics (AUPRC, AUROC, F1, BA, prevalence)

Saves one JSON per method into outputs/ablation_linear_probe/{method_name}/.

Usage:
  python scripts/parse_ablation_logs.py \
      --log-dir /project/ibi-staff/CT-JEPA/public/logs \
      --output-dir /project/ibi-staff/CT-JEPA/public/outputs/ablation_linear_probe

  # Dry-run: print what would be extracted without writing files
  python scripts/parse_ablation_logs.py --dry-run

  # Parse a single log file for debugging
  python scripts/parse_ablation_logs.py --single /path/to/log.out
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Regex patterns for parsing the log output
# ---------------------------------------------------------------------------

# "Array Task 0: Method = full_resolution"
RE_METHOD = re.compile(r"Array Task \d+: Method = (\S+)")

# "🚀 Starting Run: Pooling=[average], LR=[0.1]"
RE_RUN_START = re.compile(
    r"🚀 Starting Run: Pooling=\[(\S+)\], LR=\[([\d.eE+-]+)\]"
)

# "✅ Finished average (LR=0.1) - Best AUPRC: 0.4924"
RE_RUN_END = re.compile(
    r"✅ Finished (\S+) \(LR=([\d.eE+-]+)\) - Best AUPRC: ([\d.eE+-]+)"
)

# "Best Config: Pooling: learned_attention, LR: 0.01"
RE_BEST_CONFIG = re.compile(
    r"Best Config: Pooling: (\S+), LR: ([\d.eE+-]+)"
)

# "Best Global AUPRC: 0.5134"
RE_BEST_AUPRC = re.compile(r"Best Global AUPRC: ([\d.eE+-]+)")

# Macro table lines:
#   "AUPRC                | 0.5324     | ~0.1862 (Avg Prevalence)"
#   "AUROC                | 0.8232     | 0.5000"
#   "Macro F1             | 0.5216     | ~0.2544 (Avg Coin-Flip)"
#   "Balanced Acc (BA)    | 0.7059     | 0.5000"
RE_MACRO_LINE = re.compile(
    r"^(AUPRC|AUROC|Macro F1|Balanced Acc \(BA\))\s+\|\s+([\d.eE+-]+)"
)

# Per-class header: "🔸 MEDICAL MATERIAL"
RE_CLASS_HEADER = re.compile(r"🔸 (.+)")

# Per-class stat lines:
#   "   Prevalence: 0.0867 (8.7% of test set)"
#   "   - AUPRC: 0.4590  (Random: 0.0867)"
#   "   - AUROC: 0.8573  (Random: 0.5000)"
#   "   - F1:    0.4896  (Random: 0.1478)"
#   "   - BA:    0.7407  (Random: 0.5000)"
RE_PREVALENCE = re.compile(r"Prevalence:\s+([\d.eE+-]+)")
RE_AUPRC = re.compile(r"- AUPRC:\s+([\d.eE+-]+)")
RE_AUROC = re.compile(r"- AUROC:\s+([\d.eE+-]+)")
RE_F1 = re.compile(r"- F1:\s+([\d.eE+-]+)")
RE_BA = re.compile(r"- BA:\s+([\d.eE+-]+)")

# Sentinel marking the start of the final test evaluation
RE_FINAL_TEST_START = re.compile(r"STARTING FINAL EVALUATION ON UNSEEN TEST SET")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_single_log(log_path):
    """Parse one log file and return a dict with all extracted info."""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    result = {
        "log_file": str(log_path),
        "method": None,
        "grid_search": {
            "runs": [],
            "best_pooling": None,
            "best_lr": None,
            "best_val_auprc": None,
        },
        "test_macro": {},
        "test_per_class": {},
        "parse_errors": [],
    }

    # --- Phase 1: Find method name ---
    for line in lines:
        m = RE_METHOD.search(line)
        if m:
            result["method"] = m.group(1)
            break

    if result["method"] is None:
        result["parse_errors"].append("Could not find method name in log")
        return result

    # --- Phase 2: Parse grid search runs ---
    current_run = None
    for line in lines:
        # Start of a new run
        m_start = RE_RUN_START.search(line)
        if m_start:
            if current_run is not None:
                # Save previous run (in case it was never closed)
                result["grid_search"]["runs"].append(current_run)
            current_run = {
                "pooling": m_start.group(1),
                "lr": float(m_start.group(2)),
                "best_auprc": None,
            }
            continue

        # End of a run
        m_end = RE_RUN_END.search(line)
        if m_end and current_run is not None:
            current_run["best_auprc"] = float(m_end.group(3))
            result["grid_search"]["runs"].append(current_run)
            current_run = None
            continue

        # Best config summary
        m_best = RE_BEST_CONFIG.search(line)
        if m_best:
            result["grid_search"]["best_pooling"] = m_best.group(1)
            result["grid_search"]["best_lr"] = float(m_best.group(2))

        m_auprc = RE_BEST_AUPRC.search(line)
        if m_auprc:
            result["grid_search"]["best_val_auprc"] = float(m_auprc.group(1))

    # Don't forget the last run if still open
    if current_run is not None:
        result["grid_search"]["runs"].append(current_run)

    # --- Phase 3: Parse final test evaluation ---
    in_final_test = False
    in_macro_section = False
    in_per_class_section = False
    current_class_name = None
    current_class_data = {}

    for line in lines:
        if RE_FINAL_TEST_START.search(line):
            in_final_test = True
            continue

        if not in_final_test:
            continue

        # Detect macro section
        if "FINAL TEST METRICS (MACRO AVERAGES)" in line:
            in_macro_section = True
            in_per_class_section = False
            continue

        # Detect per-class section
        if "PER-CLASS METRICS" in line:
            in_macro_section = False
            in_per_class_section = True
            continue

        # End of per-class section (the "--- Finished" line, NOT the "-----" separators)
        if in_per_class_section and line.strip().startswith("---") and "Finished" in line:
            # Save last class
            if current_class_name is not None and current_class_data:
                result["test_per_class"][current_class_name] = current_class_data
            in_per_class_section = False
            continue

        if in_macro_section:
            m = RE_MACRO_LINE.search(line)
            if m:
                key = m.group(1).strip()
                val = float(m.group(2))
                # Normalize keys
                if key == "AUPRC":
                    result["test_macro"]["auprc"] = val
                elif key == "AUROC":
                    result["test_macro"]["auroc"] = val
                elif key == "Macro F1":
                    result["test_macro"]["macro_f1"] = val
                elif key == "Balanced Acc (BA)":
                    result["test_macro"]["balanced_accuracy"] = val

        if in_per_class_section:
            # Class header
            m_cls = RE_CLASS_HEADER.search(line)
            if m_cls:
                # Save previous class
                if current_class_name is not None and current_class_data:
                    result["test_per_class"][current_class_name] = current_class_data
                current_class_name = m_cls.group(1).strip()
                current_class_data = {}
                continue

            if current_class_name is not None:
                m_prev = RE_PREVALENCE.search(line)
                if m_prev:
                    current_class_data["prevalence"] = float(m_prev.group(1))

                m_auprc = RE_AUPRC.search(line)
                if m_auprc:
                    current_class_data["auprc"] = float(m_auprc.group(1))

                m_auroc = RE_AUROC.search(line)
                if m_auroc:
                    current_class_data["auroc"] = float(m_auroc.group(1))

                m_f1 = RE_F1.search(line)
                if m_f1:
                    current_class_data["f1"] = float(m_f1.group(1))

                m_ba = RE_BA.search(line)
                if m_ba:
                    current_class_data["ba"] = float(m_ba.group(1))

    # Save last class if still open
    if current_class_name is not None and current_class_data:
        result["test_per_class"][current_class_name] = current_class_data

    # --- Validation ---
    if not result["test_macro"]:
        result["parse_errors"].append("No macro test metrics found")
    if not result["test_per_class"]:
        result["parse_errors"].append("No per-class test metrics found")
    if result["grid_search"]["best_pooling"] is None:
        result["parse_errors"].append("No best config found in grid search")

    return result


def find_log_files(log_dir, pattern="ablation_ablation_probe-*.out"):
    """Find all matching log files in the given directory."""
    log_path = Path(log_dir)
    if not log_path.is_dir():
        print(f"ERROR: Log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(log_path.glob(pattern))
    if not files:
        print(f"WARNING: No log files matching '{pattern}' found in {log_dir}",
              file=sys.stderr)
    return files


def print_summary(results):
    """Print a human-readable summary table of all methods."""
    print("\n" + "=" * 90)
    print(f"{'Method':<30} {'Best Pool':<20} {'Best LR':<10} {'Val AUPRC':<12} {'Test AUPRC':<12}")
    print("-" * 90)

    for method_name in sorted(results.keys()):
        r = results[method_name]
        gs = r.get("grid_search", {})
        tm = r.get("test_macro", {})
        print(
            f"{method_name:<30} "
            f"{gs.get('best_pooling', 'N/A'):<20} "
            f"{gs.get('best_lr', 'N/A'):<10} "
            f"{gs.get('best_val_auprc', 'N/A'):<12} "
            f"{tm.get('auprc', 'N/A'):<12}"
        )

    print("=" * 90)

    # Print any parse errors
    errors_found = False
    for method_name in sorted(results.keys()):
        errs = results[method_name].get("parse_errors", [])
        if errs:
            if not errors_found:
                print("\n⚠️  Parse warnings:")
                errors_found = True
            for e in errs:
                print(f"  [{method_name}] {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse ablation grid search logs into structured JSON"
    )
    parser.add_argument(
        "--log-dir", type=str,
        default="/project/ibi-staff/CT-JEPA/public/logs",
        help="Directory containing ablation log files"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/project/ibi-staff/CT-JEPA/public/outputs/ablation_linear_probe",
        help="Root directory for saving parsed results (one subdir per method)"
    )
    parser.add_argument(
        "--single", type=str, default=None,
        help="Parse a single log file (for debugging)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and print summary without writing files"
    )
    args = parser.parse_args()

    # --- Collect log files ---
    if args.single:
        log_files = [Path(args.single)]
    else:
        log_files = find_log_files(args.log_dir)

    if not log_files:
        print("No log files to process. Exiting.")
        sys.exit(0)

    print(f"Found {len(log_files)} log file(s) to process.")

    # --- Parse all logs ---
    results = {}  # method_name → parsed dict

    for lf in log_files:
        print(f"  Parsing: {lf.name} ...")
        parsed = parse_single_log(str(lf))
        method = parsed["method"]
        if method is None:
            print(f"    ⚠️  Skipping {lf.name} — could not determine method")
            continue
        results[method] = parsed

    if not results:
        print("No results extracted. Exiting.")
        sys.exit(1)

    # --- Print summary ---
    print_summary(results)

    # --- Write output files ---
    if not args.dry_run:
        output_root = Path(args.output_dir)
        for method_name, data in results.items():
            method_dir = output_root / method_name
            method_dir.mkdir(parents=True, exist_ok=True)

            out_path = method_dir / "test_metrics.json"
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"  ✅ Wrote: {out_path}")

        print(f"\nDone! Results saved to {output_root}")
    else:
        print("\n[DRY RUN] No files written.")


if __name__ == "__main__":
    main()
