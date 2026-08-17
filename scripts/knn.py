import argparse
import os
import glob
import logging
import yaml
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Configure extensive logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def get_base_volume_name(volume_filename):
    return str(os.path.splitext(os.path.splitext(volume_filename)[0])[0])


def get_scan_group(volume_name):
    """
    Extracts the base scan group to identify duplicate reconstructions.
    e.g., 'train_189_a_1' -> 'train_189_a'
    """
    parts = volume_name.split('_')
    # If the last segment is just digits (the reconstruction number), strip it
    if len(parts) > 1 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return volume_name


def load_all_embeddings(emb_dir):
    """Loads and pools all embeddings in a directory into memory once."""
    emb_dict = {}
    file_paths = glob.glob(os.path.join(emb_dir, "*.npy"))

    logger.info(f"Discovered {len(file_paths)} '.npy' files in '{emb_dir}'.")

    if not file_paths:
        logger.warning(f"No embeddings found in '{emb_dir}'.")
        return emb_dict

    for emb_path in tqdm(file_paths, desc="Loading Embeddings", leave=False, unit="file"):
        base_name = get_base_volume_name(os.path.basename(emb_path))
        try:
            emb = np.load(emb_path)

            # Mean pool slice-level (2D) embeddings to volume-level (1D)
            if emb.ndim == 2:
                emb = np.mean(emb, axis=0)
            elif emb.ndim > 2:
                logger.debug(f"Skipped {base_name}: Unsupported dimensions ({emb.ndim}D).")
                continue

            emb_dict[base_name] = emb
        except Exception as e:
            logger.error(f"Failed to load {emb_path}: {e}")

    logger.info(f"Successfully loaded and pooled {len(emb_dict)} embeddings into memory.")
    return emb_dict


def evaluate_knn_cv(X, y, keys, k_values=[1, 5], n_splits=5):
    """Evaluates KNN using Stratified Group K-Fold CV and custom neighbor filtering."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Extract the base scan group for each item
    groups = np.array([get_scan_group(k) for k in keys])

    # StratifiedGroupKFold ensures reconstructions of the same scan are never split across train/test
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = {k: {'acc': [], 'ba': [], 'f1': []} for k in k_values}

    max_k = max(k_values)

    for train_idx, test_idx in sgkf.split(X, y_encoded, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        train_groups = groups[train_idx]
        test_groups = groups[test_idx]

        # Fetch extra neighbors to account for dropped duplicates (buffer of 50 is safe for recons)
        search_k = min(max_k + 50, len(X_train))
        nn = NearestNeighbors(n_neighbors=search_k, metric='cosine')
        nn.fit(X_train)
        distances, indices = nn.kneighbors(X_test)

        preds_by_k = {k: [] for k in k_values}

        # Manually filter neighbors and vote
        for i in range(len(X_test)):
            test_group = test_groups[i]
            # Track seen groups (initialized with test_group as an extra safety measure)
            seen_groups = {test_group}
            valid_neighbor_labels = []

            for train_idx_in_train in indices[i]:
                neighbor_group = train_groups[train_idx_in_train]

                # Only include this neighbor if its scan group hasn't already voted
                if neighbor_group not in seen_groups:
                    seen_groups.add(neighbor_group)
                    valid_neighbor_labels.append(y_train[train_idx_in_train])

                if len(valid_neighbor_labels) == max_k:
                    break

            # Edge-case fallback if all neighbors were exhausted
            if not valid_neighbor_labels:
                fallback_pred = Counter(y_train).most_common(1)[0][0]
                valid_neighbor_labels = [fallback_pred] * max_k
            elif len(valid_neighbor_labels) < max_k:
                # Pad with the last valid neighbor label if we ran short
                valid_neighbor_labels.extend([valid_neighbor_labels[-1]] * (max_k - len(valid_neighbor_labels)))

            # Perform the majority vote for each K
            for k in k_values:
                k_labels = valid_neighbor_labels[:k]
                pred = Counter(k_labels).most_common(1)[0][0]
                preds_by_k[k].append(pred)

        # Calculate metrics for the fold
        for k in k_values:
            fold_metrics[k]['acc'].append(accuracy_score(y_test, preds_by_k[k]))
            fold_metrics[k]['ba'].append(balanced_accuracy_score(y_test, preds_by_k[k]))
            fold_metrics[k]['f1'].append(f1_score(y_test, preds_by_k[k], average='macro'))

    # Return the mean across all 5 folds
    return {k: {
        'acc': np.mean(fold_metrics[k]['acc']),
        'ba': np.mean(fold_metrics[k]['ba']),
        'f1': np.mean(fold_metrics[k]['f1'])
    } for k in k_values}


def main():
    parser = argparse.ArgumentParser(description="Macro-Average KNN Evaluator across all CSV pathology columns.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    logger.info(f"Loading configuration from: {args.config}")
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        return

    datasets = config.get("datasets", {})
    models = config.get("models", {})
    table_data = []

    logger.info(f"Initialization complete. Evaluating {len(models)} models across {len(datasets)} datasets.")
    print("-" * 80)

    for model_name, model_paths in models.items():
        logger.info(f"🚀 STARTING EVALUATION FOR MODEL: {model_name.upper()}")
        row_dict = {"Model": model_name}

        for dataset_name, dataset_info in datasets.items():
            logger.info(f"➤ Processing dataset: {dataset_name} for model {model_name}")

            emb_dir = model_paths.get(dataset_name)
            labels_csv = dataset_info["labels_csv"]
            volume_col = dataset_info["volume_column"]

            if not emb_dir or not os.path.exists(emb_dir):
                logger.warning(f"Embedding directory missing or not found for {dataset_name}. Skipping.")
                row_dict[f"{dataset_name} (1-NN)"] = "-"
                row_dict[f"{dataset_name} (5-NN)"] = "-"
                continue

            # 1. Load DataFrame and identify pathology columns
            logger.info(f"Loading labels from: {labels_csv}")
            df = pd.read_csv(labels_csv)
            initial_rows = len(df)

            df['match_key'] = df[volume_col].apply(get_base_volume_name)
            df = df.replace(["NaN", "nan", "None", "", "[]", " "], np.nan)

            label_cols = [col for col in df.columns if col not in [volume_col, 'match_key']]
            logger.info(f"Loaded {initial_rows} rows and identified {len(label_cols)} potential label columns.")

            # 2. Load all embeddings into memory once
            emb_dict = load_all_embeddings(emb_dir)
            if not emb_dict:
                logger.warning(f"No embeddings loaded for {dataset_name}. Skipping evaluation.")
                row_dict[f"{dataset_name} (1-NN)"] = "-"
                row_dict[f"{dataset_name} (5-NN)"] = "-"
                continue

            # 3. Track cumulative metrics across all pathology columns
            macro_metrics = {1: {'acc': [], 'ba': [], 'f1': []}, 5: {'acc': [], 'ba': [], 'f1': []}}
            valid_cols_evaluated = 0
            skipped_insufficient_samples = 0
            skipped_split_constraints = 0

            # 4. Iterate over every pathology column
            column_pbar = tqdm(label_cols, desc=f"Evaluating Columns ({dataset_name})", leave=False, unit="col")
            for col in column_pbar:
                valid_df = df.dropna(subset=[col])

                features, labels, keys = [], [], []
                for _, row in valid_df.iterrows():
                    key = row['match_key']
                    if key in emb_dict:
                        features.append(emb_dict[key])
                        labels.append(row[col])
                        keys.append(key)

                if len(features) < 5:
                    skipped_insufficient_samples += 1
                    logger.debug(f"Skipped column '{col}': Only {len(features)} matched samples (needs >= 5).")
                    continue

                X = np.stack(features, axis=0)
                y = np.array(labels)

                # Check if class distribution allows 5-fold CV
                unique_classes, counts = np.unique(y, return_counts=True)
                if len(unique_classes) > 1 and min(counts) >= 5:
                    try:
                        # Pass the keys (match_keys) down into the evaluation for filtering
                        col_results = evaluate_knn_cv(X, y, keys, k_values=[1, 5], n_splits=5)
                        for k in [1, 5]:
                            macro_metrics[k]['acc'].append(col_results[k]['acc'])
                            macro_metrics[k]['ba'].append(col_results[k]['ba'])
                            macro_metrics[k]['f1'].append(col_results[k]['f1'])
                        valid_cols_evaluated += 1

                        # Update progress bar postfix with current valid count
                        column_pbar.set_postfix({'Valid Cols': valid_cols_evaluated})

                    except Exception as e:
                        skipped_split_constraints += 1
                        logger.debug(f"Skipped column '{col}': Failed StratifiedKFold constraints. Error: {e}")
                else:
                    skipped_split_constraints += 1
                    logger.debug(f"Skipped column '{col}': Insufficient minority class samples for 5-fold CV.")

            # Logging evaluation summary for this dataset
            logger.info(f"Evaluation Summary for {model_name} on {dataset_name}:")
            logger.info(f"   - Successfully Evaluated Columns: {valid_cols_evaluated}")
            logger.info(f"   - Skipped (Insufficient Samples) : {skipped_insufficient_samples}")
            logger.info(f"   - Skipped (CV Constraints)       : {skipped_split_constraints}")

            # 5. Compute the final macro-average across all evaluated columns
            if valid_cols_evaluated > 0:
                for k in [1, 5]:
                    final_acc = np.mean(macro_metrics[k]['acc'])
                    final_ba = np.mean(macro_metrics[k]['ba'])
                    final_f1 = np.mean(macro_metrics[k]['f1'])
                    row_dict[f"{dataset_name} ({k}-NN)"] = f"{final_acc:.4f} / {final_ba:.4f} / {final_f1:.4f}"
            else:
                logger.warning(f"No valid columns evaluated for {dataset_name}.")
                row_dict[f"{dataset_name} (1-NN)"] = "N/A"
                row_dict[f"{dataset_name} (5-NN)"] = "N/A"

            print("-" * 80)

        table_data.append(row_dict)

    logger.info("All evaluations completed. Generating final leaderboard...")

    df_out = pd.DataFrame(table_data)
    df_out.set_index("Model", inplace=True)

    columns = [(col, "Acc / BA / F1") for col in df_out.columns]
    df_out.columns = pd.MultiIndex.from_tuples(columns)

    print("\n" + "=" * 120)
    print(f"{'🏆 MACRO-AVERAGED KNN 5-FOLD CV LEADERBOARD (Across All Label Columns) 🏆':^120}")
    print("=" * 120)
    print(df_out.to_markdown())
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()