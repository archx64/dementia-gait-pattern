"""
Baseline: hand-engineered per-trial summary features (mean/std/min/max/median/
p10/p90 of each feet-keypoint column, plus trial length) -> per-target
RandomForestRegressor, evaluated with leave-one-subject-out cross-validation
so no subject's own trials ever leak into its held-out fold.

26 independent single-output models (one per Vicon parameter) rather than one
joint multi-output model, since each target column has its own NaN rows
(missing Vicon events) that would otherwise force dropping whole trials.
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut

DATASET_DIR = "/home/aicenter/Dev/dementia-gait-pattern/output_mahidol/ml_dataset"
DATASET_PATH = os.path.join(DATASET_DIR, "dataset.npz")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")

PERCENTILES = (10, 90)


def extract_features(seq):
    feats = []
    for col in range(seq.shape[1]):
        x = seq[:, col]
        feats.extend([x.mean(), x.std(), x.min(), x.max(), np.median(x)])
        feats.extend(np.percentile(x, PERCENTILES))
    feats.append(seq.shape[0])
    return np.array(feats, dtype=float)


def build_feature_matrix(sequences):
    return np.array([extract_features(s) for s in sequences])


def main():
    d = np.load(DATASET_PATH, allow_pickle=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    sequences = d["sequences"]
    targets = d["targets"]
    target_cols = d["target_columns"]
    groups = manifest["subject"].values

    X = build_feature_matrix(sequences)
    logo = LeaveOneGroupOut()

    results = {col: [] for col in target_cols}

    for train_idx, test_idx in logo.split(X, groups=groups):
        for j, col in enumerate(target_cols):
            y = targets[:, j]
            valid_train = train_idx[~np.isnan(y[train_idx])]
            valid_test = test_idx[~np.isnan(y[test_idx])]
            if len(valid_train) < 5 or len(valid_test) == 0:
                continue
            model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0)
            model.fit(X[valid_train], y[valid_train])
            preds = model.predict(X[valid_test])
            mae = np.mean(np.abs(preds - y[valid_test]))
            results[col].append(mae)

    print(f"{'Parameter':<28} {'MAE (LOSO mean)':>16} {'folds':>7}")
    for col in target_cols:
        vals = results[col]
        if vals:
            print(f"{col:<28} {np.mean(vals):>16.4f} {len(vals):>7}")


if __name__ == "__main__":
    main()
