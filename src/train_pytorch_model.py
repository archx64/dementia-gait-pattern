"""
GRU sequence model over per-frame feet keypoints -> 26 Vicon gait parameters,
evaluated with leave-one-subject-out cross-validation (LOSO).

Kept deliberately small (24 hidden units, dropout, weight decay) given only
~270 training trials per fold — a larger model would just memorize subjects.
Inputs and targets are z-score normalized per fold using train-fold
statistics only (no leakage of held-out subject's scale into normalization).
Missing Vicon values (NaN) are masked out of the loss and out of reported MAE.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

DATASET_DIR = "/home/aicenter/Dev/dementia-gait-pattern/output_mahidol/ml_dataset"
DATASET_PATH = os.path.join(DATASET_DIR, "dataset.npz")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")

INPUT_SIZE = 12
HIDDEN_SIZE = 24
OUTPUT_SIZE = 26
EPOCHS = 150
LR = 1e-3
WEIGHT_DECAY = 1e-4


class GaitGRU(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.head(h_n[-1])


def pad_batch(seqs):
    lengths = torch.tensor([s.shape[0] for s in seqs])
    maxlen = int(lengths.max().item())
    batch = torch.zeros(len(seqs), maxlen, seqs[0].shape[1], dtype=torch.float32)
    for i, s in enumerate(seqs):
        batch[i, : s.shape[0]] = torch.from_numpy(s.astype(np.float32))
    return batch, lengths


def masked_mse(pred, target, mask):
    diff = (pred - target) ** 2 * mask
    return diff.sum() / mask.sum().clamp(min=1)


def main():
    d = np.load(DATASET_PATH, allow_pickle=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    sequences = d["sequences"]
    targets = d["targets"].astype(np.float32)
    target_cols = d["target_columns"]
    groups = manifest["subject"].values
    unique_subjects = np.unique(groups)

    target_mask = ~np.isnan(targets)
    targets_filled = np.nan_to_num(targets, nan=0.0)

    fold_maes = {col: [] for col in target_cols}

    for held_out in unique_subjects:
        train_idx = np.where(groups != held_out)[0]
        test_idx = np.where(groups == held_out)[0]
        if len(test_idx) == 0:
            continue

        train_concat = np.concatenate([sequences[i] for i in train_idx], axis=0)
        in_mean, in_std = train_concat.mean(0), train_concat.std(0) + 1e-6

        tgt_train = targets[train_idx]
        tgt_mean = np.nanmean(tgt_train, axis=0)
        tgt_std = np.nanstd(tgt_train, axis=0) + 1e-6

        def make_batch(idx):
            seqs = [(sequences[i] - in_mean) / in_std for i in idx]
            batch, lengths = pad_batch(seqs)
            tgt = (targets_filled[idx] - tgt_mean) / tgt_std
            mask = target_mask[idx].astype(np.float32)
            return batch, lengths, torch.from_numpy(tgt.astype(np.float32)), torch.from_numpy(mask)

        Xtr, Ltr, Ytr, Mtr = make_batch(train_idx)
        Xte, Lte, Yte, Mte = make_batch(test_idx)

        model = GaitGRU()
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        model.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            pred = model(Xtr, Ltr)
            loss = masked_mse(pred, Ytr, Mtr)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_te = model(Xte, Lte).numpy()
        pred_te_denorm = pred_te * tgt_std + tgt_mean
        true_te = targets[test_idx]

        for j, col in enumerate(target_cols):
            valid = ~np.isnan(true_te[:, j])
            if valid.sum() == 0:
                continue
            mae = np.mean(np.abs(pred_te_denorm[valid, j] - true_te[valid, j]))
            fold_maes[col].append(mae)

    print(f"{'Parameter':<28} {'MAE (LOSO mean)':>16} {'folds':>7}")
    for col in target_cols:
        vals = fold_maes[col]
        if vals:
            print(f"{col:<28} {np.mean(vals):>16.4f} {len(vals):>7}")


if __name__ == "__main__":
    main()
