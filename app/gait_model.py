"""
Gait-parameter prediction via an uploaded GaitGRU checkpoint.

Ported from /home/aicenter/Dev/gait_model/pytorch_v2/infer_pytorch_model_v2.py
and run_inference_batch_v2.py (architecture + long-format output schema are
identical across that sibling repo's v1/v2/v3). The checkpoint is fully
self-contained -- weights, input/output sizes, normalization stats, and
target column names -- so nothing besides the .pt file and a skeleton CSV
produced by pose_estimation.py is needed here. This replaces gait_analysis.py
in the web app's flow entirely.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

FEET_INDICES = [17, 19, 20, 22]  # L_BigToe, L_Heel, R_BigToe, R_Heel
REQUIRED_CHECKPOINT_KEYS = [
    "model_state_dict", "input_size", "hidden_size", "output_size",
    "in_mean", "in_std", "tgt_mean", "tgt_std", "target_columns",
]

# The model is tiny (24 hidden units) -- CPU is fine and avoids contending
# with pose estimation's GPU-bound RTMW-x inference.
DEVICE = torch.device("cpu")

UNITS = {
    "Cadence": "steps/min", "Walking Speed": "m/s", "Stride Time": "s", "Step Time": "s",
    "Opposite Foot Off": "%", "Opposite Foot Contact": "%", "Foot Off": "%",
    "Single Support": "s", "Double Support": "s", "Stride Length": "m", "Step Length": "m",
    "Step Width": "m", "Limp Index": "ratio",
}


class GatedRecurrentUnit(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
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


class InvalidCheckpoint(Exception):
    pass


def load_checkpoint(model_path):
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in ckpt]
    if missing:
        raise InvalidCheckpoint(f"checkpoint is missing expected keys: {missing}")

    model = GatedRecurrentUnit(
        input_size=ckpt["input_size"], hidden_size=ckpt["hidden_size"], output_size=ckpt["output_size"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_feet_keypoints_csv(csv_path):
    """`csv_path` may be a path string or any file-like object pandas.read_csv
    accepts (e.g. an UploadFile's .file) -- used both for on-disk skeleton
    CSVs (per-round predict) and ad-hoc uploads (standalone /predict)."""
    df = pd.read_csv(csv_path)
    cols = [f"j{j}_{ax}" for j in FEET_INDICES for ax in ("x", "y", "z")]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing expected feet-keypoint columns: {missing}")
    return df[cols].to_numpy(dtype=np.float32)  # (T, 12)


def predict(model, ckpt, sequence):
    x = (sequence.astype(np.float32) - ckpt["in_mean"]) / ckpt["in_std"]
    x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([sequence.shape[0]])
    with torch.no_grad():
        pred = model(x, lengths).cpu().numpy()[0]
    pred_denorm = pred * ckpt["tgt_std"] + ckpt["tgt_mean"]
    return {str(k): float(v) for k, v in zip(ckpt["target_columns"], pred_denorm)}


def predict_from_skeleton_csv(model_path, skeleton_csv_path):
    """High-level entry point for the web app: loads the checkpoint + skeleton
    CSV and returns {target_column_name: predicted_value}."""
    model, ckpt = load_checkpoint(model_path)
    sequence = load_feet_keypoints_csv(skeleton_csv_path)
    return predict(model, ckpt, sequence)


def to_long_format(subject, result):
    """Converts a {target_column: value} prediction dict into the same
    long-format (Subject,Context,Name,Value,Units) schema used throughout the
    rest of the pipeline (gait_analysis.py's output, Vicon reference CSVs)."""
    rows = []
    for col, val in result.items():
        ctx, name = col.split("_", 1)
        rows.append({"Subject": subject, "Context": ctx, "Name": name, "Value": val, "Units": UNITS.get(name, "")})
    return pd.DataFrame(rows)
