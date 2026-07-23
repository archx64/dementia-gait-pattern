"""
Builds a paired (feet-keypoint trajectory -> Vicon gait parameters) dataset
for training a foot-ground-point correction model.

Pairing between skeleton rounds and Vicon trials is not derivable from
either source alone (round numbers are a flat per-session counter; Vicon
trials are named by task condition) — it's recovered from filenames in the
sibling dav2m-dementia/comparison_output/ project, which already verified
the correspondence per subject via method_comparison_<vicon_trial>_vs_<round>
files. A handful of entries there are known-bad (self-referential, placeholder
"subject01", or an unresolvable duplicate round) and are excluded below
without touching that project's files.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

COMPARISON_DIR = "/home/aicenter/Dev/dav2m-dementia/comparison_output"
SKELETON_DIR = "/home/aicenter/Dev/dementia-gait-pattern/output_mahidol/skeleton"
VICON_DIR = "/home/aicenter/Dev/dementia-gait-pattern/output_mahidol/vicon"
OUT_DIR = "/home/aicenter/Dev/dementia-gait-pattern/output_mahidol/ml_dataset"

FEET_INDICES = [17, 19, 20, 22]  # L_BigToe, L_Heel, R_BigToe, R_Heel

EXCLUDED_STEMS = {
    "method_comparison_07-Joe_vs_07_Joe_r1",
    "method_comparison_07-Joe_vs_subject01_r1",
    "method_comparison_12-Nut_vs_12_Nut_r1",
    "method_comparison_07_Joe_Fast5_vs_subject01_r1",
    "method_comparison_07_Joe_Com1_vs_07-Joe_r1",
    "method_comparison_07_Joe_Com2_vs_07-Joe_r1",
}

PARAM_NAMES = [
    "Cadence", "Walking Speed", "Stride Time", "Step Time", "Opposite Foot Off",
    "Opposite Foot Contact", "Foot Off", "Single Support", "Double Support",
    "Stride Length", "Step Length", "Step Width", "Limp Index",
]
TARGET_COLUMNS = [f"{ctx}_{p}" for ctx in ("Left", "Right") for p in PARAM_NAMES]


def parse_pairs():
    stems = [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(COMPARISON_DIR, "method_comparison_*.csv"))
    ]
    pairs = []
    for stem in stems:
        if stem in EXCLUDED_STEMS:
            continue
        body = stem[len("method_comparison_"):]
        left, right = body.split("_vs_")
        m = re.match(r"^(\d+)_([A-Za-z]+)_(.+)$", left)
        if m:
            code, name, trial = m.groups()
        else:
            # Takk-style variant: code lives on the right side instead,
            # e.g. "Takk_Fast_Dual3_vs_01_Takk_r3"
            m_left = re.match(r"^([A-Za-z]+)_(.+)$", left)
            m_right_code = re.match(r"^(\d+)_[A-Za-z]+_r\d+$", right)
            if not (m_left and m_right_code):
                print(f"SKIP (unparseable): {stem}")
                continue
            name, trial = m_left.groups()
            code = m_right_code.group(1)
        round_m = re.search(r"_r(\d+)$", right)
        if not round_m:
            print(f"SKIP (no round number): {stem}")
            continue
        round_num = int(round_m.group(1))
        vicon_filename = f"{code} {name} {trial.replace('_', ' ')}.csv"
        pairs.append({
            "code": code, "name": name, "trial": trial,
            "round": round_num, "vicon_filename": vicon_filename,
        })
    return pairs


def resolve_skeleton_path(code, name, round_num):
    pattern = re.compile(rf"{code}[-_]{name}_p\d+_r{round_num}\.csv$", re.IGNORECASE)
    candidates = [
        f for f in os.listdir(SKELETON_DIR)
        if pattern.search(f) and not f.endswith("_raw2d.npz")
    ]
    if len(candidates) <= 1:
        return os.path.join(SKELETON_DIR, candidates[0]) if candidates else None
    # multiple matches (e.g. Song has re-run duplicates) — prefer the
    # non-zero-padded day-month bulk convention shared by every other subject
    preferred = [c for c in candidates if not re.match(r"^0\d-0\d_", c)]
    chosen = sorted(preferred or candidates)[0]
    return os.path.join(SKELETON_DIR, chosen)


def load_vicon_targets(vicon_path):
    df = pd.read_csv(vicon_path)
    values = {}
    for _, row in df.iterrows():
        key = f"{row['Context']}_{row['Name']}"
        values[key] = row["Value"]
    return [values.get(col, np.nan) for col in TARGET_COLUMNS]


def load_feet_keypoints(skeleton_path):
    df = pd.read_csv(skeleton_path)
    cols = [f"j{j}_{ax}" for j in FEET_INDICES for ax in ("x", "y", "z")]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None
    return df[cols].to_numpy(dtype=float)  # (n_frames, 12)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = parse_pairs()
    print(f"Parsed {len(pairs)} candidate pairs (after excluding known-bad entries)")

    manifest = []
    sequences = []
    targets = []
    unresolved = []

    for p in pairs:
        skel_path = resolve_skeleton_path(p["code"], p["name"], p["round"])
        vicon_path = os.path.join(VICON_DIR, p["vicon_filename"])

        if skel_path is None or not os.path.exists(vicon_path):
            unresolved.append((p, skel_path, vicon_path))
            continue

        feet = load_feet_keypoints(skel_path)
        if feet is None or len(feet) == 0:
            unresolved.append((p, skel_path, vicon_path))
            continue

        target = load_vicon_targets(vicon_path)

        idx = len(sequences)
        sequences.append(feet)
        targets.append(target)
        manifest.append({
            "idx": idx,
            "subject": f"{p['code']}_{p['name']}",
            "round": p["round"],
            "trial": p["trial"],
            "skeleton_path": skel_path,
            "vicon_path": vicon_path,
            "n_frames": len(feet),
        })

    print(f"Resolved {len(sequences)} pairs; {len(unresolved)} unresolved")
    for p, sp, vp in unresolved:
        reason = "no skeleton match" if sp is None else "no vicon file" if not os.path.exists(vp) else "empty/missing feet cols"
        print(f"  UNRESOLVED ({reason}): {p['code']}_{p['name']} r{p['round']} trial={p['trial']}")

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(os.path.join(OUT_DIR, "manifest.csv"), index=False)

    targets_arr = np.array(targets, dtype=float)  # (N, 26)
    np.savez(
        os.path.join(OUT_DIR, "dataset.npz"),
        sequences=np.array(sequences, dtype=object),  # variable-length (n_frames, 12) per entry
        targets=targets_arr,
        target_columns=np.array(TARGET_COLUMNS),
        feet_indices=np.array(FEET_INDICES),
    )

    print(f"\nSaved {len(sequences)} examples across {manifest_df['subject'].nunique()} subjects")
    print(f"  manifest: {os.path.join(OUT_DIR, 'manifest.csv')}")
    print(f"  dataset:  {os.path.join(OUT_DIR, 'dataset.npz')}")
    print(f"\nPer-subject counts:\n{manifest_df['subject'].value_counts()}")
    print(f"\nTarget NaN rate per column (missing Vicon values):")
    nan_rate = pd.DataFrame(targets_arr, columns=TARGET_COLUMNS).isna().mean()
    print(nan_rate[nan_rate > 0])


if __name__ == "__main__":
    main()
