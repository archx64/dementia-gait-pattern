import os, numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from src.utils_floor_align import FPS_ANALYSIS, SUBJECT_NAME, ROUND, INFO, OUTPUT_CSV, DAY, MONTH, P_NO

# OUTPUT_CSV = 'output/4.csv'

class GaitAnalyzer:
    def __init__(self, csv_path, fps, height_axis="y", start_frame=None, end_frame=None):
        """
        Loads and prepares skeleton CSV data for gait analysis.

        up_direction is AUTO-DETECTED from the data:
          Y-up  (pe_floor_align output, floor at Y≈0):  head Y > heel Y → up_dir = +1
          Y-down (raw camera space, OpenCV convention):  head Y < heel Y → up_dir = -1

        height_axis : column suffix used for vertical position ("y")
        start_frame : first frame index to include  (None = beginning of file)
        end_frame   : last  frame index to include  (None = end of file)
        """
        self.fps  = fps
        self.dt   = 1.0 / fps
        self.height_axis = height_axis.lower()

        print(INFO + f"Loading CSV: {csv_path}")
        self.df = pd.read_csv(csv_path)

        if start_frame is not None:
            self.df = self.df[self.df['frame_idx'] >= start_frame]
            print(INFO + f"Cropped start to frame >= {start_frame}")
        if end_frame is not None:
            self.df = self.df[self.df['frame_idx'] <= end_frame]
            print(INFO + f"Cropped end to frame <= {end_frame}")

        self.df = self.df.reset_index(drop=True)

        # Keypoint column map  (RTMWx WholeBody 133-keypoint indices)
        self.map = {
            "L_Heel_X": "j19_x", "L_Heel_Y": "j19_y", "L_Heel_Z": "j19_z",
            "R_Heel_X": "j22_x", "R_Heel_Y": "j22_y", "R_Heel_Z": "j22_z",
            "L_Toe_X":  "j17_x", "L_Toe_Y":  "j17_y", "L_Toe_Z":  "j17_z",
            "R_Toe_X":  "j20_x", "R_Toe_Y":  "j20_y", "R_Toe_Z":  "j20_z",
        }

        # ── Auto-detect Y direction ──────────────────────────────────────
        head_y = self.df['j0_y'].dropna().mean()
        heel_y = pd.concat([self.df['j19_y'], self.df['j22_y']]).dropna().mean()
        if head_y > heel_y:
            self.up_dir = 1
            print(INFO + f"Y direction: Y-UP  (head={head_y:.2f} > heel={heel_y:.2f})")
        else:
            self.up_dir = -1
            print(INFO + f"Y direction: Y-DOWN (head={head_y:.2f} < heel={heel_y:.2f})")

        self.df.interpolate(method='linear', limit_direction='both', inplace=True)
        self.filter_data()

    def filter_data(self):
        b, a = butter(4, 6 / (0.5 * self.fps), btype="low")
        for col in self.df.columns:
            if col.startswith("j"):
                self.df[col] = self.df[col].ffill().bfill()
                self.df[col] = filtfilt(b, a, self.df[col])

    def detect_events(self, side):
        col_heel = self.map[f"{side}_Heel_{self.height_axis.upper()}"]
        col_toe  = self.map[f"{side}_Toe_{self.height_axis.upper()}"]
        heel_height = self.df[col_heel].values
        toe_height  = self.df[col_toe].values

        strike_signal = -heel_height if self.up_dir == 1 else heel_height
        strikes, _ = find_peaks(strike_signal, distance=self.fps * 0.4, prominence=0.02)

        vel_height = np.gradient(toe_height)
        off_signal = vel_height if self.up_dir == 1 else -vel_height
        offs, _ = find_peaks(off_signal, prominence=0.015, distance=self.fps * 0.4)

        return np.sort(strikes), np.sort(offs)

    # ─────────────────────────────────────────────────────────────────────────
    # collect_strides  — returns a LIST of per-stride dicts (not averaged yet)
    # ─────────────────────────────────────────────────────────────────────────
    def collect_strides(self, strikes, offs, opp_strikes, opp_offs, side,
                        min_stride_dur=None, boundary_buffer=4):
        """
        Parameters
        ----------
        min_stride_dur : float or None
            Strides shorter than this (seconds) are skipped.
            Derived from the opposite (more reliable) side's median; eliminates
            false heel strikes from the acceleration phase at the trial start.
        boundary_buffer : int
            Frames to extend the OppOff search window *before* the stride start.
            Handles the common edge case where the opposite toe-off fell 1–4 frames
            before the stride boundary due to the short trial length.
        """
        if len(strikes) < 2:
            return []

        per_stride = []

        for i in range(len(strikes) - 1):
            start        = int(strikes[i])
            end          = int(strikes[i + 1])
            stride_dur   = (end - start) / self.fps
            stride_frames = end - start

            if stride_dur == 0:
                continue

            # ── Stride duration filter ────────────────────────────────────
            if min_stride_dur is not None and stride_dur < min_stride_dur:
                print(INFO + f"  [{side}] stride {i}: {stride_dur:.2f}s < min {min_stride_dur:.2f}s  → SKIPPED")
                continue

            def get_pt(frame, name):
                return np.array([
                    self.df.iloc[frame][self.map[f"{name}_X"]],
                    self.df.iloc[frame][self.map[f"{name}_Y"]],
                    self.df.iloc[frame][self.map[f"{name}_Z"]]
                ])

            l_start = get_pt(start, "L_Heel")
            r_start = get_pt(start, "R_Heel")

            p1 = get_pt(start, f"{side}_Heel")
            p2 = get_pt(end,   f"{side}_Heel")

            stride_len = np.sqrt((p2[0]-p1[0])**2 + (p2[2]-p1[2])**2)
            step_len   = abs(l_start[2] - r_start[2])
            step_width = abs(l_start[0] - r_start[0])

            # Foot off: strictly inside the stride window
            valid_offs = offs[(offs > start) & (offs < end)]
            foot_off_pct = np.nan
            if len(valid_offs) > 0:
                foot_off_pct = (valid_offs[0] - start) / stride_frames * 100

            # Opp foot contact: strictly inside
            valid_opp_s = opp_strikes[(opp_strikes > start) & (opp_strikes < end)]
            opp_con_pct = np.nan
            step_time   = np.nan
            if len(valid_opp_s) > 0:
                opp_con_pct = (valid_opp_s[0] - start) / stride_frames * 100
                step_time   = (valid_opp_s[0] - start) / self.fps

            # ── Opp foot off: extend window backward by boundary_buffer frames ──
            # This catches the opposite toe-off that occurs just before the stride
            # start, a common edge case in short trials where the event timing is
            # close to the stride boundary.  Raw % may be slightly negative (e.g.
            # -3.8%) when the event fell 1 frame before the window; clamp to 0.
            valid_opp_o = opp_offs[
                (opp_offs >= start - boundary_buffer) & (opp_offs < end)
            ]
            opp_off_pct = np.nan
            if len(valid_opp_o) > 0:
                raw_pct     = (valid_opp_o[0] - start) / stride_frames * 100
                opp_off_pct = max(raw_pct, 0.0)   # clamp: event was ≤buffer frames early

            # Single / double support
            single_supp = np.nan
            if not (np.isnan(opp_con_pct) or np.isnan(opp_off_pct)):
                single_supp = (opp_con_pct - opp_off_pct) / 100.0 * stride_dur

            double_supp = np.nan
            if not (np.isnan(foot_off_pct) or np.isnan(opp_con_pct) or np.isnan(opp_off_pct)):
                double_supp = (opp_off_pct + (foot_off_pct - opp_con_pct)) / 100.0 * stride_dur

            limp = np.nan
            if not np.isnan(foot_off_pct):
                swing = 100 - foot_off_pct
                if swing > 0:
                    limp = foot_off_pct / swing

            per_stride.append({
                "StrideTime":     stride_dur,
                "StrideLen":      stride_len,
                "StepLen":        step_len,
                "StepWidth":      step_width,
                "WalkingSpeed":   stride_len / stride_dur,
                "Cadence":        (60 / stride_dur) * 2,
                "StepTime":       step_time,
                "FootOff":        foot_off_pct,
                "OppFootContact": opp_con_pct,
                "OppFootOff":     opp_off_pct,
                "SingleSupport":  single_supp,
                "DoubleSupport":  double_supp,
                "LimpIndex":      limp,
            })

        return per_stride

    # ─────────────────────────────────────────────────────────────────────────
    # generate_vicon_tables  — single-trial OR pooled across extra_csv_paths
    # ─────────────────────────────────────────────────────────────────────────
    def generate_vicon_tables(self, extra_csv_paths=None):
        """
        extra_csv_paths : list of additional CSV file paths (same subject/session).
            Strides from all files are pooled before computing means.
            Use this to get robust averages from multiple short trials.
        """
        BOUNDARY_BUFFER = 4   # frames; covers a 1-4 frame OppOff boundary overshoot

        l_strikes, l_offs = self.detect_events("L")
        r_strikes, r_offs = self.detect_events("R")

        # ── Cross-side minimum stride duration ───────────────────────────────
        # L side is more reliable (consistent stride times); use its median as
        # the reference.  Any stride < 80 % of that is a false detection.
        l_durs = [(l_strikes[i+1]-l_strikes[i])/self.fps
                  for i in range(len(l_strikes)-1)]
        expected_stride = float(np.median(l_durs)) if l_durs else None
        min_stride_dur  = expected_stride * 0.80 if expected_stride else None

        if expected_stride:
            print(INFO + f"Expected stride: {expected_stride:.3f}s  "
                         f"→ min allowed: {min_stride_dur:.3f}s")

        l_strides = self.collect_strides(l_strikes, l_offs, r_strikes, r_offs, "L",
                                          min_stride_dur=min_stride_dur,
                                          boundary_buffer=BOUNDARY_BUFFER)
        r_strides = self.collect_strides(r_strikes, r_offs, l_strikes, l_offs, "R",
                                          min_stride_dur=min_stride_dur,
                                          boundary_buffer=BOUNDARY_BUFFER)

        print(INFO + f"[primary]  L: {len(l_strides)} strides  R: {len(r_strides)} strides")

        # ── Pool from additional trials ───────────────────────────────────────
        if extra_csv_paths:
            for extra_path in extra_csv_paths:
                print(INFO + f"Pooling: {extra_path}")
                try:
                    extra = GaitAnalyzer(extra_path, fps=self.fps)
                    el, elo = extra.detect_events("L")
                    er, ero = extra.detect_events("R")

                    e_durs = [(el[i+1]-el[i])/self.fps for i in range(len(el)-1)]
                    e_exp  = float(np.median(e_durs)) if e_durs else expected_stride
                    e_min  = e_exp * 0.80 if e_exp else min_stride_dur

                    new_l = extra.collect_strides(el, elo, er, ero, "L",
                                                   min_stride_dur=e_min,
                                                   boundary_buffer=BOUNDARY_BUFFER)
                    new_r = extra.collect_strides(er, ero, el, elo, "R",
                                                   min_stride_dur=e_min,
                                                   boundary_buffer=BOUNDARY_BUFFER)

                    l_strides += new_l
                    r_strides += new_r
                    print(INFO + f"  added L:{len(new_l)} R:{len(new_r)} strides")

                except Exception as exc:
                    print(INFO + f"  Warning — could not load {extra_path}: {exc}")

            print(INFO + f"[pooled]   L: {len(l_strides)} strides  R: {len(r_strides)} strides")

        # ── Average across all accepted strides ───────────────────────────────
        def to_means(strides):
            if not strides:
                return {}
            keys = strides[0].keys()
            return {k: np.nanmean([s[k] for s in strides]) for k in keys}

        l_res = to_means(l_strides)
        r_res = to_means(r_strides)

        # ── Build output table ────────────────────────────────────────────────
        param_defs = [
            ("Cadence",         "Cadence",               "steps/min"),
            ("WalkingSpeed",    "Walking Speed",          "m/s"),
            ("StrideTime",      "Stride Time",            "s"),
            ("StepTime",        "Step Time",              "s"),
            ("OppFootOff",      "Opposite Foot Off",      "%"),
            ("OppFootContact",  "Opposite Foot Contact",  "%"),
            ("FootOff",         "Foot Off",               "%"),
            ("SingleSupport",   "Single Support",         "s"),
            ("DoubleSupport",   "Double Support",         "s"),
            ("StrideLen",       "Stride Length",          "m"),
            ("StepLen",         "Step Length",            "m"),
            ("StepWidth",       "Step Width",             "m"),
            ("LimpIndex",       "Limp Index",             "ratio"),
        ]

        rows = []
        def add_rows(res, ctx):
            if not res:
                return
            for k, name, unit in param_defs:
                rows.append({
                    "Subject": SUBJECT_NAME, "Context": ctx,
                    "Name": name, "Value": res.get(k, np.nan), "Units": unit
                })

        add_rows(l_res, "Left")
        add_rows(r_res, "Right")

        return pd.DataFrame(rows), pd.DataFrame()


def main():
    START_FRAME = None   # None = use all frames from the beginning
    END_FRAME   = None   # None = use all frames to the end

    # ── Multi-trial pooling ──────────────────────────────────────────────────
    # For subjects recorded in short trials: list ALL CSV paths from the same
    # session here.  Strides are pooled across all files before averaging.
    # Each file should be the pe_floor_align output (floor-aligned, Y-up).
    # Leave as an empty list [] if you have only one trial.
    EXTRA_TRIALS = [
        # 'output/01-04-p1_r2.csv',
        # 'output/01-04-p1_r3.csv',
    ]

    analyzer = GaitAnalyzer(
        OUTPUT_CSV,
        fps=FPS_ANALYSIS,
        height_axis="y",
        start_frame=START_FRAME,
        end_frame=END_FRAME,
    )

    params_df, _ = analyzer.generate_vicon_tables(
        extra_csv_paths=EXTRA_TRIALS if EXTRA_TRIALS else None
    )

    print("\n# Gait Cycle Parameters")
    print(params_df.to_markdown(index=True))

    gait_out = "output_phramongkut/gait"
    os.makedirs(gait_out, exist_ok=True)
    save_path = os.path.join(gait_out, f"{DAY}-{MONTH}_{SUBJECT_NAME}_p{P_NO}_r{ROUND}_gait.csv")
    params_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()