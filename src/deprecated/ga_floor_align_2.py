import os, numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from src.utils_floor_align import FPS_ANALYSIS, SUBJECT_NAME, ROUND, INFO, OUTPUT_CSV

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
        # Y-up  (correctly aligned): head is ABOVE feet  → head_Y > heel_Y
        # Y-down (raw camera space): head is ABOVE feet  → head_Y < heel_Y
        head_y = self.df['j0_y'].dropna().mean()
        heel_y = pd.concat([self.df['j19_y'], self.df['j22_y']]).dropna().mean()
        if head_y > heel_y:
            self.up_dir = 1      # Y-up  — floor at Y≈0, head at Y≈1.7
            print(INFO + f"Y direction: Y-UP  (head={head_y:.2f} > heel={heel_y:.2f})")
        else:
            self.up_dir = -1     # Y-down — raw OpenCV camera space
            print(INFO + f"Y direction: Y-DOWN (head={head_y:.2f} < heel={heel_y:.2f})")

        self.df.interpolate(method='linear', limit_direction='both', inplace=True)
        self.filter_data()

    def filter_data(self):
        # 4th order butterworth, 6Hz cutoff
        b, a = butter(4, 6 / (0.5 * self.fps), btype="low")
        for col in self.df.columns:
            if col.startswith("j"):
                # Forward-fill then back-fill any residual NaNs.
                # fillna(0) was here before but creates large spikes at boundaries
                # that survive the Butterworth filter and cause false event detections.
                self.df[col] = self.df[col].ffill().bfill()
                self.df[col] = filtfilt(b, a, self.df[col])

    def detect_events(self, side):
        prefix = side
        # select the correct column based on height_axis (Y)
        col_name = self.map[f"{prefix}_Heel_{self.height_axis.upper()}"]
        heel_height = self.df[col_name].values
        
        col_toe = self.map[f"{prefix}_Toe_{self.height_axis.upper()}"]
        toe_height = self.df[col_toe].values

        if self.up_dir == -1:
            strike_signal = heel_height # find maxima (ground contact)
        else:
            strike_signal = -heel_height # find maxima (if Y was Up)

        # distance: minimum frames between steps (0.5s * FPS)
        strikes, _ = find_peaks(strike_signal, distance=self.fps * 0.4, prominence=0.02)

        # toe off: max upward velocity
        vel_height = np.gradient(toe_height)
        
        if self.up_dir == -1:
            off_signal = -vel_height
        else:
            off_signal = vel_height

        # Use prominence (peak must stand out from surroundings) rather than
        # an absolute height threshold.  The heel-off event at ~33% creates a
        # small velocity blip; the real toe-off at ~62% creates a much larger
        # peak.  Prominence separates them; a flat height=0.01 threshold would
        # pick up the heel-off blip and the distance constraint would then
        # suppress the true toe-off (causing foot_off ≈ 34% instead of ≈ 62%).
        offs, _ = find_peaks(off_signal, prominence=0.015, distance=self.fps * 0.4)

        return np.sort(strikes), np.sort(offs)

    def calculate_full_metrics(self, strikes, offs, opp_strikes, opp_offs, side):
        if len(strikes) < 2:
            return None

        metrics = {k: [] for k in [
            "Cadence", "WalkingSpeed", "StrideTime", "StepTime",
            "OppFootOff", "OppFootContact", "FootOff",
            "SingleSupport", "DoubleSupport", "StrideLen",
            "StepLen", "StepWidth", "LimpIndex"
        ]}

        for i in range(len(strikes) - 1):
            start = strikes[i]
            end = strikes[i + 1]
            stride_dur = (end - start) / self.fps
            stride_frames = end - start

            if stride_dur == 0: continue

            # extract coordinates
            def get_pt(frame, name):
                return np.array([
                    self.df.iloc[frame][self.map[f"{name}_X"]],
                    self.df.iloc[frame][self.map[f"{name}_Y"]],
                    self.df.iloc[frame][self.map[f"{name}_Z"]]
                ])

            l_start = get_pt(start, "L_Heel")
            r_start = get_pt(start, "R_Heel")

            # spatial metrics
            p1 = get_pt(start, f"{side}_Heel")
            p2 = get_pt(end, f"{side}_Heel")
            
            # floor-projected stride length (ignore Y height) — result in metres
            stride_len = np.sqrt((p2[0]-p1[0])**2 + (p2[2]-p1[2])**2)

            # step length: Z-distance between HEELS at strike — metres
            step_len = abs(l_start[2] - r_start[2])
            
            # step width: X-distance — metres
            # l_heel_x_mean = self.df.iloc[start:end]['j19_x'].mean()
            # r_heel_x_mean = self.df.iloc[start:end]['j22_x'].mean()
            # step_width = abs(l_heel_x_mean - r_heel_x_mean)
            step_width = abs(l_start[0] - r_start[0])

            valid_offs = offs[(offs > start) & (offs < end)]
            foot_off_pct = np.nan
            if len(valid_offs) > 0:
                foot_off_pct = ((valid_offs[0] - start) / stride_frames) * 100

            valid_opp_s = opp_strikes[(opp_strikes > start) & (opp_strikes < end)]
            opp_con_pct = np.nan
            step_time = np.nan
            if len(valid_opp_s) > 0:
                opp_con_pct = ((valid_opp_s[0] - start) / stride_frames) * 100
                step_time = (valid_opp_s[0] - start) / self.fps

            valid_opp_o = opp_offs[(opp_offs > start) & (opp_offs < end)]
            opp_off_pct = np.nan
            if len(valid_opp_o) > 0:
                opp_off_pct = ((valid_opp_o[0] - start) / stride_frames) * 100

            # Single Support: time from opp foot off to opp foot contact — seconds
            single_supp = np.nan
            if not np.isnan(opp_con_pct) and not np.isnan(opp_off_pct):
                single_supp = (opp_con_pct - opp_off_pct) / 100.0 * stride_dur
            
            # Double Support: total double-support time per stride — seconds
            double_supp = np.nan
            if not np.isnan(foot_off_pct) and not np.isnan(opp_con_pct) and not np.isnan(opp_off_pct):
                double_supp = (opp_off_pct + (foot_off_pct - opp_con_pct)) / 100.0 * stride_dur

            limp = np.nan
            if not np.isnan(foot_off_pct):
                swing = 100 - foot_off_pct
                if swing > 0:
                    limp = foot_off_pct / swing

            metrics["StrideTime"].append(stride_dur)
            metrics["StrideLen"].append(stride_len)
            metrics["StepLen"].append(step_len)
            metrics["StepWidth"].append(step_width)
            metrics["WalkingSpeed"].append(stride_len / stride_dur)
            metrics["Cadence"].append((60 / stride_dur) * 2)
            metrics["StepTime"].append(step_time)
            metrics["FootOff"].append(foot_off_pct)
            metrics["OppFootContact"].append(opp_con_pct)
            metrics["OppFootOff"].append(opp_off_pct)
            metrics["SingleSupport"].append(single_supp)
            metrics["DoubleSupport"].append(double_supp)
            metrics["LimpIndex"].append(limp)

        return {k: np.nanmean(v) if len(v) > 0 else 0 for k, v in metrics.items()}

    def generate_vicon_tables(self):
        l_strikes, l_offs = self.detect_events("L")
        r_strikes, r_offs = self.detect_events("R")

        l_res = self.calculate_full_metrics(l_strikes, l_offs, r_strikes, r_offs, "L")
        r_res = self.calculate_full_metrics(r_strikes, r_offs, l_strikes, l_offs, "R")

        rows = []
        param_defs = [
            ("Cadence", "Cadence", "steps/min"),
            ("WalkingSpeed", "Walking Speed", "m/s"),
            ("StrideTime", "Stride Time", "s"),
            ("StepTime", "Step Time", "s"),
            ("OppFootOff", "Opposite Foot Off", "%"),
            ("OppFootContact", "Opposite Foot Contact", "%"),
            ("FootOff", "Foot Off", "%"),
            ("SingleSupport", "Single Support", "s"),
            ("DoubleSupport", "Double Support", "s"),
            ("StrideLen", "Stride Length", "m"),
            ("StepLen", "Step Length", "m"),
            ("StepWidth", "Step Width", "m"),
            ("LimpIndex", "Limp Index", "ratio"),
        ]

        def add_rows(res, ctx):
            if not res: return
            for k, name, unit in param_defs:
                rows.append({
                    "Subject": SUBJECT_NAME, "Context": ctx,
                    "Name": name, "Value": res.get(k, 0), "Units": unit
                })

        add_rows(l_res, "Left")
        add_rows(r_res, "Right")
        
        events_df = pd.DataFrame() 
        return pd.DataFrame(rows), events_df

def main():
    START_FRAME = None   # None = use all frames from the beginning
    END_FRAME   = None   # None = use all frames to the end

    analyzer = GaitAnalyzer(
        OUTPUT_CSV,
        fps=FPS_ANALYSIS,
        height_axis="y",
        start_frame=START_FRAME,
        end_frame=END_FRAME,
    )
    
    params_df, _ = analyzer.generate_vicon_tables()

    print("\n# Gait Cycle Parameters")
    print(params_df.to_markdown(index=True))

    gait_out = "gait-cycle-parameters"
    os.makedirs(gait_out, exist_ok=True)
    save_path = os.path.join(gait_out, f"{SUBJECT_NAME}_gait_{ROUND}.csv")
    params_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()