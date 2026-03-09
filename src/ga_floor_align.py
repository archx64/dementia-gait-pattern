import os, numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from src.utils_floor_align import FPS_ANALYSIS, SUBJECT_NAME, ROUND, INFO, OUTPUT_CSV

# OUTPUT_CSV = 'output/4.csv'

class GaitAnalyzer:
    def __init__(self, csv_path, fps, height_axis="y", up_direction=-1, start_frame=None, end_frame=None):
        """
        height_axis="y": In Computer Vision (OpenCV), Y is the vertical axis.
        up_direction=-1: In OpenCV, Y increases downwards. So 'Up' is negative.
        start_frame: The frame index where the person starts walking.
        end_frame: The frame index where the person stops walking.
        """
        self.fps = fps
        self.dt = 1 / fps
        self.height_axis = height_axis.lower()
        self.up_dir = up_direction

        print(INFO + f"Loading CSV: {csv_path}")
        self.df = pd.read_csv(csv_path)

        # crop dataframe base on frame index
        if start_frame is not None:
            self.df = self.df[self.df['frame_idx'] >= start_frame]
            print(INFO + f"Cropped start to frame >= {start_frame}")
            
        if end_frame is not None:
            self.df = self.df[self.df['frame_idx'] <= end_frame]
            print(INFO + f"Cropped end to frame <= {end_frame}")

        # reset index so that iloc indexing in later functions still works seamlessly
        self.df = self.df.reset_index(drop=True)

        # mapping WholeBody Keypoints
        self.map = {
            "L_Heel_X": "j19_x", "L_Heel_Y": "j19_y", "L_Heel_Z": "j19_z",
            "R_Heel_X": "j22_x", "R_Heel_Y": "j22_y", "R_Heel_Z": "j22_z",
            "L_Toe_X": "j17_x",  "L_Toe_Y": "j17_y",  "L_Toe_Z": "j17_z",
            "R_Toe_X": "j20_x",  "R_Toe_Y": "j20_y",  "R_Toe_Z": "j20_z",
        }

        # handle empty/NaN values if interpolation was OFF
        self.df.interpolate(method='linear', limit_direction='both', inplace=True)
        self.filter_data()

    def filter_data(self):
        # 4th order Butterworth, 6Hz cutoff
        b, a = butter(4, 6 / (0.5 * self.fps), btype="low")
        for col in self.df.columns:
            if col.startswith("j"):
                # fill remaining NaNs with 0 before filtering to prevent crash
                self.df[col].fillna(0, inplace=True)
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

        # distance: Minimum frames between steps (0.5s * FPS)
        strikes, _ = find_peaks(strike_signal, distance=self.fps * 0.4, prominence=0.02)

        # toe off: max upward velocity
        vel_height = np.gradient(toe_height)
        
        if self.up_dir == -1:
            off_signal = -vel_height
        else:
            off_signal = vel_height
            
        offs, _ = find_peaks(off_signal, height=0.01, distance=self.fps * 0.4)

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
            l_end = get_pt(end, "L_Heel")
            
            # spatial metrics
            p1 = get_pt(start, f"{side}_Heel")
            p2 = get_pt(end, f"{side}_Heel")
            
            # floor distance (ignore Y height)
            stride_len = np.sqrt((p2[0]-p1[0])**2 + (p2[2]-p1[2])**2) * 100 # m to cm

            # step length: Z-Distance between HEELS at strike
            step_len = abs(l_start[2] - r_start[2]) * 100 # Z-diff
            
            # step width: X-Distance
            step_width = abs(l_start[0] - r_start[0]) * 100

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

            # derived
            single_supp = np.nan
            if not np.isnan(opp_con_pct) and not np.isnan(opp_off_pct):
                single_supp = opp_con_pct - opp_off_pct
            
            double_supp = np.nan
            if not np.isnan(foot_off_pct) and not np.isnan(opp_con_pct) and not np.isnan(opp_off_pct):
                double_supp = opp_off_pct + (foot_off_pct - opp_con_pct)

            limp = np.nan
            if not np.isnan(foot_off_pct):
                swing = 100 - foot_off_pct
                if swing > 0:
                    limp = foot_off_pct / swing

            metrics["StrideTime"].append(stride_dur)
            metrics["StrideLen"].append(stride_len)
            metrics["StepLen"].append(step_len)
            metrics["StepWidth"].append(step_width)
            metrics["WalkingSpeed"].append((stride_len / 100) / stride_dur)
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
            ("SingleSupport", "Single Support", "%"),
            ("DoubleSupport", "Double Support", "%"),
            ("StrideLen", "Stride Length", "cm"),
            ("StepLen", "Step Length", "cm"),
            ("StepWidth", "Step Width", "cm"),
            ("LimpIndex", "Limp Index", "nan"),
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
    START_FRAME = 40   # set to None if we want to start from the beginning
    END_FRAME = 220     # set to None if we want to process until the end

    analyzer = GaitAnalyzer(
        OUTPUT_CSV, 
        fps=FPS_ANALYSIS, 
        height_axis="y", 
        up_direction=-1,
        start_frame=START_FRAME, 
        end_frame=END_FRAME
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
