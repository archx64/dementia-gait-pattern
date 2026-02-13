import os, numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import CubicSpline
from utils import FPS_ANALYSIS, OUTPUT_CSV, SUBJECT_NAME, ROUND, INFO, DEBUG

class GaitAnalyzer:
    def __init__(self, csv_path, original_fps, target_fps=100, height_axis="y"):
        self.original_fps = original_fps
        self.target_fps = target_fps  # We will upsample to this
        self.height_axis = height_axis.lower()

        # load and preprocess
        raw_df = pd.read_csv(csv_path)
        self.df = self.upsample_data(raw_df)
        
        self.fps = target_fps # Analysis now runs at 100Hz
        
        # wholebody keypoints mapping
        self.map = {
            "L_Heel_X": "j19_x", "L_Heel_Y": "j19_y", "L_Heel_Z": "j19_z",
            "R_Heel_X": "j22_x", "R_Heel_Y": "j22_y", "R_Heel_Z": "j22_z",
            "L_Toe_X": "j17_x",  "L_Toe_Y": "j17_y",  "L_Toe_Z": "j17_z",
            "R_Toe_X": "j20_x",  "R_Toe_Y": "j20_y",  "R_Toe_Z": "j20_z",
            "L_Hip_X": "j11_x",  "L_Hip_Z": "j11_z",
            "R_Hip_X": "j12_x",  "R_Hip_Z": "j12_z",
        }

        self.filter_data()
        self.walking_vector = self.calculate_walking_vector()

    def upsample_data(self, df):
        """Upsamples dataframe from 25Hz to 100Hz using Cubic Spline"""
        print(INFO + f"Upsampling data from {self.original_fps}Hz to {self.target_fps}Hz...")
        
        # Create old time axis
        n_frames = len(df)
        t_old = np.linspace(0, n_frames / self.original_fps, n_frames)
        
        # Create new time axis
        n_new = int(n_frames * (self.target_fps / self.original_fps))
        t_new = np.linspace(0, n_frames / self.original_fps, n_new)
        
        new_data = {}
        for col in df.columns:
            if col == "frame_idx":
                continue
                
            # Handle missing data (interpolate NaNs first)
            series = df[col].interpolate(method='linear', limit_direction='both')
            
            # Cubic Spline Interpolation
            cs = CubicSpline(t_old, series.values)
            new_data[col] = cs(t_new)
            
        new_df = pd.DataFrame(new_data)
        new_df["time"] = t_new
        return new_df

    def filter_data(self):
        # 4th order Butterworth, 6Hz cutoff
        b, a = butter(4, 6 / (0.5 * self.fps), btype="low")
        for col in self.df.columns:
            if col.startswith("j"):
                self.df[col] = filtfilt(b, a, self.df[col])

    def calculate_walking_vector(self):
        """Calculates the dominant direction of walking (2D vector in X-Z plane)"""
        # Average hip position
        hip_x = (self.df[self.map["L_Hip_X"]] + self.df[self.map["R_Hip_X"]]) / 2
        hip_z = (self.df[self.map["L_Hip_Z"]] + self.df[self.map["R_Hip_Z"]]) / 2
        
        # Fit line to path (X vs Z)
        # We use the middle 60% of data to avoid start/turn irregularities
        n = len(self.df)
        start, end = int(n*0.2), int(n*0.8)
        
        coeffs = np.polyfit(hip_z[start:end], hip_x[start:end], 1)
        # Vector is [dx, dz]. Since x = mz + c, dx = m, dz = 1
        vec = np.array([coeffs[0], 1.0]) 
        vec = vec / np.linalg.norm(vec) # Normalize
        
        print(INFO + f"Walking Vector (X, Z): {vec}")
        return vec

    def detect_events(self, side):
        prefix = side
        # Use Y axis (Height) for heel strike detection
        heel_h = self.df[self.map[f"{prefix}_Heel_{self.height_axis.upper()}"]].values
        toe_h = self.df[self.map[f"{prefix}_Toe_{self.height_axis.upper()}"]].values

        # Heel Strike: Local Minima in Height (Foot touching floor)
        # Since 'Y' is up in our aligned data, minima = floor contact
        strikes, _ = find_peaks(-heel_h, distance=self.fps * 0.5, prominence=0.005)

        # Toe Off: Max upward velocity of Toe
        vel_h = np.gradient(toe_h)
        offs, _ = find_peaks(vel_h, height=0.005, distance=self.fps * 0.5)

        return np.sort(strikes), np.sort(offs)

    def calculate_full_metrics(self, strikes, offs, opp_strikes, opp_offs, side):
        if len(strikes) < 2:
            return None

        metrics = {k: [] for k in ["Cadence", "WalkingSpeed", "StrideTime", "StepTime", "OppFootOff", "OppFootContact", "FootOff", "SingleSupport", "DoubleSupport", "StrideLen", "StepLen", "StepWidth", "LimpIndex"]}

        for i in range(len(strikes) - 1):
            start = strikes[i]
            end = strikes[i + 1]
            stride_dur = (end - start) / self.fps
            stride_frames = end - start

            if stride_dur == 0: continue

            # --- SPATIAL METRICS (Projected onto Walking Vector) ---
            
            # Step Length (Distance between heels projected on walking path)
            lx = self.df.iloc[start][self.map["L_Heel_X"]]
            lz = self.df.iloc[start][self.map["L_Heel_Z"]]
            rx = self.df.iloc[start][self.map["R_Heel_X"]]
            rz = self.df.iloc[start][self.map["R_Heel_Z"]]
            
            delta_x = lx - rx
            delta_z = lz - rz
            
            # Project vector(dx, dz) onto walking_vector
            foot_vec = np.array([delta_x, delta_z])
            step_len = abs(np.dot(foot_vec, self.walking_vector)) * 100 # cm
            
            # Step Width (Distance perpendicular to walking path)
            # Perpendicular vector is (-z, x)
            perp_vec = np.array([-self.walking_vector[1], self.walking_vector[0]])
            step_width = abs(np.dot(foot_vec, perp_vec)) * 100 # cm

            # Stride Length (Same foot displacement)
            h_x_col, h_z_col = self.map[f"{side}_Heel_X"], self.map[f"{side}_Heel_Z"]
            p1 = np.array([self.df.iloc[start][h_x_col], self.df.iloc[start][h_z_col]])
            p2 = np.array([self.df.iloc[end][h_x_col], self.df.iloc[end][h_z_col]])
            stride_vec = p2 - p1
            stride_len = abs(np.dot(stride_vec, self.walking_vector)) * 100 # cm

            # --- TEMPORAL METRICS ---
            
            # Own Foot Off
            valid_offs = offs[(offs > start) & (offs < end)]
            foot_off_pct = ((valid_offs[0] - start) / stride_frames) * 100 if len(valid_offs) > 0 else np.nan

            # Opp Contact
            valid_opp_s = opp_strikes[(opp_strikes > start) & (opp_strikes < end)]
            opp_con_pct = np.nan
            step_time = np.nan
            if len(valid_opp_s) > 0:
                opp_con_pct = ((valid_opp_s[0] - start) / stride_frames) * 100
                step_time = (valid_opp_s[0] - start) / self.fps

            # Opp Off
            valid_opp_o = opp_offs[(opp_offs > start) & (opp_offs < end)]
            opp_off_pct = ((valid_opp_o[0] - start) / stride_frames) * 100 if len(valid_opp_o) > 0 else np.nan

            # Derived
            single_supp = (opp_con_pct - opp_off_pct) if (not np.isnan(opp_con_pct) and not np.isnan(opp_off_pct)) else np.nan
            double_supp = np.nan
            if not np.isnan(foot_off_pct) and not np.isnan(opp_con_pct) and not np.isnan(opp_off_pct):
                double_supp = opp_off_pct + (foot_off_pct - opp_con_pct)
            
            limp = np.nan
            if not np.isnan(foot_off_pct):
                swing = 100 - foot_off_pct
                if swing > 0: limp = foot_off_pct / swing

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
                rows.append({"Subject": SUBJECT_NAME, "Context": ctx, "Name": name, "Value": res.get(k, 0), "Units": unit})

        add_rows(l_res, "Left")
        add_rows(r_res, "Right")
        return pd.DataFrame(rows)

def main():
    # Target FPS 100Hz is standard for Gait Labs
    analyzer = GaitAnalyzer(OUTPUT_CSV, original_fps=FPS_ANALYSIS, target_fps=100, height_axis="y")
    params_df = analyzer.generate_vicon_tables()

    print("\n# Gait Cycle Parameters")
    print(params_df.to_markdown(index=False))

    gait_out = "gait-cycle-parameters"
    os.makedirs(gait_out, exist_ok=True)
    save_path = os.path.join(gait_out, f"{SUBJECT_NAME}_gait_{ROUND}.csv")
    params_df.to_csv(save_path, index=False)
    print(INFO + f'saved gait csv file to: {save_path}')

if __name__ == "__main__":
    main()