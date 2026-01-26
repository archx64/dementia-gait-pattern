import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from utils import FPS_ANALYSIS, OUTPUT_CSV, SUBJECT_NAME

# --- CONFIGURATION ---
SUBJECT_HEIGHT_CM = 175.0  # Enter your real height in cm

class GaitAnalyzer:
    def __init__(self, csv_path, fps, height_axis='z'):
        self.fps = fps
        self.dt = 1 / fps
        self.height_axis = height_axis.lower()
        
        # Load Data
        self.df = pd.read_csv(csv_path)
        
        # 1. Define Columns
        self.raw_cols = [
            'j15_x', 'j15_y', 'j15_z', # Left Ankle
            'j16_x', 'j16_y', 'j16_z', # Right Ankle
            'j11_x', 'j11_y', 'j11_z', # Left Hip
            'j12_x', 'j12_y', 'j12_z', # Right Hip
            'j0_x', 'j0_y', 'j0_z'     # Nose
        ]
        
        # 2. Filter Data
        self.filter_data()

        # 3. AUTO-DETECT ORIENTATION (Crucial Fix)
        # We determine if 'Up' means increasing values or decreasing values
        self.up_dir = self.determine_up_direction()

        # 4. CALCULATE SCALE FACTOR (Fixed)
        self.scale_factor = self.calculate_scale_factor()
        print(f"--- CALIBRATION INFO ---")
        print(f"Subject Real Height: {SUBJECT_HEIGHT_CM} cm")
        print(f"Detected 'Up' Direction: {self.up_dir} (1=Normal, -1=Inverted)")
        print(f"Scale Factor: {self.scale_factor:.4f}")
        print("------------------------")

        # 5. Map columns
        self.map = {
            'L_Ankle_X': 'j15_x', 'L_Ankle_Y': 'j15_y', 'L_Ankle_Z': 'j15_z',
            'R_Ankle_X': 'j16_x', 'R_Ankle_Y': 'j16_y', 'R_Ankle_Z': 'j16_z'
        }

    def filter_data(self):
        b, a = butter(4, 6 / (0.5 * self.fps), btype='low')
        for col in self.raw_cols:
            if col in self.df.columns:
                self.df[col] = filtfilt(b, a, self.df[col])

    def determine_up_direction(self):
        """Auto-detects if the vertical axis is positive-up or negative-up."""
        z_col = 2 if self.height_axis == 'z' else 1
        nose_mean = self.df[f'j0_{self.height_axis}'].mean()
        ankle_mean = (self.df[f'j15_{self.height_axis}'].mean() + self.df[f'j16_{self.height_axis}'].mean()) / 2
        
        # If Nose is numerically higher than Ankle, Up is Positive (1)
        # If Nose is numerically lower than Ankle, Up is Negative (-1)
        return 1 if nose_mean > ankle_mean else -1

    def calculate_scale_factor(self):
        """Calculates ratio using Absolute difference to handle inverted axes."""
        nose = self.df[f'j0_{self.height_axis}'].values
        l_ank = self.df[f'j15_{self.height_axis}'].values
        r_ank = self.df[f'j16_{self.height_axis}'].values
        avg_ank = (l_ank + r_ank) / 2
        
        # FIX: Use ABS() to ensure height is always positive distance
        skeleton_heights = np.abs(nose - avg_ank)
        
        avg_skeleton_height = np.nanmedian(skeleton_heights)
        
        if avg_skeleton_height < 0.001: return 1.0 # Avoid div by zero
        return SUBJECT_HEIGHT_CM / avg_skeleton_height

    def detect_events(self, side='L'):
        prefix = 'L' if side == 'L' else 'R'
        z_col = self.map[f'{prefix}_Ankle_{self.height_axis.upper()}']
        z_signal = self.df[z_col].values
        
        # Heel Strike: Local Minima (Lowest Z)
        # If up_dir is 1 (Z goes up), minima is bottom.
        # If up_dir is -1 (Z goes down), maxima is bottom.
        strike_signal = -z_signal if self.up_dir == 1 else z_signal
        strikes, _ = find_peaks(strike_signal, distance=self.fps*0.25)

        # Toe Off: Max Upward Velocity
        vel_z = np.gradient(z_signal)
        # If up_dir is 1, Upward velocity is Positive.
        # If up_dir is -1, Upward velocity is Negative.
        off_signal = vel_z if self.up_dir == 1 else -vel_z
        
        # Height 0.01 ensures we only catch significant kick-offs
        offs, _ = find_peaks(off_signal, height=0.01, distance=self.fps*0.25)
        
        return np.sort(strikes), np.sort(offs)

    def generate_vicon_tables(self):
        l_strikes, l_offs = self.detect_events('L')
        r_strikes, r_offs = self.detect_events('R')
        
        # --- 1. EVENTS ---
        events_list = []
        for f in l_strikes: events_list.append({'Context': 'Left', 'Name': 'Foot Strike', 'Frame': f, 'Time (s)': f/self.fps})
        for f in l_offs:    events_list.append({'Context': 'Left', 'Name': 'Foot Off',    'Frame': f, 'Time (s)': f/self.fps})
        for f in r_strikes: events_list.append({'Context': 'Right', 'Name': 'Foot Strike', 'Frame': f, 'Time (s)': f/self.fps})
        for f in r_offs:    events_list.append({'Context': 'Right', 'Name': 'Foot Off',    'Frame': f, 'Time (s)': f/self.fps})
        
        events_df = pd.DataFrame(events_list).sort_values(by='Frame').reset_index(drop=True)
        events_df['Subject'] = SUBJECT_NAME
        events_df['Description'] = events_df['Name'].map({'Foot Strike': 'Heel strikes ground', 'Foot Off': 'Toe leaves ground'})
        events_df = events_df[['Subject', 'Context', 'Name', 'Time (s)', 'Description']]

        # --- 2. PARAMETERS ---
        params_list = []
        
        def calc_side_metrics(strikes, offs, opp_strikes, opp_offs, side):
            if len(strikes) < 2: return None
            keys = ['StepLen', 'StrideLen', 'StepTime', 'StrideTime', 'StepWidth', 'FootOff', 'OppFootContact', 'OppFootOff', 'SingleSupport', 'DoubleSupport']
            data = {k: [] for k in keys}
            
            own_x, own_y = self.map[f'{side[0]}_Ankle_X'], self.map[f'{side[0]}_Ankle_Y']
            opp_side = 'Right' if side == 'Left' else 'Left'
            opp_x, opp_y = self.map[f'{opp_side[0]}_Ankle_X'], self.map[f'{opp_side[0]}_Ankle_Y']

            for i in range(len(strikes) - 1):
                start, end = strikes[i], strikes[i+1]
                stride_frames = end - start
                
                # Time
                data['StrideTime'].append(stride_frames / self.fps)
                
                # --- GEOMETRY ---
                p1 = np.array([self.df.iloc[start][own_x], self.df.iloc[start][own_y]])
                p2 = np.array([self.df.iloc[end][own_x], self.df.iloc[end][own_y]])
                
                stride_vec = p2 - p1
                stride_raw = np.linalg.norm(stride_vec)
                data['StrideLen'].append(stride_raw * self.scale_factor) # Always positive

                # Direction Vector
                if stride_raw < 0.001: unit_vec = np.array([1, 0])
                else: unit_vec = stride_vec / stride_raw
                perp_vec = np.array([-unit_vec[1], unit_vec[0]])

                # Step Calc
                prev_opp = opp_strikes[opp_strikes < start]
                if len(prev_opp) > 0:
                    p_opp = np.array([self.df.iloc[prev_opp[-1]][opp_x], self.df.iloc[prev_opp[-1]][opp_y]])
                    step_vec = p1 - p_opp
                    
                    # FIX: Use ABS for step length/width magnitude
                    s_len = abs(np.dot(step_vec, unit_vec))
                    s_wid = abs(np.dot(step_vec, perp_vec))
                    
                    data['StepLen'].append(s_len * self.scale_factor)
                    data['StepWidth'].append(s_wid * self.scale_factor)
                else:
                    data['StepLen'].append(np.nan); data['StepWidth'].append(np.nan)

                # --- PERCENTAGES ---
                valid_offs = offs[(offs > start) & (offs < end)]
                if len(valid_offs) > 0:
                    data['FootOff'].append((valid_offs[0] - start) / stride_frames * 100)
                else:
                    data['FootOff'].append(np.nan)

                valid_opp = opp_strikes[(opp_strikes > start) & (opp_strikes < end)]
                if len(valid_opp) > 0:
                    opp_s = valid_opp[0]
                    data['StepTime'].append((opp_s - start) / self.fps)
                    data['OppFootContact'].append((opp_s - start) / stride_frames * 100)
                    
                    v_opp_off = opp_offs[(opp_offs > start) & (opp_offs < opp_s)]
                    if len(v_opp_off) > 0:
                        pct_opp_off = (v_opp_off[0] - start) / stride_frames * 100
                        data['OppFootOff'].append(pct_opp_off)
                        data['SingleSupport'].append(data['OppFootContact'][-1] - pct_opp_off)
                        data['DoubleSupport'].append(100 - data['SingleSupport'][-1])
                    else:
                        data['OppFootOff'].append(np.nan); data['SingleSupport'].append(np.nan); data['DoubleSupport'].append(np.nan)
                else:
                    data['StepTime'].append(np.nan); data['OppFootContact'].append(np.nan)

            # Average
            res = {k: np.mean([x for x in v if not np.isnan(x)]) if any(not np.isnan(x) for x in v) else 0 for k, v in data.items()}
            
            # Globals
            res['Cadence'] = 60 / res['StepTime'] if res['StepTime'] > 0 else 0
            res['WalkingSpeed'] = (res['StrideLen'] / 100) / res['StrideTime'] if res['StrideTime'] > 0 else 0
            
            # Limp Index (Stance Time / Swing Time)
            # Stance Time approx = FootOff Time
            if 0 < res['FootOff'] < 100:
                res['LimpIndex'] = res['FootOff'] / (100 - res['FootOff'])
            else:
                res['LimpIndex'] = 0
            
            return res

        l_res = calc_side_metrics(l_strikes, l_offs, r_strikes, r_offs, 'Left')
        r_res = calc_side_metrics(r_strikes, r_offs, l_strikes, l_offs, 'Right')
        
        def add_rows(res, ctx):
            if not res: return
            rows = [('Cadence', 'Cadence', 'steps/min'), ('WalkingSpeed', 'Walking Speed', 'm/s'),
                    ('StrideTime', 'Stride Time', 's'), ('StepTime', 'Step Time', 's'),
                    ('StrideLen', 'Stride Length', 'cm'), ('StepLen', 'Step Length', 'cm'),
                    ('StepWidth', 'Step Width', 'cm'), ('FootOff', 'Foot Off', '%'),
                    ('OppFootContact', 'Opposite Foot Contact', '%'), ('OppFootOff', 'Opposite Foot Off', '%'),
                    ('SingleSupport', 'Single Support', '%'), ('DoubleSupport', 'Double Support', '%'),
                    ('LimpIndex', 'Limp Index', 'ratio')]
            for k, n, u in rows: params_list.append([SUBJECT_NAME, ctx, n, res.get(k, 0), u])

        add_rows(l_res, 'Left')
        add_rows(r_res, 'Right')
        return pd.DataFrame(params_list, columns=['Subject', 'Context', 'Name', 'Value', 'Units']), events_df

if __name__ == '__main__':
    analyzer = GaitAnalyzer(OUTPUT_CSV, fps=FPS_ANALYSIS)
    p_df, e_df = analyzer.generate_vicon_tables()
    print("\n# Gait Cycle Parameters"); print(p_df.to_markdown())
    print("\n# Events"); print(e_df.to_markdown())