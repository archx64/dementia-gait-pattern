import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from utils import FPS_ANALYSIS, OUTPUT_CSV, SUBJECT_NAME

class GaitAnalyzer:
    def __init__(self, csv_path, fps, height_axis='z', up_direction=1):
        self.fps = fps
        self.dt = 1 / fps
        self.height_axis = height_axis.lower()
        self.up_dir = up_direction
        
        self.df = pd.read_csv(csv_path)
        
        # Mapping WholeBody Keypoints
        self.map = {
            'L_Heel_X': 'j19_x', 'L_Heel_Y': 'j19_y', 'L_Heel_Z': 'j19_z',
            'R_Heel_X': 'j22_x', 'R_Heel_Y': 'j22_y', 'R_Heel_Z': 'j22_z',
            'L_Toe_X':  'j17_x', 'L_Toe_Y':  'j17_y', 'L_Toe_Z':  'j17_z',
            'R_Toe_X':  'j20_x', 'R_Toe_Y':  'j20_y', 'R_Toe_Z':  'j20_z',
        }
        
        self.filter_data()

    def filter_data(self):
        # 4th order Butterworth, 6Hz cutoff
        b, a = butter(4, 6 / (0.5 * self.fps), btype='low')
        for col in self.df.columns:
            if col.startswith('j'):
                self.df[col] = filtfilt(b, a, self.df[col])

    def detect_events(self, side):
        prefix = side
        heel_z = self.df[self.map[f'{prefix}_Heel_{self.height_axis.upper()}']].values
        toe_z = self.df[self.map[f'{prefix}_Toe_{self.height_axis.upper()}']].values

        # 1. Heel Strike (Minima of Heel Z)
        strike_signal = -heel_z if self.up_dir == 1 else heel_z
        strikes, _ = find_peaks(strike_signal, distance=self.fps*0.5, prominence=0.01)

        # 2. Toe Off (Max Upward Velocity of Toe Z)
        vel_z = np.gradient(toe_z)
        off_signal = vel_z if self.up_dir == 1 else -vel_z
        offs, _ = find_peaks(off_signal, height=0.01, distance=self.fps*0.5)

        return np.sort(strikes), np.sort(offs)

    def calculate_full_metrics(self, strikes, offs, opp_strikes, opp_offs, side):
        if len(strikes) < 2: return None

        metrics = {k: [] for k in [
            'Cadence', 'WalkingSpeed', 'StrideTime', 'StepTime',
            'OppFootOff', 'OppFootContact', 'FootOff', 
            'SingleSupport', 'DoubleSupport', 
            'StrideLen', 'StepLen', 'StepWidth', 'LimpIndex'
        ]}

        for i in range(len(strikes) - 1):
            start = strikes[i]
            end = strikes[i+1]
            stride_dur = (end - start) / self.fps
            stride_frames = end - start

            if stride_dur == 0: continue

            # --- SPATIAL ---
            lx = self.df.iloc[start][self.map['L_Heel_X']]
            ly = self.df.iloc[start][self.map['L_Heel_Y']]
            rx = self.df.iloc[start][self.map['R_Heel_X']]
            ry = self.df.iloc[start][self.map['R_Heel_Y']]

            # Step Length & Width
            step_len = np.sqrt((lx - rx)**2 + (ly - ry)**2) * 100
            step_width = abs(ly - ry) * 100

            # Stride Length
            h_x, h_y = self.map[f'{side}_Heel_X'], self.map[f'{side}_Heel_Y']
            p1 = self.df.iloc[start][[h_x, h_y]]
            p2 = self.df.iloc[end][[h_x, h_y]]
            stride_len = np.linalg.norm(p2 - p1) * 100

            # --- TEMPORAL ---
            # Own Foot Off
            valid_offs = offs[(offs > start) & (offs < end)]
            foot_off_pct = np.nan
            if len(valid_offs) > 0:
                foot_off_pct = ((valid_offs[0] - start) / stride_frames) * 100

            # Opp Contact
            valid_opp_s = opp_strikes[(opp_strikes > start) & (opp_strikes < end)]
            opp_con_pct = np.nan
            step_time = np.nan
            if len(valid_opp_s) > 0:
                opp_con_pct = ((valid_opp_s[0] - start) / stride_frames) * 100
                step_time = (valid_opp_s[0] - start) / self.fps

            # Opp Off (Initial Double Support End)
            valid_opp_o = opp_offs[(opp_offs > start) & (opp_offs < end)]
            opp_off_pct = np.nan
            if len(valid_opp_o) > 0:
                opp_off_pct = ((valid_opp_o[0] - start) / stride_frames) * 100

            # Derived
            single_supp = opp_con_pct - opp_off_pct if (not np.isnan(opp_con_pct) and not np.isnan(opp_off_pct)) else np.nan
            double_supp = np.nan
            if not np.isnan(foot_off_pct) and not np.isnan(opp_con_pct) and not np.isnan(opp_off_pct):
                double_supp = opp_off_pct + (foot_off_pct - opp_con_pct)

            limp = np.nan
            if not np.isnan(foot_off_pct):
                swing = 100 - foot_off_pct
                if swing > 0: limp = foot_off_pct / swing

            # Append
            metrics['StrideTime'].append(stride_dur)
            metrics['StrideLen'].append(stride_len)
            metrics['StepLen'].append(step_len)
            metrics['StepWidth'].append(step_width)
            # metrics['WalkingSpeed'].append((stride_len/100)/stride_dur)
            metrics['WalkingSpeed'].append((stride_len/10)/stride_dur)
            metrics['Cadence'].append((60/stride_dur)*2)
            metrics['StepTime'].append(step_time)
            metrics['FootOff'].append(foot_off_pct)
            metrics['OppFootContact'].append(opp_con_pct)
            metrics['OppFootOff'].append(opp_off_pct)
            metrics['SingleSupport'].append(single_supp)
            metrics['DoubleSupport'].append(double_supp)
            metrics['LimpIndex'].append(limp)

        # Average
        return {k: np.nanmean(v) if len(v) > 0 else 0 for k, v in metrics.items()}

    def generate_vicon_tables(self):
        l_strikes, l_offs = self.detect_events('L')
        r_strikes, r_offs = self.detect_events('R')

        # 1. EVENTS TABLE
        events = []
        for f in l_strikes: events.append({'Subject': SUBJECT_NAME, 'Context': 'Left', 'Name': 'Foot Strike', 'Frame': f, 'Time (s)': f/self.fps})
        for f in l_offs:    events.append({'Subject': SUBJECT_NAME, 'Context': 'Left', 'Name': 'Foot Off',    'Frame': f, 'Time (s)': f/self.fps})
        for f in r_strikes: events.append({'Subject': SUBJECT_NAME, 'Context': 'Right', 'Name': 'Foot Strike', 'Frame': f, 'Time (s)': f/self.fps})
        for f in r_offs:    events.append({'Subject': SUBJECT_NAME, 'Context': 'Right', 'Name': 'Foot Off',    'Frame': f, 'Time (s)': f/self.fps})
        
        events_df = pd.DataFrame(events).sort_values(by='Frame').reset_index(drop=True)
        events_df['Description'] = events_df['Name'].map({
            'Foot Strike': 'Heel touches ground',
            'Foot Off': 'Toe leaves ground'
        })

        # 2. PARAMETERS TABLE
        l_res = self.calculate_full_metrics(l_strikes, l_offs, r_strikes, r_offs, 'L')
        r_res = self.calculate_full_metrics(r_strikes, r_offs, l_strikes, l_offs, 'R')
        
        rows = []
        param_defs = [
            ('Cadence', 'Cadence', 'steps/min'),
            ('WalkingSpeed', 'Walking Speed', 'm/s'),
            ('StrideTime', 'Stride Time', 's'),
            ('StepTime', 'Step Time', 's'),
            ('OppFootOff', 'Opposite Foot Off', '%'),
            ('OppFootContact', 'Opposite Foot Contact', '%'),
            ('FootOff', 'Foot Off', '%'),
            ('SingleSupport', 'Single Support', '%'),
            ('DoubleSupport', 'Double Support', '%'),
            ('StrideLen', 'Stride Length', 'cm'),
            ('StepLen', 'Step Length', 'cm'),
            ('StepWidth', 'Step Width', 'cm'),
            ('LimpIndex', 'Limp Index', 'nan'),
        ]

        def add_rows(res, ctx):
            if not res: return
            for k, name, unit in param_defs:
                rows.append({
                    'Subject': SUBJECT_NAME, 'Context': ctx, 
                    'Name': name, 'Value': res.get(k, 0), 'Units': unit
                })

        add_rows(l_res, 'Left')
        add_rows(r_res, 'Right')
        
        params_df = pd.DataFrame(rows)
        return params_df, events_df

def main():
    analyzer = GaitAnalyzer(OUTPUT_CSV, fps=FPS_ANALYSIS)
    params_df, events_df = analyzer.generate_vicon_tables()
    
    print("\n# Gait Cycle Parameters")
    print(params_df.to_markdown(index=True))
    
    print("\n# Events Table")
    print(events_df.to_markdown(index=True))
    
    # Save files
    params_df.to_csv("gait_parameters.csv", index=False)
    events_df.to_csv("gait_events.csv", index=False)

if __name__ == '__main__':
    main()