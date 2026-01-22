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
        
        # Load Data
        self.df = pd.read_csv(csv_path)
        
        # 1. Define Raw Keypoints (COCO Format)
        self.raw_cols = [
            'j15_x', 'j15_y', 'j15_z', # Left Ankle
            'j16_x', 'j16_y', 'j16_z', # Right Ankle
            'j11_x', 'j11_y', 'j11_z', # Left Hip
            'j12_x', 'j12_y', 'j12_z'  # Right Hip
        ]
        
        # 2. Filter Data
        self.filter_data()

        # 3. Compute Hip Center
        self.df['Hip_Center_X'] = (self.df['j11_x'] + self.df['j12_x']) / 2
        self.df['Hip_Center_Y'] = (self.df['j11_y'] + self.df['j12_y']) / 2
        self.df['Hip_Center_Z'] = (self.df['j11_z'] + self.df['j12_z']) / 2

        # 4. Map columns
        self.map = {
            'L_Ankle_X': 'j15_x', 'L_Ankle_Y': 'j15_y', 'L_Ankle_Z': 'j15_z',
            'R_Ankle_X': 'j16_x', 'R_Ankle_Y': 'j16_y', 'R_Ankle_Z': 'j16_z',
            'Hip_Center_X': 'Hip_Center_X', 
            'Hip_Center_Y': 'Hip_Center_Y', 
            'Hip_Center_Z': 'Hip_Center_Z'
        }

    def filter_data(self):
        # 4th order butterworth filter, 6Hz cutoff
        b, a = butter(4, 6 / (0.5 * self.fps), btype='low')
        
        for col in self.raw_cols:
            if col in self.df.columns:
                self.df[col] = filtfilt(b, a, self.df[col])
            else:
                pass # Silently skip missing columns

    def detect_events(self, side='L'):
        prefix = 'L' if side == 'L' else 'R'
        z_col = self.map[f'{prefix}_Ankle_{self.height_axis.upper()}']

        # Heel Strike (Minima)
        z_signal = self.df[z_col].values
        strike_signal = -z_signal if self.up_dir == 1 else z_signal
        strikes, _ = find_peaks(strike_signal, distance=self.fps*0.5)

        # Toe Off (Max Upward Velocity)
        vel_z = np.gradient(z_signal)
        off_signal = vel_z if self.up_dir == 1 else -vel_z
        # Look for peaks in velocity (foot kicking up)
        offs, _ = find_peaks(off_signal, height=0.01, distance=self.fps*0.5)
        
        return np.sort(strikes), np.sort(offs)

    def generate_vicon_tables(self):
        l_strikes, l_offs = self.detect_events('L')
        r_strikes, r_offs = self.detect_events('R')
        
        # --- 1. GENERATE EVENTS TABLE ---
        events_list = []
        for f in l_strikes: events_list.append({'Context': 'Left', 'Name': 'Foot Strike', 'Frame': f, 'Time (s)': f/self.fps})
        for f in l_offs:    events_list.append({'Context': 'Left', 'Name': 'Foot Off',    'Frame': f, 'Time (s)': f/self.fps})
        for f in r_strikes: events_list.append({'Context': 'Right', 'Name': 'Foot Strike', 'Frame': f, 'Time (s)': f/self.fps})
        for f in r_offs:    events_list.append({'Context': 'Right', 'Name': 'Foot Off',    'Frame': f, 'Time (s)': f/self.fps})
        
        events_df = pd.DataFrame(events_list).sort_values(by='Frame').reset_index(drop=True)
        events_df['Subject'] = SUBJECT_NAME
        events_df['Description'] = events_df['Name'].map({
            'Foot Strike': 'The instant the heel strikes the ground',
            'Foot Off': 'The instant the toe leaves the ground'
        })
        events_df = events_df[['Subject', 'Context', 'Name', 'Time (s)', 'Description']]

        # --- 2. GENERATE PARAMETERS TABLE ---
        params_list = []
        
        def calc_side_metrics(strikes, offs, opp_strikes, opp_offs, side):
            if len(strikes) < 2: return None
            
            data = {k: [] for k in ['StepLen', 'StrideLen', 'StepTime', 'StrideTime', 
                                    'StepWidth', 'OppFootOff', 'OppFootContact', 
                                    'FootOff', 'SingleSupport', 'DoubleSupport']}
            
            for i in range(len(strikes) - 1):
                start = strikes[i]
                end = strikes[i+1]
                stride_frames = end - start
                
                # Basic Time Metrics
                data['StrideTime'].append(stride_frames / self.fps)
                
                # --- SPATIAL METRICS (at Start frame) ---
                # Coordinates
                lx = self.df.iloc[start][self.map['L_Ankle_X']]
                ly = self.df.iloc[start][self.map['L_Ankle_Y']]
                rx = self.df.iloc[start][self.map['R_Ankle_X']]
                ry = self.df.iloc[start][self.map['R_Ankle_Y']]
                
                # Step Length (Distance between ankles)
                dist_cm = np.sqrt((lx-rx)**2 + (ly-ry)**2) * 100
                data['StepLen'].append(dist_cm)
                
                # Step Width (Abs diff in Y - assuming walking along X)
                width_cm = abs(ly - ry) * 100
                data['StepWidth'].append(width_cm)

                # Stride Length (Approx 2 * Step for now, or displacement)
                # Calculating displacement of the SAME foot
                foot_col_x = self.map[f'{side[0]}_Ankle_X']
                start_x = self.df.iloc[start][foot_col_x]
                end_x = self.df.iloc[end][foot_col_x]
                stride_cm = abs(end_x - start_x) * 100
                if stride_cm < 10: stride_cm = dist_cm * 2 # Fallback for treadmill
                data['StrideLen'].append(stride_cm)

                # --- TEMPORAL EVENTS (Percentages) ---
                # 1. Own Foot Off (Stance Phase end)
                # Find the 'Foot Off' that happens strictly INSIDE this stride
                valid_offs = offs[(offs > start) & (offs < end)]
                if len(valid_offs) > 0:
                    own_off = valid_offs[0]
                    pct_off = (own_off - start) / stride_frames * 100
                    data['FootOff'].append(pct_off)
                else:
                    data['FootOff'].append(np.nan)

                # 2. Opposite Foot Events
                # Opp Strike (Step Time)
                valid_opp_strikes = opp_strikes[(opp_strikes > start) & (opp_strikes < end)]
                if len(valid_opp_strikes) > 0:
                    opp_strike = valid_opp_strikes[0]
                    
                    # Step Time
                    step_time = (opp_strike - start) / self.fps
                    data['StepTime'].append(step_time)
                    
                    # Opp Contact %
                    pct_opp_contact = (opp_strike - start) / stride_frames * 100
                    data['OppFootContact'].append(pct_opp_contact)
                    
                    # Opp Off % (Must happen before Opp Contact usually, or right after start)
                    # We look for Opp Off between Start and Opp Strike
                    valid_opp_offs = opp_offs[(opp_offs > start) & (opp_offs < opp_strike)]
                    if len(valid_opp_offs) > 0:
                        opp_off = valid_opp_offs[0]
                        pct_opp_off = (opp_off - start) / stride_frames * 100
                        data['OppFootOff'].append(pct_opp_off)
                        
                        # Derived Support Phases
                        single_supp = pct_opp_contact - pct_opp_off
                        data['SingleSupport'].append(single_supp)
                        data['DoubleSupport'].append(100 - single_supp)
                    else:
                        data['OppFootOff'].append(np.nan)
                        data['SingleSupport'].append(np.nan)
                        data['DoubleSupport'].append(np.nan)
                else:
                    data['StepTime'].append(np.nan)
                    data['OppFootContact'].append(np.nan)
            
            # --- AGGREGATE ---
            # Remove NaNs before averaging
            results = {}
            for k, v in data.items():
                clean_v = [x for x in v if not np.isnan(x)]
                results[k] = np.mean(clean_v) if clean_v else 0

            # Derived Globals
            results['Cadence'] = 60 / results['StepTime'] if results['StepTime'] > 0 else 0
            results['WalkingSpeed'] = (results['StrideLen'] / 100) / results['StrideTime'] if results['StrideTime'] > 0 else 0
            
            # Limp Index (Simple Stance Time symmetry approximation)
            # Limp = Stance / Swing
            # Stance % = FootOff %
            if results['FootOff'] > 0:
                results['LimpIndex'] = results['FootOff'] / (100 - results['FootOff'])
            else:
                results['LimpIndex'] = 0

            return results

        # Calculate Both Sides
        # Notice we pass ALL events to both functions
        l_res = calc_side_metrics(l_strikes, l_offs, r_strikes, r_offs, 'Left')
        r_res = calc_side_metrics(r_strikes, r_offs, l_strikes, l_offs, 'Right')
        
        def add_rows(res, context):
            if not res: return
            # Mapping Key -> Display Name
            rows = [
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
            
            for key, name, unit in rows:
                val = res.get(key, 0)
                params_list.append([SUBJECT_NAME, context, name, val, unit])

        add_rows(l_res, 'Left')
        add_rows(r_res, 'Right')
        
        params_df = pd.DataFrame(params_list, columns=['Subject', 'Context', 'Name', 'Value', 'Units'])
        return params_df, events_df

def main():
    analyzer = GaitAnalyzer(OUTPUT_CSV, fps=FPS_ANALYSIS)
    params_df, events_df = analyzer.generate_vicon_tables()
    
    print("\n# Gait Cycle Parameters")
    print(params_df.to_markdown())
    
    print("\n\n# Events")
    print(events_df.to_markdown())

if __name__ == '__main__':
    main()