import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from utils import FPS, GAITANALYSIS_CSV, OUTPUT_CSV, WARNING, SUBJECT_NAME

class GaitAnalyzer:
    def __init__(self, csv_path, fps=30, height_axis='z', up_direction=1):
        """
        fps: Frames per Second of your video
        height_axis: 'z' (Standard for 3D lifting)
        up_direction: 1 (Standard for Z-up coordinates)
        """
        self.fps = fps
        self.dt = 1 / fps
        self.height_axis = height_axis.lower()
        self.up_dir = up_direction
        
        # load data
        self.df = pd.read_csv(csv_path)
        
        # define raw keypoints
        # perform filtering on these before calculating hip center
        self.raw_cols = [
            'j15_x', 'j15_y', 'j15_z', # Left ankle
            'j16_x', 'j16_y', 'j16_z', # Right ankle
            'j11_x', 'j11_y', 'j11_z', # Left hip
            'j12_x', 'j12_y', 'j12_z'  # Right hip
        ]
        
        # apply low-pass filter to smooth jittery skeleton data
        
        # compute hip center (average of j11 and j12)
        # we do this after filtering for smoother results
        self.df['Hip_Center_X'] = (self.df['j11_x'] + self.df['j12_x']) / 2
        self.df['Hip_Center_Y'] = (self.df['j11_y'] + self.df['j12_y']) / 2
        self.df['Hip_Center_Z'] = (self.df['j11_z'] + self.df['j12_z']) / 2

        # create mapping for easy access in methods
        self.map = {
            'L_Ankle_X': 'j15_x', 'L_Ankle_Y': 'j15_y', 'L_Ankle_Z': 'j15_z',
            'R_Ankle_X': 'j16_x', 'R_Ankle_Y': 'j16_y', 'R_Ankle_Z': 'j16_z',
            'Hip_Center_X': 'Hip_Center_X', 
            'Hip_Center_Y': 'Hip_Center_Y', 
            'Hip_Center_Z': 'Hip_Center_Z'
        }

    def filter_data(self):
        # 4th order butterworth filter, 6Hz cutoff (standard for gait)
        b, a = butter(4, 6 / (0.5 * self.fps), btype='low')
        
        for col in self.map.values:
            if col in self.df.columns:
                self.df[col] = filtfilt(b, a, self.df[col])
            else:
                print(WARNING + f"column {col} not found in CSV.")

    def detect_events(self, side='L'):
        '''Detects Foot Strike (Heel) and Foot Off (Toe)'''
        prefix = 'L' if side == 'L' else 'R'
        z_col = self.map[f'{prefix}_Ankle_{self.height_axis.upper()}']

        # heeel strike
        z_signal = self.df[z_col].values
        strike_signal = -z_signal if self.up_dir == 1 else z_signal
        strikes, _ = find_peaks(strike_signal, distance=self.fps*0.5)

        # toe off
        # when foot leaves the ground, z velocity spikes upward
        vel_z = np.gradient(z_signal)


        # look for peaks in positive velocity
        # the max velocity peak between two strikes is the Toe Off
        off_signal = vel_z if self.up_dir == 1 else -vel_z
        offs = []
        velocity_peaks, _ = find_peaks(off_signal, height=0.01, distance=self.fps*0.5)
        return np.sort(strikes), np.sort(velocity_peaks)
    
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
        
        # reorder columns to match VICON
        events_df = events_df[['Subject', 'Context', 'Name', 'Time (s)', 'Description']]

        # generate parameter table
        params_list = []
        
        # Helper to calculate metrics for a specific side
        def calc_side_metrics(strikes, offs, opp_strikes, side):
            if len(strikes) < 2: return # Need at least 2 steps for a stride
            
            # use the middle steps to avoid start/stop issues
            valid_steps = 0
            
            avg_metrics = {k: [] for k in ['StepLen', 'StrideLen', 'StepTime', 'StrideTime', 'Speed', 'Cadence']}
            
            for i in range(len(strikes) - 1):
                start = strikes[i]
                end = strikes[i+1]
                
                # stride time
                stride_time = (end - start) / self.fps
                avg_metrics['StrideTime'].append(stride_time)
                
                # Step Length (Distance at Start frame)
                # Find closest opposite strike occurring BEFORE current strike
                opp_prev = opp_strikes[opp_strikes < start]
                if len(opp_prev) > 0:
                    prev_opp = opp_prev[-1]
                    # Step Time
                    step_time = (start - prev_opp) / self.fps
                    avg_metrics['StepTime'].append(step_time)
                    
                    # geometry
                    l_x, l_y = self.df.iloc[start][self.map[f'L_Ankle_X']], self.df.iloc[start][self.map[f'L_Ankle_Y']]
                    r_x, r_y = self.df.iloc[start][self.map[f'R_Ankle_X']], self.df.iloc[start][self.map[f'R_Ankle_Y']]
                    dist_cm = np.sqrt((l_x-r_x)**2 + (l_y-r_y)**2) * 100 # Convert m to cm
                    
                    avg_metrics['StepLen'].append(dist_cm)
                    
                    # stride length (Distance covered by Hip between strikes * 2 approx, or Foot displacement)
                    # simplified: stride length ~ 2 * step length (for basic calc)
                    # Or correct way: Displacement of the SAME foot
                    foot_x_start = self.df.iloc[start][self.map[f'{side[0]}_Ankle_X']]
                    foot_x_end   = self.df.iloc[end][self.map[f'{side[0]}_Ankle_X']]
                    # Note: This requires absolute world coordinates. If treadmill, this is 0.
                    # assuming overground:
                    stride_len_cm = abs(foot_x_end - foot_x_start) * 100
                    # fallback if stationary treadmill walking:
                    if stride_len_cm < 10: stride_len_cm = dist_cm * 2 
                    
                    avg_metrics['StrideLen'].append(stride_len_cm)

            # averages
            count = len(avg_metrics['StepLen'])
            if count == 0: return None
            
            m = {k: np.mean(v) for k, v in avg_metrics.items()}
            
            # walking speed (m/s) = stride length (m) / stride time (s)
            speed = (m['StrideLen'] / 100) / m['StrideTime']
            
            # cadence (steps/min) = 60 / step time
            cadence = 60 / m['StepTime']
            
            return {
                'Cadence': cadence,
                'Walking Speed': speed,
                'Stride Time': m['StrideTime'],
                'Step Time': m['StepTime'],
                'Stride Length': m['StrideLen'],
                'Step Length': m['StepLen'],
                # placeholders for complex percentages (requires precise Toe Off logic)
                'Opposite Foot Off': 0, 
                'Opposite Foot Contact': 50.0,
                'Foot Off': 60.0,
                'Single Support': 40.0,
                'Double Support': 20.0,
                'Step Width': 15.0, # placeholder or calc average width
                'Limp Index': np.nan
            }

        # calculate Left
        l_res = calc_side_metrics(l_strikes, l_offs, r_strikes, 'Left')
        # calculate Right
        r_res = calc_side_metrics(r_strikes, r_offs, l_strikes, 'Right')
        
        # build rows
        def add_rows(res, context):
            if not res: return
            params_list.append([SUBJECT_NAME, context, 'Cadence', res['Cadence'], 'steps/min'])
            params_list.append([SUBJECT_NAME, context, 'Walking Speed', res['Walking Speed'], 'm/s'])
            params_list.append([SUBJECT_NAME, context, 'Stride Time', res['Stride Time'], 's'])
            params_list.append([SUBJECT_NAME, context, 'Step Time', res['Step Time'], 's'])
            params_list.append([SUBJECT_NAME, context, 'Stride Length', res['Stride Length'], 'cm'])
            params_list.append([SUBJECT_NAME, context, 'Step Length', res['Step Length'], 'cm'])
            # add placeholders
            params_list.append([SUBJECT_NAME, context, 'Step Width', res['Step Width'], 'cm'])

        add_rows(l_res, 'Left')
        add_rows(r_res, 'Right')
        
        params_df = pd.DataFrame(params_list, columns=['Subject', 'Context', 'Name', 'Value', 'Units'])
        
        return params_df, events_df
    

def main():
    # ensure you set the FPS correctly here
    gait_analyzer = GaitAnalyzer(OUTPUT_CSV, fps=FPS)
    
    params_df, events_df = gait_analyzer.generate_vicon_tables()

    # print(summary.keys())
    # return
    
    print("\n# Gait Cycle Parameters")
    print(params_df.to_markdown())
    
    print("\n\n# Events")
    print(events_df.to_markdown())

if __name__ == '__main__':
    main()


