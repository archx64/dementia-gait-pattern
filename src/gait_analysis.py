import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from utils import FPS, GAITANALYSIS_CSV, OUTPUT_CSV, WARNING

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
        self.filter_data()
        
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
        
        for col in self.raw_cols:
            if col in self.df.columns:
                self.df[col] = filtfilt(b, a, self.df[col])
            else:
                print(WARNING + f"column {col} not found in CSV.")

    def detect_heel_strikes(self, side='L'):
        """
        Detects Heel Strikes based on the local minima of the Ankle's vertical position.
        """
        prefix = 'L' if side == 'L' else 'R'
        z_col = self.map[f'{prefix}_Ankle_{self.height_axis.upper()}']
        
        # get vertical trajectory
        z_traj = self.df[z_col].values
        
        # invert signal if 'up' is positive, so minima become peaks for detection
        if self.up_dir == 1: 
            signal = -z_traj 
        else:
            signal = z_traj
            
        # find peaks (foot hitting the floor)
        # distance=fps*0.5 ensures we don't detect two strikes within 0.5 seconds
        peaks, _ = find_peaks(signal, distance=self.fps*0.5)
        
        return peaks

    def calculate_parameters(self):
        left_strikes = self.detect_heel_strikes('L')
        right_strikes = self.detect_heel_strikes('R')
        
        print(f"Found {len(left_strikes)} Left Strikes and {len(right_strikes)} Right Strikes.")
        
        metrics = []

        # left step calculations
        for frame in left_strikes:
            # step Length = Distance from previous right foot strike position?
            # simplified medical def: Distance between feet at moment of contact.
            
            l_x = self.df.iloc[frame][self.map['L_Ankle_X']]
            l_y = self.df.iloc[frame][self.map['L_Ankle_Y']]
            r_x = self.df.iloc[frame][self.map['R_Ankle_X']]
            r_y = self.df.iloc[frame][self.map['R_Ankle_Y']]
            
            # euclidean distance on ground plane
            dist = np.sqrt((l_x - r_x)**2 + (l_y - r_y)**2)
            
            metrics.append({
                'Side': 'Left',
                'Parameter': 'Step Length',
                'Value (m)': round(dist, 3),
                'Value (cm)': round(dist * 100, 2), # converted for VICON comparison
                'Frame': frame,
                'Time (s)': round(frame / self.fps, 3)
            })

        # right step calculation
        for frame in right_strikes:
            l_x = self.df.iloc[frame][self.map['L_Ankle_X']]
            l_y = self.df.iloc[frame][self.map['L_Ankle_Y']]
            r_x = self.df.iloc[frame][self.map['R_Ankle_X']]
            r_y = self.df.iloc[frame][self.map['R_Ankle_Y']]
            
            dist = np.sqrt((l_x - r_x)**2 + (l_y - r_y)**2)
            
            metrics.append({
                'Side': 'Right',
                'Parameter': 'Step Length',
                'Value (m)': round(dist, 3),
                'Value (cm)': round(dist * 100, 2),
                'Frame': frame,
                'Time (s)': round(frame / self.fps, 3)
            })

        # cadence = steps / minute
        total_steps = len(left_strikes) + len(right_strikes)
        duration_min = (len(self.df) / self.fps) / 60
        cadence = total_steps / duration_min if duration_min > 0 else 0
        
        # calculate gait speed (distance traveled by hip center / time)
        # using simple total displacement
        hip_start_x = self.df.iloc[0]['Hip_Center_X']
        hip_start_y = self.df.iloc[0]['Hip_Center_Y']
        hip_end_x = self.df.iloc[-1]['Hip_Center_X']
        hip_end_y = self.df.iloc[-1]['Hip_Center_Y']
        
        total_dist = np.sqrt((hip_end_x - hip_start_x)**2 + (hip_end_y - hip_start_y)**2)
        walking_speed = total_dist / (len(self.df) / self.fps)

        metrics_df = pd.DataFrame(metrics).sort_values(by='Frame')

        summary = {
            'cadence_steps_p_min': round(cadence, 2), # cadence (steps/min)
            'walking_speed_m_p_s': round(walking_speed, 2), # Walking Speed (m/s)
            'avg_left_step_length_cm': round(metrics_df[metrics_df['Side']=='Left']['Value (cm)'].mean(), 2), # Avg Left Step Length (cm)
            'avg_right_step_length_cm': round(metrics_df[metrics_df['Side']=='Right']['Value (cm)'].mean(), 2) # Avg Right Step Length (cm)
        }
        
        return metrics_df, summary
    

def main():
    # ensure you set the FPS correctly here
    gait_analyzer = GaitAnalyzer(OUTPUT_CSV, fps=FPS)
    
    events_df, summary = gait_analyzer.calculate_parameters()

    print(summary.keys())
    # return
    
    print('\nSummary')
    print(f"cadence:      {summary['cadence_steps_p_min']}")
    print(f"speed:        {summary['walking_speed_m_p_s']}")
    print(f"L Step Len:   {summary['avg_left_step_length_cm']} cm")
    print(f"R Step Len:   {summary['avg_right_step_length_cm']} cm")
    # print(f"cadence:      {summary['Cadence (steps/min)']} (Compare to VICON ~114-118)")
    # print(f"speed:        {summary['Walking Speed (m/s)']} m/s (Compare to VICON ~1.13)")
    # print(f"L Step Len:   {summary['Avg Left Step Length (cm)']} cm (Compare to VICON ~58.08)")
    # print(f"R Step Len:   {summary['Avg Right Step Length (cm)']} cm (Compare to VICON ~55.14)")
    
    print('\nDETAILED EVENTS check time column for synchronization')
    if not events_df.empty:
        print(events_df[['Side', 'Time (s)', 'Value (cm)', 'Frame']].to_markdown(index=False))
    else:
        print("no steps detected. Check if your Z-axis is correct or if data is too noisy.")

if __name__ == '__main__':
    main()


