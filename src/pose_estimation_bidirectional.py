import cv2, os, csv, torch, functools, itertools
import numpy as np
from matplotlib import pyplot as plt
from mmpose.apis import MMPoseInferencer
from scipy.spatial.transform import Rotation as R_scipy

# === CONFIGURATION ===
from utils import (
    INFO, WARNING, ERROR, DEBUG, 
    VIDEO_PATHS, CONFIG_PATH, WEIGHT_PATH, 
    OUTPUT_CSV, CALIBRATION_FILE, TILT_CORRECTION_ANGLE, 
    keypoint_names
)

# HRNet-w48 Dark (WholeBody)
MODEL_CONFIG = "td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py"
MODEL_CHECKPOINT = "hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
CONFIDENCE_THR = 0.4 
ROBUST_TRIANGULATION = True 

# ==========================================
#      TRIANGULATION 
# ==========================================
class MultiviewTriangulator:
    def __init__(self, npz_path, cam_names):
        self.cameras = {}
        self.data = np.load(npz_path)
        all_keys = list(self.data.keys())
        sorted_prefixes = sorted(list(set([k.split("_")[0] for k in all_keys])))
        for i, prefix in enumerate(sorted_prefixes):
            if i >= len(cam_names): break
            K = self.data[f"{prefix}_K"]
            R = self.data[f"{prefix}_R"]
            T = self.data[f"{prefix}_T"]
            RT = np.hstack((R, T))
            P = K @ RT
            self.cameras[i] = {"P": P}

    def triangulate_one_point(self, views):
        if len(views) < 2: return np.array([np.nan, np.nan, np.nan])
        if len(views) == 2 or not ROBUST_TRIANGULATION: return self._run_svd(views)
        candidates = []
        pairs = list(itertools.combinations(views, 2))
        for pair in pairs: candidates.append(self._run_svd(pair))
        valid_cluster = []
        for i, p1 in enumerate(candidates):
            neighbors = 0
            for j, p2 in enumerate(candidates):
                if i == j: continue
                if np.linalg.norm(p1 - p2) < 0.15: neighbors += 1
            if neighbors > 0: valid_cluster.append(p1)
        if not valid_cluster: return np.median(candidates, axis=0)
        return np.mean(valid_cluster, axis=0)

    def _run_svd(self, views):
        A = []
        for cam_idx, (u, v) in views:
            P = self.cameras[cam_idx]["P"]
            row1, row2 = u * P[2] - P[0], v * P[2] - P[1]
            A.append(row1); A.append(row2)
        u, s, vh = np.linalg.svd(np.array(A))
        X = vh[-1]
        return (X / X[3])[:3]

class PersonSelector:
    def __init__(self):
        self.selected_point = None
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: self.selected_point = (x, y)
    
    def select_frame_and_person(self, caps, inferencer):
        """Allows scrubbing through video to find a good initialization frame"""
        total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        
        cv2.namedWindow("Selection (A/D: Scrub, Space: Select)")
        cv2.setMouseCallback("Selection (A/D: Scrub, Space: Select)", self.mouse_callback)
        
        selected_results = None
        selected_idx = 0
        
        while True:
            # Update all cameras to current frame
            frames = []
            for cap in caps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, f = cap.read()
                if not ret: break
                frames.append(f)
            
            if not frames: break

            # Inference on Main View (Cam 0) ONLY for visualization
            res_gen = inferencer(frames[0], return_vis=False)
            res = next(res_gen)["predictions"][0]
            
            vis_img = frames[0].copy()
            bboxes = []
            for i, p in enumerate(res):
                bbox = p["bbox"][0]
                x1, y1, x2, y2 = map(int, bbox[:4])
                bboxes.append((x1, y1, x2, y2))
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_img, f"P{i}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.putText(vis_img, f"Frame: {current_frame}/{total_frames}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            if self.selected_point:
                cv2.circle(vis_img, self.selected_point, 5, (0,0,255), -1)

            cv2.imshow("Selection (A/D: Scrub, Space: Select)", cv2.resize(vis_img, (1280, 720)))
            
            key = cv2.waitKey(0)
            if key == ord('d'): current_frame = min(current_frame + 5, total_frames-1)
            elif key == ord('a'): current_frame = max(current_frame - 5, 0)
            elif key == 32: # Spacebar
                if self.selected_point:
                    # Identify clicked person
                    cx, cy = self.selected_point
                    found = False
                    for i, (x1, y1, x2, y2) in enumerate(bboxes):
                        if x1<=cx<=x2 and y1<=cy<=y2:
                            selected_idx = i
                            selected_results = res
                            found = True
                            break
                    if found: break
                    else: print("Clicked background. Click a box.")
                else: print("Click a person first.")
        
        cv2.destroyAllWindows()
        return current_frame, selected_idx, selected_results

    def match_person(self, ref_kpts, candidates, triangulator, ref_cam, tgt_cam):
        if not candidates: return 0
        best_idx, max_pts = 0, -1
        # Use Torso + Knees for matching (more robust than just hips)
        test_joints = [5, 6, 11, 12, 13, 14] 
        
        for idx, cand in enumerate(candidates):
            valid = 0
            for j in test_joints:
                u1, v1 = ref_kpts[j]
                u2, v2 = cand["keypoints"][j]
                pt = triangulator.triangulate_one_point([(ref_cam, (u1,v1)), (tgt_cam, (u2,v2))])
                if not np.isnan(pt[0]): valid += 1
            if valid > max_pts:
                max_pts = valid
                best_idx = idx
        return best_idx

# ==========================================
#      MAIN PIPELINE
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(INFO + f"Initializing Bidirectional Pipeline on {device}...")
    torch.load = functools.partial(torch.load, weights_only=False)
    inferencer = MMPoseInferencer(
        pose2d=os.path.join(CONFIG_PATH, MODEL_CONFIG),
        pose2d_weights=os.path.join(WEIGHT_PATH, MODEL_CHECKPOINT),
        device=device,
    )

    caps = [cv2.VideoCapture(v) for v in VIDEO_PATHS]
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    triangulator = MultiviewTriangulator(CALIBRATION_FILE, VIDEO_PATHS)

    # 1. SELECT INITIALIZATION FRAME
    selector = PersonSelector()
    print(INFO + ">>> Find a frame where the person is STANDING clearly.")
    start_frame, target_idx, res_0 = selector.select_frame_and_person(caps, inferencer)
    
    print(INFO + f"Initialized at Frame {start_frame}. Matching across views...")

    # 2. MATCH ACROSS CAMERAS (At Start Frame)
    # We must run inference on ALL cameras at this specific frame
    frames_init = []
    res_init = []
    for i, cap in enumerate(caps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        _, f = cap.read()
        frames_init.append(f)
        r = next(inferencer(f, return_vis=False))["predictions"][0]
        res_init.append(r)

    # Reference Keypoints (Cam 0)
    ref_kpts = res_init[0][target_idx]["keypoints"][:23]
    
    indices = {0: target_idx}
    init_centroids = {} # {cam_idx: (cx, cy)}
    
    # Match others
    for i in range(1, len(caps)):
        idx = selector.match_person(ref_kpts, res_init[i], triangulator, 0, i)
        indices[i] = idx
        bbox = res_init[i][idx]["bbox"][0]
        init_centroids[i] = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
        
    bbox0 = res_init[0][target_idx]["bbox"][0]
    init_centroids[0] = ((bbox0[0]+bbox0[2])/2, (bbox0[1]+bbox0[3])/2)

    # 3. PROCESSING FUNCTION
    # We define a function to process a range of frames given starting centroids
    
    final_data = {} # frame_idx -> [pts_3d]

    def process_sequence(start, end, step, start_centroids):
        print(INFO + f"Processing frames {start} to {end} (step {step})...")
        curr_centroids = start_centroids.copy()
        
        # Tilt Correction Matrix
        theta = np.radians(TILT_CORRECTION_ANGLE)
        c, s = np.cos(theta), np.sin(theta)
        R_fix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

        # Iterate
        # If step is -1, we go start-1 down to end
        # If step is 1, we go start+1 up to end
        
        rng = range(start + step, end, step)
        if len(rng) == 0: return

        for f_idx in rng:
            frames = []
            for cap in caps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                _, f = cap.read()
                frames.append(f)
            
            if any(f is None for f in frames): break

            # Inference
            all_preds = []
            for f in frames:
                r = next(inferencer(f, return_vis=False))
                all_preds.append(r["predictions"][0])

            # Track & Triangulate
            pts_3d_frame = np.zeros((23, 3)) 
            current_view_indices = {}

            for i, preds in enumerate(all_preds):
                if not preds: continue
                last_cx, last_cy = curr_centroids[i]
                best_idx, min_dist = -1, float("inf")
                
                for p_idx, p in enumerate(preds):
                    bbox = p["bbox"][0]
                    cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                    dist = np.sqrt((cx-last_cx)**2 + (cy-last_cy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = p_idx
                
                # Loose threshold for sitting/standing transitions (300px)
                if best_idx != -1 and min_dist < 300: 
                    current_view_indices[i] = best_idx
                    bbox = preds[best_idx]["bbox"][0]
                    curr_centroids[i] = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)

            for j in range(23):
                views = []
                for cam_idx in range(len(caps)):
                    if cam_idx not in current_view_indices: continue
                    p_idx = current_view_indices[cam_idx]
                    pred = all_preds[cam_idx][p_idx]
                    score = pred["keypoint_scores"][j]
                    if score > CONFIDENCE_THR:
                        u, v = pred["keypoints"][j]
                        views.append((cam_idx, (u, v)))
                
                pts_3d_frame[j] = triangulator.triangulate_one_point(views)

            # Save
            pts_3d_frame = pts_3d_frame @ R_fix.T
            final_data[f_idx] = pts_3d_frame
            print(f"Processed Frame {f_idx}", end='\r')

    # 4. RUN BIDIRECTIONAL TRACKING
    
    # A. Backward: Start -> 0
    process_sequence(start_frame, -1, -1, init_centroids)
    
    # B. Forward: Start -> End
    process_sequence(start_frame, total_frames, 1, init_centroids)
    
    # C. Don't forget the Start Frame itself!
    # (We calculated init centroids but didn't save the 3d points for start frame)
    # Quick re-calc for start frame:
    pts_start = np.zeros((23,3))
    theta = np.radians(TILT_CORRECTION_ANGLE)
    c, s = np.cos(theta), np.sin(theta)
    R_fix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    
    for j in range(23):
        views = []
        for i in range(len(caps)):
            idx = indices[i]
            pred = res_init[i][idx]
            if pred["keypoint_scores"][j] > CONFIDENCE_THR:
                views.append((i, pred["keypoints"][j]))
        pts_start[j] = triangulator.triangulate_one_point(views)
    final_data[start_frame] = pts_start @ R_fix.T

    # 5. WRITE TO CSV
    print(INFO + "\nWriting to CSV...")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["frame_idx"]
        for i in range(23): header.extend([f"j{i}_x", f"j{i}_y", f"j{i}_z"])
        writer.writerow(header)
        
        # Sort keys to write in order
        for f_idx in sorted(final_data.keys()):
            row = [f_idx]
            pts = final_data[f_idx]
            for p in pts:
                if np.isnan(p[0]): row.extend(["","",""])
                else: row.extend([f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])
            writer.writerow(row)

    print(INFO + f"Done. Saved to {OUTPUT_CSV}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()