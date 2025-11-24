import cv2
import numpy as np
import glob

# define the dimensions of checkerboard (rows - 1, columns - 1)
CHECKERBOARD = (7, 10) 
# real world square size (e.g., 25mm = 0.025 meters)
SQUARE_SIZE = 0.025 

def get_img_points(images_path, board_dims):
    """
    finds checkerboard corners in a list of images.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_dims[0] * board_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_dims[0], 0:board_dims[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    valid_indices = [] # Keep track of which images worked

    images = sorted(glob.glob(images_path))

    if not images:
        print(f"no images found at path: {images_path}")
        print("check file extensions")
        return None, None, None, None
    
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, board_dims, None)

        if ret == True:
            objpoints.append(objp)
            # refine corner locations
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            valid_indices.append(i)
            
            # draw and display the corners
            cv2.drawChessboardCorners(img, board_dims, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
    return objpoints, imgpoints, gray.shape[::-1], valid_indices

print("1. Processing Images...")
# load paths (Ensure these match your folder structure)
path1 = 'cam1/*.png'
path2 = 'cam2/*.png'
# path3 = 'cam3/*.png'

# ensure only frames where the pattern was found in ALL cameras are used
# first, get points for all cameras individually
objp1, imgp1, shape1, idx1 = get_img_points(path1, CHECKERBOARD)
objp2, imgp2, shape2, idx2 = get_img_points(path2, CHECKERBOARD)
# objp3, imgp3, shape3, idx3 = get_img_points(path3, CHECKERBOARD)

# Find intersection of valid indices (frames where board is visible in ALL 3)
# common_indices = set(idx1) & set(idx2) & set(idx3)
common_indices = set(idx1) & set(idx2)
common_indices = sorted(list(common_indices))

print(f"Found {len(common_indices)} common valid frames for stereo calibration.")

# filter the lists to only include the common frames
# we re-construct the lists based on the intersection logic
# we should optimize this to not re-read, but this is safer for tutorial purposes

final_objp = []
final_imgp1 = []
final_imgp2 = []
final_imgp3 = []

# re-populate based on common indices to ensure perfect alignment
# this relies on the arrays being populated in sorted order of filenames originally
map_1 = {idx: ptr for idx, ptr in zip(idx1, imgp1)}
map_2 = {idx: ptr for idx, ptr in zip(idx2, imgp2)}
# map_3 = {idx: ptr for idx, ptr in zip(idx3, imgp3)}

for idx in common_indices:
    final_objp.append(objp1[0]) # Object points are the same for all
    final_imgp1.append(map_1[idx])
    final_imgp2.append(map_2[idx])
    # final_imgp3.append(map_3[idx])

print("2. Calibrating Individual Cameras (Intrinsics)...")
ret1, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(final_objp, final_imgp1, shape1, None, None)
ret2, K2, D2, rvecs2, tvecs2 = cv2.calibrateCamera(final_objp, final_imgp2, shape2, None, None)
# ret3, K3, D3, rvecs3, tvecs3 = cv2.calibrateCamera(final_objp, final_imgp3, shape3, None, None)

# print(f"Cam1 RMSE: {ret1}\nCam2 RMSE: {ret2}\nCam3 RMSE: {ret3}")
print(f"Cam1 RMSE: {ret1}\nCam2 RMSE: {ret2}")

print("3. Performing Stereo Calibration (Extrinsics)...")
# stereo Calibrate Cam 1 (Master) to Cam 2
flags = cv2.CALIB_FIX_INTRINSIC # We trust our individual calibration
ret_12, _, _, _, _, R12, T12, E12, F12 = cv2.stereoCalibrate(
    final_objp, final_imgp1, final_imgp2,
    K1, D1, K2, D2, shape1, flags=flags
)

# stereo Calibrate Cam 1 (Master) to Cam 3
# ret_13, _, _, _, _, R13, T13, E13, F13 = cv2.stereoCalibrate(
#     final_objp, final_imgp1, final_imgp3,
#     K1, D1, K3, D3, shape1, flags=flags
# )

print("\n--- Calibration Results ---")
print("Camera 1 is World Origin (0,0,0)")
print(f"Cam 2 Position (relative to Cam 1):\n T: {T12.T}")
# print(f"Cam 3 Position (relative to Cam 1):\n T: {T13.T}")

# save the data
np.savez('multiview_calibration_test.npz', 
         K1=K1, D1=D1, 
         K2=K2, D2=D2, R12=R12, T12=T12,
        #  K3=K3, D3=D3, R13=R13, T13=T13)
)
print("Calibration saved to 'multiview_calibration.npz'")
