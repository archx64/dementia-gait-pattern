skeleton_links = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # face
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),  # torso (shoulders, hips and sides)
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
]

header = ['frame_idx', 'total_distance_m']

keypoint_names = [
    'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 
    'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
    'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle'
]