import bpy, os, csv, math

# configs
CSV_DIR = "/home/aicenter/Dev/dementia-gait-pattern/output"
CSV_PATH = os.path.join(CSV_DIR, "multiview_skeleton_3d.csv")
SCALE = 1
FPS = 25    
NUM_JOINTS = 133  # Changed from 17 to 23 to include feet

# definition of bone connections based on MMPose WholeBody
# 15=L_Ankle, 16=R_Ankle
# 17=L_BigToe, 18=L_SmallToe, 19=L_Heel
# 20=R_BigToe, 21=R_SmallToe, 22=R_Heel

SKELETON_LINKS = [
    # --- Body (Standard COCO) ---
    # Head
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    # Torso
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    # Arms
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    # Legs
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    # --- Feet ---
    # L_Ankle -> L_Heel -> L_BigToe -> L_SmallToe
    (15, 19),
    (19, 17),
    (17, 18),
    # R_Ankle -> R_Heel -> R_BigToe -> R_SmallToe
    (16, 22),
    (22, 20),
    (20, 21),
    # --- Face (Contours) ---
    # Jawline
    (23, 24),
    (24, 25),
    (25, 26),
    (26, 27),
    (27, 28),
    (28, 29),
    (29, 30),
    (30, 31),
    (31, 32),
    (32, 33),
    (33, 34),
    (34, 35),
    (35, 36),
    (36, 37),
    (37, 38),
    (38, 39),
    # Eyebrows
    (40, 41),
    (41, 42),
    (42, 43),
    (43, 44),
    (45, 46),
    (46, 47),
    (47, 48),
    (48, 49),
    # Nose
    (50, 51),
    (51, 52),
    (52, 53),
    (54, 55),
    (55, 56),
    (56, 57),
    (57, 58),
    # Eyes
    (59, 60),
    (60, 61),
    (61, 62),
    (62, 63),
    (63, 64),
    (64, 59),
    (65, 66),
    (66, 67),
    (67, 68),
    (68, 69),
    (69, 70),
    (70, 65),
    # Lips
    (71, 72),
    (72, 73),
    (73, 74),
    (74, 75),
    (75, 76),
    (76, 77),
    (77, 78),
    (78, 79),
    (79, 80),
    (80, 81),
    (81, 82),
    (82, 71),
    (83, 84),
    (84, 85),
    (85, 86),
    (86, 87),
    (87, 88),
    (88, 89),
    (89, 90),
    (90, 83),
    # --- Left Hand ---
    # Thumb
    (91, 92),
    (92, 93),
    (93, 94),
    (94, 95),
    # Index
    (91, 96),
    (96, 97),
    (97, 98),
    (98, 99),
    # Middle
    (91, 100),
    (100, 101),
    (101, 102),
    (102, 103),
    # Ring
    (91, 104),
    (104, 105),
    (105, 106),
    (106, 107),
    # Pinky
    (91, 108),
    (108, 109),
    (109, 110),
    (110, 111),
    # --- Right Hand ---
    # Thumb
    (112, 113),
    (113, 114),
    (114, 115),
    (115, 116),
    # Index
    (112, 117),
    (117, 118),
    (118, 119),
    (119, 120),
    # Middle
    (112, 121),
    (121, 122),
    (122, 123),
    (123, 124),
    # Ring
    (112, 125),
    (125, 126),
    (126, 127),
    (127, 128),
    # Pinky
    (112, 129),
    (129, 130),
    (130, 131),
    (131, 132),
]

NAMING_PATTERN = "Joint_{}"

def setup_render_settings(output_path, num_frames):
    scene = bpy.context.scene
    
    # Set Resolution
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    
    # Set FPS
    scene.render.fps = 14
    
    # Set Length
    scene.frame_start = 0
    scene.frame_end = num_frames
    
    # Set Output Path
    scene.render.filepath = output_path
    
    # Set Format to MP4 (H.264)
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM' # Quality
    
    print(f"Render settings configured. Output: {output_path}")

# add this to your main execution block:
# setup_render_settings("/home/aicenter/Dev/output_video.mp4", 500)
# bpy.ops.render.render(animation=True) # Uncomment to auto-render

def create_skeleton_viz():
    # 1. Remove old collection if it exists (Clean Up)
    if "SkeletonCollection" in bpy.data.collections:
        coll = bpy.data.collections["SkeletonCollection"]
        for obj in coll.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(coll)
        
    collection = bpy.data.collections.new("SkeletonCollection")
    bpy.context.scene.collection.children.link(collection)
    
    # 2. Create Spheres for all 23 joints (Set Up)
    spheres = []
    for i in range(NUM_JOINTS):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.01 * SCALE)
        obj = bpy.context.active_object
        obj.name = NAMING_PATTERN.format(i)
        
        # Unlink from default collection and link to ours
        for coll in obj.users_collection:
            coll.objects.unlink(obj)
        collection.objects.link(obj)
        
        spheres.append(obj)
        
    print(f"Reading CSV from {CSV_PATH}...")
    
    # 3. Read CSV and Keyframe (Animation)
    with open(CSV_PATH, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) 
        
        for row in reader:
            frame_idx = int(row[0])
            bpy.context.scene.frame_set(frame_idx)
            
            for i in range(NUM_JOINTS):
                # calculate CSV column index: Frame(1) + (JointIndex * 3)
                idx = 1 + (i*3)
                
                try:
                    # check if data exists (handle potential empty strings)
                    if not row[idx] or not row[idx+1] or not row[idx+2]:
                        continue

                    # parse Coordinates
                    x = float(row[idx])
                    y = float(row[idx+1])
                    z = float(row[idx+2])
                    
                    obj = spheres[i]
                    
                    # coordinate mapping: openCV (Y-Down) -> Blender (Z-Up)
                    # mapping: x=x, y=-z, z=y (rotates character to stand up)
                    obj.location = (x * SCALE, z * SCALE, -y * SCALE)
                    
                    obj.keyframe_insert(data_path="location", frame=frame_idx)
                    
                except (ValueError, IndexError):
                    pass

    print(f"Animated {NUM_JOINTS} spheres.")
    return spheres

# === execution ===
if __name__ == "__main__":
    create_skeleton_viz()