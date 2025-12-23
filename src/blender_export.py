import bpy, os, csv, math

CSV_DIR = "/home/aicenter/Dev/dementia-gait-pattern/output"

CSV_PATH = os.path.join(CSV_DIR, "multiview_skeleton_3d.csv")
SCALE = 1.0  # adjust if skeleton is too big/small
FPS = 25    # match your video FPS

# COCO Skeleton Definition
BONES = [
    (0,1), (0,2), (1,3), (2,4),      # Face
    (5,6), (5,7), (7,9), (6,8),      # Arms
    (11,12), (11,5), (12,6),         # Torso connection
    (11,13), (13,15), (12,14), (14,16) # Legs
]

SKELETON_LINKS = [
    (0, 1), (0, 2), # Nose to Eyes
    (1, 3), (2, 4), # Eyes to Ears
    (5, 6),         # Shoulders
    (5, 7), (7, 9), # Left Arm
    (6, 8), (8, 10),# Right Arm
    (5, 11), (6, 12), # Shoulder to Hip
    (11, 12),       # Hips
    (11, 13), (13, 15), # Left Leg
    (12, 14), (14, 16)  # Right Leg
]

NAMING_PATTERN = "Joint_{}"

def create_bones_from_objects():
    # create a new armature and object
    armature_data = bpy.data.armatures.new("PoseSkeleton")
    armature_obj = bpy.data.objects.new("PoseRig", armature_data)
    bpy.context.collection.objects.link(armature_obj)
    
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    bones = armature_data.edit_bones
    
    for parent_idx, child_idx in SKELETON_LINKS:
        parent_name = NAMING_PATTERN.format(parent_idx)
        child_name = NAMING_PATTERN.format(child_idx)
        
        parent_obj = bpy.data.objects.get(parent_name)
        child_obj = bpy.data.objects.get(child_name)
        
        if parent_obj and child_obj:
            # create a new bone
            bone = bones.new(f"Bone_{parent_idx}_{child_idx}")
            
            # set head (start) and tail (end)
            bone.head = parent_obj.location
            bone.tail = child_obj.location
            
            # connect bones if they share a joint
            # this logic can be complex for arbitrary graphs, 
            # but usually necessary for IK.
            
        else:
            print(f"Warning: Could not find {parent_name} or {child_name}")

    bpy.ops.object.mode_set(mode='OBJECT')
    print("Skeleton generated!")

def create_skeleton_viz():
    # cleanup existing
    if "SkeletonCollection" in bpy.data.collections:
        coll = bpy.data.collections["SkeletonCollection"]
        for obj in coll.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(coll)
        
    # setup Collections
    collection = bpy.data.collections.new("SkeletonCollection")
    bpy.context.scene.collection.children.link(collection)
    
    # create sphere objects for joints
    spheres = []
    for i in range(17):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05 * SCALE)
        obj = bpy.context.active_object
        obj.name = f"Joint_{i}"
        collection.objects.link(obj)
        bpy.context.collection.objects.unlink(obj) # Unlink from main
        spheres.append(obj)
        
    # read CSV and Animate
    with open(CSV_PATH, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) # skip header
        
        for row in reader:
            frame_idx = int(row[0])
            bpy.context.scene.frame_set(frame_idx)
            
            # read joints (starting col 1)
            for i in range(17):
                idx = 1 + (i*3)
                try:
                    # Parse X, Y, Z
                    # Note: OpenCV Y is Down, Blender Z is Up
                    # We usually Map: OpenCV X -> Blender X
                    #                 OpenCV Z -> Blender -Y (Depth)
                    #                 OpenCV Y -> Blender -Z (Height, inverted)
                    
                    x = float(row[idx])
                    y = float(row[idx+1])
                    z = float(row[idx+2])
                    
                    obj = spheres[i]
                    
                    # Coordinate Mapping (Adjust as needed)
                    obj.location = (x * SCALE, z * SCALE, -y * SCALE)
                    obj.keyframe_insert(data_path="location", frame=frame_idx)
                    
                except ValueError as e:
                    # empty data (occlusion)
                    print(str(e))

    print("animation imported!")
    
create_bones_from_objects()
create_skeleton_viz()
