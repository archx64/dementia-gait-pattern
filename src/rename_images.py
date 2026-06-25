import os
import re
from src.copy_frames import DESTINATION_DIRECTORY

def rename_image_sequence(directory, prefix="frame_"):
    # Find all files in the target directory
    files = os.listdir(directory)
    
    # Filter to only include image files
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in files if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print("No image files found in the directory.")
        return

    # Helper function to extract numbers from the old filenames so we can sort them correctly
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')

    # Sort the files numerically (e.g., 30650.png comes before 30662.png)
    image_files.sort(key=extract_number)

    print(f"Found {len(image_files)} images. Starting the renaming process...\n")

    # Loop through the sorted files and rename them
    count = 1
    for old_filename in image_files:
        # Extract the original extension (e.g., '.png')
        ext = os.path.splitext(old_filename)[1]
        
        # Format the new name with 3 digits of padding (001, 002, etc.)
        # Change {:03d} to {:04d} if you have over 999 frames and want 4 digits.
        new_filename = f"{prefix}{count:03d}{ext}"
        
        # Create the full file paths
        old_path = os.path.join(directory, old_filename)
        new_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {old_filename}  ->  {new_filename}")
        
        count += 1

    print("\nDone! All files have been renamed.")


# Point this to the folder where you copied your selected frames
# TARGET_DIRECTORY = r"C:\path\to\your\destination\folder"

def main():
    rename_image_sequence(DESTINATION_DIRECTORY)

if __name__ == "__main__":
    main()