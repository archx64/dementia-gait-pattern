import cv2, os



def capture_and_save_photo(url, save_dir, filename_prefix, ext='jpg', delay=1, window_name='Camera Feed', image_number=1):
    """
    captures photos from a webcam and saves them to a specified directory
    when a key is pressed.

    args:
        device_num (int): Index of the camera to use (e.g., 0 for default).
        save_dir (str): Path to the directory where images will be saved.
        filename_prefix (str): Prefix for saved image filenames.
        ext (str, optional): File extension for saved images. Defaults to 'jpg'.
        delay (int, optional): Delay in milliseconds for waitKey(). Defaults to 1.
        window_name (str, optional): Name of the display window. Defaults to 'Camera Feed'.
        image_number: The current number of image. It's used to restart name of the newly taken image from the last image taken.
    """
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print(f"Error: Could not open camera {url}.")
        return

    os.makedirs(save_dir, exist_ok=True)
    image_count = image_number

    print("Press 'c' to capture a photo, 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame from camera.")
            break

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(delay) & 0xFF

        if key == ord('c'):  # Press 'c' to capture
            image_filename = os.path.join(save_dir, f"{filename_prefix}_{image_count}.{ext}")
            cv2.imwrite(image_filename, frame)
            print(f"Photo saved: {image_filename}")
            image_count += 1
        elif key == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()