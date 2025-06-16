import random
import os
import numpy as np
import cv2
from tqdm import tqdm


def load_calibration_images(
    image_dir, calib_size, image_shape, data_type="float32"
) -> np.ndarray:
    """
    Load image files and process them into specified shape
    """
    if isinstance(image_shape, str):
        h, w, c = [int(s) for s in image_shape.split(",")]
    else:
        h, w, c = image_shape

    # Get all image file paths
    image_files = []
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    for file in os.listdir(image_dir):
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_extensions:
            image_files.append(os.path.join(image_dir, file))

    # Ensure there are enough images
    if len(image_files) < calib_size:
        print(
            f"Warning: Number of images ({len(image_files)}) is less than requested calib_size ({calib_size})"
        )
        print(f"Will use all available images")
        calib_size = len(image_files)

    # Randomly select calib_size images
    selected_files = random.sample(image_files, calib_size)

    # Load and process images
    dtype = np.float32 if data_type == "float32" else np.uint8

    calib_data = np.zeros((calib_size, h, w, c), dtype=dtype)

    for i, file_path in enumerate(tqdm(selected_files, desc="Processing images")):
        try:
            img = cv2.imread(file_path)
            if img is None:
                print(f"Cannot read image: {file_path}, will skip")
                continue

            # Convert to RGB
            if c == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif c == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, axis=-1)

            # Resize image
            img = cv2.resize(img, (w, h))

            # Ensure correct shape
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)

            # Normalize to [0,1]
            # img = (img.astype(np.float32) - 127.5) / 127.5

            calib_data[i] = img

        except Exception as e:
            print(f"Error processing image: {file_path}, error: {str(e)}")
    print(
        "Calibration dataset properties:",
        np.max(calib_data),
        np.min(calib_data),
        calib_data.dtype,
    )
    return calib_data
