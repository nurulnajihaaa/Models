# Install required libraries
!pip install dlib opencv-python-headless matplotlib

# Import necessary libraries
import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
import time
from google.colab import files
from PIL import Image
import io

# Upload and process the image
def upload_image():
    """Allows the user to upload an image and returns it as a NumPy array."""
    uploaded = files.upload()
    for fn in uploaded.keys():
        print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')
        image_data = uploaded[fn]
        image = Image.open(io.BytesIO(image_data))
        return np.array(image)

# Function to adjust image contrast
def adjust_contrast(image, contrast_factor):
    """
    Adjusts the contrast of an image.
    Args:
        image: Input image as a NumPy array.
        contrast_factor: Float value for contrast adjustment (>1 increases contrast, <1 decreases it).
    Returns:
        Contrast-adjusted image as a NumPy array.
    """
    contrast_factor = max(0.0, min(contrast_factor, 3.0))  # Clamp contrast factor
    return cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)

# Function to detect facial landmarks using DLIB
def detect_with_dlib(image, predictor_path):
    """
    Detects facial landmarks using the DLIB model.
    Args:
        image: Input image as a NumPy array.
        predictor_path: Path to the DLIB shape predictor model.
    Returns:
        Detected landmarks as a NumPy array.
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        print("No faces detected.")
        return None
    landmarks = predictor(gray, faces[0])
    return np.array([(p.x, p.y) for p in landmarks.parts()])

# Function to calculate Normalized Mean Error (NME)
def calculate_nme(detected_landmarks, ground_truth_landmarks):
    """
    Calculates the Normalized Mean Error (NME) between detected and ground truth landmarks.
    Args:
        detected_landmarks: Detected landmarks as a NumPy array.
        ground_truth_landmarks: Ground truth landmarks as a NumPy array.
    Returns:
        NME value as a float.
    """
    distances = np.linalg.norm(detected_landmarks - ground_truth_landmarks, axis=1)
    ipd = np.linalg.norm(ground_truth_landmarks[0] - ground_truth_landmarks[1])  # Interpupillary distance
    return np.mean(distances) / ipd

# Main script
try:
    # Step 1: Upload the image
    print("Please upload an image file...")
    image_array = upload_image()
    print("Image successfully uploaded.")

    # Step 2: Download and load the DLIB shape predictor model
    predictor_model = "shape_predictor_68_face_landmarks.dat"
    !wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    !bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

    # Step 3: Detect landmarks using DLIB
    print("Detecting facial landmarks using DLIB...")
    start_time = time.time()
    detected_landmarks = detect_with_dlib(image_array, predictor_model)
    elapsed_time = time.time() - start_time

    if detected_landmarks is not None:
        print(f"Detection time: {elapsed_time:.4f} seconds")
        print(f"Number of landmarks detected: {len(detected_landmarks)}")

        # Example ground truth landmarks (for performance evaluation)
        ground_truth_landmarks = np.array([
            [150, 200], [180, 210], [160, 240], [140, 280], [190, 290]
        ])

        # Step 4: Calculate performance metrics
        nme = calculate_nme(detected_landmarks, ground_truth_landmarks)
        print(f"Normalized Mean Error (NME): {nme:.4f}")

        # Step 5: Enhance contrast of the image
        contrast_factor = 1.5  # Adjust as needed
        print("Enhancing contrast of the image...")
        enhanced_image = adjust_contrast(image_array, contrast_factor)

        # Step 6: Display results
        print("Displaying results...")
        plt.figure(figsize=(15, 10))

        # Original image with landmarks
        for (x, y) in detected_landmarks:
            cv2.circle(image_array, (x, y), 2, (0, 255, 0), -1)
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        plt.title("Original Image with DLIB Landmarks")

        # Contrast-enhanced image
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Contrast-Enhanced Image (Factor: {contrast_factor})")

        plt.show()

    else:
        print("No landmarks detected. Unable to evaluate performance.")

except Exception as e:
    print(f"Error: {e}")
