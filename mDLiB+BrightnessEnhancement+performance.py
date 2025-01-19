# Import necessary libraries
from google.colab import files
import io
import numpy as np
import cv2
import dlib
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import time

# Function to upload and process the image
def upload_and_process_image():
    uploaded = files.upload()
    for fn in uploaded.keys():
        print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')
        image_data = uploaded[fn]

        # Open the image using PIL
        image = Image.open(io.BytesIO(image_data))

        # Convert the image to a NumPy array
        image_np = np.array(image)
        return image_np

# Function to adjust brightness (contrast enhancement)
def adjust_brightness(image, alpha=1.5):
    """
    Adjusts brightness of the image.
    alpha: brightness scale factor (>1.0 increases brightness)
    """
    brightened = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return brightened

# Function to perform Dlib facial landmark detection
def dlib_facial_landmarks(image, predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print("No faces detected.")
        return image, None, 0, None

    # Process the first detected face
    face = faces[0]
    landmarks = predictor(gray, face)
    detected_landmarks = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])

    # Draw landmarks on the image
    for (x, y) in detected_landmarks:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    return image, detected_landmarks, len(detected_landmarks), face

# Function to calculate Normalized Mean Error (NME)
def calculate_nme(detected_landmarks, ground_truth_landmarks):
    if detected_landmarks.shape != ground_truth_landmarks.shape:
        raise ValueError("Detected and ground truth landmarks must have the same shape.")
    distances = np.linalg.norm(detected_landmarks - ground_truth_landmarks, axis=1)
    ipd = np.linalg.norm(ground_truth_landmarks[0] - ground_truth_landmarks[1])  # Interpupillary distance
    nme = np.mean(distances) / ipd
    return nme

# Main code
if __name__ == "__main__":
    # Upload the image
    print("Please upload an image file...")
    image_array = upload_and_process_image()

    # Resize image for faster processing if necessary
    image_array = cv2.resize(image_array, (640, 480))

    # Download and extract Dlib's shape predictor
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    !wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    !bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

    # Perform Dlib facial landmark detection
    print("\nPerforming Dlib facial landmark detection...")
    start_time = time.time()
    annotated_image, detected_landmarks, robustness_score, face = dlib_facial_landmarks(image_array.copy(), predictor_path)
    elapsed_time = time.time() - start_time

    # Display Dlib performance
    if detected_landmarks is not None:
        print(f"Dlib Detection Time: {elapsed_time:.4f} seconds")
        print(f"Dlib Robustness Score (Detected Landmarks): {robustness_score}")
    else:
        print("Dlib did not detect any landmarks.")

    # Define simulated ground truth landmarks for NME calculation (you can replace these with real data)
    ground_truth_landmarks = np.array([
        [150, 200], [180, 210], [160, 240], [140, 280], [190, 290],
        [170, 220], [180, 230], [160, 250], [140, 270], [190, 280],
        # Add more landmarks as needed
    ])

    # Calculate Normalized Mean Error (NME) if landmarks are detected
    if detected_landmarks is not None:
        try:
            nme = calculate_nme(detected_landmarks[:len(ground_truth_landmarks)], ground_truth_landmarks)
            print(f"Normalized Mean Error (NME): {nme:.4f}")
        except Exception as e:
            print(f"Error calculating NME: {e}")

    # Apply brightness enhancement
    brightened_image = adjust_brightness(image_array)

    # Display results
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(brightened_image, cv2.COLOR_BGR2RGB))
    plt.title("Brightness Enhanced Image")
    plt.subplot(1, 3, 3)
    if detected_landmarks is not None:
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.title("Dlib Detected Landmarks")
    else:
        plt.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        plt.title("No Landmarks Detected")
    plt.show()
