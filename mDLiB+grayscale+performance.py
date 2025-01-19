# Install dependencies
!pip install dlib opencv-python-headless

# Import libraries
from google.colab import files
import matplotlib.pyplot as plt
import numpy as np
import cv2
import dlib
import time

# Download and extract Dlib's shape predictor
!wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
!bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Dlib's predictor path
predictor_path = "shape_predictor_68_face_landmarks.dat"

def upload_image():
    """Allows the user to upload an image and returns it as a NumPy array."""
    print("Please upload an image file...")
    uploaded = files.upload()
    image_path = list(uploaded.keys())[0]  # Get the uploaded file's name
    image_array = cv2.imdecode(np.frombuffer(uploaded[image_path], np.uint8), cv2.IMREAD_COLOR)

    if image_array is None:
        raise ValueError("Error: Could not load the image. Please upload a valid image.")

    # Resize the image to avoid memory issues
    image_array = cv2.resize(image_array, (640, 480))
    return image_array

def detect_with_dlib(image):
    """Detect facial landmarks using Dlib and annotate them on the image."""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        print("Dlib: No faces detected.")
        return image, None

    detected_landmarks = []
    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        detected_landmarks = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])

    return image, detected_landmarks

def calculate_nme(detected_landmarks, ground_truth_landmarks):
    """Calculate the Normalized Mean Error (NME) based on detected landmarks."""
    if detected_landmarks.shape != ground_truth_landmarks.shape:
        raise ValueError("Detected and ground truth landmarks must have the same shape.")
    distances = np.linalg.norm(detected_landmarks - ground_truth_landmarks, axis=1)
    ipd = np.linalg.norm(ground_truth_landmarks[0] - ground_truth_landmarks[1])  # Interpupillary distance
    nme = np.mean(distances) / ipd
    return nme

def convert_to_grayscale(image):
    """Convert an image to grayscale."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def display_images(original, grayscale):
    """Display the original and grayscale images side by side."""
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))  # Show BGR image in RGB format
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(grayscale, cmap='gray')
    plt.title('Grayscale Image')

    plt.show()

# Main code execution
try:
    # Step 1: Upload the image
    image_array = upload_image()
    print("Image successfully uploaded and processed as a NumPy array.")

    # Step 2: Detect facial landmarks using Dlib
    print("\nDetecting facial landmarks with Dlib...")
    start_time = time.time()
    annotated_image, detected_landmarks = detect_with_dlib(image_array.copy())
    dlib_detection_time = time.time() - start_time

    if detected_landmarks is not None:
        # Define example ground truth landmarks (replace with actual values if available)
        ground_truth_landmarks = np.array([
            [130, 210], [170, 210], [150, 250], [140, 290], [180, 290]
        ])
        dlib_nme = calculate_nme(detected_landmarks[:5], ground_truth_landmarks)
        print(f"Dlib Detection Time: {dlib_detection_time:.4f} seconds")
        print(f"Dlib Normalized Mean Error (NME): {dlib_nme:.4f}")
        print(f"Dlib Robustness Score: {len(detected_landmarks)} landmarks")
    else:
        print("No landmarks detected by Dlib.")

    # Step 3: Convert the image to grayscale
    print("\nConverting the image to grayscale...")
    grayscale_image = convert_to_grayscale(image_array)

    # Step 4: Display results
    print("\nDisplaying images...")
    display_images(image_array, grayscale_image)
    print("\nDisplaying annotated image with landmarks...")
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title('Annotated Image with Dlib Landmarks')
    plt.show()

except Exception as e:
    print(f"Error: {e}")
