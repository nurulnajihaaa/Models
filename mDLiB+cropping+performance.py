from google.colab import files
import io
from PIL import Image
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

# Function to crop the image based on coordinates
def crop_image(image, x1, y1, x2, y2):
    """Crops an image given coordinates."""
    return image[y1:y2, x1:x2]

# Load Dlib model for facial landmark detection
def load_dlib_model():
    predictor_model = "shape_predictor_68_face_landmarks.dat"
    !wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    !bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)
    return detector, predictor

# Function for Dlib facial landmark detection
def detect_with_dlib(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    landmarks = []
    for face in faces:
        # Get the landmarks for the detected face
        facial_landmarks = predictor(gray, face)
        landmarks = np.array([(facial_landmarks.part(n).x, facial_landmarks.part(n).y) for n in range(68)])
    return landmarks, faces

# Function to calculate Normalized Mean Error (NME)
def calculate_nme(detected_landmarks, ground_truth_landmarks):
    distances = np.linalg.norm(detected_landmarks - ground_truth_landmarks, axis=1)
    ipd = np.linalg.norm(ground_truth_landmarks[0] - ground_truth_landmarks[1])  # Interpupillary distance
    nme = np.mean(distances) / ipd
    return nme

# Main process
def main():
    # Step 1: Upload the image
    print("Please upload an image file...")
    image_array = upload_and_process_image()
    print("Image successfully uploaded and processed.")

    # Resize image for faster processing
    image_array = cv2.resize(image_array, (640, 480))

    # Step 2: Load Dlib model
    detector, predictor = load_dlib_model()

    # Step 3: Detect facial landmarks with Dlib
    start_time = time.time()
    detected_landmarks, faces = detect_with_dlib(image_array, detector, predictor)
    elapsed_time = time.time() - start_time
    print(f"Dlib Detection Time: {elapsed_time:.4f} seconds")

    if len(detected_landmarks) > 0:
        print(f"Number of landmarks detected: {len(detected_landmarks)}")

        # Draw landmarks on the image
        for (x, y) in detected_landmarks:
            cv2.circle(image_array, (x, y), 2, (0, 255, 0), -1)

        # Step 4: Crop the image based on selected coordinates
        print("Now, let's crop the image!")
        height, width, _ = image_array.shape
        x1 = 100  # Example cropping coordinates
        y1 = 50
        x2 = 300
        y2 = 200

        cropped_image = crop_image(image_array, x1, y1, x2, y2)

        # Display the original image with landmarks and cropped image
        plt.figure(figsize=(15, 7))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        plt.title("Image with Dlib Landmarks")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title("Cropped Image")
        plt.show()

        # Step 5: Display Dlib performance
        ground_truth_landmarks = np.array([  # Sample ground truth
            [150, 200], [180, 210], [160, 240], [140, 280], [190, 290]
        ])
        try:
            nme = calculate_nme(np.array(detected_landmarks[:5]), ground_truth_landmarks)
            print(f"Normalized Mean Error (NME): {nme:.4f}")
        except Exception as e:
            print(f"Error calculating NME: {e}")
        
    else:
        print("No landmarks detected. Please check the image.")

# Run the main function
if __name__ == "__main__":
    main()
