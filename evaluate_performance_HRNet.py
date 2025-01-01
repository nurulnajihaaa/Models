import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import torch
from torchvision import transforms
from PIL import Image

# Function to upload an image
def upload_image():
    uploaded = files.upload()
    for filename in uploaded.keys():
        print(f"Uploaded file: {filename}")
        return filename

# Preprocess the image for HRNet
def preprocess_image(image_path):
    input_size = (256, 256)  # Define the input size for HRNet
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Placeholder for loading HRNet model (replace with actual implementation)
def load_hrnet_model():
    # Ensure to load the actual HRNet model and weights here
    print("Loading HRNet model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Example
    model.eval()
    return model

# Detect landmarks using HRNet
def detect_landmarks(model, image_tensor):
    with torch.no_grad():
        start_time = time.time()
        output = model(image_tensor)  # Adjust this based on HRNet's forward method
        detection_time = time.time() - start_time
    return output, detection_time

# Calculate Normalized Mean Error (NME)
def calculate_nme(predicted_landmarks, ground_truth_landmarks, reference_distance):
    error = np.linalg.norm(predicted_landmarks - ground_truth_landmarks, axis=1)
    nme = np.mean(error) / reference_distance
    return nme

# Calculate Robustness Score
def calculate_robustness(predicted_landmarks, ground_truth_landmarks, threshold=5):
    distances = np.linalg.norm(predicted_landmarks - ground_truth_landmarks, axis=1)
    robustness = np.mean(distances < threshold) * 100
    return robustness

# Main function
def main():
    print("Upload an image to evaluate HRNet.")
    image_path = upload_image()

    # Preprocess the image
    image_tensor = preprocess_image(image_path)

    # Load HRNet model
    model = load_hrnet_model()

    # Detect landmarks and measure time
    output, detection_time = detect_landmarks(model, image_tensor)

    # Placeholder for processing model output (replace with actual parsing logic)
    predicted_landmarks = np.random.rand(68, 2) * 256  # Simulated output for testing

    # Placeholder ground truth (replace with actual ground truth input)
    ground_truth_landmarks = np.random.rand(68, 2) * 256

    # Define a reference distance (e.g., interocular distance)
    reference_distance = np.linalg.norm(ground_truth_landmarks[36] - ground_truth_landmarks[45])

    # Calculate metrics
    nme = calculate_nme(predicted_landmarks, ground_truth_landmarks, reference_distance)
    robustness = calculate_robustness(predicted_landmarks, ground_truth_landmarks)

    # Display results
    print(f"Detection Time: {detection_time:.4f} seconds")
    print(f"Normalized Mean Error (NME): {nme:.4f}")
    print(f"Robustness Score: {robustness:.2f}%")

    # Visualize the landmarks
    image = cv2.imread(image_path)
    for (x, y) in predicted_landmarks.astype(int):
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Predicted Landmarks")
    plt.axis("off")
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
