import cv2
import torch
import numpy as np
import argparse
from PIL import Image
import os
from torchvision import transforms
from hrnet import HRNetModel  # Assume you have an HRNet model implementation

# Define the model path
HRNET_MODEL_PATH = "hrnet_facial_landmarks.pth"

# Load HRNet model
def load_model():
    model = HRNetModel()
    model.load_state_dict(torch.load(HRNET_MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def process_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        landmarks = model(image_tensor)
    landmarks = landmarks.squeeze().numpy()

    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    h, w, _ = image_cv.shape
    landmarks[:, 0] *= w
    landmarks[:, 1] *= h

    for (x, y) in landmarks:
        cv2.circle(image_cv, (int(x), int(y)), 2, (0, 255, 0), -1)

    output_path = "output.jpg"
    cv2.imwrite(output_path, image_cv)
    print(f"Processed image saved as {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image")
    args = parser.parse_args()
    
    model = load_model()
    process_image(args.image, model)
