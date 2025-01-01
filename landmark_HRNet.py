# prompt: from the uploaded image, create a completed code use another model High-resolution networks (HRNets) model  for identify the facial landmark and annotate it on the face

from google.colab import files
import io
from PIL import Image
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import dlib

# Download the shape predictor model if it doesn't exist
predictor_model = "shape_predictor_68_face_landmarks.dat"
try:
    predictor = dlib.shape_predictor(predictor_model)
except RuntimeError:
    !wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    !bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
    predictor = dlib.shape_predictor(predictor_model)


def upload_and_process_image():
    uploaded = files.upload()
    for fn in uploaded.keys():
        image_data = uploaded[fn]
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        return image_np

# Example usage
try:
  image_array = upload_and_process_image()
  image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  detector = dlib.get_frontal_face_detector()
  faces = detector(gray)

  for face in faces:
      landmarks = predictor(gray, face)
      for n in range(0, 68):
          x = landmarks.part(n).x
          y = landmarks.part(n).y
          cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

  cv2_imshow(image)

except Exception as e:
  print(f"An error occurred: {e}")
