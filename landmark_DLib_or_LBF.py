# prompt: from the uploaded image, use another model DLIB model  for identify the facial landmark and annotate it ton the face

from google.colab import files
import io
from PIL import Image
import numpy as np
import cv2
import dlib
from google.colab.patches import cv2_imshow


def upload_and_process_image():
  """
  Allows user to upload an image from their local device,
  opens it using PIL, and returns the image as a NumPy array.
  """

  uploaded = files.upload()
  for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])))
    image_data = uploaded[fn]

    # Open the image using PIL
    image = Image.open(io.BytesIO(image_data))

    # Convert the image to a NumPy array
    image_np = np.array(image)
    return image_np


# Example usage:
try:
  image_array = upload_and_process_image()
  print("Image successfully uploaded and processed as a NumPy array.")
  # Further processing with image_array can be done here
except Exception as e:
  print(f"Error during image upload and processing: {e}")

!pip install dlib

# Load the pre-trained facial landmark model from dlib
predictor_model = "shape_predictor_68_face_landmarks.dat"
!wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
!bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_model)


try:
    image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    for face in faces:
        # Get the landmarks/parts for the face in box d.
        landmarks = predictor(gray, face)

        # Draw facial landmarks on the image
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    cv2_imshow(image)

except Exception as e:
    print(f"Error processing the image with dlib: {e}")
