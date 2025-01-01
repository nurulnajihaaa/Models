# prompt: create a succesfully code that can make grayscale of image processing by the uploaded image

from google.colab import files
import matplotlib.pyplot as plt
import cv2
import numpy as np

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

  # Read the image using OpenCV
  img = cv2.imdecode(np.frombuffer(uploaded[fn], np.uint8), cv2.IMREAD_COLOR)

  # Check if the image was loaded successfully
  if img is None:
      print(f"Error: Could not read image file '{fn}'. Please ensure it's a valid image format.")
      continue

  # Convert the image to grayscale
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


  # Display the original and grayscale images
  plt.figure(figsize=(10, 5))

  plt.subplot(1, 2, 1)
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Show BGR image in RGB format
  plt.title('Original Image')

  plt.subplot(1, 2, 2)
  plt.imshow(gray_img, cmap='gray')
  plt.title('Grayscale Image')

  plt.show()
