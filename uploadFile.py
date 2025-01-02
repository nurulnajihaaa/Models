# prompt: create Holistic methods that allow user to add image from the local device

from google.colab import files
import io
from PIL import Image
import numpy as np

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
