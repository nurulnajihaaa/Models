# prompt: create a succesfully code that can make cropping of image processing by the uploaded image

from google.colab import files
import matplotlib.pyplot as plt
import cv2
import numpy as np

def crop_image(image, x1, y1, x2, y2):
    """Crops an image given coordinates."""
    return image[y1:y2, x1:x2]

uploaded = files.upload()

for fn in uploaded.keys():
    print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')
    
    with open(fn, 'wb') as f:
        f.write(uploaded[fn])
    
    # Read the image using OpenCV
    img = cv2.imdecode(np.frombuffer(uploaded[fn], np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print(f"Error: Could not read image file '{fn}'.")
        continue

    # Get image dimensions
    height, width, _ = img.shape

    # Example coordinates (you can modify these interactively or provide as input)
    x1 = 100 #@param {type:"slider", min:0, max:width, step:1}
    y1 = 50 #@param {type:"slider", min:0, max:height, step:1}
    x2 = 300 #@param {type:"slider", min:0, max:width, step:1}
    y2 = 200 #@param {type:"slider", min:0, max:height, step:1}

    # Ensure valid crop coordinates
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))


    cropped_img = crop_image(img, x1, y1, x2, y2)

    # Display the original and cropped images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Image')

    plt.show()
    
    !rm {fn}
