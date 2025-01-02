from google.colab import files
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Function to adjust contrast
def adjust_contrast(image, contrast_factor):
    contrast_factor = max(0.0, min(contrast_factor, 3.0))
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
    return adjusted_image

# Function to enhance the image
def enhance_image(image, contrast_factor=1.5):
    enhanced_image = adjust_contrast(image, contrast_factor)
    return enhanced_image

# Function to crop the image
def crop_image(image, x1, y1, x2, y2):
    return image[y1:y2, x1:x2]

uploaded = files.upload()

for fn in uploaded.keys():
    print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')

    # Read the image using OpenCV
    img = cv2.imdecode(np.frombuffer(uploaded[fn], np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print(f"Error: Could not read image file '{fn}'.")
        continue

    # Step 1: Convert to Grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Convert the image to RGB color channels
    b, g, r = cv2.split(img)

    # Step 3: Adjust Contrast
    contrast_factor = 1.5  # Example contrast factor
    contrast_enhanced = enhance_image(img, contrast_factor)

    # Step 4: Enhance Image
    enhanced_image = enhance_image(contrast_enhanced)

    # Step 5: Brightness Enhancement (same as contrast enhancement)
    brightness_enhanced = enhance_image(enhanced_image)

    # Step 6: Crop Image (using example coordinates)
    height, width, _ = img.shape
    x1, y1, x2, y2 = 100, 50, 300, 200  # Example coordinates for cropping
    cropped_img = crop_image(brightness_enhanced, x1, y1, x2, y2)

    # Display the results
    plt.figure(figsize=(15, 10))

    # Original Image
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    # Grayscale Image
    plt.subplot(3, 3, 2)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Grayscale Image')

    # RGB Channels
    plt.subplot(3, 3, 3)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('RGB Image')

    # Contrast Enhanced Image
    plt.subplot(3, 3, 4)
    plt.imshow(cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2RGB))
    plt.title('Contrast Enhanced Image')

    # Enhanced Image
    plt.subplot(3, 3, 5)
    plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    plt.title('Enhanced Image')

    # Brightness Enhanced Image
    plt.subplot(3, 3, 6)
    plt.imshow(cv2.cvtColor(brightness_enhanced, cv2.COLOR_BGR2RGB))
    plt.title('Brightness Enhanced Image')

    # Cropped Image
    plt.subplot(3, 3, 7)
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Image')

    plt.show()

    # Optional: Remove the temporary file
    !rm {fn}
