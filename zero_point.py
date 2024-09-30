import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to convert an image to grayscale
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Contrast Stretching
def contrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched_image = (image - min_val) * (255 / (max_val - min_val))
    return stretched_image.astype(np.uint8)

# Thresholding
def thresholding(image, threshold_value=128):
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_image

# Digital Negative (Inverse Image)
def digital_negative(image):
    negative_image = 255 - image
    return negative_image

# Log Transformation
def log_transformation(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(1 + image))
    return np.array(log_image, dtype=np.uint8)

# Power Law (Gamma) Transformation
def power_law_transformation(image, gamma=1.0):
    normalized_image = image / 255.0
    gamma_image = np.power(normalized_image, gamma)
    return np.array(gamma_image * 255, dtype=np.uint8)

# Load and convert the image to grayscale
image_path = 'cat.jpg'  # Replace with your image file path
image = cv2.imread(image_path)
grayscale_image = convert_to_grayscale(image)

# Apply different transformations
contrast_image = contrast_stretching(grayscale_image)
threshold_image = thresholding(grayscale_image, 128)
negative_image = digital_negative(grayscale_image)
log_image = log_transformation(grayscale_image)
gamma_image = power_law_transformation(grayscale_image, gamma=2.0)

# Display all the results
titles = ['Original Grayscale', 'Contrast Stretching', 'Thresholding', 
          'Digital Negative', 'Log Transformation', 'Power Law (Gamma) Transformation']
images = [grayscale_image, contrast_image, threshold_image, 
          negative_image, log_image, gamma_image]

# Plot the images
plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# Save the images (optional)
cv2.imwrite('contrast_stretching.jpg', contrast_image)
cv2.imwrite('thresholding.jpg', threshold_image)
cv2.imwrite('digital_negative.jpg', negative_image)
cv2.imwrite('log_transformation.jpg', log_image)
cv2.imwrite('power_law_transformation.jpg', gamma_image)
