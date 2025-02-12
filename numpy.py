import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create a black image
image_height, image_width = 400, 400
image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# Draw a square
cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)

# Draw a circle
cv2.circle(image, (300, 100), 50, (255, 255, 255), -1)

# Draw a rectangle
cv2.rectangle(image, (200, 200), (350, 300), (255, 255, 255), -1)

# Draw a wavy line
for x in range(0, 400):
    y = int(200 +  20 *  np.sin(x * 0.1))
    cv2.circle(image, (x, y), 1, (255, 255, 255), -1)

# Save the image
cv2.imwrite('shapes.png', image)

# Display the created image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Shapes Image')
plt.show()
# Load the image
image = cv2.imread('shapes.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image,cmap='gray')
plt.axis('off')
plt.title('Grayscale Image')
plt.show()

# Apply a simple thresholding technique
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
plt.imshow(binary_image,cmap='gray')
plt.axis('off')
plt.title('Binary Image')
plt.show()

# Calculate and display the histogram of pixel values
histogram, bins = np.histogram(gray_image.flatten(), 256, [0, 256])

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Grayscale Image')
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Histogram')
plt.plot(histogram)
plt.xlim([0, 256])

plt.show()

# Function to detect edges using Canny edge detector
def detect_edges(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

# Detect edges in the grayscale image
edges = detect_edges(gray_image)

# Display the edges
plt.figure(figsize=(6, 6))
plt.title('Edges Detected')
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()