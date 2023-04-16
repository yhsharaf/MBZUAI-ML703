from sixdrepnet import SixDRepNet
import cv2
import numpy as np

# Create an instance of the SixdRepNet class
model = SixDRepNet()

# Load the two input images
img1 = cv2.imread("/home/yhsharaf/Desktop/MBZUAI-ML703/Clusters/KMeans/7/frame_00023_rgb.png")
img2 = cv2.imread("/home/yhsharaf/Desktop/MBZUAI-ML703/Clusters/KMeans/7/frame_00062_rgb.png")

# Compute the pose between the two images
pitch1, yaw1, roll1 = model.predict(img1)
pitch2, yaw2, roll2 = model.predict(img2)
# Define two vectors of yaw, roll, and pitch angles
vec1 = np.array([yaw1, roll1, pitch1])
vec2 = np.array([yaw2, roll2, pitch2])

# Calculate the Euclidean distance between the two vectors
distance = np.linalg.norm(vec1 - vec2)

print(distance)