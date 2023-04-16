import cv2
import numpy as np
from face_alignment import FaceAlignment, LandmarksType


# Initialize face alignment model
fa = FaceAlignment(LandmarksType._2D, device='cpu')


# Load and align two images
img1 = cv2.imread('/home/yhsharaf/Desktop/MBZUAI-ML703/BIWISubset/frame_00004_rgb.png')
img2 = cv2.imread('/home/yhsharaf/Desktop/MBZUAI-ML703/BIWISubset/frame_00004_rgb.png')
landmarks1 = fa.get_landmarks(img1)[0]
landmarks2 = fa.get_landmarks(img2)[0]


# Compute Euclidean distance between landmarks
dist = np.linalg.norm(landmarks1 - landmarks2)

print('Distance between landmarks:', dist)
