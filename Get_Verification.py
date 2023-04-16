import dlib
import cv2
import numpy as np

# Load the shape predictor model
predictor = dlib.shape_predictor("/home/yhsharaf/Desktop/MBZUAI-ML703/shape_predictor_68_face_landmarks.dat")


# Load the images you want to compare
img1_path = '/home/yhsharaf/Desktop/MBZUAI-ML703/Clusters/KMeans/7/frame_00023_rgb.png'
img2_path = '/home/yhsharaf/Desktop/MBZUAI-ML703/Clusters/KMeans/7/frame_00189_rgb.png'
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# Detect facial landmarks in the images
detector = dlib.get_frontal_face_detector()
img1_dets = detector(img1, 1)
img2_dets = detector(img2, 1)
if len(img1_dets) == 0 or len(img2_dets) == 0:
    print('No faces were detected in one or both of the images.')
else:
    img1_shape = predictor(img1, img1_dets[0])
    img2_shape = predictor(img2, img2_dets[0])

    # Compute the Euclidean distance between the facial landmarks
    img1_vec = np.empty([68, 2], dtype=int)
    img2_vec = np.empty([68, 2], dtype=int)
    for i in range(68):
        img1_vec[i] = [img1_shape.part(i).x, img1_shape.part(i).y]
        img2_vec[i] = [img2_shape.part(i).x, img2_shape.part(i).y]
    distance = np.sqrt(np.sum(np.square(img1_vec - img2_vec)))

    print(f'The Euclidean distance between the facial landmarks is {distance}.')
