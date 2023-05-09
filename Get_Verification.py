import dlib
import cv2
import numpy as np

# Load the shape predictor model
predictor = dlib.shape_predictor("/home/yhsharaf/Desktop/MBZUAI-ML703/shape_predictor_68_face_landmarks.dat")


# Load the images you want to compare
img1_path = '/home/yhsharaf/Documents/BiwiDataset/faces_0/02/frame_00023_rgb.png'
img2_path = '/home/yhsharaf/Documents/BiwiDataset/faces_0/02/frame_00189_rgb.png'
src_img_load = cv2.imread(img1_path)
dst_img_load = cv2.imread(img2_path)

# Detect facial landmarks in the images
detector = dlib.get_frontal_face_detector()
src_dets = detector(src_img_load, 1)
dst_dets = detector(dst_img_load, 1)
if len(src_dets) == 0 or len(dst_dets) == 0:
    print('No faces were detected in one or both of the images.')
else:
    src_shape = predictor(src_img_load, src_dets[0])
    dst_shape = predictor(dst_img_load, dst_dets[0])

    # Compute the Euclidean distance between the facial landmarks
    src_vec = np.empty([68, 2], dtype=int)
    dst_vec = np.empty([68, 2], dtype=int)
    for i in range(68):
        src_vec[i] = [src_shape.part(i).x, src_shape.part(i).y]
        dst_vec[i] = [dst_shape.part(i).x, dst_shape.part(i).y]
    distance = np.sqrt(np.sum(np.square(src_vec - dst_vec)))

    # Calculate the diagonal length of the face bounding box
    src_bbox = cv2.boundingRect(src_vec)
    dst_bbox = cv2.boundingRect(dst_vec)
    src_diag = np.sqrt(np.sum(np.square(np.array(src_bbox[2:]))))
    dst_diag = np.sqrt(np.sum(np.square(np.array(dst_bbox[2:]))))

    # Normalize the distance by the size of the face
    norm_distance = distance / ((src_diag + dst_diag) / 2)

    # Calculate the landmark verification score
    lms_score = -np.log(norm_distance)

    print(f'The landmark verification score is {lms_score}.')

    print(f'The Euclidean distance between the facial landmarks is {distance}.')
