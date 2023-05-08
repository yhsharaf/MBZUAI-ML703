from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import os
import numpy as np

auto_array = ['05_To_04_W_To_W_All', '05_To_04_W_To_W_Cluster_1', '05_To_04_W_To_W_GMM_5', '05_To_04_W_To_W_GMM_7', '05_To_04_W_To_W_GMM_9', '05_To_04_W_To_W_GMM_11', '05_To_04_W_To_W_Kmeans_5', '05_To_04_W_To_W_Kmeans_7', '05_To_04_W_To_W_Kmeans_9', '05_To_04_W_To_W_Kmeans_11', '05_To_04_W_To_W_Right_Left', '05_To_04_W_To_W_Static', '05_To_04_W_To_W_Up_Down']
# auto_array = ['05_To_13_W_To_M_All', '05_To_13_W_To_M_Cluster_1', '05_To_13_W_To_M_GMM_5', '05_To_13_W_To_M_GMM_7', '05_To_13_W_To_M_GMM_9', '05_To_13_W_To_M_GMM_11', '05_To_13_W_To_M_Kmeans_5', '05_To_13_W_To_M_Kmeans_7', '05_To_13_W_To_M_Kmeans_9', '05_To_13_W_To_M_Kmeans_11', '05_To_13_W_To_M_Right_Left', '05_To_13_W_To_M_Static', '05_To_13_W_To_M_Up_Down' ]
# auto_array = ['05_To_18_SW_To_SW_All', '05_To_18_SW_To_SW_Cluster_1', '05_To_18_SW_To_SW_GMM_5', '05_To_18_SW_To_SW_GMM_7', '05_To_18_SW_To_SW_GMM_9', '05_To_18_SW_To_SW_GMM_11', '05_To_18_SW_To_SW_Kmeans_5', '05_To_18_SW_To_SW_Kmeans_7', '05_To_18_SW_To_SW_Kmeans_9', '05_To_18_SW_To_SW_Kmeans_11', '05_To_18_SW_To_SW_Right_Left', '05_To_18_SW_To_SW_Static', '05_To_18_SW_To_SW_Up_Down']
for i in range(0, len(auto_array)):

    im_folder_dst = '/home/youssef.sharaf/Desktop/MBZUAI-ML703/Test_Dataset/W_To_W/01_GT/data_dst'
    im_list_dst = [os.path.join(im_folder_dst, x) for x in os.listdir(im_folder_dst) if 'png' in x]

    im_folder_src = '/home/youssef.sharaf/Desktop/DeepFaceLab/'+auto_array[i]+'/01_Test/merged'
    im_list_src = [os.path.join(im_folder_src, x) for x in os.listdir(im_folder_src) if 'png' in x]

    detector = MTCNN()

    os.makedirs('test_crop', exist_ok=True)
    os.makedirs('/home/youssef.sharaf/Desktop/DeepFaceLab/'+auto_array[i]+'/01_Test/face_track_crop', exist_ok=True)

    count = 0
    for im_path in im_list_dst:
        im = Image.open(im_path)
        boxes, _ = detector.detect(im)
        box = [int(x) for x in boxes[0]]

        im_cv = np.array(im)
        im_cv = im_cv[:, :, ::-1]
        im_cv = im_cv[box[1]:box[3], box[0]:box[2],:]

        # Open corresponding image in source folder
        im_path_src = im_path.replace(im_folder_dst, im_folder_src)
        im_src = Image.open(im_path_src)
        im_src = np.array(im_src)
        im_src = im_src[:, :, ::-1]
        im_src = im_src[box[1]:box[3], box[0]:box[2],:]

        cv2.imwrite(os.path.join('01_GT', f'{count}.png'), im_cv)
        cv2.imwrite(os.path.join('/home/youssef.sharaf/Desktop/DeepFaceLab/'+auto_array[i]+'/01_Test/face_track_crop', f'{count}.png'), im_src)
        count+=1
        # print(count)
